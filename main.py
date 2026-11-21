"""Forecast Exophase earn rates for every GTA VI PS5 trophy at a case's target.

Approach, in one paragraph: earners and players are modelled separately
and divided. Each trophy's earners are tracked as ln(earners) against
log-time since release, because early trophy growth is roughly linear
there, and the tracked-player count gets its own power-law line the same
way; the rate forecast is fitted earners over fitted players, so a refresh
that only grows the cohort dilutes the rate without touching earners. For
every trophy I fit a recency-weighted least-squares line through its
reads, then pull its slope toward the slope its tier's trophies share
(bronze is 46 trophies, so its shared growth shape is well measured;
the lone platinum barely shrinks at all) while keeping the trophy's own
level, because trophies in one tier sit at very different rates. The two
models of the rate — the saturating logit trend and the earners/players
power law — are each trusted only where their shape is right: when the
rate is falling while earners are flat or growing, the fall is dilution
and the power-law ratio is used, because the logit trend would
extrapolate the cliff into oblivion; in every other regime the logit
trend is used, because the power law is unbounded and would otherwise
print 99.99% for trophies whose true rate is ~60%. A trophy whose
description demands endgame-scale effort (platinum, 100% completion, a
finale, a large-count grind) and whose fitted earners stay below 30 is
pinned to the 0.01% floor with an upward-only spread instead: its
launch-night rate is a phantom handful of earners, not real accumulation.
The spread around the median combines three honest sources of doubt: the
random-walk term the house baseline uses, how badly the trophy's own line
fit its reads, and a dilution term for the next Exophase refresh, which
can divide every rate at once when the tracked-player count jumps.
Quantiles come from the normal z-values at the five scored levels.
Everything is plain arithmetic, no randomness, so two runs write identical
bytes.

Standard library only. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in found.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release instant the log-time axis is measured from.
EPOCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# Random-walk volatility of a rate on the logit scale, per sqrt(day).
WALK_SIGMA = 0.5
# Recency weight half-life for the line fits, in hours.
WEIGHT_HALF_LIFE_H = 24.0
# Prior strength (in reads) of a tier's shared line when shrinking a trophy.
SHRINK_PRIOR_READS = 8.0
# Floor / cap for the fit-residual part of the spread, in logit units.
RESID_FLOOR = 0.05
RESID_CAP = 1.5
# Dilution term: a refresh can hit at any horizon (the constant) and its
# uncertainty grows with horizon (the sqrt-days part).
DILUTION_BASE = 0.25
DILUTION_PER_SQRT_DAY = 0.35
# v2: separate earners/players model. An endgame trophy whose fitted
# earners stay below this count is pinned to the 0.01% floor: at final N
# (~200k) that many earners is a ~0.01% rate, so the distinction is
# invisible in the printed rate, while the launch-night phantom rates
# (4-7 earners) would otherwise forecast ~0.6%.
ENDGAME_EARNERS_GATE = 30.0
# Global multiplier on the total spread (calibrated on the 20 Nov validation).
SPREAD_SCALE = 1.0
# Floor on the total spread, in logit units: even a well-fit trophy keeps
# honest doubt about the next refresh. Set at the bottom of the 0.3-0.5
# range: at the final horizon the raw spread (~1.3) is far above it, so it
# never binds there; it only binds on short horizons, where a higher floor
# would widen the band for no benefit.
SPREAD_FLOOR = 0.3


def clamp_pct(value):
    """Keep a percent inside the range whose logit is finite."""
    if value < 0.01:
        return 0.01
    if value > 99.99:
        return 99.99
    return value


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round_half_up_4(value):
    """Round to 4 decimals, halves away from zero, like the house baseline.

    Whole results come back as ints so the JSON writes 100, not 100.0.
    """
    scaled = value * 10000.0
    whole = math.floor(scaled + 0.5)
    out = whole / 10000.0
    return int(out) if out == int(out) else out


def hours_since_epoch(stamp):
    """Hours from 2026-11-19T00:00Z to an ISO 8601 UTC timestamp."""
    moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (moment - EPOCH).total_seconds() / 3600.0


def log_time(hours):
    """Log-time axis: ln(1 + hours since release); defined back to -1 h."""
    return math.log1p(max(hours, -0.999))


# ---------------------------------------------------------------- fitting


def weighted_line(points):
    """Weighted least squares of y on x. points are (x, y, weight).

    Returns (intercept, slope, weighted RMS residual). With fewer than two
    distinct x values the slope is 0 and the line sits at the weighted mean.
    """
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        resid = math.sqrt(sum(w * (y - my) ** 2 for _, y, w in points) / sw)
        return my, 0.0, resid
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in points) / sw
    )
    return intercept, slope, resid


def weighted_means(points):
    """Weighted mean of x and of y, for re-anchoring a shrunk slope."""
    sw = sum(w for _, _, w in points)
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    return mx, my


def trophy_points(case, idx, last_hour, reads=None):
    """One trophy's usable reads as (log-time, logit rate, weight) triples.

    A read is skipped when it lacks the trophy, when the rate is 0 (a
    censored zero would drag the logit to -9.2), or when it is not a number.
    Weights decay with age so the freshest reads steer the line.
    `reads` defaults to the full history; the v2 path passes deduplicated
    reads so one flat stretch counts once.
    """
    points = []
    for read in (case["history"] if reads is None else reads):
        rates = read.get("rates") or []
        if idx >= len(rates):
            continue
        rate = rates[idx]
        if rate is None or not isinstance(rate, (int, float)):
            continue
        if rate <= 0:
            continue
        hour = hours_since_epoch(read["at"])
        age = max(0.0, last_hour - hour)
        weight = 0.5 ** (age / WEIGHT_HALF_LIFE_H)
        points.append((log_time(hour), logit_of(rate), weight))
    return points


def tier_lines(case, last_hour, reads=None):
    """One shared (intercept, slope, residual) line per tier, pooled over
    the tier's trophies. A tier with a single trophy gets that trophy's line.

    The residual is the mean of the trophies' own fit residuals, NOT the
    pooled fit's residual: the pooled residual is dominated by the trophies'
    different levels (one bronze sits at 100%, most near 1%), which is level
    dispersion, not misfit, and would inflate every short-history spread.
    `reads` defaults to the full history; the v2 path passes deduplicated
    reads.
    """
    by_tier = {}
    for idx, trophy in enumerate(case["trophies"]):
        by_tier.setdefault(trophy["tier"], []).append(
            trophy_points(case, idx, last_hour, reads))
    lines = {}
    for tier, per_trophy in by_tier.items():
        pooled = [p for pts in per_trophy for p in pts]
        own_resids = [weighted_line(pts)[2] for pts in per_trophy if len(pts) >= 2]
        resid = (sum(own_resids) / len(own_resids)) if own_resids else RESID_FLOOR
        if len(pooled) >= 2:
            intercept, slope, _ = weighted_line(pooled)
            lines[tier] = (intercept, slope, resid)
        elif pooled:
            lines[tier] = (pooled[0][1], 0.0, resid)
        else:
            lines[tier] = (logit_of(1.0), 0.0, resid)
    return lines


def usable_reads(case):
    """History with consecutive duplicate reads collapsed to one.

    Exophase reads repeat verbatim while no refresh has happened; seven
    identical reads are one observation, not seven, and counting them seven
    times would over-weight a flat stretch in every fit.
    """
    reads = []
    prev_key = None
    for read in case.get("history") or []:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if key == prev_key:
            continue
        prev_key = key
        reads.append(read)
    return reads


def is_endgame(trophy):
    """True when the description demands endgame-scale effort.

    Platinum, 100% completion, a story finale, or a large-count grind
    (250 stunts, 100 locations, 50 challenges, 8 mission strands, card
    sets, every location/spot). These are the trophies whose launch-night
    rates are phantom: a handful of earners recorded before the first
    refresh, zero once the tracked cohort grows. Numbered grinds need a
    count of at least 20 so ordinary mission trophies ("Chapter 3", "5
    enemies") are not caught.
    """
    if trophy.get("tier") == "platinum":
        return True
    desc = (trophy.get("description") or "").lower()
    if "100% completion" in desc or "final mission" in desc:
        return True
    if ("every location" in desc or "every spot" in desc
            or "card set" in desc or "mission strand" in desc
            or "shooting range" in desc):
        return True
    nums = [int(n) for n in re.findall(r"\d+", desc)]
    if any(n >= 20 for n in nums) and any(
            word in desc for word in
            ("stunt", "location", "challenge", "race", "vehicle", "vault")):
        return True
    return False


def players_points(reads, last_hour):
    """One (log-time, ln players-tracked, weight) triple per usable read."""
    points = []
    for read in reads:
        n = read.get("players_tracked")
        if not isinstance(n, (int, float)) or n <= 0:
            continue
        hour = hours_since_epoch(read["at"])
        age = max(0.0, last_hour - hour)
        weight = 0.5 ** (age / WEIGHT_HALF_LIFE_H)
        points.append((log_time(hour), math.log(n), weight))
    return points


def players_line(reads, last_hour):
    """Power-law line for the tracked-player count: ln N on ln(1+hours)."""
    points = players_points(reads, last_hour)
    if len(points) >= 2:
        return weighted_line(points)
    if points:
        return points[0][1], 0.0, RESID_FLOOR
    return math.log(638.0), 0.0, RESID_FLOOR


def earners_points(reads, idx, last_hour):
    """One trophy's earners as (log-time, ln earners, weight) triples.

    Earners = rate * players_tracked / 100, from the same read: the rate
    alone cannot tell a growing trophy from a diluted one, but earners can.
    A read is skipped when the rate is 0 (a censored zero: ln 0 is
    undefined, and a 0.00% print means "fewer than ~N/20000 earners", not
    zero earners) or when the player count is missing.
    """
    points = []
    for read in reads:
        rates = read.get("rates") or []
        n = read.get("players_tracked")
        if idx >= len(rates):
            continue
        if not isinstance(n, (int, float)) or n <= 0:
            continue
        rate = rates[idx]
        if rate is None or not isinstance(rate, (int, float)):
            continue
        if rate <= 0:
            continue
        earners = rate * n / 100.0
        hour = hours_since_epoch(read["at"])
        age = max(0.0, last_hour - hour)
        weight = 0.5 ** (age / WEIGHT_HALF_LIFE_H)
        points.append((log_time(hour), math.log(earners), weight))
    return points


def tier_earners_lines(reads, case, last_hour):
    """One shared (intercept, slope, residual) earners line per tier.

    Only the slope is used (as the shrinkage prior); the residual is the
    mean of the trophies' own fit residuals, as in tier_lines, so level
    dispersion across trophies cannot inflate it.
    """
    by_tier = {}
    for idx, trophy in enumerate(case["trophies"]):
        by_tier.setdefault(trophy["tier"], []).append(
            earners_points(reads, idx, last_hour))
    lines = {}
    for tier, per_trophy in by_tier.items():
        pooled = [p for pts in per_trophy for p in pts]
        own_resids = [weighted_line(pts)[2] for pts in per_trophy if len(pts) >= 2]
        resid = (sum(own_resids) / len(own_resids)) if own_resids else RESID_FLOOR
        if len(pooled) >= 2:
            intercept, slope, _ = weighted_line(pooled)
            lines[tier] = (intercept, slope, resid)
        elif pooled:
            lines[tier] = (pooled[0][1], 0.0, resid)
        else:
            lines[tier] = (math.log(1.0), 0.0, resid)
    return lines


def dilution_sensitivity(last_rate):
    """How much a refresh's player-count jump can move this trophy's rate.

    A low rate can be divided several-fold at once; a rate near 100% barely
    moves, because every tracked player already holds that trophy.
    """
    sens = 1.0 / (1.0 + last_rate / 2.0)
    return min(1.0, max(0.05, sens))


def forecast_trophy(points, tier_line, last_rate, target_x, horizon_days):
    """Median, spread (logit units) and shrunk slope for one trophy at the target."""
    n = len(points)
    own_intercept, own_slope, own_resid = weighted_line(points) if n >= 2 else (None, None, None)
    tier_intercept, tier_slope, tier_resid = tier_line
    if n >= 3:
        # Hierarchical shrinkage of the SLOPE only, toward the tier's slope.
        # The level is the trophy's own: trophies in one tier sit at wildly
        # different rates (one bronze is at 100%, most near 1%), so pulling
        # the intercept toward the tier mean would erase real level gaps.
        # The shrunk line is re-anchored through the trophy's own weighted
        # mean, so its level stays its own.
        prior = SHRINK_PRIOR_READS
        slope = (n * own_slope + prior * tier_slope) / (n + prior)
        mx, my = weighted_means(points)
        intercept = my - slope * mx
        resid = min(RESID_CAP, max(RESID_FLOOR, own_resid))
    elif n >= 1:
        # Too few reads to trust a slope: persist the last rate, but let the
        # tier line's level temper it slightly through the intercept.
        intercept = logit_of(last_rate)
        slope = 0.0
        resid = min(RESID_CAP, max(RESID_FLOOR, tier_resid))
    else:
        # No usable read at all: fall back to the tier line itself.
        intercept, slope = tier_intercept, tier_slope
        resid = min(RESID_CAP, max(RESID_FLOOR, tier_resid))
    median_logit = intercept + slope * target_x
    walk = WALK_SIGMA * math.sqrt(horizon_days)
    dilution = (DILUTION_BASE + DILUTION_PER_SQRT_DAY * math.sqrt(horizon_days)) \
        * dilution_sensitivity(last_rate)
    spread = walk + resid + dilution
    return median_logit, spread, slope


def forecast_trophy_v2(earn_points, earn_tier_line, n_line, last_rate,
                       last_n, target_x, horizon_days):
    """Median and spread for one trophy from separate earners/players fits.

    Returns (median_logit, spread, fitted earners at target, shrunk
    earners slope). The rate forecast is fitted earners divided by fitted
    players, so a refresh that only grows the tracked cohort dilutes the
    rate without touching earners. The slope is shrunk toward the tier's
    earners slope exactly as v1 shrinks the rate slope; the level stays
    the trophy's own. The spread is the raw walk + residual + dilution
    sum; the caller scales it.
    """
    n = len(earn_points)
    tier_intercept, tier_slope, tier_resid = earn_tier_line
    n_intercept, n_slope, _ = n_line
    n_fc = math.exp(n_intercept + n_slope * target_x)
    n_fc = min(max(n_fc, last_n), 20.0 * last_n)
    if n >= 3:
        own_intercept, own_slope, own_resid = weighted_line(earn_points)
        prior = SHRINK_PRIOR_READS
        slope = (n * own_slope + prior * tier_slope) / (n + prior)
        mx, my = weighted_means(earn_points)
        intercept = my - slope * mx
        resid = min(RESID_CAP, max(RESID_FLOOR, own_resid))
    elif n >= 1:
        mx, my = weighted_means(earn_points)
        intercept, slope = my, 0.0
        resid = min(RESID_CAP, max(RESID_FLOOR, tier_resid))
    else:
        intercept, slope = tier_intercept, tier_slope
        resid = min(RESID_CAP, max(RESID_FLOOR, tier_resid))
    earn_fc = math.exp(intercept + slope * target_x)
    median_logit = logit_of(100.0 * earn_fc / n_fc)
    walk = WALK_SIGMA * math.sqrt(horizon_days)
    base_rate = last_rate if isinstance(last_rate, (int, float)) else 1.0
    dilution = (DILUTION_BASE + DILUTION_PER_SQRT_DAY * math.sqrt(horizon_days)) \
        * dilution_sensitivity(base_rate)
    spread = walk + resid + dilution
    return median_logit, spread, earn_fc, slope


def assemble_quantiles(median_logit, spread, floored, pinned_high):
    """Five quantiles, with the floor / ceiling pinned one-sided.

    A floored trophy can only surprise upward (its lower quantiles sit on
    the 0.01% clamp); a 100% trophy can only surprise downward.
    """
    quantiles = []
    for z in Z_SCORES:
        if floored and z <= 0:
            quantiles.append(0.01)
        elif pinned_high and z >= 0:
            quantiles.append(99.99)
        else:
            quantiles.append(round_half_up_4(
                clamp_pct(pct_of_logit(median_logit + z * spread))))
    for k in range(1, len(quantiles)):
        if quantiles[k] < quantiles[k - 1]:
            quantiles[k] = quantiles[k - 1]
    return quantiles


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels other than the scored five")
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    last_hour = hours_since_epoch(history[-1]["at"])
    target_x = log_time(hours_since_epoch(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    reads = usable_reads(case)
    has_players = any(
        isinstance(r.get("players_tracked"), (int, float))
        and r["players_tracked"] > 0 for r in reads)
    last_n = 638.0
    for read in reversed(history):
        n = read.get("players_tracked")
        if isinstance(n, (int, float)) and n > 0:
            last_n = float(n)
            break

    if has_players:
        n_line = players_line(reads, last_hour)
        earn_tiers = tier_earners_lines(reads, case, last_hour)
        rate_tiers = tier_lines(case, last_hour, reads)
    else:
        # No player counts anywhere: fall back to the v1 rate-space fit.
        tiers = tier_lines(case, last_hour)

    last_rates = history[-1].get("rates") or []
    tier_last = {}
    for idx, trophy in enumerate(case["trophies"]):
        rate = last_rates[idx] if idx < len(last_rates) else None
        if isinstance(rate, (int, float)) and rate > 0:
            tier_last.setdefault(trophy["tier"], []).append(rate)
    tier_fallback = {
        tier: (sorted(rates)[len(rates) // 2] if rates else 1.0)
        for tier, rates in tier_last.items()
    }

    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        rate = last_rates[idx] if idx < len(last_rates) else None
        if not (isinstance(rate, (int, float)) and rate > 0):
            rate = tier_fallback.get(trophy["tier"], 1.0)
        if has_players:
            # Two models of the same rate: the saturating logit trend (v1)
            # and the earners-over-players power law. Each is trusted only
            # in the regime where its shape is right. When the rate is
            # FALLING while earners are flat or growing, the fall is
            # dilution: the logit trend would extrapolate the cliff into
            # oblivion, so the earners/players ratio is used. In every
            # other regime (flat, or growing) the logit trend is used,
            # because the power law is unbounded and would otherwise print
            # 99.99% for trophies whose true rate is ~60%.
            rate_points = trophy_points(case, idx, last_hour, reads)
            logit_med, logit_spread, logit_slope = forecast_trophy(
                rate_points, rate_tiers[trophy["tier"]], rate,
                target_x, horizon_days)
            earn_points = earners_points(reads, idx, last_hour)
            en_med, en_spread, earn_fc, earn_slope = forecast_trophy_v2(
                earn_points, earn_tiers[trophy["tier"]], n_line, rate,
                last_n, target_x, horizon_days)
            if logit_slope < 0.0 and earn_slope >= 0.0:
                median_logit, spread = en_med, en_spread
            else:
                median_logit, spread = logit_med, logit_spread
            spread = max(SPREAD_FLOOR, spread * SPREAD_SCALE)
            floored = is_endgame(trophy) and earn_fc < ENDGAME_EARNERS_GATE
            pinned_high = isinstance(rate, (int, float)) and rate >= 100.0
            if floored:
                median_logit = logit_of(0.01)
            elif pinned_high:
                median_logit = logit_of(99.99)
            quantiles = assemble_quantiles(median_logit, spread, floored, pinned_high)
        else:
            points = trophy_points(case, idx, last_hour)
            median_logit, spread, _slope = forecast_trophy(
                points, tiers[trophy["tier"]], rate, target_x, horizon_days)
            quantiles = []
            for z in Z_SCORES:
                q = round_half_up_4(clamp_pct(pct_of_logit(median_logit + z * spread)))
                quantiles.append(q)
            # Rounding can only move a value by 0.00005, but enforce the
            # non-decreasing rule explicitly so the row is valid by construction.
            for k in range(1, len(quantiles)):
                if quantiles[k] < quantiles[k - 1]:
                    quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


# ---------------------------------------------------------------- entry point


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
