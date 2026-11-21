# GTA 6 Trophy Rarity Forecast

Forecasts how many GTA VI players earn each trophy in week one — a Python program (standard library only) that reads Exophase case input and writes 55 rows of quantiles per trophy.

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tiVan0/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tiVan0/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

## Method

Earners and players are modelled separately (log-time power laws) and divided; a regime switch picks the logit trend or the earners/players ratio by slope; endgame trophies get a floor. Spread calibrated on out-of-sample backtests.

## Entry

`main.py` — run with `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`; deterministic, byte-identical outputs across runs.
