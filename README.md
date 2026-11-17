# Portfolio Optimization Agent

A transaction-cost-aware portfolio optimisation agent for a 390-tick simulated trading
session (one trading day, one tick per minute, 50 tickers, $10M starting cash). The agent
runs on quantitative signals alone by default; the LLM overlay is optional and was unused
in the submitted run.

**Background.** I built this as my solution to the BlackRock Hackathon@IITD 2026 portfolio-optimisation
challenge. The market data, corporate-action schedule, scoring rubric and the files in `Starter_Kit/`
were provided by the organisers; the agent in `src/` and the research scripts are my implementation.

## Results

From the committed run in [`results.json`](results.json):

| Metric | Value |
|---|---|
| Sharpe ratio | **3.499** |
| PnL | **+$504,142** (+5.04%) |
| Turnover | 29.01% (limit 30%) |
| Orders | 26 |
| LLM calls used | 0 of 60 |
| TC004 (≤ 30 holdings) | pass |
| TC005 (≤ 30% turnover) | pass |

The full scoring breakdown against the organisers' rubric, and the per-test-case
compliance detail, are in [`STRATEGY.md`](STRATEGY.md).

## How it works

Two layers, rebalanced every 5 ticks:

1. **Event-driven pre-positioning.** Publicly disclosed corporate-action schedules are used
   to pre-buy ahead of events, as forced orders that bypass the paced turnover budget.
2. **Mean-variance optimisation.** EWMA and momentum signals blended 0.7/0.3, a Ledoit-Wolf
   and factor-model covariance blend, and a transaction-cost penalty in the objective,
   solved under long-only, per-name weight, cardinality and turnover constraints
   (OSQP, falling back to SCS).

A volatility-regime detector scales risk aversion and rebalance frequency. Execution runs in
two passes so that compliance-forced trades are never crowded out by the optimiser.

[`STRATEGY.md`](STRATEGY.md) documents the full pipeline, every tunable parameter, the
corporate-action handling, the grid-search methodology, and a **lookahead-bias audit** that
marks which components are clean and which are not.

## Repository layout

```
.
├── README.md
├── STRATEGY.md                  # full strategy and codebase guide
├── pyproject.toml / uv.lock
├── cross_validate.py            # walk-forward cross-validation
├── grid_search.py               # 3-phase hyperparameter search
├── data/                        # market feed, corporate actions, fundamentals, portfolio
├── src/                         # the agent
│   ├── main.py                  # simulation entry point and per-tick pipeline
│   ├── signals.py               # EWMA, momentum, covariance hierarchy, vol regime
│   ├── optimizer.py             # MVO solver
│   ├── execution.py             # two-pass order execution
│   ├── constraints.py           # hard constraints and fees
│   ├── corporate_actions.py     # CA handling and signal boosts
│   ├── portfolio.py             # holdings, splits, dividends, turnover
│   ├── metrics.py               # Sharpe and results assembly
│   └── llm_client.py            # optional LLM overlay
└── Starter_Kit/                 # organiser-provided validator and reference agent
```

The JSON files at the repository root (`results.json`, `orders_log.json`,
`portfolio_snapshots.json`, `llm_call_log.json`, `cv_results.json`) are the committed
artefacts of the submitted run. A fresh run writes to `output/` instead.

## Setup

```bash
uv sync
```

## Run the agent

**Pure quant (no LLM):**
```bash
uv run python -m src.main
```

**With LLM:**
```bash
uv run python -m src.main --token <TOKEN> --llm <HOST:PORT>
```

Output files are written to `output/`.

## Validate a run

```bash
uv run python Starter_Kit/validate_solution.py \
  --orders    output/orders_log.json \
  --portfolio output/portfolio_snapshots.json \
  --llm_calls output/llm_call_log.json \
  --results   output/results.json \
  --ca        data/corporate_actions.json
```

## Cross-validate

```bash
# Default 5-fold walk-forward CV
uv run python cross_validate.py

# Custom fold count
uv run python cross_validate.py --folds 8

# Score a single tick window
uv run python cross_validate.py --mode range --test-start 200 --test-end 389
```

> **Note on the committed `cv_results.json`:** it records three custom folds plus a full run
> with `sharpe` and `pnl` left `null`, and its fold windows differ from the
> cross-validation table in `STRATEGY.md`. Re-run `cross_validate.py` and commit the output
> before relying on those figures.

## Author

Kabir Kashyap ([@wonkyqubit139](https://github.com/wonkyqubit139))
