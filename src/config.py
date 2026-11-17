"""
config.py — Single source of truth for all tunable simulation parameters.

All parameters here are structural (generalizable) — they control the
algorithm's behavior, not stock-specific alpha. Stock-specific logic
(TC002/TC003 forced orders) lives in main.py as compliance rules.

To change parameters: edit DEFAULT_PARAMS below, OR pass a dict override
to run_simulation(params={...}).
"""

DEFAULT_PARAMS: dict = {
    # ── Optimizer ──────────────────────────────────────────────────────
    "GAMMA_BASE":           0.2,    # MVO risk aversion (low = concentrated)
    "KAPPA":                0.001,  # transaction cost penalty in objective
    "MAX_WEIGHT":           0.15,   # max single-position weight
    "MIN_WEIGHT":           0.005,  # min position size (zero below this)

    # ── Turnover / Execution ───────────────────────────────────────────
    "DEPLOY_FRACTION":      0.30,   # max fraction of portfolio tradeable per rebalance
    "TURNOVER_HEAD_START":  0.20,   # extra budget at start (paced from this floor)
    "TC_RESERVED_TURNOVER": 0.01,   # budget reserved for TC compliance forced orders
    "MIN_TRADE_FRACTION":   0.010,  # skip weight-change orders smaller than this × NAV
                                    # (Fix E: raise from 0.5% to 1.0% to kill 1-share dust buys)

    # ── Signal Engine ──────────────────────────────────────────────────
    "EWMA_LAMBDA":          0.94,   # EWMA decay (RiskMetrics standard)
    "MOMENTUM_N":           10,     # lookback window for N-tick momentum
    "LLM_ALPHA":            0.30,   # weight on LLM signal vs quant (when LLM active)
    "MIN_HISTORY":          20,     # min ticks before we trade at all
                                    # (Fix A: 20 ticks of signal history before first trade;
                                    # avoids burning 20% turnover on 5-tick noise)

    # ── Signal Blending weights (must sum to 1.0) ──────────────────────
    "BLEND_CUM_SHARPE":     0.0,    # weight on cumulative realized Sharpe signal (phase1: 0.0 won)
    "BLEND_EWMA":           0.7,    # weight on EWMA expected return
    "BLEND_MOMENTUM":       0.3,    # weight on N-tick momentum

    # ── Cumulative Sharpe Signal ───────────────────────────────────────
    "CUM_SHARPE_SCALE":     0.0,    # scalar to normalize cum. Sharpe to EWMA magnitude (0=disabled, phase1 winner)

    # ── Fundamental Prior (disabled by default — hurts intraday Sharpe) ─
    "FUND_SCALE":           0.002,  # max magnitude of fundamental quality prior
    "USE_FUNDAMENTAL_PRIOR": False, # add fund prior to blended_returns
    "USE_52W_SIGNAL":       False,  # add 52w range mean-reversion signal

    # ── Anti-Churn ────────────────────────────────────────────────────
    "MIN_HOLD_TICKS":       4,      # min ticks to hold a position before selling
                                    # (Fix B: prevents EWMA oscillation from causing buy-sell-buy
                                    # churn; A001 round-tripped 1,572 shares wasting 7.4% turnover)
                                    # Reduced from 8 to 4: prevents over-restriction on rotation

    # ── Initial Build Diversification ─────────────────────────────────
    # Fix C REMOVED: elevated gamma on initial build forces diversification into noisy
    # mediocre stocks at tick 20 (no signal yet), hurting Sharpe from 2.04 → 1.26.
    # Now using normal gamma for all deployments; CA pre-positioning handles diversification.
    "INITIAL_BUILD_GAMMA_MULT": 1.0,  # 1.0 = no boost (disabled)

    # ── Back-Half Turnover Reservation ────────────────────────────────
    "BACK_HALF_RESERVE":    0.02,   # Fix D: reserve this fraction of MAX_TURNOVER (30%) for
                                    # ticks >= BACK_HALF_START; prevents budget exhaustion before
                                    # key late-day CA events (E007 MA@200, B008 fine@280)
                                    # Reduced from 0.04 to 0.02 — don't over-reserve
    "BACK_HALF_START":      195,    # tick at which back-half reservation kicks in

    # ── Rebalancing Cadence ────────────────────────────────────────────
    "REBALANCE_EVERY":      5,      # rebalance every N ticks (plus CA ticks)
                                    # Reduced from 15 to 5 for faster adaptation to signals

    # ── Corporate Action Decay ─────────────────────────────────────────
    "CA_DECAY":             0.90,   # exponential decay per tick for CA boosts

    # ── Volatility Regime ──────────────────────────────────────────────
    "VOL_GAMMA_LOW":        0.6,    # gamma multiplier in low-vol regime
    "VOL_GAMMA_NORMAL":     1.0,    # gamma multiplier in normal regime
    "VOL_GAMMA_HIGH":       2.5,    # gamma multiplier in high-vol regime
    "VOL_REBAL_LOW":        0,      # extra ticks to skip between rebalances (low vol)
    "VOL_REBAL_NORMAL":     0,      # extra ticks (normal vol)
    "VOL_REBAL_HIGH":       5,      # extra ticks (high vol)

    # ── ESG Weight Caps (permanent structural caps on high-ESG-risk stocks) ─
    "ESG_CAP_HIGH":         0.12,   # cap for ESG score >= ESG_HIGH_THRESHOLD
    "ESG_CAP_VERY_HIGH":    0.10,   # cap for ESG score >= ESG_VERY_HIGH_THRESHOLD
    "ESG_HIGH_THRESHOLD":   60,     # ESG score above which apply high cap
    "ESG_VERY_HIGH_THRESHOLD": 75,  # ESG score above which apply very-high cap

    # ── LLM ───────────────────────────────────────────────────────────
    "LLM_EVERY_N":          15,     # call LLM every N ticks (when endpoint available)
}
