"""
optimizer.py — Mean-Variance Optimization via cvxpy with transaction cost awareness.

Objective:
    Maximize: w^T * mu - gamma * w^T * Sigma * w - kappa * ||w - w_prev||_1

The third term penalises turnover proportionally to the transaction cost rate (kappa).
This makes the solver intrinsically reluctant to trade unless expected improvement
exceeds the cost — no need to hard-code per-tick deployment caps in the objective.

Constraints:
    sum(w) == 1, w >= 0, w[i] <= MAX_WEIGHT, cardinality <= 30, turnover <= budget

Note on sector constraints: sector-based weight caps were evaluated but removed.
With only 30% turnover budget the portfolio holds 10-20 names at most; sector
constraints created infeasibility and over-concentration when high-quality names
clustered in one sector. The covariance matrix (LW + factor model) already handles
sector risk appropriately.

Note on beta adjustment: mu / beta was evaluated but removed. The LW+factor
covariance already penalises high-beta stocks through the quadratic risk term
(w^T * Sigma * w). Double-penalising beta via mu distorts the signal.
"""
import warnings
import numpy as np
import cvxpy as cp
from typing import Optional

from src.log import vprint

GAMMA             = 1.0    # risk aversion (overridden by env var in main.py)
MAX_WEIGHT        = 0.15   # max single-position weight (15% — TC004 allows 30 holdings)
MAX_CARDINALITY   = 30     # TC004
MAX_TURNOVER_W    = 0.30   # TC005 — max sum(|w - w_prev|)
MIN_WEIGHT        = 0.005  # min position if selected

# Transaction cost penalty in the objective. Set to 0.001 (0.1%) which is
# equal to the actual one-way fee. Lower values allow more responsive rotation
# into high-return stocks as trends are identified by the EWMA signal.
KAPPA = 0.001


def optimize(
    tickers: list[str],
    mu: list[float],
    sigma: list[list[float]],
    w_prev: list[float],
    gamma: float = GAMMA,
    remaining_turnover: float = MAX_TURNOVER_W,
    kappa: float = KAPPA,
    ticker_max_weights: Optional[dict[str, float]] = None,
    sector_groups: Optional[dict[str, list[int]]] = None,   # kept for API compat, unused
    beta_vector: Optional[list[float]] = None,               # kept for API compat, unused
    beta_adjust: bool = False,                               # disabled — see module docstring
    max_weight: float = MAX_WEIGHT,                          # per-ticker weight cap from params
    min_weight: float = MIN_WEIGHT,                          # min position size from params
) -> dict[str, float]:
    """
    Run MVO and return target weights {ticker: weight}.

    Parameters
    ----------
    tickers           : ordered list of ticker symbols
    mu                : expected returns vector (same order)
    sigma             : covariance matrix (n x n)
    w_prev            : previous weights (same order, sum <= 1)
    gamma             : risk aversion coefficient (scaled by vol regime in main.py)
    remaining_turnover: hard turnover budget remaining this simulation
    kappa             : transaction cost penalty in objective (default 0.1%)
    ticker_max_weights: optional per-ticker weight caps {ticker: max_weight}
    sector_groups     : deprecated, kept for API compatibility
    beta_vector       : deprecated, kept for API compatibility
    beta_adjust       : deprecated (False), kept for API compatibility
    """
    n = len(tickers)
    mu_arr     = np.array(mu,    dtype=float)
    Sigma_arr  = np.array(sigma, dtype=float)
    w_prev_arr = np.array(w_prev, dtype=float)

    w = cp.Variable(n, nonneg=True)

    # Turnover variable for both the objective penalty and the hard constraint
    # delta[i] = |w[i] - w_prev[i]|  (non-negative by construction with aux vars)
    delta = cp.Variable(n, nonneg=True)

    objective = cp.Maximize(
        mu_arr @ w
        - gamma * cp.quad_form(w, Sigma_arr)
        - kappa * cp.sum(delta)     # transaction cost penalty
    )

    constraints = [
        cp.sum(w) == 1.0,
        w <= max_weight,
        # |w - w_prev| <= delta  (linearisation of absolute value)
        w - w_prev_arr <= delta,
        w_prev_arr - w <= delta,
    ]

    # Only add the weight-space turnover constraint when the portfolio is
    # sufficiently invested. The constraint sum(|w - w_prev|) <= budget requires
    # that budget >= sum(w) - sum(w_prev) = 1 - prev_invested, otherwise the
    # combination of sum(w)=1 and the budget makes the problem infeasible.
    # We skip it when the minimum achievable turnover (1 - prev_invested) would
    # exceed the budget; dollar-level execution handles the cap in those cases.
    prev_invested = float(np.sum(w_prev_arr))
    min_achievable_turnover = max(0.0, 1.0 - prev_invested)
    if prev_invested > 0.01 and remaining_turnover >= min_achievable_turnover - 1e-4:
        turnover_budget = max(0.0, min(remaining_turnover, MAX_TURNOVER_W))
        constraints.append(cp.sum(delta) <= turnover_budget)

    prob = cp.Problem(objective, constraints)

    # Try OSQP first (strict tolerances), then SCS.
    # OSQP accepts "optimal_inaccurate" if warm_start confuses it — we only
    # trust "optimal" from OSQP and fall through to SCS otherwise.
    solvers = [
        (cp.OSQP, dict(warm_start=True, eps_abs=1e-8, eps_rel=1e-8, verbose=False)),
        (cp.SCS,  dict(warm_start=True, verbose=False)),
    ]
    for solver, kwargs in solvers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=solver, **kwargs)
            if prob.status == "optimal" and w.value is not None:
                raw_weights = w.value
                result = _clean_weights(tickers, raw_weights, min_weight=min_weight)
                return _apply_ticker_caps(result, ticker_max_weights)
            if prob.status == "optimal_inaccurate" and w.value is not None:
                # SCS sometimes reports this but the solution is still usable
                if solver == cp.SCS:
                    raw_weights = w.value
                    result = _clean_weights(tickers, raw_weights, min_weight=min_weight)
                    return _apply_ticker_caps(result, ticker_max_weights)
                # OSQP inaccurate → try SCS
                vprint(f"[Optimizer] OSQP inaccurate — retrying with SCS")
                continue
        except Exception as e:
            vprint(f"[Optimizer] solver {solver} failed: {e}")
            continue

    # Fallback: equal weight among top-N momentum tickers
    vprint("[Optimizer] WARNING: all solvers failed — using fallback equal weights")
    result = _fallback_weights(tickers, mu_arr)
    return _apply_ticker_caps(result, ticker_max_weights)


def _clean_weights(tickers: list[str], raw: np.ndarray,
                   min_weight: float = MIN_WEIGHT) -> dict[str, float]:
    """
    Post-process raw weights:
    - Zero out positions below min_weight
    - Enforce cardinality limit (keep top MAX_CARDINALITY by weight)
    - Renormalise to sum=1
    """
    weights = np.clip(raw, 0, None)

    # Zero out sub-minimum positions
    weights[weights < min_weight] = 0.0

    # Enforce cardinality: keep top MAX_CARDINALITY by weight
    if np.sum(weights > 0) > MAX_CARDINALITY:
        threshold = np.sort(weights)[::-1][MAX_CARDINALITY - 1]
        weights[weights < threshold] = 0.0

    # Re-normalise
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        # Absolute fallback: equal weight first ticker
        weights = np.zeros(len(tickers))
        weights[0] = 1.0

    return {t: float(w) for t, w in zip(tickers, weights) if w > 0}


def _fallback_weights(tickers: list[str], mu: np.ndarray) -> dict[str, float]:
    """Equal weight among top-N highest expected return tickers."""
    n_hold = min(MAX_CARDINALITY, len(tickers))
    top_idx = np.argsort(mu)[::-1][:n_hold]
    w = np.zeros(len(tickers))
    w[top_idx] = 1.0 / n_hold
    return {tickers[i]: float(w[i]) for i in range(len(tickers)) if w[i] > 0}


def _apply_ticker_caps(
    weights: dict[str, float],
    ticker_max_weights: Optional[dict[str, float]],
) -> dict[str, float]:
    """Apply per-ticker weight caps and renormalise. No-op if caps is None/empty."""
    if not ticker_max_weights:
        return weights
    modified = False
    result = dict(weights)
    for ticker, max_w in ticker_max_weights.items():
        if ticker in result and result[ticker] > max_w:
            result[ticker] = max_w
            modified = True
    if modified:
        total = sum(result.values())
        if total > 1e-9:
            result = {t: w / total for t, w in result.items()}
    return result
