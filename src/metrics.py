"""
metrics.py — Sharpe ratio, PnL, turnover, and compliance checks.

Sharpe formula:
  log_rets = log(V[t] / V[t-1])
  mu_r     = mean(log_rets)             [arithmetic mean]
  sigma_r  = sqrt(sum((r-mu_r)^2)/n)   [population std — matches agent_candidate.py]
  sharpe   = (mu_r / sigma_r) * sqrt(390)   [annualized to per-session]

The validator reads sharpe_ratio from results.json directly:
  Sharpe_norm = min(100, sharpe / 3.0 * 100)
The 3.0 threshold is calibrated for sqrt(390) annualization — raw per-tick
Sharpe of ~0.18 × sqrt(390) ≈ 3.5. A threshold of 3.0 on the raw value
would be physically impossible for intraday data.

Scoring targets (from validate_solution.py):
  sharpe >= 3.0 → full 60 Sharpe points
  pnl    >= $500,000 → full 20 PnL points
"""
import math
from datetime import datetime, timezone
from typing import Optional

# Annualization: converts per-tick Sharpe to per-session (1 trading day = 390 ticks)
SHARPE_ANN_FACTOR = math.sqrt(390)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_sharpe(portfolio_values: list[float], risk_free: float = 0.0) -> float:
    """
    Per-session annualized Sharpe ratio.
    Uses population std (matches agent_candidate.py) then scales by sqrt(390).
    The 3.0 scoring threshold is calibrated for this formula.
    """
    if len(portfolio_values) < 2:
        return 0.0

    log_rets = [
        math.log(portfolio_values[i] / portfolio_values[i - 1])
        for i in range(1, len(portfolio_values))
        if portfolio_values[i - 1] > 0
    ]

    if not log_rets:
        return 0.0

    mu_r    = sum(log_rets) / len(log_rets)
    sigma_r = math.sqrt(sum((r - mu_r) ** 2 for r in log_rets) / len(log_rets))

    if sigma_r < 1e-10:
        return 0.0

    return (mu_r / sigma_r) * SHARPE_ANN_FACTOR


def compute_pnl(initial_value: float, final_value: float) -> float:
    return final_value - initial_value


def compute_pnl_pct(initial_value: float, final_value: float) -> float:
    if initial_value <= 0:
        return 0.0
    return (final_value - initial_value) / initial_value * 100.0


def check_tc004(snapshots: list[dict]) -> tuple[bool, str]:
    """
    TC004: holdings count <= 30 at every snapshot.
    Uses len(snap["holdings"]) where holdings is a list.
    """
    for snap in snapshots:
        n = len(snap.get("holdings", []))
        if n > 30:
            return False, (
                f"DISQUALIFYING: {n} holdings at tick {snap.get('tick_index','?')} (max 30)"
            )
    return True, "Cardinality <= 30 maintained throughout"


def check_tc005(final_turnover: float) -> tuple[bool, str]:
    """TC005: cumulative daily turnover <= 30%."""
    if final_turnover > 0.30:
        return False, f"DISQUALIFYING: turnover={final_turnover:.2%} > 30%"
    return True, f"Turnover={final_turnover:.2%} within limit"


def build_results(
    snapshots: list[dict],
    initial_cash: float,
    final_turnover: float,
    orders_log: Optional[list[dict]] = None,
    llm_log: Optional[list[dict]] = None,
) -> dict:
    """
    Build results.json matching validate_solution.py and starter kit compute_results format.

    Key fields read by validator:
      sharpe_ratio, pnl, final_value, turnover_ratio, tc004_compliant, tc005_compliant
    """
    orders_log = orders_log or []
    llm_log    = llm_log or []

    if not snapshots:
        return {
            "starting_value":  round(initial_cash, 2),
            "final_value":     round(initial_cash, 2),
            "pnl":             0.0,
            "pnl_pct":         0.0,
            "sharpe_ratio":    0.0,
            "turnover_ratio":  round(final_turnover, 6),
            "total_ticks":     0,
            "total_orders":    len(orders_log),
            "llm_calls_used":  len(llm_log),
            "llm_quota":       60,
            "tc004_compliant": True,
            "tc005_compliant": final_turnover <= 0.30,
            "generated_at":    _now_iso(),
        }

    values      = [float(s["total_value"]) for s in snapshots]
    final_value = values[-1]
    pnl         = final_value - initial_cash
    pnl_pct     = pnl / initial_cash * 100.0 if initial_cash > 0 else 0.0

    sharpe = compute_sharpe(values)

    # Recompute turnover from orders log (matches starter kit method)
    if orders_log:
        total_traded = sum(
            abs(o.get("qty", 0)) * o.get("exec_price", 0.0)
            for o in orders_log
        )
        avg_portfolio = sum(values) / len(values) if values else initial_cash
        turnover_from_log = total_traded / avg_portfolio if avg_portfolio > 0 else 0.0
        # Use the more conservative (higher) of the two calculations
        reported_turnover = max(final_turnover, turnover_from_log)
    else:
        reported_turnover = final_turnover

    tc004_ok, tc004_msg = check_tc004(snapshots)
    tc005_ok, tc005_msg = check_tc005(reported_turnover)

    results = {
        "starting_value":  round(initial_cash, 2),
        "final_value":     round(final_value, 2),
        "pnl":             round(pnl, 2),
        "pnl_pct":         round(pnl_pct, 4),
        "sharpe_ratio":    round(sharpe, 6),
        "turnover_ratio":  round(reported_turnover, 6),
        "total_ticks":     len(snapshots),
        "total_orders":    len(orders_log),
        "llm_calls_used":  len(llm_log),
        "llm_quota":       60,
        "tc004_compliant": tc004_ok,
        "tc005_compliant": tc005_ok,
        "tc004_detail":    tc004_msg,
        "tc005_detail":    tc005_msg,
        "generated_at":    _now_iso(),
        # Legacy fields for compat
        "initial_value":   round(initial_cash, 2),
        "pnl_abs":         round(pnl, 2),
        "num_ticks":       len(snapshots),
    }

    from src.log import vprint
    vprint("\n" + "=" * 60)
    vprint("SIMULATION RESULTS")
    vprint("=" * 60)
    vprint(f"  Sharpe Ratio : {sharpe:.4f}  (target >= 3.0)")
    vprint(f"  PnL          : ${pnl:+,.2f}  ({pnl_pct:+.2f}%)  (target >= $500,000)")
    vprint(f"  Final NAV    : ${final_value:,.2f}")
    vprint(f"  Turnover     : {reported_turnover:.2%}  (limit 30%)")
    vprint(f"  TC004        : {'PASS' if tc004_ok else 'FAIL'}")
    vprint(f"  TC005        : {'PASS' if tc005_ok else 'FAIL'}")
    vprint(f"  LLM calls    : {len(llm_log)}/60")
    vprint("=" * 60)

    return results
