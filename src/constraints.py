"""
constraints.py — Hard constraint checkers.

All checks return (ok: bool, reason: str).
Handles new holdings format: {ticker: {"qty": int, "avg_price": float}}
"""

MAX_HOLDINGS    = 30       # TC004 — DISQUALIFYING if breached
MAX_TURNOVER    = 0.30     # TC005 — DISQUALIFYING if breached
MIN_POSITION_W  = 0.005    # 0.5% minimum weight if held
PROP_FEE        = 0.001    # 0.1% proportional transaction cost
FIXED_FEE       = 1.00     # $1 fixed per order


def _holding_count(holdings: dict) -> int:
    """Count active positions. Works with both {ticker: int} and {ticker: {qty, avg_price}}."""
    count = 0
    for val in holdings.values():
        if isinstance(val, dict):
            if val.get("qty", 0) > 0:
                count += 1
        elif isinstance(val, (int, float)):
            if val > 0:
                count += 1
    return count


def _is_held(holdings: dict, ticker: str) -> bool:
    """Check if ticker is currently held."""
    val = holdings.get(ticker)
    if val is None:
        return False
    if isinstance(val, dict):
        return val.get("qty", 0) > 0
    return int(val) > 0


def check_cardinality(
    current_holdings: dict,
    new_ticker: str,
    side: str,
) -> tuple[bool, str]:
    """
    Ensure adding a new position does not breach the 30-holding limit.
    If SELL or ticker already held, always passes.
    """
    if side.upper() == "SELL":
        return True, "ok"
    if _is_held(current_holdings, new_ticker):
        return True, "ok"
    n = _holding_count(current_holdings)
    if n >= MAX_HOLDINGS:
        return False, f"TC004: cardinality breach — already {n} holdings (max {MAX_HOLDINGS})"
    return True, "ok"


def check_turnover(
    current_turnover: float,
    proposed_trade_value: float,
    avg_portfolio_value: float,
) -> tuple[bool, str]:
    """Ensure the proposed trade would not push turnover above 30%."""
    if avg_portfolio_value <= 0:
        return True, "ok"
    added = abs(proposed_trade_value) / avg_portfolio_value
    new_turnover = current_turnover + added
    if new_turnover > MAX_TURNOVER:
        return False, (
            f"TC005: would breach turnover — current={current_turnover:.2%}, "
            f"adds={added:.2%}, total={new_turnover:.2%} > {MAX_TURNOVER:.0%}"
        )
    return True, "ok"


def check_min_position(weight: float) -> tuple[bool, str]:
    """A non-zero position must be >= 0.5% of portfolio."""
    if weight == 0.0:
        return True, "ok"
    if weight < MIN_POSITION_W:
        return False, f"Min position breach — weight={weight:.4%} < {MIN_POSITION_W:.1%}"
    return True, "ok"


def check_long_only(qty: int, side: str, current_qty: int) -> tuple[bool, str]:
    """Ensure we never go short."""
    if side.upper() == "SELL" and qty > current_qty:
        return False, f"Long-only breach — sell {qty} but hold {current_qty}"
    return True, "ok"


def compute_fees(qty: int, price: float) -> float:
    """Transaction cost: 0.1% proportional + $1 fixed."""
    return PROP_FEE * qty * price + FIXED_FEE


def max_affordable_qty(
    cash: float,
    price: float,
    max_qty: int | None = None,
) -> int:
    """Max integer shares buyable given cash, price, and fees."""
    if price <= 0 or cash <= FIXED_FEE:
        return 0
    affordable = int((cash - FIXED_FEE) / (price * (1 + PROP_FEE)))
    affordable = max(0, affordable)
    if max_qty is not None:
        affordable = min(affordable, max_qty)
    return affordable
