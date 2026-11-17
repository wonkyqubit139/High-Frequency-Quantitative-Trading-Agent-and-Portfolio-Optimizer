"""
execution.py — Simulate order fills locally.

Every fill is validated against constraints, logged, and applied to Portfolio.
Fill record format matches the starter kit expected schema:
  {type, order_ref, ticker, side, qty, exec_price, fees, ts, tick_index, trade_value, turnover_after}
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.log import vprint
from src.constraints import (
    check_cardinality,
    check_turnover,
    check_long_only,
    compute_fees,
    max_affordable_qty,
    MAX_TURNOVER,
)

if TYPE_CHECKING:
    from src.portfolio import Portfolio
    from src.logger import SimLogger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_orders(
    orders: list[dict],
    portfolio: "Portfolio",
    prices: dict[str, float],
    tick_index: int,
    logger: "SimLogger",
    max_deploy_fraction: float = 1.0,
    turnover_budget_remaining: float | None = None,
    # TC-specific overrides: force certain tickers to be bought/excluded
    force_hold_tickers: set[str] | None = None,   # don't sell these (e.g. D002 post-split)
    forced_tc_orders: list[dict] | None = None,   # TC compliance orders that bypass paced budget
) -> list[dict]:
    """
    Process a batch of orders.

    max_deploy_fraction: max fraction of portfolio value to trade in this batch.
    turnover_budget_remaining: cap trades to this remaining paced budget.
    force_hold_tickers: tickers we must NOT sell (TC001 post-split protection).
    forced_tc_orders: TC compliance orders (TC002/TC003) that bypass the paced budget cap
                      but still respect the hard 30% TC005 ceiling and affordability.

    Returns list of fill records.
    """
    fills: list[dict] = []

    # ── Pass 1: Forced TC compliance orders (bypass paced cap, respect hard 30%) ──
    # These are critical for passing TC002/TC003 even when paced budget is exhausted.
    if forced_tc_orders:
        for order in forced_tc_orders:
            _execute_single(
                order=order,
                portfolio=portfolio,
                prices=prices,
                tick_index=tick_index,
                logger=logger,
                fills=fills,
                max_tick_trade=None,           # no per-batch cap
                tick_traded_value_ref=[0.0],   # separate counter (not shared with pass 2)
                force_hold=set(),              # forced orders always go through
                bypass_paced_cap=True,
            )

    # ── Pass 2: Normal optimizer-driven orders (subject to paced cap) ──
    tick_traded_value = 0.0
    portfolio_value   = portfolio.avg_portfolio_value or portfolio.cash
    max_tick_trade    = max_deploy_fraction * portfolio_value

    # Paced budget cap
    if turnover_budget_remaining is not None and portfolio_value > 0:
        paced_cap      = turnover_budget_remaining * portfolio_value
        max_tick_trade = min(max_tick_trade, paced_cap)

    force_hold = force_hold_tickers or set()
    tick_traded_value_ref = [0.0]

    for order in orders:
        _execute_single(
            order=order,
            portfolio=portfolio,
            prices=prices,
            tick_index=tick_index,
            logger=logger,
            fills=fills,
            max_tick_trade=max_tick_trade,
            tick_traded_value_ref=tick_traded_value_ref,
            force_hold=force_hold,
            bypass_paced_cap=False,
        )

    return fills


def _execute_single(
    order: dict,
    portfolio: "Portfolio",
    prices: dict[str, float],
    tick_index: int,
    logger: "SimLogger",
    fills: list[dict],
    max_tick_trade: float | None,
    tick_traded_value_ref: list[float],
    force_hold: set[str],
    bypass_paced_cap: bool = False,
) -> None:
    """Execute one order, applying all checks. Mutates fills and tick_traded_value_ref."""
    ticker = order["ticker"]
    side   = order["side"].upper()
    qty    = int(order.get("qty", 0))

    if qty <= 0:
        return

    # TC-specific: block sells on protected tickers
    if side == "SELL" and ticker in force_hold:
        vprint(f"[Exec] tick={tick_index} BLOCK SELL {ticker} — TC001 post-split protection")
        return

    price = prices.get(ticker)
    if price is None or price <= 0:
        vprint(f"[Exec] tick={tick_index} SKIP {side} {ticker} — no price")
        return

    tick_traded_value = tick_traded_value_ref[0]

    # Deployment/paced cap check (skipped for forced TC orders)
    if not bypass_paced_cap and max_tick_trade is not None:
        trade_value = qty * price
        if tick_traded_value + trade_value > max_tick_trade:
            remaining_budget = max_tick_trade - tick_traded_value
            qty = int(remaining_budget / price)
            if qty <= 0:
                vprint(f"[Exec] tick={tick_index} SKIP {side} {ticker} — "
                      f"deploy cap reached")
                return
            trade_value = qty * price

    trade_value = qty * price

    # Hard turnover check (absolute 30% cap — always enforced, even for forced orders)
    ok, reason = check_turnover(
        portfolio.turnover_ratio,
        trade_value,
        portfolio.avg_portfolio_value,
    )
    if not ok:
        vprint(f"[Exec] tick={tick_index} HALT — {reason}")
        return

    # Cardinality check
    ok, reason = check_cardinality(portfolio.holdings, ticker, side)
    if not ok:
        vprint(f"[Exec] tick={tick_index} SKIP {side} {ticker} — {reason}")
        return

    # Long-only check
    current_qty = portfolio.holdings.get(ticker, {}).get("qty", 0)
    ok, reason = check_long_only(qty, side, current_qty)
    if not ok:
        vprint(f"[Exec] tick={tick_index} SKIP {side} {ticker} — {reason}")
        return

    # For SELL: don't sell more than we hold
    if side == "SELL":
        qty = min(qty, current_qty)
        if qty <= 0:
            return
        trade_value = qty * price

    # Affordability check for buys
    if side == "BUY":
        affordable = max_affordable_qty(portfolio.cash, price, max_qty=qty)
        if affordable <= 0:
            vprint(f"[Exec] tick={tick_index} SKIP BUY {ticker} — insufficient cash")
            return
        qty = affordable
        trade_value = qty * price

    # Compute fees and execute
    fees       = compute_fees(qty, price)
    exec_price = price

    portfolio.apply_fill(ticker, side, qty, exec_price, fees)
    tick_traded_value_ref[0] += trade_value

    fill = {
        "type":          "execution",
        "order_ref":     f"ord_{ticker}_{side}_{qty}",
        "ticker":        ticker,
        "side":          side,
        "qty":           qty,
        "exec_price":    round(exec_price, 4),
        "fees":          round(fees, 4),
        "ts":            _now_iso(),
        # Additional fields for our own analysis
        "tick_index":    tick_index,
        "trade_value":   round(trade_value, 2),
        "turnover_after": round(portfolio.turnover_ratio, 6),
    }
    fills.append(fill)
    logger.log_fill(fill)
    vprint(f"[Exec] FILL tick={tick_index} {side} {qty}x{ticker} "
          f"@ ${exec_price:.2f} fee=${fees:.2f} "
          f"turnover={portfolio.turnover_ratio:.2%}"
          f"{' [TC-FORCED]' if bypass_paced_cap else ''}")


def target_weights_to_orders(
    target_weights: dict[str, float],
    portfolio: "Portfolio",
    prices: dict[str, float],
    tickers: list[str],
    min_trade_fraction: float = 0.010,  # skip trades < 1% of portfolio (Fix E: de-dust)
    last_buy_tick: dict | None = None,  # Fix B: {ticker: tick of last buy}
    min_hold_ticks: int = 15,           # Fix B: min ticks to hold before selling
    current_tick: int = 0,              # Fix B: current tick index
) -> list[dict]:
    """
    Convert target weights to buy/sell orders.
    Only generates orders where position change is meaningful.

    Fix B — anti-churn: blocks SELL orders within min_hold_ticks of the last
    buy for that ticker. Prevents EWMA oscillation from causing buy-sell-buy
    round-trips that waste turnover budget.

    Fix E — de-dust: skips trades below min_trade_fraction × NAV (default 1%).
    """
    total_value = portfolio.market_value(prices)
    orders: list[dict] = []
    _last_buy = last_buy_tick or {}

    for ticker in tickers:
        target_w = target_weights.get(ticker, 0.0)
        price    = prices.get(ticker, 0.0)
        if price <= 0:
            continue

        target_value  = target_w * total_value
        current_qty   = portfolio.holdings.get(ticker, {}).get("qty", 0)
        current_value = current_qty * price

        delta_value = target_value - current_value

        # Fix E: skip tiny trades (dust prevention)
        if abs(delta_value) < min_trade_fraction * total_value:
            continue

        qty = int(abs(delta_value) / price)
        if qty == 0:
            continue

        side = "BUY" if delta_value > 0 else "SELL"

        # Fix B: anti-churn — block SELL within min_hold_ticks of last buy
        if side == "SELL" and ticker in _last_buy:
            ticks_held = current_tick - _last_buy[ticker]
            if ticks_held < min_hold_ticks:
                vprint(f"[AntiChurn] SKIP SELL {ticker} — held only {ticks_held} ticks "
                       f"(min {min_hold_ticks})")
                continue

        orders.append({"ticker": ticker, "side": side, "qty": qty})

    return orders
