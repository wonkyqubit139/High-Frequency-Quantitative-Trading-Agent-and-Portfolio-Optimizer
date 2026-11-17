"""
corporate_actions.py — Handle all corporate action types.

Supported CA types (new names from hackathon data):
  STOCK_SPLIT          — adjust price history and holdings (TC001)
  EARNINGS_SURPRISE    — strong momentum buy signal (decaying)
  DIVIDEND_DECLARATION — buy before ex-date for income (decaying)
  MANAGEMENT_CHANGE    — governance risk, reduce exposure (decaying)
  MA_RUMOUR            — M&A premium, momentum buy with 5% cap (decaying)
  REGULATORY_FINE      — ESG violation, exit signal (decaying)
  INDEX_REBALANCE      — demand surge, pre-position both tickers (decaying)

Legacy type aliases also supported:
  EARNINGS_BEAT → EARNINGS_SURPRISE
  CEO_RESIGNATION → MANAGEMENT_CHANGE
  DIVIDEND → DIVIDEND_DECLARATION

CA signal boosts decay exponentially: boost *= 0.90 ^ (ticks_since_event)
After ~7 ticks the boost is halved; after ~20 ticks it is ~12% of original.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from src.log import vprint

if TYPE_CHECKING:
    from src.portfolio import Portfolio


# CA_FLAGS: {ticker: (flag_type, tick_fired)}
CA_FLAGS: dict[str, tuple[str, int]] = {}

# Per-tick exponential decay factor for CA signal boosts
CA_DECAY = 0.90  # boost halves every ~7 ticks

# Type-name normalization (old → new canonical name)
CA_TYPE_ALIASES: dict[str, str] = {
    "EARNINGS_BEAT":   "EARNINGS_SURPRISE",
    "CEO_RESIGNATION": "MANAGEMENT_CHANGE",
    "DIVIDEND":        "DIVIDEND_DECLARATION",
}

# Base signal boosts per CA type (before decay)
CA_BASE_BOOSTS: dict[str, float] = {
    "EARNINGS_SURPRISE":    +0.05,
    "DIVIDEND_DECLARATION": +0.02,
    "MA_RUMOUR":            +0.04,
    "INDEX_REBALANCE":      +0.03,
    "MANAGEMENT_CHANGE":    -0.04,
    "REGULATORY_FINE":      -0.06,
}


def _normalize_ca_type(ca_type: str) -> str:
    """Map legacy type names to canonical names."""
    upper = ca_type.upper()
    return CA_TYPE_ALIASES.get(upper, upper)


def process_corporate_actions(
    tick_index: int,
    ca_by_tick: dict[int, list[dict]],
    portfolio: "Portfolio",
    price_histories: dict[str, list[float]],
    current_prices: dict[str, float],
) -> list[dict]:
    """
    Process all CAs scheduled for tick_index.
    Returns list of processed event records.
    """
    events = ca_by_tick.get(tick_index, [])
    processed = []

    for ca in events:
        raw_type = ca.get("type", "")
        ca_type  = _normalize_ca_type(raw_type)
        ca_id    = ca.get("id", "?")
        ticker   = ca.get("ticker", "")
        tickers_list = ca.get("tickers_list", [ticker] if ticker else [])

        vprint(f"[CA] tick={tick_index} {ca_id}: {ca_type} {ticker}")

        if ca_type == "STOCK_SPLIT":
            _handle_split(ca, tickers_list[0] if tickers_list else ticker,
                          portfolio, price_histories, current_prices)

        elif ca_type == "DIVIDEND_DECLARATION":
            for t in tickers_list:
                _handle_dividend(ca, t, portfolio)

        elif ca_type == "INDEX_REBALANCE":
            # Multi-ticker: set buy flag on all tickers in the rebalance
            for t in tickers_list:
                CA_FLAGS[t] = ("INDEX_REBALANCE", tick_index)
                vprint(f"[CA] {t} added to index — buy signal (decaying)")

        else:
            # All other types: set CA flag for signal boost/reduction
            for t in tickers_list:
                CA_FLAGS[t] = (ca_type, tick_index)
                vprint(f"[CA] {t} {ca_type} — signal flag set at tick {tick_index}")

        processed.append({**ca, "processed_at_tick": tick_index, "normalized_type": ca_type})

    return processed


def _handle_split(
    ca: dict,
    ticker: str,
    portfolio: "Portfolio",
    price_histories: dict[str, list[float]],
    current_prices: dict[str, float],
) -> None:
    """
    Stock split: rescale price history, update portfolio holdings.
    The market feed already provides post-split prices.
    CRITICAL: do NOT treat the price drop as a crash signal.
    """
    ratio = int(ca.get("ratio", 1))
    if ratio <= 1:
        return

    # Rescale historical prices so EWMA/momentum aren't corrupted
    if ticker in price_histories:
        old_len = len(price_histories[ticker])
        price_histories[ticker] = [p / ratio for p in price_histories[ticker]]
        vprint(f"[CA] STOCK_SPLIT {ticker} {ratio}:1 — "
              f"rescaled {old_len} historical prices")

    # Update portfolio (qty * ratio, avg_price / ratio)
    new_price = current_prices.get(ticker, 0.0)
    portfolio.apply_split(ticker, ratio, new_price)

def _handle_dividend(
    ca: dict,
    ticker: str,
    portfolio: "Portfolio",
) -> None:
    """Pay out dividend to current holders."""
    amount = float(ca.get("amount", 0.0))
    if amount > 0:
        portfolio.apply_dividend(ticker, amount)
        vprint(f"[CA] Dividend {ticker}: ${amount:.4f}/share paid")
    else:
        vprint(f"[CA] Dividend {ticker}: amount={amount} (skipped or unknown)")


def get_ca_signal_boost(ticker: str, current_tick: int,
                        ca_decay: float = CA_DECAY) -> float:
    """
    Return a decaying signal modifier for a ticker based on active CA flags.
    Positive = buy bias, negative = sell/exit bias.

    boost(t) = base_boost × ca_decay^(current_tick - tick_fired)
    """
    entry = CA_FLAGS.get(ticker)
    if entry is None:
        return 0.0

    flag_type, tick_fired = entry
    ticks_elapsed = max(0, current_tick - tick_fired)
    decay = ca_decay ** ticks_elapsed
    base  = CA_BASE_BOOSTS.get(flag_type, 0.0)
    return base * decay


def get_all_ca_boosts(tickers: list[str], current_tick: int,
                      ca_decay: float = CA_DECAY) -> dict[str, float]:
    """Return {ticker: boost} for all tickers."""
    return {t: get_ca_signal_boost(t, current_tick, ca_decay=ca_decay) for t in tickers}



