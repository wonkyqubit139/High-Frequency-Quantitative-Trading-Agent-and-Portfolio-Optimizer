"""
portfolio.py — Portfolio state: cash, holdings (with avg_price), weights, snapshots.

Holdings format (internal): {ticker: {"qty": int, "avg_price": float}}

Snapshot format (output — matches validate_solution.py expected schema):
  {
    "tick_index": int,
    "cash": float,
    "holdings": [{"ticker": str, "qty": int, "avg_price": float}, ...],
    "total_value": float,
    "ts": str,    # ISO timestamp
  }

The validator uses qty * avg_price / total_value for weight calculation (cost basis),
NOT mark-to-market. avg_price is the weighted average cost.
"""
from datetime import datetime, timezone
from typing import Optional

from src.log import vprint


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Portfolio:
    """
    Tracks cash, share holdings with cost basis, and turnover.
    All monetary values in USD.
    """

    def __init__(
        self,
        cash: float,
        holdings: Optional[dict[str, dict]] = None,
    ):
        self.cash: float = float(cash)

        # Internal: {ticker: {"qty": int, "avg_price": float}}
        self.holdings: dict[str, dict] = {}
        if holdings:
            for ticker, h in holdings.items():
                if isinstance(h, dict):
                    qty = int(h.get("qty", 0))
                    avg_price = float(h.get("avg_price", 0.0))
                else:
                    # Legacy: plain int quantity
                    qty = int(h)
                    avg_price = 0.0
                if qty > 0:
                    self.holdings[ticker] = {"qty": qty, "avg_price": avg_price}

        # Turnover tracking
        self.total_traded_value: float = 0.0
        self._portfolio_values: list[float] = []

    # ------------------------------------------------------------------
    # Value / weight helpers
    # ------------------------------------------------------------------
    def market_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value = cash + sum(shares * current_price)."""
        equity = sum(
            self.holdings[t]["qty"] * prices.get(t, self.holdings[t]["avg_price"])
            for t in self.holdings
        )
        return self.cash + equity

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Mark-to-market weights (for optimizer)."""
        total = self.market_value(prices)
        if total <= 0:
            return {}
        return {
            t: (self.holdings[t]["qty"] * prices.get(t, self.holdings[t]["avg_price"])) / total
            for t in self.holdings
            if self.holdings[t]["qty"] > 0
        }

    def weight_vector(self, tickers: list[str], prices: dict[str, float]) -> list[float]:
        """Ordered weight list for the MVO optimizer."""
        total = self.market_value(prices)
        if total <= 0:
            return [0.0] * len(tickers)
        return [
            (self.holdings.get(t, {"qty": 0})["qty"] * prices.get(t, 0.0)) / total
            for t in tickers
        ]

    def cost_basis_weight(self, ticker: str, total_value: float) -> float:
        """
        Cost basis weight = qty * avg_price / total_value.
        This is what the validator uses for TC006/TC007/TC009.
        """
        if total_value <= 0 or ticker not in self.holdings:
            return 0.0
        h = self.holdings[ticker]
        return h["qty"] * h["avg_price"] / total_value

    # ------------------------------------------------------------------
    # Turnover
    # ------------------------------------------------------------------
    def record_tick_value(self, value: float) -> None:
        self._portfolio_values.append(value)

    @property
    def avg_portfolio_value(self) -> float:
        if not self._portfolio_values:
            return self.cash
        return sum(self._portfolio_values) / len(self._portfolio_values)

    @property
    def turnover_ratio(self) -> float:
        avg = self.avg_portfolio_value
        if avg <= 0:
            return 0.0
        return self.total_traded_value / avg

    def add_traded_value(self, value: float) -> None:
        self.total_traded_value += abs(value)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def apply_fill(
        self,
        ticker: str,
        side: str,
        qty: int,
        exec_price: float,
        fees: float,
    ) -> None:
        """
        Apply a completed fill to cash and holdings.
        Tracks weighted average cost basis.
        """
        trade_value = qty * exec_price
        side = side.upper()

        if side == "BUY":
            self.cash -= trade_value + fees
            existing = self.holdings.get(ticker)
            if existing and existing["qty"] > 0:
                # Weighted average cost basis
                old_qty    = existing["qty"]
                old_avg    = existing["avg_price"]
                new_qty    = old_qty + qty
                new_avg    = (old_qty * old_avg + qty * exec_price) / new_qty
                self.holdings[ticker] = {"qty": new_qty, "avg_price": new_avg}
            else:
                self.holdings[ticker] = {"qty": qty, "avg_price": exec_price}

        elif side == "SELL":
            self.cash += trade_value - fees
            existing = self.holdings.get(ticker)
            if existing:
                new_qty = existing["qty"] - qty
                if new_qty <= 0:
                    del self.holdings[ticker]
                else:
                    # avg_price unchanged on SELL (cost basis stays)
                    self.holdings[ticker] = {"qty": new_qty, "avg_price": existing["avg_price"]}

        self.add_traded_value(trade_value)

    def apply_split(self, ticker: str, ratio: int, new_price: float) -> None:
        """
        Stock split: multiply holdings qty, divide avg_price.
        DO NOT treat as a price crash.
        """
        if ticker not in self.holdings or ratio <= 1:
            return
        h = self.holdings[ticker]
        old_qty = h["qty"]
        new_qty = old_qty * ratio
        new_avg = h["avg_price"] / ratio
        self.holdings[ticker] = {"qty": new_qty, "avg_price": new_avg}
        vprint(f"[Portfolio] Split {ticker} {ratio}:1 — qty {old_qty} → {new_qty}, "
              f"avg_price ${h['avg_price']:.4f} → ${new_avg:.4f}")

    def apply_dividend(self, ticker: str, amount_per_share: float) -> None:
        """Add dividend cash for held shares."""
        if ticker not in self.holdings or amount_per_share <= 0:
            return
        qty = self.holdings[ticker]["qty"]
        dividend_cash = qty * amount_per_share
        if dividend_cash > 0:
            self.cash += dividend_cash
            vprint(f"[Portfolio] Dividend {ticker}: {qty} × ${amount_per_share:.4f} "
                  f"= +${dividend_cash:,.2f}")

    def holding_count(self) -> int:
        return len([t for t in self.holdings if self.holdings[t]["qty"] > 0])

    # ------------------------------------------------------------------
    # Snapshot (matches validate_solution.py expected format)
    # ------------------------------------------------------------------
    def snapshot(self, tick_index: int, prices: dict[str, float]) -> dict:
        """
        Return a serialisable dict matching the validator's expected format.

        holdings: [{ticker, qty, avg_price}]  — used by TC004 (len check) and
        TC006/TC007/TC009 (weight via qty*avg_price/total_value).
        """
        total = self.market_value(prices)
        holdings_list = [
            {
                "ticker":    t,
                "qty":       h["qty"],
                "avg_price": round(h["avg_price"], 4),
            }
            for t, h in self.holdings.items()
            if h["qty"] > 0
        ]
        return {
            "tick_index":  tick_index,
            "cash":        round(self.cash, 2),
            "holdings":    holdings_list,
            "total_value": round(total, 2),
            "ts":          _now_iso(),
        }
