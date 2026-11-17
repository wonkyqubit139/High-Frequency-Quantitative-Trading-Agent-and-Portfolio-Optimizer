"""
data_loader.py — Load and validate all JSON inputs.

Handles both old and new JSON schemas robustly.
Falls back gracefully on any field name variation.

New schemas (from hackathon starter kit):
  - market_feed: tickers as an array of objects {ticker, price, volume, ...}
  - fundamentals: list of objects [{ticker, sector, beta, ...}]
  - corporate_actions: some have tick=null (revealed at live event)
  - initial_portfolio: holdings is an array [{ticker, qty, avg_price}]
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.log import vprint

# Default CA ticks matching validate_solution.py DEFAULT_CA_TICKS
DEFAULT_CA_TICKS: dict[str, int] = {
    "CA004": 0,    # D002 stock split
    "CA008": 22,   # C002 earnings surprise (in_sample confirmed)
    "CA002": 45,   # B003 dividend declaration (in_sample confirmed)
    "CA009": 67,   # D007 dividend declaration (in_sample confirmed)
    "CA001": 90,   # A001 earnings surprise
    "CA003": 150,  # C005 management change (in_sample confirmed)
    "CA005": 200,  # E007 M&A rumour
    "CA010": 240,  # E004 management change (in_sample confirmed)
    "CA006": 280,  # B008 regulatory fine
    "CA011": 325,  # A009 regulatory fine (in_sample confirmed)
    "CA007": 370,  # A005/B001 index rebalance
}


def _load_json(path: str | Path) -> Any:
    """Load JSON file with clear error messages."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"[DataLoader] File not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"[DataLoader] Invalid JSON in {path}: {e}")


def load_portfolio(data_dir: str = "./data") -> dict:
    """
    Load initial_portfolio.json.

    Supports both formats:
      Old: {"cash": 10000000, "holdings": {}}
      New: {"portfolio_id": "...", "cash": ..., "holdings": [{ticker, qty, avg_price}],
            "seed": 42, "session_start_tick": 0, "currency": "USD"}

    Returns:
      {"cash": float, "holdings": {ticker: {"qty": int, "avg_price": float}},
       "portfolio_id": str, "seed": int}
    """
    path = Path(data_dir) / "initial_portfolio.json"
    raw = _load_json(path)

    cash = float(raw.get("cash", 10_000_000))
    raw_holdings = raw.get("holdings", [])

    holdings: dict[str, dict] = {}
    if isinstance(raw_holdings, list):
        # New format: [{ticker, qty, avg_price}, ...]
        for h in raw_holdings:
            if isinstance(h, dict) and "ticker" in h:
                ticker = str(h["ticker"])
                holdings[ticker] = {
                    "qty": int(h.get("qty", 0)),
                    "avg_price": float(h.get("avg_price", 0.0)),
                }
    elif isinstance(raw_holdings, dict):
        # Old format: {ticker: qty} or {ticker: {qty, avg_price}}
        for ticker, val in raw_holdings.items():
            if isinstance(val, dict):
                holdings[ticker] = {
                    "qty": int(val.get("qty", 0)),
                    "avg_price": float(val.get("avg_price", 0.0)),
                }
            else:
                holdings[ticker] = {
                    "qty": int(val),
                    "avg_price": 0.0,
                }

    result = {
        "cash": cash,
        "holdings": holdings,
        "portfolio_id": raw.get("portfolio_id", "unknown"),
        "seed": int(raw.get("seed", 42)),
    }
    vprint(f"[DataLoader] Portfolio: {result['portfolio_id']} | "
          f"cash=${cash:,.2f} | {len(holdings)} initial holdings")
    return result


def load_corporate_actions(data_dir: str = "./data") -> tuple[dict[int, list[dict]], list[dict]]:
    """
    Load corporate_actions.json.

    Handles:
      - tick=null (uses DEFAULT_CA_TICKS fallback for known CA IDs)
      - split_ratio vs ratio (unified to "ratio")
      - dividend_per_share vs amount (unified to "amount")
      - INDEX_REBALANCE with comma-separated tickers

    Returns:
      (by_tick, null_tick_cas)
      by_tick: {tick_index: [ca_dict, ...]}  — CAs with known ticks
      null_tick_cas: [ca_dict, ...]           — CAs whose ticks will be revealed live
                                                (already assigned DEFAULT_CA_TICKS if available)
    """
    path = Path(data_dir) / "corporate_actions.json"
    raw_list = _load_json(path)
    if not isinstance(raw_list, list):
        raw_list = raw_list.get("actions", [])

    by_tick: dict[int, list[dict]] = {}
    null_tick_cas: list[dict] = []

    for raw in raw_list:
        ca_id   = str(raw.get("id", ""))
        ca_type = str(raw.get("type", "")).upper()
        ticker  = str(raw.get("ticker", ""))

        # Normalize field names
        # split_ratio → ratio
        ratio = raw.get("ratio") or raw.get("split_ratio") or 1
        # dividend_per_share → amount
        amount = raw.get("amount") or raw.get("dividend_amount") or raw.get("div_amount") or raw.get("dividend_per_share") or 0.0
        # other fields
        surprise_pct  = raw.get("surprise_pct") or raw.get("eps_surprise") or 0.0
        price_impact  = raw.get("price_impact") or 0.0
        vol_multiplier = raw.get("vol_multiplier") or 1.0
        in_sample     = bool(raw.get("in_sample", True))
        description   = str(raw.get("description", ""))
        expected_behaviour = str(raw.get("expected_behaviour", ""))

        # Handle INDEX_REBALANCE multi-ticker (comma-separated)
        tickers_list = [t.strip() for t in ticker.split(",") if t.strip()]

        ca = {
            "id":               ca_id,
            "type":             ca_type,
            "ticker":           ticker,          # original (may be "A005,B001")
            "tickers_list":     tickers_list,    # always a list
            "ratio":            int(ratio),
            "amount":           float(amount),
            "surprise_pct":     float(surprise_pct),
            "price_impact":     float(price_impact),
            "vol_multiplier":   float(vol_multiplier),
            "in_sample":        in_sample,
            "description":      description,
            "expected_behaviour": expected_behaviour,
        }

        raw_tick = raw.get("tick")
        if raw_tick is not None:
            tick = int(raw_tick)
            ca["tick"] = tick
            by_tick.setdefault(tick, []).append(ca)
            vprint(f"[DataLoader] CA {ca_id}: {ca_type} {ticker} @ tick {tick} "
                  f"{'(in_sample)' if in_sample else ''}")
        else:
            # Null-tick CA — use DEFAULT_CA_TICKS if we know the default
            default_tick = DEFAULT_CA_TICKS.get(ca_id)
            if default_tick is not None:
                ca["tick"] = default_tick
                ca["tick_is_default"] = True
                by_tick.setdefault(default_tick, []).append(ca)
                vprint(f"[DataLoader] CA {ca_id}: {ca_type} {ticker} @ DEFAULT tick {default_tick} "
                      f"(revealed at live event)")
            else:
                ca["tick"] = None
                ca["tick_is_default"] = False
                null_tick_cas.append(ca)
                vprint(f"[DataLoader] CA {ca_id}: {ca_type} {ticker} — tick UNKNOWN "
                      f"(no default, will be revealed live)")

    vprint(f"[DataLoader] CAs with ticks: {sum(len(v) for v in by_tick.values())} | "
          f"fully unknown: {len(null_tick_cas)}")
    return by_tick, null_tick_cas


def load_market_feed(data_dir: str = "./data") -> list[dict]:
    """
    Load market feed JSON.

    Tries market_feed_full.json first, then market_feed_sample.json.

    Supports both schemas:
      Old: [{tick_index, prices: {ticker: float}, volumes: {ticker: int}}]
      New: [{ts, tick_index, tickers: [{ticker, price, volume, asset_type, sector}]}]

    Always returns internal format:
      [{tick_index: int, prices: {ticker: float}, volumes: {ticker: int}, ts: str}]
    """
    for filename in ("market_feed_full.json", "market_feed_sample.json"):
        path = Path(data_dir) / filename
        if path.exists():
            vprint(f"[DataLoader] Loading market feed from: {filename}")
            break
    else:
        raise FileNotFoundError(f"[DataLoader] No market feed file found in {data_dir}")

    raw_list = _load_json(path)
    if not isinstance(raw_list, list):
        raw_list = raw_list.get("ticks", [])

    ticks = []
    for raw in raw_list:
        tick_index = int(raw.get("tick_index", raw.get("tick", 0)))
        ts = str(raw.get("ts", datetime.now(timezone.utc).isoformat()))

        # Detect format
        if "tickers" in raw:
            # New format: tickers array
            prices  = {}
            volumes = {}
            for asset in raw["tickers"]:
                t = str(asset["ticker"])
                prices[t]  = float(asset["price"])
                volumes[t] = int(asset.get("volume", 0))
        else:
            # Old format: flat prices/volumes dicts
            raw_prices  = raw.get("prices", raw.get("price", raw.get("close", {})))
            raw_volumes = raw.get("volumes", raw.get("volume", raw.get("vol", {})))
            prices  = {k: float(v) for k, v in raw_prices.items()}
            volumes = {k: int(v)   for k, v in raw_volumes.items()}

        ticks.append({
            "tick_index": tick_index,
            "prices":     prices,
            "volumes":    volumes,
            "ts":         ts,
        })

    ticks.sort(key=lambda x: x["tick_index"])

    if ticks:
        n_tickers = len(ticks[0]["prices"])
        vprint(f"[DataLoader] Market feed: {len(ticks)} ticks × {n_tickers} tickers")
        a001 = ticks[0]["prices"].get("A001")
        if a001:
            vprint(f"[DataLoader] A001 @ tick 0: ${a001:.2f}")
    return ticks


def load_fundamentals(data_dir: str = "./data") -> dict[str, dict]:
    """
    Load fundamentals.json.

    Supports both formats:
      Old: {ticker: {sector, beta, pe_ratio, eps, esg_score, div_yield}}
      New: [{ticker, sector, pe_ratio, pb_ratio, eps, eps_growth_yoy, dividend_yield,
             debt_to_equity, roe, beta, esg_score, analyst_rating, high_52w, low_52w,
             market_cap_bn}]

    Returns {ticker: {all fields with canonical names}}.
    """
    path = Path(data_dir) / "fundamentals.json"
    raw = _load_json(path)

    result: dict[str, dict] = {}

    if isinstance(raw, list):
        # New format: list of objects
        for item in raw:
            if not isinstance(item, dict) or "ticker" not in item:
                continue
            ticker = str(item["ticker"])
            result[ticker] = {
                "sector":          str(item.get("sector", "Unknown")),
                "beta":            float(item.get("beta", 1.0)),
                "pe_ratio":        float(item.get("pe_ratio", 20.0)),
                "pb_ratio":        float(item.get("pb_ratio", 2.0)),
                "eps":             float(item.get("eps", 10.0)),
                "eps_growth_yoy":  float(item.get("eps_growth_yoy", 0.0)),
                "esg_score":       float(item.get("esg_score", 50.0)),
                "div_yield":       float(item.get("dividend_yield", item.get("div_yield", 0.0))),
                "debt_to_equity":  float(item.get("debt_to_equity", 0.5)),
                "roe":             float(item.get("roe", 0.1)),
                "analyst_rating":  str(item.get("analyst_rating", "HOLD")),
                "high_52w":        float(item.get("high_52w", 0.0)),
                "low_52w":         float(item.get("low_52w", 0.0)),
                "market_cap_bn":   float(item.get("market_cap_bn", 1.0)),
            }
    elif isinstance(raw, dict):
        # Old format: dict keyed by ticker
        for ticker, item in raw.items():
            if not isinstance(item, dict):
                continue
            result[ticker] = {
                "sector":          str(item.get("sector", "Unknown")),
                "beta":            float(item.get("beta", 1.0)),
                "pe_ratio":        float(item.get("pe_ratio", item.get("pe", 20.0))),
                "pb_ratio":        float(item.get("pb_ratio", 2.0)),
                "eps":             float(item.get("eps", 10.0)),
                "eps_growth_yoy":  float(item.get("eps_growth_yoy", 0.0)),
                "esg_score":       float(item.get("esg_score", item.get("esg", 50.0))),
                "div_yield":       float(item.get("div_yield", item.get("dividend_yield", 0.0))),
                "debt_to_equity":  float(item.get("debt_to_equity", 0.5)),
                "roe":             float(item.get("roe", 0.1)),
                "analyst_rating":  str(item.get("analyst_rating", "HOLD")),
                "high_52w":        float(item.get("high_52w", 0.0)),
                "low_52w":         float(item.get("low_52w", 0.0)),
                "market_cap_bn":   float(item.get("market_cap_bn", 1.0)),
            }

    vprint(f"[DataLoader] Fundamentals: {len(result)} tickers")
    return result


def load_all(data_dir: str = "./data") -> tuple[dict, dict, list[dict], dict[str, dict], list[dict]]:
    """
    Convenience: load all four data sources.

    Returns:
      (portfolio, ca_by_tick, market_feed, fundamentals, null_tick_cas)
    """
    portfolio    = load_portfolio(data_dir)
    ca_by_tick, null_tick_cas = load_corporate_actions(data_dir)
    market_feed  = load_market_feed(data_dir)
    fundamentals = load_fundamentals(data_dir)
    return portfolio, ca_by_tick, market_feed, fundamentals, null_tick_cas
