"""
llm_client.py — LLM client for reactive corporate action evaluation.

Strategy: Use at most 5 LLM calls per run. Each call is triggered
REACTIVELY when a corporate action fires (same tick). The LLM evaluates
the event that just happened and provides expected return estimates for
the affected ticker and its sector peers.

We prioritise high-impact CA types (sorted by |price_impact|):
  1. MA_RUMOUR             (+14.2%)
  2. REGULATORY_FINE       (-9.1%, -5.5%)
  3. EARNINGS_SURPRISE     (+8.5%, +6.2%)
  4. MANAGEMENT_CHANGE     (-6.3%, -4.7%)
  5. INDEX_REBALANCE       (+1.8%, +1.5%)

Lower-impact events (STOCK_SPLIT, DIVIDEND_DECLARATION) are NOT worth
an LLM call — they are handled mechanically.

Endpoint:
  POST {LLM_HOST}/llm/query
  Headers: Authorization: Bearer {LLM_TOKEN}
  Body: {"prompt": str, "context": dict, "deterministic_seed": int}
  Response: {"response_id": str, "text": str}
"""
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.log import vprint

# ── Configuration ─────────────────────────────────────────────────────────
LLM_QUOTA       = 5      # max calls per run (total budget = 60 across all runs)
TIMEOUT_SEC     = 30     # per-request timeout (onrender can be slow on cold start)
LLM_ALPHA       = 0.30   # blend weight of LLM signal onto quant signal

# CA types worth an LLM call (in priority order)
# STOCK_SPLIT and DIVIDEND_DECLARATION are mechanical — no LLM needed.
LLM_WORTHY_CA_TYPES = {
    "MA_RUMOUR",
    "REGULATORY_FINE",
    "EARNINGS_SURPRISE",
    "MANAGEMENT_CHANGE",
    "INDEX_REBALANCE",
}

# Module-level call counter
_call_count = 0


def reset_call_count() -> None:
    global _call_count
    _call_count = 0


def remaining_calls() -> int:
    return max(0, LLM_QUOTA - _call_count)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Reactive Trigger ──────────────────────────────────────────────────────

def should_call_for_ca(processed_events: list[dict]) -> Optional[dict]:
    """
    Given the list of CA events processed THIS tick, decide if we should
    call the LLM. Returns the highest-priority event dict, or None.

    Only triggers for LLM_WORTHY_CA_TYPES and only if quota remains.
    """
    if _call_count >= LLM_QUOTA:
        return None

    # Filter to LLM-worthy events
    worthy = [
        e for e in processed_events
        if e.get("normalized_type", e.get("type", "")).upper() in LLM_WORTHY_CA_TYPES
    ]
    if not worthy:
        return None

    # Pick highest impact (by absolute price_impact_pct)
    return max(worthy, key=lambda e: abs(float(e.get("price_impact_pct", 0) or 0)))


# ── Prompt Builder (reactive CA-focused) ──────────────────────────────────

def _build_ca_prompt(
    tick_index: int,
    ca_event: dict,
    affected_ticker_prices: dict[str, list[float]],
    sector_peer_prices: dict[str, list[float]],
    fundamentals: dict[str, dict],
    portfolio_state: Optional[dict] = None,
    turnover_remaining: float = 0.10,
) -> str:
    """Build a focused prompt for evaluating a CA that just fired."""

    ca_type = ca_event.get("normalized_type", ca_event.get("type", "UNKNOWN"))
    ca_id = ca_event.get("id", "?")
    ticker = ca_event.get("ticker", "?")
    # Handle multi-ticker CAs (INDEX_REBALANCE: "A005,B001")
    if "," in ticker:
        affected = [t.strip() for t in ticker.split(",")]
    else:
        affected = ca_event.get("tickers_list", [ticker])

    description = ca_event.get("description", "")
    expected = ca_event.get("expected_behaviour", "")
    impact = ca_event.get("price_impact_pct", "?")
    vol_mult = ca_event.get("vol_multiplier", "?")

    # Affected ticker price history
    affected_lines = []
    for t in affected:
        hist = affected_ticker_prices.get(t, [])
        if len(hist) >= 2:
            pct_10 = (hist[-1] - hist[0]) / hist[0] * 100 if hist[0] else 0
            last_price = hist[-1]
            affected_lines.append(f"  {t}: ${last_price:.2f} (10-tick change: {pct_10:+.2f}%)")
        elif hist:
            affected_lines.append(f"  {t}: ${hist[-1]:.2f}")

    # Sector peer prices
    peer_lines = []
    for t, hist in sector_peer_prices.items():
        if t in affected:
            continue
        if len(hist) >= 2 and hist[0]:
            pct = (hist[-1] - hist[0]) / hist[0] * 100
            peer_lines.append(f"  {t}: ${hist[-1]:.2f} ({pct:+.2f}%)")
    peer_lines = peer_lines[:8]

    # Fundamentals for affected tickers
    fund_lines = []
    for t in affected:
        f = fundamentals.get(t, {})
        fund_lines.append(
            f"  {t}: sector={f.get('sector','?')} beta={f.get('beta','?')} "
            f"P/E={f.get('pe_ratio','?')} EPS={f.get('eps','?')} "
            f"ESG={f.get('esg_score','?')} analyst={f.get('analyst_rating','?')}"
        )

    # Portfolio context
    portfolio_block = "Starting from cash (no positions yet)"
    if portfolio_state:
        cash  = portfolio_state.get("cash", 0)
        total = portfolio_state.get("total_value", cash) or cash
        holdings = portfolio_state.get("holdings", [])
        n_hold = len(holdings)

        held_info = []
        for h in holdings:
            if h["ticker"] in affected:
                w = h["qty"] * h["avg_price"] / total if total else 0
                held_info.append(f"{h['ticker']}: {h['qty']} shares, weight={w:.2%}")

        portfolio_block = (
            f"Cash: ${cash:,.0f} ({cash/total:.1%})\n"
            f"Holdings: {n_hold}/30\n"
            f"Turnover remaining: {turnover_remaining:.1%}\n"
            f"Affected ticker positions: {'; '.join(held_info) if held_info else 'NOT HELD'}"
        )

    # Build the list of tickers to evaluate
    eval_tickers = list(affected)
    for t in sector_peer_prices:
        if t not in eval_tickers:
            eval_tickers.append(t)
        if len(eval_tickers) >= 10:
            break

    prompt = f"""/no_think
You are a quantitative portfolio manager in an intraday trading simulation (390 ticks, 1 tick = 1 minute).

A corporate action just fired at tick {tick_index}:

=== CORPORATE ACTION EVENT ===
ID: {ca_id}
Type: {ca_type}
Ticker(s): {', '.join(affected)}
Price impact: {impact}%
Volatility multiplier: {vol_mult}x
Description: {description}
Expected behaviour: {expected}

=== AFFECTED TICKER RECENT PRICES (last 10 ticks) ===
{chr(10).join(affected_lines) if affected_lines else 'No price data'}

=== SECTOR PEERS ===
{chr(10).join(peer_lines) if peer_lines else 'No peer data'}

=== FUNDAMENTALS ===
{chr(10).join(fund_lines)}

=== PORTFOLIO ===
{portfolio_block}

=== CONSTRAINTS ===
- Max 30 holdings, max 15% weight per position
- Max 30% total session turnover ({turnover_remaining:.1%} remaining)
- Long-only (no shorting)

=== TASK ===
Given this corporate action that JUST happened, estimate the expected log return over the NEXT 20 ticks for each ticker below. Consider:
1. The direct price impact on affected ticker(s)
2. Momentum/mean-reversion after the event
3. Sector contagion or spillover effects
4. Whether to increase, hold, or reduce exposure

RESPOND with ONLY a valid JSON object (no markdown, no explanation):
{{"expected_returns": {{"TICKER": float, ...}}}}

Positive = expected to go up, negative = expected to go down. Use 0.0 for no view.

Tickers to evaluate: {', '.join(eval_tickers)}"""

    return prompt


# ── Response Parser ───────────────────────────────────────────────────────

def _parse_mu(content: str, tickers: list[str]) -> dict[str, float]:
    """
    Extract {ticker: float} from LLM response.
    Handles markdown code fences, <think> tags (qwen3), and multiple JSON schemas.
    """
    if not content:
        return {}

    text = content.strip()

    # Strip <think>...</think> tags (qwen3 reasoning model)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()

    # Strip markdown code fences
    if "```" in text:
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

    # Find first complete JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= 0:
        return {}

    try:
        raw = json.loads(text[start:end])
    except json.JSONDecodeError:
        # Fix common issues: trailing commas
        cleaned = text[start:end]
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

    # Support both {"expected_returns": {...}} and flat {ticker: float}
    data = raw.get("expected_returns", raw)
    if not isinstance(data, dict):
        return {}

    result: dict[str, float] = {}
    ticker_set = set(tickers)
    for k, v in data.items():
        if k in ticker_set:
            try:
                result[k] = float(v)
            except (TypeError, ValueError):
                pass
    return result


# ── Main LLM Call Function ────────────────────────────────────────────────

def call_llm(
    tick_index: int,
    call_number: int,
    tickers: list[str],
    ca_event: dict,
    affected_ticker_prices: dict[str, list[float]],
    sector_peer_prices: dict[str, list[float]],
    fundamentals: dict[str, dict],
    logger,
    portfolio_state: Optional[dict] = None,
    turnover_remaining: float = 0.10,
    llm_host: Optional[str] = None,
    llm_token: Optional[str] = None,
) -> Optional[dict[str, float]]:
    """
    Call the LLM to evaluate a corporate action that just fired.
    Returns {ticker: expected_return} or None on failure.
    """
    global _call_count

    if _call_count >= LLM_QUOTA:
        vprint(f"[LLM] tick={tick_index} QUOTA EXHAUSTED ({LLM_QUOTA} calls used)")
        return None

    prompt = _build_ca_prompt(
        tick_index=tick_index,
        ca_event=ca_event,
        affected_ticker_prices=affected_ticker_prices,
        sector_peer_prices=sector_peer_prices,
        fundamentals=fundamentals,
        portfolio_state=portfolio_state,
        turnover_remaining=turnover_remaining,
    )

    context = {
        "tick": tick_index,
        "ca_id": ca_event.get("id", ""),
        "ca_type": ca_event.get("normalized_type", ca_event.get("type", "")),
        "affected_tickers": ca_event.get("tickers_list",
                                          [ca_event.get("ticker", "")]),
    }

    record: dict = {
        "call_number":        call_number,
        "tick_index":         tick_index,
        "ca_id":              ca_event.get("id", ""),
        "prompt":             prompt,
        "response":           None,
        "parsed_mu":          None,
        "error":              None,
        "deterministic_seed": 42,
        "ts":                 _now_iso(),
    }

    # Determine endpoint
    host = llm_host or os.environ.get("LLM_HOST", "")
    token = llm_token or os.environ.get("LLM_TOKEN", "")

    if not host or not token:
        vprint(f"[LLM] tick={tick_index} no LLM host/token — pure quant mode")
        return None

    # Build endpoint URL — handle full URL or host:port
    if host.startswith("http://") or host.startswith("https://"):
        endpoint = f"{host.rstrip('/')}/llm/query"
    else:
        endpoint = f"http://{host}/llm/query"

    try:
        with httpx.Client(timeout=TIMEOUT_SEC) as client:
            resp = client.post(
                endpoint,
                json={
                    "prompt": prompt,
                    "context": context,
                    "deterministic_seed": 42,
                },
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()

        raw_text = data.get("text", "")
        record["response"] = raw_text
        mu = _parse_mu(raw_text, tickers)
        record["parsed_mu"] = mu
        _call_count += 1
        record["call_count_after"] = _call_count

        vprint(f"[LLM] tick={tick_index} call#{call_number} CA={ca_event.get('id','?')} — "
               f"{len(mu)} estimates, {remaining_calls()} calls left")
        logger.log_llm_call(record)
        return mu if mu else None

    except Exception as e:
        vprint(f"[LLM] tick={tick_index} ERROR: {e}")
        record["error"] = str(e)
        logger.log_llm_call(record)
        return None


# ── Legacy compatibility ──────────────────────────────────────────────────

def should_call_llm(
    tick_index: int,
    active_ca_ticks: set[int],
    volume_spike_tickers: list[str],
    call_count: int = 0,
    every_n_ticks: int = 15,
) -> bool:
    """Legacy — replaced by should_call_for_ca()."""
    return False
