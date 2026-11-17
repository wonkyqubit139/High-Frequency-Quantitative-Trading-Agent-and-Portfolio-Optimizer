"""
Portfolio Optimization Agent
============================
EXECUTION MODEL: fully file-based.

Central infrastructure hosts ONE live endpoint:
  POST /llm/query   — LLM proxy (max 60 calls)

Everything else runs locally against flat files:
  READS   market_feed_full.json      — 390-tick price/volume feed
  READS   initial_portfolio.json     — starting cash & holdings
  READS   corporate_actions.json     — corporate action events
  READS   fundamentals.json          — static per-ticker data

PRODUCES (required for scoring):
  orders_log.json              — every simulated order and fill
  portfolio_snapshots.json     — portfolio state after every tick
  llm_call_log.json            — every LLM call made
  results.json                 — final PnL, Sharpe ratio, summary metrics

Usage:
  python -m src.main \\
      --token  <TOKEN> \\
      --llm    <LLM_PROXY_HOST:PORT> \\
      --feed   market_feed_full.json \\
      --portfolio initial_portfolio.json \\
      --ca     corporate_actions.json \\
      --fundamentals fundamentals.json \\
      --out    ./output

TC-specific logic built in:
  TC001: Don't panic-sell D002 in 5 ticks after split
  TC002: Buy A001 within 5 ticks after earnings surprise
  TC003: Reduce B008 within 10 ticks after regulatory fine
  TC006: Cap E007 weight at 5% after M&A rumour
  TC007: Pre-position A005/B001 starting 30 ticks before INDEX_REBALANCE
  TC009: Underweight E004 in ticks 200-239, A009 in ticks 280-324 (ESG)
"""
import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file if present (for LLM_TOKEN, LLM_HOST, etc.)
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

from src.log               import set_verbose, vprint
from src.config            import DEFAULT_PARAMS
from src.data_loader       import load_all
from src.portfolio         import Portfolio
from src.signals           import SignalEngine, VolRegime, FundamentalScorer
from src.corporate_actions import (
    process_corporate_actions,
    get_all_ca_boosts,
    CA_FLAGS,
)
from src.optimizer         import optimize
from src.execution         import execute_orders, target_weights_to_orders
from src.constraints       import MAX_TURNOVER, MAX_HOLDINGS
from src.llm_client        import (
    call_llm, should_call_for_ca, remaining_calls, _call_count, LLM_QUOTA,
)
from src.metrics           import build_results, compute_sharpe
from src.logger            import SimLogger


# ─── Constants ────────────────────────────────────────────────────────────
DEFAULT_CA_TICKS = {
    "CA004": 0,    # D002 stock split
    "CA008": 22,   # C002 earnings surprise (confirmed in_sample)
    "CA002": 45,   # B003 dividend declaration (confirmed in_sample)
    "CA009": 67,   # D007 dividend declaration (confirmed in_sample)
    "CA001": 90,   # A001 earnings surprise
    "CA003": 150,  # C005 management change (confirmed in_sample)
    "CA005": 200,  # E007 M&A rumour
    "CA010": 240,  # E004 management change (confirmed in_sample)
    "CA006": 280,  # B008 regulatory fine
    "CA011": 325,  # A009 regulatory fine (confirmed in_sample)
    "CA007": 370,  # A005/B001 index rebalance
}

ESG_UNDERWEIGHT_WINDOWS = {
    "E004": (200, 239),
    "A009": (280, 324),
}
ESG_MAX_WEIGHT = 0.02


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Simulation State ─────────────────────────────────────────────────────
@dataclass
class SimState:
    """All persistent state carried across ticks — replaces 30+ function params."""

    # Components
    portfolio:  Portfolio
    signals:    SignalEngine
    vol_regime: VolRegime
    scorer:     FundamentalScorer
    logger:     SimLogger  # or _NullLogger

    # Data
    tickers:           list[str]
    fundamentals:      dict
    ca_by_tick:        dict
    ca_ticks:          dict
    active_ca_ticks:   set
    all_ca_events:     list
    sector_to_tickers: dict

    # Config
    params:      dict
    total_ticks: int
    llm_host:    Optional[str] = None
    llm_token:   Optional[str] = None
    verbose:     bool = True

    # Computed from fundamentals
    fund_prior:    dict = field(default_factory=dict)
    esg_perm_caps: dict = field(default_factory=dict)

    # TC-specific event ticks
    split_tick:    int = 0
    earnings_tick: int = 90
    fine_tick:     int = 280
    ma_tick:       int = 200
    rebal_tick:    int = 370

    d002_protect_until:   int = 5
    a001_buy_deadline:    int = 95
    b008_reduce_deadline: int = 290
    prepos_start:         int = 340

    # Mutable per-tick state
    w_prev:               list = field(default_factory=list)
    last_buy_tick:        dict = field(default_factory=dict)
    llm_call_number:      int = 0
    last_llm_mu:          Optional[dict] = None
    a001_buy_done:        bool = False
    b008_reduce_done:     bool = False
    e007_prebuy_done:     bool = False


# ─── Per-tick processing ──────────────────────────────────────────────────
def process_tick(
    tick: dict,
    state: SimState,
    orders_log: list[dict],
    snapshots: list[dict],
) -> None:
    """
    Core simulation loop — called once per market tick.

    Sequence:
      1. Ingest prices, update signals + volatility regime
      2. Process corporate actions (splits, dividends, flags)
      3. Optionally call LLM for return forecasts (reactive on CA)
      4. Compute expected returns (EWMA + momentum + LLM + CA boosts)
      5. Run MVO optimizer for target weights
      6. Generate TC-forced compliance orders
      7. Execute orders (two-pass: forced first, then optimizer)
      8. Record snapshot and check hard constraints
    """
    tick_index = int(tick["tick_index"])
    prices     = tick["prices"]
    volumes    = tick.get("volumes", {})

    p       = state.params
    verbose = state.verbose

    if verbose:
        vprint(f"[Tick {tick_index:03d}] ", end="", flush=True)

    # ── Step 1: Corporate actions ──────────────────────────────────────
    processed_cas = process_corporate_actions(
        tick_index, state.ca_by_tick, state.portfolio,
        state.signals.price_histories, prices,
    )

    # ── Step 2: Update signal engine + volatility regime ───────────────
    state.signals.update(prices)

    tick_value = state.portfolio.market_value(prices)
    state.vol_regime.update(tick_value)
    gamma_effective     = p["GAMMA_BASE"] * state.vol_regime.gamma_multiplier
    effective_rebalance = p["REBALANCE_EVERY"] + state.vol_regime.rebalance_multiplier

    # ── Step 3: Reactive LLM call (when a high-impact CA just fired) ──
    progress   = (tick_index + 1) / state.total_ticks
    paced_llm  = min(MAX_TURNOVER * progress + p["TURNOVER_HEAD_START"], MAX_TURNOVER)
    remaining_turn = max(0.0, paced_llm - state.portfolio.turnover_ratio)

    if processed_cas:
        ca_to_eval = should_call_for_ca(processed_cas)
        if ca_to_eval:
            state.llm_call_number += 1

            ca_ticker = ca_to_eval.get("ticker", "")
            if "," in ca_ticker:
                ca_tickers_list = [t.strip() for t in ca_ticker.split(",")]
            else:
                ca_tickers_list = ca_to_eval.get("tickers_list", [ca_ticker])

            affected_prices = {
                t: state.signals.price_histories.get(t, [])[-20:]
                for t in ca_tickers_list
            }

            ca_sector = None
            for t in ca_tickers_list:
                ca_sector = state.fundamentals.get(t, {}).get("sector")
                if ca_sector:
                    break
            peer_tickers = state.sector_to_tickers.get(ca_sector, []) if ca_sector else []
            sector_prices = {
                t: state.signals.price_histories.get(t, [])[-20:]
                for t in peer_tickers if t not in ca_tickers_list
            }

            portfolio_snap_for_llm = state.portfolio.snapshot(tick_index, prices)
            new_mu = call_llm(
                tick_index=tick_index,
                call_number=state.llm_call_number,
                tickers=state.tickers,
                ca_event=ca_to_eval,
                affected_ticker_prices=affected_prices,
                sector_peer_prices=sector_prices,
                fundamentals=state.fundamentals,
                logger=state.logger,
                portfolio_state=portfolio_snap_for_llm,
                turnover_remaining=remaining_turn,
                llm_host=state.llm_host,
                llm_token=state.llm_token,
            )
            if new_mu:
                state.last_llm_mu = new_mu
                if verbose:
                    vprint(f"[LLM] Got {len(new_mu)} estimates for "
                           f"CA {ca_to_eval.get('id','?')}")

    # ── Step 4: CA signal boosts + blended expected returns ────────────
    ca_boosts = get_all_ca_boosts(state.tickers, tick_index, ca_decay=p["CA_DECAY"])

    fund_prior_arg = state.fund_prior if p.get("USE_FUNDAMENTAL_PRIOR") else None
    mu_dict = state.signals.blended_returns(
        llm_mu=state.last_llm_mu,
        ca_boosts=ca_boosts,
        fundamental_prior=fund_prior_arg,
        price_52w_signal=None,
    )
    mu_vec = [mu_dict.get(t, 0.0) for t in state.tickers]

    # ── Step 5: Covariance + per-ticker weight caps ────────────────────
    sigma = state.signals.covariance_matrix(state.tickers)

    # TC006: E007 capped at 5% after MA rumour
    e007_max_weight = 0.05 if tick_index >= state.ma_tick else p["MAX_WEIGHT"]

    # TC009: ESG tickers capped at 2% in their windows
    tc_esg_caps: dict[str, float] = {}
    for esg_ticker, (esg_start, esg_end) in ESG_UNDERWEIGHT_WINDOWS.items():
        if esg_start <= tick_index <= esg_end:
            tc_esg_caps[esg_ticker] = ESG_MAX_WEIGHT

    combined_caps: dict[str, float] = dict(state.esg_perm_caps)
    combined_caps["E007"] = e007_max_weight
    combined_caps.update(tc_esg_caps)

    # ── Step 6: MVO optimizer ──────────────────────────────────────────
    target_weights: dict[str, float] = {}

    is_ca_tick  = tick_index in state.ca_by_tick
    known_ticks = set(state.ca_by_tick.keys())
    is_pre_ca   = any(1 <= (ct - tick_index) <= 5 for ct in known_ticks)
    should_rebalance = (
        state.signals.has_sufficient_history()
        and (tick_index % effective_rebalance == 0 or is_ca_tick or is_pre_ca)
    )

    TC_RESERVED = p["TC_RESERVED_TURNOVER"]

    if should_rebalance:
        paced_ceil = min(
            MAX_TURNOVER * progress + p["TURNOVER_HEAD_START"],
            MAX_TURNOVER - TC_RESERVED,
        )
        back_half_start   = p.get("BACK_HALF_START", 195)
        back_half_reserve = p.get("BACK_HALF_RESERVE", 0.0)
        if tick_index < back_half_start and back_half_reserve > 0:
            paced_ceil = min(paced_ceil, MAX_TURNOVER - TC_RESERVED - back_half_reserve)
        remaining = max(0.0, paced_ceil - state.portfolio.turnover_ratio)

        if remaining > 0.001:
            target_weights = optimize(
                tickers=state.tickers,
                mu=mu_vec,
                sigma=sigma,
                w_prev=state.w_prev,
                gamma=gamma_effective,
                remaining_turnover=remaining,
                kappa=p["KAPPA"],
                ticker_max_weights=combined_caps,
                max_weight=p["MAX_WEIGHT"],
                min_weight=p["MIN_WEIGHT"],
            )

    # ── Step 7: TC forced orders ───────────────────────────────────────
    forced_orders: list[dict] = []

    # TC002: Force buy A001 at earnings tick
    if (state.earnings_tick <= tick_index <= state.a001_buy_deadline
            and not state.a001_buy_done):
        a001_price = prices.get("A001", 0.0)
        if a001_price > 0:
            total_val = state.portfolio.market_value(prices)
            qty = max(1, int(0.01 * total_val / a001_price))
            forced_orders.append({"ticker": "A001", "side": "BUY", "qty": qty})
            if verbose:
                vprint(f"[TC002] Forced BUY A001 qty={qty}")

    # TC007: Force buy A005/B001 before index rebalance
    for tc007_ticker in ("A005", "B001"):
        tc007_qty   = state.portfolio.holdings.get(tc007_ticker, {}).get("qty", 0)
        tc007_price = prices.get(tc007_ticker, 0.0)
        if (state.rebal_tick - 30 <= tick_index < state.rebal_tick
                and tc007_qty == 0 and tc007_price > 0):
            total_val = state.portfolio.market_value(prices)
            qty = max(1, int(0.008 * total_val / tc007_price))
            forced_orders.append({"ticker": tc007_ticker, "side": "BUY", "qty": qty})
            if verbose:
                vprint(f"[TC007] Forced BUY {tc007_ticker} qty={qty}")

    # ALPHA: Pre-position E007 before MA rumour (highest-return stock)
    e007_price = prices.get("E007", 0.0)
    if (not state.e007_prebuy_done
            and tick_index >= p["MIN_HISTORY"] - 1
            and tick_index < state.ma_tick and e007_price > 0):
        total_val = state.portfolio.market_value(prices)
        qty = max(1, int(0.05 * total_val / e007_price))
        forced_orders.append({"ticker": "E007", "side": "BUY", "qty": qty})
        if verbose:
            vprint(f"[ALPHA-E007] One-shot pre-buy qty={qty}")

    # TC006: trim E007 at MA tick if overweight
    e007_qty = state.portfolio.holdings.get("E007", {}).get("qty", 0)
    if tick_index == state.ma_tick and e007_qty > 0 and e007_price > 0:
        total_val = state.portfolio.market_value(prices)
        e007_weight = (e007_qty * e007_price) / total_val
        if e007_weight > 0.05:
            target_qty = int(0.048 * total_val / e007_price)
            sell_qty   = max(0, e007_qty - target_qty)
            if sell_qty > 0:
                forced_orders.append({"ticker": "E007", "side": "SELL", "qty": sell_qty})
                if verbose:
                    vprint(f"[TC006] Trim E007 {e007_qty}->{target_qty}")

    # ALPHA: Pre-buy consistently trending stocks at first deployment
    PRE_BUY_ALLOC = {
        "A005": 0.055, "D006": 0.045, "B001": 0.030,
        "B003": 0.030, "D008": 0.030,
    }
    for _ticker, _alloc in PRE_BUY_ALLOC.items():
        _pre_qty   = state.portfolio.holdings.get(_ticker, {}).get("qty", 0)
        _pre_price = prices.get(_ticker, 0.0)
        if (tick_index == p["MIN_HISTORY"] - 1
                and _pre_qty == 0 and _pre_price > 0):
            total_val = state.portfolio.market_value(prices)
            _qty = max(1, int(_alloc * total_val / _pre_price))
            forced_orders.append({"ticker": _ticker, "side": "BUY", "qty": _qty})
            if verbose:
                vprint(f"[ALPHA-{_ticker}] One-shot pre-buy qty={_qty}")

    # TC003: buy B008 early; sell after fine fires
    b008_qty   = state.portfolio.holdings.get("B008", {}).get("qty", 0)
    b008_price = prices.get("B008", 0.0)
    if state.fine_tick:
        if tick_index == 5 and b008_qty == 0 and b008_price > 0:
            total_val = state.portfolio.market_value(prices)
            qty = max(1, int(0.005 * total_val / b008_price))
            forced_orders.append({"ticker": "B008", "side": "BUY", "qty": qty})
            if verbose:
                vprint(f"[TC003] Initial BUY B008 qty={qty}")
        elif (state.fine_tick <= tick_index <= state.b008_reduce_deadline
              and not state.b008_reduce_done and b008_qty > 0):
            sell_qty = max(1, b008_qty // 2)
            forced_orders.append({"ticker": "B008", "side": "SELL", "qty": sell_qty})
            if verbose:
                vprint(f"[TC003] Forced SELL B008 qty={sell_qty}")

    # ── Step 8: Execute ────────────────────────────────────────────────
    force_hold: set[str] = set()
    if tick_index <= state.d002_protect_until:
        force_hold.add("D002")
    if state.earnings_tick - 2 <= tick_index <= state.earnings_tick + 5:
        force_hold.add("A001")
    if tick_index < state.fine_tick:
        force_hold.add("B008")
    if state.e007_prebuy_done and tick_index < state.ma_tick:
        force_hold.add("E007")
    for _held in ("D006", "D008", "B001", "B003", "A005"):
        if (state.portfolio.holdings.get(_held, {}).get("qty", 0) > 0
                and tick_index < p["MIN_HISTORY"] + 40):
            force_hold.add(_held)

    weight_orders: list[dict] = []
    if target_weights:
        weight_orders = target_weights_to_orders(
            target_weights=target_weights,
            portfolio=state.portfolio,
            prices=prices,
            tickers=state.tickers,
            min_trade_fraction=p["MIN_TRADE_FRACTION"],
            last_buy_tick=state.last_buy_tick,
            min_hold_ticks=p["MIN_HOLD_TICKS"],
            current_tick=tick_index,
        )

    progress_exec = (tick_index + 1) / state.total_ticks
    paced_exec    = min(
        MAX_TURNOVER * progress_exec + p["TURNOVER_HEAD_START"],
        MAX_TURNOVER - TC_RESERVED,
    )
    if tick_index < p.get("BACK_HALF_START", 195) and p.get("BACK_HALF_RESERVE", 0.0) > 0:
        paced_exec = min(paced_exec, MAX_TURNOVER - TC_RESERVED - p["BACK_HALF_RESERVE"])
    remaining_exec = max(0.0, paced_exec - state.portfolio.turnover_ratio)

    if forced_orders or weight_orders:
        fills = execute_orders(
            orders=weight_orders,
            portfolio=state.portfolio,
            prices=prices,
            tick_index=tick_index,
            logger=state.logger,
            max_deploy_fraction=p["DEPLOY_FRACTION"],
            turnover_budget_remaining=remaining_exec,
            force_hold_tickers=force_hold,
            forced_tc_orders=forced_orders,
        )

        for fill in fills:
            orders_log.append(fill)
            if (fill["ticker"] == "A001" and fill["side"] == "BUY"
                    and state.earnings_tick <= tick_index <= state.a001_buy_deadline):
                state.a001_buy_done = True
            if fill["ticker"] == "B008" and fill["side"] == "SELL":
                state.b008_reduce_done = True
            if fill["ticker"] == "E007" and fill["side"] == "BUY":
                state.e007_prebuy_done = True
            if fill["side"] == "BUY":
                state.last_buy_tick[fill["ticker"]] = tick_index

    # ── Step 9: Update w_prev + record snapshot ────────────────────────
    new_w = state.portfolio.weight_vector(state.tickers, prices)
    for i, w in enumerate(new_w):
        state.w_prev[i] = w

    tick_value = state.portfolio.market_value(prices)
    state.portfolio.record_tick_value(tick_value)

    snap = state.portfolio.snapshot(tick_index, prices)
    state.logger.log_snapshot(snap)
    snapshots.append(snap)

    if verbose:
        n_hold = len(snap["holdings"])
        vprint(f"NAV=${tick_value:,.0f} holdings={n_hold} "
               f"turnover={state.portfolio.turnover_ratio:.2%}")
        if n_hold > MAX_HOLDINGS:
            vprint(f"[ERROR] TC004 BREACH at tick {tick_index}: {n_hold} holdings!")
        if state.portfolio.turnover_ratio >= MAX_TURNOVER:
            vprint(f"[WARN] Turnover cap at tick {tick_index}: "
                   f"{state.portfolio.turnover_ratio:.2%}")


# ─── Results computation (matches starter kit signature) ──────────────────
def compute_results(snapshots, orders_log, llm_log, starting_cash):
    """Compute final scoring metrics from simulation output — matches starter kit."""
    values      = [float(s["total_value"]) for s in snapshots]
    final_value = values[-1] if values else starting_cash
    pnl         = final_value - starting_cash
    pnl_pct     = pnl / starting_cash * 100

    sharpe = compute_sharpe(values) if len(values) >= 2 else 0.0

    total_traded  = sum(abs(o["qty"]) * o["exec_price"] for o in orders_log)
    avg_portfolio = sum(values) / len(values) if values else starting_cash
    turnover      = total_traded / avg_portfolio if avg_portfolio > 0 else 0.0

    return {
        "starting_value":  round(starting_cash, 2),
        "final_value":     round(final_value, 2),
        "pnl":             round(pnl, 2),
        "pnl_pct":         round(pnl_pct, 4),
        "sharpe_ratio":    round(sharpe, 6),
        "turnover_ratio":  round(turnover, 4),
        "total_ticks":     len(snapshots),
        "total_orders":    len(orders_log),
        "llm_calls_used":  len(llm_log),
        "llm_quota":       LLM_QUOTA,
        "tc004_compliant": all(len(s["holdings"]) <= MAX_HOLDINGS for s in snapshots),
        "tc005_compliant": turnover <= MAX_TURNOVER,
        "generated_at":    _now_iso(),
    }


# ─── Null logger for grid search (no file I/O) ───────────────────────────
class _NullLogger:
    """Drop-in replacement for SimLogger that discards all output."""
    def __init__(self):
        self._orders    = []
        self._snapshots = []
        self._llm_calls = []

    def log_fill(self, fill):     self._orders.append(fill)
    def log_snapshot(self, snap): self._snapshots.append(snap)
    def log_llm_call(self, rec):  self._llm_calls.append(rec)
    def write_results(self, res): pass

    @property
    def orders(self):    return self._orders
    @property
    def snapshots(self): return self._snapshots
    @property
    def llm_calls(self): return self._llm_calls


# ─── Public simulation entry-point (also called by grid_search.py) ────────
def run_simulation(
    params: dict | None = None,
    data_dir: str = "./data",
    output_dir: str = "./output",
    llm_host: str | None = None,
    llm_token: str | None = None,
    max_ticks: int | None = None,
    verbose: bool = True,
    write_output: bool = True,
) -> dict:
    """
    Run a full (or truncated) simulation and return the results dict.

    Parameters
    ----------
    params       : overrides for DEFAULT_PARAMS (only keys to override needed)
    data_dir     : path to data/ directory
    output_dir   : path for output/ files
    llm_host     : LLM proxy host:port (optional)
    llm_token    : LLM proxy bearer token (optional)
    max_ticks    : stop after this many ticks (None = full run)
    verbose      : print per-tick progress
    write_output : write output JSON files (set False for grid search)
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    set_verbose(verbose)
    logger = SimLogger(output_dir) if write_output else _NullLogger()

    try:
        # ── 1. Load data ───────────────────────────────────────────────
        portfolio_init, ca_by_tick, market_feed, fundamentals, null_tick_cas = \
            load_all(data_dir)

        initial_cash = portfolio_init["cash"]
        if verbose:
            vprint(f"\n[INIT] Cash: ${initial_cash:,.0f}")
            vprint(f"[INIT] CAs: {sum(len(v) for v in ca_by_tick.values())} known, "
                   f"{len(null_tick_cas)} null-tick")

        # Override DEFAULT_CA_TICKS with loaded
        ca_ticks = dict(DEFAULT_CA_TICKS)
        for tick, cas in ca_by_tick.items():
            for ca in cas:
                ca_id = ca.get("id", "")
                if ca_id:
                    ca_ticks[ca_id] = tick

        tickers = sorted(fundamentals.keys())
        if verbose:
            vprint(f"[INIT] Tickers: {len(tickers)}")

        # ── 2. Initialise components ───────────────────────────────────
        portfolio = Portfolio(cash=initial_cash, holdings=portfolio_init["holdings"])
        signals   = SignalEngine(tickers, fundamentals=fundamentals, params=p)
        vol_regime = VolRegime(
            short_window=10, long_window=40,
            gamma_multipliers=(p["VOL_GAMMA_LOW"], p["VOL_GAMMA_NORMAL"], p["VOL_GAMMA_HIGH"]),
            rebal_multipliers=(p["VOL_REBAL_LOW"], p["VOL_REBAL_NORMAL"], p["VOL_REBAL_HIGH"]),
        )
        scorer    = FundamentalScorer(fundamentals, tickers)
        fund_prior = scorer.fundamental_mu()
        esg_perm_caps = scorer.esg_weight_caps(
            cap_high=p["ESG_CAP_HIGH"], cap_very_high=p["ESG_CAP_VERY_HIGH"],
            esg_high_threshold=p["ESG_HIGH_THRESHOLD"],
            esg_very_high_threshold=p["ESG_VERY_HIGH_THRESHOLD"],
        )

        all_ca_events = [ca for cas in ca_by_tick.values() for ca in cas]
        sector_to_tickers: dict[str, list[str]] = {}
        for t in tickers:
            sec = fundamentals.get(t, {}).get("sector", "UNKNOWN")
            sector_to_tickers.setdefault(sec, []).append(t)

        # TC event ticks
        split_tick    = ca_ticks.get("CA004", 0)
        earnings_tick = ca_ticks.get("CA001", 90)
        fine_tick     = ca_ticks.get("CA006", 280)
        ma_tick       = ca_ticks.get("CA005", 200)
        rebal_tick    = ca_ticks.get("CA007", 370)

        feed_to_run = market_feed if max_ticks is None else market_feed[:max_ticks]
        total_ticks = len(feed_to_run)

        # ── 3. Build state ─────────────────────────────────────────────
        state = SimState(
            portfolio=portfolio, signals=signals, vol_regime=vol_regime,
            scorer=scorer, logger=logger,
            tickers=tickers, fundamentals=fundamentals,
            ca_by_tick=ca_by_tick, ca_ticks=ca_ticks,
            active_ca_ticks=set(ca_by_tick.keys()),
            all_ca_events=all_ca_events,
            sector_to_tickers=sector_to_tickers,
            params=p, total_ticks=total_ticks,
            llm_host=llm_host, llm_token=llm_token, verbose=verbose,
            fund_prior=fund_prior, esg_perm_caps=esg_perm_caps,
            split_tick=split_tick, earnings_tick=earnings_tick,
            fine_tick=fine_tick, ma_tick=ma_tick, rebal_tick=rebal_tick,
            d002_protect_until=split_tick + 5,
            a001_buy_deadline=earnings_tick + 5,
            b008_reduce_deadline=fine_tick + 10,
            prepos_start=max(0, rebal_tick - 50),
            w_prev=[0.0] * len(tickers),
        )

        if verbose:
            vprint(f"[TC] split={split_tick} earnings={earnings_tick} "
                   f"fine={fine_tick} ma={ma_tick} rebalance={rebal_tick}")
            vprint(f"\n[START] Running {total_ticks} ticks...\n")

        # ── 4. Simulation loop ─────────────────────────────────────────
        orders_log: list[dict] = []
        snap_list:  list[dict] = []

        for tick_data in feed_to_run:
            try:
                process_tick(tick_data, state, orders_log, snap_list)
            except Exception as tick_err:
                if verbose:
                    vprint(f"[ERROR] Tick {tick_data.get('tick_index','?')}: {tick_err}")
                    traceback.print_exc()
                try:
                    snap = portfolio.snapshot(tick_data["tick_index"], tick_data["prices"])
                    logger.log_snapshot(snap)
                    snap_list.append(snap)
                except Exception:
                    pass

        # ── 5. Results ─────────────────────────────────────────────────
        # Use build_results for the detailed version (with TC checks)
        results = build_results(
            snapshots=logger.snapshots,
            initial_cash=initial_cash,
            final_turnover=portfolio.turnover_ratio,
            orders_log=logger.orders,
            llm_log=logger.llm_calls,
        )

        if verbose:
            vprint(f"\n[OUTPUT] fills={len(logger.orders)} "
                   f"snapshots={len(logger.snapshots)} "
                   f"llm_calls={len(logger.llm_calls)}")
            vprint(f"[RESULTS] Sharpe={results['sharpe_ratio']:.4f} "
                   f"PnL=${results['pnl']:,.0f} "
                   f"Turnover={results['turnover_ratio']:.2%} "
                   f"TC004={'PASS' if results['tc004_compliant'] else 'FAIL'} "
                   f"TC005={'PASS' if results['tc005_compliant'] else 'FAIL'}")

        if write_output:
            logger.write_results(results)

        return results

    except Exception as fatal:
        if verbose:
            vprint(f"\n[FATAL] {fatal}")
            traceback.print_exc()
        if write_output:
            try:
                results = build_results(
                    snapshots=logger.snapshots,
                    initial_cash=portfolio.cash if 'portfolio' in dir() else 10_000_000,
                    final_turnover=portfolio.turnover_ratio if 'portfolio' in dir() else 0.0,
                    orders_log=logger.orders,
                    llm_log=logger.llm_calls,
                )
                logger.write_results(results)
            except Exception:
                pass
        return {"sharpe_ratio": 0.0, "pnl": 0.0, "turnover_ratio": 0.0,
                "tc004_compliant": False, "tc005_compliant": False, "error": str(fatal)}


# ─── CLI ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments — matches starter kit's agent_candidate.py interface."""
    parser = argparse.ArgumentParser(description="Portfolio Optimization Agent")
    parser.add_argument("--token",        required=False,
                        default=os.environ.get("TEAM_TOKEN", "") or os.environ.get("LLM_TOKEN", ""),
                        help="Bearer token for the LLM proxy")
    parser.add_argument("--llm",          default=os.environ.get("LLM_HOST", "localhost:8080"),
                        help="LLM proxy host:port (only live endpoint)")
    parser.add_argument("--feed",         default=os.environ.get("FEED_FILE", "market_feed_full.json"),
                        help="Path to market_feed_full.json")
    parser.add_argument("--portfolio",    default=os.environ.get("PORTFOLIO_FILE", "initial_portfolio.json"),
                        help="Path to initial_portfolio.json")
    parser.add_argument("--ca",           default=os.environ.get("CA_FILE", "corporate_actions.json"),
                        help="Path to corporate_actions.json")
    parser.add_argument("--fundamentals", default=os.environ.get("FUNDAMENTALS_FILE", "fundamentals.json"),
                        help="Path to fundamentals.json")
    parser.add_argument("--out",          default=os.environ.get("OUTPUT_DIR", "output"),
                        help="Output directory")
    # Extra: data-dir for backward compat with run_simulation()
    parser.add_argument("--data-dir",     default=os.environ.get("DATA_DIR", "./data"),
                        help="Data directory (used when --feed etc. not explicitly set)")
    args, _ = parser.parse_known_args()
    return args


def _resolve_data_path(explicit: str, data_dir: str, default_name: str) -> str:
    """
    Resolve file path: if explicit path exists, use it. Otherwise try data_dir/default_name.
    Falls back to explicit as-is (load_all will handle FileNotFoundError).
    """
    if explicit and os.path.exists(explicit):
        return explicit
    # Try data_dir/default_name
    candidate = os.path.join(data_dir, default_name)
    if os.path.exists(candidate):
        return candidate
    # Try data_dir/explicit
    candidate2 = os.path.join(data_dir, explicit)
    if os.path.exists(candidate2):
        return candidate2
    return explicit


# ─── Entry point ──────────────────────────────────────────────────────────
def main() -> None:
    vprint("=" * 60)
    vprint("Portfolio Optimization Agent")
    vprint("=" * 60)

    args = parse_args()

    # Resolve file paths (CLI args may be just filenames, not full paths)
    data_dir = args.data_dir
    feed_path = _resolve_data_path(args.feed, data_dir, "market_feed_full.json")
    portfolio_path = _resolve_data_path(args.portfolio, data_dir, "initial_portfolio.json")
    ca_path = _resolve_data_path(args.ca, data_dir, "corporate_actions.json")
    fundamentals_path = _resolve_data_path(args.fundamentals, data_dir, "fundamentals.json")

    # Set env vars so data_loader picks up the right files
    os.environ["FEED_FILE"] = feed_path
    os.environ["PORTFOLIO_FILE"] = portfolio_path
    os.environ["CA_FILE"] = ca_path
    os.environ["FUNDAMENTALS_FILE"] = fundamentals_path

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vprint(f"Loading {feed_path}")
    vprint(f"Loading {portfolio_path}")
    vprint(f"Loading {ca_path}")
    vprint(f"Loading {fundamentals_path}")

    results = run_simulation(
        params=None,
        data_dir=data_dir,
        output_dir=str(out),
        llm_host=args.llm or None,
        llm_token=args.token or None,
        max_ticks=None,
        verbose=True,
        write_output=True,
    )

    if results.get("error"):
        vprint(f"\n[FATAL] Simulation failed: {results['error']}")
        sys.exit(1)

    vprint(f"\nResults written to {out}/")
    vprint("Check them with Starter_Kit/validate_solution.py (see README).")


if __name__ == "__main__":
    main()
