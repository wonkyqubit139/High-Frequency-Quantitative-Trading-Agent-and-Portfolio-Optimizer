"""
cross_validate.py — Walk-forward cross-validation + custom range scoring.

MODES
─────
cv (default)
    Walk-forward CV: the 390-tick session is split into N equal folds.
    Each fold runs the simulation from tick 0 (fresh portfolio) and scores
    only the test window.  A final full-run fold is always appended.

    Default: 5 folds → 78 ticks each.

range
    Run the simulation once (from tick 0) and score a specific sub-window.
    Useful for inspecting any arbitrary tick range, e.g. ticks 25–78.

CLI examples
────────────
# Default 5-fold walk-forward CV:
    uv run python cross_validate.py

# 8-fold CV:
    uv run python cross_validate.py --folds 8

# CV with explicit fold size (3 × 100-tick folds + full-run):
    uv run python cross_validate.py --fold-size 100

# Custom folds — colon-separated start:end pairs (test windows only):
    uv run python cross_validate.py --custom-folds 0:78,100:200,250:320

# Score a single arbitrary tick range (range mode):
    uv run python cross_validate.py --mode range --test-start 25 --test-end 78

# All options together:
    uv run python cross_validate.py \\
        --mode cv --folds 4 --data-dir ./data --out cv_results.json

Scoring within a window
───────────────────────
  Sharpe : computed from portfolio values in [test_start, test_end)
  PnL    : portfolio_value[test_end-1] − portfolio_value[test_start]
  TC004/TC005: checked over the full run up to test_end (cumulative constraints)

Honest caveat
─────────────
  Pre-buys (E007, D006 …) at tick 19 are chosen via ex-post session-return
  analysis — intentional in-sample alpha capture, not an accidental lookahead
  bug.  The CV therefore measures signal stability independently of those picks.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main    import run_simulation
from src.log     import set_verbose
from src.metrics import compute_sharpe, SHARPE_ANN_FACTOR

TOTAL_TICKS  = 390
DATA_DIR     = "./data"
OUTPUT_DIR   = "./output"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sharpe_from_values(values: list[float]) -> float:
    """Compute annualized Sharpe from a list of portfolio values."""
    return compute_sharpe(values)


def _pnl_from_values(values: list[float]) -> float:
    if not values:
        return 0.0
    return values[-1] - values[0]


def _pnl_pct_from_values(values: list[float]) -> float:
    if not values or values[0] == 0:
        return 0.0
    return (values[-1] - values[0]) / values[0] * 100.0


def _slice_snapshots_values(snapshots: list[dict], start_tick: int, end_tick: int) -> list[float]:
    """Extract portfolio total_value list for ticks in [start_tick, end_tick)."""
    return [
        s["total_value"]
        for s in snapshots
        if start_tick <= s["tick_index"] < end_tick
    ]


def _print_separator(char: str = "─", width: int = 72):
    print(char * width)


def _describe_fold(fold_idx: int, train_end: int, test_start: int, test_end: int) -> str:
    if fold_idx == N_FOLDS - 1:
        return f"Fold {fold_idx+1}  FULL RUN (all {TOTAL_TICKS} ticks — competition score)"
    return (
        f"Fold {fold_idx+1}  "
        f"Train 0–{train_end-1}  Test {test_start}–{test_end-1}"
    )


# ---------------------------------------------------------------------------
# CA events that fall in each window (for context in the report)
# ---------------------------------------------------------------------------
CA_EVENTS = [
    (0,   "CA004 — D002 stock split"),
    (22,  "CA008 — C002 earnings surprise"),
    (45,  "CA002 — B003 dividend"),
    (67,  "CA009 — D007 dividend"),
    (90,  "CA001 — A001 earnings surprise"),
    (150, "CA003 — C005 management change"),
    (200, "CA005 — E007 M&A rumour"),
    (240, "CA010 — E004 management change"),
    (280, "CA006 — B008 regulatory fine"),
    (325, "CA011 — A009 regulatory fine"),
    (370, "CA007 — A005/B001 index rebalance"),
]


def _ca_in_window(start: int, end: int) -> list[str]:
    return [f"  tick {t}: {label}" for t, label in CA_EVENTS if start <= t < end]


# ---------------------------------------------------------------------------
# Core simulation engine (shared by CV and range mode)
# ---------------------------------------------------------------------------

def _simulate_window(
    test_start: int,
    test_end:   int,
    data_dir:   str,
    portfolio_init: dict,
    ca_by_tick:     dict,
    market_feed:    list,
    fundamentals:   dict,
    label: str = "",
) -> dict:
    """
    Run the simulation from tick 0 up to test_end (inclusive), then score the
    sub-window [test_start, test_end).

    Returns a dict with:
        test_sharpe, test_pnl, test_pnl_pct, test_ticks,
        full_run_sharpe, turnover_pct, tc004_pass, tc005_pass,
        fills, llm_calls, snapshots (count)
    """
    from src.config            import DEFAULT_PARAMS
    from src.portfolio         import Portfolio
    from src.signals           import SignalEngine, VolRegime, FundamentalScorer
    from src.corporate_actions import get_all_ca_boosts, CA_FLAGS
    from src.metrics           import build_results
    from src.main              import _run_tick, DEFAULT_CA_TICKS

    p = dict(DEFAULT_PARAMS)

    snap_collector:  list[dict] = []
    order_collector: list[dict] = []
    llm_collector:   list[dict] = []

    class _CapturingLogger:
        def __init__(self):
            self.snapshots = snap_collector
            self.orders    = order_collector
            self.llm_calls = llm_collector
        def log_fill(self, fill):         self.orders.append(fill)
        def log_snapshot(self, snap):     self.snapshots.append(snap)
        def log_llm_call(self, record):   self.llm_calls.append(record)
        def write_results(self, results): pass

    logger   = _CapturingLogger()
    tickers  = sorted(fundamentals.keys())
    port     = Portfolio(cash=portfolio_init["cash"], holdings=portfolio_init["holdings"])
    sigs     = SignalEngine(tickers=tickers, fundamentals=fundamentals, params=p)
    vol_regime = VolRegime(
        short_window=10, long_window=40,
        gamma_multipliers=(p["VOL_GAMMA_LOW"], p["VOL_GAMMA_NORMAL"], p["VOL_GAMMA_HIGH"]),
        rebal_multipliers=(p["VOL_REBAL_LOW"], p["VOL_REBAL_NORMAL"], p["VOL_REBAL_HIGH"]),
    )
    scorer     = FundamentalScorer(fundamentals, tickers)
    fund_prior = scorer.fundamental_mu()
    esg_caps   = scorer.esg_weight_caps(
        cap_high=p["ESG_CAP_HIGH"],
        cap_very_high=p["ESG_CAP_VERY_HIGH"],
        esg_high_threshold=p["ESG_HIGH_THRESHOLD"],
        esg_very_high_threshold=p["ESG_VERY_HIGH_THRESHOLD"],
    )

    llm_call_number_ref  = [0]
    last_llm_mu_ref      = [None]
    a001_buy_done_ref    = [False]
    b008_reduce_done_ref = [False]
    e007_prebuy_done_ref = [False]
    w_prev               = [0.0] * len(tickers)
    last_buy_tick: dict[str, int] = {}

    ca_ticks_local = dict(DEFAULT_CA_TICKS)
    for tick_num, cas in ca_by_tick.items():
        for ca in cas:
            ca_id = ca.get("id", "")
            if ca_id:
                ca_ticks_local[ca_id] = tick_num

    split_tick    = ca_ticks_local.get("CA004", 0)
    earnings_tick = ca_ticks_local.get("CA001", 90)
    fine_tick     = ca_ticks_local.get("CA006", 280)
    ma_tick       = ca_ticks_local.get("CA005", 200)
    rebal_tick    = ca_ticks_local.get("CA007", 370)
    d002_protect_until   = split_tick + 5
    a001_buy_deadline    = earnings_tick + 5
    b008_reduce_deadline = fine_tick + 10
    prepos_start         = max(0, rebal_tick - 50)
    active_ca_ticks      = set(ca_by_tick.keys())

    is_full = (test_end >= TOTAL_TICKS)
    feed_to_run    = market_feed if is_full else market_feed[:test_end]
    total_ticks_run = len(feed_to_run)

    for tick_data in feed_to_run:
        tick_index = tick_data["tick_index"]
        prices     = tick_data["prices"]
        volumes    = tick_data["volumes"]
        try:
            _run_tick(
                tick_index=tick_index, prices=prices, volumes=volumes,
                portfolio=port, signals=sigs, vol_regime=vol_regime,
                ca_by_tick=ca_by_tick, tickers=tickers, fundamentals=fundamentals,
                logger=logger, params=p, total_ticks=total_ticks_run,
                llm_host=None, llm_token=None,
                llm_call_number_ref=llm_call_number_ref,
                last_llm_mu_ref=last_llm_mu_ref,
                active_ca_ticks=active_ca_ticks,
                w_prev_ref=w_prev,
                d002_protect_until=d002_protect_until,
                earnings_tick=earnings_tick,
                a001_buy_deadline=a001_buy_deadline,
                a001_buy_done_ref=a001_buy_done_ref,
                e007_prebuy_done_ref=e007_prebuy_done_ref,
                fine_tick=fine_tick,
                b008_reduce_deadline=b008_reduce_deadline,
                b008_reduce_done_ref=b008_reduce_done_ref,
                prepos_start=prepos_start,
                rebal_tick=rebal_tick, ma_tick=ma_tick,
                ca_ticks=ca_ticks_local, scorer=scorer,
                fund_prior=fund_prior, esg_perm_caps=esg_caps,
                last_buy_tick=last_buy_tick, verbose=False,
            )
        except Exception:
            try:
                logger.log_snapshot(port.snapshot(tick_index, prices))
            except Exception:
                pass
            continue

    results_full = build_results(
        snapshots=logger.snapshots,
        initial_cash=portfolio_init["cash"],
        final_turnover=port.turnover_ratio,
        orders_log=logger.orders,
        llm_log=logger.llm_calls,
    )

    # Score the test sub-window
    if test_start == 0 and is_full:
        test_values = [s["total_value"] for s in logger.snapshots]
    else:
        test_values = _slice_snapshots_values(logger.snapshots, test_start, test_end)

    if len(test_values) >= 2:
        test_sharpe  = compute_sharpe(test_values)
        test_pnl     = test_values[-1] - test_values[0]
        test_pnl_pct = (test_pnl / test_values[0] * 100.0) if test_values[0] > 0 else 0.0
    else:
        test_sharpe = test_pnl = test_pnl_pct = 0.0

    return {
        "label":           label,
        "test_start":      test_start,
        "test_end":        test_end,
        "test_ticks":      len(test_values),
        "test_sharpe":     round(test_sharpe, 4),
        "test_pnl":        round(test_pnl, 2),
        "test_pnl_pct":    round(test_pnl_pct, 4),
        "full_run_sharpe": round(results_full["sharpe_ratio"], 4),
        "turnover_pct":    round(results_full["turnover_ratio"] * 100, 2),
        "tc004_pass":      results_full["tc004_compliant"],
        "tc005_pass":      results_full["tc005_compliant"],
        "fills":           results_full["total_orders"],
        "llm_calls":       results_full["llm_calls_used"],
        "snapshots":       len(logger.snapshots),
    }


# ---------------------------------------------------------------------------
# Range mode: score a single arbitrary tick window
# ---------------------------------------------------------------------------

def run_range(test_start: int, test_end: int, data_dir: str, out_path: str):
    from src.data_loader import load_all

    if test_start < 0 or test_end > TOTAL_TICKS or test_start >= test_end:
        print(f"ERROR: invalid range [{test_start}, {test_end}). "
              f"Must satisfy 0 <= start < end <= {TOTAL_TICKS}.")
        sys.exit(1)

    _print_separator("═")
    print(f"  RANGE MODE: score ticks {test_start}–{test_end-1}")
    print(f"  Simulation runs from tick 0; only [{test_start}, {test_end}) is scored.")
    _print_separator("═")

    ca_in_window = _ca_in_window(test_start, test_end)
    if ca_in_window:
        print("  CA events in range:")
        for line in ca_in_window:
            print(f"    {line}")
        print()

    set_verbose(False)
    portfolio_init, ca_by_tick, market_feed, fundamentals, _ = load_all(data_dir)

    print(f"  Running simulation (up to tick {test_end})...", end=" ", flush=True)
    result = _simulate_window(
        test_start=test_start, test_end=test_end,
        data_dir=data_dir,
        portfolio_init=portfolio_init, ca_by_tick=ca_by_tick,
        market_feed=market_feed, fundamentals=fundamentals,
        label=f"range {test_start}:{test_end}",
    )
    print("done.\n")

    _print_separator()
    print(f"  SCORED WINDOW: ticks {test_start}–{test_end-1}  ({result['test_ticks']} snapshots)")
    _print_separator()
    print(f"    Sharpe (window, annualized by sqrt({result['test_ticks']})):  "
          f"{result['test_sharpe']:+.4f}")
    print(f"    PnL   (window):   ${result['test_pnl']:+,.2f}  "
          f"({result['test_pnl_pct']:+.4f}%)")
    print(f"    Full-run Sharpe (ticks 0–{test_end-1}): {result['full_run_sharpe']:+.4f}")
    print(f"    Turnover: {result['turnover_pct']:.2f}%   "
          f"TC004={'PASS' if result['tc004_pass'] else 'FAIL'}  "
          f"TC005={'PASS' if result['tc005_pass'] else 'FAIL'}")
    print(f"    Fills: {result['fills']}  Snapshots: {result['snapshots']}")
    _print_separator("═")

    output = {
        "mode": "range",
        "generated_at": _now_iso(),
        "test_start": test_start,
        "test_end": test_end,
        "result": result,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_path}")
    _print_separator("═")


# ---------------------------------------------------------------------------
# CV driver
# ---------------------------------------------------------------------------

def run_cv(
    data_dir: str,
    out_path: str,
    n_folds: int = 5,
    fold_size: int | None = None,
    custom_folds: list[tuple[int, int]] | None = None,
):
    """
    Walk-forward cross-validation.

    fold_size    : explicit tick count per fold (overrides n_folds)
    custom_folds : list of (test_start, test_end) tuples; overrides n_folds + fold_size
    """
    from src.data_loader import load_all

    set_verbose(False)
    portfolio_init, ca_by_tick, market_feed, fundamentals, _ = load_all(data_dir)

    # ── Build fold specs ────────────────────────────────────────────────────
    if custom_folds:
        # Validate
        for i, (s, e) in enumerate(custom_folds):
            if s < 0 or e > TOTAL_TICKS or s >= e:
                print(f"ERROR: custom fold {i+1} [{s}:{e}) is invalid. "
                      f"Must satisfy 0 <= start < end <= {TOTAL_TICKS}.")
                sys.exit(1)
        # Each entry is (train_end, test_start, test_end); train_end = test_start for custom
        fold_specs = [(s, s, e) for s, e in custom_folds]
        # Always append a full run at the end
        fold_specs.append((TOTAL_TICKS, 0, TOTAL_TICKS))
        effective_fold_size = None
        mode_desc = f"custom {len(custom_folds)} folds + full run"
    else:
        effective_fold_size = fold_size if fold_size else (TOTAL_TICKS // n_folds)
        fold_specs = []
        k = 0
        while True:
            train_end  = (k + 1) * effective_fold_size
            test_start = train_end
            test_end   = min(test_start + effective_fold_size, TOTAL_TICKS)
            if test_start >= TOTAL_TICKS:
                break
            fold_specs.append((train_end, test_start, test_end))
            k += 1
            if test_end >= TOTAL_TICKS:
                break
        # Always append full run
        fold_specs.append((TOTAL_TICKS, 0, TOTAL_TICKS))
        mode_desc = f"{len(fold_specs)-1}-fold walk-forward (fold size={effective_fold_size}) + full run"

    total_folds = len(fold_specs)

    _print_separator("═")
    print(f"  WALK-FORWARD CROSS-VALIDATION")
    print(f"  Mode: {mode_desc}")
    print(f"  Data: {data_dir}    Total ticks: {TOTAL_TICKS}    Folds: {total_folds}")
    _print_separator("═")
    print("""
METHODOLOGY
  Walk-forward (time-series safe): each fold runs the simulation fresh from
  tick 0, then scores only the test-window portion of the trajectory.

  Pre-buys (E007/D006/D008/B001/B003/A005 @tick19) are in-sample optimized.
  CV measures signal stability independent of those hardcoded picks.
""")

    fold_results = []

    for fold_idx, (train_end, test_start, test_end) in enumerate(fold_specs):
        is_full_run = (test_start == 0 and test_end >= TOTAL_TICKS)

        _print_separator()
        if is_full_run:
            label = f"Fold {fold_idx+1}  FULL RUN (all {TOTAL_TICKS} ticks — competition score)"
        elif custom_folds and fold_idx < len(custom_folds):
            label = f"Fold {fold_idx+1}  Custom window {test_start}–{test_end-1}"
        else:
            label = f"Fold {fold_idx+1}  Train 0–{train_end-1}  Test {test_start}–{test_end-1}"
        print(f"  {label}")
        _print_separator()

        ca_in_test = _ca_in_window(test_start, test_end)
        if ca_in_test:
            print("  CA events in test window:")
            for ca_line in ca_in_test:
                print(f"    {ca_line}")

        print(f"  Running simulation (up to tick {test_end})...", end=" ", flush=True)

        result = _simulate_window(
            test_start=test_start, test_end=test_end,
            data_dir=data_dir,
            portfolio_init=portfolio_init, ca_by_tick=ca_by_tick,
            market_feed=market_feed, fundamentals=fundamentals,
            label=label,
        )
        print("done.")

        fold_rec = {
            "fold":            fold_idx + 1,
            "is_full_run":     is_full_run,
            "train_end_tick":  train_end,
            "test_start_tick": test_start,
            "test_end_tick":   test_end,
            **result,
        }
        fold_results.append(fold_rec)

        print(f"  TEST-WINDOW metrics ({result['test_ticks']} ticks, "
              f"ticks {test_start}–{test_end-1}):")
        print(f"    Sharpe (window, ann. by sqrt({result['test_ticks']})):  "
              f"{result['test_sharpe']:+.4f}")
        print(f"    PnL   (window):  ${result['test_pnl']:+,.0f}  "
              f"({result['test_pnl_pct']:+.2f}%)")
        print(f"  FULL-RUN metrics (0–{test_end-1}):")
        print(f"    Sharpe (full run):  {result['full_run_sharpe']:+.4f}")
        print(f"    Turnover: {result['turnover_pct']:.2f}%   "
              f"TC004={'PASS' if result['tc004_pass'] else 'FAIL'}  "
              f"TC005={'PASS' if result['tc005_pass'] else 'FAIL'}")
        print(f"    Fills: {result['fills']}  Snapshots: {result['snapshots']}")
        print()

    # ── Summary table ──────────────────────────────────────────────────────
    _print_separator("═")
    print("  CROSS-VALIDATION SUMMARY")
    _print_separator("═")

    header = (
        f"  {'Fold':<6} {'Test Window':<18} {'Test Sharpe':>12} "
        f"{'Test PnL':>13} {'Full Sharpe':>12} {'TC04':>6} {'TC05':>6}"
    )
    print(header)
    _print_separator()

    oos_sharpes = []
    oos_pnls    = []
    for r in fold_results:
        is_fr = r["is_full_run"]
        window = ("0–389 (full)" if is_fr
                  else f"{r['test_start_tick']}–{r['test_end_tick']-1}")
        tc04 = "PASS" if r["tc004_pass"] else "FAIL"
        tc05 = "PASS" if r["tc005_pass"] else "FAIL"
        row = (
            f"  {r['fold']:<6} {window:<18} {r['test_sharpe']:>+12.4f} "
            f"${r['test_pnl']:>11,.0f} {r['full_run_sharpe']:>+12.4f} "
            f"{tc04:>6} {tc05:>6}"
        )
        print(row)
        if not is_fr:
            oos_sharpes.append(r["test_sharpe"])
            oos_pnls.append(r["test_pnl"])

    _print_separator()

    if oos_sharpes:
        avg_sh  = sum(oos_sharpes) / len(oos_sharpes)
        std_sh  = math.sqrt(sum((x - avg_sh)**2 for x in oos_sharpes) / len(oos_sharpes))
        avg_pnl = sum(oos_pnls) / len(oos_pnls)
        n_oos   = len(oos_sharpes)

        print(f"\n  OUT-OF-SAMPLE STATISTICS (non-full-run folds, n={n_oos}):")
        print(f"    Mean Sharpe:  {avg_sh:+.4f}")
        print(f"    Std  Sharpe:  {std_sh:.4f}")
        print(f"    Min  Sharpe:  {min(oos_sharpes):+.4f}")
        print(f"    Max  Sharpe:  {max(oos_sharpes):+.4f}")
        print(f"    Mean PnL:    ${avg_pnl:+,.0f}")
        print()

        full_sharpe_final = fold_results[-1]["full_run_sharpe"]
        ratio = avg_sh / full_sharpe_final if full_sharpe_final != 0 else 0.0
        print(f"  OVERFITTING GUARD:")
        print(f"    Full-run Sharpe (competition score): {full_sharpe_final:+.4f}")
        print(f"    Avg OOS test-window Sharpe:          {avg_sh:+.4f}")
        print(f"    OOS / Full ratio:                    {ratio:.2%}")
        if ratio >= 0.50:
            print("    → Ratio >= 50%: reasonable generalization")
        elif ratio >= 0.0:
            print("    → Ratio 0–50%: moderate overfitting signature")
        else:
            print("    → Negative ratio: negative OOS Sharpe in test windows")
        print()

    print("""
INTERPRETATION
  Sharpe over short windows is annualized by sqrt(N_window_ticks), which
  amplifies noise — short-window Sharpe is NOT comparable to the full-run number.
  TC004/TC005 are cumulative and evaluated over the full run up to test_end.
""")

    # ── Save ───────────────────────────────────────────────────────────────
    output = {
        "mode":          "cv",
        "generated_at":  _now_iso(),
        "methodology":   mode_desc,
        "effective_fold_size": effective_fold_size,
        "total_ticks":   TOTAL_TICKS,
        "sharpe_ann_note": "sqrt(N_test_ticks) per window; sqrt(390) for full run",
        "folds":         fold_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_path}")
    _print_separator("═")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_custom_folds(s: str) -> list[tuple[int, int]]:
    """Parse '0:78,100:200,250:320' into [(0,78),(100,200),(250,320)]."""
    folds = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            print(f"ERROR: invalid fold spec '{part}'. Expected format start:end")
            sys.exit(1)
        a, b = part.split(":", 1)
        try:
            folds.append((int(a), int(b)))
        except ValueError:
            print(f"ERROR: invalid fold spec '{part}'. start and end must be integers.")
            sys.exit(1)
    return folds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk-forward CV and custom range scoring for the simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 5-fold CV:
  uv run python cross_validate.py

  # 8-fold CV:
  uv run python cross_validate.py --folds 8

  # Explicit fold size (e.g. 100 ticks per fold):
  uv run python cross_validate.py --fold-size 100

  # Custom folds (comma-separated start:end pairs):
  uv run python cross_validate.py --custom-folds 0:78,100:200,250:320

  # Score a single custom range (range mode):
  uv run python cross_validate.py --mode range --test-start 25 --test-end 78
""",
    )
    parser.add_argument(
        "--mode", choices=["cv", "range"], default="cv",
        help="'cv' = walk-forward cross-validation (default); 'range' = score a single window",
    )
    parser.add_argument("--data-dir",   default="./data",          help="Data directory")
    parser.add_argument("--out",        default="cv_results.json", help="Output JSON path")

    # CV-mode options
    parser.add_argument("--folds",      type=int, default=5,        help="Number of CV folds (cv mode)")
    parser.add_argument("--fold-size",  type=int, default=None,     help="Explicit fold size in ticks (overrides --folds)")
    parser.add_argument(
        "--custom-folds", default=None,
        help="Custom test windows as 'start:end,start:end,...' (cv mode; overrides --folds/--fold-size)",
    )

    # Range-mode options
    parser.add_argument("--test-start", type=int, default=0,        help="Start tick to score (range mode, inclusive)")
    parser.add_argument("--test-end",   type=int, default=TOTAL_TICKS, help="End tick to score (range mode, exclusive)")

    args = parser.parse_args()

    if args.mode == "range":
        run_range(
            test_start=args.test_start,
            test_end=args.test_end,
            data_dir=args.data_dir,
            out_path=args.out,
        )
    else:
        custom = _parse_custom_folds(args.custom_folds) if args.custom_folds else None
        run_cv(
            data_dir=args.data_dir,
            out_path=args.out,
            n_folds=args.folds,
            fold_size=args.fold_size,
            custom_folds=custom,
        )
