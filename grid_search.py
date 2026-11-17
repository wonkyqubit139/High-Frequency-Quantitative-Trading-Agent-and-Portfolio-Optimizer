"""
grid_search.py — Parameter sweep to maximise Sharpe.

Usage:
    uv run python grid_search.py                # phase 2 then 3
    uv run python grid_search.py --phase 1      # independent sweeps
    uv run python grid_search.py --phase 2      # cartesian
    uv run python grid_search.py --phase 3      # validate top configs
"""
import argparse
import itertools
import json
import os
import sys
import time
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DEFAULT_PARAMS
from src.main   import run_simulation
from src.log    import set_verbose

DATA_DIR    = "./data"
TRAIN_TICKS = 273
TOTAL_TICKS = 390

# ── Sweep spaces ──────────────────────────────────────────────────────────

PHASE1_SWEEPS: dict[str, list] = {
    "GAMMA_BASE":          [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0],
    "KAPPA":               [0.0, 0.0001, 0.0005, 0.001, 0.003, 0.005],
    "REBALANCE_EVERY":     [1, 2, 3, 5, 8, 10, 15],
    "TURNOVER_HEAD_START": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
    "EWMA_LAMBDA":         [0.85, 0.88, 0.90, 0.94, 0.97, 0.99],
    "MOMENTUM_N":          [3, 5, 7, 10, 15, 20],
    "CUM_SHARPE_SCALE":    [0.0, 0.001, 0.002, 0.003, 0.005, 0.010, 0.020],
    "BLEND_COMBO":         [
        (0.0, 0.7, 0.3), (0.0, 1.0, 0.0), (0.0, 0.5, 0.5),
        (0.2, 0.5, 0.3), (0.4, 0.4, 0.2), (0.5, 0.3, 0.2),
        (0.6, 0.2, 0.2), (0.0, 0.0, 1.0),
    ],
    "DEPLOY_FRACTION":     [0.10, 0.15, 0.20, 0.30],
    "CA_DECAY":            [0.80, 0.85, 0.90, 0.95, 0.99, 1.00],
    "MIN_TRADE_FRACTION":  [0.001, 0.002, 0.005, 0.010, 0.020],
    "MAX_WEIGHT":          [0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
    "MIN_WEIGHT":          [0.001, 0.003, 0.005, 0.008, 0.010],
}

PHASE2_PARAMS: dict[str, list] = {
    "REBALANCE_EVERY": [5, 10, 15, 20],
    "DEPLOY_FRACTION": [0.10, 0.15, 0.30],
    "BLEND_COMBO":     [(0.0, 1.0, 0.0), (0.0, 0.7, 0.3), (0.4, 0.4, 0.2)],
    "MAX_WEIGHT":      [0.10, 0.15, 0.20],
    "KAPPA":           [0.001, 0.003],
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _run(overrides: dict, max_ticks: int | None = None) -> dict:
    params = dict(DEFAULT_PARAMS)
    if "BLEND_COMBO" in overrides:
        c = overrides.pop("BLEND_COMBO")
        params["BLEND_CUM_SHARPE"], params["BLEND_EWMA"], params["BLEND_MOMENTUM"] = c
    params.update(overrides)
    set_verbose(False)
    try:
        return run_simulation(params=params, data_dir=DATA_DIR,
                              max_ticks=max_ticks, verbose=False, write_output=False)
    except Exception as e:
        return {"sharpe_ratio": 0.0, "pnl": 0.0, "turnover_ratio": 0.0,
                "tc004_compliant": False, "tc005_compliant": False, "error": str(e)}
    finally:
        set_verbose(True)


def _rec(overrides, result, phase, run_id):
    return {
        "run_id": run_id, "phase": phase, "params": dict(overrides),
        "sharpe": result.get("sharpe_ratio", 0.0),
        "pnl":    result.get("pnl", 0.0),
        "turnover": result.get("turnover_ratio", 0.0),
        "tc004":  result.get("tc004_compliant", False),
        "tc005":  result.get("tc005_compliant", False),
        "fills":  result.get("total_orders", 0),
        "error":  result.get("error"),
    }


def _ok(r):
    return r["tc004"] and r["tc005"] and r["error"] is None


def _save(records, path, phase):
    existing = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing = [r for r in existing if r.get("phase") != phase]
    all_r = sorted(existing + records, key=lambda r: r.get("sharpe", 0), reverse=True)
    with open(path, "w") as f:
        json.dump(all_r, f, indent=2, default=str)


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _fmt(o):
    parts = []
    for k, v in list(o.items())[:4]:
        if k == "BLEND_COMBO":
            parts.append(f"blend={v}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


# ── Phase 1 ──────────────────────────────────────────────────────────────

def run_phase1(out):
    print("Phase 1 — Independent sweeps (train)")
    results, rid = [], 0

    base = _run({}, max_ticks=TRAIN_TICKS)
    print(f"  baseline sharpe={base.get('sharpe_ratio',0):.4f}")
    results.append(_rec({"__baseline__": True}, base, 1, rid)); rid += 1

    for param, vals in PHASE1_SWEEPS.items():
        for val in vals:
            ov = {param: val}
            r = _run(dict(ov), max_ticks=TRAIN_TICKS)
            rc = _rec(ov, r, 1, rid); rid += 1; results.append(rc)
            print(f"  {param}={val!r:>10} sharpe={rc['sharpe']:.4f} pnl=${rc['pnl']:>9,.0f} {'OK' if _ok(rc) else 'FAIL'}")

    _save(results, out, 1)
    print(f"Phase 1 done. {len(results)} runs.\n")
    return results


# ── Phase 2 ──────────────────────────────────────────────────────────────

def run_phase2(p1, out):
    print("Phase 2 — Cartesian (train)")
    grid = deepcopy(PHASE2_PARAMS)

    # inject phase1 winners
    for param in list(grid):
        best = [r for r in p1 if param in r["params"] and _ok(r)]
        if best:
            bv = max(best, key=lambda r: r["sharpe"])["params"][param]
            if bv not in grid[param]:
                grid[param].append(bv)

    names = list(grid)
    combos = list(itertools.product(*(grid[n] for n in names)))
    print(f"  {len(combos)} combos")

    results, rid = [], 1000
    for i, combo in enumerate(combos):
        ov = dict(zip(names, combo))
        r = _run(dict(ov), max_ticks=TRAIN_TICKS)
        rc = _rec(ov, r, 2, rid); rid += 1; results.append(rc)
        if (i+1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(combos)}] sharpe={rc['sharpe']:.4f} {_fmt(ov)}")

    _save(results, out, 2)
    print(f"Phase 2 done. {len(results)} runs.\n")
    return results


# ── Phase 3 ──────────────────────────────────────────────────────────────

def run_phase3(p1, p2, out, top_n=20):
    print("Phase 3 — Full validation")
    pool = sorted([r for r in p1 + p2 if _ok(r)], key=lambda r: r["sharpe"], reverse=True)[:top_n]
    if not pool:
        print("  No valid configs."); return []

    results, rid = [], 9000
    for i, prev in enumerate(pool):
        ov = {k: v for k, v in prev["params"].items() if not k.startswith("__")}
        train = _run(dict(ov), max_ticks=TRAIN_TICKS)
        full  = _run(dict(ov), max_ticks=None)
        ts, fs = train.get("sharpe_ratio", 0), full.get("sharpe_ratio", 0)
        ratio = fs / ts if ts > 1e-6 else 0

        rc = _rec(ov, full, 3, rid); rid += 1
        rc.update(train_sharpe=ts, full_sharpe=fs, overfit_ratio=ratio, overfit_ok=ratio >= 0.70)
        results.append(rc)
        tag = "OK" if _ok(rc) and rc["overfit_ok"] else "FAIL"
        print(f"  [{i+1}/{len(pool)}] train={ts:.4f} full={fs:.4f} ratio={ratio:.2f} pnl=${rc['pnl']:>9,.0f} [{tag}]")

    _save(results, out, 3)

    best = [r for r in results if _ok(r) and r.get("overfit_ok", True)]
    if best:
        w = max(best, key=lambda r: r["full_sharpe"])
        print(f"\n  BEST: sharpe={w['full_sharpe']:.4f} pnl=${w['pnl']:,.0f}")
        print(f"  {json.dumps(w['params'], indent=2)}")
    print()
    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=0, help="1/2/3 or 0=all")
    ap.add_argument("--out", default="grid_results.json")
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    p1 = _load(args.out)
    p2: list = []

    if args.phase in (0, 1): p1 = run_phase1(args.out)
    if args.phase in (0, 2):
        if not p1: p1 = [r for r in _load(args.out) if r.get("phase") == 1]
        p2 = run_phase2(p1, args.out)
    if args.phase in (0, 3):
        if not p1: p1 = [r for r in _load(args.out) if r.get("phase") == 1]
        if not p2: p2 = [r for r in _load(args.out) if r.get("phase") == 2]
        run_phase3(p1, p2, args.out, top_n=args.top_n)

    print(f"Results: {args.out}")


if __name__ == "__main__":
    main()
