#!/usr/bin/env python3
"""v3 Phase 4: program representations of the ORIGINAL -O0 IR.

Writes data/v3[/quick]/features_<rep>.csv for rep in hand (the 30 v2 features), funcprops
(LLVM FunctionPropertiesAnalysis, detailed), opcodes (instruction histogram), ir2vec
(LLVM in-tree IR2Vec, 75-d). One row per program; columns prefixed f_, fp_, op_, iv_.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import _bootstrap  # noqa: F401
from src import llvm_pipeline as lp
from src.features import REPRESENTATIONS, all_representations
from src.suites import load_manifest
from src.utils import version, DATA_DIR, load_v3_config, work_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    v3 = load_v3_config()
    tc = lp.Toolchain().require()
    out = DATA_DIR / version() / ("quick" if args.quick else "")
    benches = load_manifest(DATA_DIR / version() / "benchmarks.csv",
                            v3["quick"]["programs"] if args.quick else None)
    vocab = work_dir(v3) / v3["ir2vec_vocab"]
    wd = work_dir(v3) / version(v3) / ("quick" if args.quick else "full") / "features"
    wd.mkdir(parents=True, exist_ok=True)

    # v3.1: one row per (program, snapshot), from each snapshot's own baseline IR, which the
    # experiments stage (run first) saved; v3: one row per program, from the starting IR.
    # the snapshots the experiments stage actually ran (e.g. S0, S2 in quick mode)
    snaps = list(json.loads((out / "run_config.json").read_text())["contexts"]) if "snapshots" in v3 else [None]
    exp_root = work_dir(v3) / version(v3) / ("quick" if args.quick else "full") / "experiments"

    def job(item):
        b, ctx = item
        if ctx is None:
            ll = wd / f"{b.program_id}.ll"
            r = lp.program_ir(tc, b, ll, 120)
            if not r.ok:
                return b, ctx, None, f"compile failed: {r.stderr[:200]}"
        else:
            ll = exp_root / b.program_id / ctx / "baseline.ll"
            if not ll.exists():
                return b, ctx, None, f"missing {ll} (run v3_experiments.py first)"
        try:
            return b, ctx, all_representations(ll, tc, vocab), ""
        except Exception as e:  # noqa: BLE001
            return b, ctx, None, repr(e)

    reps = {k: [] for k in REPRESENTATIONS}
    failed = []
    with ThreadPoolExecutor(v3["workers"]) as pool:
        for b, ctx, f, err in pool.map(job, [(b, c) for b in benches for c in snaps]):
            if f is None:
                failed.append((b.program_id, err))
                continue
            for k in f:
                reps[k].append({"program_id": b.program_id, **({"context": ctx} if ctx else {}), **f[k]})
    reps = {k: v for k, v in reps.items() if v}
    for k, rows in reps.items():
        df = pd.DataFrame(rows).fillna(0.0).sort_values([c for c in ("program_id", "context") if c in rows[0]])
        df.to_csv(out / f"features_{k}.csv", index=False)
        print(f"  {k:<10} {len(df)} rows x {df.shape[1] - 1 - ('context' in df)} features")
    for pid, err in failed:
        print(f"FAILED {pid}: {err}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
