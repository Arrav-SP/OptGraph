#!/usr/bin/env python3
"""v3 Phase 4: program representations of the ORIGINAL -O0 IR.

Writes data/v3[/quick]/features_<rep>.csv for rep in hand (the 30 v2 features), funcprops
(LLVM FunctionPropertiesAnalysis, detailed), opcodes (instruction histogram), ir2vec
(LLVM in-tree IR2Vec, 75-d). One row per program; columns prefixed f_, fp_, op_, iv_.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import _bootstrap  # noqa: F401
from src import llvm_pipeline as lp
from src.features import extract_features, func_properties, ir2vec_embedding, opcode_histogram
from src.suites import load_manifest
from src.utils import DATA_DIR, load_v3_config, work_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    v3 = load_v3_config()
    tc = lp.Toolchain().require()
    out = DATA_DIR / "v3" / ("quick" if args.quick else "")
    benches = load_manifest(DATA_DIR / "v3" / "benchmarks.csv",
                            v3["quick"]["programs"] if args.quick else None)
    vocab = work_dir(v3) / v3["ir2vec_vocab"]
    wd = work_dir(v3) / "v3" / ("quick" if args.quick else "full") / "features"
    wd.mkdir(parents=True, exist_ok=True)

    def job(b):
        ll = wd / f"{b.program_id}.ll"
        r = lp.compile_to_ir(tc, b.path, ll, 120, b.cflags)
        if not r.ok:
            return b, None, f"compile failed: {r.stderr[:200]}"
        try:
            hand = {"f_" + k: v for k, v in extract_features(ll, tc).items() if k != "loop_method"}
            return b, {"hand": hand, "funcprops": func_properties(ll, tc),
                       "opcodes": opcode_histogram(ll), "ir2vec": ir2vec_embedding(ll, tc, vocab)}, ""
        except Exception as e:  # noqa: BLE001
            return b, None, repr(e)

    reps = {k: [] for k in ("hand", "funcprops", "opcodes", "ir2vec")}
    failed = []
    with ThreadPoolExecutor(v3["workers"]) as pool:
        for b, f, err in pool.map(job, benches):
            if f is None:
                failed.append((b.program_id, err))
                continue
            for k in reps:
                reps[k].append({"program_id": b.program_id, **f[k]})
    for k, rows in reps.items():
        df = pd.DataFrame(rows).fillna(0.0).sort_values("program_id")
        df.to_csv(out / f"features_{k}.csv", index=False)
        print(f"  {k:<10} {len(df)} programs x {df.shape[1] - 1} features")
    for pid, err in failed:
        print(f"FAILED {pid}: {err}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
