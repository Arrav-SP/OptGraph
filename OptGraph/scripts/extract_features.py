#!/usr/bin/env python3
"""Compile every benchmark to baseline LLVM IR and write data/program_features.csv.

Features come from the ORIGINAL IR only (before any pass under study).
"""
import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src import llvm_pipeline as lp
from src.features import extract_features
from src.utils import Workspace, discover_benchmarks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--programs", nargs="+", metavar="ID")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    tc = lp.Toolchain().require()
    ws = Workspace(args.tag)
    rows = []
    with tempfile.TemporaryDirectory(prefix="optgraph_feat_") as d:
        for b in discover_benchmarks(args.programs):
            ll = Path(d) / f"{b.program_id}.ll"
            r = lp.compile_to_ir(tc, b.path, ll)
            if not r.ok:
                print(f"FAILED {b.program_id}: {r.error}\n{r.stderr}", file=sys.stderr)
                continue
            rows.append({"program_id": b.program_id, "category": b.category,
                         **extract_features(ll, tc)})
    df = pd.DataFrame(rows)
    ws.features.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ws.features, index=False)
    cols = ["program_id", "instruction_count", "basic_blocks", "functions", "branches",
            "loads", "stores", "calls", "loop_count", "loop_depth"]
    print(df[cols].to_string(index=False))
    print(f"\nwrote {ws.features}  ({len(df)} programs, {df.shape[1] - 3} features)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
