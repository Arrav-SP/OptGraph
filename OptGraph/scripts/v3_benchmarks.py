#!/usr/bin/env python3
"""v3 Phase 1: screen benchmark candidates and write data/v3/benchmarks.csv."""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src import llvm_pipeline as lp
from src.suites import candidates, screen
from src.utils import DATA_DIR, ROOT, load_v3_config, write_json


def signature(cfg: dict) -> str:
    """What the screening depends on: the suite configuration, the screening code and the
    compile/sanitizer flags."""
    parts = [json.dumps(cfg["benchmarks"], sort_keys=True), (ROOT / "src" / "suites.py").read_text(),
             " ".join(lp.BACKEND_FLAGS), " ".join(lp.SANITIZER_FLAGS)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--programs", nargs="+", metavar="ID", help="screen only these candidates")
    ap.add_argument("--out", default=str(DATA_DIR / "v3" / "benchmarks.csv"))
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel screens (runtimes are read under this load)")
    ap.add_argument("--if-stale", action="store_true",
                    help="do nothing if the manifest was built with the current signature")
    ap.add_argument("--stamp-only", action="store_true",
                    help="record the current signature for the existing manifest without screening")
    args = ap.parse_args()
    cfg = load_v3_config()
    sig_path = DATA_DIR / "v3" / "benchmarks_signature.json"
    if args.stamp_only:
        write_json(sig_path, {"signature": signature(cfg), "manifest": args.out})
        return 0
    if args.if_stale and sig_path.exists() and json.loads(sig_path.read_text())["signature"] == signature(cfg) \
            and Path(args.out).exists():
        print(f"{args.out} is up to date")
        return 0
    tc = lp.Toolchain().require()
    cands = candidates(tc, cfg)
    if args.programs:
        cands = [c for c in cands if c.program_id in set(args.programs)]
    print(f"screening {len(cands)} candidates with {args.workers} workers ...")
    start = time.perf_counter()

    def job(c):
        try:
            return screen(tc, c, cfg)
        except Exception as e:  # noqa: BLE001 - one broken candidate must not stop the rest
            return {"program_id": c.program_id, "suite": c.suite, "category": c.category,
                    "source": str(c.path), "kept": False, "drop_reason": f"unexpected error: {e!r}"}

    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for r in pool.map(job, cands):
            rows.append(r)
            print(f"  {r['program_id']:<34} {'KEPT' if r['kept'] else 'dropped':<8} "
                  f"{r.get('dataset') or '':<7} {r.get('runtime_o0_s')!s:>8}s  {r.get('drop_reason', '')[:70]}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    if not args.programs:
        write_json(sig_path, {"signature": signature(cfg), "manifest": args.out})
    kept = df[df["kept"]]
    print(f"\n{len(kept)}/{len(df)} kept ({time.perf_counter() - start:.0f}s); by suite:")
    print(df.groupby("suite")["kept"].agg(["sum", "count"]).to_string())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
