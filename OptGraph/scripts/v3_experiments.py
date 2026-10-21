#!/usr/bin/env python3
"""v3 Phases 2-3: pair experiments in every context, then the interaction dataset.

  python scripts/v3_experiments.py --quick      # config quick subset -> data/v3/quick/
  python scripts/v3_experiments.py              # all kept programs x 22 passes x 3 contexts -> data/v3/

Finished programs are skipped when nothing about them changed (resumable).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import llvm_pipeline as lp
from src.experiments_v3 import RECORD_COLUMNS, V3_METRICS, run_program_v3
from src.interaction import compute_interactions
from src.suites import load_manifest
from src.utils import DATA_DIR, load_config, load_passes, load_v3_config, now_iso, work_dir, write_json


def dynamic_subset(manifest: pd.DataFrame, size: int, max_rt: float, seed: int) -> list[str]:
    """Seeded subset for pair-level cachegrind, stratified by suite (round-robin)."""
    m = manifest[manifest["runtime_o0_s"].astype(float) <= max_rt]
    if size is None:
        return sorted(m["program_id"])
    rng = np.random.default_rng(seed)
    pools = {s: list(rng.permutation(sorted(g["program_id"]))) for s, g in m.groupby("suite")}
    out = []
    while len(out) < size and any(pools.values()):
        for s in sorted(pools):
            if pools[s] and len(out) < size:
                out.append(pools[s].pop())
    return sorted(out)


def data_dir(quick: bool):
    return DATA_DIR / "v3" / ("quick" if quick else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--programs", nargs="+", metavar="ID")
    ap.add_argument("--passes", nargs="+", metavar="PASS")
    ap.add_argument("--contexts", nargs="+", metavar="C")
    ap.add_argument("--dynamic", nargs="*", metavar="ID",
                    help="programs with pair-level dynamic counts (default: configured subset; "
                         "give no ids for none)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    cfg, v3 = load_config(), load_v3_config()
    seed = v3["seed"]
    q = v3["quick"] if args.quick else {}
    out = data_dir(args.quick)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "v3" / "benchmarks.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest = manifest[manifest["kept"].astype(str) == "True"]

    programs = args.programs or q.get("programs") or sorted(manifest["program_id"])
    benches = load_manifest(manifest_path, programs)
    passes = load_passes(cfg, args.passes or q.get("passes"), v3=True)
    contexts = {c: v3["contexts"][c] for c in (args.contexts or q.get("contexts") or v3["contexts"])}
    if args.dynamic is not None:
        dyn = set(args.dynamic)
    elif args.quick:
        dyn = set(programs)
    else:
        d = v3["dynamic_pairs"]
        dyn = set(dynamic_subset(manifest, d["subset_size"], d["max_runtime_s"], seed))
    expected = dict(zip(manifest["program_id"], manifest["output_sha256"]))
    workers = args.workers or v3["workers"]
    root = work_dir(v3) / "v3" / ("quick" if args.quick else "full") / "experiments"
    tc = lp.Toolchain().require()

    per = len(contexts) * (1 + len(passes) + len(passes) * (len(passes) - 1))
    print(f"v3 experiments: {len(benches)} programs x {len(passes)} passes x {len(contexts)} contexts "
          f"= {len(benches) * per} opt runs; dynamic counts for {len(dyn)} programs; {workers} workers")
    start = time.perf_counter()

    def job(b):
        t = time.perf_counter()
        try:
            recs, feats = run_program_v3(tc, b, passes, contexts, root / b.program_id,
                                         expected[b.program_id], b.program_id in dyn)
        except Exception as e:  # noqa: BLE001 - one broken program must not stop the run
            recs, feats = [{"program_id": b.program_id, "suite": b.suite, "category": b.category,
                            "variant": "baseline", "kind": "baseline", "status": "FAILED",
                            "error": f"unexpected error: {e!r}"}], None
        return b, recs, feats, time.perf_counter() - t

    # Longest programs first so they do not finish last on their own.
    rt = dict(zip(manifest["program_id"], manifest["runtime_o0_s"].astype(float)))
    order = sorted(benches, key=lambda b: -(rt[b.program_id] * (5 if b.program_id in dyn else 1)))
    records, feats = [], []
    with ThreadPoolExecutor(workers) as pool:
        for b, recs, f, secs in pool.map(job, order):
            records.extend(recs)
            ok = sum(r.get("status") == "SUCCESS" for r in recs)
            print(f"  {b.program_id:<34} {ok:>5}/{len(recs)} ok  {secs:7.1f}s"
                  f"{'  (dyn)' if b.program_id in dyn else ''}", flush=True)
            if f is not None:
                feats.append({"program_id": b.program_id, "category": b.category, **f})
    elapsed = time.perf_counter() - start

    # Large tables are gzip-compressed (pandas reads them directly). The interaction
    # dataset holds no feature columns; they are joined by program_id from features_<rep>.csv.
    log = pd.DataFrame(records).reindex(columns=RECORD_COLUMNS)
    log[log["status"] != "SUCCESS"].to_csv(out / "experiment_failures.csv", index=False)
    meas = log.drop(columns=["stderr"])
    meas.to_csv(out / "measurements.csv.gz", index=False)
    features = pd.DataFrame(feats).sort_values("program_id")
    features.to_csv(out / "program_features.csv", index=False)
    thr = cfg["settings"]["interaction_threshold"]
    meas = meas.astype({m: float for m in V3_METRICS})
    ds = compute_interactions(meas, thr, V3_METRICS)
    ds.to_csv(out / "interaction_dataset.csv.gz", index=False)

    n_ok = int((log["status"] == "SUCCESS").sum())
    env = lp.collect_environment(tc)
    write_json(out / "environment.json", {**env, "seed": seed})
    write_json(out / "run_config.json", {
        "timestamp": now_iso(), "elapsed_s": round(elapsed, 1), "programs": programs,
        "passes": [p.__dict__ for p in passes], "contexts": contexts,
        "dynamic_programs": sorted(dyn), "experiments_total": len(log),
        "experiments_success": n_ok, "experiments_failed": len(log) - n_ok,
        "interaction_rows": len(ds), "threshold": thr, "seed": seed,
        "work_dir": str(root), "workers": workers, "cpu_count": os.cpu_count()})
    print(f"\nexperiments: {len(log)}  success: {n_ok}  failed: {len(log) - n_ok}  ({elapsed:.0f}s)")
    if n_ok < len(log):
        bad = log[log["status"] != "SUCCESS"]
        print(bad.groupby("error").size().sort_values(ascending=False).head(15).to_string())
    print(f"interaction rows: {len(ds)}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
