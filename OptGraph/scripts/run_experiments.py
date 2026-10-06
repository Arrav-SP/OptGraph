#!/usr/bin/env python3
"""Run the controlled pass experiments and build the interaction dataset.

  python scripts/run_experiments.py --quick     # 6 programs x 5 passes  -> data/quick/
  python scripts/run_experiments.py --full      # all programs x all passes -> data/

For each program P and ordered pass pair (A, B): P, P->A, P->B, P->A->B, P->B->A.
"""
import argparse
import os
import sys
import time

import _bootstrap  # noqa: F401
from check_environment import check
from src import llvm_pipeline as lp
from src.experiments import run_all
from src.interaction import build_dataset
from src.utils import Workspace, discover_benchmarks, load_config, load_passes, now_iso, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="small subset from config (default)")
    mode.add_argument("--full", action="store_true", help="all configured programs and passes")
    ap.add_argument("--programs", nargs="+", metavar="ID", help="program ids to run")
    ap.add_argument("--program", metavar="ID", help="a single program id")
    ap.add_argument("--passes", nargs="+", metavar="PASS", help="pass names to use")
    ap.add_argument("--pass-a", metavar="PASS", help="with --pass-b: run just this pair")
    ap.add_argument("--pass-b", metavar="PASS")
    ap.add_argument("--tag", help="output workspace name (default: full / quick / custom)")
    ap.add_argument("--verify-outputs", action="store_true",
                    help="also build and run every IR and require identical program output")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    args = ap.parse_args()

    ok, env = check(write=True, quiet=True)
    if not ok:
        print("REAL EXPERIMENTS PENDING - LLVM environment unavailable.", file=sys.stderr)
        return 1
    tc = lp.Toolchain().require()
    cfg = load_config()
    settings = cfg["settings"]

    prog_names = args.programs or ([args.program] if args.program else None)
    pass_names = args.passes
    if args.pass_a or args.pass_b:
        if not (args.pass_a and args.pass_b):
            ap.error("--pass-a and --pass-b must be given together")
        pass_names = [args.pass_a, args.pass_b]
    custom = bool(prog_names or pass_names)
    if not args.full:
        prog_names = prog_names or cfg["quick"].get("programs")
        pass_names = pass_names or cfg["quick"].get("passes")
    tag = args.tag or ("full" if args.full and not custom else "custom" if custom else "quick")

    benches = discover_benchmarks(prog_names)
    passes = load_passes(cfg, pass_names)
    ws = Workspace(tag)
    per_program = 1 + len(passes) + len(passes) * (len(passes) - 1)
    print(f"OptGraph experiments [{tag}]  LLVM {env['llvm_version']}")
    print(f"  programs: {len(benches)}   passes: {len(passes)}   "
          f"experiments: {len(benches)} x {per_program} = {len(benches) * per_program}")

    start = time.perf_counter()
    meas, feats = run_all(tc, benches, passes, ws, args.verify_outputs, args.workers,
                          settings.get("opt_timeout_s", 60))
    elapsed = time.perf_counter() - start

    threshold = settings.get("interaction_threshold", 0.005)
    dataset = build_dataset(meas, feats, threshold)
    ws.dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(ws.dataset, index=False)

    n_ok = int((meas["status"] == "SUCCESS").sum())
    summary = {
        "tag": tag, "timestamp": now_iso(), "elapsed_s": round(elapsed, 1),
        "programs": [b.program_id for b in benches],
        "passes": [p.__dict__ for p in passes],
        "settings": settings, "verify_outputs": args.verify_outputs,
        "experiments_total": len(meas), "experiments_success": n_ok,
        "experiments_failed": len(meas) - n_ok,
        "interaction_rows": len(dataset),
        "undefined_rows": int((dataset["interaction_type"] == "UNDEFINED").sum()) if len(dataset) else 0,
        "environment": env,
    }
    write_json(ws.run_config, summary)

    print(f"\nExperiments: {len(meas)}  success: {n_ok}  failed: {len(meas) - n_ok}  ({elapsed:.1f}s)")
    if n_ok < len(meas):
        failed = meas[meas["status"] != "SUCCESS"]
        print(failed[["program_id", "variant", "error"]].head(20).to_string(index=False))
    print(f"Interaction rows: {len(dataset)}  (undefined, zero baseline: {summary['undefined_rows']})")
    for path in (ws.measurements, ws.experiment_log, ws.features, ws.dataset, ws.run_config):
        print("  wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
