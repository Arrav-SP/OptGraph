#!/usr/bin/env python3
"""OptGraph end-to-end demonstration on one program and one pass pair.

    python run_demo.py
    python run_demo.py --program mix_01_matrix_mul --pass-a gvn --pass-b instcombine

Everything printed is measured in this run; nothing is cached or hard-coded.
The demo works in experiments/demo/ and outputs/demo/ and never touches the
full-experiment results.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from check_environment import check  # noqa: E402
from src import llvm_pipeline as lp  # noqa: E402
from src.experiments import pair_variant, run_program  # noqa: E402
from src.graph import draw_graph, program_graph  # noqa: E402
from src.interaction import FEATURE_PREFIX, build_dataset, classify  # noqa: E402
from src.metrics import METRICS  # noqa: E402
from src.utils import OUTPUT_DIR, Workspace, discover_benchmarks, load_config, load_passes  # noqa: E402

LINE = "=" * 62


def pct(v) -> str:
    return "undefined (baseline is 0)" if v is None or pd.isna(v) else f"{100 * v:+.2f}%"


def show_direction(ds: pd.DataFrame, first: str, second: str, metric: str) -> pd.Series:
    r = ds[(ds.pass_a == first) & (ds.pass_b == second) & (ds.metric == metric)].iloc[0]
    print(f"\n  Order {first} -> {second}        [{metric}]")
    print(f"    P                    : {r.m_base:.0f}")
    print(f"    P -> {first:<16}: {r.m_a:.0f}")
    print(f"    P -> {second:<16}: {r.m_b:.0f}")
    print(f"    P -> {first} -> {second}: {r.m_ab:.0f}")
    print(f"    {second} alone,  delta({second} | P)        : {pct(r.delta_b_alone)}")
    print(f"    {second} after {first},  delta({second} | {first}, P) : {pct(r.delta_b_after_a)}")
    print(f"    Interaction I({first}, {second}, P)         : {pct(r.interaction_score)}")
    print(f"    Classification                    : {r.interaction_type}")
    return r


def main() -> int:
    cfg = load_config()
    settings = cfg["settings"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", default="loop_04_invariant")
    ap.add_argument("--pass-a", default="gvn")
    ap.add_argument("--pass-b", default="instcombine")
    ap.add_argument("--metric", default=settings["primary_metric"], choices=METRICS)
    args = ap.parse_args()
    metric, thr = args.metric, settings["interaction_threshold"]
    if args.pass_a == args.pass_b:
        ap.error("--pass-a and --pass-b must differ")
    started = time.perf_counter()

    print(LINE + "\nOptGraph Demo\n" + LINE)

    print("\n[1] Environment")
    ok, env = check(write=False, quiet=True)
    print(f"    {env['distribution']} (WSL: {env['is_wsl']}), Python {env['python_version']}, "
          f"Clang {env['clang_version']}, LLVM {env['llvm_version']}")
    if not ok:
        print("REAL EXPERIMENTS PENDING - LLVM environment unavailable.", file=sys.stderr)
        return 1
    tc = lp.Toolchain().require()

    bench = discover_benchmarks([args.program])[0]
    quick = cfg["quick"].get("passes", [])
    names = [args.pass_a, args.pass_b] + [p for p in quick if p not in (args.pass_a, args.pass_b)]
    passes = load_passes(cfg, names)
    ws = Workspace("demo")

    print(f"\n[2] Program\n    {bench.path.relative_to(ROOT)}   (category: {bench.category})")
    print(f"\n[3] Compile to LLVM IR and run P, P->A, P->B, P->A->B, P->B->A\n"
          f"    passes in this demo: {', '.join(names)}")
    records, features = run_program(tc, bench, passes, ws, verify_outputs=True,
                                    timeout=settings.get("opt_timeout_s", 60))
    meas = pd.DataFrame(records)
    n_ok = int((meas.status == "SUCCESS").sum())
    print(f"    {n_ok}/{len(meas)} experiments succeeded; "
          f"every optimised IR was linked, run and gave the baseline's output: "
          f"{bool(meas.loc[meas.status == 'SUCCESS', 'output_matches'].all())}")
    print(f"    IR files: {(ws.experiments / bench.program_id).relative_to(ROOT)}/")
    if features is None:
        print(meas[meas.status != "SUCCESS"][["variant", "error"]].to_string(index=False))
        return 1

    print("\n[4] Program features (from the ORIGINAL IR only)")
    for label, key in (("Instructions", "instruction_count"), ("Basic blocks", "basic_blocks"),
                       ("Functions", "functions"), ("Loops", "loop_count"),
                       ("Max loop depth", "loop_depth"), ("Loads", "loads"), ("Stores", "stores"),
                       ("Branches", "branches"), ("Calls", "calls"), ("CFG edges", "cfg_edges")):
        print(f"    {label:<15}: {features[key]}")

    feats = pd.DataFrame([{"program_id": bench.program_id, "category": bench.category, **features}])
    ds = build_dataset(meas, feats, thr)
    need = {(args.pass_a, args.pass_b), (args.pass_b, args.pass_a)}
    have = set(zip(ds.pass_a, ds.pass_b)) if len(ds) else set()
    if not need <= have:
        print("\nThe requested pair did not complete:")
        print(meas[meas.status != "SUCCESS"][["variant", "error"]].to_string(index=False))
        return 1

    print(f"\n[5] Interaction     Pass A = {args.pass_a}     Pass B = {args.pass_b}")
    fwd = show_direction(ds, args.pass_a, args.pass_b, metric)
    rev = show_direction(ds, args.pass_b, args.pass_a, metric)
    diff = fwd.interaction_score - rev.interaction_score
    print(f"\n  Directionality: I(A,B) - I(B,A) = {pct(diff)}  "
          f"(final size {fwd.m_ab:.0f} via A->B vs {rev.m_ab:.0f} via B->A)")

    print(f"\n[6] All metrics for {args.pass_a} -> {args.pass_b}")
    table = ds[(ds.pass_a == args.pass_a) & (ds.pass_b == args.pass_b)][
        ["metric", "m_base", "m_a", "m_b", "m_ab", "interaction_score", "interaction_type"]].copy()
    table["interaction_score"] = table["interaction_score"].map(pct)
    print("    " + table.to_string(index=False).replace("\n", "\n    "))

    print("\n[7] Program-conditioned graph")
    g = program_graph(ds, bench.program_id, metric)
    out = OUTPUT_DIR / "demo" / f"demo_{bench.program_id}_graph.png"
    drawn = draw_graph(g, out, f"OptGraph demo - {bench.program_id} ({metric})", thr)
    print(f"    {g.number_of_nodes()} passes, {g.number_of_edges()} directed edges, "
          f"{drawn} non-weak edges drawn\n    {out.relative_to(ROOT)}")

    print("\n[8] ML prediction (original-program features + pass identities only)")
    model_path = OUTPUT_DIR / "models" / "interaction_model.joblib"
    if metric != settings["primary_metric"] or not model_path.exists():
        print("    skipped: no trained model for this metric "
              "(run scripts/run_experiments.py --full, then scripts/train_model.py)")
    else:
        import joblib
        from src.model import build_matrix
        bundle = joblib.load(model_path)
        role = next((k for k, v in bundle["split"].items() if bench.program_id in v), "not in dataset")
        rows = ds[ds.metric == metric]
        rows = rows[rows.pass_a.isin(bundle["pass_names"]) & rows.pass_b.isin(bundle["pass_names"])]
        missing = [c for c in bundle["feature_columns"] if c not in rows.columns]
        if missing:
            print(f"    skipped: model expects features this run did not produce: {missing}")
        else:
            X = build_matrix(rows, bundle["pass_names"])[bundle["columns"]]
            rows = rows.assign(predicted=bundle["model"].predict(X))
            print(f"    model: {bundle['model_name']}; this program was in the model's "
                  f"{role.upper()} programs" + ("" if role == "test" else
                                                 "  (not an unseen-program prediction)"))
            for a, b in ((args.pass_a, args.pass_b), (args.pass_b, args.pass_a)):
                r = rows[(rows.pass_a == a) & (rows.pass_b == b)].iloc[0]
                print(f"    I({a}, {b}, P): actual {pct(r.interaction_score)}   "
                      f"predicted {pct(r.predicted)}   "
                      f"[{r.interaction_type} / predicted {classify(r.predicted, thr)}]")
            mae = (rows.predicted - rows.interaction_score).abs().mean()
            print(f"    mean absolute error over this program's {len(rows)} demo pairs: {100 * mae:.2f} points")

    assert not [c for c in ds.columns if c.startswith(FEATURE_PREFIX) and ds[c].isna().any()]
    print(f"\n{LINE}\nDemo finished in {time.perf_counter() - started:.1f}s - all numbers above are "
          f"real measurements from this run.\n{LINE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
