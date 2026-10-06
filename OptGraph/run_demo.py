#!/usr/bin/env python3
"""OptGraph end-to-end demonstration on one program and one pass pair.

    python run_demo.py                                  # default benchmark and pair
    python run_demo.py --list                           # show available programs and passes
    python run_demo.py --program mix_01_matrix_mul --pass-a licm --pass-b instcombine
    python run_demo.py --file path/to/your_program.c    # any C/C++ file of your own
    python run_demo.py --file my.c --pass-a mem2reg --pass-b gvn

Everything printed is measured in this run; nothing is cached or hard-coded.
The demo works in experiments/demo/ and outputs/demo/ and never touches the
full-experiment results.

A file given with --file must be a complete program (it needs main), must not
wait for keyboard input, and should finish within 10 seconds.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from check_environment import check  # noqa: E402
from src import llvm_pipeline as lp  # noqa: E402
from src.experiments import run_program  # noqa: E402
from src.graph import draw_graph, program_graph  # noqa: E402
from src.interaction import build_dataset, classify  # noqa: E402
from src.metrics import METRICS  # noqa: E402
from src.utils import (OUTPUT_DIR, SOURCE_SUFFIXES, Benchmark, Workspace,  # noqa: E402
                       discover_benchmarks, load_config, load_passes)

WIDTH = 78
COLOR = False
CODES = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1", "dim": "2"}
TYPE_COLOR = {"SYNERGY": "green", "ANTAGONISM": "red", "WEAK": "yellow", "UNDEFINED": "dim"}


# ---- presentation helpers -----------------------------------------------------------

def paint(text: str, color: str | None) -> str:
    return f"\033[{CODES[color]}m{text}\033[0m" if COLOR and color else text


def banner(title: str, subtitle: str = "") -> None:
    print(paint("╔" + "═" * (WIDTH - 2) + "╗", "cyan"))
    for line in (title, subtitle):
        if line:
            print(paint("║", "cyan") + paint(line.center(WIDTH - 2), "bold") + paint("║", "cyan"))
    print(paint("╚" + "═" * (WIDTH - 2) + "╝", "cyan"))


def step(n: int, title: str) -> None:
    head = f" STEP {n}  {title} "
    print("\n" + paint(head + "─" * max(0, WIDTH - len(head)), "cyan"))


def table(headers: list[str], rows: list[list], right: set[int] = frozenset(), indent: int = 2) -> None:
    """Print a boxed table. A cell is a string or a (string, colour) pair."""
    plain = [[c[0] if isinstance(c, tuple) else str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in plain)) for i, h in enumerate(headers)]
    pad = " " * indent

    def line(left, mid, right_):
        return pad + left + mid.join("─" * (w + 2) for w in widths) + right_

    def fmt(text, i, color=None):
        cell = text.rjust(widths[i]) if i in right else text.ljust(widths[i])
        return " " + paint(cell, color) + " "

    print(line("┌", "┬", "┐"))
    print(pad + "│" + "│".join(fmt(h, i, "bold") for i, h in enumerate(headers)) + "│")
    print(line("├", "┼", "┤"))
    for r, p in zip(rows, plain):
        cells = [fmt(p[i], i, c[1] if isinstance(c, tuple) else None) for i, c in enumerate(r)]
        print(pad + "│" + "│".join(cells) + "│")
    print(line("└", "┴", "┘"))


def pct(v, signed: bool = True) -> str:
    if v is None or pd.isna(v):
        return "undefined"
    return f"{100 * v:+.2f}%" if signed else f"{100 * v:.2f}%"


def typed(kind: str) -> tuple[str, str]:
    return kind, TYPE_COLOR.get(kind)


def in_words(kind: str, first: str, second: str) -> str:
    return {
        "SYNERGY": f"Running {first} first makes {second} MORE effective.",
        "ANTAGONISM": f"Running {first} first leaves LESS for {second} to do.",
        "WEAK": f"Running {first} first barely changes what {second} does.",
    }.get(kind, "The baseline count is 0, so a percentage cannot be computed.")


def show_path(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ---- demo parts -----------------------------------------------------------------------

def pick_benchmark(args, ap) -> Benchmark:
    if not args.file:
        try:
            return discover_benchmarks([args.program])[0]
        except ValueError:
            ap.error(f"unknown program '{args.program}'. Use --list to see the built-in "
                     f"programs, or --file PATH for your own source file.")
    path = Path(args.file).expanduser().resolve()
    if not path.is_file():
        ap.error(f"file not found: {path}")
    if path.suffix not in SOURCE_SUFFIXES:
        ap.error(f"{path.name}: expected a C/C++ source file ({', '.join(SOURCE_SUFFIXES)})")
    return Benchmark(re.sub(r"[^A-Za-z0-9_-]", "_", path.stem) or "custom", "custom", path)


def list_choices(cfg) -> None:
    banner("OptGraph — available programs and passes")
    by_cat: dict[str, list[str]] = {}
    for b in discover_benchmarks():
        by_cat.setdefault(b.category, []).append(b.program_id)
    print("\nBuilt-in programs (use with --program):")
    for cat, ids in by_cat.items():
        print(f"  {cat + ':':<14}{', '.join(ids)}")
    print("\nPasses (use with --pass-a / --pass-b):")
    table(["Pass", "Kind", "opt pipeline"], [[p.name, p.kind, p.pipeline] for p in load_passes(cfg)])
    print("\nYour own program:  python run_demo.py --file path/to/program.c")


def show_source(bench: Benchmark, limit: int) -> None:
    lines = bench.path.read_text(errors="replace").splitlines()
    for i, text in enumerate(lines[:limit], 1):
        print(paint(f"  {i:>3} │ ", "dim") + text)
    if len(lines) > limit:
        print(paint(f"      │ ... {len(lines) - limit} more lines", "dim"))


def show_direction(ds: pd.DataFrame, first: str, second: str, metric: str) -> pd.Series:
    r = ds[(ds.pass_a == first) & (ds.pass_b == second) & (ds.metric == metric)].iloc[0]
    p, a, b, ab = (int(r.m_base), int(r.m_a), int(r.m_b), int(r.m_ab))
    print(f"\n  {paint(f'Order: {first} → {second}', 'bold')}    "
          f"(does running {first} first change what {second} achieves?)")
    table(["Version of the program", metric, "removed vs P"], [
        ["P  (original)", str(p), ""],
        [f"P → {first}", str(a), pct((p - a) / p) if p else "undefined"],
        [f"P → {second}", str(b), pct((p - b) / p) if p else "undefined"],
        [f"P → {first} → {second}", str(ab), pct((p - ab) / p) if p else "undefined"],
    ], right={1, 2}, indent=4)
    if p == 0:
        print(f"    Baseline {metric} is 0: the normalised interaction is UNDEFINED.")
        return r
    name = f"Δ({second} | P)"
    after = f"Δ({second} | {first}, P)"
    w = max(len(name), len(after))
    print(f"    {name:<{w}} = ({p} − {b}) / {p}  = {pct(r.delta_b_alone)}    ← {second} alone")
    print(f"    {after:<{w}} = ({a} − {ab}) / {p}  = {pct(r.delta_b_after_a)}    ← {second} after {first}")
    print(f"    {'I(' + first + ', ' + second + ', P)':<{w}} = {pct(r.delta_b_after_a)} − ({pct(r.delta_b_alone)})"
          f"  = {paint(pct(r.interaction_score), 'bold')}")
    print(f"    Result: {paint(r.interaction_type, TYPE_COLOR.get(r.interaction_type))}"
          f"  —  {in_words(r.interaction_type, first, second)}")
    return r


def main() -> int:
    global COLOR
    cfg = load_config()
    settings = cfg["settings"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--program", default="loop_04_invariant", help="id of a built-in benchmark")
    src.add_argument("--file", metavar="PATH", help="your own C/C++ source file")
    ap.add_argument("--pass-a", default="gvn")
    ap.add_argument("--pass-b", default="instcombine")
    ap.add_argument("--metric", default=settings["primary_metric"], choices=METRICS)
    ap.add_argument("--list", action="store_true", help="list programs and passes, then exit")
    ap.add_argument("--source-lines", type=int, default=30, metavar="N",
                    help="lines of the source file to print (0 = none)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    COLOR = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    if args.list:
        list_choices(cfg)
        return 0
    metric, thr = args.metric, settings["interaction_threshold"]
    if args.pass_a == args.pass_b:
        ap.error("--pass-a and --pass-b must differ")
    known = [p.name for p in load_passes(cfg)]
    for name in (args.pass_a, args.pass_b):
        if name not in known:
            ap.error(f"unknown pass '{name}'. Available: {', '.join(known)}")
    bench = pick_benchmark(args, ap)
    A, B = args.pass_a, args.pass_b
    started = time.perf_counter()

    banner("OptGraph — pass interaction demo",
           "Does running pass A first change what pass B achieves?")

    step(1, "Environment")
    ok, env = check(write=False, quiet=True)
    table(["Component", "Version"], [
        ["Operating system", f"{env['distribution']}" + ("  (WSL)" if env["is_wsl"] else "")],
        ["Python", str(env["python_version"])],
        ["Clang", str(env["clang_version"] or "NOT FOUND")],
        ["LLVM / opt", str(env["llvm_version"] or "NOT FOUND")],
        ["C → IR → pass smoke test", ("passed", "green") if ok else ("FAILED", "red")],
    ])
    if not ok:
        print("REAL EXPERIMENTS PENDING - LLVM environment unavailable.", file=sys.stderr)
        return 1
    tc = lp.Toolchain().require()

    step(2, "Input program")
    origin = "your own file" if args.file else f"built-in benchmark, category: {bench.category}"
    print(f"  {paint(show_path(bench.path), 'bold')}   ({origin})")
    if args.source_lines > 0:
        show_source(bench, args.source_lines)

    quick = cfg["quick"].get("passes", [])
    names = [A, B] + [p for p in quick if p not in (A, B)]
    passes = load_passes(cfg, names)
    ws = Workspace("demo")

    step(3, "Compile to LLVM IR and run the experiments")
    print(f"  Pass A = {paint(A, 'bold')}    Pass B = {paint(B, 'bold')}"
          f"    (also run for the graph: {', '.join(names[2:]) or 'none'})")
    print("  For every ordered pair:  P,  P→A,  P→B,  P→A→B,  P→B→A")
    records, features = run_program(tc, bench, passes, ws, verify_outputs=True,
                                    timeout=settings.get("opt_timeout_s", 60))
    meas = pd.DataFrame(records)
    good = meas[meas.status == "SUCCESS"]
    if features is None:
        base = records[0]
        print(paint("\n  The program could not be compiled to LLVM IR.", "red"))
        print("  " + (base["stderr"].strip() or base["error"]).replace("\n", "\n  "))
        return 1
    first = meas.loc[meas.variant == "baseline", "output_matches"].iloc[0]
    baseline_ran = bool(pd.notna(first) and first)
    if baseline_ran:
        same = bool(good.output_matches.fillna(False).astype(bool).all())
        check_cell = ("identical to the original for every version", "green") if same else \
            ("some versions differ (marked FAILED)", "red")
    else:
        check_cell = ("skipped: original did not finish in 10 s (waiting for input?)", "yellow")
    table(["Check", "Result"], [
        ["opt experiments run", str(len(meas))],
        ["succeeded / failed", f"{len(good)} / {len(meas) - len(good)}"],
        ["program output check", check_cell],
        ["IR files saved in", show_path(ws.experiments / bench.program_id) + "/"],
    ])
    if len(good) < len(meas):
        print("  Failed experiments:")
        print("    " + meas[meas.status != "SUCCESS"][["variant", "error"]].head(8)
              .to_string(index=False).replace("\n", "\n    "))

    step(4, "Structural features of the ORIGINAL program")
    pairs = [("Instructions", "instruction_count"), ("Basic blocks", "basic_blocks"),
             ("Functions", "functions"), ("Loops", "loop_count"), ("Max loop depth", "loop_depth"),
             ("Loads", "loads"), ("Stores", "stores"), ("Branches", "branches"),
             ("Calls", "calls"), ("CFG edges", "cfg_edges")]
    half = (len(pairs) + 1) // 2
    table(["Feature", "Value", "Feature ", "Value "],
          [[pairs[i][0], str(features[pairs[i][1]]),
            pairs[i + half][0], str(features[pairs[i + half][1]])] for i in range(half)],
          right={1, 3})
    print(f"  ({len([k for k in features if k != 'loop_method'])} features in total; "
          f"none of them uses optimised IR)")

    feats = pd.DataFrame([{"program_id": bench.program_id, "category": bench.category, **features}])
    ds = build_dataset(meas, feats, thr)
    need = {(A, B), (B, A)}
    have = set(zip(ds.pass_a, ds.pass_b)) if len(ds) else set()
    if not need <= have:
        print(paint("\n  The requested pair did not complete, so no interaction can be computed.", "red"))
        return 1

    step(5, f"Interaction between {A} and {B}")
    print("  I(A, B, P) = Δ(B | A, P) − Δ(B | P)        Δ = (before − after) / original")
    print(f"  I > +{thr:g}: SYNERGY     I < −{thr:g}: ANTAGONISM     otherwise: WEAK")
    fwd = show_direction(ds, A, B, metric)
    rev = show_direction(ds, B, A, metric)

    step(6, "Does the order matter?")
    table(["Order", f"final {metric}", "interaction I", "type"], [
        [f"{A} → {B}", str(int(fwd.m_ab)), pct(fwd.interaction_score), typed(fwd.interaction_type)],
        [f"{B} → {A}", str(int(rev.m_ab)), pct(rev.interaction_score), typed(rev.interaction_type)],
    ], right={1, 2})
    if fwd.m_ab == rev.m_ab:
        print(f"  Both orders end at {int(fwd.m_ab)}: for this program the order does not matter.")
    else:
        best = f"{A} → {B}" if fwd.m_ab < rev.m_ab else f"{B} → {A}"
        print(f"  The orders end at different sizes ({int(fwd.m_ab)} vs {int(rev.m_ab)}); "
              f"{paint(best, 'bold')} gives the smaller program.")
        print("  This is why the graph needs two separate directed edges per pair.")

    step(7, f"The same pair ({A} → {B}) on every metric")
    rows = ds[(ds.pass_a == A) & (ds.pass_b == B)]
    table(["metric", "P", f"P→{A}", f"P→{B}", f"P→{A}→{B}", "interaction I", "type"],
          [[r.metric, f"{r.m_base:.0f}", f"{r.m_a:.0f}", f"{r.m_b:.0f}", f"{r.m_ab:.0f}",
            pct(r.interaction_score), typed(r.interaction_type)] for r in rows.itertuples()],
          right={1, 2, 3, 4, 5})
    if (rows.interaction_type == "UNDEFINED").any():
        print("  'undefined': the original count is 0, so the ratio is not computed (no fake value).")

    step(8, "Program-conditioned interaction graph")
    g = program_graph(ds, bench.program_id, metric)
    out = OUTPUT_DIR / "demo" / f"demo_{bench.program_id}_graph.png"
    drawn = draw_graph(g, out, f"OptGraph demo - {bench.program_id} ({metric})", thr)
    edges = sorted(g.edges(data="weight"), key=lambda e: abs(e[2]), reverse=True)[:6]
    table(["edge A → B", "weight I", "type"],
          [[f"{u} → {v}", pct(w), typed(classify(w, thr))] for u, v, w in edges], right={1})
    print(f"  {g.number_of_nodes()} passes, {g.number_of_edges()} directed edges, {drawn} non-weak "
          f"(strongest shown above)\n  Picture: {paint(show_path(out), 'bold')}")

    step(9, "ML prediction from the original program's features + pass names")
    model_path = OUTPUT_DIR / "models" / "interaction_model.joblib"
    if metric != settings["primary_metric"] or not model_path.exists():
        print("  Skipped: no trained model for this metric.")
        print("  (run scripts/run_experiments.py --full, then scripts/train_model.py)")
    else:
        import joblib
        from src.model import build_matrix
        bundle = joblib.load(model_path)
        role = None if args.file else next(
            (k for k, v in bundle["split"].items() if bench.program_id in v), None)
        rows = ds[ds.metric == metric]
        rows = rows[rows.pass_a.isin(bundle["pass_names"]) & rows.pass_b.isin(bundle["pass_names"])]
        X = build_matrix(rows, bundle["pass_names"])[bundle["columns"]]
        rows = rows.assign(predicted=bundle["model"].predict(X))
        if role is None:
            seen = paint("NOT in the model's data: a genuine unseen-program prediction", "green")
        elif role == "test":
            seen = paint("in the held-out TEST programs: unseen during training", "green")
        else:
            seen = paint(f"in the model's {role.upper()} programs, so NOT an unseen-program test", "yellow")
        print(f"  Model: {bundle['model_name']}.  This program is {seen}.")
        out_rows = []
        for a, b in ((A, B), (B, A)):
            r = rows[(rows.pass_a == a) & (rows.pass_b == b)].iloc[0]
            guess = classify(r.predicted, thr)
            out_rows.append([f"I({a}, {b}, P)", pct(r.interaction_score), pct(r.predicted),
                             typed(r.interaction_type), typed(guess),
                             ("yes", "green") if guess == r.interaction_type else ("no", "red")])
        table(["quantity", "actual", "predicted", "actual type", "predicted type", "match"],
              out_rows, right={1, 2})
        mae = (rows.predicted - rows.interaction_score).abs().mean()
        hits = (rows.interaction_type == rows.predicted.map(lambda v: classify(v, thr))).mean()
        print(f"  Over all {len(rows)} ordered pairs of this demo: mean absolute error "
              f"{100 * mae:.2f} percentage points, class agreement {100 * hits:.0f}%.")

    print()
    banner("Summary")
    w = max(len(f"{A} → {B}"), len("Program"))
    print(f"  {'Program':<{w}} : {bench.program_id}  ({features['instruction_count']} instructions before optimisation)")
    for first_, second_, r in ((A, B, fwd), (B, A, rev)):
        print(f"  {first_ + ' → ' + second_:<{w}} : I = {pct(r.interaction_score):>9}   "
              f"{paint(r.interaction_type, TYPE_COLOR.get(r.interaction_type))}")
    print(f"  Finished in {time.perf_counter() - started:.1f} s. Every number above was measured "
          f"in this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
