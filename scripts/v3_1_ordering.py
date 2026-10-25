#!/usr/bin/env python3
"""v3.1 Claim 4 (and Claim 1's headroom test): ordering methods under a compile budget.

Stages (resumable per program):
  search   every method on every program, objective = IR instruction count of the sequence
           applied to the program's start IR (S0), budget 200 evaluations, best-so-far recorded
           at 1/5/10/25/50/100/200 (src/search.py defines the budget unit)
  measure  each method's best-so-far sequence at every checkpoint + the -O levels: output check,
           .text size (x86-64 and AArch64), executed instructions (cachegrind)
  runtime  wall-clock (median of N pinned runs, randomised order). Refuses to start while any
           other OptGraph process is running; run it alone on an idle machine.

Methods (graphs and models are built from the other programs only):
  repredict        re-prediction: LOPO single-pass model applied to each state's own IR features
  predict_once     Claim 3 LOPO predictions of the S0 graph, capped beam
  static_graph     pair-mean graph of the other programs at S0, capped beam
  oracle_graph     the program's own measured S0 graph (headroom; ~485 compiles to obtain)
  graph_guided     static-graph proposals, top-3 per step verified by measurement
  pan_style        synergy graph walks seeding and mutating a GA            (5 seeds)
  dco_style        positive-edge sub-sequences searched by OpenTuner        (5 seeds)
  opentuner        OpenTuner over single passes                             (5 seeds)
  ga               genetic algorithm                                         (5 seeds)
  random           random search                                             (5 seeds)
  measured_greedy  upper-bound reference, budget = until it stops (true count on the curve)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _bootstrap  # noqa: F401,E402
from src import llvm_pipeline as lp  # noqa: E402
from src import search as se  # noqa: E402
from src.features import all_representations  # noqa: E402
from src.measure import build, dynamic_count, measure_variant, text_size, timed_runs  # noqa: E402
from src.metrics import measure_ir  # noqa: E402
from src.static_graphs import dco_classes, dco_subsequences, pair_mean, pan_synergy_graph, weighted_walk  # noqa: E402
from src.suites import load_manifest  # noqa: E402
from src.utils import (DATA_DIR, OUTPUT_DIR, PassSpec, load_config, load_passes, load_v3_config,  # noqa: E402
                       now_iso, rel_path, run_cmd, version, work_dir, write_json)

SEEDS = (1, 2, 3, 4, 5)
O_LEVELS = ("O1", "O2", "O3", "Os", "Oz")
METRIC = "instruction_count"


class Program:
    """One program's start IR, the IR-count objective and IR states, with shared caches."""

    def __init__(self, tc, bench, start_ll: Path, specs: dict, wd: Path, vocab):
        self.tc, self.bench, self.start, self.specs, self.wd, self.vocab = tc, bench, start_ll, specs, wd, vocab
        wd.mkdir(parents=True, exist_ok=True)
        self.values: dict[tuple, float | None] = {}
        self.states: dict[tuple, Path] = {(): start_ll}

    def _ir(self, seq: tuple) -> Path | None:
        if not seq:
            return self.start
        name = hashlib.sha1("|".join(seq).encode()).hexdigest()[:20]
        out = self.wd / f"{name}.ll"
        if not out.exists():
            pipe = lp.sequence_pipeline([self.specs[p] for p in seq])
            if not lp.run_pass(self.tc, self.start, out, PassSpec("seq", pipe), 120).ok:
                return None
        return out

    def objective(self, seq: tuple) -> float | None:
        if seq not in self.values:
            ll = self._ir(seq)
            self.values[seq] = None if ll is None else float(measure_ir(ll)[METRIC])
            if ll is not None and ll != self.start and seq not in self.states:
                ll.unlink(missing_ok=True)
        return self.values[seq]

    def state(self, seq: tuple) -> Path | None:
        if seq not in self.states:
            self.states[seq] = self._ir(seq)
        return self.states[seq]

    def features(self, path: Path):
        f = {}
        for v in all_representations(path, self.tc, self.vocab).values():
            f.update(v)
        return f, 3            # opt calls: print<loops>, print<func-properties>, print<ir2vec>

    def level_ir(self, level: str) -> Path | None:
        out = self.wd / f"{level}.ll"
        src = self.start
        if level in lp.SIZE_ATTRS:
            src = self.wd / f"{level}_in.ll"
            src.write_text(lp.add_size_attrs(self.start.read_text(), level))
        ok = lp.run_pass(self.tc, src, out, PassSpec(level, f"default<{level}>"), 300).ok
        return out if ok else None


def graphs_for(pid, ds, singles, pred_pairs, pred_singles, thr):
    """S0 graphs. Static ones use the other programs only; oracle uses the program itself."""
    d0, s0 = ds[ds["context"] == "S0"], singles[singles["context"] == "S0"]
    others, s_others = d0[d0["program_id"] != pid], s0[s0["program_id"] != pid]
    own, s_own = d0[d0["program_id"] == pid], s0[s0["program_id"] == pid]
    g = {"static_graph": pair_mean(others, s_others), "oracle_graph": pair_mean(own, s_own)}
    pp = pred_pairs[(pred_pairs["program_id"] == pid) & (pred_pairs["context"] == "S0")]
    ps = pred_singles[(pred_singles["program_id"] == pid) & (pred_singles["context"] == "S0")]
    g["predict_once"] = (dict(zip(ps["pass_b"], ps["predicted"])),
                         dict(zip(zip(pp["pass_a"], pp["pass_b"]), pp["predicted"])))
    g["pan"] = pan_synergy_graph(others, s_others, thr)
    _, inter_o = g["static_graph"]
    g["dco"] = dco_subsequences(dco_classes(others, thr), inter_o, max_len=4)
    return g


def run_methods(prog: Program, passes, graphs, model, feat_cols, budget, seed0):
    rows = []

    def ev(budget_=budget, with_state=False):
        return se.Evaluator(prog.objective, budget_, prog.state if with_state else None,
                            prog.features if with_state else None)

    def record(name, e, seed=0):
        rows.extend(se.summarise(e, method=name, seed=seed).to_dict("records"))

    def predict_delta(f):
        x = pd.DataFrame([{"pass_a": p, "pass_b": p, **f} for p in passes])
        return dict(zip(passes, model.predict(x)))

    e = ev(with_state=True)
    se.repredict(e, passes, predict_delta)
    record("repredict", e)
    for name in ("predict_once", "static_graph", "oracle_graph"):
        e = ev()
        se.graph_method(e, passes, *graphs[name])
        record(name, e)
    e = ev()
    se.graph_guided(e, passes, *graphs["static_graph"])
    record("graph_guided", e)
    nodes, edges = graphs["pan"]
    alphabet_dco = [()] + list(graphs["dco"])
    for s in SEEDS:
        rng = np.random.default_rng(seed0 * 1000 + s)
        e = ev()
        se.genetic(e, passes, rng, init=lambda: tuple(weighted_walk(nodes, edges, se.MAX_LEN, rng)),
                   mutate_segment=lambda n: weighted_walk(nodes, edges, max(n, 1), rng))
        record("pan_style", e, s)
        e = ev()
        se.opentuner_search(e, alphabet_dco, slots=8, seed=seed0 * 1000 + s)
        record("dco_style", e, s)
        e = ev()
        se.opentuner_search(e, [()] + [(p,) for p in passes], slots=se.MAX_LEN, seed=seed0 * 1000 + s)
        record("opentuner", e, s)
        e = ev()
        se.genetic(e, passes, np.random.default_rng(seed0 * 1000 + s))
        record("ga", e, s)
        e = ev()
        se.random_search(e, passes, np.random.default_rng(seed0 * 1000 + s))
        record("random", e, s)
    e = ev(budget_=10_000)
    se.measured_greedy(e, passes)
    record("measured_greedy", e)
    return rows


_CTX: dict = {}


def _init_search(args, v3, cfg, tag):
    """Per-process setup (each program's search runs in its own process: OpenTuner keeps an
    SQLite database that must not cross threads, and the GA/OpenTuner loops are pure Python)."""
    thr = cfg["settings"]["interaction_threshold"]
    data = DATA_DIR / version() / tag
    # Only what graphs_for needs (S0, one metric): 8 processes each holding the full dataset
    # would not fit in WSL's memory.
    cols = ["program_id", "context", "metric", "pass_a", "pass_b", "interaction_score", "delta_b_alone"]
    ds = pd.concat(c[(c["metric"] == METRIC) & (c["context"] == "S0") & c["interaction_score"].notna()]
                   for c in pd.read_csv(data / "interaction_dataset.csv.gz", usecols=cols, chunksize=500_000))
    claims = json.loads((OUTPUT_DIR / version() / tag / "reports" / "claims_summary.json").read_text())
    specs = {p.name: p for p in load_passes(cfg, v3=True)}
    base = work_dir(v3) / version(v3) / (tag or "full")
    _CTX.update(args=args, v3=v3, thr=thr, seed=v3["seed"], ds=ds,
                singles=ds.drop_duplicates(["program_id", "context", "pass_b"]),
                pred_pairs=pd.read_csv(data / "claims" / "lopo_pairs.csv.gz").query("context == 'S0'"),
                pred_singles=pd.read_csv(data / "claims" / "lopo_singles.csv.gz").query("context == 'S0'"),
                feat_cols=claims["single_pass_lopo"]["feature_columns"], specs=specs,
                passes=[p for p in specs if p in set(ds["pass_a"])], exp=base / "experiments",
                mdir=base / "models" / "single_lopo", root=base / "ordering",
                vocab=work_dir(v3) / v3["ir2vec_vocab"], tc=lp.Toolchain().require())


def _search_one(b):
    c = _CTX
    wd = c["root"] / b.program_id
    model_file = c["mdir"] / f"{b.program_id}.joblib"
    sig = hashlib.sha256(json.dumps([rel_path(b.path), list(b.cflags), c["passes"], c["args"].budget, SEEDS,
                                     hashlib.sha256(model_file.read_bytes()).hexdigest()]).encode()).hexdigest()
    done = wd / "done.json"
    if done.exists() and json.loads(done.read_text())["signature"] == sig:
        return b.program_id, json.loads(done.read_text())["rows"], 0.0
    shutil.rmtree(wd, ignore_errors=True)
    t = time.perf_counter()
    prog = Program(c["tc"], b, c["exp"] / b.program_id / "S0" / "baseline.ll", c["specs"], wd / "ir", c["vocab"])
    g = graphs_for(b.program_id, c["ds"], c["singles"], c["pred_pairs"], c["pred_singles"], c["thr"])
    rows = run_methods(prog, c["passes"], g, joblib.load(model_file), c["feat_cols"], c["args"].budget, c["seed"])
    base = {"O0": prog.objective(())}
    for lv in O_LEVELS:
        ll = prog.level_ir(lv)
        base[lv] = float(measure_ir(ll)[METRIC]) if ll else None
    rows = [{"program_id": b.program_id, **r} for r in rows]
    rows += [{"program_id": b.program_id, "method": lv, "seed": 0, "checkpoint": 0, "best": v,
              "best_seq": f"default<{lv}>" if lv != "O0" else "", "E": 1, "O": 1, "seconds": 0.0}
             for lv, v in base.items()]
    secs = time.perf_counter() - t
    shutil.rmtree(wd / "ir", ignore_errors=True)
    write_json(done, {"signature": sig, "rows": rows, "finished": now_iso(), "seconds": round(secs, 1)})
    return b.program_id, rows, secs


def search_stage(args, v3, cfg, benches, tag):
    from concurrent.futures import ProcessPoolExecutor
    data = DATA_DIR / version() / tag
    out = []
    with ProcessPoolExecutor(args.workers, initializer=_init_search, initargs=(args, v3, cfg, tag)) as pool:
        for pid, rows, secs in pool.map(_search_one, benches):
            out.extend(rows)
            print(f"  {pid:<34} {len(rows):>5} rows  {secs:7.0f}s", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(data / "ordering_search.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    return df


def measure_stage(args, v3, cfg, benches, tag):
    """Size (x86-64, AArch64), executed instructions and output check of every distinct best
    sequence and -O level per program."""
    data = DATA_DIR / version() / tag
    exp = work_dir(v3) / version(v3) / (tag or "full") / "experiments"
    root = work_dir(v3) / version(v3) / (tag or "full") / "measure"
    df = pd.read_csv(data / "ordering_search.csv.gz", keep_default_na=False)
    manifest = pd.read_csv(DATA_DIR / version() / "benchmarks.csv", keep_default_na=False)
    expected = dict(zip(manifest["program_id"], manifest["output_sha256"]))
    specs = {p.name: p for p in load_passes(cfg, v3=True)}
    tc = lp.Toolchain().require()

    def job(b):
        wd = root / b.program_id
        done = wd / "done.json"
        seqs = sorted(set(df[df["program_id"] == b.program_id]["best_seq"]))
        sig = hashlib.sha256(json.dumps(seqs).encode()).hexdigest()
        if done.exists() and json.loads(done.read_text())["signature"] == sig:
            return json.loads(done.read_text())["rows"]
        shutil.rmtree(wd, ignore_errors=True)
        wd.mkdir(parents=True)
        start = exp / b.program_id / "S0" / "baseline.ll"
        prog = Program(tc, b, start, specs, wd, None)
        rows, cache = [], {}
        for s in seqs:
            if s.startswith("default<"):
                ll = prog.level_ir(s[len("default<"):-1])
            else:
                ll = prog.state(tuple(x for x in s.split(",") if x)) if s else start
            if ll is None:
                rows.append({"program_id": b.program_id, "best_seq": s, "error": "opt failed"})
                continue
            m = measure_variant(tc, ll, b, expected[b.program_id], cache, dynamic=True)
            a64 = wd / "a64.o"
            r = run_cmd(tc.aarch64_object_cmd(ll, a64), 120)
            m["text_size_aarch64"] = text_size(tc, a64) if r.ok else None
            rows.append({"program_id": b.program_id, "best_seq": s, **{k: m.get(k) for k in (
                "ir_sha256", "instruction_count", "text_size", "text_size_aarch64", "dyn_instructions",
                "output_matches", "error")}})
        shutil.rmtree(wd, ignore_errors=True)
        wd.mkdir(parents=True)
        write_json(done, {"signature": sig, "rows": rows, "finished": now_iso()})
        return rows

    out = []
    with ThreadPoolExecutor(args.workers) as pool:
        for b, rows in zip(benches, pool.map(job, benches)):
            out.extend(rows)
            bad = sum(1 for r in rows if r.get("error") or r.get("output_matches") is False)
            print(f"  {b.program_id:<34} {len(rows):>4} sequences  {bad} failed", flush=True)
    m = pd.DataFrame(out)
    m.to_csv(data / "ordering_measure.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    return m


def other_optgraph_processes() -> list[str]:
    me = os.getpid()
    r = run_cmd(["ps", "-eo", "pid,args"])
    # Only python workers count: the parent shell chain's own command line also names scripts/v3*.
    return [l.strip() for l in r.stdout.splitlines()[1:]
            if l.split(None, 1)[1].startswith("python") and ("scripts/v3" in l or "pytest" in l)
            and int(l.split()[0]) != me]


def runtime_stage(args, v3, cfg, benches, tag):
    """Serial wall-clock runs, alone on the machine, random order over (program, sequence)."""
    busy = other_optgraph_processes()
    if busy and not args.force_runtime:
        print("runtime stage refused: other OptGraph processes are running:\n  " + "\n  ".join(busy))
        sys.exit(3)            # non-zero so the run chain does not mark itself complete
    data = DATA_DIR / version() / tag
    m = pd.read_csv(data / "ordering_measure.csv.gz", keep_default_na=False)
    manifest = pd.read_csv(DATA_DIR / version() / "benchmarks.csv", keep_default_na=False)
    reliable = set(manifest[manifest["runtime_reliable"].astype(str) == "True"]["program_id"])
    search = pd.read_csv(data / "ordering_search.csv.gz", keep_default_na=False)
    keep = search[search["checkpoint"].isin([0, 10, 25, 200])]
    todo = (m[m["program_id"].isin(reliable) & (m["output_matches"].astype(str) == "True")]
            .merge(keep[["program_id", "best_seq"]].drop_duplicates(), on=["program_id", "best_seq"])
            .drop_duplicates(["program_id", "ir_sha256"]))
    todo = todo.sample(frac=1.0, random_state=v3["seed"])          # randomised order
    out_csv = data / "ordering_runtime.csv"
    have = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        have = set(zip(prev["program_id"], prev["ir_sha256"]))
    exp = work_dir(v3) / version(v3) / (tag or "full") / "experiments"
    wd = work_dir(v3) / version(v3) / (tag or "full") / "runtime"
    specs = {p.name: p for p in load_passes(cfg, v3=True)}
    by_id = {b.program_id: b for b in benches}
    tc = lp.Toolchain().require()
    o = v3["ordering"]
    print(f"runtime: {len(todo)} distinct binaries, {len(todo) - len(have & set(zip(todo.program_id, todo.ir_sha256)))} to time, "
          f"{o['runtime_runs']} runs + {o['runtime_warmup']} warm-up each, core {o['runtime_core']}")
    for i, r in enumerate(todo.itertuples(), 1):
        if (r.program_id, r.ir_sha256) in have:
            continue
        b = by_id[r.program_id]
        shutil.rmtree(wd, ignore_errors=True)
        prog = Program(tc, b, exp / b.program_id / "S0" / "baseline.ll", specs, wd, None)
        s = r.best_seq
        ll = prog.level_ir(s[len("default<"):-1]) if s.startswith("default<") else \
            (prog.state(tuple(x for x in s.split(",") if x)) if s else prog.start)
        _, exe, err = build(tc, Path(ll), b.support)
        times, err2 = ([], err) if err else timed_runs(exe, o["runtime_runs"], o["runtime_warmup"],
                                                       o["runtime_core"], bench=b)
        row = {"program_id": r.program_id, "ir_sha256": r.ir_sha256, "best_seq": s,
               "times": json.dumps([round(t, 6) for t in times]), "error": err2, "timestamp": now_iso()}
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    shutil.rmtree(wd, ignore_errors=True)
    return pd.read_csv(out_csv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", choices=["search", "measure", "runtime"], required=True)
    ap.add_argument("--programs", nargs="+", metavar="ID")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--force-runtime", action="store_true", help="time even if other processes run (not for results)")
    args = ap.parse_args()
    cfg, v3 = load_config(), load_v3_config()
    tag = "quick" if args.quick else ""
    q = v3["quick"] if args.quick else {}
    args.workers = args.workers or v3["workers"]
    benches = load_manifest(DATA_DIR / version() / "benchmarks.csv", args.programs or q.get("programs"))
    t = time.perf_counter()
    {"search": search_stage, "measure": measure_stage, "runtime": runtime_stage}[args.stage](args, v3, cfg, benches, tag)
    write_json(DATA_DIR / version() / tag / f"ordering_{args.stage}_run.json",
               {"timestamp": now_iso(), "elapsed_s": round(time.perf_counter() - t, 1), "stage": args.stage,
                "programs": [b.program_id for b in benches], "budget": args.budget,
                "environment": lp.collect_environment(lp.Toolchain())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
