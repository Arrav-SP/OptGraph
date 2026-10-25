#!/usr/bin/env python3
"""v3 Phase 5 (RQ3): graph-guided pass ordering, measured for real.

Stages (all resumable):
  measure  apply every strategy's sequence to the raw -O0 IR (C0) of each program and measure
           IR count, .text size, output and dynamic instruction count of the result;
  runtime  pinned wall-clock runs of the key variants of programs with a reliable runtime;
  report   tables, statistical tests and figures.

Strategies per program and budget k: -O0/-O1/-O2/-O3/-Os, the canonical order (first k passes),
random sequences, sequences built (greedy, beam) from three graphs of the program - predicted
(LOPO model that never saw the program), program-agnostic (pair/pass means of the other
programs) and oracle (measured) - and greedy iterative search with real measurements.
"""
import argparse
import hashlib
import json
import queue
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import llvm_pipeline as lp
from src.measure import build, measure_variant, timed_runs
from src.ordering import beam_search, greedy
from src.suites import load_manifest
from src.utils import (version, DATA_DIR, OUTPUT_DIR, PassSpec, load_config, load_passes, load_v3_config, rel_path,
                       now_iso, work_dir, write_json)

O_LEVELS = {"O1": "default<O1>", "O2": "default<O2>", "O3": "default<O3>", "Os": "default<Os>"}
SOURCES = {"predicted": None, "agnostic": "PairMeanBaseline", "oracle": "interaction_score"}
MEASURED = ["instruction_count", "text_size", "dyn_instructions"]


# ---- sequences -----------------------------------------------------------------------

def random_sequences(passes: list[str], k: int, n: int, seed: int, pid: str) -> list[tuple]:
    """n random length-k sequences (no pass twice in a row), seeded per program and budget."""
    h = int(hashlib.sha256(f"{seed}:{pid}:{k}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    out = []
    for _ in range(n):
        seq = []
        while len(seq) < k:
            b = passes[rng.integers(len(passes))]
            if not seq or seq[-1] != b:
                seq.append(b)
        out.append(tuple(seq))
    return out


def graph_for(pid: str, pairs: pd.DataFrame, singles: pd.DataFrame, col: str) -> tuple[dict, dict]:
    p, s = pairs[pairs["program_id"] == pid], singles[singles["program_id"] == pid]
    inter = dict(zip(zip(p["pass_a"], p["pass_b"]), p[col]))
    delta = dict(zip(s["pass_b"], s[col]))
    return delta, inter


class Program:
    """Applies sequences to one program's raw IR (prefix-cached) and measures results."""

    def __init__(self, tc, bench, expected, specs, wd, timeout=60):
        self.tc, self.bench, self.expected, self.specs, self.wd = tc, bench, expected, specs, wd
        self.timeout = timeout
        self.prefix: dict[tuple, str] = {}
        self.cache: dict = {}
        self.evals = 0
        wd.mkdir(parents=True, exist_ok=True)
        self.raw = wd / "raw.ll"
        r = lp.program_ir(tc, bench, self.raw, 120)
        if not r.ok:
            raise RuntimeError(f"compile failed: {r.stderr[:300]}")
        self.prefix[()] = str(self.raw)

    def ir(self, seq: tuple) -> tuple[str, str]:
        """IR path after applying seq (one opt invocation per pass, as in Phase 3)."""
        if seq in self.prefix:
            return self.prefix[seq], ""
        parent, err = self.ir(seq[:-1])
        if err:
            return "", err
        out = self.wd / f"s_{hashlib.sha1('|'.join(seq).encode()).hexdigest()[:16]}.ll"
        r = lp.run_pass(self.tc, parent, out, self.specs[seq[-1]], self.timeout)
        if not r.ok:
            return "", f"opt failed at {seq[-1]}: {r.stderr.strip()[:200]}"
        self.prefix[seq] = str(out)
        return str(out), ""

    def pipeline_ir(self, name: str, pipeline: str) -> tuple[str, str]:
        out = self.wd / f"{name}.ll"
        r = lp.run_pass(self.tc, self.raw, out, PassSpec(name, pipeline), self.timeout)
        return (str(out), "") if r.ok else ("", f"opt failed: {r.stderr.strip()[:200]}")

    def measure(self, ll: str, dynamic: bool = True) -> dict:
        self.evals += 1
        return measure_variant(self.tc, Path(ll), self.bench,
                               self.expected, self.cache, dynamic, True, self.timeout)


def measured_search(prog: Program, passes: list[str], objective: str, kmax: int) -> list[dict]:
    """Greedy iterative search with real measurements: at each step evaluate every pass
    appended to the current sequence, keep the best if it improves the objective."""
    seq: tuple = ()
    cur = prog.measure(str(prog.raw))[objective]
    steps, evals, unique = [], 0, set()
    for _ in range(kmax):
        best = None
        for b in passes:
            if seq and seq[-1] == b:
                continue
            ll, err = prog.ir(seq + (b,))
            evals += 1
            if err:
                continue
            m = prog.measure(ll, dynamic=(objective == "dyn_instructions"))
            unique.add(m["ir_sha256"])
            v = m.get(objective)
            if m.get("error") or v is None:
                continue
            if best is None or v < best[1]:
                best = (b, v)
        if best is None or best[1] >= cur:
            break
        seq, cur = seq + (best[0],), best[1]
        steps.append({"seq": seq, "evals": evals, "unique_evals": len(unique)})
    return steps, {"seq": seq, "evals": evals, "unique_evals": len(unique)}


def program_job(tc, bench, expected, specs, pass_names, graphs, ocfg, seed, wd, sig):
    done = wd / "done.json"
    if done.exists():
        d = json.loads(done.read_text())
        if d.get("signature") == sig:
            return d["rows"]
    if wd.exists():
        shutil.rmtree(wd)
    prog = Program(tc, bench, expected, specs, wd)
    rows = []

    def add(strategy, objective, k, seq, ll, err, cost=1, unique_cost=None, idx=0):
        m = {} if err else prog.measure(ll)
        rows.append({"program_id": bench.program_id, "suite": bench.suite, "strategy": strategy,
                     "objective": objective, "k": k, "idx": idx, "sequence": ",".join(seq),
                     "length": len(seq), "evals": cost,
                     "unique_evals": cost if unique_cost is None else unique_cost,
                     "error": err or m.get("error", ""), "output_matches": m.get("output_matches"),
                     "ir_sha256": m.get("ir_sha256", ""), **{x: m.get(x) for x in MEASURED}})

    add("O0", "any", 0, (), str(prog.raw), "")  # the program's -O0 IR, measured like any variant
    for name, pipe in O_LEVELS.items():
        add(name, "any", 0, (pipe,), *prog.pipeline_ir(name, pipe))
    for k in ocfg["budgets"]:
        seq = tuple(ocfg["canonical"][:k])
        add("canonical", "any", k, seq, *prog.ir(seq))
        for i, seq in enumerate(random_sequences(pass_names, k, ocfg["random_per_budget"], seed,
                                                 bench.program_id)):
            add("random", "any", k, seq, *prog.ir(seq), idx=i)
    for obj in ocfg["objectives"]:
        for src, (delta, inter) in graphs[obj].items():
            for k in ocfg["budgets"]:
                for builder, width in (("greedy", 1), ("beam", ocfg["beam_width"])):
                    seq, _ = beam_search(pass_names, delta, inter, k, width)
                    seq = tuple(seq)
                    add(f"graph_{src}_{builder}", obj, k, seq, *prog.ir(seq))
        steps, final = measured_search(prog, pass_names, obj, max(ocfg["budgets"]))
        for k in ocfg["budgets"]:
            # Budget k = the search's first k steps; if it stopped earlier (no pass improved),
            # its cost includes the last, unsuccessful step.
            s = steps[k - 1] if len(steps) >= k else final
            add("search", obj, k, s["seq"], *prog.ir(s["seq"]), cost=s["evals"],
                unique_cost=s["unique_evals"])
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    write_json(done, {"signature": sig, "rows": rows, "finished": now_iso()})
    return rows


# ---- runtime ---------------------------------------------------------------------------

def key_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose runtime is measured: all non-random strategies, plus per (program, k, objective)
    the random sequence with the median and the one with the best objective value."""
    keep = df[df["strategy"] != "random"].copy()
    keep["role"] = ""
    rnd = []
    r = df[(df["strategy"] == "random") & (df["error"].fillna("") == "")]
    for obj in ("text_size", "dyn_instructions"):
        for (pid, k), g in r.groupby(["program_id", "k"]):
            g = g.sort_values([obj, "idx"])
            rnd.append(g.iloc[[(len(g) - 1) // 2]].assign(role=f"random_median_{obj}"))
            rnd.append(g.iloc[[0]].assign(role=f"random_best_{obj}"))
    return pd.concat([keep] + rnd, ignore_index=True)


def runtime_stage(tc, benches, df, ocfg, root, out_csv):
    keys = key_variants(df)
    keys = keys[(keys["error"].fillna("") == "") & keys["output_matches"].astype(bool)]
    uniq = keys.drop_duplicates(["program_id", "ir_sha256"])
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["program_id", "ir_sha256"])
    have = set(zip(done["program_id"], done["ir_sha256"]))
    todo = [r for r in uniq.itertuples() if (r.program_id, r.ir_sha256) not in have]
    print(f"runtime: {len(uniq)} unique key variants, {len(todo)} to measure on cores {ocfg['runtime_cores']}")
    cores = queue.Queue()
    for c in ocfg["runtime_cores"]:
        cores.put(c)
    by_id = {b.program_id: b for b in benches}

    def job(r):
        b = by_id[r.program_id]
        core = cores.get()
        try:
            wd = root / "runtime" / r.program_id / r.ir_sha256[:16]
            wd.mkdir(parents=True, exist_ok=True)
            p = Program(tc, b, "", SPECS, wd)
            if r.strategy in O_LEVELS:
                ll, err = p.pipeline_ir(r.strategy, O_LEVELS[r.strategy])
            else:
                ll, err = p.ir(tuple(s for s in str(r.sequence).split(",") if s and s != "nan"))
            if err:
                return {"program_id": r.program_id, "ir_sha256": r.ir_sha256, "error": err}
            _, exe, err = build(tc, Path(ll), b.support)
            if err:
                return {"program_id": r.program_id, "ir_sha256": r.ir_sha256, "error": err}
            times, err = timed_runs(exe, ocfg["runtime_runs"], ocfg["runtime_warmup"], core, bench=b)
            shutil.rmtree(wd, ignore_errors=True)
            return {"program_id": r.program_id, "ir_sha256": r.ir_sha256, "core": core,
                    "times": json.dumps([round(t, 6) for t in times]), "error": err}
        finally:
            cores.put(core)

    rows = []
    with ThreadPoolExecutor(len(ocfg["runtime_cores"])) as pool:
        for i, res in enumerate(pool.map(job, todo), 1):
            rows.append(res)
            if i % 100 == 0:
                pd.concat([done, pd.DataFrame(rows)]).to_csv(out_csv, index=False)
                print(f"  {i}/{len(todo)}", flush=True)
    pd.concat([done, pd.DataFrame(rows)]).to_csv(out_csv, index=False)


SPECS: dict = {}


def main() -> int:
    global SPECS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", choices=["measure", "runtime", "report", "all"], default="all")
    ap.add_argument("--programs", nargs="+", metavar="ID")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    cfg, v3 = load_config(), load_v3_config()
    ocfg, seed = v3["ordering"], v3["seed"]
    tag = "quick" if args.quick else ""
    if args.quick:
        ocfg = {**ocfg, **v3["quick"].get("ordering", {})}
    data, out = DATA_DIR / version() / tag, OUTPUT_DIR / version() / tag
    root = work_dir(v3) / version(v3) / (tag or "full") / "ordering"
    tc = lp.Toolchain().require()
    inputs = json.loads((out / "reports" / "phase5_inputs.json").read_text())
    objectives = [o for o in ocfg["objectives"] if o in inputs]
    ocfg = {**ocfg, "objectives": objectives}
    pairs = {o: pd.read_csv(inputs[o]["pairs"]) for o in objectives}
    singles = {o: pd.read_csv(inputs[o]["singles"]) for o in objectives}
    ds_present = set(pairs[objectives[0]]["pass_a"])
    specs = {p.name: p for p in load_passes(cfg, v3=True)}
    SPECS = specs
    pass_names = [n for n in specs if n in ds_present]
    programs = sorted(set.intersection(*(set(pairs[o]["program_id"]) for o in objectives)))
    if args.programs:
        programs = [p for p in programs if p in set(args.programs)]
    manifest = pd.read_csv(DATA_DIR / version() / "benchmarks.csv", keep_default_na=False)
    expected = dict(zip(manifest["program_id"], manifest["output_sha256"]))
    benches = load_manifest(DATA_DIR / version() / "benchmarks.csv", programs)
    meas_csv = data / "ordering_measurements.csv.gz"
    rt_csv = data / "ordering_runtime.csv"
    start = time.perf_counter()

    if args.stage in ("measure", "all"):
        print(f"Phase 5 measure: {len(benches)} programs, {len(pass_names)} passes, objectives {objectives}, "
              f"budgets {ocfg['budgets']}, {ocfg['random_per_budget']} random per budget")
        pred_col = ocfg["predictor"]

        def job(b):
            graphs = {}
            for o in objectives:
                graphs[o] = {}
                for src, col in SOURCES.items():
                    graphs[o][src] = graph_for(b.program_id, pairs[o], singles[o], col or pred_col)
            sig = hashlib.sha256(json.dumps([ocfg, pass_names, expected[b.program_id], rel_path(b.path),
                                             list(b.cflags), {o: {s: [sorted(map(list, g[0].items())),
                                                                       sorted([list(k) + [v] for k, v in g[1].items()])]
                                                                  for s, g in graphs[o].items()} for o in objectives}],
                                            default=str).encode()).hexdigest()
            t = time.perf_counter()
            try:
                rows = program_job(tc, b, expected[b.program_id], specs, pass_names, graphs, ocfg,
                                   seed, root / b.program_id, sig)
            except Exception as e:  # noqa: BLE001
                rows = [{"program_id": b.program_id, "strategy": "O0", "error": f"unexpected: {e!r}"}]
            return b, rows, time.perf_counter() - t

        rows = []
        with ThreadPoolExecutor(args.workers or v3["workers"]) as pool:
            for b, r, secs in pool.map(job, benches):
                rows.extend(r)
                bad = sum(bool(x.get("error")) for x in r)
                print(f"  {b.program_id:<34} {len(r):>4} variants  {bad} failed  {secs:6.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(meas_csv, index=False)
        print(f"measure stage: {time.perf_counter() - start:.0f}s -> {meas_csv}")

    if args.stage in ("runtime", "all"):
        df = pd.read_csv(meas_csv)
        rel = set(manifest[manifest["runtime_reliable"].astype(str) == "True"]["program_id"])
        t = time.perf_counter()
        runtime_stage(tc, [b for b in benches if b.program_id in rel], df[df["program_id"].isin(rel)],
                      ocfg, root, rt_csv)
        print(f"runtime stage: {time.perf_counter() - t:.0f}s -> {rt_csv}")

    if args.stage in ("report", "all"):
        from v3_ordering_report import report
        report(data, out, ocfg, seed)
    write_json(data / "ordering_run.json", {"timestamp": now_iso(), "elapsed_s": round(time.perf_counter() - start, 1),
                                            "stage": args.stage, "programs": programs, "ordering": ocfg,
                                            "passes": pass_names, "environment": lp.collect_environment(tc)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
