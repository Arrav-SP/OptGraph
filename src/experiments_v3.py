"""v3 pair experiments: context x pass x ordered pair, measured on IR, object and binary.

For a program and every context c (a pipeline applied to the raw -O0 IR):

    <c>/baseline.ll        P_c
    <c>/<A>.ll             P_c -> A
    <c>/<A>_then_<B>.ll    P_c -> A -> B   (B applied to the saved A.ll; deleted after measuring)

Every IR is measured by measure.measure_variant: the 8 v2 IR counts, `.text` size of the
fixed-backend object, an output check against the program's -O0 output, and, for programs
in the dynamic subset, the executed-instruction count. Results are cached by IR hash, and a
finished program is skipped on the next run if its configuration is unchanged.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from . import llvm_pipeline as lp
from .experiments import pair_variant
from .features import extract_features
from .measure import measure_variant
from .metrics import METRICS
from .utils import Benchmark, PassSpec, now_iso, rel_path, write_json

V3_METRICS = METRICS + ["text_size", "dyn_instructions"]
RECORD_COLUMNS = ["program_id", "suite", "category", "context", "variant", "kind", "pass_a",
                  "pass_b", "status", "error", "ir_changed", "output_matches", "duration_s",
                  "timestamp", "ir_sha256"] + V3_METRICS + ["stderr"]


def signature(bench: Benchmark, passes: list[PassSpec], contexts: dict, dynamic: bool,
              expected: str) -> dict:
    """Everything that determines a program's results; a change forces a rerun."""
    return {"program_id": bench.program_id, "source": rel_path(bench.path), "cflags": list(bench.cflags),
            "support": [str(s) for s in bench.support], "passes": [p.__dict__ for p in passes],
            "contexts": contexts, "dynamic": dynamic, "expected_output": expected}


def run_program_v3(tc: lp.Toolchain, bench: Benchmark, passes: list[PassSpec], contexts: dict,
                   out_dir: Path, expected: str, dynamic: bool = False, keep_pairs: bool = False,
                   timeout: float = 60) -> tuple[list[dict], dict | None]:
    """All experiments of one program. Returns (records, features of the raw -O0 IR)."""
    sig = signature(bench, passes, contexts, dynamic, expected)
    done = out_dir / "done.json"
    if done.exists() and json.loads(done.read_text()).get("signature") == sig:
        d = json.loads(done.read_text())
        return pd.read_csv(out_dir / "records.csv", keep_default_na=False,
                           na_values=[""]).to_dict("records"), d["features"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cache: dict = {}
    records: list[dict] = []

    def record(ctx, variant, kind, a, b, res, ll, parent_sha=None, error=""):
        rec = {"program_id": bench.program_id, "suite": bench.suite, "category": bench.category,
               "context": ctx, "variant": variant, "kind": kind, "pass_a": a, "pass_b": b,
               "status": "FAILED", "error": error, "ir_changed": None, "output_matches": None,
               "duration_s": round(res.duration_s, 4) if res else None,
               "timestamp": res.timestamp if res else now_iso(), "ir_sha256": "", "stderr": ""}
        if res is not None and not res.ok:
            rec["error"] = error or res.error
            rec["stderr"] = res.stderr.strip()[:2000]
        elif not error and ll is not None:
            m = measure_variant(tc, ll, bench.support, expected, cache, dynamic, True, timeout)
            rec.update(m)
            rec["ir_changed"] = None if parent_sha is None else m["ir_sha256"] != parent_sha
            if not m.get("error"):
                rec["status"] = "SUCCESS"
        records.append(rec)
        return rec

    raw = out_dir / "raw.ll"
    res = lp.compile_to_ir(tc, bench.path, raw, timeout, bench.cflags)
    features = extract_features(raw, tc) if res.ok else None
    for ctx, pipeline in contexts.items():
        cdir = out_dir / ctx
        cdir.mkdir()
        base = cdir / "baseline.ll"
        if not res.ok:
            record(ctx, "baseline", "baseline", "", "", res, None)
            continue
        cres = res
        if pipeline:
            cres = lp.run_pass(tc, raw, base, PassSpec(ctx, pipeline), timeout)
        else:
            shutil.copy(raw, base)
        brec = record(ctx, "baseline", "baseline", "", "", cres, base)
        if brec["status"] != "SUCCESS":
            continue
        singles = {}
        for a in passes:
            ll = cdir / f"{a.name}.ll"
            r = record(ctx, a.name, "single", a.name, "", lp.run_pass(tc, base, ll, a, timeout), ll,
                       brec["ir_sha256"])
            if r["status"] == "SUCCESS":
                singles[a.name] = (ll, r["ir_sha256"])
        for a in passes:
            for b in passes:
                if a.name == b.name:
                    continue
                variant = pair_variant(a.name, b.name)
                if a.name not in singles:
                    record(ctx, variant, "pair", a.name, b.name, None, None,
                           error=f"skipped: first pass {a.name} failed")
                    continue
                ll = cdir / f"{variant}.ll"
                src, sha = singles[a.name]
                record(ctx, variant, "pair", a.name, b.name, lp.run_pass(tc, src, ll, b, timeout),
                       ll, sha)
                if not keep_pairs:
                    ll.unlink(missing_ok=True)
    df = pd.DataFrame(records)
    for c in RECORD_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df[RECORD_COLUMNS].to_csv(out_dir / "records.csv", index=False)
    write_json(done, {"signature": sig, "features": features, "finished": now_iso()})
    return df[RECORD_COLUMNS].to_dict("records"), features
