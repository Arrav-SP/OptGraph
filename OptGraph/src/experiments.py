"""Controlled pass experiments.

For a program P and every configured pass A, B (A != B) this produces

    baseline.ll            P
    A.ll                   P -> A
    A_then_B.ll            P -> A -> B      (B applied to the saved A.ll)

so both orders A->B and B->A exist for every unordered pair. One `opt`
invocation is one experiment; each is logged with its status and command, and a
failure never stops the remaining experiments.
"""
from __future__ import annotations

import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from . import llvm_pipeline as lp
from .features import extract_features
from .metrics import METRICS, measure_ir
from .utils import Benchmark, CommandResult, PassSpec, Workspace, now_iso, write_json

LOG_COLUMNS = ["program_id", "category", "variant", "kind", "pass_a", "pass_b", "status",
               "error", "ir_changed", "output_matches", "duration_s", "timestamp", "ir_path",
               "command", "stdout", "stderr"]


def pair_variant(a: str, b: str) -> str:
    return f"{a}_then_{b}"


def _record(bench: Benchmark, variant: str, kind: str, pass_a: str, pass_b: str,
            res: CommandResult | None, ll: Path | None, error: str = "") -> dict:
    rec = {"program_id": bench.program_id, "category": bench.category, "variant": variant,
           "kind": kind, "pass_a": pass_a, "pass_b": pass_b,
           "status": "FAILED", "error": error, "ir_changed": None, "output_matches": None,
           "duration_s": None, "timestamp": now_iso(), "ir_path": str(ll) if ll else "",
           "command": "", "stdout": "", "stderr": ""}
    if res is not None:
        rec.update(command=res.command, stdout=res.stdout, stderr=res.stderr,
                   duration_s=round(res.duration_s, 4), timestamp=res.timestamp,
                   error=error or res.error)
        if res.ok and not error:
            rec["status"] = "SUCCESS"
    return rec


def _measure(rec: dict, ll: Path, parent_ll: Path | None) -> None:
    """Attach IR metrics to a successful record; a parse failure fails the record."""
    try:
        rec.update(measure_ir(ll))
        if parent_ll is not None:
            rec["ir_changed"] = lp.ir_body(ll.read_text()) != lp.ir_body(parent_ll.read_text())
    except Exception as e:  # noqa: BLE001 - any parse problem is recorded, not raised
        rec["status"] = "FAILED"
        rec["error"] = f"metric extraction failed: {e}"


def run_program(tc: lp.Toolchain, bench: Benchmark, passes: list[PassSpec], ws: Workspace,
                verify_outputs: bool = False, timeout: float = 60) -> tuple[list[dict], dict | None]:
    """Run every experiment for one program.

    Returns (records, features). `features` is extracted from baseline.ll only,
    before any pass under study has run, and is None if the baseline failed.
    """
    out_dir = ws.experiments / bench.program_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    records: list[dict] = []
    tmp = Path(tempfile.mkdtemp(prefix=f"optgraph_{bench.program_id}_")) if verify_outputs else None
    expected = None

    def check_output(rec: dict, ll: Path) -> None:
        if tmp is None or rec["status"] != "SUCCESS" or expected is None:
            return
        link, run = lp.build_and_run(tc, ll, tmp)
        if run is None or run.error:
            rec["output_matches"] = False
            rec["error"] = "could not build/run optimized IR: " + (link.error or (run.error if run else ""))
            rec["status"] = "FAILED"
        else:
            rec["output_matches"] = (run.stdout, run.returncode) == expected
            if not rec["output_matches"]:
                rec["status"] = "FAILED"
                rec["error"] = "program output differs from baseline"

    # ---- P ---------------------------------------------------------------
    base_ll = out_dir / "baseline.ll"
    res = lp.compile_to_ir(tc, bench.path, base_ll, timeout)
    rec = _record(bench, "baseline", "baseline", "", "", res, base_ll)
    if rec["status"] == "SUCCESS":
        v = lp.verify_ir(tc, base_ll, timeout)
        if not v.ok:
            rec.update(status="FAILED", error=f"IR verification failed: {v.stderr.strip()}")
    features = None
    if rec["status"] == "SUCCESS":
        _measure(rec, base_ll, None)
    if rec["status"] == "SUCCESS":
        features = extract_features(base_ll, tc)
        if tmp is not None:
            link, run = lp.build_and_run(tc, base_ll, tmp)
            if run is not None and not run.error:
                expected = (run.stdout, run.returncode)
                rec["output_matches"] = True
    records.append(rec)
    baseline_ok = rec["status"] == "SUCCESS"

    # ---- P -> A ----------------------------------------------------------
    single_ok: dict[str, Path] = {}
    for a in passes:
        ll = out_dir / f"{a.name}.ll"
        if not baseline_ok:
            records.append(_record(bench, a.name, "single", a.name, "", None, None,
                                   "skipped: baseline failed"))
            continue
        res = lp.run_pass(tc, base_ll, ll, a, timeout)
        rec = _record(bench, a.name, "single", a.name, "", res, ll)
        if rec["status"] == "SUCCESS":
            _measure(rec, ll, base_ll)
            check_output(rec, ll)
        if rec["status"] == "SUCCESS":
            single_ok[a.name] = ll
        records.append(rec)

    # ---- P -> A -> B, for every ordered pair -------------------------------
    for a in passes:
        for b in passes:
            if a.name == b.name:
                continue
            variant = pair_variant(a.name, b.name)
            ll = out_dir / f"{variant}.ll"
            if a.name not in single_ok:
                records.append(_record(bench, variant, "pair", a.name, b.name, None, None,
                                       f"skipped: first pass {a.name} failed"))
                continue
            res = lp.run_pass(tc, single_ok[a.name], ll, b, timeout)
            rec = _record(bench, variant, "pair", a.name, b.name, res, ll)
            if rec["status"] == "SUCCESS":
                _measure(rec, ll, single_ok[a.name])
                check_output(rec, ll)
            records.append(rec)

    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    write_json(out_dir / "metadata.json", {
        "program_id": bench.program_id, "category": bench.category,
        "source": str(bench.path), "passes": [p.__dict__ for p in passes],
        "features": features, "experiments": records})
    return records, features


def run_all(tc: lp.Toolchain, benches: list[Benchmark], passes: list[PassSpec], ws: Workspace,
            verify_outputs: bool = False, workers: int = 4, timeout: float = 60,
            progress=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all programs; write the measurement table, the log and the feature table."""
    def job(bench):
        try:
            return bench, run_program(tc, bench, passes, ws, verify_outputs, timeout)
        except Exception as e:  # noqa: BLE001 - one broken program must not stop the run
            return bench, ([_record(bench, "baseline", "baseline", "", "", None, None,
                                    f"unexpected error: {e!r}")], None)

    all_records, feature_rows = [], []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for bench, (records, features) in pool.map(job, benches):
            all_records.extend(records)
            ok = sum(r["status"] == "SUCCESS" for r in records)
            progress(f"  {bench.program_id:<26} {ok:>4}/{len(records)} experiments succeeded")
            if features is not None:
                feature_rows.append({"program_id": bench.program_id,
                                     "category": bench.category, **features})

    log = pd.DataFrame(all_records)
    for m in METRICS:
        if m not in log.columns:
            log[m] = pd.NA
    ws.measurements.parent.mkdir(parents=True, exist_ok=True)
    log[LOG_COLUMNS + METRICS].to_csv(ws.experiment_log, index=False)
    meas = log[[c for c in LOG_COLUMNS if c not in ("command", "stdout", "stderr")] + METRICS]
    meas.to_csv(ws.measurements, index=False)
    feats = pd.DataFrame(feature_rows)
    feats.to_csv(ws.features, index=False)
    return meas, feats
