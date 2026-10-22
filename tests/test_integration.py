"""End-to-end v3 pipeline on 3 programs x 4 passes x 1 context (config `quick`), in a temp dir.

experiments -> interaction dataset -> features -> LOPO prediction -> graph-built sequence,
applied and measured. Needs LLVM, valgrind and the Phase 1 manifest.
"""
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from src import llvm_pipeline as lp
from src import model as ml
from src.experiments_v3 import V3_METRICS, run_program_v3
from src.features import extract_features
from src.interaction import compute_interactions
from src.measure import measure_variant
from src.ordering import greedy
from src.suites import load_manifest
from src.utils import DATA_DIR, load_config, load_passes, load_v3_config

MANIFEST = DATA_DIR / "v3" / "benchmarks.csv"
pytestmark = pytest.mark.skipif(not (shutil.which("opt") and shutil.which("valgrind") and MANIFEST.exists()),
                                reason="needs LLVM, valgrind and data/v3/benchmarks.csv")


def test_tiny_pipeline(tmp_path):
    start = time.perf_counter()
    v3, cfg = load_v3_config(), load_config()
    q = v3["quick"]
    tc = lp.Toolchain().require()
    benches = load_manifest(MANIFEST, q["programs"])
    passes = load_passes(cfg, q["passes"], v3=True)
    contexts = {c: v3["contexts"][c] for c in q["contexts"]}
    m = pd.read_csv(MANIFEST, keep_default_na=False)
    expected = dict(zip(m["program_id"], m["output_sha256"]))
    dyn = {"arith_02_redundant", "ts_stanford_treesort"}      # gemm: static only, to stay fast

    def job(b):
        return run_program_v3(tc, b, passes, contexts, tmp_path / b.program_id,
                              expected[b.program_id], b.program_id in dyn)
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(job, benches))
    recs = pd.DataFrame([r for rs, _ in results for r in rs])
    per = 1 + len(passes) + len(passes) * (len(passes) - 1)
    assert len(recs) == len(benches) * per
    assert (recs["status"] == "SUCCESS").all(), recs[recs.status != "SUCCESS"][["variant", "error"]]
    assert recs["output_matches"].astype(bool).all()

    # resumable: a second call returns the stored records without rerunning
    t = time.perf_counter()
    again, _ = job(benches[0])
    assert time.perf_counter() - t < 2 and len(again) == per

    ds = compute_interactions(recs.astype({x: float for x in V3_METRICS}), 0.005, V3_METRICS)
    assert set(ds["context"]) == {"C0"}
    assert set(ds[ds.metric == "dyn_instructions"]["program_id"]) == dyn
    # identity on measured data: I(A,B) - I(B,A) = (m(BA) - m(AB)) / m(P)
    d = ds[(ds.metric == "text_size")].set_index(["program_id", "pass_a", "pass_b"])
    for (pid, a, b), r in d.iterrows():
        rv = d.loc[(pid, b, a)]
        assert r.interaction_score - rv.interaction_score == pytest.approx((rv.m_ab - r.m_ab) / r.m_base)

    feats = pd.DataFrame([{"program_id": b.program_id, **{"f_" + k: v for k, v in f.items() if k != "loop_method"}}
                          for b, (_, f) in zip(benches, results)])
    rows = ds[(ds.metric == "text_size") & ds.interaction_score.notna()].merge(feats, on="program_id")
    names = [p.name for p in passes]
    X, layout = ml.v3_matrix(rows.reset_index(drop=True), names, [c for c in feats if c.startswith("f_")], "both")
    pred = ml.cross_predict_v3(rows, X, layout, "lopo", ["PairMeanBaseline", "LinearRidge"], 42, "residual", False)
    assert pred[["PairMeanBaseline", "LinearRidge"]].notna().all().all()

    # build a sequence for one program from its predicted graph and measure it for real
    pid, b = "pb_gemm", next(x for x in benches if x.program_id == "pb_gemm")
    g = pred[pred.program_id == pid]
    inter = dict(zip(zip(g.pass_a, g.pass_b), g.LinearRidge))
    s = ds[(ds.metric == "text_size") & (ds.program_id == pid)].drop_duplicates("pass_b")
    seq, gain = greedy(names, dict(zip(s.pass_b, s.delta_b_alone)), inter, 4)
    assert 0 < len(seq) <= 4
    ll = tmp_path / "seq0.ll"
    shutil.copy(tmp_path / pid / "C0" / "baseline.ll", ll)
    specs = {p.name: p for p in passes}
    for i, name in enumerate(seq, 1):
        nxt = tmp_path / f"seq{i}.ll"
        assert lp.run_pass(tc, ll, nxt, specs[name]).ok
        ll = nxt
    res = measure_variant(tc, ll, b.support, expected[pid], {}, False)
    assert res["output_matches"] and res["text_size"] > 0
    assert time.perf_counter() - start < 60
