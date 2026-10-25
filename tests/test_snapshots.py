"""v3.1 snapshots: prefixes of opt's default<O2>, pair runner with dynamic counts at S0 only."""
import shutil

import pandas as pd
import pytest

from src import llvm_pipeline as lp
from src.experiments_v3 import run_program_v3
from src.measure import build, run_binary
from src.utils import BENCH_DIR, Benchmark, PassSpec, load_passes

needs_tools = pytest.mark.skipif(not (shutil.which("opt") and shutil.which("valgrind")), reason="LLVM/valgrind")


def test_split_top_level():
    assert lp.split_top_level("a,function(b,c<x;y>),cgscc(d(e,f)),g") == \
        ["a", "function(b,c<x;y>)", "cgscc(d(e,f))", "g"]


@needs_tools
def test_s4_prefix_is_default_o2(tmp_path):
    tc = lp.Toolchain()
    pipes = lp.o2_prefix_pipelines(tc, {"S0": 0, "S2": 15, "S4": 29})
    assert pipes["S0"] == "" and len(pipes["S2"]) < len(pipes["S4"])
    bench = Benchmark("p", "c", BENCH_DIR / "mixed" / "mix_01_matrix_mul.c", irflags=tuple(lp.IR_FLAGS_V3_1))
    ll = tmp_path / "p.ll"
    assert lp.program_ir(tc, bench, ll).ok
    a, b = tmp_path / "a.ll", tmp_path / "b.ll"
    assert lp.run_pass(tc, ll, a, PassSpec("S4", pipes["S4"])).ok
    assert lp.run_pass(tc, ll, b, PassSpec("O2", "default<O2>")).ok
    assert lp.ir_body(a.read_text()) == lp.ir_body(b.read_text())


@needs_tools
def test_dynamic_only_in_selected_snapshot(tmp_path):
    tc = lp.Toolchain()
    bench = Benchmark("arith_02_redundant", "arithmetic", BENCH_DIR / "arithmetic" / "arith_02_redundant.c",
                      irflags=tuple(lp.IR_FLAGS_V3_1))
    ll = tmp_path / "s0.ll"
    assert lp.program_ir(tc, bench, ll).ok
    _, exe, err = build(tc, ll)
    assert not err
    expected = run_binary(exe, 10, tmp_path).digest
    pipes = lp.o2_prefix_pipelines(tc, {"S0": 0, "S2": 15})
    passes = load_passes(names=["mem2reg", "instcombine"], v3=True)
    recs, _ = run_program_v3(tc, bench, passes, pipes, tmp_path / "out", expected, dynamic=True,
                             dynamic_contexts={"S0"})
    assert len(recs) == 2 * (1 + 2 + 2) and all(r["status"] == "SUCCESS" for r in recs)
    s0 = [r for r in recs if r["context"] == "S0"]
    s2 = [r for r in recs if r["context"] == "S2"]
    assert all(pd.notna(r.get("dyn_instructions")) for r in s0)
    s0_irs = {x["ir_sha256"] for x in s0}
    assert all(pd.isna(r.get("dyn_instructions")) for r in s2 if r["ir_sha256"] not in s0_irs)
