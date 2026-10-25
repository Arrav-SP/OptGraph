"""v3.1 starting IR, size attributes, one-invocation sequences, no-op intrinsic exclusion."""
import shutil

import pytest

from src import llvm_pipeline as lp
from src.metrics import measure_ir, parse_ir
from src.utils import PassSpec, load_passes

needs_llvm = pytest.mark.skipif(not shutil.which("opt"), reason="LLVM not installed")

SRC = """#include <stdio.h>
static int sq(int x) { int a[4]; for (int i = 0; i < 4; i++) a[i] = x * i; return a[3]; }
int main(void) { int s = 0; for (int i = 0; i < 100; i++) s += sq(i); printf("%d\\n", s); return 0; }
"""


@pytest.fixture
def ir(tmp_path):
    tc = lp.Toolchain()
    src = tmp_path / "t.c"
    src.write_text(SRC)
    ll = tmp_path / "t.ll"
    assert lp.compile_to_ir(tc, src, ll, irflags=lp.IR_FLAGS_V3_1).ok
    return tc, ll


def test_noop_intrinsics_not_counted():
    text = """define void @f(ptr %p) {
  call void @llvm.lifetime.start.p0(i64 4, ptr %p)
  store i32 1, ptr %p, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr %p)
  call void @g()
  ret void
}
"""
    p = parse_ir(text)
    assert sum(p.opcodes.values()) == 3 and p.opcodes["call"] == 1


@needs_llvm
def test_v3_1_ir_has_no_noinline_and_inline_works(ir, tmp_path):
    tc, ll = ir
    text = ll.read_text()
    attrs = "\n".join(l for l in text.splitlines() if l.startswith(("attributes", "define")))
    assert "noinline" not in attrs and "optnone" not in attrs
    out = tmp_path / "inl.ll"
    assert lp.run_pass(tc, ll, out, PassSpec("inline", "cgscc(inline)")).ok
    assert "call i32 @sq" in text and "call i32 @sq" not in out.read_text()


@needs_llvm
def test_size_attrs_make_os_differ_from_o2(ir, tmp_path):
    tc, ll = ir
    s = tmp_path / "os_in.ll"
    s.write_text(lp.add_size_attrs(ll.read_text(), "Oz"))
    assert lp.verify_ir(tc, s).ok and "optsize minsize" in s.read_text()
    o2, oz = tmp_path / "o2.ll", tmp_path / "oz.ll"
    assert lp.run_pass(tc, ll, o2, PassSpec("O2", "default<O2>")).ok
    assert lp.run_pass(tc, s, oz, PassSpec("Oz", "default<Oz>")).ok
    assert lp.ir_body(o2.read_text()) != lp.ir_body(oz.read_text())


@needs_llvm
def test_one_invocation_sequence_equals_per_pass(ir, tmp_path):
    tc, ll = ir
    specs = load_passes(names=["sroa", "inline", "instcombine", "licm", "loop-unroll", "gvn"], v3=True)
    cur = ll
    for i, s in enumerate(specs):
        nxt = tmp_path / f"p{i}.ll"
        assert lp.run_pass(tc, cur, nxt, s).ok
        cur = nxt
    one = tmp_path / "one.ll"
    assert lp.run_pass(tc, ll, one, PassSpec("seq", lp.sequence_pipeline(specs))).ok
    assert measure_ir(one) == measure_ir(cur)
