import textwrap

import pytest

from src.metrics import measure_ir, parse_cachegrind, parse_ir, parse_text_size, metrics_from_parsed
from src.suites import fp_equal, reference_text

LL = textwrap.dedent("""\
    ; ModuleID = 'x.c'
    @g = global i32 0
    declare i32 @printf(ptr, ...)

    define i32 @f(i32 %n) {
    entry:
      %c = icmp sgt i32 %n, 0
      br i1 %c, label %loop, label %exit
    loop:
      %i = phi i32 [ 0, %entry ], [ %i2, %loop ]
      %v = load i32, ptr @g, align 4
      %s = add nsw i32 %v, %i
      store i32 %s, ptr @g, align 4
      %i2 = add i32 %i, 1
      %d = icmp slt i32 %i2, %n
      br i1 %d, label %loop, label %exit
    exit:
      %r = tail call i32 (ptr, ...) @printf(ptr null)
      switch i32 %r, label %a [
        i32 0, label %b
        i32 1, label %b
      ]
    a:
      ret i32 1
    b:
      ret i32 0
    }
    """)


def test_parse_ir_counts():
    m = metrics_from_parsed(parse_ir(LL))
    assert m == {"instruction_count": 13, "basic_blocks": 5, "branches": 3, "conditional_branches": 3,
                 "loads": 1, "stores": 1, "phis": 1, "calls": 1}


def test_parse_ir_cfg_handles_multiline_switch():
    p = parse_ir(LL)
    blocks = p.cfg["f"]
    assert set(blocks["exit"]) == {"a", "b"}
    assert set(blocks["loop"]) == {"loop", "exit"}
    assert p.functions == 1


def test_measure_ir_file(tmp_path):
    f = tmp_path / "x.ll"
    f.write_text(LL)
    assert measure_ir(f)["instruction_count"] == 13


def test_parse_text_size():
    out = "x.o  :\nsection   size   addr\n.strtab    10   0\n.text     1420   0\n.data   0  0\nTotal 1430\n"
    assert parse_text_size(out) == 1420
    with pytest.raises(ValueError):
        parse_text_size("nothing")


def test_parse_cachegrind():
    out = "desc: I1 cache:\ncmd: ./a\nevents: Ir\nfl=x\n1 5\nsummary: 410031251\n"
    assert parse_cachegrind(out) == 410031251


def test_reference_text_and_fp_compare():
    assert reference_text(b"hi", 0) == b"hi\nexit 0\n"
    assert reference_text(b"", 3) == b"exit 3\n"
    assert fp_equal("x = 1.0001 y\nexit 0\n", "x = 1.0000 y\nexit 0\n", 0.001)
    assert not fp_equal("x = 1.01 y\n", "x = 1.00 y\n", 0.001)
    assert not fp_equal("x = 1.0 z\n", "x = 1.0 y\n", 0.001)
    assert not fp_equal("1 2\n", "1\n", 0.1)
