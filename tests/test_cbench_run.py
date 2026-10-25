"""Multi-file programs with a command line (cBench style): IR linking, redirections, output files."""
import shutil
from pathlib import Path

import pytest

from src import llvm_pipeline as lp
from src.measure import build, parse_run_cmd, run_binary
from src.utils import Benchmark

needs_llvm = pytest.mark.skipif(not shutil.which("llvm-link"), reason="LLVM not installed")

MAIN = """#include <stdio.h>
int work(int);
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "w"); int c, n = 0;
  while ((c = getchar()) != EOF) n += c;
  fprintf(f, "%d\\n", work(n)); fclose(f);
  printf("stdout %s\\n", argv[2]); fprintf(stderr, "stderr\\n"); return 0;
}
"""
WORK = "int work(int x) { return x * 2 + 1; }\n"


def test_parse_run_cmd():
    assert parse_run_cmd("a b < in.txt > out 2> err") == (["a", "b"], "in.txt")
    assert parse_run_cmd("-x 'two words' > o") == (["-x", "two words"], None)


@needs_llvm
def test_multi_file_program_with_redirections(tmp_path):
    src = tmp_path / "prog"
    src.mkdir()
    (src / "main.c").write_text(MAIN)
    (src / "work.c").write_text(WORK)
    (tmp_path / "in.txt").write_text("AB")
    bench = Benchmark("t", "c", src / "main.c", "cbench", ("-w",), sources=(src / "main.c", src / "work.c"),
                      run_cmd=f"result.txt hello < {tmp_path / 'in.txt'} > ftmp_out", loop=3,
                      irflags=tuple(lp.IR_FLAGS_V3_1))
    tc = lp.Toolchain()
    ll = tmp_path / "p.ll"
    r = lp.program_ir(tc, bench, ll)
    assert r.ok, r.stderr
    assert "define dso_local i32 @work" in ll.read_text()      # both files in one module
    _, exe, err = build(tc, ll)
    assert not err
    run = run_binary(exe, 10, tmp_path, bench=bench)
    assert run.returncode == 0 and not run.error
    assert b"stdout hello" in run.output and b"stderr" in run.output
    assert b"FILE result.txt\n263\n" in run.output                # (65+66)*2+1, file captured
    assert run_binary(exe, 10, tmp_path, bench=bench).digest == run.digest
    assert not any(p.name.startswith("run_") for p in tmp_path.iterdir())   # run dir removed
