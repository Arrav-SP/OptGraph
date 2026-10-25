"""v3: lower one IR file with the fixed backend, run it and measure it.

Every variant goes through the same steps, so the IR is the only difference:

    x.ll --clang BACKEND_FLAGS--> x.o --link with support objects--> x.exe

Metrics: `.text` size of x.o, executed instructions under cachegrind (deterministic),
and wall-clock runtime (warm-up + N timed runs pinned to one core). Program output is
stdout and stderr in one stream plus the exit code, as llvm-test-suite records it.
"""
from __future__ import annotations

import hashlib
import shlex
import shutil
import tempfile
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import llvm_pipeline as lp
from .metrics import parse_cachegrind, parse_text_size
from .utils import run_cmd


@dataclass
class RunResult:
    returncode: int
    output: bytes
    seconds: float
    error: str = ""

    @property
    def digest(self) -> str:
        return output_digest(self.returncode, self.output)


def output_digest(returncode: int, output: bytes) -> str:
    return hashlib.sha256(output + f"\nexit {returncode}\n".encode()).hexdigest()


def parse_run_cmd(cmd: str) -> tuple[list[str], str | None]:
    """Arguments and stdin file of a shell-style command line; `>`/`2>` targets are dropped
    (both streams are captured directly)."""
    toks, args, stdin = shlex.split(cmd), [], None
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("<", ">", "2>") and i + 1 < len(toks):
            if t == "<":
                stdin = toks[i + 1]
            i += 2
            continue
        args.append(t)
        i += 1
    return args, stdin


def run_binary(exe: Path, timeout: float = 60, cwd: Path | None = None,
               prefix: list[str] | None = None, env: dict | None = None, bench=None) -> RunResult:
    """Run once. A non-zero exit code is a result, not an error.

    Plain programs: no arguments, stdout+stderr captured together. A benchmark with a
    `run_cmd` (cBench) runs in a fresh directory holding links to its source directory's data
    files and its loop-wrap file; the output is stdout, stderr and every file it creates.
    """
    if bench is not None and bench.run_cmd:
        return _run_with_spec(exe, bench, timeout, cwd or exe.parent, prefix, env)
    cmd = [*(prefix or []), str(exe)]
    start = time.perf_counter()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=timeout, cwd=cwd, env=env)
        return RunResult(p.returncode, p.stdout, time.perf_counter() - start)
    except subprocess.TimeoutExpired:
        return RunResult(-1, b"", time.perf_counter() - start, f"timeout after {timeout}s")
    except OSError as e:
        return RunResult(-1, b"", 0.0, f"OSError: {e}")


def _run_with_spec(exe, bench, timeout, cwd, prefix, env) -> RunResult:
    args, stdin = parse_run_cmd(bench.run_cmd)
    rd = Path(tempfile.mkdtemp(prefix="run_", dir=cwd))
    try:
        src_dir = Path(bench.path).parent
        for f in src_dir.iterdir():         # data files the program opens from its cwd
            if f.is_file() and f.suffix not in (".c", ".h", ".o"):
                (rd / f.name).symlink_to(f)
        if bench.loop:
            (rd / "_finfo_dataset").write_text(f"{bench.loop}\n")
        before = {p.name for p in rd.iterdir()}
        start = time.perf_counter()
        try:
            with open(stdin, "rb") if stdin else open(os.devnull, "rb") as fin:
                p = subprocess.run([*(prefix or []), str(exe), *args], stdin=fin, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=timeout, cwd=rd, env=env)
        except subprocess.TimeoutExpired:
            return RunResult(-1, b"", time.perf_counter() - start, f"timeout after {timeout}s")
        except OSError as e:
            return RunResult(-1, b"", 0.0, f"OSError: {e}")
        secs = time.perf_counter() - start
        out = b"STDOUT\n" + p.stdout + b"\nSTDERR\n" + p.stderr
        for f in sorted(rd.iterdir()):
            if f.name not in before and f.is_file() and not f.is_symlink():
                out += b"\nFILE " + f.name.encode() + b"\n" + f.read_bytes()
        return RunResult(p.returncode, out, secs)
    finally:
        shutil.rmtree(rd, ignore_errors=True)


def build(tc: lp.Toolchain, ll: Path, support=(), timeout: float = 120) -> tuple[Path, Path, str]:
    """x.ll -> x.o -> x.exe next to x.ll. Returns (obj, exe, error text or '')."""
    obj, exe = ll.with_suffix(".o"), ll.with_suffix(".exe")
    r = run_cmd(tc.object_cmd(ll, obj), timeout)
    if not r.ok:
        return obj, exe, f"backend failed: {r.error}: {r.stderr.strip()[:500]}"
    r = run_cmd(tc.link_objects_cmd([obj, *support], exe), timeout)
    if not r.ok:
        return obj, exe, f"link failed: {r.error}: {r.stderr.strip()[:500]}"
    return obj, exe, ""


def text_size(tc: lp.Toolchain, obj: Path) -> int:
    r = run_cmd(tc.size_cmd(obj), 60)
    if not r.ok:
        raise RuntimeError(f"llvm-size failed: {r.error} {r.stderr.strip()}")
    return parse_text_size(r.stdout)


def dynamic_count(tc: lp.Toolchain, exe: Path, timeout: float = 1800, bench=None) -> int:
    """Executed instructions of the whole program (cachegrind, no cache simulation)."""
    out, log = exe.with_suffix(".cg.out").resolve(), exe.with_suffix(".cg.log").resolve()
    prefix = tc.cachegrind_cmd(exe, out, log)[:-1]
    r = run_binary(exe, timeout, exe.parent, prefix, bench=bench)
    try:
        if r.error:
            raise RuntimeError(f"cachegrind failed: {r.error}")
        return parse_cachegrind(out.read_text())
    finally:
        out.unlink(missing_ok=True)
        log.unlink(missing_ok=True)


def timed_runs(exe: Path, runs: int = 10, warmup: int = 1, core: int | None = None,
               timeout: float = 60, bench=None) -> tuple[list[float], str]:
    """Wall-clock seconds of `runs` timed executions after `warmup` untimed ones.

    Returns (times, error). Pinned with `taskset -c core` when a core is given.
    """
    prefix = ["taskset", "-c", str(core)] if core is not None else None
    times = []
    for i in range(warmup + runs):
        r = run_binary(exe, timeout, exe.parent, prefix, bench=bench)
        if r.error:
            return times, r.error
        if i >= warmup:
            times.append(r.seconds)
    return times, ""


# ---- one IR file -> every v3 metric, cached by IR content --------------------------

def ir_hash(text: str) -> str:
    return hashlib.sha256(lp.ir_body(text).encode()).hexdigest()


def measure_variant(tc: lp.Toolchain, ll: Path, bench, expected_digest: str,
                    cache: dict, dynamic: bool, check_output: bool = True,
                    timeout: float = 60) -> dict:
    """Static IR metrics, .text size, output check and (optionally) dynamic count of one IR.

    `cache` maps IR hash -> previously measured values. Identical IR lowers to an identical
    object, so reusing its values is exact. Raises nothing: failures go into 'error'.
    """
    from .metrics import measure_ir
    text = ll.read_text()
    key = ir_hash(text)
    out = {"ir_sha256": key, **measure_ir(ll)}
    hit = cache.get(key)
    if hit is not None and (not dynamic or hit.get("dyn_instructions") is not None
                            or hit.get("output_matches") is False):
        return {**out, **hit}
    res = dict(hit or {})
    obj, exe, err = build(tc, ll, bench.support)
    try:
        if err:
            return {**out, "error": err}
        res["text_size"] = text_size(tc, obj)
        if check_output and "output_matches" not in res:
            run = run_binary(exe, timeout, ll.parent, bench=bench)
            res["output_matches"] = (not run.error) and run.digest == expected_digest
            if run.error:
                res["error"] = f"run failed: {run.error}"
            elif not res["output_matches"]:
                res["error"] = "program output differs from baseline"
        if dynamic and res.get("output_matches", True) and res.get("dyn_instructions") is None:
            res["dyn_instructions"] = dynamic_count(tc, exe, timeout * 50, bench)
        res.setdefault("error", "")
        cache[key] = res
        return {**out, **res}
    except Exception as e:  # noqa: BLE001 - a measurement failure is recorded, not raised
        return {**out, **res, "error": f"measurement failed: {e}"}
    finally:
        obj.unlink(missing_ok=True)
        exe.unlink(missing_ok=True)
