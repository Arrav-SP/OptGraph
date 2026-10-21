"""v3: lower one IR file with the fixed backend, run it and measure it.

Every variant goes through the same steps, so the IR is the only difference:

    x.ll --clang BACKEND_FLAGS--> x.o --link with support objects--> x.exe

Metrics: `.text` size of x.o, executed instructions under cachegrind (deterministic),
and wall-clock runtime (warm-up + N timed runs pinned to one core). Program output is
stdout and stderr in one stream plus the exit code, as llvm-test-suite records it.
"""
from __future__ import annotations

import hashlib
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


def run_binary(exe: Path, timeout: float = 60, cwd: Path | None = None,
               prefix: list[str] | None = None, env: dict | None = None) -> RunResult:
    """Run once; stdout+stderr captured together. A non-zero exit code is a result, not an error."""
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


def dynamic_count(tc: lp.Toolchain, exe: Path, timeout: float = 1800) -> int:
    """Executed instructions of the whole program (cachegrind, no cache simulation)."""
    out, log = exe.with_suffix(".cg.out"), exe.with_suffix(".cg.log")
    r = run_cmd(tc.cachegrind_cmd(exe, out, log), timeout, cwd=exe.parent)
    try:
        if r.error and not r.error.startswith("exit code"):
            raise RuntimeError(f"cachegrind failed: {r.error}")
        return parse_cachegrind(out.read_text())
    finally:
        out.unlink(missing_ok=True)
        log.unlink(missing_ok=True)


def timed_runs(exe: Path, runs: int = 10, warmup: int = 1, core: int | None = None,
               timeout: float = 60) -> tuple[list[float], str]:
    """Wall-clock seconds of `runs` timed executions after `warmup` untimed ones.

    Returns (times, error). Pinned with `taskset -c core` when a core is given.
    """
    prefix = ["taskset", "-c", str(core)] if core is not None else None
    times = []
    for i in range(warmup + runs):
        r = run_binary(exe, timeout, exe.parent, prefix)
        if r.error:
            return times, r.error
        if i >= warmup:
            times.append(r.seconds)
    return times, ""


# ---- one IR file -> every v3 metric, cached by IR content --------------------------

def ir_hash(text: str) -> str:
    return hashlib.sha256(lp.ir_body(text).encode()).hexdigest()


def measure_variant(tc: lp.Toolchain, ll: Path, support, expected_digest: str,
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
    obj, exe, err = build(tc, ll, support)
    try:
        if err:
            return {**out, "error": err}
        res["text_size"] = text_size(tc, obj)
        if check_output and "output_matches" not in res:
            run = run_binary(exe, timeout, ll.parent)
            res["output_matches"] = (not run.error) and run.digest == expected_digest
            if run.error:
                res["error"] = f"run failed: {run.error}"
            elif not res["output_matches"]:
                res["error"] = "program output differs from baseline"
        if dynamic and res.get("output_matches", True) and res.get("dyn_instructions") is None:
            res["dyn_instructions"] = dynamic_count(tc, exe, timeout * 50)
        res.setdefault("error", "")
        cache[key] = res
        return {**out, **res}
    except Exception as e:  # noqa: BLE001 - a measurement failure is recorded, not raised
        return {**out, **res, "error": f"measurement failed: {e}"}
    finally:
        obj.unlink(missing_ok=True)
        exe.unlink(missing_ok=True)
