"""v3 benchmark suites: candidates, screening and the manifest (data/v3/benchmarks.csv).

Suites
  optgraph   the 24 v2 programs in benchmarks/
  polybench  PolyBench/C 4.2.1 (official release). The kernel module is what we optimise;
             utilities/polybench.c is compiled once with fixed flags and linked as is.
  testsuite  llvm-test-suite SingleSource/Benchmarks, C files of the configured directories.

A candidate is kept only if it compiles to IR, builds with the fixed backend, runs to the
same output twice, its -O0 runtime is inside the configured window, it matches its
reference output (test-suite), and an ASan+UBSan build of its source runs cleanly with
the same output. The reason for every drop is recorded.
"""
from __future__ import annotations

import hashlib
import os
import re
import shlex
from dataclasses import replace
from pathlib import Path

from . import llvm_pipeline as lp
from .measure import build, run_binary
from .utils import ROOT, Benchmark, discover_benchmarks, rel_path, run_cmd, version, work_dir

_NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)", re.I)


def polybench_support(tc: lp.Toolchain, cfg: dict) -> Path:
    """Compile utilities/polybench.c once (cached in the work directory)."""
    pb = work_dir(cfg) / cfg["benchmarks"]["polybench_dir"]
    obj = work_dir(cfg) / "support" / "polybench.o"
    if not obj.exists():
        obj.parent.mkdir(parents=True, exist_ok=True)
        r = run_cmd(tc.support_cmd(pb / "utilities" / "polybench.c", obj,
                                   ["-I", str(pb / "utilities")]), 120)
        if not r.ok:
            raise RuntimeError(f"polybench.c failed to compile: {r.stderr}")
    return obj


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def candidates(tc: lp.Toolchain, cfg: dict) -> list[Benchmark]:
    out = _candidates(tc, cfg)
    irflags = tuple(cfg.get("ir_flags", ()))
    return [replace(c, irflags=irflags) for c in out] if irflags else out


def _candidates(tc: lp.Toolchain, cfg: dict) -> list[Benchmark]:
    b = cfg["benchmarks"]
    out = list(discover_benchmarks())
    pb = work_dir(cfg) / b["polybench_dir"]
    support = (polybench_support(tc, cfg),)
    # The official list of the 30 kernels (the release also ships e.g. Nussinov.orig.c).
    for line in (pb / "utilities" / "benchmark_list").read_text().split():
        src = pb / line
        group = src.relative_to(pb).parts[:-2]          # e.g. linear-algebra/blas
        out.append(Benchmark(f"pb_{_slug(src.stem)}", "polybench_" + _slug(group[-1]), src,
                             "polybench", ("-I", str(pb / "utilities"), "-DPOLYBENCH_DUMP_ARRAYS"),
                             support))
    ts = work_dir(cfg) / b["test_suite_dir"]
    for sub in b["test_suite_subdirs"]:
        for src in sorted((ts / sub).glob("*.c")):
            rel = f"{sub}/{src.name}"
            if rel in b["test_suite_exclude"]:
                continue
            flags = ["-w", *b["test_suite_flags"].get(sub, []), *b["test_suite_flags"].get(rel, [])]
            out.append(Benchmark(f"ts_{_slug(sub)}_{_slug(src.stem)}", "testsuite_" + _slug(sub.split("/")[0]),
                                 src, "testsuite", tuple(flags)))
    out += cbench_candidates(cfg)
    ids = [c.program_id for c in out]
    assert len(ids) == len(set(ids)), "candidate program ids are not unique"
    return out


def cbench_candidates(cfg: dict) -> list[Benchmark]:
    """cBench V1.1 programs (config `cbench_dir`): all .c files of src/ form one module; the
    Makefile.llvm compile line gives -D/-I flags; dataset 1 of _ccc_info_datasets gives the
    command line, with its ../../ data paths made absolute."""
    b = cfg["benchmarks"]
    if not b.get("cbench_dir"):
        return []
    root = work_dir(cfg) / b["cbench_dir"]
    out = []
    for info in sorted(root.glob("*/src/_ccc_info_datasets")):
        src, name = info.parent, info.parent.parent.name
        lines = info.read_text().splitlines()
        ds1 = lines.index("1", lines.index("====="))          # "=====", "1", <cmd>, <loop>
        cmd = lines[ds1 + 1].replace("../../", f"{root}/")
        make = (src / "Makefile.llvm").read_text()
        line = next(l for l in make.splitlines() if " -c " in l and "*.c" in l)
        flags = [("-I" + str(src) if t in ("-I.", "-I./") else t)
                 for t in shlex.split(line.split("$(CCC_OPTS_ADD)")[-1]) if t.startswith(("-D", "-I"))]
        skip = set(b.get("cbench_exclude_files", {}).get(name, []))
        sources = tuple(f for f in sorted(src.glob("*.c")) if f.name not in skip)
        group = name.split("_")[0] if "_" in name else name.rstrip("de")
        flags = (*b["cbench_flags"], *flags, *b.get("cbench_program_flags", {}).get(name, []))
        # `support` holds what is linked unchanged with every variant; for cBench that is only
        # system libraries a program needs (e.g. -lz), passed to the linker as given.
        libs = tuple(Path(x) for x in b.get("cbench_link_libs", {}).get(name, []))
        out.append(Benchmark(f"cb_{_slug(name)}", "cbench_" + _slug(group), sources[0], "cbench",
                             tuple(flags), support=libs, sources=sources, run_cmd=cmd))
    return out


# ---- reference output (llvm-test-suite conventions) ----------------------------------

def reference_text(output: bytes, returncode: int) -> bytes:
    """What the test-suite stores: stdout+stderr, then 'exit N' on its own line."""
    if output and not output.endswith(b"\n"):
        output += b"\n"
    return output + f"exit {returncode}\n".encode()


def fp_equal(actual: str, expected: str, rel_tol: float) -> bool:
    """fpcmp-style comparison: text must match; numbers may differ by rel_tol (relative)."""
    a_tok, e_tok = _NUM.split(actual), _NUM.split(expected)
    a_num, e_num = _NUM.findall(actual), _NUM.findall(expected)
    if len(a_num) != len(e_num) or [t.split() for t in a_tok] != [t.split() for t in e_tok]:
        return False
    for x, y in zip(a_num, e_num):
        fx, fy = float(x), float(y)
        if fx == fy or (fx != fx and fy != fy):
            continue
        if abs(fx - fy) > rel_tol * max(abs(fx), abs(fy)):
            return False
    return True


def matches_reference(bench: Benchmark, output: bytes, returncode: int, cfg: dict) -> tuple[bool, str]:
    small = "-DSMALL_PROBLEM_SIZE" in bench.cflags
    ref_path = bench.path.with_suffix(".reference_output" + (".small" if small else ""))
    if not ref_path.exists():
        return False, "no .reference_output"
    ref = ref_path.read_bytes()
    got = reference_text(output, returncode)
    if got == ref or hashlib.md5(got).hexdigest().encode() == ref.strip():
        return True, "exact"
    sub = bench.path.parent.relative_to(work_dir(cfg) / cfg["benchmarks"]["test_suite_dir"]).as_posix()
    tol = cfg["benchmarks"]["test_suite_fp_tolerance"].get(sub)
    if tol and fp_equal(got.decode(errors="replace"), ref.decode(errors="replace"), tol):
        return True, f"fp tolerance {tol}"
    return False, "output differs from .reference_output"


# ---- screening -----------------------------------------------------------------------

def screen(tc: lp.Toolchain, bench: Benchmark, cfg: dict) -> dict:
    """Decide whether a candidate is usable. Returns one manifest row."""
    b = cfg["benchmarks"]
    wd = work_dir(cfg) / ("screen" if version(cfg) == "v3" else f"{version(cfg)}/screen") / bench.program_id
    wd.mkdir(parents=True, exist_ok=True)
    row = {"program_id": bench.program_id, "suite": bench.suite, "category": bench.category,
           "source": rel_path(bench.path), "compile_flags": " ".join(bench.cflags),
           "ir_flags": " ".join(bench.irflags),
           "support": " ".join(str(s) for s in bench.support), "dataset": "",
           "runtime_o0_s": None, "output_sha256": "", "reference_check": "", "kept": False,
           "drop_reason": "", "sources": " ".join(str(s) for s in bench.sources),
           "run_cmd": bench.run_cmd, "loop_wrap": bench.loop}
    # PolyBench: smallest dataset reaching min_runtime_s. test-suite: the default size, then the
    # suite's own SMALL_PROBLEM_SIZE (with its .reference_output.small) if the default is too slow.
    datasets = b["polybench_datasets"] if bench.suite == "polybench" else [""]
    if bench.suite == "testsuite" and bench.path.with_suffix(".reference_output.small").exists():
        datasets = ["", "SMALL_PROBLEM_SIZE"]
    max_rt = {"polybench": b["polybench_max_runtime_s"],
              "testsuite": b["test_suite_max_runtime_s"],
              "cbench": b.get("cbench_max_runtime_s")}.get(bench.suite)
    timeout = b["run_timeout_s"]

    run = None
    base = bench
    for ds in datasets:
        flag = ("-D" + ds if ds == "SMALL_PROBLEM_SIZE" else f"-D{ds}_DATASET") if ds else None
        cand = replace(base, cflags=base.cflags + ((flag,) if flag else ()))
        ll = wd / f"baseline{'_' + ds if ds else ''}.ll"
        r = lp.program_ir(tc, cand, ll, 120)
        if not r.ok:
            row["drop_reason"] = f"compile to IR failed: {r.stderr.strip()[:300]}"
            return row
        _, exe, err = build(tc, ll, cand.support)
        if err:
            row["drop_reason"] = err
            return row
        if cand.suite == "cbench":
            # Loop-wrap count chosen so one run takes about cbench_target_runtime_s at the start IR.
            # Re-measured after each scaling step: a single short run is mostly process start-up.
            target, loop = b["cbench_target_runtime_s"], 1
            for _ in range(8):
                one = run_binary(exe, timeout, wd, bench=replace(cand, loop=loop))
                if one.error:
                    row["drop_reason"] = f"-O0 run failed: {one.error}"
                    return row
                if one.seconds >= 0.8 * target or loop >= 10_000_000:
                    break
                loop = min(10_000_000, max(2 * loop, round(loop * target / max(one.seconds, 1e-4))))
            cand = replace(cand, loop=loop)
            row["loop_wrap"] = cand.loop
        run, second = run_binary(exe, timeout, wd, bench=cand), None
        if not run.error:
            second = run_binary(exe, timeout, wd, bench=cand)
        if run.error or second.error:
            row["drop_reason"] = f"-O0 run failed: {run.error or second.error}"
            return row
        bench, row["dataset"] = cand, ds
        row["runtime_o0_s"] = round(min(run.seconds, second.seconds), 4)
        if bench.suite == "polybench" and row["runtime_o0_s"] >= b["min_runtime_s"]:
            break
        if bench.suite == "testsuite" and row["runtime_o0_s"] <= max_rt:
            break
    row["compile_flags"] = " ".join(bench.cflags)
    row["output_sha256"] = run.digest
    if second.digest != run.digest:
        row["drop_reason"] = "non-deterministic output (two runs differ)"
        return row
    if bench.suite == "testsuite":
        ok, how = matches_reference(bench, run.output, run.returncode, cfg)
        row["reference_check"] = how
        if not ok:
            row["drop_reason"] = how
            return row
    elif run.returncode != 0:
        row["drop_reason"] = f"exit code {run.returncode}"
        return row
    rt = row["runtime_o0_s"]
    # Runtime statistics only use programs long enough to time (dyn. count is exact for all).
    row["runtime_reliable"] = rt >= b["min_runtime_s"]
    if bench.suite == "polybench" and rt < b["min_runtime_s"]:
        row["reference_check"] = "largest dataset still below min_runtime_s; kept, runtime not used"
    if bench.suite != "optgraph":
        if bench.suite == "testsuite" and rt < b["min_runtime_s"]:
            if not b.get("keep_fast"):
                row["drop_reason"] = f"-O0 runtime {rt}s below {b['min_runtime_s']}s"
                return row
            row["reference_check"] += "; below min_runtime_s: kept, runtime not used"
        if rt > max_rt:
            row["drop_reason"] = f"-O0 runtime {rt}s above {max_rt}s"
            return row

    # Sanitizers on the source (support sources too, so their calls are checked).
    srcs = list(bench.sources or [bench.path]) + [s for s in bench.support if str(s).startswith("-l")]
    if bench.suite == "polybench":
        srcs.append(work_dir(cfg) / b["polybench_dir"] / "utilities" / "polybench.c")
    san = wd / "sanitized.exe"
    r = run_cmd(tc.sanitizer_cmd(srcs, san, bench.cflags), 300)
    if not r.ok:
        row["drop_reason"] = f"sanitizer build failed: {r.stderr.strip()[:300]}"
        return row
    # Leaks are not undefined behaviour; several benchmarks never free before exit.
    s = run_binary(san, timeout * 10, wd, env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"},
                   bench=bench)
    if s.error or s.returncode != run.returncode or b"runtime error" in s.output \
            or b"Sanitizer" in s.output:
        detail = s.error or next((l for l in s.output.decode(errors="replace").splitlines()
                                  if "runtime error" in l or "Sanitizer" in l), f"exit {s.returncode}")
        row["drop_reason"] = f"ASan/UBSan: {detail.strip()[:300]}"
        return row
    if s.digest != run.digest:
        row["drop_reason"] = "sanitized build output differs from -O0 build"
        return row
    row["kept"] = True
    return row


def load_manifest(path: Path, programs: list[str] | None = None) -> list[Benchmark]:
    """Kept programs of data/v3/benchmarks.csv as Benchmarks (flags include the dataset)."""
    import pandas as pd
    m = pd.read_csv(path, keep_default_na=False)
    m = m[m["kept"].astype(str) == "True"]
    if programs is not None:
        unknown = sorted(set(programs) - set(m["program_id"]))
        if unknown:
            raise ValueError(f"not in the kept manifest: {unknown}")
        m = m.set_index("program_id").loc[programs].reset_index()
    return [Benchmark(r.program_id, r.category, ROOT / r.source, r.suite,
                      tuple(r.compile_flags.split()), tuple(Path(s) for s in r.support.split()),
                      tuple(getattr(r, "ir_flags", "").split()),
                      tuple(Path(s) for s in getattr(r, "sources", "").split()),
                      getattr(r, "run_cmd", ""), int(getattr(r, "loop_wrap", 0) or 0))
            for r in m.itertuples()]
