"""All interaction with Clang/LLVM lives here.

Command lines are built in one place (the `*_cmd` methods of Toolchain) so that
adapting to a different LLVM release means editing this file only.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path

from .utils import CommandResult, PassSpec, now_iso, run_cmd

# Unversioned names first, then versioned fallbacks (e.g. clang-18 on older Ubuntu).
_VERSION_SUFFIXES = [""] + [f"-{v}" for v in range(22, 13, -1)]

INSTALL_HINT = "sudo apt update && sudo apt install -y clang llvm libc6-dev"

SMOKE_SOURCE = "int main() {\n    int x = 10;\n    int y = x + 0;\n    return y;\n}\n"


class ToolchainError(RuntimeError):
    pass


def _find(tool: str) -> str | None:
    for suffix in _VERSION_SUFFIXES:
        path = shutil.which(tool + suffix)
        if path:
            return path
    return None


def _first_version(text: str) -> str | None:
    m = re.search(r"version\s+(\d+\.\d+(?:\.\d+)?)", text)
    return m.group(1) if m else None


class Toolchain:
    """Locates clang/opt/llc/llvm-config and builds their command lines."""

    def __init__(self):
        self.clang = _find("clang")
        self.clangxx = _find("clang++")
        self.opt = _find("opt")
        self.llc = _find("llc")
        self.llvm_config = _find("llvm-config")

    def missing(self) -> list[str]:
        return [n for n, p in (("clang", self.clang), ("opt", self.opt)) if not p]

    def require(self) -> "Toolchain":
        if self.missing():
            raise ToolchainError(
                f"required LLVM tool(s) not found on PATH: {', '.join(self.missing())}. "
                f"Install with: {INSTALL_HINT}")
        return self

    def versions(self) -> dict:
        def ver(tool, args=("--version",)):
            if not tool:
                return None
            r = run_cmd([tool, *args], timeout=20)
            if not r.ok:
                return None
            out = r.stdout.strip()
            return _first_version(out) or out.splitlines()[0]
        return {
            "clang_version": ver(self.clang),
            "opt_version": ver(self.opt),
            "llc_version": ver(self.llc),
            "llvm_config_version": ver(self.llvm_config),
        }

    # ---- command construction ------------------------------------------------

    def emit_ir_cmd(self, src: Path, out_ll: Path) -> list[str]:
        # -O0 keeps the IR close to the source. Clang normally tags -O0 functions
        # `optnone`, which makes every later opt pass skip them, so that is disabled.
        driver = self.clang if src.suffix == ".c" else (self.clangxx or self.clang)
        return [driver, "-O0", "-Xclang", "-disable-O0-optnone", "-S", "-emit-llvm",
                str(src), "-o", str(out_ll)]

    def opt_cmd(self, in_ll: Path, out_ll: Path, pipeline: str) -> list[str]:
        # New pass manager syntax (LLVM >= 13; the only syntax since LLVM 17).
        return [self.opt, f"-passes={pipeline}", "-S", str(in_ll), "-o", str(out_ll)]

    def verify_cmd(self, ll: Path) -> list[str]:
        return [self.opt, "-passes=verify", "-disable-output", str(ll)]

    def loops_cmd(self, ll: Path) -> list[str]:
        return [self.opt, "-passes=print<loops>", "-disable-output", str(ll)]

    def link_cmd(self, ll: Path, exe: Path) -> list[str]:
        return [self.clang, "-Wno-override-module", str(ll), "-o", str(exe), "-lm"]


# ---- operations ----------------------------------------------------------------

def compile_to_ir(tc: Toolchain, src: Path, out_ll: Path, timeout: float = 60) -> CommandResult:
    out_ll.parent.mkdir(parents=True, exist_ok=True)
    return run_cmd(tc.emit_ir_cmd(src, out_ll), timeout)


def verify_ir(tc: Toolchain, ll: Path, timeout: float = 60) -> CommandResult:
    return run_cmd(tc.verify_cmd(ll), timeout)


def run_pass(tc: Toolchain, in_ll: Path, out_ll: Path, spec: PassSpec,
             timeout: float = 60) -> CommandResult:
    """Apply one configured pass to in_ll, writing textual IR to out_ll."""
    res = run_cmd(tc.opt_cmd(in_ll, out_ll, spec.pipeline), timeout)
    if res.ok and not out_ll.exists():
        res.error = "opt reported success but wrote no output"
    return res


def compile_ir_to_exe(tc: Toolchain, ll: Path, exe: Path, timeout: float = 60) -> CommandResult:
    return run_cmd(tc.link_cmd(ll, exe), timeout)


def run_exe(exe: Path, timeout: float = 10) -> CommandResult:
    """Run a benchmark binary. A non-zero exit code is a valid program result here."""
    res = run_cmd([str(exe)], timeout)
    if res.error.startswith("exit code"):
        res.error = ""
    return res


def build_and_run(tc: Toolchain, ll: Path, workdir: Path) -> tuple[CommandResult, CommandResult | None]:
    exe = workdir / (ll.stem + ".bin")
    link = compile_ir_to_exe(tc, ll, exe)
    if not link.ok:
        return link, None
    return link, run_exe(exe)


def ir_body(text: str) -> str:
    """IR text without the header lines that only echo the file name."""
    return "\n".join(l for l in text.splitlines()
                     if not l.startswith(("; ModuleID", "source_filename")))


# ---- environment ------------------------------------------------------------------

def _os_release() -> dict:
    info = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v.strip('"')
    except OSError:
        pass
    return info


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def collect_environment(tc: Toolchain | None = None) -> dict:
    tc = tc or Toolchain()
    rel = _os_release()
    v = tc.versions()
    return {
        "os": platform.platform(),
        "system": platform.system(),
        "kernel": platform.release(),
        "is_wsl": is_wsl(),
        "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
        "distribution": rel.get("PRETTY_NAME"),
        "ubuntu_version": rel.get("VERSION_ID") if rel.get("ID") == "ubuntu" else None,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "clang_version": v["clang_version"],
        "llvm_version": v["llvm_config_version"] or v["opt_version"],
        "opt_version": v["opt_version"],
        "llvm_config_version": v["llvm_config_version"],
        "llc_version": v["llc_version"],
        "tool_paths": {"clang": tc.clang, "opt": tc.opt, "llc": tc.llc,
                       "llvm-config": tc.llvm_config},
        "timestamp": now_iso(),
    }


def smoke_test(tc: Toolchain) -> dict:
    """Compile a trivial program to IR, run one pass, and confirm the IR changed."""
    out = {"compiled": False, "pass_ran": False, "ir_changed": False, "error": ""}
    with tempfile.TemporaryDirectory(prefix="optgraph_smoke_") as d:
        d = Path(d)
        src, base, opt_ll = d / "tiny.c", d / "tiny.ll", d / "tiny.opt.ll"
        src.write_text(SMOKE_SOURCE)
        r = compile_to_ir(tc, src, base)
        if not r.ok:
            out["error"] = f"C -> LLVM IR failed: {r.error}\n{r.stderr.strip()}"
            return out
        out["compiled"] = True
        r = run_pass(tc, base, opt_ll, PassSpec("mem2reg+instcombine", "mem2reg,instcombine"))
        if not r.ok:
            out["error"] = f"opt failed: {r.error}\n{r.stderr.strip()}"
            return out
        out["pass_ran"] = True
        out["ir_changed"] = ir_body(base.read_text()) != ir_body(opt_ll.read_text())
        if not out["ir_changed"]:
            out["error"] = ("opt ran but left the IR unchanged; functions are probably "
                            "still marked optnone")
    return out


def validate_passes(tc: Toolchain, specs: list[PassSpec]) -> dict[str, str]:
    """Check each configured pipeline is accepted by the installed opt.

    Returns {pass name: error text}; empty when every pass is usable.
    """
    bad = {}
    with tempfile.TemporaryDirectory(prefix="optgraph_passes_") as d:
        d = Path(d)
        src, base = d / "tiny.c", d / "tiny.ll"
        src.write_text(SMOKE_SOURCE)
        r = compile_to_ir(tc, src, base)
        if not r.ok:
            return {s.name: "could not build the probe IR" for s in specs}
        for s in specs:
            r = run_pass(tc, base, d / "out.ll", s)
            if not r.ok:
                bad[s.name] = (r.stderr.strip() or r.error).splitlines()[0]
    return bad
