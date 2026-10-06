"""Shared paths, configuration loading and subprocess handling."""
from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
CONFIG_PATH = ROOT / "config" / "passes.yaml"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
EXPERIMENT_DIR = ROOT / "experiments"

SOURCE_SUFFIXES = (".c", ".cpp", ".cc", ".cxx")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CommandResult:
    """Outcome of one external tool invocation. Never raises on tool failure."""
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timestamp: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.error

    def to_dict(self) -> dict:
        return asdict(self)


def run_cmd(cmd: list[str], timeout: float = 60, cwd: Path | None = None) -> CommandResult:
    """Run a command, capturing output. Failures are reported in the result."""
    text = shlex.join(str(c) for c in cmd)
    stamp = now_iso()
    start = time.perf_counter()
    try:
        # stdin is closed so a benchmark that tries to read input gets EOF instead of
        # blocking on the terminal.
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                           timeout=timeout, cwd=cwd, errors="replace",
                           stdin=subprocess.DEVNULL)
        return CommandResult(text, p.returncode, p.stdout, p.stderr,
                             time.perf_counter() - start, stamp,
                             "" if p.returncode == 0 else f"exit code {p.returncode}")
    except subprocess.TimeoutExpired:
        return CommandResult(text, -1, "", "", time.perf_counter() - start, stamp,
                             f"timeout after {timeout}s")
    except FileNotFoundError:
        return CommandResult(text, -1, "", "", 0.0, stamp, f"tool not found: {cmd[0]}")
    except OSError as e:
        return CommandResult(text, -1, "", "", time.perf_counter() - start, stamp, f"OSError: {e}")


@dataclass(frozen=True)
class PassSpec:
    name: str
    pipeline: str
    kind: str = ""


@dataclass(frozen=True)
class Benchmark:
    program_id: str
    category: str
    path: Path


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("settings", {})
    cfg.setdefault("quick", {})
    return cfg


def load_passes(cfg: dict | None = None, names: list[str] | None = None) -> list[PassSpec]:
    cfg = cfg or load_config()
    specs = [PassSpec(p["name"], p.get("pipeline", p["name"]), p.get("kind", ""))
             for p in cfg["passes"]]
    if names is None:
        return specs
    by_name = {s.name: s for s in specs}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown pass(es) {unknown}; configured: {sorted(by_name)}")
    return [by_name[n] for n in names]


def discover_benchmarks(names: list[str] | None = None) -> list[Benchmark]:
    """Every source file under benchmarks/<category>/; program_id is the file stem."""
    found = [Benchmark(p.stem, p.parent.name, p)
             for p in sorted(BENCH_DIR.glob("*/*")) if p.suffix in SOURCE_SUFFIXES]
    ids = [b.program_id for b in found]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark program ids are not unique")
    if names is None:
        return found
    by_id = {b.program_id: b for b in found}
    unknown = [n for n in names if n not in by_id]
    if unknown:
        raise ValueError(f"unknown program(s) {unknown}")
    return [by_id[n] for n in names]


@dataclass(frozen=True)
class Workspace:
    """Where one experiment run (full / quick / demo) keeps its files."""
    tag: str

    @property
    def data(self) -> Path:
        return DATA_DIR if self.tag == "full" else DATA_DIR / self.tag

    @property
    def experiments(self) -> Path:
        return EXPERIMENT_DIR / self.tag

    @property
    def outputs(self) -> Path:
        return OUTPUT_DIR if self.tag == "full" else OUTPUT_DIR / self.tag

    @property
    def measurements(self) -> Path:
        return self.data / "results" / "measurements.csv"

    @property
    def experiment_log(self) -> Path:
        return self.data / "results" / "experiment_log.csv"

    @property
    def run_config(self) -> Path:
        return self.data / "results" / "run_config.json"

    @property
    def features(self) -> Path:
        return self.data / "program_features.csv"

    @property
    def dataset(self) -> Path:
        return self.data / "interaction_dataset.csv"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")
