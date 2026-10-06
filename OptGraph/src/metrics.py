"""IR-level measurements by parsing textual LLVM IR.

Definitions (all are plain counts over function bodies; declarations, globals
and metadata are ignored):

  instruction_count      every instruction, terminators included
  basic_blocks           basic blocks in defined functions
  branches               `br` + `switch` instructions
  conditional_branches   `br i1 ...` + `switch`
  loads / stores / phis / calls   instructions with that opcode

Improvement of a metric (lower is treated as "smaller IR"):

  improvement = (baseline - optimized) / baseline      (positive = reduction)

When the baseline is 0 the ratio is undefined; `improvement` returns None and
callers must handle that explicitly (see interaction.py).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

METRICS = ["instruction_count", "basic_blocks", "branches", "conditional_branches",
           "loads", "stores", "phis", "calls"]

INT_ARITH = {"add", "sub", "mul", "udiv", "sdiv", "urem", "srem",
             "shl", "lshr", "ashr", "and", "or", "xor"}
FP_ARITH = {"fadd", "fsub", "fmul", "fdiv", "frem", "fneg"}
MEMORY = {"alloca", "load", "store", "getelementptr"}
CASTS = {"trunc", "zext", "sext", "fptrunc", "fpext", "fptoui", "fptosi", "uitofp",
         "sitofp", "ptrtoint", "inttoptr", "bitcast", "addrspacecast"}
CALL_PREFIXES = {"tail", "musttail", "notail"}

_LABEL = re.compile(r'^("[^"]*"|[-a-zA-Z$._0-9]+):')
_ASSIGN = re.compile(r'^(?:%"[^"]*"|%[-a-zA-Z$._0-9]+)\s*=\s*(.*)$')
_TARGET = re.compile(r'label %("[^"]*"|[-a-zA-Z$._0-9]+)')


@dataclass
class ParsedIR:
    opcodes: Counter = field(default_factory=Counter)
    functions: int = 0
    basic_blocks: int = 0
    conditional_branches: int = 0
    # function name -> {block name: [successor block names]}
    cfg: dict = field(default_factory=dict)

    @property
    def cfg_edges(self) -> int:
        return sum(len(set(s)) for blocks in self.cfg.values() for s in blocks.values())


def parse_ir(text: str) -> ParsedIR:
    out = ParsedIR()
    blocks = None          # CFG of the function being read, or None outside functions
    current = None         # current block name
    in_switch = False      # inside a multi-line switch case table

    for raw in text.splitlines():
        if blocks is None:
            if raw.startswith("define "):
                name = re.search(r'@("[^"]*"|[-a-zA-Z$._0-9]+)', raw)
                blocks = out.cfg.setdefault(name.group(1) if name else f"fn{out.functions}", {})
                out.functions += 1
                current = None
                in_switch = False
            continue
        if raw.startswith("}"):
            blocks = None
            continue
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if in_switch:
            blocks[current].extend(_TARGET.findall(line))
            if line.startswith("]") or line.endswith("]"):
                in_switch = False
            continue
        m = _LABEL.match(raw)
        if m:
            current = m.group(1)
            blocks.setdefault(current, [])
            out.basic_blocks += 1
            continue
        if current is None:
            # The entry block of a function has no printed label.
            current = "<entry>"
            blocks[current] = []
            out.basic_blocks += 1

        m = _ASSIGN.match(line)
        body = m.group(1) if m else line
        tokens = body.split(None, 2)
        opcode = tokens[0]
        if opcode in CALL_PREFIXES and len(tokens) > 1:
            opcode = tokens[1]
        out.opcodes[opcode] += 1

        if opcode == "br":
            blocks[current].extend(_TARGET.findall(body))
            if body.startswith("br i1"):
                out.conditional_branches += 1
        elif opcode == "switch":
            blocks[current].extend(_TARGET.findall(body))
            out.conditional_branches += 1
            if "[" in body and "]" not in body:
                in_switch = True
        elif opcode in ("indirectbr", "invoke", "callbr"):
            blocks[current].extend(_TARGET.findall(body))
    return out


def metrics_from_parsed(p: ParsedIR) -> dict:
    op = p.opcodes
    return {
        "instruction_count": sum(op.values()),
        "basic_blocks": p.basic_blocks,
        "branches": op["br"] + op["switch"],
        "conditional_branches": p.conditional_branches,
        "loads": op["load"],
        "stores": op["store"],
        "phis": op["phi"],
        "calls": op["call"] + op["invoke"],
    }


def measure_ir(path: Path) -> dict:
    """The METRICS for one IR file."""
    return metrics_from_parsed(parse_ir(Path(path).read_text()))


def improvement(baseline: float, optimized: float) -> float | None:
    """(baseline - optimized) / baseline, or None when baseline is 0 (undefined)."""
    if baseline == 0:
        return None
    return (baseline - optimized) / baseline
