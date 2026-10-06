"""Structural features of a program, taken from its ORIGINAL (baseline) LLVM IR.

These are the only program-side inputs the ML model sees. Nothing here may look
at IR produced by the passes under study.
"""
from __future__ import annotations

import re
from pathlib import Path

from .llvm_pipeline import Toolchain
from .metrics import CASTS, FP_ARITH, INT_ARITH, MEMORY, ParsedIR, metrics_from_parsed, parse_ir
from .utils import run_cmd

COUNT_FEATURES = [
    "instruction_count", "basic_blocks", "functions", "branches", "conditional_branches",
    "loads", "stores", "calls", "phis", "integer_ops", "floating_ops", "arithmetic_ops",
    "memory_ops", "returns", "cfg_edges", "loop_count", "loop_depth",
    "allocas", "geps", "compares", "casts",
]
RATIO_FEATURES = [
    "load_frac", "store_frac", "branch_frac", "call_frac", "arith_frac", "memory_frac",
    "float_frac", "avg_block_size", "cyclomatic",
]
FEATURES = COUNT_FEATURES + RATIO_FEATURES


def _back_edges(blocks: dict) -> int:
    """Back edges found by an iterative DFS from the entry block."""
    if not blocks:
        return 0
    entry = next(iter(blocks))
    state = {entry: 1}                  # 1 = on stack, 2 = finished
    stack = [(entry, iter(blocks.get(entry, [])))]
    count = 0
    while stack:
        node, it = stack[-1]
        for succ in it:
            s = state.get(succ, 0)
            if s == 1:
                count += 1
            elif s == 0:
                state[succ] = 1
                stack.append((succ, iter(blocks.get(succ, []))))
                break
        else:
            state[node] = 2
            stack.pop()
    return count


def loop_info(ll: Path, parsed: ParsedIR, tc: Toolchain | None) -> tuple[int, int, str]:
    """(loop count, maximum nesting depth, method used).

    Preferred: LLVM's own LoopInfo via `opt -passes=print<loops>`, which only
    reads the IR. Fallback: count CFG back edges in the parsed text (depth is
    then only known to be >= 1).
    """
    if tc and tc.opt:
        r = run_cmd(tc.loops_cmd(ll), timeout=60)
        if r.ok:
            depths = [int(d) for d in re.findall(r"Loop at depth (\d+)", r.stdout + r.stderr)]
            return len(depths), max(depths, default=0), "llvm_loopinfo"
    n = sum(_back_edges(b) for b in parsed.cfg.values())
    return n, (1 if n else 0), "cfg_back_edges"


def extract_features(ll: Path, tc: Toolchain | None = None) -> dict:
    parsed = parse_ir(Path(ll).read_text())
    op = parsed.opcodes
    f = metrics_from_parsed(parsed)
    loops, depth, method = loop_info(ll, parsed, tc)
    int_arith = sum(op[o] for o in INT_ARITH)
    fp_arith = sum(op[o] for o in FP_ARITH)
    f.update({
        "functions": parsed.functions,
        "integer_ops": int_arith + op["icmp"],
        "floating_ops": fp_arith + op["fcmp"],
        "arithmetic_ops": int_arith + fp_arith,
        "memory_ops": sum(op[o] for o in MEMORY),
        "returns": op["ret"],
        "cfg_edges": parsed.cfg_edges,
        "loop_count": loops,
        "loop_depth": depth,
        "allocas": op["alloca"],
        "geps": op["getelementptr"],
        "compares": op["icmp"] + op["fcmp"],
        "casts": sum(op[o] for o in CASTS),
    })
    n = max(f["instruction_count"], 1)
    f.update({
        "load_frac": f["loads"] / n,
        "store_frac": f["stores"] / n,
        "branch_frac": f["branches"] / n,
        "call_frac": f["calls"] / n,
        "arith_frac": f["arithmetic_ops"] / n,
        "memory_frac": f["memory_ops"] / n,
        "float_frac": f["floating_ops"] / n,
        "avg_block_size": f["instruction_count"] / max(f["basic_blocks"], 1),
        # McCabe: E - N + 2 per function, summed.
        "cyclomatic": f["cfg_edges"] - f["basic_blocks"] + 2 * f["functions"],
    })
    out = {k: f[k] for k in FEATURES}
    out["loop_method"] = method
    return out
