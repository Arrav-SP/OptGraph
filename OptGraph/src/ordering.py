"""Pass sequences built from an interaction graph (v3 Phase 5).

The graph gives, for one program,
    delta[B]      predicted (or measured) benefit of pass B alone, Δ(B | P)
    inter[(A,B)]  predicted (or measured) interaction I(A, B, P)
Both are improvements as fractions of the program's metric (positive = smaller).

The gain of appending B to a partial sequence S is estimated additively:
    first occurrence of B : delta[B] + Σ_{A in S} inter[(A, B)]
    repeat of B           : Σ_{A in S after B's last occurrence} inter[(A, B)]
A repeat earns only what the passes run since B's last run enabled for it; B's own
standalone effect has already been taken. A sequence stops early when no pass has a
positive estimated gain, so it may be shorter than the budget k.

Greedy is beam search with width 1.
"""
from __future__ import annotations


def step_gain(seq: tuple[str, ...], b: str, delta: dict, inter: dict) -> float:
    if b in seq:
        last = len(seq) - 1 - seq[::-1].index(b)
        return sum(inter.get((a, b), 0.0) for a in seq[last + 1:] if a != b)
    return delta.get(b, 0.0) + sum(inter.get((a, b), 0.0) for a in seq if a != b)


def beam_search(passes: list[str], delta: dict, inter: dict, k: int, width: int = 1,
                allow_repeats: bool = True) -> tuple[list[str], float]:
    """Best sequence of at most k passes by estimated total gain. Returns (sequence, gain)."""
    beams = [((), 0.0)]
    finished = []
    for _ in range(k):
        expanded = []
        for seq, total in beams:
            options = [(b, step_gain(seq, b, delta, inter)) for b in passes
                       if (allow_repeats or b not in seq) and (not seq or seq[-1] != b)]
            positive = [(b, g) for b, g in options if g > 0]
            if not positive:
                finished.append((seq, total))
                continue
            expanded += [(seq + (b,), total + g) for b, g in positive]
        if not expanded:
            break
        # Deterministic order: by gain, then lexicographically by sequence.
        expanded.sort(key=lambda s: (-s[1], s[0]))
        beams = expanded[:width]
    finished += beams
    seq, total = min(finished, key=lambda s: (-s[1], s[0]))
    return list(seq), float(total)


def greedy(passes: list[str], delta: dict, inter: dict, k: int,
           allow_repeats: bool = True) -> tuple[list[str], float]:
    return beam_search(passes, delta, inter, k, 1, allow_repeats)
