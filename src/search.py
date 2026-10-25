"""v3.1 ordering methods under a compile budget (Claim 4).

Budget unit (E): one evaluation = a sequence not evaluated before is turned into IR from the
program's start IR (one `opt` invocation) and its objective, the IR instruction count without
no-op intrinsics, is measured. Repeats of a sequence are cached and not counted. Separately
counted: every `opt` invocation (O, including state construction and feature extraction) and
wall-clock seconds. Each method records best-so-far after every evaluation; results are read at
CHECKPOINTS. Sequences have at most MAX_LEN passes and obey ordering.allowed (repeat cap).
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .ordering import REPEAT_CAP, allowed, beam_candidates, step_gain

CHECKPOINTS = (1, 5, 10, 25, 50, 100, 200)
MAX_LEN = 16


def repair(seq, passes_ok=None, cap: int = REPEAT_CAP, max_len: int = MAX_LEN) -> tuple:
    """Drop passes that break the cap or repeat the previous pass; truncate to max_len."""
    out = ()
    for b in seq:
        if (passes_ok is None or b in passes_ok) and allowed(out, b, cap):
            out += (b,)
        if len(out) == max_len:
            break
    return out


class Evaluator:
    """Budget accounting around an objective. `objective(seq) -> float | None` (None = failed);
    `state(seq) -> Path` gives the IR of a sequence (for re-prediction), `features(path) -> dict`."""

    def __init__(self, objective, budget: int = 200, state=None, features=None):
        self.objective, self.budget, self._state, self._features = objective, budget, state, features
        self.cache: dict[tuple, float | None] = {}
        self.E = self.O = 0
        self.t0 = time.perf_counter()
        self.trace: list[dict] = []           # one row per evaluation
        self.best: tuple[float, tuple] | None = None

    @property
    def exhausted(self) -> bool:
        return self.E >= self.budget

    def value(self, seq) -> float | None:
        seq = tuple(seq)
        if seq in self.cache:
            return self.cache[seq]
        if self.exhausted:
            return None
        v = self.objective(seq)
        self.O += 1
        self.E += 1
        self.cache[seq] = v
        if v is not None and (self.best is None or (v, len(seq)) < (self.best[0], len(self.best[1]))):
            self.best = (v, seq)
        self.trace.append({"E": self.E, "value": v, "best": self.best[0] if self.best else None,
                           "best_seq": ",".join(self.best[1]) if self.best else "",
                           "O": self.O, "seconds": time.perf_counter() - self.t0})
        return v

    def state(self, seq):
        self.O += 1
        return self._state(tuple(seq))

    def features(self, path):
        f, n_opt = self._features(path)
        self.O += n_opt
        return f

    def checkpoints(self) -> list[dict]:
        """Best-so-far at each budget; a method that stopped early keeps its final answer
        (the row's E shows how many evaluations it actually used)."""
        rows = []
        for c in CHECKPOINTS:
            upto = [t for t in self.trace if t["E"] <= c]
            if upto:
                rows.append({"checkpoint": c, **{k: upto[-1][k] for k in ("best", "best_seq", "E", "O", "seconds")}})
        return rows


# ---- model/graph-guided: ranked candidates, evaluated best first -------------------------

def ranked(ev: Evaluator, candidates) -> None:
    for seq, _ in candidates:
        if ev.exhausted:
            break
        ev.value(seq)


def graph_method(ev: Evaluator, passes, delta, inter, width: int = 16) -> None:
    """Predict-once, static-graph and oracle-graph orderings: additive estimate, capped beam."""
    ranked(ev, beam_candidates(passes, delta, inter, MAX_LEN, width))


def repredict(ev: Evaluator, passes, predict_delta, width: int = 16, depth: int = MAX_LEN) -> None:
    """Re-prediction: every beam state's next-pass scores come from the model applied to that
    state's own IR features (`predict_delta(features) -> {pass: Δ}`); children are built with
    one opt call each. All visited sequences are ranked by summed predicted Δ and evaluated
    best first."""
    beams, seen = [((), 0.0)], {}
    for _ in range(depth):
        children = []
        for seq, total in beams:
            pred = predict_delta(ev.features(ev.state(seq)))
            children += [(seq + (b,), total + pred.get(b, 0.0)) for b in passes if allowed(seq, b)]
        if not children:
            break
        children.sort(key=lambda s: (-s[1], s[0]))
        beams = children[:width]
        for seq, total in beams:
            seen.setdefault(seq, total)
    ranked(ev, sorted(seen.items(), key=lambda s: (-s[1], s[0])))


def graph_guided(ev: Evaluator, passes, delta, inter, m: int = 3) -> None:
    """Static-graph proposals verified by measurement: at each step evaluate the top-m passes by
    the graph's estimated gain and keep the measured best; stop when none improves."""
    seq, cur = (), ev.value(())
    while len(seq) < MAX_LEN and not ev.exhausted:
        opts = sorted((b for b in passes if allowed(seq, b)), key=lambda b: (-step_gain(seq, b, delta, inter), b))[:m]
        vals = [(ev.value(seq + (b,)), b) for b in opts]
        vals = [(v, b) for v, b in vals if v is not None]
        if not vals or min(vals)[0] >= cur:
            break
        cur, b = min(vals)
        seq += (b,)


def measured_greedy(ev: Evaluator, passes) -> None:
    """Upper-bound reference: try every allowed pass at every step, keep the measured best.
    Run with a large budget; its true evaluation count is what goes on the curve."""
    seq, cur = (), ev.value(())
    while len(seq) < MAX_LEN and not ev.exhausted:
        vals = [(ev.value(seq + (b,)), b) for b in passes if allowed(seq, b)]
        vals = [(v, b) for v, b in vals if v is not None]
        if not vals or min(vals)[0] >= cur:
            break
        cur, b = min(vals)
        seq += (b,)


# ---- autotuning baselines --------------------------------------------------------------

def random_sequence(passes, rng, length: int = MAX_LEN) -> tuple:
    # The cap bounds the longest legal sequence (cap * passes); never ask for more than that.
    length = min(length, REPEAT_CAP * len(passes))
    seq = ()
    while len(seq) < length:
        ok = [p for p in passes if allowed(seq, p)]
        if not ok:              # only the pass just used has uses left
            break
        seq += (ok[rng.integers(len(ok))],)
    return seq


def random_search(ev: Evaluator, passes, rng) -> None:
    for _ in range(50 * ev.budget):           # duplicates do not use budget; bounded anyway
        if ev.exhausted:
            break
        ev.value(random_sequence(passes, rng))


def genetic(ev: Evaluator, passes, rng, init=None, mutate_segment=None, pop: int = 20,
            elite: int = 2, p_mut: float = 0.3) -> None:
    """Steady GA: tournament(3), one-point crossover, mutation (replace / insert / delete one
    pass, or replace a segment via `mutate_segment` for the Pan-style variant), elitism."""
    init = init or (lambda: random_sequence(passes, rng))
    population = [repair(init(), passes) for _ in range(pop)]
    for _ in range(50 * ev.budget):
        if ev.exhausted:
            break
        scored = [(ev.value(s), s) for s in population]
        scored = sorted([(v, s) for v, s in scored if v is not None], key=lambda t: (t[0], t[1]))
        if not scored:
            population = [repair(init(), passes) for _ in range(pop)]
            continue
        nxt = [s for _, s in scored[:elite]]
        while len(nxt) < pop:
            def pick():
                cand = [scored[i] for i in rng.integers(len(scored), size=min(3, len(scored)))]
                return min(cand, key=lambda t: (t[0], t[1]))[1]
            a, b = pick(), pick()
            cut = int(rng.integers(0, min(len(a), len(b)) + 1))
            child = list(a[:cut] + b[cut:])
            if rng.random() < p_mut:
                if mutate_segment is not None and rng.random() < 0.5:
                    i = int(rng.integers(0, len(child) + 1))
                    child = child[:i] + list(mutate_segment(MAX_LEN - i))
                else:
                    op, i = rng.integers(3), int(rng.integers(0, max(len(child), 1)))
                    if op == 0 and child:
                        child[i] = passes[rng.integers(len(passes))]
                    elif op == 1:
                        child.insert(i, passes[rng.integers(len(passes))])
                    elif child:
                        del child[i]
            nxt.append(repair(child, passes))
        population = nxt


class _BudgetDone(Exception):
    pass


def opentuner_search(ev: Evaluator, alphabet: list[tuple], slots: int, seed: int) -> None:
    """OpenTuner (Ansel et al. 2014) over `slots` enum parameters, each an element of
    `alphabet` (tuples of passes, () = none); the configuration is their concatenation, repaired.
    Single passes as the alphabet = plain OpenTuner; DCO-style sub-sequences = DCO-style."""
    import opentuner
    from opentuner import ConfigurationManipulator, EnumParameter, MeasurementInterface, Result
    from opentuner.tuningrunmain import TuningRunMain
    logging.disable(logging.WARNING)
    ids = list(range(len(alphabet)))

    class Tuner(MeasurementInterface):
        def manipulator(self):
            m = ConfigurationManipulator()
            for i in range(slots):
                m.add_parameter(EnumParameter(f"s{i}", ids))
            return m

        def run(self, desired_result, input, limit):
            if ev.exhausted:
                raise _BudgetDone
            c = desired_result.configuration.data
            seq = repair(sum((alphabet[c[f"s{i}"]] for i in range(slots)), ()))
            v = ev.value(seq)
            return Result(time=float(v) if v is not None else 1e30)

        def save_final_config(self, configuration):
            pass

    random.seed(seed)
    np.random.seed(seed)
    args = argparse.ArgumentParser(parents=opentuner.argparsers()).parse_args(
        ["--test-limit", str(ev.budget * 5), "--database", "sqlite://", "--no-dups", "--parallelism", "1"])
    t = Tuner(args)
    try:
        TuningRunMain(t, args).main()
    except _BudgetDone:
        pass
    finally:
        logging.disable(logging.NOTSET)


def summarise(ev: Evaluator, **ident) -> pd.DataFrame:
    return pd.DataFrame([{**ident, **r} for r in ev.checkpoints()])
