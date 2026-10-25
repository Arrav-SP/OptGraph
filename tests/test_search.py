"""Ordering methods on a toy objective (no LLVM): budget accounting, caps, determinism."""
import numpy as np
import pytest

from src import search as se
from src.ordering import REPEAT_CAP, beam_candidates

PASSES = ["a", "b", "c", "d"]
GAIN = {"a": 5, "b": 3, "c": 1, "d": 0}
BONUS = {("a", "b"): 4, ("b", "c"): 2}


def objective(seq):
    """100 minus gains; first use of a pass gives GAIN, a pair adjacent in order gives BONUS."""
    v, used = 100.0, set()
    for i, p in enumerate(seq):
        if p not in used:
            v -= GAIN[p]
            used.add(p)
        if i and (seq[i - 1], p) in BONUS:
            v -= BONUS[(seq[i - 1], p)]
    return v


def test_budget_cache_and_checkpoints():
    ev = se.Evaluator(objective, budget=10)
    for s in [("a",), ("a",), ("b",), ("a", "b")]:
        ev.value(s)
    assert ev.E == 3 and ev.best == (88.0, ("a", "b"))
    rows = ev.checkpoints()
    assert [r["checkpoint"] for r in rows] == list(se.CHECKPOINTS) and rows[0]["best"] == 95.0
    assert rows[-1]["E"] == 3 and rows[-1]["best"] == 88.0      # stopped early: final answer kept
    while not ev.exhausted:
        ev.value(se.random_sequence(PASSES, np.random.default_rng(ev.E)))
    assert ev.E == 10 and ev.value(("c", "d", "a", "b", "c")) is None


def test_repair_enforces_cap_and_no_immediate_repeat():
    assert se.repair(("a", "a", "b", "a", "b", "a")) == ("a", "b", "a", "b")[:4]
    s = se.repair(tuple("ab" * 20))
    assert len(s) == 2 * REPEAT_CAP and all(s.count(p) <= REPEAT_CAP for p in s)


def test_beam_candidates_ranked_and_capped():
    delta = {p: g / 100 for p, g in GAIN.items()}
    inter = {k: v / 100 for k, v in BONUS.items()}
    cands = beam_candidates(PASSES, delta, inter, 4, width=8)
    score = dict(cands)
    assert score[("a", "b")] > score[("b", "a")] and all(s.count(p) <= REPEAT_CAP for s, _ in cands for p in s)
    assert [c[1] for c in cands] == sorted((c[1] for c in cands), reverse=True)


@pytest.mark.parametrize("method", ["random", "ga"])
def test_stochastic_methods_respect_budget_and_seed(method):
    def run(seed):
        ev = se.Evaluator(objective, budget=25)
        rng = np.random.default_rng(seed)
        se.random_search(ev, PASSES, rng) if method == "random" else se.genetic(ev, PASSES, rng)
        return ev
    a, b = run(1), run(1)
    assert a.E == 25 and [t["value"] for t in a.trace] == [t["value"] for t in b.trace]


def test_repredict_uses_state_features():
    """The 'model' sees a feature of the current state: whether a was already applied."""
    ev = se.Evaluator(objective, budget=30, state=lambda seq: seq, features=lambda seq: ({"has_a": "a" in seq}, 3))
    predict = lambda f: {"a": 0.0 if f["has_a"] else 0.05, "b": 0.07 if f["has_a"] else 0.0, "c": 0.01, "d": 0.0}
    se.repredict(ev, PASSES, predict, width=4, depth=3)
    assert ev.trace[0]["best_seq"].startswith("a,b") and ev.O > ev.E


def test_graph_guided_and_measured_greedy():
    delta = {p: g / 100 for p, g in GAIN.items()}
    ev = se.Evaluator(objective, budget=100)
    se.graph_guided(ev, PASSES, delta, {}, m=2)
    assert ev.best[0] <= 88.0
    ev2 = se.Evaluator(objective, budget=500)
    se.measured_greedy(ev2, PASSES)
    assert ev2.best[0] <= ev.best[0]


def test_opentuner_runs_within_budget():
    ev = se.Evaluator(objective, budget=15)
    se.opentuner_search(ev, [()] + [(p,) for p in PASSES], slots=4, seed=3)
    assert ev.E == 15 and ev.best is not None
