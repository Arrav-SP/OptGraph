import itertools

from src.ordering import beam_search, greedy, step_gain

PASSES = ["X", "Y", "Z"]


def brute_force(delta, inter, k):
    best = ((), 0.0)
    for n in range(1, k + 1):
        for seq in itertools.product(PASSES, repeat=n):
            if any(a == b for a, b in zip(seq, seq[1:])):
                continue
            total = sum(step_gain(seq[:i], seq[i], delta, inter) for i in range(n))
            if total > best[1] + 1e-12:
                best = (seq, total)
    return list(best[0]), best[1]


def test_greedy_follows_enabling_edge():
    delta = {"X": 0.1, "Y": 0.0, "Z": 0.05}
    inter = {("X", "Y"): 0.3}
    seq, gain = greedy(PASSES, delta, inter, 2)
    assert seq == ["X", "Y"] and abs(gain - 0.4) < 1e-12


def test_beam_finds_optimum_greedy_misses():
    delta = {"X": 0.1, "Y": 0.08, "Z": 0.0}
    inter = {("Y", "Z"): 0.5}
    g_seq, g_gain = greedy(PASSES, delta, inter, 2)
    b_seq, b_gain = beam_search(PASSES, delta, inter, 2, width=3)
    opt_seq, opt_gain = brute_force(delta, inter, 2)
    assert g_seq == ["X", "Y"] and abs(g_gain - 0.18) < 1e-12
    assert b_seq == opt_seq == ["Y", "Z"] and abs(b_gain - opt_gain) < 1e-12


def test_repeat_earns_only_enabled_gain():
    delta = {"X": 0.2, "Y": 0.1, "Z": 0.0}
    inter = {("Y", "X"): 0.05}
    assert step_gain(("X", "Y"), "X", delta, inter) == 0.05
    assert step_gain(("X",), "X", delta, inter) == 0.0
    seq, gain = greedy(PASSES, delta, inter, 5)
    assert seq == ["X", "Y", "X"] and abs(gain - 0.35) < 1e-12


def test_stops_when_nothing_helps():
    seq, gain = greedy(PASSES, {"X": -0.1, "Y": -0.2, "Z": 0.0}, {}, 4)
    assert seq == [] and gain == 0.0


def test_no_repeats_option():
    delta = {"X": 0.2, "Y": 0.1, "Z": 0.0}
    seq, _ = greedy(PASSES, delta, {("Y", "X"): 0.05}, 5, allow_repeats=False)
    assert seq == ["X", "Y"]
