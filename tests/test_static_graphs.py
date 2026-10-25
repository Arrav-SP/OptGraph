import numpy as np
import pandas as pd
import pytest

from src.static_graphs import (dco_classes, dco_subsequences, pair_mean, pan_synergy_graph,
                               static_graph_error, weighted_walk)

THR = 0.005


def _rows():
    # two programs, passes x/y/z; x->y synergistic in both, y->z only in p2
    pairs = pd.DataFrame([
        ("p1", "x", "y", 0.10), ("p1", "y", "x", -0.02), ("p1", "y", "z", 0.00), ("p1", "z", "y", 0.0),
        ("p1", "x", "z", 0.00), ("p1", "z", "x", 0.00),
        ("p2", "x", "y", 0.20), ("p2", "y", "x", -0.04), ("p2", "y", "z", 0.06), ("p2", "z", "y", 0.0),
        ("p2", "x", "z", 0.00), ("p2", "z", "x", 0.00)],
        columns=["program_id", "pass_a", "pass_b", "interaction_score"])
    singles = pd.DataFrame([("p1", "x", 0.1), ("p1", "y", 0.0), ("p1", "z", -0.1),
                            ("p2", "x", 0.2), ("p2", "y", 0.1), ("p2", "z", -0.1)],
                           columns=["program_id", "pass_b", "delta_b_alone"])
    return pairs, singles


def test_pair_mean_and_classes():
    pairs, singles = _rows()
    delta, inter = pair_mean(pairs, singles)
    assert inter[("x", "y")] == pytest.approx(0.15) and delta["x"] == pytest.approx(0.15)
    c = dco_classes(pairs, THR)
    assert c[("x", "y")] == 1 and c[("y", "x")] == -1 and c[("x", "z")] == 0 and c[("y", "z")] == 1


def test_dco_subsequences_follow_positive_edges():
    pairs, singles = _rows()
    _, inter = pair_mean(pairs, singles)
    subs = dco_subsequences(dco_classes(pairs, THR), inter, max_len=3)
    assert ("x", "y", "z") in subs and ("x",) in subs and ("y", "x") not in subs


def test_pan_graph_and_walk():
    pairs, singles = _rows()
    nodes, edges = pan_synergy_graph(pairs, singles, THR)
    assert edges[("x", "y")] == 1.0 and edges[("y", "z")] == 0.5 and edges[("y", "x")] == 0.0
    assert nodes["x"] == 1.0 and nodes["z"] == 0.0
    rng = np.random.default_rng(42)
    walks = [weighted_walk(nodes, edges, 3, rng) for _ in range(200)]
    starts = pd.Series([w[0] for w in walks]).value_counts()
    assert starts.index[0] == "x" and "z" not in starts.index[:1]
    assert all(len(w) == 3 and all(a != b for a, b in zip(w, w[1:])) for w in walks)
    after_x = pd.Series([w[1] for w in walks if w[0] == "x"]).value_counts()
    assert after_x.index[0] == "y"


def test_static_graph_error():
    pairs, singles = _rows()
    p2 = pairs[pairs.program_id == "p2"]
    _, inter_p1 = pair_mean(pairs[pairs.program_id == "p1"], singles)
    e = static_graph_error(p2, inter_p1, THR, k=1)
    # p2 non-weak pairs: x>y (+), y>x (-), y>z (+); static p1 gets y>z as weak (0): no flip, 1 class error
    assert e["nonweak_pairs"] == 3 and e["sign_flip_rate"] == 0.0
    assert e["class_error_rate"] == pytest.approx(1 / 6)
    assert e["top_k_overlap"] == 1.0          # x>y is the top pair in both (k=2 would hit ties at 0)
    flipped = {k: -v for k, v in inter_p1.items()}
    assert static_graph_error(p2, flipped, THR)["sign_flip_rate"] == pytest.approx(2 / 3)
