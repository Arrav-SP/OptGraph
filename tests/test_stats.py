import numpy as np
import pandas as pd
import pytest

from src.stats import bootstrap_ci, cv, holm, ndcg_at_k, paired_test, variance_decomposition


def test_holm_matches_definition():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_bootstrap_ci_contains_median_and_is_deterministic():
    v = np.arange(1, 101)
    lo, hi = bootstrap_ci(v, seed=1)
    assert lo <= 50.5 <= hi
    assert (lo, hi) == bootstrap_ci(v, seed=1)


def test_cv():
    assert cv([1, 1, 1]) == 0.0
    assert abs(cv([1, 3]) - np.std([1, 3], ddof=1) / 2) < 1e-12


def test_paired_test_all_ties():
    r = paired_test([1, 2, 3], [1, 2, 3])
    assert r["p_value"] == 1.0 and r["ties"] == 3


def test_paired_test_direction():
    a = np.arange(20) * 0.0
    b = np.arange(1, 21) * 1.0
    r = paired_test(a, b)
    assert r["a_better"] == 20 and r["p_value"] < 0.001 and r["mean_diff"] < 0


def test_ndcg():
    y = np.array([0.3, 0.1, 0.2, 0.0])
    assert ndcg_at_k(y, y, 4) == pytest.approx(1.0)
    assert ndcg_at_k(y, -y, 4) < 1.0
    assert np.isnan(ndcg_at_k([1, 1, 1], [3, 2, 1]))


def test_variance_decomposition_recovers_additive_structure():
    rows = []
    a_eff, b_eff = {"a1": -1.0, "a2": 1.0}, {"b1": -2.0, "b2": 0.0, "b3": 2.0}
    for a in a_eff:
        for b in b_eff:
            for c in ("c1", "c2"):
                rows.append({"A": a, "B": b, "C": c, "y": a_eff[a] + b_eff[b]})
    out = variance_decomposition(pd.DataFrame(rows), "y", ["A", "B", "C"]).set_index("term")
    assert out["share"].sum() == pytest.approx(1.0)
    total = 6 * 2 * 1.0 + 4 * 8 / 1.0  # SS_A = 12*1, SS_B = 4*(4+0+4)
    assert out.loc["A", "ss"] == pytest.approx(12.0) and out.loc["B", "ss"] == pytest.approx(32.0)
    assert out.loc["C", "ss"] == pytest.approx(0.0) and out.loc["A x B x C", "ss"] == pytest.approx(0.0)
    assert out["ss"].sum() == pytest.approx(total)


def test_variance_decomposition_rejects_incomplete():
    df = pd.DataFrame({"A": ["x", "x", "y"], "B": ["p", "q", "p"], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        variance_decomposition(df, "y", ["A", "B"])
