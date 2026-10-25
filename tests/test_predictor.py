import numpy as np
import pandas as pd
import pytest

from src import predictor as pr

PASSES = ["a", "b", "c"]


@pytest.fixture(autouse=True)
def small_models(monkeypatch):
    monkeypatch.setattr(pr, "MAX_ITER", 40)      # same code path, fewer boosting rounds


def _df(n=12, seed=0):
    """Toy data where the residual from the pair mean depends on feature f: learnable."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        f = rng.random()
        for ctx in ("S0", "S2"):
            for x in PASSES:
                for y in PASSES:
                    if x != y:
                        base = 0.1 if (x, y) == ("a", "b") else 0.0
                        rows.append((f"p{i}", ctx, x, y, base + (0.2 * f if x == "a" else 0.0), f))
    return pd.DataFrame(rows, columns=["program_id", "context", "pass_a", "pass_b", "interaction_score", "f_x"])


def test_fit_fold_never_sees_test_program_and_tunes_on_train():
    df = _df()
    fm, info = pr.fit_fold(df, "p3", PASSES, ["f_x"], seed=42)
    assert "p3" not in info["val_programs"] and info["train_programs"] == 11
    assert len(info["val_scores"]) == 4 and info["params"] in [dict(p) for p in pr.ParameterGrid(pr.HGB_GRID)]


def test_lopo_beats_static_on_learnable_data_and_is_deterministic():
    df = _df()
    pred, infos = pr.lopo(df, PASSES, ["f_x"], seed=42, n_jobs=1)
    assert pred["predicted"].notna().all() and len(infos) == 12
    assert pr.gap_closed(pred) > 0.3
    again, _ = pr.lopo(df, PASSES, ["f_x"], seed=42, n_jobs=1)
    assert np.allclose(pred["predicted"], again["predicted"])


def test_static_graph_is_context_free_mean_of_training_rows():
    df = _df()
    means = pr.static_means(df[df.program_id != "p0"])
    assert means[("a", "b")] == pytest.approx(df[(df.program_id != "p0") & (df.pass_a == "a") & (df.pass_b == "b")]
                                              .interaction_score.mean())


def test_table_singles_and_feature_join():
    target = pd.DataFrame({"program_id": ["p", "p"], "context": ["S0", "S0"], "pass_a": ["a", "b"],
                           "pass_b": ["c", "c"], "interaction_score": [0.1, 0.2], "delta_b_alone": [0.3, 0.3]})
    feats = pd.DataFrame({"program_id": ["p"], "context": ["S0"], "f_x": [1.0]})
    s = pr.table(target, feats, "single")
    assert len(s) == 1 and s.iloc[0]["pass_a"] == "c" and s.iloc[0]["interaction_score"] == 0.3


def test_reduced_rank_ridge_rank():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(30, 5)), rng.normal(size=(30, 8))
    B = pr.reduced_rank_ridge(X, Y, 1.0, 2)
    assert B.shape == (5, 8) and np.linalg.matrix_rank(X @ B) == 2
