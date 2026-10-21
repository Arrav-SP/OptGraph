import numpy as np
import pandas as pd

from src import model as ml


def _df(n_prog=9, passes=("a", "b", "c")):
    rng = np.random.default_rng(0)
    rows = [(f"p{i}", x, y) for i in range(n_prog) for x in passes for y in passes if x != y]
    df = pd.DataFrame(rows, columns=["program_id", "pass_a", "pass_b"])
    df["interaction_score"] = rng.normal(size=len(df))
    df["f_x"] = df["program_id"].str[1:].astype(float)
    return df


def test_grouped_splits_never_share_programs():
    df = _df()
    groups = df["program_id"].to_numpy()
    suites = np.where(df["f_x"] < 3, "s1", np.where(df["f_x"] < 6, "s2", "s3"))
    for proto in ("lopo", "kfold", "loso", "transfer:s1:s2"):
        test_seen = []
        for _, tr, te in ml.grouped_splits(groups, proto, 42, 4, suites):
            assert not set(groups[tr]) & set(groups[te])
            assert te.any() and tr.any()
            test_seen += list(set(groups[te]))
        if proto in ("lopo", "kfold", "loso"):     # every program is tested exactly once
            assert sorted(test_seen) == sorted(set(groups))


def test_cross_predict_pair_mean_excludes_test_program():
    df = _df()
    X, layout = ml.v3_matrix(df, ["a", "b", "c"], ["f_x"], "both")
    out = ml.cross_predict_v3(df, X, layout, "lopo", ["PairMeanBaseline", "LinearRidge"], 42, "residual", False)
    for pid in ("p0", "p4"):
        g = out[out.program_id == pid]
        other = df[df.program_id != pid].groupby(["pass_a", "pass_b"]).interaction_score.mean()
        expect = g.set_index(["pass_a", "pass_b"]).index.map(other)
        assert np.allclose(g["PairMeanBaseline"], expect)
    assert out[["PairMeanBaseline", "LinearRidge"]].notna().all().all()


def test_matrix_blocks():
    df = _df(2)
    X, layout = ml.v3_matrix(df, ["a", "b", "c"], ["f_x"], "both")
    assert X.shape == (len(df), 2 + 6 + 6 + 1)
    assert (X[:, layout["onehot_pair"]].sum(axis=1) == 1).all()
    Xf, lf = ml.v3_matrix(df, ["a", "b", "c"], ["f_x"], "features")
    assert Xf.shape == (len(df), 1) and list(lf) == ["feat"]
