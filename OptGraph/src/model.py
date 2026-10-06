"""Predicting I(A, B, P) from original-program features and pass identities.

Inputs  : f_* columns (structural features of the ORIGINAL IR) and one-hot
          encodings of pass_a and pass_b. Nothing measured after optimisation.
Target  : interaction_score for one metric.
Splits  : always by program. No program contributes rows to more than one of
          train / validation / test, so test programs are genuinely unseen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .interaction import feature_columns

MODES = ("pass", "features", "both")
BASELINES = ("MeanBaseline", "PairMeanBaseline")


def load_ml_table(ds: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, int]:
    """Rows for one metric with a defined target. Returns (table, rows dropped)."""
    d = ds[ds["metric"] == metric]
    keep = d[d["interaction_score"].notna()].reset_index(drop=True)
    return keep, len(d) - len(keep)


def split_programs(programs: list[str], seed: int,
                   fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict[str, list[str]]:
    """Seeded shuffle of PROGRAM IDS into train / val / test."""
    progs = sorted(programs)
    np.random.default_rng(seed).shuffle(progs)
    n = len(progs)
    n_test = max(1, round(n * fractions[2]))
    n_val = max(1, round(n * fractions[1]))
    return {"test": sorted(progs[:n_test]), "val": sorted(progs[n_test:n_test + n_val]),
            "train": sorted(progs[n_test + n_val:])}


def build_matrix(df: pd.DataFrame, pass_names: list[str], mode: str = "both") -> pd.DataFrame:
    """Design matrix. `pass_names` fixes the one-hot columns independent of the rows given."""
    assert mode in MODES
    parts = []
    if mode in ("pass", "both"):
        for col in ("pass_a", "pass_b"):
            cat = pd.Categorical(df[col], categories=pass_names)
            parts.append(pd.get_dummies(cat, prefix=col, dtype=float).set_axis(df.index))
        # An interaction belongs to the ORDERED pair, not to A and B separately, so
        # the pair itself is also one-hot encoded (A>B and B>A are different columns).
        pairs = [f"{a}>{b}" for a in pass_names for b in pass_names if a != b]
        cat = pd.Categorical(df["pass_a"] + ">" + df["pass_b"], categories=pairs)
        parts.append(pd.get_dummies(cat, prefix="pair", dtype=float).set_axis(df.index))
    if mode in ("features", "both"):
        parts.append(df[feature_columns(df)].astype(float))
    return pd.concat(parts, axis=1)


class PairMeanRegressor(BaseEstimator, RegressorMixin):
    """Predicts the training-set mean interaction of each (pass_a, pass_b) pair.

    It knows which passes are involved but nothing about the program, so it is
    the reference for "do program features add anything?". Works on the one-hot
    pass columns of build_matrix.
    """

    def fit(self, X: pd.DataFrame, y):
        self.cols_ = [c for c in X.columns if c.startswith(("pass_a_", "pass_b_"))]
        key = X[self.cols_].astype(int).astype(str).agg("".join, axis=1)
        y = pd.Series(np.asarray(y), index=X.index)
        self.table_ = y.groupby(key).mean().to_dict()
        self.global_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame):
        key = X[self.cols_].astype(int).astype(str).agg("".join, axis=1)
        return key.map(self.table_).fillna(self.global_).to_numpy()


def make_models(seed: int) -> dict:
    return {
        "MeanBaseline": DummyRegressor(strategy="mean"),
        "PairMeanBaseline": PairMeanRegressor(),
        "LinearRidge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                              random_state=seed, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=400, max_depth=5,
                                                      learning_rate=0.05, subsample=0.8,
                                                      random_state=seed),
    }


def fit_predict(model, X_train, y_train, X_test) -> np.ndarray:
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def leave_one_program_out(df: pd.DataFrame, pass_names: list[str], seed: int,
                          model_names: list[str] | None = None,
                          mode: str = "both") -> pd.DataFrame:
    """Out-of-program predictions for every row.

    Each program is held out in turn; models are trained on all other programs.
    Returns df's identifying columns plus one prediction column per model.
    """
    X = build_matrix(df, pass_names, mode)
    y = df["interaction_score"].to_numpy()
    groups = df["program_id"].to_numpy()
    names = model_names or list(make_models(seed))
    preds = {n: np.full(len(df), np.nan) for n in names}
    for prog in sorted(set(groups)):
        test = groups == prog
        for n in names:
            if n == "PairMeanBaseline" and mode == "features":
                continue
            model = make_models(seed)[n]
            preds[n][test] = fit_predict(model, X[~test], y[~test], X[test])
    out = df[["program_id", "pass_a", "pass_b", "interaction_score"]].copy()
    for n in names:
        out[n] = preds[n]
    return out
