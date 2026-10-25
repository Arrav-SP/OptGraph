"""v3.1 predictors of per-program interactions (Claim 3) and the models re-prediction uses (Claim 4).

Rows: one per (program, snapshot, ordered pair) with target I, and one per (program, snapshot,
pass) with target Δ(B|state). Inputs: features of the IR the passes run on (snapshot-matched)
plus pass identities; the snapshot name is NOT an input, so a model can be applied to any IR
state reached during ordering.

Static graph = the context-free pair (or pass) mean over the training programs, i.e. one global
graph as in DCO / Pan et al. The model predicts the residual from it (pre-registered target).

Leave-one-program-out with validation tuning: for held-out program p, the training programs are
all others; a seeded share of them is the validation set used to pick hyperparameters; the best
setting is refitted on all training programs. p is never seen.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid

HGB_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]}
MAX_ITER = 300
THREADS = int(os.environ.get("OPTGRAPH_THREADS", "2"))


def table(target: pd.DataFrame, features: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Join targets with the features of the same (program, snapshot) baseline IR.
    kind "pair": target rows pass_a, pass_b, interaction_score. kind "single": pass_b,
    delta_b_alone (becomes pass_a = pass_b = pass, target column interaction_score)."""
    t = target.copy()
    if kind == "single":
        t = t.drop_duplicates(["program_id", "context", "pass_b"]).assign(
            pass_a=lambda d: d["pass_b"], interaction_score=lambda d: d["delta_b_alone"])
    t = t[["program_id", "context", "pass_a", "pass_b", "interaction_score"]].dropna()
    return t.merge(features, on=["program_id", "context"], how="inner", validate="many_to_one")


def static_means(train: pd.DataFrame) -> pd.Series:
    """Context-free static graph: mean target per (pass_a, pass_b) over the training rows."""
    return train.groupby(["pass_a", "pass_b"])["interaction_score"].mean()


def _static(rows: pd.DataFrame, means: pd.Series, fallback: float) -> np.ndarray:
    idx = pd.MultiIndex.from_frame(rows[["pass_a", "pass_b"]])
    return means.reindex(idx).fillna(fallback).to_numpy()


def design(rows: pd.DataFrame, passes: list[str], feat_cols: list[str]) -> np.ndarray:
    """Pass identities as two integer columns (native categoricals for HGB) + features."""
    a = pd.Categorical(rows["pass_a"], categories=passes).codes
    b = pd.Categorical(rows["pass_b"], categories=passes).codes
    return np.c_[a, b, rows.reindex(columns=feat_cols, fill_value=0.0).to_numpy(np.float32)]


def _hgb(seed: int, **p):
    return HistGradientBoostingRegressor(max_iter=MAX_ITER, early_stopping=False, random_state=seed,
                                         categorical_features=[0, 1], **p)


class FoldModel:
    """Residual model for one held-out program, plus the static graph it corrects."""

    def __init__(self, passes, feat_cols, means, fallback, model, params):
        self.passes, self.feat_cols, self.means, self.fallback = passes, feat_cols, means, fallback
        self.model, self.params = model, params

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return _static(rows, self.means, self.fallback) + self.model.predict(design(rows, self.passes, self.feat_cols))

    def static(self, rows: pd.DataFrame) -> np.ndarray:
        return _static(rows, self.means, self.fallback)


def fit_fold(df: pd.DataFrame, test_program: str, passes: list[str], feat_cols: list[str],
             seed: int, val_share: float = 0.2) -> tuple[FoldModel, dict]:
    """Validation-tuned residual HGB for one LOPO fold. Returns (model, split info)."""
    train = df[df["program_id"] != test_program]
    progs = np.array(sorted(train["program_id"].unique()))
    rng = np.random.default_rng(seed + sum(map(ord, test_program)))
    val = set(rng.choice(progs, max(1, int(round(val_share * len(progs)))), replace=False))
    fit_part, val_part = train[~train["program_id"].isin(val)], train[train["program_id"].isin(val)]
    assert test_program not in set(train["program_id"]) and not val & set(fit_part["program_id"])
    m_fit = static_means(fit_part)
    g_fit = float(fit_part["interaction_score"].mean())
    Xf, yf = design(fit_part, passes, feat_cols), fit_part["interaction_score"].to_numpy() - _static(fit_part, m_fit, g_fit)
    Xv, yv = design(val_part, passes, feat_cols), val_part["interaction_score"].to_numpy() - _static(val_part, m_fit, g_fit)
    scores = []
    for params in ParameterGrid(HGB_GRID):
        mdl = _hgb(seed, **params).fit(Xf, yf)
        scores.append((float(np.abs(mdl.predict(Xv) - yv).mean()), sorted(params.items())))
    best = dict(min(scores)[1])
    means, fallback = static_means(train), float(train["interaction_score"].mean())
    y = train["interaction_score"].to_numpy() - _static(train, means, fallback)
    model = _hgb(seed, **best).fit(design(train, passes, feat_cols), y)
    return FoldModel(passes, feat_cols, means, fallback, model, best), {
        "test_program": test_program, "train_programs": int(len(progs)), "val_programs": sorted(val),
        "params": best, "val_scores": [s for s, _ in scores]}


def lopo(df: pd.DataFrame, passes: list[str], feat_cols: list[str], seed: int, model_dir=None,
         n_jobs: int = 6) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-program predictions for every row (and fold models saved to model_dir/<p>.joblib)."""
    from sklearn.utils.parallel import Parallel, delayed

    def one(p):
        fm, info = fit_fold(df, p, passes, feat_cols, seed)
        rows = df[df["program_id"] == p]
        if model_dir is not None:
            joblib.dump(fm, model_dir / f"{p}.joblib")
        return rows.index.to_numpy(), fm.predict(rows), fm.static(rows), info

    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    res = Parallel(n_jobs=n_jobs)(delayed(one)(p) for p in sorted(df["program_id"].unique()))
    out = df[["program_id", "context", "pass_a", "pass_b", "interaction_score"]].copy()
    out["predicted"], out["static"] = np.nan, np.nan
    for idx, pred, stat, _ in res:
        out.loc[idx, "predicted"], out.loc[idx, "static"] = pred, stat
    return out, [r[3] for r in res]


def gap_closed(pred: pd.DataFrame) -> float:
    """Share of the static graph's error the model removes: 1 − MAE(model) / MAE(static).
    (The oracle, the program's own measured graph, has zero error.)"""
    e_m = (pred["predicted"] - pred["interaction_score"]).abs().mean()
    e_s = (pred["static"] - pred["interaction_score"]).abs().mean()
    return float(1 - e_m / e_s) if e_s else np.nan


# ---- cold-start recommender: reduced-rank ridge from program features to interaction vectors

def reduced_rank_ridge(X: np.ndarray, Y: np.ndarray, lam: float, rank: int) -> np.ndarray:
    """B (features × outputs) of ridge regression Y ≈ X B, truncated to `rank` (programs × pairs
    matrix completion for unseen programs, with features as side information)."""
    xtx = X.T @ X + lam * np.eye(X.shape[1])
    b = np.linalg.solve(xtx, X.T @ Y)
    u, s, vt = np.linalg.svd(X @ b, full_matrices=False)
    v = vt[:rank].T
    return b @ v @ v.T
