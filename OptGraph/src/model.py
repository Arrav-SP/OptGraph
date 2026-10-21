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


# ---- v3 --------------------------------------------------------------------------------
# Representations of the ORIGINAL program (data/v3/features_<rep>.csv), targets
#   direct   : y = I(A,B,P,context)
#   residual : y = I - mean_train(I | pair); the prediction adds the training pair mean back.
# The residual target asks directly whether program information adds anything beyond the
# pair's identity. Every split is by program (or by suite); tuning is nested and grouped.

import os  # noqa: E402

from sklearn.utils.parallel import Parallel, delayed  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold, ParameterGrid  # noqa: E402

# Threads per model fit; outer folds run in parallel processes (see cross_predict_v3).
MODEL_THREADS = int(os.environ.get("OPTGRAPH_THREADS", "2"))

REPRESENTATIONS = {"hand": "f_", "funcprops": "fp_", "opcodes": "op_", "ir2vec": "iv_"}
V3_MODELS = ("MeanBaseline", "PairMeanBaseline", "LinearRidge", "RandomForest",
             "GradientBoosting", "HistGradientBoosting")


def load_representations(data_dir, reps: list[str]) -> pd.DataFrame:
    """program_id + the columns of the requested representations, joined."""
    out = None
    for r in reps:
        f = pd.read_csv(data_dir / f"features_{r}.csv")
        out = f if out is None else out.merge(f, on="program_id", validate="one_to_one")
    return out


def v3_matrix(df: pd.DataFrame, pass_names: list[str], feat_cols: list[str],
              mode: str) -> tuple[np.ndarray, dict]:
    """Design matrix plus its column layout.

    Blocks: `codes` (pass_a, pass_b as integer categories, for HistGradientBoosting),
    `onehot_ab` (one-hot pass_a and pass_b), `onehot_pair` (one-hot ordered pair, for the
    linear model, which cannot form pass interactions itself), `feat` (program features).
    `mode` drops the pass blocks ("features") or the feature block ("pass").
    """
    assert mode in MODES
    n = len(pass_names)
    a = pd.Categorical(df["pass_a"], categories=pass_names).codes
    b = pd.Categorical(df["pass_b"], categories=pass_names).codes
    blocks = {}
    if mode in ("pass", "both"):
        rows = np.arange(len(df))
        ab = np.zeros((len(df), 2 * n), np.float32)
        ab[rows, a] = 1
        ab[rows, n + b] = 1
        off = np.flatnonzero(np.eye(n).ravel() == 0)            # ordered pairs, a != b
        pair = np.zeros((len(df), n * n), np.float32)
        pair[rows, a * n + b] = 1
        blocks["codes"] = np.c_[a, b].astype(np.float32)
        blocks["onehot_ab"] = ab
        blocks["onehot_pair"] = pair[:, off]
    if mode in ("features", "both"):
        blocks["feat"] = df[feat_cols].to_numpy(np.float32)
    layout, cols, start = {}, [], 0
    for k, v in blocks.items():
        layout[k] = np.arange(start, start + v.shape[1])
        start += v.shape[1]
        cols.append(v)
    return np.hstack(cols), layout


# Which blocks each model sees.
MODEL_BLOCKS = {"LinearRidge": ("onehot_ab", "onehot_pair", "feat"),
                "RandomForest": ("onehot_ab", "feat"), "GradientBoosting": ("onehot_ab", "feat"),
                "HistGradientBoosting": ("codes", "feat")}


def model_view(name: str, X: np.ndarray, layout: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """(columns of X for this model, categorical mask for HistGradientBoosting or None)."""
    keys = [k for k in MODEL_BLOCKS[name] if k in layout]
    idx = np.concatenate([layout[k] for k in keys])
    cat = None
    if name == "HistGradientBoosting":
        cat = np.isin(idx, layout.get("codes", []))
    return X[:, idx], cat


def v3_model_specs(seed: int) -> dict:
    """name -> (factory(categorical, **params), tuning grid). Grids are small and fixed in advance;
    the first grid point is the a-priori default used when a model is not tuned."""
    return {
        "LinearRidge": (lambda cat, **p: make_pipeline(StandardScaler(), Ridge(**p)),
                        {"alpha": [10.0, 1.0, 100.0]}),
        "RandomForest": (lambda cat, **p: RandomForestRegressor(n_estimators=200, max_features=0.3,
                                                                random_state=seed, n_jobs=MODEL_THREADS, **p),
                         {"min_samples_leaf": [5, 20]}),
        "GradientBoosting": (lambda cat, **p: GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, subsample=0.8, random_state=seed, **p),
            {"max_depth": [4, 6]}),
        "HistGradientBoosting": (lambda cat, **p: HistGradientBoostingRegressor(
            max_iter=300, early_stopping=False, random_state=seed,
            categorical_features=cat if cat is not None and cat.any() else None, **p),
            {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]}),
    }


def pair_key(df: pd.DataFrame) -> pd.Series:
    return df["pass_a"] + ">" + df["pass_b"]


def _pair_mean(train: pd.DataFrame, y_train: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    means = pd.Series(y_train, index=train.index).groupby(pair_key(train)).mean()
    return pair_key(test).map(means).fillna(float(np.mean(y_train))).to_numpy()


def _fit_predict(name, params, specs, Xtr, ytr, Xte, cat=None):
    model = specs[name][0](cat, **params)
    model.fit(Xtr, ytr)
    return np.asarray(model.predict(Xte), float)


def tune(name: str, specs: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
         inner_folds: int = 3, cat=None) -> dict:
    """Grid point with the lowest inner grouped-CV MAE (inner folds split the training programs)."""
    grid = list(ParameterGrid(specs[name][1]))
    if len(grid) == 1:
        return grid[0]
    k = min(inner_folds, len(set(groups)))
    if k < 2:
        return grid[0]
    scores = []
    for params in grid:
        err = 0.0
        for tr, te in GroupKFold(k).split(X, y, groups):
            err += np.abs(_fit_predict(name, params, specs, X[tr], y[tr], X[te], cat) - y[te]).sum()
        scores.append(err)
    return grid[int(np.argmin(scores))]


def grouped_splits(groups: np.ndarray, protocol: str, seed: int, k: int = 10,
                   suites: np.ndarray | None = None):
    """(name, train mask, test mask) per outer fold. Train and test programs never overlap.

    lopo  : one program held out per fold
    kfold : k folds of programs (seeded shuffle)
    loso  : one suite held out per fold
    transfer:<a>:<b> : train on suite a only, test on suite b
    """
    progs = np.array(sorted(set(groups)))
    if protocol == "lopo":
        for p in progs:
            yield p, groups != p, groups == p
    elif protocol == "kfold":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(progs)
        for i, chunk in enumerate(np.array_split(perm, min(k, len(progs)))):
            test = np.isin(groups, chunk)
            yield f"fold{i}", ~test, test
    elif protocol == "loso":
        for s in sorted(set(suites)):
            yield s, suites != s, suites == s
    elif protocol.startswith("transfer:"):          # transfer:<train suite>:<test suite>
        _, train, test = protocol.split(":")
        yield f"{train}->{test}", suites == train, suites == test
    else:
        raise ValueError(protocol)


def _fold_job(fold, tr, te, df, y, X, layout, groups, models, seed, target, tune_models):
    specs = v3_model_specs(seed)
    pm = _pair_mean(df[tr], y[tr], df[te])
    ytr = y[tr] - (_pair_mean(df[tr], y[tr], df[tr]) if target == "residual" else 0.0)
    preds, chosen = {}, {}
    for m in models:
        if m == "MeanBaseline":
            preds[m] = np.full(te.sum(), float(y[tr].mean()))
        elif m == "PairMeanBaseline":
            preds[m] = pm
        else:
            Xm, cat = model_view(m, X, layout)
            do_tune = tune_models is True or (bool(tune_models) and m in tune_models)
            params = tune(m, specs, Xm[tr], ytr, groups[tr], cat=cat) if do_tune else \
                list(ParameterGrid(specs[m][1]))[0]
            chosen[m] = params
            preds[m] = _fit_predict(m, params, specs, Xm[tr], ytr, Xm[te], cat) + \
                (pm if target == "residual" else 0.0)
    return fold, te, pm, preds, chosen


def cross_predict_v3(df: pd.DataFrame, X: np.ndarray, layout: dict, protocol: str,
                     models: list[str], seed: int, target: str = "residual",
                     tune_models=True, suites: np.ndarray | None = None, k: int = 10,
                     n_jobs: int = 1) -> pd.DataFrame:
    """Out-of-program predictions (on the I scale) for every row and every model.

    Returns df's identifying columns, the truth, the fold name, the training pair mean
    (`pair_mean_train`) and one column per model. Chosen hyperparameters are in
    `.attrs['params']` as {model: {fold: params}}. `tune_models` is True, False, or the set
    of model names to tune (nested, grouped by program); the others use their default.
    Outer folds run in `n_jobs` processes.
    """
    df = df.reset_index(drop=True)
    y = df["interaction_score"].to_numpy(float)
    groups = df["program_id"].to_numpy()
    folds = list(grouped_splits(groups, protocol, seed, k, suites))
    for _, tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]), "program leakage between train and test"
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fold_job)(f, tr, te, df, y, X, layout, groups, models, seed, target, tune_models)
        for f, tr, te in folds)
    out = df[["program_id", "pass_a", "pass_b", "interaction_score"]].copy()
    out["fold"] = ""
    out["pair_mean_train"] = np.nan
    chosen = {m: {} for m in models}
    for m in models:
        out[m] = np.nan
    for fold, te, pm, preds, params in results:
        out.loc[te, "fold"] = str(fold)
        out.loc[te, "pair_mean_train"] = pm
        for m, v in preds.items():
            out.loc[te, m] = v
        for m, p_ in params.items():
            chosen[m][str(fold)] = p_
    out.attrs["params"] = chosen
    return out
