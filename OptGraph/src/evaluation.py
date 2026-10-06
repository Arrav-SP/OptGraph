"""Metrics, plots and the program-similarity analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score)

from .interaction import ANTAGONISM, SYNERGY, WEAK

CLASSES = [ANTAGONISM, WEAK, SYNERGY]
BLUE, ORANGE, GREY = "#2c6fbb", "#d9822b", "#7a7a7a"


def to_class(scores, threshold: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    return np.where(s > threshold, SYNERGY, np.where(s < -threshold, ANTAGONISM, WEAK))


def _corr(fn, a, b) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None  # undefined for a constant vector (e.g. the mean predictor)
    return float(fn(a, b)[0])


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "pearson": _corr(stats.pearsonr, y_true, y_pred),
        "spearman": _corr(stats.spearmanr, y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, threshold: float) -> dict:
    t, p = to_class(y_true, threshold), to_class(y_pred, threshold)
    prec, rec, f1, sup = precision_recall_fscore_support(t, p, labels=CLASSES, zero_division=0)
    present = sup > 0  # macro average over classes that actually occur in y_true
    return {
        "accuracy": float(accuracy_score(t, p)),
        "macro_precision": float(prec[present].mean()),
        "macro_recall": float(rec[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]),
                          "f1": float(f1[i]), "support": int(sup[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(t, p, labels=CLASSES).tolist(),
        "labels": CLASSES,
    }


def evaluate(y_true, y_pred, threshold: float) -> dict:
    return {**regression_metrics(y_true, y_pred),
            "classification": classification_metrics(y_true, y_pred, threshold)}


# ---- plots -------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(panels: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.2), squeeze=False)
    for ax, (title, y, p) in zip(axes[0], panels):
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, lw=1, ls="--", label="ideal")
        ax.scatter(y, p, s=12, alpha=0.5, color=BLUE, edgecolors="none")
        m = regression_metrics(y, p)
        ax.set_title(f"{title}\nn={m['n']}  MAE={m['mae']:.4f}  R²={m['r2']:.3f}", fontsize=10)
        ax.set_xlabel("actual interaction score")
        ax.set_ylabel("predicted interaction score")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    _save(fig, path)


def plot_residuals(y, p, title: str, path: Path) -> None:
    res = p - y
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    a.scatter(p, res, s=12, alpha=0.5, color=BLUE, edgecolors="none")
    a.axhline(0, color=GREY, lw=1, ls="--")
    a.set_xlabel("predicted interaction score")
    a.set_ylabel("residual (predicted − actual)")
    a.grid(alpha=0.3)
    b.hist(res, bins=50, color=BLUE)
    b.axvline(0, color=GREY, lw=1, ls="--")
    b.set_xlabel("residual (predicted − actual)")
    b.set_ylabel("rows")
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


def plot_score_distribution(scores, threshold: float, title: str, path: Path) -> None:
    s = np.asarray(scores, float)
    counts = pd.Series(to_class(s, threshold)).value_counts().reindex(CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(s, bins=60, color=BLUE)
    ax.axvspan(-threshold, threshold, color=ORANGE, alpha=0.25, label=f"weak band (|I| ≤ {threshold:g})")
    ax.set_yscale("log")
    ax.set_xlabel("interaction score I(A, B, P)")
    ax.set_ylabel("ordered pairs (log scale)")
    ax.set_title(f"{title}\n" + "   ".join(f"{c}: {n}" for c, n in counts.items()), fontsize=10)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, title: str, path: Path) -> None:
    """table: index = model, columns include mae, rmse, r2."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, col, label in zip(axes, ("mae", "rmse", "r2"), ("MAE (lower is better)",
                                                              "RMSE (lower is better)",
                                                              "R² (higher is better)")):
        vals = table[col].astype(float)
        ax.barh(table.index, vals, color=BLUE)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel(label)
        ax.invert_yaxis()
        ax.margins(x=0.18)
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


def plot_feature_importance(importance: pd.Series, title: str, path: Path, top: int = 20) -> None:
    imp = importance.sort_values(ascending=False).head(top)[::-1]
    colors = [ORANGE if n.startswith("f_") else BLUE for n in imp.index]
    total = importance.sum() or 1.0
    share = importance[[n for n in importance.index if n.startswith("f_")]].sum() / total
    title += f"\nprogram features hold {100 * share:.1f}% of total importance"
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_xlabel("impurity-based importance")
    ax.set_title(title, fontsize=10)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["program feature", "pass identity"], frameon=False, loc="lower right")
    _save(fig, path)


def plot_confusion(cm, title: str, path: Path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("actual class")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_ablation(table: pd.DataFrame, title: str, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, label in ((a, "mae", "MAE (lower is better)"), (b, "r2", "R² (higher is better)")):
        vals = table[col].astype(float)
        ax.bar(table.index, vals, color=[GREY, ORANGE, BLUE][:len(vals)])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


# ---- program similarity vs interaction similarity ---------------------------------

def _cosine_matrix(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    u = m / norm
    return u @ u.T


def similarity_analysis(features: pd.DataFrame, ds: pd.DataFrame, metric: str,
                        seed: int, permutations: int = 5000) -> dict:
    """Does structural similarity between programs track interaction-pattern similarity?

    Structural vector  : z-scored original-IR features.
    Interaction vector : I(A, B, P) over all ordered pairs, with the across-program
                         mean of each pair removed, so similarity reflects how a
                         program DEVIATES from the average pattern rather than the
                         pattern all programs share.
    Both compared with cosine similarity; association tested with a Mantel
    permutation test (program labels shuffled), since program pairs are not
    independent observations.
    """
    d = ds[(ds["metric"] == metric) & ds["interaction_score"].notna()]
    inter = d.pivot_table(index="program_id", columns=["pass_a", "pass_b"],
                          values="interaction_score").dropna(axis=1)
    feats = features.set_index("program_id").select_dtypes("number").loc[inter.index]
    feats = feats.loc[:, feats.std(ddof=0) > 0]
    z = (feats - feats.mean()) / feats.std(ddof=0)
    s_struct = _cosine_matrix(z.to_numpy())
    s_inter = _cosine_matrix((inter - inter.mean()).to_numpy())
    s_inter_raw = _cosine_matrix(inter.to_numpy())

    iu = np.triu_indices(len(inter), k=1)
    x, y = s_struct[iu], s_inter[iu]
    rho = float(stats.spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    n = len(inter)
    null = np.empty(permutations)
    for i in range(permutations):
        perm = rng.permutation(n)
        null[i] = stats.spearmanr(x, s_inter[np.ix_(perm, perm)][iu])[0]
    p_value = float((np.sum(null >= rho) + 1) / (permutations + 1))
    return {
        "metric": metric, "n_programs": int(n), "n_program_pairs": int(len(x)),
        "n_features": int(z.shape[1]), "n_pass_pairs": int(inter.shape[1]),
        "spearman": rho, "pearson": float(stats.pearsonr(x, y)[0]),
        "mantel_p_one_sided": p_value, "permutations": permutations,
        "spearman_raw_uncentered": float(stats.spearmanr(x, s_inter_raw[iu])[0]),
        "programs": list(inter.index),
        "_x": x, "_y": y, "_s_struct": s_struct, "_s_inter": s_inter,
    }


def plot_similarity(res: dict, categories: dict[str, str], path: Path) -> None:
    progs = res["programs"]
    iu = np.triu_indices(len(progs), k=1)
    same = np.array([categories[progs[i]] == categories[progs[j]] for i, j in zip(*iu)])
    x, y = res["_x"], res["_y"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    a.scatter(x[~same], y[~same], s=18, alpha=0.6, color=BLUE, label="different category")
    a.scatter(x[same], y[same], s=22, alpha=0.8, color=ORANGE, label="same category")
    slope, icpt = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    a.plot(xs, slope * xs + icpt, color=GREY, lw=1.2, ls="--")
    a.set_xlabel("structural similarity (cosine, z-scored features)")
    a.set_ylabel("interaction-pattern similarity (cosine, centred)")
    a.set_title(f"{res['n_program_pairs']} program pairs   Spearman ρ = {res['spearman']:.3f}\n"
                f"Mantel permutation p = {res['mantel_p_one_sided']:.4f} (one-sided)", fontsize=10)
    a.grid(alpha=0.3)
    a.legend(frameon=False, fontsize=8)
    for ax, mat, title in ((b, res["_s_struct"], "structural similarity"),
                           (c, res["_s_inter"], "interaction-pattern similarity")):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(progs)), progs, rotation=90, fontsize=6)
        ax.set_yticks(range(len(progs)), progs, fontsize=6)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)
