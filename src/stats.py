"""Statistics helpers for v3: bootstrap CIs, paired tests with Holm correction, ranking
quality and a crossed-design variance decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values, stat=np.median, n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat` over `values`."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def cv(values) -> float:
    """Coefficient of variation (sample std / mean)."""
    v = np.asarray(values, float)
    return float(np.std(v, ddof=1) / np.mean(v)) if len(v) > 1 and np.mean(v) else np.nan


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as given)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired_test(a, b, n_boot: int = 2000, seed: int = 42) -> dict:
    """Wilcoxon signed-rank test of a vs b (paired by program) and a bootstrap CI of mean(a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nonzero = d[d != 0]
    if len(nonzero) == 0:
        stat, p = 0.0, 1.0
    else:
        stat, p = stats.wilcoxon(nonzero)
    lo, hi = bootstrap_ci(d, np.mean, n_boot, seed)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "ci95_mean_diff": [lo, hi], "wilcoxon_stat": float(stat), "p_value": float(p),
            "a_better": int((d < 0).sum()), "b_better": int((d > 0).sum()), "ties": int((d == 0).sum())}


def ndcg_at_k(y_true, y_pred, k: int = 10) -> float:
    """NDCG@k of ranking items by y_pred, with relevance = y_true shifted to start at 0.

    Higher y_true = more relevant (here: larger interaction, i.e. the pairs a pass
    orderer most wants to find). Returns nan when every relevance is equal.
    """
    t, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    rel = t - t.min()
    if rel.max() == 0:
        return np.nan
    k = min(k, len(t))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    # Ties in the prediction are broken by position, which favours no item in particular.
    dcg = (rel[np.argsort(-p, kind="stable")[:k]] * disc).sum()
    idcg = (np.sort(rel)[::-1][:k] * disc).sum()
    return float(dcg / idcg)


def variance_decomposition(df: pd.DataFrame, value: str, factors: list[str]) -> pd.DataFrame:
    """Share of the total sum of squares of `value` per main effect and interaction.

    `df` must be a complete crossed design with one observation per cell (every level
    combination present once). Main effects and all two-way interactions are estimated
    from cell means; with three factors the three-way term is the remainder.
    """
    assert len(factors) in (2, 3)
    d = df[factors + [value]].copy()
    sizes = [d[f].nunique() for f in factors]
    if len(d) != np.prod(sizes) or d.duplicated(factors).any():
        raise ValueError("variance_decomposition needs a complete design, one row per cell")
    y = d[value].to_numpy(float)
    mu = y.mean()
    total = ((y - mu) ** 2).sum()
    main = {f: d.groupby(f)[value].transform("mean").to_numpy() - mu for f in factors}
    rows = [{"term": f, "ss": float((main[f] ** 2).sum())} for f in factors]
    fitted = mu + sum(main.values())
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            fi, fj = factors[i], factors[j]
            cell = d.groupby([fi, fj])[value].transform("mean").to_numpy()
            eff = cell - mu - main[fi] - main[fj]
            rows.append({"term": f"{fi} x {fj}", "ss": float((eff ** 2).sum())})
            fitted = fitted + eff
    rows.append({"term": " x ".join(factors) if len(factors) == 3 else "residual",
                 "ss": float(((y - fitted) ** 2).sum())})
    out = pd.DataFrame(rows)
    out["share"] = out["ss"] / total if total else np.nan
    return out
