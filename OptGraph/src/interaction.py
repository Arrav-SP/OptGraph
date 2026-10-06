"""Pairwise interaction scores.

For a metric m (a count where smaller means a smaller IR), program P and an
ordered pass pair (A, B):

    delta(B | P)    = ( m(P)    - m(B(P))    ) / m(P)     benefit of B alone
    delta(B | A, P) = ( m(A(P)) - m(B(A(P))) ) / m(P)     benefit B adds after A

    I(A, B, P) = delta(B | A, P) - delta(B | P)

Both deltas are expressed as a fraction of the ORIGINAL program's metric m(P),
so they are directly comparable and I is the pair's departure from additivity:

    I(A, B, P) = ( m(A(P)) + m(B(P)) - m(B(A(P))) - m(P) ) / m(P)

I > +t: SYNERGY (A makes B more effective); I < -t: ANTAGONISM (A takes away
part of B's effect); otherwise WEAK. I(B, A, P) is computed separately from
the B->A experiment and is in general different.

`delta_b_after_a_local` additionally reports B's benefit relative to its actual
input, (m(A(P)) - m(B(A(P)))) / m(A(P)), for reference; it is not used in I.

Zero baseline: if m(P) == 0 (e.g. PHI count of -O0 IR is always 0) the
normalised quantities are undefined. Such rows keep the absolute interaction
(`interaction_abs`, in metric units), have empty normalised columns and
interaction_type UNDEFINED. They are never silently turned into numbers.
"""
from __future__ import annotations

import pandas as pd

from .metrics import METRICS, improvement

FEATURE_PREFIX = "f_"

SYNERGY, ANTAGONISM, WEAK, UNDEFINED = "SYNERGY", "ANTAGONISM", "WEAK", "UNDEFINED"


def classify(score: float | None, threshold: float) -> str:
    if score is None or pd.isna(score):
        return UNDEFINED
    if score > threshold:
        return SYNERGY
    if score < -threshold:
        return ANTAGONISM
    return WEAK


def interaction_row(m_p: float, m_a: float, m_b: float, m_ab: float, threshold: float) -> dict:
    """Interaction of ordered pair (A, B) from the four raw measurements."""
    d_a = improvement(m_p, m_a)
    d_b = improvement(m_p, m_b)
    gain_after = m_a - m_ab
    d_b_after = None if m_p == 0 else gain_after / m_p
    score = None if m_p == 0 else d_b_after - d_b
    return {
        "m_base": m_p, "m_a": m_a, "m_b": m_b, "m_ab": m_ab,
        "delta_a_alone": d_a,
        "delta_b_alone": d_b,
        "delta_b_after_a": d_b_after,
        "delta_b_after_a_local": improvement(m_a, m_ab),
        "interaction_score": score,
        "interaction_abs": gain_after - (m_p - m_b),
        "interaction_type": classify(score, threshold),
    }


def compute_interactions(meas: pd.DataFrame, threshold: float,
                         metrics: list[str] = METRICS) -> pd.DataFrame:
    """One row per (program, ordered pair, metric) where all four runs succeeded."""
    rows = []
    ok = meas[meas["status"] == "SUCCESS"]
    for pid, g in ok.groupby("program_id", sort=True):
        by_variant = g.set_index("variant")
        if "baseline" not in by_variant.index:
            continue
        base = by_variant.loc["baseline"]
        for _, pair in g[g["kind"] == "pair"].iterrows():
            a, b = pair["pass_a"], pair["pass_b"]
            if a not in by_variant.index or b not in by_variant.index:
                continue
            for m in metrics:
                rows.append({
                    "program_id": pid, "category": base["category"],
                    "pass_a": a, "pass_b": b, "metric": m,
                    **interaction_row(float(base[m]), float(by_variant.loc[a, m]),
                                      float(by_variant.loc[b, m]), float(pair[m]), threshold),
                })
    return pd.DataFrame(rows)


def build_dataset(meas: pd.DataFrame, features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Interaction rows joined with the ORIGINAL-program features (prefixed f_)."""
    inter = compute_interactions(meas, threshold)
    if inter.empty:
        return inter
    numeric = features.drop(columns=["category", "loop_method"], errors="ignore")
    numeric = numeric.rename(columns={c: FEATURE_PREFIX + c for c in numeric.columns
                                      if c != "program_id"})
    return inter.merge(numeric, on="program_id", how="left", validate="many_to_one")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
