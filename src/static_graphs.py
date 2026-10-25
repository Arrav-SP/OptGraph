"""Static (program-agnostic) pass-interaction graphs and how wrong they are for one program.

All graphs are built from TRAINING programs only; callers pass the training rows. They are
reimplementations in the style of prior work, not the original systems:

  pair-mean   mean I(A,B) over training programs (+ mean single-pass effect per pass)
  DCO-style   pairs classed positive / negative / neutral by their mean I (Liu et al. 2024);
              the positive-edge graph is cut into short chains ("sub-sequences") that a search
              then combines
  Pan-style   synergy knowledge graph (Pan et al. 2025): edge weight = share of training
              programs where the ordered pair is synergistic (I > threshold); sequences are
              drawn by weighted random walks on it

Rows are DataFrames with columns program_id, pass_a, pass_b, interaction_score (pairs) and
program_id, pass_b, delta_b_alone (singles).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pair_mean(pairs: pd.DataFrame, singles: pd.DataFrame) -> tuple[dict, dict]:
    """(delta {pass: mean Δ}, inter {(a, b): mean I}) over the given (training) rows."""
    inter = pairs.groupby(["pass_a", "pass_b"])["interaction_score"].mean().to_dict()
    delta = singles.groupby("pass_b")["delta_b_alone"].mean().to_dict()
    return delta, inter


def dco_classes(pairs: pd.DataFrame, threshold: float) -> dict:
    """{(a, b): +1 / -1 / 0} from the mean interaction over training programs."""
    m = pairs.groupby(["pass_a", "pass_b"])["interaction_score"].mean()
    return {k: int(np.sign(v)) if abs(v) > threshold else 0 for k, v in m.items()}


def dco_subsequences(classes: dict, inter: dict, max_len: int = 4) -> list[tuple[str, ...]]:
    """Chains along positive edges: from every pass, repeatedly follow the strongest positive
    outgoing edge to a pass not yet in the chain. Deduplicated, single passes included, so any
    sequence can still be expressed."""
    passes = sorted({a for a, _ in classes} | {b for _, b in classes})
    out = {(p,) for p in passes}
    for start in passes:
        chain = [start]
        while len(chain) < max_len:
            nxt = [(inter[(chain[-1], b)], b) for b in passes
                   if b not in chain and classes.get((chain[-1], b)) == 1]
            if not nxt:
                break
            chain.append(max(nxt)[1])
            out.add(tuple(chain))
    return sorted(out, key=lambda s: (len(s), s))


def pan_synergy_graph(pairs: pd.DataFrame, singles: pd.DataFrame, threshold: float) -> tuple[dict, dict]:
    """(node weight {pass: share of programs where the pass alone helps},
        edge weight {(a, b): share of programs where I(a,b) > threshold})."""
    edges = (pairs.assign(syn=pairs["interaction_score"] > threshold)
             .groupby(["pass_a", "pass_b"])["syn"].mean().to_dict())
    nodes = (singles.assign(h=singles["delta_b_alone"] > 0).groupby("pass_b")["h"].mean().to_dict())
    return nodes, edges


def weighted_walk(nodes: dict, edges: dict, length: int, rng: np.random.Generator) -> list[str]:
    """One sequence: start ∝ node weight, then step ∝ synergy edge weight (uniform if none)."""
    passes = sorted(nodes)
    w = np.array([nodes[p] for p in passes], float) + 1e-9
    seq = [passes[rng.choice(len(passes), p=w / w.sum())]]
    while len(seq) < length:
        cand = [p for p in passes if p != seq[-1]]
        ew = np.array([edges.get((seq[-1], p), 0.0) for p in cand], float)
        ew = ew + 1e-9 if ew.sum() == 0 else ew + 1e-9 * ew.max()
        seq.append(cand[rng.choice(len(cand), p=ew / ew.sum())])
    return seq


def static_graph_error(actual: pd.DataFrame, static_inter: dict, threshold: float, k: int = 10) -> dict:
    """How wrong a static graph is for ONE program. `actual`: that program's pair rows.

    sign_flip_rate        among pairs non-weak in the program, share where the static graph
                          predicts the opposite sign (synergy vs antagonism)
    class_error_rate      share of all pairs whose class (synergy / weak / antagonism) differs
    spearman              rank correlation over all pairs
    top_k_overlap         |top-k(static) ∩ top-k(actual)| / k, top = largest I
    mae                   mean absolute error
    """
    a = actual.set_index(["pass_a", "pass_b"])["interaction_score"]
    s = pd.Series({key: static_inter.get(key, 0.0) for key in a.index})

    def cls(v):
        return np.where(v > threshold, 1, np.where(v < -threshold, -1, 0))
    ca, cs = cls(a.to_numpy()), cls(s.to_numpy())
    nonweak = ca != 0
    rho = stats.spearmanr(a, s)[0] if a.std() > 0 and s.std() > 0 else np.nan
    return {"pairs": int(len(a)), "nonweak_pairs": int(nonweak.sum()),
            "sign_flip_rate": float((ca[nonweak] * cs[nonweak] < 0).mean()) if nonweak.any() else np.nan,
            "class_error_rate": float((ca != cs).mean()),
            "spearman": float(rho),
            "top_k_overlap": len(set(a.nlargest(k).index) & set(s.nlargest(k).index)) / k,
            "mae": float((a - s).abs().mean())}
