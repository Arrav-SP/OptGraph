#!/usr/bin/env python3
"""Descriptive statistics of the measured interactions (no modelling).

Answers, from the data alone: how often do passes interact, how much of the
variation is explained by the pass pair versus the program, which pairs change
sign between programs, and how often order matters.
"""
import argparse
import json
import sys

import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import Workspace, load_config, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--metric")
    args = ap.parse_args()
    settings = load_config()["settings"]
    metric = args.metric or settings["primary_metric"]
    thr = settings["interaction_threshold"]
    ws = Workspace(args.tag)
    if not ws.dataset.exists():
        print("dataset not found - run scripts/run_experiments.py first", file=sys.stderr)
        return 1
    ds = pd.read_csv(ws.dataset)
    run = json.loads(ws.run_config.read_text())
    d = ds[(ds.metric == metric) & ds.interaction_score.notna()].copy()
    s = d.interaction_score

    # Share of variance explained by the ordered pair alone (in-sample eta squared).
    pair_mean = d.groupby(["pass_a", "pass_b"]).interaction_score.transform("mean")
    eta_pair = 1 - ((s - pair_mean) ** 2).sum() / ((s - s.mean()) ** 2).sum()

    by_pair = d.groupby(["pass_a", "pass_b"]).agg(
        mean=("interaction_score", "mean"), std=("interaction_score", "std"),
        min=("interaction_score", "min"), max=("interaction_score", "max"),
        n_synergy=("interaction_type", lambda x: (x == "SYNERGY").sum()),
        n_antagonism=("interaction_type", lambda x: (x == "ANTAGONISM").sum()),
        n_weak=("interaction_type", lambda x: (x == "WEAK").sum())).reset_index()
    sign_flip = by_pair[(by_pair.n_synergy > 0) & (by_pair.n_antagonism > 0)]
    always_weak = by_pair[(by_pair.n_synergy == 0) & (by_pair.n_antagonism == 0)]

    # Order dependence: compare I(A,B,P) with I(B,A,P).
    rev = d.rename(columns={"pass_a": "pass_b", "pass_b": "pass_a",
                            "interaction_score": "reverse_score", "m_ab": "m_ba"})
    both = d.merge(rev[["program_id", "pass_a", "pass_b", "reverse_score", "m_ba"]],
                   on=["program_id", "pass_a", "pass_b"])
    both = both[both.pass_a < both.pass_b]
    asym = (both.interaction_score - both.reverse_score).abs() > thr

    out = {
        "metric": metric, "threshold": thr,
        "programs": int(d.program_id.nunique()),
        "passes": int(pd.concat([d.pass_a, d.pass_b]).nunique()),
        "experiments_total": run["experiments_total"],
        "experiments_success": run["experiments_success"],
        "experiments_failed": run["experiments_failed"],
        "outputs_verified": run["verify_outputs"],
        "interaction_rows_all_metrics": int(len(ds)),
        "undefined_rows_all_metrics": int((ds.interaction_type == "UNDEFINED").sum()),
        "undefined_by_metric": ds[ds.interaction_type == "UNDEFINED"].metric.value_counts().to_dict(),
        "rows_primary_metric": int(len(d)),
        "class_counts": d.interaction_type.value_counts().to_dict(),
        "score_min": float(s.min()), "score_max": float(s.max()),
        "score_mean": float(s.mean()), "score_std": float(s.std()),
        "variance_explained_by_pair_identity": float(eta_pair),
        "variance_not_explained_by_pair_identity": float(1 - eta_pair),
        "ordered_pairs": int(len(by_pair)),
        "pairs_sign_flip_across_programs": int(len(sign_flip)),
        "pairs_always_weak": int(len(always_weak)),
        "unordered_program_pairs": int(len(both)),
        "order_matters_count": int(asym.sum()),
        "order_matters_fraction": float(asym.mean()),
    }
    rdir = ws.outputs / "reports"
    write_json(rdir / "results_summary.json", out)
    by_pair.sort_values("mean").to_csv(rdir / "pair_statistics.csv", index=False)

    print(json.dumps(out, indent=2))
    cols = ["pass_a", "pass_b", "mean", "std", "min", "max", "n_synergy", "n_antagonism", "n_weak"]
    print("\nMost antagonistic pairs (mean over programs):")
    print(by_pair.nsmallest(6, "mean")[cols].round(4).to_string(index=False))
    print("\nMost synergistic pairs (mean over programs):")
    print(by_pair.nlargest(6, "mean")[cols].round(4).to_string(index=False))
    print("\nMost program-dependent pairs (largest std across programs):")
    print(by_pair.nlargest(6, "std")[cols].round(4).to_string(index=False))
    print("\nPairs that are synergy in some programs and antagonism in others:")
    print(sign_flip.sort_values("std", ascending=False).head(8)[cols].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
