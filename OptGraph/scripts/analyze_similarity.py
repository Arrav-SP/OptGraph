#!/usr/bin/env python3
"""Hypothesis check: do structurally similar programs have similar interaction patterns?"""
import argparse
import sys

import pandas as pd

import _bootstrap  # noqa: F401
from src import evaluation as ev
from src.utils import Workspace, load_config, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--metric")
    args = ap.parse_args()
    settings = load_config()["settings"]
    metric = args.metric or settings["primary_metric"]
    ws = Workspace(args.tag)
    if not (ws.dataset.exists() and ws.features.exists()):
        print("dataset not found - run scripts/run_experiments.py first", file=sys.stderr)
        return 1
    feats = pd.read_csv(ws.features)
    res = ev.similarity_analysis(feats, pd.read_csv(ws.dataset), metric, settings["seed"])
    ev.plot_similarity(res, dict(zip(feats["program_id"], feats["category"])),
                       ws.outputs / "plots" / "similarity_structure_vs_interaction.png")
    public = {k: v for k, v in res.items() if not k.startswith("_")}
    write_json(ws.outputs / "reports" / "similarity_analysis.json", public)
    print(f"programs: {res['n_programs']}   program pairs: {res['n_program_pairs']}   "
          f"features: {res['n_features']}   ordered pass pairs: {res['n_pass_pairs']}")
    print(f"Spearman(structural similarity, interaction similarity) = {res['spearman']:.3f}")
    print(f"Pearson = {res['pearson']:.3f}")
    print(f"Mantel permutation test, one-sided p = {res['mantel_p_one_sided']:.4f} "
          f"({res['permutations']} permutations)")
    print(f"(uncentred interaction vectors: Spearman = {res['spearman_raw_uncentered']:.3f})")
    print("Small sample: treat as exploratory evidence, not a significance claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
