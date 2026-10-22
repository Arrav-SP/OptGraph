#!/usr/bin/env python3
"""v3 Phase 3 analysis (RQ1): how much of an interaction depends on pair, program and context.

  - variance decomposition of I over the crossed design pair x program x context;
  - stability across contexts: per-program rank correlation of the 462 pair interactions, and
    sign flips (synergy in one context, antagonism in the other);
  - order dependence: how often I(A,B,P) and I(B,A,P) differ, per context;
  - per-context aggregate graphs, per-program graph statistics and example graphs.
"""
import argparse
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import _bootstrap  # noqa: F401
from src.graph import aggregate_graph, draw_graph, edge_table, graph_statistics, program_graph
from src.reporting import BLUE, GREEN, GREY, ORANGE, PURPLE, RED, save_fig, save_table
from src.stats import variance_decomposition
from src.utils import DATA_DIR, OUTPUT_DIR, load_config, write_json

METRICS = ["instruction_count", "text_size", "dyn_instructions"]
TERM_COLORS = {"pair": BLUE, "program_id": ORANGE, "context": GREEN, "pair x program_id": PURPLE,
               "pair x context": "#9ecae1", "program_id x context": "#fdae6b",
               "pair x program_id x context": GREY}


def complete_grid(d: pd.DataFrame, contexts: list[str]) -> pd.DataFrame:
    """Programs that have every ordered pair in every context (one value per cell)."""
    pairs = d.groupby(["program_id", "context"])["pair"].nunique()
    n_pairs = d["pair"].nunique()
    full = pairs[pairs == n_pairs].reset_index().groupby("program_id")["context"].nunique()
    keep = full[full == len(contexts)].index
    return d[d["program_id"].isin(keep)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    thr = load_config()["settings"]["interaction_threshold"]
    data = DATA_DIR / "v3" / ("quick" if args.quick else "")
    out = OUTPUT_DIR / "v3" / ("quick" if args.quick else "")
    tables, figs, reports = out / "tables", out / "figures", out / "reports"
    ds = pd.read_csv(data / "interaction_dataset.csv.gz")
    ds = ds[ds["interaction_score"].notna()].copy()
    ds["pair"] = ds["pass_a"] + ">" + ds["pass_b"]
    contexts = sorted(ds["context"].unique())
    summary = {"threshold": thr, "contexts": contexts, "metrics": {}}

    vd_rows, stab_rows, flip_rows, order_rows, class_rows = [], [], [], [], []
    per_prog_corr = {}
    for m in METRICS:
        d = ds[ds["metric"] == m]
        if d.empty:
            continue
        s = {"programs": int(d["program_id"].nunique()), "rows": int(len(d))}
        # ---- class distribution per context
        for c, g in d.groupby("context"):
            vc = g["interaction_type"].value_counts()
            class_rows.append({"metric": m, "context": c, "rows": len(g),
                               **{k: int(vc.get(k, 0)) for k in ("SYNERGY", "WEAK", "ANTAGONISM")},
                               "nonweak_frac": float((g["interaction_type"] != "WEAK").mean())})
        # ---- variance decomposition on the complete crossed design
        grid = complete_grid(d, contexts)
        if grid["program_id"].nunique() >= 2 and len(contexts) >= 2:
            vd = variance_decomposition(grid, "interaction_score", ["pair", "program_id", "context"])
            for r in vd.itertuples():
                vd_rows.append({"metric": m, "term": r.term, "share": r.share,
                                "programs": int(grid["program_id"].nunique())})
            s["variance_shares"] = dict(zip(vd["term"], vd["share"]))
        # per context: share of variance explained by the pair alone (in-sample eta^2)
        s["pair_eta2_by_context"] = {}
        for c, g in d.groupby("context"):
            mu = g.groupby("pair")["interaction_score"].transform("mean")
            tot = ((g["interaction_score"] - g["interaction_score"].mean()) ** 2).sum()
            s["pair_eta2_by_context"][c] = float(1 - ((g["interaction_score"] - mu) ** 2).sum() / tot) if tot else None
        # ---- stability across contexts
        wide = d.pivot_table(index=["program_id", "pair"], columns="context",
                             values="interaction_score")
        cls = d.pivot_table(index=["program_id", "pair"], columns="context",
                            values="interaction_type", aggfunc="first")
        for ci, cj in itertools.combinations(contexts, 2):
            both = wide[[ci, cj]].dropna()
            if both.empty:
                continue
            rhos = []
            for pid, g in both.groupby(level="program_id"):
                if g[ci].std() > 0 and g[cj].std() > 0:
                    rhos.append(stats.spearmanr(g[ci], g[cj])[0])
            per_prog_corr[(m, ci, cj)] = rhos
            pair_means = both.groupby(level="pair").mean()
            stab_rows.append({
                "metric": m, "contexts": f"{ci} vs {cj}", "programs": len(rhos),
                "per_program_spearman_median": float(np.median(rhos)) if rhos else np.nan,
                "per_program_spearman_q1": float(np.quantile(rhos, 0.25)) if rhos else np.nan,
                "per_program_spearman_q3": float(np.quantile(rhos, 0.75)) if rhos else np.nan,
                "pooled_spearman": float(stats.spearmanr(both[ci], both[cj])[0]),
                "pair_mean_spearman": float(stats.spearmanr(pair_means[ci], pair_means[cj])[0]),
            })
            cb = cls.loc[both.index, [ci, cj]]
            syn_ant = ((cb[ci] == "SYNERGY") & (cb[cj] == "ANTAGONISM")) | \
                      ((cb[ci] == "ANTAGONISM") & (cb[cj] == "SYNERGY"))
            nonweak_either = (cb[ci] != "WEAK") | (cb[cj] != "WEAK")
            flip_rows.append({
                "metric": m, "contexts": f"{ci} vs {cj}", "cells": int(len(cb)),
                "sign_flips": int(syn_ant.sum()), "sign_flip_frac": float(syn_ant.mean()),
                "sign_flip_frac_of_nonweak": float(syn_ant.sum() / max(nonweak_either.sum(), 1)),
                "class_changed_frac": float((cb[ci] != cb[cj]).mean()),
                "pairs_with_any_flip": int(syn_ant.groupby(level="pair").any().sum()),
            })
        # ---- order dependence per context
        for c, g in d.groupby("context"):
            fwd = g.set_index(["program_id", "pass_a", "pass_b"])
            rev = fwd.rename_axis(["program_id", "pass_b", "pass_a"]).reorder_levels(
                ["program_id", "pass_a", "pass_b"])
            j = fwd[["interaction_score", "m_ab", "m_base"]].join(
                rev[["interaction_score", "m_ab"]], rsuffix="_rev", how="inner")
            j = j[j.index.get_level_values("pass_a") < j.index.get_level_values("pass_b")]
            diff = (j["interaction_score"] - j["interaction_score_rev"]).abs()
            order_rows.append({"metric": m, "context": c, "unordered_cells": int(len(j)),
                               "final_metric_differs_frac": float((j["m_ab"] != j["m_ab_rev"]).mean()),
                               "order_matters_frac": float((diff > thr).mean()),
                               "median_abs_asymmetry": float(diff.median())})
        summary["metrics"][m] = s

    vd_t, stab_t, flip_t = pd.DataFrame(vd_rows), pd.DataFrame(stab_rows), pd.DataFrame(flip_rows)
    order_t, class_t = pd.DataFrame(order_rows), pd.DataFrame(class_rows)
    if len(vd_t):
        vd_wide = vd_t.pivot_table(index="term", columns="metric", values="share").reset_index()
        save_table(vd_wide, "rq1_variance_decomposition", tables,
                   "Share of the variance of I(A,B,P) per term (crossed design pair x program x context)")
    save_table(stab_t, "rq1_context_stability", tables, "Rank stability of interactions across contexts")
    save_table(flip_t, "rq1_sign_flips", tables, "Synergy/antagonism sign flips between contexts")
    save_table(order_t, "rq1_order_dependence", tables, "How often the order of a pass pair matters, per context")
    save_table(class_t, "rq1_class_distribution", tables, f"Interaction classes per context (threshold {thr})")
    summary["stability"] = stab_rows
    summary["sign_flips"] = flip_rows
    summary["order"] = order_rows
    summary["classes"] = class_rows

    # ---- figures
    if len(vd_t):
        ms = [m for m in METRICS if m in set(vd_t["metric"])]
        terms = [t for t in TERM_COLORS if t in set(vd_t["term"])]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        left = np.zeros(len(ms))
        for t in terms:
            vals = np.array([vd_t[(vd_t.metric == m) & (vd_t.term == t)]["share"].sum() for m in ms])
            ax.barh(ms, vals, left=left, color=TERM_COLORS[t], label=t.replace("program_id", "program"))
            left += vals
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of variance of I(A,B,P,context)")
        ax.invert_yaxis()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)
        save_fig(fig, "rq1_variance_decomposition", figs)
    if per_prog_corr:
        keys = list(per_prog_corr)
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(keys) + 1.2))
        ax.boxplot([per_prog_corr[k] for k in keys], orientation="horizontal", widths=0.6,
                   tick_labels=[f"{m}: {a} vs {b}" for m, a, b in keys])
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel("per-program Spearman ρ of the 462 pair interactions between contexts")
        ax.invert_yaxis()
        save_fig(fig, "rq1_context_stability", figs)

    # ---- graphs per context (primary metric text_size; instruction_count as v2)
    gdir = out / "graphs"
    for m in ("text_size", "instruction_count"):
        d = ds[ds["metric"] == m]
        for c in contexts:
            dc = d[d["context"] == c]
            if dc.empty:
                continue
            agg = aggregate_graph(dc, m)
            reports.mkdir(parents=True, exist_ok=True)
            edge_table(agg).to_csv(reports / f"aggregate_edges_{m}_{c}.csv", index=False)
            graph_statistics(agg, thr).to_csv(reports / f"aggregate_stats_{m}_{c}.csv", index=False)
            per = []
            for pid in sorted(dc["program_id"].unique()):
                st = graph_statistics(program_graph(dc, pid, m), thr)
                st.insert(0, "program_id", pid)
                per.append(st)
            pd.concat(per).to_csv(reports / f"graph_stats_per_program_{m}_{c}.csv", index=False)
            if m == "text_size":
                for ext in ("png", "pdf"):
                    draw_graph(agg, gdir / f"aggregate_{m}_{c}.{ext}",
                               f"Mean interaction over {dc['program_id'].nunique()} programs, {m}, {c}", thr)
    examples = [p for p in ("pb_gemm", "ts_misc_dt", "mix_01_matrix_mul") if p in set(ds["program_id"])]
    for pid in examples:
        for c in contexts:
            g = program_graph(ds[(ds["context"] == c) & (ds["metric"] == "text_size")], pid, "text_size")
            if g.number_of_edges():
                for ext in ("png", "pdf"):
                    draw_graph(g, figs / f"rq1_example_graph_{pid}_{c}.{ext}", f"{pid}, text_size, {c}", thr)
    summary["example_graphs"] = examples
    write_json(reports / "rq1_summary.json", summary)
    print(stab_t.round(3).to_string(index=False))
    print(flip_t.round(4).to_string(index=False))
    print(order_t.round(3).to_string(index=False))
    if len(vd_t):
        print(vd_t.pivot_table(index="term", columns="metric", values="share").round(3).to_string())
    print(f"-> {tables}, {figs}, {reports}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
