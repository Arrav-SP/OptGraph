#!/usr/bin/env python3
"""v3.1 exploratory analyses for the paper (NOT pre-registered: raw one-sided Wilcoxon p-values
over programs, no multiple-comparison correction).

  A  why re-prediction loses (Claim 4): sequence builder vs model accuracy
  B  does Claim 1's headroom carry to .text size, AArch64 size, executed instructions, runtime?
  C  drift along the -O2 trajectory: every snapshot pair, IR identity of consecutive snapshots

Writes outputs/v3_1/ANALYSIS.md, tables/v3_1_analysis_*.csv, figures/v3_1_drift_*.png.
Run with OPTGRAPH_CONFIG=v3_1.yaml after the full run.
"""
import hashlib
import json
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import _bootstrap  # noqa: F401
from src.reporting import BLUE, ORANGE, save_fig, save_table
from src.utils import DATA_DIR, OUTPUT_DIR, load_config, load_v3_config, work_dir
from v3_1_claims import load
from v3_1_report import one_sided, per_program

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
DATA, OUT = DATA_DIR / "v3_1", OUTPUT_DIR / "v3_1"
TABLES, FIGS = OUT / "tables", OUT / "figures"
SNAPS = ["S0", "S1", "S2", "S3", "S4"]
GRAPH = ["oracle_graph", "predict_once", "repredict", "static_graph", "graph_guided"]
SEARCH = ["pan_style", "dco_style", "opentuner", "ga", "random"]
SEED = 42


def fmt(r: dict) -> str:
    return (f"{r['geomean_ratio']:.3f} (95% CI {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}) | "
            f"{r['wins']}/{r['losses']}/{r['ties']} | {r['programs']} | {r['p_one_sided']:.2g}")


def strongest(search, methods, c, oz):
    r = {m: float(np.exp(np.log(per_program(search, m, c) / oz).dropna().mean())) for m in methods}
    return min(r, key=r.get)


def part_a(search, oz):
    L = ["## A. Why re-prediction loses (Claim 4)", "",
         "Geomean best-so-far IR count / -Oz by compile budget E (lower is better):", ""]
    cur = pd.read_csv(TABLES / "v3_1_budget_curves.csv")
    piv = cur.pivot(index="method", columns="checkpoint", values="geomean_ratio_vs_Oz")
    cols = [1, 5, 10, 25, 200]
    L += ["| method | " + " | ".join(f"E={c}" for c in cols) + " | mean E used at 200 |",
          "|---|" + "---|" * (len(cols) + 1)]
    used = cur[cur.checkpoint == 200].set_index("method")["mean_E"]
    for m in GRAPH + SEARCH + ["measured_greedy"]:
        L.append(f"| {m} | " + " | ".join(f"{piv.loc[m, c]:.3f}" for c in cols) + f" | {used[m]:.0f} |")
    L += ["", "Paired comparisons (ratio = geomean of a/b over programs; better/worse/tied; programs; p one-sided a < b):", "",
          "| a | b | E | ratio (95% CI) | better/worse/tied | n | p |", "|---|---|---|---|---|---|---|"]
    rows = []
    s1, s5, s10 = (strongest(search, SEARCH, c, oz) for c in (1, 5, 10))
    for a, b, c, why in (("oracle_graph", s10, 10, "perfect graph vs strongest search: is the builder the limit?"),
                         ("repredict", "static_graph", 10, "does re-conditioning beat a static graph?"),
                         ("repredict", "predict_once", 10, "does re-conditioning beat predicting once?"),
                         ("predict_once", s1, 1, "graph guidance at the smallest budget"),
                         ("repredict", s1, 1, "re-prediction at the smallest budget"),
                         ("oracle_graph", s1, 1, "headroom at the smallest budget"),
                         ("predict_once", s5, 5, "graph guidance at E = 5"),
                         ("repredict", s5, 5, "re-prediction at E = 5")):
        r = one_sided(per_program(search, a, c), per_program(search, b, c), SEED)
        rows.append({"a": a, "b": b, "E": c, "question": why, **r})
        L.append(f"| {a} | {b} | {c} | " + fmt(r) + " |")
    # Plateau: share of the E=1 objective each method still removes by E=200.
    gain = {m: float(((per_program(search, m, 1) - per_program(search, m, 200)) / per_program(search, m, 1)).median())
            for m in GRAPH + SEARCH}
    L += ["", "Median per-program improvement from E = 1 to E = 200 (graph builders plateau):", "",
          "| method | improvement |", "|---|---|"] + [f"| {m} | {v:.1%} |" for m, v in gain.items()]
    # Model accuracy for single-pass effects (what re-prediction uses), by snapshot.
    sp = pd.read_csv(DATA / "claims" / "lopo_singles.csv.gz")
    acc = []
    for (pid, ctx), g in sp.groupby(["program_id", "context"]):
        if g["interaction_score"].std() > 0:
            acc.append({"context": ctx, "model": stats.spearmanr(g["predicted"], g["interaction_score"])[0],
                        "static": stats.spearmanr(g["static"], g["interaction_score"])[0]})
    acc = pd.DataFrame(acc).groupby("context")[["model", "static"]].median()
    L += ["", "Single-pass effect ranking (median per-program Spearman with the measured effects, LOPO):", "",
          "| snapshot | model | static graph |", "|---|---|---|"] + \
         [f"| {c} | {r.model:.3f} | {r.static:.3f} |" for c, r in acc.iterrows()]
    save_table(pd.DataFrame(rows), "v3_1_analysis_A_comparisons", TABLES, "Why re-prediction loses (exploratory)")
    return L + [""]


def part_b(search):
    meas = pd.read_csv(DATA / "ordering_measure.csv.gz", keep_default_na=False)
    meas = meas[meas["output_matches"].astype(str) == "True"].copy()
    for c in ("instruction_count", "text_size", "text_size_aarch64", "dyn_instructions"):
        meas[c] = pd.to_numeric(meas[c], errors="coerce")
    rt = pd.read_csv(DATA / "ordering_runtime.csv", keep_default_na=False)
    rt = rt[rt["error"] == ""].copy()
    rt["runtime"] = rt["times"].map(lambda t: float(np.median(json.loads(t))))
    meas = meas.merge(rt[["program_id", "ir_sha256", "runtime"]].drop_duplicates(["program_id", "ir_sha256"]),
                      on=["program_id", "ir_sha256"], how="left")
    j = search.drop(columns="best").merge(meas, on=["program_id", "best_seq"], how="inner")

    def metric(m, c, col):
        s = j[(j["method"] == m) & (j["checkpoint"] == c) & (j[col] > 0)]
        return s.groupby("program_id")[col].apply(lambda v: float(np.exp(np.log(v).mean())))

    cols = {"instruction_count": "IR instructions", "text_size": ".text x86-64", "text_size_aarch64": ".text AArch64",
            "dyn_instructions": "executed instructions", "runtime": "runtime (median of 20)"}
    L = ["## B. Does the headroom carry to real metrics?", "",
         "Each method's best-so-far sequence (output-checked), per program; stochastic methods: geomean over 5 seeds. "
         "Runtime only for runtime-reliable programs and checkpoints 0/10/25/200.", "",
         "| a | b | metric | ratio (95% CI) | better/worse/tied | n | p |", "|---|---|---|---|---|---|---|"]
    rows = []
    for (a, ca), (b, cb) in ((("oracle_graph", 10), ("static_graph", 10)), (("oracle_graph", 200), ("static_graph", 200)),
                             (("repredict", 10), ("ga", 10)), (("ga", 200), ("O3", 0)), (("ga", 200), ("Oz", 0))):
        for col, name in cols.items():
            r = one_sided(metric(a, ca, col), metric(b, cb, col), SEED)
            if r["programs"] == 0:
                continue
            rows.append({"a": f"{a}@{ca}", "b": f"{b}@{cb}", "metric": col, **r})
            L.append(f"| {a} @E={ca} | {b} @E={cb} | {name} | " + fmt(r) + " |")
    save_table(pd.DataFrame(rows), "v3_1_analysis_B_metrics", TABLES, "Headroom in measured metrics (exploratory)")
    return L + [""]


def part_c():
    thr = load_config()["settings"]["interaction_threshold"]
    ds = load(DATA)
    w = ds.pivot_table(index=["program_id", "pass_a", "pass_b"], columns="context", values="interaction_score")
    M = pd.DataFrame(1.0, index=SNAPS, columns=SNAPS)
    for i, a in enumerate(SNAPS):
        for b in SNAPS[i + 1:]:
            rho = []
            for _, g in w[[a, b]].dropna().groupby(level="program_id"):
                g = g[(g[a].abs() > thr) | (g[b].abs() > thr)]
                if len(g) >= 5 and g[a].std() > 0 and g[b].std() > 0:
                    rho.append(stats.spearmanr(g[a], g[b])[0])
            M.loc[a, b] = M.loc[b, a] = float(np.nanmedian(rho))
    save_table(M.reset_index(names="snapshot"), "v3_1_analysis_C_drift_matrix", TABLES,
               "Median per-program Spearman rho between snapshots (exploratory)")
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(M.to_numpy(float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(5), SNAPS)
    ax.set_yticks(range(5), SNAPS)
    for i in range(5):
        for k in range(5):
            v = M.iat[i, k]
            ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="median Spearman ρ")
    save_fig(fig, "v3_1_drift_matrix", FIGS)
    dec = pd.read_csv(DATA / "claims" / "drift_decay.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax2 = ax.twinx()
    ax.plot(dec.snapshot, dec.median_rho_vs_S0, "o-", color=BLUE)
    ax.set_ylabel("median Spearman ρ vs S0", color=BLUE)
    ax2.plot(dec.snapshot, dec.share_pairs_near_zero, "s--", color=ORANGE)
    ax2.set_ylabel("share of pairs with |I| ≤ threshold", color=ORANGE)
    for a in (ax, ax2):
        a.set_ylim(0, 1.05)
    ax.set_xlabel("snapshot along default<O2> (S0 raw IR ... S4 full -O2)")
    save_fig(fig, "v3_1_drift_decay", FIGS)
    # How often consecutive snapshots are the exact same IR (explains rho = 1).
    exp = work_dir(load_v3_config()) / "v3_1" / "full" / "experiments"
    same = {}
    for a, b in zip(SNAPS, SNAPS[1:]):
        n = t = 0
        for d in exp.iterdir():
            fa, fb = d / a / "baseline.ll", d / b / "baseline.ll"
            if fa.exists() and fb.exists():
                t += 1
                n += hashlib.sha256(fa.read_bytes()).digest() == hashlib.sha256(fb.read_bytes()).digest()
        same[f"{a}->{b}"] = (n, t)
    L = ["## C. Drift along the -O2 trajectory", "",
         "Median per-program Spearman ρ between snapshots (pairs with |I| > threshold in either):", "",
         "| | " + " | ".join(SNAPS) + " |", "|---|" + "---|" * 5]
    L += [f"| {a} | " + " | ".join(f"{M.loc[a, b]:.2f}" for b in SNAPS) + " |" for a in SNAPS]
    L += ["", "Share of pairs with |I| <= threshold: " +
          ", ".join(f"{r.snapshot} {r.share_pairs_near_zero:.1%}" for r in dec.itertuples()), "",
          "Consecutive snapshots with byte-identical start IR (programs): " +
          ", ".join(f"{k} {n}/{t}" for k, (n, t) in same.items()), "",
          "Figures: figures/v3_1_drift_decay.png, figures/v3_1_drift_matrix.png", ""]
    return L


def main() -> int:
    search = pd.read_csv(DATA / "ordering_search.csv.gz", keep_default_na=False)
    search["best"] = pd.to_numeric(search["best"], errors="coerce")
    oz = search[search["method"] == "Oz"].set_index("program_id")["best"]
    L = ["# OptGraph v3.1: exploratory analyses", "",
         "Not pre-registered. Raw one-sided Wilcoxon p-values over programs, no correction. "
         "Generated by scripts/v3_1_analysis.py from the full run's data files; rerun it to regenerate.", ""]
    tk = OUT / "ANALYSIS_TAKEAWAYS.md"          # hand-written reading of the numbers below
    L += ([tk.read_text(), ""] if tk.exists() else []) + part_a(search, oz) + part_b(search) + part_c()
    (OUT / "ANALYSIS.md").write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
