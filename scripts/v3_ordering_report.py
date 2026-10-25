"""v3 Phase 5 report: geometric-mean improvements, win/tie/loss, tests and cost vs benefit.

All comparisons are per program on the same IR input and the same backend. A variant counts
only if it built and reproduced the program's -O0 output; a program missing a valid value for
either side of a comparison is left out of that comparison (the count is reported).
Improvement over -O0 = 1 - geomean(v / v_O0); 'vs O2' = geomean(v / v_O2) (< 1: better than O2).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.reporting import BLUE, GREEN, GREY, ORANGE, PURPLE, RED, save_fig, save_table
from src.stats import bootstrap_ci, cv, holm, paired_test
from src.utils import version, DATA_DIR, write_json

METRICS = ["text_size", "dyn_instructions", "instruction_count", "runtime"]
LEVELS = ["O1", "O2", "O3", "Os"]
GRAPH = [f"graph_{s}_{b}" for s in ("predicted", "agnostic", "oracle") for b in ("greedy", "beam")]
COLORS = {"canonical": GREY, "random_median": "#bdbdbd", "random_best": "#636363",
          "graph_predicted_greedy": BLUE, "graph_predicted_beam": "#6baed6",
          "graph_agnostic_greedy": ORANGE, "graph_agnostic_beam": "#fdae6b",
          "graph_oracle_greedy": GREEN, "graph_oracle_beam": "#74c476", "search": PURPLE,
          "O1": "#fcbba1", "O2": "#fb6a4a", "O3": RED, "Os": "#67000d"}


def gmean(x) -> float:
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean()))


def runtime_table(rt: pd.DataFrame, cv_flag: float, seed: int) -> pd.DataFrame:
    rows = []
    for r in rt.itertuples():
        if isinstance(r.error, str) and r.error:
            continue
        t = json.loads(r.times)
        lo, hi = bootstrap_ci(t, np.median, 1000, seed)
        c = cv(t)
        rows.append({"program_id": r.program_id, "ir_sha256": r.ir_sha256, "runtime": float(np.median(t)),
                     "runtime_ci_lo": lo, "runtime_ci_hi": hi, "runtime_cv": c, "noisy": bool(c > cv_flag)})
    return pd.DataFrame(rows)


def strategy_values(df: pd.DataFrame, objective: str, k: int, metric: str) -> pd.DataFrame:
    """program x strategy table of `metric` for the strategies comparable at budget k."""
    ok = df[(df["error"].fillna("") == "") & (df["output_matches"] == True) & df[metric].notna()]  # noqa: E712
    cols = {}
    for s in ["O0"] + LEVELS:
        cols[s] = ok[ok["strategy"] == s].set_index("program_id")[metric]
    cols["canonical"] = ok[(ok["strategy"] == "canonical") & (ok["k"] == k)].set_index("program_id")[metric]
    rnd = ok[(ok["strategy"] == "random") & (ok["k"] == k)]
    if metric == "runtime":   # runtime only exists for the median/best random sequences (by objective)
        roles = ok[(ok["strategy"] == "random_role") & (ok["k"] == k)]
        for role in ("median", "best"):
            r = roles[roles["role"] == f"random_{role}_{objective}"]
            cols[f"random_{role}"] = r.drop_duplicates("program_id").set_index("program_id")[metric]
    else:
        cols["random_median"] = rnd.groupby("program_id")[metric].median()
        # best = the random sequence that is best on the objective, measured on this metric
        best = rnd.sort_values([objective, "idx"]).drop_duplicates("program_id")
        cols["random_best"] = best.set_index("program_id")[metric]
    for s in GRAPH + ["search"]:
        sel = ok[(ok["strategy"] == s) & (ok["k"] == k) & (ok["objective"] == objective)]
        cols[s] = sel.drop_duplicates("program_id").set_index("program_id")[metric]
    return pd.DataFrame(cols)


def report(data, out, ocfg: dict, seed: int) -> dict:
    tables, figs, reps = out / "tables", out / "figures", out / "reports"
    df = pd.read_csv(data / "ordering_measurements.csv.gz")
    df["role"] = ""
    rt_path = data / "ordering_runtime.csv"
    noise = pd.DataFrame()
    if rt_path.exists():
        noise = runtime_table(pd.read_csv(rt_path), ocfg["runtime_cv_flag"], seed)
        df = df.merge(noise[["program_id", "ir_sha256", "runtime"]], on=["program_id", "ir_sha256"], how="left")
        # runtime of a random sequence is only kept for its key roles
        from v3_ordering import key_variants
        roles = key_variants(df.drop(columns=["role"]))
        roles = roles[roles["strategy"] == "random"][["program_id", "k", "idx", "role"]]
        # Key random sequences become extra 'random_role' rows; plain random rows carry no runtime.
        role_rows = df[df["strategy"] == "random"].drop(columns=["role"]).merge(
            roles, on=["program_id", "k", "idx"], how="inner").assign(strategy="random_role")
        df.loc[df["strategy"] == "random", "runtime"] = np.nan
        df = pd.concat([df, role_rows], ignore_index=True)
    else:
        df["runtime"] = np.nan
    manifest = pd.read_csv(DATA_DIR / version() / "benchmarks.csv", keep_default_na=False)
    reliable = set(manifest[manifest["runtime_reliable"].astype(str) == "True"]["program_id"])

    failures = df[df["error"].fillna("") != ""]
    summary = {"programs": int(df["program_id"].nunique()), "budgets": ocfg["budgets"],
               "objectives": ocfg["objectives"], "variants": int((df["strategy"] != "random_role").sum()),
               "failed_variants": int(failures.drop_duplicates(["program_id", "strategy", "objective", "k", "idx"]).shape[0]),
               "failure_examples": failures["error"].value_counts().head(10).to_dict(),
               "geomean": [], "tests": [], "wtl": []}
    if len(noise):
        rel = noise[noise["program_id"].isin(reliable)]
        summary["runtime_noise"] = {"measured": int(len(rel)), "noisy": int(rel["noisy"].sum()),
                                    "cv_median": float(rel["runtime_cv"].median()),
                                    "cv_p90": float(rel["runtime_cv"].quantile(0.9)), "cv_flag": ocfg["runtime_cv_flag"]}
        save_table(rel.describe()[["runtime", "runtime_cv"]], "rq3_runtime_noise", tables,
                   "Runtime measurement noise (median of 10 pinned runs)", index=True)

    gm_rows, test_rows, wtl_rows = [], [], []
    for obj in ocfg["objectives"]:
        for k in ocfg["budgets"]:
            for metric in METRICS:
                t = strategy_values(df, obj, k, metric)
                if metric == "runtime":
                    t = t[t.index.isin(reliable)]
                t = t.dropna(subset=["O0"])
                for s in t.columns:
                    v = t[list(dict.fromkeys([s, "O0", "O2", "O3"]))].dropna()
                    v = v[(v > 0).all(axis=1)]
                    if v.empty:
                        continue
                    gm_rows.append({"objective": obj, "k": k, "metric": metric, "strategy": s, "programs": len(v),
                                    "improvement_vs_O0": 1 - gmean(v[s] / v["O0"]),
                                    "ratio_vs_O2": gmean(v[s] / v["O2"]), "ratio_vs_O3": gmean(v[s] / v["O3"])})
                if metric != obj and not (metric == "runtime" and obj == "dyn_instructions"):
                    continue
                # key comparisons on the objective metric (and runtime as confirmation of dyn count)
                pairs = [(f"graph_predicted_{b}", other) for b in ("greedy", "beam")
                         for other in (f"graph_agnostic_{b}", "random_median", "canonical", "O2", "O3",
                                       f"graph_oracle_{b}", "search")]
                for a, b in pairs:
                    if a not in t or b not in t:
                        continue
                    v = t[[a, b]].dropna()
                    v = v[(v > 0).all(axis=1)]
                    if len(v) < 2:
                        continue
                    la, lb = np.log(v[a]), np.log(v[b])
                    res = paired_test(la, lb, 2000, seed)
                    row = {"objective": obj, "k": k, "metric": metric, "a": a, "b": b, "programs": len(v),
                           "gmean_ratio_a_over_b": float(np.exp(res["mean_diff"])),
                           "ci95_lo": float(np.exp(res["ci95_mean_diff"][0])),
                           "ci95_hi": float(np.exp(res["ci95_mean_diff"][1])),
                           "wins": res["a_better"], "ties": res["ties"], "losses": res["b_better"],
                           "wilcoxon_p": res["p_value"]}
                    (test_rows if b in (f"graph_agnostic_{a.split('_')[-1]}", "random_median") else wtl_rows).append(row)
    gm = pd.DataFrame(gm_rows)
    tests = pd.DataFrame(test_rows)
    if len(tests):
        tests["wilcoxon_p_holm"] = holm(tests["wilcoxon_p"])   # one family: every key comparison
    wtl = pd.DataFrame(wtl_rows)
    save_table(gm, "rq3_geomean_improvement", tables,
               "Geometric-mean improvement over -O0 and ratio to -O2/-O3 (same programs, same backend)")
    save_table(tests, "rq3_key_tests", tables,
               "Program-conditioned (predicted) vs program-agnostic graph and vs random median; Wilcoxon on per-program log ratios, Holm-corrected")
    save_table(wtl, "rq3_win_tie_loss", tables, "Predicted-graph sequences vs other strategies, per program")
    summary["geomean"], summary["tests"], summary["wtl"] = gm.to_dict("records"), tests.to_dict("records"), wtl.to_dict("records")

    # ---- cost vs benefit (objective metric, largest budget)
    kmax = max(ocfg["budgets"])
    cost_rows = []
    for obj in ocfg["objectives"]:
        ok = df[(df["error"].fillna("") == "") & (df["output_matches"] == True) & df[obj].notna()]  # noqa: E712
        o0 = ok[ok["strategy"] == "O0"].set_index("program_id")[obj]
        for k in ocfg["budgets"]:
            rnd = ok[(ok["strategy"] == "random") & (ok["k"] == k)]
            for n in range(1, ocfg["random_per_budget"] + 1):
                best = rnd[rnd["idx"] < n].groupby("program_id")[obj].min()
                v = pd.concat([best, o0], axis=1, keys=["v", "o0"]).dropna()
                cost_rows.append({"objective": obj, "k": k, "strategy": "random_best_of_n", "evals": n,
                                  "improvement_vs_O0": 1 - gmean(v["v"] / v["o0"]), "programs": len(v)})
            for s in GRAPH + ["search"]:
                sel = ok[(ok["strategy"] == s) & (ok["k"] == k) & (ok["objective"] == obj)].set_index("program_id")
                v = pd.concat([sel[obj], o0, sel["evals"]], axis=1, keys=["v", "o0", "e"]).dropna()
                if len(v):
                    cost_rows.append({"objective": obj, "k": k, "strategy": s, "evals": float(v["e"].mean()),
                                      "improvement_vs_O0": 1 - gmean(v["v"] / v["o0"]), "programs": len(v)})
        for s in LEVELS:
            sel = ok[ok["strategy"] == s].set_index("program_id")
            v = pd.concat([sel[obj], o0], axis=1, keys=["v", "o0"]).dropna()
            cost_rows.append({"objective": obj, "k": 0, "strategy": s, "evals": 1.0,
                              "improvement_vs_O0": 1 - gmean(v["v"] / v["o0"]), "programs": len(v)})
    cost = pd.DataFrame(cost_rows)
    save_table(cost[cost["strategy"] != "random_best_of_n"], "rq3_cost", tables,
               "Compile evaluations needed per program (graph methods: 1 compile after prediction)")
    cost.to_csv(reps / "rq3_cost_curve.csv", index=False)

    fig, axes = plt.subplots(1, len(ocfg["objectives"]), figsize=(5.4 * len(ocfg["objectives"]), 4.0), squeeze=False)
    for ax, obj in zip(axes[0], ocfg["objectives"]):
        c = cost[cost["objective"] == obj]
        r = c[(c["strategy"] == "random_best_of_n") & (c["k"] == kmax)]
        ax.plot(r["evals"], 100 * r["improvement_vs_O0"], color=GREY, label=f"best of n random (k={kmax})")
        s = c[(c["strategy"] == "search")].sort_values("k")
        ax.plot(s["evals"], 100 * s["improvement_vs_O0"], "o-", color=PURPLE, label="measured greedy search (k=8,12,16)")
        for st, mk in (("graph_predicted_greedy", "o"), ("graph_agnostic_greedy", "s"), ("graph_oracle_greedy", "^")):
            g = c[(c["strategy"] == st) & (c["k"] == kmax)]
            ax.scatter(g["evals"], 100 * g["improvement_vs_O0"], marker=mk, s=50, color=COLORS[st], zorder=3,
                       label=st.replace("graph_", "").replace("_", " ") + f" (k={kmax})")
        for st in LEVELS:
            g = c[c["strategy"] == st]
            ax.axhline(100 * g["improvement_vs_O0"].iloc[0], color=COLORS[st], lw=0.9, ls="--")
            ax.text(1.05, 100 * g["improvement_vs_O0"].iloc[0], f"-{st}",
                    color=COLORS[st], fontsize=7, va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("compile evaluations per program (log)")
        ax.set_ylabel(f"geomean {obj} reduction vs -O0 (%)")
        ax.set_title(f"objective: {obj}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    save_fig(fig, "rq3_cost_vs_benefit", figs)

    # ---- strategies vs -O levels, per metric (objective = dyn_instructions if present)
    obj = "dyn_instructions" if "dyn_instructions" in ocfg["objectives"] else ocfg["objectives"][0]
    order = ["canonical", "random_median", "random_best", "graph_agnostic_greedy", "graph_predicted_greedy",
             "graph_predicted_beam", "graph_oracle_greedy", "search"]
    mets = [m for m in METRICS if len(gm[(gm.metric == m) & (gm.objective == obj)])]
    fig, axes = plt.subplots(1, len(mets), figsize=(4.0 * len(mets), 4.6), squeeze=False)
    for ax, m in zip(axes[0], mets):
        g = gm[(gm.objective == obj) & (gm.metric == m)]
        width = 0.8 / len(ocfg["budgets"])
        for i, k in enumerate(ocfg["budgets"]):
            vals = [g[(g.k == k) & (g.strategy == s)]["improvement_vs_O0"].sum() * 100 for s in order]
            ax.barh(np.arange(len(order)) + i * width, vals, height=width, label=f"k={k}",
                    color=[BLUE, ORANGE, GREEN][i % 3], alpha=0.85)
        for st in LEVELS:
            lv = g[g.strategy == st]["improvement_vs_O0"]
            if len(lv):
                ax.axvline(100 * lv.iloc[0], color=COLORS[st], ls="--", lw=1)
                ax.text(100 * lv.iloc[0], -0.9, st, color=COLORS[st], fontsize=7, ha="center")
        ax.set_yticks(np.arange(len(order)) + width, [s.replace("graph_", "") for s in order], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"geomean {m} reduction vs -O0 (%)", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Sequences built for objective {obj}; dashed lines: -O levels on the same programs", fontsize=9)
    save_fig(fig, "rq3_ordering_vs_olevels", figs)
    write_json(reps / "rq3_summary.json", summary)
    key = gm[(gm.metric == gm.objective) & (gm.k == kmax)][["objective", "strategy", "programs",
                                                             "improvement_vs_O0", "ratio_vs_O2", "ratio_vs_O3"]]
    print(key.round(4).to_string(index=False))
    if len(tests):
        print(tests[["objective", "k", "metric", "a", "b", "programs", "gmean_ratio_a_over_b", "wins", "ties",
                     "losses", "wilcoxon_p_holm"]].round(4).to_string(index=False))
    return summary
