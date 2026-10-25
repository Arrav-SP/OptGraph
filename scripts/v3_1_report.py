#!/usr/bin/env python3
"""v3.1 statistics and report: the four pre-registered hypotheses (Holm across exactly these
four), exploratory tables and figures, outputs/v3_1/report/index.html, RESULTS.md,
results_v3_1.json. Every number is read from files the other v3.1 scripts wrote.

Pre-registered choices implemented here (outputs/v3_1/PREREGISTRATION.md is authoritative):
  objective   IR instruction count, as a ratio to the program's -Oz IR count
  H1          oracle_graph vs static_graph, best-so-far at E = 10; one-sided Wilcoxon on per-program
              log ratios (oracle < static); supported if Holm p < 0.05 and geomean ratio <= 0.98
  H2, H3      from reports/claims_summary.json (scripts/v3_1_claims.py)
  H4          repredict vs the strongest other method at E = 10, the pool being predict_once,
              static_graph, graph_guided, pan_style, dco_style, opentuner, ga, random (strongest =
              lowest geomean ratio to -Oz over all programs); stochastic methods use the per-program
              geometric mean over their 5 seeds; same test and thresholds as H1
"""
import argparse
import base64
import html
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import _bootstrap  # noqa: F401
from src.reporting import BLUE, GREEN, GREY, ORANGE, PURPLE, RED, save_fig, save_table
from src.stats import bootstrap_ci, cv, holm
from src.utils import DATA_DIR, OUTPUT_DIR, load_v3_config, now_iso, version, write_json

POOL = ["predict_once", "static_graph", "graph_guided", "pan_style", "dco_style", "opentuner", "ga", "random"]
METHODS = ["repredict"] + POOL + ["oracle_graph", "measured_greedy"]
COLORS = {"repredict": BLUE, "predict_once": "#6baed6", "static_graph": ORANGE, "graph_guided": "#fdae6b",
          "pan_style": RED, "dco_style": "#fb6a4a", "opentuner": PURPLE, "ga": "#9e9ac8", "random": GREY,
          "oracle_graph": GREEN, "measured_greedy": "#000000"}
THRESH_RATIO = 0.98


def per_program(search: pd.DataFrame, method: str, checkpoint: int) -> pd.Series:
    """Per-program objective (IR count) at a checkpoint; geometric mean over seeds."""
    s = search[(search["method"] == method) & (search["checkpoint"] == checkpoint)]
    return s.groupby("program_id")["best"].apply(lambda v: float(np.exp(np.log(v.astype(float)).mean())))


def one_sided(a: pd.Series, b: pd.Series, seed: int) -> dict:
    """a < b ? Wilcoxon signed-rank on log(a/b) over programs present in both."""
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    j = j[(j > 0).all(axis=1)]
    d = np.log(j["a"] / j["b"])
    w = stats.wilcoxon(d, alternative="less") if (d != 0).any() else None
    lo, hi = bootstrap_ci(d, np.mean, 2000, seed)
    return {"programs": int(len(d)), "geomean_ratio": float(np.exp(d.mean())),
            "ci95": [float(np.exp(lo)), float(np.exp(hi))], "wins": int((d < 0).sum()),
            "ties": int((d == 0).sum()), "losses": int((d > 0).sum()),
            "p_one_sided": float(w.pvalue) if w is not None else 1.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    v3 = load_v3_config()
    seed = v3["seed"]
    tag = "quick" if args.quick else ""
    data, out = DATA_DIR / version() / tag, OUTPUT_DIR / version() / tag
    tables, figs, rdir = out / "tables", out / "figures", out / "reports"
    claims = json.loads((rdir / "claims_summary.json").read_text())
    search = pd.read_csv(data / "ordering_search.csv.gz", keep_default_na=False)
    search["best"] = pd.to_numeric(search["best"], errors="coerce")
    oz = search[search["method"] == "Oz"].set_index("program_id")["best"]
    R = {"generated": now_iso(), "quick": args.quick, "objective": "IR instruction count / -Oz"}

    # ---- primary hypotheses
    h = {}
    h["H1"] = {**one_sided(per_program(search, "oracle_graph", 10), per_program(search, "static_graph", 10), seed),
               "claim": "Claim 1 headroom: oracle graph vs static graph, E = 10"}
    h["H2"] = {**claims.get("H2", {}), "claim": "Claim 2 drift: rho(S0, S2) < 0.5"}
    h["H3"] = {**claims.get("H3", {}), "claim": "Claim 3: LOPO model MAE < static-graph MAE"}
    ratios = {m: float(np.exp(np.log(per_program(search, m, 10) / oz).dropna().mean())) for m in POOL
              if len(per_program(search, m, 10))}
    strongest = min(ratios, key=ratios.get) if ratios else None
    h["H4"] = {**one_sided(per_program(search, "repredict", 10), per_program(search, strongest, 10), seed),
               "strongest_other": strongest, "pool_geomean_ratio_vs_Oz_at_E10": ratios,
               "claim": "Claim 4: re-prediction vs strongest other method, E = 10"}
    ps = [h[k].get("p_one_sided") if h[k].get("p_one_sided") is not None else 1.0 for k in ("H1", "H2", "H3", "H4")]
    for k, p in zip(("H1", "H2", "H3", "H4"), holm(ps)):
        h[k]["p_holm"] = float(p)
    h["H1"]["supported"] = h["H1"]["p_holm"] < 0.05 and h["H1"]["geomean_ratio"] <= THRESH_RATIO
    h["H2"]["supported"] = bool(h["H2"].get("p_holm", 1) < 0.05 and (h["H2"].get("median_rho") or 1) <= 0.5)
    h["H3"]["supported"] = bool(h["H3"].get("p_holm", 1) < 0.05 and (h["H3"].get("residual_r2") or 0) >= 0.10)
    h["H4"]["supported"] = h["H4"]["p_holm"] < 0.05 and h["H4"]["geomean_ratio"] <= THRESH_RATIO
    R["primary"] = h
    prim = pd.DataFrame([{"hypothesis": k, "claim": v["claim"], "programs": v.get("programs"),
                          "effect": v.get("geomean_ratio", v.get("median_rho", v.get("residual_r2"))),
                          "p_one_sided": v.get("p_one_sided"), "p_holm": v["p_holm"],
                          "supported": v["supported"]} for k, v in h.items()])
    save_table(prim, "v3_1_primary_hypotheses", tables, "Pre-registered hypotheses, Holm across the four")

    # ---- exploratory: budget curves (geomean ratio to -Oz, CI over programs; seed spread)
    rows = []
    for m in METHODS:
        for c in sorted(search[search["method"] == m]["checkpoint"].unique()):
            s = search[(search["method"] == m) & (search["checkpoint"] == c)]
            per_seed = []
            for sd, g in s.groupby("seed"):
                r = np.log(g.set_index("program_id")["best"] / oz).dropna()
                per_seed.append(float(np.exp(r.mean())))
            v = np.log(per_program(search, m, c) / oz).dropna()
            lo, hi = bootstrap_ci(v, np.mean, 1000, seed)
            rows.append({"method": m, "checkpoint": int(c), "programs": int(len(v)),
                         "geomean_ratio_vs_Oz": float(np.exp(v.mean())), "ci_lo": float(np.exp(lo)),
                         "ci_hi": float(np.exp(hi)), "seed_min": min(per_seed), "seed_max": max(per_seed),
                         "mean_E": float(s["E"].astype(float).mean()), "mean_opt_calls": float(s["O"].astype(float).mean()),
                         "mean_seconds": float(s["seconds"].astype(float).mean())})
    curves = pd.DataFrame(rows)
    save_table(curves, "v3_1_budget_curves", tables, "Best-so-far IR count / -Oz by compile budget (exploratory)")
    levels = {lv: float(np.exp(np.log(search[search["method"] == lv].set_index("program_id")["best"] / oz).dropna().mean()))
              for lv in ("O0", "O1", "O2", "O3", "Os", "Oz")}
    R["levels_vs_Oz"] = levels
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in METHODS:
        c = curves[curves["method"] == m]
        if m == "measured_greedy":
            ax.scatter(c["mean_E"], c["geomean_ratio_vs_Oz"], color=COLORS[m], marker="x", label=f"{m} (true E)")
            continue
        ax.plot(c["checkpoint"], c["geomean_ratio_vs_Oz"], marker="o", ms=3, color=COLORS[m], label=m)
        ax.fill_between(c["checkpoint"], c["ci_lo"], c["ci_hi"], color=COLORS[m], alpha=0.12)
    for lv in ("O2", "O3"):
        ax.axhline(levels[lv], ls="--", lw=0.8, color=GREY)
        ax.text(1.02, levels[lv], f"-{lv}", fontsize=7, va="bottom", color=GREY)
    ax.axhline(1.0, ls=":", lw=0.8, color=GREY)
    ax.set_xscale("log")
    ax.set_xlabel("compile evaluations (E)")
    ax.set_ylabel("geomean IR count / -Oz (lower is better)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    save_fig(fig, "v3_1_budget_curves", figs)

    # ---- exploratory: measured metrics of best sequences and -O levels
    mpath = data / "ordering_measure.csv.gz"
    if mpath.exists():
        meas = pd.read_csv(mpath, keep_default_na=False)
        j = search.merge(meas, on=["program_id", "best_seq"], how="left", suffixes=("", "_m"))
        for col in ("text_size", "text_size_aarch64", "dyn_instructions"):
            j[col] = pd.to_numeric(j[col], errors="coerce")
        base = j[j["method"] == "O2"].set_index("program_id")
        mrows = []
        for (m, c), g in j[j["checkpoint"].isin([0, 10, 25, 200])].groupby(["method", "checkpoint"]):
            g = g[g["output_matches"].astype(str) == "True"]
            r = {"method": m, "checkpoint": int(c), "programs": int(g["program_id"].nunique())}
            for col in ("text_size", "text_size_aarch64", "dyn_instructions"):
                gm = g.groupby("program_id")[col].apply(lambda v: float(np.exp(np.log(v[v > 0]).mean())) if (v > 0).any() else np.nan)
                rr = np.log(gm / base[col]).dropna()
                r[f"{col}_vs_O2"] = float(np.exp(rr.mean())) if len(rr) else np.nan
            mrows.append(r)
        mt = pd.DataFrame(mrows)
        save_table(mt, "v3_1_measured_vs_O2", tables, "Size, AArch64 size and executed instructions of best sequences, relative to -O2 (exploratory)")
        R["measured"] = mt.to_dict("records")
        R["output_failures"] = int((meas["output_matches"].astype(str) == "False").sum())

    rpath = data / "ordering_runtime.csv"
    if rpath.exists():
        rt = pd.read_csv(rpath, keep_default_na=False)
        rt = rt[rt["error"] == ""]
        rt["median_s"] = rt["times"].map(lambda t: float(np.median(json.loads(t))))
        rt["cv"] = rt["times"].map(lambda t: cv(json.loads(t)))
        R["runtime_noise"] = {"binaries": int(len(rt)), "median_cv": float(rt["cv"].median()),
                              "noisy": int((rt["cv"] > v3["ordering"]["runtime_cv_flag"]).sum())}

    # ---- Claims 1-3 exploratory tables from the claims summary
    for key, name, cap in (("variance_decomposition", "v3_1_variance_decomposition", "Variance decomposition, bootstrap 95% CI over programs (exploratory)"),
                           ("drift_decay", "v3_1_drift_decay", "Drift along the O2 trajectory (exploratory)"),
                           ("static_graph_error", "v3_1_static_graph_error", "Median static-graph error per program (exploratory)")):
        if key in claims:
            save_table(pd.DataFrame(claims[key]), name, tables, cap)
    R["claims_exploratory"] = {k: claims.get(k) for k in ("drift_decay", "static_graph_error", "H3_by_snapshot",
                                                         "single_pass_lopo", "representations")}

    write_json(out / "results_v3_1.json", R)
    (out / "report").mkdir(parents=True, exist_ok=True)
    (out / "report" / "index.html").write_text(render(R, prim, curves, tables, figs))
    (out / "RESULTS.md").write_text(results_md(R, prim))
    print(prim.to_string(index=False))
    return 0


def results_md(R, prim) -> str:
    lines = ["# OptGraph v3.1 results", "", f"Generated {R['generated']}. Objective: {R['objective']}.", "",
             "## Pre-registered hypotheses (Holm across the four)", "",
             "| hypothesis | claim | programs | effect | p (one-sided) | p (Holm) | supported |", "|---|---|---|---|---|---|---|"]
    for r in prim.itertuples():
        eff = "n/a" if r.effect is None or pd.isna(r.effect) else f"{r.effect:.4f}"
        p1 = "n/a" if r.p_one_sided is None or pd.isna(r.p_one_sided) else f"{r.p_one_sided:.3g}"
        lines.append(f"| {r.hypothesis} | {r.claim} | {r.programs} | {eff} | {p1} | {r.p_holm:.3g} | {r.supported} |")
    h4 = R["primary"]["H4"]
    lines += ["", f"H4 compared against the strongest other method at E = 10: **{h4.get('strongest_other')}**.",
              "", "Everything else in the report is exploratory.", ""]
    return "\n".join(lines)


def render(R, prim, curves, tables, figs) -> str:
    def img(name):
        p = figs / f"{name}.png"
        return f"<img alt='{name}' src='data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}'>" if p.exists() else ""

    def tab(name):
        p = tables / f"{name}.csv"
        return pd.read_csv(p).to_html(index=False, float_format=lambda v: f"{v:.4g}", border=0) if p.exists() else "<p>(missing)</p>"
    css = ("body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:20px 16px;background:#fbfbfa;color:#1d1d1f}"
           "table{border-collapse:collapse;font-size:12.5px}td,th{border-bottom:1px solid #ddd;padding:3px 7px;text-align:right}"
           "td:first-child,th:first-child{text-align:left}img{max-width:100%;border:1px solid #ddd}div.t{overflow-x:auto}")
    parts = [f"<title>OptGraph v3.1 results</title><style>{css}</style>", "<h1>OptGraph v3.1 results</h1>",
             f"<p>Generated {html.escape(R['generated'])}. Objective: {html.escape(R['objective'])}. "
             "Hypotheses, thresholds and analysis choices were fixed in PREREGISTRATION.md before the pilot.</p>",
             "<h2>Pre-registered hypotheses</h2>", f"<div class='t'>{tab('v3_1_primary_hypotheses')}</div>",
             f"<p>H4's comparison method (strongest other at E = 10): <b>{html.escape(str(R['primary']['H4'].get('strongest_other')))}</b>.</p>",
             "<h2>Exploratory: budget curves</h2>", img("v3_1_budget_curves"), f"<div class='t'>{tab('v3_1_budget_curves')}</div>",
             "<h2>Exploratory: measured size, AArch64 size, executed instructions</h2>", f"<div class='t'>{tab('v3_1_measured_vs_O2')}</div>",
             "<h2>Exploratory: Claims 1-3</h2>", f"<div class='t'>{tab('v3_1_variance_decomposition')}</div>",
             f"<div class='t'>{tab('v3_1_drift_decay')}</div>", f"<div class='t'>{tab('v3_1_static_graph_error')}</div>",
             f"<pre>{html.escape(json.dumps({k: R.get(k) for k in ('levels_vs_Oz', 'runtime_noise', 'output_failures')}, indent=1))}</pre>"]
    return "\n".join(parts)


if __name__ == "__main__":
    sys.exit(main())
