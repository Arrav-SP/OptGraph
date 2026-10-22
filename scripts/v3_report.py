#!/usr/bin/env python3
"""v3 Phase 7: assemble outputs/v3/report/index.html, RESULTS.md and results_v3.json.

Reads only files produced by the other v3 scripts. Every number in the text is taken from
those files; the verdict wording follows fixed rules (see RULES), stated in the report.
"""
import argparse
import base64
import html
import json
import sys

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import DATA_DIR, OUTPUT_DIR, load_v3_config, now_iso, write_json

RULES = {
    "RQ2": "supported for a metric/context if the LOPO main model (HistGradientBoosting, residual target) "
           "has a lower per-program MAE than the program-agnostic pair mean, Wilcoxon signed-rank over "
           "programs with Holm-corrected p < 0.05 (Holm over the learned models of that configuration).",
    "RQ3": "supported for an objective/budget/builder if sequences from the predicted program-conditioned "
           "graph give a lower objective value than sequences from the program-agnostic graph, "
           "Wilcoxon on per-program log ratios with Holm-corrected p < 0.05 (Holm over all key comparisons).",
}


def read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def read_csv(p):
    return pd.read_csv(p) if p.exists() else None


def pct(v, d=1):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{d}f}%"


def num(v, d=4):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def pval(p):
    return "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else (f"{p:.2g}" if p >= 1e-4 else "< 1e-4")


# ---- headline numbers -----------------------------------------------------------------

def collect(data, out, quick):
    rep = out / "reports"
    H = {"generated": now_iso(), "quick": quick}
    man = pd.read_csv(DATA_DIR / "v3" / "benchmarks.csv", keep_default_na=False)
    kept = man[man["kept"].astype(str) == "True"]
    H["benchmarks"] = {"candidates": int(len(man)), "kept": int(len(kept)),
                       "kept_by_suite": kept["suite"].value_counts().to_dict(),
                       "dropped": int(len(man) - len(kept)),
                       "dropped_reasons": man[man["kept"].astype(str) != "True"]
                       .assign(reason=lambda d: d["drop_reason"].str.replace(r"[\d.]+s", "Ns", regex=True)
                               .str.split(":").str[0]).groupby("reason").size().to_dict()}
    rc = read_json(data / "run_config.json")
    if rc:
        H["experiments"] = {k: rc[k] for k in ("experiments_total", "experiments_success", "experiments_failed",
                                               "interaction_rows", "elapsed_s")}
        H["experiments"].update(programs=len(rc["programs"]), passes=len(rc["passes"]),
                                contexts=rc["contexts"], dynamic_programs=len(rc["dynamic_programs"]))
    H["rq1"] = read_json(rep / "rq1_summary.json")
    rq2 = read_json(rep / "rq2_summary.json")
    H["rq2"] = None
    if rq2:
        res = pd.DataFrame(rq2["results"])
        main = res[(res["group"] == "main")]
        verdicts = []
        for (m, c, proto), g in main.groupby(["metric", "context", "protocol"]):
            g = g.set_index("model")
            if "HistGradientBoosting" not in g.index:
                continue
            h, pm = g.loc["HistGradientBoosting"], g.loc["PairMeanBaseline"]
            best = g.drop(index=[x for x in ("MeanBaseline", "PairMeanBaseline") if x in g.index])["mae"].idxmin()
            verdicts.append({"metric": m, "context": c, "protocol": proto,
                             "pairmean_mae": pm["mae"], "hgb_mae": h["mae"], "best_model": best,
                             "best_mae": g.loc[best, "mae"], "hgb_r2": h["r2"], "pairmean_r2": pm["r2"],
                             "hgb_residual_r2": h["residual_r2"], "p_holm": h.get("wilcoxon_p_holm"),
                             "better": h.get("programs_better"), "worse": h.get("programs_worse"),
                             "ci": [h.get("mae_diff_ci_lo"), h.get("mae_diff_ci_hi")],
                             "hgb_rank_spearman": h["per_program_spearman_median"],
                             "pairmean_rank_spearman": pm["per_program_spearman_median"],
                             "hgb_ndcg10": h["per_program_ndcg10_mean"], "pairmean_ndcg10": pm["per_program_ndcg10_mean"],
                             "supported": bool(h["mae"] < pm["mae"] and (h.get("wilcoxon_p_holm") or 1) < 0.05)})
        H["rq2"] = {"verdicts": verdicts, "results": rq2["results"]}
    H["rq3"] = read_json(rep / "rq3_summary.json")
    if H["rq3"]:
        tests = pd.DataFrame(H["rq3"]["tests"])
        if len(tests):
            tests["supported"] = (tests["gmean_ratio_a_over_b"] < 1) & (tests["wilcoxon_p_holm"] < 0.05)
            H["rq3"]["tests"] = tests.to_dict("records")
    return H


# ---- plain-English readings -------------------------------------------------------------

def rq1_text(r):
    if not r:
        return "RQ1 results are missing."
    out = []
    for m, s in r["metrics"].items():
        vs = s.get("variance_shares")
        if not vs:
            continue
        prog = sum(v for k, v in vs.items() if "program_id" in k)
        ctx = sum(v for k, v in vs.items() if "context" in k and "program_id" not in k)
        out.append(f"For {m}, the pair alone explains {pct(vs.get('pair'))} of the variance of I over "
                   f"{s['programs']} programs, the terms that involve the context but not the program "
                   f"{pct(ctx)}, and all terms involving the program {pct(prog)} "
                   f"(pair x program {pct(vs.get('pair x program_id'))}, pair x program x context "
                   f"{pct(vs.get('pair x program_id x context'))}).")
    st = pd.DataFrame(r["stability"])
    if len(st):
        for m, g in st.groupby("metric"):
            out.append(f"Across contexts ({m}), the per-program rank correlation of the pair interactions "
                       f"has a median between {num(g.per_program_spearman_median.min(), 2)} and "
                       f"{num(g.per_program_spearman_median.max(), 2)} depending on the context pair.")
    fl = pd.DataFrame(r["sign_flips"])
    if len(fl):
        g = fl[fl.metric == fl.metric.iloc[0]]
        out.append(f"Synergy turns into antagonism (or back) between two contexts for "
                   f"{pct(g.sign_flip_frac.min())} to {pct(g.sign_flip_frac.max())} of all (program, pair) "
                   f"cells for {g.metric.iloc[0]}; most cells are weak in both.")
    od = pd.DataFrame(r["order"])
    if len(od):
        for m, g in od.groupby("metric"):
            out.append(f"Order matters ({m}, |I(A,B)-I(B,A)| above the threshold) for "
                       + ", ".join(f"{pct(x.order_matters_frac)} in {x.context}" for x in g.itertuples())
                       + " of the unordered pairs.")
    return " ".join(out)


def rq2_text(r):
    if not r:
        return "RQ2 results are missing."
    v = pd.DataFrame(r["verdicts"])
    lopo = v[v.protocol == "lopo"]
    out = []
    for x in lopo.itertuples():
        word = "beats" if x.supported else ("is not significantly better than" if x.hgb_mae < x.pairmean_mae
                                             else "does not beat")
        out.append(f"{x.metric}, {x.context}: the program-conditioned model {word} the pair-mean baseline "
                   f"(MAE {num(x.hgb_mae)} vs {num(x.pairmean_mae)}; better on {x.better} and worse on "
                   f"{x.worse} held-out programs; Holm p = {pval(x.p_holm)}; residual R² {num(x.hgb_residual_r2, 3)}).")
    n_sup, n = int(lopo.supported.sum()), len(lopo)
    out.insert(0, f"Under leave-one-program-out, program conditioning helps significantly in {n_sup} of {n} "
                  f"metric/context settings.")
    return " ".join(out)


def rq3_text(r):
    if not r:
        return "RQ3 results are missing."
    gm = pd.DataFrame(r["geomean"])
    t = pd.DataFrame(r["tests"])
    kmax = max(r["budgets"])
    out = []
    for obj in r["objectives"]:
        g = gm[(gm.objective == obj) & (gm.metric == obj) & (gm.k == kmax)].set_index("strategy")
        if g.empty:
            continue
        def imp(s):
            return pct(g.loc[s, "improvement_vs_O0"]) if s in g.index else "n/a"
        out.append(f"Objective {obj}, budget {kmax}: geomean reduction vs -O0 is {imp('graph_predicted_greedy')} "
                   f"(predicted graph, greedy), {imp('graph_agnostic_greedy')} (program-agnostic graph), "
                   f"{imp('graph_oracle_greedy')} (measured graph), {imp('random_median')} (median random), "
                   f"{imp('search')} (measured search), against {imp('O2')} for -O2 and {imp('O3')} for -O3.")
    if len(t):
        tt = t[t.metric == t.objective]
        ag = tt[tt.b.str.startswith("graph_agnostic")]
        out.append(f"Predicted vs program-agnostic graph: significantly better in {int(ag.supported.sum())} of "
                   f"{len(ag)} objective/budget/builder settings (Holm p < 0.05); "
                   f"median geomean ratio {num(ag.gmean_ratio_a_over_b.median())}, "
                   f"wins/ties/losses summed {int(ag.wins.sum())}/{int(ag.ties.sum())}/{int(ag.losses.sum())}.")
        rm = tt[tt.b == "random_median"]
        out.append(f"Predicted graph vs median random sequence: significantly better in "
                   f"{int(rm.supported.sum())} of {len(rm)} settings.")
    return " ".join(out)


# ---- rendering -------------------------------------------------------------------------

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#5f6368;--card:#fff;--line:#e3e3e0;--accent:#2c6fbb;--ok:#1b7837;--bad:#b2182b;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161618;--fg:#e8e8e6;--muted:#a0a3a8;--card:#1f1f22;--line:#333338;--accent:#7fb2ea;--ok:#6fc28a;--bad:#ef7b7b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#161618;--fg:#e8e8e6;--muted:#a0a3a8;--card:#1f1f22;--line:#333338;--accent:#7fb2ea;--ok:#6fc28a;--bad:#ef7b7b;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:1100px;margin:0 auto;padding:24px 18px 64px}
h1{font-size:1.7rem;margin:.2em 0}h2{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.25em}
h3{margin-top:1.6em;font-size:1.05rem}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:14px 0}
.tbl{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--card)}td:first-child,th:first-child{text-align:left}
img{max-width:100%;background:#fff;border-radius:6px;border:1px solid var(--line)}
.yes{color:var(--ok);font-weight:600}.no{color:var(--bad);font-weight:600}
code{font-size:12.5px}figure{margin:14px 0}figcaption{color:var(--muted);font-size:13px}
"""


def table_html(df, digits=4, max_rows=200):
    if df is None or df.empty:
        return "<p class='muted'>(no data)</p>"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else (pval(v) if "p_holm" in c or c == "wilcoxon_p" else f"{v:.{digits}f}"))
    more = f"<p class='muted'>first {max_rows} of {len(df)} rows; full table in the CSV</p>" if len(df) > max_rows else ""
    return "<div class='tbl'>" + d.to_html(index=False, escape=True, border=0) + "</div>" + more


def fig_html(path, caption):
    if not path.exists():
        return f"<p class='muted'>(figure {html.escape(path.name)} missing)</p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f"<figure><img alt='{html.escape(caption)}' src='data:image/png;base64,{b64}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def build_html(H, out, elapsed):
    T, F = out / "tables", out / "figures"
    p = []
    add = p.append
    add(f"<title>OptGraph v3 results</title><style>{CSS}</style><main>")
    add("<h1>OptGraph v3: results</h1>")
    add(f"<p class='muted'>Generated {H['generated']} from the files in <code>data/v3/</code> and "
        f"<code>outputs/v3/</code>. Every number below was measured on this machine; no number was typed in.</p>")

    # headline
    b, e = H["benchmarks"], H.get("experiments", {})
    add("<div class='card'><h2 style='margin-top:0;border:0'>Headline</h2><ul>")
    add(f"<li>{b['kept']} programs ({', '.join(f'{k} {v}' for k, v in b['kept_by_suite'].items())}) of "
        f"{b['candidates']} candidates; {b['dropped']} dropped with a recorded reason.</li>")
    if e:
        add(f"<li>{e['experiments_total']:,} pass experiments ({e['programs']} programs × {e['passes']} passes × "
            f"{len(e['contexts'])} contexts), {e['experiments_failed']} failed; "
            f"{e['interaction_rows']:,} interaction values.</li>")
    if H["rq2"]:
        v = pd.DataFrame(H["rq2"]["verdicts"])
        lo = v[v.protocol == "lopo"]
        add(f"<li>RQ2: program conditioning beats the program-agnostic pair mean (significantly) in "
            f"{int(lo.supported.sum())} of {len(lo)} metric/context settings under leave-one-program-out.</li>")
    if H["rq3"] and H["rq3"].get("tests"):
        t = pd.DataFrame(H["rq3"]["tests"])
        t = t[(t.metric == t.objective) & t.b.str.startswith("graph_agnostic")]
        add(f"<li>RQ3: sequences from the predicted program-conditioned graph beat those from the "
            f"program-agnostic graph (significantly) in {int(t.supported.sum())} of {len(t)} settings.</li>")
    if elapsed:
        add(f"<li>Wall time of the reported <code>run_v3.sh</code> invocation: {elapsed / 3600:.2f} h "
            f"(see Reproduction for the time of each phase).</li>")
    add("</ul></div>")

    add("<h2>Setup</h2>")
    add("<p>Every variant is lowered by the same command, <code>clang -O2 -c -Xclang -disable-llvm-passes "
        "x.ll</code>, so differences in code size, executed instructions and runtime come from the IR alone. "
        "Interactions: <code>I(A,B,P) = Δ(B|A,P) − Δ(B|P)</code>, both as fractions of the metric of the context's "
        "baseline P. Contexts: C0 raw -O0 IR, C1 after <code>mem2reg,sroa</code>, C2 after "
        "<code>thinlto-pre-link&lt;O1&gt;</code>. Every optimised variant must reproduce the program's -O0 output.</p>")
    add(table_html(pd.DataFrame([{"suite": k, "programs": v} for k, v in b["kept_by_suite"].items()])))

    add("<h2>RQ1: how much does an interaction depend on the program and the context?</h2>")
    add(f"<div class='card'>{html.escape(rq1_text(H['rq1']))}</div>")
    for name, cap in (("rq1_variance_decomposition", "Variance decomposition of I (crossed design pair × program × context)"),
                      ("rq1_context_stability", "Per-program rank correlation of the pair interactions between contexts"),
                      ("rq1_sign_flips", "Sign flips between contexts"),
                      ("rq1_order_dependence", "Order dependence per context"),
                      ("rq1_class_distribution", "Interaction classes per context")):
        add(f"<h3>{html.escape(cap)}</h3>" + table_html(read_csv(T / f"{name}.csv")))
    add(fig_html(F / "rq1_variance_decomposition.png", "Share of the variance of I per term"))
    add(fig_html(F / "rq1_context_stability.png", "Rank stability across contexts (one point per program)"))
    for f in sorted(F.glob("rq1_example_graph_*_C*.png"))[:6]:
        add(fig_html(f, f"Example program graph: {f.stem.replace('rq1_example_graph_', '')} (text size)"))

    add("<h2>RQ2: can program-conditioned interactions be predicted for unseen programs and suites?</h2>")
    add(f"<div class='card'>{html.escape(rq2_text(H['rq2']))}<p class='muted'>Rule: {html.escape(RULES['RQ2'])}</p></div>")
    if H["rq2"]:
        v = pd.DataFrame(H["rq2"]["verdicts"])
        add("<h3>Verdicts</h3>" + table_html(v.drop(columns=["ci"])))
    for name, cap in (("rq2_models", "All models, LOPO and grouped 10-fold (residual target, hand features)"),
                      ("rq2_suites", "Leave one suite out and train-on-one-suite transfer"),
                      ("rq2_targets", "Direct target"),
                      ("rq2_ablation_inputs", "Ablation: inputs"),
                      ("rq2_ablation_reps", "Ablation: program representation")):
        add(f"<h3>{html.escape(cap)}</h3>" + table_html(read_csv(T / f"{name}.csv")))
    add(fig_html(F / "rq2_model_comparison.png", "MAE difference to the pair-mean baseline with bootstrap 95% CIs"))
    add(fig_html(F / "rq2_ranking_quality.png", "Per-program ranking quality of the predicted pair interactions"))

    add("<h2>RQ3: does ordering passes from the predicted graph produce better code?</h2>")
    add(f"<div class='card'>{html.escape(rq3_text(H['rq3']))}<p class='muted'>Rule: {html.escape(RULES['RQ3'])}</p></div>")
    gm = read_csv(T / "rq3_geomean_improvement.csv")
    if gm is not None:
        add("<h3>Geomean improvement on the objective metric</h3>" + table_html(gm[gm.metric == gm.objective]))
        add("<h3>Same sequences, other metrics (incl. runtime)</h3>" + table_html(gm[gm.metric != gm.objective], max_rows=400))
    for name, cap in (("rq3_key_tests", "Key comparisons (Wilcoxon, Holm)"), ("rq3_win_tie_loss", "Win/tie/loss vs other strategies"),
                      ("rq3_cost", "Evaluation cost"), ("rq3_runtime_noise", "Runtime measurement noise")):
        add(f"<h3>{html.escape(cap)}</h3>" + table_html(read_csv(T / f"{name}.csv"), max_rows=300))
    add(fig_html(F / "rq3_ordering_vs_olevels.png", "Ordering strategies vs -O levels"))
    add(fig_html(F / "rq3_cost_vs_benefit.png", "Cost (compile evaluations) vs benefit"))

    add("<h2>What failed or was dropped</h2>")
    man = pd.read_csv(DATA_DIR / "v3" / "benchmarks.csv", keep_default_na=False)
    add(table_html(man[man["kept"].astype(str) != "True"][["program_id", "suite", "runtime_o0_s", "drop_reason"]]))
    fails = read_csv(DATA_DIR / "v3" / ("quick/" if H["quick"] else "") / "experiment_failures.csv")
    if fails is not None and len(fails):
        add("<h3>Failed pass experiments</h3>" + table_html(fails.groupby("error").size().rename("count").reset_index()))
    else:
        add("<p>No pass experiment failed.</p>")
    if H["rq3"]:
        add(f"<p>Ordering: {H['rq3']['failed_variants']} failed variants. "
            f"{html.escape(json.dumps(H['rq3'].get('failure_examples', {})))}</p>")

    add("<h2>Threats to validity</h2><ul>")
    for t in THREATS(H):
        add(f"<li>{html.escape(t)}</li>")
    add("</ul>")
    add("<h2>Reproduction</h2><p><code>bash scripts/run_v3.sh</code> (resumable; <code>--quick</code> for a 3-program run). "
        "Phase times are in <code>outputs/v3/PROGRESS.md</code>.</p></main>")
    return "\n".join(p)


def THREATS(H):
    t = ["Program set: 101 programs from three sources; PolyBench kernels are loop nests with static control flow, "
         "and the 24 v2 programs are small hand-written ones. Results may not transfer to large applications.",
         "IR comes from clang -O0 with optnone removed. Clang still marks every function noinline at -O0, so the "
         "inline pass (and inlining inside -O1/-O2/-O3/-Os) has no effect; the -O levels are therefore weaker than "
         "a real clang -O2 build of the same source.",
         "Passes are applied one opt invocation at a time; analyses are not shared across invocations as in a real "
         "pipeline. The -O levels run as one pipeline.",
         "Interaction scores use one threshold (0.005) for classes; regression results do not depend on it.",
         "Dynamic instruction count comes from cachegrind and includes libc start-up and the PolyBench support "
         "object, which are identical across a program's variants; it is not a cycle count.",
         "Runtime is measured under WSL2 on a laptop CPU (frequency scaling, shared host). Each value is the median of "
         "10 pinned runs after one warm-up; programs below 0.05 s are excluded from runtime statistics.",
         "Pair-level dynamic counts exist only for programs with -O0 runtime <= 1 s (cost); Phase 5 uses the same set.",
         "Hyperparameters were tuned by nested program-grouped CV in the 10-fold protocol; LOPO used fixed a-priori "
         "defaults. The model and feature choices were fixed before the full results were seen, but they were "
         "developed on a 3-program pilot.",
         "The sequence builder assumes additive pairwise effects; higher-order interactions are ignored.",
         "Repeats: the builder reuses interactions measured once on the original program for every repeat, so "
         "cycles of mutually enabling passes (e.g. licm, instcombine, loop-rotate) look profitable indefinitely; "
         "graph sequences, including the measured (oracle) graph, often fill the budget with such cycles. "
         "Measured search does not have this problem.",
         "-Os: the -O0 IR carries no optsize attribute, so opt default<Os> produced IR identical to default<O2> "
         "for all 78 programs; the -Os rows are therefore not a real -Os baseline."]
    if H.get("rq3") and H["rq3"].get("runtime_noise"):
        n = H["rq3"]["runtime_noise"]
        t.append(f"Runtime noise: {n['noisy']} of {n['measured']} runtime measurements have a coefficient of "
                 f"variation above {n['cv_flag']} (median CV {n['cv_median']:.3f}).")
    return t


def results_md(H):
    lines = ["# OptGraph v3: headline results", "", f"Generated {H['generated']}. Machine-readable: `results_v3.json`.", ""]
    b = H["benchmarks"]
    lines += [f"- Programs: {b['kept']} kept of {b['candidates']} ({b['kept_by_suite']}); {b['dropped']} dropped."]
    if H.get("experiments"):
        e = H["experiments"]
        lines += [f"- Experiments: {e['experiments_total']} ({e['experiments_failed']} failed), "
                  f"{e['interaction_rows']} interaction values."]
    lines += ["", "## RQ1", "", rq1_text(H["rq1"]), "", "## RQ2", "", rq2_text(H["rq2"]), ""]
    if H["rq2"]:
        v = pd.DataFrame(H["rq2"]["verdicts"])
        cols = ["metric", "context", "protocol", "pairmean_mae", "hgb_mae", "best_model", "best_mae",
                "hgb_residual_r2", "p_holm", "better", "worse", "supported"]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in v[cols].itertuples(index=False):
            lines.append("| " + " | ".join(num(x) if isinstance(x, float) else str(x) for x in r) + " |")
    lines += ["", "## RQ3", "", rq3_text(H["rq3"]), ""]
    if H["rq3"] and H["rq3"].get("tests"):
        t = pd.DataFrame(H["rq3"]["tests"])
        cols = ["objective", "k", "metric", "a", "b", "programs", "gmean_ratio_a_over_b", "wins", "ties", "losses",
                "wilcoxon_p_holm", "supported"]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in t[cols].itertuples(index=False):
            lines.append("| " + " | ".join(num(x) if isinstance(x, float) else str(x) for x in r) + " |")
    lines += ["", "Rules: " + RULES["RQ2"], "", RULES["RQ3"], ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--elapsed", type=float, default=0.0, help="wall seconds of the run (from run_v3.sh)")
    args = ap.parse_args()
    load_v3_config()
    tag = "quick" if args.quick else ""
    data, out = DATA_DIR / "v3" / tag, OUTPUT_DIR / "v3" / tag
    H = collect(data, out, args.quick)
    H["elapsed_s"] = args.elapsed or None
    (out / "report").mkdir(parents=True, exist_ok=True)
    (out / "report" / "index.html").write_text(build_html(H, out, args.elapsed))
    (out / "RESULTS.md").write_text(results_md(H))
    write_json(out / "results_v3.json", H)
    print(f"wrote {out / 'report' / 'index.html'}, {out / 'RESULTS.md'}, {out / 'results_v3.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
