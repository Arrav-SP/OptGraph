#!/usr/bin/env python3
"""Write paper/numbers.tex: every number the paper uses, as LaTeX macros, read from the existing
v3.1 result files. Nothing is re-run. Usage: python paper/make_numbers.py (not numbers.py: that name shadows the stdlib module)"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from v3_1_report import per_program  # noqa: E402

D, O = ROOT / "data" / "v3_1", ROOT / "outputs" / "v3_1"
T = O / "tables"
M: dict[str, str] = {}


def m(name: str, v, fmt: str = "{}"):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    M[name] = fmt.format(v) if not isinstance(v, str) else v


def ratio(prefix: str, r: dict):
    """geomean ratio, CI, wins/losses, p (and Holm p if present)."""
    m(prefix + "Ratio", r["geomean_ratio"], "{:.3f}")
    m(prefix + "Lo", r["ci95"][0], "{:.3f}")
    m(prefix + "Hi", r["ci95"][1], "{:.3f}")
    m(prefix + "Pct", (1 - r["geomean_ratio"]) * 100, "{:.1f}")
    m(prefix + "Worse", (r["geomean_ratio"] - 1) * 100, "{:.1f}")
    for k in ("wins", "losses", "ties", "programs"):
        m(prefix + k.capitalize(), int(r[k]))
    m(prefix + "P", pval(r["p_one_sided"]))
    if "p_holm" in r:
        m(prefix + "PHolm", pval(r["p_holm"]))


def pval(p: float) -> str:
    if p >= 0.01:
        return f"{p:.2f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}\\times 10^{{{int(exp)}}}"


def main() -> int:
    R = json.loads((O / "results_v3_1.json").read_text())
    C = json.loads((O / "reports" / "claims_summary.json").read_text())
    run = json.loads((D / "run_config.json").read_text())
    env = json.loads((D / "ordering_search_run.json").read_text())["environment"]

    # ---- corpus and setup
    b = pd.read_csv(D / "benchmarks.csv", keep_default_na=False)
    k = b[b["kept"].astype(str) == "True"]
    m("NPrograms", len(k))
    names = {"testsuite": "TestSuite", "polybench": "PolyBench", "cbench": "CBench", "optgraph": "Own"}
    for s, n in k["suite"].value_counts().items():
        m("N" + names.get(s, s.capitalize()), int(n))
    m("NCandidates", len(b))
    m("NRuntimeReliable", int((k["runtime_reliable"].astype(str) == "True").sum()))
    lp = pd.read_csv(D / "claims" / "lopo_pairs.csv.gz", usecols=["context", "pass_a", "pass_b"])
    m("NPasses", lp["pass_a"].nunique())
    m("PassList", ", ".join(f"\\texttt{{{x}}}" for x in sorted(lp["pass_a"].unique())))
    m("NPairs", len(lp[lp.context == "S0"][["pass_a", "pass_b"]].drop_duplicates()))
    m("NExperiments", f"{run['experiments_total']:,}".replace(",", "{,}"))
    m("NExpFailed", run["experiments_failed"])
    m("NInteractionRows", f"{run['interaction_rows']:,}".replace(",", "{,}"))
    m("Threshold", C["threshold"])
    m("LLVMVersion", env["llvm_version"])
    m("NSnapshots", len(C["snapshots"]))
    m("NReps", len(C["representations"]))
    pre = (O / "PREREGISTRATION.md").read_bytes()
    sha = hashlib.sha256(pre).hexdigest()
    m("PreregSha", sha)
    m("PreregShaShort", sha[:16])

    # ---- primary hypotheses
    P = R["primary"]
    ratio("HOne", P["H1"])
    h2, h3, h4 = P["H2"], P["H3"], P["H4"]
    m("HTwoRho", h2["median_rho"], "{:.2f}")
    m("HTwoLo", h2["ci_median"][0], "{:.2f}")
    m("HTwoHi", h2["ci_median"][1], "{:.2f}")
    m("HTwoPrograms", h2["programs"])
    m("HTwoP", pval(h2["p_one_sided"]))
    m("HTwoPHolm", pval(h2["p_holm"]))
    m("HThreeRsq", h3["residual_r2"], "{:.2f}")
    m("HThreeGap", h3["gap_closed"] * 100, "{:.0f}")
    m("HThreeMaeModel", h3["mae_model"], "{:.4f}")
    m("HThreeMaeStatic", h3["mae_static"], "{:.4f}")
    m("HThreeBetter", h3["programs_better"])
    m("HThreeWorse", h3["programs_worse"])
    m("HThreePrograms", h3["programs"])
    m("HThreeP", pval(h3["p_one_sided"]))
    m("HThreePHolm", pval(h3["p_holm"]))
    ratio("HFour", h4)
    m("HFourOther", h4["strongest_other"].replace("_", "\\_"))

    # ---- exploratory: variance decomposition, static-graph error
    vd = pd.DataFrame(C["variance_decomposition"])
    a = vd[vd.subset == "all"].set_index("term")
    m("VdPair", a.loc["pair", "share"] * 100, "{:.0f}")
    m("VdProg", a.loc["program_terms", "share"] * 100, "{:.0f}")
    m("VdProgLo", a.loc["program_terms", "ci_lo"] * 100, "{:.0f}")
    m("VdProgHi", a.loc["program_terms", "ci_hi"] * 100, "{:.0f}")
    m("VdPairCtx", a.loc["pair x context", "share"] * 100, "{:.0f}")
    se = pd.DataFrame(C["static_graph_error"]).set_index(["graph", "context"])
    m("SgeRhoSZero", se.loc[("lopo", "S0"), "spearman"], "{:.2f}")
    m("SgeTopkSZero", se.loc[("lopo", "S0"), "top_k_overlap"] * 100, "{:.0f}")
    m("SgeClassErrSZero", se.loc[("lopo", "S0"), "class_error_rate"] * 100, "{:.0f}")

    # ---- drift
    dm = pd.read_csv(T / "v3_1_analysis_C_drift_matrix.csv").set_index("snapshot")
    words = dict(zip(["S0", "S1", "S2", "S3", "S4"], ["Zero", "One", "Two", "Three", "Four"]))
    for i, x in enumerate(words):
        for y in list(words)[i + 1:]:
            m(f"Rho{words[x]}{words[y]}", dm.loc[x, y], "{:.2f}")
    dec = pd.read_csv(D / "claims" / "drift_decay.csv").set_index("snapshot")
    for s, w in words.items():
        m(f"Sparse{w}", dec.loc[s, "share_pairs_near_zero"] * 100, "{:.0f}")

    # ---- ordering: curves, plateau, comparisons
    cur = pd.read_csv(T / "v3_1_budget_curves.csv").set_index(["method", "checkpoint"])["geomean_ratio_vs_Oz"]
    for meth, nm in (("ga", "GA"), ("repredict", "Repredict"), ("predict_once", "PredOnce"),
                     ("oracle_graph", "Oracle"), ("static_graph", "Static"), ("pan_style", "Pan")):
        for c, w in ((1, "One"), (10, "Ten"), (200, "TwoHundred")):
            m(f"Curve{nm}{w}", cur[(meth, c)], "{:.3f}")
    for lv, v in R["levels_vs_Oz"].items():
        m("Level" + {"O0": "OZero", "O1": "OOne", "O2": "OTwo", "O3": "OThree", "Os": "Os", "Oz": "Oz"}[lv], v, "{:.3f}")
    s = pd.read_csv(D / "ordering_search.csv.gz", keep_default_na=False)
    s["best"] = pd.to_numeric(s["best"], errors="coerce")
    for meth, nm in (("repredict", "Repredict"), ("predict_once", "PredOnce"), ("oracle_graph", "Oracle"),
                     ("static_graph", "Static"), ("ga", "GA"), ("pan_style", "Pan"), ("opentuner", "OpenTuner")):
        g = ((per_program(s, meth, 1) - per_program(s, meth, 200)) / per_program(s, meth, 1)).median()
        m(f"Plateau{nm}", g * 100, "{:.1f}")
    A = pd.read_csv(T / "v3_1_analysis_A_comparisons.csv")
    for (x, y, e), nm in ((("oracle_graph", "ga", 10), "OracleGA"), (("repredict", "static_graph", 10), "RepStatic"),
                          (("repredict", "predict_once", 10), "RepOnce"), (("predict_once", "pan_style", 1), "OnceEOne"),
                          (("repredict", "pan_style", 1), "RepEOne"), (("oracle_graph", "pan_style", 1), "OracleEOne"),
                          (("predict_once", "pan_style", 5), "OnceEFive")):
        r = A[(A.a == x) & (A.b == y) & (A.E == e)].iloc[0].to_dict()
        r["ci95"] = json.loads(r["ci95"]) if isinstance(r["ci95"], str) else r["ci95"]
        ratio("X" + nm, r)
    B = pd.read_csv(T / "v3_1_analysis_B_metrics.csv")
    for (x, y, met), nm in ((("oracle_graph@10", "static_graph@10", "text_size"), "SizeX"),
                            (("oracle_graph@10", "static_graph@10", "text_size_aarch64"), "SizeA"),
                            (("oracle_graph@200", "static_graph@200", "text_size_aarch64"), "SizeATwoHundred"),
                            (("oracle_graph@10", "static_graph@10", "dyn_instructions"), "Dyn"),
                            (("oracle_graph@10", "static_graph@10", "runtime"), "Runtime"),
                            (("ga@200", "O3@0", "text_size"), "GAOThreeSize"),
                            (("ga@200", "O3@0", "dyn_instructions"), "GAOThreeDyn"),
                            (("ga@200", "O3@0", "runtime"), "GAOThreeRuntime"),
                            (("ga@200", "Oz@0", "instruction_count"), "GAOzIR"),
                            (("ga@200", "Oz@0", "text_size"), "GAOzSize")):
        r = B[(B.a == x) & (B.b == y) & (B.metric == met)].iloc[0].to_dict()
        r["ci95"] = json.loads(r["ci95"]) if isinstance(r["ci95"], str) else r["ci95"]
        ratio("M" + nm, r)

    # ---- single-pass ranking accuracy of the LOPO model vs static graph (exploratory)
    sp = pd.read_csv(D / "claims" / "lopo_singles.csv.gz")
    acc = []
    for (pid, ctx), g in sp.groupby(["program_id", "context"]):
        if g["interaction_score"].std() > 0:
            acc.append((ctx, stats.spearmanr(g["predicted"], g["interaction_score"])[0],
                        stats.spearmanr(g["static"], g["interaction_score"])[0]))
    acc = pd.DataFrame(acc, columns=["ctx", "model", "static"]).groupby("ctx").median()
    m("RankModelSZero", acc.loc["S0", "model"], "{:.2f}")
    m("RankStaticSZero", acc.loc["S0", "static"], "{:.2f}")
    m("RankModelSTwo", acc.loc["S2", "model"], "{:.2f}")
    m("RankStaticSTwo", acc.loc["S2", "static"], "{:.2f}")

    # ---- measurement quality
    rn = R["runtime_noise"]
    m("NRuntimeBinaries", f"{rn['binaries']:,}".replace(",", "{,}"))
    m("RuntimeCV", rn["median_cv"] * 100, "{:.1f}")
    m("RuntimeNoisy", rn["noisy"])
    m("RuntimeNoisyPct", rn["noisy"] / rn["binaries"] * 100, "{:.0f}")
    m("OutputFailures", R["output_failures"])
    meas = pd.read_csv(D / "ordering_measure.csv.gz", usecols=["program_id"])
    m("NMeasured", f"{len(meas):,}".replace(",", "{,}"))

    out = ["% Generated by paper/make_numbers.py from outputs/v3_1 and data/v3_1. Do not edit."]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(out) + "\n")
    print(f"{len(M)} macros -> paper/numbers.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
