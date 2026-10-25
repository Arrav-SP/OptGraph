#!/usr/bin/env python3
"""v3 Phase 4 (RQ2): predict I(A,B,P,context) for unseen programs and unseen suites.

Every configuration (metric x context x protocol x target x inputs) produces out-of-program
predictions, cached in the work directory; summaries, tables, tests and figures go to
outputs/v3/. Protocols: lopo (leave one program out), kfold (10 program-grouped folds),
loso (leave one suite out), transfer:<a>:<b>. Tuning is nested inside the training programs.
Also writes the Phase 5 inputs: LOPO predictions of pairwise interactions and of single-pass
effects in context C0.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OPTGRAPH_THREADS", "2"))

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _bootstrap  # noqa: F401,E402
from src import model as ml  # noqa: E402
from src.evaluation import evaluate_v3, per_program_scores  # noqa: E402
from src.reporting import BLUE, GREY, ORANGE, save_fig, save_table  # noqa: E402
from src.stats import holm, paired_test  # noqa: E402
from src.utils import (version, DATA_DIR, OUTPUT_DIR, load_config, load_passes, load_v3_config,  # noqa: E402
                       work_dir, write_json)

ALL = ml.V3_MODELS
SMALL = ("MeanBaseline", "PairMeanBaseline", "LinearRidge", "HistGradientBoosting")
MAIN = "HistGradientBoosting"
REPS = ("hand", "funcprops", "opcodes", "ir2vec")


def build_configs(contexts: list[str], metrics: list[str], quick: bool) -> list[dict]:
    """The experiment grid (fixed before any result was seen).

    Tuning: nested and program-grouped, inside the training programs of each outer fold of the
    grouped 10-fold CV and the suite protocols. LOPO (101 folds) uses each model's a-priori
    default (the first grid point): tuning there would cost ~11 h on this machine; a fixed
    default involves no test information, so it is not a leak.
    Suite protocols and ablations use the cheaper models; the representation ablation and the
    direct-vs-residual comparison run in C0 only.
    """
    C = []

    def add(group, metric, ctx, protocol, models, target="residual", tune=True, mode="both", reps=("hand",)):
        C.append(dict(group=group, metric=metric, context=ctx, protocol=protocol, models=tuple(models),
                      target=target, tune=tune, mode=mode, reps=tuple(reps)))
    primary = metrics[0]
    c0 = contexts[0]
    for m in metrics:               # Phase 5 inputs first: LOPO in C0 for every metric
        add("main", m, c0, "lopo", ALL if m == primary else SMALL, tune=False)
    for c in contexts:
        for m in metrics:
            if c != c0:
                add("main", m, c, "lopo", ALL if m == primary else SMALL, tune=False)
            add("main", m, c, "kfold", ALL if m == primary else SMALL)
        for proto in ("loso", "transfer:polybench:testsuite", "transfer:testsuite:polybench"):
            add("suite", primary, c, proto, SMALL)
        for mode in ("pass", "features", "both"):
            add("ablation_inputs", primary, c, "kfold", ("PairMeanBaseline", MAIN), target="direct", mode=mode)
    add("target", primary, c0, "kfold", ("PairMeanBaseline", "RandomForest", MAIN), target="direct")
    for m in metrics:
        for reps in [(r,) for r in REPS] + [REPS]:
            add("ablation_reps", m, c0, "kfold", ("PairMeanBaseline", MAIN), reps=reps)
    if quick:   # 3 programs: no k-fold beyond 3 groups and no suite transfer is meaningful
        C = [c for c in C if c["protocol"] in ("lopo", "loso") or c["group"] == "ablation_reps"][:8]
    return C


def cfg_name(c: dict) -> str:
    proto = c["protocol"].replace(":", "-")
    return f"{c['group']}__{c['metric']}__{c['context']}__{proto}__{c['target']}__{c['mode']}__{'+'.join(c['reps'])}"


def single_pass_table(ds: pd.DataFrame, metric: str, context: str) -> pd.DataFrame:
    """Δ(B|P) per (program, pass), shaped like pair rows (pass_a = pass_b = pass) so the
    same machinery predicts it; the 'pair mean' then is the per-pass mean."""
    d = ds[(ds["metric"] == metric) & (ds["context"] == context) & ds["delta_b_alone"].notna()]
    s = d.drop_duplicates(["program_id", "pass_b"])[["program_id", "pass_b", "delta_b_alone"]]
    return s.assign(pass_a=s["pass_b"], interaction_score=s["delta_b_alone"])[
        ["program_id", "pass_a", "pass_b", "interaction_score"]]


def run(c, rows, feats, pass_names, suites_of, seed, cache_dir, sig, n_jobs, log):
    path = cache_dir / f"{cfg_name(c)}.csv.gz"
    meta = cache_dir / f"{cfg_name(c)}.json"
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get("signature") == sig:
        return pd.read_csv(path), json.loads(meta.read_text())
    d = rows.merge(feats, on="program_id", how="inner", validate="many_to_one").reset_index(drop=True)
    cols = [x for x in feats.columns if x != "program_id"]
    X, layout = ml.v3_matrix(d, pass_names, cols, c["mode"])
    t = time.perf_counter()
    pred = ml.cross_predict_v3(d, X, layout, c["protocol"], list(c["models"]), seed, c["target"],
                               c["tune"], d["program_id"].map(suites_of).to_numpy(), 10, n_jobs)
    info = {"signature": sig, "config": c, "seconds": round(time.perf_counter() - t, 1),
            "rows": len(d), "programs": int(d["program_id"].nunique()), "params": pred.attrs["params"]}
    pred.to_csv(path, index=False)
    write_json(meta, info)
    log(f"    {cfg_name(c)}: {info['rows']} rows, {info['programs']} programs, {info['seconds']}s")
    return pred, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=6, help="outer folds in parallel")
    ap.add_argument("--only", nargs="+", metavar="GROUP", help="run only these config groups")
    args = ap.parse_args()
    cfg, v3 = load_config(), load_v3_config()
    seed, thr = v3["seed"], cfg["settings"]["interaction_threshold"]
    tag = "quick" if args.quick else ""
    data, out = DATA_DIR / version() / tag, OUTPUT_DIR / version() / tag
    tables, figs, rep_dir = out / "tables", out / "figures", out / "reports"
    cache = work_dir(v3) / version(v3) / (tag or "full") / "models"
    cache.mkdir(parents=True, exist_ok=True)
    ds_path = data / "interaction_dataset.csv.gz"
    # Content, not file bytes: rewriting the same dataset must not invalidate cached models.
    sig = hashlib.sha256(gzip.decompress(ds_path.read_bytes())).hexdigest()[:16]
    ds = pd.read_csv(ds_path)
    manifest = pd.read_csv(DATA_DIR / version() / "benchmarks.csv", keep_default_na=False)
    suites_of = dict(zip(manifest["program_id"], manifest["suite"]))
    present = set(ds["pass_a"]) | set(ds["pass_b"])
    pass_names = [p.name for p in load_passes(cfg, v3=True) if p.name in present]
    contexts = sorted(ds["context"].unique())
    metrics = [m for m in ("text_size", "dyn_instructions", "instruction_count") if m in set(ds["metric"])]
    reps = {r: pd.read_csv(data / f"features_{r}.csv") for r in REPS}
    configs = build_configs(contexts, metrics, args.quick)
    if args.only:
        configs = [c for c in configs if c["group"] in args.only]
    print(f"{len(configs)} configurations, {len(pass_names)} passes, contexts {contexts}, metrics {metrics}")

    results, per_prog = [], []
    for i, c in enumerate(configs, 1):
        rows = ds[(ds["metric"] == c["metric"]) & (ds["context"] == c["context"]) & ds["interaction_score"].notna()]
        rows = rows[["program_id", "pass_a", "pass_b", "interaction_score"]]
        feats = reps[c["reps"][0]]
        for r in c["reps"][1:]:
            feats = feats.merge(reps[r], on="program_id", validate="one_to_one")
        print(f"[{i}/{len(configs)}] {cfg_name(c)}", flush=True)
        pred, info = run(c, rows, feats, pass_names, suites_of, seed, cache, sig, args.jobs, print)
        learned = [m for m in c["models"] if m not in ml.BASELINES]
        tests = {m: paired_test(per_program_scores(pred, m)["mae"],
                                per_program_scores(pred, "PairMeanBaseline")["mae"]) for m in learned}
        adj = dict(zip(tests, holm([t["p_value"] for t in tests.values()]))) if tests else {}
        for m in c["models"]:
            ev = evaluate_v3(pred, m, thr)
            row = {**{k: c[k] for k in ("group", "metric", "context", "protocol", "target", "mode")},
                   "reps": "+".join(c["reps"]), "model": m, **ev}
            if m in tests:
                t = tests[m]
                row.update(mae_diff_vs_pairmean=t["mean_diff"], mae_diff_ci_lo=t["ci95_mean_diff"][0],
                           mae_diff_ci_hi=t["ci95_mean_diff"][1], wilcoxon_p=t["p_value"],
                           wilcoxon_p_holm=adj[m], programs_better=t["a_better"],
                           programs_worse=t["b_better"], programs_tied=t["ties"])
            results.append(row)
            if c["group"] == "main":
                pp = per_program_scores(pred, m)
                per_prog.append(pp.assign(metric=c["metric"], context=c["context"],
                                          protocol=c["protocol"], model=m))
    res = pd.DataFrame(results)
    rep_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(rep_dir / "rq2_all_results.csv", index=False)
    pp_all = pd.concat(per_prog) if per_prog else pd.DataFrame()
    pp_all.to_csv(rep_dir / "rq2_per_program.csv", index=False)

    # ---- Phase 5 inputs: LOPO predictions in C0 (pairs from the main configs, singles here)
    p5 = {}
    for m in ("text_size", "dyn_instructions"):
        if m not in metrics or "C0" not in contexts:
            continue
        main_cfg = next((c for c in configs if c["group"] == "main" and c["metric"] == m
                         and c["context"] == "C0" and c["protocol"] == "lopo"), None)
        if main_cfg is None:
            continue
        single = dict(main_cfg, group="phase5_single", models=("PairMeanBaseline", MAIN))
        rows = single_pass_table(ds, m, "C0")
        pred_s, _ = run(single, rows, reps["hand"], pass_names, suites_of, seed, cache, sig, args.jobs, print)
        p5[m] = {"pairs": str(cache / f"{cfg_name(main_cfg)}.csv.gz"),
                 "singles": str(cache / f"{cfg_name(single)}.csv.gz")}
        ev_s = {mm: evaluate_v3(pred_s, mm, thr) for mm in single["models"]}
        p5[m]["single_pass_eval"] = {mm: {k: ev_s[mm][k] for k in ("mae", "r2", "spearman")} for mm in ev_s}
    write_json(rep_dir / "phase5_inputs.json", p5)

    # ---- tables
    show = ["metric", "context", "protocol", "model", "mae", "rmse", "r2", "spearman", "macro_f1",
            "residual_r2", "per_program_spearman_median", "per_program_ndcg10_mean",
            "mae_diff_vs_pairmean", "mae_diff_ci_lo", "mae_diff_ci_hi", "wilcoxon_p_holm",
            "programs_better", "programs_worse"]
    for grp, name, cap in (("main", "rq2_models", "Prediction of I for unseen programs (residual target, hand features)"),
                           ("suite", "rq2_suites", "Generalisation to unseen suites"),
                           ("target", "rq2_targets", "Direct target (compare with residual in rq2_models)"),
                           ("ablation_inputs", "rq2_ablation_inputs", "Ablation: pass identities / program features / both (direct target)"),
                           ("ablation_reps", "rq2_ablation_reps", "Ablation: program representation (residual target)")):
        t = res[res["group"] == grp]
        if len(t):
            cols = show[:3] + (["mode"] if grp == "ablation_inputs" else ["reps"] if grp == "ablation_reps" else []) + show[3:]
            save_table(t[[c for c in cols if c in t.columns]], name, tables, cap)

    # ---- figures: MAE difference vs pair mean with CIs; per-program ranking quality
    main = res[(res["group"] == "main") & res["model"].isin(ALL[2:])]
    if len(main):
        keys = main[["metric", "protocol"]].drop_duplicates().values.tolist()
        fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False)
        for ax, (m, proto) in zip(axes[0], keys):
            t = main[(main.metric == m) & (main.protocol == proto)]
            labels = [f"{r.model.replace('GradientBoosting', 'GB')} {r.context}" for r in t.itertuples()]
            yv = np.arange(len(t))
            ax.errorbar(t["mae_diff_vs_pairmean"], yv, xerr=[t["mae_diff_vs_pairmean"] - t["mae_diff_ci_lo"],
                        t["mae_diff_ci_hi"] - t["mae_diff_vs_pairmean"]], fmt="o", color=BLUE, ms=4, capsize=2)
            ax.axvline(0, color=GREY, lw=1, ls="--")
            ax.set_yticks(yv, labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{m}, {proto}", fontsize=9)
            ax.set_xlabel("MAE(model) − MAE(pair mean), 95% CI\n(< 0: program conditioning helps)", fontsize=8)
        save_fig(fig, "rq2_model_comparison", figs)
    if len(pp_all):
        sub = pp_all[(pp_all.protocol == "lopo") & (pp_all.metric == metrics[0])]
        models = [m for m in ALL if m in set(sub.model) and m != "MeanBaseline"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4 + 0.25 * len(contexts)))
        for ax, col, lab in ((axes[0], "spearman", "per-program Spearman ρ"), (axes[1], "ndcg10", "per-program NDCG@10")):
            data_, labels = [], []
            for ctx in contexts:
                for m in models:
                    v = sub[(sub.model == m) & (sub.context == ctx)][col].dropna()
                    data_.append(v)
                    labels.append(f"{m.replace('GradientBoosting', 'GB').replace('Baseline', '')} {ctx}")
            ax.boxplot(data_, orientation="horizontal", widths=0.6, tick_labels=labels)
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel(lab)
        fig.suptitle(f"Ranking of the 462 ordered pairs within each held-out program ({metrics[0]}, LOPO)", fontsize=9)
        save_fig(fig, "rq2_ranking_quality", figs)

    summary_cols = ["group", "metric", "context", "protocol", "target", "mode", "reps", "model", "mae", "r2",
                    "residual_r2", "per_program_spearman_median", "per_program_ndcg10_mean",
                    "mae_diff_vs_pairmean", "wilcoxon_p_holm", "programs_better", "programs_worse"]
    write_json(rep_dir / "rq2_summary.json", {
        "main_model": MAIN, "metrics": metrics, "contexts": contexts, "dataset_sha256_16": sig,
        "results": res[[c for c in summary_cols if c in res.columns]].to_dict("records")})
    hl = res[(res.group == "main") & (res.protocol == "lopo")]
    print(hl[["metric", "context", "model", "mae", "r2", "residual_r2", "per_program_spearman_median",
              "mae_diff_vs_pairmean", "wilcoxon_p_holm"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
