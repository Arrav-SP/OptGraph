#!/usr/bin/env python3
"""Train and evaluate interaction-score predictors on unseen programs.

Two program-level protocols are reported:
  1. hold-out   : programs split 70/15/15 into train/val/test (seeded).
  2. LOPO-CV    : leave-one-program-out; every row is predicted by models that
                  never saw its program. More stable with few programs.
The main model is the learned model with the lowest validation MAE.
Ablation (pass identities / program features / both) uses the main model.
"""
import argparse
import sys

import joblib
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import evaluation as ev
from src import model as ml
from src.interaction import feature_columns
from src.utils import Workspace, load_config, load_passes, now_iso, write_json


def metrics_table(results: dict) -> pd.DataFrame:
    rows = {name: {**{k: m[k] for k in ("mae", "rmse", "r2", "pearson", "spearman")},
                   "accuracy": m["classification"]["accuracy"],
                   "macro_precision": m["classification"]["macro_precision"],
                   "macro_recall": m["classification"]["macro_recall"],
                   "macro_f1": m["classification"]["macro_f1"]}
            for name, m in results.items()}
    return pd.DataFrame(rows).T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--metric")
    args = ap.parse_args()
    cfg = load_config()
    settings = cfg["settings"]
    seed, thr = settings["seed"], settings["interaction_threshold"]
    metric = args.metric or settings["primary_metric"]
    ws = Workspace(args.tag)
    if not ws.dataset.exists():
        print(f"{ws.dataset} not found - run scripts/run_experiments.py first", file=sys.stderr)
        return 1

    df, dropped = ml.load_ml_table(pd.read_csv(ws.dataset), metric)
    programs = sorted(df["program_id"].unique())
    pass_names = [p.name for p in load_passes(cfg) if p.name in set(df["pass_a"]) | set(df["pass_b"])]
    if len(programs) < 5:
        print(f"only {len(programs)} programs - too few for a program-level split", file=sys.stderr)
        return 1
    plots, reports, models_dir = ws.outputs / "plots", ws.outputs / "reports", ws.outputs / "models"
    for d in (plots, reports, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"metric={metric}  rows={len(df)} (dropped undefined: {dropped})  "
          f"programs={len(programs)}  passes={len(pass_names)}  features={len(feature_columns(df))}")

    # ---- 1. hold-out split by program ------------------------------------
    split = ml.split_programs(programs, seed)
    assert not (set(split["train"]) & set(split["val"]) or set(split["train"]) & set(split["test"])
                or set(split["val"]) & set(split["test"])), "program leakage between splits"
    part = {k: df[df["program_id"].isin(v)] for k, v in split.items()}
    X = {k: ml.build_matrix(v, pass_names) for k, v in part.items()}
    y = {k: v["interaction_score"].to_numpy() for k, v in part.items()}
    print(f"hold-out programs  train={len(split['train'])}  val={len(split['val'])}  test={len(split['test'])}")
    print(f"  test programs: {split['test']}")

    holdout = {"val": {}, "test": {}}
    fitted, test_pred = {}, {}
    for name, model in ml.make_models(seed).items():
        model.fit(X["train"], y["train"])
        fitted[name] = model
        holdout["val"][name] = ev.evaluate(y["val"], model.predict(X["val"]), thr)
        test_pred[name] = np.asarray(model.predict(X["test"]), float)
        holdout["test"][name] = ev.evaluate(y["test"], test_pred[name], thr)
    val_table, test_table = metrics_table(holdout["val"]), metrics_table(holdout["test"])
    best_by_val = val_table["mae"].astype(float).idxmin()
    # The main model is chosen on the VALIDATION programs only, among the learned models.
    main = val_table.drop(index=list(ml.BASELINES))["mae"].astype(float).idxmin()
    print("\nHold-out TEST set (unseen programs):")
    print(test_table.astype(float).round(4).to_string())
    print(f"lowest validation MAE overall: {best_by_val}; main (learned) model: {main}")

    # ---- 2. leave-one-program-out -------------------------------------------
    print("\nRunning leave-one-program-out cross-validation ...")
    lopo_pred = ml.leave_one_program_out(df, pass_names, seed)
    lopo = {n: ev.evaluate(lopo_pred["interaction_score"], lopo_pred[n], thr)
            for n in ml.make_models(seed)}
    lopo_table = metrics_table(lopo)
    print("LOPO-CV (all rows, each predicted with its program held out):")
    print(lopo_table.astype(float).round(4).to_string())
    per_program = (lopo_pred.assign(abs_err=(lopo_pred[main] - lopo_pred["interaction_score"]).abs(),
                                    base_err=(lopo_pred["PairMeanBaseline"] - lopo_pred["interaction_score"]).abs())
                   .groupby("program_id")[["abs_err", "base_err"]].mean()
                   .rename(columns={"abs_err": f"mae_{main}", "base_err": "mae_PairMeanBaseline"}))
    wins = int((per_program.iloc[:, 0] < per_program.iloc[:, 1]).sum())
    print(f"{main} beats PairMeanBaseline on {wins}/{len(per_program)} held-out programs")

    # ---- 3. ablation with the main model --------------------------------------
    print("\nAblation (LOPO-CV, main model) ...")
    labels = {"pass": "A: pass identities only", "features": "B: program features only",
              "both": "C: passes + features"}
    ablation = {}
    for mode, label in labels.items():
        pr = lopo_pred if mode == "both" else ml.leave_one_program_out(
            df, pass_names, seed, [main], mode)
        ablation[label] = ev.evaluate(pr["interaction_score"], pr[main], thr)
    ablation_table = metrics_table(ablation)
    print(ablation_table.astype(float).round(4).to_string())

    # ---- plots -----------------------------------------------------------------
    yl, pl = lopo_pred["interaction_score"].to_numpy(), lopo_pred[main].to_numpy()
    ev.plot_predicted_vs_actual(
        [(f"{main} - hold-out test ({len(split['test'])} unseen programs)", y["test"], test_pred[main]),
         (f"{main} - leave-one-program-out ({len(programs)} programs)", yl, pl)],
        plots / "predicted_vs_actual.png")
    ev.plot_residuals(yl, pl, f"{main} residuals, leave-one-program-out, {metric}", plots / "residuals.png")
    ev.plot_score_distribution(df["interaction_score"], thr,
                               f"Interaction scores, {metric}, {len(programs)} programs x "
                               f"{len(pass_names) * (len(pass_names) - 1)} ordered pairs",
                               plots / "interaction_distribution.png")
    ev.plot_model_comparison(lopo_table, f"Model comparison, leave-one-program-out, {metric}",
                             plots / "model_comparison.png")
    ev.plot_model_comparison(test_table, f"Model comparison, hold-out test programs, {metric}",
                             plots / "model_comparison_holdout.png")
    importance = pd.Series(fitted[main].feature_importances_, index=X["train"].columns)
    ev.plot_feature_importance(importance, f"{main} feature importance (trained on train split)",
                               plots / "feature_importance.png")
    ev.plot_confusion(lopo[main]["classification"]["confusion_matrix"],
                      f"{main}, leave-one-program-out\nthreshold |I| > {thr:g}",
                      plots / "confusion_matrix.png")
    ev.plot_ablation(ablation_table, f"Ablation, {main}, leave-one-program-out, {metric}",
                     plots / "ablation.png")

    # ---- persist ---------------------------------------------------------------
    joblib.dump({"model": fitted[main], "model_name": main, "pass_names": pass_names,
                 "feature_columns": feature_columns(df), "columns": list(X["train"].columns),
                 "metric": metric, "split": split, "threshold": thr, "trained": now_iso()},
                models_dir / "interaction_model.joblib")
    lopo_pred.to_csv(reports / "lopo_predictions.csv", index=False)
    per_program.to_csv(reports / "lopo_per_program_mae.csv")
    importance.sort_values(ascending=False).to_csv(reports / "feature_importance.csv", header=["importance"])
    for name, table in (("holdout_val", val_table), ("holdout_test", test_table),
                        ("lopo", lopo_table), ("ablation_lopo", ablation_table)):
        table.to_csv(reports / f"ml_{name}.csv")
    write_json(reports / "ml_results.json", {
        "timestamp": now_iso(), "metric": metric, "seed": seed, "threshold": thr,
        "rows": len(df), "rows_dropped_undefined": dropped, "programs": len(programs),
        "passes": pass_names, "n_features": len(feature_columns(df)), "split": split,
        "best_by_validation_mae": best_by_val, "main_model": main,
        "holdout": holdout, "lopo": lopo, "ablation_lopo": ablation,
        "main_beats_pair_baseline_programs": [wins, len(per_program)],
        "class_counts": pd.Series(ev.to_class(df["interaction_score"], thr)).value_counts().to_dict(),
    })
    print(f"\nplots -> {plots}\nreports -> {reports}\nmodel -> {models_dir / 'interaction_model.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
