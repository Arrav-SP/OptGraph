#!/usr/bin/env python3
"""v3.1 Claims 1-3 on the snapshot data (metric: IR instruction count, no-op intrinsics excluded).

Primary (pre-registered, outputs/v3_1/PREREGISTRATION.md):
  H2 drift        per-program Spearman ρ(S0, S2) over pairs non-weak in S0 or S2 < 0.5
  H3 predictable  LOPO model MAE < static-graph MAE, per program (all snapshots pooled)
(H1 and H4 come from scripts/v3_1_ordering.py.) Everything else here is exploratory.

Writes data/v3_1[/quick]/claims/*.csv(.gz) and outputs/v3_1[/quick]/reports/claims_summary.json;
saves the LOPO single-pass models (used by re-prediction) under work_dir/v3_1/<tag>/models/.
"""
import os
os.environ.setdefault("OPTGRAPH_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", os.environ["OPTGRAPH_THREADS"])

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import _bootstrap  # noqa: F401,E402
from src import predictor as pr  # noqa: E402
from src.static_graphs import pair_mean, static_graph_error  # noqa: E402
from src.stats import bootstrap_ci, variance_decomposition  # noqa: E402
from src.utils import (DATA_DIR, OUTPUT_DIR, load_config, load_passes, load_v3_config, version,  # noqa: E402
                       work_dir, write_json)

METRIC = "instruction_count"


def load(data, metric=METRIC):
    # Read in chunks and keep one metric: the full table (all metrics) does not need to be in memory.
    return pd.concat(c[(c["metric"] == metric) & c["interaction_score"].notna()]
                     for c in pd.read_csv(data / "interaction_dataset.csv.gz", chunksize=500_000)).copy()


def drift(ds, thr, a="S0", b="S2"):
    """Per program: Spearman ρ between snapshots a and b over pairs non-weak in a or b."""
    w = ds.pivot_table(index=["program_id", "pass_a", "pass_b"], columns="context", values="interaction_score")
    rows = []
    for pid, g in w.groupby(level="program_id"):
        if a not in g or b not in g:
            continue
        g = g[[a, b]].dropna()
        g = g[(g[a].abs() > thr) | (g[b].abs() > thr)]
        rho = stats.spearmanr(g[a], g[b])[0] if len(g) >= 5 and g[a].std() > 0 and g[b].std() > 0 else np.nan
        rows.append({"program_id": pid, "pairs_used": len(g), "rho": rho})
    return pd.DataFrame(rows)


def decay(ds, thr, snaps):
    """Exploratory: median ρ(S0, Sk) per snapshot and the share of pairs with |I| <= thr."""
    rows = []
    for s in snaps:
        d = drift(ds, thr, snaps[0], s) if s != snaps[0] else None
        share0 = float((ds[ds["context"] == s]["interaction_score"].abs() <= thr).mean())
        rows.append({"snapshot": s, "median_rho_vs_S0": float(d["rho"].median()) if d is not None else 1.0,
                     "programs": int(d["rho"].notna().sum()) if d is not None else int(ds["program_id"].nunique()),
                     "share_pairs_near_zero": share0})
    return pd.DataFrame(rows)


def static_errors(ds, singles, thr, suites):
    """Exploratory: for each program and snapshot, error of the static graph built from the other
    programs (LOPO) and from other suites only (cross-suite)."""
    rows = []
    for ctx, g in ds.groupby("context"):
        s_ctx = singles[singles["context"] == ctx]
        for pid, own in g.groupby("program_id"):
            for kind, mask in (("lopo", g["program_id"] != pid),
                               ("cross_suite", g["program_id"].map(suites) != suites[pid])):
                if not mask.any():
                    continue
                _, inter = pair_mean(g[mask], s_ctx[s_ctx["program_id"] != pid])
                rows.append({"program_id": pid, "suite": suites[pid], "context": ctx, "graph": kind,
                             **static_graph_error(own, inter, thr)})
    return pd.DataFrame(rows)


def decomposition(ds, programs, label, n_boot, seed):
    """Exploratory: pair x program x context shares, with a bootstrap over programs."""
    d = ds[ds["program_id"].isin(programs)].assign(pair=lambda x: x["pass_a"] + ">" + x["pass_b"])
    full = d.groupby(["program_id"])["context"].nunique()
    d = d[d["program_id"].isin(full[full == d["context"].nunique()].index)]
    cnt = d.groupby(["program_id", "context"])["pair"].nunique()
    keep = cnt.groupby(level=0).min()
    d = d[d["program_id"].isin(keep[keep == d["pair"].nunique()].index)]
    progs = sorted(d["program_id"].unique())
    if len(progs) < 3:
        return pd.DataFrame()

    def shares(sub):
        vd = variance_decomposition(sub, "interaction_score", ["pair", "program_id", "context"])
        out = dict(zip(vd["term"], vd["share"]))
        out["program_terms"] = sum(v for k, v in out.items() if "program_id" in k)
        return out
    point = shares(d)
    rng = np.random.default_rng(seed)
    boots = []
    by = {p: g for p, g in d.groupby("program_id")}
    for _ in range(n_boot):
        pick = rng.choice(progs, len(progs), replace=True)
        sub = pd.concat([by[p].assign(program_id=f"{p}#{i}") for i, p in enumerate(pick)])
        boots.append(shares(sub))
    b = pd.DataFrame(boots)
    return pd.DataFrame([{"subset": label, "programs": len(progs), "term": t, "share": v,
                          "ci_lo": float(b[t].quantile(0.025)), "ci_hi": float(b[t].quantile(0.975))}
                         for t, v in point.items()])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    # 2 parallel LOPO fits: each worker holds ~1.7 GB; 6 exhausted WSL's 7.5 GB (OOM, 23 Oct).
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    cfg, v3 = load_config(), load_v3_config()
    seed, thr = v3["seed"], cfg["settings"]["interaction_threshold"]
    tag = "quick" if args.quick else ""
    data, out = DATA_DIR / version() / tag, OUTPUT_DIR / version() / tag
    cdir, rdir = data / "claims", out / "reports"
    mdir = work_dir(v3) / version(v3) / (tag or "full") / "models" / "single_lopo"
    for d in (cdir, rdir, mdir):
        d.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(DATA_DIR / version() / "benchmarks.csv", keep_default_na=False)
    suites = dict(zip(manifest["program_id"], manifest["suite"]))
    snaps = list(v3["snapshots"]) if "snapshots" in v3 else sorted(v3["contexts"])
    sig_inputs = [data / "interaction_dataset.csv.gz"] + sorted(data.glob("features_*.csv")) + \
        [Path(__file__), Path(pr.__file__)]
    sig = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in sig_inputs)
                         + json.dumps([args.n_boot, seed]).encode()).hexdigest()
    marker = cdir / "done.json"
    if marker.exists() and json.loads(marker.read_text()).get("signature") == sig \
            and (rdir / "claims_summary.json").exists() and any(mdir.glob("*.joblib")):
        print(f"claims up to date ({marker})")
        return 0
    ds = load(data)
    snaps = [s for s in snaps if s in set(ds["context"])]
    singles = ds.drop_duplicates(["program_id", "context", "pass_b"])[["program_id", "context", "pass_b", "delta_b_alone"]]
    present = set(ds["pass_a"]) | set(ds["pass_b"])
    passes = [p.name for p in load_passes(cfg, v3=True) if p.name in present]
    summary = {"metric": METRIC, "threshold": thr, "snapshots": snaps, "programs": int(ds["program_id"].nunique())}

    # ---- H2 drift (primary) + decay curve (exploratory)
    if {"S0", "S2"} <= set(snaps):
        d = drift(ds, thr)
        d.to_csv(cdir / "drift_S0_S2.csv", index=False)
        r = d["rho"].dropna()
        w = stats.wilcoxon(r - 0.5, alternative="less") if len(r) >= 2 and (r != 0.5).any() else None
        summary["H2"] = {"programs": int(len(r)), "median_rho": float(r.median()),
                         "ci_median": list(bootstrap_ci(r, np.median, 2000, seed)),
                         "p_one_sided": float(w.pvalue) if w is not None else None,
                         "test": "Wilcoxon signed-rank of rho - 0.5, alternative less"}
        dec = decay(ds, thr, snaps)
        dec.to_csv(cdir / "drift_decay.csv", index=False)
        summary["drift_decay"] = dec.to_dict("records")

    # ---- static-graph error (exploratory)
    se = static_errors(ds, singles, thr, suites)
    se.to_csv(cdir / "static_graph_error.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary["static_graph_error"] = (se.groupby(["graph", "context"])[["sign_flip_rate", "class_error_rate",
                                                                        "spearman", "top_k_overlap", "mae"]]
                                     .median().reset_index().to_dict("records"))

    # ---- variance decomposition with program bootstrap (exploratory)
    allp = sorted(ds["program_id"].unique())
    parts = [decomposition(ds, allp, "all", args.n_boot, seed),
             decomposition(ds, [p for p in allp if suites[p] != "optgraph"], "without_v2", args.n_boot, seed),
             decomposition(ds.assign(interaction_score=ds["interaction_score"].rank()), allp, "rank_based",
                           args.n_boot, seed)]
    parts += [decomposition(ds, [p for p in allp if suites[p] == s], f"suite_{s}", args.n_boot, seed)
              for s in sorted(set(suites[p] for p in allp))]
    vd = pd.concat([p for p in parts if len(p)])
    vd.to_csv(cdir / "variance_decomposition.csv", index=False)
    summary["variance_decomposition"] = vd.to_dict("records")

    # ---- H3 predictable (primary): LOPO, all available representations, snapshot features
    feats = None
    for rep in ("hand", "funcprops", "opcodes", "ir2vec"):
        f = data / f"features_{rep}.csv"
        if f.exists():
            t = pd.read_csv(f)
            feats = t if feats is None else feats.merge(t, on=["program_id", "context"], validate="one_to_one")
    feat_cols = [c for c in feats.columns if c not in ("program_id", "context")]
    summary["representations"] = [r for r in ("hand", "funcprops", "opcodes", "ir2vec")
                                   if (data / f"features_{r}.csv").exists()]
    pairs = pr.table(ds, feats, "pair")
    pred, infos = pr.lopo(pairs, passes, feat_cols, seed, n_jobs=args.jobs)
    pred.to_csv(cdir / "lopo_pairs.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pp = pred.assign(em=(pred["predicted"] - pred["interaction_score"]).abs(),
                     es=(pred["static"] - pred["interaction_score"]).abs()).groupby("program_id")[["em", "es"]].mean()
    diff = pp["em"] - pp["es"]
    w = stats.wilcoxon(diff, alternative="less") if (diff != 0).any() else None
    sse_m = ((pred["predicted"] - pred["interaction_score"]) ** 2).sum()
    sse_s = ((pred["static"] - pred["interaction_score"]) ** 2).sum()
    summary["H3"] = {"programs": int(len(pp)), "mae_model": float(pred.eval("abs(predicted - interaction_score)").mean()),
                     "mae_static": float(pred.eval("abs(static - interaction_score)").mean()),
                     "residual_r2": float(1 - sse_m / sse_s) if sse_s else None,
                     "gap_closed": pr.gap_closed(pred),
                     "programs_better": int((diff < 0).sum()), "programs_worse": int((diff > 0).sum()),
                     "ci_mean_mae_diff": list(bootstrap_ci(diff, np.mean, 2000, seed)),
                     "p_one_sided": float(w.pvalue) if w is not None else None,
                     "test": "Wilcoxon signed-rank of per-program MAE(model) - MAE(static), alternative less",
                     "params": [i["params"] for i in infos]}
    summary["H3_by_snapshot"] = (pred.assign(em=lambda x: (x.predicted - x.interaction_score).abs(),
                                             es=lambda x: (x.static - x.interaction_score).abs())
                                 .groupby("context")[["em", "es"]].mean().reset_index().to_dict("records"))

    # ---- single-pass LOPO models for re-prediction (Claim 4); predictions kept for analysis
    single = pr.table(ds, feats, "single")
    spred, _ = pr.lopo(single, passes, feat_cols, seed, model_dir=mdir, n_jobs=args.jobs)
    spred.to_csv(cdir / "lopo_singles.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary["single_pass_lopo"] = {"gap_closed": pr.gap_closed(spred), "models": str(mdir),
                                   "feature_columns": feat_cols}
    write_json(rdir / "claims_summary.json", summary)
    write_json(marker, {"signature": sig})
    print({k: summary[k] for k in ("H2", "H3") if k in summary})
    return 0


if __name__ == "__main__":
    sys.exit(main())
