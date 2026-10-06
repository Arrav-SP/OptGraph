#!/usr/bin/env python3
"""Build and draw the per-program and aggregate interaction graphs."""
import argparse
import sys

import pandas as pd

import _bootstrap  # noqa: F401
from src.graph import aggregate_graph, draw_graph, edge_table, graph_statistics, program_graph
from src.utils import Workspace, load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--metric")
    ap.add_argument("--programs", nargs="+", metavar="ID", help="default: all programs")
    args = ap.parse_args()
    settings = load_config()["settings"]
    metric = args.metric or settings["primary_metric"]
    thr = settings["interaction_threshold"]
    ws = Workspace(args.tag)
    if not ws.dataset.exists():
        print(f"{ws.dataset} not found - run scripts/run_experiments.py first", file=sys.stderr)
        return 1
    ds = pd.read_csv(ws.dataset)
    gdir = ws.outputs / "graphs"
    rdir = ws.outputs / "reports"
    rdir.mkdir(parents=True, exist_ok=True)

    stats = []
    for pid in args.programs or sorted(ds["program_id"].unique()):
        g = program_graph(ds, pid, metric)
        if g.number_of_edges() == 0:
            print(f"  {pid}: no defined interactions, skipped")
            continue
        n = draw_graph(g, gdir / f"program_{pid}_graph.png", f"OptGraph - {pid} ({metric})", thr)
        s = graph_statistics(g, thr)
        s.insert(0, "program_id", pid)
        stats.append(s)
        print(f"  {pid:<26} {g.number_of_edges()} edges, {n} drawn")
    if stats:
        pd.concat(stats).to_csv(rdir / "graph_stats_per_program.csv", index=False)

    agg = aggregate_graph(ds, metric)
    draw_graph(agg, gdir / "aggregate_interaction_graph.png",
               f"OptGraph - mean interaction over {ds['program_id'].nunique()} programs ({metric})", thr)
    edge_table(agg).to_csv(rdir / "aggregate_graph_edges.csv", index=False)
    astats = graph_statistics(agg, thr)
    astats.to_csv(rdir / "aggregate_graph_stats.csv", index=False)
    print("\nAggregate graph, per-pass statistics:")
    print(astats.round(4).to_string(index=False))
    print(f"\ngraphs in {gdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
