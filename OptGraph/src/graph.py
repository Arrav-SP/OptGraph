"""OptGraph: a directed, weighted, program-conditioned pass-interaction graph.

Nodes are passes. The edge A -> B carries I(A, B, P): how much running A first
changes the benefit of B on program P. A -> B and B -> A are separate edges.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D

SYNERGY_COLOR = "#1b7837"
ANTAGONISM_COLOR = "#b2182b"


def _defined(ds: pd.DataFrame, metric: str) -> pd.DataFrame:
    d = ds[(ds["metric"] == metric) & ds["interaction_score"].notna()]
    return d


def program_graph(ds: pd.DataFrame, program_id: str, metric: str) -> nx.DiGraph:
    d = _defined(ds, metric)
    d = d[d["program_id"] == program_id]
    g = nx.DiGraph(program_id=program_id, metric=metric)
    g.add_nodes_from(sorted(set(d["pass_a"]) | set(d["pass_b"])))
    for r in d.itertuples():
        g.add_edge(r.pass_a, r.pass_b, weight=float(r.interaction_score),
                   interaction_type=r.interaction_type)
    return g


def aggregate_graph(ds: pd.DataFrame, metric: str) -> nx.DiGraph:
    """Edge weight = mean interaction over programs; std and counts kept as attributes."""
    d = _defined(ds, metric)
    g = nx.DiGraph(program_id="aggregate", metric=metric)
    g.add_nodes_from(sorted(set(d["pass_a"]) | set(d["pass_b"])))
    for (a, b), grp in d.groupby(["pass_a", "pass_b"]):
        s = grp["interaction_score"]
        g.add_edge(a, b, weight=float(s.mean()), std=float(s.std(ddof=0)), n_programs=int(len(s)),
                   n_synergy=int((grp["interaction_type"] == "SYNERGY").sum()),
                   n_antagonism=int((grp["interaction_type"] == "ANTAGONISM").sum()))
    return g


def graph_statistics(g: nx.DiGraph, threshold: float) -> pd.DataFrame:
    """Per-pass degree statistics.

    weighted_out: sum of I(pass -> X): the pass's net effect as an *enabler*.
    weighted_in : sum of I(X -> pass): how much the pass *benefits* from predecessors.
    Betweenness is computed on the unweighted graph of non-weak edges (|I| > threshold).
    """
    strong = nx.DiGraph()
    strong.add_nodes_from(g.nodes)
    strong.add_edges_from((u, v) for u, v, w in g.edges(data="weight") if abs(w) > threshold)
    btw = nx.betweenness_centrality(strong) if strong.number_of_edges() else {n: 0.0 for n in g}
    rows = []
    for n in g.nodes:
        out_w = [w for _, _, w in g.out_edges(n, data="weight")]
        in_w = [w for _, _, w in g.in_edges(n, data="weight")]
        rows.append({
            "pass": n,
            "weighted_out_degree": sum(out_w),
            "weighted_in_degree": sum(in_w),
            "positive_out_degree": sum(w > threshold for w in out_w),
            "negative_out_degree": sum(w < -threshold for w in out_w),
            "positive_in_degree": sum(w > threshold for w in in_w),
            "negative_in_degree": sum(w < -threshold for w in in_w),
            "betweenness": btw[n],
        })
    return pd.DataFrame(rows).sort_values("weighted_out_degree", ascending=False)


def draw_graph(g: nx.DiGraph, path: Path, title: str, threshold: float, max_edges: int = 40) -> int:
    """Draw non-weak edges (strongest `max_edges` at most). Returns edges drawn."""
    edges = [(u, v, w) for u, v, w in g.edges(data="weight") if abs(w) > threshold]
    edges.sort(key=lambda e: abs(e[2]), reverse=True)
    hidden = max(0, len(edges) - max_edges)
    edges = edges[:max_edges]
    wmax = max((abs(w) for _, _, w in edges), default=1.0)

    fig, ax = plt.subplots(figsize=(8.5, 8))
    pos = nx.circular_layout(sorted(g.nodes))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2600, node_color="#f0f0f0",
                           edgecolors="#404040", linewidths=1.2)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8.5)
    for u, v, w in reversed(edges):  # strongest drawn last, on top
        nx.draw_networkx_edges(
            g, pos, ax=ax, edgelist=[(u, v)], width=0.6 + 4.4 * abs(w) / wmax,
            edge_color=SYNERGY_COLOR if w > 0 else ANTAGONISM_COLOR, alpha=0.75,
            arrowsize=14, node_size=2600, connectionstyle="arc3,rad=0.14")
    note = f"edges with |I| > {threshold:g}"
    if hidden:
        note += f"; strongest {max_edges} shown, {hidden} weaker hidden"
    ax.set_title(f"{title}\n{note}; width ∝ |I|, max |I| = {wmax:.3f}", fontsize=10)
    ax.legend(handles=[
        Line2D([0], [0], color=SYNERGY_COLOR, lw=3, label="A → B synergy (I > 0)"),
        Line2D([0], [0], color=ANTAGONISM_COLOR, lw=3, label="A → B antagonism (I < 0)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=9)
    ax.set_axis_off()
    ax.margins(0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(edges)


def edge_table(g: nx.DiGraph) -> pd.DataFrame:
    rows = [{"pass_a": u, "pass_b": v, **d} for u, v, d in g.edges(data=True)]
    return pd.DataFrame(rows).sort_values("weight") if rows else pd.DataFrame()
