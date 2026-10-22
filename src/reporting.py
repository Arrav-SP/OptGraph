"""Paper-ready output: every table as CSV + Markdown + LaTeX (booktabs), every figure as PDF + PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})

BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2c6fbb", "#d9822b", "#1b7837", "#b2182b", "#7a7a7a", "#6a51a3"


def _fmt(v, digits: int) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "--"
        return f"{v:.{digits}f}"
    return str(v)


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"),
                 ("~", r"\~{}"), ("<", r"$<$"), (">", r"$>$")):
        s = s.replace(a, b)
    return s


def save_table(df: pd.DataFrame, name: str, out_dir: Path, caption: str = "", digits: int = 4,
               index: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:   # nothing to report (e.g. one context only): no file, and no stale one either
        for ext in ("csv", "md", "tex"):
            (out_dir / f"{name}.{ext}").unlink(missing_ok=True)
        return
    d = df.reset_index() if index else df
    d.to_csv(out_dir / f"{name}.csv", index=False)
    cols = [str(c) for c in d.columns]
    rows = [[_fmt(v, digits) for v in r] for r in d.itertuples(index=False)]
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (out_dir / f"{name}.md").write_text((f"{caption}\n\n" if caption else "") + "\n".join(md) + "\n")
    align = "".join("r" if pd.api.types.is_numeric_dtype(d[c]) else "l" for c in d.columns)
    tex = [r"\begin{table}[t]", r"\centering"]
    if caption:
        tex.append(r"\caption{" + _tex_escape(caption) + "}")
    tex += [r"\label{tab:" + name.replace("_", "-") + "}", r"\begin{tabular}{" + align + "}",
            r"\toprule", " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    tex += [" & ".join(_tex_escape(v) for v in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / f"{name}.tex").write_text("\n".join(tex) + "\n")


def save_fig(fig, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)
