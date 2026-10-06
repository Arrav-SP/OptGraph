# Review 2 — OptGraph

**OptGraph: Program-Conditioned Quantification and Prediction of LLVM Optimization Pass Interactions**

All numbers below were produced by the code in this repository on
Ubuntu 26.04 (WSL2), Clang/LLVM 21.1.8, Python 3.14.4. Sources are named next
to each table; nothing is estimated or illustrative.

## Project Objective

Measure how pairs of LLVM optimization passes interact on individual programs,
represent the result as a directed weighted graph per program, and test
whether the interaction can be predicted for unseen programs from the
structure of their unoptimized IR.

## Research Question

To what extent do interactions between compiler optimization passes depend on
the structural characteristics of the input program, and can these
program-conditioned interactions be learned and accurately predicted for
previously unseen programs?

**Hypothesis.** Programs with similar structural characteristics exhibit more
similar pass interactions than structurally dissimilar programs.

**Framing.** Pass interaction, phase ordering, optimization graphs and
ML-guided optimization are existing work. Our contribution is to examine
whether the magnitude and direction of pairwise interactions vary
systematically with program structure and whether that is predictable.

## Current Implementation

Everything listed is implemented and has been run end to end.

| Component | File | Status |
|---|---|---|
| Environment check, `environment.json` | `scripts/check_environment.py` | working |
| C → LLVM IR, pass execution, IR verification, link + run | `src/llvm_pipeline.py` | working |
| 24 benchmark programs, 5 categories (UBSan/ASan clean) | `benchmarks/` | working |
| IR metrics (8 counts) by text parsing | `src/metrics.py` | working |
| 30 structural features from the original IR | `src/features.py` | working |
| P, P→A, P→A→B, P→B→A runner with failure logging | `src/experiments.py` | working |
| Interaction score, zero-baseline handling, dataset | `src/interaction.py` | working |
| Per-program and aggregate graphs, degree statistics | `src/graph.py` | working |
| 2 baselines + 3 models, program-level splits, LOPO-CV, ablation | `src/model.py` | working |
| Metrics, 9 plots, similarity analysis | `src/evaluation.py` | working |
| End-to-end demo (~10 s) | `run_demo.py` | working |
| Streamlit dashboard | — | not built (optional) |

## Architecture

```
C program ──clang -O0──> LLVM IR (P) ──┬──> features of the ORIGINAL IR
                                       └──> opt:  P→A   P→B   P→A→B   P→B→A
                                                        │
                                                 IR measurements
                                                        │
                                         I(A,B,P) = Δ(B|A,P) − Δ(B|P)
                                                        │
                              ┌─────────────────────────┼──────────────────────┐
                        OptGraph (per program)   interaction dataset    similarity analysis
                                                        │
                                 ML model: features + (A, B) → predicted I
                                                        │
                                     actual vs predicted on unseen programs
```

## Experimental Method

For every program P and every ordered pair of distinct passes (A, B):

| Run | Meaning |
|---|---|
| P | baseline IR from `clang -O0` (with `optnone` disabled so passes can act) |
| P→A | A alone |
| P→B | B alone |
| P→A→B | B applied to the saved output of A |
| P→B→A | A applied to the saved output of B |

12 passes: mem2reg, sroa, instcombine, simplifycfg, early-cse, gvn, dce, adce,
reassociate, licm, loop-rotate, loop-unroll. That gives 132 ordered pairs and
1 + 12 + 132 = 145 `opt` runs per program.

## Interaction Formula

For a metric *m* (e.g. instruction count):

```
Δ(B | P)    = ( m(P)    − m(B(P))    ) / m(P)        benefit of B alone
Δ(B | A, P) = ( m(A(P)) − m(B(A(P))) ) / m(P)        benefit B adds after A

I(A, B, P)  = Δ(B | A, P) − Δ(B | P)
```

`I > +0.005` synergy, `I < −0.005` antagonism, otherwise weak. Both deltas are
fractions of the original m(P). If m(P) = 0 the score is undefined and the row
is labelled `UNDEFINED` (this happens only for PHI count, which is always 0 at `-O0`).

## Current Results

### Experiments (`data/results/run_config.json`)

| | |
|---|---|
| Programs × passes | 24 × 12 |
| Experiments run | 3,480 |
| Succeeded / failed | 3,480 / 0 |
| Output check | every optimized IR was linked and run; all match the baseline output |
| Interaction rows (8 metrics) | 25,344, of which 2,640 undefined (PHI, zero baseline) |
| Wall time | about 95 s |

The first full run had one real failure (`instcombine` after `licm` on
`arith_03_integer_mix` aborted on LLVM's fixpoint self-check). It was recorded
and the run continued. The pass is now invoked as
`instcombine<no-verify-fixpoint>`, as LLVM's own pipelines do.

### How passes interact — instruction count (`outputs/reports/results_summary.json`)

| | |
|---|---|
| Ordered-pair observations | 3,168 |
| Weak / antagonism / synergy | 1,938 (61%) / 769 (24%) / 461 (15%) |
| Variance explained by the pass pair alone | 56.7% |
| Variance left to differences between programs | 43.3% |
| Ordered pairs that are synergy in some programs and antagonism in others | 27 of 132 |
| Ordered pairs that are weak in every program | 38 of 132 |
| (Program, pair) cases where order changes the result | 542 of 1,584 (34%) |

Strongest mean edges: `licm → instcombine` +0.209 (synergy in all 24 programs);
`mem2reg ↔ sroa` −0.424 (they remove the same allocas, so the second has
nothing left); `gvn → loop-unroll` −0.389.

One measured example (`loop_04_invariant`, from `run_demo.py`):
instcombine alone removes 17.58% of instructions; after gvn it removes 30.77%,
so I(gvn, instcombine) = **+13.19% (synergy)**. In the other order
I(instcombine, gvn) = **−9.89% (antagonism)**. The final program has 34
instructions via gvn→instcombine and 55 via instcombine→gvn.

### Prediction on unseen programs (`outputs/reports/ml_lopo.csv`, `ml_holdout_test.csv`)

Leave-one-program-out, 24 programs, 3,168 predictions:

| Model | MAE | RMSE | R² | Spearman | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| Mean baseline | 0.0551 | 0.1183 | −0.001 | — | 0.243 | 0.130 |
| Pair-mean baseline (pass pair only) | 0.0243 | 0.0812 | 0.529 | 0.768 | 0.882 | 0.850 |
| Linear (ridge) | 0.0367 | 0.0843 | 0.492 | 0.677 | 0.474 | 0.477 |
| Gradient boosting | 0.0229 | 0.0779 | 0.566 | 0.743 | 0.740 | 0.713 |
| **Random forest** (selected on validation) | **0.0211** | **0.0779** | **0.567** | **0.779** | **0.905** | **0.876** |

Hold-out test (4 unseen programs, 528 predictions):

| Model | MAE | R² | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Pair-mean baseline | 0.0239 | 0.508 | 0.875 | 0.834 |
| Random forest | 0.0250 | 0.194 | 0.913 | 0.885 |

Ablation (random forest, leave-one-program-out):

| Inputs | MAE | R² |
|---|---|---|
| A: pass identities only | 0.0243 | 0.529 |
| B: program features only | 0.0565 | −0.004 |
| C: pass identities + program features | 0.0211 | 0.567 |

**Reading.** Most of what is predictable comes from knowing which pair is
involved. Adding program features lowers leave-one-program-out MAE by 13% and
beats the pair-mean baseline on 20 of 24 held-out programs, but on the single
4-program hold-out split the baseline has the lower MAE and higher R². The
benefit of program features is therefore small and not yet firmly established.

### Hypothesis check (`outputs/reports/similarity_analysis.json`)

Structural similarity vs interaction-pattern similarity over 276 program
pairs: Spearman ρ = 0.114, Mantel permutation p = 0.048 (one-sided). The
direction agrees with the hypothesis but the association is weak, and with 24
programs this is exploratory, not conclusive.

## Graph

- `outputs/graphs/program_<id>_graph.png` – one directed graph per program (24 files)
- `outputs/graphs/aggregate_interaction_graph.png` – mean over programs
- `outputs/reports/aggregate_graph_stats.csv` – weighted in/out degree, positive/negative degree, betweenness
- Green edge A→B: synergy; red: antagonism; width ∝ |I|; weak edges omitted.

## ML

Trained and evaluated. Inputs: 30 features of the original IR + one-hot pass A,
pass B and ordered pair. Target: instruction-count interaction. Model saved at
`outputs/models/interaction_model.joblib`. Plots in `outputs/plots/`
(predicted vs actual, residuals, score distribution, model comparison, feature
importance, confusion matrix, ablation, similarity).

## Current Limitations

- 24 small hand-written programs; the hold-out test set is 4 programs, 3 of them from the memory category.
- Only IR-level counts; no runtime or binary size.
- "Improvement" means a smaller count. `loop-unroll` deliberately grows code, so its large negative scores mean "more unrolling", not a worse program.
- Negative scores mix true interference with plain redundancy (two passes doing the same job, e.g. mem2reg and sroa).
- The baseline is `-O0` IR, so memory-promotion passes dominate the largest edges.
- `licm` and `loop-rotate` run through LLVM's loop adaptor, which also canonicalises loops; that effect is included in theirs.
- Features are whole-module counts and ratios; no dataflow or per-loop features.

## Next Steps

1. Scale to an established suite (PolyBench, cBench, MiBench) for 100+ programs.
2. Add a second baseline taken after `mem2reg`, to study interactions among mid-level passes.
3. Add binary size and, on larger programs, runtime.
4. Richer features (per-function, loop trip counts, def-use statistics).
5. Use predicted graphs to choose a pass order and compare with `-O2`.
6. Three-pass (A→B→C) interactions.
