# OptGraph

Program-Conditioned Quantification and Prediction of LLVM Optimization Pass Interactions.

Status: the full pipeline has been executed. Every figure in "Results" comes
from `outputs/reports/` and `data/`, produced on Ubuntu 26.04 (WSL2),
Clang/LLVM 21.1.8, Python 3.14.4, seed 42.

## Research Question

To what extent do interactions between compiler optimization passes depend on
the structural characteristics of the input program, and can these
program-conditioned interactions be learned and accurately predicted for
previously unseen programs?

## Hypothesis

Programs with similar structural characteristics will exhibit more similar
optimization-pass interactions than structurally dissimilar programs.

## Motivation

The effect of an optimization pass depends on what ran before it: one pass can
expose opportunities for another, or consume them. This is the root of the
phase-ordering problem, and it is well studied. What is less often quantified
is how much a given pair's interaction changes from one program to another. If
the interaction of (A, B) were a fixed property of the pair, one global pass
order would suit every program. If it depends on the program in a way that is
visible in the unoptimized code, then the interaction graph could be predicted
per program before any optimization is run. We do not claim that pass
interaction, phase ordering, optimization graphs or ML-guided compilation are
new; we examine whether the magnitude and direction of pairwise interactions
vary systematically with program structure and whether that is predictable.

## Methodology

1. 24 deterministic C programs in five categories (arithmetic, loops, memory,
   control flow, mixed), each checked with UBSan and ASan.
2. `clang -O0 -Xclang -disable-O0-optnone -S -emit-llvm` gives the baseline IR P.
3. Structural features are extracted from P only.
4. For every pass A: P→A. For every ordered pair A ≠ B: P→A→B, obtained by
   applying B to the saved IR of A. Both orders are always run.
5. Eight IR metrics are measured on every IR.
6. Every optimized IR is linked and executed; its output must equal the baseline's.
7. Interaction scores are computed, joined with the features, and analysed as
   graphs, with ML models, and with a similarity analysis.

Each `opt` invocation is one experiment, logged with command, status, stdout,
stderr, error and timestamp (`data/results/experiment_log.csv`). A failed
experiment is recorded and the run continues.

## Program Features

30 features of the original IR (`data/program_features.csv`).

Counts: instructions, basic blocks, functions, branches, conditional branches,
loads, stores, calls, PHIs, integer operations, floating-point operations,
arithmetic instructions, memory instructions, returns, CFG edges, loop count,
maximum loop depth, allocas, GEPs, compares, casts.

Ratios: load, store, branch, call, arithmetic, memory and floating-point
fractions of all instructions; average block size; cyclomatic complexity.

Counts come from parsing the textual IR. Loop count and depth come from
LLVM's LoopInfo (`opt -passes='print<loops>'`), which reads but does not
change the IR; a CFG back-edge count is the fallback.

## Optimization Passes

mem2reg, sroa, instcombine, simplifycfg, early-cse, gvn, dce, adce,
reassociate, licm, loop-rotate, loop-unroll. Exact pipeline strings are in
`config/passes.yaml` and were verified against the installed `opt` (new pass
manager). `instcombine` is run as `instcombine<no-verify-fixpoint>`, `licm` as
`loop-mssa(licm)`, `loop-rotate` as `loop(loop-rotate)`.

## Interaction Definition

For a metric m, improvement is `(baseline − optimized) / baseline`.

```
Δ(B | P)    = ( m(P)    − m(B(P))    ) / m(P)
Δ(B | A, P) = ( m(A(P)) − m(B(A(P))) ) / m(P)
I(A, B, P)  = Δ(B | A, P) − Δ(B | P)
            = ( m(A(P)) + m(B(P)) − m(B(A(P))) − m(P) ) / m(P)
```

Normalising both terms by m(P) puts them on one scale, so I is the pair's
departure from additive behaviour, in fractions of the original program.
I > 0.005: synergy. I < −0.005: antagonism. Otherwise weak. A consequence:
`I(A,B,P) − I(B,A,P) = (m(A(B(P))) − m(B(A(P)))) / m(P)`, so the asymmetry
between the two directed edges equals the phase-ordering effect of the pair.

If m(P) = 0 the normalised score is undefined. The row stores the absolute
interaction, is labelled UNDEFINED, and is left out of modelling. This occurs
for PHI count only (2,640 rows).

## Graph Representation

One directed weighted graph per program: nodes are passes, edge A→B has weight
I(A, B, P). An aggregate graph uses the mean over programs and keeps the
standard deviation and the synergy/antagonism counts per edge. For each pass
we report weighted in- and out-degree, positive and negative degree, and
betweenness on the non-weak edges.

## ML Method

- Target: I(A, B, P) for instruction count.
- Inputs: the 30 original-IR features, one-hot pass A, one-hot pass B, one-hot ordered pair.
- Models: mean baseline; pair-mean baseline (mean training score of the same ordered pair, no program information); ridge regression; random forest; gradient boosting. No deep learning: 24 programs do not justify it.
- Splits are by program. Hold-out: 16 train / 4 validation / 4 test programs. The main model is the learned model with the lowest validation MAE (random forest).
- Leave-one-program-out cross-validation: each program is predicted by models trained on the other 23.
- Ablation: pass identities only / program features only / both.

## Evaluation

Regression: MAE, RMSE, R², Pearson, Spearman. Classification into
synergy / weak / antagonism at the 0.005 threshold: accuracy and
macro-averaged precision, recall and F1, plus the confusion matrix.

## Results

**Experiments.** 3,480 run, 3,480 succeeded, 0 failed; all optimized programs
reproduce the baseline output. (One earlier run had a single genuine `opt`
failure, logged and skipped; see Optimization Passes.)

**Descriptive.** Of 3,168 ordered-pair observations for instruction count, 61%
are weak, 24% antagonistic and 15% synergistic. The pass pair alone explains
56.7% of the variance in the score; 43.3% is variation between programs for
the same pair. 27 of 132 ordered pairs are synergistic in some programs and
antagonistic in others. Order changes the result in 34% of (program, pair)
cases.

**Prediction, leave-one-program-out (3,168 predictions).**

| Model | MAE | RMSE | R² | Pearson | Spearman | Acc. | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| Mean baseline | 0.0551 | 0.1183 | −0.001 | — | — | 0.243 | 0.081 | 0.333 | 0.130 |
| Pair-mean baseline | 0.0243 | 0.0812 | 0.529 | 0.728 | 0.768 | 0.882 | 0.840 | 0.869 | 0.850 |
| Ridge | 0.0367 | 0.0843 | 0.492 | 0.707 | 0.677 | 0.474 | 0.553 | 0.637 | 0.477 |
| Gradient boosting | 0.0229 | 0.0779 | 0.566 | 0.757 | 0.743 | 0.740 | 0.691 | 0.774 | 0.713 |
| Random forest | 0.0211 | 0.0779 | 0.567 | 0.757 | 0.779 | 0.905 | 0.867 | 0.888 | 0.876 |

**Prediction, hold-out test (4 programs, 528 predictions).**

| Model | MAE | RMSE | R² | Acc. | Macro-F1 |
|---|---|---|---|---|---|
| Mean baseline | 0.0517 | 0.0841 | −0.008 | 0.237 | 0.128 |
| Pair-mean baseline | 0.0239 | 0.0587 | 0.508 | 0.875 | 0.834 |
| Ridge | 0.0647 | 0.1066 | −0.619 | 0.307 | 0.296 |
| Gradient boosting | 0.0296 | 0.0965 | −0.327 | 0.794 | 0.760 |
| Random forest | 0.0250 | 0.0752 | 0.194 | 0.913 | 0.885 |

**Ablation (random forest, leave-one-program-out).** Pass identities only:
MAE 0.0243, R² 0.529. Program features only: MAE 0.0565, R² −0.004. Both:
MAE 0.0211, R² 0.567.

**Interpretation.** The identity of the pair is by far the most informative
input. Program features alone predict nothing, as expected, since they do not
say which pair is being asked about. Combined with pass identity they reduce
leave-one-program-out MAE by 13% and improve on the pair-mean baseline for 20
of 24 held-out programs. On the 4-program hold-out split, however, the
pair-mean baseline has the lower MAE and the higher R², while the random
forest has the better class-level scores. The evidence that program structure
adds predictive information is therefore positive but small and not
consistent across protocols.

**Hypothesis.** Over 276 program pairs, structural similarity and
interaction-pattern similarity correlate at Spearman ρ = 0.114 (Mantel
permutation test, one-sided p = 0.048, 5,000 permutations). The sign agrees
with the hypothesis; the effect is weak. We do not claim statistical support
at this sample size.

## Limitations

- Small, hand-written benchmark suite; test sets of 4 programs are noisy, and 3 of the 4 hold-out test programs are from one category.
- Metrics are static IR counts. A smaller count is called an improvement, which is wrong for passes that grow code on purpose: `loop-unroll` produces the most extreme scores (down to −2.01) and they mean "A let B unroll more".
- A negative score does not separate interference from redundancy. `mem2reg` and `sroa` score −0.42 on each other because they do the same job.
- Starting from `-O0`, the dominant effects involve memory promotion; interactions among mid-level passes on already-promoted code are under-represented.
- Loop passes include LLVM's automatic loop canonicalisation.
- No systematic hyperparameter search was done. After a first evaluation showed gradient boosting under-fitting the pair structure, the ordered-pair one-hot encoding was added and its tree depth was raised from 3 to 5; the reported numbers are from that second configuration, so they are not free of evaluation-driven design choices.
- The mean baseline's correlations are reported as "—": its prediction is constant within a fold, so the pooled value (Spearman −0.03) carries no meaning.
- Only the instruction-count interaction was modelled.

## Future Work

Larger standard benchmark suites; a post-`mem2reg` baseline; binary size and
runtime; per-function and dataflow features; using predicted graphs to pick
pass orders and comparing against `-O2`/`-O3`; three-pass and higher-order
interactions (A→B→C).
