# OptGraph v3.1 pre-registration

Frozen: 2026-10-22 14:56 (IST), before the pilot and before any full v3.1 result existed.
This file is not edited after freezing; its sha256 is recorded in outputs/v3_1/PROGRESS.md.
Any deviation is reported as a deviation in PROGRESS.md and in the report.

## Data

- Corpus: the kept programs of `data/v3_1/benchmarks.csv` (sha256 c031f5b2a652095cf1baa96ab7fe593bb271a2a4134b28e5597a315610ba334f): 129 programs
  (63 testsuite, 30 polybench, 24 optgraph, 12 cbench). Starting IR: `clang -O1 -Xclang -disable-llvm-passes` (no optnone/noinline).
- 22 passes (`config/passes.yaml`, `passes` + `v3_passes`), all 462 ordered pairs.
- Snapshots: S0 = starting IR; S1, S2, S3, S4 = the first 10, 15, 22, 29 top-level stages of
  `opt -passes='default<O2>' -print-pipeline-passes` (LLVM 21.1.8). S4 equals `default<O2>`.
- Interaction: I(A,B,P) = Δ(B|A,P) − Δ(B|P), normalised by the snapshot baseline's metric.
  Weak threshold: |I| ≤ 0.005.
- Metric for Claims 1–4: IR instruction count, calls to `llvm.lifetime.*`, `llvm.dbg.*`,
  `llvm.experimental.noalias.scope.decl`, `llvm.pseudoprobe` excluded. Claim 4 objective: that count
  for the sequence applied to S0, as a ratio to the program's `-Oz` count (clang's `optsize minsize`
  attributes added before `default<Oz>`).

## Primary hypotheses (α = 0.05, Holm across exactly these four)

All tests are one-sided Wilcoxon signed-rank tests over programs. A program missing a value on
either side of a test is excluded from that test; the number used is reported.

**H1 (Claim 1, headroom).** With the same builder and budget, ordering from the oracle graph (the
program's own measured S0 graph) gives a lower objective than ordering from the static graph (mean
S0 interactions and single-pass effects of all other programs). Builder: additive estimate,
capped beam search (width 16, a pass at most twice and never twice in a row, length ≤ 16), all
visited sequences ranked by estimated gain and evaluated best first. Value per program:
best-so-far at E = 10. Test on log(oracle / static), alternative "less".
Supported if Holm p < 0.05 and geometric-mean ratio ≤ 0.98. The oracle needs ~485 compiles to
obtain and is reported as headroom, not as a method.

**H2 (Claim 2, drift).** Per program, Spearman ρ between the S0 and S2 interaction vectors over the
pairs with |I| > 0.005 in S0 or S2 (programs with at least 5 such pairs) is below 0.5. Test on
ρ − 0.5, alternative "less". Supported if Holm p < 0.05 and median ρ ≤ 0.5.

**H3 (Claim 3, predictable).** The pre-specified model predicts each held-out program's interactions
with lower MAE than the static graph. Model: HistGradientBoosting (max_iter 300, pass_a and pass_b
as native categorical features) on the residual from the context-free pair mean of the training
programs; features: every representation available (hand, LLVM func-properties, opcode histogram,
IR2Vec with the LLVM 21.1.8 seed vocabulary) of the snapshot's own baseline IR. Leave one program
out; hyperparameters (learning_rate ∈ {0.05, 0.1}, max_leaf_nodes ∈ {31, 63}) chosen on a seeded
20 % validation split of the training programs, then refitted on all training programs. Per
program: MAE over all its (snapshot, pair) rows. Test on MAE(model) − MAE(static), alternative
"less". Supported if Holm p < 0.05 and residual R² ≥ 0.10 (1 − SSE(model) / SSE(static), all rows).

**H4 (Claim 4, low-budget ordering).** Re-prediction's best-so-far objective at E = 10 is lower than
that of the strongest other method at E = 10. Re-prediction: capped beam search (width 16, depth
16) whose next-pass scores come from the LOPO single-pass model (same model family and tuning as
H3, target Δ(B|state)) applied to each state's own IR features; visited sequences ranked by summed
predicted Δ and evaluated best first. Pool: predict_once, static_graph, graph_guided, pan_style,
dco_style, opentuner, ga, random. Strongest = lowest geometric-mean ratio to -Oz over all programs
at E = 10. Stochastic methods (pan_style, dco_style, opentuner, ga, random) run 5 seeds; their
per-program value is the geometric mean over seeds. Test on log(repredict / strongest), alternative
"less". Supported if Holm p < 0.05 and geometric-mean ratio ≤ 0.98.

## Budget accounting

One evaluation (E) = a sequence not evaluated before by that method is turned into IR from S0 with
one `opt` invocation and its objective measured. Feature extraction and state construction for
re-prediction are counted as `opt` calls (O) but not as evaluations; curves report E, tables report
E, O and wall-clock seconds. Budget 200; checkpoints 1, 5, 10, 25, 50, 100, 200. Measured greedy
search is an upper-bound reference plotted at its true E.

## Exploratory (no claims; 95 % confidence intervals)

Static-graph error per program (sign flips, class errors, rank correlation, top-10 overlap; LOPO and
cross-suite), variance decomposition with a program bootstrap (all, without the 24 hand-written
programs, rank-based, per suite), the S0..S4 drift decay and share of near-zero pairs, H3 per
snapshot, every other budget, code size (x86-64 and AArch64, `.text`), executed instructions
(cachegrind) and runtime (median of 20 pinned runs after 1 warm-up, serial on an idle machine,
randomised order) of each method's best sequence at E = 10, 25, 200 and of -O1/O2/O3/Os/Oz.

## Wording

"A large share", not "most". DCO-style and Pan-style are reimplementations; their papers' numbers
are not compared. Partial support is reported as partial; unsupported hypotheses are reported.
