# OptGraph v3 progress log

## Phase 0: audit and plan (2026-10-20)

Done
- Toolchain installed by the user (apt): clang/opt/llc/llvm-size 21.1.8, valgrind 3.26.0,
  libclang-rt-21-dev (ASan/UBSan runtimes). Venv recreated at `~/.venvs/optgraph` (the v2 one was
  under `/root`) from the pinned `requirements.txt`; pytest 9.1.1 added and pinned.
- v2 `environment.json` copied to `data/v3/environment_v2_original.json` before
  `check_environment.py` overwrote it (that script always rewrites it).
- llvm-test-suite: shallow sparse clone (`SingleSource/` only) at commit `d1b7335`
  in `~/optgraph_work/llvm-test-suite`.
- PolyBench/C 4.2.1 official tarball in `~/optgraph_work/polybench-c-4.2.1-beta`
  (sha256 `426519ee8443a5f2175de6a3e9328cda8917a5e33053d0e8f59855ab56d689a4`). Reason for not using
  the test-suite copy: its `polybench.h` defines the utility functions itself (linking the official
  `polybench.c` fails with 14 redefinition errors) and its kernels run twice (StrictFP check).
- `outputs/v3/PLAN.md` written.

Run
- `scripts/check_environment.py`: OK, 12/12 v2 passes accepted.
- `scripts/run_experiments.py --quick --verify-outputs`: 156/156 experiments succeeded, 9.8 s.
- `run_demo.py`: completed in 11.4 s (gvn → instcombine I = +11.21 %, instcombine → gvn I = −8.41 %).
- Throughput probes (gemm MEDIUM): opt 20 to 35 ms per call, backend 28 ms, cachegrind ≈ 25x native.
- All 16 candidate v3 passes, `mem2reg,sroa`, `thinlto-pre-link<O1>` and `default<O1|O2|O3|Os>`
  accepted by opt 21.1.8.
- `print<func-properties>` works, including `-enable-detailed-function-properties`.
  In-tree IR2Vec (`print<ir2vec>`) needs a vocabulary file that Ubuntu does not ship; checked in Phase 4.

Failed / notes
- None blocking.

## Phase 1: benchmarks (2026-10-20)

Done
- `config/v3.yaml` (work dir, suites, per-directory test-suite flags and FP tolerances copied from
  their CMakeLists.txt, contexts). `src/suites.py` (candidates, screening, manifest loader),
  `scripts/v3_benchmarks.py`. `Benchmark` gained `suite`, `cflags`, `support` (defaults keep v2
  unchanged); `llvm_pipeline.py` gained per-program flags, support objects, the fixed backend
  command, sanitizer, size and cachegrind commands. `src/measure.py`: build, run, measure.
- PolyBench: the 30 kernels of the official `utilities/benchmark_list`; `polybench.c` compiled once
  (`clang -O2 -c`), linked unchanged; only the kernel module is optimised. Dataset = smallest of
  MINI…EXTRALARGE whose `-O0` runtime (min of 2 runs) is ≥ 0.05 s.
- Screening per candidate: IR compile, fixed-backend build, two identical runs (stdout+stderr+exit
  code hashed), runtime window, `.reference_output` check (exact, md5 for hashed suites, or fpcmp
  relative tolerance), ASan+UBSan build of the source (`-fno-sanitize-recover=all`) with the same output.

Run
- `scripts/v3_benchmarks.py --workers 1` (serial, so runtimes are not inflated by parallel load:
  a 4-worker run measured Shootout/sieve at 5.25 s versus 3.6 s alone): 688 s.
- Result: **101 programs kept** of 124 candidates (`data/v3/benchmarks.csv`):
  24 v2, 30 PolyBench, 47 llvm-test-suite. Datasets: PolyBench mostly MEDIUM, some LARGE/EXTRALARGE.
  test-suite `-O0` runtimes: median 0.53 s, max 4.1 s. Two test-suite programs use the suite's
  `SMALL_PROBLEM_SIZE` (fp-convert, nestedloop: default size above 5 s; checked against
  `.reference_output.small`).
- Runtime is flagged unreliable (`runtime_reliable=False`) for the 24 v2 programs (microseconds) and
  for pb_bicg (0.042 s) and pb_jacobi_1d (0.016 s), whose largest dataset stays under 0.05 s. They are
  kept for static, size and dynamic-count results and left out of runtime statistics.
- v2 quick mode rerun after the shared-code changes: 156/156 succeeded.

Dropped (23, reason recorded per row in the manifest)
- Undefined behaviour (UBSan): BenchmarkGame/nsieve-bits, Shootout/methcall, Shootout/objinst.
- Misc/flops: prints timings (non-deterministic; with SMALL_PROBLEM_SIZE its sanitized run differs).
- `-O0` runtime above 5 s: CoyoteBench huffbench (22.9 s), lpbench (5.3 s), Linpack (5.9 s),
  Misc lowercase (12.2 s), oourafft (6.1 s), salsa20 (5.5 s), Shootout lists (8.0 s), matrix (6.0 s).
- `-O0` runtime below 0.05 s: McGill exptree, Shootout ackermann and hello, and Stanford bubblesort,
  intmm, oscar, perm, queens, quicksort, realmm, towers.

Notes
- LeakSanitizer is disabled (`ASAN_OPTIONS=detect_leaks=0`) for the sanitizer run: a leak is not
  undefined behaviour, and 7 otherwise clean programs (e.g. heat-3d, fannkuch) never free before exit.
- Programs near the 0.05 s / 5 s limits can move in or out if the machine is loaded; the manifest
  records the runtime that decided it.

## Phase 2: measurements (2026-10-20)

Done
- `src/measure.py`: one IR → `clang -O2 -c -Xclang -disable-llvm-passes -Wno-override-module x.ll`
  (the only backend command, `llvm_pipeline.BACKEND_FLAGS`) → link with the support objects
  (`clang x.o [polybench.o] -lm`) → run. Metrics added to the 8 IR counts:
  - `text_size`: `.text` section of `x.o` from `llvm-size -A`;
  - `dyn_instructions`: `valgrind --tool=cachegrind --cache-sim=no`, `summary:` Ir of the whole run
    (program + support object + libc start-up, identical across variants of a program);
  - runtime: `timed_runs` (1 warm-up + N timed, `taskset -c <core>`), median, bootstrap 95 % CI and
    coefficient of variation in `src/stats.py`; used in Phase 5.
  - output check: stdout+stderr in one stream plus exit code, hashed, compared with the program's
    `-O0` hash from the manifest. A mismatch fails the variant.
- Measurements are cached by IR hash (header lines excluded): identical IR gives an identical object,
  so reusing its numbers is exact.
- Interaction dataset gained the `context` column (`compute_interactions` groups by program and
  context when present; v2 behaviour unchanged) and the metrics `text_size`, `dyn_instructions`.

Run
- `scripts/v3_experiments.py --quick` (3 programs × 4 passes × C0): 51/51 succeeded, outputs identical.
- Pilot, 22 passes × 3 contexts: pb_gemm (with dynamic counts) 382 s, ts_misc_dt (without) 141 s;
  2910/2910 succeeded. Each program has 1,455 variants but only 252 and 234 distinct IRs.
  Cachegrind cost: about 8.4x the native runtime plus 0.4 s per run.

Budget decision
- Static and size metrics and the output check: every variant of every program
  (estimated 0.9 h wall with 8 workers).
- Pair-level dynamic counts: all programs with `-O0` runtime ≤ 1.0 s, i.e. 78 of 101 (24 v2,
  30 PolyBench, 24 test-suite); estimated ~31 CPU-hours, ~4 h wall with 8 workers (an earlier
  version of this entry said 2 h; that was a mistake in copying the estimate). The 23 longer programs (test-suite, 1.0 to 4.1 s)
  get dynamic counts in Phase 5 only. Configurable in `config/v3.yaml` (`dynamic_pairs`).
- Phase 5: dynamic count and runtime for every variant (estimate in Phase 5's entry).

## Phase 3: contexts and bigger pass set (2026-10-20/21, set-up; results below when the run ends)

Done
- `config/passes.yaml`: `v3_passes` adds 10 passes to the 12 v2 ones (22 in total): sccp, jump-threading,
  correlated-propagation, dse, memcpyopt, indvars, loop-deletion, loop-vectorize, slp-vectorizer, inline.
  All 16 candidates were accepted by opt 21.1.8. Not used: ipsccp (module-level duplicate of sccp
  for our single-module programs), tailcallelim, mldst-motion, globalopt (narrow), loop-simplify and
  lcssa (canonicalisation that loop passes already run through their adaptors), loop-instsimplify
  (mostly covered by instcombine/simplifycfg). This keeps pairs at 462 per context.
- Contexts (`config/v3.yaml`): C0 raw `-O0` IR, C1 `mem2reg,sroa`, C2 `thinlto-pre-link<O1>`.
- `src/experiments_v3.py`: context × single × ordered pair per program; every variant measured by
  `measure_variant` (8 IR counts, text size, output check, dynamic count for the subset); resumable
  per program (signature of flags, passes, contexts, expected output); pair IR files deleted after
  measuring. `scripts/v3_experiments.py` writes `data/v3/measurements.csv.gz`,
  `interaction_dataset.csv.gz` (with `context`), `experiment_failures.csv`, `environment.json`,
  `run_config.json`. `scripts/v3_analysis.py`: variance decomposition (crossed design
  pair × program × context, `src/stats.py`), rank stability and sign flips between contexts, order
  dependence per context, aggregate and per-program graph statistics, example graphs.

Run
- Full run started 2026-10-20 21:50 UTC. The machine restarted overnight (WSL uptime reset);
  10 of 101 programs had finished. Restarted 2026-10-21 09:00 UTC; finished programs were skipped.

Note
- clang marks every function `noinline` at `-O0`, so the `inline` pass changes nothing in C0, and
  neither the C2 prefix nor the -O levels inline anything. Kept in the pass set and reported as such.

## Phase 4: prediction (set-up)

- Representations (`scripts/v3_features.py`, all from the raw `-O0` IR): hand (30 v2 features),
  funcprops (LLVM `print<func-properties>` with `-enable-detailed-function-properties`, 44 fields
  summed over functions, Max* fields maxed), opcodes (instruction-mix fractions, 43 opcodes),
  ir2vec (LLVM's in-tree IR2Vec, `print<ir2vec>` with the 75-d seed vocabulary from the LLVM 21.1.8
  source tree, function vectors summed). IR2Vec works with the installed LLVM; no pip package needed.
- Targets: direct I, and residual I − training pair mean (prediction adds the pair mean back).
- Models: mean, pair mean, ridge, random forest, gradient boosting, HistGradientBoosting (main;
  pass identities as native categorical features).
- Protocols: LOPO, grouped 10-fold, leave-one-suite-out, transfer polybench→testsuite and back.
  Train/test program sets are asserted disjoint in every fold.
- Tuning: nested, program-grouped (3 inner folds) in the 10-fold and suite protocols. LOPO uses each
  model's a-priori default (first grid point): measured fit times (HGB 9 s, RF 11 s, GB 35 s on
  46k rows) put nested tuning in 101 LOPO folds at ~11 h. Grids are fixed in `src/model.py`.
- Metrics: MAE, RMSE, R², Spearman, macro-F1, residual R² (share of the pair mean's squared error
  the model removes), per-program Spearman and NDCG@10 of the pair ranking, Wilcoxon signed-rank over
  per-program MAE vs pair mean with Holm correction and a bootstrap 95 % CI of the MAE difference.

## Phase 5: ordering (set-up)

- `src/ordering.py`: greedy and beam search (width 4) over predicted Δ(B|P) plus summed predicted
  interactions with the passes already chosen; repeats allowed (a repeat earns only the interactions
  of passes run since its last run); stops when no pass has positive estimated gain.
- `scripts/v3_ordering.py` (+ `v3_ordering_report.py`): programs = the 78 with pair-level dynamic
  counts (so the measured oracle graph exists for both objectives: text_size, dyn_instructions).
  Strategies per program: -O0/-O1/-O2/-O3/-Os (`opt -passes=default<Ox>` on the same IR, same
  backend), canonical LLVM-like order (first k of 22), 50 random sequences per budget k ∈ {8, 12, 16},
  graphs predicted (LOPO HistGradientBoosting; the model never saw the program), program-agnostic
  (pair and pass means of the other programs) and oracle (measured), each with greedy and beam,
  and greedy iterative search with real measurements (its evaluation count is the cost).
- Every variant gets IR count, text size, output check and dynamic count. Runtime (10 pinned runs
  + 1 warm-up, one per physical core 0,2,4,6,8,10) is measured for all non-random variants and the
  median and best random sequence per program/budget/objective of programs with reliable runtime
  (cost: 150 random variants × 11 runs per program was not affordable).

## Phase 6: tests

- `tests/`: 33 unit tests (interaction formula and the asymmetry identity, zero baseline, IR parser
  on a hand-written `.ll`, `llvm-size`/cachegrind parsers, reference-output comparison, hand,
  funcprops and opcode features, sequence builder on hand-made graphs including one where greedy is
  suboptimal and beam finds the brute-force optimum, grouped splits never share programs, pair mean
  excludes the test program, statistics helpers) and 1 integration test (3 programs × 4 passes × C0:
  experiments, resume, interactions, identity on measured data, features, LOPO prediction, a
  graph-built sequence applied and its output verified). All 34 pass (integration test 27 s while
  the full Phase 3 run was loading the machine).
- Found by the tests: none in the code under test; two expectations in my own test were miscounted
  (13 instructions, not 14) and were corrected.
