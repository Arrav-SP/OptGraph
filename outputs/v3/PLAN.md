# OptGraph v3 plan

Written in Phase 0 (2026-10-20) from measurements on this machine:
AMD Ryzen 5 8640HS (6 cores / 12 threads), 7 GB RAM, Ubuntu 26.04.1 (WSL2),
Clang/LLVM 21.1.8, valgrind 3.26.0, Python 3.14.4 in `~/.venvs/optgraph`.

## Ground rules this plan follows

- v2 files in `data/` and `outputs/` are not modified. v3 writes `data/v3/`, `outputs/v3/`.
  `run_demo.py` and `scripts/run_all.sh` keep working and keep their behaviour
  (v2's `passes:` list in `config/passes.yaml` is unchanged; v3 passes are a separate key).
- Heavy files (IR, objects, binaries, downloaded suites) live in the work directory,
  `~/optgraph_work/` by default, set in `config/v3.yaml` (`work_dir`).
- Seed 42 everywhere; `environment.json` is written into `data/v3/` for every full run.
- No git operations.

## Measured costs (Phase 0 probes, PolyBench gemm, MEDIUM dataset, 532 IR lines)

| step | cost |
|---|---|
| `opt` one pass | 20 to 35 ms per invocation (process start dominates) |
| backend `clang -O2 -c -Xclang -disable-llvm-passes x.ll` | 28 ms |
| `-O0` runtime, SMALL / MEDIUM / LARGE | 0.027 s / 0.064 s / 3.77 s |
| cachegrind `--cache-sim=no`, MEDIUM `-O0` | 1.7 s (410 M instructions); about 25x the native runtime |

Backend check: the command above gives the same `.text` (1420 B) as `llc -O2 -relocation-model=pic`,
and 2145 B without `-disable-llvm-passes`, so no IR pass is rerun. It is used for every variant.

## Phases, files, estimates

N ≈ 110 programs (24 v2 + 30 PolyBench + about 55 test-suite after filtering), P = 22 passes,
3 contexts. Estimates are CPU-time on this machine, wall time assumes 8 to 10 workers.
Identical IR or identical objects are measured once (hash cache); this is exact, not an approximation,
because the same object gives the same binary.

| phase | adds / changes | work | estimate |
|---|---|---|---|
| 1 benchmarks | `config/v3.yaml`, `src/suites.py` (discovery, dataset choice, sanitizer check, manifest), `src/utils.py` (Benchmark gets suite, cflags, support files), `src/llvm_pipeline.py` (per-program flags, support objects, backend command), `scripts/v3_benchmarks.py` → `data/v3/benchmarks.csv` | ~130 candidates × (5 compiles + up to 5 dataset runs + ASan/UBSan run) | 15 to 30 min |
| 2 measurements | `src/metrics.py` (text size via `llvm-size`, cachegrind count, timed runs with bootstrap CI and CV), `src/measure.py` (cached build/run/measure of one IR) | per variant: backend 30 to 100 ms, run 0.05 to 2 s, cachegrind ~25x run | per-variant costs are measured, then the pair-subset size is chosen (PROGRESS.md) |
| 3 contexts | `config/passes.yaml` (`v3_passes`), `src/experiments.py` (context-aware, resumable pair runner), `src/interaction.py` (`context` column), `scripts/v3_experiments.py`, `scripts/v3_analysis.py` (stability, variance decomposition, order) | 3 × (1 + 22 + 462) = 1,455 opt + backend per program → ~160k | static+size: 1 to 3 h wall. Output check of every variant (deduplicated): ~1 h. Dynamic count on pairs: subset only, ~2 h |
| 4 prediction | `src/features.py` (func-properties, opcode histogram, IR2Vec if usable), `src/model.py` (HistGB, residual target, grouped/nested CV, leave-one-suite-out), `src/stats.py` (Wilcoxon + Holm, bootstrap, NDCG), `scripts/v3_train.py` | ~100k rows per context and metric; LOPO × models × representations | 1 to 3 h wall (n_jobs) |
| 5 ordering | `src/ordering.py` (greedy + beam), `scripts/v3_ordering.py` | ~180 variants per program (O-levels, canonical, 50 random × 3 budgets, 3 graphs × 2 builders × 3 budgets, measured search) | dyn count ~1.5 h, runtime ~2 h (timed runs pinned, one per physical core), search cost reported |
| 6 tests | `tests/` (pytest) | unit + one integration test on 3 programs × 4 passes × 1 context | < 1 min |
| 7 full run | `scripts/run_v3.sh` (resumable, `--quick`), `scripts/v3_report.py` → tables (CSV/MD/LaTeX), figures (PDF/PNG), `report/index.html`, `RESULTS.md`, `results_v3.json` | everything above | total ≈ 6 to 10 h wall |

Disk (work directory): ~1,455 IR files per program × 30 to 100 KB ≈ 5 to 15 GB in total, plus
objects. 950 GB is free on the WSL disk. Repo additions are only CSV, JSON, figures and the report
(tens of MB).

## Design decisions (to be confirmed by the measurements, recorded in PROGRESS.md)

- PolyBench: the official PolyBench/C 4.2.1 release (SourceForge tarball, sha256 `426519ee…689a4`),
  not the llvm-test-suite copy, which inlines `polybench.c` into the header and runs every kernel
  twice (a StrictFP copy). The official kernels were checked to be identical (gemm diffed).
  `polybench.c` is compiled once with `clang -O2`; only the kernel module is optimised.
  Dataset: the smallest of MINI … EXTRALARGE whose `-O0` runtime is ≥ 0.05 s (and ≤ 2 s).
  Output check: hash of the `POLYBENCH_DUMP_ARRAYS` dump (stderr) plus stdout and exit code.
- test-suite: C programs from the listed `SingleSource/Benchmarks` directories, kept if they compile,
  run deterministically (two identical runs), take 0.05 to 5 s at `-O0`, and match
  `.reference_output` (exit code line handled as lit does).
- v2 programs stay in the static and size experiments; they run in microseconds, so their runtime is
  flagged as not meaningful and left out of runtime statistics. Dynamic count is still exact for them.
- Contexts: C0 raw `-O0` IR; C1 `mem2reg,sroa`; C2 `thinlto-pre-link<O1>` (LLVM's documented
  module-simplification pipeline at O1, i.e. `default<O1>` without its optimisation half).
- Pass set (22): the 12 v2 passes plus sccp, jump-threading, correlated-propagation, dse, memcpyopt,
  indvars, loop-deletion, loop-vectorize, slp-vectorizer, inline. All 16 candidates validated; the
  ones not used are listed with reasons in PROGRESS.md.
- Ordering objective: code size for all programs (complete pair data); dynamic instruction count on
  the programs that have pair-level dynamic data. All four metrics are measured on every resulting
  variant.
