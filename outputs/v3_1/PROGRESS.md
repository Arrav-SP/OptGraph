# OptGraph v3.1 progress log

Plan: /home/arav/.claude/plans/recursive-napping-moth.md (approved 2026-10-22). Times in IST.
v3 results (data/v3, outputs/v3) are not modified. No git operations.

## M1: starting IR, metrics, -Os/-Oz, sequences, config (22 Oct 13:14 IST)

- `config/v3_1.yaml` (selected with `OPTGRAPH_CONFIG=v3_1.yaml`), `version` key in both configs;
  every v3 script now writes to `data/<version>`, `outputs/<version>`, `work_dir/<version>`
  (`utils.version`). v3 defaults unchanged.
- Starting IR: `Benchmark.irflags` (recorded in the manifest column `ir_flags`); v3.1 uses
  `clang -O1 -Xclang -disable-llvm-passes` (no optnone/noinline; source-level noinline kept). Verified:
  `cgscc(inline)` now inlines (it was a no-op in v3).
- Metrics: calls to `llvm.lifetime.*`, `llvm.dbg.*`, `llvm.experimental.noalias.scope.decl`,
  `llvm.pseudoprobe` are not counted (no runtime effect; absent from v3's -O0 IR, so v3 numbers are
  unaffected).
- `-Os`/`-Oz`: `llvm_pipeline.add_size_attrs` adds clang's `optsize` / `optsize minsize` to every
  function definition before `default<Os|Oz>`; test shows Oz IR now differs from O2.
- `llvm_pipeline.sequence_pipeline`: one `opt` invocation per candidate sequence (explicit
  `function(...)` adaptors); test shows identical IR metrics to one invocation per pass on a probe.
  The pilot (M9) repeats this check on real programs.
- AArch64 compile-only object (`aarch64_object_cmd`) works with the installed clang.
- Experiment signatures only include irflags when set, so finished v3 results stay valid.
- Tests: 39 passed (4 new in `tests/test_v3_1_ir.py`).

## M2: cBench + re-screen (done 22 Oct 14:38 IST)

- cBench V1.1 (official tarball, sha256 8908d742…51e6), dataset 1 of each of the 32 programs. All .c
  files of a program are compiled with the v3.1 IR flags and linked with `llvm-link` into one module;
  -D/-I flags from each Makefile.llvm; the dataset command line from `_ccc_info_datasets` (arguments,
  `<` stdin, `>`/`2>` captured directly); the program runs in a fresh directory with links to its source
  directory's files and its loop-wrap count; its output is stdout + stderr + every file it creates.
- Loop-wrap counts tuned iteratively (re-measured after each scaling step) to ~0.5 s per run at the
  starting IR. The first version scaled from one run, which is mostly process start-up, and left
  dijkstra and stringsearch at ~1 ms; fixed and the whole corpus re-screened.
- Per-program fixes (config/v3_1.yaml): ghostscript `-include stdio.h` (its dprintf macro), ispell and pgp
  `-std=c89 -D_XOPEN_SOURCE=500` (getline), mad and pgp drop an unused/duplicate source file.
- v3.1 keeps test-suite programs below 0.05 s (flagged runtime-unreliable) instead of dropping them.
- Result (`data/v3_1/benchmarks.csv`, three screening runs, the last 1,081 s): **129 programs**
  (63 test-suite, 30 PolyBench, 24 v2, 12 cBench); 89 with a reliable runtime.
- cBench drops (20 of 32): ASan/UBSan found undefined behaviour or memory errors in jpeg_c, jpeg_d,
  tiff2bw, tiff2rgba, tiffdither, tiffmedian, patricia, rsynth, blowfish_d/e, rijndael_d/e, gsm, pgp_d/e;
  lame (rand/time) and sha (uninitialised data; digests differ between identical runs) are
  non-deterministic; ispell exits 1 on its dataset; mad and ghostscript need zlib headers, which are not
  installed (`sudo apt install zlib1g-dev`, not done; asked the user).

## M3–M8: code (22 Oct, tests 61 → all passing)

- M3 `llvm_pipeline.o2_prefix_pipelines`: S1–S4 = first 10/15/22/29 top-level stages of opt's printed
  `default<O2>`; test shows S4 == `default<O2>`. Dynamic counts only at S0 (`dynamic_contexts`).
- M4 `src/static_graphs.py`: pair-mean, DCO-style classes + positive-edge sub-sequences, Pan-style
  synergy graph + weighted walks, static-graph error metrics.
- M5 `src/predictor.py`: snapshot-matched features (`scripts/v3_features.py` per snapshot), residual HGB
  with validation-split tuning per LOPO fold, gap-closed metric, reduced-rank ridge (cold-start).
- M6 `src/search.py`, `src/ordering.py`: budget accounting (E, O, seconds), repeat cap, re-prediction,
  graph-guided search, GA, random, OpenTuner 0.8.8 (in-process, seeded; pinned in requirements.txt),
  measured greedy. Bugs found by tests: a dead end in random sequence generation (infinite loop) fixed.
- M7/M8 `scripts/v3_1_ordering.py` (search / measure / runtime stages; the runtime stage refuses to start
  while other OptGraph processes run), `scripts/v3_1_claims.py`, `scripts/v3_1_report.py`,
  `scripts/run_v3_1.sh`. Searches run one process per program (OpenTuner's SQLite must not cross threads).
- Quick end-to-end run (3 programs × 4 passes × S0,S2): all stages complete, 102/102 experiments, all
  best sequences reproduce the program output.

## M8b: pre-registration (frozen 22 Oct 14:56 IST)

- `outputs/v3_1/PREREGISTRATION.md`, sha256
  **71e6f03d91828c91d7fa1952af322cd1fb505c748735c92e580d1b5d97dff40e**, read-only, not edited after this.
- One choice beyond the approved plan, made stricter: H4's pool of "other methods" also contains the static
  graph itself.

## M9: pilot (started 22 Oct 14:58 IST)

- `config/v3_1_pilot.yaml` / `data/v3_1_pilot/`: 5 programs (pb_gemm, ts_misc_dt, cb_bzip2e,
  mix_01_matrix_mul, ts_shootout_heapsort), all 22 passes, all 5 snapshots, full budget.
- Run (22 Oct): experiments 15:02–15:48 IST (12,125 experiments, 0 failed; per program 147 s to
  2,758 s), features + claims 1 min, searches 15:49–16:30 (155–186 s per program; cb_bzip2e 2,488 s:
  28k-line IR), measurement 16:30–16:55 (730 sequences, 0 output mismatches), runtime stage alone
  17:23–18:35 (291 binaries, 0 errors, median CV per program 1.2–3.9 %).
- Budget accounting check (E = 200): every method used its budget except graph_guided (stops when no
  proposal improves, E 18) and measured_greedy (E 155, its true count); re-prediction used 1,164 opt
  calls and 257 s per program (feature extraction), the most expensive method in wall-clock.
- Not a result (5 programs; LOPO models trained on 4 programs): at E = 200 the search baselines reach
  0.93–0.96 of -Oz's IR count; re-prediction and the graph methods do not improve with budget.
- Recomputed full cost (fit to pilot timings: experiments 150 s + 0.076 s per IR line + 982 s per
  second of -O0 runtime; searches 155 s + 0.083 s per IR line): parallel stages 13–18 h wall, runtime
  stage 28.7 h serial (≈73 binaries per timeable program, 89 programs, 20 runs each) — total 41–47 h,
  above the plan's 37–40 h because of the runtime stage.

## Deviation D1 (22 Oct, 20:2x IST, before any full v3.1 result)

- User decision after the pilot: install zlib (`sudo apt install -y zlib1g-dev`, done by the user) so
  that cBench mad and ghostscript could be screened. Only those two rows were re-screened and merged
  (`v3_benchmarks.py --merge`; every other row is byte-identical to the frozen manifest). Both now
  build (linked with -lz, `cbench_link_libs`) and are dropped by UBSan: mad `synth.c:654` index out of
  bounds, ghostscript `gsalloc.c:125` left shift of a negative value.
- Effect: the kept corpus is unchanged (129 programs, same set). The manifest file changed (two drop
  reasons), so its sha256 is now a8d287810a00681ac41982f87eea4f36eb109d2079c75e21ec811571b8525771
  instead of c031f5b2…334f named in PREREGISTRATION.md. Screening signature re-stamped (the new config
  key only affects those two programs).
- Merge bug found while doing this (pandas 3 string columns rejected float values; the failure was hidden
  by an output filter in my command) — fixed and checked on a scratch copy first.

## Decisions for the full run (user, 22 Oct)

- Runtime stage keeps 20 runs per binary as pre-registered (~29 h serial).
- Rule agreed before the run: if re-prediction loses (H4 not supported), Claim 4 is reported as not
  supported; no tweaking and re-running.

## M10: full snapshot experiments (done 23 Oct 10:05 IST)

- 129 programs × 5 snapshots × (1 + 22 + 462) = **312,825 opt runs, 0 failed**; every variant reproduced
  its program's output. Started 22 Oct 20:16 IST, 13 h 49 min. Throughput ~2 model-CPU-hours per wall
  hour with 8 workers, half of the 4.5 assumed: the pilot ran 5 programs on 12 hardware threads, the full
  run 8 at a time on 6 physical cores.
- The machine restarted at ~10:15 IST, during the Claims 1–3 model step (no results lost; that step
  recomputes from the saved experiments). Chain restarted 10:20 IST; experiments reused from cache (4 s),
  tests 61 passed.
- 23 Oct: WSL shut down again at 10:30 IST (Windows itself did not restart; it has been up since 21 Oct
  20:56). Cause: WSL stops when no Windows-side client (VS Code / Claude Code session) is attached, and
  detached background jobs do not count; the Claude Code session had restarted. On 21 Oct Windows
  itself went down four times uncleanly (Kernel-Power event 41: crash / forced power-off / power loss)
  and once by a Start-menu power-off. Fix: a Windows-side keep-alive (`wsl.exe -- sleep infinity` in a
  minimized window, started 11:20 IST) holds WSL up independently of the editor. Chain restarted 11:2x IST.

## Root cause of the 23 Oct failures: out of memory (fixed 12:35 IST)

- `journalctl -b -1` / `-b -2`: "Out of memory: Killed process … (python) anon-rss ~1.2 GB, shmem ~0.47 GB"
  in the Claims 1–3 LOPO step (04:57 and 05:58 UTC), followed by WSL being torn down. 6 parallel LOPO
  workers × ~1.7 GB exceeded WSL's 7.5 GB RAM + 2 GB swap (host: 15.3 GB). My earlier explanation
  (WSL idle shutdown) was wrong for these two failures; the 21 Oct failures were Windows-level (event 41).
- Fixes: LOPO runs 2 workers × 3 threads (same cores, ~1/3 of the memory); claims and searches read the
  dataset in chunks and keep only the slice they need; the claims step is skipped on a restart when its
  inputs are unchanged (a refit would also have invalidated finished searches).
- Resilience: Windows scheduled task "OptGraph v3.1 watchdog" (user-level, every 15 min, headless) runs
  `scripts/watchdog_v3_1.sh`: restarts the resumable chain if it is not alive (log:
  ~/optgraph_work/v3_1_watchdog.log) and keeps a Windows-side WSL keep-alive client running. The chain
  writes data/v3_1/RUN_COMPLETE at the end; the watchdog then does nothing. Delete the task afterwards.
- Tested: the task's first run (12:34 IST) found the run dead and restarted it; 61 tests passed.

- 24 Oct 09:38 IST: runtime stage had refused at 02:25 (its busy check matched its own parent shell); fixed in v3_1_ordering.py (only python workers count; refusal now exits non-zero), RUN_COMPLETE removed, runtime stage restarted 09:40 IST. Searches + measurement complete: 129 programs, 0 failed.

- 25 Oct 10:20 IST: full v3.1 run complete (RUN_COMPLETE). Runtime: 6,432 binaries, 0 errors, median CV 1.9%, 805 flagged CV > 5% (a Windows background load slowed some hours; randomised order spreads it across methods). Pre-registered results (Holm across 4): H1 supported (oracle/static 0.949, p 2.6e-9); H2 supported (median rho 0.29, p 8e-21); H3 supported (residual R2 0.50, p 4.1e-21); H4 not supported (repredict/GA 1.107, p 1.0). Per the agreed rule, Claim 4 is reported as not supported, no re-runs.
- 25 Oct 10:26 IST: exploratory analyses (scripts/v3_1_analysis.py) -> outputs/v3_1/ANALYSIS.md (takeaways in ANALYSIS_TAKEAWAYS.md).
- 25 Oct 12:17 IST: paper draft in paper/ (make_numbers.py -> numbers.tex, sections, refs.bib, overleaf.zip); README rebuild command added and verified (data/v3_1 untouched).
