# OptGraph v3.1 milestones

Plan: `/home/arav/.claude/plans/recursive-napping-moth.md` (approved 2026-10-22). Times in IST.

| # | Milestone | Status |
|---|---|---|
| M1 | Starting IR, metric fixes, real -Os/-Oz, one-invocation sequences, v3_1 config | done 22 Oct 13:14 IST |
| M2 | cBench integration + re-screen of the whole corpus | done 22 Oct 14:38 IST (129 programs) |
| M3 | Snapshot runner (S0..S4) | code done, tested |
| M4 | Static graphs (pair-mean, DCO-style, Pan-style) + static-graph error metrics | code done, tested |
| M5 | Predictors (snapshot features, recommender, validation tuning) | code done, tested |
| M6 | Ordering methods + budget accounting + search baselines | code done, tested on toy objective |
| M7 | Final-sequence measurement + serial runtime stage | code done, quick run OK |
| M8 | Statistics + report | code done, quick run OK |
| M8b | PREREGISTRATION.md (frozen, sha256) | done 22 Oct 14:56 IST, sha256 71e6f03d…f40e |
| M9 | Pilot (5 programs) + recomputed cost → user go/no-go | done 22 Oct 18:35 IST; go given |
| M10 | Full run A: screening + snapshots | done 23 Oct 10:05 IST (312,825 runs, 0 failed) |
| M11 | Full run B: models | done 23 Oct 16:21 IST (no OOM after the memory fix) |
| M12 | Full run C: searches + size/dyn | done 24 Oct 02:25 IST (129 programs, 0 failed, 0 output mismatches) |
| M12b | Full run D: runtime (serial, idle machine) | done 25 Oct ~09:15 IST (6,432 binaries, 0 errors, median CV 1.9%, 805 with CV > 5%) |
| M13 | Final report, README, PROGRESS | done 25 Oct 10:20 IST |
