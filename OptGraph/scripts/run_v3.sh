#!/usr/bin/env bash
# OptGraph v3: reproduce every result from scratch.
#
#   bash scripts/run_v3.sh            # full run (hours; resumable - rerun to continue)
#   bash scripts/run_v3.sh --quick    # tiny subset (3 programs x 4 passes x C0), a few minutes
#
# Every phase skips work whose outputs are up to date: the benchmark manifest is rebuilt only
# when its signature (suite config, screening code, flags) changed, experiments/ordering are resumable per program, model
# configurations are cached by dataset hash. Heavy files live in work_dir (config/v3.yaml).
# Does not write the v2 files (environment.json, data/*.csv, outputs/{plots,reports,...}).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
Q=""
TAG=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; TAG="quick"; fi
OUT="outputs/v3${TAG:+/$TAG}"
DATA="data/v3${TAG:+/$TAG}"
mkdir -p "$OUT" "$DATA"
LOG="$OUT/run_v3_$(date -u +%Y%m%dT%H%M%SZ).log"
START=$(date +%s)

phase() { echo; echo "=== $* [$(date -u +%H:%M:%S)]" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

phase "environment and tests"
run $PY scripts/check_environment.py --no-write
run $PY -m pytest tests -q

phase "Phase 1: benchmarks"
# Rebuilt only when the suite config, screening code or compile flags changed (signature).
run $PY scripts/v3_benchmarks.py --workers 1 --if-stale

phase "Phases 2-3: pair experiments in every context"
run $PY scripts/v3_experiments.py $Q

phase "Phase 3: RQ1 analysis"
run $PY scripts/v3_analysis.py $Q

phase "Phase 4: representations and prediction (RQ2)"
run $PY scripts/v3_features.py $Q
run $PY scripts/v3_train.py $Q

phase "Phase 5: graph-guided ordering (RQ3)"
run $PY scripts/v3_ordering.py $Q

phase "Phase 7: report"
run $PY scripts/v3_report.py $Q --elapsed $(( $(date +%s) - START ))

phase "tests (again)"
run $PY -m pytest tests -q
echo "total $(( $(date +%s) - START ))s; log: $LOG; report: $OUT/report/index.html" | tee -a "$LOG"
