#!/usr/bin/env bash
# OptGraph v3.1 (plan: /home/arav/.claude/plans/recursive-napping-moth.md). Resumable.
#
#   bash scripts/run_v3_1.sh            # everything except the runtime stage
#   bash scripts/run_v3_1.sh --quick    # 3 programs x 4 passes x S0,S2, a few minutes
#   bash scripts/run_v3_1.sh --runtime  # runtime stage ONLY: run alone on an idle machine, last
#
# Writes data/v3_1/, outputs/v3_1/ and ~/optgraph_work/v3_1/. Never touches v2 or v3 results.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPTGRAPH_CONFIG=${OPTGRAPH_CONFIG:-v3_1.yaml}   # v3_1_pilot.yaml for the pilot
PY=${PYTHON:-python}
Q=""; TAG=""
case "${1:-}" in
  --quick) Q="--quick"; TAG="quick" ;;
  --runtime) exec $PY scripts/v3_1_ordering.py --stage runtime ;;
esac
VER=$(OPTGRAPH_CONFIG=$OPTGRAPH_CONFIG $PY -c "from src.utils import version; print(version())")
OUT="outputs/$VER${TAG:+/$TAG}"
mkdir -p "$OUT" "data/$VER${TAG:+/$TAG}"
LOG="$OUT/run_v3_1_$(TZ=Asia/Kolkata date +%Y%m%dT%H%M%S_IST).log"
START=$(date +%s)
phase() { echo; echo "=== $* [$(TZ=Asia/Kolkata date +"%H:%M:%S IST")]" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

phase "environment and tests"
run $PY scripts/check_environment.py --no-write
run env OPTGRAPH_CONFIG=v3_1.yaml $PY -m pytest tests -q   # tests use the main v3.1 manifest, also for the pilot
phase "M2: benchmarks (rebuilt only if the screening signature changed)"
run $PY scripts/v3_benchmarks.py --workers 1 --if-stale
phase "M3: snapshot experiments"
run $PY scripts/v3_experiments.py $Q
phase "M5: snapshot features"
run $PY scripts/v3_features.py $Q
phase "Claims 1-3"
run $PY scripts/v3_1_claims.py $Q
phase "Claim 4: searches"
run $PY scripts/v3_1_ordering.py --stage search $Q
phase "Claim 4: size, AArch64 size, executed instructions"
run $PY scripts/v3_1_ordering.py --stage measure $Q
phase "report (runtime results are added after: bash scripts/run_v3_1.sh --runtime, then the report again)"
run $PY scripts/v3_1_report.py $Q
echo "total $(( $(date +%s) - START ))s; log: $LOG" | tee -a "$LOG"
