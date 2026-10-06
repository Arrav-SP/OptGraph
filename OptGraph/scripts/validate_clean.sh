#!/usr/bin/env bash
# Clean-state check: wipe the quick/demo workspaces, then rerun quick mode and the demo.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf experiments/quick experiments/demo data/quick outputs/demo
python scripts/check_environment.py
python scripts/run_experiments.py --quick --verify-outputs
python run_demo.py "$@"
