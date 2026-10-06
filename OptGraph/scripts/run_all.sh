#!/usr/bin/env bash
# Full reproduction: environment check, experiments, graphs, models, analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/check_environment.py
python scripts/run_experiments.py --full --verify-outputs
python scripts/build_graph.py
python scripts/train_model.py
python scripts/analyze_similarity.py
python scripts/summarize_results.py
