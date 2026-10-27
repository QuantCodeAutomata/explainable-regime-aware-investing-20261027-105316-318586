#!/usr/bin/env bash
# Reproduce the Explainable Regime-Aware Investing replication end to end.
set -uo pipefail

cd "$(dirname "$0")"

python -m pytest -m "not market_data"

python -m exp.exp1_hmm --full
python -m exp.exp2_knn --full
python -m exp.exp3_benchmarks --full
