#!/usr/bin/env bash
# Full pipeline. Run from the project root: bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/check_env.py
python -m src.bench
python scripts/make_plots.py --update-readme
bash scripts/profile_nsys.sh || echo "TODO: nsys step failed, see README 'Profiling permissions'"
bash scripts/profile_ncu.sh  || echo "(optional) ncu step skipped/failed"
