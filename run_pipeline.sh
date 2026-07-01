#!/usr/bin/env bash
# Run estimation + plot notebooks in sequence using nbconvert.
# Each notebook is executed in-place (overwrites itself with outputs).
# The script aborts immediately if any notebook fails.

set -euo pipefail

JUPYTER=/Users/gabbi/miniconda3/envs/SaaLab/bin/jupyter
REPO="$(cd "$(dirname "$0")" && pwd)"

run_notebook() {
    local nb="$1"
    echo ""
    echo "================================================================"
    echo "  Running: $nb"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    "$JUPYTER" nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=7200 \
        "$REPO/$nb"
    echo "  DONE: $nb  ($(date '+%H:%M:%S'))"
}

run_notebook "second_estimation.ipynb"
run_notebook "plots_second_estimation.ipynb"
run_notebook "third_estimation.ipynb"
run_notebook "plots_third_estimation.ipynb"

echo ""
echo "================================================================"
echo "  All notebooks completed successfully."
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
