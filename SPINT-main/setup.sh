#!/usr/bin/env bash
# Creates the `spint` conda env for training, evaluation, and Docker builds.
# (For EvalAI submission, see README — the `evalai` CLI is installed in a
# separate env because of dependency conflicts with `spint`.)
# Usage: bash setup.sh

set -e

export PYTHONNOUSERSITE=1

if command -v mamba >/dev/null 2>&1; then
    ENV_TOOL=mamba
else
    ENV_TOOL=conda
fi

"$ENV_TOOL" env create -f environment.yaml
"$ENV_TOOL" run -n spint python -m pip install --no-deps falcon-challenge==1.0.2
"$ENV_TOOL" run -n spint python -m pip install -e .

echo
echo "Done. To use the environment, run:"
echo "    mamba activate spint"
