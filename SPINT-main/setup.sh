#!/usr/bin/env bash
# Creates the `spint` conda env for training, evaluation, and Docker builds.
# (For EvalAI submission, see README — the `evalai` CLI is installed in a
# separate env because of dependency conflicts with `spint`.)
# Usage: bash setup.sh

set -e

mamba env create -f environment.yaml
mamba run -n spint pip install -e .

echo
echo "Done. To use the environment, run:"
echo "    mamba activate spint"
