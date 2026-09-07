#!/usr/bin/env bash
# Unlaunched entry point for B1/B2/B3/B4 training-method scaffolding sweeps.
# Refuses to start without explicit GPU authorization.
set -euo pipefail

if [[ "${1:-}" != "--i-have-authorization" ]]; then
  echo "REFUSED: training-method scaffolding sweeps require GPU authorization." >&2
  echo "This entry point is inert by design. Re-run with --i-have-authorization only" >&2
  echo "after explicit protocol approval and CUDA_VISIBLE_DEVICES allocation." >&2
  exit 2
fi

shift
echo "REFUSED: this host does not have GPU authorization for training runs." >&2
echo "Configs are registered under streaming_calibration_exp/configs/experiment/." >&2
echo "No trainer was invoked." >&2
exit 3
