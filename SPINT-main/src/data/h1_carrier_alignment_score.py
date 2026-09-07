"""STATUS: DEPRIORITIZED_SEALED_NOT_ON_CRITICAL_PATH  (2026-08-09)

Reason: H-NF competes with H-C0 (0.48662), not with SPINT (0.49683). The
compact neural-only identity result already exists at 102.6x fewer identity
parameters. The only measured headroom for a frozen 4-dim per-channel
statistic on top of the learned 32-dim activity path is H-LS - H-C0 = +0.0133,
so the expected ceiling is about +0.003 against SPINT. This is a cost and
generality claim, not a performance claim.

Known defects, recorded so that a future revival does not rediscover them:
  D1 Channel 66 is dead in all 13 recordings, so the raise policy makes the
     profile builder fail on every recording. A source-declared frozen
     dead-channel list is the fix.
  D2 The second PCA stage is an identity map (see M3). Delete it and the
     constant RESPONSE_BASIS_DIM_D, or give stage 2 a different objective.
  D3 The no-label guard covers the fit but not the support selection, and it
     checks only five known keyword names. Add a hash check.
  D4 The basis fit hard-codes the fold-0 source set. A five-date extension
     would leak on four of five dates. The fit needs an outer_date argument.
  D5 The mask equivalence of M1 is assumed, not asserted.
  D6 The RMS normalizer has no numerical preflight. See the V1 numerical
     silence incident.

---

W3 alignment scorer for the H1 CarrierID carrier family (CPU-only, isolated).

This module is deliberately standalone.  It imports nothing from the live
training/producer graph: no ``h1_m4_eb_pilot``, no datamodule, no decoder, no
target file.  It takes already-materialized carrier arrays and returns a plain
dictionary.  A separate function writes the receipt.

The score is the alignment criterion proposed in
``HANDOFF_LABEL_FREE_CARRIER_AND_HEADROOM_20260809.md`` (item W3)::

    score = separability(C within one recording) / drift(C across recordings)

It is computed on the 11 fold-0 source recordings only.  It uses no target
data, no label, and no decoder training.  A better score selects a candidate
for W2; it is not a result and not an R^2.

Status string: ``IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODULE_STATUS = "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"
VERSION = "h1_carrier_alignment_score_v1"
CARRIER_WIDTH = 4
EPS = 1.0e-12

# The exactly 11 fold-0 source recordings (H1_M4_FOLD0_SOURCE).  Hard-coded so
# this scorer has no import edge into the live producer graph.  Any change to
# the source partition requires a new version string and a new module.
SOURCE_RECORDINGS: tuple[str, ...] = (
    "ses-19250108T110520",
    "ses-19250108T111022",
    "ses-19250108T111455",
    "ses-19250113T120811",
    "ses-19250113T121303",
    "ses-19250115T110633",
    "ses-19250115T111328",
    "ses-19250119T113543",
    "ses-19250119T114045",
    "ses-19250120T115044",
    "ses-19250120T115537",
)

SCORER_DEFINITION = """\
W3 alignment score (frozen definition, version h1_carrier_alignment_score_v1).

For a recording r, let C_r be its per-channel carrier array of shape [N, 4].

separability(C_r):
    The within-recording channel separability.  Defined as the total
    population variance of the carrier distribution across the N channels:
        separability(C_r) = sum_{j=1..4} var(C_r[:, j])   (population, ddof=0)
    This is the trace of the 4x4 population covariance of the N carrier rows.
    It is zero only when every channel in the recording shares one identical
    carrier.  It is invariant to a complete row permutation of C_r, so it
    measures the carrier distribution, not the channel attachment.

drift({C_r}):
    The cross-recording distribution shift.  Defined as the unweighted
    between-recording variance of the per-recording carrier means:
        mu_r    = mean_n C_r[n, :]                         (one [4] per recording)
        mu      = mean_r mu_r                              (unweighted global mean)
        drift   = (1/R) * sum_r ||mu_r - mu||^2
    This is zero if and only if every recording shares the same mean carrier.

score:
    score = mean_r separability(C_r) / drift({C_r})
    The numerator averages the per-recording separability over the R source
    recordings.  The score is an F-like alignment ratio: large when channels
    are spread within each recording while the recordings agree on the carrier
    distribution, small when the recordings drift apart.  If drift is below the
    EPS floor the scorer refuses to divide and raises DriftUndefinedError.

This score measures the carrier distribution only.  It cannot distinguish a
carrier from its complete row permutation; correct channel attachment is tested
separately by the H-NF-RS control arm in the actual evaluation.
"""


class CarrierAlignmentScoreError(ValueError):
    """Fail-closed violation of the W3 alignment scorer contract."""


class DriftUndefinedError(CarrierAlignmentScoreError):
    """Raised when cross-recording drift is zero/undefined instead of dividing."""


def _as_finite_carrier(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != CARRIER_WIDTH:
        raise CarrierAlignmentScoreError(
            f"{name} must be a finite [N, {CARRIER_WIDTH}] carrier array, got shape {array.shape}"
        )
    if array.shape[0] < 2:
        raise CarrierAlignmentScoreError(f"{name} needs at least two channels to define separability")
    if not np.isfinite(array).all():
        raise CarrierAlignmentScoreError(f"{name} contains non-finite carrier values")
    return array


def separability(carrier: np.ndarray) -> float:
    """Within-recording channel separability (total population carrier variance).

    Pure function: takes one explicit ``[N, 4]`` array, returns one scalar.
    """

    values = _as_finite_carrier(carrier, name="carrier")
    # Total population variance across the four carrier dimensions.
    return float(np.sum(np.var(values, axis=0, ddof=0)))


def drift(carriers: Sequence[np.ndarray] | Mapping[str, np.ndarray]) -> float:
    """Cross-recording distribution shift (between-recording mean variance).

    Pure function: takes an explicit sequence/mapping of ``[N, 4]`` arrays.
    Returns zero if and only if every recording shares the same mean carrier.
    """

    if isinstance(carriers, Mapping):
        recording_arrays = [np.asarray(value, dtype=np.float64) for value in carriers.values()]
        names = list(carriers.keys())
    else:
        recording_arrays = [np.asarray(value, dtype=np.float64) for value in carriers]
        names = [f"recording_{index}" for index in range(len(recording_arrays))]
    if len(recording_arrays) < 2:
        raise CarrierAlignmentScoreError(
            f"drift needs at least two recordings, got {len(recording_arrays)}"
        )
    means = np.zeros((len(recording_arrays), CARRIER_WIDTH), dtype=np.float64)
    for index, array in enumerate(recording_arrays):
        means[index] = np.mean(_as_finite_carrier(array, name=names[index]), axis=0)
    global_mean = np.mean(means, axis=0)
    return float(np.mean(np.sum(np.square(means - global_mean[None, :]), axis=1)))


def _validate_source_keys(carriers_by_recording: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    source_set = set(SOURCE_RECORDINGS)
    keys = tuple(carriers_by_recording.keys())
    unknown = [name for name in keys if name not in source_set]
    if unknown:
        raise CarrierAlignmentScoreError(
            "W3 alignment scorer accepts the 11 fold-0 source recordings only; "
            f"rejected non-source recording names: {unknown}"
        )
    if len(keys) < 2:
        raise CarrierAlignmentScoreError(
            f"W3 alignment scorer needs at least two source recordings, got {len(keys)}"
        )
    if len(set(keys)) != len(keys):
        raise CarrierAlignmentScoreError("W3 alignment scorer received duplicate recording names")
    return keys


def score(carriers_by_recording: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Compute the W3 alignment score over source recordings.

    Returns a plain dictionary.  Does not write any file and reads no target
    data.  Use :func:`write_receipt` to persist the result.
    """

    if not isinstance(carriers_by_recording, Mapping):
        raise CarrierAlignmentScoreError("carriers_by_recording must be a mapping {name: [N,4]}")
    keys = _validate_source_keys(carriers_by_recording)
    arrays = {name: _as_finite_carrier(carriers_by_recording[name], name=name) for name in keys}
    separabilities = {name: separability(arrays[name]) for name in keys}
    mean_separability = float(np.mean(list(separabilities.values()), dtype=np.float64))
    drift_value = drift({name: arrays[name] for name in keys})
    if not np.isfinite(drift_value) or drift_value <= EPS:
        raise DriftUndefinedError(
            "cross-recording drift is zero/undefined; the W3 alignment score refuses to divide "
            "(every recording shares the same mean carrier, so the alignment ratio is undefined)"
        )
    return {
        "schema": "h1_carrier_alignment_score_v1",
        "version": VERSION,
        "module_status": MODULE_STATUS,
        "definition": SCORER_DEFINITION,
        "recording_names": list(keys),
        "recording_count": len(keys),
        "separability_per_recording": separabilities,
        "mean_separability": mean_separability,
        "drift": drift_value,
        "score": mean_separability / drift_value,
        "carrier_width": CARRIER_WIDTH,
        "uses_target_data": False,
        "uses_labels": False,
        "uses_decoder_training": False,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_receipt(path: str | Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Persist one W3 score result as canonical JSON (no silent overwrite).

    The path is chosen by the caller.  This scorer never writes into
    ``sua_exploration/results/`` or ``SPINT-main/pilot_artifacts/`` on its own.
    """

    if result.get("schema") != "h1_carrier_alignment_score_v1" or result.get("version") != VERSION:
        raise CarrierAlignmentScoreError("result is not a W3 alignment-score receipt")
    body = dict(result)
    payload = _canonical_json_bytes(body)
    receipt_path = Path(path).resolve()
    output = receipt_path
    if output.exists():
        raise FileExistsError(f"W3 alignment scorer refuses to overwrite an existing receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "receipt_path": str(output),
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
