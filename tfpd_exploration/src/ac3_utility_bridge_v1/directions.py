"""AC3-U CPU stage: rebuild the frozen direction estimates and bind the anchors.

Work order §7: the direction estimates consumed by UGE (R0.5) and U2 (R2) are
rebuilt by the frozen screen's own loaders on CPU, BEFORE any GPU process:

* ``R0``  sanity  -- ``screen.load_trial_set`` + ``matrix.static_row(..., 'R0')``;
* ``R0.5`` (UGE)  -- ``matrix.static_row(..., 'R0.5')``;
* ``R2``  (U2)    -- ``matrix.learned_row_estimates(row='R2', ...)`` with the
  hyperparameters read from the sealed ``selection.json`` ``rows.R2.selected``.

Each rebuilt estimate's ``theta_sha256`` (the frozen ``RowEstimate.payload``
digest, i.e. ``matrix.array_digest``) must equal the sealed ``screen.json``
``rows.{R0,R0.5,R2}.estimate_digest.theta_sha256``.  The trajectory cache is
bound through the frozen screen's own ``verify_input_lock`` (which re-derives
every array digest of ``trajectories.npz`` against the sidecar-pinned
``materialize.json``), because the ``.npz`` carries no sidecar of its own.

This module never touches CUDA and never opens the model checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from src.ac3_action_continuity_v1 import matrix as mtx
from src.ac3_action_continuity_v1 import screen as acs

from . import plan


class AC3UDirectionsError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3UDirectionsError(message)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_sidecar(path: Path) -> str:
    """Fail closed unless ``<path>.sha256`` matches the exact file bytes."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    _require(sidecar.exists(), f"missing sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="ascii").strip()
    _require(
        expected == f"{digest}  {path.name}",
        f"sidecar drift for {path.name}: {expected!r}",
    )
    return digest


def verify_predecessors(base: Path) -> dict[str, object]:
    """Bind every immutable predecessor of work order §2 at this moment."""
    root = Path(base).absolute()
    table: dict[str, str] = {}
    for relative in plan.PREDECESSOR_FILES:
        path = root / relative
        _require(path.exists(), f"immutable predecessor missing: {relative}")
        table[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in plan.SIDEcar_PINNED_PREDECESSORS:
        _verify_sidecar(root / relative)
    return {
        "files": table,
        "closure_sha256": hashlib.sha256(
            json.dumps(table, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sidecar_pinned": list(plan.SIDEcar_PINNED_PREDECESSORS),
        "trajectories_binding": (
            "trajectories.npz has no .sha256 sidecar; it is bound bit-for-bit "
            "through the sidecar-pinned materialize.json array_digests block via "
            "the frozen screen's verify_input_lock (recorded deviation, "
            "equivalent or stronger than a whole-file sidecar)"
        ),
        "sealed_cell_d_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
    }


def session_binding(manifest: Mapping[str, object]) -> dict[str, object]:
    """Per-session offsets into the materialize trial order (the replay binding)."""
    sessions = [str(item) for item in manifest["sessions"]]
    trial_ids = [str(item) for item in manifest["trial_ids"]]
    _require(tuple(sessions) == plan.SESSIONS, "the materialized session roster drift")
    blocks: dict[str, dict[str, object]] = {}
    offset = 0
    for session in sessions:
        members = [
            index for index, item in enumerate(trial_ids)
            if item.startswith(f"{session}:trial:")
        ]
        _require(
            len(members) > 0 and members == list(range(members[0], members[-1] + 1)),
            f"the materialized trial order is not contiguous inside {session}",
        )
        _require(members[0] == offset, f"session block offset drift at {session}")
        blocks[session] = {
            "session": session,
            "start": int(members[0]),
            "stop": int(members[-1]) + 1,
            "n_trials": len(members),
            "trial_ids": [trial_ids[index] for index in members],
        }
        offset = members[-1] + 1
    _require(offset == len(trial_ids), "the session blocks do not cover the trial roster")
    return {"sessions": blocks, "n_trials": len(trial_ids)}


def rebuild_directions(
    base: Path, *, ac3_root: Optional[Path] = None, attempt_sha256: Optional[str] = None,
) -> dict[str, object]:
    """Rebuild R0/R0.5/R2 with the frozen screen's own code and verify digests."""
    root = Path(ac3_root) if ac3_root is not None else Path(base).absolute() / plan.AC3_ROOT_RELATIVE
    started = time.monotonic()
    predecessors = verify_predecessors(base)
    screen = _read_json(root / "screen.json")
    selection = _read_json(root / "selection.json")
    trial_set = acs.load_trial_set(root)
    input_lock = acs.verify_input_lock(root, trial_set)
    binding = session_binding(_read_json(root / "materialize.json"))
    _require(
        [item["session"] for item in binding["sessions"].values()] == list(trial_set.sessions),
        "the rebuilt trial set and the materialize manifest disagree on session order",
    )
    table = acs.build_table(trial_set)

    rows: dict[str, dict[str, object]] = {}
    rebuilt: dict[str, np.ndarray] = {}
    # R0 sanity + R0.5 (UGE): the frozen static rows.
    for row in ("R0", "R0.5"):
        estimate = mtx.static_row(trial_set, row)
        payload = estimate.payload()
        sealed = screen["rows"][row]["estimate_digest"]
        rows[row] = {
            "source": "matrix.static_row(trial_set, row)",
            "theta_sha256": payload["theta_sha256"],
            "sealed_theta_sha256": sealed["theta_sha256"],
            "theta_digest_matches_sealed_screen": payload["theta_sha256"] == sealed["theta_sha256"],
            "credibility_sha256": payload["credibility_sha256"],
            "sealed_credibility_sha256": sealed["credibility_sha256"],
            "credibility_digest_matches_sealed_screen": (
                payload["credibility_sha256"] == sealed["credibility_sha256"]
            ),
            "n_defined": int(payload["n_defined"]),
            "n_undefined_theta_fallback": int(trial_set.n_trials - payload["n_defined"]),
            "hyperparameters": {},
        }
        rebuilt[row] = np.asarray(estimate.theta, dtype=np.float64)
    # R2 (U2): the frozen learned row under the sealed selected hyperparameters.
    chosen = selection["rows"]["R2"]["selected"]
    hyperparameters = {
        "embedding_dim": int(chosen["embedding_dim"]),
        "temperature": float(chosen["temperature"]),
        "authority": "results/ac3_action_continuity_v0/selection.json rows.R2.selected",
    }
    learned = mtx.learned_row_estimates(
        trial_set, table, row="R2",
        embedding_dim=hyperparameters["embedding_dim"],
        temperature=hyperparameters["temperature"],
    )
    payload = learned["estimate"].payload()
    sealed = screen["rows"]["R2"]["estimate_digest"]
    rows["R2"] = {
        "source": "matrix.learned_row_estimates(trial_set, table, row='R2', ...)",
        "theta_sha256": payload["theta_sha256"],
        "sealed_theta_sha256": sealed["theta_sha256"],
        "theta_digest_matches_sealed_screen": payload["theta_sha256"] == sealed["theta_sha256"],
        "credibility_sha256": payload["credibility_sha256"],
        "sealed_credibility_sha256": sealed["credibility_sha256"],
        "credibility_digest_matches_sealed_screen": (
            payload["credibility_sha256"] == sealed["credibility_sha256"]
        ),
        "n_defined": int(payload["n_defined"]),
        "n_undefined_theta_fallback": int(trial_set.n_trials - payload["n_defined"]),
        "hyperparameters": hyperparameters,
        "fold_receipt_held_out_sessions": [
            str(item["held_out_session"]) for item in learned["receipts"]
        ],
    }
    rebuilt["R2"] = np.asarray(learned["estimate"].theta, dtype=np.float64)

    mismatched = sorted(
        row for row, item in rows.items()
        if not (item["theta_digest_matches_sealed_screen"] and item["credibility_digest_matches_sealed_screen"])
    )
    _require(
        not mismatched,
        f"rebuilt direction estimates do not reproduce the sealed screen digests: {mismatched}",
    )

    # Per-row theta arrays travel to the replay stage inside the receipt itself
    # (JSON floats round-trip exactly through repr); the array digest is
    # re-verified on load so the replay stage cannot consume a mutated vector.
    arrays: dict[str, dict[str, object]] = {}
    for row, values in rebuilt.items():
        arrays[row] = {
            "array_digest": mtx.array_digest(values),
            "theta": [float(item) for item in values.tolist()],
        }

    return {
        "schema": f"{plan.SCHEMA}_directions_v1",
        "stage": "direction_rebuild_and_digest_anchors",
        "attempt_sha256": attempt_sha256,
        "predecessors": predecessors,
        "input_lock_sha256": input_lock["input_sha256"],
        "input_lock": {
            "array_digests_matched": input_lock["array_digests_matched"],
            "trial_membership_sha256": input_lock["trial_membership_sha256"],
            "n_trials": input_lock["n_trials"],
            "sessions": list(input_lock["sessions"]),
        },
        "rows": rows,
        "theta_arrays": arrays,
        "session_binding": binding,
        "n_trials": int(trial_set.n_trials),
        "cpu_only": True,
        "cuda_initialized": False,
        "model_checkpoint_opened": False,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }


def load_theta_arrays(payload: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Reload the receipt's theta vectors and re-verify their array digests."""
    result: dict[str, np.ndarray] = {}
    for row, item in payload["theta_arrays"].items():
        values = np.asarray([float(value) for value in item["theta"]], dtype=np.float64)
        _require(
            mtx.array_digest(values) == str(item["array_digest"]),
            f"the {row} theta array drifted inside the receipt",
        )
        result[str(row)] = values
    return result
