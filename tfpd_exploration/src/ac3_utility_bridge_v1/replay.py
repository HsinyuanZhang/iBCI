"""AC3-U GPU stage: the frozen P2' coherent replay with the rotation seam.

Work order §5/§6.  The frozen ``src/learned_gate_p2prime_v1/physical.py`` is
reused exactly as the sealed driver used it:

* ``_runtime_for`` builds the reviewed runtime on one bound GPU;
* only the six SOURCE (within) assets are parsed -- the external roster is
  never opened, exactly like the AC3-0 materialization stage;
* ``_rollout_oracle`` with ``horizon_H=5`` runs the coherent greedy oracle for
  U0 (``raw``), UGE (``group_ensemble``) and U2 (``r2_head``);
* the ONLY process-local modification is the seam around
  ``policy.build_construction_predictions`` (see :mod:`.wrapper`) plus an
  instance-attribute capture of ``runtime._house_raw_r2`` that records the
  per-trial RAW governing predictions the frozen row payload already scores
  ``house_raw_r2`` from -- neither edits a frozen file.

Scoring discipline (recorded in the receipt, work order §5):

* GOVERNING utility = matrix R2 of the per-trial RAW predictions
  (``no output filter``), recomputed on the frozen float64 metric path;
* ANCHOR = the frozen row's own ``matrix_r2`` (the causal-EMA a=0.25 filtered
  matrix R2), because that is the field the sealed stage-cop O0 row carries;
  U0 must reproduce it bit-exactly.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.learned_gate_p2prime_v1 import physical as p2physical

from . import directions, gates as gates_module, plan, wrapper


class AC3UReplayError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3UReplayError(message)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_receipt(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    _require(sidecar.exists(), f"missing sidecar: {sidecar}")
    _require(
        sidecar.read_text(encoding="ascii").strip() == f"{digest}  {path.name}",
        f"sidecar drift for {path.name}",
    )
    return digest


class _RawPredictionCapture:
    """Record the per-trial raw predictions the frozen row payload scores."""

    def __init__(self, runtime: p2physical.P2PrimeOracleMatrixRuntime) -> None:
        self._runtime = runtime
        self._original = runtime._house_raw_r2
        self._owned = "_house_raw_r2" in runtime.__dict__
        self.calls = 0
        self.last: Optional[dict[str, Any]] = None

    def install(self) -> None:
        self._runtime._house_raw_r2 = self

    def restore(self) -> None:
        if self._owned:
            self._runtime._house_raw_r2 = self._original
        else:
            self._runtime.__dict__.pop("_house_raw_r2", None)

    def __call__(self, per_trial_raw, targets, masks):
        self.calls += 1
        self.last = {
            "per_trial_raw": [np.asarray(item) for item in per_trial_raw],
            "targets": [np.asarray(item) for item in targets],
            "masks": [np.asarray(item) for item in masks],
        }
        return self._original(per_trial_raw, targets, masks)


def _slim_decision(record: Mapping[str, object]) -> dict[str, object]:
    """The sealed receipt's own oracle-decision projection (``_slim_session``)."""
    return {
        "trial_id": record["trial_id"],
        "proposal_accepted_by_b8": record["proposal_accepted_by_b8"],
        "action": record["action"],
        "u_j": record["u_j"],
        "horizon_trials": record["horizon_trials"],
    }


#: The frozen ``_slim_session`` keep list plus the oracle-level extras.
_ROW_KEEP = (
    "row", "budget", "surface", "session", "n_windows", "matrix_r2", "house_raw_r2",
    "prediction_sha256_raw", "filtered_prediction_sha256", "output_filter",
    "carrier_transitions_committed", "activity_transitions_committed",
    "carrier_rejection_counts", "initial_carrier_sha256", "initial_activity_sha256",
    "final_carrier_sha256", "construction", "oracle_level", "horizon_H",
    "oracle_accept_count", "oracle_decision_count", "mean_u_j_among_proposals",
)


def _sealed_o0_sessions(base: Path) -> dict[str, Mapping[str, object]]:
    stage_cop = _read_json(Path(base).absolute() / plan.P2PRIME_ROOT_RELATIVE / "stage_cop.json")
    cells = stage_cop["matrix"]["m4"]["within"]["O0"]["sessions"]
    return {str(item["session"]): item for item in cells}


def _anchor_report(row: Mapping[str, object], sealed: Mapping[str, object]) -> dict[str, object]:
    fields: dict[str, bool] = {}
    differences: dict[str, object] = {}
    for field in plan.ANCHOR_FIELDS:
        left = row.get(field)
        right = sealed.get(field)
        equal = left == right
        fields[field] = bool(equal)
        if not equal:
            differences[field] = {"replayed": left, "sealed": right}
    reported: dict[str, object] = {}
    for field in plan.ANCHOR_REPORTED_FIELDS:
        left = row.get(field)
        right = sealed.get(field)
        reported[field] = {"replayed": left, "sealed": right, "exact": bool(left == right)}
    replayed = [_slim_decision(item) for item in row["oracle_decisions"]]
    sealed_decisions = [_slim_decision(item) for item in sealed["oracle_decisions"]]
    decisions_equal = replayed == sealed_decisions
    return {
        "label": "U0 vs sealed stage-cop within-M4 O0",
        "fields_compared": list(plan.ANCHOR_FIELDS),
        "field_matches": fields,
        "all_fields_exact": all(fields.values()),
        "reported_fields_not_gated": reported,
        "oracle_decision_records_exact": bool(decisions_equal),
        "oracle_decision_records_compared": len(sealed_decisions),
        "differences": differences if not (all(fields.values()) and decisions_equal) else {},
        "bit_exact": bool(all(fields.values()) and decisions_equal),
    }


def run_replay(
    base: Path,
    *,
    gpu_index: int,
    output_root: Optional[Path] = None,
) -> Mapping[str, object]:
    """Run the 18 frozen oracle rollouts (3 rows x 6 within sessions) at M4."""
    base = Path(base).absolute()
    output = Path(output_root) if output_root is not None else base / plan.RESULT_ROOT_RELATIVE
    _require(not (output / "terminal.json").exists(), "the AC3-U terminal receipt already exists")
    _require((output / "attempt.json").exists(), "reserve the AC3-U attempt first")
    _require((output / "directions.json").exists(), "run the AC3-U directions stage first")
    attempt_sha = _verify_receipt(output / "attempt.json")
    directions_sha = _verify_receipt(output / "directions.json")
    attempt = _read_json(output / "attempt.json")
    directions_payload = _read_json(output / "directions.json")
    _require(
        str(directions_payload.get("attempt_sha256")) == attempt_sha,
        "the directions receipt was produced under a different attempt",
    )

    # Fail closed if any immutable predecessor moved after the attempt.
    predecessors = directions.verify_predecessors(base)
    _require(
        predecessors["files"] == attempt["predecessor_sha256s"],
        "an immutable predecessor drifted between attempt and replay",
    )
    owned = plan.owned_sha256s(base)
    _require(
        owned == attempt["owned_sha256s"],
        "an owned AC3-U module drifted between attempt and replay",
    )

    theta_arrays = directions.load_theta_arrays(directions_payload)
    binding_manifest = directions_payload["session_binding"]
    ac3_root = base / plan.AC3_ROOT_RELATIVE
    with np.load(ac3_root / "trajectories.npz", allow_pickle=False) as data:
        velocity_flat = np.asarray(data["velocity_flat"], dtype=np.float64)
        row_starts = np.asarray(data["row_starts"], dtype=np.int64)
        row_counts = np.asarray(data["row_counts"], dtype=np.int64)

    environment = p2physical.validate_environment(gpu_index=gpu_index)
    started = time.monotonic()
    runtime, meta, identity = p2physical._runtime_for(base, gpu_index=gpu_index)
    try:
        runtime.prepare(identity=identity)
        fixed = v1score.derive_fixed_evaluation_authority(base)
        _require(len(fixed.within) == plan.SOURCE_SESSION_COUNT, "the within roster drifted")
        for asset in fixed.within:
            prepared = runtime._parse_session(asset=asset)
            runtime._require_state().sessions[(asset.surface, asset.session)] = prepared
        state = runtime._require_state()
        _require(
            not runtime._external_opened
            and all(key[0] == plan.SURFACE for key in state.sessions)
            and len(state.sessions) == plan.SOURCE_SESSION_COUNT,
            "AC3-U must parse the source (within) sessions only",
        )

        arm = state.modules["arm_common"]
        model_digest_before = arm.state_sha256(state.model)
        sealed_o0 = _sealed_o0_sessions(base)

        seam = wrapper.build_seam()
        capture = _RawPredictionCapture(runtime)
        specs = {
            row: p2physical.RolloutSpec(
                row_id=row, kind="oracle", construction=plan.ROWS[row]["construction"],
                horizon_H=plan.HORIZON_H,
            )
            for row in plan.ROW_ORDER
        }
        session_rows: dict[str, list[dict[str, object]]] = {row: [] for row in plan.ROW_ORDER}
        seam_records: list[dict[str, object]] = []
        anchors: dict[str, dict[str, object]] = {}
        seam.install()
        capture.install()
        try:
            for row in plan.ROW_ORDER:
                spec = specs[row]
                direction_row = plan.ROWS[row]["direction_row"]
                theta = theta_arrays[direction_row]
                for key in p2physical._ordered_session_keys(runtime, plan.SURFACE):
                    session = runtime._require_state().sessions[key]
                    session_name = str(session.session)
                    block = binding_manifest["sessions"][session_name]
                    query_ids = [str(item) for item in session.query_trial_ids[plan.BUDGET]]
                    _require(
                        query_ids == [str(item) for item in block["trial_ids"]],
                        f"trial-id binding drift for {session_name}: the P2' runtime's "
                        "query_trial_ids[4] is not the materialize.json subsequence",
                    )
                    indices = list(range(int(block["start"]), int(block["stop"])))
                    seam.begin_session(wrapper.SessionBinding(
                        row_id=row, construction=spec.construction, session=session_name,
                        trial_ids=query_ids, theta=theta[indices],
                        frozen_velocity=velocity_flat,
                        row_starts=row_starts[indices], row_counts=row_counts[indices],
                    ))
                    capture.calls = 0
                    capture.last = None
                    rollout = runtime._rollout_oracle(
                        session=session, budget=plan.BUDGET, spec=spec,
                    )
                    audit = seam.end_session()
                    seam_records.append(audit)
                    _require(capture.calls == 1 and capture.last is not None,
                             "the raw-prediction capture did not see exactly one row payload")
                    raw_r2, _joined = runtime._matrix_r2(
                        capture.last["per_trial_raw"], capture.last["targets"],
                        capture.last["masks"],
                    )
                    frozen_row = rollout["rows"][row]
                    # The sealed receipt's own slim projection plus the AC3-U
                    # scoring columns; the full per-trial transition list is not
                    # duplicated into this receipt.
                    entry = {field: frozen_row[field] for field in _ROW_KEEP if field in frozen_row}
                    entry.update({
                        "matrix_r2_filtered_anchor": float(frozen_row["matrix_r2"]),
                        "matrix_r2_raw_governing": float(raw_r2),
                        "governing_scoring": plan.SCORING["governing"]["name"],
                        "anchor_scoring": plan.SCORING["anchor"]["name"],
                        "direction_row": direction_row,
                        "oracle_decision_records": [
                            _slim_decision(item) for item in frozen_row["oracle_decisions"]
                        ],
                        "seam_audit": dict(audit),
                    })
                    session_rows[row].append(entry)
                    if row == "U0":
                        anchors[session_name] = _anchor_report(frozen_row, sealed_o0[session_name])
                        _require(
                            bool(anchors[session_name]["bit_exact"]),
                            f"the U0 anchor is not bit-exact for {session_name}: "
                            f"{anchors[session_name]['differences']}",
                        )
                    elapsed = time.monotonic() - started
                    _require(
                        elapsed <= plan.HARD_TIMEOUT_SECONDS,
                        f"the AC3-U hard timeout fired after {elapsed:.0f} s",
                    )
        finally:
            capture.restore()
            seam.restore()

        model_digest_after = arm.state_sha256(runtime._require_state().model)
        _require(
            model_digest_before == model_digest_after,
            "the sealed Cell-D model state mutated during the AC3-U replay",
        )
        resources = dict(runtime._resources())
    finally:
        runtime.close()

    governing = {
        row: {
            str(item["session"]): float(item["matrix_r2_raw_governing"])
            for item in session_rows[row]
        }
        for row in plan.ROW_ORDER
    }
    anchor_domain = {
        row: {
            str(item["session"]): float(item["matrix_r2_filtered_anchor"])
            for item in session_rows[row]
        }
        for row in plan.ROW_ORDER
    }
    payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "coherent_replay_with_rotation_seam",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_sha,
        "directions_sha256": directions_sha,
        "identity_sha256": meta["identity_sha256"],
        "environment": environment,
        "gpu_index": int(gpu_index),
        "budget": plan.BUDGET,
        "surface": plan.SURFACE,
        "horizon_H": plan.HORIZON_H,
        "sessions": list(plan.SESSIONS),
        "external_roster_opened": False,
        "rows": {
            row: {
                "construction": plan.ROWS[row]["construction"],
                "direction_row": plan.ROWS[row]["direction_row"],
                "sessions": session_rows[row],
                "governing_raw_matrix_r2": governing[row],
                "anchor_filtered_matrix_r2": anchor_domain[row],
            }
            for row in plan.ROW_ORDER
        },
        "scoring": {
            "governing": dict(plan.SCORING["governing"]),
            "anchor": dict(plan.SCORING["anchor"]),
            "note": plan.SCORING["note"],
        },
        "u0_anchor_vs_sealed_stage_cop": anchors,
        "u0_anchor_all_sessions_bit_exact": all(
            bool(item["bit_exact"]) for item in anchors.values()
        ),
        "seam": {
            "target": plan.SEAM["target"],
            "mechanism": plan.SEAM["mechanism"],
            "rotating_branch": plan.SEAM["rotating_branch"],
            "passthrough_branch": plan.SEAM["passthrough_branch"],
            "wrapper_source_sha256": owned["tfpd_exploration/src/ac3_utility_bridge_v1/wrapper.py"],
            "rotation_source_sha256": owned["tfpd_exploration/src/ac3_utility_bridge_v1/rotation.py"],
            "total_passthrough_calls": int(seam.passthrough_calls),
            "total_rotated_calls": int(seam.rotated_calls),
            "session_audits": seam_records,
        },
        "model_state_digest_before_sha256": model_digest_before,
        "model_state_digest_after_sha256": model_digest_after,
        "model_state_digest_unchanged": model_digest_before == model_digest_after,
        "resources": resources,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "model_or_checkpoint_updated": False,
        "target_optimizer_backward_update": 0,
    }
    _require(
        payload["u0_anchor_all_sessions_bit_exact"],
        "the U0 anchor failed at least one within session",
    )
    return payload


def build_terminal(*, replay_payload: Mapping[str, object]) -> Mapping[str, object]:
    """Evaluate the §8 gates on the replay receipt and assemble the terminal body."""
    rows = replay_payload["rows"]
    sessions = [str(item) for item in replay_payload["sessions"]]
    governing = {
        row: {str(key): float(value) for key, value in rows[row]["governing_raw_matrix_r2"].items()}
        for row in plan.ROW_ORDER
    }
    anchor_domain = {
        row: {str(key): float(value) for key, value in rows[row]["anchor_filtered_matrix_r2"].items()}
        for row in plan.ROW_ORDER
    }
    _require(
        all(tuple(governing[row]) == tuple(sessions) for row in plan.ROW_ORDER),
        "the replay session roster drift",
    )
    gates = gates_module.evaluate_gates(governing_r2=governing, sessions=sessions)
    anchor_gates = gates_module.evaluate_gates(
        governing_r2=anchor_domain, sessions=sessions,
    )
    return {
        "gates": gates,
        "anchor_domain_reference_non_governing": {
            "non_governing": True,
            "scoring": plan.SCORING["anchor"]["name"],
            "reason": (
                "the frozen oracle row payload's own filtered matrix R2 -- the "
                "domain the sealed stage-cop O0/O1/O2 numbers are reported in; "
                "kept for context only, the work order §5 raw scoring governs"
            ),
            "equal_session_mean_r2": anchor_gates["equal_session_mean_r2"],
            "UGE_minus_U0": anchor_gates["UGE_minus_U0"],
            "U2_minus_UGE": anchor_gates["U2_minus_UGE"],
            "disposition_if_it_governed": anchor_gates["disposition"],
        },
        "per_session_rows": {
            row: [
                {
                    "session": str(item["session"]),
                    "matrix_r2_raw_governing": float(item["matrix_r2_raw_governing"]),
                    "matrix_r2_filtered_anchor": float(item["matrix_r2_filtered_anchor"]),
                    "house_raw_r2": float(item["house_raw_r2"]),
                    "prediction_sha256_raw": str(item["prediction_sha256_raw"]),
                    "oracle_accept_count": int(item["oracle_accept_count"]),
                    "oracle_decision_count": int(item["oracle_decision_count"]),
                    "carrier_transitions_committed": int(item["carrier_transitions_committed"]),
                }
                for item in rows[row]["sessions"]
            ]
            for row in plan.ROW_ORDER
        },
    }
