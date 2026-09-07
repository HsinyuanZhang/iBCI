"""Delayed score openers for exact-14 Stage A and exact-42 full Phase-C."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from sua_exploration.mc_maze.m2_native_post33_authorization_v4 import (
    PUBLIC_KEY,
    PUBLIC_KEY_SHA256,
    consume_opening_authorization_nonce,
)
from sua_exploration.mc_maze.m2_native_post33_evaluator_v4 import validate_endpoint_payload
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    ARMS,
    FOLDS,
    PHASE_ID,
    PROTOCOL_ID,
    SEEDS,
    CellKey,
    _require_synthetic_test_fixture,
    _verify_cell_exact,
    _verify_matrix_exact,
    _verify_stage_a_exact,
    cell_paths,
    file_metadata,
    matrix_paths,
    stage_a_paths,
    validate_stage_a_cell_directory_set,
    validate_selector_payload,
    write_json_exclusive,
)


BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_RNG_SEED = 20_260_804
BOOTSTRAP_LOWER_QUANTILE = 0.025


def stage_a_decision_from_deltas(deltas: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Stage-A requires exactly seven finite paired deltas")
    mean42 = float(values.mean())
    pos42 = int(np.sum(values > 0))
    severe = bool(mean42 <= -0.03 or pos42 <= 1)
    return {
        "mean42": mean42,
        "pos42": pos42,
        "severe_negative_rule": "(mean42 <= -0.03) OR (pos42 <= 1)",
        "severe_negative_triggered": severe,
        "decision": (
            "seed42_severe_negative_futility_stop"
            if severe else "continue_without_positive_claim"
        ),
        "positive_claim_made": False,
    }


def _read_score(root: str | Path, key: CellKey, *, allow_synthetic: bool) -> float:
    # Keep the lowest-level delayed-score reader protected as well: importing
    # this private helper directly must not provide a synthetic-score opening
    # bypass for a production-like root.
    if allow_synthetic:
        _require_synthetic_test_fixture(root)
    _verify_cell_exact(root, key, allow_synthetic=allow_synthetic)
    paths = cell_paths(root, key)
    selector = json.loads(paths["selector_records"].read_text(encoding="utf-8"))
    selected = validate_selector_payload(selector, key, run_dir=paths["run"])
    payload = json.loads(paths["opaque_payload"].read_text(encoding="utf-8"))
    return validate_endpoint_payload(
        payload,
        key=key,
        selected_checkpoint=selected["checkpoint_path"],
        resolved_config=paths["resolved_config"],
    )


def open_stage_a(
    root: str | Path,
    *,
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
) -> Path:
    # Stage B must not exist when the first fourteen values are exposed.
    validate_stage_a_cell_directory_set(root)
    report = _verify_stage_a_exact(root, allow_synthetic=False)
    paths = stage_a_paths(root)
    if report["decision_present"] or paths["decision"].exists():
        raise FileExistsError("Stage-A score opening is write-once")
    # Do not accept a caller-injected "validated" mapping. The production core
    # verifies the raw detached signature and consumes its own opening nonce
    # before it reads the first opaque endpoint payload.
    _, opening_claim = consume_opening_authorization_nonce(
        root=root,
        opening_stage="stage_a_opening",
        authorization_path=authorization_path,
        signature_path=signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
    )
    return _open_stage_a_after_consumed_capability(
        root,
        authorization_path=authorization_path,
        signature_path=signature_path,
        opening_claim=opening_claim,
        allow_synthetic=False,
    )


def _open_stage_a_after_consumed_capability(
    root: str | Path,
    *,
    authorization_path: str | Path,
    signature_path: str | Path,
    opening_claim: str | Path,
    allow_synthetic: bool,
) -> Path:
    """Write the decision after the production or test-only gate consumed a nonce."""
    if allow_synthetic:
        _require_synthetic_test_fixture(root)
    paths = stage_a_paths(root)
    rows = []
    for fold, session in FOLDS.items():
        spint = _read_score(
            root, CellKey(PROTOCOL_ID, "spint", fold, 42),
            allow_synthetic=allow_synthetic,
        )
        t4 = _read_score(
            root, CellKey(PROTOCOL_ID, "t4", fold, 42),
            allow_synthetic=allow_synthetic,
        )
        rows.append(
            {"fold": fold, "outer_session": session, "spint_r2": spint, "t4_r2": t4, "delta": t4 - spint}
        )
    decision = stage_a_decision_from_deltas([row["delta"] for row in rows])
    payload = {
        "schema": "m2_post33_phase_c_opened_stage_a_decision_v4",
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "absolute_cell_root": str(Path(root).resolve()),
        "stage_a_manifest": file_metadata(paths["manifest"]),
        "opening_authorization": file_metadata(authorization_path),
        "opening_authorization_signature": file_metadata(signature_path),
        "opening_nonce_claim": file_metadata(opening_claim),
        "cell_count": 14,
        "pair_count": 7,
        "seed": 42,
        "paired_rows": rows,
        **decision,
    }
    return write_json_exclusive(paths["decision"], payload)


def _validate_stage_a_decision(
    root: str | Path,
    *,
    require_continue: bool,
    allow_synthetic: bool = False,
    require_signature: bool | None = None,
) -> Mapping[str, Any]:
    if allow_synthetic:
        _require_synthetic_test_fixture(root)
    report = _verify_stage_a_exact(root, allow_synthetic=allow_synthetic)
    if report["decision_present"] is not True:
        raise PermissionError("Stage-B/full opening requires a Stage-A decision")
    path = stage_a_paths(root)["decision"]
    decision = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "protocol_id", "phase_id", "absolute_cell_root", "stage_a_manifest",
        "opening_authorization", "opening_authorization_signature", "opening_nonce_claim",
        "cell_count", "pair_count", "seed", "paired_rows", "mean42", "pos42",
        "severe_negative_rule", "severe_negative_triggered", "decision", "positive_claim_made",
    }
    if set(decision) != required or (
        decision.get("schema") != "m2_post33_phase_c_opened_stage_a_decision_v4"
        or decision.get("protocol_id") != PROTOCOL_ID
        or decision.get("phase_id") != PHASE_ID
        or decision.get("absolute_cell_root") != str(Path(root).resolve())
        or decision.get("cell_count") != 14
        or decision.get("pair_count") != 7
        or decision.get("seed") != 42
        or decision.get("positive_claim_made") is not False
    ):
        raise ValueError("Stage-A decision schema/root/cardinality mismatch")
    if decision.get("stage_a_manifest") != file_metadata(stage_a_paths(root)["manifest"]):
        raise ValueError("Stage-A decision manifest substitution")
    opening_metadata = {
        "opening_authorization": decision.get("opening_authorization"),
        "opening_authorization_signature": decision.get("opening_authorization_signature"),
        "opening_nonce_claim": decision.get("opening_nonce_claim"),
    }
    opening_files: dict[str, Path] = {}
    for field, metadata in opening_metadata.items():
        if not isinstance(metadata, Mapping):
            raise ValueError("Stage-A decision opening capability metadata missing")
        opening_file = Path(str(metadata.get("canonical_path", "")))
        if metadata != file_metadata(opening_file):
            raise ValueError("Stage-A decision opening capability substitution")
        opening_files[field] = opening_file
    claim = json.loads(opening_files["opening_nonce_claim"].read_text(encoding="utf-8"))
    if (
        claim.get("schema") != "m2_post33_phase_c_opening_nonce_claim_v4"
        or claim.get("opening_stage") != "stage_a_opening"
        or claim.get("absolute_cell_root") != str(Path(root).resolve())
        or claim.get("authorization") != opening_metadata["opening_authorization"]
        or claim.get("signature") != opening_metadata["opening_authorization_signature"]
    ):
        raise ValueError("Stage-A decision opening nonce-claim substitution")
    signature_required = (not allow_synthetic) if require_signature is None else require_signature
    if signature_required:
        if report.get("decision_signature_present") is not True:
            raise PermissionError("Stage-A decision lacks detached root signature")
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import sha256_file
        public = PUBLIC_KEY.resolve(strict=True)
        if sha256_file(public) != PUBLIC_KEY_SHA256:
            raise PermissionError("Stage-A decision trust-anchor hash mismatch")
        key = serialization.load_pem_public_key(public.read_bytes())
        if not isinstance(key, Ed25519PublicKey):
            raise TypeError("Stage-A decision trust anchor is not Ed25519")
        try:
            signature = base64.b64decode(
                stage_a_paths(root)["decision_signature"].read_bytes(), validate=True
            )
            key.verify(signature, path.read_bytes())
        except (InvalidSignature, ValueError) as exc:
            raise PermissionError("invalid detached Ed25519 Stage-A decision signature") from exc
    rows = decision.get("paired_rows")
    if not isinstance(rows, list) or len(rows) != 7:
        raise ValueError("Stage-A decision paired row cardinality mismatch")
    observed = []
    for fold, session in FOLDS.items():
        row = rows[fold]
        if set(row) != {"fold", "outer_session", "spint_r2", "t4_r2", "delta"}:
            raise ValueError("Stage-A paired row exact key set mismatch")
        if row.get("fold") != fold or row.get("outer_session") != session:
            raise ValueError("Stage-A paired row identity/order substitution")
        spint = _read_score(root, CellKey(PROTOCOL_ID, "spint", fold, 42), allow_synthetic=allow_synthetic)
        t4 = _read_score(root, CellKey(PROTOCOL_ID, "t4", fold, 42), allow_synthetic=allow_synthetic)
        if row.get("spint_r2") != spint or row.get("t4_r2") != t4 or row.get("delta") != t4 - spint:
            raise ValueError("Stage-A decision score substitution")
        observed.append(t4 - spint)
    expected = stage_a_decision_from_deltas(observed)
    for field, value in expected.items():
        if decision.get(field) != value:
            raise ValueError("Stage-A decision rule/output substitution")
    if require_continue and decision["decision"] != "continue_without_positive_claim":
        raise PermissionError("Stage-A severe-negative stop forbids Stage B/full opening")
    return decision


def validate_stage_a_decision(
    root: str | Path,
    *,
    require_continue: bool,
) -> Mapping[str, Any]:
    """Production decision verifier: signed, non-synthetic, and fixed-policy."""
    return _validate_stage_a_decision(
        root,
        require_continue=require_continue,
        allow_synthetic=False,
        require_signature=True,
    )


def compute_full_gates(delta_matrix: np.ndarray, spint: np.ndarray, t4: np.ndarray) -> dict[str, Any]:
    deltas = np.asarray(delta_matrix, dtype=np.float64)
    spint_scores = np.asarray(spint, dtype=np.float64)
    t4_scores = np.asarray(t4, dtype=np.float64)
    if deltas.shape != (3, 7) or spint_scores.shape != (3, 7) or t4_scores.shape != (3, 7):
        raise ValueError("full opener requires exact 3-seed x 7-session matrices")
    if not (np.isfinite(deltas).all() and np.isfinite(spint_scores).all() and np.isfinite(t4_scores).all()):
        raise ValueError("full opener matrices must be finite")
    seed_means = deltas.mean(axis=1)
    session_means = deltas.mean(axis=0)
    mean_delta = float(deltas.mean())
    two_se = float(mean_delta - 2 * seed_means.std(ddof=1) / math.sqrt(3))
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    seed_indices = rng.integers(0, 3, size=(BOOTSTRAP_REPLICATES, 3))
    session_indices = rng.integers(0, 7, size=(BOOTSTRAP_REPLICATES, 7))
    samples = deltas[seed_indices[:, :, None], session_indices[:, None, :]].mean(axis=(1, 2))
    bootstrap_lower = float(np.quantile(samples, BOOTSTRAP_LOWER_QUANTILE, method="linear"))
    absolute_spint = float(spint_scores.mean())
    absolute_t4 = float(t4_scores.mean())
    gates = {
        "mean_delta": {"value": mean_delta, "threshold": 0.03, "pass": mean_delta >= 0.03},
        "seed_sign": {"values": seed_means.tolist(), "pass": bool(np.all(seed_means > 0))},
        "session_sign": {
            "values": session_means.tolist(), "positive_count": int(np.sum(session_means > 0)),
            "pass": int(np.sum(session_means > 0)) >= 6,
        },
        "paired_seed_two_se": {"lower": two_se, "pass": two_se > 0},
        "two_way_bootstrap": {
            "lower": bootstrap_lower, "quantile": BOOTSTRAP_LOWER_QUANTILE,
            "replicates": BOOTSTRAP_REPLICATES, "rng": "numpy.PCG64",
            "rng_seed": BOOTSTRAP_RNG_SEED, "pass": bootstrap_lower > 0,
        },
        "absolute_finite_positive": {
            "spint_equal_session_mean": absolute_spint,
            "t4_equal_session_mean": absolute_t4,
            "pass": math.isfinite(absolute_spint) and math.isfinite(absolute_t4) and absolute_t4 > 0,
        },
    }
    return {
        "seed_means": seed_means.tolist(),
        "session_means": session_means.tolist(),
        "gates": gates,
        "all_six_pass": all(bool(gate["pass"]) for gate in gates.values()),
    }


def open_full_matrix(
    root: str | Path,
    *,
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
) -> Path:
    matrix_report = _verify_matrix_exact(root, allow_synthetic=False)
    if matrix_report["opened_aggregate_present"] or matrix_paths(root)["opened_aggregate"].exists():
        raise FileExistsError("full score opening is write-once")
    _validate_stage_a_decision(root, require_continue=True, allow_synthetic=False)
    _, opening_claim = consume_opening_authorization_nonce(
        root=root,
        opening_stage="full_opening",
        authorization_path=authorization_path,
        signature_path=signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
    )
    return _open_full_after_consumed_capability(
        root,
        authorization_path=authorization_path,
        signature_path=signature_path,
        opening_claim=opening_claim,
        allow_synthetic=False,
    )


def _open_full_after_consumed_capability(
    root: str | Path,
    *,
    authorization_path: str | Path,
    signature_path: str | Path,
    opening_claim: str | Path,
    allow_synthetic: bool,
) -> Path:
    """Write the full aggregate after the production or test-only gate consumed a nonce."""
    if allow_synthetic:
        _require_synthetic_test_fixture(root)
    spint = np.empty((3, 7), dtype=np.float64)
    t4 = np.empty((3, 7), dtype=np.float64)
    rows = []
    for seed_index, seed in enumerate(SEEDS):
        for fold, session in FOLDS.items():
            spint_value = _read_score(
                root, CellKey(PROTOCOL_ID, "spint", fold, seed), allow_synthetic=allow_synthetic
            )
            t4_value = _read_score(
                root, CellKey(PROTOCOL_ID, "t4", fold, seed), allow_synthetic=allow_synthetic
            )
            spint[seed_index, fold] = spint_value
            t4[seed_index, fold] = t4_value
            rows.append(
                {
                    "seed": seed, "fold": fold, "outer_session": session,
                    "spint_r2": spint_value, "t4_r2": t4_value,
                    "delta": t4_value - spint_value,
                }
            )
    summary = compute_full_gates(t4 - spint, spint, t4)
    payload = {
        "schema": "m2_post33_phase_c_opened_full_aggregate_v4",
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "absolute_cell_root": str(Path(root).resolve()),
        "matrix_manifest": file_metadata(matrix_paths(root)["manifest"]),
        "stage_a_decision": file_metadata(stage_a_paths(root)["decision"]),
        "opening_authorization": file_metadata(authorization_path),
        "opening_authorization_signature": file_metadata(signature_path),
        "opening_nonce_claim": file_metadata(opening_claim),
        "cell_count": 42,
        "pair_count": 21,
        "seed_by_session_rows": rows,
        **summary,
    }
    return write_json_exclusive(matrix_paths(root)["opened_aggregate"], payload)
