"""Explicitly test-only signing helpers for Phase-C capability fixtures.

Production callers must use verify_signed_authorization and its fixed project
anchor. This module is deliberately located under tests so generated keys
cannot become a production API surface.
"""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
from typing import Any, Mapping

from sua_exploration.mc_maze import m2_native_post33_authorization_v4 as production
from sua_exploration.mc_maze import m2_native_post33_openers_v4 as openers
from sua_exploration.mc_maze import m2_native_post33_phase_c_v4 as contract


_SYNTHETIC_FIXTURE_ROOTS: set[Path] = set()


def enable_synthetic_fixture_root(root: str | Path) -> Path:
    """Create the O_EXCL test sentinel required by relaxed fixture helpers."""
    # Validate before the sentinel write.  In particular, a test that points
    # this helper at a workspace path must not be able to create even the
    # tests-only marker there.
    root_path = contract._validate_synthetic_fixture_root_location(root)
    if root_path not in _SYNTHETIC_FIXTURE_ROOTS:
        sentinel = root_path / contract._SYNTHETIC_TEST_SENTINEL
        payload = {
            "schema": "m2_post33_phase_c_v4_synthetic_fixture_sentinel",
            "absolute_fixture_root": str(root_path),
        }
        if sentinel.exists():
            if json.loads(sentinel.read_text(encoding="utf-8")) != payload:
                raise PermissionError("synthetic fixture sentinel already exists with wrong content")
        else:
            contract.write_json_exclusive(sentinel, payload)
        contract._activate_synthetic_test_fixture(root_path)
        _SYNTHETIC_FIXTURE_ROOTS.add(root_path)
    return root_path


def finalize_cell_score_sealed_with_synthetic(
    *,
    root: str | Path,
    key: contract.CellKey,
    owner_token: str,
    score_commitment_path: str | Path,
    opaque_payload_path: str | Path,
    global_cost_receipt_path: str | Path,
    cost_supplement_path: str | Path,
    source_cost_evidence_path: str | Path,
    deployment_cost_evidence_path: str | Path,
    decoder_lifecycle_path: str | Path | None = None,
    paired_spint_completion_path: str | Path | None = None,
    outer_runtime_evidence_path: str | Path | None = None,
) -> Path:
    enable_synthetic_fixture_root(root)
    return contract._finalize_cell_score_sealed(
        root=root,
        key=key,
        owner_token=owner_token,
        score_commitment_path=score_commitment_path,
        opaque_payload_path=opaque_payload_path,
        global_cost_receipt_path=global_cost_receipt_path,
        cost_supplement_path=cost_supplement_path,
        source_cost_evidence_path=source_cost_evidence_path,
        deployment_cost_evidence_path=deployment_cost_evidence_path,
        decoder_lifecycle_path=decoder_lifecycle_path,
        paired_spint_completion_path=paired_spint_completion_path,
        outer_runtime_evidence_path=outer_runtime_evidence_path,
        allow_synthetic_decoder_evidence=True,
    )


def verify_cell_exact_with_synthetic(
    root: str | Path, key: contract.CellKey
) -> dict[str, Any]:
    enable_synthetic_fixture_root(root)
    return contract._verify_cell_exact(root, key, allow_synthetic=True)


def finalize_stage_a_score_sealed_with_synthetic(root: str | Path) -> Path:
    enable_synthetic_fixture_root(root)
    return contract._finalize_stage_a_score_sealed(root, allow_synthetic=True)


def verify_stage_a_exact_with_synthetic(root: str | Path) -> dict[str, Any]:
    enable_synthetic_fixture_root(root)
    return contract._verify_stage_a_exact(root, allow_synthetic=True)


def finalize_matrix_score_sealed_with_synthetic(root: str | Path) -> Path:
    enable_synthetic_fixture_root(root)
    return contract._finalize_matrix_score_sealed(root, allow_synthetic=True)


def verify_matrix_exact_with_synthetic(root: str | Path) -> dict[str, Any]:
    enable_synthetic_fixture_root(root)
    return contract._verify_matrix_exact(root, allow_synthetic=True)


def validate_stage_a_decision_with_synthetic(
    root: str | Path,
    *,
    require_continue: bool,
    require_signature: bool = False,
) -> Mapping[str, Any]:
    enable_synthetic_fixture_root(root)
    return openers._validate_stage_a_decision(
        root,
        require_continue=require_continue,
        allow_synthetic=True,
        require_signature=require_signature,
    )


def write_decoder_lifecycle_stage_with_synthetic(
    module: Any,
    *,
    fixture_root: str | Path,
    stage_dir: str | Path,
    stage: str,
    decoder: Any,
    optimizer: Any | None,
    cell_identity: Mapping[str, Any],
) -> Path:
    """Exercise the quarantined lifecycle stage writer in a disposable root."""
    root = enable_synthetic_fixture_root(fixture_root)
    return module._write_stage_exclusive(
        stage_dir,
        stage=stage,
        decoder=decoder,
        optimizer=optimizer,
        cell_identity=cell_identity,
        synthetic_proof=True,
        synthetic_fixture_root=root,
    )


def finalize_decoder_lifecycle_evidence_with_synthetic(
    module: Any,
    *,
    fixture_root: str | Path,
    stage_dir: str | Path,
    output_path: str | Path,
    cell_identity: Mapping[str, Any],
) -> Path:
    """Exercise the quarantined lifecycle finalizer in a disposable root."""
    root = enable_synthetic_fixture_root(fixture_root)
    return module._finalize_lifecycle_evidence(
        stage_dir,
        output_path,
        cell_identity=cell_identity,
        allow_synthetic=True,
        synthetic_fixture_root=root,
    )


def verify_signed_authorization_with_test_anchor(
    authorization_path: str | Path,
    signature_path: str | Path,
    *,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cell_root: str | Path,
    cost_supplement_path: str | Path | None = None,
    now: Any = None,
    public_key_path: str | Path,
    expected_public_key_sha256: str,
) -> Mapping[str, Any]:
    """Test-only verifier; production code has no caller-selectable anchor."""
    return production._verify_signed_authorization_with_anchor(
        authorization_path,
        signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cell_root=cell_root,
        cost_supplement_path=cost_supplement_path,
        now=now,
        public_key_path=public_key_path,
        expected_public_key_sha256=expected_public_key_sha256,
    )


def consume_opening_authorization_nonce_with_test_anchor(
    *,
    root: str | Path,
    opening_stage: str,
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
    public_key_path: str | Path,
    expected_public_key_sha256: str,
) -> tuple[Mapping[str, Any], Path]:
    """Exercise the claim writer with a fixture-only detached trust anchor."""
    authorization = verify_signed_authorization_with_test_anchor(
        authorization_path,
        signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
        cell_root=root,
        public_key_path=public_key_path,
        expected_public_key_sha256=expected_public_key_sha256,
    )
    production._validate_opening_scope(authorization, opening_stage=opening_stage)
    return authorization, production._write_opening_authorization_claim(
        root=root,
        opening_stage=opening_stage,
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
    )


def _revalidate_bound_claim_with_test_anchor(
    root: str | Path,
    claim_path: str | Path,
    *,
    public_key_path: str | Path,
    expected_public_key_sha256: str,
) -> Mapping[str, Any]:
    claim_file = contract.require_canonical_regular_file(
        claim_path, within=contract.phase_root(root)
    )
    claim = json.loads(claim_file.read_text(encoding="utf-8"))
    bound = {
        field: contract.require_canonical_regular_file(
            claim[field]["canonical_path"]
        )
        for field in ("authorization", "signature", "shard_manifest")
    }
    envelope = json.loads(bound["authorization"].read_text(encoding="utf-8"))
    auth = envelope["authorization"]
    issued = production._parse_utc(auth["issued_at"], "issued_at")
    validated = verify_signed_authorization_with_test_anchor(
        bound["authorization"],
        bound["signature"],
        phase_c_program_receipt_path=auth["phase_c_program_receipt"]["canonical_path"],
        portable_manifest_path=auth["portable_manifest"]["canonical_path"],
        shard_manifest_path=bound["shard_manifest"],
        cell_root=root,
        cost_supplement_path=auth["cost_supplement"]["canonical_path"],
        now=issued + timedelta(seconds=1),
        public_key_path=public_key_path,
        expected_public_key_sha256=expected_public_key_sha256,
    )
    production.verify_authorization_nonce_claim(
        root=root,
        authorization=validated,
        authorization_path=bound["authorization"],
        signature_path=bound["signature"],
        shard_manifest_path=bound["shard_manifest"],
    )
    return validated


def validate_claim_coverage_with_test_anchor(
    root: str | Path,
    *,
    stage: str,
    public_key_path: str | Path,
    expected_public_key_sha256: str,
) -> list[Mapping[str, Any]]:
    claim_root = contract.phase_root(root) / "authorization_claims"
    if not claim_root.is_dir():
        raise PermissionError("signed authorization claims are missing")
    validated: list[Mapping[str, Any]] = []
    coverage: list[tuple[int, int]] = []
    for claim_path in sorted(claim_root.glob("*/*.json")):
        authorization = _revalidate_bound_claim_with_test_anchor(
            root,
            claim_path,
            public_key_path=public_key_path,
            expected_public_key_sha256=expected_public_key_sha256,
        )
        if authorization["stage"] != stage:
            continue
        validated.append(authorization)
        coverage.extend(
            (fold, seed)
            for fold in authorization["fold_allowlist"]
            for seed in authorization["seed_allowlist"]
        )
    production.validate_coverage_pairs(stage=stage, coverage=coverage)
    return validated


def open_stage_a_with_test_anchor(
    root: str | Path,
    *,
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
    public_key_path: str | Path,
    expected_public_key_sha256: str,
    allow_synthetic: bool,
) -> Path:
    if allow_synthetic:
        enable_synthetic_fixture_root(root)
    contract.validate_stage_a_cell_directory_set(root)
    report = contract._verify_stage_a_exact(root, allow_synthetic=allow_synthetic)
    paths = contract.stage_a_paths(root)
    if report["decision_present"] or paths["decision"].exists():
        raise FileExistsError("Stage-A score opening is write-once")
    authorization = verify_signed_authorization_with_test_anchor(
        authorization_path,
        signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
        cell_root=root,
        public_key_path=public_key_path,
        expected_public_key_sha256=expected_public_key_sha256,
    )
    production._validate_opening_scope(authorization, opening_stage="stage_a_opening")
    claim = production._write_opening_authorization_claim(
        root=root,
        opening_stage="stage_a_opening",
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
    )
    return openers._open_stage_a_after_consumed_capability(
        root,
        authorization_path=authorization_path,
        signature_path=signature_path,
        opening_claim=claim,
        allow_synthetic=allow_synthetic,
    )


def open_full_matrix_with_test_anchor(
    root: str | Path,
    *,
    authorization_path: str | Path,
    signature_path: str | Path,
    phase_c_program_receipt_path: str | Path,
    portable_manifest_path: str | Path,
    shard_manifest_path: str | Path,
    cost_supplement_path: str | Path,
    public_key_path: str | Path,
    expected_public_key_sha256: str,
    allow_synthetic: bool,
) -> Path:
    if allow_synthetic:
        enable_synthetic_fixture_root(root)
    report = contract._verify_matrix_exact(root, allow_synthetic=allow_synthetic)
    paths = contract.matrix_paths(root)
    if report["opened_aggregate_present"] or paths["opened_aggregate"].exists():
        raise FileExistsError("full score opening is write-once")
    openers._validate_stage_a_decision(
        root, require_continue=True, allow_synthetic=allow_synthetic
    )
    authorization = verify_signed_authorization_with_test_anchor(
        authorization_path,
        signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
        cell_root=root,
        public_key_path=public_key_path,
        expected_public_key_sha256=expected_public_key_sha256,
    )
    production._validate_opening_scope(authorization, opening_stage="full_opening")
    claim = production._write_opening_authorization_claim(
        root=root,
        opening_stage="full_opening",
        authorization=authorization,
        authorization_path=authorization_path,
        signature_path=signature_path,
        phase_c_program_receipt_path=phase_c_program_receipt_path,
        portable_manifest_path=portable_manifest_path,
        shard_manifest_path=shard_manifest_path,
        cost_supplement_path=cost_supplement_path,
    )
    return openers._open_full_after_consumed_capability(
        root,
        authorization_path=authorization_path,
        signature_path=signature_path,
        opening_claim=claim,
        allow_synthetic=allow_synthetic,
    )
