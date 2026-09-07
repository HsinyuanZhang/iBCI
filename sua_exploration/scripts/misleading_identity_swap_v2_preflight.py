#!/usr/bin/env python3
"""CPU-only non-authorizing preflight scaffold for misleading-identity swap-v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for entry in (REPO_ROOT, SUA_ROOT, REPO_ROOT / "streaming_calibration_exp"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from mc_maze import misleading_identity_swap_v2_core as core


IMPLEMENTATION_PATHS = {
    "config": core.CONFIG_PATH,
    "contract": core.CONTRACT_PATH,
    "primitive": SUA_ROOT / "mc_maze" / "misleading_identity_swap.py",
    "core": SUA_ROOT / "mc_maze" / "misleading_identity_swap_v2_core.py",
    "trainer_wrapper": SUA_ROOT / "mc_maze" / "misleading_identity_swap_v2_trainer.py",
    "encoder_wrapper": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "misleading_identity_swap_v2_encoder.py",
    "preflight": Path(__file__).resolve(),
    "source_authority_builder": SUA_ROOT / "scripts" / "misleading_identity_swap_v2_build_source_authority.py",
    "cell_trainer": SUA_ROOT / "scripts" / "misleading_identity_swap_v2_train_cell.py",
    "runner": SUA_ROOT / "scripts" / "misleading_identity_swap_v2_runner.sh",
    "queue_bridge": SUA_ROOT / "scripts" / "misleading_identity_swap_v2_queue_bridge.sh",
    "queue_verifier": SUA_ROOT / "scripts" / "misleading_identity_swap_v2_queue_verify.py",
    "source_sanity_regression": SUA_ROOT / "scripts" / "misleading_identity_swap_v2_source_sanity_regression.py",
    "scorer": SUA_ROOT / "scripts" / "misleading_identity_swap_v2_scorer.py",
    "aggregate": SUA_ROOT / "scripts" / "misleading_identity_swap_v2_aggregate.py",
    "shared_b3s": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_encoders.py",
    "shared_student": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_spint.py",
    "shared_lit_module": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "streaming_calibration_module.py",
    "shared_datamodule": SUA_ROOT / "mc_maze" / "multisession_datamodule.py",
    "shared_t4_features": SUA_ROOT / "mc_maze" / "unit_side_features.py",
    "a2_parent_config": SUA_ROOT / "configs" / "a2_matched_subject_shift_v2.json",
    "a2_parent_core": SUA_ROOT / "mc_maze" / "a2_matched_subject_shift_v2_core.py",
}


def current_implementation_bindings() -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, path in IMPLEMENTATION_PATHS.items():
        resolved = Path(path).resolve()
        core.require(resolved.is_file(), f"missing implementation binding {name}: {resolved}")
        bindings[name] = {"path": str(resolved), "sha256": core.sha256_file(resolved)}
    return bindings


def load_preserved_invalid_v2_attempt() -> dict[str, Any]:
    """Bind the failed v2 launch as immutable predecessor evidence, not input.

    The successor never resumes this directory.  Its sole permitted use is to
    prove the observed failure happened before target/formal access, any epoch
    checkpoint, and any terminal receipt.
    """
    path = core.INVALID_V2_ATTEMPT_PATH
    payload, digest = core.load_verified_immutable_json(path)
    core.require(digest == core.INVALID_V2_ATTEMPT_SHA256,
                 "preserved invalid v2 launch receipt SHA drift")
    core.require(payload.get("receipt_kind") == "misleading_identity_swap_v2_cell_launch" and
                 payload.get("status") == "AUTHORIZED_LIVE_CELL_INITIALIZED" and
                 payload.get("cell") == "clean_z4" and payload.get("seed") == 42 and
                 payload.get("target_nwb_opened") is False and
                 payload.get("formal_subc_test_nwb_opened") is False,
                 "preserved invalid v2 launch scientific boundary drift")
    old_root = path.parent
    core.require(not (old_root / "terminal_receipt.json").exists() and
                 not (old_root / "epoch_ckpts").exists(),
                 "invalid v2 attempt unexpectedly contains terminal/checkpoint output")
    return {
        "path": str(path.resolve()),
        "launch_receipt_sha256": digest,
        "outcome": "LIGHTNING_SANITY_VALIDATION_ABORT__NO_EPOCH_CHECKPOINT_OR_TERMINAL",
        "target_nwb_opened": False,
        "formal_subc_test_nwb_opened": False,
        "old_b117_v2_official_preflight_reusable": False,
        "old_cell_root_reusable": False,
    }


def build_official_preflight(
    *, authority_path: Path, lineage_path: Path, initial_state_path: Path,
    result_root: Path,
) -> dict[str, Any]:
    """Build the sole source-only receipt that may authorize Stage-P execution."""
    from scripts.misleading_identity_swap_v2_train_cell import _load_initial

    core.require(authority_path.resolve() == core.SOURCE_AUTHORITY_PATH.resolve(),
                 "official preflight source authority path is not canonical")
    core.require(lineage_path.resolve() == core.SOURCE_LINEAGE_PATH.resolve(),
                 "official preflight source lineage path is not canonical v3")
    core.require(initial_state_path.resolve() == core.INITIAL_STATE_PATH.resolve(),
                 "official preflight initial-state path is not canonical")
    config = core.validate_config()
    authority = core.load_verified_authority(authority_path)
    lineage, lineage_sha = core.load_verified_immutable_json(lineage_path)
    core.require(isinstance(lineage.get("implementation_bindings"), Mapping) and
                 bool(lineage.get("implementation_bindings")),
                 "source lineage construction closure missing")
    core.require(lineage.get("matching_authority_consumed_bytes_sha256") == authority.sha256,
                 "official preflight source lineage/authority drift")
    initial, initial_file_sha = _load_initial(initial_state_path)
    core.require(initial.get("matching_authority_sha256") == authority.sha256,
                 "official preflight initial-state authority drift")
    root = result_root.expanduser().resolve()
    occupied = sorted(str(path) for path in root.iterdir()) if root.exists() else []
    core.require(not occupied, f"official Stage-P result root is not empty: {occupied}")
    cell_root = core.CELL_OUTPUT_ROOT.resolve()
    cell_occupied = sorted(str(path) for path in cell_root.iterdir()) if cell_root.exists() else []
    core.require(not cell_occupied, f"official cell output root is not empty: {cell_occupied}")
    parents = {domain: core.load_a2_parent_domain_bindings(domain) for domain in core.DOMAINS}
    bindings = current_implementation_bindings()
    invalid_attempt = load_preserved_invalid_v2_attempt()
    return {
        "schema_version": 1,
        "receipt_kind": core.OFFICIAL_PREFLIGHT_KIND,
        "status": core.OFFICIAL_PREFLIGHT_STATUS,
        "screen_id": core.SCREEN_ID,
        "official": True,
        "authorizes": "only four seed42 Stage-P source cells and declared development scorers",
        "config_path": str(core.CONFIG_PATH.resolve()),
        "config_sha256": core.sha256_file(core.CONFIG_PATH),
        "contract_path": str(core.CONTRACT_PATH.resolve()),
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH),
        "result_root": str(root),
        "matching_authority_path": str(authority_path.resolve()),
        "matching_authority_sha256": authority.sha256,
        "source_lineage_path": str(lineage_path.resolve()),
        "source_lineage_sha256": lineage_sha,
        "initial_state_path": str(initial_state_path.resolve()),
        "initial_state_file_sha256": initial_file_sha,
        "initial_state_dict_sha256": initial["state_dict_sha256"],
        "stage_seed": 42,
        "cells": list(core.CELLS),
        "cell_output_root": str(core.CELL_OUTPUT_ROOT.resolve()),
        "cell_output_paths": core.canonical_cell_output_paths(),
        "epochs": 12,
        "score_epochs_one_based": list(core.SCORE_EPOCHS_ONE_BASED),
        "a2_parent_domain_bindings": parents,
        "preserved_invalid_v2_attempt": invalid_attempt,
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": core.canonical_json_sha256(bindings),
        "source_sessions_opened_by_preflight": 0,
        "validation_nwb_opened": False,
        "target_nwb_opened": False,
        "formal_subc_test_nwb_opened": False,
        "checkpoint_loaded": False,
        "gpu_used": False,
        "training_started": False,
        "config_official_preflight_minted": config["execution"]["official_preflight_minted"],
    }


def load_verified_official_preflight(path: Path) -> tuple[dict[str, Any], str]:
    from scripts.misleading_identity_swap_v2_train_cell import _load_initial

    canonical = core.require_canonical_official_preflight_path(path)
    payload, digest = core.load_verified_immutable_json(canonical)
    core.require(payload.get("receipt_kind") == core.OFFICIAL_PREFLIGHT_KIND,
                 "official preflight kind drift")
    core.require(payload.get("status") == core.OFFICIAL_PREFLIGHT_STATUS and payload.get("official") is True,
                 "official preflight status drift")
    core.require(payload.get("screen_id") == core.SCREEN_ID, "official preflight screen drift")
    bindings = current_implementation_bindings()
    core.require(payload.get("implementation_bindings") == bindings,
                 "official preflight implementation closure drift")
    core.require(payload.get("implementation_bindings_sha256") == core.canonical_json_sha256(bindings),
                 "official preflight implementation binding SHA drift")
    authority_path = Path(str(payload.get("matching_authority_path", "")))
    core.require(authority_path.resolve() == core.SOURCE_AUTHORITY_PATH.resolve(),
                 "official preflight source authority path drift")
    authority = core.load_verified_authority(authority_path)
    core.require(authority.sha256 == payload.get("matching_authority_sha256"),
                 "official preflight authority drift")
    lineage_path = Path(str(payload.get("source_lineage_path", "")))
    core.require(lineage_path.resolve() == core.SOURCE_LINEAGE_PATH.resolve(),
                 "official preflight source lineage path drift")
    lineage, lineage_sha = core.load_verified_immutable_json(lineage_path)
    core.require(isinstance(lineage.get("implementation_bindings"), Mapping) and
                 bool(lineage.get("implementation_bindings")),
                 "official source lineage construction closure missing")
    core.require(lineage_sha == payload.get("source_lineage_sha256"),
                 "official preflight lineage drift")
    core.require(lineage.get("matching_authority_consumed_bytes_sha256") == authority.sha256,
                 "official preflight lineage/authority binding drift")
    initial_path = Path(str(payload.get("initial_state_path", "")))
    core.require(initial_path.resolve() == core.INITIAL_STATE_PATH.resolve(),
                 "official preflight initial-state path drift")
    initial, initial_sha = _load_initial(initial_path)
    core.require(initial_sha == payload.get("initial_state_file_sha256") and
                 initial.get("state_dict_sha256") == payload.get("initial_state_dict_sha256"),
                 "official preflight initial-state drift")
    core.require(initial.get("matching_authority_sha256") == authority.sha256,
                 "official preflight initial-state authority drift")
    parents = {domain: core.load_a2_parent_domain_bindings(domain) for domain in core.DOMAINS}
    core.require(payload.get("a2_parent_domain_bindings") == parents,
                 "official preflight sealed A2 parent drift")
    core.require(payload.get("preserved_invalid_v2_attempt") == load_preserved_invalid_v2_attempt(),
                 "official preflight invalid v2 attempt binding drift")
    core.require(payload.get("stage_seed") == 42 and payload.get("cells") == list(core.CELLS),
                 "official preflight Stage-P topology drift")
    core.require(payload.get("cell_output_root") == str(core.CELL_OUTPUT_ROOT.resolve()) and
                 payload.get("cell_output_paths") == core.canonical_cell_output_paths(),
                 "official preflight cell output topology drift")
    core.require(payload.get("epochs") == 12 and
                 payload.get("score_epochs_one_based") == list(core.SCORE_EPOCHS_ONE_BASED),
                 "official preflight epoch policy drift")
    return payload, digest


def build_development_preflight(
    *,
    authority_payload: Mapping[str, Any],
    result_root: Path,
    config_path: Path = core.CONFIG_PATH,
    authority_consumed_bytes_sha256: str | None = None,
    authority_file_path: Path | None = None,
) -> dict[str, Any]:
    """Build, but never write, a CPU/no-data development preflight payload."""
    config = core.validate_config(config_path)
    authority = core.VerifiedMatchingAuthority.from_payload(
        authority_payload,
        consumed_bytes_sha256=authority_consumed_bytes_sha256,
        immutable_file_path=authority_file_path,
    )
    root = Path(result_root).expanduser().resolve()
    occupied: list[str] = []
    if root.exists():
        occupied = sorted(str(path) for path in root.iterdir())
    core.require(not occupied, f"fresh Stage-P result root is not empty: {occupied}")
    bindings = current_implementation_bindings()
    return {
        "schema_version": 1,
        "receipt_kind": core.PREFLIGHT_KIND,
        "status": core.PREFLIGHT_STATUS,
        "screen_id": core.SCREEN_ID,
        "non_authorizing": True,
        "official_preflight_minted": False,
        "result_root": str(root),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": core.sha256_file(config_path),
        "contract_path": str(core.CONTRACT_PATH.resolve()),
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH),
        "parent_a2_use": config["parent_protocol"]["use"],
        "matching_authority_consumed_bytes_sha256": authority.sha256,
        "matching_authority_canonical_sha256": authority.canonical_sha256,
        "matching_authority_path": authority.immutable_file_path,
        "matching_authority_session_count": len(authority.payload["sessions"]),
        "t4_z4_shared_authority_required": True,
        "stage_seed": core.STAGE_SEED,
        "fresh_training_cells": list(core.CELLS),
        "domains": list(core.DOMAINS),
        "evaluation_input_modes": list(core.EVAL_INPUT_MODES),
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": core.canonical_json_sha256(bindings),
        "data_opened": False,
        "checkpoint_loaded": False,
        "target_nwb_opened": False,
        "formal_subc_test_nwb_opened": False,
        "gpu_used": False,
        "training_started": False,
        "scoring_started": False,
        "remaining_blockers": [
            "development source authority must be independently reviewed",
            "four fresh GPU cells have not been authorized or executed",
            "development target access and GPU scoring remain separately unauthorized",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--result-root", type=Path, default=core.RESULT_ROOT)
    parser.add_argument("--check-development-authority", action="store_true")
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--official-preflight", type=Path, default=core.OFFICIAL_PREFLIGHT_PATH)
    parser.add_argument("--mint-official", action="store_true")
    parser.add_argument("--verify-official", action="store_true")
    args = parser.parse_args()
    if args.mint_official:
        if args.authority is None or args.lineage is None or args.initial_state is None:
            raise SystemExit("--authority, --lineage and --initial-state are required")
        canonical = core.require_canonical_official_preflight_path(args.official_preflight)
        core.assert_immutable_pair_fresh(canonical, label="official preflight")
        payload = build_official_preflight(
            authority_path=args.authority, lineage_path=args.lineage,
            initial_state_path=args.initial_state, result_root=args.result_root,
        )
        _body, _side, digest = core.write_immutable_json_pair(canonical, payload)
        print(json.dumps({"status": payload["status"], "path": str(canonical),
                          "sha256": digest}, indent=2, sort_keys=True))
        return 0
    if args.verify_official:
        payload, digest = load_verified_official_preflight(args.official_preflight)
        print(json.dumps({"status": payload["status"], "sha256": digest}, indent=2, sort_keys=True))
        return 0
    if not args.check_development_authority:
        print(json.dumps({
            "status": "DRY_RUN__NON_AUTHORIZING__NO_DATA_NO_GPU",
            "screen_id": core.SCREEN_ID,
            "required_authority": None if args.authority is None else str(args.authority.resolve()),
            "official_preflight_minted": False,
            "gpu_launch_implemented": True,
            "target_scorer_implemented": True,
        }, indent=2, sort_keys=True))
        return 0
    if args.authority is None:
        raise SystemExit("--authority is required with --check-development-authority")
    payload, file_sha = core.load_verified_immutable_json(args.authority)
    receipt = build_development_preflight(
        authority_payload=payload,
        result_root=args.result_root,
        authority_consumed_bytes_sha256=file_sha,
        authority_file_path=args.authority,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
