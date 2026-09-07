"""A1 implementation binding and immutable-preflight verification."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import canonical_json_sha256, load_verified_immutable_json, require, sha256_file
from .contract import SCREEN_ID

REPO = Path(__file__).resolve().parents[2]
SUA = REPO / "sua_exploration"
STREAMING = REPO / "streaming_calibration_exp"

IMPLEMENTATION_PATHS: Mapping[str, Path] = {
    "contract": SUA / "docs/A1_HIDDEN_SPACE_CARRIER_ADAPTER_CONTRACT_20260813.md",
    "package_init": SUA / "a1_hidden_carrier/__init__.py",
    "artifacts": SUA / "a1_hidden_carrier/artifacts.py",
    "a2_anchors": SUA / "a1_hidden_carrier/a2_anchors.py",
    "contract_core": SUA / "a1_hidden_carrier/contract.py",
    "cpu_proofs": SUA / "a1_hidden_carrier/cpu_proofs.py",
    "evidence": Path(__file__).resolve(),
    "production_adapter": STREAMING / "src/models/components/streaming_spint_hidden_carrier_adapter.py",
    "production_lightning_module": STREAMING / "src/models/a1_hidden_carrier_module.py",
    "production_model_config": STREAMING / "configs/model/a1_hidden_carrier_b3s.yaml",
    "production_spint_decoder": STREAMING / "src/models/components/spint.py",
    "production_streaming_encoders": STREAMING / "src/models/components/streaming_encoders.py",
    "production_run_artifacts": STREAMING / "src/metrics/run_artifacts.py",
    "shared_streaming_base": STREAMING / "src/models/components/streaming_spint.py",
    "shared_streaming_module": STREAMING / "src/models/streaming_calibration_module.py",
    "a1_multisession_datamodule": SUA / "mc_maze/multisession_datamodule.py",
    "a1_session_datamodule": SUA / "mc_maze/datamodule.py",
    "a1_unit_side_features": SUA / "mc_maze/unit_side_features.py",
    "a1_a2_scoring_helpers": SUA / "scripts/a2_matched_subject_shift_v2_score.py",
    "a1_a2_protocol_core": SUA / "mc_maze/a2_matched_subject_shift_v2_core.py",
    "a1_eval_helpers": SUA / "scripts/eval_adaptation_dandi688.py",
    "a1_gradient_free_loader": SUA / "scripts/select_gradient_free_protocol_dandi688.py",
    "sealed_a2_trainer": SUA / "scripts/train_variant_dandi688.py",
    "a1_dedicated_trainer": SUA / "a1_hidden_carrier/trainer.py",
    "a1_dedicated_scorer": SUA / "a1_hidden_carrier/scorer.py",
    "preflight": SUA / "scripts/a1_hidden_space_carrier_preflight.py",
    "runner": SUA / "scripts/run_a1_hidden_space_carrier_one_cell.sh",
    "aggregator": SUA / "scripts/aggregate_a1_hidden_space_carrier.py",
}


def current_implementation_bindings() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for name, path in IMPLEMENTATION_PATHS.items():
        resolved = path.resolve()
        require(resolved.is_file(), f"missing A1 implementation binding {name}: {resolved}")
        rows[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return rows


def implementation_bindings_sha256(bindings: Mapping[str, Any]) -> str:
    return canonical_json_sha256(bindings)


def verify_implementation_bindings(value: object) -> dict[str, dict[str, str]]:
    require(isinstance(value, Mapping), "A1 implementation bindings missing")
    current = current_implementation_bindings()
    require(dict(value) == current, "A1 implementation binding drift since preflight")
    return current


def load_verified_preflight(path: Path) -> tuple[dict[str, Any], str]:
    payload, digest = load_verified_immutable_json(path, label="A1 official CPU preflight")
    require(payload.get("schema_version") == 2, "A1 preflight schema drift")
    require(payload.get("receipt_kind") == "a1_hidden_space_carrier_official_preflight", "not an A1 preflight")
    require(payload.get("screen_id") == SCREEN_ID, "A1 preflight screen drift")
    require(payload.get("status") == "CPU_PREFLIGHT_PASSED_AWAITING_ROOT_GO", "A1 preflight did not pass")
    require(payload.get("authorizes_gpu") is False, "CPU preflight must not itself authorize GPU")
    require(payload.get("formal_subc_test_nwb_opened") is False, "preflight opened formal test")
    bindings = verify_implementation_bindings(payload.get("implementation_bindings"))
    require(
        payload.get("implementation_bindings_sha256") == implementation_bindings_sha256(bindings),
        "A1 implementation binding digest drift",
    )
    anchor = payload.get("a2_reuse_evidence")
    require(isinstance(anchor, Mapping) and anchor.get("status") == "SEALED_A2_W_REUSE_VERIFIED", "A2 reuse evidence absent")
    return payload, digest
