"""Contract helpers for the sealed-A2 SetKV-delta forward diagnostic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mc_maze import a2_matched_subject_shift_v2_core as a2


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
SCREEN_ID = "setkv_delta_forward_v1"
RESULT_ROOT = SUA_ROOT / "results" / SCREEN_ID
CONTRACT_PATH = SUA_ROOT / "docs" / "SETKV_DELTA_FORWARD_PROTOCOL_20260814.md"
A2_ROOT = SUA_ROOT / "results" / "a2_matched_subject_shift_v2"
A2_PREFLIGHT = A2_ROOT / "official_cpu_preflight.json"
A2_TERMINAL = A2_ROOT / "terminal_aggregate.json"
EXPECTED_A2_PREFLIGHT_SHA256 = "8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd"
EXPECTED_A2_TERMINAL_SHA256 = "5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc"
OFFICIAL_PREFLIGHT = RESULT_ROOT / "official_cpu_preflight.json"
PREFLIGHT_STATUS = "SETKV_DELTA_CPU_PREFLIGHT_PASSED__FORWARD_NOT_STARTED"
SEED = 42
DOMAINS = ("within_subject", "external_subject_M")
INTERVENTIONS: Mapping[str, Mapping[str, Any]] = {
    "setkv_t4": {"source_arm": "source_t4", "decode_mode": "carrier", "carrier_mode": "aligned"},
    "setkv_z4": {"source_arm": "source_z4", "decode_mode": "carrier", "carrier_mode": "zero"},
    "setkv_rs4": {"source_arm": "source_t4", "decode_mode": "carrier", "carrier_mode": "row_shuffle"},
    "duplicate_activity_t4": {"source_arm": "source_t4", "decode_mode": "duplicate_activity", "carrier_mode": None},
}

IMPLEMENTATION_BINDING_PATHS: Mapping[str, Path] = {
    "contract": CONTRACT_PATH,
    "setkv_math": SUA_ROOT / "mc_maze" / "setkv_delta.py",
    "core": Path(__file__).resolve(),
    "preflight": SUA_ROOT / "scripts" / "preflight_setkv_delta_forward.py",
    "scorer": SUA_ROOT / "scripts" / "score_setkv_delta_forward.py",
    "aggregator": SUA_ROOT / "scripts" / "aggregate_setkv_delta_forward.py",
    "queue_runner": SUA_ROOT / "scripts" / "watch_and_run_post_cebra_queue.sh",
    "a2_core": SUA_ROOT / "mc_maze" / "a2_matched_subject_shift_v2_core.py",
    "a2_scorer": SUA_ROOT / "scripts" / "a2_matched_subject_shift_v2_score.py",
    "shared_evaluator": SUA_ROOT / "scripts" / "eval_adaptation_dandi688.py",
    "frozen_model_loader": SUA_ROOT / "scripts" / "select_gradient_free_protocol_dandi688.py",
    "datamodule": SUA_ROOT / "mc_maze" / "multisession_datamodule.py",
    "session_datamodule": SUA_ROOT / "mc_maze" / "datamodule.py",
    "unit_side_features": SUA_ROOT / "mc_maze" / "unit_side_features.py",
    "streaming_module": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "streaming_calibration_module.py",
    "spint": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "spint.py",
    "streaming_spint": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_spint.py",
    "streaming_encoders": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_encoders.py",
}


class SetKVForwardContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SetKVForwardContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def current_implementation_bindings() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, path in IMPLEMENTATION_BINDING_PATHS.items():
        resolved = path.resolve()
        require(resolved.is_file(), f"missing implementation binding {name}: {resolved}")
        result[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return result


def verify_implementation_bindings(value: Any) -> dict[str, dict[str, str]]:
    require(isinstance(value, Mapping), "implementation bindings missing")
    current = current_implementation_bindings()
    require(dict(value) == current, "SetKV implementation binding drift")
    return current


def load_immutable(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        value, digest = a2.load_verified_immutable_json(path, label=label)
    except Exception as exc:  # preserve one local error type at the boundary
        raise SetKVForwardContractError(str(exc)) from exc
    return value, digest


def load_official_preflight(path: Path = OFFICIAL_PREFLIGHT) -> tuple[dict[str, Any], str]:
    payload, digest = load_immutable(path, "SetKV official CPU preflight")
    require(payload.get("receipt_kind") == "setkv_delta_forward_official_cpu_preflight", "preflight kind drift")
    require(payload.get("status") == PREFLIGHT_STATUS, "preflight status drift")
    verify_implementation_bindings(payload.get("implementation_bindings"))
    require(payload.get("implementation_bindings_sha256") == canonical_sha256(payload["implementation_bindings"]),
            "preflight implementation digest drift")
    require(payload.get("a2_official_preflight_sha256") == EXPECTED_A2_PREFLIGHT_SHA256, "A2 preflight anchor drift")
    require(payload.get("a2_terminal_aggregate_sha256") == EXPECTED_A2_TERMINAL_SHA256, "A2 terminal anchor drift")
    return payload, digest


def a2_baseline_receipt_path(source_arm: str, domain: str) -> Path:
    require(source_arm in a2.SOURCE_ARMS, f"unknown A2 source arm: {source_arm}")
    require(domain in DOMAINS, f"unknown domain: {domain}")
    return A2_ROOT / f"{domain}_{source_arm}_s{SEED}.json"


def intervention(name: str) -> Mapping[str, Any]:
    require(name in INTERVENTIONS, f"unknown SetKV intervention: {name}")
    return INTERVENTIONS[name]


def score_path(name: str, domain: str) -> Path:
    intervention(name)
    require(domain in DOMAINS, f"unknown domain: {domain}")
    return RESULT_ROOT / f"{domain}_{name}_s{SEED}.json"


def session_permutation_seed(session: str) -> int:
    raw = hashlib.sha256(f"setkv_delta_rs4_v1::{session}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big", signed=False)


def write_immutable(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    return a2.write_immutable_json(path, payload)
