"""Torch-free C1 teacher-domain ablation contract helpers.

The 2x2 is teacher {mc_maze, co_native} x carrier {t4, z4} with W-add held
fixed.  This module is import-safe without Torch, NWB, or CUDA so the
preflight, aggregator, and inert runner can stay CPU-only and source-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mc_maze.gpu_contract_common import (
    NONINFERIORITY_MARGIN,
    PREDECLARED_SEEDS,
    SEALED_FORMAL_TEST_SESSIONS,
    SUBC_VAL_SESSIONS,
    SUBM_EXTERNAL_SESSIONS,
    assert_sessions_not_sealed,
    gate_verdict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"

SCREEN_ID = "c1_teacher_domain_ablation_v1"
CONTRACT_PATH = SUA_ROOT / "docs" / "C1_TEACHER_DOMAIN_ABLATION_CONTRACT_20260813.md"
CONFIG_PATH = SUA_ROOT / "configs" / "c1_teacher_domain_ablation.json"
MANIFEST_PATH = SUA_ROOT / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
MC_MAZE_TEACHER_PATH = (
    SUA_ROOT / "checkpoints" / "teacher_mc_maze" / "best-epoch=083-val_heldin" / "r2_mean=0.9061.ckpt"
)
RESULT_ROOT = SUA_ROOT / "results" / SCREEN_ID
COMPATIBILITY_RECEIPT_NAME = "source_only_teacher_compatibility_receipt.json"
OFFICIAL_PREFLIGHT_NAME = "official_cpu_preflight.json"

EXPECTED_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
EXPECTED_TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"

TEACHER_DOMAINS: tuple[str, ...] = ("mc_maze", "co_native")
CARRIERS: tuple[str, ...] = ("t4", "z4")
SEEDS: tuple[int, ...] = PREDECLARED_SEEDS
DOMAINS: tuple[str, ...] = ("within_subject", "external_subject_M")
ADD_SITE = "W-add"
SAMPLING = "legacy"
LOSS_MODE = "task_only"
VARIANT = "B3S"
IDENTITY_MODE = "calibrated"
DECODER_MODE = "coupled"
STUDENT_LIGHTNING_TASK = "mc_maze"
ENCODER_WARMSTART = None

WINDOW_SIZE_BINS = 50
TRIAL_LENGTH_BINS = 100
IDENTITY_MLP_IN_FEATURES = 100
MODEL_DIM = 512
NUM_HEADS = 64
NUM_LAYERS = 1
NUM_ID_LAYERS = 3
NUM_COVARIATES = 2
EFFECTIVE_THRESHOLD = 0.03
WITHIN_T4_NONINFERIORITY_FLOOR = NONINFERIORITY_MARGIN

SOURCE_CELL_COUNT = len(TEACHER_DOMAINS) * len(CARRIERS) * len(SEEDS)
DOMAIN_RECEIPT_COUNT = SOURCE_CELL_COUNT * len(DOMAINS)

GPU_AUTH_ENV = "C1_TEACHER_DOMAIN_GPU_AUTHORIZATION"
GPU_AUTH_VALUE = "I_AUTHORIZE_C1_TEACHER_DOMAIN_ABLATION_GPU"
ROOT_GO_ENV = "C1_TEACHER_DOMAIN_ROOT_GO"
ROOT_GO_VALUE = "I_SIGN_C1_TEACHER_DOMAIN_ABLATION_ROOT_GO"

PRODUCTION_SPINT_HYPERPARAMETERS: Mapping[str, Any] = {
    "model_dim": MODEL_DIM,
    "num_covariates": NUM_COVARIATES,
    "window_size": WINDOW_SIZE_BINS,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "num_id_layers": NUM_ID_LAYERS,
    "use_learnable_id": True,
    "learnable_id_type": "mlp",
    "learnable_rep": True,
    "dropout_rate": 0.0,
    "dynamic_dropout": True,
    "dynamic_dropout_low": 0.0,
    "dynamic_dropout_high": 1.0,
    "tf_drop_rate": 0.1,
    "readin_layer_type": "mlp",
}

STUDENT_CONTRACT: Mapping[str, Any] = {
    "lightning_task": STUDENT_LIGHTNING_TASK,
    "variant": VARIANT,
    "loss_mode": LOSS_MODE,
    "identity_mode": IDENTITY_MODE,
    "decoder_mode": DECODER_MODE,
    "freeze_decoder": False,
    "encoder_warmstart_path": ENCODER_WARMSTART,
    "add_site": ADD_SITE,
    "sampling": SAMPLING,
    "window_size": WINDOW_SIZE_BINS,
    "trial_length": TRIAL_LENGTH_BINS,
    "side_dim": 4,
    "copies_teacher_state_dict_into_decoder_strict": True,
    "copies_teacher_state_dict_even_under_task_only": True,
    "b3s_copies_teacher_fc_id_modules": False,
    "selected_t4_encoder_warmstart": False,
    "hypothesis_status": "plausible_contributor_not_isolated_cause",
}


class C1ContractError(ValueError):
    """Raised when a C1 contract, receipt, or gate input is illegal."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C1ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(pretty_json_bytes(payload)).hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path}: expected a JSON object")
    return payload


def immutable_sidecar_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.sha256")


def _readonly_regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    requested = Path(path).expanduser()
    require(not requested.is_symlink(), f"immutable receipt path may not be a symlink: {requested}")
    body_path = requested.resolve()
    sidecar_path = immutable_sidecar_path(body_path)
    if os.path.lexists(body_path) or os.path.lexists(sidecar_path):
        raise FileExistsError(f"refusing immutable overwrite: {body_path}")
    body_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pretty_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    descriptor = os.open(body_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    os.chmod(body_path, 0o444)
    require(_readonly_regular_file(body_path), f"immutable body mode failed: {body_path}")
    sidecar_raw = f"{digest}  {body_path.name}\n".encode("ascii")
    descriptor = os.open(sidecar_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(sidecar_raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    os.chmod(sidecar_path, 0o444)
    require(_readonly_regular_file(sidecar_path), f"immutable sidecar mode failed: {sidecar_path}")
    return body_path, sidecar_path, digest


def load_strict_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = load_json_object(path)
    splits = payload.get("session_splits") or payload
    require(isinstance(splits, Mapping), f"{path}: session_splits missing")
    train = tuple(splits.get("train") or ())
    val = tuple(splits.get("val") or ())
    test = tuple(splits.get("test") or ())
    require(len(train) == 27, f"{path}: expected 27 source-train sessions")
    require(val == SUBC_VAL_SESSIONS, f"{path}: validation roster drift")
    require(set(test) == set(SEALED_FORMAL_TEST_SESSIONS) and len(test) == 6,
            f"{path}: sealed formal-test roster drift")
    assert_sessions_not_sealed(train, label=f"{path} train")
    assert_sessions_not_sealed(val, label=f"{path} val")
    return {"train": list(train), "val": list(val), "test": list(test)}


def refuse_sealed_sessions(session_names: Sequence[str], *, label: str) -> None:
    assert_sessions_not_sealed(session_names, label=label)


def refuse_target_or_sealed_paths(paths: Sequence[Path | str], *, label: str) -> None:
    """Refuse any path that names sub-M target data or a sealed test session."""
    blocked: list[str] = []
    for raw in paths:
        text = str(raw)
        lower = text.lower()
        if "sub-m" in lower or "/sub-m/" in lower or "dandi_000688/sub-m" in lower:
            blocked.append(text)
        for session in SEALED_FORMAL_TEST_SESSIONS:
            if session in text:
                blocked.append(text)
    require(not blocked, f"{label}: target or sealed path requested: {blocked}")


def source_cells() -> tuple[dict[str, Any], ...]:
    rows = []
    for teacher in TEACHER_DOMAINS:
        for carrier in CARRIERS:
            for seed in SEEDS:
                rows.append(
                    {
                        "name": f"{teacher}_{carrier}_s{seed}",
                        "teacher_domain": teacher,
                        "carrier": carrier,
                        "side_feature_group": carrier,
                        "seed": seed,
                        "add_site": ADD_SITE,
                        "sampling": SAMPLING,
                        "variant": VARIANT,
                        "loss_mode": LOSS_MODE,
                    }
                )
    require(len(rows) == SOURCE_CELL_COUNT, "source-cell topology drift")
    return tuple(rows)


def domain_receipt_name(teacher: str, carrier: str, seed: int, domain: str) -> str:
    return f"{teacher}_{carrier}_s{seed}_{domain}.json"


def expected_domain_sessions(domain: str) -> tuple[str, ...]:
    if domain == "within_subject":
        return SUBC_VAL_SESSIONS
    if domain == "external_subject_M":
        return SUBM_EXTERNAL_SESSIONS
    raise C1ContractError(f"unknown scoring domain: {domain}")


def validate_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    payload = load_json_object(config_path)
    require(payload.get("schema_version") == 1, "C1 config schema drift")
    require(payload.get("screen_id") == SCREEN_ID, "C1 config screen id drift")
    require(tuple(payload.get("teacher_domains") or ()) == TEACHER_DOMAINS, "C1 teacher-domain topology drift")
    require(tuple(payload.get("carriers") or ()) == CARRIERS, "C1 carrier topology drift")
    require(tuple(payload.get("seeds") or ()) == SEEDS, "C1 seed roster drift")
    require(payload.get("add_site") == ADD_SITE, "C1 add-site drift; W-add must stay fixed")
    require(payload.get("sampling") == SAMPLING, "C1 sampling drift; C2 is out of scope")
    require(payload.get("loss_mode") == LOSS_MODE, "C1 loss-mode drift")
    require(payload.get("hidden_space_adapter") is False, "C1 must not combine with A1 H-add")
    protocol = payload.get("frozen_protocol") or {}
    require(isinstance(protocol, Mapping), "C1 frozen_protocol missing")
    require(protocol.get("encoder_warmstart_path") is None, "C1 selected-T4 encoder warm-start is forbidden")
    require(protocol.get("lightning_task") == STUDENT_LIGHTNING_TASK, "C1 student task-string drift")
    require(protocol.get("copies_teacher_decoder_under_task_only") is True,
            "C1 must record that task_only still loads the teacher decoder")
    return payload


def empty_score_matrix() -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    return {
        teacher: {
            carrier: {seed: {domain: float("nan") for domain in DOMAINS} for seed in SEEDS}
            for carrier in CARRIERS
        }
        for teacher in TEACHER_DOMAINS
    }


def _seed_vector(matrix: Mapping[str, Any], teacher: str, carrier: str, domain: str) -> np.ndarray:
    return np.asarray([float(matrix[teacher][carrier][seed][domain]) for seed in SEEDS], dtype=np.float64)


def evaluate_c1_gates(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen C1 gate.  Must be able both to pass and to fail."""
    for teacher in TEACHER_DOMAINS:
        require(teacher in matrix, f"missing teacher domain {teacher}")
        for carrier in CARRIERS:
            require(carrier in matrix[teacher], f"missing carrier {teacher}/{carrier}")
            for seed in SEEDS:
                require(seed in matrix[teacher][carrier], f"missing seed {teacher}/{carrier}/{seed}")
                for domain in DOMAINS:
                    value = matrix[teacher][carrier][seed][domain]
                    require(np.isfinite(float(value)), f"non-finite R2 at {teacher}/{carrier}/{seed}/{domain}")

    t4_lift = _seed_vector(matrix, "co_native", "t4", "external_subject_M") - _seed_vector(
        matrix, "mc_maze", "t4", "external_subject_M"
    )
    z4_lift = _seed_vector(matrix, "co_native", "z4", "external_subject_M") - _seed_vector(
        matrix, "mc_maze", "z4", "external_subject_M"
    )
    co_gain = _seed_vector(matrix, "co_native", "t4", "external_subject_M") - _seed_vector(
        matrix, "co_native", "z4", "external_subject_M"
    )
    mc_gain = _seed_vector(matrix, "mc_maze", "t4", "external_subject_M") - _seed_vector(
        matrix, "mc_maze", "z4", "external_subject_M"
    )
    interaction = co_gain - mc_gain
    within_t4 = _seed_vector(matrix, "co_native", "t4", "within_subject") - _seed_vector(
        matrix, "mc_maze", "t4", "within_subject"
    )

    z4_reproduces = bool(float(z4_lift.mean()) >= EFFECTIVE_THRESHOLD and np.all(z4_lift > 0.0))
    gates = {
        "mean_external_t4_lift_at_least_0p03": bool(float(t4_lift.mean()) >= EFFECTIVE_THRESHOLD),
        "all_three_seed_external_t4_lifts_positive": bool(np.all(t4_lift > 0.0)),
        "z4_does_not_reproduce_the_lift": (not z4_reproduces),
        "mean_external_interaction_at_least_0p03": bool(float(interaction.mean()) >= EFFECTIVE_THRESHOLD),
        "all_three_seed_external_interactions_positive": bool(np.all(interaction > 0.0)),
        "within_subject_t4_noninferiority_floor": bool(float(within_t4.mean()) >= WITHIN_T4_NONINFERIORITY_FLOOR),
    }
    passes = bool(all(gates.values()))
    return {
        "screen_id": SCREEN_ID,
        "add_site": ADD_SITE,
        "sampling": SAMPLING,
        "external_t4_lift": {
            "per_seed": {str(seed): float(value) for seed, value in zip(SEEDS, t4_lift)},
            "mean": float(t4_lift.mean()),
        },
        "external_z4_lift": {
            "per_seed": {str(seed): float(value) for seed, value in zip(SEEDS, z4_lift)},
            "mean": float(z4_lift.mean()),
            "reproduces_t4_lift": z4_reproduces,
        },
        "external_interaction": {
            "definition": "(T4-Z4)_CO-native_external - (T4-Z4)_MC-Maze_external",
            "per_seed": {str(seed): float(value) for seed, value in zip(SEEDS, interaction)},
            "mean": float(interaction.mean()),
        },
        "within_t4_delta": {
            "per_seed": {str(seed): float(value) for seed, value in zip(SEEDS, within_t4)},
            "mean": float(within_t4.mean()),
            "noninferiority_floor": WITHIN_T4_NONINFERIORITY_FLOOR,
        },
        "seed_level_wilcoxon": {
            "computed": False,
            "reason": "prohibited: with three seeds an exact two-sided Wilcoxon cannot attain p <= 0.05",
        },
        "gates": gates,
        "passes_all_gates": passes,
        "verdict": gate_verdict(
            passes=passes,
            gate_name="teacher_domain_external_t4_effective",
            reason=(
                "CO-native teacher improves absolute external T4 by at least +0.03 "
                "on all three seeds, Z4 does not reproduce the lift, the matched "
                "teacher-by-carrier interaction is at least +0.03 on all seeds, and "
                "within-subject T4 is non-inferior"
                if passes
                else "frozen C1 gate failed; teacher/target mismatch remains a plausible contributor, not an established cause"
            ),
        ),
        "hypothesis_status": "plausible_contributor_not_isolated_cause",
    }


def _fill_base_matrix(*, mc_t4_ext: float, mc_z4_ext: float, mc_t4_within: float, mc_z4_within: float) -> dict[str, Any]:
    matrix = empty_score_matrix()
    for seed in SEEDS:
        matrix["mc_maze"]["t4"][seed]["external_subject_M"] = mc_t4_ext
        matrix["mc_maze"]["z4"][seed]["external_subject_M"] = mc_z4_ext
        matrix["mc_maze"]["t4"][seed]["within_subject"] = mc_t4_within
        matrix["mc_maze"]["z4"][seed]["within_subject"] = mc_z4_within
        matrix["co_native"]["t4"][seed]["within_subject"] = mc_t4_within
        matrix["co_native"]["z4"][seed]["within_subject"] = mc_z4_within
    return matrix


def synthetic_pass_matrix() -> dict[str, Any]:
    """External T4 lift +0.06, Z4 ~0, interaction +0.06, within T4 held."""
    matrix = _fill_base_matrix(mc_t4_ext=0.34, mc_z4_ext=-0.14, mc_t4_within=0.57, mc_z4_within=0.33)
    for seed in SEEDS:
        matrix["co_native"]["t4"][seed]["external_subject_M"] = 0.40
        matrix["co_native"]["z4"][seed]["external_subject_M"] = -0.14
        matrix["co_native"]["t4"][seed]["within_subject"] = 0.58
    return matrix


def synthetic_fail_generic_lift_matrix() -> dict[str, Any]:
    """Both T4 and Z4 lift +0.06: generic teacher improvement, not carrier use."""
    matrix = _fill_base_matrix(mc_t4_ext=0.34, mc_z4_ext=-0.14, mc_t4_within=0.57, mc_z4_within=0.33)
    for seed in SEEDS:
        matrix["co_native"]["t4"][seed]["external_subject_M"] = 0.40
        matrix["co_native"]["z4"][seed]["external_subject_M"] = -0.08
        matrix["co_native"]["t4"][seed]["within_subject"] = 0.58
        matrix["co_native"]["z4"][seed]["within_subject"] = 0.39
    return matrix


def synthetic_fail_z4_crash_matrix() -> dict[str, Any]:
    """T4 unchanged, Z4 collapses: T4-Z4 inflates without an absolute T4 lift."""
    matrix = _fill_base_matrix(mc_t4_ext=0.34, mc_z4_ext=-0.14, mc_t4_within=0.57, mc_z4_within=0.33)
    for seed in SEEDS:
        matrix["co_native"]["t4"][seed]["external_subject_M"] = 0.34
        matrix["co_native"]["z4"][seed]["external_subject_M"] = -0.50
    return matrix


def synthetic_fail_no_t4_lift_matrix() -> dict[str, Any]:
    matrix = _fill_base_matrix(mc_t4_ext=0.34, mc_z4_ext=-0.14, mc_t4_within=0.57, mc_z4_within=0.33)
    for seed in SEEDS:
        matrix["co_native"]["t4"][seed]["external_subject_M"] = 0.34
        matrix["co_native"]["z4"][seed]["external_subject_M"] = -0.14
    return matrix


def compatibility_receipt_path(result_root: Path | None = None) -> Path:
    root = RESULT_ROOT if result_root is None else Path(result_root)
    return root / COMPATIBILITY_RECEIPT_NAME


def official_preflight_path(result_root: Path | None = None) -> Path:
    root = RESULT_ROOT if result_root is None else Path(result_root)
    return root / OFFICIAL_PREFLIGHT_NAME
