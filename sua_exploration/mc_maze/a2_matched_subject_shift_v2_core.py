"""Shared, dependency-light contract helpers for A2 matched subject-shift v2.

The A2 v2 experiment has two source-trained arms (Z4 and T4) for each of
three seeds.  Each *unchanged* source run is evaluated on both the sub-C
development validation roster and the frozen external sub-M roster.  This
module deliberately contains only topology, provenance, and audit helpers;
it has no model, Torch, PyNWB, or GPU import at module import time.

All data-owning work is deferred to the preflight/evaluator entrypoints so a
dry-run can remain genuinely inert.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import site
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"

SCREEN_ID = "a2_matched_subject_shift_v2"
CONFIG_PATH = SUA_ROOT / "configs" / "a2_matched_subject_shift_v2.json"
CONTRACT_PATH = SUA_ROOT / "docs" / "A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md"
MANIFEST_PATH = SUA_ROOT / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
TEACHER_PATH = SUA_ROOT / "checkpoints" / "teacher_mc_maze" / "best-epoch=083-val_heldin" / "r2_mean=0.9061.ckpt"
SUBC_DATA_ROOT = SUA_ROOT / "data" / "dandi_000688" / "sub-C"
SUBM_DATA_ROOT = SUA_ROOT / "data" / "dandi_000688" / "sub-M"
SOURCE_CACHE_ROOT = SUA_ROOT / "cache" / "dandi688_subc_co_v1"
RESULT_ROOT = SUA_ROOT / "results" / SCREEN_ID
CHECKPOINT_ROOT = SUA_ROOT / "checkpoints"

EXPECTED_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
EXPECTED_TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"

SEEDS: tuple[int, ...] = (42, 43, 44)
SOURCE_ARMS: Mapping[str, Mapping[str, str]] = {
    "source_z4": {"arm": "z4", "variant": "B3S", "side_feature_group": "z4"},
    "source_t4": {"arm": "t4", "variant": "B3S", "side_feature_group": "t4"},
}
DOMAINS: tuple[str, ...] = ("within_subject", "external_subject_M")
WITHIN_SESSION_COUNT = 6
EXTERNAL_SESSION_COUNT = 15
ACTIVITY_CALIBRATION_TRIALS = 30
SIDE_FEATURE_POOL_TRIALS = 30
EVALUATION_START_TRIAL_INDEX = 30
WINDOW_SIZE_BINS = 50
TRIAL_LENGTH_BINS = 100
BIN_SIZE_MS = 20
TOTAL_EPOCHS = 12
EPOCH_WINDOW: tuple[int, ...] = tuple(range(5, 13))
INTERACTION_THRESHOLD = 0.03
GPU_AUTH_ENV = "A2_V2_GPU_AUTHORIZATION"
GPU_AUTH_VALUE = "I_AUTHORIZE_A2_MATCHED_SUBJECT_SHIFT_V2_GPU"
OFFICIAL_PREFLIGHT_STATUS = "CPU_PREFLIGHT_PASSED_AWAITING_ROOT_GO"
OFFICIAL_PREFLIGHT_KIND = "a2_matched_subject_shift_v2_official_preflight"
CELL_LAUNCH_KIND = "a2_matched_subject_shift_v2_cell_launch_environment"

# These are the complete executable inputs whose source bytes determine an A2
# v2 launch.  The official CPU preflight snapshots this map, and every launch,
# scorer, and aggregate refuses to continue if any one of the paths or SHA-256
# values has changed.  Keep this list explicit rather than globbing: an
# accidental file addition must never silently become a scientific authority.
IMPLEMENTATION_BINDING_PATHS: Mapping[str, Path] = {
    "config": CONFIG_PATH,
    "contract": CONTRACT_PATH,
    "core": Path(__file__).resolve(),
    "preflight": SUA_ROOT / "scripts" / "a2_matched_subject_shift_v2_preflight.py",
    "scorer": SUA_ROOT / "scripts" / "a2_matched_subject_shift_v2_score.py",
    "runner": SUA_ROOT / "scripts" / "run_a2_matched_subject_shift_v2_one_cell.sh",
    "aggregator": SUA_ROOT / "scripts" / "aggregate_a2_matched_subject_shift_v2.py",
    "trainer": SUA_ROOT / "scripts" / "train_variant_dandi688.py",
    # The A2 scorer imports these shared evaluation primitives directly.
    "shared_evaluator": SUA_ROOT / "scripts" / "eval_adaptation_dandi688.py",
    "frozen_model_loader": SUA_ROOT / "scripts" / "select_gradient_free_protocol_dandi688.py",
    # Retain the existing within-development evaluator as an independently
    # bound comparator surface; it is part of the development-score lineage.
    "within_development_evaluator": SUA_ROOT / "scripts" / "eval_epoch_window_generic_dandi688.py",
    "datamodule": SUA_ROOT / "mc_maze" / "multisession_datamodule.py",
    "session_datamodule": SUA_ROOT / "mc_maze" / "datamodule.py",
    "unit_side_features": SUA_ROOT / "mc_maze" / "unit_side_features.py",
    "gradient_free_protocol": SUA_ROOT / "scripts" / "dandi688_gradient_free_protocol.py",
    # Direct runtime imports of the trainer/evaluator.  Binding only their
    # outer scripts is insufficient: these files define the actual B3S
    # encoder/decoder, loss, attention, and run-directory semantics.
    "streaming_calibration_module": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "streaming_calibration_module.py",
    "t4_logit_residual_module": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "t4_logit_residual_module.py",
    "run_artifacts": REPO_ROOT / "streaming_calibration_exp" / "src" / "metrics" / "run_artifacts.py",
    "neuron_dropout": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "neuron_dropout.py",
    "carrier_noise_augmentation": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "carrier_noise_augmentation.py",
    "correspondence_breaking": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "correspondence_breaking.py",
    "spint_model": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "spint.py",
    "streaming_encoders": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_encoders.py",
    "streaming_spint": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_spint.py",
    "rt_ld_gain": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "rt_ld_gain.py",
    "t4_logit_residual_adapter": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "components" / "streaming_spint_t4_logit_residual_adapter.py",
    "falcon_module": REPO_ROOT / "streaming_calibration_exp" / "src" / "models" / "falcon_module.py",
}


class A2V2ContractError(ValueError):
    """A frozen A2 v2 invariant has drifted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise A2V2ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly_regular_file(path: Path) -> bool:
    """Return true only for a non-symlink regular file mode 0444."""
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444


def immutable_sidecar_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.sha256")


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    """Write one JSON body and SHA sidecar using O_EXCL, fsync, then chmod 0444.

    A crash can leave an orphaned body before its sidecar, but it can never
    create a valid artifact accidentally: all readers require both immutable
    files and verify their exact body hash.  Deliberately do not clean up an
    orphan, because doing so would turn a failed scientific write into an
    overwrite-capable path.
    """
    requested_path = Path(path).expanduser()
    # Receipts are never addressed through a final-component symlink.  The
    # project paths below are all ordinary paths, and rejecting this unusual
    # form prevents a caller from making an immutable receipt appear to have
    # been written at one location while its bytes live at another.
    require(not requested_path.is_symlink(), f"immutable receipt path may not be a symlink: {requested_path}")
    body_path = requested_path.resolve()
    sidecar_path = immutable_sidecar_path(body_path)
    if os.path.lexists(body_path) or os.path.lexists(sidecar_path):
        raise FileExistsError(f"refusing immutable overwrite: {body_path}")
    body_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pretty_json_bytes(dict(payload))
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
    # File fsync makes the bytes durable; fsyncing the parent also persists the
    # O_EXCL directory entries and chmod metadata across a crash.
    try:
        directory_fd = os.open(body_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise A2V2ContractError(f"cannot fsync immutable receipt directory: {body_path.parent}") from exc
    return body_path, sidecar_path, digest


def load_verified_immutable_json(path: Path, *, label: str = "immutable JSON") -> tuple[dict[str, Any], str]:
    """Load a body only after its strict SHA sidecar and both 0444 modes pass."""
    requested_path = Path(path).expanduser()
    require(not requested_path.is_symlink(), f"{label} path may not be a symlink: {requested_path}")
    body_path = requested_path.resolve()
    sidecar_path = immutable_sidecar_path(body_path)
    require(_readonly_regular_file(body_path), f"{label} is not immutable mode 0444: {body_path}")
    require(_readonly_regular_file(sidecar_path), f"{label} sidecar is not immutable mode 0444: {sidecar_path}")
    try:
        sidecar = sidecar_path.read_text(encoding="ascii")
    except OSError as exc:
        raise A2V2ContractError(f"cannot read {label} sidecar: {sidecar_path}") from exc
    observed = sha256_file(body_path)
    expected_sidecar = f"{observed}  {body_path.name}\n"
    require(sidecar == expected_sidecar, f"{label} SHA sidecar/body integrity mismatch: {body_path}")
    return load_json_object(body_path), observed


def current_implementation_bindings() -> dict[str, dict[str, str]]:
    """Return the exact path/SHA snapshot that an official preflight seals."""
    bindings: dict[str, dict[str, str]] = {}
    for name, path in IMPLEMENTATION_BINDING_PATHS.items():
        resolved = Path(path).expanduser().resolve()
        require(resolved.is_file(), f"missing implementation binding {name}: {resolved}")
        bindings[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return bindings


def implementation_bindings_sha256(bindings: Mapping[str, Any]) -> str:
    return canonical_json_sha256(bindings)


def verify_implementation_bindings(bindings: Any) -> dict[str, dict[str, str]]:
    """Fail closed if an official snapshot differs from the current code bytes."""
    require(isinstance(bindings, Mapping), "implementation bindings missing")
    expected_names = set(IMPLEMENTATION_BINDING_PATHS)
    require(set(bindings) == expected_names, "implementation binding key set drift")
    normalized: dict[str, dict[str, str]] = {}
    for name in IMPLEMENTATION_BINDING_PATHS:
        row = bindings[name]
        require(isinstance(row, Mapping), f"implementation binding malformed: {name}")
        path = row.get("path")
        digest = row.get("sha256")
        require(isinstance(path, str) and isinstance(digest, str), f"implementation binding fields malformed: {name}")
        normalized[name] = {"path": path, "sha256": digest}
    current = current_implementation_bindings()
    require(normalized == current, "implementation binding SHA/path drift since official preflight")
    return normalized


def python_isolation_binding() -> dict[str, Any]:
    """Capture the interpreter isolation that prevents user-site Torch leakage."""
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 is required")
    require(site.ENABLE_USER_SITE is False, "Python user site remains enabled despite PYTHONNOUSERSITE=1")
    return {
        "PYTHONNOUSERSITE": "1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "user_site_enabled": False,
    }


def verify_python_isolation_binding(binding: Any) -> dict[str, Any]:
    require(isinstance(binding, Mapping), "Python isolation binding missing")
    current = python_isolation_binding()
    require(dict(binding) == current, "Python isolation interpreter/environment drift")
    return current


def validate_official_preflight_payload(
    payload: Mapping[str, Any],
    *,
    result_root: Path,
) -> dict[str, Any]:
    """Validate the one, non-authorizing preflight trusted by all six cells.

    This intentionally does *not* repeat a whole-result-root freshness scan.
    Once the immutable official preflight is minted, a later cell must be able
    to run beside earlier cells.  The runner calls :func:`assert_cell_fresh`
    for its own arm/seed instead.
    """
    require(payload.get("schema_version") == 3, "official preflight schema drift")
    require(payload.get("receipt_kind") == OFFICIAL_PREFLIGHT_KIND, "not an official A2 v2 preflight")
    require(payload.get("status") == OFFICIAL_PREFLIGHT_STATUS, "official preflight is not a non-authorizing passed status")
    require(payload.get("official_preflight") is True, "official preflight flag missing")
    require(payload.get("non_authorizing_status") is True, "official preflight became authorizing")
    require(payload.get("root_go_required_before_gpu") is True, "official preflight root-GO gate missing")
    require(payload.get("cpu_only") is True and payload.get("gpu_used") is False,
            "official preflight must remain CPU-only")
    require(payload.get("training_started") is False and payload.get("checkpoint_loaded") is False,
            "official preflight cannot include training/checkpoint activity")
    require(payload.get("formal_subc_test_nwb_opened") is False, "official preflight opened sealed formal test")
    require(payload.get("screen_id") == SCREEN_ID, "official preflight screen id drift")
    require(payload.get("result_root") == str(Path(result_root).expanduser().resolve()),
            "official preflight result-root drift")
    require(payload.get("expected_fresh_gpu_cells") == len(SOURCE_ARMS) * len(SEEDS),
            "official preflight fresh GPU-cell count drift")
    require(payload.get("forbidden_duplicate_domain_training_cells") == len(SOURCE_ARMS) * len(SEEDS) * len(DOMAINS),
            "official preflight domain-retraining prohibition drift")
    require(payload.get("source_training_cells") == list(SOURCE_ARMS), "official preflight source-cell roster drift")
    require(payload.get("seeds") == list(SEEDS), "official preflight seed roster drift")
    require(payload.get("scoring_domains") == list(DOMAINS), "official preflight scoring-domain roster drift")
    require(payload.get("query_policy") == frozen_query_policy(), "official preflight M30/trial-30 query policy drift")
    require(payload.get("config_validated") is True, "official preflight did not validate frozen config")
    for key, expected_count in (
        ("within_subject_audit", WITHIN_SESSION_COUNT),
        ("external_subject_M_audit", EXTERNAL_SESSION_COUNT),
    ):
        audit = payload.get(key)
        require(isinstance(audit, Mapping), f"official preflight {key} missing")
        require(audit.get("expected_count") == expected_count, f"official preflight {key} expected count drift")
        require(audit.get("admissible_count") == expected_count, f"official preflight {key} admissible count drift")
        rows = audit.get("sessions")
        require(isinstance(rows, list) and len(rows) == expected_count, f"official preflight {key} session audit drift")
        require(all(isinstance(row, Mapping) and row.get("admissible") is True for row in rows),
                f"official preflight {key} contains an inadmissible session")
    normalizer = payload.get("normalizer_authority")
    require(isinstance(normalizer, Mapping), "official preflight normalizer authority missing")
    require(normalizer.get("policy") == frozen_normalizer_policy(), "official preflight normalizer policy drift")
    require(normalizer.get("source_train_sessions") == load_strict_manifest()["train"],
            "official preflight normalizer source roster drift")
    require(normalizer.get("source_train_session_count") == 27,
            "official preflight normalizer source cardinality drift")
    # The official receipt is minted before any target session is scored.  It
    # seals the source-only authority and the refit prohibition; the explicit
    # score-time `*_performed` provenance belongs to each domain receipt.
    require(normalizer.get("target_domain_normalizer_refit_forbidden") is True,
            "official preflight did not prohibit target normalizer refit")
    require(payload.get("implementation_blockers") == [], "official preflight contains blockers")
    bindings = verify_implementation_bindings(payload.get("implementation_bindings"))
    require(payload.get("implementation_bindings_sha256") == implementation_bindings_sha256(bindings),
            "official preflight implementation binding digest drift")
    verify_python_isolation_binding(payload.get("python_isolation"))
    require(payload.get("contract_sha256") == sha256_file(CONTRACT_PATH), "official preflight contract SHA drift")
    require(payload.get("config_sha256") == sha256_file(CONFIG_PATH), "official preflight config SHA drift")
    return dict(payload)


def load_verified_official_preflight(path: Path, *, result_root: Path) -> tuple[dict[str, Any], str]:
    canonical_path = official_preflight_path(result_root=result_root).expanduser().resolve()
    require(
        Path(path).expanduser().resolve() == canonical_path,
        f"official preflight must use its one canonical path: {canonical_path}",
    )
    payload, digest = load_verified_immutable_json(path, label="official A2 v2 preflight")
    return validate_official_preflight_payload(payload, result_root=result_root), digest


def assert_cell_fresh(*, source_arm: str, seed: int, result_root: Path) -> dict[str, str]:
    """Reject only the candidate arm/seed's own outputs after global preflight.

    Earlier A2 cells are deliberately ignored.  This is the companion to the
    one-time root audit and prevents cell 2--6 from being blocked by cell 1.
    """
    source_arm_for_name(source_arm)
    require(seed in SEEDS, f"unknown A2 seed: {seed}")
    root = Path(result_root).expanduser().resolve()
    targets = {
        "source_run_dir": source_run_dir(source_arm, seed),
        "cell_launch_receipt": cell_launch_receipt_path(source_arm, seed, result_root=root),
        "within_subject_receipt": domain_result_path(source_arm, seed, "within_subject", result_root=root),
        "external_subject_M_receipt": domain_result_path(source_arm, seed, "external_subject_M", result_root=root),
        "trainer_global_summary": source_summary_path(source_arm, seed),
    }
    occupied: list[str] = []
    for label, target in targets.items():
        if target.exists() or immutable_sidecar_path(target).exists():
            occupied.append(f"{label}={target}")
    require(not occupied, "candidate A2 source cell is not fresh: " + "; ".join(occupied))
    return {name: str(path) for name, path in targets.items()}


def torch_runtime_binding(*, visible_device_index: int = 0) -> dict[str, Any]:
    """Verify the isolated conda Torch/CUDA runtime immediately before training.

    This deliberately lives behind an explicit launch-only call.  CPU preflight
    never imports Torch or asks CUDA for a device, while an authorized launcher
    records the exact Torch origin/version and visible RTX device before any
    trainer code can run.
    """
    isolation = python_isolation_binding()
    import torch

    torch_path = Path(str(torch.__file__)).resolve()
    prefix = Path(sys.prefix).resolve()
    try:
        torch_path.relative_to(prefix)
    except ValueError as exc:
        raise A2V2ContractError(f"Torch is outside the active environment prefix: {torch_path}") from exc
    version = str(torch.__version__)
    cuda_build = str(torch.version.cuda)
    require(version.startswith("2.5.1.post303"), f"unexpected Torch version: {version}")
    require(cuda_build == "11.8", f"unexpected Torch CUDA build: {cuda_build}")
    require(torch.cuda.is_available(), "isolated Torch reports CUDA unavailable")
    require(torch.cuda.device_count() > visible_device_index, "requested visible CUDA device is unavailable")
    device = torch.device(f"cuda:{visible_device_index}")
    properties = torch.cuda.get_device_properties(device)
    name = str(properties.name)
    require(name, "CUDA device name is empty")
    require("RTX 3090" in name, f"unexpected launch GPU; A2 v2 is pinned to an RTX 3090 class device, got {name!r}")
    return {
        "python_isolation": isolation,
        "torch_path": str(torch_path),
        "torch_version": version,
        "torch_cuda_build": cuda_build,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_index": visible_device_index,
        "device": str(device),
        "device_name": name,
        "device_capability": list(torch.cuda.get_device_capability(device)),
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    """Human-readable immutable JSON bytes (the SHA covers these exact bytes)."""
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _float32_normalizer_sha256(mean: Any, std: Any) -> str:
    """Hash numeric normalizer values with the project's float32 semantics.

    NumPy is intentionally imported only by a caller that is already doing an
    explicit data audit or score.  The helper works with ndarray-like values.
    """
    import numpy as np

    payload = {
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "std": np.asarray(std, dtype=np.float32).tolist(),
    }
    # Keep byte-for-byte compatibility with
    # unit_side_features.side_feature_stats_sha256(), which is the digest
    # train_variant_dandi688.py records in source run_metadata.json.  In
    # particular, that established provenance hash intentionally has no
    # trailing newline (unlike immutable JSON receipts in this module).
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalizer_value_sha256(mean: Any, std: Any) -> str:
    return _float32_normalizer_sha256(mean, std)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A2V2ContractError(f"cannot parse JSON object: {path}") from exc
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def source_arm_for_name(source_arm: str) -> Mapping[str, str]:
    try:
        return SOURCE_ARMS[source_arm]
    except KeyError as exc:
        raise A2V2ContractError(f"unsupported source arm: {source_arm!r}") from exc


def source_run_name(source_arm: str, seed: int) -> str:
    source_arm_for_name(source_arm)
    require(seed in SEEDS, f"seed must be one of {SEEDS}, got {seed}")
    return f"{SCREEN_ID}_{source_arm}_dandi688_co_s{seed}"


def source_run_dir(source_arm: str, seed: int) -> Path:
    return CHECKPOINT_ROOT / source_run_name(source_arm, seed)


def source_summary_path(source_arm: str, seed: int) -> Path:
    """Global summary path written unconditionally by the shared trainer."""
    return SUA_ROOT / "results" / f"p3_{source_run_name(source_arm, seed)}_seed{seed}.json"


def domain_result_path(source_arm: str, seed: int, domain: str, *, result_root: Path = RESULT_ROOT) -> Path:
    source_arm_for_name(source_arm)
    require(seed in SEEDS, f"seed must be one of {SEEDS}, got {seed}")
    require(domain in DOMAINS, f"unsupported domain: {domain!r}")
    return Path(result_root) / f"{domain}_{source_arm}_s{seed}.json"


def official_preflight_path(*, result_root: Path = RESULT_ROOT) -> Path:
    """The one immutable CPU preflight shared by the complete six-cell matrix."""
    return Path(result_root) / "official_cpu_preflight.json"


def cell_launch_receipt_path(source_arm: str, seed: int, *, result_root: Path = RESULT_ROOT) -> Path:
    source_arm_for_name(source_arm)
    require(seed in SEEDS, f"seed must be one of {SEEDS}, got {seed}")
    return Path(result_root) / f"cell_launch_{source_arm}_s{seed}.json"


def source_epoch_checkpoint_paths(run_dir: Path) -> dict[int, Path]:
    run_dir = Path(run_dir).expanduser().resolve()
    paths = {epoch: run_dir / "epoch_ckpts" / f"epoch_{epoch - 1:03d}.ckpt" for epoch in EPOCH_WINDOW}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    require(not missing, f"missing source epoch checkpoints: {missing}")
    return paths


def source_checkpoint_sha256_bundle(run_dir: Path) -> dict[str, str]:
    paths = source_epoch_checkpoint_paths(run_dir)
    return {str(epoch): sha256_file(path) for epoch, path in paths.items()}


def validate_checkpoint_sha256_bundle(bundle: Any, *, label: str = "checkpoint bundle") -> dict[str, str]:
    require(isinstance(bundle, Mapping), f"{label}: expected object")
    expected = {str(epoch) for epoch in EPOCH_WINDOW}
    require(set(bundle) == expected, f"{label}: expected epoch keys {sorted(expected)}")
    normalized: dict[str, str] = {}
    for epoch in EPOCH_WINDOW:
        value = bundle[str(epoch)]
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                f"{label}: invalid SHA-256 for epoch {epoch}")
        normalized[str(epoch)] = value
    return normalized


def load_strict_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, list[str]]:
    payload = load_json_object(manifest_path)
    require(payload.get("schema_version") == 1, "strict source manifest schema drift")
    splits = payload.get("session_splits")
    require(isinstance(splits, Mapping), "strict source manifest lacks session_splits")
    expected_counts = {"train": 27, "val": 6, "test": 6}
    normalized: dict[str, list[str]] = {}
    for key, count in expected_counts.items():
        rows = splits.get(key)
        require(isinstance(rows, list) and len(rows) == count and all(isinstance(row, str) for row in rows),
                f"strict source manifest {key} roster drift")
        require(len(set(rows)) == count, f"strict source manifest {key} contains duplicate sessions")
        normalized[key] = list(rows)
    require(len(set(normalized["train"] + normalized["val"] + normalized["test"])) == 39,
            "strict source manifest rosters are not disjoint")
    return normalized


def expected_within_sessions(manifest_path: Path = MANIFEST_PATH) -> tuple[str, ...]:
    from mc_maze.gpu_contract_common import SUBC_VAL_SESSIONS

    manifest = load_strict_manifest(manifest_path)
    observed = tuple(manifest["val"])
    require(observed == tuple(SUBC_VAL_SESSIONS), "strict source validation roster differs from shared A2 roster")
    return observed


def expected_external_sessions() -> tuple[str, ...]:
    from mc_maze.gpu_contract_common import SUBM_EXTERNAL_SESSIONS, assert_sessions_not_sealed

    sessions = tuple(SUBM_EXTERNAL_SESSIONS)
    require(len(sessions) == EXTERNAL_SESSION_COUNT and len(set(sessions)) == EXTERNAL_SESSION_COUNT,
            "external sub-M roster must contain exactly 15 unique sessions")
    assert_sessions_not_sealed(sessions, label="A2 v2 external sub-M roster")
    return sessions


def expected_domain_sessions(domain: str, manifest_path: Path = MANIFEST_PATH) -> tuple[str, ...]:
    if domain == "within_subject":
        return expected_within_sessions(manifest_path)
    if domain == "external_subject_M":
        return expected_external_sessions()
    raise A2V2ContractError(f"unsupported domain: {domain!r}")


def active_source_session_paths(manifest_path: Path = MANIFEST_PATH, data_root: Path = SUBC_DATA_ROOT) -> tuple[list[Path], list[Path], list[str]]:
    """Resolve only train/validation NWBs; formal-test names stay inert strings."""
    manifest = load_strict_manifest(manifest_path)
    root = Path(data_root).expanduser().resolve()
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for split, result in (("train", train_paths), ("val", val_paths)):
        for session in manifest[split]:
            path = (root / f"{session}_behavior+ecephys.nwb").resolve()
            require(path.parent == root and path.is_file(), f"missing source {split} NWB: {path}")
            result.append(path)
    # Deliberately do not turn the six formal-test strings into paths here.
    return train_paths, val_paths, list(manifest["test"])


def external_session_paths(data_root: Path = SUBM_DATA_ROOT) -> list[Path]:
    root = Path(data_root).expanduser().resolve()
    paths: list[Path] = []
    for session in expected_external_sessions():
        path = (root / f"{session}_behavior+ecephys.nwb").resolve()
        require(path.parent == root and path.is_file(), f"missing frozen external sub-M NWB: {path}")
        paths.append(path)
    return paths


def frozen_query_policy() -> dict[str, Any]:
    """Matched M30 policy with explicit target-session carrier provenance.

    The first 30 rewarded target-direction labels construct each scored
    session's carrier (including Z4 before its standardized-T4 mask).  That is
    deliberately distinct from fitting a target normalizer or adapting the
    frozen decoder, both of which remain forbidden below.
    """
    return {
        "activity_calibration_trials": ACTIVITY_CALIBRATION_TRIALS,
        "side_feature_label_pool_trials": SIDE_FEATURE_POOL_TRIALS,
        "selection_mode": "first",
        "evaluation_start_trial_index": EVALUATION_START_TRIAL_INDEX,
        "window_size_bins": WINDOW_SIZE_BINS,
        "trial_length_bins": TRIAL_LENGTH_BINS,
        "bin_size_ms": BIN_SIZE_MS,
        "trial_result_filter": "R",
        "query_rule": (
            "usable rewarded trials[30:] only; each 50-bin query window is contained in a "
            "trial strictly after chronological rewarded trial 30"
        ),
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_velocity_labels_used_for_weight_updates": False,
        "backward_gradients": False,
        "decoder_weight_updates": False,
    }


def frozen_normalizer_policy() -> dict[str, Any]:
    return {
        "behavior_authority": "strict_subc_source_train_27_only",
        "side_feature_authority": "strict_subc_source_train_27_only",
        "behavior_fit_scope": "source train sessions only; binned cursor velocity at 20 ms",
        "side_feature_fit_scope": "source train sessions only; raw T4 statistics over chronological first 30 rewarded trials",
        "target_domain_normalizer_refit_performed": False,
        "target_domain_normalizer_refit_forbidden": True,
    }


def trial30_semantics_from_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    require_target_labels: bool = True,
    session: str = "session",
) -> dict[str, Any]:
    """Validate the common M30/trial-30 query contract without scoring anything."""
    require(len(trials) > EVALUATION_START_TRIAL_INDEX,
            f"{session}: need more than 30 usable rewarded trials for a post-30 query")
    support = list(trials[:ACTIVITY_CALIBRATION_TRIALS])
    query_trials = list(trials[EVALUATION_START_TRIAL_INDEX:])
    require(len(support) == ACTIVITY_CALIBRATION_TRIALS, f"{session}: M30 support cardinality drift")
    if require_target_labels:
        missing = [index for index, row in enumerate(support) if not _finite_target_direction(row.get("target_dir"))]
        require(not missing, f"{session}: first-30 T4 label pool has missing/non-finite target_dir at {missing}")
    query_window_count = 0
    for index, row in enumerate(query_trials, start=EVALUATION_START_TRIAL_INDEX):
        try:
            start = int(row["start"])
            stop = int(row["stop"])
        except (KeyError, TypeError, ValueError) as exc:
            raise A2V2ContractError(f"{session}: invalid start/stop at usable rewarded trial {index}") from exc
        query_window_count += max(0, stop - start - WINDOW_SIZE_BINS + 1)
    require(query_window_count > 0, f"{session}: no 50-bin query windows strictly after trial 30")
    original_trial_indices: list[int | None] = []
    for row in support:
        value = row.get("trial_index")
        original_trial_indices.append(int(value) if value is not None else None)
    return {
        "usable_rewarded_trial_count": len(trials),
        "activity_support_usable_indices": list(range(ACTIVITY_CALIBRATION_TRIALS)),
        "activity_support_original_trial_indices": original_trial_indices,
        "side_feature_label_pool_usable_indices": list(range(SIDE_FEATURE_POOL_TRIALS)),
        "first30_target_dir_all_finite": bool(require_target_labels),
        "query_usable_trial_indices_start": EVALUATION_START_TRIAL_INDEX,
        "query_usable_trial_count": len(query_trials),
        "post30_query_window_count": query_window_count,
    }


def _finite_target_direction(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    payload = load_json_object(config_path)
    require(payload.get("schema_version") == 2, "A2 v2 config schema drift")
    require(payload.get("screen_id") == SCREEN_ID, "A2 v2 config screen id drift")
    source_rows = payload.get("source_training_cells")
    require(isinstance(source_rows, list) and len(source_rows) == len(SOURCE_ARMS), "A2 v2 source-arm topology drift")
    observed_arms = {
        str(row.get("name")): {
            "arm": row.get("arm"),
            "variant": row.get("variant"),
            "side_feature_group": row.get("side_feature_group"),
        }
        for row in source_rows
        if isinstance(row, Mapping)
    }
    require(observed_arms == {key: dict(value) for key, value in SOURCE_ARMS.items()}, "A2 v2 source-arm mapping drift")
    require(tuple(payload.get("seeds") or ()) == SEEDS, "A2 v2 seed roster drift")
    domains = payload.get("scoring_domains")
    require(isinstance(domains, list) and tuple(row.get("name") for row in domains if isinstance(row, Mapping)) == DOMAINS,
            "A2 v2 scoring-domain topology drift")
    expected_protocol = frozen_query_policy()
    observed_protocol = payload.get("frozen_protocol")
    require(isinstance(observed_protocol, Mapping), "A2 v2 missing frozen_protocol")
    for key in (
        "activity_calibration_trials", "side_feature_label_pool_trials", "evaluation_start_trial_index",
        "window_size_bins", "trial_length_bins", "bin_size_ms", "selection_mode", "epochs", "epoch_window",
        "loss_mode", "identity_mode", "decoder_mode", "learning_rate", "batch_size", "chronological_calibration",
        "no_early_stopping", "checkpoint_every_epoch",
    ):
        expected = {
            "activity_calibration_trials": ACTIVITY_CALIBRATION_TRIALS,
            "side_feature_label_pool_trials": SIDE_FEATURE_POOL_TRIALS,
            "evaluation_start_trial_index": EVALUATION_START_TRIAL_INDEX,
            "window_size_bins": WINDOW_SIZE_BINS,
            "trial_length_bins": TRIAL_LENGTH_BINS,
            "bin_size_ms": BIN_SIZE_MS,
            "selection_mode": "first",
            "epochs": TOTAL_EPOCHS,
            "epoch_window": list(EPOCH_WINDOW),
            "loss_mode": "task_only",
            "identity_mode": "calibrated",
            "decoder_mode": "coupled",
            "learning_rate": 1e-4,
            "batch_size": 32,
            "chronological_calibration": True,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
        }[key]
        require(observed_protocol.get(key) == expected, f"A2 v2 frozen protocol drift: {key}")
    require(payload.get("checkpoint_reuse", {}).get("one_source_run_per_arm_seed") is True,
            "A2 v2 checkpoint reuse must be required")
    require(payload.get("normalizer_authority", {}).get("target_fit_forbidden") is True,
            "A2 v2 target normalizer fit must be forbidden")
    execution = payload.get("execution")
    require(isinstance(execution, Mapping), "A2 v2 execution receipt policy missing")
    expected_execution = {
        "fresh_gpu_training_cells": len(SOURCE_ARMS) * len(SEEDS),
        "gpu_authorization_environment": GPU_AUTH_ENV,
        "gpu_authorization_value": GPU_AUTH_VALUE,
        "root_go_environment": "A2_V2_ROOT_GO",
        "root_go_value": "I_SIGN_A2_MATCHED_SUBJECT_SHIFT_V2_ROOT_GO",
        "preflight_receipt_required_for_launch": True,
        "one_official_immutable_preflight_for_all_six_cells": True,
        "official_preflight_path": "results/a2_matched_subject_shift_v2/official_cpu_preflight.json",
        "python_no_user_site_required": True,
    }
    for key, expected in expected_execution.items():
        require(execution.get(key) == expected, f"A2 v2 execution policy drift: {key}")
    return payload


def validate_source_run_metadata(
    metadata: Mapping[str, Any],
    *,
    source_arm: str,
    seed: int,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    """Require that a completed source run matches every frozen training control."""
    arm = source_arm_for_name(source_arm)
    require(metadata.get("status") == "completed", "source run is not completed")
    require(metadata.get("variant") == arm["variant"], "source run variant drift")
    require(metadata.get("seed") == seed, "source run seed drift")
    require(metadata.get("task") == "CO", "source run task drift")
    require(metadata.get("signal_view", "sua") == "sua", "source run signal view drift")
    require(metadata.get("split_counts") == [27, 6, 6], "source run split count drift")
    require(metadata.get("max_units_exclusive") == 100, "source run unit cap drift")
    require(metadata.get("held_out_test_evaluated") is False, "source run opened a formal test")
    require(metadata.get("train_val_manifest") == str(Path(manifest_path).resolve()), "source run manifest path drift")
    require(metadata.get("train_val_manifest_sha256") == sha256_file(manifest_path), "source run manifest SHA drift")
    source_data = Path(str(metadata.get("data_dir", ""))).expanduser().resolve()
    require(source_data == SUBC_DATA_ROOT.resolve(), "source run data root drift")
    require(metadata.get("teacher_sha256") == EXPECTED_TEACHER_SHA256, "source run teacher SHA drift")
    side = metadata.get("side_features") or {}
    require(side.get("group") == arm["side_feature_group"], "source run side-feature arm drift")
    require(side.get("pool_size") == SIDE_FEATURE_POOL_TRIALS, "source run T4 label-pool drift")
    require(side.get("normalization_base_feature_group") == "t4",
            "source run side-normalizer substrate drift")
    require(isinstance(side.get("normalization_sha256"), str) and len(side["normalization_sha256"]) == 64,
            "source run side normalizer provenance missing")
    session_splits = metadata.get("session_splits") or {}
    manifest = load_strict_manifest(manifest_path)
    require(session_splits.get("train") == manifest["train"], "source run train-session roster drift")
    require(session_splits.get("val") == manifest["val"], "source run validation-session roster drift")
    require(session_splits.get("test") == manifest["test"], "source run formal-test names receipt drift")
    training = metadata.get("training") or {}
    expected_training = {
        "max_epochs": TOTAL_EPOCHS,
        "no_early_stopping": True,
        "checkpoint_every_epoch": True,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "window_size": WINDOW_SIZE_BINS,
        "calibration_n_trials": ACTIVITY_CALIBRATION_TRIALS,
        "random_calibration": False,
        "calibration_selection": "chronological_first_n",
        "trial_length": TRIAL_LENGTH_BINS,
        "bin_size_ms": BIN_SIZE_MS,
        "loss_mode": "task_only",
        "identity_mode": "calibrated",
    }
    for key, expected in expected_training.items():
        require(training.get(key) == expected, f"source run training drift: {key}")
    decoder = metadata.get("decoder_architecture") or {}
    require(decoder.get("mode") == "coupled", "source run decoder-mode drift")
    validation = metadata.get("validation_protocol") or {}
    require(validation.get("calibration_trials") == "trials[0:calibration_n_trials]", "source run M30 calibration declaration drift")
    require(validation.get("evaluation_windows") == "trials[calibration_n_trials:] only", "source run post-M30 query declaration drift")


def source_run_receipt_fingerprint(
    run_dir: Path,
    *,
    source_arm: str,
    seed: int,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    require(metadata_path.is_file(), f"source run metadata missing: {metadata_path}")
    metadata = load_json_object(metadata_path)
    validate_source_run_metadata(metadata, source_arm=source_arm, seed=seed, manifest_path=manifest_path)
    bundle = source_checkpoint_sha256_bundle(run_dir)
    return {
        "source_run_dir": str(run_dir),
        "source_run_metadata_path": str(metadata_path),
        "source_run_metadata_sha256": sha256_file(metadata_path),
        "source_checkpoint_sha256_bundle": bundle,
        "source_checkpoint_sha256_bundle_sha256": canonical_json_sha256(bundle),
    }


def receipt_protocol_invariant(receipt: Mapping[str, Any]) -> None:
    require(receipt.get("query_policy") == frozen_query_policy(), "receipt query-policy drift")
    require(receipt.get("target_session_carrier_fit_performed") is True,
            "receipt must declare target-session carrier construction")
    require(receipt.get("target_direction_labels_used_for_carrier") is True,
            "receipt must declare target direction labels used for carrier construction")
    require(receipt.get("target_velocity_labels_used_for_weight_updates") is False,
            "receipt indicates target velocity labels updated weights")
    require(receipt.get("backward_gradients") is False,
            "receipt indicates backward gradients during target score")
    require(receipt.get("decoder_weight_updates") is False,
            "receipt indicates decoder weight updates during target score")
    require(receipt.get("target_domain_normalizer_refit_performed") is False,
            "receipt indicates target-domain normalizer refit")
    authority = receipt.get("normalizer_authority")
    require(isinstance(authority, Mapping), "receipt normalizer authority missing")
    policy = authority.get("policy")
    require(policy == frozen_normalizer_policy(), "receipt normalizer-policy drift")
    require(authority.get("target_domain_normalizer_refit_performed") is False,
            "receipt normalizer authority indicates target refit")
    require(authority.get("target_domain_normalizer_refit_forbidden") is True,
            "receipt does not prohibit target normalizer refit")
    require(authority.get("source_train_session_count") == 27, "receipt source normalizer roster cardinality drift")
    sessions = authority.get("source_train_sessions")
    require(isinstance(sessions, list) and len(sessions) == 27 and len(set(sessions)) == 27,
            "receipt source normalizer roster drift")
    for key in ("behavior_normalizer_value_sha256", "side_normalizer_value_sha256"):
        value = authority.get(key)
        require(isinstance(value, str) and len(value) == 64, f"receipt missing {key}")
    for key in (
        "cached_training_path_behavior_normalizer_value_sha256",
        "cached_training_path_side_normalizer_value_sha256",
    ):
        value = authority.get(key)
        require(isinstance(value, str) and len(value) == 64, f"receipt missing {key}")
    require(authority.get("cached_vs_uncached_behavior_values_bitwise_identical") is True,
            "receipt behavior cached/uncached normalizer parity failed")
    require(authority.get("cached_vs_uncached_side_values_bitwise_identical") is True,
            "receipt side cached/uncached normalizer parity failed")
    require(authority.get("behavior_normalizer_value_sha256") == authority.get("cached_training_path_behavior_normalizer_value_sha256"),
            "receipt behavior cached/uncached digest mismatch")
    require(authority.get("side_normalizer_value_sha256") == authority.get("cached_training_path_side_normalizer_value_sha256"),
            "receipt side cached/uncached digest mismatch")
    require(authority.get("training_cache_root") == str(SOURCE_CACHE_ROOT.resolve()),
            "receipt training cache-root drift")
    for key in ("training_path_behavior_cache_sha256", "training_path_side_cache_sha256"):
        value = authority.get(key)
        require(isinstance(value, str) and len(value) == 64, f"receipt missing {key}")
