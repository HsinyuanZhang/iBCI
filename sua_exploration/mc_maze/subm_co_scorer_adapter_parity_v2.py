"""Shared observation-only C1 batch trace for future sub-M scorer-adapter parity.

Nothing in this module executes at import time.  The execution-facing functions
are intentionally unreachable from the v2 CLI until a later append-only,
root-reviewed authorization is supplied.  When that happens, both the C1
reference and a future adapter receive exactly the same prepared C1 batch and
use the one trace/metric/seal path below; no second data loader or metric loop
is allowed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VIEW = "sua"
SEED = 44
SUPPORT_TRIALS = 50
IDENTITY_TRIALS = 30
BIN_SIZE_MS = 20
HISTORY_BINS = 50
TRIAL_LENGTH_BINS = 100
PAD_VALUE = -1.0
LOADER_BATCH_SIZE = 128
TRACE_ROWS = 16
R2_ATOL = 1.0e-6
PARITY_RECEIPT_SCHEMA = "dandi_000688_subc_scorer_adapter_parity_receipt_v1"

CHECKPOINT = {
    "path": "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_t4_s44/epoch_ckpts/epoch_011.ckpt",
    "sha256": "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6",
    "bytes": 64769167,
}
TEACHER = {
    "path": "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt",
    "sha256": "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d",
    "bytes": 55195903,
}
DEV_SESSION = {
    "path": "sua_exploration/data/dandi_000688/sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb",
    "session": "sub-C_ses-CO-20151103",
    "sha256": "7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7",
    "bytes": 62145872,
}
NORMALIZERS = {
    "behavior": {
        "path": "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/sua/behavior_stats/be50f588491c004f721e.npz",
        "sha256": "821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd",
        "bytes": 397,
    },
    "t4": {
        "path": "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/sua/side_feature_stats/dd3da1f59700c1b96ab8.npz",
        "sha256": "32d32a7fe1b80a139571aae0ce3c3a1d802aec21c99b23a4cff72b8a60261701",
        "bytes": 614,
        "semantic_sha256": "ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7",
    },
}


class ParityError(RuntimeError):
    """Raised for a fixture, deterministic-trace, or seal-contract violation."""


class ParityAuthorizationError(ParityError):
    """Raised before any checkpoint or NWB access is allowed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ParityError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _tensor_digest(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    cpu = value.detach().to("cpu").contiguous()
    raw = cpu.numpy().tobytes(order="C")
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(torch.isfinite(cpu).all().item()),
    }


def _array_digest(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    raw = contiguous.tobytes(order="C")
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(np.isfinite(contiguous).all()),
    }


def _pin_path(root: Path, pin: Mapping[str, Any], *, label: str) -> Path:
    relative = str(pin["path"])
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ParityError(f"{label} path escapes repository: {relative}") from exc
    require(candidate.is_file() and not candidate.is_symlink(), f"missing or unsafe {label}: {relative}")
    require(candidate.stat().st_size == int(pin["bytes"]), f"{label} byte-size drift")
    require(sha256_file(candidate) == str(pin["sha256"]), f"{label} SHA-256 drift")
    return candidate


def _c1_runtime_modules() -> dict[str, Any]:
    """Import C1 owners only inside a future authorized execution path."""
    scripts = REPO_ROOT / "sua_exploration/scripts"
    import sys

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return {
        "eval": importlib.import_module("eval_adaptation_dandi688"),
        "selection": importlib.import_module("select_gradient_free_protocol_dandi688"),
        "side": importlib.import_module("mc_maze.unit_side_features"),
        "protocol": importlib.import_module("dandi688_gradient_free_protocol"),
    }


def _load_pinned_normalizer(root: Path, pin: Mapping[str, Any], *, label: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read a precomputed NPZ exactly; never invoke a fitting/cache-writing API."""
    path = _pin_path(root, pin, label=label)
    try:
        with np.load(path, allow_pickle=False) as bundle:
            mean = bundle["mean"].astype(np.float32, copy=False)
            std = bundle["std"].astype(np.float32, copy=False)
    except (KeyError, OSError, ValueError) as exc:
        raise ParityError(f"cannot load pinned {label} mean/std") from exc
    require(mean.ndim == 1 and std.ndim == 1 and mean.shape == std.shape, f"pinned {label} shape drift")
    require(np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 0), f"pinned {label} has invalid values")
    return mean, std, {"file": dict(pin), "mean": _array_digest(mean), "std": _array_digest(std)}


@dataclass(frozen=True)
class PreparedC1Batch:
    """The one prepared C1 batch passed unchanged to both trace labels."""

    neural: torch.Tensor
    behavior: torch.Tensor
    calibration: torch.Tensor
    side_features: torch.Tensor | None
    electrode_ids: torch.Tensor | None
    trial_trace: dict[str, Any]
    normalizer_trace: dict[str, Any]
    unit_row_trace: dict[str, Any]

    def input_trace(self) -> dict[str, Any]:
        tensors = {
            "neural": _tensor_digest(self.neural),
            "behavior": _tensor_digest(self.behavior),
            "calibration": _tensor_digest(self.calibration),
            "side_features": _tensor_digest(self.side_features),
            "electrode_ids": _tensor_digest(self.electrode_ids),
        }
        return {
            "trial_trace": self.trial_trace,
            "normalizer_trace": self.normalizer_trace,
            "unit_row_trace": self.unit_row_trace,
            "tensors": tensors,
            "input_sha256": hashlib.sha256(_canonical_bytes(tensors)).hexdigest(),
        }


class PreparedBatchAdapter(Protocol):
    """Future adapter boundary: it may only consume the already-prepared C1 batch."""

    def predict_raw(self, batch: PreparedC1Batch, *, device: torch.device) -> torch.Tensor:
        """Return unscaled raw decoder output with C1's [B,T,2] contract."""


@dataclass
class RuntimeCounters:
    checkpoint_hash_checks: int = 0
    checkpoint_load_calls: int = 0
    normalizer_hash_checks: int = 0
    normalizer_fit_calls: int = 0
    nwb_hash_checks: int = 0
    nwb_open_calls: int = 0
    model_forward_calls: int = 0
    torchmetrics_update_calls: int = 0
    torchmetrics_compute_calls: int = 0
    gpu_used: bool = False
    optimizer_or_backward_calls: int = 0
    subm_nwb_paths_constructed: bool = False
    subm_nwb_files_accessed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchTrace:
    label: str
    input_trace: dict[str, Any]
    raw_output_trace: dict[str, Any]
    prediction_trace: dict[str, Any]
    target_trace: dict[str, Any]
    r2: float
    torch_version: str
    torchmetrics_version: str
    predictions: np.ndarray = field(repr=False, compare=False)
    targets: np.ndarray = field(repr=False, compare=False)

    def json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "input_trace": self.input_trace,
            "raw_output_trace": self.raw_output_trace,
            "prediction_trace": self.prediction_trace,
            "target_trace": self.target_trace,
            "r2": self.r2,
            "torch_version": self.torch_version,
            "torchmetrics_version": self.torchmetrics_version,
        }


def _configure_cpu_determinism() -> torch.device:
    """Apply the predeclared CPU-only compatibility configuration."""
    require(not torch.cuda.is_initialized(), "parity execution refuses a pre-initialized CUDA runtime")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in {"", None}, "parity execution requires CUDA_VISIBLE_DEVICES to be empty")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch.device("cpu")


def prepare_fixed_c1_batch(root: Path, *, counters: RuntimeCounters) -> PreparedC1Batch:
    """Use C1's own data/session functions to make the fixed post-50 batch once.

    This function is execution-only.  It has no call path from a prelaunch or
    dry run, and direct NPZ loading makes a normalizer fit/cache write impossible.
    """
    root = root.resolve()
    runtime = _c1_runtime_modules()
    c1 = runtime["eval"]
    selection = runtime["selection"]
    side = runtime["side"]
    protocol = runtime["protocol"]
    session_path = _pin_path(root, DEV_SESSION, label="fixed consumed C1 development NWB")
    counters.nwb_hash_checks += 1
    counters.nwb_open_calls += 1
    behavior_mean, behavior_std, behavior_trace = _load_pinned_normalizer(root, NORMALIZERS["behavior"], label="behavior normalizer")
    side_mean, side_std, side_trace = _load_pinned_normalizer(root, NORMALIZERS["t4"], label="SUA T4 normalizer")
    counters.normalizer_hash_checks += 2
    record = c1.load_session_with_trials(
        session_path,
        BIN_SIZE_MS,
        HISTORY_BINS,
        SUPPORT_TRIALS,
        TRIAL_LENGTH_BINS,
        PAD_VALUE,
        behavior_mean,
        behavior_std,
        cache_dir=None,
        signal_view=VIEW,
    )
    require(record.get("name") == DEV_SESSION["session"] and record.get("signal_view") == VIEW, "fixed C1 session/view drift")
    record = c1.attach_side_features(
        record,
        session_path,
        side_feature_group="t4",
        waveform_feature_group=side.base_feature_group("t4"),
        pool_size=SUPPORT_TRIALS,
        permutation_seed=None,
        mean=side_mean,
        std=side_std,
        cache_dir=None,
    )
    trials = record["trials"]
    require(len(trials) > SUPPORT_TRIALS, "fixed fixture has no post50 query trial")
    chosen = protocol.select_calibration_trial_indices(trials, IDENTITY_TRIALS, SUPPORT_TRIALS, "first")
    require(chosen == list(range(IDENTITY_TRIALS)), "C1 first_n30 selection drift")
    record["calib_trials"] = c1.build_calib_trials_for_indices(record, chosen, IDENTITY_TRIALS)
    dataset = c1.make_subset_dataset(record, trials[SUPPORT_TRIALS:], record["name"])
    require(len(dataset) >= TRACE_ROWS, "fixed post50 dataset has fewer than 16 windows")
    loader = torch.utils.data.DataLoader(dataset, batch_size=LOADER_BATCH_SIZE, shuffle=False, num_workers=0)
    raw_batch = next(iter(loader))
    neural, behavior, calibration, side_features, electrode_ids = c1._unpack_loader_batch(raw_batch)
    require(neural.shape[0] >= TRACE_ROWS, "first fixed C1 DataLoader batch has fewer than 16 rows")
    slice16 = slice(0, TRACE_ROWS)
    prepared = PreparedC1Batch(
        neural=neural[slice16].contiguous(),
        behavior=behavior[slice16].contiguous(),
        calibration=calibration[slice16].contiguous(),
        side_features=None if side_features is None else side_features[slice16].contiguous(),
        electrode_ids=None if electrode_ids is None else electrode_ids[slice16].contiguous(),
        trial_trace={
            "rewarded_support": "trials[0:50]",
            "identity_selection": chosen,
            "query": "trials[50:]",
            "first_query_loader_batch": {"batch_size": LOADER_BATCH_SIZE, "shuffle": False, "num_workers": 0},
            "selected_rows": list(range(TRACE_ROWS)),
            "trial_indices_sha256": hashlib.sha256(json.dumps([item["trial_index"] for item in trials], separators=(",", ":")).encode()).hexdigest(),
        },
        normalizer_trace={"behavior": behavior_trace, "t4": side_trace, "fitting_calls": 0},
        unit_row_trace={
            "n_units": int(record["n_units"]),
            "identity_row_map": list(range(int(record["n_units"]))),
            "t4_side_feature_rows": _array_digest(np.asarray(record["side_features"], dtype=np.float32)),
        },
    )
    return prepared


def _reference_raw_output(model: Any, batch: PreparedC1Batch, *, device: torch.device) -> torch.Tensor:
    """The C1 `eval_r2` forward semantics, centralized once for observability."""
    c1 = _c1_runtime_modules()["eval"]
    neural = batch.neural.to(device)
    calibration = batch.calibration.to(device)
    side_features = None if batch.side_features is None else batch.side_features.to(device)
    electrode_ids = None if batch.electrode_ids is None else batch.electrode_ids.to(device)
    decoder_key_features = model.decoder_key_features(side_features)
    raw, _ = model.student(
        neural,
        calib_trials=calibration,
        side_features=side_features,
        decoder_key_features=decoder_key_features,
        electrode_ids=electrode_ids,
    )
    require(raw.ndim == 3 and raw.shape[0] == TRACE_ROWS and raw.shape[-1] == 2, "C1 raw decoder output shape drift")
    # The shared C1 owner performs the only last-bin/scaling operation.
    return raw


def _capture(label: str, batch: PreparedC1Batch, raw_output: torch.Tensor, *, counters: RuntimeCounters) -> BatchTrace:
    """The one observation/R² path used by C1 reference and future adapter traces."""
    c1 = _c1_runtime_modules()["eval"]
    metrics = importlib.import_module("torchmetrics.regression")
    scaled = c1.decode_last_behavior(raw_output)
    target = batch.behavior.to(raw_output.device)[:, -1:, :]
    require(scaled.shape == target.shape == (TRACE_ROWS, 1, 2), "prediction/target C1 last-bin shape drift")
    metric = metrics.R2Score(multioutput="variance_weighted").to(raw_output.device)
    metric.update(scaled.flatten(start_dim=0, end_dim=1), target.flatten(start_dim=0, end_dim=1))
    counters.torchmetrics_update_calls += 1
    value = float(metric.compute().detach().cpu().item())
    counters.torchmetrics_compute_calls += 1
    require(math.isfinite(value), "parity batch TorchMetrics R2 is nonfinite")
    predictions = scaled.detach().cpu().numpy().reshape(TRACE_ROWS, 2).astype(np.float32, copy=False)
    targets = target.detach().cpu().numpy().reshape(TRACE_ROWS, 2).astype(np.float32, copy=False)
    return BatchTrace(
        label=label,
        input_trace=batch.input_trace(),
        raw_output_trace=_tensor_digest(raw_output),
        prediction_trace=_array_digest(predictions),
        target_trace=_array_digest(targets),
        r2=value,
        torch_version=torch.__version__,
        torchmetrics_version=str(importlib.import_module("torchmetrics").__version__),
        predictions=predictions,
        targets=targets,
    )


def capture_c1_reference_trace(model: Any, batch: PreparedC1Batch, *, device: torch.device, counters: RuntimeCounters) -> BatchTrace:
    with torch.no_grad():
        raw = _reference_raw_output(model, batch, device=device)
    counters.model_forward_calls += 1
    return _capture("c1_reference", batch, raw, counters=counters)


def capture_future_adapter_trace(adapter: PreparedBatchAdapter, batch: PreparedC1Batch, *, device: torch.device, counters: RuntimeCounters) -> BatchTrace:
    """Capture a future adapter with the same prepared C1 inputs and output observer."""
    with torch.no_grad():
        raw = adapter.predict_raw(batch, device=device)
    require(isinstance(raw, torch.Tensor), "future adapter must return a Tensor raw decoder output")
    counters.model_forward_calls += 1
    return _capture("future_adapter", batch, raw, counters=counters)


def assert_trace_parity(reference: BatchTrace, adapter: BatchTrace) -> dict[str, Any]:
    require(reference.label == "c1_reference" and adapter.label == "future_adapter", "trace labels drift")
    require(reference.input_trace == adapter.input_trace, "adapter prepared inputs differ from C1 reference")
    require(reference.raw_output_trace["shape"] == adapter.raw_output_trace["shape"], "raw output shape differs")
    require(reference.target_trace == adapter.target_trace, "adapter target trace differs")
    require(np.array_equal(reference.predictions, adapter.predictions), "adapter prediction values differ exactly from C1 reference")
    require(np.array_equal(reference.targets, adapter.targets), "adapter target values differ exactly from C1 reference")
    delta = abs(reference.r2 - adapter.r2)
    require(delta <= R2_ATOL, f"adapter/reference R2 delta exceeds {R2_ATOL}: {delta}")
    return {
        "inputs_exact": True,
        "predictions_exact": True,
        "targets_exact": True,
        "r2_reference": reference.r2,
        "r2_adapter": adapter.r2,
        "r2_abs_delta": delta,
        "r2_atol": R2_ATOL,
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"failed to make immutable: {path.name}")
    return hashlib.sha256(payload).hexdigest()


def _write_immutable_npz(path: Path, *, predictions: np.ndarray, targets: np.ndarray) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        np.savez_compressed(handle, predictions=predictions, targets=targets)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"failed to make immutable: {path.name}")
    return sha256_file(path)


def seal_single_use_execution(output_root: Path, *, prelaunch_sha256: str, reference: BatchTrace, adapter: BatchTrace, counters: RuntimeCounters) -> dict[str, Any]:
    """Seal a future completed parity run once; never called by the current CLI."""
    output_root = output_root.resolve()
    require(not output_root.exists(), f"parity output root already exists: {output_root}")
    outcome = assert_trace_parity(reference, adapter)
    output_root.mkdir(parents=True, exist_ok=False)
    input_path = output_root / "input_trace.json"
    reference_trace_path = output_root / "c1_reference_trace.json"
    adapter_trace_path = output_root / "adapter_trace.json"
    reference_npz = output_root / "reference_prediction_target.npz"
    adapter_npz = output_root / "adapter_prediction_target.npz"
    input_sha = _write_immutable_json(input_path, reference.input_trace)
    reference_sha = _write_immutable_json(reference_trace_path, reference.json())
    adapter_sha = _write_immutable_json(adapter_trace_path, adapter.json())
    reference_npz_sha = _write_immutable_npz(reference_npz, predictions=reference.predictions, targets=reference.targets)
    adapter_npz_sha = _write_immutable_npz(adapter_npz, predictions=adapter.predictions, targets=adapter.targets)
    receipt = {
        "schema": PARITY_RECEIPT_SCHEMA,
        "status": "PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION",
        "prelaunch_sha256": prelaunch_sha256,
        "comparison": outcome,
        "counters": counters.as_dict(),
        "subm_nwb_paths_constructed": False,
        "subm_nwb_files_accessed": False,
        "subm_endpoint_scores_computed": False,
        "external_score_authorized": False,
    }
    receipt_path = output_root / "receipt.json"
    receipt_sha = _write_immutable_json(receipt_path, receipt)
    artifacts = []
    for path, sha in ((input_path, input_sha), (reference_trace_path, reference_sha), (adapter_trace_path, adapter_sha), (reference_npz, reference_npz_sha), (adapter_npz, adapter_npz_sha), (receipt_path, receipt_sha)):
        artifacts.append({"path": path.name, "sha256": sha, "bytes": path.stat().st_size, "mode": "0444"})
    seal_path = output_root / "seal.json"
    seal_sha = _write_immutable_json(seal_path, {"schema_version": 1, "kind": "dandi_000688_subc_scorer_adapter_parity_execution_seal_v2", "status": receipt["status"], "artifacts": artifacts})
    return {"output_root": str(output_root), "receipt_sha256": receipt_sha, "seal_sha256": seal_sha, "status": receipt["status"]}


def execute_parity_once_not_authorized(*_: Any, **__: Any) -> None:
    """Hard fence for this delivery before it can import C1 runtime/data code."""
    raise ParityAuthorizationError(
        "PARITY_EXECUTION_NOT_AUTHORIZED: root must bind the final parity-v2 prelaunch, a real Ed25519 authorization, and a future adapter implementation before checkpoint/NWB/forward access"
    )
