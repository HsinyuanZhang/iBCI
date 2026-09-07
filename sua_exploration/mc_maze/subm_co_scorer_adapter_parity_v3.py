"""Concrete, independently-loaded C1/sub-M data-adapter parity v3.

The only callable command delivered with v3 is a fail-closed runner. This
module contains the future execution choreography for static review, but does
no work at import time and is not imported by the current runner. C1 and
adapter fixtures are built from independent owner-loader calls; no C1 batch is
passed to the adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


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
CALIBRATION_GATE = "LOADER_CALIBRATION_50_REBUILT_TO_C1_FIRST_N30"

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

# Score-only-v1 dependency sources used by the concrete adapter path, followed
# by the C1 owners defining reference preprocessing and fixed forward semantics.
OWNER_SOURCE_PINS = {
    "sua_exploration/mc_maze/subm_co_score_only.py": "8e6aa7b9efdeb894dd0f04a06aae0b01226aafb9ea3a6c493b9cb571648a9be4",
    "sua_exploration/mc_maze/multisession_datamodule.py": "674fb4c235ba8f9393a6d1614f1f6f4260177ed9751e88acb4c05c4396d81e2d",
    "sua_exploration/mc_maze/datamodule.py": "0c93359991c32e81b552e00169fa1f31a5b71d345782c9bf81b136bd5c708506",
    "sua_exploration/mc_maze/unit_side_features.py": "059faefcd766dfc8e25253d9ded2b619a46dea408e6f00a30cfa5b2ecd185ab6",
    "sua_exploration/scripts/eval_adaptation_dandi688.py": "e452d19d738316a2ff54074585b22e526bb1cc9275bfcfe5d33aa1becc5ccc30",
    "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py": "1fee6f482b1cd93e867b8f1b64bd13bdb967881b56f5a5563feaa635091daeb9",
    "sua_exploration/scripts/dandi688_gradient_free_protocol.py": "a0d1b331c0548a967bbd08804231c2045a360e8947f869dea2d8c251e4d70e68",
}


class ParityV3Error(RuntimeError):
    """A concrete fixture or exact-observation invariant failed."""


class ParityV3AuthorizationError(ParityV3Error):
    """Raised before this delivery may access a checkpoint or NWB."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ParityV3Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _pin_path(root: Path, pin: Mapping[str, Any], *, label: str) -> Path:
    candidate = (root / str(pin["path"])).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ParityV3Error(f"{label} escapes repository") from exc
    require(candidate.is_file() and not candidate.is_symlink(), f"missing/unsafe {label}")
    require(candidate.stat().st_size == int(pin["bytes"]), f"{label} byte-size drift")
    require(sha256_file(candidate) == str(pin["sha256"]), f"{label} SHA-256 drift")
    return candidate


def _verify_owner_source_pins(root: Path) -> None:
    for relative, expected in OWNER_SOURCE_PINS.items():
        path = (root / relative).resolve()
        require(path.is_file() and not path.is_symlink(), f"missing/unsafe owner source: {relative}")
        require(sha256_file(path) == expected, f"owner source SHA-256 drift: {relative}")


def _load_normalizer(root: Path, pin: Mapping[str, Any], *, label: str) -> tuple[np.ndarray, np.ndarray]:
    path = _pin_path(root, pin, label=label)
    try:
        with np.load(path, allow_pickle=False) as bundle:
            mean = bundle["mean"].astype(np.float32, copy=False)
            std = bundle["std"].astype(np.float32, copy=False)
    except (KeyError, OSError, ValueError) as exc:
        raise ParityV3Error(f"invalid pinned {label} normalizer") from exc
    require(mean.ndim == 1 and mean.shape == std.shape, f"{label} normalizer shape drift")
    require(np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 0), f"{label} normalizer invalid")
    return mean, std


def _runtime_owners() -> dict[str, Any]:
    """Import existing owners only in a future authorized runtime path."""
    import sys

    for location in (
        REPO_ROOT / "sua_exploration",
        REPO_ROOT / "sua_exploration/scripts",
        REPO_ROOT / "streaming_calibration_exp",
    ):
        text = str(location)
        if text not in sys.path:
            sys.path.insert(0, text)
    c1 = importlib.import_module("eval_adaptation_dandi688")
    selection = importlib.import_module("dandi688_gradient_free_protocol")
    model = importlib.import_module("select_gradient_free_protocol_dandi688")
    data = importlib.import_module("mc_maze.multisession_datamodule")
    side = importlib.import_module("mc_maze.unit_side_features")
    dataset = importlib.import_module("mc_maze.datamodule")
    torch = importlib.import_module("torch")
    torchmetrics = importlib.import_module("torchmetrics.regression")
    return {
        "c1": c1,
        "selection": selection,
        "model": model,
        "load_dandi688_session": data.load_dandi688_session,
        "list_datamodule_rewarded_trials": data.list_datamodule_rewarded_trials,
        "load_unit_side_features": side.load_unit_side_features,
        "MCMazeSessionDataset": dataset.MCMazeSessionDataset,
        "DataLoader": importlib.import_module("torch.utils.data").DataLoader,
        "torch": torch,
        "R2Score": torchmetrics.R2Score,
    }


def _array_digest(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    contiguous = np.ascontiguousarray(value)
    raw = contiguous.tobytes(order="C")
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(np.isfinite(contiguous).all()),
    }


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").contiguous().numpy()
    return np.asarray(value)


def _canonical_trial_chronology(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Complete ordered chronology common to both existing owner APIs."""
    rows: list[dict[str, Any]] = []
    for trial in trials:
        target_dir = trial.get("target_dir")
        target_value = None if target_dir is None or not np.isfinite(target_dir) else float(target_dir)
        rows.append(
            {
                "trial_index": int(trial["trial_index"]),
                "start": int(trial["start"]),
                "stop": int(trial["stop"]),
                "target_dir": target_value,
            }
        )
    return rows


def _first_batch_observation(dataset: Any, owners: Mapping[str, Any]) -> dict[str, Any]:
    loader = owners["DataLoader"](
        dataset, batch_size=LOADER_BATCH_SIZE, shuffle=False, num_workers=0
    )
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise ParityV3Error("post-50 dataset has no batch") from exc
    c1 = owners["c1"]
    neural, behavior, calibration, side_features, electrode_ids = c1._unpack_loader_batch(batch)
    values = {
        "neural": _as_numpy(neural),
        "behavior": _as_numpy(behavior),
        "calibration": _as_numpy(calibration),
        "side_features": _as_numpy(side_features),
        "electrode_ids": _as_numpy(electrode_ids),
    }
    names = [str(value) for value in batch[3]]
    require(len(names) == int(values["neural"].shape[0]), "batch session-name cardinality drift")
    return {
        "full": {key: _array_digest(value) for key, value in values.items()},
        "first16": {
            key: _array_digest(None if value is None else value[:TRACE_ROWS])
            for key, value in values.items()
        },
        "session_names": names,
        "values": values,
    }


@dataclass(frozen=True)
class IndependentlyPreparedFixture:
    """One side of parity, populated only by its own independent owner chain."""

    label: str
    dataset: Any
    chronology: list[dict[str, Any]]
    selection_indices: list[int]
    neural: np.ndarray
    behavior: np.ndarray
    calibration: np.ndarray
    valid_starts: np.ndarray
    side_features: np.ndarray | None
    electrode_ids: np.ndarray | None
    first_batch: dict[str, Any]

    def input_trace(self) -> dict[str, Any]:
        arrays = {
            "neural": _array_digest(self.neural),
            "behavior": _array_digest(self.behavior),
            "calibration": _array_digest(self.calibration),
            "valid_starts": _array_digest(self.valid_starts),
            "side_features": _array_digest(self.side_features),
            "electrode_ids": _array_digest(self.electrode_ids),
        }
        chronology_bytes = _canonical_bytes(self.chronology)
        return {
            "label": self.label,
            "calibration_gate": CALIBRATION_GATE,
            "selection_indices": list(self.selection_indices),
            "support_trials": SUPPORT_TRIALS,
            "identity_trials": IDENTITY_TRIALS,
            "query_trials_start": SUPPORT_TRIALS,
            "chronology": {
                "rows": len(self.chronology),
                "sha256": hashlib.sha256(chronology_bytes).hexdigest(),
                "value": self.chronology,
            },
            "arrays": arrays,
            "first_batch": {
                key: value for key, value in self.first_batch.items() if key != "values"
            },
        }


def prepare_c1_reference_fixture(
    root: Path, *, owners: Mapping[str, Any]
) -> IndependentlyPreparedFixture:
    """C1 owner loader → first_n30 rebuild → T4 attach → post-50 subset."""
    behavior_mean, behavior_std = _load_normalizer(root, NORMALIZERS["behavior"], label="behavior")
    side_mean, side_std = _load_normalizer(root, NORMALIZERS["t4"], label="T4")
    nwb_path = _pin_path(root, DEV_SESSION, label="C1 consumed development NWB")
    c1 = owners["c1"]
    selection = owners["selection"]
    record = c1.load_session_with_trials(
        nwb_path,
        bin_size_ms=BIN_SIZE_MS,
        window_size=HISTORY_BINS,
        calib_n=SUPPORT_TRIALS,
        max_trial_length=TRIAL_LENGTH_BINS,
        pad_value=PAD_VALUE,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        trial_result_filter="R",
        cache_dir=None,
        signal_view=VIEW,
    )
    chronology = _canonical_trial_chronology(record["trials"])
    indices = selection.select_calibration_trial_indices(
        record["trials"], IDENTITY_TRIALS, SUPPORT_TRIALS, "first"
    )
    record["calib_trials"] = c1.build_calib_trials_for_indices(record, indices, IDENTITY_TRIALS)
    record = c1.attach_side_features(
        record,
        nwb_path,
        side_feature_group="t4",
        waveform_feature_group="t4",
        pool_size=SUPPORT_TRIALS,
        permutation_seed=None,
        mean=side_mean,
        std=side_std,
        cache_dir=None,
    )
    dataset = c1.make_subset_dataset(
        record, record["trials"][SUPPORT_TRIALS:], record["name"]
    )
    require(len(dataset) > 0, "C1 post-50 dataset is empty")
    return IndependentlyPreparedFixture(
        label="c1_reference",
        dataset=dataset,
        chronology=chronology,
        selection_indices=list(indices),
        neural=np.asarray(record["neural"]),
        behavior=np.asarray(record["behavior"]),
        calibration=np.asarray(record["calib_trials"]),
        valid_starts=np.asarray(dataset.valid_starts),
        side_features=_as_numpy(getattr(dataset, "side_features", None)),
        electrode_ids=_as_numpy(getattr(dataset, "electrode_ids", None)),
        first_batch=_first_batch_observation(dataset, owners),
    )


def prepare_score_only_adapter_fixture(
    root: Path, *, owners: Mapping[str, Any]
) -> IndependentlyPreparedFixture:
    """Run score-only-v1 owners, then rebuild only C1 first_n30 activity.

    The owner loader receives 50 and therefore owns post-50 valid_starts. C1's
    existing builder receives independent adapter neural data plus the
    adapter-owner chronology and replaces only its returned 50-row activity
    calibration tensor. No C1 prepared batch enters this pipeline.
    """
    behavior_mean, behavior_std = _load_normalizer(root, NORMALIZERS["behavior"], label="behavior")
    side_mean, side_std = _load_normalizer(root, NORMALIZERS["t4"], label="T4")
    nwb_path = _pin_path(root, DEV_SESSION, label="C1 consumed development NWB")
    record = owners["load_dandi688_session"](
        nwb_path,
        bin_size_ms=BIN_SIZE_MS,
        window_size=HISTORY_BINS,
        calibration_n_trials=SUPPORT_TRIALS,
        max_trial_length=TRIAL_LENGTH_BINS,
        pad_value=PAD_VALUE,
        interpolate_trials=True,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        trial_result_filter="R",
        exclude_calibration_trials_from_windows=True,
        cache_dir=None,
        signal_view=VIEW,
    )
    chronology_source = owners["list_datamodule_rewarded_trials"](
        nwb_path,
        bin_size_ms=BIN_SIZE_MS,
        window_size=HISTORY_BINS,
        trial_result_filter="R",
    )
    chronology = _canonical_trial_chronology(chronology_source)
    indices = owners["selection"].select_calibration_trial_indices(
        chronology_source, IDENTITY_TRIALS, SUPPORT_TRIALS, "first"
    )
    c1 = owners["c1"]
    # Existing C1 code remains the only interpolation/padding implementation;
    # source arrays and chronology here come from the independent adapter path.
    rebuild_record = {
        "neural": record.neural,
        "trials": chronology_source,
        "n_units": int(record.neural.shape[1]),
    }
    rebuilt_calibration = c1.build_calib_trials_for_indices(
        rebuild_record, indices, IDENTITY_TRIALS
    )
    features, _metadata = owners["load_unit_side_features"](
        nwb_path,
        feature_group="t4",
        pool_size=SUPPORT_TRIALS,
        mean=side_mean,
        std=side_std,
        cache_dir=None,
        permutation_seed=None,
        bin_size_ms=BIN_SIZE_MS,
        window_size=HISTORY_BINS,
        trial_result_filter="R",
        signal_view=VIEW,
    )
    require(features.shape == (record.neural.shape[1], 4), "adapter T4 shape drift")
    dataset = owners["MCMazeSessionDataset"](
        neural_data=record.neural,
        behavior_data=record.behavior,
        valid_starts=record.valid_starts,
        calib_trials=rebuilt_calibration,
        window_size=HISTORY_BINS,
        session_name=record.name,
        side_features=features,
        electrode_ids=None,
    )
    require(len(dataset) > 0, "adapter post-50 dataset is empty")
    return IndependentlyPreparedFixture(
        label="score_only_adapter",
        dataset=dataset,
        chronology=chronology,
        selection_indices=list(indices),
        neural=np.asarray(record.neural),
        behavior=np.asarray(record.behavior),
        calibration=np.asarray(rebuilt_calibration),
        valid_starts=np.asarray(record.valid_starts),
        side_features=_as_numpy(getattr(dataset, "side_features", None)),
        electrode_ids=_as_numpy(getattr(dataset, "electrode_ids", None)),
        first_batch=_first_batch_observation(dataset, owners),
    )


def _assert_array_exact(
    label: str, reference: np.ndarray | None, adapter: np.ndarray | None
) -> None:
    if reference is None or adapter is None:
        require(reference is None and adapter is None, f"{label} None/value mismatch")
        return
    require(reference.dtype == adapter.dtype, f"{label} dtype mismatch")
    require(reference.shape == adapter.shape, f"{label} shape mismatch")
    require(np.array_equal(reference, adapter, equal_nan=True), f"{label} value mismatch")
    require(_array_digest(reference) == _array_digest(adapter), f"{label} digest mismatch")


@dataclass
class SharedAdapterParityObserver:
    """The sole gate between independent inputs and shared forward execution."""

    input_comparisons: int = 0
    forward_calls: int = 0
    metric_updates: int = 0
    metric_computes: int = 0

    def compare_inputs(
        self, reference: IndependentlyPreparedFixture, adapter: IndependentlyPreparedFixture
    ) -> dict[str, Any]:
        require(
            reference.selection_indices == adapter.selection_indices == list(range(IDENTITY_TRIALS)),
            "C1 first_n30 selection drift",
        )
        require(reference.chronology == adapter.chronology, "full owner trial chronology mismatch")
        for label in (
            "neural",
            "behavior",
            "calibration",
            "valid_starts",
            "side_features",
            "electrode_ids",
        ):
            _assert_array_exact(label, getattr(reference, label), getattr(adapter, label))
        require(
            reference.first_batch["session_names"] == adapter.first_batch["session_names"],
            "first-batch session-name mismatch",
        )
        for scope in ("full", "first16"):
            require(
                reference.first_batch[scope] == adapter.first_batch[scope],
                f"first batch {scope} dtype/shape/hash mismatch",
            )
            for key, reference_value in reference.first_batch["values"].items():
                adapter_value = adapter.first_batch["values"][key]
                if scope == "first16":
                    reference_value = None if reference_value is None else reference_value[:TRACE_ROWS]
                    adapter_value = None if adapter_value is None else adapter_value[:TRACE_ROWS]
                _assert_array_exact(
                    f"first batch {scope}/{key}", reference_value, adapter_value
                )
        self.input_comparisons += 1
        return {
            "reference": reference.input_trace(),
            "adapter": adapter.input_trace(),
            "input_exact": True,
        }


def _configure_cpu_determinism(owners: Mapping[str, Any]) -> Any:
    torch = owners["torch"]
    require(not torch.cuda.is_initialized(), "preinitialized CUDA is forbidden")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in {"", None}, "CUDA_VISIBLE_DEVICES must be empty")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch.device("cpu")


def _forward_and_observe(
    model: Any,
    dataset: Any,
    *,
    owners: Mapping[str, Any],
    observer: SharedAdapterParityObserver,
) -> dict[str, Any]:
    """The one fixed C1 forward/target/R2 body used for both datasets."""
    torch = owners["torch"]
    device = torch.device("cpu")
    c1 = owners["c1"]
    metric = owners["R2Score"](multioutput="variance_weighted").to(device)
    prediction_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in owners["DataLoader"](
            dataset, batch_size=LOADER_BATCH_SIZE, shuffle=False, num_workers=0
        ):
            neural, behavior, calibration, side_features, electrode_ids = c1._unpack_loader_batch(batch)
            neural, behavior, calibration = (
                neural.to(device),
                behavior.to(device),
                calibration.to(device),
            )
            side_features = None if side_features is None else side_features.to(device)
            electrode_ids = None if electrode_ids is None else electrode_ids.to(device)
            decoder_key_features = model.decoder_key_features(side_features)
            raw, _ = model.student(
                neural,
                calib_trials=calibration,
                side_features=side_features,
                decoder_key_features=decoder_key_features,
                electrode_ids=electrode_ids,
            )
            prediction = c1.decode_last_behavior(raw)
            target = behavior[:, -1:, :]
            metric.update(
                prediction.flatten(start_dim=0, end_dim=1),
                target.flatten(start_dim=0, end_dim=1),
            )
            observer.metric_updates += 1
            observer.forward_calls += 1
            prediction_rows.append(
                prediction[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            target_rows.append(
                target[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)
            )
    observer.metric_computes += 1
    predictions = np.concatenate(prediction_rows, axis=0)
    targets = np.concatenate(target_rows, axis=0)
    score = float(metric.compute().item())
    require(math.isfinite(score), "nonfinite R2")
    return {
        "prediction": predictions,
        "target": targets,
        "r2": score,
        "prediction_digest": _array_digest(predictions),
        "target_digest": _array_digest(targets),
    }


def compare_prediction_target_exact(reference: Mapping[str, Any], adapter: Mapping[str, Any]) -> None:
    _assert_array_exact(
        "prediction", np.asarray(reference["prediction"]), np.asarray(adapter["prediction"])
    )
    _assert_array_exact(
        "target", np.asarray(reference["target"]), np.asarray(adapter["target"])
    )
    require(
        abs(float(reference["r2"]) - float(adapter["r2"])) <= R2_ATOL,
        "shared metric R2 mismatch",
    )


def concrete_parity_after_future_authorization(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Future execution body; never called by the v3 runner in this delivery."""
    root = root.resolve()
    _verify_owner_source_pins(root)
    owners = _runtime_owners()
    device = _configure_cpu_determinism(owners)
    require(device.type == "cpu", "CPU-only execution required")
    reference = prepare_c1_reference_fixture(root, owners=owners)
    adapter = prepare_score_only_adapter_fixture(root, owners=owners)
    observer = SharedAdapterParityObserver()
    input_trace = observer.compare_inputs(reference, adapter)
    checkpoint = _pin_path(root, CHECKPOINT, label="fixed C1 checkpoint")
    teacher = _pin_path(root, TEACHER, label="fixed C1 teacher checkpoint")
    model = owners["model"].load_frozen_model(
        checkpoint, teacher, "B3S", device, identity_mode="calibrated"
    )
    reference_result = _forward_and_observe(
        model, reference.dataset, owners=owners, observer=observer
    )
    adapter_result = _forward_and_observe(
        model, adapter.dataset, owners=owners, observer=observer
    )
    compare_prediction_target_exact(reference_result, adapter_result)
    return {
        "status": "PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION_PENDING_SEPARATE_SEAL_REVIEW",
        "input_trace": input_trace,
        "reference": {
            key: value
            for key, value in reference_result.items()
            if key not in {"prediction", "target"}
        },
        "adapter": {
            key: value
            for key, value in adapter_result.items()
            if key not in {"prediction", "target"}
        },
        "observer": asdict(observer),
    }


def execute_parity_once_not_authorized(*_args: Any, **_kwargs: Any) -> None:
    """The current release's explicit runtime fence."""
    raise ParityV3AuthorizationError(
        "PARITY_EXECUTION_NOT_AUTHORIZED: v3 is static evidence only; a future "
        "append-only authorization and review are required before any runtime "
        "owner import, checkpoint/NWB access, or forward"
    )

