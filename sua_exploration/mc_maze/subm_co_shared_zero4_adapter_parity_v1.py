"""Consumed-sub-C input parity for a future shared-zero4 external adapter.

This module is deliberately not an external scorer.  It has one pinned input:
the already-consumed sub-C development session used by the V5/V5R2 adapter
parity work.  It never loads a checkpoint, runs a model, computes R2, accepts an
NWB path from a caller, or refers to the formal sub-M data root.

The experiment checks the input contract needed by a later, separately
authorized three-arm scorer:

* retain the existing V5 owner chronology and its start/stop-only bridge;
* retain the owner loader's first-50 query exclusion and rebuild activity from
  the first 30 trials;
* attach direct standardized Z4 by channel count only;
* prove exact input invariance after target-direction shuffle and removal;
* cover both SUA and pseudo-MUA views without a model forward or score.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("sua", "pseudo_mua")
VARIANTS = ("original", "label_shuffle", "label_drop")
SUPPORT_TRIALS = 50
ACTIVITY_TRIALS = 30
BIN_SIZE_MS = 20
HISTORY_BINS = 50
TRIAL_LENGTH_BINS = 100
PAD_VALUE = -1.0
SIDE_DIM = 4

CONSUMED_SUBC = {
    "path": "sua_exploration/data/dandi_000688/sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb",
    "session": "sub-C_ses-CO-20151103",
    "sha256": "7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7",
    "bytes": 62145872,
}
BEHAVIOR_NORMALIZER = {
    "path": "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/sua/behavior_stats/be50f588491c004f721e.npz",
    "sha256": "821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd",
    "bytes": 397,
}

# Existing owners are immutable dependencies of this append-only parity.  The
# new source package may be reviewed independently without rewriting them.
DEPENDENCY_SOURCE_PINS = {
    "sua_exploration/mc_maze/paired_view_c1_shared_zero4.py":
        "ee754cb279fae1a72e88b51b1f11f94b22380e953dde972ef36eabcaf4295f88",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py":
        "563f095bdfab38e5df4cd6087e9dac27f2fb58a81190e0b9ad920215fba7adb4",
    "sua_exploration/mc_maze/multisession_datamodule.py":
        "674fb4c235ba8f9393a6d1614f1f6f4260177ed9751e88acb4c05c4396d81e2d",
    "sua_exploration/mc_maze/datamodule.py":
        "0c93359991c32e81b552e00169fa1f31a5b71d345782c9bf81b136bd5c708506",
    "sua_exploration/scripts/eval_adaptation_dandi688.py":
        "e452d19d738316a2ff54074585b22e526bb1cc9275bfcfe5d33aa1becc5ccc30",
    "sua_exploration/scripts/dandi688_gradient_free_protocol.py":
        "a0d1b331c0548a967bbd08804231c2045a360e8947f869dea2d8c251e4d70e68",
}


class SharedZero4ParityError(RuntimeError):
    """The fixed consumed-sub-C zero4 input contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SharedZero4ParityError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin_path(root: Path, pin: Mapping[str, Any], *, label: str) -> Path:
    root = root.resolve()
    candidate = (root / str(pin["path"])).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SharedZero4ParityError(f"{label} escapes repository") from exc
    require(candidate.is_file() and not candidate.is_symlink(), f"missing/unsafe {label}")
    require(candidate.stat().st_size == int(pin["bytes"]), f"{label} byte-size drift")
    require(sha256_file(candidate) == str(pin["sha256"]), f"{label} SHA-256 drift")
    return candidate


def verify_dependency_sources(root: Path = REPO_ROOT) -> dict[str, str]:
    root = root.resolve()
    observed: dict[str, str] = {}
    for relative, expected in DEPENDENCY_SOURCE_PINS.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SharedZero4ParityError(f"dependency escapes repository: {relative}") from exc
        require(path.is_file() and not path.is_symlink(), f"missing/unsafe dependency: {relative}")
        digest = sha256_file(path)
        require(digest == expected, f"dependency source SHA-256 drift: {relative}")
        observed[relative] = digest
    return observed


def _load_behavior_normalizer(root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = _pin_path(root, BEHAVIOR_NORMALIZER, label="behavior normalizer")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            mean = bundle["mean"].astype(np.float32, copy=False)
            std = bundle["std"].astype(np.float32, copy=False)
    except (KeyError, OSError, ValueError) as exc:
        raise SharedZero4ParityError("invalid pinned behavior normalizer") from exc
    require(mean.ndim == 1 and mean.shape == std.shape, "behavior normalizer shape drift")
    require(bool(np.isfinite(mean).all() and np.isfinite(std).all()), "non-finite normalizer")
    require(bool(np.all(std > 0)), "non-positive normalizer scale")
    return mean, std


def _runtime_owners(root: Path) -> dict[str, Any]:
    """Load only data/chronology owners; no model or checkpoint owner is imported."""

    verify_dependency_sources(root)
    for location in (
        root / "sua_exploration",
        root / "sua_exploration/scripts",
        root / "streaming_calibration_exp",
    ):
        text = str(location)
        if text not in sys.path:
            sys.path.insert(0, text)
    data = importlib.import_module("mc_maze.multisession_datamodule")
    dataset = importlib.import_module("mc_maze.datamodule")
    c1 = importlib.import_module("eval_adaptation_dandi688")
    selection = importlib.import_module("dandi688_gradient_free_protocol")
    bridge = importlib.import_module(
        "sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5"
    )
    zero4 = importlib.import_module("mc_maze.paired_view_c1_shared_zero4")
    torch = importlib.import_module("torch")
    return {
        "load_dandi688_session": data.load_dandi688_session,
        "list_datamodule_rewarded_trials": data.list_datamodule_rewarded_trials,
        "MCMazeSessionDataset": dataset.MCMazeSessionDataset,
        "build_calib_trials_for_indices": c1.build_calib_trials_for_indices,
        "select_calibration_trial_indices": selection.select_calibration_trial_indices,
        "bridge_owner_chronology_for_c1_builder": bridge.bridge_owner_chronology_for_c1_builder,
        "attach_standardized_zero4_to_evaluation_record":
            zero4.attach_standardized_zero4_to_evaluation_record,
        "require_standardized_zero4": zero4.require_standardized_zero4,
        "torch": torch,
    }


def _array_digest(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    raw = array.tobytes(order="C")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(np.isfinite(array).all()),
    }


def _bitwise_positive_float32_zero(value: np.ndarray) -> bool:
    array = np.asarray(value)
    return (
        array.dtype == np.dtype(np.float32)
        and array.ndim == 2
        and array.shape[1] == SIDE_DIM
        and bool(np.all(array.view(np.uint32) == np.uint32(0)))
    )


def _structural_trial_digest(trials: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        {
            "position": position,
            "trial_index": int(trial["trial_index"]),
            "start": int(trial["start"]),
            "stop": int(trial["stop"]),
        }
        for position, trial in enumerate(trials)
    ]
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def _label_digest(trials: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        {
            "position": position,
            "target_dir_present": "target_dir" in trial,
            "target_dir": (
                None
                if trial.get("target_dir") is None
                else float(trial["target_dir"])
            ),
        }
        for position, trial in enumerate(trials)
    ]
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def label_variant(
    trials: Sequence[Mapping[str, Any]], variant: str
) -> list[dict[str, Any]]:
    """Change labels only; slice coordinates/order remain byte-equivalent as integers."""

    require(variant in VARIANTS, f"unsupported label variant: {variant}")
    rows = [dict(trial) for trial in trials]
    if variant == "original":
        return rows
    if variant == "label_drop":
        for row in rows:
            row.pop("target_dir", None)
            row.pop("target_id", None)
        return rows
    values = [row.get("target_dir") for row in rows]
    require(len(values) > 1, "label shuffle requires at least two trials")
    rotated = values[1:] + values[:1]
    require(rotated != values, "label shuffle did not change the label sequence")
    for row, value in zip(rows, rotated, strict=True):
        row["target_dir"] = value
    return rows


def _expected_query_valid_starts(
    trials: Sequence[Mapping[str, Any]], *, support_trials: int = SUPPORT_TRIALS
) -> np.ndarray:
    starts: list[int] = []
    for trial in trials[support_trials:]:
        starts.extend(range(int(trial["start"]), int(trial["stop"]) - HISTORY_BINS + 1))
    return np.asarray(starts, dtype=np.int64)


@dataclass(frozen=True)
class PredictionInputRecord:
    view: str
    variant: str
    session: str
    neural: np.ndarray
    behavior: np.ndarray
    calibration: np.ndarray
    valid_starts: np.ndarray
    side_features: np.ndarray
    selection_indices: tuple[int, ...]
    structural_trial_sha256: str
    label_sha256: str
    descriptor_receipt: Mapping[str, Any]
    source_unit_count: int

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "neural": self.neural,
            "behavior": self.behavior,
            "calibration": self.calibration,
            "valid_starts": self.valid_starts,
            "side_features": self.side_features,
        }

    def trace(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "variant": self.variant,
            "session": self.session,
            "source_unit_count": self.source_unit_count,
            "channel_count": int(self.neural.shape[1]),
            "selection_indices": list(self.selection_indices),
            "structural_trial_sha256": self.structural_trial_sha256,
            "label_sha256": self.label_sha256,
            "arrays": {key: _array_digest(value) for key, value in self.arrays().items()},
            "descriptor_receipt": dict(self.descriptor_receipt),
        }


def _assert_same_prediction_input(
    reference: PredictionInputRecord, candidate: PredictionInputRecord
) -> None:
    require(reference.view == candidate.view, "cross-view prediction-input comparison")
    require(reference.session == candidate.session, "session drift after label mutation")
    require(
        reference.selection_indices == candidate.selection_indices == tuple(range(ACTIVITY_TRIALS)),
        "activity first_n30 selection drift",
    )
    require(
        reference.structural_trial_sha256 == candidate.structural_trial_sha256,
        "label mutation changed structural chronology",
    )
    for key, expected in reference.arrays().items():
        observed = candidate.arrays()[key]
        require(expected.dtype == observed.dtype, f"{key} dtype changed after label mutation")
        require(expected.shape == observed.shape, f"{key} shape changed after label mutation")
        require(
            np.array_equal(expected, observed, equal_nan=True),
            f"{key} values changed after label mutation",
        )
        require(_array_digest(expected) == _array_digest(observed), f"{key} digest drift")


def _descriptor_receipt_checked(receipt: Mapping[str, Any]) -> dict[str, Any]:
    expected_zero = {
        "target_direction_label_reads_for_descriptor": 0,
        "t4_trial_rate_reads_for_descriptor": 0,
        "target_t4_rate_fit_calls": 0,
    }
    for key, expected in expected_zero.items():
        require(receipt.get(key) == expected, f"zero4 descriptor access counter drift: {key}")
    require(receipt.get("raw_t4_constructed") is False, "zero4 constructed raw T4")
    require(
        receipt.get("source_t4_normalizer_arithmetic_performed") is False,
        "zero4 performed T4 normalizer arithmetic",
    )
    require(receipt.get("bitwise_float32_zero") is True, "zero4 receipt lacks bitwise proof")
    return dict(receipt)


def build_view_records(
    *, root: Path, view: str, owners: Mapping[str, Any]
) -> tuple[dict[str, PredictionInputRecord], dict[str, Any]]:
    require(view in VIEWS, f"unsupported signal view: {view}")
    behavior_mean, behavior_std = _load_behavior_normalizer(root)
    nwb_path = _pin_path(root, CONSUMED_SUBC, label="consumed sub-C development NWB")
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
        signal_view=view,
    )
    raw_owner_trials = owners["list_datamodule_rewarded_trials"](
        nwb_path,
        bin_size_ms=BIN_SIZE_MS,
        window_size=HISTORY_BINS,
        trial_result_filter="R",
    )
    _raw_evidence, builder_trials, bridge_trace = owners[
        "bridge_owner_chronology_for_c1_builder"
    ](raw_owner_trials)
    expected_starts = _expected_query_valid_starts(builder_trials)
    owner_starts = np.asarray(record.valid_starts)
    require(owner_starts.dtype == np.dtype(np.int64), "owner valid_starts dtype drift")
    require(
        np.array_equal(owner_starts, expected_starts),
        "owner valid_starts do not equal post-first-50 chronology",
    )
    require(len(owner_starts) > 0, f"{view} has no post-first-50 query windows")

    records: dict[str, PredictionInputRecord] = {}
    baseline_structure = _structural_trial_digest(builder_trials)
    baseline_label = _label_digest(builder_trials)
    for variant in VARIANTS:
        trials = label_variant(builder_trials, variant)
        require(
            _structural_trial_digest(trials) == baseline_structure,
            f"{variant} changed structural chronology",
        )
        indices = owners["select_calibration_trial_indices"](
            trials, ACTIVITY_TRIALS, SUPPORT_TRIALS, "first"
        )
        require(indices == list(range(ACTIVITY_TRIALS)), "activity first_n30 selection drift")
        rebuild_record = {
            "name": record.name,
            "neural": record.neural,
            "behavior": record.behavior,
            "trials": trials,
            "n_units": int(record.neural.shape[1]),
            "source_unit_count": int(record.source_unit_count or record.neural.shape[1]),
            "signal_view": view,
        }
        calibration = owners["build_calib_trials_for_indices"](
            rebuild_record, indices, ACTIVITY_TRIALS
        )
        require(
            calibration.shape
            == (ACTIVITY_TRIALS, TRIAL_LENGTH_BINS, int(record.neural.shape[1])),
            f"{view}/{variant} activity calibration shape drift",
        )
        zero_record = owners["attach_standardized_zero4_to_evaluation_record"](
            rebuild_record
        )
        side = np.asarray(zero_record["side_features"])
        owners["require_standardized_zero4"](
            side, context=f"consumed-subC/{view}/{variant}"
        )
        require(_bitwise_positive_float32_zero(side), f"{view}/{variant} non-bitwise zero4")
        descriptor_receipt = _descriptor_receipt_checked(
            zero_record["zero4_descriptor_receipt"]
        )
        dataset = owners["MCMazeSessionDataset"](
            neural_data=record.neural,
            behavior_data=record.behavior,
            valid_starts=owner_starts,
            calib_trials=calibration,
            window_size=HISTORY_BINS,
            session_name=record.name,
            side_features=side,
            electrode_ids=None,
        )
        require(len(dataset) == len(owner_starts), f"{view}/{variant} dataset window drift")
        first = dataset[0]
        require(len(first) == 5, f"{view}/{variant} first input arity drift")
        first_side = first[4].detach().cpu().contiguous().numpy()
        require(
            _bitwise_positive_float32_zero(first_side),
            f"{view}/{variant} first input has non-bitwise zero4",
        )
        records[variant] = PredictionInputRecord(
            view=view,
            variant=variant,
            session=record.name,
            neural=np.asarray(dataset.neural),
            behavior=np.asarray(dataset.behavior),
            calibration=dataset.calib_trials.detach().cpu().contiguous().numpy(),
            valid_starts=np.asarray(dataset.valid_starts),
            side_features=dataset.side_features.detach().cpu().contiguous().numpy(),
            selection_indices=tuple(indices),
            structural_trial_sha256=_structural_trial_digest(trials),
            label_sha256=_label_digest(trials),
            descriptor_receipt=descriptor_receipt,
            source_unit_count=int(record.source_unit_count or record.neural.shape[1]),
        )

    original = records["original"]
    require(original.label_sha256 == baseline_label, "original label digest drift")
    for variant in ("label_shuffle", "label_drop"):
        require(
            records[variant].label_sha256 != original.label_sha256,
            f"{variant} failed to alter label evidence",
        )
        _assert_same_prediction_input(original, records[variant])
    return records, {
        "view": view,
        "raw_owner_trial_count": len(raw_owner_trials),
        "builder_trial_count": len(builder_trials),
        "owner_schema_bridge": bridge_trace,
        "activity_policy": {
            "selection": "first_n30",
            "activity_trials": ACTIVITY_TRIALS,
            "indices": list(range(ACTIVITY_TRIALS)),
        },
        "comparator_and_query_policy": {
            "t4_comparator_pool_trials": SUPPORT_TRIALS,
            "zero4_descriptor_pool_trials": None,
            "query_trials_start": SUPPORT_TRIALS,
            "valid_starts_source": "existing owner loader after first 50 rewarded trials",
            "valid_starts": _array_digest(owner_starts),
            "valid_starts_exact_post50_reconstruction": True,
        },
        "channel_contract": {
            "source_unit_count": int(record.source_unit_count or record.neural.shape[1]),
            "channel_count": int(record.neural.shape[1]),
            "neural_shape": list(record.neural.shape),
            "side_shape": list(original.side_features.shape),
            "side_matches_signal_view_channel_axis": (
                original.side_features.shape[0] == record.neural.shape[1]
            ),
        },
        "label_invariance": {
            "variants": list(VARIANTS),
            "prediction_input_exact_after_shuffle": True,
            "prediction_input_exact_after_drop": True,
            "model_forward_calls": 0,
            "r2_computations": 0,
        },
        "records": {variant: value.trace() for variant, value in records.items()},
    }


def execute_consumed_subc_zero4_parity(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run input-only parity on the fixed consumed sub-C fixture."""

    root = root.resolve()
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CUDA must be hidden")
    _pin_path(root, CONSUMED_SUBC, label="consumed sub-C development NWB")
    dependencies = verify_dependency_sources(root)
    owners = _runtime_owners(root)
    torch = owners["torch"]
    require(not torch.cuda.is_initialized(), "CUDA was initialized in CPU-only parity")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A previous in-process test may already have frozen the same setting.
        require(torch.get_num_interop_threads() == 1, "Torch interop thread policy drift")

    view_traces: dict[str, Any] = {}
    view_records: dict[str, dict[str, PredictionInputRecord]] = {}
    for view in VIEWS:
        records, trace = build_view_records(root=root, view=view, owners=owners)
        view_records[view] = records
        view_traces[view] = trace

    sua = view_records["sua"]["original"]
    pseudo = view_records["pseudo_mua"]["original"]
    require(sua.neural.shape[1] == sua.side_features.shape[0], "SUA side/channel mismatch")
    require(
        pseudo.neural.shape[1] == pseudo.side_features.shape[0],
        "pseudo-MUA side/channel mismatch",
    )
    require(
        pseudo.source_unit_count >= pseudo.neural.shape[1],
        "pseudo-MUA source/channel count relation drift",
    )
    require(
        _bitwise_positive_float32_zero(sua.side_features)
        and _bitwise_positive_float32_zero(pseudo.side_features),
        "view-level zero4 bit contract failed",
    )
    return {
        "schema": "dandi_000688_consumed_subc_shared_zero4_adapter_input_parity_v1",
        "status": "PASS_CONSUMED_SUBC_SHARED_ZERO4_INPUT_PARITY_NO_MODEL_NO_R2",
        "scope": {
            "fixture": dict(CONSUMED_SUBC),
            "views": list(VIEWS),
            "external_subm_accessed": False,
            "external_subm_scored": False,
            "checkpoint_files_opened": 0,
            "model_forward_calls": 0,
            "r2_computations": 0,
            "gpu_used": False,
        },
        "zero4_descriptor_contract": {
            "construction_input": ["channel_count"],
            "dtype": "float32",
            "side_dim": SIDE_DIM,
            "positive_zero_bits_only": True,
            "target_direction_label_reads_for_descriptor": 0,
            "t4_trial_rate_reads_for_descriptor": 0,
            "target_t4_rate_fit_calls": 0,
            "source_t4_normalizer_value_reads": 0,
            "source_t4_normalizer_arithmetic_performed": False,
            "raw_t4_constructed": False,
        },
        "dependencies": dependencies,
        "views": view_traces,
        "cross_view": {
            "views_present": list(VIEWS),
            "sua_channels": int(sua.neural.shape[1]),
            "pseudo_mua_channels": int(pseudo.neural.shape[1]),
            "pseudo_mua_source_units": pseudo.source_unit_count,
            "both_side_shapes_match_their_channel_axes": True,
        },
        "scientific_result": None,
        "external_scoring_capability_created": False,
    }

