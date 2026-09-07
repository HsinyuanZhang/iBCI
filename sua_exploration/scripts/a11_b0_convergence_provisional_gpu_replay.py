"""Provisional CUDA replay for the pinned A11 B0 validation curve.

This module is deliberately additive.  It imports the authority/audit helpers
from :mod:`a11_b0_convergence_full_access_evaluator`, but never edits or calls
its CPU driver.  A temporary exact-source materialization is shared with that
evaluator and a second, uniquely named compatibility driver changes only the
execution device from CPU to a caller-selected CUDA device.

The CUDA result is provisional: the immutable CPU smoke receipt remains the
authority.  A seed-42/epoch-000 GPU run must first match that receipt's mean and
all six per-session R2 values (including the query/window hashes) before this
module will launch the 36-checkpoint shard plan.  Every output receipt is
``PROVISIONAL_GPU_REPLAY``, created with O_EXCL, made read-only, and bound to a
SHA-256 sidecar.

No training, checkpoint writes, backward pass, formal-test NWB opens, or train
NWB opens are permitted by this path.  In particular, a CUDA runtime failure
is a blocked result, never permission to fall back to CPU or to skip parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


# The imported evaluator is the sole source of A11 authority constants and
# materialization/data/query bindings.  The module lives beside this file, so
# its import works both when invoked as a script and from focused tests.
import a11_b0_convergence_full_access_evaluator as a11


PROGRAM_ID = "a11_b0_convergence_provisional_gpu_replay_v2"
RECEIPT_KIND = "PROVISIONAL_GPU_REPLAY"
PARITY_MODE = "gpu_parity"
FULL_MODE = "gpu_full_replay"
EXPECTED_CPU_SMOKE_MEAN_R2 = 0.29514842480421066
DEFAULT_PARITY_TOLERANCE = 1.0e-5
GPU_IDS: tuple[int, int] = (0, 1)
EXPECTED_GPU_NAME_FRAGMENT = "RTX 3090"
RESULT_MARKER = "A11_GPU_RESULT_JSON="


class GPUReplayError(RuntimeError):
    """A CUDA, authority, parity, or immutability violation that must block."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(_canonical_bytes(value))


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_tolerance(value: float) -> float:
    tolerance = float(value)
    # A caller may choose a stricter tolerance but may not weaken the parity
    # gate beyond the stated 1e-5 contract.
    if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance > DEFAULT_PARITY_TOLERANCE:
        raise GPUReplayError(
            f"parity tolerance must be finite, positive, and <= {DEFAULT_PARITY_TOLERANCE:g}; "
            f"found {value!r}"
        )
    return tolerance


def _read_only(path: Path, label: str) -> None:
    """Require a receipt/sidecar to have no owner/group/other write bit."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        raise GPUReplayError(f"{label} is not readable: {path}: {exc}") from exc
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise GPUReplayError(f"{label} is writable; refusing mutable CPU authority: {path}")


def _verify_sidecar(receipt_path: Path) -> str:
    """Verify an immutable JSON receipt against its SHA sidecar."""
    sidecar = Path(f"{receipt_path}.sha256")
    if not receipt_path.is_file() or not sidecar.is_file():
        raise GPUReplayError(f"CPU smoke receipt or SHA sidecar is missing: {receipt_path}")
    _read_only(receipt_path, "CPU smoke receipt")
    _read_only(sidecar, "CPU smoke receipt SHA sidecar")
    try:
        line = sidecar.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise GPUReplayError(f"cannot read CPU smoke SHA sidecar: {sidecar}: {exc}") from exc
    fields = line.split()
    if len(fields) != 2 or len(fields[0]) != 64 or fields[1] != receipt_path.name:
        raise GPUReplayError(f"malformed CPU smoke SHA sidecar: {sidecar}")
    observed = sha256_bytes(receipt_path.read_bytes())
    if fields[0] != observed:
        raise GPUReplayError(
            f"CPU smoke receipt SHA mismatch: sidecar={fields[0]}, observed={observed}"
        )
    return observed


def _cpu_smoke_reference_path(repo_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        _verify_sidecar(path)
        return path
    result_dir = repo_root / "sua_exploration" / "results" / a11.PROGRAM_ID
    candidates = sorted(
        path
        for path in result_dir.glob(f"{a11.PROGRAM_ID}_cpu_smoke_*.json")
        if "_blocked_" not in path.name
    )
    if not candidates:
        raise GPUReplayError(
            "no completed immutable CPU smoke receipt was found; GPU parity cannot start"
        )
    if len(candidates) != 1:
        raise GPUReplayError(
            "CPU smoke authority is ambiguous; expected one completed receipt, found "
            f"{[str(path) for path in candidates]}"
        )
    _verify_sidecar(candidates[0])
    return candidates[0]


def load_cpu_smoke_reference(
    repo_root: Path,
    authority_fingerprint: str,
    explicit: Path | None = None,
) -> tuple[Path, str, Mapping[str, Any]]:
    """Load and validate the immutable completed CPU smoke receipt.

    The authority fingerprint is checked against the current read-only
    preflight, preventing parity against a receipt from a different source,
    checkpoint, query, or normalizer binding.
    """
    receipt_path = _cpu_smoke_reference_path(repo_root, explicit)
    digest = _verify_sidecar(receipt_path)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GPUReplayError(f"cannot parse CPU smoke receipt: {receipt_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GPUReplayError("CPU smoke receipt must be a JSON object")
    if payload.get("program_id") != a11.PROGRAM_ID:
        raise GPUReplayError("CPU smoke receipt program binding drifted")
    if payload.get("status") != "completed" or payload.get("receipt_kind") != "non_scientific_cpu_forward_smoke":
        raise GPUReplayError("CPU smoke receipt is not a completed forward-smoke authority")
    if payload.get("authority_fingerprint_sha256") != authority_fingerprint:
        raise GPUReplayError(
            "CPU smoke receipt authority fingerprint differs from current preflight: "
            f"expected {authority_fingerprint}, found {payload.get('authority_fingerprint_sha256')}"
        )
    if payload.get("formal_test_nwb_access") not in (False, None):
        raise GPUReplayError("CPU smoke authority records formal-test NWB access")
    forward = payload.get("cpu_forward_result")
    if not isinstance(forward, dict):
        raise GPUReplayError("CPU smoke receipt has no forward result")
    if forward.get("device") != "cpu" or forward.get("torch_grad_enabled") is not False:
        raise GPUReplayError("CPU smoke authority is not CPU/no-grad")
    per_seed = forward.get("per_seed")
    if not isinstance(per_seed, dict) or set(per_seed) != {"42", "43", "44"}:
        raise GPUReplayError("CPU smoke receipt has an unexpected seed scope")
    epoch_zero = per_seed.get("42", {}).get("per_epoch", {}).get("0")
    if not isinstance(epoch_zero, dict):
        raise GPUReplayError("CPU smoke receipt is missing seed 42 epoch_000")
    mean_r2 = epoch_zero.get("mean_r2")
    if not isinstance(mean_r2, (int, float)) or not math.isfinite(float(mean_r2)):
        raise GPUReplayError("CPU smoke mean R2 is not finite")
    if float(mean_r2) != EXPECTED_CPU_SMOKE_MEAN_R2:
        raise GPUReplayError(
            "immutable CPU smoke mean differs from the pinned value: "
            f"expected {EXPECTED_CPU_SMOKE_MEAN_R2!r}, found {mean_r2!r}"
        )
    sessions = epoch_zero.get("per_session_r2")
    if not isinstance(sessions, dict) or set(sessions) != set(a11._require_list(
        payload.get("data_authority", {}).get("allowed_validation_sessions"),
        "CPU smoke allowed validation sessions",
    )):
        raise GPUReplayError("CPU smoke receipt does not contain exactly the six validation sessions")
    for name, value in sessions.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise GPUReplayError(f"CPU smoke session R2 is not finite: {name}")
    query_windows = forward.get("query_windows")
    if not isinstance(query_windows, dict) or set(query_windows) != set(sessions):
        raise GPUReplayError("CPU smoke receipt query/window scope is not exactly six sessions")
    if forward.get("normalizer_arrays_sha256") != payload.get("data_authority", {}).get(
        "normalizer", {}
    ).get("array_contract_sha256"):
        raise GPUReplayError("CPU smoke normalizer array binding drifted")
    return receipt_path, digest, payload


def _gpu_driver_source() -> str:
    """Return the exact-source GPU compatibility driver.

    This intentionally mirrors the historical CPU compatibility driver's
    data/query/metric loop.  Device checks and receipt-facing metadata are the
    only execution differences.
    """
    return r'''
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


RESULT_MARKER = "A11_GPU_RESULT_JSON="


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def tensor_state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    config = read_config(sys.argv[1])
    source_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(source_root / "sua_exploration" / "scripts"))
    sys.path.insert(0, str(source_root / "sua_exploration"))
    sys.path.insert(0, str(source_root / "streaming_calibration_exp"))

    expected_visible = str(config["physical_cuda_device"])
    observed_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if observed_visible != expected_visible:
        raise RuntimeError(
            "A11 GPU device-scope violation: expected CUDA_VISIBLE_DEVICES="
            + expected_visible
            + ", found "
            + repr(observed_visible)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A11 GPU CUDA is unavailable; refusing CPU fallback")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "A11 GPU process must expose exactly one logical CUDA device; found "
            + str(torch.cuda.device_count())
        )
    try:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        capability = tuple(int(v) for v in torch.cuda.get_device_capability(0))
    except Exception as exc:
        raise RuntimeError("A11 GPU CUDA initialization failed; refusing CPU fallback") from exc
    if "RTX 3090" not in str(device_name):
        raise RuntimeError("A11 GPU device is not the required RTX 3090: " + repr(device_name))
    torch.set_grad_enabled(False)
    if torch.is_grad_enabled():
        raise RuntimeError("A11 GPU could not disable autograd")

    import eval_adaptation_dandi688 as adaptation
    import mc_maze.datamodule as datamodule
    import mc_maze.multisession_datamodule as multisession
    from dandi688_gradient_free_protocol import select_calibration_trial_indices
    from eval_adaptation_dandi688 import (
        build_calib_trials_for_indices,
        eval_r2,
        load_session_with_trials,
        make_subset_dataset,
    )
    from select_gradient_free_protocol_dandi688 import load_frozen_model

    allowed_paths = {str(Path(path).resolve()) for path in config["allowed_validation_nwb_paths"]}
    opened_paths = []
    original_nwb_io = {
        "adaptation": adaptation.NWBHDF5IO,
        "datamodule": datamodule.NWBHDF5IO,
        "multisession": multisession.NWBHDF5IO,
    }

    class GuardedNWBHDF5IO:
        def __init__(self, path, *args, **kwargs):
            canonical = str(Path(path).resolve())
            if canonical not in allowed_paths:
                raise RuntimeError(
                    "A11 GPU data-scope violation: attempted NWB open outside the six "
                    "validation files: "
                    + canonical
                )
            opened_paths.append(canonical)
            self._wrapped = original_nwb_io["adaptation"](path, *args, **kwargs)

        def __enter__(self):
            return self._wrapped.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            return self._wrapped.__exit__(exc_type, exc_value, traceback)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    adaptation.NWBHDF5IO = GuardedNWBHDF5IO
    datamodule.NWBHDF5IO = GuardedNWBHDF5IO
    multisession.NWBHDF5IO = GuardedNWBHDF5IO

    normalizer_path = Path(config["normalizer_cache_path"])
    with np.load(normalizer_path, allow_pickle=False) as normalizer:
        mean = normalizer["mean"].astype(np.float32, copy=False)
        std = normalizer["std"].astype(np.float32, copy=False)
    if (
        mean.shape != (2,)
        or std.shape != (2,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise RuntimeError("A11 GPU normalizer cache failed shape/finiteness validation")

    records = []
    for raw_path in config["allowed_validation_nwb_paths"]:
        path = Path(raw_path).resolve()
        rec = load_session_with_trials(
            path,
            config["bin_size_ms"],
            config["window_size"],
            config["pool_size"],
            config["trial_length"],
            config["pad_value"],
            mean,
            std,
            trial_result_filter="R",
            cache_dir=Path(config["private_cache_dir"]),
            signal_view=config["signal_view"],
        )
        if rec["name"] not in config["allowed_validation_sessions"]:
            raise RuntimeError("A11 GPU record escaped the validation roster: " + repr(rec["name"]))
        if rec["signal_view"] != "sua":
            raise RuntimeError("A11 GPU record signal view drifted from SUA")
        selection = select_calibration_trial_indices(
            rec["trials"], config["calibration_n"], config["pool_size"], "first"
        )
        expected_selection = list(range(config["calibration_n"]))
        if selection != expected_selection:
            raise RuntimeError("A11 GPU fixed-first calibration selection drifted")
        rec["calib_trials"] = build_calib_trials_for_indices(
            rec, selection, config["calibration_n"]
        )
        query_trials = rec["trials"][config["pool_size"] :]
        if not query_trials:
            raise RuntimeError("A11 GPU query is empty after the fixed support pool")
        dataset = make_subset_dataset(rec, query_trials, rec["name"])
        if not len(dataset):
            raise RuntimeError("A11 GPU query has zero scored windows")
        support_payload = [
            {
                "usable_trial_list_index": int(index),
                "original_trial_index": int(rec["trials"][index]["trial_index"]),
            }
            for index in selection
        ]
        query_payload = [
            {
                "original_trial_index": int(trial["trial_index"]),
                "start": int(trial["start"]),
                "stop": int(trial["stop"]),
                "target_dir": trial.get("target_dir"),
                "target_id": trial.get("target_id"),
            }
            for trial in query_trials
        ]
        starts = np.asarray(dataset.valid_starts, dtype="<i8")
        records.append(
            {
                "name": rec["name"],
                "dataset": dataset,
                "source_unit_count": int(rec["source_unit_count"]),
                "support_trial_count": len(support_payload),
                "support_trial_sha256": sha256_json(support_payload),
                "query_trial_count": len(query_payload),
                "query_trial_sha256": sha256_json(query_payload),
                "scored_window_count": int(len(dataset)),
                "scored_window_start_sha256": sha256_bytes(starts.tobytes()),
            }
        )

    result = {
        "device": device.type,
        "cuda_device": str(device),
        "cuda_device_logical_index": 0,
        "cuda_device_physical_index": int(config["physical_cuda_device"]),
        "cuda_device_name": str(device_name),
        "cuda_compute_capability": list(capability),
        "cuda_visible_devices": observed_visible,
        "runtime_binding": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
            "torch_version": str(torch.__version__),
            "torch_file": str(Path(torch.__file__).resolve()),
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_is_built": bool(torch.backends.cuda.is_built()),
            "torch_cuda_visible_device_count": int(torch.cuda.device_count()),
        },
        "torch_grad_enabled": bool(torch.is_grad_enabled()),
        "normalizer_arrays_sha256": sha256_json(
            {"mean": mean.astype("<f4").tolist(), "std": std.astype("<f4").tolist()}
        ),
        "query_windows": {
            item["name"]: {
                key: value
                for key, value in item.items()
                if key != "dataset"
            }
            for item in records
        },
        "per_seed": {},
    }
    for run in config["runs"]:
        seed = int(run["seed"])
        per_epoch = {}
        for epoch_text, checkpoint_path in run["epoch_checkpoints"].items():
            epoch = int(epoch_text)
            model = load_frozen_model(
                Path(checkpoint_path),
                Path(config["teacher_checkpoint"]),
                config["variant"],
                device,
            )
            if model.training:
                raise RuntimeError("A11 GPU frozen model unexpectedly remained in training mode")
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("A11 GPU frozen model has a trainable parameter")
            if any(parameter.device != device for parameter in model.parameters()):
                raise RuntimeError("A11 GPU frozen model parameter escaped its CUDA device")
            if any(value.device != device for value in model.state_dict().values()):
                raise RuntimeError("A11 GPU frozen model state escaped its CUDA device")
            before = tensor_state_sha256(model)
            per_session = {}
            with torch.no_grad():
                for item in records:
                    per_session[item["name"]] = float(eval_r2(model, item["dataset"], device))
            after = tensor_state_sha256(model)
            if before != after:
                raise RuntimeError("A11 GPU forward-only invariant failed: checkpoint state changed")
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("A11 GPU forward-only invariant failed: a parameter received a gradient")
            per_epoch[str(epoch)] = {
                "cuda_device": str(device),
                "cuda_device_physical_index": int(config["physical_cuda_device"]),
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
                "checkpoint_sha256": config["checkpoint_sha256_by_seed"][str(seed)][str(epoch)],
                "per_session_r2": per_session,
                "mean_r2": float(sum(per_session.values()) / len(per_session)),
                "model_state_sha256_before": before,
                "model_state_sha256_after": after,
            }
            del model
        result["per_seed"][str(seed)] = {
            "cuda_device": str(device),
            "cuda_device_physical_index": int(config["physical_cuda_device"]),
            "per_epoch": per_epoch,
        }

    if set(opened_paths) != allowed_paths:
        raise RuntimeError(
            "A11 GPU validation-scope violation: expected each of the six validation NWBs exactly "
            "at least once, observed "
            + repr(sorted(set(opened_paths)))
        )
    result["opened_nwb_paths"] = sorted(set(opened_paths))
    result["opened_nwb_path_count"] = len(result["opened_nwb_paths"])
    result["formal_test_nwb_access"] = False
    result["training_nwb_access"] = False
    print(RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
'''


def _materialize_gpu_driver(repo_root: Path, target_root: Path) -> dict[str, str]:
    """Materialize the exact historical closure plus this new GPU driver."""
    source_hashes = a11._materialize_historical_runtime(repo_root, target_root)
    driver_path = target_root / "a11_provisional_gpu_compatibility_driver.py"
    driver_path.write_text(_gpu_driver_source(), encoding="utf-8")
    source_hashes[driver_path.name] = a11.sha256_file(driver_path)
    return source_hashes


def _device_inventory(gpu_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Read-only nvidia-smi inventory; fail closed when the requested GPUs differ."""
    ids = [int(value) for value in gpu_ids]
    if not ids or len(set(ids)) != len(ids) or any(value < 0 for value in ids):
        raise GPUReplayError(f"GPU ids must be distinct non-negative integers: {ids!r}")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise GPUReplayError(f"nvidia-smi is unavailable; refusing provisional GPU replay: {exc}") from exc
    if completed.returncode != 0:
        raise GPUReplayError(
            "nvidia-smi failed; refusing provisional GPU replay: " + completed.stderr.strip()
        )
    inventory: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise GPUReplayError(f"unexpected nvidia-smi inventory row: {line!r}")
        try:
            index = int(fields[0])
            used = int(fields[2])
            total = int(fields[3])
        except ValueError as exc:
            raise GPUReplayError(f"unexpected nvidia-smi numeric row: {line!r}") from exc
        inventory.append({"index": index, "name": fields[1], "memory_used_mib": used, "memory_total_mib": total})
    by_id = {item["index"]: item for item in inventory}
    missing = [value for value in ids if value not in by_id]
    if missing:
        raise GPUReplayError(f"requested GPU ids are absent from nvidia-smi: {missing}")
    selected = [by_id[value] for value in ids]
    wrong_name = [item for item in selected if EXPECTED_GPU_NAME_FRAGMENT not in item["name"]]
    if wrong_name:
        raise GPUReplayError(f"requested devices are not RTX 3090s: {wrong_name}")
    return selected


def _driver_config(
    preflight: Mapping[str, Any],
    private_cache_dir: Path,
    *,
    seed_epochs: Mapping[int, Sequence[int]],
    physical_cuda_device: int,
) -> dict[str, Any]:
    """Build a driver config from the evaluator's private, audited bindings."""
    runtime = a11._require_mapping(preflight["_runtime"], "preflight runtime")
    runs = runtime["runs"]
    if not isinstance(runs, list) or not all(isinstance(run, a11.RunAuthority) for run in runs):
        raise GPUReplayError("internal preflight run authority is malformed")
    normalizer_target = a11._copy_normalizer_into_private_cache(
        normalizer_path=Path(runtime["normalizer_path"]),
        private_cache_dir=private_cache_dir,
        train_paths=[Path(str(path)) for path in runs[0].metadata["session_files"]["train"]],
    )
    selected_runs: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, dict[str, str]] = {}
    for run in runs:
        wanted = {int(epoch) for epoch in seed_epochs.get(run.seed, ())}
        selected_epochs = {
            str(epoch): str(run.epoch_checkpoints[epoch])
            for epoch in a11.LIGHTNING_EPOCHS
            if epoch in wanted
        }
        if not selected_epochs:
            continue
        selected_runs.append({"seed": run.seed, "epoch_checkpoints": selected_epochs})
        checkpoint_hashes[str(run.seed)] = {
            str(epoch): run.checkpoint_sha256[epoch]
            for epoch in a11.LIGHTNING_EPOCHS
            if epoch in wanted
        }
    return {
        "physical_cuda_device": int(physical_cuda_device),
        "allowed_validation_nwb_paths": preflight["data_authority"]["allowed_validation_nwb_paths"],
        "allowed_validation_sessions": preflight["data_authority"]["allowed_validation_sessions"],
        "normalizer_cache_path": str(normalizer_target),
        "private_cache_dir": str(private_cache_dir),
        "teacher_checkpoint": runs[0].metadata["teacher_checkpoint"],
        "variant": a11.VARIANT,
        "signal_view": a11.SIGNAL_VIEW,
        "bin_size_ms": a11.BIN_SIZE_MS,
        "window_size": a11.WINDOW_SIZE,
        "pool_size": a11.POOL_SIZE,
        "calibration_n": a11.CALIBRATION_N,
        "trial_length": a11.TRIAL_LENGTH,
        "pad_value": a11.PAD_VALUE,
        "checkpoint_sha256_by_seed": checkpoint_hashes,
        "runs": selected_runs,
    }


def _run_gpu_process(
    repo_root: Path,
    preflight: Mapping[str, Any],
    *,
    seed_epochs: Mapping[int, Sequence[int]],
    physical_cuda_device: int,
) -> dict[str, Any]:
    """Run one isolated CUDA process and return its marker payload."""
    with tempfile.TemporaryDirectory(prefix="a11_b0_provisional_gpu_") as temporary_dir:
        root = Path(temporary_dir)
        materialized_root = root / "historical_source"
        source_hashes = _materialize_gpu_driver(repo_root, materialized_root)
        private_cache_dir = root / "private_validation_cache"
        config = _driver_config(
            preflight,
            private_cache_dir,
            seed_epochs=seed_epochs,
            physical_cuda_device=physical_cuda_device,
        )
        config_path = root / "driver_config.json"
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(physical_cuda_device)
        # The compatible torch build is installed in the existing conda
        # environment.  Isolate the child from ~/.local, which otherwise
        # shadows it with an incompatible cu130 torch build.
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = ""
        completed = subprocess.run(
            [sys.executable, str(materialized_root / "a11_provisional_gpu_compatibility_driver.py"), str(config_path)],
            cwd=str(repo_root),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        marker_lines = [
            line[len(RESULT_MARKER) :]
            for line in completed.stdout.splitlines()
            if line.startswith(RESULT_MARKER)
        ]
        if completed.returncode != 0:
            raise GPUReplayError(
                f"GPU {physical_cuda_device} driver failed with exit code {completed.returncode}; "
                f"stdout tail={completed.stdout[-4000:]!r}; stderr tail={completed.stderr[-4000:]!r}"
            )
        if len(marker_lines) != 1:
            raise GPUReplayError(
                f"GPU {physical_cuda_device} driver did not emit exactly one result marker; "
                f"stdout tail={completed.stdout[-4000:]!r}; stderr tail={completed.stderr[-4000:]!r}"
            )
        try:
            payload = json.loads(marker_lines[0])
        except json.JSONDecodeError as exc:
            raise GPUReplayError(f"GPU driver emitted malformed JSON: {exc}") from exc
        if payload.get("device") != "cuda" or payload.get("torch_grad_enabled") is not False:
            raise GPUReplayError("GPU driver violated CUDA/no-grad execution contract")
        if payload.get("cuda_device_physical_index") != physical_cuda_device:
            raise GPUReplayError("GPU driver reported the wrong physical CUDA device")
        runtime_binding = payload.get("runtime_binding")
        if not isinstance(runtime_binding, dict):
            raise GPUReplayError("GPU driver omitted its interpreter/torch runtime binding")
        if runtime_binding.get("python_no_user_site") != "1":
            raise GPUReplayError("GPU driver was not isolated from the user-site torch installation")
        if runtime_binding.get("python_executable") != str(Path(sys.executable).resolve()):
            raise GPUReplayError("GPU driver interpreter differs from the wrapper interpreter")
        if not runtime_binding.get("torch_cuda_is_built") or not runtime_binding.get("torch_cuda_version"):
            raise GPUReplayError("GPU driver runtime binding is not CUDA-enabled")
        if payload.get("formal_test_nwb_access") is not False or payload.get("training_nwb_access") is not False:
            raise GPUReplayError("GPU driver reported forbidden NWB access")
        allowed = set(preflight["data_authority"]["allowed_validation_nwb_paths"])
        opened = set(payload.get("opened_nwb_paths", []))
        if opened != allowed:
            raise GPUReplayError(
                f"GPU {physical_cuda_device} opened an unexpected validation scope: "
                f"expected {sorted(allowed)}, found {sorted(opened)}"
            )
        return {
            "driver_payload": payload,
            "physical_cuda_device": physical_cuda_device,
            "materialized_source_sha256": source_hashes,
            "driver_stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
            "driver_stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
        }


def _epoch_payload(payload: Mapping[str, Any], seed: int, epoch: int) -> Mapping[str, Any]:
    try:
        per_seed = payload["per_seed"]
        return per_seed[str(seed)]["per_epoch"][str(epoch)]
    except (KeyError, TypeError) as exc:
        raise GPUReplayError(f"GPU payload is missing seed {seed} epoch {epoch}") from exc


def compare_gpu_parity(
    cpu_reference_payload: Mapping[str, Any],
    gpu_payload: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Fail closed unless GPU seed42/epoch000 matches all CPU-smoke fields."""
    tolerance = _ensure_tolerance(tolerance)
    cpu_forward = cpu_reference_payload.get("cpu_forward_result")
    if not isinstance(cpu_forward, dict):
        raise GPUReplayError("CPU reference has no forward result")
    cpu_epoch = _epoch_payload(cpu_forward, 42, 0)
    gpu_epoch = _epoch_payload(gpu_payload, 42, 0)
    cpu_sessions = cpu_epoch.get("per_session_r2")
    gpu_sessions = gpu_epoch.get("per_session_r2")
    if not isinstance(cpu_sessions, dict) or not isinstance(gpu_sessions, dict):
        raise GPUReplayError("CPU/GPU parity payload lacks per-session R2 values")
    if set(cpu_sessions) != set(gpu_sessions) or len(cpu_sessions) != 6:
        raise GPUReplayError(
            "GPU parity session roster differs from the six-session CPU smoke roster: "
            f"cpu={sorted(cpu_sessions)}, gpu={sorted(gpu_sessions)}"
        )
    deltas = {
        name: abs(float(gpu_sessions[name]) - float(cpu_sessions[name])) for name in sorted(cpu_sessions)
    }
    gpu_mean = float(gpu_epoch.get("mean_r2"))
    cpu_mean = float(cpu_epoch.get("mean_r2"))
    deltas["__mean_r2__"] = abs(gpu_mean - cpu_mean)
    if not all(math.isfinite(value) for value in deltas.values()):
        raise GPUReplayError(f"GPU parity produced a non-finite R2 delta: {deltas}")
    max_abs = max(deltas.values())
    cpu_query = cpu_forward.get("query_windows")
    gpu_query = gpu_payload.get("query_windows")
    if cpu_query != gpu_query:
        raise GPUReplayError("GPU parity query/window hashes differ from the immutable CPU smoke")
    if cpu_forward.get("normalizer_arrays_sha256") != gpu_payload.get("normalizer_arrays_sha256"):
        raise GPUReplayError("GPU parity normalizer array binding differs from the CPU smoke")
    if gpu_payload.get("opened_nwb_path_count") != 6 or gpu_payload.get("formal_test_nwb_access") is not False:
        raise GPUReplayError("GPU parity did not open exactly six validation NWBs only")
    passed = max_abs <= tolerance
    result = {
        "passed": bool(passed),
        "tolerance": tolerance,
        "max_abs_r2_difference": float(max_abs),
        "per_session_abs_r2_difference": {key: float(value) for key, value in deltas.items() if key != "__mean_r2__"},
        "mean_abs_r2_difference": float(deltas["__mean_r2__"]),
        "cpu_mean_r2": cpu_mean,
        "gpu_mean_r2": gpu_mean,
        "query_window_hashes_identical": True,
        "normalizer_arrays_identical": True,
        "validation_session_count": 6,
    }
    if not passed:
        raise GPUReplayError(
            f"GPU parity failed: max absolute R2 difference {max_abs:g} exceeds tolerance {tolerance:g}"
        )
    return result


def _validate_full_payloads(
    payloads: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge three one-seed processes and recheck exact 36-checkpoint scope."""
    if len(payloads) != 3:
        raise GPUReplayError(f"full GPU replay requires three seed payloads, found {len(payloads)}")
    merged_per_seed: dict[str, Any] = {}
    query_windows: Mapping[str, Any] | None = None
    normalizer_hash: str | None = None
    devices: dict[str, int] = {}
    runtime_bindings: dict[str, Mapping[str, Any]] = {}
    source_hashes: list[Mapping[str, str]] = []
    for payload in payloads:
        per_seed = payload.get("per_seed")
        if not isinstance(per_seed, dict) or len(per_seed) != 1:
            raise GPUReplayError("each GPU shard must contain exactly one seed")
        seed_text = next(iter(per_seed))
        seed = int(seed_text)
        if seed not in a11.SEEDS or seed_text in merged_per_seed:
            raise GPUReplayError(f"unexpected or duplicate GPU shard seed: {seed_text}")
        seed_value = per_seed[seed_text]
        epochs = seed_value.get("per_epoch") if isinstance(seed_value, dict) else None
        if not isinstance(epochs, dict) or set(epochs) != {str(epoch) for epoch in a11.LIGHTNING_EPOCHS}:
            raise GPUReplayError(f"GPU shard seed {seed} does not contain all twelve epochs")
        for epoch_text, epoch_record in epochs.items():
            if not isinstance(epoch_record, dict):
                raise GPUReplayError(f"GPU shard seed {seed} epoch {epoch_text} is malformed")
            if epoch_record.get("model_state_sha256_before") != epoch_record.get("model_state_sha256_after"):
                raise GPUReplayError(f"GPU shard seed {seed} epoch {epoch_text} mutated state")
            if epoch_record.get("checkpoint_sha256") != preflight["source_configuration_authority"]["run_receipts"][seed_text]["epoch_checkpoints"][f"epoch_{int(epoch_text):03d}"]["sha256"]:
                raise GPUReplayError(f"GPU shard seed {seed} epoch {epoch_text} checkpoint binding drifted")
        merged_per_seed[seed_text] = seed_value
        physical = int(seed_value.get("cuda_device_physical_index"))
        devices[seed_text] = physical
        runtime_binding = payload.get("runtime_binding")
        if not isinstance(runtime_binding, dict) or runtime_binding.get("python_no_user_site") != "1":
            raise GPUReplayError(f"GPU shard seed {seed} omitted a user-site-isolated runtime binding")
        runtime_bindings[seed_text] = runtime_binding
        current_query = payload.get("query_windows")
        if query_windows is None:
            query_windows = current_query
        elif current_query != query_windows:
            raise GPUReplayError("GPU shards disagree on query/window hashes")
        current_normalizer = payload.get("normalizer_arrays_sha256")
        if normalizer_hash is None:
            normalizer_hash = current_normalizer
        elif current_normalizer != normalizer_hash:
            raise GPUReplayError("GPU shards disagree on normalizer arrays")
        source_hashes.append(payload.get("materialized_source_sha256", {}))
    if set(merged_per_seed) != {str(seed) for seed in a11.SEEDS}:
        raise GPUReplayError("GPU full replay is missing one or more B0 seeds")
    if query_windows is None or normalizer_hash is None:
        raise GPUReplayError("GPU full replay omitted query/normalizer bindings")
    if set(devices.values()) != set(GPU_IDS):
        raise GPUReplayError(f"GPU shards did not use both requested devices: {devices}")
    expected_query_hash = preflight["query_and_metric_contract"]["query_contract_sha256"]
    if expected_query_hash != a11.sha256_json(a11.query_contract()):
        raise GPUReplayError("query contract hash unexpectedly drifted during full replay")
    return {
        "per_seed": merged_per_seed,
        "query_windows": query_windows,
        "normalizer_arrays_sha256": normalizer_hash,
        "cuda_device_by_seed": devices,
        "runtime_binding_by_seed": runtime_bindings,
        "materialized_source_sha256_by_shard": source_hashes,
    }


def run_gpu_parity(
    repo_root: Path,
    preflight: Mapping[str, Any],
    cpu_reference_payload: Mapping[str, Any],
    *,
    gpu_id: int,
    tolerance: float,
) -> dict[str, Any]:
    inventory = _device_inventory([gpu_id])
    process_result = _run_gpu_process(
        repo_root,
        preflight,
        seed_epochs={42: (0,), 43: (), 44: ()},
        physical_cuda_device=gpu_id,
    )
    payload = process_result["driver_payload"]
    parity = compare_gpu_parity(cpu_reference_payload, payload, tolerance=tolerance)
    return {
        "parity_gate": parity,
        "gpu_inventory": inventory,
        "process_result": process_result,
    }


def run_gpu_full_replay(
    repo_root: Path,
    preflight: Mapping[str, Any],
    cpu_reference_payload: Mapping[str, Any],
    *,
    tolerance: float,
    gpu_ids: Sequence[int] = GPU_IDS,
) -> dict[str, Any]:
    """Run parity first, then safely shard seed processes over GPU0/GPU1."""
    ids = tuple(int(value) for value in gpu_ids)
    if ids != GPU_IDS:
        raise GPUReplayError(f"full replay is pinned to the two RTX 3090 ids {GPU_IDS}; found {ids}")
    inventory = _device_inventory(ids)
    parity = run_gpu_parity(
        repo_root,
        preflight,
        cpu_reference_payload,
        gpu_id=ids[0],
        tolerance=tolerance,
    )
    # Seed 42 parity has already run, so keep its checked result and run its
    # remaining eleven epochs as part of the first full shard.  Seed 43 runs on
    # GPU1 concurrently; seed 44 runs only after the first two complete.
    shard_specs = [
        ({42: tuple(range(1, 12))}, ids[0]),
        ({43: tuple(range(12))}, ids[1]),
    ]
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="a11-gpu-shard") as executor:
        futures = [
            executor.submit(
                _run_gpu_process,
                repo_root,
                preflight,
                seed_epochs=epochs,
                physical_cuda_device=physical,
            )
            for epochs, physical in shard_specs
        ]
        shard_results = [future.result() for future in futures]
    seed42_parity_result = parity["process_result"]["driver_payload"]
    seed42_full_result = shard_results[0]["driver_payload"]
    seed42_full_result["materialized_source_sha256"] = shard_results[0]["materialized_source_sha256"]
    shard_results[1]["driver_payload"]["materialized_source_sha256"] = shard_results[1][
        "materialized_source_sha256"
    ]
    # The parity process and the full shard must agree on epoch_000 before the
    # merged receipt is constructed.  This catches a changed cache or source
    # tree between the gate and expansion.
    parity_epoch = _epoch_payload(seed42_parity_result, 42, 0)
    full_seed42_epochs = seed42_full_result.get("per_seed", {}).get("42", {}).get("per_epoch", {})
    if "0" not in full_seed42_epochs:
        # The first shard intentionally excludes epoch zero; merge it below.
        full_seed42_epochs = dict(full_seed42_epochs)
        parity_epoch = dict(parity_epoch)
        parity_epoch["cuda_device"] = f"cuda:0"
        parity_epoch["cuda_device_physical_index"] = ids[0]
        parity_checkpoint = preflight["source_configuration_authority"]["run_receipts"]["42"][
            "epoch_checkpoints"
        ]["epoch_000"]
        parity_epoch["checkpoint_path"] = parity_checkpoint["path"]
        parity_epoch["checkpoint_sha256"] = parity_checkpoint["sha256"]
        full_seed42_epochs["0"] = parity_epoch
        seed42_full_result["per_seed"]["42"]["per_epoch"] = full_seed42_epochs
    else:
        if full_seed42_epochs["0"] != parity_epoch:
            raise GPUReplayError("seed 42 epoch_000 changed between parity and expansion")
    seed44_result = _run_gpu_process(
        repo_root,
        preflight,
        seed_epochs={42: (), 43: (), 44: tuple(range(12))},
        physical_cuda_device=ids[0],
    )
    seed44_result["driver_payload"]["materialized_source_sha256"] = seed44_result[
        "materialized_source_sha256"
    ]
    merged = _validate_full_payloads(
        [seed42_full_result, shard_results[1]["driver_payload"], seed44_result["driver_payload"]],
        preflight,
    )
    return {
        "parity": parity,
        "gpu_inventory": inventory,
        "merged_result": merged,
        "shard_processes": [
            shard_results[0],
            shard_results[1],
            seed44_result,
        ],
    }


def _output_path(repo_root: Path, mode: str, fingerprint: str, out_dir: Path | None) -> Path:
    destination = out_dir.resolve() if out_dir is not None else repo_root / "sua_exploration" / "results" / PROGRAM_ID
    return destination / f"{PROGRAM_ID}_{mode}_{fingerprint[:16]}.json"


def write_provisional_receipt(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    """Create a read-only O_EXCL JSON receipt and matching SHA sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    digest = sha256_bytes(data)
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH,
        )
    except FileExistsError as exc:
        raise GPUReplayError(f"provisional receipt already exists and will not be overwritten: {path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    sidecar = Path(f"{path}.sha256")
    sidecar_data = f"{digest}  {path.name}\n".encode("ascii")
    try:
        fd = os.open(
            sidecar,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH,
        )
    except FileExistsError as exc:
        raise GPUReplayError(
            f"provisional receipt SHA sidecar already exists and will not be overwritten: {sidecar}"
        ) from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(sidecar_data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path, sidecar, digest


def _failure_receipt(repo_root: Path, out_dir: Path | None, mode: str, error: Exception) -> tuple[Path, Path, str]:
    fingerprint = sha256_json({"program_id": PROGRAM_ID, "mode": mode, "error": f"{type(error).__name__}: {error}"})
    payload = {
        "program_id": PROGRAM_ID,
        "receipt_kind": RECEIPT_KIND,
        "status": "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "error_type": type(error).__name__,
        "error": str(error),
        "forward_evaluation_started": False,
        "formal_test_nwb_access": False,
        "training_nwb_access": False,
        "science_claim_allowed": False,
    }
    return write_provisional_receipt(_output_path(repo_root, f"{mode}_blocked", fingerprint, out_dir), payload)


def _public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in preflight.items() if key != "_runtime"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gpu-parity", action="store_true", help="Run seed 42 epoch_000 parity only.")
    mode.add_argument(
        "--gpu-full-replay",
        action="store_true",
        help="Run parity, then the 36-checkpoint two-GPU provisional replay.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id for --gpu-parity (default: 0).")
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=DEFAULT_PARITY_TOLERANCE,
        help="Maximum absolute CPU/GPU R2 delta; must be <= 1e-5.",
    )
    parser.add_argument("--cpu-reference", type=Path, default=None, help="Explicit immutable CPU smoke receipt path.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Additive provisional receipt directory.")
    args = parser.parse_args(argv)
    selected_mode = FULL_MODE if args.gpu_full_replay else PARITY_MODE
    repo_root = project_root()
    try:
        tolerance = _ensure_tolerance(args.parity_tolerance)
        preflight = a11.build_preflight(repo_root)
        authority_fingerprint = str(preflight["authority_fingerprint_sha256"])
        reference_path, reference_digest, reference_payload = load_cpu_smoke_reference(
            repo_root,
            authority_fingerprint,
            args.cpu_reference,
        )
        if selected_mode == PARITY_MODE:
            run = run_gpu_parity(
                repo_root,
                preflight,
                reference_payload,
                gpu_id=args.gpu_id,
                tolerance=tolerance,
            )
            public_result = {
                "parity_gate": run["parity_gate"],
                "gpu_inventory": run["gpu_inventory"],
                "gpu_forward_result": run["process_result"]["driver_payload"],
                "gpu_runtime_binding": run["process_result"]["driver_payload"]["runtime_binding"],
                "materialized_source_sha256": run["process_result"]["materialized_source_sha256"],
                "driver_stdout_sha256": run["process_result"]["driver_stdout_sha256"],
                "driver_stderr_sha256": run["process_result"]["driver_stderr_sha256"],
            }
        else:
            run = run_gpu_full_replay(
                repo_root,
                preflight,
                reference_payload,
                tolerance=tolerance,
            )
            public_result = {
                "parity_gate": run["parity"]["parity_gate"],
                "gpu_inventory": run["gpu_inventory"],
                "gpu_full_result": run["merged_result"],
                "gpu_runtime_bindings_by_seed": run["merged_result"]["runtime_binding_by_seed"],
                "shard_processes": [
                    {
                        "physical_cuda_device": item["physical_cuda_device"],
                        "materialized_source_sha256": item["materialized_source_sha256"],
                        "driver_stdout_sha256": item["driver_stdout_sha256"],
                        "driver_stderr_sha256": item["driver_stderr_sha256"],
                    }
                    for item in run["shard_processes"]
                ],
            }
        payload = _public_preflight(preflight)
        payload.update(
            {
                "program_id": PROGRAM_ID,
                "receipt_kind": RECEIPT_KIND,
                "status": "completed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": selected_mode,
                "science_claim_allowed": False,
                "forward_evaluation_started": True,
                "formal_test_nwb_access": False,
                "training_nwb_access": False,
                "provisional_authority": {
                    "cpu_smoke_receipt_path": str(reference_path),
                    "cpu_smoke_receipt_sha256": reference_digest,
                    "cpu_smoke_mean_r2": EXPECTED_CPU_SMOKE_MEAN_R2,
                    "parity_tolerance": tolerance,
                    "source_binding": payload["historical_source"],
                    "query_metric_binding": payload["query_and_metric_contract"],
                    "normalizer_binding": payload["data_authority"]["normalizer"],
                    "checkpoint_binding": payload["source_configuration_authority"]["run_receipts"],
                    "runtime_binding": public_result.get(
                        "gpu_runtime_binding", public_result.get("gpu_runtime_bindings_by_seed")
                    ),
                },
                "gpu_replay_result": public_result,
            }
        )
        fingerprint = sha256_json(
            {
                "program_id": PROGRAM_ID,
                "mode": selected_mode,
                "authority_fingerprint_sha256": authority_fingerprint,
                "cpu_smoke_receipt_sha256": reference_digest,
            }
        )
        receipt, sidecar, digest = write_provisional_receipt(
            _output_path(repo_root, selected_mode, fingerprint, args.out_dir), payload
        )
        print(f"A11 provisional GPU receipt: {receipt}")
        print(f"A11 provisional GPU receipt SHA-256: {digest}")
        print(f"A11 provisional GPU receipt sidecar: {sidecar}")
        return 0
    except Exception as exc:
        try:
            receipt, sidecar, digest = _failure_receipt(repo_root, args.out_dir, selected_mode, exc)
        except Exception as receipt_error:
            print(
                f"A11 provisional GPU blocked before receipt: {type(exc).__name__}: {exc}; "
                f"receipt error: {type(receipt_error).__name__}: {receipt_error}",
                file=sys.stderr,
            )
            return 2
        print(f"A11 provisional GPU blocked receipt: {receipt}", file=sys.stderr)
        print(f"A11 provisional GPU blocked receipt SHA-256: {digest}", file=sys.stderr)
        print(f"A11 provisional GPU blocked receipt sidecar: {sidecar}", file=sys.stderr)
        print(f"A11 provisional GPU block reason: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
