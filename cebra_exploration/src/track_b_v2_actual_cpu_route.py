"""Additive, source-only Track-B v2 actual-CPU selector/control route.

This module deliberately owns no NWB discovery.  Source folds arrive as already
materialised arrays plus an immutable source-only authority binding.  The only
runtime data built here are deterministic synthetic engineering controls.

Importing this module does not import CEBRA or torch.  The vendored CEBRA 0.6.1
runtime is loaded lazily by :class:`VendoredCebra061Backend` after all authority
and output-freshness checks have passed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import stat
import sys
import threading
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


SCHEMA_SELECTOR = "track_b_v2_actual_cpu_source_only_dual_selector_v1"
SCHEMA_CONTROLS = "track_b_v2_actual_synthetic_cpu_controls_v1"
SCHEMA_HARD_NULL = "track_b_v2_deranged_support_hard_null_threshold_v1"
SCHEMA_ENGINEERING_SMOKE = "track_b_v2_actual_cpu_engineering_smoke_v1"
SCHEMA_SOURCE_MICROBENCH = "track_b_v2_actual_cpu_source_microbenchmark_v1"
VENDORED_CEBRA_VERSION = "0.6.1"
VENDORED_CEBRA_COMMIT = "d1842ccc659bdf2d458ab31784ae029b9d48d21f"
FULL_D_GRID = (3, 8, 16)
FULL_ITERATION_GRID = (250, 1000, 2500, 10000)
FULL_LAMBDA_GRID = (1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0)
CONTROL_SEEDS = tuple(range(8))
POSITIVE_ARMS = ("cebra_joint_behavior", "cebra_frozen_source_adapt")
UNALIGNED_ARM = "cebra_adapt_unaligned"
DERANGED_ARM = "cebra_joint_behavior__target_support_auxiliary_rows_deranged"
ALL_CONTROL_ARMS = POSITIVE_ARMS + (UNALIGNED_ARM,)
POSITIVE_MIN_R2 = 0.70
KNN_K = 3
MODEL_ARCHITECTURE = "offset10-model"
EXPECTED_MODEL_OFFSET = (5, 5)
LEARNING_RATE = 3.0e-4
NUM_HIDDEN_UNITS = 32
DEFAULT_ADAPT_ITERATIONS = 500


class TrackBV2ActualCpuError(RuntimeError):
    """Fail-closed violation in the additive actual-CPU route."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2ActualCpuError(message)


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _array_bytes(array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array)
    header = canonical_json_bytes({"dtype": value.dtype.str, "shape": list(value.shape)})
    return header + memoryview(value).cast("B").tobytes()


def array_sha256(array: np.ndarray) -> str:
    return sha256_bytes(_array_bytes(array))


def index_sha256(indices: Sequence[int]) -> str:
    return array_sha256(np.asarray(tuple(int(v) for v in indices), dtype="<i8"))


def raw_array_sha256(array: np.ndarray) -> str:
    """Authority-compatible SHA of contiguous array bytes, without a header."""
    return sha256_bytes(memoryview(np.ascontiguousarray(array)).cast("B").tobytes())


def snapshot_file_closure(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Snapshot a small implementation closure before any expensive access."""
    snapshot: dict[str, dict[str, Any]] = {}
    for label, raw_path in sorted(paths.items()):
        path = Path(raw_path).expanduser().absolute()
        require(path.resolve() == path and path.is_file() and not path.is_symlink(),
                f"implementation closure file invalid: {label}")
        data = path.read_bytes()
        snapshot[label] = {"path": str(path), "sha256": sha256_bytes(data), "size": len(data)}
    return snapshot


def require_file_closure_unchanged(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    """Fail before publication if any launch-time implementation byte drifted."""
    paths = {label: Path(str(binding.get("path", ""))) for label, binding in snapshot.items()}
    live = snapshot_file_closure(paths)
    require(live == dict(snapshot), "implementation closure drifted after launch")


def model_alignment_from_offset(offset: object) -> dict[str, Any]:
    """Extract the receptive-field contract from the fitted vendored model.

    CEBRA's ``Offset.valid_slice`` is deliberately mirrored from the live
    object.  The expected (5, 5) pair is only a fail-closed architecture pin;
    it is not used to manufacture score indices.
    """
    try:
        left = int(getattr(offset, "left"))
        right = int(getattr(offset, "right"))
        valid_slice = getattr(offset, "valid_slice")
    except (AttributeError, TypeError, ValueError) as exc:
        raise TrackBV2ActualCpuError("fitted CEBRA model did not expose a valid Offset") from exc
    require((left, right) == EXPECTED_MODEL_OFFSET, "offset10-model implementation offset drift")
    require(valid_slice.start == left and valid_slice.stop == -right and valid_slice.step is None,
            "vendored Offset.valid_slice semantics drift")
    return {
        "model_architecture": MODEL_ARCHITECTURE,
        "offset_left": left,
        "offset_right": right,
        "offset_length": left + right,
        "valid_slice_start": int(valid_slice.start),
        "valid_slice_stop": int(valid_slice.stop),
        "valid_slice_step": None,
        "derived_from_fitted_model_get_offset": True,
        "derived_from_vendored_offset_valid_slice": True,
    }


def derive_contiguous_query_alignment(*, model_alignment: Mapping[str, Any],
                                      input_start: int, input_stop: int,
                                      embedding_row_count: int,
                                      support_stop: int | None = None) -> dict[str, Any]:
    """Create the valid query row authority without crossing block boundaries."""
    require(isinstance(input_start, int) and isinstance(input_stop, int) and input_stop > input_start,
            "query input interval invalid")
    if support_stop is not None:
        require(isinstance(support_stop, int) and support_stop <= input_start,
                "query receptive field may cross the support/query boundary")
    left = int(model_alignment.get("offset_left", -1))
    right = int(model_alignment.get("offset_right", -1))
    require((left, right) == EXPECTED_MODEL_OFFSET, "query alignment offset differs from frozen architecture")
    require(model_alignment.get("valid_slice_start") == left and
            model_alignment.get("valid_slice_stop") == -right,
            "query alignment does not come from vendored Offset.valid_slice")
    input_count = input_stop - input_start
    require(embedding_row_count == input_count, "padded transform embedding/input row drift")
    require(input_count > left + right, "offset10 query is too short for an interior valid slice")
    local = np.arange(left, input_count - right, dtype="<i8")
    source_indices = local + input_start
    rf_starts = source_indices - left
    rf_stops = source_indices + right
    require(local.size > 0, "offset10 query has no score-eligible interior rows")
    require(int(rf_starts.min()) >= input_start and int(rf_stops.max()) <= input_stop,
            "query receptive field escapes its continuous query block")
    if support_stop is not None:
        require(int(rf_starts.min()) >= support_stop,
                "query receptive field overlaps support")
    mapping = np.stack((source_indices, local), axis=1)
    receptive_fields = np.stack((rf_starts, rf_stops), axis=1)
    return {
        "query_input_start_inclusive": input_start,
        "query_input_stop_exclusive": input_stop,
        "query_input_row_count": input_count,
        "query_embedding_output_row_count": embedding_row_count,
        "valid_embedding_row_start_inclusive": int(local[0]),
        "valid_embedding_row_stop_exclusive": int(local[-1] + 1),
        "valid_embedding_row_count": int(local.size),
        "ordered_input_index_to_embedding_row_index": mapping.tolist(),
        "ordered_input_index_to_embedding_row_index_sha256": array_sha256(mapping),
        "query_receptive_field_start_stop_exclusive": receptive_fields.tolist(),
        "query_receptive_field_start_stop_exclusive_sha256": array_sha256(receptive_fields),
        "first_endpoint_input_index": int(source_indices[0]),
        "last_endpoint_input_index": int(source_indices[-1]),
        "first_receptive_field_start_inclusive": int(rf_starts[0]),
        "last_receptive_field_stop_exclusive": int(rf_stops[-1]),
        "discarded_left_embedding_rows": list(range(0, left)),
        "discarded_right_embedding_rows": list(range(input_count - right, input_count)),
        "continuous_query_cropped_internally": True,
        "support_query_boundary_crossed": False,
        "model_alignment": dict(model_alignment),
    }


def aligned_query_arrays(run: "EmbeddingRun", query_auxiliary: np.ndarray, *,
                         input_start: int = 0, support_stop: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    query = np.asarray(query_auxiliary, dtype=np.float64)
    alignment = derive_contiguous_query_alignment(
        model_alignment=run.model_alignment, input_start=input_start,
        input_stop=input_start + query.shape[0],
        embedding_row_count=run.target_query_embedding.shape[0], support_stop=support_stop)
    start = alignment["valid_embedding_row_start_inclusive"]
    stop = alignment["valid_embedding_row_stop_exclusive"]
    return run.target_query_embedding[start:stop], query[start:stop], alignment


def _regular_readonly_fd(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    require(path.is_absolute(), f"{label} path must be absolute")
    require(path.resolve() == path, f"{label} path may not traverse a symlink")
    lst = path.lstat()
    require(not stat.S_ISLNK(lst.st_mode), f"{label} must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        fst = os.fstat(fd)
        require(stat.S_ISREG(fst.st_mode), f"{label} must be a regular file")
        require(stat.S_IMODE(fst.st_mode) == 0o444, f"{label} mode must be 0444")
        require((fst.st_dev, fst.st_ino) == (lst.st_dev, lst.st_ino), f"{label} inode changed during open")
        return fd, fst
    except Exception:
        os.close(fd)
        raise


def _read_fd_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def load_immutable_json(path: Path, *, expected_schema: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read body and sidecar exactly once from verified immutable descriptors."""
    body = path.expanduser().absolute()
    sidecar = body.with_name(f"{body.name}.sha256")
    body_fd, body_stat = _regular_readonly_fd(body, label="receipt body")
    side_fd, side_stat = _regular_readonly_fd(sidecar, label="receipt sidecar")
    try:
        body_bytes = _read_fd_all(body_fd)
        side_bytes = _read_fd_all(side_fd)
    finally:
        os.close(body_fd)
        os.close(side_fd)
    digest = sha256_bytes(body_bytes)
    tokens = side_bytes.decode("ascii").split()
    require(len(tokens) == 2 and tokens[1] == body.name, "receipt sidecar format/name mismatch")
    require(tokens[0] == digest, "receipt body SHA disagrees with sidecar")
    try:
        payload = json.loads(body_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2ActualCpuError("receipt body is not valid JSON") from exc
    require(isinstance(payload, dict), "receipt body must be a JSON object")
    if expected_schema is not None:
        require(payload.get("schema") == expected_schema, f"unexpected receipt schema: {payload.get('schema')!r}")
    return payload, {
        "path": str(body), "sha256": digest, "mode": "0444", "size": len(body_bytes),
        "device": body_stat.st_dev, "inode": body_stat.st_ino,
        "sidecar_path": str(sidecar), "sidecar_sha256": sha256_bytes(side_bytes),
        "sidecar_device": side_stat.st_dev, "sidecar_inode": side_stat.st_ino,
        "read_once_from_verified_fd": True,
    }


def _write_fsync(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)


def write_immutable_pair(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Publish a body/sidecar pair with O_EXCL and clean rollback on conflicts."""
    body = path.expanduser().absolute()
    sidecar = body.with_name(f"{body.name}.sha256")
    body.parent.mkdir(parents=True, exist_ok=True)
    require(body.parent.resolve() == body.parent, "output directory may not traverse a symlink")
    require(not os.path.lexists(body) and not os.path.lexists(sidecar), "output body/sidecar must both be fresh")
    data = canonical_json_bytes(dict(payload))
    digest = sha256_bytes(data)
    nonce = f"{os.getpid()}.{time.time_ns()}"
    tmp_body = body.parent / f".{body.name}.{nonce}.body.tmp"
    tmp_side = body.parent / f".{body.name}.{nonce}.side.tmp"
    created_body = False
    created_side = False
    try:
        _write_fsync(tmp_body, data)
        _write_fsync(tmp_side, f"{digest}  {body.name}\n".encode("ascii"))
        os.link(tmp_body, body)
        created_body = True
        os.link(tmp_side, sidecar)
        created_side = True
        dir_fd = os.open(body.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        if created_side:
            sidecar.unlink(missing_ok=True)
        if created_body:
            body.unlink(missing_ok=True)
        raise
    finally:
        tmp_body.unlink(missing_ok=True)
        tmp_side.unlink(missing_ok=True)
    return {"path": str(body), "sha256": digest, "sidecar_path": str(sidecar),
            "sidecar_sha256": sha256_bytes(sidecar.read_bytes())}


@dataclass(frozen=True)
class Geometry:
    output_dimension: int
    source_iterations: int

    def __post_init__(self) -> None:
        require(isinstance(self.output_dimension, int) and self.output_dimension > 0, "output dimension invalid")
        require(isinstance(self.source_iterations, int) and self.source_iterations > 0, "source iterations invalid")

    def key(self) -> str:
        return f"d{self.output_dimension}-it{self.source_iterations}"

    def as_dict(self) -> dict[str, int | str]:
        return {"output_dimension": self.output_dimension, "source_iterations": self.source_iterations,
                "encoder_geometry_key": self.key()}


@dataclass(frozen=True)
class SelectorSpec:
    d_grid: tuple[int, ...] = FULL_D_GRID
    iteration_grid: tuple[int, ...] = FULL_ITERATION_GRID
    lambda_grid: tuple[float, ...] = FULL_LAMBDA_GRID
    cebra_seed: int = 42
    mode: str = "official_candidate"

    def __post_init__(self) -> None:
        require(self.mode in {"official_candidate", "engineering_smoke"}, "selector mode invalid")
        require(self.d_grid and self.iteration_grid and self.lambda_grid, "selector grids must be nonempty")
        require(len(set(self.d_grid)) == len(self.d_grid), "d grid duplicate")
        require(len(set(self.iteration_grid)) == len(self.iteration_grid), "iteration grid duplicate")
        require(len(set(self.lambda_grid)) == len(self.lambda_grid), "lambda grid duplicate")
        require(all(int(v) > 0 for v in self.d_grid + self.iteration_grid), "selector integer grid invalid")
        require(all(math.isfinite(float(v)) and float(v) > 0 for v in self.lambda_grid), "lambda grid invalid")
        if self.mode == "official_candidate":
            require(self.d_grid == FULL_D_GRID, "official d grid drift")
            require(self.iteration_grid == FULL_ITERATION_GRID, "official iteration grid drift")
            require(self.lambda_grid == FULL_LAMBDA_GRID, "official lambda grid drift")
            require(self.cebra_seed == 42, "official selector seed must be 42")

    def geometries(self) -> tuple[Geometry, ...]:
        return tuple(Geometry(d, it) for d in self.d_grid for it in self.iteration_grid)

    def as_dict(self) -> dict[str, Any]:
        return {"d_grid": list(self.d_grid), "iteration_grid": list(self.iteration_grid),
                "linear_ridge_lambda_grid": list(self.lambda_grid),
                "linear_candidate_count": len(self.geometries()) * len(self.lambda_grid),
                "knn_candidate_count": len(self.geometries()), "cebra_seed": self.cebra_seed,
                "mode": self.mode, "target_data_permitted": False}


def _matrix(value: np.ndarray, *, label: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.ndim == 2 and array.shape[0] >= 4 and array.shape[1] >= 1, f"{label} shape invalid")
    if rows is not None:
        require(array.shape[0] == rows, f"{label} row mismatch")
    require(np.isfinite(array).all(), f"{label} nonfinite")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class SourcePseudoTargetFold:
    fold_id: str
    held_source_session_id: str
    peer_source_session_ids: tuple[str, ...]
    peer_neural: tuple[np.ndarray, ...]
    peer_auxiliary: tuple[np.ndarray, ...]
    held_support_neural: np.ndarray
    held_support_auxiliary: np.ndarray
    held_query_neural: np.ndarray
    held_query_auxiliary: np.ndarray
    support_trial_count: int
    expected_support_trial_count: int
    outer_target_opened: bool = False
    formal_data_opened: bool = False
    query_neural_in_fit: bool = False
    query_auxiliary_in_fit: bool = False

    def validated(self) -> "SourcePseudoTargetFold":
        require(self.fold_id and self.held_source_session_id, "fold/session id missing")
        require(len(self.peer_source_session_ids) >= 1, "pseudo-target fold needs peer source sessions")
        require(len(set(self.peer_source_session_ids)) == len(self.peer_source_session_ids), "peer session duplicate")
        require(self.held_source_session_id not in self.peer_source_session_ids, "held source is in peer fit roster")
        require(len(self.peer_neural) == len(self.peer_auxiliary) == len(self.peer_source_session_ids),
                "peer arrays/roster length mismatch")
        for sid, x, y in zip(self.peer_source_session_ids, self.peer_neural, self.peer_auxiliary, strict=True):
            neural = _matrix(x, label=f"peer neural {sid}")
            aux = _matrix(y, label=f"peer auxiliary {sid}", rows=neural.shape[0])
        support_x = _matrix(self.held_support_neural, label="held support neural")
        _matrix(self.held_support_auxiliary, label="held support auxiliary", rows=support_x.shape[0])
        query_x = _matrix(self.held_query_neural, label="held query neural")
        query_y = _matrix(self.held_query_auxiliary, label="held query auxiliary", rows=query_x.shape[0])
        require(query_y.shape[1] == np.asarray(self.held_support_auxiliary).shape[1], "support/query auxiliary width mismatch")
        require(self.support_trial_count == self.expected_support_trial_count, "M50/M24 support trial boundary drift")
        require(self.expected_support_trial_count in {24, 50}, "only M50/M24 source pseudo-target boundaries are legal")
        require(not self.outer_target_opened and not self.formal_data_opened, "outer target/formal data touched")
        require(not self.query_neural_in_fit and not self.query_auxiliary_in_fit, "held-source query entered fit")
        return self

    def lineage(self) -> dict[str, Any]:
        self.validated()
        return {
            "fold_id": self.fold_id, "held_source_session_id": self.held_source_session_id,
            "peer_source_session_ids": list(self.peer_source_session_ids),
            "support_trial_count": self.support_trial_count,
            "outer_target_opened": False, "formal_data_opened": False,
            "query_neural_in_fit": False, "query_auxiliary_in_fit": False,
            "peer_neural_sha256": [array_sha256(x) for x in self.peer_neural],
            "peer_auxiliary_sha256": [array_sha256(y) for y in self.peer_auxiliary],
            "held_support_neural_sha256": array_sha256(self.held_support_neural),
            "held_support_auxiliary_sha256": array_sha256(self.held_support_auxiliary),
            "held_query_neural_sha256": array_sha256(self.held_query_neural),
            "held_query_auxiliary_sha256": array_sha256(self.held_query_auxiliary),
        }


@dataclass(frozen=True)
class EmbeddingRun:
    peer_fit_embeddings: tuple[np.ndarray, ...]
    target_support_embedding: np.ndarray
    target_query_embedding: np.ndarray
    fit_calls: tuple[dict[str, Any], ...]
    model_alignment: Mapping[str, Any]
    source_query_neural_seen_by_fit: bool = False
    source_query_auxiliary_seen_by_fit: bool = False


class CpuBackend(Protocol):
    identity: Mapping[str, Any]

    def run_arm(self, *, arm: str, peer_neural: Sequence[np.ndarray], peer_auxiliary: Sequence[np.ndarray],
                target_support_neural: np.ndarray, target_support_auxiliary: np.ndarray,
                target_query_neural: np.ndarray, geometry: Geometry, seed: int,
                adapt_iterations: int = DEFAULT_ADAPT_ITERATIONS,
                transform_peers: bool = True) -> EmbeddingRun: ...


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


class VendoredCebra061Backend:
    """Actual vendored CEBRA backend; importing it is an explicit runtime action."""

    def __init__(self) -> None:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "actual CPU route requires blank CUDA_VISIBLE_DEVICES")
        import cebra
        import torch
        require(getattr(cebra, "__version__", None) == VENDORED_CEBRA_VERSION, "CEBRA version drift")
        module_path = Path(cebra.__file__).resolve()
        require("cebra_exploration/third_party/cebra" in str(module_path), "CEBRA is not the vendored tree")
        torch_path = Path(torch.__file__).resolve()
        require("/.local/" not in str(torch_path), "user-site torch is forbidden")
        sklearn_impl = module_path.parent / "integrations" / "sklearn" / "cebra.py"
        model_impl = module_path.parent / "models" / "model.py"
        offset_impl = module_path.parent / "data" / "datatypes.py"
        provenance = module_path.parents[2] / "CEBRA_PROVENANCE.txt"
        require(sklearn_impl.is_file() and model_impl.is_file() and offset_impl.is_file() and provenance.is_file(),
                "vendored CEBRA implementation/provenance missing")
        self.cebra = cebra
        self.identity = {"backend": "vendored_cebra", "version": VENDORED_CEBRA_VERSION,
                         "commit": VENDORED_CEBRA_COMMIT, "module_path": str(module_path),
                         "module_sha256": sha256_bytes(module_path.read_bytes()),
                         "sklearn_implementation_path": str(sklearn_impl),
                         "sklearn_implementation_sha256": sha256_bytes(sklearn_impl.read_bytes()),
                         "model_offset_implementation_path": str(model_impl),
                         "model_offset_implementation_sha256": sha256_bytes(model_impl.read_bytes()),
                         "offset_datatype_implementation_path": str(offset_impl),
                         "offset_datatype_implementation_sha256": sha256_bytes(offset_impl.read_bytes()),
                         "provenance_path": str(provenance),
                         "provenance_sha256": sha256_bytes(provenance.read_bytes()),
                         "python": sys.version, "numpy_version": np.__version__,
                         "torch_path": str(torch_path), "torch_version": torch.__version__,
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "device": "cpu", "actual_cebra": True}

    def _estimator(self, geometry: Geometry, *, iterations: int, adapt_iterations: int,
                   batch_size: int) -> Any:
        return self.cebra.CEBRA(model_architecture=MODEL_ARCHITECTURE, device="cpu",
                                batch_size=int(batch_size), learning_rate=LEARNING_RATE,
                                output_dimension=geometry.output_dimension,
                                num_hidden_units=NUM_HIDDEN_UNITS, max_iterations=int(iterations),
                                max_adapt_iterations=int(adapt_iterations), verbose=False)

    @staticmethod
    def _transform(estimator: Any, array: np.ndarray, session_id: int | None = None) -> np.ndarray:
        kwargs = {} if session_id is None else {"session_id": session_id}
        out = np.asarray(estimator.transform(np.asarray(array, dtype=np.float64), **kwargs), dtype=np.float64)
        require(out.shape[0] == array.shape[0] and out.ndim == 2, "CEBRA embedding shape drift")
        return out

    @staticmethod
    def _model_alignment(estimator: Any) -> dict[str, Any]:
        alignment = model_alignment_from_offset(estimator.offset_)
        models = ([estimator.model_] if hasattr(estimator.model_, "get_offset")
                  else list(estimator.model_))
        require(models, "fitted estimator has no session model")
        offsets = [model_alignment_from_offset(model.get_offset()) for model in models]
        require(all(value == alignment for value in offsets), "fitted session model offsets disagree")
        return alignment

    def run_arm(self, *, arm: str, peer_neural: Sequence[np.ndarray], peer_auxiliary: Sequence[np.ndarray],
                target_support_neural: np.ndarray, target_support_auxiliary: np.ndarray,
                target_query_neural: np.ndarray, geometry: Geometry, seed: int,
                adapt_iterations: int = DEFAULT_ADAPT_ITERATIONS,
                transform_peers: bool = True) -> EmbeddingRun:
        require(arm in ALL_CONTROL_ARMS + (DERANGED_ARM,), f"unsupported actual arm {arm}")
        peers_x = [np.asarray(x, dtype=np.float64) for x in peer_neural]
        peers_y = [np.asarray(y, dtype=np.float64) for y in peer_auxiliary]
        support_x = np.asarray(target_support_neural, dtype=np.float64)
        support_y = np.asarray(target_support_auxiliary, dtype=np.float64)
        query_x = np.asarray(target_query_neural, dtype=np.float64)
        joint_batch = min(512, min([x.shape[0] for x in peers_x] + [support_x.shape[0]]))
        source_batch = min(512, min(x.shape[0] for x in peers_x))
        fit_calls: list[dict[str, Any]] = []

        def record(label: str, started: float, iterations: int) -> None:
            fit_calls.append({"label": label, "iterations": int(iterations),
                              "wall_clock_s": time.monotonic() - started})

        _seed_all(seed)
        if arm in {"cebra_joint_behavior", DERANGED_ARM}:
            estimator = self._estimator(geometry, iterations=geometry.source_iterations,
                                        adapt_iterations=adapt_iterations, batch_size=joint_batch)
            started = time.monotonic()
            estimator.fit(peers_x + [support_x], peers_y + [support_y])
            record("joint_multisession_fit", started, geometry.source_iterations)
            target_sid = len(peers_x)
            alignment = self._model_alignment(estimator)
            return EmbeddingRun(
                tuple(self._transform(estimator, x, i) for i, x in enumerate(peers_x)) if transform_peers else (),
                self._transform(estimator, support_x, target_sid),
                self._transform(estimator, query_x, target_sid), tuple(fit_calls), alignment)

        source = self._estimator(geometry, iterations=geometry.source_iterations,
                                 adapt_iterations=adapt_iterations, batch_size=source_batch)
        started = time.monotonic()
        source.fit(peers_x, peers_y)
        record(f"{arm}__source_multisession_fit", started, geometry.source_iterations)
        if arm == "cebra_frozen_source_adapt":
            estimator = self._estimator(geometry, iterations=adapt_iterations,
                                        adapt_iterations=adapt_iterations, batch_size=joint_batch)
            started = time.monotonic()
            estimator.fit(peers_x + [support_x], peers_y + [support_y],
                          freeze_sessions=list(range(len(peers_x))), init_from=source)
            record("frozen_source_joint_target_fit", started, adapt_iterations)
            target_sid = len(peers_x)
            alignment = self._model_alignment(estimator)
            return EmbeddingRun(
                tuple(self._transform(estimator, x, i) for i, x in enumerate(peers_x)) if transform_peers else (),
                self._transform(estimator, support_x, target_sid),
                self._transform(estimator, query_x, target_sid), tuple(fit_calls), alignment)

        require(arm == UNALIGNED_ARM, "unhandled actual arm")
        template = self._estimator(geometry, iterations=1, adapt_iterations=adapt_iterations,
                                   batch_size=min(512, peers_x[0].shape[0]))
        started = time.monotonic()
        template.fit(peers_x[0], peers_y[0])
        record("unaligned_template_initialisation_fit", started, 1)
        template.model_.load_state_dict(source.model_[0].state_dict())
        started = time.monotonic()
        template.fit(support_x, support_y, adapt=True)
        record("unaligned_single_session_adapt_fit", started, adapt_iterations)
        alignment = self._model_alignment(template)
        return EmbeddingRun(tuple(self._transform(source, x, i) for i, x in enumerate(peers_x)) if transform_peers else (),
                            self._transform(template, support_x), self._transform(template, query_x),
                            tuple(fit_calls), alignment)


def pooled_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    require(y.shape == p.shape and y.ndim == 2 and y.shape[0] >= 3, "R2 arrays invalid")
    denominator = float(np.square(y - y.mean(axis=0, keepdims=True)).sum())
    require(denominator > 0 and math.isfinite(denominator), "R2 denominator invalid")
    score = 1.0 - float(np.square(y - p).sum()) / denominator
    require(math.isfinite(score), "R2 is nonfinite")
    return score


def _fit_ridge(x: np.ndarray, y: np.ndarray, normalized_lambda: float) -> dict[str, np.ndarray]:
    features = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    z = (features - mean) / scale
    z1 = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
    penalty = float(normalized_lambda) * float(z.shape[0])
    regularizer = np.eye(z1.shape[1]) * penalty
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(z1.T @ z1 + regularizer, z1.T @ target)
    return {"mean": mean, "scale": scale, "weights": weights}


def _predict_ridge(x: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    return np.concatenate([z, np.ones((z.shape[0], 1))], axis=1) @ model["weights"]


def _predict_knn(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    x = np.asarray(train_x, dtype=np.float64)
    q = np.asarray(query_x, dtype=np.float64)
    require(x.shape[0] >= KNN_K, "kNN needs at least three fit rows")
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    x_norm[x_norm == 0] = 1.0
    q_norm[q_norm == 0] = 1.0
    similarity = (q / q_norm) @ (x / x_norm).T
    nearest = np.argpartition(-similarity, KNN_K - 1, axis=1)[:, :KNN_K]
    return np.asarray(train_y, dtype=np.float64)[nearest].mean(axis=1)


def _runtime_metrics(started: float, fit_calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    env_threads = {name: os.environ.get(name) for name in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    return {"wall_clock_s": time.monotonic() - started, "cebra_fit_call_count": len(fit_calls),
            "cebra_fit_calls": list(fit_calls), "python_active_thread_count": threading.active_count(),
            "thread_environment": env_threads,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


def implementation_binding() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {"core_path": str(path), "core_sha256": sha256_bytes(path.read_bytes()),
            "schema_selector": SCHEMA_SELECTOR, "schema_controls": SCHEMA_CONTROLS,
            "vendored_cebra_commit": VENDORED_CEBRA_COMMIT,
            "vendored_cebra_version": VENDORED_CEBRA_VERSION}


def _authority_guard(authority: Mapping[str, Any], folds: Sequence[SourcePseudoTargetFold], *, official: bool,
                     binding: Mapping[str, Any] | None) -> None:
    require(authority.get("outer_target_opened") is False, "source authority reports outer target opened")
    require(authority.get("formal_data_opened") is False, "source authority reports formal data opened")
    require(authority.get("source_only") is True, "source authority is not source-only")
    require(authority.get("fold_lineage") == [fold.lineage() for fold in folds], "source fold lineage differs from authority")
    if official:
        require(authority.get("status") == "ROOT_SETTLED_SOURCE_ONLY_FOLD_AUTHORITY",
                "official selector requires root-settled source authority")
        require(isinstance(binding, Mapping), "official selector requires immutable source-authority binding")
        require(binding.get("read_once_from_verified_fd") is True, "source authority was not read once from verified fd")
        require(binding.get("mode") == "0444" and _valid_sha(binding.get("sha256")),
                "source authority immutable mode/SHA binding invalid")
        require(Path(str(binding.get("path", ""))).is_absolute(), "source authority binding path must be absolute")


def execute_source_only_selector(*, folds: Sequence[SourcePseudoTargetFold], spec: SelectorSpec,
                                 backend: CpuBackend, source_authority: Mapping[str, Any],
                                 source_authority_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    require(folds, "selector needs source pseudo-target folds")
    checked = tuple(fold.validated() for fold in folds)
    require(len({fold.fold_id for fold in checked}) == len(checked), "selector fold ids duplicate")
    official = spec.mode == "official_candidate"
    _authority_guard(source_authority, checked, official=official, binding=source_authority_binding)
    actual_backend = backend.identity.get("actual_cebra") is True
    if official:
        require(actual_backend, "official selector requires actual vendored CEBRA")
    geometry_runs: list[dict[str, Any]] = []
    linear_predictions: dict[tuple[str, float], list[np.ndarray]] = {}
    linear_truth: dict[tuple[str, float], list[np.ndarray]] = {}
    knn_predictions: dict[str, list[np.ndarray]] = {}
    knn_truth: dict[str, list[np.ndarray]] = {}
    all_fit_calls: list[dict[str, Any]] = []
    for geometry in spec.geometries():
        for fold in checked:
            started = time.monotonic()
            run = backend.run_arm(arm="cebra_joint_behavior", peer_neural=fold.peer_neural,
                                  peer_auxiliary=fold.peer_auxiliary,
                                  target_support_neural=fold.held_support_neural,
                                  target_support_auxiliary=fold.held_support_auxiliary,
                                  target_query_neural=fold.held_query_neural,
                                  geometry=geometry, seed=spec.cebra_seed, transform_peers=False)
            require(not run.source_query_neural_seen_by_fit and not run.source_query_auxiliary_seen_by_fit,
                    "held-source post-M query entered selector fit")
            require(run.target_support_embedding.shape[0] == fold.held_support_auxiliary.shape[0],
                    "selector support embedding row drift")
            require(run.target_query_embedding.shape[0] == fold.held_query_auxiliary.shape[0],
                    "selector query embedding row drift")
            fit_calls = [dict(call) | {"fold_id": fold.fold_id, "geometry": geometry.key()}
                         for call in run.fit_calls]
            all_fit_calls.extend(fit_calls)
            geometry_runs.append({"fold_id": fold.fold_id, "geometry": geometry.as_dict(),
                                  **_runtime_metrics(started, fit_calls),
                                  "held_query_neural_in_fit": False, "held_query_auxiliary_in_fit": False})
            for lam in spec.lambda_grid:
                model = _fit_ridge(run.target_support_embedding, fold.held_support_auxiliary, lam)
                key = (geometry.key(), float(lam))
                query_embedding, query_truth, _ = aligned_query_arrays(run, fold.held_query_auxiliary)
                linear_predictions.setdefault(key, []).append(_predict_ridge(query_embedding, model))
                linear_truth.setdefault(key, []).append(query_truth)
            knn_predictions.setdefault(geometry.key(), []).append(
                _predict_knn(run.target_support_embedding, fold.held_support_auxiliary,
                             aligned_query_arrays(run, fold.held_query_auxiliary)[0]))
            knn_truth.setdefault(geometry.key(), []).append(
                aligned_query_arrays(run, fold.held_query_auxiliary)[1])
    linear_cells = []
    for geometry in spec.geometries():
        for lam in spec.lambda_grid:
            key = (geometry.key(), float(lam))
            linear_cells.append({"geometry": geometry.as_dict(), "normalized_lambda": float(lam),
                                 "candidate_key": f"{geometry.key()}-lambda{float(lam):.12g}-linear_ridge",
                                 "source_inner_query_pooled_r2": pooled_r2(
                                     np.concatenate(linear_truth[key]), np.concatenate(linear_predictions[key])),
                                 "target_data_used": False})
    knn_cells = []
    for geometry in spec.geometries():
        knn_cells.append({"geometry": geometry.as_dict(), "normalized_lambda": "NOT_APPLICABLE",
                          "candidate_key": f"{geometry.key()}-lambdaNA-knn_cosine_k3",
                          "source_inner_query_pooled_r2": pooled_r2(
                              np.concatenate(knn_truth[geometry.key()]),
                              np.concatenate(knn_predictions[geometry.key()])),
                          "target_data_used": False})
    linear_winner = sorted(linear_cells, key=lambda cell: (-cell["source_inner_query_pooled_r2"], cell["candidate_key"]))[0]
    knn_winner = sorted(knn_cells, key=lambda cell: (-cell["source_inner_query_pooled_r2"], cell["candidate_key"]))[0]
    return {"schema": SCHEMA_SELECTOR,
            "status": "OFFICIAL_CANDIDATE__ROOT_MINT_REQUIRED" if official else "ENGINEERING_SMOKE_ONLY__NON_AUTHORISING",
            "official_authority_minted": False, "target_execution_authorised": False,
            "mode": spec.mode, "device": "cpu", "backend": dict(backend.identity),
            "implementation_binding": implementation_binding(),
            "source_authority_binding": dict(source_authority_binding or {}),
            "source_authority_payload_sha256": sha256_bytes(canonical_json_bytes(source_authority)),
            "source_only": True, "outer_target_opened": False, "formal_data_opened": False,
            "target_discovery_performed": False, "query_neural_in_fit": False, "query_auxiliary_in_fit": False,
            "selector_spec": spec.as_dict(), "fold_count": len(checked),
            "fold_lineage": [fold.lineage() for fold in checked],
            "linear_ridge": {"all_candidates": linear_cells, "selected": linear_winner,
                             "candidate_count": len(linear_cells)},
            "knn_cosine_k3": {"all_candidates": knn_cells, "selected": knn_winner,
                              "candidate_count": len(knn_cells), "normalized_lambda": "NOT_APPLICABLE"},
            "independent_decoder_selection": True, "geometry_runs": geometry_runs,
            "cebra_fit_call_count": len(all_fit_calls),
            "expected_cebra_fit_call_count": len(checked) * len(spec.geometries()),
            "all_thresholds_frozen": False, "hard_null_threshold_status": "PENDING_REAL_EIGHT_SEED_SMOKE"}


def load_selector_winners(path: Path, *, allow_engineering: bool = False) -> tuple[Geometry, float, Geometry, dict[str, Any]]:
    payload, binding = load_immutable_json(path, expected_schema=SCHEMA_SELECTOR)
    allowed = {"OFFICIAL_SELECTOR_AUTHORITY"}
    if allow_engineering:
        allowed.add("ENGINEERING_SMOKE_ONLY__NON_AUTHORISING")
    require(payload.get("status") in allowed, "selector receipt status is not usable")
    binding["selector_status"] = payload.get("status")
    require(payload.get("outer_target_opened") is False and payload.get("formal_data_opened") is False,
            "selector receipt target/formal lineage invalid")
    spec = payload.get("selector_spec")
    require(isinstance(spec, dict), "selector spec missing")
    d_grid = tuple(int(v) for v in spec.get("d_grid", ()))
    it_grid = tuple(int(v) for v in spec.get("iteration_grid", ()))
    lambda_grid = tuple(float(v) for v in spec.get("linear_ridge_lambda_grid", ()))
    if payload.get("status") == "OFFICIAL_SELECTOR_AUTHORITY":
        require(d_grid == FULL_D_GRID and it_grid == FULL_ITERATION_GRID and lambda_grid == FULL_LAMBDA_GRID,
                "official selector grid drift")
        require(payload.get("backend", {}).get("actual_cebra") is True, "official selector is not actual CEBRA")
        require(payload.get("official_authority_minted") is True, "official selector mint flag missing")
        require(payload.get("root_authorised") is True, "official selector lacks root authorisation")
    expected_geometries = {(d, it) for d in d_grid for it in it_grid}
    expected_linear = {(d, it, lam) for d, it in expected_geometries for lam in lambda_grid}
    linear_cells = payload.get("linear_ridge", {}).get("all_candidates")
    knn_cells = payload.get("knn_cosine_k3", {}).get("all_candidates")
    require(isinstance(linear_cells, list) and isinstance(knn_cells, list), "selector candidate cells missing")
    actual_linear = {(int(cell["geometry"]["output_dimension"]),
                      int(cell["geometry"]["source_iterations"]), float(cell["normalized_lambda"]))
                     for cell in linear_cells}
    actual_knn = {(int(cell["geometry"]["output_dimension"]),
                   int(cell["geometry"]["source_iterations"])) for cell in knn_cells}
    require(len(actual_linear) == len(linear_cells) and actual_linear == expected_linear,
            "selector linear candidate coverage drift")
    require(len(actual_knn) == len(knn_cells) and actual_knn == expected_geometries,
            "selector kNN candidate coverage drift")
    require(payload["linear_ridge"].get("candidate_count") == len(linear_cells), "linear candidate count drift")
    require(payload["knn_cosine_k3"].get("candidate_count") == len(knn_cells), "kNN candidate count drift")
    linear = payload["linear_ridge"]["selected"]
    knn = payload["knn_cosine_k3"]["selected"]
    linear_geometry = Geometry(int(linear["geometry"]["output_dimension"]), int(linear["geometry"]["source_iterations"]))
    knn_geometry = Geometry(int(knn["geometry"]["output_dimension"]), int(knn["geometry"]["source_iterations"]))
    lam = float(linear["normalized_lambda"])
    require(knn.get("normalized_lambda") == "NOT_APPLICABLE", "kNN winner illegally carries ridge lambda")
    require((linear_geometry.output_dimension, linear_geometry.source_iterations, lam) in actual_linear,
            "linear winner is not a frozen candidate")
    require((knn_geometry.output_dimension, knn_geometry.source_iterations) in actual_knn,
            "kNN winner is not a frozen candidate")
    return linear_geometry, lam, knn_geometry, binding


def build_official_selector_authority(*, candidate_payload: Mapping[str, Any],
                                      root_authorised: bool) -> dict[str, Any]:
    """Promote only a complete actual/full-grid candidate; publishing stays separate."""
    require(root_authorised, "official selector promotion requires root authorisation")
    require(candidate_payload.get("schema") == SCHEMA_SELECTOR, "selector candidate schema drift")
    require(candidate_payload.get("status") == "OFFICIAL_CANDIDATE__ROOT_MINT_REQUIRED",
            "only an official candidate may be promoted")
    require(candidate_payload.get("backend", {}).get("actual_cebra") is True,
            "official selector candidate must use actual CEBRA")
    spec = candidate_payload.get("selector_spec", {})
    require(tuple(spec.get("d_grid", ())) == FULL_D_GRID, "official selector d grid drift")
    require(tuple(spec.get("iteration_grid", ())) == FULL_ITERATION_GRID, "official selector iteration grid drift")
    require(tuple(spec.get("linear_ridge_lambda_grid", ())) == FULL_LAMBDA_GRID,
            "official selector lambda grid drift")
    linear_cells = candidate_payload.get("linear_ridge", {}).get("all_candidates", ())
    knn_cells = candidate_payload.get("knn_cosine_k3", {}).get("all_candidates", ())
    require(isinstance(linear_cells, list) and len(linear_cells) == 60,
            "official selector linear candidate coverage drift")
    require(isinstance(knn_cells, list) and len(knn_cells) == 12,
            "official selector kNN candidate coverage drift")
    expected_linear = {(d, it, lam) for d in FULL_D_GRID for it in FULL_ITERATION_GRID
                       for lam in FULL_LAMBDA_GRID}
    actual_linear = {(int(cell["geometry"]["output_dimension"]),
                      int(cell["geometry"]["source_iterations"]), float(cell["normalized_lambda"]))
                     for cell in linear_cells}
    expected_knn = {(d, it) for d in FULL_D_GRID for it in FULL_ITERATION_GRID}
    actual_knn = {(int(cell["geometry"]["output_dimension"]),
                   int(cell["geometry"]["source_iterations"])) for cell in knn_cells}
    require(actual_linear == expected_linear and actual_knn == expected_knn,
            "official selector candidate keys drift")
    require(candidate_payload.get("cebra_fit_call_count") == candidate_payload.get("expected_cebra_fit_call_count"),
            "official selector fit count drift")
    require(candidate_payload.get("expected_cebra_fit_call_count") ==
            int(candidate_payload.get("fold_count", 0)) * 12,
            "official selector fold/fit topology drift")
    source_binding = candidate_payload.get("source_authority_binding", {})
    require(source_binding.get("read_once_from_verified_fd") is True and source_binding.get("mode") == "0444" and
            _valid_sha(source_binding.get("sha256")), "official selector source authority binding drift")
    require(candidate_payload.get("outer_target_opened") is False and
            candidate_payload.get("formal_data_opened") is False, "official selector touched target/formal")
    promoted = dict(candidate_payload)
    promoted.update({"status": "OFFICIAL_SELECTOR_AUTHORITY", "official_authority_minted": True,
                     "root_authorised": True,
                     "promoted_candidate_payload_sha256": sha256_bytes(canonical_json_bytes(candidate_payload)),
                     "target_execution_authorised": False,
                     "hard_null_threshold_status": "PENDING_REAL_EIGHT_SEED_SMOKE"})
    return promoted


def fixed_derangement(n_rows: int, authority_material: str) -> tuple[np.ndarray, dict[str, Any]]:
    require(n_rows >= 4, "derangement needs at least four support rows")
    require(authority_material and authority_material.strip() == authority_material, "derangement authority missing")
    digest = hashlib.sha256(authority_material.encode("utf-8")).digest()
    offset = 1 + int.from_bytes(digest[:8], "big") % (n_rows - 1)
    permutation = np.roll(np.arange(n_rows, dtype=np.int64), offset)
    require(np.all(permutation != np.arange(n_rows)), "derangement is identity at some row")
    return permutation, {"authority_material_sha256": sha256_bytes(authority_material.encode("utf-8")),
                         "permutation_sha256": array_sha256(permutation), "row_count": n_rows,
                         "nonidentity_derangement": True, "cebra_seed_independent": True}


def apply_deranged_auxiliary(*, support_neural: np.ndarray, support_auxiliary: np.ndarray,
                             permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    neural = np.ascontiguousarray(np.asarray(support_neural, dtype=np.float64))
    labels = np.ascontiguousarray(np.asarray(support_auxiliary, dtype=np.float64))
    perm = np.asarray(permutation, dtype=np.int64)
    require(perm.shape == (labels.shape[0],), "derangement permutation shape drift")
    require(np.array_equal(np.sort(perm), np.arange(labels.shape[0])), "derangement is not a permutation")
    require(np.all(perm != np.arange(labels.shape[0])), "derangement must be nonidentity at every row")
    deranged = np.ascontiguousarray(labels[perm])
    before_multiset = sorted(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in labels)
    after_multiset = sorted(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in deranged)
    require(before_multiset == after_multiset, "derangement changed auxiliary label multiset")
    proof = {"support_neural_sha256_before": array_sha256(neural),
             "support_neural_sha256_after": array_sha256(neural.copy()),
             "support_neural_rows_exact_equal": True,
             "auxiliary_sha256_before": array_sha256(labels), "auxiliary_sha256_after": array_sha256(deranged),
             "auxiliary_label_multiset_sha256_before": sha256_bytes(canonical_json_bytes(before_multiset)),
             "auxiliary_label_multiset_sha256_after": sha256_bytes(canonical_json_bytes(after_multiset)),
             "auxiliary_label_multiset_exact_equal": True, "only_auxiliary_row_correspondence_changed": True}
    return neural.copy(), deranged, proof


def verify_derangement_arrays(*, neural_before: np.ndarray, neural_after: np.ndarray,
                              auxiliary_before: np.ndarray, auxiliary_after: np.ndarray,
                              permutation: np.ndarray) -> dict[str, Any]:
    """Adversarially useful verifier for externally materialised hard-null arrays."""
    before_x = np.ascontiguousarray(np.asarray(neural_before, dtype=np.float64))
    after_x = np.ascontiguousarray(np.asarray(neural_after, dtype=np.float64))
    before_y = np.ascontiguousarray(np.asarray(auxiliary_before, dtype=np.float64))
    after_y = np.ascontiguousarray(np.asarray(auxiliary_after, dtype=np.float64))
    perm = np.asarray(permutation, dtype=np.int64)
    require(np.array_equal(before_x, after_x), "derangement changed support neural rows")
    require(perm.shape == (before_y.shape[0],), "derangement permutation shape drift")
    require(np.array_equal(np.sort(perm), np.arange(before_y.shape[0])), "derangement is not a permutation")
    require(np.all(perm != np.arange(before_y.shape[0])), "derangement is identity at some row")
    require(np.array_equal(after_y, before_y[perm]), "deranged auxiliary rows do not match permutation")
    before_multiset = sorted(hashlib.sha256(row.tobytes()).hexdigest() for row in before_y)
    after_multiset = sorted(hashlib.sha256(row.tobytes()).hexdigest() for row in after_y)
    require(before_multiset == after_multiset, "derangement changed auxiliary label multiset")
    return {"support_neural_rows_exact_equal": True, "auxiliary_label_multiset_exact_equal": True,
            "only_auxiliary_row_correspondence_changed": True,
            "permutation_sha256": array_sha256(perm)}


def make_synthetic_control_fold(*, seed_material: str = "track-b-v2-actual-controls-v1",
                                peer_widths: tuple[int, ...] = (8, 11), target_width: int = 13,
                                peer_rows: int = 96, support_rows: int = 48,
                                query_rows: int = 48) -> SourcePseudoTargetFold:
    require(peer_rows >= 8 and support_rows >= 8 and query_rows >= 8, "synthetic control rows too small")
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    total = max(peer_rows, support_rows + query_rows)
    t = np.linspace(0.0, 6.0 * np.pi, total, endpoint=False)
    latent = np.stack([np.cos(t), np.sin(t)], axis=1)
    peers_x = []
    peers_y = []
    for width in peer_widths:
        mix = rng.normal(size=(2, width))
        peers_x.append(np.ascontiguousarray(latent[:peer_rows] @ mix + 0.03 * rng.normal(size=(peer_rows, width))))
        peers_y.append(np.ascontiguousarray(latent[:peer_rows]))
    target_mix = rng.normal(size=(2, target_width))
    target = latent[:support_rows + query_rows] @ target_mix
    target += 0.03 * rng.normal(size=target.shape)
    return SourcePseudoTargetFold(
        fold_id="synthetic_source_only_fold0", held_source_session_id="synthetic_held_source",
        peer_source_session_ids=tuple(f"synthetic_peer_{i}" for i in range(len(peer_widths))),
        peer_neural=tuple(peers_x), peer_auxiliary=tuple(peers_y),
        held_support_neural=np.ascontiguousarray(target[:support_rows]),
        held_support_auxiliary=np.ascontiguousarray(latent[:support_rows]),
        held_query_neural=np.ascontiguousarray(target[support_rows:support_rows + query_rows]),
        held_query_auxiliary=np.ascontiguousarray(latent[support_rows:support_rows + query_rows]),
        support_trial_count=50, expected_support_trial_count=50).validated()


def build_source_authority(folds: Sequence[SourcePseudoTargetFold], *, engineering: bool) -> dict[str, Any]:
    return {"schema": "track_b_v2_source_fold_array_authority_v1",
            "status": "ENGINEERING_SYNTHETIC_SOURCE_ONLY" if engineering else "ROOT_SETTLED_SOURCE_ONLY_FOLD_AUTHORITY",
            "source_only": True, "outer_target_opened": False, "formal_data_opened": False,
            "fold_lineage": [fold.lineage() for fold in folds]}


def _score_control_run(run: EmbeddingRun, peer_auxiliary: Sequence[np.ndarray], query_y: np.ndarray,
                       *, normalized_lambda: float, decoder: str) -> tuple[float, float, dict[str, Any]]:
    source_x = np.concatenate(run.peer_fit_embeddings)
    source_y = np.concatenate([np.asarray(y, dtype=np.float64) for y in peer_auxiliary])
    scored_query_embedding, scored_query_y, alignment = aligned_query_arrays(run, query_y)
    require(scored_query_embedding.shape[0] >= 3, "offset10 query has too few score-eligible rows")
    if decoder == "linear_ridge":
        model = _fit_ridge(source_x, source_y, normalized_lambda)
        source_pred = _predict_ridge(source_x, model)
        query_pred = _predict_ridge(scored_query_embedding, model)
    else:
        require(decoder == "knn_cosine_k3", "unknown decoder")
        source_pred = _predict_knn(source_x, source_y, source_x)
        query_pred = _predict_knn(source_x, source_y, scored_query_embedding)
    return pooled_r2(source_y, source_pred), pooled_r2(scored_query_y, query_pred), {
        "source_embedding_sha256": array_sha256(source_x),
        "target_support_embedding_sha256": array_sha256(run.target_support_embedding),
        "target_query_embedding_full_sha256": array_sha256(run.target_query_embedding),
        "target_query_embedding_scored_sha256": array_sha256(scored_query_embedding),
        "target_query_prediction_sha256": array_sha256(query_pred),
        "query_alignment": alignment}


def execute_synthetic_controls(*, selector_path: Path, backend: CpuBackend,
                               fold: SourcePseudoTargetFold, seeds: tuple[int, ...] = CONTROL_SEEDS,
                               adapt_iterations: int = DEFAULT_ADAPT_ITERATIONS,
                               allow_engineering_selector: bool = False,
                               enforce_positive_gate: bool = True) -> dict[str, Any]:
    fold.validated()
    require(not fold.outer_target_opened and not fold.formal_data_opened, "synthetic control cannot touch real target/formal")
    require(seeds and len(set(seeds)) == len(seeds), "control seeds invalid")
    if seeds != CONTROL_SEEDS:
        require(allow_engineering_selector and not enforce_positive_gate,
                "only nonauthorising engineering smoke may use fewer than eight seeds")
    linear_geometry, lam, knn_geometry, selector_binding = load_selector_winners(
        selector_path, allow_engineering=allow_engineering_selector)
    geometry_by_decoder = {"linear_ridge": linear_geometry, "knn_cosine_k3": knn_geometry}
    unique_geometries = {geometry.key(): geometry for geometry in geometry_by_decoder.values()}
    measurements: list[dict[str, Any]] = []
    hard_null: list[dict[str, Any]] = []
    all_fit_calls: list[dict[str, Any]] = []
    permutation, permutation_authority = fixed_derangement(
        fold.held_support_auxiliary.shape[0], array_sha256(fold.held_support_auxiliary))
    unchanged_neural, deranged_aux, derangement_proof = apply_deranged_auxiliary(
        support_neural=fold.held_support_neural, support_auxiliary=fold.held_support_auxiliary,
        permutation=permutation)
    require(array_sha256(unchanged_neural) == array_sha256(fold.held_support_neural), "hard-null neural drift")
    for seed in seeds:
        for geometry_key, geometry in unique_geometries.items():
            for arm in ALL_CONTROL_ARMS + (DERANGED_ARM,):
                labels = deranged_aux if arm == DERANGED_ARM else fold.held_support_auxiliary
                started = time.monotonic()
                run = backend.run_arm(arm=arm, peer_neural=fold.peer_neural,
                                      peer_auxiliary=fold.peer_auxiliary,
                                      target_support_neural=fold.held_support_neural,
                                      target_support_auxiliary=labels,
                                      target_query_neural=fold.held_query_neural,
                                      geometry=geometry, seed=seed, adapt_iterations=adapt_iterations)
                require(not run.source_query_neural_seen_by_fit and not run.source_query_auxiliary_seen_by_fit,
                        "control query entered fit")
                calls = [dict(call) | {"seed": seed, "arm": arm, "geometry": geometry_key}
                         for call in run.fit_calls]
                all_fit_calls.extend(calls)
                runtime = _runtime_metrics(started, calls)
                for decoder, selected_geometry in geometry_by_decoder.items():
                    if selected_geometry.key() != geometry_key:
                        continue
                    source_r2, target_r2, hashes = _score_control_run(
                        run, fold.peer_auxiliary, fold.held_query_auxiliary,
                        normalized_lambda=lam, decoder=decoder)
                    row = {"arm": arm, "seed": seed, "decoder": decoder,
                           "geometry": selected_geometry.as_dict(),
                           "normalized_lambda": lam if decoder == "linear_ridge" else "NOT_APPLICABLE",
                           "source_query_r2": source_r2, "target_query_r2": target_r2,
                           "query_neural_in_fit": False, "query_auxiliary_in_fit": False,
                           "support_index_sha256": index_sha256(range(fold.held_support_neural.shape[0])),
                           "query_index_sha256": index_sha256(range(
                               fold.held_support_neural.shape[0],
                               fold.held_support_neural.shape[0] + fold.held_query_neural.shape[0])),
                           "runtime": runtime, **hashes}
                    if arm == DERANGED_ARM:
                        hard_null.append(row)
                    else:
                        measurements.append(row)
    expected_measurements = len(seeds) * len(ALL_CONTROL_ARMS) * 2
    expected_hard_null = len(seeds) * 2
    require(len(measurements) == expected_measurements, "ordinary control coverage drift")
    require(len(hard_null) == expected_hard_null, "hard-null coverage drift")
    positive_failures = [row for row in measurements if row["arm"] in POSITIVE_ARMS
                         and row["target_query_r2"] < POSITIVE_MIN_R2]
    if enforce_positive_gate:
        require(not positive_failures, "positive synthetic control failed frozen R2 threshold")
    actual = backend.identity.get("actual_cebra") is True
    expected_actual_fit_calls = len(seeds) * len(unique_geometries) * 7
    if actual:
        require(len(all_fit_calls) == expected_actual_fit_calls, "actual control CEBRA.fit call count drift")
    selector_official = selector_binding.get("selector_status") == "OFFICIAL_SELECTOR_AUTHORITY"
    real_complete = (seeds == CONTROL_SEEDS and actual and selector_official and enforce_positive_gate)
    return {"schema": SCHEMA_CONTROLS,
            "status": "REAL_EIGHT_SEED_SMOKE_COMPLETE__THRESHOLD_NOT_FROZEN" if
                      real_complete
                      else "ENGINEERING_SMOKE_ONLY__NON_AUTHORISING",
            "official_authority_minted": False, "target_execution_authorised": False,
            "selector_binding": selector_binding, "backend": dict(backend.identity), "device": "cpu",
            "implementation_binding": implementation_binding(),
            "source_only_synthetic": True, "outer_target_opened": False, "formal_data_opened": False,
            "target_discovery_performed": False, "seeds": list(seeds),
            "positive_threshold": POSITIVE_MIN_R2, "positive_gate_enforced": enforce_positive_gate,
            "positive_failure_count": len(positive_failures),
            "unaligned_role": "diagnostic_distribution_only__not_a_hard_null_gate",
            "unaligned_threshold": None, "linear_ridge_geometry": linear_geometry.as_dict(),
            "linear_ridge_normalized_lambda": lam, "knn_cosine_k3_geometry": knn_geometry.as_dict(),
            "knn_normalized_lambda": "NOT_APPLICABLE", "geometry_cli_override_permitted": False,
            "measurements": measurements, "hard_null_measurements": hard_null,
            "derangement_authority": permutation_authority | derangement_proof,
            "hard_null_threshold_status": "PENDING_SEPARATE_IMMUTABLE_FREEZE_AFTER_REVIEW",
            "hard_null_threshold": None, "cebra_fit_call_count": len(all_fit_calls),
            "expected_actual_cebra_fit_call_count": expected_actual_fit_calls,
            "fit_calls": all_fit_calls, "threshold_frozen_before_real_target": False}


def validate_real_eight_seed_control_smoke(payload: Mapping[str, Any]) -> None:
    require(payload.get("schema") == SCHEMA_CONTROLS, "control smoke schema drift")
    require(payload.get("status") == "REAL_EIGHT_SEED_SMOKE_COMPLETE__THRESHOLD_NOT_FROZEN",
            "threshold requires actual complete eight-seed smoke")
    require(payload.get("backend", {}).get("actual_cebra") is True, "control smoke is not actual CEBRA")
    require(payload.get("selector_binding", {}).get("selector_status") == "OFFICIAL_SELECTOR_AUTHORITY",
            "control smoke lacks official selector geometry")
    require(payload.get("outer_target_opened") is False and payload.get("formal_data_opened") is False and
            payload.get("target_discovery_performed") is False, "control smoke touched target/formal")
    require(payload.get("positive_gate_enforced") is True and payload.get("positive_failure_count") == 0,
            "positive control gate was not passed")
    ordinary = payload.get("measurements")
    hard = payload.get("hard_null_measurements")
    require(isinstance(ordinary, list) and isinstance(hard, list), "control measurements missing")
    expected_ordinary = {(arm, seed, decoder) for arm in ALL_CONTROL_ARMS for seed in CONTROL_SEEDS
                         for decoder in ("linear_ridge", "knn_cosine_k3")}
    actual_ordinary = {(row.get("arm"), row.get("seed"), row.get("decoder")) for row in ordinary}
    require(len(ordinary) == len(actual_ordinary) == 48 and actual_ordinary == expected_ordinary,
            "ordinary control coverage drift")
    expected_hard = {(seed, decoder) for seed in CONTROL_SEEDS for decoder in ("linear_ridge", "knn_cosine_k3")}
    actual_hard = {(row.get("seed"), row.get("decoder")) for row in hard}
    require(len(hard) == len(actual_hard) == 16 and actual_hard == expected_hard,
            "hard-null control coverage drift")
    require(all(row.get("query_neural_in_fit") is False and row.get("query_auxiliary_in_fit") is False
                for row in ordinary + hard), "control query entered fit")
    proof = payload.get("derangement_authority", {})
    require(proof.get("nonidentity_derangement") is True and proof.get("cebra_seed_independent") is True,
            "hard-null permutation authority invalid")
    require(proof.get("support_neural_rows_exact_equal") is True and
            proof.get("auxiliary_label_multiset_exact_equal") is True and
            proof.get("only_auxiliary_row_correspondence_changed") is True,
            "hard-null exact-array proof invalid")
    require(payload.get("cebra_fit_call_count") == payload.get("expected_actual_cebra_fit_call_count"),
            "control CEBRA.fit count drift")


def freeze_hard_null_threshold(*, controls_payload: Mapping[str, Any], controls_binding: Mapping[str, Any],
                               threshold: float, target_opened_before_freeze: bool,
                               root_authorised: bool) -> dict[str, Any]:
    require(root_authorised, "hard-null threshold freeze requires root authorisation")
    validate_real_eight_seed_control_smoke(controls_payload)
    require(controls_binding.get("read_once_from_verified_fd") is True and
            controls_binding.get("mode") == "0444" and _valid_sha(controls_binding.get("sha256")),
            "threshold freeze requires immutable verified control receipt")
    require(not target_opened_before_freeze, "hard-null threshold was frozen after target access")
    require(math.isfinite(float(threshold)) and float(threshold) < POSITIVE_MIN_R2,
            "hard-null threshold must be finite and below positive threshold")
    rows = controls_payload.get("hard_null_measurements")
    require(isinstance(rows, list) and len(rows) == 16, "hard-null freeze needs 8 seeds x 2 decoders")
    require({int(row["seed"]) for row in rows} == set(CONTROL_SEEDS), "hard-null seed coverage drift")
    require({row["decoder"] for row in rows} == {"linear_ridge", "knn_cosine_k3"}, "hard-null decoder coverage drift")
    hard_scores = [float(row["target_query_r2"]) for row in rows]
    require(all(math.isfinite(score) for score in hard_scores), "hard-null scores must be finite")
    require(max(hard_scores) < float(threshold), "frozen hard-null threshold does not reject every smoke score")
    return {"schema": SCHEMA_HARD_NULL, "status": "ROOT_AUTHORISED_THRESHOLD_FREEZE_CANDIDATE__MINT_REQUIRED",
            "official_authority_minted": False,
            "controls_payload_sha256": sha256_bytes(canonical_json_bytes(controls_payload)),
            "controls_receipt_binding": dict(controls_binding),
            "threshold": float(threshold), "target_opened_before_freeze": False,
            "threshold_may_be_relaxed_or_backfilled": False, "seeds": list(CONTROL_SEEDS),
            "decoders": ["linear_ridge", "knn_cosine_k3"],
            "unaligned_remains_non_gating_diagnostic": True}
