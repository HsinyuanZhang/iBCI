#!/usr/bin/env python3
"""Run the CSCS cross-session consistency screen on CPU only.

Frozen protocol: ``sua_exploration/docs/CROSS_SESSION_CONSISTENCY_SCREEN_PROTOCOL_20260813.md``.

The screen measures cross-session consistency of the pooled representation ``y_hat`` of paper
eq. (7) -- post-FFN, pre-readout -- for the sealed A2-v2 source-trained T4/Z4 bundles, against a
matched permutation null and a within-session split-half ceiling, and relates it to the sealed
external sub-M outcomes.

It trains nothing.  There is no optimizer, no backward pass, and no checkpoint write.  It never
resolves a sealed sub-C formal-test NWB path.  CUDA is disabled before Torch is imported and the
runner fails closed if a CUDA device is nevertheless visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Must precede Torch and the scientific stack.  Verified again after import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for _path in (REPO_ROOT, SUA_ROOT, SUA_ROOT / "scripts", REPO_ROOT / "streaming_calibration_exp"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import importlib.util  # noqa: E402

from mc_maze import a2_matched_subject_shift_v2_core as core  # noqa: E402
from mc_maze import cross_session_consistency_screen as cscs  # noqa: E402
from mc_maze.multisession_datamodule import session_name_from_path  # noqa: E402
from scripts.eval_adaptation_dandi688 import (  # noqa: E402
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    attach_side_features,
    build_calib_trials_for_indices,
    load_session_with_trials,
)
from scripts.select_gradient_free_protocol_dandi688 import load_frozen_model  # noqa: E402


PROTOCOL_PATH = SUA_ROOT / "docs" / "CROSS_SESSION_CONSISTENCY_SCREEN_PROTOCOL_20260813.md"
RESULT_ROOT = SUA_ROOT / "results" / "cross_session_consistency_screen"
CELL_ROOT = RESULT_ROOT / "cells"
RECEIPT_PATH = RESULT_ROOT / "cross_session_consistency_screen_receipt.json"
EXTERNAL_RECEIPT_ROOT = SUA_ROOT / "results" / "a2_matched_subject_shift_v2"

YHAT_HOOK_MODULE_PATH = "student.decoder.fc_out"

# Every file whose content determines a number in the receipt is hashed into the receipt.
IMPLEMENTATION_FILES: tuple[str, ...] = (
    "sua_exploration/mc_maze/cross_session_consistency_screen.py",
    "sua_exploration/scripts/run_cross_session_consistency_screen.py",
    "sua_exploration/tests/test_cross_session_consistency_screen.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/scripts/a2_matched_subject_shift_v2_score.py",
    "sua_exploration/scripts/eval_adaptation_dandi688.py",
    "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
)
FORWARD_BATCH_WINDOWS = 512
PCA_DIMS: tuple[int, ...] = (8, 16, 32, 64)
PRIMARY_PCA_DIM = cscs.PRIMARY_PCA_DIM
PRIMARY_NULL = "condition_pairing_permutation"
SECONDARY_NULL = "direction_label_permutation"
SOURCE_ARMS: tuple[str, ...] = ("source_t4", "source_z4")


# ======================================================================================
# Provenance and environment
# ======================================================================================
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_bindings() -> dict[str, str]:
    """SHA-256 of every source file this screen's numbers depend on."""
    bindings: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative
        cscs.require(path.is_file(), f"implementation file missing: {relative}")
        bindings[relative] = sha256_file(path)
    return bindings


def environment_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_num_threads": int(torch.get_num_threads()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "pythonnouserssite": os.environ.get("PYTHONNOUSERSITE", "<unset>"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    """Publish one write-once artifact plus a required SHA-256 sidecar."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), 0o444)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(f"{digest}  {target.name}\n".encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), 0o444)
    return target, sidecar, digest


def assert_cpu_only() -> None:
    cscs.require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 is mandatory")
    cscs.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty")
    cscs.require(not torch.cuda.is_available(), "a CUDA device is visible; this screen is CPU-only")


# ======================================================================================
# Session data
# ======================================================================================
class SessionData:
    """One loaded session with everything the screen needs and nothing it does not."""

    __slots__ = (
        "name", "domain", "nwb_path", "nwb_sha256", "n_units", "neural", "calib_trials",
        "side_features", "selection", "plan", "condition_ids", "mean_calibration_rate",
        "first30_finite_target_dir_count",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def receipt(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "nwb_path": str(self.nwb_path),
            "nwb_sha256": self.nwb_sha256,
            "n_units": int(self.n_units),
            "query_trial_count": int(self.selection["query_trial_count"]),
            "eligible_trials_per_direction": {str(k): int(v) for k, v in self.selection["eligible_counts"].items()},
            "complete_eight_direction_grid": bool(self.selection["complete_eight_direction_grid"]),
            "available_directions": [int(d) for d in self.selection["available_directions"]],
            "condition_count": int(self.condition_ids.size),
            "window_count": int(self.plan["starts"].size),
            "window_start_sha256": cscs.array_sha256(self.plan["starts"]),
            "mean_calibration_prefix_rate": float(self.mean_calibration_rate),
            "first30_finite_target_dir_count": int(self.first30_finite_target_dir_count),
            "first30_target_dir_all_finite": bool(
                self.first30_finite_target_dir_count == core.ACTIVITY_CALIBRATION_TRIALS
            ),
        }


def load_session(nwb_path: Path, domain: str, behavior_stats, side_stats, *, hash_nwb: bool) -> SessionData:
    behavior_mean, behavior_std = behavior_stats
    side_mean, side_std = side_stats
    name = session_name_from_path(nwb_path)
    cscs.assert_sessions_not_sealed([name], label="CSCS session roster")
    record = load_session_with_trials(
        nwb_path, core.BIN_SIZE_MS, core.WINDOW_SIZE_BINS, core.ACTIVITY_CALIBRATION_TRIALS,
        core.TRIAL_LENGTH_BINS, PAD_VALUE, behavior_mean, behavior_std,
        trial_result_filter="R", cache_dir=None, signal_view="sua",
    )
    cscs.require(record["name"] == name, "session identity drift")
    # Same M30 support and post-30 query boundary as the sealed A2-v2 scorer.  The scorer's
    # strict all-finite first-30 label requirement governs exactly the two domains it scores;
    # sub-C source-*train* sessions were never subject to it, and one of them
    # (sub-C_ses-CO-20150313) has a single non-finite target_dir in its calibration prefix that
    # the sealed side-feature path already absorbs.  The count is recorded either way.
    prefix = record["trials"][: core.ACTIVITY_CALIBRATION_TRIALS]
    finite_labels = sum(
        1 for row in prefix
        if row.get("target_dir") is not None and np.isfinite(float(row["target_dir"]))
    )
    core.trial30_semantics_from_trials(
        record["trials"],
        require_target_labels=(domain in {"source_val", "external_subject_M"}),
        session=name,
    )
    record["calib_trials"] = build_calib_trials_for_indices(
        record, list(range(core.ACTIVITY_CALIBRATION_TRIALS)), core.ACTIVITY_CALIBRATION_TRIALS
    )
    side_by_arm: dict[str, np.ndarray] = {}
    for arm, group in (("source_t4", "t4"), ("source_z4", "z4")):
        enriched = attach_side_features(
            record, nwb_path, side_feature_group=group, waveform_feature_group="t4",
            pool_size=core.SIDE_FEATURE_POOL_TRIALS, permutation_seed=None,
            mean=side_mean, std=side_std, cache_dir=None,
        )
        side_by_arm[arm] = np.ascontiguousarray(enriched["side_features"])
    selection = cscs.select_condition_trials(record["trials"])
    cscs.require(bool(selection["selected_query_positions"]), f"{name}: no direction reaches the trial quota")
    plan = cscs.condition_window_plan(record["trials"], selection)
    condition_ids = np.unique(cscs.condition_index(plan["direction_index"], plan["offset_index"]))
    calib = np.asarray(record["calib_trials"], dtype=np.float32)
    finite = calib[calib != PAD_VALUE]
    return SessionData(
        name=name,
        domain=domain,
        nwb_path=nwb_path,
        nwb_sha256=sha256_file(nwb_path) if hash_nwb else None,
        n_units=int(record["n_units"]),
        neural=np.ascontiguousarray(record["neural"], dtype=np.float32),
        calib_trials=calib,
        side_features=side_by_arm,
        selection=selection,
        plan=plan,
        condition_ids=condition_ids,
        mean_calibration_rate=float(finite.mean()) if finite.size else 0.0,
        first30_finite_target_dir_count=int(finite_labels),
    )


# ======================================================================================
# y_hat extraction
# ======================================================================================
class YhatCapture:
    """Forward pre-hook on the readout layer: its input is exactly eq. (7)'s y_hat."""

    def __init__(self, model) -> None:
        target = model.student.decoder.fc_out
        self.buffer: list[torch.Tensor] = []
        self.enabled = False
        self._handle = target.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        if self.enabled:
            self.buffer.append(inputs[0].detach().clone())
        return None

    def drain(self) -> torch.Tensor:
        stacked = torch.cat(self.buffer, dim=0)
        self.buffer.clear()
        return stacked

    def close(self) -> None:
        self._handle.remove()


@torch.no_grad()
def extract_yhat(model, session: SessionData, arm: str, *, parity_check: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-window y_hat for one session's condition grid under one frozen checkpoint."""
    cscs.require(model.student.decoder_mode == "coupled", "CSCS assumes the coupled decoder path")
    side = torch.from_numpy(session.side_features[arm]).float().unsqueeze(0)
    calib = torch.from_numpy(session.calib_trials).float().unsqueeze(0)
    identity = model.student.compute_identity(calib, side_features=side, electrode_ids=None)
    starts = session.plan["starts"]
    capture = YhatCapture(model)
    parity = {"performed": False, "bitwise_identical": None}
    try:
        capture.enabled = True
        chunks: list[np.ndarray] = []
        for begin in range(0, starts.size, FORWARD_BATCH_WINDOWS):
            block = starts[begin : begin + FORWARD_BATCH_WINDOWS]
            neural = torch.from_numpy(
                np.stack([session.neural[start : start + WINDOW_SIZE] for start in block])
            ).float()
            model.student.decode_with_identity(neural, identity.expand(neural.shape[0], -1, -1))
            captured = capture.drain()
            chunks.append(captured.reshape(captured.shape[0], -1).numpy().astype(np.float64))
        values = np.concatenate(chunks, axis=0)

        if parity_check:
            # Prove identity reuse is not an approximation: one batch through the full
            # student(...) path must give a bitwise-identical y_hat.
            block = starts[:FORWARD_BATCH_WINDOWS]
            neural = torch.from_numpy(
                np.stack([session.neural[start : start + WINDOW_SIZE] for start in block])
            ).float()
            batch_side = side.expand(neural.shape[0], -1, -1)
            model.student(
                neural, calib_trials=calib.expand(neural.shape[0], -1, -1, -1),
                side_features=batch_side,
                decoder_key_features=model.decoder_key_features(batch_side),
                electrode_ids=None,
            )
            full_path = capture.drain().reshape(block.size, -1).numpy().astype(np.float64)
            parity = {
                "performed": True,
                "bitwise_identical": bool(np.array_equal(full_path, values[: block.size])),
                "max_abs_difference": float(np.abs(full_path - values[: block.size]).max()),
                "windows_compared": int(block.size),
            }
    finally:
        capture.enabled = False
        capture.close()
    cscs.require(values.shape[0] == starts.size, "y_hat row count drift")
    cscs.require(np.isfinite(values).all(), "y_hat contains non-finite values")
    return values, parity


# ======================================================================================
# Pair engine
# ======================================================================================
class MapOperator:
    """Cached cross-validated OLS prediction operator for one fixed map source matrix."""

    def __init__(self, x: np.ndarray, folds: np.ndarray, *, n_folds: int = cscs.CV_FOLDS) -> None:
        self.n_folds = n_folds
        self.train_index: list[np.ndarray] = []
        self.test_index: list[np.ndarray] = []
        self.projector: list[np.ndarray] = []
        for fold in range(n_folds):
            test = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            cscs.require(test.size > 0, f"fold {fold} is empty")
            cscs.require(train.size > x.shape[1], f"fold {fold} has fewer training rows than map parameters")
            design_train = np.concatenate([np.ones((train.size, 1)), x[train]], axis=1)
            design_test = np.concatenate([np.ones((test.size, 1)), x[test]], axis=1)
            self.projector.append(design_test @ np.linalg.pinv(design_train))
            self.train_index.append(train)
            self.test_index.append(test)

    def residual_ss(self, targets: Sequence[np.ndarray]) -> np.ndarray:
        """Held-out residual sum of squares for a batch of target matrices sharing this source."""
        count = len(targets)
        width = targets[0].shape[1]
        total = np.zeros(count, dtype=np.float64)
        for fold in range(self.n_folds):
            train, test = self.train_index[fold], self.test_index[fold]
            stacked_train = np.concatenate([target[train] for target in targets], axis=1)
            stacked_test = np.concatenate([target[test] for target in targets], axis=1)
            residual = (stacked_test - self.projector[fold] @ stacked_train) ** 2
            total += residual.reshape(test.size, count, width).sum(axis=(0, 2))
        return total


def _null_orders(condition_ids: np.ndarray, null_kind: str, seed_parts: Sequence[Any], n_permutations: int) -> list[np.ndarray]:
    permute = cscs.NULL_KINDS[null_kind]
    orders = []
    for index in range(n_permutations):
        rng = np.random.Generator(np.random.PCG64(cscs.derived_seed(null_kind, *seed_parts, index)))
        orders.append(permute(condition_ids, rng))
    return orders


def directed_consistency(
    operator: MapOperator,
    target: np.ndarray,
    condition_ids: np.ndarray,
    *,
    null_kind: str,
    seed_parts: Sequence[Any],
    n_permutations: int,
) -> dict[str, float]:
    """Raw / null / adv for one ordered pair, with the null batched against a cached operator."""
    ss_total = float(np.sum((target - target.mean(axis=0)) ** 2))
    cscs.require(ss_total > 0.0, "target condition means have zero total variance")
    orders = _null_orders(condition_ids, null_kind, seed_parts, n_permutations)
    batch = [target] + [target[order] for order in orders]
    residuals = operator.residual_ss(batch)
    values = 1.0 - residuals / ss_total
    raw = float(values[0])
    null = float(np.mean(values[1:]))
    return {"raw": raw, "null": null, "adv": float(raw - null)}


class PairEngine:
    """Symmetric pair consistency over a fixed set of projected condition-mean matrices."""

    def __init__(
        self,
        matrices: Mapping[tuple[str, str], np.ndarray],
        condition_ids: Mapping[str, np.ndarray],
        *,
        null_kind: str,
        n_permutations: int,
        seed_prefix: Sequence[Any],
    ) -> None:
        self.matrices = matrices
        self.condition_ids = condition_ids
        self.null_kind = null_kind
        self.n_permutations = n_permutations
        self.seed_prefix = list(seed_prefix)
        self._operators: dict[tuple[str, str, bytes], MapOperator] = {}

    def _shared(self, session_a: str, session_b: str) -> np.ndarray:
        return cscs.shared_condition_ids(self.condition_ids[session_a], self.condition_ids[session_b])

    def _view(self, session: str, view: str, shared: np.ndarray) -> np.ndarray:
        matrix = self.matrices[(session, view)]
        if shared.size == self.condition_ids[session].size:
            return matrix
        return cscs.restrict_to_conditions(matrix, self.condition_ids[session], shared)

    def _operator(self, session: str, view: str, shared: np.ndarray) -> MapOperator:
        key = (session, view, shared.tobytes())
        operator = self._operators.get(key)
        if operator is None:
            operator = MapOperator(self._view(session, view, shared), cscs.fold_assignment(shared))
            self._operators[key] = operator
        return operator

    def symmetric(self, session_a: str, view_a: str, session_b: str, view_b: str) -> dict[str, Any]:
        shared = self._shared(session_a, session_b)
        forward = directed_consistency(
            self._operator(session_a, view_a, shared), self._view(session_b, view_b, shared), shared,
            null_kind=self.null_kind, n_permutations=self.n_permutations,
            seed_parts=[*self.seed_prefix, session_a, view_a, session_b, view_b],
        )
        backward = directed_consistency(
            self._operator(session_b, view_b, shared), self._view(session_a, view_a, shared), shared,
            null_kind=self.null_kind, n_permutations=self.n_permutations,
            seed_parts=[*self.seed_prefix, session_b, view_b, session_a, view_a],
        )
        return {
            "session_a": session_a, "session_b": session_b,
            "n_conditions": int(shared.size),
            "raw": 0.5 * (forward["raw"] + backward["raw"]),
            "null": 0.5 * (forward["null"] + backward["null"]),
            "adv": 0.5 * (forward["adv"] + backward["adv"]),
            "forward": forward, "backward": backward,
        }


def _summarize(values: Iterable[Mapping[str, float]]) -> dict[str, float]:
    rows = list(values)
    cscs.require(bool(rows), "cannot summarize an empty pair set")
    return {
        "n": len(rows),
        "raw": float(np.mean([row["raw"] for row in rows])),
        "null": float(np.mean([row["null"] for row in rows])),
        "adv": float(np.mean([row["adv"] for row in rows])),
        "adv_median": float(np.median([row["adv"] for row in rows])),
        "adv_min": float(np.min([row["adv"] for row in rows])),
        "adv_max": float(np.max([row["adv"] for row in rows])),
    }


# ======================================================================================
# One (arm, seed, epoch) cell
# ======================================================================================
def analyze_cell(
    means_by_session: Mapping[str, Mapping[str, np.ndarray]],
    *,
    arm: str,
    seed: int,
    epoch: int,
    source_sessions: Sequence[str],
    val_sessions: Sequence[str],
    external_sessions: Sequence[str],
) -> dict[str, Any]:
    complete = {
        name for name, payload in means_by_session.items()
        if int(payload["condition_ids"].size) == cscs.N_CONDITIONS
    }
    primary_source = [s for s in source_sessions if s in complete]
    primary_val = [s for s in val_sessions if s in complete]
    primary_external = [s for s in external_sessions if s in complete]
    cscs.require(len(primary_source) == len(source_sessions), "a source session lost the complete grid")

    result: dict[str, Any] = {
        "arm": arm, "seed": seed, "epoch": epoch,
        "complete_grid_sessions": sorted(complete),
        "incomplete_grid_sessions": sorted(set(means_by_session) - complete),
        "by_pca_dim": {},
    }

    for pca_dim in PCA_DIMS:
        basis = cscs.fit_source_pca(
            {name: payload["full"] for name, payload in means_by_session.items()},
            n_components=pca_dim, source_sessions=primary_source,
        )
        projected: dict[tuple[str, str], np.ndarray] = {}
        ids: dict[str, np.ndarray] = {}
        for name, payload in means_by_session.items():
            ids[name] = payload["condition_ids"]
            for view in ("full", "half_1", "half_2"):
                projected[(name, view)] = basis.project(payload[view])

        null_kinds = [PRIMARY_NULL] + ([SECONDARY_NULL] if pca_dim == PRIMARY_PCA_DIM else [])
        per_dim: dict[str, Any] = {"pca_basis": basis.as_receipt(), "by_null": {}}

        for null_kind in null_kinds:
            engine = PairEngine(
                projected, ids, null_kind=null_kind,
                n_permutations=cscs.NULL_PERMUTATIONS, seed_prefix=[arm, seed, epoch, pca_dim],
            )
            source_pairs = [
                engine.symmetric(a, "full", b, "full")
                for i, a in enumerate(primary_source) for b in primary_source[i + 1 :]
            ]
            source_pairs_half = [
                engine.symmetric(a, "half_1", b, "half_1")
                for i, a in enumerate(primary_source) for b in primary_source[i + 1 :]
            ]
            ceiling = [engine.symmetric(name, "half_1", name, "half_2") for name in primary_source]
            external_pairs = {
                name: [engine.symmetric(name, "full", source, "full") for source in primary_source]
                for name in external_sessions
            }
            block: dict[str, Any] = {
                "source_source_full": _summarize(source_pairs),
                "source_source_half": _summarize(source_pairs_half),
                "within_session_split_half_ceiling": _summarize(ceiling),
                "per_source_session_mean_adv": {
                    name: float(np.mean([p["adv"] for p in source_pairs if name in (p["session_a"], p["session_b"])]))
                    for name in primary_source
                },
                "per_source_session_ceiling_adv": {
                    row["session_a"]: float(row["adv"]) for row in ceiling
                },
                "per_external_session_mean_adv": {
                    name: float(np.mean([p["adv"] for p in rows])) for name, rows in external_pairs.items()
                },
                "per_external_session_mean_raw": {
                    name: float(np.mean([p["raw"] for p in rows])) for name, rows in external_pairs.items()
                },
                "external_complete_grid_sessions": primary_external,
                "headroom_fraction": cscs.headroom_fraction(
                    _summarize(source_pairs_half)["adv"], _summarize(ceiling)["adv"]
                ),
            }
            if pca_dim == PRIMARY_PCA_DIM and null_kind == PRIMARY_NULL and primary_val:
                val_pairs = [
                    engine.symmetric(a, "full", b, "full")
                    for i, a in enumerate(primary_val) for b in primary_val[i + 1 :]
                ]
                val_to_source = [
                    engine.symmetric(v, "full", s, "full") for v in primary_val for s in primary_source
                ]
                block["val_val_full"] = _summarize(val_pairs)
                block["val_to_source_full"] = _summarize(val_to_source)
                block["val_ceiling"] = _summarize(
                    [engine.symmetric(name, "half_1", name, "half_2") for name in primary_val]
                )
            per_dim["by_null"][null_kind] = block
        result["by_pca_dim"][str(pca_dim)] = per_dim
    return result


# ======================================================================================
# External R2 binding
# ======================================================================================
def load_external_r2(arm: str, seed: int) -> tuple[dict[str, float], dict[str, Any]]:
    path = EXTERNAL_RECEIPT_ROOT / f"external_subject_M_{arm}_s{seed}.json"
    cscs.require(path.is_file(), f"sealed external receipt missing: {path}")
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    cscs.require(sidecar.is_file(), f"sealed external receipt sidecar missing: {sidecar}")
    cscs.require(digest == sidecar.read_text(encoding="utf-8").split()[0], "sealed external receipt digest drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cscs.require(payload.get("domain") == "external_subject_M", "external receipt domain drift")
    cscs.require(list(payload["protocol"]["epoch_window"]) == list(core.EPOCH_WINDOW), "external receipt epoch-window drift")
    return {str(k): float(v) for k, v in payload["per_session_mean_r2"].items()}, {
        "path": str(path), "sha256": digest, "mean_r2": float(payload["mean_r2"]),
    }


# ======================================================================================
# Driver
# ======================================================================================
def fit_source_normalizers(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], tuple, tuple]:
    """Reuse the sealed A2-v2 scorer's normalizer authority unchanged."""
    spec = importlib.util.spec_from_file_location(
        "a2_matched_subject_shift_v2_score", SUA_ROOT / "scripts" / "a2_matched_subject_shift_v2_score.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    authority, behavior_stats, side_stats, _test_names, _train_paths = module._fit_source_normalizers(metadata)
    return authority, behavior_stats, side_stats


def run(args: argparse.Namespace) -> int:
    assert_cpu_only()
    torch.set_num_threads(int(args.threads))
    started = time.time()
    CELL_ROOT.mkdir(parents=True, exist_ok=True)

    protocol_sha = sha256_file(PROTOCOL_PATH)
    implementation_sha = implementation_bindings()
    print(f"[cscs] protocol {PROTOCOL_PATH.name} sha256={protocol_sha}", flush=True)
    print(f"[cscs] implementation bindings: {len(implementation_sha)} files hashed", flush=True)

    bundles: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in SOURCE_ARMS:
        for seed in core.SEEDS:
            run_dir = core.CHECKPOINT_ROOT / core.source_run_name(arm, seed)
            fingerprint = core.source_run_receipt_fingerprint(run_dir, source_arm=arm, seed=seed)
            metadata = core.load_json_object(run_dir / "run_metadata.json")
            cscs.require(
                str((metadata.get("side_features") or {}).get("group")) == core.source_arm_for_name(arm)["side_feature_group"],
                f"{arm} s{seed}: checkpoint side-feature group does not match the declared arm",
            )
            cscs.require(bool(metadata.get("held_out_test_evaluated")) is False, "source run claims a formal test evaluation")
            bundles[(arm, seed)] = {
                "run_dir": run_dir, "fingerprint": fingerprint, "metadata": metadata,
                "checkpoints": core.source_epoch_checkpoint_paths(run_dir),
            }
    print(f"[cscs] bound {len(bundles)} sealed source bundles, epochs {list(core.EPOCH_WINDOW)}", flush=True)

    reference_metadata = bundles[("source_t4", 42)]["metadata"]
    normalizer_authority, behavior_stats, side_stats = fit_source_normalizers(reference_metadata)
    print(f"[cscs] source normalizers verified ({time.time() - started:.0f}s)", flush=True)

    train_paths, val_paths, formal_test_names = core.active_source_session_paths()
    external_paths = core.external_session_paths()
    cscs.require(len(formal_test_names) == 6, "formal-test name receipt drift")
    rosters = [("source_train", train_paths), ("source_val", val_paths), ("external_subject_M", external_paths)]

    sessions: dict[str, SessionData] = {}
    order: dict[str, list[str]] = {}
    for domain, paths in rosters:
        names = []
        for path in paths:
            data = load_session(path, domain, behavior_stats, side_stats, hash_nwb=not args.skip_nwb_hashes)
            sessions[data.name] = data
            names.append(data.name)
            print(f"[cscs] loaded {data.name:<26} units={data.n_units:<3} "
                  f"windows={data.plan['starts'].size:<5} complete={data.selection['complete_eight_direction_grid']}", flush=True)
        order[domain] = names
    print(f"[cscs] loaded {len(sessions)} sessions ({time.time() - started:.0f}s)", flush=True)

    source_sessions = order["source_train"]
    val_sessions = order["source_val"]
    external_sessions = order["external_subject_M"]

    parity_receipts: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    representation_shape: dict[str, int] = {}
    for (arm, seed), bundle in bundles.items():
        for epoch in core.EPOCH_WINDOW:
            cell_key = f"{arm}_s{seed}_e{epoch}"
            cell_path = CELL_ROOT / f"{cell_key}.json"
            if cell_path.is_file() and not args.recompute_cells:
                cells[cell_key] = json.loads(cell_path.read_text(encoding="utf-8"))
                print(f"[cscs] cell {cell_key}: reused", flush=True)
                continue
            checkpoint = bundle["checkpoints"][epoch]
            model = load_frozen_model(checkpoint, core.TEACHER_PATH, "B3S", torch.device("cpu"), identity_mode="calibrated")
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            means: dict[str, dict[str, np.ndarray]] = {}
            cell_started = time.time()
            for index, name in enumerate(sessions):
                session = sessions[name]
                do_parity = (epoch == core.EPOCH_WINDOW[0]) and index == 0
                values, parity = extract_yhat(model, session, arm, parity_check=do_parity)
                if parity["performed"]:
                    cscs.require(bool(parity["bitwise_identical"]), "identity-reuse parity failed")
                    parity_receipts.append({"arm": arm, "seed": seed, "epoch": epoch, "session": name, **parity})
                representation_shape.setdefault("flattened_dim", int(values.shape[1]))
                cscs.require(int(values.shape[1]) == representation_shape["flattened_dim"], "y_hat width drift")
                means[name] = cscs.condition_means(values, session.plan)
            cell = analyze_cell(
                means, arm=arm, seed=seed, epoch=epoch,
                source_sessions=source_sessions, val_sessions=val_sessions, external_sessions=external_sessions,
            )
            cell["checkpoint_path"] = str(checkpoint)
            cell["checkpoint_sha256"] = bundle["fingerprint"]["source_checkpoint_sha256_bundle"][str(epoch)]
            cell["seconds"] = float(time.time() - cell_started)
            cell_path.write_text(json.dumps(cell, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            cells[cell_key] = cell
            headroom = cell["by_pca_dim"][str(PRIMARY_PCA_DIM)]["by_null"][PRIMARY_NULL]["headroom_fraction"]
            summary = cell["by_pca_dim"][str(PRIMARY_PCA_DIM)]["by_null"][PRIMARY_NULL]["source_source_full"]
            print(f"[cscs] cell {cell_key}: adv={summary['adv']:.4f} raw={summary['raw']:.4f} "
                  f"null={summary['null']:.4f} headroom={headroom} ({cell['seconds']:.0f}s)", flush=True)
            del model, means

    aggregate = aggregate_cells(cells, sessions, order)
    payload = {
        "schema_version": cscs.SCHEMA_VERSION,
        "screen": cscs.SCREEN_NAME,
        "purpose": "cross_session_consistency_of_pooled_representation_yhat_screen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_document": {"path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)), "sha256": protocol_sha},
        "implementation_bindings": implementation_sha,
        "environment": environment_fingerprint(),
        "trained_anything": False,
        "backward_gradients": False,
        "checkpoint_writes": False,
        "formal_subc_test_nwb_opened": False,
        "formal_subc_test_session_names_only": list(formal_test_names),
        "resolved_config": {
            "epoch_window": list(core.EPOCH_WINDOW),
            "source_arms": list(SOURCE_ARMS),
            "seeds": list(core.SEEDS),
            "n_directions": cscs.N_DIRECTIONS,
            "phase_offsets": list(cscs.PHASE_OFFSETS),
            "n_conditions": cscs.N_CONDITIONS,
            "trials_per_direction": cscs.TRIALS_PER_DIRECTION,
            "min_trial_duration_bins": cscs.MIN_TRIAL_DURATION_BINS,
            "evaluation_start_trial_index": cscs.EVALUATION_START_TRIAL_INDEX,
            "window_size_bins": WINDOW_SIZE,
            "trial_length_bins": TRIAL_LENGTH,
            "bin_size_ms": core.BIN_SIZE_MS,
            "cv_folds": cscs.CV_FOLDS,
            "pca_dims": list(PCA_DIMS),
            "primary_pca_dim": PRIMARY_PCA_DIM,
            "null_kinds": [PRIMARY_NULL, SECONDARY_NULL],
            "primary_null": PRIMARY_NULL,
            "n_permutations": cscs.NULL_PERMUTATIONS,
            "bootstrap_resamples": cscs.BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": cscs.BOOTSTRAP_SEED,
            "kill_headroom_fraction_min": cscs.KILL_HEADROOM_FRACTION_MIN,
            "kill_spearman_min": cscs.KILL_SPEARMAN_MIN,
            "yhat_hook_module_path": YHAT_HOOK_MODULE_PATH,
            "yhat_flattened_dim": representation_shape.get("flattened_dim"),
            "forward_batch_windows": FORWARD_BATCH_WINDOWS,
        },
        "source_bundles": {
            f"{arm}_s{seed}": {
                "run_dir": str(bundle["run_dir"]),
                "run_metadata_sha256": bundle["fingerprint"]["source_run_metadata_sha256"],
                "checkpoint_sha256_bundle": bundle["fingerprint"]["source_checkpoint_sha256_bundle"],
                "checkpoint_sha256_bundle_sha256": bundle["fingerprint"]["source_checkpoint_sha256_bundle_sha256"],
            }
            for (arm, seed), bundle in bundles.items()
        },
        "normalizer_authority": normalizer_authority,
        "sessions": {name: data.receipt() for name, data in sessions.items()},
        "session_rosters": order,
        "identity_reuse_parity": parity_receipts,
        "per_cell": cells,
        **aggregate,
        "elapsed_seconds": float(time.time() - started),
    }
    body, sidecar, digest = write_immutable_json(Path(args.out_path), payload)
    print(f"[cscs] receipt {body} sha256={digest}")
    print(f"[cscs] sidecar {sidecar}")
    print(json.dumps(aggregate["verdict"], indent=2, sort_keys=True))
    return 0


def aggregate_cells(
    cells: Mapping[str, Any],
    sessions: Mapping[str, SessionData],
    order: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Pool the 48 (arm, seed, epoch) cells into the pre-declared §7 analyses and §8 verdict."""
    external_r2: dict[tuple[str, int], dict[str, float]] = {}
    external_receipts: dict[str, Any] = {}
    for arm in SOURCE_ARMS:
        for seed in core.SEEDS:
            values, receipt = load_external_r2(arm, seed)
            external_r2[(arm, seed)] = values
            external_receipts[f"{arm}_s{seed}"] = receipt

    def cell(arm: str, seed: int, epoch: int) -> Mapping[str, Any]:
        return cells[f"{arm}_s{seed}_e{epoch}"]

    def block(arm: str, seed: int, epoch: int, pca_dim: int, null_kind: str) -> Mapping[str, Any]:
        return cell(arm, seed, epoch)["by_pca_dim"][str(pca_dim)]["by_null"][null_kind]

    out: dict[str, Any] = {"external_r2_receipts": external_receipts}

    # A1/A2/A4 -- consistency levels and headroom, per arm, averaged over epochs.
    levels: dict[str, Any] = {}
    for pca_dim in PCA_DIMS:
        for null_kind in ([PRIMARY_NULL, SECONDARY_NULL] if pca_dim == PRIMARY_PCA_DIM else [PRIMARY_NULL]):
            for arm in SOURCE_ARMS:
                per_seed = {}
                for seed in core.SEEDS:
                    blocks = [block(arm, seed, epoch, pca_dim, null_kind) for epoch in core.EPOCH_WINDOW]
                    headrooms = [b["headroom_fraction"] for b in blocks if b["headroom_fraction"] is not None]
                    per_seed[str(seed)] = {
                        "source_source_full": {
                            key: float(np.mean([b["source_source_full"][key] for b in blocks]))
                            for key in ("raw", "null", "adv")
                        },
                        "source_source_half": {
                            key: float(np.mean([b["source_source_half"][key] for b in blocks]))
                            for key in ("raw", "null", "adv")
                        },
                        "ceiling": {
                            key: float(np.mean([b["within_session_split_half_ceiling"][key] for b in blocks]))
                            for key in ("raw", "null", "adv")
                        },
                        "headroom_fraction": float(np.mean(headrooms)) if headrooms else None,
                        "headroom_defined_epochs": len(headrooms),
                    }
                    if "val_val_full" in blocks[0]:
                        for key in ("val_val_full", "val_to_source_full", "val_ceiling"):
                            per_seed[str(seed)][key] = {
                                metric: float(np.mean([b[key][metric] for b in blocks]))
                                for metric in ("raw", "null", "adv")
                            }
                defined = [v["headroom_fraction"] for v in per_seed.values() if v["headroom_fraction"] is not None]
                levels[f"pca{pca_dim}_{null_kind}_{arm}"] = {
                    "per_seed": per_seed,
                    "mean_over_seeds": {
                        "source_source_full_adv": float(np.mean([v["source_source_full"]["adv"] for v in per_seed.values()])),
                        "source_source_full_raw": float(np.mean([v["source_source_full"]["raw"] for v in per_seed.values()])),
                        "source_source_full_null": float(np.mean([v["source_source_full"]["null"] for v in per_seed.values()])),
                        "source_source_half_adv": float(np.mean([v["source_source_half"]["adv"] for v in per_seed.values()])),
                        "ceiling_adv": float(np.mean([v["ceiling"]["adv"] for v in per_seed.values()])),
                        "ceiling_raw": float(np.mean([v["ceiling"]["raw"] for v in per_seed.values()])),
                        "headroom_fraction": float(np.mean(defined)) if defined else None,
                    },
                }
    out["consistency_levels"] = levels

    # A3 -- T4 vs Z4 paired by source session, sign counts and a bootstrap interval over sessions.
    source_sessions = order["source_train"]
    contrasts: dict[str, Any] = {}
    for seed in core.SEEDS:
        t4 = [float(np.mean([block("source_t4", seed, e, PRIMARY_PCA_DIM, PRIMARY_NULL)["per_source_session_mean_adv"][s]
                             for e in core.EPOCH_WINDOW])) for s in source_sessions]
        z4 = [float(np.mean([block("source_z4", seed, e, PRIMARY_PCA_DIM, PRIMARY_NULL)["per_source_session_mean_adv"][s]
                             for e in core.EPOCH_WINDOW])) for s in source_sessions]
        contrasts[str(seed)] = {
            "sessions": source_sessions,
            "t4_mean_adv": float(np.mean(t4)),
            "z4_mean_adv": float(np.mean(z4)),
            "paired": cscs.paired_contrast(t4, z4),
        }
    out["t4_vs_z4_source_session_contrast"] = contrasts

    # A5 -- does consistency predict external performance.
    association: dict[str, Any] = {}
    for pca_dim in PCA_DIMS:
        for arm in SOURCE_ARMS:
            complete = block(arm, core.SEEDS[0], core.EPOCH_WINDOW[0], pca_dim, PRIMARY_NULL)["external_complete_grid_sessions"]
            consistency_by_seed: dict[int, list[float]] = {}
            external_by_seed: dict[int, list[float]] = {}
            for seed in core.SEEDS:
                consistency_by_seed[seed] = [
                    float(np.mean([block(arm, seed, e, pca_dim, PRIMARY_NULL)["per_external_session_mean_adv"][name]
                                   for e in core.EPOCH_WINDOW]))
                    for name in complete
                ]
                external_by_seed[seed] = [external_r2[(arm, seed)][name] for name in complete]
            bootstrap = cscs.bootstrap_mean_spearman_over_seeds(consistency_by_seed, external_by_seed)
            entry: dict[str, Any] = {
                "sessions": complete,
                "consistency_adv_by_seed": {str(k): v for k, v in consistency_by_seed.items()},
                "external_r2_by_seed": {str(k): v for k, v in external_by_seed.items()},
                "bootstrap": bootstrap,
            }
            if pca_dim == PRIMARY_PCA_DIM:
                units = np.array([sessions[name].n_units for name in complete], dtype=np.float64)
                rate = np.array([sessions[name].mean_calibration_rate for name in complete], dtype=np.float64)
                entry["confound_controls"] = {
                    "partial_spearman_controlling_unit_count": {
                        str(seed): float(cscs.partial_spearman(
                            np.array(consistency_by_seed[seed]), np.array(external_by_seed[seed]), units))
                        for seed in core.SEEDS
                    },
                    "partial_spearman_controlling_mean_rate": {
                        str(seed): float(cscs.partial_spearman(
                            np.array(consistency_by_seed[seed]), np.array(external_by_seed[seed]), rate))
                        for seed in core.SEEDS
                    },
                    "spearman_unit_count_vs_external_r2": {
                        str(seed): float(cscs.spearman(units, np.array(external_by_seed[seed]))) for seed in core.SEEDS
                    },
                    "spearman_unit_count_vs_consistency": {
                        str(seed): float(cscs.spearman(units, np.array(consistency_by_seed[seed]))) for seed in core.SEEDS
                    },
                }
                excluded = [name for name in order["external_subject_M"] if name not in complete]
                entry["excluded_external_sessions"] = {
                    name: {
                        "external_r2_by_seed": {str(seed): external_r2[(arm, seed)][name] for seed in core.SEEDS},
                        "available_directions": [int(d) for d in sessions[name].selection["available_directions"]],
                        "consistency_adv_shared_grid_by_seed": {
                            str(seed): float(np.mean([
                                block(arm, seed, e, pca_dim, PRIMARY_NULL)["per_external_session_mean_adv"][name]
                                for e in core.EPOCH_WINDOW
                            ]))
                            for seed in core.SEEDS
                        },
                    }
                    for name in excluded
                }
                all_names = list(order["external_subject_M"])
                all_consistency = {
                    seed: [float(np.mean([block(arm, seed, e, pca_dim, PRIMARY_NULL)["per_external_session_mean_adv"][n]
                                          for e in core.EPOCH_WINDOW])) for n in all_names]
                    for seed in core.SEEDS
                }
                all_external = {seed: [external_r2[(arm, seed)][n] for n in all_names] for seed in core.SEEDS}
                entry["secondary_all_15_sessions_shared_direction_grid"] = {
                    "sessions": all_names,
                    "bootstrap": cscs.bootstrap_mean_spearman_over_seeds(all_consistency, all_external),
                }
            association[f"pca{pca_dim}_{arm}"] = entry
    out["consistency_vs_external_r2"] = association

    # A6 -- arm-level dose response, 6 points, explicitly underpowered.
    arm_points = []
    for arm in SOURCE_ARMS:
        for seed in core.SEEDS:
            arm_points.append({
                "arm": arm, "seed": seed,
                "source_consistency_adv": float(np.mean([
                    block(arm, seed, e, PRIMARY_PCA_DIM, PRIMARY_NULL)["source_source_full"]["adv"]
                    for e in core.EPOCH_WINDOW
                ])),
                "external_mean_r2": float(np.mean(list(external_r2[(arm, seed)].values()))),
            })
    out["arm_level_dose_response"] = {
        "points": arm_points,
        "spearman": float(cscs.spearman(
            np.array([p["source_consistency_adv"] for p in arm_points]),
            np.array([p["external_mean_r2"] for p in arm_points]),
        )),
        "n": len(arm_points),
        "power_note": "n=6 (2 arms x 3 seeds); reported, not leaned on.",
    }

    # §8 verdict -- T4 arm, primary PCA dim, primary null.
    t4_levels = levels[f"pca{PRIMARY_PCA_DIM}_{PRIMARY_NULL}_source_t4"]["mean_over_seeds"]
    t4_association = association[f"pca{PRIMARY_PCA_DIM}_source_t4"]["bootstrap"]
    out["verdict"] = {
        **cscs.evaluate_kill_criterion(
            headroom=t4_levels["headroom_fraction"],
            mean_spearman=t4_association["mean_spearman"],
            spearman_ci_lower=t4_association["ci_lower_95"],
        ),
        "basis": {
            "arm": "source_t4",
            "pca_dim": PRIMARY_PCA_DIM,
            "null_kind": PRIMARY_NULL,
            "source_source_full_adv": t4_levels["source_source_full_adv"],
            "source_source_half_adv": t4_levels["source_source_half_adv"],
            "ceiling_adv": t4_levels["ceiling_adv"],
            "spearman_ci_upper_95": t4_association["ci_upper_95"],
            "spearman_per_seed": t4_association["per_seed_spearman"],
        },
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-path", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-nwb-hashes", action="store_true", help="Skip NWB SHA-256 (debug only; receipt records it).")
    parser.add_argument("--recompute-cells", action="store_true")
    parser.add_argument("--launch", action="store_true", help="Execute; otherwise print an inert preflight.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.launch:
        assert_cpu_only()
        print(json.dumps({
            "mode": "preflight",
            "status": "NOT_LAUNCHED",
            "protocol_document": str(PROTOCOL_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "environment": environment_fingerprint(),
            "implementation_bindings": implementation_bindings(),
            "out_path": str(args.out_path),
        }, indent=2, sort_keys=True))
        return 0
    try:
        return run(args)
    except (cscs.CscsError, core.A2V2ContractError, FileNotFoundError, ValueError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
