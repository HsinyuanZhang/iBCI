#!/usr/bin/env python3
"""External sub-M follow-up controls: frozen F0, PV50, and Ridge50.

This is deliberately an *additive* scorer.  It never trains a network and it
does not rerun or write inside the completed V9 T4/zero4/TS4 result root.
Instead it reopens the frozen V9 cohort and target traces, then writes new
immutable artifacts in a separately named result root.

The three arms answer different questions and must not be conflated:

``f0_b3``
    Reuse the existing per-view, three-seed B3 F0 checkpoints.  It sees the
    normal first-30 activity calibration but no target-direction/behavior
    label is passed to its forward call.  It is a useful system-level F0
    comparator.  V9's ``shared_zero4`` remains the architecture-matched B3S
    no-label control, because the historical F0 checkpoints were independently
    trained B3 systems.

``pv50``
    A conventional baseline-subtracted, preferred-direction population vector
    fit from the first 50 trials, followed by a two-dimensional affine OLS gain
    fit on causal windows in those same 50 trials.

``ridge50``
    A conventional causal 50-bin flattened-spike-history ridge/Wiener decoder,
    fit only on those first-50 calibration windows.  The standardized ridge
    objective and lambda=1 are fixed before scoring; there is no validation or
    query-dependent tuning.

PV50 and Ridge50 use dense behavior samples from the 50 calibration trials;
T4 uses one target-direction label per trial.  Thus PV/Ridge are deliberately
stronger-label conventional performance comparators, *not* equal-information
carrier attribution controls.  The equal-width/equal-architecture mechanism
control remains V9 T4 versus V9 TS4.

Modes
-----
``preflight`` performs the CPU source-data audit only and writes nothing.
``forward`` seals prediction/target artifacts only; it deliberately contains
no R2 implementation.  ``finalize`` is a separate CPU-only operation and
requires the complete 15-session, 150-cell matrix plus TorchMetrics 1.5.1.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for _import_root in (ROOT, ROOT / "sua_exploration", ROOT / "sua_exploration/scripts"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as numerical


SCHEMA = "dandi_000688_subm_f0_pv_ridge_controls_v1"
V9_SCHEMA = "dandi_000688_subm_co_three_arm_v9_runtime_v1"
VIEWS = ("sua", "pseudo_mua")
F0_SEEDS = (42, 43, 44)
F0_ARM = "f0_b3"
PV_ARM = "pv50"
RIDGE_ARM = "ridge50"
ARMS = (F0_ARM, PV_ARM, RIDGE_ARM)
ACTIVITY_IDENTITY_TRIALS = 30
FIT_TRIALS = 50
HISTORY_BINS = numerical.HISTORY_BINS
OUTPUT_DIM = 2
EXPECTED_SESSIONS = 15
EXPECTED_V9_CELLS = EXPECTED_SESSIONS * len(VIEWS) * 3 * len(F0_SEEDS)
EXPECTED_NEW_CELLS = EXPECTED_SESSIONS * len(VIEWS) * (len(F0_SEEDS) + 2)


class ControlRunError(RuntimeError):
    """A frozen sub-M control invariant was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlRunError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly_regular(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444
    except OSError:
        return False


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlRunError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_exclusive(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Preserve a partial artifact as an incident.  Never replace an
        # uncertain output on a retry.
        raise
    os.chmod(path, 0o444)
    require(readonly_regular(path), f"immutable write failed: {path}")
    return sha256_bytes(raw)


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    return write_exclusive(path, canonical_bytes(dict(payload)))


def write_npz_exclusive(path: Path, prediction: np.ndarray, target: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, predictions=prediction, targets=target)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    os.chmod(path, 0o444)
    require(readonly_regular(path), f"immutable NPZ write failed: {path}")
    return sha256_file(path)


def safe_output_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ControlRunError(f"output path escapes result root: {relative}") from exc
    return candidate


@dataclass(frozen=True)
class CohortRow:
    asset_id: str
    session_id: str
    frozen_path: str
    nwb_sha256: str
    nwb_bytes: int
    query_window_count: int

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> "CohortRow":
        result = cls(
            asset_id=str(row["asset_id"]),
            session_id=str(row["session_id"]),
            frozen_path=str(row["frozen_path"]),
            nwb_sha256=str(row["nwb_sha256"]),
            nwb_bytes=int(row["nwb_bytes"]),
            query_window_count=int(row["query_window_count"]),
        )
        require(len(result.nwb_sha256) == 64 and result.nwb_bytes > 0 and result.query_window_count > 0, "invalid V9 cohort row")
        return result


@dataclass(frozen=True)
class F0Checkpoint:
    view: str
    seed: int
    path: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V9Inputs:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    cohort: tuple[CohortRow, ...]
    behavior_normalizers: Mapping[str, tuple[Path, str]]
    teacher: tuple[Path, str, int]


@dataclass(frozen=True)
class CellKey:
    asset_id: str
    session_id: str
    view: str
    arm: str
    seed: int | None

    def __post_init__(self) -> None:
        require(self.view in VIEWS, f"invalid view: {self.view}")
        require(self.arm in ARMS, f"invalid arm: {self.arm}")
        if self.arm == F0_ARM:
            require(self.seed in F0_SEEDS, "F0 requires seed 42,43,44")
        else:
            require(self.seed is None, f"{self.arm} is deterministic and has no seed")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def arm_seed_token(self) -> str:
        return f"seed_{self.seed}" if self.seed is not None else "deterministic"

    @property
    def artifact_relative_path(self) -> str:
        return f"artifacts/{self.asset_id}/{self.view}/{self.arm}/{self.arm_seed_token}/predictions_targets.npz"

    @property
    def commit_relative_path(self) -> str:
        return f"commits/{self.asset_id}/{self.view}/{self.arm}/{self.arm_seed_token}.json"


def _resolve_pinned_path(raw_path: str, *, repo_root: Path, label: str) -> Path:
    """Resolve a V9 absolute source path on either the local or remote clone.

    The completed V9 manifest pins its *original* absolute source location. A
    copied V9 result tree can legitimately live under a different clone root,
    so resolve the unambiguous ``sua_exploration/...`` suffix and subsequently
    check the original content SHA.  This is path relocation, not input
    substitution.
    """
    direct = Path(raw_path).expanduser()
    if direct.is_file() and not direct.is_symlink():
        return direct.resolve()
    parts = direct.parts
    try:
        index = parts.index("sua_exploration")
    except ValueError as exc:
        raise ControlRunError(f"cannot relocate V9 {label} path: {raw_path}") from exc
    candidate = (repo_root / Path(*parts[index:])).resolve()
    require(candidate.is_file() and not candidate.is_symlink(), f"missing relocated V9 {label}: {candidate}")
    return candidate


def load_v9_inputs(v9_root: Path, *, repo_root: Path) -> V9Inputs:
    root = v9_root.expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), f"missing V9 run manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == V9_SCHEMA, "not the completed V9 runtime manifest")
    require(manifest.get("expected_cell_count") == EXPECTED_V9_CELLS, "V9 expected-cell topology drift")
    contract = manifest.get("contract")
    require(isinstance(contract, Mapping), "V9 contract missing")
    chronology = contract.get("chronology")
    require(
        chronology == {
            "activity_identity_trials": ACTIVITY_IDENTITY_TRIALS,
            "t4_fit_pool_trials": FIT_TRIALS,
            "query_rule": "strictly_after_rewarded_trial_50",
            "selection": "chronological_first",
        },
        "V9 chronology contract drift",
    )
    raw_cohort = contract.get("cohort")
    require(isinstance(raw_cohort, list) and len(raw_cohort) == EXPECTED_SESSIONS, "V9 cohort must have 15 sessions")
    cohort = tuple(CohortRow.from_json(row) for row in raw_cohort)
    require(len({row.asset_id for row in cohort}) == EXPECTED_SESSIONS, "duplicate V9 cohort asset")
    require(sum(row.query_window_count for row in cohort) == 708_795, "V9 frozen query cardinality drift")
    normalizers = contract.get("normalizers")
    require(isinstance(normalizers, Mapping) and set(normalizers) == set(VIEWS), "V9 behavior normalizer topology drift")
    behavior_normalizers: dict[str, tuple[Path, str]] = {}
    for view in VIEWS:
        item = normalizers[view]
        require(isinstance(item, Mapping), f"V9 {view} normalizer row invalid")
        path = _resolve_pinned_path(str(item["behavior_path"]), repo_root=repo_root, label=f"{view} behavior normalizer")
        expected_sha = str(item["behavior_sha256"])
        require(sha256_file(path) == expected_sha, f"V9 {view} behavior normalizer SHA drift")
        behavior_normalizers[view] = (path, expected_sha)
    teacher = contract.get("teacher")
    require(isinstance(teacher, Mapping), "V9 teacher pin missing")
    teacher_path = _resolve_pinned_path(str(teacher["path"]), repo_root=repo_root, label="teacher")
    teacher_sha = str(teacher["sha256"])
    teacher_bytes = int(teacher["bytes"])
    require(teacher_path.stat().st_size == teacher_bytes and sha256_file(teacher_path) == teacher_sha, "V9 teacher pin drift")
    return V9Inputs(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
        cohort=cohort,
        behavior_normalizers=behavior_normalizers,
        teacher=(teacher_path, teacher_sha, teacher_bytes),
    )


def load_mean_std(path: Path, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            mean = np.ascontiguousarray(archive["mean"], dtype=np.float32)
            std = np.ascontiguousarray(archive["std"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise ControlRunError(f"invalid {label} normalizer: {path}") from exc
    require(mean.ndim == std.ndim == 1 and mean.shape == std.shape == (OUTPUT_DIM,), f"{label} normalizer shape drift")
    require(np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 0), f"{label} normalizer values invalid")
    return mean, std


def load_v9_target(v9: V9Inputs, row: CohortRow, view: str) -> tuple[np.ndarray, str, Path]:
    artifact = v9.root / "artifacts" / row.asset_id / view / "shared_t4" / "seed_42" / "predictions_targets.npz"
    require(artifact.is_file() and not artifact.is_symlink(), f"missing V9 target trace: {artifact}")
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            target = np.ascontiguousarray(archive["targets"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise ControlRunError(f"invalid V9 target trace: {artifact}") from exc
    require(target.shape == (row.query_window_count, OUTPUT_DIM), f"V9 target count drift: {row.session_id}/{view}")
    require(np.isfinite(target).all(), f"nonfinite V9 target: {row.session_id}/{view}")
    return target, sha256_bytes(target.tobytes(order="C")), artifact


def parse_f0_checkpoint(value: str) -> tuple[tuple[str, int], Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("--f0-checkpoint must be VIEW:SEED=PATH")
    raw_key, raw_path = value.split("=", 1)
    view, raw_seed = raw_key.split(":", 1)
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("F0 seed must be 42, 43, or 44") from exc
    if view not in VIEWS or seed not in F0_SEEDS or not raw_path:
        raise argparse.ArgumentTypeError("--f0-checkpoint requires sua|pseudo_mua:42|43|44=PATH")
    return (view, seed), Path(raw_path).expanduser().resolve()


def pin_f0_checkpoints(values: Sequence[str]) -> dict[tuple[str, int], F0Checkpoint]:
    parsed: dict[tuple[str, int], F0Checkpoint] = {}
    for text in values:
        key, path = parse_f0_checkpoint(text)
        require(key not in parsed, f"duplicate F0 checkpoint: {key}")
        require(path.is_file() and not path.is_symlink(), f"missing unsafe F0 checkpoint: {path}")
        parsed[key] = F0Checkpoint(
            view=key[0], seed=key[1], path=str(path), sha256=sha256_file(path), bytes=int(path.stat().st_size)
        )
    expected = {(view, seed) for view in VIEWS for seed in F0_SEEDS}
    require(set(parsed) == expected, "F0 requires exactly six checkpoints: each view x 42/43/44")
    return parsed


def validate_f0_checkpoint_architecture(checkpoint: F0Checkpoint) -> dict[str, Any]:
    """Cheap CPU-only guard that this is a genuine side_dim=0 B3 checkpoint."""
    import torch

    try:
        payload = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - torch gives heterogeneous load errors
        raise ControlRunError(f"cannot inspect F0 checkpoint {checkpoint.path}: {exc}") from exc
    require(isinstance(payload, Mapping), f"F0 checkpoint payload is not a mapping: {checkpoint.path}")
    hparams = payload.get("hyper_parameters") or {}
    require(isinstance(hparams, Mapping), f"F0 checkpoint hparams invalid: {checkpoint.path}")
    side_dim = int(hparams.get("side_dim", 0))
    require(side_dim == 0, f"F0 checkpoint side_dim must be 0, got {side_dim}: {checkpoint.path}")
    state = payload.get("state_dict")
    require(isinstance(state, Mapping) and state, f"F0 checkpoint state_dict invalid: {checkpoint.path}")
    return {
        "side_dim": side_dim,
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "checkpoint_sha256": checkpoint.sha256,
    }


def selected_cohort(cohort: Sequence[CohortRow], args: argparse.Namespace) -> tuple[CohortRow, ...]:
    selected = list(cohort)
    if args.asset_id:
        selected = [row for row in selected if row.asset_id == args.asset_id]
    if args.max_sessions is not None:
        selected = selected[: args.max_sessions]
    require(selected, "filters selected zero V9 sessions")
    return tuple(selected)


def selected_views(args: argparse.Namespace) -> tuple[str, ...]:
    return (args.view,) if args.view else VIEWS


def selected_arms(args: argparse.Namespace) -> tuple[str, ...]:
    raw = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    require(raw and set(raw) <= set(ARMS), f"--arms must be a nonempty subset of {','.join(ARMS)}")
    return tuple(arm for arm in ARMS if arm in raw)


def expected_keys(rows: Sequence[CohortRow], views: Sequence[str], arms: Sequence[str]) -> tuple[CellKey, ...]:
    return tuple(
        CellKey(row.asset_id, row.session_id, view, arm, seed if arm == F0_ARM else None)
        for row in rows
        for view in views
        for arm in arms
        for seed in (F0_SEEDS if arm == F0_ARM else (None,))
    )


def _runtime_owners(repo_root: Path) -> dict[str, Any]:
    from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as v9_runtime

    return v9_runtime._runtime_owners(repo_root)


def _build_view_base(
    *, repo_root: Path, nwb_path: Path, view: str, mean: np.ndarray, std: np.ndarray, owners: Mapping[str, Any]
) -> tuple[Any, np.ndarray, list[dict[str, Any]], Mapping[str, Any]]:
    from sua_exploration.mc_maze import subm_co_score_only_v3r2 as v3r2

    record, rebuilt, builder_trials, bridge = v3r2._build_view_base(
        nwb_path=nwb_path,
        view=view,
        normalizer={"behavior_mean": mean, "behavior_std": std},
        owners=owners,
    )
    require(record.signal_view == view, f"view loader drift: expected {view}, got {record.signal_view}")
    require(int(rebuilt.shape[0]) == ACTIVITY_IDENTITY_TRIALS, "F0 activity calibration is not chronological first 30")
    require(len(builder_trials) >= FIT_TRIALS + 1, "fewer than 51 usable rewarded trials")
    require(int(record.valid_starts.size) == int(bridge["query_window_count"]), "V3R2 query bridge count drift")
    return record, np.ascontiguousarray(rebuilt, dtype=np.float32), builder_trials, bridge


def _as_trial_bounds(trials: Sequence[Mapping[str, Any]], *, label: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    previous_stop = -1
    for index, trial in enumerate(trials):
        try:
            start, stop = int(trial["start"]), int(trial["stop"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlRunError(f"{label} trial {index} has unsafe bounds") from exc
        require(start >= 0 and stop - start >= HISTORY_BINS, f"{label} trial {index} is too short")
        require(start >= previous_stop, f"{label} trials are not chronological/nonoverlapping")
        result.append((start, stop))
        previous_stop = stop
    require(result, f"{label} is empty")
    return result


def validate_windows_in_trials(starts: np.ndarray, trials: Sequence[Mapping[str, Any]], *, label: str) -> None:
    """Prove every causal history stays within exactly one declared trial."""
    values = np.ascontiguousarray(starts, dtype=np.int64)
    require(values.ndim == 1 and values.size > 0, f"{label} starts missing")
    bounds = _as_trial_bounds(trials, label=label)
    cursor = 0
    for start in values:
        while cursor < len(bounds) and int(start) >= bounds[cursor][1]:
            cursor += 1
        require(cursor < len(bounds), f"{label} start occurs after the final trial")
        low, high = bounds[cursor]
        require(low <= int(start) and int(start) + HISTORY_BINS <= high, f"{label} history crosses an invalid trial boundary")


def calibration_starts_first50(builder_trials: Sequence[Mapping[str, Any]], record: Any) -> tuple[np.ndarray, Sequence[Mapping[str, Any]]]:
    from sua_exploration.mc_maze.multisession_datamodule import _compute_valid_starts

    support = builder_trials[:FIT_TRIALS]
    future = builder_trials[FIT_TRIALS:]
    starts = np.ascontiguousarray(_compute_valid_starts(list(support), HISTORY_BINS), dtype=np.int64)
    validate_windows_in_trials(starts, support, label="first50 calibration")
    validate_windows_in_trials(record.valid_starts, future, label="V9 query")
    require(int(starts.max()) + HISTORY_BINS <= int(support[-1]["stop"]), "first50 calibration reaches beyond trial 50")
    require(int(record.valid_starts.min()) >= int(future[0]["start"]), "V9 query begins before trial 51")
    require(not np.intersect1d(starts, record.valid_starts, assume_unique=True).size, "first50 calibration/query starts overlap")
    return starts, support


def cosine_trial_rates(record: Any, support_trials: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """T4-equivalent per-trial rates and direction rows, confined to first 50."""
    from sua_exploration.mc_maze.unit_side_features import _nearest_canonical_direction_index

    bounds = _as_trial_bounds(support_trials, label="PV first50")
    require(len(bounds) == FIT_TRIALS, "PV must consume exactly the chronological first 50 trials")
    duration = np.asarray([(stop - start) * numerical.BIN_SIZE_S for start, stop in bounds], dtype=np.float64)
    require(np.all(duration > 0), "nonpositive PV trial exposure")
    count = np.stack([record.neural[start:stop].sum(axis=0, dtype=np.float64) for start, stop in bounds], axis=1)
    rates = np.ascontiguousarray(count / duration[None, :], dtype=np.float64)
    raw_directions = [trial.get("target_dir") for trial in support_trials]
    require(all(value is not None and math.isfinite(float(value)) for value in raw_directions), "PV first50 has missing/nonfinite target direction")
    direction_indices = np.asarray([_nearest_canonical_direction_index(float(value)) for value in raw_directions], dtype=np.int64)
    present = sorted({int(value) for value in direction_indices})
    require(len(present) >= 2, "PV cosine fit is direction-degenerate in first50")
    return rates, direction_indices, present


def build_pv_readout(
    *, record: Any, support_trials: Sequence[Mapping[str, Any],], calibration_starts: np.ndarray
) -> tuple[numerical.PopulationVectorReadout, dict[str, Any]]:
    from sua_exploration.mc_maze.unit_side_features import _unit_tuning_features

    trial_rates, direction_indices, present = cosine_trial_rates(record, support_trials)
    n_channels = int(record.neural.shape[1])
    a = np.empty(n_channels, dtype=np.float64)
    c = np.empty(n_channels, dtype=np.float64)
    m = np.empty(n_channels, dtype=np.float64)
    b = np.empty(n_channels, dtype=np.float64)
    zero_spike = 0
    zero_modulation = 0
    for channel in range(n_channels):
        feature, _t8, is_zero_spike, is_zero_modulation = _unit_tuning_features(
            trial_rates[channel], direction_indices, present
        )
        a[channel], c[channel], m[channel], b[channel] = (float(value) for value in feature)
        zero_spike += int(is_zero_spike)
        zero_modulation += int(is_zero_modulation)
    preferred, zero_from_pd = numerical.preferred_directions_from_cosine(a, c, m)
    prefix = numerical.prefix_sums(record.neural)
    calibration_rates = numerical.window_rates_from_prefix(prefix, calibration_starts)
    calibration_vectors = numerical.population_vectors(calibration_rates, preferred, b)
    calibration_target = numerical.targets_at_window_end(record.behavior, calibration_starts)
    gain, intercept, rank = numerical.fit_population_vector_gain(calibration_vectors, calibration_target)
    readout = numerical.PopulationVectorReadout(
        preferred_direction=preferred,
        baseline_rate=np.ascontiguousarray(b, dtype=np.float64),
        gain=gain,
        intercept=intercept,
        calibration_rank=rank,
        zero_modulation_channels=zero_from_pd,
    )
    return readout, {
        "fit_trials": FIT_TRIALS,
        "target_direction_labels_used": FIT_TRIALS,
        "dense_behavior_windows_used": int(calibration_starts.size),
        "present_direction_count": len(present),
        "zero_spike_channels": zero_spike,
        "zero_modulation_channels": zero_modulation,
        "affine_gain_rank": rank,
        "channel_count": n_channels,
    }


def predict_pv_query(record: Any, readout: numerical.PopulationVectorReadout, *, batch_size: int) -> np.ndarray:
    prefix = numerical.prefix_sums(record.neural)
    parts: list[np.ndarray] = []
    for starts in numerical.batched(record.valid_starts, batch_size):
        rates = numerical.window_rates_from_prefix(prefix, starts)
        vector = numerical.population_vectors(rates, readout.preferred_direction, readout.baseline_rate)
        parts.append(numerical.predict_population_vector(vector, readout.gain, readout.intercept))
    prediction = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
    require(prediction.shape == (record.valid_starts.size, OUTPUT_DIM) and np.isfinite(prediction).all(), "PV query prediction drift")
    return prediction


def predict_ridge_query(record: Any, readout: numerical.RidgeReadout, *, batch_size: int, device: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for starts in numerical.batched(record.valid_starts, batch_size):
        features = numerical.raw_window_features(record.neural, starts)
        parts.append(numerical.predict_ridge(features, readout, device=device))
    prediction = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
    require(prediction.shape == (record.valid_starts.size, OUTPUT_DIM) and np.isfinite(prediction).all(), "ridge query prediction drift")
    return prediction


def collect_f0_predictions(*, model: Any, dataset: Any, device: str, owners: Mapping[str, Any], batch_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Exact frozen B3 scoring path with no side feature or label input."""
    torch = owners["torch"]
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested for F0 but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.synchronize(runtime_device)
    model.eval()
    for parameter in model.parameters():
        require(parameter.requires_grad is False, "F0 evaluator has a trainable parameter")
    loader = owners["DataLoader"](dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    t0 = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            require(len(batch) == 4, f"F0 dataset must be a no-side four-tuple, got arity={len(batch)}")
            neural, behavior, calibration, _names = batch
            neural = neural.to(runtime_device)
            behavior = behavior.to(runtime_device)
            calibration = calibration.to(runtime_device)
            decoder_key_features = model.decoder_key_features(None)
            raw_prediction, _ = model.student(
                neural,
                calib_trials=calibration,
                side_features=None,
                decoder_key_features=decoder_key_features,
                electrode_ids=None,
            )
            prediction = raw_prediction[:, -1:, :] / 5.0
            target = behavior[:, -1:, :]
            predictions.append(prediction[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False))
            targets.append(target[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False))
    if runtime_device.type == "cuda":
        torch.cuda.synchronize(runtime_device)
    prediction_np = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    target_np = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    require(prediction_np.shape == target_np.shape == (len(dataset), OUTPUT_DIM), "F0 forward output shape/count drift")
    require(np.isfinite(prediction_np).all() and np.isfinite(target_np).all(), "F0 forward contains nonfinite values")
    return prediction_np, target_np, {
        "forward_seconds": time.monotonic() - t0,
        "device": str(runtime_device),
        "batch_size": batch_size,
        "metric_computed": False,
        "backward_called": False,
        "target_direction_labels_passed_to_model": 0,
        "dense_behavior_labels_passed_to_model": 0,
    }


def build_f0_dataset(record: Any, rebuilt: np.ndarray, owners: Mapping[str, Any]) -> Any:
    dataset = owners["MCMazeSessionDataset"](
        neural_data=record.neural,
        behavior_data=record.behavior,
        valid_starts=record.valid_starts,
        calib_trials=rebuilt,
        window_size=HISTORY_BINS,
        session_name=record.name,
        side_features=None,
        electrode_ids=None,
    )
    require(len(dataset) == int(record.valid_starts.size), "F0 query count drift")
    return dataset


def output_manifest_payload(
    *, v9: V9Inputs, rows: Sequence[CohortRow], views: Sequence[str], arms: Sequence[str], f0: Mapping[tuple[str, int], F0Checkpoint], device: str,
) -> dict[str, Any]:
    import torch

    runtime = torch.device(device)
    device_info: dict[str, Any] = {
        "requested_device": str(runtime),
        "torch": str(torch.__version__),
        "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "tf32_disabled": True,
        "f0_deterministic_algorithms": True,
        "ridge_normalized_lambda": numerical.RIDGE_NORMALIZED_LAMBDA,
    }
    if runtime.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        device_info["cuda_device_name"] = str(torch.cuda.get_device_name(runtime))
    return {
        "schema": SCHEMA,
        "status": "PHASE_A_FORWARD_ONLY_NO_METRIC",
        "v9_source": {
            "root": str(v9.root),
            "run_manifest": str(v9.manifest_path),
            "run_manifest_sha256": v9.manifest_sha256,
            "contract_sha256": v9.manifest.get("contract_sha256"),
        },
        "cohort": [asdict(row) for row in rows],
        "topology": {
            "views": list(views),
            "arms": list(arms),
            "f0_seeds": list(F0_SEEDS) if F0_ARM in arms else [],
            "expected_new_cells": len(expected_keys(rows, views, arms)),
        },
        "chronology": {
            "activity_identity_trials_for_f0": ACTIVITY_IDENTITY_TRIALS,
            "calibration_fit_trials_for_pv_ridge": FIT_TRIALS,
            "calibration_selection": "chronological_first",
            "history_bins": HISTORY_BINS,
            "query_rule": "strictly_after_rewarded_trial_50",
            "query_target_trace": "byte-identical reuse of V9 shared_t4 seed_42 target trace",
        },
        "arms": {
            F0_ARM: {
                "kind": "historical_frozen_B3_forward_only_system_comparator",
                "training_relation": "independently trained per-view B3 checkpoint; not paired/shared B3S",
                "label_scope": "no target-direction or dense behavior label reaches target forward call",
                "checkpoints": [f0[(view, seed)].as_dict() for view in VIEWS for seed in F0_SEEDS] if F0_ARM in arms else [],
            },
            PV_ARM: {
                "kind": "classical_baseline_subtracted_PD_population_vector_plus_affine_OLS",
                "target_direction_label_scope": "first50 only; one target direction per calibration trial",
                "dense_behavior_label_scope": "causal last-bin targets from first50 windows only",
            },
            RIDGE_ARM: {
                "kind": "closed_form_causal_50bin_flattened_spike_history_ridge",
                "target_direction_label_scope": "none",
                "dense_behavior_label_scope": "causal last-bin targets from first50 windows only",
                "feature_standardization": "calibration-only first50 feature mean/std",
                "objective": "mean(||Y-XW-b||^2)+1.0||W||^2; b unpenalized",
            },
        },
        "execution_device": device_info,
        "phase_a_metric_policy": "FORBIDDEN_UNTIL_COMPLETE_FULL_15_SESSION_MATRIX",
        "overwrite_policy": "immutable_missing_only",
    }


def initialize_output(root: Path, payload: Mapping[str, Any]) -> None:
    root = root.resolve()
    manifest = root / "run_manifest.json"
    if manifest.exists():
        require(readonly_regular(manifest), "unsafe existing control run manifest")
        require(read_json(manifest) == dict(payload), "existing control run manifest drift")
        return
    if root.exists():
        # A typo should not attach the program to an unrelated result folder.
        require(not any(root.iterdir()), f"new result root is nonempty without a run manifest: {root}")
    write_json_exclusive(manifest, payload)


def _read_control_artifact(path: Path, *, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    require(readonly_regular(path), f"missing/unsafe control artifact: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            prediction = np.ascontiguousarray(archive["predictions"], dtype=np.float32)
            target = np.ascontiguousarray(archive["targets"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise ControlRunError(f"invalid control artifact: {path}") from exc
    require(prediction.shape == target.shape == (expected_rows, OUTPUT_DIM), "control artifact shape drift")
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), "control artifact nonfinite")
    return prediction, target


def validate_existing_cell(output_root: Path, row: CohortRow, key: CellKey, *, v9_target_sha: str) -> None:
    artifact = safe_output_path(output_root, key.artifact_relative_path)
    commit_path = safe_output_path(output_root, key.commit_relative_path)
    require(readonly_regular(artifact) and readonly_regular(commit_path), f"orphan/unsafe existing control cell: {key}")
    commit_raw = commit_path.read_bytes()
    commit = read_json(commit_path)
    require(canonical_bytes(commit) == commit_raw, "existing control commit is not canonical JSON")
    require(commit.get("schema") == SCHEMA and commit.get("cell") == key.as_dict() and commit.get("metric_computed") is False, "existing control cell identity drift")
    require(commit.get("query_window_count") == row.query_window_count, "existing control query count drift")
    require(commit.get("v9_target_trace_sha256") == v9_target_sha, "existing control V9 target pin drift")
    prediction, target = _read_control_artifact(artifact, expected_rows=row.query_window_count)
    require(sha256_file(artifact) == commit.get("artifact", {}).get("sha256"), "existing control artifact SHA drift")
    require(sha256_bytes(target.tobytes(order="C")) == v9_target_sha, "existing control target bytes drift")
    del prediction


def write_cell_artifact(
    *, output_root: Path, row: CohortRow, key: CellKey, prediction: np.ndarray, target: np.ndarray, v9_target_artifact: Path, evidence: Mapping[str, Any]
) -> None:
    prediction = np.ascontiguousarray(prediction, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)
    require(prediction.shape == target.shape == (row.query_window_count, OUTPUT_DIM), "control artifact output count drift")
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), "control artifact nonfinite")
    artifact_path = safe_output_path(output_root, key.artifact_relative_path)
    commit_path = safe_output_path(output_root, key.commit_relative_path)
    require(not artifact_path.exists() and not commit_path.exists(), f"existing/orphan output prevents replacement: {key}")
    artifact_sha = write_npz_exclusive(artifact_path, prediction, target)
    reopened_prediction, reopened_target = _read_control_artifact(artifact_path, expected_rows=row.query_window_count)
    require(reopened_prediction.tobytes(order="C") == prediction.tobytes(order="C"), "control artifact prediction reopen drift")
    require(reopened_target.tobytes(order="C") == target.tobytes(order="C"), "control artifact target reopen drift")
    target_sha = sha256_bytes(target.tobytes(order="C"))
    payload = {
        "schema": SCHEMA,
        "status": "ARTIFACT_COMMITTED_NO_METRIC",
        "cell": key.as_dict(),
        "query_window_count": row.query_window_count,
        "artifact": {
            "relative_path": key.artifact_relative_path,
            "sha256": artifact_sha,
            "bytes": artifact_path.stat().st_size,
            "dtype": "float32",
            "shape": [row.query_window_count, OUTPUT_DIM],
        },
        "v9_target_trace": {
            "path": str(v9_target_artifact),
            "sha256": target_sha,
            "reuse_policy": "byte-identical V9 shared_t4/seed_42 target trace",
        },
        "v9_target_trace_sha256": target_sha,
        "evidence": dict(evidence),
        "metric_computed": False,
    }
    write_json_exclusive(commit_path, payload)


def _base_trace(record: Any, rebuilt: np.ndarray, calibration_starts: np.ndarray) -> dict[str, Any]:
    return {
        "neural_sha256": sha256_bytes(np.ascontiguousarray(record.neural).tobytes(order="C")),
        "behavior_sha256": sha256_bytes(np.ascontiguousarray(record.behavior).tobytes(order="C")),
        "query_valid_starts_sha256": sha256_bytes(np.ascontiguousarray(record.valid_starts).tobytes(order="C")),
        "activity_first30_calibration_sha256": sha256_bytes(rebuilt.tobytes(order="C")),
        "classical_first50_window_starts_sha256": sha256_bytes(calibration_starts.tobytes(order="C")),
        "classical_first50_window_count": int(calibration_starts.size),
    }


def _nwb_path_and_pin(nwb_root: Path, row: CohortRow) -> Path:
    path = (nwb_root / row.frozen_path).resolve()
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe V9 NWB: {path}")
    require(path.stat().st_size == row.nwb_bytes, f"V9 NWB bytes drift: {row.session_id}")
    require(sha256_file(path) == row.nwb_sha256, f"V9 NWB SHA drift: {row.session_id}")
    return path


def audit_session_view(
    *, repo_root: Path, nwb_path: Path, row: CohortRow, view: str, mean: np.ndarray, std: np.ndarray, owners: Mapping[str, Any], v9_target: np.ndarray
) -> tuple[Any, np.ndarray, list[dict[str, Any]], np.ndarray, Mapping[str, Any], dict[str, Any]]:
    record, rebuilt, builder_trials, bridge = _build_view_base(
        repo_root=repo_root, nwb_path=nwb_path, view=view, mean=mean, std=std, owners=owners
    )
    require(record.name == row.session_id, f"runtime session identity drift: {record.name}")
    require(record.neural.ndim == 2 and record.behavior.shape == (record.neural.shape[0], OUTPUT_DIM), "record array shape drift")
    require(int(record.valid_starts.size) == row.query_window_count, f"V9 query-map drift: {row.session_id}/{view}")
    recomputed_target = numerical.targets_at_window_end(record.behavior, record.valid_starts)
    require(recomputed_target.tobytes(order="C") == v9_target.tobytes(order="C"), f"V9 target trace mismatch: {row.session_id}/{view}")
    calibration_starts, support_trials = calibration_starts_first50(builder_trials, record)
    evidence = {
        "bridge": dict(bridge),
        "base_trace": _base_trace(record, rebuilt, calibration_starts),
        "query_target_v9_byte_identical": True,
        "query_target_trace_sha256": sha256_bytes(v9_target.tobytes(order="C")),
        "signal_view": view,
        "channels": int(record.neural.shape[1]),
        "time_bins": int(record.neural.shape[0]),
        "first50_support_trial_count": len(support_trials),
        "strict_trial_boundary_checks": True,
    }
    return record, rebuilt, builder_trials, calibration_starts, support_trials, evidence


def forward(
    *, repo_root: Path, nwb_root: Path, output_root: Path, v9: V9Inputs, rows: Sequence[CohortRow], views: Sequence[str], arms: Sequence[str], f0: Mapping[tuple[str, int], F0Checkpoint], device: str, f0_batch_size: int, linear_batch_size: int,
) -> dict[str, Any]:
    payload = output_manifest_payload(v9=v9, rows=rows, views=views, arms=arms, f0=f0, device=device)
    initialize_output(output_root, payload)
    owners = _runtime_owners(repo_root)
    behavior_stats = {view: load_mean_std(v9.behavior_normalizers[view][0], label=f"{view} behavior") for view in VIEWS}
    # Loading and strict architecture reconstruction is deferred until the first
    # missing cell; resume therefore never wastes GPU time on completed cells.
    models: dict[tuple[str, int], Any] = {}
    completed = 0
    skipped = 0
    f0_architecture: dict[tuple[str, int], dict[str, Any]] = {}
    targets_by_asset: dict[str, bytes] = {}
    with owners["torch"].no_grad():
        for row in rows:
            nwb_path = _nwb_path_and_pin(nwb_root, row)
            for view in views:
                v9_target, v9_target_sha, v9_target_artifact = load_v9_target(v9, row, view)
                peer = targets_by_asset.setdefault(row.asset_id, v9_target.tobytes(order="C"))
                require(peer == v9_target.tobytes(order="C"), f"SUA/pseudo V9 target mismatch: {row.session_id}")
                keys_here = [key for key in expected_keys((row,), (view,), arms)]
                absent: list[CellKey] = []
                for key in keys_here:
                    artifact = safe_output_path(output_root, key.artifact_relative_path)
                    commit = safe_output_path(output_root, key.commit_relative_path)
                    if artifact.exists() or commit.exists():
                        require(artifact.exists() and commit.exists(), f"orphan existing cell: {key}")
                        validate_existing_cell(output_root, row, key, v9_target_sha=v9_target_sha)
                        skipped += 1
                    else:
                        absent.append(key)
                if not absent:
                    continue
                mean, std = behavior_stats[view]
                record, rebuilt, _trials, calibration_starts, support_trials, base_evidence = audit_session_view(
                    repo_root=repo_root, nwb_path=nwb_path, row=row, view=view, mean=mean, std=std, owners=owners, v9_target=v9_target
                )
                if PV_ARM in {key.arm for key in absent}:
                    pv, pv_evidence = build_pv_readout(record=record, support_trials=support_trials, calibration_starts=calibration_starts)
                    pv_prediction = predict_pv_query(record, pv, batch_size=linear_batch_size)
                    require(np.isfinite(pv_prediction).all(), "PV output nonfinite")
                    key = CellKey(row.asset_id, row.session_id, view, PV_ARM, None)
                    if key in absent:
                        write_cell_artifact(
                            output_root=output_root, row=row, key=key, prediction=pv_prediction, target=v9_target,
                            v9_target_artifact=v9_target_artifact,
                            evidence={
                                "base": base_evidence,
                                "method": pv_evidence,
                                "fit_policy": "first50 only; no query rows, labels, or metrics in fit",
                                "no_backpropagation": True,
                            },
                        )
                        completed += 1
                if RIDGE_ARM in {key.arm for key in absent}:
                    calibration_features = numerical.raw_window_features(record.neural, calibration_starts)
                    calibration_target = numerical.targets_at_window_end(record.behavior, calibration_starts)
                    ridge = numerical.fit_ridge(
                        calibration_features, calibration_target,
                        normalized_lambda=numerical.RIDGE_NORMALIZED_LAMBDA, device=device,
                    )
                    ridge_prediction = predict_ridge_query(record, ridge, batch_size=linear_batch_size, device=device)
                    key = CellKey(row.asset_id, row.session_id, view, RIDGE_ARM, None)
                    if key in absent:
                        write_cell_artifact(
                            output_root=output_root, row=row, key=key, prediction=ridge_prediction, target=v9_target,
                            v9_target_artifact=v9_target_artifact,
                            evidence={
                                "base": base_evidence,
                                "method": {
                                    "fit_trials": FIT_TRIALS,
                                    "target_direction_labels_used": 0,
                                    "dense_behavior_windows_used": int(calibration_starts.size),
                                    "raw_feature_shape": list(calibration_features.shape),
                                    "normalized_lambda": numerical.RIDGE_NORMALIZED_LAMBDA,
                                    "feature_standardization": "mean/std computed from first50 calibration rows only",
                                    "intercept": "target mean, unpenalized",
                                    "solver_device": ridge.solver_device,
                                },
                                "fit_policy": "first50 only; lambda fixed before query; no query rows, labels, or metrics in fit",
                                "no_backpropagation": True,
                            },
                        )
                        completed += 1
                    del calibration_features, calibration_target, ridge_prediction
                f0_missing = [key for key in absent if key.arm == F0_ARM]
                if f0_missing:
                    dataset = build_f0_dataset(record, rebuilt, owners)
                    for key in f0_missing:
                        spec = f0[(view, int(key.seed))]
                        require(Path(spec.path).stat().st_size == spec.bytes and sha256_file(Path(spec.path)) == spec.sha256, f"F0 checkpoint changed before load: {view}/{key.seed}")
                        model_key = (view, int(key.seed))
                        if model_key not in f0_architecture:
                            f0_architecture[model_key] = validate_f0_checkpoint_architecture(spec)
                        if model_key not in models:
                            models[model_key] = owners["model"].load_frozen_model(
                                Path(spec.path), v9.teacher[0], "B3", owners["torch"].device(device)
                            )
                        f0_prediction, f0_target, f0_evidence = collect_f0_predictions(
                            model=models[model_key], dataset=dataset, device=device, owners=owners, batch_size=f0_batch_size
                        )
                        require(f0_target.tobytes(order="C") == v9_target.tobytes(order="C"), f"F0 query target mismatch vs V9: {row.session_id}/{view}/{key.seed}")
                        write_cell_artifact(
                            output_root=output_root, row=row, key=key, prediction=f0_prediction, target=v9_target,
                            v9_target_artifact=v9_target_artifact,
                            evidence={
                                "base": base_evidence,
                                "method": {
                                    "checkpoint": spec.as_dict(),
                                    "architecture": f0_architecture[model_key],
                                    "activity_identity": "chronological first30 rebuilt calibration tensor",
                                    "side_features": "None",
                                    "target_direction_labels_used_by_forward": 0,
                                    "dense_behavior_labels_used_by_forward": 0,
                                },
                                "forward": f0_evidence,
                                "no_backpropagation": True,
                            },
                        )
                        completed += 1
                    del dataset
                del record, rebuilt
    return {"status": "PHASE_A_FORWARD_ARTIFACTS_COMMITTED_NO_METRIC", "completed_cells": completed, "skipped_existing_cells": skipped, "output_root": str(output_root)}


def preflight(
    *, repo_root: Path, nwb_root: Path, v9: V9Inputs, rows: Sequence[CohortRow], views: Sequence[str], arms: Sequence[str], f0: Mapping[tuple[str, int], F0Checkpoint]
) -> dict[str, Any]:
    """CPU audit.  It validates source bytes, chronology, targets, and designs."""
    owners = _runtime_owners(repo_root)
    behavior_stats = {view: load_mean_std(v9.behavior_normalizers[view][0], label=f"{view} behavior") for view in VIEWS}
    f0_architecture: dict[str, Any] = {}
    if F0_ARM in arms:
        for view in VIEWS:
            for seed in F0_SEEDS:
                f0_architecture[f"{view}:{seed}"] = validate_f0_checkpoint_architecture(f0[(view, seed)])
    session_rows: list[dict[str, Any]] = []
    for row in rows:
        nwb_path = _nwb_path_and_pin(nwb_root, row)
        v9_targets: dict[str, bytes] = {}
        for view in views:
            target, target_sha, _target_path = load_v9_target(v9, row, view)
            mean, std = behavior_stats[view]
            record, _rebuilt, _trials, calibration_starts, support_trials, evidence = audit_session_view(
                repo_root=repo_root, nwb_path=nwb_path, row=row, view=view, mean=mean, std=std, owners=owners, v9_target=target
            )
            if PV_ARM in arms:
                _pv, pv_info = build_pv_readout(record=record, support_trials=support_trials, calibration_starts=calibration_starts)
            else:
                pv_info = None
            v9_targets[view] = target.tobytes(order="C")
            session_rows.append({
                "asset_id": row.asset_id,
                "session_id": row.session_id,
                "view": view,
                "query_window_count": row.query_window_count,
                "v9_target_trace_sha256": target_sha,
                "first50_calibration_window_count": int(calibration_starts.size),
                "channel_count": int(record.neural.shape[1]),
                "pv_design": pv_info,
                "evidence": evidence,
            })
        if set(views) == set(VIEWS):
            require(v9_targets["sua"] == v9_targets["pseudo_mua"], f"SUA/pseudo target trace differs: {row.session_id}")
    return {
        "status": "CPU_PREFLIGHT_PASS_NO_OUTPUT_WRITTEN",
        "schema": SCHEMA,
        "selected_sessions": len(rows),
        "selected_views": list(views),
        "selected_arms": list(arms),
        "expected_new_cells": len(expected_keys(rows, views, arms)),
        "f0_architecture": f0_architecture,
        "sessions": session_rows,
    }


def _metric_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    import torchmetrics

    require(torchmetrics.__version__ == "1.5.1", f"finalize requires torchmetrics 1.5.1, got {torchmetrics.__version__}")
    from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu

    result = float(recompute_torchmetrics_r2_cpu(prediction, target))
    require(math.isfinite(result), "nonfinite TorchMetrics R2")
    return result


def _bootstrap_sessions(delta: np.ndarray, *, seed: int = 68_820_260_805, draws: int = 100_000) -> dict[str, float | int]:
    require(delta.ndim == 1 and delta.size == EXPECTED_SESSIONS, "bootstrap requires 15 session deltas")
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(draws, dtype=np.float64)
    chunk = 10_000
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, delta.size, size=(stop - start, delta.size))
        values[start:stop] = delta[indices].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
    return {"seed": seed, "draws": draws, "lower_95": float(lower), "upper_95": float(upper)}


def finalize(*, output_root: Path, v9: V9Inputs) -> dict[str, Any]:
    """CPU-only complete-matrix finalizer; intentionally unavailable for smoke roots."""
    root = output_root.resolve()
    manifest_path = root / "run_manifest.json"
    require(readonly_regular(manifest_path), "missing/unsafe control run manifest")
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == SCHEMA, "not a control run manifest")
    topology = manifest.get("topology")
    require(isinstance(topology, Mapping), "control topology missing")
    require(topology.get("views") == list(VIEWS) and topology.get("arms") == list(ARMS), "finalize requires all controls/views")
    cohort = tuple(CohortRow.from_json(row) for row in manifest.get("cohort", []))
    require(len(cohort) == EXPECTED_SESSIONS, "finalize requires full 15-session cohort, not a smoke subset")
    require(topology.get("expected_new_cells") == EXPECTED_NEW_CELLS, "finalize requires full 150-cell matrix")
    require(manifest.get("v9_source", {}).get("run_manifest_sha256") == v9.manifest_sha256, "V9 source manifest drift at finalization")
    final_path = root / "aggregate" / "endpoint_aggregate_torchmetrics151.json"
    require(not final_path.exists(), f"aggregate already exists: {final_path}")
    cells: list[dict[str, Any]] = []
    grids: dict[str, dict[str, Any]] = {
        F0_ARM: {view: np.empty((EXPECTED_SESSIONS, len(F0_SEEDS)), dtype=np.float64) for view in VIEWS},
        PV_ARM: {view: np.empty(EXPECTED_SESSIONS, dtype=np.float64) for view in VIEWS},
        RIDGE_ARM: {view: np.empty(EXPECTED_SESSIONS, dtype=np.float64) for view in VIEWS},
    }
    for session_index, row in enumerate(cohort):
        for view in VIEWS:
            v9_target, target_sha, _ = load_v9_target(v9, row, view)
            for arm in ARMS:
                seeds: Iterable[int | None] = F0_SEEDS if arm == F0_ARM else (None,)
                for seed_index, seed in enumerate(seeds):
                    key = CellKey(row.asset_id, row.session_id, view, arm, seed)
                    artifact = safe_output_path(root, key.artifact_relative_path)
                    commit = safe_output_path(root, key.commit_relative_path)
                    require(readonly_regular(commit), f"missing unsafe commit: {commit}")
                    payload = read_json(commit)
                    require(payload.get("v9_target_trace_sha256") == target_sha, "commit V9 target pin drift")
                    prediction, target = _read_control_artifact(artifact, expected_rows=row.query_window_count)
                    require(target.tobytes(order="C") == v9_target.tobytes(order="C"), "finalizer target byte drift")
                    require(sha256_file(artifact) == payload.get("artifact", {}).get("sha256"), "finalizer artifact SHA drift")
                    score = _metric_r2(prediction, target)
                    if arm == F0_ARM:
                        grids[arm][view][session_index, seed_index] = score
                    else:
                        grids[arm][view][session_index] = score
                    cells.append({"asset_id": row.asset_id, "session_id": row.session_id, "view": view, "arm": arm, "seed": seed, "query_window_count": row.query_window_count, "r2": score})
    require(len(cells) == EXPECTED_NEW_CELLS, "finalizer control cell count drift")
    v9_aggregate_path = v9.root / "aggregate" / "endpoint_aggregate_torchmetrics151.json"
    v9_aggregate = read_json(v9_aggregate_path)
    require(v9_aggregate.get("status") == "FULL_270_LOCAL_TORCHMETRICS_1_5_1_FINALIZED", "V9 authoritative aggregate unavailable")
    v9_rows = v9_aggregate.get("cells")
    require(isinstance(v9_rows, list) and len(v9_rows) == EXPECTED_V9_CELLS, "V9 authoritative cells drift")
    contrasts: dict[str, dict[str, Any]] = {}
    for view in VIEWS:
        t4 = np.empty((EXPECTED_SESSIONS, len(F0_SEEDS)), dtype=np.float64)
        z4 = np.empty_like(t4)
        for session_index, row in enumerate(cohort):
            for seed_index, seed in enumerate(F0_SEEDS):
                matching_t4 = [item for item in v9_rows if item["asset_id"] == row.asset_id and item["view"] == view and item["arm"] == "shared_t4" and item["seed"] == seed]
                matching_z4 = [item for item in v9_rows if item["asset_id"] == row.asset_id and item["view"] == view and item["arm"] == "shared_zero4" and item["seed"] == seed]
                require(len(matching_t4) == len(matching_z4) == 1, "V9 paired cell missing")
                t4[session_index, seed_index] = float(matching_t4[0]["r2"])
                z4[session_index, seed_index] = float(matching_z4[0]["r2"])
        t4_mean = t4.mean(axis=1)
        comparison = {
            "t4_minus_v9_shared_zero4_matched_B3S": t4.mean(axis=1) - z4.mean(axis=1),
            "t4_minus_historical_f0_b3_system_level": t4.mean(axis=1) - grids[F0_ARM][view].mean(axis=1),
            "t4_minus_pv50_stronger_dense_label_classical": t4_mean - grids[PV_ARM][view],
            "t4_minus_ridge50_stronger_dense_label_classical": t4_mean - grids[RIDGE_ARM][view],
        }
        contrasts[view] = {}
        for label, delta in comparison.items():
            contrasts[view][label] = {
                "mean_delta_r2": float(delta.mean(dtype=np.float64)),
                "positive_session_count": int(np.sum(delta > 0)),
                "per_session_delta_r2": {cohort[i].asset_id: float(value) for i, value in enumerate(delta)},
                "bootstrap_session_only": _bootstrap_sessions(delta),
            }
    payload = {
        "schema": "dandi_000688_subm_f0_pv_ridge_torchmetrics151_v1",
        "status": "FULL_150_LOCAL_TORCHMETRICS_1_5_1_FINALIZED",
        "source_control_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "source_v9_aggregate": {"path": str(v9_aggregate_path), "sha256": sha256_file(v9_aggregate_path)},
        "verified_cell_count": len(cells),
        "cells": cells,
        "mean_r2": {
            F0_ARM: {view: {str(seed): float(grids[F0_ARM][view][:, index].mean()) for index, seed in enumerate(F0_SEEDS)} for view in VIEWS},
            PV_ARM: {view: float(grids[PV_ARM][view].mean()) for view in VIEWS},
            RIDGE_ARM: {view: float(grids[RIDGE_ARM][view].mean()) for view in VIEWS},
        },
        "contrasts": contrasts,
        "claim_boundary": {
            "matched_no_label_control": "V9 shared_t4 minus V9 shared_zero4",
            "mechanism_control": "V9 shared_t4 minus V9 shared_ts4 (not recomputed here)",
            "historical_f0": "system-level comparator only: B3 independently trained per view, versus paired shared B3S V9 T4",
            "pv_ridge": "classical 50-trial direct-calibration comparators with richer dense behavior labels than T4",
        },
    }
    write_json_exclusive(final_path, payload)
    return {"status": payload["status"], "aggregate_path": str(final_path), "verified_cell_count": len(cells)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("preflight", "forward", "finalize"))
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--v9-run-root", type=Path, required=True)
    parser.add_argument("--nwb-root", type=Path, required=True, help="DANDI root containing sub-M/")
    parser.add_argument("--output-root", type=Path, help="new immutable control result root")
    parser.add_argument("--f0-checkpoint", action="append", default=[], metavar="VIEW:SEED=PATH")
    parser.add_argument("--device", default="cuda:0", help="F0 forward and closed-form ridge solver device")
    parser.add_argument("--f0-batch-size", type=int, default=128)
    parser.add_argument("--linear-batch-size", type=int, default=2048)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--asset-id")
    parser.add_argument("--max-sessions", type=int, help="development smoke only; use a separate output root")
    args = parser.parse_args()
    if args.mode in {"forward", "finalize"} and args.output_root is None:
        parser.error("--output-root is required for forward/finalize")
    if args.max_sessions is not None and args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    if args.f0_batch_size <= 0 or args.linear_batch_size <= 0:
        parser.error("batch sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    nwb_root = args.nwb_root.expanduser().resolve()
    require(repo_root == ROOT.resolve(), f"runner must execute from its own repository root: {ROOT}")
    require(nwb_root.is_dir(), f"missing NWB root: {nwb_root}")
    v9 = load_v9_inputs(args.v9_run_root, repo_root=repo_root)
    rows = selected_cohort(v9.cohort, args)
    views = selected_views(args)
    arms = selected_arms(args)
    f0 = pin_f0_checkpoints(args.f0_checkpoint) if F0_ARM in arms else {}
    if args.mode == "preflight":
        result = preflight(repo_root=repo_root, nwb_root=nwb_root, v9=v9, rows=rows, views=views, arms=arms, f0=f0)
    elif args.mode == "forward":
        result = forward(
            repo_root=repo_root, nwb_root=nwb_root, output_root=args.output_root.expanduser().resolve(),
            v9=v9, rows=rows, views=views, arms=arms, f0=f0, device=args.device,
            f0_batch_size=args.f0_batch_size, linear_batch_size=args.linear_batch_size,
        )
    else:
        # ``finalize`` deliberately ignores filters: it requires a complete
        # full-cohort output manifest written by a prior forward invocation.
        require(args.max_sessions is None and args.view is None and args.asset_id is None and set(arms) == set(ARMS), "finalize accepts only the complete unfiltered matrix")
        result = finalize(output_root=args.output_root.expanduser().resolve(), v9=v9)
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()

