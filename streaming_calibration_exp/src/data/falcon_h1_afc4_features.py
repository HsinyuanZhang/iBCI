"""Source-only H1 q=3 velocity carriers.

This module is intentionally independent from the RT/K4 implementation.  H1
uses the first chronological :class:`TrialNum` in each held-in calibration
record as its support (all ``eval_mask``-valid, non-still bins).  A fold plan
fits a raw-covariance q=3 PCA basis from the *inner-train* support bins only,
whitens the three scores by their source eigenvalues, and fits one closed-form
affine carrier per neural channel::

    [w1, w2, w3, b] = argmin_beta || [1, z1, z2, z3] beta - neural ||^2

The intercept is unpenalised and the three slopes use the frozen ridge
``lambda=1``.  The source descriptor normalizer is also fit only from the
inner-train descriptors.  Target support is never used to refit either
transform.  Five fixed-width arms are provided: ``afc4_h1q3`` (full),
``afc4_h1_b4`` (baseline only), ``zero4``, ``afc4_h1_rs`` (complete row
permutation), and ``afc4_h1_xs`` (cross-segment velocity-label derangement).

The functions accept in-memory records so CPU tests can exercise all leakage
guards without opening a second H1 file.  File loading is restricted to the
explicit DANDI 000954 held-in calibration/minival roots; held-out, query, and
EvalAI paths fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb
from pynwb import NWBHDF5IO


H1_VELOCITY_DIM = 7
H1_Q = 3
H1_AFC4_DIM = 4
H1_NUM_NEURONS = 176
H1_ACTIVE_THRESHOLD = 1.0e-3
H1_DESCRIPTOR_RIDGE = 1.0
H1_PROTOCOL_VERSION = "h1_source_only_rawcov_q3_afc4_v1"

H1_HELDIN_SESSION_NAMES: tuple[str, ...] = (
    "ses-19250101T111740",
    "ses-19250101T112404",
    "ses-19250108T110520",
    "ses-19250108T111022",
    "ses-19250108T111455",
    "ses-19250113T120811",
    "ses-19250113T121303",
    "ses-19250115T110633",
    "ses-19250115T111328",
    "ses-19250119T113543",
    "ses-19250119T114045",
    "ses-19250120T115044",
    "ses-19250120T115537",
)

H1_ARMS: tuple[str, ...] = (
    "afc4_h1q3",
    "afc4_h1_b4",
    "zero4",
    "afc4_h1_rs",
    "afc4_h1_xs",
)


def _finite_2d(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) <= 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty [rows, columns] matrix")
    return array


def _finite_1d(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size <= 0:
        raise ValueError(f"{name} must be a non-empty vector")
    return array


def _session_name(path: Path) -> str:
    marker = "_ses-"
    if marker not in path.name:
        raise ValueError(f"cannot derive H1 session from {path.name}")
    value = path.stem.split(marker, 1)[1]
    value = value.split("_behavior", 1)[0]
    return "ses-" + value


def _require_h1_path(path: Path, *, split: str) -> None:
    resolved = path.resolve()
    lower = str(resolved).lower()
    if "spint-main/data/000954/" not in lower:
        raise ValueError(f"H1 path leaves DANDI 000954 scope: {resolved}")
    if any(token in lower for token in ("held-out", "heldout", "evalai", "formal", "test")):
        raise ValueError(f"H1 path contains a forbidden held-out/formal token: {resolved}")
    required_dir = {
        "calib": "sub-humanpitt-held-in-calib",
        "minival": "sub-humanpitt-held-in-minival",
    }.get(str(split).lower())
    if required_dir is None or required_dir not in lower:
        raise ValueError(f"H1 path is not in the explicit held-in {split} scope: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)


def index_h1_heldin_pairs(data_dir: str | Path) -> dict[str, tuple[Path, Path]]:
    """Index exactly the 13 held-in calibration/minival pairs, no file reads."""

    root = Path(data_dir).resolve()
    calib_root = root / "sub-HumanPitt-held-in-calib"
    minival_root = root / "sub-HumanPitt-held-in-minival"
    calib = {_session_name(path): path.resolve() for path in sorted(calib_root.glob("*.nwb"))}
    minival = {_session_name(path): path.resolve() for path in sorted(minival_root.glob("*.nwb"))}
    expected = set(H1_HELDIN_SESSION_NAMES)
    if set(calib) != expected or set(minival) != expected:
        raise FileNotFoundError(
            "H1 clean LOSO requires exactly the 13 held-in calibration/minival pairs; "
            f"calib={sorted(calib)}, minival={sorted(minival)}"
        )
    for path in calib.values():
        _require_h1_path(path, split="calib")
    for path in minival.values():
        _require_h1_path(path, split="minival")
    return {name: (calib[name], minival[name]) for name in H1_HELDIN_SESSION_NAMES}


def _active_support_mask(
    covariates: np.ndarray,
    eval_mask: np.ndarray,
    trial_num: np.ndarray,
    *,
    threshold: float = H1_ACTIVE_THRESHOLD,
) -> np.ndarray:
    cov = _finite_2d(covariates, name="covariates")
    evaluation = np.asarray(eval_mask, dtype=bool).reshape(-1)
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    if cov.shape[1] != H1_VELOCITY_DIM or evaluation.shape != (cov.shape[0],) or labels.shape != evaluation.shape:
        raise ValueError("H1 covariates/eval_mask/TrialNum shapes disagree")
    finite = np.isfinite(labels)
    if not finite.any():
        raise ValueError("H1 TrialNum contains no finite values")
    # The chronological first TrialNum is the first finite, eval-valid value,
    # not the numerical minimum (pre-trial sentinels may be smaller).  The
    # finite label stream must be monotone and the first value may not reappear
    # after a later label; otherwise support boundaries are ambiguous.
    valid_indices = np.flatnonzero(evaluation & finite)
    if valid_indices.size == 0:
        raise ValueError("H1 TrialNum has no finite eval-valid bin")
    ordered = labels[valid_indices]
    if np.any(np.diff(ordered) < 0.0):
        raise ValueError("H1 TrialNum is not chronological/nondecreasing")
    first_trial = float(ordered[0])
    first_positions = np.flatnonzero(ordered == first_trial)
    if first_positions.size == 0 or first_positions[-1] != first_positions.size - 1:
        raise ValueError("H1 first TrialNum is not one contiguous chronological support trial")
    active = np.any(np.abs(cov) >= float(threshold), axis=1)
    support = (labels == first_trial) & evaluation & active
    if int(support.sum()) < H1_Q + 1:
        raise ValueError(
            f"H1 first-TrialNum support needs at least {H1_Q + 1} active bins, got {int(support.sum())}"
        )
    return support


def _contiguous_segments(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return global index arrays for each contiguous ``True`` block."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    segments: list[np.ndarray] = []
    start: int | None = None
    for index, active in enumerate(values):
        if active and start is None:
            start = index
        elif not active and start is not None:
            segments.append(np.arange(start, index, dtype=np.int64))
            start = None
    if start is not None:
        segments.append(np.arange(start, values.size, dtype=np.int64))
    return tuple(segments)


@dataclass(frozen=True)
class H1SessionRecord:
    """One H1 recording with only the arrays needed by this protocol."""

    session_name: str
    split: str
    path: Path | None
    neural: np.ndarray
    covariates: np.ndarray
    trial_change: np.ndarray
    eval_mask: np.ndarray
    trial_num: np.ndarray
    support_mask: np.ndarray
    support_segment_indices: tuple[np.ndarray, ...]

    @property
    def support_neural(self) -> np.ndarray:
        return np.asarray(self.neural[self.support_mask], dtype=np.float64)

    @property
    def support_velocity(self) -> np.ndarray:
        return np.asarray(self.covariates[self.support_mask], dtype=np.float64)

    @property
    def support_bins(self) -> int:
        return int(self.support_mask.sum())

    @property
    def support_segments(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Velocity/neural blocks for the first TrialNum active segments."""

        return tuple(
            (np.asarray(self.covariates[index], dtype=np.float64), np.asarray(self.neural[index], dtype=np.float64))
            for index in self.support_segment_indices
        )


def record_from_arrays(
    *,
    session_name: str,
    split: str,
    neural: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    trial_num: np.ndarray | None = None,
    path: Path | None = None,
) -> H1SessionRecord:
    """Build a validated record for CPU tests or a caller-owned loader."""

    spikes = _finite_2d(neural, name="neural")
    velocity = _finite_2d(covariates, name="covariates")
    if spikes.shape[0] != velocity.shape[0] or spikes.shape[1] != H1_NUM_NEURONS:
        raise ValueError(f"H1 neural/covariate shape mismatch or neuron count: {spikes.shape}/{velocity.shape}")
    if velocity.shape[1] != H1_VELOCITY_DIM:
        raise ValueError(f"H1 requires seven velocity channels, got {velocity.shape}")
    n_bins = spikes.shape[0]
    tc = np.zeros(n_bins, dtype=bool) if trial_change is None else np.asarray(trial_change, dtype=bool).reshape(-1)
    em = np.ones(n_bins, dtype=bool) if eval_mask is None else np.asarray(eval_mask, dtype=bool).reshape(-1)
    tn = np.ones(n_bins, dtype=np.float64) if trial_num is None else np.asarray(trial_num, dtype=np.float64).reshape(-1)
    if tc.shape != (n_bins,) or em.shape != (n_bins,) or tn.shape != (n_bins,):
        raise ValueError("H1 auxiliary arrays must have one value per bin")
    support = _active_support_mask(velocity, em, tn)
    segments = _contiguous_segments(support)
    if not segments:
        raise ValueError("H1 first-TrialNum support has no contiguous active segment")
    return H1SessionRecord(
        str(session_name), str(split), None if path is None else Path(path).resolve(),
        spikes, velocity, tc, em, tn, support, segments,
    )


def load_h1_record(path: str | Path, *, split: str) -> H1SessionRecord:
    """Load one explicit held-in H1 NWB and extract the first-TrialNum support."""

    path = Path(path)
    _require_h1_path(path, split=split)
    neural, covariates, trial_change, eval_mask = load_nwb(path, FalconTask.h1)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        if "TrialNum" not in nwb.acquisition:
            raise ValueError(f"{path}: H1 TrialNum acquisition is missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    return record_from_arrays(
        session_name=_session_name(path), split=split, path=path,
        neural=np.asarray(neural, dtype=np.float64),
        covariates=np.asarray(covariates, dtype=np.float64),
        trial_change=np.asarray(trial_change, dtype=bool),
        eval_mask=np.asarray(eval_mask, dtype=bool),
        trial_num=trial_num,
    )


@dataclass(frozen=True)
class H1VelocityQ3Basis:
    """Raw-covariance q=3 basis with source eigenvalue whitening."""

    mean: np.ndarray
    components: np.ndarray  # [3, 7]
    eigenvalues: np.ndarray  # [3]
    explained_variance_ratio: np.ndarray  # [3]
    sign_anchor_indices: np.ndarray  # [3]
    fit_sessions: tuple[str, ...]

    def project(self, velocity: np.ndarray) -> np.ndarray:
        values = _finite_2d(velocity, name="velocity")
        if values.shape[1] != H1_VELOCITY_DIM:
            raise ValueError("H1 velocity dimensionality differs from the source basis")
        raw = (values - self.mean) @ self.components.T
        return raw / np.sqrt(self.eigenvalues)[None, :]


def fit_source_velocity_basis(records: Mapping[str, H1SessionRecord]) -> H1VelocityQ3Basis:
    """Fit raw-covariance PCA from first-trial support bins of source sessions."""

    if not records:
        raise ValueError("H1 source basis needs at least one inner-train session")
    names = tuple(str(name) for name in records)
    if tuple(sorted(names)) != names or len(set(names)) != len(names):
        raise ValueError("H1 source basis session names must be unique and sorted")
    chunks = [record.support_velocity for record in records.values()]
    stacked = np.concatenate(chunks, axis=0)
    if stacked.shape[0] < H1_Q + 1:
        raise ValueError("H1 source basis has too few support bins")
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    covariance = centered.T @ centered / float(stacked.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.any(eigenvalues[:H1_Q] <= 1.0e-12) or not np.isfinite(eigenvalues).all():
        raise ValueError("H1 source velocity covariance is rank-deficient for q=3")
    components = eigenvectors[:, :H1_Q].T.copy()
    anchors = np.argmax(np.abs(components), axis=1).astype(np.int64)
    for row, anchor in enumerate(anchors):
        if components[row, anchor] < 0.0:
            components[row] *= -1.0
    ratio = eigenvalues[:H1_Q] / float(eigenvalues.sum())
    return H1VelocityQ3Basis(
        mean.astype(np.float64), components.astype(np.float64),
        eigenvalues[:H1_Q].astype(np.float64), ratio.astype(np.float64),
        anchors, names,
    )


def _deterministic_permutation(size: int, *, context: str, seed: int, kind: str) -> np.ndarray:
    if int(size) < 2:
        raise ValueError(f"{kind} permutation needs at least two rows/segments")
    payload = f"{H1_PROTOCOL_VERSION}:{kind}:{int(seed)}:{context}:{int(size)}".encode()
    digest = hashlib.sha256(payload).digest()
    # An explicit non-zero cyclic offset is a proof of derangement; random
    # permutations can retain fixed points even after a fallback roll.
    offset = 1 + (int.from_bytes(digest[:4], "little") % (int(size) - 1))
    order = np.roll(np.arange(int(size), dtype=np.int64), offset)
    if np.any(order == np.arange(int(size))):
        raise RuntimeError(f"{kind} permutation construction retained a fixed point")
    return order


def deterministic_row_permutation(num_neurons: int, *, session_name: str, seed: int) -> np.ndarray:
    """A complete deterministic row permutation for the H1 RS null."""

    return _deterministic_permutation(num_neurons, context=session_name, seed=seed, kind="row")


def deterministic_segment_permutation(num_segments: int, *, session_name: str, seed: int) -> np.ndarray:
    """A complete non-identity cross-segment label derangement."""

    return _deterministic_permutation(num_segments, context=session_name, seed=seed, kind="segment")


def _global_cross_segment_derangement(
    segment_ids: np.ndarray, *, session_name: str, seed: int
) -> np.ndarray:
    """Return a global source-index permutation with no same-segment pair.

    The returned indices are a bijection of all support bins, so XS preserves
    the exact velocity-label multiset.  A deterministic hash order of cyclic
    offsets is searched; each candidate is accepted only when every target bin
    receives a source bin from a different segment.  H1's first-trial support
    has multiple blocks, but callers still fail closed if no legal offset is
    available.
    """

    labels = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    if labels.size < 2 or len(np.unique(labels)) < 2:
        raise ValueError("H1 XS requires at least two active support segments")
    size = int(labels.size)
    digest = hashlib.sha256(f"{H1_PROTOCOL_VERSION}:xs-global:{int(seed)}:{session_name}:{size}".encode()).digest()
    start = 1 + (int.from_bytes(digest[:4], "little") % (size - 1))
    candidates = [1 + ((start - 1 + step) % (size - 1)) for step in range(size - 1)]
    target = np.arange(size, dtype=np.int64)
    for offset in candidates:
        source = (target + int(offset)) % size
        if np.all(labels[source] != labels[target]):
            if not np.array_equal(np.sort(source), target):
                raise RuntimeError("H1 XS source-index mapping is not bijective")
            return source
    raise ValueError("H1 XS has no legal global cross-segment derangement")


def _fit_affine(
    scores: np.ndarray,
    neural: np.ndarray,
    *,
    source: str,
    ridge: float = H1_DESCRIPTOR_RIDGE,
) -> tuple[np.ndarray, np.ndarray, float]:
    z = _finite_2d(scores, name=f"{source}.scores")
    y = _finite_2d(neural, name=f"{source}.neural")
    if z.shape[1] != H1_Q or z.shape[0] != y.shape[0]:
        raise ValueError(f"{source}: score/neural shape mismatch {z.shape}/{y.shape}")
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))
    rank = int(np.linalg.matrix_rank(design))
    if rank != H1_AFC4_DIM:
        raise ValueError(f"{source}: H1 AFC4 design rank is {rank}, not {H1_AFC4_DIM}")
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition):
        raise ValueError(f"{source}: H1 AFC4 design condition is non-finite")
    penalty = np.diag(np.asarray([0.0, ridge, ridge, ridge], dtype=np.float64))
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    # beta is [intercept, w1, w2, w3] x neurons; public descriptor is [w1,w2,w3,b].
    descriptor = np.column_stack((beta[1], beta[2], beta[3], beta[0]))
    if descriptor.shape != (y.shape[1], H1_AFC4_DIM) or not np.isfinite(descriptor).all():
        raise ValueError(f"{source}: H1 AFC4 descriptor is invalid: {descriptor.shape}")
    return descriptor.astype(np.float32), design, condition


def _support_segments(record: H1SessionRecord) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return every contiguous active block within the first TrialNum."""

    return record.support_segments


def fit_h1_descriptor(
    record: H1SessionRecord,
    basis: H1VelocityQ3Basis,
    *,
    cross_segment_label_shuffle: bool = False,
    shuffle_seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one per-channel H1 descriptor from first-trial support only."""

    segments = _support_segments(record)
    projected = [(basis.project(velocity), neural) for velocity, neural in segments]
    if cross_segment_label_shuffle:
        if len(projected) < 2:
            raise ValueError("H1 XS requires at least two active segments in the first TrialNum support")
        score_chunks = [value[0] for value in projected]
        neural_chunks = [value[1] for value in projected]
        scores = np.concatenate(score_chunks, axis=0)
        neural = np.concatenate(neural_chunks, axis=0)
        segment_ids = np.concatenate(
            [np.full(value.shape[0], index, dtype=np.int64) for index, value in enumerate(score_chunks)]
        )
        permutation = _global_cross_segment_derangement(
            segment_ids, session_name=record.session_name, seed=shuffle_seed
        )
        scores = scores[permutation]
        descriptor, design, condition = _fit_affine(scores, neural, source=f"{record.session_name}:xs")
        return descriptor, {
            "session_name": record.session_name,
            "support_bins": int(neural.shape[0]),
            "segment_count": len(projected),
            "design_rank": int(np.linalg.matrix_rank(design)),
            "design_condition": condition,
            "label_shuffle": True,
            "label_shuffle_policy": "global_nonidentity_cross_segment_bin_derangement",
            "source_index_permutation": permutation.tolist(),
            "fit_scope": "first_trialnum_active_eval_valid_bins",
        }
    scores = np.concatenate([value[0] for value in projected], axis=0)
    neural = np.concatenate([value[1] for value in projected], axis=0)
    descriptor, design, condition = _fit_affine(scores, neural, source=record.session_name)
    return descriptor, {
        "session_name": record.session_name,
        "support_bins": int(neural.shape[0]),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_condition": condition,
        "label_shuffle": False,
        "fit_scope": "first_trialnum_active_eval_valid_bins",
    }


def _later_trial_active_mask(record: H1SessionRecord) -> np.ndarray:
    """Return later-TrialNum, eval-valid, non-static bins for a diagnostic.

    This is intentionally separate from the support mask: the support remains
    the first chronological trial, while the diagnostic asks whether that
    frozen first-trial carrier extrapolates to later active bins.  The first
    label is selected chronologically (the same rule as ``_active_support_mask``)
    rather than by taking a numerical minimum.
    """

    evaluation = np.asarray(record.eval_mask, dtype=bool).reshape(-1)
    labels = np.asarray(record.trial_num, dtype=np.float64).reshape(-1)
    if evaluation.shape != labels.shape:
        raise ValueError(f"{record.session_name}: diagnostic TrialNum/eval_mask shape mismatch")
    valid = np.flatnonzero(evaluation & np.isfinite(labels))
    if valid.size == 0:
        raise ValueError(f"{record.session_name}: diagnostic has no finite eval-valid TrialNum bins")
    first_trial = float(labels[valid[0]])
    active = np.any(np.abs(np.asarray(record.covariates, dtype=np.float64)) >= H1_ACTIVE_THRESHOLD, axis=1)
    mask = evaluation & np.isfinite(labels) & (labels > first_trial) & active
    if int(mask.sum()) == 0:
        raise ValueError(f"{record.session_name}: diagnostic has no later active eval-valid bins")
    return mask


def evaluate_h1_later_trial_diagnostic(
    record: H1SessionRecord,
    basis: H1VelocityQ3Basis,
    *,
    raw_descriptor: np.ndarray | None = None,
    include_rs: bool = True,
    shuffle_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a first-trial carrier on later active calibration bins.

    The basis is expected to be a source-only fold basis.  ``raw_descriptor``
    is fitted from ``record``'s first-trial support when omitted; it is never
    refit on the later bins.  ``normalized_mse_gain`` is the dimensionless
    relative gain ``1 - SSE_full/SSE_b_only`` and ``flattened_correlation``
    is computed over all later-bin/channel entries.  The b-only reference uses
    the same fitted intercept and is therefore a strict matched baseline.

    The optional RS diagnostic applies the exact deterministic row permutation
    to the *raw* descriptor (before any source normalizer), preserving the
    direct-prediction interpretation of that null arm.
    """

    if record.split != "calib":
        raise ValueError("H1 later-trial diagnostic requires a calibration record")
    descriptor = (
        fit_h1_descriptor(record, basis)[0]
        if raw_descriptor is None
        else np.asarray(raw_descriptor, dtype=np.float64)
    )
    descriptor = _finite_2d(descriptor, name=f"{record.session_name}.raw_descriptor")
    if descriptor.shape != (H1_NUM_NEURONS, H1_AFC4_DIM):
        raise ValueError(
            f"{record.session_name}: raw descriptor must have shape {(H1_NUM_NEURONS, H1_AFC4_DIM)}, got {descriptor.shape}"
        )
    mask = _later_trial_active_mask(record)
    scores = basis.project(record.covariates[mask])
    target = np.asarray(record.neural[mask], dtype=np.float64)
    full = scores @ descriptor[:, :H1_Q].T + descriptor[:, H1_Q][None, :]
    baseline = np.broadcast_to(descriptor[:, H1_Q][None, :], target.shape)
    full_sse = float(np.square(full - target).sum())
    baseline_sse = float(np.square(baseline - target).sum())
    if not np.isfinite(full_sse) or not np.isfinite(baseline_sse) or baseline_sse <= 0.0:
        raise ValueError(f"{record.session_name}: later-trial diagnostic MSE denominator is invalid")

    def _flattened_corr(prediction: np.ndarray) -> float:
        flat_prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        flat_target = target.reshape(-1)
        if np.std(flat_prediction) <= 1.0e-12 or np.std(flat_target) <= 1.0e-12:
            return float("nan")
        return float(np.corrcoef(flat_prediction, flat_target)[0, 1])

    result: dict[str, Any] = {
        "session_name": record.session_name,
        "first_trial_num": float(record.trial_num[np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num))[0]]),
        "later_active_eval_valid_bins": int(mask.sum()),
        "fit_scope": "target_first_trialnum_support_only",
        "basis_scope": "source_inner_train_first_trialnum_support_only",
        "full_mse": float(full_sse / target.size),
        "b_only_mse": float(baseline_sse / target.size),
        "normalized_mse_gain": float(1.0 - full_sse / baseline_sse),
        "flattened_correlation_full": _flattened_corr(full),
        "flattened_correlation_b_only": _flattened_corr(baseline),
    }
    if include_rs:
        order = deterministic_row_permutation(
            H1_NUM_NEURONS, session_name=record.session_name, seed=int(shuffle_seed)
        )
        rs_descriptor = descriptor[order]
        rs_prediction = scores @ rs_descriptor[:, :H1_Q].T + rs_descriptor[:, H1_Q][None, :]
        rs_sse = float(np.square(rs_prediction - target).sum())
        result.update(
            {
                "rs4_normalized_mse_gain": float(1.0 - rs_sse / baseline_sse),
                "rs4_flattened_correlation": _flattened_corr(rs_prediction),
                "rs4_row_permutation_bijective": bool(np.array_equal(np.sort(order), np.arange(H1_NUM_NEURONS))),
            }
        )
    return result


def fit_h1_descriptor_from_segments(
    *,
    session_name: str,
    velocity_segments: Sequence[np.ndarray],
    neural_segments: Sequence[np.ndarray],
    basis: H1VelocityQ3Basis,
    shuffle_seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a full descriptor or strong XS using explicit support segments.

    XS applies one global segment-constrained bin derangement: each target
    neural bin is paired with a velocity bin from a different segment, with
    every source velocity bin used exactly once.  Thus the complete source
    velocity-label multiset is preserved and no interpolation or
    within-segment shuffle is accepted.
    """

    velocities = [_finite_2d(value, name=f"{session_name}.velocity_segment[{i}]") for i, value in enumerate(velocity_segments)]
    neurals = [_finite_2d(value, name=f"{session_name}.neural_segment[{i}]") for i, value in enumerate(neural_segments)]
    if len(velocities) != len(neurals) or len(velocities) < 1:
        raise ValueError("H1 descriptor segments must be non-empty and paired")
    if any(v.shape[1] != H1_VELOCITY_DIM for v in velocities):
        raise ValueError("H1 descriptor segments require seven velocity channels")
    if any(v.shape[0] != y.shape[0] or y.shape[1] != H1_NUM_NEURONS for v, y in zip(velocities, neurals)):
        raise ValueError("H1 descriptor segment velocity/neural shapes disagree")
    projected = [basis.project(value) for value in velocities]
    score_chunks = list(projected)
    neural_chunks = list(neurals)
    segment_ids = np.concatenate(
        [np.full(value.shape[0], index, dtype=np.int64) for index, value in enumerate(score_chunks)]
    )
    flat_scores = np.concatenate(score_chunks, axis=0)
    flat_neural = np.concatenate(neural_chunks, axis=0)
    permutation = _global_cross_segment_derangement(
        segment_ids, session_name=session_name, seed=shuffle_seed
    ) if len(velocities) >= 2 else None
    if permutation is not None:
        flat_scores = flat_scores[permutation]
    descriptor, design, condition = _fit_affine(
        flat_scores,
        flat_neural,
        source=f"{session_name}:xs" if len(velocities) >= 2 else f"{session_name}:full",
    )
    return descriptor, {
        "session_name": session_name,
        "support_bins": int(sum(value.shape[0] for value in neurals)),
        "segment_count": len(velocities),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_condition": condition,
        "label_shuffle": bool(len(velocities) >= 2),
        "label_shuffle_policy": (
            "global_nonidentity_cross_segment_bin_derangement"
            if len(velocities) >= 2 else "none"
        ),
        "source_index_permutation": None if permutation is None else permutation.tolist(),
    }


class H1AFC4SourcePlan:
    """One immutable source-only q=3/AFC4 plan for all H1 arms."""

    def __init__(self, source_records: Mapping[str, H1SessionRecord], *, shuffle_seed: int = 42) -> None:
        if not source_records:
            raise ValueError("H1 AFC4 source plan needs at least one inner-train record")
        names = tuple(str(name) for name in source_records)
        if tuple(sorted(names)) != names or len(set(names)) != len(names):
            raise ValueError("H1 AFC4 source records must be unique and sorted")
        if any(record.session_name != name for name, record in source_records.items()):
            raise ValueError("H1 AFC4 source record keys disagree with session names")
        if any(record.split != "calib" for record in source_records.values()):
            raise ValueError("H1 AFC4 source plan accepts held-in calibration records only")
        self.source_records = dict(source_records)
        self.source_session_names = names
        self.shuffle_seed = int(shuffle_seed)
        self.basis = fit_source_velocity_basis(self.source_records)
        raw: dict[str, np.ndarray] = {}
        audits: dict[str, dict[str, Any]] = {}
        for name, record in self.source_records.items():
            raw[name], audits[name] = fit_h1_descriptor(record, self.basis)
        stacked = np.concatenate(list(raw.values()), axis=0)
        self.normalizer_mean = stacked.mean(axis=0).astype(np.float32)
        self.normalizer_std = stacked.std(axis=0, ddof=0).astype(np.float32)
        self.normalizer_std[self.normalizer_std <= 1.0e-6] = 1.0
        self.raw_descriptors = raw
        self.source_audits = audits

    def _normalized(self, raw: np.ndarray) -> np.ndarray:
        values = (np.asarray(raw, dtype=np.float32) - self.normalizer_mean) / self.normalizer_std
        if not np.isfinite(values).all():
            raise ValueError("H1 AFC4 source-only normalizer produced non-finite values")
        return values

    def descriptor_for(self, record: H1SessionRecord, arm: str) -> tuple[np.ndarray, dict[str, Any]]:
        """Build one arm descriptor without changing the frozen plan."""

        requested = str(arm).lower()
        aliases = {
            "full": "afc4_h1q3",
            "afc4_h1q3": "afc4_h1q3",
            "b4": "afc4_h1_b4",
            "afc4_h1_b4": "afc4_h1_b4",
            "zero4": "zero4",
            "rs4": "afc4_h1_rs",
            "afc4_h1_rs": "afc4_h1_rs",
            "xs4": "afc4_h1_xs",
            "afc4_h1_xs": "afc4_h1_xs",
        }
        canonical = aliases.get(requested)
        if canonical is None:
            raise ValueError(f"Unsupported H1 AFC4 arm {arm!r}; choices={H1_ARMS}")
        if record.split != "calib":
            raise ValueError("H1 AFC4 descriptors require a held-in calibration support record")
        if canonical == "zero4":
            return np.zeros((H1_NUM_NEURONS, H1_AFC4_DIM), dtype=np.float32), {
                "arm": canonical,
                "session_name": record.session_name,
                "support_bins": record.support_bins,
                "fit_scope": "no_label_derived_side_values",
                "normalizer_used": False,
                "normalizer_fit_scope": "source_records_only",
                "normalizer_fit_sessions": list(self.source_session_names),
            }
        if canonical == "afc4_h1_xs":
            raw, audit = fit_h1_descriptor(
                record, self.basis, cross_segment_label_shuffle=True, shuffle_seed=self.shuffle_seed
            )
            values = self._normalized(raw)
            return values.astype(np.float32, copy=False), {
                **audit,
                "arm": canonical,
                "normalizer_fit_scope": "source_records_only",
                "normalizer_fit_sessions": list(self.source_session_names),
                "normalizer_mean": self.normalizer_mean.tolist(),
                "normalizer_std": self.normalizer_std.tolist(),
            }
        raw, audit = fit_h1_descriptor(record, self.basis)
        values = self._normalized(raw)
        if canonical == "afc4_h1_b4":
            values = values.copy()
            values[:, :H1_Q] = 0.0
        elif canonical == "afc4_h1_rs":
            order = deterministic_row_permutation(
                values.shape[0], session_name=record.session_name, seed=self.shuffle_seed
            )
            values = values[order]
            audit = {**audit, "row_permutation": order.tolist()}
        audit = {
            **audit,
            "arm": canonical,
            "normalizer_fit_scope": "source_records_only",
            "normalizer_fit_sessions": list(self.source_session_names),
            "normalizer_mean": self.normalizer_mean.tolist(),
            "normalizer_std": self.normalizer_std.tolist(),
        }
        return values.astype(np.float32, copy=False), audit

    def normalize_arm(self, raw_descriptor: np.ndarray, *, arm: str, session_name: str) -> tuple[np.ndarray, dict[str, Any]]:
        """Normalize an externally fitted raw descriptor (used by XS)."""

        raw = _finite_2d(raw_descriptor, name="raw_descriptor")
        if raw.shape != (H1_NUM_NEURONS, H1_AFC4_DIM):
            raise ValueError(f"H1 raw descriptor shape must be {(H1_NUM_NEURONS, H1_AFC4_DIM)}, got {raw.shape}")
        canonical = str(arm).lower()
        if canonical != "afc4_h1_xs":
            raise ValueError("normalize_arm is reserved for the H1 XS descriptor")
        values = self._normalized(raw)
        return values.astype(np.float32, copy=False), {
            "arm": canonical,
            "session_name": str(session_name),
            "normalizer_fit_scope": "source_records_only",
            "normalizer_fit_sessions": list(self.source_session_names),
            "label_shuffle": True,
        }

    def manifest(self, *, excluded_outer_target: str | None = None) -> dict[str, Any]:
        """Return JSON-safe fold provenance for a clean nested-LOSO receipt."""

        return {
            "schema": "h1_afc4_source_only_plan_v1",
            "protocol_version": H1_PROTOCOL_VERSION,
            "source_sessions": list(self.source_session_names),
            "excluded_outer_target_session": excluded_outer_target,
            "basis": {
                "fit_scope": "inner_train_first_trialnum_active_eval_valid_bins",
                "fit_sessions": list(self.source_session_names),
                "velocity_dim": H1_VELOCITY_DIM,
                "q": H1_Q,
                "mean": self.basis.mean.tolist(),
                "components": self.basis.components.tolist(),
                "eigenvalues": self.basis.eigenvalues.tolist(),
                "explained_variance_ratio": self.basis.explained_variance_ratio.tolist(),
                "sign_anchor_indices": self.basis.sign_anchor_indices.tolist(),
                "eigenvalue_whitening": True,
            },
            "descriptor": {
                "estimator": "closed_form_ridge_affine_per_channel",
                "ridge_lambda": H1_DESCRIPTOR_RIDGE,
                "intercept_penalized": False,
                "feature_order": ["w1", "w2", "w3", "b"],
                "fit_scope": "first_trialnum_active_eval_valid_bins",
                "lag_bins": 0,
            },
            "normalizer": {
                "fit_scope": "source_records_only",
                "fit_sessions": list(self.source_session_names),
                "mean": self.normalizer_mean.tolist(),
                "std": self.normalizer_std.tolist(),
            },
            "arms": list(H1_ARMS),
            "outer_target_used_for_basis_or_normalizer": False,
            "outer_target_used_for_checkpoint_selection": False,
        }
