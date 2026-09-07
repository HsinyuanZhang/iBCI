"""Unsupervised FA-alignment comparator skeleton (CPU only).

Part A audits constructibility without fitting an alignment.  Part B implements
the FALCON-style unsupervised-alignment-plus-linear-readout arms but is not
executed on real data from this module's runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze.source_pooled_ridge import (
    SessionChannelInfo,
    audit_view_correspondence,
    correspondence_audit_to_dict,
    detect_correspondence,
)
from sua_exploration.mc_maze.subm_v9_f0_pv_ridge import (
    FEATURE_STD_EPS,
    fit_ridge,
    predict_ridge,
)


PROTOCOL_DATE = "20260813"
SEED_MATERIAL = "fa-alignment-comparator-20260813"
SUBM_CALIBRATION_TRIALS = 50
RT_CALIBRATION_TRIALS = 24
HISTORY_BINS = 50
BIN_SIZE_MS = 20
BIN_SIZE_S = 0.020
OUTPUT_DIM = 2
VIEWS = ("sua", "pseudo_mua")
D_GRID = (4, 8, 16)
VE_THRESHOLD = 0.80
D_SPREAD_MAX = 4
MIN_SAMPLES_PER_PARAMETER = 2.0
FA_EM_MAX_ITER = 200
FA_EM_TOL = 1.0e-5
FA_UNIQUENESS_FLOOR = 1.0e-6
VE_CURVE_MAX_D = 96
ANISOTROPY_UNIQUE_ROTATION = 1.05
RT_EXPECTED_FOLDS = 15
SUBM_EXPECTED_SESSIONS = 15
NORMALIZED_LAMBDA_FIXED = 1.0
FORBIDDEN_PATH_TOKENS = ("heldout", "held-out", "minival", "formal", "private", "evalai", "test_ecephys")

# Sealed 15-session subject-M cohort used by every existing comparator arm.
SUBM_COHORT_SESSIONS: tuple[str, ...] = (
    "sub-M_ses-CO-20140307",
    "sub-M_ses-CO-20140626",
    "sub-M_ses-CO-20140627",
    "sub-M_ses-CO-20141203",
    "sub-M_ses-CO-20150511",
    "sub-M_ses-CO-20150512",
    "sub-M_ses-CO-20150610",
    "sub-M_ses-CO-20150611",
    "sub-M_ses-CO-20150612",
    "sub-M_ses-CO-20150615",
    "sub-M_ses-CO-20150616",
    "sub-M_ses-CO-20150617",
    "sub-M_ses-CO-20150623",
    "sub-M_ses-CO-20150625",
    "sub-M_ses-CO-20150626",
)

SEALED_REFERENCES = {
    "subject_m": {
        "carrier_sua": 0.3568,
        "carrier_pseudo_mua": 0.3061,
        "dense_ridge_sua": 0.4179,
        "dense_ridge_pseudo_mua": 0.4102,
    },
    "rt": {
        "t4d": 0.448176,
        "ridge": 0.200202,
        "zero4": 0.179272,
    },
}

ARMS = ("faa_a2_align", "faa_a1_anchored", "faa_no_align")
SCORING_ARMS_RT = ("faa_a2_align", "faa_no_align")
SELECTED_RT_D = 8
RT_SHARED_D_LIMITATION = (
    "A shared latent dimensionality is not defensible on RT: d80 ranges 15-45 "
    "(spread 30). The selected grid value is 8 because d=16 is infeasible on "
    "3 sessions. Forcing shared d=8 under-models some sessions and over-models others."
)


class FaAlignmentError(RuntimeError):
    """Raised when an FA-alignment contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FaAlignmentError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def seed_from_material(material: str = SEED_MATERIAL) -> int:
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def rng_from_material(material: str = SEED_MATERIAL) -> np.random.Generator:
    return np.random.default_rng(seed_from_material(material))


def path_is_forbidden(path: Path | str) -> bool:
    lowered = str(path).lower()
    return any(token in lowered for token in FORBIDDEN_PATH_TOKENS)


def assert_path_in_scope(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    require(not path_is_forbidden(resolved), f"refusing forbidden path: {resolved}")
    return resolved


def fa_parameter_count(n_channels: int, d: int) -> int:
    require(n_channels > 0 and d > 0, "FA parameter count requires positive n and d")
    return int(n_channels) * (int(d) + 2)


def samples_per_parameter(n_samples: int, n_channels: int, d: int) -> float:
    params = fa_parameter_count(n_channels, d)
    return float(n_samples) / float(params) if params else float("nan")


def r2_unweighted(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    require(pred.shape == target.shape and pred.ndim == 2 and pred.shape[0] >= 3, "invalid R2 arrays")
    residual = np.square(target - pred).sum(axis=0)
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum(axis=0)
    require(np.all(np.isfinite(total)) and np.all(total > 0.0), "per-dimension R2 denominator is not positive")
    scores = 1.0 - residual / total
    require(np.isfinite(scores).all(), "unweighted R2 is non-finite")
    return float(scores.mean())


def r2_variance_weighted(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    require(pred.shape == target.shape and pred.ndim == 2 and pred.shape[0] >= 3, "invalid R2 arrays")
    residual = np.square(target - pred).sum(axis=0)
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum(axis=0)
    denominator = float(total.sum())
    require(math.isfinite(denominator) and denominator > 0.0, "variance-weighted R2 denominator is not positive")
    score = 1.0 - float(residual.sum()) / denominator
    require(math.isfinite(score), "variance-weighted R2 is non-finite")
    return float(score)


def metric_for_dataset(dataset: str) -> str:
    if dataset == "rt":
        return "variance_weighted_r2"
    if dataset == "subject_m":
        return "variance_weighted_r2"
    raise FaAlignmentError(f"unknown dataset for metric: {dataset}")


def score_predictions(predictions: np.ndarray, targets: np.ndarray, *, dataset: str) -> float:
    name = metric_for_dataset(dataset)
    if name == "variance_weighted_r2":
        return r2_variance_weighted(predictions, targets)
    raise FaAlignmentError(f"unhandled metric: {name}")


@dataclass(frozen=True)
class FactorAnalysisFit:
    mean: np.ndarray
    loadings: np.ndarray
    uniquenesses: np.ndarray
    n_iter: int
    n_components: int

    def transform(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations, dtype=np.float64)
        require(x.ndim == 2 and x.shape[1] == self.mean.size, "FA transform channel mismatch")
        centered = x - self.mean
        psi_inv = 1.0 / self.uniquenesses
        wt_psi = self.loadings.T * psi_inv
        posterior_precision = np.eye(self.n_components, dtype=np.float64) + wt_psi @ self.loadings
        posterior_cov = np.linalg.inv(posterior_precision)
        return np.ascontiguousarray((posterior_cov @ wt_psi @ centered.T).T, dtype=np.float64)


@dataclass(frozen=True)
class AlignmentTransform:
    """Affine map ``z_aligned = (z - target_mean) @ matrix.T + source_mean``.

    ``faa_no_align`` stores the identity matrix and zero means so the map is
    exactly the identity on latents.
    """

    matrix: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    arm: str

    def apply(self, latents: np.ndarray) -> np.ndarray:
        z = np.asarray(latents, dtype=np.float64)
        require(z.ndim == 2 and z.shape[1] == self.matrix.shape[0], "alignment latent width mismatch")
        aligned = (z - self.target_mean) @ self.matrix.T + self.source_mean
        return np.ascontiguousarray(aligned, dtype=np.float64)

    def is_identity(self) -> bool:
        d = int(self.matrix.shape[0])
        return bool(
            np.array_equal(self.matrix, np.eye(d))
            and np.array_equal(self.source_mean, np.zeros(d))
            and np.array_equal(self.target_mean, np.zeros(d))
        )


@dataclass(frozen=True)
class SessionAuditRow:
    session_name: str
    view: str
    n_channels: int
    n_units: int
    n_samples: int
    channel_ids: tuple[int, ...]
    electrode_ids_per_unit: tuple[int, ...] | None
    identifiers_are_positional_only: bool
    channel_ids_sha256: str
    variance_explained: tuple[float, ...]
    cumulative_variance_explained: tuple[float, ...]
    d80: int | None
    covariance_rank: int
    top_d_anisotropy: dict[str, float]
    samples_per_parameter_by_d: dict[str, float]
    fa_feasible_by_d: dict[str, bool]


def identity_alignment(d: int, *, arm: str = "faa_no_align") -> AlignmentTransform:
    require(d > 0, "identity alignment requires positive d")
    zeros = np.zeros(int(d), dtype=np.float64)
    return AlignmentTransform(
        matrix=np.eye(int(d), dtype=np.float64),
        source_mean=zeros,
        target_mean=zeros.copy(),
        arm=arm,
    )


def _signed_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    """Deterministic sign convention: largest-magnitude entry of each column is positive."""
    signed = np.array(eigenvectors, dtype=np.float64, copy=True)
    for column in range(signed.shape[1]):
        pivot = int(np.argmax(np.abs(signed[:, column])))
        if signed[pivot, column] < 0.0:
            signed[:, column] *= -1.0
    return signed


def pca_variance_explained(observations: np.ndarray, *, max_d: int = VE_CURVE_MAX_D) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] >= 2 and x.shape[1] >= 2, "PCA variance explained needs a 2-D block")
    centered = x - x.mean(axis=0, keepdims=True)
    n_samples = int(centered.shape[0])
    gram = centered.T @ centered / max(n_samples - 1, 1)
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
    total = float(eigenvalues.sum())
    require(math.isfinite(total) and total > 0.0, "calibration covariance has no positive variance")
    ratios = eigenvalues / total
    limit = min(int(max_d), int(ratios.size))
    ve = np.ascontiguousarray(ratios[:limit], dtype=np.float64)
    return ve, np.cumsum(ve)


def covariance_rank(observations: np.ndarray, *, rtol: float = 1.0e-8) -> int:
    x = np.asarray(observations, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.size == 0:
        return 0
    return int(np.sum(singular > rtol * float(singular[0])))


def top_d_anisotropy(observations: np.ndarray, d: int) -> dict[str, float]:
    x = np.asarray(observations, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    n_samples = int(centered.shape[0])
    gram = centered.T @ centered / max(n_samples - 1, 1)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(gram), 0.0))[::-1]
    take = min(int(d), int(eigenvalues.size))
    head = eigenvalues[:take]
    positive = head[head > 0.0]
    if positive.size < 2:
        ratio = float("nan")
        unique = False
    else:
        ratio = float(positive[0] / positive[-1])
        unique = bool(ratio >= ANISOTROPY_UNIQUE_ROTATION)
    return {
        "d": float(take),
        "lambda_max": float(head[0]) if head.size else float("nan"),
        "lambda_min_in_top_d": float(head[-1]) if head.size else float("nan"),
        "anisotropy": ratio,
        "rotation_uniquely_identifiable": unique,
    }


def d80_from_cumulative(cumulative: np.ndarray, *, threshold: float = VE_THRESHOLD) -> int | None:
    hits = np.flatnonzero(np.asarray(cumulative, dtype=np.float64) >= float(threshold))
    if hits.size == 0:
        return None
    return int(hits[0] + 1)


def fit_factor_analysis(
    observations: np.ndarray,
    d: int,
    *,
    max_iter: int = FA_EM_MAX_ITER,
    tol: float = FA_EM_TOL,
) -> FactorAnalysisFit:
    """EM factor analysis with SVD initialisation.  Deterministic; no RNG."""
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 2 and np.isfinite(x).all(), "FA observations must be finite 2-D")
    n_samples, n_channels = int(x.shape[0]), int(x.shape[1])
    require(d >= 1 and d < n_channels, "FA d must satisfy 1 <= d < n_channels")
    require(n_samples > d, "FA requires more samples than factors")
    mean = x.mean(axis=0)
    centered = x - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    scale = np.sqrt(max(n_samples, 1))
    loadings = (vt[:d].T * singular[:d] / scale).astype(np.float64, copy=False)
    variance = np.mean(centered * centered, axis=0)
    uniquenesses = np.maximum(variance - np.sum(loadings * loadings, axis=1), FA_UNIQUENESS_FLOOR)
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        psi_inv = 1.0 / uniquenesses
        wt_psi = loadings.T * psi_inv
        posterior_precision = np.eye(d, dtype=np.float64) + wt_psi @ loadings
        posterior_cov = np.linalg.inv(posterior_precision)
        ez = (posterior_cov @ wt_psi @ centered.T).T
        ezz = posterior_cov + (ez.T @ ez) / n_samples
        loadings_new = (centered.T @ ez / n_samples) @ np.linalg.inv(ezz)
        uniquenesses = np.maximum(variance - np.sum(loadings_new * (loadings_new @ ezz), axis=1), FA_UNIQUENESS_FLOOR)
        delta = float(np.linalg.norm(loadings_new - loadings) / max(np.linalg.norm(loadings), 1.0e-12))
        loadings = loadings_new
        if delta < float(tol):
            break
    return FactorAnalysisFit(
        mean=np.ascontiguousarray(mean, dtype=np.float64),
        loadings=np.ascontiguousarray(loadings, dtype=np.float64),
        uniquenesses=np.ascontiguousarray(uniquenesses, dtype=np.float64),
        n_iter=int(n_iter),
        n_components=int(d),
    )


def estimate_latent_alignment(
    source_latents: np.ndarray,
    target_latents: np.ndarray,
    *,
    target_labels: np.ndarray | None = None,
    arm: str = "faa_a2_align",
) -> AlignmentTransform:
    """Match latent second-order statistics with a signed eigenbasis map.

    Consumes **zero** target labels.  Any non-None ``target_labels`` fails closed.
    """
    if target_labels is not None:
        raise FaAlignmentError(
            "alignment estimation must consume zero target labels; "
            f"got target_labels with shape {np.asarray(target_labels).shape}"
        )
    source = np.asarray(source_latents, dtype=np.float64)
    target = np.asarray(target_latents, dtype=np.float64)
    require(source.ndim == 2 and target.ndim == 2, "alignment latents must be 2-D")
    require(source.shape[1] == target.shape[1], "source/target latent dimensionality mismatch")
    require(source.shape[0] >= 3 and target.shape[0] >= 3, "alignment needs at least three latent rows")
    d = int(source.shape[1])
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_c = source - source_mean
    target_c = target - target_mean
    source_cov = (source_c.T @ source_c) / max(source.shape[0] - 1, 1)
    target_cov = (target_c.T @ target_c) / max(target.shape[0] - 1, 1)
    source_eval, source_evec = np.linalg.eigh(source_cov)
    target_eval, target_evec = np.linalg.eigh(target_cov)
    source_order = np.argsort(source_eval)[::-1]
    target_order = np.argsort(target_eval)[::-1]
    source_eval = np.maximum(source_eval[source_order], 0.0)
    target_eval = np.maximum(target_eval[target_order], 0.0)
    source_evec = _signed_eigenvectors(source_evec[:, source_order])
    target_evec = _signed_eigenvectors(target_evec[:, target_order])
    scale = np.sqrt(source_eval / np.maximum(target_eval, 1.0e-12))
    matrix = source_evec * scale @ target_evec.T
    require(np.isfinite(matrix).all(), "alignment matrix is non-finite")
    return AlignmentTransform(
        matrix=np.ascontiguousarray(matrix, dtype=np.float64),
        source_mean=np.ascontiguousarray(source_mean, dtype=np.float64),
        target_mean=np.ascontiguousarray(target_mean, dtype=np.float64),
        arm=arm,
    )


def estimate_channel_anchored_alignment(
    source_loadings: np.ndarray,
    target_loadings: np.ndarray,
    *,
    target_labels: np.ndarray | None = None,
    arm: str = "faa_a1_anchored",
) -> AlignmentTransform:
    """Orthogonal Procrustes on FA loadings of correspondent channels."""
    if target_labels is not None:
        raise FaAlignmentError(
            "alignment estimation must consume zero target labels; "
            f"got target_labels with shape {np.asarray(target_labels).shape}"
        )
    source = np.asarray(source_loadings, dtype=np.float64)
    target = np.asarray(target_loadings, dtype=np.float64)
    require(source.ndim == 2 and source.shape == target.shape, "anchored alignment loading shape mismatch")
    cross = target.T @ source
    u, _s, vt = np.linalg.svd(cross, full_matrices=False)
    matrix = vt.T @ u.T
    d = int(source.shape[1])
    zeros = np.zeros(d, dtype=np.float64)
    return AlignmentTransform(matrix=matrix, source_mean=zeros, target_mean=zeros.copy(), arm=arm)


def require_shared_d(per_session_d: Mapping[str, int]) -> int:
    values = {int(value) for value in per_session_d.values()}
    require(len(per_session_d) > 0, "shared d requires at least one session")
    if len(values) != 1:
        raise FaAlignmentError(
            "shared latent dimensionality d mismatch across sessions; "
            f"observed {dict(per_session_d)}"
        )
    return next(iter(values))


def resolve_shared_d_from_feasibility(
    per_session_feasible: Mapping[str, Sequence[int]],
    *,
    grid: Sequence[int] = D_GRID,
) -> int:
    require(per_session_feasible, "shared-d resolution requires sessions")
    common: set[int] | None = None
    for feasible in per_session_feasible.values():
        as_set = set(int(value) for value in feasible)
        common = as_set if common is None else common & as_set
    require(common is not None, "shared-d resolution produced no set")
    grid_common = [int(value) for value in grid if int(value) in common]
    if not grid_common:
        raise FaAlignmentError(
            "shared latent dimensionality d mismatch across sessions; "
            f"empty intersection of feasible d among { {key: list(val) for key, val in per_session_feasible.items()} }"
        )
    return int(max(grid_common))


def _rewarded_calibration_bins(
    *,
    trials_df: Any,
    bin_edges: np.ndarray,
    calibration_trials: int,
    window_size: int,
) -> np.ndarray:
    selected: list[tuple[int, int]] = []
    for _index, trial in trials_df.iterrows():
        if trial["result"] != "R":
            continue
        start_bin = int(np.searchsorted(bin_edges, trial["start_time"]))
        stop_bin = int(np.searchsorted(bin_edges, trial["stop_time"]))
        start_bin = max(0, start_bin)
        stop_bin = min(int(bin_edges.size - 1), stop_bin)
        if stop_bin - start_bin >= window_size:
            selected.append((start_bin, stop_bin))
        if len(selected) >= calibration_trials:
            break
    require(
        len(selected) >= calibration_trials,
        f"session has {len(selected)} usable rewarded trials, need {calibration_trials}",
    )
    bins: list[int] = []
    for start_bin, stop_bin in selected[:calibration_trials]:
        bins.extend(range(start_bin, stop_bin))
    return np.ascontiguousarray(bins, dtype=np.int64)


def load_subject_m_session_audit(
    nwb_path: Path,
    *,
    view: str,
    calibration_trials: int = SUBM_CALIBRATION_TRIALS,
) -> tuple[SessionAuditRow, np.ndarray]:
    from pynwb import NWBHDF5IO

    from sua_exploration.mc_maze.datamodule import bin_spikes
    from sua_exploration.mc_maze.multisession_datamodule import (
        electrode_ids_from_units,
        pool_spikes_by_electrode,
        session_name_from_path,
    )

    require(view in VIEWS, f"unsupported subject-M view: {view}")
    nwb_path = assert_path_in_scope(nwb_path)
    session_name = session_name_from_path(nwb_path)
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        require(nwb.units is not None, f"{nwb_path}: units table missing")
        units_df = nwb.units.to_dataframe()
        spike_columns = [np.asarray(times, dtype=np.float64).reshape(-1) for times in units_df["spike_times"].values]
        electrode_ids = electrode_ids_from_units(units_df)
        trials_df = nwb.intervals["trials"].to_dataframe()
    n_units = len(spike_columns)
    require(n_units >= 2, f"{session_name}: fewer than two units")
    all_spikes = np.concatenate(spike_columns)
    t_min = float(all_spikes.min())
    t_max = float(all_spikes.max())
    bin_size_s = BIN_SIZE_MS / 1000.0
    bin_edges = np.arange(t_min, t_max + bin_size_s, bin_size_s)
    calib_bins = _rewarded_calibration_bins(
        trials_df=trials_df,
        bin_edges=bin_edges,
        calibration_trials=calibration_trials,
        window_size=HISTORY_BINS,
    )
    first = int(calib_bins.min())
    last = int(calib_bins.max()) + 1
    local_edges = bin_edges[first : last + 1]
    unit_block = np.zeros((last - first, n_units), dtype=np.float64)
    for unit_index, spikes in enumerate(spike_columns):
        unit_block[:, unit_index] = bin_spikes(spikes, local_edges)
    shifted = calib_bins - first
    calib_units = unit_block[shifted]
    if view == "pseudo_mua":
        neural, channel_ids = pool_spikes_by_electrode(calib_units.astype(np.float32), electrode_ids)
        neural = np.asarray(neural, dtype=np.float64)
        positional_only = False
        ids_tuple = tuple(int(value) for value in np.asarray(channel_ids, dtype=np.int64).tolist())
    else:
        neural = np.asarray(calib_units, dtype=np.float64)
        channel_ids = np.arange(n_units, dtype=np.int64)
        positional_only = True
        ids_tuple = tuple(int(value) for value in channel_ids.tolist())
    n_channels = int(neural.shape[1])
    n_samples = int(neural.shape[0])
    ve, cumulative = pca_variance_explained(neural)
    row = _session_audit_row(
        session_name=session_name,
        view=view,
        n_channels=n_channels,
        n_units=n_units,
        n_samples=n_samples,
        channel_ids=ids_tuple,
        electrode_ids_per_unit=tuple(int(value) for value in np.asarray(electrode_ids, dtype=np.int64).tolist()),
        identifiers_are_positional_only=positional_only,
        variance_explained=ve,
        cumulative_variance_explained=cumulative,
    )
    return row, neural


def load_rt_unit_identifiers(nwb_path: Path) -> tuple[np.ndarray, tuple[int, ...] | None, bool]:
    from pynwb import NWBHDF5IO

    from sua_exploration.mc_maze.multisession_datamodule import electrode_ids_from_units

    nwb_path = assert_path_in_scope(nwb_path)
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        require(nwb.units is not None, f"{nwb_path}: units table missing")
        units_df = nwb.units.to_dataframe()
        n_units = int(len(units_df))
        electrode_ids = None
        if "electrodes" in units_df:
            try:
                electrode_ids = electrode_ids_from_units(units_df)
            except ValueError:
                electrode_ids = None
    require(n_units >= 2, f"{nwb_path}: fewer than two units")
    if electrode_ids is not None:
        channel_ids = np.asarray(electrode_ids, dtype=np.int64)
        return channel_ids, tuple(int(value) for value in channel_ids.tolist()), False
    channel_ids = np.arange(n_units, dtype=np.int64)
    return channel_ids, None, True


def load_rt_session_audit(
    nwb_path: Path,
    *,
    raw_loader: Any,
    calibration_trials: int = RT_CALIBRATION_TRIALS,
) -> tuple[SessionAuditRow, np.ndarray, dict[str, Any]]:
    from sua_exploration.mc_maze import rt_classical_comparators as rt

    nwb_path = assert_path_in_scope(nwb_path)
    channel_ids, electrode_tuple, positional_only = load_rt_unit_identifiers(nwb_path)
    raw = raw_loader(nwb_path)
    session_name = str(raw["session_name"])
    neural = np.asarray(raw["neural"], dtype=np.float64)
    trial_change = np.asarray(raw["trial_change"], dtype=bool)
    eval_mask = np.asarray(raw["eval_mask"], dtype=bool)
    starts = np.flatnonzero(trial_change)
    require(starts.size > calibration_trials, f"{session_name}: not enough RT trials for M24")
    boundary = int(starts[calibration_trials])
    calib_bins = np.flatnonzero(eval_mask[:boundary])
    require(calib_bins.size >= 3, f"{session_name}: empty RT calibration block")
    calib = neural[calib_bins]
    require(int(calib.shape[1]) == int(channel_ids.size), f"{session_name}: identifier/neural width mismatch")
    ve, cumulative = pca_variance_explained(calib)
    layout = rt.rt_outer_window_layout(
        session_name, raw["neural"], raw["covariates"], raw["trial_change"], raw["eval_mask"]
    )
    row = _session_audit_row(
        session_name=session_name,
        view="rt",
        n_channels=int(calib.shape[1]),
        n_units=int(calib.shape[1]),
        n_samples=int(calib.shape[0]),
        channel_ids=tuple(int(value) for value in np.asarray(channel_ids, dtype=np.int64).tolist()),
        electrode_ids_per_unit=electrode_tuple,
        identifiers_are_positional_only=positional_only,
        variance_explained=ve,
        cumulative_variance_explained=cumulative,
    )
    return row, calib, layout.query_window_audit


def _session_audit_row(
    *,
    session_name: str,
    view: str,
    n_channels: int,
    n_units: int,
    n_samples: int,
    channel_ids: tuple[int, ...],
    electrode_ids_per_unit: tuple[int, ...] | None,
    identifiers_are_positional_only: bool,
    variance_explained: np.ndarray,
    cumulative_variance_explained: np.ndarray,
) -> SessionAuditRow:
    spp = {str(d): samples_per_parameter(n_samples, n_channels, int(d)) for d in D_GRID}
    feasible = {
        str(d): bool(n_channels > int(d) and n_samples > int(d) and spp[str(d)] >= MIN_SAMPLES_PER_PARAMETER)
        for d in D_GRID
    }
    anisotropy = {str(d): {"d": float("nan")} for d in D_GRID}
    return SessionAuditRow(
        session_name=session_name,
        view=view,
        n_channels=n_channels,
        n_units=n_units,
        n_samples=n_samples,
        channel_ids=channel_ids,
        electrode_ids_per_unit=electrode_ids_per_unit,
        identifiers_are_positional_only=identifiers_are_positional_only,
        channel_ids_sha256=sha256_array(np.asarray(channel_ids, dtype=np.int64)),
        variance_explained=tuple(float(value) for value in variance_explained.tolist()),
        cumulative_variance_explained=tuple(float(value) for value in cumulative_variance_explained.tolist()),
        d80=d80_from_cumulative(cumulative_variance_explained),
        covariance_rank=0,
        top_d_anisotropy=anisotropy,
        samples_per_parameter_by_d=spp,
        fa_feasible_by_d=feasible,
    )


def _complete_session_row(row: SessionAuditRow, neural: np.ndarray) -> SessionAuditRow:
    anisotropy = {str(d): top_d_anisotropy(neural, int(d)) for d in D_GRID}
    return SessionAuditRow(
        session_name=row.session_name,
        view=row.view,
        n_channels=row.n_channels,
        n_units=row.n_units,
        n_samples=row.n_samples,
        channel_ids=row.channel_ids,
        electrode_ids_per_unit=row.electrode_ids_per_unit,
        identifiers_are_positional_only=row.identifiers_are_positional_only,
        channel_ids_sha256=row.channel_ids_sha256,
        variance_explained=row.variance_explained,
        cumulative_variance_explained=row.cumulative_variance_explained,
        d80=row.d80,
        covariance_rank=covariance_rank(neural),
        top_d_anisotropy=anisotropy,
        samples_per_parameter_by_d=row.samples_per_parameter_by_d,
        fa_feasible_by_d=row.fa_feasible_by_d,
    )


def session_row_to_dict(row: SessionAuditRow) -> dict[str, Any]:
    return {
        "session_name": row.session_name,
        "view": row.view,
        "n_channels": row.n_channels,
        "n_units": row.n_units,
        "n_samples": row.n_samples,
        "n_channel_ids": len(row.channel_ids),
        "identifiers_are_positional_only": row.identifiers_are_positional_only,
        "channel_ids_sha256": row.channel_ids_sha256,
        "channel_ids_head": list(row.channel_ids[:8]),
        "channel_ids_tail": list(row.channel_ids[-8:]) if len(row.channel_ids) > 8 else list(row.channel_ids),
        "electrode_ids_unique_count": (
            len(set(row.electrode_ids_per_unit)) if row.electrode_ids_per_unit is not None else None
        ),
        "variance_explained": list(row.variance_explained),
        "cumulative_variance_explained": list(row.cumulative_variance_explained),
        "d80": row.d80,
        "covariance_rank": row.covariance_rank,
        "top_d_anisotropy": row.top_d_anisotropy,
        "samples_per_parameter_by_d": row.samples_per_parameter_by_d,
        "fa_feasible_by_d": row.fa_feasible_by_d,
        "fa_parameter_count_by_d": {str(d): fa_parameter_count(row.n_channels, int(d)) for d in D_GRID},
    }


def _channel_info_from_row(row: SessionAuditRow, *, view: str) -> SessionChannelInfo:
    return SessionChannelInfo(
        asset_id=row.session_name,
        session_id=row.session_name,
        view=view,
        n_channels=row.n_channels,
        source_unit_count=row.n_units,
        channel_ids=row.channel_ids,
        electrode_ids_per_unit=row.electrode_ids_per_unit,
        channel_ids_sha256=row.channel_ids_sha256,
    )


def correspondence_from_rows(
    rows: Sequence[SessionAuditRow],
    *,
    view: str,
    identifiers_are_positional_only: bool | None = None,
) -> dict[str, Any]:
    require(rows, "correspondence requires sessions")
    positional = (
        bool(identifiers_are_positional_only)
        if identifiers_are_positional_only is not None
        else bool(view == "sua" or all(row.identifiers_are_positional_only for row in rows))
    )
    if view in VIEWS:
        audit = audit_view_correspondence([_channel_info_from_row(row, view=view) for row in rows], view=view)
        payload = correspondence_audit_to_dict(audit)
        payload["identifiers_are_positional_only"] = bool(view == "sua")
        return payload
    channel_rows = [np.asarray(row.channel_ids, dtype=np.int64) for row in rows]
    stable, identical, correspondent_count = detect_correspondence(
        channel_rows, identifiers_are_positional_only=positional
    )
    sets = [set(row.channel_ids) for row in rows]
    intersection = set.intersection(*sets)
    union = set.union(*sets)
    pairwise = [len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    dims_match = len({row.n_channels for row in rows}) == 1
    definable = bool(stable and correspondent_count > 0)
    return {
        "view": view,
        "spr_definable": definable,
        "status": "SPR_DEFINABLE" if definable else "SPR_UNDEFINED_NO_CHANNEL_CORRESPONDENCE",
        "n_sessions": len(rows),
        "channel_id_sets_intersection_size": len(intersection),
        "channel_id_sets_union_size": len(union),
        "stable_index_mapping_exists": bool(stable),
        "correspondent_channel_count": int(correspondent_count),
        "pairwise_channel_id_set_intersection_min": min(pairwise) if pairwise else len(intersection),
        "pairwise_channel_id_set_intersection_max": max(pairwise) if pairwise else len(intersection),
        "all_sessions_identical_channel_ids": bool(identical),
        "feature_dimension_matches_across_sessions": bool(dims_match),
        "identifiers_are_positional_only": bool(positional),
        "per_session": [session_row_to_dict(row) for row in rows],
    }


def shared_d_assessment(rows: Sequence[SessionAuditRow]) -> dict[str, Any]:
    d80_values = [row.d80 for row in rows]
    present = [int(value) for value in d80_values if value is not None]
    feasible = {row.session_name: [int(d) for d, ok in row.fa_feasible_by_d.items() if ok] for row in rows}
    spread = (max(present) - min(present)) if present else None
    defensible = False
    reason = "no session reached the variance-explained threshold"
    shared_d = None
    try:
        shared_d = resolve_shared_d_from_feasibility(feasible)
        if present and spread is not None and spread <= D_SPREAD_MAX:
            defensible = True
            reason = (
                f"feasible d intersection is nonempty (selected {shared_d} from {D_GRID}); "
                f"d80 spread {spread} <= {D_SPREAD_MAX}"
            )
        elif present:
            defensible = False
            reason = f"feasible d intersection selected {shared_d}, but d80 spread {spread} > {D_SPREAD_MAX}"
        else:
            defensible = True
            reason = (
                f"no session reached VE threshold {VE_THRESHOLD}; "
                f"falling back to feasible-grid intersection d={shared_d}"
            )
    except FaAlignmentError as exc:
        reason = str(exc)
    return {
        "defensible": bool(defensible),
        "reason": reason,
        "shared_d_from_feasible_grid": shared_d,
        "d80_per_session": {row.session_name: row.d80 for row in rows},
        "d80_min": min(present) if present else None,
        "d80_max": max(present) if present else None,
        "d80_spread": spread,
        "d80_sessions_reaching_threshold": len(present),
        "ve_threshold": VE_THRESHOLD,
        "d_spread_max": D_SPREAD_MAX,
        "d_grid": list(D_GRID),
        "feasible_d_per_session": feasible,
        "min_n_channels": min(row.n_channels for row in rows),
        "max_n_channels": max(row.n_channels for row in rows),
        "min_n_samples": min(row.n_samples for row in rows),
        "max_n_samples": max(row.n_samples for row in rows),
    }


def alignment_estimability(rows: Sequence[SessionAuditRow], *, shared_d: int | None) -> dict[str, Any]:
    d = int(shared_d) if shared_d is not None else int(D_GRID[0])
    ranks = [row.covariance_rank for row in rows]
    anisotropies = [
        float(row.top_d_anisotropy[str(d)]["anisotropy"])
        for row in rows
        if str(d) in row.top_d_anisotropy
        and math.isfinite(float(row.top_d_anisotropy[str(d)]["anisotropy"]))
    ]
    unique_rotation = [
        bool(row.top_d_anisotropy[str(d)]["rotation_uniquely_identifiable"])
        for row in rows
        if str(d) in row.top_d_anisotropy
    ]
    estimable = all(row.n_samples >= 3 and row.covariance_rank >= 2 for row in rows)
    return {
        "estimable_from_unlabelled_second_order_stats": bool(estimable),
        "evaluated_d": d,
        "min_covariance_rank": min(ranks) if ranks else 0,
        "max_covariance_rank": max(ranks) if ranks else 0,
        "sessions_with_rank_ge_2": int(sum(rank >= 2 for rank in ranks)),
        "min_top_d_anisotropy": min(anisotropies) if anisotropies else None,
        "max_top_d_anisotropy": max(anisotropies) if anisotropies else None,
        "sessions_with_unique_rotation": int(sum(unique_rotation)),
        "rotation_identifiability_note": (
            "Gaussian moment matching of FA posterior latents is estimable from unlabelled "
            "mean and covariance whenever rank >= 2.  A unique rotation additionally requires "
            f"anisotropy >= {ANISOTROPY_UNIQUE_ROTATION} in the top-d spectrum."
        ),
    }


def emit_variant_verdicts(
    *,
    correspondence: dict[str, Any],
    shared_d: dict[str, Any],
    estimability: dict[str, Any],
    rows: Sequence[SessionAuditRow],
) -> dict[str, Any]:
    correspondent = int(correspondence["correspondent_channel_count"])
    a1_definable = bool(correspondence.get("stable_index_mapping_exists") and correspondent > 0)
    if a1_definable:
        a1 = "FAA_DEFINABLE"
        a1_reason = None
    else:
        a1 = "FAA_UNDEFINED_NO_CHANNEL_CORRESPONDENCE"
        a1_reason = "stable cross-session channel correspondence is absent"

    fa_any = all(any(row.fa_feasible_by_d.values()) for row in rows)
    if not fa_any:
        a2 = "FAA_UNDEFINED_LATENT_FIT_INFEASIBLE"
        a2_reason = "no d in the predeclared grid is feasible on every session"
    elif not estimability["estimable_from_unlabelled_second_order_stats"]:
        a2 = "FAA_UNDEFINED_ALIGNMENT_NOT_ESTIMABLE"
        a2_reason = "unlabelled second-order statistics are rank-deficient on at least one session"
    elif not shared_d["defensible"] and shared_d["shared_d_from_feasible_grid"] is None:
        a2 = "FAA_UNDEFINED_SHARED_D_UNSTABLE"
        a2_reason = shared_d["reason"]
    else:
        a2 = "FAA_DEFINABLE"
        a2_reason = None
        if not shared_d["defensible"]:
            a2_reason = (
                "A2 remains definable at the feasible-grid intersection; "
                f"shared-d defensible flag is False because {shared_d['reason']}"
            )
    return {
        "A1": {
            "verdict": a1,
            "reason": a1_reason,
            "requires_channel_correspondence": True,
            "correspondent_channel_count": correspondent,
            "channel_id_sets_intersection_size": int(correspondence["channel_id_sets_intersection_size"]),
            "channel_id_sets_union_size": int(correspondence["channel_id_sets_union_size"]),
            "stable_index_mapping_exists": bool(correspondence["stable_index_mapping_exists"]),
            "all_sessions_identical_channel_ids": bool(correspondence["all_sessions_identical_channel_ids"]),
            "feature_dimension_matches_across_sessions": bool(
                correspondence["feature_dimension_matches_across_sessions"]
            ),
            "n_sessions": int(correspondence["n_sessions"]),
        },
        "A2": {
            "verdict": a2,
            "reason": a2_reason,
            "requires_channel_correspondence": False,
            "n_sessions": len(rows),
            "shared_d_from_feasible_grid": shared_d["shared_d_from_feasible_grid"],
            "shared_d_defensible": bool(shared_d["defensible"]),
            "fa_feasible_session_count": int(sum(any(row.fa_feasible_by_d.values()) for row in rows)),
            "min_n_channels": min(row.n_channels for row in rows),
            "max_n_channels": max(row.n_channels for row in rows),
            "min_n_samples": min(row.n_samples for row in rows),
            "max_n_samples": max(row.n_samples for row in rows),
            "estimable_from_unlabelled_second_order_stats": bool(
                estimability["estimable_from_unlabelled_second_order_stats"]
            ),
        },
    }


def build_part_a_payload(
    *,
    dataset: str,
    view: str,
    rows: Sequence[SessionAuditRow],
    correspondence: dict[str, Any],
    query_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    shared = shared_d_assessment(rows)
    estimability = alignment_estimability(rows, shared_d=shared["shared_d_from_feasible_grid"])
    verdicts = emit_variant_verdicts(
        correspondence=correspondence, shared_d=shared, estimability=estimability, rows=rows
    )
    a1_ok = verdicts["A1"]["verdict"] == "FAA_DEFINABLE"
    a2_ok = verdicts["A2"]["verdict"] == "FAA_DEFINABLE"
    return {
        "dataset": dataset,
        "view": view,
        "n_sessions": len(rows),
        "calibration_trials": SUBM_CALIBRATION_TRIALS if dataset == "subject_m" else RT_CALIBRATION_TRIALS,
        "correspondence": correspondence,
        "per_session": [session_row_to_dict(row) for row in rows],
        "latent_dimensionality": shared,
        "alignment_estimability": estimability,
        "verdicts": verdicts,
        "buildable_arms": {
            "faa_a2_align": bool(a2_ok),
            "faa_a1_anchored": bool(a1_ok),
            "faa_no_align": bool(a2_ok),
        },
        "query_identity": query_identity,
        "sealed_references": SEALED_REFERENCES[dataset],
        "target_labels_used_in_alignment": False,
        "alignment_fitted": False,
    }


def discover_subject_m_sessions(data_dir: Path) -> dict[str, Path]:
    data_dir = assert_path_in_scope(data_dir)
    indexed: dict[str, Path] = {}
    for session_name in SUBM_COHORT_SESSIONS:
        path = data_dir / f"{session_name}_behavior+ecephys.nwb"
        require(path.is_file(), f"missing subject-M NWB: {path}")
        indexed[session_name] = assert_path_in_scope(path)
    require(len(indexed) == SUBM_EXPECTED_SESSIONS, "subject-M cohort is not 15 sessions")
    return indexed


def audit_subject_m(*, data_dir: Path, view: str) -> dict[str, Any]:
    require(view in VIEWS, f"unsupported view: {view}")
    sessions = discover_subject_m_sessions(data_dir)
    rows: list[SessionAuditRow] = []
    for session_name in SUBM_COHORT_SESSIONS:
        row, neural = load_subject_m_session_audit(sessions[session_name], view=view)
        rows.append(_complete_session_row(row, neural))
        del neural
    correspondence = correspondence_from_rows(rows, view=view)
    query_identity = {
        "status": "NOT_OPENED_DURING_AUDIT",
        "calibration_budget_trials": SUBM_CALIBRATION_TRIALS,
        "query_boundary": "strictly_after_rewarded_trial_50",
        "binding_when_scored": (
            "kalman_comparator.build_subject_m_session_blocks / SPR load_v9_target "
            "on the sealed V9 query starts and targets"
        ),
        "missing_if_unbound": (
            "audit-only does not open the V9 result tree; scoring must bind the "
            "same sealed query-identity triple as the existing subject-M arms"
        ),
    }
    return build_part_a_payload(
        dataset="subject_m",
        view=view,
        rows=rows,
        correspondence=correspondence,
        query_identity=query_identity,
    )


def bind_rt_query_identity(
    *,
    layout_audit: Mapping[str, Any],
    sealed: Mapping[str, Any],
) -> dict[str, Any]:
    sealed_q = sealed["query_identity"]
    bound = (
        layout_audit["ordered_window_start_sha256"] == sealed_q["ordered_window_start_sha256"]
        and layout_audit["ordered_target_covariate_evalmask_sha256"]
        == sealed_q["ordered_target_covariate_evalmask_sha256"]
        and layout_audit["ordered_query_identity_sha256"] == sealed_q["ordered_query_identity_sha256"]
    )
    return {
        "session_name": sealed["session_name"],
        "fold": int(sealed["fold"]),
        "bound": bool(bound),
        "sealed": {
            "ordered_window_start_sha256": sealed_q["ordered_window_start_sha256"],
            "ordered_target_covariate_evalmask_sha256": sealed_q["ordered_target_covariate_evalmask_sha256"],
            "ordered_query_identity_sha256": sealed_q["ordered_query_identity_sha256"],
        },
        "computed": {
            "ordered_window_start_sha256": layout_audit["ordered_window_start_sha256"],
            "ordered_target_covariate_evalmask_sha256": layout_audit["ordered_target_covariate_evalmask_sha256"],
            "ordered_query_identity_sha256": layout_audit["ordered_query_identity_sha256"],
        },
    }


def audit_rt(*, data_dir: Path, stage2_cell_root: Path, raw_loader: Any) -> dict[str, Any]:
    from sua_exploration.mc_maze import rt_classical_comparators as rt

    data_dir = assert_path_in_scope(data_dir)
    stage2_cell_root = assert_path_in_scope(stage2_cell_root)
    from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions

    sessions = {
        rt.session_name_from_nwb_path(Path(path)): assert_path_in_scope(Path(path))
        for path in find_rt_sessions(data_dir)
    }
    require(len(sessions) == RT_EXPECTED_FOLDS, f"expected {RT_EXPECTED_FOLDS} RT sessions")
    rows: list[SessionAuditRow] = []
    bindings: list[dict[str, Any]] = []
    for fold in range(RT_EXPECTED_FOLDS):
        sealed = rt.load_sealed_stage2_cell(stage2_cell_root, fold)
        nwb_path = sessions[sealed["session_name"]]
        row, neural, layout_audit = load_rt_session_audit(nwb_path, raw_loader=raw_loader)
        rows.append(_complete_session_row(row, neural))
        bindings.append(bind_rt_query_identity(layout_audit=layout_audit, sealed=sealed))
        del neural
    positional = all(row.identifiers_are_positional_only for row in rows)
    correspondence = correspondence_from_rows(
        rows, view="rt", identifiers_are_positional_only=positional
    )
    query_identity = {
        "all_folds_bound_to_sealed_stage2": all(item["bound"] for item in bindings),
        "per_fold": bindings,
        "calibration_budget_trials": RT_CALIBRATION_TRIALS,
        "query_start_trial": RT_CALIBRATION_TRIALS,
    }
    return build_part_a_payload(
        dataset="rt",
        view="rt",
        rows=rows,
        correspondence=correspondence,
        query_identity=query_identity,
    )


def select_d_on_source_only(
    source_rows: Sequence[SessionAuditRow],
    *,
    target_query_used: bool = False,
) -> tuple[int, dict[str, Any]]:
    require(not target_query_used, "d selection must not use any target query window")
    feasible = {row.session_name: [int(d) for d, ok in row.fa_feasible_by_d.items() if ok] for row in source_rows}
    selected = resolve_shared_d_from_feasibility(feasible)
    return selected, {
        "selection_policy": "max_d_in_predeclared_grid_feasible_on_every_source_session",
        "selected_d": int(selected),
        "grid": list(D_GRID),
        "target_query_consumed": False,
        "feasible_d_per_source_session": feasible,
    }


def _require_a1_definable(part_a: Mapping[str, Any]) -> None:
    verdict = part_a["verdicts"]["A1"]["verdict"]
    require(verdict == "FAA_DEFINABLE", f"faa_a1_anchored is not definable: {verdict}")


def run_faa_no_align(
    *,
    source_latents: np.ndarray,
    source_labels: np.ndarray,
    target_latents: np.ndarray,
    normalized_lambda: float = NORMALIZED_LAMBDA_FIXED,
) -> tuple[np.ndarray, AlignmentTransform]:
    require(source_latents.shape[1] == target_latents.shape[1], "no-align latent d mismatch")
    transform = identity_alignment(int(source_latents.shape[1]), arm="faa_no_align")
    require(transform.is_identity(), "faa_no_align transform is not identity")
    readout = fit_ridge(source_latents, source_labels, normalized_lambda=normalized_lambda, device="cpu")
    aligned = transform.apply(target_latents)
    require(np.array_equal(aligned, np.asarray(target_latents, dtype=np.float64)), "faa_no_align modified latents")
    return predict_ridge(aligned.astype(np.float32), readout, device="cpu"), transform


def run_faa_a2_align(
    *,
    source_latents: np.ndarray,
    source_labels: np.ndarray,
    target_latents: np.ndarray,
    target_labels: np.ndarray | None = None,
    normalized_lambda: float = NORMALIZED_LAMBDA_FIXED,
) -> tuple[np.ndarray, AlignmentTransform]:
    transform = estimate_latent_alignment(
        source_latents, target_latents, target_labels=target_labels, arm="faa_a2_align"
    )
    readout = fit_ridge(source_latents, source_labels, normalized_lambda=normalized_lambda, device="cpu")
    aligned = transform.apply(target_latents)
    return predict_ridge(aligned.astype(np.float32), readout, device="cpu"), transform


def run_faa_a1_anchored(
    *,
    part_a: Mapping[str, Any],
    source_loadings: np.ndarray,
    target_loadings: np.ndarray,
    source_latents: np.ndarray,
    source_labels: np.ndarray,
    target_latents: np.ndarray,
    target_labels: np.ndarray | None = None,
    normalized_lambda: float = NORMALIZED_LAMBDA_FIXED,
) -> tuple[np.ndarray, AlignmentTransform]:
    _require_a1_definable(part_a)
    transform = estimate_channel_anchored_alignment(
        source_loadings, target_loadings, target_labels=target_labels, arm="faa_a1_anchored"
    )
    readout = fit_ridge(source_latents, source_labels, normalized_lambda=normalized_lambda, device="cpu")
    aligned = transform.apply(target_latents)
    return predict_ridge(aligned.astype(np.float32), readout, device="cpu"), transform


def bind_subject_m_sealed_query_identities(
    *,
    repo_root: Path,
    data_dir: Path,
    v9_run_root: Path,
    view: str,
    budget: int = SUBM_CALIBRATION_TRIALS,
) -> dict[str, Any]:
    """Bind the sealed subject-M query identities used by existing comparator arms.

    Implemented against the same V9 helpers as ``kalman_comparator`` / SPR.
    Refuses to open a forbidden path; if the only known sealed artifact root is
    out of scope, this raises rather than inventing hashes.
    """
    v9_run_root = v9_run_root.expanduser()
    if path_is_forbidden(v9_run_root):
        raise NotImplementedError(
            "subject-M sealed query identities live in the V9 result tree used by "
            "kalman_comparator.build_subject_m_session_blocks and "
            "run_source_pooled_ridge.load_v9_target; that tree's path contains a "
            "forbidden token for this session, so hashes cannot be computed here. "
            "Scoring must bind query_starts_sha256 / query_target_sha256 / "
            "ordered_query_identity_sha256 from those helpers without guessing."
        )
    raise NotImplementedError(
        "subject-M sealed query-identity binding is implemented only as the V9 "
        "helper path used by kalman_comparator.build_subject_m_session_blocks; "
        f"no alternate in-scope artifact root was supplied (got {v9_run_root})"
    )


def evaluate_arm_on_session_blocks(
    *,
    arm: str,
    source_neural: np.ndarray,
    source_labels: np.ndarray,
    target_neural: np.ndarray,
    d: int,
    part_a: Mapping[str, Any] | None = None,
    target_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Synthetic/session-block scoring helper.  Not the real-data runner."""
    require(arm in ARMS, f"unknown arm: {arm}")
    source_fa = fit_factor_analysis(source_neural, d)
    target_fa = fit_factor_analysis(target_neural, d)
    source_latents = source_fa.transform(source_neural)
    target_latents = target_fa.transform(target_neural)
    if arm == "faa_no_align":
        prediction, transform = run_faa_no_align(
            source_latents=source_latents,
            source_labels=source_labels,
            target_latents=target_latents,
        )
    elif arm == "faa_a2_align":
        prediction, transform = run_faa_a2_align(
            source_latents=source_latents,
            source_labels=source_labels,
            target_latents=target_latents,
            target_labels=target_labels,
        )
    else:
        require(part_a is not None, "faa_a1_anchored requires a Part A payload")
        prediction, transform = run_faa_a1_anchored(
            part_a=part_a,
            source_loadings=source_fa.loadings,
            target_loadings=target_fa.loadings,
            source_latents=source_latents,
            source_labels=source_labels,
            target_latents=target_latents,
            target_labels=target_labels,
        )
    return {
        "arm": arm,
        "d": int(d),
        "prediction": prediction,
        "transform_is_identity": transform.is_identity(),
        "transform_matrix": transform.matrix,
        "target_labels_used_in_alignment": False,
    }


def dataset_query_identity_sha256(per_fold_ordered_hashes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in per_fold_ordered_hashes:
        digest.update(str(value).encode("ascii"))
    return digest.hexdigest()


def paired_contrast(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    require(a.shape == b.shape and a.size > 0, "paired contrast length mismatch")
    delta = a - b
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "n_positive": int(np.sum(delta > 0.0)),
        "n_negative": int(np.sum(delta < 0.0)),
        "n_zero": int(np.sum(delta == 0.0)),
        "n": int(delta.size),
        "per_session_delta": [float(value) for value in delta.tolist()],
    }


def load_rt_scoring_session(
    nwb_path: Path,
    *,
    raw_loader: Any,
    sealed: Mapping[str, Any],
    calibration_trials: int = RT_CALIBRATION_TRIALS,
) -> dict[str, Any]:
    from sua_exploration.mc_maze import rt_classical_comparators as rt

    nwb_path = assert_path_in_scope(nwb_path)
    raw = raw_loader(nwb_path)
    session_name = str(raw["session_name"])
    require(session_name == sealed["session_name"], f"session name drift vs sealed cell: {session_name}")
    neural = np.asarray(raw["neural"], dtype=np.float64)
    covariates = np.asarray(raw["covariates"], dtype=np.float64)
    trial_change = np.asarray(raw["trial_change"], dtype=bool)
    eval_mask = np.asarray(raw["eval_mask"], dtype=bool)
    starts = np.flatnonzero(trial_change)
    require(starts.size > calibration_trials, f"{session_name}: not enough RT trials for M24")
    boundary = int(starts[calibration_trials])
    calib_bins = np.flatnonzero(eval_mask[:boundary])
    require(calib_bins.size >= 3, f"{session_name}: empty RT calibration block")
    layout = rt.rt_outer_window_layout(session_name, raw["neural"], raw["covariates"], trial_change, eval_mask)
    neural_p, cov_p, _trial_p, _eval_p, _starts = rt.pad_rt_like_falcon(
        raw["neural"], raw["covariates"], trial_change, eval_mask
    )
    query_bins = np.ascontiguousarray(layout.query_window_starts + HISTORY_BINS - 1, dtype=np.int64)
    binding = bind_rt_query_identity(layout_audit=layout.query_window_audit, sealed=sealed)
    require(binding["bound"], f"{session_name}: query identity failed to bind to sealed Stage-2")
    return {
        "session_name": session_name,
        "fold": int(sealed["fold"]),
        "n_channels": int(neural.shape[1]),
        "calib_neural": np.ascontiguousarray(neural[calib_bins], dtype=np.float64),
        "calib_labels": np.ascontiguousarray(covariates[calib_bins], dtype=np.float64),
        "query_neural": np.ascontiguousarray(neural_p[query_bins], dtype=np.float64),
        "query_labels": np.ascontiguousarray(cov_p[query_bins], dtype=np.float64),
        "query_identity": binding,
        "t4d_reference_r2": float(sealed["t4d_r2"]),
        "n_calib": int(calib_bins.size),
        "n_query": int(query_bins.size),
    }


def _fit_rt_session_fa(session: Mapping[str, Any], *, d: int) -> dict[str, Any]:
    require(int(session["n_channels"]) > int(d), f"{session['session_name']}: n_channels <= d")
    fa = fit_factor_analysis(session["calib_neural"], d)
    return {
        "fa": fa,
        "calib_latents": fa.transform(session["calib_neural"]),
        "query_latents": fa.transform(session["query_neural"]),
        "n_iter": int(fa.n_iter),
    }


def _score_rt_fold(
    *,
    target: Mapping[str, Any],
    target_fit: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    source_fits: Sequence[Mapping[str, Any]],
    d: int,
    arms: Sequence[str],
) -> dict[str, Any]:
    require(all(arm in SCORING_ARMS_RT for arm in arms), "RT scoring permits only faa_a2_align and faa_no_align")
    require("faa_a1_anchored" not in arms, "faa_a1_anchored is refused on RT")
    source_latents = [fit["calib_latents"] for fit in source_fits]
    source_labels = [source["calib_labels"] for source in sources]
    source_fa_iters = [int(fit["n_iter"]) for fit in source_fits]
    pooled_source_latents = np.concatenate(source_latents, axis=0)
    pooled_source_labels = np.concatenate(source_labels, axis=0)
    target_calib_latents = target_fit["calib_latents"]
    target_query_latents = target_fit["query_latents"]
    target_fa = target_fit["fa"]
    arm_rows: dict[str, Any] = {}
    for arm in arms:
        if arm == "faa_no_align":
            prediction, transform = run_faa_no_align(
                source_latents=pooled_source_latents,
                source_labels=pooled_source_labels,
                target_latents=target_query_latents,
            )
        else:
            transform = estimate_latent_alignment(
                pooled_source_latents,
                target_calib_latents,
                target_labels=None,
                arm="faa_a2_align",
            )
            readout = fit_ridge(
                pooled_source_latents, pooled_source_labels, normalized_lambda=NORMALIZED_LAMBDA_FIXED, device="cpu"
            )
            aligned_query = transform.apply(target_query_latents)
            prediction = predict_ridge(aligned_query.astype(np.float32), readout, device="cpu")
        r2 = score_predictions(prediction, target["query_labels"], dataset="rt")
        arm_rows[arm] = {
            "r2": float(r2),
            "transform_is_identity": bool(transform.is_identity()),
            "target_labels_used_in_alignment": False,
        }
    return {
        "session_name": target["session_name"],
        "fold": int(target["fold"]),
        "n_channels": int(target["n_channels"]),
        "n_source_sessions": int(len(sources)),
        "n_source_latent_rows": int(pooled_source_latents.shape[0]),
        "n_calib": int(target["n_calib"]),
        "n_query": int(target["n_query"]),
        "target_fa_n_iter": int(target_fa.n_iter),
        "source_fa_n_iter": source_fa_iters,
        "t4d_reference_r2": float(target["t4d_reference_r2"]),
        "query_identity": target["query_identity"],
        "arms": arm_rows,
        "target_labels_used_in_alignment": False,
    }


def score_rt(
    *,
    data_dir: Path,
    stage2_cell_root: Path,
    raw_loader: Any,
    d: int = SELECTED_RT_D,
    arms: Sequence[str] = SCORING_ARMS_RT,
) -> dict[str, Any]:
    from sua_exploration.mc_maze import rt_classical_comparators as rt

    require("faa_a1_anchored" not in arms, "faa_a1_anchored remains refused on RT")
    data_dir = assert_path_in_scope(data_dir)
    stage2_cell_root = assert_path_in_scope(stage2_cell_root)
    from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions

    sessions = {
        rt.session_name_from_nwb_path(Path(path)): assert_path_in_scope(Path(path))
        for path in find_rt_sessions(data_dir)
    }
    require(len(sessions) == RT_EXPECTED_FOLDS, f"expected {RT_EXPECTED_FOLDS} RT sessions")
    loaded: list[dict[str, Any]] = []
    for fold in range(RT_EXPECTED_FOLDS):
        sealed = rt.load_sealed_stage2_cell(stage2_cell_root, fold)
        loaded.append(load_rt_scoring_session(sessions[sealed["session_name"]], raw_loader=raw_loader, sealed=sealed))
    require(all(item["query_identity"]["bound"] for item in loaded), "query identity failed to bind on at least one fold")
    per_fold_hashes = [
        item["query_identity"]["sealed"]["ordered_query_identity_sha256"] for item in loaded
    ]
    computed_hashes = [
        item["query_identity"]["computed"]["ordered_query_identity_sha256"] for item in loaded
    ]
    require(per_fold_hashes == computed_hashes, "computed query identity drifted from sealed Stage-2")
    dataset_hash = dataset_query_identity_sha256(per_fold_hashes)
    session_fits = [_fit_rt_session_fa(item, d=int(d)) for item in loaded]
    fold_rows = []
    for index, target in enumerate(loaded):
        sources = [item for j, item in enumerate(loaded) if j != index]
        source_fits = [item for j, item in enumerate(session_fits) if j != index]
        fold_rows.append(
            _score_rt_fold(
                target=target,
                target_fit=session_fits[index],
                sources=sources,
                source_fits=source_fits,
                d=int(d),
                arms=arms,
            )
        )
    per_arm: dict[str, Any] = {}
    t4d = [float(row["t4d_reference_r2"]) for row in fold_rows]
    for arm in arms:
        scores = [float(row["arms"][arm]["r2"]) for row in fold_rows]
        per_arm[arm] = {
            "mean_r2": float(np.mean(scores)),
            "median_r2": float(np.median(scores)),
            "per_session_r2": {row["session_name"]: float(row["arms"][arm]["r2"]) for row in fold_rows},
            "vs_t4d": paired_contrast(t4d, scores),
            "target_labels_used_in_alignment": False,
        }
    contrasts = {}
    if "faa_a2_align" in per_arm and "faa_no_align" in per_arm:
        a2 = [float(row["arms"]["faa_a2_align"]["r2"]) for row in fold_rows]
        no = [float(row["arms"]["faa_no_align"]["r2"]) for row in fold_rows]
        contrasts["faa_a2_align_minus_faa_no_align"] = paired_contrast(a2, no)
        contrasts["carrier_t4d_minus_faa_a2_align"] = paired_contrast(t4d, a2)
    return {
        "dataset": "rt",
        "metric": metric_for_dataset("rt"),
        "selected_d": int(d),
        "shared_d_defensible": False,
        "shared_d_limitation": RT_SHARED_D_LIMITATION,
        "source_pooling": (
            "per-session FA on each source calibration block, then concatenate latents; "
            "observation-space pooling is impossible because channel counts differ"
        ),
        "arms_run": list(arms),
        "arms_refused": {"faa_a1_anchored": "FAA_UNDEFINED_NO_CHANNEL_CORRESPONDENCE"},
        "target_labels_used_in_alignment": False,
        "query_identity": {
            "all_folds_bound_to_sealed_stage2": True,
            "dataset_query_identity_sha256": dataset_hash,
            "per_fold_ordered_query_identity_sha256": per_fold_hashes,
            "calibration_budget_trials": RT_CALIBRATION_TRIALS,
        },
        "per_fold": fold_rows,
        "arms": per_arm,
        "contrasts": contrasts,
        "sealed_references": SEALED_REFERENCES["rt"],
    }
