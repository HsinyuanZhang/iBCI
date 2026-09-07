"""Source-frozen log / per-frequency standardizer / PCA8 with sign lock."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import N_FREQ, VALID_END, VALID_START
from .metric import log_from_raw, standardize
from .util import sha256_array


@dataclass
class AcousticBasis:
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray  # [8, 158]
    singular_values: np.ndarray
    pca_center: np.ndarray
    train_dates: tuple[str, ...]
    n_frames: int
    mean_sha256: str
    std_sha256: str
    basis_sha256: str
    sign_sha256: str

    def B(self, q: int) -> np.ndarray:
        if q not in (3, 8):
            raise ValueError("q must be 3 or 8")
        return self.components[:q]

    def transform_log(self, log_spec: np.ndarray) -> np.ndarray:
        """log_spec [..., 158] -> standardized [..., 158]. Frequency is last axis."""
        return standardize(log_spec, self.mean, self.std)

    def project(self, log_spec: np.ndarray, q: int) -> np.ndarray:
        z_log = self.transform_log(log_spec) - self.pca_center
        return z_log @ self.B(q).T

    def inverse_pca(self, z_q: np.ndarray, q: int) -> np.ndarray:
        """z_q [..., q] -> standardized-log [..., 158]."""
        return z_q @ self.B(q) + self.pca_center

    def to_raw(self, z_log: np.ndarray) -> np.ndarray:
        from .metric import standardized_log_to_raw

        return standardized_log_to_raw(z_log, self.mean, self.std)


def _valid_log_frames(trials) -> np.ndarray:
    frames = []
    for trial in trials:
        log_spec = log_from_raw(trial.spectrogram)  # [158, 880]
        frames.append(log_spec[:, VALID_START:VALID_END].T)  # [700, 158]
    return np.concatenate(frames, axis=0)


def fit_acoustic_basis(train_trials, train_dates: tuple[str, ...]) -> AcousticBasis:
    log_frames = _valid_log_frames(train_trials)
    mean = log_frames.mean(axis=0)
    std = log_frames.std(axis=0, ddof=0)
    std = np.where(std == 0.0, 1.0, std)
    z = (log_frames - mean) / std
    pca_center = z.mean(axis=0)
    xc = z - pca_center
    _, singular, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:8].copy()
    signs = []
    for i in range(8):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] *= -1.0
            signs.append(-1)
        else:
            signs.append(1)
    signs = np.asarray(signs, dtype=np.int8)
    return AcousticBasis(
        mean=mean.astype(np.float64),
        std=std.astype(np.float64),
        components=components.astype(np.float64),
        singular_values=singular[:8].astype(np.float64),
        pca_center=pca_center.astype(np.float64),
        train_dates=tuple(train_dates),
        n_frames=int(log_frames.shape[0]),
        mean_sha256=sha256_array(mean.astype(np.float64)),
        std_sha256=sha256_array(std.astype(np.float64)),
        basis_sha256=sha256_array(components.astype(np.float64)),
        sign_sha256=sha256_array(signs),
    )


def basis_excludes_date(basis: AcousticBasis, date: str) -> bool:
    return date not in basis.train_dates
