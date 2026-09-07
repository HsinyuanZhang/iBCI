"""Fold-local naming and the target-support dictionary immutability probe."""
from __future__ import annotations

import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as parent_syn3

from . import plan

RECTIFIED_TRIAL_MEAN_PCA3_LABEL = "rectified trial-mean PCA3"


def dictionary_immutability_probe(basis, target_support_emg: np.ndarray) -> dict[str, object]:
    """NNLS-project real target-support EMG; the frozen dictionary must not move."""
    emg = np.asarray(target_support_emg, dtype=np.float64)
    if emg.ndim != 2 or emg.shape[0] < 1:
        raise rsyn3.Syn3Error("immutability probe needs target-support EMG bins")
    before = np.array(basis.dictionary, copy=True)
    digest_before = rsyn3.array_digest(before)
    rsyn3.nnls_activations(rsyn3.apply_scale(emg, basis.scale), basis.dictionary)
    digest_after = rsyn3.array_digest(np.asarray(basis.dictionary))
    mutated = bool(not np.array_equal(before, basis.dictionary))
    return {
        "source": "target_support_emg",
        "n_bins": int(emg.shape[0]),
        "n_channels": int(emg.shape[1]),
        "dictionary_digest_before": digest_before,
        "dictionary_digest_after": digest_after,
        "mutated": mutated,
    }


def rectified_trial_mean_pca3_reliability(
    emg_bins: np.ndarray, rates: np.ndarray, trial_ids: np.ndarray,
) -> dict[str, object]:
    """Disclosure-only diagnostic. Trial means are taken after ReLU, not before."""
    emg = rectify.relu_nonnegative_projection(emg_bins)
    r = np.asarray(rates, dtype=np.float64)
    ids = np.asarray(trial_ids)
    unique = np.unique(ids)
    if unique.size < plan.RANK:
        return {
            "label": RECTIFIED_TRIAL_MEAN_PCA3_LABEL,
            "bin_law": "mean_of_rectified_bins",
            "weight_flattened_pearson": None,
            "intercept_pearson": None,
            "split": "unavailable",
            "reason": "trial-mean PCA rank exceeds n_trials",
            "n_trials": int(unique.size),
        }
    means = np.stack([emg[ids == trial].mean(axis=0) for trial in unique])
    rate_means = np.stack([r[ids == trial].mean(axis=0) for trial in unique])
    report = dict(parent_syn3.trial_mean_pca_reliability(means, rate_means))
    report["label"] = RECTIFIED_TRIAL_MEAN_PCA3_LABEL
    report["bin_law"] = "mean_of_rectified_bins"
    report["n_trials"] = int(unique.size)
    return report


def zero4_never_fits_target(controls: dict[str, object]) -> dict[str, object]:
    payload = dict(controls)
    payload["zero4_target_fit_calls"] = 0
    return payload
