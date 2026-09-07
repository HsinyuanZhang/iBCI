"""EMG-rSyn3 wrappers: ReLU first, then parent Syn3 helpers."""
from __future__ import annotations

import numpy as np

from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as parent_syn3

from . import plan
from . import rectify

Syn3Error = parent_syn3.Syn3Error
SourceBasis = parent_syn3.SourceBasis
array_digest = parent_syn3.array_digest
positive_scale = parent_syn3.positive_scale
apply_scale = parent_syn3.apply_scale
nnls_activations = parent_syn3.nnls_activations
fit_unit_ridge = parent_syn3.fit_unit_ridge
fit_all_units = parent_syn3.fit_all_units
carrier_from_encoding = parent_syn3.carrier_from_encoding
require_support_bins = parent_syn3.require_support_bins
coverage_report = parent_syn3.coverage_report
trial_stratified_split_half = parent_syn3.trial_stratified_split_half
normalize_carriers = parent_syn3.normalize_carriers
deterministic_row_permutation = parent_syn3.deterministic_row_permutation
deterministic_trial_derangement = parent_syn3.deterministic_trial_derangement
source_normalizer = parent_syn3.source_normalizer


def derange_trial_association(
    scores: np.ndarray, rates: np.ndarray, trial_ids: np.ndarray, session_name: str, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trial-level EMG/neural derangement that allows unequal trial lengths."""
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    ids = np.asarray(trial_ids)
    unique = np.unique(ids)
    order = deterministic_trial_derangement(len(unique), session_name=session_name, seed=seed)
    remapped = np.empty_like(z)
    for dest_index, source_index in enumerate(order):
        dest_mask = ids == unique[dest_index]
        source_vals = z[ids == unique[source_index]]
        dest_n = int(np.sum(dest_mask))
        if source_vals.shape[0] == dest_n:
            remapped[dest_mask] = source_vals
        else:
            take = np.arange(dest_n) % int(source_vals.shape[0])
            remapped[dest_mask] = source_vals[take]
    return remapped, r


def fit_source_nmf(emg: np.ndarray) -> parent_syn3.SourceBasis:
    rectified = rectify.relu_nonnegative_projection(emg)
    return parent_syn3.fit_source_nmf(rectified)


def fit_source_pca(emg: np.ndarray) -> parent_syn3.SourceBasis:
    rectified = rectify.relu_nonnegative_projection(emg)
    return parent_syn3.fit_source_pca(rectified)


def project_basis(emg: np.ndarray, basis: parent_syn3.SourceBasis) -> np.ndarray:
    rectified = rectify.relu_nonnegative_projection(emg)
    return parent_syn3.project_basis(rectified, basis)


def trial_mean_pca_reliability(emg_means: np.ndarray, rates: np.ndarray) -> dict[str, object]:
    """Historical trial-mean signed-PCA split-half. M2 is typed unavailable."""
    means = np.asarray(emg_means, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    if means.shape[0] < plan.RANK:
        return {
            "weight_flattened_pearson": None,
            "intercept_pearson": None,
            "split": "unavailable",
            "reason": "trial-mean PCA rank exceeds n_trials",
            "n_trials": int(means.shape[0]),
        }
    rectified = rectify.relu_nonnegative_projection(means)
    return parent_syn3.trial_mean_pca_reliability(rectified, r)


def build_controls(**kwargs) -> dict[str, object]:
    payload = parent_syn3.build_controls(**kwargs)
    payload["rSyn3"] = payload["Syn3"]
    return payload
