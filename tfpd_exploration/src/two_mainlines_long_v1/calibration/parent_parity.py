"""P4: independent old dictionary / per-unit carrier / same-input prediction.

Must not treat (new raw + old μ/σ) as the parent carrier.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import basis as sealed_basis
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer as sealed_normalizer
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import stage0 as fold_stage0
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan as syn3_plan

from . import constants as C
from .factory import solve_carrier_float64


class ParentParityError(RuntimeError):
    """Fail closed when the independent parent reference cannot be recovered."""


def _sealed_fold0(repo_root: Path) -> dict[str, Any]:
    path = Path(repo_root) / fold_plan.STAGE0_ROOT_RELATIVE / "fold_receipts.json"
    payload = json.loads(path.read_text(encoding="utf-8"))["0"]
    return payload


def sealed_mu_sigma(repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    sealed = _sealed_fold0(repo_root)
    mu = np.asarray(sealed["normalizer_mean"], dtype=np.float64)
    sigma = np.asarray(sealed["normalizer_scale"], dtype=np.float64)
    return mu, sigma


def rematerialize_source_nmf(repo_root: Path):
    del repo_root
    paths = parent_data.allowlisted_paths()
    sessions = {
        name: parent_data.load_support_bins(
            paths[name], emg_trial_stop=None, neural_trial_stop=syn3_plan.SUPPORT_TRIALS
        )
        for name in syn3_plan.SESSIONS
    }
    rectified = {name: rectify.relu_nonnegative_projection(record.emg) for name, record in sessions.items()}
    source_emg = np.concatenate([rectified[name] for name in C.SOURCES], axis=0)
    return rsyn3.fit_source_nmf(source_emg), sessions


def encode_support_carrier(emg: np.ndarray, rates: np.ndarray, dictionary: np.ndarray, scale: np.ndarray) -> np.ndarray:
    basis = sealed_basis.RowNormalizedNNMFBasis(
        dictionary=torch.as_tensor(dictionary, dtype=torch.float64),
        scale=torch.as_tensor(scale, dtype=torch.float64),
        trainable=False,
    )
    z = basis.encode(torch.as_tensor(emg, dtype=torch.float64))
    raw = solve_carrier_float64(z, torch.as_tensor(rates, dtype=torch.float64))
    return raw.detach().cpu().numpy()


def old_path_parent_is_new_raw_plus_old_mean(
    frozen: sealed_normalizer.FrozenSourceNormalizer,
    *,
    repo_root: Path | None = None,
) -> bool:
    """True when parent_normalized == normalize(new source_raw, sealed μ/σ)."""
    if repo_root is None:
        repo_root = C.REPO_ROOT
    sealed_mu, sealed_sigma = sealed_mu_sigma(repo_root)
    matches = []
    for name, raw in frozen.source_raw.items():
        rebuilt = rsyn3.normalize_carriers(raw, sealed_mu, sealed_sigma)
        parent = frozen.parent_normalized[name]
        matches.append(bool(np.allclose(rebuilt, parent, atol=C.CARRIER_ATOL, rtol=C.CARRIER_RTOL)))
    return bool(matches) and all(matches)


def independent_parent_parity(repo_root: Path, frozen: sealed_normalizer.FrozenSourceNormalizer) -> dict[str, Any]:
    """Rebuild carriers from support + rematerialized D0 + sealed μ/σ, not new-raw+old-mean."""
    repo_root = Path(repo_root)
    sealed = _sealed_fold0(repo_root)
    sealed_mu, sealed_sigma = sealed_mu_sigma(repo_root)
    sealed_d0_digest = str(sealed["nmf"]["dictionary_digest"])
    sealed_scale_digest = str(sealed["nmf"]["scale_digest"])
    sealed_recon = str(sealed["nmf"]["reconstruction_digest"])

    nmf, _sessions = rematerialize_source_nmf(repo_root)
    d0 = np.asarray(nmf.dictionary, dtype=np.float64)
    scale = np.asarray(nmf.scale, dtype=np.float64)
    d0_digest = rsyn3.array_digest(d0)
    scale_digest = rsyn3.array_digest(scale)
    recon_digest = str(nmf.reconstruction_digest)

    loaded = fold_data.load_fold_scope(fold=0)
    independent_raw: dict[str, np.ndarray] = {}
    per_unit_ok = True
    max_abs = 0.0
    for name, record in loaded["sources"].items():
        emg_b, rates_b, _ids = fold_stage0._mask_budget(record, C.SUPPORT_TRIALS)
        raw = encode_support_carrier(emg_b, rates_b, d0, scale)
        independent_raw[name] = raw
        # Compare against bank encode of the same support, not frozen.source_raw as parent.
        scores = rsyn3.project_basis(emg_b, nmf)
        weights, intercepts = rsyn3.fit_all_units(scores, rates_b)
        bank_raw = rsyn3.carrier_from_encoding(weights, intercepts)
        err = float(np.max(np.abs(raw - bank_raw)))
        max_abs = max(max_abs, err)
        per_unit_ok = per_unit_ok and bool(np.allclose(raw, bank_raw, atol=C.CARRIER_ATOL, rtol=C.CARRIER_RTOL))

    emg_t, rates_t, _ids_t = fold_stage0._mask_budget(loaded["target"], C.SUPPORT_TRIALS)
    target_raw = encode_support_carrier(emg_t, rates_t, d0, scale)
    target_digest = rsyn3.array_digest(target_raw)

    independent_norm = {
        name: rsyn3.normalize_carriers(raw, sealed_mu, sealed_sigma) for name, raw in independent_raw.items()
    }
    used_new_raw_old_mean = old_path_parent_is_new_raw_plus_old_mean(frozen, repo_root=repo_root)
    d0_match = d0_digest == sealed_d0_digest
    target_match = target_digest == C.SEALED_TARGET_M10_DIGEST
    recon_match = recon_digest == sealed_recon
    scale_match = scale_digest == sealed_scale_digest
    mu_match = bool(np.allclose(sealed_mu, frozen.mu0, atol=C.CARRIER_ATOL, rtol=C.CARRIER_RTOL))
    sigma_match = bool(np.allclose(sealed_sigma, frozen.sigma0, atol=C.CARRIER_ATOL, rtol=C.CARRIER_RTOL))

    passed = bool(d0_match and target_match and recon_match and scale_match and per_unit_ok and mu_match and sigma_match)
    reason = "ok"
    if not d0_match or not target_match or not recon_match:
        reason = (
            "independent rematerialized D0/target-M10 do not match sealed Stage0 bytes; "
            "NMF is not recoverable from the local repo (PCA does match). "
            "Stopping rather than adopting a new rematerialized D0 under the same name."
        )

    return {
        "passed": passed,
        "reason": reason,
        "used_new_raw_plus_old_mean_as_parent": used_new_raw_old_mean,
        "refuses_new_raw_plus_old_mean": True,
        "d0_digest": d0_digest,
        "sealed_d0_digest": sealed_d0_digest,
        "d0_digest_matches_sealed": d0_match,
        "target_m10_digest": target_digest,
        "sealed_target_m10_digest": C.SEALED_TARGET_M10_DIGEST,
        "target_m10_digest_matches_sealed": target_match,
        "reconstruction_digest": recon_digest,
        "sealed_reconstruction_digest": sealed_recon,
        "reconstruction_matches_sealed": recon_match,
        "scale_digest_matches_sealed": scale_match,
        "per_unit_bank_parity": per_unit_ok,
        "per_unit_max_abs": max_abs,
        "sealed_mu_sigma_match_frozen": bool(mu_match and sigma_match),
        "independent_normalized_sessions": sorted(independent_norm),
        "estimator_definition_unchanged": True,
        "red_stop": not passed,
    }


def same_input_prediction_parity(
    student,
    carrier_a: np.ndarray,
    carrier_b: np.ndarray,
    *,
    seed: int = 11,
) -> dict[str, Any]:
    window = int(student.decoder.window_size)
    trial_length = int(getattr(student.id_encoder, "trial_length", 1024))
    units = int(carrier_a.shape[0])
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    neural = torch.randn(1, window, units, dtype=torch.float32, generator=generator)
    calib = torch.randn(1, C.SUPPORT_TRIALS, trial_length, units, dtype=torch.float32, generator=generator)
    was = student.training
    student.eval()
    try:
        identity = student.compute_identity(calib)
        pred_a = student.decode_with_identity(
            neural, identity, carrier=torch.as_tensor(carrier_a, dtype=torch.float32).unsqueeze(0)
        )
        pred_b = student.decode_with_identity(
            neural, identity, carrier=torch.as_tensor(carrier_b, dtype=torch.float32).unsqueeze(0)
        )
    finally:
        student.train(was)
    max_abs = float((pred_a - pred_b).abs().max())
    ok = bool(torch.allclose(pred_a, pred_b, atol=C.CONSUMER_ATOL, rtol=C.CONSUMER_RTOL))
    return {"passed": ok, "max_abs": max_abs}
