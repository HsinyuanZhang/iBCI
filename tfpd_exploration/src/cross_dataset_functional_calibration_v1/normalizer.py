"""SOURCE_INITIAL_DICTIONARY_FROZEN_NORMALIZER_V1.

Rematerialize source-pooled μ0,σ0 once from source-frozen D0 + RMS + M10
support. P-FIX and P-CA share the same bytes. Never update per step/epoch
and never refit on target.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import stage0 as fold_stage0
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as bank

from . import plan


class NormalizerError(RuntimeError):
    """Fail closed for the frozen source normalizer."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NormalizerError(message)


def _code_sha(repo_root: Path) -> str:
    relatives = (
        plan.M1_SYN3_OPERATOR_RELATIVE,
        plan.M1_CARRIER_BANK_RELATIVE,
        "tfpd_exploration/src/cross_dataset_functional_calibration_v1/normalizer.py",
        "tfpd_exploration/src/cross_dataset_functional_calibration_v1/basis.py",
        "tfpd_exploration/src/cross_dataset_functional_calibration_v1/carrier_solver.py",
    )
    digest = plan.sha256_bytes(b"".join((Path(repo_root) / relative).read_bytes() for relative in relatives))
    return digest


def _input_sha(
    *,
    source_path_shas: dict[str, str],
    d0: np.ndarray,
    scale: np.ndarray,
    raw: dict[str, np.ndarray],
) -> str:
    payload = plan.canonical_json_bytes(
        {
            "source_path_shas": source_path_shas,
            "d0": bank.array_digest(d0),
            "scale": bank.array_digest(scale),
            "raw": {name: bank.array_digest(array) for name, array in sorted(raw.items())},
        }
    )
    return plan.sha256_bytes(payload)


def _encode_session(record, basis) -> np.ndarray:
    emg_b, rates_b, ids_b = fold_stage0._mask_budget(record, plan.M1_SUPPORT_TRIALS)
    rsyn3.require_support_bins(ids_b, budget=plan.M1_SUPPORT_TRIALS)
    scores = rsyn3.project_basis(emg_b, basis)
    weights, intercepts = rsyn3.fit_all_units(scores, rates_b)
    return rsyn3.carrier_from_encoding(weights, intercepts)


def _support_view(record) -> dict[str, np.ndarray]:
    emg_b, rates_b, ids_b = fold_stage0._mask_budget(record, plan.M1_SUPPORT_TRIALS)
    return {"emg": np.asarray(emg_b, dtype=np.float64), "rates": np.asarray(rates_b, dtype=np.float64), "ids": ids_b}


@dataclass
class FrozenSourceNormalizer:
    name: str
    mu0: np.ndarray
    sigma0: np.ndarray
    d0: np.ndarray
    scale: np.ndarray
    d0_positive_count: int
    d0_zero_count: int
    relu_lock: bool
    updated_per_step: bool
    refit_on_target: bool
    parent_carrier_parity_passed: bool
    parent_max_abs_err: float
    array_sha256: str
    input_sha256: str
    code_sha256: str
    source_raw: dict[str, np.ndarray]
    source_supports: dict[str, dict[str, np.ndarray]]
    target_support: dict[str, np.ndarray]
    parent_normalized: dict[str, np.ndarray]
    source_sessions: tuple[str, ...]
    d0_digest: str
    sealed_d0_digest: str
    d0_digest_matches_sealed: bool

    def shared_mu0_for(self, _arm: str) -> np.ndarray:
        return self.mu0

    def shared_sigma0_for(self, _arm: str) -> np.ndarray:
        return self.sigma0

    def transform(self, raw_carrier: np.ndarray) -> np.ndarray:
        return rsyn3.normalize_carriers(raw_carrier, self.mu0, self.sigma0)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "source_initial_dictionary_frozen_normalizer_v1",
            "name": self.name,
            "mu0_sha256": bank.array_digest(self.mu0),
            "sigma0_sha256": bank.array_digest(self.sigma0),
            "array_sha256": self.array_sha256,
            "input_sha256": self.input_sha256,
            "code_sha256": self.code_sha256,
            "d0_sha256": bank.array_digest(self.d0),
            "scale_sha256": bank.array_digest(self.scale),
            "d0_positive_count": self.d0_positive_count,
            "d0_zero_count": self.d0_zero_count,
            "relu_lock": self.relu_lock,
            "softplus_used": False,
            "updated_per_step": self.updated_per_step,
            "refit_on_target": self.refit_on_target,
            "parent_carrier_parity_passed": self.parent_carrier_parity_passed,
            "parent_max_abs_err": self.parent_max_abs_err,
            "carrier_atol": plan.P_CARRIER_ATOL,
            "carrier_rtol": plan.P_CARRIER_RTOL,
            "source_sessions": list(self.source_sessions),
            "p_fix_pca_share_mu0_sigma0": True,
            "d0_digest": self.d0_digest,
            "sealed_d0_digest": self.sealed_d0_digest,
            "d0_digest_matches_sealed": self.d0_digest_matches_sealed,
            "scale_digest": bank.array_digest(self.scale),
        }


_CACHE: FrozenSourceNormalizer | None = None


def materialize(repo_root: Path) -> FrozenSourceNormalizer:
    """Compute μ0,σ0 once. Subsequent calls return the same object."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    repo_root = Path(repo_root)
    # Do not call carrier_bank.build_fold0_carrier_bank: it refits NMF then
    # fail-closes on a sealed *target* digest. Source μ0,σ0 are bound to the
    # sealed fold0 receipts instead.
    receipts_path = repo_root / fold_plan.STAGE0_ROOT_RELATIVE / "fold_receipts.json"
    _require(receipts_path.is_file(), "fold-local Stage0 receipts missing")
    import json

    sealed = json.loads(receipts_path.read_text(encoding="utf-8"))["0"]
    parent_mean = np.asarray(sealed["normalizer_mean"], dtype=np.float64)
    parent_scale = np.asarray(sealed["normalizer_scale"], dtype=np.float64)
    sealed_d0_digest = str(sealed["nmf"]["dictionary_digest"])
    sealed_scale_digest = str(sealed["nmf"]["scale_digest"])
    loaded = fold_data.load_fold_scope(fold=0)
    sources = loaded["sources"]
    _require(tuple(sources) == plan.M1_FOLD0_SOURCES, "source session drift")
    source_emg = np.concatenate(
        [rectify.relu_nonnegative_projection(record.emg) for record in sources.values()],
        axis=0,
    )
    nmf = rsyn3.fit_source_nmf(source_emg)
    source_raw = {name: _encode_session(record, nmf) for name, record in sources.items()}
    mu0, sigma0 = rsyn3.source_normalizer(list(source_raw.values()))
    max_err = max(float(np.max(np.abs(mu0 - parent_mean))), float(np.max(np.abs(sigma0 - parent_scale))))
    passed = bool(
        np.allclose(mu0, parent_mean, atol=plan.P_CARRIER_ATOL, rtol=plan.P_CARRIER_RTOL)
        and np.allclose(sigma0, parent_scale, atol=plan.P_CARRIER_ATOL, rtol=plan.P_CARRIER_RTOL)
    )
    if not passed:
        raise NormalizerError(
            "FAIL CLOSED: rematerialized values substantially change parent normalization; "
            f"max_abs_err={max_err}"
        )
    _require(rsyn3.array_digest(nmf.scale) == sealed_scale_digest, "source RMS scale digest drifted")
    parent_normalized = {name: rsyn3.normalize_carriers(raw, parent_mean, parent_scale) for name, raw in source_raw.items()}
    d0 = np.asarray(nmf.dictionary, dtype=np.float64)
    scale = np.asarray(nmf.scale, dtype=np.float64)
    _require(np.isfinite(scale).all() and bool(np.all(scale >= plan.M1_SCALE_FLOOR)), "bank scale floor")
    source_path_shas = {name: record.path_sha256 for name, record in sources.items()}
    array_sha = plan.sha256_bytes(
        plan.canonical_json_bytes({"mu0": bank.array_digest(mu0), "sigma0": bank.array_digest(sigma0)})
    )
    frozen = FrozenSourceNormalizer(
        name=plan.P_NORMALIZER_NAME,
        mu0=mu0,
        sigma0=sigma0,
        d0=d0,
        scale=scale,
        d0_positive_count=int(np.sum(d0 > 0.0)),
        d0_zero_count=int(np.sum(d0 == 0.0)),
        relu_lock=True,
        updated_per_step=False,
        refit_on_target=False,
        parent_carrier_parity_passed=True,
        parent_max_abs_err=max_err,
        array_sha256=array_sha,
        input_sha256=_input_sha(
            source_path_shas=source_path_shas, d0=d0, scale=scale, raw=source_raw
        ),
        code_sha256=_code_sha(repo_root),
        source_raw=source_raw,
        source_supports={name: _support_view(record) for name, record in sources.items()},
        target_support=_support_view(loaded["target"]),
        parent_normalized=parent_normalized,
        source_sessions=plan.M1_FOLD0_SOURCES,
        d0_digest=rsyn3.array_digest(d0),
        sealed_d0_digest=sealed_d0_digest,
        d0_digest_matches_sealed=rsyn3.array_digest(d0) == sealed_d0_digest,
    )
    _CACHE = frozen
    return frozen
