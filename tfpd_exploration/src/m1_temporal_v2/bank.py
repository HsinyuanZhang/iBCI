"""Fit and load the named rSyn3-refit-v1 source bank. Never refit at train time."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.carrier_bank import _encode_session
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan as parent_plan
from tfpd_exploration.src.m1_emg_syn3_fcm_v1.syn3 import SourceBasis

from . import plan


def _lock_numeric_env() -> dict[str, str]:
    locked = {
        "OMP_NUM_THREADS": str(plan.OMP_THREADS),
        "MKL_NUM_THREADS": str(plan.OMP_THREADS),
        "OPENBLAS_NUM_THREADS": str(plan.OMP_THREADS),
        "NUMEXPR_NUM_THREADS": str(plan.OMP_THREADS),
    }
    for key, value in locked.items():
        os.environ[key] = value
    return locked


def _array_digest(array: np.ndarray) -> str:
    return rsyn3.array_digest(np.ascontiguousarray(array))


def _source_paths() -> dict[str, Any]:
    """Resolve only the three fold-0 source files. Target is not hashed or opened."""
    root = parent_data.repo_root()
    paths: dict[str, Any] = {}
    for name in plan.SOURCE_SESSIONS:
        if name == plan.TARGET_SESSION:
            raise RuntimeError("target listed as source")
        relative = parent_plan.SOURCE_RELATIVE[name]
        path = parent_data.require_source_path(root / relative)
        digest = parent_data.file_sha256(path)
        if digest != parent_plan.SOURCE_FILE_SHA256[name]:
            raise RuntimeError(f"source sha drift {name}")
        paths[name] = path
    return paths


def _reconstruct_basis(blob: Any) -> SourceBasis:
    return SourceBasis(
        kind="nnmf",
        scale=np.asarray(blob["scale"], dtype=np.float64),
        dictionary=np.asarray(blob["d0"], dtype=np.float64),
        activations=np.asarray(blob["activations"], dtype=np.float64),
        order=tuple(int(x) for x in np.asarray(blob["nmf_order"]).tolist()),
        reconstruction_digest=str(blob["reconstruction_digest"][0]),
        library={"estimator": "sklearn.decomposition.NMF", "sealed": True},
        extra={"n_iter": int(blob["nmf_n_iter"][0])},
    )


def seal_source_bank() -> dict[str, Any]:
    """Fit NMF once on fold0 sources 26/27/28. Does not open the outer target."""
    env = _lock_numeric_env()
    fold_plan.verify_bound_documents(plan.REPO_ROOT)
    rectify.assert_frozen_law()
    if plan.TARGET_SESSION in plan.SOURCE_SESSIONS:
        raise RuntimeError("target listed as source")
    paths = _source_paths()
    sources = {name: fold_data.load_fold_session(paths[name], role="source") for name in plan.SOURCE_SESSIONS}
    source_emg = np.concatenate(
        [rectify.relu_nonnegative_projection(record.emg) for record in sources.values()],
        axis=0,
    )
    nmf = rsyn3.fit_source_nmf(source_emg)
    raw = {name: _encode_session(record, nmf) for name, record in sources.items()}
    norm_mean, norm_scale = rsyn3.source_normalizer(list(raw.values()))
    normalized = {name: rsyn3.normalize_carriers(carrier, norm_mean, norm_scale) for name, carrier in raw.items()}
    plan.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": np.asarray([plan.SCHEMA]),
        "revision": np.asarray([plan.REVISION]),
        "carrier_name": np.asarray([plan.CARRIER_NAME]),
        "d0": np.ascontiguousarray(nmf.dictionary, dtype=np.float64),
        "scale": np.ascontiguousarray(nmf.scale, dtype=np.float64),
        "activations": np.ascontiguousarray(nmf.activations, dtype=np.float64),
        "normalizer_mean": np.ascontiguousarray(norm_mean, dtype=np.float64),
        "normalizer_scale": np.ascontiguousarray(norm_scale, dtype=np.float64),
        "source_sessions": np.asarray(plan.SOURCE_SESSIONS),
        "nmf_order": np.asarray(nmf.order, dtype=np.int64),
        "nmf_n_iter": np.asarray([int(nmf.extra["n_iter"])], dtype=np.int64),
        "reconstruction_digest": np.asarray([nmf.reconstruction_digest]),
    }
    for name in plan.SOURCE_SESSIONS:
        payload[f"raw/{name}"] = np.ascontiguousarray(raw[name], dtype=np.float64)
        payload[f"normalized/{name}"] = np.ascontiguousarray(normalized[name], dtype=np.float64)
        payload[f"support_emg_trial_ids/{name}"] = np.ascontiguousarray(sources[name].emg_trial_ids)
        payload[f"support_rate_trial_ids/{name}"] = np.ascontiguousarray(sources[name].rate_trial_ids)
        payload[f"unit_order/{name}"] = np.arange(int(raw[name].shape[0]), dtype=np.int64)
        payload[f"channel_names/{name}"] = np.asarray(sources[name].channel_names)
    np.savez_compressed(plan.BANK_NPZ, **payload)
    digests = {
        "d0": _array_digest(payload["d0"]),
        "scale": _array_digest(payload["scale"]),
        "activations": _array_digest(payload["activations"]),
        "normalizer_mean": _array_digest(payload["normalizer_mean"]),
        "normalizer_scale": _array_digest(payload["normalizer_scale"]),
        "raw": {name: _array_digest(payload[f"raw/{name}"]) for name in plan.SOURCE_SESSIONS},
        "normalized": {name: _array_digest(payload[f"normalized/{name}"]) for name in plan.SOURCE_SESSIONS},
        "npz": hashlib.sha256(plan.BANK_NPZ.read_bytes()).hexdigest(),
    }
    if digests["d0"] == plan.OLD_D0_DIGEST:
        raise RuntimeError("refit D0 collided with old sealed digest; do not treat as P repair")
    receipt = {
        "schema": plan.SCHEMA,
        "revision": plan.REVISION,
        "carrier_name": plan.CARRIER_NAME,
        "not_old_stage0": True,
        "old_d0_digest": plan.OLD_D0_DIGEST,
        "old_target_m10_digest": plan.OLD_TARGET_M10_DIGEST,
        "source_sessions": list(plan.SOURCE_SESSIONS),
        "target_session_excluded": plan.TARGET_SESSION,
        "target_file_opened": False,
        "target_loaded": False,
        "query_values_read": False,
        "sklearn": dict(nmf.library),
        "nmf_law": "m1_emg_syn3_fcm_v1.NNMF_LAW",
        "seed": plan.SEED,
        "omp_threads": plan.OMP_THREADS,
        "env": env,
        "support_trials": plan.SUPPORT_TRIALS,
        "nmf_n_iter": int(nmf.extra["n_iter"]),
        "reconstruction_digest": nmf.reconstruction_digest,
        "s_fix_path": str(plan.S_FIX_PATH),
        "s_fix_sha256": plan.S_FIX_SHA256,
        "s_fix_decoder_weights_copied": False,
        "digests": digests,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    plan.BANK_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def load_source_bank() -> dict[str, Any]:
    if not plan.BANK_NPZ.is_file() or not plan.BANK_RECEIPT.is_file():
        raise FileNotFoundError("rSyn3-refit-v1 bank is not sealed")
    receipt = json.loads(plan.BANK_RECEIPT.read_text(encoding="utf-8"))
    blob = np.load(plan.BANK_NPZ, allow_pickle=False)
    if str(receipt.get("schema")) != plan.SCHEMA:
        raise RuntimeError(f"bank schema {receipt.get('schema')}")
    if str(receipt.get("revision")) != plan.REVISION:
        raise RuntimeError("bank revision drift")
    if receipt.get("target_loaded") is True or receipt.get("query_values_read") is True:
        raise RuntimeError("sealed bank claims target/query read")
    actual = hashlib.sha256(plan.BANK_NPZ.read_bytes()).hexdigest()
    if actual != receipt["digests"]["npz"]:
        raise RuntimeError("bank bytes drifted vs receipt")
    if receipt["digests"]["d0"] == plan.OLD_D0_DIGEST:
        raise RuntimeError("bank claims old Stage-0 D0; v2 must be a new named bank")
    normalized = {name: np.asarray(blob[f"normalized/{name}"], dtype=np.float32) for name in plan.SOURCE_SESSIONS}
    return {
        "schema": plan.SCHEMA,
        "revision": plan.REVISION,
        "carrier_name": plan.CARRIER_NAME,
        "receipt": receipt,
        "normalized": normalized,
        "d0": np.asarray(blob["d0"], dtype=np.float64),
        "scale": np.asarray(blob["scale"], dtype=np.float64),
        "normalizer_mean": np.asarray(blob["normalizer_mean"], dtype=np.float64),
        "normalizer_scale": np.asarray(blob["normalizer_scale"], dtype=np.float64),
        "source_sessions": list(plan.SOURCE_SESSIONS),
        "npz_sha256": actual,
    }


def carrier_for_session(bank: dict[str, Any], session: str) -> np.ndarray:
    if session not in bank["normalized"]:
        raise KeyError(f"{session} is not a sealed source session")
    return np.ascontiguousarray(bank["normalized"][session], dtype=np.float32)


def encode_legal_target_m10(bank: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build target M10 from sealed D0 + source μ/σ. Support only; not used in train."""
    loaded = bank or load_source_bank()
    blob = np.load(plan.BANK_NPZ, allow_pickle=False)
    basis = _reconstruct_basis(blob)
    root = parent_data.repo_root()
    path = parent_data.require_source_path(root / parent_plan.SOURCE_RELATIVE[plan.TARGET_SESSION])
    record = fold_data.load_fold_session(path, role="target")
    raw = _encode_session(record, basis)
    normalized = rsyn3.normalize_carriers(raw, loaded["normalizer_mean"], loaded["normalizer_scale"])
    return {
        "session": plan.TARGET_SESSION,
        "role": "target_m10_support_only",
        "query_values_read": False,
        "used_in_train": False,
        "used_in_normalizer": False,
        "raw": np.ascontiguousarray(raw, dtype=np.float64),
        "normalized": np.ascontiguousarray(normalized, dtype=np.float32),
        "raw_digest": _array_digest(raw),
        "source_bank_npz": loaded["npz_sha256"],
    }
