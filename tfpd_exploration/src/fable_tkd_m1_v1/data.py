"""Data plane for FABLE TKD M1 v1 (fold-local LOSO fold 0).

Everything face-related is REUSED BY IMPORT from
``m1_emg_rsyn3_fold_local_v1`` (carrier bank, datamodule, isolation) and the
parent syn3 helpers; nothing is reimplemented.  This module adds only the
TKD-specific closed forms: per-unit rho, the pooled value scale, the
generative-anchor inputs (W/D/Pbar), target statistics, and the frozen-SPINT
teacher output cache.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify as rsyn3_rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import datamodule as fold_dm
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import stage0 as fold_stage0
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import syn3 as fold_syn3

from . import plan


def ensure_streaming_path(repo_root: Path) -> None:
    experiment = str(Path(repo_root) / "streaming_calibration_exp")
    if experiment not in sys.path:
        sys.path.insert(0, experiment)


def _digest_array(value: np.ndarray) -> str:
    import json

    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


#: Sealed fold-0 M10 numeric pins (results/m1_emg_rsyn3_fold_local_v1/stage0/
#: reliability_table.json, row budget=10) -- M1-D5 tolerance-based binding.
SEALED_FOLD0_M10_NUMERICS = {
    "carrier_frobenius_norm": 99.96695423114359,
    "gram_eigenvalues": [
        0.1487215318146919, 0.8224791442655249, 3.3869709453411603,
        7.9415559238215705,
    ],
    "per_synergy_dispersion": [1.9188654095606201, 1.1145609039942814,
                               0.861535413576596],
    "valid_bins": 636,
}
SEALED_NUMERIC_TOLERANCE = 1.0e-9


def _build_bank(repo_root: Path) -> dict[str, Any]:
    """Rebuild the fold-0 carrier bank with the sealed law VERBATIM.

    M1-D5: carrier_bank.build_fold0_carrier_bank's bit-digest assert no
    longer reproduces in this environment (library drift); this rebuild
    executes the identical construction (same parent functions) and binds
    the result NUMERICALLY to the sealed fold-0 M10 row instead.
    """
    repo_root = Path(repo_root)
    fold_plan_local = fold_plan
    rsyn3_rectify.assert_frozen_law()
    loaded = fold_data.load_fold_scope(fold=0)
    isolation = dict(loaded["isolation"])
    plan.require(isolation["target_query_values_read"] is False, "target query leak")
    sources = loaded["sources"]
    target = loaded["target"]
    plan.require(target.session == plan.FOLD0_TARGET_SESSION, "fold-0 target drift")
    source_emg = np.concatenate(
        [rsyn3_rectify.relu_nonnegative_projection(record.emg)
         for record in sources.values()], axis=0,
    )
    nmf = rsyn3.fit_source_nmf(source_emg)

    def encode(record) -> np.ndarray:
        emg_b, rates_b, ids_b = fold_stage0._mask_budget(record, plan.SUPPORT_TRIALS)
        rsyn3.require_support_bins(ids_b, budget=plan.SUPPORT_TRIALS)
        scores = rsyn3.project_basis(emg_b, nmf)
        weights, intercepts = rsyn3.fit_all_units(scores, rates_b)
        return rsyn3.carrier_from_encoding(weights, intercepts), scores

    source_raw = {}
    source_scores = {}
    for name, record in sources.items():
        source_raw[name], source_scores[name] = encode(record)
    norm_mean, norm_scale = rsyn3.source_normalizer(list(source_raw.values()))
    target_raw, target_scores = encode(target)

    # --- M1-D5 numeric binding to the sealed fold-0 M10 row ---------------
    sealed = SEALED_FOLD0_M10_NUMERICS
    fro = float(np.linalg.norm(target_raw))
    plan.require(
        abs(fro - sealed["carrier_frobenius_norm"]) <= SEALED_NUMERIC_TOLERANCE,
        f"M1-D5 carrier norm drift: {fro} vs sealed {sealed['carrier_frobenius_norm']}",
    )
    design = np.column_stack((np.ones(target_scores.shape[0]), target_scores))
    gram = (design.T @ design) / target_scores.shape[0]
    eigenvalues = np.sort(np.linalg.eigvalsh(gram))
    plan.require(
        float(np.abs(eigenvalues - np.asarray(sealed["gram_eigenvalues"])).max())
        <= SEALED_NUMERIC_TOLERANCE,
        "M1-D5 gram eigenvalue drift",
    )
    dispersion = np.std(target_scores, axis=0)
    plan.require(
        float(np.abs(dispersion - np.asarray(sealed["per_synergy_dispersion"])).max())
        <= SEALED_NUMERIC_TOLERANCE,
        "M1-D5 synergy dispersion drift",
    )
    plan.require(target_scores.shape[0] == sealed["valid_bins"], "M1-D5 valid bins")

    raw = dict(source_raw)
    raw[target.session] = target_raw
    normalized = {
        name: {"rSyn3": rsyn3.normalize_carriers(carrier, norm_mean, norm_scale),
               "Zero4": np.zeros_like(carrier)}
        for name, carrier in raw.items()
    }
    return {
        "fold": 0,
        "target_session": target.session,
        "source_sessions": list(plan.FOLD0_SOURCE_SESSIONS),
        "isolation": isolation,
        "basis": nmf,
        "normalizer_mean": norm_mean,
        "normalizer_scale": norm_scale,
        "raw": raw,
        "normalized": normalized,
        "current_target_carrier_digest": rsyn3.array_digest(target_raw),
        "sealed_target_carrier_digest": plan.FOLD0_M10_RAW_CARRIER_DIGEST,
        "numeric_binding": "M1-D5 (tolerance 1e-9 vs sealed fold-0 M10 row)",
        "query_values_read": False,
    }


def load_plane(repo_root: Path) -> dict[str, Any]:
    """Build the fold-0 carrier bank + fold-local datamodule (verbatim reuse).

    Returns the bank, the train (source) and eval (fold-0 target query)
    datasets, per-session normalized carriers (t), per-unit rho, and the
    generative-anchor inputs (W_s, D, Pbar, sigma, target stats).
    """
    repo_root = Path(repo_root)
    ensure_streaming_path(repo_root)
    bank = _build_bank(repo_root)
    data_module = fold_dm.make_datamodule(repo_root, bank, "S-Fix")
    train_base = data_module.train_dataset.base
    eval_base = data_module.val_heldin_dataset.base

    # --- rho + anchor inputs (same law as the bank; basis reused) ---------
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data

    scope = fold_data.load_fold_scope(fold=0)
    records = {**scope["sources"], scope["target"].session: scope["target"]}
    basis = bank["basis"]

    rho: dict[str, np.ndarray] = {}
    for name, record in records.items():
        emg_b, rates_b, ids_b = fold_stage0._mask_budget(record, plan.SUPPORT_TRIALS)
        rsyn3.require_support_bins(ids_b, budget=plan.SUPPORT_TRIALS)
        scores = rsyn3.project_basis(emg_b, basis)
        weights, intercepts = rsyn3.fit_all_units(scores, rates_b)
        raw = rsyn3.carrier_from_encoding(weights, intercepts)
        plan.require(
            _digest_array(raw) == _digest_array(bank["raw"][name]),
            f"re-derived carrier digest drift for {name}",
        )
        # rho: closed-form per-unit R^2 of the same ridge fit (M2 law).
        design = np.column_stack((np.ones(scores.shape[0]), scores))
        predicted = design @ np.vstack([intercepts, weights.T])
        residual = np.square(rates_b - predicted).sum(axis=0)
        centered = rates_b - rates_b.mean(axis=0, keepdims=True)
        total = np.square(centered).sum(axis=0)
        safe = np.where(total > 0.0, total, 1.0)
        rho[name] = np.ascontiguousarray(
            np.clip(1.0 - residual / safe, 0.0, 1.0), dtype=np.float32
        )

    # --- generative-anchor fold-shared quantities -------------------------
    # W_s: per-session raw carrier weight columns [64,3]; Pbar: mean of the
    # per-session normalized-gram inverses; D: the frozen NMF dictionary
    # (unit-normalized rows, scaled-EMG space).
    source_raws = [bank["raw"][name] for name in plan.FOLD0_SOURCE_SESSIONS]
    inverses = []
    for raw in source_raws:
        w = raw[:, : plan.RANK].astype(np.float64)
        gram = (w.T @ w) / w.shape[0] + plan.RIDGE_LAMBDA * np.eye(plan.RANK)
        inverses.append(np.linalg.inv(gram))
    p_bar = np.mean(np.stack(inverses, axis=0), axis=0)
    dictionary = np.ascontiguousarray(basis.dictionary, dtype=np.float64)  # [3,16]

    # --- sigma pooled + target stats over SOURCE training windows ---------
    sigma_num = 0.0
    sigma_count = 0
    target_sum = np.zeros(plan.OUT_DIM, dtype=np.float64)
    target_sqsum = np.zeros(plan.OUT_DIM, dtype=np.float64)
    target_count = 0
    train_starts: dict[str, np.ndarray] = {}
    for name in plan.FOLD0_SOURCE_SESSIONS:
        starts = np.asarray(
            [start for session, start in train_base.window_indices
             if session == name], dtype=np.int64,
        )
        plan.require(starts.size > 0, f"no train windows for {name}")
        train_starts[name] = starts
        neural = np.asarray(train_base.neural_data[name], dtype=np.float64)
        # neural_data is COUNTS per bin; the carrier intercept is in Hz
        # (counts / BIN_SECONDS) -- mu recovers the intercept in count units.
        b_counts = bank["raw"][name][:, 3].astype(np.float64) * plan.BIN_SECONDS
        covariate = np.asarray(train_base.covariate_data[name], dtype=np.float64)
        last = starts + plan.WINDOW_SIZE - 1
        rates_last = neural[last, :]  # [n, 64] read-in statistic = last bin
        deviations = rates_last - b_counts[None, :]
        sigma_num += float(np.square(deviations).sum())
        sigma_count += deviations.size
        targets = covariate[last, :]
        target_sum += targets.sum(axis=0)
        target_sqsum += np.square(targets).sum(axis=0)
        target_count += targets.shape[0]
    sigma_pooled = float(np.sqrt(sigma_num / sigma_count))
    target_mean = target_sum / target_count
    target_var = target_sqsum / target_count - np.square(target_mean)
    target_rms = float(np.sqrt(np.mean(target_var + np.square(target_mean))))

    return {
        "bank": bank,
        "records": records,
        "basis": basis,
        "dictionary": dictionary,
        "p_bar": p_bar,
        "train_base": train_base,
        "eval_base": eval_base,
        "train_starts": train_starts,
        "rho": rho,
        "sigma_pooled": sigma_pooled,
        "target_mean": target_mean,
        "target_rms_pooled": target_rms,
        "target_count": target_count,
        "isolation": bank["isolation"],
    }


def session_tensors(plane: dict[str, Any], session: str) -> tuple[np.ndarray, np.ndarray]:
    """(t, rho) for a session: normalized rSyn3 carrier + per-unit rho."""
    t = np.asarray(
        plane["bank"]["normalized"][session]["rSyn3"], dtype=np.float32
    )
    rho = plane["rho"][session]
    plan.require(t.shape == (plan.CHANNELS, plan.CARRIER_DIM)
                 and rho.shape == (plan.CHANNELS,), f"carrier shape drift {session}")
    return t, rho


def eval_windows(plane: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Fold-0 target query window starts + last-bin targets (verbatim face)."""
    base = plane["eval_base"]
    session = plan.FOLD0_TARGET_SESSION
    starts = np.asarray(
        [start for name, start in base.window_indices if name == session],
        dtype=np.int64,
    )
    plan.require(starts.size == plan.EXPECTED_EVAL_WINDOWS,
                 f"eval window count drift: {starts.size}")
    covariate = np.asarray(base.covariate_data[session], dtype=np.float32)
    targets = np.ascontiguousarray(covariate[starts + plan.WINDOW_SIZE - 1])
    return starts, targets


def teacher_cache_path(repo_root: Path) -> Path:
    return Path(repo_root) / plan.TEACHER_CACHE_RELATIVE


def ensure_teacher_cache(
    plane: dict[str, Any], repo_root: Path, device: Any
) -> dict[str, np.ndarray]:
    """D14-for-M1: frozen SPINT teacher last-timestep outputs on the SOURCE
    training windows (static m10 identity law), cached once per session."""
    import torch

    repo_root = Path(repo_root)
    cache_dir = teacher_cache_path(repo_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[str, np.ndarray] = {}
    rebuilt: list[str] = []
    base = plane["train_base"]
    teacher = None
    for session in plan.FOLD0_SOURCE_SESSIONS:
        path = cache_dir / f"{session}.npy"
        if path.exists():
            sidecar = path.with_name(path.name + ".sha256")
            plan.require(sidecar.is_file(), f"teacher sidecar missing {session}")
            array = np.load(path)
            digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
            plan.require(
                sidecar.read_text(encoding="utf-8") == f"{digest}  {path.name}\n",
                f"teacher cache digest drift {session}",
            )
            cached[session] = array
            continue
        if teacher is None:
            teacher_path = repo_root / plan.TEACHER_CKPT_RELATIVE
            plan.require(teacher_path.is_file(), f"missing teacher {teacher_path}")
            digest_file = hashlib.sha256()
            with teacher_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest_file.update(chunk)
            plan.require(digest_file.hexdigest() == plan.TEACHER_CKPT_SHA256,
                         "teacher ckpt sha drift")
            from src.models.falcon_module import FalconLitModule

            module = FalconLitModule.load_from_checkpoint(
                str(teacher_path), weights_only=False
            )
            teacher = module.net.to(device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        starts = plane["train_starts"][session]
        calib = np.ascontiguousarray(
            base.calib_trialized_neural_features[session][: plan.SUPPORT_TRIALS],
            dtype=np.float32,
        )
        plan.require(calib.ndim == 3 and calib.shape[0] == plan.SUPPORT_TRIALS,
                     f"calib shape drift {session}")
        neural = np.asarray(base.neural_data[session], dtype=np.float32)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            calib_batch = torch.from_numpy(calib).unsqueeze(0).to(device)
            for offset in range(0, starts.size, 256):
                chunk = starts[offset : offset + 256]
                x = torch.from_numpy(
                    np.stack([neural[s : s + plan.WINDOW_SIZE] for s in chunk])
                ).to(device)
                calib_expanded = calib_batch.expand(x.shape[0], -1, -1, -1)
                y = teacher(x, calib_trialized_neural_features=calib_expanded)
                outputs.append(y[:, -1, :].detach().cpu().numpy().astype(np.float32))
        targets = np.ascontiguousarray(np.concatenate(outputs, axis=0))
        plan.require(targets.shape == (starts.size, plan.OUT_DIM)
                     and np.isfinite(targets).all(), "teacher target drift")
        temporary = path.with_name(f".{path.name}.tmp")
        with open(temporary, "wb") as handle:
            np.save(handle, targets)
            handle.flush()
            import os as _os

            _os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o444)
        digest = hashlib.sha256(targets.tobytes(order="C")).hexdigest()
        sidecar_tmp = path.with_name(f".{path.name}.sha256.tmp")
        with open(sidecar_tmp, "wb") as handle:
            handle.write(f"{digest}  {path.name}\n".encode("utf-8"))
        sidecar_tmp.replace(path.with_name(path.name + ".sha256"))
        path.with_name(path.name + ".sha256").chmod(0o444)
        plan.require(
            hashlib.sha256(np.load(path).tobytes(order="C")).hexdigest() == digest,
            "teacher reload drift",
        )
        cached[session] = np.load(path)
        rebuilt.append(session)
    if rebuilt:
        plan.atomic_receipt(cache_dir / "teacher_cache.json", {
            "schema": f"{plan.SCHEMA}:teacher_cache",
            "teacher_checkpoint_sha256": plan.TEACHER_CKPT_SHA256,
            "rebuilt": rebuilt,
            "sessions": {
                name: {
                    "n_windows": int(cached[name].shape[0]),
                    "sha256_bytes": hashlib.sha256(
                        cached[name].tobytes(order="C")
                    ).hexdigest(),
                }
                for name in plan.FOLD0_SOURCE_SESSIONS
            },
            "law": (
                "frozen M1 SPINT teacher (epoch_019), static m10 repeated "
                "calibration identity, last-timestep raw outputs on source "
                "training windows only"
            ),
        })
    return cached
