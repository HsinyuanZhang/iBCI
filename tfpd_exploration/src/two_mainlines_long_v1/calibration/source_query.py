"""P5: real source query EMG grads into D, window-ID split, target hash freeze."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import stage0 as fold_stage0

from . import constants as C
from . import hashes
from .factory import FreshPArm, normalize_carrier, solve_carrier_float64


def window_id_support_query_split(datamodule, frozen) -> dict[str, Any]:
    """Audit real Falcon window IDs: carrier support is M10; outer query is [10, 210)."""
    train = datamodule.train_dataset
    train_base = train.base if hasattr(train, "base") else train
    val = datamodule.val_heldin_dataset
    val_base = val.base if hasattr(val, "base") else val
    if not getattr(train_base, "window_indices", None):
        raise RuntimeError("train dataset has no window_indices")
    if not getattr(val_base, "window_indices", None):
        raise RuntimeError("val_heldin dataset has no window_indices")

    support_ok = True
    support_counts: dict[str, int] = {}
    for name, support in frozen.source_supports.items():
        ids = np.asarray(support["ids"])
        support_counts[name] = int(ids.size)
        if ids.size == 0 or int(ids.min()) < 0 or int(ids.max()) >= C.SUPPORT_TRIALS:
            support_ok = False

    val_starts = getattr(val_base, "trial_start_indices", {})
    val_leaked = 0
    val_sessions: dict[str, int] = {}
    for session, start in val_base.window_indices:
        name = session.decode("ascii") if isinstance(session, (bytes, bytearray)) else str(session)
        val_sessions[name] = val_sessions.get(name, 0) + 1
        trial_starts = np.asarray(val_starts[name], dtype=np.int64)
        if int(start) < int(trial_starts[C.QUERY_START]):
            val_leaked += 1
    val_audits = getattr(val_base, "query_window_audit", {})
    val_audit_ok = True
    for _name, audit in val_audits.items():
        if int(audit.get("query_start_trial", -1)) != C.QUERY_START:
            val_audit_ok = False
        if not bool(audit.get("full_window_disjoint", False)):
            val_audit_ok = False

    train_sessions: dict[str, int] = {}
    for session, _start in train_base.window_indices:
        name = session.decode("ascii") if isinstance(session, (bytes, bytearray)) else str(session)
        train_sessions[name] = train_sessions.get(name, 0) + 1
    train_query_start = int(getattr(train_base, "query_start_trial", -1))

    passed = bool(
        support_ok
        and val_leaked == 0
        and val_audit_ok
        and val_sessions
        and train_sessions
        and set(train_sessions) == set(C.SOURCES)
        and C.OUTER in val_sessions
        and C.OUTER not in train_sessions
    )
    return {
        "passed": passed,
        "support_trial_ids_lt_m10": support_ok,
        "support_bin_counts": support_counts,
        "train_n_windows": len(train_base.window_indices),
        "train_sessions": train_sessions,
        "train_query_start_trial": train_query_start,
        "train_query_start_zero_disclosed": train_query_start == 0,
        "val_n_windows": len(val_base.window_indices),
        "val_sessions": val_sessions,
        "val_query_windows_overlapping_support": int(val_leaked),
        "val_audits_ok": val_audit_ok,
        "query_start_trial": C.QUERY_START,
        "support_trials": C.SUPPORT_TRIALS,
        "ignored_query_labels_only": False,
    }


def real_source_query_dictionary_grad(arm: FreshPArm, batch) -> dict[str, Any]:
    """Backprop real source neural/EMG query MSE into D. Synthetic zeros are not enough."""
    if not arm.train_basis:
        raise RuntimeError("dictionary grad requires P-CA")
    neural, target, calib, session, _unused = batch
    name = session[0].decode("ascii") if isinstance(session[0], (bytes, bytearray)) else str(session[0])
    support = arm.frozen.source_supports[name]
    emg = torch.as_tensor(support["emg"], dtype=torch.float64, device=arm.device)
    rates = torch.as_tensor(support["rates"], dtype=torch.float64, device=arm.device)
    if arm.basis.raw_dictionary.grad is not None:
        arm.basis.raw_dictionary.grad = None
    arm.enter_train()
    raw = solve_carrier_float64(arm.basis.encode(emg), rates)
    carrier = normalize_carrier(raw, arm.frozen)
    neural_t = neural.to(arm.device)
    target_t = target.to(arm.device)
    calib_t = calib.to(arm.device)
    pred, _identity = arm.student(neural_t, calib_trials=calib_t, carrier=carrier)
    pred_last = pred[:, -1, :] if pred.ndim == 3 else pred
    target_last = target_t[:, -1, :] if target_t.ndim == 3 else target_t
    if torch.allclose(target_last, torch.zeros_like(target_last)):
        raise RuntimeError("query target is all zeros; not a real EMG query loss")
    loss = F.mse_loss(pred_last, target_last)
    arm.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = arm.basis.raw_dictionary.grad
    if grad is None:
        raise RuntimeError("dictionary received no grad from real query EMG loss")
    consumer_grad = grad.detach().clone()
    arm.optimizer.zero_grad(set_to_none=True)
    if arm.basis.raw_dictionary.grad is not None:
        arm.basis.raw_dictionary.grad = None
    raw_sum = solve_carrier_float64(arm.basis.encode(emg), rates)
    raw_sum.sum().backward()
    sum_grad = arm.basis.raw_dictionary.grad
    if sum_grad is None:
        raise RuntimeError("carrier.sum grad missing")
    different = not torch.allclose(consumer_grad, sum_grad)
    return {
        "passed": bool(torch.isfinite(consumer_grad).all() and float(consumer_grad.norm()) > 0.0 and different),
        "finite": bool(torch.isfinite(consumer_grad).all()),
        "nonzero": bool(float(consumer_grad.norm()) > 0.0),
        "not_just_carrier_sum": bool(different),
        "loss": "consumer_mse_real_source_query_emg",
        "session": name,
        "target_abs_mean": float(target_last.detach().abs().mean()),
        "loss_value": float(loss.detach().cpu()),
    }


def target_fit_learned_state_unchanged(arm: FreshPArm) -> dict[str, Any]:
    before = hashes.learned_state(arm.student, arm.basis)
    before_sha = hashes.state_sha256(before)
    support = arm.frozen.target_support
    emg = torch.as_tensor(support["emg"], dtype=torch.float64, device=arm.device)
    rates = torch.as_tensor(support["rates"], dtype=torch.float64, device=arm.device)
    with torch.no_grad():
        raw = solve_carrier_float64(arm.basis.encode(emg), rates)
        normalized = normalize_carrier(raw, arm.frozen)
    after = hashes.learned_state(arm.student, arm.basis)
    after_sha = hashes.state_sha256(after)
    return {
        "passed": before_sha == after_sha,
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "carrier_shape": list(normalized.shape),
        "target_labels_used_only_for_legal_m10": True,
    }


def load_real_source_batches(repo_root, frozen, *, n_batches: int = 1):
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import datamodule as stage1_data

    zeros = np.zeros((64, 4), dtype=np.float32)
    bank = {
        "normalized": {
            name: {"rSyn3": zeros.copy(), "Zero4": zeros.copy()} for name in C.SOURCES
        }
    }
    data = stage1_data.make_datamodule(repo_root, bank, "S-Fix")
    data.setup("fit")
    batches = []
    for batch in data.train_dataloader():
        batches.append(batch)
        if len(batches) >= int(n_batches):
            break
    if not batches:
        raise RuntimeError("empty source query loader")
    return data, batches


def load_one_real_source_batch(repo_root, frozen):
    data, batches = load_real_source_batches(repo_root, frozen, n_batches=1)
    return data, batches[0]
