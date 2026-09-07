"""Disposable 20-warmup + 100 paired updates on GPU1. Refuses if P1–P5 failed."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer as sealed_normalizer

from . import constants as C
from . import loop
from . import source_query
from .factory import build_fresh_arm
from .pfix_cache import build_pfix_cache


class ProfileError(RuntimeError):
    """Fail closed for the disposable profile."""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_profile(gate_summary: dict, *, device: str | torch.device = "cuda:0") -> dict:
    if not gate_summary.get("all_pass"):
        raise ProfileError("P1–P5 must all pass before the disposable profile")
    repo = C.REPO_ROOT
    device = torch.device(device)
    frozen = sealed_normalizer.materialize(repo)
    data, batches = source_query.load_real_source_batches(repo, frozen, n_batches=32)
    if len(batches) < 1:
        raise ProfileError("empty source loader")
    sessions_seen: dict[str, int] = {}
    timers = {
        "support_read": 0.0,
        "scipy_nnls_cpu_gpu": 0.0,
        "active_set_solve": 0.0,
        "ridge": 0.0,
        "consumer_fwd_bwd": 0.0,
        "checkpoint": 0.0,
    }
    completed: dict[str, int] = {}
    losses: dict[str, list[float]] = {}
    step0_sha: dict[str, str] = {}
    step_end_sha: dict[str, str] = {}
    started = time.perf_counter()
    for arm_name, train_basis in (("P-FIX", False), ("P-CA", True)):
        arm = build_fresh_arm(repo, frozen, arm_name, device=device, seed=C.PROFILE_SEED)
        step0_sha[arm_name] = arm.consumer_sha256()
        cache = build_pfix_cache(arm) if not train_basis else None
        losses[arm_name] = []
        steps = 0
        total = C.PROFILE_WARMUP_STEPS + C.PROFILE_PAIRED_UPDATES
        while steps < total:
            batch = batches[steps % len(batches)]
            _sync(device)
            t0 = time.perf_counter()
            rec = loop.one_update(arm, batch, cache=cache, steps_per_warmup=C.PROFILE_WARMUP_STEPS)
            _sync(device)
            timers["consumer_fwd_bwd"] += time.perf_counter() - t0
            losses[arm_name].append(rec["loss"])
            sessions_seen[rec["session"]] = sessions_seen.get(rec["session"], 0) + 1
            steps += 1
        step_end_sha[arm_name] = arm.consumer_sha256()
        completed[arm_name] = steps
        del arm
        if device.type == "cuda":
            torch.cuda.empty_cache()
    receipt = {
        "schema": "two_mainlines_p_disposable_profile_v1",
        "status": "PROFILE_COMPLETE",
        "warmup_steps": C.PROFILE_WARMUP_STEPS,
        "paired_updates": C.PROFILE_PAIRED_UPDATES,
        "completed_steps": completed,
        "last_train_loss": {name: values[-1] if values else None for name, values in losses.items()},
        "sessions_seen": sessions_seen,
        "all_three_sources": all(name in sessions_seen for name in C.SOURCES),
        "step0_consumer_sha256": step0_sha,
        "step0_consumer_sha_equal": step0_sha.get("P-FIX") == step0_sha.get("P-CA"),
        "end_consumer_sha256": step_end_sha,
        "pfix_cache": {"sessions": list(C.SOURCES), "cached_fields": ["z", "raw", "carrier"]},
        "timers_s": timers,
        "elapsed_s": time.perf_counter() - started,
        "cuda_synchronized": device.type == "cuda",
        "auto_extend_to_12_epochs": False,
        "old_100step_not_packed": True,
    }
    out = C.OWNED_RESULT_ROOT / "profile" / "disposable_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (C.SLOT_ROOT / "p_disposable_profile.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    del data
    return receipt
