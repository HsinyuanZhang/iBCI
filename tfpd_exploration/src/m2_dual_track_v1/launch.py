"""Coordinator-owned formal training launch. Shared training.py stays a skeleton."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from . import champion, contracts, data, jobs, plan, sampler, training
from .calibration_memory import AQMEM
from .monitor import snapshot, write_snapshot

FULL_CKPT_SCHEMA = "m2_dual_track_v1_full_ckpt_v1"
SHUFFLED_RUN_TAG = "shuffled"
CONTINUATION_RUN_TAG = "shuffled_e13_24"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(payload: Mapping[str, Any] | None) -> None:
    if not payload:
        return
    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"])
    if payload.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda"])


def _arm_dest(root: Path, arm: str, seed: int, run_tag: str = "") -> Path:
    name = f"seed{seed}" if not run_tag else f"seed{seed}_{run_tag}"
    return root / "arms" / arm / name


def _bind_visible_device() -> tuple[torch.device, str, int]:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    plan.require(torch.cuda.is_available(), "CUDA unavailable")
    uuid = jobs.visible_gpu_uuid()
    index = jobs.visible_gpu_index()
    torch.set_num_threads(plan.TORCH_THREADS)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    return device, uuid, index


def shared_manifest_path(root: Path | None = None) -> Path:
    root = root or plan.active_run_root()
    return root / "sampler" / "shuffled_batch_manifest.json"


def ensure_shared_manifest(root: Path | None = None) -> dict[str, Any]:
    root = root or plan.active_run_root()
    path = shared_manifest_path(root)
    lengths = sampler.session_lengths_from_cache(root / "cache", "source_train")
    manifest = sampler.build_shuffled_manifest(
        lengths,
        batch_size=plan.EFFECTIVE_BATCH,
        epochs=plan.EPOCHS,
        sampler_seed=sampler.SAMPLER_SEED,
    )
    sampler.save_manifest(path, manifest)
    return sampler.load_manifest(path)


def shared_manifest_24_path(root: Path | None = None) -> Path:
    root = root or plan.active_run_root()
    return root / "sampler" / "shuffled_batch_manifest_24.json"


def ensure_shared_manifest_24(root: Path | None = None) -> dict[str, Any]:
    root = root or plan.active_run_root()
    parent = ensure_shared_manifest(root)
    path = shared_manifest_24_path(root)
    grown = sampler.extend_shuffled_manifest(parent, epochs=plan.EPOCHS_EXTENDED)
    if path.exists():
        existing = sampler.load_manifest(path)
        plan.require(existing["digest"] == grown["digest"], "24-epoch manifest digest drift")
        return existing
    sampler.save_manifest(path, grown)
    return sampler.load_manifest(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_full_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    opt = payload.get("optimizer") or {}
    rng = payload.get("rng") or {}
    n_state = 0
    if isinstance(opt, dict):
        n_state = len(opt.get("state") or {})
    return {
        "schema": payload.get("schema"),
        "epoch_one_based": payload.get("epoch_one_based"),
        "global_step": payload.get("global_step"),
        "sampler": payload.get("sampler"),
        "has_optimizer": bool(opt),
        "has_rng": bool(rng.get("torch") is not None),
        "n_optimizer_states": int(n_state),
        "state_keys": sorted(payload.get("state_dict", {}).keys()),
        "bytes": int(Path(path).stat().st_size),
        "sha256": file_sha256(path),
    }


def assert_optimizer_named_correspondence(model: Any, optimizer: torch.optim.Optimizer, payload: Mapping[str, Any]) -> None:
    names = list(model.trainable_parameters())
    saved_names = payload.get("param_names")
    if saved_names:
        plan.require(list(saved_names) == names, "optimizer parameter-name order drifted")
    saved = payload.get("optimizer") or {}
    groups = optimizer.param_groups
    saved_groups = saved.get("param_groups") or []
    plan.require(len(groups) == len(saved_groups), "optimizer group count mismatch")
    for live, frozen in zip(groups, saved_groups):
        plan.require(len(live["params"]) == len(frozen["params"]), "optimizer group width mismatch")
    live_params = {param for group in groups for param in group["params"]}
    plan.require(set(optimizer.state).issubset(live_params), "optimizer state params not in live groups")
    if saved.get("state"):
        plan.require(len(saved["state"]) == len(live_params), "optimizer state count mismatch")


def _save_full_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extras: dict[str, Any],
    global_step: int,
    manifest_digest: str | None,
) -> None:
    torch.save(
        {
            "schema": FULL_CKPT_SCHEMA,
            "epoch_one_based": epoch,
            "global_step": global_step,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _capture_rng(),
            "sampler": {"manifest_digest": manifest_digest, "completed_epoch": epoch},
            "param_names": list(model.trainable_parameters()),
            "extras": extras,
        },
        path,
    )


def _load_banks(surface: str, device: torch.device) -> dict[str, contracts.SessionBank]:
    return {
        session: data.load_session_bank(surface, session, device=device)
        for session in plan.HELDIN_SESSIONS
    }


def _score_minival(candidate: Any, banks: dict[str, contracts.SessionBank], device: torch.device) -> dict[str, Any]:
    per_session: dict[str, float] = {}
    losses: list[float] = []
    candidate.eval()
    for session, bank in banks.items():
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for batch in data.iter_session_batches(
            bank, batch_size=plan.EFFECTIVE_BATCH, device=device, target_space=plan.SCORING_TARGET_SPACE
        ):
            with torch.inference_mode():
                raw = candidate.forward_last(batch.X, batch.bank, batch.unit_mask)
            native = raw.detach().cpu().numpy() / plan.BEHAVIOR_SCALE
            preds.append(np.ascontiguousarray(native, dtype=np.float32))
            targets.append(batch.last_target.detach().cpu().numpy())
        target = np.concatenate(targets, axis=0)
        pred = np.concatenate(preds, axis=0)
        per_session[session] = contracts.variance_weighted_r2(target, pred)
        losses.append(float(np.mean(np.square(target - pred))))
    summary = contracts.summarize_sessions(per_session)
    return {
        "per_session_r2": summary["per_session_r2"],
        "equal_session_mean": summary["equal_session_mean"],
        "native_mse": float(np.mean(losses)),
    }


def _build_a(device: torch.device, seed: int) -> AQMEM:
    _set_seeds(seed)
    frozen = champion.load_frozen_champion(device=device)
    model = AQMEM(frozen.student).to(device)
    model.train()
    return model


def _save_epoch(dest: Path, model: AQMEM, epoch: int, extras: dict[str, Any]) -> None:
    allow = {name: param.detach().cpu() for name, param in model.trainable_parameters().items()}
    torch.save(
        {
            "epoch_one_based": epoch,
            "state_dict": allow,
            "extras": extras,
        },
        dest / f"epoch_{epoch:03d}.pt",
    )


def run_a_qmem(*, seed: int = plan.SEED_PRIMARY, max_epochs: int = plan.EPOCHS) -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    plan.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "A launch requires CUDA_VISIBLE_DEVICES=0")
    plan.require(torch.cuda.is_available(), "CUDA unavailable for A")
    torch.set_num_threads(plan.TORCH_THREADS)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    run_id = f"A-QMEM_s{seed}"
    root = plan.active_run_root()
    dest = root / "arms" / "A-QMEM" / f"seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    jobs.acquire_lease(gpu_uuid=plan.GPU0_UUID, lessee="A", run_id=run_id, run_root=root)
    jobs.append_job(
        jobs.JobRecord(
            run_id=run_id,
            owner="A",
            arm="A-QMEM",
            status="TRAIN_STARTED",
            gpu_uuid=plan.GPU0_UUID,
            pid=os.getpid(),
            started_unix=time.time(),
            log_path=str(log_path),
            ckpt_path=str(dest),
        ),
        root,
    )
    started = time.monotonic()
    deadline = started + plan.JOB_GPU_HOUR_LIMIT * 3600.0

    train_banks = _load_banks("source_train", device)
    minival_banks = _load_banks("source_minival", device)
    updates_per_epoch = training.count_updates(train_banks)
    warmup_steps = max(1, int(round(plan.A_WARMUP_FRAC * updates_per_epoch * max_epochs)))

    def _throwaway() -> tuple[AQMEM, torch.optim.AdamW]:
        model = _build_a(device, seed=seed + 999)
        opt = training.build_optimizer(
            model.trainable_parameters().items(),
            lr=plan.A_LR,
            weight_decay=plan.A_WEIGHT_DECAY,
        )
        return model, opt

    first_batch = next(training.epoch_batches(train_banks, device=device))

    def _step(model: AQMEM, opt: torch.optim.AdamW) -> None:
        model.train()
        pred = model.forward_last(first_batch.X, first_batch.bank, first_batch.unit_mask)
        loss = nn.functional.mse_loss(pred, first_batch.last_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.A_GRAD_CLIP)
        opt.step()

    t0 = time.monotonic()
    preflight = training.isolated_preflight(_throwaway, _step)
    preflight["seconds"] = time.monotonic() - t0
    preflight["examples_per_s"] = (120 * plan.EFFECTIVE_BATCH) / max(preflight["seconds"], 1e-6)
    if torch.cuda.is_available():
        preflight["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated(device))
    _write_json(dest / "preflight.json", preflight)
    write_snapshot(root / "monitor" / f"{run_id}_preflight.json")

    model = _build_a(device, seed=seed)
    optimizer = training.build_optimizer(
        model.trainable_parameters().items(),
        lr=plan.A_LR,
        weight_decay=plan.A_WEIGHT_DECAY,
    )
    global_step = 0
    epoch_scores: dict[int, float] = {}
    metrics_path = dest / "metrics.jsonl"
    budget_hit = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        epoch_t0 = time.monotonic()
        for batch in training.epoch_batches(train_banks, device=device):
            if time.monotonic() >= deadline:
                budget_hit = True
                break
            lr = training.apply_warmup_lr(optimizer, plan.A_LR, global_step, warmup_steps)
            pred = model.forward_last(batch.X, batch.bank, batch.unit_mask)
            loss = nn.functional.mse_loss(pred, batch.last_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.A_GRAD_CLIP)
            optimizer.step()
            global_step += 1
            running += float(loss.detach().cpu())
            n_batches += 1
            if global_step % 50 == 0:
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "step",
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "lr": lr,
                        "unix": time.time(),
                    },
                )
        if n_batches == 0:
            break
        minival = _score_minival(model, minival_banks, device)
        epoch_scores[epoch] = float(minival["equal_session_mean"])
        extras = {
            "train_mse": running / n_batches,
            "minival": minival,
            "seconds": time.monotonic() - epoch_t0,
            "global_step": global_step,
        }
        _save_epoch(dest, model, epoch, extras)
        _append_jsonl(metrics_path, {"event": "epoch", "epoch": epoch, **extras, "unix": time.time()})
        write_snapshot(root / "monitor" / f"{run_id}_epoch{epoch:03d}.json")
        jobs.append_job(
            jobs.JobRecord(
                run_id=run_id,
                owner="A",
                arm="A-QMEM",
                status="EPOCH",
                gpu_uuid=plan.GPU0_UUID,
                pid=os.getpid(),
                note=f"epoch {epoch} minival={epoch_scores[epoch]:.6f}",
            ),
            root,
        )
        if budget_hit:
            break

    selected = training.select_checkpoint(epoch_scores, max_epoch=max(epoch_scores))
    summary = {
        "run_id": run_id,
        "arm": "A-QMEM",
        "seed": seed,
        "epochs_completed": sorted(epoch_scores),
        "selected_epoch": selected,
        "endpoint_epoch": max(epoch_scores),
        "epoch_minival": epoch_scores,
        "updates_per_epoch": updates_per_epoch,
        "warmup_steps": warmup_steps,
        "budget_hit": budget_hit,
        "elapsed_s": time.monotonic() - started,
        "preflight": preflight,
        "hardware": snapshot(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "ext4_scored": False,
        "note": "ext-4 scored only after selected+endpoint12; development evidence",
    }
    _write_json(dest / "summary.json", summary)
    jobs.append_job(
        jobs.JobRecord(
            run_id=run_id,
            owner="A",
            arm="A-QMEM",
            status="TRAIN_FINISHED" if not budget_hit else "TRAIN_BUDGET",
            gpu_uuid=plan.GPU0_UUID,
            pid=os.getpid(),
            ckpt_path=str(dest),
            extras={"selected_epoch": selected},
        ),
        root,
    )
    jobs.release_lease(gpu_uuid=plan.GPU0_UUID, lessee="A", run_root=root)
    print(json.dumps({"status": "TRAIN_FINISHED", "selected_epoch": selected, "budget_hit": budget_hit}, indent=2))
    return 0


def _load_a_checkpoint(model: AQMEM, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    allow = set(model.trainable_parameters())
    plan.require(not unexpected, f"unexpected A keys: {unexpected}")
    plan.require(allow.issubset(set(payload["state_dict"])), f"missing trained keys: {allow - set(payload['state_dict'])}")
    del missing
    return payload


def _score_ext4(candidate: Any, device: torch.device) -> dict[str, Any]:
    banks = {
        session: data.load_session_bank("ext4", session, device=device)
        for session in plan.EXT4_SESSIONS
    }
    per_session: dict[str, float] = {}
    rows: dict[str, Any] = {}
    candidate.eval()
    for session, bank in banks.items():
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for batch in data.iter_session_batches(
            bank, batch_size=plan.EFFECTIVE_BATCH, device=device, target_space=plan.SCORING_TARGET_SPACE
        ):
            with torch.inference_mode():
                raw = candidate.forward_last(batch.X, batch.bank, batch.unit_mask)
            preds.append(np.ascontiguousarray(raw.detach().cpu().numpy() / plan.BEHAVIOR_SCALE, dtype=np.float32))
            targets.append(batch.last_target.detach().cpu().numpy())
        target = np.concatenate(targets, axis=0)
        pred = np.concatenate(preds, axis=0)
        r2 = contracts.variance_weighted_r2(target, pred)
        per_session[session] = float(r2)
        rows[session] = {
            "r2": float(r2),
            "window_count": int(target.shape[0]),
            "prediction_digest": champion.array_sha256(pred),
        }
    summary = contracts.summarize_sessions(per_session)
    dates = contracts.date_equal_mean(per_session)
    return {
        "per_session": rows,
        "summary": summary,
        "date_sensitivity": dates,
        "R_session_equal_mean": summary["equal_session_mean"],
        "R_date_equal_mean": dates["equal_date_mean"],
        "development_evidence": True,
    }


def score_a_qmem(*, seed: int = plan.SEED_PRIMARY, device: str = "cuda:0") -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    device_t = torch.device(device)
    root = plan.active_run_root()
    dest = root / "arms" / "A-QMEM" / f"seed{seed}"
    summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
    selected = int(summary["selected_epoch"])
    endpoint = int(summary["endpoint_epoch"])
    ref = json.loads((root / "stage0" / "ref_clean.json").read_text(encoding="utf-8"))
    r_ref = float(ref["R_REF_clean"])
    frozen = champion.load_frozen_champion(device=device_t)
    reports = {}
    for label, epoch in (("selected", selected), ("endpoint12", endpoint)):
        if label == "endpoint12" and epoch == selected:
            reports[label] = {"duplicate_of": "selected", "epoch": epoch}
            continue
        model = AQMEM(frozen.student).to(device_t)
        _load_a_checkpoint(model, dest / f"epoch_{epoch:03d}.pt")
        report = _score_ext4(model, device_t)
        report["epoch"] = epoch
        report["delta_vs_ref"] = float(report["R_session_equal_mean"]) - r_ref
        reports[label] = report
    payload = {
        "arm": "A-QMEM",
        "seed": seed,
        "R_REF_clean": r_ref,
        "reports": reports,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(dest / "ext4_score.json", payload)
    jobs.append_job(
        jobs.JobRecord(run_id=f"A-QMEM_s{seed}", owner="A", arm="A-QMEM", status="EXT4_SCORED", extras=payload),
        root,
    )
    print(json.dumps({
        "selected": reports.get("selected", {}).get("R_session_equal_mean"),
        "delta": reports.get("selected", {}).get("delta_vs_ref"),
        "endpoint12": reports.get("endpoint12", {}).get("R_session_equal_mean"),
    }, indent=2))
    return 0


def load_full_checkpoint(
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    plan.require(payload.get("schema") == FULL_CKPT_SCHEMA, f"not a full ckpt: {path}")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True)
    plan.require(not missing and not unexpected, f"state_dict mismatch {missing} {unexpected}")
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
    assert_optimizer_named_correspondence(model, optimizer, payload)
    _restore_rng(payload.get("rng"))
    return payload


def run_b_decoder(
    *,
    arm: str,
    seed: int = plan.SEED_PRIMARY,
    max_epochs: int = plan.EPOCHS,
    run_tag: str = SHUFFLED_RUN_TAG,
    manifest_path: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> int:
    plan.require(bool(run_tag), "new B training requires a run_tag; will not overwrite the sequential record")
    device, gpu_uuid, gpu_index = _bind_visible_device()
    del gpu_index
    root = plan.active_run_root()
    dest = _arm_dest(root, arm, seed, run_tag)
    original = root / "arms" / arm / f"seed{seed}"
    parent_shuffled = root / "arms" / arm / f"seed{seed}_{SHUFFLED_RUN_TAG}"
    plan.require(dest.resolve() != original.resolve(), "refusing to write into the original sequential B run")
    plan.require(dest.resolve() != parent_shuffled.resolve(), "refusing to write into the frozen 12ep shuffled run")
    if dest.exists() and (dest / "summary.json").exists():
        raise plan.DualTrackError(f"revised B dest already finished: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    if resume_path:
        manifest = sampler.load_manifest(Path(manifest_path) if manifest_path else shared_manifest_24_path(root))
        plan.require(int(manifest["epochs"]) >= int(max_epochs), "resume manifest too short")
    else:
        manifest = sampler.load_manifest(Path(manifest_path) if manifest_path else shared_manifest_path(root))
    run_id = f"{arm}_s{seed}_{run_tag}"
    jobs.acquire_lease(gpu_uuid=gpu_uuid, lessee="B", run_id=run_id, run_root=root)
    jobs.append_job(
        jobs.JobRecord(
            run_id=run_id, owner="B", arm=arm, status="TRAIN_STARTED",
            gpu_uuid=gpu_uuid, pid=os.getpid(), started_unix=time.time(), ckpt_path=str(dest),
            extras={"run_tag": run_tag, "sampler_digest": manifest["digest"], "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"), "resume": str(resume_path) if resume_path else None},
        ),
        root,
    )
    started = time.monotonic()
    try:
        return _train_b_loop(
            arm=arm,
            seed=seed,
            max_epochs=max_epochs,
            run_tag=run_tag,
            manifest=manifest,
            manifest_path=manifest_path,
            device=device,
            gpu_uuid=gpu_uuid,
            root=root,
            dest=dest,
            run_id=run_id,
            started=started,
            resume_path=Path(resume_path) if resume_path else None,
        )
    except Exception:
        jobs.append_job(jobs.JobRecord(run_id=run_id, owner="B", arm=arm, status="TRAIN_FAILED", gpu_uuid=gpu_uuid, pid=os.getpid()), root)
        raise
    finally:
        jobs.release_lease(gpu_uuid=gpu_uuid, lessee="B", run_root=root)


def _train_b_loop(
    *,
    arm: str,
    seed: int,
    max_epochs: int,
    run_tag: str,
    manifest: dict[str, Any],
    manifest_path: str | Path | None,
    device: torch.device,
    gpu_uuid: str,
    root: Path,
    dest: Path,
    run_id: str,
    started: float,
    resume_path: Path | None = None,
) -> int:
    from . import decoders

    deadline = started + plan.JOB_GPU_HOUR_LIMIT * 3600.0

    train_banks = _load_banks("source_train", device)
    minival_banks = _load_banks("source_minival", device)
    updates_per_epoch = training.count_updates(train_banks)
    warmup_steps = updates_per_epoch  # 1 epoch warmup

    def _build() -> Any:
        _set_seeds(seed)
        model = decoders.build_candidate(arm, seed=seed).to(device)
        return model

    start_epoch = 1
    global_step = 0
    epoch_scores: dict[int, float] = {}
    preflight: dict[str, Any] = {"skipped_on_resume": bool(resume_path)}
    parent_history: dict[str, Any] = {}

    if resume_path is None:
        first_batch = next(training.epoch_batches_shuffled(train_banks, manifest, 1, device=device))

        def _fp32_smoke(model: Any) -> None:
            model.train()
            pred = model.forward_last(first_batch.X, first_batch.bank, first_batch.unit_mask)
            plan.require(torch.isfinite(pred).all(), "B FP32 smoke nonfinite")

        throw = _build()
        _fp32_smoke(throw)
        opt = torch.optim.AdamW(decoders.adamw_param_groups(throw), lr=plan.B_LR, betas=plan.ADAM_BETAS, eps=plan.ADAM_EPS)

        def _step(model: Any, optimizer: torch.optim.AdamW) -> None:
            model.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model.forward_last(first_batch.X, first_batch.bank, first_batch.unit_mask)
                loss = nn.functional.mse_loss(pred.float(), first_batch.last_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.B_GRAD_CLIP)
            optimizer.step()

        t0 = time.monotonic()
        preflight = training.isolated_preflight(lambda: (throw, opt), _step)
        preflight["seconds"] = time.monotonic() - t0
        preflight["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated(device))
        _write_json(dest / "preflight.json", preflight)
        del throw, opt
        torch.cuda.empty_cache()
        model = _build()
        optimizer = torch.optim.AdamW(decoders.adamw_param_groups(model), lr=plan.B_LR, betas=plan.ADAM_BETAS, eps=plan.ADAM_EPS)
    else:
        parent_summary = json.loads((resume_path.parent / "summary.json").read_text(encoding="utf-8"))
        parent_history = {
            "parent_dest": str(resume_path.parent),
            "parent_summary": str(resume_path.parent / "summary.json"),
            "parent_ckpt": str(resume_path),
            "parent_ckpt_sha256": file_sha256(resume_path),
            "parent_summary_sha256": file_sha256(resume_path.parent / "summary.json"),
            "parent_selected_epoch": parent_summary.get("selected_epoch"),
            "parent_endpoint_epoch": parent_summary.get("endpoint_epoch"),
            "parent_sampler_digest": parent_summary.get("sampler_digest"),
        }
        _write_json(dest / "parent_ref.json", parent_history)
        _write_json(
            resume_path.parent / "continuation.json",
            {"child_dest": str(dest), "resume_ckpt": str(resume_path), "child_run_tag": run_tag},
        )
        model = _build()
        optimizer = torch.optim.AdamW(decoders.adamw_param_groups(model), lr=plan.B_LR, betas=plan.ADAM_BETAS, eps=plan.ADAM_EPS)
        payload = load_full_checkpoint(resume_path, model, optimizer, device=device)
        plan.require(int(payload["epoch_one_based"]) == plan.EPOCHS, "resume must start from epoch12, not selected")
        plan.require(int(payload["global_step"]) == updates_per_epoch * plan.EPOCHS, "global_step is not 12 full epochs")
        start_epoch = int(payload["epoch_one_based"]) + 1
        global_step = int(payload["global_step"])
        epoch_scores = {int(k): float(v) for k, v in parent_summary.get("epoch_minival", {}).items()}
        _write_json(dest / "resume_receipt.json", inspect_full_checkpoint(resume_path) | {"restored_start_epoch": start_epoch, "restored_global_step": global_step})
        _write_json(dest / "preflight.json", {"skipped_on_resume": True, "reason": "disposable smoke must not advance formal e12 state"})

    metrics_path = dest / "metrics.jsonl"
    budget_hit = False
    param_count = decoders.count_trainable_parameters(model)
    _write_json(dest / "param_count.json", param_count)
    _write_json(
        dest / "sampler_ref.json",
        {
            "path": str((shared_manifest_24_path(root) if resume_path else shared_manifest_path(root)) if manifest_path is None else manifest_path),
            "digest": manifest["digest"],
            "parent_12_digest": manifest.get("parent_12_digest"),
        },
    )

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        epoch_t0 = time.monotonic()
        for batch in training.epoch_batches_shuffled(train_banks, manifest, epoch, device=device):
            if time.monotonic() >= deadline:
                budget_hit = True
                break
            lr = training.apply_b_lr(
                optimizer,
                plan.B_LR,
                global_step=global_step,
                warmup_steps=warmup_steps,
                epoch=epoch,
                max_epochs=max_epochs,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model.forward_last(batch.X, batch.bank, batch.unit_mask)
                loss = nn.functional.mse_loss(pred.float(), batch.last_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.B_GRAD_CLIP)
            optimizer.step()
            global_step += 1
            running += float(loss.detach().cpu())
            n_batches += 1
            if global_step % 20 == 0:
                _append_jsonl(metrics_path, {
                    "event": "step", "epoch": epoch, "global_step": global_step,
                    "loss": float(loss.detach().cpu()), "lr": lr, "unix": time.time(),
                })
        if n_batches == 0:
            break
        minival = _score_minival(model, minival_banks, device)
        epoch_scores[epoch] = float(minival["equal_session_mean"])
        extras = {
            "train_mse": running / n_batches,
            "minival": minival,
            "seconds": time.monotonic() - epoch_t0,
            "global_step": global_step,
        }
        _save_full_checkpoint(
            dest / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            extras=extras,
            global_step=global_step,
            manifest_digest=manifest["digest"],
        )
        _append_jsonl(metrics_path, {"event": "epoch", "epoch": epoch, **extras, "unix": time.time()})
        write_snapshot(root / "monitor" / f"{run_id}_epoch{epoch:03d}.json")
        jobs.append_job(jobs.JobRecord(
            run_id=run_id, owner="B", arm=arm, status="EPOCH", gpu_uuid=gpu_uuid,
            pid=os.getpid(), note=f"epoch {epoch} minival={epoch_scores[epoch]:.6f}",
        ), root)
        if budget_hit:
            break

    selected = training.select_checkpoint(epoch_scores, max_epoch=max(epoch_scores))
    summary = {
        "run_id": run_id, "arm": arm, "seed": seed, "run_tag": run_tag,
        "sampler": "shuffled_session_pure_v1",
        "sampler_digest": manifest["digest"],
        "epochs_completed": sorted(epoch_scores),
        "selected_epoch": selected,
        "endpoint_epoch": max(epoch_scores),
        "epoch_minival": epoch_scores,
        "updates_per_epoch": updates_per_epoch,
        "warmup_steps": warmup_steps,
        "param_count": param_count,
        "budget_hit": budget_hit,
        "elapsed_s": time.monotonic() - started,
        "preflight": preflight,
        "hardware": snapshot(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "ext4_scored": False,
        "precision": "bf16_autocast",
        "resume_ready": True,
        "ckpt_schema": FULL_CKPT_SCHEMA,
        "cosine_13_24_armed": max_epochs > plan.EPOCHS,
        "resumed_from": str(resume_path) if resume_path else None,
        "start_epoch": start_epoch,
        "parent_history": parent_history,
        "note": (
            "resume of shuffled epoch12 through epoch24; parent 12ep root is immutable"
            if resume_path
            else "from-scratch shuffled run; not a resume of the sequential 12ep record"
        ),
        "leased_gpu_uuid": gpu_uuid,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _write_json(dest / "summary.json", summary)
    jobs.append_job(jobs.JobRecord(
        run_id=run_id, owner="B", arm=arm,
        status="TRAIN_BUDGET" if budget_hit else "TRAIN_FINISHED",
        gpu_uuid=gpu_uuid, pid=os.getpid(), ckpt_path=str(dest),
        extras={"selected_epoch": selected, "run_tag": run_tag},
    ), root)
    print(json.dumps({"status": "TRAIN_FINISHED", "selected_epoch": selected, "budget_hit": budget_hit, "params": param_count, "run_tag": run_tag}, indent=2))
    return 0


def score_a_all_epochs(*, seeds: tuple[int, ...] = (plan.SEED_PRIMARY, plan.SEED_CONFIRM), device: str = "cuda:0") -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    device_t, gpu_uuid, _ = _bind_visible_device()
    del device
    root = plan.active_run_root()
    run_id = "A-QMEM_ext4_scan"
    jobs.acquire_lease(gpu_uuid=gpu_uuid, lessee="A", run_id=run_id, run_root=root)
    ref = json.loads((root / "stage0" / "ref_clean.json").read_text(encoding="utf-8"))
    r_ref = float(ref["R_REF_clean"])
    frozen = champion.load_frozen_champion(device=device_t)
    combined: dict[str, Any] = {"R_REF_clean": r_ref, "seeds": {}, "pick_rule": "max_ext4_session_equal_mean_tie_earlier"}
    try:
        for seed in seeds:
            dest = root / "arms" / "A-QMEM" / f"seed{seed}"
            source_summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
            source_score = json.loads((dest / "ext4_score.json").read_text(encoding="utf-8"))
            epochs = [int(p.stem.split("_")[1]) for p in sorted(dest.glob("epoch_*.pt"))]
            plan.require(epochs == list(range(1, 13)), f"A seed{seed} missing epoch files: {epochs}")
            by_epoch: dict[int, dict[str, Any]] = {}
            for epoch in epochs:
                model = AQMEM(frozen.student).to(device_t)
                _load_a_checkpoint(model, dest / f"epoch_{epoch:03d}.pt")
                report = _score_ext4(model, device_t)
                report["epoch"] = epoch
                report["delta_vs_ref"] = float(report["R_session_equal_mean"]) - r_ref
                by_epoch[epoch] = report
                del model
            visible = training.select_checkpoint(
                {epoch: float(row["R_session_equal_mean"]) for epoch, row in by_epoch.items()},
                max_epoch=12,
            )
            payload = {
                "arm": "A-QMEM",
                "seed": seed,
                "R_REF_clean": r_ref,
                "source_pick_epoch": int(source_summary["selected_epoch"]),
                "source_pick_ext4": source_score["reports"].get("selected", {}),
                "source_pick_kept": True,
                "visible_development_epoch_pick": visible,
                "visible_development_ext4": by_epoch[visible],
                "all_epochs": {str(epoch): row for epoch, row in by_epoch.items()},
                "finished": datetime.now(timezone.utc).isoformat(),
                "note": "visible-development pick uses clean ext-4; original source-minival pick is retained",
            }
            _write_json(dest / "ext4_epoch_scan.json", payload)
            combined["seeds"][str(seed)] = {
                "source_pick_epoch": int(source_summary["selected_epoch"]),
                "source_pick_R": source_score["reports"].get("selected", {}).get("R_session_equal_mean"),
                "source_pick_delta": source_score["reports"].get("selected", {}).get("delta_vs_ref"),
                "visible_pick_epoch": visible,
                "visible_pick_R": by_epoch[visible]["R_session_equal_mean"],
                "visible_pick_delta": by_epoch[visible]["delta_vs_ref"],
                "scan_path": str(dest / "ext4_epoch_scan.json"),
            }
        _write_json(root / "arms" / "A-QMEM" / "ext4_visible_epoch_pick.json", combined)
        jobs.append_job(jobs.JobRecord(run_id=run_id, owner="A", arm="A-QMEM", status="EXT4_SCAN", gpu_uuid=gpu_uuid, extras=combined), root)
        print(json.dumps(combined["seeds"], indent=2))
    finally:
        jobs.release_lease(gpu_uuid=gpu_uuid, lessee="A", run_root=root)
    return 0


def score_b_decoder(*, arm: str, seed: int = plan.SEED_PRIMARY, run_tag: str = "", device: str = "cuda:0") -> int:
    from . import decoders

    del device
    device_t, gpu_uuid, _ = _bind_visible_device()
    root = plan.active_run_root()
    dest = _arm_dest(root, arm, seed, run_tag)
    summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
    selected = int(summary["selected_epoch"])
    endpoint = int(summary["endpoint_epoch"])
    ref = json.loads((root / "stage0" / "ref_clean.json").read_text(encoding="utf-8"))
    r_ref = float(ref["R_REF_clean"])
    run_id = f"{arm}_s{seed}{('_' + run_tag) if run_tag else ''}_ext4"
    jobs.acquire_lease(gpu_uuid=gpu_uuid, lessee="B", run_id=run_id, run_root=root)
    reports: dict[str, Any] = {}
    try:
        for label, epoch in (("selected", selected), ("endpoint12", endpoint)):
            if label == "endpoint12" and epoch == selected:
                reports[label] = {"duplicate_of": "selected", "epoch": epoch}
                continue
            model = decoders.build_candidate(arm, seed=seed).to(device_t)
            payload = torch.load(dest / f"epoch_{epoch:03d}.pt", map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True)
            del missing, unexpected
            report = _score_ext4(model, device_t)
            report["epoch"] = epoch
            report["delta_vs_ref"] = float(report["R_session_equal_mean"]) - r_ref
            reports[label] = report
            del model
        out = {
            "arm": arm,
            "seed": seed,
            "run_tag": run_tag or "original_sequential",
            "R_REF_clean": r_ref,
            "reports": reports,
            "development_evidence": True,
            "architecture_verdict": "none",
            "finished": datetime.now(timezone.utc).isoformat(),
            "note": "source-selected + endpoint12 only; not an SSM architecture verdict",
        }
        _write_json(dest / "ext4_score.json", out)
        jobs.append_job(jobs.JobRecord(run_id=run_id, owner="B", arm=arm, status="EXT4_SCORED", gpu_uuid=gpu_uuid, extras={"selected": reports.get("selected", {}).get("R_session_equal_mean")}), root)
        print(json.dumps({
            "selected": reports.get("selected", {}).get("R_session_equal_mean"),
            "delta": reports.get("selected", {}).get("delta_vs_ref"),
            "endpoint12": reports.get("endpoint12", {}).get("R_session_equal_mean"),
        }, indent=2))
    finally:
        jobs.release_lease(gpu_uuid=gpu_uuid, lessee="B", run_root=root)
    return 0


def _b_epoch_ckpt(root: Path, arm: str, seed: int, epoch: int, run_tag: str) -> Path:
    if int(epoch) <= plan.EPOCHS:
        return root / "arms" / arm / f"seed{seed}_{SHUFFLED_RUN_TAG}" / f"epoch_{epoch:03d}.pt"
    return _arm_dest(root, arm, seed, run_tag) / f"epoch_{epoch:03d}.pt"


def score_b_extended(*, arm: str, seed: int = plan.SEED_PRIMARY, run_tag: str = CONTINUATION_RUN_TAG) -> int:
    from . import decoders

    device_t, gpu_uuid, _ = _bind_visible_device()
    root = plan.active_run_root()
    dest = _arm_dest(root, arm, seed, run_tag)
    parent = root / "arms" / arm / f"seed{seed}_{SHUFFLED_RUN_TAG}"
    summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
    parent_score = json.loads((parent / "ext4_score.json").read_text(encoding="utf-8"))
    ref = json.loads((root / "stage0" / "ref_clean.json").read_text(encoding="utf-8"))
    r_ref = float(ref["R_REF_clean"])
    selected = int(summary["selected_epoch"])
    endpoint = int(summary["endpoint_epoch"])
    run_id = f"{arm}_s{seed}_{run_tag}_ext4_scan"
    jobs.acquire_lease(gpu_uuid=gpu_uuid, lessee="B", run_id=run_id, run_root=root)
    try:
        by_epoch: dict[int, dict[str, Any]] = {}
        for epoch in range(1, endpoint + 1):
            path = _b_epoch_ckpt(root, arm, seed, epoch, run_tag)
            plan.require(path.is_file(), f"missing {path}")
            model = decoders.build_candidate(arm, seed=seed).to(device_t)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"], strict=True)
            report = _score_ext4(model, device_t)
            report["epoch"] = epoch
            report["delta_vs_ref"] = float(report["R_session_equal_mean"]) - r_ref
            report["ckpt"] = str(path)
            by_epoch[epoch] = report
            del model
        visible = training.select_checkpoint(
            {epoch: float(row["R_session_equal_mean"]) for epoch, row in by_epoch.items()},
            max_epoch=endpoint,
        )
        source_row = by_epoch[selected]
        end_row = by_epoch[endpoint]
        payload = {
            "arm": arm,
            "seed": seed,
            "run_tag": run_tag,
            "R_REF_clean": r_ref,
            "source_pick_epoch": selected,
            "source_pick_ext4": source_row,
            "endpoint24": end_row,
            "original12_source_pick_kept": parent_score["reports"].get("selected"),
            "original12_endpoint12_kept": parent_score["reports"].get("endpoint12"),
            "visible_development_epoch_pick": visible,
            "visible_development_ext4": by_epoch[visible],
            "candidate_count": len(by_epoch),
            "all_epochs": {str(epoch): row for epoch, row in by_epoch.items()},
            "pick_rule": "max_source_minival_1_24_tie_earlier; visible uses ext4 over 1-24",
            "development_evidence": True,
            "finished": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(dest / "ext4_epoch_scan.json", payload)
        _write_json(
            dest / "ext4_score.json",
            {
                "arm": arm,
                "seed": seed,
                "run_tag": run_tag,
                "R_REF_clean": r_ref,
                "reports": {
                    "selected": {**source_row, "view": "source_pick_1_24"},
                    "endpoint24": {**end_row, "view": "endpoint24"},
                    "visible_ext4": {**by_epoch[visible], "view": "visible_development_1_24"},
                },
                "finished": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(json.dumps({
            "source_pick": selected,
            "source_R": source_row["R_session_equal_mean"],
            "source_delta": source_row["delta_vs_ref"],
            "endpoint24": end_row["R_session_equal_mean"],
            "visible": visible,
            "visible_R": by_epoch[visible]["R_session_equal_mean"],
        }, indent=2))
    finally:
        jobs.release_lease(gpu_uuid=gpu_uuid, lessee="B", run_root=root)
    return 0


def resume_next_step_parity(*, arm: str, seed: int = plan.SEED_PRIMARY) -> int:
    """Disposable same-GPU interrupted vs uninterrupted first step after epoch12."""
    from . import decoders

    device, gpu_uuid, _ = _bind_visible_device()
    root = plan.active_run_root()
    parent = root / "arms" / arm / f"seed{seed}_{SHUFFLED_RUN_TAG}"
    ckpt_path = parent / "epoch_012.pt"
    manifest = ensure_shared_manifest_24(root)
    run_id = f"{arm}_s{seed}_resume_smoke"
    jobs.acquire_lease(gpu_uuid=gpu_uuid, lessee="B", run_id=run_id, run_root=root)
    try:
        train_banks = _load_banks("source_train", device)
        batch = next(training.epoch_batches_shuffled(train_banks, manifest, 13, device=device))

        def _one_step(model, optimizer):
            model.train()
            lr = training.apply_b_lr(
                optimizer, plan.B_LR, global_step=37980, warmup_steps=3165, epoch=13, max_epochs=24
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model.forward_last(batch.X, batch.bank, batch.unit_mask)
                loss = nn.functional.mse_loss(pred.float(), batch.last_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.B_GRAD_CLIP)
            optimizer.step()
            return lr, float(loss.detach().cpu()), pred.detach().float().cpu()

        def _fresh():
            _set_seeds(seed)
            model = decoders.build_candidate(arm, seed=seed).to(device)
            optimizer = torch.optim.AdamW(decoders.adamw_param_groups(model), lr=plan.B_LR, betas=plan.ADAM_BETAS, eps=plan.ADAM_EPS)
            load_full_checkpoint(ckpt_path, model, optimizer, device=device)
            return model, optimizer

        model_u, opt_u = _fresh()
        lr_u, loss_u, pred_u = _one_step(model_u, opt_u)
        del model_u, opt_u
        torch.cuda.empty_cache()
        model_r, opt_r = _fresh()
        lr_r, loss_r, pred_r = _one_step(model_r, opt_r)
        max_diff = float((pred_u - pred_r).abs().max())
        receipt = {
            "arm": arm,
            "same_gpu": True,
            "lr_uninterrupted": lr_u,
            "lr_resumed": lr_r,
            "loss_uninterrupted": loss_u,
            "loss_resumed": loss_r,
            "max_pred_abs_diff": max_diff,
            "window_ids": list(batch.window_ids),
            "session": batch.session_id,
            "bf16_contract_atol": 1.0e-4,
            "bitwise_same_gpu": max_diff == 0.0,
            "passed": max_diff <= 1.0e-4 and abs(loss_u - loss_r) <= 1.0e-6 and abs(lr_u - lr_r) < 1e-12,
        }
        _write_json(parent / "resume_parity_smoke.json", receipt)
        plan.require(receipt["passed"], f"resume parity failed: {receipt}")
        print(json.dumps(receipt, indent=2))
    finally:
        jobs.release_lease(gpu_uuid=gpu_uuid, lessee="B", run_root=root)
    return 0


def run_training(
    arm: str,
    seed: int,
    device: str,
    max_epochs: int,
    run_tag: str = "",
    manifest: str | None = None,
    resume: str | None = None,
) -> int:
    del device
    if arm == "A-QMEM":
        return run_a_qmem(seed=seed, max_epochs=max_epochs)
    if arm in {"B-MAMBA", "B-TRANSFORMER"}:
        tag = run_tag or (CONTINUATION_RUN_TAG if resume else SHUFFLED_RUN_TAG)
        return run_b_decoder(
            arm=arm,
            seed=seed,
            max_epochs=max_epochs,
            run_tag=tag,
            manifest_path=manifest,
            resume_path=resume,
        )
    raise plan.DualTrackError(f"unknown arm {arm}")
