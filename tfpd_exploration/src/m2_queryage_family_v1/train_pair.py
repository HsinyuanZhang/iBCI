"""Launch-gated prospective M2 FW-QueryAge16 + prefix paired trainer.

This is a *new* finite named family, never a continuation of CRST-B4's causal
checkpoints.  It is deliberately separate from both the frozen M2 trainer and
the resource smoke: a passed, hash-bound smoke receipt is necessary but is not
selection evidence.  The only gradient surface is ``source_train`` and the
only selection surface is the fixed ``source_minival`` positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from tfpd_exploration.src.m2_b_small_stability_v1 import training
from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
from tfpd_exploration.src.m2_dual_track_v1 import data, plan, sampler
from tfpd_exploration.src.m2_dual_track_v1 import training as source_training
from tfpd_exploration.src.m2_dual_track_v1.contracts import summarize_sessions, variance_weighted_r2

from .model import make_paired_queryage_decoders, shared_parameter_max_abs_diff
from .resource_smoke import prefix


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json"
GO_ENV = "M2_QUERYAGE_PREFIX_PAIR_GO"
EPOCHS, UPDATES_PER_EPOCH, EMA_DECAY = 24, 3165, 0.9995
HARD_SECONDS, HARD_MEMORY_BYTES = 21_600, 22 << 30
SCHEMA = "m2_queryage_prefix_pair_source_only_v1"
SMOKE_STATUS = "PASS_100_SOURCE_TRAIN_PAIRED_UPDATES_NO_SCORE"
ARMS = ("FLAT", "ROUTE")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def validate_smoke_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Require the exact passed resource smoke, not merely any JSON receipt."""
    path = Path(path)
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise RuntimeError("smoke SHA-256 must be lowercase hexadecimal")
    if not path.is_absolute() or not path.is_file() or sha(path) != expected_sha256:
        raise RuntimeError("hash-bound resource smoke receipt is absent or changed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "m2_queryage_pair_resource_smoke_v1"
            or receipt.get("status") != SMOKE_STATUS
            or receipt.get("no_minival_loaded_or_scored") is not True
            or receipt.get("no_checkpoint_selection_or_promotion") is not True):
        raise RuntimeError("resource smoke is not the required passed no-selection receipt")
    forecast = receipt.get("forecast_24x3165_pair_updates_seconds_with_50pct_margin_plus_1800_eval_checkpoint")
    memory = receipt.get("max_memory_bytes")
    if (not isinstance(receipt.get("updates"), list) or len(receipt["updates"]) != 100
            or not isinstance(forecast, (int, float)) or not np.isfinite(forecast) or forecast >= HARD_SECONDS
            or not isinstance(memory, int) or memory < 1 or memory >= HARD_MEMORY_BYTES):
        raise RuntimeError("resource smoke resource forecast/update/memory guard failed")
    return receipt


def validate_smoke_closure_current(receipt: Mapping[str, Any]) -> dict[str, str]:
    """Re-hash the frozen resource-smoke closure before any source bank load."""
    pre, post = receipt.get("authority_pre"), receipt.get("authority_post")
    if not isinstance(pre, dict) or pre != post:
        raise RuntimeError("resource smoke lacks identical pre/post authority")
    closure = pre.get("closure")
    if not isinstance(closure, dict) or not closure:
        raise RuntimeError("resource smoke lacks a frozen closure")
    current: dict[str, str] = {}
    for raw_path, expected in closure.items():
        path = Path(raw_path)
        if not isinstance(expected, str) or not path.is_file():
            raise RuntimeError("resource smoke closure path is invalid")
        current[raw_path] = sha(path)
    if current != closure:
        raise RuntimeError("resource smoke closure changed after PASS")
    return current


def _existing_surface_folders(surface: str) -> list[Path]:
    root = plan.active_run_root() / "cache" / surface
    folders = [root / session for session in plan.HELDIN_SESSIONS]
    if not root.is_dir() or any(not folder.is_dir() for folder in folders):
        raise RuntimeError(f"existing {surface} cache directories required; no creation permitted")
    return folders


def _closure_paths() -> list[Path]:
    """All code and source-only data used by construction, train, or selection."""
    from tfpd_exploration.src.m2_b_small_stability_v1 import config as small_config, decoder as small_decoder, ema
    from tfpd_exploration.src.m2_dual_track_v1 import contracts, decoders
    from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import core as query_core
    from tfpd_exploration.src.m2_family_v1 import config as m2_config, decoder as m2_decoder, routing
    from tfpd_exploration.src.m2_family_v1 import launch as m2_launch

    paths = [
        Path(__file__), Path(__file__).with_name("__init__.py"), Path(__file__).with_name("model.py"),
        Path(__file__).with_name("resource_smoke.py"), MANIFEST,
        Path(training.__file__), Path(ema.__file__),
        Path(data.__file__), Path(plan.__file__), Path(sampler.__file__), Path(source_training.__file__),
        Path(small_config.__file__), Path(small_decoder.__file__), Path(contracts.__file__), Path(decoders.__file__),
        Path(query_core.__file__), Path(m2_config.__file__), Path(m2_decoder.__file__), Path(routing.__file__),
        Path(m2_launch.__file__), ROOT / "tfpd_exploration/results/m2/family_v1/paired_source_only_v1/pretrain_manifest.json",
    ]
    for surface in ("source_train", "source_minival"):
        for folder in _existing_surface_folders(surface):
            paths.extend(folder / name for name in (
                "X_store.npy", "target_store.npy", "eligible_starts.npy", "T.npy", "e0_u.pt",
                "mapping.json", "provenance.json", "extra.json", "calib_activity.npy",
            ))
    if any(not item.is_file() for item in paths):
        raise RuntimeError("required source/code closure file is missing")
    return paths


def closure_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha(path) for path in paths}


def protocol(smoke_path: Path, smoke_sha256: str, smoke: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "PROTOCOL_SAVED_PRE_CONSTRUCTION",
        "family": "CRST-B4[temporal=FW-QueryAge16,routing=FLAT/ROUTE,W50] + deterministic prefix p=.5",
        "compound_recipe_disclosure": (
            "This is a fresh QueryAge-plus-prefix compound family.  It cannot attribute any result, "
            "resource use, or comparison difference to QueryAge or prefix separately."
        ),
        "initialization": "fresh seed42 M2 frontend/readout and fresh FW-QueryAge16; no causal or M1 state loading",
        "temporal": "QueryTemporalStack(width=256,heads=8,layers=4,ffn=512,window=50,age_buckets=16,seed=42)",
        "pair": "FLAT/ROUTE share all non-routing state byte-exactly; ROUTE g=0; static routing only",
        "source_train": "only gradient surface; existing immutable shuffled 24-epoch manifest",
        "source_minival": "only selection surface; all fixed source-minival positions every epoch",
        "selection": "highest EMA equal-session R2 among epochs1..24, earliest epoch tie; raw and endpoint24 reported only",
        "prefix": "resource_smoke.prefix p=.5 once per batch, same shortened tensor for FLAT and ROUTE; only left history zeroed",
        "dropout": "existing deterministic BxN p=.1 whole-unit mask, same mask for FLAT and ROUTE",
        "target": "unchanged M2 decoder_raw final-bin target = native x5; MSE in decoder_raw domain",
        "epochs": EPOCHS,
        "updates_per_epoch_per_arm": UPDATES_PER_EPOCH,
        "total_paired_updates": EPOCHS * UPDATES_PER_EPOCH,
        "optimizer": "existing AdamW parameter grouping/S1-SMALL-COS/LR/clip1",
        "ema_decay": EMA_DECAY,
        "forbidden": ["ext4", "heldout", "official", "public original predictions", "external completed diagnostics for selection", "old causal state loading"],
        "hard_limits": {"seconds": HARD_SECONDS, "cuda_peak_allocated_bytes": HARD_MEMORY_BYTES, "one_visible_gpu": True},
        "resource_smoke": {"path": str(smoke_path), "sha256": smoke_sha256, "status": smoke["status"]},
    }


def _launch_gate(out: Path, physical_gpu: int, threads: int) -> torch.device:
    if out.exists():
        raise FileExistsError(f"refuse overwrite existing output root {out}")
    if not out.is_absolute() or os.environ.get(GO_ENV) != "1":
        raise RuntimeError("absolute output and explicit reviewed GO are required")
    if physical_gpu not in (0, 1) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("exactly one leased physical GPU (0 or 1) is required")
    if threads != 1 or not torch.cuda.is_available():
        raise RuntimeError("one CPU thread and visible CUDA are required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return torch.device("cuda:0")


def _check_budget(started: float) -> None:
    if time.monotonic() - started > HARD_SECONDS:
        raise RuntimeError("hard 21600-second prospective-pair limit reached; checkpoints preserved")
    if torch.cuda.max_memory_allocated() > HARD_MEMORY_BYTES:
        raise RuntimeError("hard 22-GiB allocated-memory limit reached; checkpoints preserved")


def validate_source_batch(batch: Any) -> tuple[int, ...]:
    """Strict actual-source geometry/identity gate, before either arm sees it."""
    starts = tuple(int(value) for value in batch.window_ids)
    if (not isinstance(batch.session_id, str) or not batch.session_id or not starts
            or len(starts) != batch.X.size(0) or len(set(starts)) != len(starts)
            or min(starts) < 0 or max(starts) + 50 > len(batch.bank.X_store)
            or batch.X.dtype != torch.float32 or tuple(batch.X.shape) != (len(starts), 50, 96)
            or batch.last_target.dtype != torch.float32 or tuple(batch.last_target.shape) != (len(starts), 2)
            or not bool(torch.isfinite(batch.X).all()) or not bool(torch.isfinite(batch.last_target).all())):
        raise RuntimeError("actual source batch session/start/FP32 finite geometry drift")
    return starts


def paired_optimizer_step(
    batch: Any,
    models: Mapping[str, torch.nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    emas: Mapping[str, DecoderEMA],
    *, epoch: int, batch_id: int,
    prefix_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = prefix,
    budget_check: Callable[[], None] | None = None,
    digest: Any | None = None,
) -> dict[str, Any]:
    """One paired update: exactly one prefix and one BxN mask shared by arms."""
    starts = validate_source_batch(batch)
    base = batch.unit_mask if batch.unit_mask is not None else batch.bank.unit_mask
    if base.ndim == 1:
        base = base.unsqueeze(0).expand(batch.X.size(0), -1)
    keep = training.unit_dropout_mask(base, seed=42, epoch=epoch, batch_id=batch_id).to(batch.X.device)
    shortened, lengths = prefix_fn(batch.X, seed=42, epoch=epoch, batch_id=batch_id, probability=.5)
    if not torch.equal(shortened[:, -1], batch.X[:, -1]):
        raise RuntimeError("prefix must preserve each observed current bin")
    if digest is not None:
        digest["order"].update(batch.session_id.encode()); digest["order"].update(np.asarray(starts, dtype=np.int64).tobytes())
        digest["target"].update(batch.last_target.detach().cpu().numpy().tobytes())
        digest["prefix"].update(lengths.detach().cpu().numpy().tobytes()); digest["keep"].update(keep.detach().cpu().numpy().tobytes())
    row = {"rows": int(batch.X.size(0)), "prefix_shortened_rows": int((lengths < 50).sum()), "loss": {}}
    for arm in ARMS:
        if budget_check is not None: budget_check()
        model, optimizer = models[arm], optimizers[arm]
        model.train(); optimizer.zero_grad(set_to_none=True)
        training.apply_cell_lr(optimizer, "S1-SMALL-COS", (epoch - 1) * UPDATES_PER_EPOCH + batch_id + 1)
        prediction = model.forward_last(shortened, batch.bank, dropout_keep=keep)
        loss = torch.nn.functional.mse_loss(prediction.float(), batch.last_target.float())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite decoder_raw paired loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step(); emas[arm].update_after_step(model)
        if budget_check is not None: budget_check()
        row["loss"][arm] = float(loss.detach().cpu())
    return row


def score_equal_session(model: torch.nn.Module, banks: Mapping[str, Any], device: torch.device,
                        budget_check: Callable[[], None] | None = None) -> float:
    """The exact M2 source-minival selection positions, in native units only."""
    values: dict[str, float] = {}
    model.eval()
    for session, bank in banks.items():
        prediction, target = [], []
        for batch in data.iter_session_batches(bank, batch_size=plan.EFFECTIVE_BATCH, device=device,
                                               target_space=plan.SCORING_TARGET_SPACE):
            if budget_check is not None: budget_check()
            with torch.inference_mode():
                prediction.append((model.forward_last(batch.X, batch.bank, batch.unit_mask) / plan.BEHAVIOR_SCALE).cpu().numpy())
            if budget_check is not None: budget_check()
            target.append(batch.last_target.cpu().numpy())
        values[session] = variance_weighted_r2(np.concatenate(target), np.concatenate(prediction))
    return float(summarize_sessions(values)["equal_session_mean"])


def select_earliest_ema(ema_history: Mapping[int, float], *, epochs: int = EPOCHS) -> int:
    normalized = {int(epoch): float(value) for epoch, value in ema_history.items()}
    if tuple(sorted(normalized)) != tuple(range(1, epochs + 1)):
        raise RuntimeError("selection requires contiguous EMA epochs 1..24")
    if not all(np.isfinite(value) for value in normalized.values()):
        raise RuntimeError("selection requires finite EMA values")
    return min(normalized, key=lambda epoch: (-normalized[epoch], epoch))


def checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: DecoderEMA,
                       *, arm: str, epoch: int, manifest_digest: str, authority: Mapping[str, Any],
                       protocol_sha256: str, recipe: Mapping[str, Any], updates_per_epoch: int = UPDATES_PER_EPOCH) -> dict[str, Any]:
    state = training.TrainState(cell=f"QUERYAGE_PREFIX_{arm}", seed=42, epoch=epoch,
                                batch_id=updates_per_epoch, global_step=epoch * updates_per_epoch,
                                manifest_digest=manifest_digest)
    payload = training.save_checkpoint(model, optimizer, ema, state)
    payload.update(schema="m2_queryage_prefix_pair_checkpoint_v1", authority=dict(authority),
                   protocol_sha256=protocol_sha256, recipe=dict(recipe), actual_epoch_batches=updates_per_epoch)
    if payload["ema"]["n_updates"] != epoch * updates_per_epoch or payload["ema"]["decay"] != EMA_DECAY:
        raise RuntimeError("checkpoint EMA count/decay drift")
    return payload


def _record_epoch(root: Path, *, arm: str, epoch: int, model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer, ema: DecoderEMA, raw: float, ema_value: float,
                  manifest_digest: str, authority: Mapping[str, Any], protocol_sha256: str,
                  recipe: Mapping[str, Any], epoch_digest: Mapping[str, str], updates_per_epoch: int) -> dict[str, Any]:
    payload = checkpoint_payload(model, optimizer, ema, arm=arm, epoch=epoch,
                                 manifest_digest=manifest_digest, authority=authority, protocol_sha256=protocol_sha256,
                                 recipe=recipe, updates_per_epoch=updates_per_epoch)
    checkpoint = root / arm / f"epoch_{epoch:03d}.pt"
    _atomic_torch(checkpoint, payload)
    receipt = {"schema": "m2_queryage_prefix_pair_epoch_receipt_v1", "arm": arm, "epoch": epoch,
               "raw_equal_session_r2": raw, "ema_equal_session_r2": ema_value,
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
               "checkpoint_has_raw_ema_optimizer_rng": all(key in payload for key in ("raw_state_dict", "ema", "optimizer", "rng")),
               "endpoint_checkpoint": epoch == EPOCHS, "actual_epoch_batches": updates_per_epoch,
               "actual_batch_sha256": dict(epoch_digest), "protocol_sha256": protocol_sha256}
    receipt_path = root / arm / f"epoch_{epoch:03d}_receipt.json"
    _atomic_json(receipt_path, receipt)
    receipt["receipt_sha256"] = sha(receipt_path)
    return receipt


def _checkpoint_roundtrip_once(path: Path, *, arm: str, protocol_sha256: str) -> None:
    """Re-read the saved first-epoch artifact and strictly rebuild its topology."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol_sha256") != protocol_sha256 or payload.get("cell") != f"QUERYAGE_PREFIX_{arm}":
        raise RuntimeError("saved checkpoint protocol/cell drift")
    rng_before = training.capture_rng()
    try:
        fresh = make_paired_queryage_decoders(42)[0 if arm == "FLAT" else 1]
        optimizer = training.make_optimizer(fresh.trainable_parameters().items())
        ema = DecoderEMA(fresh, decay=EMA_DECAY); state = training.TrainState(cell="")
        training.load_checkpoint(payload, fresh, optimizer, ema, state)
        if (set(payload["raw_state_dict"]) != set(fresh.state_dict()) or set(payload["ema"]["shadow"]) != set(fresh.trainable_parameters())
                or ema.n_updates != payload["ema"]["n_updates"]
                or any(not torch.equal(v, fresh.state_dict()[k]) for k, v in payload["raw_state_dict"].items())
                or any(not torch.equal(v, ema.shadow[k]) for k, v in payload["ema"]["shadow"].items())):
            raise RuntimeError("saved checkpoint strict RAW/EMA topology roundtrip drift")
    finally:
        training.restore_rng(rng_before)


def verify_epoch_artifacts(root: Path, history: Mapping[str, Any], protocol_sha256: str) -> None:
    if sha(root / "protocol_preconstruction.json") != protocol_sha256:
        raise RuntimeError("preconstruction protocol changed during training")
    for arm in ARMS:
        for epoch, receipt in history[arm]["receipts"].items():
            path = root / arm / f"epoch_{int(epoch):03d}_receipt.json"
            if sha(path) != receipt["receipt_sha256"] or sha(Path(receipt["checkpoint"])) != receipt["checkpoint_sha256"]:
                raise RuntimeError("persisted epoch artifact changed after its sealed receipt")
            if json.loads(path.read_text()) != {k: v for k, v in receipt.items() if k != "receipt_sha256"}:
                raise RuntimeError("persisted epoch receipt content drift")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*")) if path.is_file()}


def preflight_only(*, smoke_receipt: Path, smoke_sha256: str) -> dict[str, Any]:
    """CPU/read-only authority check; no GO, GPU, output root, or bank creation."""
    smoke = validate_smoke_receipt(Path(smoke_receipt), smoke_sha256)
    smoke_closure = validate_smoke_closure_current(smoke)
    paths = _closure_paths()  # checks actual source train/minival roster directories and files only
    manifest = sampler.load_manifest(MANIFEST)
    batches = manifest.get("batches", {})
    counts = {epoch: len(batches.get(str(epoch), ())) for epoch in range(1, EPOCHS + 1)}
    if tuple(counts) != tuple(range(1, EPOCHS + 1)) or any(value != UPDATES_PER_EPOCH for value in counts.values()):
        raise RuntimeError("immutable manifest must contain exactly 24x3165 batches")
    return {"schema": SCHEMA, "status": "PREFLIGHT_ONLY_NO_GPU_NO_BANKS_NO_OUTPUT",
            "smoke_receipt_sha256": smoke_sha256, "smoke_closure": smoke_closure,
            "closure": closure_hashes(paths), "manifest_sha256": sha(MANIFEST),
            "manifest_digest": manifest.get("digest"), "epoch_batch_counts": counts}


def run(output: Path, *, physical_gpu: int, smoke_receipt: Path, smoke_sha256: str, threads: int = 1) -> dict[str, Any]:
    """Actual 24x3165 paired-run entry point; deliberately impossible without GO."""
    started = time.monotonic()
    output = Path(output)
    device = _launch_gate(output, physical_gpu, threads)
    preflight = preflight_only(smoke_receipt=Path(smoke_receipt), smoke_sha256=smoke_sha256)
    smoke = validate_smoke_receipt(Path(smoke_receipt), smoke_sha256)
    smoke_closure = preflight["smoke_closure"]
    paths = _closure_paths(); authority_pre = preflight["closure"]
    authority_pre["hash_bound_resource_smoke_receipt"] = smoke_sha256
    authority_pre["resource_smoke_closure_current"] = smoke_closure
    output.mkdir(parents=True, exist_ok=False)
    frozen_protocol = protocol(Path(smoke_receipt), smoke_sha256, smoke)
    frozen_protocol["authority_pre"] = authority_pre
    _atomic_json(output / "protocol_preconstruction.json", frozen_protocol)
    protocol_sha256 = sha(output / "protocol_preconstruction.json")
    _check_budget(started)
    torch.manual_seed(42); np.random.seed(42); torch.cuda.reset_peak_memory_stats()
    flat, route = make_paired_queryage_decoders(seed=42)
    if shared_parameter_max_abs_diff(flat, route) != 0.0:
        raise RuntimeError("paired shared initialization drift")
    models = {"FLAT": flat.to(device), "ROUTE": route.to(device)}
    optimizers = {arm: training.make_optimizer(model.trainable_parameters().items()) for arm, model in models.items()}
    emas = {arm: DecoderEMA(model, decay=EMA_DECAY) for arm, model in models.items()}
    manifest = sampler.load_manifest(MANIFEST)
    banks = {surface: {session: data.load_session_bank(surface, session, device=device) for session in plan.HELDIN_SESSIONS}
             for surface in ("source_train", "source_minival")}
    _check_budget(started)
    if source_training.count_updates(banks["source_train"]) != UPDATES_PER_EPOCH:
        raise RuntimeError("exact 3165 source-train updates per epoch required")
    history = {arm: {"RAW": {}, "EMA": {}, "receipts": {}} for arm in ARMS}
    for epoch in range(1, EPOCHS + 1):
        epoch_digest = {key: hashlib.sha256() for key in ("order", "target", "prefix", "keep")}
        last_row: dict[str, Any] | None = None
        for batch_id, batch in enumerate(source_training.epoch_batches_shuffled(
                banks["source_train"], manifest, epoch, device=device, target_space="decoder_raw")):
            _check_budget(started)
            last_row = paired_optimizer_step(batch, models, optimizers, emas, epoch=epoch, batch_id=batch_id,
                                             budget_check=lambda: _check_budget(started), digest=epoch_digest)
            if (batch_id + 1) % 20 == 0:
                _atomic_json(output / "heartbeat.json", {"status": "RUNNING", "epoch": epoch,
                    "batch_id": batch_id + 1, "loss": last_row["loss"], "unix": time.time(),
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        if batch_id + 1 != UPDATES_PER_EPOCH:
            raise RuntimeError("immutable manifest/source sampler cardinality drift")
        digest_hex = {key: value.hexdigest() for key, value in epoch_digest.items()}
        for arm in ARMS:
            _check_budget(started)
            raw = score_equal_session(models[arm], banks["source_minival"], device, lambda: _check_budget(started))
            ema_value = emas[arm].score_with_ema(models[arm], lambda current: score_equal_session(
                current, banks["source_minival"], device, lambda: _check_budget(started)))
            if not np.isfinite(raw) or not np.isfinite(ema_value):
                raise RuntimeError("non-finite source-minival score")
            history[arm]["RAW"][epoch] = raw; history[arm]["EMA"][epoch] = ema_value
            history[arm]["receipts"][epoch] = _record_epoch(output, arm=arm, epoch=epoch, model=models[arm],
                optimizer=optimizers[arm], ema=emas[arm], raw=raw, ema_value=ema_value,
                manifest_digest=manifest["digest"], authority=authority_pre, protocol_sha256=protocol_sha256,
                recipe=frozen_protocol, epoch_digest=digest_hex, updates_per_epoch=UPDATES_PER_EPOCH)
            if epoch == 1:
                _checkpoint_roundtrip_once(Path(history[arm]["receipts"][epoch]["checkpoint"]), arm=arm,
                                           protocol_sha256=protocol_sha256)
        _atomic_json(output / "progress.json", {"status": "RUNNING", "completed_epoch": epoch,
            "elapsed_seconds": time.monotonic() - started, "history": history})
    picks = {arm: select_earliest_ema(history[arm]["EMA"]) for arm in ARMS}
    verify_epoch_artifacts(output, history, protocol_sha256)
    authority_post = closure_hashes(paths)
    # The receipt itself is an explicit launch dependency, even though it is
    # not an input to the gradient/selection arrays.
    validate_smoke_receipt(Path(smoke_receipt), smoke_sha256)
    if validate_smoke_closure_current(smoke) != smoke_closure:
        raise RuntimeError("resource smoke closure changed during prospective pair")
    authority_post["hash_bound_resource_smoke_receipt"] = smoke_sha256
    authority_post["resource_smoke_closure_current"] = smoke_closure
    _check_budget(started)
    if authority_post != authority_pre:
        raise RuntimeError("code/source closure changed during prospective pair")
    result = {"schema": SCHEMA, "status": "SOURCE_MINIVAL_SELECTION_ONLY_COMPLETE", "protocol": frozen_protocol,
              "authority_pre": authority_pre, "authority_post": authority_post, "manifest_sha256": sha(MANIFEST),
              "manifest_digest": manifest["digest"], "history": history, "selected_primary_ema_epoch": picks,
              "endpoint_epoch": EPOCHS, "selection_disclosure": "EMA only; earliest tie; source-minival is selection-exposed, not untouched generalization",
              "elapsed_seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "outer_or_ext4_or_public_predictions_opened": False}
    _atomic_json(output / "selection_summary.json", result)
    _atomic_json(output / "artifact_hashes_pre_completion.json", _tree_hashes(output))
    _atomic_json(output / "completion.json", {"schema": SCHEMA, "status": result["status"],
        "selection_summary_sha256": sha(output / "selection_summary.json"),
        "artifact_hash_manifest_sha256": sha(output / "artifact_hashes_pre_completion.json"),
        "authority_pre": authority_pre, "authority_post": authority_post})
    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1))
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--smoke-sha256", required=True)
    parser.add_argument("--threads", type=int, choices=(1,))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        return preflight_only(smoke_receipt=args.smoke_receipt, smoke_sha256=args.smoke_sha256)
    if args.output is None or args.physical_gpu is None or args.threads is None:
        parser.error("--output, --physical-gpu, and --threads are required unless --preflight-only")
    return run(args.output, physical_gpu=args.physical_gpu, smoke_receipt=args.smoke_receipt,
               smoke_sha256=args.smoke_sha256, threads=args.threads)


if __name__ == "__main__":
    main()
