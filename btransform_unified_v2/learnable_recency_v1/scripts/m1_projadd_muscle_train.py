#!/usr/bin/env python3
"""M1 muscle_response16_svd4/global_rms proj_add learnable-recency trainer.

Binds the official M1 mainline carrier pack
(``results/m1_muscle_r100_v1/carrier_official4``) through the frozen muscle
runner's data path: the legacy fullsession banks keep their frozen b3-identity
E0 and ``frozen._replace_carriers`` swaps only the 64x4 T carrier.  The
recency-bias mechanism, P8/P16/P32 proj_add decoder build, stage structure,
pairing assertions, and smoke discipline mirror
``m1_projadd_learnable_train.py``; the muscle data/carrier/HO path mirrors
``m1_full_learnable_train.py``.  Does not edit any existing file.

Pairing: shared ``named_parameters`` of the learnable model stay byte-equal to
a *locally built* same-seed, same-proj_dim stock ``RiftDecoder`` (only the
``recency_slopes`` buffer differs under the scaled M1 1/3 ladder); init-forward
parity is checked against a same-ladder fixed reference via ``install_temporal``.
No external formal run is used for initialization pairing.  The frozen muscle
formal run is loaded only to pin the source/HO data contracts; its carrier pack
receipt SHA and the official 582205 anchor are recorded in ``run_meta`` as
provenance context and do not participate in contract validation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
WS = ROOT.parent
M1_FLAT = ROOT / "scripts/recency_flat_ablation_v1/m1_full_flat_train.py"
FLAT_DIR = M1_FLAT.parent
FROZEN_DIR = ROOT / "scripts/m1_muscle_r100_v1"
RESULTS = PKG / "results"

for candidate in (PKG / "src", FLAT_DIR, FROZEN_DIR, ROOT / "src", WS / "btransform_unified_v1/src", WS / "btransform_unified_v1/scripts", WS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

spec = importlib.util.spec_from_file_location("_m1_full_flat_projadd_muscle", M1_FLAT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot privately load m1_full_flat_train.py")
m1_flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1_flat)
frozen = m1_flat.frozen

from btransform_unified_v1 import plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.config import add_learnable_flags, config_from_args, config_from_run_meta, dataset_config
from learnable_recency_v1.p8 import maybe_truncate_p8
from learnable_recency_v1.temporal import LearnableRecencyTemporal
from learnable_recency_v1.wrap import (
    LearnableRiftDecoder,
    LearnableRiftStreamDecoder,
    apply_group_lrs,
    assert_shared_byte_equal,
    install_temporal,
    new_parameter_names,
    recency_bias_snapshot,
    trainable_new_parameter_count,
)

SEED = 42
CONTEXT, EPOCHS, BATCH, LR = frozen.CONTEXT, frozen.EPOCHS, frozen.BATCH, frozen.LR
QUERY_PAD_BINS = CONTEXT - 1
UPDATES_PER_EPOCH = frozen.UPDATES_PER_EPOCH
EXPECTED_WINDOWS = frozen.EXPECTED_WINDOWS
HO = frozen.HO
SOURCE = frozen.SOURCE_SESSIONS
FROZEN_SAMPLER_SEED = frozen.SEED
PROJ_DIM = 16
CARRIER_VARIANT = "muscle_response16_svd4/global_rms"
REFERENCE_ARM = "D_JOINT"
MAINLINE_NOTE = "EvalAI submission 582205; original FULL formal e3 reference"
OFFICIAL_SUBMISSION_ID = 582205
OFFICIAL_TEST_HELD_OUT_R2_MEAN = 0.6259910151098528  # submission 582205 test_split_m1, docs/M1_MUSCLE_OFFICIAL_RETRAIN_20260909.md

TRAIN_SCHEMA = "m1_projadd_muscle_learnable_train_v1"
CHECKPOINT_SCHEMA = "m1_projadd_muscle_learnable_epoch_checkpoint_v1"
PROGRESS_SCHEMA = "m1_projadd_muscle_learnable_score_progress_v1"
SCORE_SCHEMA = "m1_projadd_muscle_learnable_ho_calib_epoch_scan_v1"


def _sha_file(path: Path) -> str:
    return frozen._sha_file(path)


def _atomic_json(path: Path, payload: Any) -> None:
    frozen._atomic_json(path, payload)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    frozen._append_jsonl(path, payload)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    return frozen._atomic_checkpoint(path, payload)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return frozen._rng_state(device)


def _source_hashes() -> dict[str, str]:
    """Frozen muscle dependency binding plus the private helper and this runner."""
    values = dict(frozen._source_hashes())
    values[str(M1_FLAT.resolve())] = _sha_file(M1_FLAT.resolve())
    values[str(Path(__file__).resolve())] = _sha_file(Path(__file__).resolve())
    return values


def _carrier_binding(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bind the exact official muscle carrier pack, retaining its receipt fields.

    The frozen runner's binding is authoritative; this line only appends its own
    provenance fields (the ablation-style source hashes are additional fields
    binding these checkpoints without claiming their code made the formal run).
    """
    carriers, original_binding = frozen._carrier_binding(path)
    binding = dict(original_binding)
    binding.pop("binding_sha256", None)
    binding["carrier_variant"] = CARRIER_VARIANT
    binding["identity_interface"] = "proj_add"
    binding["projadd_muscle_runner_py_sha256"] = _sha_file(Path(__file__).resolve())
    binding["binding_sha256"] = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return carriers, binding


def _paired_summary(paired: Mapping[str, Any]) -> dict[str, Any]:
    return {key: paired[key] for key in (
        "path", "run_meta_sha256", "train_receipt_sha256", "score_receipt_sha256",
        "mainline", "reference_selection", "reference_selected_checkpoint_sha256",
    )}


def muscle_reference(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the official pack plus the frozen muscle formal run and its anchor.

    The formal run is used to pin the data contracts (source_contract /
    ho_contract equality) exactly like ``m1_full_learnable_train.py``; the pack
    receipt SHA and the official 582205 anchor are returned as provenance
    context only and never enter a contract comparison.
    """
    carriers, binding = _carrier_binding(args.carrier_pack)
    if binding["carrier_variant"] != CARRIER_VARIANT:
        raise RuntimeError("this runner accepts only the frozen official muscle-response carrier pack")
    paired = m1_flat._paired_reference(args.paired_reference, arm=REFERENCE_ARM, binding=binding)
    if paired["mainline"] != MAINLINE_NOTE:
        raise RuntimeError("paired reference is not the recorded 582205 muscle mainline")
    summary = _paired_summary(paired)
    anchor = {
        "carrier_variant": CARRIER_VARIANT,
        "carrier_pack_npz": binding["carrier_pack_npz"],
        "carrier_pack_npz_sha256": binding["carrier_pack_npz_sha256"],
        "carrier_pack_receipt_sha256": binding["carrier_pack_receipt_sha256"],
        "fit_sha256": binding["fit_sha256"],
        "official_submission_id": OFFICIAL_SUBMISSION_ID,
        "official_test_split_m1_held_out_r2_mean": OFFICIAL_TEST_HELD_OUT_R2_MEAN,
        "paired_reference_summary": summary,
        "provenance_only": True,
        "participates_in_contract_validation": False,
        "note": "582205 anchor and pack receipt recorded as provenance context; "
                "contracts are pinned against the frozen muscle run's recorded source/HO contracts",
    }
    return carriers, binding, dict(paired), anchor


def shared_init_sha(model: nn.Module, names: list[str]) -> str:
    digest = hashlib.sha256()
    named = dict(model.named_parameters())
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(named[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def learnable_decoder(device: torch.device, recency_cfg, proj_dim: int = PROJ_DIM) -> LearnableRiftDecoder:
    # P8 is a rank-8 truncation of the P16 build (the v1 operator requires a
    # multiple of 16); see learnable_recency_v1.p8.
    build_dim = 16 if proj_dim == 8 else proj_dim
    model = LearnableRiftDecoder("m1", recency_cfg, context_bins=CONTEXT, seed=SEED, proj_dim=build_dim).to(device)
    maybe_truncate_p8(model, proj_dim)
    model.temporal.set_attention_backend("local")
    expected = tuple(recency_cfg.temporal_config.windows)
    if tuple(model.temporal_config.windows) != expected:
        raise RuntimeError(f"M1 R100 D{recency_cfg.layers} windows drifted: {tuple(model.temporal_config.windows)} != {expected}")
    if recency_cfg.layers == 4 and expected != (25, 25, 25, 24):
        raise RuntimeError(f"M1 R100 D4 layer windows drifted: {expected}")
    if int(model.proj_dim) != proj_dim:
        raise RuntimeError(f"M1 muscle proj_add decoder proj_dim drifted: {model.proj_dim} != {proj_dim}")
    return model


def stock_decoder(device: torch.device, layers: int, proj_dim: int = PROJ_DIM) -> RiftDecoder:
    """Same-seed stock RiftDecoder with the DEFAULT (unscaled) ladder."""
    build_dim = 16 if proj_dim == 8 else proj_dim
    model = RiftDecoder("m1", context_bins=CONTEXT, bias_mode="recency", seed=SEED, proj_dim=build_dim).to(device)
    maybe_truncate_p8(model, proj_dim)
    if layers != 4:
        install_temporal(model, dataset_config("m1", tier="fixed", layers=layers, ladder="default"), SEED)
    model.temporal.set_attention_backend("local")
    return model


def fixed_ladder_reference(device: torch.device, recency_cfg, proj_dim: int = PROJ_DIM) -> RiftDecoder:
    """Plain fixed-recency RiftDecoder on the *same* ladder as the learnable run."""
    build_dim = 16 if proj_dim == 8 else proj_dim
    model = RiftDecoder("m1", context_bins=CONTEXT, bias_mode="recency", seed=SEED, proj_dim=build_dim).to(device)
    maybe_truncate_p8(model, proj_dim)
    ladder_cfg = dataset_config(
        "m1", tier="fixed", layers=recency_cfg.layers, half_life_seconds=recency_cfg.half_life_seconds
    )
    install_temporal(model, ladder_cfg, SEED)
    model.temporal.set_attention_backend("local")
    return model


def assert_paired(model: nn.Module, device: torch.device, recency_cfg, proj_dim: int = PROJ_DIM) -> dict[str, Any]:
    """Pair against a locally built same-seed stock decoder (no external formal run)."""
    stock = stock_decoder(device, recency_cfg.layers, proj_dim)
    shared = assert_shared_byte_equal(model, stock)
    extra = new_parameter_names(model)
    hashed = shared_init_sha(model, shared)
    ladder_ref = fixed_ladder_reference(device, recency_cfg, proj_dim)
    z = torch.randn(1, CONTEXT, model.temporal_config.width, device=device)
    mask = torch.ones(1, CONTEXT, dtype=torch.bool, device=device)
    with torch.inference_mode():
        left = model.temporal(z, mask)
        right = ladder_ref.temporal(z, mask)
    if not torch.allclose(left, right, atol=1e-6, rtol=1e-6):
        raise RuntimeError("M1 muscle proj_add learnable/same-ladder init forward drift")
    del stock, ladder_ref
    return {
        "pairing": "local same-seed same-proj_dim stock RiftDecoder (DEFAULT ladder); no external formal run",
        "shared_parameter_names": shared,
        "new_parameter_names": extra,
        "new_parameter_counts": {name: int(dict(model.named_parameters())[name].numel()) for name in extra},
        "trainable_new_parameter_count": trainable_new_parameter_count(model),
        "shared_initialization_sha256": hashed,
        "init_forward_max_abs": float((left - right).abs().max().cpu()),
        "ladder": recency_cfg.ladder_metadata(),
    }


def assert_full_stream_parity(model: nn.Module, material: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Full-window versus token-by-token parity on a left-padded W100 context.

    Mirrors the frozen muscle parity probe, including the true-zero valid
    coordinate, but selects the learnable stream decoder when the temporal
    stack is learnable.
    """
    from btransform_unified_v2.streaming import RiftStreamDecoder

    item = material[HO[0]]
    x = np.ascontiguousarray(item["dataset"][0][0], dtype=np.float32).copy()
    valid = frozen.valid_mask_from_padded_starts((97,), device=device)
    # This zero is in a *valid* final-bin coordinate, so it proves zero-valued
    # activity is kept separate from the startup validity mask.
    x[-1, 0] = 0.0
    raw = torch.from_numpy(x).unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            full = model(raw, item["bank"], input_valid_mask=valid)
        stream = LearnableRiftStreamDecoder(model) if isinstance(model.temporal, LearnableRecencyTemporal) else RiftStreamDecoder(model)
        streamed = None
        for offset in range(CONTEXT):
            streamed = stream.stream_step(raw[:, offset], item["bank"], ["m1-projadd-muscle-parity"], valid_mask=valid[:, offset])
        if streamed is None or not torch.allclose(full, streamed, rtol=2e-5, atol=2e-5):
            raise RuntimeError("M1 muscle proj_add learnable full-window versus streaming parity failed")
        return {"status": "PASSED", "query_start": 97, "valid_bins": int(valid.sum()),
                "startup_masked_bins": int((~valid).sum()), "true_zero_valid_coordinate": [CONTEXT - 1, 0],
                "max_abs": float((full - streamed).abs().max().cpu())}
    finally:
        model.train(was_training)


def bias_snapshot(model: nn.Module, x: torch.Tensor, bank: Any, valid: torch.Tensor) -> dict[str, Any]:
    was = model.training
    model.eval()
    with torch.inference_mode():
        tokens = model.frontend_tokens(x[:1], bank)
        stats = recency_bias_snapshot(model.temporal, tokens, valid[:1])
    model.train(was)
    return stats


def _checkpoint_payload(model, optimizer, ema, *, epoch: int, step: int, smoke: bool, meta: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA, "cell": meta["cell"], "epoch": epoch, "global_step": step, "smoke": smoke,
            "tier": meta["tier"], "bias_mode": meta["bias_mode"], "new_parameter_names": meta["new_parameter_names"],
            "config": {"seed": SEED, "sampler_seed": FROZEN_SAMPLER_SEED, "context_bins": CONTEXT,
                       "proj_dim": int(meta["proj_dim"]), "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                       "bias_mode": meta["bias_mode"], "attention_backend": "local", "carrier_variant": CARRIER_VARIANT},
            "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"],
            "carrier_binding": meta["carrier_binding"], "fit_sha256": meta["carrier_binding"]["fit_sha256"],
            "shared_initialization_sha256": meta["initialization_pairing"]["shared_initialization_sha256"],
            "raw_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.state_dict(),
            "rng": _rng_state(device)}


def _validate_checkpoint(path: Path, meta: Mapping[str, Any], *, expected_epoch: int | None = None) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"schema": CHECKPOINT_SCHEMA, "cell": meta["cell"], "smoke": False, "tier": meta["tier"]}
    if any(state.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"checkpoint contract mismatch: {path}")
    epoch = int(state.get("epoch", 0))
    if expected_epoch is not None and epoch != expected_epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {path}")
    if epoch < 1 or int(state.get("global_step", -1)) != epoch * UPDATES_PER_EPOCH:
        raise RuntimeError(f"checkpoint step/epoch mismatch: {path}")
    config = state.get("config", {})
    if config != {"seed": SEED, "sampler_seed": FROZEN_SAMPLER_SEED, "context_bins": CONTEXT,
                  "proj_dim": int(meta["proj_dim"]), "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                  "bias_mode": meta["bias_mode"], "attention_backend": "local", "carrier_variant": CARRIER_VARIANT}:
        raise RuntimeError(f"checkpoint configuration mismatch: {path}")
    for key in ("source_hashes", "source_contract", "new_parameter_names", "carrier_binding"):
        if state.get(key) != meta.get(key):
            raise RuntimeError(f"checkpoint {key} mismatch: {path}")
    if state.get("fit_sha256") != meta["carrier_binding"]["fit_sha256"]:
        raise RuntimeError(f"checkpoint carrier fit SHA mismatch: {path}")
    if state.get("shared_initialization_sha256") != meta["initialization_pairing"]["shared_initialization_sha256"]:
        raise RuntimeError(f"checkpoint shared initialization sha mismatch: {path}")
    return state


def rebuild_model(meta: Mapping[str, Any], device: torch.device) -> LearnableRiftDecoder:
    """Rebuild the learnable decoder for scoring from the run's recorded proj_dim."""
    recency_cfg = config_from_run_meta(meta, "m1")
    return learnable_decoder(device, recency_cfg, int(meta.get("proj_dim", PROJ_DIM)))


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    recency_cfg = config_from_args(args, "m1")
    proj_dim = int(args.proj_dim)
    smoke = args.max_updates_smoke is not None
    if not smoke and args.epochs != EPOCHS:
        raise ValueError("formal M1 muscle proj_add learnable requires exactly 24 epochs")
    if args.resume is not None:
        raise RuntimeError("resume is not used in this drop; rerun into a fresh --dest")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dest = args.dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("new learnable destination must be empty")
    carriers, binding, paired, anchor = muscle_reference(args)
    dataset, sampler = frozen.legacy.build_fullsession_face()
    legacy_banks, legacy_report = frozen.legacy.build_fullsession_banks(dataset)
    banks = frozen._replace_carriers(legacy_banks, carriers, SOURCE)
    contract = frozen._source_contract(
        dataset,
        sampler,
        banks,
        {
            "legacy_frozen_activity_bank_report": legacy_report,
            "carrier_replacement": binding["carrier_variant"],
            "actual_carrier_metadata": "source_contract.bank_hashes + carrier_binding",
        },
    )
    source_calib = frozen.m1_plan.calib_trials_from_dataset(dataset)
    contract["raw_m10_calib_sha256"] = {name: frozen._array_sha(value) for name, value in source_calib.items()}
    if contract != paired["source_contract"]:
        raise RuntimeError("muscle proj_add source contract differs from frozen muscle reference")
    model = learnable_decoder(device, recency_cfg, proj_dim)
    init = assert_paired(model, device, recency_cfg, proj_dim)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    # Shared parameters keep the frozen trainer's name-based no-decay grouping
    # so this arm's recipe matches the frozen muscle pipeline; new recency
    # params are a third group with wd=0 and the tier's lr multiplier.
    new_names = set(new_parameter_names(model))
    shared_named = [(n, p) for n, p in model.named_parameters() if n not in new_names and p.requires_grad]
    extra_named = [(n, p) for n, p in model.named_parameters() if n in new_names and p.requires_grad]
    groups = frozen.optimizer_factory.adamw_param_groups(shared_named, weight_decay=plan.WEIGHT_DECAY)
    if extra_named:
        groups.append({"params": [p for _, p in extra_named], "weight_decay": 0.0,
                       "lr_multiplier": recency_cfg.lr_multiplier})
    for group in groups:
        group.setdefault("lr_multiplier", 1.0)
    optimizer = torch.optim.AdamW(groups, lr=LR, betas=frozen.optimizer_factory.plan.ADAM_BETAS,
                                  eps=frozen.optimizer_factory.plan.ADAM_EPS)
    smoke_preflight = None
    material = None
    if smoke:
        material = frozen._ho_material(carriers)
        ho = frozen._ho_contract(material)
        if ho != paired["ho_contract"]:
            raise RuntimeError("muscle proj_add HO contract differs from frozen muscle reference")
        smoke_preflight = {
            "initialization_pairing": init,
            "ho_contract": ho,
            "streaming_parity": assert_full_stream_parity(model, material, device),
        }
    meta = {
        "schema": TRAIN_SCHEMA, "status": "SMOKE" if smoke else "FORMAL",
        "cell": f"M1-MUSCLE-R100-D{recency_cfg.layers}-P{proj_dim}-PROJADD-LEARNABLE-{recency_cfg.tier.upper()}-V1",
        "task": "m1", "identity_interface": "proj_add", "variant": "learnable_recency", "tier": recency_cfg.tier,
        "bias_mode": "recency" if recency_cfg.tier == "fixed" else f"learnable_{recency_cfg.tier}",
        "carrier_variant": CARRIER_VARIANT,
        "identity_e0": "frozen legacy fullsession bank E0 (b3_identity, [64,100]); carrier replacement swaps only the 64x4 T carrier",
        "learnable_config": recency_cfg.__dict__,
        "new_parameter_names": init["new_parameter_names"],
        "new_parameter_counts": init["new_parameter_counts"],
        "trainable_new_parameter_count": init["trainable_new_parameter_count"],
        "seed": SEED, "sampler_seed": FROZEN_SAMPLER_SEED, "context_bins": CONTEXT, "query_pad_bins": QUERY_PAD_BINS,
        "proj_dim": proj_dim, "layer_windows": list(recency_cfg.temporal_config.windows), "depth": recency_cfg.layers,
        "width": int(model.temporal_config.width), "ladder": recency_cfg.ladder_metadata(), "attention_backend": "local",
        "epochs": args.epochs, "batch": BATCH, "updates_per_epoch": UPDATES_PER_EPOCH,
        "total_updates": EPOCHS * UPDATES_PER_EPOCH,
        "optimizer": {"name": "AdamW", "betas": list(frozen.optimizer_factory.plan.ADAM_BETAS),
                      "eps": frozen.optimizer_factory.plan.ADAM_EPS, "weight_decay": plan.WEIGHT_DECAY,
                      "new_param_weight_decay": 0.0, "clip": 1.0,
                      "grouping": "reference name-based no-decay groups (shared) + wd=0 new-recency group"},
        "lr": {"peak": LR, "min": LR * plan.LR_MIN_FACTOR, "warmup_updates": UPDATES_PER_EPOCH,
               "new_param_multiplier": recency_cfg.lr_multiplier},
        "ema_decay": plan.EMA_DECAY, "unit_dropout": 0.1,
        "initialization_pairing": init,
        "paired_reference": _paired_summary(paired),
        "reference_anchor": anchor,
        "source_train_only_for_gradients": True,
        "development_surface": "visible HO3 calibration (3881 windows), post-training EMA1..24 scan only",
        "official_test_used": False,
        "source_hashes": _source_hashes(),
        "runner_sha256": _sha_file(Path(__file__)),
        "source_contract": dict(contract),
        "carrier_binding": dict(binding), "fit_sha256": binding["fit_sha256"],
        "launch": {"argv": sys.argv, "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "python_no_user_site": os.environ.get("PYTHONNOUSERSITE")},
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    _atomic_json(dest / "run_meta.json", meta)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=frozen.legacy._collate, num_workers=0)
    started = time.monotonic()
    step, last_bias, last_loss = 0, None, None
    for epoch in range(1, args.epochs + 1):
        model.train(); losses: list[float] = []
        last_x = last_bank = last_valid = None
        for batch_id, (x, y, sessions) in enumerate(loader):
            if any(session != sessions[0] for session in sessions):
                raise RuntimeError("M1 source sampler produced mixed-session batch")
            step += 1
            lr = frozen.warmup_cosine_lr(step, total_steps=EPOCHS * UPDATES_PER_EPOCH, warmup_steps=UPDATES_PER_EPOCH,
                                         peak=LR, min_factor=plan.LR_MIN_FACTOR)
            apply_group_lrs(optimizer, lr)
            keep_rng = torch.Generator(device="cpu"); keep_rng.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
            keep = whole_unit_dropout(banks[sessions[0]].unit_mask, p=0.1, generator=keep_rng)
            valid = torch.ones((len(x), CONTEXT), dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                prediction = model(x.float().to(device), banks[sessions[0]], dropout_keep=keep, input_valid_mask=valid)
                loss = nn.functional.mse_loss(prediction.float(), y.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite learnable loss epoch={epoch} batch={batch_id}")
            loss.backward(); grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            optimizer.step(); ema.update_after_step(model)
            losses.append(float(loss.detach().cpu()))
            last_loss = losses[-1]
            last_x, last_bank, last_valid = x.float().to(device), banks[sessions[0]], valid
            if step == 1 or step % 100 == 0:
                _atomic_json(dest / "heartbeat.json", {"status": "SMOKE" if smoke else "TRAINING", "pid": os.getpid(),
                                                       "event": "step", "epoch": epoch, "global_step": step,
                                                       "loss": losses[-1], "lr": lr, "grad_norm": grad_norm,
                                                       "elapsed_seconds": time.monotonic() - started,
                                                       "utc": datetime.now(timezone.utc).isoformat()})
            if smoke and step >= args.max_updates_smoke:
                break
        last_bias = bias_snapshot(model, last_x, last_bank, last_valid)
        checkpoint = _checkpoint_payload(model, optimizer, ema, epoch=epoch, step=step, smoke=smoke, meta=meta, device=device)
        checkpoint_sha = _atomic_checkpoint(dest / f"epoch_{epoch:03d}.pt", checkpoint)
        row = {"status": "SMOKE" if smoke else "TRAINING", "event": "epoch", "epoch": epoch, "global_step": step,
               "train_mse": float(np.mean(losses)), "checkpoint_sha256": checkpoint_sha, "bias": last_bias,
               "utc": datetime.now(timezone.utc).isoformat()}
        _atomic_json(dest / "heartbeat.json", row); _append_jsonl(dest / "metrics.jsonl", row)
        if smoke:
            receipt = {
                "schema": TRAIN_SCHEMA, "status": "COMPLETED", "cell": meta["cell"], "tier": recency_cfg.tier,
                "steps": step, "finite_loss": last_loss is not None and bool(np.isfinite(last_loss)),
                "new_parameter_names": init["new_parameter_names"],
                "new_parameter_counts": init["new_parameter_counts"],
                "trainable_new_parameter_count": init["trainable_new_parameter_count"],
                "carrier_binding": {key: binding[key] for key in (
                    "carrier_variant", "carrier_pack_npz_sha256", "carrier_pack_receipt_sha256", "fit_sha256")},
                "reference_anchor": anchor,
                "preflight": smoke_preflight,
                "postupdate": {"full_vs_stream": assert_full_stream_parity(model, material, device)},
                "bias": last_bias, "checkpoint_sha256": checkpoint_sha,
                "cuda_initialized": torch.cuda.is_initialized(),
            }
            _atomic_json(dest / "smoke_receipt.json", receipt)
            return {"status": "SMOKE_COMPLETED", "steps": step,
                    "trainable_new_parameter_count": init["trainable_new_parameter_count"]}
        if len(losses) != UPDATES_PER_EPOCH:
            raise RuntimeError(f"epoch {epoch} had {len(losses)} updates, expected {UPDATES_PER_EPOCH}")
    if step != EPOCHS * UPDATES_PER_EPOCH:
        raise RuntimeError(f"formal total updates {step} != {EPOCHS * UPDATES_PER_EPOCH}")
    _atomic_json(dest / "train_receipt.json", {"schema": TRAIN_SCHEMA, "status": "COMPLETED", "cell": meta["cell"],
                                               "tier": recency_cfg.tier, "epochs": EPOCHS, "steps": step,
                                               "carrier_variant": CARRIER_VARIANT,
                                               "fit_sha256": binding["fit_sha256"],
                                               "paired_reference": meta["paired_reference"],
                                               "reference_anchor": anchor,
                                               "post_training_scoring_required": True})
    return {"status": "TRAIN_COMPLETED", "steps": step}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    dest = args.dest.resolve()
    meta = frozen._read_json(dest / "run_meta.json")
    if (meta.get("status") != "FORMAL" or meta.get("schema") != TRAIN_SCHEMA or meta.get("task") != "m1"
            or meta.get("identity_interface") != "proj_add"):
        raise RuntimeError("score requires matching formal M1 muscle proj_add learnable run")
    recency_cfg = config_from_run_meta(meta, "m1")
    carriers, binding, _paired, anchor = muscle_reference(args)
    if meta.get("carrier_binding") != dict(binding):
        raise RuntimeError("recorded carrier binding changed against the official pack")
    if meta.get("paired_reference") != _paired_summary(_paired):
        raise RuntimeError("recorded paired reference summary changed")
    if meta.get("reference_anchor") != anchor:
        raise RuntimeError("recorded reference anchor changed")
    if meta.get("source_hashes") != _source_hashes():
        raise RuntimeError("score source hashes differ from the frozen muscle pipeline plus this runner")
    if meta.get("source_contract", {}).get("total_windows") != EXPECTED_WINDOWS:
        raise RuntimeError("score source contract mismatch")
    receipt = frozen._read_json(dest / "train_receipt.json")
    required_receipt = {"status": "COMPLETED", "cell": meta["cell"], "epochs": EPOCHS, "steps": EPOCHS * UPDATES_PER_EPOCH}
    if any(receipt.get(key) != value for key, value in required_receipt.items()):
        raise RuntimeError("score requires a completed matching formal train receipt")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    model = rebuild_model(meta, device)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    material = frozen._ho_material(carriers)
    ho_contract = frozen._ho_contract(material)
    progress_path = dest / "score_progress.json"
    initial_progress = {"schema": PROGRESS_SCHEMA, "cell": meta["cell"], "ho_contract": ho_contract, "completed": {}}
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else initial_progress
    if any(progress.get(key) != initial_progress[key] for key in ("schema", "cell", "ho_contract")):
        raise RuntimeError("score progress provenance mismatch")
    completed = progress.get("completed", {})
    if not isinstance(completed, dict) or not set(completed).issubset({str(epoch) for epoch in range(1, EPOCHS + 1)}):
        raise RuntimeError("malformed score progress")
    repeat: dict[str, Any] | None = None
    for epoch in range(1, EPOCHS + 1):
        path = dest / f"epoch_{epoch:03d}.pt"
        state = _validate_checkpoint(path, meta, expected_epoch=epoch)
        checkpoint_sha = _sha_file(path)
        model.load_state_dict(state["raw_state_dict"], strict=True)
        ema.load_state_dict(state["ema"])
        if repeat is None:
            repeat = frozen._assert_ho_repeatable(model, ema, material, device)
            repeat["full_vs_stream_parity"] = assert_full_stream_parity(model, material, device)
        previous = completed.get(str(epoch))
        if previous is not None:
            if previous.get("checkpoint_sha256") != checkpoint_sha:
                raise RuntimeError(f"checkpoint hash drift after scored epoch {epoch}")
            frozen._validate_scored_report(previous.get("ema_ho_calib", {}))
            continue
        report = frozen._with_ema_eval(model, ema, lambda: frozen._ho_score(model, material, device, capture_predictions=False))
        frozen._validate_scored_report(report)
        completed[str(epoch)] = {"checkpoint_sha256": checkpoint_sha, "ema_ho_calib": report}
        _atomic_json(progress_path, {**initial_progress, "status": "SCORING", "completed": completed,
                                     "last_completed_epoch": epoch, "repeatability": repeat})
        _atomic_json(dest / "heartbeat.json", {"status": "SCORING", "event": "ho_calib_score", "epoch": epoch,
                                               "completed_epochs": len(completed),
                                               "utc": datetime.now(timezone.utc).isoformat()})
    if set(completed) != {str(epoch) for epoch in range(1, EPOCHS + 1)}:
        raise RuntimeError("score scan did not produce exactly 24 epochs")
    scores = {epoch: row["ema_ho_calib"] for epoch, row in completed.items()}
    best = min(range(1, EPOCHS + 1), key=lambda epoch: (-scores[str(epoch)]["equal_session_mean_channel_variance_weighted_r2"], epoch))
    legacy_best = min(range(1, EPOCHS + 1), key=lambda epoch: (-scores[str(epoch)]["equal_session_mean"], epoch))
    score_receipt = {"schema": SCORE_SCHEMA, "status": "COMPLETED", "cell": meta["cell"], "tier": recency_cfg.tier,
                     "proj_dim": int(meta.get("proj_dim", PROJ_DIM)),
                     "ema_by_epoch": scores,
                     "checkpoint_sha256_by_epoch": {epoch: completed[epoch]["checkpoint_sha256"] for epoch in sorted(completed, key=int)},
                     "selection": {"epoch": best, "metric": "channel_variance_weighted_r2",
                                   "rule": "earliest maximum equal-session mean EMA on visible HO3 calibration"},
                     "legacy_selection": {"epoch": legacy_best, "metric": "legacy_flattened_r2",
                                          "rule": "earliest maximum equal-session mean EMA on visible HO3 calibration"},
                     "repeatability": repeat, "ho_contract": ho_contract,
                     "carrier_binding": dict(binding), "fit_sha256": binding["fit_sha256"],
                     "carrier_variant": CARRIER_VARIANT,
                     "paired_reference": meta["paired_reference"], "reference_anchor": meta["reference_anchor"],
                     "official_test_used": False}
    _atomic_json(dest / "score_receipt.json", score_receipt)
    _atomic_json(dest / "ho_calib_epoch_scan.json", score_receipt)
    _atomic_json(dest / "heartbeat.json", {"status": "COMPLETED", "event": "score_complete", "epoch": best,
                                           "completed_epochs": EPOCHS,
                                           "utc": datetime.now(timezone.utc).isoformat()})
    return {"status": "SCORE_COMPLETED", "best_epoch": best}


def default_dest(args: argparse.Namespace) -> Path:
    name = f"m1_projadd_muscle_{args.tier}_p{args.proj_dim}_s{args.seed}"
    return (RESULTS / "smoke" / f"{name}_v1") if args.max_updates_smoke is not None else (RESULTS / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(parser)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--stage", choices=("train", "score"), default="train")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--proj-dim", type=int, default=PROJ_DIM)
    parser.add_argument("--carrier-pack", type=Path, default=m1_flat.DEFAULT_PACK)
    parser.add_argument("--paired-reference", type=Path, default=m1_flat.DEFAULT_REFERENCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.tier not in ("learned_slope", "fixed"):
        parser.error("this runner accepts only --tier learned_slope or fixed")
    if args.epochs < 1 or args.epochs > EPOCHS:
        parser.error("--epochs must be 1..24")
    if args.proj_dim <= 0:
        parser.error("proj_dim must be positive")
    if args.max_updates_smoke is not None and (args.max_updates_smoke < 1 or args.stage != "train"):
        parser.error("smoke requires positive updates and --stage train")
    if args.stage == "score" and args.max_updates_smoke is not None:
        parser.error("smoke is a train-stage-only mode")
    if args.dest is None:
        args.dest = default_dest(args)
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    result = run_train(args) if args.stage == "train" else run_score(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
