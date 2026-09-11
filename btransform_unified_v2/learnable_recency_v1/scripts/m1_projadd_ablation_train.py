#!/usr/bin/env python3
"""M1 muscle-carrier R100/D4/P16 proj_add learnable-recency ACTIVITY_ONLY ablation runner.

Mirrors the M1 muscle mainline data path exactly as ``m1_full_learnable_train.py``
uses it: the frozen muscle runner (``scripts/m1_muscle_r100_v1/train.py``, loaded
privately via ``scripts/recency_flat_ablation_v1/m1_full_flat_train.py``) supplies
``legacy.build_fullsession_face/build_fullsession_banks``, the official
``muscle_response16_svd4/global_rms`` carrier pack
(``results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz`` — the
EvalAI-582205 generation), ``_replace_carriers``, ``_ho_material(carriers)``,
``_ho_contract`` and the HO scoring helpers.  The decoder side is borrowed from
``m1_projadd_learnable_train.py``: ``LearnableRiftDecoder`` P16 +
learned_slope@default-ladder + seed 42 (no P8 truncation).  The recipe is frozen
and ONLY the identity injection is changed:

  activity_only: E0 kept bit-for-bit (the legacy bank E0 is the b3_identity
                 activity encoder output over the M10 calibration support —
                 label-free), carrier zeroed — every transformed bank carries a
                 fresh ``carrier = zeros((64, 4), float32)``.

Transforms always build NEW ``TaskBank`` objects with fresh identity arrays; the
frozen source banks (legacy and muscle-replaced) and the held-out material banks
are never mutated, and non-sharing is asserted.  The SAME carrier-zeroing
transform is applied to the held-out material (``frozen._ho_material(carriers)``
banks) in the train-stage smoke preflight and in the score stage.  Pairing binds
the carrier pack receipt plus the data/HO contracts of the completed muscle
formal run ``results/m1_muscle_r100_v1/formal_s42_gpu1`` (validated through
``m1_full_flat_train._paired_reference``); no initialization SHA is pinned
against it because that reference is a different decoder family — decoder
initialization is instead paired structurally (same-seed stock P16 proj_add
build byte-equal on shared parameters + same-ladder init-forward parity).
``--identity`` is required and accepts only ``activity_only``.

This file does not modify any existing script; the frozen runner and the flat
runner are loaded privately and only their helpers are reused.
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
V1 = WS / "btransform_unified_v1"
M1_FLAT = ROOT / "scripts/recency_flat_ablation_v1/m1_full_flat_train.py"
FLAT_DIR = M1_FLAT.parent
FROZEN_DIR = ROOT / "scripts/m1_muscle_r100_v1"
RESULTS = PKG / "results"

for candidate in (PKG / "src", FLAT_DIR, FROZEN_DIR, ROOT / "src", V1 / "src", V1 / "scripts", WS, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

_spec = importlib.util.spec_from_file_location("_m1_full_flat_ablation", M1_FLAT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot privately load m1_full_flat_train.py")
m1_flat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m1_flat)
frozen = m1_flat.frozen

from btransform_unified_v1 import plan
from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from btransform_unified_v2.streaming import RiftStreamDecoder
from learnable_recency_v1.config import add_learnable_flags, config_from_args, config_from_run_meta
from learnable_recency_v1.temporal import LearnableRecencyTemporal
from learnable_recency_v1.wrap import (
    LearnableRiftStreamDecoder,
    apply_group_lrs,
    assert_shared_byte_equal,
    new_parameter_names,
)

import m1_projadd_learnable_train as template

SEED = 42
CONTEXT, EPOCHS, BATCH, LR = frozen.CONTEXT, frozen.EPOCHS, frozen.BATCH, frozen.LR
UPDATES_PER_EPOCH, EXPECTED_WINDOWS = frozen.UPDATES_PER_EPOCH, frozen.EXPECTED_WINDOWS
QUERY_PAD_BINS = frozen.QUERY_PAD_BINS
HO = frozen.HO
SOURCE = frozen.SOURCE_SESSIONS
FROZEN_SAMPLER_SEED = frozen.SEED
PROJ_DIM = 16  # fixed; the template's P8 knob is deliberately not a parameter here
REFERENCE_ARM = "D_JOINT"  # arm recorded in the paired muscle formal reference
CARRIER_VARIANT = "muscle_response16_svd4/global_rms"

IDENTITY = "activity_only"
ABLATION_IDENTITIES = (IDENTITY,)
CARRIER_SHAPE = (64, 4)  # M1_UNITS x M1_CARRIER_DIM (m1_projadd geometry)

TRAIN_SCHEMA = "m1_rift_projadd_ablation_train_v1"
CHECKPOINT_SCHEMA = "m1_rift_projadd_ablation_epoch_checkpoint_v1"
PROGRESS_SCHEMA = "m1_rift_projadd_ablation_score_progress_v1"
SCORE_SCHEMA = "m1_rift_projadd_ablation_ho_calib_epoch_scan_v1"

DEFAULT_PACK = m1_flat.DEFAULT_PACK
DEFAULT_REFERENCE = m1_flat.DEFAULT_REFERENCE
DEFAULT_DEST_NAME = "m1_projadd_activity_only_s42"

ABLATION_DEFINITION = (
    "activity_only: E0 kept bit-for-bit (legacy bank b3_identity over the M10 calibration "
    "support — label-free), muscle carrier replaced by zeros((64,4), float32)"
)


def _sha_file(path: Path) -> str:
    return frozen._sha_file(path)


def source_hashes() -> dict[str, str]:
    """Frozen muscle-runner data seal plus the learnable sources, the proj_add template, and this runner."""
    values = dict(frozen._source_hashes())
    values[str(Path(__file__).resolve())] = _sha_file(Path(__file__).resolve())
    values[str(Path(template.__file__).resolve())] = _sha_file(Path(template.__file__).resolve())
    for path in sorted((PKG / "src/learnable_recency_v1").glob("*.py")):
        values[str(path)] = _sha_file(path)
    return values


def carrier_binding(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Original frozen pack/receipt binding plus this ablation's provenance fields.

    The frozen binding fields (pack/receipt/fit/implementation digests) stay
    verbatim so ``m1_flat._paired_reference`` can pin them against the muscle
    formal run; the added fields identify this arm.  ``binding_sha256`` is
    recomputed over the extended record.
    """
    carriers, original = frozen._carrier_binding(path)
    binding = dict(original)
    binding.pop("binding_sha256", None)
    binding["carrier_variant"] = CARRIER_VARIANT
    binding["temporal_bias_mode"] = "learnable_recency_learned_slope_default_p16"
    binding["identity_ablation"] = IDENTITY
    binding["ablation_runner_py_sha256"] = _sha_file(Path(__file__).resolve())
    binding["binding_sha256"] = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return carriers, binding


def transform_bank(bank: TaskBank, identity: str) -> TaskBank:
    """Return a NEW bank with the activity_only ablation identity applied; ``bank`` is never mutated.

    E0 is copied bit-for-bit into a fresh C-contiguous array; carrier becomes a
    fresh ``zeros((64, 4), float32)``.  X_store/target_store/window_ids are
    observational stores shared read-only with the input bank; unit_mask is
    copied.  ``calibration_meta`` is copied with the ORIGINAL input digests
    preserved in the ``identity_ablation`` record; ``carrier_sha256`` is
    updated to the zeroed carrier so the meta stays self-consistent with the
    arrays the new bank actually holds.
    """
    if identity != IDENTITY:
        raise ValueError(f"identity must be {IDENTITY!r}, got {identity!r}")
    if tuple(bank.carrier.shape) != CARRIER_SHAPE:
        raise RuntimeError(f"{bank.session_id}: carrier shape {bank.carrier.shape} != {CARRIER_SHAPE}")
    e0 = np.array(bank.E0, dtype=np.float32, order="C", copy=True)
    carrier = np.zeros(CARRIER_SHAPE, dtype=np.float32, order="C")
    if not np.array_equal(e0, bank.E0):
        raise RuntimeError(f"{bank.session_id}: E0 copy is not bitwise identical to the input")
    record = {
        "identity": identity,
        "definition": ABLATION_DEFINITION,
        "input_e0_sha256": array_sha256(bank.E0),
        "input_carrier_sha256": array_sha256(bank.carrier),
        "transformed_e0_sha256": array_sha256(e0),
        "transformed_carrier_sha256": array_sha256(carrier),
        "e0_bitwise_unchanged": True,
    }
    meta = dict(bank.calibration_meta)
    meta["carrier_sha256"] = record["transformed_carrier_sha256"]
    meta["identity_ablation"] = record
    return TaskBank(
        session_id=bank.session_id,
        E0=e0,
        carrier=carrier,
        unit_mask=np.array(bank.unit_mask, dtype=np.bool_, copy=True),
        X_store=bank.X_store,
        target_store=bank.target_store,
        window_ids=bank.window_ids,
        calibration_meta=meta,
    )


def _assert_fresh_transform(session: str, bank: TaskBank, transformed: TaskBank, e0_digest: str, carrier_digest: str) -> None:
    if array_sha256(bank.E0) != e0_digest or array_sha256(bank.carrier) != carrier_digest:
        raise RuntimeError(f"{session}: transform mutated the frozen bank")
    if transformed.E0 is bank.E0 or transformed.carrier is bank.carrier:
        raise RuntimeError(f"{session}: transform must not share identity arrays with the frozen bank")


def transform_banks(banks: Mapping[str, TaskBank], identity: str) -> tuple[dict[str, TaskBank], dict[str, Any]]:
    """Transform every source bank; assert originals unchanged and identity arrays fresh."""
    out: dict[str, TaskBank] = {}
    rows: dict[str, Any] = {}
    for session, bank in banks.items():
        e0_digest, carrier_digest = array_sha256(bank.E0), array_sha256(bank.carrier)
        transformed = transform_bank(bank, identity)
        _assert_fresh_transform(session, bank, transformed, e0_digest, carrier_digest)
        row = {
            "input_e0_sha256": e0_digest,
            "input_carrier_sha256": carrier_digest,
            "transformed_e0_sha256": array_sha256(transformed.E0),
            "transformed_carrier_sha256": array_sha256(transformed.carrier),
            "e0_sha256_unchanged": array_sha256(transformed.E0) == e0_digest,
            "carrier_all_zero": bool(np.all(transformed.carrier == 0.0)),
            "identity_arrays_not_shared": True,
        }
        if not row["e0_sha256_unchanged"] or not row["carrier_all_zero"]:
            raise RuntimeError(f"{session}: activity_only transform contract violated")
        out[session] = transformed
        rows[session] = row
    return out, rows


def transform_material(material: Mapping[str, Any], identity: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the SAME identity transform to ``frozen._ho_material`` banks (fresh rows).

    Only the ``bank`` entry and its ``e0_sha256``/``carrier_sha256`` digests
    change; dataset/calib10/starts/target/body/neural/covariate fields are
    carried over untouched so ``frozen._ho_contract`` on the transformed
    material describes the ablation surface exactly.
    """
    out: dict[str, Any] = {}
    rows: dict[str, Any] = {}
    for session, item in material.items():
        bank = item["bank"]
        e0_digest, carrier_digest = array_sha256(bank.E0), array_sha256(bank.carrier)
        transformed = transform_bank(bank, identity)
        _assert_fresh_transform(f"heldout/{session}", bank, transformed, e0_digest, carrier_digest)
        row = dict(item)
        row["bank"] = transformed
        row["e0_sha256"] = array_sha256(transformed.E0)
        row["carrier_sha256"] = array_sha256(transformed.carrier)
        record = {
            "input_e0_sha256": e0_digest,
            "input_carrier_sha256": carrier_digest,
            "transformed_e0_sha256": row["e0_sha256"],
            "transformed_carrier_sha256": row["carrier_sha256"],
            "e0_sha256_unchanged": row["e0_sha256"] == e0_digest,
            "carrier_all_zero": bool(np.all(transformed.carrier == 0.0)),
            "identity_arrays_not_shared": True,
        }
        if not record["e0_sha256_unchanged"] or not record["carrier_all_zero"]:
            raise RuntimeError(f"heldout/{session}: activity_only transform contract violated")
        out[session] = row
        rows[session] = record
    return out, rows


def identity_ablation_block(
    identity: str, source_rows: Mapping[str, Any], heldout_rows: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """run_meta block: identity, per-session carrier-zero summary, E0-unchanged assertion."""
    block: dict[str, Any] = {
        "identity": identity,
        "definition": ABLATION_DEFINITION,
        "carrier_variant": CARRIER_VARIANT,
        "carrier_shape": list(CARRIER_SHAPE),
        "source_sessions": dict(source_rows),
        "e0_sha256_unchanged": all(row["e0_sha256_unchanged"] for row in source_rows.values()),
        "carrier_zeroed": all(row["carrier_all_zero"] for row in source_rows.values()),
        "identity_arrays_not_shared": all(row["identity_arrays_not_shared"] for row in source_rows.values()),
        "ho_side_transform": (
            "the same carrier-zeroing transform is applied to frozen._ho_material(carriers) banks "
            "in the train smoke preflight and the score stage"
        ),
    }
    if heldout_rows:
        block["heldout_sessions"] = dict(heldout_rows)
        block["heldout_e0_sha256_unchanged"] = all(row["e0_sha256_unchanged"] for row in heldout_rows.values())
        block["heldout_carrier_zeroed"] = all(row["carrier_all_zero"] for row in heldout_rows.values())
    return block


def ensure_frozen_recipe(args: argparse.Namespace) -> None:
    """The ablation keeps the frozen recipe verbatim: learned_slope/default/4-layer/seed 42/P16."""
    drifted = []
    if args.tier != "learned_slope":
        drifted.append("--tier learned_slope")
    if args.ladder != "default":
        drifted.append("--ladder default")
    if int(args.layers) != 4:
        drifted.append("--layers 4")
    if float(args.lr_multiplier) != 1.0:
        drifted.append("--lr-multiplier 1.0")
    if args.learn_flat_heads:
        drifted.append("--learn-flat-heads off")
    if args.cable_nw:
        drifted.append("--cable-nw off")
    if args.per_layer is not True:
        drifted.append("--per-layer")
    if args.half_lives is not None:
        drifted.append("no --half-lives override")
    if drifted:
        raise ValueError("identity ablation freezes the paired recipe; expected " + ", ".join(drifted))


def decoder_pairing(model: nn.Module, device: torch.device, recency_cfg) -> dict[str, Any]:
    """Structural init pairing against a same-seed stock P16 proj_add build.

    No external initialization SHA is pinned: the muscle formal reference is a
    different decoder family.  The teeth here are the byte-equality of shared
    named parameters against the same-seed stock build plus the same-ladder
    init-forward parity (both enforced inside ``template.assert_paired``).
    """
    stock = template.stock_decoder(device, recency_cfg.layers, PROJ_DIM)
    shared = assert_shared_byte_equal(model, stock)
    expected = template.shared_init_sha(model, shared)
    init = template.assert_paired(model, device, {"initialization_sha256": expected}, recency_cfg, PROJ_DIM)
    del stock
    return init


def assert_full_stream_parity(model: nn.Module, material: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Full-window versus token-by-token parity on a left-padded W100 context."""
    item = material[HO[0]]
    x = item["dataset"][0][0]
    raw = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(device)
    valid = frozen.valid_mask_from_padded_starts((97,), device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            full = model(raw, item["bank"], input_valid_mask=valid)
        stream = LearnableRiftStreamDecoder(model) if isinstance(model.temporal, LearnableRecencyTemporal) else RiftStreamDecoder(model)
        streamed = None
        for offset in range(CONTEXT):
            streamed = stream.stream_step(raw[:, offset], item["bank"], ["m1-projadd-ablation-parity"], valid_mask=valid[:, offset])
        if streamed is None or not torch.allclose(full, streamed, rtol=2e-5, atol=2e-5):
            raise RuntimeError("M1 muscle proj_add ablation full-window versus streaming parity failed")
        return {"status": "PASSED", "query_start": 97, "valid_bins": int(valid.sum())}
    finally:
        model.train(was_training)


def _checkpoint_payload(model, optimizer, ema, *, epoch: int, step: int, smoke: bool, meta: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA, "cell": meta["cell"], "identity": IDENTITY, "epoch": epoch, "global_step": step, "smoke": smoke,
            "tier": meta["tier"], "bias_mode": meta["bias_mode"], "new_parameter_names": meta["new_parameter_names"],
            "config": {"seed": SEED, "sampler_seed": FROZEN_SAMPLER_SEED, "context_bins": CONTEXT, "proj_dim": PROJ_DIM,
                       "epochs": EPOCHS, "batch": BATCH, "lr": LR, "bias_mode": meta["bias_mode"], "attention_backend": "local"},
            "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"],
            "carrier_binding": meta["carrier_binding"], "fit_sha256": meta["fit_sha256"],
            "shared_initialization_sha256": meta["initialization_pairing"]["shared_initialization_sha256"],
            "raw_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.state_dict(),
            "rng": frozen._rng_state(device)}


def _validate_checkpoint(path: Path, meta: Mapping[str, Any], *, expected_epoch: int | None = None) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"schema": CHECKPOINT_SCHEMA, "cell": meta["cell"], "identity": IDENTITY, "smoke": False, "tier": meta["tier"]}
    if any(state.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"checkpoint contract mismatch: {path}")
    if expected_epoch is not None and int(state.get("epoch", 0)) != expected_epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {path}")
    config = state.get("config", {})
    if config != {"seed": SEED, "sampler_seed": FROZEN_SAMPLER_SEED, "context_bins": CONTEXT, "proj_dim": PROJ_DIM,
                  "epochs": EPOCHS, "batch": BATCH, "lr": LR, "bias_mode": meta["bias_mode"], "attention_backend": "local"}:
        raise RuntimeError(f"checkpoint configuration mismatch: {path}")
    for key in ("source_hashes", "source_contract", "new_parameter_names", "carrier_binding", "fit_sha256"):
        if state.get(key) != meta.get(key):
            raise RuntimeError(f"checkpoint {key} mismatch: {path}")
    if state.get("shared_initialization_sha256") != meta["initialization_pairing"]["shared_initialization_sha256"]:
        raise RuntimeError(f"checkpoint shared initialization sha mismatch: {path}")
    return state


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    ensure_frozen_recipe(args)
    recency_cfg = config_from_args(args, "m1")
    smoke = args.max_updates_smoke is not None
    if not smoke and args.epochs != EPOCHS:
        raise ValueError("formal M1 muscle proj_add ablation requires exactly 24 epochs")
    if args.resume is not None:
        raise RuntimeError("resume is not used in this drop; rerun into a fresh --dest")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dest = args.dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("new ablation destination must be empty")
    carriers, binding = carrier_binding(args.carrier_pack)
    paired = m1_flat._paired_reference(args.paired_reference, arm=REFERENCE_ARM, binding=binding)
    dataset, sampler = frozen.legacy.build_fullsession_face()
    legacy_banks, legacy_report = frozen.legacy.build_fullsession_banks(dataset)
    banks_muscle = frozen._replace_carriers(legacy_banks, carriers, SOURCE)
    contract = frozen._source_contract(dataset, sampler, banks_muscle, {
        "legacy_frozen_activity_bank_report": legacy_report,
        "carrier_replacement": CARRIER_VARIANT,
        "actual_carrier_metadata": "source_contract.bank_hashes + carrier_binding",
    })
    source_calib = frozen.m1_plan.calib_trials_from_dataset(dataset)
    contract["raw_m10_calib_sha256"] = {name: frozen._array_sha(value) for name, value in source_calib.items()}
    if contract != paired["source_contract"]:
        raise RuntimeError("ablation source contract differs from paired muscle reference")
    banks_abl, source_rows = transform_banks(banks_muscle, IDENTITY)
    model = template.learnable_decoder(device, recency_cfg, PROJ_DIM)
    init = decoder_pairing(model, device, recency_cfg)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    # Identical optimizer recipe to the proj_add learnable template: shared
    # parameters keep the reference trainer's name-based no-decay grouping;
    # new recency params are a third group with wd=0 and the tier's lr multiplier.
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
    material_abl = None
    if smoke:
        material = frozen._ho_material(carriers)
        ho_original = frozen._ho_contract(material)
        if ho_original != paired["ho_contract"]:
            raise RuntimeError("ablation HO contract (pre-transform) differs from paired muscle reference")
        material_abl, ho_rows = transform_material(material, IDENTITY)
        smoke_preflight = {
            "initialization_pairing": init,
            "ho_contract_pre_transform": ho_original,
            "ho_contract": frozen._ho_contract(material_abl),
            "identity_transform": ho_rows,
            "streaming_parity": assert_full_stream_parity(model, material_abl, device),
        }
    else:
        ho_rows = None
    ablation_record = identity_ablation_block(IDENTITY, source_rows, ho_rows)
    cell = f"M1-MUSCLE-R100-D{recency_cfg.layers}-P{PROJ_DIM}-PROJADD-LEARNABLE-ABLATION-{args.identity.upper()}-V1"
    meta = {
        "schema": TRAIN_SCHEMA, "status": "SMOKE" if smoke else "FORMAL",
        "cell": cell,
        "task": "m1", "identity_interface": "proj_add", "variant": "learnable_recency",
        "identity": args.identity, "identity_ablation": ablation_record,
        "carrier_variant": CARRIER_VARIANT,
        "tier": recency_cfg.tier,
        "bias_mode": "recency" if recency_cfg.tier == "fixed" else f"learnable_{recency_cfg.tier}",
        "learnable_config": recency_cfg.__dict__,
        "new_parameter_names": init["new_parameter_names"],
        "new_parameter_counts": init["new_parameter_counts"],
        "trainable_new_parameter_count": init["trainable_new_parameter_count"],
        "seed": SEED, "sampler_seed": FROZEN_SAMPLER_SEED, "context_bins": CONTEXT, "query_pad_bins": QUERY_PAD_BINS,
        "proj_dim": PROJ_DIM, "layer_windows": list(recency_cfg.temporal_config.windows), "depth": recency_cfg.layers,
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
        "source_train_only_for_gradients": True,
        "development_surface": "visible HO3 calibration (3881 windows), post-training EMA scan only",
        "official_selection_metric": "equal-session mean channel-centered variance-weighted R2",
        "official_test_used": False,
        "source_hashes": source_hashes(),
        "runner_sha256": _sha_file(Path(__file__)),
        "source_contract": dict(contract),
        "carrier_binding": dict(binding), "fit_sha256": binding["fit_sha256"],
        "paired_reference": dict(paired),
        "launch": {"argv": sys.argv, "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "python_no_user_site": os.environ.get("PYTHONNOUSERSITE")},
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    frozen._atomic_json(dest / "run_meta.json", meta)
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
            lr = warmup_cosine_lr(step, total_steps=EPOCHS * UPDATES_PER_EPOCH, warmup_steps=UPDATES_PER_EPOCH,
                                  peak=LR, min_factor=plan.LR_MIN_FACTOR)
            apply_group_lrs(optimizer, lr)
            keep_rng = torch.Generator(device="cpu"); keep_rng.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
            keep = whole_unit_dropout(banks_abl[sessions[0]].unit_mask, p=0.1, generator=keep_rng)
            valid = torch.ones((len(x), CONTEXT), dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                prediction = model(x.float().to(device), banks_abl[sessions[0]], dropout_keep=keep, input_valid_mask=valid)
                loss = nn.functional.mse_loss(prediction.float(), y.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite ablation loss epoch={epoch} batch={batch_id}")
            loss.backward(); grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            optimizer.step(); ema.update_after_step(model)
            losses.append(float(loss.detach().cpu()))
            last_loss = losses[-1]
            last_x, last_bank, last_valid = x.float().to(device), banks_abl[sessions[0]], valid
            if step == 1 or step % 100 == 0:
                frozen._atomic_json(dest / "heartbeat.json", {"status": "SMOKE" if smoke else "TRAINING", "pid": os.getpid(),
                                                              "event": "step", "identity": IDENTITY, "epoch": epoch,
                                                              "global_step": step, "loss": losses[-1], "lr": lr,
                                                              "grad_norm": grad_norm,
                                                              "elapsed_seconds": time.monotonic() - started,
                                                              "utc": datetime.now(timezone.utc).isoformat()})
            if smoke and step >= args.max_updates_smoke:
                break
        last_bias = template.bias_snapshot(model, last_x, last_bank, last_valid)
        checkpoint = _checkpoint_payload(model, optimizer, ema, epoch=epoch, step=step, smoke=smoke, meta=meta, device=device)
        checkpoint_sha = frozen._atomic_checkpoint(dest / f"epoch_{epoch:03d}.pt", checkpoint)
        row = {"status": "SMOKE" if smoke else "TRAINING", "event": "epoch", "identity": IDENTITY, "epoch": epoch,
               "global_step": step, "train_mse": float(np.mean(losses)), "checkpoint_sha256": checkpoint_sha,
               "bias": last_bias, "utc": datetime.now(timezone.utc).isoformat()}
        frozen._atomic_json(dest / "heartbeat.json", row); frozen._append_jsonl(dest / "metrics.jsonl", row)
        if smoke:
            receipt = {
                "schema": TRAIN_SCHEMA, "status": "COMPLETED", "cell": meta["cell"], "identity": IDENTITY,
                "carrier_variant": CARRIER_VARIANT, "tier": recency_cfg.tier, "steps": step,
                "finite_loss": last_loss is not None and bool(np.isfinite(last_loss)),
                "identity_ablation": ablation_record,
                "carrier_binding": dict(binding), "fit_sha256": binding["fit_sha256"],
                "new_parameter_names": init["new_parameter_names"],
                "trainable_new_parameter_count": init["trainable_new_parameter_count"],
                "preflight": smoke_preflight,
                "postupdate": {"full_vs_stream": assert_full_stream_parity(model, material_abl, device)},
                "bias": last_bias, "checkpoint_sha256": checkpoint_sha,
                "cuda_initialized": torch.cuda.is_initialized(),
            }
            frozen._atomic_json(dest / "smoke_receipt.json", receipt)
            return {"status": "SMOKE_COMPLETED", "steps": step, "identity": IDENTITY}
        if len(losses) != UPDATES_PER_EPOCH:
            raise RuntimeError(f"epoch {epoch} had {len(losses)} updates, expected {UPDATES_PER_EPOCH}")
    if step != EPOCHS * UPDATES_PER_EPOCH:
        raise RuntimeError(f"formal total updates {step} != {EPOCHS * UPDATES_PER_EPOCH}")
    frozen._atomic_json(dest / "train_receipt.json", {
        "schema": TRAIN_SCHEMA, "status": "COMPLETED", "cell": meta["cell"], "identity": IDENTITY,
        "carrier_variant": CARRIER_VARIANT, "tier": recency_cfg.tier, "epochs": EPOCHS, "steps": step,
        "identity_ablation": {"identity": IDENTITY,
                              "transformed_e0_sha256": {s: row["transformed_e0_sha256"] for s, row in source_rows.items()},
                              "transformed_carrier_sha256": {s: row["transformed_carrier_sha256"] for s, row in source_rows.items()},
                              "e0_sha256_unchanged": ablation_record["e0_sha256_unchanged"],
                              "carrier_zeroed": ablation_record["carrier_zeroed"]},
        "carrier_binding": dict(binding), "fit_sha256": binding["fit_sha256"],
        "paired_reference": meta["paired_reference"],
        "post_training_scoring_required": True})
    return {"status": "TRAIN_COMPLETED", "steps": step}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    dest = args.dest.resolve()
    meta = frozen._read_json(dest / "run_meta.json")
    if (meta.get("status") != "FORMAL" or meta.get("schema") != TRAIN_SCHEMA or meta.get("task") != "m1"
            or meta.get("identity") != IDENTITY):
        raise RuntimeError("score requires matching formal M1 muscle proj_add activity_only ablation run")
    if int(meta.get("proj_dim", -1)) != PROJ_DIM:
        raise RuntimeError("score requires the frozen P16 ablation geometry")
    recency_cfg = config_from_run_meta(meta, "m1")
    carriers, binding = carrier_binding(args.carrier_pack)
    paired = m1_flat._paired_reference(args.paired_reference, arm=REFERENCE_ARM, binding=binding)
    if binding != meta.get("carrier_binding"):
        raise RuntimeError("score carrier binding differs from the training run")
    if paired != meta.get("paired_reference"):
        raise RuntimeError("score paired muscle reference differs from the training run")
    if meta.get("source_hashes") != source_hashes():
        raise RuntimeError("score source hashes differ from the ablation runner seal")
    if meta.get("source_contract", {}).get("total_windows") != EXPECTED_WINDOWS:
        raise RuntimeError("score source contract mismatch")
    receipt = frozen._read_json(dest / "train_receipt.json")
    required_receipt = {"status": "COMPLETED", "cell": meta["cell"], "identity": IDENTITY,
                        "epochs": EPOCHS, "steps": EPOCHS * UPDATES_PER_EPOCH}
    if any(receipt.get(key) != value for key, value in required_receipt.items()):
        raise RuntimeError("score requires a completed matching formal train receipt")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    model = template.rebuild_model(meta, device)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    material = frozen._ho_material(carriers)
    ho_original = frozen._ho_contract(material)
    if ho_original != paired["ho_contract"]:
        raise RuntimeError("score HO contract (pre-transform) differs from paired muscle reference")
    material_abl, ho_rows = transform_material(material, IDENTITY)
    ho_contract = frozen._ho_contract(material_abl)
    progress_path = dest / "score_progress.json"
    initial_progress = {"schema": PROGRESS_SCHEMA, "cell": meta["cell"], "identity": IDENTITY,
                        "source_hashes": meta["source_hashes"], "ho_contract": ho_contract,
                        "carrier_binding": meta["carrier_binding"], "fit_sha256": meta["fit_sha256"], "completed": {}}
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else initial_progress
    if any(progress.get(key) != initial_progress[key] for key in ("schema", "cell", "identity", "source_hashes", "ho_contract", "carrier_binding", "fit_sha256")):
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
            repeat = frozen._assert_ho_repeatable(model, ema, material_abl, device)
            repeat["full_vs_stream_parity"] = assert_full_stream_parity(model, material_abl, device)
        previous = completed.get(str(epoch))
        if previous is not None:
            if previous.get("checkpoint_sha256") != checkpoint_sha:
                raise RuntimeError(f"checkpoint hash drift after scored epoch {epoch}")
            frozen._validate_scored_report(previous.get("ema_ho_calib", {}))
            continue
        report = frozen._with_ema_eval(model, ema, lambda: frozen._ho_score(model, material_abl, device))
        frozen._validate_scored_report(report)
        completed[str(epoch)] = {"checkpoint_sha256": checkpoint_sha, "ema_ho_calib": report}
        frozen._atomic_json(progress_path, {**initial_progress, "status": "SCORING", "completed": completed,
                                            "last_completed_epoch": epoch, "repeatability": repeat})
        frozen._atomic_json(dest / "heartbeat.json", {"status": "SCORING", "event": "ho_calib_score", "epoch": epoch,
                                                      "identity": IDENTITY, "completed_epochs": len(completed),
                                                      "utc": datetime.now(timezone.utc).isoformat()})
    if set(completed) != {str(epoch) for epoch in range(1, EPOCHS + 1)}:
        raise RuntimeError("score scan did not produce exactly 24 epochs")
    scores = {epoch: row["ema_ho_calib"] for epoch, row in completed.items()}
    best = min(range(1, EPOCHS + 1), key=lambda epoch: (-scores[str(epoch)]["equal_session_mean_channel_variance_weighted_r2"], epoch))
    legacy_best = min(range(1, EPOCHS + 1), key=lambda epoch: (-scores[str(epoch)]["equal_session_mean"], epoch))
    score_receipt = {"schema": SCORE_SCHEMA, "status": "COMPLETED", "cell": meta["cell"], "identity": IDENTITY,
                     "carrier_variant": CARRIER_VARIANT, "tier": recency_cfg.tier,
                     "ema_by_epoch": scores,
                     "checkpoint_sha256_by_epoch": {epoch: completed[epoch]["checkpoint_sha256"] for epoch in sorted(completed, key=int)},
                     "selection": {"epoch": best, "metric": "channel_variance_weighted_r2",
                                   "rule": "earliest maximum equal-session mean EMA on visible HO3 calibration"},
                     "legacy_selection": {"epoch": legacy_best, "metric": "legacy_flattened_r2",
                                          "rule": "earliest maximum equal-session mean EMA on visible HO3 calibration"},
                     "repeatability": repeat, "ho_contract": ho_contract,
                     "ho_contract_pre_transform": ho_original,
                     "identity_ablation": identity_ablation_block(IDENTITY, meta["identity_ablation"]["source_sessions"], ho_rows),
                     "carrier_binding": meta["carrier_binding"], "fit_sha256": meta["fit_sha256"],
                     "paired_reference": meta["paired_reference"],
                     "official_test_used": False}
    frozen._atomic_json(dest / "score_receipt.json", score_receipt)
    frozen._atomic_json(dest / "ho_calib_epoch_scan.json", score_receipt)
    frozen._atomic_json(dest / "heartbeat.json", {"status": "COMPLETED", "event": "score_complete", "epoch": best,
                                                  "identity": IDENTITY, "completed_epochs": EPOCHS,
                                                  "utc": datetime.now(timezone.utc).isoformat()})
    return {"status": "SCORE_COMPLETED", "best_epoch": best}


def default_dest(args: argparse.Namespace) -> Path:
    return (RESULTS / "smoke" / DEFAULT_DEST_NAME) if args.max_updates_smoke is not None else (RESULTS / DEFAULT_DEST_NAME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(parser)
    parser.set_defaults(tier="learned_slope", ladder="default")
    parser.add_argument("--identity", choices=ABLATION_IDENTITIES, required=True,
                        help="ablation identity (this runner implements activity_only only)")
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--stage", choices=("train", "score"), default="train")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--carrier-pack", type=Path, default=DEFAULT_PACK,
                        help="frozen official muscle_response16_svd4/global_rms carrier pack NPZ")
    parser.add_argument("--paired-reference", type=Path, default=DEFAULT_REFERENCE,
                        help="completed muscle formal run whose data/HO contracts pin this arm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > EPOCHS:
        parser.error("--epochs must be 1..24")
    if args.max_updates_smoke is not None and (args.max_updates_smoke < 1 or args.stage != "train"):
        parser.error("smoke requires positive updates and --stage train")
    if args.stage == "score" and args.max_updates_smoke is not None:
        parser.error("smoke is a train-stage-only mode")
    try:
        ensure_frozen_recipe(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dest is None:
        args.dest = default_dest(args)
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    result = run_train(args) if args.stage == "train" else run_score(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
