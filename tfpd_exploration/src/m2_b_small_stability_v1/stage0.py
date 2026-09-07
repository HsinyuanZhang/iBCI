"""Stage0 technical gates. CPU only. Does not launch 24-epoch training."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tfpd_exploration.src.m2_dual_track_v1 import plan
from tfpd_exploration.src.m2_dual_track_v1.champion import sha256_file
from tfpd_exploration.src.m2_dual_track_v1.contracts import SessionBank, make_stub_bank
from tfpd_exploration.src.m2_dual_track_v1.sampler import load_manifest

from . import config as cfg
from .decoder import SmallTransformerDecoder, count_decoder_parameters, init_state_sha256
from .ema import DecoderEMA
from .report import file_sha256, write_historical_trajectory_summary
from .training import (
    TrainState,
    capture_rng,
    disposable_profile_hook,
    disposable_smoke_steps,
    load_checkpoint,
    lr_at_update,
    restore_rng,
    run_optimizer_step,
    save_checkpoint,
    unit_dropout_mask,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _code_shas() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    rows = {}
    for path in sorted(package.glob("*.py")):
        rows[path.name] = file_sha256(path)
    return rows


def _window_id_digest() -> dict[str, Any]:
    meta_path = cfg.OLD_ROOT / "cache" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ids = meta["frozen_ext4_window_ids"]
    payload = {session: ids[session] for session in plan.EXT4_SESSIONS}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    counts = {session: len(payload[session]) for session in plan.EXT4_SESSIONS}
    return {
        "digest": digest,
        "counts": counts,
        "total": int(sum(counts.values())),
        "expected": dict(plan.EXT4_EXPECTED_WINDOWS),
        "match_expected": counts == dict(plan.EXT4_EXPECTED_WINDOWS),
        "meta_sha256": file_sha256(meta_path),
    }


def gate_authority_bind() -> dict[str, Any]:
    workorder = cfg.REPO_ROOT / cfg.WORKORDER_RELATIVE
    ckpt = cfg.REPO_ROOT / plan.CHAMPION_CKPT_RELATIVE
    head = cfg.REPO_ROOT / plan.SELECTED_HEAD_RELATIVE
    film = cfg.REPO_ROOT / plan.FILM_STATES_RELATIVE
    split = cfg.REPO_ROOT / plan.M33_SPLIT_MANIFEST_RELATIVE
    normalizer = cfg.OLD_ROOT / "cache" / "move_t4_normalizer.json"
    ref = cfg.OLD_ROOT / "stage0" / "ref_clean.json"
    manifest = load_manifest(cfg.OLD_MANIFEST_24_PATH)
    ref_payload = json.loads(ref.read_text(encoding="utf-8"))
    film_keys = []
    if film.is_file():
        states = torch.load(film, map_location="cpu", weights_only=False)
        film_keys = sorted(str(key) for key in states)
    windows = _window_id_digest()
    bank_sessions = sorted(
        (cfg.OLD_ROOT / "cache" / "source_train").glob("ses-*/provenance.json")
    )
    bank_e0 = {}
    for path in bank_sessions:
        prov = json.loads(path.read_text(encoding="utf-8"))
        bank_e0[path.parent.name] = {
            "e0_sha256": prov.get("e0_sha256"),
            "eligible_start_sha256": prov.get("eligible_start_sha256"),
        }
    rows = {
        "workorder_sha256": file_sha256(workorder),
        "workorder_match": file_sha256(workorder) == cfg.WORKORDER_SHA256,
        "code_shas": _code_shas(),
        "sampler_path": str(cfg.OLD_MANIFEST_24_PATH.relative_to(cfg.REPO_ROOT)),
        "sampler_digest": manifest["digest"],
        "sampler_digest_match": manifest["digest"] == cfg.MANIFEST_24_DIGEST,
        "parent_12_digest": manifest.get("parent_12_digest"),
        "parent_12_match": manifest.get("parent_12_digest") == cfg.MANIFEST_12_DIGEST,
        "updates_per_epoch": len(manifest["batches"]["1"]),
        "total_updates": sum(len(manifest["batches"][str(e)]) for e in range(1, 25)),
        "champion_ckpt_sha256": sha256_file(ckpt) if ckpt.is_file() else None,
        "champion_ckpt_match": ckpt.is_file() and sha256_file(ckpt) == cfg.CHAMPION_CKPT_SHA256,
        "selected_head_exists": head.is_file(),
        "film_states_sha256": file_sha256(film) if film.is_file() else None,
        "film_has_p0": "p0" in film_keys,
        "film_keys": film_keys,
        "split_manifest_exists": split.is_file(),
        "normalizer_sha256": file_sha256(normalizer),
        "normalizer_inherited_m33": json.loads(normalizer.read_text(encoding="utf-8")).get(
            "inherited_m33_fold_normalizer"
        ),
        "ref_clean_sha256": file_sha256(ref),
        "R_REF_session": ref_payload.get("R_REF_clean"),
        "R_REF_match": abs(float(ref_payload["R_REF_clean"]) - cfg.R_REF_SESSION) < 1e-12,
        "window_ids": windows,
        "bank_e0": bank_e0,
        "heldin_sessions": list(plan.HELDIN_SESSIONS),
        "no_new_train_sessions": list(plan.HELDIN_SESSIONS)
        == [
            "ses-2020-10-19-Run1",
            "ses-2020-10-19-Run2",
            "ses-2020-10-20-Run1",
            "ses-2020-10-20-Run2",
            "ses-2020-10-27-Run1",
            "ses-2020-10-27-Run2",
            "ses-2020-10-28-Run1",
        ],
        "old_plan_temporal_width_untouched": plan.B_TEMPORAL_WIDTH == 512,
        "new_train_sessions": False,
    }
    rows["pass"] = bool(
        rows["workorder_match"]
        and rows["sampler_digest_match"]
        and rows["parent_12_match"]
        and rows["updates_per_epoch"] == 3165
        and rows["total_updates"] == 75960
        and rows["champion_ckpt_match"]
        and rows["film_has_p0"]
        and rows["R_REF_match"]
        and rows["window_ids"]["match_expected"]
        and rows["window_ids"]["total"] == 2069
        and rows["no_new_train_sessions"]
        and rows["old_plan_temporal_width_untouched"]
        and rows["normalizer_inherited_m33"] is False
    )
    return rows


def gate_param_count() -> dict[str, Any]:
    model = SmallTransformerDecoder(seed=42)
    counts = count_decoder_parameters(model)
    width_ok = (
        model.frontend.slot_proj.out_features == 256
        and model.frontend.slot_proj.in_features == 2048
        and model.readout[0].in_features == 256
        and model.temporal.blocks[0].attn.qkv.in_features == 256
        and model.temporal.pe.shape[-1] == 256
    )
    rows = {
        **counts,
        "width_ok": width_ok,
        "output_units": 2,
        "pass": (
            counts["decoder_total"] == cfg.DECODER_PARAMS
            and counts["temporal"] == cfg.TEMPORAL_PARAMS
            and counts["frontend_plus_readout"] == cfg.FRONTEND_READOUT_PARAMS
            and counts["decoder_plus_shared_calib"] == cfg.DECODER_PLUS_CALIB_PARAMS
            and width_ok
        ),
    }
    return rows


def _permute_bank(bank: SessionBank, perm: torch.Tensor) -> SessionBank:
    return SessionBank(
        session_id=bank.session_id,
        support_trial_ids=bank.support_trial_ids,
        raw_trial_ids=bank.raw_trial_ids,
        X_store=bank.X_store,
        target_store=bank.target_store,
        eligible_starts=bank.eligible_starts,
        E0=bank.E0[perm],
        T=bank.T[perm],
        unit_mask=bank.unit_mask[perm],
        provenance=dict(bank.provenance),
        frozen_u=None if bank.frozen_u is None else bank.frozen_u[:, perm],
    )


def gate_permutation_causal() -> dict[str, Any]:
    model = SmallTransformerDecoder(seed=42).eval()
    bank = make_stub_bank(seed=4)
    x = torch.randn(2, cfg.SMALL.window, plan.CHANNELS)
    perm = torch.randperm(plan.CHANNELS)
    y = model.forward_last(x, bank, bank.unit_mask)
    y_p = model.forward_last(x[:, :, perm], _permute_bank(bank, perm), bank.unit_mask[perm])
    perm_ok = bool(torch.allclose(y, y_p, atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL))
    x_future = x.clone()
    x_future[:, 26:] += 4.0
    h = model.forward_hidden(x, bank, bank.unit_mask)
    h_f = model.forward_hidden(x_future, bank, bank.unit_mask)
    causal_ok = bool(torch.allclose(h[:, :26], h_f[:, :26], atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL))
    prefix = model.forward_hidden(x[:, :26], bank, bank.unit_mask)
    prefix_ok = bool(torch.allclose(prefix, h[:, :26], atol=plan.PERM_ATOL, rtol=plan.PERM_RTOL))
    y_shape = list(y.shape)
    return {
        "permutation_ok": perm_ok,
        "causal_ok": causal_ok,
        "prefix_ok": prefix_ok,
        "output_shape": y_shape,
        "pass": perm_ok and causal_ok and prefix_ok and y_shape == [2, 2],
    }


def gate_init_and_dropout() -> dict[str, Any]:
    a = SmallTransformerDecoder(seed=42)
    b = SmallTransformerDecoder(seed=42)
    sha = init_state_sha256(a)
    sha_b = init_state_sha256(b)
    mask = torch.ones(4, plan.CHANNELS, dtype=torch.bool)
    m0 = unit_dropout_mask(mask, seed=42, epoch=3, batch_id=17)
    m1 = unit_dropout_mask(mask, seed=42, epoch=3, batch_id=17)
    formal = capture_rng()
    smoke = disposable_smoke_steps(n_steps=2)
    after = capture_rng()
    restore_rng(formal)
    return {
        "init_sha": sha,
        "init_sha_pair_match": sha == sha_b,
        "dropout_pair_match": bool(torch.equal(m0, m1)),
        "smoke": smoke,
        "formal_rng_unchanged": bool(torch.equal(after["torch"], formal["torch"])),
        "pass": sha == sha_b and bool(torch.equal(m0, m1)) and smoke["advanced_formal_rng"] is False,
    }


def gate_lr_ema() -> dict[str, Any]:
    s0_1 = lr_at_update(cfg.CELL_S0, 1)
    s0_w = lr_at_update(cfg.CELL_S0, cfg.UPDATES_PER_EPOCH)
    s0_w1 = lr_at_update(cfg.CELL_S0, cfg.UPDATES_PER_EPOCH + 1)
    s0_t = lr_at_update(cfg.CELL_S0, cfg.TOTAL_UPDATES)
    s1_t = lr_at_update(cfg.CELL_S1, cfg.TOTAL_UPDATES)
    model = torch.nn.Linear(2, 2)
    ema = DecoderEMA(model, decay=0.9995)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update_after_step(model)
    first_copy = bool(torch.equal(ema.shadow["weight"], model.weight.detach().float()))
    aliased = ema.shadow["weight"].data_ptr() == model.weight.data_ptr()
    return {
        "s0_t1": s0_1,
        "s0_tW": s0_w,
        "s0_tW1": s0_w1,
        "s0_tT": s0_t,
        "s1_tT": s1_t,
        "first_copy": first_copy,
        "not_aliased": not aliased,
        "pass": (
            abs(s0_1 - cfg.LR_MAX / cfg.UPDATES_PER_EPOCH) < 1e-15
            and abs(s0_w - cfg.LR_MAX) < 1e-15
            and abs(s0_w1 - cfg.LR_MAX) < 1e-15
            and abs(s0_t - 0.1 * cfg.LR_MAX) < 1e-15
            and abs(s1_t - cfg.LR_MIN) < 1e-15
            and first_copy
            and not aliased
        ),
    }


def gate_raw_step_invariant() -> dict[str, Any]:
    bank = make_stub_bank(seed=7)
    x = torch.randn(2, cfg.SMALL.window, plan.CHANNELS)
    y = torch.randn(2, 2)
    torch.manual_seed(0)
    a = SmallTransformerDecoder(seed=42)
    b = SmallTransformerDecoder(seed=42)
    opt_a = torch.optim.AdamW(list(a.parameters()), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    opt_b = torch.optim.AdamW(list(b.parameters()), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    opt_b.load_state_dict(opt_a.state_dict())
    ema = DecoderEMA(a, decay=0.9995)
    run_optimizer_step(b, opt_b, x, y, bank, cell=cfg.CELL_S1, t=1, seed=42, epoch=1, batch_id=0, ema=None)
    run_optimizer_step(a, opt_a, x, y, bank, cell=cfg.CELL_S1, t=1, seed=42, epoch=1, batch_id=0, ema=ema)
    raw_match = all(torch.equal(p1, p2) for (_, p1), (_, p2) in zip(a.named_parameters(), b.named_parameters()))
    raw_before = {k: v.detach().clone() for k, v in a.named_parameters()}
    ema.score_with_ema(a, lambda m: m.forward_last(x, bank, bank.unit_mask))
    restored = all(torch.equal(p, raw_before[n]) for n, p in a.named_parameters())
    return {"raw_match": raw_match, "ema_score_restores_raw": restored, "pass": raw_match and restored}


def gate_interrupt_resume() -> dict[str, Any]:
    bank = make_stub_bank(seed=7)
    x = torch.randn(2, cfg.SMALL.window, plan.CHANNELS)
    y = torch.randn(2, 2)
    torch.manual_seed(11)
    model = SmallTransformerDecoder(seed=42)
    opt = torch.optim.AdamW(list(model.parameters()), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    ema = DecoderEMA(model, decay=0.9995)
    state = TrainState(cell=cfg.CELL_S0, seed=42, manifest_digest=cfg.MANIFEST_24_DIGEST)
    run_optimizer_step(model, opt, x, y, bank, cell=cfg.CELL_S0, t=1, seed=42, epoch=1, batch_id=0, ema=ema, state=state)
    ckpt = save_checkpoint(model, opt, ema, state)
    model_u = SmallTransformerDecoder(seed=42)
    model_u.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    opt_u = torch.optim.AdamW(list(model_u.parameters()), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    opt_u.load_state_dict(opt.state_dict())
    ema_u = DecoderEMA(model_u, decay=0.9995)
    ema_u.load_checkpoint_state(ema.checkpoint_state())
    rng = capture_rng()
    run_optimizer_step(model_u, opt_u, x, y, bank, cell=cfg.CELL_S0, t=2, seed=42, epoch=1, batch_id=1, ema=ema_u)
    model_r = SmallTransformerDecoder(seed=7)
    opt_r = torch.optim.AdamW(list(model_r.parameters()), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    ema_r = DecoderEMA(model_r, decay=0.9995)
    state_r = TrainState(cell=cfg.CELL_S0, seed=42, manifest_digest=cfg.MANIFEST_24_DIGEST)
    load_checkpoint(ckpt, model_r, opt_r, ema_r, state_r)
    restore_rng(rng)
    run_optimizer_step(model_r, opt_r, x, y, bank, cell=cfg.CELL_S0, t=2, seed=42, epoch=1, batch_id=1, ema=ema_r)
    bitwise = all(torch.equal(p1, p2) for (_, p1), (_, p2) in zip(model_u.named_parameters(), model_r.named_parameters()))
    close = all(
        torch.allclose(p1, p2, atol=1e-6, rtol=1e-5)
        for (_, p1), (_, p2) in zip(model_u.named_parameters(), model_r.named_parameters())
    )
    return {
        "bitwise": bitwise,
        "close_atol_1e6": close,
        "numeric_policy": "prefer_bitwise; fallback atol=1e-6 rtol=1e-5 documented if needed",
        "pass": bitwise or close,
    }


def gate_profile_hook() -> dict[str, Any]:
    return {
        "implemented": callable(disposable_profile_hook),
        "launched_on_gpu": False,
        "pass": callable(disposable_profile_hook),
        "note": "100-step GPU profile hook is implemented in training.disposable_profile_hook; Stage0 does not launch it",
    }


def run_stage0(*, root: Path | None = None) -> dict[str, Any]:
    dest = Path(root) if root is not None else cfg.ACTIVE_RUN_ROOT
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "S0_SMALL_LEGACY" / "seed42").mkdir(parents=True, exist_ok=True)
    (dest / "S1_SMALL_COS" / "seed42").mkdir(parents=True, exist_ok=True)
    historical = write_historical_trajectory_summary(dest)
    gates = {
        "authority_bind": gate_authority_bind(),
        "param_count": gate_param_count(),
        "permutation_causal": gate_permutation_causal(),
        "init_and_dropout_pairing": gate_init_and_dropout(),
        "lr_ema": gate_lr_ema(),
        "raw_step_invariant": gate_raw_step_invariant(),
        "interrupt_resume": gate_interrupt_resume(),
        "profile_hook_present": gate_profile_hook(),
    }
    ok = all(block.get("pass") for block in gates.values())
    report = {
        "schema": cfg.SCHEMA,
        "status": "READY" if ok else "BLOCKED",
        "decoder_params": gates["param_count"]["decoder_total"],
        "temporal_params": gates["param_count"]["temporal"],
        "frontend_plus_readout": gates["param_count"]["frontend_plus_readout"],
        "primary_candidate": cfg.PRIMARY_CANDIDATE,
        "formal_training_launched": False,
        "gates": gates,
        "profile_hook": {
            "implemented": True,
            "launched_on_gpu": False,
        },
        "historical_last_k": {
            "transformer_source_pick": historical["transformer"]["source_pick_ext4"],
            "mamba_same_surface_gap": historical["mamba"]["same_surface_gap"],
        },
        "finished": datetime.now(timezone.utc).isoformat(),
        "root": str(dest),
    }
    _write_json(dest / "stage0.json", report)
    _write_json(
        dest / "parent_ref.json",
        {
            "old_root": str(cfg.OLD_ROOT.relative_to(cfg.REPO_ROOT)),
            "old_root_read_only": True,
            "analysis_sha256": cfg.OLD_ANALYSIS_SHA256,
            "comparison_csv_sha256": cfg.OLD_COMPARISON_SHA256,
            "manifest_24_path": str(cfg.OLD_MANIFEST_24_PATH.relative_to(cfg.REPO_ROOT)),
            "manifest_24_digest": cfg.MANIFEST_24_DIGEST,
            "appended_to_old_root": False,
        },
    )
    _write_json(dest / "sampler_ref.json", {
        "referenced_not_rewritten": True,
        "path": str(cfg.OLD_MANIFEST_24_PATH.relative_to(cfg.REPO_ROOT)),
        "digest": cfg.MANIFEST_24_DIGEST,
        "parent_12_digest": cfg.MANIFEST_12_DIGEST,
    })
    return report
