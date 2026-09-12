#!/usr/bin/env python3
"""B3S two-stage FULL arm: unified activity trunk E0 + real task carrier T.

B3S = the bare ``pre_pool -> trial mean -> post_pool`` activity trunk of the
M1 method family (``learnable_recency_v1.activity_model``), calibrated from
activity, with each task's original T carrier travelling the independent
token channel of the existing proj_add frontend (``local conv + e0_proj(E0)
+ carrier concat``); there is no FiLM and no TaskBank.

Stage 1 (``--stage 1``, ``stage: 1_pretain``) jointly trains trunk+decoder
with the real carrier injected.  With seed 42 / P16 / default ladder it is
same-instance subtractable against the completed ACT v2 arm (identical seeds,
sampler, construction; only the carrier differs), so FULL-ACT isolates the
carrier contribution.

Stage 2 (``--stage 2 --freeze-trunk-from ...``, ``stage: 2_b3s``) loads the
trunk weights from the selected checkpoint of a stage-1 run (any
activity-joint run with ``identity_encoder.*`` state works, including the ACT
v2 schema), freezes them (eval + requires_grad False, excluded from the
optimizer and the EMA shadow), and retrains a fresh decoder under the same
carrier injection.  Seeds act on the decoder side and sampling only; for M2
the frozen seed-42 24x3165 manifest stays the sampler at every seed and this
is recorded in run_meta.

Trunk routes (``--trunk-side``): ``none`` (default) is the bare B3S trunk
above; ``concat`` additionally feeds the same frozen carrier into the trunk at
the post_pool entrance — ``E0 = post_pool(cat(pre_pool(activity).mean(trials),
carrier))``, the H1 C2 ``fused_identity`` semantics — while the token-channel
carrier concat stays in place.  Both stages and all binding checks
(checkpoint side match for stage 2) apply unchanged to either route.

Carrier sources (all frozen, byte-asserted against their provenance):
  - M2 source/minival: dual-track cache ``SessionBank.T`` MOVE-T4 [96,4]
    (exactly ``adapters.build_m2_bank``'s carrier); M2 EXT6: per-session
    ``T.npy`` of the frozen raw-M33 directory (the file
    ``m2_ext6_epoch_pick.load_query_pair`` reads).
  - M1 (held ready): the official muscle pack
    ``results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz`` [64,4],
    loaded through the frozen fail-closed ``load_carrier_pack`` binding.
  - H1: per-session constant m3 profile of the 27-tag signed_state14 plan at
    the ACT v2 support set (first three eval-valid trials, asserted equal to
    ``activity_data``'s support ``trial_ids``); the per-(session,start,budget)
    rotation of the signed-state mainline is intentionally not reproduced and
    this concession is disclosed in run_meta.  The HO-M3 face uses its 27-tag
    bank payload carrier keyed by falcon group.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PKG, ROOT, WS = HERE.parent, HERE.parent.parent, HERE.parent.parent.parent
V1 = WS / "btransform_unified_v1"
for path in (HERE, PKG / "src", ROOT / "src", ROOT, V1 / "src", WS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import activity_data
import activity_train as act
from btransform_unified_v1 import plan as v1_plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v1.schedule import warmup_cosine_lr
from learnable_recency_v1.activity_model import ActivityIdentityEncoder, ActivityLearnableRiftDecoder
from learnable_recency_v1.config import add_learnable_flags, config_from_args
from learnable_recency_v1.wrap import apply_group_lrs, split_optimizer_parameters
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan

SEED, BATCH = 42, 32
TASK_EPOCHS = act.TASK_EPOCHS
M2_UPDATES = act.M2_UPDATES
SCHEMA = "b3s_full_train_v1"
CKPT_SCHEMA = "b3s_full_epoch_checkpoint_v1"
H1_BANKS_DEFAULT = ROOT / "results/h1_signed_state_r300_v1/banks_official13_20260909"
M1_CARRIER_PACK_DEFAULT = ROOT / "results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz"
ACT_CKPT_SCHEMAS = ("activity_joint_early_pool_epoch_checkpoint_v2", CKPT_SCHEMA)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _ah(a: Any) -> str:
    x = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(x.dtype.str.encode())
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def _atom(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _append(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), HERE / "activity_full_score.py", *list(act.source_hashes())]
    seen: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        if p.is_file() and str(p) not in seen:
            seen[str(p)] = _sha(p)
    return seen


# ---------------------------------------------------------------------------
# Model: the ACT v2 decoder with the zero carrier replaced by a real task T.
# ---------------------------------------------------------------------------

TensorLike = torch.Tensor  # documentation alias used by _carrier_rows

CONCAT_SIDE_SEED_OFFSET = 0x434F4E43  # "CONC": isolated RNG domain for the concat post_pool


def _concat_three_linear_post_pool(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    """Same three-affine stack as the bare trunk, widened for the carrier join."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ConcatSideTrunk(ActivityIdentityEncoder):
    """Concat route trunk: ``E0 = post_pool(cat(pre_pool(activity).mean(trials), carrier))``.

    Exactly the H1 C2 materializer semantics
    (``tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_calibration.py::
    fused_identity``): per-trial ``pre_pool`` features are averaged over
    trials, the frozen ``[N,4]`` carrier is concatenated at the post_pool
    entrance (hidden+4 dims; H1: 32+4=36), and the widened three-affine
    ``post_pool`` produces E0.  ``pre_pool`` keeps the bare trunk's
    construction and RNG domain (identical bytes at the same seed);
    ``post_pool`` is rebuilt under a fresh forked offset.  The carrier enters
    detached (frozen input, never trained); ``activity_model.py`` is not
    modified.
    """

    def __init__(self, input_bins: int, output_dim: int, hidden_dim: int = 64, seed: int = 42) -> None:
        super().__init__(input_bins, output_dim, hidden_dim=hidden_dim, seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + CONCAT_SIDE_SEED_OFFSET)
            self.post_pool = _concat_three_linear_post_pool(hidden_dim + 4, output_dim, hidden_dim)

    def forward(self, activity: torch.Tensor, trial_mask: torch.Tensor | None = None,
                carrier: torch.Tensor | None = None) -> torch.Tensor:
        if activity.ndim != 3 or activity.shape[0] < 1 or activity.shape[1] != self.input_bins or activity.shape[2] < 1:
            raise ValueError(f"activity must be nonempty [M,{self.input_bins},N], got {tuple(activity.shape)}")
        if not torch.is_floating_point(activity):
            activity = activity.to(dtype=torch.float32)
        if not torch.isfinite(activity).all():
            raise ValueError("activity contains non-finite values")
        trials, _bins, units = activity.shape
        if trial_mask is None:
            mask = torch.ones(trials, dtype=torch.bool, device=activity.device)
        else:
            if trial_mask.dtype != torch.bool or trial_mask.shape != (trials,):
                raise ValueError("trial_mask must be bool [M]")
            mask = trial_mask.to(device=activity.device)
        if not bool(mask.any()):
            raise ValueError("trial_mask selects no support trials")
        features = self.pre_pool(activity.permute(0, 2, 1))
        weights = mask.to(dtype=features.dtype).view(trials, 1, 1)
        pooled = (features * weights).sum(dim=0) / weights.sum(dim=0)
        if carrier is None:
            side = torch.zeros(units, 4, dtype=pooled.dtype, device=pooled.device)
        else:
            if not torch.is_tensor(carrier):
                carrier = torch.as_tensor(np.asarray(carrier, np.float32))
            if carrier.ndim != 2 or tuple(carrier.shape) != (units, 4):
                raise ValueError(f"concat carrier must be [{units},4], got {tuple(carrier.shape)}")
            side = carrier.to(device=pooled.device, dtype=pooled.dtype).detach()
        identity = self.post_pool(torch.cat((pooled, side), dim=-1))
        if identity.shape != (units, self.output_dim):
            raise RuntimeError("concat identity output geometry drift")
        return identity


class B3SFullRiftDecoder(ActivityLearnableRiftDecoder):
    """proj_add RIFT decoder: activity-trunk E0 plus an injectable frozen T.

    ``carrier=None`` keeps the literal-zero activity carrier buffer and is
    bit-identical to the ACT v2 decoder; a real ``[N,4]`` tensor travels the
    same token channel as ``TaskBank.carrier`` in the stock frontend.  With
    ``trunk_state`` the pretrained trunk weights are loaded and frozen
    (requires_grad False, eval-only, outside optimizer/EMA).
    """

    def __init__(
        self,
        task: str,
        cfg: Any,
        *,
        context_bins: int | None = None,
        seed: int = 42,
        support_bins: int | None = None,
        identity_hidden: int | None = None,
        trunk_state: Mapping[str, torch.Tensor] | None = None,
        trunk_side: str = "none",
    ) -> None:
        super().__init__(
            task,
            cfg,
            context_bins=context_bins,
            seed=seed,
            support_bins=support_bins,
            identity_hidden=identity_hidden,
        )
        if trunk_side not in ("none", "concat"):
            raise ValueError("trunk_side must be 'none' or 'concat'")
        self.trunk_side = trunk_side
        if trunk_side == "concat":
            # Replace only the trunk; the discarded bare encoder lived in its own
            # forked RNG domain, so decoder bytes are untouched.
            self.identity_encoder = ConcatSideTrunk(
                self.support_bins, self.geometry["e0_dim"], hidden_dim=self.identity_hidden, seed=seed
            )
        self.frozen_trunk = False
        if trunk_state is not None:
            self.load_frozen_trunk(trunk_state)

    def load_frozen_trunk(self, trunk_state: Mapping[str, torch.Tensor]) -> None:
        """Overwrite the trunk with pretrained weights and freeze it."""
        incompatible = self.identity_encoder.load_state_dict(dict(trunk_state), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"trunk checkpoint key mismatch: {incompatible}")
        self.freeze_trunk()

    def freeze_trunk(self) -> None:
        for p in self.identity_encoder.parameters():
            p.requires_grad_(False)
        self.frozen_trunk = True
        self.identity_encoder.eval()

    def train(self, mode: bool = True) -> "B3SFullRiftDecoder":
        super().train(mode)
        if self.frozen_trunk:
            self.identity_encoder.eval()
        return self

    def trunk_parameter_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, p in sorted(self.identity_encoder.named_parameters()):
            digest.update(name.encode())
            digest.update(p.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def _carrier_rows(self, carrier: Any, local: TensorLike) -> torch.Tensor:
        if not torch.is_tensor(carrier):
            carrier = torch.as_tensor(np.asarray(carrier, np.float32))
        if carrier.ndim != 2 or tuple(carrier.shape) != (self.units, 4):
            raise ValueError(f"carrier must be [{self.units},4], got {tuple(carrier.shape)}")
        carrier = carrier.to(device=local.device, dtype=local.dtype)
        return carrier.unsqueeze(0).expand(local.shape[0], -1, -1)

    def _fuse_local_with_carrier(self, local, identity, keep, carrier) -> torch.Tensor:
        if carrier is None:
            return self._fuse_activity_local(local, identity, keep)
        return self._fuse_batched_local(local, identity, self._carrier_rows(carrier, local), keep)

    def frontend_tokens(
        self,
        x,
        identity,
        unit_mask=None,
        dropout_generator=None,
        dropout_keep=None,
        carrier=None,
    ):
        x = self._check_input(x, None)
        identity = self._identity_batch(identity, x.shape[0], x.device, x.dtype)
        keep = self._resolve_activity_keep(unit_mask, x.shape[0], x.device, dropout_generator, dropout_keep)
        return self._fuse_local_with_carrier(self.frontend.local_conv(x), identity, keep, carrier)

    def frontend_last(self, raw5, identity, unit_mask=None, carrier=None):
        if raw5.ndim != 3 or raw5.shape[1:] != (5, self.units):
            raise ValueError(f"raw5 must be [B,5,{self.units}]")
        raw5 = self._check_input(raw5, None)
        identity = self._identity_batch(identity, raw5.shape[0], raw5.device, raw5.dtype)
        keep = self._resolve_activity_keep(unit_mask, raw5.shape[0], raw5.device, None, None)
        batch = raw5.shape[0]
        flat = raw5.permute(0, 2, 1).reshape(batch * self.units, 1, 5)
        conv = self.frontend.local_conv
        local = conv.act(conv.conv(flat)).reshape(batch, self.units, 16, 1).permute(0, 3, 1, 2)
        return self._fuse_local_with_carrier(local, identity, keep, carrier)[:, 0]

    def encode_activity(self, activity, trial_mask=None, carrier=None):
        """Encode support with gradients intact; the concat route also eats T."""
        if self.trunk_side == "concat":
            return self.identity_encoder(activity, trial_mask, carrier)
        if carrier is not None:
            raise ValueError("the bare trunk takes no carrier into E0; use --trunk-side concat")
        return super().encode_activity(activity, trial_mask)

    def calibrate(self, activity, trial_mask=None, carrier=None):
        """Produce a detached, label-free identity only in evaluation mode."""
        if self.training:
            raise RuntimeError("calibrate is eval-only; call model.eval()")
        with torch.no_grad():
            return self.encode_activity(activity, trial_mask, carrier).detach()

    def forward_scores(
        self,
        x,
        activity=None,
        *,
        identity=None,
        unit_mask=None,
        dropout_keep=None,
        input_valid_mask=None,
        trial_mask=None,
        carrier=None,
    ):
        if (activity is None) == (identity is None):
            raise ValueError("provide exactly one of activity or identity")
        if identity is not None and trial_mask is not None:
            raise ValueError("trial_mask is only valid with activity")
        x = self._check_input(x, input_valid_mask)
        e0 = (self.encode_activity(activity, trial_mask, carrier=(carrier if self.trunk_side == "concat" else None))
              if activity is not None else identity)
        assert e0 is not None
        if input_valid_mask is not None:
            x = torch.where(input_valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
        z = self.frontend_tokens(x, e0, unit_mask=unit_mask, dropout_keep=dropout_keep, carrier=carrier)
        hidden = self.temporal(z, input_valid_mask)
        return self.readout(self.final_norm(hidden))

    def forward(
        self,
        x,
        activity=None,
        *,
        identity=None,
        unit_mask=None,
        dropout_keep=None,
        input_valid_mask=None,
        trial_mask=None,
        carrier=None,
    ):
        return self.forward_scores(
            x,
            activity,
            identity=identity,
            unit_mask=unit_mask,
            dropout_keep=dropout_keep,
            input_valid_mask=input_valid_mask,
            trial_mask=trial_mask,
            carrier=carrier,
        )[:, -1]


# ---------------------------------------------------------------------------
# Trunk checkpoint resolution (stage 2).
# ---------------------------------------------------------------------------


def extract_trunk_state(state: Mapping[str, Any], view: str) -> dict[str, torch.Tensor]:
    if state.get("schema") not in ACT_CKPT_SCHEMAS:
        raise RuntimeError(f"not an activity-joint checkpoint: {state.get('schema')}")
    prefix = "identity_encoder."
    if view == "raw":
        source = state["raw_state_dict"]
    elif view == "ema":
        source = state.get("ema", {}).get("shadow")
        if source is None:
            raise RuntimeError("checkpoint carries no EMA shadow")
    else:
        raise ValueError("view must be 'raw' or 'ema'")
    trunk = {k[len(prefix):]: v.detach().clone().to(torch.float32) for k, v in source.items() if k.startswith(prefix)}
    if not trunk:
        raise RuntimeError("checkpoint contains no identity_encoder weights")
    return trunk


def checkpoint_trunk_side(state: Mapping[str, Any]) -> str:
    """Trunk route recorded in a checkpoint (ACT v2 checkpoints are 'none')."""
    side = state.get("trunk_side") or (state.get("run_meta") or {}).get("trunk_side")
    if side is None:
        return "none"
    if side not in ("none", "concat"):
        raise RuntimeError(f"unknown trunk_side in checkpoint: {side!r}")
    return str(side)


def _require_trunk_side(state: Mapping[str, Any], expected_side: str | None) -> str:
    side = checkpoint_trunk_side(state)
    if expected_side is not None and side != expected_side:
        raise RuntimeError(f"trunk side mismatch: checkpoint is '{side}', run requested '{expected_side}'")
    return side


def resolve_trunk(source: Path, *, view: str = "ema", epoch: int | None = None,
                  expected_side: str | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Resolve a trunk from a checkpoint file, a selection dest, or a run dir.

    Returns the trunk tensors (with provenance).  A selection dest is any
    directory holding a ``score_receipt.json`` with ``selection.epoch`` and
    ``run_dir``; a bare run dir needs ``epoch`` (``--freeze-trunk-epoch``).
    ``expected_side`` gates the route: a concat run can only freeze a concat
    trunk and a bare run a bare trunk.
    """
    source = Path(source).resolve()
    if source.suffix == ".pt":
        checkpoint = source
        run_dir = source.parent
        selected = None
    elif (source / "score_receipt.json").is_file():
        receipt = json.loads((source / "score_receipt.json").read_text())
        run_dir = Path(receipt["run_dir"]).resolve()
        selected = int(receipt["selection"]["epoch"])
        checkpoint = run_dir / f"epoch_{selected:03d}.pt"
    else:
        run_dir = source
        selected = None if epoch is None else int(epoch)
        if selected is None:
            raise ValueError(f"{source} is neither a checkpoint nor a selection dest; pass --freeze-trunk-epoch")
        checkpoint = run_dir / f"epoch_{selected:03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    side = _require_trunk_side(state, expected_side)
    trunk = extract_trunk_state(state, view)
    digest = hashlib.sha256()
    for name in sorted(trunk):
        digest.update(name.encode())
        digest.update(trunk[name].numpy().tobytes())
    provenance = {
        "source": str(source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint),
        "checkpoint_schema": str(state.get("schema")),
        "run_dir": str(run_dir),
        "epoch": int(state.get("epoch", selected if selected is not None else -1)),
        "view": view,
        "trunk_side": side,
        "trunk_weights_sha256": digest.hexdigest(),
        "trunk_parameter_count": len(trunk),
        "selection_receipt_used": selected is not None,
    }
    return trunk, provenance


def resolve_formal_trunk(source: Path, *, task: str, side: str = "none") -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Fail-closed trunk source for FORMAL stage 2.

    The source must be a completed stage-1 B3S selection receipt: same task,
    stage ``1_pretain``, FORMAL run with a COMPLETED train receipt, a complete
    every-epoch EMA scan, and the earliest-max selected epoch.  The frozen
    checkpoint must match the curve's recorded sha, carry the EMA view, and
    record the same trunk side (``side``) as the requesting run.
    """
    SCORE_SCHEMA = "b3s_full_score_v1"
    source = Path(source).resolve()
    if source.suffix == ".pt" or not (source / "score_receipt.json").is_file():
        raise ValueError("formal stage 2 requires the stage-1 selection receipt directory (not a bare checkpoint/run dir)")
    receipt = json.loads((source / "score_receipt.json").read_text())
    if receipt.get("schema") != SCORE_SCHEMA or receipt.get("status") != "COMPLETED":
        raise RuntimeError("trunk source receipt is not a completed B3S score receipt")
    if receipt.get("stage") != "1_pretain" or receipt.get("task") != task:
        raise RuntimeError(f"trunk source is not a stage-1 pretrain selection for task {task}")
    if receipt.get("view") != "EMA" or receipt.get("partial"):
        raise RuntimeError("trunk source selection must be a complete EMA scan")
    if str(receipt.get("trunk_side", "none")) != str(side):
        raise RuntimeError(f"trunk source receipt side '{receipt.get('trunk_side', 'none')}' != requested '{side}'")
    run_dir = Path(receipt["run_dir"]).resolve()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if meta.get("schema") != SCHEMA or meta.get("task") != task or str(meta.get("stage")) != "1_pretain" or meta.get("status") != "FORMAL":
        raise RuntimeError("trunk source run is not a FORMAL stage-1 B3S pretrain of this task")
    if str(meta.get("trunk_side", "none")) != str(side):
        raise RuntimeError(f"trunk source run side '{meta.get('trunk_side', 'none')}' != requested '{side}'")
    train_receipt = json.loads((run_dir / "train_receipt.json").read_text())
    if train_receipt.get("status") != "COMPLETED":
        raise RuntimeError("trunk source training is incomplete")
    epochs_expected = int(meta.get("epochs", TASK_EPOCHS[task]))
    curve = receipt.get("ema_by_epoch") or {}
    if sorted(int(e) for e in curve) != list(range(1, epochs_expected + 1)):
        raise RuntimeError("trunk source selection did not scan every epoch")
    selected = int(receipt["selection"]["epoch"])
    checkpoint = run_dir / f"epoch_{selected:03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if curve[str(selected)].get("checkpoint_sha256") != _sha(checkpoint):
        raise RuntimeError("selected checkpoint bytes differ from the selection receipt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("schema") != CKPT_SCHEMA or state.get("task") != task or str(state.get("stage")) != "1_pretain" \
            or bool(state.get("smoke")) or int(state.get("epoch", -1)) != selected:
        raise RuntimeError("selected checkpoint is not the formal stage-1 epoch named by the receipt")
    recorded_side = _require_trunk_side(state, side)
    trunk = extract_trunk_state(state, "ema")
    digest = hashlib.sha256()
    for name in sorted(trunk):
        digest.update(name.encode())
        digest.update(trunk[name].numpy().tobytes())
    provenance = {
        "source": str(source), "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint),
        "checkpoint_schema": str(state.get("schema")), "run_dir": str(run_dir), "epoch": selected,
        "view": "ema", "trunk_side": recorded_side,
        "trunk_weights_sha256": digest.hexdigest(), "trunk_parameter_count": len(trunk),
        "selection_receipt_used": True,
        "formal_binding": {"receipt_schema": receipt.get("schema"), "run_status": meta.get("status"),
                           "scan_epochs": sorted(int(e) for e in curve), "selected_epoch": selected,
                           "receipt_dir": str(source)},
    }
    return trunk, provenance


# ---------------------------------------------------------------------------
# Carrier loading (per task; all sources frozen and hash-pinned).
# ---------------------------------------------------------------------------


def _validate_carriers(carriers: Mapping[str, np.ndarray], units: int, sessions: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for session in sessions:
        if session not in carriers:
            raise KeyError(f"carrier missing for session {session}")
        value = np.ascontiguousarray(np.asarray(carriers[session], np.float32))
        if value.shape != (units, 4) or not np.isfinite(value).all():
            raise RuntimeError(f"carrier geometry/nonfinite drift: {session} {value.shape}")
        out[session] = value
    return out


def load_m2_carriers() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Source MOVE-T4 carriers from the frozen dual-track cache."""
    from tfpd_exploration.src.m2_dual_track_v1 import data as old_data

    carriers: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for session in old_plan.HELDIN_SESSIONS:
        bank = old_data.load_session_bank("source_train", session, device="cpu")
        value = np.ascontiguousarray(bank.T.detach().cpu().numpy(), np.float32)
        if value.shape != (96, 4) or not np.isfinite(value).all():
            raise RuntimeError(f"M2 source carrier drift: {session}")
        carriers[session] = value
        provenance[session] = {
            "source": "dual_track cache 20260905_101500 source_train SessionBank.T (MOVE-T4, = adapters.build_m2_bank carrier)",
            "sha256": _ah(value),
        }
    return carriers, {
        "variant": "m2_move_t4_dual_track_cache",
        "per_session": provenance,
        "source_note": "identical bytes to adapters.build_m2_bank(surface='source_train').carrier",
    }


def load_m2_ext6_carriers(ext6_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    carriers: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for directory in sorted(p for p in ext6_root.iterdir() if p.is_dir()):
        path = directory / "T.npy"
        value = np.ascontiguousarray(np.load(path), np.float32)
        if value.shape != (96, 4) or not np.isfinite(value).all():
            raise RuntimeError(f"M2 EXT6 carrier drift: {directory.name}")
        carriers[directory.name] = value
        provenance[directory.name] = {
            "source": str(path),
            "file_sha256": _sha(path),
            "sha256": _ah(value),
            "same_file_as": "m2_ext6_epoch_pick.load_query_pair bank.carrier",
        }
    if len(carriers) != 6:
        raise RuntimeError(f"M2 EXT6 carrier roster drift: {sorted(carriers)}")
    return carriers, {"variant": "m2_move_t4_ext6_Tnpy", "root": str(ext6_root), "per_session": provenance}


def load_m1_carriers(pack_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Official M1 muscle pack through the frozen fail-closed loader."""
    frozen_dir = ROOT / "scripts/m1_muscle_r100_v1"
    for candidate in (frozen_dir, ROOT / "scripts", V1 / "src", WS):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location("_activity_full_m1_carrier", frozen_dir / "carrier.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot privately load the frozen M1 carrier module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    carriers, body = module.load_carrier_pack(Path(pack_path))
    return carriers, {
        "variant": f"m1_{module.METHOD}",
        "carrier_pack_npz": str(Path(pack_path).resolve()),
        "carrier_pack_npz_sha256": _sha(Path(pack_path)),
        "carrier_pack_receipt_sha256": _sha(Path(pack_path).with_suffix(".json")),
        "fit_sha256": body.get("fit_sha256"),
        "loader": "frozen m1_muscle_r100_v1/carrier.py::load_carrier_pack (fail-closed)",
        "per_session": {s: {"sha256": _ah(v)} for s, v in sorted(carriers.items())},
    }


def _load_h1_signed():
    """Privately load the signed-state module with SPINT-main's ``src`` prebound."""
    for name in [key for key in sys.modules if key == "src" or key.startswith("src.")]:
        del sys.modules[name]
    spint_main = WS / "SPINT-main"
    if str(spint_main) not in sys.path:
        sys.path.insert(0, str(spint_main))
    import src.data.h1_m4_eb_pilot  # noqa: F401  (binding is the point)
    signed_dir = ROOT / "scripts/h1_signed_state_r300_v1"
    if str(signed_dir) not in sys.path:
        sys.path.insert(0, str(signed_dir))
    spec = importlib.util.spec_from_file_location("_activity_full_h1_signed", signed_dir / "train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot privately load the signed-state H1 trainer")
    signed = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = signed
    spec.loader.exec_module(signed)
    return signed


def load_h1_carriers(banks_dir: Path, support_trial_ids: Mapping[str, Any], *, include_train: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """H1 carriers: per-session m3 profile at the ACT v2 support; HO payload carriers."""
    signed = _load_h1_signed()
    plan, receipt, carriers27 = signed._load_banks(Path(banks_dir))
    legacy = signed.h1_profiles._legacy()
    paths = legacy.index_heldin_calib(signed.b2.DATA_ROOT)
    train: dict[str, np.ndarray] = {}
    per_session: dict[str, Any] = {}
    if include_train:
        if not support_trial_ids:
            raise ValueError("H1 training carriers require the activity support trial ids")
        for session in sorted(support_trial_ids):
            record = legacy.load_record(paths[session])
            m3_vals = tuple(float(v) for v in record.trial_values[:3])
            ids = tuple(float(v) for v in support_trial_ids[session])
            if m3_vals != ids:
                raise RuntimeError(f"{session}: m3 profile support {m3_vals} != activity support {ids}")
            carrier, _raw, _info = signed.h1_profiles.deploy_profile(record, plan, m3_vals)
            carrier = np.ascontiguousarray(carrier, np.float32)
            if carrier.shape != (176, 4) or not np.isfinite(carrier).all():
                raise RuntimeError(f"H1 m3 carrier drift: {session}")
            train[session] = carrier
            per_session[session] = {
                "source": "signed_state14 plan deploy_profile(record, plan, first-3 eval-valid trials) == m3 profile at start=0",
                "support_trials": list(m3_vals),
                "sha256": _ah(carrier),
            }
    heldout: dict[str, np.ndarray] = {}
    for session, key in signed.HELDOUT_SESSION_TO_FALCON_KEY:
        carrier = np.ascontiguousarray(carriers27[key], np.float32)
        if carrier.shape != (176, 4) or not np.isfinite(carrier).all():
            raise RuntimeError(f"H1 HO carrier drift: {key}")
        heldout[session] = carrier
        heldout[session.removeprefix("ses-")] = carrier
    return {**train, **heldout}, {
        "variant": "h1_signed_state14_m3_profile_constant_plus_ho_payload",
        "banks_dir": str(Path(banks_dir).resolve()),
        "banks_receipt_sha256": _sha(Path(banks_dir) / "receipt.json"),
        "train_support_alignment": "per-session first-3 eval-valid trials asserted equal to activity_data support trial_ids",
        "concession": "the signed-state per-(session,start,budget) carrier rotation is not reproduced; "
                      "training uses the session-constant m3 (budget=3) profile at the ACT v2 support set "
                      "(ACT v2 support is M3-shaped); the HO-M3 face uses its 27-tag payload carrier",
        "per_session": per_session,
        "per_session_ho": {k: {"sha256": _ah(v)} for k, v in sorted(heldout.items())},
    }


def load_task_carriers(task: str, *, train_sessions: list[str], eval_sessions: list[str] | None = None,
                       support_trial_ids: Mapping[str, Any] | None = None,
                       m1_pack: Path = M1_CARRIER_PACK_DEFAULT, h1_banks: Path = H1_BANKS_DEFAULT,
                       m2_ext6_root: Path | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    task = task.lower()
    units = {"m1": 64, "m2": 96, "h1": 176}[task]
    if task == "m2":
        carriers, binding = load_m2_carriers()
        if eval_sessions:
            if m2_ext6_root is None:
                m2_ext6_root = resolve_m2_ext6_root()
            ext6, ext6_binding = load_m2_ext6_carriers(m2_ext6_root)
            carriers = {**carriers, **ext6}
            binding = {"source_train": binding, "evaluation": ext6_binding}
    elif task == "m1":
        carriers, binding = load_m1_carriers(m1_pack)
    elif task == "h1":
        carriers, binding = load_h1_carriers(h1_banks, support_trial_ids or {}, include_train=bool(train_sessions))
    else:
        raise ValueError(task)
    if eval_sessions:
        missing = [s for s in eval_sessions if s not in carriers]
        if missing:
            raise RuntimeError(f"{task} carrier faces incomplete: {missing}")
    return _validate_carriers(carriers, units, sorted(carriers)), binding


def resolve_m2_ext6_root() -> Path:
    """Delegates to :func:`activity_data.resolve_m2_ext6_root` (shared locator)."""
    return activity_data.resolve_m2_ext6_root()


# ---------------------------------------------------------------------------
# Training / scoring surfaces.
# ---------------------------------------------------------------------------


def _trial_mask(item: Mapping[str, Any], device: torch.device) -> torch.Tensor | None:
    return act._trial_mask(item, device)


def _support(item: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    return act._support(item, device)


def score_surface(model, surface: Mapping[str, Any], *, context: int, device: torch.device,
                  behavior_scale: float, max_batches: int | None = None,
                  save_predictions: Path | None = None, identities: dict[str, torch.Tensor] | None = None,
                  carriers: Mapping[str, torch.Tensor] | None = None, capture_arrays: bool = False) -> dict[str, Any]:
    model.eval(); rows = {}; ps = []; ts = []
    side = str(getattr(model, "trunk_side", "none") or "none")
    for session, item in surface.items():
        carrier = None if carriers is None else carriers[session]
        if identities is not None and session in identities:
            identity = identities[session]
        else:
            activity, trial_mask = _support(item, device)
            with torch.no_grad():
                # Concat route: the trunk eats the target session's carrier as a
                # frozen side input when producing E0; bare route ignores it.
                if side == "concat":
                    identity = model.calibrate(activity, trial_mask, carrier=carrier)
                else:
                    identity = model.calibrate(activity, trial_mask)
            if identities is not None:
                identities[session] = identity
        pred = []; target = []
        for start in range(0, len(item["starts"]), BATCH):
            if max_batches is not None and start // BATCH >= max_batches:
                break
            ids = np.arange(start, min(start + BATCH, len(item["starts"])), dtype=np.int64)
            x, _y, valid = activity_data.windows(item, ids, context, device, behavior_scale)
            with torch.no_grad():
                p = model(x, identity=identity, input_valid_mask=valid, carrier=carrier).float().cpu().numpy() / behavior_scale
            pred.append(p); target.append(np.asarray(item["Y"], np.float32)[ids])
        if not pred:
            continue
        p = np.concatenate(pred); t = np.concatenate(target); r = float(variance_weighted_r2(t, p))
        rows[session] = {"r2": r, "window_count": len(t), "identity_norm": float(identity.norm().cpu()),
                         "prediction_sha256": hashlib.sha256(p.tobytes()).hexdigest()}
        if capture_arrays:
            rows[session]["_prediction"], rows[session]["_target"] = p, t
        if save_predictions is not None:
            save_predictions.mkdir(parents=True, exist_ok=True)
            np.save(save_predictions / f"{session}.npy", p)
        ps.append(p); ts.append(t)
    if not rows:
        return {"per_session": {}, "equal_session_mean": None, "pooled_r2": None, "n_windows": 0, "partial": max_batches is not None}
    return {"per_session": rows, "equal_session_mean": float(np.mean([r["r2"] for r in rows.values()])),
            "pooled_r2": float(variance_weighted_r2(np.concatenate(ts), np.concatenate(ps))),
            "n_windows": int(sum(r["window_count"] for r in rows.values())), "partial": max_batches is not None}


def _batches(task: str, train: Mapping[str, Any], epoch: int, seed: int):
    if task == "m2":
        return act._m2_manifest_batches(train, epoch)
    rng = np.random.default_rng(seed + epoch); out = []
    for session in sorted(train):
        ids = rng.permutation(len(train[session]["starts"]))
        out += [(session, ids[i:i + BATCH]) for i in range(0, len(ids), BATCH)]
    rng.shuffle(out)
    return out


def _with_ema(model, ema: DecoderEMA, fn):
    return act._with_ema(model, ema, fn)


def _assert_carrier_bytes(carriers: Mapping[str, np.ndarray], binding: Mapping[str, Any], meta_carriers: Mapping[str, Any] | None) -> None:
    """Every carrier must match its recorded source bytes exactly."""
    if meta_carriers is None:
        return
    recorded = meta_carriers.get("sha256_per_session") or {}
    for session, value in carriers.items():
        digest = recorded.get(session)
        if digest is not None and digest != _ah(value):
            raise RuntimeError(f"carrier bytes drifted vs run_meta: {session}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    task = args.task.lower(); smoke = args.smoke_steps is not None
    stage = int(args.stage)
    if task not in TASK_EPOCHS:
        raise ValueError("--task must be m1, m2, or h1")
    if stage not in (1, 2):
        raise ValueError("--stage must be 1 (joint pretrain) or 2 (frozen-trunk B3S)")
    if not smoke and args.epochs != TASK_EPOCHS[task]:
        raise ValueError(f"formal {task} needs {TASK_EPOCHS[task]} epochs")
    if args.proj_dim != 16:
        raise ValueError("B3S full recipe is P16")
    if stage == 1 and args.seed not in (42, 43):
        # 42 pins the B3S bare-trunk route (same-instance subtractable vs ACT v2);
        # 43 is allowed for the Concat route's learned+full multi-seed policy.
        raise ValueError("stage-1 seeds are 42 or 43")
    seed = int(args.seed)
    if seed < 1:
        raise ValueError("--seed must be positive")
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    data = activity_data.load_task_data(task, include_eval=False); train = data["train"]; mini = data.get("validation", {})
    if not train:
        raise RuntimeError("source training surface is empty")
    context = int(data["metadata"]["context"]); scale = float(data["metadata"].get("behavior_scale", 5.0))
    cfg = config_from_args(args, task)
    side = str(getattr(args, "trunk_side", "none") or "none")
    if side not in ("none", "concat"):
        raise ValueError("--trunk-side must be none or concat")
    trunk_state, trunk_provenance = None, None
    if stage == 2:
        if args.freeze_trunk_from is None:
            raise ValueError("stage 2 requires --freeze-trunk-from")
        if not smoke:
            # Formal stage 2 binds the stage-1 SELECTION receipt (complete EMA
            # scan, earliest-max epoch, EMA view, same trunk side).  Bare
            # checkpoints, run dirs and non-B3S-stage-1 sources (e.g. ACT v2
            # runs) are rejected; the permissive resolver below stays smoke-only.
            if args.freeze_trunk_view != "ema":
                raise ValueError("formal stage 2 freezes the EMA view (the selection basis); --freeze-trunk-view must be ema")
            trunk_state, trunk_provenance = resolve_formal_trunk(args.freeze_trunk_from, task=task, side=side)
        else:
            trunk_state, trunk_provenance = resolve_trunk(args.freeze_trunk_from, view=args.freeze_trunk_view,
                                                          epoch=args.freeze_trunk_epoch, expected_side=side)
    model = B3SFullRiftDecoder(task, cfg, context_bins=context, seed=seed, support_bins=args.support_bins,
                               identity_hidden=args.identity_hidden, trunk_state=trunk_state, trunk_side=side).to(device)
    model.temporal.set_attention_backend("local")
    support_ids = {s: train[s]["support_provenance"].get("trial_ids", []) for s in train}
    carriers, carrier_binding = load_task_carriers(task, train_sessions=sorted(train), support_trial_ids=support_ids)
    missing = [s for s in sorted(train) if s not in carriers] + [s for s in sorted(mini) if s not in carriers]
    if missing:
        raise RuntimeError(f"carrier coverage incomplete: {missing}")
    trunk_params = {n: p for n, p in model.named_parameters() if n.startswith("identity_encoder.")}
    if stage == 2:
        if not all(not p.requires_grad for p in trunk_params.values()):
            raise RuntimeError("stage-2 trunk is not frozen")
        trunk_sha_initial = model.trunk_parameter_sha256()
        if trunk_sha_initial != trunk_provenance["trunk_weights_sha256"]:
            raise RuntimeError("loaded trunk bytes differ from the resolved checkpoint")
    ema = DecoderEMA(model, decay=v1_plan.EMA_DECAY)
    groups = split_optimizer_parameters(model, peak_lr=v1_plan.LR_PEAK, weight_decay=v1_plan.WEIGHT_DECAY, lr_multiplier=cfg.lr_multiplier)
    if stage == 2:
        group_ids = {id(p) for group in groups for p in group["params"]}
        leaked = [n for n, p in trunk_params.items() if id(p) in group_ids]
        if leaked or any(n.startswith("identity_encoder.") for n in ema.shadow):
            raise RuntimeError("frozen trunk leaked into optimizer/EMA")
    opt = torch.optim.AdamW(groups, lr=v1_plan.LR_PEAK, weight_decay=v1_plan.WEIGHT_DECAY, betas=old_plan.ADAM_BETAS, eps=old_plan.ADAM_EPS)
    dest = args.dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("fresh destination must be empty")
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": SCHEMA, "status": "SMOKE" if smoke else "FORMAL", "stage": f"{stage}_pretain" if stage == 1 else "2_b3s",
        "task": task, "seed": seed, "context_bins": context, "proj_dim": 16, "epochs": TASK_EPOCHS[task], "batch": BATCH,
        "trunk_side": side,
        "identity_interface": "concat_route" if side == "concat" else ("b3s_joint_pretain" if stage == 1 else "b3s_frozen_trunk_proj_add"),
        "identity_architecture": "existing_early_pool_activity_trunk_v2",
        "b3s_definition": "bare pre_pool -> trial mean -> post_pool activity trunk (M1 method family); "
                          "activity-calibrated; carrier T travels the token channel; no FiLM, no TaskBank",
        "side_injection": ("concat route: E0 = f(activity concatenated with carrier); the frozen [N,4] carrier joins at the "
                           "post_pool input, dim hidden+4 (H1: 32+4=36); semantics from the H1 C2 materializer "
                           "fused_identity (two_mainlines_long_v1/decoder/h1_calibration.py:64): "
                           "pooled = pre_pool(activity.permute).mean(trials); E0 = post_pool(cat(pooled, carrier)); "
                           "the token-channel carrier concat is retained in parallel"
                           if side == "concat" else
                           "none: the bare trunk E0 sees activity only; the carrier travels the token channel alone"),
        "trunk": {"trainable": stage == 1, "trunk_side": side,
                  **({"source": trunk_provenance} if trunk_provenance else {"note": "joint pretraining; trunk trained from scratch with the carrier present"})},
        "carrier": {"frozen": True, "trained": False,
                    "channel": "proj_add token concat (local conv + e0_proj(E0) + T)"
                               + (" + concat-side trunk input (E0 = post_pool(cat(pooled_activity, T)))" if side == "concat" else ""),
                    "binding": carrier_binding, "sha256_per_session": {s: _ah(v) for s, v in sorted(carriers.items())}},
        "zero_carrier": False, "no_T": False, "carrier_frozen": True, "no_TaskBank": True,
        "support_bins": model.support_bins, "identity_hidden": model.identity_hidden, "learnable_config": cfg.__dict__,
        "selection": {"primary": "FULL-compatible public calibration EMA scan; final claims EvalAI",
                      "source_minival": "audit only; never checkpoint selection"},
        "optimizer": {"name": "AdamW", "lr": v1_plan.LR_PEAK, "weight_decay": v1_plan.WEIGHT_DECAY,
                      "betas": list(old_plan.ADAM_BETAS), "eps": old_plan.ADAM_EPS, "grad_clip": v1_plan.GRAD_CLIP,
                      "ema_decay": v1_plan.EMA_DECAY, "unit_dropout": v1_plan.UNIT_DROPOUT,
                      "warmup_updates": M2_UPDATES if task == "m2" else None},
        "behavior_scale": scale,
        "sampler": "frozen M2 manifest 24x3165/B32 seed42 (manifest stays seed 42 at every --seed)" if task == "m2"
                   else "per-epoch deterministic global session-batch shuffle at --seed",
        "source_hashes": source_hashes(),
        "input_hashes": {k: {s: v.get("hashes", {}) for s, v in data.get(k, {}).items()} for k in ("train", "validation")},
        "support_provenance": {k: {s: v.get("support_provenance", {}) for s, v in data.get(k, {}).items()} for k in ("train", "validation")},
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    _atom(dest / "run_meta.json", meta)
    step = 0; started = time.monotonic()
    trunk_initial = {n: p.detach().clone() for n, p in trunk_params.items()}
    total = (M2_UPDATES if task == "m2" else max(1, len(_batches(task, train, 1, seed)))) * TASK_EPOCHS[task]
    row = None
    for epoch in range(1, TASK_EPOCHS[task] + 1):
        model.train(); losses = []; grad_norms = []; encoder_grad = {"pre_pool": [], "post_pool": []}
        for bi, (session, ids) in enumerate(_batches(task, train, epoch, seed)):
            item = train[session]; x, y, valid = activity_data.windows(item, ids, context, device, scale)
            a, tm = _support(item, device); carrier = torch.as_tensor(carriers[session], device=device)
            step += 1
            apply_group_lrs(opt, warmup_cosine_lr(step, total_steps=total,
                                                  warmup_steps=(M2_UPDATES if task == "m2" else max(1, total // TASK_EPOCHS[task])),
                                                  peak=v1_plan.LR_PEAK, min_factor=v1_plan.LR_MIN_FACTOR))
            g = torch.Generator(device="cpu"); g.manual_seed(unit_dropout_seed(seed, epoch, bi))
            keep = whole_unit_dropout(torch.ones(model.units, dtype=torch.bool), p=v1_plan.UNIT_DROPOUT, generator=g)
            opt.zero_grad(set_to_none=True)
            with (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()):
                loss = nn.functional.mse_loss(model(x, a, trial_mask=tm, dropout_keep=keep, input_valid_mask=valid,
                                                    carrier=carrier).float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            for branch in encoder_grad:
                grads = [p.grad.detach().norm() for n, p in model.named_parameters()
                         if n.startswith(f"identity_encoder.{branch}.") and p.grad is not None]
                encoder_grad[branch].append(float(torch.stack(grads).norm().cpu()) if grads else 0.0)
            grad_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), v1_plan.GRAD_CLIP, error_if_nonfinite=True)))
            opt.step(); ema.update_after_step(model); losses.append(float(loss.detach().cpu()))
            if smoke and step >= args.smoke_steps:
                break
        raw = score_surface(model, mini, context=context, device=device, behavior_scale=scale,
                            max_batches=1 if smoke else None, carriers=carriers) if mini else None
        es = _with_ema(model, ema, lambda: score_surface(model, mini, context=context, device=device, behavior_scale=scale,
                                                         max_batches=1 if smoke else None, carriers=carriers)) if mini else None
        ckpt = dest / f"epoch_{epoch:03d}.pt"
        torch.save({"schema": CKPT_SCHEMA, "stage": meta["stage"], "task": task, "epoch": epoch, "global_step": step,
                    "smoke": smoke, "trunk_side": side, "raw_state_dict": model.state_dict(), "ema": ema.state_dict(),
                    "optimizer": opt.state_dict(), "run_meta": meta}, ckpt)
        if stage == 2:
            drifted = {n: float((p.detach() - trunk_initial[n]).norm().cpu()) for n, p in trunk_params.items()}
            if any(v != 0.0 for v in drifted.values()):
                raise RuntimeError("frozen trunk weights changed during training")
            grads_on_trunk = [n for n, p in trunk_params.items() if p.grad is not None]
            if grads_on_trunk:
                raise RuntimeError(f"frozen trunk received gradients: {grads_on_trunk}")
            trunk_audit = {"trunk_sha256": model.trunk_parameter_sha256(), "trunk_unchanged": True,
                           "trunk_parameter_delta_norms": {k: 0.0 for k in trunk_params}}
        else:
            audit = {k: {"mean": float(np.mean(v)), "max": float(np.max(v)), "finite": bool(np.isfinite(v).all()),
                         "nonzero": bool(np.max(v) > 0)} for k, v in encoder_grad.items()}
            if not all(v["finite"] and v["nonzero"] for v in audit.values()):
                raise RuntimeError("joint trunk pre_pool/post_pool did not receive finite nonzero gradients")
        changed = {n: float((p.detach() - trunk_initial[n]).norm().cpu()) for n, p in trunk_params.items()} if stage == 1 else {}
        row = {"epoch": epoch, "global_step": step, "train_mse": float(np.mean(losses)), "grad_norm": float(np.mean(grad_norms)),
               "encoder_grad_audit": audit if stage == 1 else None,
               "trunk_frozen_audit": trunk_audit if stage == 2 else None,
               "encoder_parameter_delta_norms": changed,
               "source_minival_raw": raw, "source_minival_ema": es}
        _append(dest / "metrics.jsonl", row); _atom(dest / "heartbeat.json", row)
        if not smoke and task == "m2" and (len(losses) != M2_UPDATES or step != epoch * M2_UPDATES or ema.n_updates != step):
            raise RuntimeError("M2 exact manifest/EMA accounting drift")
        if smoke:
            break
    final_epoch = epoch
    if not smoke and task == "m2" and step != 24 * M2_UPDATES:
        raise RuntimeError("M2 total update count drift")
    state = torch.load(dest / f"epoch_{final_epoch:03d}.pt", map_location=device, weights_only=False)
    restored = B3SFullRiftDecoder(task, cfg, context_bins=context, seed=seed, support_bins=model.support_bins,
                                  identity_hidden=args.identity_hidden, trunk_side=side).to(device)
    restored.temporal.set_attention_backend("local")
    restored.load_state_dict(state["raw_state_dict"])
    if stage == 2:
        restored.freeze_trunk()
    sample = next(iter(train.values()))
    sx, _, sv = activity_data.windows(sample, np.arange(1), context, device, scale)
    sa, stm = _support(sample, device)
    fixed_keep = torch.ones(model.units, dtype=torch.bool, device=device)
    scarrier = torch.as_tensor(carriers[next(iter(train))], device=device)
    with torch.no_grad():
        raw_parity = bool(torch.equal(
            model.eval()(sx, sa, trial_mask=stm, dropout_keep=fixed_keep, input_valid_mask=sv, carrier=scarrier),
            restored.eval()(sx, sa, trial_mask=stm, dropout_keep=fixed_keep, input_valid_mask=sv, carrier=scarrier)))
        shadow = state["ema"]["shadow"]
        for n, p in restored.named_parameters():
            if n in shadow:
                p.copy_(shadow[n].to(p.device, p.dtype))
        ema_parity = bool(torch.equal(
            _with_ema(model, ema, lambda: model.eval()(sx, sa, trial_mask=stm, dropout_keep=fixed_keep,
                                                       input_valid_mask=sv, carrier=scarrier)),
            restored.eval()(sx, sa, trial_mask=stm, dropout_keep=fixed_keep, input_valid_mask=sv, carrier=scarrier)))
    parity = raw_parity and ema_parity
    receipt = {
        "schema": "b3s_full_smoke_receipt_v1" if smoke else "b3s_full_train_receipt_v1",
        "status": "COMPLETED", "stage": meta["stage"], "trunk_side": side, "formal_claim": not smoke,
        "selection": {"checkpoint": f"epoch_{final_epoch:03d}.pt", "view": "EMA",
                      "rule": "smoke only" if smoke else "last completed training checkpoint; local selection performed by FULL-compatible EMA scan"},
        "global_step": step, "checkpoint_parity": {"raw": raw_parity, "ema": ema_parity},
        "finite_loss": bool(math.isfinite(row["train_mse"])),
        "carrier": {"frozen": True, "sessions": len(carriers), "sha256_per_session": meta["carrier"]["sha256_per_session"]},
        "trunk": ({"frozen": True, "weights_sha256": model.trunk_parameter_sha256(),
                   "unchanged_from_source": model.trunk_parameter_sha256() == trunk_provenance["trunk_weights_sha256"],
                   "source": trunk_provenance} if stage == 2 else {"trainable": True, "joint_pretrain": True}),
        "runtime_seconds": time.monotonic() - started,
    }
    if stage == 2 and not receipt["trunk"]["unchanged_from_source"]:
        raise RuntimeError("stage-2 trunk bytes differ from the source checkpoint after training")
    if smoke and not (parity and receipt["finite_loss"]):
        raise RuntimeError("B3S full smoke validation failed")
    _atom(dest / ("smoke_receipt.json" if smoke else "train_receipt.json"), receipt)
    return receipt


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_learnable_flags(p); p.set_defaults(ladder="default")
    p.add_argument("--task", required=True, choices=("m1", "m2", "h1"))
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--stage", type=int, default=1, choices=(1, 2))
    p.add_argument("--trunk-side", choices=("none", "concat"), default="none",
                   help="none: bare B3S trunk, carrier token channel only; concat: carrier also joins the trunk "
                        "at the post_pool input (H1 C2 fused_identity semantics)")
    p.add_argument("--freeze-trunk-from", type=Path, default=None,
                   help="stage 2: checkpoint .pt, selection dest with score_receipt.json, or run dir (+ --freeze-trunk-epoch)")
    p.add_argument("--freeze-trunk-epoch", type=int, default=None)
    p.add_argument("--freeze-trunk-view", choices=("ema", "raw"), default="ema")
    p.add_argument("--m1-carrier-pack", type=Path, default=M1_CARRIER_PACK_DEFAULT)
    p.add_argument("--h1-banks", type=Path, default=H1_BANKS_DEFAULT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--proj-dim", type=int, default=16)
    p.add_argument("--support-bins", type=int)
    p.add_argument("--identity-hidden", type=int, default=None)
    p.add_argument("--smoke-steps", type=int)
    return p


def main():
    a = build_parser().parse_args()
    a.epochs = TASK_EPOCHS[a.task] if a.epochs is None else a.epochs
    if a.smoke_steps is not None and a.smoke_steps < 1:
        raise ValueError("--smoke-steps must be positive")
    print(json.dumps(run(a), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
