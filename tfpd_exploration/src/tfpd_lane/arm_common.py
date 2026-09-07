"""Shared utilities for the Gate-2 three-arm admission trainers (§3/§4/§6/§9).

Implements the frozen, arm-independent pieces so that arms A/B/C are strict
recombinations of one implementation surface:

- the step-level LR schedule of HANDOFF_CARRIER_ADMISSION_CURRICULUM_20260816.md
  §6.2/§6.4: linear warmup 1e-5 -> 1e-4 over the first TWO epochs' optimizer
  steps, then cosine decay to 1e-6 at the final step of the last epoch.  B's
  T4 phase and arm C call the SAME function with the SAME phase-local
  arguments, so their "LR value at every T4-phase step" equality is structural;
  B phase 1 is constant lr=1e-4, exactly matching the sealed boundary pilot;
- the phase plans (arm, epochs per phase, schedule kind, alpha in {0,1});
- post-normalization admission (§4): the visible side is either the exact
  normalized-T4 tensor (identity) or ``zeros_like`` of it (canonical Z4);
  raw T4 is never multiplied by an alpha;
- byte hashing with §3 signed-zero normalization, W_side block/moment access;
- §9 fixed-diagnostic-batch branch summaries (post_pool[0] T4-vs-calibration
  contribution norms and their ratio) and attention entropy/concentration.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence

import numpy as np
import torch

# ---- frozen optimizer/LR contract (§6.1/§6.2/§6.3/§6.4) --------------------
ADAM_CONSTRUCTOR = {
    "cls": "torch.optim.Adam",
    "lr": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
}
PHASE1_LR = 1e-4  # constant; identical to the sealed boundary pilot
WARMUP_EPOCHS = 2
LR_WARMUP_START = 1e-5
LR_WARMUP_END = 1e-4
LR_FINAL = 1e-6
TOTAL_BUDGET_EPOCHS = 48

ARMS = ("A", "B", "C")


def warmup_steps(steps_per_epoch: int) -> int:
    return WARMUP_EPOCHS * int(steps_per_epoch)


def lr_at_step(step: int, n_epochs: int, steps_per_epoch: int) -> float:
    """Step-level warmup+cosine LR for a T4 phase of ``n_epochs`` epochs.

    ``step`` is PHASE-LOCAL (0-based optimizer step within the T4 phase).  At
    step 0 the LR is 1e-5; it reaches 1e-4 at the end of the second epoch's
    steps and decays by cosine to exactly 1e-6 at the final step.
    """
    spe = int(steps_per_epoch)
    total = int(n_epochs) * spe
    if spe <= 0 or int(n_epochs) <= 0:
        raise ValueError("n_epochs and steps_per_epoch must be positive")
    if step < 0 or step >= total:
        raise ValueError(f"step {step} outside phase [0, {total})")
    warm = warmup_steps(spe)
    if step < warm:
        return LR_WARMUP_START + (LR_WARMUP_END - LR_WARMUP_START) * (step / warm)
    span = total - warm
    progress = (step - warm) / span
    return LR_FINAL + 0.5 * (LR_WARMUP_END - LR_FINAL) * (1.0 + math.cos(math.pi * progress))


def schedule_params(n_epochs: int, steps_per_epoch: int) -> dict:
    spe = int(steps_per_epoch)
    return {
        "kind": "warmup_then_cosine",
        "phase_local_steps": True,
        "n_epochs": int(n_epochs),
        "steps_per_epoch": spe,
        "total_steps": int(n_epochs) * spe,
        "warmup_epochs": WARMUP_EPOCHS,
        "warmup_steps": warmup_steps(spe),
        "lr_warmup_start": LR_WARMUP_START,
        "lr_warmup_end": LR_WARMUP_END,
        "lr_final": LR_FINAL,
        "lr_at_step_0": lr_at_step(0, n_epochs, spe),
        # Smoke budgets (n_epochs*spe shorter than warmup) clip the probe steps
        # to the phase; the full frozen budget is unchanged.
        "lr_at_last_warmup_step": lr_at_step(min(warmup_steps(spe) - 1, n_epochs * spe - 1), n_epochs, spe),
        "lr_at_first_cosine_step": lr_at_step(min(warmup_steps(spe), n_epochs * spe - 1), n_epochs, spe),
        "lr_at_final_step": lr_at_step(int(n_epochs) * spe - 1, n_epochs, spe),
    }


def build_arm_plan(arm: str, t_pre: int, e_t4: int) -> dict:
    """Frozen phase plan.  B's T4 phase and arm C are the same plan prefix."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if t_pre + e_t4 != TOTAL_BUDGET_EPOCHS:
        raise ValueError(f"T_pre + E_t4 must be 48, got {t_pre} + {e_t4}")
    if arm == "A":
        phases = [
            {"phase": "t4_direct", "visible_side": "t4", "epochs": TOTAL_BUDGET_EPOCHS,
             "lr_schedule": "warmup_then_cosine", "alpha": 1}
        ]
    elif arm == "B":
        phases = [
            {"phase": "z4_pretrain", "visible_side": "z4", "epochs": int(t_pre),
             "lr_schedule": "constant_1e-4", "alpha": 0},
            {"phase": "t4_finetune", "visible_side": "t4", "epochs": int(e_t4),
             "lr_schedule": "warmup_then_cosine", "alpha": 1},
        ]
    else:
        phases = [
            {"phase": "t4_direct_exposure_matched", "visible_side": "t4", "epochs": int(e_t4),
             "lr_schedule": "warmup_then_cosine", "alpha": 1}
        ]
    return {
        "arm": arm,
        "t_pre": int(t_pre),
        "e_t4": int(e_t4),
        "total_model_training_epochs": sum(p["epochs"] for p in phases),
        "adam_constructor": dict(ADAM_CONSTRUCTOR),
        "phases": phases,
    }


def assert_b_phase2_matches_c_plan(plan_b: dict, plan_c: dict) -> dict:
    """§6.4 equality: B's T4 phase and arm C share every schedule axis."""
    b_t4 = [p for p in plan_b["phases"] if p["visible_side"] == "t4"]
    assert len(b_t4) == 1 and len(plan_c["phases"]) == 1
    b_t4 = b_t4[0]
    c_t4 = plan_c["phases"][0]
    for field in ("epochs", "lr_schedule", "alpha"):
        if b_t4[field] != c_t4[field]:
            raise ValueError(f"B/C T4-phase drift at {field}: {b_t4[field]} vs {c_t4[field]}")
    if plan_b["adam_constructor"] != plan_c["adam_constructor"]:
        raise ValueError("B/C Adam constructor drift")
    return {
        "t4_phase_epochs_equal": True,
        "lr_schedule_equal": True,
        "adam_constructor_equal": True,
        "batch_order_rule": "same sampler constructor (dataset, batch=32, shuffle, seed=42)",
        "dropout_rng_note": (
            "dropout draws are NOT equalized across arms (B's stream continues from phase 1); "
            "the contract equalizes constructor, LR at every step, batch order, and step count"
        ),
    }


# ---- §4 admission: strictly post-normalization -----------------------------
def admit_side(side_normalized: torch.Tensor, mode: str) -> torch.Tensor:
    """Choose the model-visible side from the ALIGNED NORMALIZED T4 tensor.

    mode='t4': the exact normalized-T4 tensor, returned by identity (never a
    product with alpha).  mode='z4': ``zeros_like`` of it — bitwise positive
    zero, proven bitwise equal to the production z4 loader at preflight.
    """
    if mode == "t4":
        return side_normalized
    if mode == "z4":
        return torch.zeros_like(side_normalized)
    raise ValueError(f"visible-side mode must be 't4' or 'z4', got {mode!r}")


# ---- §3/§9 hashing and branch accessors ------------------------------------
def tensor_sha256(tensor) -> str:
    digest = hashlib.sha256()
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.is_floating_point():
        flat = flat + 0  # IEEE: -0.0 + 0.0 == +0.0; identity otherwise
    digest.update(str(flat.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    if flat.numel():
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(model) -> str:
    from torch.nn.parameter import UninitializedParameter

    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        tensor = state[key]
        if isinstance(tensor, UninitializedParameter):
            digest.update(b"|uninitialized-lazy|")
            continue
        digest.update(tensor_sha256(tensor).encode("utf-8"))
    return digest.hexdigest()


def optimizer_sha256(optimizer) -> str:
    payload = optimizer.state_dict()
    digest = hashlib.sha256()
    digest.update(json.dumps(payload.get("param_groups", []), sort_keys=True).encode("utf-8"))
    for index in sorted(payload.get("state", {}), key=int):
        entry = payload["state"][index]
        digest.update(f"|{index}|".encode("utf-8"))
        step = entry.get("step")
        step_bytes = (
            tensor_sha256(step).encode("utf-8")
            if torch.is_tensor(step)
            else repr(step).encode("utf-8")
        )
        digest.update(b"step:" + step_bytes)
        for moment in ("exp_avg", "exp_avg_sq"):
            if moment in entry:
                digest.update(
                    moment.encode("utf-8") + b":" + tensor_sha256(entry[moment]).encode("utf-8")
                )
            else:
                digest.update(moment.encode("utf-8") + b":absent")
    return digest.hexdigest()


def w_side_block(model) -> torch.Tensor:
    encoder = model.id_encoder
    weight = encoder.post_pool[0].weight
    return weight[:, encoder.hidden_dim : encoder.hidden_dim + encoder.side_dim]


def w_side_moment(model, optimizer, name: str):
    param = model.id_encoder.post_pool[0].weight
    encoder = model.id_encoder
    state = optimizer.state.get(param, {})
    tensor = state.get(name)
    if tensor is None:
        return None
    return tensor[:, encoder.hidden_dim : encoder.hidden_dim + encoder.side_dim]


def param_groups_by_branch(model):
    """(encoder params, decoder params) for per-branch gradient norms (§9)."""
    encoder_params = [p for p in model.id_encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    return encoder_params, decoder_params


# ---- §9 fixed-batch diagnostics -------------------------------------------
def post_pool_contribution(model, calib: torch.Tensor, side_t4: torch.Tensor) -> dict:
    """T4-vs-calibration contribution norms at post_pool[0] on a fixed batch.

    Replicates the encoder's own pooling (reset_stream/push_trial, the exact
    functions forward_batch calls) so ``mean_feat`` is the true calibration
    branch input; contributions are the two column blocks of the same first
    affine layer applied to their respective inputs.
    """
    encoder = model.id_encoder
    hidden = encoder.hidden_dim
    with torch.no_grad():
        batch, n_trials, _, n_units = calib.shape
        state = encoder.reset_stream(batch, n_units, calib.device, calib.dtype)
        for trial in range(n_trials):
            state = encoder.push_trial(state, calib[:, trial])
        mean_feat = state["sum_feat"] / state["trial_count"]  # [B, N, hidden]
        weight = encoder.post_pool[0].weight  # [hidden, hidden + side_dim]
        calib_contrib = torch.einsum("bnh,oh->bno", mean_feat, weight[:, :hidden])
        t4_contrib = torch.einsum("bns,os->bno", side_t4, weight[:, hidden:])
        calib_norm = float(calib_contrib.norm().item())
        t4_norm = float(t4_contrib.norm().item())
    return {
        "norm_w_side_times_t4_fixed_batch": t4_norm,
        "norm_calibration_contribution_fixed_batch": calib_norm,
        "ratio_t4_to_calibration_at_post_pool0": (t4_norm / calib_norm) if calib_norm > 0 else 0.0,
    }


def attention_summary(model, neural: torch.Tensor, calib: torch.Tensor, side: torch.Tensor) -> dict:
    """Mean cross-attention entropy and max-weight concentration (§9).

    Captures per-head attention weights by wrapping each decoder
    ``MultiheadAttention.forward`` at instance level for one forward pass,
    restoring the modules immediately afterwards.  Eval mode, no gradients.
    """
    layers = model.decoder.transformer.layers
    captured: list[torch.Tensor] = []
    patched = []
    for layer in layers:
        attn = layer.cross_attn
        original = attn.forward

        def _make_wrapper(orig, store):
            def wrapper(*args, **kwargs):
                kwargs.pop("need_weights", None)
                kwargs["average_attn_weights"] = False
                out, weights = orig(*args, need_weights=True, **kwargs)
                store.append(weights.detach())
                return out, weights

            return wrapper

        attn.forward = _make_wrapper(original, captured)
        patched.append(attn)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(neural, calib_trials=calib, side_features=side)
        entropies: list[float] = []
        maxima: list[float] = []
        for weights in captured:
            probs = weights.float().clamp_min(1e-12)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            entropy = -(probs * probs.log()).sum(dim=-1)
            entropies.append(float(entropy.mean().item()))
            maxima.append(float(probs.max(dim=-1).values.mean().item()))
        return {
            "n_layers_captured": len(captured),
            "mean_attention_entropy": float(np.mean(entropies)) if entropies else None,
            "mean_max_attention_weight": float(np.mean(maxima)) if maxima else None,
        }
    finally:
        for attn in patched:
            attn.__dict__.pop("forward", None)
        if was_training:
            model.train()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def t4_authority_fingerprint(records: dict) -> dict:
    """Per-session SHA of the aligned normalized-T4 side bytes (§4 authority)."""
    return {
        name: tensor_sha256(torch.from_numpy(record.side_features.copy()))
        for name, record in records.items()
    }
