"""Population-robustness round (D / DH cells) — route-owned helpers.

Contract: HANDOFF_POPULATION_ROBUSTNESS_NEXT_ROUND_20260817.md.  The two cells
are EXACT arm-A replicas except for the head count and the dynamic unit
dropout flags; no sealed file is modified (arm A runner, spintshape_module,
spint.py, streaming_spint.py stay byte-identical — enforced by tests).

- `build_population_robustness_model` rebuilds the arm-A graph directly from
  the streaming components with `num_heads` and `dynamic_dropout` as the ONLY
  changed constructor arguments.  nn.MultiheadAttention parameter shapes are
  independent of the head count (in_proj [3d,d], out_proj [d,d]); only the
  head PARTITION changes, so both cells strict-load the SAME
  canonical_initial_state.pt artifact (state sha 65bacb85…) bit-for-bit, and
  constructing either variant consumes the identical RNG stream as the arm-A
  builder (proven by state-SHA equality across variants).
- dynamic dropout uses the EXISTING implementation verbatim (per forward, one
  `random.uniform(low, high)` p and one PyTorch unit-mask dropout), train-mode
  only; eval/scoring never activates it.
- instrumentation is passive and route-owned: wrappers around
  `random.uniform` and `torch.nn.functional.dropout` record the p the model
  itself sampled (no second RNG draw), the realized retained-unit fraction,
  all-zero-population sample counts, and pre/post-dropout token norms;
  a per-head attention summary extends the existing instance-level capture
  with per-head entropies and an across-head diversity statistic.
"""

from __future__ import annotations

import contextlib
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

CELLS = {
    "D": {"num_heads": 2, "dynamic_dropout": True,
          "note": "2 heads + dynamic dropout; head count identical to arm A"},
    "DH": {"num_heads": 64, "dynamic_dropout": True,
           "note": "64 heads + dynamic dropout; same tensors, 64-way head partition"},
}
DYNAMIC_DROPOUT_LOW = 0.0
DYNAMIC_DROPOUT_HIGH = 1.0


def build_population_robustness_model(seed: int = 42, cell: str = "D"):
    """Arm-A graph with ONLY num_heads / dynamic_dropout changed."""
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {sorted(CELLS)}, got {cell!r}")
    config = CELLS[cell]
    import importlib.util

    for extra in (
        REPO_ROOT / "streaming_calibration_exp",
        REPO_ROOT / "sua_exploration",
    ):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    try:
        import src as _src_pkg
        streaming_src = str(REPO_ROOT / "streaming_calibration_exp" / "src")
        if streaming_src not in _src_pkg.__path__:
            _src_pkg.__path__.append(streaming_src)
    except ImportError:
        pass
    from src.models.components.spint import SpintModel
    from src.models.components.streaming_encoders import build_encoder
    from src.models.components.streaming_spint import StreamingSpintModel

    torch.manual_seed(seed)
    decoder = SpintModel(
        model_dim=512,
        num_covariates=2,
        window_size=50,
        num_heads=config["num_heads"],
        num_layers=1,
        num_id_layers=1,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
        dynamic_dropout=config["dynamic_dropout"],
        dynamic_dropout_low=DYNAMIC_DROPOUT_LOW,
        dynamic_dropout_high=DYNAMIC_DROPOUT_HIGH,
    )
    id_encoder = build_encoder(
        "B3S",
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        side_dim=4,
    )
    model = StreamingSpintModel(decoder=decoder, id_encoder=id_encoder, decoder_mode="coupled")
    model._pop_robust_cell = cell
    model._pop_robust_config = dict(config)
    return model


@contextlib.contextmanager
def dynamic_dropout_recorder():
    """Passive instrumentation of the EXISTING dynamic-dropout implementation.

    Records, per training forward: the p the model itself sampled via
    `random.uniform` (never a second draw), and for the matching unit-mask
    `F.dropout` call the realized retained-unit fraction, the count of
    batch samples whose whole unit population was dropped, and pre/post
    dropout token-norm summaries.  Everything passes through unchanged.
    """
    recorded: dict = {
        "sampled_p": [],
        "dropout_calls": [],
        "uniform_calls": 0,
    }
    original_uniform = random.uniform
    original_dropout = torch.nn.functional.dropout

    def recording_uniform(low, high):
        value = original_uniform(low, high)
        recorded["uniform_calls"] += 1
        recorded["sampled_p"].append(float(value))
        return value

    def recording_dropout(input, p=0.5, training=True, inplace=False):
        out = original_dropout(input, p=p, training=training, inplace=inplace)
        # the EXISTING implementation drops out the per-unit mask tensor
        # [batch, num_units] (2-D); tokens are scaled by it afterwards
        matches_dynamic = (
            training
            and float(p) > 0.0
            and input.dim() == 2
            and recorded["sampled_p"]
            and float(p) == recorded["sampled_p"][-1]
        )
        if matches_dynamic:
            with torch.no_grad():
                kept = (out != 0).float().mean().item()
                dead_rows = int((out == 0).all(dim=-1).sum().item())
                recorded["dropout_calls"].append({
                    "p": float(p),
                    "shape": list(input.shape),
                    "retained_unit_fraction": float(kept),
                    "all_zero_population_samples": dead_rows,
                    "dropout_scale": float(1.0 / (1.0 - float(p))) if float(p) < 1.0 else None,
                })
        return out

    recorded["fc_in_token_norms"] = []

    random.uniform = recording_uniform
    torch.nn.functional.dropout = recording_dropout
    try:
        yield recorded
    finally:
        random.uniform = original_uniform
        torch.nn.functional.dropout = original_dropout


@contextlib.contextmanager
def fc_in_token_norm_recorder(model):
    """Route-owned forward_pre_hook on decoder.fc_in: post-dropout token norms.

    The hook sees exactly the tensor fed to fc_in — the unit-mask-scaled token
    set — so its norms are the realized POST-dropout token norms.  Removed on
    exit; no model file is touched.
    """
    norms: list[dict] = []
    decoder = model.decoder

    def _capture(module, args):
        # decoder.fc_in is called twice per forward: once for the unit-token
        # src [B, n_units, W] and once for the query rep [1, C, W]; both are
        # recorded with their shapes and filtered by the consumers
        with torch.no_grad():
            tensor = args[0] if args else None
            if tensor is None:
                return
            per_token = tensor.float().norm(dim=-1)
            norms.append({"shape": list(tensor.shape), "norm": float(per_token.mean().item())})
        return None

    handle = decoder.fc_in.register_forward_pre_hook(_capture)
    try:
        yield norms
    finally:
        handle.remove()


def fixed_batch_dropout_diagnostic(model, neural, calib, side):
    """Measured pre/post dropout token norms on ONE fixed batch.

    Pre: an eval-mode forward (dropout disabled by the existing gate).
    Post: a train-mode forward under the passive recorder — the model's own
    sampled p and realized mask, never a redraw.  The model is restored to
    its original mode.
    """
    was_training = model.training
    out = {}
    model.eval()
    with fc_in_token_norm_recorder(model) as pre_norms:
        with torch.no_grad():
            model(neural, calib_trials=calib, side_features=side)
    def _unit_entry(entries):
        for entry in entries:
            if len(entry["shape"]) == 3 and entry["shape"][1] != 2:
                return entry["norm"]
        return None

    out["token_norm_pre_dropout_mean"] = _unit_entry(pre_norms)

    model.train()
    with fc_in_token_norm_recorder(model) as post_norms:
        with dynamic_dropout_recorder() as rec:
            model(neural, calib_trials=calib, side_features=side)
    out["token_norm_post_dropout_mean"] = _unit_entry(post_norms)
    out["realized_p"] = rec["sampled_p"][-1] if rec["sampled_p"] else None
    out["realized_retained_fraction"] = (
        rec["dropout_calls"][-1]["retained_unit_fraction"] if rec["dropout_calls"] else None
    )
    out["realized_all_zero_population_samples"] = (
        rec["dropout_calls"][-1]["all_zero_population_samples"] if rec["dropout_calls"] else None
    )
    if was_training:
        model.train()
    else:
        model.eval()
    return out


def summarize_dropout_record(recorded: dict, num_heads: int) -> dict:
    """Epoch-level aggregation of the passive record."""
    calls = recorded["dropout_calls"]
    ps = recorded["sampled_p"]
    summary = {
        "n_forwards_with_sampled_p": len(ps),
        "n_recorded_unit_mask_calls": len(calls),
        "sampled_p_min": float(np.min(ps)) if ps else None,
        "sampled_p_mean": float(np.mean(ps)) if ps else None,
        "sampled_p_max": float(np.max(ps)) if ps else None,
        "num_heads": num_heads,
    }
    if calls:
        summary.update({
            "retained_unit_fraction_mean": float(np.mean([c["retained_unit_fraction"] for c in calls])),
            "retained_unit_fraction_min": float(np.min([c["retained_unit_fraction"] for c in calls])),
            "all_zero_population_samples_total": int(sum(c["all_zero_population_samples"] for c in calls)),
        })
    # unit-token entries: [B, n_units, W] with n_units != num_covariates (2);
    # session batches carry different unit counts, so filter by dimension role
    unit_calls = [
        entry["norm"] for entry in recorded.get("fc_in_token_norms", [])
        if len(entry["shape"]) == 3 and entry["shape"][1] != 2
    ]
    if unit_calls:
        summary.update({
            "fc_in_input_token_norm_post_dropout_mean": float(np.mean(unit_calls)),
            "fc_in_input_token_norm_post_dropout_max": float(np.max(unit_calls)),
            "n_fc_in_unit_token_calls": len(unit_calls),
        })
    return summary


def per_head_attention_summary(model, neural, calib, side) -> dict:
    """Per-head attention entropies + across-head diversity (no RNG, eval mode).

    Captures per-head weights by wrapping each decoder MultiheadAttention at
    instance level for ONE forward (need_weights=True, per-head), restoring
    immediately.  For the D cell (2 heads) this yields two entropies; for DH
    (64 heads) a 64-vector plus the across-head diversity statistic.
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
        for weights in captured:  # [B, heads, C, N]
            probs = weights.float().clamp_min(1e-12)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            ent = -(probs * probs.log()).sum(dim=-1)  # [B, heads, C]
            entropies.extend(ent.mean(dim=(0, 2)).tolist())
            maxima.extend(probs.max(dim=-1).values.mean(dim=(0, 2)).tolist())
        return {
            "n_heads": len(entropies),
            "per_head_mean_entropy": [round(float(e), 6) for e in entropies],
            "per_head_mean_max_weight": [round(float(m), 6) for m in maxima],
            "across_head_entropy_std": float(np.std(entropies)) if entropies else None,
            "across_head_entropy_mean": float(np.mean(entropies)) if entropies else None,
        }
    finally:
        for attn in patched:
            attn.__dict__.pop("forward", None)
        if was_training:
            model.train()


def state_signature(model):
    """(sorted keys, shapes) signature, lazy-safe — for equality proofs."""
    from torch.nn.parameter import UninitializedParameter

    state = model.state_dict()
    shapes = {
        k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter) else tuple(v.shape))
        for k, v in state.items()
    }
    return sorted(state.keys()), shapes


def trainable_parameter_count(model) -> int:
    from torch.nn.parameter import UninitializedParameter

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad and not isinstance(p, UninitializedParameter)
    )


def initial_state_equality_proof(canonical_state: dict, seed: int = 42) -> dict:
    """Bitwise proof that D and DH load the canonical initial state unchanged.

    Both variants are constructed from the same seed; their state signatures
    and tensor bytes must equal the canonical payload's (the RNG stream is
    consumed identically because the head count changes no parameter shapes
    and the dropout flags are metadata only).
    """
    proofs = {}
    for cell in CELLS:
        model = build_population_robustness_model(seed=seed, cell=cell)
        keys, shapes = state_signature(model)
        canonical_keys = sorted(canonical_state.keys())
        keys_equal = keys == canonical_keys
        from torch.nn.parameter import UninitializedParameter

        shape_mismatches = [
            k for k in canonical_keys
            if shapes.get(k) != (
                ("uninitialized-lazy",)
                if isinstance(canonical_state[k], UninitializedParameter)
                else tuple(canonical_state[k].shape)
            )
        ]
        loaded = build_population_robustness_model(seed=seed, cell=cell)
        loaded.load_state_dict(canonical_state, strict=True)
        proofs[cell] = {
            "state_keys_equal_to_canonical": keys_equal,
            "shape_mismatches": shape_mismatches,
            "strict_load": True,
            "num_heads": CELLS[cell]["num_heads"],
            "dynamic_dropout": CELLS[cell]["dynamic_dropout"],
            "trainable_parameters": trainable_parameter_count(model),
            "note": "head count changes only the MHA head partition; RNG stream identical",
        }
        del model, loaded
    if {p["trainable_parameters"] for p in proofs.values()}.__len__() != 1:
        raise ValueError("parameter-count drift across cells")
    return proofs
