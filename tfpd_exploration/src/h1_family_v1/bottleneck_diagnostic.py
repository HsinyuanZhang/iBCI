"""Read-only pooling/bottleneck audit for the sealed CRST-B4 scale-1 gate.

The diagnostic does not optimize parameters, read minival rows, or modify the
preflight receipt/checkpoints.  CPU mode writes activation/attention/update
facts; GPU mode, invoked separately, adds the requested exact FP32 frontend
Jacobian on one declared source window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT as H1_ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.capacity_probe import batch
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from .model import make_v2_initialized_pair


ATTEMPT = H1_ROOT / "family_v1" / "crst_b4_scale1_source208_preflight_v1"
OUT = H1_ROOT / "family_v1" / "crst_b4_scale1_pooling_bottleneck_diagnostic_v1"
IDS_PATH = H1_ROOT / "capacity_probe_208_source_v2" / "frozen_ids.json"
SEED = 42
MICRO = 4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().float()).item())


def _fixed_ids() -> dict[str, np.ndarray]:
    if _sha(IDS_PATH) != "da4bf975a5c1023bbffe26da87d0d4977f437d7db1db012283511ef140d96211":
        raise RuntimeError("the sealed V2 fixed ID list changed")
    loaded = json.loads(IDS_PATH.read_text())["ids"]
    fixed = {name: np.asarray(values, dtype=np.int64) for name, values in loaded.items()}
    if len(fixed) != 13 or sum(len(v) for v in fixed.values()) != 208 or any(len(v) != 16 for v in fixed.values()):
        raise RuntimeError("expected exactly 13 source banks x 16 frozen endpoints")
    return fixed


def _models(device: torch.device) -> tuple[dict[str, torch.nn.Module], dict[str, dict], dict[str, torch.nn.Module]]:
    flat0, route0 = make_v2_initialized_pair(seed=SEED)
    initial = {"flat": flat0, "route": route0}
    flat, route = make_v2_initialized_pair(seed=SEED)
    models = {"flat": flat.to(device).eval(), "route": route.to(device).eval()}
    payloads: dict[str, dict] = {}
    for arm in models:
        path = ATTEMPT / f"{arm}_260.pt"
        payload = torch.load(path, map_location=device, weights_only=False)
        models[arm].load_state_dict(payload["model"], strict=True)
        if int(payload["updates"]) != 260:
            raise RuntimeError(f"{arm} checkpoint is not the sealed 260 state")
        payloads[arm] = payload
    return models, payloads, initial


def _bank(row: dict, device: torch.device) -> H1Bank:
    return H1Bank(row["bank"]["E0"].to(device), row["bank"]["T"].to(device), row["bank"]["unit_mask"].to(device))


def _attention(model: torch.nn.Module, tokens_last: torch.Tensor, bank: H1Bank) -> dict:
    frontend = model.frontend
    attn = frontend.attn
    slots = frontend.slot_norm(frontend.slots).unsqueeze(0)  # [1,S,D]
    query = attn.q_proj(slots).view(1, frontend.cfg.slots, attn.n_heads, attn.head_dim).transpose(1, 2)
    key = attn.k_proj(tokens_last).view(1, tokens_last.size(1), attn.n_heads, attn.head_dim).transpose(1, 2)
    logits = torch.matmul(query, key.transpose(-2, -1)) * (attn.head_dim ** -0.5)
    route_bonus_l2 = 0.0
    if attn.routed:
        bonus = attn.routing_bonus(bank.E0.to(tokens_last), bank.T.to(tokens_last), batch=1)
        logits = logits + bonus
        route_bonus_l2 = _l2(bonus)
    valid = bank.unit_mask.to(device=logits.device, dtype=torch.bool).view(1, 1, 1, -1)
    weights = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)[0]  # [H,S,N]
    active = int(bank.unit_mask.sum().item())
    entropy = -(weights.clamp_min(1e-30) * weights.clamp_min(1e-30).log()).sum(dim=-1)
    mean_slot = weights.mean(dim=0).double()  # [S,N]
    mean_slot = mean_slot / mean_slot.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    cosine = mean_slot @ mean_slot.T
    pair = cosine[~torch.eye(cosine.size(0), dtype=torch.bool, device=cosine.device)]
    singular = torch.linalg.svdvals(weights.reshape(-1, weights.size(-1)).double())
    sq = singular.square()
    effective_rank = float(sq.sum().square().div(sq.square().sum().clamp_min(1e-30)).item())
    threshold = float(singular.max().item()) * 1e-4
    return {
        "active_units": active,
        "attention_entropy_nats_mean": float(entropy.mean().item()),
        "attention_entropy_fraction_of_uniform_mean": float((entropy / np.log(active)).mean().item()),
        "attention_entropy_nats_by_head_slot": entropy.cpu().tolist(),
        "mean_slot_pairwise_cosine_mean": float(pair.mean().item()),
        "mean_slot_pairwise_cosine_min": float(pair.min().item()),
        "mean_slot_matrix_rank_threshold_1e-4": int((torch.linalg.svdvals(mean_slot) > threshold).sum().item()),
        "head_slot_attention_effective_rank": effective_rank,
        "head_slot_attention_singular_values": singular.cpu().tolist(),
        "head_slot_attention_near_zero_count_1e-4_relative": int((singular <= threshold).sum().item()),
        "route_bonus_l2": route_bonus_l2,
    }


def _activation_bank_metrics(model: torch.nn.Module, x: torch.Tensor, bank: H1Bank) -> dict:
    frontend = model.frontend
    with torch.no_grad():
        local = frontend.local_conv(x)
        local_static = frontend.local_conv(torch.zeros_like(x))
        token = frontend.token_norm(frontend.token_mlp(local, bank.E0.to(x), bank.T.to(x)))
        token_static = frontend.token_norm(frontend.token_mlp(local_static, bank.E0.to(x), bank.T.to(x)))
        return {
            "local_activity_minus_static_rms": _l2(local - local_static) / np.sqrt(local.numel()),
            "local_static_rms": _l2(local_static) / np.sqrt(local_static.numel()),
            "local_activity_to_static_rms_ratio": (_l2(local - local_static) / max(_l2(local_static), 1e-30)),
            "token_activity_minus_static_rms": _l2(token - token_static) / np.sqrt(token.numel()),
            "token_static_rms": _l2(token_static) / np.sqrt(token_static.numel()),
            "token_activity_to_static_rms_ratio": (_l2(token - token_static) / max(_l2(token_static), 1e-30)),
            "attention": _attention(model, token[:, -1], bank),
        }


def _module_name(name: str) -> str:
    if name.startswith("frontend.local_conv"):
        return "frontend.local_conv"
    if name.startswith("frontend.token_mlp"):
        return "frontend.token_mlp"
    if name.startswith("frontend.token_norm"):
        return "frontend.token_norm"
    if name.startswith("frontend.slot_norm") or name.startswith("frontend.slots"):
        return "frontend.slot_queries_and_norm"
    if name.startswith("frontend.attn"):
        return "frontend.slot_attention"
    if name.startswith("frontend.slot_ffn") or name.startswith("frontend.slot_ffn_norm"):
        return "frontend.slot_ffn"
    if name.startswith("frontend.slot_proj"):
        return "frontend.slot_projection"
    if name.startswith("temporal"):
        return "temporal"
    if name.startswith("final_norm"):
        return "final_norm"
    if name.startswith("readout"):
        return "readout"
    return "other"


def _update_l2(model: torch.nn.Module, initial: torch.nn.Module) -> dict[str, dict[str, float]]:
    before = dict(initial.named_parameters())
    out: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        key = _module_name(name)
        delta = param.detach().cpu() - before[name].detach().cpu()
        row = out.setdefault(key, {"delta_l2_sq": 0.0, "initial_l2_sq": 0.0, "parameter_count": 0.0})
        row["delta_l2_sq"] += float(delta.float().square().sum().item())
        row["initial_l2_sq"] += float(before[name].detach().cpu().float().square().sum().item())
        row["parameter_count"] += float(param.numel())
    for row in out.values():
        row["delta_l2"] = float(np.sqrt(row.pop("delta_l2_sq")))
        initial_l2 = float(np.sqrt(row.pop("initial_l2_sq")))
        row["delta_to_initial_l2"] = row["delta_l2"] / max(initial_l2, 1e-30)
        row["parameter_count"] = int(row["parameter_count"])
    return out


def _source_gradients(model: torch.nn.Module, cache: dict, fixed: dict[str, np.ndarray], device: torch.device) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    losses = []
    for name in sorted(fixed):
        row = cache["train"][name]
        x, target, _ = batch(row, fixed[name], device)
        bank = _bank(row, device)
        # The average over 13 actual source batches is one real source loss;
        # microbatching is only to keep the diagnostic's activation footprint small.
        session_loss = 0.0
        for offset in range(0, len(x), MICRO):
            loss = F.mse_loss(model.forward_last(x[offset:offset + MICRO], bank), target[offset:offset + MICRO])
            (loss / (len(fixed) * (len(x) // MICRO))).backward()
            session_loss += float(loss.detach().item())
        losses.append(session_loss / (len(x) // MICRO))
    grouped: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = _module_name(name)
        grouped[key] = grouped.get(key, 0.0) + float(param.grad.detach().float().square().sum().item())
    model.eval()
    return {"mean_per_session_batch_loss": float(np.mean(losses)), "gradient_l2_by_module": {key: float(np.sqrt(value)) for key, value in sorted(grouped.items())}}


def _jacobian(model: torch.nn.Module, x: torch.Tensor, bank: H1Bank) -> dict:
    """Exact FP32 d(frontend last token 256)/d(latest raw unit values 176)."""
    model.eval()
    x = x.detach().clone().requires_grad_(True)
    zlast = model.encode_frontend(x, bank)[:, -1, :]
    rows = []
    for column in range(zlast.size(-1)):
        grad = torch.autograd.grad(zlast[0, column], x, retain_graph=column + 1 < zlast.size(-1), create_graph=False)[0]
        rows.append(grad[0, -1, :].detach().cpu())
    jac = torch.stack(rows).float()  # [256,176], calculated in trained FP32 model
    singular = torch.linalg.svdvals(jac.double())
    threshold = float(singular.max().item()) * 1e-4
    sq = singular.square()
    return {
        "semantic": "FP32 autograd Jacobian d(frontend final token[256])/d(current raw x[176]); singular analysis float64",
        "shape": list(jac.shape),
        "frobenius_norm": float(torch.linalg.vector_norm(jac).item()),
        "singular_values_float64": singular.tolist(),
        "effective_rank": float(sq.sum().square().div(sq.square().sum().clamp_min(1e-30)).item()),
        "rank_threshold_1e-4_relative": int((singular > threshold).sum().item()),
        "near_zero_count_1e-4_relative": int((singular <= threshold).sum().item()),
        "max_singular": float(singular.max().item()),
        "min_singular": float(singular.min().item()),
    }


def cpu_stage() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic root: {OUT}")
    device = torch.device("cpu")
    cache = build_or_load()
    recorded = json.loads((H1_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, recorded)
    fixed = _fixed_ids()
    models, payloads, initial = _models(device)
    report = {
        "schema": "h1_crst_b4_scale1_pooling_bottleneck_diagnostic_v1",
        "status": "CPU_STAGE_COMPLETE_AWAITING_OPTIONAL_GPU_JACOBIAN",
        "read_only": True,
        "source_only": True,
        "minival_opened": False,
        "preflight_report_sha256": _sha(ATTEMPT / "report_260.json"),
        "checkpoint_sha256": {arm: _sha(ATTEMPT / f"{arm}_260.pt") for arm in models},
        "fixed_ids_sha256": _sha(IDS_PATH),
        "cache_authority": recorded,
        "bank_window_declaration": "For each source bank: the first of the sealed 16 source endpoints and its full W=700 raw window.  The activity/static RMS quantities in this CPU stage aggregate all 700 positions; they are not last-five-only statistics.",
        "arms": {},
    }
    for arm, model in models.items():
        rows = {}
        for name in sorted(fixed):
            row = cache["train"][name]
            x, _, _ = batch(row, fixed[name][:1], device)
            rows[name] = _activation_bank_metrics(model, x, _bank(row, device))
        report["arms"][arm] = {
            "per_source_bank": rows,
            "aggregate": _aggregate(rows),
            "parameter_update_l2_by_module": _update_l2(model, initial[arm]),
            "actual_source_loss_gradient": _source_gradients(model, cache, fixed, device),
            "route_gate": (None if arm == "flat" else {"values": model.frontend.attn.g.detach().cpu().tolist(), "abs_max": float(model.frontend.attn.g.detach().abs().max().item()), "l2": _l2(model.frontend.attn.g)}),
        }
    OUT.mkdir(parents=True)
    (OUT / "cpu_stage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _aggregate(rows: dict[str, dict]) -> dict:
    keys = ["local_activity_minus_static_rms", "local_static_rms", "local_activity_to_static_rms_ratio", "token_activity_minus_static_rms", "token_static_rms", "token_activity_to_static_rms_ratio"]
    output = {key: {"mean": float(np.mean([row[key] for row in rows.values()])), "min": float(np.min([row[key] for row in rows.values()])), "max": float(np.max([row[key] for row in rows.values()]))} for key in keys}
    attention_keys = ["attention_entropy_nats_mean", "attention_entropy_fraction_of_uniform_mean", "mean_slot_pairwise_cosine_mean", "mean_slot_pairwise_cosine_min", "mean_slot_matrix_rank_threshold_1e-4", "head_slot_attention_effective_rank", "head_slot_attention_near_zero_count_1e-4_relative", "route_bonus_l2"]
    output["attention"] = {key: {"mean": float(np.mean([row["attention"][key] for row in rows.values()])), "min": float(np.min([row["attention"][key] for row in rows.values()])), "max": float(np.max([row["attention"][key] for row in rows.values()]))} for key in attention_keys}
    return output


def gpu_jacobian() -> None:
    cpu = json.loads((OUT / "cpu_stage.json").read_text())
    if cpu["status"] != "CPU_STAGE_COMPLETE_AWAITING_OPTIONAL_GPU_JACOBIAN":
        raise RuntimeError("unexpected CPU stage state")
    device = torch.device("cuda:0")
    cache = build_or_load()
    fixed = _fixed_ids()
    models, _, _ = _models(device)
    name = sorted(fixed)[0]
    row = cache["train"][name]
    x, _, _ = batch(row, fixed[name][:1], device)
    bank = _bank(row, device)
    last5 = x[:, -5:].contiguous()
    equivalence = {}
    for arm, model in models.items():
        with torch.no_grad():
            full_last = model.encode_frontend(x, bank)[:, -1]
            local_last = model.encode_frontend(last5, bank)[:, -1]
        maximum = float((full_last - local_last).abs().max().item())
        if not torch.allclose(full_last, local_last, atol=1e-5, rtol=1e-4):
            raise RuntimeError(f"causal-k5 frontend final-token equivalence failed for {arm}: {maximum}")
        equivalence[arm] = {"full700_vs_last5_final_token_max_abs": maximum, "atol": 1e-5, "rtol": 1e-4}
    cpu["jacobian_window"] = {"session": name, "query_start": int(fixed[name][0]), "latest_raw_bin": int(fixed[name][0] + 699)}
    cpu["jacobian_input_reduction"] = {"law": "asserted exact final frontend-token equivalence: full W=700 vs causal last five raw bins; Jacobian then uses only last5 because SharedSetFrontend is per-time-step and local Conv is causal k5", "equivalence": equivalence}
    cpu["jacobian"] = {arm: _jacobian(model, last5, bank) for arm, model in models.items()}
    cpu["status"] = "COMPLETE_READ_ONLY"
    (OUT / "report.json").write_text(json.dumps(cpu, indent=2, sort_keys=True) + "\n")


def last5_activation_supplement() -> None:
    """Small read-only supplement for the literal final-five-bin request."""
    cpu = json.loads((OUT / "cpu_stage.json").read_text())
    device = torch.device("cpu")
    cache = build_or_load()
    fixed = _fixed_ids()
    models, _, _ = _models(device)
    supplement = {
        "schema": "h1_crst_b4_scale1_pooling_bottleneck_last5_supplement_v1",
        "read_only": True,
        "source_only": True,
        "cpu_stage_sha256": _sha(OUT / "cpu_stage.json"),
        "declaration": "Same first sealed endpoint per source bank as CPU stage; only the literal final five raw bins are passed.  These are valid for local/token activity-static quantities because the shared causal local Conv has k=5; no temporal/readout quantity is claimed here.",
        "arms": {},
    }
    for arm, model in models.items():
        rows = {}
        for name in sorted(fixed):
            row = cache["train"][name]
            x, _, _ = batch(row, fixed[name][:1], device)
            rows[name] = _activation_bank_metrics(model, x[:, -5:].contiguous(), _bank(row, device))
        supplement["arms"][arm] = {"per_source_bank": rows, "aggregate": _aggregate(rows)}
    (OUT / "last5_activation_supplement.json").write_text(json.dumps(supplement, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cpu", "gpu-jacobian", "last5"), required=True)
    args = parser.parse_args()
    if args.stage == "cpu":
        cpu_stage()
    elif args.stage == "gpu-jacobian":
        gpu_jacobian()
    else:
        last5_activation_supplement()
