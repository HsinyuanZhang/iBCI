"""Bounded CPU-only source diagnostic for CRST-B4 token block balance.

This makes no parameter update.  It audits one declared source bank and tests
one *initialization-only* algebraic reparameterization of the existing local
FC1 block.  It is not a training authorization or an architecture result.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT as H1_ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.capacity_probe import batch
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from .model import make_v2_unscaled_dot_pair


ATTEMPT = H1_ROOT / "family_v1" / "crst_b4_set_v2_unscaled_dot_source208_preflight_v1"
IDS = H1_ROOT / "capacity_probe_208_source_v2" / "frozen_ids.json"
OUT = H1_ROOT / "family_v1" / "crst_b4_set_v2_local_block_balance_diagnostic_v1"
SEED = 42
CPU_TOTAL_LIMIT_S = 180.0
TRAINED_JACOBIAN_LIMIT_S = 60.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().square().mean().sqrt().item())


def _fixed() -> tuple[str, int]:
    if sha(IDS) != "da4bf975a5c1023bbffe26da87d0d4977f437d7db1db012283511ef140d96211":
        raise RuntimeError("sealed source IDs changed")
    ids = json.loads(IDS.read_text())["ids"]
    name = sorted(ids)[0]
    return name, int(ids[name][0])


def _bank(row: dict) -> H1Bank:
    return H1Bank(row["bank"]["E0"], row["bank"]["T"], row["bank"]["unit_mask"])


def _load_trained() -> torch.nn.Module:
    flat, _ = make_v2_unscaled_dot_pair(seed=SEED)
    payload = torch.load(ATTEMPT / "flat_260.pt", map_location="cpu", weights_only=False)
    if int(payload["updates"]) != 260:
        raise RuntimeError("wrong trained checkpoint")
    flat.load_state_dict(payload["model"], strict=True)
    return flat.eval()


def _attention(model: torch.nn.Module, x5: torch.Tensor, bank: H1Bank) -> dict:
    frontend = model.frontend
    with torch.no_grad():
        local = frontend.local_conv(x5)
        tokens = frontend.token_norm(frontend.token_mlp(local, bank.E0.to(x5), bank.T.to(x5)))[:, -1]
        attn = frontend.attn
        slots = frontend.slot_norm(frontend.slots).unsqueeze(0)
        q = attn.q_proj(slots).view(1, frontend.cfg.slots, attn.n_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(tokens).view(1, tokens.size(1), attn.n_heads, attn.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (attn.head_dim ** -0.5)
        if attn.routed:
            logits = logits + attn.routing_bonus(bank.E0.to(x5), bank.T.to(x5), batch=1)
        if getattr(attn, "logit_multiplier_name", None) == "sqrt_head_dim":
            logits = logits * (attn.head_dim ** 0.5)
        keep = bank.unit_mask.bool().view(1, 1, 1, -1)
        weights = torch.softmax(logits.masked_fill(~keep, float("-inf")), dim=-1)[0]
        entropy = -(weights.clamp_min(1e-30) * weights.clamp_min(1e-30).log()).sum(dim=-1)
        matrix = weights.reshape(-1, weights.size(-1)).double()
        singular = torch.linalg.svdvals(matrix)
        sq = singular.square()
        slot = weights.mean(dim=0).double(); slot = slot / slot.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        cosine = slot @ slot.T
        pair = cosine[~torch.eye(cosine.size(0), dtype=torch.bool)]
    return {
        "entropy_fraction_uniform": float((entropy / math.log(int(bank.unit_mask.sum()))).mean().item()),
        "head_slot_effective_rank": float(sq.sum().square().div(sq.square().sum().clamp_min(1e-30)).item()),
        "mean_slot_pairwise_cosine": float(pair.mean().item()),
    }


def _jacobian(model: torch.nn.Module, x5: torch.Tensor, bank: H1Bank, limit_s: float) -> dict:
    started = time.monotonic()
    x = x5.detach().clone().requires_grad_(True)
    z = model.encode_frontend(x, bank)[:, -1]
    rows = []
    for i in range(z.size(-1)):
        if time.monotonic() - started > limit_s:
            raise TimeoutError("trained frontend Jacobian exceeded its 60-second CPU bound")
        gradient = torch.autograd.grad(z[0, i], x, retain_graph=i + 1 < z.size(-1))[0]
        rows.append(gradient[0, -1].detach())
    jac = torch.stack(rows).float()
    singular = torch.linalg.svdvals(jac.double())
    sq = singular.square()
    cutoff = float(singular.max().item()) * 1e-4
    return {
        "semantic": "true FP32 autograd d(frontend final token[256])/d(latest raw x[176]); SVD analysis float64; x input is final five bins after exact causal-k5 reduction",
        "shape": list(jac.shape),
        "elapsed_s": time.monotonic() - started,
        "effective_rank": float(sq.sum().square().div(sq.square().sum().clamp_min(1e-30)).item()),
        "rank_above_1e-4_relative": int((singular > cutoff).sum().item()),
        "near_zero_count_1e-4_relative": int((singular <= cutoff).sum().item()),
        "max_singular": float(singular.max().item()),
        "min_singular": float(singular.min().item()),
    }


def _causal_last5_equivalence(model: torch.nn.Module, x: torch.Tensor, x5: torch.Tensor, bank: H1Bank) -> float:
    with torch.no_grad():
        full = model.encode_frontend(x, bank)[:, -1]
        short = model.encode_frontend(x5, bank)[:, -1]
    maximum = float((full - short).abs().max().item())
    if not torch.allclose(full, short, atol=1e-5, rtol=1e-4):
        raise RuntimeError(f"causal-k5 final frontend token mismatch: {maximum}")
    return maximum


def _fc1_blocks(model: torch.nn.Module, x: torch.Tensor, bank: H1Bank) -> dict:
    frontend = model.frontend
    with torch.no_grad():
        local = frontend.local_conv(x)
        local0 = frontend.local_conv(torch.zeros_like(x))
        weight = frontend.token_mlp.fc1.weight
        w_local, w_e0, w_t = weight.split([16, 700, 4], dim=1)
        e0 = F.linear(bank.E0, w_e0, None).view(1, 1, 176, -1)
        t = F.linear(bank.T, w_t, None).view(1, 1, 176, -1)
        dynamic = F.linear(local, w_local, None) - F.linear(local0, w_local, None)
        local_static = F.linear(local0, w_local, None)
        static = local_static + e0 + t + frontend.token_mlp.fc1.bias.view(1, 1, 1, -1)
        token = frontend.token_norm(frontend.token_mlp(local, bank.E0, bank.T))
        token0 = frontend.token_norm(frontend.token_mlp(local0, bank.E0, bank.T))
        bound = 1.0 / math.sqrt(720)
    return {
        "fc1_full_fan_in": 720,
        "block_dims": {"local": 16, "E0": 700, "T": 4},
        "uniform_kaiming_a_sqrt5_expected_bound": bound,
        "weight_rms": {"local": rms(w_local), "E0": rms(w_e0), "T": rms(w_t)},
        "weight_rms_ratio_local_to_E0": rms(w_local) / rms(w_e0),
        "projection_rms": {"local_dynamic": rms(dynamic), "local_static_conv_bias_path": rms(local_static), "E0_static": rms(e0), "T_static": rms(t), "combined_static_pre_gelu": rms(static)},
        "token_rms": {"activity_minus_static": rms(token - token0), "static": rms(token0), "activity_to_static": rms(token - token0) / max(rms(token0), 1e-30)},
    }


def _rescaled_init() -> torch.nn.Module:
    flat, _ = make_v2_unscaled_dot_pair(seed=SEED)
    factor = math.sqrt(720 / 16)
    with torch.no_grad():
        flat.frontend.token_mlp.fc1.weight[:, :16].mul_(factor)
    flat.eval()
    flat.local_block_rescale_factor = factor
    return flat


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    cache = build_or_load()
    authority = json.loads((H1_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, authority)
    name, start = _fixed()
    row = cache["train"][name]
    x, _, _ = batch(row, np.asarray([start]), torch.device("cpu"))
    x5 = x[:, -5:].contiguous()
    bank = _bank(row)
    trained = _load_trained()
    report = {
        "schema": "h1_crst_b4_set_v2_local_block_balance_diagnostic_v1",
        "status": "RUNNING",
        "read_only": True,
        "source_only": True,
        "minival_opened": False,
        "limits": {"total_cpu_seconds": CPU_TOTAL_LIMIT_S, "trained_jacobian_seconds": TRAINED_JACOBIAN_LIMIT_S},
        "declared_window": {"session": name, "query_start": start, "last5_only_for_attention_and_jacobian": True},
        "bindings": {"trained_checkpoint_sha256": sha(ATTEMPT / "flat_260.pt"), "source_ids_sha256": sha(IDS), "cache_authority": authority, "code_sha256": sha(Path(__file__))},
        "trained_set_v2": {"full700_vs_last5_frontend_last_max_abs": _causal_last5_equivalence(trained, x, x5, bank), "attention_last5": _attention(trained, x5, bank), "jacobian_last5": _jacobian(trained, x5, bank, TRAINED_JACOBIAN_LIMIT_S), "fc1_blocks_full700": _fc1_blocks(trained, x, bank)},
    }
    if time.monotonic() - started > CPU_TOTAL_LIMIT_S:
        raise TimeoutError("CPU budget exhausted before init reparameterization")
    baseline, _ = make_v2_unscaled_dot_pair(seed=SEED)
    baseline.eval()
    rescaled = _rescaled_init()
    report["initialization_only_comparison"] = {
        "baseline_v2_unscaled_dot": {"full700_vs_last5_frontend_last_max_abs": _causal_last5_equivalence(baseline, x, x5, bank), "attention_last5": _attention(baseline, x5, bank), "jacobian_last5": _jacobian(baseline, x5, bank, CPU_TOTAL_LIMIT_S - (time.monotonic() - started)), "fc1_blocks_full700": _fc1_blocks(baseline, x, bank)},
        "proposed_local_block_rescaled": {"formula": "W_local <- sqrt(720/16) * W_local; E0/T blocks and all other seed-42 tensors unchanged", "factor": math.sqrt(720 / 16), "full700_vs_last5_frontend_last_max_abs": _causal_last5_equivalence(rescaled, x, x5, bank), "attention_last5": _attention(rescaled, x5, bank), "jacobian_last5": _jacobian(rescaled, x5, bank, CPU_TOTAL_LIMIT_S - (time.monotonic() - started)), "fc1_blocks_full700": _fc1_blocks(rescaled, x, bank)},
    }
    report["elapsed_s"] = time.monotonic() - started
    if report["elapsed_s"] > CPU_TOTAL_LIMIT_S:
        raise TimeoutError("CPU bound exceeded")
    report["status"] = "COMPLETE_READ_ONLY"
    OUT.mkdir(parents=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"elapsed_s": report["elapsed_s"], "trained": report["trained_set_v2"]["attention_last5"], "rescaled": report["initialization_only_comparison"]["proposed_local_block_rescaled"]["attention_last5"]}, sort_keys=True))


if __name__ == "__main__":
    main()
