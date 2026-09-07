"""Frozen CRST-B4 H1 source-only 260/1040 learnability gate.

No minival iterator exists in this module.  It exclusively opens the owned
source cache's ``train`` rows and refuses a nonempty attempt root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT as H1_ROOT, authority, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.capacity_probe import batch, ids, r2
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from .model import initialization_receipt, make_v2_initialized_pair, make_v2_unscaled_dot_pair, make_v2_unscaled_dot_localbalanced_pair, route_gate_gradient_l1, zero_gate_parity


OUT = H1_ROOT / "family_v1" / "crst_b4_scale1_source208_preflight_v1"
V2_IDS = H1_ROOT / "capacity_probe_208_source_v2" / "frozen_ids.json"
SEED = 42
W = 700
LR = 2e-4
MICRO = 4
EFFECTIVE = 16
REPORT_UPDATES = 260
EXTEND_UPDATES = 1040
SAFETY_SECONDS_PER_ARM = 1800
VARIANTS = {
    "scale1_baseline": {"factory": make_v2_initialized_pair, "name": "CRST-B4-common-set-v1", "rule": "legacy scaled qk/sqrt(head_dim), scale1 source gate"},
    "set_v2_unscaled_dot": {"factory": make_v2_unscaled_dot_pair, "name": "CRST-B4-common-set-v2-unscaled-dot", "rule": "(qk/sqrt(head_dim) + legal route bonus) * sqrt(head_dim) before unchanged mask/softmax"},
    "set_v2_unscaled_dot_localbalanced": {"factory": make_v2_unscaled_dot_localbalanced_pair, "name": "CRST-B4-common-set-v3-unscaled-dot-localbalanced-init", "rule": "set-v2 unscaled-dot plus initialization-only W_fc1_local *= sqrt((16+700+4)/16)=sqrt(45); E0/T blocks and all runtime operators unchanged"},
}
UNSCALED_PROTOCOL = H1_ROOT / "family_v1" / "crst_b4_set_v2_unscaled_dot_source208_preflight_protocol_v1.json"
LOCALBALANCED_MANIFEST = H1_ROOT / "family_v1" / "crst_b4_set_v2_localbalanced_prospective_v1" / "manifest.json"
LOCALBALANCED_AUTHORIZATION = H1_ROOT / "family_v1" / "crst_b4_set_v2_localbalanced_prospective_v1" / "launch_authorization.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_ids(cache: dict) -> tuple[dict[str, np.ndarray], dict]:
    frozen = json.loads(V2_IDS.read_text())
    expected = "da4bf975a5c1023bbffe26da87d0d4977f437d7db1db012283511ef140d96211"
    if sha(V2_IDS) != expected:
        raise RuntimeError("the V2 frozen IDs file hash differs from the frozen protocol")
    fixed = {name: np.asarray(starts, dtype=np.int64) for name, starts in frozen["ids"].items()}
    recomputed = ids(cache)
    if set(fixed) != set(recomputed) or any(not np.array_equal(fixed[k], recomputed[k]) for k in fixed):
        raise RuntimeError("fixed source IDs are not exactly the V2 deterministic 208 endpoints")
    if sum(len(v) for v in fixed.values()) != 208 or any(len(v) != 16 for v in fixed.values()):
        raise RuntimeError("expected 13 sessions x 16 fixed source endpoints")
    return fixed, frozen


def _score(model: torch.nn.Module, cache: dict, fixed: dict[str, np.ndarray], device: torch.device) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for name, starts in sorted(fixed.items()):
            row = cache["train"][name]
            x, _, native = batch(row, starts, device)
            bank = H1Bank(row["bank"]["E0"].to(device), row["bank"]["T"].to(device), row["bank"]["unit_mask"].to(device))
            for offset in range(0, len(x), MICRO):
                predictions.append((model.forward_last(x[offset:offset + MICRO], bank) / 20.0).cpu().numpy())
                targets.append(native[offset:offset + MICRO])
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    return {
        "r2_concat": r2(pred, target),
        "prediction_std": float(pred.std()),
        "target_std": float(target.std()),
        "prediction_mean": float(pred.mean()),
        "target_mean": float(target.mean()),
    }


def _initial_attention_summary(model: torch.nn.Module, cache: dict, fixed: dict[str, np.ndarray], device: torch.device) -> dict[str, float]:
    """Read-only last-token attention facts on one declared endpoint per bank."""
    entropies = []
    ranks = []
    cosines = []
    model.eval()
    with torch.no_grad():
        for name in sorted(fixed):
            row = cache["train"][name]
            x, _, _ = batch(row, fixed[name][:1], device)
            bank = H1Bank(row["bank"]["E0"].to(device), row["bank"]["T"].to(device), row["bank"]["unit_mask"].to(device))
            frontend = model.frontend
            local = frontend.local_conv(x)
            tokens = frontend.token_norm(frontend.token_mlp(local, bank.E0.to(x), bank.T.to(x)))[:, -1]
            attn = frontend.attn
            slots = frontend.slot_norm(frontend.slots).unsqueeze(0)
            q = attn.q_proj(slots).view(1, frontend.cfg.slots, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(tokens).view(1, tokens.size(1), attn.n_heads, attn.head_dim).transpose(1, 2)
            logits = torch.matmul(q, k.transpose(-2, -1)) * (attn.head_dim ** -0.5)
            if attn.routed:
                logits = logits + attn.routing_bonus(bank.E0.to(x), bank.T.to(x), batch=1)
            # The model's rule is read from its class rather than inferred from
            # labels: only common-set-v2 has this multiplier.
            if getattr(attn, "logit_multiplier_name", None) == "sqrt_head_dim":
                logits = logits * (attn.head_dim ** 0.5)
            keep = bank.unit_mask.to(device=device, dtype=torch.bool).view(1, 1, 1, -1)
            weights = torch.softmax(logits.masked_fill(~keep, float("-inf")), dim=-1)[0]
            entropy = -(weights.clamp_min(1e-30) * weights.clamp_min(1e-30).log()).sum(dim=-1)
            entropies.append(float((entropy / np.log(int(bank.unit_mask.sum()))).mean().item()))
            mat = weights.reshape(-1, weights.size(-1)).double()
            singular = torch.linalg.svdvals(mat)
            sq = singular.square(); ranks.append(float(sq.sum().square().div(sq.square().sum().clamp_min(1e-30)).item()))
            slots_mean = weights.mean(dim=0).double(); slots_mean = slots_mean / slots_mean.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            cosine = slots_mean @ slots_mean.T
            cosines.append(float(cosine[~torch.eye(cosine.size(0), dtype=torch.bool, device=device)].mean().item()))
    return {"attention_entropy_fraction_uniform_mean": float(np.mean(entropies)), "head_slot_attention_effective_rank_mean": float(np.mean(ranks)), "mean_slot_pairwise_cosine_mean": float(np.mean(cosines))}


def _stable_decline(losses: list[float]) -> bool:
    # Fixed, intentionally weak optimizer-health condition: means of the last
    # and first 32 updates.  No score is selected or tuned with this condition.
    if len(losses) < 64 or not all(np.isfinite(losses)):
        return False
    return float(np.mean(losses[-32:])) < float(np.mean(losses[:32]))


def _checkpoint(model: torch.nn.Module, opt: torch.optim.Optimizer, updates: int, losses: list[float]) -> dict:
    return {"model": model.state_dict(), "optimizer": opt.state_dict(), "updates": updates, "losses": losses}


def main(*, attempt: str = OUT.name, extend_only: bool = False, variant: str = "scale1_baseline") -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    spec = VARIANTS[variant]
    if variant == "set_v2_unscaled_dot_localbalanced":
        if not LOCALBALANCED_AUTHORIZATION.is_file():
            raise FileNotFoundError("localbalanced launch authorization is required before any source update")
        launch = json.loads(LOCALBALANCED_AUTHORIZATION.read_text())
        actual_hashes = {"manifest.json": sha(LOCALBALANCED_MANIFEST), "model.py": sha(Path(__file__).with_name("model.py")), "source_preflight.py": sha(Path(__file__))}
        if launch.get("attempt") != attempt or launch.get("variant") != variant or launch.get("bindings") != actual_hashes:
            raise RuntimeError("localbalanced launch authorization does not bind this exact attempt/operator")
    out = OUT.parent / attempt
    if out.exists() and any(out.iterdir()) and not extend_only:
        raise FileExistsError(f"refusing to overwrite nonempty preflight root: {out}")
    if extend_only and not (out / "report_260.json").is_file():
        raise FileNotFoundError("an exact 260-stage report is required for extension")
    out.mkdir(parents=True, exist_ok=True)
    cache = build_or_load()
    recorded = json.loads((H1_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, recorded)
    fixed, frozen = _fixed_ids(cache)
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    flat, route = spec["factory"](seed=SEED)
    arms = {"flat": flat.to(device), "route": route.to(device)}
    opts = {name: torch.optim.AdamW(groups(model), lr=LR) for name, model in arms.items()}
    losses = {name: [] for name in arms}
    done = 0
    preflight: dict = {
        "schema": "h1_crst_b4_scale1_source208_preflight_v1",
        "status": "RUNNING",
        "source_only": True,
        "minival_opened": False,
        "cache_authority": recorded,
        "cache_code_authority": authority(cache)["code_sha256"],
        "v2_frozen_ids_sha256": sha(V2_IDS),
        "sessions": len(fixed),
        "windows": sum(len(v) for v in fixed.values()),
        "model_variant": {"id": variant, "name": spec["name"], "only_frontend_rule_change": spec["rule"]},
        "operator_code_sha256": {
            "model.py": sha(Path(__file__).with_name("model.py")),
            "source_preflight.py": sha(Path(__file__)),
        },
        "recipe": {"seed": SEED, "activity_scale": 1.0, "W": W, "target_multiplier": 20, "lr": LR, "microbatch": MICRO, "effective_batch": EFFECTIVE, "dropout": 0.0, "reporting_updates": REPORT_UPDATES, "extension_updates": EXTEND_UPDATES, "per_arm_safety_seconds": SAFETY_SECONDS_PER_ARM},
        "initialization": initialization_receipt(flat, route),
    }
    if variant in {"set_v2_unscaled_dot", "set_v2_unscaled_dot_localbalanced"}:
        preflight["frozen_protocol_sha256"] = sha(UNSCALED_PROTOCOL)
    if variant == "set_v2_unscaled_dot_localbalanced":
        preflight["frozen_localbalanced_manifest_sha256"] = sha(LOCALBALANCED_MANIFEST)
        preflight["launch_authorization_sha256"] = sha(LOCALBALANCED_AUTHORIZATION)
    first_name = sorted(fixed)[0]
    first_row = cache["train"][first_name]
    first_x, first_y, _ = batch(first_row, fixed[first_name][:MICRO], device)
    first_bank = H1Bank(first_row["bank"]["E0"].to(device), first_row["bank"]["T"].to(device), first_row["bank"]["unit_mask"].to(device))
    preflight["zero_gate_parity"] = zero_gate_parity(flat, route, first_x, first_bank)
    route.train(); opts["route"].zero_grad(set_to_none=True)
    F.mse_loss(route.forward_last(first_x, first_bank), first_y).backward()
    gate_grad = route_gate_gradient_l1(route)
    opts["route"].zero_grad(set_to_none=True)
    if gate_grad <= 0.0:
        raise RuntimeError("ROUTE gate gradient is zero at the required g=0 preflight")
    preflight["route_gate_gradient_l1_at_g0"] = gate_grad
    preflight["sealed_bank_initial_attention"] = {name: _initial_attention_summary(model, cache, fixed, device) for name, model in arms.items()}
    if variant in {"set_v2_unscaled_dot", "set_v2_unscaled_dot_localbalanced"}:
        baseline_flat, baseline_route = make_v2_initialized_pair(seed=SEED)
        baseline = {"flat": baseline_flat.to(device), "route": baseline_route.to(device)}
        preflight["sealed_bank_initial_attention_scale1_baseline"] = {name: _initial_attention_summary(model, cache, fixed, device) for name, model in baseline.items()}
        del baseline
    (out / "input_authority.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")

    if extend_only:
        stage = json.loads((out / "report_260.json").read_text())
        if not stage["extension_eligible"]:
            raise RuntimeError("260-stage receipt is not extension eligible")
        for name in arms:
            payload = torch.load(out / f"{name}_260.pt", map_location=device, weights_only=False)
            arms[name].load_state_dict(payload["model"])
            opts[name].load_state_dict(payload["optimizer"])
            losses[name] = list(payload["losses"])
        done = REPORT_UPDATES

    # V2's capacity gate made one effective-16 update from a session's entire
    # fixed endpoint block, accumulating four microbatches of four.  Keep that
    # geometry exactly; cycling microbatches as independent updates would be a
    # recipe change.
    names = sorted(fixed)
    arm_elapsed = {name: 0.0 for name in arms}
    start = time.monotonic()
    try:
        target_updates = EXTEND_UPDATES if extend_only else REPORT_UPDATES
        while done < target_updates:
            name = names[done % len(names)]
            row = cache["train"][name]
            x, target, _ = batch(row, fixed[name], device)
            bank = H1Bank(row["bank"]["E0"].to(device), row["bank"]["T"].to(device), row["bank"]["unit_mask"].to(device))
            for arm, model in arms.items():
                arm_start = time.monotonic()
                model.train(); opts[arm].zero_grad(set_to_none=True)
                pieces = []
                for offset in range(0, EFFECTIVE, MICRO):
                    loss = F.mse_loss(model.forward_last(x[offset:offset + MICRO], bank), target[offset:offset + MICRO])
                    (loss * (MICRO / EFFECTIVE)).backward()
                    pieces.append(float(loss.detach().item()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opts[arm].step()
                losses[arm].append(float(np.mean(pieces)))
                arm_elapsed[arm] += time.monotonic() - arm_start
                if arm_elapsed[arm] > SAFETY_SECONDS_PER_ARM:
                    raise TimeoutError(f"{arm} exceeded safety limit; runtime is reported separately from quality")
            done += 1
            if done % 20 == 0:
                (out / "stage_log.json").write_text(json.dumps({"updates_completed": done, "wall_elapsed_s": time.monotonic() - start, "per_arm_elapsed_s": arm_elapsed, "last_loss": {k: v[-1] for k, v in losses.items()}}, indent=2, sort_keys=True) + "\n")
    except BaseException as error:
        preflight.update({"status": "INTERRUPTED_OR_SAFETY_STOP", "updates_completed": done, "error": repr(error), "wall_elapsed_s": time.monotonic() - start, "per_arm_elapsed_s": arm_elapsed})
        for name in arms:
            torch.save(_checkpoint(arms[name], opts[name], done, losses[name]), out / f"{name}_latest.pt")
        (out / "failure_receipt.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
        raise

    after = {name: _score(model, cache, fixed, device) for name, model in arms.items()}
    loss_decline = {name: _stable_decline(values) for name, values in losses.items()}
    eligible = any(after[name]["r2_concat"] >= 0.10 and after[name]["prediction_std"] >= 0.25 * after[name]["target_std"] and loss_decline[name] for name in arms)
    label = "report_1040.json" if extend_only else "report_260.json"
    preflight.update({"status": "COMPLETE", "updates_completed": done, "wall_elapsed_s": time.monotonic() - start, "per_arm_elapsed_s": arm_elapsed, "after": after, "loss": {name: {"first32_mean": float(np.mean(values[:32])), "last32_mean": float(np.mean(values[-32:])), "stable_decline": loss_decline[name]} for name, values in losses.items()}, "extension_eligible": eligible if not extend_only else None, "meaningful_learnability_pass": (all(after[name]["r2_concat"] >= .5 and after[name]["prediction_std"] >= .5 * after[name]["target_std"] for name in arms) if extend_only else None)})
    for name in arms:
        torch.save(_checkpoint(arms[name], opts[name], done, losses[name]), out / f"{name}_{done}.pt")
        torch.save(_checkpoint(arms[name], opts[name], done, losses[name]), out / f"{name}_latest.pt")
    (out / label).write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"attempt": attempt, "stage": done, "after": after, "extension_eligible": preflight["extension_eligible"], "meaningful_learnability_pass": preflight["meaningful_learnability_pass"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default=OUT.name)
    parser.add_argument("--extend-only", action="store_true")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="scale1_baseline")
    args = parser.parse_args()
    main(attempt=args.attempt, extend_only=args.extend_only, variant=args.variant)
