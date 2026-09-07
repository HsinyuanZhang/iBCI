"""No-update CPU feasibility receipt for the prospective local-balanced pair."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT as H1_ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.capacity_probe import batch
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from .model import initialization_receipt, local_fc1_balance_factor, make_v2_unscaled_dot_localbalanced_pair, route_gate_gradient_l1, zero_gate_parity


OUT = H1_ROOT / "family_v1" / "crst_b4_set_v2_localbalanced_prospective_v1"
IDS = H1_ROOT / "capacity_probe_208_source_v2" / "frozen_ids.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT / "cpu_one_step_feasibility.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    cache = build_or_load()
    authority = json.loads((H1_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, authority)
    if sha(IDS) != "da4bf975a5c1023bbffe26da87d0d4977f437d7db1db012283511ef140d96211":
        raise RuntimeError("frozen source IDs changed")
    ids = json.loads(IDS.read_text())["ids"]
    name = sorted(ids)[0]
    row = cache["train"][name]
    starts = np.asarray(ids[name], dtype=np.int64)
    device = torch.device("cpu")
    x, target, _ = batch(row, starts, device)
    bank = H1Bank(row["bank"]["E0"], row["bank"]["T"], row["bank"]["unit_mask"])
    flat, route = make_v2_unscaled_dot_localbalanced_pair(seed=42)
    result = {
        "schema": "h1_crst_b4_localbalanced_cpu_one_step_feasibility_v1",
        "status": "RUNNING",
        "read_only_no_optimizer_step": True,
        "source_only": True,
        "minival_opened": False,
        "source_ids_sha256": sha(IDS),
        "cache_authority": authority,
        "operator_code_sha256": {"model.py": sha(Path(__file__).with_name("model.py")), "preflight.py": sha(Path(__file__))},
        "local_fc1_factor": local_fc1_balance_factor(flat.cfg),
        "initialization": initialization_receipt(flat, route),
        "zero_gate_parity": zero_gate_parity(flat, route, x[:4], bank),
    }
    started = time.monotonic()
    arms = {"flat": flat, "route": route}
    for arm, model in arms.items():
        model.train(); model.zero_grad(set_to_none=True)
        # Exact prospective geometry: four microbatches, effective source batch 16.
        pieces = []
        for offset in range(0, 16, 4):
            loss = F.mse_loss(model.forward_last(x[offset:offset + 4], bank), target[offset:offset + 4])
            (loss * .25).backward(); pieces.append(float(loss.detach().item()))
        result.setdefault("loss_before_any_update", {})[arm] = float(np.mean(pieces))
    result["route_gate_gradient_l1_at_g0"] = route_gate_gradient_l1(route)
    if result["route_gate_gradient_l1_at_g0"] <= 0.0:
        raise RuntimeError("zero route gate gradient")
    result["elapsed_s"] = time.monotonic() - started
    result["status"] = "COMPLETE_NO_UPDATE"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"elapsed_s": result["elapsed_s"], "loss": result["loss_before_any_update"], "gate_grad": result["route_gate_gradient_l1_at_g0"]}, sort_keys=True))


if __name__ == "__main__":
    main()
