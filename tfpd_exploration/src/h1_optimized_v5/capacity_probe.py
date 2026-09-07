"""Predeclared source-only V5 gate: fixed 208 train endpoints, 1040 updates.

This module never opens minival data, never selects an epoch, and never
warms from V4/probe weights.  It is intentionally not executed on import.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.capacity_probe import batch, ids, r2
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .model import make_matched_pair

OUT = ROOT / "capacity_probe_208_source_v5_logage1040"
LR, UPDATES, MICRO, EFFECTIVE, DEADLINE = 2e-4, 1040, 4, 16, 600


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def score(model, cache, fixed, device):
    prediction, target = [], []
    model.eval()
    with torch.no_grad():
        for name, starts in fixed.items():
            row = cache["train"][name]; x, _, y = batch(row, starts, device)
            bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
            prediction.extend((model.forward_last(x[index:index + MICRO], bank) / 20).cpu().numpy() for index in range(0, len(x), MICRO))
            target.extend(y[index:index + MICRO] for index in range(0, len(y), MICRO))
    p, y = np.concatenate(prediction), np.concatenate(target)
    return {"r2_concat": r2(p, y), "prediction_std": float(p.std()), "target_std": float(y.std())}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite a V5 gate: {OUT}")
    cache = build_or_load(); authority = json.loads((ROOT / "source_cache_authority.json").read_text()); validate_authority(cache, authority)
    fixed = ids(cache)
    recipe = {"schema": "h1_v5_logage_source_gate_v1", "iterative_design_disclosure": "known-source capacity gate for a predeclared single temporal-variable design; not a formal or hidden/test score", "variable": "V5 T uses read-time fixed log16 age bucket indices; V4 signed frontend and matched FULL remain unchanged", "frontend_contract_version": 4, "temporal_contract_version": 3, "source": "exact fixed 208 train endpoints only; minival never opened", "fresh_init": "V4 fresh seed42 matched FULL then V3 query initialized from that FULL; no V4/probe warmstart", "updates": UPDATES, "lr": LR, "effective_batch": EFFECTIVE, "microbatch": MICRO, "dropout": "disabled", "target": "20x native velocity; score prediction divided by 20", "acceptance": "PASS iff R2>=0.5 and prediction_std>=0.5*target_std; 0.1<=R2<0.5 LEARNING_HOLD; else FAIL"}
    OUT.mkdir(parents=True)
    frozen = {"ids": {name: values.tolist() for name, values in fixed.items()}, "source_authority": authority, "recipe": recipe}
    (OUT / "frozen_ids.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    device = torch.device("cuda:0"); full, query = make_matched_pair(); arms = {"full_v4": full.to(device), "t_v5_logage": query.to(device)}
    code = {"model.py": sha(Path(__file__).with_name("model.py")), "capacity_probe.py": sha(Path(__file__)), "current_query_v3/core.py": sha(Path(__file__).parents[1] / "two_mainlines_long_v1/current_query_v3/core.py")}
    optimizer = {name: torch.optim.AdamW(groups(model), lr=LR) for name, model in arms.items()}; initial = {name: state_sha(model) for name, model in arms.items()}; before = {name: score(model, cache, fixed, device) for name, model in arms.items()}; started = time.time(); done = 0; names = list(fixed)
    try:
        for step in range(UPDATES):
            if time.time() - started > DEADLINE: break
            name = names[step % len(names)]; row = cache["train"][name]; x, y, _ = batch(row, fixed[name], device)
            bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
            for model_name, model in arms.items():
                model.train(); optimizer[model_name].zero_grad(set_to_none=True)
                for index in range(0, len(x), MICRO):
                    (F.mse_loss(model.forward_last(x[index:index + MICRO], bank), y[index:index + MICRO]) * (MICRO / len(x))).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer[model_name].step()
            done = step + 1
    finally:
        after = {name: score(model, cache, fixed, device) for name, model in arms.items()}
        decision = {name: "PASS" if value["r2_concat"] >= .5 and value["prediction_std"] >= .5 * value["target_std"] else "LEARNING_HOLD" if value["r2_concat"] >= .1 else "FAIL" for name, value in after.items()}
        for name, model in arms.items(): torch.save({"model": model.state_dict(), "optimizer": optimizer[name].state_dict(), "updates": done, "recipe": recipe}, OUT / f"{name}_latest.pt")
        report = {"schema": recipe["schema"], "status": "COMPLETE" if done == UPDATES else "TIMEOUT_OR_INTERRUPTED", "input_authority": authority, "code_sha256": code, "frozen_ids_sha256": sha(OUT / "frozen_ids.json"), "initial_state_sha256": initial, "trained_state_sha256": {name: state_sha(model) for name, model in arms.items()}, "recipe": recipe, "before": before, "after": after, "updates_completed": done, "elapsed_s": time.time() - started, "decision": decision}
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"after": after, "decision": decision, "updates": done}, sort_keys=True))


if __name__ == "__main__":
    main()
