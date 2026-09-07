"""Sealed query-only V5 formal train: V4 schedule/data, V3 log-age reader only."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.h1_optimized_v2.score import evaluate
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .model import make_matched_pair

OUT = ROOT / "paired_v5_logage_queryonly_12ep_v1"
V4 = ROOT / "paired_v4_signed_12ep_v1"
SEED, W, EPOCHS, LR, MICRO, EFFECTIVE, EMA = 42, 700, 12, 1e-4, 8, 32, .9995


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rng(*values):
    return np.random.default_rng(int.from_bytes(hashlib.sha256("|".join(map(str, (SEED, *values))).encode()).digest()[:8], "little"))


def batches(cache, epoch):
    ordered = []
    for name, row in sorted(cache["train"].items()):
        starts = rng("sampler", epoch, name).permutation(row["query_starts"])
        ordered.extend((name, starts[index:index + EFFECTIVE]) for index in range(0, len(starts), EFFECTIVE))
    return [ordered[index] for index in rng("order", epoch).permutation(len(ordered))]


def collate(row, starts, device):
    x = np.stack([row["neural"][int(start):int(start) + W] for start in starts]).astype("float32")
    y = np.stack([row["velocity"][int(start) + W - 1] for start in starts]).astype("float32") * 20
    return torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)


def keep(n, epoch, batch_index, device):
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|keep|{epoch}|{batch_index}".encode()).digest()[:8], "little")
    values = torch.rand((n, 176), generator=torch.Generator(device="cpu").manual_seed(seed)) >= .1
    values[values.sum(-1) == 0, 0] = True
    return values.to(device)


def checkpoint(model, optimizer, ema, epoch, step):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.checkpoint_state(), "epoch": epoch, "step": step,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(), "cuda": torch.cuda.get_rng_state_all()},
            "frontend_contract_version": 4, "temporal_contract_version": 3, "recipe": "paired_v5_logage_queryonly_12ep_v1"}


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    for required in (V4 / "input_authority.json", V4 / "selection_freeze.json", V4 / "final.json"):
        if not required.is_file(): raise FileNotFoundError(required)
    cache = build_or_load(); authority = json.loads((ROOT / "source_cache_authority.json").read_text()); validate_authority(cache, authority)
    v4_final = json.loads((V4 / "final.json").read_text()); v4_input = json.loads((V4 / "input_authority.json").read_text())
    recipe = {"seed": SEED, "design_disclosure": "known-source iterative design: a single query-time fixed log16 age-bias variable relative to sealed V4; not a hidden/test result", "arm": "V5 query-only; V4 FULL is sealed external comparator and is never trained or mutated here", "init": "fresh V4 seed42 FULL, fresh V3 log-age query initialized from that FULL; no V4/probe warmstart", "frontend_contract_version": 4, "temporal_contract_version": 3, "activity_scale": 1, "window": W, "units": 176, "out_dim": 7, "target": "20x native velocity, predict divide20", "epochs": EPOCHS, "lr": LR, "weight_decay": .01, "optimizer": "AdamW decay except bias/norm/1d zero", "schedule": "linear warmup epoch1 then constant", "clip_norm": 1, "dropout": "deterministic per-example whole-unit p=.10; same mask generator as V4", "effective_batch": EFFECTIVE, "microbatch": MICRO, "ema": EMA, "selection": "predeclared all 12 frozen minival 2908 endpoints, EMA pooled R2, earliest epoch tie", "complete": "all 20325 minival eval-mask bins only after selection freeze"}
    OUT.mkdir(parents=True)
    code = {name: sha(Path(__file__).with_name(name)) for name in ("model.py", "paired_train.py")}
    control = {"sealed_v4_input_authority_sha256": sha(V4 / "input_authority.json"), "sealed_v4_selection_freeze_sha256": sha(V4 / "selection_freeze.json"), "sealed_v4_final_sha256": sha(V4 / "final.json"), "sealed_v4_full_initial_state_sha256": v4_input.get("initial_state_sha256", v4_final.get("initial_state_sha256", {})).get("full", v4_final["initial_state_sha256"]["full"]), "sealed_v4_full_selection_ema_r2": .6021138429641724, "sealed_v4_full_complete_r2": .5360636711120605}
    (OUT / "input_authority.json").write_text(json.dumps({"source_cache_authority": authority, "code_sha256": code, "recipe": recipe, "v4_control": control}, indent=2, sort_keys=True) + "\n")
    protocol = {"schema": "h1_v5_selection_protocol_freeze_v1", "frozen_before_training": True, "candidate_epochs": list(range(1, EPOCHS + 1)), "selection_surface": "exact frozen minival W700 stride4 2908 endpoints", "governing": "EMA r2_concat", "tie_break": "earliest epoch", "complete_surface": "all 20325 minival eval-mask bins after selection"}
    (OUT / "selection_protocol_freeze.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED); device = torch.device("cuda:0")
    full, model = make_matched_pair(); full = full.to(device); model = model.to(device)
    for name in ("frontend", "final_norm", "readout"):
        left, right = getattr(full, name).state_dict(), getattr(model, name).state_dict()
        if any(not torch.equal(left[key], right[key]) for key in left): raise RuntimeError(f"fresh matched init drift: {name}")
    initial = state_sha(model)
    if state_sha(full) != v4_final["initial_state_sha256"]["full"]: raise RuntimeError("V4 FULL fresh-init equivalence drift")
    if int(model.frontend_contract_version.item()) != 4 or int(model.temporal.temporal_contract_version.item()) != 3: raise RuntimeError("V5 contract marker drift")
    optimizer = torch.optim.AdamW(groups(model), lr=LR); ema = DecoderEMA(model, decay=EMA); step = 0
    report = {"schema": "h1_v5_logage_queryonly_formal_v1", "status": "RUNNING", "recipe": recipe, "v4_control": control, "initial_state_sha256": initial, "epochs": []}; started = time.time(); updates = len(batches(cache, 1))
    for epoch in range(1, EPOCHS + 1):
        for batch_index, (name, starts) in enumerate(batches(cache, epoch)):
            row = cache["train"][name]; x, y = collate(row, starts, device); bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")]); masks = keep(len(starts), epoch, batch_index, device)
            model.train(); optimizer.zero_grad(set_to_none=True); step += 1
            for group in optimizer.param_groups: group["lr"] = LR * min(1., step / updates)
            for index in range(0, len(starts), MICRO): (F.mse_loss(model.forward_last(x[index:index + MICRO], bank, masks[index:index + MICRO]), y[index:index + MICRO]) * (len(x[index:index + MICRO]) / len(x))).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); ema.update_after_step(model)
        payload = checkpoint(model, optimizer, ema, epoch, step); torch.save(payload, OUT / f"t_v5_epoch_{epoch:03d}.pt"); torch.save(payload, OUT / "t_v5_latest.pt")
        raw = evaluate(model, device=device, mode="selection"); scored = ema.score_with_ema(model, lambda view: evaluate(view, device=device, mode="selection"))
        report["epochs"].append({"epoch": epoch, "elapsed_s": time.time() - started, "raw_selection_diagnostic": raw, "ema_selection_governing": scored}); (OUT / "progress.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    winner = max(report["epochs"], key=lambda value: value["ema_selection_governing"]["r2_concat"])
    freeze = {"schema": "h1_v5_selection_freeze_v1", "protocol_freeze_sha256": sha(OUT / "selection_protocol_freeze.json"), "selected": {"epoch": winner["epoch"], "ema_r2_concat": winner["ema_selection_governing"]["r2_concat"], "checkpoint": str(OUT / f"t_v5_epoch_{winner['epoch']:03d}.pt"), "tie_break": "earliest"}}
    (OUT / "selection_freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n"); report["selection_freeze"] = freeze
    report["complete_selected"], report["complete_epoch12"] = {}, {}
    for label, path in (("complete_selected", Path(freeze["selected"]["checkpoint"])), ("complete_epoch12", OUT / "t_v5_epoch_012.pt")):
        payload = torch.load(path, map_location=device, weights_only=False); model.load_state_dict(payload["model"], strict=True)
        if int(model.temporal.temporal_contract_version.item()) != 3: raise RuntimeError("checkpoint temporal contract drift")
        candidate_ema = DecoderEMA(model, decay=EMA); candidate_ema.load_checkpoint_state(payload["ema"]); report[label] = candidate_ema.score_with_ema(model, lambda view: evaluate(view, device=device, mode="complete"))
    report["status"] = "COMPLETE"; report["trained_state_sha256"] = state_sha(model); (OUT / "final.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
