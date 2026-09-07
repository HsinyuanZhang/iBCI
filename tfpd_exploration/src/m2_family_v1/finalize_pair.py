"""Post-completion-only finalizer for the frozen M2 FLAT/ROUTE pair.

This module deliberately separates audit/selection from execution.  It cannot
load an epoch checkpoint, write an export, or score a query unless the pair's
immutable summary first proves a complete 24-epoch source-only run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m2_b_small_stability_v1 import training
from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_same_query_comparator_v1 import core
from . import config
from .decoder import M2FamilyDecoder, make_paired_decoders

ROOT = Path(__file__).resolve().parents[3]
PAIR_ROOT = config.RESULT_ROOT
BASELINE = ROOT / "tfpd_exploration/results/m2/family_v1/source_minival_e8_spint_m30_replay_v1.json"
OUT_ROOT = ROOT / "tfpd_exploration/results/m2/family_v1/finalize_pair_v1"
AUTH = "M2_FAMILY_V1_FINALIZE"
MEMBERS = ("FLAT", "ROUTE")
COUNTS = (173, 129, 117, 116, 141, 141, 194)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and bool(np.isfinite(value))


def validate_summary(root: Path = PAIR_ROOT) -> dict[str, Any]:
    """Validate source-only completion and independently freeze earliest EMA picks.

    Does not load checkpoint tensors.  Thus refusal happens before a selected
    model can influence any downstream operation.
    """
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("REFUSED: missing immutable SOURCE_ONLY_COMPLETE summary")
    summary = _read_json(summary_path)
    if summary.get("status") != "SOURCE_ONLY_COMPLETE":
        raise RuntimeError("REFUSED: pair is not SOURCE_ONLY_COMPLETE")
    scores = summary.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(MEMBERS):
        raise RuntimeError("REFUSED: exact FLAT/ROUTE score arms required")
    picks: dict[str, int] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for arm in MEMBERS:
        arm_scores = scores[arm]
        if not isinstance(arm_scores, dict) or set(arm_scores) != {"RAW", "EMA"}:
            raise RuntimeError(f"REFUSED: {arm} RAW/EMA histories required")
        for kind in ("RAW", "EMA"):
            history = arm_scores[kind]
            normalized = {int(k): v for k, v in history.items()} if isinstance(history, dict) else {}
            if tuple(sorted(normalized)) != tuple(range(1, 25)) or not all(_finite_number(v) for v in normalized.values()):
                raise RuntimeError(f"REFUSED: {arm} must contain 24 contiguous finite {kind} epochs")
            arm_scores[kind] = normalized
        pick = min(range(1, 25), key=lambda epoch: (-float(arm_scores["EMA"][epoch]), epoch))
        picks[arm] = pick
        immutable = {}
        for epoch in range(1, 25):
            receipt_path = root / arm / f"epoch_{epoch:03d}_metrics.json"
            sidecar_hash = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
            if not receipt_path.is_file() or not sidecar_hash.is_file():
                raise RuntimeError(f"REFUSED: {arm} epoch{epoch} immutable receipt/hash missing")
            stated_hash = sidecar_hash.read_text(encoding="utf-8").strip().split()[0]
            if stated_hash != sha(receipt_path): raise RuntimeError(f"REFUSED: {arm} epoch{epoch} receipt hash mismatch")
            receipt = _read_json(receipt_path)
            if receipt.get("arm") != arm or int(receipt.get("epoch", -1)) != epoch: raise RuntimeError(f"REFUSED: {arm} immutable receipt identity mismatch")
            for key, value in (("raw_equal_session_r2", arm_scores["RAW"][epoch]), ("ema_equal_session_r2", arm_scores["EMA"][epoch])):
                if not _finite_number(receipt.get(key)) or abs(float(receipt[key])-float(value)) > 1e-12: raise RuntimeError(f"REFUSED: {arm} epoch{epoch} receipt does not reconcile {key}")
            immutable[epoch] = {"metrics_path": str(receipt_path), "metrics_sha256": stated_hash}
        checkpoints = {}
        for epoch in sorted({pick, 24}):
            checkpoint = root / arm / f"epoch_{epoch:03d}.pt"
            if not checkpoint.is_file(): raise RuntimeError(f"REFUSED: {arm} epoch{epoch} checkpoint missing")
            checkpoints[epoch] = {"path": str(checkpoint), "sha256": sha(checkpoint)}
        receipts[arm] = {"selected_epoch": pick, "immutable_epoch_receipts": immutable, "checkpoint_hashes": checkpoints}
    disclosure_path = root / "snapshot_completion_audit_v1.json"
    disclosure = None
    if disclosure_path.is_file():
        detail = _read_json(disclosure_path)
        if detail.get("completed_summary_sha256") != sha(summary_path):
            raise RuntimeError("REFUSED: snapshot sealing addendum summary binding drift")
        disclosure = {"path": str(disclosure_path), "sha256": sha(disclosure_path), "detail": detail}
    return {"summary_path": str(summary_path), "summary_sha256": sha(summary_path),
            "picks": picks, "selected_receipts": receipts, "summary": summary,
            "snapshot_sealing_addendum": disclosure}


def checkpoint_contract(payload: dict[str, Any], model: torch.nn.Module, arm: str, expected_epoch: int, manifest_digest: str) -> dict[str, Any]:
    ema = payload.get("ema")
    raw_state = payload.get("raw_state_dict")
    if not isinstance(ema, dict) or not isinstance(raw_state, dict):
        raise RuntimeError("REFUSED: checkpoint EMA/raw_state_dict payload missing")
    expected_steps = expected_epoch * 3165
    if payload.get("schema") != "m2_b_small_stability_v1_ckpt" or payload.get("cell") != f"CRST_B4_{arm}": raise RuntimeError("REFUSED: checkpoint schema/cell drift")
    if int(payload.get("epoch", -1)) != expected_epoch or int(payload.get("global_step", -1)) != expected_steps:
        raise RuntimeError("REFUSED: selected checkpoint epoch/global step drift")
    if int(payload.get("seed", -1)) != 42 or payload.get("manifest_digest") != manifest_digest: raise RuntimeError("REFUSED: checkpoint seed/manifest drift")
    if int(ema.get("n_updates", -1)) != expected_steps or float(ema.get("decay", -1.0)) != config.EMA_DECAY:
        raise RuntimeError("REFUSED: EMA updates/decay drift")
    if not raw_state or not all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in raw_state.values()):
        raise RuntimeError("REFUSED: non-finite or empty model tensors")
    shadow = ema.get("shadow"); parameters = dict(model.trainable_parameters())
    if not isinstance(shadow, dict) or set(shadow) != set(parameters):
        raise RuntimeError("REFUSED: EMA key set drift")
    if set(raw_state) != set(model.state_dict()): raise RuntimeError("REFUSED: raw state keys not strict model state")
    if any(tuple(shadow[k].shape) != tuple(parameters[k].shape) or not bool(torch.isfinite(shadow[k]).all()) for k in parameters):
        raise RuntimeError("REFUSED: EMA shape/finite drift")
    return {"state": {k: payload[k] for k in ("epoch", "global_step", "batch_id", "seed", "manifest_digest") if k in payload}, "batch_id_limitation": "runner checkpoints leave batch_id=0 at epoch boundary; not interpreted as 3165", "ema_n_updates": int(ema["n_updates"]), "ema_decay": float(ema["decay"]),
            "model_keys": sorted(raw_state), "model_shapes": {k: list(v.shape) for k, v in raw_state.items()}}


def resource_preflight() -> dict[str, Any]:
    """Read-only launch plan; score is never automatically launched."""
    return {"status": "PLANNED_ONLY_NO_SCORE", "authorization_env": AUTH,
            "planned_device": "cuda:0 (physical GPU0) only after M2 trainer completion",
            "fallback": "CPU permitted only after explicit review; no automatic fallback",
            "forbidden": ["ext4", "heldout", "official", "promotion", "automatic_launch"],
            "surface": "source_minival exact 1011 windows / W50 / native divide-by-5"}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); tmp = Path(f.name)
    tmp.replace(path)

def _atomic_torch(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as f: tmp=Path(f.name)
    torch.save(state, tmp); tmp.replace(path)

def _code_and_source_authority() -> dict[str, Any]:
    code={name:sha(Path(__file__).parent/name) for name in ("config.py","decoder.py","routing.py","launch.py","finalize_pair.py")}
    for module in (data,plan,core,training): code[Path(module.__file__).name]=sha(Path(module.__file__))
    from tfpd_exploration.src.m2_dual_track_v1 import contracts, decoders
    from tfpd_exploration.src.m2_b_small_stability_v1 import config as small_config, decoder as small_decoder
    from tfpd_exploration.src.m2_b_small_stability_v1 import ema
    for module in (contracts,decoders,small_config,small_decoder,ema): code[str(Path(module.__file__).relative_to(ROOT))]=sha(Path(module.__file__))
    rows=[]
    for session in plan.HELDIN_SESSIONS:
        d=data._session_dir("source_minival",session)
        names=("X_store.npy","eligible_starts.npy","target_store.npy","e0_u.pt","T.npy","mapping.json","provenance.json")
        rows.append({"session":session,"files":{n:sha(d/n) for n in names},"starts":core.array_sha256(np.asarray(np.load(d/"eligible_starts.npy"),dtype=np.int64)),"target":core.array_sha256(np.asarray(np.load(d/"target_store.npy"),dtype=np.float32))})
    return {"code":code,"source_minival":rows}


def _verify_pretrain_recipe(pretrain: dict[str, Any]) -> dict[str, Any]:
    """Compare current training inputs to hashes recorded before update one."""
    bound = pretrain.get("recipe_hashes", {})
    package = Path(__file__).parent
    expected_names = ("config.py", "decoder.py", "routing.py", "launch.py")
    code = {name: sha(package / name) for name in expected_names}
    manifest = ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json"
    source = {}
    file_names = ("X_store.npy", "target_store.npy", "eligible_starts.npy", "T.npy", "e0_u.pt", "mapping.json", "provenance.json")
    for session in plan.HELDIN_SESSIONS:
        folder = data._session_dir("source_train", session)
        row = {name: sha(folder / name) for name in file_names}
        provenance = _read_json(folder / "provenance.json")
        row["provenance_e0_sha256"] = provenance.get("e0_sha256")
        row["provenance_t4_sha256"] = provenance.get("t4_sha256")
        source[session] = row
    current = {"code_sha256": code, "sampler_sha256": sha(manifest), "source_cache_sha256": source}
    if current != bound:
        raise RuntimeError("REFUSED: current training recipe/source differs from pre-update-one hashes")
    return current

def export_ema_strict(payload: dict[str, Any], model: M2FamilyDecoder, path: Path) -> dict[str, Any]:
    """Strict raw load, then replace trainable parameters only; buffers stay RAW."""
    model.load_state_dict(payload["raw_state_dict"], strict=True)
    full = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for name, parameter in model.trainable_parameters().items(): full[name] = payload["ema"]["shadow"][name].detach().cpu().clone().to(parameter.dtype)
    _atomic_torch(path, full)
    fresh = M2FamilyDecoder(routed=model.routed, seed=config.SEED)
    exported = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = fresh.load_state_dict(exported, strict=True)
    if missing or unexpected or set(exported) != set(full) or any(not torch.equal(full[k], fresh.state_dict()[k]) for k in full): raise RuntimeError("REFUSED: strict exported EMA reload drift")
    return {"export_path": str(path), "export_sha256": sha(path), "state_keys": sorted(full), "strict_reload": True}

def score_source_minival(model: M2FamilyDecoder, device: torch.device) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Raw source-minival only: fixed W50, native output, ordered sealed archive."""
    rows=[]; prediction=[]; target=[]; starts_all=[]; sessions=[]; model.to(device).eval()
    with torch.inference_mode():
        for session, expected in zip(plan.HELDIN_SESSIONS, COUNTS):
            bank=data.load_session_bank("source_minival", session, device=device); root=data._session_dir("source_minival", session)
            x=np.load(root/"X_store.npy", mmap_mode="r"); starts=np.asarray(np.load(root/"eligible_starts.npy"),dtype=np.int64); y=np.asarray(np.load(root/"target_store.npy",mmap_mode="r"),dtype=np.float32)
            if len(starts)!=expected or y.shape!=(expected,2) or starts.min()<0 or starts.max()+50>len(x): raise RuntimeError("REFUSED: source-minival W50 geometry/count drift")
            if bank.E0.shape[-1] != 50 or bank.T.shape[-1] != 4 or bank.E0.shape[-2] != x.shape[-1] or bank.unit_mask.shape[-1] != x.shape[-1] or not bool(bank.unit_mask.any()) or not bool(torch.isfinite(bank.E0).all()) or not bool(torch.isfinite(bank.T).all()): raise RuntimeError("REFUSED: source-minival bank/mask geometry or finite drift")
            out=[]
            for off in range(0, expected, 16):
                windows=np.stack([x[i:i+50] for i in starts[off:off+16]]).astype(np.float32)
                out.append((model.forward_last(torch.from_numpy(windows).to(device),bank,bank.unit_mask)/5).cpu().numpy())
            p=np.concatenate(out).astype(np.float32)
            if not np.isfinite(p).all() or not np.isfinite(y).all(): raise RuntimeError("REFUSED: non-finite native prediction/target")
            rows.append({"session":session,"window_count":expected,"ordered_window_starts_sha256":core.array_sha256(starts),"target_sha256":core.array_sha256(y),"prediction_sha256":core.array_sha256(p),"native_r2":core.variance_weighted_r2(y,p)})
            prediction.append(p); target.append(y); starts_all.append(starts); sessions.extend([session]*expected)
    p=np.concatenate(prediction); y=np.concatenate(target); starts=np.concatenate(starts_all)
    if p.shape!=(1011,2): raise RuntimeError("REFUSED: exact1011 native archive drift")
    return {"rows":rows,"equal_session_r2":float(np.mean([r["native_r2"] for r in rows])),"pooled_r2":core.variance_weighted_r2(y,p)}, {"prediction":p,"target":y,"start":starts,"session":np.asarray(sessions)}


def prepare() -> dict[str, Any]:
    audit = validate_summary()
    return {"schema": "m2_family_v1_finalize_pair_v1", "status": "PREPARED_NOT_EXECUTED",
            "audit": {k: v for k, v in audit.items() if k != "summary"},
            "resource_preflight": resource_preflight(), "baseline_receipt_sha256": sha(BASELINE) if BASELINE.is_file() else None}


def run(device: str | None = None, *, allow_cpu_for_test: bool = False) -> dict[str, Any]:
    """Intentionally gated execution entry point; not invoked by preparation/tests."""
    if os.environ.get(AUTH) != "1":
        raise RuntimeError(f"REFUSED: set {AUTH}=1 only after root review")
    audit = validate_summary()  # Freeze selection before *any* selected load.
    if OUT_ROOT.exists():
        raise RuntimeError(f"REFUSED: will not overwrite {OUT_ROOT}")
    requested=device or "cuda:0"
    if requested == "cpu":
        if not allow_cpu_for_test: raise RuntimeError("REFUSED: CPU requires explicit test/resource authorization")
    elif requested == "cuda:0":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available(): raise RuntimeError("REFUSED: planned physical GPU0 requires CUDA_VISIBLE_DEVICES=0")
    else: raise RuntimeError("REFUSED: only planned GPU0 or explicit test CPU route")
    device=torch.device(requested); pretrain=_read_json(PAIR_ROOT/"pretrain_manifest.json")
    if pretrain.get("schema")!=config.SCHEMA or pretrain.get("updates_per_epoch")!=3165: raise RuntimeError("REFUSED: pretrain manifest drift")
    frozen={"schema":"m2_family_v1_finalize_pair_v1","status":"SELECTION_FROZEN_PRE_SCORE","audit":{k:v for k,v in audit.items() if k!="summary"},"pretrain_sha256":sha(PAIR_ROOT/"pretrain_manifest.json"),"baseline_sha256":"0646d63801230cdc467c9ac9a4a1441b7380d43ea3a9fdf9a37e4cdd63fbcb46","authority_pre":_code_and_source_authority(),"verified_training_recipe":_verify_pretrain_recipe(pretrain),"exports":{},"scores":{}}
    if sha(BASELINE)!=frozen["baseline_sha256"]: raise RuntimeError("REFUSED: fixed baseline receipt SHA drift")
    frozen["immutable_pre"]={"baseline":sha(BASELINE),"pretrain":sha(PAIR_ROOT/"pretrain_manifest.json"),"summary":audit["summary_sha256"],"receipts":audit["selected_receipts"]}
    _atomic_json(OUT_ROOT/"selection_freeze.json",frozen)  # durable before checkpoint loads/prediction
    for arm in MEMBERS:
        epoch=audit["picks"][arm]; model=make_paired_decoders(config.SEED)[0 if arm=="FLAT" else 1]
        endpoint=torch.load(PAIR_ROOT/arm/"epoch_024.pt",map_location="cpu",weights_only=False)
        frozen.setdefault("epoch24_checkpoint_contract",{})[arm]=checkpoint_contract(endpoint,model,arm,24,pretrain["manifest_digest"])
        for label,item_epoch in (("selected",epoch),("endpoint24",24)):
            payload=torch.load(PAIR_ROOT/arm/f"epoch_{item_epoch:03d}.pt",map_location="cpu",weights_only=False)
            key=f"{arm}_{label}_epoch_{item_epoch:03d}"
            frozen.setdefault("checkpoint_contract",{})[key]=checkpoint_contract(payload,model,arm,item_epoch,pretrain["manifest_digest"])
            frozen["exports"][key]=export_ema_strict(payload,model,OUT_ROOT/f"{key}_ema_state.pt")
    for key, export in frozen["exports"].items():
        arm=key.split("_")[0]; epoch=int(key[-3:]); model=make_paired_decoders(config.SEED)[0 if arm=="FLAT" else 1]
        model.load_state_dict(torch.load(export["export_path"],map_location="cpu",weights_only=True),strict=True)
        frozen["scores"][key], archive=score_source_minival(model,device)
        recorded=audit["summary"]["scores"][arm]["EMA"][epoch]
        frozen["scores"][key]["recorded_ema_r2"]=recorded; frozen["scores"][key]["reproduction_delta"]=frozen["scores"][key]["equal_session_r2"]-recorded
        if not np.isfinite(frozen["scores"][key]["reproduction_delta"]) or abs(frozen["scores"][key]["reproduction_delta"])>1e-5: raise RuntimeError("REFUSED: recorded EMA score reproduction drift")
        archive_path=OUT_ROOT/f"{key}_source_minival.npz"
        with tempfile.NamedTemporaryFile(dir=OUT_ROOT, suffix=".npz", delete=False) as h: tmp=Path(h.name)
        np.savez_compressed(tmp,**archive); tmp.replace(archive_path)
        frozen["scores"][key]["npz_sha256"]=sha(archive_path)
    baseline=_read_json(BASELINE)
    for key, score in frozen["scores"].items():
        if len(score["rows"])!=7: raise RuntimeError("REFUSED: seven baseline rows required")
        for got, old in zip(score["rows"],baseline["rows"]):
            if got["session"]!=old["session"] or got["window_count"]!=old["window_count"] or got["ordered_window_starts_sha256"]!=old["ordered_window_starts_sha256"] or got["target_sha256"]!=old["target_sha256"]: raise RuntimeError("REFUSED: baseline rows/starts/targets mismatch")
        score["vs_spint_equal_session_delta"]=score["equal_session_r2"]-baseline["spint_equal_session_r2"]; score["vs_spint_pooled_delta"]=score["pooled_r2"]-baseline["spint_pooled_r2"]
        score["vs_e8_equal_session_delta"]=score["equal_session_r2"]-baseline["e8_equal_session_r2"]; score["vs_e8_pooled_delta"]=score["pooled_r2"]-baseline["e8_pooled_r2"]
    # Re-read actual files; never call an old in-memory audit a post-read.
    audit_after=validate_summary()
    immutable_post={"baseline":sha(BASELINE),"pretrain":sha(PAIR_ROOT/"pretrain_manifest.json"),"summary":audit_after["summary_sha256"],"receipts":audit_after["selected_receipts"]}
    if _code_and_source_authority()!=frozen["authority_pre"] or immutable_post!=frozen["immutable_pre"]: raise RuntimeError("REFUSED: code/source/immutable authority changed during inference")
    if _verify_pretrain_recipe(_read_json(PAIR_ROOT / "pretrain_manifest.json")) != frozen["verified_training_recipe"]:
        raise RuntimeError("REFUSED: original training inputs changed during finalization")
    for item in frozen["exports"].values():
        if sha(Path(item["export_path"])) != item["export_sha256"]: raise RuntimeError("REFUSED: exported state changed during replay")
    result={**frozen,"schema":"m2_family_v1_finalize_pair_v1","status":"SOURCE_MINIVAL_SELECTION_DIAGNOSTIC_NOT_UNTOUCHED_GENERALIZATION",
            "exposure_qualification":{"current_pair_gradient_queries":"held-in-calib only; excludes held-in-minival query arrays",
                                      "current_pair_selection":"all24 EMA scores use these1011 held-in-minival queries",
                                      "historical_spint_exact_label_window_training_exposure":"unknown from the bound checkpoint",
                                      "historical_e8_exact_label_window_training_exposure":"not established by payload/bank equality",
                                      "same_training_recipe_comparison":False},
            "baseline_receipt_sha256":sha(BASELINE),"authority_post":_code_and_source_authority(),"immutable_post":immutable_post}
    _atomic_json(OUT_ROOT/"receipt.json",result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--prepare", action="store_true"); p.add_argument("--run", action="store_true"); a = p.parse_args()
    if a.run: print(json.dumps(run(), sort_keys=True))
    elif a.prepare: print(json.dumps(prepare(), sort_keys=True))
    else: print(json.dumps(resource_preflight(), sort_keys=True))
