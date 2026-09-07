#!/usr/bin/env python3
"""Isolated target evaluator for the context Full cell and its controls.

It can score Full alone (a preliminary receipt), but only a separately trained
context Zero5 permits the content comparison and terminal content conclusion.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import hydra, numpy as np, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.data.h1_context_event_carrier import H1ContextEventDataModule, build_context_target_dataset
from src.data.h1_context_event_source_snapshot import load_snapshot
from src.h1_m4_eb_normalized_v2_contract import assert_immutable_receipt, assert_state_immutable, sha256_file, state_hash, write_immutable_json
from src.models.h1_sparse_event_module import CHECKPOINT_SCHEMA

HSE5_RECEIPT = ROOT / "pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1.json"
HSE5_ZERO = ROOT / "logs/h1_sparse_event_endpoint_m4_zero_s42_v1/checkpoints/fixed_epoch50/epoch_049.ckpt"
HSE5_ZERO_CONFIG = ROOT / "logs/h1_sparse_event_endpoint_m4_zero_s42_v1/.hydra/config.yaml"
EXPECTED_QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"

def _need(ok: bool, msg: str) -> None:
    if not ok: raise ValueError(msg)
def _r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64); tss = float(np.square(y-y.mean(axis=0, keepdims=True)).sum()); _need(tss > 0, "undefined R2")
    return 1.0-float(np.square(y-p).sum())/tss
def _load(path: Path, config_path: Path, arm: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False); meta = ckpt.get("h1_sparse_event_endpoint")
    _need(isinstance(meta, dict) and meta.get("schema") == CHECKPOINT_SCHEMA and meta.get("arm") == arm and ckpt.get("epoch") == 49, f"invalid terminal context {arm} checkpoint")
    _need(meta.get("fold_date") == "19250101" and meta.get("epochs_completed") == 50 and meta.get("carrier_dim") == 5 and meta.get("target_session_backward_steps") == 0, "checkpoint topology/target legality drift")
    _need(meta.get("config_sha256") == sha256_file(config_path), "checkpoint config SHA mismatch")
    cfg = OmegaConf.load(config_path); _need(cfg.pilot.arm == arm and cfg.model.net.carrier_dim == 5, "checkpoint config arm/topology drift")
    return ckpt, meta, cfg
def _model(cfg, ckpt, device):
    value = hydra.utils.instantiate(cfg.model); value.load_state_dict(ckpt["state_dict"], strict=True); return value.to(device).eval()
def _score(model, dataset, device, label):
    before = state_hash(model.state_dict()); ps=[]; ys=[]; names=[]
    with torch.no_grad():
        for neural, target, identity, name, carrier in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
            out = model(neural.to(device, dtype=torch.float32), calib_trialized_neural_features=identity.to(device, dtype=torch.float32), carrier=carrier.to(device, dtype=torch.float32))
            if model.hparams.decode_last_timestep_only: out, target = out[:, -1:, :], target[:, -1:, :]
            if model.hparams.predict_scaled_behavior: out = out / model.hparams.behavior_scaling_factor
            ps.append(out[:, -1].cpu().numpy()); ys.append(target[:, -1].numpy()); names.extend(name)
    after = state_hash(model.state_dict()); assert_state_immutable(before, after, label); p,y=np.concatenate(ps),np.concatenate(ys)
    return {"pooled_r2": _r2(y,p), "samples": len(dataset), "state_immutable": before == after, "state_sha256_before": before, "state_sha256_after": after,
            "per_session": {n: {"samples": int(np.sum(np.asarray(names)==n)), "r2": _r2(y[np.asarray(names)==n],p[np.asarray(names)==n])} for n in sorted(set(names))}, "query_window_indices_sha256": dataset.window_indices_sha256}

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-dir",type=Path,default=ROOT/"data/000954"); p.add_argument("--full-checkpoint",type=Path,required=True); p.add_argument("--full-config",type=Path,required=True)
    p.add_argument("--source-snapshot-receipt",type=Path,required=True); p.add_argument("--common-zero-preflight",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--zero-checkpoint",type=Path,default=HSE5_ZERO); p.add_argument("--zero-config",type=Path,default=HSE5_ZERO_CONFIG); p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args()
    full_ckpt, full_meta, full_cfg = _load(a.full_checkpoint,a.full_config,"full")
    snapshot=load_snapshot(a.source_snapshot_receipt)
    source=H1ContextEventDataModule(task="h1",data_dir=str(a.data_dir.resolve()),cache_dir=str(ROOT/"pilot_artifacts/h1_context_event_carrier/shared_source_cache"),source_snapshot_receipt=str(a.source_snapshot_receipt)); source.setup("fit")
    _need(full_meta["source_manifest_sha256"] == source.pilot_manifest_sha256 and full_meta["normalizer_sha256"] == source.normalizer.normalizer_sha256, "Full/source snapshot mismatch")
    # No target NWB is opened until checkpoint/source bindings above have passed.
    target=build_context_target_dataset(data_dir=a.data_dir,source_module=source); _need(target.window_indices_sha256 == EXPECTED_QUERY_SHA,"query windows drift from sealed H-SE5 pool")
    device=torch.device(a.device); model=_model(full_cfg,full_ckpt,device); scores={name:_score(model,target.with_intervention(name),device,f"Context/{name}") for name in target.INTERVENTIONS}
    sealed=json.loads(HSE5_RECEIPT.read_text()); hse=float(sealed["metrics"]["hse5_same_checkpoint_interventions"]["full"]["pooled_r2"])
    preflight=assert_immutable_receipt(a.common_zero_preflight,"READY_CONTEXT_FULL_COMMON_HSE5_ZERO")
    _need(preflight["source_snapshot"]["receipt_sha256"]==sha256_file(a.source_snapshot_receipt),"common-zero parity preflight/snapshot mismatch")
    _need(preflight["noncarrier_zero_parity"]["reusable_common_hse5_zero5"] is True,"common Zero lacks parity authorization")
    zero_ckpt,zero_meta,zero_cfg=_load(a.zero_checkpoint,a.zero_config,"zero")
    _need(preflight["checkpoints"]["sealed_hse5_zero"]["sha256"]==sha256_file(a.zero_checkpoint),"common Zero/preflight checkpoint mismatch")
    common_zero=_score(_model(zero_cfg,zero_ckpt,device),target.with_intervention("full"),device,"Context/common-HSE5-Zero5")
    margins={"context_full_minus_sealed_hse5_full":scores["full"]["pooled_r2"]-hse,"full_minus_same_checkpoint_zero":scores["full"]["pooled_r2"]-scores["zero"]["pooled_r2"],"full_minus_row_shuffle":scores["full"]["pooled_r2"]-scores["row"]["pooled_r2"],"full_minus_endpoint_label_shuffle":scores["full"]["pooled_r2"]-scores["label"]["pooled_r2"],"full_minus_tag_shuffle":scores["full"]["pooled_r2"]-scores["tag"]["pooled_r2"]}
    margins["context_full_minus_common_independent_zero5"]=scores["full"]["pooled_r2"]-common_zero["pooled_r2"]
    hse_session=sealed["metrics"]["hse5_same_checkpoint_interventions"]["full"]["per_session"]
    hse_session_deltas={n:scores["full"]["per_session"][n]["r2"]-hse_session[n]["r2"] for n in hse_session}
    clauses={"common_independent_zero":margins["context_full_minus_common_independent_zero5"]>0,"same_checkpoint_zero":margins["full_minus_same_checkpoint_zero"]>0,"row_shuffle":margins["full_minus_row_shuffle"]>0,"endpoint_label_shuffle":margins["full_minus_endpoint_label_shuffle"]>0,"tag_shuffle":margins["full_minus_tag_shuffle"]>0,"label_two_of_two":all(scores["full"]["per_session"][n]["r2"]>scores["label"]["per_session"][n]["r2"] for n in hse_session),"tag_two_of_two":all(scores["full"]["per_session"][n]["r2"]>scores["tag"]["per_session"][n]["r2"] for n in hse_session)}
    primary=margins["context_full_minus_sealed_hse5_full"]; signs=all(x>0 for x in hse_session_deltas.values()); controls=all(clauses.values())
    status="PASS_CONTEXT_MATERIAL" if controls and primary>=.010 and signs else ("SMALL_POSITIVE_NONMATERIAL" if controls and 0<primary<.010 and signs else "STOP_CONTEXT")
    evaluator_path = Path(__file__).resolve()
    receipt={"schema":"h1_context_event_carrier_m4_fold0_terminal_v2","status":status,"fold_date":"19250101","seed":42,"evaluation_device":str(device),"evaluator":{"path":str(evaluator_path),"sha256":sha256_file(evaluator_path)},"checkpoint_binding_completed_before_target_open":True,"source_manifest":source.pilot_manifest(),"source_manifest_sha256":source.pilot_manifest_sha256,"source_snapshot":{"receipt":str(snapshot["receipt_path"]),"sha256":sha256_file(snapshot["receipt_path"]),"snapshot":str(snapshot["snapshot_path"]),"snapshot_sha256":sha256_file(snapshot["snapshot_path"])},"common_zero_parity_preflight":{"path":str(a.common_zero_preflight),"sha256":sha256_file(a.common_zero_preflight)},"checkpoints":{"full":{"path":str(a.full_checkpoint),"sha256":sha256_file(a.full_checkpoint),"metadata":full_meta},"common_hse5_zero5":{"path":str(a.zero_checkpoint),"sha256":sha256_file(a.zero_checkpoint),"metadata":zero_meta,"resolved_hydra_config":str(a.zero_config),"resolved_hydra_config_sha256":sha256_file(a.zero_config)}},"target":{"query_window_indices_sha256":target.window_indices_sha256,"support_and_carrier_hashes":target.support_and_carrier_hashes(),"post_four_trial_query":True},"metrics":{"context_same_checkpoint_interventions":scores,"common_independent_hse5_zero5":common_zero},"sealed_hse5":{"receipt":str(HSE5_RECEIPT),"sha256":sha256_file(HSE5_RECEIPT),"full_pooled_r2":hse,"per_recording_delta":hse_session_deltas},"margins":margins,"gate":{"clauses":clauses,"primary_material_translation":{"pooled_minimum":.010,"both_recordings_positive":signs},"pass":status=="PASS_CONTEXT_MATERIAL"},"scope":{"target_optimizer_steps":0,"target_backward_steps":0,"dense_velocity_used_for_carrier":False},"interpretation":{"same_checkpoint_controls_are_diagnostics_not_independent_nulls":True,"common_zero_authorized_only_by_bound_literal_zero_preflight":True,"midpoint_ablation_is_no_new_training_diagnostic":True}}
    out,digest=write_immutable_json(a.output,receipt); print(json.dumps({"status":receipt["status"],"receipt":str(out),"sha256":digest},indent=2)); return 0 if status=="PASS_CONTEXT_MATERIAL" else 2
if __name__=="__main__": raise SystemExit(main())
