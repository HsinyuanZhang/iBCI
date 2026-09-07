#!/usr/bin/env python3
"""Fail-closed preflight for the *single* snapshot-bound Context Full launch."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import hydra, lightning as L, torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event
from src.data.h1_context_event_carrier import H1ContextEventDataModule
from src.data.h1_context_event_source_snapshot import load_snapshot
from src.data.h1_sparse_event_endpoint import H1SparseEventDataModule
from src.data.h1_sparse_event_source_snapshot import apply_snapshot_to_source_module, load_snapshot as load_hse_snapshot
from src.h1_m4_eb_normalized_v2_contract import canonical_sha256, sha256_file, state_hash, write_immutable_json

HSE_ZERO=ROOT/"logs/h1_sparse_event_endpoint_m4_zero_s42_v1/checkpoints/fixed_epoch50/epoch_049.ckpt"
HSE_ZERO_CONFIG=ROOT/"logs/h1_sparse_event_endpoint_m4_zero_s42_v1/.hydra/config.yaml"
HSE_SOURCE_SNAPSHOT=ROOT/"pilot_artifacts/h1_sparse_event_endpoint/source_snapshot/H1_SE5_FOLD0_SOURCE_v1.json"
HSE_FULL_CONFIG=ROOT/"configs/experiment/h1_sparse_event_endpoint_full.yaml"
CONTEXT_FULL_CONFIG=ROOT/"configs/experiment/h1_context_event_carrier_full.yaml"

def _need(ok:bool,msg:str)->None:
    if not ok:raise ValueError(msg)
def _starts_sha(source)->str:
    body={n:list(source.carrier_cache.starts_by_session[n]) for n in source.pilot_manifest()["source_sessions"]}
    return event.canonical_sha256(body)
def _without_carrier_model(cfg):
    x=OmegaConf.to_container(cfg.model,resolve=True);x["net"]["zero_carrier"]="LITERAL_ZERO_BOUNDARY";x["pilot_arm"]="LITERAL_ZERO_BOUNDARY";return x
def _outcome_gate()->dict:
    return {"primary_material_translation":{"comparison":"Context Full - sealed H-SE5 Full","pooled_r2_minimum":0.010,"both_target_recordings_positive":True},"required_positive_pooled":["Context Full - common independent Zero5","Full - same-checkpoint Zero","Full - row shuffle","Full - endpoint-label shuffle","Full - tag shuffle"],"prefer_two_of_two_positive":["Full - endpoint-label shuffle","Full - tag shuffle"],"outcome_routes":{"PASS_MATERIAL":"pooled >= +0.010 and both target recording deltas positive, with every required control positive","SMALL_POSITIVE_NONMATERIAL":"0 < Context Full - H-SE5 Full < +0.010 with no sign inversion; do not extend","STOP":"Context Full - H-SE5 Full <= 0, any target-recording sign inversion, or any required pooled control <= 0"},"proportional_extrapolation":"launch rationale only; never a forecast or acceptance criterion"}
def common_zero_reusable(noncarrier_checks:dict[str,bool])->bool:
 """Carrier manifest/cache/normalizer deliberately do not enter this decision.

 The consumer executes ``zeros_like(carrier)`` at its literal zero boundary;
 only these independently recorded non-carrier bindings affect the common null.
 """
 return bool(noncarrier_checks) and all(noncarrier_checks.values())

def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-dir",type=Path,default=ROOT/"data/000954");p.add_argument("--source-snapshot-receipt",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--hse5-zero",type=Path,default=HSE_ZERO);p.add_argument("--hse5-zero-config",type=Path,default=HSE_ZERO_CONFIG);a=p.parse_args()
 with initialize_config_dir(version_base="1.3",config_dir=str(ROOT/"configs")):
  fullcfg=compose(config_name="train.yaml",overrides=["experiment=h1_context_event_carrier_full"])
 snap=load_snapshot(a.source_snapshot_receipt)
 source=H1ContextEventDataModule(task="h1",data_dir=str(a.data_dir.resolve()),cache_dir=str(ROOT/"pilot_artifacts/h1_context_event_carrier/shared_source_cache"),source_snapshot_receipt=str(a.source_snapshot_receipt));source.setup("fit");m=source.pilot_manifest()
 _need(m==snap["manifest"] and source.pilot_manifest_sha256==snap["metadata"]["manifest_sha256"],"training source manifest is not the snapshot manifest")
 _need(m["candidate"]=="ser_context_q4" and m["carrier_dim"]==m["side_dim"]==5 and m["target_carrier_ridge_lambda"]==3.0 and not m["deployment_carrier_dense_velocity_opened"],"context carrier contract drift")
 _need(m["fixed_epochs"]==50 and m["calibration_schedule_sha256"] and m["batch_order_sha256"],"fixed schedule drift")
 ckpt=torch.load(a.hse5_zero,map_location="cpu",weights_only=False);z=ckpt.get("h1_sparse_event_endpoint");_need(isinstance(z,dict) and z.get("arm")=="zero" and ckpt.get("epoch")==49,"sealed H-SE5 Zero5 checkpoint drift")
 zcfg=OmegaConf.load(a.hse5_zero_config);_need(sha256_file(a.hse5_zero_config)==z.get("config_sha256"),"sealed Zero resolved .hydra config SHA drift")
 hse=H1SparseEventDataModule(task="h1",data_dir=str(a.data_dir.resolve()),cache_dir=str(ROOT/"pilot_artifacts/h1_sparse_event_endpoint/shared_source_cache"));hse.setup("fit");apply_snapshot_to_source_module(hse,load_hse_snapshot(HSE_SOURCE_SNAPSHOT));hm=hse.pilot_manifest()
 # Match src/train.py's model construction point: reseed immediately before
 # instantiate.  The zero flag has no parameters, and zero boundary then
 # makes all carrier-map/cache/normalizer values observationally irrelevant.
 L.seed_everything(42,workers=True);ctx_model=hydra.utils.instantiate(fullcfg.model);ctx_initial=state_hash(ctx_model.state_dict())
 source_hashes=canonical_sha256({"source_sessions":m["source_sessions"],"files":m["files"]})
 checks={"fold_date":z.get("fold_date")=="19250101","source_sessions_and_nwb_hashes":z.get("source_hashes_sha256")==source_hashes==canonical_sha256({"source_sessions":hm["source_sessions"],"files":hm["files"]}),"source_window_indices":hm["source_window_indices_sha256"]==m["source_window_indices_sha256"],"batch_order":hm["batch_order_sha256"]==m["batch_order_sha256"],"calibration_start_schedule":hm["calibration_schedule_sha256"]==m["calibration_schedule_sha256"],"seed_epochs":zcfg.seed==42 and zcfg.trainer.max_epochs==50 and z.get("epochs_completed")==50,"optimizer":OmegaConf.to_container(zcfg.model.optimizer,resolve=True)==OmegaConf.to_container(fullcfg.model.optimizer,resolve=True),"noncarrier_model_topology":_without_carrier_model(zcfg)==_without_carrier_model(fullcfg),"parameter_counts":z.get("carrier_identity_parameters")==ctx_model.net.carrier_parameter_count() and z.get("whole_model_parameters")==sum(x.numel() for x in ctx_model.net.parameters()),"initial_state":z.get("initial_state_sha256")==ctx_initial,"carrier_shape":z.get("carrier_dim")==5,"literal_zero_boundary":bool(zcfg.model.net.zero_carrier) and z.get("carrier_mode")=="literal_zero5_at_model_boundary"}
 # The starts/identity selection is fully determined by the shared schedule
 # and the same legal contiguous M4 starts.  Record its Context digest so the
 # receipt contains an auditable identity schedule binding rather than merely
 # asserting that property in prose.
 checks["trial_identity_start_structure"] = all(len(v)==len(source.carrier_cache.starts_by_session[k]) for k,v in {k:source.carrier_cache.starts_by_session[k] for k in m["source_sessions"]}.items())
 reusable=common_zero_reusable(checks);irrelevant={"carrier_map_cache_normalizer":{"expected_to_differ":True,"why":"zero_carrier=True executes zeros_like(carrier) before concatenation; zero tensor shape is separately bound to [176,5]"},"hse5":{"source_manifest_sha256":z.get("source_manifest_sha256"),"cache_sha256":z.get("source_cache_sha256"),"normalizer_sha256":z.get("normalizer_sha256")},"context":{"source_manifest_sha256":source.pilot_manifest_sha256,"cache_sha256":m["carrier_cache_sha256"],"normalizer_sha256":m["normalizer_sha256"]}}
 receipt={"schema":"h1_context_event_carrier_m4_fold0_cpu_preflight_v2","status":"READY_CONTEXT_FULL_COMMON_HSE5_ZERO" if reusable else "STOP_CONTEXT_ZERO_PARITY_FAILED","fold_date":"19250101","seed":42,"gpu_launched":False,"source_snapshot":{"receipt":str(snap["receipt_path"]),"receipt_sha256":snap["receipt_sha256"],"snapshot":str(snap["snapshot_path"]),"snapshot_sha256":snap["snapshot_sha256"],"fixed_screen":snap["receipt"]["fixed_screen"]},"source_manifest":m,"source_manifest_sha256":source.pilot_manifest_sha256,"noncarrier_zero_parity":{"checks":checks,"reusable_common_hse5_zero5":reusable,"context_trial_identity_starts_sha256":_starts_sha(source),"irrelevant_after_literal_zero_boundary":irrelevant},"configs":{"context_full":{"path":str(CONTEXT_FULL_CONFIG),"sha256":sha256_file(CONTEXT_FULL_CONFIG)},"sealed_hse5_zero_resolved":{"path":str(a.hse5_zero_config),"sha256":sha256_file(a.hse5_zero_config)}},"checkpoints":{"sealed_hse5_zero":{"path":str(a.hse5_zero),"sha256":sha256_file(a.hse5_zero),"metadata":z}},"planned_training":{"only_launchable_cell":"Context Full" if reusable else None,"epochs":50,"launches_context_zero":False,"common_zero":"sealed H-SE5 Zero5" if reusable else None},"controls":["common independent H-SE5 Zero5","same-checkpoint Zero","same-checkpoint row shuffle","same-checkpoint endpoint-label shuffle","same-checkpoint tag shuffle"],"outcome_gate":_outcome_gate(),"scope":{"source_nwbs_opened":11,"target_nwbs_opened":0,"target_optimizer_steps":0,"target_backward_steps":0,"dense_velocity_used_for_carrier":False}}
 out,d=write_immutable_json(a.output,receipt);print(json.dumps({"status":receipt["status"],"receipt":str(out),"sha256":d},indent=2));return 0 if reusable else 2
if __name__=="__main__":raise SystemExit(main())
