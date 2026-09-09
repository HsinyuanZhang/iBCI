#!/usr/bin/env python3
"""Independently replay the sealed formal M1 R100/D4 calibration scan.

This does not train or alter the formal run.  It materializes only the old
held-out calibration M10 face, applies each sealed EMA state, retains arrays
needed to audit the metric, and never opens an official-test surface.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; V2=HERE.parent.parent; WORKSPACE=V2.parent; V1=WORKSPACE/"btransform_unified_v1"
for candidate in (V2/"src",V2/"scripts",V1/"src",V1/"scripts",WORKSPACE):
    if str(candidate) not in sys.path: sys.path.insert(0,str(candidate))
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.r2 import variance_weighted_r2
import rift_v1.m1_joint_train as baseline

SCHEMA="m1_muscle_r100_baseline_replay_v1"
BASELINE_DEFAULT=V2/"results/rift_v1/m1_r100_joint_d_s42_formal_v1"
EPOCHS=tuple(range(1,baseline.EPOCHS+1))
def need(ok:bool,msg:str)->None:
    if not ok: raise RuntimeError(msg)
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def array_sha(value:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
def atomic_json(path:Path,value:Mapping[str,Any])->None:
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");tmp.replace(path)
def atomic_npz(path:Path,**arrays:np.ndarray)->None:
    tmp=path.with_suffix(".tmp.npz");np.savez_compressed(tmp,**arrays);tmp.replace(path)
def read_json(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text());need(isinstance(value,dict),f"JSON object required: {path}");return value

def validate_formal(run:Path)->tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
    """Bind all 24 checkpoint bytes to the completed old run receipts."""
    meta,train,score=(read_json(run/name) for name in ("run_meta.json","train_receipt.json","score_receipt.json"))
    expected={"schema":"m1_rift_joint_train_v1","status":"FORMAL","cell":baseline.CELL,"task":"m1","arm":"D_JOINT","seed":baseline.SEED,"sampler_seed":baseline.SEED,"epochs":baseline.EPOCHS,"updates_per_epoch":baseline.UPDATES_PER_EPOCH,"official_test_used":False}
    need(all(meta.get(k)==v for k,v in expected.items()),"formal run metadata identity mismatch")
    need(meta.get("source_hashes")==baseline._source_hashes(),"formal source-code binding drift")
    expected_train={"schema":"m1_rift_joint_train_receipt_v1","status":"COMPLETED","cell":baseline.CELL,"arm":"D_JOINT","seed":baseline.SEED,"sampler_seed":baseline.SEED,"epochs":baseline.EPOCHS,"steps":baseline.EPOCHS*baseline.UPDATES_PER_EPOCH,"source_hashes":meta["source_hashes"],"source_contract":meta["source_contract"],"b3s":meta["b3s"]}
    need(all(train.get(k)==v for k,v in expected_train.items()),"formal train receipt mismatch")
    expected_score={"schema":"m1_rift_joint_ho_calib_epoch_scan_v1","status":"COMPLETED","cell":baseline.CELL,"arm":"D_JOINT","seed":baseline.SEED,"sampler_seed":baseline.SEED,"source_hashes":meta["source_hashes"],"source_contract":meta["source_contract"],"b3s":meta["b3s"],"official_test_used":False}
    need(all(score.get(k)==v for k,v in expected_score.items()),"formal score receipt mismatch")
    keys={str(e) for e in EPOCHS};need(set(score.get("ema_by_epoch",{}))==keys and set(score.get("checkpoint_sha256_by_epoch",{}))==keys,"formal score/checkpoint scan incomplete")
    need(score.get("selection",{}).get("rule")=="earliest maximum equal-session mean EMA","formal selection rule drift")
    for epoch in EPOCHS:
        path=run/f"epoch_{epoch:03d}.pt";need(path.is_file() and sha(path)==score["checkpoint_sha256_by_epoch"][str(epoch)],f"formal checkpoint SHA drift at epoch {epoch}");baseline._validate_checkpoint(path,meta,expected_epoch=epoch)
    return meta,train,score

def initial_progress(run:Path,meta:Mapping[str,Any],score:Mapping[str,Any])->dict[str,Any]:
    return {"schema":SCHEMA,"status":"REPLAYING","baseline_run":str(run),"baseline_run_meta_sha256":sha(run/"run_meta.json"),"baseline_train_receipt_sha256":sha(run/"train_receipt.json"),"baseline_score_receipt_sha256":sha(run/"score_receipt.json"),"source_code_sha256":{str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(Path(baseline.__file__).resolve()):sha(Path(baseline.__file__).resolve())},"formal_cell":baseline.CELL,"arm":meta["arm"],"seed":meta["seed"],"ho_contract":score["ho_contract"],"completed":{}}
def check_progress(value:Mapping[str,Any],expect:Mapping[str,Any])->dict[str,Any]:
    # ``status``, ``completed``, and ``last_completed_epoch`` advance during a
    # resumable replay.  Every provenance binding remains immutable.
    static_keys=set(expect)-{"status","completed"}
    need(all(value.get(k)==expect[k] for k in static_keys),"replay progress provenance drift");done=value.get("completed");need(isinstance(done,dict) and set(done)<={str(e) for e in EPOCHS},"malformed replay progress");return dict(done)

def score_epoch(model:torch.nn.Module,material:Mapping[str,Any],device:torch.device,epoch_dir:Path)->dict[str,Any]:
    per={}; was=model.training;model.eval()
    try:
      for session in baseline.HO:
        item=material[session];starts=np.asarray(item["starts"],dtype=np.int64);ps=[];ys=[]
        for bid,batch in enumerate(DataLoader(item["dataset"],batch_size=baseline.BATCH,shuffle=False,num_workers=0)):
            x,y=batch[0],batch[1];offset=bid*baseline.BATCH;valid=baseline.valid_mask_from_padded_starts(item["starts"][offset:offset+len(x)],device=device)
            with torch.inference_mode(): out=model(x.float().to(device),item["bank"],input_valid_mask=valid)
            ps.append(np.ascontiguousarray(out.float().cpu().numpy(),dtype=np.float32));ys.append(np.ascontiguousarray(y[:,-1,:].numpy(),dtype=np.float32))
        pred=np.ascontiguousarray(np.concatenate(ps),dtype=np.float32);target=np.ascontiguousarray(np.concatenate(ys),dtype=np.float32)
        need(pred.shape==target.shape==(len(starts),16) and np.isfinite(pred).all() and np.isfinite(target).all(),f"{session}: replay geometry/finite drift")
        artifact=epoch_dir/f"{session}_predictions.npz";need(not artifact.exists(),f"refuse overwrite {artifact}");atomic_npz(artifact,pred=pred,y=target,starts=starts)
        per[session]={"window_count":int(len(target)),"legacy_variance_weighted_r2":float(variance_weighted_r2(target,pred)),"sklearn_channel_centered_variance_weighted_r2":float(r2_score(target,pred,multioutput="variance_weighted")),"prediction_sha256":array_sha(pred),"target_sha256":array_sha(target),"starts_sha256":array_sha(starts),"artifact":artifact.name,"artifact_sha256":sha(artifact)}
    finally: model.train(was)
    return {"per_session":per,"equal_session_mean_legacy":float(np.mean([per[s]["legacy_variance_weighted_r2"] for s in baseline.HO])),"equal_session_mean_sklearn_channel_centered":float(np.mean([per[s]["sklearn_channel_centered_variance_weighted_r2"] for s in baseline.HO])),"n_windows":int(sum(per[s]["window_count"] for s in baseline.HO))}

def replay(args:argparse.Namespace)->dict[str,Any]:
    run=args.baseline_run.resolve();need(run.is_dir(),f"formal baseline missing: {run}");meta,_train,oldscore=validate_formal(run)
    dest=args.dest.resolve();receipt_path=dest/"replay_receipt.json";need(not receipt_path.exists(),"completed replay receipt already exists; refuse overwrite")
    if not dest.exists():dest.mkdir(parents=True)
    expected=initial_progress(run,meta,oldscore);progress_path=dest/"replay_progress.json";completed=check_progress(read_json(progress_path) if progress_path.is_file() else expected,expected)
    torch.set_num_threads(args.cpu_threads);device=torch.device(args.device);material=baseline._ho_material();need(baseline._ho_contract(material)==oldscore["ho_contract"],"held-out calibration material contract drift")
    model=baseline._decoder(device,"D_JOINT",baseline.SEED);model.install_session_memory({s:material[s]["bank"] for s in baseline.HO},{s:material[s]["calib10"] for s in baseline.HO});model.to(device);ema=DecoderEMA(model,decay=baseline.plan.EMA_DECAY)
    for epoch in EPOCHS:
      path=run/f"epoch_{epoch:03d}.pt";old=oldscore["ema_by_epoch"][str(epoch)]
      if str(epoch) in completed:
        row=completed[str(epoch)];epoch_dir=dest/row.get("artifact_dir","");metrics=row.get("metrics",{});per=metrics.get("per_session",{})
        need(row.get("checkpoint_sha256")==sha(path) and epoch_dir.is_dir() and set(per)==set(baseline.HO),f"replay resume binding drift epoch {epoch}")
        for session in baseline.HO:
            artifact=epoch_dir/per[session].get("artifact","")
            need(artifact.is_file() and per[session].get("artifact_sha256")==sha(artifact),f"replay artifact SHA drift epoch {epoch} session {session}")
        continue
      state=baseline._validate_checkpoint(path,meta,expected_epoch=epoch);model.load_state_dict(state["raw_state_dict"],strict=True);ema.load_state_dict(state["ema"]);epoch_dir=dest/f"epoch_{epoch:03d}";need(not epoch_dir.exists(),f"partial epoch without progress: {epoch_dir}");epoch_dir.mkdir(parents=True)
      observed=baseline._with_ema_eval(model,ema,lambda:score_epoch(model,material,device,epoch_dir));old_per=old.get("per_session",{});need(set(old_per)==set(baseline.HO),f"old score roster drift epoch {epoch}")
      diffs={s:abs(observed["per_session"][s]["legacy_variance_weighted_r2"]-float(old_per[s]["r2"])) for s in baseline.HO};delta=max(*diffs.values(),abs(observed["equal_session_mean_legacy"]-float(old["equal_session_mean"])))
      need(delta<=1e-10,f"epoch {epoch}: replay legacy R2 differs by {delta:.3g}")
      exact={s:observed["per_session"][s]["prediction_sha256"]==old_per[s].get("prediction_sha256") for s in baseline.HO}
      completed[str(epoch)]={"checkpoint_sha256":sha(path),"artifact_dir":epoch_dir.name,"legacy_metric_max_abs_difference":delta,"exact_prediction_sha256":exact,"all_prediction_sha256_exact":all(exact.values()),"metrics":observed}
      atomic_json(progress_path,{**expected,"status":"REPLAYING","completed":completed,"last_completed_epoch":epoch})
    need(set(completed)=={str(e) for e in EPOCHS},"replay incomplete")
    selected=min(EPOCHS,key=lambda e:(-completed[str(e)]["metrics"]["equal_session_mean_sklearn_channel_centered"],e))
    receipt={**expected,"status":"COMPLETED","completed":completed,"original_legacy_selection":oldscore["selection"],"surface":"visible HO3 calibration, query trial0 including M10 support","selection":{"epoch":selected,"metric":"channel_variance_weighted_r2","rule":"earliest maximum equal-session mean sklearn channel-centered variance-weighted R2"},"max_legacy_metric_abs_difference":max(float(x["legacy_metric_max_abs_difference"]) for x in completed.values()),"all_prediction_sha256_exact":all(x["all_prediction_sha256_exact"] for x in completed.values()),"official_test_used":False}
    atomic_json(receipt_path,receipt);return {"status":"COMPLETED","receipt":str(receipt_path),"selected_epoch":selected}
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--baseline-run",type=Path,default=BASELINE_DEFAULT);p.add_argument("--dest",type=Path,required=True);p.add_argument("--device",default="cuda:1");p.add_argument("--cpu-threads",type=int,default=4);a=p.parse_args();
    if a.cpu_threads<1:p.error("--cpu-threads must be positive")
    print(json.dumps(replay(a),sort_keys=True))
if __name__=="__main__":main()
