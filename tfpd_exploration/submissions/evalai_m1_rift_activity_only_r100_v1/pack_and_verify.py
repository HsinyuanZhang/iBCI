#!/usr/bin/env python3
"""Build/verify/package the independently sealed M1 ACTIVITY_ONLY e3 payload.

No stage submits, pushes, registers, or opens an official-test surface.  Build
uses only the already-audited B formal run and public seven-tag M10 activity.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, shutil, subprocess, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

ROOT=Path("/home/xinyuan/Work_host/SPINT"); DEST=Path(__file__).resolve().parent
RUN=ROOT/"btransform_unified_v2/results/rift_v1/m1_r100_joint_b_s42_formal_v1"
CKPT=RUN/"epoch_003.pt"; REPLAY=ROOT/"btransform_unified_v2/results/carrier_v4/m1_activity_baseline_replay_v1/replay_receipt.json"
BANK_PAYLOAD=ROOT/"tfpd_exploration/submissions/evalai_m1_projadd_depth2_ort_v1/artifacts/m1_projadd_p16_d2_s42_ema_e21.pkl"
V1_SRC=ROOT/"btransform_unified_v1/src"; V2_SRC=ROOT/"btransform_unified_v2/src"; V1_SCRIPTS=ROOT/"btransform_unified_v1/scripts"; V2_RIFT=ROOT/"btransform_unified_v2/scripts/rift_v1"
SOURCE=("ses-20120924","ses-20120926","ses-20120927","ses-20120928"); HO=("20121004","20121017","20121024"); TAGS={s.removeprefix("ses-") for s in SOURCE}|set(HO)
EPOCHS=tuple(range(1,25)); UPDATES=159960; PAYLOAD=DEST/"artifacts/m1_rift_activity_only_e3.pkl"; IMAGE_TAG="m1-rift-activity-only-r100-e3:v1"; GATE=1e-5; QUERY_PAD_BINS=99; HOST_STEPS=128
SKIP={"joint_m2_model.py","m2_mechanism_model.py","joint_m1_model.py"}
SAFE_INIT='''"""Container-safe RIFT decoder package."""\nfrom .config import RiftTemporalConfig\nfrom .model import RiftDecoder\nfrom .streaming import RiftStreamDecoder\nfrom .temporal import RiftTemporal, RiftTemporalState\nfrom .cpu_temporal import CpuRiftTemporalRuntime\n'''
for p in (V2_SRC,V1_SRC,V1_SCRIPTS,V2_RIFT,DEST,ROOT):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
def need(ok:bool,msg:str)->None:
 if not ok: raise RuntimeError(msg)
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
 return h.hexdigest()
def asha(value:Any)->str:return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()
def read(path:Path)->dict[str,Any]:
 value=json.loads(path.read_text());need(isinstance(value,dict),f"JSON object required: {path}");return value
def atomic(path:Path,value:Mapping[str,Any])->None:
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");tmp.replace(path)
def _copy_pkg()->None:
 out=DEST/"artifacts/pkg"; need(not out.exists(),f"refusing existing package tree: {out}")
 for source,name in ((V1_SRC/"btransform_unified_v1","btransform_unified_v1"),(V2_SRC/"btransform_unified_v2","btransform_unified_v2")):
  target=out/name;target.mkdir(parents=True)
  for item in source.rglob("*.py"):
   if item.name in SKIP:continue
   dst=target/item.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(item,dst)
 (out/"btransform_unified_v2/__init__.py").write_text(SAFE_INIT)
def _validate_inputs()->tuple[dict,dict,dict,dict]:
 meta,train,score=(read(RUN/name) for name in ("run_meta.json","train_receipt.json","score_receipt.json")); replay=read(REPLAY)
 expected={"schema":"m1_rift_joint_train_v1","status":"FORMAL","cell":"M1-RIFT-R100-D4-JOINT-B3S-CONCAT-V1","arm":"B_ACTIVITY_ONLY","seed":42,"sampler_seed":42,"epochs":24,"context_bins":100,"depth":4,"layer_windows":[25,25,25,24],"official_test_used":False}
 need(all(meta.get(k)==v for k,v in expected.items()),"B run metadata identity drift")
 need(train.get("schema")=="m1_rift_joint_train_receipt_v1" and train.get("status")=="COMPLETED" and train.get("arm")=="B_ACTIVITY_ONLY" and train.get("epochs")==24 and train.get("steps")==UPDATES,"B train receipt incomplete")
 need(score.get("schema")=="m1_rift_joint_ho_calib_epoch_scan_v1" and score.get("status")=="COMPLETED" and score.get("arm")=="B_ACTIVITY_ONLY","B original score identity drift")
 need(set(score.get("ema_by_epoch",{}))=={str(e) for e in EPOCHS} and set(score.get("checkpoint_sha256_by_epoch",{}))=={str(e) for e in EPOCHS},"B old score scan incomplete")
 need(int(score.get("selection",{}).get("epoch",-1))==3,"B old-metric selected epoch must be e3")
 need(replay.get("schema")=="carrier_v4_m1_activity_only_baseline_replay_v1" and replay.get("status")=="COMPLETED" and replay.get("arm")=="B_ACTIVITY_ONLY","B channel replay identity drift")
 need(replay.get("baseline_run")==str(RUN.resolve()) and replay.get("baseline_train_receipt_sha256")==sha(RUN/"train_receipt.json") and replay.get("baseline_score_receipt_sha256")==sha(RUN/"score_receipt.json"),"replay binding to B run drift")
 need(set(replay.get("completed",{}))=={str(e) for e in EPOCHS} and int(replay.get("selection",{}).get("epoch",-1))==3,"B channel replay selected epoch must be e3")
 need(CKPT.is_file() and sha(CKPT)==score["checkpoint_sha256_by_epoch"]["3"] and sha(CKPT)==replay["completed"]["3"]["checkpoint_sha256"],"B e3 checkpoint binding drift")
 return meta,train,score,replay
def _official_banks()->dict:
 from m1_rift_falcon_decoder import CPUUnpickler
 with BANK_PAYLOAD.open("rb") as f:payload=CPUUnpickler(f).load()
 banks=payload["bank_by_dataset_tag"];need(set(banks)==TAGS,"official seven-tag M10/activity roster drift");return banks
def _load_b_model():
 from btransform_unified_v1.ema import DecoderEMA
 from btransform_unified_v2.joint_m1_model import ARM_B,JointM1ConcatDecoder
 import m1_joint_train as joint
 state=torch.load(CKPT,map_location="cpu",weights_only=False);need(state.get("schema")=="m1_rift_joint_epoch_checkpoint_v1" and state.get("config",{}).get("arm")==ARM_B and state.get("epoch")==3,"checkpoint is not B e3")
 model=JointM1ConcatDecoder(ARM_B,seed=42);model.load_state_dict(state["raw_state_dict"],strict=True);ema=DecoderEMA(model,decay=.9995);ema.load_state_dict(state["ema"]);ema.apply_to(model);model.eval()
 return model,joint
def _install(model,joint):
 import m1_projadd_depth2_series as legacy
 from btransform_unified_v1 import m1_projadd as mp
 dataset,_=legacy.build_fullsession_face(); src,_=legacy.build_fullsession_banks(dataset);cal=mp.calib_trials_from_dataset(dataset);ho=joint._ho_material()
 banks={s:src[s] for s in SOURCE}|{s:ho[s]["bank"] for s in HO}; calibs={s:cal[s] for s in SOURCE}|{s:ho[s]["calib10"] for s in HO};model.install_session_memory(banks,calibs);model.eval();return banks,calibs,ho,dataset
def _seal(model,banks,official):
 from btransform_unified_v1.m1_b3s_joint import encode_b3s
 sealed={}
 for tag,row in official.items():
  session=tag if tag in banks else f"ses-{tag}";bank=banks[session]
  with torch.inference_mode():e0,direct=model._identity([bank.session_id],torch.device("cpu"))
  raw=getattr(model,f"calib_{bank.session_id.replace('-','_')}").cpu();zero=torch.zeros((64,4),dtype=torch.float32)
  with torch.inference_mode():expected=encode_b3s(model.encoder,raw,zero)
  e=np.ascontiguousarray(e0[0].numpy(),np.float32);t=np.ascontiguousarray(direct[0].numpy(),np.float32)
  need(np.array_equal(t,np.zeros_like(t)) and np.array_equal(e,np.ascontiguousarray(expected.numpy(),np.float32)),f"{tag}: B route is not literal zero side/direct T")
  sealed[tag]={"E0":e,"T":t,"unit_mask":np.ascontiguousarray(row["unit_mask"],np.bool_),"session":tag,"e0_sha256":asha(e),"t_sha256":asha(t),"zero_direct_t":True,"zero_encoder_side":True}
 return sealed
def build()->dict:
 need(not PAYLOAD.exists(),f"refusing overwrite payload: {PAYLOAD}");meta,train,score,replay=_validate_inputs();official=_official_banks();model,joint=_load_b_model();banks,_cal,ho,_dataset=_install(model,joint);sealed=_seal(model,banks,official)
 from btransform_unified_v2.concat_model import RiftConcatDecoder
 concat=RiftConcatDecoder("m1",context_bins=100,bias_mode="recency",seed=42);allowed=set(concat.state_dict());state={k:v.detach().cpu().float().clone() for k,v in model.state_dict().items() if k in allowed};missing,unexpected=concat.load_state_dict(state,strict=True);need(not missing and not unexpected,"B EMA-to-static concat state mismatch");concat.eval()
 payload={"schema":"m1_rift_activity_only_r100_cached_falcon_payload_v1","task":"m1","arm":"B_ACTIVITY_ONLY","context_bins":100,"bias_mode":"recency","identity_interface":"ema_b3s_zero_side_static_e0_concat","information_route":{"encoder_side":"literal_zero","direct_t":"literal_zero","e0":"ema_b3s(activity, literal_zero_side)"},"behavior_scaling_factor":1.0,"checkpoint":str(CKPT),"checkpoint_sha256":sha(CKPT),"input_receipts_sha256":{"run_meta.json":sha(RUN/"run_meta.json"),"train_receipt.json":sha(RUN/"train_receipt.json"),"score_receipt.json":sha(RUN/"score_receipt.json"),"replay_receipt.json":sha(REPLAY)},"bank_source_payload_sha256":sha(BANK_PAYLOAD),"bank_by_dataset_tag":sealed,"ema_state_dict":{k:v.detach().cpu().float().clone() for k,v in concat.state_dict().items()},"selection":{"epoch":3,"old_metric_equal_session_mean":float(score["ema_by_epoch"]["3"]["equal_session_mean"]),"channel_metric_equal_session_mean":float(replay["completed"]["3"]["metrics"]["equal_session_mean_channel_variance_weighted_r2"]),"original_score_selected_epoch":3,"channel_replay_selected_epoch":3,"official_test_used":False}}
 PAYLOAD.parent.mkdir(parents=True,exist_ok=True)
 with PAYLOAD.open("wb") as f:pickle.dump(payload,f,protocol=4)
 _copy_pkg();return {"status":"BUILT_NOT_PACKAGED","payload":str(PAYLOAD),"payload_sha256":sha(PAYLOAD),"arm":"B_ACTIVITY_ONLY"}
def _full_context(raw: np.ndarray, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one W100 context and validity solely from raw-time geometry."""
    need(raw.ndim == 2 and raw.shape[1] == 64 and end >= 0, "invalid M1 host raw timeline")
    width=min(end+1,100); out=np.zeros((1,100,64),np.float32); out[0,100-width:]=raw[end-width+1:end+1]
    valid=torch.zeros((1,100),dtype=torch.bool);valid[:,100-width:]=True
    return torch.from_numpy(out),valid

def _cached_row_state(runtime: Any, row: int) -> tuple[torch.Tensor, ...]:
    """Snapshot every persistent cached-runtime field for one fixed row."""
    cached=runtime.cached_temporal
    need(cached is not None and runtime.last is not None,"B8 cached runtime state unavailable")
    return (runtime.raw4[row].clone(), runtime.last[row].clone(),
            *(value[row].clone() for value in cached.keys),
            *(value[row].clone() for value in cached.values),
            *(value[row].clone() for value in cached.lengths))

def _same_cached_row_state(before: tuple[torch.Tensor, ...], after: tuple[torch.Tensor, ...]) -> bool:
    return len(before)==len(after) and all(torch.equal(left,right) for left,right in zip(before,after))

def _host_inputs():
    """Return true raw bins after the known 99-bin M1 query padding."""
    model,joint=_load_b_model();banks,_cal,ho,dataset=_install(model,joint);raw={}
    for session in SOURCE:
        padded=np.ascontiguousarray(dataset.neural_data[session],np.float32)
        need(padded.ndim==2 and padded.shape[1]==64 and len(padded)>QUERY_PAD_BINS+HOST_STEPS,"source padded host timeline too short")
        raw[session.removeprefix("ses-")]=np.ascontiguousarray(padded[QUERY_PAD_BINS:])
    for session in HO:
        padded=np.ascontiguousarray(ho[session]["dataset"].neural_data[session],np.float32)
        need(padded.ndim==2 and padded.shape[1]==64 and len(padded)>QUERY_PAD_BINS+HOST_STEPS,"HO padded host timeline too short")
        raw[session]=np.ascontiguousarray(padded[QUERY_PAD_BINS:])
    return model,banks,raw

def host()->dict:
    from falcon_challenge.config import FalconConfig,FalconTask
    from btransform_unified_v2.concat_model import RiftConcatDecoder
    from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
    from btransform_unified_v2.streaming import RiftStreamDecoder
    from btransform_unified_v1.bank import TaskBank
    from m1_rift_falcon_decoder import _task_bank,load_payload,validate_payload,M1RiftActivityOnlyCachedFalconDecoder
    need(PAYLOAD.is_file(),"run --stage build first");payload=load_payload(PAYLOAD);validate_payload(payload);live,live_banks,raw=_host_inputs();config=FalconConfig(task=FalconTask.m1)
    static=RiftConcatDecoder("m1",context_bins=100,bias_mode="recency",seed=42);static.load_state_dict({k:torch.as_tensor(v) for k,v in payload["ema_state_dict"].items()},strict=True);static.eval()
    # Keep a deliberately all-zero but valid probe at a real time coordinate
    # in every independent trace. Every full/stream route below consumes the
    # same probe trace, so validity is never inferred from amplitude.
    traces={tag:np.ascontiguousarray(value[:HOST_STEPS].copy()) for tag,value in raw.items()}
    for value in traces.values(): value[1,:]=0.0
    per_tag={}; max_live_static=0.0; max_packed_full=0.0
    for tag in HO:
        session=tag; source_bank=live_banks[session]; sealed=_task_bank(tag,payload["bank_by_dataset_tag"][tag]); stem=f"sub-MonkeyL-held-out-calib_ses-{tag}_behavior+ecephys"
        packed=M1RiftActivityOnlyCachedFalconDecoder(config,str(PAYLOAD),1);packed.reset([stem])
        trace=traces[tag];steps=len(trace); tag_live_static=0.0;tag_packed_full=0.0
        for end in range(steps):
            x,valid=_full_context(trace,end)
            with torch.inference_mode(): live_out=live(x,source_bank,input_valid_mask=valid); static_out=static(x,sealed,input_valid_mask=valid)
            live_delta=float((live_out-static_out).abs().max());tag_live_static=max(tag_live_static,live_delta)
            packed_out=packed.predict(x[0,-1:].numpy()); full_delta=float(np.max(np.abs(packed_out-static_out.numpy())));tag_packed_full=max(tag_packed_full,full_delta)
        need(tag_live_static<=GATE,f"{tag}: live B EMA/static payload parity failed")
        need(tag_packed_full<=GATE,f"{tag}: packed/full R100 parity failed")
        per_tag[tag]={"steps":steps,"live_b_ema_vs_static_max_abs":tag_live_static,"packed_vs_full_max_abs":tag_packed_full,"startup_endpoints_checked":[0,1],"full_context_endpoint_checked":99,"true_zero_valid_endpoint":1}
        max_live_static=max(max_live_static,tag_live_static);max_packed_full=max(max_packed_full,tag_packed_full)
    # Eight independent bank instances: seven real payload tags plus a separately
    # addressed clone solely for host batching, never an eighth official tag.
    tags=sorted(TAGS);banks=[_task_bank(tag,payload["bank_by_dataset_tag"][tag]) for tag in tags]
    clone_row=payload["bank_by_dataset_tag"][tags[0]];banks.append(_task_bank(tags[0],clone_row,session_id=f"{tags[0]}#host-clone"))
    ids=[f"host-b8-{i}" for i in range(8)];rt8=CpuRiftRuntime(static,banks,ids,temporal_backend="cached");histories=[[] for _ in range(8)];max_b8_full=0.0;inactive_checked=False;zero_valid_checked=False
    for end in range(HOST_STEPS):
        observed=np.stack([traces[tags[i%7]][end] for i in range(8)]).astype(np.float32,copy=False);valid_rows=torch.ones(8,dtype=torch.bool)
        if end==1: zero_valid_checked=bool(not np.any(observed[6]))
        inactive_before=None
        if end==2:
            valid_rows[7]=False
            inactive_before=_cached_row_state(rt8,7)
        got=rt8.advance(torch.from_numpy(observed),valid_mask=valid_rows)
        for row,bank in enumerate(banks):
            if not bool(valid_rows[row]):
                # The returned inactive score is not used by Falcon; meanwhile
                # ``runtime.last`` retains the prior score.
                # Verify the actual persistent state: raw window, output cache,
                # and every cached temporal KV buffer/length for this row.
                need(inactive_before is not None and _same_cached_row_state(inactive_before,_cached_row_state(rt8,row)),"inactive B8 row advanced cached state")
                inactive_checked=True
                continue
            histories[row].append(np.ascontiguousarray(observed[row].copy()))
            consumed=np.stack(histories[row])
            # Reference is each row's valid-consumed history. An inactive step
            # is absent from both the cached runtime history and this context.
            x,mask=_full_context(consumed,len(consumed)-1)
            with torch.inference_mode(): want=static(x,bank,input_valid_mask=mask)[0]
            max_b8_full=max(max_b8_full,float((got[row]-want).abs().max()))
    need(max_b8_full<=GATE and inactive_checked and zero_valid_checked,"B8 cached/full or inactive/true-zero route failed")
    # Query-pad parity calls all 99 invalid positions explicitly. Validity is
    # constructed from the time coordinate alone; the t1 observed bin is an
    # all-zero but valid raw observation in ``traces``.
    tag="20121004";bank=_task_bank(tag,payload["bank_by_dataset_tag"][tag]);stream_pad=RiftStreamDecoder(static);padded_out=None
    for time_index in range(100):
        row=torch.zeros((1,64),dtype=torch.float32) if time_index < QUERY_PAD_BINS else torch.from_numpy(traces[tag][0:1])
        padded_out=stream_pad.stream_step(row,bank,["querypad"],valid_mask=torch.tensor([time_index >= QUERY_PAD_BINS]))
    x0,mask0=_full_context(traces[tag],0)
    with torch.inference_mode(): full0=static(x0,bank,input_valid_mask=mask0)
    querypad_delta=float((full0-padded_out).abs().max()) if padded_out is not None else float("inf")
    need(querypad_delta<=2e-5,"t0 query-pad full/stream parity failed")
    stream_full=RiftStreamDecoder(static);streamed=None
    for end in range(100): streamed=stream_full.stream_step(torch.from_numpy(traces[tag][end:end+1]),bank,["full-context"],valid_mask=torch.ones(1,dtype=torch.bool))
    x,mask=_full_context(traces[tag],99)
    with torch.inference_mode(): full=static(x,bank,input_valid_mask=mask)
    fullstream_delta=float((full-streamed).abs().max()) if streamed is not None else float("inf")
    need(fullstream_delta<=2e-5,"full-context stream true-zero parity failed")
    # Smoke intentionally uses unmodified raw observations. The separate probe
    # checks above cover literal-zero-valid routing without changing this input.
    smoke=DEST/"artifacts/smoke_window.npz";need(not smoke.exists(),f"refusing overwrite smoke evidence: {smoke}");stem=f"sub-MonkeyL-held-out-calib_ses-{tag}_behavior+ecephys";smoke_raw=np.ascontiguousarray(raw[tag][:40]);packed=M1RiftActivityOnlyCachedFalconDecoder(config,str(PAYLOAD),1);packed.reset([stem]);expected=None
    for row in smoke_raw:expected=packed.predict(row[None,:])
    need(expected is not None,"smoke endpoint missing");np.savez(smoke,tag_stem=np.asarray(stem),window=smoke_raw,expected=expected)
    route={"sealed_t_literal_zero":all(not bool(np.any(np.asarray(row["T"]))) for row in payload["bank_by_dataset_tag"].values()),"live_b_ema_zero_side_vs_static_checked":max_live_static<=GATE,"static_e0_sha_checked_by_runtime":all(asha(row["E0"]) == row["e0_sha256"] and asha(row["T"]) == row["t_sha256"] for row in payload["bank_by_dataset_tag"].values())}
    need(all(route.values()),"payload route/SHA validation report is false")
    report={"status":"HOST_PACK_VERIFY_PASS","payload_sha256":sha(PAYLOAD),"per_ho_tag":per_tag,"max_live_b_ema_vs_static":max_live_static,"max_packed_vs_full":max_packed_full,"B8_cached_vs_full_max_abs":max_b8_full,"B8_inactive_row_checked":inactive_checked,"B8_true_zero_valid_checked":zero_valid_checked,"querypad_t0_full_stream_max_abs":querypad_delta,"full_context_stream_truezero_max_abs":fullstream_delta,"full_stream_startup_truezero_checked":fullstream_delta<=2e-5,"route":route,"raw_input_contract":{"query_pad_bins_stripped":QUERY_PAD_BINS,"validity":"derived from raw end geometry, never neural amplitude"},"smoke_window":str(smoke)};atomic(DEST/"artifacts/host_verify.json",report);return report
def docker()->dict:
 need(PAYLOAD.is_file() and (DEST/"artifacts/pkg").is_dir(),"run --stage build first");digest=sha(PAYLOAD);subprocess.run(["docker","build","--build-arg",f"PAYLOAD_SHA256={digest}","-t",IMAGE_TAG,str(DEST)],check=True);info=subprocess.check_output(["docker","image","inspect",IMAGE_TAG,"--format","{{.Id}} {{.Size}}"],text=True).strip().split();return {"status":"DOCKER_BUILT_NOT_PUBLISHED","image_tag":IMAGE_TAG,"image_id":info[0],"image_size":int(info[1]),"payload_sha256":digest}
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("build","host","docker"),required=True);a=p.parse_args();os.environ.setdefault("PYTHONNOUSERSITE","1");os.environ.setdefault("CUDA_VISIBLE_DEVICES","");print(json.dumps({"build":build,"host":host,"docker":docker}[a.stage](),indent=2,sort_keys=True))
if __name__=="__main__":main()
