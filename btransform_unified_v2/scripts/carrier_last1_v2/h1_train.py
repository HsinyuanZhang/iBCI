"""Train/score H1 on the fixed chronological leave-last-one-dates split."""
from __future__ import annotations

import argparse, contextlib, hashlib, inspect, json, random, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CROSS = ROOT / "scripts" / "cross_session_v1"
STREAMING_ROOT = ROOT.parent / "streaming_calibration_exp"
# Fresh-process import order.  The v2 package initializer imports the M2
# module, whose encoder intentionally imports ``src.models...`` from this
# checkout; make that origin explicit before importing any v2 module.
for item in (str(ROOT.parent), str(ROOT.parent / "btransform_unified_v1" / "src"),
             str(CROSS), str(ROOT / "src"), str(STREAMING_ROOT), str(HERE)):
    if item in sys.path: sys.path.remove(item)
    sys.path.insert(0, item)
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
streaming_origin = Path(inspect.getfile(SideFeatureEarlyPoolEncoder)).resolve()
if STREAMING_ROOT not in streaming_origin.parents:
    raise RuntimeError(f"streaming encoder must come from {STREAMING_ROOT}, got {streaming_origin}")
from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import whole_unit_dropout, unit_dropout_seed
from btransform_unified_v1.schedule import warmup_cosine_lr
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v2.cross_session_h1_model import ARMS, ARM_B, ARM_D, ARM_Z, CrossSessionH1Decoder
import h1_prepare as prep

BATCH, EPOCHS, SCALE, EMA_DECAY = 32, 32, 20.0, .9995

def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
def sha_array(x: np.ndarray) -> str:
    a=np.ascontiguousarray(x); return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); tmp.replace(path)
def torch_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); torch.save(value,tmp); tmp.replace(path)
def parameter_hash(model: nn.Module) -> str:
    h=hashlib.sha256()
    for n,v in model.named_parameters(): h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def _manifest(prepared: Path, surface: str) -> dict:
    row=json.loads((prepared/surface/"manifest.json").read_text()); c=prep.split_contract()
    if row.get("schema")!=prep.PREPARE_SCHEMA or not prep.matches_split_contract(row): raise RuntimeError("chronological manifest/split mismatch")
    expected=c["source_sessions"] if surface=="source" else c["target_sessions"]
    if sorted(row.get("records",{}))!=sorted(expected): raise RuntimeError(f"{surface} roster mismatch")
    authority=row["source_authority"]
    if tuple(authority.get("source_sessions",()))!=tuple(c["source_sessions"]) or authority.get("split_id")!=prep.SPLIT_ID or authority.get("target_records_opened")!=0: raise RuntimeError("source authority roster/leak mismatch")
    return row

def audit_npz(path:Path,row:dict,arm:str)->None:
    with np.load(path,allow_pickle=False) as z:
        required={"X","valid","y","ends","starts","segment_starts","support","query","native_query_indices","stride"}
        if not required.issubset(z.files): raise RuntimeError(f"{path}: prepared fields missing")
        X,y,valid=z["X"],z["y"],z["valid"]; starts,ends,seg=z["starts"],z["ends"],z["segment_starts"]
        if sha_array(X)!=row["X_sha256"] or sha_array(y)!=row["y_sha256"] or sha_array(starts)!=row["starts_sha256"] or sha_array(ends)!=row["endpoint_sha256"] or sha_array(seg)!=row["segment_starts_sha256"]: raise RuntimeError(f"{path}: array hash mismatch")
        if int(z["stride"])!=int(row["stride"]) or set(map(float,z["support"]))&set(map(float,z["query"])): raise RuntimeError(f"{path}: support/query overlap")
        widths=ends-starts+1
        if not (len(X)==len(y)==len(valid)==len(starts)==len(ends)==len(seg) and np.all(starts>=seg) and np.all(ends>=starts) and np.all(valid.sum(1)==widths) and not np.any(widths>300) and not np.any(valid[:,:-1] & ~valid[:,1:])): raise RuntimeError(f"{path}: native geometry/reset/padding mismatch")
        if arm in (ARM_B,ARM_D) and ("activity" not in z.files or sha_array(z["activity"])!=row["activity_sha256"]): raise RuntimeError(f"{path}: activity binding")
        if arm==ARM_D and ("carrier" not in z.files or sha_array(z["carrier"])!=row["carrier_sha256"]): raise RuntimeError(f"{path}: carrier binding")

def load_surface(prepared:Path,surface:str,arm:str,validation:bool=False):
    manifest=_manifest(prepared,surface); banks={};data={};activity={}
    for session,row in sorted(manifest["records"].items()):
        selected=row["validation"] if validation else row; suffix=".val.npz" if validation else ".npz"; path=prepared/surface/f"{session}{suffix}";audit_npz(path,selected,arm)
        if validation and set(map(float,row["query_trials"])) & set(map(float,selected["query_trials"])):
            raise RuntimeError(f"{session}: source train/validation trial overlap")
        with np.load(path,allow_pickle=False) as z:
            X=np.ascontiguousarray(z["X"],np.float32);y=np.ascontiguousarray(z["y"],np.float32);valid=np.ascontiguousarray(z["valid"],np.bool_); ends=np.ascontiguousarray(z["ends"],np.int64)
            carrier=np.ascontiguousarray(z["carrier"],np.float32) if arm==ARM_D else np.zeros((176,4),np.float32); e0=np.zeros((176,700),np.float32)
            banks[session]=TaskBank(session,e0,carrier,np.ones(176,bool),X,y,ends,{"shape":(176,700),"trial_count":3,"budget":3,"estimator":"chrono source-only H-C","array_sha256":array_sha256(e0)})
            data[session]=(X,y,valid)
            if arm in (ARM_B,ARM_D): activity[session]=torch.from_numpy(np.ascontiguousarray(z["activity"],np.float32))
    return manifest,banks,data,activity

def make_model(arm,seed,device,banks,activity):
    m=CrossSessionH1Decoder(arm,seed=seed).to(device);m.install_memory(banks,activity);return m
def strict_ema(model,ema,allow_uninitialized=False):
    p=dict(model.named_parameters())
    if set(p)!=set(ema.shadow): raise RuntimeError("EMA parameter key mismatch")
    pointers=set()
    for n,v in ema.shadow.items():
        if tuple(p[n].shape)!=tuple(v.shape) or v.data_ptr() in pointers: raise RuntimeError("invalid EMA shape/storage")
        pointers.add(v.data_ptr())
        if not (allow_uninitialized and ema.n_updates==0) and not torch.isfinite(v).all(): raise RuntimeError("invalid EMA")
    if ema.n_updates==0 and not allow_uninitialized: raise RuntimeError("EMA has no optimizer update")
def swap_ema(model,ema):
    strict_ema(model,ema); saved={n:v.detach().clone() for n,v in model.named_parameters()}
    with torch.no_grad():
        for n,v in model.named_parameters(): v.copy_(ema.shadow[n].to(v.device,v.dtype))
    return saved
def restore(model,saved):
    with torch.no_grad():
        for n,v in model.named_parameters():v.copy_(saved[n])

def date_aggregate(per_session:dict[str,dict])->dict:
    dates={d:[per_session[s]["r2"] for s in prep.H1_SESSIONS_BY_DATE[d] if s in per_session] for d in (*prep.SOURCE_DATES,*prep.TARGET_DATES)}
    dates={d:float(np.mean(v)) for d,v in dates.items() if v}
    return {"per_date":dates,"equal_date_mean":float(np.mean(list(dates.values()))),"equal_session_mean":float(np.mean([r["r2"] for r in per_session.values()]))}

def evaluate(model,ema,banks,data,device,artifact_dir:Path|None=None):
    saved=swap_ema(model,ema); was=model.training; model.eval(); rows={}
    try:
        for s,(X,y,valid) in data.items():
            chunks=[]
            for off in range(0,len(X),BATCH):
                with torch.inference_mode(): chunks.append((model(torch.from_numpy(X[off:off+BATCH]).to(device),banks[s],input_valid_mask=torch.from_numpy(valid[off:off+BATCH]).to(device))/SCALE).float().cpu().numpy())
            pred=np.ascontiguousarray(np.concatenate(chunks),np.float32)
            artifact=None
            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True,exist_ok=True);artifact=artifact_dir/f"{s}.npz";np.savez_compressed(artifact,prediction=pred,target=y,endpoints=banks[s].window_ids)
            rows[s]={"r2":float(variance_weighted_r2(y,pred)),"windows":int(len(y)),"y_sha256":sha_array(y),"prediction_sha256":sha_array(pred),"endpoint_sha256":sha_array(banks[s].window_ids),"artifact":None if artifact is None else str(artifact),"artifact_sha256":None if artifact is None else sha_file(artifact)}
    finally: restore(model,saved);model.train(was)
    return {"per_session":rows,**date_aggregate(rows),"n_windows":int(sum(x["windows"] for x in rows.values()))}

def binding(args,manifest,model):
    return {"schema":"h1_carrier_last1_train_v1","split":prep.split_contract(),"arm":args.arm,"seed":args.seed,"epochs":EPOCHS,"batch":BATCH,"source_manifest_sha256":sha_file(args.prepared/"source"/"manifest.json"),"initial_parameter_sha256":parameter_hash(model),"source_code_sha256":{str(p):sha_file(p) for p in (Path(__file__),HERE/"h1_prepare.py",CROSS/"h1_prepare.py",CROSS/"h1_train.py",ROOT/"src/btransform_unified_v2/cross_session_h1_model.py")}}
def bind_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def rng_state(): return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])

def perturbation_preflight(args,device,banks,data,activity):
    models={a:make_model(a,args.seed,device,banks,activity if a in (ARM_B,ARM_D) else {}) for a in ARMS};params={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()}
    states={a:dict(m.named_parameters()) for a,m in models.items()};common=set(states[ARM_Z]);enc_b={n for n in states[ARM_B] if n.startswith("encoder.")};enc_d={n for n in states[ARM_D] if n.startswith("encoder.")}
    if common != set(states[ARM_B])-enc_b or common != set(states[ARM_D])-enc_d or not enc_b or enc_b!=enc_d: raise RuntimeError("Z/B/D decoder/encoder topology mismatch")
    if not all(torch.equal(states[ARM_Z][n].cpu(),states[ARM_B][n].cpu()) and torch.equal(states[ARM_Z][n].cpu(),states[ARM_D][n].cpu()) for n in common): raise RuntimeError("shared decoder initial states differ")
    if not all(torch.equal(states[ARM_B][n].cpu(),states[ARM_D][n].cpu()) for n in enc_b): raise RuntimeError("B/D encoder initial states differ")
    if max(params.values())/min(params.values())-1 >= .05: raise RuntimeError("arm parameter difference >=5%")
    first=next(iter(data));X,y,v=data[first]; raw=torch.from_numpy(X[:2]).to(device);mask=torch.from_numpy(v[:2]).to(device); checks={}
    for arm,m in models.items():
        m.train();m.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(m(raw,banks[first],input_valid_mask=mask).float(),torch.from_numpy(y[:2]*SCALE).to(device));loss.backward();checks[arm]=bool(torch.isfinite(loss) and any(q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum()>0 for q in m.parameters()))
    if not all(checks.values()):raise RuntimeError("nonfinite/no-gradient preflight")
    for m in models.values():m.eval()
    def altered(bank):return TaskBank(bank.session_id,bank.E0,bank.carrier+.25,bank.unit_mask,bank.X_store,bank.target_store,bank.window_ids,bank.calibration_meta)
    with torch.inference_mode():
        z0=models[ARM_Z](raw,banks[first],input_valid_mask=mask);zc=models[ARM_Z](raw,altered(banks[first]),input_valid_mask=mask);b0=models[ARM_B](raw,banks[first],input_valid_mask=mask);bc=models[ARM_B](raw,altered(banks[first]),input_valid_mask=mask);d0=models[ARM_D](raw,banks[first],input_valid_mask=mask)
        ba,da,dc=next(iter(models[ARM_B]._activity.values())),next(iter(models[ARM_D]._activity.values())),next(iter(models[ARM_D]._carrier.values()));oba,oda,odc=ba.clone(),da.clone(),dc.clone();ba.add_(.25);b_a=models[ARM_B](raw,banks[first],input_valid_mask=mask);ba.copy_(oba);da.add_(.25);d_a=models[ARM_D](raw,banks[first],input_valid_mask=mask);da.copy_(oda);dc.add_(.25);d_c=models[ARM_D](raw,banks[first],input_valid_mask=mask);dc.copy_(odc);da.add_(.25);dc.add_(.25);d_both=models[ARM_D](raw,banks[first],input_valid_mask=mask);da.copy_(oda);dc.copy_(odc)
    changed=lambda x,y:not torch.equal(x,y);effects={"z_calibration_insensitive":not changed(z0,zc),"b_activity_sensitive":changed(b0,b_a),"b_carrier_insensitive":not changed(b0,bc),"d_activity_sensitive":changed(d0,d_a),"d_carrier_sensitive":changed(d0,d_c),"d_dual_path_sensitive":changed(d0,d_both)}
    if not all(effects.values()):raise RuntimeError(f"invalid arm dataflow: {effects}")
    return {"trainable_parameter_counts":params,"parameter_delta_fraction":max(params.values())/min(params.values())-1,"shared_decoder_initial_states_byte_equal":True,"b_d_encoder_initial_state_equal":True,"z_has_no_encoder":not hasattr(models[ARM_Z],"encoder"),"gradient_checks":checks,"effects":effects,"z_output_sha256":hashlib.sha256(z0.cpu().numpy().tobytes()).hexdigest()}

def preflight(args):
    manifest,banks,data,activity=load_surface(args.prepared,"source",ARM_D);device=torch.device(args.device);checks=perturbation_preflight(args,device,banks,data,activity)
    row={"schema":"h1_carrier_last1_preflight_v1","status":"PASSED","split":prep.split_contract(),"arm":args.arm,"source_only":True,"target_records_opened":0,"source_manifest_sha256":sha_file(args.prepared/"source"/"manifest.json"),**checks}
    json_atomic(args.dest/"preflight.json",row);return row

def train(args):
    pf=json.loads((args.dest/"preflight.json").read_text());
    if pf.get("status")!="PASSED" or pf.get("split")!=prep.split_contract():raise RuntimeError("passed chronological source-only preflight required")
    manifest,banks,data,activity=load_surface(args.prepared,"source",args.arm);_,vbanks,vdata,_=load_surface(args.prepared,"source",args.arm,True);device=torch.device(args.device);model=make_model(args.arm,args.seed,device,banks,activity);ema=DecoderEMA(model,EMA_DECAY);strict_ema(model,ema,allow_uninitialized=True);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);meta=binding(args,manifest,model);recipe=bind_hash(meta);run=args.dest
    if run.exists() and args.resume is None and any(p.name not in {"preflight.json"} for p in run.iterdir()):raise FileExistsError("fresh chronological run refuses existing artifacts")
    run.mkdir(parents=True,exist_ok=True);json_atomic(run/"run_meta.json",{**meta,"recipe_binding":recipe})
    start=1;step=0;curve=[]
    if args.resume:
        state=torch.load(args.resume,map_location="cpu",weights_only=False)
        if state.get("recipe_binding")!=recipe or any(k not in state for k in ("model","opt","ema","epoch","step","curve","rng")):raise RuntimeError("resume rejected: old LODO or different chronological source authority")
        model.load_state_dict(state["model"],strict=True);opt.load_state_dict(state["opt"]);ema.load_state_dict(state["ema"]);strict_ema(model,ema);restore_rng(state["rng"]);start=int(state["epoch"])+1;step=int(state["step"]);curve=list(state["curve"])
    updates=sum((len(X)+BATCH-1)//BATCH for X,_,_ in data.values()); began=time.monotonic()
    for epoch in range(start,EPOCHS+1):
        model.train();rng=np.random.default_rng(args.seed+epoch);sessions=sorted(data);rng.shuffle(sessions);losses=[];sampler_digest=hashlib.sha256();sampler_digest.update(np.asarray(sessions,dtype="S").tobytes())
        for s in sessions:
            X,y,v=data[s];order=rng.permutation(len(X));sampler_digest.update(s.encode());sampler_digest.update(order.astype(np.int64).tobytes())
            for off in range(0,len(order),BATCH):
                ix=order[off:off+BATCH];step+=1;opt.param_groups[0]["lr"]=warmup_cosine_lr(step,EPOCHS*updates,updates,peak=1e-4,min_factor=.1);keep=whole_unit_dropout(torch.ones(176,dtype=torch.bool),p=.1,generator=torch.Generator().manual_seed(unit_dropout_seed(args.seed,epoch,step)));sampler_digest.update(keep.numpy().tobytes());opt.zero_grad(set_to_none=True);amp=torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" else contextlib.nullcontext()
                with amp: pred=model(torch.from_numpy(X[ix]).to(device),banks[s],dropout_keep=keep,input_valid_mask=torch.from_numpy(v[ix]).to(device));loss=nn.functional.mse_loss(pred.float(),torch.from_numpy(y[ix]*SCALE).to(device))
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite loss")
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();ema.update_after_step(model);losses.append(float(loss.detach()))
        val=evaluate(model,ema,vbanks,vdata,device);epoch_row={"epoch":epoch,"step":step,"train_mse":float(np.mean(losses)),"source_val_ema":val,"sampler_endpoint_keep_sha256":sampler_digest.hexdigest(),"elapsed_seconds":time.monotonic()-began};curve.append(epoch_row);print(json.dumps(epoch_row,sort_keys=True),flush=True);torch_atomic(run/f"ema_epoch_{epoch:03d}.pt",{"schema":"h1_carrier_last1_ema_v1","epoch":epoch,"ema":ema.state_dict(),"recipe_binding":recipe});torch_atomic(run/"resume_latest.pt",{"schema":"h1_carrier_last1_resume_v1","epoch":epoch,"step":step,"model":model.state_dict(),"opt":opt.state_dict(),"ema":ema.state_dict(),"curve":curve,"rng":rng_state(),"recipe_binding":recipe})
    selected=max(curve,key=lambda r:(r["source_val_ema"]["equal_date_mean"],-r["epoch"]));receipt={"schema":"h1_carrier_last1_train_receipt_v1","meta":meta,"recipe_binding":recipe,"selection":"earliest maximum source validation equal_date_mean; equal_session_mean diagnostic only","selected":selected,"curve":curve,"source_files":{s:{"train":sha_file(args.prepared/"source"/f"{s}.npz"),"validation":sha_file(args.prepared/"source"/f"{s}.val.npz")} for s in data},"checkpoints":{f"ema_epoch_{i:03d}":sha_file(run/f"ema_epoch_{i:03d}.pt") for i in range(1,EPOCHS+1)}};json_atomic(run/"selection.json",selected);json_atomic(run/"train_receipt.json",receipt);json_atomic(args.prepared/"source"/f"selection_seal_{args.arm}.json",{"schema":"h1_carrier_last1_selection_seal_v1","split":prep.split_contract(),"arm":args.arm,"source_manifest_sha256":sha_file(args.prepared/"source"/"manifest.json"),"train_receipt":str((run/"train_receipt.json").resolve()),"train_receipt_sha256":sha_file(run/"train_receipt.json"),"selected_epoch":selected["epoch"],"selected":selected});return receipt

def score_one(args,epoch,banks,data,activity,device,label):
    meta=json.loads((args.dest/"run_meta.json").read_text());receipt=json.loads((args.dest/"train_receipt.json").read_text());path=args.dest/f"ema_epoch_{epoch:03d}.pt";state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("recipe_binding")!=meta.get("recipe_binding") or state.get("schema")!="h1_carrier_last1_ema_v1" or int(state.get("epoch",-1))!=epoch or receipt.get("checkpoints",{}).get(f"ema_epoch_{epoch:03d}")!=sha_file(path):raise RuntimeError("checkpoint receipt/epoch binding mismatch")
    model=make_model(args.arm,args.seed,device,banks,activity);ema=DecoderEMA(model,EMA_DECAY);ema.load_state_dict(state["ema"]);strict_ema(model,ema);return {"epoch":epoch,"checkpoint_sha256":sha_file(args.dest/f"ema_epoch_{epoch:03d}.pt"),"target":evaluate(model,ema,banks,data,device,args.dest/"target_score_artifacts"/label)}
def score(args):
    receipt=json.loads((args.dest/"train_receipt.json").read_text());meta=json.loads((args.dest/"run_meta.json").read_text());
    curve=receipt.get("curve",[]);selected=receipt.get("selected",{})
    if receipt.get("recipe_binding")!=meta.get("recipe_binding") or receipt["meta"].get("split")!=prep.split_contract() or receipt["meta"].get("arm")!=args.arm or len(curve)!=EPOCHS or [r.get("epoch") for r in curve]!=list(range(1,EPOCHS+1)) or selected!=max(curve,key=lambda r:(r["source_val_ema"]["equal_date_mean"],-r["epoch"])):raise RuntimeError("unbound or incomplete chronological source selection")
    manifest,banks,data,activity=load_surface(args.prepared,"target",args.arm);epoch=int(receipt["selected"]["epoch"]);device=torch.device(args.device);out={"schema":"h1_carrier_last1_target_score_v1","split":prep.split_contract(),"arm":args.arm,"target_query_labels_used_for_gradients":False,"target_query_labels_used_for_selection":False,"selected_source_epoch":epoch,"selected_ema":score_one(args,epoch,banks,data,activity,device,"selected_ema"),"fixed_e32_ema":score_one(args,32,banks,data,activity,device,"fixed_e32_ema"),"target_manifest_sha256":sha_file(args.prepared/"target"/"manifest.json")};json_atomic(args.dest/"target_score.json",out);return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--dest",type=Path,required=True);p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--stage",choices=("preflight","train","score"),required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda:0");p.add_argument("--cpu-threads",type=int,default=2);p.add_argument("--resume",type=Path);a=p.parse_args()
    if a.seed!=42 or a.cpu_threads<1:raise ValueError("requires seed42 and positive cpu threads")
    a.prepared=a.prepared.resolve();a.dest=a.dest.resolve();torch.set_num_threads(a.cpu_threads);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    print(json.dumps({"preflight":preflight,"train":train,"score":score}[a.stage](a),indent=2,sort_keys=True))
if __name__=="__main__":main()
