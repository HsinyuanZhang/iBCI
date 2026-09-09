#!/usr/bin/env python3
"""Chronological two-source M1 Z/B/D trainer; target scoring is separate."""
from __future__ import annotations
import argparse, hashlib, json, random, time
from contextlib import nullcontext
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT/"src"),str(ROOT.parent/"btransform_unified_v1/src"),str(ROOT.parent)]
from btransform_unified_v2.cross_session_m1_model import ARMS,CrossSessionM1Decoder,Z_NONE,B_ACTIVITY_ONLY,D_JOINT
from m1_data import SPLIT_ID, SOURCE_SESSIONS, TARGET_SESSIONS, contract, assert_contract, materialize_sources
EPOCHS,BATCH,LR,EMA_DECAY,UNIT_DROPOUT=24,32,1e-4,.9995,.1

def atomic_json(p:Path,v:object)->None:
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n"); t.replace(p)
def atomic_torch(p:Path,v:object)->None:
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp"); torch.save(v,t); t.replace(p)
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def state_hash(model:nn.Module, prefix:str="")->str:
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):
        if n.startswith(prefix): h.update(n.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def common_init_hash(*models:nn.Module)->str:
    states=[m.state_dict() for m in models]; keys=set.intersection(*(set(s) for s in states))
    h=hashlib.sha256()
    for n in sorted(keys):
        values=[s[n].detach().cpu() for s in states]
        if not all(torch.equal(values[0],v) for v in values[1:]): raise RuntimeError(f"shared initialization drift at {n}")
        h.update(n.encode());h.update(values[0].contiguous().numpy().tobytes())
    return h.hexdigest()
def array_hash(x)->str:
    a=np.ascontiguousarray(np.asarray(x)); return hashlib.sha256(a.view(np.uint8)).hexdigest()
def chain_digest(previous:str,payload:bytes)->str:
    h=hashlib.sha256();h.update(bytes.fromhex(previous));h.update(payload);return h.hexdigest()
def source_evidence(m)->dict[str,object]:
    out={"calib":{s:array_hash(v) for s,v in m["calib"].items()},"carrier":{s:array_hash(v.carrier) for s,v in m["banks"].items()},"rsyn3_dictionary_and_normalizer":{s:array_hash(v) for s,v in m["rsyn3"].items()}}
    for split in ("train","val"):
        ds=m[split];out[split]={"neural":{s:array_hash(v) for s,v in ds.neural_data.items()},"targets":{s:array_hash(v) for s,v in ds.covariate_data.items()},"window_starts":array_hash(np.asarray(ds.window_indices,dtype=object).astype(str))}
    return out
def metadata(a:argparse.Namespace)->dict[str,object]:
    row=contract(); assert_contract(row)
    return {"schema":"m1_chronological_last2_v1","split_id":SPLIT_ID,"sources":list(SOURCE_SESSIONS),"targets":list(TARGET_SESSIONS),"arm":a.arm,"seed":a.seed,"data_contract":row,"recipe":{"epochs":EPOCHS,"batch":BATCH,"lr":LR,"weight_decay":.01,"ema":EMA_DECAY,"unit_dropout":UNIT_DROPOUT,"context":100,"precision":"bf16 decoder; encoder fp32","attention_backend":"local"},"selection":{"surface":"source validation only [310,end)","rule":"earliest maximum equal-session EMA","target_query_labels_used":False},"reporting":{"target":"selected source-val EMA and predeclared EMA epoch24","target_gradients":0}}

class IndexedWindows(Dataset):
    def __init__(self,base): self.base=base; self.window_indices=base.window_indices; self.window_size=base.window_size
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        x,y,_,s=self.base[i]; _,start=self.window_indices[i]; valid=np.arange(self.window_size)>=max(0,self.window_size-1-int(start))
        return x,y,s,np.ascontiguousarray(valid),int(start)
class FixedSessionSampler(Sampler[list[int]]):
    def __init__(self,ds:IndexedWindows,seed:int,epoch:int,shuffle:bool,drop_last:bool=True):
        groups={}
        for i,(s,_) in enumerate(ds.window_indices): groups.setdefault(s,[]).append(i)
        out=[]
        for s,ix in sorted(groups.items()):
            r=random.Random(f"m1-cross-session|{seed}|{epoch}|{s}"); ix=list(ix); r.shuffle(ix) if shuffle else None
            limit=len(ix)-BATCH+1 if drop_last else len(ix)
            out += [ix[j:min(j+BATCH,len(ix))] for j in range(0,limit,BATCH)]
        r=random.Random(f"m1-cross-session-batches|{seed}|{epoch}"); r.shuffle(out) if shuffle else None; self.batches=out
    def __iter__(self): yield from self.batches
    def __len__(self): return len(self.batches)
def collate(rows):
    return (torch.stack([torch.as_tensor(r[0]) for r in rows]),torch.stack([torch.as_tensor(r[1]) for r in rows]),[str(r[2].decode() if isinstance(r[2],bytes) else r[2]) for r in rows],torch.stack([torch.as_tensor(r[3],dtype=torch.bool) for r in rows]),torch.tensor([r[4] for r in rows],dtype=torch.int64))
def loader(ds,seed:int,epoch:int=0,shuffle:bool=False):
    wrap=IndexedWindows(ds); return DataLoader(wrap,batch_sampler=FixedSessionSampler(wrap,seed,epoch,shuffle,drop_last=shuffle),collate_fn=collate,num_workers=0)
def _ema_apply(model,ema):
    params=dict(model.named_parameters()); canonical=dict(model.named_parameters(remove_duplicate=True)); aliases=dict(model.named_parameters(remove_duplicate=False))
    if set(canonical)!=set(ema.shadow) or set(params)!=set(canonical): raise RuntimeError("EMA must cover every named parameter exactly")
    for n,p in aliases.items():
        if n not in params and not any(p.data_ptr()==q.data_ptr() for q in params.values()): raise RuntimeError("unrecognised parameter alias")
    before={n:b.detach().clone() for n,b in model.named_buffers()}
    with torch.no_grad():
        for n,p in params.items(): p.copy_(ema.shadow[n].to(p.device,p.dtype))
    for n,b in model.named_buffers():
        if not torch.equal(before[n],b): raise RuntimeError(f"EMA changed fixed buffer {n}")
def _ema_eval(model,ema,ds,banks,device,audit_path:Path|None=None):
    raw={n:p.detach().clone() for n,p in model.named_parameters()}; model.eval(); _ema_apply(model,ema); out={}
    with torch.inference_mode():
        for x,y,s,v,starts in loader(ds,42,shuffle=False):
            for name in dict.fromkeys(s):
                ix=[i for i,z in enumerate(s) if z==name]; pr=model(x[ix].to(device),banks[name],input_valid_mask=v[ix].to(device)).cpu().numpy(); out.setdefault(name,[[],[],[]]); out[name][0].append(y[ix,-1].numpy()); out[name][1].append(pr); out[name][2].append(starts[ix].numpy())
    with torch.no_grad():
        for n,p in model.named_parameters(): p.copy_(raw[n])
    from btransform_unified_v1.r2 import variance_weighted_r2
    vals={s:float(variance_weighted_r2(np.concatenate(a),np.concatenate(b))) for s,(a,b,_starts) in out.items()}
    if audit_path is not None:
        payload={}
        for s,(ys,ps,starts) in out.items(): payload[f"target/{s}"]=np.concatenate(ys);payload[f"prediction/{s}"]=np.concatenate(ps);payload[f"window_start/{s}"]=np.concatenate(starts)
        tmp=audit_path.with_suffix(".tmp.npz");np.savez_compressed(tmp,**payload);tmp.replace(audit_path)
    return {"per_session":vals,"equal_session_mean":float(np.mean(list(vals.values())))}
def _rng_state(): return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def _set_rng(s):
 random.setstate(s["python"]);np.random.set_state(s["numpy"]);torch.set_rng_state(s["torch"].cpu())
 if s["cuda"] is not None: torch.cuda.set_rng_state_all([state.cpu() for state in s["cuda"]])
def _configure(model): model.temporal.set_attention_backend("local"); model._frontend_owner.unit_dropout_p=0.0 # explicit paired keep is passed by loop
def _install(model,arm,banks,calib): model.install_calibration(banks,{} if arm=="Z_NONE" else calib)
def _keep(seed,epoch,bid,device):
    from btransform_unified_v1.model import unit_dropout_seed,whole_unit_dropout
    g=torch.Generator(device="cpu");g.manual_seed(unit_dropout_seed(seed,epoch,bid)); return whole_unit_dropout(torch.ones((1,64),dtype=torch.bool),p=UNIT_DROPOUT,generator=g).to(device)
def _source_cache(a): return a.dest.parent.parent/"source_prepare_cache"

def prepare(a):
    if (a.dest/"prepare.json").exists() and not a.resume: raise FileExistsError("destination already prepared; use --resume")
    m=materialize_sources(cache=_source_cache(a)); atomic_json(a.dest/"prepare.json",{**metadata(a),"status":"PREPARED","source_sessions":list(m["sources"]),"basis_fit_scope":m["basis_scope"],"source_cache_files":sorted(p.name for p in _source_cache(a).glob("*"))})
def preflight(a):
    """Real, small numerical assertions; source-only materialization intentionally opens no target."""
    m=materialize_sources(cache=_source_cache(a)); device=torch.device(a.device); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    models={arm:CrossSessionM1Decoder(arm,seed=a.seed).to(device) for arm in ARMS}
    for arm,z in models.items(): _configure(z); _install(z,arm,m["banks"],m["calib"])
    try:
        models[Z_NONE].install_calibration(m["banks"],m["calib"])
    except RuntimeError:
        z_calibration_forbidden=True
    else:
        z_calibration_forbidden=False
        raise RuntimeError("Z accepted calibration IO")
    counts={k:sum(p.numel() for p in v.parameters()) for k,v in models.items()}; ratio=max(counts.values())/min(counts.values()); shared=common_init_hash(models[Z_NONE],models[B_ACTIVITY_ONLY],models[D_JOINT])
    if ratio>1.05 or state_hash(models[B_ACTIVITY_ONLY],"encoder.")!=state_hash(models[D_JOINT],"encoder."): raise RuntimeError("parameter/init parity failure")
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    x,y,s,v,_starts=next(iter(loader(m["train"],a.seed,shuffle=False))); name=s[0]; keep=_keep(a.seed,1,0,device); expected_padding=any(st < 99 for _,st in m["train"].window_indices); actual_padding=any(bool((~IndexedWindows(m["train"])[i][3]).any()) for i in range(len(m["train"])))
    if expected_padding != actual_padding: raise RuntimeError("raw start to valid-mask mapping drift")
    report={"counts":counts,"parameter_ratio":ratio,"init":{"B3S_bytes_equal":True,"shared_ZBD_hash":shared},"Z_calibration_forbidden":z_calibration_forbidden,"valid_mask_mapping_checked":True}
    for arm,z in models.items():
        z.train(); z.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            p=z(x.to(device),m["banks"][name],dropout_keep=keep,input_valid_mask=v.to(device))
            loss=nn.functional.mse_loss(p.float(),y[:,-1].to(device))
        loss.backward(); report[arm]={"finite":bool(torch.isfinite(p).all() and torch.isfinite(loss)),"has_grad":bool(any(q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum()>0 for q in z.parameters()))}
        if not all(report[arm].values()): raise RuntimeError(f"{arm} forward/gradient preflight failed")
    with torch.inference_mode():
        b=models[B_ACTIVITY_ONLY]; d=models[D_JOINT]; b.eval(); d.eval(); baseb=b(x.to(device),m["banks"][name],input_valid_mask=v.to(device)); based=d(x.to(device),m["banks"][name],input_valid_mask=v.to(device)); oldb=b._carrier[name].clone(); oldd=d._carrier[name].clone(); b._carrier[name].add_(1.); d._carrier[name].add_(1.); pertb=b(x.to(device),m["banks"][name],input_valid_mask=v.to(device)); pertd=d(x.to(device),m["banks"][name],input_valid_mask=v.to(device)); b._carrier[name].copy_(oldb); d._carrier[name].copy_(oldd)
    report["carrier"]={"B_max_delta":float((baseb-pertb).abs().max()),"D_max_delta":float((based-pertd).abs().max())}
    if report["carrier"]["B_max_delta"] != 0. or report["carrier"]["D_max_delta"] <= 0.: raise RuntimeError("carrier arm isolation failure")
    atomic_json(a.dest/"preflight.json",{**metadata(a),"status":"PASSED","checks":report,"source_only":True,"rawtrial_fullwindow_disjoint":True,"source_evidence":source_evidence(m),"source_code_sha256":{str(Path(__file__)):sha(Path(__file__)),str(Path(__file__).with_name("m1_data.py")):sha(Path(__file__).with_name("m1_data.py")),str(ROOT/"src/btransform_unified_v2/cross_session_m1_model.py"):sha(ROOT/"src/btransform_unified_v2/cross_session_m1_model.py")}})
def _require_ready(a):
    p=a.dest/"preflight.json"
    if not p.is_file(): raise RuntimeError("run --stage preflight successfully first")
    row=json.loads(p.read_text());m=materialize_sources(cache=_source_cache(a))
    if row.get("status")!="PASSED" or any(row.get(k)!=metadata(a)[k] for k in ("split_id","sources","targets","arm","seed","data_contract")) or row.get("source_evidence")!=source_evidence(m): raise RuntimeError("preflight binding differs from current chronological source split")
def train(a):
    _require_ready(a); from btransform_unified_v1.ema import DecoderEMA; from tfpd_exploration.src.m2_dual_track_v1 import training as factory
    m=materialize_sources(cache=_source_cache(a));device=torch.device(a.device);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);model=CrossSessionM1Decoder(a.arm,seed=a.seed).to(device);_configure(model);_install(model,a.arm,m["banks"],m["calib"]);initialization_hash=state_hash(model);ema=DecoderEMA(model,EMA_DECAY);opt=factory.build_optimizer(model.named_parameters(),lr=LR,weight_decay=.01);start_epoch=1;step=0;best=None;resume_state=None
    cp=a.dest/"resume_latest.pt"
    if cp.exists() and not a.resume: raise FileExistsError("checkpoint exists; use --resume")
    if a.resume and not cp.exists(): raise FileNotFoundError("--resume requires resume_latest.pt")
    if a.resume and cp.exists():
        q=torch.load(cp,map_location=device,weights_only=False);resume_state=q
        if q.get("meta") != metadata(a) or q.get("source_evidence") != source_evidence(m) or q.get("initialization_hash") != initialization_hash or not isinstance(q.get("batch_order_sha256"),str): raise RuntimeError("resume checkpoint belongs to another chronological split/source roster")
        model.load_state_dict(q["model"],strict=True);ema.load_state_dict(q["ema"]);opt.load_state_dict(q["optimizer"]);_set_rng(q["rng"]);start_epoch=q["epoch"]+1;step=q["step"];best=q["best"]
    evidence=source_evidence(m); batch_order_sha256=resume_state["batch_order_sha256"] if resume_state else hashlib.sha256(b"m1_chronological_batch_chain_v1").hexdigest(); started=time.monotonic(); batches=len(FixedSessionSampler(IndexedWindows(m["train"]),a.seed,1,True)); total=EPOCHS*batches
    for ep in range(start_epoch,EPOCHS+1):
        model.train()
        for bid,(x,y,s,v,_starts) in enumerate(loader(m["train"],a.seed,ep,True)):
            if len(set(s))!=1: raise RuntimeError("mixed-session batch")
            step+=1; opt.zero_grad(set_to_none=True); warm=batches; frac=min(1.,step/warm); cosine=.1+.9*.5*(1+np.cos(np.pi*max(0,step-warm)/max(1,total-warm))); lr=LR*(frac if step<=warm else cosine)
            for g in opt.param_groups:g["lr"]=lr
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                pred=model(x.to(device),m["banks"][s[0]],dropout_keep=_keep(a.seed,ep,bid,device),input_valid_mask=v.to(device))
                loss=nn.functional.mse_loss(pred.float(),y[:,-1].to(device))
            if not bool(torch.isfinite(loss) and torch.isfinite(pred).all()): raise FloatingPointError("nonfinite train forward/loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError("nonfinite train gradient")
            batch_order_sha256=chain_digest(batch_order_sha256,f"{ep}|{bid}|{s[0]}|".encode()+_keep(a.seed,ep,bid,torch.device("cpu")).numpy().tobytes()); nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();ema.update_after_step(model)
            if step%50==0: atomic_json(a.dest/"heartbeat.json",{"status":"TRAINING","epoch":ep,"step":step,"elapsed_seconds":time.monotonic()-started,"loss":float(loss.detach().cpu())});print(f"epoch={ep} step={step} elapsed={time.monotonic()-started:.1f}s loss={float(loss):.6g}",flush=True)
        val=_ema_eval(model,ema,m["val"],m["banks"],device);row={"epoch":ep,"step":step,"source_val":val};atomic_json(a.dest/f"ema_epoch_{ep:03d}.json",row)
        atomic_torch(a.dest/f"ema_epoch_{ep:03d}.pt",{"ema":ema.state_dict(),"epoch":ep,"step":step,"meta":metadata(a)})
        if best is None or val["equal_session_mean"]>best["source_val"]["equal_session_mean"]: best=row;atomic_torch(a.dest/"selected_ema.pt",{"ema":ema.state_dict(),"epoch":ep,"meta":metadata(a)})
        atomic_torch(cp,{"model":model.state_dict(),"ema":ema.state_dict(),"optimizer":opt.state_dict(),"epoch":ep,"step":step,"best":best,"rng":_rng_state(),"meta":metadata(a),"source_evidence":evidence,"initialization_hash":initialization_hash,"batch_order_sha256":batch_order_sha256})
    hashes={p.name:sha(p) for p in [cp,a.dest/"selected_ema.pt"]}; sources={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name("m1_data.py"),ROOT/"src/btransform_unified_v2/cross_session_m1_model.py",ROOT.parent/"streaming_calibration_exp/src/data/falcon_datamodule.py",ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/data.py",ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py"]};atomic_json(a.dest/"train_receipt.json",{**metadata(a),"status":"COMPLETED","selected":best,"epochs":EPOCHS,"steps":step,"initialization_hash":initialization_hash,"batch_order_sha256":batch_order_sha256,"checkpoint_sha256":hashes,"source_code_sha256":sources,"actual_source_arrays":evidence,"source_arrays_scope":m["basis_scope"]})
def main():
    p=argparse.ArgumentParser();p.add_argument("--dest",type=Path,required=True);p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--stage",choices=("prepare","preflight","train"),required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--resume",action="store_true");a=p.parse_args()
    if a.seed != 42: raise ValueError("protocol primary seed is 42")
    {"prepare":prepare,"preflight":preflight,"train":train}[a.stage](a)
if __name__=="__main__":main()
