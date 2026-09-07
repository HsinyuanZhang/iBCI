#!/usr/bin/env python3
"""Formal DANDI 688 RIFT training, resumable only from an identical formal run.

The only evaluation surface is the 6-session validation split.  This file never
constructs or scores a formal-test split.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; WS = ROOT.parent
for p in (ROOT/'src', WS/'sua_exploration', WS/'streaming_calibration_exp', WS/'btransform_unified_v1'/'src', WS):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import torch
from torch import nn
from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1 import plan as v1_plan
from tfpd_exploration.src.m2_dual_track_v1.training import build_optimizer
from mc_maze.dandi688_sparse_event_t4_v1 import plan
from mc_maze.dandi688_sparse_event_t4_v1.production import prepare_source_surface
from btransform_unified_v2 import RiftDecoder, RiftStreamDecoder

SEED=42; W=50; BATCH=32; EPOCHS=12; AVG=(8,9,10,11)
SCHEMA='dandi688_rift_v2'

def ah(x: Any) -> str: return array_sha256(np.ascontiguousarray(np.asarray(x)))
def obj_hash(x: Any) -> str: return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def file_hash(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(value,tmp); os.replace(tmp,path)
def heartbeat(dest: Path, *, status: str, event: str, **payload: Any) -> None:
    dump(dest/'heartbeat.json', {'status':status,'event':event,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),**payload})
def load_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text())
def bank(s: str, r: dict[str, Any]) -> TaskBank:
    e0_hash=ah(r['e0'])
    return TaskBank(s,r['e0'],r['carrier'],r['mask'],np.zeros((1,W,len(r['mask'])),np.float32),np.zeros((1,2),np.float32),np.zeros(1,np.int64),{'shape':tuple(r['e0'].shape),'trial_count':30,'budget':30,'estimator':'frozen B3S post_pool(M30) + profile-M10','array_sha256':e0_hash,'e0_sha256':e0_hash,'carrier_sha256':ah(r['carrier'])})
def xy(r: dict[str, Any], starts: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    return (np.stack([r['neural'][s:s+W] for s in starts]).astype(np.float32), np.stack([r['behavior'][s+W-1] for s in starts]).astype(np.float32), np.ones((len(starts),W),bool))
def windows(r, starts): return xy(r, starts)
def batches(rows: dict[str,dict[str,Any]], epoch: int):
    names=sorted(s for s,r in rows.items() if r['split']=='train')
    seed=int.from_bytes(hashlib.sha256(f'{plan.ROUTE_NAME}:stage2-q50:{SEED}:{epoch}'.encode()).digest()[:8],'little')
    rng=np.random.Generator(np.random.PCG64(seed))
    for i in rng.permutation(len(names)):
        s=names[int(i)]; q=rows[s]['starts'][rng.permutation(len(rows[s]['starts']))]
        for o in range(0,len(q),BATCH): yield s,np.ascontiguousarray(q[o:o+BATCH])
def rng_state() -> dict[str, Any]:
    return {'python':random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(state: dict[str,Any]) -> None:
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None:
        if not torch.cuda.is_available(): raise RuntimeError('resume requires CUDA RNG state but CUDA is unavailable')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
def data_source_hashes() -> dict[str,str]:
    """Only code/config that materializes the immutable prepared data bundle."""
    files=[WS/'sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/plan.py', WS/'sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/production.py', WS/'sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/core.py', WS/'sua_exploration/mc_maze/dandi688_cp_film_v1/runner.py', WS/'sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json', WS/plan.TEACHER_RELATIVE]
    return {str(f.relative_to(WS)):file_hash(f) for f in files}
def rows_hash(rows: dict[str,dict[str,Any]]) -> dict[str,dict[str,str]]:
    return {s:{'neural':ah(r['neural']),'behavior':ah(r['behavior']),'starts':ah(r['starts']),'e0':ah(r['e0']),'carrier':ah(r['carrier']),'mask':ah(r['mask'])} for s,r in sorted(rows.items())}
def make_contract(rows: dict[str,dict[str,Any]], n: int, updates: int, device: str, prepared:dict[str,Any]) -> dict[str,Any]:
    model_config={'task':'dandi688_co','units':n,'e0_dim':50,'carrier_dim':4,'out_dim':2,'context_bins':W,'bias_mode':'recency','seed':SEED,'proj_dim':16,'attention_backend':'local'}
    model_files=[ROOT/'src/btransform_unified_v2/model.py',ROOT/'src/btransform_unified_v2/temporal.py',ROOT/'src/btransform_unified_v2/streaming.py',ROOT/'src/btransform_unified_v2/config.py',WS/'btransform_unified_v1/src/btransform_unified_v1/model.py',WS/'btransform_unified_v1/src/btransform_unified_v1/plan.py']
    return {'schema':SCHEMA,'status':'FORMAL','seed':SEED,'manifest_sha256':plan.MANIFEST_SHA256,'split_counts':{'train':27,'val':6},'formal_test_used':False,'window_bins':W,'activity_support_m30':30,'carrier_candidate_m10':10,'query_start_trial':50,'unit_geometry':{'nmax':n,'padding':'right','padded_unit_mask':False},'updates_per_epoch':updates,'epochs':EPOCHS,'fixed_average_epochs_zero_based':list(AVG),'optimizer':{'name':'AdamW','lr_peak':v1_plan.LR_PEAK,'weight_decay':v1_plan.WEIGHT_DECAY,'grad_clip':v1_plan.GRAD_CLIP,'unit_dropout':v1_plan.UNIT_DROPOUT},'baseline_comparability':{'status':'NEW_RIFT_BASELINE_NOT_ONLY_DECODER_MATCHED','fixed_data_sample_epochs':{'train_sessions':27,'validation_sessions':6,'epochs':EPOCHS,'batch':BATCH,'window_bins':W},'old_sparse_event_optimizer':{'name':'Adam','lr':plan.STAGE2_BASE_LEARNING_RATE},'new_rift_optimizer':{'name':'AdamW','lr_peak':v1_plan.LR_PEAK}},'model_config':model_config,'model_source_hashes':{str(f.relative_to(WS)):file_hash(f) for f in model_files},'data_source_hashes':prepared['data_source_hashes'],'prepared_contract_sha256':obj_hash(prepared),'target_cache_hashes':prepared['rows_hashes'],'runner_sha256':file_hash(Path(__file__))}
def assert_contract(dest: Path, contract: dict[str,Any]) -> str:
    path=dest/'data_contract.json'
    if not path.is_file(): raise RuntimeError('missing data_contract.json')
    saved=load_json(path)
    if saved != contract: raise RuntimeError('source/data/config contract mismatch; refusing incompatible destination')
    return obj_hash(contract)
def checkpoint_state(epoch:int, step:int, model, optimizer, contract_sha256:str) -> dict[str,Any]:
    return {'schema':SCHEMA+'_checkpoint','status':'FORMAL_PARTIAL','smoke':False,'epoch_zero_based':epoch,'next_epoch_zero_based':epoch+1,'global_step':step,'epochs':EPOCHS,'contract_sha256':contract_sha256,'raw_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state()}
def validate_resume(state: dict[str,Any], contract_sha256: str, dest:Path, updates_per_epoch: int) -> None:
    required={'schema':SCHEMA+'_checkpoint','status':'FORMAL_PARTIAL','smoke':False,'epochs':EPOCHS,'contract_sha256':contract_sha256}
    if any(state.get(k)!=v for k,v in required.items()): raise RuntimeError('resume checkpoint contract mismatch')
    epoch=state.get('epoch_zero_based'); nxt=state.get('next_epoch_zero_based')
    if not isinstance(epoch,int) or nxt != epoch+1 or not 0 <= epoch < EPOCHS-1: raise RuntimeError('resume checkpoint has invalid epoch')
    if not isinstance(state.get('global_step'),int) or state['global_step'] != (epoch+1)*updates_per_epoch: raise RuntimeError('resume checkpoint global step does not exactly match completed epochs')
    if not all(k in state for k in ('raw_state_dict','optimizer','rng')): raise RuntimeError('resume checkpoint missing model/optimizer/RNG state')
    if (dest/'train_receipt.json').exists() or (dest/'average_e8_e11.pt').exists(): raise RuntimeError('destination is already complete; resume refused')
def average_checkpoint(dest:Path, contract_sha256:str) -> Path:
    states=[]
    for epoch in AVG:
        path=dest/f'epoch_{epoch:03d}.pt'
        if not path.is_file(): raise RuntimeError(f'missing fixed averaging checkpoint {path.name}')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('contract_sha256') != contract_sha256 or state.get('epoch_zero_based') != epoch or state.get('smoke'): raise RuntimeError('averaging checkpoint contract mismatch')
        states.append(state['raw_state_dict'])
    avg={k:torch.stack([s[k].double() for s in states]).mean(0).to(states[0][k].dtype) if states[0][k].is_floating_point() else states[0][k] for k in states[0]}
    path=dest/'average_e8_e11.pt'; atomic_torch_save(path, {'schema':SCHEMA+'_average','status':'FORMAL','average_epochs_zero_based':list(AVG),'state_dict':avg,'contract_sha256':contract_sha256}); return path
def build_rows(device:torch.device) -> dict[str,dict[str,Any]]:
    surface=prepare_source_surface(WS,signal_view='sua',reliability_mask=(True,)*4)
    from mc_maze.dandi688_cp_film_v1.runner import _prepare_student
    student=_prepare_student(WS,SEED,device); n=max(r.record.neural.shape[1] for r in surface.sessions.values()); rows={}
    for s,row in surface.sessions.items():
        k=row.record.neural.shape[1]; cal=torch.from_numpy(row.record.calib_trials.astype(np.float32)).unsqueeze(0).to(device); c=torch.from_numpy(row.profile_m10.astype(np.float32)).unsqueeze(0).to(device)
        with torch.inference_mode(): e=student.id_encoder.post_pool(torch.cat((student.id_encoder.pre_pool(cal.permute(0,1,3,2)).mean(1),c),-1))[0].cpu().numpy()
        neural=np.zeros((len(row.record.neural),n),np.float32); neural[:,:k]=row.record.neural; e0=np.zeros((n,50),np.float32); e0[:k]=e; carrier=np.zeros((n,4),np.float32); carrier[:k]=row.profile_m10; mask=np.zeros(n,bool); mask[:k]=1
        rows[s]={'split':row.split,'neural':neural,'behavior':row.record.behavior.astype(np.float32),'starts':np.asarray(row.q50_starts,np.int64),'e0':e0,'carrier':carrier,'mask':mask}
    del student; return rows
def prepared_rows(cache:Path, device:torch.device) -> tuple[dict[str,dict[str,Any]],dict[str,Any]]:
    """Load or atomically create the data-only 27+6 cache; never includes test data."""
    meta_path=cache/'prepared_contract.json'
    if meta_path.is_file():
        meta=load_json(meta_path)
        required={'schema':SCHEMA+'_prepared','manifest_sha256':plan.MANIFEST_SHA256,'split_counts':{'train':27,'val':6},'formal_test_used':False,'data_source_hashes':data_source_hashes()}
        if any(meta.get(k)!=v for k,v in required.items()): raise RuntimeError('prepared cache source/manifest contract mismatch')
        rows={}
        for s,info in meta.get('sessions',{}).items():
            path=cache/'sessions'/f'{s}.npz'
            if not path.is_file(): raise RuntimeError(f'prepared cache missing session {s}')
            z=np.load(path); rows[s]={'split':info['split'],'neural':z['neural'],'behavior':z['behavior'],'starts':z['starts'],'e0':z['e0'],'carrier':z['carrier'],'mask':z['mask']}
        if rows_hash(rows)!=meta.get('rows_hashes') or len(rows)!=33: raise RuntimeError('prepared cache array hash/count mismatch')
        return rows,meta
    if cache.exists(): raise RuntimeError('prepared cache path exists without immutable prepared_contract.json')
    rows=build_rows(device); meta={'schema':SCHEMA+'_prepared','status':'FROZEN','manifest_sha256':plan.MANIFEST_SHA256,'split_counts':{'train':27,'val':6},'formal_test_used':False,'data_source_hashes':data_source_hashes(),'rows_hashes':rows_hash(rows),'sessions':{s:{'split':r['split']} for s,r in sorted(rows.items())}}
    tmp=cache.parent/(cache.name+'.tmp'); tmp.mkdir(parents=True,exist_ok=False); (tmp/'sessions').mkdir()
    for s,r in rows.items(): np.savez_compressed(tmp/'sessions'/f'{s}.npz',neural=r['neural'],behavior=r['behavior'],starts=r['starts'],e0=r['e0'],carrier=r['carrier'],mask=r['mask'])
    dump(tmp/'prepared_contract.json',meta)
    size=sum(p.stat().st_size for p in tmp.rglob('*') if p.is_file())
    if size >= 2*1024**3: raise RuntimeError(f'prepared cache exceeds 2 GiB ({size} bytes)')
    os.replace(tmp,cache); return rows,meta
def build_model(n:int, device:torch.device):
    m=RiftDecoder({'task':'dandi688_co','units':n,'e0_dim':50,'carrier_dim':4,'out_dim':2},context_bins=W,bias_mode='recency',seed=SEED,proj_dim=16).to(device); m.temporal.set_attention_backend('local'); return m
def assert_full_stream_parity(model, sample:dict[str,Any], device:torch.device) -> dict[str,Any]:
    """Real one-window oracle: batch decoder output equals stream-step output."""
    model.eval(); starts=sample['starts'][:1]; x,_,v=xy(sample,starts); tx=torch.from_numpy(x).to(device); tv=torch.from_numpy(v).to(device); b=bank('preflight-parity',sample)
    with torch.inference_mode():
        offline=model.forward_scores(tx,b,input_valid_mask=tv)[0]
        stream=RiftStreamDecoder(model); online=torch.stack([stream.stream_step(tx[:,t],b,['preflight-parity'],valid_mask=tv[:,t])[0] for t in range(W)])
    diff=float((offline-online).abs().max());
    if not math.isfinite(diff) or diff > 1e-5: raise RuntimeError(f'batch/stream parity failed: max_abs={diff}')
    model.train(); return {'status':'PASSED','max_abs':diff,'bins':W}
def score(dest:Path, rows:dict[str,dict[str,Any]], model, device:torch.device, contract_sha256:str) -> None:
    receipt_path=dest/'train_receipt.json'; avg_path=dest/'average_e8_e11.pt'
    if (dest/'score_receipt.json').exists(): raise RuntimeError('score receipt already exists; refusing to overwrite a scored contract')
    if not receipt_path.is_file() or not avg_path.is_file(): raise RuntimeError('score requires complete formal train receipt and average checkpoint')
    receipt=load_json(receipt_path); required={'schema':SCHEMA+'_train_receipt','status':'TRAIN_COMPLETED','epochs':EPOCHS,'updates_per_epoch':28076,'contract_sha256':contract_sha256,'formal_test_used':False}
    if any(receipt.get(k)!=v for k,v in required.items()): raise RuntimeError('full formal train receipt contract mismatch')
    avg=torch.load(avg_path,map_location=device,weights_only=False)
    if avg.get('schema')!=SCHEMA+'_average' or avg.get('status')!='FORMAL' or avg.get('contract_sha256')!=contract_sha256 or avg.get('average_epochs_zero_based')!=list(AVG): raise RuntimeError('formal average checkpoint contract mismatch')
    model.load_state_dict(avg['state_dict']); model.eval(); out={}
    with torch.inference_mode():
        for s,r in sorted(rows.items()):
            if r['split']!='val': continue
            pred=[]; target=[]; coords=[]
            for off in range(0,len(r['starts']),BATCH):
                starts=r['starts'][off:off+BATCH]; x,y,v=xy(r,starts); p=model(torch.from_numpy(x).to(device),bank(s,r),input_valid_mask=torch.from_numpy(v).to(device)).cpu().numpy()
                pred.append(p); target.append(y); coords.append(starts)
            pred=np.concatenate(pred).astype(np.float32); target=np.concatenate(target).astype(np.float32); coords=np.concatenate(coords).astype(np.int64)
            if len(pred)!=len(r['starts']) or not (np.isfinite(pred).all() and np.isfinite(target).all()): raise RuntimeError(f'incomplete/nonfinite validation output: {s}')
            path=dest/'scores'/f'{s}.npz'; path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,prediction=pred,target=target,query_start=coords)
            total=float(np.square(target-target.mean(axis=0,keepdims=True)).sum()); r2=float(1.0-np.square(target-pred).sum()/total) if total>0 else float('nan')
            out[s]={'n_query':int(len(coords)),'prediction_sha256':ah(pred),'target_sha256':ah(target),'query_start_sha256':ah(coords),'mse':float(np.mean((pred-target)**2)),'r2':r2}
    if len(out)!=6: raise RuntimeError(f'expected 6 validation sessions, got {len(out)}')
    means=[x['mse'] for x in out.values()]; r2s=[x['r2'] for x in out.values()]
    if not all(math.isfinite(x) for x in means+r2s): raise RuntimeError('nonfinite validation score')
    all_pred=[]; all_target=[]
    for s in sorted(out):
        z=np.load(dest/'scores'/f'{s}.npz'); all_pred.append(z['prediction']); all_target.append(z['target'])
    pp=np.concatenate(all_pred); tt=np.concatenate(all_target); denom=float(np.square(tt-tt.mean(axis=0,keepdims=True)).sum()); pooled=float(1-np.square(tt-pp).sum()/denom)
    dump(dest/'score_receipt.json',{'schema':SCHEMA+'_score_receipt','status':'SCORED','contract_sha256':contract_sha256,'average_checkpoint_sha256':file_hash(avg_path),'sessions':out,'equal_session_mean_mse':float(np.mean(means)),'equal_session_mean_r2':float(np.mean(r2s)),'pooled_r2':pooled,'formal_test_used':False})
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--dest',type=Path,required=True); p.add_argument('--prepared-cache',type=Path,default=WS/'btransform_unified_v2/results/rift_v1/dandi688_prepared_cache_v1'); p.add_argument('--stage',choices=('preflight','train','score'),default='train'); p.add_argument('--device',default='cuda:0'); p.add_argument('--epochs',type=int,default=EPOCHS); p.add_argument('--resume',type=Path); p.add_argument('--max-updates-smoke',type=int); p.add_argument('--cpu-threads',type=int,default=2); a=p.parse_args()
    if a.stage=='train' and (a.max_updates_smoke is not None or a.epochs!=EPOCHS): raise ValueError('this formal-only runner refuses smoke or non-12-epoch training')
    if a.stage!='train' and a.resume: raise ValueError('--resume is valid only for train')
    if a.resume and a.resume.parent.resolve()!=a.dest.resolve(): raise RuntimeError('resume checkpoint must belong to --dest')
    torch.set_num_threads(a.cpu_threads); d=torch.device(a.device); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if not a.resume and a.stage=='train' and a.dest.exists() and any(a.dest.iterdir()): raise FileExistsError('new training destination must be empty')
    rows,prepared=prepared_rows(a.prepared_cache,d); n=next(iter(rows.values()))['neural'].shape[1]; updates=sum((len(r['starts'])+BATCH-1)//BATCH for r in rows.values() if r['split']=='train')
    if updates!=28076: raise RuntimeError(f'Q50 update count drift: {updates}')
    contract=make_contract(rows,n,updates,str(d),prepared); cs=obj_hash(contract)
    if a.stage=='train' and not a.resume:
        a.dest.mkdir(parents=True,exist_ok=True); dump(a.dest/'data_contract.json',contract)
    elif a.stage=='preflight' and not (a.dest/'data_contract.json').exists():
        if a.dest.exists() and any(a.dest.iterdir()): raise FileExistsError('new preflight destination must be empty')
        a.dest.mkdir(parents=True,exist_ok=True); dump(a.dest/'data_contract.json',contract)
    else: assert_contract(a.dest,contract)
    model=build_model(n,d); first=next(r for r in rows.values() if r['split']=='train')
    if a.stage=='preflight':
        heartbeat(a.dest,status='PREFLIGHT',event='gradient_start',updates_per_epoch=updates,contract_sha256=cs)
        x,y,v=xy(first,first['starts'][:BATCH]); loss=nn.functional.mse_loss(model(torch.from_numpy(x).to(d),bank('preflight',first),input_valid_mask=torch.from_numpy(v).to(d)),torch.from_numpy(y).to(d))
        if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite preflight loss')
        loss.backward(); grad_norm=float(nn.utils.clip_grad_norm_(model.parameters(),v1_plan.GRAD_CLIP,error_if_nonfinite=True)); gradients_finite=all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in model.parameters())
        if not gradients_finite: raise FloatingPointError('nonfinite preflight gradient')
        parity=assert_full_stream_parity(model,first,d); report={'status':'PASSED','gradient_loss':float(loss.detach()),'gradient_finite':gradients_finite,'gradient_norm':grad_norm,'stream_parity':parity,'updates_per_epoch':updates,'contract_sha256':cs}
        dump(a.dest/'preflight.json',report); heartbeat(a.dest,status='PREFLIGHT_COMPLETED',event='complete',**{key:value for key,value in report.items() if key!='status'}); return
    if a.stage=='score': score(a.dest,rows,model,d,cs); return
    opt=build_optimizer(model.named_parameters(),lr=v1_plan.LR_PEAK,weight_decay=v1_plan.WEIGHT_DECAY); begin=0; step=0
    if a.resume:
        state=torch.load(a.resume,map_location=d,weights_only=False); validate_resume(state,cs,a.dest,updates); model.load_state_dict(state['raw_state_dict']); opt.load_state_dict(state['optimizer']); restore_rng(state['rng']); begin=state['next_epoch_zero_based']; step=state['global_step']
    heartbeat(a.dest,status='TRAINING',event='start',epoch=begin,global_step=step,updates_per_epoch=updates,contract_sha256=cs)
    for e in range(begin,EPOCHS):
        done=0; losses=[]; epoch_started=time.monotonic()
        for bi,(s,starts) in enumerate(batches(rows,e)):
            x,y,v=xy(rows[s],starts); opt.zero_grad(); keep=whole_unit_dropout(torch.from_numpy(rows[s]['mask']),p=v1_plan.UNIT_DROPOUT,generator=torch.Generator().manual_seed(unit_dropout_seed(SEED,e,bi))).to(d)
            loss=nn.functional.mse_loss(model(torch.from_numpy(x).to(d),bank(s,rows[s]),dropout_keep=keep,input_valid_mask=torch.from_numpy(v).to(d)),torch.from_numpy(y).to(d))
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f'nonfinite loss epoch={e} batch={bi}')
            loss.backward(); grad_norm=float(nn.utils.clip_grad_norm_(model.parameters(),v1_plan.GRAD_CLIP,error_if_nonfinite=True)); opt.step(); step+=1; done+=1; losses.append(float(loss.detach()))
            if step==1 or step%100==0: heartbeat(a.dest,status='TRAINING',event='step',epoch=e,batch=bi,global_step=step,loss=losses[-1],grad_norm=grad_norm)
        if done!=updates or step!=(e+1)*updates: raise RuntimeError('short or inconsistent formal epoch')
        state=checkpoint_state(e,step,model,opt,cs); atomic_torch_save(a.dest/f'epoch_{e:03d}.pt',state)
        heartbeat(a.dest,status='TRAINING',event='epoch_checkpoint',epoch=e,global_step=step,mean_loss=float(np.mean(losses)),elapsed_seconds=time.monotonic()-epoch_started,checkpoint_sha256=file_hash(a.dest/f'epoch_{e:03d}.pt'))
    avg=average_checkpoint(a.dest,cs); receipt={'schema':SCHEMA+'_train_receipt','status':'TRAIN_COMPLETED','epochs':EPOCHS,'global_step':step,'updates_per_epoch':updates,'average_epochs_zero_based':list(AVG),'average_checkpoint_sha256':file_hash(avg),'contract_sha256':cs,'formal_test_used':False}; dump(a.dest/'train_receipt.json',receipt); heartbeat(a.dest,status='TRAIN_COMPLETED',event='complete',**receipt)
if __name__=='__main__': main()
