#!/usr/bin/env python3
"""Build or host-verify one fresh sealed H1 R300 ACTIVITY_ONLY/NONE package.
No stage pushes, registers, submits, or invokes Docker automatically.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, shutil, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path('/home/xinyuan/Work_host/SPINT'); DEST=Path(__file__).resolve().parent
V1=ROOT/'btransform_unified_v1/src'; V2=ROOT/'btransform_unified_v2/src'; sys.path[:0]=[str(V2),str(V1),str(DEST),str(ROOT)]
ARMS=('ACTIVITY_ONLY','NONE'); EPOCHS=32; TAGS=27
CANONICAL_ROSTER_PAYLOAD=ROOT/'tfpd_exploration/submissions/evalai_h1_c2_cal1_b2_ort_v1/artifacts/h1_c2_cal1_b2_s42_ema_e18_L200.pkl'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(x):
 a=np.ascontiguousarray(x); return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def need(x,m):
 if not x: raise RuntimeError(m)
def canonical_h1_tag_sessions():
 """Return the immutable official 13-HI + 14-HO Falcon-hash tag/session map."""
 from falcon_challenge.config import FalconConfig, FalconTask
 from btransform_unified_v1.c2_protocol import HELDOUT_SESSION_TO_FALCON_KEY
 from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
 config=FalconConfig(task=FalconTask.h1)
 pairs=[(f'sub-HumanPitt-held-in-minival_{session}',str(session)) for session in HELDIN_SESSIONS]
 pairs += [(f'sub-HumanPitt-held-out-calib_{session}',str(session)) for session,_key in HELDOUT_SESSION_TO_FALCON_KEY]
 out={config.hash_dataset(Path(stem).stem):session for stem,session in pairs}
 need(len(pairs)==TAGS and len(out)==TAGS,'official H1 inventory/hash collision')
 return out
def build(a):
 from btransform_unified_v1.ema import DecoderEMA
 from btransform_unified_v2.model import RiftDecoder
 from h1_rift_falcon_decoder import PAYLOAD_SCHEMA
 dest=a.dest.resolve(); need(not dest.exists(),'--dest must be fresh'); banks=a.banks.resolve(); run=a.run_dir.resolve()
 receipt=json.loads((banks/'receipt.json').read_text()); meta=json.loads((run/'run_meta.json').read_text()); train=json.loads((run/'train_receipt.json').read_text()); score=json.loads((run/'ho_m3_selection.json').read_text())
 need(receipt.get('schema')=='h1_signed_state_r300_ablation_banks_v1' and receipt.get('status')=='BUILT' and receipt.get('arm')==a.arm and receipt.get('tag_count')==TAGS,'sealed bank receipt mismatch')
 from btransform_unified_v1 import h1_config
 need(tuple(receipt.get('source_sessions',()))==tuple(h1_config.H1_ALL_SESSIONS),'exact 27 official roster source-session drift')
 need(receipt.get('banks_27_sha256')==sha(banks/'banks_27.npz'),'bank NPZ file SHA mismatch')
 need(meta.get('schema')=='rift_h1_signed_state_r300_ablation_v1' and meta.get('status')=='FORMAL' and meta.get('arm')==a.arm and meta.get('epochs')==EPOCHS,'formal run metadata mismatch')
 need(train.get('status')=='COMPLETED' and train.get('epochs')==EPOCHS and train.get('selected_checkpoint_sha256'),'complete selected receipt required')
 curve=score.get('curve'); need(isinstance(curve,list) and len(curve)==EPOCHS and {int(x.get('epoch',-1)) for x in curve}==set(range(1,EPOCHS+1)),'all 32 grouped-seven score sidecars required')
 need(score.get('status')=='HO_M3_DEVELOPMENT_SELECTION','selection status mismatch')
 from btransform_unified_v1.c2_protocol import HO_SELECTION_METRIC, select_epoch
 for x in curve: need(all(k in x for k in (HO_SELECTION_METRIC,'worst_session_r2','session_std_population','epoch_zero_based','checkpoint_sha256')),'selection row incomplete')
 selected=select_epoch(curve)
 need(score.get('selected')==selected,'C2 select_epoch drift')
 ep=int(selected.get('epoch',-1)); ckpt=run/f'epoch_{ep:03d}.pt'; need(1<=ep<=EPOCHS and train.get('selected_epoch')==ep and sha(ckpt)==train['selected_checkpoint_sha256'],'selected checkpoint binding mismatch')
 for row in curve:
  checkpoint=run/f"epoch_{int(row['epoch']):03d}.pt"; sidecar=checkpoint.with_suffix('.pt.binding.json'); need(checkpoint.is_file() and sidecar.is_file() and row.get('checkpoint_sha256')==sha(checkpoint),'score/checkpoint sidecar binding mismatch')
  binding=json.loads(sidecar.read_text()); need(binding.get('arm')==a.arm and binding.get('epoch')==int(row['epoch']) and binding.get('checkpoint_sha256')==sha(checkpoint) and binding.get('run_meta_sha256')==sha(run/'run_meta.json') and binding.get('banks_receipt_sha256')==receipt.get('banks_27_sha256') and binding.get('source_manifest_sha256')==meta.get('source_manifest_sha256'),'checkpoint provenance binding mismatch')
 state=torch.load(ckpt,map_location='cpu',weights_only=False); need(state.get('smoke') is False and state.get('epoch')==ep,'formal checkpoint mismatch')
 model=RiftDecoder('h1',context_bins=300,bias_mode='recency',seed=42,proj_dim=16);model.load_state_dict(state['raw_state_dict'],strict=True);ema=DecoderEMA(model,decay=.9995);ema.load_state_dict(state['ema']);ema.apply_to(model);model.eval()
 from h1_rift_falcon_decoder import CPUUnpickler
 with open(CANONICAL_ROSTER_PAYLOAD,'rb') as handle: canonical=CPUUnpickler(handle).load()['bank_by_dataset_tag']
 # The sealed main payload establishes only the tag set; it contains no session fields.
 canonical_sessions=canonical_h1_tag_sessions()
 need(len(canonical_sessions)==TAGS and set(canonical_sessions)==set(canonical),'official fixed tag/session roster drift')
 need(set(receipt.get('tags',{}))==set(canonical_sessions),'receipt exact canonical 27-tag roster drift')
 need(all(str(receipt['tags'][tag].get('session'))==canonical_sessions[tag] for tag in canonical_sessions),'canonical tag/session mapping drift')
 sealed={}
 with np.load(banks/'banks_27.npz') as z:
  need(set(z.files)=={f'{kind}/{tag}' for tag in canonical for kind in ('E0','T')},'exact NPZ tag roster drift')
  for tag in canonical:
   row=receipt['tags'][tag]; raw_e=z[f'E0/{tag}']; raw_t=z[f'T/{tag}']; need(raw_e.dtype==np.float32 and raw_t.dtype==np.float32 and np.isfinite(raw_e).all() and np.isfinite(raw_t).all(),'raw bank dtype/finiteness drift')
   e=np.ascontiguousarray(raw_e);t=np.ascontiguousarray(raw_t);need(e.shape==(176,700) and t.shape==(176,4) and ash(e)==row['E0_sha256'] and ash(t)==row['T_sha256'] and np.array_equal(t,np.zeros_like(t)),'bank array binding drift')
   if a.arm=='NONE':need(np.array_equal(e,np.zeros_like(e)),'NONE E0 drift')
   sealed[tag]={'E0':e,'T':t,'unit_mask':np.ones(176,bool),'session':canonical_sessions[tag],'e0_sha256':ash(e),'t_sha256':ash(t)}
 dest.mkdir(parents=True);
 # Fresh standalone context includes its exact runtime and entry files.
 for name in ('h1_rift_falcon_decoder.py','decode.py','Dockerfile','README.md'):
  shutil.copy2(DEST/name,dest/name)
 payload={'schema':PAYLOAD_SCHEMA,'task':'h1','arm':a.arm,'context_bins':300,'bias_mode':'recency','proj_dim':16,'behavior_scaling_factor':20.,'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'banks_receipt_sha256':sha(banks/'receipt.json'),'banks_npz_sha256':sha(banks/'banks_27.npz'),'bank_by_dataset_tag':sealed,'canonical_dataset_tags':sorted(canonical),'canonical_tag_to_session':canonical_sessions,'ema_state_dict':{k:v.detach().cpu().float().clone() for k,v in model.state_dict().items()},'selection':selected,'official_test_used':False,'binding':{'run_meta_sha256':sha(run/'run_meta.json'),'train_receipt_sha256':sha(run/'train_receipt.json'),'selection_sha256':sha(run/'ho_m3_selection.json'),'source_manifest_sha256':meta.get('source_manifest_sha256'),'ablation_implementation_sha256':meta.get('ablation_implementation_sha256'),'banks_27_sha256':receipt.get('banks_27_sha256'),'banks_receipt_sha256':sha(banks/'receipt.json')}}
 with (dest/'payload.pkl').open('wb') as f:pickle.dump(payload,f,protocol=4)
 (dest/'receipt.json').write_text(json.dumps({'status':'BUILT_NOT_HOST_VERIFIED','arm':a.arm,'payload_sha256':sha(dest/'payload.pkl'),'selection_epoch':ep,'official_test_used':False,'binding':{'run_meta_sha256':sha(run/'run_meta.json'),'train_receipt_sha256':sha(run/'train_receipt.json'),'selection_sha256':sha(run/'ho_m3_selection.json'),'source_manifest_sha256':meta.get('source_manifest_sha256'),'ablation_implementation_sha256':meta.get('ablation_implementation_sha256'),'banks_27_sha256':receipt.get('banks_27_sha256'),'banks_receipt_sha256':sha(banks/'receipt.json')}},indent=2)+'\n'); return {'payload':str(dest/'payload.pkl')}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--banks',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a),indent=2))
def _cache_row(runtime, row):
    cache=runtime.cached_temporal; need(cache is not None and runtime.last is not None,'cached state absent')
    return (runtime.raw4[row].clone(), runtime.last[row].clone(), *(x[row].clone() for x in cache.keys), *(x[row].clone() for x in cache.values), *(x[row].clone() for x in cache.lengths))
def _same_state(left,right): return len(left)==len(right) and all(torch.equal(a,b) for a,b in zip(left,right))
def _full_h1(model, bank, rows):
    sys.path.insert(0,str(ROOT/'btransform_unified_v2/scripts/rift_v1'))
    import h1_train as ht
    raw=np.ascontiguousarray(np.stack(rows),np.float32); x,valid=ht.endpoint_context(raw,np.asarray([len(raw)-1],np.int64),300)
    with torch.inference_mode(): return model(torch.from_numpy(x),bank,input_valid_mask=torch.from_numpy(valid)).numpy()/20.
def host_verify(package: Path, arm: str) -> dict:
    """Read-only real-public parity: HI B1/B8 plus mixed 2HI+2HO, no test data."""
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from btransform_unified_v1 import adapters
    from btransform_unified_v1.c2_protocol import HELDOUT_SESSION_TO_FALCON_KEY
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
    from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
    from btransform_unified_v2.model import RiftDecoder
    from h1_rift_falcon_decoder import H1RiftCachedFalconDecoder, _task_bank, load_payload, validate_payload
    sys.path.insert(0,str(ROOT/'btransform_unified_v2/scripts/rift_v1'))
    import h1_train as ht
    payload=load_payload(package/'payload.pkl'); validate_payload(payload)
    need(str(payload.get('arm'))==arm,'host arm/payload mismatch')
    config=FalconConfig(task=FalconTask.h1); cache=adapters._h1_source_cache(); endpoints=(0,1,79,299,319,359); report={}
    def check(label, stems, raws, inactive_at=None):
        tags=[config.hash_dataset(Path(stem).stem) for stem in stems]
        need(len(tags)==len(set(tags)) and all(t in payload['bank_by_dataset_tag'] for t in tags),'receipt-derived canonical tag missing')
        batch=len(tags); packed=H1RiftCachedFalconDecoder(config,str(package/'payload.pkl'),batch);packed.reset(stems)
        model=RiftDecoder('h1',context_bins=300,bias_mode='recency',seed=42,proj_dim=16);model.load_state_dict({k:torch.as_tensor(v) for k,v in payload['ema_state_dict'].items()},strict=True);model.eval()
        banks=[_task_bank(t,payload['bank_by_dataset_tag'][t]) for t in tags]; stream=CpuRiftRuntime(model,banks,[str(t) for t in tags],temporal_backend='cached')
        histories=[[] for _ in tags]; maximum=0.; inactive=False
        for step in range(360):
            observed=np.stack([x[step] for x in raws]); valid=torch.ones(batch,dtype=torch.bool); before=None
            if inactive_at is not None and step==inactive_at: valid[-1]=False; before=_cache_row(stream,batch-1); inactive=True
            got=packed.predict(observed[:int(valid.sum())]); want=stream.advance(torch.from_numpy(observed),valid_mask=valid)
            if before is not None: need(_same_state(before,_cache_row(stream,batch-1)),'inactive row advanced persistent state')
            for row in range(batch):
                if valid[row]: histories[row].append(observed[row].copy())
            if step in endpoints:
                for row,bank in enumerate(banks):
                    full=_full_h1(model,bank,histories[row]); maximum=max(maximum,float(np.max(np.abs(want[row:row+1].numpy()/20.-full))))
                need(np.allclose(got,want[:len(got)].numpy()/20.,atol=1e-5,rtol=1e-5),'packed/cache stream mismatch')
        packed.reset(stems); fresh=H1RiftCachedFalconDecoder(config,str(package/'payload.pkl'),batch);fresh.reset(stems)
        need(np.allclose(packed.predict(np.stack([x[0] for x in raws])),fresh.predict(np.stack([x[0] for x in raws])),atol=1e-6),'reset/fresh parity failed')
        need(maximum<=1e-5,f'{label} full-R300 endpoint gate failed')
        report[label]={'full_endpoints':list(endpoints),'max_full_abs':maximum,'inactive_resume_checked':inactive,'reset_fresh_checked':True,'roster_stems':stems}
    hi=list(HELDIN_SESSIONS); hi_stems=[f'sub-HumanPitt-held-in-minival_{x}' for x in hi]
    hi_raw=[np.ascontiguousarray(cache['minival'][x]['neural'][:360],np.float32) for x in hi]
    check('B1',hi_stems[:1],hi_raw[:1]); check('B8',hi_stems[:8],hi_raw[:8],inactive_at=120)
    # Exact held-out identifiers come from C2 mapping and train.py's public heldout-calib loader; no tag is guessed.
    ho_pairs=list(HELDOUT_SESSION_TO_FALCON_KEY[:2]); ho_stems=[f'sub-HumanPitt-held-out-calib_{session}' for session,_key in ho_pairs]
    ho_raw=[]
    for session,_key in ho_pairs:
        neural,_velocity,_change,_eval_mask=load_nwb(ht.b2.HO_DIR/f'sub-HumanPitt-held-out-calib_{session}.nwb',FalconTask.h1)
        ho_raw.append(np.ascontiguousarray(neural[:360],np.float32))
    check('MIXED_2HI_2HO',hi_stems[:2]+ho_stems,hi_raw[:2]+ho_raw,inactive_at=120)
    tag=config.hash_dataset(Path(hi_stems[0]).stem); bank=_task_bank(tag,payload['bank_by_dataset_tag'][tag]); model=RiftDecoder('h1',context_bins=300,bias_mode='recency',seed=42,proj_dim=16);model.load_state_dict({k:torch.as_tensor(v) for k,v in payload['ema_state_dict'].items()},strict=True);model.eval()
    probe=np.zeros((1,176),np.float32); rt=CpuRiftRuntime(model,[bank],['zero'],temporal_backend='cached'); need(np.allclose(rt.advance(torch.from_numpy(probe)).numpy()/20.,_full_h1(model,bank,[probe[0]]),atol=1e-5),'valid literal-zero probe/full mismatch')
    raw=np.ascontiguousarray(hi_raw[0][:40],np.float32); dec=H1RiftCachedFalconDecoder(config,str(package/'payload.pkl'),1);dec.reset([hi_stems[0]]); expected=np.stack([dec.predict(row[None]).copy() for row in raw]); smoke=package/'smoke_window.npz';np.savez(smoke,tag_stem=np.asarray(hi_stems[0]),window=raw,expected=expected)
    body={'status':'HOST_PACK_VERIFY_PASS','arm':arm,'payload_sha256':sha(package/'payload.pkl'),'B':report,'true_zero_valid_checked':True,'smoke_window':str(smoke),'fixture':'unaltered_first_40_raw_bins_with_per_bin_expected','official_test_used':False};(package/'host_verify.json').write_text(json.dumps(body,indent=2)+'\n');return body

def docker_context(package: Path, arm: str) -> dict:
    pkg=package/'artifacts/pkg'; need(not pkg.exists(),'refusing existing Docker package tree'); pkg.mkdir(parents=True)
    for src,name in ((V1/'btransform_unified_v1','btransform_unified_v1'),(V2/'btransform_unified_v2','btransform_unified_v2')): shutil.copytree(src,pkg/name,ignore=shutil.ignore_patterns('__pycache__','joint_m2_model.py','joint_m1_model.py','m2_mechanism_model.py'))
    (pkg/'btransform_unified_v2/__init__.py').write_text('from .model import RiftDecoder\nfrom .cpu_runtime import CpuRiftRuntime\n__all__ = ["RiftDecoder", "CpuRiftRuntime"]\n')
    need(all((package/name).is_file() for name in ('h1_rift_falcon_decoder.py','decode.py','Dockerfile','README.md')),'standalone runtime files missing from fresh package')
    shutil.copy2(package/'payload.pkl',package/'artifacts/payload.pkl')
    return {'status':'DOCKER_CONTEXT_PREPARED_NOT_BUILT','arm':arm,'payload_sha256':sha(package/'payload.pkl')}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('build','host','docker-context'),required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--banks',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args()
 if a.stage=='build': print(json.dumps(build(a),indent=2))
 elif a.stage=='host': print(json.dumps(host_verify(a.dest.resolve(),a.arm),indent=2))
 else: print(json.dumps(docker_context(a.dest.resolve(),a.arm),indent=2))

# The following explicit stages are intentionally separate from build. They use
# public minival neural data only and never invoke Docker or EvalAI implicitly.