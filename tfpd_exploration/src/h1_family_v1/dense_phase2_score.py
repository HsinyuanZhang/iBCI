"""Gated, post-completion-only scoring for the paired dense Phase-2 run.

This module never trains, selects, promotes, or resumes.  Its only mutable
outputs are a fresh evaluator directory after a completed, hash-bound run.
"""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile, time
from pathlib import Path
import numpy as np

ARMS=("flat","route"); MODES=("RAW","EMA"); LIMIT=2400; MEMORY=22<<30
ROOT=Path(__file__).resolve().parents[2]
SOURCE208_REFERENCE_SHA='f3ac9a999fef786ca435e0ff6eefcf679b82e3b3439a535cd66ae6e65c80e614'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):
 value=json.loads(Path(path).read_text())
 if not isinstance(value,dict): raise RuntimeError("JSON object required")
 return value
def atomic(value,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,mode='w',delete=False) as h:
  tmp=Path(h.name);json.dump(value,h,sort_keys=True,indent=2,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
 os.replace(tmp,path)
def atomic_npz(arrays,path):
 with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npz',delete=False) as h: tmp=Path(h.name)
 try: np.savez_compressed(tmp,**arrays);os.replace(tmp,path)
 finally: tmp.unlink(missing_ok=True)

def _bind(files,path,expected=None):
 path=Path(path).resolve(); actual=sha(path)
 if expected is not None and actual!=expected: raise RuntimeError('bound input SHA drift: '+str(path))
 if files.setdefault(str(path),actual)!=actual: raise RuntimeError('conflicting input binding')
 return path

def _validate_dense_receipt(dense, authority_sha):
 from . import dense_phase2_pair as dense_run
 if (dense.get('schema')!='h1_dense_phase2_formal_train_v1' or dense.get('status')!='TRAINING_COMPLETE_NO_SCORE_OR_SELECTION' or dense.get('pre')!=dense.get('post') or dense.get('ema_updates')!={'flat':1462,'route':1462} or len(dense.get('updates',[]))!=1462 or len(dense.get('checkpoints',[]))!=2): raise RuntimeError('completed dense receipt contract drift')
 b=dense['pre']
 if b.get('protocol')!=dense_run.PROTOCOL or b.get('protocol_sha256')!=dense_run.digest(dense_run.PROTOCOL): raise RuntimeError('dense fixed fresh selected-EMA protocol drift')
 if b.get('formal_epoch_identities',{}).keys()!={'1','2'} or set(dense.get('identities',{}))!={'1','2'}: raise RuntimeError('dense formal sampler/dropout identity topology drift')
 for epoch in ('1','2'):
  actual,formal=dense['identities'][epoch],b['formal_epoch_identities'][epoch]
  if actual.get('sampler_sha256')!=formal.get('sampler_sha256') or actual.get('keep_sha256')!=formal.get('keep_sha256') or actual.get('batches')!=731 or actual.get('windows')!=23212: raise RuntimeError('dense formal sampler/dropout identity drift')
 if not all(x.get('epoch')==e and x.get('identities')==dense['identities'][str(e)] for e,x in enumerate(dense['checkpoints'],1)): raise RuntimeError('dense epoch checkpoint identity drift')
 rows=dense['updates']
 if ([(x.get('epoch'),x.get('batch')) for x in rows]!=[(e,i) for e in (1,2) for i in range(731)] or any(not isinstance(x.get('rows'),int) or not 1<=x['rows']<=32 or not isinstance(x.get('session'),str) or x.get('endpoint_valid') is not True or not np.isfinite(x.get('valid_fraction',np.nan)) or not 0<x['valid_fraction']<=1 or set(x.get('loss',{}))!=set(ARMS) or any(not np.isfinite(v) for v in x['loss'].values()) for x in rows) or any(sum(x['rows'] for x in rows[(e-1)*731:e*731])!=23212 for e in (1,2))):raise RuntimeError('dense complete 2x731 source/loss topology drift')
 if not isinstance(authority_sha,str) or len(authority_sha)!=64: raise RuntimeError('dense input authority SHA required')
 return b

def collect_bindings(formal,dense_root,dense_receipt_sha,cold_summary,cold_summary_sha,output):
 """Read-only audit; execute before torch/cache/model imports."""
 if any(not Path(p).is_absolute() for p in (formal,dense_root,cold_summary,output)): raise RuntimeError('absolute paths required')
 files={}
 here=Path(__file__).resolve()
 selected_receipt=ROOT/'results/family_runtime_v1/h1_selected_source208_diagnostic_v1/receipt.json'
 for p in (here,here.with_name('dense_phase2_pair.py'),ROOT/'src/family_runtime_v1/complete_h1_family_source.py',ROOT/'src/family_runtime_v1/diagnose_h1_selected_source208.py',ROOT/'src/family_runtime_v1/diagnose_h1_endpoint_raw.py',ROOT/'src/h1_family_v1/familyformal_split_train.py',ROOT/'src/h1_family_v1/cold_phase_summary.py',dense_root/'receipt.json',dense_root/'input_authority.json',cold_summary,selected_receipt): _bind(files,p)
 _bind(files,selected_receipt,SOURCE208_REFERENCE_SHA)
 _bind(files,dense_root/'receipt.json',dense_receipt_sha); _bind(files,cold_summary,cold_summary_sha)
 authority_sha=sha(dense_root/'input_authority.json'); dense=read(dense_root/'receipt.json'); b=_validate_dense_receipt(dense,authority_sha)
 for item in dense['checkpoints']:
  _bind(files,item['path'],item['sha256'])
 authority=read(dense_root/'input_authority.json')
 if dense.get('pre',{}).get('formal')!=str(formal) or dense['initial']!=authority.get('initial') or authority.get('bindings')!=b or authority.get('mode')!='formal2': raise RuntimeError('dense root/formal/initial authority drift')
 from . import dense_phase2_pair as dense_run
 if dense_run.collect_bindings(formal,dense_root,smoke_receipt=Path(b['smoke_receipt']['path']))!=b:raise RuntimeError('fresh dense source/smoke/complete closure drift')
 from tfpd_exploration.src.family_runtime_v1.complete_h1_family_source import artifact_audit,code_source_audit
 artifact,source=artifact_audit(formal),code_source_audit(formal)
 if b.get('formal_artifact')!=artifact or b.get('formal_source')!=source: raise RuntimeError('dense current formal artifact/source closure drift')
 for file_map in (b['files'],artifact['files'],source['files']):
  for path,value in file_map.items(): _bind(files,path,value)
 cold=read(cold_summary)
 if cold.get('schema')!='h1_cold_history_phase_archive_summary_v2' or cold.get('status')!='COMPLETE_ARCHIVE_ONLY_NO_SELECTION': raise RuntimeError('matched cold CONTROL summary schema drift')
 selected_ref=read(selected_receipt)
 if selected_ref.get('status')!='COMPLETE_READ_ONLY_DESCRIPTIVE': raise RuntimeError('selected source208 reference receipt drift')
 from . import cold_phase_summary as summary
 baselines={};source208={}
 for arm in ARMS:
  cold_receipt=Path(cold_summary).parent/'h1_cold_phase2x2_v1'/arm/'receipt.json'
  cold_sha=cold.get('inputs_sha256_pre',{}).get(str(cold_receipt))
  if not isinstance(cold_sha,str) or len(cold_sha)!=64:raise RuntimeError('cold worker must have explicit summary SHA binding')
  _bind(files,cold_receipt,cold_sha); cold_worker=summary.verify_worker(cold_receipt.parent.parent,arm,files)
  selected_export=str(formal/'exports'/f'{arm}_selected_plain_ema.pt')
  if cold_worker.get('initial_state_digest') is None or cold_worker.get('pre',{}).get('files',{}).get(selected_export)!=b['files'].get(selected_export): raise RuntimeError('dense/cold CONTROL selected-export byte binding drift')
  if any(cold_worker['pre'].get(k)!=b[k] for k in ('formal_artifact','formal_source','formal_epoch_identities')):raise RuntimeError('cold/dense source and sampler authority mismatch')
  item=cold.get('workers',{}).get(arm,{}).get('CONTROL',{}).get('EMA')
  if not item: item=cold.get('archives',{}).get(arm,{}).get('CONTROL_EMA')
  # The checked-in summary is an analysis document; accept its explicit archive
  # locator only, never infer a comparison from an unbound result.
  if not isinstance(item,dict) or 'archive' not in item: raise RuntimeError('bound matched cold CONTROL EMA archive required')
  archive=item['archive']; path=_bind(files,archive['path'],archive['sha256']);baselines[arm]=str(path)
  source_item=selected_ref.get('archives',{}).get(arm,{})
  source208[arm]=str(_bind(files,source_item.get('path'),source_item.get('sha256')))
 if {p:sha(p) for p in files}!=files:raise RuntimeError('input changed during dense score preflight')
 return {'formal':str(formal),'dense_root':str(dense_root),'dense_receipt_sha256':dense_receipt_sha,'dense_authority_sha256':authority_sha,'cold_summary':str(cold_summary),'cold_summary_sha256':cold_summary_sha,'cold_control_ema_archives':baselines,'selected_source208_archives':source208,'dense':b,'dense_receipt_identities':dense['identities'],'artifact':artifact,'source':source,'files':files,'output':str(output)}

def preflight(formal,dense_root,dense_receipt_sha,cold_summary,cold_summary_sha,output,authorization,authorization_sha):
 if output.exists(): raise FileExistsError(output)
 if os.environ.get('H1_DENSE_PHASE2_SCORE_GO')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or sha(authorization)!=authorization_sha: raise RuntimeError('explicit GO/GPU0/authorization gate')
 binding=collect_bindings(formal,dense_root,dense_receipt_sha,cold_summary,cold_summary_sha,output)
 if read(authorization).get('bindings')!=binding: raise RuntimeError('external evaluator authorization drift')
 return binding

def state_digest(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()):
  a=t.detach().cpu().contiguous().numpy();h.update(n.encode());h.update(str(a.dtype).encode());h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
 return h.hexdigest()
class Raw:
 def score_with_ema(self,model,callback): return callback(model)
def validate_dense_payload(payload,binding):
 from . import dense_phase2_pair as dense
 if (payload.get('schema')!='h1_dense_phase2_end_epoch_v1' or payload.get('epoch')!=2 or payload.get('global_step')!=1462 or payload.get('protocol_sha256')!=dense.digest(dense.PROTOCOL) or payload.get('authority_sha256')!=binding['dense_authority_sha256'] or payload.get('identities')!=binding['dense_receipt_identities']['2']): raise RuntimeError('dense stage2 payload identity drift')
 if set(payload.get('models',{}))!=set(ARMS) or set(payload.get('optimizers',{}))!=set(ARMS) or set(payload.get('emas',{}))!=set(ARMS): raise RuntimeError('dense stage2 payload arm topology drift')
 for arm in ARMS:
  raw,ema=payload['models'][arm],payload['emas'][arm]
  import torch
  if not raw or any(not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or not bool(torch.isfinite(t).all()) for t in raw.values()) or ema.get('n_updates')!=1462 or ema.get('decay')!=.9995 or set(ema.get('shadow',{}))!=set(raw) or any(not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or t.shape!=raw[n].shape or not bool(torch.isfinite(t).all()) for n,t in ema['shadow'].items()): raise RuntimeError('dense RAW/EMA checkpoint tensor contract drift')
 return payload

def check_archive(arrays,baseline,*,source208):
 key='start' if source208 else 'end';n=208 if source208 else 20325
 if set(arrays)!={'prediction','target','session_id',key} or arrays['prediction'].shape!=(n,7) or arrays['target'].shape!=(n,7) or arrays['prediction'].dtype!=np.float64 or arrays['target'].dtype!=np.float64 or arrays[key].shape!=(n,) or arrays['session_id'].shape!=(n,) or arrays[key].dtype!=np.int64 or arrays['session_id'].dtype.kind!='U' or not np.isfinite(arrays['prediction']).all() or not np.isfinite(arrays['target']).all() or any(not np.array_equal(arrays[k],baseline[k]) for k in ('target','session_id',key)): raise RuntimeError('native archive metadata/FP64 contract drift')
 from . import cold_phase_summary as summary
 summary._validate_session_geometry(arrays['session_id'],arrays[key],source208=source208)
def segments(arrays,metric):
 out={}
 for label,mask,n in (('cold_history_lt_699',arrays['end']<699,8702),('full_w700_ge_699',arrays['end']>=699,11623)):
  if int(mask.sum())!=n: raise RuntimeError('fixed segment cardinality drift')
  out[label]=metric({k:v[mask] for k,v in arrays.items()})
 return out

def run(formal,dense_root,dense_receipt_sha,cold_summary,cold_summary_sha,output,authorization,authorization_sha):
 started=time.monotonic()
 binding=preflight(formal,dense_root,dense_receipt_sha,cold_summary,cold_summary_sha,output,authorization,authorization_sha)
 import torch
 if not torch.cuda.is_available() or torch.cuda.current_device()!=0: raise RuntimeError('GPU0 unavailable')
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.reset_peak_memory_stats()
 from tfpd_exploration.src.h1_optimized_v2.cache import CACHE,ROOT as CACHE_ROOT,validate_authority
 from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
 from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
 from tfpd_exploration.src.h1_family_v1.familyformal_split_train import score_complete_cached_ema
 from tfpd_exploration.src.family_runtime_v1 import diagnose_h1_selected_source208 as selected
 from tfpd_exploration.src.family_runtime_v1.complete_h1_family_source import check_metadata,metric,validate_archive
 from tfpd_exploration.src.family_runtime_v1.diagnose_h1_endpoint_raw import GuardedModel
 from . import cold_phase_summary as summary
 def guard():
  torch.cuda.synchronize()
  if time.monotonic()-started>LIMIT or torch.cuda.max_memory_allocated()>MEMORY: raise RuntimeError('score resource bound')
 guard()
 cache=torch.load(CACHE,map_location='cpu',weights_only=False);authority=read(CACHE_ROOT/'source_cache_authority.json');validate_authority(cache,authority);fixed=selected._require_ids();selected.validate_manifest_against_cache(cache,authority,fixed)
 def load_bound(path,source208=False):
  if sha(path)!=binding['files'][path]:raise RuntimeError('baseline changed before archive load')
  return summary.load(path,source208=source208)
 baselines={a:load_bound(binding['cold_control_ema_archives'][a]) for a in ARMS}
 source_baselines={a:load_bound(binding['selected_source208_archives'][a],True) for a in ARMS}
 for a in ARMS: check_metadata(baselines[a],cache)
 cold=read(cold_summary)
 for a in ARMS:
  reported=cold['workers'][a]['CONTROL']['EMA']
  if summary.analysis(baselines[a])!={k:v for k,v in reported.items() if k!='archive'}:raise RuntimeError('recomputed cold CONTROL analysis differs from frozen summary')
 for key in ('target','session_id','end'):
  if not np.array_equal(baselines['flat'][key],baselines['route'][key]): raise RuntimeError('cold CONTROL metadata disagreement')
 ck=Path(dense_root)/'dense_stage_epoch02.pt'
 if sha(ck)!=binding['files'].get(str(ck.resolve())): raise RuntimeError('bound dense checkpoint changed before load')
 payload=validate_dense_payload(torch.load(ck,map_location='cpu',weights_only=False),binding)
 output.mkdir(parents=True);atomic({'bindings':binding},output/'input_authority.json');results={};dev=torch.device('cuda:0')
 guard()
 for arm,index in zip(ARMS,(0,1)):
  pair=make_v2_unscaled_dot_localbalanced_pair(seed=42);model=pair[index].to(dev);del pair;model.load_state_dict(payload['models'][arm],strict=True);before=state_digest(model.state_dict());ema=DecoderEMA(model,decay=.9995);ema.load_checkpoint_state(payload['emas'][arm]);ema.shadow={k:v.to(dev) for k,v in ema.shadow.items()};archives={};scores={}
  for mode in MODES:
   scorer=Raw() if mode=='RAW' else ema
   def guarded(candidate):
    last=[-1]
    def check(calls,rows):
     guard()
     if (calls==0 or calls%128==0) and last[0]!=calls:
      atomic({'status':'SCORING','arm':arm,'mode':mode,'calls':calls,'rows':rows,'elapsed_seconds':time.monotonic()-started},output/'live.json');last[0]=calls
    return GuardedModel(candidate,check)
   e1,a1=scorer.score_with_ema(model,lambda candidate:selected.evaluate_source208(guarded(candidate),cache,fixed,dev,lambda row,d: __import__('tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal',fromlist=['H1Bank']).H1Bank(*[row['bank'][k].to(d) for k in ('E0','T','unit_mask')])))
   c=scorer.score_with_ema(model,lambda candidate:score_complete_cached_ema(model=guarded(candidate),ema=Raw(),cache=cache,device=dev));a2={k:c.pop('_'+k) for k in ('prediction','target','session_id','end')};check_archive(a2,baselines[arm],source208=False);check_archive(a1,source_baselines[arm],source208=True)
   summary._check_e1(e1,summary.metric(a1));summary._check_e2(c,summary.metric(a2))
   if state_digest(model.state_dict())!=before: raise RuntimeError('RAW state changed during scoring')
   paths={};
   for label,arrays in (('source208',a1),('complete',a2)):
    p=output/f'{arm}_dense_e2_{mode.lower()}_{label}_float64.npz';atomic_npz(arrays,p);paths[label]={'path':str(p),'sha256':sha(p),'metric':summary.metric(arrays)}
   candidate_analysis,control_analysis=summary.analysis(a2),summary.analysis(baselines[arm])
   archives[mode]=paths;scores[mode]={'source208':e1,'complete':paths['complete']['metric'],'analysis':candidate_analysis,'cold_control_ema_analysis':control_analysis,'minus_cold_control_ema':summary._delta(candidate_analysis,control_analysis)};guard()
  results[arm]={'raw_state_digest':before,'archives':archives,'scores':scores,'primary_ema_minus_cold_control':scores['EMA']['complete']['r2_concat_float64']-metric(baselines[arm])['r2_concat_float64'],'raw_secondary_minus_cold_control':scores['RAW']['complete']['r2_concat_float64']-metric(baselines[arm])['r2_concat_float64']}
 post=collect_bindings(formal,dense_root,dense_receipt_sha,cold_summary,cold_summary_sha,output)
 if post!=binding: raise RuntimeError('fresh post authority drift')
 for arm in results.values():
  for mode in MODES:
   for item in arm['archives'][mode].values():
    if sha(item['path'])!=item['sha256']: raise RuntimeError('output archive mutation before receipt')
 guard()
 receipt={'schema':'h1_dense_phase2_postcompleted_fixed_e2_score_v1','status':'COMPLETE_NO_SELECTION_OR_PROMOTION','pre':binding,'post':post,'arms':results,'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'parameter_updates':0,'no_selection_or_promotion':True}
 atomic(receipt,output/'receipt.json');return receipt

def main(argv=None):
 p=argparse.ArgumentParser();
 for x in ('formal','dense-root','dense-receipt-sha256','cold-summary','cold-summary-sha256','output','authorization','authorization-sha256'): p.add_argument('--'+x,required=True)
 a=p.parse_args(argv);return run(Path(a.formal),Path(a.dense_root),a.dense_receipt_sha256,Path(a.cold_summary),a.cold_summary_sha256,Path(a.output),Path(a.authorization),a.authorization_sha256)
if __name__=='__main__': main()
