"""Read-only native-unit export for a completed V5 query-only formal run."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.score import CHUNK, _windows
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .model import make_matched_pair


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def state_sha(model):
 h=hashlib.sha256()
 for n,t in sorted(model.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def hashes():
 here=Path(__file__).resolve();src=here.parents[1]
 files={'v5_model':here.with_name('model.py'),'v5_paired_train':here.with_name('paired_train.py'),'v5_export':here,'v2_cache':src/'h1_optimized_v2/cache.py','v2_score':src/'h1_optimized_v2/score.py','ema':src/'h1_temporal_decoder_quick_product_v1/ema.py','h1_temporal_root':src/'two_mainlines_long_v1/decoder/h1_temporal.py','query_v2_core':src/'two_mainlines_long_v1/current_query_v2/core.py','query_v2_streaming':src/'two_mainlines_long_v1/current_query_v2/streaming.py','query_v3_core':src/'two_mainlines_long_v1/current_query_v3/core.py'}
 if any(not p.is_file() for p in files.values()):raise FileNotFoundError('missing operator source')
 return {k:sha(v) for k,v in files.items()}
def masks(cache):
 ans={}
 for split in ('train','minival'):
  ans[split]={}
  for name,row in cache[split].items():
   x=row['bank']['unit_mask'].detach().cpu().contiguous().numpy();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.tobytes());ans[split][name]=h.hexdigest()
 return ans
def arrays(model,cache,device,mode):
 out={k:[] for k in ('prediction_native_velocity','target_native_velocity','session_id','bin_timestep')};model.eval()
 with torch.inference_mode():
  for name,row in cache['minival'].items():
   ends=row['query_starts']+699 if mode=='selection' else np.flatnonzero(row['eval_mask']);b=H1Bank(*[row['bank'][q].to(device) for q in ('E0','T','unit_mask')]);p=[]
   for off in range(0,len(ends),CHUNK):
    x=torch.as_tensor(_windows(row['neural'],ends[off:off+CHUNK]),device=device);p.append((model.forward_last(x,b)/20).cpu().numpy().astype(np.float32,copy=False))
   out['prediction_native_velocity'].append(np.concatenate(p));out['target_native_velocity'].append(np.asarray(row['velocity'][ends],np.float32));out['session_id'].append(np.full(len(ends),name,dtype='U32'));out['bin_timestep'].append(np.asarray(ends,np.int64))
 return {k:np.concatenate(v) for k,v in out.items()}
def check_ids(cache,a,mode):
 ss=[];tt=[]
 for name,row in cache['minival'].items():
  ends=row['query_starts']+699 if mode=='selection' else np.flatnonzero(row['eval_mask']);ss.append(np.full(len(ends),name,dtype='U32'));tt.append(np.asarray(ends,np.int64))
 s,t=np.concatenate(ss),np.concatenate(tt)
 if not np.array_equal(s,a['session_id']) or not np.array_equal(t,a['bin_timestep']):raise RuntimeError('source identifiers drift')
 pairs=np.char.add(np.char.add(s,':'),t.astype(str)); expected=2908 if mode=='selection' else 20325
 if len(s)!=expected or len(np.unique(s))!=13 or len(np.unique(pairs))!=len(pairs):raise RuntimeError('invalid scored endpoint cardinality')
def r2(p,y):
 p,y=p.astype(np.float64),y.astype(np.float64);return float(1-np.square(p-y).sum()/np.square(y-y.mean(0,keepdims=True)).sum())
def contract(model,ema):
 expected={n for n,p in model.named_parameters() if p.requires_grad}
 if expected!=set(ema['shadow']):raise RuntimeError('EMA shadow key set does not equal trainable parameter set')
 if int(model.frontend_contract_version.item())!=4 or int(model.temporal.temporal_contract_version.item())!=3:raise RuntimeError('strict V5 contract buffers are not 4/3')
def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',default='paired_v5_logage_queryonly_12ep_v1');p.add_argument('--device',default='cpu');args=p.parse_args();root=ROOT/args.attempt
 needed=[root/'input_authority.json',root/'selection_freeze.json',root/'final.json'];
 if any(not x.is_file() for x in needed):raise FileNotFoundError('completed V5 final/freeze/input required')
 output=root/'independent_score_export'
 if output.exists():raise FileExistsError(f'refusing overwrite {output}')
 cache=build_or_load();authority=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,authority);freeze=json.loads((root/'selection_freeze.json').read_text());device=torch.device(args.device);output.mkdir()
 full,model=make_matched_pair();control=json.loads((root/'input_authority.json').read_text())['v4_control'];factory_full=state_sha(full);expected=control['sealed_v4_full_initial_state_sha256']
 if factory_full!=expected:raise RuntimeError('fresh V5 source FULL does not equal sealed V4 initial FULL')
 manifest={'schema':'h1_v5_independent_native_score_export_v1','attempt':args.attempt,'committed_input_bindings':{'input_authority.json':sha(root/'input_authority.json'),'selection_freeze.json':sha(root/'selection_freeze.json'),'final.json':sha(root/'final.json'),'source_cache_authority.json':sha(ROOT/'source_cache_authority.json')},'input_authority':authority,'operator_source_sha256':hashes(),'per_session_bank_unit_mask_sha256':masks(cache),'v4_control':{**control,'fresh_v5_factory_full_state_sha256':factory_full,'fresh_full_equals_sealed_v4_initial':True},'write_semantics':'trainer files are treated as completed committed files; exporter makes no atomic-write claim','surfaces':{'selection':{'n_bins':2908},'complete':{'n_bins':20325}},'arms':{'t':{}}};model=model.to(device)
 paths={'selected':Path(freeze['selected']['checkpoint']),'epoch12':root/'t_v5_epoch_012.pt'}
 for label,path in paths.items():
  payload=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'],strict=True);contract(model,payload['ema']);ema=DecoderEMA(model,decay=float(payload['ema']['decay']));ema.load_checkpoint_state(payload['ema']);state=output/f't_{label}_plain_ema_model_state.pt';meta=output/f't_{label}_ema_metadata.pt';record={'checkpoint':str(path),'checkpoint_sha256':sha(path),'ema_shadow_key_count':len(payload['ema']['shadow']),'frontend_contract_version':int(model.frontend_contract_version.item()),'temporal_contract_version':int(model.temporal.temporal_contract_version.item()),'surfaces':{}}
  def export(view):
   torch.save({k:v.detach().cpu().clone() for k,v in view.state_dict().items()},state)
   for mode in ('selection','complete'):
    a=arrays(view,cache,device,mode);check_ids(cache,a,mode);archive=output/f't_{label}_{mode}_native.npz';np.savez_compressed(archive,**a);record['surfaces'][mode]={'npz':str(archive),'sha256':sha(archive),'n_bins':len(a['bin_timestep']),'session_count':13,'r2_concat_float64':r2(a['prediction_native_velocity'],a['target_native_velocity'])}
  ema.score_with_ema(model,export);torch.save({'schema':'decoder_ema_metadata_v1','arm':'t','checkpoint':str(path),'checkpoint_sha256':sha(path),'ema':ema.checkpoint_state()},meta);record.update({'plain_ema_model_state':str(state),'plain_ema_model_state_sha256':sha(state),'ema_metadata':str(meta),'ema_metadata_sha256':sha(meta)});manifest['arms']['t'][label]=record
 target=output/'manifest.json';target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({'manifest':str(target),'sha256':sha(target)}))
if __name__=='__main__':main()
