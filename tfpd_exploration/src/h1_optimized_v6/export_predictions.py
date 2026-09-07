"""Read-only strict native export for completed V6 query-only formal result."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v4.export_predictions import native_predictions,assert_ids,r2
from .model import make_matched_pair
from .ema import V6EMA
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def masks(cache):
 out={}
 for split in ('train','minival'):
  out[split]={}
  for n,row in cache[split].items():
   a=row['bank']['unit_mask'].detach().cpu().contiguous().numpy();h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.tobytes());out[split][n]=h.hexdigest()
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',default='paired_v6_recency_queryonly_12ep_v1');p.add_argument('--device',default='cpu');a=p.parse_args();root=ROOT/a.attempt;need=[root/'input_authority.json',root/'selection_protocol_freeze.json',root/'selection_freeze.json',root/'final.json']
 if any(not x.is_file() for x in need):raise FileNotFoundError('completed V6 receipt missing')
 out=root/'independent_score_export'
 if out.exists():raise FileExistsError(out)
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);freeze=json.loads((root/'selection_freeze.json').read_text());out.mkdir();device=torch.device(a.device)
 here=Path(__file__).resolve();src=here.parents[1];files={'v6_model':here.with_name('model.py'),'v6_ema':here.with_name('ema.py'),'v6_train':here.with_name('paired_train.py'),'v6_export':here,'v4_model':src/'h1_optimized_v4/model.py','v4_export_helpers':src/'h1_optimized_v4/export_predictions.py','v2_cache':src/'h1_optimized_v2/cache.py','v2_score':src/'h1_optimized_v2/score.py','query_v4_core':src/'two_mainlines_long_v1/current_query_v4/core.py'}
 manifest={'schema':'h1_v6_independent_native_score_export_v1','attempt':a.attempt,'committed_input_bindings':{x.name:sha(x) for x in need}|{'source_cache_authority.json':sha(ROOT/'source_cache_authority.json')},'input_authority':auth,'operator_source_sha256':{k:sha(v) for k,v in files.items()},'per_session_bank_unit_mask_sha256':masks(cache),'arms':{'t':{}},'surfaces':{'selection':{'n_bins':2908},'complete':{'n_bins':20325}},'write_semantics':'completed files only; no atomic write claim'}
 model=make_matched_pair()[1].to(device);paths={'selected':Path(freeze['selected']['checkpoint']),'epoch12':root/'t_v6_epoch_012.pt'}
 for label,path in paths.items():
  data=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(data['model'],strict=True)
  if int(model.frontend_contract_version)!=4 or int(model.temporal.temporal_contract_version)!=4:raise RuntimeError('V6 contract buffers drift')
  ema=V6EMA(model,.9995);ema.load_checkpoint_state(data['ema']);state=out/f't_{label}_plain_ema_model_state.pt';meta=out/f't_{label}_ema_metadata.pt';rec={'checkpoint':str(path),'checkpoint_sha256':sha(path),'frontend_contract_version':4,'temporal_contract_version':4,'ema_shadow_key_count':len(data['ema']['shadow']),'immutable_temporal_buffer_names':sorted(data['ema']['immutable_temporal_buffers']),'surfaces':{}}
  def export(view):
   torch.save({k:v.detach().cpu().clone() for k,v in view.state_dict().items()},state)
   for mode in ('selection','complete'):
    arr=native_predictions(view,cache,device,mode);assert_ids(cache,arr,mode);npz=out/f't_{label}_{mode}_native.npz';np.savez_compressed(npz,**arr);rec['surfaces'][mode]={'npz':str(npz),'sha256':sha(npz),'n_bins':len(arr['bin_timestep']),'session_count':13,'r2_concat_float64':r2(arr['prediction_native_velocity'],arr['target_native_velocity'])}
  ema.score_with_ema(model,export);torch.save({'schema':'v6_ema_metadata_v1','checkpoint':str(path),'checkpoint_sha256':sha(path),'ema':ema.checkpoint_state()},meta);rec.update({'plain_ema_model_state':str(state),'plain_ema_model_state_sha256':sha(state),'ema_metadata':str(meta),'ema_metadata_sha256':sha(meta)});manifest['arms']['t'][label]=rec
 target=out/'manifest.json';target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({'manifest':str(target),'sha256':sha(target)}))
if __name__=='__main__':main()
