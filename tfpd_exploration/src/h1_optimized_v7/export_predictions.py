"""Independent strict native exports for both completed V7 p=.30 arms."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v4.export_predictions import native_predictions,assert_ids,r2
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from .model import make_matched_pair
from .ema import V7EMA

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
def _validate_checkpoint(data,arm):
 if set(data.get('rng',{}))!={'torch','numpy','python','cuda'}:raise RuntimeError('V7 checkpoint RNG receipt drift')
 if arm=='full_v4':
  if data.get('operator_contract')!='v4_causal_full_window_no_query_temporal_contract' or 'temporal_contract_version' in data:raise RuntimeError('V7 FULL operator tag drift')
 else:
  if data.get('operator_contract')!='v6_recency_query_temporal_contract4' or data.get('temporal_contract_version')!=4:raise RuntimeError('V7 T operator tag drift')
  if 'immutable_temporal_buffers' not in data.get('ema',{}):raise RuntimeError('V7 T immutable buffers absent')
def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',default='v7_dropout30_paired12_v1');p.add_argument('--device',default='cpu');a=p.parse_args();root=ROOT/a.attempt
 need=[root/'input_authority.json',root/'selection_protocol_freeze.json',root/'selection_freeze.json',root/'final.json']
 if any(not x.is_file() for x in need):raise FileNotFoundError('completed V7 formal receipt missing')
 out=root/'independent_score_export'
 if out.exists():raise FileExistsError(out)
 cache=build_or_load();authority=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,authority);freeze=json.loads((root/'selection_freeze.json').read_text());out.mkdir();device=torch.device(a.device)
 here=Path(__file__).resolve();src=here.parents[1];files={'v7_model':here.with_name('model.py'),'v7_ema':here.with_name('ema.py'),'v7_train':here.with_name('paired_train.py'),'v7_export':here,'v6_model':src/'h1_optimized_v6/model.py','v6_ema':src/'h1_optimized_v6/ema.py','v4_model':src/'h1_optimized_v4/model.py','v4_export_helpers':src/'h1_optimized_v4/export_predictions.py','v2_cache':src/'h1_optimized_v2/cache.py','v2_score':src/'h1_optimized_v2/score.py','query_v4_core':src/'two_mainlines_long_v1/current_query_v4/core.py'}
 manifest={'schema':'h1_v7_dropout30_independent_native_score_export_v1','attempt':a.attempt,'committed_input_bindings':{x.name:sha(x) for x in need}|{'source_cache_authority.json':sha(ROOT/'source_cache_authority.json')},'input_authority':authority,'operator_source_sha256':{k:sha(v) for k,v in files.items()},'per_session_bank_unit_mask_sha256':masks(cache),'arms':{'full_v4':{},'t_v6':{}},'surfaces':{'selection':{'n_bins':2908},'complete':{'n_bins':20325}},'write_semantics':'completed files only; no atomic write claim'}
 fresh=dict(zip(('full_v4','t_v6'),(x.to(device) for x in make_matched_pair())))
 for arm,model in fresh.items():
  paths={'selected':Path(freeze['selected'][arm]['checkpoint']),'epoch12':root/f'{arm}_epoch_012.pt'}
  for label,path in paths.items():
   data=torch.load(path,map_location='cpu',weights_only=False);_validate_checkpoint(data,arm);model.load_state_dict(data['model'],strict=True)
   if int(model.frontend_contract_version)!=4:raise RuntimeError('V7 frontend contract drift')
   if arm=='t_v6':
    if int(model.temporal.temporal_contract_version)!=4:raise RuntimeError('V7 query temporal contract drift')
    ema=V7EMA(model,.9995)
   else: ema=DecoderEMA(model,.9995)
   ema.load_checkpoint_state(data['ema']);state=out/f'{arm}_{label}_plain_ema_model_state.pt';meta=out/f'{arm}_{label}_ema_metadata.pt';rec={'checkpoint':str(path),'checkpoint_sha256':sha(path),'operator_contract':data['operator_contract'],'frontend_contract_version':4,'ema_shadow_key_count':len(data['ema']['shadow']),'rng_keys':sorted(data['rng']),'surfaces':{}}
   if arm=='t_v6':rec.update({'temporal_contract_version':4,'immutable_temporal_buffer_names':sorted(data['ema']['immutable_temporal_buffers'])})
   def export(view):
    torch.save({k:v.detach().cpu().clone() for k,v in view.state_dict().items()},state)
    for mode in ('selection','complete'):
     arr=native_predictions(view,cache,device,mode);assert_ids(cache,arr,mode);npz=out/f'{arm}_{label}_{mode}_native.npz';np.savez_compressed(npz,**arr);rec['surfaces'][mode]={'npz':str(npz),'sha256':sha(npz),'n_bins':len(arr['bin_timestep']),'session_count':13,'r2_concat_float64':r2(arr['prediction_native_velocity'],arr['target_native_velocity'])}
   ema.score_with_ema(model,export);torch.save({'schema':'v7_ema_metadata_v1','arm':arm,'checkpoint':str(path),'checkpoint_sha256':sha(path),'ema':ema.checkpoint_state()},meta);rec.update({'plain_ema_model_state':str(state),'plain_ema_model_state_sha256':sha(state),'ema_metadata':str(meta),'ema_metadata_sha256':sha(meta)});manifest['arms'][arm][label]=rec
 target=out/'manifest.json';target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({'manifest':str(target),'sha256':sha(target)}))
if __name__=='__main__':main()
