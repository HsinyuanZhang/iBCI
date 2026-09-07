"""Read-only independent native export for deterministic V4 FULL continuation."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v4.export_predictions import assert_ids,native_predictions,r2
from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def mask_hashes(cache):
 ans={}
 for split in ('train','minival'):
  ans[split]={}
  for n,row in cache[split].items():
   a=row['bank']['unit_mask'].detach().cpu().contiguous().numpy();h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.tobytes());ans[split][n]=h.hexdigest()
 return ans
def source_hashes():
 here=Path(__file__).resolve();src=here.parents[1];files={'continuation_train':here.with_name('v4_full_continue24.py'),'continuation_export':here,'v4_model':src/'h1_optimized_v4/model.py','v4_paired_train':src/'h1_optimized_v4/paired_train.py','v4_export_helpers':src/'h1_optimized_v4/export_predictions.py','v2_cache':src/'h1_optimized_v2/cache.py','v2_score':src/'h1_optimized_v2/score.py','ema':src/'h1_temporal_decoder_quick_product_v1/ema.py','h1_temporal_root':src/'two_mainlines_long_v1/decoder/h1_temporal.py'}
 if any(not x.is_file() for x in files.values()):raise FileNotFoundError('operator source missing')
 return {k:sha(v) for k,v in files.items()}
def contract(model,ema):
 if {n for n,p in model.named_parameters() if p.requires_grad}!=set(ema['shadow']):raise RuntimeError('EMA/trainable keys differ')
 if int(model.frontend_contract_version.item())!=4:raise RuntimeError('frontend contract must be 4')
def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',default='v4_full_continue24_deterministic_v1');p.add_argument('--device',default='cpu');a=p.parse_args();root=ROOT/a.attempt
 needed=[root/'input_authority.json',root/'protocol_freeze.json',root/'selection_freeze.json',root/'final.json']
 if any(not x.is_file() for x in needed):raise FileNotFoundError('completed continuation receipts required')
 out=root/'independent_score_export'
 if out.exists():raise FileExistsError(out)
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);inp=json.loads((root/'input_authority.json').read_text());freeze=json.loads((root/'selection_freeze.json').read_text());device=torch.device(a.device);out.mkdir()
 manifest={'schema':'h1_v4_continuation_independent_native_score_export_v1','attempt':a.attempt,'committed_input_bindings':{x.name:sha(x) for x in needed}|{'source_cache_authority.json':sha(ROOT/'source_cache_authority.json')},'input_authority':auth,'operator_source_sha256':source_hashes(),'per_session_bank_unit_mask_sha256':mask_hashes(cache),'numerical_runtime_disclosure':inp['protocol']['numerical_runtime_disclosure'],'protocol_freeze_sha256':sha(root/'protocol_freeze.json'),'arms':{'full':{}},'surfaces':{'selection':{'n_bins':2908},'complete':{'n_bins':20325}},'write_semantics':'completed files only; no atomic-write claim'}
 model=make_matched_pair()[0].to(device);paths={'selected':Path(freeze['selected']['checkpoint']),'epoch24':root/'full_epoch_024.pt'}
 for label,path in paths.items():
  data=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(data['model'],strict=True);contract(model,data['ema']);ema=DecoderEMA(model,decay=float(data['ema']['decay']));ema.load_checkpoint_state(data['ema']);state=out/f'full_{label}_plain_ema_model_state.pt';meta=out/f'full_{label}_ema_metadata.pt';record={'checkpoint':str(path),'checkpoint_sha256':sha(path),'frontend_contract_version':4,'ema_shadow_key_count':len(data['ema']['shadow']),'surfaces':{}}
  def export(view):
   torch.save({k:v.detach().cpu().clone() for k,v in view.state_dict().items()},state)
   for mode in ('selection','complete'):
    arrays=native_predictions(view,cache,device,mode);assert_ids(cache,arrays,mode);npz=out/f'full_{label}_{mode}_native.npz';np.savez_compressed(npz,**arrays);record['surfaces'][mode]={'npz':str(npz),'sha256':sha(npz),'n_bins':len(arrays['bin_timestep']),'session_count':13,'r2_concat_float64':r2(arrays['prediction_native_velocity'],arrays['target_native_velocity'])}
  ema.score_with_ema(model,export);torch.save({'schema':'decoder_ema_metadata_v1','arm':'full','checkpoint':str(path),'checkpoint_sha256':sha(path),'ema':ema.checkpoint_state()},meta);record.update({'plain_ema_model_state':str(state),'plain_ema_model_state_sha256':sha(state),'ema_metadata':str(meta),'ema_metadata_sha256':sha(meta)});manifest['arms']['full'][label]=record
 target=out/'manifest.json';target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({'manifest':str(target),'sha256':sha(target)}))
if __name__=='__main__':main()
