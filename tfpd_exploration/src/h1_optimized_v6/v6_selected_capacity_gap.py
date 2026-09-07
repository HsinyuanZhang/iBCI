"""Read-only source-capacity versus matched-minival diagnostic for sealed V6.

This neither trains nor selects.  The train side is exactly the frozen 208
endpoint ID set from V6's completed source gate.  Because train/minival are
separate chronological arrays of different lengths, minival uses the same
predeclared count (16 evenly-positioned frozen query windows/session), not
invalid train indices.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.capacity_probe import r2
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .model import H1RecencyPriorQuery

ATTEMPT=ROOT/'paired_v6_recency_queryonly_12ep_v1'
EMA=ATTEMPT/'independent_score_export/t_selected_plain_ema_model_state.pt'
FROZEN=ROOT/'capacity_probe_208_source_v6_recency1040/frozen_ids.json'
OUT=ROOT/'v7_readonly_v6_selected_capacity_gap_v1/report.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def starts(row):
 q=np.asarray(row['query_starts'],dtype=np.int64)
 return q[np.linspace(0,len(q)-1,16,dtype=np.int64)]
def score(model,cache,partition,ids,dev):
 ps=[];ys=[]; desc={}
 with torch.inference_mode():
  for session,row in sorted(cache[partition].items()):
   ss=np.asarray(ids[session] if partition=='train' else starts(row),dtype=np.int64)
   x=torch.stack([torch.as_tensor(row['neural'][s:s+700]) for s in ss]).to(dev,dtype=torch.float32)
   b=H1Bank(*[row['bank'][k].to(dev) for k in ('E0','T','unit_mask')])
   p=model.forward_last(x,b).div(20).cpu().numpy(); y=np.asarray([row['velocity'][s+699] for s in ss],dtype=np.float32)
   ps.append(p);ys.append(y);desc[session]={'count':int(len(ss)),'min_start':int(ss.min()),'max_start':int(ss.max())}
 p,y=np.concatenate(ps),np.concatenate(ys)
 return {'examples':int(len(y)),'r2_concat_native':r2(p,y),'prediction_std':float(p.std()),'target_std':float(y.std()),'sessions':desc}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth)
 frozen=json.loads(FROZEN.read_text());ids={k:np.asarray(v,dtype=np.int64) for k,v in frozen['ids'].items()}
 if sum(map(len,ids.values()))!=208:raise RuntimeError('frozen source cardinality drift')
 dev=torch.device('cuda:0');model=H1RecencyPriorQuery().to(dev);model.load_state_dict(torch.load(EMA,map_location='cpu',weights_only=True),strict=True);model.eval()
 report={'schema':'h1_v6_selected_e12_readonly_capacity_gap_v1','status':'COMPLETE_READ_ONLY','prohibitions':['no training','no optimizer','no model selection','no checkpoint writes'],'model':'V6 query-only selected epoch12 EMA','checkpoint':str(EMA),'checkpoint_sha256':sha(EMA),'epoch_checkpoint':str(ATTEMPT/'t_v6_epoch_012.pt'),'epoch_checkpoint_sha256':sha(ATTEMPT/'t_v6_epoch_012.pt'),'source_frozen_ids':str(FROZEN),'source_frozen_ids_sha256':sha(FROZEN),'source_description':'exact V6 capacity-probe frozen 208 train query starts','minival_description':'matched cardinality: 16 deterministic evenly-positioned valid minival query starts per session; train numeric starts are invalid for shorter minival streams','input_authority_sha256':sha(ROOT/'source_cache_authority.json'),'train':score(model,cache,'train',ids,dev),'minival':score(model,cache,'minival',ids,dev)}
 report['r2_train_minus_minival']=report['train']['r2_concat_native']-report['minival']['r2_concat_native']
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:report[k] for k in ('train','minival','r2_train_minus_minival')},sort_keys=True))
if __name__=='__main__':main()
