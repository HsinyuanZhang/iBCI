"""Read-only float64 rank/conditioning audit of sealed signed frontend maps.

Only source-cache bank tensors and sealed plain EMA state dictionaries are
consumed.  Neural arrays, targets, starts, query features, NWB files and
models' temporal/readout paths are never accessed.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from tfpd_exploration.src.h1_optimized_v6.model import H1RecencyPriorQuery
from tfpd_exploration.src.h1_optimized_v4.model import H1SignedFull

OUT=ROOT/'v8_readonly_operator_hypothesis_v1/rank_conditioning_v1.json'
CASES={
 'v6_t_selected_e12':(H1RecencyPriorQuery,ROOT/'paired_v6_recency_queryonly_12ep_v1/independent_score_export/t_selected_plain_ema_model_state.pt'),
 'v7_t_selected_e12':(H1RecencyPriorQuery,ROOT/'v7_dropout30_paired12_v1/independent_score_export/t_v6_selected_plain_ema_model_state.pt'),
 'v7_full_selected_e12':(H1SignedFull,ROOT/'v7_dropout30_paired12_v1/independent_score_export/full_v4_selected_plain_ema_model_state.pt'),
}
RELATIVE_TOLERANCES=(1e-12,1e-10,1e-8,1e-6,1e-4)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ahash(x):
 a=np.ascontiguousarray(torch.as_tensor(x).detach().cpu().numpy());h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.tobytes());return h.hexdigest()
def summary(a):
 # Exact float32 frontend operator is cast to float64 before SVD.
 s=np.linalg.svd(np.asarray(a,dtype=np.float64),compute_uv=False);smax=float(s[0]);eps_cut=max(a.shape)*np.finfo(np.float64).eps*smax
 ranks={str(t):int(np.count_nonzero(s>smax*t)) for t in RELATIVE_TOLERANCES};ranks['machine_eps_scaled']=int(np.count_nonzero(s>eps_cut))
 def condition(t):
  kept=s[s>smax*t]
  return float(smax/kept[-1]) if len(kept) else float('inf')
 return {'shape':list(a.shape),'singular_values_float64':[float(x) for x in s],'smax':smax,'machine_eps_scaled_cutoff':float(eps_cut),'rank_at_relative_tolerances':ranks,'condition_number_at_relative_tolerances':{str(t):condition(t) for t in RELATIVE_TOLERANCES},'condition_number_machine_eps_scaled':condition(eps_cut/smax)}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 cache=build_or_load();authority=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,authority)
 report={'schema':'h1_signed_map_rank_conditioning_v1','status':'COMPLETE_READ_ONLY','scope':'source train bank E0/T/unit_mask plus sealed plain EMA frontend parameters only; no neural/velocity/starts/query/NWB/model forward','float_operator':'exact float32 frontend weights cast to float64 solely for SVD','relative_tolerances':list(RELATIVE_TOLERANCES),'cases':{},'source_cache_authority_sha256':sha(ROOT/'source_cache_authority.json'),'code_sha256':sha(Path(__file__))}
 for label,(klass,path) in CASES.items():
  if not path.is_file():raise FileNotFoundError(path)
  model=klass();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);front=model.frontend;case={'plain_ema_state':str(path),'plain_ema_state_sha256':sha(path),'frontend_contract_version':int(model.frontend_contract_version.item()),'sessions':{}}
  for session,row in sorted(cache['train'].items()):
   bank=H1Bank(row['bank']['E0'],row['bank']['T'],row['bank']['unit_mask']);active=np.asarray(bank.unit_mask.detach().cpu(),dtype=bool)
   if active.ndim!=1 or not active.any():raise RuntimeError('active unit mask drift')
   # x is a dtype/device sentinel only; frontend.weights is bank-only.
   w=front.weights(bank,torch.zeros(1,dtype=torch.float32)).detach().cpu().numpy()[active].astype(np.float64,copy=False)
   n=int(active.sum());mean=np.full((n,1),1.0/n,dtype=np.float64);p=front.pop_projection.weight.detach().cpu().numpy().reshape(-1).astype(np.float64,copy=False)
   augmented=np.concatenate((w,mean),axis=1)
   actual=w+mean*p.reshape(1,-1)
   case['sessions'][session]={'active_units':n,'bank_sha256':{'E0':ahash(bank.E0),'T':ahash(bank.T),'unit_mask':ahash(bank.unit_mask)},'signed_only_W':summary(w),'augmented_W_plus_mean_upperbound':summary(augmented),'actual_composed_A_W_plus_mean_outer_pop_weight':summary(actual)}
  report['cases'][label]=case
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(OUT),'cases':list(report['cases'])}))
if __name__=='__main__':main()
