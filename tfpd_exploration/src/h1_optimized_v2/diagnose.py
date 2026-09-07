"""Causal activity-sensitivity receipt for the H1 frontend revisions."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from .data import build_window_manifest, collate_runtime_target, materialize_banks, shuffled_batches
from .model import H1CurrentQueryDecoder

ROOT=Path(__file__).resolve().parents[2]/"results/decoder_validation_v2/20260905_190000/h1"
def main():
 p=build_window_manifest(); s=p['train_sessions']; b=materialize_banks(s)
 items=shuffled_batches(s,seed=42,epoch=1,batch_size=8)[0]; name,x,y,_,_=collate_runtime_target(s,items)
 dev=torch.device('cuda:0'); bank=type(b[name])(b[name].E0.to(dev),b[name].T.to(dev),b[name].unit_mask.to(dev)); x=x.to(dev)
 rows=[]
 for scale in (1.0,4.0):
  m=H1CurrentQueryDecoder(activity_scale=scale).to(dev).eval()
  with torch.no_grad():
   z=m.frontend(x*scale,bank.E0,bank.T,bank.unit_mask.unsqueeze(0).expand(x.size(0),-1)); out=m.forward_last(x,bank)
   probes={}
   for age in (0,1,4,20,100,699):
    q=x.clone();q[:,-1-age,:]=0
    probes[str(age)]=float((out-m.forward_last(q,bank)).abs().mean().cpu())
  rows.append({'frontend_revision':'original_v1' if scale==1 else 'activity_balanced_scalar_v1_scale_4','scale':scale,'z_sample_std':float(z.std(0).mean().cpu()),'z_sample_std_last':float(z[:,-1].std(0).mean().cpu()),'prediction_sample_std':float(out.std(0).mean().cpu()),'mean_abs_prediction_change_by_age':probes})
 report={'schema':'h1_activity_sensitivity_v2','source_only':True,'rows':rows,'note':'age 0 is current bin; all probes zero an observed past/current bin only, never expose future input'}
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'activity_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
