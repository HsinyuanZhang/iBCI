#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import yaml
R=Path(__file__).resolve().parents[2];X='m2_m24_b3ts_t4_v1';E={'ses-2020-10-30-Run1','ses-2020-10-30-Run2','ses-2020-11-18-Run1','ses-2020-11-19-Run1','ses-2020-11-24-Run1','ses-2020-11-24-Run2'};V={'t4':('B3S','t4'),'b3ts':('B3TS','t4')};PROTOCOL_SHA='5779aaee90f1a4050b135803ef6a84bd2b556330ef00ed171ca8f071c6e2f286'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 q=argparse.ArgumentParser();q.add_argument('--artifact',type=Path,required=True);q.add_argument('--group',choices=V,required=True);a=q.parse_args();p=a.artifact.resolve();src=R/f'sua_exploration/results/{X}/aggregate_source.json';z=json.loads(src.read_text());receipt=R/f'sua_exploration/results/{X}/protocol_stage0_and_conditional_expansion.json'
 if not receipt.is_file() or sha(receipt)!=PROTOCOL_SHA:raise ValueError('frozen protocol receipt')
 c=yaml.safe_load((p/'resolved_config.yaml').read_text());d,m=c['data'],c['model'];v,s=V[a.group]
 if (d.get('task'),d.get('loso_fold'),c.get('seed'),m.get('variant'),d.get('side_feature_group'),d.get('calibration_n_trials'),d.get('random_calibration'),d.get('include_heldout_in_fit'),d.get('include_heldout_in_test'),d.get('query_start_trial'),c.get('train'),c.get('test'))!=('m2',1,42,v,s,24,False,False,True,24,False,True):raise ValueError('config contract')
 k=z['arms'][a.group]['checkpoint'];cp=Path(k['path']);
 if not cp.is_file() or sha(cp)!=k['sha256'] or str(Path(c['ckpt_path']).resolve())!=str(cp.resolve()):raise ValueError('source checkpoint contract')
 sp=json.loads((p/'split_manifest.json').read_text());w=sp.get('heldout_query_window_audit',{});
 if set(w)!=E:raise ValueError('six audits')
 for n,x in w.items():
  if not(x.get('support_trials')==x.get('query_start_trial')==24 and x.get('window_size')==50 and x.get('full_window_disjoint') and x.get('minimum_window_start_padded_bin')==x.get('raw_query_start_bin')+49):raise ValueError('window')
 files=[]
 for f in sorted((R/'SPINT-main/data/000953').rglob('*held-out-calib*.nwb')):files.append({'session':f.name.split('_')[1].split('.')[0],'path':str(f.resolve()),'sha256':sha(f)})
 if {x['session'] for x in files}!=E:raise ValueError('NWB set')
 out=p/'heldout_b3ts_t4_provenance.json';
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps({'local_heldout_calib_evaluated':True,'formal_heldout_evaluated':False,'hidden_evalai_evaluated':False,'group':a.group,'frozen_protocol_receipt':str(receipt.resolve()),'frozen_protocol_receipt_sha256':PROTOCOL_SHA,'source_aggregate_sha256':sha(src),'source_checkpoint':k,'six_calibration_nwbs':files,'query_contract':w,'label_budget':'M24 neural activity plus trial-level target-direction labels only; no continuous velocity/K4 labels','no_backward_optimizer_or_checkpoint_selection_on_heldout':True},indent=2)+'\n');print(out)
if __name__=='__main__':main()
