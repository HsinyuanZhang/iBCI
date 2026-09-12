import numpy as np
from dataclasses import dataclass
from dandi688_bench_v2.baseline_runner import run_baselines

@dataclass
class R:
 session_id:str; split:str; representation:str; neural:object; velocity:object; query_indices:object; support_indices:object; carrier_indices:object; channel_indices:object; metadata:dict

def _r(name, rep):
 rng=np.random.RandomState(abs(hash(name+rep))%999); x=rng.poisson(2,size=(80,4)).astype(float)
 return R(name,'train',rep,x,np.c_[np.arange(80.),-np.arange(80.)],np.arange(20,80),np.arange(20),np.arange(10),np.arange(4),{'raw_nwb_sha256':'x','array_sha256':{'query_indices':'q','velocity':'v'},'canonical_electrode_keys':[]})

def test_smoke_runner_writes_only_smoke_dev_surrogate_artifacts(tmp_path):
 src={'pmua':[_r('20150716','pmua')], 'sua':[_r('20150716','sua')]}; dev={'pmua':[_r('20150715','pmua')], 'sua':[_r('20150715','sua')]}
 got=run_baselines(tmp_path,tmp_path/'out',smoke=True,methods=('wf_zs_h0',),source_records=src,dev_records=dev)
 assert got['status']=='SMOKE' and got['final_loaded'] is False
 assert (tmp_path/'out'/'selection.json').exists()
