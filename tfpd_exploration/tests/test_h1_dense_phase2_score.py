import os,pytest,numpy as np,torch
from tfpd_exploration.src.h1_family_v1 import dense_phase2_score as s

def test_archive_contract_requires_native_fp64_and_metadata():
 a={'prediction':np.zeros((208,7),np.float64),'target':np.ones((208,7),np.float64),'session_id':np.repeat([f'ses-{i:02d}' for i in range(13)],16),'start':np.arange(208,dtype=np.int64)}
 s.check_archive(a,a,source208=True)
 b=dict(a);b['prediction']=b['prediction'].astype(np.float32)
 with pytest.raises(RuntimeError):s.check_archive(b,a,source208=True)

def test_dense_payload_strictly_binds_e2_protocol_authority_and_ema():
 from tfpd_exploration.src.h1_family_v1 import dense_phase2_pair as d
 state={'w':torch.ones(1)}; ident={'sampler_sha256':'s','keep_sha256':'k'}
 p={'schema':'h1_dense_phase2_end_epoch_v1','protocol_sha256':d.digest(d.PROTOCOL),'authority_sha256':'a'*64,'epoch':2,'global_step':1462,'identities':ident,'models':{'flat':state,'route':state},'optimizers':{'flat':{},'route':{}},'emas':{a:{'shadow':state,'n_updates':1462,'decay':.9995} for a in s.ARMS},'rng':{}}
 b={'dense_authority_sha256':'a'*64,'dense_receipt_identities':{'2':ident}}
 assert s.validate_dense_payload(p,b) is p
 p['emas']['flat']['n_updates']=1
 with pytest.raises(RuntimeError):s.validate_dense_payload(p,b)

def test_gate_precedes_authorization_or_dense_load(tmp_path,monkeypatch):
 monkeypatch.delenv('H1_DENSE_PHASE2_SCORE_GO',raising=False)
 monkeypatch.setattr(s,'collect_bindings',lambda *x:pytest.fail('audit before gate'))
 with pytest.raises(RuntimeError,match='explicit GO'):
  s.preflight(tmp_path.resolve(),tmp_path.resolve(),'0'*64,tmp_path/'cold','0'*64,(tmp_path/'out').resolve(),tmp_path/'auth','0'*64)

def test_actual_complete_and_source_helpers_on_synthetic_fixed_cardinalities():
 from tfpd_exploration.src.h1_family_v1.familyformal_split_train import score_complete_cached_ema
 from tfpd_exploration.src.family_runtime_v1.diagnose_h1_selected_source208 import evaluate_source208
 class M(torch.nn.Module):
  def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(20.))
  def forward_last(self,x,bank):return x[:,-1,:7]*self.p
 cache={'train':{},'minival':{}};fixed={}
 for i in range(13):
  n=1563 if i<12 else 1569; cold=670 if i<5 else 669; timeline=699+n-cold
  neural=np.zeros((timeline,176),np.float32);neural[:,:7]=np.arange(timeline,dtype=np.float32)[:,None]/1000
  mask=np.zeros(timeline,bool);mask[:cold]=True;mask[699:]=True
  name=f'ses-{i:02d}';row={'neural':neural,'velocity':neural[:,:7].copy(),'eval_mask':mask,'query_starts':np.arange(timeline-699,dtype=np.int64),'bank':{'E0':torch.zeros(176,700),'T':torch.zeros(176,4),'unit_mask':torch.ones(176,dtype=torch.bool)}}
  cache['train'][name]=cache['minival'][name]=row;fixed[name]=np.linspace(0,timeline-700,16,dtype=np.int64)
 model=M();complete=score_complete_cached_ema(model=model,ema=s.Raw(),cache=cache,device=torch.device('cpu')); a2={k:complete.pop('_'+k) for k in ('prediction','target','session_id','end')}
 source,a1=evaluate_source208(model,cache,fixed,torch.device('cpu'),lambda row,d:None)
 assert a2['prediction'].shape==(20325,7) and a1['prediction'].shape==(208,7)
 from tfpd_exploration.src.h1_family_v1 import cold_phase_summary as summary
 s.check_archive(a1,a1,source208=True);s.check_archive(a2,a2,source208=False)
 summary._check_e1(source,summary.metric(a1));summary._check_e2(complete,summary.metric(a2))
 analysis=summary.analysis(a2)
 assert analysis['groups']['cold_history_lt_699']['n_bins']==8702
 assert analysis['groups']['full_w700_ge_699']['n_bins']==11623
 assert all(abs(value['r2_concat_float64']-1)<1e-10 for value in analysis['groups'].values())
