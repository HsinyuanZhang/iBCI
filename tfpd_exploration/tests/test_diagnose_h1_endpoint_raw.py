import pytest,torch,numpy as np
from tfpd_exploration.src.family_runtime_v1 import diagnose_h1_endpoint_raw as d
def test_gate_precedes_reads(tmp_path,monkeypatch):
 monkeypatch.delenv('H1_ENDPOINT_RAW_GO',raising=False);monkeypatch.setattr(d,'sha',lambda p:pytest.fail('read before gate'))
 with pytest.raises(RuntimeError,match='explicit GO'):d.preflight(tmp_path.resolve(),(tmp_path/'o').resolve(),tmp_path/'p','0'*64,tmp_path/'r','0'*64)
def test_payload_schema_accepts_exact_raw_state_and_rejects_epoch():
 frozen={'protocol':{'x':1},'bindings':{'source_authority_sha256':'a'*64},'code_closure':{'x':'y'}}
 import hashlib,json
 protocol=hashlib.sha256((json.dumps(frozen['protocol'],sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
 binding={'source':{'complete_source':{'frozen':frozen}},'ready':{'flat':{'identities':{'12':{'sampler_sha256':'b'}},'shared_sha256':'c'}}}
 p={'schema':'h1_crst_b4_splitarm_end_epoch_checkpoint_v1','arm':'flat','epoch':12,'global_step':8772,'next_batch_index':731,'ema':{'n_updates':8772,'decay':.9995},'model':{'x':torch.ones(1,dtype=torch.float32)},'protocol_sha256':protocol,'source_authority_sha256':'a'*64,'code_sha256':{'x':'y'},'sampler_identity_sha256':'b','dropout_identity_sha256':'d','shared_init_sha256':'c'}
 binding['ready']['flat']['identities']['12']['keep_sha256']='d'
 assert d.validate_payload(p,'flat',binding)['x'].item()==1
 p['epoch']=11
 with pytest.raises(RuntimeError):d.validate_payload(p,'flat',{})

def test_cli_requires_both_selected_runtime_bindings():
 with pytest.raises(SystemExit):d.main(['--formal','/a','--output','/b'])

def test_guarded_model_rawscore_and_archive_metadata_contract():
 class M(torch.nn.Module):
  def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(1.))
  def forward_last(self,x,bank):return x[:,-1,:7]*self.p
 events=[];m=M();g=d.GuardedModel(m,lambda calls,rows:events.append((calls,rows)))
 out=g.forward_last(torch.ones(3,700,176),None)
 assert out.shape==(3,7) and g.calls==1 and g.rows==3 and events==[(0,0),(1,3)]
 assert d.RawScore().score_with_ema(m,lambda q:q is m)
 arrays={'prediction':np.zeros((208,7),np.float64),'target':np.ones((208,7),np.float64),'session_id':np.array(['s']*208),'start':np.arange(208,dtype=np.int64)}
 d.check_archive(arrays,arrays,source208=True)
 bad={**arrays,'target':arrays['target'].astype(np.float32)}
 with pytest.raises(RuntimeError):d.check_archive(bad,arrays,source208=True)


def test_guard_rejects_before_forward_and_coordinate_drift():
 class Never:
  def forward_last(self, x, bank):pytest.fail('forward after resource failure')
 def reject(calls,rows):raise RuntimeError('resource bound')
 model=d.GuardedModel(Never(),reject)
 with pytest.raises(RuntimeError,match='resource bound'):
  model.forward_last(torch.ones(1,700,176),None)
 assert model.calls==model.rows==0
 arrays={'prediction':np.zeros((20325,7),np.float64),'target':np.ones((20325,7),np.float64),
         'session_id':np.array(['s']*20325),'end':np.arange(20325,dtype=np.int64)}
 bad={**arrays,'end':arrays['end']+1}
 with pytest.raises(RuntimeError,match='exact query metadata'):
  d.check_archive(bad,arrays,source208=False)


def test_real_native_helpers_with_guarded_fake_model_and_all_fixed_cardinalities():
 """The actual B8 complete and B16 source helpers run, with no trained model/data."""
 from tfpd_exploration.src.h1_family_v1.familyformal_split_train import score_complete_cached_ema
 from tfpd_exploration.src.family_runtime_v1.diagnose_h1_selected_source208 import evaluate_source208,state_digest
 from tfpd_exploration.src.h1_family_v1 import cold_phase_summary as summary
 class Identity(torch.nn.Module):
  def __init__(self):
   super().__init__();self.scale=torch.nn.Parameter(torch.tensor(20.,dtype=torch.float32))
  def forward_last(self,x,bank):return x[:,-1,:7]*self.scale
 cache={'train':{},'minival':{}};fixed={}
 for index in range(13):
  count=1563 if index<12 else 1569
  name=f'ses-{index:02d}'
  neural=np.zeros((count,176),np.float32)
  neural[:,:7]=np.arange(count,dtype=np.float32)[:,None]/1000+np.arange(7,dtype=np.float32)[None,:]/10
  starts=np.arange(count-699,dtype=np.int64)
  bank={'E0':torch.zeros(176,700),'T':torch.zeros(176,4),'unit_mask':torch.ones(176,dtype=torch.bool)}
  row={'neural':neural,'velocity':neural[:,:7].copy(),'eval_mask':np.ones(count,bool),
       'query_starts':starts,'bank':bank}
  cache['train'][name]=row;cache['minival'][name]=row
  fixed[name]=np.linspace(0,count-700,16,dtype=np.int64)
 model=Identity();before=state_digest(model.state_dict());events=[]
 complete_proxy=d.GuardedModel(model,lambda calls,rows:events.append((calls,rows)))
 complete=score_complete_cached_ema(model=complete_proxy,ema=d.RawScore(),cache=cache,device=torch.device('cpu'))
 arrays={key:complete.pop('_'+key) for key in ('prediction','target','session_id','end')}
 d.check_archive(arrays,arrays,source208=False)
 assert complete_proxy.rows==20325 and complete_proxy.calls==sum((len(r['neural'])+7)//8 for r in cache['minival'].values())
 summary._check_e2(complete,summary.metric(arrays))
 assert complete['r2_concat_float64']>.999999
 source_proxy=d.GuardedModel(model,lambda calls,rows:None)
 source,source_arrays=evaluate_source208(source_proxy,cache,fixed,torch.device('cpu'),lambda row,device:None)
 d.check_archive(source_arrays,source_arrays,source208=True)
 summary._check_e1(source,summary.metric(source_arrays))
 assert source_proxy.rows==208 and source_proxy.calls==13
 assert state_digest(model.state_dict())==before
