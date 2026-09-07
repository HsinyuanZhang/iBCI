import hashlib,pytest,torch,numpy as np
from tfpd_exploration.src.h1_family_v1 import dense_phase2_pair as d
def test_protocol_is_fixed_dense_endpoint_contract():
 assert d.PROTOCOL['positions']==[349,466,582,699] and d.PROTOCOL['loss'].startswith('equal .25') and d.PROTOCOL['epochs']==2 and d.PROTOCOL['lr']==1e-5
def test_gate_precedes_authorization_read(tmp_path,monkeypatch):
 monkeypatch.delenv('H1_DENSE_PHASE2_GO',raising=False);monkeypatch.setattr(d,'sha',lambda p:pytest.fail('read before gate'))
 with pytest.raises(RuntimeError,match='explicit GO'):d.preflight(tmp_path.resolve(),(tmp_path/'o').resolve(),tmp_path/'a','0'*64)
class E:
 def __init__(self):self.n=0
 def update_after_step(self,m):self.n+=1
class M(torch.nn.Module):
 def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.))
 def encode_frontend(self,x,b,k):return x[:,:,:1].expand(-1,-1,2)
 def temporal(self,z):return z
 def final_norm(self,z):return z
 def readout(self,z):return z[...,:1].expand(*z.shape[:-1],7)*self.w
def test_dense_step_microbatch_validity():
 x=torch.ones(10,700,176);rec={'velocity':np.ones((710,7),np.float32),'eval_mask':np.ones(710,bool)};starts=np.arange(10,dtype=np.int64);models={a:M() for a in ('flat','route')}
 class O:
  def __init__(self,m):self.m=m
  def zero_grad(self,set_to_none=True):self.m.w.grad=None
  def step(self):pass
 out=d.dense_step(torch,models,{a:O(m) for a,m in models.items()},{a:E() for a in models},x,rec,starts,None,torch.ones(10,176,dtype=torch.bool));assert out['endpoint_valid'] and out['valid_fraction']==1

def test_dense_micro_accumulation_uses_one_full_ragged_denominator():
 # Position 349 is invalid for the last two rows; positions 466/582/699 are
 # valid.  Splitting at MICRO must still equal the single full-batch MSE.
 x=torch.ones(10,700,176); rec={'velocity':np.ones((710,7),np.float32),'eval_mask':np.ones(710,bool)};rec['eval_mask'][np.arange(8)+349]=False
 starts=np.arange(10,dtype=np.int64);models={a:M() for a in ('flat','route')}
 class O:
  def __init__(self,m):self.m=m
  def zero_grad(self,set_to_none=True):self.m.w.grad=None
  def step(self):pass
 out=d.dense_step(torch,models,{a:O(m) for a,m in models.items()},{a:E() for a in models},x,rec,starts,None,torch.ones(10,176,dtype=torch.bool))
 # predictions are 1 and all valid targets are 20, hence (19^2), independent
 # of the ragged position count because numerator and denominator match.
 assert out['loss']['flat']==pytest.approx(361.) and out['loss']['route']==pytest.approx(361.)

def test_checkpoint_strict_restore_roundtrip_cpu():
 class CE:
  def __init__(self,m):self.shadow={k:v.detach().clone() for k,v in m.state_dict().items()};self.n_updates=d.UPDATES
  def checkpoint_state(self):return {'shadow':self.shadow,'n_updates':self.n_updates,'decay':.9995}
  def load_checkpoint_state(self,p):self.shadow={k:v.clone() for k,v in p['shadow'].items()};self.n_updates=p['n_updates']
 models={a:M() for a in ('flat','route')};opts={a:torch.optim.SGD(m.parameters(),lr=.1) for a,m in models.items()};emas={a:CE(m) for a,m in models.items()};ident={'sampler_sha256':'s','keep_sha256':'k'}
 payload=d._checkpoint(torch,models,opts,emas,1,ident)
 with torch.no_grad():models['flat'].w.fill_(9.)
 d._restore_checkpoint(torch,payload,models,opts,emas,1,ident)
 assert models['flat'].w.item()==1. and emas['route'].n_updates==d.UPDATES

def test_tiny_real_loop_then_first_checkpoint(monkeypatch):
 # Exercise the runner's actual sampler -> W700 -> keep -> paired dense-step
 # path with a one-update injected protocol, then its first disk-state payload.
 import tfpd_exploration.src.h1_optimized_v4.paired_train as pairtrain
 import tfpd_exploration.src.h1_family_v1.familyformal_split_train as split
 import tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal as temporal
 monkeypatch.setattr(d,'UPDATES',1);monkeypatch.setattr(d,'SOURCE_WINDOWS',1)
 monkeypatch.setattr(pairtrain,'batches',lambda cache,epoch:[('s',np.array([0],dtype=np.int64))])
 monkeypatch.setattr(split,'sampler_identity_digest',lambda ordered:'sampler')
 monkeypatch.setattr(split,'dropout_keep',lambda **kw:torch.ones(kw['n'],176,dtype=torch.bool))
 monkeypatch.setattr(temporal,'H1Bank',lambda *args:None)
 class CE(E):
  def __init__(self):super().__init__();self.shadow={'w':torch.tensor(1.)};self.n_updates=0
  def update_after_step(self,m):super().update_after_step(m);self.n_updates=self.n
  def checkpoint_state(self):return {'shadow':self.shadow,'n_updates':self.n_updates,'decay':.9995}
  def load_checkpoint_state(self,p):self.shadow=p['shadow'];self.n=self.n_updates=p['n_updates']
 models={a:M() for a in ('flat','route')};opts={a:torch.optim.SGD(m.parameters(),lr=.01) for a,m in models.items()};emas={a:CE() for a in models}
 row={'neural':np.ones((700,176),np.float32),'velocity':np.ones((700,7),np.float32),'eval_mask':np.ones(700,bool),'bank':{k:torch.tensor(1.) for k in ('E0','T','unit_mask')}}
 expected={'1':{'sampler_sha256':'sampler','keep_sha256':hashlib.sha256(b's'+np.array([0],dtype=np.int64).tobytes()+np.ones((1,176),bool).tobytes()).hexdigest()}}
 rows,ids=d._loop(torch,{'train':{'s':row}},models,opts,emas,epochs=1,limit_updates=1,device=torch.device('cpu'),expected=expected)
 assert len(rows)==1 and all(ids['1'][k]==v for k,v in expected['1'].items()) and all(e.n==1 for e in emas.values())
 payload=d._checkpoint(torch,models,opts,emas,1,ids['1']);assert payload['global_step']==1 and set(payload['models'])=={'flat','route'}
