import numpy as np,pytest
from tfpd_exploration.src.family_runtime_v1.m1_replay_contract import replay_session,W
class R:
 def __init__(self):self.raw=None
 def predict(self,x):self.raw=np.concatenate((self.raw[:,1:],x[:,None]),1);return np.pad(x[:,:4],((0,0),(0,12))).astype(np.float32,copy=True)
def test_gap_replay():
 raw=np.arange(250*64,dtype=np.float32).reshape(250,64);starts=np.array([0,3,20,100],np.int64);pred=np.pad(raw[starts+W-1,:4],((0,0),(0,12))).astype(np.float32);a={'start':starts,'session':np.array(['s']*4),'prediction':pred,'target':np.zeros((4,16),np.float32)};r=R()
 def init(x):r.raw=x.copy()
 z=replay_session(r,raw,'s',a,lambda q:np.pad(q[0,-1:,:4],((0,0),(0,12))).astype(np.float32),initialize_history=init,current_prediction=lambda:pred[:1].copy());assert z['public_calls']==100 and z['scored_count']==4 and z['prediction'].shape==(4,16)


def test_replay_checks_exact_endpoints_and_native_parity():
 raw=np.arange(250*64,dtype=np.float32).reshape(250,64);starts=np.array([0,100],np.int64);pred=np.pad(raw[starts+W-1,:4],((0,0),(0,12))).astype(np.float32);a={'start':starts,'session':np.array(['s']*2),'prediction':pred,'target':np.zeros((2,16),np.float32)};r=R();queries=[];events=[]
 def init(x):r.raw=x.copy()
 def native(q):
  queries.append(q.copy());return np.pad(q[0,-1:,:4],((0,0),(0,12))).astype(np.float32)
 result=replay_session(r,raw,'s',a,native,initialize_history=init,current_prediction=lambda:pred[:1].copy(),progress=events.append)
 # The initialized window covers endpoint 99.  The replay directly checks
 # endpoints 100, 103, 198 and 199 in addition to that initialization.
 assert result['direct_count']==5
 assert [q.shape for q in queries]==[(1,W,64)]*5
 assert [int(q[0,-1,0]//64) for q in queries]==[99,100,103,198,199]
 assert events==[{'public_calls':100,'end':199,'session':'s'}]


def test_replay_rejects_initial_raw_drift():
 raw=np.zeros((100,64),np.float32);a={'start':np.array([0],np.int64),'session':np.array(['s']),'prediction':np.zeros((1,16),np.float32),'target':np.zeros((1,16),np.float32)}
 class Broken:
  raw=np.ones((1,W,64),np.float32)
  def predict(self,x):return np.zeros((1,16),np.float32)
 with pytest.raises(RuntimeError,match='initial raw drift'):
  replay_session(Broken(),raw,'s',a,lambda q:np.zeros((1,16),np.float32),initialize_history=lambda x:None,current_prediction=lambda:np.zeros((1,16),np.float32))


def test_replay_rejects_runtime_raw_drift_and_bad_native_output():
 raw=np.arange(101*64,dtype=np.float32).reshape(101,64);starts=np.array([0,1],np.int64);pred=np.pad(raw[starts+W-1,:4],((0,0),(0,12))).astype(np.float32);a={'start':starts,'session':np.array(['s']*2),'prediction':pred,'target':np.zeros((2,16),np.float32)}
 class Drift(R):
  def predict(self,x):
   super().predict(x);self.raw[:]=0;return np.pad(x[:,:4],((0,0),(0,12))).astype(np.float32)
 r=Drift()
 with pytest.raises(RuntimeError,match='raw drift'):
  replay_session(r,raw,'s',a,lambda q:np.pad(q[0,-1:,:4],((0,0),(0,12))).astype(np.float32),initialize_history=lambda x:setattr(r,'raw',x.copy()),current_prediction=lambda:pred[:1].copy())
 r=R()
 with pytest.raises(RuntimeError,match='public output'):
  replay_session(r,raw,'s',a,lambda q:np.zeros((1,16),np.float64),initialize_history=lambda x:setattr(r,'raw',x.copy()),current_prediction=lambda:pred[:1].copy())


def test_replay_rejects_native_prediction_mismatch():
 raw=np.zeros((100,64),np.float32);a={'start':np.array([0],np.int64),'session':np.array(['s']),'prediction':np.zeros((1,16),np.float32),'target':np.zeros((1,16),np.float32)};r=R()
 with pytest.raises(RuntimeError,match='native mismatch'):
  replay_session(r,raw,'s',a,lambda q:np.ones((1,16),np.float32),initialize_history=lambda x:setattr(r,'raw',x.copy()),current_prediction=lambda:np.zeros((1,16),np.float32))
def test_bad_metadata():
 with pytest.raises(ValueError):replay_session(None,np.zeros((1,64),np.float32),'s',{'start':np.array([],np.int64),'session':[],'prediction':np.zeros((0,16)),'target':np.zeros((0,16))},lambda x:None,initialize_history=lambda x:None,current_prediction=lambda:None)
