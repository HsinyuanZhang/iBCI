import numpy as np,pytest
from tfpd_exploration.src.family_runtime_v1.h1_replay_contract import replay_session,raw_window
class R:
 def __init__(self):self.raw=np.zeros((1,700,3),np.float32)
 def predict(self,x):self.raw[:,:-1]=self.raw[:,1:];self.raw[:,-1]=x[0];return np.pad(x[:,:],((0,0),(0,4))).astype(np.float32,copy=True)
def fixture():
 x=np.arange(702*3,dtype=np.float32).reshape(702,3);v=np.arange(702*7,dtype=np.float32).reshape(702,7);m=np.zeros(702,bool);m[[1,700,701]]=1;e=np.flatnonzero(m);return x,v,m,{'end':e,'session_id':np.array(['s']*3),'target':v[e],'prediction':np.pad(x[e],((0,0),(0,4))).astype(np.float32)}
def test_all_bins_raw_archive_and_native():
 x,v,m,a=fixture();r=R();z=replay_session(r,x,v,m,'s',a,lambda q:np.pad(q[-1:],((0,0),(0,4))).astype(np.float32));assert z['public_calls']==702 and z['scored_count']==3 and z['direct_count']==5
@pytest.mark.parametrize('key,value',[('end',np.array([1,2,701])),('session_id',np.array(['x']*3)),('target',np.zeros((3,7),np.float32))])
def test_archive_identity_reject(key,value):
 x,v,m,a=fixture();a[key]=value
 with pytest.raises(RuntimeError):replay_session(R(),x,v,m,'s',a)
class BadRaw(R):
 def predict(self,x):return np.zeros((1,7),np.float32)
def test_raw_native_and_output_contract_reject():
 x,v,m,a=fixture()
 with pytest.raises(RuntimeError):replay_session(BadRaw(),x,v,m,'s',a)
 with pytest.raises(RuntimeError):replay_session(R(),x,v,m,'s',a,lambda q:np.full((1,7),np.nan,np.float32))
 with pytest.raises(RuntimeError):replay_session(R(),x,v,m,'s',a,lambda q:np.zeros((2,7),np.float32))
