import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'static_controls.py'; spec=importlib.util.spec_from_file_location('static_controls',P); sc=importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
def test_causal_transform_and_literal_padding():
 src={'a':{'support':np.array([[1.,2.],[2.,4.],[3.,6.]])}}
 target={'support':np.array([[10.,20.],[12.,24.],[14.,28.]])}
 f=sc.fit(src,target,'diag_z'); a=np.array([[11.,22.]],np.float32); b=np.array([[99.,198.]],np.float32)
 assert np.allclose(sc.transform(a,f),sc.transform(a,f)); assert not np.allclose(sc.transform(a,f),sc.transform(b,f))
 item={'X':np.ones((5,2),np.float32),'pad':2,'starts':np.array([0])}; x,_=sc.contexts(item,np.full((3,2),7,np.float32)); assert np.array_equal(x[:2],np.zeros((2,2),np.float32))
def test_identity_and_coral_shapes():
 src={'a':{'support':np.arange(20,dtype=np.float32).reshape(10,2)}}; target={'support':np.arange(30,50,dtype=np.float32).reshape(10,2)}; z=target['support']
 assert np.array_equal(sc.transform(z,sc.fit(src,target,'identity')),z)
 assert sc.transform(z,sc.fit(src,target,'coral')).shape==z.shape
