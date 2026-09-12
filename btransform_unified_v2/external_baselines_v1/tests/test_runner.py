import numpy as np
from types import SimpleNamespace
from pathlib import Path
from btransform_unified_v2.external_baselines_v1.wiener import WienerRidge
from btransform_unified_v2.external_baselines_v1 import run, data
from btransform_unified_v1.r2 import variance_weighted_r2
def test_wiener_source_only_fit_is_finite():
 rng=np.random.default_rng(0);x=rng.normal(size=(30,8));y=x[:,:2]@np.array([[2.,0.],[0.,3.]])
 p=WienerRidge(1e-3).fit(x,y).predict(x)
 assert p.shape==y.shape and np.isfinite(p).all() and np.mean((p-y)**2)<1e-4

def test_complete_runner_never_uses_target_labels_for_adapters_or_predictions(tmp_path,monkeypatch):
 rng=np.random.default_rng(1)
 def item(y):
  return {'X':rng.normal(size=(12,2)).astype('f4'),'Y':np.asarray(y,'f4'),'starts':np.arange(2,10),
          'pad':0,'activity':rng.normal(size=(3,4,2)).astype('f4'),'hashes':{},'support_provenance':{}}
 source=item(rng.normal(size=(8,2))); target=item(rng.normal(size=(8,2)))
 surface={'train':{'ses-2020-01-01':source},'evaluation':{'target':target},'metadata':{'context':2}}
 monkeypatch.setattr(run.data,'load',lambda task,evaluation:surface)
 a=SimpleNamespace(task='m2',dest=tmp_path/'one',history=2,ridge=1.,coral_ridge=1e-3,coral_shrinkage=.1,fa_dim=1,fa_max_iter=20,fa_n_init=1,fa_stable_fraction=.5,max_batches=1)
 one=run.run(a); first={p.name:np.load(p) for p in a.dest.glob('pred_*.npy')}
 target['Y']+=1000; a.dest=tmp_path/'two'; two=run.run(a); second={p.name:np.load(p) for p in a.dest.glob('pred_*.npy')}
 assert first.keys()==second.keys() and all(np.array_equal(first[k],second[k]) for k in first)
 assert all(v['n_windows']==8 and np.isfinite(v['standard_variance_weighted_equal_session_mean']) for v in two['reports'].values())

def test_transformed_padding_stays_literal_zero():
 item={'X':np.ones((6,2),np.float32),'starts':np.array([0]),'pad':2}
 out=run._fit_features(item,np.array([0]),context=4,history=4,transform=lambda x:x+7)
 assert np.all(out.reshape(1,4,2)[0,:2]==0)

def test_multioutput_variance_weighted_is_not_flattened_metric():
 y=np.array([[0.,0.],[1.,100.],[2.,200.]])
 p=np.array([[0.,10.],[1.,110.],[2.,210.]])
 assert run._r2(y,p) != variance_weighted_r2(y,p)
