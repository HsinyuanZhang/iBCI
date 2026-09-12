import numpy as np
import pytest
from btransform_unified_v2.external_baselines_v1.core import CoralAligner,AlignedFA

def test_coral_covariance_and_identity():
 r=np.random.default_rng(1);s=r.normal(size=(400,5));t=s@np.diag([2,.5,3,1.5,.8])+3
 a=CoralAligner(s,ridge=1e-6).calibrate(t);z=a.transform(t)
 assert np.allclose(np.cov(z,rowvar=False),np.cov(s,rowvar=False),atol=.08)
 assert np.allclose(CoralAligner(s).calibrate(s).transform(s),s,atol=2e-4)
def test_aligned_fa_rotation_and_api_has_no_y():
 r=np.random.default_rng(3);q,_=np.linalg.qr(r.normal(size=(3,3))); L=r.normal(size=(8,3));
 assert np.allclose(AlignedFA._rotation(L@q,L),q.T,atol=1e-10)
 z=r.normal(size=(600,3));s=z@L.T+.03*r.normal(size=(600,8));t=(z@q)@L.T+.03*r.normal(size=(600,8))
 m=AlignedFA(s,latent_dim=3,max_iter=500,stable_fraction=.75);a=m.calibrate(t)
 assert a.transform(t).shape==(600,3) and len(a.stable_rows)==6 and a.diagnostics['target_labels_used'] is False
 assert a.transform(t[:1]).shape==(1,3)
def test_stable_pruning_removes_bad_loading_rows():
 r=np.random.default_rng(4);a=r.normal(size=(10,3));b=a.copy();b[-2:]+=100
 keep=np.arange(10)
 while len(keep)>8:
  rot=AlignedFA._rotation(b[keep],a[keep]);err=np.sum((b[keep]@rot-a[keep])**2,1);keep=np.delete(keep,np.argmax(err))
 assert set(keep)==set(range(8))
def test_errors_and_rank_deficient_are_finite():
 x=np.ones((30,4));assert np.isfinite(CoralAligner(x).calibrate(x).transform(x)).all()
 with pytest.raises(ValueError):AlignedFA(np.ones((1,2)))
