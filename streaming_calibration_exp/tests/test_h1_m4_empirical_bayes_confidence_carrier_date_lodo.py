from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'streaming_calibration_exp/scripts/audit_h1_m4_empirical_bayes_confidence_carrier_date_lodo.py'
SPEC=importlib.util.spec_from_file_location('h1_m4_empirical_bayes',SCRIPT)
assert SPEC and SPEC.loader
audit=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=audit; SPEC.loader.exec_module(audit)

def test_analytic_shrinkage_is_four_dimensional_and_row_shuffle_is_complete():
 rng=np.random.default_rng(4); trials=[]; mix=rng.normal(size=(7,176))*.2
 for n in range(1,8):
  z=np.linspace(0,2*np.pi,30,endpoint=False)+n*.1; y=np.column_stack([np.sin((j+1)*z) for j in range(7)])
  trials.append(audit.raw.m2.h1.TrialBlocks(float(n),30+y@mix,y,{}))
 rec=audit.raw.m2.FullRecord('ses-19250101T000001','19250101',None,'synthetic',tuple(trials))
 p=audit.Frozen('19250101',(),np.full(176,30.),np.ones(176),np.eye(16,176),16,1.,np.eye(7,4),np.zeros(4),.5,'synthetic')
 beta,G,sig=audit._fit(rec,p); e,w,v=audit._shrink(beta,G,sig,p)
 assert e.shape==(176,4) and np.all((w>0)&(w<=1)) and np.all(v>=0)
 shuffled,permsha=audit._shuffle(e,rec,p)
 assert permsha and not np.array_equal(shuffled,e)
 assert audit._predict_metric(rec,e,beta[0],p)['r2'] is not None
