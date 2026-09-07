from __future__ import annotations
import numpy as np, pytest, torch
from tfpd_exploration.src.m2_anchored_postfusion_capacity_audit_refit_v2 import audit,plan
from tfpd_exploration.src.m2_anchored_postfusion_capacity_screen_v1.gates import CapacityGate,disagreement_statistic
def _screen():
 rows={}
 for arm,n in [('A-S1',1),('A-TB4',4),('A-DC2',2)]:
  params=[-.2]*n; rows[arm]=[{'epoch':1,'params':params,'mean_r2':.1,'per_session_r2':{'z':.1,'y':.1}},{'epoch':2,'params':[x+.1 for x in params],'mean_r2':.2,'per_session_r2':{'z':.2,'y':.2}}]
 return {'history':rows,'selected':{a:rows[a][1] for a in rows}}
def test_selected_params_restore_exact_and_reject_drift():
 s=_screen(); row=audit.selected_history_row(s,'A-TB4'); assert row['epoch']==2 and len(row['params'])==4
 audit.validate_restore(expected=row['params'],observed=row['params'])
 with pytest.raises(audit.AuditError):audit.validate_restore(expected=row['params'],observed=[0]*4)
def test_scalar_and_capacity_frozen_gates_and_tiebreak():
 z={'a':.1,'b':.1}; s={'a':.103,'b':.102}; c={'a':.107,'b':.106}
 scalar=audit.scalar_control(scalar=s,zero=z); assert scalar['both_nonnegative'] and scalar['within_v1_tolerance']
 gate=audit.capacity_gate(candidate=c,scalar=s,zero=z); assert gate['passed']
 assert audit.choose_winner({'A-TB4':gate,'A-DC2':gate})=='A-DC2'
def test_zero_anchor_dc2_permutation_and_coordinated_equivalence():
 torch.manual_seed(2); n=torch.randn(1,96,50); p=torch.randn(1,96,50); values=torch.randn(7,96,50)
 for arm in plan.ARMS:
  g=CapacityGate(arm,source_fit_mean=.1,source_fit_std=.2); z=(disagreement_statistic(values)-.1)/.2 if arm=='A-DC2' else None
  assert torch.equal(g.gate(z=z,native=n,post=p,training=False),n)
 assert torch.equal(disagreement_statistic(values),disagreement_statistic(values[torch.tensor([4,1,6,0,2,5,3])]))
 x=np.linspace(-1,1,200); audit.equivalence(coordinated_prediction=x,separate_prediction=x+1e-7,coordinated_r2=.2,separate_r2=.2000001)
 with pytest.raises(audit.AuditError):audit.equivalence(coordinated_prediction=x,separate_prediction=x+1e-3,coordinated_r2=.2,separate_r2=.2)
def test_worker_exposes_bound_v1_literals():
 from tfpd_exploration.src.m2_anchored_postfusion_capacity_audit_refit_v2 import worker
 assert worker.require_bound_v1_graph()==plan.V1_BODIES
