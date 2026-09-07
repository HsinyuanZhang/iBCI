"""Pure APFC V2 restoration, control, and source-gate codecs."""
from __future__ import annotations
import math
from typing import Mapping,Sequence
from . import plan
class AuditError(ValueError):pass
def _need(ok,msg):
 if not ok:raise AuditError(msg)
def _finite(values):return all(math.isfinite(float(x)) for x in values)
def selected_history_row(screen:Mapping[str,object],arm:str)->Mapping[str,object]:
 _need(arm in plan.ARMS,'V2 unknown APFC arm')
 history=screen.get('history'); selected=screen.get('selected')
 _need(isinstance(history,Mapping) and isinstance(selected,Mapping),'V2 V1 screen history missing')
 rows=history.get(arm); best=selected.get(arm); _need(isinstance(rows,Sequence) and isinstance(best,Mapping),'V2 selected arm evidence missing')
 epoch=int(best.get('epoch',-1)); hits=[r for r in rows if isinstance(r,Mapping) and int(r.get('epoch',-2))==epoch]
 _need(len(hits)==1 and list(hits[0].get('params',()))==list(best.get('params',())),'V2 selected history vector drift')
 params=tuple(float(x) for x in hits[0]['params']); expected={'A-S1':1,'A-TB4':4,'A-DC2':2}[arm]
 _need(len(params)==expected and _finite(params),'V2 selected parameter geometry drift')
 return {'arm':arm,'epoch':epoch,'params':params,'v1_mean_r2':float(hits[0]['mean_r2']),'v1_per_session_r2':dict(hits[0]['per_session_r2'])}
def validate_restore(*,expected:Sequence[float],observed:Sequence[float])->None:
 _need(tuple(float(x) for x in expected)==tuple(float(x) for x in observed),'V2 restored selected gate vector drift')
def scalar_control(*,scalar:Mapping[str,float],zero:Mapping[str,float])->dict[str,object]:
 _need(set(scalar)==set(zero) and len(scalar)==2,'V2 scalar validation roster drift')
 d={s:float(scalar[s])-float(zero[s]) for s in sorted(scalar)}; mean=sum(d.values())/2
 _need(_finite(d.values()),'V2 scalar nonfinite')
 return {'per_session_delta':d,'mean_gain':mean,'both_nonnegative':all(x>=0 for x in d.values()),
         'within_v1_tolerance':abs(mean-plan.V1_SCALAR_GAIN)<=.001}
def capacity_gate(*,candidate:Mapping[str,float],scalar:Mapping[str,float],zero:Mapping[str,float])->dict[str,object]:
 _need(set(candidate)==set(scalar)==set(zero) and len(candidate)==2,'V2 capacity validation roster drift')
 ds={s:float(candidate[s])-float(scalar[s]) for s in sorted(candidate)}; dz={s:float(candidate[s])-float(zero[s]) for s in sorted(candidate)}
 mean_s=sum(ds.values())/2; mean_z=sum(dz.values())/2
 out={'per_session_vs_scalar':ds,'per_session_vs_zero':dz,'mean_vs_scalar':mean_s,'mean_vs_zero':mean_z,'positive_sessions':sum(x>0 for x in ds.values()),'worst_vs_scalar':min(ds.values())}
 out['passed']=mean_s>=.003 and out['positive_sessions']==2 and out['worst_vs_scalar']>=-.002 and mean_z>=.005
 return out
def choose_winner(gates:Mapping[str,Mapping[str,object]])->str|None:
 good=[a for a in ('A-TB4','A-DC2') if bool(gates.get(a,{}).get('passed'))]
 if not good:return None
 if len(good)==1:return good[0]
 a,b=good; da=float(gates[a]['mean_vs_scalar']); db=float(gates[b]['mean_vs_scalar'])
 return 'A-DC2' if abs(da-db)<.001 else (a if da>db else b)
def equivalence(*,coordinated_prediction,separate_prediction,coordinated_r2:float,separate_r2:float)->dict[str,float]:
 import numpy as np
 p=float(np.max(np.abs(np.asarray(coordinated_prediction)-np.asarray(separate_prediction))))
 r=abs(float(coordinated_r2)-float(separate_r2)); _need(p<=plan.PREDICTION_TOLERANCE and r<=plan.R2_TOLERANCE,'V2 coordinated/separate equivalence drift')
 return {'prediction_max_abs':p,'r2_abs':r}
