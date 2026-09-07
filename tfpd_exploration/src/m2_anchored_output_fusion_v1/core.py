"""Pure closed-form AOF algebra and static decision gate."""
from __future__ import annotations
import math
from typing import Mapping
import numpy as np
class AOFError(ValueError):pass
def _need(ok,msg):
 if not ok:raise AOFError(msg)
def beta_equal_session(pairs:Mapping[str,tuple[np.ndarray,np.ndarray,np.ndarray]], *, expected_sessions=None)->dict[str,object]:
 """Fit `beta` with declared-order float64, equal session MSE weighting."""
 if expected_sessions is None:
  ordered=tuple(sorted(pairs))
 else:
  ordered=tuple(expected_sessions);_need(tuple(pairs)==ordered and set(pairs)==set(ordered),'AOF fit lexical session authority drift')
 terms=[]
 for session in ordered:
  native,post,target=(np.asarray(v,dtype=np.float64) for v in pairs[session]); _need(native.shape==post.shape==target.shape and native.size>0,'AOF pair geometry')
  _need(np.isfinite(native).all() and np.isfinite(post).all() and np.isfinite(target).all(),'AOF pair finite authority')
  d=post-native;e=target-native;n=int(native.shape[0]); terms.append((session,n,float(np.sum(d*e,dtype=np.float64)/n),float(np.sum(d*d,dtype=np.float64)/n)))
 num=sum(x[2] for x in terms);den=sum(x[3] for x in terms);_need(math.isfinite(num) and math.isfinite(den) and den>0,'AOF beta denominator')
 beta=num/den;_need(math.isfinite(beta),'AOF beta nonfinite')
 scale=max(abs(num),abs(den),1.0)
 return {'beta':float(beta),'numerator':float(num),'denominator':float(den),'denominator_scale':float(scale),'conditioning_ratio':float(den/scale),'per_session':[{'session':s,'windows':n,'numerator':a,'denominator':b} for s,n,a,b in terms]}
def fuse(native,post,beta:float):
 if float(beta)==0. and not bool(np.signbit(np.float64(beta))):return native
 return native+float(beta)*(post-native)
def r2(target,pred):
 from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1.core import variance_weighted_r2
 return float(variance_weighted_r2(np.asarray(target),np.asarray(pred)))
def gate(rows:Mapping[str,tuple[np.ndarray,np.ndarray,np.ndarray]],beta:float, *, expected_sessions=None)->dict[str,object]:
 _need(len(rows)>0,'AOF validation has no sessions')
 if expected_sessions is not None:
  ordered=tuple(expected_sessions);_need(tuple(rows)==ordered and set(rows)==set(ordered),'AOF validation lexical session authority drift')
 else:
  ordered=tuple(sorted(rows))
 _need(math.isfinite(float(beta)),'AOF validation beta nonfinite')
 out={}
 for s in ordered:
  n,p,t=rows[s]
  _need(np.isfinite(n).all() and np.isfinite(p).all() and np.isfinite(t).all(),'AOF gate finite authority')
  zero=fuse(n,p,0.);_need(np.array_equal(zero,n),'AOF +0 native bitwise drift'); a=fuse(n,p,beta)
  native_r2,post_r2,aof_r2=r2(t,n),r2(t,p),r2(t,a)
  _need(math.isfinite(native_r2) and math.isfinite(post_r2) and math.isfinite(aof_r2),'AOF validation R2 nonfinite')
  out[s]={'native_r2':native_r2,'post_r2':post_r2,'aof_r2':aof_r2,'zero_prediction_exact':True,'beta':float(beta)}
 d={s:x['aof_r2']-x['native_r2'] for s,x in out.items()};mean=float(np.mean(list(d.values())))
 _need(np.isfinite(np.asarray(list(d.values()),dtype=np.float64)).all() and math.isfinite(mean),'AOF validation delta nonfinite')
 return {'rows':out,'mean_delta':mean,'positive_sessions':sum(v>0 for v in d.values()),'worst_delta':min(d.values()),'passed':mean>=.005 and sum(v>0 for v in d.values())==2 and min(d.values())>=.001}
