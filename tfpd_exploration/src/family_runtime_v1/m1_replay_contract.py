"""Pure NumPy M1 W100 chronological replay contract."""
from __future__ import annotations
import numpy as np
W=100
def _raw(x):
 if hasattr(x,'detach'):x=x.detach().cpu().numpy()
 return np.asarray(x)
def _check(p):
 p=np.asarray(p)
 if p.shape!=(1,16) or p.dtype!=np.float32 or not p.flags.owndata or not np.isfinite(p).all():raise RuntimeError('public output')
 return p
def replay_session(runtime,raw,session,archive,native_forward,*,native_targets=None,initialize_history=None,current_prediction=None,progress=None):
 raw=np.asarray(raw);a=archive;starts=np.asarray(a['start']);n=len(starts)
 if raw.dtype!=np.float32 or raw.ndim!=2 or raw.shape[1]!=64 or starts.dtype!=np.int64 or starts.ndim!=1 or not n or not np.all(np.diff(starts)>0):raise ValueError('metadata')
 if len(a['session'])!=n or not np.all(a['session']==session) or np.asarray(a['prediction']).shape!=(n,16) or np.asarray(a['target']).shape!=(n,16):raise ValueError('archive')
 if native_targets is not None and not np.array_equal(a['target'],native_targets):raise RuntimeError('target identity')
 first,last=int(starts[0]),int(starts[-1]);
 if initialize_history is None or current_prediction is None or first<0 or last+W>len(raw):raise ValueError('callbacks/window')
 initialize_history(raw[None,first:first+W].copy());expect0=raw[first:first+W]
 if _raw(runtime.raw).shape!=(1,W,64) or not np.array_equal(_raw(runtime.raw)[0],expect0):raise RuntimeError('initial raw drift')
 init=_check(current_prediction());
 if not np.allclose(init,a['prediction'][0],atol=1e-5,rtol=1e-5):raise RuntimeError('initial archive')
 base=first+W-1;direct={i for i in (base,base+1,base+4,base+99,base+100,last+W-1) if base<=i<=last+W-1};row=1;calls=0;maxerr=float(np.max(np.abs(init-a['prediction'][0])));nativeerr=0.;preds=[init.copy()];direct_count=0
 q=_check(native_forward(expect0[None].copy()));nativeerr=float(np.max(np.abs(q-init)));direct_count=1
 if not np.allclose(q,init,atol=1e-5,rtol=1e-5):raise RuntimeError('native mismatch')
 for end in range(first+W,last+W):
  p=_check(runtime.predict(np.ascontiguousarray(raw[end:end+1])));calls+=1;expect=raw[end-W+1:end+1]
  if not np.array_equal(_raw(runtime.raw)[0],expect):raise RuntimeError('raw drift')
  if end in direct:
   q=_check(native_forward(expect[None].copy()));nativeerr=max(nativeerr,float(np.max(np.abs(q-p))));direct_count+=1
   if not np.allclose(q,p,atol=1e-5,rtol=1e-5):raise RuntimeError('native mismatch')
  start=end-W+1
  if row<n and start==starts[row]:
   e=float(np.max(np.abs(p-a['prediction'][row])));maxerr=max(maxerr,e)
   if not np.allclose(p,a['prediction'][row],atol=1e-5,rtol=1e-5):raise RuntimeError('archive prediction')
   preds.append(p.copy());row+=1
  if progress and (calls%1024==0 or end==last+W-1):progress({'public_calls':calls,'end':end,'session':session})
 if row!=n:raise RuntimeError('missed scored endpoint')
 return {'prediction':np.concatenate(preds),'public_calls':calls,'initial_current_predictions':1,'scored_count':n,'max_abs_error':maxerr,'max_native_abs_error':nativeerr,'direct_count':direct_count}
