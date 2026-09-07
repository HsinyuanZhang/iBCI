"""Pure NumPy contract for chronological H1 public-runtime replay."""
from __future__ import annotations
import numpy as np
W=700
def raw_window(neural,end):
 z=np.zeros((W,neural.shape[1]),np.float32);n=min(end+1,W);z[-n:]=neural[end+1-n:end+1];return z
def _raw(runtime):
 x=runtime.raw
 if hasattr(x,'detach'):x=x.detach().cpu().numpy()
 return np.asarray(x)
def replay_session(runtime,neural,velocity,eval_mask,session,archive_slice,native_forward=None):
 neural=np.asarray(neural);velocity=np.asarray(velocity);mask=np.asarray(eval_mask);a=archive_slice
 if neural.dtype!=np.float32 or neural.ndim!=2 or velocity.shape!=(len(neural),7) or mask.dtype!=np.bool_ or mask.ndim!=1 or len(mask)!=len(neural):raise ValueError('geometry')
 ends=np.flatnonzero(mask)
 if a['prediction'].shape!=(len(ends),7) or a['target'].shape!=(len(ends),7) or len(a['end'])!=len(ends) or len(a['session_id'])!=len(ends) or not np.isfinite(a['prediction']).all():raise RuntimeError('archive shape')
 if not(np.array_equal(a['end'],ends) and np.all(a['session_id']==session) and np.array_equal(a['target'],velocity[ends])):raise RuntimeError('archive identity')
 preds=[];maxerr=0.;nativeerr=0.;direct=0;subset={i for i in (0,4,699,700,len(neural)-1) if 0<=i<len(neural)};row=0
 for i in range(len(neural)):
  pred=np.asarray(runtime.predict(np.ascontiguousarray(neural[i:i+1])));expect=raw_window(neural,i)
  if not np.array_equal(_raw(runtime)[0],expect):raise RuntimeError('runtime raw drift')
  if pred.shape!=(1,7) or pred.dtype!=np.float32 or not pred.flags.owndata or not np.isfinite(pred).all():raise RuntimeError('public prediction contract')
  if i in subset and native_forward is not None:
   q=np.asarray(native_forward(expect));q=q.reshape(1,7) if q.shape==(7,) else q
   if q.shape!=(1,7) or not np.isfinite(q).all():raise RuntimeError('native output contract')
   nativeerr=max(nativeerr,float(np.max(np.abs(q-pred))))
   if not np.allclose(q,pred,atol=1e-5,rtol=1e-5):raise RuntimeError('native mismatch')
   direct+=1
  if mask[i]:
   err=float(np.max(np.abs(pred-a['prediction'][row])));maxerr=max(maxerr,err)
   if not np.allclose(pred,a['prediction'][row],atol=1e-5,rtol=1e-5):raise RuntimeError('archive prediction')
   preds.append(pred.copy());row+=1
 return {'prediction':np.concatenate(preds) if preds else np.empty((0,7),np.float32),'max_abs_error':maxerr,'max_native_abs_error':nativeerr,'public_calls':len(neural),'scored_count':row,'direct_count':direct}
