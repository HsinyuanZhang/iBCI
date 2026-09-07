"""Pure W700 dense causal supervision helpers; no cache/model construction."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

POSITIONS=(349,466,582,699); WINDOW,UNITS,OUTPUTS=700,176,7

def forward_positions(model,x,bank,dropout_keep=None,positions=POSITIONS):
    positions=tuple(positions)
    if not positions or any(not isinstance(p,int) or not 0<=p<WINDOW for p in positions): raise RuntimeError('invalid dense positions')
    expected_units=getattr(getattr(model,'cfg',None),'n_units',UNITS)
    if x.ndim!=3 or x.shape[1:]!=(WINDOW,expected_units): raise RuntimeError('W700 input geometry required')
    z=model.encode_frontend(x,bank,dropout_keep); hidden=model.temporal(z)
    out=model.readout(model.final_norm(hidden[:,positions]))
    if out.shape!=(len(x),len(positions),OUTPUTS): raise RuntimeError('dense decoder output geometry drift')
    return out

def dense_targets(record,starts,positions=POSITIONS):
    starts=np.asarray(starts)
    velocity=np.asarray(record['velocity']); mask=np.asarray(record['eval_mask'])
    if starts.dtype!=np.int64 or starts.ndim!=1 or not len(starts) or len(np.unique(starts))!=len(starts) or np.any(starts<0): raise RuntimeError('int64 unique source starts required')
    if velocity.dtype!=np.float32 or velocity.ndim!=2 or velocity.shape[1]!=OUTPUTS or mask.dtype!=np.bool_ or mask.shape!=(len(velocity),) or not np.isfinite(velocity).all(): raise RuntimeError('source velocity/eval mask contract drift')
    positions=tuple(positions); idx=starts[:,None]+np.asarray(positions,dtype=np.int64)[None,:]
    in_range=(idx>=0)&(idx<len(velocity)); valid=np.zeros_like(in_range,dtype=bool);valid[in_range]=mask[idx[in_range]]
    if not np.all(valid[:,-1]): raise RuntimeError('every source endpoint must be eligible scored velocity')
    values=np.zeros((len(starts),len(positions),OUTPUTS),dtype=np.float32);values[valid]=velocity[idx[valid]]*20
    return torch.from_numpy(values),torch.from_numpy(valid)

def weighted_loss(prediction,target,valid,positions=POSITIONS):
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[-1]!=OUTPUTS or valid.shape!=prediction.shape[:2]: raise RuntimeError('dense loss geometry drift')
    weights=torch.full((len(positions),),1/len(positions),dtype=prediction.dtype,device=prediction.device)
    numerator,denom=weighted_terms(prediction,target,valid,positions)
    return numerator/denom

def weighted_terms(prediction,target,valid,positions=POSITIONS):
    """Return additive numerator/denominator for exact ragged microbatch scaling."""
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[-1]!=OUTPUTS or valid.shape!=prediction.shape[:2]: raise RuntimeError('dense loss geometry drift')
    weights=torch.full((len(positions),),1/len(positions),dtype=prediction.dtype,device=prediction.device)
    point=F.mse_loss(prediction,target.to(prediction),reduction='none').mean(-1); active=valid.to(prediction.device)
    if not bool(active[:,-1].all()): raise RuntimeError('endpoint supervision absent')
    denom=(active*weights).sum()
    if float(denom)==0: raise RuntimeError('no dense valid targets')
    return (point*active*weights).sum(),denom

def assert_last_parity(model,x,bank,dropout_keep=None):
    dense=forward_positions(model,x,bank,dropout_keep,(699,))[:,0]
    direct=model.forward_last(x,bank,dropout_keep)
    if not torch.equal(dense,direct): raise RuntimeError('position699 does not exactly equal forward_last')
    return dense
