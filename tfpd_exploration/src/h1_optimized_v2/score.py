"""Frozen source-only selection and distinct complete-stream evaluation."""
from __future__ import annotations
import numpy as np
import torch
from .cache import build_or_load
from .model import H1FullWindowControl, H1CurrentQueryDecoder
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

WINDOW=700; CHUNK=16
def _r2(p,y):
    denom=((y-y.mean(0))**2).sum()
    return float(1-((p-y)**2).sum()/denom) if denom else float('nan')
def _windows(a, ends):
    out=np.zeros((len(ends),WINDOW,a.shape[1]),np.float32)
    for i,e in enumerate(ends):
        take=min(int(e)+1,WINDOW);out[i,-take:]=a[int(e)+1-take:int(e)+1]
    return out
def evaluate(model, *, device, split='minival', mode='selection'):
    """selection=frozen W700/stride4 2908 endpoints; complete=all 20325 mask bins."""
    cache=build_or_load(); rows=cache[split]; ps=[]; ys=[]; per={}
    model.eval()
    with torch.inference_mode():
      for name,row in rows.items():
        ends=row['query_starts']+WINDOW-1 if mode=='selection' else np.flatnonzero(row['eval_mask'])
        bank=H1Bank(row['bank']['E0'].to(device),row['bank']['T'].to(device),row['bank']['unit_mask'].to(device))
        part=[]
        for off in range(0,len(ends),CHUNK):
          x=torch.as_tensor(_windows(row['neural'],ends[off:off+CHUNK]),device=device)
          part.append((model.forward_last(x,bank)/20.0).cpu().numpy())
        p=np.concatenate(part); y=row['velocity'][ends]
        ps.append(p);ys.append(y);per[name]=_r2(p,y)
    p=np.concatenate(ps); y=np.concatenate(ys)
    return {'split':split,'mode':mode,'n_bins':int(len(y)),'r2_concat':_r2(p,y),'equal_session_mean_r2':float(np.mean(list(per.values()))),
      'worst_session':min(per,key=per.get),'worst_session_r2':float(min(per.values())),'per_session_r2':per,
      'disclosure':'known-source development; not an official hidden/test score'}
