"""Label-separated data access for external gradient-free baselines."""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; SCRIPTS=ROOT/"learnable_recency_v1/scripts"
for p in (SCRIPTS,ROOT/'src',ROOT,ROOT.parent):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
# Prefer the source package over the workspace namespace directory.
sys.path.remove(str(ROOT/'src')); sys.path.insert(0,str(ROOT/'src'))
# The launcher may have first created a workspace namespace package.  Replace
# it with the installed source package before activity_data imports its config.
sys.modules.pop('btransform_unified_v2',None)
import activity_data

def load(task: str, *, evaluation: bool) -> dict[str,Any]:
    return activity_data.load_task_data(task,include_eval=evaluation)

def reference_source_day(train: Mapping[str,Any]) -> str:
    """Latest lexicographic ISO-like held-in session, recorded in the receipt."""
    if not train: raise ValueError("empty source surface")
    return max(train)

def causal_features(item: Mapping[str,Any], indices: np.ndarray, *, context: int, history: int=10) -> np.ndarray:
    """Endpoint causal bins, left-padding invalid bins with zero; never reads Y."""
    starts=np.asarray(item["starts"],np.int64)[np.asarray(indices,np.int64)]
    raw=np.asarray(item["X"],np.float32); out=np.zeros((len(starts),history*raw.shape[1]),np.float32)
    for i,s in enumerate(starts):
        end=int(s)+context; lo=max(int(s),end-history,int(item.get("pad",0)))
        take=raw[lo:end]
        out[i,-take.shape[0]*raw.shape[1]:]=take.reshape(-1)
    return out

def sha_array(a: Any)->str:
    x=np.ascontiguousarray(np.asarray(a)); h=hashlib.sha256(); h.update(x.dtype.str.encode()); h.update(str(x.shape).encode()); h.update(x.tobytes()); return h.hexdigest()
