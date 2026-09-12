"""CPU-only unlabeled neural alignment baselines.

``AlignedFA`` uses scikit-learn's iterative FactorAnalysis implementation.  It
is a no-backprop practical implementation of the Aligned-FA procedure, not a
claim that sklearn's optimizer is byte-identical to Degenhart et al.'s FA fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time
import numpy as np
from sklearn.decomposition import FactorAnalysis


def _x(value: Any, name: str, *, calibration: bool=True) -> np.ndarray:
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]<(2 if calibration else 1) or a.shape[1]<1: raise ValueError(f"{name} must be [N,C]")
    if not np.isfinite(a).all(): raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(a)

def _sqrt_psd(a: np.ndarray, ridge: float, inverse: bool=False) -> np.ndarray:
    q,v=np.linalg.eigh((a+a.T)*.5 + np.eye(a.shape[0])*ridge)
    q=np.maximum(q,ridge)
    return (v * (q**(-.5) if inverse else q**.5)) @ v.T

@dataclass
class CoralAdapter:
    source_mean: np.ndarray; target_mean: np.ndarray; source_root: np.ndarray; target_invroot: np.ndarray; diagnostics: dict[str,Any]
    def transform(self, activity: Any) -> np.ndarray:
        x=_x(activity,"activity",calibration=False)
        if x.shape[1]!=self.source_mean.size: raise ValueError("channel mismatch")
        return np.asarray((x-self.target_mean)@self.target_invroot@self.source_root+self.source_mean,dtype=np.float32)

class CoralAligner:
    def __init__(self, source_activity: Any, *, ridge: float=1e-5, shrinkage: float=0.0):
        self.source=_x(source_activity,"source_activity"); self.ridge=float(ridge);self.shrinkage=float(shrinkage)
        if self.ridge<=0 or not 0<=self.shrinkage<=1: raise ValueError("ridge/shrinkage")
        self.source_mean=self.source.mean(0); self.source_cov=self._cov(self.source)
    def _cov(self,x):
        c=np.cov(x,rowvar=False,bias=False); c=np.atleast_2d(c); return (1-self.shrinkage)*c+self.shrinkage*np.eye(c.shape[0])*np.trace(c)/c.shape[0]
    def calibrate(self,target_activity: Any) -> CoralAdapter:
        t=_x(target_activity,"target_activity")
        if t.shape[1]!=self.source.shape[1]: raise ValueError("channel mismatch")
        tc=self._cov(t)
        return CoralAdapter(self.source_mean,t.mean(0),_sqrt_psd(self.source_cov,self.ridge),_sqrt_psd(tc,self.ridge,True),{"method":"CORAL","channels":int(t.shape[1]),"target_samples":int(t.shape[0]),"ridge":self.ridge,"shrinkage":self.shrinkage,"target_labels_used":False,"target_backpropagation":False})

@dataclass
class AlignedFAAdapter:
    target_fa: FactorAnalysis; rotation: np.ndarray; stable_rows: np.ndarray; diagnostics: dict[str,Any]
    def transform(self, activity: Any) -> np.ndarray:
        x=_x(activity,"activity",calibration=False)
        if x.shape[1]!=self.target_fa.components_.shape[1]: raise ValueError("channel mismatch")
        return np.asarray(self.target_fa.transform(x)@self.rotation,dtype=np.float32)

class AlignedFA:
    def __init__(self, source_activity: Any, *, latent_dim:int=10, seed:int=42, max_iter:int=200, tol:float=1e-3, stable_fraction:float=.5, n_init:int=1):
        self.source=_x(source_activity,"source_activity"); self.latent_dim=int(latent_dim);self.seed=int(seed);self.max_iter=int(max_iter);self.tol=float(tol);self.stable_fraction=float(stable_fraction);self.n_init=int(n_init)
        if not 1<=self.latent_dim<=min(self.source.shape): raise ValueError("latent_dim")
        if not 0<self.stable_fraction<=1 or self.n_init<1 or self.max_iter<1 or self.tol<=0: raise ValueError("FA hyperparameters")
        self.source_fa,self.source_fit=self._fit(self.source,0);self.source_loadings=self.source_fa.components_.T
    def _fit(self,x,offset):
        candidates=[]
        for i in range(self.n_init):
            rng=np.random.default_rng(self.seed+offset+i); init=np.maximum(np.var(x,axis=0)*(1+rng.normal(0,.01,x.shape[1])),1e-8)
            fa=FactorAnalysis(n_components=self.latent_dim,max_iter=self.max_iter,tol=self.tol,svd_method="lapack",noise_variance_init=init)
            z=fa.fit_transform(x)
            if not (np.isfinite(z).all() and np.isfinite(fa.components_).all() and np.isfinite(fa.noise_variance_).all()): raise FloatingPointError("nonfinite FA")
            candidates.append((float(fa.score(x)),fa))
        score,fa=max(candidates,key=lambda z:z[0]);return fa,{"log_likelihood_per_sample":score,"n_iter":int(fa.n_iter_),"converged":bool(fa.n_iter_<self.max_iter),"n_init":self.n_init,"implementation":"sklearn FactorAnalysis iterative SVD solver; no target backprop"}
    def source_transform(self,activity:Any)->np.ndarray:
        x=_x(activity,"activity",calibration=False);
        if x.shape[1]!=self.source.shape[1]:raise ValueError("channel mismatch")
        return np.asarray(self.source_fa.transform(x),dtype=np.float32)
    @staticmethod
    def _rotation(a,b):
        u,_,vt=np.linalg.svd(a.T@b,full_matrices=False);return u@vt
    def calibrate(self,target_activity:Any)->AlignedFAAdapter:
        t=_x(target_activity,"target_activity")
        if t.shape[1]!=self.source.shape[1]:raise ValueError("channel mismatch")
        started=time.monotonic();fa,fit=self._fit(t,10000); tl=fa.components_.T; keep=np.arange(t.shape[1]);wanted=max(self.latent_dim,int(np.ceil(t.shape[1]*self.stable_fraction)))
        # Degenhart-style iterative prune: align rows, discard the largest row residual.
        while len(keep)>wanted:
            r=self._rotation(tl[keep],self.source_loadings[keep]); err=np.sum((tl[keep]@r-self.source_loadings[keep])**2,1); keep=np.delete(keep,int(np.argmax(err)))
        r=self._rotation(tl[keep],self.source_loadings[keep])
        d={"method":"AlignedFA stable-loading-row orthogonal Procrustes","channels":int(t.shape[1]),"latent_dim":self.latent_dim,"target_samples":int(t.shape[0]),"stable_rows":keep.tolist(),"stable_fraction":self.stable_fraction,"source_fit":self.source_fit,"target_fit":fit,"seconds":time.monotonic()-started,"target_labels_used":False,"target_backpropagation":False,"procrustes":"closed-form SVD"}
        return AlignedFAAdapter(fa,r,keep,d)
