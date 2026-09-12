"""Fixed source-only causal ridge/Wiener decoder."""
from __future__ import annotations
import numpy as np

class WienerRidge:
    def __init__(self, ridge: float=1.0): self.ridge=float(ridge)
    def fit(self,x:np.ndarray,y:np.ndarray):
        x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
        self.mean_=x.mean(0); self.scale_=x.std(0); self.scale_[self.scale_<1e-6]=1.
        z=(x-self.mean_)/self.scale_; z=np.c_[z,np.ones(len(z))]
        reg=np.eye(z.shape[1])*self.ridge; reg[-1,-1]=0.
        self.coef_=np.linalg.solve(z.T@z+reg,z.T@y); return self
    def predict(self,x:np.ndarray)->np.ndarray:
        z=(np.asarray(x,np.float64)-self.mean_)/self.scale_; return (np.c_[z,np.ones(len(z))]@self.coef_).astype(np.float32)
