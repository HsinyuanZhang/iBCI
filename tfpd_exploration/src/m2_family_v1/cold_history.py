"""Pure deterministic cold-history prefix dropout; no data/model dependency."""
from __future__ import annotations
import hashlib
import torch

DOMAIN=b'm2_reset_prefix_v1'
def _seed(seed:int,epoch:int,batch_id:int)->int:
 if not all(isinstance(v,int) and v>=0 for v in (seed,epoch,batch_id)):raise ValueError('seed/epoch/batch_id must be nonnegative ints')
 return int.from_bytes(hashlib.sha256(DOMAIN+b'|'+f'{seed}|{epoch}|{batch_id}'.encode()).digest()[:8],'little')
def apply_prefix_dropout(x:torch.Tensor,*,seed:int,epoch:int,batch_id:int,probability:float=.5)->tuple[torch.Tensor,torch.Tensor]:
 _seed(seed,epoch,batch_id)  # validate counters even for the p=0 control.
 if x.dtype!=torch.float32 or x.ndim!=3 or tuple(x.shape[1:])!=(50,96):raise ValueError('expected finite FP32 [B,50,96]')
 if x.shape[0]==0:raise ValueError('empty batch')
 if not bool(torch.isfinite(x).all()):raise ValueError('nonfinite x')
 if probability not in (0.,.5):raise ValueError('probability must be 0 or .5')
 out=x.clone()
 if probability==0.:return out,torch.full((x.shape[0],),50,dtype=torch.int64,device=x.device)
 g=torch.Generator(device='cpu').manual_seed(_seed(seed,epoch,batch_id));chosen=torch.rand((x.shape[0],),generator=g)<.5;lengths=torch.full((x.shape[0],),50,dtype=torch.int64)
 if bool(chosen.any()):lengths[chosen]=torch.randint(1,50,(int(chosen.sum()),),generator=g)
 prefix=torch.arange(50).view(1,50)<(50-lengths).view(-1,1)
 out.masked_fill_(prefix.to(out.device).unsqueeze(-1),0.)
 return out,lengths.to(x.device)
