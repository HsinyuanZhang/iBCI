"""V6 EMA with explicit immutable temporal-buffer integrity checks."""
from __future__ import annotations
import torch
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
class V6EMA(DecoderEMA):
 def __init__(self,module,decay):
  super().__init__(module,decay);self.fixed={n:b.detach().cpu().clone() for n,b in module.named_buffers() if 'temporal_contract_version' in n or 'recency_tau_bins' in n or 'recency_prior_by_age' in n or 'age_bucket_by_age' in n}
  if not self.fixed:raise RuntimeError('V6 immutable temporal buffers absent')
  self._check(module)
 def _check(self,module):
  got=dict(module.named_buffers())
  if set(got)<set(self.fixed) or any(not torch.equal(got[n].detach().cpu(),v) for n,v in self.fixed.items()):raise RuntimeError('V6 immutable temporal buffer drift')
 def update_after_step(self,module):self._check(module);super().update_after_step(module);self._check(module)
 def score_with_ema(self,module,fn):self._check(module);out=super().score_with_ema(module,fn);self._check(module);return out
 def checkpoint_state(self):
  p=super().checkpoint_state();p['immutable_temporal_buffers']={n:v.clone() for n,v in self.fixed.items()};return p
 def load_checkpoint_state(self,payload):
  saved=payload.get('immutable_temporal_buffers')
  if saved is None or set(saved)!=set(self.fixed) or any(not torch.equal(saved[n].cpu(),v) for n,v in self.fixed.items()):raise RuntimeError('V6 EMA immutable buffer receipt drift')
  super().load_checkpoint_state(payload)
