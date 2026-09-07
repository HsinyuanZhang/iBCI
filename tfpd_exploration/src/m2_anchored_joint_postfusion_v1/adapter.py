"""Route-owned post-fusion adapters installed only after strict native load."""
from __future__ import annotations

from typing import Any


class AdapterError(RuntimeError):
    pass


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise AdapterError(message)


def install_after_strict_load(student: Any, arm: str) -> Any:
    """Install a fresh AJPF adapter without touching shared streaming sources."""
    import torch
    import torch.nn as nn
    _need(arm in ("J-R1", "J-MEAN"), "AJPF adapter only serves post-fusion arms")
    native = student.id_encoder

    class AJPFIdentityAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.native = native
            self.arm = arm
            self.variant = getattr(native, "variant", None)
            self.trial_length = getattr(native, "trial_length", None)
            self.window_size = getattr(native, "window_size", None)
            self.hidden_dim = getattr(native, "hidden_dim", None)
            self.side_dim = getattr(native, "side_dim", None)
            self.electrode_embed_dim = getattr(native, "electrode_embed_dim", 0)
            if arm == "J-R1":
                self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
            else:
                self.register_parameter("alpha", None)

        def _post(self, calibration: Any, side_features: Any) -> Any:
            _need(side_features is not None and calibration.ndim == 4, "AJPF post-fusion input topology")
            values = self.native.pre_pool(calibration.permute(0, 1, 3, 2))
            side = side_features.unsqueeze(1).expand(-1, values.shape[1], -1, -1)
            pooled = self.native.post_pool(torch.cat((values, side), dim=-1))
            total = pooled[:, 0]
            for index in range(1, int(pooled.shape[1])):
                total = total + pooled[:, index]
            return total / int(pooled.shape[1])

        def forward_batch(self, calibration: Any, trial_lengths: Any = None, side_features: Any = None,
                          electrode_ids: Any = None) -> Any:
            native_identity = self.native.forward_batch(calibration, trial_lengths=trial_lengths,
                                                        side_features=side_features, electrode_ids=electrode_ids)
            post = self._post(calibration, side_features)
            if self.arm == "J-MEAN":
                return post
            return native_identity + torch.tanh(self.alpha) * (post - native_identity)

        def __getattr__(self, name: str) -> Any:
            # Preserve the native streaming interface only where model setup
            # asks for it; continual scoring remains route-owned.
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.native, name)

    adapter = AJPFIdentityAdapter()
    student.id_encoder = adapter
    return adapter


def require_positive_zero(alpha: Any) -> None:
    import torch
    _need(tuple(alpha.shape) == () and bool(torch.equal(alpha.detach(), torch.zeros_like(alpha)))
          and not bool(torch.signbit(alpha.detach()).item()), "AJPF alpha is not IEEE +0")
