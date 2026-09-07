"""Exact optimized execution for the sealed M2 S1/S2 transformer weights.

This imports the sealed runtime only as an oracle/module definition.  It does
not edit or replace it.  Optimizations are: true multi-session frontend batch,
boundary-correct frontend cache, and last-query execution in temporal layer L.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tfpd_exploration.src.two_mainlines_long_v1.latency_opt_v2 import FrontendWindowCache
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.container_decoder import (
    CausalTransformerDecoder, SessionBank,
)


def _version(value: torch.Tensor) -> int:
    """Reject inference-mode tensors whose mutation version cannot be audited."""
    try:
        return int(value._version)
    except RuntimeError as exc:
        raise RuntimeError(
            "M2 optimized cache requires tensors created outside torch.inference_mode(); "
            "clone/load them normally before constructing the adapter"
        ) from exc


def stack_banks(banks: list[SessionBank], *, device: torch.device, dtype: torch.dtype) -> SessionBank:
    """Make independent sessions a batch without mixing identities or masks."""
    if not banks:
        raise ValueError("at least one session bank is required")
    n = banks[0].E0.shape[0]
    if any(b.E0.shape[0] != n or b.T.shape[0] != n for b in banks):
        raise ValueError("vectorized sessions require the same unit roster width")
    return SessionBank(
        E0=torch.stack([b.E0.to(device=device, dtype=dtype) for b in banks]),
        T=torch.stack([b.T.to(device=device, dtype=dtype) for b in banks]),
        unit_mask=torch.stack([b.unit_mask.to(device=device, dtype=torch.bool) for b in banks]),
    )


class M2ExactOptimizedDecoder:
    """Inference-only M2 adapter; call rebuild after reset/gap/mask/bank changes."""

    def __init__(self, model: CausalTransformerDecoder, banks: list[SessionBank]) -> None:
        self.model = model.eval()
        self.banks = banks
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self._bank_signature: tuple[object, ...] = ()
        self.bank = self._compile_banks(banks)
        self.front_cache = FrontendWindowCache(self._frontend, window=50, kernel=5)

    def _compile_banks(self, banks: list[SessionBank]) -> SessionBank:
        """Compile a cache key containing every bank/order-sensitive input."""
        signature: list[object] = [id(self.model), self.dtype, self.device]
        for bank in banks:
            for value in (bank.E0, bank.T, bank.unit_mask):
                signature.extend((value.data_ptr(), tuple(value.shape), value.dtype, value.device, _version(value)))
        for parameter in self.model.parameters():
            signature.extend((parameter.data_ptr(), tuple(parameter.shape), parameter.dtype, parameter.device, _version(parameter)))
        self._bank_signature = tuple(signature)
        return stack_banks(banks, device=self.device, dtype=self.dtype)

    def _ensure_fresh(self) -> None:
        """Invalidate on any in-place bank/mask/order/weight mutation.

        PyTorch's tensor version counter is incremented by ordinary in-place
        updates.  The tensor identity/shape/device terms also catch replacement,
        dtype, and roster changes.  Recompilation deliberately copies the new
        frozen-bank values before allowing a cached z sequence to be reused.
        """
        parameter = next(self.model.parameters())
        # Device/dtype migration is supported only as an invalidating event;
        # never reuse CPU cache tensors after `model.to(...)`.
        if parameter.device != self.device or parameter.dtype != self.dtype:
            self.device, self.dtype = parameter.device, parameter.dtype
            self.bank = self._compile_banks(self.banks)
            self.invalidate()
            return
        signature: list[object] = [id(self.model), self.dtype, self.device]
        for bank in self.banks:
            for value in (bank.E0, bank.T, bank.unit_mask):
                signature.extend((value.data_ptr(), tuple(value.shape), value.dtype, value.device, _version(value)))
        for parameter in self.model.parameters():
            signature.extend((parameter.data_ptr(), tuple(parameter.shape), parameter.dtype, parameter.device, _version(parameter)))
        if tuple(signature) != self._bank_signature:
            self.bank = self._compile_banks(self.banks)
            self.invalidate()

    def replace_banks(self, banks: list[SessionBank]) -> None:
        """Mandatory route for reset/bank/mask/unit-order changes; invalidates z."""
        self.banks = banks
        self.bank = self._compile_banks(banks)
        self.invalidate()

    def invalidate(self) -> None:
        self.front_cache.reset()

    def _frontend(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.frontend(x, self.bank, self.bank.unit_mask)

    def _temporal_last(self, z: torch.Tensor) -> torch.Tensor:
        h = z + self.model.temporal.pe[: z.size(1)].unsqueeze(0).to(device=z.device, dtype=z.dtype)
        blocks = self.model.temporal.blocks
        for block in blocks[:-1]:
            # The S1 training module returns (hidden, ephemeral_cache), while
            # the Falcon/LARGE oracle returns hidden.  Neither cache is kept.
            stepped = block(h)
            h = stepped[0] if isinstance(stepped, tuple) else stepped
        # Final self-attention needs all K/V but only the final Q/output row.
        block = blocks[-1]
        normed = block.norm1(h)
        attn = block.attn
        b, w, d = normed.shape
        qkv = attn.qkv(normed).view(b, w, 3, attn.n_heads, attn.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q[:, -1:].transpose(1, 2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        # The sole query is position w-1 and may attend every key 0..w-1.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = attn.proj(out.transpose(1, 2).contiguous().view(b, 1, d))
        last = h[:, -1:] + out
        last = last + block.ffn(block.norm2(last))
        return self.model.readout(self.model.final_norm(last))[:, 0, :]

    def rebuild(self, window: torch.Tensor) -> torch.Tensor:
        self._ensure_fresh()
        return self._temporal_last(self.front_cache.rebuild(window))

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        self._ensure_fresh()
        return self._temporal_last(self.front_cache.advance(next_bin))
