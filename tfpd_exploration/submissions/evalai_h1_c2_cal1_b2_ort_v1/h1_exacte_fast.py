"""Fast exact-E runtime for H1 proj_add P16 L=200 (same weights, same operators).

Port of the sealed M1 Runtime-line optimization
(``m1_projadd_runtime_v1/20260907T020150Z/scripts/m1_exacte_fast.py``) onto H1
geometry (submission 582025): L=200, N=176, P(proj_out)=16, token_in=20,
out_dim=7, identity_dim=700. NO semantic change vs the frozen packed decoder
(``h1_trf_falcon_decoder.H1ProjAddFalconDecoder``): same model classes, same
weights, same observation/predict protocol, same window/boundary semantics
(k=5 causal-conv left-boundary recompute kept), same last-layer last-query
trick, no cross-window temporal KV reuse.

What is optimized (mirrors the M1 recipe):
  1. advance() runs the causal conv ONCE over the full [B,200,176] window
     and feeds only the 5 genuinely-changed positions (window-start
     boundary 0..3 + the new bin) through ONE batched set-attention call.
     The packed reference (``_FrontendWindowCache.advance``) instead calls
     the full frontend twice (4-position left call + 5-position right call
     that discards 4) -- same receptive fields, more redundant attention
     work. Receptive fields are identical (conv is position-wise causal;
     boundary zero-padding semantics preserved).
  2. Static folding per reset: P(E0) = e0_proj(E0), carrier T4 expansion,
     normalized slots are computed once per active-bank batch instead of
     per call; the packed payload's unit masks are all-true for all 27
     H1 dataset tags (verified), so the always-on key_padding_mask branch
     in the packed frontend is dropped when true, which is bit-exact (mask
     adds 0.0 to unmasked scores; softmax over an all-False pad matches the
     unmasked softmax).
  3. Preallocated rolling raw/z buffers (no cat/clone churn).
  4. Last temporal block computes only K,V for the full window and Q for
     the last position (packed reference computes Q for all 200 then
     discards 199 via ``[:, -1:]`` at the very end).

Equivalence is enforced by the external FP32 gate (1e-5 + 1e-5*|ref|) against
the frozen packed decoder (``h1_trf_falcon_decoder.py``, submission 582025).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent


def _load_packed_module():
    """Import the frozen packed H1 decoder module (read-only) for its classes."""
    import importlib.util

    packed = Path(
        os.environ.get(
            "RT_PACKED_DECODER",
            "/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/"
            "evalai_h1_projadd_exacte_v1/h1_trf_falcon_decoder.py",
        )
    )
    spec = importlib.util.spec_from_file_location("h1_trf_falcon_decoder_frozen", packed)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h1_trf_falcon_decoder_frozen"] = mod
    spec.loader.exec_module(mod)
    return mod


tfd = _load_packed_module()


class FastExactEEngine:
    """Batched exact-E engine; drop-in for the packed ``_ExactEEngine``.

    Generalized over ``window`` (L) and ``n_units`` (N) -- the packed M1
    template hardcoded L=100, N=64; this reads both from the bank / model
    config at construction time instead.
    """

    def __init__(self, model, banks, window: int) -> None:
        self.model = model
        self.bank = tfd.stack_banks(banks)
        cfg = model.cfg
        self.window = int(window)
        self.kernel = cfg.conv_kernel
        self.device = next(model.parameters()).device
        B = self.bank.E0.shape[0]
        n_units = self.bank.E0.shape[1]
        # --- static folds (per active-bank batch) -------------------------
        e0 = self.bank.E0.to(device=self.device, dtype=torch.float32)
        t4 = self.bank.T.to(device=self.device, dtype=torch.float32)
        if cfg.proj_out % cfg.conv_channels != 0:
            raise ValueError("proj_out must be a multiple of conv_channels")
        n_groups = cfg.proj_out // cfg.conv_channels
        self.proj_static = model.frontend.e0_proj(e0).view(B, 1, n_units, n_groups, cfg.conv_channels)
        self.t4_static = t4.view(B, 1, n_units, cfg.t4_dim)
        self.slots_static = model.frontend.slot_norm(model.frontend.slots).view(1, 1, cfg.slots, cfg.set_dim)
        keep = self.bank.unit_mask
        if keep.dim() == 1:
            keep = keep.unsqueeze(0).expand(B, -1)
        self.keep_all_true = bool(keep.all())
        self.keep_expanded = keep.contiguous() if not self.keep_all_true else None
        self.raw = None
        self.z = None

    # -- full-window frontend (used at rebuild; mirrors SharedSetFrontend) --
    def _frontend_full(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.model.cfg
        batch, width, n_units = x.shape
        local = self.model.frontend.local_conv(x)
        fused = (local.unsqueeze(-2) + self.proj_static).reshape(batch, width, n_units, cfg.proj_out)
        tokens = self.model.frontend.token_norm(
            self.model.frontend.token_mlp(
                torch.cat([fused, self.t4_static.expand(batch, width, n_units, cfg.t4_dim)], dim=-1)
            )
        )
        slots = self.slots_static.expand(batch, width, cfg.slots, cfg.set_dim)
        q = slots.reshape(batch * width, cfg.slots, cfg.set_dim)
        k = tokens.reshape(batch * width, n_units, cfg.set_dim)
        if self.keep_all_true:
            attn_out, _ = self.model.frontend.mha(q, k, k, need_weights=False)
        else:
            pad = (~self.keep_expanded).unsqueeze(1).expand(batch, width, n_units).reshape(batch * width, n_units)
            attn_out, _ = self.model.frontend.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attn_out
        slots_out = slots_out + self.model.frontend.slot_ffn(self.model.frontend.slot_ffn_norm(slots_out))
        return self.model.frontend.slot_proj(slots_out.reshape(batch, width, cfg.slots * cfg.set_dim))

    # -- set-attention over a small set of window positions ---------------
    def _set_attention(self, local_sel: torch.Tensor) -> torch.Tensor:
        """local_sel: [B, P, n_units, conv_channels] conv outputs at P positions."""
        cfg = self.model.cfg
        batch, P, n_units, _ = local_sel.shape
        fused = (local_sel.unsqueeze(-2) + self.proj_static).reshape(batch, P, n_units, cfg.proj_out)
        tokens = self.model.frontend.token_norm(
            self.model.frontend.token_mlp(
                torch.cat([fused, self.t4_static.expand(batch, P, n_units, cfg.t4_dim)], dim=-1)
            )
        )
        slots = self.slots_static.expand(batch, P, cfg.slots, cfg.set_dim)
        q = slots.reshape(batch * P, cfg.slots, cfg.set_dim)
        k = tokens.reshape(batch * P, n_units, cfg.set_dim)
        if self.keep_all_true:
            attn_out, _ = self.model.frontend.mha(q, k, k, need_weights=False)
        else:
            pad = (~self.keep_expanded).unsqueeze(1).expand(batch, P, n_units).reshape(batch * P, n_units)
            attn_out, _ = self.model.frontend.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attn_out
        slots_out = slots_out + self.model.frontend.slot_ffn(self.model.frontend.slot_ffn_norm(slots_out))
        return self.model.frontend.slot_proj(slots_out.reshape(batch, P, cfg.slots * cfg.set_dim))

    # -- temporal stack with last-query last block -------------------------
    def _temporal_last(self, z: torch.Tensor) -> torch.Tensor:
        model = self.model
        hidden = z + model.temporal.pe[: z.size(1)].unsqueeze(0).to(device=z.device, dtype=z.dtype)
        for block in model.temporal.blocks[:-1]:
            hidden = block(hidden)
        block = model.temporal.blocks[-1]
        normed = block.norm1(hidden)
        attn = block.attn
        batch, width, dim = normed.shape
        qkv = attn.qkv(normed).view(batch, width, 3, attn.n_heads, attn.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query[:, -1:].transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        attn_out = attn.proj(attn_out.transpose(1, 2).contiguous().view(batch, 1, dim))
        last = hidden[:, -1:] + attn_out
        last = last + block.ffn(block.norm2(last))
        return model.readout(model.final_norm(last))[:, 0, :]

    # -- protocol entry points ---------------------------------------------
    def rebuild(self, window: torch.Tensor) -> torch.Tensor:
        self.raw = window.detach().clone()
        self.z = self._frontend_full(self.raw).detach().clone()
        return self._temporal_last(self.z)

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        raw = self.raw
        z_prev = self.z
        conv = self.model.frontend.local_conv
        k = self.kernel
        # roll raw buffer in place (no cat/clone)
        raw.copy_(torch.roll(raw, shifts=-1, dims=1))
        raw[:, -1] = next_bin[:, 0]
        # boundary convs only (identical receptive fields as the packed
        # left/right calls): window-start 0..3 with zero pad, and the last
        # position with its k-bin context.
        left = conv(raw[:, : k - 1])                    # [B, 4, N, conv_channels]
        right = conv(raw[:, -k:])[:, -1:]                # [B, 1, N, conv_channels]
        local_sel = torch.cat((left, right), dim=1)      # [B, 5, N, conv_channels]
        z5 = self._set_attention(local_sel)              # [B, 5, temporal_width]
        # assemble z: positions 0..3 fresh, 4..window-2 reused from prev, window-1 fresh
        z = torch.empty_like(z_prev)
        z[:, :4] = z5[:, :4]
        z[:, 4:-1] = z_prev[:, 5:]
        z[:, -1] = z5[:, 4]
        self.z = z
        return self._temporal_last(z)


class FastH1ProjAddFalconDecoder(tfd.H1ProjAddFalconDecoder):
    """Protocol-identical decoder with the fast engine.

    ``observe()`` is inherited unchanged from the packed class (H1 has no
    ``smooth_observations`` branch in its observe path, unlike M1). Only
    ``predict()`` is overridden to route through ``FastExactEEngine``.
    """

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if not self.local_banks:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        if self._kv_cache_enabled:
            raise RuntimeError("cross-window KV cache is forbidden without a written parity proof")
        active = int(np.asarray(neural_observations).shape[0])
        self.observe(neural_observations)
        n_active = len(self.local_banks)
        decoder_input = torch.as_tensor(
            np.ascontiguousarray(self.observation_buffer[:, :n_active, :].transpose(1, 0, 2)),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            if self._engine is None:
                self._engine = FastExactEEngine(self.decoder, self.local_banks, self.window_size)
                prediction = self._engine.rebuild(decoder_input)
            else:
                prediction = self._engine.advance(decoder_input[:, -1:, :])
        native = prediction.detach().cpu().numpy() / self.behavior_scaling_factor
        if not np.isfinite(native).all():
            raise RuntimeError("non-finite H1 proj_add prediction")
        if n_active < self.batch_size:
            pad = np.zeros((self.batch_size - n_active, native.shape[1]), dtype=np.float32)
            native = np.concatenate([native, pad], axis=0)
        self._n_predicts += 1
        if self._n_predicts % 2000 == 0:
            print(f"h1-fast-e predict_steps={self._n_predicts} n_active={n_active}", flush=True)
        return native[:active].astype(np.float32, copy=False)
