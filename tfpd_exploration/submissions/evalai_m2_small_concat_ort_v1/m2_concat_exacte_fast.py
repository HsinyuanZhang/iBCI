"""Fast exact-E runtime for M2 SMALL concat (sealed 581973 weights).

Port of the M1 Runtime-line optimizations onto concat geometry. NOT proj_add:
tokens = concat(local16, E0 50, T4 4) -> token_in=70. There is no e0_proj add
and no grouped P. Same sealed SMALL classes/weights/protocol as
``trf_falcon_decoder.py`` (EvalAI 581973). Same window/boundary semantics
(k=5 causal-conv left-boundary recompute), last-layer last-query, no
cross-window temporal KV reuse.

What is optimized:
  1. advance() runs the causal conv ONCE over the full [B,50,96] window and
     feeds only the 5 changed positions (0..3 + last) through one set-attn.
  2. Static folding per reset: E0 / T4 / normalized slots computed once;
     all-true unit masks drop key_padding_mask (bit-exact).
  3. Preallocated rolling raw/z buffers.
  4. Last temporal block: K,V full window, Q last position only.
  5. observe() skips dead raw-history roll when smooth_observations is False.

Equivalence is enforced by the host FP32 gate (max_abs <= 1e-5) against the
frozen packed adapter.
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
    import importlib.util

    packed = Path(
        os.environ.get(
            "RT_PACKED_DECODER",
            str(_HERE / "trf_falcon_decoder.py"),
        )
    )
    spec = importlib.util.spec_from_file_location("trf_falcon_decoder_frozen_m2_concat", packed)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trf_falcon_decoder_frozen_m2_concat"] = mod
    spec.loader.exec_module(mod)
    return mod


tfd = _load_packed_module()


class FastExactEEngine:
    """Batched exact-E engine for concat SMALL; drop-in for ``_ExactEEngine``."""

    def __init__(self, model, banks) -> None:
        self.model = model
        self.bank = tfd.stack_banks(banks)
        cfg = model.cfg
        if int(cfg.token_in) != int(cfg.conv_channels + cfg.identity_dim + cfg.t4_dim):
            raise ValueError(
                f"concat token_in={cfg.token_in} != local{cfg.conv_channels}+E0{cfg.identity_dim}+T4{cfg.t4_dim}"
            )
        if hasattr(cfg, "proj_out") or hasattr(model.frontend, "e0_proj"):
            raise ValueError("refusing proj_add geometry; this engine is concat SMALL only")
        self.window = tfd.WINDOW
        self.kernel = cfg.conv_kernel
        self.device = next(model.parameters()).device
        B = self.bank.E0.shape[0]
        n_units = self.bank.E0.shape[1]
        e0 = self.bank.E0.to(device=self.device, dtype=torch.float32)
        t4 = self.bank.T.to(device=self.device, dtype=torch.float32)
        self.e0_static = e0.view(B, 1, n_units, cfg.identity_dim)
        self.t4_static = t4.view(B, 1, n_units, cfg.t4_dim)
        self.slots_static = model.frontend.slot_norm(model.frontend.slots).view(1, 1, cfg.slots, cfg.set_dim)
        keep = self.bank.unit_mask
        if keep.dim() == 1:
            keep = keep.unsqueeze(0).expand(B, -1)
        self.keep_all_true = bool(keep.all())
        self.keep_expanded = keep.contiguous() if not self.keep_all_true else None
        self.raw = None
        self.z = None
        self.needs = torch.tensor(
            list(range(self.kernel - 1)) + [self.window - 1], dtype=torch.long, device=self.device
        )

    def _tokens(self, local: torch.Tensor) -> torch.Tensor:
        """local: [B, P, N, conv_channels] -> tokens [B, P, N, set_dim]."""
        fe = self.model.frontend
        batch, width, n_units, _ = local.shape
        e0 = self.e0_static.expand(batch, width, n_units, self.e0_static.shape[-1])
        t4 = self.t4_static.expand(batch, width, n_units, self.t4_static.shape[-1])
        return fe.token_norm(fe.token_mlp(torch.cat([local, e0, t4], dim=-1)))

    def _set_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        cfg = self.model.cfg
        fe = self.model.frontend
        batch, width, n_units, _ = tokens.shape
        slots = self.slots_static.expand(batch, width, cfg.slots, cfg.set_dim)
        q = slots.reshape(batch * width, cfg.slots, cfg.set_dim)
        k = tokens.reshape(batch * width, n_units, cfg.set_dim)
        if self.keep_all_true:
            attn_out, _ = fe.mha(q, k, k, need_weights=False)
        else:
            pad = (~self.keep_expanded).unsqueeze(1).expand(batch, width, n_units).reshape(batch * width, n_units)
            attn_out, _ = fe.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attn_out
        slots_out = slots_out + fe.slot_ffn(fe.slot_ffn_norm(slots_out))
        return fe.slot_proj(slots_out.reshape(batch, width, cfg.slots * cfg.set_dim))

    def _frontend_full(self, x: torch.Tensor) -> torch.Tensor:
        local = self.model.frontend.local_conv(x)
        return self._set_from_tokens(self._tokens(local))

    def _set_attention(self, local_sel: torch.Tensor) -> torch.Tensor:
        return self._set_from_tokens(self._tokens(local_sel))

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

    def rebuild(self, window: torch.Tensor) -> torch.Tensor:
        self.raw = window.detach().clone()
        self.z = self._frontend_full(self.raw).detach().clone()
        return self._temporal_last(self.z)

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        raw = self.raw
        z_prev = self.z
        conv = self.model.frontend.local_conv
        k = self.kernel
        raw.copy_(torch.roll(raw, shifts=-1, dims=1))
        raw[:, -1] = next_bin[:, 0]
        local_full = conv(raw)
        local_sel = local_full[:, self.needs]
        z5 = self._set_attention(local_sel)
        z = torch.empty_like(z_prev)
        z[:, : k - 1] = z5[:, : k - 1]
        z[:, k - 1 : -1] = z_prev[:, k:]
        z[:, -1] = z5[:, k - 1]
        self.z = z
        return self._temporal_last(z)


class FastTrfFalconDecoder(tfd.TrfFalconDecoder):
    """Protocol-identical decoder with the concat fast engine."""

    def observe(self, neural_observations: np.ndarray):
        observations = np.ascontiguousarray(neural_observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.task_config.n_channels:
            raise ValueError(
                f"Expected neural observations [B,{self.task_config.n_channels}], got {observations.shape}"
            )
        if observations.shape[0] > self.batch_size:
            raise ValueError("Evaluator batch exceeds configured decoder batch size")
        if observations.shape[0] < self.batch_size:
            observations = np.pad(
                observations, ((0, self.batch_size - observations.shape[0]), (0, 0))
            )
        if self.smooth_observations:
            self.raw_history_buffer = np.roll(self.raw_history_buffer, -1, axis=0)
            self.raw_history_buffer[-1] = observations
            history, batch, channels = self.raw_history_buffer.shape
            flat = self.raw_history_buffer.reshape(history, batch * channels)
            from falcon_challenge.filtering import NEURAL_TAU_MS, apply_exponential_filter  # type: ignore

            smoothed = apply_exponential_filter(flat, tau=NEURAL_TAU_MS, bin_size=self.task_config.bin_size_ms)
            latest = smoothed[-1].reshape(batch, channels).astype(np.float32)
        else:
            latest = observations
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = latest

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if not self.local_banks:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        if self._kv_cache_enabled:
            raise RuntimeError("cross-window KV cache is forbidden without a written parity proof")
        self.observe(neural_observations)
        n_active = len(self.local_banks)
        decoder_input = torch.as_tensor(
            np.ascontiguousarray(self.observation_buffer[:, :n_active, :].transpose(1, 0, 2)),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            if self._engine is None:
                self._engine = FastExactEEngine(self.decoder, self.local_banks)
                prediction = self._engine.rebuild(decoder_input)
            else:
                prediction = self._engine.advance(decoder_input[:, -1:, :])
        native = prediction.detach().cpu().numpy() / self.behavior_scaling_factor
        if not np.isfinite(native).all():
            raise RuntimeError("non-finite Transformer prediction")
        if n_active < self.batch_size:
            pad = np.zeros((self.batch_size - n_active, native.shape[1]), dtype=np.float32)
            native = np.concatenate([native, pad], axis=0)
        self._n_predicts += 1
        if self._n_predicts % 2000 == 0:
            print(f"fast-e concat predict_steps={self._n_predicts} n_active={n_active}", flush=True)
        return native.astype(np.float32, copy=False)
