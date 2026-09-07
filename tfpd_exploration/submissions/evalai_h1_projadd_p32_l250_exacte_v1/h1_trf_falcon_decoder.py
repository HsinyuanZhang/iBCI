"""Self-contained Falcon runtime for H1 proj_add P32 + exact-E.

Identity: tokens = token_mlp([local16+P_g | ... | carrier4]), token_in=36,
P = Linear(700->32, bias=False), two 16-d groups added onto local conv.
Exact-E is causal-k5 frontend reuse plus last-layer last-query.
Window comes from the payload (200 or 250).
Does not import tfpd_exploration. Not SPINT. Separate dest from P16 e24.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from falcon_challenge.interface import BCIDecoder

PAYLOAD_SCHEMA = "h1_projadd_exacte_falcon_payload_v1"
CHANNELS = 176
BEHAVIOR_SCALE = 20.0
EXPECTED_SESSION_COUNT = 27
OFFICIAL_BATCH = 8
IDENTITY_DIM = 700
PE_MAX_LEN = 700


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


@dataclass
class SessionBank:
    E0: torch.Tensor
    T: torch.Tensor
    unit_mask: torch.Tensor


@dataclass(frozen=True)
class TrfConfig:
    kind: str
    conv_channels: int = 16
    conv_kernel: int = 5
    token_in: int = 36
    identity_dim: int = IDENTITY_DIM
    proj_out: int = 32
    t4_dim: int = 4
    set_dim: int = 256
    slots: int = 8
    heads: int = 8
    layers: int = 4
    temporal_width: int = 256
    ffn: int = 512
    readout_hidden: int = 128
    out_dim: int = 7


PROJ_ADD_CFG = TrfConfig(kind="proj_add")


def _sinusoidal_pe(max_len: int, width: int) -> torch.Tensor:
    pe = torch.zeros(max_len, width)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SharedCausalConv(nn.Module):
    def __init__(self, cfg: TrfConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(1, cfg.conv_channels, kernel_size=cfg.conv_kernel, padding=0, bias=True)
        self.act = nn.SiLU()
        self.left_pad = cfg.conv_kernel - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, width, n_units = x.shape
        hidden = x.permute(0, 2, 1).reshape(batch * n_units, 1, width)
        hidden = F.pad(hidden, (self.left_pad, 0))
        hidden = self.act(self.conv(hidden))
        return hidden.reshape(batch, n_units, self.cfg.conv_channels, width).permute(0, 3, 1, 2)


class SharedSetFrontend(nn.Module):
    def __init__(self, cfg: TrfConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.local_conv = SharedCausalConv(cfg)
        self.token_mlp = nn.Sequential(
            nn.Linear(cfg.token_in, cfg.set_dim),
            nn.GELU(),
            nn.Linear(cfg.set_dim, cfg.set_dim),
        )
        self.e0_proj = nn.Linear(cfg.identity_dim, cfg.proj_out, bias=False)
        self.slots = nn.Parameter(torch.zeros(cfg.slots, cfg.set_dim))
        self.slot_norm = nn.LayerNorm(cfg.set_dim)
        self.token_norm = nn.LayerNorm(cfg.set_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.set_dim, num_heads=cfg.heads, dropout=0.0, batch_first=True
        )
        self.slot_ffn_norm = nn.LayerNorm(cfg.set_dim)
        self.slot_ffn = nn.Sequential(
            nn.Linear(cfg.set_dim, 4 * cfg.set_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.set_dim, cfg.set_dim),
        )
        self.slot_proj = nn.Linear(cfg.slots * cfg.set_dim, cfg.temporal_width)

    def forward(self, x: torch.Tensor, bank: SessionBank, unit_keep: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        batch, width, n_units = x.shape
        local = self.local_conv(x)
        e0 = bank.E0.to(device=x.device, dtype=local.dtype)
        t4 = bank.T.to(device=x.device, dtype=local.dtype)
        n_groups = cfg.proj_out // cfg.conv_channels
        if cfg.proj_out != n_groups * cfg.conv_channels:
            raise ValueError(f"proj_out {cfg.proj_out} must be a multiple of {cfg.conv_channels}")
        if e0.dim() == 2:
            proj = self.e0_proj(e0).view(1, 1, n_units, n_groups, cfg.conv_channels)
        else:
            proj = self.e0_proj(e0).view(e0.shape[0], 1, n_units, n_groups, cfg.conv_channels)
        fused_local = (local.unsqueeze(-2) + proj).reshape(
            batch, width, n_units, n_groups * cfg.conv_channels
        )
        if t4.dim() == 2:
            t4 = t4.view(1, 1, n_units, cfg.t4_dim).expand(batch, width, n_units, cfg.t4_dim)
        else:
            t4 = t4.unsqueeze(1).expand(batch, width, n_units, cfg.t4_dim)
        tokens = self.token_norm(self.token_mlp(torch.cat([fused_local, t4], dim=-1)))
        slots = self.slot_norm(self.slots).view(1, 1, cfg.slots, cfg.set_dim).expand(
            batch, width, cfg.slots, cfg.set_dim
        )
        query = slots.reshape(batch * width, cfg.slots, cfg.set_dim)
        key = tokens.reshape(batch * width, n_units, cfg.set_dim)
        pad = (~unit_keep).unsqueeze(1).expand(batch, width, n_units).reshape(batch * width, n_units)
        attn_out, _ = self.mha(query, key, key, key_padding_mask=pad, need_weights=False)
        slots_out = query + attn_out
        slots_out = slots_out + self.slot_ffn(self.slot_ffn_norm(slots_out))
        return self.slot_proj(slots_out.reshape(batch, width, cfg.slots * cfg.set_dim))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrfConfig) -> None:
        super().__init__()
        self.n_heads = cfg.heads
        self.head_dim = cfg.temporal_width // cfg.heads
        self.qkv = nn.Linear(cfg.temporal_width, 3 * cfg.temporal_width)
        self.proj = nn.Linear(cfg.temporal_width, cfg.temporal_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, width, dim = x.shape
        qkv = self.qkv(x).view(batch, width, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(batch, width, dim))


class CausalTransformerBlock(nn.Module):
    def __init__(self, cfg: TrfConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.temporal_width)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.temporal_width)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.temporal_width, cfg.ffn),
            nn.GELU(),
            nn.Linear(cfg.ffn, cfg.temporal_width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x + self.attn(self.norm1(x))
        return hidden + self.ffn(self.norm2(hidden))


class CausalTransformerStack(nn.Module):
    def __init__(self, cfg: TrfConfig, max_len: int = PE_MAX_LEN) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(CausalTransformerBlock(cfg) for _ in range(cfg.layers))
        self.register_buffer("pe", _sinusoidal_pe(max_len, cfg.temporal_width), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x + self.pe[: x.size(1)].unsqueeze(0).to(dtype=x.dtype)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class CausalTransformerDecoder(nn.Module):
    def __init__(self, cfg: TrfConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = SharedSetFrontend(cfg)
        self.final_norm = nn.LayerNorm(cfg.temporal_width)
        self.readout = nn.Sequential(
            nn.Linear(cfg.temporal_width, cfg.readout_hidden),
            nn.GELU(),
            nn.Linear(cfg.readout_hidden, cfg.out_dim),
        )
        self.temporal = CausalTransformerStack(cfg)

    def forward_last(self, x: torch.Tensor, bank: SessionBank, unit_mask: torch.Tensor | None = None) -> torch.Tensor:
        keep = bank.unit_mask if unit_mask is None else unit_mask
        if keep.dtype != torch.bool:
            keep = keep.bool()
        if keep.dim() == 1:
            keep = keep.unsqueeze(0).expand(x.size(0), -1)
        hidden = self.temporal(self.frontend(x, bank, keep.contiguous()))
        return self.readout(self.final_norm(hidden))[:, -1, :]


def load_payload(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        return CPUUnpickler(handle).load()


def stack_banks(banks: list[SessionBank]) -> SessionBank:
    if not banks:
        raise ValueError("exact-E requires at least one session bank")
    n_units = banks[0].E0.shape[0]
    if any(bank.E0.shape[0] != n_units or bank.T.shape[0] != n_units for bank in banks):
        raise ValueError("vectorized sessions require the same unit roster width")
    return SessionBank(
        E0=torch.stack([bank.E0 for bank in banks]),
        T=torch.stack([bank.T for bank in banks]),
        unit_mask=torch.stack([bank.unit_mask for bank in banks]),
    )


@dataclass
class _WindowCacheState:
    raw: torch.Tensor
    frontend: torch.Tensor


class _FrontendWindowCache:
    def __init__(self, frontend, *, window: int, kernel: int = 5) -> None:
        if window < kernel or kernel < 2:
            raise ValueError("require window >= kernel >= 2")
        self.frontend = frontend
        self.window = int(window)
        self.kernel = int(kernel)
        self.state: _WindowCacheState | None = None

    def reset(self) -> None:
        self.state = None

    def rebuild(self, raw_window: torch.Tensor) -> torch.Tensor:
        if raw_window.ndim != 3 or raw_window.size(1) != self.window:
            raise ValueError(f"expected [B,{self.window},N] window, got {tuple(raw_window.shape)}")
        fused = self.frontend(raw_window)
        self.state = _WindowCacheState(raw=raw_window.detach().clone(), frontend=fused.detach().clone())
        return fused

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("advance requires rebuild after reset")
        old = self.state
        if next_bin.ndim != 3 or next_bin.shape[1:] != (1, old.raw.size(2)):
            raise ValueError("next_bin must be [B,1,N] with the compiled unit roster")
        raw = torch.cat((old.raw[:, 1:], next_bin), dim=1)
        left = self.frontend(raw[:, : self.kernel - 1])
        right = self.frontend(raw[:, -self.kernel :])[:, -1:]
        fused = torch.cat((left, old.frontend[:, self.kernel :], right), dim=1)
        self.state = _WindowCacheState(raw=raw.detach().clone(), frontend=fused.detach().clone())
        return fused


class _ExactEEngine:
    def __init__(self, model: CausalTransformerDecoder, banks: list[SessionBank], window: int) -> None:
        self.model = model
        self.bank = stack_banks(banks)
        self.cache = _FrontendWindowCache(self._frontend, window=window, kernel=5)

    def _frontend(self, x: torch.Tensor) -> torch.Tensor:
        keep = self.bank.unit_mask
        if keep.dim() == 1:
            keep = keep.unsqueeze(0).expand(x.size(0), -1)
        return self.model.frontend(x, self.bank, keep.contiguous())

    def _temporal_last(self, fused: torch.Tensor) -> torch.Tensor:
        hidden = fused + self.model.temporal.pe[: fused.size(1)].unsqueeze(0).to(
            device=fused.device, dtype=fused.dtype
        )
        blocks = self.model.temporal.blocks
        for block in blocks[:-1]:
            hidden = block(hidden)
        block = blocks[-1]
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
        return self.model.readout(self.model.final_norm(last))[:, 0, :]

    def rebuild(self, window: torch.Tensor) -> torch.Tensor:
        return self._temporal_last(self.cache.rebuild(window))

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        return self._temporal_last(self.cache.advance(next_bin))


class H1ProjAddFalconDecoder(BCIDecoder):
    """BCIDecoder that runs H1 proj_add with exact-E caching."""

    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        if self.batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {self.batch_size} exceeds official H1 max {OFFICIAL_BATCH}")
        payload = load_payload(model_path)
        if payload.get("schema_version") != PAYLOAD_SCHEMA:
            raise ValueError(f"unsupported payload schema {payload.get('schema_version')}")
        if payload.get("task") != task_config.task:
            raise ValueError("payload task does not match evaluator task")
        window = int(payload.get("window_size", -1))
        if window not in {200, 250}:
            raise ValueError(f"payload window_size must be 200 or 250, got {window}")
        kind = str(payload["kind"])
        if kind != "proj_add":
            raise ValueError(f"unknown kind {kind}")
        self.kind = kind
        self.window_size = window
        self.decoder = CausalTransformerDecoder(PROJ_ADD_CFG)
        state = {
            name: torch.as_tensor(value, dtype=torch.float32) for name, value in payload["state_dict"].items()
        }
        missing, unexpected = self.decoder.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise ValueError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.bank_by_dataset_tag = payload["bank_by_dataset_tag"]
        if len(self.bank_by_dataset_tag) != EXPECTED_SESSION_COUNT:
            raise ValueError("H1 payload must cover exactly 27 official dataset tags")
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        if abs(self.behavior_scaling_factor - BEHAVIOR_SCALE) > 1.0e-12:
            raise ValueError("behavior scale must be 20")
        if int(self.decoder.temporal.pe.size(0)) < self.window_size:
            raise ValueError("runtime PE does not cover the payload window")
        self.observation_buffer = np.zeros(
            (self.window_size, self.batch_size, task_config.n_channels), dtype=np.float32
        )
        self.history_count = np.zeros(self.batch_size, dtype=np.int64)
        self.device = torch.device("cpu")
        self.local_banks: list[SessionBank] = []
        self._engine: _ExactEEngine | None = None
        self._kv_cache_enabled = False
        self._n_predicts = 0

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        hashed_tags = [self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        missing = [tag for tag in hashed_tags if tag not in self.bank_by_dataset_tag]
        if missing:
            raise ValueError(f"Dataset tags {missing} are absent from the H1 proj_add bank payload")
        self.device = torch.device("cpu")
        self.decoder = self.decoder.to(self.device).eval()
        self.local_banks = []
        for tag in hashed_tags:
            row = self.bank_by_dataset_tag[tag]
            self.local_banks.append(
                SessionBank(
                    E0=torch.as_tensor(row["E0"], dtype=torch.float32, device=self.device),
                    T=torch.as_tensor(row["T"], dtype=torch.float32, device=self.device),
                    unit_mask=torch.as_tensor(row["unit_mask"], dtype=torch.bool, device=self.device),
                )
            )
        self.observation_buffer.fill(0.0)
        self.history_count.fill(0)
        self._engine = None

    def on_done(self, dones: np.ndarray):
        return None

    def set_batch_size(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {batch_size} exceeds official H1 max {OFFICIAL_BATCH}")
        self.batch_size = batch_size
        self.observation_buffer = np.zeros(
            (self.window_size, self.batch_size, self.task_config.n_channels), dtype=np.float32
        )
        self.history_count = np.zeros(self.batch_size, dtype=np.int64)
        self._engine = None

    def observe(self, neural_observations: np.ndarray):
        observations = np.asarray(neural_observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.task_config.n_channels:
            raise ValueError(
                f"Expected neural observations [B,{self.task_config.n_channels}], got {observations.shape}"
            )
        if observations.shape[0] > self.batch_size:
            raise ValueError("Evaluator batch exceeds configured decoder batch size")
        active = observations.shape[0]
        if active < self.batch_size:
            observations = np.pad(observations, ((0, self.batch_size - active), (0, 0)))
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = observations
        self.history_count[:active] += 1

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
                self._engine = _ExactEEngine(self.decoder, self.local_banks, self.window_size)
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
        return native[:active].astype(np.float32, copy=False)


def smoke(payload_path: str, window_path: str) -> None:
    from falcon_challenge.config import FalconConfig, FalconTask

    bundle = np.load(window_path)
    config = FalconConfig(task=FalconTask.h1)
    decoder = H1ProjAddFalconDecoder(task_config=config, model_path=payload_path, batch_size=1)
    decoder.reset(dataset_tags=[str(bundle["tag_stem"])])
    window = np.asarray(bundle["window"], dtype=np.float32)
    if window.ndim != 2:
        raise ValueError("smoke window must be [L, N]")
    pred = None
    history_before_done = None
    for row in window:
        pred = decoder.predict(row.reshape(1, -1))
        history_before_done = int(decoder.history_count[0])
        decoder.on_done(np.ones((1,), dtype=bool))
        if int(decoder.history_count[0]) != history_before_done:
            raise RuntimeError("on_done reset continual history")
    expected = np.asarray(bundle["expected"], dtype=np.float32)
    if pred is None or pred.shape != expected.shape or not np.allclose(pred, expected, atol=1.0e-5, rtol=1.0e-5):
        raise RuntimeError(f"container smoke mismatch {pred} vs {expected}")
    if not np.isfinite(pred).all():
        raise RuntimeError("container smoke non-finite")
    print(
        json.dumps(
            {
                "status": "CONTAINER_SMOKE_PASS",
                "pred": pred.tolist(),
                "on_done_noop": True,
                "window": int(decoder.window_size),
                "kind": decoder.kind,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-payload", default="")
    parser.add_argument("--smoke-window", default="")
    args = parser.parse_args()
    if args.smoke_payload and args.smoke_window:
        smoke(args.smoke_payload, args.smoke_window)
    else:
        raise SystemExit("use --smoke-payload and --smoke-window")
