"""Self-contained Falcon runtime for the M1 current-query Transformer.

Copied into the EvalAI image. Reconstructs Conv1→16 k5, concat(local,E0,H-C),
8-slot set attention d=256, 4-layer causal Transformer, 256→128→7 last-bin,
output /20. Does not import tfpd_exploration, does not instantiate the C2
SPINT decoder, and does not reuse KV cache across windows. on_done is a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
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
from v3_runtime import BankBatch, HeterogeneousCurrentQueryStream

PAYLOAD_SCHEMA = "m1_optimized_v2_trf_falcon_payload_v1"
WINDOW = 100
CHANNELS = 64
BEHAVIOR_SCALE = 1.0
EXPECTED_SESSION_COUNT = 7
OFFICIAL_BATCH = 4
PE_MAX_LEN = 100


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
class M1TrfConfig:
    kind: str
    conv_channels: int = 16
    conv_kernel: int = 5
    e0_dim: int = 100
    hc_dim: int = 4
    set_dim: int = 256
    slots: int = 8
    heads: int = 8
    layers: int = 4
    temporal_width: int = 256
    ffn: int = 512
    readout_hidden: int = 128
    out_dim: int = 16
    window: int = 100
    pe_max_len: int = 100
    route_key_dim: int = 32

    @property
    def token_in(self) -> int:
        return self.conv_channels + self.e0_dim + self.hc_dim


QUERY_CFG = M1TrfConfig(kind="current_query")


def _sinusoidal_pe(max_len: int, width: int) -> torch.Tensor:
    pe = torch.zeros(max_len, width)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _sdpa_causal(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=True)


class SharedCausalConv(nn.Module):
    def __init__(self, cfg: M1TrfConfig) -> None:
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


class FactoredTokenMLP(nn.Module):
    def __init__(self, cfg: M1TrfConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.token_in, cfg.set_dim)
        self.fc2 = nn.Linear(cfg.set_dim, cfg.set_dim)

    def forward(self, local: torch.Tensor, e0: torch.Tensor, hc: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        weight = self.fc1.weight
        w_loc, w_e0, w_hc = weight.split([cfg.conv_channels, cfg.e0_dim, cfg.hc_dim], dim=1)
        hidden = F.linear(local, w_loc, None)
        e0_term = F.linear(e0, w_e0, None)
        hc_term = F.linear(hc, w_hc, None)
        if e0_term.dim() == 2:
            e0_term = e0_term.unsqueeze(0).unsqueeze(1)
        elif e0_term.dim() == 3:
            e0_term = e0_term.unsqueeze(1)
        if hc_term.dim() == 2:
            hc_term = hc_term.unsqueeze(0).unsqueeze(1)
        elif hc_term.dim() == 3:
            hc_term = hc_term.unsqueeze(1)
        hidden = hidden + e0_term + hc_term + self.fc1.bias
        return self.fc2(F.gelu(hidden))


class SlotUnitAttention(nn.Module):
    def __init__(self, cfg: M1TrfConfig, routed: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.routed = routed
        self.n_heads = cfg.heads
        self.head_dim = cfg.set_dim // cfg.heads
        self.q_proj = nn.Linear(cfg.set_dim, cfg.set_dim)
        self.k_proj = nn.Linear(cfg.set_dim, cfg.set_dim)
        self.v_proj = nn.Linear(cfg.set_dim, cfg.set_dim)
        self.out_proj = nn.Linear(cfg.set_dim, cfg.set_dim)
        if routed:
            self.route_proj = nn.ModuleList(
                nn.Linear(cfg.e0_dim + cfg.hc_dim, cfg.route_key_dim) for _ in range(cfg.heads)
            )
            self.q_cal = nn.Parameter(torch.zeros(cfg.heads, cfg.slots, cfg.route_key_dim))
            self.g = nn.Parameter(torch.zeros(cfg.heads))

    def routing_bonus(self, e0: torch.Tensor, hc: torch.Tensor, batch: int) -> torch.Tensor:
        if e0.dim() == 2:
            e0 = e0.unsqueeze(0).expand(batch, -1, -1)
        if hc.dim() == 2:
            hc = hc.unsqueeze(0).expand(batch, -1, -1)
        joined = torch.cat([e0, hc], dim=-1)
        calibrated = F.layer_norm(joined, (joined.shape[-1],), weight=None, bias=None)
        parts = [proj(calibrated) for proj in self.route_proj]
        projected = torch.stack(parts, dim=1)
        scale = self.cfg.route_key_dim ** -0.5
        bonus = torch.einsum("hks,bhns->bhkn", self.q_cal, projected) * scale
        gate = torch.tanh(self.g).view(1, self.n_heads, 1, 1)
        return gate * bonus

    def forward(
        self,
        slots: torch.Tensor,
        tokens: torch.Tensor,
        unit_keep: torch.Tensor,
        e0: torch.Tensor | None = None,
        hc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_width, n_slots, dim = slots.shape
        n_units = tokens.size(1)
        query = self.q_proj(slots).view(batch_width, n_slots, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(tokens).view(batch_width, n_units, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(tokens).view(batch_width, n_units, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        logits = torch.matmul(query, key.transpose(-2, -1)) * scale
        if self.routed:
            if e0 is None or hc is None:
                raise RuntimeError("ROUTE needs E0 and H-C")
            batch = unit_keep.size(0)
            width = batch_width // batch
            bonus = self.routing_bonus(e0, hc, batch)
            logits = logits + bonus.unsqueeze(1).expand(-1, width, -1, -1, -1).reshape(
                batch_width, self.n_heads, n_slots, n_units
            )
        pad = (~unit_keep).unsqueeze(1).expand(-1, batch_width // unit_keep.size(0), -1)
        pad = pad.reshape(batch_width, n_units)
        logits = logits.masked_fill(pad.unsqueeze(1).unsqueeze(2), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        hidden = torch.matmul(weights, value)
        hidden = hidden.transpose(1, 2).contiguous().view(batch_width, n_slots, dim)
        return self.out_proj(hidden)


class SharedSetFrontend(nn.Module):
    def __init__(self, cfg: M1TrfConfig, routed: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.local_conv = SharedCausalConv(cfg)
        self.token_mlp = FactoredTokenMLP(cfg)
        self.slots = nn.Parameter(torch.zeros(cfg.slots, cfg.set_dim))
        self.slot_norm = nn.LayerNorm(cfg.set_dim)
        self.token_norm = nn.LayerNorm(cfg.set_dim)
        self.attn = SlotUnitAttention(cfg, routed=routed)
        self.slot_ffn_norm = nn.LayerNorm(cfg.set_dim)
        self.slot_ffn = nn.Sequential(
            nn.Linear(cfg.set_dim, 4 * cfg.set_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.set_dim, cfg.set_dim),
        )
        self.slot_proj = nn.Linear(cfg.slots * cfg.set_dim, cfg.temporal_width)

    def forward(self, x: torch.Tensor, e0: torch.Tensor, hc: torch.Tensor, unit_keep: torch.Tensor) -> torch.Tensor:
        local = self.local_conv(x)
        batch, width, n_units, _ = local.shape
        tokens = self.token_norm(self.token_mlp(local, e0, hc))
        slots = self.slot_norm(self.slots).view(1, 1, self.cfg.slots, self.cfg.set_dim)
        slots = slots.expand(batch, width, self.cfg.slots, self.cfg.set_dim)
        query = slots.reshape(batch * width, self.cfg.slots, self.cfg.set_dim)
        key = tokens.reshape(batch * width, n_units, self.cfg.set_dim)
        attended = self.attn(query, key, unit_keep, e0=e0, hc=hc)
        slots_out = query + attended
        slots_out = slots_out + self.slot_ffn(self.slot_ffn_norm(slots_out))
        fused = slots_out.reshape(batch, width, self.cfg.slots * self.cfg.set_dim)
        return self.slot_proj(fused)



class QueryReadBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn: int, window: int, age_buckets: int) -> None:
        super().__init__()
        self.width, self.heads = width, heads
        self.head_dim = width // heads
        self.window, self.age_buckets = window, age_buckets
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width)
        self.kv_proj = nn.Linear(width, 2 * width)
        self.out_proj = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))
        self.age_bias = nn.Parameter(torch.zeros(heads, age_buckets))

    def project_memory(self, z: torch.Tensor):
        b, t, _ = z.shape
        kv = self.kv_proj(self.memory_norm(z)).reshape(b, t, 2, self.heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def read(self, query: torch.Tensor, memory, valid_mask=None) -> torch.Tensor:
        k, v = memory
        b, _, t, _ = k.shape
        q = self.q_proj(self.query_norm(query)).reshape(b, 1, self.heads, self.head_dim).transpose(1, 2)
        age = torch.arange(t - 1, -1, -1, device=query.device)
        buckets = torch.div(age * self.age_buckets, self.window, rounding_mode="floor").clamp(max=self.age_buckets - 1)
        bias = self.age_bias[:, buckets].to(dtype=query.dtype).unsqueeze(0).unsqueeze(2)
        if valid_mask is not None:
            bias = bias.masked_fill(~valid_mask[:, None, None, :], float("-inf"))
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
        context = context.transpose(1, 2).reshape(b, 1, self.width)
        hidden = query + self.out_proj(context)
        return hidden + self.ffn(self.ffn_norm(hidden))

    def forward(self, query: torch.Tensor, z: torch.Tensor, valid_mask=None) -> torch.Tensor:
        return self.read(query, self.project_memory(z), valid_mask)


class QueryTemporalStack(nn.Module):
    def __init__(self, cfg: M1TrfConfig) -> None:
        super().__init__()
        self.width = cfg.temporal_width
        self.window = cfg.window
        self.blocks = nn.ModuleList(
            QueryReadBlock(cfg.temporal_width, cfg.heads, cfg.ffn, cfg.window, 16)
            for _ in range(cfg.layers)
        )

    def forward(self, z: torch.Tensor, valid_mask=None) -> torch.Tensor:
        query = z[:, -1:]
        for block in self.blocks:
            query = block(query, z, valid_mask)
        return query


class M1TemporalDecoder(nn.Module):
    def __init__(self, cfg: M1TrfConfig, routed: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.routed = routed
        self.frontend = SharedSetFrontend(cfg, routed=routed)
        self.temporal = QueryTemporalStack(cfg)
        self.final_norm = nn.LayerNorm(cfg.temporal_width)
        self.readout = nn.Sequential(
            nn.Linear(cfg.temporal_width, cfg.readout_hidden),
            nn.GELU(),
            nn.Linear(cfg.readout_hidden, cfg.out_dim),
        )

    def forward_last(self, x: torch.Tensor, bank: SessionBank, unit_mask: torch.Tensor | None = None) -> torch.Tensor:
        keep = bank.unit_mask if unit_mask is None else unit_mask
        if keep.dtype != torch.bool:
            keep = keep.bool()
        if keep.dim() == 1:
            keep = keep.unsqueeze(0).expand(x.size(0), -1)
        e0 = bank.E0.to(device=x.device, dtype=x.dtype)
        hc = bank.T.to(device=x.device, dtype=x.dtype)
        hidden = self.temporal(self.frontend(x, e0, hc, keep.contiguous()))
        return self.readout(self.final_norm(hidden))[:, -1, :]


def build_decoder(kind: str) -> M1TemporalDecoder:
    if kind in {"current_query", "flat"}:
        return M1TemporalDecoder(QUERY_CFG, routed=False)
    raise ValueError(f"unknown decoder kind {kind}")


def load_payload(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        return CPUUnpickler(handle).load()


class M1TemporalFalconDecoder(BCIDecoder):
    """BCIDecoder that runs the new temporal Transformer, one full W=700 window per step."""

    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        if self.batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {self.batch_size} exceeds official M1 max {OFFICIAL_BATCH}")
        payload = load_payload(model_path)
        if payload.get("schema_version") != PAYLOAD_SCHEMA:
            raise ValueError(f"unsupported payload schema {payload.get('schema_version')}")
        if payload.get("task") != task_config.task:
            raise ValueError("payload task does not match evaluator task")
        if int(payload.get("window_size", -1)) != WINDOW:
            raise ValueError("payload window_size must be 100")
        if int(payload.get("window_size", -1)) != WINDOW:
            raise ValueError("window must be 100")
        kind = str(payload["kind"])
        if kind not in {"current_query", "flat"}:
            raise ValueError(f"unknown kind {kind}")
        self.kind = kind
        self.decoder = build_decoder(kind)
        state = {name: torch.as_tensor(value, dtype=torch.float32) for name, value in payload["state_dict"].items()}
        missing, unexpected = self.decoder.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise ValueError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.bank_by_dataset_tag = payload["bank_by_dataset_tag"]
        if len(self.bank_by_dataset_tag) != EXPECTED_SESSION_COUNT:
            raise ValueError("M1 payload must cover exactly 7 public calibration sessions")
        self.window_size = WINDOW
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        if abs(self.behavior_scaling_factor - BEHAVIOR_SCALE) > 1.0e-12:
            raise ValueError("behavior scale must be 1")
        self.observation_buffer = np.zeros(
            (self.window_size, self.batch_size, task_config.n_channels), dtype=np.float32
        )
        self.history_count = np.zeros(self.batch_size, dtype=np.int64)
        self.device = torch.device("cpu")
        self.local_banks: list[SessionBank] = []
        self._stream: HeterogeneousCurrentQueryStream | None = None
        self._last_output: np.ndarray | None = None

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        hashed_tags = [self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        missing = [tag for tag in hashed_tags if tag not in self.bank_by_dataset_tag]
        if missing:
            raise ValueError(f"Dataset tags {missing} are absent from the H1 temporal bank payload")
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
        # The evaluator can have a final partial batch.  Its inactive rows use
        # a repeated legal bank but V3 freezes all their causal state; their
        # values cannot contaminate active rows.
        rows = self.local_banks + [self.local_banks[-1]] * (self.batch_size - len(self.local_banks))
        bank = BankBatch(
            E0=torch.stack([row.E0 for row in rows]), T=torch.stack([row.T for row in rows]),
            unit_mask=torch.stack([row.unit_mask for row in rows]), session_ids=tuple(hashed_tags) + tuple([hashed_tags[-1]] * (self.batch_size-len(hashed_tags))),
            unit_ids=tuple(tuple(range(self.task_config.n_channels)) for _ in rows),
        )
        self._stream = HeterogeneousCurrentQueryStream(self.decoder, bank)
        self._last_output = None
        self.observation_buffer.fill(0.0)
        self.history_count.fill(0)

    def on_done(self, dones: np.ndarray):
        return None

    def set_batch_size(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size > OFFICIAL_BATCH:
            raise ValueError(f"batch_size {batch_size} exceeds official M1 max {OFFICIAL_BATCH}")
        self.batch_size = batch_size
        self.observation_buffer = np.zeros(
            (self.window_size, self.batch_size, self.task_config.n_channels), dtype=np.float32
        )
        self.history_count = np.zeros(self.batch_size, dtype=np.int64)
        self._stream = None
        self._last_output = None

    def observe(self, neural_observations: np.ndarray):
        observations = np.asarray(neural_observations)
        if observations.ndim != 2 or observations.shape[1] != self.task_config.n_channels:
            raise ValueError(
                f"Expected neural observations [B,{self.task_config.n_channels}], got {observations.shape}"
            )
        if observations.dtype != np.float32 or not observations.flags.c_contiguous or not np.isfinite(observations).all():
            raise ValueError("V3 requires finite C-contiguous float32 observations")
        if observations.shape[0] > self.batch_size:
            raise ValueError("Evaluator batch exceeds configured decoder batch size")
        active = observations.shape[0]
        if self._stream is None: raise RuntimeError("reset(dataset_tags) must be called before observe")
        self._last_output = self._stream.predict(observations, active=active)
        self.history_count[:active] += 1

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if not self.local_banks:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        active = int(np.asarray(neural_observations).shape[0])
        self.observe(neural_observations)
        assert self._stream is not None
        # observe made exactly one public V3 call and retained its copied
        # native output; do not make a second stateful advance here.
        assert self._last_output is not None
        native = self._last_output / self.behavior_scaling_factor
        if not np.isfinite(native).all():
            raise RuntimeError("non-finite M1 current-query Transformer prediction")
        # Slicing a B4 backing allocation for a final partial batch can expose
        # a non-owning view.  The public Falcon boundary always returns an
        # independent native float32 allocation.
        return native[:active].astype(np.float32, copy=True)


def smoke(payload_path: str, window_path: str) -> None:
    from falcon_challenge.config import FalconConfig, FalconTask

    bundle = np.load(window_path)
    config = FalconConfig(task=FalconTask.m1)
    decoder = M1TemporalFalconDecoder(task_config=config, model_path=payload_path, batch_size=1)
    decoder.reset(dataset_tags=[str(bundle["tag_stem"])])
    window = np.asarray(bundle["window"], dtype=np.float32)
    if window.ndim != 2:
        raise ValueError("smoke window must be [100, N]")
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
