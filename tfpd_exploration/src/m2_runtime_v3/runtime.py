"""Exact-E M2 runtime with a final-row Q projection and direct host input.

The first L-1 temporal blocks remain full-window computations.  The last
block has no temporal KV cache: K and V are recomputed for every row on every
call.  It is therefore mathematically the same full-window causal decoder.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


WINDOW = 50
OFFICIAL_BATCH = 7
_REFERENCE = Path(__file__).parents[2] / "submissions/evalai_m2_small_trf_e_opt_v1/trf_falcon_decoder.py"


class RuntimeV3Error(ValueError):
    pass


def _reference_module():
    """Load the frozen submitted implementation under a private module name."""
    key = "_m2_runtime_v3_frozen_reference"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeV3Error(f"frozen reference unavailable: {_REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class StateBytes:
    dynamic_owned_tensor_bytes: int
    static_model_tensor_bytes: int
    static_active_bank_tensor_bytes: int
    static_active_projection_bytes: int
    host_observation_bytes: int


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _version(value: torch.Tensor) -> int:
    """Make mutation auditing fail closed for inference-mode tensors."""
    try:
        return int(value._version)
    except RuntimeError as exc:
        raise RuntimeV3Error(
            "runtime-v3 tensors must be created outside torch.inference_mode so mutations are auditable"
        ) from exc


class _ExactEFinalQ:
    """Frontend window reuse plus full-prefix / final-row temporal evaluation."""

    def __init__(self, model, bank) -> None:
        self.model = model
        self.bank = bank
        self.raw: torch.Tensor | None = None
        self.frontend: torch.Tensor | None = None
        front = model.frontend
        first = front.token_mlp[0]
        # E0 and T never change between reset calls.  Their contribution to
        # the first token linear layer is therefore a legal static projection.
        static_features = torch.cat((bank.E0, bank.T), dim=-1)
        self.static_token = F.linear(static_features, first.weight[:, 16:], first.bias)
        self._signature = self._make_signature()

    def _make_signature(self) -> tuple[object, ...]:
        first_parameter = next(self.model.parameters())
        parts: list[object] = [id(self.model), first_parameter.device, first_parameter.dtype]
        for value in (
            *self.model.parameters(), *self.model.buffers(), self.bank.E0, self.bank.T,
            self.bank.unit_mask, self.static_token,
        ):
            parts.extend((value.data_ptr(), tuple(value.shape), value.dtype, value.device, _version(value)))
        return tuple(parts)

    def _refresh_if_mutated(self) -> None:
        """Refresh legal CPU/float32 mutations; reject device/dtype migration."""
        parameter = next(self.model.parameters())
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise RuntimeV3Error("runtime-v3 supports only CPU float32 model state; reset/rebuild after migration")
        signature = self._make_signature()
        if signature == self._signature:
            return
        # E0/T/mask mutation, order replacement, or parameter/buffer mutation
        # invalidates both the frontend cache and immutable bank projection.
        front = self.model.frontend
        first = front.token_mlp[0]
        features = torch.cat((self.bank.E0, self.bank.T), dim=-1)
        self.static_token = F.linear(features, first.weight[:, 16:], first.bias)
        if self.raw is not None:
            self.frontend = self._frontend(self.raw)
        self._signature = self._make_signature()

    def _frontend(self, raw: torch.Tensor) -> torch.Tensor:
        """Frozen frontend, factoring only immutable E0/T into static_token."""
        front = self.model.frontend
        cfg = front.cfg
        batch, width, n_units = raw.shape
        local = front.local_conv(raw)
        first, activation, second = front.token_mlp
        tokens = F.linear(local, first.weight[:, :16], None)
        tokens = tokens + self.static_token.unsqueeze(1)
        tokens = front.token_norm(second(activation(tokens)))
        slots = front.slot_norm(front.slots).view(1, 1, cfg.slots, cfg.set_dim).expand(
            batch, width, cfg.slots, cfg.set_dim
        )
        q = slots.reshape(batch * width, cfg.slots, cfg.set_dim)
        k = tokens.reshape(batch * width, n_units, cfg.set_dim)
        keep = self.bank.unit_mask
        if keep.dim() == 1:
            keep = keep.unsqueeze(0).expand(raw.size(0), -1)
        pad = (~keep).unsqueeze(1).expand(batch, width, n_units).reshape(batch * width, n_units)
        attn_out, _ = front.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attn_out
        slots_out = slots_out + front.slot_ffn(front.slot_ffn_norm(slots_out))
        return front.slot_proj(slots_out.reshape(batch, width, cfg.slots * cfg.set_dim))

    def _last(self, z: torch.Tensor) -> torch.Tensor:
        hidden = z + self.model.temporal.pe[: z.size(1)].unsqueeze(0).to(
            device=z.device, dtype=z.dtype
        )
        blocks = self.model.temporal.blocks
        for block in blocks[:-1]:
            hidden = block(hidden)
        block = blocks[-1]
        normed = block.norm1(hidden)
        attn = block.attn
        batch, width, dim = normed.shape
        # The frozen path projected all [B,50] rows to QKV then discarded 49
        # Q rows.  Q is independent per row, so this keeps only the useful Q.
        q_weight, kv_weight = attn.qkv.weight[:dim], attn.qkv.weight[dim:]
        q_bias, kv_bias = attn.qkv.bias[:dim], attn.qkv.bias[dim:]
        query = F.linear(normed[:, -1:], q_weight, q_bias).view(
            batch, 1, attn.n_heads, attn.head_dim
        ).transpose(1, 2)
        key_value = F.linear(normed, kv_weight, kv_bias).view(
            batch, width, 2, attn.n_heads, attn.head_dim
        )
        key, value = key_value.unbind(dim=2)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        out = attn.proj(out.transpose(1, 2).contiguous().view(batch, 1, dim))
        last = hidden[:, -1:] + out
        last = last + block.ffn(block.norm2(last))
        return self.model.readout(self.model.final_norm(last))[:, 0, :]

    def rebuild(self, raw: torch.Tensor) -> torch.Tensor:
        self._refresh_if_mutated()
        if raw.ndim != 3 or raw.shape[1] != WINDOW:
            raise RuntimeV3Error(f"expected raw [B,{WINDOW},N], got {tuple(raw.shape)}")
        self.raw = raw
        self.frontend = self._frontend(raw)
        return self._last(self.frontend)

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        self._refresh_if_mutated()
        if self.raw is None or self.frontend is None:
            raise RuntimeV3Error("advance before rebuild")
        if next_bin.shape != (self.raw.shape[0], 1, self.raw.shape[2]):
            raise RuntimeV3Error("new bin shape / roster drift")
        # Avoid the reference adapter's full numpy history transpose and
        # host-to-torch conversion of 50 bins: only the direct newest host
        # bin crosses into torch.  A new tensor is required here because
        # PyTorch forbids an overlapping in-place left shift.
        self.raw = torch.cat((self.raw[:, 1:], next_bin), dim=1)
        left = self._frontend(self.raw[:, :4])
        right = self._frontend(self.raw[:, -5:])[:, -1:]
        self.frontend = torch.cat((left, self.frontend[:, 5:], right), dim=1)
        return self._last(self.frontend)

    def dynamic_bytes(self) -> int:
        return sum(_tensor_bytes(value) for value in (self.raw, self.frontend) if value is not None)

    def static_projection_bytes(self) -> int:
        return _tensor_bytes(self.static_token)


class RuntimeV3Decoder:
    """Read-only payload adapter with strict host-input and roster guards."""

    def __init__(self, payload_path: str | Path, *, batch_size: int = 1) -> None:
        if not 1 <= int(batch_size) <= OFFICIAL_BATCH:
            raise RuntimeV3Error(f"batch_size must be 1..{OFFICIAL_BATCH}")
        ref = _reference_module()
        payload = ref.load_payload(payload_path)
        if payload.get("schema_version") != ref.PAYLOAD_SCHEMA or int(payload.get("window_size", -1)) != WINDOW:
            raise RuntimeV3Error("unrecognized frozen M2 payload")
        self._ref = ref
        self.batch_size = int(batch_size)
        self.kind = str(payload["kind"])
        self.model = ref.build_decoder(self.kind).eval()
        missing, unexpected = self.model.load_state_dict(
            {name: torch.as_tensor(value, dtype=torch.float32) for name, value in payload["state_dict"].items()},
            strict=True,
        )
        if missing or unexpected:
            raise RuntimeV3Error("payload/model state mismatch")
        for value in self.model.parameters():
            value.requires_grad_(False)
        self._banks = payload["bank_by_dataset_tag"]
        if len(self._banks) != 13 or float(payload["behavior_scaling_factor"]) != 5.0:
            raise RuntimeV3Error("frozen bank or output-scale contract drift")
        if bool(payload.get("smooth_observations", False)):
            raise RuntimeV3Error("v3 direct-input path requires the frozen unsmoothed payload")
        self._engine: _ExactEFinalQ | None = None
        self._active = 0
        self._channels: int | None = None

    def _require_eval(self) -> None:
        """Fail closed: cached inference semantics are invalid in train mode."""
        if self.model.training:
            raise RuntimeV3Error("runtime-v3 requires model.eval(); train-mode execution is forbidden")

    def reset(self, dataset_tags: Iterable[str | Path]) -> None:
        self._require_eval()
        tags = [Path(tag).stem for tag in dataset_tags]
        if not tags or len(tags) > self.batch_size:
            raise RuntimeV3Error("active dataset count must be 1..configured batch")
        missing = [tag for tag in tags if tag not in self._banks]
        if missing:
            raise RuntimeV3Error(f"unknown frozen bank tag(s): {missing}")
        rows = [self._banks[tag] for tag in tags]
        channels = int(np.asarray(rows[0]["E0"]).shape[0])
        if any(np.asarray(row["E0"]).shape != (channels, 50) or np.asarray(row["T"]).shape != (channels, 4) for row in rows):
            raise RuntimeV3Error("heterogeneous unit roster is not vectorizable")
        mask = [np.asarray(row["unit_mask"], dtype=np.bool_) for row in rows]
        if any(value.shape != (channels,) for value in mask):
            raise RuntimeV3Error("unit-mask roster drift")
        bank = self._ref.SessionBank(
            E0=torch.stack([torch.as_tensor(row["E0"], dtype=torch.float32) for row in rows]),
            T=torch.stack([torch.as_tensor(row["T"], dtype=torch.float32) for row in rows]),
            unit_mask=torch.stack([torch.as_tensor(value, dtype=torch.bool) for value in mask]),
        )
        raw = torch.zeros((len(rows), WINDOW, channels), dtype=torch.float32)
        self._engine = _ExactEFinalQ(self.model, bank)
        self._engine.rebuild(raw)
        self._active, self._channels = len(rows), channels

    def predict(self, observations: np.ndarray) -> np.ndarray:
        self._require_eval()
        if self._engine is None or self._channels is None:
            raise RuntimeV3Error("reset must precede predict")
        value = np.asarray(observations)
        if value.dtype != np.float32 or value.ndim != 2 or value.shape != (self._active, self._channels):
            raise RuntimeV3Error(f"expected float32 [{self._active},{self._channels}] direct host input")
        if not value.flags.c_contiguous:
            raise RuntimeV3Error("direct host input must be C-contiguous")
        prediction = self._engine.advance(torch.from_numpy(value).unsqueeze(1))
        native = prediction.numpy() / 5.0
        if not np.isfinite(native).all():
            raise RuntimeV3Error("non-finite result")
        if self._active == self.batch_size:
            return native.astype(np.float32, copy=False)
        result = np.zeros((self.batch_size, native.shape[1]), dtype=np.float32)
        result[: self._active] = native
        return result

    def observe(self, observations: np.ndarray) -> None:
        """Consume a legal M2 gap bin through the identical public state path."""
        self.predict(observations)

    def on_done(self, dones: np.ndarray) -> None:
        """M2 is continual; trial-boundary flags do not reset causal state."""
        self._require_eval()
        value = np.asarray(dones)
        if value.ndim != 1 or value.shape[0] != self._active:
            raise RuntimeV3Error("on_done expects one flag per active session")

    def state_bytes(self) -> StateBytes:
        dynamic = 0 if self._engine is None else self._engine.dynamic_bytes()
        bank = 0
        if self._engine is not None:
            bank = sum(_tensor_bytes(getattr(self._engine.bank, key)) for key in ("E0", "T", "unit_mask"))
        model = sum(_tensor_bytes(value) for value in self.model.state_dict().values())
        host = 0 if self._channels is None else self._active * self._channels * np.dtype(np.float32).itemsize
        projection = 0 if self._engine is None else self._engine.static_projection_bytes()
        return StateBytes(dynamic, model, bank, projection, host)
