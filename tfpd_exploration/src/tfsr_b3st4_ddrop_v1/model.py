"""CPU-tensor-only TF-SR architecture scaffold; no data loading or launch behavior.

The normalized T4 input is deliberately a capability rather than a bare tensor.
That keeps the source-normalizer authority and unit-axis lineage attached to the
only representation that B3S and the decoder are allowed to consume.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tfpd.bilinear_readin import CausalActivityEncoder


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DIAGNOSTIC_MODES = frozenset(("aligned", "zero", "wrong_pair"))


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact lowercase 64-hex SHA-256")


def ordered_unit_digest(ordered_unit_ids: tuple[str, ...]) -> str:
    """Return the canonical digest used to bind the ordered TF-SR unit axis."""
    if not isinstance(ordered_unit_ids, tuple) or not ordered_unit_ids:
        raise ValueError("ordered_unit_ids must be a nonempty tuple")
    if any(not isinstance(unit_id, str) or not unit_id for unit_id in ordered_unit_ids):
        raise ValueError("ordered_unit_ids must contain nonempty strings")
    if len(set(ordered_unit_ids)) != len(ordered_unit_ids):
        raise ValueError("ordered_unit_ids must be unique")
    payload = json.dumps(ordered_unit_ids, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class NormalizedT4Batch:
    """Immutable authority-bound, normalized T4 input for one ordered unit batch.

    ``wrong_pair`` is intentionally a diagnostic capability: its tensor values
    are paired to the receiving unit axis incorrectly, while its recorded axis
    remains the decoder's real ordered axis.  It cannot be mistaken for an
    aligned production capability because the mode is explicit.
    """

    tensor: Tensor
    raw_authority_sha256: str
    normalizer_authority_sha256: str
    roster_digest: str
    ordered_unit_digest: str
    ordered_unit_ids: tuple[str, ...]
    lineage: tuple[str, ...]
    diagnostic_mode: Literal["aligned", "zero", "wrong_pair"] = "aligned"

    def __post_init__(self) -> None:
        self.validate()

    @property
    def normalized(self) -> Tensor:
        """A named alias that makes the model-visible domain explicit."""
        return self.tensor

    @property
    def batch_size(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def num_units(self) -> int:
        return int(self.tensor.shape[1])

    def validate(self) -> None:
        if not isinstance(self.tensor, Tensor) or self.tensor.ndim != 3 or self.tensor.shape[-1] != 4:
            raise ValueError("normalized T4 tensor must be floating [B,N,4]")
        if not self.tensor.is_floating_point() or self.tensor.shape[0] < 1 or self.tensor.shape[1] < 1:
            raise ValueError("normalized T4 tensor requires nonempty floating batch and unit axes")
        if not torch.isfinite(self.tensor).all().item():
            raise ValueError("normalized T4 tensor must be finite")
        _require_sha256(self.raw_authority_sha256, "raw_authority_sha256")
        _require_sha256(self.normalizer_authority_sha256, "normalizer_authority_sha256")
        _require_sha256(self.roster_digest, "roster_digest")
        _require_sha256(self.ordered_unit_digest, "ordered_unit_digest")
        if not isinstance(self.ordered_unit_ids, tuple) or len(self.ordered_unit_ids) != self.num_units:
            raise ValueError("ordered_unit_ids must match the normalized T4 unit axis")
        expected_digest = ordered_unit_digest(self.ordered_unit_ids)
        if self.ordered_unit_digest != expected_digest:
            raise ValueError("ordered_unit_digest does not bind ordered_unit_ids")
        if not isinstance(self.lineage, tuple) or not self.lineage:
            raise ValueError("T4 lineage must be a nonempty tuple")
        if any(not isinstance(item, str) or not item for item in self.lineage):
            raise ValueError("T4 lineage entries must be nonempty strings")
        if self.diagnostic_mode not in _DIAGNOSTIC_MODES:
            raise ValueError("diagnostic_mode must be aligned, zero, or wrong_pair")

    def joint_permute(self, permutation: Tensor) -> "NormalizedT4Batch":
        """Jointly permute an aligned capability's rows and its ordered lineage."""
        self.validate()
        if self.diagnostic_mode != "aligned":
            raise ValueError("joint_permute is only defined for an aligned T4 capability")
        if permutation.ndim != 1 or permutation.numel() != self.num_units or permutation.dtype != torch.long:
            raise ValueError("permutation must be a torch.long [N] tensor")
        if permutation.device != self.tensor.device:
            raise ValueError("permutation and normalized T4 must share a device")
        expected = torch.arange(self.num_units, device=permutation.device, dtype=torch.long)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError("permutation must contain each unit index exactly once")
        ids = tuple(self.ordered_unit_ids[index] for index in permutation.detach().cpu().tolist())
        return NormalizedT4Batch(
            tensor=self.tensor.index_select(1, permutation).detach().clone(),
            raw_authority_sha256=self.raw_authority_sha256,
            normalizer_authority_sha256=self.normalizer_authority_sha256,
            roster_digest=self.roster_digest,
            ordered_unit_digest=ordered_unit_digest(ids),
            ordered_unit_ids=ids,
            lineage=self.lineage,
            diagnostic_mode="aligned",
        )


class T4Normalizer(nn.Module):
    """Frozen source-normalizer authority that constructs T4 capabilities."""

    def __init__(self, mean: Tensor, std: Tensor, raw_authority_sha256: str, normalizer_authority_sha256: str) -> None:
        super().__init__()
        if not isinstance(mean, Tensor) or not isinstance(std, Tensor):
            raise ValueError("T4 normalizer mean/std must be tensors")
        if mean.shape != (4,) or std.shape != (4,) or not mean.is_floating_point() or not std.is_floating_point():
            raise ValueError("T4 normalizer requires finite floating mean/std [4]")
        if not torch.isfinite(mean).all().item() or not torch.isfinite(std).all().item() or not torch.all(std > 0).item():
            raise ValueError("T4 normalizer requires finite mean and strictly positive finite std")
        _require_sha256(raw_authority_sha256, "raw_authority_sha256")
        _require_sha256(normalizer_authority_sha256, "normalizer_authority_sha256")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        self.raw_authority_sha256 = raw_authority_sha256
        self.normalizer_authority_sha256 = normalizer_authority_sha256

    def forward(
        self,
        raw: Tensor,
        *,
        roster_digest: str,
        ordered_unit_ids: tuple[str, ...],
        lineage: tuple[str, ...],
    ) -> NormalizedT4Batch:
        if not isinstance(raw, Tensor) or raw.ndim != 3 or raw.shape[-1] != 4 or not raw.is_floating_point():
            raise ValueError("raw T4 must be a floating [B,N,4] tensor")
        if raw.shape[0] < 1 or raw.shape[1] < 1 or not torch.isfinite(raw).all().item():
            raise ValueError("raw T4 must have nonempty finite batch and unit axes")
        if raw.device != self.mean.device:
            raise ValueError("raw T4 and normalizer buffers must share a device")
        normalized = ((raw - self.mean) / self.std).detach().clone()
        return NormalizedT4Batch(
            tensor=normalized,
            raw_authority_sha256=self.raw_authority_sha256,
            normalizer_authority_sha256=self.normalizer_authority_sha256,
            roster_digest=roster_digest,
            ordered_unit_digest=ordered_unit_digest(ordered_unit_ids),
            ordered_unit_ids=ordered_unit_ids,
            lineage=lineage,
            diagnostic_mode="aligned",
        )

    def _validate_own_capability(self, normalized: NormalizedT4Batch) -> None:
        if not isinstance(normalized, NormalizedT4Batch):
            raise ValueError("T4 controls require a NormalizedT4Batch capability")
        normalized.validate()
        if (
            normalized.raw_authority_sha256 != self.raw_authority_sha256
            or normalized.normalizer_authority_sha256 != self.normalizer_authority_sha256
        ):
            raise ValueError("T4 capability authority does not match this normalizer")

    def zero(self, normalized: NormalizedT4Batch) -> NormalizedT4Batch:
        """Construct the explicitly typed zero-T4 diagnostic post-normalization."""
        self._validate_own_capability(normalized)
        return NormalizedT4Batch(
            tensor=torch.zeros_like(normalized.tensor),
            raw_authority_sha256=normalized.raw_authority_sha256,
            normalizer_authority_sha256=normalized.normalizer_authority_sha256,
            roster_digest=normalized.roster_digest,
            ordered_unit_digest=normalized.ordered_unit_digest,
            ordered_unit_ids=normalized.ordered_unit_ids,
            lineage=normalized.lineage,
            diagnostic_mode="zero",
        )

    def wrong_pair(self, normalized: NormalizedT4Batch, permutation: Tensor | None = None) -> NormalizedT4Batch:
        """Construct an explicitly typed row-mispairing diagnostic post-normalization."""
        self._validate_own_capability(normalized)
        if permutation is None:
            permutation = torch.arange(normalized.num_units - 1, -1, -1, device=normalized.tensor.device)
        if permutation.ndim != 1 or permutation.numel() != normalized.num_units or permutation.dtype != torch.long:
            raise ValueError("wrong-pair permutation must be a torch.long [N] tensor")
        if permutation.device != normalized.tensor.device:
            raise ValueError("wrong-pair permutation and T4 capability must share a device")
        expected = torch.arange(normalized.num_units, device=permutation.device, dtype=torch.long)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError("wrong-pair permutation must contain each unit index exactly once")
        if torch.equal(permutation, expected):
            raise ValueError("wrong-pair diagnostic requires a nonidentity permutation")
        return NormalizedT4Batch(
            tensor=normalized.tensor.index_select(1, permutation).detach().clone(),
            raw_authority_sha256=normalized.raw_authority_sha256,
            normalizer_authority_sha256=normalized.normalizer_authority_sha256,
            roster_digest=normalized.roster_digest,
            ordered_unit_digest=normalized.ordered_unit_digest,
            ordered_unit_ids=normalized.ordered_unit_ids,
            lineage=normalized.lineage,
            diagnostic_mode="wrong_pair",
        )


class B3S(nn.Module):
    """Exact shared M30 B3S dimensions: 18,290 trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_pool = nn.Sequential(nn.Linear(100, 64), nn.ReLU())
        self.post_pool = nn.Sequential(nn.Linear(68, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 50))

    def forward(self, calib: Tensor, normalized_t4: NormalizedT4Batch) -> Tensor:
        if not isinstance(normalized_t4, NormalizedT4Batch):
            raise ValueError("B3S requires a NormalizedT4Batch capability, never a bare T4 tensor")
        normalized_t4.validate()
        if (
            not isinstance(calib, Tensor)
            or calib.ndim != 4
            or calib.shape[1:3] != (30, 100)
            or not calib.is_floating_point()
            or not torch.isfinite(calib).all().item()
        ):
            raise ValueError("calib must be finite floating [B,30,100,N]")
        if calib.shape[0] != normalized_t4.batch_size or calib.shape[3] != normalized_t4.num_units:
            raise ValueError("calib/T4 unit alignment mismatch")
        if calib.device != normalized_t4.tensor.device or calib.dtype != normalized_t4.tensor.dtype:
            raise ValueError("calib and normalized T4 capability must share device and dtype")
        features = self.pre_pool(calib.permute(0, 1, 3, 2)).mean(dim=1)
        return self.post_pool(torch.cat((features, normalized_t4.tensor), dim=-1))


class TFSRDecoder(nn.Module):
    """Frozen TF-SR topology; state is zeroed for every forward/window."""

    def __init__(
        self,
        dropout_low: float = 0.0,
        dropout_high: float = 1.0,
        *,
        capture_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        if (
            isinstance(dropout_low, bool)
            or isinstance(dropout_high, bool)
            or not isinstance(dropout_low, (int, float))
            or not isinstance(dropout_high, (int, float))
            or not math.isfinite(float(dropout_low))
            or not math.isfinite(float(dropout_high))
            or not 0.0 <= float(dropout_low) <= float(dropout_high) <= 1.0
        ):
            raise ValueError("dropout bounds must be finite with 0 <= low <= high <= 1")
        if type(capture_diagnostics) is not bool:
            raise ValueError("capture_diagnostics must be an exact bool")
        self.dropout_low, self.dropout_high = float(dropout_low), float(dropout_high)
        self.capture_diagnostics = capture_diagnostics
        self.b3s = B3S()
        self.activity_encoder = CausalActivityEncoder(20, 256, 64)
        self.unit_mlp = nn.Sequential(nn.Linear(114, 256), nn.ReLU(), nn.Linear(256, 256))
        self.query_base = nn.Parameter(torch.empty(2, 256))
        nn.init.normal_(self.query_base)
        self.state_query = nn.Linear(256, 512)
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(256), nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 256))
        self.gru = nn.GRU(515, 256, batch_first=True)
        self.head = nn.Linear(256, 2)
        self.last_dropout_p: Tensor | None = None
        self.last_unit_gain_mask: Tensor | None = None
        self.last_unit_survivor_mask: Tensor | None = None
        self.last_unit_mask: Tensor | None = None  # Compatibility alias: the gain-valued Cell-D mask.
        self.last_pre_mask_tokens: Tensor | None = None
        self.last_post_mask_tokens: Tensor | None = None
        self.last_activity_features: Tensor | None = None
        self.last_activity_mass: Tensor | None = None
        self.last_mass: Tensor | None = None
        self.last_readin: Tensor | None = None
        self.last_prediction: Tensor | None = None
        self.last_states: Tensor | None = None

    @staticmethod
    def _finite_floating(tensor: Tensor, name: str) -> None:
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must be a finite floating tensor")

    def _validate(self, x: Tensor, calib: Tensor, t4: NormalizedT4Batch) -> None:
        self._finite_floating(x, "x")
        self._finite_floating(calib, "calib")
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] != 50 or x.shape[2] < 1:
            raise ValueError("x must be finite floating [B,50,N] with nonempty B/N")
        if calib.shape != (x.shape[0], 30, 100, x.shape[2]):
            raise ValueError("query/calibration unit-axis alignment failure")
        if not isinstance(t4, NormalizedT4Batch):
            raise ValueError("TFSRDecoder requires a NormalizedT4Batch capability, never a bare T4 tensor")
        t4.validate()
        if t4.batch_size != x.shape[0] or t4.num_units != x.shape[2]:
            raise ValueError("query/T4 capability unit-axis alignment failure")
        if x.device != calib.device or x.device != t4.tensor.device or x.dtype != calib.dtype or x.dtype != t4.tensor.dtype:
            raise ValueError("query, calibration, and T4 capability must share device and dtype")

    def _mask(self, tokens: Tensor) -> Tensor:
        """Apply the exact Cell-D law once, after complete fused-token formation."""
        batch, _, units, _ = tokens.shape
        gain_mask = torch.ones((batch, units), device=tokens.device, dtype=tokens.dtype)
        if self.training:
            # Deliberately Python RNG plus exactly one F.dropout call on [B,N].
            sampled_p = random.uniform(self.dropout_low, self.dropout_high)
            gain_mask = F.dropout(gain_mask, p=sampled_p, training=True)
            self.last_dropout_p = tokens.new_tensor(sampled_p).detach()
        else:
            self.last_dropout_p = None
        survivor_mask = gain_mask.ne(0)
        self.last_unit_gain_mask = gain_mask.detach()
        self.last_unit_survivor_mask = survivor_mask.detach()
        self.last_unit_mask = self.last_unit_gain_mask
        return tokens * gain_mask[:, None, :, None]

    @staticmethod
    def _population_mass(activity: Tensor, survivor_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return disclosure channels and unscaled retained-only activity mass."""
        retained = survivor_mask.to(dtype=activity.dtype)
        retained_count = retained.sum(dim=1)
        retained_fraction = retained.mean(dim=1)
        retained_abs_sum = (activity.abs() * retained[:, None, :, None]).sum(dim=(2, 3))
        denominator = retained_count[:, None] * activity.shape[-1]
        activity_mean = torch.where(
            denominator > 0,
            retained_abs_sum / denominator.clamp_min(1),
            torch.zeros_like(retained_abs_sum),
        )
        time_steps = activity.shape[1]
        count_channel = torch.log1p(retained_count)[:, None].expand(-1, time_steps)
        fraction_channel = retained_fraction[:, None].expand(-1, time_steps)
        return torch.stack((count_channel, fraction_channel, torch.log1p(activity_mean)), dim=-1), activity_mean

    def forward(
        self,
        x: Tensor,
        calib: Tensor,
        normalized_t4: NormalizedT4Batch,
        return_states: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        self._validate(x, calib, normalized_t4)
        identity = self.b3s(calib, normalized_t4)
        activity = self.activity_encoder(x)
        pre_mask_tokens = self.unit_mlp(torch.cat((activity, identity[:, None].expand(-1, 50, -1, -1)), dim=-1))
        tokens = self._mask(pre_mask_tokens)
        if self.last_unit_survivor_mask is None:  # Defensive; _mask always sets it.
            raise RuntimeError("unit-mask diagnostic was not set")
        mass, activity_mass = self._population_mass(activity, self.last_unit_survivor_mask)
        hidden = x.new_zeros(1, x.shape[0], 256)
        outputs: list[Tensor] = []
        states: list[Tensor] | None = [] if return_states or self.capture_diagnostics else None
        readins: list[Tensor] | None = [] if self.capture_diagnostics else None
        for time_index in range(50):
            queries = self.query_base[None] + self.state_query(hidden[0]).view(x.shape[0], 2, 256)
            read, _ = self.attn(queries, tokens[:, time_index], tokens[:, time_index], need_weights=False)
            read = self.norm1(queries + read)
            read = self.norm2(read + self.ffn(read))
            step = torch.cat((read.flatten(1), mass[:, time_index]), dim=1).unsqueeze(1)
            _, hidden = self.gru(step, hidden)
            if states is not None:
                states.append(hidden[0])
            outputs.append(self.head(hidden[0]))
            if readins is not None:
                readins.append(read)
        prediction = torch.stack(outputs, dim=1)
        state_tensor = torch.stack(states, dim=1) if states is not None else None
        if self.capture_diagnostics:
            self.last_pre_mask_tokens = pre_mask_tokens.detach()
            self.last_post_mask_tokens = tokens.detach()
            self.last_activity_features = activity.detach()
            self.last_activity_mass = activity_mass.detach()
            self.last_mass = mass.detach()
            self.last_readin = torch.stack(readins, dim=1).detach() if readins is not None else None
            self.last_prediction = prediction.detach()
            self.last_states = state_tensor.detach() if state_tensor is not None else None
        else:
            self.last_pre_mask_tokens = None
            self.last_post_mask_tokens = None
            self.last_activity_features = None
            self.last_activity_mass = None
            self.last_mass = None
            self.last_readin = None
            self.last_prediction = None
            self.last_states = None
        if return_states:
            if state_tensor is None:  # Defensive; return_states makes the history mandatory.
                raise RuntimeError("state history was not retained for return_states=True")
            return prediction, state_tensor
        return prediction

    @staticmethod
    def dense_valid_bin_mse(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        if not isinstance(prediction, Tensor) or not isinstance(target, Tensor) or not isinstance(valid, Tensor):
            raise ValueError("prediction, target, and valid must be tensors")
        if prediction.shape != target.shape or prediction.ndim != 3 or valid.shape != prediction.shape[:2]:
            raise ValueError("output/loss interface mismatch")
        if prediction.device != target.device or prediction.device != valid.device:
            raise ValueError("prediction, target, and valid must share a device")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError("prediction and target must be floating tensors")
        if not torch.isfinite(prediction).all().item() or not torch.isfinite(target).all().item():
            raise ValueError("prediction and target must be finite")
        if valid.dtype == torch.bool:
            valid_bool = valid
        elif valid.is_floating_point() or valid.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            if not torch.isfinite(valid).all().item() or not torch.logical_or(valid == 0, valid == 1).all().item():
                raise ValueError("valid mask must contain only finite 0/1 values")
            valid_bool = valid.to(dtype=torch.bool)
        else:
            raise ValueError("valid mask must be bool or a finite 0/1 numeric tensor")
        valid_count = int(valid_bool.sum().item())
        if valid_count == 0:
            raise ValueError("valid mask must select at least one time bin")
        per_bin_mse = (prediction - target).square().mean(dim=-1)
        return (per_bin_mse * valid_bool.to(dtype=prediction.dtype)).sum() / valid_count

    def accounting(self, representative_n: int = 128) -> dict[str, float | str]:
        if isinstance(representative_n, bool) or not isinstance(representative_n, int) or representative_n < 1:
            raise ValueError("representative_n must be a positive integer")
        blocks = {
            "b3s": self.b3s,
            "activity": self.activity_encoder,
            "token": self.unit_mlp,
            "readin": nn.ModuleList([self.state_query, self.attn, self.norm1, self.norm2, self.ffn]),
            "gru": self.gru,
            "head": self.head,
        }
        counts = {name: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad) for name, module in blocks.items()}
        counts["readin"] += self.query_base.numel()
        total = sum(counts.values())
        n = representative_n
        analytic_mac = n * (
            30 * 100 * 64 + 68 * 64 + 64 * 64 + 64 * 50 + 50 * (20 * 256 + 256 * 64 + 114 * 256 + 256 * 256)
        )
        analytic_mac += 50 * (
            2 * 256 * 256 + 2 * n * 256 * 256 + 4 * n * 256 + 2 * 256 * 256
            + 2 * (256 * 1024 + 1024 * 256) + 3 * (515 * 256 + 256 * 256) + 256 * 2
        )
        return {
            "trainable_params": float(total),
            **{f"params_{name}": float(value) for name, value in counts.items()},
            "analytic_mac_estimate_n": float(analytic_mac),
            "analytic_mac_representative_n": float(n),
            "persistent_state_bytes": float(256 * 4),
            "state_bytes": float(256 * 4),
            "analytic_activation_estimate_bytes": float(50 * representative_n * 256 * 4),
            "gpu_memory_envelope": "DEFERRED_NO_GPU_STAGE0",
        }

    def cpu_latency_ms(
        self,
        x: Tensor,
        calib: Tensor,
        normalized_t4: NormalizedT4Batch,
        *,
        warmup: int = 1,
        repeats: int = 3,
    ) -> float:
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("warmup must be an integer >= 1")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
            raise ValueError("repeats must be an integer >= 3")
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                for _ in range(warmup):
                    self(x, calib, normalized_t4)
                elapsed_ms: list[float] = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    self(x, calib, normalized_t4)
                    elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        finally:
            self.train(was_training)
        return float(statistics.median(elapsed_ms))
