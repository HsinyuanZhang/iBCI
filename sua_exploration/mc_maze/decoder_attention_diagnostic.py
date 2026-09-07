"""Checkpoint-bound, descriptive A12 attention capture helpers.

This module deliberately implements only the ordinary coupled B3S decoder
path used by the canonical SUA T4/Z4 comparison.  It is *not* a generic SPINT
diagnostic framework: it refuses decoupled, fixed-slot, M2, H1, synthetic
receipt, identity-zeroing, and within-checkpoint intervention paths.

The capture patch mirrors the actual
``streaming_calibration_exp/src/models/components/spint.py``
``CrossAttentionLayer.forward`` exactly, except that it asks
``nn.MultiheadAttention`` for per-head weights.  It records the normalized
Q/K/V inputs and their exact projections, while preserving the prediction
tensor bit-for-bit; callers must enforce that parity before publishing any
receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F


SCHEMA_VERSION = "a12_descriptive_attention_audit_v4"
STRICT_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
INPUT_SHAPE_CONTRACT = "neural=[B,W,N], calibration=[B,M,T,N], side=[B,N,4]"
CANONICAL_WINDOW_SIZE = 50
CANONICAL_SUPPORT_M = 30
CANONICAL_TRIAL_LENGTH = 100

SEALED_FORMAL_TEST_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
)

DEFAULT_VALIDATION_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
)


class A12ForwardError(RuntimeError):
    """Raised when an A12 descriptive-forward invariant is violated."""


class SealedSessionError(A12ForwardError, ValueError):
    """Raised when a sealed formal-test session is requested."""


def sha256_file(path: Path) -> str:
    """Hash a regular input file without deserializing it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_sealed_sessions(session_names: Sequence[str]) -> None:
    """Reject every formal-test session before any data/model operation."""

    blocked = sorted(set(session_names) & set(SEALED_FORMAL_TEST_SESSIONS))
    if blocked:
        raise SealedSessionError(
            "refusing sealed formal-test session(s): " + ", ".join(blocked)
        )


def assert_cpu_only() -> torch.device:
    """Assert the no-GPU boundary used by the real A12 runner."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise A12ForwardError("A12 requires CUDA_VISIBLE_DEVICES='' before Torch use")
    if torch.cuda.is_available():
        raise A12ForwardError("A12 descriptive forward refuses an available CUDA runtime")
    return torch.device("cpu")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A12ForwardError(message)


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def stack_windows_bwn(windows: Sequence[torch.Tensor], *, window_size: int = CANONICAL_WINDOW_SIZE) -> torch.Tensor:
    """Stack ``[W,N]`` decoder windows as ``[B,W,N]`` without a transpose.

    The explicit leading-window assertion catches the historical accidental
    ``window.T`` conversion, which produced ``[B,N,W]``.
    """

    _require(bool(windows), "at least one decoder window is required")
    normalized: list[torch.Tensor] = []
    units: int | None = None
    for index, window in enumerate(windows):
        _require(isinstance(window, torch.Tensor), f"window {index} is not a Tensor")
        _require(window.ndim == 2, f"window {index} must be [W,N], got {_shape(window)}")
        _require(
            int(window.shape[0]) == window_size,
            f"window {index} must have W={window_size} in axis 0; refusing [N,W] input {_shape(window)}",
        )
        if units is None:
            units = int(window.shape[1])
        _require(int(window.shape[1]) == units, "all decoder windows must share N")
        normalized.append(window)
    result = torch.stack(normalized, dim=0)
    _require(result.ndim == 3 and int(result.shape[1]) == window_size, "BWN stack drift")
    return result


def assert_b3s_input_shapes(
    neural: torch.Tensor,
    calibration: torch.Tensor,
    side_features: torch.Tensor,
    *,
    canonical: bool = True,
) -> None:
    """Validate the actual B3S loader tensor contract.

    ``MCMazeSessionDataset`` yields individual neural windows as ``[W,N]``;
    its DataLoader creates the required ``[B,W,N]`` batch.  Calibration and
    per-unit side features retain their B3S batch dimensions.
    """

    _require(neural.ndim == 3, f"neural must be [B,W,N], got {_shape(neural)}")
    _require(
        calibration.ndim == 4,
        f"calibration must be [B,M,T,N], got {_shape(calibration)}",
    )
    _require(side_features.ndim == 3, f"side must be [B,N,4], got {_shape(side_features)}")
    batch, width, units = _shape(neural)
    cal_batch, support, trial_length, cal_units = _shape(calibration)
    side_batch, side_units, side_dim = _shape(side_features)
    _require(batch > 0 and units > 0, "neural B and N must be positive")
    _require((cal_batch, cal_units) == (batch, units), "calibration B/N must match neural")
    _require((side_batch, side_units, side_dim) == (batch, units, 4), "side must be [B,N,4]")
    if canonical:
        _require(width == CANONICAL_WINDOW_SIZE, f"A12 requires W={CANONICAL_WINDOW_SIZE}, got {width}")
        _require(support == CANONICAL_SUPPORT_M, f"A12 requires M={CANONICAL_SUPPORT_M}, got {support}")
        _require(
            trial_length == CANONICAL_TRIAL_LENGTH,
            f"A12 requires T={CANONICAL_TRIAL_LENGTH}, got {trial_length}",
        )


def _copy_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Stable CPU digest including dtype and shape, suitable for provenance."""

    normalized = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(json.dumps(list(normalized.shape), separators=(",", ":")).encode("ascii"))
    digest.update(normalized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash all state tensors without serialization or mutation."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _require_actual_coupled_b3s(student: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    """Return the actual ordinary coupled decoder layers or fail closed.

    The check intentionally uses the runtime B3S object graph rather than a
    stale import from ``SPINT-main``.  The runner separately asserts that the
    imported ``src`` package originates in ``streaming_calibration_exp``.
    """

    _require(
        getattr(student, "decoder_mode", None) == "coupled",
        "A12 accepts only the ordinary coupled B3S decoder",
    )
    _require(
        getattr(student, "fixed_slot_router", None) is None,
        "A12 refuses fixed-slot decoder paths",
    )
    _require(
        getattr(student, "decoupled_transformer", None) is None,
        "A12 refuses decoupled K/V decoder paths",
    )
    decoder = getattr(student, "decoder", None)
    transformer = getattr(decoder, "transformer", None)
    raw_layers = getattr(transformer, "layers", None)
    _require(raw_layers is not None, "B3S decoder transformer.layers is missing")
    layers = tuple(raw_layers)
    _require(len(layers) == 1, "A12 is bound to the canonical one-layer B3S decoder")
    layer = layers[0]
    mha = getattr(layer, "cross_attn", None)
    _require(isinstance(mha, torch.nn.MultiheadAttention), "B3S cross_attn must be nn.MultiheadAttention")
    _require(bool(mha.batch_first), "B3S cross_attn must be batch_first=True")
    _require(mha.bias_k is None and mha.bias_v is None and not mha.add_zero_attn,
             "A12 refuses noncanonical MultiheadAttention key/value augmentation")
    _require(hasattr(layer, "norm1") and hasattr(layer, "norm2") and hasattr(layer, "ffn") and hasattr(layer, "dropout"),
             "B3S CrossAttentionLayer structure drift")
    return layers


def resolve_b3s_student(model: torch.nn.Module) -> torch.nn.Module:
    """Resolve the actual Lightning checkpoint wrapper's B3S student module."""

    student = getattr(model, "student", None)
    _require(student is not None and isinstance(student, torch.nn.Module), "A12 requires a loaded B3S checkpoint wrapper with .student")
    _require_actual_coupled_b3s(student)
    return student


def project_qkv(
    mha: torch.nn.MultiheadAttention,
    query_normalized: torch.Tensor,
    key_normalized: torch.Tensor,
    value_normalized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the actual fused MHA matrix in exact Q, K, V order."""

    _require(bool(mha._qkv_same_embed_dim), "A12 requires one fused B3S Q/K/V projection")
    _require(mha.in_proj_weight is not None, "B3S MHA in_proj_weight is missing")
    embed_dim = int(mha.embed_dim)
    for name, tensor in (
        ("query", query_normalized),
        ("key", key_normalized),
        ("value", value_normalized),
    ):
        _require(tensor.ndim == 3 and int(tensor.shape[-1]) == embed_dim,
                 f"{name} normalized input must be [B,tokens,{embed_dim}], got {_shape(tensor)}")
    _require(int(query_normalized.shape[0]) == int(key_normalized.shape[0]) == int(value_normalized.shape[0]),
             "Q/K/V batch dimensions drift")
    weight = mha.in_proj_weight
    bias = mha.in_proj_bias
    _require(tuple(weight.shape) == (3 * embed_dim, embed_dim), "B3S fused projection shape drift")
    if bias is not None:
        _require(tuple(bias.shape) == (3 * embed_dim,), "B3S fused projection bias shape drift")
    q = F.linear(query_normalized, weight[0:embed_dim], None if bias is None else bias[0:embed_dim])
    k = F.linear(key_normalized, weight[embed_dim : 2 * embed_dim], None if bias is None else bias[embed_dim : 2 * embed_dim])
    v = F.linear(value_normalized, weight[2 * embed_dim : 3 * embed_dim], None if bias is None else bias[2 * embed_dim : 3 * embed_dim])
    return q, k, v


@dataclass(frozen=True)
class CapturedCrossAttention:
    """One exact coupled B3S layer invocation and its per-head map."""

    layer_index: int
    mha: torch.nn.MultiheadAttention
    query_normalized: torch.Tensor
    key_normalized: torch.Tensor
    value_normalized: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attention: torch.Tensor


@dataclass(frozen=True)
class ForwardCapture:
    """A no-grad B3S prediction plus its captured Q/K/V attention records."""

    prediction: torch.Tensor
    identity: torch.Tensor
    layers: tuple[CapturedCrossAttention, ...]


@contextmanager
def capture_coupled_cross_attention(student: torch.nn.Module) -> Iterator[list[CapturedCrossAttention]]:
    """Temporarily retain exact Q/K/V and per-head maps from real B3S layers.

    No parameters, buffers, input tensor, identity, or decoder path are
    changed.  The only reporting change is ``average_attn_weights=False``.
    """

    layers = _require_actual_coupled_b3s(student)
    captures: list[CapturedCrossAttention] = []
    originals: list[tuple[torch.nn.Module, Any]] = []
    for layer_index, layer in enumerate(layers):
        original = layer.forward

        def patched_forward(
            query: torch.Tensor,
            key_value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None,
            *,
            _layer: torch.nn.Module = layer,
            _layer_index: int = layer_index,
            _original: Any = original,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Preserve the production layer invocation itself verbatim.  The
            # second MHA call below is reporting-only and obtains the per-head
            # map that the production call otherwise averages before return.
            # This makes prediction parity exact by construction rather than
            # relying on two nominally equivalent MHA code paths.
            x, production_attention = _original(
                query,
                key_value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            query_normalized = _layer.norm1(query)
            key_normalized = _layer.norm1(key_value)
            # The canonical coupled B3S decoder uses exactly the same normalized
            # key-value input for K and V.  We retain distinct fields so any
            # future architectural drift is visible rather than mislabeled.
            value_normalized = key_normalized
            _, attention = _layer.cross_attn(
                query=query_normalized,
                key=key_normalized,
                value=value_normalized,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            _require(attention.ndim == 4, f"per-head attention must be [B,H,C,N], got {_shape(attention)}")
            q, k, v = project_qkv(
                _layer.cross_attn,
                query_normalized,
                key_normalized,
                value_normalized,
            )
            captures.append(
                CapturedCrossAttention(
                    layer_index=_layer_index,
                    mha=_layer.cross_attn,
                    query_normalized=_copy_tensor(query_normalized),
                    key_normalized=_copy_tensor(key_normalized),
                    value_normalized=_copy_tensor(value_normalized),
                    q=_copy_tensor(q),
                    k=_copy_tensor(k),
                    v=_copy_tensor(v),
                    attention=_copy_tensor(attention),
                )
            )
            # Keep the production attention-return contract unchanged too:
            # MultiLayerCrossAttention callers may inspect its averaged map.
            return x, production_attention

        originals.append((layer, original))
        layer.forward = patched_forward  # type: ignore[method-assign]
    try:
        yield captures
    finally:
        for layer, original in originals:
            layer.forward = original  # type: ignore[method-assign]


def plain_b3s_forward(
    model: torch.nn.Module,
    neural: torch.Tensor,
    calibration: torch.Tensor,
    side_features: torch.Tensor,
    *,
    electrode_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the actual unpatched B3S checkpoint forward with no query labels."""

    assert_b3s_input_shapes(neural, calibration, side_features)
    student = resolve_b3s_student(model)
    _require(not model.training and not student.training, "A12 requires model.eval() before forward")
    with torch.no_grad():
        prediction, identity = student(
            neural,
            calib_trials=calibration,
            side_features=side_features,
            electrode_ids=electrode_ids,
        )
    return prediction, identity


def forward_b3s_with_capture(
    model: torch.nn.Module,
    neural: torch.Tensor,
    calibration: torch.Tensor,
    side_features: torch.Tensor,
    *,
    electrode_ids: torch.Tensor | None = None,
) -> ForwardCapture:
    """Run the same B3S forward while recording one descriptive attention map."""

    assert_b3s_input_shapes(neural, calibration, side_features)
    student = resolve_b3s_student(model)
    _require(not model.training and not student.training, "A12 requires model.eval() before forward")
    with torch.no_grad(), capture_coupled_cross_attention(student) as layers:
        prediction, identity = student(
            neural,
            calib_trials=calibration,
            side_features=side_features,
            electrode_ids=electrode_ids,
        )
    _require(len(layers) == 1, f"canonical one-layer B3S forward produced {len(layers)} captured layers")
    return ForwardCapture(prediction=_copy_tensor(prediction), identity=_copy_tensor(identity), layers=tuple(layers))


def assert_output_parity(reference: torch.Tensor, captured: torch.Tensor) -> None:
    """Fail closed unless capture-on/off predictions are exactly equal."""

    if torch.equal(reference, captured):
        return
    if reference.shape != captured.shape:
        raise A12ForwardError(
            f"A12 capture changed prediction shape: {_shape(reference)} != {_shape(captured)}"
        )
    difference = float((reference - captured).abs().max().detach().cpu().item())
    raise A12ForwardError(
        "A12 capture changed B3S prediction values; max absolute difference="
        f"{difference:.9g}"
    )


def head_contribution_norms(
    mha: torch.nn.MultiheadAttention,
    value_projection: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    """Return per-window/head/covariate contribution norms using **V**, not K.

    The canonical attention output concatenates all head value mixtures and
    multiplies them by ``out_proj``.  This function preserves that geometry by
    applying each head's input-column block of the real ``out_proj`` to its
    value-weighted output before taking its L2 norm.
    """

    _require(attention.ndim == 4, f"attention must be [B,H,C,N], got {_shape(attention)}")
    _require(value_projection.ndim == 3, f"V must be [B,N,E], got {_shape(value_projection)}")
    batch, heads, covariates, units = _shape(attention)
    v_batch, v_units, embed_dim = _shape(value_projection)
    _require((v_batch, v_units) == (batch, units), "V B/N must match attention")
    _require(int(mha.num_heads) == heads and int(mha.embed_dim) == embed_dim,
             "MHA head/embed dimensions drift")
    _require(embed_dim % heads == 0, "MHA embed dimension must divide heads")
    head_dim = embed_dim // heads
    values_by_head = value_projection.reshape(batch, units, heads, head_dim).permute(0, 2, 1, 3)
    attended = torch.einsum("bhcn,bhnd->bhcd", attention, values_by_head)
    output_weight = mha.out_proj.weight
    _require(tuple(output_weight.shape) == (embed_dim, embed_dim), "MHA out_proj shape drift")
    per_head_outputs: list[torch.Tensor] = []
    for head in range(heads):
        block = output_weight[:, head * head_dim : (head + 1) * head_dim]
        per_head_outputs.append(torch.einsum("bcd,ed->bce", attended[:, head], block))
    stacked = torch.stack(per_head_outputs, dim=1)
    return torch.linalg.vector_norm(stacked, ord=2, dim=-1)


def _mean_pairwise_head_cosine(attention: torch.Tensor) -> float:
    _, heads, _, _ = _shape(attention)
    if heads < 2:
        return 1.0
    flattened = attention.mean(dim=0).reshape(heads, -1)
    normalized = F.normalize(flattened, p=2, dim=1, eps=1.0e-12)
    cosine = normalized @ normalized.transpose(0, 1)
    mask = torch.triu(torch.ones((heads, heads), dtype=torch.bool, device=cosine.device), diagonal=1)
    return float(cosine[mask].mean().detach().cpu().item())


def summarize_attention_captures(captures: Sequence[CapturedCrossAttention]) -> dict[str, Any]:
    """Produce descriptive session metrics from one layer across query windows."""

    _require(bool(captures), "cannot summarize an empty attention capture")
    _require({record.layer_index for record in captures} == {0}, "A12 expects exactly layer 0")
    first = captures[0]
    attention = torch.cat([record.attention for record in captures], dim=0)
    query_normalized = torch.cat([record.query_normalized for record in captures], dim=0)
    key_normalized = torch.cat([record.key_normalized for record in captures], dim=0)
    value_normalized = torch.cat([record.value_normalized for record in captures], dim=0)
    q_projection = torch.cat([record.q for record in captures], dim=0)
    k_projection = torch.cat([record.k for record in captures], dim=0)
    values = torch.cat([record.v for record in captures], dim=0)
    batch, heads, covariates, units = _shape(attention)
    _require(units > 1, "attention entropy requires at least two units")
    _require(
        all(record.mha is first.mha for record in captures),
        "all session captures must originate from one loaded checkpoint layer",
    )
    entropy = -(attention * attention.clamp_min(1.0e-12).log()).sum(dim=-1)
    normalized_entropy = entropy / math.log(units)
    effective_support = entropy.exp()
    inverse_participation_ratio = attention.square().sum(dim=-1).reciprocal()
    contribution = head_contribution_norms(first.mha, values, attention)
    summary = {
        "layer_index": 0,
        "num_windows": batch,
        "num_heads": heads,
        "num_covariates": covariates,
        "num_units": units,
        "mean_normalized_entropy": float(normalized_entropy.mean().detach().cpu().item()),
        "mean_effective_attended_units": float(effective_support.mean().detach().cpu().item()),
        "mean_inverse_participation_ratio": float(inverse_participation_ratio.mean().detach().cpu().item()),
        "mean_max_probability": float(attention.max(dim=-1).values.mean().detach().cpu().item()),
        "head_contribution_l2_mean": float(contribution.mean().detach().cpu().item()),
        "head_contribution_l2_per_head": [
            float(value) for value in contribution.mean(dim=(0, 2)).detach().cpu().tolist()
        ],
        "mean_pairwise_head_cosine": _mean_pairwise_head_cosine(attention),
        "variance_across_windows": float(attention.var(dim=0, unbiased=False).mean().detach().cpu().item()),
        "variance_across_covariates": float(attention.var(dim=2, unbiased=False).mean().detach().cpu().item()),
        "attention_tensor_sha256": tensor_sha256(attention),
        "query_normalized_sha256": tensor_sha256(query_normalized),
        "key_normalized_sha256": tensor_sha256(key_normalized),
        "value_normalized_sha256": tensor_sha256(value_normalized),
        "q_projection_sha256": tensor_sha256(q_projection),
        "k_projection_sha256": tensor_sha256(k_projection),
        "v_projection_sha256": tensor_sha256(values),
        "qkv_order": ["Q", "K", "V"],
        "value_projection_used_for_head_contribution": True,
    }
    return summary


class AttentionSummaryAccumulator:
    """Streaming equivalent of :func:`summarize_attention_captures`.

    A12 query pools are much larger than the final descriptive statistics.
    Retaining every Q/K/V and attention tensor until the end can consume many
    gigabytes on CPU, even though all reported quantities are first/second
    moments plus a small attention-map mean.  This accumulator consumes one
    dataloader batch at a time and keeps only those sufficient statistics.  A
    declared total window count lets its tensor digests use exactly the same
    ``dtype + final-shape + raw-bytes`` convention as ``tensor_sha256``.

    It is intentionally scoped to the same single coupled B3S layer as the
    non-streaming summary.  The runner still performs the ordinary forward,
    capture parity check, and provenance checks for every batch.
    """

    _TENSOR_FIELDS = (
        ("attention", "attention"),
        ("query_normalized", "query_normalized"),
        ("key_normalized", "key_normalized"),
        ("value_normalized", "value_normalized"),
        ("q_projection", "q"),
        ("k_projection", "k"),
        ("v_projection", "v"),
    )

    def __init__(self, *, expected_num_windows: int) -> None:
        _require(expected_num_windows > 0, "expected_num_windows must be positive")
        self.expected_num_windows = int(expected_num_windows)
        self.num_windows = 0
        self._mha: torch.nn.MultiheadAttention | None = None
        self._heads: int | None = None
        self._covariates: int | None = None
        self._units: int | None = None
        self._attention_sum: torch.Tensor | None = None
        self._attention_sum_sq_by_position: torch.Tensor | None = None
        self._covariate_variance_sum = 0.0
        self._covariate_variance_count = 0
        self._entropy_sum = 0.0
        self._effective_support_sum = 0.0
        self._inverse_participation_sum = 0.0
        self._max_probability_sum = 0.0
        self._contribution_sum = 0.0
        self._contribution_per_head_sum: torch.Tensor | None = None
        self._summary_hashers: dict[str, hashlib._Hash] = {}
        self._summary_shapes: dict[str, tuple[int, ...]] = {}
        self._summary_dtypes: dict[str, torch.dtype] = {}

    def _register_tensor(self, name: str, tensor: torch.Tensor) -> None:
        normalized = tensor.detach().cpu().contiguous()
        _require(normalized.ndim >= 1, f"{name} must have a batch dimension")
        trailing = tuple(int(value) for value in normalized.shape[1:])
        if name not in self._summary_hashers:
            final_shape = (self.expected_num_windows,) + trailing
            digest = hashlib.sha256()
            digest.update(str(normalized.dtype).encode("ascii"))
            digest.update(json.dumps(list(final_shape), separators=(",", ":")).encode("ascii"))
            self._summary_hashers[name] = digest
            self._summary_shapes[name] = final_shape
            self._summary_dtypes[name] = normalized.dtype
        else:
            _require(self._summary_shapes[name][1:] == trailing, f"{name} trailing shape drift")
            _require(self._summary_dtypes[name] == normalized.dtype, f"{name} dtype drift")
        self._summary_hashers[name].update(normalized.view(torch.uint8).numpy().tobytes())

    def update(self, captures: Sequence[CapturedCrossAttention]) -> None:
        """Consume one or more batches, releasing their tensors after return."""

        _require(bool(captures), "cannot update an empty attention capture batch")
        for record in captures:
            _require(record.layer_index == 0, "A12 expects exactly layer 0")
            if self._mha is None:
                self._mha = record.mha
            _require(record.mha is self._mha, "all captures must use one checkpoint layer")
            attention = record.attention
            _require(attention.ndim == 4, f"attention must be [B,H,C,N], got {_shape(attention)}")
            batch, heads, covariates, units = _shape(attention)
            _require(units > 1, "attention entropy requires at least two units")
            if self._heads is None:
                self._heads, self._covariates, self._units = heads, covariates, units
                self._attention_sum = torch.zeros((heads, covariates, units), dtype=attention.dtype)
                self._attention_sum_sq_by_position = torch.zeros((heads, covariates, units), dtype=attention.dtype)
                self._contribution_per_head_sum = torch.zeros((heads,), dtype=attention.dtype)
            _require((heads, covariates, units) == (self._heads, self._covariates, self._units),
                     "attention H/C/N shape drift")

            for output_name, field_name in self._TENSOR_FIELDS:
                self._register_tensor(output_name, getattr(record, field_name))

            entropy = -(attention * attention.clamp_min(1.0e-12).log()).sum(dim=-1)
            normalized_entropy = entropy / math.log(units)
            effective_support = entropy.exp()
            inverse_participation_ratio = attention.square().sum(dim=-1).reciprocal()
            contribution = head_contribution_norms(self._mha, record.v, attention)

            self._entropy_sum += float(normalized_entropy.sum().detach().cpu().item())
            self._effective_support_sum += float(effective_support.sum().detach().cpu().item())
            self._inverse_participation_sum += float(inverse_participation_ratio.sum().detach().cpu().item())
            self._max_probability_sum += float(attention.max(dim=-1).values.sum().detach().cpu().item())
            self._contribution_sum += float(contribution.sum().detach().cpu().item())
            assert self._contribution_per_head_sum is not None
            self._contribution_per_head_sum += contribution.sum(dim=(0, 2)).detach().cpu()

            assert self._attention_sum is not None
            self._attention_sum += attention.detach().cpu().sum(dim=0)
            assert self._attention_sum_sq_by_position is not None
            self._attention_sum_sq_by_position += attention.detach().cpu().square().sum(dim=0)
            covariate_var = attention.var(dim=2, unbiased=False)
            self._covariate_variance_sum += float(covariate_var.sum().detach().cpu().item())
            self._covariate_variance_count += int(covariate_var.numel())
            self.num_windows += batch
        _require(self.num_windows <= self.expected_num_windows, "streaming attention window count exceeded declaration")

    def finalize(self) -> dict[str, Any]:
        """Return the same descriptive schema as ``summarize_attention_captures``."""

        _require(self.num_windows == self.expected_num_windows,
                 f"streaming attention saw {self.num_windows} windows; expected {self.expected_num_windows}")
        _require(self._mha is not None and self._attention_sum is not None and self._heads is not None
                 and self._covariates is not None and self._units is not None,
                 "cannot finalize an empty attention accumulator")
        _require(self._covariate_variance_count > 0 and self._attention_sum_sq_by_position is not None,
                 "streaming attention sufficient statistics are empty")
        count = float(self.num_windows)
        attention_mean = self._attention_sum / float(self.num_windows)
        flattened = attention_mean.reshape(self._heads, -1)
        normalized = F.normalize(flattened, p=2, dim=1, eps=1.0e-12)
        cosine = normalized @ normalized.transpose(0, 1)
        if self._heads < 2:
            pairwise_cosine = 1.0
        else:
            mask = torch.triu(torch.ones((self._heads, self._heads), dtype=torch.bool), diagonal=1)
            pairwise_cosine = float(cosine[mask].mean().item())
        variance_windows = (
            self._attention_sum_sq_by_position / count
            - (self._attention_sum / count).square()
        ).mean().item()
        contribution_per_head = self._contribution_per_head_sum / float(self.num_windows * self._covariates)  # type: ignore[operator]
        return {
            "layer_index": 0,
            "num_windows": self.num_windows,
            "num_heads": self._heads,
            "num_covariates": self._covariates,
            "num_units": self._units,
            "mean_normalized_entropy": self._entropy_sum / (self.num_windows * self._heads * self._covariates),
            "mean_effective_attended_units": self._effective_support_sum / (self.num_windows * self._heads * self._covariates),
            "mean_inverse_participation_ratio": self._inverse_participation_sum / (self.num_windows * self._heads * self._covariates),
            "mean_max_probability": self._max_probability_sum / (self.num_windows * self._heads * self._covariates),
            "head_contribution_l2_mean": self._contribution_sum / (self.num_windows * self._heads * self._covariates),
            "head_contribution_l2_per_head": [float(value) for value in contribution_per_head.tolist()],
            "mean_pairwise_head_cosine": pairwise_cosine,
            "variance_across_windows": variance_windows,
            "variance_across_covariates": self._covariate_variance_sum / self._covariate_variance_count,
            "attention_tensor_sha256": self._summary_hashers["attention"].copy().hexdigest(),
            "query_normalized_sha256": self._summary_hashers["query_normalized"].copy().hexdigest(),
            "key_normalized_sha256": self._summary_hashers["key_normalized"].copy().hexdigest(),
            "value_normalized_sha256": self._summary_hashers["value_normalized"].copy().hexdigest(),
            "q_projection_sha256": self._summary_hashers["q_projection"].copy().hexdigest(),
            "k_projection_sha256": self._summary_hashers["k_projection"].copy().hexdigest(),
            "v_projection_sha256": self._summary_hashers["v_projection"].copy().hexdigest(),
            "qkv_order": ["Q", "K", "V"],
            "value_projection_used_for_head_contribution": True,
        }
