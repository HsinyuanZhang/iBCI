"""Route-owned Cell W: the temporal latent residual decoder (handoff 2026-08-18 §3 / directions 2026-08-17 §5 Priority 2).

Implements HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3
"Cell W - output head and within-window structure" to the frozen spec of
HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md §5 Priority 2, without touching
any shared SPINT / Arm A / D / DH / R / S2 / T / G / C file:

    unit_tokens = fc_in(activity + identity)                  # unchanged path
    latent_k    = cross_attention(K learned queries, unit_tokens)   # K = 8, FROZEN
    delta[t, c] = sum_k temporal_basis[t, k] * value_head(latent_k)[k, c]
    prediction  = base_output + delta                          # delta == 0 at init

- The new cross-attention is ONE ``nn.MultiheadAttention(512, 2,
  batch_first=True)`` (heads = 2, dropout = 0.0, PyTorch default init); query =
  the K learned query slots ``[K, 512]`` expanded over batch, key/value =
  ``unit_tokens``.  The existing decoder transformer is NOT reused, NOT
  modified, and no other attention machinery is added.
- ``value_head = nn.Linear(512, 2)`` with weight AND bias zero-initialized, so
  ``delta`` is EXACTLY zero at initialization and the model is bitwise equal
  to Arm A at initialization (proven at launch and in tests, torch.equal plus
  the lane's signed-zero-normalized tensor SHA equality).
- ``temporal_basis = nn.Parameter [window_size=50, K=8]``, standard-normal
  init; ``queries = nn.Parameter [K=8, 512]``, normal init with std =
  ``model_dim ** -0.5`` (the same scaled-normal convention as
  ``CalibrationFixedSlotRouter.slot_queries``).  Both are dead at
  initialization because ``value_head`` is zero.
- ``delta`` is computed as ``[B, W, C]`` and added AFTER the existing permute
  (layout-matched to the base output ``[B, W, C]``).

Caveat recorded verbatim per the handoff: "temporal_basis mixes the whole
window, so the residual is non-causal, matching the current decoder".

Gradient flow at initialization (documented, standard zero-init residual
topology, cf. LoRA's zero B): ``value_head`` receives nonzero gradient and
wakes at optimizer step 1; ``queries``, ``temporal_basis`` and the
cross-attention receive EXACTLY zero gradients while ``value_head`` is zero
(nothing is permanently dead - from step 2 onward every parameter is live).
Adam makes no spurious move on the zero-grad parameters (weight_decay = 0).
The new head consumes no RNG (cross_attention dropout = 0.0), so the decoder's
internal dropout stream is identical to Arm A's.

Canonical-initial-state discipline (W ADDS parameters, so it differs from the
T/C/G strict=True template): load with ``strict=False``, then assert that the
missing keys are EXACTLY the W-head keys (sorted equality) and the unexpected
keys are empty, then verify the shared-key state is byte-exact by recomputing
``state_sha256`` semantics over the canonical-named subset and comparing
against the artifact's recorded ``state_sha256``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---- frozen cell contract ----------------------------------------------------
NUM_HEADS = 2  # the existing decoder transformer head count, unchanged
MODEL_DIM = 512
WINDOW_SIZE = 50
NUM_COVARIATES = 2
K_LATENT = 8  # FROZEN: no sweep (directions 2026-08-17 §5 Priority 2)
CELL_NAME = "cellW_temporal_residual"
W_HEAD_PREFIX = "temporal_head"

# The exact state keys the W head adds to the canonical Arm A graph, sorted.
W_HEAD_STATE_KEYS = (
    f"{W_HEAD_PREFIX}.cross_attention.in_proj_bias",
    f"{W_HEAD_PREFIX}.cross_attention.in_proj_weight",
    f"{W_HEAD_PREFIX}.cross_attention.out_proj.bias",
    f"{W_HEAD_PREFIX}.cross_attention.out_proj.weight",
    f"{W_HEAD_PREFIX}.queries",
    f"{W_HEAD_PREFIX}.temporal_basis",
    f"{W_HEAD_PREFIX}.value_head.bias",
    f"{W_HEAD_PREFIX}.value_head.weight",
)
W_HEAD_PARAMETER_COUNT = 1_056_146  # 4096 + 400 + 1,050,624 + 1,026

NON_CAUSAL_CAVEAT = (
    "temporal_basis mixes the whole window, so the residual is non-causal, "
    "matching the current decoder"
)

GRADIENT_FLOW_NOTE = (
    "value_head (zero weight AND bias) receives nonzero gradient and wakes at "
    "optimizer step 1; queries, temporal_basis and the cross-attention receive "
    "EXACTLY zero gradients at initialization (zero-init residual topology, as "
    "in LoRA's zero B) and are live from step 2 onward; Adam performs no "
    "update on the zero-grad parameters (weight_decay = 0)"
)


# ---------------------------------------------------------------------------
def _ensure_component_paths():
    """Same package-merge discipline as src/tfpd_lane/sparsification.py."""
    for extra in (
        REPO_ROOT / "streaming_calibration_exp",
        REPO_ROOT / "sua_exploration",
    ):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    try:
        import src as _src_pkg

        streaming_src = str(REPO_ROOT / "streaming_calibration_exp" / "src")
        if streaming_src not in _src_pkg.__path__:
            _src_pkg.__path__.append(streaming_src)
    except ImportError:
        pass


def _arm_common():
    """THE lane arm_common module - runner registration or package import."""
    module = sys.modules.get("tfpd_lane_arm_common")
    if module is not None and hasattr(module, "state_sha256"):
        return module
    from src.tfpd_lane import arm_common as module

    return module


def state_sha256_over_keys(state: dict, keys) -> str:
    """arm_common.state_sha256 semantics over a key-restricted subset.

    Identical construction (sorted keys; per key: name bytes, then dtype +
    shape + signed-zero-normalized tensor bytes, with the
    ``|uninitialized-lazy|`` marker for unmaterialized LazyLinear entries), so
    a digest over exactly the canonical key set is directly comparable to the
    ``state_sha256`` recorded in the canonical initial-state artifact.
    """
    from torch.nn.parameter import UninitializedParameter

    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        tensor = state[key]
        if isinstance(tensor, UninitializedParameter):
            digest.update(b"|uninitialized-lazy|")
            continue
        digest.update(_arm_common().tensor_sha256(tensor).encode("utf-8"))
    return digest.hexdigest()


def load_canonical_initial_state(model, canonical_state: dict,
                                 expected_state_sha256: str) -> dict:
    """The W canonical-load discipline (strict=False + exact proofs).

    1. ``load_state_dict(canonical_state, strict=False)``;
    2. missing keys must be EXACTLY ``W_HEAD_STATE_KEYS`` (sorted equality);
    3. unexpected keys must be empty;
    4. the shared-key state must be byte-exact: ``state_sha256`` semantics
       recomputed over the canonical-named subset must equal the artifact's
       recorded ``expected_state_sha256``.

    Raises ``RuntimeError`` on any violation; returns the receipt record.
    """
    incompat = model.load_state_dict(canonical_state, strict=False)
    missing = sorted(incompat.missing_keys)
    unexpected = sorted(incompat.unexpected_keys)
    if missing != sorted(W_HEAD_STATE_KEYS):
        raise RuntimeError(
            "canonical load discipline failure: missing keys are not exactly "
            f"the W-head keys: {missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"canonical load discipline failure: unexpected keys {unexpected}"
        )
    canonical_keys = set(canonical_state)
    shared_sha = state_sha256_over_keys(model.state_dict(), canonical_keys)
    if shared_sha != expected_state_sha256:
        raise RuntimeError(
            "canonical load discipline failure: shared-subset state SHA "
            f"{shared_sha} != artifact state SHA {expected_state_sha256}"
        )
    added = sorted(set(model.state_dict()) - canonical_keys)
    return {
        "strict_load": False,
        "missing_keys_exactly_w_head": True,
        "unexpected_keys_empty": True,
        "missing_keys": missing,
        "added_keys": added,
        "shared_subset_state_sha256": shared_sha,
        "shared_subset_matches_artifact_state_sha256": True,
        "compared": (
            "state SHA recomputed over exactly the canonical-named key subset "
            "of the W model's state_dict (arm_common.state_sha256 semantics: "
            "sorted keys, per-tensor dtype + shape + signed-zero-normalized "
            "bytes, |uninitialized-lazy| marker for LazyLinear entries) vs the "
            "artifact's recorded state_sha256"
        ),
    }


# ---------------------------------------------------------------------------
class TemporalLatentResidualHead(nn.Module):
    """The zero-initialized temporal latent residual (the ONE W change).

    ``forward(unit_tokens [B, N, model_dim]) -> delta [B, W, C]`` with
    ``delta[t, c] = sum_k temporal_basis[t, k] * value_head(latent_k)[k, c]``
    and ``value_head`` zero-initialized in weight AND bias, so the returned
    delta is exactly zero at initialization.
    """

    def __init__(self, model_dim: int = MODEL_DIM, window_size: int = WINDOW_SIZE,
                 num_covariates: int = NUM_COVARIATES, k_latent: int = K_LATENT,
                 num_heads: int = NUM_HEADS):
        super().__init__()
        self.model_dim = int(model_dim)
        self.window_size = int(window_size)
        self.num_covariates = int(num_covariates)
        self.k_latent = int(k_latent)
        self.num_heads = int(num_heads)
        # learned query slots [K, model_dim]; scaled-normal init (documented):
        # the CalibrationFixedSlotRouter.slot_queries convention
        self.queries = nn.Parameter(torch.empty(self.k_latent, self.model_dim))
        nn.init.normal_(self.queries, std=self.model_dim ** -0.5)
        # learned temporal basis [window_size, K]; standard-normal init
        # (documented); dead at init because value_head is zero
        self.temporal_basis = nn.Parameter(
            torch.randn(self.window_size, self.k_latent)
        )
        # ONE new cross-attention, PyTorch default init, dropout = 0.0
        self.cross_attention = nn.MultiheadAttention(
            self.model_dim, self.num_heads, batch_first=True
        )
        # zero-initialized in weight AND bias -> delta == 0 exactly at init
        self.value_head = nn.Linear(self.model_dim, self.num_covariates)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, unit_tokens: torch.Tensor) -> torch.Tensor:
        query = self.queries.unsqueeze(0).expand(unit_tokens.size(0), -1, -1)
        latent, _ = self.cross_attention(query, unit_tokens, unit_tokens)
        values = self.value_head(latent)  # [B, K, C]
        return torch.einsum("tk,bkc->btc", self.temporal_basis, values)


class TemporalResidualStreamingSpintModel(nn.Module):
    """Route-owned wrapper adding the temporal latent residual to the decode.

    `from_parent` reuses the parent's modules by reference (same objects, same
    names), so every canonical state key is shared byte-for-byte and the W head
    is the ONLY addition.  `decode_components` replicates the parent decode
    exactly and returns ``(base_output [B, W, C], delta [B, W, C])``;
    `decode_with_identity` returns ``base + delta``.
    """

    def __init__(self, parent, head: TemporalLatentResidualHead | None = None):
        super().__init__()
        # reuse the parent's modules by reference (same objects, same names)
        self.decoder = parent.decoder
        self.id_encoder = parent.id_encoder
        # constructed AFTER the parent inside the builder so the parent's
        # canonical initial bytes are consumed first from the seeded stream
        self.temporal_head = head if head is not None else TemporalLatentResidualHead()

    @classmethod
    def from_parent(cls, parent, head: TemporalLatentResidualHead | None = None):
        return cls(parent, head=head)

    # -- graph parity bookkeeping ------------------------------------------
    @property
    def window_size(self):
        return self.decoder.window_size

    def compute_identity(self, calib_trials, side_features=None, electrode_ids=None):
        return self.id_encoder.forward_batch(
            calib_trials, side_features=side_features, electrode_ids=electrode_ids
        )

    def forward(self, neural, calib_trials=None, identity=None, side_features=None,
                **kwargs):
        if identity is None:
            identity = self.compute_identity(calib_trials, side_features=side_features)
        behavior = self.decode_with_identity(neural, identity)
        return behavior, identity

    # -- the exact decode: parent path verbatim + the residual ---------------
    def decode_components(self, neural, identity, neuron_gate=None):
        """Parent decode replica returning ``(base [B, W, C], delta [B, W, C])``."""
        src = neural.permute(0, 2, 1)
        if neuron_gate is not None:
            src = src * neuron_gate
        src = src + identity
        unit_tokens = self.decoder.fc_in(src)  # [B, N, model_dim]
        rep = self.decoder.fc_in(self.decoder.rep).to(unit_tokens)
        transformer_output, _ = self.decoder.transformer(
            rep.repeat(unit_tokens.size(0), 1, 1), unit_tokens
        )
        output = self.decoder.fc_out(transformer_output)  # [B, C, W]
        base = output.permute(0, 2, 1)  # [B, W, C]
        delta = self.temporal_head(unit_tokens)  # [B, W, C]
        return base, delta

    def delta_only(self, neural, identity) -> torch.Tensor:
        """The residual alone (diagnostics/proofs; still exact at zero-init)."""
        return self.decode_components(neural, identity)[1]

    def decode_with_identity(self, neural, identity, neuron_gate=None,
                             live_gain_features=None, live_gain_state=None):
        """Parent path replica with ONE insertion: the temporal residual add."""
        if live_gain_features is not None or live_gain_state is not None:
            raise ValueError("the temporal-residual family never uses live gain paths")
        base, delta = self.decode_components(neural, identity, neuron_gate=neuron_gate)
        return base + delta


def build_temporal_residual_model(seed: int = 42) -> TemporalResidualStreamingSpintModel:
    """The exact Arm A / D graph (2 heads) plus the route-owned W head.

    The decoder/encoder construction is byte-identical to the sibling route
    builders (same seeded draw order), so the shared subset of a seed-42 build
    reproduces the canonical initial bytes; the head draws come afterwards and
    are seed-determined (documented, not canonical).
    """
    _ensure_component_paths()
    from src.models.components.spint import SpintModel
    from src.models.components.streaming_encoders import build_encoder
    from src.models.components.streaming_spint import StreamingSpintModel

    torch.manual_seed(seed)
    decoder = SpintModel(
        model_dim=MODEL_DIM, num_covariates=NUM_COVARIATES,
        window_size=WINDOW_SIZE, num_heads=NUM_HEADS,
        num_layers=1, num_id_layers=1, use_learnable_id=True,
        learnable_id_type="mlp", learnable_rep=True,
        # Arm A: no dynamic dropout; the built-in decoder train-mode dropout
        # (cross-attention/FFN/residual, tf_drop_rate=0.1) is unchanged
        dynamic_dropout=False,
    )
    id_encoder = build_encoder(
        "B3S", window_size=WINDOW_SIZE, trial_length=100, id_hidden_dim=128,
        hidden_dim=64, side_dim=4,
    )
    inner = StreamingSpintModel(decoder=decoder, id_encoder=id_encoder,
                                decoder_mode="coupled")
    return TemporalResidualStreamingSpintModel.from_parent(inner)


# ---------------------------------------------------------------------------
# launch-time proofs and per-epoch diagnostics (shared by runner and tests)
# ---------------------------------------------------------------------------
def parent_decode_from_src(model, src: torch.Tensor) -> torch.Tensor:
    """The plain parent decode from an already-assembled `src` [B,N,W]."""
    unit_tokens = model.decoder.fc_in(src)
    rep = model.decoder.fc_in(model.decoder.rep).to(unit_tokens)
    out, _ = model.decoder.transformer(
        rep.repeat(unit_tokens.size(0), 1, 1), unit_tokens
    )
    return model.decoder.fc_out(out).permute(0, 2, 1)


def parent_decode(model, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    """The plain parent decode path through the same modules, same order."""
    return parent_decode_from_src(model, neural.permute(0, 2, 1) + identity)


def _raw_bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    # .cpu() first: launch proofs run on the training device, and .numpy()
    # requires host memory (no-op for CPU tensors)
    return (a.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            == b.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


def prove_zero_init_bitwise(model, neural: torch.Tensor, calib: torch.Tensor,
                            side: torch.Tensor) -> dict:
    """W launch/test proof: bitwise equality to the parent path at init.

    "Bitwise" is the lane's §3 convention: torch.equal value equality AND
    equal ``arm_common.tensor_sha256`` (signed-zero-normalized bytes).  The
    raw-byte comparison is also recorded; a difference there can only be a
    signed-zero artifact of adding exact +0.0/-0.0 deltas.
    """
    arm_common = _arm_common()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            mine, _ = model(neural, calib_trials=calib, side_features=side)
            identity = model.compute_identity(calib, side_features=side)
            parent = parent_decode(model, neural, identity)
            delta = model.delta_only(neural, identity)
    finally:
        if was_training:
            model.train()
    return {
        "torch_equal": bool(torch.equal(mine, parent)),
        "tensor_sha256_equal": bool(
            arm_common.tensor_sha256(mine) == arm_common.tensor_sha256(parent)
        ),
        "raw_bytes_equal": _raw_bytes_equal(mine, parent),
        "bitwise_convention": (
            "torch.equal AND equal arm_common.tensor_sha256 (signed-zero-"
            "normalized bytes); raw-byte differences can only be -0.0/+0.0"
        ),
        "delta_exactly_zero": int(torch.count_nonzero(delta).item()) == 0,
        "delta_nonzero_count": int(torch.count_nonzero(delta).item()),
        "delta_abs_mean": float(delta.abs().mean().item()),
        "delta_abs_max": float(delta.abs().max().item()),
        "prediction_shape": list(mine.shape),
    }


def head_parameter_manifest(model) -> dict:
    """Exact added-parameter names/shapes/counts (integrity-block record)."""
    state = model.state_dict()
    entries = [
        {"name": key, "shape": list(state[key].shape),
         "numel": int(state[key].numel())}
        for key in sorted(W_HEAD_STATE_KEYS)
    ]
    total = sum(entry["numel"] for entry in entries)
    live = sum(
        int(p.numel()) for p in model.temporal_head.parameters()
        if p.requires_grad and not isinstance(p, torch.nn.parameter.UninitializedParameter)
    )
    return {
        "names": [entry["name"] for entry in entries],
        "components": entries,
        "total_parameters": total,
        "total_trainable_head_parameters": live,
        "counts_match": total == live == W_HEAD_PARAMETER_COUNT,
    }


def head_probe_diagnostics(model, neural: torch.Tensor, calib: torch.Tensor,
                           side: torch.Tensor) -> dict:
    """Per-epoch probe: delta wake-up magnitudes and W-head parameter norms.

    Eval mode, no gradients, on a fixed probe batch.  ``delta_abs_mean/max``
    show the residual waking up from exact zero; ``w_head_param_norms`` and
    ``value_head_weight_max_abs`` track the head parameters themselves.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base, delta = model.decode_components(
                neural, model.compute_identity(calib, side_features=side)
            )
            state = model.state_dict()
            norms = {
                key: float(state[key].norm().item())
                for key in sorted(W_HEAD_STATE_KEYS)
            }
            value_weight = state[f"{W_HEAD_PREFIX}.value_head.weight"]
            value_bias = state[f"{W_HEAD_PREFIX}.value_head.bias"]
            return {
                "delta_abs_mean": float(delta.abs().mean().item()),
                "delta_abs_max": float(delta.abs().max().item()),
                "delta_exactly_zero": int(torch.count_nonzero(delta).item()) == 0,
                "base_abs_mean": float(base.abs().mean().item()),
                "delta_to_base_scale_ratio": (
                    float(delta.abs().mean().item() / base.abs().mean().item())
                    if float(base.abs().mean().item()) > 0 else 0.0
                ),
                "w_head_param_norms": norms,
                "value_head_weight_max_abs": float(value_weight.abs().max().item()),
                "value_head_bias_max_abs": float(value_bias.abs().max().item()),
            }
    finally:
        if was_training:
            model.train()
