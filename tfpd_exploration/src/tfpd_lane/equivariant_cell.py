"""Route-owned SO(2)-equivariant carrier cells (roadmap family B, sub-M T4 route).

Implements the frozen equivariant-decoder cell spec (canonical-frame /
equivariance family) without touching any shared SPINT / Arm A / D file:

- the equivariance property (the whole point): for every rotation R in SO(2),
  ``f({(x_i, R*beta_i, m_i, b_i)}_i) = R * f({(x_i, beta_i, m_i, b_i)}_i)`` where
  ``beta_i = (a_i, c_i)`` is the RAW carrier pair of unit i and the output is
  the 2-D velocity prediction;
- architecture rule: invariants feed the attention logits; equivariants span
  the value/output stream.  The equivariant stream is realized with complex
  arithmetic in R^2 (a complex number z is the 2-vector (Re z, Im z); complex
  multiplication by a REAL coefficient is a shared-coefficient 2-vector map,
  which is exactly what SO(2)-equivariance permits);
- the #1 design trap: the existing pipeline z-scores (a,c) per component with
  source-fit statistics, and per-component z-scoring does NOT commute with
  rotation.  The equivariant path therefore NEVER consumes the z-scored
  (a,c): the carrier direction enters as the per-unit unit-normalized pair
  ``beta_i/|beta_i|`` (a rotation-covariant complex unit) and the only
  carrier magnitudes consumed are rotation-INVARIANT scalars (the canonical
  z-scores of m and b — rotation acts on a different subspace — and
  ``log1p(|beta_i|/m_scale)`` with m_scale a frozen global scalar constant).

Cells (one factor each, everything else the sealed Cell-D contract verbatim):

- ``B_augmentation``: sealed Cell D graph unchanged (strict canonical load,
  2 heads, dynamic unit dropout); the ONE factor is training-time joint SO(2)
  rotation augmentation — one ``theta ~ U(-pi, pi)`` per batch, shared across
  the batch, rotating the normalized side columns (a,c) and the standardized
  behavior labels jointly; activity/calibration/(m,b) invariant; never applied
  at eval.
- ``C_equivariant``: the Cell-D base verbatim (B3S encoder, coupled decoder
  modules, whole-unit dropout law, strict-27, M30, seed 42) with the ordinary
  fused-token consumer of the carrier REPLACED by the equivariant consumer
  (``EquivariantCarrierConsumer``).  Every canonical tensor is strict-loaded
  bitwise (the ordinary consumer modules stay present, frozen, and inactive);
  the new consumer head is initialized in its own seeded namespace.  The
  fused path receives the canonical Z4 (zeros_like normalized T4), so the
  carrier enters ONLY through the equivariant consumer and the property is
  structural.  Parameter count differs from canonical — a DISCLOSED deviation
  (family-B new-architecture cell); the parity claims are instead
  (a) at R = identity the model is its own baseline (bitwise) and
  (b) the equivariance property holds to float tolerance (1e-5) for random
  rotations.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---- frozen constants / dedicated generator namespaces ---------------------
B_ROTATION_SEED = 42_010          # arm B: one SO(2) angle per training batch
EQUIVARIANT_HEAD_SEED = 42_020    # arm C: fresh namespace for the consumer head
EQUIVARIANCE_PROBE_SEED = 42_030  # launch/per-epoch equivariance probe angles
NUM_HEADS = 2
HEAD_KEY_DIM = 64
IOTA_DIM = 6
MODULATION_EPS = 1e-6             # same constant as mc_maze.unit_side_features
EXPECTED_STEPS_PER_EPOCH = 33_925
EXPECTED_EPOCHS = 48
EQUIVARIANCE_TOLERANCE = 1e-5
CARRIER_AUTHORITY_KIND = "tfpd_equivariant_carrier_authority_v1"

CELLS = {
    "B_augmentation": {
        "cell_name": "cellB_rotation_augmentation",
        "factor": "training-time joint SO(2) rotation augmentation of carrier+labels",
    },
    "C_equivariant": {
        "cell_name": "cellC_equivariant",
        "factor": "equivariant consumer of the carrier replacing the fused-token consumer",
    },
}

SYMMETRY_STATEMENT = (
    "for every R in SO(2): f({(x_i, R*beta_i, m_i, b_i)}_i) = "
    "R * f({(x_i, beta_i, m_i, b_i)}_i), with beta_i=(a_i,c_i) the RAW carrier "
    "pair and the output the 2-D velocity prediction. Invariants feed the "
    "attention logits; equivariants span the value/output stream (the complex "
    "carrier units with real coefficients only)."
)

CARRIER_NORMALIZATION_CHOICE = (
    "per-unit unit-normalization beta_i/|beta_i| (the complex carrier unit; "
    "|beta_i| is rotation-invariant so the normalization commutes with "
    "rotation), plus rotation-invariant scalars only: the canonical "
    "source-normalizer z-scores of m and b (rotation acts on the (a,c) "
    "subspace, never on m/b), and log1p(|beta_i|/m_scale) with m_scale a "
    "frozen global scalar constant (mean raw |beta| over all strict-27 "
    "units). The per-component z-scored (a,c) is NEVER consumed."
)

PREREGISTERED_GATES = {
    "primary": {
        "comparison": "C_minus_A_external_governing",
        "arm_A_reference": (
            "sealed Cell D: results/pop_robust_v1/cellD_2heads_dynamic_dropout "
            "(exists; no rerun needed)"
        ),
        "surface": (
            "external sub-M governing R2 (last bin, per-session "
            "variance-weighted R2, equal-session mean)"
        ),
        "gate": "mean paired delta >= +0.03 AND >= 10/15 sessions positive",
    },
    "also_reported": [
        {"comparison": "C_minus_B", "question": "equivariant symmetry beyond augmentation"},
        {"comparison": "B_minus_A", "question": "is any gain just rotation augmentation"},
    ],
    "granularity": ["both granularities (within-6 / external-15)", "date blocks"],
    "statistics": (
        "paired per-session deltas + 10,000-draw session bootstrap 95% CI + "
        "exact sign pattern (tfpd_lane.matched_scorer.paired_session_stats, seed 42)"
    ),
    "eval_policy": (
        "no augmentation at eval for either arm; arm C eval = plain forward"
    ),
}


# ---------------------------------------------------------------------------
def _ensure_component_paths():
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


# ---- rotation utilities (deterministic; float64 by construction) -----------
def rotation_matrix_2d(theta: float) -> np.ndarray:
    """The SO(2) matrix [[cos, -sin], [sin, cos]] in float64."""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)


def probe_thetas(n: int, seed: int = EQUIVARIANCE_PROBE_SEED) -> list[float]:
    """Deterministic probe angles ~ Uniform(-pi, pi) from a dedicated namespace."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return [float(v) for v in rng.uniform(-math.pi, math.pi, int(n))]


class RotationStream:
    """Arm B augmentation law: one SO(2) angle per training batch.

    One ``theta ~ Uniform(-pi, pi)`` per training forward, drawn from a
    dedicated numpy PCG64 namespace, shared across the whole batch.  The full
    drawn sequence is SHA-recorded exactly like the sparsification p stream.
    """

    def __init__(self, seed: int = B_ROTATION_SEED):
        self.seed = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._values: list[float] = []

    def next(self) -> float:
        value = float(self._rng.uniform(-math.pi, math.pi))
        self._values.append(value)
        return value

    def stats(self) -> dict:
        values = np.asarray(self._values, dtype=np.float64)
        if values.size == 0:
            return {"n": 0}
        return {
            "n": int(values.size),
            "min": float(values.min()),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(values.max()),
            "abs_median": float(np.median(np.abs(values))),
        }

    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self._values, dtype=np.float64).tobytes())
        return digest.hexdigest()


def rotate_side_behavior(
    side: torch.Tensor,
    behavior: torch.Tensor,
    theta: float,
    pad_value: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arm B's joint rotation: normalized T4 (a,c) columns AND behavior labels.

    ``side`` is the aligned NORMALIZED T4 tensor [.., N, 4]; columns 0:2 are
    the z-scored carrier pair (the model-visible fused carrier — an
    augmentation input, never an equivariance claim).  ``behavior`` is the
    standardized label tensor [.., W, 2]; pad rows (all-channel ``pad_value``
    fill) are detected BEFORE rotation and restored unchanged so downstream
    validity masking sees the identical rows.  theta == 0.0 returns bitwise
    clones (the model is its own baseline).
    """
    if theta == 0.0:
        return side.clone(), behavior.clone()
    rot = torch.from_numpy(rotation_matrix_2d(theta)).to(
        device=side.device, dtype=side.dtype
    )
    new_side = side.clone()
    new_side[..., 0:2] = side[..., 0:2] @ rot.t()
    valid = (behavior != pad_value).all(dim=-1)
    new_behavior = behavior.clone()
    rotated_pairs = behavior[..., 0:2] @ rot.t()
    new_behavior[..., 0:2] = torch.where(
        valid.unsqueeze(-1), rotated_pairs, behavior[..., 0:2]
    )
    return new_side, new_behavior


# ---- the rotation-covariant carrier view -----------------------------------
class CarrierBundle:
    """Per-unit SO(2)-covariant view of one session's RAW T4 authority row.

    Equivariant stream: ``unit2[i] = beta_i / |beta_i|`` in R^2 — the complex
    carrier unit ``e^{i phi_i}`` realized as a 2-vector; under a global
    rotation it transforms as ``unit2 -> R @ unit2``.

    Invariant stream (functions of the rotation orbit only):
    - the canonical source-normalizer z-scores of m and b (rotation acts on a
      different subspace, so these never move),
    - ``log1p(|beta_i| / m_scale)`` with m_scale a frozen global scalar,
    - the direction-validity flag (raw m > MODULATION_EPS, the theta-authority
      validity law),
    - ``rho_cos/rho_sin`` — the pairwise angular-difference projections of
      unit i's direction onto the population phase resultant
      ``Psi = mean_j e^{i phi_j}``: functions of ``phi_i - phi_j`` ONLY.
    """

    __slots__ = (
        "raw", "m_norm", "b_norm", "m_scale",
        "beta_norm", "valid", "unit2", "rho_cos", "rho_sin",
    )

    def __init__(self, raw, m_norm, b_norm, m_scale, beta_norm, valid, unit2,
                 rho_cos, rho_sin):
        self.raw = raw
        self.m_norm = m_norm
        self.b_norm = b_norm
        self.m_scale = m_scale
        self.beta_norm = beta_norm
        self.valid = valid
        self.unit2 = unit2
        self.rho_cos = rho_cos
        self.rho_sin = rho_sin

    @classmethod
    def from_raw(cls, raw, m_norm, b_norm, m_scale: float) -> "CarrierBundle":
        raw = torch.as_tensor(raw, dtype=torch.float64)
        if raw.dim() != 2 or raw.shape[1] != 4:
            raise ValueError(f"raw T4 must be [N,4], got {tuple(raw.shape)}")
        a, c = raw[:, 0], raw[:, 1]
        beta = torch.hypot(a, c)
        valid = raw[:, 2] > MODULATION_EPS  # theta-authority validity law
        safe = torch.where(beta > 0.0, beta, torch.ones_like(beta))
        unit2 = torch.stack([a, c], dim=-1) / safe.unsqueeze(-1)
        unit2 = torch.where(valid.unsqueeze(-1), unit2, torch.zeros_like(unit2))
        n_valid = max(int(valid.sum().item()), 1)
        psi = unit2.sum(dim=0) / n_valid
        dot = unit2[:, 0] * psi[0] + unit2[:, 1] * psi[1]
        cross = unit2[:, 0] * psi[1] - unit2[:, 1] * psi[0]
        zeros = torch.zeros_like(beta)
        return cls(
            raw=raw,
            m_norm=torch.as_tensor(m_norm, dtype=torch.float32).clone(),
            b_norm=torch.as_tensor(b_norm, dtype=torch.float32).clone(),
            m_scale=float(m_scale),
            beta_norm=beta,
            valid=valid,
            unit2=unit2,
            rho_cos=torch.where(valid, dot, zeros),
            rho_sin=torch.where(valid, cross, zeros),
        )

    def rotated(self, theta: float) -> "CarrierBundle":
        """The genuinely rotated RAW carrier, with every derived quantity
        recomputed (invariants must come out equal, the unit vector rotated)."""
        rot = rotation_matrix_2d(theta)
        raw = self.raw.clone()
        a, c = self.raw[:, 0], self.raw[:, 1]
        raw[:, 0] = a * rot[0, 0] + c * rot[0, 1]
        raw[:, 1] = a * rot[1, 0] + c * rot[1, 1]
        return CarrierBundle.from_raw(raw, self.m_norm, self.b_norm, self.m_scale)

    def unit_two_vector(self) -> torch.Tensor:
        """The equivariant value stream: float64 [N,2] complex carrier units."""
        return self.unit2

    def invariant_features(self, dtype=torch.float32, device=None) -> torch.Tensor:
        """Per-unit invariant carrier features [N, IOTA_DIM] (float64 core)."""
        features = torch.stack(
            [
                self.m_norm.to(torch.float64),
                self.b_norm.to(torch.float64),
                torch.log1p(self.beta_norm / self.m_scale),
                self.valid.to(torch.float64),
                self.rho_cos,
                self.rho_sin,
            ],
            dim=-1,
        )
        if device is not None:
            features = features.to(device)
        return features.to(dtype)


# ---- the equivariant consumer (arm C's ONE factor) -------------------------
class EquivariantCarrierConsumer(torch.nn.Module):
    """SO(2)-equivariant consumer of the T4 carrier.

    Invariant pathway (logits + gains, all real): the carrier-blind activity
    tokens (B3S Z4-identity + neural windows, whole-unit dropout, fc_in) and
    the per-unit carrier invariants ``iota``.  Equivariant pathway (values):
    the complex carrier units with REAL coefficients only — softmax attention
    weights, a real per-unit/per-bin gain, and a real head mixing; the output
    is the float64 complex sum cast once to float32.

    Every learnable tensor is real and is applied with shared coefficients to
    the two carrier components, so rotating every carrier unit by R rotates
    the output by exactly R and nothing else moves.
    """

    def __init__(
        self,
        *,
        window_size: int = 50,
        model_dim: int = 512,
        num_heads: int = NUM_HEADS,
        head_key_dim: int = HEAD_KEY_DIM,
        num_queries: int = 2,
        iota_dim: int = IOTA_DIM,
    ):
        super().__init__()
        if head_key_dim <= 0 or model_dim <= 0 or window_size <= 0:
            raise ValueError("consumer dims must be positive")
        self.window_size = int(window_size)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_key_dim = int(head_key_dim)
        self.num_queries = int(num_queries)
        self.iota_dim = int(iota_dim)
        self.norm_q = torch.nn.LayerNorm(self.model_dim)
        self.norm_k = torch.nn.LayerNorm(self.model_dim)
        self.q_proj = torch.nn.Linear(
            self.model_dim, self.num_heads * self.head_key_dim, bias=False
        )
        self.k_proj = torch.nn.Linear(
            self.model_dim + self.iota_dim,
            self.num_heads * self.head_key_dim,
            bias=False,
        )
        self.gain = torch.nn.Linear(
            self.model_dim + self.iota_dim, self.num_queries * self.window_size
        )
        self.head_mix = torch.nn.Parameter(
            torch.full((self.num_heads, self.num_queries), 1.0 / self.num_heads)
        )
        self.out_scale = torch.nn.Parameter(torch.ones((), dtype=torch.float32))

    def _logits(self, tokens: torch.Tensor, queries: torch.Tensor,
                carrier: CarrierBundle) -> torch.Tensor:
        batch, n_units, _ = tokens.shape
        n_queries = queries.shape[1]
        iota = carrier.invariant_features(dtype=tokens.dtype, device=tokens.device)
        if iota.shape != (n_units, self.iota_dim):
            raise ValueError(
                f"carrier invariant features {tuple(iota.shape)} do not match "
                f"[{n_units}, {self.iota_dim}]"
            )
        iota_b = iota.unsqueeze(0).expand(batch, -1, -1)
        q = self.q_proj(self.norm_q(queries)).view(
            batch, n_queries, self.num_heads, self.head_key_dim
        )
        k = self.k_proj(torch.cat([self.norm_k(tokens), iota_b], dim=-1)).view(
            batch, n_units, self.num_heads, self.head_key_dim
        )
        return torch.einsum("bchd,bnhd->bchn", q, k) / math.sqrt(self.head_key_dim)

    def attention_weights(self, tokens: torch.Tensor, queries: torch.Tensor,
                          carrier: CarrierBundle) -> torch.Tensor:
        """The INVARIANT softmax attention weights [B, C, H, N] (eval use)."""
        with torch.no_grad():
            return torch.softmax(
                self._logits(tokens.float(), queries.float(), carrier), dim=-1
            )

    def forward(self, tokens: torch.Tensor, queries: torch.Tensor,
                carrier: CarrierBundle) -> torch.Tensor:
        tokens = tokens.float()
        queries = queries.float()
        if tokens.dim() != 3 or queries.dim() != 3:
            raise ValueError("tokens [B,N,d] and queries [B,C,d] required")
        if tokens.shape[-1] != self.model_dim or queries.shape[-1] != self.model_dim:
            raise ValueError("token/query feature width does not match model_dim")
        if queries.shape[1] != self.num_queries:
            raise ValueError(
                f"expected {self.num_queries} query tokens, got {queries.shape[1]}"
            )
        batch, n_units, _ = tokens.shape
        weights = torch.softmax(
            self._logits(tokens, queries, carrier), dim=-1
        )  # [B, C, H, N] — REAL and rotation-invariant
        mixed = torch.einsum(
            "bchn,hc->bcn", weights, self.head_mix.to(weights.dtype)
        )  # [B, C, N] real
        iota = carrier.invariant_features(dtype=tokens.dtype, device=tokens.device)
        iota_b = iota.unsqueeze(0).expand(batch, -1, -1)
        gain = self.gain(torch.cat([tokens, iota_b], dim=-1)).view(
            batch, n_units, self.num_queries, self.window_size
        )  # [B, N, C, W] real, invariant
        unit2 = carrier.unit_two_vector().to(device=tokens.device, dtype=torch.float64)
        # complex sum, accumulated in float64: z = sum_{c,i} mixed * gain * u_i
        z = torch.einsum(
            "bcn,bncw,nu->bwu",
            mixed.to(torch.float64),
            gain.to(torch.float64),
            unit2,
        )
        out = z * self.out_scale.to(torch.float64)
        return out.to(tokens.dtype)  # [B, W, 2] — channel 0 = Re, channel 1 = Im


class EquivariantStreamingSpintModel(torch.nn.Module):
    """Route-owned wrapper: the Cell-D base with the consumer swapped.

    ``from_parent`` reuses the parent StreamingSpintModel's decoder and B3S
    encoder by reference, so every canonical state key strict-loads.  The
    fused side path is structurally carrier-blind: ``forward`` feeds the B3S
    encoder ``zeros_like(side)`` (the canonical Z4) and never the normalized
    T4 itself, so the ONLY carrier-dependent computation in the whole graph is
    the equivariant consumer.  The whole-unit dynamic dropout law is the
    parent implementation verbatim (one ``random.uniform`` p and one PyTorch
    unit-mask dropout per training forward, eval never drops).
    """

    def __init__(self, parent):
        super().__init__()
        self.decoder = parent.decoder
        self.id_encoder = parent.id_encoder
        self._decoder_frozen = False
        self.consumer: EquivariantCarrierConsumer | None = None
        self.carrier: CarrierBundle | None = None

    @classmethod
    def from_parent(cls, parent) -> "EquivariantStreamingSpintModel":
        return cls(parent)

    @property
    def window_size(self):
        return self.decoder.window_size

    def attach_consumer(self, consumer: EquivariantCarrierConsumer):
        self.consumer = consumer
        return self

    def set_carrier(self, bundle: CarrierBundle):
        self.carrier = bundle

    def compute_identity(self, calib_trials, side_features=None, electrode_ids=None):
        return self.id_encoder.forward_batch(
            calib_trials, side_features=side_features, electrode_ids=electrode_ids
        )

    def forward(self, neural, calib_trials=None, identity=None, side_features=None,
                carrier=None, **kwargs):
        if carrier is None:
            carrier = self.carrier
        if carrier is None:
            raise ValueError("an SO(2)-covariant carrier bundle is required")
        if self.consumer is None:
            raise ValueError("the equivariant consumer head is not attached")
        if identity is None:
            if side_features is None:
                raise ValueError(
                    "side_features required (the fused path consumes its "
                    "zeros_like — the canonical Z4 — never the T4 itself)"
                )
            identity = self.compute_identity(
                calib_trials, side_features=torch.zeros_like(side_features)
            )
        behavior = self.decode_with_identity(neural, identity, carrier)
        return behavior, identity

    def decode_with_identity(self, neural, identity, carrier):
        """Parent decode path replica up to fc_in, then the equivariant consumer."""
        src = neural.permute(0, 2, 1)
        src = src + identity
        # ---- verbatim whole-unit dropout law (StreamingSpintModel path) ----
        batch_size = src.size(0)
        num_neurons = src.size(1)
        dropout_mask = torch.ones(
            batch_size, num_neurons, device=src.device, dtype=src.dtype
        )
        if not self._decoder_frozen:
            if self.decoder.dynamic_dropout and self.training:
                import random

                p = random.uniform(
                    self.decoder.dynamic_dropout_low,
                    self.decoder.dynamic_dropout_high,
                )
                dropout_mask = torch.nn.functional.dropout(
                    dropout_mask, p=p, training=True
                )
            elif self.decoder.dropout_rate > 0.0 and self.training:
                dropout_mask = torch.nn.functional.dropout(
                    dropout_mask, p=self.decoder.dropout_rate, training=True
                )
        src = src * dropout_mask.unsqueeze(-1)
        tokens = self.decoder.fc_in(src)  # [B, N, d] carrier-blind
        queries = self.decoder.fc_in(self.decoder.rep).to(tokens)
        queries = queries.repeat(tokens.size(0), 1, 1)  # [B, C, d] invariant
        return self.consumer(tokens, queries, carrier)


def build_equivariant_model(seed: int = 42) -> EquivariantStreamingSpintModel:
    """The Cell-D graph verbatim + the equivariant consumer head.

    The decoder/encoder construction consumes the identical RNG stream as the
    Cell-D builder (decoder, then B3S encoder — head-count/dropout flags carry
    no parameters), so every canonical tensor strict-loads bitwise.  The ONE
    new factor (the consumer) is then built under its own dedicated seeded
    namespace (a disclosed extra-seed deviation from the Cell-D contract).
    """
    _ensure_component_paths()
    from src.models.components.spint import SpintModel
    from src.models.components.streaming_encoders import build_encoder
    from src.models.components.streaming_spint import StreamingSpintModel

    torch.manual_seed(seed)
    decoder = SpintModel(
        model_dim=512, num_covariates=2, window_size=50, num_heads=NUM_HEADS,
        num_layers=1, num_id_layers=1, use_learnable_id=True,
        learnable_id_type="mlp", learnable_rep=True,
        dynamic_dropout=True, dynamic_dropout_low=0.0, dynamic_dropout_high=1.0,
    )
    id_encoder = build_encoder(
        "B3S", window_size=50, trial_length=100, id_hidden_dim=128,
        hidden_dim=64, side_dim=4,
    )
    inner = StreamingSpintModel(decoder=decoder, id_encoder=id_encoder,
                                decoder_mode="coupled")
    model = EquivariantStreamingSpintModel.from_parent(inner)
    torch.manual_seed(EQUIVARIANT_HEAD_SEED)
    model.attach_consumer(
        EquivariantCarrierConsumer(
            window_size=50, model_dim=512, num_heads=NUM_HEADS,
            head_key_dim=HEAD_KEY_DIM, num_queries=2, iota_dim=IOTA_DIM,
        )
    )
    return model


INACTIVE_CANONICAL_MODULES = (
    "decoder.transformer", "decoder.fc_out", "decoder.fc_id_in", "decoder.fc_id_out",
)


def _tensor_sha256(tensor) -> str:
    digest = hashlib.sha256()
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.is_floating_point():
        flat = flat + 0  # IEEE: -0.0 + 0.0 == +0.0; identity otherwise
    digest.update(str(flat.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    if flat.numel():
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def subset_state_sha256(state_dict: dict, keys) -> str:
    """arm_common.state_sha256's law restricted to ``keys`` (sorted)."""
    digest = hashlib.sha256()
    from torch.nn.parameter import UninitializedParameter

    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        tensor = state_dict[key]
        if isinstance(tensor, UninitializedParameter):
            digest.update(b"|uninitialized-lazy|")
            continue
        digest.update(_tensor_sha256(tensor).encode("utf-8"))
    return digest.hexdigest()


def load_canonical_prefix(model, canonical_state: dict) -> dict:
    """Strict-load every canonical tensor into the equivariant graph, bitwise.

    The canonical keys must all exist with identical shapes (lazy entries stay
    lazy).  After the load, the canonical-subset SHA must equal the canonical
    payload's ``state_sha256`` — the bitwise-prefix proof.
    """
    from torch.nn.parameter import UninitializedParameter

    own = model.state_dict()
    missing = [k for k in canonical_state if k not in own]
    if missing:
        raise SystemExit(f"canonical keys missing from the equivariant graph: {missing[:5]}")
    for key, value in canonical_state.items():
        target = own[key]
        if isinstance(value, UninitializedParameter) or isinstance(target, UninitializedParameter):
            if type(value) is not type(target):
                raise SystemExit(f"lazy mismatch at {key}")
            continue
        if tuple(target.shape) != tuple(value.shape):
            raise SystemExit(
                f"canonical shape mismatch at {key}: "
                f"{tuple(target.shape)} vs {tuple(value.shape)}"
            )
        with torch.no_grad():
            target.copy_(value)
    own = model.state_dict()
    for key, value in canonical_state.items():
        if isinstance(value, UninitializedParameter):
            continue
        if not torch.equal(own[key], value):
            raise SystemExit(f"canonical prefix not bitwise at {key}")
    subset_sha = subset_state_sha256(own, canonical_state.keys())
    return {
        "canonical_keys": len(canonical_state),
        "strict_load": True,
        "bitwise_equal_all_keys": True,
        "canonical_subset_sha256": subset_sha,
    }


def freeze_inactive_consumer_modules(model) -> dict:
    """Freeze the ordinary fused-token consumer modules (present, canonical,
    never used by the equivariant decode path; disclosed)."""
    from torch.nn.parameter import UninitializedParameter

    frozen = 0
    for name in INACTIVE_CANONICAL_MODULES:
        module = model
        for attr in name.split("."):
            module = getattr(module, attr)
        for param in module.parameters():
            if isinstance(param, UninitializedParameter):
                continue  # lazy entries never materialize on this path
            if param.requires_grad:
                param.requires_grad_(False)
                frozen += param.numel()
    return {
        "modules": list(INACTIVE_CANONICAL_MODULES),
        "frozen_parameter_count": int(frozen),
        "note": (
            "the ordinary fused-token consumer stays in the graph (canonical "
            "bytes, strict-loaded) but is disconnected from the equivariant "
            "decode path; it receives no gradient and is never invoked"
        ),
    }


def parameter_accounting(model) -> dict:
    from torch.nn.parameter import UninitializedParameter

    def _count(params) -> int:
        return sum(
            p.numel() for p in params if not isinstance(p, UninitializedParameter)
        )

    all_params = list(model.parameters())
    consumer = list(model.consumer.parameters())
    frozen = [p for p in all_params if not p.requires_grad]
    return {
        "canonical_tensors_strict_loaded": _count(all_params) - _count(consumer),
        "new_consumer_parameters": _count(consumer),
        "frozen_canonical_parameters": _count(frozen),
        "total_trainable": _count(p for p in all_params if p.requires_grad),
        "total_parameters": _count(all_params),
    }


# ---- the raw carrier authority (theta-authority pattern, verbatim rule) ----
def build_carrier_authority(train_nwbs) -> dict:
    """Source-only RAW T4 authority aligned to canonical unit order.

    Same authority the sealed theta authority used:
    ``compute_unit_side_features_uncached(feature_group='t4', pool_size=30,
    bin 20 ms, window 50, 'R', 'sua')`` -> raw ``[m cos phi, m sin phi, m, b]``.
    Never z-scored; per-session ``raw_t4_sha256`` (float32 bytes, the theta
    authority's hash law) binds this artifact to the sealed theta authority.
    """
    from mc_maze.multisession_datamodule import session_name_from_path
    from mc_maze.unit_side_features import (
        MODULATION_EPS as _EPS,
        compute_unit_side_features_uncached,
    )

    authority = {}
    for nwb_path in train_nwbs:
        raw, _meta = compute_unit_side_features_uncached(
            Path(nwb_path),
            feature_group="t4",
            pool_size=30,
            bin_size_ms=20,
            window_size=50,
            trial_result_filter="R",
            signal_view="sua",
        )
        raw = np.asarray(raw, dtype=np.float64)
        beta = np.hypot(raw[:, 0], raw[:, 1])
        valid = raw[:, 2] > _EPS
        authority[session_name_from_path(nwb_path)] = {
            "raw": raw,
            "valid": valid.astype(bool),
            "n_units": int(raw.shape[0]),
            "raw_t4_sha256": hashlib.sha256(
                raw.astype(np.float32).tobytes()
            ).hexdigest(),
            "beta_norm": beta,
        }
    return authority


def authority_sha256(authority: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(authority):
        entry = authority[name]
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(entry["raw"], dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(entry["valid"], dtype=bool).tobytes())
        digest.update(str(entry["n_units"]).encode("utf-8"))
    return digest.hexdigest()


def global_m_scale(authority: dict) -> float:
    """The frozen rotation-invariant global scalar: mean raw |beta| over all
    units of every strict-27 session (a function of the orbit only)."""
    values = np.concatenate([entry["beta_norm"].reshape(-1) for entry in authority.values()])
    scale = float(values.mean())
    if not math.isfinite(scale) or scale <= 0.0:
        raise SystemExit("degenerate global m_scale (nonpositive or nonfinite)")
    return scale


def build_carrier_bundles(authority: dict, session_records, m_scale: float) -> dict:
    """Per-session carrier bundles from the authority + normalized side rows.

    ``m_norm``/``b_norm`` are the canonical source-normalizer z-scores of the
    T4 m/b columns taken from the datamodule's aligned normalized side tensor
    (rotation never touches them); the direction comes from the RAW authority.
    """
    bundles = {}
    for name, entry in sorted(authority.items()):
        record = session_records[name]
        side = np.asarray(record.side_features)
        if side.shape[0] != entry["n_units"] or side.shape[1] != 4:
            raise SystemExit(
                f"{name}: side features {side.shape} not aligned with authority "
                f"n_units={entry['n_units']}"
            )
        bundles[name] = CarrierBundle.from_raw(
            torch.from_numpy(np.ascontiguousarray(entry["raw"])),
            m_norm=torch.from_numpy(np.ascontiguousarray(side[:, 2], dtype=np.float32)),
            b_norm=torch.from_numpy(np.ascontiguousarray(side[:, 3], dtype=np.float32)),
            m_scale=m_scale,
        )
    return bundles


# ---- equivariance probe -----------------------------------------------------
def equivariance_probe(model, neural, calib_trials, side_features,
                       thetas) -> dict:
    """Max |f(x, R beta) - R f(x, beta)| over probe rotations, eval mode.

    The rotated inputs are genuinely rotated RAW carriers with every derived
    quantity recomputed — the honest form of the property, not a cached one.
    theta == 0.0 must reproduce the baseline bitwise (self-baseline claim).
    """
    if model.carrier is None:
        raise ValueError("set the session carrier bundle before probing")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base, _ = model(neural, calib_trials=calib_trials, side_features=side_features)
            violations = []
            identity_bitwise = None
            for theta in thetas:
                pred, _ = model(
                    neural, calib_trials=calib_trials, side_features=side_features,
                    carrier=model.carrier.rotated(float(theta)),
                )
                if float(theta) == 0.0:
                    identity_bitwise = bool(torch.equal(pred, base))
                rot = torch.from_numpy(rotation_matrix_2d(float(theta)))
                expected = base.to(torch.float64) @ rot.t().to(base.device)
                violations.append(
                    float((pred.to(torch.float64) - expected).abs().max().item())
                )
        return {
            "n_rotations": len(thetas),
            "max_violation": (max(violations) if violations else None),
            "per_rotation_violations": violations,
            "identity_rotation_bitwise_equal": identity_bitwise,
        }
    finally:
        if was_training:
            model.train()


# ---- budget guard + receipt integrity blocks --------------------------------
def assert_training_budget(steps_per_epoch: int, epochs: int, smoke: bool,
                           expected_steps: int = EXPECTED_STEPS_PER_EPOCH,
                           expected_epochs: int = EXPECTED_EPOCHS) -> dict:
    """Fail-closed budget guard: the sealed 33,925 x 48 recipe (smoke exempt).

    The final-four SWA window always needs at least four epochs.
    """
    if epochs < 4:
        raise SystemExit("final-four SWA requires at least 4 epochs")
    if smoke:
        return {
            "smoke": True,
            "epochs": int(epochs),
            "steps_per_epoch": int(steps_per_epoch),
            "note": "smoke budget exempt from the sealed 33,925 x 48 guard (non-authoritative)",
        }
    if steps_per_epoch != expected_steps or epochs != expected_epochs:
        raise SystemExit(
            f"budget drift: expected {expected_steps} steps/epoch over "
            f"{expected_epochs} epochs, got {steps_per_epoch} x {epochs}"
        )
    return {
        "smoke": False,
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "total_optimizer_steps": int(epochs) * int(steps_per_epoch),
    }


def arm_b_integrity_block(*, num_heads: int, total_parameters: int,
                          canonical_parameters: int) -> dict:
    return {
        "cell": "B_augmentation",
        "family": "B_canonical_frame_equivariance",
        "cell_name": CELLS["B_augmentation"]["cell_name"],
        "num_heads": int(num_heads),
        "architecture": (
            "sealed Cell D verbatim (B3S encoder, coupled decoder, 2 heads, "
            "dynamic whole-unit dropout law unchanged); NO architecture change"
        ),
        "one_factor": CELLS["B_augmentation"]["factor"],
        "augmentation_law": {
            "draw": (
                f"one theta ~ Uniform(-pi, pi) per training batch, numpy "
                f"PCG64({B_ROTATION_SEED}) dedicated namespace, full sequence "
                f"SHA-recorded"
            ),
            "shared_across_batch": True,
            "rotated": [
                "normalized T4 side columns 0:2 (the z-scored carrier pair (a,c))",
                "standardized behavior label channels 0:2",
            ],
            "invariant": [
                "neural activity windows", "calibration trials",
                "normalized side columns 2:4 (m,b)",
            ],
            "padding_rule": (
                "pad rows (all-channel -1 fill) are detected before rotation "
                "and restored unchanged, so validity masking is unchanged"
            ),
            "identity_case": "theta == 0.0 returns bitwise clones (own baseline)",
            "eval": "never applied at eval/scoring; eval = plain forward",
        },
        "parameter_count": {
            "total": int(total_parameters),
            "canonical_cell_d": int(canonical_parameters),
            "disclosed_deviation": False,
        },
        "symmetry_claim": (
            "none at inference; this arm isolates 'is any gain just augmentation'"
        ),
        "gates": PREREGISTERED_GATES,
    }


def arm_c_integrity_block(*, num_heads: int, canonical_parameters: int,
                          consumer_parameters: int, total_trainable: int,
                          m_scale: float, launch_max_violation: float,
                          normalization_commutes: bool) -> dict:
    return {
        "cell": "C_equivariant",
        "family": "B_canonical_frame_equivariance",
        "cell_name": CELLS["C_equivariant"]["cell_name"],
        "num_heads": {
            "consumer_attention_heads": int(num_heads),
            "decoder_num_heads": int(num_heads),
            "note": "both are 2; the decoder head partition is inherited from Cell D",
        },
        "symmetry_statement": SYMMETRY_STATEMENT,
        "architecture_rule": (
            "invariants feed the attention logits; equivariants span the "
            "value/output stream"
        ),
        "one_factor": CELLS["C_equivariant"]["factor"],
        "carrier_normalization": {
            "choice": CARRIER_NORMALIZATION_CHOICE,
            "z_scored_ac_used": False,
            "per_component_z_scoring_commutes_with_rotation": False,
            "normalization_commutes_with_rotation": bool(normalization_commutes),
            "m_scale_frozen_global_scalar": float(m_scale),
            "raw_authority": (
                "compute_unit_side_features_uncached(feature_group='t4', "
                "pool_size=30, bin 20ms, window 50, 'R', 'sua') — the same "
                "authority the sealed theta authority used; per-session "
                "raw_t4_sha256 must equal the theta authority's"
            ),
        },
        "fused_path": (
            "the canonical Z4 (zeros_like normalized T4): the carrier never "
            "enters the fused token path, so the equivariance is structural"
        ),
        "whole_unit_dropout_law": (
            "the parent implementation verbatim: one random.uniform p in "
            "[0,1) and one PyTorch unit-mask dropout per training forward; "
            "never active at eval (the law is carrier-independent)"
        ),
        "parameter_disclosure": {
            "canonical_tensors_strict_loaded_bitwise": int(canonical_parameters),
            "canonical_tensors_inactive_requires_grad_false": list(
                INACTIVE_CANONICAL_MODULES
            ),
            "new_consumer_parameters": int(consumer_parameters),
            "total_trainable": int(total_trainable),
            "bitwise_parity_at_init_vs_cell_d_claimed": False,
            "note": (
                "DISCLOSED deviation (family-B new-architecture cell): "
                "parameter count differs from canonical Cell D; the parity "
                "claims are (a) identity rotation is the model's own baseline "
                "and (b) the equivariance property to 1e-5 under random rotations"
            ),
        },
        "equivariance": {
            "tolerance": EQUIVARIANCE_TOLERANCE,
            "launch_max_violation": float(launch_max_violation),
            "identity_rotation_own_baseline": True,
        },
        "augmentation": "none — the symmetry is built in",
        "eval_policy": "plain forward; augmentation never used",
        "gates": PREREGISTERED_GATES,
    }
