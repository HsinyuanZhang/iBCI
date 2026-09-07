"""Route-owned R / S2 population-sparsification cells (workorder 2026-08-18).

Implements WORKORDER_BEAT_A2_SPARSIFICATION_20260818.md §4-§6 without touching
any shared SPINT / Arm A / D / DH file:

- `PStream`: the exact shared p stream — one `p ~ Uniform(0,1)` per training
  forward from a dedicated numpy PCG64(42) generator.  R and S2 construct
  identical streams (same seed, same call order), so their
  `p_sequence_sha256` values are equal; relative to sealed D (which sampled p
  on the global RNG without recording draws) the cells are DISTRIBUTION-
  matched, not bitwise random-path-matched.
- `SparsifiedStreamingSpintModel`: a route-owned subclass whose
  `decode_with_identity` applies the mask at the exact D site — after
  `src = activity + identity`, before `fc_in` — with gain `1/(1-p)`, no
  clamp, no floor; `p == 1` produces the all-zero tensor without division.
  With sparsification disabled (evaluation), the mask is exact ones and the
  forward is bitwise equal to the parent path.
- Cell R: elementwise Bernoulli keep-mask over `B x N x W` (one shared p).
- Cell S2: carrier-ordered sector dropout on the 2-head D graph — per batch
  example, `k_drop ~ Binomial(N, p)`, `k_valid ~ Hypergeometric(N, |V|,
  k_drop)`, an independent sector centre `phi ~ Uniform(-pi, pi)`, drop the
  `k_valid` direction-valid units closest to `phi` (ties by canonical unit
  index), drop the remaining undefined units uniformly without replacement.
  `theta = atan2(raw_c, raw_a)` from the RAW T4 authority only (never
  z-scored values); theta is a masking authority, never model-visible.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
P_STREAM_SEED = 42
SECTOR_SEED = 42_002
ELEMENTWISE_MASK_SEED = 42_001
CELLS = {
    "R": {"mask_structure": "elementwise_BxNxW", "num_heads": 2},
    "S2": {"mask_structure": "carrier_sector_whole_unit", "num_heads": 2},
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


class PStream:
    """Exact shared p stream: one draw per training forward."""

    def __init__(self, seed: int = P_STREAM_SEED):
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._values: list[float] = []

    def next(self) -> float:
        value = float(self._rng.random())
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
        }

    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self._values, dtype=np.float64).tobytes())
        return digest.hexdigest()


class SectorMasker:
    """Frozen §6.2 construction; one call per batch example.

    Accumulates only lightweight numeric records (k_drop / n_dropped /
    sector width per example) so a 1.6M-step run stays memory-flat; `reset()`
    is called by the runner each epoch and `summary()` aggregates.
    """

    def __init__(self, seed: int = SECTOR_SEED):
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self.k_drop: list[int] = []
        self.n_dropped: list[int] = []
        self.width: list[float] = []
        self.count = 0

    def draw(self, theta: np.ndarray, valid: np.ndarray, p: float) -> np.ndarray:
        """Return a keep-mask [N] (float32, 1 keep / 0 drop) for one example."""
        n = theta.shape[0]
        n_valid = int(valid.sum())
        k_drop = int(self._rng.binomial(n, p))
        k_valid = int(self._rng.hypergeometric(n_valid, n - n_valid, k_drop)) if k_drop > 0 else 0
        phi = float(self._rng.uniform(-math.pi, math.pi))
        drop = np.zeros(n, dtype=bool)
        valid_idx = np.flatnonzero(valid)
        undefined_idx = np.flatnonzero(~valid)
        if k_valid > 0 and valid_idx.size:
            distances = np.abs(np.angle(np.exp(1j * (theta[valid_idx] - phi))))
            order = np.lexsort((valid_idx, distances))  # ties by canonical index
            drop[valid_idx[order[:k_valid]]] = True
        remaining = k_drop - int(drop.sum())
        if remaining > 0 and undefined_idx.size:
            pick = self._rng.choice(undefined_idx, size=min(remaining, undefined_idx.size),
                                    replace=False)
            drop[pick] = True
        mask = (~drop).astype(np.float32)
        dropped_theta = theta[drop & valid]
        width = None
        if dropped_theta.size >= 2:
            sorted_theta = np.sort(dropped_theta)
            gaps = np.diff(sorted_theta)
            wrap_gap = (sorted_theta[0] + 2 * math.pi) - sorted_theta[-1]
            width = float(min(gaps.max(), wrap_gap))
        elif dropped_theta.size == 1:
            width = 0.0
        self.k_drop.append(k_drop)
        self.n_dropped.append(int(drop.sum()))
        self.width.append(width if width is not None else float("nan"))
        self.count += 1
        return mask

    def reset(self) -> None:
        self.k_drop.clear()
        self.n_dropped.clear()
        self.width.clear()
        self.count = 0

    def summary(self) -> dict:
        widths = np.asarray([w for w in self.width if np.isfinite(w)], dtype=np.float64)
        drops = np.asarray(self.n_dropped, dtype=np.float64)
        return {
            "n_examples": self.count,
            "k_drop_mean": float(drops.mean()) if drops.size else None,
            "k_drop_max": int(drops.max()) if drops.size else None,
            "sector_width_rad_mean": float(widths.mean()) if widths.size else None,
            "sector_width_rad_max": float(widths.max()) if widths.size else None,
        }


class ElementwiseMasker:
    """Cell R mask: Bernoulli keep over the full token tensor."""

    def __init__(self, seed: int = ELEMENTWISE_MASK_SEED):
        self.generator = torch.Generator().manual_seed(seed)
        self.stats: list[dict] = []

    def draw(self, shape, p: float, device) -> torch.Tensor:
        keep_prob = 1.0 - p
        if p >= 1.0:
            mask = torch.zeros(shape, device=device)
            self.stats.append({"p": p, "kept_fraction": 0.0, "all_zero": True})
            return mask
        mask = torch.bernoulli(
            torch.full(shape, keep_prob), generator=self.generator
        ).to(device)
        self.stats.append({
            "p": p,
            "kept_fraction": float((mask != 0).float().mean().item()),
            "all_zero": bool((mask == 0).all().item()),
        })
        return mask


def build_sparsified_model(seed: int = 42, cell: str = "R"):
    """The exact Arm A/D graph (2 heads) with the route-owned decode override."""
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {sorted(CELLS)}, got {cell!r}")
    _ensure_component_paths()
    from src.models.components.spint import SpintModel
    from src.models.components.streaming_encoders import build_encoder
    from src.models.components.streaming_spint import StreamingSpintModel

    torch.manual_seed(seed)
    decoder = SpintModel(
        model_dim=512, num_covariates=2, window_size=50, num_heads=2,
        num_layers=1, num_id_layers=1, use_learnable_id=True,
        learnable_id_type="mlp", learnable_rep=True,
        # the ROUTE mask replaces D's built-in dynamic dropout at the same
        # site; the flag itself carries no parameters
        dynamic_dropout=False,
    )
    id_encoder = build_encoder(
        "B3S", window_size=50, trial_length=100, id_hidden_dim=128,
        hidden_dim=64, side_dim=4,
    )
    inner = StreamingSpintModel(decoder=decoder, id_encoder=id_encoder, decoder_mode="coupled")
    model = SparsifiedStreamingSpintModel.from_parent(inner, cell=cell)
    return model


class SparsifiedStreamingSpintModel(torch.nn.Module):
    """Route-owned wrapper applying the sparsification at the exact D site.

    `from_parent` copies every parameter/buffer of a freshly built parent
    StreamingSpintModel (identical state keys/shapes/count), so strict-loading
    the canonical Arm A initial state works unchanged.  `decode_with_identity`
    replicates the parent path exactly and inserts ONE mask application
    between `src = src + identity` and `fc_in`.
    """

    def __init__(self, parent, cell: str):
        super().__init__()
        # reuse the parent's modules by reference (same objects, same names)
        self.decoder = parent.decoder
        self.id_encoder = parent.id_encoder
        self._cell = cell
        self.sparsification_enabled = False
        self.current_p: float | None = None
        self.masker = None  # set by the runner when enabling
        self.theta = None
        self.valid = None
        self.mask_stats: list[dict] = []

    @classmethod
    def from_parent(cls, parent, cell: str):
        model = cls(parent, cell=cell)
        return model

    # -- graph parity bookkeeping ------------------------------------------
    @property
    def window_size(self):
        return self.decoder.window_size

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse=recurse)

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

    def set_session_authority(self, theta: torch.Tensor, valid: torch.Tensor):
        self.theta = theta
        self.valid = valid

    # -- the exact sparsification site --------------------------------------
    def apply_sparsification(self, src: torch.Tensor) -> torch.Tensor:
        if not self.sparsification_enabled or self.current_p is None:
            return src  # evaluation path: exact ones, bitwise parent-equal
        p = self.current_p
        if p >= 1.0:
            out = torch.zeros_like(src)
            self.mask_stats.append({"p": p, "all_zero_population": True,
                                    "kept_fraction": 0.0, "max_gain": None})
            return out
        gain = 1.0 / (1.0 - p)  # reported; the applied formula is exact division
        if self._cell == "R":
            mask = self.masker.draw(tuple(src.shape), p, src.device)  # [B,N,W]
            out = src * mask / (1.0 - p)
            kept = float((mask != 0).float().mean().item())
            self.mask_stats.append({
                "p": p, "structure": "elementwise",
                "kept_fraction": kept,
                "all_zero_population": bool((mask == 0).all().item()),
                "max_gain": float(gain),
                "post_mask_token_norm_mean": float(out.detach().norm(dim=-1).mean().item()),
            })
            return out
        # S2: whole-unit keep-mask [B, N] from the frozen sector construction
        theta = self.theta.detach().cpu().numpy()
        valid = self.valid.detach().cpu().numpy()
        masks = np.stack(
            [self.masker.draw(theta, valid, p) for _ in range(src.shape[0])], axis=0
        )
        mask = torch.from_numpy(masks).to(src.device)  # [B, N]
        out = src * mask.unsqueeze(-1) / (1.0 - p)
        self.mask_stats.append({
            "p": p, "structure": "sector_whole_unit",
            "kept_fraction": float((mask != 0).float().mean().item()),
            "all_zero_population": bool((mask == 0).all(dim=1).any().item()),
            "max_gain": float(gain),
            "post_mask_token_norm_mean": float(out.detach().norm(dim=-1).mean().item()),
        })
        return out

    def decode_with_identity(self, neural, identity, neuron_gate=None,
                             live_gain_features=None, live_gain_state=None):
        """Parent path replica with ONE insertion at the D site."""
        src = neural.permute(0, 2, 1)
        if live_gain_features is not None or live_gain_state is not None:
            raise ValueError("the sparsification family never uses live gain paths")
        if neuron_gate is not None:
            src = src * neuron_gate
        src = src + identity
        src = self.apply_sparsification(src)  # <-- the exact site, before fc_in
        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        transformer_output, _ = self.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        output = self.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)


# ---------------------------------------------------------------------------
def build_theta_authority(train_nwbs) -> dict:
    """Source-only RAW T4 direction authority aligned to canonical unit order.

    theta = atan2(raw_c, raw_a) from the un-normalized closed-form T4
    (`[m*cos(phi), m*sin(phi), m, b]`), valid = raw_m > MODULATION_EPS.  Never
    z-scored values; never model-visible.
    """
    from mc_maze.multisession_datamodule import session_name_from_path
    from mc_maze.unit_side_features import (
        MODULATION_EPS,
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
        theta = np.arctan2(raw[:, 1], raw[:, 0])
        valid = raw[:, 2] > MODULATION_EPS
        name = session_name_from_path(nwb_path)
        authority[name] = {
            "theta": theta.astype(np.float64),
            "valid": valid.astype(bool),
            "n_units": int(raw.shape[0]),
            "raw_t4_sha256": hashlib.sha256(raw.astype(np.float32).tobytes()).hexdigest(),
        }
    return authority


def authority_sha256(authority: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(authority):
        entry = authority[name]
        digest.update(name.encode("utf-8"))
        digest.update(entry["theta"].astype(np.float64).tobytes())
        digest.update(entry["valid"].astype(bool).tobytes())
        digest.update(str(entry["n_units"]).encode("utf-8"))
    return digest.hexdigest()
