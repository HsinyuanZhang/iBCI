"""Route-owned T / G sub-population decomposition cells (handoff 2026-08-18).

Implements HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3 cells
T and G without touching any shared SPINT / Arm A / D / DH / R / S2 file:

- `PStream` is NOT duplicated: the exact p stream is imported from the
  route-owned `src/tfpd_lane/sparsification.py` (numpy PCG64(42), one
  `p ~ U(0,1)` per training forward), so T and G draw the same sequence as
  R / S2 and their `p_sequence_sha256` values are directly comparable.

- Cell T (true removal via `key_padding_mask`): one shared `p` per training
  forward plus a per-(batch, unit) whole-unit Bernoulli keep mask from a
  route-owned `torch.Generator` namespace (CPU, seed 42_003).  The boolean
  complement is passed as `key_padding_mask` ([B, N], True = excluded) to
  `decoder.transformer(...)`, so dropped units are genuinely absent from the
  attention softmax.  `src` is NOT zeroed (no constant placeholder token) and
  no `1/(1-p)` rescaling is applied (attention renormalizes over survivors).
  `min_keep = 4` per row: after the Bernoulli draw, any batch row keeping
  fewer than 4 units randomly restores (4 - kept) dropped units using the same
  generator, guarding the fully-masked-row NaN softmax that sealed D can never
  hit (its placeholder tokens always exist).

- Cell G (random global gain, claim-killer control): nothing is masked.  Every
  unit window is multiplied by `1/(1 - clamp(p, 0, 0.95))` at the exact same
  site (after `src = activity + identity`, before `fc_in`).  The clamp keeps
  the gain finite at 20x; the realized gain distribution and the clamp trigger
  rate (raw `p > 0.95`) are recorded per epoch.

Both cells: with the perturbation disabled (evaluation / scoring) no mask and
no gain is applied, `key_padding_mask` is not passed at all, and the forward is
bitwise equal to the unsparsified parent path.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

# ---- frozen cell contract ---------------------------------------------------
P_STREAM_SEED = 42  # MUST equal sparsification.P_STREAM_SEED (asserted below)
UNIT_MASK_SEED = 42_003  # route-owned namespace for cell T's whole-unit draws
MIN_KEEP = 4  # cell T: minimum surviving units per batch row
GAIN_CLAMP = 0.95  # cell G: p is clamped to [0, 0.95] before the gain

CELLS = {
    "T": {
        "mask_structure": "whole_unit_bernoulli_key_padding_mask",
        "num_heads": 2,
        "min_keep": MIN_KEEP,
        "p_clamp": None,
        "rescaling_policy": (
            "none: src is not zeroed and no 1/(1-p) rescaling is applied; "
            "attention renormalizes over the surviving keys"
        ),
    },
    "G": {
        "mask_structure": "none_global_gain_all_units",
        "num_heads": 2,
        "min_keep": "n/a (no masking)",
        "p_clamp": GAIN_CLAMP,
        "rescaling_policy": (
            "all unit windows multiplied by 1/(1-clamp(p, 0, 0.95)) at the "
            "exact D site (after activity+identity, before fc_in)"
        ),
    },
}


# ---------------------------------------------------------------------------
def resolve_sparsification_module():
    """Return THE route-owned sparsification module — never a copy of PStream.

    Prefers the runner's file-path registration (`tfpd_lane_sparsification`)
    and falls back to the package import used by the test suite.
    """
    module = sys.modules.get("tfpd_lane_sparsification")
    if module is not None and hasattr(module, "PStream"):
        return module
    from src.tfpd_lane import sparsification as module

    return module


def pstream(seed: int | None = None):
    """A fresh exact p stream from the shared (never duplicated) definition."""
    module = resolve_sparsification_module()
    if P_STREAM_SEED != module.P_STREAM_SEED:
        raise RuntimeError(
            "p-stream seed drift vs src/tfpd_lane/sparsification.py: "
            f"{P_STREAM_SEED} != {module.P_STREAM_SEED}"
        )
    return module.PStream(seed=module.P_STREAM_SEED if seed is None else seed)


def _ensure_component_paths():
    """Delegates to the sibling module so `src` stays ONE package object."""
    return resolve_sparsification_module()._ensure_component_paths()


class UnitRemovalMasker:
    """Cell T whole-unit Bernoulli keep mask with the `min_keep` restoration.

    All draws happen on CPU with a dedicated `torch.Generator` (seed 42_003),
    so the realized masks are device-independent and never perturb data
    ordering, the global torch RNG, or any other generator.  Per-epoch
    bookkeeping is an exact surviving-count histogram (memory-flat across a
    1.6M-step run); `reset_epoch()` is called by the runner every epoch.
    """

    def __init__(self, seed: int = UNIT_MASK_SEED, min_keep: int = MIN_KEEP):
        self.generator = torch.Generator().manual_seed(int(seed))
        self.min_keep = int(min_keep)
        self.seed = int(seed)
        self._histogram: dict[int, int] = {}
        self.rows_total = 0
        self.forwards = 0
        self.min_keep_trigger_rows = 0
        self.units_restored = 0

    # -- the draw ------------------------------------------------------------
    def draw(self, batch_size: int, n_units: int, p: float,
             device) -> torch.Tensor:
        """Bernoulli whole-unit keep mask [B, N] (bool, True = kept)."""
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"p must lie in [0, 1], got {p!r}")
        batch_size, n_units = int(batch_size), int(n_units)
        if batch_size <= 0 or n_units <= 0:
            raise ValueError("batch_size and n_units must be positive")
        if n_units < self.min_keep:
            # a population smaller than the floor is never thinned
            keep = torch.ones(batch_size, n_units, dtype=torch.bool)
        else:
            keep_prob = min(1.0, max(0.0, 1.0 - float(p)))
            keep = torch.bernoulli(
                torch.full((batch_size, n_units), keep_prob),
                generator=self.generator,
            ).bool()
            counts = keep.sum(dim=1)
            for row in torch.nonzero(counts < self.min_keep).flatten().tolist():
                kept = int(counts[row].item())
                need = self.min_keep - kept
                dropped = torch.nonzero(~keep[row]).flatten()
                order = torch.randperm(dropped.numel(), generator=self.generator)
                keep[row, dropped[order[:need]]] = True
                self.min_keep_trigger_rows += 1
                self.units_restored += need
        for count in keep.sum(dim=1).tolist():
            key = int(count)
            self._histogram[key] = self._histogram.get(key, 0) + 1
        self.rows_total += batch_size
        self.forwards += 1
        return keep.to(device)

    # -- per-epoch bookkeeping -------------------------------------------------
    def reset_epoch(self) -> None:
        self._histogram.clear()
        self.rows_total = 0
        self.forwards = 0
        self.min_keep_trigger_rows = 0
        self.units_restored = 0

    def summary(self) -> dict:
        """Realized surviving-unit statistics (handoff §5.6), exact histogram."""
        total = sum(self._histogram.values())
        out = {
            "min_keep": self.min_keep,
            "masker_seed": self.seed,
            "rows_total": int(total),
            "min_keep_trigger_rows": int(self.min_keep_trigger_rows),
            "min_keep_trigger_rate": (
                self.min_keep_trigger_rows / total if total else None
            ),
            "units_restored_by_min_keep": int(self.units_restored),
        }
        if not total:
            return {**out, "surviving_units_histogram": {}}
        keys = sorted(self._histogram)

        def quantile(q: float) -> int:
            threshold = q * total
            cumulative = 0
            for key in keys:
                cumulative += self._histogram[key]
                if cumulative >= threshold:
                    return key
            return keys[-1]

        out.update(
            {
                "surviving_units_min": int(keys[0]),
                "surviving_units_q05": int(quantile(0.05)),
                "surviving_units_q25": int(quantile(0.25)),
                "surviving_units_median": int(quantile(0.50)),
                "surviving_units_q75": int(quantile(0.75)),
                "surviving_units_q95": int(quantile(0.95)),
                "surviving_units_max": int(keys[-1]),
                "surviving_units_mean": float(
                    sum(k * n for k, n in self._histogram.items()) / total
                ),
                "surviving_units_histogram": {str(k): self._histogram[k] for k in keys},
                "rows_below_min_keep_after_guard": int(
                    sum(n for k, n in self._histogram.items() if k < self.min_keep)
                ),
                "all_masked_rows_after_guard": int(self._histogram.get(0, 0)),
            }
        )
        return out


class GlobalGainRecorder:
    """Cell G realized-gain bookkeeping (handoff §5.6)."""

    def __init__(self, clamp: float = GAIN_CLAMP):
        self.clamp = float(clamp)
        self.p_raw: list[float] = []
        self.gain: list[float] = []

    def record(self, p_raw: float, p_clamped: float, gain: float) -> None:
        self.p_raw.append(float(p_raw))
        self.gain.append(float(gain))

    def reset_epoch(self) -> None:
        self.p_raw.clear()
        self.gain.clear()

    def summary(self) -> dict:
        gains = np.asarray(self.gain, dtype=np.float64)
        raw = np.asarray(self.p_raw, dtype=np.float64)
        out = {
            "gain_rule": f"1/(1-clamp(p, 0, {self.clamp}))",
            "clamp": self.clamp,
            "n_forwards": int(gains.size),
        }
        if not gains.size:
            return out
        out.update(
            {
                "gain_min": float(gains.min()),
                "gain_q25": float(np.quantile(gains, 0.25)),
                "gain_median": float(np.quantile(gains, 0.50)),
                "gain_q75": float(np.quantile(gains, 0.75)),
                "gain_max": float(gains.max()),
                "gain_mean": float(gains.mean()),
                "p_raw_min": float(raw.min()),
                "p_raw_median": float(np.quantile(raw, 0.50)),
                "p_raw_max": float(raw.max()),
                "clamp_trigger_count": int((raw > self.clamp).sum()),
                "clamp_trigger_rate": float((raw > self.clamp).mean()),
            }
        )
        return out


def build_subpop_model(seed: int = 42, cell: str = "T"):
    """The exact Arm A / D graph (2 heads) with the route-owned decode override."""
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
        # the ROUTE perturbation replaces D's built-in dynamic dropout at the
        # same site; the flag itself carries no parameters
        dynamic_dropout=False,
    )
    id_encoder = build_encoder(
        "B3S", window_size=50, trial_length=100, id_hidden_dim=128,
        hidden_dim=64, side_dim=4,
    )
    inner = StreamingSpintModel(decoder=decoder, id_encoder=id_encoder, decoder_mode="coupled")
    model = SubpopStreamingSpintModel.from_parent(inner, cell=cell)
    if cell == "T":
        model.masker = UnitRemovalMasker()
    else:
        model.gain_recorder = GlobalGainRecorder()
    return model


class SubpopStreamingSpintModel(torch.nn.Module):
    """Route-owned wrapper applying the T / G perturbation at the exact D site.

    `from_parent` reuses the parent's modules by reference (same objects, same
    names), so the state keys/shapes/parameter count are identical to the
    canonical Arm A graph and strict-loading the canonical initial state works
    unchanged.  `decode_with_identity` replicates the parent path exactly and
    inserts ONE perturbation between `src = src + identity` and `fc_in`.
    """

    def __init__(self, parent, cell: str):
        super().__init__()
        self.decoder = parent.decoder
        self.id_encoder = parent.id_encoder
        self.cell = cell
        self.perturbation_enabled = False
        self.current_p: float | None = None
        self.masker = None  # cell T: UnitRemovalMasker (set at build time)
        self.gain_recorder = None  # cell G: GlobalGainRecorder
        self.perturbation_stats: list[dict] = []
        self.theta = None  # authority plumbing kept for interface parity only
        self.valid = None

    @classmethod
    def from_parent(cls, parent, cell: str):
        return cls(parent, cell=cell)

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

    # -- the exact perturbation site ----------------------------------------
    def apply_perturbation(self, src: torch.Tensor):
        """Return ``(src, key_padding_mask)`` after this cell's ONE change.

        Cell T returns ``src`` untouched together with the boolean complement
        of the whole-unit keep mask ([B, N], True = excluded from attention).
        Cell G returns ``src * 1/(1-clamp(p, 0, 0.95))`` and no mask.  With the
        perturbation disabled both return ``src`` unchanged and no mask, which
        makes the evaluation path bitwise equal to the parent path.
        """
        if not self.perturbation_enabled or self.current_p is None:
            return src, None  # evaluation path: no mask, no gain, no rescale
        p = float(self.current_p)
        if self.cell == "T":
            if self.masker is None:
                raise RuntimeError("cell T requires a UnitRemovalMasker")
            keep = self.masker.draw(src.shape[0], src.shape[1], p, src.device)
            key_padding_mask = ~keep  # [B, N]; True = excluded from the softmax
            counts = keep.sum(dim=1)
            self.perturbation_stats.append({
                "cell": "T",
                "p": p,
                "n_units": int(src.shape[1]),
                "kept_fraction": float(keep.float().mean().item()),
                "surviving_units_min": int(counts.min().item()),
                "surviving_units_mean": float(counts.float().mean().item()),
                "surviving_units_max": int(counts.max().item()),
                "src_zeroed": False,
                "rescaling": "none",
            })
            return src, key_padding_mask
        if self.gain_recorder is None:
            raise RuntimeError("cell G requires a GlobalGainRecorder")
        p_clamped = min(max(p, 0.0), GAIN_CLAMP)
        gain = 1.0 / (1.0 - p_clamped)
        self.gain_recorder.record(p, p_clamped, gain)
        self.perturbation_stats.append({
            "cell": "G",
            "p_raw": p,
            "p_clamped": p_clamped,
            "gain": gain,
            "clamped": bool(p > GAIN_CLAMP),
            "mask": "none",
        })
        return src * gain, None

    def decode_with_identity(self, neural, identity, neuron_gate=None,
                             live_gain_features=None, live_gain_state=None):
        """Parent path replica with ONE insertion at the D site."""
        src = neural.permute(0, 2, 1)
        if live_gain_features is not None or live_gain_state is not None:
            raise ValueError("the sub-population family never uses live gain paths")
        if neuron_gate is not None:
            src = src * neuron_gate
        src = src + identity
        src, key_padding_mask = self.apply_perturbation(src)  # <-- the exact site
        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        if key_padding_mask is None:
            # evaluation path: the literal parent call, bitwise equal
            transformer_output, _ = self.decoder.transformer(
                rep.repeat(src.size(0), 1, 1), src
            )
        else:
            transformer_output, _ = self.decoder.transformer(
                rep.repeat(src.size(0), 1, 1), src, key_padding_mask=key_padding_mask
            )
        output = self.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# launch-time proofs (shared by the runner and the test suite)
# ---------------------------------------------------------------------------
def parent_decode_from_src(model, src: torch.Tensor) -> torch.Tensor:
    """The unsparsified parent decode from an already-assembled `src` [B,N,W]."""
    x = model.decoder.fc_in(src)
    rep = model.decoder.fc_in(model.decoder.rep).to(x)
    out, _ = model.decoder.transformer(rep.repeat(x.size(0), 1, 1), x)
    return model.decoder.fc_out(out).permute(0, 2, 1)


def parent_decode(model, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    """The unsparsified parent decode path through the same modules, same order."""
    return parent_decode_from_src(model, neural.permute(0, 2, 1) + identity)


def prove_true_removal(model, src: torch.Tensor, excluded_unit: int = 0,
                       delta: float = 1.0) -> dict:
    """Cell T launch/test proof: a padding-masked unit is genuinely absent.

    With `key_padding_mask` excluding unit `u`, changing `src[:, u, :]` leaves
    the transformer output bitwise unchanged, while changing a kept unit's
    window does change it.  Eval mode (deterministic attention, no dropout).
    """
    was_training = model.training
    model.eval()
    batch, n_units = src.shape[0], src.shape[1]
    if n_units < 2:
        raise ValueError("the removal proof needs at least two units")
    probe_unit = 1 if excluded_unit != 1 else 0
    key_padding_mask = torch.zeros(batch, n_units, dtype=torch.bool,
                                   device=src.device)
    key_padding_mask[:, excluded_unit] = True
    try:
        with torch.no_grad():
            query = model.decoder.fc_in(model.decoder.rep).to(src).repeat(batch, 1, 1)

            def _decode(source):
                out, _ = model.decoder.transformer(
                    query, model.decoder.fc_in(source), key_padding_mask=key_padding_mask
                )
                return out

            base = _decode(src)
            changed_excluded = src.clone()
            changed_excluded[:, excluded_unit] = changed_excluded[:, excluded_unit] + delta
            out_excluded = _decode(changed_excluded)
            changed_kept = src.clone()
            changed_kept[:, probe_unit] = changed_kept[:, probe_unit] + delta
            out_kept = _decode(changed_kept)
    finally:
        if was_training:
            model.train()
    return {
        "excluded_unit": int(excluded_unit),
        "probe_unit": int(probe_unit),
        "excluded_unit_change_bitwise_invisible": bool(torch.equal(base, out_excluded)),
        "kept_unit_change_visible": bool(not torch.equal(base, out_kept)),
        "kept_unit_change_max_abs": float((base - out_kept).abs().max().item()),
        "output_finite": bool(torch.isfinite(base).all().item()),
        "min_keep": MIN_KEEP,
    }


def prove_gain_rule(model, neural: torch.Tensor, identity: torch.Tensor,
                    p: float) -> dict:
    """Cell G launch/test proof: the site applies exactly `src * 1/(1-clamp(p))`."""
    p_clamped = min(max(float(p), 0.0), GAIN_CLAMP)
    gain = 1.0 / (1.0 - p_clamped)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model.current_p = float(p)
            model.perturbation_enabled = True
            mine, _ = model(neural, identity=identity)
            model.perturbation_enabled = False
            expected = parent_decode_from_src(
                model, (neural.permute(0, 2, 1) + identity) * gain
            )
    finally:
        model.perturbation_enabled = False
        if was_training:
            model.train()
    return {
        "p_raw": float(p),
        "p_clamped": p_clamped,
        "gain": gain,
        "clamped": bool(float(p) > GAIN_CLAMP),
        "bitwise_equal_to_parent_times_gain": bool(torch.equal(mine, expected)),
    }


def summarize_perturbation(model) -> dict:
    """Per-epoch realized perturbation statistics (handoff §5.6)."""
    forwards = model.perturbation_stats
    base = {"n_forwards": len(forwards)}
    if model.cell == "T":
        kept = [entry["kept_fraction"] for entry in forwards]
        return {
            **base,
            **model.masker.summary(),
            "kept_fraction_mean": float(np.mean(kept)) if kept else None,
        }
    return {**base, **model.gain_recorder.summary()}
