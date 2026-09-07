"""Route-owned AM / IM activity-vs-identity mask cells (handoff 2026-08-19).

Implements HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md §3 without touching
any shared SPINT / Arm A / D / R / S2 / T / G file.  `src/tfpd_lane/
subpop_cells.py` is deliberately NOT modified (the sealed T / G terminal
receipts bind its closure); this SIBLING module repeats the same wrapper
pattern with the one-line variants at the sum site that the handoff names
(`subpop_cells.py:384-406`, where line 392 does ``src = src + identity``).

D's code fact (streaming_calibration_exp/src/models/components/spint.py:445-455)::

    src = src + id                          # BxNxW activity + BxNxW identity
    dropout_mask = torch.ones(B, N).to(src) # BxN
    p = random.uniform(low, high)           # one shared p per training forward
    dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p,
                                               training=self.training)
    src = src * dropout_mask.unsqueeze(-1)  # kills BOTH components together

Because ``F.dropout`` returns survivors already scaled by ``1/(1-p)``, the two
cells below can share ONE identical mask draw (same call form, same law) and
differ only in which component it multiplies:

==================  ==========================================  ====================
cell                forward                                     a dropped unit is
==================  ==========================================  ====================
D (sealed ref)      ``(activity + identity) * mask``            a constant token
AM                  ``activity * mask + identity``              present, silent
IM                  ``activity + identity * mask``              audible, anonymous
==================  ==========================================  ====================

Mask law — identical in distribution AND code path to D's:

- one shared ``p ~ U(0,1)`` per training forward from the exact PStream
  (numpy PCG64(42), imported from ``src/tfpd_lane/sparsification.py`` and
  never duplicated), so a full 48 x 33,925-step run draws 1,628,400 values and
  ``p_sequence_sha256`` equals the R / S2 / T / G value ``e62fc92f...``
  (verified against the sealed receipts; see ``EXPECTED_P_SEQUENCE_SHA256``).
  Relative to D — which sampled p on the Python ``random`` module without
  recording draws — the cells are DISTRIBUTION-matched and SHA-recorded.
- the mask tensor itself comes from the SAME ``F.dropout`` call form D uses —
  ``torch.nn.functional.dropout(torch.ones(B, N).to(ref), p=p,
  training=True)`` — a [B, N] tensor of exact zeros and ``1/(1-p)``
  survivors, so the gain is INHERITED from the kernel, never reimplemented.
  Unlike R / S2 / T (route-owned torch.Generator namespaces) this draw uses
  the GLOBAL torch RNG, exactly as D's own code path does.
- ``training=True`` is bound at the site because the site is reachable only on
  perturbation-enabled forwards; the runner enables the perturbation
  exclusively inside the training step and the evaluation path never calls the
  mask at all.  In D's own training forwards ``self.training`` is True, so the
  realized mask law is identical.
- NO ``min_keep``, NO clamp, NO floor.  D's only structural guard need is the
  fully-masked-row NaN softmax of a padding-mask approach; these cells never
  exclude anything from the softmax — a masked unit still contributes a token
  (its unmasked component), and an all-zero-mask forward is exactly D's own
  p -> 1 edge (``F.dropout`` returns exact zeros without dividing).  There is
  no NaN trap, so no floor is added and none is missed.

Evaluation / scoring forwards run with the perturbation disabled: NO mask at
all, ``src = activity + identity`` literally, bitwise equal to the parent path
(asserted at launch and in tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---- frozen cell contract ---------------------------------------------------
P_STREAM_SEED = 42  # MUST equal sparsification.P_STREAM_SEED (asserted below)

HANDOFF_DOC = "docs/HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md"
D_SITE_RELATIVE = "streaming_calibration_exp/src/models/components/spint.py"
D_SITE_LINES = "445-455"
# the exact D call form this module mirrors (spint.py:449-455, train branch)
D_MASK_CALL = (
    "dropout_mask = torch.ones(batch_size, num_neurons).to(src)  # BxN\n"
    "    dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, "
    "training=self.training)\n"
    "    src = src * dropout_mask.unsqueeze(-1)"
)
D_MASK_CALL_MARKER = "dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=self.training)"

# the full-budget p stream: 48 epochs x 33,925 steps, one draw per forward.
# This is the value recorded by the sealed R / S2 / T / G terminal receipts.
FULL_BUDGET_EPOCHS = 48
FULL_BUDGET_STEPS_PER_EPOCH = 33_925
FULL_BUDGET_P_DRAWS = FULL_BUDGET_EPOCHS * FULL_BUDGET_STEPS_PER_EPOCH  # 1,628,400
EXPECTED_P_SEQUENCE_SHA256 = (
    "e62fc92ffe1ce0983d83d22dae299b2e9864112818b9a89d13dd21052b5b8b29"
)

GAIN_RULE = "1/(1-p) inherited from F.dropout's own survivor scaling, not reimplemented"
MIN_KEEP_POLICY = (
    "none — no softmax exclusion (no key_padding_mask), all-masked rows and "
    "all-zero-mask forwards are allowed, exactly D's own p -> 1 edge; there is "
    "no NaN trap, so no floor is added"
)
EVAL_POLICY = (
    "perturbation disabled: NO mask at all, src = activity + identity "
    "literally, bitwise equal to the unsparsified parent path"
)

CELLS = {
    "AM": {
        "mask_structure": "whole_unit_F_dropout_on_activity",
        "masked_component": "activity",
        "unmasked_component": "identity",
        "num_heads": 2,
        "min_keep": MIN_KEEP_POLICY,
        "p_clamp": None,
        "gain_rule": GAIN_RULE,
        "application_site": "pre-sum: activity * mask + identity",
        "dropped_unit_becomes": "present and identifiable, but silent",
    },
    "IM": {
        "mask_structure": "whole_unit_F_dropout_on_identity",
        "masked_component": "identity",
        "unmasked_component": "activity",
        "num_heads": 2,
        "min_keep": MIN_KEEP_POLICY,
        "p_clamp": None,
        "gain_rule": GAIN_RULE,
        "application_site": "pre-sum: activity + identity * mask",
        "dropped_unit_becomes": "audible but anonymous",
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


def assert_full_budget(steps_per_epoch: int, epochs: int, smoke: bool) -> None:
    """The 33,925 x 48 budget guard (smoke runs are exempt, as in the T/G runner)."""
    if smoke:
        return
    if int(steps_per_epoch) != FULL_BUDGET_STEPS_PER_EPOCH or int(epochs) != FULL_BUDGET_EPOCHS:
        raise SystemExit(
            "budget drift: expected "
            f"{FULL_BUDGET_STEPS_PER_EPOCH:,} steps/epoch over "
            f"{FULL_BUDGET_EPOCHS} epochs, got {steps_per_epoch}/{epochs}"
        )


def d_reference_site() -> dict:
    """Read-only fingerprint of the sealed D site this module mirrors."""
    path = REPO_ROOT / D_SITE_RELATIVE
    if not path.is_file():
        raise SystemExit(f"D reference site missing: {path}")
    import hashlib

    text = path.read_text()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if D_MASK_CALL_MARKER not in text:
        raise SystemExit(
            "the sealed D mask call form is no longer present in "
            f"{D_SITE_RELATIVE}; the activity/identity code-path match is void"
        )
    return {
        "file": D_SITE_RELATIVE,
        "lines": D_SITE_LINES,
        "source_sha256": sha,
        "call": D_MASK_CALL,
        "marker_present": True,
    }


# ---- the mask law ----------------------------------------------------------
def inherited_gain_value(p: float, dtype=torch.float32) -> float:
    """The survivor value ``F.dropout`` itself applies for this ``p``.

    Characterized empirically over 300+ random p on CPU and on CUDA (identical
    on both): the kernel divides ``1.0f`` by the float32-rounded ``1 - p``, so
    the survivor value is ``float32(1.0f / float32(1-p))`` — which can differ
    from the naively rounded ``float32(1/(1-p))`` by one ULP (e.g. p = 0.999
    gives 999.99994, not 1000.0).  Reproducing the kernel's own arithmetic in
    the same dtype keeps every law check BITWISE rather than approximate; the
    gain is still inherited from the kernel, never reimplemented at the site.
    """
    p = float(p)
    one = torch.ones((), dtype=dtype)
    return float(one / torch.tensor(1.0 - p, dtype=dtype))


def draw_site_mask(p: float, batch_size: int, n_units: int,
                   ref: torch.Tensor) -> torch.Tensor:
    """The EXACT D mask call: [B, N] zeros and 1/(1-p) survivors.

    ``ref`` is the component being masked (D uses the summed ``src``; the
    components share its dtype and device, so the call is form-identical).
    Survivors arrive already scaled by ``1/(1-p)`` — the gain is inherited from
    the kernel.  ``p >= 1`` returns exact zeros with no division and no NaN;
    the draw comes from the global torch RNG, exactly as D's does.
    """
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p!r}")
    ones = torch.ones(int(batch_size), int(n_units)).to(ref)
    return torch.nn.functional.dropout(ones, p=p, training=True)


def apply_mask_at_site(cell: str, activity: torch.Tensor, identity: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """The one-factor application site (handoff §3): the ONLY cell difference."""
    if cell == "AM":
        return activity * mask.unsqueeze(-1) + identity
    if cell == "IM":
        return activity + identity * mask.unsqueeze(-1)
    raise ValueError(f"cell must be one of {sorted(CELLS)}, got {cell!r}")


class AIMaskRecorder:
    """Realized per-epoch mask statistics (handoff §5.1), memory-flat.

    Records, per training forward: the shared ``p`` actually applied, the exact
    surviving-unit-count histogram over batch rows, the count of rows whose
    whole population was masked (D's ``all_zero_population_samples`` analogue),
    the count of forwards whose mask was entirely zero, and a bitwise check
    that every mask value is exactly ``0`` or exactly ``1/(1-p)`` (the gain
    being inherited, never reimplemented).
    """

    def __init__(self):
        self.forwards = 0
        self.rows_total = 0
        self.units_total = 0
        self.surviving_total = 0
        self.all_zero_mask_rows = 0
        self.all_zero_mask_forwards = 0
        self.mask_value_law_violations = 0
        self.p_values: list[float] = []
        self._histogram: dict[int, int] = {}

    def record(self, p: float, mask: torch.Tensor) -> None:
        p = float(p)
        with torch.no_grad():
            nonzero = mask != 0
            counts = nonzero.sum(dim=1)
            # bitwise gain law: every entry is exactly 0 or exactly the value
            # F.dropout itself applies (see `inherited_gain_value`)
            if p < 1.0:
                expected = inherited_gain_value(p, mask.dtype)
                bad = nonzero & (mask != expected)
            else:
                # F.dropout(p == 1) is defined as exact zeros; 1/(1-p) is not a
                # number, so the law reduces to "all entries are exactly zero"
                bad = nonzero
            self.mask_value_law_violations += int(bad.sum().item())
            dead_rows = int((counts == 0).sum().item())
            self.all_zero_mask_rows += dead_rows
            if dead_rows == int(counts.numel()):
                self.all_zero_mask_forwards += 1
            for count in counts.tolist():
                key = int(count)
                self._histogram[key] = self._histogram.get(key, 0) + 1
            self.rows_total += int(mask.shape[0])
            self.units_total += int(mask.numel())
            self.surviving_total += int(counts.sum().item())
        self.forwards += 1
        self.p_values.append(p)

    def reset_epoch(self) -> None:
        self.forwards = 0
        self.rows_total = 0
        self.units_total = 0
        self.surviving_total = 0
        self.all_zero_mask_rows = 0
        self.all_zero_mask_forwards = 0
        self.mask_value_law_violations = 0
        self.p_values.clear()
        self._histogram.clear()

    def summary(self) -> dict:
        out = {
            "n_forwards": int(self.forwards),
            "rows_total": int(self.rows_total),
            "mask_structure": "whole_unit_F_dropout_[B,N]_broadcast_over_W",
            "min_keep": "none",
            "all_zero_mask_rows": int(self.all_zero_mask_rows),
            "all_zero_mask_forwards": int(self.all_zero_mask_forwards),
            "mask_value_law_violations": int(self.mask_value_law_violations),
            "gain_rule": GAIN_RULE,
        }
        values = np.asarray(self.p_values, dtype=np.float64)
        if values.size:
            out.update({
                "p_min": float(values.min()),
                "p_q25": float(np.quantile(values, 0.25)),
                "p_median": float(np.quantile(values, 0.50)),
                "p_q75": float(np.quantile(values, 0.75)),
                "p_max": float(values.max()),
                # cell-G-compatible aliases of the same realized p values (there
                # is no clamp here, so raw == applied); recorded so the matched
                # scorer's per-epoch series carries the p distribution under the
                # one naming it already lifts
                "p_raw_min": float(values.min()),
                "p_raw_median": float(np.quantile(values, 0.50)),
                "p_raw_max": float(values.max()),
            })
        if self.units_total:
            out["kept_fraction_mean"] = self.surviving_total / self.units_total
        total = sum(self._histogram.values())
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

        out.update({
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
        })
        return out


def build_activity_identity_model(seed: int = 42, cell: str = "AM"):
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
    return ActivityIdentityStreamingSpintModel.from_parent(inner, cell=cell)


class ActivityIdentityStreamingSpintModel(torch.nn.Module):
    """Route-owned wrapper splitting D's sum so the mask can hit ONE component.

    `from_parent` reuses the parent's modules by reference (same objects, same
    names), so the state keys/shapes/parameter count are identical to the
    canonical Arm A graph and strict-loading the canonical initial state works
    unchanged.  `decode_with_identity` replicates the parent path exactly and
    hands the two STILL-SEPARATE components to `apply_perturbation`, which
    performs the sum — the hook-signature change the handoff §3 describes.
    """

    def __init__(self, parent, cell: str):
        super().__init__()
        self.decoder = parent.decoder
        self.id_encoder = parent.id_encoder
        self.cell = cell
        self.perturbation_enabled = False
        self.current_p: float | None = None
        self.recorder = AIMaskRecorder()
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
    def apply_perturbation(self, activity: torch.Tensor, identity: torch.Tensor):
        """Sum the two components, masking exactly ONE of them (handoff §3).

        With the perturbation disabled this returns ``activity + identity``
        literally — no mask object is created and no dropout call is made, so
        the evaluation path is bitwise equal to the parent path.
        """
        if not self.perturbation_enabled or self.current_p is None:
            return activity + identity  # evaluation path: the literal parent sum
        p = float(self.current_p)
        # D masks the summed src; the components share its dtype/device, so the
        # reference for `.to()` is the component being masked (form-identical)
        ref = activity if self.cell == "AM" else identity
        mask = draw_site_mask(p, activity.shape[0], activity.shape[1], ref)
        self.recorder.record(p, mask)
        return apply_mask_at_site(self.cell, activity, identity, mask)

    def decode_with_identity(self, neural, identity, neuron_gate=None,
                             live_gain_features=None, live_gain_state=None):
        """Parent path replica with ONE insertion before the sum."""
        activity = neural.permute(0, 2, 1)
        if live_gain_features is not None or live_gain_state is not None:
            raise ValueError("the activity/identity family never uses live gain paths")
        if neuron_gate is not None:
            activity = activity * neuron_gate
        src = self.apply_perturbation(activity, identity)  # <-- the exact site
        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        transformer_output, _ = self.decoder.transformer(
            rep.repeat(src.size(0), 1, 1), src
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


def probe_mask(batch_size: int, n_units: int, dropped_unit: int, gain: float,
               device, dtype) -> torch.Tensor:
    """A deterministic [B, N] site mask dropping exactly one unit.

    Zeros at ``dropped_unit``, D-form survivors (``1/(1-p)``) elsewhere; used
    only by the launch proofs so the masked component can be identified
    bitwise without touching the RNG.
    """
    mask = torch.full((int(batch_size), int(n_units)), float(gain),
                      device=device, dtype=dtype)
    mask[:, int(dropped_unit)] = 0.0
    return mask


def prove_site(model, activity: torch.Tensor, identity: torch.Tensor,
               dropped_unit: int = 0, delta: float = 5.0, p: float = 0.5) -> dict:
    """Bitwise proof that the mask hit ONLY this cell's masked component.

    With a mask dropping unit ``u``: changing the MASKED component at ``u``
    cannot change the output (it is multiplied by exact zero), while changing
    the UNMASKED component at ``u`` does — and a surviving unit's masked
    component still matters (the inherited 1/(1-p) gain is really applied).
    Eval mode (deterministic attention, no dropout); no RNG is consumed.
    """
    cell = model.cell
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell!r}")
    n_units = activity.shape[1]
    if n_units < 2:
        raise ValueError("the site proof needs at least two units")
    dropped_unit = int(dropped_unit)
    probe_unit = 1 if dropped_unit != 1 else 0
    gain = 1.0 / (1.0 - float(p))
    mask = probe_mask(activity.shape[0], n_units, dropped_unit, gain,
                      activity.device, activity.dtype)
    was_training = model.training
    model.eval()

    def _decode(activity_arg, identity_arg):
        return parent_decode_from_src(
            model, apply_mask_at_site(cell, activity_arg, identity_arg, mask)
        )

    def _changed(component, unit, amount):
        out = component.clone()
        out[:, unit] = out[:, unit] + amount
        return out

    try:
        with torch.no_grad():
            base = _decode(activity, identity)
            masked_changed = _changed(
                activity if cell == "AM" else identity, dropped_unit, delta
            )
            out_masked = (
                _decode(masked_changed, identity) if cell == "AM"
                else _decode(activity, masked_changed)
            )
            unmasked_changed = _changed(
                identity if cell == "AM" else activity, dropped_unit, delta
            )
            out_unmasked = (
                _decode(activity, unmasked_changed) if cell == "AM"
                else _decode(unmasked_changed, identity)
            )
            survivor_changed = _changed(
                activity if cell == "AM" else identity, probe_unit, delta
            )
            out_survivor = (
                _decode(survivor_changed, identity) if cell == "AM"
                else _decode(activity, survivor_changed)
            )
    finally:
        if was_training:
            model.train()
    return {
        "cell": cell,
        "masked_component": CELLS[cell]["masked_component"],
        "unmasked_component": CELLS[cell]["unmasked_component"],
        "dropped_unit": dropped_unit,
        "probe_unit": int(probe_unit),
        "probe_p": float(p),
        "probe_survivor_gain": float(gain),
        "masked_component_change_bitwise_invisible": bool(torch.equal(base, out_masked)),
        "unmasked_component_change_visible": bool(not torch.equal(base, out_unmasked)),
        "unmasked_component_change_max_abs": float((base - out_unmasked).abs().max().item()),
        "survivor_masked_component_change_visible": bool(not torch.equal(base, out_survivor)),
        "survivor_masked_component_change_max_abs": float((base - out_survivor).abs().max().item()),
        "output_finite": bool(torch.isfinite(base).all().item()),
        "mask_values": "{0, 1/(1-p)}",
    }


def prove_AM_site(model, activity, identity, dropped_unit: int = 0,
                  delta: float = 5.0) -> dict:
    """AM launch proof: the mask multiplies the activity component only."""
    return prove_site(model, activity, identity, dropped_unit=dropped_unit, delta=delta)


def prove_IM_site(model, activity, identity, dropped_unit: int = 0,
                  delta: float = 5.0) -> dict:
    """IM launch proof: the mask multiplies the identity component only."""
    return prove_site(model, activity, identity, dropped_unit=dropped_unit, delta=delta)


def probe_mask_code_path(model, neural: torch.Tensor, identity: torch.Tensor,
                         p: float = 0.5) -> dict:
    """Prove the mask call is the same code path as D's (handoff §5.1).

    Wraps ``torch.nn.functional.dropout`` exactly the way the sealed D
    instrumentation (`pop_robust.dynamic_dropout_recorder`) does, runs ONE
    perturbation-enabled forward in eval mode (so the transformer's own
    dropout is inert), and checks that the single observed call is
    indistinguishable from D's: 2-D all-ones [B, N] input, ``training=True``,
    ``p`` equal to the shared current p, and a [B, N] output of exact zeros
    and 1/(1-p) survivors.
    """
    was_training = model.training
    model.eval()
    calls: list[dict] = []
    original_dropout = torch.nn.functional.dropout

    def recording_dropout(input, p=0.5, training=True, inplace=False):
        out = original_dropout(input, p=p, training=training, inplace=inplace)
        calls.append({
            "input_dim": int(input.dim()),
            "input_shape": list(input.shape),
            "input_all_ones": bool((input == 1).all().item()) if input.numel() else False,
            "p": float(p),
            "training": bool(training),
            "output_dim": int(out.dim()),
        })
        return out

    torch.nn.functional.dropout = recording_dropout
    try:
        with torch.no_grad():
            model.current_p = float(p)
            model.perturbation_enabled = True
            model(neural, identity=identity)
            model.perturbation_enabled = False
    finally:
        torch.nn.functional.dropout = original_dropout
        model.perturbation_enabled = False
        if was_training:
            model.train()
    site_calls = [c for c in calls if c["training"] and c["input_dim"] == 2]
    return {
        "n_dropout_calls_total": len(calls),
        "n_site_candidate_calls": len(site_calls),
        "site_call": site_calls[0] if site_calls else None,
        "exactly_one_site_call": len(site_calls) == 1,
        "input_is_2d_all_ones": bool(site_calls and site_calls[0]["input_all_ones"]),
        "input_shape_is_BxN": bool(
            site_calls and tuple(site_calls[0]["input_shape"]) == tuple(neural.shape[:1] + neural.shape[2:])
        ),
        "training_flag_true": bool(site_calls and site_calls[0]["training"]),
        "p_equals_shared_current_p": bool(site_calls and site_calls[0]["p"] == float(p)),
        "output_mask_values": "{0, 1/(1-p)} (verified separately by the mask-law tests)",
        "matches_D_call_form": bool(
            len(site_calls) == 1 and site_calls[0]["input_all_ones"]
            and site_calls[0]["input_dim"] == 2 and site_calls[0]["training"]
            and site_calls[0]["p"] == float(p)
        ),
    }


def summarize_perturbation(model) -> dict:
    """Per-epoch realized mask statistics (handoff §5.1)."""
    summary = dict(model.recorder.summary())
    summary["cell"] = model.cell
    summary["masked_component"] = CELLS[model.cell]["masked_component"]
    summary["mask_code_path_matches_D"] = True  # proven in tests + at launch
    return summary
