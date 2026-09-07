"""Route-owned matched-capacity NON-equivariant control (arm C-prime).

Operator follow-up to the family-B equivariant cells: the C - A gate is
confounded by construction, because exact equivariance required disconnecting
and freezing the ordinary fused-token consumer (decoder.transformer + fc_out +
learnable-id stacks, 3,203,684 parameters), so arm C trains 492,951 parameters
while sealed Cell D trains 3,510,842.  C - A therefore cannot isolate
equivariance from capacity.  Arm C-prime is the matched-capacity control:

- IDENTICAL to arm C in every respect EXCEPT the carrier view is not
  covariant.  Same ``EquivariantCarrierConsumer`` class (same layer dims, the
  same 185,793-parameter head), same carrier-blind canonical-Z4 fused path,
  same frozen inactive modules, same trainable set (492,951), same training
  contract / runner / seed / receipt discipline, same raw-T4 carrier authority
  and the same frozen global scalar m_scale.
- The carrier enters as ORDINARY real per-unit features: the per-component
  z-scored RAW (a, c) computed with the source-fit per-component statistics —
  exactly the ordinary pipeline's own quantity, and provably NOT
  rotation-covariant (per-component z-scoring does not commute with rotation;
  see tests/test_equivariant_cell.py) — plus the same invariant scalars as
  arm C (the canonical z-scores of m and b, log1p(|beta|/m_scale), and the
  direction-validity flag).  There is no complex-unit value stream and no
  equivariance claim: the consumer's 2-vector value stream carries the
  z-scored pair instead of the covariant complex units.

Pre-registered reading (in every C-prime receipt): C - C-prime is the
equivariance-at-matched-capacity readout; C-prime - A bounds the capacity
confound; C-prime - B is also reported.  Any superiority claim keeps the same
gate style (external governing mean paired delta >= +0.03 AND >= 10/15
sessions positive).

This module is a NEW file by construction: the already-launched arm B / C runs
hash ``src/tfpd_lane/equivariant_cell.py`` and ``scripts/run_equivariant_cell.py``
in their source closures, so those files stay byte-frozen and everything here
reuses them by import only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
CELL_NAME = "cellCprime_matched_control"
CELL_KEY = "Cprime_matched_control"
# The matched control deliberately shares the consumer namespace/seed with arm C
# (same class, same dims, same init stream) so the ONLY difference is the view.
MATCHED_CONTROL_HEAD_SEED_NOTE = (
    "same consumer class and seed namespace as arm C (EQUIVARIANT_HEAD_SEED); "
    "the graph is identical by construction — only the carrier view differs"
)

MATCHED_CONTROL_GATES = {
    "primary_reading": {
        "comparison": "C_minus_Cprime_external_governing",
        "question": (
            "equivariance at matched capacity: C and C-prime share the "
            "architecture, the 185,793-parameter consumer, the 492,951 "
            "trainable set, the Z4 fused path, the raw-T4 authority and "
            "m_scale; only the carrier view (covariant complex units vs "
            "ordinary z-scored reals) differs"
        ),
        "gate": (
            "mean paired delta >= +0.03 AND >= 10/15 sessions positive "
            "(same style as the pre-registered C - A gate) for any superiority claim"
        ),
    },
    "capacity_confound_bound": {
        "comparison": "Cprime_minus_A_external_governing",
        "question": (
            "bounds the capacity confound: how much of any C - A gap is the "
            "reduced trainable set (492,951 vs sealed Cell D's 3,510,842) "
            "rather than equivariance"
        ),
    },
    "also_reported": [
        {
            "comparison": "Cprime_minus_B",
            "question": "matched-capacity non-covariant control vs rotation augmentation",
        }
    ],
    "granularity": ["both granularities (within-6 / external-15)", "date blocks"],
    "statistics": (
        "paired per-session deltas + 10,000-draw session bootstrap 95% CI + "
        "exact sign pattern (tfpd_lane.matched_scorer.paired_session_stats, seed 42)"
    ),
    "eval_policy": "plain forward; no augmentation",
    "note": (
        "the already-launched C - A / C - B / B - A gates stay pre-registered "
        "as launched; C - C-prime and C-prime - A decompose the C - A gate "
        "into equivariance and capacity"
    ),
}


def _equivariant_cell():
    """Load the (byte-frozen, launch-bound) equivariant_cell module."""
    try:
        from src.tfpd_lane import equivariant_cell

        return equivariant_cell
    except ImportError:
        module = sys.modules.get("tfpd_lane_equivariant")
        if module is not None:
            return module
        import importlib.util

        path = Path(__file__).parent / "equivariant_cell.py"
        spec = importlib.util.spec_from_file_location("tfpd_lane_equivariant", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["tfpd_lane_equivariant"] = module
        spec.loader.exec_module(module)
        return module


class MatchedControlCarrierBundle:
    """Arm C-prime's carrier view: the ordinary real per-unit features.

    Equivariant stream REPLACED: the consumer's 2-vector value stream carries
    the per-component z-scored RAW pair ``(a, c)`` computed with the
    source-fit per-component statistics — the ordinary pipeline's own
    quantity.  Because per-component z-scoring does not commute with
    rotation, ``value2(R beta) != R value2(beta)`` and the model is NOT
    rotation-equivariant (proved by tests and by a launch probe).

    Feature layout (``invariant_features``, IOTA_DIM = 6, mirroring arm C's
    slot count so the consumer shapes are identical):
    ``[a_z, c_z, m_norm, b_norm, log1p(|beta|/m_scale), valid]`` — the first
    two slots are the ordinary z-scored carrier pair (covariant-family slots
    rho_cos/rho_sin of arm C are replaced by the raw z-scored pair).
    """

    __slots__ = (
        "raw", "mean_ac", "std_ac", "m_norm", "b_norm", "m_scale",
        "beta_norm", "valid", "value2",
    )

    def __init__(self, raw, mean_ac, std_ac, m_norm, b_norm, m_scale,
                 beta_norm, valid, value2):
        self.raw = raw
        self.mean_ac = mean_ac
        self.std_ac = std_ac
        self.m_norm = m_norm
        self.b_norm = b_norm
        self.m_scale = m_scale
        self.beta_norm = beta_norm
        self.valid = valid
        self.value2 = value2

    @classmethod
    def from_raw(cls, raw, mean_ac, std_ac, m_norm, b_norm,
                 m_scale: float) -> "MatchedControlCarrierBundle":
        eq = _equivariant_cell()
        raw = torch.as_tensor(raw, dtype=torch.float64)
        if raw.dim() != 2 or raw.shape[1] != 4:
            raise ValueError(f"raw T4 must be [N,4], got {tuple(raw.shape)}")
        mean_ac = torch.as_tensor(mean_ac, dtype=torch.float64).reshape(2)
        std_ac = torch.as_tensor(std_ac, dtype=torch.float64).reshape(2)
        if bool((std_ac <= 0).any()) or not bool(torch.isfinite(std_ac).all()):
            raise ValueError("per-component statistics must be positive and finite")
        a, c = raw[:, 0], raw[:, 1]
        beta = torch.hypot(a, c)
        valid = raw[:, 2] > eq.MODULATION_EPS  # theta-authority validity law
        value2 = torch.stack(
            [(a - mean_ac[0]) / std_ac[0], (c - mean_ac[1]) / std_ac[1]], dim=-1
        )
        return cls(
            raw=raw,
            mean_ac=mean_ac,
            std_ac=std_ac,
            m_norm=torch.as_tensor(m_norm, dtype=torch.float32).clone(),
            b_norm=torch.as_tensor(b_norm, dtype=torch.float32).clone(),
            m_scale=float(m_scale),
            beta_norm=beta,
            valid=valid,
            value2=value2,
        )

    def rotated(self, theta: float) -> "MatchedControlCarrierBundle":
        """The rotated RAW carrier pushed through the SAME ordinary view
        (fixed per-component statistics): the honest input to a
        non-equivariance check — the model sees what the ordinary pipeline
        would see after a physical rotation of the carrier."""
        eq = _equivariant_cell()
        rot = eq.rotation_matrix_2d(theta)
        raw = self.raw.clone()
        a, c = self.raw[:, 0], self.raw[:, 1]
        raw[:, 0] = a * rot[0, 0] + c * rot[0, 1]
        raw[:, 1] = a * rot[1, 0] + c * rot[1, 1]
        return MatchedControlCarrierBundle.from_raw(
            raw, self.mean_ac, self.std_ac, self.m_norm, self.b_norm, self.m_scale
        )

    def unit_two_vector(self) -> torch.Tensor:
        """The consumer's 2-vector VALUE-STREAM hook.

        Deliberately NOT a covariant complex unit here: it returns the
        ordinary per-component z-scored raw pair (float64 [N,2]).
        """
        return self.value2

    def value_two_vector(self) -> torch.Tensor:
        return self.value2

    def invariant_features(self, dtype=torch.float32, device=None) -> torch.Tensor:
        """Per-unit carrier features [N, IOTA_DIM] (float64 core)."""
        features = torch.stack(
            [
                self.value2[:, 0],
                self.value2[:, 1],
                self.m_norm.to(torch.float64),
                self.b_norm.to(torch.float64),
                torch.log1p(self.beta_norm / self.m_scale),
                self.valid.to(torch.float64),
            ],
            dim=-1,
        )
        if device is not None:
            features = features.to(device)
        return features.to(dtype)


def build_matched_control_model(seed: int = 42):
    """The arm C graph VERBATIM (same class, dims, seeds, parameter count).

    The matched control differs from arm C ONLY in the carrier view fed to
    this graph at runtime (a ``MatchedControlCarrierBundle`` instead of a
    covariant ``CarrierBundle``), so the consumer parameter count (185,793),
    the frozen inactive modules, and the trainable set (492,951) are exactly
    arm C's.
    """
    eq = _equivariant_cell()
    return eq.build_equivariant_model(seed=seed)


def build_matched_control_bundles(authority: dict, session_records, m_scale: float,
                                  mean_ac, std_ac) -> dict:
    """Per-session matched-control bundles from the (shared) raw authority.

    ``mean_ac``/``std_ac`` are the SOURCE-FIT per-component statistics of the
    raw (a, c) columns — the same statistics the datamodule's normalizer used
    — so ``value2`` equals the aligned normalized side columns 0:2 bitwise
    (asserted here, the ordinary-pipeline identity).
    """
    mean_ac = np.asarray(mean_ac, dtype=np.float64).reshape(2)
    std_ac = np.asarray(std_ac, dtype=np.float64).reshape(2)
    bundles = {}
    for name, entry in sorted(authority.items()):
        record = session_records[name]
        side = np.asarray(record.side_features)
        if side.shape[0] != entry["n_units"] or side.shape[1] != 4:
            raise SystemExit(
                f"{name}: side features {side.shape} not aligned with authority "
                f"n_units={entry['n_units']}"
            )
        raw = np.asarray(entry["raw"], dtype=np.float64)
        expected_z = ((raw[:, 0:2] - mean_ac) / std_ac).astype(np.float32)
        delta = float(np.abs(expected_z - side[:, 0:2]).max())
        if delta > 1e-6:
            raise SystemExit(
                f"{name}: matched-control z-scored (a,c) drift vs the aligned "
                f"normalized side columns ({delta})"
            )
        bundles[name] = MatchedControlCarrierBundle.from_raw(
            torch.from_numpy(raw),
            torch.from_numpy(mean_ac),
            torch.from_numpy(std_ac),
            m_norm=torch.from_numpy(np.ascontiguousarray(side[:, 2], dtype=np.float32)),
            b_norm=torch.from_numpy(np.ascontiguousarray(side[:, 3], dtype=np.float32)),
            m_scale=m_scale,
        )
    return bundles


def matched_parameter_parity(model_c_prime, model_c) -> dict:
    """Exact-match proof that C-prime's capacity equals arm C's."""
    eq = _equivariant_cell()
    from torch.nn.parameter import UninitializedParameter

    def _count(params):
        return sum(
            p.numel() for p in params if not isinstance(p, UninitializedParameter)
        )

    keys_prime = sorted(model_c_prime.state_dict().keys())
    keys_c = sorted(model_c.state_dict().keys())
    if keys_prime != keys_c:
        raise SystemExit("matched-control state keys differ from arm C")
    shapes_prime = {
        k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter)
            else tuple(v.shape))
        for k, v in model_c_prime.state_dict().items()
    }
    shapes_c = {
        k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter)
            else tuple(v.shape))
        for k, v in model_c.state_dict().items()
    }
    if shapes_prime != shapes_c:
        raise SystemExit("matched-control state shapes differ from arm C")
    accounting_prime = eq.parameter_accounting(model_c_prime)
    accounting_c = eq.parameter_accounting(model_c)
    parity = {
        "state_keys_equal": True,
        "state_shapes_equal": True,
        "consumer_parameters": accounting_prime["new_consumer_parameters"],
        "consumer_parameters_equal": (
            accounting_prime["new_consumer_parameters"]
            == accounting_c["new_consumer_parameters"]
        ),
        "total_trainable": accounting_prime["total_trainable"],
        "total_trainable_equal": (
            accounting_prime["total_trainable"] == accounting_c["total_trainable"]
        ),
        "frozen_canonical_parameters": accounting_prime["frozen_canonical_parameters"],
        "canonical_tensors_strict_loaded": (
            accounting_prime["canonical_tensors_strict_loaded"]
        ),
    }
    if not (parity["consumer_parameters_equal"] and parity["total_trainable_equal"]):
        raise SystemExit(
            f"matched-capacity parity failure vs arm C: {parity}"
        )
    return parity


def nonequivariance_probe(model, neural, calib_trials, side_features,
                          thetas) -> dict:
    """The mirror of the equivariance probe: rotate the carrier through the
    ORDINARY view (fixed per-component statistics) and measure how far the
    output is from rotating — the matched control must violate grossly."""
    if model.carrier is None:
        raise ValueError("set the session carrier bundle before probing")
    eq = _equivariant_cell()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base, _ = model(neural, calib_trials=calib_trials, side_features=side_features)
            base_scale = float(base.abs().max().item())
            violations = []
            identity_bitwise = None
            for theta in thetas:
                pred, _ = model(
                    neural, calib_trials=calib_trials, side_features=side_features,
                    carrier=model.carrier.rotated(float(theta)),
                )
                if float(theta) == 0.0:
                    identity_bitwise = bool(torch.equal(pred, base))
                rot = torch.from_numpy(eq.rotation_matrix_2d(float(theta)))
                expected = base.to(torch.float64) @ rot.t().to(base.device)
                violations.append(
                    float((pred.to(torch.float64) - expected).abs().max().item())
                )
        scale = max(base_scale, 1e-12)
        return {
            "n_rotations": len(thetas),
            "base_output_max_abs": base_scale,
            "max_violation": max(violations) if violations else None,
            "min_violation": min(violations) if violations else None,
            "relative_max_violation": (max(violations) / scale) if violations else None,
            "per_rotation_violations": violations,
            "identity_rotation_bitwise_equal": identity_bitwise,
            "equivariance_claim": False,
        }
    finally:
        if was_training:
            model.train()


def matched_control_integrity_block(*, num_heads: int, canonical_parameters: int,
                                    consumer_parameters: int, total_trainable: int,
                                    m_scale: float, launch_max_violation: float,
                                    launch_relative_violation: float) -> dict:
    eq = _equivariant_cell()
    return {
        "cell": CELL_KEY,
        "family": "B_canonical_frame_equivariance",
        "cell_name": CELL_NAME,
        "role": "matched-capacity NON-equivariant control for the C - A gate",
        "num_heads": {
            "consumer_attention_heads": int(num_heads),
            "decoder_num_heads": int(num_heads),
            "note": "both are 2, identical to arm C",
        },
        "matched_to_arm_c": {
            "consumer_class": "EquivariantCarrierConsumer (the SAME class)",
            "consumer_parameters": int(consumer_parameters),
            "total_trainable": int(total_trainable),
            "fused_path": (
                "canonical Z4 (zeros_like normalized T4) — identical to arm C"
            ),
            "frozen_inactive_modules": list(eq.INACTIVE_CANONICAL_MODULES),
            "trainable_set_equal_to_arm_c": True,
            "head_seed_note": MATCHED_CONTROL_HEAD_SEED_NOTE,
        },
        "carrier_view": {
            "equivariant_stream": "NONE — no complex-unit value stream",
            "value_stream": (
                "the consumer's 2-vector value stream carries the per-component "
                "z-scored RAW (a, c) — an ordinary real feature, not a covariant unit"
            ),
            "per_unit_features": (
                "[a_z, c_z, m_norm, b_norm, log1p(|beta|/m_scale), valid] — the "
                "z-scored raw pair replaces arm C's rho_cos/rho_sin slots, so the "
                "consumer input width is unchanged"
            ),
            "z_scoring": (
                "source-fit per-component statistics (the ordinary pipeline's own "
                "normalizer); provably NOT rotation-covariant (per-component "
                "z-scoring does not commute with rotation)"
            ),
            "invariants_shared_with_arm_c": [
                "m/b source-normalizer z-scores", "log1p(|beta|/m_scale)",
                "direction-validity flag",
            ],
            "m_scale_frozen_global_scalar": float(m_scale),
        },
        "symmetry_claim": "none — deliberately NOT rotation-equivariant",
        "nonequivariance": {
            "launch_max_violation": float(launch_max_violation),
            "launch_relative_max_violation": float(launch_relative_violation),
            "note": (
                "violations are the point: the control must fail the rotation "
                "property that arm C satisfies to 1e-5"
            ),
        },
        "parameter_disclosure": {
            "canonical_tensors_strict_loaded_bitwise": int(canonical_parameters),
            "canonical_tensors_inactive_requires_grad_false": list(
                eq.INACTIVE_CANONICAL_MODULES
            ),
            "new_consumer_parameters": int(consumer_parameters),
            "total_trainable": int(total_trainable),
            "bitwise_parity_at_init_vs_cell_d_claimed": False,
            "capacity_matched_to_arm_c": True,
        },
        "augmentation": "none",
        "eval_policy": "plain forward",
        "gates": MATCHED_CONTROL_GATES,
    }
