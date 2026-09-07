"""Deliverable B: forward-only mechanism diagnostics over a frozen Stage-1
checkpoint (HANDOFF_POST_BILINEAR_CONSUMER_EXTENSION_20260815.md §5.7).

Four same-checkpoint, no-retraining diagnostics:

- ``aligned``           : the T4 carrier as produced by the pipeline;
- ``zero``              : exact zero carrier (``torch.zeros_like``) — the Z4
                          activity fallback;
- ``wrong_pair``        : row-permuted carrier (valid content, wrong unit
                          pairing), via the model's own frozen
                          ``wrong_pair_carrier`` static method;
- ``activity_destroyed``: each unit's count sequence circularly shifted by
                          a random per-unit offset (frozen seed), destroying
                          the unit/time alignment while preserving each
                          unit's marginal count sequence.

Scoring uses `src.tfpd.synth.r2_score` — the same pooled variance-weighted
metric family as `src/tfpd/gates.py`.  T4 minus Z4 remains the necessary
carrier-content check; the wrong-pair contrast is the distinctive one for
the per-unit task-correspondence claim (§5.7).  A wrong-pair penalty LARGER
than the zero penalty is strong evidence but deliberately not required
(parent §8.6 — a good model may conservatively attenuate a corrupt carrier).

Checkpoint loading rebuilds the underlying decoder directly from the
Lightning checkpoint's ``hyper_parameters`` / ``state_dict`` (prefix
``model.``) via ``torch.load``; the LightningModule is never instantiated
and no Lightning import is needed on this path.  Real-datamodule scoring is
an interface only (``--datamodule-config``): it refuses before instantiating
anything.
"""

from __future__ import annotations

import torch
from torch import nn

from src.tfpd.bilinear_readin import BilinearTaskFrameDecoder
from src.tfpd.population_vector import LearnedPopulationVectorDecoder
from src.tfpd.synth import SyntheticSession, generate_session, r2_score

DIAGNOSTICS = ("aligned", "zero", "wrong_pair", "activity_destroyed")

DEFAULT_WRONG_PAIR_SEED = 0
DEFAULT_DESTROY_SEED = 0


# ---- checkpoint loading -------------------------------------------------------


def load_stage1_model(checkpoint_path) -> nn.Module:
    """Rebuild a Stage-1 decoder from a Lightning checkpoint, CPU-only.

    ``torch.load`` -> ``hyper_parameters``/``state_dict`` -> rebuild
    ``BilinearTaskFrameDecoder`` / ``LearnedPopulationVectorDecoder``
    directly (no LightningModule instantiation).  The re-initialization seed
    is irrelevant: every trained tensor is overwritten by the checkpoint
    state_dict (strict load).
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"not a Lightning checkpoint: {checkpoint_path!r}")
    hparams = ckpt.get("hyper_parameters") or {}
    model_name = hparams.get("model_name")
    if model_name not in ("bilinear", "population_vector"):
        raise ValueError(f"checkpoint does not declare a known Stage-1 model: {model_name!r}")
    model = (
        BilinearTaskFrameDecoder(
            window_size=20, carrier_dim=4, feature_dim=16, embed_dim=16,
            hidden_dim=64, latent_dim=32, gru_hidden=64, num_covariates=2,
        )
        if model_name == "bilinear"
        else LearnedPopulationVectorDecoder(
            window_size=20, carrier_dim=4, hidden_dim=64,
            basis_dim=16, gru_hidden=64, num_covariates=2,
        )
    )
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    if not state:
        raise ValueError("checkpoint state_dict has no 'model.'-prefixed entries")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---- diagnostic arms ----------------------------------------------------------


def destroy_activity(counts: torch.Tensor, seed: int = DEFAULT_DESTROY_SEED) -> torch.Tensor:
    """Random per-unit circular shift along time; [T, N] -> [T, N].

    Preserves each unit's marginal count sequence (same values, same unit),
    destroys the alignment between unit activity and behaviour.  The seed is
    frozen so the arm is reproducible.
    """
    gen = torch.Generator().manual_seed(seed)
    t_len, n = counts.shape
    offsets = torch.randint(0, t_len, (n,), generator=gen)
    out = counts.clone()
    for i in range(n):
        out[:, i] = torch.roll(counts[:, i], shifts=int(offsets[i].item()))
    return out


def diagnostic_inputs(
    model: nn.Module,
    session: SyntheticSession,
    wrong_pair_seed: int = DEFAULT_WRONG_PAIR_SEED,
    destroy_seed: int = DEFAULT_DESTROY_SEED,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per-diagnostic batched (counts, carrier) inputs.

    The wrong-pair arm reuses the model's own frozen ``wrong_pair_carrier``
    static method (both Stage-1 decoders implement it).
    """
    x = session.counts.unsqueeze(0)  # [1, T, N]
    carrier = session.carrier.unsqueeze(0)  # [1, N, 4]
    return {
        "aligned": (x, carrier.clone()),
        "zero": (x, torch.zeros_like(carrier)),
        "wrong_pair": (x, model.wrong_pair_carrier(carrier, seed=wrong_pair_seed)),
        "activity_destroyed": (
            destroy_activity(session.counts, destroy_seed).unsqueeze(0),
            carrier.clone(),
        ),
    }


def run_mech_diag(
    model: nn.Module,
    session: SyntheticSession,
    wrong_pair_seed: int = DEFAULT_WRONG_PAIR_SEED,
    destroy_seed: int = DEFAULT_DESTROY_SEED,
) -> dict:
    """Forward-only four diagnostics; per-diagnostic R^2 plus deltas.

    R^2 uses `src.tfpd.synth.r2_score`, matching the house metric family.
    """
    model.eval()
    behaviour = session.behaviour
    inputs = diagnostic_inputs(model, session, wrong_pair_seed, destroy_seed)
    r2: dict[str, float] = {}
    with torch.no_grad():
        for name, (counts, carrier) in inputs.items():
            pred = model(counts, carrier)[0]
            r2[name] = r2_score(pred, behaviour)
    return {
        "per_diagnostic_r2": r2,
        "deltas": {
            "t4_minus_z4": r2["aligned"] - r2["zero"],
            "t4_minus_wrong_pair": r2["aligned"] - r2["wrong_pair"],
            "t4_minus_activity_destroyed": r2["aligned"] - r2["activity_destroyed"],
        },
        "seeds": {"wrong_pair": wrong_pair_seed, "destroy": destroy_seed},
        "num_units": session.num_units,
        "num_bins": int(session.counts.shape[0]),
    }


def run_mech_diag_cohort(
    model: nn.Module,
    sessions: list[SyntheticSession],
    wrong_pair_seed: int = DEFAULT_WRONG_PAIR_SEED,
    destroy_seed: int = DEFAULT_DESTROY_SEED,
) -> dict:
    per_session = [
        run_mech_diag(model, s, wrong_pair_seed, destroy_seed) for s in sessions
    ]
    means = {
        name: sum(p["per_diagnostic_r2"][name] for p in per_session) / len(per_session)
        for name in DIAGNOSTICS
    }
    return {
        "schema": "tfpd_mech_diag_v1",
        "num_sessions": len(sessions),
        "per_diagnostic_r2_mean": means,
        "per_session": per_session,
        "interpretation_note": (
            "t4_minus_z4 is the necessary carrier-content check; wrong_pair isolates "
            "per-unit task correspondence; a wrong-pair penalty larger than the zero "
            "penalty is strong evidence but not required (parent 8.6)"
        ),
    }


def synthetic_cohort(seed: int, num_sessions: int, **kwargs) -> list[SyntheticSession]:
    """Synthetic smoke cohort (never touches real NWB / Dandi data)."""
    return [generate_session(seed=seed * 1000 + k, **kwargs) for k in range(num_sessions)]
