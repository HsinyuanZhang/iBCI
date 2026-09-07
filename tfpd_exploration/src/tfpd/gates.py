"""Stage-0 gate implementations shared by the pytest suite and the receipt runner.

Each function returns (passed: bool, detail: dict).  Thresholds and fairness rules
are the frozen values from docs/TFPD_STAGE0_SYNTHETIC_GATE_CONTRACT_20260815.md
(amendment block included).

Fairness rules (G5):

- the aligned / zero-carrier / wrong-pair arms start from ONE initial state
  (deep-copied per arm), removing initialization luck from the contrast;
- the headline metric is QUERY-region R^2 only; the support-region R^2 is
  reported separately and never enters the pass rule;
- per-session numbers are recorded, not just the cohort mean.

Non-vacuousness rules (G3):

- branch liveness is measured on the branch's own parameters (named per model)
  AND on the input tensors, not on name-prefix heuristics over all parameters;
- a wrong-pair negative control (row-permuted carrier) must fail to reach the
  aligned arm, proving the carrier is consumed as per-unit content.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from src.tfpd.bilinear_readin import BilinearTaskFrameDecoder
from src.tfpd.bilinear_lowrank_readin import BilinearLowRankTaskFrameDecoder
from src.tfpd.bilinear_state_gain import BilinearStateGainTaskFrameDecoder
from src.tfpd.population_vector import LearnedPopulationVectorDecoder
from src.tfpd.synth import SyntheticSession, generate_session, r2_score

PERMUTATION_TOL = 1e-5
VARIABLE_N_SET = (24, 96, 192)
VARIABLE_N_VARIANCE_RATIO_MAX = 10.0
CARRIER_EFFECT_MIN_DELTA = 0.10
TRAIN_STEPS = 400
TRAIN_SESSIONS = 8
EVAL_SESSIONS = 4
QUERY_ONLY_HEADLINE = True

# Branch parameter name prefixes, per model, for the G3 liveness check.
BRANCH_PARAMS: dict[str, dict[str, tuple[str, ...]]] = {
    "bilinear": {
        "activity": ("activity_encoder", "readin_activity"),
        "carrier": ("carrier_map", "readin_carrier"),
    },
    "population_vector": {
        "activity": ("activity",),
        "carrier": ("confidence",),
    },
    "bilinear_lowrank": {
        "activity": ("activity_encoder", "readin.readin_activity", "readin.readout_latent"),
        "carrier": ("readin.carrier_map",),
    },
    "bilinear_state_gain": {
        "activity": ("activity_encoder", "readin_activity"),
        "carrier": ("carrier_map", "readin_carrier"),
        # state_gain (W) is the new mechanism under test; G3's finite check
        # covers it, but it belongs to neither the activity nor carrier branch.
    },
}


def build_models(seed: int = 42) -> dict[str, nn.Module]:
    torch.manual_seed(seed)
    return {
        "bilinear": BilinearTaskFrameDecoder(),
        "population_vector": LearnedPopulationVectorDecoder(),
        "bilinear_lowrank": BilinearLowRankTaskFrameDecoder(),
        "bilinear_state_gain": BilinearStateGainTaskFrameDecoder(),
    }


def _branch_grad_norm(model: nn.Module, prefixes: tuple[str, ...]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            if parameter.grad is None:
                return float("nan")
            total += float(parameter.grad.norm())
    return total


def gate1_permutation_invariance(model: nn.Module, seed: int = 0) -> tuple[bool, dict]:
    session = generate_session(seed=seed, num_units=96)
    x = session.counts.unsqueeze(0)
    carrier = session.carrier.unsqueeze(0)
    with torch.no_grad():
        reference = model(x, carrier)
    generator = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(100):
        perm = torch.randperm(x.shape[-1], generator=generator)
        with torch.no_grad():
            permuted = model(x[:, :, perm], carrier[:, perm])
        worst = max(worst, (permuted - reference).abs().max().item())
    return worst < PERMUTATION_TOL, {"max_abs_diff": worst, "tolerance": PERMUTATION_TOL}


def gate2_variable_n(model: nn.Module, seed: int = 0) -> tuple[bool, dict]:
    outputs: dict[int, torch.Tensor] = {}
    grad_ok = True
    for n in VARIABLE_N_SET:
        session = generate_session(seed=seed + n, num_units=n)
        x = session.counts.unsqueeze(0).requires_grad_(True)
        carrier = session.carrier.unsqueeze(0).requires_grad_(True)
        out = model(x, carrier)
        if not torch.isfinite(out).all():
            return False, {"failed_at": n, "reason": "non-finite output"}
        loss = (out**2).mean()
        grads = torch.autograd.grad(loss, [x, carrier], retain_graph=False)
        if not all(torch.isfinite(g).all() for g in grads):
            grad_ok = False
        outputs[n] = out.detach()
    variances = {n: float(o.var()) for n, o in outputs.items()}
    ratio = max(variances.values()) / max(min(variances.values()), 1e-12)
    passed = grad_ok and ratio < VARIABLE_N_VARIANCE_RATIO_MAX
    return passed, {"variances": variances, "ratio": ratio, "grads_finite": grad_ok}


def gate3_finite_live_gradients(model_name: str, seed: int = 0) -> tuple[bool, dict]:
    model = build_models(seed=7)[model_name]
    session = generate_session(seed=seed, num_units=96)
    x = session.counts.unsqueeze(0).requires_grad_(True)
    carrier = session.carrier.unsqueeze(0).requires_grad_(True)
    target = session.behaviour.unsqueeze(0)
    prediction = model(x, carrier)
    loss = nn.functional.mse_loss(prediction, target)
    loss.backward()

    finite = True
    for parameter in model.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            finite = False
    branches = BRANCH_PARAMS[model_name]
    activity_norm = _branch_grad_norm(model, branches["activity"])
    carrier_norm = _branch_grad_norm(model, branches["carrier"])
    input_activity_grad = float(x.grad.norm())
    input_carrier_grad = float(carrier.grad.norm())
    detail = {
        "all_params_have_finite_grad": finite,
        "activity_branch_param_grad_norm": activity_norm,
        "carrier_branch_param_grad_norm": carrier_norm,
        "activity_input_grad_norm": input_activity_grad,
        "carrier_input_grad_norm": input_carrier_grad,
    }
    passed = (
        finite
        and np.isfinite([activity_norm, carrier_norm, input_activity_grad, input_carrier_grad]).all()
        and activity_norm > 0.0
        and carrier_norm > 0.0
        and input_activity_grad > 0.0
        and input_carrier_grad > 0.0
    )
    return bool(passed), detail


def gate4_query_support_separation(model: nn.Module, seed: int = 0) -> tuple[bool, dict]:
    """Causality + support-only carrier on the ACTUAL noisy pipeline.

    (a) Causality: perturbing every bin after index q leaves outputs at t <= q
    bit-identical.  (Outputs at t legitimately depend on all bins <= t through
    the recurrent decoder; older-bin invariance is not the contract.)
    (b) The actual emitted carrier is query-independent: generating the same
    session with query counts perturbed before the carrier stage (same rng
    draw order) must reproduce the identical noisy carrier bit for bit.
    """
    session = generate_session(seed=seed, num_units=96)
    x = session.counts.unsqueeze(0)
    carrier = session.carrier.unsqueeze(0)
    q = x.shape[1] // 2
    with torch.no_grad():
        reference = model(x, carrier)
    perturbed = x.clone()
    perturbed[:, q + 1 :, :] += 1.0
    with torch.no_grad():
        after = model(perturbed, carrier)
    causality_diff = float((after[:, : q + 1, :] - reference[:, : q + 1, :]).abs().max())

    perturbed_session = generate_session(seed=seed, num_units=96, query_perturb=3.0)
    carrier_diff = float((perturbed_session.carrier - session.carrier).abs().max())
    counts_query_diff = float(
        (perturbed_session.counts[~session.support_mask] - session.counts[~session.support_mask]).abs().max()
    )

    passed = causality_diff == 0.0 and carrier_diff == 0.0 and counts_query_diff > 0.0
    return passed, {
        "causality_max_diff": causality_diff,
        "noisy_carrier_refit_max_diff": carrier_diff,
        "query_counts_actually_perturbed": counts_query_diff,
    }


def _arm_carrier(session: SyntheticSession, arm: str, seed: int) -> torch.Tensor:
    carrier = session.carrier
    if arm == "zero":
        return torch.zeros_like(carrier)
    if arm == "wrong_pair":
        generator = torch.Generator().manual_seed(seed)
        return carrier[torch.randperm(carrier.shape[0], generator=generator)]
    return carrier


def _query_support_split(session: SyntheticSession, window: int) -> tuple[slice, slice]:
    support_end = int(session.support_mask.sum())
    support_slice = slice(window, support_end)
    query_slice = slice(max(support_end, window), None)
    return support_slice, query_slice


def _train(model: nn.Module, sessions: list[SyntheticSession], arm: str, seed: int) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad()
        for session in sessions:
            x = session.counts.unsqueeze(0)
            carrier = _arm_carrier(session, arm, seed).unsqueeze(0)
            loss = nn.functional.mse_loss(model(x, carrier), session.behaviour.unsqueeze(0))
            loss.backward()
        optimizer.step()
    model.eval()


def _score(model: nn.Module, sessions: list[SyntheticSession], arm: str, seed: int) -> dict:
    window = getattr(model, "window_size")
    per_session = {"query": [], "support": []}
    with torch.no_grad():
        for session in sessions:
            carrier = _arm_carrier(session, arm, seed).unsqueeze(0)
            prediction = model(session.counts.unsqueeze(0), carrier)[0]
            support_slice, query_slice = _query_support_split(session, window)
            per_session["query"].append(r2_score(prediction[query_slice], session.behaviour[query_slice]))
            per_session["support"].append(r2_score(prediction[support_slice], session.behaviour[support_slice]))
    return {
        "query_mean": sum(per_session["query"]) / len(per_session["query"]),
        "support_mean": sum(per_session["support"]) / len(per_session["support"]),
        "query_per_session": per_session["query"],
        "support_per_session": per_session["support"],
    }


def gate5_attainable_carrier_effect(model_name: str, seed: int = 42) -> tuple[bool, dict]:
    train = [generate_session(seed=seed * 1000 + k) for k in range(TRAIN_SESSIONS)]
    eval_sessions = [generate_session(seed=900_000 + seed * 100 + k) for k in range(EVAL_SESSIONS)]

    # One initial state, deep-copied per arm: initialization cannot explain a
    # contrast between arms.
    initial = build_models(seed)[model_name]
    arms = {}
    for arm in ("aligned", "zero", "wrong_pair"):
        model = copy.deepcopy(initial)
        _train(model, train, arm, seed)
        arms[arm] = _score(model, eval_sessions, arm, seed)

    delta_zero = arms["aligned"]["query_mean"] - arms["zero"]["query_mean"]
    delta_wrong = arms["aligned"]["query_mean"] - arms["wrong_pair"]["query_mean"]
    detail = {
        "arms": arms,
        "delta_aligned_minus_zero_query": delta_zero,
        "delta_aligned_minus_wrong_pair_query": delta_wrong,
        "required_min_delta": CARRIER_EFFECT_MIN_DELTA,
        "headline": "query_mean",
    }
    return delta_zero >= CARRIER_EFFECT_MIN_DELTA and delta_wrong >= CARRIER_EFFECT_MIN_DELTA, detail


def run_all_gates(seed: int = 42, models: tuple[str, ...] | None = None) -> dict:
    results = {}
    available = build_models(seed)
    selected = available if models is None else {name: available[name] for name in models}
    for name, model in selected.items():
        results[name] = {
            "G1_permutation_invariance": gate1_permutation_invariance(model),
            "G2_variable_n": gate2_variable_n(model),
            "G3_finite_live_gradients": gate3_finite_live_gradients(name),
            "G4_query_support_separation": gate4_query_support_separation(model),
            "G5_attainable_carrier_effect": gate5_attainable_carrier_effect(name, seed),
        }
    return results
