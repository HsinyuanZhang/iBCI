"""Deliverable A: zero-GPU pre-gate for the time-varying weighting premise.

Implements HANDOFF_POST_BILINEAR_CONSUMER_EXTENSION_20260815.md §5.5
(respecified pre-gate; the first draft's alignment-oracle kill was withdrawn
because it leaks the target and points the wrong way for a Poisson cosine
unit, whose local Fisher information about movement direction is
``m^2 sin^2(theta - theta_pref) / lambda`` — maximal +/-90 degrees from the
preferred direction, minimal at alignment).

Pre-registered estimator family — shared value pipeline and shared readout
head, ONLY the weights differ:

Shared pipeline (identical for every estimator; carrier + counts only, no
true target):
  activity  ``a_i(t)``   causal trailing-window mean of spike counts (for
             the prior read); residuals use the instantaneous count;
  value map ``u_hat_i, m_hat_i, b_hat_i`` from the noisy T4-style carrier;
  prior     ``y0_hat``   the shared model's own causal position estimate:
             ridge head on the lagged UNIFORM-weight population read,
             fitted on support bins (this is the ``y_hat`` the handoff's
             causal-evaluation clause refers to; a direction-space prior
             cannot be locally linearized here because the true direction
             rotates ~0.6 rad/bin, while position is smooth);
  evidence  Poisson-cosine observation model at the causal prior
             ``y_prev = y0_hat_{t-1}``:
             ``lambda_hat_i(y) = softplus(b_hat_i + m_hat_i <u_hat_i, y>/|y|) + floor``
             with gradient ``g_i(y) = d lambda_hat_i / d y`` and residual
             ``z_i(t) = n_i(t) - lambda_hat_i(y_prev)``.
Weighted channel (the ONLY weight-dependent part):
             ``delta_w(t) = sum_i w_i(t) g_i(y_prev) z_i(t) / (sum_i w_i(t) + eps)``
             (2-D) plus ``log1p(sum_i w_i(t))``.  Final prediction =
             ``y0_hat + H(corr_w)`` with the same anchored residual ridge H
             fitted on support for every arm.

Estimators (only ``w`` differs):
1. ``static`` : ``w_i = softplus(gamma . features(c_i))``, features
   ``[1, m_hat, b_hat, m_hat^2, m_hat^2 / b_hat]`` — a non-negative ridge on
   source (the time-averaged Fisher precision ``m^2 / (2b)`` is inside the
   basis, so the static arm is NOT a strawman), fitted by one-step
   regression on support bins, then frozen.
2. ``fisher`` : time-varying Fisher/slope weights at the model's own causal
   prior, ``w_i(t) = |g_i(y_prev)|^2 / lambda_hat_i(y_prev)`` — up to the
   shared ``softplus'^2 / |y|^2`` factor exactly the handoff's
   ``m^2 sin^2(theta - theta_pref) / lambda``.  No fitting, no true target.
3. ``align``  : LEAKED UPPER BOUND ONLY.  ``w_i(t) = cos^2(phi_t - theta_hat_i)``
   with the true instantaneous direction.  Reported separately, never a
   kill.

PASS/FAIL CONTRACT (frozen; the gate is non-vacuous — each construction can
pass and fail):

- construction (i)  isotropic cosine population — the sealed
  `src.tfpd/synth.py` generator (unit directions uniform, every unit
  informative at all times, broad cosine tuning):
  PASS requires ``R2_fisher - R2_static >= +0.05`` on query bins AND
  ``R2_align - R2_static < +0.05``.
- construction (ii) state-matched population — narrow von-Mises-like tuning
  (kappa = 0.4) with high baselines: at any bin only units whose preferred
  direction is near the current movement direction carry signal, the rest
  are baseline noise:
  PASS requires ``R2_align - R2_static >= +0.05``.

Overall PASS requires both constructions to pass.  Failing construction (i)
means the time-varying premise is not supported on the population class it
is claimed for; failing construction (ii) would mean the synthetic pair
cannot discriminate the weighting schemes (vacuous gate).  Per §5.5-5 this
pre-gate is never a sole kill for the lane.

Design disclosures (frozen into this file because they shaped the family;
all measured during development, before this contract was frozen):
- a weighted-PV read family inverts the expected pattern: aligned units'
  Poisson noise is radial and harmless for a read direction, so the leaked
  alignment oracle WINS on construction (i) there (+0.18 R2) while Fisher
  ties static — the handoff's flank-information argument only holds for a
  LOCAL estimator, hence the residual/gradient family above;
- an oracle Fisher weighting evaluated at the TRUE state gains nothing over
  static on construction (i) (0.667 vs 0.675 R2): the static carrier ridge
  can represent the time-averaged precision of a broad isotropic cosine
  population, so the time-varying premise is second-order there by
  construction, consistent with the prior-evidence caution in handoff §4.3;
- the premise IS first-order when instantaneous informativeness is sparse
  (construction (ii) regime), where the static arm collapses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.tfpd.synth import (
    SyntheticSession,
    _smooth_trajectory,
    fit_carrier_support,
    generate_session,
    r2_score,
)

# ---- frozen hyperparameters -------------------------------------------------

WINDOW = 8  # trailing bins for the prior read's causal activity
LAGS = 6  # prior head input: read channels at lags 0..LAGS-1
HEAD_RIDGE = 1.0  # ridge for the prior head and the anchored residual head
STATIC_FIT_STEPS = 300
STATIC_FIT_LR = 0.05
STATIC_FIT_RIDGE = 0.1
EPS = 1e-6
RATE_FLOOR = 0.05

LENGTH = 512
NUM_UNITS_ISOTROPIC = 64
NUM_UNITS_STATE_MATCHED = 96
SUPPORT_FRACTION = 0.3
CARRIER_NOISE = 0.10
STATE_MATCHED_KAPPA = 0.40
STATE_MATCHED_BASELINE = (1.0, 3.0)

DELTA_MIN = 0.05

ESTIMATORS = ("static", "fisher", "align")


# ---- construction (ii): state-matched population ----------------------------


def generate_state_matched_session(
    seed: int,
    length: int = LENGTH,
    num_units: int = NUM_UNITS_STATE_MATCHED,
    support_fraction: float = SUPPORT_FRACTION,
    carrier_noise: float = CARRIER_NOISE,
    kappa: float = STATE_MATCHED_KAPPA,
    baseline_range: tuple[float, float] = STATE_MATCHED_BASELINE,
) -> SyntheticSession:
    """Narrow-tuning population: only state-matched units carry signal.

    Rates are ``lambda_i(t) = softplus(b_i + g_i * exp(-(1 - cos(phi_t -
    theta_i)) / kappa)) + 0.05`` with HIGH baselines ``b_i ~ U[1, 3]``.  The
    modulation is confined to an arc around each unit's preferred direction,
    so at any time bin only units whose preferred direction is near the
    current movement direction are elevated; the many remaining units fire
    at a noisy baseline.  The carrier pipeline (closed-form T4-style support
    fit + noise) is identical to `src.tfpd.synth.generate_session`.
    """
    rng = np.random.default_rng(seed)
    n = num_units

    theta = rng.uniform(0, 2 * np.pi, size=n)
    gain = rng.uniform(0.5, 2.0, size=n)
    baseline = rng.uniform(baseline_range[0], baseline_range[1], size=n)

    traj = _smooth_trajectory(length, rng)  # [T, 2]
    direction = traj / np.clip(np.linalg.norm(traj, axis=1, keepdims=True), 1e-8, None)
    phi = np.arctan2(direction[:, 1], direction[:, 0])  # [T]

    bump = np.exp(-(1.0 - np.cos(phi[:, None] - theta[None, :])) / kappa)
    modulation = baseline[None, :] + gain[None, :] * bump
    rate = np.logaddexp(0.0, modulation) + RATE_FLOOR
    if not np.all(np.isfinite(rate)) or np.any(rate <= 0.0):
        raise AssertionError("state-matched lambda positivity violated")
    counts = rng.poisson(rate).astype(np.float32)

    support_end = int(length * support_fraction)
    support_mask = np.zeros(length, dtype=bool)
    support_mask[:support_end] = True

    a_hat, c_hat, m_hat, b_hat = fit_carrier_support(counts, phi, support_mask)
    noise = rng.normal(0.0, carrier_noise, size=(3, n)) * np.stack(
        [a_hat + 0.5, c_hat + 0.5, b_hat + 0.5]
    )
    a_noisy = a_hat + noise[0]
    c_noisy = c_hat + noise[1]
    b_noisy = b_hat + noise[2]
    m_noisy = np.sqrt(a_noisy**2 + c_noisy**2)
    carrier = np.stack([a_noisy, c_noisy, m_noisy, b_noisy], axis=1)

    return SyntheticSession(
        counts=torch.from_numpy(counts),
        behaviour=torch.from_numpy(traj.astype(np.float32)),
        carrier=torch.from_numpy(carrier.astype(np.float32)),
        support_mask=torch.from_numpy(support_mask),
    )


# ---- shared pipeline ---------------------------------------------------------


def trailing_mean(x: torch.Tensor, window: int = WINDOW) -> torch.Tensor:
    """Causal a_i(t) = mean(x_i[max(0, t-window+1) : t+1]); [T,N] -> [T,N]."""
    t_len, n = x.shape
    csum = torch.cat([x.new_zeros(1, n), x.cumsum(dim=0)], dim=0)
    idx = torch.arange(t_len)
    lo = (idx + 1 - window).clamp_min(0)
    counts = (idx + 1 - lo).to(x.dtype).unsqueeze(1)
    return (csum[idx + 1] - csum[lo]) / counts


def value_map(carrier: torch.Tensor) -> torch.Tensor:
    """Shared frozen value map: value_i = [a_hat_i, c_hat_i]; [N,4] -> [N,2]."""
    return carrier[:, 0:2]


def population_read(activity: torch.Tensor, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted population read; returns [T, 3] = [r_x, r_y, log1p(mass)]."""
    weighted = activity * weights  # [T, N]
    denom = weighted.sum(dim=1, keepdim=True) + EPS
    r = torch.einsum("tn,nd->td", weighted, values) / denom
    mass = torch.log1p(weighted.sum(dim=1, keepdim=True))
    return torch.cat([r, mass], dim=1)


def lag_stack(channels: torch.Tensor, lags: int = LAGS) -> torch.Tensor:
    """Causal lag stack; [T, d] -> [T, d*lags] (front zero-padded)."""
    t_len, d = channels.shape
    pad = channels.new_zeros(lags - 1, d)
    stacked = torch.cat([pad, channels], dim=0)
    windows = stacked.unfold(dimension=0, size=lags, step=1)  # [T, d, lags]
    return windows.permute(0, 2, 1).reshape(t_len, d * lags)


def rate_and_gradient(carrier: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Poisson cosine rate and gradient at positions ``y``; -> ([T,N], [T,N,2]).

    ``lambda_hat_i(y) = softplus(b_hat_i + m_hat_i <u_hat_i, y> / |y|) + floor``
    ``g_i(y)          = softplus'(net) m_hat_i (u_hat_i - cos(Delta) y_hat) / |y|``
    with norm ``softplus'(net) m_hat_i |sin Delta| / |y|`` — the cosine
    tuning slope at the causal position estimate.
    """
    a_c = carrier[:, 0:2]
    m_hat = carrier[:, 2].clamp_min(0.0)
    b_hat = carrier[:, 3]
    u_hat = a_c / a_c.norm(dim=1, keepdim=True).clamp_min(EPS)  # [N, 2]

    radius = y.norm(dim=1, keepdim=True).clamp_min(EPS)  # [T, 1]
    y_dir = y / radius
    cos_delta = y_dir @ u_hat.T  # [T, N]
    net = b_hat.unsqueeze(0) + m_hat.unsqueeze(0) * cos_delta
    rate = torch.nn.functional.softplus(net) + RATE_FLOOR
    dsigmoid = torch.sigmoid(net)
    gradient = (
        dsigmoid.unsqueeze(-1)
        * m_hat.unsqueeze(0).unsqueeze(-1)
        * (u_hat.unsqueeze(0) - cos_delta.unsqueeze(-1) * y_dir.unsqueeze(1))
        / radius.unsqueeze(-1)
    )
    return rate, gradient


class RidgeHead:
    """Linear ridge head fitted on support rows, frozen thereafter."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor, lam: float = HEAD_RIDGE) -> None:
        self.x_mean = features.mean(dim=0, keepdim=True)
        self.x_std = features.std(dim=0, keepdim=True).clamp_min(1e-8)
        self.y_mean = targets.mean(dim=0, keepdim=True)
        xs = (features - self.x_mean) / self.x_std
        design = torch.cat([xs, features.new_ones(len(xs), 1)], dim=1)
        gram = design.T @ design + lam * torch.eye(design.shape[1], dtype=design.dtype)
        gram[-1, -1] -= lam  # intercept unpenalized
        self.beta = torch.linalg.solve(gram, design.T @ (targets - self.y_mean))

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        xs = (features - self.x_mean) / self.x_std
        design = torch.cat([xs, features.new_ones(len(xs), 1)], dim=1)
        return design @ self.beta + self.y_mean


def true_direction(behaviour: torch.Tensor) -> torch.Tensor:
    """Direction of the true position vector y_t (this is the leak channel)."""
    return torch.atan2(behaviour[:, 1], behaviour[:, 0])


# ---- weights -----------------------------------------------------------------


def static_weight_features(carrier: torch.Tensor) -> torch.Tensor:
    """Carrier-side basis for the static non-negative ridge.

    Includes the theoretical time-averaged Fisher precision ``m^2 / b`` so
    the static arm can represent the best static approximation of the
    Fisher weighting — it is not a strawman.
    """
    m_hat = carrier[:, 2].clamp_min(0.0)
    b_hat = carrier[:, 3]
    return torch.stack(
        [
            torch.ones_like(m_hat),
            m_hat,
            b_hat,
            m_hat**2,
            m_hat**2 / (b_hat + RATE_FLOOR),
        ],
        dim=1,
    )


def fit_static_weights(
    counts: torch.Tensor,
    carrier: torch.Tensor,
    behaviour: torch.Tensor,
    support_rows: torch.Tensor,
    steps: int = STATIC_FIT_STEPS,
    lr: float = STATIC_FIT_LR,
    lam: float = STATIC_FIT_RIDGE,
) -> torch.Tensor:
    """Non-negative static ridge weights, fitted on support bins only.

    One-step regression: with ``y_prev`` the TRUE previous-bin position
    (source calibration only), the ideal weighted correction satisfies
    ``delta_w(y_prev) ~= beta * (y_t - y_{t-1})``; gamma and beta are fitted
    to that relation on support rows.  Query bins and query targets never
    enter.  Frozen after return.
    """
    y_prev = torch.cat([behaviour[:1], behaviour[:-1]], dim=0)
    rate, grad = rate_and_gradient(carrier, y_prev)
    resid = counts - rate
    target = behaviour - y_prev  # one-step displacement
    feats = static_weight_features(carrier)
    gamma = torch.zeros(feats.shape[1], requires_grad=True)
    beta = torch.zeros(1, 2, requires_grad=True)
    optimizer = torch.optim.Adam([gamma, beta], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        w = torch.nn.functional.softplus(feats @ gamma)  # [N]
        corr = torch.einsum("n,tn,tni->ti", w, resid[support_rows], grad[support_rows]) / (w.sum() + EPS)
        loss = ((corr - target[support_rows] * beta) ** 2).sum() + lam * (gamma**2).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.nn.functional.softplus(feats @ gamma)


def fisher_weights(rate: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Time-varying Fisher/slope weights at the causal prior; -> [T, N].

    ``w_i(t) = |g_i(y_hat_{t-1})|^2 / lambda_hat_i(y_hat_{t-1})`` — the
    per-unit local Fisher information of the Poisson cosine observation
    model evaluated at the model's own causal position estimate (up to the
    shared ``softplus'^2 / |y|^2`` factor exactly the handoff's
    ``m^2 sin^2(theta - theta_pref) / lambda``).  Never touches the true y_t.
    """
    return (gradient**2).sum(dim=-1) / rate


def alignment_weights(carrier: torch.Tensor, phi_true: torch.Tensor) -> torch.Tensor:
    """LEAKED ORACLE (upper bound only, never a kill): cos^2(true phi_t - theta_hat_i)."""
    a_c = carrier[:, 0:2]
    theta_hat = torch.atan2(a_c[:, 1], a_c[:, 0])
    delta = phi_true.unsqueeze(1) - theta_hat.unsqueeze(0)
    return torch.cos(delta) ** 2


def correction_channel(
    counts: torch.Tensor, rate: torch.Tensor, gradient: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Weighted 2-D correction + population-information channels; -> [T, 3]."""
    resid = counts - rate
    wsum = weights.sum(dim=1, keepdim=True) + EPS
    delta = torch.einsum("tn,tni->ti", weights * resid, gradient) / wsum
    log_j = torch.log1p(weights.sum(dim=1, keepdim=True))
    return torch.cat([delta, log_j], dim=1)


# ---- session pipeline ---------------------------------------------------------


def _row_masks(session: SyntheticSession) -> tuple[torch.Tensor, torch.Tensor]:
    support_end = int(session.support_mask.sum().item())
    all_rows = torch.arange(session.counts.shape[0])
    support_rows = all_rows[(all_rows >= LAGS) & (all_rows < support_end)]
    query_rows = all_rows[all_rows >= max(support_end, LAGS)]
    return support_rows, query_rows


@dataclass
class SessionResult:
    query_r2: dict[str, float]
    support_r2: dict[str, float]
    predictions: dict[str, torch.Tensor]


def run_session(session: SyntheticSession) -> SessionResult:
    """Fit all three estimators on support, freeze, score query.

    Shared two-stage pipeline: a prior head on the lagged uniform read
    (support-fitted) provides the causal position estimate; the Poisson
    cosine observation model is evaluated at ``y0_hat_{t-1}``; the weighted
    correction channel is the only arm-dependent input, consumed by the
    same anchored residual-ridge readout.  The true target ``y_t`` never
    enters any weight (the static arm's support-only one-step regression is
    source calibration, frozen before query scoring).
    """
    counts = session.counts.float()
    behaviour = session.behaviour.float()
    carrier = session.carrier.float()
    support_rows, query_rows = _row_masks(session)

    # stage A: shared causal prior from the uniform read (support-fitted).
    activity = trailing_mean(counts)
    prior_read = population_read(activity, value_map(carrier), torch.ones_like(activity))
    prior_features = lag_stack(prior_read)
    prior_head = RidgeHead(prior_features[support_rows], behaviour[support_rows])
    with torch.no_grad():
        y0 = prior_head.predict(prior_features)
    y_prev = torch.cat([y0[:1], y0[:-1]], dim=0)  # y0_hat_{t-1}

    # stage B: shared evidence at the causal prior.
    rate, gradient = rate_and_gradient(carrier, y_prev)

    # arm weights; static fitted on support only, frozen.
    w_static = fit_static_weights(counts, carrier, behaviour, support_rows)
    weights = {
        "static": w_static.unsqueeze(0).expand(counts.shape[0], -1),
        "fisher": fisher_weights(rate, gradient),
        "align": alignment_weights(carrier, true_direction(behaviour)),
    }

    predictions = {}
    for name, w in weights.items():
        corr = correction_channel(counts, rate, gradient, w)
        residual_head = RidgeHead(corr[support_rows], (behaviour - y0)[support_rows])
        with torch.no_grad():
            predictions[name] = y0 + residual_head.predict(corr)

    query_r2 = {name: r2_score(pred[query_rows], behaviour[query_rows]) for name, pred in predictions.items()}
    support_r2 = {name: r2_score(pred[support_rows], behaviour[support_rows]) for name, pred in predictions.items()}
    return SessionResult(query_r2=query_r2, support_r2=support_r2, predictions=predictions)


# ---- gate --------------------------------------------------------------------


def run_construction(construction: str, seed: int, num_sessions: int) -> dict:
    if construction == "isotropic_cosine":
        sessions = [
            generate_session(seed=seed * 1000 + k, length=LENGTH, num_units=NUM_UNITS_ISOTROPIC)
            for k in range(num_sessions)
        ]
    elif construction == "state_matched":
        sessions = [
            generate_state_matched_session(seed=seed * 1000 + k, length=LENGTH)
            for k in range(num_sessions)
        ]
    else:
        raise ValueError(f"unknown construction {construction}")

    per_session = [run_session(s) for s in sessions]
    means = {
        name: sum(r.query_r2[name] for r in per_session) / num_sessions for name in ESTIMATORS
    }
    deltas = {
        "fisher_minus_static": means["fisher"] - means["static"],
        "align_minus_static": means["align"] - means["static"],
    }
    if construction == "isotropic_cosine":
        passed = (
            deltas["fisher_minus_static"] >= DELTA_MIN
            and deltas["align_minus_static"] < DELTA_MIN
        )
        rule = (
            "pass requires fisher - static >= +0.05 AND align - static < +0.05 "
            "(Fisher time-varying beats static; leaked alignment oracle weak or null)"
        )
    else:
        passed = deltas["align_minus_static"] >= DELTA_MIN
        rule = (
            "pass requires align - static >= +0.05 "
            "(leaked alignment oracle must beat static: only state-matched units are informative)"
        )
    return {
        "construction": construction,
        "num_sessions": num_sessions,
        "query_r2_mean": means,
        "query_r2_per_session": {name: [r.query_r2[name] for r in per_session] for name in ESTIMATORS},
        "support_r2_mean": {name: sum(r.support_r2[name] for r in per_session) / num_sessions for name in ESTIMATORS},
        "deltas": deltas,
        "delta_min": DELTA_MIN,
        "rule": rule,
        "passed": bool(passed),
    }


def run_pregate(seed: int = 42, num_sessions: int = 12) -> dict:
    """Both constructions; overall pass requires both construction passes."""
    constructions = [run_construction(name, seed, num_sessions) for name in ("isotropic_cosine", "state_matched")]
    all_passed = all(c["passed"] for c in constructions)
    return {
        "schema": "tfpd_pregate_timevarying_v1",
        "seed": seed,
        "hyperparameters": {
            "window": WINDOW, "lags": LAGS, "head_ridge": HEAD_RIDGE,
            "static_fit_steps": STATIC_FIT_STEPS, "static_fit_ridge": STATIC_FIT_RIDGE,
            "length": LENGTH, "num_units_isotropic": NUM_UNITS_ISOTROPIC,
            "num_units_state_matched": NUM_UNITS_STATE_MATCHED,
            "state_matched_kappa": STATE_MATCHED_KAPPA,
            "state_matched_baseline": list(STATE_MATCHED_BASELINE),
        },
        "constructions": {c["construction"]: c for c in constructions},
        "status": "PASS" if all_passed else "FAIL",
        "note": (
            "align is a leaked upper bound (true y inside the weights) and is never a kill; "
            "per handoff 5.5 this pre-gate is not a sole kill for the lane"
        ),
    }
