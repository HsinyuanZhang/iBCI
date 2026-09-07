"""Source-only filter selection: F2 alpha CV, F3 simplex FIR fit, F4 logistic fit.

§6 discipline (all enforced here):

* outer unit = SESSION, never window; every fold is leave-one-session-out over
  the approved source sessions (the within-6 sub-C roster of this protocol);
* no external surface is ever read by any of these functions — callers pass
  only source streams and the functions have no other input;
* the fitting loss is the §6.2 session-balanced normalized SSE;
* optimizers are deterministic (fixed init, finite-difference gradients, no
  RNG); every chosen parameter payload is hashed into the receipts.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from . import ladder, metrics, plan


class SelectionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionError(message)


# ---------------------------------------------------------------------------
# F2: fixed-alpha EMA selected on the frozen grid by session-grouped CV.
# ---------------------------------------------------------------------------


def select_alpha(
    streams: Sequence[ladder.SessionStream],
    *, grid: Sequence[float] = plan.ALPHA_GRID,
) -> dict[str, Any]:
    """LOSO over source sessions: each fold's alpha is chosen by the others."""
    sessions = [stream.session for stream in streams]
    _require(len(set(sessions)) == len(sessions) >= 2, "source grid CV needs distinct sessions")
    per_session_r2: dict[str, dict[float, float]] = {}
    for stream in streams:
        per_session_r2[stream.session] = {
            float(alpha): metrics.matrix_r2(ladder.apply_filter(stream, ladder.f2(alpha)).blocks, stream)
            for alpha in grid
        }
    folds: list[dict[str, Any]] = []
    oof_r2: list[float] = []
    for index, held in enumerate(streams):
        others = [item for item in streams if item.session != held.session]
        chosen = max(
            grid,
            key=lambda alpha: metrics.equal_session_mean(
                [per_session_r2[item.session][float(alpha)] for item in others]
            ),
        )
        held_r2 = per_session_r2[held.session][float(chosen)]
        oof_r2.append(held_r2)
        folds.append({
            "fold": index, "held_out_session": held.session,
            "chosen_alpha": float(chosen),
            "held_out_r2_at_chosen_alpha": held_r2,
            "held_out_r2_at_all_grid_points": {
                f"{float(alpha):.2f}": per_session_r2[held.session][float(alpha)] for alpha in grid
            },
        })
    oof_mean = metrics.equal_session_mean(oof_r2)
    winner = max(grid, key=lambda alpha: metrics.equal_session_mean(
        [per_session_r2[item.session][float(alpha)] for item in streams]
    ))
    return {
        "level": "F2",
        "grid": [float(item) for item in grid],
        "selection": "session-grouped LOSO on source sessions only",
        "folds": folds,
        "oof_equal_session_mean_r2": oof_mean,
        "full_source_mean_r2_by_alpha": {
            f"{float(alpha):.2f}": metrics.equal_session_mean(
                [per_session_r2[item.session][float(alpha)] for item in streams]
            ) for alpha in grid
        },
        "pooled_alpha": float(winner),
        "payload": {"level": "F2", "alpha": float(winner)},
    }


# ---------------------------------------------------------------------------
# F3: source-learned FIR-K4 on the softmax simplex.
# ---------------------------------------------------------------------------


def _simplex_from_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def fit_fir(streams: Sequence[ladder.SessionStream], *, maxiter: int | None = None) -> dict[str, Any]:
    """Fit the K4 FIR simplex weights by minimizing the §6.2 loss."""
    from scipy.optimize import minimize

    maxiter = int(maxiter or plan.FIR_OPTIMIZER["maxiter"])

    def objective(logits: np.ndarray) -> float:
        weights = _simplex_from_logits(np.asarray(logits, dtype=np.float64))
        return metrics.balanced_nsse(ladder.f3(weights), streams)

    result = minimize(
        objective, np.zeros(plan.FIR_TAPS, dtype=np.float64), method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10},
    )
    weights = _simplex_from_logits(np.asarray(result.x, dtype=np.float64))
    return {
        "weights": [float(item) for item in weights],
        "loss": float(result.fun),
        "n_iterations": int(result.nit),
        "converged": bool(result.success),
        "message": str(result.message),
    }


def cv_fir(streams: Sequence[ladder.SessionStream]) -> dict[str, Any]:
    """LOSO fit-and-score of the F3 filter on the source sessions."""
    folds: list[dict[str, Any]] = []
    oof_r2: list[float] = []
    for index, held in enumerate(streams):
        others = [item for item in streams if item.session != held.session]
        fit = fit_fir(others)
        spec = ladder.f3(fit["weights"])
        held_r2 = metrics.matrix_r2(ladder.apply_filter(held, spec).blocks, held)
        oof_r2.append(held_r2)
        folds.append({
            "fold": index, "held_out_session": held.session,
            "fit_sessions": [item.session for item in others],
            "fitted_weights": fit["weights"],
            "fit_loss": fit["loss"],
            "converged": fit["converged"],
            "held_out_r2": held_r2,
        })
    pooled = fit_fir(streams)
    return {
        "level": "F3",
        "folds": folds,
        "oof_equal_session_mean_r2": metrics.equal_session_mean(oof_r2),
        "pooled_fit": pooled,
        "payload": {"level": "F3", "weights": pooled["weights"]},
    }


# ---------------------------------------------------------------------------
# F4: adaptive scalar gain, features only from the §4 whitelist.
# ---------------------------------------------------------------------------


def _f4_batches(
    streams: Sequence[ladder.SessionStream],
) -> list[tuple[ladder.BatchView, float]]:
    """Padded batch + session SST per stream (SST precomputed once)."""
    return [
        (ladder.BatchView.from_stream(stream), metrics.session_sst(stream)) for stream in streams
    ]


def _f4_objective_and_grad(
    parameters: np.ndarray, batches: Sequence[tuple[ladder.BatchView, float]],
) -> tuple[float, np.ndarray]:
    """§6.2 balanced loss and its exact gradient (analytic adjoint recursion)."""
    total = 0.0
    gradient = np.zeros_like(np.asarray(parameters, dtype=np.float64))
    for batch, sst in batches:
        loss, grad = ladder.f4_loss_and_grad(np.asarray(parameters, dtype=np.float64), batch, sst=sst)
        total += loss
        gradient += grad
    n = len(batches)
    return total / n, gradient / n


def _f4_objective(parameters: np.ndarray, streams: Sequence[ladder.SessionStream]) -> float:
    spec = ladder.f4(parameters[:-1], float(parameters[-1]))
    return metrics.balanced_nsse(spec, streams)


def fit_f4(streams: Sequence[ladder.SessionStream], *, maxiter: int | None = None) -> dict[str, Any]:
    from scipy.optimize import minimize

    maxiter = int(maxiter or plan.F4_OPTIMIZER["maxiter"])
    batches = _f4_batches(streams)
    start = np.zeros(plan.F4_PARAM_COUNT, dtype=np.float64)
    result = minimize(
        _f4_objective_and_grad, start, args=(batches,), method="L-BFGS-B", jac=True,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10},
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    # the reported loss is recomputed through the same objective (deterministic)
    final_loss = _f4_objective_and_grad(parameters, batches)[0]
    return {
        "theta": [float(item) for item in parameters[:-1]],
        "bias": float(parameters[-1]),
        "feature_names": list(plan.F4_FEATURES),
        "loss": float(final_loss),
        "n_iterations": int(result.nit),
        "converged": bool(result.success),
        "message": str(result.message),
        "param_count": int(parameters.size),
        "gradient": "analytic adjoint recursion (unit-tested against finite differences)",
    }


def cv_f4(
    streams: Sequence[ladder.SessionStream],
    *, reference: ladder.FilterSpec,
) -> dict[str, Any]:
    """LOSO fit-and-score of F4; primary readout = OOF gain over ``reference``."""
    folds: list[dict[str, Any]] = []
    oof_f4: list[float] = []
    oof_reference: list[float] = []
    oof_raw: list[float] = []
    for index, held in enumerate(streams):
        others = [item for item in streams if item.session != held.session]
        fit = fit_f4(others)
        spec = ladder.f4(fit["theta"], fit["bias"])
        held_result = ladder.apply_filter(held, spec)
        held_f4 = metrics.matrix_r2(held_result.blocks, held)
        held_reference = metrics.matrix_r2(ladder.apply_filter(held, reference).blocks, held)
        held_raw = metrics.matrix_r2(ladder.apply_filter(held, ladder.F0).blocks, held)
        gains = held_result.gains
        active = gains > 0.0
        oof_f4.append(held_f4)
        oof_reference.append(held_reference)
        oof_raw.append(held_raw)
        folds.append({
            "fold": index, "held_out_session": held.session,
            "fit_sessions": [item.session for item in others],
            "fitted_theta": fit["theta"], "fitted_bias": fit["bias"],
            "converged": fit["converged"], "fit_loss": fit["loss"],
            "held_out_r2_f4": held_f4,
            "held_out_r2_reference": held_reference,
            "held_out_r2_raw": held_raw,
            "held_out_gain_mean": float(gains[active].mean()) if bool(active.any()) else None,
            "held_out_gain_max": float(gains.max()),
        })
    pooled = fit_f4(streams)
    return {
        "level": "F4",
        "reference_filter": reference.payload(),
        "folds": folds,
        "oof_equal_session_mean_r2_f4": metrics.equal_session_mean(oof_f4),
        "oof_equal_session_mean_r2_reference": metrics.equal_session_mean(oof_reference),
        "oof_equal_session_mean_r2_raw": metrics.equal_session_mean(oof_raw),
        "oof_gain_over_reference": metrics.equal_session_mean(oof_f4) - metrics.equal_session_mean(oof_reference),
        "oof_gain_over_reference_paired": metrics.paired_session_deltas(
            oof_f4, oof_reference, label="F4_minus_selected_fixed_source_oof",
        ),
        "within_recovery_vs_raw_paired": metrics.paired_session_deltas(
            oof_f4, oof_raw, label="F4_minus_F0_source_oof",
        ),
        "within_recovery_vs_reference_paired": metrics.paired_session_deltas(
            oof_f4, oof_reference, label="F4_minus_selected_fixed_source_oof_paired",
        ),
        "pooled_fit": pooled,
        "payload": {"level": "F4", "theta": pooled["theta"], "bias": pooled["bias"]},
    }


# ---------------------------------------------------------------------------
# The §10.2 fixed-filter selection gate.
# ---------------------------------------------------------------------------


def select_fixed_filter(
    *, f1_oof: float, f2_oof: float, fir_oof: float, fir_payload: dict[str, Any],
    best_f2_payload: dict[str, Any],
) -> dict[str, Any]:
    """At most one fixed filter advances; F3 needs >= +0.005 over best F1/F2."""
    best_simple = max(
        (("F1", f1_oof), ("F2", f2_oof)), key=lambda item: item[1],
    )
    margin = fir_oof - float(best_simple[1])
    gate = margin >= plan.FIR_GATE_OVER_F1_F2
    if gate:
        selected, payload = "F3", fir_payload
    else:
        selected, payload = best_simple[0], (
            {"level": "F1"} if best_simple[0] == "F1" else best_f2_payload
        )
    return {
        "f1_oof": float(f1_oof),
        "f2_oof": float(f2_oof),
        "f3_oof": float(fir_oof),
        "best_nonlearned": {"level": best_simple[0], "oof": float(best_simple[1])},
        "f3_margin_over_best_nonlearned": float(margin),
        "gate_threshold": plan.FIR_GATE_OVER_F1_F2,
        "gate_passed": bool(gate),
        "selected_level": selected,
        "selected_payload": payload,
        "selected_payload_sha256": hashlib.sha256(
            ladder.canonical_json(payload).encode("utf-8")
        ).hexdigest(),
        "rule": (
            "F3 advances only if it beats the best nonlearned F1/F2 by at least "
            "+0.005 source-grouped OOF (§10.2); otherwise the simpler filter is retained"
        ),
    }


def oracle_disposition(*, oracle_minus_fixed: float, surface: str, budget: int) -> dict[str, Any]:
    """§8.3 disposition of the coherent adaptive oracle over the best fixed."""
    value = float(oracle_minus_fixed)
    if value < plan.ORACLE_STOP_BELOW:
        disposition = "STOP_ADAPTIVE_LEARNING"
        reading = "fixed filter is sufficient"
    elif value < plan.ORACLE_PROCEED_ABOVE:
        disposition = "CONDITIONAL_F4"
        reading = "fit F4 only if grouped source predictability is stable"
    else:
        disposition = "PROCEED_F4"
        reading = "adaptive gain has material headroom"
    return {
        "surface": surface, "budget": int(budget),
        "coherent_oracle_minus_best_fixed": value,
        "stop_below": plan.ORACLE_STOP_BELOW,
        "proceed_above": plan.ORACLE_PROCEED_ABOVE,
        "disposition": disposition,
        "reading": reading,
    }
