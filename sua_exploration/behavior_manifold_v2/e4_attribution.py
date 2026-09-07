"""E4: per-output manifold-loading attribution.

Two per-output quantities are compared across the 16 EMG channels:

  delta_o   = R2_o(q8 ensemble route) - R2_o(DirectRidge), each pooled across
              the four sessions as 1 - sum_sessions(SSE)/sum_sessions(TSS)
              (the combine.py sufficient-statistic convention), on the frozen
              M10 deployment comparison;
  loading_o = R2_o(dec(enc(y))) of the frozen decoder on each session's full
              behaviour, three-seed mean, pooled across sessions the same way.

Prediction: delta grows with loading (gains concentrate where the manifold
reconstructs the output well) and PECmaj (known mean-negative, -0.016) sits at
the low-loading end.  Reported as a Pearson correlation with a t-test p-value
and the full per-output table.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sua_exploration.behavior_autoencoder_v1.core import need
from sua_exploration.h1_m1_priority_v1.core import array_sha256, regression_metrics

from .frozen import FrozenDeployment
from .protocol import full_session_arrays


def _pooled_per_output(metrics_per_session: list[dict[str, Any]]) -> np.ndarray:
    sse = np.sum([np.asarray(m["sse_per_output"], dtype=np.float64) for m in metrics_per_session], axis=0)
    tss = np.sum([np.asarray(m["tss_per_output"], dtype=np.float64) for m in metrics_per_session], axis=0)
    return 1.0 - sse / tss


def _pearson(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson r with the exact two-sided t-test p-value (incomplete beta)."""

    from scipy.special import betainc as regularized_incomplete_beta

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    n = int(x.size)
    need(n >= 3, "attribution correlation needs at least three outputs")
    need(float((xm * xm).sum()) > 0.0 and float((ym * ym).sum()) > 0.0, "attribution correlation is degenerate (zero variance)")
    r = float((xm * ym).sum() / np.sqrt(float((xm * xm).sum()) * float((ym * ym).sum())))
    need(abs(r) < 1.0, "attribution correlation is degenerate (|r| = 1)")
    t = float(r * np.sqrt((n - 2) / (1.0 - r * r)))
    p = float(regularized_incomplete_beta((n - 2) / 2.0, 0.5, (n - 2) / ((n - 2) + t * t)))
    return {"pearson_r": r, "t_statistic": t, "p_value_two_sided": p, "n": n}


def experiment_e4(frozen: FrozenDeployment, *, sessions) -> dict[str, Any]:
    names = frozen.names
    output_names = list(sessions[names[0]].output_names)
    # Family per-output pooled R2 on the frozen M10 deployment comparison.
    q8_metrics = [frozen.targets[name].ensemble_metrics for name in names]
    dr_metrics = [frozen.targets[name].directridge_metrics for name in names]
    r2_q8 = _pooled_per_output(q8_metrics)
    r2_dr = _pooled_per_output(dr_metrics)
    delta = r2_q8 - r2_dr

    # Manifold reconstruction quality per output, pooled across sessions.
    per_session_recon_metrics = []
    per_session_recon_rows: dict[str, Any] = {}
    for name in names:
        target = frozen.targets[name]
        _, full_y, full_indices = full_session_arrays(sessions[name])
        seed_recon = [seed.manifold.decode(seed.manifold.encode(full_y)) for seed in target.seeds]
        reconstruction = np.mean(np.stack(seed_recon, axis=0), axis=0, dtype=np.float64)
        metrics = regression_metrics(full_y, reconstruction)
        metrics.update(
            {
                "full_indices_sha256": array_sha256(full_indices),
                "prediction_sha256": array_sha256(reconstruction),
                "per_seed_reconstruction_sha256": [
                    array_sha256(value) for value in seed_recon
                ],
            }
        )
        per_session_recon_metrics.append(metrics)
        per_session_recon_rows[name] = {
            "pooled_variance_weighted_r2": float(metrics["pooled_variance_weighted_r2"]),
            "r2_per_output": dict(zip(output_names, metrics["r2_per_output"])),
        }
    loading = _pooled_per_output(per_session_recon_metrics)

    correlation = _pearson(loading, delta)
    order = np.argsort(loading)
    table = [
        {
            "output": output_names[index],
            "manifold_reconstruction_r2": float(loading[index]),
            "q8_route_r2": float(r2_q8[index]),
            "directridge_r2": float(r2_dr[index]),
            "delta_q8_minus_directridge": float(delta[index]),
        }
        for index in order
    ]
    pecmaj_index = output_names.index("PECmaj")
    pecmaj_rank_low = int(np.sum(loading < loading[pecmaj_index]))
    return {
        "schema": "m1_behavior_manifold_v2_e4_output_attribution_v1",
        "status": "COMPLETE_E4_PER_OUTPUT_ATTRIBUTION",
        "prediction_to_record": "gains concentrate in outputs the manifold reconstructs well; PECmaj at the low-reconstruction end",
        "correlation_loading_vs_delta": correlation,
        "pecmaj_check": {
            "delta": float(delta[pecmaj_index]),
            "manifold_reconstruction_r2": float(loading[pecmaj_index]),
            "rank_from_bottom_by_reconstruction": pecmaj_rank_low,
            "total_outputs": len(output_names),
            "known_mean_delta_from_handoff": -0.016162,
        },
        "per_output_table_sorted_by_reconstruction": table,
        "per_session_reconstruction_summary": per_session_recon_rows,
        "aggregation_convention": "per-output 1 - sum_sessions(SSE)/sum_sessions(TSS); three-seed mean decode of encoded full-session behaviour",
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
        },
    }
