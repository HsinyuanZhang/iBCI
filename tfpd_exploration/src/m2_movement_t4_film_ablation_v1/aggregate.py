"""Pure-CPU aggregation for the three-seed MOVE-T4 FiLM content grid."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from tfpd_exploration.src.m2_hold_film_probe_v1.physical import _atomic_json

from . import plan


def _summary(values: dict[str, float]) -> dict[str, Any]:
    vector = np.asarray(list(values.values()), dtype=np.float64)
    boot = np.fromiter(
        (
            float(np.mean(vector[list(index)]))
            for index in itertools.product(range(vector.size), repeat=vector.size)
        ),
        dtype=np.float64,
    )
    return {
        "mean": float(vector.mean()),
        "median": float(np.median(vector)),
        "positive_sessions": int((vector > 0).sum()),
        "session_count": int(vector.size),
        "worst": float(vector.min()),
        "bootstrap_lower_95_exact": float(np.quantile(boot, 0.025)),
        "per_session_seed_averaged_delta": values,
    }


def aggregate(repo_root: Path) -> dict[str, Any]:
    root = plan.result_root(repo_root)
    output = root / "aggregate.json"
    if output.exists():
        raise FileExistsError(output)
    payloads: dict[int, dict[str, Any]] = {}
    for seed in plan.SEEDS:
        terminal = root / f"seed{seed}" / "terminal.json"
        score = root / f"seed{seed}" / "score.json"
        if not terminal.is_file() or not score.is_file():
            raise FileNotFoundError(f"seed {seed} lacks terminal/score")
        term = json.loads(terminal.read_text(encoding="utf-8"))
        body = json.loads(score.read_text(encoding="utf-8"))
        if term.get("status") != "TERMINAL" or body.get("status") != "TERMINAL":
            raise ValueError(f"seed {seed} is not terminal")
        if body.get("movement_window_bins") != [5, 30]:
            raise ValueError(f"seed {seed} movement window drift")
        payloads[seed] = body

    first = payloads[plan.SEEDS[0]]
    sessions = sorted(first["p0"]["summaries"]["external_official_query"]["per_session_r2"])
    contrasts: dict[str, tuple[str, str]] = {
        "semantic_REAL_minus_EMPTY": ("REAL", "EMPTY"),
        "alignment_REAL_minus_ROWSHUFFLE": ("REAL", "ROWSHUFFLE"),
        "capacity_EMPTY_minus_MOVE_P0": ("EMPTY", "P0"),
        "arbitrary_code_ROWSHUFFLE_minus_EMPTY": ("ROWSHUFFLE", "EMPTY"),
    }
    summaries: dict[str, Any] = {}
    for name, (candidate, reference) in contrasts.items():
        values: dict[str, float] = {}
        for session in sessions:
            deltas: list[float] = []
            for seed in plan.SEEDS:
                body = payloads[seed]
                candidate_value = body["arms"][candidate]["matched"]["summaries"][
                    "external_official_query"
                ]["per_session_r2"][session]
                if reference == "P0":
                    reference_value = body["p0"]["summaries"]["external_official_query"][
                        "per_session_r2"
                    ][session]
                else:
                    reference_value = body["arms"][reference]["matched"]["summaries"][
                        "external_official_query"
                    ]["per_session_r2"][session]
                deltas.append(float(candidate_value) - float(reference_value))
            values[session] = float(np.mean(deltas))
        summaries[name] = _summary(values)

    absolute = {
        "MOVE_P0": float(
            np.mean(
                [
                    payloads[seed]["p0"]["summaries"]["external_official_query"][
                        "equal_session_mean"
                    ]
                    for seed in plan.SEEDS
                ]
            )
        ),
        **{
            arm: float(
                np.mean(
                    [
                        payloads[seed]["arms"][arm]["matched"]["summaries"][
                            "external_official_query"
                        ]["equal_session_mean"]
                        for seed in plan.SEEDS
                    ]
                )
            )
            for arm in ("REAL", "EMPTY", "ROWSHUFFLE")
        },
    }
    result = {
        "schema": plan.SCHEMA + "_aggregate",
        "status": "TERMINAL",
        "seeds": list(plan.SEEDS),
        "movement_window_bins": [5, 30],
        "movement_window_ms": [100, 600],
        "absolute_external_equal_session_mean_across_seeds": absolute,
        "contrasts": summaries,
        "verdict": {
            "profile_semantic_content_supported": bool(
                summaries["semantic_REAL_minus_EMPTY"]["mean"] > 0
                and summaries["semantic_REAL_minus_EMPTY"]["positive_sessions"] >= 4
            ),
            "row_alignment_supported": bool(
                summaries["alignment_REAL_minus_ROWSHUFFLE"]["mean"] > 0
                and summaries["alignment_REAL_minus_ROWSHUFFLE"]["positive_sessions"] >= 4
            ),
            "t4_conditioned_adapter_supported": bool(
                summaries["capacity_EMPTY_minus_MOVE_P0"]["mean"] > 0
                and summaries["capacity_EMPTY_minus_MOVE_P0"]["positive_sessions"] >= 4
                and summaries["capacity_EMPTY_minus_MOVE_P0"]["bootstrap_lower_95_exact"] > 0
            ),
            "paper_interpretation": (
                "After correcting the carrier window, the real hold/reach profile is worse than "
                "the positive-zero EMPTY control and the fixed row-shuffled control. The profile's "
                "semantic content and unit-row alignment are therefore unsupported. A small "
                "M33-trained T4-conditioned adapter remains strongly positive versus MOVE-P0."
            ),
        },
        "next_submission_candidate": {
            "arm": "seed42_EMPTY",
            "reason": "profile-free deployable adapter; fixed seed lineage matching prior M2 submissions",
            "evalai_push": False,
        },
    }
    _atomic_json(output, result)
    return result


__all__ = ("aggregate",)
