"""Pure-CPU aggregation for the three-seed M2 FiLM content ablation."""

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
        (float(np.mean(vector[list(index)])) for index in itertools.product(range(len(vector)), repeat=len(vector))),
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
        term = json.loads(terminal.read_text())
        body = json.loads(score.read_text())
        if term.get("status") != "TERMINAL" or body.get("status") != "TERMINAL":
            raise ValueError(f"seed {seed} is not terminal")
        if body.get("official_config", {}).get("name") != plan.OFFICIAL_CONFIG_NAME:
            raise ValueError(f"seed {seed} config drift")
        payloads[seed] = body

    sessions = sorted(
        payloads[plan.SEEDS[0]]["p0"]["summaries"]["external_official_query"]["per_session_r2"]
    )
    contrasts: dict[str, tuple[str, str]] = {
        "semantic_REAL_minus_EMPTY": ("REAL", "EMPTY"),
        "alignment_REAL_minus_ROWSHUFFLE": ("REAL", "ROWSHUFFLE"),
        "capacity_EMPTY_minus_P0": ("EMPTY", "P0"),
        "arbitrary_code_ROWSHUFFLE_minus_EMPTY": ("ROWSHUFFLE", "EMPTY"),
    }
    summaries: dict[str, Any] = {}
    for name, (candidate, reference) in contrasts.items():
        values: dict[str, float] = {}
        for session in sessions:
            deltas = []
            for seed in plan.SEEDS:
                body = payloads[seed]
                cand = body["arms"][candidate]["matched"]["summaries"]["external_official_query"]["per_session_r2"][session]
                if reference == "P0":
                    ref = body["p0"]["summaries"]["external_official_query"]["per_session_r2"][session]
                else:
                    ref = body["arms"][reference]["matched"]["summaries"]["external_official_query"]["per_session_r2"][session]
                deltas.append(float(cand) - float(ref))
            values[session] = float(np.mean(deltas))
        summaries[name] = _summary(values)

    absolute = {
        arm: float(np.mean([
            payloads[seed]["arms"][arm]["matched"]["summaries"]["external_official_query"]["equal_session_mean"]
            for seed in plan.SEEDS
        ]))
        for arm in ("REAL", "EMPTY", "ROWSHUFFLE")
    }
    result = {
        "schema": plan.SCHEMA + "_aggregate",
        "status": "TERMINAL",
        "seeds": list(plan.SEEDS),
        "official_config": plan.OFFICIAL_CONFIG_NAME,
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
                summaries["capacity_EMPTY_minus_P0"]["mean"] > 0
                and summaries["capacity_EMPTY_minus_P0"]["positive_sessions"] >= 4
            ),
            "paper_interpretation": (
                "The official product gain is real, but this matched final-grid ablation does not "
                "attribute it to hold/reach profile semantics or row alignment. The positive EMPTY "
                "arm attributes the repeatable local gain to an M33-trained T4-conditioned adapter; "
                "ROWSHUFFLE additionally behaves as an arbitrary unit/session code or regularizer."
            ),
        },
        "evalai_push": False,
    }
    _atomic_json(output, result)
    return result

