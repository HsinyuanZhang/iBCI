#!/usr/bin/env python3
"""Apply the MOVE-T4 epoch-pick rule to the frozen M1 formal12 EMA table.

The 12-epoch EMA scores are already sealed. This program only re-ranks them.
It does not open ses-20120924, does not contact EvalAI, and does not mutate
the formal12 result root. Hidden/test files are not read.

Surface (disclosed): chronological 80/20 source-dev tails on
ses-20120926/27/28. This is not a six-session outer face; M1 has no legal
visible outer analogue to M2's Nov-24 held-out-calib without opening outer-24
query values.

Tie-break matches MOVE-T4 581919: higher equal-session mean, then higher
worst session, then earlier epoch, then lower seed. Seed is only 42.
The submission image prefers the current-query (T) operator for CPU timeout.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/xinyuan/Work_host/SPINT")
SRC = ROOT / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1"
RESULT = ROOT / "tfpd_exploration/results/m1_optimized_v2_source_dev_epoch_pick_v1"
SESSIONS = ("ses-20120926", "ses-20120927", "ses-20120928")
OPERATORS = ("full_window", "current_query")
TAG = "formal12_chron80_v2"
SEED = 42


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _postscore(kind: str) -> dict[str, Any]:
    path = SRC / f"formal_{TAG}_{kind}_postscore.json"
    require(path.is_file(), f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["tag"] == TAG and payload["operator"] == kind, "postscore identity drift")
    require(payload["outer_query_opened"] is False, "outer query leaked into formal postscore")
    return payload


def _ema_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in payload["all_rows"]:
        if raw["variant"] != "ema":
            continue
        per = {session: float(raw["per_session"][session]["r2"]["model"]) for session in SESSIONS}
        require(set(per) == set(SESSIONS), "source-dev session drift")
        mean = float(np.mean([per[s] for s in SESSIONS]))
        require(abs(mean - float(raw["equal_session_mean_r2"])) < 1.0e-10, "equal-session mean drift")
        rows.append(
            {
                "operator": payload["operator"],
                "seed": SEED,
                "epoch_one_based": int(raw["epoch"]),
                "variant": "EMA",
                "external_per_session_r2": per,
                "external_equal_session_mean": mean,
                "external_worst_session": min(per, key=per.get),
                "external_worst_session_r2": float(min(per.values())),
                "checkpoint": raw["checkpoint"],
                "checkpoint_sha256": raw["checkpoint_sha256"],
                "plain_ema_model_state": raw.get("plain_ema_model_state"),
                "plain_ema_model_state_sha256": raw.get("plain_ema_model_state_sha256"),
                "formal_selected_match": (
                    int(raw["epoch"]) == int(payload["selected"]["epoch"])
                    and raw["variant"] == payload["selected"]["variant"]
                ),
            }
        )
    require([row["epoch_one_based"] for row in rows] == list(range(1, 13)), "need EMA epochs 1..12")
    return rows


def _pick(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (
            -row["external_equal_session_mean"],
            -row["external_worst_session_r2"],
            row["epoch_one_based"],
            row["seed"],
        ),
    )[0]


def execute() -> dict[str, Any]:
    RESULT.mkdir(parents=True, exist_ok=True)
    out = RESULT / "selection.json"
    require(not out.exists(), f"refusing to overwrite {out}")
    by_op = {kind: _ema_rows(_postscore(kind)) for kind in OPERATORS}
    selected = {kind: _pick(by_op[kind]) for kind in OPERATORS}
    formal = {
        kind: {
            "epoch_one_based": int(_postscore(kind)["selected"]["epoch"]),
            "equal_session_mean_r2": float(_postscore(kind)["selected"]["equal_session_mean_r2"]),
        }
        for kind in OPERATORS
    }
    t_pick = selected["current_query"]
    full_pick = selected["full_window"]
    require(t_pick["plain_ema_model_state"], "missing T selected plain EMA state")
    require(
        Path(t_pick["plain_ema_model_state"]).is_file(),
        f"missing T state {t_pick['plain_ema_model_state']}",
    )
    result = {
        "schema": "m1_optimized_v2_source_dev_epoch_pick_v1",
        "status": "SELECTED_BEFORE_EVALAI",
        "selection_rule": (
            "max visible equal-session mean; higher worst-session, "
            "earlier epoch, lower seed tie-breaks"
        ),
        "selection_surface": (
            "chronological 80/20 source-dev tails on ses-20120926/27/28; "
            "ses-20120924 query unread"
        ),
        "copied_from": "tfpd_exploration/scripts/run_m2_movement_t4_empty_epoch_pick_v1.py",
        "tag": TAG,
        "seed": SEED,
        "view": "EMA",
        "epochs_one_based": list(range(1, 13)),
        "sessions": list(SESSIONS),
        "outer_session": "ses-20120924",
        "outer_query_opened": False,
        "evalai_opened": False,
        "hidden_evalai_score_used": False,
        "submit_operator": "current_query",
        "submit_operator_reason": (
            "T already passed the frozen FULL-retention gate and is the faster "
            "official-CPU operator; FULL remains the accuracy reference only"
        ),
        "formal_earliest_max_ema": formal,
        "selected": selected,
        "t4_rule_matches_formal_selected": {
            kind: int(selected[kind]["epoch_one_based"]) == formal[kind]["epoch_one_based"]
            for kind in OPERATORS
        },
        "curve": by_op,
        "image_state": {
            "operator": "current_query",
            "epoch_one_based": t_pick["epoch_one_based"],
            "path": t_pick["plain_ema_model_state"],
            "sha256": t_pick["plain_ema_model_state_sha256"],
        },
        "full_reference": {
            "operator": "full_window",
            "epoch_one_based": full_pick["epoch_one_based"],
            "equal_session_mean": full_pick["external_equal_session_mean"],
            "delta_t_minus_full": (
                t_pick["external_equal_session_mean"] - full_pick["external_equal_session_mean"]
            ),
        },
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out, result)
    return result


if __name__ == "__main__":
    payload = execute()
    print(
        json.dumps(
            {
                "selected": payload["selected"],
                "t4_rule_matches_formal_selected": payload["t4_rule_matches_formal_selected"],
                "image_state": payload["image_state"],
            },
            indent=2,
        ),
        flush=True,
    )
