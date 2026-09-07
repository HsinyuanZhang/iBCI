"""Fail-closed aggregate for B1 M2 T4 loss-mode sweep (contract + aggregator only)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

SUA_ROOT = Path(__file__).resolve().parents[1]
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze.gpu_contract_common import (
    PREDECLARED_SEEDS,
    contract_sha256,
    fp32_mainline_gates,
    gate_verdict,
    load_json,
    require,
)

SCREEN_ID = "m2_t4_loss_mode_sweep_v1"
LOSS_MODES = ("task_only", "task_plus_y", "task_plus_y_plus_E")
LAMBDA_EXPECTATIONS = {
    "task_only": (0.0, 0.0),
    "task_plus_y": (1.0, 0.0),
    "task_plus_y_plus_E": (1.0, 0.1),
}


def load_cell(path: Path, loss_mode: str, seed: int) -> tuple[list[str], np.ndarray]:
    artifact = load_json(path)
    require(f"{path}: screen", artifact.get("screen_id"), SCREEN_ID)
    require(f"{path}: loss_mode", artifact.get("loss_mode"), loss_mode)
    require(f"{path}: seed", artifact.get("seed"), seed)
    require(f"{path}: variant", artifact.get("variant"), "B3S")
    require(f"{path}: side", artifact.get("side_feature_group"), "t4")
    lam_y, lam_e = LAMBDA_EXPECTATIONS[loss_mode]
    require(f"{path}: lambda_y", artifact.get("lambda_y"), lam_y)
    require(f"{path}: lambda_E", artifact.get("lambda_E"), lam_e)
    require(f"{path}: calib trials", artifact.get("calibration_n_trials"), 33)
    require(f"{path}: epochs", artifact.get("max_epochs"), 12)
    sessions = sorted((artifact.get("per_session_r2") or {}).keys())
    if len(sessions) != 4:
        raise ValueError(f"{path}: expected 4 held-out sessions, got {len(sessions)}")
    values = np.asarray([float(artifact["per_session_r2"][s]) for s in sessions], dtype=np.float64)
    return sessions, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    contract = args.contract.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {out_path}")

    matrices: dict[str, np.ndarray] = {}
    session_names: list[str] | None = None
    for mode in LOSS_MODES:
        rows = []
        for seed in PREDECLARED_SEEDS:
            path = result_dir / f"{mode}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, values = load_cell(path, mode, seed)
            if session_names is None:
                session_names = sessions
            elif sessions != session_names:
                raise ValueError(f"{path}: session set drift")
            rows.append(values)
        matrices[mode] = np.asarray(rows, dtype=np.float64)

    assert session_names is not None
    primary = fp32_mainline_gates(
        matrices["task_only"] - matrices["task_plus_y_plus_E"],
        session_names,
    )
    if primary["passes_all_gates"]:
        primary_verdict = gate_verdict(True, "task_only beats task_plus_y_plus_E", "task_only_beats_distill")
    else:
        primary_verdict = gate_verdict(
            False,
            "kill criterion: distillation confound not supported",
            "task_only_fails_to_beat_distill",
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "screen_id": SCREEN_ID,
        "contract_path": str(contract),
        "contract_sha256": contract_sha256(contract),
        "arm_mean_r2": {mode: float(matrices[mode].mean()) for mode in LOSS_MODES},
        "primary_contrast_task_only_minus_task_plus_y_plus_E": primary,
        "primary_verdict": primary_verdict,
        "secondary_task_plus_y_minus_task_plus_y_plus_E": fp32_mainline_gates(
            matrices["task_plus_y"] - matrices["task_plus_y_plus_E"],
            session_names,
            seed_draws=20260815,
        ),
        "forecloses_b2_b3_on_failure": primary_verdict["gate_name"] == "task_only_fails_to_beat_distill",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"primary_verdict": primary_verdict}, indent=2))


if __name__ == "__main__":
    main()
