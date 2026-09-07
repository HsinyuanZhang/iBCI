"""Fail-closed aggregate for A10 matched no-backprop cost matrix."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SUA_ROOT = Path(__file__).resolve().parents[1]
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze.gpu_contract_common import (
    PREDECLARED_SEEDS,
    SUBC_VAL_SESSIONS,
    assert_sessions_not_sealed,
    contract_sha256,
    fp32_mainline_gates,
    gate_verdict,
    load_json,
    paired_sigma_delta,
    require,
)

SCREEN_ID = "a10_no_backprop_cost_v1"
ARMS = ("gradient_free", "readout_probe", "full_finetune")
HEADROOM_THRESHOLD = 0.03


def load_arm(path: Path, arm: str, seed: int) -> tuple[list[str], dict[str, float], float]:
    artifact = load_json(path)
    if arm == "gradient_free" and artifact.get("generated_by") == "eval_epoch_window_generic_dandi688.py":
        require(f"{path}: schema", artifact.get("schema_version"), 1)
        require(f"{path}: no formal test", artifact.get("no_test_files_evaluated"), True)
        require(f"{path}: seed", artifact.get("seed"), seed)
        protocol = artifact.get("protocol") or {}
        require(f"{path}: calib", protocol.get("calibration_n"), 30)
        require(f"{path}: pool", protocol.get("pool_size"), 30)
        per_epoch = artifact.get("per_epoch") or {}
        last_epoch = str(max(int(k) for k in per_epoch))
        sessions = sorted(per_epoch[last_epoch]["per_session_r2"].keys())
        if list(sessions) != list(SUBC_VAL_SESSIONS):
            raise ValueError(f"{path}: validation session drift")
        assert_sessions_not_sealed(sessions, label=f"{path}")
        per_session = {
            s: float(per_epoch[last_epoch]["per_session_r2"][s]) for s in sessions
        }
        mean_r2 = float(np.mean(list(per_session.values())))
        require(f"{path}: backward", artifact.get("uses_backward_gradients"), False)
        return sessions, per_session, mean_r2

    artifact = load_json(path)
    require(f"{path}: schema", artifact.get("schema_version"), 1)
    require(f"{path}: screen", artifact.get("screen_id"), SCREEN_ID)
    require(f"{path}: arm", artifact.get("arm"), arm)
    require(f"{path}: seed", artifact.get("seed"), seed)
    require(f"{path}: no formal test", artifact.get("no_test_files_evaluated"), True)
    protocol = artifact.get("protocol") or {}
    require(f"{path}: calib", protocol.get("calibration_n"), 30)
    require(f"{path}: eval start", protocol.get("evaluation_start_trial"), 30)
    sessions = sorted((artifact.get("per_session_r2") or {}).keys())
    if sessions != list(SUBC_VAL_SESSIONS):
        raise ValueError(f"{path}: validation session drift")
    assert_sessions_not_sealed(sessions, label=f"{path}")
    if arm == "gradient_free":
        require(f"{path}: backward", artifact.get("uses_backward_gradients"), False)
        require(f"{path}: label updates", artifact.get("uses_behavior_labels_for_weight_updates"), False)
    else:
        require(f"{path}: backward", artifact.get("uses_backward_gradients"), True)
        require(f"{path}: label updates", artifact.get("uses_behavior_labels_for_weight_updates"), True)
    mean_r2 = float(artifact.get("mean_r2"))
    observed = float(np.mean([float(artifact["per_session_r2"][s]) for s in sessions]))
    if abs(mean_r2 - observed) > 1e-9:
        raise ValueError(f"{path}: mean_r2 inconsistent with sessions")
    return sessions, {s: float(artifact["per_session_r2"][s]) for s in sessions}, mean_r2


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

    means: dict[str, list[float]] = {arm: [] for arm in ARMS}
    matrices: dict[str, np.ndarray] = {}
    for arm in ARMS:
        values = []
        for seed in PREDECLARED_SEEDS:
            path = result_dir / f"{arm}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, per_session_map, mean_r2 = load_arm(path, arm, seed)
            values.append(mean_r2)
            means[arm].append(mean_r2)
        matrices[arm] = np.asarray(values, dtype=np.float64)

    best_headroom = []
    for seed_index, seed in enumerate(PREDECLARED_SEEDS):
        gf = matrices["gradient_free"][seed_index]
        best = max(
            matrices["readout_probe"][seed_index] - gf,
            matrices["full_finetune"][seed_index] - gf,
        )
        best_headroom.append(best)
    headroom_arr = np.asarray(best_headroom, dtype=np.float64)
    mean_headroom = float(headroom_arr.mean())
    sigma_paired = paired_sigma_delta(headroom_arr)

    if mean_headroom >= HEADROOM_THRESHOLD and float(np.min(headroom_arr)) > 0.0 and mean_headroom - 2.0 * sigma_paired > 0.0:
        headroom_verdict = gate_verdict(True, "headroom_large — decoder redesign may be reviewed", "headroom_large")
    elif mean_headroom + 2.0 * sigma_paired < HEADROOM_THRESHOLD:
        headroom_verdict = gate_verdict(False, "headroom_small — thesis confirmed; B13-B16 foreclosed", "headroom_small")
    else:
        headroom_verdict = gate_verdict(False, "headroom indeterminate", "headroom_indeterminate")

    # Build session-level matrices for secondary gates from per-session data
    session_matrix: dict[str, np.ndarray] = {}
    for arm in ARMS:
        rows = []
        for seed in PREDECLARED_SEEDS:
            path = result_dir / f"{arm}_s{seed}.json"
            _sessions, per_session_map, _mean = load_arm(path, arm, seed)
            rows.append([per_session_map[s] for s in SUBC_VAL_SESSIONS])
        session_matrix[arm] = np.asarray(rows, dtype=np.float64)

    secondary = {
        "readout_probe_minus_gradient_free": fp32_mainline_gates(
            session_matrix["readout_probe"] - session_matrix["gradient_free"],
            list(SUBC_VAL_SESSIONS),
        ),
        "full_finetune_minus_gradient_free": fp32_mainline_gates(
            session_matrix["full_finetune"] - session_matrix["gradient_free"],
            list(SUBC_VAL_SESSIONS),
            seed_draws=20260814,
        ),
    }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "screen_id": SCREEN_ID,
        "contract_path": str(contract),
        "contract_sha256": contract_sha256(contract),
        "arm_mean_r2": {arm: float(np.mean(means[arm])) for arm in ARMS},
        "headroom": {
            "mean_best_headroom": mean_headroom,
            "sigma_paired": sigma_paired,
            "per_seed_best_headroom": {
                str(seed): float(v) for seed, v in zip(PREDECLARED_SEEDS, headroom_arr)
            },
            "verdict": headroom_verdict,
        },
        "secondary_contrasts": secondary,
        "formal_test_files_opened": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"headroom_verdict": headroom_verdict}, indent=2))


if __name__ == "__main__":
    main()
