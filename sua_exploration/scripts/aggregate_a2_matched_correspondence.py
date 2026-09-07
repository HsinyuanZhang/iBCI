"""Fail-closed aggregate for A2 matched 2×2 correspondence matrix."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SUA_ROOT = Path(__file__).resolve().parents[1]
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze.gpu_contract_common import (
    PREDECLARED_SEEDS,
    SUBC_VAL_SESSIONS,
    SUBM_EXTERNAL_SESSIONS,
    assert_sessions_not_sealed,
    contract_sha256,
    fp32_mainline_gates,
    gate_verdict,
    hierarchical_bootstrap_ci,
    load_json,
    paired_sigma_delta,
    require,
)

SCREEN_ID = "a2_matched_correspondence_v1"
CELLS = ("within_z4", "within_t4", "cross_z4", "cross_t4")
WITHIN_SESSIONS = list(SUBC_VAL_SESSIONS)
CROSS_SESSIONS = list(SUBM_EXTERNAL_SESSIONS)
INTERACTION_THRESHOLD = 0.03


SIDE_FOR_CELL = {
    "within_z4": "z4",
    "within_t4": "t4",
    "cross_z4": "z4",
    "cross_t4": "t4",
}


def load_cell(path: Path, cell: str, seed: int, sessions: list[str]) -> np.ndarray:
    artifact = load_json(path)
    require(f"{path}: schema", artifact.get("schema_version"), 1)
    require(f"{path}: no formal test", artifact.get("no_test_files_evaluated"), True)
    require(f"{path}: seed", artifact.get("seed"), seed)
    require(f"{path}: variant", artifact.get("variant"), "B3S")
    protocol = artifact.get("protocol") or {}
    require(f"{path}: calibration_n", protocol.get("calibration_n"), 30)
    require(f"{path}: pool_size", protocol.get("pool_size"), 30)
    require(f"{path}: epochs", protocol.get("total_epochs"), 12)
    require(f"{path}: epoch window", protocol.get("epoch_window"), list(range(5, 13)))
    metadata_path = Path(artifact.get("run_metadata_path", ""))
    if metadata_path.is_file():
        metadata = load_json(metadata_path)
        side_group = (metadata.get("side_features") or {}).get("group")
        require(f"{path}: side group", side_group, SIDE_FOR_CELL[cell])
        training = metadata.get("training") or {}
        require(f"{path}: loss mode", training.get("loss_mode"), "task_only")
    per_epoch = artifact.get("per_epoch") or {}
    if set(per_epoch) != {str(e) for e in range(5, 13)}:
        raise ValueError(f"{path}: per_epoch window drift")
    rows = []
    for epoch in range(5, 13):
        values = per_epoch[str(epoch)].get("per_session_r2") or {}
        if set(values) != set(sessions):
            raise ValueError(f"{path}: session set drift at epoch {epoch}")
        rows.append([float(values[s]) for s in sessions])
    assert_sessions_not_sealed(sessions, label=f"{path}")
    return np.asarray(rows, dtype=np.float64).mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    contract = args.contract.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {out_path}")

    expected_contract_sha = contract_sha256(contract)
    matrices: dict[str, np.ndarray] = {}
    for cell in CELLS:
        sessions = WITHIN_SESSIONS if cell.startswith("within_") else CROSS_SESSIONS
        rows = []
        for seed in PREDECLARED_SEEDS:
            path = result_dir / f"{cell}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(load_cell(path, cell, seed, sessions))
        matrices[cell] = np.asarray(rows, dtype=np.float64)

    interaction = []
    for seed_index, seed in enumerate(PREDECLARED_SEEDS):
        within_gain = matrices["within_t4"][seed_index].mean() - matrices["within_z4"][seed_index].mean()
        cross_gain = matrices["cross_t4"][seed_index].mean() - matrices["cross_z4"][seed_index].mean()
        interaction.append(within_gain - cross_gain)
    interaction_arr = np.asarray(interaction, dtype=np.float64)
    mean_interaction = float(interaction_arr.mean())
    sigma_paired = paired_sigma_delta(interaction_arr)

    # bootstrap on synthetic 3x1 "session" matrix for interaction
    delta_matrix = interaction_arr.reshape(3, 1)
    ci = hierarchical_bootstrap_ci(delta_matrix, seed=20260812)
    try:
        wilcoxon_p = float(
            wilcoxon(interaction_arr, alternative="two-sided", zero_method="wilcox", method="exact").pvalue
        )
    except ValueError:
        wilcoxon_p = 1.0

    interaction_gates = {
        "mean_interaction_at_least_0p03": mean_interaction >= INTERACTION_THRESHOLD,
        "all_three_seed_interactions_positive": bool(np.all(interaction_arr > 0.0)),
        "bootstrap_95ci_lower_positive": ci[0] > 0.0,
        "exact_wilcoxon_le_0p05": wilcoxon_p <= 0.05,
    }
    interaction_effective = all(interaction_gates.values())

    secondary = {
        "within_t4_minus_z4": fp32_mainline_gates(
            matrices["within_t4"] - matrices["within_z4"],
            WITHIN_SESSIONS,
        ),
        "cross_t4_minus_z4": fp32_mainline_gates(
            matrices["cross_t4"] - matrices["cross_z4"],
            CROSS_SESSIONS,
            seed_draws=20260813,
        ),
    }

    if interaction_effective:
        interaction_verdict = gate_verdict(
            passes=True,
            reason="all pre-registered interaction gates passed",
            gate_name="interaction_effective",
        )
    elif mean_interaction + 2.0 * sigma_paired < INTERACTION_THRESHOLD:
        interaction_verdict = gate_verdict(
            passes=False,
            reason="confidently excludes +0.03 interaction",
            gate_name="interaction_ineffective",
        )
    else:
        interaction_verdict = gate_verdict(
            passes=False,
            reason="underpowered or indeterminate; report descriptive 2x2 only",
            gate_name="interaction_indeterminate",
            details={"power_note": "3 seeds cannot support reliable formal interaction inference"},
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "screen_id": SCREEN_ID,
        "contract_path": str(contract),
        "contract_sha256": expected_contract_sha,
        "cell_mean_r2": {
            cell: float(matrices[cell].mean()) for cell in CELLS
        },
        "carrier_gain_within_per_seed": {
            str(seed): float(
                matrices["within_t4"][i].mean() - matrices["within_z4"][i].mean()
            )
            for i, seed in enumerate(PREDECLARED_SEEDS)
        },
        "carrier_gain_cross_per_seed": {
            str(seed): float(
                matrices["cross_t4"][i].mean() - matrices["cross_z4"][i].mean()
            )
            for i, seed in enumerate(PREDECLARED_SEEDS)
        },
        "interaction": {
            "mean": mean_interaction,
            "sigma_paired": sigma_paired,
            "per_seed": {str(s): float(v) for s, v in zip(PREDECLARED_SEEDS, interaction_arr)},
            "bootstrap_95ci": ci,
            "wilcoxon_p": wilcoxon_p,
            "gates": interaction_gates,
            "passes_all_gates": interaction_effective,
            "verdict": interaction_verdict,
        },
        "secondary_contrasts": secondary,
        "formal_test_files_opened": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"interaction_verdict": interaction_verdict, "mean_interaction": mean_interaction}, indent=2))


if __name__ == "__main__":
    main()
