"""Fail-closed aggregate for A14 B3T efficiency branch completion."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SUA_ROOT = Path(__file__).resolve().parents[1]
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from scripts.aggregate_sua_b3t_t4_efficiency import (
    ARMS,
    check_aligned_first_gate,
    load_cell,
    summarize,
)
from mc_maze.gpu_contract_common import contract_sha256, gate_verdict, require


SCREEN_ID = "sua_b3t_t4_efficiency_v1"
COMPLETION_SEED = 42


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

    seeds = (COMPLETION_SEED,)
    for arm in ARMS:
        path = result_dir / f"{arm}_s{COMPLETION_SEED}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing required cell: {path}")

    gate = check_aligned_first_gate(result_dir, seed=COMPLETION_SEED)
    require("aligned-first gate", gate.get("control_permitted"), True)

    matrices: dict[str, np.ndarray] = {}
    session_names: list[str] | None = None
    for arm in ARMS:
        sessions, values, _receipt = load_cell(
            result_dir / f"{arm}_s{COMPLETION_SEED}.json",
            arm,
            COMPLETION_SEED,
        )
        if session_names is None:
            session_names = sessions
        elif sessions != session_names:
            raise ValueError(f"{arm}: session set drift")
        matrices[arm] = np.asarray([values], dtype=np.float64)

    assert session_names is not None
    accuracy = summarize(
        matrices["b3t_t4"],
        matrices["t4"],
        seeds=seeds,
        sessions=session_names,
    )
    content = summarize(
        matrices["b3t_t4"],
        matrices["b3t_ts4"],
        seeds=seeds,
        sessions=session_names,
    )
    seed_delta = (matrices["b3t_t4"] - matrices["t4"]).mean(axis=1)
    ni_pass = float(seed_delta.mean()) >= -0.03
    content_pass = bool(content.get("passes_strict_superiority"))

    if content_pass and ni_pass:
        branch_verdict = gate_verdict(True, "content and efficiency gates pass", "branch_claimable")
    elif ni_pass and not content_pass:
        branch_verdict = gate_verdict(False, "efficiency only; no T4 content claim", "efficiency_only")
    elif content_pass and not ni_pass:
        branch_verdict = gate_verdict(False, "content only; no efficiency claim", "content_only")
    else:
        branch_verdict = gate_verdict(False, "branch closes cleanly", "branch_closed")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "screen_id": SCREEN_ID,
        "contract_path": str(contract),
        "contract_sha256": contract_sha256(contract),
        "completion_seed": COMPLETION_SEED,
        "aligned_first_gate": gate,
        "branch_verdict": branch_verdict,
        "content_gate_pass": content_pass,
        "efficiency_gate_pass": ni_pass,
        "contrasts": {
            "b3t_t4_vs_fresh_t4": accuracy,
            "b3t_t4_vs_b3t_ts4": content,
        },
        "mean_seed_paired_delta_r2": float(seed_delta.mean()),
        "formal_test_files_opened": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"branch_verdict": branch_verdict}, indent=2))


if __name__ == "__main__":
    main()
