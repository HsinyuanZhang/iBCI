"""P3 Stage A: verify sealed RS4/LS4 aggregates against §2.1b baseline numbers.

CPU-only receipt. Does not re-run inference (that would need GPU / frozen ckpt
forward). Confirms on-disk aggregates still match the handoff structural claim
that both RS4 and LS4 land below Z4.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "carrier_perf_p3_stage_a_baseline_receipt_v1"

# Handoff §2.1b rounded display values; absolute_r2 means are the sealed truth.
EXPECTED = {
    "rs4": {
        "aggregate_rel": "sua_exploration/results/sua_t4_m30_ac4_rs4_v1/aggregate_v1.json",
        "keys": {"rs4": ("absolute_r2", "ac4_rs4", "mean_r2"), "z4": ("absolute_r2", "z4", "mean_r2")},
        "display_rs4": 0.2686,
        "display_z4": 0.3260,
        "exact_rs4": 0.26858999305922127,
        "exact_z4": 0.32600806570715374,
    },
    "ls4": {
        "aggregate_rel": "sua_exploration/results/sua_t4_m30_component_attribution_v10/aggregate_r11.json",
        "keys": {"ls4": ("absolute_r2", "ls4", "mean_r2"), "z4": ("absolute_r2", "z4", "mean_r2")},
        "display_ls4": 0.2656,
        "display_z4": 0.3260,
        "exact_ls4": 0.2656259809931119,
        "exact_z4": 0.32600806570715374,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dig(obj: Mapping[str, Any], path: tuple[str, ...]) -> float:
    cur: Any = obj
    for key in path:
        cur = cur[key]
    return float(cur)


def verify_aggregate(
    *,
    path: Path,
    value_key: str,
    nested: tuple[str, ...],
    exact: float,
    z4_nested: tuple[str, ...],
    exact_z4: float,
    atol: float = 1e-12,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = _dig(payload, nested)
    z4 = _dig(payload, z4_nested)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        value_key: value,
        "z4": z4,
        "matches_exact": math.isclose(value, exact, rel_tol=0.0, abs_tol=atol),
        "z4_matches_exact": math.isclose(z4, exact_z4, rel_tol=0.0, abs_tol=atol),
        "below_z4": value < z4,
        "delta_vs_z4": value - z4,
    }


def build_p3_stage_a_receipt(*, spint_root: Path) -> dict[str, Any]:
    rs4_path = spint_root / EXPECTED["rs4"]["aggregate_rel"]
    ls4_path = spint_root / EXPECTED["ls4"]["aggregate_rel"]
    rs4 = verify_aggregate(
        path=rs4_path,
        value_key="ac4_rs4",
        nested=EXPECTED["rs4"]["keys"]["rs4"],
        exact=EXPECTED["rs4"]["exact_rs4"],
        z4_nested=EXPECTED["rs4"]["keys"]["z4"],
        exact_z4=EXPECTED["rs4"]["exact_z4"],
    )
    ls4 = verify_aggregate(
        path=ls4_path,
        value_key="ls4",
        nested=EXPECTED["ls4"]["keys"]["ls4"],
        exact=EXPECTED["ls4"]["exact_ls4"],
        z4_nested=EXPECTED["ls4"]["keys"]["z4"],
        exact_z4=EXPECTED["ls4"]["exact_z4"],
    )
    structural = bool(rs4["below_z4"] and ls4["below_z4"])
    exact_ok = bool(
        rs4["matches_exact"]
        and rs4["z4_matches_exact"]
        and ls4["matches_exact"]
        and ls4["z4_matches_exact"]
    )
    return {
        "schema": SCHEMA,
        "status": "completed_cpu_only" if exact_ok and structural else "failed_baseline_mismatch",
        "no_gpu": True,
        "no_inference_rerun": True,
        "structural_claim_rs4_and_ls4_below_z4": structural,
        "exact_means_match_sealed_aggregates": exact_ok,
        "rs4": rs4,
        "ls4": ls4,
        "display_rounded": {
            "rs4": EXPECTED["rs4"]["display_rs4"],
            "ls4": EXPECTED["ls4"]["display_ls4"],
            "z4": EXPECTED["rs4"]["display_z4"],
        },
        "note": (
            "Stage A seals baseline consistency for the corruption hypothesis. "
            "Training-side corruption GPU cells remain blocked until authorized."
        ),
    }


def write_receipt(receipt: Mapping[str, Any], output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    path = output_dir / "receipt.json"
    tmp = output_dir / "receipt.json.tmp"
    tmp.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
