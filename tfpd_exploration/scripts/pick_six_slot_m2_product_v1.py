#!/usr/bin/env python3
"""Seal today's M2 visible-product picks. Does not rewrite old comparison files."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/xinyuan/Work_host/SPINT")
PACK = REPO / "tfpd_exploration/results/m2_b_small_stability_v1/astra_pack_v1"
OLD = REPO / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500"
DEST = REPO / "tfpd_exploration/results/six_evalai_slots_v1/20260905_155800"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _small_rows() -> list[dict[str, object]]:
    grouped: dict[tuple, dict[str, object]] = {}
    for seed in (42, 43):
        path = PACK / f"seed{seed}_ext4_per_session.csv"
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["view"] != "EMA":
                    continue
                key = (int(row["seed"]), row["cell"], int(row["epoch"]))
                item = grouped.setdefault(
                    key,
                    {
                        "seed": key[0],
                        "cell": key[1],
                        "epoch": key[2],
                        "equal_session_mean": float(row["R_session_equal_mean"]),
                        "per_session": {},
                    },
                )
                item["per_session"][row["session"]] = float(row["r2"])
    out = []
    for item in grouped.values():
        sessions = item["per_session"]
        item["worst_session"] = min(sessions.values()) if sessions else float("-inf")
        item["n_session"] = len(sessions)
        out.append(item)
    return out


def _pick(rows: list[dict[str, object]]) -> dict[str, object]:
    def key(row: dict[str, object]) -> tuple:
        return (
            -float(row["equal_session_mean"]),
            -float(row["worst_session"]),
            int(row["epoch"]),
            int(row["seed"]),
            str(row["cell"]),
        )

    ranked = sorted(rows, key=key)
    return ranked[0]


def _large_rows() -> list[dict[str, object]]:
    path = OLD / "arms/B-TRANSFORMER/seed42_shuffled_e13_24/ext4_epoch_scan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for epoch_s, row in payload["all_epochs"].items():
        per = {name: float(body["r2"]) for name, body in row["per_session"].items()}
        out.append(
            {
                "seed": 42,
                "cell": "B-TRANSFORMER",
                "view": "RAW",
                "epoch": int(epoch_s),
                "equal_session_mean": float(row["R_session_equal_mean"]),
                "worst_session": min(per.values()),
                "per_session": per,
                "ckpt": row["ckpt"],
            }
        )
    return out


def _cell_root(cell: str, seed: int) -> Path:
    from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg

    return cfg.run_root_for_cell(cell) / cell.replace("-", "_") / f"seed{seed}"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    small = _small_rows()
    large = _large_rows()
    s1 = _pick(small)
    s2 = _pick(large)
    ckpt_s1 = _cell_root(str(s1["cell"]), int(s1["seed"])) / f"epoch_{int(s1['epoch']):03d}.pt"
    ckpt_s2 = Path(str(s2["ckpt"]))
    payload = {
        "schema": "six_slot_m2_product_pick_v1",
        "t0": "2026-09-05T15:57:50+08:00",
        "rule": [
            "equal_session_mean desc",
            "worst_session desc",
            "epoch asc",
            "seed asc",
            "cell lex",
        ],
        "s1": {
            **s1,
            "view": "EMA",
            "ckpt": str(ckpt_s1),
            "ckpt_exists": ckpt_s1.is_file(),
            "note": "visible-product pick; does not rewrite source-pick routing",
        },
        "s2": {
            **s2,
            "ckpt_exists": ckpt_s2.is_file(),
            "note": "RAW only; shuffled 1-24; architecture probe",
        },
        "small_n": len(small),
        "large_n": len(large),
        "pack_csv_sha256": {
            "seed42": _sha256(PACK / "seed42_ext4_per_session.csv"),
            "seed43": _sha256(PACK / "seed43_ext4_per_session.csv"),
        },
        "large_scan_sha256": _sha256(
            OLD / "arms/B-TRANSFORMER/seed42_shuffled_e13_24/ext4_epoch_scan.json"
        ),
        "sealed_at": datetime.now(timezone.utc).isoformat(),
    }
    dest = DEST / "m2_product_picks.json"
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = DEST / "m2_s1_ema_candidate_table.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "cell", "epoch", "equal_session_mean", "worst_session"],
        )
        writer.writeheader()
        for row in sorted(small, key=lambda r: (r["seed"], r["cell"], r["epoch"])):
            writer.writerow(
                {
                    "seed": row["seed"],
                    "cell": row["cell"],
                    "epoch": row["epoch"],
                    "equal_session_mean": f"{row['equal_session_mean']:.10f}",
                    "worst_session": f"{row['worst_session']:.10f}",
                }
            )
    print(json.dumps({"s1": payload["s1"], "s2": payload["s2"], "dest": str(dest)}, indent=2))


if __name__ == "__main__":
    main()
