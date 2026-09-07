#!/usr/bin/env python3
"""Run score-free live setup probes for both post-33 data modules and all folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "SPINT-main/data/000953"
OLD_AUDIT = ROOT / "sua_exploration/results/m2_heldin_postsupport_endpoint_v1/audit.json"
DEFAULT_OUT = ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_live_plumbing_20260804/live_plumbing.json"
MARKER = "POST33_LIVE_JSON="


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(side: str, fold: int) -> dict[str, Any]:
    if side == "spint":
        cwd = ROOT / "SPINT-main"
        class_name = "M2Post33ConfirmSPINTDataModule"
    elif side == "t4":
        cwd = ROOT / "streaming_calibration_exp"
        class_name = "M2Post33ConfirmT4DataModule"
    else:
        raise ValueError(side)
    code = f"""
import json
from src.data.falcon_post33_confirm_v1_datamodule import {class_name}
dm={class_name}(task='m2',data_dir={str(DATA)!r},validation_protocol='loso',loso_fold={fold},calibration_n_trials=33,heldin_query_start_trial=33,random_calibration=False,include_heldout_in_fit=False,num_workers=0)
dm.setup('fit')
print({MARKER!r}+json.dumps({{'manifest':dm.get_split_manifest(),'query_len':len(dm.post33_query_dataset),'query_audit':dm.post33_query_dataset.query_window_audit}},sort_keys=True))
"""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = "."
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{side} fold{fold} live setup failed ({completed.returncode}):\n{completed.stderr}"
        )
    rows = [line for line in completed.stdout.splitlines() if line.startswith(MARKER)]
    if len(rows) != 1:
        raise ValueError(f"{side} fold{fold} emitted {len(rows)} JSON markers")
    payload = json.loads(rows[0][len(MARKER) :])
    payload["stderr_was_empty"] = not completed.stderr.strip()
    return payload


def build() -> dict[str, Any]:
    structural = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))[
        "heldin_postsupport_window_audit"
    ]["per_session"]
    evidence: dict[str, dict[str, Any]] = {"spint": {}, "t4": {}}
    for side in evidence:
        for fold in range(7):
            record = probe(side, fold)
            manifest = record["manifest"]
            outer = manifest["outer_left_out_session"]
            expected = structural[outer]["query_window_audit"]
            if manifest["outer_counts"] != {
                "train": 0,
                "normalizer": 0,
                "checkpoint_selection": 0,
                "post33_query": 1,
            }:
                raise ValueError(f"{side} fold{fold} outer role leak")
            if record["query_len"] != expected["eligible_windows"]:
                raise ValueError(f"{side} fold{fold} query count mismatch")
            query = record["query_audit"]
            if list(query) != [outer]:
                raise ValueError(f"{side} fold{fold} query is not the unique outer session")
            audit = query[outer]
            if (
                audit["query_start_trial"] != 33
                or audit["window_size"] != 50
                or audit["full_window_disjoint"] is not True
                or audit["minimum_window_start_padded_bin"] - audit["raw_query_start_bin"] != 49
            ):
                raise ValueError(f"{side} fold{fold} full-history boundary failed")
            evidence[side][str(fold)] = record
    totals = {
        side: sum(record["query_len"] for record in per_fold.values())
        for side, per_fold in evidence.items()
    }
    if totals != {"spint": 101_171, "t4": 101_171}:
        raise ValueError(f"live all-fold window totals mismatch: {totals}")
    return {
        "schema_version": 1,
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "PASS_SCORE_FREE_LIVE_PLUMBING",
        "execution_scope": {
            "gpu_used": False,
            "training_started": False,
            "optimizer_steps": 0,
            "new_endpoint_r2_values_read": 0,
            "scorer_modules_imported": 0,
            "formal_sua_paths_resolved": 0,
            "evalai_calls": 0,
        },
        "sides": evidence,
        "eligible_window_totals": totals,
        "probe_count": 14,
    }


def write(out: Path) -> Path:
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    with out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with out.with_suffix(out.suffix + ".sha256").open("x", encoding="utf-8") as handle:
        handle.write(f"{sha256(out)}  {out.name}\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    path = write(args.out)
    print(path)
    print(sha256(path))


if __name__ == "__main__":
    main()

