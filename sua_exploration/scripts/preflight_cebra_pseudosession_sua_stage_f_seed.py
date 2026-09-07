#!/usr/bin/env python3
"""Mint one source-only pseudo-session schedule authority for seed 43 or 44.

This is intentionally additive: the running Stage-P implementation and its
immutable seed-42 authority are never rewritten.  A Stage-F seed is admissible
only after the frozen Stage-P aggregate says to expand.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import numpy as np


SUA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SUA_ROOT / "scripts"))

from mc_maze.cebra_pseudosession import CebraPseudoSessionDataModule  # noqa: E402
from mc_maze.unit_side_features import side_feature_stats_sha256  # noqa: E402
from preflight_cebra_pseudosession_sua import (  # noqa: E402
    CONTRACT,
    CONFIG,
    EXPECTED,
    MANIFEST,
    MODULE,
    SCREEN_ID,
    sha256_file,
    verify_readonly_pair,
    write_immutable_pair,
)
from train_cebra_pseudosession_sua import (  # noqa: E402
    EXPECTED_PREFLIGHT_SHA256,
    PREFLIGHT,
)


RESULT_ROOT = SUA_ROOT / "results" / SCREEN_ID
STAGE_P = RESULT_ROOT / "stage_p_seed42_aggregate.json"
EXPECTED_NORMALIZER = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
STAGE_F_SEEDS = (43, 44)


class StageFScheduleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StageFScheduleError(message)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def load_stage_p(path: Path = STAGE_P) -> tuple[dict[str, Any], str]:
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    require(path.is_file() and sidecar.is_file(), "Stage-P aggregate is missing")
    require(not path.is_symlink() and not sidecar.is_symlink(), "Stage-P symlink forbidden")
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, "Stage-P body is not immutable")
    require(stat.S_IMODE(sidecar.stat().st_mode) == 0o444, "Stage-P sidecar is not immutable")
    digest = sha256_file(path)
    require(sidecar.read_text(encoding="ascii").strip() == digest, "Stage-P sidecar drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("receipt_kind") == "cebra_pseudosession_sua_stage_p_aggregate",
            "Stage-P receipt kind drift")
    require(payload.get("seed") == 42, "Stage-P seed drift")
    require(payload.get("passes_stage_p") is True, "Stage-P did not authorize expansion")
    require(payload.get("verdict") == "STAGE_P_PASS__EXPAND_SEEDS_43_44",
            "Stage-P verdict did not authorize expansion")
    gates = payload.get("frozen_gates") or {}
    require(gates.get("external_absolute_t4_delta_at_least_0p03") is True,
            "external absolute-T4 gate failed")
    require(gates.get("within_t4_noninferior_at_minus_0p03") is True,
            "within non-inferiority gate failed")
    require(payload.get("formal_subc_test_nwb_opened") is False,
            "Stage-P opened formal data")
    return payload, digest


def datamodule(group: str, seed: int) -> CebraPseudoSessionDataModule:
    return CebraPseudoSessionDataModule(
        data_dir=str(SUA_ROOT / "data" / "dandi_000688" / "sub-C"),
        task="CO", split_counts=(27, 6, 6), batch_size=32, window_size=50,
        calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=0, pin_memory=False, random_calibration=False, seed=seed,
        max_units_exclusive=100,
        cache_dir=str(SUA_ROOT / "cache" / "dandi688_subc_co_v1"),
        signal_view="sua", side_feature_group=group, side_feature_pool_size=30,
        train_val_manifest_path=str(MANIFEST), pseudo_mix_probability=0.5,
        pseudo_contributor_count=3, pseudo_max_behavior_residual=0.25,
    )


def compare_plan(left: Any, right: Any) -> bool:
    if (left.anchor_session, left.anchor_start, left.mix_candidate, left.mixed) != (
        right.anchor_session, right.anchor_start, right.mix_candidate, right.mixed
    ):
        return False
    if not np.array_equal(left.final_permutation, right.final_permutation):
        return False
    return all(
        (a.session, a.start) == (b.session, b.start)
        and np.array_equal(a.unit_indices, b.unit_indices)
        and a.behavior_residual == b.behavior_residual
        for a, b in zip(left.contributors, right.contributors, strict=True)
    )


def validate_schedule_manifest(manifest: Mapping[str, Any], seed: int) -> None:
    require(manifest.get("seed") == seed, "schedule seed drift")
    require(manifest.get("examples_audited") == 1_086_007, "schedule is not full")
    require(0.47 <= float(manifest.get("mixed_fraction_observed", -1.0)) <= 0.51,
            "accepted mix fraction outside frozen gate")
    rejected = int(manifest.get("rejected_candidate_examples", -1))
    require(0 <= rejected <= 10_860, "behavior fallback exceeds one percent")
    residual = manifest.get("accepted_endpoint_residual") or {}
    require(float(residual.get("max", 1.0)) <= 0.25, "accepted residual exceeds 0.25")
    require(float(residual.get("p99", 1.0)) <= 0.10, "accepted residual P99 exceeds 0.10")
    donor_uses = manifest.get("accepted_donor_uses") or {}
    require(len(donor_uses) == 27 and all(int(value) > 0 for value in donor_uses.values()),
            "not every source session contributes accepted donor units")
    schedule_sha = manifest.get("schedule_sha256")
    require(isinstance(schedule_sha, str) and len(schedule_sha) == 64,
            "schedule SHA is missing")


def implementation_bindings() -> dict[str, str]:
    paths = {
        "stage_f_schedule_preflight": Path(__file__).resolve(),
        "pseudo_session_module": MODULE,
        "config": CONFIG,
        "contract": CONTRACT,
        "parent_manifest": MANIFEST,
        "seed42_constructibility_preflight": PREFLIGHT,
        "stage_p_aggregate": STAGE_P,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run(seed: int) -> dict[str, Any]:
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    require(seed in STAGE_F_SEEDS, "only seeds 43 and 44 are Stage-F schedules")
    verify_readonly_pair(PREFLIGHT, EXPECTED_PREFLIGHT_SHA256)
    require(sha256_file(MANIFEST) == EXPECTED["manifest"], "source manifest drift")
    stage_p, stage_p_sha = load_stage_p()
    before = implementation_bindings()

    t4 = datamodule("t4", seed)
    t4.setup("fit")
    require(len(t4.session_splits["train"]) == 27 and len(t4.session_splits["val"]) == 6,
            "source/development roster size drift")
    require(len(t4.session_splits["test"]) == 6 and len(t4.session_files["test"]) == 0,
            "formal paths were resolved")
    schedule = t4.pseudo_session_manifest()
    validate_schedule_manifest(schedule, seed)

    z4 = datamodule("z4", seed)
    z4.setup("fit")
    require(len(z4.session_files["test"]) == 0, "Z4 resolved formal paths")
    t4_mean, t4_std = t4._get_side_feature_stats()
    z4_mean, z4_std = z4._get_side_feature_stats()
    require(np.array_equal(t4_mean, z4_mean) and np.array_equal(t4_std, z4_std),
            "T4/Z4 source normalizer bit drift")
    normalizer_sha = side_feature_stats_sha256(t4_mean, t4_std)
    require(normalizer_sha == EXPECTED_NORMALIZER, "A2 T4 normalizer drift")
    require(all(np.count_nonzero(row.side_features) == 0
                for row in z4.train_dataset.sessions.values()),
            "Z4 source rows are not exact zero")
    parity_count = min(10_000, len(t4.train_dataset), len(z4.train_dataset))
    require(len(t4.train_dataset) == len(z4.train_dataset) == 1_086_007,
            "T4/Z4 dataset cardinality drift")
    require(all(compare_plan(t4.train_dataset.plan_for_index(index),
                             z4.train_dataset.plan_for_index(index))
                for index in range(parity_count)),
            "T4/Z4 pseudo-session plan prefix drift")

    after = implementation_bindings()
    require(before == after, "implementation drifted during Stage-F schedule audit")
    return {
        "schema_version": 1,
        "receipt_kind": "cebra_pseudosession_sua_stage_f_seed_schedule_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": SCREEN_ID,
        "seed": seed,
        "status": "STAGE_F_SOURCE_SCHEDULE_PASSED__GPU_NOT_LAUNCHED",
        "stage_p_aggregate_sha256": stage_p_sha,
        "stage_p_deltas": stage_p["deltas"],
        "source_schedule": schedule,
        "source_schedule_sha256": schedule["schedule_sha256"],
        "t4_z4_plan_prefix_compared": parity_count,
        "t4_z4_plan_prefix_exact": True,
        "t4_z4_source_normalizer_bit_equal": True,
        "t4_normalizer_sha256": normalizer_sha,
        "z4_source_rows_exact_zero": True,
        "formal_subc_test_nwb_opened": False,
        "cuda_used": False,
        "training_started": False,
        "implementation_bindings": after,
        "implementation_bindings_sha256": canonical_sha256(after),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=STAGE_F_SEEDS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    expected_output = RESULT_ROOT / f"stage_f_seed{args.seed}_source_schedule_preflight.json"
    output = args.output or expected_output
    require(output.resolve() == expected_output.resolve(), "Stage-F authority path drift")
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN__NO_DATA",
            "seed": args.seed,
            "requires_stage_p_pass": str(STAGE_P),
            "output": str(output),
        }, indent=2, sort_keys=True))
        return 0
    payload = run(args.seed)
    digest = write_immutable_pair(output, payload)
    print(json.dumps({"status": payload["status"], "seed": args.seed,
                      "output": str(output.resolve()), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
