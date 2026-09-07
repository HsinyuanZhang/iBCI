#!/usr/bin/env python3
"""Additive source trainer for the frozen SUA pseudo-session intervention.

The established A2 trainer is imported without modification.  This entrypoint
replaces only its source DataModule constructor and decorates run metadata with
the immutable pseudo-session authority and the seed-specific full schedule.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


SUA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SUA_ROOT / "scripts"))

from mc_maze.cebra_pseudosession import CebraPseudoSessionDataModule  # noqa: E402
import train_variant_dandi688 as base  # noqa: E402


SCREEN_ID = "cebra_pseudosession_sua_v1"
CONFIG = SUA_ROOT / "configs" / f"{SCREEN_ID}.json"
CONTRACT = SUA_ROOT / "docs" / "CEBRA_PSEUDOSESSION_SUA_CONTRACT_20260814.md"
PREFLIGHT = SUA_ROOT / "results" / SCREEN_ID / "official_cpu_preflight.json"
EXPECTED_PREFLIGHT_SHA256 = "6f1e522053b4e45f2999a852a96d4c3b9b4de3ef318f0ce5a65980fe3dcbd770"
EXPECTED_SEED42_SCHEDULE_SHA256 = "070a1463b5495d7d3adb094c3ba962e4d9710da7491cbccd82109e0b4dbb2894"


class TrainerContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainerContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flag_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    require(index + 1 < len(argv), f"missing value for {flag}")
    return argv[index + 1]


def validate_argv(argv: list[str]) -> tuple[str, int]:
    exact = {
        "--variant": "B3S",
        "--task": "CO",
        "--split_counts": "27,6,6",
        "--signal_view": "sua",
        "--side_feature_pool_size": "30",
        "--calibration_n_trials": "30",
        "--max_units_exclusive": "100",
        "--batch_size": "32",
        "--num_workers": "4",
        "--lr": "0.0001",
        "--max_epochs": "12",
        "--loss_mode": "task_only",
        "--identity_mode": "calibrated",
    }
    for flag, expected in exact.items():
        require(_flag_value(argv, flag) == expected, f"frozen trainer argument drift: {flag}")
    arm = _flag_value(argv, "--side_features")
    require(arm in {"t4", "z4"}, "fresh pseudo-session arm must be t4 or z4")
    seed_text = _flag_value(argv, "--seed")
    require(seed_text is not None and int(seed_text) in {42, 43, 44}, "seed drift")
    expected_name = f"{SCREEN_ID}_mix_{arm}_s{seed_text}"
    require(_flag_value(argv, "--out_name") == expected_name, "source run name drift")
    require("--chronological_calibration" in argv, "chronological M30 is required")
    require("--no_early_stopping" in argv, "no-early-stopping rule is required")
    require("--checkpoint_every_epoch" in argv, "all epoch checkpoints are required")
    require("--require_gpu" in argv, "GPU execution must fail instead of falling back to CPU")
    require("--disable_progress_bar" in argv, "immutable log mode requires disabled progress bar")
    forbidden = {
        "--freeze_decoder", "--freeze_encoder_base", "--limit_train_batches",
        "--limit_val_batches", "--random_calibration",
    }
    require(not forbidden.intersection(argv), "forbidden source-training argument present")
    data_dir = Path(str(_flag_value(argv, "--data_dir"))).expanduser().resolve()
    manifest = Path(str(_flag_value(argv, "--train_val_manifest"))).expanduser().resolve()
    teacher = Path(str(_flag_value(argv, "--teacher_ckpt"))).expanduser().resolve()
    cache = Path(str(_flag_value(argv, "--cache_dir"))).expanduser().resolve()
    require(data_dir == (SUA_ROOT / "data" / "dandi_000688" / "sub-C").resolve(),
            "source data root drift")
    require(manifest == (SUA_ROOT / "configs" / "subc_co_27_6_strict_train_val_manifest.json").resolve(),
            "strict source manifest drift")
    require(teacher == (SUA_ROOT / "checkpoints" / "teacher_mc_maze" /
                         "best-epoch=083-val_heldin" / "r2_mean=0.9061.ckpt").resolve(),
            "teacher path drift")
    require(cache == (SUA_ROOT / "cache" / "dandi688_subc_co_v1").resolve(),
            "source cache root drift")
    return str(arm), int(seed_text)


def load_preflight() -> dict[str, Any]:
    require(PREFLIGHT.is_file() and Path(str(PREFLIGHT) + ".sha256").is_file(),
            "official constructibility preflight is missing")
    require(sha256_file(PREFLIGHT) == EXPECTED_PREFLIGHT_SHA256, "preflight body SHA drift")
    require(Path(str(PREFLIGHT) + ".sha256").read_text(encoding="ascii").strip() ==
            EXPECTED_PREFLIGHT_SHA256, "preflight sidecar drift")
    payload = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    require(payload.get("status") == "CPU_PREFLIGHT_PASSED__GPU_NOT_LAUNCHED",
            "preflight status drift")
    require(payload.get("formal_subc_test_nwb_opened") is False, "preflight opened formal data")
    current = {
        "config": sha256_file(CONFIG),
        "contract": sha256_file(CONTRACT),
        "pseudo_session_module": sha256_file(SUA_ROOT / "mc_maze" / "cebra_pseudosession.py"),
        "focused_test": sha256_file(SUA_ROOT / "tests" / "test_cebra_pseudosession.py"),
        "preflight": sha256_file(SUA_ROOT / "scripts" / "preflight_cebra_pseudosession_sua.py"),
    }
    bound = payload.get("implementation_bindings", {})
    require(all(bound.get(name) == digest for name, digest in current.items()),
            "constructibility implementation binding drift")
    return payload


def main() -> None:
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    arm, seed = validate_argv(sys.argv[1:])
    preflight = load_preflight()
    active_dm: CebraPseudoSessionDataModule | None = None
    runtime_manifest: dict[str, Any] | None = None
    original_write_json = base.write_json

    def factory(*args: Any, **kwargs: Any) -> CebraPseudoSessionDataModule:
        nonlocal active_dm
        require(active_dm is None, "base trainer instantiated the source DataModule more than once")
        active_dm = CebraPseudoSessionDataModule(
            *args,
            pseudo_mix_probability=0.5,
            pseudo_contributor_count=3,
            pseudo_max_behavior_residual=0.25,
            **kwargs,
        )
        return active_dm

    def decorated_write_json(path: Path, payload: dict[str, Any]) -> None:
        nonlocal runtime_manifest
        if path.name == "run_metadata.json":
            if active_dm is not None and active_dm.train_dataset is not None and runtime_manifest is None:
                runtime_manifest = dict(active_dm.pseudo_session_manifest())
                if seed == 42:
                    require(runtime_manifest["schedule_sha256"] == EXPECTED_SEED42_SCHEDULE_SHA256,
                            "seed42 full pseudo-session schedule drift")
            payload["cebra_pseudosession"] = {
                "screen_id": SCREEN_ID,
                "source_training_only": True,
                "arm": f"mix_{arm}",
                "seed": seed,
                "config_path": str(CONFIG.resolve()),
                "config_sha256": sha256_file(CONFIG),
                "contract_path": str(CONTRACT.resolve()),
                "contract_sha256": sha256_file(CONTRACT),
                "constructibility_preflight_path": str(PREFLIGHT.resolve()),
                "constructibility_preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
                "mix_probability": 0.5,
                "contributor_count": 3,
                "max_behavior_residual": 0.25,
                "anchor_sampling_unchanged": True,
                "validation_and_deployment_unmixed": True,
                "runtime_source_schedule": runtime_manifest,
            }
        original_write_json(path, payload)

    base.Dandi688MultiSessionDataModule = factory
    base.write_json = decorated_write_json
    # Retain a local reference so an accidental preflight mutation cannot be
    # optimized away as an unused validation.
    require(preflight["pseudo_session_audit"]["schedule_sha256"] ==
            EXPECTED_SEED42_SCHEDULE_SHA256, "preflight schedule authority drift")
    base.main()


if __name__ == "__main__":
    main()
