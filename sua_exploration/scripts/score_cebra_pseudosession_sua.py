#!/usr/bin/env python3
"""Score one completed pseudo-session source run on an unchanged A2 domain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SUA_ROOT / "scripts"))

from mc_maze import a2_matched_subject_shift_v2_core as a2core  # noqa: E402
import a2_matched_subject_shift_v2_score as a2score  # noqa: E402
from preflight_cebra_pseudosession_sua import (  # noqa: E402
    EXPECTED,
    write_immutable_pair,
)
from train_cebra_pseudosession_sua import (  # noqa: E402
    EXPECTED_PREFLIGHT_SHA256,
    EXPECTED_SEED42_SCHEDULE_SHA256,
    PREFLIGHT,
    SCREEN_ID,
    load_preflight,
)
from execute_cebra_pseudosession_sua_cell import paths_for  # noqa: E402


RESULT_ROOT = SUA_ROOT / "results" / SCREEN_ID
A2_PREFLIGHT = SUA_ROOT / "results/a2_matched_subject_shift_v2/official_cpu_preflight.json"


class ScoreError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_immutable(path: Path, expected_kind: str) -> tuple[dict[str, Any], str]:
    sidecar = Path(str(path) + ".sha256")
    require(path.is_file() and sidecar.is_file(), f"missing immutable receipt: {path}")
    digest = sha256_file(path)
    require(sidecar.read_text(encoding="ascii").strip() == digest, f"receipt sidecar drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("receipt_kind") == expected_kind, f"receipt kind drift: {path}")
    return payload, digest


def validate_a2_measurement_surface() -> dict[str, Any]:
    require(sha256_file(A2_PREFLIGHT) == EXPECTED["a2_preflight"], "A2 preflight body drift")
    payload = json.loads(A2_PREFLIGHT.read_text(encoding="utf-8"))
    a2core.verify_implementation_bindings(payload.get("implementation_bindings"))
    return payload


def validate_source(arm: str, seed: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    require(arm in {"t4", "z4"} and seed in {42, 43, 44}, "source cell drift")
    targets = paths_for(arm, seed)
    launch, launch_sha = load_immutable(
        targets["launch"], "cebra_pseudosession_sua_cell_launch"
    )
    completion, completion_sha = load_immutable(
        targets["completion"], "cebra_pseudosession_sua_cell_completion"
    )
    require(launch.get("arm") == f"mix_{arm}" and launch.get("seed") == seed,
            "launch cell identity drift")
    require(launch.get("constructibility_preflight_sha256") == EXPECTED_PREFLIGHT_SHA256,
            "launch preflight drift")
    require(completion.get("launch_receipt_sha256") == launch_sha,
            "completion/launch chain drift")
    require(completion.get("successful") is True and completion.get("return_code") == 0,
            "source training did not complete successfully")
    run_dir = targets["run"].resolve()
    metadata_path = run_dir / "run_metadata.json"
    require(metadata_path.is_file(), "source run metadata missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    a2core.validate_source_run_metadata(
        metadata, source_arm=f"source_{arm}", seed=seed
    )
    pseudo = metadata.get("cebra_pseudosession") or {}
    require(pseudo.get("screen_id") == SCREEN_ID, "pseudo-session metadata missing")
    require(pseudo.get("arm") == f"mix_{arm}" and pseudo.get("seed") == seed,
            "pseudo-session arm/seed metadata drift")
    require(pseudo.get("mix_probability") == 0.5 and pseudo.get("contributor_count") == 3,
            "pseudo-session topology drift")
    require(pseudo.get("max_behavior_residual") == 0.25, "residual gate drift")
    require(pseudo.get("constructibility_preflight_sha256") == EXPECTED_PREFLIGHT_SHA256,
            "source metadata preflight drift")
    runtime = pseudo.get("runtime_source_schedule") or {}
    require(runtime.get("examples_audited") == 1_086_007, "source schedule is not full")
    require(runtime.get("seed") == seed, "source schedule seed drift")
    require(runtime.get("mixed_examples", 0) > 0, "source schedule contains no mixing")
    require(runtime.get("accepted_endpoint_residual", {}).get("max", 1.0) <= 0.25,
            "source schedule accepted an invalid donor")
    if seed == 42:
        require(runtime.get("schedule_sha256") == EXPECTED_SEED42_SCHEDULE_SHA256,
                "seed42 runtime schedule drift")
    fingerprint = a2core.source_run_receipt_fingerprint(
        run_dir, source_arm=f"source_{arm}", seed=seed
    )
    fingerprint.update({
        "pseudo_session_schedule_sha256": runtime["schedule_sha256"],
        "launch_receipt_sha256": launch_sha,
        "completion_receipt_sha256": completion_sha,
    })
    return run_dir, metadata, fingerprint


def implementation_bindings() -> dict[str, str]:
    paths = {
        "scorer": Path(__file__).resolve(),
        "a2_scorer": SUA_ROOT / "scripts/a2_matched_subject_shift_v2_score.py",
        "a2_core": SUA_ROOT / "mc_maze/a2_matched_subject_shift_v2_core.py",
        "shared_evaluator": SUA_ROOT / "scripts/eval_adaptation_dandi688.py",
        "model_loader": SUA_ROOT / "scripts/select_gradient_free_protocol_dandi688.py",
        "pseudo_datamodule": SUA_ROOT / "mc_maze/cebra_pseudosession.py",
        "pseudo_trainer": SUA_ROOT / "scripts/train_cebra_pseudosession_sua.py",
        "base_datamodule": SUA_ROOT / "mc_maze/multisession_datamodule.py",
        "unit_side_features": SUA_ROOT / "mc_maze/unit_side_features.py",
        "streaming_module": REPO_ROOT / "streaming_calibration_exp/src/models/streaming_calibration_module.py",
        "streaming_spint": REPO_ROOT / "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "streaming_encoders": REPO_ROOT / "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def result_path(arm: str, seed: int, domain: str) -> Path:
    return RESULT_ROOT / f"{domain}_mix_{arm}_s{seed}.json"


def execute(arm: str, seed: int, domain: str, device_name: str) -> dict[str, Any]:
    import torch
    from mc_maze.multisession_datamodule import session_name_from_path

    load_preflight()
    a2_payload = validate_a2_measurement_surface()
    run_dir, metadata, fingerprint = validate_source(arm, seed)
    output = result_path(arm, seed, domain)
    require(not output.exists() and not Path(str(output) + ".sha256").exists(),
            "score output is not fresh")
    bindings_before = implementation_bindings()
    authority, behavior_stats, side_stats, formal_names, _ = a2score._fit_source_normalizers(metadata)
    domain_paths = a2score._load_domain_paths(domain)
    expected_sessions = a2core.expected_domain_sessions(domain)
    require(tuple(session_name_from_path(path) for path in domain_paths) == expected_sessions,
            "domain path roster drift")
    require(torch.cuda.is_available(), "CUDA scoring required")
    device = torch.device(device_name)
    require(device.type == "cuda", "score device must be CUDA")
    checkpoints = a2core.source_epoch_checkpoint_paths(run_dir)
    per_epoch: dict[str, dict[str, Any]] = {}
    query_receipts = None
    for epoch in a2core.EPOCH_WINDOW:
        per_session, receipts = a2score._evaluate_epoch(
            checkpoint_path=checkpoints[epoch], source_arm=f"source_{arm}",
            metadata=metadata, domain_paths=domain_paths, behavior_stats=behavior_stats,
            side_stats=side_stats, device=device,
        )
        require(tuple(per_session) == expected_sessions, "score session order drift")
        if query_receipts is None:
            query_receipts = receipts
        else:
            require(query_receipts == receipts, "query semantics drifted across epochs")
        per_epoch[str(epoch)] = {
            "checkpoint_path": str(checkpoints[epoch].resolve()),
            "checkpoint_sha256": fingerprint["source_checkpoint_sha256_bundle"][str(epoch)],
            "per_session_r2": per_session,
            "mean_r2": sum(per_session.values()) / len(per_session),
        }
    require(query_receipts is not None, "empty epoch score")
    per_session_mean = {
        session: sum(per_epoch[str(epoch)]["per_session_r2"][session]
                     for epoch in a2core.EPOCH_WINDOW) / len(a2core.EPOCH_WINDOW)
        for session in expected_sessions
    }
    bindings_after = implementation_bindings()
    require(bindings_before == bindings_after, "score implementation drifted during target access")
    payload = {
        "schema_version": 1,
        "receipt_kind": "cebra_pseudosession_sua_domain_score",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": SCREEN_ID,
        "arm": f"mix_{arm}",
        "side_feature_group": arm,
        "seed": seed,
        "domain": domain,
        "domain_sessions": list(expected_sessions),
        "mean_r2": sum(per_session_mean.values()) / len(per_session_mean),
        "per_session_mean_r2": per_session_mean,
        "per_epoch": per_epoch,
        "session_query_receipts": query_receipts,
        "source_run": fingerprint,
        "normalizer_authority": authority,
        "parent_a2_measurement_implementation_bindings_sha256":
            a2_payload["implementation_bindings_sha256"],
        "implementation_bindings": bindings_after,
        "implementation_bindings_sha256": hashlib.sha256(
            json.dumps(bindings_after, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "protocol": {
            "epoch_window": list(a2core.EPOCH_WINDOW),
            "activity_calibration_trials": 30,
            "carrier_label_trials": 30,
            "selection": "first_chronological",
            "query_start_trial": 30,
            "query_and_metric_identical_to_a2": True,
        },
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_velocity_labels_used_for_weight_updates": False,
        "target_normalizer_refit": False,
        "target_backward_gradients": False,
        "target_decoder_or_encoder_weight_updates": False,
        "formal_subc_test_nwb_opened": False,
        "formal_subc_test_names_only": list(formal_names),
        "execution_device": str(device),
    }
    write_immutable_pair(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("t4", "z4"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--domain", choices=("within_subject", "external_subject_M"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        payload = {
            "status": "DRY_RUN__NO_DATA_NO_MODEL",
            "arm": f"mix_{args.arm}", "seed": args.seed, "domain": args.domain,
            "source_paths": {name: str(path) for name, path in paths_for(args.arm, args.seed).items()},
            "output": str(result_path(args.arm, args.seed, args.domain)),
            "formal_subc_test_nwb_opened": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    result = execute(args.arm, args.seed, args.domain, args.device)
    print(json.dumps({"status": "COMPLETE", "mean_r2": result["mean_r2"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
