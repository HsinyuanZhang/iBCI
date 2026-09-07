#!/usr/bin/env python3
"""Read-only third reference: clean full-SPINT on frozen M2/M24 held-out query.

This is deliberately not a student training/evaluation path.  The full clean
teacher receives only the first 24 calibration trials' neural tensor.  The
``side_feature_group=none`` datamodule contract prevents any T4/``tgt_loc`` or
other target labels from entering calibration.  Query covariates are consumed
only as scored test targets under ``torch.inference_mode()``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torchmetrics.regression import R2Score

STREAMING_ROOT = Path(__file__).resolve().parents[1]
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from src.data.falcon_datamodule import FalconDataModule
from src.models.falcon_module import FalconLitModule


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SHA256 = "c769f39703f33e1e1f1f0c02a77d7cc9cd28356e1af2b51ec47fbc214aec24dc"
EXPECTED_SESSIONS = {
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def verify_clean_teacher(teacher: Path, receipt: Path) -> dict:
    """Reuse the one authoritative validator in an isolated streaming process."""
    validator = ROOT / "sua_exploration/scripts/validate_m2_clean_teacher_receipt.py"
    subprocess.run(
        [sys.executable, str(validator), "--checkpoint", str(teacher), "--receipt", str(receipt)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return json.loads(receipt.read_text())


def six_nwb_fingerprints() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "SPINT-main/data/000953").rglob("*held-out-calib*.nwb")):
        session = path.name.split("_")[1].split(".")[0]
        rows.append({"session": session, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    require(len(rows) == 6 and {row["session"] for row in rows} == EXPECTED_SESSIONS, "M2 held-out NWB set is not the frozen six")
    return rows


def validate_audit(audit: object) -> dict[str, dict[str, object]]:
    require(isinstance(audit, dict) and set(audit) == EXPECTED_SESSIONS, "missing exact six-session query audit")
    for session, row in audit.items():
        require(isinstance(row, dict), f"{session}: malformed query audit")
        require(row.get("support_trials") == 24 and row.get("query_start_trial") == 24, f"{session}: not M24/query24")
        require(row.get("window_size") == 50 and row.get("full_window_disjoint") is True, f"{session}: 50-bin query contract failed")
        require(int(row.get("query_trials", 0)) > 0 and int(row.get("eligible_windows", 0)) > 0, f"{session}: empty query")
        require(row.get("minimum_window_start_padded_bin") == row.get("raw_query_start_bin", -50) + 49, f"{session}: history boundary failed")
    return {str(name): dict(row) for name, row in audit.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--source-arms", nargs="+", default=["ordinary_t4", "ssc_t4"],
                        help="Source-gate arms that must bind the supplied clean teacher.")
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha", default=PROTOCOL_SHA256)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", default="clean_full_spint_third_reference_not_primary")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite reference {args.out}")
    require(sha256(args.protocol_receipt) == args.expected_protocol_sha, "frozen protocol receipt SHA drift")
    source_gate = json.loads(args.source_gate.read_text())
    require(source_gate.get("catastrophic_stop") is False and source_gate.get("next_action") == "one_frozen_local_heldout_replay_required", "source gate does not authorize reference replay")
    teacher = args.teacher_checkpoint.resolve()
    receipt = args.teacher_receipt.resolve()
    require(teacher.is_file() and receipt.is_file(), "teacher checkpoint/receipt missing")
    receipt_json = verify_clean_teacher(teacher, receipt)
    selected_teacher = receipt_json.get("selected_checkpoint", {})
    require(selected_teacher.get("sha256") == sha256(teacher), "clean teacher receipt does not bind supplied teacher")
    require(args.source_arms and len(set(args.source_arms)) == len(args.source_arms), "source arms must be nonempty and unique")
    source_teachers = [source_gate.get("arms", {}).get(arm, {}).get("teacher", {}) for arm in args.source_arms]
    require(all(row.get("checkpoint_sha256") == selected_teacher["sha256"] and row.get("receipt_sha256") == sha256(receipt) for row in source_teachers), "source arms are not bound to this clean teacher receipt")

    # This is the exact student-heldout data contract except that no side feature
    # is constructed.  Instantiation/setup is intentionally after all receipts
    # verify, because it opens the six local held-out calibration NWBs.
    datamodule = FalconDataModule(
        task="m2", data_dir=str(ROOT / "SPINT-main/data/000953"), heldin_session_names=[""],
        batch_size=32, window_size=50, calibration_n_trials=24, random_calibration=False,
        smooth_calibration=False, max_trial_length=100, standardize_covariates=False,
        use_intertrials=True, use_calib_intertrials=False, trial_feature_type="raw",
        interpolate_trials=True, interpolate_trials_kind="cubic", pad_value=-1.0,
        validation_protocol="loso", loso_fold=1, include_heldout_in_fit=False,
        include_heldout_in_test=True, query_start_trial=24, num_workers=0,
        pin_memory=False, sampler_seed=args.sampler_seed, balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False, side_feature_group="none", side_feature_shuffle_seed=args.sampler_seed,
    )
    datamodule.prepare_data()
    datamodule.setup("test")
    audit = validate_audit(datamodule.get_split_manifest().get("heldout_query_window_audit"))
    loaders = datamodule.test_dataloader()
    require(isinstance(loaders, list) and len(loaders) == 2, "reference requires held-in + held-out test loaders")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = FalconLitModule.load_from_checkpoint(str(teacher), weights_only=False, map_location="cpu")
    model.eval().to(device)
    score_by_session: dict[str, R2Score] = {}
    with torch.inference_mode():
        for batch in loaders[1]:
            neural, behavior_target, calib, session_names = batch
            require(len(set(session_names)) == 1, "held-out batch mixes sessions")
            session = str(session_names[0])
            neural, behavior_target, calib = neural.to(device), behavior_target.to(device), calib.to(device)
            prediction = model(neural, calib_trialized_neural_features=calib)
            if model.hparams.decode_last_timestep_only:
                prediction, behavior_target = prediction[:, -1:, :], behavior_target[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                prediction = prediction / model.hparams.behavior_scaling_factor
            metric = score_by_session.setdefault(session, R2Score(multioutput="variance_weighted").to(device))
            metric.update(prediction.flatten(0, 1), behavior_target.flatten(0, 1))
    require(set(score_by_session) == EXPECTED_SESSIONS, "reference did not score exactly six held-out sessions")
    scores = {session: float(metric.compute().item()) for session, metric in sorted(score_by_session.items())}
    payload = {
        "schema_version": 1,
        "role": args.role,
        "formal_heldout_evaluated": False,
        "local_heldout_calib_evaluated": True,
        "hidden_evalai_evaluated": False,
        "protocol_receipt": {"path": str(args.protocol_receipt.resolve()), "sha256": args.expected_protocol_sha},
        "source_gate": {"path": str(args.source_gate.resolve()), "sha256": sha256(args.source_gate)},
        "clean_teacher_receipt": {"path": str(receipt), "sha256": sha256(receipt), "selected_teacher_checkpoint_sha256": selected_teacher["sha256"]},
        "descriptor_contract": {"calibration_trials": 24, "query_start_trial": 24, "window_size_bins": 50, "side_feature_group": "none", "calibration_target_labels_used": False, "backward_gradients": False, "optimizer_updates": False, "checkpoint_selection": False},
        "six_heldout_calibration_nwbs": six_nwb_fingerprints(),
        "query_window_audit": audit,
        "per_session_r2": scores,
        "mean_r2_equal_session": sum(scores.values()) / len(scores),
        "evaluation_disclosure": "Query covariates are loaded only as scored targets; no target labels or covariates enter the M24 calibration tensor.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
