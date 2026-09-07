#!/usr/bin/env python3
"""REF-SHUF replay for FABLE TKD M2 v1 (Wave 2 spec section 5).

1. Reproduce the probe p0 arm (champion ckpt + zero FiLM, exact probe code
   path) on the external face; require equal_session_mean == SEALED_M30_EXTERNAL
   within 1e-6 (face-drift guard; G3 calibration is untrustworthy otherwise).
2. Swap the student checkpoint for the sealed ts4 whole-row-permutation ckpt
   (sha e385e2f4...), score the SAME external face with the ts4 permutation
   applied to the normalized T4 side rows (falcon_datamodule
   deterministic_row_permutation law, seed 42, session-scoped), and report
   Delta_ref = SEALED - ts4_mean (the G3 identity-gate calibration quantity).

GPU1 (same arithmetic environment as the probe) with full preflight.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "tfpd_exploration" / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

TS4_CKPT = (
    REPO_ROOT / "streaming_calibration_exp/outputs/streaming_calibration/"
    "e8_ts4_m2_submission_control_m33q33_v1_s42_20260801_162010/checkpoints/best.ckpt"
)


def gpu1_preflight() -> dict[str, object]:
    def query(arguments: list[str]) -> str:
        completed = subprocess.run(arguments, check=True, text=True,
                                   capture_output=True, timeout=30)
        return completed.stdout

    rows = query(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                  "--format=csv,noheader,nounits"])
    target = None
    for line in rows.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts[0] == "1":
            target = {"uuid": parts[1], "util": float(parts[2]), "mem": float(parts[3])}
    if target is None:
        raise SystemExit("GPU index 1 absent")
    if target["uuid"] != "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86":
        raise SystemExit(f"GPU1 UUID drift: {target['uuid']}")
    if target["util"] > 5.0 or target["mem"] > 500.0:
        raise SystemExit(f"GPU1 busy: util {target['util']}% mem {target['mem']}MiB")
    apps = query(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                  "--format=csv,noheader"])
    pids = [[p.strip() for p in line.split(",")]
            for line in apps.strip().splitlines() if line.strip()]
    import os

    own_pid = os.getpid()
    foreign = [row for row in pids
               if row[1] == target["uuid"] and int(row[0]) != own_pid]
    if foreign:
        raise SystemExit(f"GPU1 has foreign compute apps: {[r[0] for r in foreign]}")
    return {"gpu1_uuid": target["uuid"], "util": target["util"],
            "mem_mib": target["mem"], "checked_at_utc":
            datetime.now(timezone.utc).isoformat()}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from tfpd_exploration.src.fable_tkd_m2_v1 import plan

    if not args.execute:
        print(json.dumps({
            "dry": True,
            "steps": ["p0 reproduction (champion, probe law, |err|<=1e-6)",
                      "ts4 ckpt swap (sha e385e2f4...)",
                      "external face with ts4 row permutation",
                      "Delta_ref = SEALED - ts4_mean"],
            "sealed_m30_external": plan.SEALED_M30_EXTERNAL,
            "ts4_ckpt": str(TS4_CKPT),
        }, indent=2))
        return 0

    preflight = gpu1_preflight()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan.require(torch.backends.cuda.matmul.allow_tf32 is False, "TF32 not off")
    plan.require(torch.cuda.is_available(), "GPU1 unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    root = plan.result_root(REPO_ROOT)
    receipt_path = root / "ref_shuf_replay.json"

    def _attempt_slot() -> Path:
        """O_EXCL slot; retry only after a recorded failure (D4 convention)."""
        first = root / "ref_shuf_attempt.json"
        if not first.exists():
            return first
        if not (root / "ref_shuf_failure.json").exists():
            raise SystemExit(f"refusing: {receipt_path.name} sealed without failure")
        index = 1
        while (root / f"ref_shuf_attempt_r{index}.json").exists():
            index += 1
        return root / f"ref_shuf_attempt_r{index}.json"

    def _failure(error: BaseException) -> None:
        target = root / "ref_shuf_failure.json"
        if target.exists():
            index = 1
            while (root / f"ref_shuf_failure_r{index}.json").exists():
                index += 1
            target = root / f"ref_shuf_failure_r{index}.json"
        plan.atomic_receipt(target, {
            "schema": f"{plan.SCHEMA}:ref_shuf_failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })

    try:
        plan.require(not receipt_path.exists(), f"receipt already sealed: {receipt_path}")
        plan.atomic_receipt(_attempt_slot(), {
            "schema": f"{plan.SCHEMA}:ref_shuf_attempt",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_preflight": preflight,
            "ts4_ckpt": str(TS4_CKPT),
        }, exclusive=True)

        from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

        model, data_module, _task_config, metadata = load_frozen_model_and_data()
        plan.require(metadata["checkpoint_sha256"] == plan.CHAMPION_CKPT_SHA256,
                     "champion checkpoint drift")
        plan.require(metadata["normalization_sha256"]
                     == plan.CHAMPION_NORMALIZATION_SHA256, "normalizer drift")
        student = model.student.to(device)
        original_encoder = copy.deepcopy(student.id_encoder)

        external = data_module.val_heldout_dataset
        plan.require(external is not None, "external dataset missing")

        # ---- step 1: p0 reproduction via the probe's exact code path ----
        from tfpd_exploration.src.m2_hold_film_probe_v1 import physical as probe_physical

        probe_physical.install_film(student, device)
        p0 = probe_physical.score_arm(
            student=student,
            datasets={"external_official_query": external},
            device=device,
            batch_size=1024,
            shuffle=False,
            arm="p0_replay",
        )
        p0_mean = float(
            p0["summaries"]["external_official_query"]["equal_session_mean"]
        )
        p0_error = abs(p0_mean - plan.SEALED_M30_EXTERNAL)
        plan.require(p0_error <= plan.P0_R2_TOLERANCE,
                     f"p0 reproduction drift {p0_error} > {plan.P0_R2_TOLERANCE}")

        # ---- step 2: restore champion student, swap in the ts4 ckpt ----
        student.id_encoder = original_encoder
        ts4_sha = sha_file(TS4_CKPT)
        plan.require(ts4_sha == plan.TS4_CKPT_SHA256, f"ts4 ckpt drift: {ts4_sha}")
        manifest_path = TS4_CKPT.parent.parent / "checkpoint_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plan.require(manifest.get("artifact_checkpoint_sha256") == ts4_sha,
                     "ts4 manifest does not bind the artifact")
        ts4_state = torch.load(TS4_CKPT, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(ts4_state, strict=True)
        model.student.to(device)
        model.student.eval()
        plan.require(model.student.decoder_mode == "coupled", "ts4 student not coupled")
        plan.require(model.student._decoder_frozen, "ts4 decoder unexpectedly trainable")

        from src.data.falcon_t4_features import deterministic_row_permutation
        from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as screen_core
        from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import physical as screen_physical

        # ---- step 3: ts4 on the same external face, permuted side rows ----
        ts4_rows: list[dict[str, object]] = []
        ts4_values: dict[str, float] = {}
        with torch.inference_mode():
            for session in sorted(external.calib_trialized_neural_features):
                calibration = np.asarray(
                    external.calib_trialized_neural_features[session], dtype=np.float32
                )
                activity = np.ascontiguousarray(
                    calibration[: plan.ACTIVITY_HORIZON], dtype=np.float32
                )
                selected = np.arange(plan.ACTIVITY_HORIZON, dtype=np.int64)
                side, ridge_evidence = screen_physical._ridge_side(external, session, selected)
                perm = deterministic_row_permutation(
                    plan.CHANNELS, session_name=session, seed=42
                )
                plan.require(not bool((perm == np.arange(plan.CHANNELS)).all()),
                             f"ts4 perm identity for {session}")
                side = np.ascontiguousarray(side[perm], dtype=np.float32)
                support = torch.from_numpy(activity).unsqueeze(0).to(device)
                side_tensor = torch.from_numpy(side).unsqueeze(0).to(device)
                identity = student.compute_identity(support, side_features=side_tensor)
                starts = np.asarray(
                    [start for name, start in external.window_indices if name == session],
                    dtype=np.int64,
                )
                predictions: list[np.ndarray] = []
                targets: list[np.ndarray] = []
                for offset in range(0, starts.size, 1024):
                    chunk = starts[offset : offset + 1024]
                    neural_np = np.stack(
                        [external.neural_data[session][start : start + plan.WINDOW]
                         for start in chunk], axis=0
                    ).astype(np.float32, copy=False)
                    target_np = np.stack(
                        [external.covariate_data[session][start + plan.WINDOW - 1]
                         for start in chunk], axis=0
                    ).astype(np.float32, copy=False)
                    neural = torch.from_numpy(neural_np).to(device)
                    prediction, _ = student(neural, identity=identity)
                    predictions.append(
                        prediction[:, -1, :].cpu().numpy().astype(np.float32)
                        / np.float32(plan.BEHAVIOR_SCALE)
                    )
                    targets.append(target_np)
                prediction_np = np.ascontiguousarray(np.concatenate(predictions, axis=0))
                target_np = np.ascontiguousarray(np.concatenate(targets, axis=0))
                r2 = screen_core.variance_weighted_r2(target_np, prediction_np)
                ts4_values[session] = r2
                ts4_rows.append({
                    "session": session,
                    "window_count": int(starts.size),
                    "ordered_window_starts_sha256": screen_core.array_sha256(starts),
                    "target_sha256": screen_core.array_sha256(target_np),
                    "prediction_sha256": screen_core.array_sha256(prediction_np),
                    "perm_sha256": screen_core.array_sha256(perm),
                    "r2": r2,
                    "ridge": ridge_evidence,
                })
        ts4_summary = screen_core.summarize_sessions(ts4_values)
        ts4_mean = float(ts4_summary["equal_session_mean"])
        delta_ref = plan.SEALED_M30_EXTERNAL - ts4_mean
        contrast = screen_core.paired_contrast(
            ts4_values, p0["session_maps"]["external_official_query"]
        )

        plan.atomic_receipt(receipt_path, {
            "schema": f"{plan.SCHEMA}:ref_shuf_replay",
            "status": "TERMINAL",
            "gpu": {"preflight": preflight, "device": "cuda:0 (GPU1)",
                    "name": torch.cuda.get_device_name(0)},
            "champion_ckpt_sha256": plan.CHAMPION_CKPT_SHA256,
            "ts4_ckpt_sha256": ts4_sha,
            "normalization_sha256": metadata["normalization_sha256"],
            "teacher_sha256": metadata["teacher_checkpoint_sha256"],
            "p0_reproduction": {
                "equal_session_mean": p0_mean,
                "sealed_m30_external": plan.SEALED_M30_EXTERNAL,
                "abs_error": p0_error,
                "tolerance": plan.P0_R2_TOLERANCE,
                "per_session_r2":
                    p0["summaries"]["external_official_query"]["per_session_r2"],
            },
            "ts4_external": {
                "equal_session_mean": ts4_mean,
                "per_session_r2": ts4_summary["per_session_r2"],
                "rows": ts4_rows,
            },
            "delta_ref": delta_ref,
            "delta_ref_formula": "SEALED_M30_EXTERNAL - ts4_equal_session_mean",
            "ts4_minus_p0_paired": contrast,
            "g3_rule": (
                "if delta_ref >= 0.20 the identity gate stays A'-SHUF >= +0.10 "
                "and A'-POOL >= +0.10; otherwise the gate becomes "
                "A'-SHUF >= delta_ref/2 and A'-POOL >= delta_ref/2 (workorder G3)"
            ),
        })
        print(json.dumps({
            "p0_mean": p0_mean, "p0_abs_error": p0_error,
            "ts4_mean": ts4_mean, "delta_ref": delta_ref,
            "receipt": str(receipt_path),
        }, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001
        _failure(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
