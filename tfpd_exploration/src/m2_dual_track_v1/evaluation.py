"""Disjoint ext-4 evaluator. Development evidence only.

``Candidate.forward_last`` is decoder_raw.  This module divides by
``plan.BEHAVIOR_SCALE`` (5.0) before variance-weighted R2 against native
covariates.  Never subtract 0.360449 or 0.349453 from the new scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from . import champion, contracts, data, plan


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prediction_digest(values: np.ndarray) -> str:
    return champion.array_sha256(np.ascontiguousarray(values, dtype=np.float32))


def decoder_raw_to_native(prediction_raw: np.ndarray | torch.Tensor) -> np.ndarray:
    """The only authorized /5. Applied at score time, never inside Candidate.forward_last."""
    array = np.asarray(
        prediction_raw.detach().cpu().numpy() if hasattr(prediction_raw, "detach") else prediction_raw,
        dtype=np.float64,
    )
    return np.ascontiguousarray(array / plan.BEHAVIOR_SCALE, dtype=np.float32)


class DualTrackEvaluator:
    """Evaluator(candidate, scoring_manifest) -> per-session R2 + prediction digest."""

    def __init__(self, *, device: str | torch.device = "cpu", batch_size: int = 32) -> None:
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

    def _banks(self, scoring_manifest: Mapping[str, Any]) -> dict[str, contracts.SessionBank]:
        surface = str(scoring_manifest.get("surface", "ext4"))
        sessions = tuple(scoring_manifest.get("sessions") or (
            plan.EXT4_SESSIONS if surface == "ext4" else plan.HELDIN_SESSIONS
        ))
        return {
            session: data.load_session_bank(surface, session, device="cpu")
            for session in sessions
        }

    def __call__(
        self,
        candidate: contracts.Candidate,
        scoring_manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        plan.require(
            getattr(candidate, "training_target_space", None) == plan.TRAINING_TARGET_SPACE,
            "candidate must declare decoder_raw",
        )
        banks = self._banks(scoring_manifest)
        per_session: dict[str, float] = {}
        rows: dict[str, Any] = {}
        candidate.eval()
        for session, bank in banks.items():
            targets: list[np.ndarray] = []
            preds: list[np.ndarray] = []
            window_ids: list[int] = []
            for batch in data.iter_session_batches(
                bank,
                batch_size=self.batch_size,
                device=self.device,
                target_space=plan.SCORING_TARGET_SPACE,
            ):
                x = batch.X
                with torch.inference_mode():
                    raw = candidate.forward_last(x, batch.bank, batch.unit_mask)
                native_pred = decoder_raw_to_native(raw)
                native_target = batch.last_target.detach().cpu().numpy().astype(np.float32, copy=False)
                preds.append(native_pred)
                targets.append(native_target)
                window_ids.extend(batch.window_ids)
            target = np.concatenate(targets, axis=0)
            prediction = np.concatenate(preds, axis=0)
            r2 = contracts.variance_weighted_r2(target, prediction)
            per_session[session] = float(r2)
            rows[session] = {
                "session_id": session,
                "r2": float(r2),
                "window_count": int(target.shape[0]),
                "prediction_digest": prediction_digest(prediction),
                "target_digest": prediction_digest(target),
                "window_ids": window_ids,
                "scoring_space": plan.SCORING_TARGET_SPACE,
                "divided_by_behavior_scale": True,
            }
        summary = contracts.summarize_sessions(per_session)
        dates = contracts.date_equal_mean(per_session)
        return {
            "schema": plan.SCHEMA,
            "contract_version": plan.CONTRACT_VERSION,
            "candidate": getattr(candidate, "name", type(candidate).__name__),
            "surface": scoring_manifest.get("surface", "ext4"),
            "development_evidence": True,
            "forbidden_baseline_subtractions": list(plan.FORBIDDEN_BASELINE_SUBTRACTIONS),
            "subtracted_historical_six_session": False,
            "subtracted_official_ho": False,
            "behavior_scale": plan.BEHAVIOR_SCALE,
            "divide_applied_at": "evaluator_only",
            "per_session": rows,
            "summary": summary,
            "date_sensitivity": dates,
            "R_session_equal_mean": summary["equal_session_mean"],
            "R_date_equal_mean": dates["equal_date_mean"],
        }


def gpu_ref_command(*, gpu_index: int = 0) -> str:
    return (
        "PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID "
        f"CUDA_VISIBLE_DEVICES={gpu_index} "
        f"{plan.PYTHON} "
        "tfpd_exploration/scripts/run_m2_dual_track_v1.py score-ref --device cuda:0"
    )


def score_reference(*, device: str = "cpu") -> int:
    """CLI hook. Writes stage0/ref_clean.json with R_REF_clean. Development evidence."""
    dest = plan.active_run_root() / "stage0"
    dest.mkdir(parents=True, exist_ok=True)
    command = gpu_ref_command(gpu_index=0)
    (dest / "run_ref_gpu0.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "cd /home/xinyuan/Work_host/SPINT\n"
        "# Coordinator only. Requires a GPU0 lease. Pipeline worker does not occupy a GPU.\n"
        f"{gpu_ref_command(gpu_index=0)}\n",
        encoding="utf-8",
    )
    cache_meta = data.cache_root() / "meta.json"
    if not cache_meta.is_file():
        _write_json(
            dest / "ref_clean.json",
            {
                "status": "PREPARED_NOT_SCORED",
                "reason": "cache/meta.json missing; run stage0_data() first",
                "gpu_command": command,
                "R_REF_clean": None,
                "development_evidence": True,
            },
        )
        print(json.dumps({"status": "PREPARED_NOT_SCORED", "gpu_command": command}, indent=2))
        return 0

    try:
        cand = champion.load_frozen_champion(device=device)
        evaluator = DualTrackEvaluator(device=device, batch_size=32)
        report = evaluator(cand, data.scoring_manifest(surface="ext4"))
        payload = {
            "status": "SCORED",
            "device": device,
            "R_REF_clean": report["R_session_equal_mean"],
            "R_REF_clean_date_equal_mean": report["R_date_equal_mean"],
            "development_evidence": True,
            "subtracted_0_360449": False,
            "subtracted_0_349453": False,
            "report": report,
            "champion_authority": cand.authority,
            "gpu_command": command,
        }
        _write_json(dest / "ref_clean.json", payload)
        print(json.dumps({
            "status": "SCORED",
            "R_REF_clean": payload["R_REF_clean"],
            "device": device,
        }, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        _write_json(
            dest / "ref_clean.json",
            {
                "status": "PREPARED_NOT_SCORED",
                "reason": f"{type(exc).__name__}: {exc}",
                "gpu_command": command,
                "R_REF_clean": None,
                "development_evidence": True,
            },
        )
        print(json.dumps({
            "status": "PREPARED_NOT_SCORED",
            "reason": str(exc),
            "gpu_command": command,
        }, indent=2))
        return 0
