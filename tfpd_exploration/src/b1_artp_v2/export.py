"""Build and locally verify the six-date ARTP-P payload without hidden queries."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from tfpd_exploration.src.b1_sfcj_v1 import data
from tfpd_exploration.src.b1_sfcj_v1.constants import ALL_DATES, HELD_IN_DATES
from tfpd_exploration.src.b1_sfcj_v1.metric import official_metric_from_trials
from tfpd_exploration.src.b1_sfcj_v1.util import sha256_array, sha256_bytes, sha256_file

from .core import build_profiled_payload
from .decoder import B1ARTPProfileDecoder


def build_six_date_payloads() -> tuple[dict, dict]:
    payloads = {}
    rows = []
    for date in ALL_DATES:
        trials = data.first_m3(date)
        payloads[date] = build_profiled_payload(trials)
        paths = sorted(set(trial.path for trial in trials))
        rows.append(
            {
                "date": date,
                "n_calibration_trials": 3,
                "calibration_trial_indices": [int(trial.trial_index) for trial in trials],
                "calibration_file": paths,
                "calibration_file_sha256": [sha256_file(Path(path)) for path in paths],
                "payload_array_sha256": dict(payloads[date]["array_sha256"]),
                "query_labels_used": False,
            }
        )
    if set(payloads) != set(ALL_DATES):
        raise RuntimeError("six-date payload roster mismatch")
    return payloads, {
        "schema": "b1-artp-profile-payload-authority-v2",
        "dates": list(ALL_DATES),
        "rows": rows,
        "held_out_reads": "released held-out-calib M3 only",
        "hidden_query_reads": False,
        "model_updates": 0,
    }


def serialize_payloads(payloads: dict) -> bytes:
    envelope = {
        "schema": "b1-artp-profile-six-date-envelope-v2",
        "method": "ARTP-P",
        "payloads": payloads,
    }
    return pickle.dumps(envelope, protocol=4)


def deserialize_payloads(body: bytes) -> dict:
    envelope = pickle.loads(body)
    if envelope.get("schema") != "b1-artp-profile-six-date-envelope-v2":
        raise RuntimeError("payload envelope schema mismatch")
    if set(envelope.get("payloads", {})) != set(ALL_DATES):
        raise RuntimeError("payload envelope date roster mismatch")
    return envelope["payloads"]


def local_source_parity(payloads: dict, source_screen: dict) -> dict:
    expected = {
        row["date"]: row["systems"]["ARTP-P"] for row in source_screen["rows"]
    }
    decoder = B1ARTPProfileDecoder(payloads)
    rows = []
    for date in HELD_IN_DATES:
        decoder.reset([date])
        calibration = data.calib_trials(date)
        query = calibration[3:] + data.minival_trials(date)
        predictions = [decoder.predict(trial.tx.T[None]) for trial in query]
        targets = [trial.spectrogram for trial in query]
        score = official_metric_from_trials(predictions, targets)
        prediction_sha256 = sha256_array(np.stack(predictions))
        row = {
            "date": date,
            "n_query": len(query),
            "mse_mean": float(score["MSE Mean"]),
            "prediction_sha256": prediction_sha256,
            "expected_mse_mean": float(expected[date]["mse_mean"]),
            "expected_prediction_sha256": expected[date]["prediction_sha256"],
            "mse_exact": bool(float(score["MSE Mean"]) == float(expected[date]["mse_mean"])),
            "prediction_sha256_exact": bool(
                prediction_sha256 == expected[date]["prediction_sha256"]
            ),
        }
        if not row["mse_exact"] or not row["prediction_sha256_exact"]:
            raise RuntimeError(f"decoder/source-screen mismatch for {date}: {row}")
        rows.append(row)
    return {
        "schema": "b1-artp-profile-local-source-parity-v2",
        "rows": rows,
        "all_exact": True,
        "query_label_access_count": decoder.query_label_access_count,
        "model_updates": decoder.model_updates,
    }


def export_and_verify(*, artifact: Path, receipt: Path, source_screen_path: Path) -> dict:
    if artifact.exists() or receipt.exists():
        raise FileExistsError(f"payload output already exists: {artifact} / {receipt}")
    source_body = source_screen_path.read_bytes()
    source_screen = json.loads(source_body)
    if source_screen.get("schema") != "b1-artp-profile-source-screen-v2" or not source_screen.get(
        "all_primary_gates_pass"
    ):
        raise RuntimeError("source screen is missing or did not pass")
    payloads, authority = build_six_date_payloads()
    body = serialize_payloads(payloads)
    restored = deserialize_payloads(body)
    parity = local_source_parity(restored, source_screen)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(artifact.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(artifact)
    result = {
        "schema": "b1-artp-profile-package-receipt-v2",
        "status": "LOCAL_PACKAGE_COMPLETE_NOT_SUBMITTED",
        "artifact": str(artifact),
        "artifact_sha256": sha256_bytes(body),
        "artifact_size": len(body),
        "source_screen": str(source_screen_path),
        "source_screen_sha256": sha256_bytes(source_body),
        "payload_authority": authority,
        "local_source_parity": parity,
        "evalai_submitted": False,
    }
    temporary_receipt = receipt.with_suffix(receipt.suffix + ".tmp")
    temporary_receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary_receipt.replace(receipt)
    return result
