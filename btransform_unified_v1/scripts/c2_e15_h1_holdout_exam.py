"""H1 holdout-date (1925-01-20) exam-face eval of the frozen C2 e15 system.

Level-1 fair-comparison eval per ADDENDUM_FAIR_COMPARISON_H1_20260906.md.
Evaluator-only replication of the fixed-reference path
(tfpd_exploration/src/h1_optimized_v2/c2_reference.py) restricted to the LODO
holdout date's minival eval_mask (session, end) coordinates (2,952 windows —
bitwise the same coordinate rule as the M-F250 exam face):

  - immutable deployment package evalai_h1_c2_ho_epoch15_v1 (sha 91ef13cc...)
    carrying the C2 epoch-15 state (ckpt sha ce46267e...)
  - native streaming contract: one API predict() call per chronological bin,
    no reset at trial boundaries; score only eval_mask bins
  - window: native W=700 streaming (M-F250 used L=250; window definitions
    differ across systems — part of the compared system, P0-1)
  - exposure audit: the C2 training receipt chain (terminal.json epoch-15
    entry + source_authority sha binding) must list the 01-20 sessions among
    the trained held-in calib recordings

No training, no EvalAI, no modification of any historical root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path("/home/xinyuan/Work_host/SPINT")
SPINT_MAIN = WORKSPACE / "SPINT-main"
if str(SPINT_MAIN) not in sys.path:
    sys.path.insert(0, str(SPINT_MAIN))

PACKAGE = WORKSPACE / "tfpd_exploration/submissions/evalai_h1_c2_ho_epoch15_v1/artifacts/decoder.pt"
PACKAGE_SHA256 = "91ef13cc94b9ab865c8f926dcbb6d33e9628bf9cbc9b757665d10e33cfda144a"
CHECKPOINT_SHA256 = "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215"
SOURCE_CACHE = (
    WORKSPACE / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/source_cache.pt"
)
SOURCE_CACHE_SHA256 = "51ff9ebfcd10a032f9c173ec426bfb4c421b751502271577582239c51bcc91b4"
C2_TRAINING_ROOT = Path(
    "/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/h1_series_20260830/"
    "results/h1_cal_aug_m3_aware_dual_selection_v2_a1"
)

HOLDOUT_SESSIONS = ("ses-19250120T115044", "ses-19250120T115537")
RESET_TAG_FMT = "sub-HumanPitt-held-in-minival_{session}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Per-column-mean centered variance-weighted R2 (C2 reference convention)."""
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    denominator = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("zero/nonfinite target variance")
    return float(1.0 - np.square(pred - target).sum() / denominator)


def _exposure_audit() -> dict:
    """Bind the C2 e15 training receipt chain and read its session exposure."""
    terminal_path = C2_TRAINING_ROOT / "training/c2/terminal.json"
    authority_path = C2_TRAINING_ROOT / "source_authority/authority.json"
    ckpt_entry = None
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    for entry in terminal["checkpoints"]:
        if int(entry["epoch_zero_based"]) == 15:
            ckpt_entry = entry
            break
    if ckpt_entry is None:
        raise RuntimeError("C2 terminal.json lacks epoch 15")
    ckpt_sha_ok = str(ckpt_entry["sha256"]) == CHECKPOINT_SHA256
    ckpt_step_ok = int(ckpt_entry["global_step"]) == 66128
    authority_sha = _sha256_file(authority_path)
    authority_sha_ok = authority_sha == str(terminal["source_authority_sha256"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    sessions = sorted({str(e["session"]) for e in authority["m3_carrier_entries"]})
    holdout_in_train = {s: (s in sessions) for s in HOLDOUT_SESSIONS}
    exposed = all(holdout_in_train.values())
    return {
        "verdict": "exposed" if exposed else "clean_or_unknown",
        "training_receipt": {
            "terminal_json": str(terminal_path),
            "terminal_schema": terminal.get("schema"),
            "terminal_status": terminal.get("status"),
            "config_sha256": terminal.get("config_sha256"),
            "finished_at_utc": terminal.get("finished_at_utc"),
        },
        "epoch15_entry": ckpt_entry,
        "epoch15_sha_matches_frozen_ckpt": bool(ckpt_sha_ok and ckpt_step_ok),
        "source_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
            "sha_matches_terminal_binding": bool(authority_sha_ok),
            "heldin_calib_recordings_opened": authority.get("heldin_calib_recordings_opened"),
            "heldin_minival_recordings_opened": authority.get("heldin_minival_recordings_opened"),
            "heldout_calib_recordings_opened": authority.get("heldout_calib_recordings_opened"),
            "c2_prefix_cycle": authority.get("c2_cycle"),
            "trained_sessions": sessions,
            "holdout_date_sessions_in_train": holdout_in_train,
        },
        "method": (
            "C2 e15 ckpt sha ce46267e... bound to terminal.json epoch-15 entry; terminal's "
            "source_authority_sha256 binds the training source authority whose "
            "m3_carrier_sessions enumerate the trained held-in recordings"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from falcon_challenge.config import FalconConfig, FalconTask
    from third_party.falcon_challenge.h1_epfilm_spint_decoder import H1EPFiLMSpintDecoder

    started = time.monotonic()
    if _sha256_file(PACKAGE) != PACKAGE_SHA256:
        raise RuntimeError("immutable C2 epoch-15 package/checkpoint hash drift")
    cache_sha = _sha256_file(SOURCE_CACHE)
    if cache_sha != SOURCE_CACHE_SHA256:
        raise RuntimeError("source cache drift against frozen authority")
    cache = torch.load(SOURCE_CACHE, map_location="cpu", weights_only=False)
    exposure = _exposure_audit()
    if exposure["verdict"] != "exposed":
        raise RuntimeError("C2 exposure audit did not close; refusing to run unlabeled")

    decoder = H1EPFiLMSpintDecoder(FalconConfig(task=FalconTask.h1), PACKAGE, batch_size=1, device=args.device)

    rows = []
    all_pred: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    arrays_for_export: dict[str, np.ndarray] = {}
    parity_rows = []
    for session in HOLDOUT_SESSIONS:
        row = cache["minival"][session]
        tag = RESET_TAG_FMT.format(session=session)
        decoder.reset([Path(tag)])
        with torch.inference_mode():
            native_identity = decoder._film_identity()[0].detach().cpu()
        cached_identity = row["bank"]["E0"].detach().cpu()
        cached_carrier = row["bank"]["T"].detach().cpu()
        package_carrier = decoder.local_carrier[0].detach().cpu()
        identity_max_abs = float((native_identity - cached_identity).abs().max())
        carrier_max_abs = float((package_carrier - cached_carrier).abs().max())
        if identity_max_abs > 1.0e-5 or carrier_max_abs > 1.0e-6:
            raise RuntimeError(f"C2 support/bank mismatch for {session}")
        parity_rows.append({
            "session": session,
            "dataset_tag": tag,
            "payload_key": decoder.local_keys[0],
            "identity_max_abs": identity_max_abs,
            "carrier_max_abs": carrier_max_abs,
        })
        decoder.reset([Path(tag)])
        score_mask = np.asarray(row["eval_mask"], dtype=bool)
        ends = np.flatnonzero(score_mask).astype(np.int64)
        neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
        velocity = np.ascontiguousarray(row["velocity"], dtype=np.float32)
        prediction: list[np.ndarray] = []
        target: list[np.ndarray] = []
        for index in range(len(neural)):
            # predict() both observes and returns the stateful W700 estimate.
            value = decoder.predict(neural[index][None])[0]
            if score_mask[index]:
                prediction.append(np.asarray(value, dtype=np.float32))
                target.append(velocity[index])
        part_prediction = np.asarray(prediction, dtype=np.float32)
        part_target = np.asarray(target, dtype=np.float32)
        if len(part_prediction) != int(score_mask.sum()):
            raise RuntimeError(f"C2 stream count mismatch for {session}")
        if not np.isfinite(part_prediction).all():
            raise RuntimeError("nonfinite C2 prediction")
        all_pred.append(part_prediction)
        all_target.append(part_target)
        arrays_for_export[f"pred__{session}"] = part_prediction
        arrays_for_export[f"target__{session}"] = part_target
        arrays_for_export[f"ends__{session}"] = ends
        rows.append({
            "session": session,
            "reset_tag": tag,
            "n": int(len(ends)),
            "n_stream_bins": int(len(neural)),
            "window": 700,
            "ends_sha256": _array_digest(ends),
            "target_sha256": _array_digest(part_target),
            "pred_sha256": _array_digest(part_prediction),
            "r2_percolumn_mean0": _r2(part_prediction, part_target),
        })

    pred = np.concatenate(all_pred)
    target = np.concatenate(all_target)
    per_session = [row["r2_percolumn_mean0"] for row in rows]
    result = {
        "schema": "c2_e15_h1_holdout_exam_exposed_eval_v1",
        "system": "C2 e15 (frozen deployment package, ckpt ce46267e...)",
        "task": "h1",
        "surface": "LODO holdout date 1925-01-20 exam (minival eval_mask (session,end) coordinates)",
        "holdout_sessions": list(HOLDOUT_SESSIONS),
        "n_points": int(len(target)),
        "window": 700,
        "window_note": (
            "C2 scored with its native W=700 stateful streaming (one predict per chronological "
            "bin, no trial-boundary reset); M-F250 exam used L=250. Window definitions differ "
            "across systems (P0-1) — the join locks only (session,end)."
        ),
        "coordinate_rule": "ends = flatnonzero(eval_mask) per session from the frozen source_cache (M-F250 exam rule)",
        "package": str(PACKAGE),
        "package_sha256": PACKAGE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_cache_sha256": cache_sha,
        "device": args.device,
        "streaming_contract": "one API predict call per chronological bin; no reset at trial boundary; score only frozen mask",
        "bank_parity": parity_rows,
        "exposure": exposure,
        "r2_convention": "per-column mean(0) centered variance-weighted (C2 reference convention)",
        "pooled_r2": _r2(pred, target),
        "equal_session_mean_r2": float(np.mean(per_session)),
        "sessions": rows,
        "elapsed_seconds": time.monotonic() - started,
        "status": "COMPLETE_INFERENCE_ONLY",
        "parameter_updates": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.out)
    npz_path = args.out.with_name(args.out.stem + "_arrays.npz")
    np.savez_compressed(npz_path, **arrays_for_export)
    print(json.dumps({
        "status": result["status"], "n_points": result["n_points"],
        "pooled_r2": result["pooled_r2"], "equal_session_mean_r2": result["equal_session_mean_r2"],
        "exposure": result["exposure"]["verdict"], "elapsed_s": result["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
