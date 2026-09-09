#!/usr/bin/env python3
"""Replay all 24 sealed M1 B_ACTIVITY_ONLY EMA checkpoints on HO3 calibration.

This does not train, mutate the formal run, or open an official-test surface.
It retains prediction, target, and padded-window identifiers for the current
channel-centered metric, while preserving the old score as authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
WS = V2.parent
V1 = WS / "btransform_unified_v1"
for entry in (V2 / "src", V2 / "scripts", V1 / "src", V1 / "scripts", WS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.r2 import variance_weighted_r2
import rift_v1.m1_joint_train as baseline

SCHEMA = "carrier_v4_m1_activity_only_baseline_replay_v1"
ARM = "B_ACTIVITY_ONLY"
BASELINE_DEFAULT = V2 / "results/rift_v1/m1_r100_joint_b_s42_formal_v1"
D_REPLAY_DEFAULT = V2 / "results/m1_muscle_r100_v1/baseline_replay/replay_receipt.json"
EPOCHS = tuple(range(1, baseline.EPOCHS + 1))


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def validate_formal(run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Bind B-only full training, its 24 checkpoint bytes, and its old scan."""
    meta, train, score = (read_json(run / name) for name in ("run_meta.json", "train_receipt.json", "score_receipt.json"))
    expected = {"schema": "m1_rift_joint_train_v1", "status": "FORMAL", "cell": baseline.CELL, "task": "m1", "arm": ARM,
                "seed": 42, "sampler_seed": 42, "epochs": 24, "context_bins": 100, "depth": 4,
                "layer_windows": [25, 25, 25, 24], "official_test_used": False}
    need(all(meta.get(key) == value for key, value in expected.items()), "B formal metadata identity mismatch")
    need(meta.get("source_hashes") == baseline._source_hashes(), "B formal source-code binding drift")
    source = meta.get("source_contract", {})
    need(tuple(source.get("sessions", ())) == baseline.SOURCE_SESSIONS and source.get("total_windows") == baseline.EXPECTED_WINDOWS and source.get("updates_per_epoch") == baseline.UPDATES_PER_EPOCH, "B source4/R100 contract drift")
    b3 = meta.get("b3s", {})
    need(b3.get("encoder") == "SideFeatureEarlyPoolEncoder" and b3.get("decoder_weights_copied") is False and b3.get("side_columns_initialized_zero") is True, "B live B3 encoder initialization contract drift")
    expected_train = {"schema": "m1_rift_joint_train_receipt_v1", "status": "COMPLETED", "cell": baseline.CELL, "arm": ARM,
                      "seed": 42, "sampler_seed": 42, "epochs": 24, "steps": 24 * baseline.UPDATES_PER_EPOCH,
                      "source_hashes": meta["source_hashes"], "source_contract": source, "b3s": b3}
    need(all(train.get(key) == value for key, value in expected_train.items()), "B completed training receipt mismatch")
    expected_score = {"schema": "m1_rift_joint_ho_calib_epoch_scan_v1", "status": "COMPLETED", "cell": baseline.CELL, "arm": ARM,
                      "seed": 42, "sampler_seed": 42, "source_hashes": meta["source_hashes"], "source_contract": source,
                      "b3s": b3, "official_test_used": False}
    need(all(score.get(key) == value for key, value in expected_score.items()), "B formal score receipt mismatch")
    keys = {str(epoch) for epoch in EPOCHS}
    need(set(score.get("ema_by_epoch", ())) == keys and set(score.get("checkpoint_sha256_by_epoch", ())) == keys, "B 24-epoch scan incomplete")
    for epoch in EPOCHS:
        checkpoint = run / f"epoch_{epoch:03d}.pt"
        need(checkpoint.is_file() and sha(checkpoint) == score["checkpoint_sha256_by_epoch"][str(epoch)], f"B checkpoint SHA drift: {epoch}")
        state = baseline._validate_checkpoint(checkpoint, meta, expected_epoch=epoch)
        need(state.get("config", {}).get("arm") == ARM, f"B checkpoint arm drift: {epoch}")
    return meta, train, score


def validate_d_authority(meta: Mapping[str, Any], d_path: Path) -> dict[str, Any] | None:
    """Bind D replay to the D run it declares, never to B by a fixed path."""
    if not d_path.is_file():
        return None
    receipt = read_json(d_path)
    need(receipt.get("schema") == "m1_muscle_r100_baseline_replay_v1" and receipt.get("status") == "COMPLETED", "D replay schema/status mismatch")
    need(receipt.get("arm") == "D_JOINT" and receipt.get("all_prediction_sha256_exact") is True, "D replay authority mismatch")
    d_run = Path(receipt.get("baseline_run", "")).resolve()
    need(d_run.is_dir(), "D replay did not declare a readable baseline run")
    d_meta, _d_train, d_score = validate_formal_d(d_run)
    need(receipt.get("ho_contract") == d_score.get("ho_contract"), "D replay/run HO contract drift")
    need(meta.get("source_contract") == d_meta.get("source_contract"), "B/D source roster or source query contract drift")
    need(meta.get("initialization_sha256") == d_meta.get("initialization_sha256"), "B/D initial live encoder/decoder state drift")
    completed = receipt.get("completed")
    need(isinstance(completed, Mapping) and "1" in completed and isinstance(completed["1"], Mapping), "D replay epoch-1 evidence missing")
    metrics = completed["1"].get("metrics", {})
    per = metrics.get("per_session", {}) if isinstance(metrics, Mapping) else {}
    need(set(per) == set(baseline.HO), "D replay target/window roster drift")
    for session in baseline.HO:
        row = per[session]
        need(isinstance(row, Mapping) and all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in ("target_sha256", "starts_sha256")), f"D replay hashes missing: {session}")
    return {"path": str(d_path.resolve()), "sha256": sha(d_path), "declared_baseline_run": str(d_run), "declared_baseline_run_meta_sha256": sha(d_run / "run_meta.json"),
            "ho_contract": receipt["ho_contract"], "source_contract_match": True, "initialization_sha256_match": True,
            "target_start_authority": per, "authority_inference_batch_size": baseline.BATCH}


def validate_formal_d(run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Minimal formal D identity used only to bind the authority receipt."""
    meta, train, score = (read_json(run / name) for name in ("run_meta.json", "train_receipt.json", "score_receipt.json"))
    need(meta.get("schema") == "m1_rift_joint_train_v1" and meta.get("status") == "FORMAL" and meta.get("arm") == "D_JOINT" and meta.get("seed") == 42, "D formal metadata mismatch")
    need(train.get("status") == "COMPLETED" and train.get("arm") == "D_JOINT" and train.get("epochs") == 24 and train.get("steps") == 24 * baseline.UPDATES_PER_EPOCH, "D full training receipt mismatch")
    need(score.get("status") == "COMPLETED" and score.get("arm") == "D_JOINT" and set(score.get("ema_by_epoch", {})) == {str(e) for e in EPOCHS}, "D score scan mismatch")
    return meta, train, score

def score_epoch(model: torch.nn.Module, material: Mapping[str, Any], device: torch.device, batch_size: int, epoch_dir: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}; was_training = model.training; model.eval()
    try:
        for session in baseline.HO:
            item = material[session]; starts = np.asarray(item["starts"], dtype=np.int64)
            predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []
            for batch_index, batch in enumerate(DataLoader(item["dataset"], batch_size=batch_size, shuffle=False, num_workers=0)):
                x, y = batch[0], batch[1]; offset = batch_index * batch_size
                valid = baseline.valid_mask_from_padded_starts(item["starts"][offset:offset + len(x)], device=device)
                with torch.inference_mode():
                    output = model(x.float().to(device), item["bank"], input_valid_mask=valid)
                predictions.append(np.ascontiguousarray(output.float().cpu().numpy(), dtype=np.float32))
                targets.append(np.ascontiguousarray(y[:, -1, :].numpy(), dtype=np.float32))
            prediction = np.ascontiguousarray(np.concatenate(predictions), dtype=np.float32)
            target = np.ascontiguousarray(np.concatenate(targets), dtype=np.float32)
            need(prediction.shape == target.shape == (len(starts), 16) and np.isfinite(prediction).all() and np.isfinite(target).all(), f"{session}: replay geometry/nonfinite drift")
            artifact = epoch_dir / f"{session}_predictions.npz"; need(not artifact.exists(), f"refuse overwrite {artifact}")
            atomic_npz(artifact, prediction=prediction, target=target, window_ids=starts)
            rows[session] = {"window_count": int(len(target)), "legacy_variance_weighted_r2": float(variance_weighted_r2(target, prediction)),
                "channel_variance_weighted_r2": float(r2_score(target, prediction, multioutput="variance_weighted")),
                "prediction_sha256": array_sha(prediction), "target_sha256": array_sha(target), "window_ids_sha256": array_sha(starts),
                "artifact": artifact.name, "artifact_sha256": sha(artifact)}
    finally:
        model.train(was_training)
    return {"per_session": rows, "equal_session_mean_legacy": float(np.mean([rows[s]["legacy_variance_weighted_r2"] for s in baseline.HO])),
            "equal_session_mean_channel_variance_weighted_r2": float(np.mean([rows[s]["channel_variance_weighted_r2"] for s in baseline.HO])),
            "n_windows": int(sum(rows[s]["window_count"] for s in baseline.HO))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_DEFAULT)
    parser.add_argument("--d-replay-receipt", type=Path, default=D_REPLAY_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.inference_batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch size and CPU threads must be positive")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"--output-dir must be fresh: {out}")
    meta, train, old_score = validate_formal(args.baseline_run.resolve())
    d_authority = validate_d_authority(meta, args.d_replay_receipt.resolve())
    out.mkdir(parents=True)
    progress_path = out / "replay_progress.json"
    static_progress = {"schema": SCHEMA, "baseline_run": str(args.baseline_run.resolve()), "arm": ARM, "seed": 42,
        "inference_batch_size": args.inference_batch_size, "authority_inference_batch_size": baseline.BATCH,
        "batching_matches_old_authority": args.inference_batch_size == baseline.BATCH, "source_contract": meta["source_contract"],
        "ho_contract": old_score["ho_contract"], "d_replay_authority": d_authority}
    atomic_json(progress_path, {**static_progress, "status": "REPLAYING", "completed": {}, "last_completed_epoch": None})
    torch.set_num_threads(args.cpu_threads); device = torch.device(args.device)
    material = baseline._ho_material(); need(baseline._ho_contract(material) == old_score["ho_contract"], "B replay HO material contract drift")
    model = baseline._decoder(device, ARM, baseline.SEED)
    model.install_session_memory({s: material[s]["bank"] for s in baseline.HO}, {s: material[s]["calib10"] for s in baseline.HO}); model.to(device)
    ema = DecoderEMA(model, decay=baseline.plan.EMA_DECAY)
    completed: dict[str, Any] = {}
    for epoch in EPOCHS:
        checkpoint = args.baseline_run.resolve() / f"epoch_{epoch:03d}.pt"
        state = baseline._validate_checkpoint(checkpoint, meta, expected_epoch=epoch)
        model.load_state_dict(state["raw_state_dict"], strict=True); ema.load_state_dict(state["ema"])
        epoch_dir = out / f"epoch_{epoch:03d}"; epoch_dir.mkdir()
        observed = baseline._with_ema_eval(model, ema, lambda: score_epoch(model, material, device, args.inference_batch_size, epoch_dir))
        if d_authority is not None:
            for session in baseline.HO:
                expected_input = d_authority["target_start_authority"][session]
                observed_input = observed["per_session"][session]
                need(expected_input["window_count"] == observed_input["window_count"] and expected_input["target_sha256"] == observed_input["target_sha256"] and expected_input["starts_sha256"] == observed_input["window_ids_sha256"], f"D authority target/window drift: {session}")
        authority = old_score["ema_by_epoch"][str(epoch)]["per_session"]
        legacy_difference = max(abs(observed["per_session"][session]["legacy_variance_weighted_r2"] - float(authority[session]["r2"])) for session in baseline.HO)
        need(legacy_difference <= 1e-10, f"B replay old metric mismatch epoch={epoch}: {legacy_difference}")
        completed[str(epoch)] = {"checkpoint_sha256": sha(checkpoint), "artifact_dir": epoch_dir.name, "legacy_metric_max_abs_difference": legacy_difference, "metrics": observed}
        atomic_json(progress_path, {**static_progress, "status": "REPLAYING", "completed": completed, "last_completed_epoch": epoch})
    selected = min(EPOCHS, key=lambda epoch: (-completed[str(epoch)]["metrics"]["equal_session_mean_channel_variance_weighted_r2"], epoch))
    receipt = {"schema": SCHEMA, "status": "COMPLETED", "baseline_run": str(args.baseline_run.resolve()), "arm": ARM, "seed": 42,
        "formal_cell": baseline.CELL, "inference_batch_size": args.inference_batch_size, "authority_inference_batch_size": baseline.BATCH,
        "batching_matches_old_authority": args.inference_batch_size == baseline.BATCH, "baseline_run_meta_sha256": sha(args.baseline_run.resolve() / "run_meta.json"),
        "baseline_train_receipt_sha256": sha(args.baseline_run.resolve() / "train_receipt.json"), "baseline_score_receipt_sha256": sha(args.baseline_run.resolve() / "score_receipt.json"),
        "source_contract": meta["source_contract"], "b3s": meta["b3s"], "ho_contract": old_score["ho_contract"], "d_replay_authority": d_authority,
        "completed": completed, "selection": {"epoch": selected, "metric": "channel_variance_weighted_r2", "rule": "earliest maximum equal-session mean current channel variance-weighted R2"},
        "original_legacy_selection": old_score["selection"], "official_test_used": False,
        "code_sha256": {str(Path(__file__).resolve()): sha(Path(__file__).resolve()), str(Path(baseline.__file__).resolve()): sha(Path(baseline.__file__).resolve())}}
    atomic_json(out / "replay_receipt.json", receipt)
    atomic_json(progress_path, {**static_progress, "status": "COMPLETED", "completed": completed, "last_completed_epoch": 24})
    print(json.dumps({"status": "COMPLETED", "receipt": str(out / "replay_receipt.json"), "selected_epoch": selected}, sort_keys=True))


if __name__ == "__main__":
    main()
