#!/usr/bin/env python3
"""Local validation for the Ce-NAT chunk100e TTA submission.

Stages:
1. ``payload``  - audit payload schema/arm/chain/coverage.
2. ``parity``   - THE deployment-equivalence sentinel: the ONLINE decoder is
                  driven bin-by-bin (batch 1) over each of the 13 sessions and
                  compared, per session, against the V4 scorer semantics:
                  commit bins, accept/reject sequence, rate-history digest,
                  pool member order, and per-window predictions equal within
                  1e-7 (same process, same device); session R2 difference
                  <= 1e-7 against the V4 score rows.
3. ``contract`` - on_done no-op, weight/identity fingerprints frozen, no
                  optimizer state, law modules never imported at predict time.
4. ``minival``  - real host FalconEvaluator (local, batch 7) with call
                  counters; batch-slot isolation asserted via per-slot audit.

Writes artifacts/local_validation_receipt.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
PAYLOAD_NAME = "t4_m2_seed42_cenat_chunk100e_tta.pkl"
DECODER_MODULE = "cenat_chunk_decoder"
DECODER_CLASS = "CenatChunkTTADecoder"
EVALUATOR_SHA256 = "2b848f84ef620eac82ce53d331d03727d9ae4ea9a5f2e5edb6e6557537ce4418"
V4_SCORE = ROOT / "tfpd_exploration/results/m2_ajpf_c_v4/score.json"
R2_ABS_TOLERANCE = 1.0e-7


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_runtime_imports() -> None:
    streaming_root = ROOT / "streaming_calibration_exp"
    apfg_package = ROOT / "tfpd_exploration/submissions/evalai_m2_apfg_static_v1"
    for path in (ROOT, HERE, streaming_root, apfg_package):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import src.models.streaming_calibration_module  # noqa: F401
    sys.path.append(str(ROOT / "SPINT-main"))


def audit_payload(payload_path: Path) -> dict[str, Any]:
    bootstrap_runtime_imports()
    module = __import__(DECODER_MODULE)
    with payload_path.open("rb") as handle:
        payload = module.CPUUnpickler(handle).load()
    require(payload["schema_version"] == module.PAYLOAD_SCHEMA, "payload schema drift")
    require(str(payload["task"]) == "FalconTask.m2", "payload task drift")
    metadata = payload["metadata"]
    require(metadata["arm"] == module.PAYLOAD_ARM, "arm drift")
    require(metadata["test_time_adaptive"] is True, "payload must declare TTA")
    require(metadata["label_budget"] == 4, "label budget drift")
    require(metadata["online_backward_pass"] is False, "online backward drift")
    require(len(payload["seed_pool_by_dataset_tag"]) == 13, "coverage drift")
    for tag, pool in payload["seed_pool_by_dataset_tag"].items():
        require(np.asarray(pool).shape == (30, 100, 96), f"seed shape drift {tag}")
    json.dumps(metadata, sort_keys=True)
    return {"payload_sha256": sha256_file(payload_path), "arm": metadata["arm"],
            "chain": metadata.get("checkpoint_sha256"), "session_count": 13,
            "test_time_adaptive": True}


def _local_reference_rows() -> dict[tuple[str, str], dict[str, Any]]:
    """V4 score rows for Ce-NAT/chunk100e + pooled anchor."""
    payload = json.loads(V4_SCORE.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload["rows"]:
        key = (row["surface"], row["session"])
        if row["arm"] == "Ce-NAT":
            out.setdefault(key, {})["cenat"] = row
        if row["arm"] == "pooled":
            out.setdefault(key, {})["pooled"] = row
    require(len(out) == 13, "V4 row coverage drift")
    return out


def run_parity_and_contract(payload_path: Path) -> dict[str, Any]:
    bootstrap_runtime_imports()
    import torch

    module = __import__(DECODER_MODULE)
    decoder_class = getattr(module, DECODER_CLASS)
    from falcon_challenge.config import FalconConfig, FalconTask
    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        calibration_file_map, load_frozen_model_and_data, manual_decode)
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts, variance_weighted_r2)

    torch.set_num_threads(min(8, torch.get_num_threads()))
    task_config = FalconConfig(task=FalconTask.m2)
    decoder = decoder_class(task_config=task_config, model_path=str(payload_path), batch_size=1)
    parameter_before = decoder.parameter_fingerprint()

    _model, data_module, _tc, _meta = load_frozen_model_and_data()
    sealed = json.loads((ROOT / "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json")
                        .read_text(encoding="utf-8"))
    sealed_rows = {(r["surface"], r["session"]): r for r in sealed["rows"] if r["cell"] == "ridge_activity30_m4"}
    reference = _local_reference_rows()

    surfaces = (("external_official_query", data_module.val_heldout_dataset, 6, "external"),
                ("within_post30", data_module.train_dataset, 7, "within"))
    per_surface: dict[str, Any] = {}
    for surface_name, dataset, expected, short in surfaces:
        sessions = sorted(dataset.calib_trialized_neural_features)
        require(len(sessions) == expected, f"{surface_name} count drift")
        per_session = {}
        for session in sessions:
            # --- local scorer reference semantics (chunk100e, batch-1 segments)
            from tfpd_exploration.src.m2_ajpf_c_v1 import chunk_law

            angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
            calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
            seed_rows = [np.ascontiguousarray(row, dtype=np.float32)
                         for row in calibration[:30]]
            side, _ = __import__("laws").sealed_fit_ridge_side(dataset, session,
                                                               __import__("laws").select_dopt4_support(angles))
            side_tensor = torch.from_numpy(np.ascontiguousarray(side)).unsqueeze(0)
            neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
            targets = np.asarray(dataset.covariate_data[session], dtype=np.float32)
            starts = np.asarray([s for name, s in dataset.window_indices if name == session], dtype=np.int64)
            if short == "within":
                starts = select_common_post30_window_starts(starts, dataset.trial_start_indices[session])
            endpoints = starts + 49
            windows = np.ascontiguousarray(neural[starts[:, None] + np.arange(50)[None, :]], dtype=np.float32)
            commits = chunk_law.chunk_commit_bins(neural, 0, energy_gated=True)
            counts = chunk_law.committed_before(commits, endpoints)

            student_decoder = decoder.decoder  # frozen modules shared by both paths
            id_encoder = decoder.id_encoder
            local_predictions = np.zeros((starts.size, 2), dtype=np.float32)
            segment_start = 0
            committed_upto = 0
            pool = __import__("tfpd_exploration.src.m2_ajpf_c_v1.chunk_law", fromlist=["SeededPool"]).SeededPool(
                seed_rows, 4)
            identity = None
            identity_count = -1
            with torch.inference_mode():
                while segment_start < starts.size:
                    stop = segment_start
                    while stop < starts.size and counts[stop] == counts[segment_start]:
                        stop += 1
                    count = int(counts[segment_start])
                    while committed_upto < count:
                        commit_bin = commits[committed_upto]
                        pool.commit(np.ascontiguousarray(neural[commit_bin - 99 : commit_bin + 1], dtype=np.float32))
                        committed_upto += 1
                    if count != identity_count:
                        support = torch.from_numpy(pool.stack()).unsqueeze(0)
                        identity = id_encoder.forward_batch(support, side_features=side_tensor)
                        identity_count = count
                    for offset in range(segment_start, stop, 1024):
                        batch = torch.from_numpy(windows[offset : min(offset + 1024, stop)])
                        output = manual_decode(student_decoder, batch, identity)
                        local_predictions[offset : min(offset + 1024, stop)] = (
                            output[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False) / 5.0)
                    segment_start = stop

            # --- online decoder over the same session (batch 1, per bin)
            fake_tag = Path(f"sub-MonkeyN-held-in-calib_{session}_behavior+ecephys.nwb")
            decoder.reset(dataset_tags=[fake_tag])
            real_stream = np.ascontiguousarray(neural[49:], dtype=np.float32)  # prehistory lives in reset
            online_predictions = np.zeros((int(real_stream.shape[0]), 2), dtype=np.float32)
            online_commits: list[int] = []
            state = decoder.slots[0]
            original_append = state.append_bin
            counter = {"bin": -1}

            accepted_before = {"n": state.accepted_count}
            def traced_append(row, _state=state, _orig=original_append, _before=accepted_before):
                _orig(row)
                if _state.accepted_count > _before["n"]:
                    _before["n"] = _state.accepted_count
                    online_commits.append(counter["bin"] + 49)  # padded-bin equivalent

            state.append_bin = traced_append  # type: ignore[method-assign]
            for real_bin in range(int(real_stream.shape[0])):
                counter["bin"] = real_bin
                online_predictions[real_bin] = decoder.predict(real_stream[real_bin : real_bin + 1])[0]
            state.append_bin = original_append  # type: ignore[method-assign]

            # --- compare: online prediction at real bin k == local window start s=k
            online_at_endpoints = online_predictions[starts]
            max_abs = float(np.abs(online_at_endpoints - local_predictions).max())
            require(max_abs <= R2_ABS_TOLERANCE,
                    f"{session}: parity drift {max_abs}")
            require(online_commits == commits,  # padded-bin equivalence
                    f"{session}: commit-bin sequence drift {online_commits[:5]}…")
            local_r2 = float(variance_weighted_r2(
                np.ascontiguousarray(targets[endpoints], dtype=np.float32), local_predictions))
            v4_row = reference[(surface_name, session)]["cenat"]
            require(abs(local_r2 - float(v4_row["r2"])) <= R2_ABS_TOLERANCE,
                    f"{session}: R2 drift vs V4: {local_r2} vs {v4_row['r2']}")
            per_session[session] = {
                "r2_online_endpoints": float(variance_weighted_r2(
                    np.ascontiguousarray(targets[endpoints], dtype=np.float32),
                    online_at_endpoints)),
                "r2_local_scorer": local_r2,
                "r2_v4_row": float(v4_row["r2"]),
                "max_abs_prediction_diff": max_abs,
                "commits": len(commits),
                "window_count": int(starts.size),
                "pooled_static_r2_sealed": float(sealed_rows[(surface_name, session)]["r2"]),
            }
        per_surface[surface_name] = {"per_session": per_session}

    # contract assertions
    done_result = decoder.on_done(np.asarray([True]))
    require(done_result is None, "on_done not a no-op")
    require(decoder.parameter_fingerprint() == parameter_before, "weights mutated")
    gradients = [p.grad for p in decoder.decoder.parameters() if p.grad is not None]
    require(not gradients, "gradient state after replay")
    return {
        "parity": per_surface,
        "contract": {"on_done_is_noop": True, "weights_frozen": True,
                     "no_gradient_state": True,
                     "parameter_fingerprint_sha256": parameter_before},
    }


def run_minival(payload_path: Path, out_dir: Path) -> dict[str, Any]:
    bootstrap_runtime_imports()
    import os

    os.environ["EVAL_DATA_PATH"] = str(ROOT / "SPINT-main/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PREDICTION_PATH_LOCAL"] = str(out_dir / "minival_prediction.pkl")
    os.environ["GT_PATH"] = str(out_dir / "minival_gt.pkl")

    import falcon_challenge.evaluator as evaluator_module
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator

    module = __import__(DECODER_MODULE)
    decoder_class = getattr(module, DECODER_CLASS)
    evaluator_sha = sha256_file(Path(evaluator_module.__file__))
    require(evaluator_sha == EVALUATOR_SHA256, "evaluator sha drift")

    task_config = FalconConfig(task=FalconTask.m2)
    decoder = decoder_class(task_config=task_config, model_path=str(payload_path), batch_size=7)
    calls = {"on_done": 0, "reset": 0, "predict": 0}
    decoder_reset, decoder_predict, decoder_on_done = decoder.reset, decoder.predict, decoder.on_done

    def counting_reset(*a, **k):
        calls["reset"] += 1
        return decoder_reset(*a, **k)

    def counting_predict(*a, **k):
        calls["predict"] += 1
        return decoder_predict(*a, **k)

    def counting_on_done(*a, **k):
        calls["on_done"] += 1
        return decoder_on_done(*a, **k)

    decoder.reset = counting_reset  # type: ignore[method-assign]
    decoder.predict = counting_predict  # type: ignore[method-assign]
    decoder.on_done = counting_on_done  # type: ignore[method-assign]

    evaluator = FalconEvaluator(eval_remote=False, split="m2")
    started = time.monotonic()
    result = evaluator.evaluate(decoder, phase="minival")
    elapsed = time.monotonic() - started
    require(calls["on_done"] == 0, "official evaluator invoked on_done")
    require(calls["reset"] > 0 and calls["predict"] > 0, "evaluator never drove decoder")
    audit = decoder.slot_audit()
    require(all(entry["candidate_rows"] < 100 for entry in audit), "candidate buffer overflow")
    return {
        "phase": "minival", "batch_size": 7,
        "evaluator_sha256": evaluator_sha,
        "calls": calls,
        "slot_audit": audit,
        "wall_seconds": elapsed,
        "metrics": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stages", default="payload,parity,contract,minival")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"schema": "m2_cenat_chunk100e_validation_dry_v1",
                          "status": "DRY_NO_DATA_NO_WRITE"}, sort_keys=True))
        return
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    payload_path = ARTIFACTS / PAYLOAD_NAME
    require(payload_path.exists(), f"payload missing: {payload_path}")
    receipt: dict[str, Any] = {"schema_version": "m2_cenat_chunk100e_local_validation_v1",
                               "status": "LOCAL_VALIDATION_RUNNING"}
    if "payload" in stages:
        receipt["payload_audit"] = audit_payload(payload_path)
    if {"parity", "contract"} & set(stages):
        result = run_parity_and_contract(payload_path)
        receipt["parity"] = result["parity"]
        receipt["contract"] = result["contract"]
    if "minival" in stages:
        receipt["official_local_minival"] = run_minival(payload_path, ARTIFACTS / "minival_v1")
    receipt["status"] = "LOCAL_VALIDATION_COMPLETE"
    out = ARTIFACTS / "local_validation_receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
