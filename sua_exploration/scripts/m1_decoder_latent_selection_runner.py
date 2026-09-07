#!/usr/bin/env python3
"""Reviewed-only M1 selection data boundary; default mode never opens an NWB/query tensor.

The executable stage is deliberately guarded.  Its constants are imported by the
prelaunch receipt and are the only supported data boundary for a later root-approved
CPU selection run.  There is no route to `[210,end)`, held-out data, or EvalAI here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCE = ROOT / "streaming_calibration_exp"
DATA = ROOT / "SPINT-main" / "data" / "000941" / "sub-MonkeyL-held-in-calib"
F0 = ROOT / "streaming_calibration_exp" / "outputs" / "streaming_calibration" / "m1_clean_selection_v1_f0_m1_f1_s42_20260801_192017" / "checkpoints" / "best.ckpt"
TEACHER = ROOT / "SPINT-main" / "logs" / "train" / "runs" / "2026-07-21-19-11-01" / "checkpoints" / "best_ckpt" / "epoch_019.ckpt"
F0_SHA256 = "1ec318f81cfaa9f6eb5e998a2b34135e2bde9941c48fb47bc46f47632f0d6cd8"
TEACHER_SHA256 = "c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2"
OUTER_LEFT_OUT = "ses-20120926"
SOURCE_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
SUPPORT = (0, 10)
SELECTION = (10, 210)
REPORT_START = 210
ARMS = ("full", "rate_only", "label_shuffle", "rate_residualized_condition_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_selection_only_request(*, start_trial: int, end_trial: int, heldout: bool, evalai: bool) -> None:
    if (start_trial, end_trial) != SELECTION:
        raise ValueError("M1 DLA runner accepts exactly held-in query trials [10,210), never report/post-hoc ranges")
    if heldout or evalai:
        raise ValueError("M1 DLA selection runner forbids held-out and EvalAI access")


def frozen_manifest() -> dict[str, Any]:
    if sha256(F0) != F0_SHA256 or sha256(TEACHER) != TEACHER_SHA256:
        raise ValueError("selection runner checkpoint hash mismatch")
    files = sorted(DATA.glob("*held-in-calib*.nwb"))
    names = tuple(path.name.split("_")[1].split(".")[0] for path in files)
    if names != SOURCE_SESSIONS:
        raise ValueError(f"expected exactly the four locked M1 source sessions, got {names}")
    return {"source_sessions": list(SOURCE_SESSIONS), "outer_left_out": OUTER_LEFT_OUT,
            "support": list(SUPPORT), "selection": list(SELECTION), "report_forbidden_from": REPORT_START,
            "include_heldout": False, "evalai": False,
            "f0_checkpoint_sha256": F0_SHA256, "teacher_checkpoint_sha256": TEACHER_SHA256}


def load_reviewed_selection_datamodule() -> Any:
    """Open only four held-in-calib NWBs and build query windows constrained to [10,210).

    This function is intentionally never invoked by default/prelaunch.  A future
    reviewed executor must additionally construct `E0` for all sessions, `Delta*`
    only for the three outer-train sessions, and decoder predictions only from this
    datamodule's query windows.  It must reject any output window at/after 210.
    """
    assert_selection_only_request(start_trial=10, end_trial=210, heldout=False, evalai=False)
    if str(SCE) not in sys.path:
        sys.path.insert(0, str(SCE))
    from src.data.falcon_datamodule import FalconDataModule
    dm = FalconDataModule(task="m1", data_dir=str(DATA.parent) + "/", heldin_session_names=[""], batch_size=2,
                          window_size=100, calibration_n_trials=10, random_calibration=False, smooth_calibration=False,
                          max_trial_length=1024, standardize_covariates=False, use_intertrials=True,
                          use_calib_intertrials=False, trial_feature_type="raw", interpolate_trials=True,
                          interpolate_trials_kind="cubic", pad_value=-1.0, validation_protocol="loso", loso_fold=1,
                          include_heldout_in_fit=False, include_heldout_in_test=False, query_start_trial=0,
                          heldin_query_start_trial=10, heldin_query_end_trial=210, num_workers=0, pin_memory=False,
                          side_feature_group="d4")
    dm.trainer = SimpleNamespace(world_size=1)
    dm.setup("fit")
    audit = dm.val_heldin_dataset.query_window_audit
    if set(audit) != {OUTER_LEFT_OUT} or any(record.get("query_end_trial") != 210 or record.get("query_start_trial") != 10 for record in audit.values()):
        raise ValueError("outer selection datamodule escaped the locked [10,210) query boundary")
    return dm


def _selection_api() -> Any:
    import importlib.util
    path = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_stage.py"
    spec = importlib.util.spec_from_file_location("m1_dla_selection_stage", path)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load selection-stage API")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through ``sys.modules`` while
    # the module body is executing.  Register the isolated module before
    # ``exec_module`` so the real runner follows the same import contract as
    # the focused tests.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dataset_all_sessions(dm: Any) -> Any:
    """Rebuild query windows for all four sources, exactly [10,210), no held-out object."""
    from src.data.falcon_datamodule import FalconDataset
    dataset = FalconDataset(sessions_dict=dm.train_calib_heldin_sessions, calib_sessions_dict=dm.train_calib_heldin_sessions,
                            window_size=100, split="m1_dla_selection", calibration_n_trials=10,
                            random_calibration=False, smooth_calibration=False, max_trial_length=1024,
                            use_calib_intertrials=False, trial_feature_type="raw", remove_still_times=False,
                            remove_calib_still_times=False, use_calib_active_segments=False, calib_n_active_segments=1,
                            interpolate_trials=True, interpolate_trials_kind="cubic", pad_value=-1.0,
                            side_feature_group="d4", side_feature_shuffle_seed=42, query_start_trial=10, query_end_trial=210)
    if tuple(sorted(dataset.query_window_audit)) != SOURCE_SESSIONS:
        raise ValueError("selection dataset did not contain exactly the four locked source sessions")
    for name, audit in dataset.query_window_audit.items():
        if audit.get("query_start_trial") != 10 or audit.get("query_end_trial") != 210 or not audit.get("full_window_disjoint"):
            raise ValueError(f"{name}: selection query audit is not fully disjoint [10,210)")
    return dataset


def _load_frozen_models(device: str) -> tuple[Any, Any, Any, Any]:
    """Return F0/B3 encoder, F0 decoder, and teacher identity modules; caller must restrict teacher use."""
    import torch
    if str(SCE) not in sys.path:
        sys.path.insert(0, str(SCE))
    from src.models.falcon_module import FalconLitModule
    from src.models.components.streaming_encoders import EarlyPoolEncoder
    teacher_module = FalconLitModule.load_from_checkpoint(TEACHER, map_location="cpu", weights_only=False).eval()
    teacher = teacher_module.net.eval()
    f0_payload = torch.load(F0, map_location="cpu", weights_only=False)
    state = f0_payload["state_dict"]
    encoder = EarlyPoolEncoder(1024, 100, 64).eval()
    encoder.load_state_dict({key[len("student.id_encoder."):]: value for key, value in state.items() if key.startswith("student.id_encoder.")}, strict=True)
    decoder = copy.deepcopy(teacher).eval()
    decoder_state = {key[len("student.decoder."):]: value for key, value in state.items() if key.startswith("student.decoder.")}
    decoder.load_state_dict(decoder_state, strict=True)
    for module in (encoder, decoder, teacher.fc_id_in, teacher.fc_id_out):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.to(device).eval()
    return encoder, decoder, teacher.fc_id_in.eval(), teacher.fc_id_out.eval()


def _teacher_delta_for_source_only(support: np.ndarray, e0: np.ndarray, teacher_in: Any, teacher_out: Any, device: str) -> np.ndarray:
    """Private source-only supervisor; outer-left-out construction has no call to this API."""
    import torch
    with torch.no_grad():
        c = torch.from_numpy(np.asarray(support, dtype=np.float32)).unsqueeze(0).to(device)
        target = teacher_out(teacher_in(c.permute(0, 1, 3, 2)).mean(dim=1)).cpu().numpy()[0]
    return target - e0


def _e0_from_support(support: np.ndarray, encoder: Any, device: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        return encoder.forward_batch(torch.from_numpy(np.asarray(support, dtype=np.float32)).unsqueeze(0).to(device)).cpu().numpy()[0]


def _query_windows(dataset: Any, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return windows/targets/trial-id only from constrained raw query arrays.

    The dataset is constructed with ``side_feature_group='d4'`` solely so that
    the calibration object IDs required by the feature arms are loaded.  Calling
    ``dataset[index]`` would also construct a normalized D4 side feature and
    therefore both require an unrelated D4-normalization fit and risk silently
    injecting that feature into this experiment.  Slice the already-audited raw
    neural/covariate arrays directly instead.
    """
    neural, behavior, trial_ids = [], [], []
    starts = np.asarray(dataset.trial_start_indices[name])
    window_size = int(dataset.window_size)
    for session, start in dataset.window_indices:
        if session != name:
            continue
        end = int(start) + window_size
        neural_window = np.asarray(dataset.neural_data[name][int(start):end], dtype=np.float32)
        behavior_window = np.asarray(dataset.covariate_data[name][int(start):end], dtype=np.float32)
        if neural_window.shape[0] != window_size or behavior_window.shape[0] != window_size:
            raise ValueError(f"{name}: incomplete raw query window at start={start}")
        trial = int(np.searchsorted(starts, start, side="right") - 1)
        if trial < SELECTION[0] or trial >= SELECTION[1]:
            raise ValueError(f"{name}: query window escaped selection trial range")
        neural.append(neural_window)
        behavior.append(behavior_window[-1])  # frozen M1 decoder convention: last timestep only
        trial_ids.append(trial)
    if not neural:
        raise ValueError(f"{name}: no eligible selection windows")
    return np.stack(neural), np.stack(behavior), np.asarray(trial_ids, dtype=np.int64)


def _frozen_f0_decode(decoder: Any, neural: np.ndarray, identity: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
    import torch
    if batch_size <= 0:
        raise ValueError("decode batch size must be positive")
    outputs = []
    with torch.no_grad():
        e = torch.from_numpy(np.asarray(identity, dtype=np.float32)).unsqueeze(0).to(device)
        for start in range(0, neural.shape[0], batch_size):
            x = torch.from_numpy(np.asarray(neural[start:start + batch_size], dtype=np.float32)).to(device)
            src = decoder.fc_in(x.permute(0, 2, 1) + e)
            rep = decoder.fc_in(decoder.rep).to(src).repeat(src.shape[0], 1, 1)
            output, _ = decoder.transformer(rep, src)
            outputs.append(decoder.fc_out(output).permute(0, 2, 1)[:, -1, :].cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_selection_inputs(device: str) -> tuple[list[Any], Any, dict[str, Any], Any, np.ndarray]:
    """Build actual held-in input objects.  Never call before explicit root-reviewed execution.

    The outer object is constructed in a separate branch without a teacher argument,
    so `ses-20120926` cannot acquire a Delta target through this function's API.
    """
    api = _selection_api()
    dm = load_reviewed_selection_datamodule()
    dataset = _dataset_all_sessions(dm)
    encoder, decoder, teacher_in, teacher_out = _load_frozen_models(device)
    outer_train, outer, outer_trial_ids = [], None, None
    audits: dict[str, Any] = {}
    for name in SOURCE_SESSIONS:
        support = np.asarray(dataset.calib_trialized_neural[name][:10], dtype=np.float32)
        sums, lengths, labels = (np.asarray(dataset.calib_trial_spike_sums[name][:10]), np.asarray(dataset.calib_trial_lengths[name][:10]), np.asarray(dataset.calib_trial_obj_ids[name][:10]))
        if set(labels.tolist()) != {1, 2, 3, 4}:
            raise ValueError(f"{name}: locked M1 selection requires all four M10 obj labels")
        base = api.support_unit_features(sums, lengths, labels, session_name=name, seed=42)
        shuffled = api.support_unit_features(sums, lengths, labels, session_name=name, seed=42, shuffle_labels=True)
        features = {"full": base, "rate_only": base, "label_shuffle": shuffled, "rate_residualized_condition_only": base}
        e0 = _e0_from_support(support, encoder, device)
        neural, behavior, trial_ids = _query_windows(dataset, name)
        audits[name] = {"support_trials": 10, "query_trial_range": [10, 210], "query_windows": int(neural.shape[0]),
                        "query_trial_ids_minmax": [int(trial_ids.min()), int(trial_ids.max())], "label_counts": {str(k): int((labels == k).sum()) for k in (1, 2, 3, 4)}}
        if name == OUTER_LEFT_OUT:
            outer = api.OuterSelectionSession(name, e0, features, neural, behavior)
            outer_trial_ids = trial_ids
        else:
            delta = _teacher_delta_for_source_only(support, e0, teacher_in, teacher_out, device)
            outer_train.append(api.SourceSelectionSession(name, e0, delta, features, neural, behavior))
    if outer is None or {item.name for item in outer_train} != set(SOURCE_SESSIONS) - {OUTER_LEFT_OUT}:
        raise RuntimeError("outer split construction failed")
    if outer_trial_ids is None:
        raise RuntimeError("outer selection trial IDs missing")
    return outer_train, outer, audits, decoder, outer_trial_ids


def _block_bootstrap_windows(api: Any, f0: np.ndarray, dla: np.ndarray, target: np.ndarray, trial_ids: np.ndarray, *, block_trials: int = 10, replicates: int = 1000) -> dict[str, Any]:
    """Convert trial-labelled selection windows to contiguous trial-block bootstrap samples."""
    unique = np.unique(trial_ids)
    if unique.tolist() != list(range(10, 210)):
        raise ValueError("outer bootstrap requires all and only chronological selection trials 10..209")
    blocks = [unique[start:start + block_trials] for start in range(0, len(unique), block_trials)]
    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(replicates):
        selected_trials = np.concatenate([blocks[i] for i in rng.randint(0, len(blocks), size=len(blocks))])
        indices = np.concatenate([np.flatnonzero(trial_ids == trial) for trial in selected_trials])
        deltas.append(api.variance_weighted_r2(dla[indices], target[indices]) - api.variance_weighted_r2(f0[indices], target[indices]))
    values = np.asarray(deltas)
    se = float(values.std(ddof=1))
    return {"unit": "aggregate_R2_recomputed_after_contiguous_trial_block_resample", "block_trials": block_trials,
            "replicates": replicates, "seed": 42, "delta_r2_mean": float(values.mean()), "standard_error": se,
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))], "two_sided_mde_r2": float(1.96 * se)}


def score_selection_arms(arms: tuple[str, ...], *, device: str, decode_batch_size: int) -> dict[str, Any]:
    """Actual CPU scorer; only reachable through reviewed CLI and never reads report trials."""
    api = _selection_api()
    outer_train, outer, audits, decoder, outer_trial_ids = build_selection_inputs(device)
    decode = lambda neural, identity: _frozen_f0_decode(decoder, neural, identity, device=device, batch_size=decode_batch_size)
    results: dict[str, Any] = {}
    for arm in arms:
        candidates = api.inner_loso_candidates(arm, outer_train, decoder=decode, outer_left_out_name=OUTER_LEFT_OUT)
        locked = max(candidates, key=lambda item: (item.mean_delta_r2, -item.rank, -item.ridge_lambda))
        fitted = api.fit_reduced_rank_map(arm, [item.features[arm] for item in outer_train], [item.delta_star for item in outer_train], rank=locked.rank, ridge_lambda=locked.ridge_lambda)
        ehat = outer.e0 + fitted.predict_delta(outer.features[arm])
        f0_prediction, dla_prediction = decode(outer.query_neural, outer.e0), decode(outer.query_neural, ehat)
        f0_r2, arm_r2 = api.variance_weighted_r2(f0_prediction, outer.query_behavior), api.variance_weighted_r2(dla_prediction, outer.query_behavior)
        results[arm] = {"inner_candidates": [{"rank": item.rank, "lambda": item.ridge_lambda, "per_inner_validation_delta_r2": dict(item.inner_validation_delta_r2), "mean_delta_r2": item.mean_delta_r2} for item in candidates],
                        "locked": {"rank": locked.rank, "lambda": locked.ridge_lambda, "criterion": "mean_inner_validation_frozen_decoder_behavior_delta_R2"},
                        "outer_selection": {"window": [10, 210], "identity_r2": f0_r2, "arm_r2": arm_r2, "delta_r2": arm_r2 - f0_r2,
                                            "bootstrap": _block_bootstrap_windows(api, f0_prediction, dla_prediction, outer.query_behavior, outer_trial_ids)},
                        "state_accounting": api.feature_state_accounting(arm, locked.rank),
                        "calibration_accounting": {"labels": "M10 obj_id only for full/shuffle/residual; none for rate_only", "label_counts": audits[OUTER_LEFT_OUT]["label_counts"],
                                                    "feature_rate_divides": 10 * 64, "conditioned_rate_accumulates": 10 * 64,
                                                    "gradients": 0, "decoder_training": False, "teacher_used_on_outer_leftout": False}}
    gates = None
    if {"full", "rate_only", "label_shuffle"} <= set(results):
        full, rate, shuffle = (results[name]["outer_selection"]["delta_r2"] for name in ("full", "rate_only", "label_shuffle"))
        boot = results["full"]["outer_selection"]["bootstrap"]
        gates = {"full_minus_identity_ge_0p015": full >= .015, "full_minus_rate_ge_0p010": full - rate >= .010,
                 "full_minus_label_shuffle_ge_0p010": full - shuffle >= .010, "full_ci_excludes_zero": boot["ci95"][0] > 0.0,
                 "full_mde_le_0p015": boot["two_sided_mde_r2"] <= .015, "diagnostic_only_arm": "rate_residualized_condition_only"}
    return {"schema_version": "m1_dla_selection_result_v1", "manifest": frozen_manifest(), "source_audits": audits,
            "arms": results, "gates_if_full_rate_shuffle_scored_together": gates,
            "no_report_access": True, "heldout_access": False, "EvalAI_access": False,
            "execution_ledger": {"forward_device": device, "optimizer_steps": 0, "backward_calls": 0, "trainable_parameters": 0,
                                 "decoder_eval": True, "torch_no_grad": True, "decode_batch_size": decode_batch_size}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-reviewed-data-load", action="store_true")
    parser.add_argument("--execute-scoring", action="store_true")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1"), default="cpu")
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--smoke-mini-batches", type=int, default=0)
    args = parser.parse_args()
    manifest = frozen_manifest()
    if args.execute_scoring and not args.execute_reviewed_data_load:
        raise ValueError("actual scoring requires the explicit reviewed data-load flag")
    if not args.execute_reviewed_data_load:
        print(json.dumps({"status": "preflight_only_no_data_opened", "manifest": manifest}, sort_keys=True))
        return
    if os.environ.get("M1_DLA_ROOT_REVIEWED_EXECUTION") != "1":
        raise PermissionError("set M1_DLA_ROOT_REVIEWED_EXECUTION=1 only after root review; default cannot open data")
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available() or int(args.device.split(":")[1]) >= torch.cuda.device_count():
            raise RuntimeError(f"requested unavailable device {args.device}")
    if not args.execute_scoring:
        dm = load_reviewed_selection_datamodule()
        print(json.dumps({"status": "reviewed_data_boundary_loaded_no_fit_or_scoring", "manifest": manifest,
                          "outer_selection_audit": dm.val_heldin_dataset.query_window_audit}, sort_keys=True))
        return
    arms = ARMS if args.arm == "all" else (args.arm,)
    if args.smoke_mini_batches:
        # smoke is forward-only: construct the immutable inputs and decode at most
        # the requested number of batches from outer [10,210), without fit/score.
        _, outer, audits, decoder, _ = build_selection_inputs(args.device)
        count = min(outer.query_neural.shape[0], args.smoke_mini_batches * args.decode_batch_size)
        _frozen_f0_decode(decoder, outer.query_neural[:count], outer.e0, device=args.device, batch_size=args.decode_batch_size)
        print(json.dumps({"status": "selection_forward_smoke_complete_no_fit_or_score", "device": args.device, "decoded_windows": count, "audits": audits}, sort_keys=True))
        return
    default_out = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v2" / "selection_runs" / ("all_arms" if args.arm == "all" else f"arm_{args.arm}")
    out = args.out or default_out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable selection result: {out}")
    out.mkdir(parents=True, exist_ok=False)
    result = score_selection_arms(arms, device=args.device, decode_batch_size=args.decode_batch_size)
    result["execution"] = {"arm_request": args.arm, "shared_manifest_sha256": sha256(ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v2" / "selection_stage_prelaunch.json")}
    target = out / "result.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "result.sha256").write_text(f"{sha256(target)}  result.json\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
