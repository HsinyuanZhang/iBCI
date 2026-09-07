"""No-NWB contracts for executable-but-not-launched RT R4 infrastructure."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import prepare_rt_r4_execution as preflight
from scripts import rt_r4_pilot as pilot
import src.data.rt_nested_loso_datamodule as nested
import src.rt_r4_budget_response_eval as r4_eval


SESSION_NAMES = (
    "ses-RT-20131009", "ses-RT-20131010", "ses-RT-20131011",
    "ses-RT-20131028", "ses-RT-20131029", "ses-RT-20131209",
    "ses-RT-20131210", "ses-RT-20131212", "ses-RT-20131213",
    "ses-RT-20131217", "ses-RT-20131218", "ses-RT-20150316",
    "ses-RT-20150317", "ses-RT-20150318", "ses-RT-20150320",
)


def _path(name: str) -> Path:
    return Path(f"/synthetic/sub-C_{name}_behavior+ecephys.nwb")


def _session(name: str, units: int = 5) -> dict:
    trials, bins = 32, 20
    total = trials * bins
    trial_change = np.zeros(total, dtype=bool)
    velocity = np.zeros((total, 2), dtype=np.float32)
    segment = np.empty(total, dtype=np.int64)
    neural = np.zeros((total, units), dtype=np.float32)
    for trial in range(trials):
        left = trial * bins
        trial_change[left] = True
        # Both design and unit coefficients drift across the prefix, making
        # the M6 descriptor deterministically distinct from M24.
        theta = 0.31 * trial
        v = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
        velocity[left:left + bins] = v
        segment[left:left + bins] = trial
        for unit in range(units):
            rate = 1.0 + 0.3 * unit + (0.2 + 0.04 * trial) * v[0] + (0.1 * unit - 0.03 * trial) * v[1]
            pattern = ((np.arange(bins) + trial + unit) % 3 == 0).astype(np.float32)
            neural[left:left + bins, unit] = np.maximum(0.0, rate + pattern)
    return {
        "session_name": name,
        "neural": neural,
        "covariates": velocity,
        "trial_change": trial_change,
        "eval_mask": np.ones(total, dtype=bool),
        "k4_segment_id": segment,
        "rt_segment_audit": {
            "complete_cue_trials": trials,
            "accepted_reach_segments": trials,
            "event_qualified_bins": total,
            "trial_records": [
                {
                    "trial_index": trial, "complete_cue": True,
                    "accepted_segments": 1, "declared_segments": 1,
                    "excluded_segments": 0, "exclusion_reason": None,
                    "segment_exclusion_reasons": {},
                }
                for trial in range(trials)
            ],
        },
        "rt_velocity_audit": {"loader_standardization": "none"},
    }


def _patch_target(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    indexed = OrderedDict((name, _path(name)) for name in SESSION_NAMES)
    monkeypatch.setattr(nested, "index_rt_session_paths", lambda *_args, **_kwargs: indexed)

    def load(path: Path) -> dict:
        name = nested.session_name_from_path(path)
        calls.append(name)
        return _session(name)

    monkeypatch.setattr(nested, "load_rt_session", load)


def test_r4_target_builder_uses_m6_carrier_not_m24_but_keeps_activity_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _patch_target(monkeypatch, calls)
    kwargs = dict(
        data_dir="/synthetic", outer_loso_fold=0, side_feature_shuffle_seed=42,
        carrier_calibration_n_trials=6, activity_calibration_n_trials=24,
        query_start_trial=24, window_size=50, max_trial_length=100,
        interpolate_trials=True, interpolate_trials_kind="cubic", pad_value=-1.0,
        expected_session_count=15, side_feature_mean=np.zeros(4, dtype=np.float32),
        side_feature_std=np.ones(4, dtype=np.float32),
    )
    full, split, _path_value, full_audit = r4_eval.build_r4_outer_target_dataset(
        side_feature_group="afc4_vel", **kwargs
    )
    mb4, mb_split, _path_value, mb4_audit = r4_eval.build_r4_outer_target_dataset(
        side_feature_group="afc4_mb4", **kwargs
    )
    assert calls == [SESSION_NAMES[0], SESSION_NAMES[0]]
    assert split == mb_split
    name = split.outer_target_session
    assert full.calib_n_trials[name] == mb4.calib_n_trials[name] == 24
    assert full.query_start_trial == mb4.query_start_trial == 24
    assert full_audit["carrier_calibration_trials"] == 6
    assert full_audit["prefix_fit"]["trial_index_range"] == [0, 6]
    assert full_audit["raw_prefix_is_not_m24"] is True
    assert full_audit["raw_prefix_descriptor_sha256"] == mb4_audit["raw_prefix_descriptor_sha256"]
    assert full_audit["query_window_identity_sha256"] == mb4_audit["query_window_identity_sha256"]
    full_side = full[0][4]
    mb4_side = mb4[0][4]
    np.testing.assert_array_equal(mb4_side[:, :2], np.zeros_like(mb4_side[:, :2]))
    np.testing.assert_array_equal(full_side[:, 2:], mb4_side[:, 2:])
    assert full[0][2].shape[0] == mb4[0][2].shape[0] == 24


def _manifest(*, budget: int, arm: str, fold: int = 0) -> dict:
    split = nested.nested_loso_partition(SESSION_NAMES, fold, expected_session_count=15)
    return {
        "validation_protocol": "nested_loso",
        "outer_loso_fold": fold,
        "loso_fold": fold,
        "requested_side_feature_group": arm,
        "target_session": split.outer_target_session,
        "session_names": list(split.all_sessions),
        "inner_train_sessions": list(split.inner_train_sessions),
        "inner_validation_session": split.inner_validation_session,
        "nested_selection": {
            "clean": True,
            "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
            "inner_validation_only_for_checkpoint_selection": True,
            "checkpoint_metric": "val_heldin/r2_mean",
            "checkpoint_metric_scope": "inner_validation_session_only",
        },
        "source_only_normalizer": {
            "fit_scope": "inner_train_sessions_only",
            "fit_sessions": list(split.inner_train_sessions),
            "excluded_inner_validation_session": split.inner_validation_session,
            "excluded_outer_target_session": split.outer_target_session,
            "activity_calibration_trials": 24,
            "carrier_calibration_trials": budget,
            "common_query_start_trial": 24,
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0],
        },
        "rt_r4_budget_response": {
            "schema": "rt_r4_budget_response_common_q24_split_v1",
            "activity_calibration_trials": 24,
            "activity_trial_index_range": [0, 24],
            "carrier_calibration_trials": budget,
            "carrier_trial_index_range": [0, budget],
            "unused_for_carrier_and_query_trial_range": [budget, 24],
            "common_query_start_trial": 24,
            "only_carrier_fit_prefix_varies": True,
            "neural_activity_tensor_budget_varies": False,
            "outer_target_loaded_during_fit": False,
        },
    }


def _config(path: Path, *, budget: int, arm: str, fold: int = 0) -> None:
    path.write_text(
        "\n".join([
            "run_id: synthetic_r4", "seed: 42", "test: false", "model: {}", "data:",
            f"  _target_: {r4_eval.R4_DATAMODULE_TARGET}",
            "  data_dir: /synthetic", f"  side_feature_group: {arm}",
            f"  side_feature_calibration_n_trials: {budget}",
            "  calibration_n_trials: 24", "  query_start_trial: 24",
            f"  outer_loso_fold: {fold}", "  side_feature_shuffle_seed: 42",
            "  batch_size: 4", "  window_size: 50", "  max_trial_length: 100",
            "  interpolate_trials: true", "  interpolate_trials_kind: cubic",
            "  pad_value: -1.0", "  expected_session_count: 15", "",
        ]), encoding="utf-8",
    )


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def setup(self, _stage: str) -> None:
        return None

    def model_step(self, batch):
        target = batch[1]
        return {"behavior_pred": target * self.scale, "behavior_target": target}


def test_r4_evaluator_is_one_shot_no_bp_state_identical_and_immutable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[str] = []
    _patch_target(monkeypatch, calls)
    config = tmp_path / "config.yaml"
    split = tmp_path / "split.json"
    checkpoint = tmp_path / "model.ckpt"
    selection = tmp_path / "selection.json"
    output = tmp_path / "outer.json"
    _config(config, budget=6, arm="afc4_vel")
    split.write_text(json.dumps(_manifest(budget=6, arm="afc4_vel")), encoding="utf-8")
    checkpoint.write_bytes(b"synthetic-checkpoint")
    selection.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(r4_eval, "_validate_selection_receipt", lambda **_kwargs: {})
    monkeypatch.setattr(r4_eval, "_restore_selected_checkpoint", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(r4_eval.hydra.utils, "instantiate", lambda _cfg: _TinyModel())
    result = r4_eval.evaluate_r4_outer_target(
        config_path=config, checkpoint_path=checkpoint, split_manifest_path=split,
        selection_receipt_path=selection, output_path=output, outer_loso_fold=0,
        device="cpu",
    )
    assert calls == [SESSION_NAMES[0]]
    assert result["status"] == r4_eval.R4_OUTER_STATUS
    assert result["activity_calibration_trials"] == 24
    assert result["carrier_calibration_trials"] == 6
    assert result["target_carrier_prefix_fit"]["prefix_fit"]["trial_index_range"] == [0, 6]
    assert result["model_state_sha256_before"] == result["model_state_sha256_after"]
    assert result["target_backpropagation"] is False
    assert result["optimizer_present"] is False and result["trainer_present"] is False
    assert result["r2_variance_weighted"] == pytest.approx(1.0)
    assert output.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError, match="overwrite"):
        r4_eval.evaluate_r4_outer_target(
            config_path=config, checkpoint_path=checkpoint, split_manifest_path=split,
            selection_receipt_path=selection, output_path=output, outer_loso_fold=0,
            device="cpu",
        )
    assert calls == [SESSION_NAMES[0]]  # overwrite refusal occurs before another target open


def test_bad_selection_is_rejected_before_target_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    split = tmp_path / "split.json"
    _config(config, budget=12, arm="afc4_mb4")
    split.write_text(json.dumps(_manifest(budget=12, arm="afc4_mb4")), encoding="utf-8")
    opened: list[str] = []
    monkeypatch.setattr(nested, "load_rt_session", lambda _path: opened.append("target"))
    monkeypatch.setattr(
        r4_eval, "_validate_selection_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad selection")),
    )
    with pytest.raises(ValueError, match="bad selection"):
        r4_eval.evaluate_r4_outer_target(
            config_path=config, checkpoint_path=tmp_path / "missing.ckpt",
            split_manifest_path=split, selection_receipt_path=tmp_path / "missing.json",
        )
    assert opened == []


def test_static_plan_is_exact_balanced_and_does_not_inspect_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        pilot.subprocess, "check_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GPU queried")),
    )
    plan = pilot.build_plan(work_root=tmp_path / "runs")
    assert plan["pilot_cell_count"] == 12
    assert plan["scope"]["cuda_queried"] == 0
    assert len(plan["lanes"]["0"]["serial_cells"]) == 6
    assert len(plan["lanes"]["1"]["serial_cells"]) == 6
    matrix = {
        (row["budget"], row["fold"], row["arm"])
        for lane in plan["lanes"].values() for row in lane["serial_cells"]
    }
    assert matrix == {
        (budget, fold, arm)
        for budget in (6, 12) for fold in (0, 7, 14)
        for arm in ("afc4_vel", "afc4_mb4")
    }
    commands = [row["train_command"] for lane in plan["lanes"].values() for row in lane["serial_cells"]]
    assert all("ckpt_path=null" in command and "test=false" in command for command in commands)
    assert all(str(pilot.TRAIN_ENTRY) in command for command in commands)


def _make_cell(
    root: Path, *, budget: int, arm: str, fold: int, score: float,
    query_hash: str | None = None,
) -> None:
    paths = pilot.cell_paths(root, budget=budget, arm=arm, fold=fold)
    paths.fit.mkdir(parents=True, exist_ok=True)
    checkpoint = paths.fit / "checkpoints/best.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-m{budget}-{arm}-{fold}".encode())
    paths.config.parent.mkdir(parents=True, exist_ok=True)
    paths.config.write_text("synthetic-config", encoding="utf-8")
    paths.split.write_text("synthetic-split", encoding="utf-8")
    paths.selection.write_text("synthetic-selection", encoding="utf-8")
    raw_sha = f"{budget:02x}{fold:02x}".ljust(64, "a")
    query = query_hash or f"{fold:02x}".ljust(64, "b")
    normalizer = f"{budget:02x}{fold:02x}".ljust(64, "c")
    state = "d" * 64
    outer = {
        "schema": pilot.OUTER_SCHEMA,
        "status": pilot.OUTER_STATUS,
        "seed": 42,
        "arm": arm,
        "outer_loso_fold": fold,
        "outer_target_session": SESSION_NAMES[PILOT_FOLDS_TO_INDEX[fold]],
        "checkpoint_path": str(checkpoint.resolve()),
        "activity_calibration_trials": 24,
        "carrier_calibration_trials": budget,
        "query_start_trial": 24,
        "query_windows_evaluated": 1000 + fold,
        "query_window_identity_sha256": query,
        "r2_variance_weighted": score,
        "source_only_normalizer_sha256": normalizer,
        "target_carrier_prefix_fit": {
            "status": "PASS_TARGET_ACTIVITY_M24_CARRIER_PREFIX_QUERY_Q24",
            "carrier_calibration_trials": budget,
            "raw_prefix_descriptor_sha256": raw_sha,
            "prefix_fit": {"trial_index_range": [0, budget]},
        },
        "target_backpropagation": False,
        "optimizer_present": False,
        "trainer_present": False,
        "model_training_mode": False,
        "model_state_unchanged": True,
        "model_state_sha256_before": state,
        "model_state_sha256_after": state,
    }
    pilot._write_immutable(paths.outer, outer)
    pilot._write_cell_terminal(
        paths=paths, budget=budget, arm=arm, fold=fold, outer=outer
    )


PILOT_FOLDS_TO_INDEX = {0: 0, 7: 7, 14: 14}


def test_budgetwise_gate_retains_signs_and_only_expands_passing_budget(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    # M6: +.04 on all three -> expand. M12: one negative -> stop.
    for fold in pilot.PILOT_FOLDS:
        _make_cell(root, budget=6, arm="afc4_vel", fold=fold, score=0.54)
        _make_cell(root, budget=6, arm="afc4_mb4", fold=fold, score=0.50)
    m12_deltas = {0: 0.08, 7: -0.01, 14: 0.05}
    for fold, delta in m12_deltas.items():
        _make_cell(root, budget=12, arm="afc4_vel", fold=fold, score=0.50 + delta)
        _make_cell(root, budget=12, arm="afc4_mb4", fold=fold, score=0.50)
    output = tmp_path / "aggregate.json"
    result = pilot.aggregate_pilot(work_root=root, output=output)
    assert result["budget_decisions"]["6"]["decision"].startswith("EXPAND")
    assert result["budget_decisions"]["6"]["gate"]["passed"] is True
    assert result["budget_decisions"]["12"]["decision"] == "STOP_THIS_BUDGET"
    assert result["budget_decisions"]["12"]["gate"]["passed"] is False
    assert result["budget_decisions"]["12"]["signed_full_minus_mb4"] == pytest.approx(
        [0.08, -0.01, 0.05]
    )
    assert output.stat().st_mode & 0o777 == 0o444
    body = json.loads(output.read_text())
    assert len(body["rows"]) == 6
    assert all(row["state_identity_verified"] for row in body["rows"])


def test_gate_rejects_query_hash_drift_across_budget(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    for budget in pilot.BUDGETS:
        for fold in pilot.PILOT_FOLDS:
            query = "e" * 64 if (budget, fold) == (12, 7) else f"{fold:02x}".ljust(64, "b")
            for arm, score in (("afc4_vel", 0.55), ("afc4_mb4", 0.50)):
                _make_cell(root, budget=budget, arm=arm, fold=fold, score=score, query_hash=query)
    with pytest.raises(pilot.RtR4PilotError, match="query hash differs"):
        pilot.aggregate_pilot(work_root=root, output=tmp_path / "aggregate.json")


def test_superseding_execution_preflight_binds_old_receipt_and_launches_nothing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "execution_preflight.json"
    body = preflight.build_receipt(output=output)
    assert body["supersedes_without_mutating"]["sha256"] == pilot.PREPARE_RECEIPT_SHA256
    assert body["original_prepare_receipt_retained_as_historical_source_only_contract"] is True
    assert body["static_dual_3090_pilot"]["pilot_cell_count"] == 12
    assert body["scope"] == {
        "nwb_files_opened": 0, "outer_target_payloads_opened": 0,
        "trainer_constructed": 0, "optimizer_constructed": 0,
        "cuda_queried": 0, "gpu_processes_started": 0,
        "tmux_sessions_created": 0, "pilot_commands_executed": 0,
    }
    path, _digest = preflight.write_immutable(output, body)
    assert path.stat().st_mode & 0o777 == 0o444
    validated = pilot.validate_execution_preflight(path)
    assert validated["status"] == pilot.EXECUTION_PREFLIGHT_STATUS

