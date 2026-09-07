"""Focused no-target/no-GPU tests for the isolated h=32 H-LS implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import numpy as np
from omegaconf import OmegaConf
import pytest

from scripts import h1_carrierid_date_lodo_hls_source_executor as executor
from scripts import h1_carrierid_date_lodo_hls_source_preflight as source_preflight
from scripts import h1_carrierid_date_lodo_hls_terminal_evaluate as terminal_evaluator
from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    LAUNCH_SCHEMA, LAUNCH_STATUS, SOURCE_PREFLIGHT_SCHEMA, SOURCE_PREFLIGHT_STATUS,
)
from src.data import h1_carrierid_date_lodo_hls_target as hls_target
from src.data.h1_carrierid_date_lodo_target import DateLodoTargetSupport


ROOT = Path(__file__).resolve().parents[1]


def _immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hls_network_config_is_exact_hc_h32_topology_not_ci64():
    hc = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hc.yaml"), resolve=False)
    hls = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hls.yaml"), resolve=False)
    assert hls["net"] == hc["net"]
    assert hls["net"]["_target_"] == "src.models.components.h1_carrierid_spint.H1CarrierIdSpint"
    assert hls["net"]["carrier_hidden_dim"] == 32
    assert "ci" not in hls["net"]["_target_"].lower()
    for field in (
        "decode_last_timestep_only", "predict_scaled_behavior", "behavior_scaling_factor",
        "optimizer", "scheduler", "compile", "clean_teacher", "scheduler_monitor",
    ):
        assert hls[field] == hc[field]


def test_hls_source_preflight_composes_exact_fixed_route_without_data_access(tmp_path: Path):
    _cfg, raw = source_preflight._compose(
        outer_date="19250113", phase1_preflight=tmp_path / "phase1.json",
        hls_source_preflight=tmp_path / "hls.json",
    )
    assert str(raw["phase2"]["outer_date"]) == "19250113"
    assert raw["phase2"]["arm"] == "H-LS"
    assert raw["model"]["net"]["carrier_hidden_dim"] == 32
    assert raw["trainer"]["max_epochs"] == raw["trainer"]["min_epochs"] == 50
    assert raw["trainer"]["limit_val_batches"] == 0


def test_hls_source_module_has_no_target_or_ci_dependency_and_only_one_intervention():
    source = (ROOT / "src/data/h1_carrierid_date_lodo_hls.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith("from ") or line.startswith("import ")]
    assert not any("target" in line for line in imports)
    assert not any("date_lodo_ci" in line for line in imports)
    assert "label_rotation_carrier" in source
    assert "complete_row_shuffle" not in source
    assert '"same_h_c_source_windows": True' in source
    assert '"same_h_c_source_schedule": True' in source
    assert '"same_h_c_normalizer": True' in source


def test_hls_target_changes_only_carrier(monkeypatch: pytest.MonkeyPatch):
    identity = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    full = np.full((4, 4), 2.0, dtype=np.float32)
    support = DateLodoTargetSupport(
        "session", (1.0, 2.0, 3.0, 4.0), 5.0, 17, identity, full,
        "support-sha", hashlib.sha256(full.tobytes()).hexdigest(),
    )

    def fake_parent_init(self, records, plan, normalizer, *, outer_date):
        self.records = dict(records); self.plan = plan; self.normalizer = normalizer; self.outer_date = outer_date
        self.support = {"session": support}; self.window_indices = [("session", 17)]
        self.window_indices_sha256 = "w" * 64

    def fake_parent_manifest(self):
        return {
            "schema": "base", "outer_date": self.outer_date, "sessions": ["session"],
            "window_indices_sha256": self.window_indices_sha256, "samples": 1,
            "all_query_histories_start_at_or_after_fifth_trial": True,
            "support": {"session": {"support_trials": [1.0, 2.0, 3.0, 4.0],
                                      "fifth_trial": 5.0, "query_first_bin": 17,
                                      "support_sha256": "support-sha",
                                      "normalized_carrier_sha256": support.carrier_sha256,
                                      "identity_sha256": "old"}},
        }

    monkeypatch.setattr(hls_target.H1CarrierIdDateLodoStrictTargetDataset, "__init__", fake_parent_init)
    monkeypatch.setattr(hls_target.H1CarrierIdDateLodoStrictTargetDataset, "manifest", fake_parent_manifest)
    changed = np.full((4, 4), 3.0, dtype=np.float64)
    monkeypatch.setattr(hls_target, "label_rotation_carrier", lambda record, plan, values: changed)

    class Normalizer:
        @staticmethod
        def normalize(value):
            return np.asarray(value) / 3.0

    dataset = hls_target.H1CarrierIdDateLodoHlsStrictTargetDataset(
        {"session": object()}, object(), Normalizer(), outer_date="19250108",
    )
    observed = dataset.support["session"]
    assert observed.support_trials == support.support_trials
    assert observed.query_first_bin == support.query_first_bin
    assert observed.support_sha256 == support.support_sha256
    assert np.array_equal(observed.identity, identity)
    assert np.array_equal(observed.normalized_carrier, np.ones((4, 4), dtype=np.float32))
    assert not np.array_equal(observed.normalized_carrier, full)
    manifest = dataset.manifest()
    assert manifest["carrier_intervention"] == "temporal_velocity_label_rotation"
    assert manifest["same_h_c_support_identity_and_query_windows"] is True


def test_source_executor_default_only_builds_exact_fresh_command(tmp_path: Path):
    code_paths = {
        "data": ROOT / "src/data/h1_carrierid_date_lodo_hls.py",
        "model": ROOT / "src/models/h1_carrierid_date_lodo_hls_module.py",
        "component": ROOT / "src/models/components/h1_carrierid_spint.py",
        "experiment": ROOT / "configs/experiment/h1_carrierid_date_lodo_hls_phase2.yaml",
        "data_config": ROOT / "configs/data/falcon_h1_carrierid_date_lodo_hls.yaml",
        "model_config": ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hls.yaml",
        "terminal_callback": ROOT / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml",
    }
    preflight_body = {
        "schema": SOURCE_PREFLIGHT_SCHEMA, "status": SOURCE_PREFLIGHT_STATUS,
        "outer_date": "19250108", "source_binding_sha256": "b" * 64,
        "source_binding": {"preflight_path": str(tmp_path / "phase1.json")},
        "source_controls": {"carrier_intervention": "temporal_velocity_label_rotation"},
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0},
        "code_sha256": {name: _sha(path) for name, path in code_paths.items()},
    }
    preflight = _immutable(tmp_path / "source.json", preflight_body)
    launch_body = {
        "schema": LAUNCH_SCHEMA, "status": LAUNCH_STATUS,
        "route": "H1-HLS-FIVEDATE", "not_a_gpu_launcher": True, "launch_authorized": False,
        "training_contract": {"arm": "H-LS", "fresh_seed": 42, "epochs": 50,
                              "fixed_terminal_epoch_zero_based": 49, "checkpoint_warm_start": False},
        "fixed_grid": ["19250108", "19250113", "19250115", "19250119", "19250120"],
        "source_preflights": {"19250108": {"path": str(preflight), "sha256": _sha(preflight),
                                                "source_binding_sha256": "b" * 64}},
    }
    launch = _immutable(tmp_path / "launch.json", launch_body)
    plan = executor.build_command(launch_receipt=launch, outer_date="19250108",
                                  run_dir=tmp_path / "new_run", python_executable="python")
    command = plan["command"]
    assert command[:3] == ["python", str(ROOT / "src/train.py"),
                           "experiment=h1_carrierid_date_lodo_hls_phase2"]
    assert "phase2.outer_date=19250108" in command
    assert "seed=42" in command and "ckpt_path=null" in command and "test=false" in command
    assert not (tmp_path / "new_run").exists()


def test_target_evaluator_is_explicit_and_imports_target_only_after_receipt_binding():
    source = (ROOT / "scripts/h1_carrierid_date_lodo_hls_terminal_evaluate.py").read_text(encoding="utf-8")
    binding_check = source.index('"H-C and H-LS do not share the exact Phase-1 source estimator binding"')
    target_import = source.index('importlib.import_module("src.data.h1_carrierid_date_lodo_target")')
    assert binding_check < target_import
    assert "--execute-target-evaluation" in source
    assert "with torch.no_grad()" in source
    assert '"optimizer_steps": 0, "backward_steps": 0' in source
    assert "state_sha256_before" in source and "state_sha256_after" in source
    reproduction_call = source.index("hc_reproduction = _validate_original_hc_reproduction")
    receipt_write = source.index("written, digest = write_immutable_json")
    assert reproduction_call < receipt_write
    assert '"original_hc_reproduction_check": hc_reproduction' in source


def _hc_metric(*, pooled_r2: float = 0.625) -> dict[str, object]:
    return {
        "pooled_r2": pooled_r2,
        "per_session": {
            "session_a": {"samples": 32, "r2": 0.5},
            "session_b": {"samples": 17, "r2": 0.75},
        },
        "samples": 49, "batches": 2, "last_batch_size": 17,
        "r2_accumulator_dtype": "float64",
        "query_window_indices_sha256": "q" * 64,
    }


def test_original_hc_reproduction_check_records_tolerances_and_differences():
    expected = _hc_metric()
    observed = _hc_metric(
        pooled_r2=float(expected["pooled_r2"]) + terminal_evaluator.HC_REPRODUCTION_R2_ABS_TOLERANCE / 2,
    )
    observed["per_session"]["session_b"]["r2"] += terminal_evaluator.HC_REPRODUCTION_R2_ABS_TOLERANCE / 2
    result = terminal_evaluator._validate_original_hc_reproduction(
        {"metrics": {"h_c": expected}}, observed,
    )
    assert result["passed"] is True
    assert result["required_before_hls_paired_result_publication"] is True
    assert result["tolerance"] == {
        "r2_absolute": terminal_evaluator.HC_REPRODUCTION_R2_ABS_TOLERANCE,
        "r2_relative": 0.0,
    }
    assert result["exact_checks"]["samples"]["equal"] is True
    assert result["pooled_r2"]["absolute_difference"] > 0.0
    assert result["per_recording"]["session_b"]["r2"]["absolute_difference"] > 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("metrics"), "lacks metrics mapping"),
        (lambda value: value["metrics"]["h_c"].update(query_window_indices_sha256="x" * 64),
         "query_window_indices_sha256"),
        (lambda value: value["metrics"]["h_c"].update(samples=48), "samples"),
        (lambda value: value["metrics"]["h_c"].update(batches=3), "batches"),
        (lambda value: value["metrics"]["h_c"].update(last_batch_size=16), "last_batch_size"),
        (lambda value: value["metrics"]["h_c"].update(
            pooled_r2=0.625 + 2 * terminal_evaluator.HC_REPRODUCTION_R2_ABS_TOLERANCE), "pooled standard R2"),
        (lambda value: value["metrics"]["h_c"]["per_session"].update(
            renamed=value["metrics"]["h_c"]["per_session"].pop("session_b")), "names/order"),
        (lambda value: value["metrics"]["h_c"]["per_session"]["session_b"].update(samples=16),
         "recording sample count"),
        (lambda value: value["metrics"]["h_c"]["per_session"]["session_b"].update(
            r2=0.75 + 2 * terminal_evaluator.HC_REPRODUCTION_R2_ABS_TOLERANCE), "recording R2"),
    ],
)
def test_original_hc_reproduction_check_fails_closed_on_every_required_mismatch(mutation, message):
    original = {"metrics": {"h_c": _hc_metric()}}
    mutation(original)
    with pytest.raises(terminal_evaluator.HlsTerminalEvaluationError, match=message):
        terminal_evaluator._validate_original_hc_reproduction(original, _hc_metric())


def test_all_actual_hls_modules_compile_without_opening_data_or_cuda():
    paths = [
        "src/data/h1_carrierid_date_lodo_hls.py",
        "src/data/h1_carrierid_date_lodo_hls_target.py",
        "src/models/h1_carrierid_date_lodo_hls_module.py",
        "scripts/h1_carrierid_date_lodo_hls_source_preflight.py",
        "scripts/h1_carrierid_date_lodo_hls_source_executor.py",
        "scripts/h1_carrierid_date_lodo_hls_source_terminal_audit.py",
        "scripts/h1_carrierid_date_lodo_hls_target_evaluator_closure.py",
        "scripts/h1_carrierid_date_lodo_hls_terminal_evaluate.py",
        "scripts/h1_carrierid_date_lodo_hls_fivedate_aggregate.py",
    ]
    for relative in paths:
        compile((ROOT / relative).read_text(encoding="utf-8"), relative, "exec")
