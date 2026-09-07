"""Target-free contracts for the strict H1 date-LODO terminal evaluator.

None of these tests calls :func:`evaluate` with a valid preflight or imports
the isolated target-data module.  The only tensors below are synthetic test
values, so a passing suite cannot open a date-LODO target recording.
"""
from __future__ import annotations

import json
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from scripts import h1_carrierid_date_lodo_phase2_terminal_evaluate as evaluator
from scripts import h1_carrierid_date_lodo_phase2_terminal_preflight as terminal_preflight
from scripts.h1_carrierid_date_lodo_phase2_terminal_preflight import (
    CLOSURE_FILES,
    DateLodoEvaluatorPreflightError,
    _require_terminal_pair,
)
from src.data import h1_carrierid_date_lodo_target as target_view
from src.h1_m4_cce_contract import array_sha256, sha256_file


def _immutable_json(path: Path, body: dict[str, object]) -> Path:
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)
    return path


def _row(tmp_path: Path, arm: str, *, epoch: int = 49, warm_start: bool = False) -> dict[str, object]:
    checkpoint, config = tmp_path / f"{arm}.ckpt", tmp_path / f"{arm}.yaml"
    checkpoint.write_bytes(b"synthetic checkpoint only; never loaded by this test")
    config.write_text("synthetic: true\n", encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": str(config), "config_sha256": sha256_file(config),
        "metadata": {
            "arm": arm, "checkpoint_epoch_zero_based": epoch,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
            "checkpoint_warm_start": warm_start,
            "phase2_source_binding_sha256": "a" * 64,
        },
    }


def _valid_preflight_rows(tmp_path: Path) -> dict[str, object]:
    return {"checkpoints": {"H-S": _row(tmp_path, "H-S"), "H-C": _row(tmp_path, "H-C")}}


def _synthetic_source_bundle(tmp_path: Path, *, tamper: str | None = None) -> Path:
    """Minimal immutable Phase-1 bundle fixture; never contains a recording."""

    directory = tmp_path / "source_bundle"; directory.mkdir()
    arrays = {
        "mean": np.arange(176, dtype=np.float64), "scale": np.ones(176, dtype=np.float64),
        "pcs": np.ones((2, 176), dtype=np.float64), "U": np.ones((7, 4), dtype=np.float64),
        "mu": np.zeros(4, dtype=np.float64),
    }
    arrays_path = directory / "frozen_m4_plan.npz"
    np.savez(arrays_path, **arrays, q=np.asarray(2, dtype=np.int64), **{"lambda": np.asarray(1.0)}, tau2=np.asarray(0.5))
    arrays_path.chmod(0o444)
    plan = {
        "outer_date": "19250108", "source_sessions": ["source-a"], "source_input_sha256": ["s" * 64],
        "q": 2, "lambda": 1.0, "tau2": 0.5, "transform_sha256": "t" * 64,
        "raw_receipt_sha256": "r" * 64, "eb_receipt_sha256": "e" * 64,
        "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
        "array_shape": {name: list(value.shape) for name, value in arrays.items()},
        # Historical real bundles deliberately carry null here; immutable
        # per-array digest/shape authority is what the target loader must use.
        "arrays_file_sha256": None,
    }
    if tamper == "array":
        plan["array_sha256"] = {**plan["array_sha256"], "U": "0" * 64}
    if tamper == "scalar":
        plan["q"] = 3
    plan_path = _immutable_json(directory / "frozen_m4_plan.manifest.json", plan)
    normalizer = {
        "s_src": 1.0, "source_cache_sha256": "c" * 64, "entries": 1, "rows": 176, "dims": 4,
        "normalizer_sha256": "n" * 64,
    }
    normalizer_path = _immutable_json(directory / "source_rms_normalizer.manifest.json", normalizer)
    source = {
        "source_sessions": ["source-a"], "source_files": [{"sha256": "s" * 64}],
        "frozen_plan": {"manifest_path": str(plan_path), "manifest_sha256": sha256_file(plan_path),
                        "transform_sha256": "t" * 64, "raw_receipt_sha256": "r" * 64, "eb_receipt_sha256": "e" * 64},
        "normalizer": {"manifest_path": str(normalizer_path), "manifest_file_sha256": sha256_file(normalizer_path),
                       "normalizer_sha256": "n" * 64},
    }
    return _immutable_json(directory / "shared_source_manifest.json", source)


class _SyntheticStrictDataset(Dataset):
    """A local four-sample, two-session dataset; it is not an H1 data view."""

    window_indices_sha256 = "synthetic_same_support_query_windows"

    def __init__(self) -> None:
        self._sessions = ("date-session-a", "date-session-a", "date-session-b", "date-session-b")

    def __len__(self) -> int:
        return len(self._sessions)

    def __getitem__(self, index: int):
        value = np.float32(index + 1)
        neural = np.full((2, 7), value, dtype=np.float32)
        target = neural.copy()
        identity = np.zeros((4, 2, 7), dtype=np.float32)
        carrier = np.zeros((7, 4), dtype=np.float32)
        return neural, target, identity, self._sessions[index], carrier


class _FrozenEcho(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.hparams = SimpleNamespace(decode_last_timestep_only=True, predict_scaled_behavior=False)

    def forward(self, neural, *, calib_trialized_neural_features, carrier):
        del calib_trialized_neural_features, carrier
        return neural + self.bias


def test_evaluator_requires_explicit_execute_flag_before_any_evaluation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = False

    def _must_not_run(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("evaluate must not be reached without explicit flag")

    monkeypatch.setattr(evaluator, "evaluate", _must_not_run)
    monkeypatch.setattr("sys.argv", ["terminal_evaluate.py", "--preflight", str(tmp_path / "none.json"),
                                      "--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit, match="refusing target access"):
        evaluator.main()
    assert called is False


def test_missing_preflight_or_any_hs_hc_terminal_binding_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(evaluator.DateLodoEvaluatorError, match="immutable mode-0444 preflight"):
        evaluator._read_preflight(tmp_path / "absent_preflight.json")

    valid = _valid_preflight_rows(tmp_path)
    checked = evaluator._checkpoint_config_rows(valid)
    assert tuple(checked) == ("H-S", "H-C")
    for broken in (
        {"checkpoints": {"H-S": valid["checkpoints"]["H-S"]}},
        {"checkpoints": {"H-S": {**valid["checkpoints"]["H-S"], "checkpoint_path": str(tmp_path / "missing.ckpt")},
                         "H-C": valid["checkpoints"]["H-C"]}},
        {"checkpoints": {"H-S": _row(tmp_path, "H-S", epoch=48), "H-C": _row(tmp_path, "H-C")}},
        {"checkpoints": {"H-S": _row(tmp_path, "H-S", warm_start=True), "H-C": _row(tmp_path, "H-C")}},
    ):
        with pytest.raises(evaluator.DateLodoEvaluatorError):
            evaluator._checkpoint_config_rows(broken)


def test_preflight_runtime_path_binding_keeps_isolated_code_and_canonical_data_separate(tmp_path: Path) -> None:
    """The remote isolated stage intentionally has no ``data/000954`` copy."""

    canonical = tmp_path / "SPINT-main"
    target_data_root = canonical / "data" / "000954"
    target_data_root.mkdir(parents=True)
    source_manifest = canonical / "pilot_artifacts" / "phase1" / "shared_source_manifest.json"
    source_manifest.parent.mkdir(parents=True)
    _immutable_json(source_manifest, {"source_only": True})
    preflight = {
        "runtime": {
            "isolated_stage_root": str(evaluator.ROOT.resolve()),
            "canonical_data_repository_root": str(canonical),
            "target_data_root": str(target_data_root),
        },
        "source_binding": {
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
        },
    }
    assert evaluator._target_data_root_from_preflight(preflight) == target_data_root.resolve()
    assert terminal_preflight._canonical_target_data_root(target_data_root) == (target_data_root.resolve(), canonical.resolve())

    bad_stage = {**preflight, "runtime": {**preflight["runtime"], "isolated_stage_root": str(tmp_path / "other-stage")}}
    with pytest.raises(evaluator.DateLodoEvaluatorError, match="exact isolated-stage"):
        evaluator._target_data_root_from_preflight(bad_stage)
    with pytest.raises(terminal_preflight.DateLodoEvaluatorPreflightError, match="canonical SPINT-main/data/000954"):
        terminal_preflight._canonical_target_data_root(tmp_path / "not_the_data_root")


def test_preopen_pair_binding_rejects_missing_hs_hc_e49_or_mismatched_checker(tmp_path: Path) -> None:
    pair = _immutable_json(tmp_path / "pair_preflight.json", {"source_only": True})
    rows = _valid_preflight_rows(tmp_path)["checkpoints"]
    checker = {
        "pair_preflight": {"path": str(pair), "sha256": sha256_file(pair)},
        "h_s": rows["H-S"], "h_c": rows["H-C"],
        "equal_schedule_verified_fields": ["phase2_source_binding_sha256", "phase1_source_manifest_sha256", "phase1_preflight_sha256"],
    }
    result = _require_terminal_pair(checker, pair, sha256_file(pair))
    assert set(result) == {"H-S", "H-C"}
    broken = dict(checker)
    broken["h_c"] = None
    with pytest.raises(DateLodoEvaluatorPreflightError, match="H-C"):
        _require_terminal_pair(broken, pair, sha256_file(pair))
    broken = dict(checker)
    broken["pair_preflight"] = {"path": str(pair), "sha256": "0" * 64}
    with pytest.raises(DateLodoEvaluatorPreflightError, match="another pair preflight"):
        _require_terminal_pair(broken, pair, sha256_file(pair))


def test_target_dependency_loader_uses_manifest_array_scalar_authority_and_accepts_real_7_by_4_u(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This isolates the array/scalar integrity logic from Phase-1's broader
    # source-partition validator; no NWB loader can be reached by this test.
    monkeypatch.setattr(target_view, "validate_source_bundle_manifest", lambda *_args, **_kwargs: None)
    plan, normalizer, _manifest = target_view.load_target_dependencies(_synthetic_source_bundle(tmp_path), outer_date="19250108")
    assert plan.U.shape == (7, 4)
    assert plan.U.shape[1] == 4
    assert normalizer.denominator == pytest.approx(1.0)
    for tamper, needle in (("array", "U array shape/SHA drift"), ("scalar", "scalar q/lambda/tau2 drift")):
        broken_root = tmp_path / tamper; broken_root.mkdir()
        with pytest.raises(target_view.DateLodoTargetError, match=needle):
            target_view.load_target_dependencies(_synthetic_source_bundle(broken_root, tamper=tamper), outer_date="19250108")


def test_target_open_is_textually_after_all_receipt_checkpoint_source_and_model_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/h1_carrierid_date_lodo_phase2_terminal_evaluate.py").read_text(encoding="utf-8")
    first_open = source.index("load_outer_date_target_records")
    for prerequisite in ("_read_preflight(preflight_path)", "_checkpoint_config_rows(preflight)",
                         "_require_one_shot_evaluation_slot", "_target_data_root_from_preflight(preflight)", "load_target_dependencies",
                         "_instantiate(rows[\"H-S\"]", "_instantiate(rows[\"H-C\"]"):
        assert source.index(prerequisite) < first_open
    assert 'ROOT / "data/000954"' not in source
    preflight_source = (root / "scripts/h1_carrierid_date_lodo_phase2_terminal_preflight.py").read_text(encoding="utf-8")
    assert "load_outer_date_target_records" not in preflight_source
    assert any(path.endswith("test_h1_carrierid_date_lodo_phase2_terminal_evaluator_contract.py") for path in CLOSURE_FILES)
    required_runtime = {
        "src/data/h1_carrierid_date_lodo_source.py", "src/data/h1_m4_eb_pilot.py", "src/h1_m4_cce_contract.py",
        "src/models/h1_carrierid_date_lodo_phase2_module.py", "src/models/falcon_module.py",
        "src/models/components/spint.py", "src/models/components/h1_carrierid_spint.py",
    }
    assert required_runtime.issubset(set(CLOSURE_FILES))
    required_control_path = {
        "scripts/h1_carrierid_date_lodo_phase2_preflight.py",
        "scripts/h1_carrierid_date_lodo_future_remote_queue.py",
        "scripts/h1_carrierid_date_lodo_phase2_terminal_checker.py",
        "tests/test_h1_carrierid_date_lodo_phase2_contract.py",
        "tests/test_h1_carrierid_date_lodo_future_remote_queue.py",
        "tests/test_h1_carrierid_date_lodo_phase2_terminal_checker_contract.py",
    }
    assert required_control_path.issubset(set(CLOSURE_FILES))
    checker_source = (root / "scripts/h1_carrierid_date_lodo_phase2_terminal_checker.py").read_text(encoding="utf-8")
    for target_loader in ("load_outer_date_target_records", "from src.data.h1_carrierid_date_lodo_target", "import src.data.h1_carrierid_date_lodo_target"):
        assert target_loader not in preflight_source
        assert target_loader not in checker_source


def test_terminal_evaluator_one_shot_slot_rejects_alternate_filename_and_any_prior_same_date_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This exercises only receipt paths; it cannot reach a target loader."""

    stage = tmp_path / "isolated-stage"
    artifact_dir = stage / "pilot_artifacts" / "h1_carrierid_date_lodo_phase2" / "terminal_evaluations"
    monkeypatch.setattr(evaluator, "ROOT", stage)
    monkeypatch.setattr(evaluator, "EVALUATION_ARTIFACT_DIR", artifact_dir)
    canonical = evaluator._canonical_evaluation_output("19250113")
    assert evaluator._require_one_shot_evaluation_slot(outer_date="19250113", output_path=canonical) == canonical
    with pytest.raises(evaluator.DateLodoEvaluatorError, match="canonical per-date"):
        evaluator._require_one_shot_evaluation_slot(
            outer_date="19250113", output_path=tmp_path / "different_name.json"
        )

    # This represents a receipt emitted by an older evaluator that let the
    # caller choose an alternate output filename.  Its mere existence blocks a
    # second target opening even though the new canonical slot is still empty.
    artifact_dir.mkdir(parents=True)
    _immutable_json(artifact_dir / "old_alternate_name.json", {
        "schema": evaluator.EVALUATION_SCHEMA,
        "status": evaluator._evaluation_status("19250113"),
        "outer_date": "19250113",
    })
    with pytest.raises(evaluator.DateLodoEvaluatorError, match="repeat target evaluation"):
        evaluator._require_one_shot_evaluation_slot(outer_date="19250113", output_path=canonical)


def test_evaluator_has_no_optimizer_or_backward_path_and_requires_frozen_no_warmstart_metadata() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/h1_carrierid_date_lodo_phase2_terminal_evaluate.py").read_text(encoding="utf-8")
    for forbidden in ("torch.optim", ".backward(", "configure_optimizers", ".fit(", "optimizer.step("):
        assert forbidden not in source
    assert 'meta.get("checkpoint_warm_start") is False' in source
    assert '"optimizer_steps": 0' in source and '"backward_steps": 0' in source
    assert "parameter.requires_grad_(False)" in source


def test_float64_pooled_per_recording_r2_delta_state_hash_and_shared_window_contract() -> None:
    dataset, model = _SyntheticStrictDataset(), _FrozenEcho()
    result = evaluator._evaluate(model, dataset, torch.device("cpu"), ("date-session-a", "date-session-b"))
    assert result["pooled_r2"] == pytest.approx(1.0)
    assert set(result["per_session"]) == {"date-session-a", "date-session-b"}
    assert result["r2_accumulator_dtype"] == "float64"
    assert result["state_immutable"] is True
    assert result["state_sha256_before"] == result["state_sha256_after"]
    assert result["query_window_indices_sha256"] == dataset.window_indices_sha256
    source = (Path(__file__).resolve().parents[1] / "scripts/h1_carrierid_date_lodo_phase2_terminal_evaluate.py").read_text(encoding="utf-8")
    assert '"h_c_minus_h_s"' in source
    assert "hs_metrics, hc_metrics = _evaluate(hs_model, dataset" in source
    assert "_evaluate(hc_model, dataset" in source
    assert "H-S/H-C target datasets do not share exact strict support/query windows" in source
