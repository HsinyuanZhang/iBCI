"""CPU-only contract tests for C2's sampling-objective factorial."""
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

from src.metrics import c2_m2_sampling as core
from scripts.preflight_c2_m2_sampling import build_preflight


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
PY = sys.executable


def _init_args(path: Path, class_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return [arg.arg for arg in item.args.args]
    raise AssertionError(f"{class_name}.__init__ not found in {path}")


def _receipt(spec: core.CellSpec, score: float, *, preflight_sha: str, session_name: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": core.SCHEMA_VERSION,
        "screen_id": core.SCREEN_ID,
        "sampling": spec.sampling,
        "carrier": spec.carrier,
        "fold": spec.fold,
        "seed": spec.seed,
        "session_name": session_name or f"m2-dev-fold-{spec.fold}",
        "score": score,
        "epoch_window": list(core.EPOCH_WINDOW),
        "loss_mode": core.LOSS_MODE,
        "r2_native_loss": False,
        "balance_session_batches": spec.balance_session_batches,
        "window_budget_per_session": None,
        "development_scope_only": True,
        "formal_test_or_external_heldout_opened": False,
        "sealed_formal_test_sessions_opened": False,
        "official_preflight_sha256": preflight_sha,
    }


def _write_receipts(root: Path, scores: dict[core.CellSpec, float], *, preflight_sha: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec, score in scores.items():
        path = root / f"{spec.sampling}_{spec.carrier}_f{spec.fold}_s{spec.seed}.json"
        core.write_immutable_json(path, _receipt(spec, score, preflight_sha=preflight_sha))
        paths[spec.key] = path
    return paths


def test_equal_session_lever_exists_only_on_streaming_m2_sampler() -> None:
    streaming = REPO / "streaming_calibration_exp/src/data/falcon_datamodule.py"
    sua = REPO / "sua_exploration/mc_maze/multisession_datamodule.py"
    spint_main = REPO / "SPINT-main/src/data/falcon_datamodule.py"
    streaming_args = _init_args(streaming, "SessionBatchSampler")
    assert "balance_sessions" in streaming_args
    assert "window_budget_per_session" in streaming_args
    assert "balance_sessions" not in _init_args(sua, "SessionBatchSampler")
    assert "window_budget_per_session" not in _init_args(sua, "SessionBatchSampler")
    assert "balance_sessions" not in _init_args(spint_main, "SessionBatchSampler")
    assert "window_budget_per_session" not in _init_args(spint_main, "SessionBatchSampler")
    source = streaming.read_text(encoding="utf-8")
    assert source.splitlines()[core.EXISTING_LEVER["sampler_class_line"] - 1].startswith("class SessionBatchSampler")
    assert "balance_sessions=False" in source.splitlines()[core.EXISTING_LEVER["balance_sessions_argument_line"] - 1]
    assert "balance_session_batches: float | bool = False" in source.splitlines()[
        core.EXISTING_LEVER["datamodule_argument_line"] - 1
    ]


def test_ordinary_m2_config_keeps_balance_default_off() -> None:
    text = (PROJECT / "configs/data/falcon_m2.yaml").read_text(encoding="utf-8")
    assert "balance_session_batches: false" in text
    assert "window_budget_per_session" not in text


def test_frozen_gate_can_pass_and_fail_with_distinct_verdicts() -> None:
    passing = core.evaluate_sampling_gates(
        core.synthetic_score_matrix(t4_delta=0.05, z4_delta=0.01),
        bootstrap_draws=101,
    )
    too_small = core.evaluate_sampling_gates(
        core.synthetic_score_matrix(t4_delta=0.01, z4_delta=-0.04),
        bootstrap_draws=101,
    )
    generic = core.evaluate_sampling_gates(
        core.synthetic_score_matrix(t4_delta=0.05, z4_delta=0.05),
        bootstrap_draws=101,
    )
    assert passing["passed"] is True
    assert passing["classification"] == "sampling_objective_pass"
    assert passing["mean_t4_delta"] == pytest.approx(0.05)
    assert passing["mean_interaction"] == pytest.approx(0.04)
    assert too_small["passed"] is False
    assert too_small["classification"] == "t4_session_mean_r2_lift_below_floor_stop"
    assert generic["passed"] is False
    assert generic["classification"] == "generic_z4_lift_or_no_carrier_specificity_stop"
    assert len({passing["classification"], too_small["classification"], generic["classification"]}) == 3
    assert passing["wilcoxon_computed"] is False
    assert generic["gates"]["t4_mean_delta_at_least_floor"] is True
    assert generic["gates"]["interaction_mean_at_least_floor"] is False


def test_lattice_is_exactly_thirty_six_cells_and_incomplete_matrix_fails_closed() -> None:
    cells = core.expected_cells()
    assert len(cells) == 36
    assert {cell.seed for cell in cells} == {42, 43, 44}
    assert {cell.fold for cell in cells} == {0, 1, 2}
    scores = core.synthetic_score_matrix(t4_delta=0.05, z4_delta=0.01)
    scores.pop(next(iter(scores)))
    with pytest.raises(core.C2ContractError, match="incomplete or extra C2 matrix"):
        core.evaluate_sampling_gates(scores, bootstrap_draws=11)


def test_sealed_subc_sessions_are_refused() -> None:
    for name in sorted(core.SEALED_FORMAL_TEST_SESSIONS):
        with pytest.raises(core.C2ContractError, match="sealed sub-C"):
            core.refuse_sealed_sessions([name])
    spec = core.CellSpec("legacy", "t4", 0, 42)
    receipt = _receipt(spec, 0.4, preflight_sha="a" * 64, session_name="sub-C_ses-CO-20151113")
    with pytest.raises(core.C2ContractError, match="sealed sub-C"):
        core.validate_cell_receipt(receipt, spec)


def test_receipt_rejects_r2_loss_and_window_budget() -> None:
    spec = core.CellSpec("equal_session", "t4", 1, 43)
    r2 = _receipt(spec, 0.4, preflight_sha="a" * 64)
    r2["r2_native_loss"] = True
    with pytest.raises(core.C2ContractError, match="R2 loss"):
        core.validate_cell_receipt(r2, spec)
    budget = _receipt(spec, 0.4, preflight_sha="a" * 64)
    budget["window_budget_per_session"] = 64
    with pytest.raises(core.C2ContractError, match="window_budget"):
        core.validate_cell_receipt(budget, spec)
    wrong_flag = _receipt(spec, 0.4, preflight_sha="a" * 64)
    wrong_flag["balance_session_batches"] = False
    with pytest.raises(core.C2ContractError, match="sampler flag"):
        core.validate_cell_receipt(wrong_flag, spec)


def test_z4_wrapper_is_c2_local_and_masks_only_the_side_tensor() -> None:
    from src.data.c2_m2_matched_z4_datamodule import (
        C2M2MatchedZ4DataModule,
        _MaskStandardizedT4Dataset,
    )
    from src.data.falcon_datamodule import FalconDataModule
    import numpy as np

    assert issubclass(C2M2MatchedZ4DataModule, FalconDataModule)
    assert inspect.signature(C2M2MatchedZ4DataModule.__init__).parameters[
        "c2_z4_mask_after_standardization"
    ].default is True
    source = inspect.getsource(C2M2MatchedZ4DataModule)
    assert "b1_z4" not in source

    class _Row:
        window_indices = [("s", 0)]

        def __getitem__(self, index: int):
            side = np.ones((3, 4), dtype=np.float32)
            return np.zeros(2), np.zeros(2), np.zeros(2), "s", side

    masked = _MaskStandardizedT4Dataset(_Row())
    neural, behavior, calib, session, side = masked[0]
    del neural, behavior, calib, session
    assert side.shape == (3, 4)
    assert not side.any()


def test_preflight_composes_four_templates_and_keeps_gpu_unauthorized() -> None:
    payload = build_preflight(REPO)
    assert payload["status"] == "CPU_PREFLIGHT_READY_GPU_NOT_AUTHORIZED"
    assert payload["gpu_authorized"] is False
    assert payload["operations"]["training_started"] is False
    assert payload["operations"]["torch_imported"] is False
    assert payload["factorial"]["cells"] == 36
    assert payload["scope"]["r2_native_loss"] is False
    assert payload["scope"]["window_budget_per_session_used"] is False
    assert payload["scope"]["sua_a2_sampler_claim"] is False
    by_arm = {(row["sampling"], row["carrier"]): row for row in payload["templates"]}
    assert by_arm[("legacy", "t4")]["balance_session_batches"] is False
    assert by_arm[("equal_session", "t4")]["balance_session_batches"] is True
    assert by_arm[("equal_session", "z4")]["data_target"].endswith("C2M2MatchedZ4DataModule")
    assert payload["implementation_bindings"] == core.source_bindings(REPO)


def test_experiment_configs_flip_only_the_existing_sampler_switch() -> None:
    cfg_dir = str(PROJECT / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        legacy = compose(config_name="train", overrides=["experiment=c2_v1_m2_t4_legacy", "train=true", "test=false"])
        equal = compose(
            config_name="train",
            overrides=["experiment=c2_v1_m2_t4_equal_session", "train=true", "test=false"],
        )
        z4 = compose(config_name="train", overrides=["experiment=c2_v1_m2_z4_equal_session", "train=true", "test=false"])
    assert legacy.data.balance_session_batches is False
    assert equal.data.balance_session_batches is True
    assert legacy.model.loss_mode == "task_plus_y_plus_E"
    assert equal.model.loss_mode == "task_plus_y_plus_E"
    assert "window_budget_per_session" not in legacy.data
    assert z4.data._target_.endswith("C2M2MatchedZ4DataModule")
    assert z4.data.balance_session_batches is True


def test_inert_runner_refuses_launch_and_dry_run_binds_contract(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    payload = build_preflight(REPO)
    digest = core.write_immutable_json(preflight_path, payload)
    dry = subprocess.run(
        [PY, str(PROJECT / "scripts/run_c2_m2_sampling.py"), "--official-preflight", str(preflight_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT),
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    printed = json.loads(dry.stdout)
    assert printed["gpu_launched"] is False
    assert printed["official_preflight_sha256"] == digest
    assert core.sha256_file(REPO / "sua_exploration/docs/C2_M2_SAMPLING_OBJECTIVE_CONTRACT_20260813.md") in printed["contract_sha256"]
    assert len(printed["cells"]) == 36
    launched = subprocess.run(
        [
            PY,
            str(PROJECT / "scripts/run_c2_m2_sampling.py"),
            "--official-preflight",
            str(preflight_path),
            "--launch",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT),
        check=False,
    )
    assert launched.returncode != 0
    assert "Refusing" in launched.stderr


def test_fail_closed_aggregator_accepts_passing_matrix_and_rejects_sealed_or_incomplete(
    tmp_path: Path,
) -> None:
    preflight_path = tmp_path / "preflight.json"
    preflight_sha = core.write_immutable_json(preflight_path, build_preflight(REPO))
    scores = core.synthetic_score_matrix(t4_delta=0.05, z4_delta=0.01)
    receipt_dir = tmp_path / "receipts"
    paths = _write_receipts(receipt_dir, scores, preflight_sha=preflight_sha)
    out = tmp_path / "aggregate.json"
    cmd = [
        PY,
        str(PROJECT / "scripts/aggregate_c2_m2_sampling.py"),
        "--official-preflight",
        str(preflight_path),
        "--out",
        str(out),
        "--bootstrap-draws",
        "51",
    ]
    for key, path in paths.items():
        cmd.extend(["--receipt", f"{key}={path}"])
    ok = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT), check=False)
    assert ok.returncode == 0, ok.stderr
    printed = json.loads(ok.stdout)
    assert printed["passed"] is True
    assert printed["classification"] == "sampling_objective_pass"

    incomplete_cmd = [
        PY,
        str(PROJECT / "scripts/aggregate_c2_m2_sampling.py"),
        "--official-preflight",
        str(preflight_path),
        "--out",
        str(tmp_path / "incomplete.json"),
        "--bootstrap-draws",
        "11",
    ]
    for key, path in list(paths.items())[:-1]:
        incomplete_cmd.extend(["--receipt", f"{key}={path}"])
    missing = subprocess.run(incomplete_cmd, capture_output=True, text=True, cwd=str(PROJECT), check=False)
    assert missing.returncode != 0
    assert "incomplete" in (missing.stderr + missing.stdout)

    sealed_spec = next(iter(scores))
    sealed_path = tmp_path / "sealed.json"
    sealed_payload = _receipt(
        sealed_spec,
        scores[sealed_spec],
        preflight_sha=preflight_sha,
        session_name="sub-C_ses-CO-20151201",
    )
    core.write_immutable_json(sealed_path, sealed_payload)
    sealed_cmd = [
        PY,
        str(PROJECT / "scripts/aggregate_c2_m2_sampling.py"),
        "--official-preflight",
        str(preflight_path),
        "--out",
        str(tmp_path / "sealed_agg.json"),
        "--receipt",
        f"{sealed_spec.key}={sealed_path}",
    ]
    sealed = subprocess.run(sealed_cmd, capture_output=True, text=True, cwd=str(PROJECT), check=False)
    assert sealed.returncode != 0
    assert "sealed" in (sealed.stderr + sealed.stdout).lower()


def test_generic_synthetic_matrix_fails_the_frozen_gate_through_the_aggregator(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    preflight_sha = core.write_immutable_json(preflight_path, build_preflight(REPO))
    scores = core.synthetic_score_matrix(t4_delta=0.05, z4_delta=0.05)
    paths = _write_receipts(tmp_path / "receipts", scores, preflight_sha=preflight_sha)
    out = tmp_path / "generic.json"
    cmd = [
        PY,
        str(PROJECT / "scripts/aggregate_c2_m2_sampling.py"),
        "--official-preflight",
        str(preflight_path),
        "--out",
        str(out),
        "--bootstrap-draws",
        "21",
    ]
    for key, path in paths.items():
        cmd.extend(["--receipt", f"{key}={path}"])
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT), check=False)
    assert proc.returncode == 0, proc.stderr
    printed = json.loads(proc.stdout)
    assert printed["passed"] is False
    assert printed["classification"] == "generic_z4_lift_or_no_carrier_specificity_stop"
    body, _digest = core.load_verified_immutable_json(out)
    assert body["passed"] is False
    assert body["gates"]["t4_mean_delta_at_least_floor"] is True
    assert body["gates"]["interaction_mean_at_least_floor"] is False
