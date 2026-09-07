"""CPU tests for C1 teacher-domain ablation scaffolding.

Proves the frozen gate can pass and fail, that a CO-native teacher of the
existing student architecture can be constructed and loaded without touching
target or sealed data, and that the inert runner never launches.
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
SCE = ROOT / "streaming_calibration_exp"
PY = "/home/xinyuan/miniconda3/envs/spint/bin/python"
for path in (ROOT, SUA, SCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mc_maze import c1_teacher_domain_ablation as core

RUNNER = SUA / "scripts" / "run_c1_teacher_domain_ablation_one_cell.sh"
PREFLIGHT = SUA / "scripts" / "c1_teacher_domain_ablation_preflight.py"
AGGREGATOR = SUA / "scripts" / "aggregate_c1_teacher_domain_ablation.py"
RECEIPT_CLI = SUA / "scripts" / "write_c1_teacher_compatibility_receipt.py"
CONTRACT = core.CONTRACT_PATH

TINY_ARCHITECTURE = {
    "model_dim": 32,
    "num_covariates": 2,
    "window_size": 50,
    "num_heads": 2,
    "num_layers": 1,
    "num_id_layers": 2,
    "use_learnable_id": True,
    "learnable_id_type": "mlp",
    "learnable_rep": True,
    "dropout_rate": 0.0,
    "dynamic_dropout": True,
    "dynamic_dropout_low": 0.0,
    "dynamic_dropout_high": 1.0,
    "tf_drop_rate": 0.1,
    "readin_layer_type": "mlp",
}


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    merged = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHON_BIN": PY}
    if env:
        merged.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=merged, cwd=str(cwd or ROOT))


def test_frozen_gate_can_pass_and_fail_with_different_verdicts() -> None:
    passed = core.evaluate_c1_gates(core.synthetic_pass_matrix())
    generic = core.evaluate_c1_gates(core.synthetic_fail_generic_lift_matrix())
    crash = core.evaluate_c1_gates(core.synthetic_fail_z4_crash_matrix())
    flat = core.evaluate_c1_gates(core.synthetic_fail_no_t4_lift_matrix())
    assert passed["passes_all_gates"] is True
    assert passed["verdict"]["gate_name"] == "teacher_domain_external_t4_effective"
    assert passed["verdict"]["passes"] is True
    assert generic["passes_all_gates"] is False
    assert generic["gates"]["z4_does_not_reproduce_the_lift"] is False
    assert crash["passes_all_gates"] is False
    assert crash["gates"]["mean_external_t4_lift_at_least_0p03"] is False
    assert crash["external_interaction"]["mean"] > core.EFFECTIVE_THRESHOLD
    assert flat["passes_all_gates"] is False
    assert passed["verdict"]["passes"] != generic["verdict"]["passes"]
    assert passed["hypothesis_status"] == "plausible_contributor_not_isolated_cause"
    assert generic["hypothesis_status"] == "plausible_contributor_not_isolated_cause"


def test_config_holds_w_add_and_rejects_h_add() -> None:
    payload = core.validate_config()
    assert payload["add_site"] == "W-add"
    assert payload["hidden_space_adapter"] is False
    assert payload["sampling"] == "legacy"
    assert payload["frozen_protocol"]["copies_teacher_decoder_under_task_only"] is True
    assert payload["frozen_protocol"]["encoder_warmstart_path"] is None


def test_source_only_helpers_refuse_sealed_and_target_paths() -> None:
    with pytest.raises(Exception, match="sealed"):
        core.refuse_sealed_sessions(["sub-C_ses-CO-20151113"], label="probe")
    with pytest.raises(core.C1ContractError, match="target or sealed"):
        core.refuse_target_or_sealed_paths(
            [SUA / "data" / "dandi_000688" / "sub-M" / "fake.nwb"],
            label="probe",
        )
    with pytest.raises(core.C1ContractError, match="target or sealed"):
        core.refuse_target_or_sealed_paths(
            [SUA / "data" / "dandi_000688" / "sub-C" / "sub-C_ses-CO-20151201.nwb"],
            label="probe",
        )


def test_manifest_source_train_excludes_sealed_sessions() -> None:
    from mc_maze.gpu_contract_common import SEALED_FORMAL_TEST_SESSIONS

    manifest = core.load_strict_manifest()
    assert len(manifest["train"]) == 27
    assert set(manifest["test"]) == set(SEALED_FORMAL_TEST_SESSIONS)
    assert not set(manifest["train"]) & set(SEALED_FORMAL_TEST_SESSIONS)
    assert not set(manifest["val"]) & set(SEALED_FORMAL_TEST_SESSIONS)


def test_synthetic_co_native_teacher_loads_into_student_contract_under_task_only(tmp_path: Path) -> None:
    from mc_maze import c1_teacher_compatibility as compat

    ckpt = tmp_path / "co_native_tiny.ckpt"
    written = compat.write_matching_teacher_checkpoint(
        ckpt,
        architecture=TINY_ARCHITECTURE,
        identity_in_features=core.TRIAL_LENGTH_BINS,
        lightning_task=core.STUDENT_LIGHTNING_TASK,
        seed=1,
    )
    assert written["trained"] is False
    loaded = compat.load_into_student_contract(
        ckpt,
        side_dim=4,
        loss_mode="task_only",
        require_production_architecture=False,
    )
    assert loaded["loaded"] is True
    assert loaded["loss_mode"] == "task_only"
    assert loaded["encoder_warmstart_path"] is None
    assert loaded["decoder_state_dict_strict_copy_of_teacher"] is True
    assert loaded["selected_t4_encoder_warmstart"] is False
    assert loaded["b3s_encoder_contains_teacher_fc_id"] is False
    assert loaded["lightning_task"] == "mc_maze"


def test_window_size_mismatch_fails_student_contract(tmp_path: Path) -> None:
    from mc_maze import c1_teacher_compatibility as compat

    bad = dict(TINY_ARCHITECTURE)
    bad["window_size"] = 40
    ckpt = tmp_path / "bad_window.ckpt"
    compat.write_matching_teacher_checkpoint(
        ckpt,
        architecture=bad,
        identity_in_features=core.TRIAL_LENGTH_BINS,
        seed=2,
    )
    with pytest.raises(core.C1ContractError, match="window_size"):
        compat.load_into_student_contract(ckpt, require_production_architecture=False)


def test_compatibility_receipt_is_source_only_and_records_mc_maze_diff(tmp_path: Path) -> None:
    from mc_maze import c1_teacher_compatibility as compat

    out = tmp_path / "compat.json"
    work = tmp_path / "work"
    body, sidecar, digest, receipt = compat.write_compatibility_receipt(
        out,
        work_dir=work,
        teacher_path=core.MC_MAZE_TEACHER_PATH,
        use_production_architecture=False,
        tiny_architecture=TINY_ARCHITECTURE,
    )
    assert body.is_file() and sidecar.is_file()
    assert digest == core.sha256_file(body)
    assert stat.S_IMODE(body.stat().st_mode) == 0o444
    assert receipt["source_only"] is True
    assert receipt["target_or_sealed_data_opened"] is False
    assert receipt["operations"]["nwb_opened"] is False
    assert receipt["operations"]["gpu_used"] is False
    assert receipt["operations"]["training_run"] is False
    assert receipt["trained_co_native_teacher_exists"] is False
    assert receipt["interface_constructible"] is True
    assert receipt["gpu_authorized"] is False
    assert receipt["hypothesis_status"] == "plausible_contributor_not_isolated_cause"
    assert receipt["student_contract"]["copies_teacher_state_dict_even_under_task_only"] is True
    assert receipt["student_contract"]["selected_t4_encoder_warmstart"] is False
    diffs = receipt["decoder_initialization_diff"]
    assert diffs["isolated_cause_claimed"] is False
    assert diffs["weight_origin"]["mc_maze"]["behavior"] == "hand_vel"
    assert diffs["weight_origin"]["co_native_spec"]["behavior"] == "cursor_vel"
    assert diffs["student_load_path"]["does_so_under_task_only"] is True
    assert "sub-M" not in json.dumps(receipt["source_train_session_names_only"])
    with pytest.raises(FileExistsError):
        compat.write_compatibility_receipt(
            out,
            work_dir=work / "other",
            teacher_path=core.MC_MAZE_TEACHER_PATH,
            use_production_architecture=False,
            tiny_architecture=TINY_ARCHITECTURE,
        )


def test_compatibility_cli_tiny_architecture(tmp_path: Path) -> None:
    out = tmp_path / "cli.json"
    work = tmp_path / "work"
    proc = _run(
        [
            PY,
            str(RECEIPT_CLI),
            "--out",
            str(out),
            "--work-dir",
            str(work),
            "--tiny-architecture",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["interface_constructible"] is True
    assert payload["operations"]["sealed_formal_test_sessions_opened"] is False


def test_preflight_fail_closed_without_compatibility_receipt(tmp_path: Path) -> None:
    result_root = tmp_path / "c1"
    receipt = result_root / "official_cpu_preflight.json"
    proc = _run(
        [
            PY,
            str(PREFLIGHT),
            "--result-root",
            str(result_root),
            "--receipt",
            str(receipt),
            "--compatibility-receipt",
            str(result_root / "missing.json"),
        ]
    )
    assert proc.returncode == 2
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "STOP_C1_PREFLIGHT_BLOCKERS"
    assert payload["gpu_authorized"] is False
    assert any(row["code"] == "MISSING_SOURCE_ONLY_COMPATIBILITY_RECEIPT" for row in payload["implementation_blockers"])


def test_preflight_fail_closed_on_nonempty_cell_root(tmp_path: Path) -> None:
    from mc_maze import c1_teacher_compatibility as compat

    result_root = tmp_path / "c1"
    result_root.mkdir()
    compat_path = result_root / "source_only_teacher_compatibility_receipt.json"
    compat.write_compatibility_receipt(
        compat_path,
        work_dir=tmp_path / "work",
        teacher_path=core.MC_MAZE_TEACHER_PATH,
        use_production_architecture=False,
        tiny_architecture=TINY_ARCHITECTURE,
    )
    (result_root / "mc_maze_t4_s42_within_subject.json").write_text("{}", encoding="utf-8")
    receipt = result_root / "official_cpu_preflight.json"
    proc = _run(
        [
            PY,
            str(PREFLIGHT),
            "--result-root",
            str(result_root),
            "--receipt",
            str(receipt),
            "--compatibility-receipt",
            str(compat_path),
        ]
    )
    assert proc.returncode == 2
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert any(row["code"] == "NONEMPTY_OUTPUT_ROOT" for row in payload["implementation_blockers"])


def test_preflight_passes_cpu_status_when_compatibility_exists(tmp_path: Path) -> None:
    from mc_maze import c1_teacher_compatibility as compat

    result_root = tmp_path / "c1"
    compat_path = result_root / "source_only_teacher_compatibility_receipt.json"
    compat.write_compatibility_receipt(
        compat_path,
        work_dir=tmp_path / "work",
        teacher_path=core.MC_MAZE_TEACHER_PATH,
        use_production_architecture=False,
        tiny_architecture=TINY_ARCHITECTURE,
    )
    receipt = result_root / "official_cpu_preflight.json"
    proc = _run(
        [
            PY,
            str(PREFLIGHT),
            "--result-root",
            str(result_root),
            "--receipt",
            str(receipt),
            "--compatibility-receipt",
            str(compat_path),
        ]
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "CPU_PREFLIGHT_PASSED_AWAITING_TRAINED_CO_NATIVE_TEACHER_AND_ROOT_GO"
    assert payload["gpu_authorized"] is False
    assert payload["trained_co_native_teacher_exists"] is False
    assert payload["nwb_opened"] is False


def _write_matrix(result_dir: Path, matrix: dict, *, add_site: str = "W-add", extra: dict | None = None) -> None:
    contract_sha = core.sha256_file(CONTRACT)
    result_dir.mkdir(parents=True, exist_ok=True)
    for teacher in core.TEACHER_DOMAINS:
        for carrier in core.CARRIERS:
            for seed in core.SEEDS:
                for domain in core.DOMAINS:
                    sessions = core.expected_domain_sessions(domain)
                    value = float(matrix[teacher][carrier][seed][domain])
                    per_session = {session: value for session in sessions}
                    mean = sum(float(per_session[session]) for session in sessions) / len(sessions)
                    payload = {
                        "schema_version": 1,
                        "screen_id": core.SCREEN_ID,
                        "contract_sha256": contract_sha,
                        "teacher_domain": teacher,
                        "carrier": carrier,
                        "seed": seed,
                        "domain": domain,
                        "add_site": add_site,
                        "sampling": "legacy",
                        "hidden_space_adapter": False,
                        "loss_mode": "task_only",
                        "variant": "B3S",
                        "encoder_warmstart_path": None,
                        "sealed_formal_test_sessions_opened": False,
                        "formal_test_sessions_opened": False,
                        "per_session_r2": per_session,
                        "mean_r2": mean,
                    }
                    if extra:
                        payload.update(extra)
                    path = result_dir / core.domain_receipt_name(teacher, carrier, seed, domain)
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_aggregator_pass_and_fail_and_missing_cell(tmp_path: Path) -> None:
    aggregator = _module("c1_agg_test", AGGREGATOR)
    good_dir = tmp_path / "good"
    _write_matrix(good_dir, core.synthetic_pass_matrix())
    good = aggregator.aggregate(good_dir)
    assert good["gate"]["passes_all_gates"] is True

    bad_dir = tmp_path / "bad"
    _write_matrix(bad_dir, core.synthetic_fail_generic_lift_matrix())
    bad = aggregator.aggregate(bad_dir)
    assert bad["gate"]["passes_all_gates"] is False
    assert good["gate"]["verdict"]["passes"] != bad["gate"]["verdict"]["passes"]

    missing = tmp_path / "missing"
    _write_matrix(missing, core.synthetic_pass_matrix())
    (missing / core.domain_receipt_name("co_native", "t4", 42, "external_subject_M")).unlink()
    with pytest.raises(core.C1ContractError, match="missing domain receipts"):
        aggregator.aggregate(missing)


def test_aggregator_fail_closed_on_sealed_session_and_h_add(tmp_path: Path) -> None:
    aggregator = _module("c1_agg_test2", AGGREGATOR)
    sealed_dir = tmp_path / "sealed"
    _write_matrix(sealed_dir, core.synthetic_pass_matrix())
    path = sealed_dir / core.domain_receipt_name("mc_maze", "t4", 42, "within_subject")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["per_session_r2"]["sub-C_ses-CO-20151113"] = 0.1
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises((core.C1ContractError, ValueError), match="sealed|roster"):
        aggregator.aggregate(sealed_dir)

    hadd = tmp_path / "hadd"
    _write_matrix(hadd, core.synthetic_pass_matrix(), extra={"hidden_space_adapter": True})
    with pytest.raises(core.C1ContractError, match="H-add"):
        aggregator.aggregate(hadd)


def test_aggregator_cli_writes_immutable_fail_and_pass(tmp_path: Path) -> None:
    good_dir = tmp_path / "good"
    _write_matrix(good_dir, core.synthetic_pass_matrix())
    out = tmp_path / "agg.json"
    proc = _run([PY, str(AGGREGATOR), "--result-dir", str(good_dir), "--out", str(out)])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gate"]["passes_all_gates"] is True
    assert stat.S_IMODE(out.stat().st_mode) == 0o444
    proc2 = _run([PY, str(AGGREGATOR), "--result-dir", str(good_dir), "--out", str(out)])
    assert proc2.returncode == 2
    assert "refusing immutable overwrite" in proc2.stderr


def test_runner_dry_run_is_inert_and_launch_refuses() -> None:
    env = {
        "TEACHER_DOMAIN": "co_native",
        "CARRIER": "t4",
        "SEED": "42",
        "GPU": "0",
        "PYTHON_BIN": PY,
    }
    dry = _run(["bash", str(RUNNER), "--dry-run"], env=env)
    assert dry.returncode == 0, dry.stderr
    assert "DRY_RUN_INERT=true" in dry.stdout
    assert "TORCH_IMPORTED=false" in dry.stdout
    assert "GPU_LAUNCHED=false" in dry.stdout
    assert "ADD_SITE=W-add" in dry.stdout
    assert core.sha256_file(CONTRACT) in dry.stdout
    assert "train_variant_dandi688.py" in dry.stdout
    launch = _run(["bash", str(RUNNER), "--launch"], env=env)
    assert launch.returncode == 3
    assert "Refusing GPU launch" in (launch.stderr + launch.stdout)


def test_real_mc_maze_teacher_matches_frozen_student_architecture() -> None:
    from mc_maze import c1_teacher_compatibility as compat

    snapshot = compat.snapshot_teacher_checkpoint(core.MC_MAZE_TEACHER_PATH)
    assert snapshot["sha256"] == core.EXPECTED_TEACHER_SHA256
    assert snapshot["lightning_task"] == "mc_maze"
    assert snapshot["architecture"] == dict(core.PRODUCTION_SPINT_HYPERPARAMETERS)
    assert snapshot["identity_mlp_in_features"] == 100
    assert snapshot["fc_in_in_features"] == 50
    assert snapshot["n_independent_readin"] is True
    assert snapshot["device"] == "cpu"
