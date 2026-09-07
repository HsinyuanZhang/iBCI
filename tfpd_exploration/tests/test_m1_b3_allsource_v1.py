"""CPU tests for m1_b3_allsource_v1. No 12-epoch training, no GPU jobs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from tfpd_exploration.src.m1_b3_allsource_v1 import gpu as cell_gpu
from tfpd_exploration.src.m1_b3_allsource_v1 import plan
from tfpd_exploration.src.m1_b3_allsource_v1 import receipts as cell_receipts


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tfpd_exploration/scripts/run_m1_b3_allsource_v1.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"


def _cli_env() -> dict[str, str]:
    return {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(ROOT)}


def test_plan_literals() -> None:
    assert plan.TRAIN_ARMS == (
        "b3", "b3s_rsyn3", "b3s_t4", "b3s_t4_encoder",
        "b3s_rsyn3_freeze", "b3s_rsyn3_acyc",
    )
    assert plan.WAVE1_ARMS == ("b3", "b3s_rsyn3")
    assert plan.PACK_LIMIT == 2
    assert plan.COSINE_50EP_FORBIDDEN is True
    assert plan.SCHEDULER is None
    assert plan.SEED == 42 and plan.EPOCHS == 12
    assert plan.ADAM_LR == 1.0e-4 and plan.TRAIN_BATCH_SIZE == 32
    assert plan.SOURCE_SESSIONS == ("20120924", "20120926", "20120927", "20120928")
    assert plan.TEACHER_SHA256 == (
        "c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2"
    )
    assert plan.OFFICIAL_ORIGINAL_HELDOUT_R2 == 0.648591
    assert plan.GPU0_ALLOWED is False
    dry = plan.dry_plan()
    assert dry["four_session_all_source_in_train_roster"] is True
    assert dry["local_20120924_is_not_official_heldout"] is True
    assert dry["pack_limit"] == 2
    assert dry["imports_torch"] is False
    assert dry["formal_benchmark_verdict"] is False
    assert dry["wave2_arms"] == ["b3s_rsyn3_freeze", "b3s_rsyn3_acyc"]
    assert plan.ARM_SPECS["b3"]["freeze_decoder"] is False
    assert plan.ARM_SPECS["b3"]["side_feature_group"] == "none"
    assert plan.ARM_SPECS["b3s_t4"]["side_feature_group"] == "t4"
    assert plan.ARM_SPECS["b3s_rsyn3"]["side_feature_group"] == "rsyn3"
    assert plan.ARM_SPECS["b3s_t4_encoder"]["freeze_decoder"] is True
    assert plan.WAVE2_ARMS == ("b3s_rsyn3_freeze", "b3s_rsyn3_acyc")
    assert plan.ARM_SPECS["b3s_rsyn3_freeze"]["freeze_decoder"] is True
    assert plan.ARM_SPECS["b3s_rsyn3_freeze"]["side_feature_group"] == "rsyn3"
    assert plan.ARM_SPECS["b3s_rsyn3_freeze"]["activity_prefix_cycle"] is None
    assert plan.ARM_SPECS["b3s_rsyn3_acyc"]["freeze_decoder"] is False
    assert plan.ARM_SPECS["b3s_rsyn3_acyc"]["side_feature_group"] == "rsyn3"
    assert plan.ARM_SPECS["b3s_rsyn3_acyc"]["activity_prefix_cycle"] == (10, 5, 2)
    assert plan.ACTIVITY_CYCLE == (10, 5, 2)


def test_owned_paths_exist_and_teacher_bytes_match() -> None:
    for relative in plan.OWNED_PATHS:
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink(), relative
    teacher = ROOT / plan.TEACHER_CHECKPOINT_RELATIVE
    assert teacher.is_file()
    assert plan.sha256_bytes(teacher.read_bytes()) == plan.TEACHER_SHA256
    closure = plan.implementation_closure(ROOT)
    assert len(closure["paths"]) == len(plan.OWNED_PATHS)


def test_hydra_experiments_pin_joint_recipe() -> None:
    configs = ROOT / "streaming_calibration_exp/configs"
    b3 = (configs / "experiment/m1_b3_allsource_b3.yaml").read_text(encoding="utf-8")
    t4 = (configs / "experiment/m1_b3_allsource_b3s_t4.yaml").read_text(encoding="utf-8")
    rsyn3 = (configs / "experiment/m1_b3_allsource_b3s_rsyn3.yaml").read_text(encoding="utf-8")
    enc = (configs / "experiment/m1_b3_allsource_b3s_t4_encoder.yaml").read_text(encoding="utf-8")
    for text in (b3, t4, rsyn3, enc):
        assert "max_epochs: 12" in text
        assert "num_sanity_val_steps: 0" in text
        assert "loss_mode: task_only" in text
        assert "test: false" in text
        assert "require_baseline_validation: false" in text
        assert "m1_teacher_ckpt_path" in text
    assert "freeze_decoder: false" in b3
    assert "freeze_decoder: false" in t4
    assert "freeze_decoder: false" in rsyn3
    assert "freeze_decoder: true" in enc
    assert "falcon_m1_all_source_b3s_rsyn3" in rsyn3
    assert "streaming_b3s_rsyn3_m1" in rsyn3
    assert "falcon_m1_all_source_b3s_t4" not in rsyn3
    assert "side_feature_group: none" in (
        configs / "data/falcon_m1_all_source_b3.yaml"
    ).read_text(encoding="utf-8")
    assert "side_feature_group: t4" in (
        configs / "data/falcon_m1_all_source_b3s_t4.yaml"
    ).read_text(encoding="utf-8")
    assert "side_feature_group: rsyn3" in (
        configs / "data/falcon_m1_all_source_b3s_rsyn3.yaml"
    ).read_text(encoding="utf-8")


def test_new_paths_avoid_forbidden_tokens() -> None:
    forbidden = ("held-out", "minival", "evalai", "formal")
    for relative in plan.OWNED_PATHS:
        lowered = relative.lower()
        for token in forbidden:
            assert token not in lowered, relative
    assert "evalai" not in plan.RESULT_ROOT_RELATIVE


def test_gpu_pack_two_own_third_refused_foreign_refused() -> None:
    own = "tfpd_exploration/scripts/run_m1_b3_allsource_v1.py --arm b3"
    rows = (
        f"0, {plan.GPU0_UUID}, 453, 0\n"
        f"1, {plan.GPU1_UUID}, 987, 54\n"
    )

    def busy(_arguments):
        query = " ".join(_arguments)
        if "compute-apps" in query:
            return f"{plan.GPU1_UUID}, 9, python, 1024\n"
        return (
            f"0, {plan.GPU0_UUID}, 453, 0\n"
            f"1, {plan.GPU1_UUID}, 23, 0\n"
        )

    with pytest.raises(cell_gpu.GpuError, match="compute apps"):
        cell_gpu.assert_target_gpu_launchable(1, cmdline_runner=busy)

    def one_own(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return f"{plan.GPU1_UUID}, 11, python, 958\n"
        return rows

    receipt = cell_gpu.assert_target_gpu_launchable(
        1, pack=True, cmdline_runner=one_own,
        pid_cmdline=lambda pid: own if pid == "11" else "other",
    )
    assert receipt["pack"] is True
    assert receipt["own_occupants"] == 1

    def two_own(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return (
                f"{plan.GPU1_UUID}, 11, python, 958\n"
                f"{plan.GPU1_UUID}, 12, python, 958\n"
            )
        return rows

    with pytest.raises(cell_gpu.GpuError, match="pack limit"):
        cell_gpu.assert_target_gpu_launchable(
            1, pack=True, cmdline_runner=two_own, pid_cmdline=lambda pid: own,
        )

    def foreign(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return f"{plan.GPU1_UUID}, 99, python, 1024\n"
        return rows

    with pytest.raises(cell_gpu.GpuError, match="foreign"):
        cell_gpu.assert_target_gpu_launchable(
            1, pack=True, cmdline_runner=foreign,
            pid_cmdline=lambda pid: "python tfpd_exploration/scripts/run_m1_tier12_pack_v1.py",
        )

    with pytest.raises(cell_gpu.GpuError, match="GPU 1"):
        cell_gpu.assert_target_gpu_launchable(0, cmdline_runner=lambda _: rows)


def test_receipts_refuse_sealed_roots(tmp_path: Path) -> None:
    with pytest.raises(cell_receipts.ReceiptError, match="sealed"):
        cell_receipts.refuse_sealed_roots(ROOT / "tfpd_exploration/results/m1_t0c1_prefix_v1")
    with pytest.raises(cell_receipts.ReceiptError, match="sealed"):
        cell_receipts.refuse_sealed_roots(ROOT / "tfpd_exploration/results/m1_tier12_pack_v1")
    cell_receipts.refuse_sealed_roots(tmp_path / "m1_b3_allsource_v1")
    cell_receipts.refuse_sealed_roots(ROOT / plan.RESULT_ROOT_RELATIVE)


def test_cli_dry_and_execute_gpu_error() -> None:
    dry = subprocess.run(
        [PYTHON, "-S", str(CLI), "--dry-run"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True, check=True,
    )
    payload = json.loads(dry.stdout)
    assert payload["phase"] == "m1_b3_allsource_v1"
    assert payload["wave1_arms"] == ["b3", "b3s_rsyn3"]
    denied = subprocess.run(
        [PYTHON, str(CLI), "--execute-gpu"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True,
    )
    assert denied.returncode != 0
    assert "cannot mint a GPU capability" in (denied.stderr + denied.stdout)
    gpu0 = subprocess.run(
        [PYTHON, str(CLI), "--execute", "--gpu-authorized", "--gpu-index", "0",
         "--mode", "train", "--arm", "b3"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True,
    )
    assert gpu0.returncode != 0
    assert "GPU 1" in (gpu0.stderr + gpu0.stdout)


def test_pack_token_is_this_cli() -> None:
    assert plan.OWN_TOKEN == "run_m1_b3_allsource_v1.py"
    assert "tier12" not in plan.OWN_TOKEN
    assert plan.GPU1_UUID == "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"


def test_hydra_compose_rsyn3_pins_b3s_carrier_not_t4() -> None:
    from hydra import compose, initialize_config_dir

    configs = ROOT / "streaming_calibration_exp/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(configs)):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=m1_b3_allsource_b3s_rsyn3"],
        )
    assert str(cfg.data._target_).endswith("M1AllSourceB3RSyn3DataModule")
    assert str(cfg.data.side_feature_group).lower() == "rsyn3"
    assert str(cfg.model.variant) == "B3S"
    assert int(cfg.model.side_dim) == 4
    assert bool(cfg.model.freeze_decoder) is False
    assert str(cfg.model.loss_mode) == "task_only"
    assert int(cfg.trainer.max_epochs) == 12
    assert bool(cfg.test) is False
    assert "t4" not in str(cfg.data._target_).lower()


def test_rsyn3_package_helpers_keep_frozen_source_carriers() -> None:
    import numpy as np

    from tfpd_exploration.src.m1_b3_allsource_v1 import package as package_module
    from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank

    assert package_module.neural_loader_side_group("rsyn3") == "none"
    assert package_module.neural_loader_side_group("t4") == "t4"
    assert package_module.neural_loader_side_group("none") == "none"
    source = np.arange(256, dtype=np.float32).reshape(64, 4)
    bank = {
        "later_day_in_fit": False,
        "normalized": {name: np.zeros((64, 4), dtype=np.float32) for name in plan.SOURCE_SESSION_NAMES},
    }
    bank["normalized"]["ses-20120924"] = source
    chrono = plan.ARM_SPECS["b3s_rsyn3"]
    carrier, selected = package_module.rsyn3_carrier_for_session(
        "ses-20120924", Path("/unused.nwb"), bank, spec=chrono,
    )
    assert carrier.dtype == np.float32
    assert np.array_equal(carrier, source)
    assert selected == list(range(10))
    bank["later_day_in_fit"] = True
    with pytest.raises(package_module.PackageError, match="later-day"):
        package_module.rsyn3_carrier_for_session(
            "ses-20120924", Path("/unused.nwb"), bank, spec=chrono,
        )

    fake = {
        "schema": "m1_all_source_rsyn3_bank_v1",
        "source_sessions": list(plan.SOURCE_SESSION_NAMES),
        "support_trials": 10,
        "query_values_read": False,
        "nmf_fit_sessions": list(plan.SOURCE_SESSION_NAMES),
        "later_day_in_fit": False,
        "basis_kind": "nnmf",
        "basis_scale": np.ones(8, dtype=np.float64),
        "basis_dictionary": np.eye(3, 8, dtype=np.float64),
        "basis_order": [0, 1, 2],
        "normalizer_mean": np.zeros(4, dtype=np.float64),
        "normalizer_scale": np.ones(4, dtype=np.float64),
        "normalized": {
            name: np.zeros((64, 4), dtype=np.float32) for name in plan.SOURCE_SESSION_NAMES
        },
        "source_files": {
            name: {"path": f"/tmp/{name}.nwb", "sha256": "0" * 64}
            for name in plan.SOURCE_SESSION_NAMES
        },
    }
    payload = rsyn3_bank.manifest_payload(fake)
    json.dumps(payload)
    restored = rsyn3_bank.bank_from_manifest(payload)
    assert restored["later_day_in_fit"] is False
    assert tuple(restored["source_sessions"]) == plan.SOURCE_SESSION_NAMES


def test_package_sys_path_prefers_streaming_src() -> None:
    import sys
    from tfpd_exploration.src.m1_b3_allsource_v1 import package as package_module

    package_module._ensure_streaming_paths(ROOT)
    import src.models.streaming_calibration_module as module

    assert "streaming_calibration_exp" in Path(module.__file__).resolve().parts
    streaming = str(ROOT / "streaming_calibration_exp")
    spint_main = str(ROOT / "SPINT-main")
    assert sys.path.index(streaming) < sys.path.index(spint_main)
    assert plan.package_root_relative("b3").endswith("export/b3")
    assert package_module.PAYLOAD_LEAF == "decoder.pt"


def test_rsyn3_datamodule_rewrites_side_and_returns_five_tuple() -> None:
    import numpy as np
    import sys

    streaming = ROOT / "streaming_calibration_exp"
    if str(streaming) not in sys.path:
        sys.path.insert(0, str(streaming))
    from src.data.falcon_m1_all_source_b3_rsyn3_datamodule import (  # noqa: E402
        M1AllSourceB3RSyn3DataModule,
        RSyn3SideDataset,
    )

    module = M1AllSourceB3RSyn3DataModule(
        task="m1",
        data_dir=".",
        calibration_n_trials=10,
        random_calibration=False,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=0,
        heldin_query_end_trial=None,
        side_feature_group="rsyn3",
        validation_protocol="all_source",
    )
    assert str(module.hparams.side_feature_group).lower() == "none"
    with pytest.raises(ValueError, match="none/rsyn3"):
        M1AllSourceB3RSyn3DataModule(
            task="m1",
            data_dir=".",
            calibration_n_trials=10,
            random_calibration=False,
            include_heldout_in_fit=False,
            include_heldout_in_test=False,
            query_start_trial=0,
            heldin_query_start_trial=0,
            heldin_query_end_trial=None,
            side_feature_group="t4",
            validation_protocol="all_source",
        )

    class _Base:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            neural = np.zeros((100, 64), dtype=np.float32)
            target = np.zeros((100, 2), dtype=np.float32)
            calib = np.zeros((10, 1024, 64), dtype=np.float32)
            return neural, target, calib, "ses-20120924"

    carrier = np.ones((64, 4), dtype=np.float32)
    wrapped = RSyn3SideDataset(_Base(), {"ses-20120924": carrier})
    neural, target, calib, session, side = wrapped[0]
    assert session == "ses-20120924"
    assert side.shape == (64, 4)
    assert np.array_equal(side, carrier)


def test_m4_query_protocol_is_local_diagnostic() -> None:
    import inspect

    from tfpd_exploration.src.m1_b3_allsource_v1 import m4_query
    from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank

    assert plan.M4_QUERY_SUPPORT == 4
    assert plan.M4_QUERY_START == 4
    assert plan.CALIBRATION_N_TRIALS == 10
    assert m4_query.SUPPORT_TRIALS == 4
    assert m4_query.QUERY_START == 4
    assert plan.M4_QUERY_INCLUDE_SOURCE_IN_TRAIN_DEFAULT is False
    assert plan.M4_QUERY_LATER_DAY_QUERY_TRIALS == 6
    assert m4_query.session_role("ses-20121004") == "later_day_public_calib"
    assert m4_query.session_role("ses-20120924") == "source_in_train"
    assert "held-out" not in plan.M4_QUERY_ROOT_RELATIVE
    assert "minival" not in plan.M4_QUERY_ROOT_RELATIVE
    assert "evalai" not in plan.M4_QUERY_ROOT_RELATIVE
    assert inspect.signature(rsyn3_bank.encode_public_session).parameters["support_trials"].default == 10
    assert "support_trials" in inspect.signature(rsyn3_bank.load_public_calib_support).parameters
    assert 10 - plan.M4_QUERY_SUPPORT == 6
    identical = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
    assert abs(m4_query.last_bin_r2(identical, identical) - 1.0) < 1e-6
    dry = plan.dry_plan()
    assert dry["m4_query"]["support_trials"] == 4
    assert dry["m4_query"]["formal_benchmark_verdict"] is False
    assert dry["prospective_roots"]["m4_query_b3"].endswith("m4_query_v2/b3")
    assert plan.M4_QUERY_ARMS == plan.WAVE1_ARMS + plan.WAVE2_ARMS
    assert dry["prospective_roots"]["m4_query_b3s_rsyn3_freeze"].endswith(
        "m4_query_v2/b3s_rsyn3_freeze"
    )
    assert dry["prospective_roots"]["m4_query_b3s_rsyn3_acyc"].endswith(
        "m4_query_v2/b3s_rsyn3_acyc"
    )
    with pytest.raises(plan.PlanError, match="M4 query is not defined"):
        plan.m4_query_root_relative("b3s_t4")


def test_remote_push_mirrors_successful_m2_cached_identity() -> None:
    from tfpd_exploration.src.m1_b3_allsource_v1 import runtime
    from tfpd_exploration.src.m1_b3_allsource_v1 import submit_remote

    assert plan.REMOTE_CHALLENGE_ID == 2319
    assert plan.REMOTE_PHASE_ID == 4599
    assert plan.REMOTE_TEAM_ID == 41975
    assert plan.REMOTE_BATCH_SIZE == 4
    assert plan.REMOTE_BASE_IMAGE.startswith("spint-original-m1:")
    assert "evalai" not in plan.REMOTE_PUSH_ROOT_RELATIVE
    assert "held-out" not in plan.REMOTE_PUSH_ROOT_RELATIVE
    assert runtime.PAYLOAD_SCHEMA == plan.PAYLOAD_SCHEMA
    assert runtime.EXPECTED_SESSION_COUNT == 7
    assert runtime.EXPECTED_ARMS == plan.REMOTE_ARMS
    assert runtime.WINDOW_SIZE == 100
    report = submit_remote.plan_report("b3")
    assert report["mode"] == "plan_only"
    assert report["challenge_id"] == 2319
    assert report["phase_id"] == 4599
    assert report["private"] is True
    assert report["required_runtime"]["task"] == "m1"
    assert report["required_runtime"]["batch_size"] == 4
    assert report["required_runtime"]["cached_identity_only"] is True
    assert report["required_runtime"]["backpropagation"] is False
    dockerfile = (ROOT / "tfpd_exploration/src/m1_b3_allsource_v1/Dockerfile").read_text(encoding="utf-8")
    assert "TASK=m1" in dockerfile
    assert "BATCH_SIZE=4" in dockerfile
    assert "/data/decoder.pkl" in dockerfile
    assert "spint-original-m1:" in dockerfile
    fake = {"schema_version": "nope", "task": "m1", "arm": "b3", "decoder": object(),
            "identity_by_dataset_tag": {}, "window_size": 100, "behavior_scaling_factor": 1.0}
    with pytest.raises(ValueError, match="Unsupported"):
        runtime._require_payload(fake, expected_arm="b3", task="m1")


CARRIER_K_CLI = ROOT / "tfpd_exploration/scripts/run_m1_b3_allsource_carrier_k_probe.py"


def test_carrier_k_probe_grid_and_local_surface() -> None:
    from tfpd_exploration.src.m1_b3_allsource_v1 import carrier_k

    assert plan.CARRIER_K_PROBE_ROOT_RELATIVE == f"{plan.RESULT_ROOT_RELATIVE}/carrier_k_probe_v1"
    assert "evalai" not in plan.CARRIER_K_PROBE_ROOT_RELATIVE
    assert "held-out" not in plan.CARRIER_K_PROBE_ROOT_RELATIVE
    assert "minival" not in plan.CARRIER_K_PROBE_ROOT_RELATIVE
    assert "formal" not in plan.CARRIER_K_PROBE_ROOT_RELATIVE
    assert "test" not in plan.CARRIER_K_PROBE_ROOT_RELATIVE
    assert plan.CARRIER_K_ARM == "b3s_rsyn3"
    assert plan.CARRIER_K_METHODS == ("chronological", "dopt_tgt_loc", "dopt_emg_syn3")
    assert plan.CARRIER_K_BUDGETS == (3, 4, 5, 6)
    assert plan.CARRIER_K_POOL_TRIALS == 10
    assert plan.CARRIER_K_QUERY_START == 10
    grid = carrier_k.variant_grid()
    names = [row["name"] for row in grid]
    assert len(grid) == 13
    assert names.count("chronological_k10") == 1
    assert "dopt_tgt_loc_k10" not in names
    assert "dopt_emg_syn3_k10" not in names
    for method in plan.CARRIER_K_METHODS:
        for budget in plan.CARRIER_K_BUDGETS:
            assert f"{method}_k{budget}" in names
    dry = plan.dry_plan()
    assert dry["carrier_k_probe"]["formal_benchmark_verdict"] is False
    assert dry["carrier_k_probe"]["arm"] == "b3s_rsyn3"
    assert dry["carrier_k_probe"]["query_start_trial"] == 10
    assert dry["carrier_k_probe"]["encoder_uses_all_pool_neural"] is True
    assert dry["prospective_roots"]["carrier_k_probe"].endswith("carrier_k_probe_v1/b3s_rsyn3")
    assert "tfpd_exploration/src/m1_b3_allsource_v1/carrier_k.py" in plan.OWNED_PATHS
    assert "tfpd_exploration/src/m1_b3_allsource_v1/carrier_k_probe.py" in plan.OWNED_PATHS
    assert "tfpd_exploration/scripts/run_m1_b3_allsource_carrier_k_probe.py" in plan.OWNED_PATHS


def test_carrier_k_later_day_is_m4_remaining_query() -> None:
    from tfpd_exploration.src.m1_b3_allsource_v1 import carrier_k

    assert plan.CARRIER_K_LATER_DAY_ARMS == ("b3s_rsyn3", "b3s_rsyn3_freeze")
    assert plan.CARRIER_K_LATER_DAY_SUPPORT == 4
    assert plan.CARRIER_K_LATER_DAY_QUERY_START == 4
    assert plan.CARRIER_K_LATER_DAY_BUDGETS == (2, 3)
    assert plan.CARRIER_K_LATER_DAY_BASELINE_K == 4
    assert "held-out" not in plan.CARRIER_K_LATER_DAY_ROOT_RELATIVE
    assert "minival" not in plan.CARRIER_K_LATER_DAY_ROOT_RELATIVE
    assert "evalai" not in plan.CARRIER_K_LATER_DAY_ROOT_RELATIVE
    grid = carrier_k.later_day_variant_grid()
    names = [row["name"] for row in grid]
    assert len(grid) == 7
    assert names.count("chronological_k4") == 1
    assert next(row for row in grid if row["name"] == "chronological_k4")["is_baseline"] is True
    assert "dopt_tgt_loc_k4" not in names
    assert "dopt_emg_syn3_k4" not in names
    for method in plan.CARRIER_K_METHODS:
        for budget in (2, 3):
            assert f"{method}_k{budget}" in names
    dry = plan.dry_plan()
    later = dry["carrier_k_later_day"]
    assert later["formal_benchmark_verdict"] is False
    assert later["selector_pool_is_support_only"] is True
    assert later["encoder_neural_trials"] == 4
    assert later["n_variants"] == 7
    assert dry["prospective_roots"]["carrier_k_later_day_b3s_rsyn3_freeze"].endswith(
        "carrier_k_later_day_v1/b3s_rsyn3_freeze"
    )
    with pytest.raises(plan.PlanError, match="later-day carrier-k"):
        plan.carrier_k_later_day_root_relative("b3")
    assert "tfpd_exploration/src/m1_b3_allsource_v1/carrier_k_later_day.py" in plan.OWNED_PATHS
    assert "tfpd_exploration/scripts/run_m1_b3_allsource_carrier_k_later_day.py" in plan.OWNED_PATHS
    cli = ROOT / "tfpd_exploration/scripts/run_m1_b3_allsource_carrier_k_later_day.py"
    denied = subprocess.run(
        [PYTHON, str(cli), "--execute-gpu"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True,
    )
    assert denied.returncode != 0
    assert "cannot mint a GPU capability" in (denied.stderr + denied.stdout)


def test_top4_package_arms_keep_m10_encoder() -> None:
    from tfpd_exploration.src.m1_b3_allsource_v1 import runtime
    from tfpd_exploration.src.m1_b3_allsource_v1 import submit_remote

    assert plan.TOP4_PACKAGE_ARMS == ("b3s_rsyn3_freeze_top4", "b3s_rsyn3_acyc_top4")
    assert plan.TOP4_CARRIER_METHOD == "dopt_tgt_loc"
    assert plan.TOP4_CARRIER_K == 4
    assert plan.source_train_arm("b3s_rsyn3_freeze_top4") == "b3s_rsyn3_freeze"
    assert plan.source_train_arm("b3s_rsyn3_acyc_top4") == "b3s_rsyn3_acyc"
    assert plan.ARM_SPECS["b3s_rsyn3_freeze_top4"]["encoder_neural_trials"] == 10
    assert plan.ARM_SPECS["b3s_rsyn3_acyc_top4"]["activity_prefix_cycle"] == (10, 5, 2)
    with pytest.raises(plan.PlanError, match="cannot train"):
        plan.validate_mode_arm("train", "b3s_rsyn3_freeze_top4")
    plan.validate_mode_arm("package", "b3s_rsyn3_freeze_top4")
    assert plan.package_root_relative("b3s_rsyn3_freeze_top4").endswith(
        "export/b3s_rsyn3_freeze_top4"
    )
    assert plan.remote_push_root_relative("b3s_rsyn3_acyc_top4").endswith(
        "remote_push_v1/b3s_rsyn3_acyc_top4"
    )
    assert runtime.EXPECTED_ARMS == plan.REMOTE_ARMS
    assert "b3s_rsyn3_freeze_top4" in submit_remote.ARM_META
    tag = submit_remote.image_tag_for("b3s_rsyn3_freeze_top4", "abcd" * 16)
    assert tag.startswith("spint-b3s-rsyn3-freeze-top4-m1:allsource-s42-")
    dry = plan.dry_plan()
    assert dry["top4_package_arms"] == ["b3s_rsyn3_freeze_top4", "b3s_rsyn3_acyc_top4"]
    timed = ROOT / "tfpd_exploration/scripts/run_m1_b3_allsource_remote_timed.py"
    denied = subprocess.run(
        [PYTHON, str(timed), "--execute-gpu"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True,
    )
    assert denied.returncode != 0
    assert "cannot mint a GPU capability" in (denied.stderr + denied.stdout)


def test_carrier_k_selectors_and_selected_encode() -> None:
    import numpy as np

    from tfpd_exploration.src.m1_b3_allsource_v1 import carrier_k
    from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1.data import SessionBins
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1.syn3 import SourceBasis

    np.testing.assert_array_equal(carrier_k.select_chronological(4), np.arange(4))
    np.testing.assert_array_equal(carrier_k.select_chronological(10), np.arange(10))

    clustered = np.zeros(10, dtype=np.float64)
    clustered[7] = 0.5 * np.pi
    clustered[8] = np.pi
    clustered[9] = 1.5 * np.pi
    tgt = carrier_k.select_dopt_tgt_loc(clustered, 3)
    assert tgt.shape == (3,)
    assert not np.array_equal(np.sort(tgt), np.arange(3))
    assert {7, 8, 9}.issuperset(set(tgt.tolist()) - {0, 1, 2, 3, 4, 5, 6})

    sparse = np.zeros((10, 3), dtype=np.float64)
    sparse[:7] = (1.0, 0.0, 0.0)
    sparse[7] = (0.0, 1.0, 0.0)
    sparse[8] = (0.0, 0.0, 1.0)
    sparse[9] = (1.0, 1.0, 0.0)
    emg_sel = carrier_k.select_dopt_emg_syn3(sparse, 3)
    assert emg_sel.shape == (3,)
    assert not np.array_equal(np.sort(emg_sel), np.arange(3))
    assert len(set(emg_sel.tolist()) & {7, 8, 9}) >= 2

    short = np.array([0.0, np.nan, np.nan, np.nan], dtype=np.float64)
    with pytest.raises(carrier_k.CarrierKError, match="finite"):
        carrier_k.select_dopt_tgt_loc(short, 3)

    n_trials, bins, n_emg, n_units = 10, 8, 6, 4
    emg_rows = []
    rate_rows = []
    ids = []
    for trial in range(n_trials):
        for _ in range(bins):
            row = np.zeros(n_emg, dtype=np.float64)
            row[trial % 3] = 1.0 + 0.2 * trial
            emg_rows.append(row)
            rate_rows.append(np.full(n_units, float(trial + 1), dtype=np.float64))
            ids.append(trial)
    ids_np = np.asarray(ids, dtype=np.int64)
    record = SessionBins(
        session="ses-toy",
        path=Path("/tmp/toy.nwb"),
        path_sha256="0" * 64,
        emg=np.vstack(emg_rows),
        emg_trial_ids=ids_np,
        rates=np.vstack(rate_rows),
        rate_trial_ids=ids_np.copy(),
        n_trials=n_trials,
        channel_names=tuple(f"ch{i}" for i in range(n_emg)),
        signal_view={"query_neural_or_emg_values_read": False},
    )
    basis = SourceBasis(
        kind="nnmf",
        scale=np.ones(n_emg, dtype=np.float64),
        dictionary=np.eye(3, n_emg, dtype=np.float64),
        activations=np.zeros((1, 3), dtype=np.float64),
        order=(0, 1, 2),
        reconstruction_digest="toy",
        library={},
        extra={},
    )
    prefix = rsyn3_bank._encode_record(record, basis, budget=4)
    selected_prefix = rsyn3_bank._encode_record(
        record, basis, selected_trial_ids=np.arange(4, dtype=np.int64),
    )
    later = rsyn3_bank._encode_record(
        record, basis, selected_trial_ids=np.asarray([0, 2, 5, 7], dtype=np.int64),
    )
    assert prefix.shape == (n_units, 4)
    np.testing.assert_allclose(prefix, selected_prefix)
    assert not np.allclose(prefix, later)

    means = carrier_k.trial_mean_synergy(record, basis, pool_trials=10)
    assert means.shape == (10, 3)
    assert np.isfinite(means).all()


def test_carrier_k_cli_refuses_gpu() -> None:
    denied = subprocess.run(
        [PYTHON, str(CARRIER_K_CLI), "--execute-gpu"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True,
    )
    assert denied.returncode != 0
    assert "cannot mint a GPU capability" in (denied.stderr + denied.stdout)
    dry = subprocess.run(
        [PYTHON, "-S", str(CARRIER_K_CLI), "--dry-run"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True, check=True,
    )
    payload = json.loads(dry.stdout)
    assert payload["carrier_k_probe"]["formal_benchmark_verdict"] is False
    assert payload["carrier_k_probe"]["n_variants"] == 13


def test_wave2_freeze_and_acyc_are_one_factor_vs_submitted_rsyn3() -> None:
    import torch

    from tfpd_exploration.src.m1_b3_allsource_v1 import acyc

    configs = ROOT / "streaming_calibration_exp/configs"
    freeze = (configs / "experiment/m1_b3_allsource_b3s_rsyn3_freeze.yaml").read_text(encoding="utf-8")
    prefix = (configs / "experiment/m1_b3_allsource_b3s_rsyn3_acyc.yaml").read_text(encoding="utf-8")
    for text in (freeze, prefix):
        assert "max_epochs: 12" in text
        assert "loss_mode: task_only" in text
        assert "falcon_m1_all_source_b3s_rsyn3" in text
        assert "streaming_b3s_rsyn3_m1" in text
        assert "test: false" in text
    assert "freeze_decoder: true" in freeze
    assert "freeze_decoder: false" in prefix
    assert "m1_all_source_rsyn3_acyc" in prefix
    assert "m1_all_source_rsyn3_acyc" not in freeze
    callback = (configs / "callbacks/m1_all_source_rsyn3_acyc.yaml").read_text(encoding="utf-8")
    assert "ActivityPrefixCallback" in callback
    assert "[10, 5, 2]" in callback or "10, 5, 2" in callback

    operator = acyc.ActivityPrefixOperator(cycle=(10, 5, 2), pool_trials=10)
    calib = torch.zeros(2, 10, 8, 4)

    class _Student:
        def __init__(self) -> None:
            self.training = True

        def register_forward_pre_hook(self, fn, with_kwargs=True):
            del with_kwargs
            self.hook = fn
            return object()

    student = _Student()
    operator.attach(student)
    widths = []
    for _ in range(6):
        _args, kwargs = student.hook(student, (), {"calib_trials": calib})
        widths.append(int(kwargs["calib_trials"].shape[1]))
    assert widths == [10, 5, 2, 10, 5, 2]
    student.training = False
    assert student.hook(student, (), {"calib_trials": calib}) is None
    assert "tfpd_exploration/src/m1_b3_allsource_v1/acyc.py" in plan.OWNED_PATHS

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(configs)):
        freeze_cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=m1_b3_allsource_b3s_rsyn3_freeze"],
        )
        acyc_cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=m1_b3_allsource_b3s_rsyn3_acyc"],
        )
    assert bool(freeze_cfg.model.freeze_decoder) is True
    assert bool(acyc_cfg.model.freeze_decoder) is False
    assert str(freeze_cfg.data.side_feature_group).lower() == "rsyn3"
    assert str(acyc_cfg.data.side_feature_group).lower() == "rsyn3"
    assert "ActivityPrefixCallback" in str(acyc_cfg.callbacks)
