"""CPU tests for m1_tier12_pack_v1. No 20-epoch training, no GPU jobs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from tfpd_exploration.src.m1_t0c1_prefix_v1 import plan as t0c1_plan
from tfpd_exploration.src.m1_t0c1_prefix_v1 import schedule as t0c1_schedule
from tfpd_exploration.src.m1_tier12_pack_v1 import chunk as chunk_module
from tfpd_exploration.src.m1_tier12_pack_v1 import gpu as pack_gpu
from tfpd_exploration.src.m1_tier12_pack_v1 import plan
from tfpd_exploration.src.m1_tier12_pack_v1 import receipts as pack_receipts


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tfpd_exploration/scripts/run_m1_tier12_pack_v1.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"


def _cli_env() -> dict[str, str]:
    return {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(ROOT)}


def test_plan_literals() -> None:
    assert plan.TRAIN_ARMS == ("t0_swa", "c1_swa", "t0_w32", "c1_w32")
    assert len(plan.TRAIN_ARMS) == 4
    assert plan.PACK_LIMIT == 2
    assert plan.CHUNK_BINS == 1024
    assert plan.CYCLE == (10, 5, 2)
    assert plan.CYCLE == t0c1_plan.CYCLE
    assert plan.COSINE_50EP_FORBIDDEN is True
    assert plan.SCHEDULER is None
    assert plan.SEED == 42 and plan.EPOCHS == 20 and plan.STEPS_PER_EPOCH == 4951
    assert plan.ADAM_LR == 1.0e-5 and plan.TRAIN_BATCH_SIZE == 32
    assert plan.OBJECTIVE_LAMBDA == 0.0
    assert plan.HELDIN_TRAINING_SESSIONS == ("20120926", "20120927", "20120928")
    assert "20121004" not in plan.HELDIN_TRAINING_SESSIONS
    assert plan.FOUR_SESSION_ALL_SOURCE_OUT_OF_SCOPE["in_train_roster"] is False
    dry = plan.dry_plan()
    assert dry["train_arms"] == list(plan.TRAIN_ARMS)
    assert dry["pack_limit"] == 2
    assert dry["chunk_bins"] == 1024
    assert dry["cycle"] == [10, 5, 2]
    assert dry["cosine_50ep_forbidden"] is True
    assert dry["four_session_all_source_in_train_roster"] is False
    assert dry["public_execution_authorized"] is True
    assert dry["imports_torch"] is False


def test_prefix_cycle_identical_to_t0c1() -> None:
    assert t0c1_schedule.sequence(7) == [10, 5, 2, 10, 5, 2, 10]
    assert plan.CYCLE == t0c1_plan.CYCLE == (10, 5, 2)
    assert plan.prefix_kind("t0_swa") == "t0"
    assert plan.prefix_kind("c1_w32") == "c1"
    assert plan.prefix_kind("c1_swa") == "c1"
    from tfpd_exploration.src.m1_t0c1_prefix_v1 import hook as t0c1_hook
    import torch

    operator = t0c1_hook.M1CalPrefixOperator("c1", record_steps=6)
    module = type("M", (), {"training": True})()
    seen = []
    for _ in range(6):
        calibration = torch.ones(4, 10, 4, 4)
        _args, kwargs_out = operator.forward_pre_hook(module, (), {
            "calib_trialized_neural_features": calibration,
        })
        seen.append(int(kwargs_out["calib_trialized_neural_features"].shape[1]))
    assert seen == [10, 5, 2, 10, 5, 2]


def test_gpu_pack_two_own_third_refused_foreign_refused_nonpack_refuses_apps() -> None:
    own = "tfpd_exploration/scripts/run_m1_tier12_pack_v1.py --arm t0_swa"
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

    with pytest.raises(pack_gpu.GpuError, match="compute apps"):
        pack_gpu.assert_target_gpu_launchable(1, cmdline_runner=busy)

    def one_own(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return f"{plan.GPU1_UUID}, 11, python, 958\n"
        return rows

    receipt = pack_gpu.assert_target_gpu_launchable(
        1, pack=True, cmdline_runner=one_own,
        pid_cmdline=lambda pid: own if pid == "11" else "other",
    )
    assert receipt["pack"] is True
    assert receipt["own_occupants"] == 1
    assert receipt["pack_limit"] == 2

    def two_own(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return (
                f"{plan.GPU1_UUID}, 11, python, 958\n"
                f"{plan.GPU1_UUID}, 12, python, 958\n"
            )
        return rows

    with pytest.raises(pack_gpu.GpuError, match="pack limit"):
        pack_gpu.assert_target_gpu_launchable(
            1, pack=True, cmdline_runner=two_own, pid_cmdline=lambda pid: own,
        )

    def foreign(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return f"{plan.GPU1_UUID}, 99, python, 1024\n"
        return rows

    with pytest.raises(pack_gpu.GpuError, match="foreign"):
        pack_gpu.assert_target_gpu_launchable(
            1, pack=True, cmdline_runner=foreign,
            pid_cmdline=lambda pid: "python tfpd_exploration/scripts/run_pit_m2_v1.py",
        )


def test_chunk_index_map_and_bootstrap() -> None:
    import numpy as np

    n_bins = 1024 * 5 + 100
    neural = np.zeros((n_bins, 64), dtype=np.float32)
    chunks = chunk_module.extract_chunks(neural)
    assert chunks.shape == (5, 1024, 64)
    assert chunk_module.n_complete_chunks(n_bins) == 5
    last_in_chunk_3 = 3 * 1024
    decision = chunk_module.selection_for_last_bin(last_in_chunk_3, 5)
    assert decision["chunk_k"] == 3
    assert decision["usable_chunk_indices"] == (0, 1, 2)
    assert decision["n_completed_before"] == 3
    assert decision["bootstrap"] is True
    assert decision["support_selection"] == tuple(range(10))
    assert "frozen support" in str(decision["hybrid_bootstrap_disclosure"]).lower() or (
        "support trials" in str(decision["hybrid_bootstrap_disclosure"])
    )
    later = chunk_module.selection_for_last_bin(12 * 1024, 12)
    assert later["bootstrap"] is False
    assert later["chunk_selection"] == tuple(range(2, 12))
    assert all(index < 12 for index in later["chunk_selection"])


def test_apply_swa_final_four_mean(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from tfpd_exploration.src.m1_tier12_pack_v1.train import apply_swa_final_four

    paths = []
    for index, fill in enumerate((1.0, 2.0, 3.0, 4.0)):
        path = tmp_path / f"epoch_{16 + index}.pt"
        torch.save({"state_dict": {"w": torch.ones(3, 3) * fill}}, path)
        paths.append(path)
    out = tmp_path / "swa_final4.pt"
    manifest = apply_swa_final_four(paths, out)
    assert manifest["fp64_arithmetic"] is True
    loaded = torch.load(out, map_location="cpu", weights_only=False)["state_dict"]
    assert torch.allclose(loaded["w"], torch.full((3, 3), 2.5))


def test_cli_dry_and_execute_gpu_error() -> None:
    dry = subprocess.run(
        [PYTHON, "-S", str(CLI), "--dry-run"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True, check=True,
    )
    payload = json.loads(dry.stdout)
    assert payload["phase"] == "m1_tier12_pack_v1"
    assert payload["public_execution_authorized"] is True
    assert payload["pack_limit"] == 2
    assert payload["train_arms"] == ["t0_swa", "c1_swa", "t0_w32", "c1_w32"]
    denied = subprocess.run(
        [PYTHON, str(CLI), "--execute-gpu"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True,
    )
    assert denied.returncode != 0
    assert "cannot mint a GPU capability" in (denied.stderr + denied.stdout)
    missing = subprocess.run(
        [PYTHON, str(CLI), "--execute"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True,
    )
    assert missing.returncode != 0
    text = missing.stderr + missing.stdout
    assert "in-process root-reviewed" not in text
    assert "--gpu-authorized" in text or "gpu-authorized" in text


def test_w32_constructor_kwargs() -> None:
    torch = pytest.importorskip("torch")
    from tfpd_exploration.src.m1_tier12_pack_v1.train import load_w32_model, w32_constructor_kwargs

    kwargs = w32_constructor_kwargs()
    assert kwargs["identity_width"] == 32
    assert kwargs["model_dim"] == 1024
    assert kwargs["window_size"] == 100
    model = load_w32_model(ROOT)
    last = list(model.fc_id_out.children())[-1]
    assert isinstance(last, torch.nn.Linear)
    assert last.out_features == 100
    assert model.identity_width == 32
    assert hasattr(model, "fc_id_in") and hasattr(model, "fc_id_out")


def test_receipts_refuse_sealed_t0c1_roots(tmp_path: Path) -> None:
    with pytest.raises(pack_receipts.ReceiptError, match="sealed"):
        pack_receipts.refuse_sealed_roots(ROOT / "tfpd_exploration/results/m1_t0c1_prefix_v1")
    with pytest.raises(pack_receipts.ReceiptError, match="sealed"):
        pack_receipts.refuse_sealed_roots(ROOT / "tfpd_exploration/results/m1_t0c1_prefix_v1/t0")
    pack_receipts.refuse_sealed_roots(tmp_path / "m1_tier12_pack_v1")
    pack_receipts.refuse_sealed_roots(ROOT / plan.RESULT_ROOT_RELATIVE)


def test_own_token_is_this_cli_not_rsyn3() -> None:
    assert plan.OWN_TOKEN == "run_m1_tier12_pack_v1.py"
    assert "rsyn3" not in plan.OWN_TOKEN
    assert plan.GPU0_UUID == "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
    assert plan.GPU1_UUID == "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"


def test_q10_end_is_score_time_slice_not_training() -> None:
    assert plan.QUERY_Q10_END == (10, None)
    assert plan.QUERY_Q10_END_SESSIONS == ("20120924",)
    assert plan.READ_RULES["governing_window"] == "full_session"
    assert plan.READ_RULES["comparison_window_q10_end"]["governing"] is False
    dry = plan.dry_plan()
    assert dry["query_q10_end"]["score_time_only"] is True
    assert dry["query_q10_end"]["governing"] is False


def test_q10_end_mask_keeps_trials_from_10() -> None:
    from tfpd_exploration.src.m1_tier12_pack_v1 import score as pack_score

    trials = tuple(range(20))
    mask = pack_score.query_row_mask(trials, start=10, stop=None)
    assert mask.tolist() == [False] * 10 + [True] * 10
    half = pack_score.query_row_mask(trials, start=10, stop=210)
    assert half.tolist() == [False] * 10 + [True] * 10


def test_restrict_opened_to_q10_end_filters_windows() -> None:
    from tfpd_exploration.src.m1_tier12_pack_v1 import score as pack_score
    import numpy as np

    class FakeDataset:
        def __init__(self) -> None:
            self.trial_start_indices = {"20120924": np.arange(20, dtype=np.int64) * 200}
            self.window_indices = [("20120924", int(start)) for start in self.trial_start_indices["20120924"]]
            self.neural_data = {"20120924": np.zeros((4000, 64), np.float32)}
            self.covariate_data = {"20120924": np.zeros((4000, 16), np.float32)}
            self.calib_trialized_neural_features = {"20120924": np.zeros((20, 1024, 64), np.float32)}

        def __len__(self) -> int:
            return len(self.window_indices)

    opened = {
        "dataset": FakeDataset(),
        "trials": np.zeros((20, 1024, 64), np.float32),
        "session_id": "20120924",
    }
    sliced = pack_score.restrict_opened_to_query(opened, start=10, stop=None)
    assert sliced["n_windows_full"] == 20
    assert sliced["n_windows_query"] == 10
    assert sliced["output_trial_min"] == 10
    assert sliced["output_trial_max"] == 19
    assert len(sliced["dataset"]) == 10
    assert sliced["trials"].shape[0] == 20


def test_swa_read_rules_ignore_q10_end_cells() -> None:
    from tfpd_exploration.src.m1_tier12_pack_v1 import score as pack_score

    rules = pack_score.evaluate_read_rules(
        sealed={"t0_static_20120924": 0.5707439184},
        cells={
            "t0_swa_static_m10_20120924": {"governing_r2": 0.5807439184},
            "t0_swa_static_m10_q10_end_20120924": {"governing_r2": 0.99},
        },
    )
    assert rules["swa_t0"]["swa_static"] == 0.5807439184
    assert rules["swa_t0"]["verdict"] == "HELPS"
