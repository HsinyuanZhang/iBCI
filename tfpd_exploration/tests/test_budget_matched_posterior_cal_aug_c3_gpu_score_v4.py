from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "tfpd_exploration",
    ROOT / "tfpd_exploration/src",
    ROOT / "sua_exploration",
    ROOT / "streaming_calibration_exp/src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from budget_matched_posterior_cal_aug_c3_v1 import score_gpu_v4 as v4


def test_static_dry_cli_is_torch_free() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_score_v4.py"
    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_MODEL_NO_CUDA_NO_WRITE"
    assert payload["gpu_batch_candidates"] == [1024, 512, 128]


def test_closures_are_split_and_current() -> None:
    execution = v4.execution_closure(ROOT)
    review = v4.review_closure(ROOT)
    assert set(execution["files"]).isdisjoint(review["files"])
    assert execution["closure_sha256"] != review["closure_sha256"]


class _Inputs:
    def __init__(self) -> None:
        self.starts = np.arange(0, 65, dtype=np.int64)
        self.neural = np.arange((65 + 49) * 5, dtype=np.float32).reshape(65 + 49, 5) / 1000.0


class _Model:
    def compute_identity(self, activity, *, side_features):
        return (activity.mean() + side_features.mean()).reshape(1, 1)

    def decode_with_identity(self, neural, identity):
        import torch
        base = neural.mean(dim=-1, keepdim=True)
        return torch.cat((base + identity[:, None, :], base - identity[:, None, :]), dim=-1)


class _Runtime:
    def __init__(self, torch):
        self._torch = torch


def test_decode_batch_change_is_bitwise_on_cpu_synthetic() -> None:
    import torch
    inputs = _Inputs()
    runtime = _Runtime(torch)
    model = _Model()
    activity = torch.arange(4 * 100 * 5, dtype=torch.float32).reshape(4, 100, 5) / 100.0
    side = torch.arange(5 * 4, dtype=torch.float32).reshape(1, 5, 4) / 10.0
    small = v4._decode_on_device(
        runtime, inputs, model, activity, side,
        device=torch.device("cpu"), batch_size=32,
    )
    large = v4._decode_on_device(
        runtime, inputs, model, activity, side,
        device=torch.device("cpu"), batch_size=128,
    )
    assert torch.equal(small, large)


def test_parity_evidence_pass_and_fail() -> None:
    import torch

    class Inputs:
        surface = "within"
        session = "s"
        last_targets = np.arange(16, dtype=np.float32).reshape(8, 2)
        last_valid_mask = np.ones(8, dtype=np.bool_)

    class Metric:
        @staticmethod
        def session_r2(prediction, target):
            return float(((prediction - target) ** 2).mean().item())

    base = torch.arange(8 * 50 * 2, dtype=torch.float32).reshape(8, 50, 2) / 100.0
    evidence = v4._parity_evidence(base, base.clone(), base.clone(), inputs=Inputs(), matched_scorer=Metric())
    assert evidence["passed"] is True
    changed = base.clone()
    changed[0, 0, 0] += 1e-3
    with pytest.raises(v4.GPUScoreV4Error, match="parity"):
        v4._parity_evidence(base, changed, changed, inputs=Inputs(), matched_scorer=Metric())


def test_predecessor_and_producer_metadata_remain_valid() -> None:
    from budget_matched_posterior_cal_aug_c3_v1 import score as v1
    from budget_matched_posterior_cal_aug_c3_v1 import score_v2 as v2
    from budget_matched_posterior_cal_aug_c3_v1 import score_v3 as v3
    assert v2.validate_failed_v1(ROOT)["exact_leaf_count"] == 4
    assert v3.validate_failed_v2(ROOT)["exact_leaf_count"] == 4
    assert v1.validate_c2_producer(ROOT)["exact_leaf_count"] == v1.EXPECTED_FULL_LEAVES
    assert v1.validate_c3_producer(ROOT, "real")["arm"] == "real"
