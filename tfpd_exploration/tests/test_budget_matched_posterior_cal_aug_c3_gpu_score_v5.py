from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
from budget_matched_posterior_cal_aug_c3_v1 import score_gpu_v5 as v5


def test_failed_v4_graph_is_exact() -> None:
    binding = v5.validate_failed_v4(ROOT)
    assert binding["attempt_sha256"] == v5.FAILED_V4_ATTEMPT_SHA256
    assert binding["exact_leaf_count"] == 4
    assert binding["disposition"].endswith("UUID_PREFIX_FORMAT_ONLY")


def test_torch_uuid_is_canonicalized_after_attempt() -> None:
    class UUID:
        def __str__(self):
            return v4.EXPECTED_GPU_UUID.removeprefix("GPU-")

    class Properties:
        name = v4.EXPECTED_GPU_NAME
        uuid = UUID()
        total_memory = 25_438_126_080
        major = 8
        minor = 6

    class Matmul:
        allow_tf32 = True

    class CudaBackend:
        matmul = Matmul()

    class Cudnn:
        allow_tf32 = True
        benchmark = True
        deterministic = False

    class Backends:
        cuda = CudaBackend()
        cudnn = Cudnn()

    class Cuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 1
        @staticmethod
        def get_device_properties(index): return Properties()

    class Version:
        cuda = "test"

    class Torch:
        __version__ = "test"
        cuda = Cuda()
        backends = Backends()
        version = Version()
        enabled = False
        @classmethod
        def use_deterministic_algorithms(cls, value): cls.enabled = bool(value)
        @classmethod
        def are_deterministic_algorithms_enabled(cls): return cls.enabled

    result = v4._post_attempt_device_contract(Torch)
    assert result["uuid"] == v4.EXPECTED_GPU_UUID
    assert result["torch_uuid_raw"] == v4.EXPECTED_GPU_UUID.removeprefix("GPU-")


def test_static_v5_dry_cli() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_score_v5.py"
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
    assert payload["uuid_repair"] == "canonical_GPU_prefix_only"


def test_v5_closure_is_split() -> None:
    execution = v5.execution_closure(ROOT)
    review = v5.review_closure(ROOT)
    assert set(execution["files"]).isdisjoint(review["files"])
