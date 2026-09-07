from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

from . import launcher
from .runtime import stable_state_hash


class _Parameter:
    def __init__(self, count: int) -> None:
        self._count = count

    def numel(self) -> int:
        return self._count


class _Student:
    def __init__(self) -> None:
        self._state = {
            "decoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "encoder.bias": torch.zeros(3, dtype=torch.float32),
        }

    def state_dict(self):
        return self._state

    def parameters(self):
        return [_Parameter(9), _Parameter(3)]

    def decoder_cost_comparison_receipt(self, *, batch_size: int, num_neurons: int):
        assert batch_size == 1 and num_neurons == 64
        return {
            "active_mode": "coupled",
            "coupled": {
                "total": 9876,
                "persistent_state_bytes_fp32": 2048,
            },
        }


class _Module:
    def __init__(self) -> None:
        self.student = _Student()


def test_stable_hash_is_order_independent() -> None:
    first = {"b": torch.ones(2), "a": torch.zeros(1)}
    second = {"a": torch.zeros(1), "b": torch.ones(2)}
    assert stable_state_hash(first) == stable_state_hash(second)


def test_initial_state_callback_is_synthetic_and_pre_fit(tmp_path: Path) -> None:
    output = tmp_path / "source_initial.json"
    # The repository's system Python intentionally has no Lightning.  Run
    # the real callback under the pinned spint interpreter instead of
    # skipping: this catches the actual import and callback target contract.
    provenance = launcher.provenance_bindings()
    provenance_args = ", ".join(
        f"{field}={value!r}" for field, value in provenance.items()
    )
    code = f'''
import json, torch
from sua_exploration.rt_seed_robustness_annex_v2.runtime import AnnexInitialStateCallback
class Parameter:
    def __init__(self, count): self.count = count
    def numel(self): return self.count
class Student:
    def __init__(self):
        self._state = {{"decoder.weight": torch.arange(6, dtype=torch.float32).reshape(2,3), "encoder.bias": torch.zeros(3)}}
    def state_dict(self): return self._state
    def parameters(self): return [Parameter(9), Parameter(3)]
    def decoder_cost_comparison_receipt(self, *, batch_size, num_neurons):
        assert batch_size == 1 and num_neurons == 64
        return {{"active_mode": "coupled", "coupled": {{"total": 9876, "persistent_state_bytes_fp32": 2048}}}}
class Module:
    def __init__(self): self.student = Student()
callback = AnnexInitialStateCallback(output_path={str(output)!r}, arm="afc4_mb4", fold=0, seed=43, run_id="synthetic-v2-1", initial_state_phase="before_first_optimizer_step", {provenance_args})
callback.on_fit_start(None, Module())
print(json.dumps(json.load(open({str(output)!r}))))
'''
    completed = subprocess.run(
        [str(launcher.DEFAULT_PYTHON), "-c", code],
        cwd=launcher.STREAMING_ROOT,
        env=launcher.execution_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_SOURCE_INITIAL_STATE_RECORDED"
    assert receipt["initial_state_scope"].startswith("student.state_dict")
    assert receipt["initial_state_phase"] == "before_first_optimizer_step"
    assert receipt["run_id"] == "synthetic-v2-1"
    assert receipt["accounting"]["parameter_count"] == 12
    assert receipt["accounting"]["macs_per_decode_call"] == 9876
    assert receipt["accounting"]["cached_state_bytes"] == 2048
    assert receipt["implementation_snapshot_sha256"] == provenance["implementation_snapshot_sha256"]
