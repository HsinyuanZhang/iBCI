"""Tests for swa_utils_dandi688.average_student_state_dicts (E1 SWA weight averaging).

Pure-tensor tests, no GPU, no NWB data, no Lightning module construction: the function under
test only ever sees plain state_dict-shaped ``dict[str, torch.Tensor]`` objects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import swa_utils_dandi688 as swa  # noqa: E402


def _sd(**tensors: torch.Tensor) -> dict[str, torch.Tensor]:
    return dict(tensors)


def test_averages_float_tensors_arithmetically():
    a = _sd(**{"student.w": torch.tensor([1.0, 2.0, 3.0])})
    b = _sd(**{"student.w": torch.tensor([3.0, 4.0, 5.0])})
    c = _sd(**{"student.w": torch.tensor([2.0, 0.0, 1.0])})
    out = swa.average_student_state_dicts([a, b, c])
    assert torch.allclose(out["student.w"], torch.tensor([2.0, 2.0, 3.0]))


def test_averaged_dtype_matches_input_dtype():
    a = _sd(**{"student.w": torch.tensor([1.0, 2.0], dtype=torch.float32)})
    b = _sd(**{"student.w": torch.tensor([3.0, 4.0], dtype=torch.float32)})
    out = swa.average_student_state_dicts([a, b])
    assert out["student.w"].dtype == torch.float32
    assert torch.allclose(out["student.w"], torch.tensor([2.0, 3.0]))


def test_single_checkpoint_average_is_identity():
    a = _sd(**{"student.w": torch.tensor([1.0, 2.0, 3.0])})
    out = swa.average_student_state_dicts([a])
    assert torch.equal(out["student.w"], a["student.w"])


def test_teacher_tensors_copied_not_averaged_when_identical():
    a = _sd(**{"teacher.w": torch.tensor([5.0, 5.0]), "student.w": torch.tensor([1.0])})
    b = _sd(**{"teacher.w": torch.tensor([5.0, 5.0]), "student.w": torch.tensor([3.0])})
    out = swa.average_student_state_dicts([a, b])
    # Teacher unchanged (not averaged to itself trivially -- explicitly copied).
    assert torch.equal(out["teacher.w"], torch.tensor([5.0, 5.0]))
    # Student IS averaged.
    assert torch.allclose(out["student.w"], torch.tensor([2.0]))


def test_teacher_tensor_mismatch_raises_instead_of_averaging():
    # If the teacher ever differs across the window (it shouldn't -- it's frozen), this must
    # be treated as an integrity failure, not silently averaged or silently accepted.
    a = _sd(**{"teacher.w": torch.tensor([5.0, 5.0])})
    b = _sd(**{"teacher.w": torch.tensor([5.0, 6.0])})
    with pytest.raises(ValueError, match="teacher tensor"):
        swa.average_student_state_dicts([a, b])


def test_non_floating_point_tensor_copied_when_identical():
    a = _sd(**{"student.count": torch.tensor([3, 3], dtype=torch.int64)})
    b = _sd(**{"student.count": torch.tensor([3, 3], dtype=torch.int64)})
    out = swa.average_student_state_dicts([a, b])
    assert out["student.count"].dtype == torch.int64
    assert torch.equal(out["student.count"], torch.tensor([3, 3], dtype=torch.int64))


def test_non_floating_point_tensor_mismatch_raises_instead_of_averaging():
    # An integer buffer (e.g. a step counter) must never be averaged -- confirm the function
    # refuses rather than silently doing float-style arithmetic on it.
    a = _sd(**{"student.step": torch.tensor(3, dtype=torch.int64)})
    b = _sd(**{"student.step": torch.tensor(5, dtype=torch.int64)})
    with pytest.raises(ValueError, match="non-floating-point tensor"):
        swa.average_student_state_dicts([a, b])


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        swa.average_student_state_dicts([])


def test_rejects_mismatched_key_sets():
    a = _sd(**{"student.w": torch.tensor([1.0])})
    b = _sd(**{"student.v": torch.tensor([1.0])})
    with pytest.raises(ValueError, match="key set"):
        swa.average_student_state_dicts([a, b])


def test_rejects_mismatched_shapes():
    a = _sd(**{"student.w": torch.tensor([1.0, 2.0])})
    b = _sd(**{"student.w": torch.tensor([1.0, 2.0, 3.0])})
    with pytest.raises(ValueError, match="shape/dtype"):
        swa.average_student_state_dicts([a, b])


def test_rejects_mismatched_dtypes():
    a = _sd(**{"student.w": torch.tensor([1.0, 2.0], dtype=torch.float32)})
    b = _sd(**{"student.w": torch.tensor([1.0, 2.0], dtype=torch.float64)})
    with pytest.raises(ValueError, match="shape/dtype"):
        swa.average_student_state_dicts([a, b])


def test_averaging_is_order_independent():
    a = _sd(**{"student.w": torch.tensor([1.0, 5.0])})
    b = _sd(**{"student.w": torch.tensor([3.0, 1.0])})
    c = _sd(**{"student.w": torch.tensor([2.0, 0.0])})
    out_abc = swa.average_student_state_dicts([a, b, c])
    out_cab = swa.average_student_state_dicts([c, a, b])
    assert torch.allclose(out_abc["student.w"], out_cab["student.w"])


def test_large_window_matches_manual_mean():
    torch.manual_seed(0)
    tensors = [torch.randn(4, 4) for _ in range(20)]
    state_dicts = [_sd(**{"student.w": t}) for t in tensors]
    out = swa.average_student_state_dicts(state_dicts)
    manual_mean = torch.stack(tensors, dim=0).mean(dim=0)
    assert torch.allclose(out["student.w"], manual_mean, atol=1e-6)
