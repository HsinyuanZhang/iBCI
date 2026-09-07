import numpy as np
import pytest
from tfpd_exploration.src.m2_family_v1 import evaluate_frozen_ext4 as evaluator


def inputs():
    raw = np.zeros((300, 96), dtype=np.float32)
    starts = np.asarray([200, 205, 210, 215], dtype=np.int64)
    target = np.arange(8, dtype=np.float32).reshape(4, 2)
    mapping = {"support_horizon": 33, "support_trial_ids": list(range(33)),
               "apply_disjoint_on_this_query_file": True, "eligible_starts_padded": starts.tolist(),
               "query_window_audit": {"full_window_disjoint": True}, "support_boundary_padded_bin": 200}
    sealed = {"ordered_window_starts_sha256": evaluator.core.array_sha256(starts),
              "target_sha256": evaluator.core.array_sha256(target)}
    return raw, starts, target, mapping, sealed


def test_fixed_ext4_geometry_target_and_support():
    args = inputs()
    evaluator.verify_geometry(*args, 4)
    args[3]["support_horizon"] = 30
    with pytest.raises(RuntimeError, match="support/disjoint"):
        evaluator.verify_geometry(*args, 4)
    args = inputs()
    args[2][0, 0] = 100
    with pytest.raises(RuntimeError, match="target/start"):
        evaluator.verify_geometry(*args, 4)
    args = inputs()
    args[3]["support_boundary_padded_bin"] = 201
    with pytest.raises(RuntimeError, match="disjoint"):
        evaluator.verify_geometry(*args, 4)


def test_missing_go_refuses_before_model_or_authority(monkeypatch, tmp_path):
    monkeypatch.delenv("M2_FROZEN_EXT4_GO", raising=False)
    monkeypatch.setattr(evaluator, "OUT", tmp_path / "new-result")
    monkeypatch.setattr(evaluator, "authority", lambda: pytest.fail("no authority work without GO"))
    with pytest.raises(RuntimeError, match="explicit review GO"):
        evaluator.run(tmp_path / "absent-authorization.json")
