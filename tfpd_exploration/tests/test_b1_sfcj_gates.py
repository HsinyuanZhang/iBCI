"""CPU tests for §10.3 revised dimension logic and independent L_A0 selection."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tfpd_exploration.src.b1_sfcj_v1 import gates


def _dual(pass_flag: bool) -> dict:
    inner = {"pass": pass_flag, "mean_relative_gain": 0.05 if pass_flag else 0.0}
    return gates.dual_law_content_success(inner, inner)


def test_dimension_rule_only_among_content_gate_survivors():
    dim = {"mean_relative_dimension_gain": 0.02, "positive_dates": 3, "per_date": {}}
    # only SFC9 of native passes
    out = gates.apply_dimension_after_content_gate(
        _dual(False),
        _dual(True),
        _dual(False),
        _dual(False),
        dim,
        dim,
    )
    assert out["native"]["status"] == "selected_sole_survivor"
    assert out["native"]["q"] == 9
    assert out["jr1"]["status"] == "exited"
    assert out["jr1"]["q"] is None
    # both q pass -> dimension rule may pick 9
    both = gates.apply_dimension_after_content_gate(
        _dual(True),
        _dual(True),
        _dual(True),
        _dual(True),
        dim,
        dim,
    )
    assert both["native"]["q"] == 9
    assert both["jr1"]["status"] == "selected"


def test_L_A0_selected_independently_of_method():
    mse_fixed = {"20210626": 0.10, "20210627": 0.10, "20210628": 0.10}
    mse_grow_a0 = {"20210626": 0.08, "20210627": 0.08, "20210628": 0.08}  # +20%
    mse_grow_method = {"20210626": 0.11, "20210627": 0.11, "20210628": 0.11}  # worse
    l_a0 = gates.select_L_arm(mse_fixed, mse_grow_a0)
    l_method = gates.select_L_arm(mse_fixed, mse_grow_method)
    assert l_a0["L_arm"] == "GROWING"
    assert l_method["L_arm"] == "FIXED3"
    assert l_a0["L_arm"] != l_method["L_arm"]
