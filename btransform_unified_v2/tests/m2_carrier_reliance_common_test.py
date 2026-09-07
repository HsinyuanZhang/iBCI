import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "m2_carrier_common", Path(__file__).parents[1] / "scripts/m2_carrier_reliance_v1/common.py"
)
common = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(common)


def test_direct_t4_arms_never_mutate_e0_or_mask():
    record = {"E0": np.arange(12, dtype=np.float32).reshape(3, 4),
              "carrier": np.arange(12, dtype=np.float32).reshape(3, 4),
              "unit_mask": np.array([True, False, True])}
    zero = common.clone_bank_for_arm(record, "zero")
    shuffled = common.clone_bank_for_arm(record, "shuffle", 101)
    assert np.array_equal(record["E0"], zero["E0"]) and np.array_equal(record["unit_mask"], zero["unit_mask"])
    assert np.all(zero["carrier"][[0, 2]] == 0) and np.array_equal(zero["carrier"][1], record["carrier"][1])
    assert np.array_equal(shuffled["E0"], record["E0"]) and np.array_equal(shuffled["unit_mask"], record["unit_mask"])


def test_m2_tag_and_three_date_bootstrap_contract():
    assert common.dataset_tag("ses-2020-10-30-Run2") == "Run2_20201030"
    result = common.paired_date_bootstrap({"2020-10-30": 1.0, "2020-11-18": 2.0, "2020-11-19": 3.0}, draws=20)
    assert result["equal_date_group_mean"] == 2.0 and result["draws"] == 20
