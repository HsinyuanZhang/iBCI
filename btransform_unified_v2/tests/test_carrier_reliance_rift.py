"""Independent no-dataset contracts for the RIFT carrier-reliance runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"carrier_reliance_v1"/"run_h1_rift.py"
sys.path[:0]=[str(ROOT/"scripts"/"carrier_reliance_v1"),str(ROOT/"src"),str(ROOT.parent/"btransform_unified_v1"/"src")]
spec=importlib.util.spec_from_file_location("rift_carrier_runner_test",SCRIPT)
assert spec and spec.loader
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def _record() -> dict:
    return {"session":"fixture","key":"S6_set_1","group":"S6","neural":np.arange(800*176,dtype=np.float32).reshape(800,176),"E0":np.ones((176,700),np.float32),"carrier":np.arange(176*4,dtype=np.float32).reshape(176,4),"unit_mask":np.ones(176,bool),"selected_endpoints":np.asarray([0,3,299,450],np.int64)}


def test_windows_is_bounded_left_padded_and_uses_real_endpoints() -> None:
    record=_record(); x,valid=runner.windows(record,record["selected_endpoints"])
    assert x.shape==(4,300,176) and valid.shape==(4,300)
    assert valid[0].sum()==1 and np.array_equal(x[0,-1],record["neural"][0])
    assert valid[2].all() and np.array_equal(x[2],record["neural"][:300])
    assert valid[3].all() and np.array_equal(x[3],record["neural"][151:451])


def test_direct_carrier_arms_copy_every_other_bank_array() -> None:
    record=_record(); original_e0=record["E0"].copy(); original_c=record["carrier"].copy(); original_m=record["unit_mask"].copy()
    real=runner.make_bank(record,"REAL"); zero=runner.make_bank(record,"C_ZERO"); shuffled=runner.make_bank(record,"C_SHUF101")
    assert np.array_equal(real.E0,original_e0) and np.array_equal(zero.E0,original_e0) and np.array_equal(shuffled.E0,original_e0)
    assert np.array_equal(real.unit_mask,original_m) and np.array_equal(zero.unit_mask,original_m)
    assert np.array_equal(zero.carrier,np.zeros_like(original_c))
    assert np.array_equal(np.sort(shuffled.carrier,axis=0),np.sort(original_c,axis=0))
    assert np.array_equal(record["E0"],original_e0) and np.array_equal(record["carrier"],original_c) and np.array_equal(record["unit_mask"],original_m)


def test_arm_names_are_fixed_and_bootstrap_is_group_not_record_level() -> None:
    assert runner.ARMS==("REAL","C_ZERO","C_SHUF101","C_SHUF102","C_SHUF103")
    values={f"S{n}":float(n) for n in range(6,13)}
    out=runner.common.grouped_bootstrap(values)
    assert out["draws"]==2000 and out["seed"]==20260907 and out["equal_group_mean"]==9.0
