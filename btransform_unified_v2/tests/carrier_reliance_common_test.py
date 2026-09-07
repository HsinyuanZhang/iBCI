import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('carrier_common',Path(__file__).parents[1]/'scripts/carrier_reliance_v1/common.py')
common=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(common)

def test_carrier_arms_copy_without_mutating_e0_raw_or_mask():
    record={'E0':np.arange(12,dtype=np.float32).reshape(3,4),'carrier':np.arange(12,dtype=np.float32).reshape(3,4),'unit_mask':np.array([True,False,True])}
    normal=common.clone_bank_for_arm(record,'normal'); zero=common.clone_bank_for_arm(record,'zero'); shuffled=common.clone_bank_for_arm(record,'shuffle',101)
    assert np.array_equal(record['carrier'],normal['carrier'])
    assert np.all(zero['carrier'][[0,2]] == 0)
    assert np.array_equal(zero['carrier'][1],record['carrier'][1])
    assert np.array_equal(shuffled['E0'],record['E0']) and np.array_equal(shuffled['unit_mask'],record['unit_mask'])
