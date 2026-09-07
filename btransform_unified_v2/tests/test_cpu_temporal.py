import pytest
import torch
from btransform_unified_v2.config import RiftTemporalConfig
from btransform_unified_v2.temporal import RiftTemporal
from btransform_unified_v2.cpu_temporal import CpuRiftTemporalRuntime


@pytest.mark.parametrize("context,bias", [(200, "recency"), (300, "recency"), (300, "flat")])
def test_preallocated_cpu_runtime_matches_reference_across_two_capacity_cycles(context, bias):
    torch.manual_seed(context)
    config = RiftTemporalConfig.for_context(context, width=8, heads=2, ffn_width=16,
        half_life_seconds=(.08, None), bias_mode=bias)
    model = RiftTemporal(config).eval(); rt = CpuRiftTemporalRuntime(model, 2)
    ref = model.init_state(2, "cpu", torch.float32)
    steps = 2 * max(config.windows) + 3
    for t in range(steps):
        z = torch.randn(2, 8); valid = torch.tensor([t != 2, t > 1])
        got = rt.step(z, valid); want, ref = model.step(z, ref, valid)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert all(key.shape[1] == window - 1 for key, window in zip(rt.keys, config.windows))


def test_reset_and_reorder_match_reference():
    torch.manual_seed(9); config = RiftTemporalConfig(width=8, heads=2, ffn_width=16, windows=(3,2), half_life_seconds=(.1,None))
    model=RiftTemporal(config).eval(); rt=CpuRiftTemporalRuntime(model,2); ref=model.init_state(2,"cpu",torch.float32)
    for _ in range(5):
        z=torch.randn(2,8); got=rt.step(z); want,ref=model.step(z,ref); torch.testing.assert_close(got,want)
    rt.reorder([1,0]); ref=model.select_rows(ref,[1,0]); rt.reset_rows([1]); ref=model.reset_rows(ref,[1])
    for keys, values, lengths in zip(rt.keys, rt.values, rt.lengths):
        assert torch.count_nonzero(keys[1]) == 0
        assert torch.count_nonzero(values[1]) == 0
        assert lengths[1] == 0
    z=torch.randn(2,8); got=rt.step(z); want,ref=model.step(z,ref); torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-5)
