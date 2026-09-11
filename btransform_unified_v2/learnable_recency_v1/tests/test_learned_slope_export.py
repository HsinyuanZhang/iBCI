from __future__ import annotations

import torch

from btransform_unified_v2.cpu_temporal import CpuRiftTemporalRuntime
from learnable_recency_v1.cpu_temporal import exported_cpu_runtime

from helpers import make_pair, scramble_new_params, stream


def test_export_constant_slopes_match_rift_temporal_and_cpu_runtime() -> None:
    _fixed, learnable = make_pair(50, "learned_slope", seed=23, width=16, heads=4)
    scramble_new_params(learnable, torch.Generator().manual_seed(23))
    learnable.eval()
    exported, runtime = exported_cpu_runtime(learnable, batch_size=2)
    exported_slopes = learnable.export_constant_slopes()
    torch.testing.assert_close(exported.recency_slopes, exported_slopes)
    z = torch.randn(2, 15, 16)
    torch.testing.assert_close(learnable(z), exported(z), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(learnable(z, backend="dense"), exported(z, backend="dense"), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(stream(learnable, z), stream(exported, z), atol=1e-6, rtol=1e-6)
    ref_state = exported.init_state(2, "cpu", torch.float32)
    for time in range(z.shape[1]):
        got = runtime.step(z[:, time])
        want, ref_state = exported.step(z[:, time], ref_state)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    assert isinstance(runtime, CpuRiftTemporalRuntime)


def test_per_layer_export_is_table_and_shared_vector_requires_agreement() -> None:
    _fixed, learnable = make_pair(50, "learned_slope", seed=23, width=16, heads=4, per_layer=True)
    table = learnable.export_constant_slopes()
    assert tuple(table.shape) == (4, 4)
    shared = learnable.export_shared_slope_vector()
    assert tuple(shared.shape) == (4,)
    scramble_new_params(learnable, torch.Generator().manual_seed(23))
    table = learnable.export_constant_slopes()
    assert tuple(table.shape) == (4, 4)
    try:
        learnable.export_shared_slope_vector()
    except RuntimeError as exc:
        assert "CpuLearnableRecencyRuntime" in str(exc)
    else:
        raise AssertionError("diverged per_layer slopes must not export a single vector")
