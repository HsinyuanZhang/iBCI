from __future__ import annotations

import numpy as np

from scripts.h1_carrierid_evaluate import _r2


def test_carrierid_r2_accumulates_float32_predictions_in_float64():
    rng = np.random.default_rng(0)
    truth = (rng.normal(size=(8965, 7)) * 1.0e-3 + 1.0).astype(np.float32)
    estimate = (truth + rng.normal(scale=1.0e-4, size=truth.shape).astype(np.float32)).astype(np.float32)

    truth64 = truth.astype(np.float64)
    estimate64 = estimate.astype(np.float64)
    sse64 = float(np.square(truth64 - estimate64).sum())
    centered64 = truth64 - truth64.mean(axis=0, keepdims=True)
    expected64 = float(1.0 - sse64 / float(np.square(centered64).sum()))

    sse32 = float(np.square(truth - estimate).sum())
    centered32 = truth - truth.mean(axis=0, keepdims=True)
    old_float32_result = float(1.0 - sse32 / float(np.square(centered32).sum()))

    assert _r2(truth, estimate) == expected64
    assert expected64 != old_float32_result
