"""Small end-to-end contracts for the source-selected FAIR V2 WF pipeline."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
EXTERNAL = Path(__file__).resolve().parents[2]
if str(EXTERNAL) not in sys.path:
    sys.path.insert(0, str(EXTERNAL))
from fair_v2 import metrics, run


def _item(seed: int, *, scale: float = 1.0, shift: float = 0.0, y_shift: float = 0.0) -> dict:
    rng = np.random.RandomState(seed)
    n, channels, context = 80, 6, 3
    raw = rng.normal(scale=scale, size=(n, channels)).astype(np.float32) + shift
    # Each target depends on a causal raw signal, so a pooled ridge has a real
    # signal to learn.  Source labels differ by session on purpose.
    y = np.column_stack((raw[2:, 0] + .35 * raw[1:-1, 1] + y_shift,
                         -.45 * raw[2:, 2] + .20 * raw[:-2, 3] - y_shift)).astype(np.float32)
    starts = np.arange(context - 1, n, dtype=np.int64)
    support = np.arange(0, 45, dtype=np.int64)
    x = np.pad(raw, ((context - 1, 0), (0, 0))).astype(np.float32)
    return {"X": x, "Y": y, "starts": starts, "pad": context - 1,
            "support_indices": support, "support": raw[support],
            "support_segments": [support],
            "support_provenance": {"raw_nwb": f"/synthetic/ses-{seed}.nwb", "trial_ids": [0]}}


def _config() -> dict:
    return {**run.DEFAULT_CONFIG, "bin_ms": 20, "tau_ms": 20, "filter_extent": 1,
            "history_bins": 3, "alphas": [100., 1000.], "coral_shrinkages": [0., .5, 1.],
            "fa_dims": [2], "stable_fractions": [1., .5], "fa_n_init": 1,
            "fa_max_iter": 10_000, "fa_retry_max_iter": 10_000, "fa_tol": 1e-5,
            "methods": list(run.METHODS), "seed": 17}


def _adapter_fingerprint(adapters: dict) -> tuple:
    got = []
    for session in sorted(adapters):
        normalizer = adapters[session]["normalizer"]
        got.extend((session, normalizer.mean_.tobytes(), normalizer.scale_.tobytes()))
        for method, transform in sorted(adapters[session]["methods"].items()):
            if transform is None:
                got.extend((method, None))
            else:
                got.extend((method, transform.matrix_.tobytes()))
                for name in ("intercept_", "source_mean_", "target_mean_"):
                    value = getattr(transform, name, None)
                    got.append(None if value is None else value.tobytes())
    return tuple(got)


def test_source_selection_is_pooled_chronological_and_target_label_blind():
    config = _config()
    train = {"a": _item(1, y_shift=0.), "b": _item(2, scale=1.2, y_shift=3.), "c": _item(3, shift=.8, y_shift=-2.)}
    metadata = {"context": 3}
    fitted, selection, source = run.source_fit(train, metadata, config)

    assert selection["source_session_count"] == 3
    assert selection["source_windows"] == sum(len(x["Y"]) for x in train.values())
    for session, split in selection["source_splits"].items():
        cutoff = int(np.floor(len(train[session]["Y"]) * .8))
        assert split["fit_rows"] == cutoff
        # tau=20 gives one smoothing tap, so the purge is exactly history=3.
        assert split["validation_first_raw_endpoint"] - split["fit_last_raw_endpoint"] >= 3
    assert selection["target_labels_used_for_fit"] is False
    assert selection["target_labels_used_for_selection"] is False

    # Change only a non-reference source label.  Pooled refitting must change
    # the learned decoder, proving it did not silently train on the reference.
    altered = copy.deepcopy(train)
    altered["a"]["Y"] += np.float32(29.)
    fitted_changed, _, _ = run.source_fit(altered, metadata, config)
    assert not np.array_equal(fitted["methods"]["diag_z_wf"]["readout"].coef_,
                              fitted_changed["methods"]["diag_z_wf"]["readout"].coef_)

    evaluation = {"target": _item(9, scale=.7, shift=1.4, y_shift=.5)}
    pred_a, adapters_a, diagnostics_a = run.target_fit_and_predict(fitted, evaluation, metadata)
    y_replaced = copy.deepcopy(evaluation)
    y_replaced["target"]["Y"] = np.full_like(y_replaced["target"]["Y"], -999.)
    pred_b, adapters_b, diagnostics_b = run.target_fit_and_predict(fitted, y_replaced, metadata)
    for method in run.METHODS:
        np.testing.assert_array_equal(pred_a[method]["target"], pred_b[method]["target"])
    assert _adapter_fingerprint(adapters_a) == _adapter_fingerprint(adapters_b)
    assert diagnostics_a["target"]["input_hashes"]["Y"] != diagnostics_b["target"]["input_hashes"]["Y"]


def test_metrics_match_sklearn_legacy_and_real_h1_grouping():
    y = np.array([[0., 1.], [1., 3.], [3., 0.], [6., 2.]], dtype=np.float32)
    p = np.array([[.2, 1.4], [1.3, 2.8], [2.5, .1], [5.8, 2.3]], dtype=np.float32)
    got = metrics.score_arrays(y, p)
    assert got["standard_variance_weighted_r2"] == r2_score(y.astype(np.float64), p.astype(np.float64), multioutput="variance_weighted")
    assert got["legacy_flattened_r2"] == 1.0 - np.square(y - p).sum() / np.square(y - y.mean()).sum()

    root = Path("SPINT-main/data/000954/sub-HumanPitt-held-out-calib")
    paths = sorted(root.glob("*.nwb"))[:2]
    assert len(paths) == 2
    items = {path.stem: {"Y": y + i, "support_provenance": {"raw_nwb": str(path)}} for i, path in enumerate(paths)}
    predictions = {path.stem: p + i for i, path in enumerate(paths)}
    report = metrics.prediction_report("h1", items, predictions)
    assert report["n_sessions"] == 2 and report["n_windows"] == 8
    grouped = report["grouped_seven"]
    assert grouped["n_groups"] >= 1
    assert sum(row["n_windows"] for row in grouped["per_group"].values()) == 8
