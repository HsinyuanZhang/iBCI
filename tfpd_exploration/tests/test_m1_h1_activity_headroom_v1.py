from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tfpd_exploration.src.m1_h1_activity_headroom_v1 import core
from tfpd_exploration.src.m1_h1_activity_headroom_v1.m1 import write_once
from tfpd_exploration.src.m1_h1_activity_headroom_v1.m1_breadth import OFFICIAL_M1_FOLDS


def test_selection_contract_distinguishes_all_four_arms() -> None:
    static = core.selection_for_output_trial(
        core.ActivityArm.STATIC_SUPPORT, output_trial_index=40,
        total_trials=100, support_trials=10,
    )
    rolling = core.selection_for_output_trial(
        core.ActivityArm.ROLLING_FIXED_M, output_trial_index=40,
        total_trials=100, support_trials=10,
    )
    growing = core.selection_for_output_trial(
        core.ActivityArm.CAUSAL_GROWING_CAP30, output_trial_index=40,
        total_trials=100, support_trials=10,
    )
    oracle = core.selection_for_output_trial(
        core.ActivityArm.FULL_SESSION_ORACLE, output_trial_index=40,
        total_trials=100, support_trials=10,
    )
    assert static == tuple(range(10))
    assert rolling == tuple(range(30, 40))
    assert growing == tuple(range(10, 40))
    assert oracle == tuple(range(100))


@pytest.mark.parametrize("arm", [
    core.ActivityArm.STATIC_SUPPORT,
    core.ActivityArm.ROLLING_FIXED_M,
    core.ActivityArm.CAUSAL_GROWING_CAP30,
])
def test_causal_arm_never_reads_current_or_future_trial(arm: core.ActivityArm) -> None:
    for current in range(10, 80):
        selected = core.selection_for_output_trial(
            arm, output_trial_index=current, total_trials=80, support_trials=10,
        )
        assert selected and max(selected) < current


def test_pre_support_windows_retain_original_static_support() -> None:
    expected = tuple(range(10))
    for arm in (
        core.ActivityArm.STATIC_SUPPORT,
        core.ActivityArm.ROLLING_FIXED_M,
        core.ActivityArm.CAUSAL_GROWING_CAP30,
    ):
        assert core.selection_for_output_trial(
            arm, output_trial_index=3, total_trials=414, support_trials=10,
        ) == expected


def test_variance_weighted_r2_is_pooled_sse_over_per_output_tss() -> None:
    truth = np.asarray([[0.0, 1.0], [1.0, 4.0], [2.0, 9.0]], dtype=np.float32)
    pred = truth + np.asarray([[0.1, -0.2], [0.0, 0.3], [-0.1, -0.1]], dtype=np.float32)
    centered = truth.astype(np.float64) - truth.astype(np.float64).mean(axis=0, keepdims=True)
    expected = 1.0 - np.square(truth.astype(np.float64) - pred).sum() / np.square(centered).sum()
    assert core.variance_weighted_r2(pred, truth) == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_array_digest_binds_dtype_shape_and_values() -> None:
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert core.array_digest(value) != core.array_digest(value.astype(np.float64))
    assert core.array_digest(value) != core.array_digest(value.reshape(2, 6))
    changed = value.copy(); changed[0, 0] = 1.0
    assert core.array_digest(value) != core.array_digest(changed)


def test_grouped_indices_preserves_first_seen_state_and_row_order() -> None:
    rows = core.grouped_indices(((0, 1), (1, 2), (0, 1), (2, 3)))
    assert rows == (((0, 1), (0, 2)), ((1, 2), (1,)), ((2, 3), (3,)))


def test_cached_m1_identity_is_exact_direct_projection() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc_id_in = torch.nn.Sequential(torch.nn.Linear(5, 6), torch.nn.ReLU())
            self.fc_id_out = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.ReLU())

    net = Net().eval()
    trials = np.random.default_rng(4).normal(size=(7, 5, 3)).astype(np.float32)
    encoded = core.encode_trial_activity(net, trials, family="m1", device="cpu", chunk_size=2)
    selection = (1, 3, 4, 6)
    cached = core.identity_from_encoded_trials(net, encoded, selection, family="m1")
    direct_trials = torch.as_tensor(trials[list(selection)]).permute(0, 2, 1)
    direct = net.fc_id_out(net.fc_id_in(direct_trials).mean(dim=0))
    assert float(torch.max(torch.abs(cached - direct))) <= 1.0e-7


def test_cached_h1_identity_keeps_carrier_fixed_and_is_exact() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(8)

    class Net(torch.nn.Module):
        carrier_dim = 2
        zero_carrier = False

        def __init__(self) -> None:
            super().__init__()
            self.carrier_pre_pool = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.ReLU())
            self.carrier_post_pool = torch.nn.Sequential(torch.nn.Linear(6, 3), torch.nn.ReLU())

    net = Net().eval()
    trials = np.random.default_rng(5).normal(size=(6, 5, 3)).astype(np.float32)
    carrier = np.random.default_rng(6).normal(size=(3, 2)).astype(np.float32)
    encoded = core.encode_trial_activity(net, trials, family="h1", device="cpu", chunk_size=3)
    selection = (0, 2, 5)
    cached = core.identity_from_encoded_trials(
        net, encoded, selection, family="h1", carrier=carrier,
    )
    direct_trials = torch.as_tensor(trials[list(selection)]).permute(0, 2, 1)
    pooled = net.carrier_pre_pool(direct_trials).mean(dim=0)
    direct = net.carrier_post_pool(torch.cat((pooled, torch.as_tensor(carrier)), dim=-1))
    assert float(torch.max(torch.abs(cached - direct))) <= 1.0e-7


def test_raw_state_identity_preserves_selected_trial_mlp_reduction() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(9)

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc_id_in = torch.nn.Sequential(
                torch.nn.Linear(5, 7), torch.nn.ReLU(), torch.nn.Linear(7, 7),
            )
            self.fc_id_out = torch.nn.Sequential(torch.nn.Linear(7, 4), torch.nn.ReLU())

    net = Net().eval()
    trials = np.random.default_rng(10).normal(size=(8, 5, 3)).astype(np.float32)
    selection = (1, 2, 5, 7)
    observed = core.identity_from_raw_trials(
        net, trials, selection, family="m1", device="cpu",
    )
    selected = torch.as_tensor(trials[list(selection)]).unsqueeze(0).permute(0, 1, 3, 2)
    expected = net.fc_id_out(net.fc_id_in(selected).mean(dim=1)[0])
    assert torch.equal(observed, expected)


def test_write_once_is_immutable_and_has_canonical_sidecar(tmp_path: Path) -> None:
    path, digest = write_once(tmp_path / "receipt.json", {"b": 2, "a": 1})
    assert path.stat().st_mode & 0o777 == 0o444
    sidecar = path.with_name(path.name + ".sha256")
    assert sidecar.stat().st_mode & 0o777 == 0o444
    assert sidecar.read_text() == f"{digest}  receipt.json\n"
    with pytest.raises(FileExistsError):
        write_once(path, {"a": 1})


def test_official_m1_breadth_specs_are_three_distinct_outer_targets() -> None:
    assert tuple(OFFICIAL_M1_FOLDS) == (0, 1, 2)
    assert [spec.target_session for spec in OFFICIAL_M1_FOLDS.values()] == [
        "20120924", "20120926", "20120927",
    ]
    assert len({spec.checkpoint_sha256 for spec in OFFICIAL_M1_FOLDS.values()}) == 3
    for fold, spec in OFFICIAL_M1_FOLDS.items():
        assert spec.fold == fold
        assert f"fold{fold}" in spec.manifest_relative
        assert len(spec.target_sha256) == len(spec.checkpoint_sha256) == 64
