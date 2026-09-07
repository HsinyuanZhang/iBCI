"""No-data/no-CUDA unit tests for the two act30 EvalAI M2 submission variants.

Covers both packages built from the validated v1 skeleton
(``submissions/evalai_m2_dopt_static_v1``):

- ``evalai_m2_act30_dopt4_v1``: D-opt-4 carrier (first-30 finite-angle
  candidates) + label-free first-30 B3S activity pool (sealed screen cell
  ``ridge_activity30_m4``).
- ``evalai_m2_act30_full_v1``: chronological first-30 (m30) carrier + the
  same label-free first-30 activity pool (sealed screen cell
  ``ridge_static_m30``).

Parameterized over the two variants: frozen selection/activity laws on
synthetic calibration sets, the ridge lambda=0.1 T4 fit, the BCIDecoder
adapter shapes and the zero-online-update contract, the packaging wiring
(Dockerfile/decode/label sync), the frozen submission helper, and the
export/terminal receipts.  Only synthetic tensors are used; no checkpoint,
no NWB data, no CUDA, no network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
for _path in (REPO_ROOT, REPO_ROOT / "SPINT-main"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

VARIANTS = {
    "act30_dopt4": {
        "package": TFPD_ROOT / "submissions/evalai_m2_act30_dopt4_v1",
        "laws_name": "laws_act30_dopt4_v1",
        "decoder_file": "act30_dopt4_decoder.py",
        "decoder_module": "act30_dopt4_decoder",
        "decoder_class": "Act30Dopt4M2Decoder",
        "submit_helper": "submit_evalai_act30_dopt4.py",
        "exporter": "export_act30_dopt4_payload.py",
        "payload_name": "t4_m2_seed42_dopt4_act30_identity.pkl",
        "result_root": TFPD_ROOT / "results/evalai_m2_act30_dopt4_v1",
        "arm": "dopt4_static_act30",
        "label_budget": 4,
        "activity_budget": 30,
        "sealed_cell": "ridge_activity30_m4",
        "anchor_external": 0.2909923623168425,
        "anchor_within": 0.6772200181830803,
        "image_tag": "spint-t4-m2:dopt4-act30-s42-e4ff17e8",
        "image_id": "sha256:79ac29ca71a84eb97be59411fc80d2a567127e95fd96b63dfa042d22b5a4421f",
        "payload_sha256": "e4ff17e857c0bab9bbd900bc737ca7c48476a44ed03725cefc5377d92b959261",
    },
    "act30_full": {
        "package": TFPD_ROOT / "submissions/evalai_m2_act30_full_v1",
        "laws_name": "laws_act30_full_v1",
        "decoder_file": "act30_full_decoder.py",
        "decoder_module": "act30_full_decoder",
        "decoder_class": "Act30FullM2Decoder",
        "submit_helper": "submit_evalai_act30_full.py",
        "exporter": "export_act30_full_payload.py",
        "payload_name": "t4_m2_seed42_m30_act30_identity.pkl",
        "result_root": TFPD_ROOT / "results/evalai_m2_act30_full_v1",
        "arm": "m30_static_act30",
        "label_budget": 30,
        "activity_budget": 30,
        "sealed_cell": "ridge_static_m30",
        "anchor_external": 0.29521985196829853,
        "anchor_within": 0.6892605728315127,
        "image_tag": "spint-t4-m2:m30-act30-s42-d60f38d5",
        "image_id": "sha256:ee4fc32075cb054494d69110d58f0b21654f733227fda36d1352007d585a07d9",
        "payload_sha256": "d60f38d5cefe45ea8fdccf9853d5034a15d7896b81b0fff3ebe983b7bd7b78a8",
    },
}

SEALED_SCREEN = TFPD_ROOT / "results/m2_t4_activity_budget_screen_v1/score.json"

_LAWS_CACHE: dict[str, object] = {}
_DECODER_CACHE: dict[str, object] = {}
_HELPER_CACHE: dict[str, object] = {}


def _load_file_module(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def laws_for(variant: str):
    if variant not in _LAWS_CACHE:
        _LAWS_CACHE[variant] = _load_file_module(
            VARIANTS[variant]["package"] / "laws.py", VARIANTS[variant]["laws_name"]
        )
    return _LAWS_CACHE[variant]


def decoder_module_for(variant: str):
    if variant not in _DECODER_CACHE:
        _DECODER_CACHE[variant] = _load_file_module(
            VARIANTS[variant]["package"] / VARIANTS[variant]["decoder_file"],
            VARIANTS[variant]["decoder_module"],
        )
    return _DECODER_CACHE[variant]


def helper_for(variant: str):
    if variant not in _HELPER_CACHE:
        _HELPER_CACHE[variant] = _load_file_module(
            VARIANTS[variant]["package"] / VARIANTS[variant]["submit_helper"],
            f"submit_evalai_{variant}",
        )
    return _HELPER_CACHE[variant]


def _synthetic_angles(seed: int, n_pool: int = 30, finite: int = 12, total: int = 35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.full(total, np.nan, dtype=np.float64)
    positions = rng.permutation(n_pool)[:finite]
    angles[positions] = (
        -3.0 * np.pi / 4.0 + rng.integers(0, 8, finite) * (np.pi / 4.0)
        + rng.normal(0.0, 0.05, finite)
    )
    return angles


def _synthetic_calib(seed: int, trials: int = 33) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(trials, 100, 96)).astype(np.float32)


# ---------------------------------------------------------------------------
# Law: selection (per variant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_selection_matches_the_sealed_screen_law(variant: str) -> None:
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import _support_indices

    laws = laws_for(variant)
    budget = VARIANTS[variant]["label_budget"]
    for seed in (0, 1, 2, 3, 4):
        angles = _synthetic_angles(seed)

        class _Shim:
            pass

        shim = _Shim()
        shim.calib_trial_target_angles = {"s": angles}
        sealed = _support_indices(shim, "s", budget)
        mirror = laws.select_dopt4_support(angles) if budget == 4 else laws.select_m30_support(angles)
        assert np.array_equal(mirror, sealed), f"seed {seed}: {mirror} vs {sealed}"


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_selection_is_deterministic(variant: str) -> None:
    laws = laws_for(variant)
    angles = _synthetic_angles(11)
    first = laws.select_dopt4_support(angles) if VARIANTS[variant]["label_budget"] == 4 else laws.select_m30_support(angles)
    second = laws.select_dopt4_support(angles) if VARIANTS[variant]["label_budget"] == 4 else laws.select_m30_support(angles)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_selection_stays_inside_the_first30_pool(variant: str) -> None:
    laws = laws_for(variant)
    angles = _synthetic_angles(5)
    selected = laws.select_dopt4_support(angles) if VARIANTS[variant]["label_budget"] == 4 else laws.select_m30_support(angles)
    assert int(selected.max()) < 30
    assert np.unique(selected).size == selected.size


def test_dopt4_selection_is_sorted_absolute_indices() -> None:
    laws = laws_for("act30_dopt4")
    angles = np.full(30, np.nan)
    angles[[2, 7, 11, 29]] = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    assert laws.select_dopt4_support(angles).tolist() == [2, 7, 11, 29]


def test_dopt4_selection_is_not_the_chronological_alias() -> None:
    laws = laws_for("act30_dopt4")
    angles = np.full(35, np.nan)
    angles[[1, 5, 11, 18, 24, 29]] = [0.0, np.pi / 3, 2 * np.pi / 3, np.pi, -2 * np.pi / 3, -np.pi / 3]
    selected = laws.select_dopt4_support(angles)
    assert selected.tolist() != [0, 1, 2, 3]


def test_dopt4_selection_fails_closed_below_four_candidates() -> None:
    laws = laws_for("act30_dopt4")
    angles = np.full(35, np.nan)
    angles[[3, 9, 21]] = [0.0, np.pi / 2, np.pi]
    with pytest.raises(laws.LawError):
        laws.select_dopt4_support(angles)


def test_dopt4_selection_fails_closed_without_first30_metadata() -> None:
    laws = laws_for("act30_dopt4")
    with pytest.raises(laws.LawError):
        laws.select_dopt4_support(np.full(29, np.nan))


def test_dopt4_greedy_beats_chronological_prefix_on_design_determinant() -> None:
    laws = laws_for("act30_dopt4")
    rng = np.random.default_rng(5)
    angles = -3.0 * np.pi / 4.0 + rng.integers(0, 8, 30) * (np.pi / 4.0) + rng.normal(0.0, 0.05, 30)
    selected = laws.select_dopt4_support(angles)
    design_d = laws.design_matrix_from_thetas(angles[selected])
    design_c = laws.design_matrix_from_thetas(angles[[0, 1, 2, 3]])
    assert np.linalg.det(design_d.T @ design_d) >= np.linalg.det(design_c.T @ design_c)


def test_dopt4_greedy_core_matches_the_sealed_doptimal_module() -> None:
    laws = laws_for("act30_dopt4")
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices as sealed_greedy,
    )

    thetas = _synthetic_angles(7)[np.isfinite(_synthetic_angles(7))]
    assert np.array_equal(
        laws.greedy_forward_d_optimal_indices(thetas, 4),
        sealed_greedy(thetas, 4),
    )


def test_m30_selection_is_the_chronological_first30_block() -> None:
    laws = laws_for("act30_full")
    assert laws.select_m30_support(_synthetic_angles(2)).tolist() == list(range(30))


def test_m30_selection_fails_closed_below_three_directional_trials() -> None:
    laws = laws_for("act30_full")
    angles = np.full(35, np.nan)
    angles[[0, 14]] = [0.0, np.pi / 2]
    with pytest.raises(laws.LawError):
        laws.select_m30_support(angles)


def test_m30_selection_ignores_direction_metadata_beyond_the_block() -> None:
    laws = laws_for("act30_full")
    angles = np.full(35, np.nan)
    angles[[0, 1, 2]] = [0.0, np.pi / 2, np.pi]
    angles[32] = np.pi  # outside the block; must not matter
    assert laws.select_m30_support(angles).tolist() == list(range(30))


# ---------------------------------------------------------------------------
# Law: activity pool (per variant, identical sealed law)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_activity_pool_is_exactly_the_first30_block(variant: str) -> None:
    laws = laws_for(variant)
    calib = _synthetic_calib(21)
    pool = laws.select_first30_activity_pool(calib)
    assert pool.shape == (30, 100, 96)
    assert pool.dtype == np.float32
    assert np.array_equal(pool, calib[:30])


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_activity_pool_matches_the_sealed_select_activity_rows(variant: str) -> None:
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import select_activity_rows

    laws = laws_for(variant)
    calib = _synthetic_calib(22)
    selected = (
        laws.select_dopt4_support(_synthetic_angles(22))
        if VARIANTS[variant]["label_budget"] == 4
        else laws.select_m30_support(_synthetic_angles(22))
    )
    assert np.array_equal(
        laws.select_first30_activity_pool(calib),
        select_activity_rows(calib, selected_indices=selected, activity_budget=30),
    )


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_activity_pool_fails_closed_below_30_trials(variant: str) -> None:
    laws = laws_for(variant)
    with pytest.raises(laws.LawError):
        laws.select_first30_activity_pool(_synthetic_calib(23, trials=29))


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_activity_pool_fails_closed_on_trial_shape_drift(variant: str) -> None:
    laws = laws_for(variant)
    rng = np.random.default_rng(24)
    with pytest.raises(laws.LawError):
        laws.select_first30_activity_pool(rng.normal(size=(33, 90, 96)).astype(np.float32))


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_verify_against_sealed_sources_covers_selection_and_pool(variant: str) -> None:
    laws = laws_for(variant)
    angles = _synthetic_angles(25)
    calib = _synthetic_calib(25)
    report = laws.verify_against_sealed_sources(angles, calib)
    assert report["mirror_equals_sealed"] is True
    assert report["activity_pool_rows"] == 30
    assert "activity_pool" in report["provenance"]
    assert report["provenance"]["ridge_t4"].startswith(
        "tfpd_exploration/src/calibration_budget_comparators_v1.py"
    )


# ---------------------------------------------------------------------------
# Ridge T4 law (sealed import exercised through each package)
# ---------------------------------------------------------------------------


def test_ridge_t4_zero_lambda_matches_lstsq() -> None:
    from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4

    theta = np.asarray([0.0, 0.5, 1.2, 2.0, 2.8, 4.0])
    design = np.column_stack((np.cos(theta), np.sin(theta), np.ones(theta.size)))
    coefficients = np.asarray([[2.0, -1.0], [-0.5, 3.0], [4.0, 7.0]])
    rates = design @ coefficients
    raw, evidence = fit_ridge_t4(rates, theta, normalized_lambda=0.0)
    assert np.allclose(raw[:, :2], coefficients[:2].T, atol=1e-6)
    assert np.allclose(raw[:, 3], coefficients[2], atol=1e-6)
    assert np.allclose(raw[:, 2], np.linalg.norm(coefficients[:2].T, axis=1), atol=1e-6)
    assert evidence["design_rank"] == 3


def test_ridge_t4_lambda_penalizes_only_direction_terms() -> None:
    from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4

    rng = np.random.default_rng(2)
    theta = np.linspace(-np.pi, np.pi, 9, endpoint=False)
    rates = rng.normal(size=(9, 5))
    ridge, _ = fit_ridge_t4(rates, theta, normalized_lambda=0.1)
    heavy, _ = fit_ridge_t4(rates, theta, normalized_lambda=1.0e6)
    assert np.allclose(heavy[:, :2], 0.0, atol=1e-6)
    assert np.allclose(heavy[:, 3], rates.mean(axis=0), atol=1e-3)
    assert np.linalg.norm(ridge[:, :2]) > np.linalg.norm(heavy[:, :2])


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_ridge_side_uses_the_variant_support_law(variant: str) -> None:
    laws = laws_for(variant)

    class _Dataset:
        pass

    dataset = _Dataset()
    n_trials, n_units = 33, 96
    rng = np.random.default_rng(9)
    angles = _synthetic_angles(9, finite=14)
    dataset.calib_trial_target_angles = {"s": angles}
    dataset.calib_trial_lengths = {"s": np.full(n_trials, 40, dtype=np.int64)}
    dataset.calib_trial_spike_sums = {
        "s": rng.normal(4.0, 1.0, size=(n_trials, n_units)).astype(np.float32)
    }
    dataset.side_feature_mean = np.zeros(4, dtype=np.float32)
    dataset.side_feature_std = np.ones(4, dtype=np.float32)
    if VARIANTS[variant]["label_budget"] == 4:
        selected = laws.select_dopt4_support(angles)
        expected_usable = 4
    else:
        selected = laws.select_m30_support(angles)
        expected_usable = int(np.isfinite(angles[:30]).sum())
    side, evidence = laws.sealed_fit_ridge_side(dataset, "s", selected)
    assert side.shape == (n_units, 4)
    assert np.isfinite(side).all()
    assert evidence["selected_indices"] == selected.tolist()
    assert evidence["normalized_lambda"] == 0.1
    assert evidence["usable_directional_trials"] == expected_usable


# ---------------------------------------------------------------------------
# Decoder adapter: shapes, contract, no-online-update (per variant)
# ---------------------------------------------------------------------------


import torch  # noqa: E402


class _IdentityTransformer(torch.nn.Module):
    def forward(self, query, source):
        return query, None


class _TinyDecoder(torch.nn.Module):
    def __init__(self, window: int, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.window_size = window
        self.fc_in = torch.nn.Linear(window, d_model)
        self.rep = torch.nn.Parameter(torch.randn(1, out_dim, window))
        self.fc_out = torch.nn.Linear(d_model, window)
        self.transformer = _IdentityTransformer()


class _StubTaskConfig:
    """FalconConfig stand-in with a synthetic channel count."""

    def __init__(self, n_channels: int) -> None:
        from falcon_challenge.config import FalconConfig, FalconTask

        self._config = FalconConfig(task=FalconTask.m2)
        self.task = FalconTask.m2
        self.n_channels = n_channels
        self.bin_size_ms = 20

    def hash_dataset(self, handle):
        return self._config.hash_dataset(handle)


_REAL_M2_TAGS = (
    "Run1_20201019",
    "Run2_20201019",
    "Run1_20201020",
    "Run2_20201020",
    "Run1_20201027",
    "Run2_20201027",
    "Run1_20201028",
    "Run1_20201030",
    "Run2_20201030",
    "Run1_20201118",
    "Run1_20201119",
    "Run1_20201124",
    "Run2_20201124",
)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Keep every test in this module off any visible GPU (no-CUDA charter)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


def _write_tiny_payload(variant: str, tmp_path: Path, **overrides) -> Path:
    import pickle

    decoder_module = decoder_module_for(variant)
    window, channels = 8, 6
    metadata = {
        "arm": decoder_module.PAYLOAD_ARM,
        "label_budget": decoder_module.PAYLOAD_LABEL_BUDGET,
        "activity_budget": decoder_module.PAYLOAD_ACTIVITY_BUDGET,
        "online_state": "cached_E[N,50]",
        "online_backward_pass": False,
    }
    metadata.update(overrides)
    payload = {
        "schema_version": "e8_t4_m2_cached_identity_v1",
        "task": _StubTaskConfig(channels).task,
        "decoder": _TinyDecoder(window, 4, 2),
        "identity_by_dataset_tag": {
            tag: np.zeros((channels, window), dtype=np.float32) for tag in _REAL_M2_TAGS
        },
        "window_size": window,
        "behavior_scaling_factor": 5.0,
        "smooth_observations": False,
        "metadata": metadata,
    }
    path = tmp_path / f"tiny_payload_{variant}.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _make_decoder(variant: str, tmp_path: Path, **payload_kwargs):
    decoder_module = decoder_module_for(variant)
    decoder_class = getattr(decoder_module, VARIANTS[variant]["decoder_class"])
    payload_path = _write_tiny_payload(variant, tmp_path, **payload_kwargs)
    return decoder_class(
        task_config=_StubTaskConfig(6), model_path=str(payload_path), batch_size=2
    )


def _calib_tag() -> str:
    return "sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb"


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_predict_requires_reset_first(variant: str, tmp_path: Path) -> None:
    decoder = _make_decoder(variant, tmp_path)
    with pytest.raises(RuntimeError):
        decoder.predict(np.zeros((1, 6), dtype=np.float32))


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_reset_and_predict_shapes(variant: str, tmp_path: Path) -> None:
    decoder = _make_decoder(variant, tmp_path)
    decoder.reset(dataset_tags=[_calib_tag(), _calib_tag()])
    output = decoder.predict(np.ones((2, 6), dtype=np.float32))
    assert output.shape == (2, 2)
    assert np.isfinite(output).all()
    small = decoder.predict(np.ones((1, 6), dtype=np.float32))
    assert small.shape == (2, 2)


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_predict_matches_the_manual_decode_law(variant: str, tmp_path: Path) -> None:
    decoder = _make_decoder(variant, tmp_path)
    decoder.reset(dataset_tags=[_calib_tag(), _calib_tag()])
    stream = np.arange(2 * 20 * 6, dtype=np.float32).reshape(20, 2, 6) / 97.0
    for step in range(19):
        decoder.predict(stream[step])
    output = decoder.predict(stream[19])

    module = decoder.decoder
    history = np.zeros((8, 2, 6), dtype=np.float32)
    for step in range(20):
        history = np.roll(history, -1, axis=0)
        history[-1] = stream[step]
    neural = torch.tensor(history.transpose(1, 0, 2), dtype=torch.float32)
    identity = torch.zeros(1, 6, 8)
    source = neural.permute(0, 2, 1) + identity
    source = module.fc_in(source)
    query = module.fc_in(module.rep).to(source)
    transformed, _ = module.transformer(query.repeat(source.shape[0], 1, 1), source)
    expected = module.fc_out(transformed).permute(0, 2, 1)[:, -1, :].detach().numpy() / 5.0
    assert np.allclose(output, expected, atol=1e-6)


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_unknown_dataset_tag_fails_closed(variant: str, tmp_path: Path) -> None:
    decoder = _make_decoder(variant, tmp_path)
    with pytest.raises(ValueError):
        decoder.reset(dataset_tags=[Path("sub-MonkeyN-held-in-calib_ses-1999-01-01-Run9_behavior+ecephys.nwb")])


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_wrong_arm_payload_is_rejected(variant: str, tmp_path: Path) -> None:
    other = "act30_full" if variant == "act30_dopt4" else "act30_dopt4"
    wrong_arms = {VARIANTS[other]["arm"], "dopt4_static_m4", "ridge_m4_activity30"}
    for arm in sorted(wrong_arms):
        with pytest.raises(ValueError):
            _make_decoder(variant, tmp_path, arm=arm)


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_wrong_budget_payload_is_rejected(variant: str, tmp_path: Path) -> None:
    other = "act30_full" if variant == "act30_dopt4" else "act30_dopt4"
    with pytest.raises(ValueError):
        _make_decoder(variant, tmp_path, label_budget=VARIANTS[other]["label_budget"])
    with pytest.raises(ValueError):
        _make_decoder(variant, tmp_path, activity_budget=4)


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_on_done_is_a_noop_and_updates_nothing(variant: str, tmp_path: Path) -> None:
    decoder = _make_decoder(variant, tmp_path)
    before = decoder.parameter_fingerprint()
    assert decoder.on_done(np.asarray([True, False])) is None
    assert decoder.on_done(np.asarray([1, 1])) is None
    assert decoder.parameter_fingerprint() == before


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_no_online_updates_across_predict_stream(variant: str, tmp_path: Path) -> None:
    decoder = _make_decoder(variant, tmp_path)
    decoder.reset(dataset_tags=[_calib_tag(), _calib_tag()])
    parameters_before = decoder.parameter_fingerprint()
    identity_before = decoder.identity_fingerprint()
    rng = np.random.default_rng(1)
    for _ in range(50):
        decoder.predict(rng.normal(size=(2, 6)).astype(np.float32))
    assert decoder.parameter_fingerprint() == parameters_before
    assert decoder.identity_fingerprint() == identity_before
    assert all(p.grad is None for p in decoder.decoder.parameters())


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_predict_never_imports_the_selection_law(variant: str, tmp_path: Path) -> None:
    import importlib.abc

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            for token in (
                "d_optimal_calibration_design",
                "calibration_budget_comparators_v1",
                "m2_t4_activity_budget_screen_v1",
                "laws",
            ):
                if token in fullname:
                    raise AssertionError(f"predict-time law import: {fullname}")
            return None

    decoder = _make_decoder(variant, tmp_path)
    decoder.reset(dataset_tags=[_calib_tag(), _calib_tag()])
    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        output = decoder.predict(np.ones((2, 6), dtype=np.float32))
    finally:
        sys.meta_path.remove(blocker)
    assert output.shape == (2, 2)


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_decoder_source_has_no_selection_law_reference(variant: str) -> None:
    source = (VARIANTS[variant]["package"] / VARIANTS[variant]["decoder_file"]).read_text(
        encoding="utf-8"
    )
    for token in (
        "greedy_forward_d_optimal",
        "d_optimal_calibration_design",
        "calib_trial_target_angles",
        "calib_trialized_neural",
        "fit_ridge_t4",
        "select_dopt4",
        "select_m30",
        "select_first30",
        "torch.optim",
        "backward(",
    ):
        assert token not in source, f"runtime references calibration-phase concept: {token}"


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_identity_fingerprint_is_content_bound(variant: str, tmp_path: Path) -> None:
    decoder = _make_decoder(variant, tmp_path)
    first = decoder.identity_fingerprint()
    tag = sorted(decoder.identity_by_dataset_tag)[0]
    decoder.identity_by_dataset_tag[tag] = decoder.identity_by_dataset_tag[tag] + 1.0
    assert decoder.identity_fingerprint() != first


# ---------------------------------------------------------------------------
# Packaging wiring (per variant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_dockerfile_binds_validated_base_payload_and_labels(variant: str) -> None:
    dockerfile = (VARIANTS[variant]["package"] / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2" in dockerfile
    assert (
        f"COPY artifacts/{VARIANTS[variant]['payload_name']} /data/decoder.pkl" in dockerfile
    )
    assert f"COPY {VARIANTS[variant]['decoder_file']} /{VARIANTS[variant]['decoder_file']}" in dockerfile
    assert "COPY decode.py /decode.py" in dockerfile
    assert (
        "ARG CHECKPOINT_SHA256=25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
        in dockerfile
    )
    assert 'LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"' in dockerfile
    assert f'LABEL ai.eval.label_budget="{VARIANTS[variant]["label_budget"]}"' in dockerfile
    assert 'LABEL ai.eval.activity_budget="30"' in dockerfile
    assert "ENV BATCH_SIZE=7" in dockerfile


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_dockerfile_method_label_matches_the_submission_helper(variant: str) -> None:
    """The v1 packaging lost a day to exactly this drift; pin it."""
    dockerfile = (VARIANTS[variant]["package"] / "Dockerfile").read_text(encoding="utf-8")
    line = next(
        l for l in dockerfile.splitlines() if l.startswith("LABEL ai.eval.method=")
    )
    label = line.split("=", 1)[1].strip().strip('"')
    assert label == helper_for(variant).CANDIDATE["method_label"]


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_decode_entrypoint_defaults(variant: str) -> None:
    source = (VARIANTS[variant]["package"] / "decode.py").read_text(encoding="utf-8")
    assert 'default="/data/decoder.pkl"' in source
    assert '"--batch-size", type=int, default=7' in source
    assert VARIANTS[variant]["decoder_class"] in source


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_exporter_dry_cli_is_torch_free_and_writes_nothing(variant: str, tmp_path: Path) -> None:
    script = VARIANTS[variant]["package"] / VARIANTS[variant]["exporter"]
    output = tmp_path / "candidate.pkl"
    probe = (
        "import runpy,sys; "
        f"sys.argv=['export','--output',{str(output)!r}]; "
        f"runpy.run_path({str(script)!r},run_name='__main__'); "
        "assert 'torch' not in sys.modules"
    )
    import os

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=VARIANTS[variant]["package"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE_NO_SUBMISSION"
    assert payload["arm"] == VARIANTS[variant]["arm"]
    assert payload["label_budget"] == VARIANTS[variant]["label_budget"]
    assert payload["activity_budget"] == 30
    assert payload["normalized_lambda"] == 0.1
    assert not output.exists()


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_submission_helper_freezes_the_built_candidate(variant: str) -> None:
    module = helper_for(variant)
    report = module.plan_report()
    spec = VARIANTS[variant]
    assert report["arm"] == spec["arm"]
    assert report["image_tag"] == spec["image_tag"]
    assert report["image_id"] == spec["image_id"]
    assert report["payload_sha256"] == spec["payload_sha256"]
    assert report["checkpoint_sha256"] == (
        "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
    )
    assert (report["label_budget"], report["activity_budget"]) == (
        spec["label_budget"],
        30,
    )
    assert report["private"] is True
    assert report["phase_slug"] == "few-shot-test-2319" and report["challenge_id"] == 2319
    attributes = {item["name"]: item["value"] for item in report["submission_attributes"]}
    assert attributes == {
        "IsHeldOutZeroShot": False,
        "IsTestTimeAdaptive": False,
        "IsPretrained": False,
    }
    assert not (spec["package"] / "artifacts").joinpath(
        f"evalai_push_state_{'dopt4_act30' if variant == 'act30_dopt4' else 'm30_act30'}_v1.json"
    ).exists()


def test_variant_table_is_not_cross_wired() -> None:
    a, b = VARIANTS["act30_dopt4"], VARIANTS["act30_full"]
    assert a["arm"] != b["arm"]
    assert a["sealed_cell"] != b["sealed_cell"]
    assert abs(a["anchor_external"] - b["anchor_external"]) > 1e-3
    assert a["label_budget"] != b["label_budget"]
    assert a["package"] != b["package"]


# ---------------------------------------------------------------------------
# Receipts (require the completed export + terminal stages)
# ---------------------------------------------------------------------------


def _sealed_rows(cell: str) -> dict[tuple[str, str], dict]:
    payload = json.loads(SEALED_SCREEN.read_text(encoding="utf-8"))
    rows = {
        (row["surface"], row["session"]): row
        for row in payload["rows"]
        if row["cell"] == cell
    }
    assert len(rows) == 13
    return rows


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_payload_receipt_reproduces_the_sealed_rows_bitwise(variant: str) -> None:
    spec = VARIANTS[variant]
    from pathlib import PurePosixPath

    receipt_name = PurePosixPath(str(spec["payload_name"])).with_suffix(".receipt.json")
    receipt = json.loads(
        (spec["package"] / "artifacts" / receipt_name).read_text(encoding="utf-8")
    )
    assert receipt["arm"] == spec["arm"]
    assert receipt["label_budget"] == spec["label_budget"]
    assert receipt["activity_budget"] == 30
    assert receipt["session_count"] == 13
    assert receipt["max_direct_vs_cached_identity_abs"] == 0.0
    assert receipt["max_direct_vs_decoder_only_abs"] == 0.0
    rows = _sealed_rows(spec["sealed_cell"])
    assert set(receipt["session_records"]) == {session for _, session in rows}
    for session, record in receipt["session_records"].items():
        row = rows[(record["surface"], session)]
        assert record["selected_indices"] == row["side_evidence"]["selected_indices"]
        assert (
            record["side_evidence"]["activity_sha256_screen_law"] == row["activity_sha256"]
        ), f"{session}: activity pool bytes drifted from the sealed screen"
        assert (
            record["side_evidence"]["normalized_t4_sha256"]
            == row["side_evidence"]["normalized_t4_sha256"]
        ), f"{session}: ridge T4 bytes drifted from the sealed screen"
        assert record["activity_budget"] == 30
    # Continuity with the previously deployed official image of the same law.
    assert receipt["prior_deployed_identity_sha256_matches"] in (0, 13)


@pytest.mark.parametrize("variant", ("act30_dopt4", "act30_full"))
def test_terminal_receipt_binds_the_package(variant: str) -> None:
    spec = VARIANTS[variant]
    receipt = json.loads(
        (spec["result_root"] / "terminal.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "TERMINAL_PACKAGED_NOT_SUBMITTED"
    assert receipt["docker"]["pushed"] is False
    assert receipt["submission"]["performed_by_this_cell"] is False
    assert receipt["docker"]["image_id"] == spec["image_id"]
    assert receipt["docker"]["tag"] == spec["image_tag"]
    assert receipt["artifact_tree"]["payload"]["sha256"] == spec["payload_sha256"]
    assert receipt["calibration_law_provenance"]["activity_pool"].endswith(
        "mirrored verbatim and cross-checked against the sealed import"
    )
    external = receipt["local_reference_scores"]["external_official_query"]
    assert abs(external["equal_session_mean"] - spec["anchor_external"]) < 1e-2
    within = receipt["local_reference_scores"]["within_post30"]
    assert abs(within["equal_session_mean"] - spec["anchor_within"]) < 1e-2
    contract = receipt["contract_compliance"]
    for key in (
        "predict_time_selection_law_imports_blocked",
        "weights_unchanged_across_replay",
        "identities_unchanged_across_replay",
        "on_done_is_noop",
        "no_gradient_state_after_replay",
        "no_optimizer_or_calibration_state_on_decoder",
    ):
        assert contract[key] is True
    assert contract["trial_metadata_available_at_predict_time"] is False
    proof = receipt["evaluator_semantics_proof"]
    assert proof["on_done_calls"] == 0 and proof["on_done_never_called_by_official_loop"] is True
    container = receipt["container_validation"]
    assert container["entered_expected_300s_wait"] is True
    # the seven minival sessions plus the evaluator's normalized_latency channel
    assert set(container["session_keys"]) == {
        "Run1_20201019",
        "Run2_20201019",
        "Run1_20201020",
        "Run2_20201020",
        "Run1_20201027",
        "Run2_20201027",
        "Run1_20201028",
        "normalized_latency",
    }
    assert container["task_key"] == ["m2"]
