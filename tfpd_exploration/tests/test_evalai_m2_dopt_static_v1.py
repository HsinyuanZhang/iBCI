"""No-data/no-CUDA unit tests for the D-opt-4 static M2 EvalAI package.

Covers: the frozen greedy D-opt selection law on synthetic calibration sets,
the ridge lambda=0.1 T4 fit, the BCIDecoder adapter shapes and the
zero-online-update contract, and the packaging wiring (Dockerfile/decode).
Only synthetic tensors are used; no checkpoint, no NWB data, no CUDA.
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
PACKAGE_DIR = REPO_ROOT / "tfpd_exploration/submissions/evalai_m2_dopt_static_v1"
for _path in (REPO_ROOT, REPO_ROOT / "SPINT-main", PACKAGE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import laws  # noqa: E402
from laws import LawError  # noqa: E402


def _synthetic_angles(seed: int, n_pool: int = 30, finite: int = 12, total: int = 35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.full(total, np.nan, dtype=np.float64)
    positions = rng.permutation(n_pool)[:finite]
    angles[positions] = (
        -3.0 * np.pi / 4.0 + rng.integers(0, 8, finite) * (np.pi / 4.0)
        + rng.normal(0.0, 0.05, finite)
    )
    return angles


# ---------------------------------------------------------------------------
# D-opt selection law
# ---------------------------------------------------------------------------


def test_selection_is_four_first30_finite_candidates() -> None:
    selected = laws.select_dopt4_support(_synthetic_angles(3))
    assert selected.shape == (4,)
    assert selected.dtype == np.int64
    assert int(selected.max()) < 30
    assert np.unique(selected).size == 4


def test_selection_is_sorted_absolute_indices() -> None:
    angles = np.full(30, np.nan)
    angles[[2, 7, 11, 29]] = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    assert laws.select_dopt4_support(angles).tolist() == [2, 7, 11, 29]


def test_selection_is_deterministic() -> None:
    angles = _synthetic_angles(11)
    first = laws.select_dopt4_support(angles)
    second = laws.select_dopt4_support(angles)
    assert np.array_equal(first, second)


def test_selection_is_not_the_chronological_alias() -> None:
    angles = np.full(35, np.nan)
    angles[[1, 5, 11, 18, 24, 29]] = [0.0, np.pi / 3, 2 * np.pi / 3, np.pi, -2 * np.pi / 3, -np.pi / 3]
    selected = laws.select_dopt4_support(angles)
    assert selected.tolist() != [0, 1, 2, 3]


def test_selection_fails_closed_below_four_candidates() -> None:
    angles = np.full(35, np.nan)
    angles[[3, 9, 21]] = [0.0, np.pi / 2, np.pi]
    with pytest.raises(LawError):
        laws.select_dopt4_support(angles)


def test_selection_fails_closed_without_first30_metadata() -> None:
    with pytest.raises(LawError):
        laws.select_dopt4_support(np.full(29, np.nan))


def test_greedy_beats_chronological_prefix_on_design_determinant() -> None:
    rng = np.random.default_rng(5)
    # A fully finite pool makes the chronological-prefix comparator defined.
    angles = -3.0 * np.pi / 4.0 + rng.integers(0, 8, 30) * (np.pi / 4.0) + rng.normal(0.0, 0.05, 30)
    selected = laws.select_dopt4_support(angles)
    design_d = laws.design_matrix_from_thetas(angles[selected])
    design_c = laws.design_matrix_from_thetas(angles[[0, 1, 2, 3]])
    gram_d = design_d.T @ design_d
    gram_c = design_c.T @ design_c
    assert np.linalg.det(gram_d) >= np.linalg.det(gram_c)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Keep every test in this module off any visible GPU (no-CUDA charter)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


def test_mirror_matches_sealed_screen_law() -> None:
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical import _support_indices

    for seed in (0, 1, 2, 3, 4):
        angles = _synthetic_angles(seed)

        class _Shim:
            pass

        shim = _Shim()
        shim.calib_trial_target_angles = {"s": angles}
        sealed = _support_indices(shim, "s", 4)
        mirror = laws.select_dopt4_support(angles)
        assert np.array_equal(mirror, sealed), f"seed {seed}: {mirror} vs {sealed}"


def test_greedy_core_matches_sealed_doptimal_module() -> None:
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices as sealed_greedy,
    )

    thetas = _synthetic_angles(7)[np.isfinite(_synthetic_angles(7))]
    assert np.array_equal(
        laws.greedy_forward_d_optimal_indices(thetas, 4),
        sealed_greedy(thetas, 4),
    )


# ---------------------------------------------------------------------------
# Ridge T4 law (sealed import exercised through the package)
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
    # With a large lambda the direction terms shrink toward zero while the
    # intercept converges to the mean rate (unpenalized intercept).
    heavy, _ = fit_ridge_t4(rates, theta, normalized_lambda=1.0e6)
    assert np.allclose(heavy[:, :2], 0.0, atol=1e-6)
    assert np.allclose(heavy[:, 3], rates.mean(axis=0), atol=1e-3)
    assert np.linalg.norm(ridge[:, :2]) > np.linalg.norm(heavy[:, :2])


def test_static_ridge_side_uses_only_selected_support() -> None:
    class _Dataset:
        pass

    dataset = _Dataset()
    n_trials, n_units = 30, 96  # the sealed ridge side law hard-requires 96 channels
    rng = np.random.default_rng(9)
    dataset.calib_trial_target_angles = {"s": _synthetic_angles(9, finite=14)}
    dataset.calib_trial_lengths = {"s": np.full(n_trials, 40, dtype=np.int64)}
    dataset.calib_trial_spike_sums = {
        "s": rng.normal(4.0, 1.0, size=(n_trials, n_units)).astype(np.float32)
    }
    dataset.side_feature_mean = np.zeros(4, dtype=np.float32)
    dataset.side_feature_std = np.ones(4, dtype=np.float32)
    selected = laws.select_dopt4_support(dataset.calib_trial_target_angles["s"])
    side, evidence = laws.sealed_fit_ridge_side(dataset, "s", selected)
    assert side.shape == (n_units, 4)
    assert np.isfinite(side).all()
    assert evidence["selected_indices"] == selected.tolist()
    assert evidence["normalized_lambda"] == 0.1
    assert evidence["usable_directional_trials"] == 4


# ---------------------------------------------------------------------------
# Decoder adapter: shapes, contract, no-online-update
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


def _build_tiny_decoder(window: int, d_model: int, n_channels: int, out_dim: int):
    """Minimal stand-in matching the frozen decoder's interface and shapes.

    Mirrors ``src.models.components.spint.SpintModel`` (coupled mode):
    ``rep`` is ``[1, C, W]`` (C covariate queries), ``fc_in`` is
    ``Linear(W, H)``, ``fc_out`` is ``Linear(H, W)``, and the runtime's
    ``permute(0, 2, 1)[:, -1, :]`` therefore returns ``[B, C]``.
    """
    return _TinyDecoder(window, d_model, out_dim)


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


def _write_tiny_payload(tmp_path: Path, *, arm: str = "dopt4_static_m4") -> Path:
    import pickle

    window, channels = 8, 6
    payload = {
        "schema_version": "e8_t4_m2_cached_identity_v1",
        "task": _StubTaskConfig(channels).task,
        "decoder": _build_tiny_decoder(window, 4, channels, 2),
        "identity_by_dataset_tag": {
            tag: np.zeros((channels, window), dtype=np.float32) for tag in _REAL_M2_TAGS
        },
        "window_size": window,
        "behavior_scaling_factor": 5.0,
        "smooth_observations": False,
        "metadata": {
            "arm": arm,
            "label_budget": 4,
            "activity_budget": 4,
            "online_state": "cached_E[N,50]",
            "online_backward_pass": False,
        },
    }
    path = tmp_path / "tiny_payload.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _make_decoder(tmp_path: Path, **payload_kwargs):
    from dopt_static_decoder import DoptStaticM2Decoder

    payload_path = _write_tiny_payload(tmp_path, **payload_kwargs)
    return DoptStaticM2Decoder(
        task_config=_StubTaskConfig(6), model_path=str(payload_path), batch_size=2
    )


def _calib_tag() -> str:
    return "sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb"


def test_predict_requires_reset_first(tmp_path: Path) -> None:
    decoder = _make_decoder(tmp_path)
    with pytest.raises(RuntimeError):
        decoder.predict(np.zeros((1, 6), dtype=np.float32))


def test_reset_and_predict_shapes(tmp_path: Path) -> None:
    decoder = _make_decoder(tmp_path)
    decoder.reset(dataset_tags=[_calib_tag(), _calib_tag()])
    output = decoder.predict(np.ones((2, 6), dtype=np.float32))
    assert output.shape == (2, 2)
    assert np.isfinite(output).all()
    # batch smaller than configured is zero-padded, not rejected
    small = decoder.predict(np.ones((1, 6), dtype=np.float32))
    assert small.shape == (2, 2)


def test_predict_matches_the_manual_decode_law(tmp_path: Path) -> None:
    import torch

    decoder = _make_decoder(tmp_path)
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


def test_unknown_dataset_tag_fails_closed(tmp_path: Path) -> None:
    decoder = _make_decoder(tmp_path)
    with pytest.raises(ValueError):
        decoder.reset(dataset_tags=[Path("sub-MonkeyN-held-in-calib_ses-1999-01-01-Run9_behavior+ecephys.nwb")])


def test_wrong_arm_payload_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _make_decoder(tmp_path, arm="ridge_m4_activity30")


def test_on_done_is_a_noop_and_updates_nothing(tmp_path: Path) -> None:
    decoder = _make_decoder(tmp_path)
    before = decoder.parameter_fingerprint()
    assert decoder.on_done(np.asarray([True, False])) is None
    assert decoder.on_done(np.asarray([1, 1])) is None
    assert decoder.parameter_fingerprint() == before


def test_no_online_updates_across_predict_stream(tmp_path: Path) -> None:
    decoder = _make_decoder(tmp_path)
    decoder.reset(dataset_tags=[_calib_tag(), _calib_tag()])
    parameters_before = decoder.parameter_fingerprint()
    identity_before = decoder.identity_fingerprint()
    rng = np.random.default_rng(1)
    for _ in range(50):
        decoder.predict(rng.normal(size=(2, 6)).astype(np.float32))
    assert decoder.parameter_fingerprint() == parameters_before
    assert decoder.identity_fingerprint() == identity_before
    assert all(p.grad is None for p in decoder.decoder.parameters())


def test_predict_never_imports_the_selection_law(tmp_path: Path) -> None:
    import importlib.abc

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            for token in ("d_optimal_calibration_design", "calibration_budget_comparators_v1", "m2_t4_activity_budget_screen_v1", "laws"):
                if token in fullname:
                    raise AssertionError(f"predict-time law import: {fullname}")
            return None

    decoder = _make_decoder(tmp_path)
    decoder.reset(dataset_tags=[_calib_tag(), _calib_tag()])
    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        output = decoder.predict(np.ones((2, 6), dtype=np.float32))
    finally:
        sys.meta_path.remove(blocker)
    assert output.shape == (2, 2)


def test_decoder_source_has_no_selection_law_reference() -> None:
    source = (PACKAGE_DIR / "dopt_static_decoder.py").read_text(encoding="utf-8")
    for token in (
        "greedy_forward_d_optimal",
        "d_optimal_calibration_design",
        "calib_trial_target_angles",
        "calib_trialized_neural",
        "fit_ridge_t4",
        "select_dopt4",
        "torch.optim",
        "backward(",
    ):
        assert token not in source, f"runtime references calibration-phase concept: {token}"


def test_identity_fingerprint_is_content_bound(tmp_path: Path) -> None:
    decoder = _make_decoder(tmp_path)
    first = decoder.identity_fingerprint()
    tag = sorted(decoder.identity_by_dataset_tag)[0]
    decoder.identity_by_dataset_tag[tag] = decoder.identity_by_dataset_tag[tag] + 1.0
    assert decoder.identity_fingerprint() != first


# ---------------------------------------------------------------------------
# Packaging wiring
# ---------------------------------------------------------------------------


def test_dockerfile_binds_validated_base_and_payload() -> None:
    dockerfile = (PACKAGE_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE=spint-m2:e8-epoch027-76f0fb2" in dockerfile
    assert "COPY artifacts/t4_m2_seed42_dopt4_static_identity.pkl /data/decoder.pkl" in dockerfile
    assert "COPY dopt_static_decoder.py /dopt_static_decoder.py" in dockerfile
    assert "COPY decode.py /decode.py" in dockerfile
    assert (
        "ARG CHECKPOINT_SHA256=25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
        in dockerfile
    )
    assert 'LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"' in dockerfile
    assert 'LABEL ai.eval.label_budget="4"' in dockerfile
    assert "ENV BATCH_SIZE=7" in dockerfile


def test_decode_entrypoint_defaults() -> None:
    source = (PACKAGE_DIR / "decode.py").read_text(encoding="utf-8")
    assert 'default="/data/decoder.pkl"' in source
    assert '"--batch-size", type=int, default=7' in source


def test_exporter_dry_cli_is_torch_free_and_writes_nothing(tmp_path: Path) -> None:
    script = PACKAGE_DIR / "export_dopt_static_payload.py"
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
        cwd=PACKAGE_DIR,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE_NO_SUBMISSION"
    assert payload["label_budget"] == 4 and payload["activity_budget"] == 4
    assert payload["normalized_lambda"] == 0.1
    assert not output.exists()


def test_submission_helper_freezes_the_built_candidate() -> None:
    spec = importlib.util.spec_from_file_location(
        "submit_evalai_dopt_static", PACKAGE_DIR / "submit_evalai_dopt_static.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.plan_report()
    assert report["image_tag"] == "spint-t4-m2:dopt4-static-s42-4f68b7b8"
    assert report["image_id"] == (
        "sha256:280f295ea6a46b681295d8c738938972010f067ed455dd0440f9052061a39c9f"
    )
    assert report["payload_sha256"] == (
        "4f68b7b8d0a64c66b53810dd73a125b9021b0b029dc57f00bb888b783151ecf4"
    )
    assert report["checkpoint_sha256"] == (
        "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
    )
    assert (report["label_budget"], report["activity_budget"]) == (4, 4)
    assert report["private"] is True
    assert report["phase_slug"] == "few-shot-test-2319" and report["challenge_id"] == 2319
    attributes = {item["name"]: item["value"] for item in report["submission_attributes"]}
    assert attributes == {
        "IsHeldOutZeroShot": False,
        "IsTestTimeAdaptive": False,
        "IsPretrained": False,
    }
    assert not (PACKAGE_DIR / "artifacts" / "evalai_push_state_dopt4_static_v1.json").exists()


def test_terminal_receipt_binds_the_package() -> None:
    receipt = json.loads(
        (REPO_ROOT / "tfpd_exploration/results/evalai_m2_dopt_static_v1/terminal.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "TERMINAL_PACKAGED_NOT_SUBMITTED"
    assert receipt["docker"]["pushed"] is False
    assert receipt["submission"]["performed_by_this_cell"] is False
    assert receipt["docker"]["image_id"] == (
        "sha256:280f295ea6a46b681295d8c738938972010f067ed455dd0440f9052061a39c9f"
    )
    external = receipt["local_reference_scores"]["external_official_query"]
    assert abs(external["equal_session_mean"] - 0.22271999429945652) < 1e-6
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
