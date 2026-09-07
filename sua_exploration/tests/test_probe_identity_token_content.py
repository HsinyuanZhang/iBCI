"""Synthetic, CPU-only contract tests for the root-audited A4-v2 probe."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import identity_token_content_probe as probe


REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOW_SIZE = 24
N_UNITS = 32
SEED = 20260812
SESSIONS = [f"sub-C_ses-CO-201511{day:02d}" for day in (3, 4, 6, 9, 10, 12)]


def _load_aggregate_module():
    script = REPO_ROOT / "sua_exploration/scripts/aggregate_identity_token_content_probe.py"
    spec = importlib.util.spec_from_file_location("aggregate_identity_token_content_probe", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_module():
    script = REPO_ROOT / "sua_exploration/scripts/probe_identity_token_content.py"
    spec = importlib.util.spec_from_file_location("probe_identity_token_content", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_embedded_integrity_receipt(path: Path, receipt: dict) -> None:
    body = dict(receipt)
    body["receipt_body_sha256"] = probe.sha256_bytes(probe.canonical_json_bytes(body))
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")


def _hyperparameters() -> probe.ProbeHyperparameters:
    return probe.ProbeHyperparameters(random_seed=SEED, min_session_units=15)


def _session_carriers() -> dict[str, np.ndarray]:
    return {
        session: probe.synthetic_carrier(N_UNITS, np.random.default_rng(SEED + index))
        for index, session in enumerate(SESSIONS)
    }


def _identities(carriers: dict[str, np.ndarray], *, kind: str) -> dict[str, np.ndarray]:
    factory = {
        "phase": probe.synthetic_phase_carrying_identity,
        "activity": probe.synthetic_activity_only_identity,
    }[kind]
    return {
        session: factory(
            carrier,
            window_size=WINDOW_SIZE,
            rng=np.random.default_rng(SEED + 100 + index),
        )
        for index, (session, carrier) in enumerate(carriers.items())
    }


def _fit(kind: str) -> dict:
    carriers = _session_carriers()
    return probe.fit_session_loso_probe_suite(
        _identities(carriers, kind=kind),
        carriers,
        sessions=SESSIONS,
        hyperparameters=_hyperparameters(),
    )


def test_session_loso_positive_control_recovers_cross_session_phase() -> None:
    result = _fit("phase")
    assert result["sessions"] == SESSIONS
    assert set(result["per_session"]) == set(SESSIONS)
    assert result["pooled"]["probes"]["phase_unit"]["mean_cosine"] >= 0.80
    assert result["pooled"]["probes"]["b"]["r2"] >= 0.80
    assert (
        result["pooled"][probe.PRIMARY_NULL_NAME]["phase_unit"]["mean_cosine"]
        <= 0.25
    )
    for held_out, fold in result["per_session"].items():
        assert fold["held_out_session"] == held_out
        assert held_out not in fold["training_sessions"]
        assert len(fold["training_sessions"]) == 5
        assert fold["n_held_out_units"] == N_UNITS


def test_session_loso_activity_only_control_has_no_global_phase_coordinate() -> None:
    result = _fit("activity")
    assert result["pooled"]["probes"]["b"]["r2"] >= 0.50
    assert abs(result["pooled"]["probes"]["phase_unit"]["mean_cosine"]) <= 0.30


def test_pairing_permutation_manifest_is_deterministic_and_rejects_target_row_drift() -> None:
    carriers = _session_carriers()
    first = probe.build_pairing_permutation_manifest(carriers, sessions=SESSIONS, random_seed=SEED)
    second = probe.build_pairing_permutation_manifest(carriers, sessions=SESSIONS, random_seed=SEED)
    assert first == second
    result = probe.fit_session_loso_probe_suite(
        _identities(carriers, kind="phase"),
        carriers,
        sessions=SESSIONS,
        hyperparameters=_hyperparameters(),
        pairing_permutation_manifest=first,
    )
    assert result["pairing_permutation"] == first
    altered = {session: values.copy() for session, values in carriers.items()}
    altered[SESSIONS[0]][0, 0] += 0.25
    with pytest.raises(ValueError, match="pairing-permutation manifest drifted"):
        probe.fit_session_loso_probe_suite(
            _identities(altered, kind="phase"),
            altered,
            sessions=SESSIONS,
            hyperparameters=_hyperparameters(),
            pairing_permutation_manifest=first,
        )


def test_sealed_session_guard_raises_before_any_fit() -> None:
    carriers = _session_carriers()
    identities = _identities(carriers, kind="phase")
    sealed = list(SESSIONS)
    sealed[-1] = probe.SEALED_TEST_SESSIONS[0]
    carriers[sealed[-1]] = carriers.pop(SESSIONS[-1])
    identities[sealed[-1]] = identities.pop(SESSIONS[-1])
    with pytest.raises(ValueError, match="refusing sealed"):
        probe.fit_session_loso_probe_suite(
            identities,
            carriers,
            sessions=sealed,
            hyperparameters=_hyperparameters(),
        )


def test_refuses_insufficient_units_in_any_held_out_session() -> None:
    carriers = _session_carriers()
    identities = _identities(carriers, kind="phase")
    carriers[SESSIONS[0]] = carriers[SESSIONS[0]][:14]
    identities[SESSIONS[0]] = identities[SESSIONS[0]][:14]
    with pytest.raises(ValueError, match="N=14 < required 15"):
        probe.fit_session_loso_probe_suite(
            identities,
            carriers,
            sessions=SESSIONS,
            hyperparameters=_hyperparameters(),
        )


def test_fixed_epoch_window_averages_each_held_out_session_not_an_argmax() -> None:
    base = _fit("phase")
    epoch_results = {}
    for epoch, offset in ((5, 0.0), (6, 0.2)):
        row = json.loads(json.dumps(base))
        for session in SESSIONS:
            row["per_session"][session]["probes"]["phase_unit"]["mean_cosine"] += offset
            row["per_session"][session]["pairing_permutation_null"]["phase_unit"]["mean_cosine"] += offset / 2.0
        # The source-level pooled metrics are intentionally not used by the
        # epoch aggregator; recompute them from its retained per-session rows.
        row["pooled"] = probe._pool_per_session_scores(row["per_session"])
        epoch_results[epoch] = row
    aggregate = probe.aggregate_fixed_epoch_window(epoch_results, sessions=SESSIONS)
    expected = (
        epoch_results[5]["per_session"][SESSIONS[0]]["probes"]["phase_unit"]["mean_cosine"]
        + epoch_results[6]["per_session"][SESSIONS[0]]["probes"]["phase_unit"]["mean_cosine"]
    ) / 2.0
    assert aggregate["per_session"][SESSIONS[0]]["probes"]["phase_unit"]["mean_cosine"] == pytest.approx(expected)
    assert aggregate["epoch_window"] == [5, 6]
    assert aggregate["pairing_permutation"] == base["pairing_permutation"]


def test_paired_v2_read_rule_passes_synthetic_carrier_vs_activity_pair() -> None:
    carrier = _fit("phase")
    activity = _fit("activity")
    verdict = probe.evaluate_paired_gate(
        activity["pooled"],
        carrier["pooled"],
        activity_only_per_session=activity["per_session"],
        carrier_per_session=carrier["per_session"],
    )
    assert verdict["cross_session_phase_content_gate_pass"] is True
    assert verdict["positive_held_out_sessions"] >= 5
    assert verdict["mean_delta_phase"] >= 0.10
    assert verdict["ac4_advantage_over_pairing_permutation"] >= 0.10


def test_canonical_hash_excludes_only_requested_keys() -> None:
    first = {"created_at": "a", "value": 3}
    second = {"created_at": "b", "value": 3}
    assert probe.sha256_bytes(probe.canonical_json_bytes(first, exclude_keys=("created_at",))) == probe.sha256_bytes(
        probe.canonical_json_bytes(second, exclude_keys=("created_at",))
    )


def test_mean_normalized_ridge_is_invariant_to_training_row_duplication() -> None:
    rng = np.random.default_rng(20260812)
    features = rng.normal(size=(17, 5))
    target = rng.normal(size=(17, 2))
    standardized, _ignored, _mean, _scale = probe._standardize_train_test(features, features)
    intercept, weights = probe._fit_ridge_coefficients(
        standardized,
        target,
        normalized_lambda=1.0,
    )
    duplicated_intercept, duplicated_weights = probe._fit_ridge_coefficients(
        np.concatenate([standardized, standardized], axis=0),
        np.concatenate([target, target], axis=0),
        normalized_lambda=1.0,
    )
    assert np.allclose(duplicated_intercept, intercept, rtol=1e-12, atol=1e-12)
    assert np.allclose(duplicated_weights, weights, rtol=1e-12, atol=1e-12)


def test_activity_calibration_contract_uses_m_t_n_not_the_time_axis_as_m() -> None:
    runner = _load_runner_module()
    real_layout = np.zeros((30, 100, 38), dtype=np.float32)
    assert runner._validate_activity_calibration_shape(
        real_layout,
        session="sub-C_ses-CO-20151103",
        n_units=38,
    ) == (30, 100, 38)

    for malformed in (
        np.zeros((38, 30, 100), dtype=np.float32),
        np.zeros((30, 99, 38), dtype=np.float32),
        np.zeros((30, 100, 37), dtype=np.float32),
    ):
        with pytest.raises(ValueError, match=r"activity calibration must be \[M,T,N\]"):
            runner._validate_activity_calibration_shape(
                malformed,
                session="sub-C_ses-CO-20151103",
                n_units=38,
            )


def test_runner_rejects_frozen_ridge_or_null_hyperparameter_drift_before_loading_inputs() -> None:
    runner = _load_runner_module()
    with pytest.raises(ValueError, match="frozen probe hyperparameters drift"):
        runner.run_paired_v10_preflight(
            ac4_checkpoint=Path("/nonexistent/ac4.ckpt"),
            z4_checkpoint=Path("/nonexistent/z4.ckpt"),
            cache_dir=None,
            hyperparameters=probe.ProbeHyperparameters(
                normalized_ridge_lambda=2.0,
                random_seed=43,
            ),
        )


def test_aggregator_rejects_frozen_hyperparameter_drift_before_checkpoint_reads(tmp_path: Path) -> None:
    aggregate = _load_aggregate_module()
    receipt = {
        "schema_version": probe.SCHEMA_VERSION,
        "probe_name": probe.PROBE_NAME,
        "sealed_test_sessions_opened": False,
        "execution_scope": {
            "cpu_only": True,
            "cuda_visible_devices": "",
            "torch_cuda_available": False,
            "no_training_no_backward": True,
            "opened_nwb_sessions": SESSIONS,
            "opened_nwb_session_count": 6,
        },
        "sessions": SESSIONS,
        "protocol": dict(aggregate.REQUIRED_PROTOCOL),
        "probe_hyperparameters": {
            **aggregate.FROZEN_PROBE_HYPERPARAMETERS,
            "normalized_ridge_lambda": 2.0,
            "random_seed": 43,
        },
    }
    path = tmp_path / "hyperparameter_drift.json"
    _write_embedded_integrity_receipt(path, receipt)
    with pytest.raises(ValueError, match="frozen probe hyperparameters drift"):
        aggregate.validate_receipt(path)


def test_receipt_writer_uses_exclusive_immutable_receipt_and_sha_sidecar(tmp_path: Path) -> None:
    runner = _load_runner_module()
    path = tmp_path / "receipt.json"
    runner.write_receipt({"example": "a4"}, path)
    sidecar = path.with_name(path.name + ".sha256")
    assert (path.stat().st_mode & 0o777) == 0o444
    assert (sidecar.stat().st_mode & 0o777) == 0o444
    assert sidecar.read_text(encoding="ascii") == f"{runner.sha256_file(path)}  {path.name}\n"
    with pytest.raises(FileExistsError, match="refusing to overwrite existing A4 receipt"):
        runner.write_receipt({"example": "a4"}, path)


def test_aggregate_writer_uses_exclusive_immutable_output_and_sha_sidecar(tmp_path: Path) -> None:
    aggregate = _load_aggregate_module()
    path = tmp_path / "aggregate.json"
    aggregate.write_aggregate({"example": "a4 aggregate"}, path)
    sidecar = path.with_name(path.name + ".sha256")
    assert (path.stat().st_mode & 0o777) == 0o444
    assert (sidecar.stat().st_mode & 0o777) == 0o444
    assert sidecar.read_text(encoding="ascii") == f"{aggregate._sha256_file(path)}  {path.name}\n"
    with pytest.raises(FileExistsError, match="refusing to overwrite existing A4 aggregate"):
        aggregate.write_aggregate({"example": "changed"}, path)

    # A filesystem-level modification must be detectable from the publication
    # sidecar even though the normal writer leaves the file read-only.
    path.chmod(0o644)
    path.write_text('{"example":"tampered"}\n', encoding="utf-8")
    assert sidecar.read_text(encoding="ascii") != f"{aggregate._sha256_file(path)}  {path.name}\n"


def test_aggregator_rejects_receipt_integrity_drift_before_schema_checks(tmp_path: Path) -> None:
    aggregate = _load_aggregate_module()
    path = tmp_path / "tampered.json"
    path.write_text(
        json.dumps({"receipt_body_sha256": "0" * 64, "schema_version": probe.SCHEMA_VERSION}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical body SHA drift"):
        aggregate.validate_receipt(path)


def test_aggregator_accepts_verified_sha_sidecar_as_an_integrity_binding(tmp_path: Path) -> None:
    aggregate = _load_aggregate_module()
    runner = _load_runner_module()
    path = tmp_path / "sidecar_only.json"
    path.write_text(json.dumps({"schema_version": 0}), encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        f"{runner.sha256_file(path)}  {path.name}\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="schema version drift"):
        aggregate.validate_receipt(path)


def test_aggregator_rejects_sealed_or_unpaired_input_before_gate(tmp_path: Path) -> None:
    aggregate = _load_aggregate_module()
    receipt = {
        "schema_version": probe.SCHEMA_VERSION,
        "probe_name": probe.PROBE_NAME,
        "sealed_test_sessions_opened": False,
        "execution_scope": {
            "cpu_only": True,
            "cuda_visible_devices": "",
            "torch_cuda_available": False,
            "no_training_no_backward": True,
            "opened_nwb_sessions": [probe.SEALED_TEST_SESSIONS[0]] * 6,
            "opened_nwb_session_count": 6,
        },
        "sessions": [probe.SEALED_TEST_SESSIONS[0]] * 6,
        "protocol": {},
    }
    path = tmp_path / "sealed.json"
    _write_embedded_integrity_receipt(path, receipt)
    with pytest.raises(ValueError, match="six unique development sessions"):
        aggregate.validate_receipt(path)
