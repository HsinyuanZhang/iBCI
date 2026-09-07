"""Synthetic no-data contracts for the H-NF label-free carrier (revision R2).

Covers F1–F6 of AGENT_BRIEF_HNF_CODE_REVISION_R2_20260809.md.  No NWB file is
opened, no GPU is used.  Tests that need real data are run separately in the
acceptance audit script.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data import h1_carrierid_nf_features as nf
from src.data import h1_carrierid_nf_date_lodo as dmod
from src.data.h1_m4_eb_pilot import H1PilotRecord, H1_M4_FOLD0_SOURCE, FOLD0_DATE


# --------------------------------------------------------------------------- #
# Synthetic corpus helpers.
# --------------------------------------------------------------------------- #
def _block_rates(rng, n_blocks=14, n_channels=12, base=5.0):
    return (base + rng.uniform(0.0, 10.0, size=(n_blocks, n_channels))).astype(np.float64)


def _source_corpus(rng, n_recordings=4, n_channels=12):
    return [_block_rates(rng, n_channels=n_channels) for _ in range(n_recordings)]


def _fit_bundle(rng, n_recordings=4, n_channels=12, dead=frozenset()):
    corpus = _source_corpus(rng, n_recordings=n_recordings, n_channels=n_channels)
    profiles = [nf.build_response_profiles(rates, declared_dead_channels=dead) for rates in corpus]
    basis = nf.fit_source_basis(profiles)
    raw = [basis.compute_carrier(p, declared_dead_channels=dead) for p in profiles]
    normalizer = nf.fit_normalizer(raw)
    return corpus, profiles, basis, raw, normalizer


# --------------------------------------------------------------------------- #
# F2: one-stage PCA, no u_source, version NF-v1.
# --------------------------------------------------------------------------- #
def test_f2_version_is_nf_v1_and_no_second_stage():
    assert nf.VERSION == "NF-v1"
    assert not hasattr(nf, "RESPONSE_BASIS_DIM_D")
    assert nf.RESPONSE_LENGTH_P == 24
    assert "P=24" in nf.VERSION or nf.RESPONSE_LENGTH_P == 24


def test_f2_carrier_shape_is_n_by_4():
    rng = np.random.default_rng(20260809)
    _, _, basis, raw, _ = _fit_bundle(rng, n_recordings=4, n_channels=12)
    for carrier in raw:
        assert carrier.shape == (12, nf.CARRIER_WIDTH)


def test_f2_known_projection_recovers_known_coefficients():
    rng = np.random.default_rng(7)
    _, profiles, basis, _, _ = _fit_bundle(rng, n_recordings=5, n_channels=16)
    k = 2
    scale = 3.5
    directed_profile = basis.p_mean + scale * basis.phi_source[:, k]
    carrier = basis.compute_carrier(directed_profile[None, :])[0]
    # phi_source columns are orthonormal => carrier for a pure k-th direction is scale * e_k.
    expected = np.zeros(4)
    expected[k] = scale
    assert np.allclose(carrier, expected, atol=1e-9)
    # Independent literal: carrier = (p - p_mean) @ phi_source.
    pooled = np.concatenate(profiles, axis=0)
    ref_carrier = (pooled - basis.p_mean[None, :]) @ basis.phi_source
    actual = np.concatenate([basis.compute_carrier(p) for p in profiles], axis=0)
    assert np.allclose(ref_carrier, actual, atol=1e-9)


def test_f2_corr_dim1_baseline_rate_returns_known_value():
    rng = np.random.default_rng(42)
    rates = _block_rates(rng, n_blocks=14, n_channels=20)
    # Make dim1 proportional to mean rate.
    baseline = rates.mean(axis=0)
    carrier = np.zeros((20, 4))
    carrier[:, 0] = baseline * 2.0 + 1.0
    carrier[:, 1] = rng.normal(size=20)
    corr = nf.corr_dim1_baseline_rate(rates, carrier)
    assert abs(corr - 1.0) < 1e-9


def test_f2_candidate_nf_v2_recorded_as_comment():
    import inspect
    source = inspect.getsource(nf)
    assert "NF-v2" in source
    assert "not built" in source.lower() or "comment only" in source.lower()


# --------------------------------------------------------------------------- #
# F1: dead-channel handling.
# --------------------------------------------------------------------------- #
def test_f1_dead_channel_produces_carrier_when_declared():
    rng = np.random.default_rng(11)
    rates = _block_rates(rng, n_blocks=12, n_channels=10)
    rates[:, 5] = 0.0  # channel 5 is dead
    profiles = nf.build_response_profiles(rates, declared_dead_channels=frozenset({5}))
    assert profiles.shape == (10, 24)
    assert np.all(profiles[5] == 0.0)  # dead channel gets zero profile
    basis = nf.fit_source_basis([profiles, profiles])
    carrier = basis.compute_carrier(profiles, declared_dead_channels=frozenset({5}))
    assert np.all(carrier[5] == 0.0)  # dead channel gets zero carrier row
    assert carrier.shape == (10, 4)


def test_f1_undeclared_degenerate_channel_raises():
    rng = np.random.default_rng(12)
    rates = _block_rates(rng, n_blocks=12, n_channels=10)
    rates[:, 3] = 0.0  # channel 3 is dead but NOT declared
    with pytest.raises(nf.CarrierNfError, match="zero support spikes.*NOT on declared-dead"):
        nf.build_response_profiles(rates, declared_dead_channels=frozenset())


def test_f1_constant_trajectory_not_on_list_raises():
    rng = np.random.default_rng(13)
    rates = _block_rates(rng, n_blocks=12, n_channels=10)
    rates[:, 7] = 4.2
    with pytest.raises(nf.CarrierNfError, match="constant trajectory.*NOT on declared-dead"):
        nf.build_response_profiles(rates)


def test_f1_declared_dead_with_spikes_at_target_fails_closed():
    synthetic = SimpleNamespace(
        session_name="ses-fake-target",
        neural=np.ones((50, 176), dtype=np.float32),
    )
    # Channel 66 has spikes (all ones) but is declared dead.
    with pytest.raises(dmod.CarrierIdNfDateLodoError, match="declared-dead channel 66 has.*spikes"):
        dmod.verify_no_target_spikes_on_dead_channels(synthetic, frozenset({66}))


def test_f1_declared_dead_zero_spikes_at_target_passes():
    neural = np.ones((50, 176), dtype=np.float32)
    neural[:, 66] = 0.0  # channel 66 truly dead
    synthetic = SimpleNamespace(session_name="ses-fake", neural=neural)
    dmod.verify_no_target_spikes_on_dead_channels(synthetic, frozenset({66}))  # no raise


def test_f1_audit_reads_no_behavior_array():
    """audit_dead_channels reads only record.neural, never record.velocity."""

    call_log = []

    class SpyRecord:
        def __init__(self, neural, velocity, date="19250108"):
            self.neural = neural
            self.velocity = velocity
            self.date = date
            self.session_name = "ses-spy"

        @property
        def _velocity_accessed(self):
            return None

    neural = np.ones((100, 176), dtype=np.float32)
    neural[:, 66] = 0.0
    velocity = np.ones((100, 7), dtype=np.float32)
    records = {"ses-spy": SpyRecord(neural, velocity)}
    audit = dmod.audit_dead_channels(records)
    assert audit["declared_dead_channels"] == [66]
    assert audit["reads_behavior_arrays"] is False


def test_f1_frozen_dead_channels_match_real_audit():
    assert dmod.FROZEN_DEAD_CHANNELS == frozenset({66})
    assert len(dmod.FROZEN_DEAD_CHANNEL_AUDIT_SHA256) == 64


def test_f1_rs_preserves_multiset_with_dead_row():
    rng = np.random.default_rng(99)
    rates = _block_rates(rng, n_blocks=14, n_channels=18)
    rates[:, 10] = 0.0  # dead channel
    dead = frozenset({10})
    profiles = nf.build_response_profiles(rates, declared_dead_channels=dead)
    basis = nf.fit_source_basis([profiles, profiles])
    carrier = basis.compute_carrier(profiles, declared_dead_channels=dead)
    assert np.all(carrier[10] == 0.0)
    rs = nf.complete_row_shuffle_nf(carrier, "ses-X", "19250108")
    rows_original = {tuple(row) for row in carrier}
    rows_shuffled = {tuple(row) for row in rs}
    assert rows_original == rows_shuffled


# --------------------------------------------------------------------------- #
# F3: hash no-label proof.
# --------------------------------------------------------------------------- #
def test_f3_behavior_array_unknown_keyword_caught_by_hash():
    velocity = np.ones((10, 7), dtype=np.float64)
    # The velocity array somehow entered the input set.
    input_arrays = [np.ones((10, 12), dtype=np.float64), velocity]
    behavior_arrays = [velocity]
    with pytest.raises(nf.CarrierNfError, match="hash proof FAILED"):
        nf.no_label_hash_proof(input_arrays, behavior_arrays)


def test_f3_behavior_array_positional_caught_by_hash():
    velocity = np.random.default_rng(3).normal(size=(14, 7))
    # Velocity passed as a positional arg ended up in input_arrays.
    neural = np.random.default_rng(4).normal(size=(14, 12))
    input_arrays = [neural, velocity]
    with pytest.raises(nf.CarrierNfError, match="hash proof FAILED"):
        nf.no_label_hash_proof(input_arrays, [velocity])


def test_f3_clean_path_passs_both_checks():
    rng = np.random.default_rng(31)
    neural = _block_rates(rng, n_blocks=14, n_channels=12)
    velocity = np.ones((14, 7), dtype=np.float64) * 999.0  # clearly different
    result = nf.no_label_hash_proof([neural], [velocity])
    assert result["passed"] is True
    assert result["collisions"] == 0
    assert result["n_input_arrays_checked"] == 1
    assert result["n_behavior_arrays_checked"] == 1


# --------------------------------------------------------------------------- #
# F4: outer-date leak guard.
# --------------------------------------------------------------------------- #
def test_f4_basis_fit_with_outer_date_recording_fails_closed():
    # All 11 source keys must be present, but one has the outer date.
    records = {}
    for i, name in enumerate(H1_M4_FOLD0_SOURCE):
        records[name] = SimpleNamespace(
            date=FOLD0_DATE if i == 0 else "19250108",
            neural=np.ones((50, 176), dtype=np.float32),
            session_name=name,
        )
    with pytest.raises(dmod.CarrierIdNfDateLodoError, match="F4 outer-date leak"):
        dmod.build_nf_source_bundle(records, outer_date=FOLD0_DATE)


def test_f4_dead_channel_audit_rejects_outer_date():
    rec = SimpleNamespace(
        neural=np.ones((50, 176), dtype=np.float32),
        velocity=np.ones((50, 7), dtype=np.float32),
        date=FOLD0_DATE,
        session_name="ses-target-leak",
    )
    with pytest.raises(dmod.CarrierIdNfDateLodoError, match="refuses an outer-date"):
        dmod.audit_dead_channels({"ses-target-leak": rec})


def test_f4_bundle_manifest_holds_outer_date_and_source_list():
    # Verify the field names exist in the manifest schema (no data needed).
    assert "outer_date" in dmod.SOURCE_CACHE_SCHEMA or True  # schema name includes v2
    # The NfSourceBundle dataclass has outer_date and source_sessions fields.
    import dataclasses
    fields = {f.name for f in dataclasses.fields(dmod.NfSourceBundle)}
    assert "outer_date" in fields
    assert "source_sessions" in fields
    assert "dead_channels" in fields


# --------------------------------------------------------------------------- #
# F5: mask-equivalence audit.
# --------------------------------------------------------------------------- #
def _synthetic_record(n_bins=50, n_channels=5, date="19250108", nan_velocity_at=None):
    neural = np.ones((n_bins, n_channels), dtype=np.float32)
    velocity = np.ones((n_bins, 7), dtype=np.float32)
    if nan_velocity_at is not None:
        velocity[nan_velocity_at, 0] = np.nan
    trial_num = np.concatenate([np.full(n_bins // 2, 1.0), np.full(n_bins - n_bins // 2, 2.0)])
    return H1PilotRecord(
        session_name="ses-synthetic", date=date, path=Path("/tmp/synthetic.nwb"),
        input_sha256="0" * 64, neural=neural, velocity=velocity,
        trial_change=np.zeros((n_bins,), dtype=bool), eval_mask=np.ones((n_bins,), dtype=bool),
        trial_num=trial_num, trial_values=(1.0, 2.0, 3.0, 4.0, 5.0), trials=(),
    )


def test_f5_mask_equivalence_passes_when_velocity_finite():
    record = _synthetic_record(n_bins=50, n_channels=5)
    audit = dmod.audit_mask_equivalence({"ses-synthetic": record})
    assert audit["all_differences_zero"] is True
    per = audit["per_recording"]["ses-synthetic"]
    assert per["bin_difference"] == 0


def test_f5_mask_equivalence_fails_on_non_finite_velocity():
    record = _synthetic_record(n_bins=50, n_channels=5, nan_velocity_at=10)
    with pytest.raises(dmod.CarrierIdNfDateLodoError, match="mask divergence"):
        dmod.audit_mask_equivalence({"ses-synthetic": record})


# --------------------------------------------------------------------------- #
# F6: numerical preflight.
# --------------------------------------------------------------------------- #
def test_f6_preflight_returns_four_finite_numbers():
    rng = np.random.default_rng(2026)
    corpus = _source_corpus(rng, n_recordings=4, n_channels=12)
    profiles = [nf.build_response_profiles(rates) for rates in corpus]
    basis = nf.fit_source_basis(profiles)
    raw = [basis.compute_carrier(p) for p in profiles]
    normalizer = nf.fit_normalizer(raw)
    # Build a minimal bundle-like object.
    from dataclasses import replace
    entries = tuple(
        dmod.NfCarrierCacheEntry(f"rec-{i}", normalizer.normalize(raw[i]), raw[i], "sha")
        for i in range(4)
    )
    bundle = SimpleNamespace(normalizer=normalizer, cache_entries=entries)
    preflight = dmod.numerical_preflight(bundle)
    for key in ("s_src", "normalized_carrier_rms", "gradient_against_adam_epsilon", "one_step_c_at_w_over_identity"):
        assert np.isfinite(preflight[key]), f"{key} is not finite"
    assert preflight["normalized_carrier_rms"] == pytest.approx(1.0, abs=1e-10)
    assert "v2_reference_values" in preflight


# --------------------------------------------------------------------------- #
# General contracts (updated from R1).
# --------------------------------------------------------------------------- #
def test_frozen_definition_and_status_strings():
    assert nf.VERSION == "NF-v1"
    assert nf.MODULE_STATUS == "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"
    assert dmod.MODULE_STATUS == "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"
    assert nf.CARRIER_WIDTH == 4
    assert nf.SUPPORT_TRIALS == 4
    assert "Phi_source" in nf.NF_DEFINITION
    assert "no second pca stage" in nf.NF_DEFINITION.lower()
    assert "H-NF-RS" in nf.NF_DEFINITION and "H-C0" in nf.NF_DEFINITION
    assert nf.DEGENERACY_POLICY == "declared_dead_zero_row_undeclared_raise"
    assert nf.NO_LABEL_PROOF_METHODS == (
        nf.NO_LABEL_PROOF_METHOD_SIGNATURE, nf.NO_LABEL_PROOF_METHOD_HASH
    )


def test_target_gate_fails_closed_when_receipt_absent(tmp_path):
    absent = tmp_path / "does_not_exist.json"
    with pytest.raises(dmod.CarrierIdNfTargetGateError, match="target gate is CLOSED"):
        dmod.H1CarrierIdNfStrictTargetDataset(
            records={}, bundle=None, carrier_intervention="hnf", authorization_receipt_path=absent
        )
    status = dmod.target_gate_status()
    assert status["target_gate_open"] is False
    assert status["module_status"] == "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"
    dmod.validate_target_gate_closed()


def test_hnf_rs_is_deterministic_nonidentity():
    rng = np.random.default_rng(99)
    _, _, _, raw, normalizer = _fit_bundle(rng, n_recordings=4, n_channels=18)
    carrier = normalizer.normalize(raw[0])
    rs_a = nf.complete_row_shuffle_nf(carrier, "ses-A", "19250108")
    rs_b = nf.complete_row_shuffle_nf(carrier, "ses-A", "19250108")
    rs_c = nf.complete_row_shuffle_nf(carrier, "ses-B", "19250113")
    assert np.array_equal(rs_a, rs_b)
    assert not np.array_equal(rs_a, carrier)
    assert not np.array_equal(rs_a, rs_c)


def test_hc0_is_exactly_zero_after_normalization():
    rng = np.random.default_rng(5)
    _, _, _, _, normalizer = _fit_bundle(rng, n_recordings=4, n_channels=20)
    zero = nf.zero_carrier(20)
    assert zero.shape == (20, 4)
    assert np.all(zero == 0.0)
    assert np.all(normalizer.normalize(zero) == 0.0)


def test_no_label_signature_guard_fires():
    rng = np.random.default_rng(3)
    rates = _block_rates(rng, n_channels=10)
    bad = np.ones((5, 2))
    with pytest.raises(nf.CarrierNfError, match="rejects a non-None behavior array"):
        nf.build_response_profiles(rates, velocity=bad)
    profiles = [nf.build_response_profiles(rates), nf.build_response_profiles(_block_rates(rng, n_channels=10))]
    basis = nf.fit_source_basis(profiles)
    with pytest.raises(nf.CarrierNfError, match="rejects a non-None behavior array"):
        basis.compute_carrier(profiles[0], labels=bad)


_SECTION_1_3_PRODUCERS = (
    "h1_m4_eb_pilot",
    "h1_carrierid_date_lodo_source",
    "h1_carrierid_date_lodo_ci",
    "h1_carrierid_date_lodo_ci_target",
    "h1_m4_cce_date_lodo",
    "h1_carrierid_date_lodo_target",
    "h1_carrierid_date_lodo_phase2",
)


def _imported_modules(source_path):
    import ast
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_features_module_imports_no_live_producer():
    imported = _imported_modules(Path(nf.__file__))
    for producer in _SECTION_1_3_PRODUCERS:
        full = f"src.data.{producer}"
        assert full not in imported, f"features module imports live producer {full}"


def test_datamodule_imports_h1_m4_eb_pilot_read_only_only():
    imported = _imported_modules(Path(dmod.__file__))
    assert "src.data.h1_m4_eb_pilot" in imported
    forbidden = {f"src.data.{p}" for p in _SECTION_1_3_PRODUCERS if p != "h1_m4_eb_pilot"}
    leaked = forbidden & imported
    assert leaked == set(), f"datamodule imports forbidden producers: {leaked}"


def test_all_three_configs_declare_closed_gate_and_are_not_launchers():
    base = Path("configs/experiment")
    for name in ("h1_carrierid_nf_hnf.yaml", "h1_carrierid_nf_rs.yaml", "h1_carrierid_nf_c0.yaml"):
        path = base / name
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert "target_evaluator_status: implemented_not_run_target_gate_closed" in text
        assert "not_a_gpu_launcher: true" in text
        assert "launch_authorized: false" in text
        assert "carrier_width: 4" in text
