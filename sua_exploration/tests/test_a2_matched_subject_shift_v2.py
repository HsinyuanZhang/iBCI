"""Focused, no-GPU tests for the additive A2 matched subject-shift v2 package."""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SUA) not in sys.path:
    sys.path.insert(0, str(SUA))

from sua_exploration.mc_maze import a2_matched_subject_shift_v2_core as core


AGGREGATOR_PATH = SUA / "scripts" / "aggregate_a2_matched_subject_shift_v2.py"
PREFLIGHT_PATH = SUA / "scripts" / "a2_matched_subject_shift_v2_preflight.py"
SCORER_PATH = SUA / "scripts" / "a2_matched_subject_shift_v2_score.py"
RUNNER_PATH = SUA / "scripts" / "run_a2_matched_subject_shift_v2_one_cell.sh"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _aggregator():
    return _module("a2_v2_aggregator_test", AGGREGATOR_PATH)


def _preflight():
    return _module("a2_v2_preflight_test", PREFLIGHT_PATH)


def _source_bundle(fill: str) -> dict[str, str]:
    return {str(epoch): fill * 64 for epoch in core.EPOCH_WINDOW}


def _normalizer_authority() -> dict:
    return {
        "policy": core.frozen_normalizer_policy(),
        "source_train_sessions": core.load_strict_manifest()["train"],
        "source_train_session_count": 27,
        "behavior_normalizer_value_sha256": "b" * 64,
        "side_normalizer_value_sha256": "c" * 64,
        "source_training_side_normalizer_value_sha256": "c" * 64,
        "cached_training_path_behavior_normalizer_value_sha256": "b" * 64,
        "cached_training_path_side_normalizer_value_sha256": "c" * 64,
        "cached_vs_uncached_behavior_values_bitwise_identical": True,
        "cached_vs_uncached_side_values_bitwise_identical": True,
        "training_cache_root": str(core.SOURCE_CACHE_ROOT.resolve()),
        "training_path_behavior_cache": "/fake/behavior.npz",
        "training_path_behavior_cache_sha256": "d" * 64,
        "training_path_side_cache": "/fake/t4.npz",
        "training_path_side_cache_sha256": "e" * 64,
        "target_domain_normalizer_refit_performed": False,
        "target_domain_normalizer_refit_forbidden": True,
        "formal_test_sessions_resolved_or_opened": False,
        "formal_test_session_names_only": core.load_strict_manifest()["test"],
        "normalizer_cache_dir": None,
    }


def _session_trace(index: int) -> dict:
    return {
        "usable_rewarded_trial_count": 40 + index,
        "activity_support_usable_indices": list(range(30)),
        "activity_support_original_trial_indices": list(range(30)),
        "side_feature_label_pool_usable_indices": list(range(30)),
        "first30_target_dir_all_finite": True,
        "query_usable_trial_indices_start": 30,
        "query_usable_trial_count": 10 + index,
        "post30_query_window_count": 100 + index,
        "dataset_query_window_count": 100 + index,
        "activity_calibration_trial_indices": list(range(30)),
        "side_feature_label_pool_trial_indices": list(range(30)),
        "behavior_normalizer_authority": "strict_subc_source_train_27_only",
        "side_normalizer_authority": "strict_subc_source_train_27_only",
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_velocity_labels_used_for_weight_updates": False,
        "backward_gradients": False,
        "decoder_weight_updates": False,
    }


def _receipt(
    source_arm: str,
    seed: int,
    domain: str,
    *,
    offset: float = 0.0,
    official_sha: str = "p" * 64,
    bindings: dict | None = None,
) -> dict:
    arm = core.source_arm_for_name(source_arm)
    sessions = core.expected_domain_sessions(domain)
    bundle = _source_bundle("a" if source_arm == "source_z4" else "d")
    values = {session: offset + 0.01 * index for index, session in enumerate(sessions)}
    per_epoch = {
        str(epoch): {
            "checkpoint_path": f"/fake/{source_arm}/s{seed}/epoch_{epoch}.ckpt",
            "checkpoint_sha256": bundle[str(epoch)],
            "per_session_r2": values,
            "mean_r2": float(np.mean(list(values.values()))),
        }
        for epoch in core.EPOCH_WINDOW
    }
    normalizer = _normalizer_authority()
    return {
        "schema_version": 3,
        "screen_id": core.SCREEN_ID,
        "contract_path": str(core.CONTRACT_PATH),
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH),
        "official_preflight_sha256": official_sha,
        "implementation_bindings": bindings if bindings is not None else {},
        "implementation_bindings_sha256": core.implementation_bindings_sha256(bindings or {}),
        "cell_launch_receipt_path": f"/fake/{source_arm}/s{seed}/cell_launch.json",
        "cell_launch_receipt_sha256": "l" * 64,
        "source_arm": source_arm,
        "arm": arm["arm"],
        "variant": arm["variant"],
        "seed": seed,
        "domain": domain,
        "domain_sessions": list(sessions),
        "domain_session_count": len(sessions),
        "source_run": {
            "source_run_dir": f"/fake/{source_arm}/s{seed}",
            "source_run_metadata_path": f"/fake/{source_arm}/s{seed}/run_metadata.json",
            "source_run_metadata_sha256": ("e" if source_arm == "source_z4" else "f") * 64,
            "source_checkpoint_sha256_bundle": bundle,
            "source_checkpoint_sha256_bundle_sha256": core.canonical_json_sha256(bundle),
        },
        "source_run_metadata_path": f"/fake/{source_arm}/s{seed}/run_metadata.json",
        "source_run_metadata_sha256": ("e" if source_arm == "source_z4" else "f") * 64,
        "source_checkpoint_scored_unchanged_on_both_domains_required": True,
        "source_checkpoint_sha256_bundle": bundle,
        "source_checkpoint_sha256_bundle_sha256": core.canonical_json_sha256(bundle),
        "query_policy": core.frozen_query_policy(),
        "normalizer_authority": normalizer,
        "protocol": {
            "total_epochs": 12,
            "epoch_window": list(range(5, 13)),
            "epoch_score_rule": "unweighted mean session R2 over exactly source epochs 5..12",
            "activity_calibration_n": 30,
            "pool_size": 30,
            "selection_mode": "first",
            "evaluation_start_trial_index": 30,
            "loss_mode": "task_only",
            "identity_mode": "calibrated",
            "signal_view": "sua",
        },
        "per_epoch": per_epoch,
        "per_session_mean_r2": values,
        "mean_r2": float(np.mean(list(values.values()))),
        "session_query_receipts": {session: _session_trace(index) for index, session in enumerate(sessions)},
        "no_test_files_evaluated": True,
        "formal_subc_test_nwb_opened": False,
        "formal_subc_test_session_names_only": core.load_strict_manifest()["test"],
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_velocity_labels_used_for_weight_updates": False,
        "backward_gradients": False,
        "decoder_weight_updates": False,
        "target_domain_normalizer_refit_performed": False,
    }


def _official_preflight_payload(root: Path, *, bindings: dict | None = None) -> dict:
    bindings = bindings if bindings is not None else core.current_implementation_bindings()
    within_sessions = core.expected_domain_sessions("within_subject")
    external_sessions = core.expected_domain_sessions("external_subject_M")
    normalizer = _normalizer_authority()
    # An official preflight deliberately seals authority/rosters but defers
    # numeric normalizer values until a completed source run exists.
    normalizer["behavior_normalizer_value_sha256"] = None
    normalizer["side_normalizer_value_sha256"] = None
    return {
        "schema_version": 3,
        "receipt_kind": core.OFFICIAL_PREFLIGHT_KIND,
        "official_preflight": True,
        "screen_id": core.SCREEN_ID,
        "status": core.OFFICIAL_PREFLIGHT_STATUS,
        "non_authorizing_status": True,
        "root_go_required_before_gpu": True,
        "cpu_only": True,
        "gpu_used": False,
        "training_started": False,
        "checkpoint_loaded": False,
        "formal_subc_test_nwb_opened": False,
        "result_root": str(root.resolve()),
        "expected_fresh_gpu_cells": 6,
        "forbidden_duplicate_domain_training_cells": 12,
        "source_training_cells": list(core.SOURCE_ARMS),
        "seeds": list(core.SEEDS),
        "scoring_domains": list(core.DOMAINS),
        "query_policy": core.frozen_query_policy(),
        "config_validated": True,
        "normalizer_authority": normalizer,
        "within_subject_audit": {
            "expected_count": len(within_sessions),
            "admissible_count": len(within_sessions),
            "sessions": [{"session": session, "admissible": True} for session in within_sessions],
        },
        "external_subject_M_audit": {
            "expected_count": len(external_sessions),
            "admissible_count": len(external_sessions),
            "sessions": [{"session": session, "admissible": True} for session in external_sessions],
        },
        "implementation_blockers": [],
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH),
        "config_sha256": core.sha256_file(core.CONFIG_PATH),
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": core.implementation_bindings_sha256(bindings),
        "python_isolation": core.python_isolation_binding(),
    }


def _write_immutable(path: Path, payload: dict) -> str:
    _body, sidecar, digest = core.write_immutable_json(path, payload)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    assert sidecar.read_text(encoding="ascii") == f"{digest}  {path.name}\n"
    return digest


def _write_matrix(root: Path, *, gains: dict[tuple[str, int], float] | None = None) -> None:
    """Write a valid synthetic 12-receipt matrix with controlled carrier gains."""
    gains = gains or {}
    root.mkdir(parents=True, exist_ok=True)
    bindings = core.current_implementation_bindings()
    official_sha = _write_immutable(core.official_preflight_path(result_root=root), _official_preflight_payload(root, bindings=bindings))
    for domain in core.DOMAINS:
        for seed in core.SEEDS:
            z4 = _receipt("source_z4", seed, domain, offset=0.1, official_sha=official_sha, bindings=bindings)
            gain = gains.get((domain, seed), 0.04 if domain == "external_subject_M" else 0.01)
            t4 = _receipt("source_t4", seed, domain, offset=0.1 + gain, official_sha=official_sha, bindings=bindings)
            for source_arm, receipt in (("source_z4", z4), ("source_t4", t4)):
                _write_immutable(core.domain_result_path(source_arm, seed, domain, result_root=root), receipt)


def test_core_config_has_two_source_arms_three_seeds_and_m30_contract() -> None:
    config = core.validate_config()
    assert config["screen_id"] == core.SCREEN_ID
    assert tuple(config["seeds"]) == (42, 43, 44)
    assert tuple(row["name"] for row in config["source_training_cells"]) == ("source_z4", "source_t4")
    assert config["frozen_protocol"]["activity_calibration_trials"] == 30
    assert config["frozen_protocol"]["side_feature_label_pool_trials"] == 30
    assert config["frozen_protocol"]["evaluation_start_trial_index"] == 30
    assert config["checkpoint_reuse"]["one_source_run_per_arm_seed"] is True
    assert config["normalizer_authority"]["target_fit_forbidden"] is True


def test_trial30_contract_accepts_post30_only_and_rejects_short_or_missing_labels() -> None:
    trials = [
        {"start": i * 80, "stop": i * 80 + 70, "trial_index": i, "target_dir": float(i)}
        for i in range(31)
    ]
    audit = core.trial30_semantics_from_trials(trials, session="synthetic")
    assert audit["activity_support_usable_indices"] == list(range(30))
    assert audit["side_feature_label_pool_usable_indices"] == list(range(30))
    assert audit["query_usable_trial_indices_start"] == 30
    assert audit["post30_query_window_count"] == 21
    with pytest.raises(core.A2V2ContractError, match="more than 30"):
        core.trial30_semantics_from_trials(trials[:30], session="short")
    broken = list(trials)
    broken[5] = {**broken[5], "target_dir": None}
    with pytest.raises(core.A2V2ContractError, match="target_dir"):
        core.trial30_semantics_from_trials(broken, session="missing-label")


def test_external_session_roster_is_exactly_15_and_disjoint_from_sealed_subc() -> None:
    external = core.expected_external_sessions()
    within = core.expected_within_sessions()
    assert len(external) == 15
    assert len(within) == 6
    assert all(name.startswith("sub-M_ses-CO-") for name in external)
    assert not set(external) & set(core.load_strict_manifest()["test"])


def test_hierarchical_interaction_bootstrap_resamples_sessions_independently_by_domain() -> None:
    aggregate = _aggregator()
    within = np.asarray([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]], dtype=np.float64)
    external = np.asarray([[0.06, 0.07, 0.08], [0.09, 0.10, 0.11], [0.12, 0.13, 0.14]], dtype=np.float64)
    ci, samples = aggregate.hierarchical_interaction_bootstrap_ci(within, external, draws=301, seed=9)
    assert len(samples) == 301
    assert len(ci) == 2 and ci[0] <= ci[1]
    assert np.isfinite(samples).all()
    assert samples.mean() > 0


def test_aggregate_prohibits_impossible_three_seed_wilcoxon_and_requires_checkpoint_reuse(tmp_path: Path) -> None:
    aggregate = _aggregator()
    result_dir = tmp_path / "results"
    _write_matrix(result_dir)
    payload = aggregate.aggregate(result_dir=result_dir, bootstrap_draws=301, bootstrap_seed=21)
    assert payload["receipt_pairing_validation"]["passed"] is True
    assert payload["interaction"]["seed_level_wilcoxon"]["computed"] is False
    assert "three seed-level" in payload["interaction"]["seed_level_wilcoxon"]["reason"]
    assert payload["interaction"]["mean_interaction"] == pytest.approx(0.03)
    assert payload["interaction"]["verdict"]["name"] in {"subject_shift_interaction_effective", "interaction_indeterminate"}
    assert payload["secondary_contrasts"]["within_subject_t4_minus_z4"]["wilcoxon_scope"].startswith("session means")


def test_bootstrap_treats_session_as_crossed_cluster_shared_across_seeds() -> None:
    aggregate = _aggregator()
    delta = np.arange(18, dtype=np.float64).reshape(3, 6)
    draws = 17
    seed = 91
    _ci, observed = aggregate._hierarchical_ci_paired_delta(delta, draws=draws, seed=seed)
    rng = np.random.default_rng(seed)
    selected_seeds = rng.integers(0, 3, size=(draws, 3))
    # One session roster per draw, shared across the three sampled seeds.
    selected_sessions = rng.integers(0, 6, size=(draws, 6))
    expected = delta[selected_seeds[:, :, None], selected_sessions[:, None, :]].mean(axis=(1, 2))
    np.testing.assert_array_equal(observed, expected)


def test_aggregate_fails_closed_when_external_receipt_changes_source_checkpoint_bundle(tmp_path: Path) -> None:
    aggregate = _aggregator()
    result_dir = tmp_path / "results"
    _write_matrix(result_dir)
    path = core.domain_result_path("source_t4", 42, "external_subject_M", result_root=result_dir)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["source_checkpoint_sha256_bundle"]["5"] = "0" * 64
    receipt["source_run"]["source_checkpoint_sha256_bundle"]["5"] = "0" * 64
    receipt["source_checkpoint_sha256_bundle_sha256"] = core.canonical_json_sha256(receipt["source_checkpoint_sha256_bundle"])
    receipt["source_run"]["source_checkpoint_sha256_bundle_sha256"] = receipt["source_checkpoint_sha256_bundle_sha256"]
    receipt["per_epoch"]["5"]["checkpoint_sha256"] = "0" * 64
    # A post-write body mutation cannot retain a valid immutable sidecar.
    # Make the adversarial mutation possible only in this test, preserve the
    # original sidecar, then demand that the aggregate rejects integrity before
    # it can consume altered scientific content.
    path.chmod(0o644)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises((ValueError, core.A2V2ContractError), match="integrity mismatch|SHA sidecar/body"):
        aggregate.aggregate(result_dir=result_dir, bootstrap_draws=101)


def test_aggregate_fails_closed_on_target_normalizer_refit_claim(tmp_path: Path) -> None:
    aggregate = _aggregator()
    result_dir = tmp_path / "results"
    _write_matrix(result_dir)
    path = core.domain_result_path("source_z4", 43, "external_subject_M", result_root=result_dir)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["target_domain_normalizer_refit_performed"] = True
    path.chmod(0o644)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises((ValueError, core.A2V2ContractError), match="integrity mismatch|SHA sidecar/body"):
        aggregate.aggregate(result_dir=result_dir, bootstrap_draws=101)


def test_receipt_provenance_requires_target_carrier_but_forbids_adaptation() -> None:
    receipt = _receipt("source_t4", 42, "external_subject_M", bindings=core.current_implementation_bindings())
    core.receipt_protocol_invariant(receipt)
    assert receipt["target_session_carrier_fit_performed"] is True
    assert receipt["target_direction_labels_used_for_carrier"] is True
    assert receipt["target_velocity_labels_used_for_weight_updates"] is False
    assert receipt["backward_gradients"] is False
    assert receipt["decoder_weight_updates"] is False

    missing_carrier = dict(receipt)
    missing_carrier["target_session_carrier_fit_performed"] = False
    with pytest.raises(core.A2V2ContractError, match="carrier construction"):
        core.receipt_protocol_invariant(missing_carrier)

    adapted = dict(receipt)
    adapted["decoder_weight_updates"] = True
    with pytest.raises(core.A2V2ContractError, match="decoder weight updates"):
        core.receipt_protocol_invariant(adapted)


def test_preflight_reports_exact_six_fresh_gpu_cells_and_15_external_m30_rows(tmp_path: Path) -> None:
    preflight = _preflight()
    receipt = preflight.build_receipt(result_root=tmp_path / "fresh", audit_data=True)
    assert receipt["expected_fresh_gpu_cells"] == 6
    assert receipt["forbidden_duplicate_domain_training_cells"] == 12
    assert receipt["external_subject_M_audit"]["expected_count"] == 15
    assert receipt["external_subject_M_audit"]["admissible_count"] == 15
    assert receipt["within_subject_audit"]["admissible_count"] == 6
    # Preflight seals source-only normalizer authority before any target
    # session is scored; its pre-existing `target_domain_fit_performed` field
    # is retained there for compatibility.  Domain-score receipts use the
    # explicit carrier-vs-adaptation provenance asserted above.
    assert receipt["normalizer_authority"]["target_domain_fit_performed"] is False
    assert receipt["normalizer_authority"]["behavior_normalizer_value_sha256"] is None
    assert receipt["normalizer_authority"]["value_digest_status"].startswith("DEFERRED_UNTIL_SOURCE_RUN_EXISTS")
    assert receipt["formal_subc_test_nwb_opened"] is False


def test_cell_freshness_includes_trainer_global_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = tmp_path / "global_summary.json"
    monkeypatch.setattr(core, "source_run_dir", lambda source_arm, seed: tmp_path / "checkpoint")
    monkeypatch.setattr(core, "source_summary_path", lambda source_arm, seed: summary)
    targets = core.assert_cell_fresh(source_arm="source_z4", seed=42, result_root=tmp_path / "matrix")
    assert targets["trainer_global_summary"] == str(summary)
    summary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(core.A2V2ContractError, match="trainer_global_summary"):
        core.assert_cell_fresh(source_arm="source_z4", seed=42, result_root=tmp_path / "matrix")


def test_runner_dry_run_is_inert_and_names_six_source_cells() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER_PATH), "--dry-run"],
        cwd=ROOT,
        env={**__import__("os").environ, "SOURCE_ARM": "source_z4", "SEED": "42", "GPU": "0"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "EXPECTED_FRESH_GPU_CELLS=6" in result.stdout
    assert "SAME_UNCHANGED_SOURCE_EPOCH_BUNDLE_SCORED_ON=within_subject,external_subject_M" in result.stdout
    assert "--chronological_calibration" in result.stdout
    assert "--side_feature_pool_size 30" in result.stdout
    assert "--calibration_n_trials 30" in result.stdout
    assert "--domain external_subject_M" in result.stdout


def test_real_frozen_model_loader_import_path_exists() -> None:
    scorer = _module("a2_v2_scorer_import_test", SCORER_PATH)
    # The scorer's late import must resolve the real loader module before any
    # checkpoint/NWB/GPU work.  This caught an earlier import from a module
    # that did not define load_frozen_model at all.
    from scripts.select_gradient_free_protocol_dandi688 import load_frozen_model

    assert callable(load_frozen_model)
    assert str(scorer.SCRIPTS_ROOT) in sys.path


def test_scorer_dry_run_requires_existing_valid_source_run_before_claiming_readiness(tmp_path: Path) -> None:
    # Dry run may inspect a source run fingerprint, but it must never invent a
    # scorer-ready run when provenance is absent.
    result = subprocess.run(
        [
            sys.executable,
            str(SCORER_PATH),
            "--source-arm", "source_z4", "--seed", "42", "--domain", "within_subject",
            "--run-dir", str(tmp_path / "missing"), "--out-path", str(tmp_path / "out.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (tmp_path / "out.json").exists()
