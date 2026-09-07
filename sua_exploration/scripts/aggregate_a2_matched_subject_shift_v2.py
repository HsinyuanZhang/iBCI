#!/usr/bin/env python3
"""Fail-closed aggregation for the A2 matched subject-shift v2 matrix.

The primary statistic is the external-minus-within interaction in carrier gain:

    [mean_M(T4 - Z4)] - [mean_C(T4 - Z4)]

There are only three training seeds.  This aggregate deliberately does *not*
compute a seed-level Wilcoxon test: a two-sided exact Wilcoxon with n=3 cannot
reach p <= .05.  Its primary uncertainty gate is instead the contract's
hierarchical bootstrap, which resamples paired seeds and then resamples
sessions independently inside each scoring domain.

The secondary within/external carrier contrasts retain their prescribed
session-level exact Wilcoxon gate.  Those are descriptives/secondary gates,
not a substitute for the primary interaction gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUA_ROOT = REPO_ROOT / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze import a2_matched_subject_shift_v2_core as core


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_receipt(
    path: Path,
    *,
    source_arm: str,
    seed: int,
    domain: str,
    contract_sha256: str,
    official_preflight_sha256: str,
    implementation_bindings: Mapping[str, Any],
    implementation_bindings_sha256: str,
) -> dict[str, Any]:
    receipt, receipt_sha = core.load_verified_immutable_json(path, label="A2 v2 domain receipt")
    arm = core.source_arm_for_name(source_arm)
    _require(receipt.get("schema_version") == 3, f"{path}: schema drift")
    _require(receipt.get("screen_id") == core.SCREEN_ID, f"{path}: screen id drift")
    _require(receipt.get("contract_sha256") == contract_sha256, f"{path}: contract SHA drift")
    _require(receipt.get("official_preflight_sha256") == official_preflight_sha256,
             f"{path}: official preflight SHA drift")
    _require(receipt.get("implementation_bindings") == implementation_bindings,
             f"{path}: implementation bindings drift")
    _require(receipt.get("implementation_bindings_sha256") == implementation_bindings_sha256,
             f"{path}: implementation binding digest drift")
    _require(isinstance(receipt.get("cell_launch_receipt_sha256"), str) and len(receipt["cell_launch_receipt_sha256"]) == 64,
             f"{path}: cell launch receipt digest missing")
    _require(receipt.get("source_arm") == source_arm, f"{path}: source arm drift")
    _require(receipt.get("arm") == arm["arm"], f"{path}: arm drift")
    _require(receipt.get("variant") == arm["variant"], f"{path}: variant drift")
    _require(receipt.get("seed") == seed, f"{path}: seed drift")
    _require(receipt.get("domain") == domain, f"{path}: domain drift")
    _require(receipt.get("no_test_files_evaluated") is True, f"{path}: formal-test isolation drift")
    _require(receipt.get("formal_subc_test_nwb_opened") is False, f"{path}: formal-test NWB was opened")
    _require(receipt.get("target_session_carrier_fit_performed") is True,
             f"{path}: target-session carrier construction was not declared")
    _require(receipt.get("target_direction_labels_used_for_carrier") is True,
             f"{path}: target direction-label carrier use was not declared")
    _require(receipt.get("target_velocity_labels_used_for_weight_updates") is False,
             f"{path}: target velocity labels updated weights")
    _require(receipt.get("backward_gradients") is False, f"{path}: backward gradients were used")
    _require(receipt.get("decoder_weight_updates") is False, f"{path}: decoder weights were updated")
    _require(receipt.get("target_domain_normalizer_refit_performed") is False,
             f"{path}: target normalizer refit")
    _require(receipt.get("source_checkpoint_scored_unchanged_on_both_domains_required") is True,
             f"{path}: missing source-checkpoint reuse declaration")
    _require(receipt.get("query_policy") == core.frozen_query_policy(), f"{path}: query policy drift")
    core.receipt_protocol_invariant(receipt)
    source = receipt.get("source_run")
    _require(isinstance(source, Mapping), f"{path}: source run receipt missing")
    bundle = core.validate_checkpoint_sha256_bundle(receipt.get("source_checkpoint_sha256_bundle"), label=f"{path}: source bundle")
    _require(source.get("source_checkpoint_sha256_bundle") == bundle, f"{path}: top/source checkpoint bundle disagreement")
    _require(source.get("source_checkpoint_sha256_bundle_sha256") == core.canonical_json_sha256(bundle),
             f"{path}: source checkpoint bundle digest drift")
    _require(receipt.get("source_checkpoint_sha256_bundle_sha256") == core.canonical_json_sha256(bundle),
             f"{path}: top checkpoint bundle digest drift")
    protocol = receipt.get("protocol")
    _require(isinstance(protocol, Mapping), f"{path}: protocol missing")
    for key, expected in {
        "total_epochs": core.TOTAL_EPOCHS,
        "epoch_window": list(core.EPOCH_WINDOW),
        "activity_calibration_n": core.ACTIVITY_CALIBRATION_TRIALS,
        "pool_size": core.SIDE_FEATURE_POOL_TRIALS,
        "selection_mode": "first",
        "evaluation_start_trial_index": core.EVALUATION_START_TRIAL_INDEX,
        "loss_mode": "task_only",
        "identity_mode": "calibrated",
        "signal_view": "sua",
    }.items():
        _require(protocol.get(key) == expected, f"{path}: protocol drift: {key}")
    expected_sessions = core.expected_domain_sessions(domain)
    _require(tuple(receipt.get("domain_sessions") or ()) == expected_sessions, f"{path}: domain roster drift")
    _require(receipt.get("domain_session_count") == len(expected_sessions), f"{path}: domain count drift")
    per_epoch = receipt.get("per_epoch")
    _require(isinstance(per_epoch, Mapping) and set(per_epoch) == {str(epoch) for epoch in core.EPOCH_WINDOW},
             f"{path}: epoch window receipt drift")
    for epoch in core.EPOCH_WINDOW:
        row = per_epoch[str(epoch)]
        _require(isinstance(row, Mapping), f"{path}: malformed epoch {epoch}")
        _require(row.get("checkpoint_sha256") == bundle[str(epoch)], f"{path}: epoch {epoch} checkpoint SHA drift")
        values = row.get("per_session_r2")
        _require(isinstance(values, Mapping) and tuple(values) == expected_sessions,
                 f"{path}: epoch {epoch} per-session roster/order drift")
        parsed = np.asarray([float(values[session]) for session in expected_sessions], dtype=np.float64)
        _require(np.isfinite(parsed).all(), f"{path}: epoch {epoch} has non-finite R2")
        declared_epoch_mean = float(row.get("mean_r2"))
        _require(np.isfinite(declared_epoch_mean), f"{path}: epoch {epoch} non-finite mean R2")
        # The scorer deliberately uses Python's left-to-right `sum` for its
        # receipt.  Repeat that arithmetic here rather than NumPy's reduction
        # so exact provenance verification cannot fail only because a vector
        # backend chose a different floating summation order.
        expected_epoch_mean = sum(float(values[session]) for session in expected_sessions) / len(expected_sessions)
        _require(declared_epoch_mean == expected_epoch_mean, f"{path}: epoch {epoch} declared mean is not exact")
    sessions = receipt.get("session_query_receipts")
    _require(isinstance(sessions, Mapping) and tuple(sessions) == expected_sessions,
             f"{path}: session query receipt roster/order drift")
    for session in expected_sessions:
        session_row = sessions[session]
        _require(isinstance(session_row, Mapping), f"{path}: malformed session query receipt")
        _require(session_row.get("activity_calibration_trial_indices") == list(range(30)),
                 f"{path}: {session} activity M30 selection drift")
        _require(session_row.get("side_feature_label_pool_trial_indices") == list(range(30)),
                 f"{path}: {session} T4 label M30 pool drift")
        _require(session_row.get("query_usable_trial_indices_start") == 30,
                 f"{path}: {session} query start drift")
        _require(session_row.get("dataset_query_window_count") == session_row.get("post30_query_window_count"),
                 f"{path}: {session} query window count drift")
        _require(int(session_row.get("dataset_query_window_count", 0)) > 0,
                 f"{path}: {session} has no post-M30 query windows")
        _require(session_row.get("target_session_carrier_fit_performed") is True,
                 f"{path}: {session} target-session carrier construction was not declared")
        _require(session_row.get("target_direction_labels_used_for_carrier") is True,
                 f"{path}: {session} target direction-label carrier use was not declared")
        _require(session_row.get("target_velocity_labels_used_for_weight_updates") is False,
                 f"{path}: {session} target velocity labels updated weights")
        _require(session_row.get("backward_gradients") is False,
                 f"{path}: {session} backward gradients were used")
        _require(session_row.get("decoder_weight_updates") is False,
                 f"{path}: {session} decoder weights were updated")
    means = np.asarray(
        [
            sum(float(per_epoch[str(epoch)]["per_session_r2"][session]) for epoch in core.EPOCH_WINDOW)
            / len(core.EPOCH_WINDOW)
            for session in expected_sessions
        ],
        dtype=np.float64,
    )
    declared_means = receipt.get("per_session_mean_r2")
    _require(isinstance(declared_means, Mapping) and tuple(declared_means) == expected_sessions,
             f"{path}: declared session mean roster/order drift")
    declared = np.asarray([float(declared_means[session]) for session in expected_sessions], dtype=np.float64)
    _require(np.allclose(means, declared, rtol=0.0, atol=1.0e-15),
             f"{path}: declared session averages are not exact epoch averages")
    expected_grand_mean = sum(float(value) for value in declared) / len(declared)
    _require(float(receipt.get("mean_r2")) == expected_grand_mean, f"{path}: declared grand mean is not exact")
    receipt["_immutable_receipt_sha256"] = receipt_sha
    return receipt


def _hierarchical_ci_paired_delta(delta: np.ndarray, *, draws: int, seed: int) -> tuple[list[float], np.ndarray]:
    """Crossed seed x session bootstrap for one paired arm contrast.

    A session is a repeated cluster observed under every training seed.  Each
    draw therefore samples one seed roster and one session roster, with the
    latter shared across sampled seeds rather than independently duplicated
    inside each seed.
    """
    _require(delta.ndim == 2 and delta.shape[0] == len(core.SEEDS), "delta must be [3 seeds, sessions]")
    _require(draws >= 1, "bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    n_seed, n_session = delta.shape
    selected_seeds = rng.integers(0, n_seed, size=(draws, n_seed))
    selected_sessions = rng.integers(0, n_session, size=(draws, n_session))
    sampled = delta[selected_seeds[:, :, None], selected_sessions[:, None, :]].mean(axis=(1, 2))
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])], sampled


def hierarchical_interaction_bootstrap_ci(
    within_delta: np.ndarray,
    external_delta: np.ndarray,
    *,
    draws: int = 50_000,
    seed: int = 20260812,
) -> tuple[list[float], np.ndarray]:
    """Crossed seed x session bootstrap, paired across domains and arms.

    `within_delta` and `external_delta` are each [seed, session] matrices of
    T4−Z4.  A single seed draw is shared across both domains and arms to keep
    the interaction paired; independent session draws are made per sampled
    domain but shared across the sampled seed roster, preserving session as a
    repeated cluster instead of pseudoreplicating it once per seed.
    """
    _require(within_delta.ndim == external_delta.ndim == 2, "interaction matrices must be two-dimensional")
    _require(within_delta.shape[0] == external_delta.shape[0] == len(core.SEEDS), "interaction matrices must have three seeds")
    _require(draws >= 1, "bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    n_seed = len(core.SEEDS)
    seed_indices = rng.integers(0, n_seed, size=(draws, n_seed))
    within_sessions = rng.integers(0, within_delta.shape[1], size=(draws, within_delta.shape[1]))
    external_sessions = rng.integers(0, external_delta.shape[1], size=(draws, external_delta.shape[1]))
    sampled_within = within_delta[seed_indices[:, :, None], within_sessions[:, None, :]].mean(axis=(1, 2))
    sampled_external = external_delta[seed_indices[:, :, None], external_sessions[:, None, :]].mean(axis=(1, 2))
    sampled_interaction = sampled_external - sampled_within
    return [float(value) for value in np.quantile(sampled_interaction, [0.025, 0.975])], sampled_interaction


def _exact_session_wilcoxon(session_means: np.ndarray) -> float:
    """The valid secondary n=6/n=15 session-level Wilcoxon, never n=3 seeds."""
    try:
        return float(wilcoxon(session_means, alternative="two-sided", zero_method="wilcox", method="exact").pvalue)
    except ValueError:
        return 1.0


def _secondary_gates(delta: np.ndarray, sessions: tuple[str, ...], *, bootstrap_seed: int, bootstrap_draws: int) -> dict[str, Any]:
    _require(delta.shape == (len(core.SEEDS), len(sessions)), "secondary delta topology drift")
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    ci, _samples = _hierarchical_ci_paired_delta(delta, draws=bootstrap_draws, seed=bootstrap_seed)
    p_value = _exact_session_wilcoxon(session_means)
    gates = {
        "mean_paired_delta_at_least_0p03": bool(float(delta.mean()) >= core.INTERACTION_THRESHOLD),
        "all_three_seed_means_positive": bool(np.all(seed_means > 0.0)),
        "all_session_means_positive": bool(np.all(session_means > 0.0)),
        "hierarchical_seed_then_session_bootstrap_95ci_lower_positive": bool(ci[0] > 0.0),
        "session_paired_exact_wilcoxon_two_sided_le_0p05": bool(p_value <= 0.05),
    }
    return {
        "mean_paired_delta_r2": float(delta.mean()),
        "per_seed_mean_delta_r2": {str(seed): float(value) for seed, value in zip(core.SEEDS, seed_means)},
        "per_session_mean_delta_r2": {session: float(value) for session, value in zip(sessions, session_means)},
        "positive_seed_count": int((seed_means > 0.0).sum()),
        "positive_session_count": int((session_means > 0.0).sum()),
        "hierarchical_bootstrap": {
            "method": "crossed_seed_x_session_bootstrap; one session resample shared across sampled seeds",
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
            "95ci": ci,
        },
        "session_paired_exact_wilcoxon_two_sided_p": p_value,
        "wilcoxon_scope": "session means over 3 seed-paired values; never the 3 seed means themselves",
        "gates": gates,
        "passes_all_gates": bool(all(gates.values())),
    }


def _validate_cross_receipt_invariants(receipts: Mapping[tuple[str, int, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Prove pairing/reuse before numerical aggregation is permitted."""
    normalizer_authority_reference: Mapping[str, Any] | None = None
    query_policy_reference: Mapping[str, Any] | None = None
    source_bundle_pairs: list[dict[str, Any]] = []
    domain_arm_pairs: list[dict[str, Any]] = []
    session_query_pairs: list[dict[str, Any]] = []
    for source_arm in core.SOURCE_ARMS:
        for seed in core.SEEDS:
            within = receipts[(source_arm, seed, "within_subject")]
            external = receipts[(source_arm, seed, "external_subject_M")]
            for key in (
                "source_run_metadata_sha256",
                "source_checkpoint_sha256_bundle",
                "source_checkpoint_sha256_bundle_sha256",
                "cell_launch_receipt_sha256",
                "query_policy",
                "normalizer_authority",
            ):
                _require(within.get(key) == external.get(key),
                         f"{source_arm}/s{seed}: source-domain pair mismatch: {key}")
            source_bundle_pairs.append({
                "source_arm": source_arm,
                "seed": seed,
                "within_external_source_bundle_identical": True,
                "source_checkpoint_sha256_bundle_sha256": within["source_checkpoint_sha256_bundle_sha256"],
            })
    for domain in core.DOMAINS:
        for seed in core.SEEDS:
            z4 = receipts[("source_z4", seed, domain)]
            t4 = receipts[("source_t4", seed, domain)]
            _require(z4.get("domain_sessions") == t4.get("domain_sessions"), f"{domain}/s{seed}: Z4/T4 domain roster mismatch")
            _require(z4.get("query_policy") == t4.get("query_policy"), f"{domain}/s{seed}: Z4/T4 query policy mismatch")
            _require(z4.get("normalizer_authority") == t4.get("normalizer_authority"),
                     f"{domain}/s{seed}: Z4/T4 normalizer authority mismatch")
            _require(z4.get("session_query_receipts") == t4.get("session_query_receipts"),
                     f"{domain}/s{seed}: Z4/T4 M30/trial-30 query trace mismatch")
            domain_arm_pairs.append({"domain": domain, "seed": seed, "z4_t4_session_roster_identical": True,
                                     "z4_t4_query_policy_identical": True, "z4_t4_normalizer_authority_identical": True})
            session_query_pairs.append({"domain": domain, "seed": seed, "z4_t4_session_query_receipts_identical": True})
    for receipt in receipts.values():
        if normalizer_authority_reference is None:
            normalizer_authority_reference = receipt["normalizer_authority"]
            query_policy_reference = receipt["query_policy"]
        else:
            _require(receipt["normalizer_authority"] == normalizer_authority_reference,
                     "all A2 v2 receipts must share one source-train normalizer authority")
            _require(receipt["query_policy"] == query_policy_reference,
                     "all A2 v2 receipts must share one M30/trial-30 query policy")
    return {
        "all_required_receipts_present": True,
        "within_external_same_source_epoch_sha_bundle": source_bundle_pairs,
        "z4_t4_pairings": domain_arm_pairs,
        "z4_t4_query_trace_pairings": session_query_pairs,
        "all_receipts_share_normalizer_authority": True,
        "all_receipts_share_query_policy": True,
        "passed": True,
    }


def aggregate(
    *,
    result_dir: Path,
    contract: Path = core.CONTRACT_PATH,
    bootstrap_draws: int = 50_000,
    bootstrap_seed: int = 20260812,
) -> dict[str, Any]:
    result_dir = result_dir.expanduser().resolve()
    contract = contract.expanduser().resolve()
    _require(contract.is_file(), f"contract missing: {contract}")
    core.validate_config()
    contract_sha256 = core.sha256_file(contract)
    official_preflight_path = core.official_preflight_path(result_root=result_dir)
    official_preflight, official_preflight_sha256 = core.load_verified_official_preflight(
        official_preflight_path,
        result_root=result_dir,
    )
    implementation_bindings = core.verify_implementation_bindings(official_preflight["implementation_bindings"])
    implementation_bindings_sha256 = core.implementation_bindings_sha256(implementation_bindings)
    receipts: dict[tuple[str, int, str], dict[str, Any]] = {}
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for source_arm in core.SOURCE_ARMS:
        for domain in core.DOMAINS:
            sessions = core.expected_domain_sessions(domain)
            rows: list[np.ndarray] = []
            for seed in core.SEEDS:
                path = core.domain_result_path(source_arm, seed, domain, result_root=result_dir)
                _require(path.is_file(), f"missing required domain result: {path}")
                receipt = _load_receipt(
                    path,
                    source_arm=source_arm,
                    seed=seed,
                    domain=domain,
                    contract_sha256=contract_sha256,
                    official_preflight_sha256=official_preflight_sha256,
                    implementation_bindings=implementation_bindings,
                    implementation_bindings_sha256=implementation_bindings_sha256,
                )
                receipts[(source_arm, seed, domain)] = receipt
                rows.append(np.asarray([float(receipt["per_session_mean_r2"][session]) for session in sessions], dtype=np.float64))
            matrices[(source_arm, domain)] = np.asarray(rows, dtype=np.float64)
    pairing = _validate_cross_receipt_invariants(receipts)

    within_sessions = core.expected_domain_sessions("within_subject")
    external_sessions = core.expected_domain_sessions("external_subject_M")
    within_delta = matrices[("source_t4", "within_subject")] - matrices[("source_z4", "within_subject")]
    external_delta = matrices[("source_t4", "external_subject_M")] - matrices[("source_z4", "external_subject_M")]
    carrier_gain_within = within_delta.mean(axis=1)
    carrier_gain_external = external_delta.mean(axis=1)
    interaction_per_seed = carrier_gain_external - carrier_gain_within
    mean_interaction = float(interaction_per_seed.mean())
    sigma_paired = float(interaction_per_seed.std(ddof=1) / np.sqrt(len(core.SEEDS)))
    interaction_ci, _interaction_samples = hierarchical_interaction_bootstrap_ci(
        within_delta, external_delta, draws=bootstrap_draws, seed=bootstrap_seed
    )
    interaction_gates = {
        "mean_interaction_at_least_0p03": bool(mean_interaction >= core.INTERACTION_THRESHOLD),
        "all_three_seed_interactions_strictly_positive": bool(np.all(interaction_per_seed > 0.0)),
        "hierarchical_bootstrap_95ci_lower_positive": bool(interaction_ci[0] > 0.0),
        "receipt_pairing_and_reuse_invariants_pass": bool(pairing["passed"]),
    }
    interaction_effective = bool(all(interaction_gates.values()))
    if interaction_effective:
        verdict = {
            "name": "subject_shift_interaction_effective",
            "passes": True,
            "reason": "all pre-registered interaction gates passed",
        }
    elif mean_interaction + 2.0 * sigma_paired < core.INTERACTION_THRESHOLD:
        verdict = {
            "name": "interaction_ineffective",
            "passes": False,
            "reason": "mean_interaction + 2*sigma_paired is below +0.03",
        }
    else:
        verdict = {
            "name": "interaction_indeterminate",
            "passes": False,
            "reason": "descriptive 2x2 only; three seeds do not support a population-level subject-generalization p-value",
        }
    secondary = {
        "within_subject_t4_minus_z4": _secondary_gates(
            within_delta, within_sessions, bootstrap_seed=bootstrap_seed + 1, bootstrap_draws=bootstrap_draws
        ),
        "external_subject_M_t4_minus_z4": _secondary_gates(
            external_delta, external_sessions, bootstrap_seed=bootstrap_seed + 2, bootstrap_draws=bootstrap_draws
        ),
    }
    cell_means = {
        "within_subject_z4": float(matrices[("source_z4", "within_subject")].mean()),
        "within_subject_t4": float(matrices[("source_t4", "within_subject")].mean()),
        "external_subject_M_z4": float(matrices[("source_z4", "external_subject_M")].mean()),
        "external_subject_M_t4": float(matrices[("source_t4", "external_subject_M")].mean()),
    }
    payload = {
        "schema_version": 3,
        "screen_id": core.SCREEN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract),
        "contract_sha256": contract_sha256,
        "official_preflight_path": str(official_preflight_path),
        "official_preflight_sha256": official_preflight_sha256,
        "implementation_bindings": implementation_bindings,
        "implementation_bindings_sha256": implementation_bindings_sha256,
        "result_dir": str(result_dir),
        "formal_subc_test_nwb_opened": False,
        "receipt_pairing_validation": pairing,
        "descriptive_2x2_cell_mean_r2": cell_means,
        "carrier_gain": {
            "within_subject_per_seed": {str(seed): float(value) for seed, value in zip(core.SEEDS, carrier_gain_within)},
            "external_subject_M_per_seed": {str(seed): float(value) for seed, value in zip(core.SEEDS, carrier_gain_external)},
            "within_subject_per_session_mean": {session: float(value) for session, value in zip(within_sessions, within_delta.mean(axis=0))},
            "external_subject_M_per_session_mean": {session: float(value) for session, value in zip(external_sessions, external_delta.mean(axis=0))},
        },
        "interaction": {
            "definition": "mean_session_R2(t4 external_subject_M) - mean_session_R2(z4 external_subject_M) - [mean_session_R2(t4 within_subject) - mean_session_R2(z4 within_subject)]",
            "per_seed": {str(seed): float(value) for seed, value in zip(core.SEEDS, interaction_per_seed)},
            "mean_interaction": mean_interaction,
            "sigma_interaction_paired": sigma_paired,
            "hierarchical_bootstrap": {
                "method": "paired seed resampling shared across domains; one independent session resample per domain shared across sampled seeds",
                "draws": bootstrap_draws,
                "seed": bootstrap_seed,
                "95ci": interaction_ci,
            },
            "seed_level_wilcoxon": {
                "computed": False,
                "reason": "prohibited: with three seed-level interactions an exact two-sided Wilcoxon cannot attain p <= 0.05",
            },
            "gates": interaction_gates,
            "passes_all_gates": interaction_effective,
            "verdict": verdict,
        },
        "secondary_contrasts": secondary,
        "historical_hints_table_only": {
            "within_subject_activity_only_z4": 0.326008,
            "cross_subject_activity_only_zero4_m50_trial50": -0.057766,
            "cross_subject_carrier_t4_m50_trial50": 0.356828,
            "claim": "historical protocol-mismatched values only; no replication claim from this table",
        },
    }
    return payload


def _write_once(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    return core.write_immutable_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=core.CONTRACT_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = aggregate(
            result_dir=args.result_dir,
            contract=args.contract,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
        )
        _body, sidecar, digest = _write_once(args.out, payload)
    except (ValueError, core.A2V2ContractError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"interaction": payload["interaction"], "out": str(args.out), "sidecar": str(sidecar), "sha256": digest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
