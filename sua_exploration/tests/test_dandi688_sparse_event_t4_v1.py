from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_maze.dandi688_sparse_event_t4_v1 import plan
from mc_maze.dandi688_sparse_event_t4_v1.core import (
    apply_reliability_mask,
    array_sha256,
    deterministic_nonidentity_permutation,
    phase_r_profile,
    row_shuffle_profile,
)
from mc_maze.dandi688_sparse_event_t4_v1.descriptors import (
    classify_era_from_first30,
    is_legal_phase_trial,
    materialize_sparse_event_t4,
)
from mc_maze.dandi688_sparse_event_t4_v1.reliability import (
    SourceReliabilityAudit,
    aggregate_column_reliability,
    assert_audit_is_disjoint,
    retention_mask,
)
from mc_maze.dandi688_sparse_event_t4_v1.runner import (
    GPUAuthorizationRequired,
    average_last_four_trainable_states,
    require_explicit_gpu_authorization,
    executable_cells,
    stage2_estimator_contract,
    stage_plan,
)
from mc_maze.dandi688_sparse_event_t4_v1.scoring import (
    average_paired_seeds,
    baseline_claim_gate,
    film_gate,
    paired_bootstrap,
)
from mc_maze.dandi688_sparse_event_t4_v1.stage0 import validate_era_counts, verify_frozen_inputs
from mc_maze.dandi688_sparse_event_t4_v1.stage0 import execute_stage0
from mc_maze.dandi688_sparse_event_t4_v1.descriptors import SparseEventMaterialization
from mc_maze.dandi688_sparse_event_t4_v1.production import prepare_source_surface
from mc_maze.dandi688_sparse_event_t4_v1.production import run_stage2_estimator_coordinated
from mc_maze.dandi688_sparse_event_t4_v1.production import _publish_stage2_full_state_artifacts
from mc_maze.dandi688_sparse_event_t4_v1.production import run_stage2_film_coordinated
from mc_maze.dandi688_sparse_event_t4_v1.production import _publish_stage2_film_full_state_artifacts
from mc_maze.dandi688_sparse_event_t4_v1.production import _all_q50_training_batches
from mc_maze.dandi688_sparse_event_t4_v1.production import _score_stage2_film_q50
from mc_maze.dandi688_sparse_event_t4_v1.production import _admit_stage2_film_opening
from mc_maze.dandi688_sparse_event_t4_v1.production import execute_seed_stage
from mc_maze.dandi688_sparse_event_t4_v1.admission import admit_stage1_successor
from mc_maze.dandi688_sparse_event_t4_v1.aggregate import aggregate_stage1_v2
from mc_maze.dandi688_sparse_event_t4_v1.core import SparseEventContractError
from mc_maze.dandi688_sparse_event_t4_v1.stage2_aggregate import (
    aggregate_stage2,
    admit_stage3_branch,
    admit_stage2_v2_successor,
    aggregate_stage2_v2,
    admit_stage2_v3_successor,
    aggregate_stage2_v3,
)
from mc_maze.dandi688_sparse_event_t4_v1.stage3_aggregate import aggregate_stage3
from mc_maze.dandi688_sparse_event_t4_v1.final_evidence import (
    parse_gpu_dmon,
    surface_evidence,
    validate_sidecar_inventory,
)


def _trial(position: int, *, legal: bool = True, delay: float = 1.0) -> dict[str, object]:
    directions = tuple(-3.0 * math.pi / 4.0 + index * math.pi / 4.0 for index in range(8))
    return {
        "trial_index": position,
        "start_time": 0.0,
        "stop_time": 3.0 if legal else 1.6,
        "target_dir": directions[position % len(directions)],
        "target_on_time": 0.5,
        "go_cue_time": 0.5 + delay,
    }


def test_final_evidence_gpu_monitor_and_sidecar_inventory(tmp_path):
    body = tmp_path / "receipt.json"
    body.write_bytes(b"{}\n")
    digest = hashlib.sha256(body.read_bytes()).hexdigest()
    body.with_name("receipt.json.sha256").write_text(f"{digest}  receipt.json\n", encoding="ascii")
    inventory = validate_sidecar_inventory(tmp_path)
    assert inventory == [{
        "body_relative": "receipt.json",
        "sha256": digest,
        "size_bytes": 3,
        "sidecar_relative": "receipt.json.sha256",
    }]

    summary = parse_gpu_dmon([
        "#Date Time gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk fb bar1 ccpm",
        "20260904 22:48:29 0 300 75 - 96 30 0 0 0 0 9501 1890 2193 14 0",
        "20260904 22:48:39 0 320 76 - 98 34 0 0 0 0 9501 1890 2193 14 0",
    ])
    assert summary["sample_count"] == 2
    assert summary["sm_percent"] == {"mean": 97.0, "minimum": 96.0, "maximum": 98.0}
    assert summary["power_w"]["mean"] == 310.0


def test_final_surface_evidence_closes_27_6_activity_carrier_profile_digests():
    sessions = {}
    for index in range(33):
        value = np.asarray([[float(index), float(index + 1)]], dtype=np.float32)
        sessions[f"session-{index:02d}"] = SimpleNamespace(
            split="train" if index < 27 else "val",
            record=SimpleNamespace(calib_trials=value + 1),
            q50_starts=np.asarray([50, 51], dtype=np.int64),
            parent_m30_t4=value + 2,
            whole_m10_raw=value + 3,
            post700_m10_raw=value + 4,
            whole_m10_parentnorm=value + 5,
            whole_m10_t4=value + 6,
            post700_m10_t4=value + 7,
            profile_m10=value + 8,
            receipt={"session_id": f"session-{index:02d}"},
        )
    surface = SimpleNamespace(
        sessions=sessions,
        normalizers={"whole_m10": {"mean": np.zeros(2), "scale": np.ones(2)}},
        parent_m30_normalizer={"mean": np.zeros(2), "scale": np.ones(2)},
        reliability_mask=(True, True, True, True),
        signal_view="sua",
    )
    evidence = surface_evidence(surface)
    assert evidence["view"] == "sua"
    assert len(evidence["sessions"]) == 33
    assert sum(row["split"] == "train" for row in evidence["sessions"].values()) == 27
    first = evidence["sessions"]["session-00"]
    assert first["activity_m30_sha256"] == array_sha256(sessions["session-00"].record.calib_trials)
    assert first["post700_m10_t4_sha256"] == array_sha256(sessions["session-00"].post700_m10_t4)


def test_actual_stage1_successor_admission_binds_frozen_incident_and_v1_failures():
    repo_root = Path(__file__).resolve().parents[2]
    admitted = admit_stage1_successor(repo_root)
    assert admitted["successor_incident_sha256"] == plan.STAGE1_SUCCESSOR_INCIDENT_SHA256
    assert admitted["correction"] == "ONE_TEMPLATE_DEEPCOPIED_TO_FOUR_FILM_ARMS"
    assert set(admitted["v1_failure_witnesses"]) == {"42", "43"}
    assert set(admitted["v2_failure_witnesses"]) == {"42", "43"}
    assert admitted["v3_incident_sha256"] == plan.STAGE1_V3_INCIDENT_SHA256
    assert admitted["production_pythonpath"] == plan.PRODUCTION_PYTHONPATH
    for seed in (42, 43):
        witness = admitted["v1_failure_witnesses"][str(seed)]
        expected = plan.STAGE1_V1_FAILURE_WITNESSES[seed]
        assert witness["attempt_sha256"] == expected["attempt_sha256"]
        assert witness["failure_sha256"] == expected["failure_sha256"]
    for seed in (42, 43):
        witness = admitted["v2_failure_witnesses"][str(seed)]
        expected = plan.STAGE1_V2_FAILURE_WITNESSES[seed]
        assert witness["attempt_sha256"] == expected["attempt_sha256"]
        assert witness["failure_sha256"] == expected["failure_sha256"]


def _trials(*, delay: float = 1.0) -> list[dict[str, object]]:
    return [_trial(position, delay=delay) for position in range(30)]


def _rate_pooler(_: Path, intervals):
    starts = np.asarray([item["start_time"] for item in intervals], dtype=np.float64)
    stops = np.asarray([item["stop_time"] for item in intervals], dtype=np.float64)
    duration = stops - starts
    n = duration.size
    angles = np.asarray(tuple(-3.0 * math.pi / 4.0 + (index % 8) * math.pi / 4.0 for index in range(n)))
    if np.allclose(duration, plan.H300_SECONDS):
        return np.vstack((np.full(n, 4.0), np.full(n, 3.0), np.full(n, 2.0))), 3
    if np.allclose(duration, plan.R700_SECONDS):
        return np.vstack((7.0 + 2.0 * np.cos(angles), 8.0 + 3.0 * np.sin(angles), 5.0 + np.cos(angles) - np.sin(angles))), 3
    return np.vstack((6.0 + 0.5 * np.cos(angles), 6.0 + 0.5 * np.sin(angles), 5.0 + np.cos(angles))), 3


def _lister(_: Path, **_kwargs):
    return _trials()


def test_half_open_window_and_label_horizon_materialization_has_no_dense_behavior_surface():
    assert is_legal_phase_trial(_trial(0)) == (True, None)
    assert is_legal_phase_trial(_trial(0, legal=False))[0] is False
    materialized = materialize_sparse_event_t4(Path("sub-C_ses-CO-20131003.nwb"), trial_lister=_lister, rate_pooler=_rate_pooler)
    assert materialized.whole_t4.shape == materialized.raw_profile.shape == (3, 4)
    assert np.array_equal(materialized.post700_t4, materialized.r700_t4)
    assert np.all(materialized.legal_candidate_positions < 10)
    assert materialized.receipt["dense_velocity_scalars_consumed"] == 0
    assert materialized.receipt["candidate_pool_n"] == 10
    assert materialized.receipt["activity_support_n"] == 30
    assert materialized.receipt["carrier_profile_label_horizon"] == 10
    names = set(inspect.signature(materialize_sparse_event_t4).parameters)
    assert "velocity" not in names and "position" not in names and "nwb" not in names


def test_dense_behavior_trap_is_never_touched_by_descriptor_materializer():
    class BehaviorTrap:
        accesses = 0

        def __getitem__(self, _key):
            self.accesses += 1
            raise AssertionError("dense behavior access is forbidden")

    trap = BehaviorTrap()

    def guarded_lister(path, **kwargs):
        assert trap.accesses == 0
        return _lister(path, **kwargs)

    def guarded_pooler(path, intervals):
        assert trap.accesses == 0
        return _rate_pooler(path, intervals)

    materialize_sparse_event_t4(Path("sub-C_ses-CO-20131003.nwb"), trial_lister=guarded_lister, rate_pooler=guarded_pooler)
    assert trap.accesses == 0


def test_profile_order_is_a_r_c_r_m_r_delta_b_and_pseudomua_refits_after_pooling():
    materialized = materialize_sparse_event_t4(Path("sub-C_ses-CO-20131003.nwb"), trial_lister=_lister, rate_pooler=_rate_pooler)
    assert np.allclose(materialized.raw_profile[:, :3], materialized.r700_t4[:, :3])
    assert np.allclose(materialized.raw_profile[:, 3], materialized.r700_t4[:, 3] - materialized.h300_t4[:, 3])
    pseudo = materialize_sparse_event_t4(Path("sub-C_ses-CO-20131003.nwb"), signal_view="pseudo_mua", trial_lister=_lister, rate_pooler=_rate_pooler, electrode_id_loader=lambda _: np.asarray([7, 7, 11]))
    assert pseudo.raw_profile.shape == (2, 4)
    assert pseudo.receipt["view"] == "pseudo_mua"


def test_era_assignment_and_frozen_era_count_contract():
    assert classify_era_from_first30(_trials(delay=0.001)) == "no_delay"
    assert classify_era_from_first30(_trials(delay=0.3)) == "short_delay"
    assert classify_era_from_first30(_trials(delay=1.0)) == "long_delay"
    rows = []
    rows += [{"split": "train", "era": "no_delay"}] * 9
    rows += [{"split": "train", "era": "short_delay"}] * 5
    rows += [{"split": "train", "era": "long_delay"}] * 13
    rows += [{"split": "val", "era": "long_delay"}] * 6
    assert validate_era_counts(rows) == plan.ERA_EXPECTED_COUNTS


def test_mask_phase_r_and_row_shuffle_are_exact_deterministic_and_nonidentity():
    profile = np.arange(20, dtype=np.float32).reshape(5, 4)
    masked = apply_reliability_mask(profile, (1, 0, 1, 0))
    assert np.array_equal(masked[:, 1], np.zeros(5, dtype=np.float32))
    assert np.array_equal(masked[:, 3], np.zeros(5, dtype=np.float32))
    phase = phase_r_profile(profile, (1, 1, 1, 1))
    assert np.array_equal(phase[:, 3], np.zeros(5, dtype=np.float32))
    first, permutation = row_shuffle_profile(profile, session_id="sub-C_ses-CO-20131003", view="sua", training_seed=42)
    second, repeat = row_shuffle_profile(profile, session_id="sub-C_ses-CO-20131003", view="sua", training_seed=42)
    assert np.array_equal(first, second) and np.array_equal(permutation, repeat)
    assert not np.array_equal(permutation, np.arange(5))
    assert np.array_equal(permutation, deterministic_nonidentity_permutation(rows=5, session_id="sub-C_ses-CO-20131003", view="sua", training_seed=42))


def test_reliability_audit_namespace_is_memory_disjoint_and_uses_both_gates():
    candidate = np.ones((3, 4), dtype=np.float32)
    split, reference = candidate.copy(), candidate.copy()
    audit = SourceReliabilityAudit(split_half={"a": split}, deployment_reference={"a": reference})
    assert_audit_is_disjoint([candidate], audit)
    split_summary = aggregate_column_reliability({name: [0.8] * 27 for name in plan.PROFILE_COLUMN_NAMES}, gate="split_half")
    reference_summary = aggregate_column_reliability({name: [0.7] * 27 for name in plan.PROFILE_COLUMN_NAMES}, gate="reference")
    assert retention_mask(split_summary, reference_summary) == (True, True, True, True)
    with pytest.raises(Exception, match="share memory"):
        assert_audit_is_disjoint([candidate], SourceReliabilityAudit(split_half={"a": candidate}, deployment_reference={"a": reference}))


def test_stage_plans_are_bound_to_epoch11_paired_seeds_and_gate_cell_execution():
    plan1 = stage_plan(1, film_admitted=True)
    plan2 = stage_plan(2, film_admitted=False)
    assert plan1["parent_checkpoint"] == plan2["parent_checkpoint"] == "epoch_011.ckpt"
    assert plan1["seeds"] == [42, 43, 44]
    assert plan2["epoch_average_zero_based"] == [8, 9, 10, 11]
    stage1 = executable_cells(stage=1, film_admitted=False, estimator_admitted=False, stage1_film_open=False)
    stage2 = executable_cells(stage=2, film_admitted=False, estimator_admitted=True, stage1_film_open=False)
    assert len(stage1) == 9  # Stage-1 estimator is descriptive/non-gating.
    assert len(stage2) == 6
    with pytest.raises(GPUAuthorizationRequired):
        require_explicit_gpu_authorization(False)


def _publish_synthetic_stage1_v3_seed(root: Path, seed: int, *, evidence_mismatch: bool = False) -> dict[str, str]:
    """Write one complete immutable Stage-1-v3 seed topology for aggregation."""
    from mc_maze.dandi688_sparse_event_t4_v1.lifecycle import publish_immutable_json

    root.mkdir(parents=True)
    attempt = publish_immutable_json(root, "attempt.json", {"status": "STARTED", "stage": "stage1_v3", "seed": seed})
    artifacts = {
        arm: publish_immutable_json(root, f"stage1_{arm.lower()}.pt", {"seed": seed, "arm": arm})
        for arm in ("EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE")
    }
    trained = ("EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE")
    scored = ("WHOLE-NATIVE", "WHOLE-PARENTNORM", "WHOLE-M10NORM", "POST700-M10NORM", *trained)
    baseline = {
        "WHOLE-NATIVE": 0.10,
        "WHOLE-PARENTNORM": 0.11,
        "WHOLE-M10NORM": 0.12,
        "POST700-M10NORM": 0.15,
        "EMPTY": 0.10,
        "PHASE-R": 0.11,
        "SE-T4": 0.13,
        "ROW-SHUFFLE": 0.12,
    }
    sessions = tuple(f"val{index}" for index in range(6))
    scores = {arm: {session: baseline[arm] + seed * 1.0e-8 for session in sessions} for arm in scored}
    evidence = {}
    for arm in scored:
        evidence[arm] = {}
        for session in sessions:
            target = hashlib.sha256(f"target:{session}".encode()).hexdigest()
            if evidence_mismatch and arm == "SE-T4" and session == "val0":
                target = hashlib.sha256(b"mismatched-target").hexdigest()
            evidence[arm][session] = {
                "prediction_sha256": hashlib.sha256(f"prediction:{seed}:{arm}:{session}".encode()).hexdigest(),
                "target_sha256": target,
                "query_sha256": hashlib.sha256(f"query:{session}".encode()).hexdigest(),
                "window_count": 257,
            }
    parent_sha = hashlib.sha256(f"parent:{seed}".encode()).hexdigest()
    training = publish_immutable_json(root, "training.json", {
        "attempt_sha256": attempt,
        "epochs": plan.EPOCHS,
        "average_epochs_zero_based": list(plan.AVERAGE_EPOCHS_ZERO_BASED),
        "head_template_factory_calls": 1,
        "initial_head_states_identical": True,
        "initial_head_state_sha256": {arm: "a" * 64 for arm in trained},
        "checkpoints": artifacts,
        "zero_anchor": {"passed": True, "window_count": 32},
        "parent_state_before": parent_sha,
        "parent_state_after_training": parent_sha,
    })
    score = publish_immutable_json(root, "score.json", {
        "attempt_sha256": attempt,
        "parent_state_before": parent_sha,
        "parent_state_after_training": parent_sha,
        "parent_state_after_score": parent_sha,
        "per_session_r2": scores,
        "per_session_evidence": evidence,
    })
    terminal = publish_immutable_json(root, "terminal.json", {
        "status": "PASS",
        "attempt_sha256": attempt,
        "training_sha256": training,
        "score_sha256": score,
    })
    return {"attempt": attempt, "training": training, "score": score, "terminal": terminal}


def test_aggregate_stage1_v2_publishes_three_seed_v3_opening_and_six_contrasts(tmp_path):
    root = tmp_path / plan.STAGE1_SUCCESSOR_RELATIVE
    expected = {
        seed: _publish_synthetic_stage1_v3_seed(root / "sua" / f"seed{seed}", seed)
        for seed in plan.SEEDS
    }

    aggregate = aggregate_stage1_v2(tmp_path)
    aggregate_path = root / "aggregate.json"
    body = aggregate_path.read_bytes()

    assert aggregate_path.is_file() and aggregate_path.with_name("aggregate.json.sha256").is_file()
    assert aggregate["aggregate_sha256"] == hashlib.sha256(body).hexdigest()
    assert aggregate["stage"] == "stage1_v3_aggregate"
    assert aggregate["stage1_terminal_sha256"] == {str(seed): expected[seed]["terminal"] for seed in plan.SEEDS}
    assert set(aggregate["summaries"]) == {
        "estimator_ood",
        "semantic_ood",
        "baseline_ood",
        "attachment_ood",
        "product_ood",
        "capacity_or_global_correction_ood",
    }
    assert {name: summary["grand_mean"] for name, summary in aggregate["summaries"].items()} == pytest.approx({
        "estimator_ood": 0.03,
        "semantic_ood": 0.03,
        "baseline_ood": 0.02,
        "attachment_ood": 0.01,
        "product_ood": 0.03,
        "capacity_or_global_correction_ood": 0.0,
    })
    opening = aggregate["stage2_film_opening"]
    assert opening["status"] == "OPEN"
    assert opening["semantic_mean"] == pytest.approx(0.03)
    assert opening["attachment_mean"] == pytest.approx(0.01)
    assert opening["semantic_positive_sessions"] == 6
    assert opening["predicate"] == "semantic_mean>0__attachment_mean>0__semantic_positive_sessions>=4"
    assert aggregate_path.with_name("aggregate.json.sha256").read_text().strip() == f"{aggregate['aggregate_sha256']}  aggregate.json"


def test_aggregate_stage1_v2_rejects_q50_evidence_mismatch_without_publishing(tmp_path):
    root = tmp_path / plan.STAGE1_SUCCESSOR_RELATIVE
    for seed in plan.SEEDS:
        _publish_synthetic_stage1_v3_seed(
            root / "sua" / f"seed{seed}",
            seed,
            evidence_mismatch=(seed == 43),
        )

    with pytest.raises(SparseEventContractError, match="scoring surface mismatch"):
        aggregate_stage1_v2(tmp_path)
    assert not (root / "aggregate.json").exists()
    assert not (root / "aggregate.json.sha256").exists()


def _publish_synthetic_stage2_seed(root: Path, *, branch: str, seed: int, stage_label: str = "stage2", stage2_admission=None) -> dict[str, str]:
    from mc_maze.dandi688_sparse_event_t4_v1.lifecycle import publish_immutable_json

    arms = ("WHOLE-T4", "POST700-T4") if branch == "estimator" else ("WHOLE-NATIVE", "EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE")
    prefix = f"{stage_label}_{branch}"
    stage = f"{stage_label}_{branch}"
    root.mkdir(parents=True)
    attempt_body = {
        "status": "STARTED", "stage": stage, "seed": seed,
        "view": "pseudo_mua" if stage_label == "stage3" else "sua",
        "stage0_admission": {"reliability_mask": [True, True, True, True]},
    }
    if stage2_admission is not None:
        attempt_body["stage2_aggregate_admission"] = stage2_admission
    attempt = publish_immutable_json(root, "attempt.json", attempt_body)
    artifact = {arm: publish_immutable_json(root, f"{prefix}_{arm.lower()}.pt", {"seed": seed, "arm": arm}) for arm in arms}
    epoch_receipts = [{"epoch_zero_based": epoch, "optimizer_steps": {arm: 1 for arm in arms}} for epoch in range(plan.EPOCHS)]
    training = {
        "attempt_sha256": attempt,
        "average_epochs_zero_based": list(plan.AVERAGE_EPOCHS_ZERO_BASED),
        "epoch_receipts": epoch_receipts,
        "full_state_artifacts": {arm: {"artifact_sha256": artifact[arm]} for arm in arms},
    }
    if branch == "film":
        training.update({
            "head_template_factory_calls": 1,
            "initial_head_states_identical": True,
            "initial_head_state_sha256": {arm: "b" * 64 for arm in arms[1:]},
            "zero_anchor": {"passed": True, "window_count": 32},
        })
    training_sha = publish_immutable_json(root, "training.json", training)
    sessions = tuple(f"val{index}" for index in range(6))
    values = (
        {"WHOLE-T4": .10, "POST700-T4": .13}
        if branch == "estimator" else
        {"WHOLE-NATIVE": .10, "EMPTY": .10, "PHASE-R": .11, "SE-T4": .13, "ROW-SHUFFLE": .11}
    )
    score = {arm: {} for arm in arms}
    for arm in arms:
        for session in sessions:
            score[arm][session] = {
                "r2": values[arm] + seed * 1.0e-8,
                "prediction_sha256": hashlib.sha256(f"p:{branch}:{seed}:{arm}:{session}".encode()).hexdigest(),
                "target_sha256": hashlib.sha256(f"t:{session}".encode()).hexdigest(),
                "query_sha256": hashlib.sha256(f"q:{session}".encode()).hexdigest(),
                "window_count": 257,
            }
    score_sha = publish_immutable_json(root, "score.json", {"attempt_sha256": attempt, "per_session": score})
    terminal = publish_immutable_json(root, "terminal.json", {
        "status": "PASS", "attempt_sha256": attempt, "training_sha256": training_sha,
        "score_sha256": score_sha, "full_state_artifacts": training["full_state_artifacts"],
    })
    return {"terminal": terminal, "training": training_sha, "score": score_sha}


def test_stage2_three_seed_aggregate_gates_estimator_and_film_then_admits_stage3(tmp_path):
    root = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage2"
    for branch in ("estimator", "film"):
        for seed in plan.SEEDS:
            _publish_synthetic_stage2_seed(root / branch / "sua" / f"seed{seed}", branch=branch, seed=seed)

    aggregate = aggregate_stage2(tmp_path)

    assert aggregate["estimator"]["status"] == "PASS" and aggregate["estimator"]["gate"] is True
    assert aggregate["film"]["status"] == "PASS" and aggregate["film"]["gate"] is True
    assert aggregate["film"]["baseline_claim_gate"] is True
    assert aggregate["scope"] == "SUA_SOURCE_27_TRAIN__SUA_SOURCE_6_VALIDATION__NO_TEST_EXTERNAL_EVALAI"
    assert aggregate["estimator"]["summary"]["grand_mean"] == pytest.approx(.03)
    assert {name: item["grand_mean"] for name, item in aggregate["film"]["summaries"].items()} == pytest.approx({
        "semantic": .03, "baseline": .02, "attachment": .02, "product": .03, "capacity": .0,
    })
    assert (root / "aggregate.json").is_file() and (root / "aggregate.json.sha256").is_file()
    for branch in ("estimator", "film"):
        admitted = admit_stage3_branch(tmp_path, branch)
        assert admitted["branch"] == branch and admitted["branch_gate"] ["gate"] is True


def test_film_route_gate_does_not_require_positive_delta_b_subclaim():
    semantic = {
        "grand_mean": .020,
        "positive_sessions": 5,
        "worst": -.010,
        "bootstrap": {"lower_95": .001},
        "seed_grand_means": {"42": .010, "43": .020, "44": .030},
    }
    attachment = {"grand_mean": .005}
    baseline = {"grand_mean": -.002}

    assert film_gate(semantic, attachment) is True
    assert baseline_claim_gate(baseline, delta_b_retained=True) is False
    assert baseline_claim_gate({"grand_mean": .002}, delta_b_retained=True) is True
    assert baseline_claim_gate({"grand_mean": .002}, delta_b_retained=False) is False


def test_actual_stage2_v2_incident_admission_holds_exact_abandoned_attempts():
    repo_root = Path(__file__).resolve().parents[2]
    admitted = admit_stage2_v2_successor(repo_root, "film")
    assert admitted["incident_sha256"] == plan.STAGE2_V2_INCIDENT_SHA256
    assert admitted["held_interrupted_attempt_sha256"] == {
        "estimator:44": plan.STAGE2_V1_INTERRUPTED_ATTEMPTS["estimator"][44],
        "film:42": plan.STAGE2_V1_INTERRUPTED_ATTEMPTS["film"][42],
    }


def test_actual_stage2_v3_incident_admission_holds_exact_user_reallocation_failure():
    repo_root = Path(__file__).resolve().parents[2]
    admitted = admit_stage2_v3_successor(repo_root)
    assert admitted["incident_sha256"] == plan.STAGE2_V3_INCIDENT_SHA256
    assert admitted["held_interrupted_v2_estimator"] == plan.STAGE2_V2_INTERRUPTED_ESTIMATOR


def test_stage2_v2_mixed_aggregate_uses_v1_estimator_42_43_and_successor_remainders(tmp_path, monkeypatch):
    import mc_maze.dandi688_sparse_event_t4_v1.stage2_aggregate as stage2_aggregate

    old = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage2"
    for seed in (42, 43):
        _publish_synthetic_stage2_seed(old / "estimator" / "sua" / f"seed{seed}", branch="estimator", seed=seed)
    successor = tmp_path / plan.STAGE2_V2_RELATIVE
    _publish_synthetic_stage2_seed(successor / "estimator" / "sua" / "seed44", branch="estimator", seed=44, stage_label="stage2_v2")
    for seed in plan.SEEDS:
        _publish_synthetic_stage2_seed(successor / "film" / "sua" / f"seed{seed}", branch="film", seed=seed, stage_label="stage2_v2")
    monkeypatch.setattr(stage2_aggregate, "admit_stage2_v2_successor", lambda _root, branch: {"incident_sha256": "i" * 64, "branch": branch})

    # The public aggregate entrypoint must select the successor authority once
    # any stage2_v2 root is present; callers cannot accidentally publish V1.
    aggregate = aggregate_stage2(tmp_path)

    assert aggregate["stage"] == "stage2_v2_aggregate"
    assert aggregate["estimator"]["authority"] == {"42": "stage2_v1", "43": "stage2_v1", "44": "stage2_v2"}
    assert aggregate["film"]["authority"] == {"42": "stage2_v2", "43": "stage2_v2", "44": "stage2_v2"}
    assert (successor / "aggregate.json").is_file()


def test_stage2_v2_public_route_requires_successor_admission_before_any_executor(tmp_path, monkeypatch):
    """The public surface only recognizes successor roots and remains CPU-only."""
    import mc_maze.dandi688_sparse_event_t4_v1.admission as admission
    import mc_maze.dandi688_sparse_event_t4_v1.production as production
    import mc_maze.dandi688_sparse_event_t4_v1.stage2_aggregate as stage2_aggregate

    events, dispatched = [], {}
    def hold_incident(_repo_root, branch):
        events.append(("incident", branch))
        return {"branch": branch, "incident_sha256": "i" * 64}
    def hold_stage0(_repo_root):
        events.append(("stage0", None))
        return {"reliability_mask": [True, True, True, True]}
    def fake_estimator(*_args, **kwargs):
        dispatched["estimator"] = kwargs
        return {"branch": "estimator"}
    def fake_film(*_args, **kwargs):
        dispatched["film"] = kwargs
        return {"branch": "film"}
    monkeypatch.setattr(stage2_aggregate, "admit_stage2_v2_successor", hold_incident)
    monkeypatch.setattr(admission, "admit_stage0_graph", hold_stage0)
    monkeypatch.setattr(production, "_execute_stage3_estimator", fake_estimator)
    monkeypatch.setattr(production, "_execute_stage2_film", fake_film)

    estimator_root = tmp_path / plan.STAGE2_V2_RELATIVE / "estimator" / "sua" / "seed44"
    assert execute_seed_stage(tmp_path, stage=2, seed=44, view="sua", result_root=estimator_root) == {"branch": "estimator"}
    assert events == [("incident", "estimator"), ("stage0", None)]
    assert dispatched["estimator"]["stage_label"] == "stage2_v2"
    assert dispatched["estimator"]["stage2_admission"]["branch_gate"]["reliability_mask"] == [True] * 4

    events.clear()
    film_root = tmp_path / plan.STAGE2_V2_RELATIVE / "film" / "sua" / "seed42"
    assert execute_seed_stage(tmp_path, stage=2, seed=42, view="sua", result_root=film_root) == {"branch": "film"}
    assert events == [("incident", "film"), ("stage0", None)]
    assert dispatched["film"]["stage_label"] == "stage2_v2"
    assert not estimator_root.exists() and not film_root.exists()

    with pytest.raises(SparseEventContractError, match="reruns only interrupted seed44"):
        execute_seed_stage(
            tmp_path,
            stage=2,
            seed=42,
            view="sua",
            result_root=tmp_path / plan.STAGE2_V2_RELATIVE / "estimator" / "sua" / "seed42",
        )


def test_stage2_v3_mixed_aggregate_prefers_v3_and_stage3_reads_latest_authority(tmp_path, monkeypatch):
    import mc_maze.dandi688_sparse_event_t4_v1.stage2_aggregate as stage2_aggregate

    v1 = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage2"
    for seed in (42, 43):
        _publish_synthetic_stage2_seed(v1 / "estimator" / "sua" / f"seed{seed}", branch="estimator", seed=seed)
    v2 = tmp_path / plan.STAGE2_V2_RELATIVE
    for seed in plan.SEEDS:
        _publish_synthetic_stage2_seed(v2 / "film" / "sua" / f"seed{seed}", branch="film", seed=seed, stage_label="stage2_v2")
    v3 = tmp_path / plan.STAGE2_V3_RELATIVE
    _publish_synthetic_stage2_seed(v3 / "estimator" / "sua" / "seed44", branch="estimator", seed=44, stage_label="stage2_v3")
    monkeypatch.setattr(
        stage2_aggregate,
        "admit_stage2_v3_successor",
        lambda _root: {"incident_sha256": "i" * 64, "held_interrupted_v2_estimator": dict(plan.STAGE2_V2_INTERRUPTED_ESTIMATOR)},
    )

    aggregate = aggregate_stage2(tmp_path)

    assert aggregate["stage"] == "stage2_v3_aggregate"
    assert aggregate["estimator"]["authority"] == {"42": "stage2_v1", "43": "stage2_v1", "44": "stage2_v3"}
    assert aggregate["film"]["authority"] == {"42": "stage2_v2", "43": "stage2_v2", "44": "stage2_v2"}
    estimator_admission = admit_stage3_branch(tmp_path, "estimator")
    film_admission = admit_stage3_branch(tmp_path, "film")
    assert estimator_admission["stage2_aggregate_relative"] == plan.STAGE2_V3_RELATIVE + "/aggregate.json"
    assert film_admission["stage2_aggregate_relative"] == plan.STAGE2_V3_RELATIVE + "/aggregate.json"


def test_stage2_v3_public_route_held_admits_then_dispatches_seed44_only_without_data(tmp_path, monkeypatch):
    import mc_maze.dandi688_sparse_event_t4_v1.admission as admission
    import mc_maze.dandi688_sparse_event_t4_v1.production as production
    import mc_maze.dandi688_sparse_event_t4_v1.stage2_aggregate as stage2_aggregate

    events, dispatched = [], {}
    monkeypatch.setattr(stage2_aggregate, "admit_stage2_v3_successor", lambda _root: events.append("incident") or {"incident_sha256": "i" * 64})
    monkeypatch.setattr(admission, "admit_stage0_graph", lambda _root: events.append("stage0") or {"reliability_mask": [True] * 4})
    monkeypatch.setattr(production, "_execute_stage3_estimator", lambda *_args, **kwargs: dispatched.update(kwargs) or {"ok": True})
    result_root = tmp_path / plan.STAGE2_V3_RELATIVE / "estimator" / "sua" / "seed44"
    assert execute_seed_stage(tmp_path, stage=2, seed=44, view="sua", result_root=result_root) == {"ok": True}
    assert events == ["incident", "stage0"]
    assert dispatched["stage_label"] == "stage2_v3" and dispatched["stage2_admission"]["branch_gate"]["reliability_mask"] == [True] * 4
    assert not result_root.exists()
    with pytest.raises(SparseEventContractError, match="reruns only interrupted estimator seed44"):
        execute_seed_stage(tmp_path, stage=2, seed=42, view="sua", result_root=tmp_path / plan.STAGE2_V3_RELATIVE / "estimator" / "sua" / "seed42")


def test_stage2_v3_private_executor_requires_gpu_zero_before_data_or_cuda(tmp_path, monkeypatch):
    import mc_maze.dandi688_sparse_event_t4_v1.production as production

    seen = {}
    def stop_before_data(_repo_root, _seed, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("attestation stop")
    monkeypatch.setattr(production, "_attest_loads_before_cuda", stop_before_data)
    monkeypatch.setattr(production, "prepare_source_surface", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data must not open")))
    root = tmp_path / plan.STAGE2_V3_RELATIVE / "estimator" / "sua" / "seed44"
    with pytest.raises(RuntimeError, match="attestation stop"):
        production._execute_stage3_estimator(
            tmp_path, seed=44, view="sua", result_root=root,
            stage2_admission={"branch_gate": {"reliability_mask": [True] * 4}}, stage_label="stage2_v3",
        )
    assert seen["required_physical_gpu"] == "0"
    assert {item.name for item in root.iterdir()} == {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}


def test_stage3_pseudomua_final_aggregate_binds_passed_stage2_branches_and_all_six_seed_roots(tmp_path):
    source = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage2"
    for branch in ("estimator", "film"):
        for seed in plan.SEEDS:
            _publish_synthetic_stage2_seed(source / branch / "sua" / f"seed{seed}", branch=branch, seed=seed)
    stage2 = aggregate_stage2(tmp_path)
    estimator_admission = admit_stage3_branch(tmp_path, "estimator")
    film_admission = admit_stage3_branch(tmp_path, "film")
    stage3 = tmp_path / plan.STAGE3_RELATIVE
    for branch, admission in (("estimator", estimator_admission), ("film", film_admission)):
        for seed in plan.SEEDS:
            _publish_synthetic_stage2_seed(
                stage3 / branch / "pseudo_mua" / f"seed{seed}",
                branch=branch, seed=seed, stage_label="stage3", stage2_admission=admission,
            )

    final = aggregate_stage3(tmp_path)

    assert final["source_stage2_aggregate"]["sha256"] == stage2["aggregate_sha256"]
    assert final["branches"]["estimator"]["gate"] is True
    assert final["branches"]["film"]["gate"] is True
    assert final["branches"]["film"]["baseline_claim_gate"] is True
    assert final["within_view_outcome"] == "PSEUDO_MUA_FILM_REPLICATION_POSITIVE"
    target = tmp_path / plan.FINAL_AGGREGATE_RELATIVE / plan.STAGE3_PSEUDO_MUA_AGGREGATE_NAME
    assert target.is_file() and target.with_name(target.name + ".sha256").is_file()


def test_stage3_pseudomua_final_aggregate_allows_estimator_only_and_rejects_unbound_seed_receipts(tmp_path):
    source = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage2"
    for seed in plan.SEEDS:
        _publish_synthetic_stage2_seed(source / "estimator" / "sua" / f"seed{seed}", branch="estimator", seed=seed)
    aggregate_stage2(tmp_path)
    estimator_admission = admit_stage3_branch(tmp_path, "estimator")
    stage3 = tmp_path / plan.STAGE3_RELATIVE
    for seed in plan.SEEDS:
        _publish_synthetic_stage2_seed(
            stage3 / "estimator" / "pseudo_mua" / f"seed{seed}",
            branch="estimator", seed=seed, stage_label="stage3", stage2_admission=estimator_admission,
        )
    final = aggregate_stage3(tmp_path)
    assert final["branches"]["estimator"]["gate"] is True
    assert final["branches"]["film"]["status"] == "NOT_ADMITTED"
    assert final["within_view_outcome"] == "PSEUDO_MUA_ESTIMATOR_ONLY__FILM_NOT_ADMITTED_BY_SUA"

    bad = tmp_path / "bad"
    for seed in plan.SEEDS:
        _publish_synthetic_stage2_seed(
            bad / plan.RESULT_PARENT_RELATIVE / "stage2" / "estimator" / "sua" / f"seed{seed}",
            branch="estimator", seed=seed,
        )
    aggregate_stage2(bad)
    for seed in plan.SEEDS:
        _publish_synthetic_stage2_seed(
            bad / plan.STAGE3_RELATIVE / "estimator" / "pseudo_mua" / f"seed{seed}",
            branch="estimator", seed=seed, stage_label="stage3", stage2_admission={"wrong": True},
        )
    # A distinct repository retains immutable input leaves while proving that
    # an admission mismatch emits no final aggregate leaf.
    with pytest.raises(SparseEventContractError, match="parent aggregate binding drift"):
        aggregate_stage3(bad)
    assert not (bad / plan.FINAL_AGGREGATE_RELATIVE / plan.STAGE3_PSEUDO_MUA_AGGREGATE_NAME).exists()


def test_stage3_canonical_pseudomua_route_dispatches_only_after_aggregate_admission(tmp_path, monkeypatch):
    import mc_maze.dandi688_sparse_event_t4_v1.production as production

    stage2 = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage2"
    for branch in ("estimator", "film"):
        for seed in plan.SEEDS:
            _publish_synthetic_stage2_seed(stage2 / branch / "sua" / f"seed{seed}", branch=branch, seed=seed)
    aggregate_stage2(tmp_path)
    dispatched = {}
    def fake_estimator(*args, **kwargs):
        dispatched["estimator"] = {"args": args, **kwargs}
        return {"branch": "estimator"}
    def fake_film(*args, **kwargs):
        dispatched["film"] = {"args": args, **kwargs}
        return {"branch": "film"}
    monkeypatch.setattr(production, "_execute_stage3_estimator", fake_estimator)
    monkeypatch.setattr(production, "_execute_stage2_film", fake_film)
    for branch in ("estimator", "film"):
        result_root = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage3" / branch / "pseudo_mua" / "seed42"
        result = execute_seed_stage(tmp_path, stage=3, seed=42, view="pseudo_mua", result_root=result_root)
        assert result["branch"] == branch
        assert not result_root.exists()
    assert dispatched["estimator"]["stage2_admission"]["branch"] == "estimator"
    assert dispatched["film"]["branch_admission"]["branch"] == "film"
    assert dispatched["film"]["stage_label"] == "stage3"


def test_stage3_pseudomua_attempt_precedes_attestation_and_never_touches_cuda_or_data(tmp_path, monkeypatch):
    import mc_maze.dandi688_sparse_event_t4_v1.production as production

    stage2 = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage2"
    for branch in ("estimator", "film"):
        for seed in plan.SEEDS:
            _publish_synthetic_stage2_seed(stage2 / branch / "sua" / f"seed{seed}", branch=branch, seed=seed)
    aggregate_stage2(tmp_path)
    calls = []
    def stop_before_torch(repo_root, seed, **kwargs):
        calls.append((Path(repo_root), seed, kwargs))
        raise RuntimeError("attestation stop before torch/data/cuda")
    monkeypatch.setattr(production, "_attest_loads_before_cuda", stop_before_torch)
    monkeypatch.setattr(production, "prepare_source_surface", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data must not open")))
    for branch in ("estimator", "film"):
        result_root = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage3" / branch / "pseudo_mua" / "seed42"
        with pytest.raises(RuntimeError, match="attestation stop"):
            execute_seed_stage(tmp_path, stage=3, seed=42, view="pseudo_mua", result_root=result_root)
        assert (result_root / "attempt.json").is_file() and (result_root / "failure.json").is_file()
        attempt = __import__("json").loads((result_root / "attempt.json").read_text())
        assert attempt["stage"] == f"stage3_{branch}" and attempt["view"] == "pseudo_mua"
    assert calls == [
        (tmp_path.resolve(), 42, {"required_physical_gpu": "0"}),
        (tmp_path.resolve(), 42, {"required_physical_gpu": "0"}),
    ]


def test_paired_scoring_uses_three_seed_average_and_exact_10000_draw_bootstrap():
    baseline = {seed: {f"v{session}": 0.0 for session in range(6)} for seed in (42, 43, 44)}
    candidate = {seed: {f"v{session}": .02 + .001 * seed for session in range(6)} for seed in (42, 43, 44)}
    summary = average_paired_seeds(candidate, baseline)
    assert summary["positive_sessions"] == 6
    assert summary["bootstrap"]["draws"] == 10_000
    assert paired_bootstrap({f"v{index}": .01 for index in range(6)}) == paired_bootstrap({f"v{index}": .01 for index in range(6)})


def test_stage2_estimator_contract_is_two_arm_complete_student_and_paired():
    contract = stage2_estimator_contract(42)
    assert contract["arms"] == ["WHOLE-T4", "POST700-T4"]
    assert contract["complete_student_trainable"] is True
    assert contract["paired_rng_restore_before_each_arm"] is True
    assert contract["average_epochs_zero_based"] == [8, 9, 10, 11]


def test_stage2_estimator_cpu_one_batch_loop_captures_full_states_and_average():
    import torch
    model=torch.nn.Linear(1,1,bias=False)
    result=run_stage2_estimator_coordinated(parent_student=model,batches=lambda _:[(torch.ones(1,1),torch.ones(1,1))],carrier_for_arm=lambda _a,b:b[0],predict_loss=lambda m,b,c:torch.mean((m(c)-b[1])**2),score=lambda m:float(m.weight.detach().sum()),seed=42,optimizer_factory=torch.optim.SGD)
    assert set(result["averaged"]) == {"WHOLE-T4","POST700-T4"}
    assert len(result["epoch_states"]["WHOLE-T4"]) == 12
    assert len(result["epoch_receipts"]) == 12


def test_stage2_estimator_cpu_pairing_never_touches_cuda_rng(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: (_ for _ in ()).throw(AssertionError("CUDA RNG touched")))
    model=torch.nn.Linear(1,1,bias=False)
    result=run_stage2_estimator_coordinated(parent_student=model,batches=lambda _:[(torch.ones(1,1),torch.ones(1,1))],carrier_for_arm=lambda _a,b:b[0],predict_loss=lambda m,b,c:torch.mean((m(c)-b[1])**2),score=lambda m:0.0,seed=42,optimizer_factory=torch.optim.SGD)
    assert len(result["epoch_receipts"]) == 12


def test_stage2_complete_student_restores_frozen_decoder_and_decoupled_trainability():
    import torch

    class FrozenStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder=torch.nn.Linear(1,1,bias=False)
            self.decoupled_transformer=torch.nn.Linear(1,1,bias=False)
            self._decoder_frozen=True

        def train(self, mode=True):
            super().train(mode)
            if self._decoder_frozen:
                self.decoder.eval()
                self.decoupled_transformer.eval()
            return self

        def forward(self, value):
            return self.decoder(value)+self.decoupled_transformer(value)

    batch=torch.ones(1,1)
    estimator_execution=[]
    estimator=run_stage2_estimator_coordinated(
        parent_student=FrozenStudent(),
        batches=lambda _:[batch],
        carrier_for_arm=lambda _arm,_batch:batch,
        predict_loss=lambda model,_batch,carrier:(estimator_execution.append((model._decoder_frozen,model.decoder.training,model.decoupled_transformer.training)) or torch.mean((model(carrier)-1)**2)),
        score=lambda _model:0.0,
        seed=42,
        optimizer_factory=torch.optim.SGD,
    )
    film_execution=[]
    film=run_stage2_film_coordinated(
        parent_student=FrozenStudent(),
        film_factory=lambda:torch.nn.Linear(1,1,bias=False),
        batches=lambda _:[batch],
        predict_loss=lambda model,head,_arm,value:(film_execution.append((model._decoder_frozen,model.decoder.training,model.decoupled_transformer.training)) or torch.mean((model(value)+(head(value) if head is not None else 0)-1)**2)),
        seed=42,
        optimizer_factory=torch.optim.SGD,
    )
    for result, execution in ((estimator,estimator_execution),(film,film_execution)):
        assert execution and all(state == (False,True,True) for state in execution)
        for model in result["models"].values():
            assert model._decoder_frozen is False
            assert model.decoder.training is True and model.decoupled_transformer.training is True
            assert all(parameter.requires_grad for parameter in model.decoder.parameters())
            assert all(parameter.requires_grad for parameter in model.decoupled_transformer.parameters())
        assert all(row["decoder_frozen"] is False and row["decoder_training"] is True and row["decoupled_training"] is True for row in result["student_trainability"].values())


def test_stage2_full_state_artifacts_publish_two_immutable_torch_leaves(tmp_path):
    import torch
    model=torch.nn.Linear(1,1,bias=False)
    result=run_stage2_estimator_coordinated(parent_student=model,batches=lambda _:[(torch.ones(1,1),torch.ones(1,1))],carrier_for_arm=lambda _a,b:b[0],predict_loss=lambda m,b,c:torch.mean((m(c)-b[1])**2),score=lambda m:0.0,seed=42,optimizer_factory=torch.optim.SGD)
    artifact=_publish_stage2_full_state_artifacts(tmp_path,result,42)
    assert set(artifact)=={"WHOLE-T4","POST700-T4"}
    for arm in artifact:
        assert len(artifact[arm]["per_epoch_state_sha256"]) == 12
        path=tmp_path/f"stage2_estimator_{arm.lower()}.pt"
        assert path.is_file() and path.with_name(path.name+".sha256").is_file()


def test_stage2_film_cpu_coordinator_five_arms_four_heads_and_no_cuda_rng(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,"get_rng_state_all",lambda:(_ for _ in ()).throw(AssertionError("CUDA RNG touched")))
    base=torch.nn.Linear(1,1,bias=False)
    result=run_stage2_film_coordinated(parent_student=base,film_factory=lambda:torch.nn.Linear(1,1,bias=False),batches=lambda _:[torch.ones(1,1)],predict_loss=lambda model,head,name,batch:torch.mean((model(batch)+(head(batch) if head is not None else 0)-1)**2),seed=42,optimizer_factory=torch.optim.SGD)
    names={"WHOLE-NATIVE","EMPTY","PHASE-R","SE-T4","ROW-SHUFFLE"}
    assert set(result["models"]) == names and set(result["heads"]) == names
    assert set(result["averaged"]) == names and set(result["averaged_heads"]) == names-{"WHOLE-NATIVE"}
    assert all(len(result["epoch_states"][name])==12 for name in names)
    assert all(len(result["head_epoch_states"][name])==12 for name in names-{"WHOLE-NATIVE"})
    assert all(set(row["optimizer_steps"].values())=={1} for row in result["epoch_receipts"])


def test_stage2_film_zero_anchor_uses_first_governing_batch_and_restores_rng(monkeypatch):
    import random
    import torch

    monkeypatch.setattr(torch.cuda,"get_rng_state_all",lambda:(_ for _ in ()).throw(AssertionError("CUDA RNG touched")))
    random.seed(813); np.random.seed(813); torch.manual_seed(813)
    resident_batch=torch.ones(1,1)
    governing_batch_ids=[]; governing_rng={}; sentinel_modes=[]; expected_rng={}

    def film_factory():
        head=torch.nn.Linear(1,1,bias=True)
        with torch.no_grad():
            head.weight.zero_(); head.bias.zero_()
        return head

    def zero_anchor(models,heads,batch):
        assert batch is resident_batch
        sentinel_modes.extend([model.training for model in models.values()])
        sentinel_modes.extend(head.training for head in heads.values() if head is not None)
        python_state,numpy_state,torch_state=random.getstate(),np.random.get_state(),torch.random.get_rng_state()
        expected_rng["next"]=(random.random(),float(np.random.random()),float(torch.rand(())))
        random.setstate(python_state); np.random.set_state(numpy_state); torch.random.set_rng_state(torch_state)
        # Deliberately consume every paired RNG stream: the coordinator must
        # restore all three before this same batch reaches optimizer.step.
        random.random(); np.random.random(); torch.rand(())
        native=models["WHOLE-NATIVE"](batch)
        identity={}; prediction={}; direct={}
        for arm in ("EMPTY","PHASE-R","SE-T4","ROW-SHUFFLE"):
            identity[arm]=bool(torch.equal(models[arm](batch)+heads[arm](batch),native))
            prediction[arm]=bool(torch.equal(models[arm](batch)+heads[arm](batch),native))
            direct[arm]=bool(torch.equal(heads[arm].weight,torch.zeros_like(heads[arm].weight)))
        return {"passed":all(identity.values()) and all(prediction.values()) and all(direct.values()),"identity_bitwise":identity,"prediction_bitwise":prediction,"direct_native_branch":direct}

    def loss(model,head,name,batch):
        governing_batch_ids.append(id(batch))
        values=(random.random(),float(np.random.random()),float(torch.rand(())))
        if name not in governing_rng:
            governing_rng[name]=values
        prediction=model(batch)+(head(batch) if head is not None else 0)
        return torch.mean((prediction-1)**2)

    result=run_stage2_film_coordinated(
        parent_student=torch.nn.Linear(1,1,bias=False),
        film_factory=film_factory,
        batches=lambda _:[resident_batch],
        predict_loss=loss,
        seed=42,
        optimizer_factory=torch.optim.SGD,
        zero_anchor=zero_anchor,
    )

    assert result["zero_anchor"]["passed"] is True
    assert all(value is False for value in sentinel_modes)
    assert governing_batch_ids == [id(resident_batch)] * (5 * plan.EPOCHS)
    assert set(governing_rng) == {"WHOLE-NATIVE","EMPTY","PHASE-R","SE-T4","ROW-SHUFFLE"}
    assert all(values == pytest.approx(expected_rng["next"]) for values in governing_rng.values())


def test_stage2_film_heads_clone_one_factory_template_with_identical_initial_state():
    import hashlib
    import torch

    factory_calls = 0
    initial_head_bytes = {}
    initial_head_sha = {}

    def film_factory():
        nonlocal factory_calls
        factory_calls += 1
        return torch.nn.Linear(1, 1, bias=True)

    def state_bytes(module):
        payload = bytearray()
        for name, value in module.state_dict().items():
            payload.extend(name.encode("utf-8"))
            payload.extend(value.detach().cpu().numpy().tobytes())
        return bytes(payload)

    def loss(model, head, name, batch):
        if head is not None and name not in initial_head_sha:
            initial_head_bytes[name] = state_bytes(head)
            initial_head_sha[name] = hashlib.sha256(initial_head_bytes[name]).hexdigest()
        return torch.mean((model(batch) + (head(batch) if head is not None else 0) - 1) ** 2)

    run_stage2_film_coordinated(
        parent_student=torch.nn.Linear(1, 1, bias=False),
        film_factory=film_factory,
        batches=lambda _: [torch.ones(1, 1)],
        predict_loss=loss,
        seed=42,
        optimizer_factory=torch.optim.SGD,
    )

    assert factory_calls == 1
    assert set(initial_head_sha) == {"EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE"}
    assert len(set(initial_head_bytes.values())) == 1
    assert len(set(initial_head_sha.values())) == 1


def test_stage2_film_opening_admission_binds_three_stage1_v2_terminals(tmp_path):
    from mc_maze.dandi688_sparse_event_t4_v1.lifecycle import publish_immutable_json

    aggregate_root = tmp_path / plan.STAGE1_SUCCESSOR_RELATIVE
    aggregate_root.mkdir(parents=True)
    terminal_sha = {}
    for seed in plan.SEEDS:
        root = aggregate_root / "sua" / f"seed{seed}"
        root.mkdir(parents=True)
        terminal_sha[str(seed)] = publish_immutable_json(
            root,
            "terminal.json",
            {"status": "PASS", "seed": seed, "stage": "stage1_v2"},
        )
    aggregate_sha = publish_immutable_json(
        aggregate_root,
        "aggregate.json",
        {
            "status": "PASS",
            "route": plan.ROUTE_NAME,
            "stage": "stage1_v3_aggregate",
            "seeds": list(plan.SEEDS),
            "stage1_terminal_sha256": terminal_sha,
            "stage2_film_opening": {
                "status": "OPEN",
                "semantic_mean": 0.01,
                "attachment_mean": 0.01,
                "semantic_positive_sessions": 4,
            },
        },
    )

    admitted = _admit_stage2_film_opening(
        tmp_path,
        stage0_admission={"film_route": "OPEN", "reliability_mask": [True] * 4},
    )

    assert admitted["stage1_v3_aggregate_sha256"] == aggregate_sha
    assert admitted["stage1_terminal_sha256"] == terminal_sha
    assert admitted["stage2_film_opening"]["status"] == "OPEN"


def test_stage2_film_canonical_route_fails_closed_before_lifecycle_when_opening_is_absent(tmp_path, monkeypatch):
    import mc_maze.dandi688_sparse_event_t4_v1.production as production

    result_root = tmp_path / plan.RESULT_PARENT_RELATIVE / "stage2" / "film" / "sua" / "seed42"
    with pytest.raises(FileNotFoundError):
        _admit_stage2_film_opening(
            tmp_path,
            stage0_admission={"film_route": "OPEN", "reliability_mask": [True] * 4},
        )
    monkeypatch.setattr(
        production,
        "_admit_stage2_film_opening",
        lambda _repo_root: (_ for _ in ()).throw(FileNotFoundError("Stage1-v2 aggregate absent")),
    )
    with pytest.raises(FileNotFoundError):
        execute_seed_stage(tmp_path, stage=2, seed=42, view="sua", result_root=result_root)
    assert not result_root.exists()


def test_stage2_film_full_state_artifacts_publish_five_bodies_and_sidecars(tmp_path):
    import torch
    base=torch.nn.Linear(1,1,bias=False)
    result=run_stage2_film_coordinated(parent_student=base,film_factory=lambda:torch.nn.Linear(1,1,bias=False),batches=lambda _:[torch.ones(1,1)],predict_loss=lambda model,head,name,batch:torch.mean((model(batch)+(head(batch) if head is not None else 0)-1)**2),seed=42,optimizer_factory=torch.optim.SGD)
    artifacts=_publish_stage2_film_full_state_artifacts(tmp_path,result,42)
    assert len(artifacts)==5
    for arm,row in artifacts.items():
        path=tmp_path/f"stage2_film_{arm.lower()}.pt"
        assert path.is_file() and path.with_name(path.name+".sha256").is_file()
        assert len(row["base_epoch_state_sha256"])==12 and row["base_average_state_sha256"]
        if arm=="WHOLE-NATIVE": assert row["head_epoch_state_sha256"] is None and row["head_average_state_sha256"] is None
        else: assert len(row["head_epoch_state_sha256"])==12 and row["head_average_state_sha256"]


def test_all_q50_training_batches_cover_27_sessions_once_in_deterministic_chunks():
    from types import SimpleNamespace
    sessions={f"s{i:02}":SimpleNamespace(split="train",q50_starts=np.arange(65,dtype=np.int64)) for i in range(27)}
    surface=SimpleNamespace(sessions=sessions)
    first=list(_all_q50_training_batches(surface,42,3)); repeat=list(_all_q50_training_batches(surface,42,3)); other=list(_all_q50_training_batches(surface,42,4))
    assert [(n,c.tolist()) for n,_,c in first]==[(n,c.tolist()) for n,_,c in repeat]
    assert [(n,c.tolist()) for n,_,c in first] != [(n,c.tolist()) for n,_,c in other]
    seen={name:[] for name in sessions}
    for name,_row,chunk in first:
        assert chunk.size<=32;seen[name].extend(chunk.tolist())
    assert all(sorted(values)==list(range(65)) and len(values)==65 for values in seen.values())


def test_score_stage2_film_q50_scores_six_validation_sessions_with_fresh_arm_identities():
    """The scorer is CPU-safe and records Q50 evidence for all five arms."""
    import torch
    from types import SimpleNamespace

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_pool_calls = 0

        def pre_pool(self, value):
            self.pre_pool_calls += 1
            return torch.ones((value.shape[0], value.shape[1], value.shape[2], plan.HIDDEN_DIM), device=value.device)

        def post_pool(self, value):
            return value[..., : plan.HIDDEN_DIM]

    class FakeStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.id_encoder = FakeEncoder()
            self.decode_batch_sizes = []

        def decode_with_identity(self, neural, identity):
            del identity
            self.decode_batch_sizes.append(int(neural.shape[0]))
            return neural[..., :2]

    class RecordingHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_profile = None

        def forward(self, context):
            self.last_profile = context[..., 4:].detach().cpu().numpy().copy()
            return torch.zeros((*context.shape[:-1], 2 * plan.HIDDEN_DIM), dtype=context.dtype, device=context.device)

    starts = np.arange(257, dtype=np.int64)
    neural = np.arange((starts[-1] + 50) * 3, dtype=np.float32).reshape(starts[-1] + 50, 3)
    behavior = neural[:, :2] / 5.0
    sessions = {}
    for index in range(6):
        sessions[f"val{index}"] = SimpleNamespace(
            split="val",
            q50_starts=starts,
            whole_m10_parentnorm=np.full((3, 4), index + 1, dtype=np.float32),
            profile_m10=np.arange(12, dtype=np.float32).reshape(3, 4) + index,
            record=SimpleNamespace(
                calib_trials=np.ones((2, 4, 3), dtype=np.float32),
                neural=neural,
                behavior=behavior,
            ),
        )
    surface = SimpleNamespace(
        sessions=sessions,
        reliability_mask=(True, True, True, True),
        signal_view="sua",
    )
    names = ("WHOLE-NATIVE", "EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE")
    models = {name: FakeStudent() for name in names}
    heads = {name: RecordingHead() for name in names if name != "WHOLE-NATIVE"}

    scores = _score_stage2_film_q50(surface, models, heads, torch.device("cpu"), 42)

    assert set(scores) == set(names)
    for name in names:
        assert len(scores[name]) == 6
        for evidence in scores[name].values():
            assert evidence["r2"] == pytest.approx(1.0)
            assert evidence["window_count"] == starts.size
            assert len(evidence["prediction_sha256"]) == len(evidence["target_sha256"]) == len(evidence["query_sha256"]) == 64
        assert models[name].training is False
        assert models[name].id_encoder.pre_pool_calls == 6
        assert models[name].decode_batch_sizes == [256, 1] * 6
    assert all(head.training is False for head in heads.values())
    assert np.array_equal(heads["EMPTY"].last_profile, np.zeros((1, 3, 4), dtype=np.float32))
    assert np.array_equal(heads["PHASE-R"].last_profile[..., 3], np.zeros((1, 3), dtype=np.float32))
    assert np.array_equal(heads["SE-T4"].last_profile, sessions["val5"].profile_m10[None, ...])
    assert not np.array_equal(heads["ROW-SHUFFLE"].last_profile, sessions["val5"].profile_m10[None, ...])


def test_last_four_epoch_average_float64_and_frozen_state_invariance():
    epochs = [{"weight": np.asarray([float(index)], dtype=np.float32), "counter": np.asarray([9], dtype=np.int64)} for index in range(12)]
    averaged = average_last_four_trainable_states(epochs, trainable_names=("weight",), frozen_reference={"weight": epochs[0]["weight"], "counter": epochs[0]["counter"]})
    assert np.allclose(averaged["weight"], [9.5])
    assert np.array_equal(averaged["counter"], [9])


def test_frozen_design_and_root_owned_workorder_are_bound():
    repo_root = Path(__file__).resolve().parents[2]
    observed = verify_frozen_inputs(repo_root)
    assert observed[plan.DESIGN_RELATIVE] == plan.DESIGN_SHA256
    assert observed[plan.WORKORDER_RELATIVE] == plan.WORKORDER_SHA256


def test_stage0_executor_publishes_attempt_before_materializer_and_terminalizes(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    short = {"sub-C_ses-CO-20131003", "sub-C_ses-CO-20131022", "sub-C_ses-CO-20131023", "sub-C_ses-CO-20131031", "sub-C_ses-CO-20131101"}
    calls = []

    def fake_materializer(path, *, namespace, support_positions=None, **_kwargs):
        assert (tmp_path / "stage0" / "attempt.json").is_file()
        calls.append((path.name, namespace, support_positions))
        session = path.name.removesuffix("_behavior+ecephys.nwb")
        index = len(calls)
        raw = np.column_stack([np.arange(3) + index + col for col in range(4)]).astype(np.float32)
        if namespace == "reliability_audit" and support_positions == tuple(range(50, 110)):
            raw = raw.copy()
        whole = raw + (0.5 if namespace == "candidate" else 0.0)
        post = raw.copy()
        if session.startswith("sub-C_ses-CO-201511"):
            era = "long_delay"
        elif session in short:
            era = "short_delay"
        elif sum(1 for name, ns, _ in calls if ns == "candidate" and name.removesuffix("_behavior+ecephys.nwb") not in short and not name.startswith("sub-C_ses-CO-201511")) <= 9:
            era = "no_delay"
        else:
            era = "long_delay"
        return SparseEventMaterialization(session, "sua", whole, post, raw, post, raw, np.asarray([0, 1, 2]), np.asarray([0, 1, 2]), namespace, {"session_id": session, "era": era, "dense_velocity_scalars_consumed": 0})

    receipt = execute_stage0(repo_root, tmp_path / "stage0", materializer=fake_materializer)
    assert (tmp_path / "stage0" / "attempt.json").is_file()
    assert (tmp_path / "stage0" / "stage0.json").is_file()
    assert (tmp_path / "stage0" / "terminal.json").is_file()
    assert len(calls) == 33 + 27 * 3
    assert receipt["gpu_opened"] is False


def test_route_owned_cpu_preparation_replaces_only_carrier_profile_and_never_needs_cuda():
    from types import SimpleNamespace

    train, val, test = [f"tr{index}" for index in range(27)], [f"va{index}" for index in range(6)], [f"te{index}" for index in range(6)]
    def record():
        return SimpleNamespace(side_features=np.arange(12, dtype=np.float32).reshape(3, 4), valid_starts=np.arange(0, 5100, dtype=np.int64))
    train_sessions, val_sessions = ({name: record() for name in train}, {name: record() for name in val})
    train_dataset = SimpleNamespace(sessions=train_sessions, window_indices={(name, start) for name in train_sessions for start in range(0, 5100)})
    val_dataset = SimpleNamespace(sessions=val_sessions, window_indices={(name, start) for name in val_sessions for start in range(0, 5100)})
    dm = SimpleNamespace(test_dataset=None, _side_feature_stats=(np.zeros(4), np.ones(4)), session_splits={"train": train, "val": val, "test": test}, train_dataset=train_dataset, val_dataset=val_dataset)
    def descriptor(path, **kwargs):
        value = float(sum(path.name.encode()) % 17)
        matrix = np.arange(12, dtype=np.float32).reshape(3, 4) + value
        return SparseEventMaterialization(path.stem, kwargs["signal_view"], matrix, matrix + 1, matrix, matrix + 1, matrix + 2, np.asarray([0, 1, 2]), np.arange(10), kwargs["namespace"], {"dense_velocity_scalars_consumed": 0})
    trials = [{"start": index * 100, "stop": index * 100 + 100} for index in range(51)]
    surface = prepare_source_surface(Path("/unused"), signal_view="sua", reliability_mask=(1, 1, 0, 1), datamodule_factory=lambda *_: dm, descriptor_materializer=descriptor, trial_lister=lambda *_args, **_kwargs: trials)
    assert len(surface.sessions) == 33
    assert surface.sessions[train[0]].whole_m10_t4.shape == (3, 4)
    assert np.array_equal(surface.sessions[train[0]].profile_m10[:, 2], np.zeros(3, dtype=np.float32))
    assert all(row.receipt["dense_velocity_scalars_consumed"] == 0 for row in surface.sessions.values())
