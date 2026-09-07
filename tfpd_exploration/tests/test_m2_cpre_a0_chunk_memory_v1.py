"""Focused no-data/no-CUDA tests for M2 C-Pre/A0 V1."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_ROOT),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import chunk_memory, inventory, physical, plan, receipts


def _rows(seed: int = 0, channels: int = 3) -> list[np.ndarray]:
    return [np.full((100, channels), float(seed + index), dtype=np.float32) for index in range(30)]


def _memory(seed: int = 0) -> chunk_memory.SeededActivity30RollingMemory:
    return chunk_memory.SeededActivity30RollingMemory(first30_b3s=_rows(seed), carrier_support_indices=tuple(range(10)))


def _meta(surface: str, session: str, trials: int, *, window_shift: int = 0) -> inventory.ChronologyMetadata:
    starts = tuple(range(0, trials * 100, 100))
    windows = tuple(range(window_shift, trials * 100 - 10, 10))
    return inventory.ChronologyMetadata(surface, session, starts, windows)


def test_cpre_inventory_exact33_exposes_reasons_without_targets() -> None:
    record = _meta("external_post30_local", "s33", 33)
    row = inventory.inventory_session(record)
    by_n = {item["exposure"]: item for item in row["exposures"]}
    assert row["post_support_native_activity_units"] == 3
    assert by_n[60]["reason"] == "insufficient_total_past_exposure:33<=60"
    assert by_n[10]["eligible"] is True
    assert by_n["all-past"]["eligible"] is True
    assert row["target_values_read"] is False


def test_inventory_reduced_grid_is_target_free_and_canonical() -> None:
    records = [
        _meta("within_post30", "z", 50), _meta("within_post30", "a", 33),
        _meta("external_post30_local", "z", 45), _meta("external_post30_local", "a", 33),
    ]
    result = inventory.inventory_dataset(records)
    assert [item["surface"] for item in result["surfaces"]] == list(plan.SURFACE_ORDER)
    assert [item["session_id"] for item in result["surfaces"][0]["sessions"]] == ["a", "z"]
    assert result["target_values_read"] is False
    assert result["surfaces"][0]["dataset_reduced_exposure_grid"] == [10, 30, "all-past"]


def test_cpre_v2_full_w50_disjoint_post30_repairs_the_actual_coordinate_mismatch() -> None:
    """The V1 raw-boundary count is deliberately different from the V2 anchor law."""
    starts = tuple(range(0, 7000, 100))
    windows = tuple(range(0, 7000))
    record = inventory.ChronologyMetadata("within_post30", "actual-shaped", starts, windows)
    old = inventory.metadata_window_evidence(record)
    fixed = inventory.metadata_window_evidence_v2(record)
    assert fixed["post30_raw_trial_boundary"] == 3000
    assert fixed["post30_full_window_disjoint_floor"] == 3049
    assert old["post30_window_count"] == fixed["post30_window_count"] + 49
    assert old["post30_window_starts_sha256"] != fixed["post30_window_starts_sha256"]
    authority = {"actual-shaped": {
        "anchor_post30_window_count": fixed["post30_window_count"],
        "anchor_post30_window_anchor_core_sha256": fixed["post30_window_anchor_core_sha256"],
    }}
    with pytest.raises(inventory.InventoryError, match="within post30"):
        inventory.crosscheck_anchor_window_authority(
            records=(record,), surface="within_post30", authority_by_session=authority)
    verified = inventory.crosscheck_anchor_window_authority_v2(
        records=(record,), surface="within_post30", authority_by_session=authority)
    assert verified["actual-shaped"]["post30_full_window_disjoint_floor"] == 3049


def test_cpre_v2_suffix_intersects_every_exposure_with_post30_and_keeps_chunk_origin_raw() -> None:
    starts = tuple(range(0, 8000, 100))
    record = inventory.ChronologyMetadata("external_post30_local", "s", starts, tuple(range(8000)))
    rows = {row["exposure"]: row for row in inventory.inventory_session_v2(record)["exposures"]}
    assert rows[10]["suffix_start"] == 3049
    assert rows[30]["suffix_start"] == 3049
    assert rows[60]["suffix_start"] == 6049
    assert rows["all-past"]["suffix_start"] == 3049
    assert rows[10]["window_count"] == rows[30]["window_count"]
    # This is eligibility only: the raw chunk phase origin remains trial[30].
    assert starts[30] == 3000 and starts[30] != rows[30]["suffix_start"]


def test_cpre_v2_external_full_authority_is_unchanged_while_post30_is_a_subset() -> None:
    starts = tuple(range(0, 7000, 100)); windows = tuple(range(0, 7000, 2))
    record = inventory.ChronologyMetadata("external_post30_local", "ext", starts, windows)
    old, fixed = inventory.metadata_window_evidence(record), inventory.metadata_window_evidence_v2(record)
    assert old["full_window_count"] == fixed["full_window_count"]
    assert old["full_window_anchor_core_sha256"] == fixed["full_window_anchor_core_sha256"]
    assert fixed["post30_window_count"] < fixed["full_window_count"]
    authority = {"ext": {"anchor_full_window_count": fixed["full_window_count"],
                          "anchor_full_window_anchor_core_sha256": fixed["full_window_anchor_core_sha256"]}}
    result = inventory.crosscheck_anchor_window_authority_v2(
        records=(record,), surface="external_post30_local", authority_by_session=authority)
    assert result["ext"]["post30_is_deterministic_subset_of_full"] is True


def test_cpre_v2_binds_exact_v1_failure_literals_and_validates_actual_shaped_semantics() -> None:
    assert plan.CPRE_V1_FAILURE_ATTEMPT_SHA256 == "47dabd4b1f364afb8e83402c1f7f098603e6fc3c6decbe2e1248f07699038e67"
    assert plan.CPRE_V1_FAILURE_SHA256 == "f91383045b4c7ee63b37272419b753d8779bbb456928d2ada081176e922f150a"
    attempt = {"schema": "m2_cpre_attempt_v1", "status": "ATTEMPT_RESERVED_BEFORE_METADATA",
               "route": plan.ROUTE_SCHEMA, "root_relative": plan.CPRE_ROOT_RELATIVE,
               "target_access": False, "no_model_checkpoint_cuda_r2_or_target_read_before_inventory": True}
    failure = {"schema": "m2_cpre_failure_v1", "status": "FAIL_CLOSED",
               "attempt_sha256": plan.CPRE_V1_FAILURE_ATTEMPT_SHA256,
               "stage": "metadata_provider",
               "error": "InventoryError: within post30 anchor window count/digest drift",
               "progress": {"target_access": False, "target_values_read": False,
                            "model_or_checkpoint_opened": False, "cuda_initialized": False,
                            "metadata_inventory_published": False}}
    physical._validate_v1_cpre_failure_payloads(attempt=attempt, failure=failure)
    failure["progress"] = {**failure["progress"], "cuda_initialized": True}
    with pytest.raises(physical.PhysicalContractError, match="no-access"):
        physical._validate_v1_cpre_failure_payloads(attempt=attempt, failure=failure)


def test_cpre_v2_synthetic_lifecycle_is_attempt_first_and_uses_v2_schemas(tmp_path: Path) -> None:
    closure = physical.source_closure(_ROOT)
    assert closure["cpre_v2_repair_sha256"] == plan.CPRE_V2_REPAIR_SHA256
    predecessor = {"root_relative": plan.CPRE_ROOT_RELATIVE,
                   "attempt_sha256": plan.CPRE_V1_FAILURE_ATTEMPT_SHA256,
                   "failure_sha256": plan.CPRE_V1_FAILURE_SHA256}
    binding = {"root_relative": plan.CPRE_V2_ROOT_RELATIVE,
               "closure_sha256": closure["closure_sha256"],
               "v1_failure_predecessor": predecessor}
    parent = tmp_path / "receipts"; parent.mkdir()
    records = (_meta("external_post30_local", "e", 70), _meta("within_post30", "w", 70))
    result = physical.execute_cpre_v2_synthetic(
        root=parent / "m2_cpre_metadata_v2", repo_root=_ROOT,
        capability=physical._mint_test_capability(binding), records=records,
        v1_failure_predecessor=predecessor)
    descriptors = receipts.verify_topology(parent / "m2_cpre_metadata_v2",
                                           bodies=("attempt.json", "metadata_inventory.json", "terminal.json"))
    assert descriptors["attempt.json"].payload["schema"] == "m2_cpre_attempt_v2"
    assert descriptors["metadata_inventory.json"].payload["schema"] == "m2_cpre_metadata_inventory_v2"
    assert descriptors["terminal.json"].payload["schema"] == "m2_cpre_terminal_v2"
    assert result["terminal"] == descriptors["terminal.json"].sha256


def test_cpre_v2_held_descriptor_predecessor_rejects_v1_failure_topology_or_progress_drift(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the O_NOFOLLOW topology reader with a temp, actual-shaped graph."""
    repo = tmp_path / "repo"; root = repo / plan.CPRE_ROOT_RELATIVE
    root.mkdir(parents=True)
    attempt_payload = {"schema": "m2_cpre_attempt_v1", "status": "ATTEMPT_RESERVED_BEFORE_METADATA",
                       "route": plan.ROUTE_SCHEMA, "root_relative": plan.CPRE_ROOT_RELATIVE,
                       "target_access": False, "no_model_checkpoint_cuda_r2_or_target_read_before_inventory": True}
    attempt = receipts.publish_pair(root, "attempt.json", attempt_payload)
    failure = receipts.publish_pair(root, "failure.json", {
        "schema": "m2_cpre_failure_v1", "status": "FAIL_CLOSED", "attempt_sha256": attempt.sha256,
        "stage": "metadata_provider", "error": "InventoryError: within post30 anchor window count/digest drift",
        "progress": {"target_access": False, "target_values_read": False,
                     "model_or_checkpoint_opened": False, "cuda_initialized": False,
                     "metadata_inventory_published": False},
    })
    monkeypatch.setattr(plan, "CPRE_V1_FAILURE_ATTEMPT_SHA256", attempt.sha256)
    monkeypatch.setattr(plan, "CPRE_V1_FAILURE_SHA256", failure.sha256)
    witness = physical._future_v1_cpre_failure_predecessor(repo)
    assert witness["attempt_sha256"] == attempt.sha256 and witness["failure_sha256"] == failure.sha256
    (root / "failure.json.sha256").chmod(0o644)
    with pytest.raises(receipts.ReceiptError, match="0444"):
        physical._future_v1_cpre_failure_predecessor(repo)


def test_cpre_v2_post_metadata_failure_preserves_only_complete_immutable_prefix(tmp_path: Path) -> None:
    closure = physical.source_closure(_ROOT)
    predecessor = {"root_relative": plan.CPRE_ROOT_RELATIVE,
                   "attempt_sha256": plan.CPRE_V1_FAILURE_ATTEMPT_SHA256,
                   "failure_sha256": plan.CPRE_V1_FAILURE_SHA256}
    root_parent = tmp_path / "receipts"; root_parent.mkdir(); root = root_parent / "m2_cpre_metadata_v2"
    with pytest.raises(physical.PhysicalContractError, match="synthetic lifecycle"):
        physical.execute_cpre_v2_synthetic(
            root=root, repo_root=_ROOT,
            capability=physical._mint_test_capability({
                "root_relative": plan.CPRE_V2_ROOT_RELATIVE,
                "closure_sha256": closure["closure_sha256"], "v1_failure_predecessor": predecessor}),
            records=(_meta("external_post30_local", "e", 70), _meta("within_post30", "w", 70)),
            v1_failure_predecessor=predecessor, fail_after_metadata_for_test=True)
    descriptors = receipts.verify_terminal_xor(
        root, success_bodies=("attempt.json", "metadata_inventory.json", "terminal.json"))
    assert set(descriptors) == {"attempt.json", "metadata_inventory.json", "failure.json"}
    assert descriptors["failure.json"].payload["schema"] == "m2_cpre_failure_v2"


def test_a0_accepts_historical_cpre_v2_closure_without_equating_it_to_current_a0_closure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A current transport-only closure change cannot force a C-Pre rerun."""
    repo = tmp_path / "repo"
    v1_root = repo / plan.CPRE_ROOT_RELATIVE; v1_root.mkdir(parents=True)
    v1_attempt = receipts.publish_pair(v1_root, "attempt.json", {
        "schema": "m2_cpre_attempt_v1", "status": "ATTEMPT_RESERVED_BEFORE_METADATA",
        "route": plan.ROUTE_SCHEMA, "root_relative": plan.CPRE_ROOT_RELATIVE,
        "target_access": False, "no_model_checkpoint_cuda_r2_or_target_read_before_inventory": True})
    v1_failure = receipts.publish_pair(v1_root, "failure.json", {
        "schema": "m2_cpre_failure_v1", "status": "FAIL_CLOSED", "attempt_sha256": v1_attempt.sha256,
        "stage": "metadata_provider", "error": "InventoryError: within post30 anchor window count/digest drift",
        "progress": {"target_access": False, "target_values_read": False,
                     "model_or_checkpoint_opened": False, "cuda_initialized": False,
                     "metadata_inventory_published": False}})
    monkeypatch.setattr(plan, "CPRE_V1_FAILURE_ATTEMPT_SHA256", v1_attempt.sha256)
    monkeypatch.setattr(plan, "CPRE_V1_FAILURE_SHA256", v1_failure.sha256)
    predecessor = {"root_relative": plan.CPRE_ROOT_RELATIVE, "attempt_sha256": v1_attempt.sha256,
                   "failure_sha256": v1_failure.sha256}
    v2_root = repo / plan.CPRE_V2_ROOT_RELATIVE; v2_root.mkdir(parents=True)
    historical_closure = {"closure_sha256": "c" * 64, "historical": "CPU-only-CPre"}
    v2_attempt = receipts.publish_pair(v2_root, "attempt.json", {
        "schema": "m2_cpre_attempt_v2", "status": "ATTEMPT_RESERVED_BEFORE_METADATA",
        "root_relative": plan.CPRE_V2_ROOT_RELATIVE, "v1_failure_predecessor": predecessor})
    inventory_body = receipts.publish_pair(v2_root, "metadata_inventory.json", {
        "schema": "m2_cpre_metadata_inventory_v2", "target_values_read": False,
        "window_coordinate_law": "endpoint_s_ge_raw_trial_start_n_plus_49"})
    terminal = receipts.publish_pair(v2_root, "terminal.json", {
        "schema": "m2_cpre_terminal_v2", "status": "TERMINAL", "attempt_sha256": v2_attempt.sha256,
        "metadata_inventory_sha256": inventory_body.sha256, "v1_failure_predecessor": predecessor,
        "target_access": False, "target_values_read": False, "model_or_checkpoint_opened": False,
        "cuda_initialized": False, "source_closure": historical_closure})
    monkeypatch.setattr(plan, "CPRE_V2_HISTORICAL_ATTEMPT_SHA256", v2_attempt.sha256)
    monkeypatch.setattr(plan, "CPRE_V2_HISTORICAL_INVENTORY_SHA256", inventory_body.sha256)
    monkeypatch.setattr(plan, "CPRE_V2_HISTORICAL_TERMINAL_SHA256", terminal.sha256)
    monkeypatch.setattr(plan, "CPRE_V2_HISTORICAL_CLOSURE_SHA256", "c" * 64)
    current = physical.source_closure(_ROOT)
    witness = physical._future_cpre_completion(repo_root=repo, closure=current)
    assert witness["terminal_sha256"] == terminal.sha256
    assert witness["historical_cpre_closure_sha256"] == "c" * 64
    # A0's own capability still rejects a changed current closure independently.
    cap = physical._mint_test_capability({"root_relative": plan.A0_ROOTS["external_post30_local"],
                                          "closure_sha256": current["closure_sha256"]})
    bad_current = {**current, "closure_sha256": "0" * 64}
    with pytest.raises(physical.PhysicalContractError, match="closure"):
        physical._consume(cap, expected_root=plan.A0_ROOTS["external_post30_local"], closure=bad_current)
    (v2_root / "metadata_inventory.json.sha256").chmod(0o644)
    with pytest.raises(receipts.ReceiptError, match="0444"):
        physical._future_cpre_completion(repo_root=repo, closure=current)


def test_root_only_cpu_aggregate_issuer_binds_completed_canonical_shards_without_torch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"; current = physical.source_closure(_ROOT)
    roots = {"external_post30_local": "results/external", "within_post30": "results/within",
             "aggregate": "results/aggregate"}
    monkeypatch.setattr(plan, "A0_ROOTS", roots)
    monkeypatch.setattr(physical, "source_closure", lambda _: current)
    for surface in ("external_post30_local", "within_post30"):
        root = repo / roots[surface]; root.mkdir(parents=True)
        attempt = receipts.publish_pair(root, "attempt.json", {"schema": "test"})
        receipts.publish_pair(root, "launch.json", {"schema": "test"})
        receipts.publish_pair(root, "input_authority.json", {"schema": "test"})
        replay = receipts.publish_pair(root, "replay.json", {"schema": "test", "surface": surface})
        receipts.publish_pair(root, "terminal.json", {
            "schema": "m2_a0_terminal_v1", "status": "TERMINAL", "attempt_sha256": attempt.sha256,
            "replay_sha256": replay.sha256, "source_closure": current})
    cap = physical._issue_live_aggregate_capability(repo_root=repo, root=repo / roots["aggregate"])
    assert cap._binding["cpu_only"] is True
    assert cap._binding["external_shard"]["root_relative"] == roots["external_post30_local"]
    with pytest.raises(physical.PhysicalContractError, match="closure"):
        physical._completed_a0_shard_witness(
            repo_root=repo, surface="external_post30_local", closure={**current, "closure_sha256": "0" * 64})


def test_metadata_only_nwb_provider_derives_w50_windows_without_behavior_values(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "synthetic.nwb"
    timestamps = np.arange(5000, dtype=np.float64) * .02
    mask = np.arange(5000) % 3 != 0
    starts = timestamps[np.arange(0, 4000, 100)]
    with h5py.File(path, "w") as handle:
        finger = handle.create_group("acquisition/finger_vel")
        for key in ("index", "mrs"):
            member = finger.create_group(key)
            member.create_dataset("timestamps", data=timestamps)
            # A target-value looking dataset exists but the provider has no path
            # to it; this makes accidental finger_vel.data access fail in h5py.
            member.create_dataset("data", data=np.full((5000,), np.nan))
        handle.create_dataset("acquisition/eval_mask/data", data=mask)
        handle.create_dataset("intervals/trials/start_time", data=starts)
    provisional = inventory.ChronologyMetadata("external_post30_local", "synthetic", tuple(range(0, 4000, 100)),
                                                tuple(np.flatnonzero(mask)))
    evidence = inventory.metadata_window_evidence(provisional)
    authority = {"synthetic": {"anchor_full_window_count": evidence["full_window_count"],
                                "anchor_full_window_anchor_core_sha256": evidence["full_window_anchor_core_sha256"]}}
    records = inventory.metadata_only_nwb_records(
        paths=(path,), surface="external_post30_local",
        resolved_metadata_facts={"task": "m2", "window_size": 50, "use_intertrials": True,
                                 "remove_still_times": False, "resolved_config_sha256": "a" * 64},
        anchor_window_authority=authority, session_name=lambda _: "synthetic",
    )
    assert records == (provisional,)
    bad = {"synthetic": {"anchor_full_window_count": 1,
                          "anchor_full_window_anchor_core_sha256": "0" * 64}}
    with pytest.raises(inventory.InventoryError, match="anchor"):
        inventory.metadata_only_nwb_records(
            paths=(path,), surface="external_post30_local",
            resolved_metadata_facts={"task": "m2", "window_size": 50, "use_intertrials": True,
                                     "remove_still_times": False, "resolved_config_sha256": "a" * 64},
            anchor_window_authority=bad, session_name=lambda _: "synthetic")


def test_cpre_metadata_only_production_entry_is_capability_bound(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    paths: dict[str, list[Path]] = {}
    authority: dict[str, dict[str, dict[str, object]]] = {}
    for surface, name in (("external_post30_local", "external"), ("within_post30", "within")):
        path = tmp_path / f"{name}.nwb"; paths[surface] = [path]
        time_axis = np.arange(5000, dtype=np.float64) * .02
        mask = np.ones(5000, dtype=bool); starts = time_axis[np.arange(0, 4000, 100)]
        with h5py.File(path, "w") as handle:
            component = handle.create_group("acquisition/finger_vel").create_group("index")
            component.create_dataset("timestamps", data=time_axis)
            handle.create_dataset("acquisition/eval_mask/data", data=mask)
            handle.create_dataset("intervals/trials/start_time", data=starts)
        record = inventory.ChronologyMetadata(surface, name, tuple(range(0, 4000, 100)), tuple(np.flatnonzero(mask)))
        item = inventory.metadata_window_evidence(record)
        authority[surface] = {name: ({"anchor_full_window_count": item["full_window_count"],
                                      "anchor_full_window_anchor_core_sha256": item["full_window_anchor_core_sha256"]}
                                     if surface == "external_post30_local" else
                                     {"anchor_post30_window_count": item["post30_window_count"],
                                      "anchor_post30_window_anchor_core_sha256": item["post30_window_anchor_core_sha256"]})}
    facts = {"task": "m2", "window_size": 50, "use_intertrials": True,
             "remove_still_times": False, "resolved_config_sha256": "a" * 64}
    closure = physical.source_closure(_ROOT)
    binding = {"root_relative": plan.CPRE_ROOT_RELATIVE, "closure_sha256": closure["closure_sha256"],
               "metadata_facts": facts, "anchor_window_authority_sha256": physical._canonical_json_sha(authority)}
    root_parent = tmp_path / "receipts"; root_parent.mkdir()
    result = physical.execute_cpre_metadata_only(
        root=root_parent / "m2_cpre_metadata_v1", repo_root=_ROOT,
        capability=physical._mint_test_capability(binding), paths_by_surface=paths,
        resolved_metadata_facts=facts, anchor_window_authority=authority,
        session_name=lambda path: path.stem,
    )
    assert len(result["terminal"]) == 64


def test_chunk_phase0_is_raw_boundary_aligned_and_predict_then_updates() -> None:
    memory = _memory()
    stream = chunk_memory.FixedChunkStream(memory=memory, phase=0, calibration_boundary_bin=3000)
    initial = stream.begin_causal_bin(3000)
    assert initial == memory.identity_sha256()
    for coordinate in range(3000, 3099):
        stream.append_after_causal_bin(coordinate, np.ones(3, np.float32))
        stream.begin_causal_bin(coordinate + 1)
    update = stream.append_after_causal_bin(3099, np.ones(3, np.float32))
    assert update is not None and update.coordinate == 3099
    assert stream.completed_chunks == 1 and memory.update_count == 1
    assert stream.finish()["partial_final_bins_discarded"] == 0


def test_chunk_phase50_skips_raw_prefix_but_never_uses_metric_or_target() -> None:
    memory = _memory()
    stream = chunk_memory.FixedChunkStream(memory=memory, phase=50, calibration_boundary_bin=100)
    for coordinate in range(100, 249):
        stream.begin_causal_bin(coordinate)
        update = stream.append_after_causal_bin(coordinate, np.arange(3, dtype=np.float32))
        assert update is None
    stream.begin_causal_bin(249)
    update = stream.append_after_causal_bin(249, np.arange(3, dtype=np.float32))
    assert update is not None and update.coordinate == 249
    assert stream.completed_chunks == 1


def test_chunk_rejects_skips_duplicates_and_partial_discard_is_counted() -> None:
    stream = chunk_memory.FixedChunkStream(memory=_memory(), phase=0, calibration_boundary_bin=0)
    stream.begin_causal_bin(0)
    stream.append_after_causal_bin(0, np.zeros(3, np.float32))
    with pytest.raises(chunk_memory.ChunkMemoryError, match="skipped or duplicated"):
        stream.begin_causal_bin(2)
    for coordinate in range(1, 18):
        stream.begin_causal_bin(coordinate)
        stream.append_after_causal_bin(coordinate, np.zeros(3, np.float32))
    assert stream.finish()["partial_final_bins_discarded"] == 18


def test_chunk_streams_are_independent_and_reset_only_after_done() -> None:
    left = chunk_memory.FixedChunkStream(memory=_memory(1), phase=0, calibration_boundary_bin=0)
    right = chunk_memory.FixedChunkStream(memory=_memory(2), phase=50, calibration_boundary_bin=0)
    with pytest.raises(chunk_memory.ChunkMemoryError, match="only after done"):
        left.reset()
    for coordinate in range(3):
        left.begin_causal_bin(coordinate); left.append_after_causal_bin(coordinate, np.ones(3, np.float32))
        right.begin_causal_bin(coordinate); right.append_after_causal_bin(coordinate, np.ones(3, np.float32))
    assert left.memory.identity_sha256() != right.memory.identity_sha256()
    left.finish(); left.reset()
    assert left.completed_chunks == 0 and not left.done


def test_m10_carrier_authority_is_separate_from_m30_activity_capacity() -> None:
    memory = _memory()
    assert memory.retained_count == 30
    assert memory.carrier_support_indices == tuple(range(10))
    with pytest.raises(chunk_memory.ChunkMemoryError, match="exactly ten"):
        chunk_memory.SeededActivity30RollingMemory(first30_b3s=_rows(), carrier_support_indices=tuple(range(30)))


def test_first_prediction_parity_validator_actual_shaped_rows() -> None:
    digest = "a" * 64
    rows = []
    for arm, phase, r2 in (("static", None, .2), ("true_trial", None, .3), ("chunk", 0, .296), ("chunk", 50, .28)):
        rows.append(physical.A0Row(
            surface="external_post30_local", session_id="s", arm=arm, phase=phase, r2=r2,
            first_prediction_sha256=digest, prediction_sha256=digest,
            initial_b3s_identity_sha256=digest, final_b3s_identity_sha256=digest,
            initial_activity_stack_sha256=digest, final_activity_stack_sha256=digest,
            query_starts_sha256=digest, target_window_sha256=digest, window_count=10,
            update_count=0, eviction_count=0, partial_final_bins_discarded=0,
            parameter_updates=0, target_gradients=0, wall_seconds=0.0, peak_memory_bytes=0,
        ))
    output = physical.validate_a0_rows(rows, surface="external_post30_local", roster=("s",))
    assert output["gate"]["external_noninferiority_mean"] is True
    bad = list(rows); bad[3] = physical.A0Row(
        surface="external_post30_local", session_id="s", arm="chunk", phase=50, r2=.28,
        first_prediction_sha256="b" * 64, prediction_sha256=digest,
        initial_b3s_identity_sha256=digest, final_b3s_identity_sha256=digest,
        initial_activity_stack_sha256=digest, final_activity_stack_sha256=digest,
        query_starts_sha256=digest, target_window_sha256=digest, window_count=10,
        update_count=0, eviction_count=0, partial_final_bins_discarded=0,
        parameter_updates=0, target_gradients=0, wall_seconds=0.0, peak_memory_bytes=0,
    )
    with pytest.raises(physical.PhysicalContractError, match="parity"):
        physical.validate_a0_rows(bad, surface="external_post30_local", roster=("s",))


def test_static_profiles_pin_only_gpu0_and_surface_affinity() -> None:
    assert physical.static_device_profile()["physical_index"] == 0
    assert physical.static_device_profile()["uuid"] == plan.GPU0_UUID
    assert physical.scheduler_profile("external_post30_local")["cpu_affinity"] == [0, 1, 2, 3, 16, 17, 18, 19]
    with pytest.raises(physical.PhysicalContractError):
        physical.DeviceAttestation(1, "1", "GPU1", "cuda:0", False).validate()


def test_gpu0_uuid_transport_normalizer_accepts_only_the_exact_optional_prefix() -> None:
    canonical = plan.GPU0_UUID
    bare = canonical.removeprefix("GPU-")
    assert physical.normalize_gpu0_uuid(canonical) == {
        "raw_uuid": canonical, "canonical_uuid": canonical}
    assert physical.normalize_gpu0_uuid(bare) == {
        "raw_uuid": bare, "canonical_uuid": canonical}
    for drift in ("GPU-" + bare + "x", bare.upper(), "GPU1", "", "GPU-GPU-" + bare):
        with pytest.raises(physical.PhysicalContractError, match="UUID"):
            physical.normalize_gpu0_uuid(drift)


def test_runtime_scheduler_attestation_requires_actual_env_and_affinity() -> None:
    profile = physical.scheduler_profile("external_post30_local")
    environment = {
        "CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    observed = physical.attest_a0_runtime_scheduler(
        surface="external_post30_local", environ=environment, affinity=profile["cpu_affinity"], pid=123)
    assert observed["pid"] == 123 and observed["cpu_affinity"] == profile["cpu_affinity"]
    with pytest.raises(physical.PhysicalContractError, match="affinity"):
        physical.attest_a0_runtime_scheduler(
            surface="external_post30_local", environ=environment, affinity=[0], pid=123)
    bad = dict(environment); bad["CUDA_VISIBLE_DEVICES"] = "1"
    with pytest.raises(physical.PhysicalContractError, match="visibility"):
        physical.attest_a0_runtime_scheduler(
            surface="external_post30_local", environ=bad, affinity=profile["cpu_affinity"], pid=123)


def test_precomputed_identity_adapter_cpu_actual_shaped_and_no_cuda() -> None:
    torch = pytest.importorskip("torch")
    class Student:
        decoder_mode = "coupled"
        def decode_with_identity(self, neural, identity):
            # [B,50,N] -> [B,50,2], shape matching frozen velocity output.
            return neural[..., :2] + identity[:, :1, :2]
    neural = np.arange(3 * 50 * 3, dtype=np.float32).reshape(3, 50, 3)
    identity = torch.zeros((1, 1, 2), dtype=torch.float32)
    output = physical.decode_precomputed_identity_gpu_safe(
        torch=torch, student=Student(), neural_windows=neural, identity=identity,
        device=torch.device("cpu"), batch_size=2, behavior_scale=1.0,
    )
    assert output.shape == (3, 2) and np.all(np.isfinite(output))
    with pytest.raises(physical.PhysicalContractError, match="W50"):
        physical.decode_precomputed_identity_gpu_safe(
            torch=torch, student=Student(), neural_windows=np.zeros((1, 100, 3), dtype=np.float32),
            identity=identity, device=torch.device("cpu"), batch_size=2, behavior_scale=1.0,
        )
    assert torch.cuda.is_initialized() is False


def test_identity_snapshot_batching_matches_eager_prediction_and_r2_cpu() -> None:
    """Batching may change forward grouping, never endpoint state/order/output."""
    torch = pytest.importorskip("torch")

    class Student:
        decoder_mode = "coupled"

        def decode_with_identity(self, neural, identity):
            return neural[..., :2] * 0.25 + identity[:, :1, :2]

    source = np.arange(800 * 3, dtype=np.float32).reshape(800, 3) / 100.0
    identity_a = torch.zeros((1, 1, 2), dtype=torch.float32)
    identity_b = torch.full((1, 1, 2), 0.5, dtype=torch.float32)
    requests = tuple(
        physical.IdentityDecodeRequest(
            ordinal=index, raw_window_start=start,
            identity=identity_a if index < 3 or index == 5 else identity_b,
            identity_sha256=("a" if index < 3 or index == 5 else "b") * 64,
        )
        for index, start in enumerate((100, 150, 200, 250, 300, 350))
    )
    eager, eager_evidence = physical.decode_identity_snapshot_segments(
        torch=torch, student=Student(), neural_source=source, requests=requests,
        device=torch.device("cpu"), batch_size=1, behavior_scale=1.0,
        array_sha256=lambda array: physical._canonical_json_sha({"starts": np.asarray(array).tolist()}),
    )
    batched, batched_evidence = physical.decode_identity_snapshot_segments(
        torch=torch, student=Student(), neural_source=source, requests=requests,
        device=torch.device("cpu"), batch_size=4, behavior_scale=1.0,
        array_sha256=lambda array: physical._canonical_json_sha({"starts": np.asarray(array).tolist()}),
    )
    assert np.max(np.abs(eager - batched)) <= 1e-6
    target = np.linspace(-1.0, 1.0, eager.size, dtype=np.float32).reshape(eager.shape)
    r2_eager = 1.0 - float(np.square(target - eager).sum() / np.square(target - target.mean(axis=0)).sum())
    r2_batched = 1.0 - float(np.square(target - batched).sum() / np.square(target - target.mean(axis=0)).sum())
    assert r2_eager == r2_batched
    assert eager_evidence["actual_scored_prediction_count"] == batched_evidence["actual_scored_prediction_count"] == 6
    assert [segment["pre_update_identity_sha256"] for segment in batched_evidence["identity_segments"]] == [
        "a" * 64, "a" * 64, "b" * 64, "a" * 64,
    ]
    assert batched_evidence["first_prediction_parity_singleton_batch"] is True
    assert torch.cuda.is_initialized() is False


def test_cubic_reconstruction_uses_full_raw_trial_not_eval_mask() -> None:
    pytest.importorskip("scipy")
    raw = np.stack((np.arange(8000, dtype=np.float32), np.arange(8000, dtype=np.float32) ** 2), axis=1)
    starts = np.arange(0, 31 * 250, 250, dtype=np.int64)
    # An adversarial evaluation mask is present in the raw authority but is
    # intentionally irrelevant to sealed calibration cubic materialization.
    rebuilt = physical._cubic_activity_reconstruction(
        raw_session={"neural": raw, "eval_mask": np.zeros(8000, dtype=bool)}, raw_starts=starts)
    assert rebuilt.shape == (31, 100, 2)
    assert np.isfinite(rebuilt).all()
    assert not np.allclose(rebuilt[0], 0.0)


def test_receipt_descriptor_mode_sidecar_topology_and_corruption(tmp_path: Path) -> None:
    root = tmp_path / "root"; root.mkdir()
    desc = receipts.publish_pair(root, "attempt.json", {"x": 1})
    assert receipts.descriptor_read(root, "attempt.json").sha256 == desc.sha256
    os.chmod(root / "attempt.json", 0o644)
    with pytest.raises(receipts.ReceiptError, match="0444"):
        receipts.descriptor_read(root, "attempt.json")
    os.chmod(root / "attempt.json", 0o444)
    (root / "extra").write_text("x")
    with pytest.raises(receipts.ReceiptError, match="topology"):
        receipts.verify_topology(root, bodies=("attempt.json",))


def test_failure_is_attempt_plus_failure_only_and_never_overwrites_partial(tmp_path: Path) -> None:
    root = tmp_path / "failure"; root.mkdir()
    attempt = receipts.publish_pair(root, "attempt.json", {"attempt": True})
    failure = receipts.publish_failure_after_attempt(
        root=root, attempt_sha256=attempt.sha256, schema="m2_test_failure_v1", stage="pre_runtime",
        error="synthetic", progress={"data_opened": False, "cuda_initialized": False},
    )
    assert failure.payload["status"] == "FAIL_CLOSED"
    receipts.verify_terminal_xor(root, success_bodies=("attempt.json", "terminal.json"))
    root2 = tmp_path / "partial"; root2.mkdir()
    attempt2 = receipts.publish_pair(root2, "attempt.json", {"attempt": True})
    receipts.publish_pair(root2, "launch.json", {"launch": True})
    with pytest.raises(receipts.ReceiptError, match="topology"):
        receipts.publish_failure_after_attempt(root=root2, attempt_sha256=attempt2.sha256,
                                               schema="x", stage="late", error="x", progress={})
    preserved = receipts.publish_failure_preserving_prefix(
        root=root2, prefix_bodies=("attempt.json", "launch.json"), attempt_sha256=attempt2.sha256,
        schema="m2_a0_failure_v1", stage="launch", error="synthetic", progress={"target_updates": 0})
    assert preserved.payload["retained_immutable_prefix"]["launch.json"]
    verified = receipts.verify_terminal_xor(root2, success_bodies=(
        "attempt.json", "launch.json", "input_authority.json", "replay.json", "terminal.json"))
    assert set(verified) == {"attempt.json", "launch.json", "failure.json"}


def test_structured_predecessor_semantics_rejects_wrong_selected_or_negative_verdict() -> None:
    payloads = {
        "memory_law": {
            "schema": "m2_memory_law_scan_v1_terminal_v1", "status": "TERMINAL",
            "official_contract_claimed": False,
            "verdict": {"winner": "UNIFORM_UNCAPPED", "verdict": "UNCAPPED_WINS",
                        "eligible_policies": ["UNIFORM_UNCAPPED"]},
            "gates": {"policy_results": {"UNIFORM_UNCAPPED": {
                "disposition": "SELECTED_ELIGIBLE", "primary_passed": True, "safety_passed": True}}},
        },
        "chrono4": {
            "schema": "m2_chrono4_strict_v1_terminal_v1", "status": "TERMINAL",
            "official_contract_claimed": False, "headline": {"verdict": "SELECTION_CONTRIBUTION_LARGE"},
            "selection_contribution_dopt_minus_chrono4": {"cdm_family": {"external_post30_local": {
                "equal_session_mean_delta": .1, "positive_sessions_dopt_better": 5}}},
        },
        "reblock10": {
            "schema": "m2_reblock10_v1_terminal_v1", "status": "TERMINAL",
            "official_contract_claimed": False, "primary_verdict": "REBLOCK_CDM_NOT_WORTH",
            "primary_gate_cdm_k4_external": {"verdict": "REBLOCK_CDM_NOT_WORTH", "delta": -.02,
                                               "positive_sessions": 2, "breadth_denominator": 6},
        },
    }
    receipts.semantic_predecessor_validator(payloads)
    payloads["memory_law"] = {**payloads["memory_law"], "verdict": {"winner": "EMA_A095"}}
    with pytest.raises(receipts.ReceiptError, match="selected winner"):
        receipts.semantic_predecessor_validator(payloads)


def test_cpre_synthetic_lifecycle_is_attempt_first_and_one_shot(tmp_path: Path) -> None:
    closure = physical.source_closure(_ROOT)
    binding = {"root_relative": plan.CPRE_ROOT_RELATIVE, "closure_sha256": closure["closure_sha256"]}
    cap = physical._mint_test_capability(binding)
    parent = tmp_path / "results"; parent.mkdir()
    root = parent / "m2_cpre_metadata_v1"
    records = [_meta("external_post30_local", "e", 33), _meta("within_post30", "w", 33)]
    result = physical.execute_cpre_synthetic(root=root, repo_root=_ROOT, capability=cap, records=records)
    assert len(result["terminal"]) == 64
    assert stat.S_IMODE((root / "terminal.json").stat().st_mode) == 0o444
    with pytest.raises(physical.PhysicalContractError, match="already consumed"):
        physical.execute_cpre_synthetic(root=root, repo_root=_ROOT, capability=cap, records=records)


def test_cpre_post_metadata_failure_retains_only_complete_immutable_prefix(tmp_path: Path) -> None:
    closure = physical.source_closure(_ROOT)
    binding = {"root_relative": plan.CPRE_ROOT_RELATIVE, "closure_sha256": closure["closure_sha256"]}
    parent = tmp_path / "results"; parent.mkdir()
    root = parent / "m2_cpre_metadata_v1"
    with pytest.raises(physical.PhysicalContractError, match="synthetic lifecycle failed"):
        physical.execute_cpre_synthetic(
            root=root, repo_root=_ROOT, capability=physical._mint_test_capability(binding),
            records=[_meta("external_post30_local", "e", 33), _meta("within_post30", "w", 33)],
            fail_after_metadata_for_test=True,
        )
    descriptors = receipts.verify_terminal_xor(root, success_bodies=(
        "attempt.json", "metadata_inventory.json", "terminal.json"))
    assert set(descriptors) == {"attempt.json", "metadata_inventory.json", "failure.json"}
    failure = descriptors["failure.json"].payload
    assert failure["stage"] == "terminal"
    assert failure["retained_immutable_prefix"] == {
        "attempt.json": descriptors["attempt.json"].sha256,
        "metadata_inventory.json": descriptors["metadata_inventory.json"].sha256,
    }


def test_cpre_metadata_factory_cannot_run_before_immutable_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    facts = {"task": "m2", "window_size": 50, "use_intertrials": True,
             "remove_still_times": False, "resolved_config_sha256": "a" * 64}
    records = {
        "external_post30_local": inventory.ChronologyMetadata("external_post30_local", "e", tuple(range(0, 4000, 100)), tuple(range(5000))),
        "within_post30": inventory.ChronologyMetadata("within_post30", "w", tuple(range(0, 4000, 100)), tuple(range(5000))),
    }
    authority: dict[str, dict[str, dict[str, object]]] = {}
    for surface, record in records.items():
        evidence = inventory.metadata_window_evidence(record)
        authority[surface] = {record.session_id: (
            {"anchor_full_window_count": evidence["full_window_count"],
             "anchor_full_window_anchor_core_sha256": evidence["full_window_anchor_core_sha256"]}
            if surface == "external_post30_local" else
            {"anchor_post30_window_count": evidence["post30_window_count"],
             "anchor_post30_window_anchor_core_sha256": evidence["post30_window_anchor_core_sha256"]}
        )}
    parent = tmp_path / "receipts"; parent.mkdir(); root = parent / "m2_cpre_metadata_v1"
    observed: list[str] = []
    def factory(*, paths, surface, resolved_metadata_facts, anchor_window_authority, session_name=None):
        assert (root / "attempt.json").is_file()
        observed.append(surface)
        return (records[surface],)
    monkeypatch.setattr(inventory, "metadata_only_nwb_records", factory)
    closure = physical.source_closure(_ROOT)
    binding = {"root_relative": plan.CPRE_ROOT_RELATIVE, "closure_sha256": closure["closure_sha256"],
               "metadata_facts": facts, "anchor_window_authority_sha256": physical._canonical_json_sha(authority)}
    physical.execute_cpre_metadata_only(
        root=root, repo_root=_ROOT, capability=physical._mint_test_capability(binding),
        paths_by_surface={surface: [tmp_path / f"{surface}.nwb"] for surface in plan.SURFACE_ORDER},
        resolved_metadata_facts=facts, anchor_window_authority=authority,
    )
    assert observed == list(plan.SURFACE_ORDER)


def test_capability_wrong_root_and_closure_fail_closed(tmp_path: Path) -> None:
    closure = physical.source_closure(_ROOT)
    cap = physical._mint_test_capability({"root_relative": "wrong", "closure_sha256": closure["closure_sha256"]})
    with pytest.raises(physical.PhysicalContractError, match="root"):
        physical._consume(cap, expected_root=plan.CPRE_ROOT_RELATIVE, closure=closure)
    cap = physical._mint_test_capability({"root_relative": plan.CPRE_ROOT_RELATIVE, "closure_sha256": "0" * 64})
    with pytest.raises(physical.PhysicalContractError, match="closure"):
        physical._consume(cap, expected_root=plan.CPRE_ROOT_RELATIVE, closure=closure)


def test_live_root_witness_rejects_noncanonical_or_parent_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; parent = repo / "results"; parent.mkdir(parents=True)
    relative = "results/live"
    root = repo / relative
    witness = physical._prospective_root_witness(repo_root=repo, relative=relative)
    physical._verify_prospective_root_witness(
        witness=witness, repo_root=repo, expected_relative=relative, supplied_root=root)
    with pytest.raises(physical.PhysicalContractError, match="noncanonical"):
        physical._verify_prospective_root_witness(
            witness=witness, repo_root=repo, expected_relative=relative, supplied_root=tmp_path / "other")
    moved = tmp_path / "moved"; parent.rename(moved); parent.mkdir()
    with pytest.raises(physical.PhysicalContractError, match="parent"):
        physical._verify_prospective_root_witness(
            witness=witness, repo_root=repo, expected_relative=relative, supplied_root=root)


def _anchor_payload() -> dict[str, object]:
    return {
        "schema": "m2_t4_activity_budget_screen_v1", "status": "TERMINAL",
        "checkpoint_sha256": plan.ANCHOR_CHECKPOINT_SHA256,
        "normalization_sha256": plan.ANCHOR_NORMALIZER_SHA256,
        "parameter_updates": 0, "target_gradients": 0,
        "rows": [{"cell": "ridge_activity30_m10"}],
    }


def _carrier_evidence(session: str = "s") -> dict[str, dict[str, object]]:
    return {session: {
        "budget": 10, "selection": "chronological_first_m", "selected_indices": list(range(10)),
        "selected_indices_sha256": plan.M10_CHRONOLOGICAL_SELECTED_INDICES_SHA256,
        "usable_directional_trials": 3, "raw_t4_sha256": "e" * 64,
        "normalized_t4_sha256": "f" * 64,
    }}


def _seed_evidence(session: str = "s") -> dict[str, dict[str, object]]:
    return {session: {
        "seed_source": "dataset.calib_trialized_neural_features[:30]", "activity_rows": 30,
        "reconstruction_law": "frozen_cubic_interpolation",
        "sealed_first30_activity_stack_sha256": "1" * 64,
        "used_seed_activity_stack_sha256": "1" * 64,
        "reconstructed_first30_activity_stack_sha256": "2" * 64,
        "reconstructed_vs_sealed_max_abs": 0.0,
    }}


def _query_evidence(session: str = "s", *, within: bool = False) -> dict[str, dict[str, object]]:
    segments = [{
        "first_ordinal": 0, "last_ordinal": 4, "scored_prediction_count": 5,
        "pre_update_identity_sha256": "a" * 64, "governed_starts_sha256": "b" * 64,
    }]
    arm = {
        "frozen_batch_size": 32, "first_prediction_parity_singleton_batch": True,
        "actual_scored_prediction_count": 5,
        "identity_segments": segments,
        "identity_segments_sha256": physical._canonical_json_sha({"identity_segments": segments}),
    }
    return {session: {
        "governed_post30_cpre_int64_sha256": "4" * 64,
        "governed_post30_anchor_core_sha256": "5" * 64,
        "governed_post30_window_count": 5,
        "cpre_anchor_reference_int64_sha256": "6" * 64,
        "anchor_score_ordered_window_starts_sha256": "7" * 64,
        "anchor_reference_window_count": 9,
        "governed_post30_subset_of_anchor_full_query": True,
        "within_post30_exact_anchor_parity": True if within else None,
        "batch_decode_evidence": {"frozen_batch_size": 32,
                                  "arms": {key: dict(arm) for key in (
                                      "static", "true_trial", "chunk_phase0", "chunk_phase50")}},
    }}


def test_a0_synthetic_shard_lifecycle_has_exact_links_and_gpu0_only(tmp_path: Path) -> None:
    closure = physical.source_closure(_ROOT)
    roster = ("s",)
    cpre_sha = "c" * 64
    binding = {
        "root_relative": plan.A0_ROOTS["external_post30_local"],
        "closure_sha256": closure["closure_sha256"], "surface": "external_post30_local",
        "roster": list(roster), "cpre_terminal_sha256": cpre_sha,
        "device_profile": physical.static_device_profile(),
        "scheduler_profile": physical.scheduler_profile("external_post30_local"),
        "frozen_decode_batch_size": plan.A0_FROZEN_DECODE_BATCH_SIZE,
    }
    digest = "d" * 64
    rows = [physical.A0Row(
        surface="external_post30_local", session_id="s", arm=arm, phase=phase, r2=r2,
        first_prediction_sha256=digest, prediction_sha256=digest,
        initial_b3s_identity_sha256=digest, final_b3s_identity_sha256=digest,
        initial_activity_stack_sha256=digest, final_activity_stack_sha256=digest,
        query_starts_sha256=digest, target_window_sha256=digest, window_count=5,
        update_count=0, eviction_count=0, partial_final_bins_discarded=0,
        parameter_updates=0, target_gradients=0, wall_seconds=0.1, peak_memory_bytes=0,
    ) for arm, phase, r2 in (("static", None, .2), ("true_trial", None, .3), ("chunk", 0, .299), ("chunk", 50, .29))]
    parent = tmp_path / "a0"; parent.mkdir(); root = parent / "external"
    result = physical.execute_a0_shard_synthetic(
        root=root, repo_root=_ROOT, capability=physical._mint_test_capability(binding),
        surface="external_post30_local", roster=roster, cpre_terminal_sha256=cpre_sha,
        static_anchor=_anchor_payload(), rows=rows, actual_scored_prediction_count={"s": 5},
        carrier_evidence=_carrier_evidence(), activity_seed_evidence=_seed_evidence(),
        query_surface_evidence=_query_evidence(),
    )
    assert set(result) == {"attempt", "launch", "input_authority", "replay", "terminal"}
    terminal = receipts.descriptor_read(root, "terminal.json").payload
    assert terminal["attempt_sha256"] == result["attempt"]
    assert terminal["parameter_updates"] == terminal["target_gradients"] == 0


def test_a0_authority_rejects_wrong_anchor_or_carrier30() -> None:
    authority = physical.build_a0_input_authority(
        surface="within_post30", roster=("s",), cpre_terminal_sha256="c" * 64,
        static_anchor=_anchor_payload(), carrier_evidence=_carrier_evidence(), activity_seed_evidence=_seed_evidence(),
        query_surface_evidence=_query_evidence(within=True),
    )
    assert authority["carrier_label_budget"] == 10 and authority["activity_retained_capacity"] == 30
    bad = _anchor_payload(); bad["checkpoint_sha256"] = "0" * 64
    with pytest.raises(physical.PhysicalContractError, match="checkpoint"):
        physical.build_a0_input_authority(surface="within_post30", roster=("s",),
                                          cpre_terminal_sha256="c" * 64, static_anchor=bad,
                                          carrier_evidence=_carrier_evidence(), activity_seed_evidence=_seed_evidence(),
                                          query_surface_evidence=_query_evidence(within=True))
    bad_carrier = _carrier_evidence(); bad_carrier["s"]["selected_indices"] = list(range(1, 11))
    with pytest.raises(physical.PhysicalContractError, match="chronological"):
        physical.build_a0_input_authority(surface="within_post30", roster=("s",),
                                          cpre_terminal_sha256="c" * 64, static_anchor=_anchor_payload(),
                                          carrier_evidence=bad_carrier, activity_seed_evidence=_seed_evidence(),
                                          query_surface_evidence=_query_evidence(within=True))
    bad_seed = _seed_evidence(); bad_seed["s"]["used_seed_activity_stack_sha256"] = "3" * 64
    with pytest.raises(physical.PhysicalContractError, match="sealed direct"):
        physical.build_a0_input_authority(surface="within_post30", roster=("s",),
                                          cpre_terminal_sha256="c" * 64, static_anchor=_anchor_payload(),
                                          carrier_evidence=_carrier_evidence(), activity_seed_evidence=bad_seed,
                                          query_surface_evidence=_query_evidence(within=True))


def test_anchor_shaped_window_authority_binds_named_cpre_and_anchor_hashes() -> None:
    # This mirrors immutable activity-budget score-row placement rather than
    # manufacturing both authorities through the A0 route itself.
    metadata = {"anchor_window_crosschecks": {"external_post30_local": {"s": {
        "full_window_count": 9, "full_window_starts_sha256": "1" * 64,
        "full_window_anchor_core_sha256": "2" * 64,
        "post30_window_count": 5, "post30_window_starts_sha256": "3" * 64,
        "post30_window_anchor_core_sha256": "4" * 64,
        "post30_is_deterministic_subset_of_full": True,
    }}}}
    anchor = _anchor_payload()
    anchor["rows"] = [{"cell": plan.ANCHOR_CELL, "surface": "external_official_query", "session": "s",
                       "window_count": 9, "ordered_window_starts_sha256": "2" * 64}]
    authority = physical._derive_a0_query_window_authority(
        metadata=metadata, anchor_payload=anchor, surface="external_post30_local", roster=("s",))
    assert authority["s"] == {
        "cpre_anchor_reference_int64_sha256": "1" * 64,
        "anchor_score_ordered_window_starts_sha256": "2" * 64,
        "anchor_reference_window_count": 9,
        "within_post30_exact_anchor_parity": None,
    }
    anchor["rows"][0]["ordered_window_starts_sha256"] = "1" * 64
    with pytest.raises(physical.PhysicalContractError, match="exact anchor score"):
        physical._derive_a0_query_window_authority(
            metadata=metadata, anchor_payload=anchor, surface="external_post30_local", roster=("s",))


def test_public_cli_is_inert_and_torch_free_under_python_S() -> None:
    command = [sys.executable, "-S", str(_ROOT / "tfpd_exploration/scripts/run_m2_cpre_a0_chunk_memory_v1.py"), "--dry-run"]
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    output = subprocess.run(command, text=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert output.returncode == 0 and "no data" in output.stdout
    denied = subprocess.run(command[:-1] + ["--execute"], text=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert denied.returncode == 2 and "opaque" in denied.stderr


def test_package_import_does_not_import_torch() -> None:
    command = [sys.executable, "-S", "-c", "import sys; sys.path[:0]=['%s']; import tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1; assert 'torch' not in sys.modules and 'src' not in sys.modules" % _ROOT]
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, text=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr


def test_qualified_route_does_not_poison_streaming_top_level_src_namespace() -> None:
    code = (
        "import sys; from pathlib import Path; "
        "root=Path(r'%s'); sys.path[:0]=[str(root)]; "
        "import tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1.inventory; "
        "assert 'src' not in sys.modules; "
        "sys.path.insert(0, str(root/'streaming_calibration_exp')); "
        "import src.models.components.streaming_spint as streaming; "
        "assert str(Path(streaming.__file__)).startswith(str(root/'streaming_calibration_exp')); "
        "print('isolated')"
    ) % _ROOT
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], text=True, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"
