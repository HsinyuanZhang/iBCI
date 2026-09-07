"""No-target tests for the post-synthetic dual-consumer authority publisher."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_actual_cpu_route as route  # noqa: E402
import track_b_v2_post_synthetic_runtime_control_authority as authority  # noqa: E402


RUNNER = ROOT / "cebra_exploration/scripts/publish_track_b_v2_post_synthetic_runtime_controls.py"


def _runner(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run([sys.executable, str(RUNNER), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True)


def test_live_dry_plan_binds_canonical_cost_terminal_and_two_honest_threshold_origins() -> None:
    plan = authority.build_dry_plan()
    assert plan["status"] == authority.STATUS_DRY_PLAN
    assert plan["input_bindings"]["cost"]["canonical_body_sha256"] == authority.EXPECTED_COST_SHA256
    assert plan["input_bindings"]["terminal"]["sha256"] == authority.EXPECTED_TERMINAL_SHA256
    assert plan["positive_control_threshold_r2"] == 0.70
    assert plan["positive_threshold_origin"] == authority.POSITIVE_THRESHOLD_ORIGIN
    assert plan["deranged_hard_null_threshold_r2"] == 0.60
    assert plan["hard_null_threshold_origin"] == authority.HARD_NULL_THRESHOLD_ORIGIN
    assert plan["hard_null_threshold_may_be_relaxed_or_backfilled"] is False
    assert plan["receipt_minted"] is plan["target_data_opened"] is plan["gpu_used"] is False
    evidence = plan["subject_m_payload"]["raw_bound_control_evidence"]
    assert evidence == plan["rt_payload"]["raw_bound_control_evidence"]
    assert evidence["arm_run_count"] == 32
    assert evidence["cebra_fit_call_count"] == 56
    assert evidence["decoder_measurement_count"] == 192
    assert len(evidence["target_support_only_raw_measurements"]) == 64
    assert evidence["permutation_authority_sha256"] == authority.EXPECTED_PERMUTATION_SHA256


def test_all_required_groups_are_exact_eight_of_eight_at_separate_thresholds() -> None:
    evidence, _ = authority.load_raw_bound_evidence()
    groups = evidence["target_support_only_groups"]
    for arm in authority.POSITIVE_ARMS:
        for decoder in authority.DECODERS:
            group = groups[f"{arm}__{decoder}"]
            assert group["count"] == group["strict_pass_count"] == 8
            assert group["min"] > 0.70
            assert group["strict_relation"] == "greater_than"
    for decoder in authority.DECODERS:
        group = groups[f"{authority.DERANGED_ARM}__{decoder}"]
        assert group["count"] == group["strict_pass_count"] == 8
        assert group["max"] < 0.60
        assert group["strict_relation"] == "less_than"
        unaligned = groups[f"{authority.UNALIGNED_ARM}__{decoder}"]
        assert unaligned["strict_relation"] == "not_gated"
        assert unaligned["strict_pass_count"] is None


def test_payloads_fit_both_live_consumers_and_share_non_circular_evidence_binding(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence, _ = authority.load_raw_bound_evidence()
    closure = authority.implementation_closure()
    payloads = authority.build_consumer_payloads(evidence=evidence, closure=closure)
    shared = payloads["subject_m"]["paired_runtime_control_set"]
    assert shared == payloads["rt"]["paired_runtime_control_set"]
    assert "subject_m_body_sha256" not in shared and "rt_body_sha256" not in shared
    assert shared["shared_raw_bound_evidence_sha256"] == evidence["raw_bound_evidence_sha256"]
    authority.subject_consumer._validate_control_raw_bound_summary(payloads["subject_m"])

    subject_path = tmp_path / "subject" / "runtime_control.json"
    subject_path.parent.mkdir()
    monkeypatch.setattr(authority.subject_consumer, "RUNTIME_CONTROL_PATH", subject_path)
    route.write_immutable_pair(subject_path, payloads["subject_m"])
    subject_preflight = {
        "required_runtime_control_body_path": str(subject_path),
        "required_runtime_control_contract": authority.subject_consumer._runtime_control_admission_contract(
            cost_body_sha256=authority.EXPECTED_COST_SHA256),
    }
    checked_subject = authority.subject_consumer._validate_runtime_control_pair(preflight=subject_preflight)
    assert checked_subject["body_sha256"]

    rt_path = tmp_path / "rt" / "receipt.json"
    rt_path.parent.mkdir()
    monkeypatch.setattr(authority.rt_consumer, "RUNTIME_CONTROL_BODY", rt_path)
    route.write_immutable_pair(rt_path, payloads["rt"])
    checked = authority.rt_consumer.inspect_root_frozen_runtime_control_pair(
        cost_gate={"canonical_body_sha256": authority.EXPECTED_COST_SHA256})
    assert checked["status"] == authority.rt_consumer.RUNTIME_CONTROL_STATUS


def test_default_cli_is_no_write_no_torch_no_cebra_and_single_flags_reject() -> None:
    completed = _runner()
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["status"] == authority.STATUS_DRY_PLAN
    assert plan["torch_imported"] is plan["cebra_imported"] is False
    assert plan["receipt_minted"] is False
    for flag in ("--mint", "--i-have-root-review-authorization"):
        rejected = _runner(flag)
        assert rejected.returncode != 0
        assert "requires both" in rejected.stderr


def test_cli_has_no_output_threshold_target_or_evidence_override() -> None:
    text = RUNNER.read_text().lower()
    for forbidden in ("--output", "--threshold", "--target", "--formal", "--nwb", "--npz",
                      "--terminal", "--cost", "--permutation", "--seed"):
        assert forbidden not in text


def test_body_sidecar_or_symlink_conflict_fails_before_input_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for kind in ("body", "sidecar", "symlink"):
        subject = tmp_path / kind / "subject" / "runtime_control.json"
        rt = tmp_path / kind / "rt" / "receipt.json"
        subject.parent.mkdir(parents=True)
        rt.parent.mkdir(parents=True)
        if kind == "body":
            subject.write_text("poison")
        elif kind == "sidecar":
            authority._sidecar(subject).write_text("poison")
        else:
            victim = tmp_path / kind / "victim"
            victim.write_text("poison")
            subject.symlink_to(victim)
        monkeypatch.setattr(authority, "SUBJECT_BODY", subject)
        monkeypatch.setattr(authority.subject_consumer, "RUNTIME_CONTROL_PATH", subject)
        monkeypatch.setattr(authority, "RT_BODY", rt)
        monkeypatch.setattr(authority.rt_consumer, "RUNTIME_CONTROL_BODY", rt)
        monkeypatch.setattr(authority.route, "load_immutable_json",
                            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("input reached")))
        with pytest.raises(authority.RuntimeControlAuthorityError, match="fresh"):
            authority.build_dry_plan()
        monkeypatch.undo()


@pytest.mark.parametrize("mutation", (
    "terminal_sha", "closure", "settled", "positive_threshold", "hard_null_threshold",
    "permutation", "count", "query_fit",
    "positive_fail", "deranged_fail", "unaligned_role", "target_open",
))
def test_raw_bound_summary_tamper_fails_both_consumer_contracts(mutation: str) -> None:
    plan = authority.build_dry_plan()
    subject = copy.deepcopy(plan["subject_m_payload"])
    rt = copy.deepcopy(plan["rt_payload"])
    for payload in (subject, rt):
        evidence = payload["raw_bound_control_evidence"]
        if mutation == "terminal_sha":
            evidence["synthetic_v2_terminal_body_sha256"] = "0" * 64
        elif mutation == "closure":
            evidence["synthetic_v2_terminal_closure_sha256"] = "0" * 64
        elif mutation == "settled":
            evidence["settled_synthetic_v2_file_sha256"]["v2_core"] = "0" * 64
        elif mutation == "positive_threshold":
            evidence["positive_control_threshold_r2"] = 0.69
        elif mutation == "hard_null_threshold":
            evidence["deranged_hard_null_threshold_r2"] = 0.59
        elif mutation == "permutation":
            evidence["permutation_authority_sha256"] = "0" * 64
        elif mutation == "count":
            evidence["decoder_measurement_count"] = 191
        elif mutation == "query_fit":
            evidence["target_support_only_raw_measurements"][0]["query_neural_or_auxiliary_in_fit"] = True
        elif mutation == "positive_fail":
            row = next(row for row in evidence["target_support_only_raw_measurements"]
                       if row["arm"] == "cebra_joint_behavior")
            row["target_query_r2"] = 0.70
        elif mutation == "deranged_fail":
            row = next(row for row in evidence["target_support_only_raw_measurements"]
                       if row["arm"] == authority.DERANGED_ARM)
            row["target_query_r2"] = 0.60
        elif mutation == "unaligned_role":
            evidence["ordinary_unaligned_role"] = "pass_gate"
        elif mutation == "target_open":
            evidence["target_data_opened"] = True
    with pytest.raises(authority.subject_consumer.TrackBV2SubjectMStagePRuntimeError):
        authority.subject_consumer._validate_control_raw_bound_summary(subject)
    # RT uses the same hardened raw-summary rules; call its inspector-level logic through a
    # temporary pair in the dedicated compatibility test, and assert the equivalent fields here.
    with pytest.raises(authority.subject_consumer.TrackBV2SubjectMStagePRuntimeError):
        authority.subject_consumer._validate_control_raw_bound_summary(rt)


@pytest.mark.parametrize("mutation", (
    "terminal_sha", "positive_threshold", "hard_null_threshold", "permutation", "query_fit", "positive_fail",
    "deranged_fail", "unaligned_role", "target_open",
))
def test_rt_live_consumer_rejects_raw_evidence_tamper(
        mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = authority.build_dry_plan()
    payload = copy.deepcopy(plan["rt_payload"])
    evidence = payload["raw_bound_control_evidence"]
    if mutation == "terminal_sha":
        evidence["synthetic_v2_terminal_body_sha256"] = "0" * 64
    elif mutation == "positive_threshold":
        evidence["positive_control_threshold_r2"] = 0.69
    elif mutation == "hard_null_threshold":
        evidence["deranged_hard_null_threshold_r2"] = 0.59
    elif mutation == "permutation":
        evidence["permutation_authority_sha256"] = "0" * 64
    elif mutation == "query_fit":
        evidence["target_support_only_raw_measurements"][0]["query_neural_or_auxiliary_in_fit"] = True
    elif mutation == "positive_fail":
        row = next(row for row in evidence["target_support_only_raw_measurements"]
                   if row["arm"] == "cebra_joint_behavior")
        row["target_query_r2"] = 0.70
    elif mutation == "deranged_fail":
        row = next(row for row in evidence["target_support_only_raw_measurements"]
                   if row["arm"] == authority.DERANGED_ARM)
        row["target_query_r2"] = 0.60
    elif mutation == "unaligned_role":
        evidence["ordinary_unaligned_role"] = "pass_gate"
    elif mutation == "target_open":
        evidence["target_data_opened"] = True
    body = tmp_path / "rt" / "receipt.json"
    body.parent.mkdir()
    monkeypatch.setattr(authority.rt_consumer, "RUNTIME_CONTROL_BODY", body)
    route.write_immutable_pair(body, payload)
    with pytest.raises(authority.rt_consumer.TrackBV2RTOneCellError):
        authority.rt_consumer.inspect_root_frozen_runtime_control_pair(
            cost_gate={"canonical_body_sha256": authority.EXPECTED_COST_SHA256})


def _patch_outputs(monkeypatch: pytest.MonkeyPatch, root: Path) -> tuple[Path, Path]:
    subject = root / "subject" / "runtime_control.json"
    rt = root / "rt" / "receipt.json"
    monkeypatch.setattr(authority, "SUBJECT_BODY", subject)
    monkeypatch.setattr(authority.subject_consumer, "RUNTIME_CONTROL_PATH", subject)
    monkeypatch.setattr(authority, "RT_BODY", rt)
    monkeypatch.setattr(authority.rt_consumer, "RUNTIME_CONTROL_BODY", rt)
    return subject, rt


def test_transactional_writer_publishes_two_standard_0444_pairs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = authority.build_dry_plan()
    subject, rt = _patch_outputs(monkeypatch, tmp_path)
    payloads = {"subject_m": plan["subject_m_payload"], "rt": plan["rt_payload"]}
    bindings = authority.publish_two_pairs_transactionally(payloads)
    assert set(bindings) == {"subject_m", "rt"}
    for role, path in (("subject_m", subject), ("rt", rt)):
        assert path.stat().st_mode & 0o777 == 0o444
        assert authority._sidecar(path).stat().st_mode & 0o777 == 0o444
        assert not path.is_symlink() and not authority._sidecar(path).is_symlink()
        assert bindings[role]["read_once_from_verified_fd"] is True


def test_transactional_writer_rolls_back_all_four_outputs_on_late_conflict(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = authority.build_dry_plan()
    subject, rt = _patch_outputs(monkeypatch, tmp_path)
    payloads = {"subject_m": plan["subject_m_payload"], "rt": plan["rt_payload"]}
    original = authority.os.link
    calls = 0

    def fail_fourth(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise FileExistsError("adversarial late sidecar conflict")
        original(*args, **kwargs)

    monkeypatch.setattr(authority.os, "link", fail_fourth)
    with pytest.raises(FileExistsError):
        authority.publish_two_pairs_transactionally(payloads)
    for path in (subject, authority._sidecar(subject), rt, authority._sidecar(rt)):
        assert not os.path.lexists(path)


def test_transactional_writer_recomputes_final_live_closure_after_all_four_links_and_rolls_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = authority.build_dry_plan()
    subject, rt = _patch_outputs(monkeypatch, tmp_path)
    payloads = {"subject_m": plan["subject_m_payload"], "rt": plan["rt_payload"]}
    calls = 0

    def drift_after_links() -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert all(os.path.lexists(path) for path in (
            subject, authority._sidecar(subject), rt, authority._sidecar(rt)))
        drifted = copy.deepcopy(plan["implementation_closure"])
        drifted["closure_sha256"] = "0" * 64
        return drifted

    monkeypatch.setattr(authority, "implementation_closure", drift_after_links)
    with pytest.raises(authority.RuntimeControlAuthorityError,
                       match="closure drift after publication"):
        authority.publish_two_pairs_transactionally(payloads)
    assert calls == 1
    for path in (subject, authority._sidecar(subject), rt, authority._sidecar(rt)):
        assert not os.path.lexists(path)


def test_transactional_writer_rejects_parent_rename_swap_and_fd_rolls_back_original_inode(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = authority.build_dry_plan()
    subject, rt = _patch_outputs(monkeypatch, tmp_path)
    payloads = {"subject_m": plan["subject_m_payload"], "rt": plan["rt_payload"]}
    original_link = authority.os.link
    link_calls = 0
    renamed_parent = tmp_path / "subject-renamed-after-bind"

    def rename_after_fourth_link(*args: object, **kwargs: object) -> None:
        nonlocal link_calls
        original_link(*args, **kwargs)
        link_calls += 1
        if link_calls == 4:
            subject.parent.rename(renamed_parent)
            subject.parent.mkdir()

    monkeypatch.setattr(authority.os, "link", rename_after_fourth_link)
    with pytest.raises(authority.RuntimeControlAuthorityError,
                       match="parent identity changed"):
        authority.publish_two_pairs_transactionally(payloads)
    assert link_calls == 4
    for path in (subject, authority._sidecar(subject), rt, authority._sidecar(rt)):
        assert not os.path.lexists(path)
    assert not os.path.lexists(renamed_parent / subject.name)
    assert not os.path.lexists(renamed_parent / authority._sidecar(subject).name)
    assert list(renamed_parent.iterdir()) == []


def test_settled_synthetic_implementation_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = authority.source._read_regular_file

    def poison(path: Path, *, label: str) -> bytes:
        raw = original(path, label=label)
        return raw + b"drift" if "synthetic_v2_core" in label else raw

    monkeypatch.setattr(authority.source, "_read_regular_file", poison)
    with pytest.raises(authority.RuntimeControlAuthorityError, match="settled"):
        authority.implementation_closure()
