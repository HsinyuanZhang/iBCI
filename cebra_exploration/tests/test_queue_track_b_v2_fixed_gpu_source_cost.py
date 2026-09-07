"""Shell-level no-data/no-GPU tests for the fixed engineering wait queue."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUE = REPO_ROOT / "cebra_exploration/scripts/queue_track_b_v2_fixed_gpu_source_cost.sh"
CANONICAL_OUTPUT = (
    REPO_ROOT / "cebra_exploration/results"
    / "track_b_v2_fixed_gpu_cost_sua_firstfold_d8it250_s42_gpu1_v1/receipt.json"
)


def _mask_payload() -> dict:
    return {
        "receipt_kind": "carrier_value_mask_terminal_aggregate",
        "formal_subc_test_nwb_opened": False,
        "screen_id": "carrier_value_mask_v1",
        "seed": 42,
        "parameter_delta": 0,
        "positive_forward_mask_signal": False,
        "verdict": "VALUE_WEIGHTED_LOW_GAIN_MASK_FLAT_OR_NONSPECIFIC__ADVANCE_QUEUE",
        "routing_checks": {
            "external_low_t4_absolute_delta_positive": False,
            "external_low_mask_carrier_interaction_positive": False,
            "external_low_t4_beats_random_same_count": False,
            "external_low_t4_beats_high_gain_mask": False,
            "within_low_t4_delta_at_least_minus_0p03": True,
        },
        "domains": {"within_subject": {}, "external_subject_M": {}},
    }


def _write_mask(root: Path, *, mode: int = 0o444, payload: dict | None = None) -> Path:
    body = root / "terminal_mask_aggregate.json"
    payload = _mask_payload() if payload is None else payload
    body_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode()
    body.write_bytes(body_bytes)
    side = body.with_name(body.name + ".sha256")
    side.write_text(f"{hashlib.sha256(body_bytes).hexdigest()}  {body.name}\n", encoding="ascii")
    body.chmod(mode)
    side.chmod(mode)
    return body


def _fake_smi(root: Path, *, occupied: bool) -> Path:
    script = root / "nvidia-smi"
    script.write_text("#!/bin/sh\n" + ("printf '424242\\n'\n" if occupied else "exit 0\n"))
    script.chmod(0o755)
    return script


def _queue_sha() -> str:
    return hashlib.sha256(QUEUE.read_bytes()).hexdigest()


def _run_live_test(mask: Path, smi: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update({
        "TRACK_B_QUEUE_EXPECTED_SHA256": _queue_sha(),
        "TRACK_B_QUEUE_TEST_MODE": "1",
        "TRACK_B_QUEUE_MASK_RECEIPT": str(mask),
        "TRACK_B_QUEUE_NVIDIA_SMI": str(smi),
        "TRACK_B_QUEUE_POLL_SECONDS": "0",
        "TRACK_B_QUEUE_MAX_POLLS": "1",
        "TRACK_B_QUEUE_TEST_NO_EXEC": "1",
    })
    return subprocess.run(
        [str(QUEUE), "--execute", "--i-have-authorization"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


def test_default_is_dry_no_wait_no_write_and_needs_no_anchor(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    completed = subprocess.run(
        [str(QUEUE)], cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DRY_PLAN_ONLY__NO_WAIT__NO_DATA__NO_GPU__NO_WRITE" in completed.stdout
    assert set(tmp_path.iterdir()) == before
    assert not os.path.lexists(CANONICAL_OUTPUT)
    assert not os.path.lexists(CANONICAL_OUTPUT.with_name(CANONICAL_OUTPUT.name + ".sha256"))


def test_standard_immutable_sidecar_and_idle_gpu_reach_test_only_ready(tmp_path: Path) -> None:
    mask = _write_mask(tmp_path)
    completed = _run_live_test(mask, _fake_smi(tmp_path, occupied=False))
    assert completed.returncode == 0, completed.stderr
    assert "READY_TEST_ONLY__NO_EXEC" in completed.stdout
    assert hashlib.sha256(mask.read_bytes()).hexdigest() in completed.stdout
    assert stat.S_IMODE(mask.stat().st_mode) == 0o444


def test_mutable_mask_receipt_waits_and_never_executes(tmp_path: Path) -> None:
    mask = _write_mask(tmp_path, mode=0o644)
    completed = _run_live_test(mask, _fake_smi(tmp_path, occupied=False))
    assert completed.returncode == 75
    assert "WAIT_MASK_INVALID" in completed.stderr
    assert "reason=mask_missing_or_invalid" in completed.stderr
    assert "READY_TEST_ONLY" not in completed.stdout


def test_occupied_physical_gpu1_waits_and_never_executes(tmp_path: Path) -> None:
    mask = _write_mask(tmp_path)
    completed = _run_live_test(mask, _fake_smi(tmp_path, occupied=True))
    assert completed.returncode == 75
    assert "physical_gpu1_compute_process_present_or_query_failed" in completed.stderr
    assert "READY_TEST_ONLY" not in completed.stdout


def test_symlink_or_nonstandard_sidecar_is_rejected(tmp_path: Path) -> None:
    immutable = tmp_path / "immutable"
    immutable.mkdir()
    real_mask = _write_mask(immutable)
    alias = tmp_path / real_mask.name
    alias.symlink_to(real_mask)
    (tmp_path / f"{alias.name}.sha256").symlink_to(real_mask.with_name(real_mask.name + ".sha256"))
    completed = _run_live_test(alias, _fake_smi(tmp_path, occupied=False))
    assert completed.returncode == 75
    assert "WAIT_MASK_INVALID" in completed.stderr


@pytest.mark.parametrize(
    ("field", "poison"),
    (
        ("screen_id", "other_screen"),
        ("seed", 43),
        ("parameter_delta", 1),
        ("positive_forward_mask_signal", 0),
        ("verdict", "UNREVIEWED_BRANCH"),
        ("formal_subc_test_nwb_opened", True),
        ("domains", {"within_subject": {}}),
        ("routing_checks", {"external_low_t4_absolute_delta_positive": False}),
    ),
)
def test_mask_terminal_completion_contract_tamper_waits(
        tmp_path: Path, field: str, poison: object) -> None:
    payload = _mask_payload()
    payload[field] = poison
    mask = _write_mask(tmp_path, payload=payload)
    completed = _run_live_test(mask, _fake_smi(tmp_path, occupied=False))
    assert completed.returncode == 75
    assert "WAIT_MASK_INVALID" in completed.stderr
    assert "READY_TEST_ONLY" not in completed.stdout


def test_both_allowed_verdicts_are_completion_only_not_branching(tmp_path: Path) -> None:
    payload = _mask_payload()
    payload["positive_forward_mask_signal"] = True
    payload["verdict"] = "VALUE_WEIGHTED_LOW_GAIN_MASK_POSITIVE__DESIGN_SEPARATE_TRAINED_ROUTE"
    mask = _write_mask(tmp_path, payload=payload)
    completed = _run_live_test(mask, _fake_smi(tmp_path, occupied=False))
    assert completed.returncode == 0, completed.stderr
    assert "READY_TEST_ONLY__NO_EXEC" in completed.stdout


def test_single_authorization_flag_and_wrong_self_anchor_fail_closed(tmp_path: Path) -> None:
    for flag in ("--execute", "--i-have-authorization"):
        completed = subprocess.run([str(QUEUE), flag], capture_output=True, text=True, timeout=10)
        assert completed.returncode == 2
        assert "requires both" in completed.stderr
    env = dict(os.environ)
    env["TRACK_B_QUEUE_EXPECTED_SHA256"] = "0" * 64
    completed = subprocess.run(
        [str(QUEUE), "--execute", "--i-have-authorization"],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode != 0
    assert "SHA drift" in completed.stderr


def test_copied_queue_cannot_substitute_for_canonical_script(tmp_path: Path) -> None:
    copied = tmp_path / QUEUE.name
    copied.write_bytes(QUEUE.read_bytes())
    copied.chmod(0o755)
    env = dict(os.environ)
    env["TRACK_B_QUEUE_EXPECTED_SHA256"] = hashlib.sha256(copied.read_bytes()).hexdigest()
    completed = subprocess.run(
        [str(copied), "--execute", "--i-have-authorization"],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode != 0
    assert "canonical regular script" in completed.stderr
