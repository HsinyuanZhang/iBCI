from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


SUA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUA_ROOT / "scripts"))

import preflight_cebra_pseudosession_sua_stage_f_seed as subject  # noqa: E402


def _stage_p(tmp_path: Path, *, passed: bool = True) -> Path:
    path = tmp_path / "stage_p.json"
    payload = {
        "receipt_kind": "cebra_pseudosession_sua_stage_p_aggregate",
        "seed": 42,
        "passes_stage_p": passed,
        "verdict": "STAGE_P_PASS__EXPAND_SEEDS_43_44" if passed else "STAGE_P_STOP",
        "frozen_gates": {
            "external_absolute_t4_delta_at_least_0p03": passed,
            "within_t4_noninferior_at_minus_0p03": passed,
        },
        "formal_subc_test_nwb_opened": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = subject.sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(digest + "\n", encoding="ascii")
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return path


def _manifest(seed: int = 43) -> dict:
    return {
        "seed": seed,
        "examples_audited": 1_086_007,
        "mixed_fraction_observed": 0.492,
        "rejected_candidate_examples": 8_000,
        "accepted_endpoint_residual": {"max": 0.249, "p99": 0.09},
        "accepted_donor_uses": {f"session_{index}": index + 1 for index in range(27)},
        "schedule_sha256": "a" * 64,
    }


def test_stage_p_gate_accepts_only_immutable_positive_receipt(tmp_path: Path) -> None:
    good = _stage_p(tmp_path, passed=True)
    payload, digest = subject.load_stage_p(good)
    assert payload["passes_stage_p"] is True
    assert digest == subject.sha256_file(good)

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad = _stage_p(bad_dir, passed=False)
    with pytest.raises(subject.StageFScheduleError, match="did not authorize"):
        subject.load_stage_p(bad)


def test_stage_p_gate_rejects_mutable_or_tampered_receipt(tmp_path: Path) -> None:
    mutable = _stage_p(tmp_path, passed=True)
    mutable.chmod(0o644)
    with pytest.raises(subject.StageFScheduleError, match="not immutable"):
        subject.load_stage_p(mutable)

    other = tmp_path / "other"
    other.mkdir()
    tampered = _stage_p(other, passed=True)
    tampered.chmod(0o644)
    tampered.write_text("{}", encoding="utf-8")
    tampered.chmod(0o444)
    with pytest.raises(subject.StageFScheduleError, match="sidecar drift"):
        subject.load_stage_p(tampered)


def test_schedule_manifest_has_acting_pass_and_fail_gates() -> None:
    subject.validate_schedule_manifest(_manifest(), 43)

    wrong_seed = _manifest(seed=44)
    with pytest.raises(subject.StageFScheduleError, match="seed drift"):
        subject.validate_schedule_manifest(wrong_seed, 43)

    high_residual = _manifest()
    high_residual["accepted_endpoint_residual"]["max"] = 0.251
    with pytest.raises(subject.StageFScheduleError, match="exceeds 0.25"):
        subject.validate_schedule_manifest(high_residual, 43)

    missing_donor = _manifest()
    missing_donor["accepted_donor_uses"]["session_0"] = 0
    with pytest.raises(subject.StageFScheduleError, match="not every source session"):
        subject.validate_schedule_manifest(missing_donor, 43)


def test_stage_f_seed_scope_is_exact() -> None:
    assert subject.STAGE_F_SEEDS == (43, 44)
