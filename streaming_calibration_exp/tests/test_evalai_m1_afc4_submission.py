from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "sua_exploration" / "evalai_m1_threeway"))

import submit_evalai as helper  # noqa: E402


def _manifest(tmp_path: Path, *, duplicate_ids: bool = False, phase_id: int | None = None) -> Path:
    payload = {
        "schema_version": helper.CANDIDATE_MANIFEST_SCHEMA,
        "challenge_id": helper.CHALLENGE_ID,
        "phase_id": helper.PHASE_ID if phase_id is None else phase_id,
        "team_id": helper.TEAM_ID,
        "candidates": {},
    }
    for arm, image_char, payload_char in (("full", "a", "c"), ("b4", "b", "d")):
        candidate = dict(helper.CANDIDATES[arm])
        candidate.update(
            {
                "image_id": "sha256:" + ("a" if duplicate_ids else image_char) * 64,
                "payload_sha256": payload_char * 64,
                "checkpoint_sha256": "e" * 64,
                "teacher_sha256": "f" * 64,
            }
        )
        payload["candidates"][arm] = candidate
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_full_b4_plan_is_placeholder_until_manifest_and_has_no_side_effects():
    for arm in ("full", "b4"):
        report = helper.plan_report(arm)
        assert report["mode"] == "plan_only"
        assert report["ready_for_docker_preflight"] is False
        assert report["challenge_id"] == 2319
        assert report["phase_id"] == 4599
        assert report["team_id"] == 41975
        assert report["required_runtime"]["query_labels_used"] is False
        assert report["required_runtime"]["backpropagation"] is False
    assert helper.plan_report("m1_afc4_full_allsource")["arm"] == "full"
    assert helper.plan_report("m1_afc4_b4_allsource")["arm"] == "b4"


def test_manifest_binds_independent_hashes_without_network(tmp_path: Path):
    path = _manifest(tmp_path)
    bound = helper.load_candidate_manifest(path)
    assert set(bound) == {"full", "b4"}
    assert bound["full"]["image_id"] != bound["b4"]["image_id"]
    assert helper.plan_report("full", path)["ready_for_docker_preflight"] is True


def test_manifest_fails_closed_on_phase_or_duplicate_image(tmp_path: Path):
    with pytest.raises(ValueError, match="phase_id"):
        helper.load_candidate_manifest(_manifest(tmp_path, phase_id=9999))
    with pytest.raises(ValueError, match="independent immutable image IDs"):
        helper.load_candidate_manifest(_manifest(tmp_path, duplicate_ids=True))


def test_non_plan_full_preflight_requires_manifest(monkeypatch):
    def fail_if_docker_called():
        raise AssertionError("Docker must not be touched before a manifest is bound")

    monkeypatch.setattr(helper.docker, "from_env", fail_if_docker_called)
    with pytest.raises(RuntimeError, match="not bound"):
        helper.preflight("full")
