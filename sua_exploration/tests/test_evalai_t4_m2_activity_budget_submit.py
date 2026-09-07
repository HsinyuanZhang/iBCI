from __future__ import annotations

import importlib
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


submit = importlib.import_module(
    "sua_exploration.evalai_t4_m2_activity_budget.submit_evalai"
)


def test_plan_is_static_and_binds_all_candidates(monkeypatch):
    monkeypatch.setattr(submit, "_runtime", lambda: (_ for _ in ()).throw(AssertionError()))
    m10 = submit.plan_report("m10")
    m4 = submit.plan_report("m4")
    m30 = submit.plan_report("m30")
    assert m10["image_id"].startswith("sha256:")
    assert len(m10["image_id"]) == 71
    assert m10["label_budget"] == 10
    assert m4["label_budget"] == 4
    assert m10["activity_budget"] == m4["activity_budget"] == 30
    assert m30["label_budget"] == m30["activity_budget"] == 30
    assert "chronological cue" in m10["budget_disclosure"]
    assert "selected from first30 cue" in m4["budget_disclosure"]
    assert m10["image_id"] != m4["image_id"]
    assert m10["payload_sha256"] != m4["payload_sha256"]
    assert len({m10["image_id"], m4["image_id"], m30["image_id"]}) == 3
    assert len({m10["payload_sha256"], m4["payload_sha256"], m30["payload_sha256"]}) == 3
    assert m10["private"] is m4["private"] is m30["private"] is True
    assert all(item["value"] is False for item in m10["submission_attributes"])


def test_unknown_arm_and_state_separation():
    with pytest.raises(ValueError, match="unknown arm"):
        submit.plan_report("m33")
    assert submit.state_path("m10") != submit.state_path("m4")
    assert submit.state_path("m30") not in {submit.state_path("m10"), submit.state_path("m4")}


@pytest.mark.parametrize("arm", ["m10", "m4", "m30"])
def test_image_guard_requires_every_label(arm):
    candidate = submit.CANDIDATES[arm]

    class Image:
        id = candidate["image_id"]
        attrs = {
            "Config": {
                "Labels": {
                    "ai.eval.payload.sha256": candidate["payload_sha256"],
                    "ai.eval.checkpoint.sha256": submit.CHECKPOINT_SHA256,
                    "ai.eval.label_budget": str(candidate["budget"]),
                    "ai.eval.activity_budget": "30",
                    "ai.eval.method": candidate["method_label"],
                }
            }
        }

    class Images:
        def get(self, tag):
            assert tag == candidate["image_tag"]
            return Image()

    class Client:
        images = Images()

    class Docker:
        @staticmethod
        def from_env():
            return Client()

    _, _, observed = submit._validate_local_image(arm, Docker)
    assert observed["payload_sha256"] == candidate["payload_sha256"]
    Image.attrs["Config"]["Labels"]["ai.eval.activity_budget"] = "4"
    with pytest.raises(RuntimeError, match="activity_budget"):
        submit._validate_local_image(arm, Docker)


def test_plan_cli_imports_no_external_runtime(tmp_path):
    script = Path(submit.__file__).resolve()
    command = [sys.executable, "-S", str(script), "--arm", "m4", "--plan"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)
    assert payload["mode"] == "plan_only"
    assert payload["label_budget"] == 4
    assert "docker" not in sys.modules
    assert "boto3" not in sys.modules


@pytest.mark.parametrize(
    "arm,submission_id,digest,metrics",
    [
        (
            "m10",
            581359,
            "a747ece42f8b7b467767e4d86debc0368dcc3df0db81195f0849967b35f78c07",
            {
                "held_in_r2_mean": 0.5584457832666476,
                "held_in_r2_std": 0.02817107511468847,
                "held_out_r2_mean": 0.26383289570479845,
                "held_out_r2_std": 0.0990401582746693,
                "normalized_latency": 0.044338009469217544,
            },
        ),
        (
            "m4",
            581361,
            "68da5427e2d27169b8b6e1e08a487113b65b734a7e7644a5d1778f993febbb97",
            {
                "held_in_r2_mean": 0.5627788928403332,
                "held_in_r2_std": 0.017415236899653543,
                "held_out_r2_mean": 0.2897439880338965,
                "held_out_r2_std": 0.10386880330226296,
                "normalized_latency": 0.04701159050299251,
            },
        ),
        (
            "m30",
            581362,
            "ea7d4185943602b964accd98e4c8a90758350fb689227efe0fa8ff64fbbe9c5e",
            {
                "held_in_r2_mean": 0.5754404413011437,
                "held_in_r2_std": 0.026353584924937428,
                "held_out_r2_mean": 0.2951607131264304,
                "held_out_r2_std": 0.10888410118544638,
                "normalized_latency": 0.044834055592910246,
            },
        ),
    ],
)
def test_official_terminal_receipts_are_immutable_and_exact(
    arm, submission_id, digest, metrics
):
    artifact_dir = Path(submit.__file__).resolve().parent / "artifacts"
    path = artifact_dir / f"evalai_submission_{submission_id}_terminal_receipt_v1.json"
    sidecar = path.with_suffix(path.suffix + ".sha256")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    assert sidecar.read_text(encoding="utf-8") == (
        f"{digest}  {path.name}\n"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    payload = json.loads(raw)
    assert payload["arm"] == arm
    assert payload["official_api"]["submission_id"] == submission_id
    assert payload["official_api"]["status"] == "finished"
    assert payload["integrity"]["retry_or_second_submission_created"] is False
    assert payload["official_metrics"] == metrics


def test_completion_receipt_rehashes_every_named_authority():
    project_root = Path(__file__).resolve().parents[2]
    artifact_dir = Path(submit.__file__).resolve().parent / "artifacts"
    path = artifact_dir / "m2_sua_complete_experiment_audit_v1.json"
    sidecar = path.with_suffix(path.suffix + ".sha256")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == "a090e5ba973d860982deac0abb6b0211f324756283bb46d12a3093ef4434ed97"
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  {path.name}\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    payload = json.loads(raw)
    assert payload["completion"]["status"] == "COMPLETE"
    assert [row["submission_id"] for row in payload["official_results"]] == [
        581359,
        581361,
        581362,
    ]
    for authority in payload["local_authorities"]:
        body = project_root / authority["path"]
        assert hashlib.sha256(body.read_bytes()).hexdigest() == authority["sha256"]
    document = project_root / payload["documentation"]["path"]
    assert hashlib.sha256(document.read_bytes()).hexdigest() == payload["documentation"][
        "sha256"
    ]
