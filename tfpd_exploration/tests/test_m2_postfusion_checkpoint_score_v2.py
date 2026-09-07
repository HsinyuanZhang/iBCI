from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from tfpd_exploration.src.m2_postfusion_checkpoint_score_v2 import binding, driver, plan
from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import lifecycle


ROOT = Path(__file__).resolve().parents[2]


def _publish(path: Path, payload: object) -> str:
    body = payload if isinstance(payload, bytes) else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(body)
    path.chmod(0o444)
    digest = hashlib.sha256(body).hexdigest()
    side = path.with_name(path.name + ".sha256")
    side.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    side.chmod(0o444)
    return digest


def _synthetic_v1_failure(root: Path) -> dict[str, str]:
    target = root / plan.V1_FAILURE_ROOT_RELATIVE
    target.mkdir(parents=True)
    attempt = {"schema": "m2_postfusion_checkpoint_score_v1_attempt_v1", "status": "ATTEMPT_RESERVED",
               "target_or_checkpoint_opened": False, "cuda_initialized": False, "closure_sha256": "a" * 64}
    attempt_sha = _publish(target / "attempt.json", attempt)
    launch = {"schema": "m2_postfusion_checkpoint_score_v1_launch_v1", "cuda_visible_devices": "", "cuda_initialized": False}
    launch_sha = _publish(target / "launch.json", launch)
    error = "b" * 64
    failure = {"schema": "m2_postfusion_checkpoint_score_v1_failure_v1", "status": "FAILED", "terminal_xor_failure": True,
               "attempt_sha256": attempt_sha, "error_sha256": error,
               "progress": {"checkpoint_strict_load_attempted": True, "checkpoint_strict_loaded": True,
                            "cuda_initialized": False, "pooled_descriptor_attempted": True, "pooled_descriptor_opened": True,
                            "rows": 0, "screen_descriptor_attempted": True, "screen_descriptor_opened": True,
                            "target_materialization_attempted": True, "target_materialized": False,
                            "target_or_checkpoint_opened": True}}
    failure_sha = _publish(target / "failure.json", failure)
    return {"attempt.json": attempt_sha, "launch.json": launch_sha, "failure.json": failure_sha, "error": error}


def test_static_workorder_and_actual_immutable_v1_failure_graph_are_exact_and_no_torch() -> None:
    assert plan.validate_static(ROOT)["workorder_sha256"] == plan.WORKORDER_SHA256
    witness = binding.validate_v1_failure_graph(ROOT)
    assert witness["body_sha256"] == plan.V1_FAILURE_BODIES
    assert witness["error_sha256"] == plan.V1_FAILURE_ERROR_SHA256
    # This has to be process-local: the combined suite intentionally runs a
    # real V1 strict-load test first, so parent-process ``sys.modules`` is not
    # a valid assertion about V2's no-Torch admission import.
    purelib = str(Path(sysconfig.get_paths()["purelib"]))
    env = {"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join((str(ROOT), purelib))}
    clean = subprocess.run(
        [sys.executable, "-S", "-c", "import sys; import tfpd_exploration.src.m2_postfusion_checkpoint_score_v2.driver; assert 'torch' not in sys.modules; print('no-torch')"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    assert clean.stdout.strip() == "no-torch"


def test_v1_held_fd_validator_rejects_extra_leaf_and_progress_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hashes = _synthetic_v1_failure(tmp_path)
    monkeypatch.setattr(plan, "V1_FAILURE_BODIES", {key: hashes[key] for key in ("attempt.json", "launch.json", "failure.json")})
    monkeypatch.setattr(plan, "V1_FAILURE_ERROR_SHA256", hashes["error"])
    binding.validate_v1_failure_graph(tmp_path)
    target = tmp_path / plan.V1_FAILURE_ROOT_RELATIVE
    _publish(target / "extra.json", {})
    with pytest.raises(binding.BindingError, match="missing/extra"):
        binding.validate_v1_failure_graph(tmp_path)
    (target / "extra.json").unlink(); (target / "extra.json.sha256").unlink()
    # Replacing a canonical leaf with a correctly-modeled but different body
    # is caught by the fixed body literal before semantic acceptance.
    (target / "failure.json").chmod(0o644)
    (target / "failure.json.sha256").chmod(0o644)
    _publish(target / "failure.json", {"schema": "wrong"})
    with pytest.raises(binding.BindingError, match="body SHA"):
        binding.validate_v1_failure_graph(tmp_path)


def test_v2_profile_is_distinct_and_failure_receipt_has_class_and_bounded_message(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; (repo / "tfpd_exploration/results").mkdir(parents=True)
    long_error = "x" * 500
    result = lifecycle.execute(
        repo,
        attempt={"schema": "attempt"},
        launch=lambda: {"schema": "launch"},
        bodies=lambda _artifact: (_ for _ in ()).throw(ValueError(long_error)),
        terminal=lambda _published: {},
        progress=lambda: {"rows": 0, "cuda_initialized": False},
        root_relative=plan.RESULT_ROOT_RELATIVE,
        schema=plan.SCHEMA,
        include_failure_diagnostic=True,
    )
    assert result[0] is None and isinstance(result[1], str)
    failure = json.loads((repo / plan.RESULT_ROOT_RELATIVE / "failure.json").read_text())
    assert failure["schema"] == f"{plan.SCHEMA}_failure_v1"
    assert failure["exception_class"] == "builtins.ValueError"
    assert failure["diagnostic_message"] == "x" * 240


def test_successor_failure_revalidates_its_held_predecessor_before_publication(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; (repo / "tfpd_exploration/results").mkdir(parents=True)
    result = lifecycle.execute(
        repo, attempt={"schema": "attempt"}, launch=lambda: {"schema": "launch"},
        bodies=lambda _artifact: (_ for _ in ()).throw(RuntimeError("materialize failed")),
        terminal=lambda _published: {}, progress=lambda: {"rows": 0},
        root_relative=plan.RESULT_ROOT_RELATIVE, schema=plan.SCHEMA,
        include_failure_diagnostic=True, failure_revalidate=lambda: None,
    )
    assert result[0] is None
    failure = json.loads((repo / plan.RESULT_ROOT_RELATIVE / "failure.json").read_text())
    assert failure["failure_revalidation"] == {"passed": True}


def test_clean_qualified_import_does_not_create_top_level_src_and_legacy_still_imports() -> None:
    qualified = """
import sys
from tfpd_exploration.src.cdm_p1_m2_local_v1 import replay
assert 'src' not in sys.modules
print('qualified-ok')
"""
    purelib = str(Path(sysconfig.get_paths()["purelib"]))
    env = {"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join((str(ROOT), purelib))}
    out = subprocess.run([sys.executable, "-S", "-c", qualified], cwd=ROOT, env=env,
                         check=True, text=True, capture_output=True)
    assert out.stdout.strip() == "qualified-ok"
    legacy = "from src.cdm_p1_m2_local_v1 import replay; print('legacy-ok')"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "tfpd_exploration"), purelib))
    out = subprocess.run([sys.executable, "-S", "-c", legacy], cwd=ROOT, env=env,
                         check=True, text=True, capture_output=True)
    assert out.stdout.strip() == "legacy-ok"


def test_public_v2_cli_is_inert_and_dry_import_has_no_torch() -> None:
    env = os.environ | {"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": str(ROOT)}
    command = [sys.executable, "-S", "tfpd_exploration/scripts/run_m2_postfusion_checkpoint_score_v2.py", "--dry-run"]
    out = subprocess.run(command, cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    payload = json.loads(out.stdout)
    assert payload["schema"] == plan.SCHEMA and payload["status"] == "INERT_READY_REQUIRES_OPAQUE_CAPABILITY"
    trace = subprocess.run([sys.executable, "-S", "-c", "import sys; import tfpd_exploration.src.m2_postfusion_checkpoint_score_v2.driver; assert 'torch' not in sys.modules; print('clean')"],
                           cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    assert trace.stdout.strip() == "clean"


def test_v2_closure_binds_import_recovery_sources() -> None:
    closed = driver.closure(ROOT)
    assert len(closed["files"]) == len(plan.CLOSURE_RELATIVES)
    assert "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py" in closed["files"]
    assert "tfpd_exploration/src/support_anchored_t4_stage_p_v1/replay.py" in closed["files"]


def test_v2_opaque_admission_binds_actual_v1_failure_without_reserving_or_writing_v2_root() -> None:
    # Issuance is deliberately descriptor-only: no screen/checkpoint/data
    # reader and no successor root reservation occurs before execute.
    assert not (ROOT / plan.RESULT_ROOT_RELATIVE).exists()
    capability = driver.issue_live_capability(ROOT, token=driver._V2_ISSUER_TOKEN)
    assert capability.root == ROOT
    assert capability.predecessor["body_sha256"] == plan.V1_FAILURE_BODIES
    assert capability._inner.profile is driver.V2_PROFILE
    assert capability._inner.lineage_witness == {"v1_import_recovery_failure": capability.predecessor}
    assert not capability.consumed
    assert not (ROOT / plan.RESULT_ROOT_RELATIVE).exists()
