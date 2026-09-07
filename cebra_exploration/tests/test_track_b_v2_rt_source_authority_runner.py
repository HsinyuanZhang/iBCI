"""Synthetic fail-closed tests for the isolated RT source-authority runner.

No test opens an NWB file, target, formal data, CEBRA model, checkpoint, or
GPU.  The worker's source materializer is replaced by tiny float32 records;
the test still verifies that the runner's guarded path layer receives exactly
the 14 source IDs and never the opaque held-out ID.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_rt_source_authority_runner as runner  # noqa: E402
import track_b_v2_source_adapter as source  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _redirect_results_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(runner, "_RESULTS_ROOT", results)
    monkeypatch.setattr(runner, "_FULL_ROOT", results / "rt_full_dev")


def _pointer_validation() -> dict[str, object]:
    return {
        "schema": "track_b_v2_metric_pointer_validation_v2",
        "pointer_body_sha256": "d2e53165cba76aea5f28b893592b7bcf2951b8a4f13bafe8d5a14061b30072b6",
    }


def _synthetic_source_sessions(ids: tuple[str, ...]) -> tuple[source.SourceSessionMaterialization, ...]:
    return tuple(
        source.SourceSessionMaterialization(
            dataset="rt", view=None, session_id=session_id,
            source_path=f"/synthetic/source/{session_id}.nwb", source_nwb_sha256=_sha(session_id),
            neural=np.full((25 + index % 2, 2 + index % 3), float(index + 1), dtype=np.float32),
            dense_behavior=np.stack((np.arange(25 + index % 2), -np.arange(25 + index % 2)), axis=1).astype(np.float32),
            source_trial_count=24,
            rt_t4d_label_provenance={
                "m24_trial_event_count": 24,
                "eligible_endpoint_reach_row_count": 24,
                "unique_endpoint_coordinate_scalar_count": 48,
            },
        )
        for index, session_id in enumerate(ids)
    )


def _install_synthetic_materializer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def fake_canonical_path(dataset: str, session_id: str) -> Path:
        assert dataset == "rt"
        return tmp_path / f"{session_id}.nwb"

    def fake_materialize(*, dataset: str, view: str | None, source_session_ids, source_only_smoke: bool = False):
        assert dataset == "rt" and view is None and source_only_smoke is False
        ids = tuple(source_session_ids)
        calls.append(ids)
        # Go through the runner-installed guard.  If it ever permits a held
        # target, the worker's derivation count/order assertion will fail.
        for session_id in ids:
            source._canonical_source_path("rt", session_id)
        request = source.canonical_source_only_adapter_spec("rt", None, source_session_ids=ids)
        return request, _synthetic_source_sessions(ids)

    monkeypatch.setattr(source, "_canonical_source_path", fake_canonical_path)
    monkeypatch.setattr(source, "materialize_canonical_source_sessions", fake_materialize)
    return calls


def test_worker_materializes_exactly_one_fold_14_sources_and_writes_immutable_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_results_root(monkeypatch, tmp_path)
    calls = _install_synthetic_materializer(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_validate_rt_metric_pointer", _pointer_validation)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root = runner._canonical_root(run_kind="smoke", outer_fold_id="rt_outer_fold_00")
    root.mkdir()

    result = runner.run_rt_source_authority_worker(
        run_kind="smoke", output_root=root, outer_fold_id="rt_outer_fold_00",
    )
    plan, fold = runner._validated_plan_and_fold("rt_outer_fold_00")
    assert calls == [tuple(fold["source_session_ids"])]
    assert len(calls[0]) == 14
    assert fold["opaque_held_out_target_session_id"] not in calls[0]
    assert result["rt_metric_pointer"]["body_sha256"] == _pointer_validation()["pointer_body_sha256"]
    assert result["rt_15fold_source_authority_plan_sha256"] == plan["rt_15fold_source_authority_plan_sha256"]
    isolation = result["process_isolation"]
    assert isolation["source_path_derivation_call_count"] == 14
    assert isolation["source_path_derivation_session_ids"] == fold["source_session_ids"]
    assert isolation["held_out_target_path_derivation_count"] == 0
    assert isolation["held_out_target_data_opened"] is False
    assert result["target_data_opened"] is False
    assert result["cebra_imported"] is False
    assert result["gpu_used"] is False
    assert result["score_emitted"] is False
    assert result["cost"]["cebra_training_mac_count"] == 0

    fold_dir = root / "folds" / "rt_outer_fold_00"
    expected = [runner._BUNDLE_FILENAMES[name] for name in runner._BUNDLE_MEMBER_NAMES]
    expected.append("rt_source_only_fold_execution_receipt.json")
    for filename in expected:
        body = fold_dir / filename
        sidecar = body.with_name(f"{body.name}.sha256")
        assert stat.S_IMODE(body.stat().st_mode) == 0o444
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
        payload, sha = runner._same_fd_pair_payload(body, role="test")
        assert sha
        assert payload["target_data_opened"] is False


def test_smoke_fails_before_reserving_root_or_spawning_child_when_pointer_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_results_root(monkeypatch, tmp_path)

    def invalid_pointer() -> dict[str, object]:
        raise runner.TrackBV2RTSourceAuthorityRunnerError("sealed RT pointer validation failed")

    def no_child(*args, **kwargs):
        raise AssertionError("invalid pointer must block subprocess spawn")

    monkeypatch.setattr(runner, "_validate_rt_metric_pointer", invalid_pointer)
    monkeypatch.setattr(runner.subprocess, "run", no_child)
    with pytest.raises(runner.TrackBV2RTSourceAuthorityRunnerError, match="sealed RT pointer"):
        runner.run_rt_source_authority_smoke()
    assert not list((tmp_path / "results").iterdir())


def test_full_expansion_is_root_flag_gated_and_worker_command_has_no_target_or_gpu_arguments() -> None:
    with pytest.raises(runner.TrackBV2RTSourceAuthorityRunnerError, match="root_authorized_full_run"):
        runner.run_full_rt_15fold_source_authority()
    command = runner._worker_command(
        run_kind="smoke", root=Path("/canonical/output"), outer_fold_id="rt_outer_fold_00",
    )
    assert "--target" not in " ".join(command)
    assert "--gpu" not in command
    assert "--_worker-fold-id" in command


def test_public_smoke_cli_offers_no_target_data_gpu_or_full_batch_option() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_rt_source_authority.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    assert "--smoke" in result.stdout
    assert "--target" not in result.stdout
    assert "--data" not in result.stdout
    assert "--gpu" not in result.stdout
    assert "--full" not in result.stdout
