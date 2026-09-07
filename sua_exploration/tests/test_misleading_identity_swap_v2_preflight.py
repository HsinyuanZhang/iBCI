from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

from mc_maze import misleading_identity_swap_v2_core as core
from scripts.misleading_identity_swap_v2_preflight import (
    build_development_preflight, load_verified_official_preflight,
)
from scripts.misleading_identity_swap_v2_scorer import (
    build_synthetic_score_receipt,
    validate_synthetic_score_receipt,
)
from mc_maze.misleading_identity_swap_v2_trainer import (
    MisleadingIdentitySwapV2LitModule,
)


def _authority() -> dict:
    descriptor = torch.arange(8 * 4, dtype=torch.float64).reshape(8, 4)
    return core.build_matching_authority({"synthetic_source_session": descriptor})


def test_cpu_preflight_is_non_authorizing_and_opens_nothing(tmp_path: Path) -> None:
    result_root = tmp_path / "fresh_result_root"
    receipt = build_development_preflight(
        authority_payload=_authority(), result_root=result_root
    )
    assert receipt["status"] == core.PREFLIGHT_STATUS
    assert receipt["non_authorizing"] is True
    assert receipt["official_preflight_minted"] is False
    assert receipt["data_opened"] is False
    assert receipt["checkpoint_loaded"] is False
    assert receipt["target_nwb_opened"] is False
    assert receipt["formal_subc_test_nwb_opened"] is False
    assert receipt["gpu_used"] is False
    assert receipt["training_started"] is False
    assert receipt["fresh_training_cells"] == list(core.CELLS)
    assert len(receipt["implementation_bindings"]) >= 10
    assert not result_root.exists()


def test_preflight_rejects_nonfresh_result_root(tmp_path: Path) -> None:
    result_root = tmp_path / "occupied"
    result_root.mkdir()
    (result_root / "partial.json").write_text("{}", encoding="utf-8")
    try:
        build_development_preflight(
            authority_payload=_authority(), result_root=result_root
        )
    except core.SwapV2ContractError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("occupied result root did not fail closed")


def test_canonical_official_preflight_is_immutable_live_and_tamper_closed(tmp_path: Path) -> None:
    payload, digest = load_verified_official_preflight(core.OFFICIAL_PREFLIGHT_PATH)
    assert payload["official"] is True
    assert payload["status"] == core.OFFICIAL_PREFLIGHT_STATUS
    assert len(digest) == 64
    assert core.OFFICIAL_PREFLIGHT_PATH.stat().st_mode & 0o777 == 0o444
    body, _side, _sha = core.write_immutable_json_pair(tmp_path / "copied.json", payload)
    try:
        load_verified_official_preflight(body)
    except core.SwapV2ContractError as exc:
        assert "not canonical" in str(exc)
    else:
        raise AssertionError("caller-supplied official preflight copy was accepted")


def test_successor_binds_invalid_v2_attempt_and_rejects_old_official_path() -> None:
    """V3 may cite the failed launch but may never resume its V2 authority/root."""
    payload, _digest = load_verified_official_preflight(core.OFFICIAL_PREFLIGHT_PATH)
    assert payload["preserved_invalid_v2_attempt"]["launch_receipt_sha256"] == (
        core.INVALID_V2_ATTEMPT_SHA256
    )
    old = (
        core.SUA_ROOT / "results" / "misleading_identity_swap_v2_source_authority_dev" /
        "official_cpu_preflight_v2.json"
    )
    try:
        load_verified_official_preflight(old)
    except core.SwapV2ContractError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("predecessor V2 official preflight was accepted by V3")


def test_runner_default_is_inert_and_live_launch_needs_explicit_cell(tmp_path: Path) -> None:
    runner = core.SUA_ROOT / "scripts" / "misleading_identity_swap_v2_runner.sh"
    dry = subprocess.run(
        ["bash", str(runner), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "NO_DATA_NO_GPU" in dry.stdout
    assert "LIVE_INTEGRATION_PLAN__NO_DATA_NO_GPU" in dry.stdout
    assert "official_cpu_preflight_v3.json" in dry.stdout
    assert "stage_p_successor_v3_seed42" in dry.stdout
    launch = subprocess.run(
        ["bash", str(runner), "--launch-cell"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert launch.returncode != 0
    assert "CELL" in launch.stderr
    assert list(tmp_path.iterdir()) == []


def test_synthetic_scorer_schema_cannot_claim_target_or_gpu() -> None:
    authority = core.VerifiedMatchingAuthority.from_payload(_authority())
    receipt = build_synthetic_score_receipt(
        cell="swap_t4",
        domain="external_subject_M",
        evaluation_input_mode="swapped_diagnostic",
        pooled_r2=0.25,
        authority_sha256=authority.sha256,
    )
    assert validate_synthetic_score_receipt(receipt) == receipt
    assert receipt["status"] == "SYNTHETIC_TEST_ONLY__NOT_A_RESULT"
    assert receipt["target_nwb_opened"] is False
    assert receipt["formal_subc_test_nwb_opened"] is False
    assert receipt["gpu_used"] is False
    assert receipt["target_optimizer_or_backward_steps"] == 0


def test_trainer_session_guard_runs_on_batch_metadata_before_forward() -> None:
    single = (
        torch.empty(2, 1, 1),
        torch.empty(2, 1, 1),
        torch.empty(2, 1, 1, 1),
        ["session_a", "session_a"],
    )
    assert MisleadingIdentitySwapV2LitModule._single_session(single) == "session_a"
    mixed = (*single[:3], ["session_a", "session_b"])
    try:
        MisleadingIdentitySwapV2LitModule._single_session(mixed)
    except ValueError as exc:
        assert "single nonempty session" in str(exc)
    else:
        raise AssertionError("mixed-session batch was not rejected before forward")


def test_lightning_sanity_validation_is_clean_eval_not_standalone_score() -> None:
    """Regression for the v2 source launch abort before epoch zero."""
    import torch

    module = object.__new__(MisleadingIdentitySwapV2LitModule)
    torch.nn.Module.__init__(module)
    module.training = False
    module._swap_v2_identity_training_mode = "swap"
    module._swap_v2_evaluation_input_mode = "clean"
    module._swap_v2_explicit_evaluation_epoch = None
    module._trainer = type("Trainer", (), {
        "validating": False, "sanity_checking": True, "current_epoch": 0,
    })()
    assert module._runtime_phase_and_epoch() == ("eval_clean", 0)

    module._swap_v2_evaluation_input_mode = "swapped_diagnostic"
    try:
        module._runtime_phase_and_epoch()
    except RuntimeError as exc:
        assert "sanity/validation" in str(exc)
    else:
        raise AssertionError("sanity validation accepted the swapped diagnostic mode")

    module._swap_v2_evaluation_input_mode = "clean"
    module._trainer = type("Trainer", (), {
        "validating": False, "sanity_checking": False, "current_epoch": 0,
    })()
    try:
        module._runtime_phase_and_epoch()
    except RuntimeError as exc:
        assert "stand-alone" in str(exc)
    else:
        raise AssertionError("stand-alone score escaped its explicit checkpoint-epoch requirement")
