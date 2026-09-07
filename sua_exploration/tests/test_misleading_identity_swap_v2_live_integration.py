from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mc_maze import misleading_identity_swap_v2_core as core
from scripts import misleading_identity_swap_v2_build_source_authority as builder
from scripts import misleading_identity_swap_v2_scorer as scorer
from scripts.misleading_identity_swap_v2_scorer import _source_stats
from scripts import misleading_identity_swap_v2_aggregate as aggregate
from scripts import misleading_identity_swap_v2_queue_verify as queue_verify
from scripts import misleading_identity_swap_v2_train_cell as train_cell
from scripts.misleading_identity_swap_v2_train_cell import cell_factors


@pytest.mark.parametrize("cell,expected", [
    ("clean_z4", ("clean", "z4")), ("clean_t4", ("clean", "t4")),
    ("swap_z4", ("swap", "z4")), ("swap_t4", ("swap", "t4")),
])
def test_four_cell_factorization(cell: str, expected: tuple[str, str]) -> None:
    assert cell_factors(cell) == expected


def test_source_lineage_normalizer_is_value_bound() -> None:
    from mc_maze.unit_side_features import side_feature_stats_sha256
    mean = np.asarray([1, 2, 3, 4], dtype=np.float32)
    std = np.asarray([2, 3, 4, 5], dtype=np.float32)
    lineage = {"descriptor": {
        "normalizer_mean_float32": mean.tolist(),
        "normalizer_std_float32": std.tolist(),
        "normalizer_value_sha256": side_feature_stats_sha256(mean, std),
    }}
    observed = _source_stats(lineage)
    assert np.array_equal(observed[0], mean)
    lineage["descriptor"]["normalizer_mean_float32"][0] += 1
    with pytest.raises(core.SwapV2ContractError, match="value SHA drift"):
        _source_stats(lineage)


def test_source_builder_dry_run_is_inert() -> None:
    result = subprocess.run(
        ["/home/xinyuan/miniconda3/envs/spint/bin/python", str(Path(builder.__file__))],
        check=True, capture_output=True, text=True,
        env={"PYTHONNOUSERSITE": "1", "PYTHONPATH": "sua_exploration:streaming_calibration_exp"},
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRY_RUN__NO_DATA_NO_GPU"
    assert payload["source_session_count"] == 27
    assert payload["target_nwb_opened"] is False
    assert payload["formal_subc_test_nwb_opened"] is False


def test_queue_bridge_is_plan_only_and_never_opens_target() -> None:
    path = core.SUA_ROOT / "scripts" / "misleading_identity_swap_v2_queue_bridge.sh"
    result = subprocess.run(["bash", str(path), "--dry-run"], check=True,
                            capture_output=True, text=True)
    assert "DRY_RUN__NO_TARGET_NO_GPU" in result.stdout
    denied = subprocess.run(["bash", str(path), "--launch"], check=False,
                            capture_output=True, text=True)
    assert denied.returncode == 3
    assert "independent ROOT GO" in denied.stderr


def test_target_authority_sibling_bytes_are_exact(tmp_path: Path) -> None:
    import torch
    descriptors = {"session": torch.arange(32, dtype=torch.float64).reshape(8, 4)}
    payload = core.build_target_diagnostic_authority(descriptors, domain="within_subject")
    body, _side, digest = core.write_immutable_json_pair(tmp_path / "authority.json", payload)
    left = core.load_verified_runtime_authority(body, expected_kind="target_diagnostic")
    right = core.load_verified_runtime_authority(body, expected_kind="target_diagnostic")
    assert left.sha256 == right.sha256 == digest
    core.verify_sibling_authority_bindings({"t4": digest, "z4": digest}, expected_sha256=digest)


def test_full_a2_trial30_semantics_and_normalizer_poison_fail_closed() -> None:
    parent = core.load_a2_parent_domain_bindings("within_subject")
    session = parent["domain_sessions"][0]
    observed = dict(parent["session_query_receipts"][session])
    scorer.require_exact_a2_trial30_semantics(
        observed, parent["session_query_receipts"], session=session
    )
    observed["activity_support_original_trial_indices"] = list(
        reversed(observed["activity_support_original_trial_indices"])
    )
    with pytest.raises(core.SwapV2ContractError, match="full trial-30 semantics"):
        scorer.require_exact_a2_trial30_semantics(
            observed, parent["session_query_receipts"], session=session
        )
    normalizer = dict(parent["normalizer_authority"])
    scorer.require_exact_a2_normalizer_authority(normalizer, parent["normalizer_authority"])
    normalizer["behavior_normalizer_value_sha256"] = "0" * 64
    with pytest.raises(core.SwapV2ContractError, match="normalizer authority"):
        scorer.require_exact_a2_normalizer_authority(normalizer, parent["normalizer_authority"])


def test_target_and_score_output_conflicts_fail_before_any_access(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target.json"; target.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(scorer, "_domain_paths", lambda _domain: (_ for _ in ()).throw(AssertionError("target resolved")))
    with pytest.raises(FileExistsError):
        scorer.prepare_target_authority(
            domain="within_subject", source_lineage_path=tmp_path / "missing-lineage.json",
            authority_out=target, lineage_out=tmp_path / "lineage.json",
            official_preflight_path=tmp_path / "missing-official.json",
        )


def test_target_prepare_rejects_nonofficial_source_lineage_before_target_resolution(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        scorer, "_domain_paths",
        lambda _domain: (_ for _ in ()).throw(AssertionError("target resolved")),
    )
    legacy = core.SUA_ROOT / "results" / "misleading_identity_swap_v2_source_authority_dev" / "strict27_m30_source_lineage_v2.json"
    with pytest.raises(core.SwapV2ContractError, match="source lineage differs"):
        scorer.prepare_target_authority(
            domain="within_subject", source_lineage_path=legacy,
            authority_out=tmp_path / "target.json", lineage_out=tmp_path / "lineage.json",
            official_preflight_path=core.OFFICIAL_PREFLIGHT_PATH,
        )
    score = tmp_path / "score.json"; core.sidecar_path(score).write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        scorer.execute_score(
            cell="clean_t4", domain="within_subject", mode="clean",
            terminal_receipt_path=tmp_path / "missing-terminal.json",
            target_authority_path=tmp_path / "missing-target.json",
            target_lineage_path=tmp_path / "missing-target-lineage.json",
            source_lineage_path=tmp_path / "missing-source-lineage.json",
            out_path=score, device_name="cuda:0",
            official_preflight_path=tmp_path / "missing-official.json",
        )


def _terminal_fixture(cell: str) -> tuple[dict, dict, str, dict]:
    from scripts.misleading_identity_swap_v2_preflight import (
        current_implementation_bindings, load_verified_official_preflight,
    )
    official, official_sha = load_verified_official_preflight(core.OFFICIAL_PREFLIGHT_PATH)
    live = current_implementation_bindings()
    terminal = {
        "receipt_kind": "misleading_identity_swap_v2_cell_terminal",
        "status": "CELL_TRAINING_COMPLETE__DEVELOPMENT_NOT_OFFICIAL",
        "cell": cell, "seed": 42, "epochs": 12,
        "score_epochs_one_based": list(range(5, 13)),
        "matching_authority_path": official["matching_authority_path"],
        "matching_authority_sha256": official["matching_authority_sha256"],
        "source_lineage_path": official["source_lineage_path"],
        "source_lineage_sha256": official["source_lineage_sha256"],
        "initial_state_path": official["initial_state_path"],
        "initial_state_file_sha256": official["initial_state_file_sha256"],
        "initial_state_dict_sha256": official["initial_state_dict_sha256"],
        "official_preflight_path": str(core.OFFICIAL_PREFLIGHT_PATH.resolve()),
        "official_preflight_sha256": official_sha,
        "implementation_bindings": live,
        "implementation_bindings_sha256": official["implementation_bindings_sha256"],
        "cell_output_root": official["cell_output_root"],
        "cell_output_path": official["cell_output_paths"][cell],
        "python_isolation": {"python": "synthetic-shared"},
        "torch_runtime": {"runtime": "synthetic-shared"},
    }
    return terminal, official, official_sha, live


def test_canonical_cell_output_and_initial_freshness_precede_setup(tmp_path: Path, monkeypatch) -> None:
    _terminal, official, _sha, _live = _terminal_fixture("clean_t4")
    expected = Path(official["cell_output_paths"]["clean_t4"])
    assert train_cell.require_canonical_cell_output(
        official, cell="clean_t4", output_dir=expected
    ) == expected.resolve()
    with pytest.raises(core.SwapV2ContractError, match="unique canonical"):
        train_cell.require_canonical_cell_output(
            official, cell="clean_t4", output_dir=tmp_path / "sibling"
        )
    occupied = tmp_path / "initial.pt"; occupied.write_bytes(b"occupied")
    monkeypatch.setattr(train_cell, "_model", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("model setup")))
    with pytest.raises(FileExistsError):
        train_cell.mint_initial_state(tmp_path / "missing-authority.json", occupied)


@pytest.mark.parametrize("poison,pattern", [
    ("implementation_bindings", "implementation closure drift"),
    ("matching_authority_path", "matching_authority_path differs"),
    ("initial_state_file_sha256", "initial_state_file_sha256 differs"),
])
def test_aggregate_terminal_official_binding_poison(
    poison: str, pattern: str,
) -> None:
    terminal, official, official_sha, live = _terminal_fixture("clean_t4")
    if poison == "implementation_bindings": terminal[poison] = {}
    else: terminal[poison] = "0" * 64
    with pytest.raises(core.SwapV2ContractError, match=pattern):
        aggregate.validate_terminal_against_official(
            terminal, cell="clean_t4", official=official,
            official_sha=official_sha, current_bindings=live,
        )


def test_four_terminal_runtime_and_path_drift_fail_closed() -> None:
    terminals = {cell: _terminal_fixture(cell)[0] for cell in core.CELLS}
    aggregate.validate_four_terminal_common(terminals)
    terminals["swap_t4"]["torch_runtime"] = {"runtime": "poison"}
    with pytest.raises(core.SwapV2ContractError, match="torch_runtime"):
        aggregate.validate_four_terminal_common(terminals)


@pytest.mark.parametrize("poison,pattern", [
    ("partial", "partial terminal"),
    ("symlink", "symlink"),
    ("status", "status/kind"),
    ("preflight", "official preflight drift"),
    ("closure", "live closure drift"),
])
def test_queue_terminal_adversarial_validation(
    tmp_path: Path, poison: str, pattern: str,
) -> None:
    terminal, official, official_sha, live = _terminal_fixture("clean_t4")
    path = tmp_path / "terminal.json"
    if poison == "partial":
        path.write_text("{}", encoding="utf-8")
    elif poison == "symlink":
        real, real_side, _ = core.write_immutable_json_pair(tmp_path / "real.json", terminal)
        path.symlink_to(real); core.sidecar_path(path).symlink_to(real_side)
    else:
        if poison == "status": terminal["status"] = "POISON"
        if poison == "preflight": terminal["official_preflight_sha256"] = "0" * 64
        if poison == "closure": terminal["implementation_bindings"] = {}
        core.write_immutable_json_pair(path, terminal)
    with pytest.raises((core.SwapV2ContractError, FileExistsError), match=pattern):
        queue_verify.verify_terminal(
            path, cell="clean_t4", official=official, official_sha=official_sha,
            live=live, require_canonical_path=False,
        )
