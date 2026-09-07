"""No-data contracts for immutable remote D-Q4e import preparation."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError, sha256_file, write_immutable_json
from scripts import h1_carrierid_distribution_exposure_remote_q4e_import as remote_import
from scripts.h1_carrierid_distribution_exposure_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS
from scripts.h1_carrierid_distribution_exposure_remote_stage_preflight import STAGE_SCHEMA, STAGE_STATUS
from scripts.h1_carrierid_distribution_exposure_terminal_checker import (
    TERMINAL_PREFLIGHT_SCHEMA,
    TERMINAL_PREFLIGHT_STATUS,
)


def _immutable(path: Path, payload: dict[str, object]) -> Path:
    write_immutable_json(path, payload)
    return path


def _valid_preflight_chain(tmp_path: Path):
    source = _immutable(tmp_path / "source.json", {"schema": PREFLIGHT_SCHEMA, "status": PREFLIGHT_STATUS})
    closure = {"scripts/a.py": "a" * 64, "configs/q4.yaml": "b" * 64}
    terminal = _immutable(
        tmp_path / "terminal.json",
        {
            "schema": TERMINAL_PREFLIGHT_SCHEMA,
            "status": TERMINAL_PREFLIGHT_STATUS,
            "launch": {"authorized": False},
            "source_sha256": closure,
        },
    )
    stage = {
        "schema": STAGE_SCHEMA,
        "status": STAGE_STATUS,
        "scope": {
            "source_nwb_opened": 0,
            "target_opened_or_enumerated": False,
            "minival_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "datamodule_constructed": False,
            "trainer_constructed": False,
            "cuda_constructed_or_launched": False,
            "gpu_queued": False,
        },
        "local_exposure_preflight": {"sha256": sha256_file(source), "status": PREFLIGHT_STATUS},
        "source_closure": closure,
        "terminal_checkpoint_requirement": {
            "path": "q4e/checkpoints/fixed_epoch50/epoch_049.ckpt",
            "schema": "h1_carrierid_h32_fresh_distribution_exposure_terminal_checkpoint_v1",
            "arm": "D-Q4E",
            "fixed_epoch": 49,
            "no_target_before_pair_checker": True,
        },
    }
    return source, terminal, stage


def test_remote_sha_command_is_batch_noninteractive_and_hashes_exact_three_inputs():
    paths = {
        "checkpoint": Path("/remote/q4e/epoch_049.ckpt"),
        "config": Path("/remote/q4e/.hydra/config.yaml"),
        "stage_preflight": Path("/remote/stage.json"),
    }
    command = remote_import._remote_sha256_command(
        ssh_binary="ssh", remote_host="xinyuan@100.103.97.12", paths=paths
    )
    assert command[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "xinyuan@100.103.97.12"]
    assert command[-1].startswith("sha256sum -- ")
    assert "epoch_049.ckpt" in command[-1] and "config.yaml" in command[-1] and "stage.json" in command[-1]


def test_import_requires_remote_stage_to_bind_both_local_immutable_preflights(tmp_path):
    source, terminal, stage = _valid_preflight_chain(tmp_path)
    binding = remote_import._validate_preflight_chain(
        source_preflight=source,
        terminal_preflight=terminal,
        stage=stage,
        expected_remote_stage_sha256="c" * 64,
    )
    assert binding["stage_closure_count"] == 2
    bad = dict(stage)
    bad["source_closure"] = {"scripts/a.py": "f" * 64}
    with pytest.raises(NormalizedV2ContractError, match="closure disagrees"):
        remote_import._validate_preflight_chain(
            source_preflight=source,
            terminal_preflight=terminal,
            stage=bad,
            expected_remote_stage_sha256="c" * 64,
        )


def test_remote_import_tool_has_no_target_loader_or_training_path():
    source = Path(remote_import.__file__).read_text(encoding="utf-8")
    assert "load_target_records" not in source
    assert "DataModule(" not in source
    assert "trainer.fit" not in source
    assert "--expected-remote-stage-preflight-sha256" in source
    assert "sha256sum --" in source
