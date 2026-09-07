"""No-target unit contracts for the H1 date-LODO Phase-2 terminal checker."""
from __future__ import annotations

import json
from pathlib import Path
import stat

from omegaconf import OmegaConf
import pytest
import torch

from scripts.h1_carrierid_date_lodo_phase2_terminal_checker import (
    CHECKPOINT_SCHEMA,
    DateLodoTerminalError,
    RUNTIME_INIT_PROBE_SCHEMA,
    RUNTIME_INIT_PROBE_STATUS,
    _finite_state_dict,
    check_pair,
    check_single,
)
from src.h1_m4_cce_contract import canonical_sha256, sha256_file


def _immutable_json(path: Path, body: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)
    return path


def _fixture_receipts(tmp_path: Path, *, outer_date: str = "19250108") -> tuple[Path, Path, Path]:
    phase1 = _immutable_json(tmp_path / "phase1.json", {"schema": "source", "status": "pass"})
    source = {
        "preflight_path": str(phase1), "preflight_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64, "outer_date": outer_date,
    }
    pair = _immutable_json(tmp_path / "pair.json", {
        "schema": "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1",
        "status": "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED",
        "outer_date": outer_date,
        "source_binding": source, "source_binding_sha256": canonical_sha256(source),
        "fresh_models": {
            "h_s": {"initial_state_sha256": "c" * 64},
            "h_c": {"initial_state_sha256": "d" * 64},
        },
        "code_sha256": {
            "model": sha256_file(Path(__file__).resolve().parents[1] / "src/models/h1_carrierid_date_lodo_phase2_module.py"),
            "data": sha256_file(Path(__file__).resolve().parents[1] / "src/data/h1_carrierid_date_lodo_phase2.py"),
        },
    })
    launch = _immutable_json(tmp_path / "launch.json", {
        "schema": "h1_carrierid_date_lodo_phase2_paired_source_launch_receipt_v1",
        "status": "PASS_PAIRED_SOURCE_TRAINING_PREPARED_NOT_LAUNCHED",
        "outer_date": outer_date, "pair_preflight_path": str(pair), "pair_preflight_sha256": sha256_file(pair),
    })
    return phase1, pair, launch


def _runtime_probe(*, path: Path, pair: Path, config: Path, observed_hash: str,
                   outer_date: str | None = None, pair_path: Path | None = None) -> Path:
    """Make a synthetic immutable source-only H-S bridge receipt for checker tests."""

    pair_body = json.loads(pair.read_text(encoding="utf-8"))
    bound_pair = pair if pair_path is None else pair_path
    return _immutable_json(path, {
        "schema": RUNTIME_INIT_PROBE_SCHEMA, "status": RUNTIME_INIT_PROBE_STATUS,
        "arm": "H-S", "outer_date": str(pair_body["outer_date"] if outer_date is None else outer_date),
        "pair_preflight": {"path": str(bound_pair), "sha256": sha256_file(bound_pair)},
        "source_binding_sha256": pair_body["source_binding_sha256"],
        "production_hs_config": {"path": str(config), "sha256": sha256_file(config)},
        "runtime_initialization": {
            "hash_domain": "lightning_wrapper_state_dict_after_gpu_transfer_and_lazy_materialization",
            "initial_state_sha256": observed_hash,
            "capture_hook": "on_train_batch_start(epoch=0,batch_idx=0)",
            "capture_precedes_training_step_and_first_optimizer_step": True,
            "optimizer_steps_before_capture": 0, "backward_steps_before_capture": 0,
            "probe_optimizer_steps_after_capture": 1, "seed": 42, "accelerator": "gpu", "precision": "32-true",
        },
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                  "formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False,
                  "checkpoint_loaded": False, "checkpoint_created": False},
        "code_sha256": {
            **pair_body["code_sha256"],
            "runtime_probe": sha256_file(
                Path(__file__).resolve().parents[1] / "scripts/h1_carrierid_date_lodo_phase2_preflight.py"
            ),
        },
    })


def _write_checkpoint(tmp_path: Path, *, arm: str, pair: Path, phase1: Path) -> Path:
    pair_body = json.loads(pair.read_text(encoding="utf-8"))
    outer_date = str(pair_body["outer_date"])
    run = tmp_path / arm.lower().replace("-", "")
    config_path = run / ".hydra" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    cfg = {
        "protocol_id": f"h1_carrierid_date_lodo_phase2_{arm.lower().replace('-', '')}_{outer_date}_source_only_v1",
        "train": True, "test": False, "ckpt_path": None, "seed": 42,
        "phase2": {"outer_date": outer_date, "arm": arm, "phase1_preflight_path": str(phase1),
                   "pair_preflight_path": str(pair)},
        "data": {"phase1_preflight_path": str(phase1),
                 "_target_": "src.data.h1_carrierid_date_lodo_phase2.H1CarrierIdDateLodoSourceDataModule"},
        "model": {"arm": arm, "fixed_seed": 42, "net": {"_target_": {
            "H-S": "src.models.components.spint.SpintModel",
            "H-C": "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
        }[arm]}},
        "trainer": {"max_epochs": 50, "min_epochs": 50, "limit_val_batches": 0,
                    "num_sanity_val_steps": 0, "accelerator": "gpu", "devices": 1, "precision": "32-true"},
        "callbacks": {"fixed_epoch50": {"monitor": None, "every_n_epochs": 50, "save_top_k": -1, "save_last": False}},
    }
    OmegaConf.save(config=OmegaConf.create(cfg), f=config_path)
    source = pair_body["source_binding"]
    fresh = pair_body["fresh_models"]["h_s" if arm == "H-S" else "h_c"]
    checkpoint = run / "checkpoints" / "fixed_epoch50" / "epoch_049.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({
        "epoch": 49, "global_step": 50, "state_dict": {"weight": torch.tensor([1.0])},
        "h1_carrierid_date_lodo_phase2": {
            "schema": CHECKPOINT_SCHEMA, "arm": arm, "outer_date": outer_date, "fresh_seed": 42,
            "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
            "initial_state_sha256": fresh["initial_state_sha256"],
            "phase2_source_binding_sha256": canonical_sha256(source),
            "phase1_source_manifest_sha256": source["source_manifest_sha256"],
            "phase1_preflight_sha256": source["preflight_sha256"],
            "config_sha256": sha256_file(config_path), "target_optimizer_steps": 0,
            "target_backward_steps": 0, "checkpoint_warm_start": False,
        },
    }, checkpoint)
    return checkpoint


def test_terminal_checker_is_checkpoint_and_receipt_only() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/h1_carrierid_date_lodo_phase2_terminal_checker.py").read_text(encoding="utf-8")
    assert "from src.data" not in source
    assert "import lightning" not in source
    assert "load_target" not in source
    assert "_hc_wrapper_initial_hash" in source
    assert "hs-runtime-init-probe" in source
    assert "torch.cuda" not in source
    assert '"target_evaluator_status": "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"' in source


def test_terminal_checker_rejects_nonfinite_checkpoint_state() -> None:
    _finite_state_dict({"finite": torch.tensor([1.0])})
    with pytest.raises(DateLodoTerminalError, match="nonfinite"):
        _finite_state_dict({"nan": torch.tensor([float("nan")])})
    with pytest.raises(DateLodoTerminalError, match="not a tensor"):
        _finite_state_dict({"bad": 1})


def test_single_and_pair_checker_bind_e49_config_and_common_source_schedule(tmp_path: Path) -> None:
    phase1, pair, launch = _fixture_receipts(tmp_path)
    hs = _write_checkpoint(tmp_path, arm="H-S", pair=pair, phase1=phase1)
    hc = _write_checkpoint(tmp_path, arm="H-C", pair=pair, phase1=phase1)
    hs_result = check_single(checkpoint_path=hs, arm="H-S", pair_preflight_path=pair, launch_receipt_path=launch,
                             output_path=tmp_path / "hs_terminal.json")
    assert hs_result["status"].startswith("PASS_")
    assert stat.S_IMODE(Path(hs_result["receipt_path"]).stat().st_mode) == 0o444
    pair_result = check_pair(hs_checkpoint_path=hs, hc_checkpoint_path=hc, pair_preflight_path=pair,
                             launch_receipt_path=launch, output_path=tmp_path / "pair_terminal.json")
    assert pair_result["status"].startswith("PASS_")
    assert pair_result["h_s"]["metadata"]["phase2_source_binding_sha256"] == pair_result["h_c"]["metadata"]["phase2_source_binding_sha256"]


def test_hs_wrapper_lazy_mismatch_requires_exact_date_immutable_runtime_probe(tmp_path: Path) -> None:
    phase1, pair, launch = _fixture_receipts(tmp_path, outer_date="19250113")
    hs = _write_checkpoint(tmp_path, arm="H-S", pair=pair, phase1=phase1)
    payload = torch.load(hs, map_location="cpu", weights_only=False)
    observed_hash = "e" * 64
    payload["h1_carrierid_date_lodo_phase2"]["initial_state_sha256"] = observed_hash
    torch.save(payload, hs)
    config = hs.parent.parent.parent / ".hydra" / "config.yaml"
    probe = _runtime_probe(path=tmp_path / "hs_runtime_probe.json", pair=pair, config=config,
                           observed_hash=observed_hash)
    with pytest.raises(DateLodoTerminalError, match="runtime-init probe"):
        check_single(checkpoint_path=hs, arm="H-S", pair_preflight_path=pair, launch_receipt_path=launch,
                     output_path=tmp_path / "without_probe.json")
    result = check_single(checkpoint_path=hs, arm="H-S", pair_preflight_path=pair, launch_receipt_path=launch,
                          output_path=tmp_path / "with_probe.json", hs_runtime_init_probe_path=probe)
    assert result["checkpoint"]["initial_state_verification"]["mode"] == "hs_gpu_lazy_runtime_probe"


def test_hs_runtime_probe_rejects_wrong_date_tamper_and_another_pair_receipt(tmp_path: Path) -> None:
    """A hash-domain bridge may not be reused across dates or immutable pairs."""

    phase1, pair, launch = _fixture_receipts(tmp_path / "first", outer_date="19250113")
    hs = _write_checkpoint(tmp_path / "first", arm="H-S", pair=pair, phase1=phase1)
    payload = torch.load(hs, map_location="cpu", weights_only=False)
    observed_hash = "e" * 64
    payload["h1_carrierid_date_lodo_phase2"]["initial_state_sha256"] = observed_hash
    torch.save(payload, hs)
    config = hs.parent.parent.parent / ".hydra" / "config.yaml"
    wrong_date = _runtime_probe(path=tmp_path / "wrong_date.json", pair=pair, config=config,
                                observed_hash=observed_hash, outer_date="19250115")
    with pytest.raises(DateLodoTerminalError, match="outer date/arm"):
        check_single(checkpoint_path=hs, arm="H-S", pair_preflight_path=pair, launch_receipt_path=launch,
                     output_path=tmp_path / "wrong_date_out.json", hs_runtime_init_probe_path=wrong_date)

    _, other_pair, _ = _fixture_receipts(tmp_path / "second", outer_date="19250113")
    wrong_pair = _runtime_probe(path=tmp_path / "wrong_pair.json", pair=pair, config=config,
                                observed_hash=observed_hash, pair_path=other_pair)
    with pytest.raises(DateLodoTerminalError, match="another pair preflight"):
        check_single(checkpoint_path=hs, arm="H-S", pair_preflight_path=pair, launch_receipt_path=launch,
                     output_path=tmp_path / "wrong_pair_out.json", hs_runtime_init_probe_path=wrong_pair)

    tampered = _runtime_probe(path=tmp_path / "tampered.json", pair=pair, config=config, observed_hash=observed_hash)
    body = json.loads(tampered.read_text(encoding="utf-8"))
    tampered.chmod(0o644)
    body["runtime_initialization"]["initial_state_sha256"] = "f" * 64
    tampered.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    tampered.chmod(0o444)
    with pytest.raises(DateLodoTerminalError, match="does not equal terminal checkpoint metadata"):
        check_single(checkpoint_path=hs, arm="H-S", pair_preflight_path=pair, launch_receipt_path=launch,
                     output_path=tmp_path / "tampered_out.json", hs_runtime_init_probe_path=tampered)


def test_terminal_checker_derives_a_nonlegacy_outer_date_from_pair_preflight(tmp_path: Path) -> None:
    phase1, pair, launch = _fixture_receipts(tmp_path)
    pair_body = json.loads(pair.read_text(encoding="utf-8"))
    pair_body["outer_date"] = "19250113"
    pair_body["source_binding"]["outer_date"] = "19250113"
    pair_body["source_binding_sha256"] = canonical_sha256(pair_body["source_binding"])
    pair.chmod(0o644)
    pair.write_text(json.dumps(pair_body, sort_keys=True), encoding="utf-8")
    pair.chmod(0o444)
    launch_body = json.loads(launch.read_text(encoding="utf-8"))
    launch_body["outer_date"] = "19250113"
    launch_body["pair_preflight_sha256"] = sha256_file(pair)
    launch.chmod(0o644)
    launch.write_text(json.dumps(launch_body, sort_keys=True), encoding="utf-8")
    launch.chmod(0o444)
    hs = _write_checkpoint(tmp_path, arm="H-S", pair=pair, phase1=phase1)
    hc = _write_checkpoint(tmp_path, arm="H-C", pair=pair, phase1=phase1)
    result = check_pair(hs_checkpoint_path=hs, hc_checkpoint_path=hc, pair_preflight_path=pair,
                        launch_receipt_path=launch, output_path=tmp_path / "pair_terminal_19250113.json")
    assert result["status"].startswith("PASS_")
    body = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert body["outer_date"] == "19250113"
