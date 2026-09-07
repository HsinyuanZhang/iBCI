"""Fail-closed source-checkpoint contracts for the H-C0 five-date evaluator.

These tests construct only a tiny Lightning-shaped checkpoint in memory.  No
NWB, target dataset, model instantiation, or evaluator execution is involved.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts import h1_carrierid_hc0_fivedate_evaluate as hc0


DATE = "19250108"


def _counter(value: int, *, include_started: bool = True) -> dict[str, int]:
    row = {"ready": value, "completed": value}
    if include_started:
        row["started"] = value
    return row


def _payload(config_sha: str) -> dict[str, Any]:
    batches = hc0._expected_source_batches(DATE)
    steps = hc0._expected_global_step(DATE)
    return {
        "epoch": 49,
        "global_step": steps,
        "pytorch-lightning_version": hc0.EXPECTED_LIGHTNING_VERSION,
        "optimizer_states": [{"state": {"0": {"step": torch.tensor(steps, dtype=torch.float32)}}, "param_groups": []}],
        "lr_schedulers": [],
        "state_dict": {"weight": torch.ones(2, dtype=torch.float32)},
        "h1_carrierid_date_lodo_phase2": {
            "checkpoint_epoch_zero_based": 49,
            "epochs_completed": 50,
            "config_sha256": config_sha,
        },
        "loops": {
            "fit_loop": {
                "epoch_progress": {
                    "total": {"ready": 50, "completed": 49, "started": 50, "processed": 50},
                    "current": {"ready": 50, "completed": 49, "started": 50, "processed": 50},
                },
                "epoch_loop.state_dict": {"_batches_that_stepped": steps},
                "epoch_loop.batch_progress": {
                    "total": {"ready": steps, "completed": steps, "started": steps, "processed": steps},
                    "current": {"ready": batches, "completed": batches, "started": batches, "processed": batches},
                    "is_last_batch": True,
                },
                "epoch_loop.automatic_optimization.optim_progress": {
                    "optimizer": {
                        "step": {"total": _counter(steps, include_started=False), "current": _counter(batches, include_started=False)},
                        "zero_grad": {"total": _counter(steps), "current": _counter(batches)},
                    },
                },
                "epoch_loop.scheduler_progress": {
                    "total": _counter(0, include_started=False), "current": _counter(0, include_started=False),
                },
            },
        },
    }


@pytest.fixture
def terminal_payload(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    config = tmp_path / "config.yaml"
    config.write_text("protocol_id: synthetic-hc0\n", encoding="utf-8")
    return _payload(hc0.sha256_file(config)), config


def test_valid_terminal_checkpoint_contract_passes(terminal_payload: tuple[dict[str, Any], Path]) -> None:
    payload, config = terminal_payload
    metadata = hc0._validate_terminal_checkpoint_payload(payload, date=DATE, config_path=config)
    assert metadata["epochs_completed"] == 50
    assert payload["global_step"] == 167800


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda value: value.update(epoch=48), "top-level epoch"),
        (lambda value: value.update(global_step=1), "global_step"),
        (lambda value: value.update(**{"pytorch-lightning_version": "2.3.0"}), "Lightning version"),
        (lambda value: value["h1_carrierid_date_lodo_phase2"].update(epochs_completed=49), "50 completed"),
        (lambda value: value["h1_carrierid_date_lodo_phase2"].update(config_sha256="0" * 64), "config SHA"),
        (lambda value: value["state_dict"].update(weight=torch.tensor([float("nan")])), "state_dict entry"),
        (lambda value: value["optimizer_states"][0]["state"]["0"].update(step=torch.tensor(float("nan"))), "optimizer_states"),
        (lambda value: value["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"].update(completed=1), "batch_progress"),
        (lambda value: value.update(lr_schedulers=[{}]), "scheduler"),
    ],
)
def test_terminal_checkpoint_contract_rejects_each_provenance_gap(
    terminal_payload: tuple[dict[str, Any], Path], mutate, match: str,
) -> None:
    payload, config = terminal_payload
    mutate(deepcopy(payload))
    broken = deepcopy(payload)
    mutate(broken)
    with pytest.raises(ValueError, match=match):
        hc0._validate_terminal_checkpoint_payload(broken, date=DATE, config_path=config)


def test_fixed_batch_schedule_is_explicit_and_complete() -> None:
    assert tuple(hc0.EXPECTED_SOURCE_BATCHES_PER_EPOCH) == tuple(hc0.CONFIRMATORY_DATES)
    assert {date: hc0._expected_global_step(date) for date in hc0.CONFIRMATORY_DATES} == {
        "19250108": 167800,
        "19250113": 171100,
        "19250115": 172700,
        "19250119": 170200,
        "19250120": 170950,
    }


def _resolved_config(run_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        train=True,
        test=False,
        ckpt_path=None,
        trainer=SimpleNamespace(min_epochs=50, max_epochs=50, limit_val_batches=0, num_sanity_val_steps=0),
        data=SimpleNamespace(fixed_epochs=50),
        phase2=SimpleNamespace(outer_date=int(DATE)),
        callbacks={
            "fixed_epoch50": {
                "dirpath": "${paths.output_dir}/checkpoints/fixed_epoch50",
                "monitor": None,
                "every_n_epochs": 50,
                "save_last": False,
                "save_top_k": -1,
                "filename": "epoch_{epoch:03d}",
                "auto_insert_metric_name": False,
            },
        },
    )


def test_resolved_config_lifecycle_gate_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    hc0._validate_resolved_hc0_config(_resolved_config(run_dir), date=DATE, run_dir=run_dir)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda cfg: setattr(cfg, "train", False), "train/test lifecycle"),
        (lambda cfg: setattr(cfg.trainer, "max_epochs", 49), "exactly 50"),
        (lambda cfg: setattr(cfg.trainer, "limit_val_batches", 1), "validation/sanity"),
        (lambda cfg: setattr(cfg.data, "fixed_epochs", 49), "fixed_epochs"),
        (lambda cfg: setattr(cfg.phase2, "outer_date", 19250113), "outer_date"),
        (lambda cfg: cfg.callbacks.update({"early_stopping": {}}), "unexpected/early-stopping"),
        (lambda cfg: cfg.callbacks["fixed_epoch50"].update({"every_n_epochs": 1}), "callback every_n_epochs"),
        (lambda cfg: cfg.callbacks["fixed_epoch50"].update({"dirpath": "other"}), "callback path"),
    ],
)
def test_resolved_config_lifecycle_gate_rejects_drift(tmp_path: Path, mutate, match: str) -> None:
    run_dir = tmp_path / "run"
    config = _resolved_config(run_dir)
    mutate(config)
    with pytest.raises(ValueError, match=match):
        hc0._validate_resolved_hc0_config(config, date=DATE, run_dir=run_dir)
