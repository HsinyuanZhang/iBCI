#!/usr/bin/env python3
"""Prepare—but never execute—the paired H-S/H-C source-training launch receipt.

This receipt-only launcher composes the two fixed Hydra configurations and
binds them to one immutable, passed Phase-2 CPU pair preflight.  It exposes no
``--execute`` option, creates no Trainer, and has no CUDA/target/checkpoint
load path.  A later explicit GPU executor must consume this immutable receipt
instead of independently choosing an outer date, source schedule, or
checkpoint policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import hydra
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_phase2_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, write_immutable_json


LAUNCH_RECEIPT_SCHEMA = "h1_carrierid_date_lodo_phase2_paired_source_launch_receipt_v1"
LAUNCH_RECEIPT_STATUS = "PASS_PAIRED_SOURCE_TRAINING_PREPARED_NOT_LAUNCHED"
LEGACY_OUTER_DATE = "19250108"
LEGACY_ARM_CONFIGS = {
    "H-S": "h1_carrierid_date_lodo_hs_19250108",
    "H-C": "h1_carrierid_date_lodo_hc_19250108",
}
GENERIC_ARM_CONFIGS = {
    "H-S": "h1_carrierid_date_lodo_hs_phase2",
    "H-C": "h1_carrierid_date_lodo_hc_phase2",
}
MODEL_TARGETS = {
    "H-S": "src.models.components.spint.SpintModel",
    "H-C": "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _immutable(path: Path) -> bool:
    return path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _read_pair_preflight(path: Path) -> tuple[Path, dict[str, Any], str]:
    candidate = path.resolve()
    _need(_immutable(candidate), f"pair preflight must be immutable mode 0444: {candidate}")
    try:
        receipt = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid pair preflight JSON: {candidate}") from error
    _need(receipt.get("schema") == PREFLIGHT_SCHEMA and receipt.get("status") == PREFLIGHT_STATUS,
          "pair preflight does not prove a passed source-only Phase-2 pair")
    scope = receipt.get("scope")
    _need(isinstance(scope, Mapping) and all(scope.get(field) in (False, 0) for field in (
        "target_recordings_opened", "target_bytes_read", "trainer_constructed_or_launched",
        "checkpoint_created_or_loaded", "cuda_constructed_or_launched",
    )), "pair preflight violates source-only CPU boundary")
    contract = receipt.get("phase2_training_contract")
    _need(isinstance(contract, Mapping), "pair preflight lacks Phase-2 training contract")
    _need(contract.get("arms") == ["H-S", "H-C"] and contract.get("fresh_seed") == 42,
          "pair preflight arm/seed drift")
    _need(contract.get("fixed_terminal_epoch_zero_based") == 49 and contract.get("epochs") == 50,
          "pair preflight terminal epoch contract drift")
    _need(contract.get("checkpoint_warm_start_forbidden") is True and contract.get("target_evaluator_status") == "NOT_IMPLEMENTED",
          "pair preflight permits a forbidden warm-start or target route")
    source = receipt.get("source_binding")
    _need(isinstance(source, Mapping) and source.get("target_recordings_opened") == 0 and source.get("target_bytes_read") == 0,
          "pair preflight source binding violates target boundary")
    _need(canonical_sha256(source) == receipt.get("source_binding_sha256"), "pair preflight source-binding SHA drift")
    outer_date = str(receipt.get("outer_date", ""))
    _need(outer_date in CONFIRMATORY_DATES, "pair preflight outer date is not a confirmatory date")
    _need(source.get("outer_date") == outer_date,
          "pair preflight and source binding disagree on the immutable outer date")
    return candidate, receipt, sha256_file(candidate)


def _experiment_for_arm(*, arm: str, outer_date: str) -> str:
    """Keep the active 19250108 config byte-identical; generalize only new dates."""

    _need(arm in MODEL_TARGETS, f"unsupported arm: {arm}")
    _need(str(outer_date) in CONFIRMATORY_DATES, "outer date is not confirmatory")
    return (LEGACY_ARM_CONFIGS if str(outer_date) == LEGACY_OUTER_DATE else GENERIC_ARM_CONFIGS)[arm]


def _compose_experiment(experiment: str, *, overrides: tuple[str, ...] = ()) -> Any:
    with hydra.initialize_config_dir(version_base=None, config_dir=str((ROOT / "configs").resolve())):
        return hydra.compose(config_name="train", overrides=[f"experiment={experiment}", *overrides])


def _compose_arm_config(
    *, arm: str, outer_date: str, pair_preflight: Path, phase1_preflight: Path,
) -> tuple[str, Any, tuple[str, ...]]:
    """Compose one receipt-bound arm without constructing data, CUDA, or a Trainer."""

    experiment = _experiment_for_arm(arm=arm, outer_date=outer_date)
    # The legacy 19250108 configs are part of an active run and consequently
    # retain their original literal/interpolated bindings.  Later dates use
    # generic templates whose only date-bearing settings are supplied from the
    # immutable pair preflight passed to this launcher.
    overrides: tuple[str, ...] = ()
    if str(outer_date) != LEGACY_OUTER_DATE:
        overrides = (
            f"phase2.outer_date={outer_date}",
            f"phase2.pair_preflight_path={pair_preflight}",
            f"phase2.phase1_preflight_path={phase1_preflight}",
        )
    return experiment, _compose_experiment(experiment, overrides=overrides), overrides


def _validate_arm_config(
    cfg: Any, *, arm: str, outer_date: str, pair_preflight: Path,
    phase1_preflight: Path | None = None, experiment: str | None = None, compose_overrides: tuple[str, ...] = (),
) -> dict[str, Any]:
    # A no-run Hydra compose has no runtime ``HydraConfig``; retain the raw
    # interpolation AST rather than resolving `${paths.work_dir}`.  The exact
    # binding is still checked below and resolves only inside a later explicit
    # training invocation.
    raw = OmegaConf.to_container(cfg, resolve=False)
    _need(isinstance(raw, Mapping), f"{arm}: composed Hydra config is malformed")
    phase2, data, model, trainer, callbacks = (
        raw["phase2"], raw["data"], raw["model"], raw["trainer"], raw["callbacks"],
    )
    _need(phase2["arm"] == arm and str(phase2["outer_date"]) == str(outer_date),
          f"{arm}: Hydra config arm/date mismatch")
    legacy_pair = "${paths.work_dir}/pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIR_CPU_PREFLIGHT_v2.json"
    pair_binding = phase2["pair_preflight_path"]
    _need(
        pair_binding == str(pair_preflight) or
        (str(outer_date) == LEGACY_OUTER_DATE and pair_binding == legacy_pair and pair_preflight.name in legacy_pair),
        f"{arm}: Hydra config does not bind this pair preflight",
    )
    _need(phase2["warm_start_forbidden"] is True and phase2["target_evaluator_status"] == "NOT_IMPLEMENTED",
          f"{arm}: config enables forbidden warm-start/target route")
    _need(raw["train"] is True and raw["test"] is False and raw["ckpt_path"] in (None, "", "null"),
          f"{arm}: source config train/test/checkpoint contract drift")
    _need(int(raw["seed"]) == 42 and int(model["fixed_seed"]) == 42 and int(data["seed"]) == 42,
          f"{arm}: fresh seed=42 contract drift")
    _need(model["_target_"] == "src.models.h1_carrierid_date_lodo_phase2_module.H1CarrierIdDateLodoPhase2LitModule",
          f"{arm}: wrong Phase-2 model wrapper")
    _need(model["net"]["_target_"] == MODEL_TARGETS[arm], f"{arm}: wrong paired consumer")
    _need(data["_target_"] == "src.data.h1_carrierid_date_lodo_phase2.H1CarrierIdDateLodoSourceDataModule",
          f"{arm}: wrong Phase-2 source DataModule")
    _need(data["phase1_preflight_path"] == "${phase2.phase1_preflight_path}",
          f"{arm}: data config does not bind Phase-1 preflight")
    _need(
        int(data["calibration_n_trials"]) == 4 and int(data["fixed_epochs"]) == 50 and
        int(data["batch_size"]) == 32 and int(data["window_size"]) == 700 and
        int(data["max_trial_length"]) == 1024,
        f"{arm}: fixed M=4 chronological support/source schedule contract drift",
    )
    if phase1_preflight is not None and str(outer_date) != LEGACY_OUTER_DATE:
        _need(phase2["phase1_preflight_path"] == str(phase1_preflight),
              f"{arm}: generic config does not bind the source Phase-1 preflight")
    _need(
        int(trainer["max_epochs"]) == int(trainer["min_epochs"]) == 50 and
        int(trainer["limit_val_batches"]) == 0 and int(trainer["num_sanity_val_steps"]) == 0 and
        trainer["enable_checkpointing"] is True,
        f"{arm}: trainer e49/no-validation contract drift",
    )
    terminal = callbacks.get("fixed_epoch50")
    _need(terminal is not None and terminal.get("monitor") is None and int(terminal.get("every_n_epochs")) == 50 and
          int(terminal.get("save_top_k")) == -1 and terminal.get("save_last") is False,
          f"{arm}: terminal e49 callback contract drift")
    return {
        "experiment": raw["protocol_id"],
        "experiment_config": experiment or "UNSPECIFIED",
        "compose_overrides": list(compose_overrides),
        "task_name": raw["task_name"],
        "config_sha256": canonical_sha256(raw),
        "consumer_target": model["net"]["_target_"],
        "model_wrapper": model["_target_"],
        "data_target": data["_target_"],
        "fresh_seed": 42,
        "fixed_terminal_epoch_zero_based": 49,
        "epochs": 50,
        "calibration_support_trials": 4,
        "strict_target_query_starts_at_trial": 5,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "checkpoint_warm_start_forbidden": True,
        "planned_command": [
            sys.executable, str(ROOT / "src/train.py"),
            f"experiment={experiment or _experiment_for_arm(arm=arm, outer_date=outer_date)}",
            *compose_overrides,
            "ckpt_path=null", "test=false",
        ],
    }


def run(*, pair_preflight: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite paired launch receipt: {output}")
    preflight_path, receipt, preflight_sha = _read_pair_preflight(pair_preflight)
    outer_date = str(receipt["outer_date"])
    _need(outer_date in CONFIRMATORY_DATES, "pair preflight outer date is not confirmatory")
    source = receipt["source_binding"]
    phase1_preflight = Path(str(source.get("preflight_path", ""))).resolve()
    _need(_immutable(phase1_preflight), "pair preflight source binding lacks an immutable Phase-1 receipt")
    _need(sha256_file(phase1_preflight) == source.get("preflight_sha256"),
          "pair preflight Phase-1 receipt byte identity drift")
    arms: dict[str, dict[str, Any]] = {}
    for arm in MODEL_TARGETS:
        config, composed, overrides = _compose_arm_config(
            arm=arm, outer_date=outer_date, pair_preflight=preflight_path, phase1_preflight=phase1_preflight,
        )
        arms[arm] = _validate_arm_config(
            composed, arm=arm, outer_date=outer_date, pair_preflight=preflight_path,
            phase1_preflight=phase1_preflight, experiment=config, compose_overrides=overrides,
        )
    _need(arms["H-S"]["fresh_seed"] == arms["H-C"]["fresh_seed"] == 42, "pair fresh seed mismatch")
    launch = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "status": LAUNCH_RECEIPT_STATUS,
        "mode": "prepare_only_no_subprocess_no_trainer_no_cuda_no_target_no_checkpoint_load",
        "outer_date": outer_date,
        "pair_preflight_path": str(preflight_path),
        "pair_preflight_sha256": preflight_sha,
        "source_binding_sha256": receipt["source_binding_sha256"],
        "source_binding": source,
        "pair": {
            "arms": arms,
            "same_source_binding_for_h_s_h_c": True,
            "same_phase1_schedule_for_h_s_h_c": True,
            "fresh_seed": 42,
            "terminal_checkpoint": "epoch_049 only after 50 source epochs",
            "warm_start_forbidden": True,
        },
        "scope": {
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
            "trainer_constructed_or_launched": False,
            "cuda_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False,
            "target_evaluator_status": "NOT_IMPLEMENTED",
        },
        "code_sha256": {
            "launcher": sha256_file(Path(__file__).resolve()),
            "data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_phase2.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_phase2_module.py"),
            "hs_experiment": sha256_file(ROOT / "configs/experiment" / f"{arms['H-S']['experiment_config']}.yaml"),
            "hc_experiment": sha256_file(ROOT / "configs/experiment" / f"{arms['H-C']['experiment_config']}.yaml"),
            "terminal_callback": sha256_file(ROOT / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml"),
        },
    }
    write_immutable_json(output, launch)
    return {**launch, "output": str(output.resolve()), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-preflight", type=Path,
        default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIR_CPU_PREFLIGHT_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(pair_preflight=args.pair_preflight, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
