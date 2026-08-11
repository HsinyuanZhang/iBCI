"""Training entry point for streaming calibration experiments."""
import json
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.baseline import validate_baseline_prerequisites
from src.metrics.run_artifacts import (
    METRICS_PER_SESSION_FIELDS,
    METRICS_SUMMARY_FIELDS,
    assert_run_dir_is_fresh,
    build_test_metric_rows,
    ensure_run_dir,
    export_teacher_metadata,
    load_baseline_session_r2,
    make_run_id,
    parse_test_metrics,
    write_baseline_reference,
    write_checkpoint_manifest,
    write_environment,
    write_git_state,
    write_hardware_cost,
    write_metrics_table,
    write_resolved_config,
    write_run_metadata,
    write_source_manifest,
    write_split_manifest,
)
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _allow_known_checkpoint_globals(ckpt_path: str) -> None:
    """Allow only the known standard metadata types in trusted local Lightning checkpoints."""
    safe_globals = [dict, list, int, Path, ContainerMetadata, Metadata, ListConfig, AnyNode, Any, defaultdict]
    # Serialized Linux paths use the concrete PosixPath class rather than pathlib.Path.
    from pathlib import PosixPath

    safe_globals.append(PosixPath)
    allowed = {f"{item.__module__}.{item.__qualname__}" for item in safe_globals}
    inspect_globals = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if inspect_globals is not None:
        unsafe = set(inspect_globals(ckpt_path))
        unexpected = unsafe.difference(allowed)
        if unexpected:
            raise RuntimeError(f"Refusing checkpoint with unexpected globals: {sorted(unexpected)}")
    torch.serialization.add_safe_globals(safe_globals)


def _encoder_cost_profile(model: LightningModule, cfg: DictConfig):
    if not hasattr(model, "encoder_cost_profile"):
        return None
    return model.encoder_cost_profile(
        num_neurons=96,
        trial_length=cfg.data.max_trial_length,
        num_trials=cfg.data.calibration_n_trials,
    )


def _write_hardware_cost_artifact(model: LightningModule, artifact_root: Path, cfg: DictConfig) -> None:
    profile = _encoder_cost_profile(model, cfg)
    if profile is None:
        return
    write_hardware_cost(
        artifact_root,
        profile,
        extra={
            "variant": cfg.model.variant,
            "seed": cfg.seed,
            "note": "B4-B6 MAC counts valid bins only; cubic-interpolation flag reflects encoder capability.",
        },
    )


def _drop_early_stopping(callbacks: List[Callback], *, enabled: bool) -> List[Callback]:
    """M2 fixed-epoch-budget mode: drop EarlyStopping callbacks when requested.

    Does not touch the callbacks config or delete the early-stopping callback
    definition -- this only filters the already-instantiated callback list for this
    run, so the default (``enabled=False``) path is unchanged. See
    sua_exploration/docs/CURRENT_RESULTS.md section H.2 (unequal max-of-N selection
    bias from variants training different numbers of epochs).
    """
    if not enabled:
        return callbacks
    kept = [callback for callback in callbacks if not isinstance(callback, EarlyStopping)]
    dropped = len(callbacks) - len(kept)
    if dropped:
        log.info(f"no_early_stopping=true: dropped {dropped} EarlyStopping callback(s)")
    return kept


def _best_checkpoint_callback(callbacks: List[Callback]) -> Optional[ModelCheckpoint]:
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint) and getattr(callback, "monitor", None):
            return callback
    return None


def _resolve_test_checkpoint(cfg: DictConfig, trainer: Trainer, callbacks: List[Callback]) -> Optional[str]:
    if cfg.get("ckpt_path") not in (None, "", "null"):
        ckpt_path = str(cfg.ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Configured ckpt_path does not exist: {ckpt_path}")
        return ckpt_path

    best_callback = _best_checkpoint_callback(callbacks)
    if best_callback is not None and best_callback.best_model_path:
        return best_callback.best_model_path

    if trainer.checkpoint_callback is not None and trainer.checkpoint_callback.best_model_path:
        return trainer.checkpoint_callback.best_model_path

    if cfg.get("train"):
        log.warning("Best checkpoint not found after training; testing current weights.")
        return None

    raise ValueError("test=true requires ckpt_path when train=false")


def _selected_checkpoint_score(callbacks: List[Callback], trainer: Trainer) -> Tuple[str, Optional[float]]:
    best_callback = _best_checkpoint_callback(callbacks)
    if best_callback is not None and best_callback.monitor:
        score = best_callback.best_model_score
        if score is not None and hasattr(score, "item"):
            score = float(score.item())
        return str(best_callback.monitor), score
    if trainer.checkpoint_callback is not None and trainer.checkpoint_callback.monitor:
        score = trainer.checkpoint_callback.best_model_score
        if score is not None and hasattr(score, "item"):
            score = float(score.item())
        return str(trainer.checkpoint_callback.monitor), score
    return "val_heldin/r2_mean", None


def _export_run_metrics(
    artifact_root: Path,
    cfg: DictConfig,
    metric_dict: Dict[str, Any],
    profile,
    ckpt_path: Optional[str],
    callbacks: List[Callback],
    trainer: Trainer,
    datamodule: LightningDataModule,
) -> None:
    if profile is None:
        return

    baseline_path = Path(cfg.get("baseline_metrics_path", ""))
    baseline = load_baseline_session_r2(baseline_path if baseline_path.exists() else None)
    split_manifest = datamodule.get_split_manifest() if hasattr(datamodule, "get_split_manifest") else {}
    write_split_manifest(artifact_root, split_manifest)
    if baseline_path.exists() and not (artifact_root / "baseline_reference.csv").exists():
        write_baseline_reference(artifact_root, baseline_path)

    parsed = parse_test_metrics(metric_dict)
    summary_rows, per_session_rows = build_test_metric_rows(
        run_id=artifact_root.name,
        variant=str(cfg.model.variant),
        seed=int(cfg.seed),
        calibration_trials=int(cfg.data.calibration_n_trials),
        parsed=parsed,
        profile=profile,
        baseline=baseline,
        validation_protocol=str(split_manifest.get("validation_protocol", cfg.data.get("validation_protocol", ""))),
        fold_id=split_manifest.get("fold_id", cfg.data.get("loso_fold")),
    )
    write_metrics_table(artifact_root, summary_rows, "metrics_summary.csv", METRICS_SUMMARY_FIELDS)
    write_metrics_table(artifact_root, per_session_rows, "metrics_per_session.csv", METRICS_PER_SESSION_FIELDS)

    checkpoint_manifest: Dict[str, Any] = {}
    if ckpt_path:
        monitor, selected_value = _selected_checkpoint_score(callbacks, trainer)
        write_checkpoint_manifest(
            artifact_root,
            ckpt_path,
            selected_by_metric=monitor,
            selected_metric_value=selected_value,
            copy_checkpoint=True,
        )
        checkpoint_manifest = json.loads((artifact_root / "checkpoint_manifest.json").read_text())
        write_run_metadata(
            artifact_root,
            cfg=cfg,
            run_id=artifact_root.name,
            artifact_root=artifact_root,
            split_manifest=split_manifest,
            checkpoint_manifest=checkpoint_manifest,
            selected_metric=monitor,
            selected_metric_value=selected_value,
        )


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # M1 hard assertion (sua_exploration/docs/CURRENT_RESULTS.md section H.4): the Hydra
    # run directory is named to be unique per (run_id, loso_fold, seed) launch (see
    # configs/hydra/default.yaml), but that is a naming precaution, not a guarantee.
    # Refuse to train into a directory that already holds another run's checkpoints or
    # tfevents rather than silently commingling two runs the way B15P/B3 fold1 did.
    assert_run_dir_is_fresh(Path(cfg.paths.output_dir))

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    run_id = make_run_id(cfg)
    artifact_root = ensure_run_dir(cfg.paths.artifact_dir, run_id)
    write_resolved_config(artifact_root, cfg)
    write_environment(artifact_root)
    write_git_state(artifact_root, Path(cfg.paths.root_dir))
    write_source_manifest(artifact_root, Path(cfg.paths.root_dir))
    if cfg.model.get("teacher_ckpt_path"):
        export_teacher_metadata(artifact_root, cfg.model.teacher_ckpt_path)

    if cfg.get("require_baseline_validation", True) and cfg.get("train", True):
        validate_baseline_prerequisites(cfg)
        write_baseline_reference(artifact_root, Path(cfg.baseline_metrics_path))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    callbacks = _drop_early_stopping(callbacks, enabled=bool(cfg.get("no_early_stopping", False)))
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, use_distributed_sampler=False
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "artifact_dir": artifact_root,
        "run_id": run_id,
    }

    if logger:
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        # The strict M1 source-only decoder must emit its provenance receipt
        # from the exact loader state before any optimizer step.  Data modules
        # have no Lightning ``on_fit_start`` hook, hence this explicit, narrow
        # capability check.  Ordinary data modules are unaffected.
        if hasattr(datamodule, "write_source_only_manifest"):
            datamodule.setup("fit")
            receipt = datamodule.write_source_only_manifest(cfg.paths.output_dir)
            log.info("Wrote source-only decoder manifest: %s", receipt)
        log.info("Starting training!")
        if cfg.get("ckpt_path") not in (None, "", "null"):
            _allow_known_checkpoint_globals(str(cfg.ckpt_path))
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if hasattr(model, "setup"):
        model.setup("fit")
    profile = _encoder_cost_profile(model, cfg)
    _write_hardware_cost_artifact(model, artifact_root, cfg)

    ckpt_path: Optional[str] = None
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = _resolve_test_checkpoint(cfg, trainer, callbacks)
        if ckpt_path:
            _allow_known_checkpoint_globals(ckpt_path)
        test_kwargs = {"model": model, "datamodule": datamodule, "ckpt_path": ckpt_path}
        if "weights_only" in inspect.signature(trainer.test).parameters:
            test_kwargs["weights_only"] = True
        trainer.test(**test_kwargs)

    metric_dict = {**train_metrics, **trainer.callback_metrics}
    _export_run_metrics(artifact_root, cfg, metric_dict, profile, ckpt_path, callbacks, trainer, datamodule)
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
