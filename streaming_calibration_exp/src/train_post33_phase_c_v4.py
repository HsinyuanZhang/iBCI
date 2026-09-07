"""Deterministic Phase-C wrapper around the historical streaming trainer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Optional

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for bootstrap_path in (PROJECT_ROOT, WORKSPACE):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import src.train as legacy
from src.utils import extras, get_metric_value
from src.metrics.run_artifacts import METRICS_PER_SESSION_FIELDS


from sua_exploration.mc_maze.m2_native_post33_authorization_v4 import (
    require_cell_execution_capability_from_environment,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey,
    PROTOCOL_ID,
    cell_paths,
    require_canonical_directory,
    require_canonical_regular_file,
    require_phase_c_training_wrapper_pre_hydra_gate,
    require_same_root_paired_spint_teacher,
    validate_phase_c_training_plan,
    write_bytes_exclusive,
)


def _require_phase_c_hydra_runtime(run: Path) -> None:
    if not HydraConfig.initialized():
        raise RuntimeError("Phase-C training wrapper requires initialized Hydra runtime")
    hydra_cfg = HydraConfig.get()
    if hydra_cfg.output_subdir is not None:
        raise ValueError("Phase-C Hydra output_subdir must be null")
    for label in ("hydra_logging", "job_logging"):
        logging_cfg = hydra_cfg.get(label)
        if logging_cfg is None or logging_cfg.get("disable_existing_loggers") is not True:
            raise ValueError(f"Phase-C Hydra {label} must be disabled")
    runtime_output = Path(str(hydra_cfg.runtime.output_dir))
    if runtime_output.is_symlink() or str(runtime_output) != str(run):
        raise ValueError("Phase-C Hydra runtime output directory substitution")
    if require_canonical_directory(runtime_output) != run:
        raise ValueError("Phase-C Hydra runtime directory is not the canonical run")


def _write_resolved_exclusive(cfg: DictConfig) -> None:
    owner_path = Path(str(cfg.cell_paths.owner)).resolve(strict=True)
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    expected = (
        owner.get("schema"), owner.get("phase_id"), owner.get("arm"),
        owner.get("fold"), owner.get("seed"), owner.get("owner_token"),
    )
    observed = (
        "m2_post33_phase_c_cell_ownership_v4", "PHASE_C_V4", "t4",
        int(cfg.data.loso_fold), int(cfg.seed), str(cfg.cell_owner_token),
    )
    if expected != observed:
        raise PermissionError("streaming Phase-C trainer ownership mismatch")
    cell = owner_path.parent.parent.resolve()
    key = CellKey(PROTOCOL_ID, "t4", int(cfg.data.loso_fold), int(cfg.seed))
    root = cell.parents[5]
    if cell_paths(root, key)["cell_dir"].resolve() != cell:
        raise PermissionError("streaming Phase-C trainer cell root derivation mismatch")
    authorization = require_cell_execution_capability_from_environment(root=root, key=key)
    require_same_root_paired_spint_teacher(
        root,
        key,
        Path(str(cfg.model.paired_spint_completion_receipt)),
    )
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise ValueError("T4 Phase-C resolved config is not a mapping")
    validate_phase_c_training_plan(
        resolved,
        root=root,
        key=key,
        authorized_cost_supplement=authorization["_validated_cost_supplement_path"],
        project_root=PROJECT_ROOT,
    )
    run = require_canonical_directory(cell / "run", within=cell)
    _require_phase_c_hydra_runtime(run)
    output = Path(str(cfg.cell_paths.resolved_config)).resolve()
    if output != run / "resolved_config.yaml":
        raise ValueError("resolved config path is outside the canonical run directory")
    secondary_root = Path(str(cfg.paths.artifact_dir)).resolve()
    if secondary_root != run / "secondary_artifacts":
        raise ValueError("secondary artifact root is not deterministic/canonical")
    data = OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        cursor = 0
        while cursor < len(data):
            cursor += os.write(fd, data[cursor:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _canonical_run_id(cfg: DictConfig) -> str:
    if str(cfg.phase_id) != "PHASE_C_V4" or str(cfg.arm) != "t4":
        raise ValueError("deterministic run id is reserved for Phase-C T4")
    return "canonical"


def _ensure_t4_secondary_metrics_per_session(cfg: DictConfig) -> None:
    """Make the legacy secondary export surface fixed even for fit-only runs.

    ``src.train._export_run_metrics`` normally emits this table alongside the
    summary when per-session rows exist.  A Phase-C fit intentionally performs
    no outer test, so some Lightning versions leave that table empty.  The
    closed run-tree contract nevertheless reserves one deterministic CSV (with
    its canonical header) rather than conditionally changing the artifact set.
    """
    secondary = require_canonical_directory(
        Path(str(cfg.cell_paths.secondary_artifacts))
    )
    output = secondary / "metrics_per_session.csv"
    if output.exists():
        require_canonical_regular_file(output, within=secondary)
        return
    write_bytes_exclusive(output, (",".join(METRICS_PER_SESSION_FIELDS) + "\n").encode("utf-8"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    _write_resolved_exclusive(cfg)
    legacy.make_run_id = _canonical_run_id
    extras(cfg)
    model = None
    active_model = None
    instantiate = legacy.hydra.utils.instantiate

    def _instantiate_with_phase_c_cleanup(*args, **kwargs):
        """Capture the one Phase-C model before historical ``train`` can fail.

        The frozen legacy trainer instantiates the model before ``Trainer.fit``
        and later runs its profile/export work after fit.  Keeping this narrowly
        scoped interposition in the Phase-C wrapper lets its outer ``finally``
        release a retained teacher snapshot after *either* a normal return or an
        exception, without modifying the frozen legacy source map.
        """
        nonlocal active_model
        instance = instantiate(*args, **kwargs)
        if callable(getattr(instance, "finalize_phase_c_teacher_snapshot", None)):
            if active_model is not None and active_model is not instance:
                raise RuntimeError("Phase-C wrapper instantiated multiple teacher snapshot models")
            active_model = instance
        return instance

    legacy.hydra.utils.instantiate = _instantiate_with_phase_c_cleanup
    try:
        metric_dict, _ = legacy.train(cfg)
        model = active_model
        _ensure_t4_secondary_metrics_per_session(cfg)
        return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))
    finally:
        legacy.hydra.utils.instantiate = instantiate
        model_to_finalize = model if model is not None else active_model
        finalize_teacher_snapshot = getattr(model_to_finalize, "finalize_phase_c_teacher_snapshot", None)
        if callable(finalize_teacher_snapshot):
            finalize_teacher_snapshot()


if __name__ == "__main__":
    require_phase_c_training_wrapper_pre_hydra_gate(sys.argv[1:], arm="t4")
    main()
