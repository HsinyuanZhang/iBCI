"""Owned deterministic Phase-C wrapper around the historical SPINT trainer."""
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


from sua_exploration.mc_maze.m2_native_post33_authorization_v5_r9 import (
    require_cell_execution_capability_from_environment,
    require_r9_training_wrapper_pre_hydra_gate,
    verify_r9_training_wrapper_capability_only,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey,
    PROTOCOL_ID,
    cell_paths,
    require_canonical_directory,
    validate_phase_c_training_plan,
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
        "m2_post33_phase_c_cell_ownership_v4", "PHASE_C_V4", "spint",
        int(cfg.data.loso_fold), int(cfg.seed), str(cfg.cell_owner_token),
    )
    if expected != observed:
        raise PermissionError("SPINT Phase-C trainer ownership mismatch")
    cell = owner_path.parent.parent.resolve()
    key = CellKey(PROTOCOL_ID, "spint", int(cfg.data.loso_fold), int(cfg.seed))
    root = cell.parents[5]
    if cell_paths(root, key)["cell_dir"].resolve() != cell:
        raise PermissionError("SPINT Phase-C trainer cell root derivation mismatch")
    authorization = require_cell_execution_capability_from_environment(root=root, key=key)
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise ValueError("SPINT Phase-C resolved config is not a mapping")
    validate_phase_c_training_plan(
        resolved,
        root=root,
        key=key,
        authorized_cost_supplement=authorization["_validated_cost_supplement_path"],
        project_root=PROJECT_ROOT,
    )
    output = Path(str(cfg.cell_paths.resolved_config)).resolve()
    run = require_canonical_directory(cell / "run", within=cell)
    _require_phase_c_hydra_runtime(run)
    if output != run / "resolved_config.yaml":
        raise ValueError("resolved config path is outside the canonical run directory")
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    _write_resolved_exclusive(cfg)
    extras(cfg)
    metric_dict, _ = legacy.train(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    if verify_r9_training_wrapper_capability_only(sys.argv[1:], arm="spint"):
        raise SystemExit(0)
    require_r9_training_wrapper_pre_hydra_gate(sys.argv[1:], arm="spint")
    main()
