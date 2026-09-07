#!/usr/bin/env python3
"""Exact selected-checkpoint SPINT outer evaluator worker for Phase-C."""
from __future__ import annotations

import argparse
from io import StringIO
import json
import math
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for bootstrap_path in (PROJECT_ROOT, WORKSPACE):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

import hydra
from lightning.pytorch import Trainer
from omegaconf import OmegaConf


from sua_exploration.mc_maze.m2_native_post33_evaluator_v4 import write_endpoint_payload
from sua_exploration.mc_maze.m2_native_post33_deployment_profiler_v4 import DeploymentProfilerV4
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey, PROTOCOL_ID, cell_paths, file_metadata,
    prepare_phase_c_evaluator_handoff, release_selected_checkpoint_snapshot,
    validate_deployment_constants_origin, validate_deployment_constants_snapshot,
    validate_phase_c_training_plan, validate_selected_checkpoint_origin,
    validate_selected_checkpoint_snapshot,
    write_json_exclusive,
)
from sua_exploration.mc_maze.m2_native_post33_authorization_v4 import (
    require_cell_execution_capability,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(7), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--opaque-payload-out", type=Path, required=True)
    parser.add_argument("--deployment-cost-evidence-out", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-signature", type=Path, required=True)
    parser.add_argument("--program-receipt", type=Path, required=True)
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--cost-supplement", type=Path, required=True)
    args = parser.parse_args()
    key = CellKey(PROTOCOL_ID, "spint", args.fold, args.seed)
    paths = cell_paths(args.cell_root, key)
    require_cell_execution_capability(
        root=args.cell_root,
        key=key,
        authorization_path=args.authorization,
        signature_path=args.authorization_signature,
        phase_c_program_receipt_path=args.program_receipt,
        portable_manifest_path=args.portable_manifest,
        shard_manifest_path=args.shard_manifest,
        cost_supplement_path=args.cost_supplement,
    )
    owner = json.loads(paths["owner"].read_text(encoding="utf-8"))
    if owner.get("owner_token") != args.owner_token:
        raise PermissionError("SPINT evaluator worker ownership mismatch")
    if args.opaque_payload_out.resolve() != paths["opaque_payload_run"]:
        raise ValueError("SPINT opaque payload output is outside the canonical run")
    if args.deployment_cost_evidence_out.resolve() != paths["deployment_cost_evidence_run"]:
        raise ValueError("SPINT deployment-cost output is outside the canonical run")
    (
        selected,
        checkpoint_snapshot,
        deployment_constants_snapshot,
        config_path,
        config_bytes,
    ) = prepare_phase_c_evaluator_handoff(
        root=args.cell_root,
        key=key,
        cost_supplement_path=args.cost_supplement,
    )
    try:
        # These two checks are deliberately before parsing/config/model/data
        # construction.  A named temp replacement is rejected; a later swap
        # cannot redirect Trainer because it receives the pinned FD path.
        validate_selected_checkpoint_snapshot(checkpoint_snapshot)
        validate_selected_checkpoint_origin(checkpoint_snapshot)
        validate_deployment_constants_snapshot(deployment_constants_snapshot)
        validate_deployment_constants_origin(deployment_constants_snapshot)
        cfg = OmegaConf.load(StringIO(config_bytes.decode("utf-8")))
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        if not isinstance(resolved, dict):
            raise ValueError("SPINT resolved evaluator config is not a mapping")
        validate_phase_c_training_plan(
            resolved,
            root=args.cell_root,
            key=key,
            authorized_cost_supplement=args.cost_supplement,
            project_root=PROJECT_ROOT,
        )
        validate_selected_checkpoint_snapshot(checkpoint_snapshot)
        validate_selected_checkpoint_origin(checkpoint_snapshot)
        validate_deployment_constants_snapshot(deployment_constants_snapshot)
        validate_deployment_constants_origin(deployment_constants_snapshot)
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.bind_phase_c_deployment_constants(deployment_constants_snapshot)
        model = hydra.utils.instantiate(cfg.model)
        model.bind_phase_c_selected_checkpoint_snapshot(checkpoint_snapshot)
        profiler = DeploymentProfilerV4()
        model.enable_deployment_profiler(profiler)
        validate_selected_checkpoint_snapshot(checkpoint_snapshot)
        trainer = Trainer(
            accelerator="gpu", devices=1, precision="32-true", logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
            deterministic=True, use_distributed_sampler=False,
        )
        trainer.test(
            model=model, datamodule=datamodule,
            ckpt_path=str(checkpoint_snapshot.trainer_checkpoint_path),
            verbose=False,
        )
        validate_selected_checkpoint_snapshot(checkpoint_snapshot)
        if not isinstance(model.deployment_cache_evidence, dict):
            raise RuntimeError("SPINT evaluator produced no cached-deployment evidence")
        # Lightning may restore the module's pre-test training flag after
        # ``Trainer.test`` returns.  The deployment microbenchmark must retain
        # the exact inference semantics used by the test loop (in particular,
        # the historical dynamic-dropout branch must remain disabled).
        model.eval()
        profiler.benchmark_online_b1(
            model.cached_online_forward, datamodule.deployment_neural_window()
        )
        split = datamodule.get_split_manifest()
        write_json_exclusive(
            paths["deployment_cost_evidence_run"],
            profiler.payload(
                arm="spint", fold=args.fold, seed=args.seed,
                outer_session=key.outer_session, resolved_config=file_metadata(config_path),
                cache_evidence=model.deployment_cache_evidence,
                descriptor_fit={
                    "applicable": False, "execution_device": "none", "invocations": 0,
                    "wall_time_ns": 0, "persistent_state_bytes": 0,
                },
                integrity_audit=split["deployment_calibration_integrity_audit"],
            ),
        )
        score = model.outer_test_score_for_payload
        metric = model.outer_test_runtime_metric_evidence
        if not isinstance(score, float) or not math.isfinite(score) or not isinstance(metric, dict):
            raise RuntimeError("SPINT exact-one outer evaluator produced no finite payload score")
        query = datamodule.post33_query_dataset.query_window_audit
        if set(query) != {key.outer_session}:
            raise ValueError("SPINT evaluator query scope is not exact-one outer session")
        scope = {
            "outer_session": key.outer_session,
            "outer_counts": split["outer_counts"],
            "query_window_audit": query[key.outer_session],
            "target_labels_used": False,
            "query_targets_used_for_calibration": False,
            "query_targets_used_for_normalization": False,
            "query_targets_used_for_selection": False,
        }
        validate_selected_checkpoint_origin(checkpoint_snapshot)
        validate_deployment_constants_origin(deployment_constants_snapshot)
        write_endpoint_payload(
            paths["opaque_payload_run"], key=key, score=score,
            metric_total=int(metric["metric_total"]),
            selected_checkpoint=checkpoint_snapshot.canonical_selected_checkpoint,
            resolved_config=config_path,
            execution_capability_evidence=paths["execution_capability_evidence_run"],
            source_and_query_scope=scope,
        )
        validate_selected_checkpoint_origin(checkpoint_snapshot)
        validate_deployment_constants_origin(deployment_constants_snapshot)
    finally:
        release_selected_checkpoint_snapshot(checkpoint_snapshot)


if __name__ == "__main__":
    main()
