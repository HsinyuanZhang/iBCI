#!/usr/bin/env python3
"""Exact selected-checkpoint T4 outer evaluator worker for Phase-C v5.

This is an append-only v4 evaluator copy.  Its only runtime-semantic delta is
the post-``Trainer.test`` cached B=1 deployment preparation: v5 explicitly
co-locates ``model.student`` and the plain cached identity on CUDA, binds
checks to the profiler's actual CUDA input, and revalidates cache bytes after
profiling.  All checkpoint, score, chronology, label, query, and teacher
lifecycle behavior below is intentionally identical to v4.
"""
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


from src.callbacks.decoder_lifecycle_phase_c_v4 import DecoderLifecycleEvalStagesV4
from src.utils.t4_outer_runtime_phase_c_v4 import write_t4_outer_runtime_evidence
from sua_exploration.mc_maze.m2_native_post33_deployment_device_prep_v5 import (
    prepare_cached_online_b1_after_trainer_test_v5,
)
from sua_exploration.mc_maze.m2_native_post33_deployment_profiler_v4 import DeploymentProfilerV4
from sua_exploration.mc_maze.m2_native_post33_evaluator_v4 import write_endpoint_payload
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey, PROTOCOL_ID, cell_paths, file_metadata,
    prepare_phase_c_evaluator_handoff, release_selected_checkpoint_snapshot,
    require_same_root_paired_spint_teacher, validate_deployment_constants_origin,
    validate_deployment_constants_snapshot, validate_phase_c_training_plan,
    validate_selected_checkpoint_origin, validate_selected_checkpoint_snapshot,
    write_json_exclusive,
)
from sua_exploration.mc_maze.m2_native_post33_authorization_v5_r7 import (
    require_cell_execution_capability,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(7), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--opaque-payload-out", type=Path, required=True)
    parser.add_argument("--decoder-evidence-out", type=Path, required=True)
    parser.add_argument("--outer-runtime-evidence-out", type=Path, required=True)
    parser.add_argument("--deployment-cost-evidence-out", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-signature", type=Path, required=True)
    parser.add_argument("--program-receipt", type=Path, required=True)
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--cost-supplement", type=Path, required=True)
    args = parser.parse_args()
    key = CellKey(PROTOCOL_ID, "t4", args.fold, args.seed)
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
        raise PermissionError("T4 evaluator worker ownership mismatch")
    exact_outputs = (
        args.opaque_payload_out.resolve() == paths["opaque_payload_run"],
        args.decoder_evidence_out.resolve() == paths["decoder_lifecycle_evidence_run"],
        args.outer_runtime_evidence_out.resolve() == paths["outer_runtime_evidence_run"],
        args.deployment_cost_evidence_out.resolve() == paths["deployment_cost_evidence_run"],
    )
    if not all(exact_outputs):
        raise ValueError("T4 evaluator output is outside the canonical run")
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
    model = None
    teacher_snapshot_finalized = False
    try:
        validate_selected_checkpoint_snapshot(checkpoint_snapshot)
        validate_selected_checkpoint_origin(checkpoint_snapshot)
        validate_deployment_constants_snapshot(deployment_constants_snapshot)
        validate_deployment_constants_origin(deployment_constants_snapshot)
        cfg = OmegaConf.load(StringIO(config_bytes.decode("utf-8")))
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        if not isinstance(resolved, dict):
            raise ValueError("T4 resolved evaluator config is not a mapping")
        validate_phase_c_training_plan(
            resolved,
            root=args.cell_root,
            key=key,
            authorized_cost_supplement=args.cost_supplement,
            project_root=PROJECT_ROOT,
        )
        require_same_root_paired_spint_teacher(
            args.cell_root,
            key,
            Path(str(cfg.model.paired_spint_completion_receipt)),
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
        lifecycle = DecoderLifecycleEvalStagesV4(
            stage_dir=str(paths["decoder_lifecycle_stages"]),
            evidence_output=str(paths["decoder_lifecycle_evidence_run"]),
            cell_owner_path=str(paths["owner"]), owner_token=args.owner_token,
            fold=args.fold, seed=args.seed,
        )
        validate_selected_checkpoint_snapshot(checkpoint_snapshot)
        trainer = Trainer(
            accelerator="gpu", devices=1, precision="32-true", logger=False,
            callbacks=[lifecycle], enable_checkpointing=False, enable_progress_bar=False,
            deterministic=True, use_distributed_sampler=False,
        )
        trainer.test(
            model=model, datamodule=datamodule,
            ckpt_path=str(checkpoint_snapshot.trainer_checkpoint_path),
            verbose=False,
        )
        finalize_teacher_snapshot = getattr(model, "finalize_phase_c_teacher_snapshot", None)
        if callable(finalize_teacher_snapshot):
            # The model releases its teacher FD in a ``finally`` even when its
            # terminal canonical-origin revalidation fails.  Mark it finalized
            # on both outcomes so the outer exception cleanup does not call a
            # released snapshot a second time and mask that original failure.
            try:
                finalize_teacher_snapshot()
            finally:
                teacher_snapshot_finalized = True
        else:
            teacher_snapshot_finalized = True
        validate_selected_checkpoint_snapshot(checkpoint_snapshot)
        validate_deployment_constants_snapshot(deployment_constants_snapshot)
        score = model.outer_test_score_for_payload
        metric = model.outer_test_runtime_metric_evidence
        if not isinstance(score, float) or not math.isfinite(score) or not isinstance(metric, dict):
            raise RuntimeError("T4 exact-one outer evaluator produced no finite payload score")
        split = datamodule.get_split_manifest()
        if not isinstance(model.deployment_cache_evidence, dict):
            raise RuntimeError("T4 evaluator produced no cached-deployment evidence")
        # Lightning 2.4 teardown moves registered model state to CPU while the
        # plain cached identity remains outside that migration.  v5 restores
        # the deployment student/cache/device/mode boundary before the
        # profiler owns the same CPU->CUDA B=1 transfer as the v4 evaluator.
        neural_cpu = datamodule.deployment_neural_window()
        prepared = prepare_cached_online_b1_after_trainer_test_v5(
            model, arm="t4", neural_cpu=neural_cpu,
        )
        profiler.benchmark_online_b1(
            prepared.bind_checked_forward(model.cached_online_forward), neural_cpu
        )
        prepared.assert_ready()
        write_json_exclusive(
            paths["deployment_cost_evidence_run"],
            profiler.payload(
                arm="t4", fold=args.fold, seed=args.seed,
                outer_session=key.outer_session, resolved_config=file_metadata(config_path),
                cache_evidence=model.deployment_cache_evidence,
                descriptor_fit=split["t4_descriptor_fit_runtime"],
                integrity_audit=split["deployment_calibration_integrity_audit"],
            ),
        )
        write_t4_outer_runtime_evidence(
            paths["outer_runtime_evidence_run"], fold=args.fold, seed=args.seed,
            metric_evidence=metric, split_manifest=split,
        )
        runtime = split["outer_runtime_evidence"]
        scope = {
            "outer_session": key.outer_session,
            "outer_counts": split["outer_counts"],
            "query_window_audit": runtime["query_window_audit"],
            "target_labels_used": True,
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
        try:
            if model is not None and not teacher_snapshot_finalized:
                finalize_teacher_snapshot = getattr(
                    model, "finalize_phase_c_teacher_snapshot", None
                )
                if callable(finalize_teacher_snapshot):
                    finalize_teacher_snapshot()
        finally:
            release_selected_checkpoint_snapshot(checkpoint_snapshot)


if __name__ == "__main__":
    main()
