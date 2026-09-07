#!/usr/bin/env python3
"""Forward-score B1's retained logical epochs 5--12 and write one receipt.

This is the only intended way to convert a completed B1 training directory to
a factorial cell score.  It does no training and uses no target-session
backpropagation.  Unlike ``src/train.py test=true``, it does not resolve a
best checkpoint: every retained epoch 5--12 is independently loaded and
scored on the fixed internal-LOSO validation session.

The executable path is intentionally explicit (`--execute`).  The default
dry-run validates only immutable metadata/paths and opens neither CUDA, NWB,
teacher, nor an epoch checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import PosixPath
from pathlib import Path
from typing import Any, Mapping

PROJECT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.metrics import b1_m2_factorial as core


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise core.B1ContractError(message)


def _spec(stage: str, fold: int, seed: int, carrier: str, loss_mode: str) -> core.CellSpec:
    spec = core.CellSpec(stage, fold, seed, carrier, loss_mode)
    if spec not in core.cells_for_stage(stage):
        raise core.B1ContractError(f"Not a predeclared B1 cell: {spec.key}")
    return spec


def _load_preflight(path: Path) -> tuple[dict[str, Any], str]:
    payload, digest = core.load_verified_immutable_json(path)
    core.validate_preflight_payload(payload)
    bindings = payload["implementation_bindings"]
    core.validate_live_source_bindings(REPO_ROOT, bindings)
    return payload, digest


def _verify_reserved_score_output(post_training_binding: Path, output: Path) -> Path:
    """Require the scorer to consume exactly the O_EXCL-reserved score path.

    The post-training binder is the immutable bridge from an actual training
    artifact to its earlier future launch contract.  Reopen that bridge here,
    before artifact inspection and before any Torch/Lightning/data import, so
    a direct scorer invocation cannot redirect a legitimate cell into a
    different receipt path.  Rechecking output freshness here also prevents a
    stale body or stranded sidecar from being discovered only after a forward
    pass has run.
    """
    binding, _binding_sha = core.load_verified_immutable_json(post_training_binding)
    _need(
        binding.get("receipt_kind") == "b1_m2_factorial_post_training_path_binding"
        and binding.get("screen_id") == core.SCREEN_ID,
        "B1 scorer requires a B1 post-training path binding",
    )
    future_raw = binding.get("future_launch_receipt_path")
    future_sha_expected = binding.get("future_launch_receipt_sha256")
    _need(
        isinstance(future_raw, str) and bool(future_raw)
        and isinstance(future_sha_expected, str) and len(future_sha_expected) == 64,
        "B1 post-training binding lacks future launch path/SHA",
    )
    future_path = Path(future_raw)
    future, future_sha = core.load_verified_immutable_json(future_path)
    _need(
        future_sha == future_sha_expected,
        "B1 scorer future launch SHA drift from post-training binding",
    )
    _need(
        future.get("receipt_kind") == "b1_m2_factorial_future_launch_contract"
        and future.get("screen_id") == core.SCREEN_ID,
        "B1 scorer future launch contract identity drift",
    )
    reserved_raw = future.get("future_score_receipt")
    _need(
        isinstance(reserved_raw, str) and bool(reserved_raw),
        "B1 future launch contract lacks future_score_receipt",
    )
    reserved_unresolved = Path(reserved_raw)
    _need(
        reserved_unresolved.is_absolute(),
        "B1 future launch future_score_receipt must be absolute",
    )
    reserved = reserved_unresolved.resolve()
    actual = output.resolve()
    _need(
        actual == reserved,
        "B1 scorer --out must exactly match the bound future_score_receipt: "
        f"expected {reserved}, got {actual}",
    )
    _need(
        not actual.exists() and not Path(f"{actual}.sha256").exists(),
        f"B1 scorer reserved score output is not fresh: {actual}",
    )
    return reserved


def _expected_checkpoint_paths(epoch_dir: Path) -> dict[int, Path]:
    # Lightning's epoch_004 corresponds to logical epoch 5.  Require one
    # unambiguous checkpoint for each epoch rather than picking by mtime/name.
    paths: dict[int, Path] = {}
    for logical_epoch in core.EPOCH_WINDOW:
        stored = logical_epoch - 1
        matches = sorted(epoch_dir.glob(f"epoch_{stored:03d}.ckpt"))
        _need(len(matches) == 1, f"Need exactly one checkpoint for logical epoch {logical_epoch}: {matches}")
        paths[logical_epoch] = matches[0]
    return paths


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise core.B1ContractError(f"B1 {label} is unreadable: {path}") from exc
    _need(isinstance(value, Mapping), f"B1 {label} is not a JSON object")
    return value


def _verify_post_training_binding(
    path: Path,
    *,
    spec: core.CellSpec,
    preflight_sha: str,
    artifact: Path,
    checkpoint_run_dir: Path,
) -> tuple[Mapping[str, Any], str]:
    payload, digest = core.load_verified_immutable_json(path)
    _need(payload.get("receipt_kind") == "b1_m2_factorial_post_training_path_binding", "B1 post-training binding kind drift")
    _need(payload.get("screen_id") == core.SCREEN_ID and payload.get("cell") == spec.key,
          "B1 launch receipt screen/cell drift")
    _need(payload.get("official_preflight_sha256") == preflight_sha, "B1 launch receipt preflight drift")
    future_path = Path(str(payload.get("future_launch_receipt_path", "")))
    future, future_sha = core.load_verified_immutable_json(future_path)
    _need(future_sha == payload.get("future_launch_receipt_sha256"), "B1 post-training future launch SHA drift")
    _need(future.get("receipt_kind") == "b1_m2_factorial_future_launch_contract" and future.get("cell") == spec.key,
          "B1 post-training future launch contract drift")
    _need(future.get("official_preflight_sha256") == preflight_sha, "B1 future launch preflight drift")
    start_raw = payload.get("execution_start_receipt_path")
    completion_raw = payload.get("execution_completion_receipt_path")
    _need(isinstance(start_raw, str) and bool(start_raw), "B1 binding lacks mandatory execution-start path")
    _need(isinstance(completion_raw, str) and bool(completion_raw), "B1 binding lacks mandatory completion path")
    start_path = Path(start_raw)
    completion_path = Path(completion_raw)
    (
        chained_launch, chained_launch_sha, _start, start_sha,
        _completion, completion_sha,
    ) = core.validate_execution_chain(
        future_launch_receipt=future_path,
        execution_start_receipt=start_path,
        execution_completion_receipt=completion_path,
        require_success=True,
    )
    _need(chained_launch_sha == future_sha and chained_launch.get("cell") == spec.key,
          "B1 independently reopened execution chain launch drift")
    _need(payload.get("execution_start_receipt_sha256") == start_sha,
          "B1 binding execution-start SHA drift")
    _need(payload.get("execution_completion_receipt_sha256") == completion_sha,
          "B1 binding execution-completion SHA drift")
    _need(Path(str(payload.get("artifact", ""))).resolve() == artifact.resolve(), "B1 post-training artifact path drift")
    _need(Path(str(payload.get("checkpoint_run_dir", ""))).resolve() == checkpoint_run_dir.resolve(),
          "B1 post-training checkpoint-run-dir drift")
    _need(Path(str(future.get("future_post_training_binding", ""))).resolve() == path.resolve(),
          "B1 future launch post-training-binding path drift")
    _need(Path(str(future.get("unique_explicit_log_dir", ""))).resolve() == checkpoint_run_dir.resolve(),
          "B1 future launch log-dir contract drift")
    _need(artifact.parent.resolve() == Path(str(future.get("unique_artifact_parent", ""))).resolve(),
          "B1 future launch artifact-parent contract drift")
    context = payload.get("execution_context")
    future_context = future.get("execution_context")
    _need(isinstance(context, Mapping) and isinstance(future_context, Mapping),
          "B1 execution-context binding missing")
    _need(dict(context) == dict(future_context), "B1 post-training execution-context drift")
    for key in ("interpreter", "script", "working_dir"):
        value = context.get(key)
        _need(isinstance(value, str) and Path(value).is_absolute(),
              f"B1 execution context {key} is not absolute")
    _need(Path(str(context["working_dir"])).resolve() == PROJECT.resolve(),
          "B1 scorer only accepts a streaming_calibration_exp-bound working directory")
    _need((artifact / "resolved_config.yaml").is_file(), "B1 artifact lacks resolved config")
    _need(checkpoint_run_dir.is_dir(), "B1 committed checkpoint run directory is absent")
    return payload, digest


def _resolve_bound_path(raw: str, *, working_dir: Path) -> Path:
    """Resolve a saved relative path under the immutable launch working directory."""
    candidate = Path(raw)
    return candidate.resolve() if candidate.is_absolute() else (working_dir / candidate).resolve()


def _safe_torch_load(path: Path, *, map_location: str = "cpu") -> Mapping[str, Any]:
    """Read a trusted local Lightning payload without passing loader flags to a module."""
    import torch
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.listconfig import ListConfig
    from omegaconf.nodes import AnyNode

    # Current local Lightning checkpoints retain harmless Hydra/path metadata.
    # Permit only those concrete metadata types, after inspecting the checkpoint
    # when the installed torch exposes that API.  This mirrors train.py's
    # trusted-local policy while keeping the scorer independent of train.py.
    safe_globals = [
        dict, list, int, Path, PosixPath, defaultdict, Any,
        ContainerMetadata, Metadata, ListConfig, AnyNode,
    ]
    allowed = {f"{item.__module__}.{item.__qualname__}" for item in safe_globals}
    inspect_globals = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if inspect_globals is not None:
        unexpected = set(inspect_globals(path)).difference(allowed)
        _need(not unexpected, f"B1 checkpoint has unexpected serialized globals: {sorted(unexpected)}")
    torch.serialization.add_safe_globals(safe_globals)

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Compatibility with installed torch versions without weights_only.
        payload = torch.load(path, map_location=map_location)
    _need(isinstance(payload, Mapping), f"B1 checkpoint malformed: {path}")
    return payload


def _checkpoint_bundle(
    spec: core.CellSpec, checkpoint_run_dir: Path, preflight: Mapping[str, Any],
    post_training_binding: Mapping[str, Any], *, inspect_hparams: bool,
) -> dict[str, Any]:
    """Bind every scored epoch and, in execute mode, its stored hparams."""
    paths = _expected_checkpoint_paths(checkpoint_run_dir / "checkpoints" / "epoch_ckpts")
    bundle: dict[str, Any] = {}
    expected_teacher = preflight["implementation_bindings"]["teacher"]["checkpoint"]
    for logical_epoch, path in paths.items():
        record: dict[str, Any] = {
            "logical_epoch": logical_epoch,
            "stored_epoch": logical_epoch - 1,
            "path": str(path.resolve()),
            "sha256": core.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        expected_post_hash = (post_training_binding.get("epochs004_011_sha256") or {}).get(str(logical_epoch))
        _need(record["sha256"] == expected_post_hash, f"B1 post-training checkpoint SHA drift at epoch {logical_epoch}")
        if inspect_hparams:
            loaded = _safe_torch_load(path)
            _need(int(loaded.get("epoch", -1)) == logical_epoch - 1, f"B1 checkpoint epoch metadata drift at {logical_epoch}")
            hp = loaded.get("hyper_parameters")
            _need(isinstance(hp, Mapping), f"B1 checkpoint hparams missing at logical epoch {logical_epoch}")
            # These fields are model-side facts; fold/seed/normalizer are
            # separately bound by the source science config + split manifest.
            for key, expected in {
                "variant": "B3S", "side_dim": 4, "freeze_decoder": True,
                "loss_mode": spec.loss_mode,
            }.items():
                _need(hp.get(key) == expected, f"B1 checkpoint {logical_epoch} hparams.{key} drift")
            working_dir = Path(str(post_training_binding["execution_context"]["working_dir"])).resolve()
            _need(
                _resolve_bound_path(str(hp.get("teacher_ckpt_path", "")), working_dir=working_dir)
                == (REPO_ROOT / expected_teacher["path"]).resolve(),
                  f"B1 checkpoint {logical_epoch} teacher path drift")
            _need(float(hp.get("lambda_y")) == 1.0, f"B1 checkpoint {logical_epoch} lambda_y drift")
            _need(float(hp.get("lambda_E")) == (0.0 if spec.loss_mode == "task_plus_y" else 0.1),
                  f"B1 checkpoint {logical_epoch} lambda_E drift")
            record["hparams_verified"] = {
                "variant": hp["variant"], "side_dim": hp["side_dim"], "freeze_decoder": hp["freeze_decoder"],
                "loss_mode": hp["loss_mode"], "lambda_y": hp["lambda_y"], "lambda_E": hp["lambda_E"],
                "teacher_ckpt_path": hp["teacher_ckpt_path"],
            }
        else:
            record["hparams_verified"] = "DEFERRED_TO_EXECUTE_NO_CHECKPOINT_DESERIALIZATION"
        bundle[str(logical_epoch)] = record
    return bundle


def _instantiate_and_strict_load_checkpoint(checkpoint: Path, *, working_dir: Path):
    """Build the streaming module and strict-load it after explicit setup.

    This deliberately avoids Lightning's ``load_from_checkpoint`` shortcut:
    B1 needs the real teacher/student construction path to execute before
    state restoration.  ``weights_only`` is a torch-load option, never a
    module-constructor keyword.
    """
    from src.models.falcon_module import FalconLitModule
    from src.models.streaming_calibration_module import StreamingCalibrationLitModule

    payload = _safe_torch_load(checkpoint)
    hparams = payload.get("hyper_parameters")
    state = payload.get("state_dict")
    _need(isinstance(hparams, Mapping), "B1 checkpoint hyper_parameters missing")
    _need(isinstance(state, Mapping), "B1 checkpoint state_dict missing")
    kwargs = dict(hparams)
    raw_teacher = kwargs.get("teacher_ckpt_path")
    _need(isinstance(raw_teacher, str) and raw_teacher, "B1 checkpoint teacher_ckpt_path missing")
    kwargs["teacher_ckpt_path"] = str(_resolve_bound_path(raw_teacher, working_dir=working_dir))
    kwargs.pop("weights_only", None)
    model = StreamingCalibrationLitModule(**kwargs)
    # The shared module's teacher setup intentionally uses Lightning's normal
    # loader.  Scope a CPU map-location wrapper here so a GPU-saved teacher
    # cannot initialize CUDA during an audit/scoring process.  This does not
    # alter shared runtime source or the checkpoint; it only constrains this
    # scorer's local read path and is restored immediately.
    original_loader = FalconLitModule.load_from_checkpoint
    def _cpu_teacher_loader(cls, *args: Any, **loader_kwargs: Any):
        if "map_location" in loader_kwargs and loader_kwargs["map_location"] != "cpu":
            raise core.B1ContractError("B1 scorer refuses a non-CPU teacher map location")
        loader_kwargs["map_location"] = "cpu"
        return original_loader(*args, **loader_kwargs)
    FalconLitModule.load_from_checkpoint = classmethod(_cpu_teacher_loader)
    try:
        model.setup("test")
    finally:
        FalconLitModule.load_from_checkpoint = original_loader
    incompatible = model.load_state_dict(dict(state), strict=True)
    _need(not incompatible.missing_keys and not incompatible.unexpected_keys,
          "B1 explicit strict state load reported incompatible keys")
    return model


def _audit_artifact(
    spec: core.CellSpec,
    preflight: Mapping[str, Any],
    preflight_sha: str,
    artifact: Path,
    checkpoint_run_dir: Path,
    post_training_binding: Path,
    *,
    inspect_hparams: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove an artifact is exactly the declared scientific cell, fail closed."""
    for name in ("resolved_config.yaml", "source_manifest.json", "split_manifest.json", "teacher_metadata.json"):
        _need((artifact / name).is_file(), f"B1 artifact missing {name}")
    binding_payload, binding_digest = _verify_post_training_binding(
        post_training_binding, spec=spec, preflight_sha=preflight_sha, artifact=artifact,
        checkpoint_run_dir=checkpoint_run_dir,
    )
    cfg = _verify_source_artifact_config(spec, preflight, artifact)
    declared = core.expected_preflight_cell(preflight, spec)
    science = core.science_config_projection(cfg)
    science_sha = core.sha256_payload(science)
    _need(science_sha == declared["science_config_sha256"], "B1 full science config differs from preflight cell")
    source = _read_json(artifact / "source_manifest.json", label="source manifest")
    source_subset_audit = core.validate_exact_artifact_source_manifest(
        source, preflight["implementation_bindings"]
    )
    split = _read_json(artifact / "split_manifest.json", label="split manifest")
    _need(split.get("validation_protocol") == "loso" and split.get("fold_id") == spec.fold,
          "B1 split LOSO/fold provenance drift")
    _need(split.get("heldout_evaluated_in_fit") is False and split.get("heldout_evaluated_in_test") is False,
          "B1 split opened external held-out data")
    train_sessions = split.get("train_sessions")
    validation_sessions = split.get("validation_sessions")
    _need(isinstance(train_sessions, list) and len(train_sessions) == 6, "B1 split must have exactly 6 source sessions")
    _need(isinstance(validation_sessions, list) and len(validation_sessions) == 1, "B1 split must have one LOSO validation session")
    _need(not set(train_sessions).intersection(validation_sessions), "B1 source/validation sessions overlap")
    _need(split.get("query_start_trial") == 0 and split.get("heldin_query_start_trial") == 0
          and split.get("heldin_query_end_trial") is None, "B1 query-window provenance drift")
    normalizer = split.get("native_t4_normalization")
    _need(isinstance(normalizer, Mapping), "B1 T4 normalizer binding absent")
    _need(normalizer.get("feature_group") == "t4" and list(normalizer.get("train_sessions") or []) == train_sessions,
          "B1 T4 normalizer source-session provenance drift")
    _need(len(normalizer.get("mean") or []) == 4 and len(normalizer.get("std") or []) == 4
          and isinstance(normalizer.get("sha256"), str), "B1 T4 normalizer dimensions/digest drift")
    teacher = _read_json(artifact / "teacher_metadata.json", label="teacher metadata")
    teacher_binding = preflight["implementation_bindings"]["teacher"]["checkpoint"]
    _need(teacher.get("teacher_checkpoint_path") == str((REPO_ROOT / teacher_binding["path"]).resolve()),
          "B1 teacher path drift")
    _need(teacher.get("teacher_checkpoint_sha256") == teacher_binding["sha256"], "B1 teacher checkpoint SHA drift")
    bundle = _checkpoint_bundle(spec, checkpoint_run_dir, preflight, binding_payload, inspect_hparams=inspect_hparams)
    artifact_binding = {
        "resolved_config_sha256": core.sha256_file(artifact / "resolved_config.yaml"),
        "science_config_sha256": science_sha,
        "source_manifest_sha256": core.sha256_file(artifact / "source_manifest.json"),
        "source_manifest_relevant_subset_audit": source_subset_audit,
        "split_manifest_sha256": core.sha256_file(artifact / "split_manifest.json"),
        "teacher_metadata_sha256": core.sha256_file(artifact / "teacher_metadata.json"),
        "post_training_path_binding_sha256": binding_digest,
        "execution_context": dict(binding_payload["execution_context"]),
        "checkpoint_run_dir": str(checkpoint_run_dir.resolve()),
        "split_semantics": {
            "validation_protocol": "loso", "fold_id": spec.fold, "train_sessions": train_sessions,
            "validation_sessions": validation_sessions, "heldout_evaluated_in_fit": False,
            "heldout_evaluated_in_test": False, "query_start_trial": 0, "heldin_query_start_trial": 0,
            "heldin_query_end_trial": None,
        },
        "normalizer_binding": dict(normalizer),
    }
    return artifact_binding, bundle


def _dry_run_payload(
    spec: core.CellSpec, preflight_path: Path, preflight_sha: str, preflight: Mapping[str, Any],
    artifact: Path, checkpoint_run_dir: Path, post_training_binding: Path,
) -> dict[str, Any]:
    artifact_binding, bundle = _audit_artifact(
        spec, preflight, preflight_sha, artifact, checkpoint_run_dir, post_training_binding, inspect_hparams=False,
    )
    return {
        "screen_id": core.SCREEN_ID,
        "cell": spec.key,
        "gpu_or_cpu_forward_started": False,
        "formal_test_or_external_heldout_opened": False,
        "preflight_path": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "source_artifact": str(artifact.resolve()),
        "artifact_binding": artifact_binding,
        "logical_epoch_checkpoint_bundle": bundle,
        "fixed_epoch_rule": "logical 5--12 only; no best-validation checkpoint selection",
    }


def _verify_source_artifact_config(spec: core.CellSpec, preflight: Mapping[str, Any], artifact: Path) -> Mapping[str, Any]:
    """Reject an epoch directory whose recorded factor levels are not B1's."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(artifact / "resolved_config.yaml"), resolve=False)
    _need(isinstance(cfg, Mapping), "B1 artifact resolved config is malformed")
    data = cfg.get("data")
    model = cfg.get("model")
    _need(isinstance(data, Mapping) and isinstance(model, Mapping), "B1 artifact lacks data/model config")
    for key, expected in {
        "task": "m2", "validation_protocol": "loso", "loso_fold": spec.fold,
        "calibration_n_trials": 33, "random_calibration": False,
        "include_heldout_in_fit": False, "include_heldout_in_test": False,
        "side_feature_group": "t4",
    }.items():
        _need(data.get(key) == expected, f"B1 source artifact data.{key} drift")
    expected_data_target = (
        "src.data.falcon_datamodule.FalconDataModule" if spec.carrier == "t4"
        else "src.data.b1_m2_matched_z4_datamodule.B1M2MatchedZ4DataModule"
    )
    _need(data.get("_target_") == expected_data_target, "B1 source artifact carrier path drift")
    _need(model.get("loss_mode") == spec.loss_mode and model.get("variant") == "B3S", "B1 source artifact loss/variant drift")
    expected_lambda_e = 0.0 if spec.loss_mode == "task_plus_y" else 0.1
    _need(float(model.get("lambda_y")) == 1.0 and float(model.get("lambda_E")) == expected_lambda_e,
          "B1 source artifact loss-weight drift")
    _need(int(cfg.get("trainer", {}).get("max_epochs", -1)) == 12 and cfg.get("no_early_stopping") is True,
          "B1 source artifact epoch policy drift")
    declared = core.expected_preflight_cell(preflight, spec)
    _need(core.science_config_sha256(cfg) == declared["science_config_sha256"],
          "B1 artifact full science configuration drift")
    return cfg


def _execute(
    spec: core.CellSpec,
    preflight: Mapping[str, Any],
    preflight_sha: str,
    artifact: Path,
    checkpoint_run_dir: Path,
    post_training_binding: Path,
    device: str,
) -> dict[str, Any]:
    # All heavyweight imports are confined to execute mode so a dry run remains
    # a metadata-only audit.
    import lightning.pytorch as pl
    import torch

    artifact_binding, checkpoint_bundle = _audit_artifact(
        spec, preflight, preflight_sha, artifact, checkpoint_run_dir, post_training_binding, inspect_hparams=True,
    )
    # Reuse the recorded *data* configuration from source training rather than
    # compose a second potentially drifting Hydra runtime config.
    from omegaconf import OmegaConf
    import hydra
    saved_cfg = OmegaConf.load(artifact / "resolved_config.yaml")
    datamodule = hydra.utils.instantiate(saved_cfg.data)
    # `setup('test')` has no external held-out files because both matching
    # configs hard-code include_heldout_in_test=false.
    datamodule.setup("test")
    expected_session = list(datamodule.get_split_manifest().get("validation_sessions") or [])
    _need(len(expected_session) == 1, "B1 exact one-session LOSO evaluator required")
    checkpoints = _expected_checkpoint_paths(checkpoint_run_dir / "checkpoints" / "epoch_ckpts")
    working_dir = Path(str(artifact_binding["execution_context"]["working_dir"])).resolve()
    accelerator = "gpu" if device.startswith("cuda") else "cpu"
    device_count = 1
    values: dict[str, dict[str, float]] = {}
    for logical_epoch, checkpoint in checkpoints.items():
        model = _instantiate_and_strict_load_checkpoint(checkpoint, working_dir=working_dir)
        trainer = pl.Trainer(
            accelerator=accelerator, devices=device_count, logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
        )
        trainer.test(model=model, datamodule=datamodule, verbose=False)
        metric_key = f"test_heldin_{expected_session[0]}/r2"
        value = trainer.callback_metrics.get(metric_key)
        if value is None:
            raise core.B1ContractError(f"B1 scorer missing R2 metric {metric_key} for epoch {logical_epoch}")
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        values[str(logical_epoch)] = {expected_session[0]: float(value)}
    return core.make_cell_score_receipt(
        spec=spec, preflight=preflight, run_artifact=artifact, per_epoch_session_r2=values,
        artifact_binding=artifact_binding, checkpoint_bundle=checkpoint_bundle,
        official_preflight_sha256=preflight_sha,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("P", "F"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--carrier", choices=core.CARRIERS, required=True)
    parser.add_argument("--loss-mode", choices=core.LOSS_MODES, required=True)
    parser.add_argument("--official-preflight", type=Path, required=True)
    parser.add_argument("--checkpoint-run-dir", type=Path, required=True)
    parser.add_argument("--post-training-binding", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    spec = _spec(args.stage, args.fold, args.seed, args.carrier, args.loss_mode)
    preflight, preflight_sha = _load_preflight(args.official_preflight)
    # This must precede the artifact audit below: the audit is deliberately
    # metadata-only in dry mode, but execute mode subsequently imports
    # Torch/Lightning and opens the B1 development data.
    _verify_reserved_score_output(args.post_training_binding, args.out)
    dry = _dry_run_payload(
        spec, args.official_preflight, preflight_sha, preflight, args.artifact,
        args.checkpoint_run_dir, args.post_training_binding,
    )
    if not args.execute:
        print(json.dumps(dry, indent=2))
        return
    if args.out.exists() or Path(f"{args.out}.sha256").exists():
        raise SystemExit(f"Refusing existing B1 cell score receipt: {args.out}")
    receipt = _execute(
        spec, preflight, preflight_sha, args.artifact, args.checkpoint_run_dir,
        args.post_training_binding, args.device,
    )
    receipt["official_preflight_sha256"] = preflight_sha
    digest = core.write_immutable_json(args.out, receipt)
    print(json.dumps({"receipt": str(args.out.resolve()), "sha256": digest, "cell": spec.key}, indent=2))


if __name__ == "__main__":
    main()
