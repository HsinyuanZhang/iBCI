"""CPU-only source-only C1 teacher compatibility probe.

Establishes whether a CO-native teacher of the existing student architecture
can be constructed and loaded into StreamingCalibrationLitModule, and records
exactly what differs from the current MC-Maze decoder initialization.

This module never opens an NWB, never touches sub-M or sealed test sessions,
never trains, and never uses a GPU.  A matching-architecture synthetic
checkpoint proves the *interface*.  It is not a trained CO-native teacher and
is not an accuracy result.  Teacher/target mismatch remains a hypothesis to
test, not a cause declared by this receipt.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"

import torch

from mc_maze import c1_teacher_domain_ablation as core


def _require_cpu() -> None:
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CUDA_VISIBLE_DEVICES must be empty")
    if torch.cuda.is_available():
        raise core.C1ContractError("C1 compatibility probe refuses a visible CUDA device")


def _tensor_shapes(state: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: list(tensor.shape) for name, tensor in state.items()}


def _tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu").contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def spint_architecture_from_net(net: Any) -> dict[str, Any]:
    return {
        "model_dim": int(net.model_dim),
        "num_covariates": int(net.num_covariates),
        "window_size": int(net.window_size),
        "num_heads": int(net.num_heads),
        "num_layers": int(net.num_layers),
        "num_id_layers": int(net.num_id_layers),
        "use_learnable_id": bool(net.use_learnable_id),
        "learnable_id_type": str(net.learnable_id_type),
        "learnable_rep": bool(net.learnable_rep),
        "dropout_rate": float(net.dropout_rate),
        "dynamic_dropout": bool(net.dynamic_dropout),
        "dynamic_dropout_low": float(net.dynamic_dropout_low),
        "dynamic_dropout_high": float(net.dynamic_dropout_high),
        "tf_drop_rate": float(net.tf_drop_rate),
        "readin_layer_type": str(net.readin_layer_type),
    }


def materialized_identity_in_features(net: Any) -> int:
    weight = net.fc_id_in[0].weight
    core.require(weight.ndim == 2, "teacher fc_id_in[0] is not a materialized Linear")
    return int(weight.shape[1])


def snapshot_teacher_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Deserialize a FalconLitModule teacher on CPU and snapshot the student-facing contract."""
    _require_cpu()
    path = Path(path).expanduser().resolve()
    core.require(path.is_file(), f"teacher checkpoint missing: {path}")
    core.refuse_target_or_sealed_paths([path], label="teacher checkpoint")

    from src.models.falcon_module import FalconLitModule

    module = FalconLitModule.load_from_checkpoint(str(path), weights_only=False, map_location=map_location)
    module.eval()
    net = module.net
    device = next(net.parameters()).device
    core.require(device.type == "cpu", f"teacher loaded onto {device}, not cpu")
    architecture = spint_architecture_from_net(net)
    state = {name: tensor.detach().cpu() for name, tensor in net.state_dict().items()}
    identity_in = materialized_identity_in_features(net)
    fc_in_in = int(net.fc_in[0].weight.shape[1])
    return {
        "path": str(path),
        "sha256": core.sha256_file(path),
        "bytes": path.stat().st_size,
        "lightning_task": str(module.hparams.task),
        "decode_last_timestep_only": bool(module.hparams.decode_last_timestep_only),
        "predict_scaled_behavior": bool(module.hparams.predict_scaled_behavior),
        "behavior_scaling_factor": float(module.hparams.behavior_scaling_factor),
        "architecture": architecture,
        "identity_mlp_in_features": identity_in,
        "fc_in_in_features": fc_in_in,
        "state_dict_keys": sorted(state),
        "state_dict_shapes": _tensor_shapes(state),
        "state_dict_sha256": _tensor_sha256(state),
        "n_independent_readin": fc_in_in == core.WINDOW_SIZE_BINS,
        "variable_n_loadable": True,
        "device": str(device),
        "map_location": map_location,
    }


def build_spint_net(architecture: Mapping[str, Any], *, identity_in_features: int, materialize_n: int = 4) -> Any:
    from src.models.components.spint import SpintModel

    net = SpintModel(**dict(architecture))
    window = int(architecture["window_size"])
    with torch.no_grad():
        src = torch.zeros(1, window, materialize_n)
        calib = torch.zeros(1, 1, int(identity_in_features), materialize_n)
        net.eval()
        net(src, calib)
    observed = materialized_identity_in_features(net)
    core.require(observed == int(identity_in_features),
                 f"identity MLP in_features {observed} != required {identity_in_features}")
    return net


def write_matching_teacher_checkpoint(
    dest: Path,
    *,
    architecture: Mapping[str, Any],
    identity_in_features: int,
    lightning_task: str = core.STUDENT_LIGHTNING_TASK,
    seed: int = 0,
) -> dict[str, Any]:
    """Construct a FalconLitModule checkpoint that the existing student loader can ingest.

    Weights are synthetic.  This proves format/architecture loadability, not a
    trained CO-native teacher.
    """
    _require_cpu()
    dest = Path(dest).expanduser()
    core.require(not dest.exists(), f"refusing to overwrite teacher checkpoint: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    net = build_spint_net(architecture, identity_in_features=identity_in_features)
    import lightning.pytorch as pl
    from src.models.falcon_module import FalconLitModule

    module = FalconLitModule(
        task=lightning_task,
        net=net,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        optimizer=partial(torch.optim.Adam, lr=5e-5, weight_decay=0.0),
        scheduler=None,
        compile=False,
    )
    payload = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "hparams_name": "kwargs",
        "pytorch-lightning_version": pl.__version__,
        "c1_synthetic_teacher": True,
        "c1_not_a_trained_teacher": True,
    }
    torch.save(payload, dest)
    return {
        "path": str(dest.resolve()),
        "sha256": core.sha256_file(dest),
        "bytes": dest.stat().st_size,
        "synthetic": True,
        "trained": False,
        "architecture": dict(architecture),
        "identity_mlp_in_features": int(identity_in_features),
        "lightning_task": lightning_task,
        "seed": int(seed),
    }


def load_into_student_contract(
    teacher_ckpt: Path,
    *,
    side_dim: int = 4,
    loss_mode: str = core.LOSS_MODE,
    hidden_dim: int | None = None,
    require_production_architecture: bool = False,
) -> dict[str, Any]:
    """Run StreamingCalibrationLitModule.setup on CPU under the frozen student contract."""
    _require_cpu()
    teacher_ckpt = Path(teacher_ckpt).expanduser().resolve()
    core.refuse_target_or_sealed_paths([teacher_ckpt], label="student teacher_ckpt")
    from src.models.falcon_module import FalconLitModule
    from src.models.streaming_calibration_module import StreamingCalibrationLitModule

    teacher = FalconLitModule.load_from_checkpoint(str(teacher_ckpt), weights_only=False, map_location="cpu")
    architecture = spint_architecture_from_net(teacher.net)
    identity_in = materialized_identity_in_features(teacher.net)
    core.require(
        int(architecture["window_size"]) == core.WINDOW_SIZE_BINS,
        "teacher window_size is not the frozen student W=50 contract",
    )
    core.require(
        identity_in == core.TRIAL_LENGTH_BINS,
        "teacher identity MLP in_features is not the frozen student trial_length=100 contract",
    )
    if require_production_architecture:
        core.require(
            architecture == dict(core.PRODUCTION_SPINT_HYPERPARAMETERS),
            "teacher architecture is not the production SpintModel contract",
        )
    hidden = int(hidden_dim) if hidden_dim is not None else min(64, int(architecture["model_dim"]))
    student = StreamingCalibrationLitModule(
        task=core.STUDENT_LIGHTNING_TASK,
        variant=core.VARIANT,
        teacher_ckpt_path=str(teacher_ckpt),
        window_size=int(architecture["window_size"]),
        trial_length=core.TRIAL_LENGTH_BINS,
        id_hidden_dim=hidden,
        hidden_dim=hidden,
        pad_value=-1.0,
        freeze_decoder=False,
        encoder_warmstart_path=None,
        loss_mode=loss_mode,  # type: ignore[arg-type]
        lambda_y=1.0,
        lambda_E=0.1,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        identity_mode=core.IDENTITY_MODE,
        side_dim=int(side_dim),
        decoder_mode=core.DECODER_MODE,
        optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0.0),
        scheduler=None,
        compile=False,
    )
    student.setup("fit")
    core.require(student.student is not None, "student.setup did not construct a student")
    decoder_state = student.student.decoder.state_dict()
    teacher_state = teacher.net.state_dict()
    core.require(set(decoder_state) == set(teacher_state), "student decoder keys drifted from teacher")
    mismatched = [
        name
        for name, tensor in teacher_state.items()
        if not torch.equal(tensor.detach().cpu(), decoder_state[name].detach().cpu())
    ]
    core.require(not mismatched, f"task_only student decoder is not a strict copy of the teacher: {mismatched[:8]}")
    encoder_names = list(student.student.id_encoder.state_dict())
    copied_id = any(name.startswith("fc_id_") for name in encoder_names)
    any_cuda = any(tensor.device.type == "cuda" for tensor in decoder_state.values())
    core.require(not any_cuda, "student decoder tensors landed on CUDA")
    return {
        "loaded": True,
        "lightning_task": core.STUDENT_LIGHTNING_TASK,
        "variant": core.VARIANT,
        "loss_mode": loss_mode,
        "encoder_warmstart_path": None,
        "freeze_decoder": False,
        "add_site": core.ADD_SITE,
        "side_dim": int(side_dim),
        "decoder_state_dict_strict_copy_of_teacher": True,
        "b3s_encoder_contains_teacher_fc_id": copied_id,
        "selected_t4_encoder_warmstart": False,
        "student_device": "cpu",
        "architecture": architecture,
        "identity_mlp_in_features": materialized_identity_in_features(teacher.net),
        "teacher_state_dict_sha256": _tensor_sha256(teacher_state),
        "decoder_state_dict_sha256": _tensor_sha256(decoder_state),
    }


def diff_decoder_initialization(mc_snapshot: Mapping[str, Any], co_load: Mapping[str, Any]) -> dict[str, Any]:
    """Record exact student-contract differences.  Do not interpret them as a cause."""
    mc_arch = dict(mc_snapshot.get("architecture") or {})
    co_arch = dict(co_load.get("architecture") or {})
    mc_shapes = dict(mc_snapshot.get("state_dict_shapes") or {})
    return {
        "hypothesis_status": "plausible_contributor_not_isolated_cause",
        "isolated_cause_claimed": False,
        "architecture_hyperparameters_identical": mc_arch == co_arch,
        "lightning_task_string": {
            "mc_maze_teacher": mc_snapshot.get("lightning_task"),
            "co_native_and_student": co_load.get("lightning_task"),
            "note": "the existing student constructor hard-codes task='mc_maze' even on CO data",
        },
        "identity_mlp_in_features": {
            "mc_maze_teacher": mc_snapshot.get("identity_mlp_in_features"),
            "co_native_required": co_load.get("identity_mlp_in_features"),
            "must_equal_trial_length": core.TRIAL_LENGTH_BINS,
        },
        "readin_fc_in": {
            "in_features": mc_snapshot.get("fc_in_in_features"),
            "n_independent": mc_snapshot.get("n_independent_readin"),
            "note": "decoder accepts variable unit counts even though the MC-Maze teacher was trained in a fixed-N regime",
        },
        "weight_origin": {
            "mc_maze": {
                "dataset": "DANDI 000128 MC_Maze",
                "behavior": "hand_vel",
                "unit_regime": "fixed_N_training_distribution",
                "checkpoint_sha256": mc_snapshot.get("sha256"),
            },
            "co_native_spec": {
                "dataset": "DANDI 000688 sub-C CO source-train 27 only",
                "behavior": "cursor_vel",
                "unit_regime": "variable_N_max_units_exclusive_100",
                "trained_teacher_exists": False,
            },
        },
        "student_load_path": {
            "copies_teacher_state_dict_strict": True,
            "does_so_under_task_only": True,
            "selected_t4_encoder_warmstart": False,
            "b3s_copies_teacher_fc_id_modules": bool(co_load.get("b3s_encoder_contains_teacher_fc_id")),
            "loss_mode_does_not_skip_decoder_init": True,
        },
        "state_dict_keys_compared_to_mc_maze": sorted(mc_shapes),
        "what_this_diff_is_not": [
            "not an accuracy result",
            "not proof that MC-Maze mismatch is the remaining error",
            "not authorization to train or launch C1",
        ],
    }


def build_compatibility_receipt(
    *,
    work_dir: Path,
    teacher_path: Path = core.MC_MAZE_TEACHER_PATH,
    use_production_architecture: bool = True,
    tiny_architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_cpu()
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    core.refuse_target_or_sealed_paths([work_dir, teacher_path], label="compatibility work")
    operations = {
        "nwb_opened": False,
        "target_subject_m_opened": False,
        "sealed_formal_test_sessions_opened": False,
        "gpu_used": False,
        "training_run": False,
        "torch_imported": True,
        "mc_maze_teacher_deserialized": False,
        "synthetic_co_native_teacher_constructed": False,
        "student_contract_setup_run": False,
    }
    blockers: list[dict[str, str]] = []
    manifest = core.load_strict_manifest()
    source_train_only = list(manifest["train"])
    core.refuse_sealed_sessions(source_train_only, label="source-train roster")

    mc_snapshot: dict[str, Any] | None = None
    try:
        mc_snapshot = snapshot_teacher_checkpoint(teacher_path)
        operations["mc_maze_teacher_deserialized"] = True
        if mc_snapshot["sha256"] != core.EXPECTED_TEACHER_SHA256:
            blockers.append({"code": "MC_MAZE_TEACHER_SHA_DRIFT", "detail": mc_snapshot["sha256"]})
    except Exception as exc:
        blockers.append({"code": "MC_MAZE_TEACHER_LOAD_FAILED", "detail": str(exc)})

    architecture: Mapping[str, Any]
    identity_in = core.IDENTITY_MLP_IN_FEATURES
    if use_production_architecture:
        architecture = dict(core.PRODUCTION_SPINT_HYPERPARAMETERS)
        if mc_snapshot is not None and mc_snapshot["architecture"] != architecture:
            blockers.append({"code": "PRODUCTION_ARCHITECTURE_DRIFT", "detail": str(mc_snapshot["architecture"])})
        if mc_snapshot is not None:
            identity_in = int(mc_snapshot["identity_mlp_in_features"])
    else:
        core.require(tiny_architecture is not None, "tiny architecture required when not using production")
        architecture = dict(tiny_architecture)

    synthetic_ckpt = work_dir / "synthetic_co_native_teacher.ckpt"
    co_teacher: dict[str, Any] | None = None
    student_load: dict[str, Any] | None = None
    if not blockers:
        try:
            co_teacher = write_matching_teacher_checkpoint(
                synthetic_ckpt,
                architecture=architecture,
                identity_in_features=identity_in,
                lightning_task=core.STUDENT_LIGHTNING_TASK,
                seed=0,
            )
            operations["synthetic_co_native_teacher_constructed"] = True
            student_load = load_into_student_contract(
                synthetic_ckpt,
                side_dim=4,
                loss_mode=core.LOSS_MODE,
                require_production_architecture=use_production_architecture,
            )
            operations["student_contract_setup_run"] = True
        except Exception as exc:
            blockers.append({"code": "CO_NATIVE_STUDENT_CONTRACT_LOAD_FAILED", "detail": str(exc)})

    diffs = (
        diff_decoder_initialization(mc_snapshot, student_load)
        if mc_snapshot is not None and student_load is not None
        else None
    )
    interface_ok = not blockers and student_load is not None and student_load.get("loaded") is True
    receipt = {
        "schema_version": 1,
        "screen_id": core.SCREEN_ID,
        "kind": "c1_source_only_teacher_compatibility_receipt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu_authorized": False,
        "training_authorized": False,
        "source_only": True,
        "target_or_sealed_data_opened": False,
        "operations": operations,
        "student_contract": dict(core.STUDENT_CONTRACT),
        "source_train_session_names_only": source_train_only,
        "source_train_session_count": len(source_train_only),
        "formal_test_session_names_only": list(manifest["test"]),
        "formal_test_sessions_opened": False,
        "mc_maze_teacher": mc_snapshot,
        "synthetic_co_native_teacher": co_teacher,
        "trained_co_native_teacher_exists": False,
        "student_contract_load": student_load,
        "decoder_initialization_diff": diffs,
        "interface_constructible": bool(interface_ok),
        "implementation_blockers": blockers,
        "hypothesis_status": "plausible_contributor_not_isolated_cause",
        "status": (
            "SOURCE_ONLY_COMPATIBILITY_PASSED_INTERFACE_ONLY"
            if interface_ok
            else "SOURCE_ONLY_COMPATIBILITY_FAILED"
        ),
        "note": (
            "A matching-architecture synthetic CO-native teacher loaded into the "
            "existing student contract under task_only.  This is not a trained "
            "teacher, not an accuracy result, and not GPU authorization.  The "
            "recorded MC-Maze vs CO-native differences are the contract to test, "
            "not an isolated cause."
            if interface_ok
            else "CO-native teacher could not be constructed and loaded into the existing student contract."
        ),
    }
    return receipt


def write_compatibility_receipt(
    out_path: Path,
    *,
    work_dir: Path,
    teacher_path: Path = core.MC_MAZE_TEACHER_PATH,
    use_production_architecture: bool = True,
    tiny_architecture: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, str, dict[str, Any]]:
    receipt = build_compatibility_receipt(
        work_dir=work_dir,
        teacher_path=teacher_path,
        use_production_architecture=use_production_architecture,
        tiny_architecture=tiny_architecture,
    )
    body, sidecar, digest = core.write_immutable_json(out_path, receipt)
    return body, sidecar, digest, receipt
