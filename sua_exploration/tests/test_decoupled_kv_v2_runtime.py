"""CPU contracts for the standalone v2 train/evaluation entrypoints."""
from __future__ import annotations

import sys
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

_SUA_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _SUA_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

from decoupled_kv_v2_runtime import checkpoint_v2_config
from eval_epoch_window_decoupled_v2_dandi688 import (
    _validate_metadata,
    compute_protocol_epochs,
    compute_variant_score,
    epoch_checkpoint_path,
)
from select_teacher_readin_decoupled_kv_v2_protocol_dandi688 import (
    _validate_v2_run_metadata,
    evaluate_fixed_v2_protocol_over_validation_sessions,
    validate_t4_side_receipt,
)
from mc_maze.unit_side_features import (
    feature_semantics_version,
    side_feature_stats_sha256,
)
from train_variant_dandi688_decoupled_v2 import (
    build_parser,
    validate_args,
)
from src.models.decoupled_kv_v2_module import (
    TeacherReadinDecoupledLitModule,
)
from validate_v1_decoupled_gate import (
    ARMS,
    validate_gate,
    validate_result_payload,
)


def _checkpoint_config(
    *,
    mode: str = "e_t4",
    key_dim: int = 48,
    value_dim: int = 64,
    seed=None,
    teacher_sha: str = "d" * 64,
) -> dict:
    receipt = {
        "module": "TeacherReadinDecoupledLitModule",
        "v2_key_mode": mode,
        "v2_key_dim": key_dim,
        "v2_value_dim": value_dim,
        "v2_key_permutation_seed": seed,
        "active_factor_sha256": "a" * 64,
        "teacher_checkpoint_sha256": teacher_sha,
    }
    return {
        "hyper_parameters": {
            "task": "mc_maze",
            "variant": "B3S",
            "window_size": 50,
            "trial_length": 100,
            "id_hidden_dim": 128,
            "hidden_dim": 64,
            "num_emas": 4,
            "num_filters": 4,
            "kernel_size": 5,
            "learnable_ema_alpha": False,
            "sparsity_k": 16,
            "pad_value": -1.0,
            "freeze_decoder": False,
            "freeze_encoder_base": False,
            "tune_encoder_fusion": False,
            "fusion_mean_lr_scale": 1.0,
            "loss_mode": "task_only",
            "lambda_y": 1.0,
            "lambda_E": 0.1,
            "decode_last_timestep_only": True,
            "predict_scaled_behavior": True,
            "behavior_scaling_factor": 5.0,
            "neuron_dropout_mode": "none",
            "neuron_dropout_p_low": 0.0,
            "neuron_dropout_p_high": 0.3,
            "neuron_dropout_block_size": 4,
            "neuron_dropout_warmup_epochs": 10,
            "support_prediction_consistency_weight": 0.0,
            "side_dim": 4,
            "electrode_embed_dim": 0,
            "num_electrodes": 0,
            "identity_mode": "calibrated",
            "decoder_mode": "coupled",
            "fixed_slot_count": 0,
            "fixed_slot_dim": 32,
            "fixed_slot_mode": "soft",
            "fixed_slot_fusion": "film",
            "fixed_slot_temperature": 1.0,
            "encoder_warmstart_path": None,
            "compile": False,
            "decoupled_key_mode": "e_t4",
            "decoupled_key_dim": 32,
            "decoupled_value_dim": 32,
            "decoupled_num_heads": 2,
            "decoupled_key_permutation_seed": None,
            "v2_key_mode": mode,
            "v2_key_dim": key_dim,
            "v2_value_dim": value_dim,
            "v2_key_permutation_seed": seed,
        },
        "teacher_readin_decoupled_v2_receipt": receipt,
    }


def _train_args(*extra: str):
    return build_parser().parse_args([
        "--v2_key_mode", "e_t4",
        "--out_name", "unit-test-v2",
        "--data_dir", "/data/not-opened",
        "--train_val_manifest", "/manifest/not-opened.json",
        "--no_early_stopping",
        "--checkpoint_every_epoch",
        *extra,
    ])


def _metadata(manifest: Path, teacher: Path) -> dict:
    return {
        "schema_version": 2,
        "runner_family": "teacher_readin_decoupled_kv_v2",
        "lightning_module_class": (
            "src.models.decoupled_kv_v2_module."
            "TeacherReadinDecoupledLitModule"
        ),
        "status": "completed",
        "seed": 42,
        "variant": "B3S",
        "signal_view": "sua",
        "task": "CO",
        "max_units_exclusive": 100,
        "held_out_test_evaluated": False,
        "split_counts": [27, 6, 6],
        "train_val_manifest": str(manifest),
        "train_val_manifest_sha256": _sha(manifest),
        "teacher_sha256": _sha(teacher),
        "training": {
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
            "calibration_n_trials": 30,
            "world_size": 1,
        },
        "side_features": {
            "group": "t4",
            "pool_size": 50,
            "side_dim": 4,
            "permutation_seed": None,
            "feature_version": feature_semantics_version("t4"),
            "normalization_sha256": side_feature_stats_sha256(
                np.zeros(4, dtype=np.float32),
                np.ones(4, dtype=np.float32),
            ),
        },
        "decoder_architecture": {
            "architecture_family": "teacher_readin_decoupled_kv_v2",
            "base_decoder_mode_argument": "coupled",
            "active_decoder_mode": "teacher_readin_decoupled_v2",
            "key_mode": "e_t4",
            "key_width": 48,
            "value_width": 64,
            "attention_heads": 1,
            "key_permutation_seed": None,
            "legacy_decoder_transformer_active": False,
            "encoder_side_input": "aligned_real_t4",
            "direct_t4_branch": "additive_4_to_48_zero_initialized",
        },
    }


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_v2_config_binds_hparams_receipt_and_fixed_ranks():
    config = checkpoint_v2_config(_checkpoint_config())
    assert config == {
        "v2_key_mode": "e_t4",
        "v2_key_dim": 48,
        "v2_value_dim": 64,
        "v2_key_permutation_seed": None,
    }
    checkpoint = _checkpoint_config()
    checkpoint["teacher_readin_decoupled_v2_receipt"][
        "active_factor_sha256"
    ] = "short"
    with pytest.raises(ValueError, match="active factor SHA256"):
        checkpoint_v2_config(checkpoint)
    with pytest.raises(ValueError, match="Dk=48"):
        checkpoint_v2_config(_checkpoint_config(key_dim=32))
    with pytest.raises(ValueError, match="permutation seed"):
        checkpoint_v2_config(_checkpoint_config(mode="e_ts4"))


def test_production_wrapper_hparams_satisfy_runtime_whitelist():
    module = TeacherReadinDecoupledLitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path="/not/opened.ckpt",
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        pad_value=-1.0,
        freeze_decoder=False,
        freeze_encoder_base=False,
        loss_mode="task_only",
        lambda_y=1.0,
        lambda_E=0.1,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        identity_mode="calibrated",
        fixed_slot_count=0,
        fixed_slot_dim=32,
        fixed_slot_mode="soft",
        fixed_slot_fusion="film",
        fixed_slot_temperature=1.0,
        decoder_mode="coupled",
        side_dim=4,
        electrode_embed_dim=0,
        num_electrodes=0,
        encoder_warmstart_path=None,
        optimizer=partial(torch.optim.Adam, lr=1.0e-4),
        scheduler=None,
        compile=False,
        v2_key_mode="e_t4",
        v2_key_dim=48,
        v2_value_dim=64,
        v2_key_permutation_seed=None,
    )
    checkpoint = {
        "hyper_parameters": dict(module.hparams),
        "teacher_readin_decoupled_v2_receipt": {
            "module": "TeacherReadinDecoupledLitModule",
            "v2_key_mode": "e_t4",
            "v2_key_dim": 48,
            "v2_value_dim": 64,
            "v2_key_permutation_seed": None,
            "active_factor_sha256": "a" * 64,
            "teacher_checkpoint_sha256": "d" * 64,
        },
    }
    assert checkpoint_v2_config(checkpoint)["v2_key_mode"] == "e_t4"


def test_runtime_loader_calls_strict_restore_hooks_in_order(monkeypatch):
    import decoupled_kv_v2_runtime as runtime

    checkpoint = _checkpoint_config()
    checkpoint["state_dict"] = {}
    events = []

    class FakeModule:
        def __init__(self, **kwargs):
            events.append("construct")

        def setup(self, stage):
            assert stage == "fit"
            events.append("setup")

        def on_load_checkpoint(self, loaded):
            assert loaded is checkpoint
            events.append("on_load_checkpoint")

        def load_state_dict(self, state, strict):
            assert state == {} and strict is True
            events.append("strict_load")

        def validate_loaded_v2_checkpoint_receipt(self):
            events.append("validate_factor_receipt")

        def parameters(self):
            return []

        def to(self, device):
            events.append("to")
            return self

        def eval(self):
            events.append("eval")
            return self

    monkeypatch.setattr(runtime.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        runtime, "_sha256_file", lambda path: "d" * 64
    )
    monkeypatch.setattr(
        runtime, "TeacherReadinDecoupledLitModule", FakeModule
    )
    runtime.load_frozen_v2_model(
        Path("/not/opened.ckpt"),
        Path("/teacher/not/opened.ckpt"),
        "B3S",
        torch.device("cpu"),
    )
    assert events[:5] == [
        "construct",
        "setup",
        "on_load_checkpoint",
        "strict_load",
        "validate_factor_receipt",
    ]


def test_standalone_train_parser_fails_closed_on_matched_contract():
    validate_args(_train_args())
    validate_args(_train_args(
        "--v2_key_mode", "e_ts4",
        "--v2_key_permutation_seed", "42",
    ))
    with pytest.raises(ValueError, match="calibration_n_trials"):
        validate_args(_train_args("--calibration_n_trials", "20"))
    with pytest.raises(ValueError, match="permutation seed"):
        validate_args(_train_args(
            "--v2_key_mode", "e_ts4",
            "--v2_key_permutation_seed", "17",
        ))


def test_v2_epoch_math_matches_frozen_m3_window(tmp_path: Path):
    epochs = compute_protocol_epochs(12, 4)
    assert epochs == tuple(range(5, 13))
    assert epoch_checkpoint_path(tmp_path, 5).name == "epoch_004.ckpt"
    values = {epoch: float(epoch) for epoch in epochs}
    assert compute_variant_score(values, epochs) == 8.5
    with pytest.raises(ValueError, match="exact declared"):
        compute_variant_score({5: 0.1}, epochs)


def test_metadata_and_checkpoint_topology_are_cross_bound(
    tmp_path: Path,
):
    manifest = tmp_path / "manifest.json"
    teacher = tmp_path / "teacher.ckpt"
    manifest.write_text("{}")
    teacher.write_bytes(b"teacher")
    metadata = _metadata(manifest, teacher)
    checkpoint = _checkpoint_config(teacher_sha=_sha(teacher))
    _validate_metadata(
        metadata,
        manifest_path=manifest,
        teacher_ckpt=teacher,
        total_epochs=12,
    )
    _validate_v2_run_metadata(
        metadata,
        checkpoint=checkpoint,
        manifest_path=manifest,
    )
    metadata["decoder_architecture"]["key_mode"] = "x_only"
    with pytest.raises(ValueError, match="does not match checkpoint"):
        _validate_v2_run_metadata(
            metadata,
            checkpoint=checkpoint,
            manifest_path=manifest,
        )


@pytest.mark.parametrize(
    ("selection_mode", "calibration_n", "pool_size"),
    [
        ("direction_coverage", 30, 50),
        ("first", 20, 50),
        ("first", 30, 40),
    ],
)
def test_fixed_v2_helper_rejects_protocol_tuning(
    selection_mode,
    calibration_n,
    pool_size,
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="fixes first / n=30 / pool=50"):
        evaluate_fixed_v2_protocol_over_validation_sessions(
            ckpt_path=tmp_path / "unused.ckpt",
            teacher_ckpt=tmp_path / "teacher.ckpt",
            variant="B3S",
            data_dir=tmp_path,
            task="CO",
            split_counts=(27, 6, 6),
            max_units_exclusive=100,
            cache_dir=None,
            pool_size=pool_size,
            selection_mode=selection_mode,
            calibration_n=calibration_n,
            signal_view="sua",
            train_val_manifest=tmp_path / "manifest.json",
        )


def test_direct_v2_gate_binds_manifest_teacher_and_key_seed(
    tmp_path: Path,
):
    manifest = tmp_path / "manifest.json"
    teacher = tmp_path / "teacher.ckpt"
    manifest.write_text("{}")
    teacher.write_bytes(b"teacher")
    metadata = _metadata(manifest, teacher)
    checkpoint = _checkpoint_config(teacher_sha=_sha(teacher))
    _validate_v2_run_metadata(
        metadata, checkpoint=checkpoint, manifest_path=manifest
    )

    manifest.write_text('{"drift": true}')
    with pytest.raises(ValueError, match="manifest SHA256"):
        _validate_v2_run_metadata(
            metadata, checkpoint=checkpoint, manifest_path=manifest
        )
    manifest.write_text("{}")

    wrong_teacher = _checkpoint_config(teacher_sha="e" * 64)
    with pytest.raises(ValueError, match="checkpoint teacher SHA256"):
        _validate_v2_run_metadata(
            metadata, checkpoint=wrong_teacher, manifest_path=manifest
        )

    e_ts4 = _checkpoint_config(
        mode="e_ts4", seed=17, teacher_sha=_sha(teacher)
    )
    metadata["decoder_architecture"].update({
        "key_mode": "e_ts4",
        "key_permutation_seed": 17,
    })
    with pytest.raises(ValueError, match="must equal run seed"):
        _validate_v2_run_metadata(
            metadata, checkpoint=e_ts4, manifest_path=manifest
        )


def test_t4_normalization_feature_and_alignment_receipts(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    teacher = tmp_path / "teacher.ckpt"
    manifest.write_text("{}")
    teacher.write_bytes(b"teacher")
    metadata = _metadata(manifest, teacher)
    mean = np.zeros(4, dtype=np.float32)
    std = np.ones(4, dtype=np.float32)
    validate_t4_side_receipt(
        metadata,
        side_feature_group="t4",
        side_pool_size=50,
        permutation_seed=None,
        side_mean=mean,
        side_std=std,
    )
    with pytest.raises(ValueError, match="normalization SHA256"):
        validate_t4_side_receipt(
            metadata,
            side_feature_group="t4",
            side_pool_size=50,
            permutation_seed=None,
            side_mean=mean + 1.0,
            side_std=std,
        )
    metadata["side_features"]["feature_version"] = -1
    with pytest.raises(ValueError, match="feature semantics"):
        validate_t4_side_receipt(
            metadata,
            side_feature_group="t4",
            side_pool_size=50,
            permutation_seed=None,
            side_mean=mean,
            side_std=std,
        )
    with pytest.raises(ValueError, match="aligned T4"):
        validate_t4_side_receipt(
            metadata,
            side_feature_group="t4",
            side_pool_size=50,
            permutation_seed=42,
            side_mean=mean,
            side_std=std,
        )


def _write_v1_gate_fixture(tmp_path: Path) -> tuple[Path, Path]:
    result_dir = tmp_path / "results"
    result_dir.mkdir(parents=True)
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text("{}")
    metadata_sha = _sha(metadata_path)
    artifacts = {}
    for arm in ARMS:
        result_path = (result_dir / f"{arm}_m50_s42.json").resolve()
        payload = {
            "schema_version": 1,
            "purpose": "epoch_window_deterministic_checkpoint_selection",
            "variant": "B3S",
            "seed": 42,
            "signal_view": "sua",
            "split_counts": [27, 6, 6],
            "max_units_exclusive": 100,
            "epoch_list": list(range(5, 13)),
            "run_metadata_path": str(metadata_path),
            "run_metadata_sha256": metadata_sha,
            "no_test_files_evaluated": True,
            "uses_behavior_labels_for_weight_updates": False,
            "uses_backward_gradients": False,
            "protocol": {
                "total_epochs": 12,
                "burn_in_epochs": 4,
                "epoch_window": list(range(5, 13)),
                "selection_mode": "first",
                "calibration_n": 30,
                "evaluation_forward_calibration_n": 30,
                "train_activity_calibration_n": 30,
                "label_feature_calibration_n": 50,
                "pool_size": 50,
            },
        }
        result_path.write_text(json.dumps(payload))
        artifacts[arm] = {"42": str(result_path)}
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps({
        "schema_version": 1,
        "purpose": "fresh_t4_decoupled_key_value_screen",
        "protocol": {
            "seeds": [42],
            "M_activity": 30,
            "M_T4": 50,
            "common_evaluation_start": 50,
            "epochs": 12,
            "scored_epoch_window": list(range(5, 13)),
            "formal_test_evaluated": False,
        },
        "artifacts": artifacts,
        "arm_mean_r2": {arm: 0.1 for arm in ARMS},
        "stage0_descriptive_mechanism_pass": False,
    }))
    return aggregate_path, result_dir


def test_v1_gate_rejects_bad_json_seed_epoch_and_missing_arm(
    tmp_path: Path,
):
    aggregate_path, result_dir = _write_v1_gate_fixture(tmp_path)
    calls = []
    receipt = validate_gate(
        aggregate_path=aggregate_path,
        result_dir=result_dir,
        arm_validator=lambda path, arm, seed: calls.append(
            (path, arm, seed)
        ),
    )
    assert len(calls) == 5
    assert receipt["formal_test_evaluated"] is False

    aggregate = json.loads(aggregate_path.read_text())
    aggregate["protocol"]["seeds"] = [43]
    aggregate_path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="aggregate seeds"):
        validate_gate(
            aggregate_path=aggregate_path,
            result_dir=result_dir,
            arm_validator=lambda *args: None,
        )

    aggregate_path.write_text("{bad json")
    with pytest.raises(ValueError, match="invalid JSON"):
        validate_gate(
            aggregate_path=aggregate_path,
            result_dir=result_dir,
            arm_validator=lambda *args: None,
        )

    aggregate_path, result_dir = _write_v1_gate_fixture(
        tmp_path / "second"
    )
    result_path = result_dir / "coupled_t4_m50_s42.json"
    result = json.loads(result_path.read_text())
    result["protocol"]["epoch_window"] = [5]
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="epoch window"):
        validate_gate(
            aggregate_path=aggregate_path,
            result_dir=result_dir,
            arm_validator=lambda *args: None,
        )

    aggregate_path, result_dir = _write_v1_gate_fixture(
        tmp_path / "third"
    )
    (result_dir / "kv_x_only_m50_s42.json").unlink()
    with pytest.raises(FileNotFoundError):
        validate_gate(
            aggregate_path=aggregate_path,
            result_dir=result_dir,
            arm_validator=lambda *args: None,
        )


def test_v2_entrypoints_do_not_import_active_v1_entrypoints():
    train_source = (
        _SCRIPT_DIR / "train_variant_dandi688_decoupled_v2.py"
    ).read_text()
    eval_source = (
        _SCRIPT_DIR / "eval_epoch_window_decoupled_v2_dandi688.py"
    ).read_text()
    protocol_source = (
        _SCRIPT_DIR
        / "select_teacher_readin_decoupled_kv_v2_protocol_dandi688.py"
    ).read_text()
    assert "import train_variant_dandi688" not in train_source
    assert "eval_epoch_window_generic_dandi688" not in eval_source
    assert "select_gradient_free_protocol_dandi688" not in protocol_source
    scheduler_source = (
        _SCRIPT_DIR / "schedule_after_decoupled_v1_stage0.sh"
    ).read_text()
    assert "run_sua_t4_shrinkage_one_cell" not in scheduler_source
    assert "eval_adaptation_dandi688" not in scheduler_source
    assert "aggregate_seed42.json" in scheduler_source
    assert "aggregate_sua_decoupled_kv_v2.py" in scheduler_source
    assert "aggregate_seeds42_43_44.json" in scheduler_source
    assert "stage0_descriptive_candidate_pass" in scheduler_source
    assert "run_v2_replication_seed 43 0" in scheduler_source
    assert "run_v2_replication_seed 44 1" in scheduler_source
    assert "run_v1_coupled_seed" in scheduler_source
    assert "--seeds 42,43,44" in scheduler_source
    for arm in (
        "coupled_t4", "kv_e_t4", "kv_e_ts4", "kv_e_only", "kv_x_only"
    ):
        assert arm in scheduler_source
    for arm in ("kv2_e_t4", "kv2_e_ts4", "kv2_e_only", "kv2_x_only"):
        assert arm in scheduler_source


def test_strict_v2_protocol_never_counts_or_opens_formal_paths(
    tmp_path: Path,
    monkeypatch,
):
    import select_teacher_readin_decoupled_kv_v2_protocol_dandi688 as protocol

    run_dir = tmp_path / "run"
    epoch_dir = run_dir / "epoch_ckpts"
    epoch_dir.mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    teacher = tmp_path / "teacher.ckpt"
    manifest.write_text("{}")
    teacher.write_bytes(b"teacher")
    checkpoint_path = epoch_dir / "epoch_004.ckpt"
    checkpoint = _checkpoint_config(teacher_sha=_sha(teacher))
    import torch

    torch.save(checkpoint, checkpoint_path)
    (run_dir / "run_metadata.json").write_text(
        __import__("json").dumps(_metadata(manifest, teacher))
    )
    train_path = tmp_path / "train.nwb"
    val_path = tmp_path / "val.nwb"
    counted = []
    monkeypatch.setattr(
        protocol,
        "load_frozen_train_val_manifest",
        lambda manifest_path, data_dir: (
            [train_path],
            [val_path],
            ["formal-session-name-only"],
        ),
    )
    monkeypatch.setattr(
        protocol,
        "fit_behavior_stats",
        lambda *args, **kwargs: (0.0, 1.0),
    )
    monkeypatch.setattr(
        protocol,
        "load_side_feature_stats_for_run_metadata",
        lambda *args, **kwargs: (
            "t4",
            None,
            50,
            None,
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        protocol, "load_frozen_v2_model", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        protocol,
        "load_session_with_trials",
        lambda *args, **kwargs: {
            "name": "validation-session",
            "trials": [{} for _ in range(51)],
        },
    )
    monkeypatch.setattr(
        protocol, "attach_side_features", lambda rec, *args, **kwargs: rec
    )
    monkeypatch.setattr(
        protocol,
        "evaluate_session_configs",
        lambda *args, **kwargs: (
            {"gradient_free_calibrated_first_n30": 0.5},
            {"gradient_free_calibrated_first_n30": {"indices": []}},
        ),
    )
    monkeypatch.setattr(
        protocol,
        "nwb_unit_count",
        lambda path: counted.append(path) or 64,
    )
    monkeypatch.setattr(
        protocol,
        "session_name_from_path",
        lambda path: path.stem,
    )
    result = evaluate_fixed_v2_protocol_over_validation_sessions(
        ckpt_path=checkpoint_path,
        teacher_ckpt=teacher,
        variant="B3S",
        data_dir=tmp_path,
        task="CO",
        split_counts=(27, 6, 6),
        max_units_exclusive=100,
        cache_dir=None,
        pool_size=50,
        selection_mode="first",
        calibration_n=30,
        signal_view="sua",
        train_val_manifest=manifest,
    )
    assert counted == [train_path, val_path]
    assert result["session_splits"]["test"] == [
        "formal-session-name-only"
    ]
    assert result["formal_test_paths_resolved"] is False
    assert result["formal_test_files_opened"] == 0
