"""Standalone validation epoch-window evaluator for decoupled K/V v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from select_teacher_readin_decoupled_kv_v2_protocol_dandi688 import (  # noqa: E402
    evaluate_fixed_v2_protocol_over_validation_sessions,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_protocol_epochs(
    total_epochs: int, burn_in: int
) -> tuple[int, ...]:
    if total_epochs < 1 or burn_in < 0 or burn_in >= total_epochs:
        raise ValueError(
            "require total_epochs>=1 and 0<=burn_in<total_epochs"
        )
    return tuple(range(burn_in + 1, total_epochs + 1))


def epoch_checkpoint_path(
    epoch_ckpt_dir: Path, protocol_epoch: int
) -> Path:
    if protocol_epoch < 1:
        raise ValueError("protocol epoch must be positive")
    return epoch_ckpt_dir / f"epoch_{protocol_epoch - 1:03d}.ckpt"


def select_epoch_window_checkpoints(
    run_dir: Path,
    epochs: Sequence[int],
    total_epochs: int,
) -> dict[int, Path]:
    epoch_dir = run_dir / "epoch_ckpts"
    selected = {
        epoch: epoch_checkpoint_path(epoch_dir, epoch)
        for epoch in epochs
    }
    missing = [path.name for path in selected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing v2 epoch checkpoints for {total_epochs}-epoch "
            f"protocol: {missing}"
        )
    return selected


def compute_variant_score(
    per_epoch_mean_r2: Mapping[int, float],
    epochs: Sequence[int],
) -> float:
    if sorted(per_epoch_mean_r2) != sorted(epochs):
        raise ValueError("variant score requires the exact declared epoch set")
    return sum(per_epoch_mean_r2[epoch] for epoch in epochs) / len(epochs)


def _validate_metadata(
    metadata: dict,
    *,
    manifest_path: Path,
    teacher_ckpt: Path,
    total_epochs: int,
) -> None:
    expected_scalar = {
        "schema_version": 2,
        "runner_family": "teacher_readin_decoupled_kv_v2",
        "lightning_module_class": (
            "src.models.decoupled_kv_v2_module."
            "TeacherReadinDecoupledLitModule"
        ),
        "status": "completed",
        "variant": "B3S",
        "signal_view": "sua",
        "task": "CO",
        "max_units_exclusive": 100,
        "held_out_test_evaluated": False,
    }
    for name, value in expected_scalar.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"v2 metadata {name}={metadata.get(name)!r}, "
                f"expected {value!r}"
            )
    if metadata.get("split_counts") != [27, 6, 6]:
        raise ValueError("v2 metadata must record strict 27/6/6")
    if str(manifest_path) != metadata.get("train_val_manifest"):
        raise ValueError("strict manifest path differs from training metadata")
    if sha256_file(manifest_path) != metadata.get(
        "train_val_manifest_sha256"
    ):
        raise ValueError("strict manifest SHA256 differs from training metadata")
    if sha256_file(teacher_ckpt) != metadata.get("teacher_sha256"):
        raise ValueError("teacher checkpoint SHA256 differs from training metadata")
    training = metadata.get("training") or {}
    if (
        training.get("max_epochs") != total_epochs
        or training.get("no_early_stopping") is not True
        or training.get("checkpoint_every_epoch") is not True
        or training.get("calibration_n_trials") != 30
        or training.get("world_size") != 1
    ):
        raise ValueError("v2 training protocol receipt is inconsistent")
    side = metadata.get("side_features") or {}
    if (
        side.get("group") != "t4"
        or side.get("pool_size") != 50
        or side.get("side_dim") != 4
    ):
        raise ValueError("v2 evaluator requires aligned T4@50")
    decoder = metadata.get("decoder_architecture") or {}
    if (
        decoder.get("architecture_family")
        != "teacher_readin_decoupled_kv_v2"
        or decoder.get("base_decoder_mode_argument") != "coupled"
        or decoder.get("active_decoder_mode")
        != "teacher_readin_decoupled_v2"
        or decoder.get("key_width") != 48
        or decoder.get("value_width") != 64
        or decoder.get("attention_heads") != 1
        or decoder.get("legacy_decoder_transformer_active") is not False
    ):
        raise ValueError("v2 decoder metadata is inconsistent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--train_val_manifest", required=True)
    parser.add_argument("--total_epochs", type=int, required=True)
    parser.add_argument("--burn_in", type=int, required=True)
    parser.add_argument("--calibration_n", type=int, required=True)
    parser.add_argument("--pool_size", type=int, required=True)
    parser.add_argument("--out_path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if (
        args.total_epochs != 12
        or args.burn_in != 4
        or args.calibration_n != 30
        or args.pool_size != 50
    ):
        raise ValueError(
            "v2 matched score fixes epochs=12, burn-in=4, "
            "M_activity=30 and evaluation start=50"
        )
    epochs = compute_protocol_epochs(args.total_epochs, args.burn_in)
    run_dir = Path(args.run_dir).expanduser().resolve()
    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest_path = Path(args.train_val_manifest).expanduser().resolve()
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir else None
    )
    metadata_path = run_dir / "run_metadata.json"
    for path, description in (
        (metadata_path, "v2 run metadata"),
        (teacher_ckpt, "teacher checkpoint"),
        (manifest_path, "strict manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {description}: {path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing data directory: {data_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_metadata(
        metadata,
        manifest_path=manifest_path,
        teacher_ckpt=teacher_ckpt,
        total_epochs=args.total_epochs,
    )
    checkpoints = select_epoch_window_checkpoints(
        run_dir, epochs, args.total_epochs
    )

    per_epoch: dict[str, dict] = {}
    session_splits = None
    session_unit_counts = None
    for epoch in epochs:
        checkpoint = checkpoints[epoch]
        result = evaluate_fixed_v2_protocol_over_validation_sessions(
            ckpt_path=checkpoint,
            teacher_ckpt=teacher_ckpt,
            variant="B3S",
            data_dir=data_dir,
            task="CO",
            split_counts=(27, 6, 6),
            max_units_exclusive=100,
            cache_dir=cache_dir,
            pool_size=50,
            selection_mode="first",
            calibration_n=30,
            signal_view="sua",
            train_val_manifest=manifest_path,
        )
        if session_splits is None:
            session_splits = result["session_splits"]
            session_unit_counts = result["session_unit_counts"]
        elif result["session_splits"] != session_splits:
            raise ValueError(f"session split drift at protocol epoch {epoch}")
        elif result["session_unit_counts"] != session_unit_counts:
            raise ValueError(f"unit-count drift at protocol epoch {epoch}")
        if result.get("formal_test_files_opened") != 0:
            raise ValueError("v2 evaluator opened a formal test file")
        per_epoch[str(epoch)] = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "per_session_r2": result["per_session_r2"],
            "mean_r2": result["mean_r2"],
        }

    per_epoch_mean = {
        epoch: per_epoch[str(epoch)]["mean_r2"] for epoch in epochs
    }
    variant_score = compute_variant_score(per_epoch_mean, epochs)
    decoder = metadata["decoder_architecture"]
    payload = {
        "schema_version": 2,
        "purpose": (
            "teacher_readin_decoupled_kv_v2_validation_epoch_window"
        ),
        "generated_by": (
            "eval_epoch_window_decoupled_v2_dandi688.py"
        ),
        "protocol_metric_source": (
            "select_teacher_readin_decoupled_kv_v2_protocol_dandi688."
            "evaluate_fixed_v2_protocol_over_validation_sessions"
        ),
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "run_metadata_path": str(metadata_path),
        "run_metadata_sha256": sha256_file(metadata_path),
        "variant": "B3S",
        "seed": metadata["seed"],
        "task": "CO",
        "data_dir": str(data_dir),
        "teacher_ckpt": str(teacher_ckpt),
        "teacher_ckpt_sha256": sha256_file(teacher_ckpt),
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "signal_view": "sua",
        "train_val_manifest": str(manifest_path),
        "train_val_manifest_sha256": sha256_file(manifest_path),
        "decoder_architecture": {
            "architecture_family": decoder["architecture_family"],
            "key_mode": decoder["key_mode"],
            "key_width": decoder["key_width"],
            "value_width": decoder["value_width"],
            "attention_heads": decoder["attention_heads"],
            "key_permutation_seed": decoder["key_permutation_seed"],
            "online_cost_receipt_reference_n64": decoder[
                "online_cost_receipt_reference_n64"
            ],
        },
        "protocol": {
            "name": "fixed_epoch_window_no_argmax",
            "total_epochs": 12,
            "epoch_window": list(epochs),
            "burn_in_epochs": 4,
            "selection_mode": "first",
            "train_activity_calibration_n": 30,
            "evaluation_forward_calibration_n": 30,
            "label_feature_calibration_n": 50,
            "pool_size": 50,
            "evaluation_trials": "chronological trials[50:]",
        },
        "epoch_list": list(epochs),
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": {
            str(epoch): per_epoch_mean[epoch] for epoch in epochs
        },
        "variant_score": variant_score,
        "variant_score_definition": (
            "unweighted mean of six-session validation mean R2 over "
            "protocol epochs 5..12"
        ),
        "session_splits": session_splits,
        "session_unit_counts": session_unit_counts,
        "calibration_trial_selection_uses_behavior_labels": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_feature_label_scope": (
            "chronological_rewarded_trials[0:50]"
        ),
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "active_factor_sha_verified_per_checkpoint": True,
        "formal_test_paths_resolved": False,
        "formal_test_files_opened": 0,
        "no_test_files_evaluated": True,
    }
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote v2 validation epoch-window result: {out_path}")
    print(f"Variant score: {variant_score:.6f}")


if __name__ == "__main__":
    main()
