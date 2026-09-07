#!/usr/bin/env python3
"""CPU-only preflight for the isolated M1 Version-B pilot.

The preflight is deliberately narrower than the training entry point.  It
resolves and hashes exactly the four ``held-in-calib`` files for fold 0,
checks that no forbidden scope appears in the resolved paths, and composes all
three Hydra arms without constructing a datamodule dataset.  A trusted local
teacher checkpoint is loaded on CPU only to verify the B0/B3S decoder and
initial-state invariants.  No minival, held-out, formal, or EvalAI path is
enumerated or opened.

This file is an audit/preflight, not a launcher.  It never calls datamodule
``setup`` and never starts a Trainer.  Use ``run_m1_version_b_5070ti.sh`` only
after this receipt has been reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = REPO_ROOT / "streaming_calibration_exp"
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

EXPECTED = {
    "ses-20120924": "sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb",
    "ses-20120926": "sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb",
    "ses-20120927": "sub-MonkeyL-held-in-calib_ses-20120927_behavior+ecephys.nwb",
    "ses-20120928": "sub-MonkeyL-held-in-calib_ses-20120928_behavior+ecephys.nwb",
}
ARMS = (
    ("m1_version_b_hs_continuation", "none", "B0"),
    ("m1_version_b_c0", "zero4", "B3S"),
    ("m1_version_b_c", "full", "B3S"),
)
FORBIDDEN = ("minival", "held-out", "heldout", "formal", "evalai", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sessions(data_root: Path) -> dict[str, Path]:
    """Resolve only the exact source directory; never use a recursive glob."""
    root = data_root.resolve()
    if root.name != "000941":
        raise ValueError(f"expected canonical .../data/000941 root, got {root}")
    directory = (root / "sub-MonkeyL-held-in-calib").resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = sorted(directory.glob("*.nwb"))
    if {path.name for path in paths} != set(EXPECTED.values()):
        raise RuntimeError(
            "source scope mismatch; expected exactly the four M1 held-in-calib files "
            f"but found {[path.name for path in paths]}"
        )
    out: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(directory)
        except ValueError as exc:
            raise RuntimeError(f"source symlink escapes allow-listed directory: {path}") from exc
        if any(token in str(resolved.relative_to(root)).lower() for token in FORBIDDEN):
            raise RuntimeError(f"forbidden path token in Version-B source: {resolved}")
        session = next((name for name, filename in EXPECTED.items() if filename == path.name), None)
        if session is None:
            raise RuntimeError(f"unexpected source filename: {path.name}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        out[session] = resolved
    if set(out) != set(EXPECTED):
        raise RuntimeError(f"source session set mismatch: {sorted(out)}")
    return out


def compose_arm(exp_name: str):
    from hydra import compose, initialize_config_dir

    config_dir = (STREAMING_ROOT / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                f"experiment={exp_name}",
                f"paths.root_dir={REPO_ROOT}",
                "hydra.job.chdir=false",
            ],
        )


def stable_state_hash(state: Any) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def config_invariants() -> dict[str, Any]:
    """Validate the terminal, single-query evaluation policy in all arms."""
    checks: dict[str, Any] = {}
    for exp_name, arm, variant in ARMS:
        cfg = compose_arm(exp_name)
        if str(cfg.optimized_metric) != "test_heldin/r2_mean":
            raise AssertionError(
                f"{exp_name}: optimized_metric must be test_heldin/r2_mean, "
                f"got {cfg.optimized_metric!r}"
            )
        if int(cfg.trainer.limit_val_batches) != 0:
            raise AssertionError(f"{exp_name}: limit_val_batches must be zero")
        if int(cfg.trainer.num_sanity_val_steps) != 0:
            raise AssertionError(f"{exp_name}: num_sanity_val_steps must be zero")
        checkpoint = cfg.callbacks.fixed_last_checkpoint
        if checkpoint.monitor is not None:
            raise AssertionError(f"{exp_name}: fixed checkpoint must not monitor a query metric")
        if bool(checkpoint.save_last):
            raise AssertionError(f"{exp_name}: save_last must be false")
        if int(checkpoint.every_n_epochs) != 12:
            raise AssertionError(f"{exp_name}: checkpoint must save only terminal epoch 11")
        if int(checkpoint.save_top_k) != -1:
            raise AssertionError(f"{exp_name}: fixed checkpoint must retain the terminal file only")
        if cfg.callbacks.early_stopping is not None:
            raise AssertionError(f"{exp_name}: early stopping must be disabled")
        if not bool(cfg.train) or not bool(cfg.test):
            raise AssertionError(f"{exp_name}: train/test must both be enabled")
        checks[exp_name] = {
            "carrier_arm": arm,
            "variant": variant,
            "optimized_metric": str(cfg.optimized_metric),
            "limit_val_batches": int(cfg.trainer.limit_val_batches),
            "num_sanity_val_steps": int(cfg.trainer.num_sanity_val_steps),
            "fixed_checkpoint": {
                "monitor": None,
                "save_last": bool(checkpoint.save_last),
                "save_top_k": int(checkpoint.save_top_k),
                "every_n_epochs": int(checkpoint.every_n_epochs),
                "terminal_epoch": 11,
            },
            "early_stopping": False,
        }
    return checks


def model_invariants() -> dict[str, Any]:
    """Build the models on CPU and verify common decoder/seed invariants.

    The teacher checkpoint is trusted local experiment output.  Lightning's
    restricted classmethod is unwrapped so the checkpoint is explicitly mapped
    to CPU; this avoids any CUDA initialization during a preflight.
    """
    import lightning.pytorch as pl
    import torch
    from hydra.utils import instantiate
    from src.models.falcon_module import FalconLitModule

    original_loader = FalconLitModule.load_from_checkpoint
    unwrapped_loader = original_loader.__wrapped__

    def cpu_loader(cls, checkpoint_path, *args, **kwargs):
        kwargs["map_location"] = "cpu"
        return unwrapped_loader(cls, checkpoint_path, *args, **kwargs)

    FalconLitModule.load_from_checkpoint = classmethod(cpu_loader)
    try:
        models: dict[str, Any] = {}
        for exp_name, _arm, _variant in ARMS:
            cfg = compose_arm(exp_name)
            model = instantiate(cfg.model)
            model.setup("fit")
            if model._freeze_decoder:
                raise AssertionError(f"{exp_name}: freeze_decoder must be false")
            if model._loss_mode != "task_only" or model._lambda_y != 0.0 or model._lambda_E != 0.0:
                raise AssertionError(f"{exp_name}: source-only task loss contract drifted")
            models[exp_name] = model

        decoder_hashes = {
            name: stable_state_hash(model.student.decoder.state_dict())
            for name, model in models.items()
        }
        if len(set(decoder_hashes.values())) != 1:
            raise AssertionError(f"three arms do not share the same initialized decoder: {decoder_hashes}")

        # Re-seeding must make the two B3S arms byte-identical before training.
        cfg_c0 = compose_arm("m1_version_b_c0")
        cfg_c = compose_arm("m1_version_b_c")
        # Seed immediately before each construction.  Instantiating the first
        # model consumes the global initializer RNG, so seeding only once (or
        # before both instantiations) would make this check a false negative.
        pl.seed_everything(42, workers=True)
        c0 = instantiate(cfg_c0.model)
        c0.setup("fit")
        pl.seed_everything(42, workers=True)
        c = instantiate(cfg_c.model)
        c.setup("fit")
        b3s_c0_hash = stable_state_hash(c0.student.id_encoder.state_dict())
        b3s_c_hash = stable_state_hash(c.student.id_encoder.state_dict())
        if b3s_c0_hash != b3s_c_hash:
            raise AssertionError("B-C0/B-C B3S initial encoder weights differ under seed 42")

        # Compare B0's decoder against the actual teacher net state, not an
        # inferred parameter count or a checkpoint filename.
        teacher_path = Path(cfg_c.model.teacher_ckpt_path)
        teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
        teacher_net_state = {
            key[len("net.") :]: value
            for key, value in teacher_payload["state_dict"].items()
            if key.startswith("net.")
        }
        teacher_hash = stable_state_hash(teacher_net_state)
        if decoder_hashes["m1_version_b_hs_continuation"] != teacher_hash:
            raise AssertionError("H-S B0 decoder is not byte-identical to the teacher checkpoint")
        return {
            "decoder_sha256": decoder_hashes,
            "teacher_net_sha256": teacher_hash,
            "b3s_c0_initial_encoder_sha256": b3s_c0_hash,
            "b3s_c_initial_encoder_sha256": b3s_c_hash,
            "params": {
                name: {
                    "student": sum(parameter.numel() for parameter in model.student.parameters()),
                    "decoder": sum(parameter.numel() for parameter in model.student.decoder.parameters()),
                    "encoder": sum(parameter.numel() for parameter in model.student.id_encoder.parameters()),
                }
                for name, model in models.items()
            },
        }
    finally:
        FalconLitModule.load_from_checkpoint = original_loader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "SPINT-main" / "data" / "000941",
        help="canonical data/000941 root (only held-in-calib is resolved)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "sua_exploration" / "results" / "m1_version_b_preflight" / "receipt_v2.json",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="skip checkpoint construction; useful on a machine without the SPINT Python environment",
    )
    args = parser.parse_args()

    sessions = canonical_sessions(args.data_root)
    source_hashes = {name: sha256(path) for name, path in sorted(sessions.items())}
    receipt: dict[str, Any] = {
        "schema": "m1_version_b_preflight_v2",
        "scope": {
            "task": "m1",
            "fold": 0,
            "source_sessions": ["ses-20120926", "ses-20120927", "ses-20120928"],
            "target_session": "ses-20120924",
            "support_trials": [0, 10],
            "query_trials": [10, 210],
            "formal_test_opened": False,
            "heldout_files_opened": False,
            "minival_files_opened": False,
            "target_backpropagation": False,
            "target_query_values_read_by_preflight": False,
        },
        "source_files": {
            name: {"path": str(path), "sha256": source_hashes[name], "bytes": path.stat().st_size}
            for name, path in sorted(sessions.items())
        },
        "arms": [
            {"experiment": exp_name, "carrier_arm": arm, "variant": variant, "seed": 42, "fixed_epoch": 11}
            for exp_name, arm, variant in ARMS
        ],
        "checks": {
            "source_directory_enumeration": "exact held-in-calib/*.nwb only",
            "forbidden_tokens": list(FORBIDDEN),
            "no_datamodule_setup": True,
            "no_target_query_validation_or_checkpoint_selection": True,
        },
    }
    receipt["config_invariants"] = config_invariants()
    if not args.skip_model:
        receipt["model_invariants"] = model_invariants()

    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"refusing to overwrite incompatible receipt: {args.output}")
    args.output.write_text(encoded, encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
