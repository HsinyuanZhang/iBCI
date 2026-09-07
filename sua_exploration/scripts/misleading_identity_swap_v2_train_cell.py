#!/usr/bin/env python3
"""Dedicated four-cell swap-v2 trainer; dry-run unless explicitly authorized."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
SCE_ROOT = REPO_ROOT / "streaming_calibration_exp"
for entry in (SUA_ROOT, SCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from mc_maze import a2_matched_subject_shift_v2_core as a2
from mc_maze import misleading_identity_swap_v2_core as core

AUTH_ENV = "SWAP_V2_STAGE_P_GPU_AUTHORIZATION"
AUTH_VALUE = "I_AUTHORIZE_SWAP_V2_STAGE_P_GPU"
DEFAULT_INITIAL = SUA_ROOT / "results" / "misleading_identity_swap_v2_source_authority_dev" / "shared_initial_state_v2.pt"


def cell_factors(cell: str) -> tuple[str, str]:
    core.require(cell in core.CELLS, "unknown swap-v2 cell")
    identity, carrier = cell.split("_", 1)
    return identity, carrier


def require_canonical_cell_output(
    official: dict[str, Any], *, cell: str, output_dir: Path,
) -> Path:
    core.require(cell in core.CELLS, "unknown swap-v2 cell")
    core.require(official.get("cell_output_root") == str(core.CELL_OUTPUT_ROOT.resolve()) and
                 official.get("cell_output_paths") == core.canonical_cell_output_paths(),
                 "official cell output topology drift")
    out = output_dir.resolve()
    expected = Path(official["cell_output_paths"][cell]).resolve()
    core.require(out == expected and out.parent == core.CELL_OUTPUT_ROOT.resolve(),
                 "cell output-dir is not the unique canonical official path")
    return out


def _model(authority: Path, identity: str):
    import torch
    from mc_maze.misleading_identity_swap_v2_trainer import MisleadingIdentitySwapV2LitModule

    return MisleadingIdentitySwapV2LitModule(
        task="mc_maze", variant="B3S", teacher_ckpt_path=str(a2.TEACHER_PATH.resolve()),
        window_size=50, trial_length=100, id_hidden_dim=128, hidden_dim=64,
        pad_value=-1.0, freeze_decoder=False, freeze_encoder_base=False,
        loss_mode="task_only", lambda_y=1.0, lambda_E=0.1,
        decode_last_timestep_only=True, predict_scaled_behavior=True,
        behavior_scaling_factor=5.0, identity_mode="calibrated",
        side_dim=4, electrode_embed_dim=0, num_electrodes=0,
        optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0.0),
        scheduler=None, compile=False,
        matching_authority_path=str(authority.resolve()),
        matching_authority_kind="source",
        identity_training_mode=identity,
        evaluation_input_mode="clean",
    )


def _state_sha(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _binary_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _write_immutable_binary(path: Path, raw: bytes) -> str:
    path = path.resolve()
    side = _binary_sidecar(path)
    if os.path.lexists(path) or os.path.lexists(side):
        raise FileExistsError(f"immutable initial-state pair is not fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    temp: list[Path] = []
    made_body = False
    try:
        for suffix, data in ((".pt.tmp", raw), (".sha.tmp", f"{digest}  {path.name}\n".encode("ascii"))):
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
            tmp = Path(name); temp.append(tmp)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            os.chmod(tmp, 0o444)
        os.link(temp[0], path); made_body = True
        os.link(temp[1], side)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except BaseException:
        if made_body and not os.path.lexists(side): path.unlink(missing_ok=True)
        raise
    finally:
        for tmp in temp: tmp.unlink(missing_ok=True)
    return digest


def _load_initial(path: Path) -> tuple[dict[str, Any], str]:
    import torch
    requested = path.expanduser()
    core.require(not requested.is_symlink(), "initial-state path may not be a symlink")
    resolved = requested.resolve(); side = _binary_sidecar(resolved)
    for candidate in (resolved, side):
        info = candidate.lstat()
        core.require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444,
                     "initial-state pair must be regular mode 0444")
    raw = resolved.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    core.require(side.read_bytes() == f"{digest}  {resolved.name}\n".encode("ascii"),
                 "initial-state sidecar mismatch")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    core.require(isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict),
                 "initial-state payload malformed")
    return payload, digest


def mint_initial_state(authority: Path, out: Path) -> dict[str, Any]:
    import lightning.pytorch as pl
    import torch
    core.assert_immutable_pair_fresh(out, label="shared initial state")
    core.load_verified_authority(authority)
    pl.seed_everything(42, workers=True)
    model = _model(authority, "clean"); model.setup("fit")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state_sha = _state_sha(state)
    buffer = io.BytesIO()
    torch.save({
        "schema_version": 1,
        "receipt_kind": "misleading_identity_swap_v2_shared_initial_state",
        "seed": 42,
        "matching_authority_sha256": core.sha256_file(authority),
        "state_dict_sha256": state_sha,
        "state_dict": state,
    }, buffer)
    file_sha = _write_immutable_binary(out, buffer.getvalue())
    return {"path": str(out.resolve()), "file_sha256": file_sha, "state_dict_sha256": state_sha}


def execute_cell(args: argparse.Namespace) -> dict[str, Any]:
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
    from scripts.train_variant_dandi688 import configure_multisession_metrics
    from scripts.misleading_identity_swap_v2_preflight import current_implementation_bindings
    from scripts.misleading_identity_swap_v2_preflight import load_verified_official_preflight

    official, official_sha = load_verified_official_preflight(args.official_preflight)
    identity, carrier = cell_factors(args.cell)
    out = require_canonical_cell_output(official, cell=args.cell, output_dir=args.output_dir)
    core.require(not out.exists(), f"fresh cell output directory already exists: {out}")
    core.require(os.environ.get(AUTH_ENV) == AUTH_VALUE, "explicit swap-v2 GPU authorization missing")
    core.require(torch.cuda.is_available(), "swap-v2 live training refuses CPU fallback")
    authority = core.load_verified_authority(args.authority)
    lineage, lineage_sha = core.load_verified_immutable_json(args.lineage)
    core.require(isinstance(lineage.get("implementation_bindings"), dict) and
                 bool(lineage.get("implementation_bindings")),
                 "source lineage construction closure missing")
    core.require(lineage.get("matching_authority_consumed_bytes_sha256") == authority.sha256,
                 "source lineage/authority byte binding drift")
    initial, initial_file_sha = _load_initial(args.initial_state)
    core.require(initial.get("matching_authority_sha256") == authority.sha256,
                 "initial state authority drift")
    core.require(official.get("matching_authority_path") == str(args.authority.resolve()) and
                 official.get("matching_authority_sha256") == authority.sha256,
                 "cell authority differs from official preflight")
    core.require(official.get("source_lineage_path") == str(args.lineage.resolve()) and
                 official.get("source_lineage_sha256") == lineage_sha,
                 "cell lineage differs from official preflight")
    core.require(official.get("initial_state_path") == str(args.initial_state.resolve()) and
                 official.get("initial_state_file_sha256") == initial_file_sha and
                 official.get("initial_state_dict_sha256") == initial.get("state_dict_sha256"),
                 "cell initial state differs from official preflight")
    out.mkdir(parents=True, exist_ok=False)
    launch_path = out / "launch_receipt.json"
    terminal_path = out / "terminal_receipt.json"
    pl.seed_everything(42, workers=True)
    dm = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBC_DATA_ROOT), task="CO", split_counts=(27, 6, 6), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=4, random_calibration=False, seed=42, max_units_exclusive=100,
        cache_dir=str(a2.SOURCE_CACHE_ROOT), signal_view="sua", side_feature_group=carrier,
        side_feature_pool_size=30, train_val_manifest_path=str(a2.MANIFEST_PATH),
    )
    dm.setup("fit")
    model = _model(args.authority, identity); model.setup("fit")
    model.load_state_dict(initial["state_dict"], strict=True)
    observed_state_sha = _state_sha(model.state_dict())
    core.require(observed_state_sha == initial.get("state_dict_sha256"), "shared initial-state load drift")
    configure_multisession_metrics(model, dm)
    launch = {
        "schema_version": 1, "receipt_kind": "misleading_identity_swap_v2_cell_launch",
        "status": "AUTHORIZED_LIVE_CELL_INITIALIZED", "cell": args.cell, "seed": 42,
        "identity_training_mode": identity, "carrier": carrier, "loss_mode": "task_only",
        "epochs": 12, "score_epochs_one_based": list(range(5, 13)), "learning_rate": 1e-4,
        "matching_authority_path": str(args.authority.resolve()),
        "matching_authority_sha256": authority.sha256,
        "source_lineage_path": str(args.lineage.resolve()), "source_lineage_sha256": lineage_sha,
        "initial_state_path": str(args.initial_state.resolve()), "initial_state_file_sha256": initial_file_sha,
        "initial_state_dict_sha256": observed_state_sha,
        "source_train_sessions": dm.session_splits["train"],
        "formal_test_session_names_only": dm.session_splits["test"],
        "formal_subc_test_nwb_opened": False, "target_nwb_opened": False,
        "implementation_bindings": current_implementation_bindings(),
        "implementation_bindings_sha256": official["implementation_bindings_sha256"],
        "official_preflight_path": str(core.OFFICIAL_PREFLIGHT_PATH.resolve()),
        "official_preflight_sha256": official_sha,
        "cell_output_root": str(core.CELL_OUTPUT_ROOT.resolve()),
        "cell_output_path": str(out),
        "python_isolation": a2.python_isolation_binding(),
        "torch_runtime": a2.torch_runtime_binding(visible_device_index=0),
    }
    core.write_immutable_json_pair(launch_path, launch)
    epoch_dir = out / "epoch_ckpts"
    callbacks = [
        ModelCheckpoint(dirpath=str(out), filename="best-{epoch:03d}-{val_heldin/r2_mean:.4f}",
                        monitor="val_heldin/r2_mean", mode="max", save_top_k=3),
        ModelCheckpoint(dirpath=str(epoch_dir), filename="epoch_{epoch:03d}", auto_insert_metric_name=False,
                        every_n_epochs=1, save_top_k=-1),
    ]
    trainer = pl.Trainer(max_epochs=12, accelerator="gpu", devices=1, callbacks=callbacks,
                         check_val_every_n_epoch=1, log_every_n_steps=50,
                         default_root_dir=str(out), deterministic=True, enable_progress_bar=False)
    trainer.fit(model, datamodule=dm)
    checkpoints = {str(epoch + 1): str((epoch_dir / f"epoch_{epoch:03d}.ckpt").resolve()) for epoch in range(12)}
    for path in checkpoints.values(): core.require(Path(path).is_file(), f"missing epoch checkpoint: {path}")
    terminal = {
        **launch, "receipt_kind": "misleading_identity_swap_v2_cell_terminal",
        "status": "CELL_TRAINING_COMPLETE__DEVELOPMENT_NOT_OFFICIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256_by_epoch_one_based": {
            epoch: core.sha256_file(Path(path)) for epoch, path in checkpoints.items()
        },
        "checkpoint_paths_by_epoch_one_based": checkpoints,
        "target_optimizer_or_backward_steps": 0,
    }
    core.write_immutable_json_pair(terminal_path, terminal)
    return {"status": terminal["status"], "cell": args.cell, "terminal_receipt": str(terminal_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=core.CELLS)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--initial-state", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--official-preflight", type=Path, default=core.OFFICIAL_PREFLIGHT_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mint-development-initial-state", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.mint_development_initial_state:
        print(json.dumps(mint_initial_state(args.authority, args.initial_state), indent=2, sort_keys=True)); return 0
    if not args.launch:
        print(json.dumps({"status": "DRY_RUN__NO_DATA_NO_GPU", "cell": args.cell,
                          "same_initial_state_required": True, "epochs": 12,
                          "score_epochs_one_based": list(range(5, 13))}, indent=2, sort_keys=True)); return 0
    core.require(args.cell is not None and args.lineage is not None and args.output_dir is not None,
                 "live cell requires --cell, --lineage, and --output-dir")
    print(json.dumps(execute_cell(args), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
