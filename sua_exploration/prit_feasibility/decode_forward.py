"""Forward-only decode of A2 validation sessions for the PRI-T feasibility gate.

Produces, per validation session, the frozen decoder's own velocity output for every
evaluation window, grouped by trial, plus the true ``target_dir`` retained ONLY as the
evaluation label for target-inference agreement.

Label-constraint contract (the point of this file):
  * For every *validation* (target) session this script opens ``nwb.units`` and
    ``nwb.intervals['trials']`` and nothing else.  It never touches
    ``processing/behavior/{Velocity,Position,Acceleration}``, so the measured
    ``cursor_vel`` / ``cursor_pos`` / ``cursor_acc`` streams cannot enter the HMM in any
    role -- not as an observation, not as a mask, not as a normalizer.
  * ``behavior_data`` handed to the windowing dataset is an explicit zero array, so even
    an accidental read of the dataset's behavior channel yields no target-session signal.
  * ``behavior_mean``/``behavior_std`` are fit on the 27 *source/train* sessions only
    (``fit_behavior_stats``), which is source-side information the frozen decoder was
    trained with; it is used solely to map the decoder's normalized output back into
    cursor-velocity units.  ``--raw_normalized`` skips even that.
  * ``target_dir`` from the target session's calibration pool trials[0:30] is read by the
    T4 side-feature estimator.  That is T4's own declared label budget, not new
    information.  ``target_dir`` on evaluation trials[30:] is written to the output as
    ``true_dir`` and is used only for scoring in prit_gate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pynwb import NWBHDF5IO
from torch.utils.data import DataLoader

_SUA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUA_ROOT))

from mc_maze.datamodule import MCMazeSessionDataset, bin_spikes
from mc_maze.multisession_datamodule import (
    _build_calib_trials,
    _compute_valid_starts,
    fit_behavior_stats,
    session_name_from_path,
)

sys.path.insert(0, str(_SUA_ROOT / "scripts"))
from eval_adaptation_dandi688 import (  # noqa: E402
    attach_side_features,
    checkpoint_architecture_kwargs,
    load_side_feature_stats_for_run_metadata,
)

_SCE_ROOT = _SUA_ROOT.parent / "streaming_calibration_exp"
sys.path.insert(0, str(_SCE_ROOT))
from src.models.streaming_calibration_module import StreamingCalibrationLitModule  # noqa: E402

WINDOW_SIZE = 50
TRIAL_LENGTH = 100
BIN_SIZE_MS = 20
PAD_VALUE = -1.0
BEHAVIOR_SCALING_FACTOR = 5.0
ID_HIDDEN_DIM = 128
HIDDEN_DIM = 64

FORBIDDEN_BEHAVIOR_KEYS = ("Velocity", "Position", "Acceleration")


def load_spikes_and_trials(nwb_path: Path, pool_size: int) -> dict:
    """Bin spikes and list rewarded trials without opening any kinematic stream."""
    bin_size_s = BIN_SIZE_MS / 1000.0
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        units_df = nwb.units.to_dataframe()
        n_units = len(units_df)

        all_spikes = np.concatenate(units_df["spike_times"].values)
        t_min = float(all_spikes.min())
        t_max = float(all_spikes.max())
        bin_edges = np.arange(t_min, t_max + bin_size_s, bin_size_s)
        num_bins = len(bin_edges) - 1

        binned = np.zeros((num_bins, n_units), dtype=np.float32)
        for i, (_, unit) in enumerate(units_df.iterrows()):
            binned[:, i] = bin_spikes(unit["spike_times"], bin_edges)

        trials_df = nwb.intervals["trials"].to_dataframe()
        trial_info = []
        for _, trial in trials_df.iterrows():
            if trial["result"] != "R":
                continue
            start_bin = max(0, int(np.searchsorted(bin_edges, trial["start_time"])))
            stop_bin = min(num_bins, int(np.searchsorted(bin_edges, trial["stop_time"])))
            if stop_bin - start_bin >= WINDOW_SIZE:
                trial_info.append(
                    {
                        "start": start_bin,
                        "stop": stop_bin,
                        "trial_index": int(trial.name),
                        "target_dir": (
                            None
                            if trial.get("target_dir") is None
                            or not np.isfinite(trial["target_dir"])
                            else float(trial["target_dir"])
                        ),
                        "target_id": (
                            None
                            if trial.get("target_id") is None
                            or not np.isfinite(trial["target_id"])
                            else int(trial["target_id"])
                        ),
                    }
                )

    calib_trials = _build_calib_trials(
        binned, trial_info, pool_size, TRIAL_LENGTH, n_units, PAD_VALUE, True
    )
    return {
        "name": session_name_from_path(nwb_path),
        "n_units": n_units,
        "neural": binned,
        # Explicit zeros: no target-session kinematics anywhere in this record.
        "behavior": np.zeros((binned.shape[0], 2), dtype=np.float32),
        "trials": trial_info,
        "calib_trials": calib_trials,
        "signal_view": "sua",
    }


def assert_no_kinematic_read(nwb_path: Path) -> dict:
    """Record that the forbidden streams exist and were deliberately not read."""
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        behavior = nwb.processing.get("behavior")
        present = [key for key in FORBIDDEN_BEHAVIOR_KEYS if behavior is not None and key in behavior.data_interfaces]
    return {"present_but_unread": present}


@torch.no_grad()
def decode_session(
    model,
    rec: dict,
    trials: list[dict],
    device: torch.device,
    zero_identity: bool,
    batch_size: int = 512,
) -> dict[int, np.ndarray]:
    """Return {trial_index: [T, 2] decoded normalized velocity} for these trials."""
    valid_starts = _compute_valid_starts(trials, WINDOW_SIZE)
    dataset = MCMazeSessionDataset(
        neural_data=rec["neural"],
        behavior_data=rec["behavior"],
        valid_starts=valid_starts,
        calib_trials=rec["calib_trials"],
        window_size=WINDOW_SIZE,
        session_name=rec["name"],
        side_features=rec.get("side_features"),
        electrode_ids=rec.get("electrode_ids"),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    outputs = np.zeros((len(valid_starts), 2), dtype=np.float64)
    cursor = 0
    for batch in loader:
        if len(batch) == 6:
            neural, _behavior, calib, _name, side_features, electrode_ids = batch
        elif len(batch) == 5:
            neural, _behavior, calib, _name, side_features = batch
            electrode_ids = None
        else:
            neural, _behavior, calib, _name = batch
            side_features, electrode_ids = None, None
        neural = neural.to(device)
        calib = calib.to(device)
        if side_features is not None:
            side_features = side_features.to(device)
        if electrode_ids is not None:
            electrode_ids = electrode_ids.to(device)

        if zero_identity:
            identity = torch.zeros(
                neural.shape[0], neural.shape[2], neural.shape[1],
                device=neural.device, dtype=neural.dtype,
            )
            y = model.student.decode_with_identity(neural, identity)
        else:
            decoder_key_features = model.decoder_key_features(side_features)
            y, _ = model.student(
                neural,
                calib_trials=calib,
                side_features=side_features,
                decoder_key_features=decoder_key_features,
                electrode_ids=electrode_ids,
            )
        y = (y[:, -1, :] / BEHAVIOR_SCALING_FACTOR).double().cpu().numpy()
        outputs[cursor : cursor + y.shape[0]] = y
        cursor += y.shape[0]
    assert cursor == len(valid_starts)

    # Window whose first bin is `start` decodes the bin `start + WINDOW_SIZE - 1`.
    per_trial: dict[int, np.ndarray] = {}
    offset = 0
    for trial in trials:
        n_windows = trial["stop"] - trial["start"] - WINDOW_SIZE + 1
        per_trial[trial["trial_index"]] = outputs[offset : offset + n_windows]
        offset += n_windows
    assert offset == len(valid_starts)
    return per_trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="checkpoints/<name> directory")
    parser.add_argument("--tag", required=True, help="output tag, e.g. t4 or z4")
    parser.add_argument(
        "--zero_identity",
        action="store_true",
        help="Bypass the identity encoder with an all-zero identity (cold-start control).",
    )
    parser.add_argument("--raw_normalized", action="store_true",
                        help="Skip the train-only un-normalization of decoder output.")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    run_metadata = json.loads((ckpt_dir / "run_metadata.json").read_text())
    ckpt_path = Path(run_metadata["best_checkpoint"])
    pool_size = int(run_metadata["side_features"]["pool_size"])
    data_dir = Path(run_metadata["data_dir"])
    manifest = json.loads(Path(run_metadata["train_val_manifest"]).read_text())
    cache_dir = Path(run_metadata["cache_dir"])

    train_names = manifest["session_splits"]["train"]
    val_names = manifest["session_splits"]["val"]
    forbidden = set(run_metadata["session_splits"]["test"])
    assert not (set(val_names) & forbidden), "validation split collides with formal test"
    train_files = [data_dir / f"{name}_behavior+ecephys.nwb" for name in train_names]
    val_files = [data_dir / f"{name}_behavior+ecephys.nwb" for name in val_names]
    for path in train_files + val_files:
        assert path.is_file(), path
        assert session_name_from_path(path) not in forbidden

    behavior_mean, behavior_std = fit_behavior_stats(
        train_files, bin_size_ms=BIN_SIZE_MS, cache_dir=cache_dir
    )
    side_feature_config = load_side_feature_stats_for_run_metadata(
        run_metadata, train_files, cache_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = StreamingCalibrationLitModule(
        task="mc_maze",
        variant=run_metadata["variant"],
        teacher_ckpt_path=run_metadata["teacher_checkpoint"],
        window_size=WINDOW_SIZE,
        trial_length=TRIAL_LENGTH,
        id_hidden_dim=ID_HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_value=PAD_VALUE,
        freeze_decoder=False,
        loss_mode="task_only",
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=BEHAVIOR_SCALING_FACTOR,
        **checkpoint_architecture_kwargs(ckpt),
        compile=False,
    )
    model.setup("fit")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    model.eval()

    out_dir = Path(__file__).resolve().parent / "decoded"
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "created_at": datetime.now().astimezone().isoformat(),
        "tag": args.tag,
        "ckpt_dir": str(ckpt_dir),
        "ckpt": str(ckpt_path),
        "side_feature_group": run_metadata["side_features"]["group"],
        "zero_identity": args.zero_identity,
        "raw_normalized": args.raw_normalized,
        "pool_size": pool_size,
        "behavior_mean": behavior_mean.tolist(),
        "behavior_std": behavior_std.tolist(),
        "val_sessions": val_names,
        "forbidden_test_sessions_opened": [],
        "target_session_kinematics_read": "none",
        "sessions": {},
    }

    for nwb_path in val_files:
        rec = load_spikes_and_trials(nwb_path, pool_size)
        stream_note = assert_no_kinematic_read(nwb_path)
        if side_feature_config is not None and not args.zero_identity:
            (group, waveform_group, side_pool, permutation_seed, side_mean, side_std) = side_feature_config
            rec = attach_side_features(
                rec,
                nwb_path,
                side_feature_group=group,
                waveform_feature_group=waveform_group,
                pool_size=side_pool,
                permutation_seed=permutation_seed,
                mean=side_mean,
                std=side_std,
                cache_dir=cache_dir,
            )
        name = rec["name"]
        eval_trials = rec["trials"][pool_size:]
        per_trial = decode_session(
            model, rec, eval_trials, device, args.zero_identity, args.batch_size
        )

        payload: dict[str, np.ndarray] = {}
        dirs, ids, keys = [], [], []
        for trial in eval_trials:
            index = trial["trial_index"]
            vel = per_trial[index]
            if not args.raw_normalized:
                vel = vel * behavior_std[None, :] + behavior_mean[None, :]
            payload[f"vel_{index}"] = vel.astype(np.float32)
            keys.append(index)
            dirs.append(np.nan if trial["target_dir"] is None else trial["target_dir"])
            ids.append(-1 if trial["target_id"] is None else trial["target_id"])
        np.savez_compressed(
            out_dir / f"{name}__{args.tag}.npz",
            trial_indices=np.asarray(keys, dtype=np.int64),
            true_dir=np.asarray(dirs, dtype=np.float64),
            true_id=np.asarray(ids, dtype=np.int64),
            **payload,
        )
        receipt["sessions"][name] = {
            "n_units": rec["n_units"],
            "n_usable_trials": len(rec["trials"]),
            "n_calibration_pool_trials": pool_size,
            "n_eval_trials": len(eval_trials),
            "n_eval_windows": int(sum(v.shape[0] for v in per_trial.values())),
            "n_eval_trials_with_finite_target_dir": int(np.isfinite(dirs).sum()),
            "forbidden_streams": stream_note,
        }
        print(
            f"[{name}] units={rec['n_units']} eval_trials={len(eval_trials)} "
            f"windows={receipt['sessions'][name]['n_eval_windows']}"
        )

    receipt_path = out_dir / f"decode_receipt_{args.tag}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"Saved {receipt_path}")


if __name__ == "__main__":
    main()
