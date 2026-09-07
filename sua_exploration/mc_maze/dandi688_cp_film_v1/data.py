"""Source-only DANDI data and profile materialization."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import core, plan


@dataclass(frozen=True)
class SessionMaterial:
    session: str
    split: str
    record: object
    q50_starts: np.ndarray
    profile10: np.ndarray
    profile30: np.ndarray
    evidence10: dict[str, object]
    evidence30: dict[str, object]


def prepare_datamodule(repo_root: Path):
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(repo_root / plan.DATA_RELATIVE),
        task="CO",
        split_counts=(plan.TRAIN_SESSIONS, plan.VALIDATION_SESSIONS, plan.FORMAL_TEST_SESSIONS),
        batch_size=plan.BATCH_SIZE,
        window_size=plan.WINDOW,
        calibration_n_trials=plan.ACTIVITY_SUPPORT,
        max_trial_length=plan.TRIAL_LENGTH,
        bin_size_ms=plan.BIN_MS,
        num_workers=0,
        pin_memory=False,
        random_calibration=False,
        seed=42,
        max_units_exclusive=100,
        cache_dir=str(repo_root / plan.CACHE_RELATIVE),
        signal_view="sua",
        side_feature_group="t4",
        side_feature_pool_size=plan.T4_SUPPORT,
        train_val_manifest_path=str(repo_root / plan.MANIFEST_RELATIVE),
    )
    dm.setup("fit")
    core.require(dm.test_dataset is None, "formal-test dataset was constructed")
    core.require(
        len(dm.session_splits.get("train", [])) == plan.TRAIN_SESSIONS
        and len(dm.session_splits.get("val", [])) == plan.VALIDATION_SESSIONS
        and len(dm.session_splits.get("test", [])) == plan.FORMAL_TEST_SESSIONS,
        "strict roster drift",
    )
    return dm


def _profile_for_horizon(record, trials, horizon: int):
    selected = trials[:horizon]
    bins = np.concatenate(
        [np.arange(int(row["start"]), int(row["stop"]), dtype=np.int64) for row in selected]
    )
    profile, evidence = core.profile_from_bins(record.neural[bins], record.behavior[bins])
    return profile, {
        **{key: value for key, value in evidence.items() if key != "raw_profile"},
        "horizon": int(horizon),
        "trial_indices": [int(row["trial_index"]) for row in selected],
        "bin_indices_sha256": core.array_sha256(bins),
        "raw_profile_sha256": core.array_sha256(evidence["raw_profile"]),
        "masked_profile_sha256": core.array_sha256(profile),
    }


def materialize(repo_root: Path, dm) -> dict[str, SessionMaterial]:
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials

    out = {}
    for split, dataset in (("train", dm.train_dataset), ("val", dm.val_dataset)):
        core.require(dataset is not None, f"missing {split} dataset")
        for session, record in dataset.sessions.items():
            path = repo_root / plan.DATA_RELATIVE / f"{session}_behavior+ecephys.nwb"
            trials = list_datamodule_rewarded_trials(
                path,
                bin_size_ms=plan.BIN_MS,
                window_size=plan.WINDOW,
                trial_result_filter="R",
            )
            core.require(len(trials) > plan.QUERY_START, f"{session}: insufficient Q50 trials")
            p10, e10 = _profile_for_horizon(record, trials, 10)
            p30, e30 = _profile_for_horizon(record, trials, 30)
            q50 = np.concatenate(
                [
                    np.arange(int(row["start"]), int(row["stop"]) - plan.WINDOW + 1, dtype=np.int64)
                    for row in trials[plan.QUERY_START :]
                ]
            )
            available = {(session, int(start)) for start in dataset.sessions[session].valid_starts}
            if split == "train":
                # Train records expose all windows; validation records expose Q30.
                available = set(dataset.window_indices)
            core.require(all((session, int(start)) in available for start in q50), f"{session}: Q50 not in dataset")
            out[session] = SessionMaterial(session, split, record, np.ascontiguousarray(q50), p10, p30, e10, e30)
    core.require(sum(row.split == "train" for row in out.values()) == plan.TRAIN_SESSIONS, "train material count drift")
    core.require(sum(row.split == "val" for row in out.values()) == plan.VALIDATION_SESSIONS, "val material count drift")
    return out


def shuffled_profile(profile: np.ndarray, session: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    payload = f"{plan.PROFILE_SHUFFLE_DOMAIN}:{seed}:{session}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    permutation = np.random.Generator(np.random.PCG64(derived)).permutation(profile.shape[0])
    if profile.shape[0] > 1 and np.array_equal(permutation, np.arange(profile.shape[0])):
        permutation = np.roll(permutation, 1)
    return np.ascontiguousarray(profile[permutation]), np.ascontiguousarray(permutation, dtype=np.int64)


def epoch_starts(material: SessionMaterial, seed: int, epoch: int) -> np.ndarray:
    values = material.q50_starts
    count = min(values.size, plan.WINDOWS_PER_SESSION_PER_EPOCH)
    payload = f"{plan.TRAIN_SAMPLE_DOMAIN}:{seed}:{epoch}:{material.session}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(derived))
    picked = rng.choice(values.size, size=count, replace=False)
    return np.ascontiguousarray(np.sort(values[picked]), dtype=np.int64)

