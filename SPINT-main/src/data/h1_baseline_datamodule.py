"""Isolated H1 baseline data path for reproducing the released SPINT code.

The upstream ``FalconDataModule`` is intentionally broad: its normal setup
also enumerates the formal held-out calibration directory and returns a
two-loader validation list.  That is useful for the benchmark submission,
but it is not an acceptable preparation path for this baseline reproduction.
This module keeps the original ``FalconDataset``/``SessionBatchSampler``
semantics (including random contiguous calibration blocks) while explicitly
opening only the 13 H1 held-in calibration and held-in-minival files.

The path is H1-only and does not change the shared RT/M1 data module.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import lightning.pytorch as pl
import numpy as np
from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb
from torch.utils.data import DataLoader

from src.data.falcon_datamodule import FalconDataset, SessionBatchSampler


H1_BASELINE_HELDIN_SESSIONS: tuple[str, ...] = (
    "ses-19250101T111740",
    "ses-19250101T112404",
    "ses-19250108T110520",
    "ses-19250108T111022",
    "ses-19250108T111455",
    "ses-19250113T120811",
    "ses-19250113T121303",
    "ses-19250115T110633",
    "ses-19250115T111328",
    "ses-19250119T113543",
    "ses-19250119T114045",
    "ses-19250120T115044",
    "ses-19250120T115537",
)

H1_BASELINE_HELDOUT_SESSIONS: tuple[str, ...] = (
    "ses-19250126T113454",
    "ses-19250126T114029",
    "ses-19250127T120333",
    "ses-19250127T120826",
    "ses-19250129T112555",
    "ses-19250129T113059",
    "ses-19250202T113958",
    "ses-19250202T114452",
    "ses-19250203T113515",
    "ses-19250203T114018",
    "ses-19250206T112219",
    "ses-19250206T112712",
    "ses-19250209T111826",
    "ses-19250209T112327",
)

H1_BASELINE_PROTOCOL = "spint_h1_baseline_reproduction_v1"


def h1_session_date(session_name: str) -> str:
    """Return the YYYYMMDD grouping key used for H1 date audits."""

    token = str(session_name)
    if not token.startswith("ses-") or len(token) < 12 or token[12] != "T":
        raise ValueError(f"not an H1 session name: {session_name!r}")
    return token[4:12]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session_from_path(path: Path) -> str:
    stem = path.stem
    marker = "_ses-"
    if marker not in stem:
        raise ValueError(f"cannot parse H1 session from {path}")
    return stem[stem.index(marker) + 1 :]


class H1BaselineDataModule(pl.LightningDataModule):
    """Released-SPINT H1 data semantics with a held-in-only safety boundary."""

    def __init__(
        self,
        task: str,
        data_dir: str,
        heldin_session_names: list[str] | tuple[str, ...] | None = None,
        batch_size: int = 32,
        window_size: int = 700,
        calibration_n_trials: float = 2,
        random_calibration: bool = True,
        smooth_calibration: bool = False,
        max_trial_length: int = 1024,
        standardize_covariates: bool = False,
        use_intertrials: bool = True,
        use_calib_intertrials: bool = False,
        trial_feature_type: str = "raw",
        remove_still_times: bool = False,
        remove_calib_still_times: bool = False,
        use_calib_active_segments: bool = False,
        calib_n_active_segments: int = 1,
        interpolate_trials: bool = True,
        interpolate_trials_kind: str = "cubic",
        pad_value: float = -1.0,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        if str(task).lower() != "h1":
            raise ValueError("H1 baseline path accepts task='h1' only")
        sessions = tuple(heldin_session_names or H1_BASELINE_HELDIN_SESSIONS)
        if sessions != H1_BASELINE_HELDIN_SESSIONS:
            raise ValueError(
                "H1 baseline requires the explicit 13-session held-in set; "
                f"got {sessions!r}"
            )
        if int(calibration_n_trials) != 2 or float(calibration_n_trials) != 2.0:
            raise ValueError("H1 baseline protocol fixes calibration_n_trials=2")
        if not bool(random_calibration):
            raise ValueError("H1 baseline protocol fixes random_calibration=true for training")
        if int(window_size) != 700 or int(max_trial_length) != 1024:
            raise ValueError("H1 baseline protocol fixes window_size=700 and max_trial_length=1024")
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = int(batch_size)
        self._fit_loaded_paths: list[str] = []

    @staticmethod
    def _enumerate(root: Path, directory_name: str, expected: tuple[str, ...]) -> list[Path]:
        directory = (root / directory_name).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"H1 baseline missing held-in directory: {directory}")
        paths = sorted(directory.glob("*.nwb"))
        by_session: dict[str, Path] = {}
        for path in paths:
            resolved = path.resolve()
            try:
                resolved.relative_to(directory)
            except ValueError as error:
                raise ValueError(f"H1 baseline rejects symlink escape: {path}") from error
            session = _session_from_path(resolved)
            if session in by_session:
                raise ValueError(f"duplicate H1 baseline session {session} in {directory}")
            by_session[session] = resolved
        if set(by_session) != set(expected) or len(by_session) != len(expected):
            raise ValueError(
                f"H1 baseline {directory_name} session set mismatch: "
                f"observed={sorted(by_session)} expected={sorted(expected)}"
            )
        return [by_session[session] for session in expected]

    @staticmethod
    def _load(path: Path) -> dict[str, np.ndarray]:
        neural, covariates, trial_change, eval_mask = load_nwb(path, FalconTask.h1)
        arrays = {
            "neural": np.asarray(neural, dtype=np.float32),
            "covariates": np.asarray(covariates, dtype=np.float32),
            "trial_change": np.asarray(trial_change, dtype=bool),
            "eval_mask": np.asarray(eval_mask, dtype=bool),
        }
        if arrays["neural"].ndim != 2 or arrays["neural"].shape[1] != 176:
            raise ValueError(f"H1 baseline expected [T,176] neural array from {path}")
        if arrays["covariates"].ndim != 2 or arrays["covariates"].shape[1] != 7:
            raise ValueError(f"H1 baseline expected [T,7] velocity array from {path}")
        if any(np.isnan(value).any() for value in arrays.values()):
            raise ValueError(f"H1 baseline found NaN in {path}")
        lengths = {value.shape[0] for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"H1 baseline array length mismatch in {path}: {lengths}")
        if not np.any(arrays["trial_change"]):
            raise ValueError(f"H1 baseline found no TrialNum changes in {path}")
        return arrays

    @staticmethod
    def _prepare(paths: list[Path]) -> OrderedDict[str, dict[str, np.ndarray]]:
        output: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        for path in paths:
            output[_session_from_path(path)] = H1BaselineDataModule._load(path)
        return output

    def setup(self, stage: str | None = None) -> None:
        if stage in {"test", "predict"}:
            raise RuntimeError("H1 baseline path forbids test/predict and formal held-out access")
        if stage not in {None, "fit", "validate"}:
            raise ValueError(f"unsupported H1 baseline setup stage {stage!r}")
        root = Path(self.hparams.data_dir).resolve()
        calib_paths = self._enumerate(
            root, "sub-HumanPitt-held-in-calib", H1_BASELINE_HELDIN_SESSIONS
        )
        minival_paths = self._enumerate(
            root, "sub-HumanPitt-held-in-minival", H1_BASELINE_HELDIN_SESSIONS
        )
        self._calib_paths = calib_paths
        self._minival_paths = minival_paths
        self._fit_loaded_paths = [str(path) for path in (*calib_paths, *minival_paths)]
        train_sessions = self._prepare(calib_paths)
        val_sessions = self._prepare(minival_paths)
        common = dict(
            window_size=int(self.hparams.window_size),
            calibration_n_trials=int(self.hparams.calibration_n_trials),
            smooth_calibration=bool(self.hparams.smooth_calibration),
            max_trial_length=int(self.hparams.max_trial_length),
            use_calib_intertrials=bool(self.hparams.use_calib_intertrials),
            trial_feature_type=str(self.hparams.trial_feature_type),
            remove_still_times=bool(self.hparams.remove_still_times),
            remove_calib_still_times=bool(self.hparams.remove_calib_still_times),
            use_calib_active_segments=bool(self.hparams.use_calib_active_segments),
            calib_n_active_segments=int(self.hparams.calib_n_active_segments),
            interpolate_trials=bool(self.hparams.interpolate_trials),
            interpolate_trials_kind=str(self.hparams.interpolate_trials_kind),
            pad_value=float(self.hparams.pad_value),
        )
        self.train_dataset = FalconDataset(
            sessions_dict=train_sessions,
            calib_sessions_dict=train_sessions,
            split="train",
            random_calibration=True,
            **common,
        )
        self.val_heldin_dataset = FalconDataset(
            sessions_dict=val_sessions,
            calib_sessions_dict=train_sessions,
            split="val_heldin",
            random_calibration=False,
            **common,
        )
        self.val_heldout_dataset = None
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset, self.batch_size_per_device, shuffle=True
        )
        self.val_heldin_batch_sampler = SessionBatchSampler(
            self.val_heldin_dataset, self.batch_size_per_device, shuffle=False
        )
        self.val_heldout_batch_sampler = None

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_heldin_dataset,
            batch_sampler=self.val_heldin_batch_sampler,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        raise RuntimeError("H1 baseline path forbids test_dataloader and formal held-out access")

    def input_manifest(self) -> dict[str, Any]:
        if not hasattr(self, "_calib_paths"):
            raise RuntimeError("call setup before requesting the H1 baseline input manifest")
        rows = []
        for role, paths in (("heldin_calib", self._calib_paths), ("heldin_minival", self._minival_paths)):
            for path in paths:
                session = _session_from_path(path)
                rows.append(
                    {
                        "role": role,
                        "session": session,
                        "date_group": h1_session_date(session),
                        "path": str(path),
                        "sha256": _sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        return {
            "schema": "spint_h1_baseline_input_manifest_v1",
            "protocol": H1_BASELINE_PROTOCOL,
            "task": "h1",
            "fit_input_scope": "all_13_heldin_calib_recordings",
            "validation_only_scope": "all_13_matching_heldin_minival_recordings",
            "discrete_direction_labels_available": False,
            "dense_velocity_covariates_available": True,
            "spint_calibration_features": "neural_only_trialized_features",
            "afc4_kinematic_comparator": "dense_7d_velocity_labels_are_a_separate_carrier_input",
            "formal_heldout_opened": False,
            "heldin_sessions": list(H1_BASELINE_HELDIN_SESSIONS),
            "heldout_sessions_forbidden": list(H1_BASELINE_HELDOUT_SESSIONS),
            "generalization_grouping_note": (
                "This reproduction uses all 13 held-in recordings. Any later cross-recording "
                "generalization analysis should group by the six held-in recording dates "
                "before considering a session-level split."
            ),
            "files": rows,
        }
