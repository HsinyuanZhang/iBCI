"""Fail-closed all-source native-M1 training data modules.

The ordinary FALCON data module discovers minival and held-out files as part of
``setup``.  That lifecycle is useful for development, but it is not an
appropriate input contract for the final M1 teacher or the Full/B4 AFC4
students: those runs must fit all four held-in calibration sessions, use only
the first ten support trials for identity construction, and have no validation
or query loader at all.  This module is deliberately separate from the
LOSO/source-only modules so a final run cannot accidentally alter an active
LOSO process or discover a new file through a broad ``rglob``.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from falcon_challenge.config import FalconConfig, FalconTask

from src.data.falcon_datamodule import FalconDataModule, FalconDataset, SessionBatchSampler


M1_ALL_SOURCE_SESSIONS: tuple[str, ...] = (
    "ses-20120924",
    "ses-20120926",
    "ses-20120927",
    "ses-20120928",
)
_FORBIDDEN_TOKENS = ("minival", "held-out", "heldout", "formal", "evalai", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_name(path: Path) -> str:
    try:
        return f"ses-{path.name.split('_ses-')[1].split('_behavior')[0]}"
    except IndexError as exc:
        raise ValueError(f"cannot parse M1 session name from {path}") from exc


class M1AllSourceDataModule(FalconDataModule):
    """Train-only native-M1 module over the exact four held-in sessions.

    ``val_dataloader`` is intentionally empty and ``test_dataloader`` raises.
    A caller that asks Lightning to score a target therefore fails closed rather
    than silently opening a minival, held-out, or formal query file.
    """

    def __init__(
        self,
        *args: Any,
        source_session_names: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        supplied = tuple(M1_ALL_SOURCE_SESSIONS if source_session_names is None else source_session_names)
        if supplied != M1_ALL_SOURCE_SESSIONS:
            raise ValueError(
                "all-source M1 training requires exactly the canonical held-in sessions "
                f"{M1_ALL_SOURCE_SESSIONS}, got {supplied}"
            )
        if str(kwargs.get("task", "")).lower() != "m1":
            raise ValueError("all-source module is native M1 only")
        if str(kwargs.get("validation_protocol", "all_source")).lower() != "all_source":
            raise ValueError("all-source M1 requires validation_protocol=all_source")
        if int(kwargs.get("calibration_n_trials", 10)) != 10:
            raise ValueError("all-source M1 requires chronological M10 support")
        if bool(kwargs.get("random_calibration", False)):
            raise ValueError("all-source M1 requires random_calibration=false")
        if bool(kwargs.get("include_heldout_in_fit", False)) or bool(kwargs.get("include_heldout_in_test", False)):
            raise ValueError("all-source M1 forbids held-out/formal data")
        if int(kwargs.get("query_start_trial", 0)) != 0:
            raise ValueError("all-source M1 never consumes a query prefix")
        if int(kwargs.get("heldin_query_start_trial", 0)) != 0 or kwargs.get("heldin_query_end_trial") is not None:
            raise ValueError("all-source M1 has no held-in validation/query loader")
        supplied_group = str(kwargs.pop("side_feature_group", "none")).lower()
        if supplied_group != "none":
            raise ValueError("the base all-source module has no side-feature adapter")
        # ``validation_protocol`` is retained in hparams for receipts, but no
        # inherited split resolver is called by this module.
        kwargs["heldin_session_names"] = list(M1_ALL_SOURCE_SESSIONS)
        super().__init__(*args, side_feature_group="none", **kwargs)
        self.source_session_names = M1_ALL_SOURCE_SESSIONS
        self.train_session_names = list(M1_ALL_SOURCE_SESSIONS)
        self.val_heldin_session_names: list[str] = []
        self.val_heldout_session_names: list[str] = []

    def _source_paths(self) -> OrderedDict[str, Path]:
        root = Path(self.hparams.data_dir).resolve()
        if not (
            root.name == "000941"
            and root.parent.name == "data"
            and root.parent.parent.name == "SPINT-main"
        ):
            raise ValueError(f"all-source M1 requires the canonical SPINT-main/data/000941 root, got {root}")
        directory = (root / "sub-MonkeyL-held-in-calib").resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths: dict[str, Path] = {}
        for path in sorted(directory.glob("*.nwb")):
            resolved = path.resolve()
            try:
                resolved.relative_to(directory)
            except ValueError as exc:
                raise ValueError(f"source symlink escapes held-in-calib directory: {path}") from exc
            text = str(resolved).lower()
            if any(token in text for token in _FORBIDDEN_TOKENS):
                raise ValueError(f"forbidden source path: {resolved}")
            session = _session_name(resolved)
            if session in paths:
                raise ValueError(f"duplicate source session {session}: {resolved}")
            paths[session] = resolved
        if tuple(sorted(paths)) != M1_ALL_SOURCE_SESSIONS:
            raise ValueError(
                "all-source M1 requires exactly four held-in files; "
                f"found {tuple(sorted(paths))}"
            )
        return OrderedDict((name, paths[name]) for name in M1_ALL_SOURCE_SESSIONS)

    def _assert_fit_stage(self, stage: Optional[str]) -> None:
        if stage not in (None, "fit"):
            raise ValueError("all-source M1 module supports fit only")
        h = self.hparams
        if str(h.task).lower() != "m1" or int(h.calibration_n_trials) != 10:
            raise ValueError("all-source M1 contract drifted from native M10")
        if str(h.validation_protocol).lower() != "all_source":
            raise ValueError("all-source M1 requires validation_protocol=all_source")
        if bool(h.random_calibration) or bool(h.include_heldout_in_fit) or bool(h.include_heldout_in_test):
            raise ValueError("all-source M1 contract forbids random/held-out inputs")
        if int(h.query_start_trial) != 0 or int(h.heldin_query_start_trial) != 0 or h.heldin_query_end_trial is not None:
            raise ValueError("all-source M1 contract has no query windows")

    def setup(self, stage: Optional[str] = None) -> None:
        self._assert_fit_stage(stage)
        if getattr(self, "train_dataset", None) is not None:
            return
        paths = self._source_paths()
        task = FalconConfig(task=FalconTask.m1).task
        sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        covariates_mean = covariates_std = None
        for index, (name, path) in enumerate(paths.items()):
            record = self.prepare_session_data(
                path,
                task,
                standardize_covariates=bool(self.hparams.standardize_covariates),
                covariates_mean=covariates_mean,
                covariates_std=covariates_std,
                use_intertrials=bool(self.hparams.use_intertrials),
            )
            if index == 0:
                covariates_mean, covariates_std = record["covariates_mean"], record["covariates_std"]
            sessions[name] = record
        self.source_paths = paths
        self.train_calib_heldin_sessions = sessions
        self.train_dataset = FalconDataset(
            sessions_dict=sessions,
            calib_sessions_dict=sessions,
            window_size=self.hparams.window_size,
            split="train",
            calibration_n_trials=self.hparams.calibration_n_trials,
            random_calibration=False,
            smooth_calibration=self.hparams.smooth_calibration,
            max_trial_length=self.hparams.max_trial_length,
            use_calib_intertrials=self.hparams.use_calib_intertrials,
            trial_feature_type=self.hparams.trial_feature_type,
            remove_still_times=self.hparams.remove_still_times,
            remove_calib_still_times=self.hparams.remove_calib_still_times,
            use_calib_active_segments=self.hparams.use_calib_active_segments,
            calib_n_active_segments=self.hparams.calib_n_active_segments,
            interpolate_trials=self.hparams.interpolate_trials,
            interpolate_trials_kind=self.hparams.interpolate_trials_kind,
            pad_value=self.hparams.pad_value,
            side_feature_group="none",
            query_start_trial=0,
        )
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset,
            self.batch_size_per_device,
            shuffle=True,
            seed=self.hparams.sampler_seed,
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
        )
        self.val_heldin_dataset = None
        self.val_heldout_dataset = None

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        raise RuntimeError("all-source M1 training forbids local validation/test evaluation")

    def get_split_manifest(self) -> dict[str, Any]:
        paths = getattr(self, "source_paths", OrderedDict())
        return {
            "schema": "m1_all_source_train_only_v1",
            "task": "m1",
            "validation_protocol": "all_source",
            "fold_id": None,
            "train_sessions": list(M1_ALL_SOURCE_SESSIONS),
            "validation_sessions": [],
            "source_files": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
            },
            "source_only": True,
            "all_source": True,
            "minival_opened": False,
            "heldout_opened": False,
            "formal": False,
            "evalai": False,
            "calibration_n_trials": 10,
            "query_start_trial": 0,
            "checkpoint_selection": "fixed_epoch_train_loss_only",
            "teacher_epochs": 20,
            "target_backpropagation": False,
        }

    def write_source_only_manifest(self, output_dir: str | Path) -> Path:
        """Write the all-source receipt using the train.py source hook name."""
        manifest = self.get_split_manifest()
        output = Path(output_dir) / "all_source_train_manifest.json"
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if output.exists() and output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite incompatible all-source manifest: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        (output.with_suffix(".sha256")).write_text(
            f"{_sha256(output)}  {output.name}\n", encoding="utf-8"
        )
        return output


__all__ = ["M1_ALL_SOURCE_SESSIONS", "M1AllSourceDataModule"]
