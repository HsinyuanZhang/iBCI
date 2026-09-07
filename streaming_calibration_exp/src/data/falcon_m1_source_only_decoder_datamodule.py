"""Strict source-only native-M1 base-decoder data path for approved folds.

This is intentionally separate from ``FalconDataModule.setup``.  The latter
discovers minival files as part of its normal lifecycle, while the AFC4 base
decoder must not open the outer-left-out session, minival, held-out, formal, or
EvalAI data at all.  This module permits only the three exact fold-specific M1
``held-in-calib`` files and exposes training batches only; checkpoint selection
therefore cannot inspect a target-session metric.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from falcon_challenge.config import FalconConfig, FalconTask
from torch.utils.data import DataLoader

from src.data.falcon_datamodule import FalconDataModule, FalconDataset, SessionBatchSampler


M1_SOURCE_ONLY_FOLDS = {
    0: ("ses-20120924", ("ses-20120926", "ses-20120927", "ses-20120928")),
    1: ("ses-20120926", ("ses-20120924", "ses-20120927", "ses-20120928")),
    2: ("ses-20120927", ("ses-20120924", "ses-20120926", "ses-20120928")),
}
FOLD0_TARGET, FOLD0_SOURCES = M1_SOURCE_ONLY_FOLDS[0]
FOLD1_TARGET, FOLD1_SOURCES = M1_SOURCE_ONLY_FOLDS[1]
FOLD2_TARGET, FOLD2_SOURCES = M1_SOURCE_ONLY_FOLDS[2]
_FORBIDDEN_TOKENS = ("minival", "held-out", "formal", "evalai", "test")


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


class M1SourceOnlyDecoderFold0DataModule(FalconDataModule):
    """No-target/no-minival input for an explicitly allow-listed M1 fold.

    The legacy class name is retained to preserve the fold-0 config target.
    Unknown folds fail closed; this module never infers a source set.
    """

    def __init__(self, *args: Any, source_session_names: list[str], **kwargs: Any) -> None:
        supplied_fold = kwargs.get("loso_fold", 0)
        fold = 0 if supplied_fold is None else int(supplied_fold)
        if fold not in M1_SOURCE_ONLY_FOLDS:
            raise ValueError(f"source decoder has no approved source-only policy for fold {fold}")
        target, expected_sources = M1_SOURCE_ONLY_FOLDS[fold]
        supplied = tuple(source_session_names)
        if supplied != expected_sources:
            raise ValueError(f"fold-{fold} source decoder requires exactly {expected_sources}, got {supplied}")
        if str(kwargs.get("task", "")).lower() != "m1":
            raise ValueError("source decoder is native M1 only")
        if bool(kwargs.get("random_calibration", False)):
            raise ValueError("source decoder requires deterministic chronological M10")
        if bool(kwargs.get("include_heldout_in_fit", False)) or bool(kwargs.get("include_heldout_in_test", False)):
            raise ValueError("source decoder forbids held-out data")
        if str(kwargs.get("side_feature_group", "none")).lower() != "none":
            raise ValueError("base decoder has no AFC4 side input")
        kwargs["loso_fold"] = fold
        kwargs["heldin_session_names"] = list(expected_sources)
        super().__init__(*args, **kwargs)
        self.outer_fold = fold
        self.outer_left_out = target
        self.source_session_names = expected_sources

    def _source_paths(self) -> OrderedDict[str, Path]:
        root = Path(self.hparams.data_dir).resolve()
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
            if any(token in str(resolved).lower() for token in _FORBIDDEN_TOKENS):
                raise ValueError(f"forbidden source path: {resolved}")
            paths[_session_name(resolved)] = resolved
        selected = OrderedDict((name, paths[name]) for name in self.source_session_names if name in paths)
        if tuple(selected) != self.source_session_names:
            raise ValueError(f"exact fold-{self.outer_fold} source files missing: found {tuple(selected)}")
        if self.outer_left_out in selected:
            raise RuntimeError(f"outer fold-{self.outer_fold} target entered base decoder source paths")
        return selected

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit"):
            raise ValueError("source decoder supports fit only; no validation/test/formal lifecycle")
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
        self.train_session_names = list(self.source_session_names)
        self.val_heldin_session_names = []
        self.val_heldout_session_names = []
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
        # Deliberately no held-in target/minival or held-out dataset/sampler.
        self.val_heldin_dataset = None
        self.val_heldout_dataset = None

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        raise RuntimeError("source decoder forbids test/formal evaluation")

    def get_split_manifest(self) -> dict[str, Any]:
        paths = getattr(self, "source_paths", OrderedDict())
        return {
            "schema": f"m1_afc4_source_only_decoder_fold{self.outer_fold}_v1",
            "task": "m1",
            "outer_fold": self.outer_fold,
            "outer_left_out": self.outer_left_out,
            "train_sessions": list(self.source_session_names),
            "validation_sessions": [],
            "source_files": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
            },
            "source_only": True,
            "minival_opened": False,
            "heldout_opened": False,
            "formal": False,
            "evalai": False,
            "checkpoint_selection": "fixed_epoch_train_loss_only",
            "target_backpropagation": False,
        }

    def write_source_only_manifest(self, output_dir: str | Path) -> Path:
        """Persist a source-only receipt beside the fixed-epoch checkpoint.

        Lightning does not dispatch an ``on_fit_start`` hook on data modules,
        so the training entrypoint calls this explicitly after ``setup('fit')``
        has constructed the permitted source sessions.  Keeping the writer on
        the data module ensures that the receipt is derived from the exact
        paths the loader used, rather than from separately duplicated config.
        """
        manifest = self.get_split_manifest()
        output = Path(output_dir) / "source_only_decoder_manifest.json"
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if output.exists() and output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite incompatible source-only manifest: {output}")
        output.write_text(encoded, encoding="utf-8")
        (output.with_suffix(".sha256")).write_text(f"{_sha256(output)}  {output.name}\n", encoding="utf-8")
        return output
