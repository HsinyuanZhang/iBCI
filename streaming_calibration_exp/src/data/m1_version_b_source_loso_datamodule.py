"""Isolated M1 Version-B source-LOSO data path.

This module is intentionally independent of ``FalconDataModule.setup``.  The
legacy M1 AFC4 module calls the broad setup first, which discovers public
``held-in-minival`` files before replacing the validation dataset.  Version-B
must not even enumerate those files.  The only files resolved here are the
four explicitly allow-listed native ``held-in-calib`` NWBs.

The train split contains the three source sessions for one approved M1 LOSO
fold.  The validation/test split is the left-out *held-in-calib* session,
strictly after its first ten support trials.  It is a development endpoint;
no held-out, minival, formal, or EvalAI file is permitted.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.data.falcon_datamodule import FalconDataModule, FalconDataset, SessionBatchSampler
from src.data.falcon_emg_afc4_features import SourceFrozenEMGAFC4Plan


M1_VERSION_B_FOLDS: dict[int, tuple[str, tuple[str, str, str]]] = {
    0: ("ses-20120924", ("ses-20120926", "ses-20120927", "ses-20120928")),
    1: ("ses-20120926", ("ses-20120924", "ses-20120927", "ses-20120928")),
    2: ("ses-20120927", ("ses-20120924", "ses-20120926", "ses-20120928")),
}

_FORBIDDEN_PATH_TOKENS = ("minival", "held-out", "heldout", "formal", "evalai", "test")
_ARMS = {"none", "zero4", "full", "b4", "rs4", "ls4"}


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


class M1VersionBDataset(FalconDataset):
    """Falcon windows with an optional fixed-width Version-B side matrix."""

    def __init__(self, *args: Any, carrier_plan: SourceFrozenEMGAFC4Plan | None,
                 carrier_arm: str, **kwargs: Any) -> None:
        if carrier_arm not in _ARMS:
            raise ValueError(f"unknown Version-B carrier arm {carrier_arm!r}")
        if carrier_arm not in {"none", "zero4"} and carrier_plan is None:
            raise ValueError(f"{carrier_arm} Version-B arm requires a source-frozen AFC4 plan")
        # The wrapper owns the side-feature return contract.  Ignore any
        # inherited caller value rather than allowing a duplicate keyword to
        # silently select a legacy feature path.
        kwargs.pop("side_feature_group", None)
        super().__init__(*args, side_feature_group="none", **kwargs)
        self.carrier_plan = carrier_plan
        self.carrier_arm = carrier_arm

    def __getitem__(self, index: int):
        neural, target, calibration, session_name = super().__getitem__(index)
        if self.carrier_arm == "none":
            return neural, target, calibration, session_name
        num_neurons = int(calibration.shape[-1])
        if self.carrier_arm == "zero4":
            side = np.zeros((num_neurons, 4), dtype=np.float32)
        else:
            assert self.carrier_plan is not None
            side = self.carrier_plan.normalized(str(session_name), arm=self.carrier_arm)
            if side.shape != (num_neurons, 4):
                raise RuntimeError(
                    f"Version-B AFC4 shape mismatch for {session_name}: "
                    f"side={side.shape}, calibration={tuple(calibration.shape)}"
                )
        return neural, target, calibration, session_name, side


class M1VersionBSourceLOSODataModule(FalconDataModule):
    """No-minival source-LOSO data module for the Version-B pilot.

    The inherited ``__init__`` is used only for hyperparameter storage and
    utility methods such as ``prepare_session_data``.  ``setup`` is fully
    replaced and never calls the inherited broad file-discovery path.
    """

    def __init__(self, *args: Any, source_session_names: list[str],
                 afc4_arm: str = "none", **kwargs: Any) -> None:
        fold = int(kwargs.get("loso_fold", 0))
        if fold not in M1_VERSION_B_FOLDS:
            raise ValueError(f"Version-B supports approved M1 folds {sorted(M1_VERSION_B_FOLDS)}")
        target, expected_sources = M1_VERSION_B_FOLDS[fold]
        if tuple(source_session_names) != expected_sources:
            raise ValueError(
                f"fold-{fold} requires source sessions {expected_sources}, got {tuple(source_session_names)}"
            )
        if str(kwargs.get("task", "")).lower() != "m1":
            raise ValueError("Version-B source-LOSO is native M1 only")
        if afc4_arm not in _ARMS:
            raise ValueError(f"afc4_arm must be one of {sorted(_ARMS)}, got {afc4_arm!r}")
        # The parent constructor does not inspect data; it only stores hparams.
        # Keep side_feature_group=none because this module appends its own side
        # matrix and must not activate any legacy FALCON side path.
        kwargs["heldin_session_names"] = list(expected_sources)
        kwargs["side_feature_group"] = "none"
        super().__init__(*args, **kwargs)
        self.outer_fold = fold
        self.outer_left_out = target
        self.source_session_names = expected_sources
        self.afc4_arm = afc4_arm
        self.carrier_plan: SourceFrozenEMGAFC4Plan | None = None
        self._setup_stage: str | None = None

    def _assert_contract(self, stage: Optional[str]) -> None:
        h = self.hparams
        if stage not in (None, "fit", "validate", "test"):
            raise ValueError(f"Version-B does not support stage={stage!r}")
        if str(h.task).lower() != "m1":
            raise ValueError("Version-B task must be m1")
        if str(h.validation_protocol).lower() != "loso":
            raise ValueError("Version-B requires validation_protocol=loso")
        if int(h.loso_fold) != self.outer_fold:
            raise ValueError("Version-B fold/hparams mismatch")
        if int(h.calibration_n_trials) != 10 or bool(h.random_calibration):
            raise ValueError("Version-B requires deterministic chronological M10")
        if bool(h.include_heldout_in_fit) or bool(h.include_heldout_in_test):
            raise ValueError("Version-B forbids held-out/formal data")
        if int(h.query_start_trial) != 0:
            raise ValueError("Version-B target query uses held-in query_start_trial only")
        if int(h.heldin_query_start_trial) != 10 or int(h.heldin_query_end_trial) != 210:
            raise ValueError("Version-B requires the frozen held-in-calib query [10,210)")
        if bool(h.smooth_calibration) or not bool(h.use_intertrials):
            raise ValueError("Version-B requires raw unsmoothed contiguous calibration")
        if str(h.trial_feature_type).lower() != "raw":
            raise ValueError("Version-B requires raw calibration trials")
        if self.afc4_arm not in _ARMS:
            raise ValueError("Version-B carrier arm drifted")

    def _source_paths(self) -> OrderedDict[str, Path]:
        root = Path(self.hparams.data_dir).resolve()
        if root.name != "000941":
            raise ValueError(f"Version-B requires canonical .../data/000941 root, got {root}")
        directory = (root / "sub-MonkeyL-held-in-calib").resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths: dict[str, Path] = {}
        # This is the only directory enumerated by this module.  No rglob or
        # sibling minival/held-out directory is ever touched.
        for path in sorted(directory.glob("*.nwb")):
            resolved = path.resolve()
            try:
                resolved.relative_to(directory)
            except ValueError as exc:
                raise ValueError(f"source symlink escapes held-in-calib: {path}") from exc
            # Check the path *inside* the canonical 000941 root.  Looking at
            # the entire absolute path would reject harmless parent names such
            # as pytest's temporary ``test_*`` directory while providing no
            # additional scope protection.
            text = str(resolved.relative_to(root)).lower()
            if any(token in text for token in _FORBIDDEN_PATH_TOKENS):
                raise ValueError(f"forbidden Version-B source path: {resolved}")
            session = _session_name(resolved)
            if session in paths:
                raise ValueError(f"duplicate Version-B session {session}")
            paths[session] = resolved
        expected = (self.outer_left_out,) + tuple(self.source_session_names)
        if set(paths) != set(expected):
            raise ValueError(f"Version-B requires exactly sessions {sorted(expected)}, found {sorted(paths)}")
        return OrderedDict((name, paths[name]) for name in expected)

    def _dataset(self, sessions: OrderedDict[str, dict[str, Any]],
                 calibration: OrderedDict[str, dict[str, Any]], *, split: str,
                 carrier_plan: SourceFrozenEMGAFC4Plan | None) -> M1VersionBDataset:
        h = self.hparams
        is_query = split != "train"
        return M1VersionBDataset(
            sessions_dict=sessions,
            calib_sessions_dict=calibration,
            window_size=h.window_size,
            split=split,
            calibration_n_trials=h.calibration_n_trials,
            random_calibration=False,
            smooth_calibration=h.smooth_calibration,
            max_trial_length=h.max_trial_length,
            use_calib_intertrials=h.use_calib_intertrials,
            trial_feature_type=h.trial_feature_type,
            remove_still_times=h.remove_still_times,
            remove_calib_still_times=h.remove_calib_still_times,
            use_calib_active_segments=h.use_calib_active_segments,
            calib_n_active_segments=h.calib_n_active_segments,
            interpolate_trials=h.interpolate_trials,
            interpolate_trials_kind=h.interpolate_trials_kind,
            pad_value=h.pad_value,
            query_start_trial=h.heldin_query_start_trial if is_query else 0,
            query_end_trial=h.heldin_query_end_trial if is_query else None,
            allow_empty_query_sessions=False,
            carrier_plan=carrier_plan,
            carrier_arm=self.afc4_arm,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self._assert_contract(stage)
        if getattr(self, "train_dataset", None) is not None:
            return
        paths = self._source_paths()
        task = self._falcon_task_m1()
        source_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        covariates_mean = covariates_std = None
        for index, name in enumerate(self.source_session_names):
            record = self.prepare_session_data(
                paths[name], task,
                standardize_covariates=bool(self.hparams.standardize_covariates),
                covariates_mean=covariates_mean,
                covariates_std=covariates_std,
                use_intertrials=bool(self.hparams.use_intertrials),
            )
            if index == 0:
                covariates_mean = record["covariates_mean"]
                covariates_std = record["covariates_std"]
            source_records[name] = record
        self.source_paths = OrderedDict((name, paths[name]) for name in self.source_session_names)
        self.target_path = paths[self.outer_left_out]
        self.train_session_names = list(self.source_session_names)
        self.val_heldin_session_names = [self.outer_left_out]
        self.val_heldout_session_names = []
        self.train_calib_heldin_sessions = source_records

        # Every compact arm binds one source-only PCA and correctly-paired Full
        # normalizer. Zero4 does not add the target and therefore never reads
        # target EMG or target support spikes for carrier construction. The
        # B4/RS4/LS4 controls fit from the same target M10 support as Full.
        if self.afc4_arm != "none":
            self.carrier_plan = SourceFrozenEMGAFC4Plan(
                self.source_paths, shuffle_seed=int(self.hparams.side_feature_shuffle_seed)
            )
            if self.afc4_arm != "zero4":
                self.carrier_plan.add_target(self.target_path)

        # Only after the carrier state is frozen may the ordinary evaluator
        # materialize the left-out held-in query record. For Full/B4/RS4/LS4,
        # ``add_target`` above has already read exactly target trials [0,10)
        # through the narrow AFC4 loader. No target query value can therefore
        # influence PCA, normalization, carrier fitting, or model selection.
        target_record = self.prepare_session_data(
            paths[self.outer_left_out], task,
            standardize_covariates=bool(self.hparams.standardize_covariates),
            covariates_mean=covariates_mean,
            covariates_std=covariates_std,
            use_intertrials=bool(self.hparams.use_intertrials),
        )
        target_records = OrderedDict([(self.outer_left_out, target_record)])

        self.train_dataset = self._dataset(source_records, source_records, split="train", carrier_plan=self.carrier_plan)
        self.val_heldin_dataset = self._dataset(target_records, target_records, split="val_heldin", carrier_plan=self.carrier_plan)
        audits = self.val_heldin_dataset.query_window_audit
        target_audit = audits.get(self.outer_left_out)
        if target_audit is None or not target_audit["full_window_disjoint"] or target_audit["eligible_windows"] <= 0:
            raise RuntimeError(f"Version-B target query audit invalid: {target_audit}")
        self.val_heldout_dataset = None
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset, self.batch_size_per_device, shuffle=True,
            seed=int(self.hparams.sampler_seed),
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
        )
        self.val_heldin_batch_sampler = SessionBatchSampler(
            self.val_heldin_dataset, self.batch_size_per_device, shuffle=False
        )
        self._setup_stage = str(stage or "fit")

    def _falcon_task_m1(self):
        # Import lazily so importing this module for a CPU contract test does
        # not instantiate or inspect any data file.
        from falcon_challenge.config import FalconConfig, FalconTask

        return FalconConfig(task=FalconTask.m1).task

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset, batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.val_heldin_dataset, batch_sampler=self.val_heldin_batch_sampler,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    @staticmethod
    def _sampler_batch_counts(sampler: SessionBatchSampler) -> dict[str, int]:
        """Return realized per-session batches across supported sampler revisions.

        The remote Version-B workspace has a frozen ``SessionBatchSampler``
        revision that predates ``session_batch_counts``.  Because the frozen
        pilot disables session balancing, ``original_session_batch_counts`` is
        then exactly the realized count.  The structural fallback reads the
        already-built batches and deliberately does not iterate the sampler,
        which could mutate an epoch-aware sampler during terminal export.
        """
        counts = getattr(sampler, "session_batch_counts", None)
        if counts is None:
            if bool(getattr(sampler, "balance_sessions", False)):
                raise RuntimeError(
                    "cannot infer realized Version-B batch counts from an older balanced sampler"
                )
            counts = getattr(sampler, "original_session_batch_counts", None)
        if counts is None:
            batches = getattr(sampler, "batched_indices", None)
            dataset = getattr(sampler, "dataset", None)
            if batches is None or dataset is None or not hasattr(dataset, "window_indices"):
                raise RuntimeError("Version-B sampler exposes no auditable batch-count state")
            reconstructed = {str(name): 0 for name in getattr(sampler, "session_to_indices", {})}
            for batch in batches:
                if not batch:
                    continue
                name = str(dataset.window_indices[batch[0]][0])
                reconstructed[name] = reconstructed.get(name, 0) + 1
            counts = reconstructed
        output = {str(name): int(count) for name, count in dict(counts).items()}
        if not output or any(count <= 0 for count in output.values()):
            raise RuntimeError(f"Version-B sampler batch counts are invalid: {output}")
        return output

    @staticmethod
    def _sampler_sha256(sampler: SessionBatchSampler) -> tuple[str, int]:
        """Hash the frozen batch order without iterating/mutating the sampler."""

        batches = getattr(sampler, "batched_indices", None)
        if batches is None:
            raise RuntimeError("Version-B sampler exposes no frozen batched_indices for hashing")
        digest = hashlib.sha256()
        scored_windows = 0
        for batch_index, batch in enumerate(batches):
            indices = np.asarray(batch, dtype=np.int64).reshape(-1)
            if indices.size == 0:
                raise RuntimeError(f"Version-B sampler contains an empty batch at {batch_index}")
            digest.update(np.asarray([batch_index, indices.size], dtype=np.int64).tobytes())
            digest.update(indices.tobytes())
            scored_windows += int(indices.size)
        if scored_windows <= 0:
            raise RuntimeError("Version-B sampler hash saw no scored windows")
        return digest.hexdigest(), scored_windows

    def get_split_manifest(self) -> dict[str, Any]:
        if not hasattr(self, "source_paths"):
            return {
                "schema": "m1_version_b_source_loso_v1",
                "source_only": True,
                "formal": False,
                "minival_files_opened": False,
                "minival_values_used": False,
            }
        train_sampler_sha, train_scored_windows = self._sampler_sha256(self.train_batch_sampler)
        query_sampler_sha, query_scored_windows = self._sampler_sha256(self.val_heldin_batch_sampler)
        manifest: dict[str, Any] = {
            "schema": "m1_version_b_source_loso_v1",
            "task": "m1",
            "outer_fold": self.outer_fold,
            "outer_left_out": self.outer_left_out,
            "train_sessions": list(self.source_session_names),
            "validation_sessions": [self.outer_left_out],
            "source_only": True,
            "formal": False,
            "evalai": False,
            "heldout_files_opened": False,
            "heldout_values_used": False,
            "minival_files_opened": False,
            "minival_values_used": False,
            # The held-in query is necessarily read by Lightning validation
            # and the final fixed-window report.  It is never used for an
            # optimizer step, early stopping, or checkpoint selection.
            "target_query_values_read_by_validation_or_evaluator": True,
            "target_query_values_used_for_optimizer_or_checkpoint_selection": False,
            "carrier_arm": self.afc4_arm,
            "support_trials": [0, 10],
            "query_trials": [10, 210],
            "calibration_n_trials": 10,
            "checkpoint_selection": "fixed_last_epoch_11_train_source_only",
            "target_backpropagation": False,
            "source_files": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in self.source_paths.items()
            },
            "target_file": {
                "path": str(self.target_path), "sha256": _sha256(self.target_path)
            },
            "query_window_audit": self.val_heldin_dataset.query_window_audit,
            "train_batch_counts": self._sampler_batch_counts(self.train_batch_sampler),
            "train_sampler_sha256": train_sampler_sha,
            "train_scored_windows_per_epoch": train_scored_windows,
            "query_batch_count": len(self.val_heldin_batch_sampler),
            "query_sampler_sha256": query_sampler_sha,
            "query_scored_windows": query_scored_windows,
        }
        if self.carrier_plan is not None:
            receipt = self.carrier_plan.receipt(arm=self.afc4_arm)
            # A Zero4 control binds the source PCA/normalizer, but no target
            # support record is added to its plan. Other controls retain their
            # exact intervention in this arm-specific receipt.
            manifest["carrier_plan_receipt"] = receipt
            manifest["carrier_plan_receipt"]["target_fit_calls"] = dict(self.carrier_plan.target_fit_calls)
        return manifest

    def write_source_only_manifest(self, output_dir: str | Path) -> Path:
        output = Path(output_dir) / "m1_version_b_source_loso_manifest.json"
        encoded = json.dumps(self.get_split_manifest(), indent=2, sort_keys=True) + "\n"
        if output.exists() and output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite incompatible manifest: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        (output.with_suffix(".sha256")).write_text(f"{_sha256(output)}  {output.name}\n", encoding="utf-8")
        return output


class M1VersionBSourceOnlyFitDataModule(M1VersionBSourceLOSODataModule):
    """Strict source-only fit variant used by the fold-1 continuation.

    ``M1VersionBSourceLOSODataModule`` is intentionally also usable by the
    independent evaluator, where ``setup('test')`` materializes the left-out
    held-in query.  That dual-use class must therefore resolve the target when
    its ordinary ``setup('fit')`` path is called.  Training the fold-1 v2
    proposal uses this subclass instead: fit resolves exactly the three source
    filenames, prepares only those records, and exposes no validation/test
    dataset.  The target path is not stored, hashed, or passed to a loader.

    The evaluator switches back to the parent class *after* both terminal
    checkpoints exist.  Keeping that transition explicit prevents
    ``test=false`` or ``limit_val_batches=0`` from being mistaken for a data
    isolation guarantee.
    """

    _SOURCE_ONLY_ARMS = {"none", "zero4"}

    def _approved_source_paths(self) -> OrderedDict[str, Path]:
        """Resolve only allow-listed source filenames; never list target files."""
        root = Path(self.hparams.data_dir).resolve()
        if root.name != "000941":
            raise ValueError(f"Version-B requires canonical .../data/000941 root, got {root}")
        directory = (root / "sub-MonkeyL-held-in-calib").resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths: OrderedDict[str, Path] = OrderedDict()
        for session in self.source_session_names:
            # The native M1 source directory has one canonical filename per
            # session.  Constructing this path avoids glob/rglob enumeration
            # of the outer-left-out file during fit.
            suffix = session.removeprefix("ses-")
            path = (directory / f"sub-MonkeyL-held-in-calib_ses-{suffix}_behavior+ecephys.nwb").resolve()
            try:
                path.relative_to(directory)
            except ValueError as exc:
                raise ValueError(f"source path escapes held-in-calib directory: {path}") from exc
            relative = str(path.relative_to(root)).lower()
            if any(token in relative for token in _FORBIDDEN_PATH_TOKENS):
                raise ValueError(f"forbidden Version-B source path: {path}")
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            paths[session] = path
        if tuple(paths) != tuple(self.source_session_names):
            raise ValueError(f"source order drift: {tuple(paths)}")
        return paths

    def setup(self, stage: Optional[str] = None) -> None:
        self._assert_contract(stage)
        if stage not in (None, "fit"):
            raise ValueError("source-only Version-B fit supports only stage='fit'")
        if getattr(self, "train_dataset", None) is not None:
            return
        if self.afc4_arm not in self._SOURCE_ONLY_ARMS:
            raise ValueError(
                f"source-only fit cannot construct target-dependent carrier arm {self.afc4_arm!r}"
            )
        paths = self._approved_source_paths()
        task = self._falcon_task_m1()
        source_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        covariates_mean = covariates_std = None
        for index, name in enumerate(self.source_session_names):
            record = self.prepare_session_data(
                paths[name], task,
                standardize_covariates=bool(self.hparams.standardize_covariates),
                covariates_mean=covariates_mean,
                covariates_std=covariates_std,
                use_intertrials=bool(self.hparams.use_intertrials),
            )
            if index == 0:
                covariates_mean = record["covariates_mean"]
                covariates_std = record["covariates_std"]
            source_records[name] = record

        self.source_paths = paths
        # Deliberately do not set ``target_path``.  A source-only manifest must
        # make it impossible to accidentally imply that the target was read.
        self.train_session_names = list(self.source_session_names)
        self.val_heldin_session_names = []
        self.val_heldout_session_names = []
        self.train_calib_heldin_sessions = source_records
        self.carrier_plan = None
        self.train_dataset = self._dataset(
            source_records, source_records, split="train", carrier_plan=None
        )
        self.val_heldin_dataset = None
        self.val_heldout_dataset = None
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset, self.batch_size_per_device, shuffle=True,
            seed=int(self.hparams.sampler_seed),
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
        )
        self._setup_stage = "fit"

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        raise RuntimeError("source-only Version-B fit forbids test/formal evaluation")

    def get_split_manifest(self) -> dict[str, Any]:
        paths = getattr(self, "source_paths", OrderedDict())
        return {
            "schema": "m1_version_b_source_only_fit_v2",
            "task": "m1",
            "outer_fold": self.outer_fold,
            "outer_left_out": self.outer_left_out,
            "train_sessions": list(self.source_session_names),
            "validation_sessions": [],
            "source_only": True,
            "formal": False,
            "evalai": False,
            "heldout_files_opened": False,
            "heldout_values_used": False,
            "minival_files_opened": False,
            "minival_values_used": False,
            "target_path_resolved_during_fit": False,
            "target_query_values_read_by_fit": False,
            "target_backpropagation": False,
            "checkpoint_selection": "fixed_last_epoch_11_train_source_only",
            "carrier_arm": self.afc4_arm,
            "source_files": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
        }

    def write_source_only_manifest(self, output_dir: str | Path) -> Path:
        """Write an unambiguous source-only-fit receipt beside the run log."""
        output = Path(output_dir) / "m1_version_b_source_only_fit_manifest.json"
        encoded = json.dumps(self.get_split_manifest(), indent=2, sort_keys=True) + "\n"
        if output.exists() and output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite incompatible manifest: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        (output.with_suffix(".sha256")).write_text(f"{_sha256(output)}  {output.name}\n", encoding="utf-8")
        return output


__all__ = [
    "M1_VERSION_B_FOLDS",
    "M1VersionBSourceLOSODataModule",
    "M1VersionBSourceOnlyFitDataModule",
]
