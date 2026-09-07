"""Clean nested-LOSO H1 data plumbing for the source-only AFC4 pilot.

The module is append-only and does not alter the RT/M2 data paths.  During
``fit`` it opens only the 11 inner-train held-in calibration NWBs and one
inner-validation calibration/minival pair.  The outer target's matching
calibration/minival files are indexed by name but not opened.  A separate
post-selection helper opens that pair exactly once for an outer one-shot
evaluation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.data.falcon_datamodule import SessionBatchSampler
from src.data.falcon_h1_afc4_features import (
    H1_ARMS,
    H1_NUM_NEURONS,
    H1AFC4SourcePlan,
    H1SessionRecord,
    index_h1_heldin_pairs,
    load_h1_record,
)


H1_EXPECTED_SESSION_COUNT = 13


def _source_plan_sha256(plan: H1AFC4SourcePlan, *, excluded_outer_target: str) -> str:
    payload = json.dumps(
        plan.manifest(excluded_outer_target=excluded_outer_target),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_shared_teacher_binding(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Require Full/B4/Zero manifests to bind one teacher and one source plan."""

    required = ("afc4_h1q3", "afc4_h1_b4", "zero4")
    missing = [arm for arm in required if arm not in manifests]
    if missing:
        raise ValueError(f"H1 student teacher binding missing arms: {missing}")
    teacher = {str(manifests[arm].get("teacher_checkpoint_sha256", "")) for arm in required}
    plan = {str(manifests[arm].get("source_plan_sha256", "")) for arm in required}
    if len(teacher) != 1 or not next(iter(teacher)):
        raise ValueError("H1 Full/B4/Zero do not share one non-empty teacher checkpoint SHA")
    if len(plan) != 1 or not next(iter(plan)):
        raise ValueError("H1 Full/B4/Zero do not share one non-empty source-plan SHA")
    return {"teacher_checkpoint_sha256": next(iter(teacher)), "source_plan_sha256": next(iter(plan))}


@dataclass(frozen=True)
class H1NestedLossoSplit:
    all_sessions: tuple[str, ...]
    outer_fold: int
    outer_target_session: str
    outer_source_sessions: tuple[str, ...]
    inner_validation_session: str
    inner_train_sessions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_sessions": list(self.all_sessions),
            "outer_fold": int(self.outer_fold),
            "outer_target_session": self.outer_target_session,
            "outer_source_sessions": list(self.outer_source_sessions),
            "inner_validation_session": self.inner_validation_session,
            "inner_train_sessions": list(self.inner_train_sessions),
            "inner_selection_rule": "sorted_all_sessions[(outer_fold + 1) % 13]",
        }


def h1_nested_loso_partition(
    session_names: Sequence[str], outer_fold: int, *, expected_session_count: int = H1_EXPECTED_SESSION_COUNT
) -> H1NestedLossoSplit:
    sessions = tuple(sorted(str(name) for name in session_names))
    if len(sessions) != int(expected_session_count) or len(set(sessions)) != len(sessions):
        raise ValueError(f"H1 clean nested LOSO expects {expected_session_count} unique sessions, got {sessions}")
    fold = int(outer_fold)
    if fold < 0 or fold >= len(sessions):
        raise ValueError(f"outer_loso_fold must be in [0,{len(sessions)-1}], got {outer_fold}")
    outer_target = sessions[fold]
    outer_source = tuple(name for name in sessions if name != outer_target)
    # Match the frozen clean nested-LOSO rule used by RT: choose the next
    # session in the full sorted ring, then remove the outer target from the
    # resulting inner-train pool.  Selecting from ``outer_source`` would shift
    # every fold after the target and silently change checkpoint selection.
    inner_validation = sessions[(fold + 1) % len(sessions)]
    inner_train = tuple(name for name in outer_source if name != inner_validation)
    if outer_target in inner_train or outer_target == inner_validation:
        raise RuntimeError("H1 nested LOSO outer target leaked into inner split")
    if len(outer_source) != 12 or len(inner_train) != 11:
        raise RuntimeError("H1 nested LOSO cardinality contract failed")
    return H1NestedLossoSplit(
        sessions, fold, outer_target, outer_source, inner_validation, inner_train
    )


def _support_trial_array(record: H1SessionRecord, *, max_trial_length: int, pad_value: float) -> np.ndarray:
    values = record.support_neural
    if values.shape[1] != H1_NUM_NEURONS:
        raise ValueError("H1 support neural channel count mismatch")
    if values.shape[0] > int(max_trial_length):
        raise ValueError(
            f"H1 first-TrialNum support has {values.shape[0]} bins, exceeds max_trial_length={max_trial_length}"
        )
    output = np.full((1, int(max_trial_length), H1_NUM_NEURONS), float(pad_value), dtype=np.float32)
    output[0, : values.shape[0]] = values.astype(np.float32)
    return output


class H1AFC4Dataset(Dataset):
    """Window dataset with one first-TrialNum support tensor per session."""

    def __init__(
        self,
        *,
        support_records: Mapping[str, H1SessionRecord],
        query_records: Mapping[str, H1SessionRecord],
        side_features: Mapping[str, np.ndarray] | None,
        query_mode: str,
        include_side_features: bool = True,
        window_size: int = 700,
        max_trial_length: int = 1024,
        pad_value: float = -1.0,
    ) -> None:
        if query_mode not in {"calib_post_support", "minival"}:
            raise ValueError("H1 query_mode must be calib_post_support or minival")
        if int(window_size) <= 0 or int(max_trial_length) <= 0:
            raise ValueError("H1 window/max_trial_length must be positive")
        self.window_size = int(window_size)
        self.max_trial_length = int(max_trial_length)
        self.pad_value = float(pad_value)
        self.query_mode = query_mode
        self.include_side_features = bool(include_side_features)
        self.support_records = dict(support_records)
        self.query_records = dict(query_records)
        if set(self.support_records) != set(self.query_records):
            raise ValueError("H1 support/query session sets must match")
        if side_features is None:
            if self.include_side_features:
                raise ValueError("H1 student dataset requires side_features")
            side_features = {
                name: np.zeros((H1_NUM_NEURONS, 4), dtype=np.float32)
                for name in self.query_records
            }
        if set(self.query_records) != set(side_features):
            raise ValueError("H1 side-feature/session sets must match")
        self.session_names = tuple(sorted(self.query_records))
        self.support_features = {}
        self.support_trialized = {}
        self.neural_data: dict[str, np.ndarray] = {}
        self.behavior_data: dict[str, np.ndarray] = {}
        self.eval_mask: dict[str, np.ndarray] = {}
        self.window_indices: list[tuple[str, int]] = []
        self.query_window_audit: dict[str, dict[str, Any]] = {}

        history = self.window_size - 1
        for name in self.session_names:
            support = self.support_records[name]
            query = self.query_records[name]
            side = np.asarray(side_features[name], dtype=np.float32)
            if side.shape != (H1_NUM_NEURONS, 4) or not np.isfinite(side).all():
                raise ValueError(f"{name}: H1 side feature shape/finite contract failed: {side.shape}")
            self.support_features[name] = side
            self.support_trialized[name] = _support_trial_array(
                support, max_trial_length=self.max_trial_length, pad_value=self.pad_value
            )
            neural = np.asarray(query.neural, dtype=np.float32)
            behavior = np.asarray(query.covariates, dtype=np.float32)
            evaluation = np.asarray(query.eval_mask, dtype=bool)
            self.neural_data[name] = np.pad(neural, ((history, 0), (0, 0)), constant_values=0.0)
            self.behavior_data[name] = np.pad(behavior, ((history, 0), (0, 0)), constant_values=0.0)
            self.eval_mask[name] = np.pad(evaluation, (history, 0), constant_values=False)

            if query_mode == "calib_post_support":
                valid_trial = np.flatnonzero(query.eval_mask & np.isfinite(query.trial_num))
                if valid_trial.size == 0:
                    raise ValueError(f"{name}: query record has no valid TrialNum")
                first_value = query.trial_num[valid_trial[0]]
                first_indices = np.flatnonzero(query.trial_num == first_value)
                if first_indices.size == 0:
                    raise ValueError(f"{name}: query first TrialNum boundary is missing")
                minimum_start = int(first_indices[-1] + history + 1)
                query_trials = int(np.unique(query.trial_num[query.trial_num > first_value]).size)
            else:
                minimum_start = 0
                query_trials = int(np.unique(query.trial_num).size)
            maximum_start = self.neural_data[name].shape[0] - self.window_size
            count_before = len(self.window_indices)
            for start in range(max(0, minimum_start), maximum_start + 1):
                last = start + self.window_size - 1
                if bool(self.eval_mask[name][last]):
                    self.window_indices.append((name, start))
            eligible = len(self.window_indices) - count_before
            self.query_window_audit[name] = {
                "query_mode": query_mode,
                "support_trial_count": 1,
                "query_trials": query_trials,
                "minimum_window_start_padded_bin": minimum_start,
                "maximum_window_start_padded_bin": maximum_start,
                "window_size": self.window_size,
                "eligible_windows": eligible,
                "full_window_disjoint": query_mode == "calib_post_support",
            }
            if eligible <= 0:
                raise ValueError(f"{name}: H1 query has no eligible windows")

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int):
        session_name, start = self.window_indices[int(index)]
        end = start + self.window_size
        batch = (
            self.neural_data[session_name][start:end].copy(),
            self.behavior_data[session_name][start:end].copy(),
            self.support_trialized[session_name].copy(),
            session_name,
        )
        if self.include_side_features:
            return (*batch, self.support_features[session_name].copy())
        return batch


class H1CleanNestedLossoDataModule(pl.LightningDataModule):
    """H1 clean nested-LOSO fit data with an explicit outer-target boundary."""

    def __init__(
        self,
        *,
        task: str = "h1",
        data_dir: str = "",
        outer_loso_fold: int | None = None,
        loso_fold: int | None = None,
        calibration_n_trials: int = 1,
        side_feature_group: str = "afc4_h1q3",
        side_feature_shuffle_seed: int = 42,
        batch_size: int = 8,
        window_size: int = 700,
        max_trial_length: int = 1024,
        pad_value: float = -1.0,
        session_window_budget: int = 512,
        session_balanced_sampling: bool = True,
        sampler_reshuffle_each_epoch: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampler_seed: int = 42,
        include_side_features: bool = True,
        teacher_checkpoint_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if task != "h1":
            raise ValueError("H1 clean nested LOSO requires task='h1'")
        if outer_loso_fold is None:
            outer_loso_fold = loso_fold
        elif loso_fold is not None and int(outer_loso_fold) != int(loso_fold):
            raise ValueError("outer_loso_fold and loso_fold disagree")
        if outer_loso_fold is None:
            raise ValueError("outer_loso_fold must be explicit")
        if str(side_feature_group).lower() not in H1_ARMS:
            raise ValueError(f"Unsupported H1 side_feature_group {side_feature_group!r}; choices={H1_ARMS}")
        if int(batch_size) <= 0 or int(session_window_budget) <= 0:
            raise ValueError("H1 batch_size/session_window_budget must be positive")
        if int(calibration_n_trials) != 1:
            raise ValueError("H1 AFC4 protocol freezes calibration_n_trials=1")
        if int(session_window_budget) % int(batch_size):
            raise ValueError("H1 session_window_budget must be divisible by batch_size")
        self.save_hyperparameters()
        self.hparams.outer_loso_fold = int(outer_loso_fold)
        self.hparams.loso_fold = int(outer_loso_fold)
        self._setup_complete = False
        self._outer_target_loaded = False
        self._outer_target_query_labels_read = False

    def setup(self, stage: str | None = None) -> None:
        if self._setup_complete:
            return
        pairs = index_h1_heldin_pairs(self.hparams.data_dir)
        split = h1_nested_loso_partition(tuple(pairs), int(self.hparams.outer_loso_fold))
        self.split = split
        self.session_pairs = pairs
        # Only inner-train calibration and inner-validation calib/minival are
        # opened.  The outer target pair remains a path-only record.
        inner_train_calib = OrderedDict(
            (name, load_h1_record(pairs[name][0], split="calib")) for name in split.inner_train_sessions
        )
        inner_val_calib = load_h1_record(pairs[split.inner_validation_session][0], split="calib")
        inner_val_minival = load_h1_record(pairs[split.inner_validation_session][1], split="minival")
        self.outer_target_path_pair = pairs[split.outer_target_session]
        if split.outer_target_session in inner_train_calib or split.outer_target_session == inner_val_calib.session_name:
            raise RuntimeError("H1 outer target was opened during fit setup")
        self.source_plan = H1AFC4SourcePlan(inner_train_calib, shuffle_seed=int(self.hparams.side_feature_shuffle_seed))
        arm = str(self.hparams.side_feature_group).lower()
        train_side = {
            name: self.source_plan.descriptor_for(record, arm)[0] for name, record in inner_train_calib.items()
        }
        val_side = {
            inner_val_calib.session_name: self.source_plan.descriptor_for(inner_val_calib, arm)[0]
        }
        self.train_dataset = H1AFC4Dataset(
            support_records=inner_train_calib,
            query_records=inner_train_calib,
            side_features=train_side,
            query_mode="calib_post_support",
            include_side_features=bool(self.hparams.include_side_features),
            window_size=int(self.hparams.window_size),
            max_trial_length=int(self.hparams.max_trial_length),
            pad_value=float(self.hparams.pad_value),
        )
        self.val_inner_dataset = H1AFC4Dataset(
            support_records={inner_val_calib.session_name: inner_val_calib},
            query_records={inner_val_calib.session_name: inner_val_minival},
            side_features=val_side,
            query_mode="minival",
            include_side_features=bool(self.hparams.include_side_features),
            window_size=int(self.hparams.window_size),
            max_trial_length=int(self.hparams.max_trial_length),
            pad_value=float(self.hparams.pad_value),
        )
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset,
            int(self.hparams.batch_size),
            shuffle=True,
            seed=int(self.hparams.sampler_seed),
            balance_sessions=bool(self.hparams.session_balanced_sampling),
            reshuffle_each_epoch=bool(self.hparams.sampler_reshuffle_each_epoch),
            window_budget_per_session=int(self.hparams.session_window_budget),
            require_full_window_budget=True,
        )
        self.val_inner_batch_sampler = SessionBatchSampler(
            self.val_inner_dataset,
            int(self.hparams.batch_size),
            shuffle=False,
        )
        self.loaded_fit_sessions = tuple(inner_train_calib) + (inner_val_calib.session_name,)
        self._setup_complete = True

    def train_dataloader(self) -> DataLoader:
        self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
        )

    def val_dataloader(self) -> DataLoader:
        self.setup("fit")
        return DataLoader(
            self.val_inner_dataset,
            batch_sampler=self.val_inner_batch_sampler,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
        )

    def test_dataloader(self) -> DataLoader:
        raise RuntimeError(
            "H1 clean nested LOSO has no fit-time outer loader; run the explicit post-selection outer evaluator"
        )

    @property
    def outer_target_loaded(self) -> bool:
        return bool(self._outer_target_loaded)

    @property
    def outer_target_query_labels_read(self) -> bool:
        return bool(self._outer_target_query_labels_read)

    def get_split_manifest(self) -> dict[str, Any]:
        self.setup("fit")
        split = self.split
        return {
            "schema": "h1_afc4_clean_nested_loso_split_manifest_v1",
            "task": "h1",
            "protocol": "clean_nested_loso",
            "validation_protocol": "nested_loso",
            "outer_loso_fold": int(split.outer_fold),
            "loso_fold": int(split.outer_fold),
            "split": split.as_dict(),
            "arm": str(self.hparams.side_feature_group).lower(),
            "include_side_features": bool(self.hparams.include_side_features),
            "teacher_checkpoint_sha256": self.hparams.teacher_checkpoint_sha256,
            "source_plan_sha256": _source_plan_sha256(
                self.source_plan, excluded_outer_target=split.outer_target_session
            ),
            "loaded_fit_sessions": list(self.loaded_fit_sessions),
            "outer_target_session": split.outer_target_session,
            "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
            "source_only_basis_or_normalizer": self.source_plan.manifest(
                excluded_outer_target=split.outer_target_session
            ),
            "checkpoint_selection": {
                "scope": "inner_validation_session_only",
                "inner_validation_session": split.inner_validation_session,
                "outer_target_used": False,
            },
            "support": {
                "trial_count": 1,
                "contract": "first TrialNum AND eval_mask-valid AND non-static 7D velocity bins",
                "query_mode_train": "calib_post_support",
                "query_mode_inner_validation": "matching_minival",
            },
            "query_window_audit": {
                "inner_train": self.train_dataset.query_window_audit,
                "inner_validation": self.val_inner_dataset.query_window_audit,
            },
        }

    def write_source_only_manifest(self, output_dir: str | Path) -> dict[str, Any]:
        """Write the fit/source-plan manifest before the first optimizer step.

        ``src/train.py`` calls this narrow capability immediately after data
        module construction and before ``Trainer.fit``.  H1 teacher runs have
        no encoder-cost profile, so relying on the generic post-fit export
        would otherwise lose the split provenance.  The manifest is immutable
        and includes the exact inner-validation monitor contract.
        """

        self.setup("fit")
        from src.callbacks.h1_nested_selection import h1_inner_validation_monitor

        manifest = self.get_split_manifest()
        monitor = h1_inner_validation_monitor(self.split.inner_validation_session)
        manifest["status"] = "PASS_H1_SOURCE_ONLY_MANIFEST_PRE_OPTIMIZER"
        manifest["source_only_manifest_written_before_optimizer"] = True
        manifest["checkpoint_selection"] = {
            **dict(manifest.get("checkpoint_selection", {})),
            "monitor": monitor,
            "scope": "inner_validation_session_only",
            "outer_target_used": False,
        }
        manifest["selection_contract"] = {
            "arm": str(self.hparams.side_feature_group).lower(),
            "role": "student" if bool(self.hparams.include_side_features) else "teacher",
            "monitor": monitor,
            "selection_scope": "inner_validation_session_only",
            "inner_validation_session": self.split.inner_validation_session,
            "outer_target_session": self.split.outer_target_session,
            "outer_target_used": False,
        }
        path = Path(output_dir) / "h1_split_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite H1 source-only manifest: {path}")
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **manifest}


def build_h1_outer_target_dataset(
    *,
    data_dir: str | Path,
    outer_loso_fold: int,
    source_records: Mapping[str, H1SessionRecord],
    side_feature_group: str,
    side_feature_shuffle_seed: int = 42,
    window_size: int = 700,
    max_trial_length: int = 1024,
    pad_value: float = -1.0,
    include_side_features: bool = True,
) -> tuple[H1AFC4Dataset, H1NestedLossoSplit, H1SessionRecord, H1SessionRecord]:
    """Open one outer target pair after checkpoint selection only."""

    pairs = index_h1_heldin_pairs(data_dir)
    split = h1_nested_loso_partition(tuple(pairs), int(outer_loso_fold))
    if split.outer_target_session in source_records:
        raise ValueError("H1 outer target is present in source_records")
    if tuple(sorted(source_records)) != split.inner_train_sessions:
        raise ValueError("H1 outer evaluator source_records must equal inner_train_sessions exactly")
    plan = H1AFC4SourcePlan(source_records, shuffle_seed=int(side_feature_shuffle_seed))
    support = load_h1_record(pairs[split.outer_target_session][0], split="calib")
    query = load_h1_record(pairs[split.outer_target_session][1], split="minival")
    arm = str(side_feature_group).lower()
    side = plan.descriptor_for(support, arm)[0]
    dataset = H1AFC4Dataset(
        support_records={support.session_name: support},
        query_records={query.session_name: query},
        side_features={support.session_name: side},
        query_mode="minival",
        include_side_features=bool(include_side_features),
        window_size=int(window_size),
        max_trial_length=int(max_trial_length),
        pad_value=float(pad_value),
    )
    return dataset, split, support, query


# Conventional acronym alias for callers that prefer LOSO spelling.
H1CleanNestedLOSODataModule = H1CleanNestedLossoDataModule
