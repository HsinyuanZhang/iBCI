"""Clean nested-LOSO data plumbing for the RT development program.

The existing :mod:`src.data.rt_datamodule` is intentionally retained for the
historical pilot receipts.  It constructs an outer LOSO validation dataset in
``setup`` and therefore cannot be used to make an unbiased epoch-selection
claim.  This module is the append-only clean path:

* the outer target path is identified from the sorted file list, but its NWB is
  never opened during ``fit``;
* the next source session in cyclic sorted order is the sole inner validation
  session; the remaining 13 source sessions are training sessions;
* AFC4/K4 carrier statistics are fitted from those 13 sessions only; and
* ``test_dataloader`` is deliberately unavailable.  The outer target is
  evaluated by the separate one-shot evaluator after a checkpoint is selected.

No model architecture is changed here.  The boundary is entirely a loader and
checkpoint-selection contract.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader

from src.data.falcon_datamodule import FalconDataset, SessionBatchSampler
from src.data.falcon_k4_features import fit_train_k4_stats
from src.data.afc4_xls_v2_adapter import AUDIT_SHA256 as XLS_V2_SUPPORT_AUDIT_SHA256
from src.data.rt_k4_loader import (
    RT_EXPECTED_SESSION_COUNT,
    RT_GATES,
    RT_PROTOCOL,
    find_rt_sessions,
    load_rt_session,
    summarize_rt_trial_budget,
)
from src.data.rt_sparse_endpoint_loader import load_rt_sparse_endpoint_t4d_session


logger = logging.getLogger(__name__)


_K4_FEATURE_GROUPS = {
    "k4",
    "ks4",
    "k4ls",
    "afc4_vel",
    "afc4_rs",
    "afc4_ls",
    "afc4_mb4",
    "afc4_b4",
    "afc4_w4",
    "afc4_xls_v2",
}
_T4D_FEATURE_GROUPS = {"rt_sparse_endpoint_t4d"}


_RT_ARM_SPECS: dict[str, dict[str, str]] = {
    "none": {
        "canonical_arm": "none",
        "implementation_group": "none",
        "side_feature_semantics": "no side feature; native B2/LatePool identity reference",
    },
    "zero4": {
        "canonical_arm": "zero4",
        "implementation_group": "zero4",
        "side_feature_semantics": "[N,4] all zero; width-matched no-label control",
    },
    "rt_sparse_endpoint_t4d": {
        "canonical_arm": "rt_sparse_endpoint_t4d",
        "implementation_group": "rt_sparse_endpoint_t4d",
        "side_feature_semantics": "precomputed endpoint-only per-unit [a,c,0,0] carrier; no dense velocity in carrier",
    },
    "k4": {
        "canonical_arm": "afc4_vel",
        "implementation_group": "k4",
        "side_feature_semantics": "aligned per-unit [wx,wy,||w||,b] velocity carrier",
    },
    "afc4_vel": {
        "canonical_arm": "afc4_vel",
        "implementation_group": "k4",
        "side_feature_semantics": "aligned per-unit [wx,wy,||w||,b] velocity carrier",
    },
    "ks4": {
        "canonical_arm": "afc4_rs",
        "implementation_group": "ks4",
        "side_feature_semantics": "complete deterministic descriptor-row shuffle after source-only normalization",
    },
    "afc4_rs": {
        "canonical_arm": "afc4_rs",
        "implementation_group": "ks4",
        "side_feature_semantics": "complete deterministic descriptor-row shuffle after source-only normalization",
    },
    "k4ls": {
        "canonical_arm": "afc4_ls",
        "implementation_group": "k4ls",
        "side_feature_semantics": "segment-preserving continuous velocity label-association null",
    },
    "afc4_ls": {
        "canonical_arm": "afc4_ls",
        "implementation_group": "k4ls",
        "side_feature_semantics": "segment-preserving continuous velocity label-association null",
    },
    "afc4_mb4": {
        "canonical_arm": "afc4_mb4",
        "implementation_group": "k4__normalized_component_mask",
        "side_feature_semantics": "aligned normalized [0,0,||w||,b] component ablation",
    },
    "afc4_b4": {
        "canonical_arm": "afc4_b4",
        "implementation_group": "k4__normalized_component_mask",
        "side_feature_semantics": "aligned normalized [0,0,0,b] component ablation",
    },
    "afc4_w4": {
        "canonical_arm": "afc4_w4",
        "implementation_group": "k4__normalized_component_mask",
        "side_feature_semantics": "aligned normalized [wx,wy,0,0] component ablation",
    },
    "afc4_xls_v2": {
        "canonical_arm": "afc4_xls_v2",
        "implementation_group": "afc4_xls_v2",
        "side_feature_semantics": "audited session-namespaced cross-reach support-velocity pairing null [wx,wy,||w||,b]",
    },
}


@dataclass(frozen=True)
class NestedRtSplit:
    """The complete deterministic outer/inner session partition."""

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
            "inner_selection_rule": "sorted_sessions[(outer_fold + 1) % session_count]",
        }


def session_name_from_path(path: str | Path) -> str:
    """Derive an RT session name without opening its NWB file."""

    name = Path(path).name
    suffix = "_behavior+ecephys.nwb"
    if not name.startswith("sub-C_ses-RT-") or not name.endswith(suffix):
        raise ValueError(f"Not an RT sub-C session NWB: {path}")
    return name.removesuffix(suffix).removeprefix("sub-C_")


def index_rt_session_paths(
    data_dir: str | Path,
    *,
    expected_session_count: int = RT_EXPECTED_SESSION_COUNT,
) -> OrderedDict[str, Path]:
    """Index RT paths by name without reading any session payload.

    This function is intentionally the only operation used to identify the
    outer target during training.  Calling it cannot read query behaviour or
    construct a ``FalconDataset``.
    """

    paths = find_rt_sessions(data_dir)
    if len(paths) != int(expected_session_count):
        raise FileNotFoundError(
            f"RT nested LOSO requires exactly {expected_session_count} RT NWBs in "
            f"{Path(data_dir)}, found {len(paths)}"
        )
    indexed: OrderedDict[str, Path] = OrderedDict()
    for path in sorted(paths):
        name = session_name_from_path(path)
        if name in indexed:
            raise ValueError(f"Duplicate RT session name in file index: {name}")
        indexed[name] = Path(path)
    if tuple(indexed) != tuple(sorted(indexed)):
        raise RuntimeError("RT session path index is not deterministically sorted")
    return indexed


def nested_loso_partition(
    session_names: Sequence[str], outer_fold: int, *, expected_session_count: int | None = None
) -> NestedRtSplit:
    """Return the fail-closed cyclic next-source nested LOSO partition.

    For sorted sessions ``S`` and outer target ``S[i]``, inner validation is
    ``S[(i+1) % len(S)]``.  It is therefore always a source session, and the
    inner training set has exactly ``len(S)-2`` sessions (13 for RT's 15).
    """

    sessions = tuple(sorted(str(name) for name in session_names))
    if len(set(sessions)) != len(sessions):
        raise ValueError("RT nested LOSO requires unique session names")
    if expected_session_count is not None and len(sessions) != int(expected_session_count):
        raise ValueError(
            f"RT nested LOSO expects {expected_session_count} sessions, got {len(sessions)}"
        )
    if len(sessions) < 3:
        raise ValueError("Nested LOSO requires at least three sessions")
    fold = int(outer_fold)
    if fold < 0 or fold >= len(sessions):
        raise ValueError(f"outer_loso_fold must be in [0, {len(sessions) - 1}], got {outer_fold}")
    outer_target = sessions[fold]
    inner_validation = sessions[(fold + 1) % len(sessions)]
    outer_source = tuple(name for name in sessions if name != outer_target)
    inner_train = tuple(name for name in outer_source if name != inner_validation)
    if outer_target in inner_train or outer_target == inner_validation:
        raise RuntimeError("Nested LOSO accidentally included the outer target in inner data")
    if len(outer_source) != len(sessions) - 1 or len(inner_train) != len(sessions) - 2:
        raise RuntimeError("Nested LOSO cardinality contract failed")
    return NestedRtSplit(
        all_sessions=sessions,
        outer_fold=fold,
        outer_target_session=outer_target,
        outer_source_sessions=outer_source,
        inner_validation_session=inner_validation,
        inner_train_sessions=inner_train,
    )


def _normalizer_jsonable(normalizer: dict[str, Any] | None) -> dict[str, Any] | None:
    if normalizer is None:
        return None
    return {
        **normalizer,
        "mean": np.asarray(normalizer["mean"], dtype=np.float32).tolist(),
        "std": np.asarray(normalizer["std"], dtype=np.float32).tolist(),
    }


def _dataset_kwargs(hparams: Mapping[str, Any], feature_group: str, query_start_trial: int) -> dict[str, Any]:
    values = {
        "window_size": int(hparams["window_size"]),
        "calibration_n_trials": int(hparams["calibration_n_trials"]),
        "random_calibration": False,
        "smooth_calibration": False,
        "max_trial_length": int(hparams["max_trial_length"]),
        "use_calib_intertrials": True,
        "remove_calib_still_times": False,
        "interpolate_trials": bool(hparams["interpolate_trials"]),
        "interpolate_trials_kind": str(hparams["interpolate_trials_kind"]),
        "pad_value": float(hparams["pad_value"]),
        "side_feature_group": feature_group,
        "side_feature_shuffle_seed": int(hparams["side_feature_shuffle_seed"]),
        "query_start_trial": int(query_start_trial),
    }
    if feature_group == "afc4_xls_v2":
        path = hparams.get("xls_v2_support_audit_path")
        if not path:
            raise ValueError("afc4_xls_v2 requires xls_v2_support_audit_path")
        values["xls_v2_support_audit_path"] = str(path)
    return values


class RtNestedLossoDataModule(pl.LightningDataModule):
    """RT fit DataModule whose outer target is structurally absent from fit."""

    def __init__(
        self,
        task: str = "rt",
        data_dir: str = "",
        batch_size: int = 32,
        window_size: int = 50,
        calibration_n_trials: int = 24,
        query_start_trial: int | None = None,
        random_calibration: bool = False,
        smooth_calibration: bool = False,
        max_trial_length: int = 100,
        interpolate_trials: bool = True,
        interpolate_trials_kind: str = "cubic",
        pad_value: float = -1.0,
        validation_protocol: str = "nested_loso",
        outer_loso_fold: int | None = None,
        loso_fold: int | None = None,
        side_feature_group: str | None = None,
        side_feature_shuffle_seed: int = 42,
        session_window_budget: int = 4096,
        session_balanced_sampling: bool = True,
        sampler_reshuffle_each_epoch: bool = True,
        expected_session_count: int = RT_EXPECTED_SESSION_COUNT,
        num_workers: int = 4,
        pin_memory: bool = True,
        sampler_seed: int = 42,
        xls_v2_support_audit_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if task != "rt":
            raise ValueError(f"RtNestedLossoDataModule only supports task='rt', got {task!r}")
        if validation_protocol not in {"nested_loso", "loso_nested"}:
            raise ValueError("RT clean path requires validation_protocol='nested_loso'")
        if int(expected_session_count) != RT_EXPECTED_SESSION_COUNT:
            raise ValueError(
                f"RT protocol expects exactly {RT_EXPECTED_SESSION_COUNT} sessions, not {expected_session_count}"
            )
        if outer_loso_fold is None:
            outer_loso_fold = loso_fold
        elif loso_fold is not None and int(outer_loso_fold) != int(loso_fold):
            raise ValueError("outer_loso_fold and loso_fold aliases disagree")
        if outer_loso_fold is None:
            raise ValueError("outer_loso_fold must be set; clean nested LOSO never defaults a fold")
        if int(calibration_n_trials) != 24:
            raise ValueError("RT clean nested LOSO is frozen to chronological M24 support")
        effective_query_start = int(calibration_n_trials) if query_start_trial is None else int(query_start_trial)
        if effective_query_start != int(calibration_n_trials):
            raise ValueError("query_start_trial must equal calibration_n_trials=24")
        if random_calibration or smooth_calibration:
            raise ValueError("RT clean nested LOSO requires chronological raw, unsmoothed calibration")
        if int(window_size) != 50:
            raise ValueError("RT clean nested LOSO requires window_size=50")
        if int(max_trial_length) != 100 or not interpolate_trials:
            raise ValueError("RT clean nested LOSO requires max_trial_length=100 and interpolation")
        if int(batch_size) <= 0 or int(session_window_budget) <= 0:
            raise ValueError("batch_size and session_window_budget must be positive")
        if int(session_window_budget) % int(batch_size):
            raise ValueError("session_window_budget must be divisible by batch_size")
        if side_feature_group is None:
            raise ValueError("RT clean nested LOSO requires an explicit side_feature_group/arm")
        feature_group = str(side_feature_group).lower()
        if feature_group not in _RT_ARM_SPECS:
            raise ValueError(f"Unsupported RT side_feature_group {side_feature_group!r}")
        if feature_group == "afc4_xls_v2" and not xls_v2_support_audit_path:
            raise ValueError("RT XLSv2 requires an explicit immutable support-audit path")
        self._outer_fold = int(outer_loso_fold)
        self._query_start_trial = effective_query_start
        self._feature_group = feature_group
        self._setup_complete = False
        self._outer_target_loader_opened = False
        self._outer_target_query_labels_read = False

        # Keep both names in hparams for Hydra's historical run-directory
        # interpolation while exposing one unambiguous canonical fold.
        self.hparams.outer_loso_fold = self._outer_fold
        self.hparams.loso_fold = self._outer_fold

    def setup(self, stage: str | None = None) -> None:
        if self._setup_complete:
            return
        indexed_paths = index_rt_session_paths(
            self.hparams.data_dir,
            expected_session_count=int(self.hparams.expected_session_count),
        )
        partition = nested_loso_partition(
            tuple(indexed_paths),
            self._outer_fold,
            expected_session_count=int(self.hparams.expected_session_count),
        )
        self.split = partition
        self.session_paths = indexed_paths
        self.outer_target_path = indexed_paths[partition.outer_target_session]

        # This is the critical guard: only the 14 outer-source NWBs are opened
        # during fit, and the outer target is not even passed to load_rt_session.
        loaded_names = partition.inner_train_sessions + (partition.inner_validation_session,)
        all_loaded: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for name in loaded_names:
            raw = (load_rt_sparse_endpoint_t4d_session(indexed_paths[name])
                   if self._feature_group in _T4D_FEATURE_GROUPS else load_rt_session(indexed_paths[name]))
            if str(raw.get("session_name")) != name:
                raise RuntimeError(
                    f"RT loader returned session {raw.get('session_name')!r} for indexed name {name!r}"
                )
            all_loaded[name] = raw
            segment_audit = raw["rt_segment_audit"]
            logger.info(
                "%s: units=%d trials=%d complete_cue=%d accepted_segments=%d event_bins=%d",
                name,
                raw["neural"].shape[1],
                int(raw["trial_change"].sum()),
                segment_audit["complete_cue_trials"],
                segment_audit["accepted_reach_segments"],
                segment_audit["event_qualified_bins"],
            )
        if partition.outer_target_session in all_loaded:
            raise RuntimeError("Outer target was loaded during nested fit setup")

        train_dict = OrderedDict((name, all_loaded[name]) for name in partition.inner_train_sessions)
        val_dict = OrderedDict([(partition.inner_validation_session, all_loaded[partition.inner_validation_session])])
        ds_kwargs = _dataset_kwargs(self.hparams, self._feature_group, self._query_start_trial)
        self.train_dataset = FalconDataset(
            sessions_dict=train_dict,
            calib_sessions_dict=train_dict,
            split="rt_nested_inner_train",
            **ds_kwargs,
        )

        side_feature_mean = None
        side_feature_std = None
        self.native_k4_normalization: dict[str, Any] | None = None
        if self._feature_group in (_K4_FEATURE_GROUPS | _T4D_FEATURE_GROUPS):
            raw_features = self.train_dataset.native_k4_statistics_inputs(partition.inner_train_sessions)
            side_feature_mean, side_feature_std = fit_train_k4_stats(
                raw_features, partition.inner_train_sessions
            )
            if self._feature_group in _T4D_FEATURE_GROUPS:
                # The two trailing dimensions are architectural zero pads,
                # never statistics-bearing features.
                side_feature_mean = side_feature_mean.copy(); side_feature_std = side_feature_std.copy()
                side_feature_mean[2:] = 0.0; side_feature_std[2:] = 1.0
            # Explicitly assert that the fitter received the 13 inner-train
            # names; this prevents accidental use of the full outer source set.
            if tuple(raw_features) != tuple(partition.inner_train_sessions):
                raise RuntimeError("Carrier statistics input order/scope is not inner-train-only")
            self.train_dataset.set_native_k4_normalization(side_feature_mean, side_feature_std)
            self.native_k4_normalization = {
                "fit_scope": "inner_train_sessions_only",
                "fit_sessions": list(partition.inner_train_sessions),
                "excluded_inner_validation_session": partition.inner_validation_session,
                "excluded_outer_target_session": partition.outer_target_session,
                "feature_group": self._feature_group,
                "mean": side_feature_mean.copy(),
                "std": side_feature_std.copy(),
            }

        self.val_inner_dataset = FalconDataset(
            sessions_dict=val_dict,
            calib_sessions_dict=val_dict,
            split="rt_nested_inner_validation",
            side_feature_mean=side_feature_mean,
            side_feature_std=side_feature_std,
            **ds_kwargs,
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
        expected_batches = int(self.hparams.session_window_budget) // int(self.hparams.batch_size)
        if set(self.train_batch_sampler.session_batch_counts.values()) != {expected_batches}:
            raise RuntimeError(
                "Nested RT source sampler did not allocate equal exposure to every inner-train session: "
                f"{self.train_batch_sampler.session_batch_counts}"
            )
        if not self.val_inner_batch_sampler.session_batch_counts:
            raise RuntimeError("Nested RT inner validation produced no eligible query windows")
        self.loaded_sessions = all_loaded
        self.session_names = list(partition.all_sessions)
        self.train_session_names = list(partition.inner_train_sessions)
        self.val_session_name = partition.inner_validation_session
        self._setup_complete = True
        logger.info(
            "RT clean nested LOSO fold %d: outer target=%s (not loaded), inner train=%s, inner val=%s",
            self._outer_fold,
            partition.outer_target_session,
            list(partition.inner_train_sessions),
            partition.inner_validation_session,
        )

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
            "RT clean nested LOSO has no fit-time test loader; run the explicit one-shot "
            "outer-target evaluator after checkpoint selection"
        )

    @property
    def outer_target_loaded(self) -> bool:
        return bool(self._outer_target_loader_opened)

    @property
    def outer_target_query_labels_read(self) -> bool:
        return bool(self._outer_target_query_labels_read)

    def get_split_manifest(self) -> dict[str, Any]:
        """Return a receipt that proves the target was absent from fit."""

        self.setup("fit")
        partition = self.split
        arm = _RT_ARM_SPECS[self._feature_group]
        source_k4_audits = (
            {name: self.train_dataset.k4_audits[name] for name in self.train_session_names}
            if self._feature_group in (_K4_FEATURE_GROUPS | _T4D_FEATURE_GROUPS)
            else {}
        )
        inner_k4_audits = (
            dict(self.val_inner_dataset.k4_audits)
            if self._feature_group in (_K4_FEATURE_GROUPS | _T4D_FEATURE_GROUPS)
            else {}
        )
        source_t4d_audits = (
            {name: self.loaded_sessions[name]["t4d_audit"] for name in self.train_session_names}
            if self._feature_group in _T4D_FEATURE_GROUPS else {}
        )
        inner_t4d_audits = (
            {name: self.loaded_sessions[name]["t4d_audit"] for name in (partition.inner_validation_session,)}
            if self._feature_group in _T4D_FEATURE_GROUPS else {}
        )
        xls_v2_permutation_sha256: dict[str, str] = {}
        if self._feature_group == "afc4_xls_v2":
            for name, audit in {**source_k4_audits, **inner_k4_audits}.items():
                permutation_sha = audit.get("label_permutation_sha256")
                if not isinstance(permutation_sha, str) or len(permutation_sha) != 64:
                    raise RuntimeError(f"XLSv2 permutation SHA absent for {name}")
                if audit.get("xls_v2_support_audit_sha256") != XLS_V2_SUPPORT_AUDIT_SHA256:
                    raise RuntimeError(f"XLSv2 support-audit SHA drift for {name}")
                xls_v2_permutation_sha256[name] = permutation_sha
        return {
            "protocol": RT_PROTOCOL,
            "gates": RT_GATES,
            "task": "rt",
            "development_only": True,
            "formal_heldout_opened": False,
            "validation_protocol": "nested_loso",
            "outer_loso_fold": int(partition.outer_fold),
            "loso_fold": int(partition.outer_fold),
            "arm": arm,
            "requested_side_feature_group": self._feature_group,
            "nested_selection": {
                "clean": True,
                "inner_selection_rule": "sorted_sessions[(outer_fold + 1) % session_count]",
                "outer_target_loaded_during_fit": False,
                "outer_target_query_labels_read_during_fit": False,
                "inner_validation_only_for_checkpoint_selection": True,
                "checkpoint_metric": "val_heldin/r2_mean",
                "checkpoint_metric_scope": "inner_validation_session_only",
            },
            "calibration": {
                "budget_trials": int(self.hparams.calibration_n_trials),
                "trial_index_range": [0, int(self.hparams.calibration_n_trials)],
                "target_calibration_optimizer_steps": 0,
                "estimator": "closed_form_raw_rate_OLS",
            },
            "query": {
                "query_start_trial": int(self._query_start_trial),
                "full_window_after_support_required": True,
                "window_size_bins": int(self.hparams.window_size),
                "event_qualified_query_endpoint": True,
            },
            "source_sessions": list(partition.outer_source_sessions),
            "outer_source_sessions": list(partition.outer_source_sessions),
            "inner_train_sessions": list(partition.inner_train_sessions),
            "inner_validation_session": partition.inner_validation_session,
            "target_session": partition.outer_target_session,
            "session_names": list(partition.all_sessions),
            "all_sessions": list(partition.all_sessions),
            "target_session_loaded_during_fit": False,
            "target_query_window_audit": None,
            "session_count": len(partition.all_sessions),
            "loaded_fit_sessions": list(self.loaded_sessions),
            "source_query_window_audit": self.train_dataset.query_window_audit,
            "inner_validation_query_window_audit": self.val_inner_dataset.query_window_audit,
            "rt_event_segment_audit": {
                name: self.loaded_sessions[name]["rt_segment_audit"] for name in self.loaded_sessions
            },
            "m24_event_support_audit": {
                name: summarize_rt_trial_budget(
                    self.loaded_sessions[name]["rt_segment_audit"],
                    budget_trials=int(self.hparams.calibration_n_trials),
                )
                for name in self.loaded_sessions
            },
            "rt_velocity_audit": {
                name: self.loaded_sessions[name]["rt_velocity_audit"] for name in self.loaded_sessions
            },
            "source_k4_calibration_audit": source_k4_audits,
            "inner_validation_k4_calibration_audit": inner_k4_audits,
            "source_t4d_access_audit": source_t4d_audits,
            "inner_validation_t4d_access_audit": inner_t4d_audits,
            "source_only_normalizer": _normalizer_jsonable(self.native_k4_normalization),
            "xls_v2_support_audit": (
                {
                    "path": str(Path(self.hparams.xls_v2_support_audit_path).resolve()),
                    "sha256": XLS_V2_SUPPORT_AUDIT_SHA256,
                    "mode_required": "0444",
                    "per_session_permutation_sha256": xls_v2_permutation_sha256,
                    "query_labels_available_to_generator": False,
                    "common_inverse_or_alignment_map": False,
                }
                if self._feature_group == "afc4_xls_v2"
                else None
            ),
            "carrier_transform_fit_sessions": list(partition.inner_train_sessions)
            if self._feature_group in (_K4_FEATURE_GROUPS | _T4D_FEATURE_GROUPS)
            else [],
            "pca_enabled": False,
            "pca_fit_sessions": [],
            "source_sampler": {
                "session_balanced_sampling": bool(self.hparams.session_balanced_sampling),
                "window_budget_per_session": int(self.hparams.session_window_budget),
                "batch_size": int(self.hparams.batch_size),
                "batches_per_inner_train_session": dict(self.train_batch_sampler.session_batch_counts),
                "available_batches_per_inner_train_session_before_budget": dict(
                    self.train_batch_sampler.original_session_batch_counts
                ),
                "reshuffle_each_epoch": bool(self.hparams.sampler_reshuffle_each_epoch),
                "sampler_seed": int(self.hparams.sampler_seed),
            },
            "decoder_transfer": {
                "decoder_source": "FALCON-M2 checkpoint initialization for RT source fit",
                "decoder_joint_retrained_during_fit": True,
                "rt_native_decoder": False,
                "target_session_backpropagation": False,
            },
        }


def build_outer_target_dataset(
    *,
    data_dir: str | Path,
    outer_loso_fold: int,
    side_feature_group: str,
    side_feature_shuffle_seed: int,
    calibration_n_trials: int = 24,
    query_start_trial: int | None = None,
    window_size: int = 50,
    max_trial_length: int = 100,
    interpolate_trials: bool = True,
    interpolate_trials_kind: str = "cubic",
    pad_value: float = -1.0,
    expected_session_count: int = RT_EXPECTED_SESSION_COUNT,
    side_feature_mean: np.ndarray | None = None,
    side_feature_std: np.ndarray | None = None,
    xls_v2_support_audit_path: str | Path | None = None,
) -> tuple[FalconDataset, NestedRtSplit, Path]:
    """Open exactly one outer target for the post-selection evaluator.

    The caller must provide a normalizer fitted by the 13 inner-train
    sessions.  This function intentionally does not fit any transform from the
    target payload.
    """

    if int(calibration_n_trials) != 24:
        raise ValueError("RT outer evaluator is frozen to M24 support")
    effective_query = int(calibration_n_trials) if query_start_trial is None else int(query_start_trial)
    if effective_query != int(calibration_n_trials):
        raise ValueError("outer target query_start_trial must equal calibration_n_trials")
    indexed = index_rt_session_paths(data_dir, expected_session_count=expected_session_count)
    split = nested_loso_partition(tuple(indexed), int(outer_loso_fold), expected_session_count=expected_session_count)
    target_path = indexed[split.outer_target_session]
    # This is the explicit post-selection boundary: target labels are opened
    # only here, never through RtNestedLossoDataModule.setup.
    raw = (load_rt_sparse_endpoint_t4d_session(target_path)
           if str(side_feature_group).lower() in _T4D_FEATURE_GROUPS else load_rt_session(target_path))
    if str(raw.get("session_name")) != split.outer_target_session:
        raise RuntimeError("Outer evaluator target name disagrees with indexed path")
    group = str(side_feature_group).lower()
    if group in (_K4_FEATURE_GROUPS | _T4D_FEATURE_GROUPS) and (side_feature_mean is None or side_feature_std is None):
        raise ValueError("Outer AFC4 evaluator requires the inner-train-only normalizer")
    hparams = {
        "window_size": int(window_size),
        "calibration_n_trials": int(calibration_n_trials),
        "max_trial_length": int(max_trial_length),
        "interpolate_trials": bool(interpolate_trials),
        "interpolate_trials_kind": str(interpolate_trials_kind),
        "pad_value": float(pad_value),
        "side_feature_shuffle_seed": int(side_feature_shuffle_seed),
        "xls_v2_support_audit_path": (
            None if xls_v2_support_audit_path is None else str(xls_v2_support_audit_path)
        ),
    }
    ds = FalconDataset(
        sessions_dict=OrderedDict([(split.outer_target_session, raw)]),
        calib_sessions_dict=OrderedDict([(split.outer_target_session, raw)]),
        split="rt_nested_outer_target_one_shot",
        side_feature_mean=side_feature_mean,
        side_feature_std=side_feature_std,
        **_dataset_kwargs(hparams, group, effective_query),
    )
    return ds, split, target_path


# Both spellings are kept as import aliases: ``LOSO`` is the protocol's usual
# acronym, while the Hydra target above uses the PEP-8-friendly historical
# ``Losso`` spelling.  They resolve to the exact same fail-closed class.
RtNestedLOSODataModule = RtNestedLossoDataModule
