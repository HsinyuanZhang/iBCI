"""Minimal paired SUA/pseudo-MUA data contract for C1.

This module intentionally contains no descriptor fitting and no model-side
fusion.  The two views remain two independently constructed
``Dandi688MultiSessionDataModule`` instances: that is what keeps their
train-only normalizers and on-disk cache namespaces separate.  C1 only adds a
strict zipper over their *already materialized* window datasets.

The zipper is deliberately stricter than matching session names.  It requires
the complete ``(session_name, window_start)`` sequence to be identical, and
the training module additionally checks that every collated pair has the same
session names and exactly equal behaviour target tensor.  The neural tensors
are allowed to have a different last dimension because pseudo-MUA is the
deterministic electrode pooling of sorted SUA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from mc_maze.multisession_datamodule import (
    Dandi688MultiSessionDataModule,
    Dandi688MultiSessionDataset,
    SessionBatchSampler,
)


class PairedViewContractError(ValueError):
    """Raised when SUA and pseudo-MUA cannot be used as one paired exposure."""


@dataclass(frozen=True)
class PairedExposureReceipt:
    """Small, serializable statement of the view-pairing invariant."""

    split: str
    sample_count: int
    batch_count: int
    session_window_counts: dict[str, int]
    paired_loss_weights: tuple[float, float]
    lambda_consistency: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "sample_count": self.sample_count,
            "batch_count": self.batch_count,
            "session_window_counts": dict(self.session_window_counts),
            "paired_loss_weights": list(self.paired_loss_weights),
            "lambda_consistency": self.lambda_consistency,
        }


def _window_sequence(dataset: Dandi688MultiSessionDataset) -> tuple[tuple[str, int], ...]:
    return tuple((str(name), int(start)) for name, start in dataset.window_indices)


def _require_equal_window_sequences(
    sua_dataset: Dandi688MultiSessionDataset,
    pseudo_mua_dataset: Dandi688MultiSessionDataset,
    *,
    split: str,
) -> tuple[tuple[str, int], ...]:
    sua_indices = _window_sequence(sua_dataset)
    pseudo_indices = _window_sequence(pseudo_mua_dataset)
    if sua_indices != pseudo_indices:
        if len(sua_indices) != len(pseudo_indices):
            detail = f"length SUA={len(sua_indices)} pseudo-MUA={len(pseudo_indices)}"
        else:
            mismatch = next(
                index
                for index, (sua_item, pseudo_item) in enumerate(zip(sua_indices, pseudo_indices))
                if sua_item != pseudo_item
            )
            detail = (
                f"first mismatch index={mismatch}: SUA={sua_indices[mismatch]!r}, "
                f"pseudo-MUA={pseudo_indices[mismatch]!r}"
            )
        raise PairedViewContractError(
            f"{split}: paired view window indices differ ({detail})"
        )
    return sua_indices


def _require_equal_record_axes(
    sua_dataset: Dandi688MultiSessionDataset,
    pseudo_mua_dataset: Dandi688MultiSessionDataset,
    *,
    split: str,
) -> dict[str, Any]:
    """Audit the full session/time/target axes before exposing a paired loader.

    The pseudo-MUA construction is allowed to change only the neural channel
    axis.  Behaviour, valid window starts, calibration trial/time axes, and
    total binned counts must remain aligned.  The total-count check complements
    the exact electrode-membership tests around ``pool_spikes_by_electrode``;
    it catches stale or view-crossed caches on real materialized records.
    """
    if tuple(sua_dataset.sessions) != tuple(pseudo_mua_dataset.sessions):
        raise PairedViewContractError(
            f"{split}: SUA/pseudo-MUA session order differs"
        )
    session_rows: dict[str, Any] = {}
    for session_name, sua_record in sua_dataset.sessions.items():
        pseudo_record = pseudo_mua_dataset.sessions[session_name]
        if sua_record.signal_view != "sua" or pseudo_record.signal_view != "pseudo_mua":
            raise PairedViewContractError(
                f"{split}/{session_name}: signal-view identity mismatch"
            )
        if not np.array_equal(sua_record.valid_starts, pseudo_record.valid_starts):
            raise PairedViewContractError(
                f"{split}/{session_name}: valid time/window indices differ"
            )
        if not np.array_equal(sua_record.behavior, pseudo_record.behavior):
            raise PairedViewContractError(
                f"{split}/{session_name}: normalized behaviour targets differ"
            )
        if sua_record.neural.shape[0] != pseudo_record.neural.shape[0]:
            raise PairedViewContractError(
                f"{split}/{session_name}: neural time axes differ"
            )
        if sua_record.calib_trials.shape[:2] != pseudo_record.calib_trials.shape[:2]:
            raise PairedViewContractError(
                f"{split}/{session_name}: calibration trial/time axes differ"
            )
        if pseudo_record.neural.shape[1] > sua_record.neural.shape[1]:
            raise PairedViewContractError(
                f"{split}/{session_name}: pooling increased the channel count"
            )
        neural_error = float(
            np.max(
                np.abs(
                    sua_record.neural.sum(axis=1, dtype=np.float64)
                    - pseudo_record.neural.sum(axis=1, dtype=np.float64)
                )
            )
        )
        calibration_error = float(
            np.max(
                np.abs(
                    sua_record.calib_trials.sum(axis=2, dtype=np.float64)
                    - pseudo_record.calib_trials.sum(axis=2, dtype=np.float64)
                )
            )
        )
        if neural_error > 1e-6:
            raise PairedViewContractError(
                f"{split}/{session_name}: electrode pooling violates online count "
                f"conservation (max error {neural_error})"
            )
        if calibration_error > 1e-4:
            raise PairedViewContractError(
                f"{split}/{session_name}: electrode pooling violates calibration count "
                f"conservation (max error {calibration_error})"
            )
        if sua_record.side_features is None or pseudo_record.side_features is None:
            raise PairedViewContractError(
                f"{split}/{session_name}: C1 requires a T4/TS4 descriptor in both views"
            )
        if (
            sua_record.side_features.shape != (sua_record.neural.shape[1], 4)
            or pseudo_record.side_features.shape != (pseudo_record.neural.shape[1], 4)
        ):
            raise PairedViewContractError(
                f"{split}/{session_name}: descriptor rows do not match view channels"
            )
        if not (
            np.isfinite(sua_record.side_features).all()
            and np.isfinite(pseudo_record.side_features).all()
        ):
            raise PairedViewContractError(
                f"{split}/{session_name}: descriptor contains non-finite values"
            )
        session_rows[session_name] = {
            "time_bins": int(sua_record.neural.shape[0]),
            "source_units": int(sua_record.neural.shape[1]),
            "pooled_channels": int(pseudo_record.neural.shape[1]),
            "window_count": int(sua_record.valid_starts.size),
            "max_online_count_error": neural_error,
            "max_calibration_count_error": calibration_error,
        }
    return {
        "session_count": len(session_rows),
        "sessions": session_rows,
        "behavior_targets_bitwise_equal": True,
        "valid_time_indices_bitwise_equal": True,
        "count_conservation_checked": True,
    }


def _session_window_counts(indices: Iterable[tuple[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for session_name, _ in indices:
        counts[session_name] = counts.get(session_name, 0) + 1
    return counts


def validate_pair_batch(
    sua_batch: Sequence[Any], pseudo_mua_batch: Sequence[Any]
) -> None:
    """Prove the per-step common target/session/time-axis invariant.

    Batch components follow ``Dandi688MultiSessionDataset``'s current public
    tuple contract.  There may be five or six components depending on whether
    electrode ids are present; C1's ordinary T4/TS4 path currently has five.
    The validation remains general so a future tested data-loader change fails
    loudly rather than silently removing the pairing guarantee.
    """
    if len(sua_batch) not in {4, 5, 6} or len(pseudo_mua_batch) not in {4, 5, 6}:
        raise PairedViewContractError(
            "paired batch must contain the standard 4/5/6-item session batch tuples"
        )
    sua_neural, sua_target, sua_calib, sua_sessions = sua_batch[:4]
    pseudo_neural, pseudo_target, pseudo_calib, pseudo_sessions = pseudo_mua_batch[:4]
    if tuple(sua_sessions) != tuple(pseudo_sessions):
        raise PairedViewContractError(
            f"paired batch session names differ: SUA={tuple(sua_sessions)!r}, "
            f"pseudo-MUA={tuple(pseudo_sessions)!r}"
        )
    if not torch.equal(sua_target, pseudo_target):
        raise PairedViewContractError(
            "paired batch behaviour targets differ; views are not exposed to the same target/time indices"
        )
    if sua_neural.ndim != 3 or pseudo_neural.ndim != 3:
        raise PairedViewContractError("paired neural inputs must have shape [B,T,N]")
    if sua_neural.shape[:2] != pseudo_neural.shape[:2]:
        raise PairedViewContractError(
            "paired neural batch/time axes differ; pseudo-MUA pooling may change N only"
        )
    if sua_calib.ndim != 4 or pseudo_calib.ndim != 4:
        raise PairedViewContractError("paired calibration inputs must have shape [B,Q,T,N]")
    if sua_calib.shape[:3] != pseudo_calib.shape[:3]:
        raise PairedViewContractError(
            "paired calibration batch/trial/time axes differ; pseudo-MUA pooling may change N only"
        )
    if sua_target.shape != pseudo_target.shape:
        raise PairedViewContractError("paired targets must have equal shape")


class PairedViewDataset(Dataset):
    """Zip two view datasets after a complete alignment audit."""

    def __init__(
        self,
        sua_dataset: Dandi688MultiSessionDataset,
        pseudo_mua_dataset: Dandi688MultiSessionDataset,
        *,
        split: str,
    ) -> None:
        self.sua_dataset = sua_dataset
        self.pseudo_mua_dataset = pseudo_mua_dataset
        self.split = split
        self.window_indices = _require_equal_window_sequences(
            sua_dataset, pseudo_mua_dataset, split=split
        )
        self.axis_receipt = _require_equal_record_axes(
            sua_dataset, pseudo_mua_dataset, split=split
        )

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.sua_dataset[index], self.pseudo_mua_dataset[index]


def paired_collate(samples: list[tuple[Any, Any]]) -> tuple[Sequence[Any], Sequence[Any]]:
    """Collate both views independently, then validate their common exposure."""
    if not samples:
        raise PairedViewContractError("cannot collate an empty paired-view batch")
    sua_samples, pseudo_samples = zip(*samples)
    sua_batch = default_collate(list(sua_samples))
    pseudo_batch = default_collate(list(pseudo_samples))
    validate_pair_batch(sua_batch, pseudo_batch)
    return sua_batch, pseudo_batch


class PairedViewC1DataModule:
    """A strict paired wrapper around separate SUA and pseudo-MUA data modules.

    It intentionally is not a LightningDataModule subclass.  Training C1 only
    needs its paired train loader and keeping this wrapper thin prevents a new
    loader hierarchy from becoming a second preprocessing implementation.
    """

    def __init__(
        self,
        sua: Dandi688MultiSessionDataModule,
        pseudo_mua: Dandi688MultiSessionDataModule,
        *,
        paired_loss_weights: tuple[float, float] = (0.5, 0.5),
        lambda_consistency: float = 0.0,
    ) -> None:
        if paired_loss_weights != (0.5, 0.5):
            raise PairedViewContractError(
                "C1 fixes equal task-loss weights exactly to (0.5, 0.5)"
            )
        if lambda_consistency != 0.0:
            raise PairedViewContractError(
                "C1 is lambda=0 only; consistency is conditional C2 work"
            )
        if sua.signal_view != "sua" or pseudo_mua.signal_view != "pseudo_mua":
            raise PairedViewContractError(
                "Paired C1 requires one SUA datamodule and one pseudo-MUA datamodule"
            )
        self.sua = sua
        self.pseudo_mua = pseudo_mua
        self.paired_loss_weights = paired_loss_weights
        self.lambda_consistency = lambda_consistency
        self.train_dataset: PairedViewDataset | None = None
        self.val_dataset: PairedViewDataset | None = None
        self.session_splits: dict[str, list[str]] = {}

    def setup(self) -> None:
        self.sua.setup("fit")
        self.pseudo_mua.setup("fit")
        if self.sua.session_splits != self.pseudo_mua.session_splits:
            raise PairedViewContractError(
                "SUA and pseudo-MUA session splits differ; C1 cannot share exposure"
            )
        if self.sua.train_dataset is None or self.pseudo_mua.train_dataset is None:
            raise RuntimeError("paired C1 train datasets did not initialize")
        if self.sua.val_dataset is None or self.pseudo_mua.val_dataset is None:
            raise RuntimeError("paired C1 validation datasets did not initialize")
        self.session_splits = {
            split: list(names) for split, names in self.sua.session_splits.items()
        }
        self.train_dataset = PairedViewDataset(
            self.sua.train_dataset, self.pseudo_mua.train_dataset, split="train"
        )
        self.val_dataset = PairedViewDataset(
            self.sua.val_dataset, self.pseudo_mua.val_dataset, split="validation"
        )

    def _loader(self, dataset: PairedViewDataset, *, shuffle: bool) -> DataLoader:
        sampler = SessionBatchSampler(
            dataset.sua_dataset,
            batch_size=self.sua.batch_size,
            shuffle=shuffle,
            seed=self.sua.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.sua.num_workers,
            pin_memory=self.sua.pin_memory,
            collate_fn=paired_collate,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("call setup() before train_dataloader()")
        return self._loader(self.train_dataset, shuffle=True)

    def exposure_receipt(self) -> dict[str, dict[str, Any]]:
        if self.train_dataset is None or self.val_dataset is None:
            raise RuntimeError("call setup() before exposure_receipt()")
        result: dict[str, dict[str, Any]] = {}
        for split, dataset in (("train", self.train_dataset), ("validation", self.val_dataset)):
            sampler = SessionBatchSampler(
                dataset.sua_dataset,
                batch_size=self.sua.batch_size,
                shuffle=(split == "train"),
                seed=self.sua.seed,
            )
            receipt = PairedExposureReceipt(
                split=split,
                sample_count=len(dataset),
                batch_count=len(sampler),
                session_window_counts=_session_window_counts(dataset.window_indices),
                paired_loss_weights=self.paired_loss_weights,
                lambda_consistency=self.lambda_consistency,
            )
            result[split] = receipt.as_dict()
            result[split]["axis_alignment"] = dataset.axis_receipt
        return result


def normalizer_hashes_are_distinct(sua_hash: str | None, pseudo_mua_hash: str | None) -> bool:
    """Small receipt guard; absent/identical statistics are both contract failures."""
    return (
        isinstance(sua_hash, str)
        and isinstance(pseudo_mua_hash, str)
        and len(sua_hash) == 64
        and len(pseudo_mua_hash) == 64
        and sua_hash != pseudo_mua_hash
    )
