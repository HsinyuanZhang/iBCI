"""Direct-standardized Z4 data path for the shared C1 source program.

This module exists because the historical generic ``z4`` token is *not* an
appropriate control for the new shared-weight C1 experiment.  In that legacy
path the loader first constructs a raw T4 descriptor from target-direction
labels and trial rates, applies the source T4 normalizer, and only then masks
the result to zero.  That is numerically a zero input, but it still performs a
label/rate fit on the evaluation session.

The program below has a deliberately smaller contract:

* the underlying :class:`Dandi688MultiSessionDataModule` is created with
  ``side_feature_group=None``;
* after the ordinary neural/activity record is materialized, this module
  directly writes ``np.zeros((N, 4), dtype=np.float32)`` into the already
  standardized B3S side coordinate;
* no raw T4 vector exists and no side-feature normalizer arithmetic is
  performed by this module;
* the source-only T4 coordinate is a *provenance reference* to the shared-T4
  program's two source normalizer hashes, not an operation on target data.

The ordinary data loader still reads neural activity and behavioural targets
needed for the decoding task.  The counters below deliberately say only that
the *descriptor* reads zero target-direction labels and makes zero T4 trial
rate fits; they do not incorrectly claim that the decoder has no behaviour
target exposure at all.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mc_maze.multisession_datamodule import (
    Dandi688MultiSessionDataModule,
    Dandi688MultiSessionDataset,
    SessionRecord,
)
from mc_maze.paired_view_c1 import PairedViewC1DataModule


SIDE_DIM = 4
STANDARDIZED_COORDINATE_NAME = "source_only_t4_standardized_coordinate"
DIRECT_ZERO_CONSTRUCTION = (
    "np.zeros((N,4), dtype=np.float32) directly in standardized coordinate; "
    "no raw T4 descriptor and no normalizer arithmetic"
)


class SharedZero4ContractError(ValueError):
    """Raised when the label-free, direct-standardized Z4 contract is broken."""


@dataclass(frozen=True)
class Zero4DescriptorReceipt:
    """Serializable declaration of what the Z4 descriptor path did and did not do."""

    split: str
    view: str
    session_count: int
    channel_count: int
    side_dim: int
    construction: str
    target_direction_label_reads_for_descriptor: int
    t4_trial_rate_reads_for_descriptor: int
    target_t4_rate_fit_calls: int
    raw_t4_constructed: bool
    source_t4_normalizer_arithmetic_performed: bool
    exact_float32_zero: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "view": self.view,
            "session_count": self.session_count,
            "channel_count": self.channel_count,
            "side_dim": self.side_dim,
            "coordinate": STANDARDIZED_COORDINATE_NAME,
            "construction": self.construction,
            "target_direction_label_reads_for_descriptor": self.target_direction_label_reads_for_descriptor,
            "t4_trial_rate_reads_for_descriptor": self.t4_trial_rate_reads_for_descriptor,
            "label_access_scope": "descriptor_only",
            "target_t4_rate_fit_calls": self.target_t4_rate_fit_calls,
            "raw_t4_constructed": self.raw_t4_constructed,
            "source_t4_normalizer_arithmetic_performed": self.source_t4_normalizer_arithmetic_performed,
            "exact_float32_zero": self.exact_float32_zero,
        }


def standardized_zero4(channel_count: int) -> np.ndarray:
    """Return the neutral B3S/T4-coordinate value directly, bitwise zero.

    ``channel_count`` is intentionally the only datum this construction
    consumes.  In particular, it receives neither an NWB path nor a trial
    table, target direction, rate, raw descriptor, mean, or standard
    deviation.  That makes raw-zero-then-standardize impossible by API.
    """

    # Do not accept numerically integral floats: accepting and coercing them
    # would hide a malformed evaluator record before the shape cross-check.
    if isinstance(channel_count, bool) or not isinstance(channel_count, (int, np.integer)):
        raise SharedZero4ContractError("channel_count must be an integer")
    count = int(channel_count)
    if count < 0:
        raise SharedZero4ContractError("channel_count must be non-negative")
    return np.zeros((count, SIDE_DIM), dtype=np.float32)


def _zero_float32_bits(values: np.ndarray) -> bool:
    """Require positive IEEE float32 zero bytes, not merely numeric equality."""

    array = np.asarray(values)
    return (
        array.dtype == np.dtype(np.float32)
        and array.ndim == 2
        and array.shape[1] == SIDE_DIM
        and bool(np.all(array.view(np.uint32) == np.uint32(0)))
    )


def require_standardized_zero4(values: np.ndarray, *, context: str) -> None:
    if not _zero_float32_bits(values):
        array = np.asarray(values)
        raise SharedZero4ContractError(
            f"{context}: expected bitwise float32 standardized Z4 [N,{SIDE_DIM}], "
            f"got dtype={array.dtype}, shape={array.shape}"
        )


def _attach_zero4_record(record: SessionRecord, *, split: str) -> SessionRecord:
    """Attach direct Z4 without consulting any descriptor feature function."""

    if record.side_features is not None:
        raise SharedZero4ContractError(
            f"{split}/{record.name}: base record already has side features; "
            "shared_zero4 requires side_feature_group=None before direct attachment"
        )
    if record.neural.ndim != 2:
        raise SharedZero4ContractError(
            f"{split}/{record.name}: expected neural [time,N], got {record.neural.shape}"
        )
    zero4 = standardized_zero4(record.neural.shape[1])
    require_standardized_zero4(zero4, context=f"{split}/{record.name}")
    return replace(record, side_features=zero4)


def attach_standardized_zero4_to_dataset(
    dataset: Dandi688MultiSessionDataset,
    *,
    split: str,
) -> dict[str, Any]:
    """Mutate only a newly materialized dataset's record mapping with direct Z4.

    ``SessionRecord`` itself is frozen, so each replacement retains all base
    neural/behaviour/trial axes unchanged.  The function does not receive an
    NWB path or a normalizer and has no route to a T4 fitter.
    """

    if not dataset.sessions:
        raise SharedZero4ContractError(f"{split}: cannot attach Z4 to no sessions")
    rows: dict[str, Any] = {}
    for session_name, record in tuple(dataset.sessions.items()):
        if str(session_name) != record.name:
            raise SharedZero4ContractError(
                f"{split}: session mapping key/name mismatch: {session_name!r}/{record.name!r}"
            )
        updated = _attach_zero4_record(record, split=split)
        dataset.sessions[session_name] = updated
        rows[session_name] = {
            "channels": int(updated.neural.shape[1]),
            "side_shape": list(updated.side_features.shape),
            "side_dtype": str(updated.side_features.dtype),
            "bitwise_float32_zero": _zero_float32_bits(updated.side_features),
        }
    if not all(row["bitwise_float32_zero"] for row in rows.values()):
        raise SharedZero4ContractError(f"{split}: direct Z4 row audit failed")
    return {
        "split": split,
        "coordinate": STANDARDIZED_COORDINATE_NAME,
        "construction": DIRECT_ZERO_CONSTRUCTION,
        "sessions": rows,
        "session_count": len(rows),
        "channel_count": sum(row["channels"] for row in rows.values()),
        "side_dim": SIDE_DIM,
        "target_direction_label_reads_for_descriptor": 0,
        "t4_trial_rate_reads_for_descriptor": 0,
        "label_access_scope": "descriptor_only",
        "target_t4_rate_fit_calls": 0,
        "raw_t4_constructed": False,
        "source_t4_normalizer_arithmetic_performed": False,
        "all_records_bitwise_float32_zero": True,
    }


def attach_standardized_zero4_to_evaluation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return an evaluator record with direct Z4, without inspecting trials/labels.

    The evaluator needs the record's ``trials`` later to choose chronology and
    query windows.  This descriptor attachment intentionally reads only
    ``n_units`` and copies the mapping; it never accesses ``trials``, target
    directions, rates, a normalizer, or an NWB path.
    """

    if "n_units" not in record or "neural" not in record:
        raise SharedZero4ContractError("evaluation record is missing n_units or neural")
    n_units = record["n_units"]
    neural = np.asarray(record["neural"])
    if neural.ndim != 2:
        raise SharedZero4ContractError(
            f"evaluation record neural must be [time,N], got {neural.shape}"
        )
    # In pseudo-MUA, ``source_unit_count`` can exceed this value.  The B3S
    # side rows must match the *pooled channel* axis held in ``n_units``.
    if not isinstance(n_units, (int, np.integer)) or isinstance(n_units, bool):
        raise SharedZero4ContractError("evaluation record n_units must be an integer")
    if int(n_units) != int(neural.shape[1]):
        raise SharedZero4ContractError(
            f"evaluation record n_units={n_units} does not match neural channels={neural.shape[1]}"
        )
    zero4 = standardized_zero4(n_units)
    require_standardized_zero4(zero4, context="evaluation record")
    updated = dict(record)
    updated["side_features"] = zero4
    updated["zero4_descriptor_receipt"] = {
        "coordinate": STANDARDIZED_COORDINATE_NAME,
        "construction": DIRECT_ZERO_CONSTRUCTION,
        "target_direction_label_reads_for_descriptor": 0,
        "t4_trial_rate_reads_for_descriptor": 0,
        "label_access_scope": "descriptor_only",
        "target_t4_rate_fit_calls": 0,
        "raw_t4_constructed": False,
        "source_t4_normalizer_arithmetic_performed": False,
        "bitwise_float32_zero": True,
    }
    return updated


def require_zero4_batch(batch: Sequence[Any], *, view: str, context: str) -> dict[str, Any]:
    """Check every B3S side tensor in a collated batch at the bit level."""

    if len(batch) != 5:
        raise SharedZero4ContractError(
            f"{context}/{view}: Z4 batch must have five items (neural,target,calib,session,side)"
        )
    side = batch[4]
    if not isinstance(side, torch.Tensor):
        raise SharedZero4ContractError(f"{context}/{view}: side is not a tensor")
    if side.dtype != torch.float32 or side.ndim != 3 or side.shape[-1] != SIDE_DIM:
        raise SharedZero4ContractError(
            f"{context}/{view}: expected float32 [B,N,{SIDE_DIM}] side, got {side.dtype}/{tuple(side.shape)}"
        )
    cpu = side.detach().cpu().contiguous()
    # ``torch.equal(..., zeros_like(...))`` alone would accept negative zero.
    raw = cpu.view(torch.uint8)
    if int(torch.count_nonzero(raw).item()) != 0:
        raise SharedZero4ContractError(f"{context}/{view}: side contains non-zero bits")
    return {
        "view": view,
        "batch_size": int(side.shape[0]),
        "channels": int(side.shape[1]),
        "side_dim": SIDE_DIM,
        "dtype": "float32",
        "bitwise_float32_zero": True,
    }


class SharedZero4PairedDataModule:
    """C1 paired loader whose only side input is direct-standardized Z4.

    The ordinary pair wrapper still performs the strong SUA/pseudo-MUA target,
    time-axis, and count-conservation checks.  This class adds the direct-Z4
    construction/audit before that wrapper sees a side tensor.
    """

    def __init__(
        self,
        sua: Dandi688MultiSessionDataModule,
        pseudo_mua: Dandi688MultiSessionDataModule,
    ) -> None:
        if sua.signal_view != "sua" or pseudo_mua.signal_view != "pseudo_mua":
            raise SharedZero4ContractError("requires one SUA and one pseudo-MUA base datamodule")
        if sua.side_feature_group is not None or pseudo_mua.side_feature_group is not None:
            raise SharedZero4ContractError(
                "shared_zero4 requires base datamodules with side_feature_group=None"
            )
        self.sua = sua
        self.pseudo_mua = pseudo_mua
        self._paired: PairedViewC1DataModule | None = None
        self.session_splits: dict[str, list[str]] = {}
        self._descriptor_receipts: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _require_split_dataset(
        datamodule: Dandi688MultiSessionDataModule,
        split: str,
    ) -> Dandi688MultiSessionDataset:
        dataset = datamodule.train_dataset if split == "train" else datamodule.val_dataset
        if dataset is None:
            raise SharedZero4ContractError(f"{split}: base dataset did not initialize")
        return dataset

    def setup(self) -> None:
        if self._paired is not None:
            return
        self.sua.setup("fit")
        self.pseudo_mua.setup("fit")
        if self.sua.session_splits != self.pseudo_mua.session_splits:
            raise SharedZero4ContractError("SUA/pseudo-MUA split drift before Z4 attachment")
        for split in ("train", "validation"):
            dm_split = "train" if split == "train" else "validation"
            sua_dataset = self._require_split_dataset(self.sua, "train" if split == "train" else "val")
            pseudo_dataset = self._require_split_dataset(
                self.pseudo_mua, "train" if split == "train" else "val"
            )
            self._descriptor_receipts[f"{split}/sua"] = attach_standardized_zero4_to_dataset(
                sua_dataset, split=f"{dm_split}/sua"
            )
            self._descriptor_receipts[f"{split}/pseudo_mua"] = attach_standardized_zero4_to_dataset(
                pseudo_dataset, split=f"{dm_split}/pseudo_mua"
            )
        self._paired = PairedViewC1DataModule(self.sua, self.pseudo_mua)
        self._paired.setup()
        self.session_splits = {
            split: list(names) for split, names in self._paired.session_splits.items()
        }

    def train_dataloader(self):
        if self._paired is None:
            raise SharedZero4ContractError("call setup() before train_dataloader()")
        return self._paired.train_dataloader()

    def exposure_receipt(self) -> dict[str, Any]:
        if self._paired is None:
            raise SharedZero4ContractError("call setup() before exposure_receipt()")
        return {
            "paired_exposure": self._paired.exposure_receipt(),
            "descriptor_construction": dict(self._descriptor_receipts),
            "descriptor_contract": {
                "coordinate": STANDARDIZED_COORDINATE_NAME,
                "construction": DIRECT_ZERO_CONSTRUCTION,
                "target_direction_label_reads_for_descriptor": 0,
                "t4_trial_rate_reads_for_descriptor": 0,
                "label_access_scope": "descriptor_only",
                "target_t4_rate_fit_calls": 0,
                "raw_t4_constructed": False,
                "source_t4_normalizer_arithmetic_performed": False,
            },
        }

    def audit_all_train_batches(self) -> dict[str, Any]:
        """Materialize every source training batch before optimizer construction.

        This is deliberately a no-model/no-gradient audit.  A fresh loader is
        created for later fitting, so consuming this deterministic audit loader
        cannot change support sampling or training order.
        """

        if self._paired is None:
            raise SharedZero4ContractError("call setup() before audit_all_train_batches()")
        # SessionBatchSampler fixes the full index order at construction.  Hash
        # it before materialization and compare against a newly-created formal
        # training loader after the audit.  This guards against a future loader
        # change where reading the audit could perturb random state/order.
        def sampler_digest(loader: Any) -> str:
            batches = [list(map(int, batch)) for batch in loader.batch_sampler]
            encoded = json.dumps(batches, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        audit_loader = self._paired.train_dataloader()
        audit_order_sha256 = sampler_digest(audit_loader)
        per_view = {"sua": 0, "pseudo_mua": 0}
        batch_count = 0
        sample_count = 0
        for batch_count, pair in enumerate(audit_loader, start=1):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise SharedZero4ContractError("paired Z4 loader produced an invalid pair")
            sua_batch, pseudo_batch = pair
            sua_row = require_zero4_batch(sua_batch, view="sua", context=f"train_batch_{batch_count}")
            pseudo_row = require_zero4_batch(
                pseudo_batch, view="pseudo_mua", context=f"train_batch_{batch_count}"
            )
            if sua_row["batch_size"] != pseudo_row["batch_size"]:
                raise SharedZero4ContractError("paired Z4 batch sizes differ")
            per_view["sua"] += sua_row["batch_size"]
            per_view["pseudo_mua"] += pseudo_row["batch_size"]
            sample_count += sua_row["batch_size"]
        if batch_count == 0:
            raise SharedZero4ContractError("Z4 source training loader is empty")
        if per_view["sua"] != sample_count or per_view["pseudo_mua"] != sample_count:
            raise SharedZero4ContractError("Z4 all-batch audit sample accounting drift")
        formal_train_loader = self._paired.train_dataloader()
        formal_order_sha256 = sampler_digest(formal_train_loader)
        if audit_order_sha256 != formal_order_sha256:
            raise SharedZero4ContractError(
                "all-batch zero audit changed the formal source-training sampler order"
            )
        return {
            "scope": "all_source_training_batches_before_model_or_optimizer",
            "batch_count": batch_count,
            "paired_sample_count": sample_count,
            "views": {
                "sua": {"sample_count": per_view["sua"], "bitwise_float32_zero": True},
                "pseudo_mua": {
                    "sample_count": per_view["pseudo_mua"],
                    "bitwise_float32_zero": True,
                },
            },
            "target_direction_label_reads_for_descriptor": 0,
            "t4_trial_rate_reads_for_descriptor": 0,
            "label_access_scope": "descriptor_only",
            "target_t4_rate_fit_calls": 0,
            "model_constructed": False,
            "optimizer_constructed": False,
            "backward_called": False,
            "audit_batch_index_order_sha256": audit_order_sha256,
            "formal_train_batch_index_order_sha256": formal_order_sha256,
            "audit_preserves_formal_train_batch_index_order": True,
        }
