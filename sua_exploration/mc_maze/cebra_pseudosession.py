"""CEBRA-inspired, size-matched pseudo-session source augmentation.

The deployment/evaluation path remains the ordinary DANDI-688 multi-session
path.  Only source-training examples are changed.  A mixed example keeps the
anchor session's behaviour target and total unit count, while its unit set is
assembled from the anchor and behaviour-matched windows from other source
sessions.  Neural activity, calibration trials, and side features always move
together under exactly the same unit selection and final permutation.

This module deliberately contains no model or loss change.  It is compatible
with the existing permutation-invariant B3S T4/Z4 consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from mc_maze.multisession_datamodule import (
    Dandi688MultiSessionDataModule,
    Dandi688MultiSessionDataset,
    SessionRecord,
)


SCHEMA = "cebra_size_matched_pseudosession_v1"
MATCH_KIND = "continuous_last_bin_velocity_l2"


class PseudoSessionError(ValueError):
    """A pseudo-session constructibility or provenance invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PseudoSessionError(message)


def _stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ContributorPlan:
    session: str
    start: int
    unit_indices: np.ndarray
    behavior_residual: float


@dataclass(frozen=True)
class PseudoSessionPlan:
    anchor_session: str
    anchor_start: int
    mix_candidate: bool
    mixed: bool
    contributors: tuple[ContributorPlan, ...]
    final_permutation: np.ndarray

    @property
    def total_units(self) -> int:
        return int(sum(item.unit_indices.size for item in self.contributors))


class SizeMatchedPseudoSessionDataset(Dataset):
    """Wrap one ordinary source dataset with deterministic pseudo-session examples.

    The ordinary ``SessionBatchSampler`` can still be used because
    ``window_indices`` is unchanged and every mixed item retains the anchor
    session name and anchor unit count.  Thus all examples collated in a batch
    have the same final ``N`` even though their contributors differ.
    """

    def __init__(
        self,
        base: Dandi688MultiSessionDataset,
        *,
        seed: int,
        mix_probability: float = 0.5,
        contributor_count: int = 3,
        max_behavior_residual: float = 0.25,
    ) -> None:
        require(isinstance(base, Dandi688MultiSessionDataset), "base dataset type drift")
        require(0.0 <= float(mix_probability) <= 1.0, "mix_probability must lie in [0,1]")
        require(int(contributor_count) >= 2, "pseudo-session needs at least two contributors")
        require(np.isfinite(max_behavior_residual) and float(max_behavior_residual) > 0.0,
                "max_behavior_residual must be finite and positive")
        require(not base.random_calibration, "pseudo-session v1 requires frozen chronological calibration")
        require(base.window_size >= 1, "window_size must be positive")
        require(len(base.sessions) >= int(contributor_count), "not enough source sessions for contributor_count")
        self.base = base
        self.sessions = base.sessions
        self.window_indices = base.window_indices
        self.window_size = int(base.window_size)
        self.calibration_n_trials = int(base.calibration_n_trials)
        self.seed = int(seed)
        self.mix_probability = float(mix_probability)
        self.contributor_count = int(contributor_count)
        self.max_behavior_residual = float(max_behavior_residual)
        self.session_names = tuple(sorted(self.sessions))

        self._starts: dict[str, np.ndarray] = {}
        self._endpoint_behavior: dict[str, np.ndarray] = {}
        self._trees: dict[str, cKDTree] = {}
        for name in self.session_names:
            record = self.sessions[name]
            require(record.side_features is not None, f"{name}: T4/Z4 side features are required")
            require(record.electrode_ids is None, f"{name}: v1 is SUA B3S only; electrode ids are out of scope")
            require(record.calib_trials.shape[0] == self.calibration_n_trials,
                    f"{name}: chronological calibration count drift")
            require(record.calib_trials.shape[-1] == record.neural.shape[-1],
                    f"{name}: calibration/neural unit mismatch")
            require(record.side_features.shape == (record.neural.shape[-1], 4),
                    f"{name}: side feature shape must be [N,4]")
            starts = np.asarray(record.valid_starts, dtype=np.int64)
            require(starts.ndim == 1 and starts.size > 0, f"{name}: no source query windows")
            endpoints = np.asarray(record.behavior[starts + self.window_size - 1], dtype=np.float64)
            require(endpoints.ndim == 2 and endpoints.shape[1] == 2,
                    f"{name}: continuous velocity matching requires two behaviour coordinates")
            require(np.isfinite(endpoints).all(), f"{name}: non-finite matching behaviour")
            self._starts[name] = starts
            self._endpoint_behavior[name] = np.ascontiguousarray(endpoints)
            self._trees[name] = cKDTree(endpoints)

    def __len__(self) -> int:
        return len(self.window_indices)

    def _is_mixed(self, index: int) -> bool:
        if self.mix_probability <= 0.0:
            return False
        if self.mix_probability >= 1.0:
            return True
        threshold = int(math.floor(self.mix_probability * (1 << 64)))
        return _stable_u64(SCHEMA, self.seed, "mix", int(index)) < threshold

    def _donor_sessions(self, index: int, anchor: str) -> tuple[str, ...]:
        candidates = [name for name in self.session_names if name != anchor]
        order = sorted(
            candidates,
            key=lambda name: (_stable_u64(SCHEMA, self.seed, "donor", int(index), name), name),
        )
        donors = tuple(order[: self.contributor_count - 1])
        require(len(donors) == self.contributor_count - 1 and len(set(donors)) == len(donors),
                "donor selection is not distinct")
        return donors

    def _quotas(self, total_units: int, index: int) -> tuple[int, ...]:
        require(total_units >= self.contributor_count, "anchor has fewer units than contributors")
        base, remainder = divmod(total_units, self.contributor_count)
        quotas = [base] * self.contributor_count
        rotation = _stable_u64(SCHEMA, self.seed, "quota", int(index)) % self.contributor_count
        for offset in range(remainder):
            quotas[(rotation + offset) % self.contributor_count] += 1
        require(sum(quotas) == total_units and max(quotas) - min(quotas) <= 1,
                "size-matched quota construction failed")
        return tuple(quotas)

    def _unit_indices(self, *, index: int, session: str, quota: int) -> np.ndarray:
        n_units = int(self.sessions[session].neural.shape[-1])
        require(0 < quota <= n_units, f"{session}: requested {quota} of {n_units} units")
        rng = np.random.default_rng(_stable_u64(SCHEMA, self.seed, "units", int(index), session))
        selected = np.sort(rng.choice(n_units, size=quota, replace=False).astype(np.int64))
        require(np.unique(selected).size == quota, f"{session}: duplicate unit selection")
        return selected

    def plan_for_index(self, index: int) -> PseudoSessionPlan:
        anchor, anchor_start = self.window_indices[int(index)]
        anchor_start = int(anchor_start)
        anchor_units = int(self.sessions[anchor].neural.shape[-1])
        if not self._is_mixed(int(index)):
            indices = np.arange(anchor_units, dtype=np.int64)
            return PseudoSessionPlan(
                anchor_session=anchor,
                anchor_start=anchor_start,
                mix_candidate=False,
                mixed=False,
                contributors=(ContributorPlan(anchor, anchor_start, indices, 0.0),),
                final_permutation=indices.copy(),
            )

        contributor_names = (anchor, *self._donor_sessions(int(index), anchor))
        quotas = self._quotas(anchor_units, int(index))
        anchor_target = np.asarray(
            self.sessions[anchor].behavior[anchor_start + self.window_size - 1], dtype=np.float64
        )
        contributors: list[ContributorPlan] = []
        for slot, (name, quota) in enumerate(zip(contributor_names, quotas, strict=True)):
            if slot == 0:
                start, residual = anchor_start, 0.0
            else:
                distance, tree_index = self._trees[name].query(anchor_target, k=1)
                start = int(self._starts[name][int(tree_index)])
                residual = float(distance / math.sqrt(anchor_target.size))
            contributors.append(ContributorPlan(
                session=name,
                start=start,
                unit_indices=self._unit_indices(index=int(index), session=name, quota=int(quota)),
                behavior_residual=residual,
            ))
        rng = np.random.default_rng(_stable_u64(SCHEMA, self.seed, "final-permutation", int(index)))
        permutation = rng.permutation(anchor_units).astype(np.int64)
        accepted = all(
            item.behavior_residual <= self.max_behavior_residual
            for item in contributors[1:]
        )
        return PseudoSessionPlan(
            anchor_session=anchor,
            anchor_start=anchor_start,
            mix_candidate=True,
            mixed=accepted,
            contributors=tuple(contributors),
            final_permutation=permutation,
        )

    def __getitem__(self, index: int):
        plan = self.plan_for_index(int(index))
        if not plan.mixed:
            return self.base[int(index)]
        neural_blocks: list[np.ndarray] = []
        calibration_blocks: list[np.ndarray] = []
        side_blocks: list[np.ndarray] = []
        for item in plan.contributors:
            record = self.sessions[item.session]
            unit = item.unit_indices
            neural_blocks.append(record.neural[item.start : item.start + self.window_size, unit])
            calibration_blocks.append(record.calib_trials[..., unit])
            assert record.side_features is not None
            side_blocks.append(record.side_features[unit])
        neural = np.concatenate(neural_blocks, axis=-1)[:, plan.final_permutation]
        calibration = np.concatenate(calibration_blocks, axis=-1)[..., plan.final_permutation]
        side = np.concatenate(side_blocks, axis=0)[plan.final_permutation]
        anchor = self.sessions[plan.anchor_session]
        behavior = anchor.behavior[plan.anchor_start : plan.anchor_start + self.window_size]
        require(neural.shape == (self.window_size, plan.total_units), "mixed neural shape drift")
        require(calibration.shape[-1] == side.shape[0] == plan.total_units,
                "mixed neural/calibration/side unit alignment drift")
        return (
            torch.from_numpy(np.ascontiguousarray(neural)).float(),
            torch.from_numpy(np.ascontiguousarray(behavior)).float(),
            torch.from_numpy(np.ascontiguousarray(calibration)).float(),
            plan.anchor_session,
            torch.from_numpy(np.ascontiguousarray(side)).float(),
        )

    def audit_manifest(self, *, max_examples: int | None = None) -> dict[str, Any]:
        count = len(self) if max_examples is None else min(len(self), int(max_examples))
        require(count > 0, "audit requires at least one example")
        schedule = hashlib.sha256()
        mixed_count = 0
        candidate_count = 0
        rejected_count = 0
        accepted_residuals: list[float] = []
        candidate_residuals: list[float] = []
        accepted_donor_counts = {name: 0 for name in self.session_names}
        candidate_donor_counts = {name: 0 for name in self.session_names}
        for index in range(count):
            plan = self.plan_for_index(index)
            schedule.update(plan.anchor_session.encode("utf-8"))
            schedule.update(np.asarray(
                [plan.anchor_start, int(plan.mix_candidate), int(plan.mixed)], dtype="<i8"
            ).tobytes())
            for item in plan.contributors:
                schedule.update(item.session.encode("utf-8"))
                schedule.update(np.asarray([item.start], dtype="<i8").tobytes())
                schedule.update(np.asarray(item.unit_indices, dtype="<i8").tobytes())
                schedule.update(np.asarray([item.behavior_residual], dtype="<f8").tobytes())
                if item.session != plan.anchor_session:
                    candidate_donor_counts[item.session] += 1
                    candidate_residuals.append(item.behavior_residual)
                    if plan.mixed:
                        accepted_donor_counts[item.session] += 1
                        accepted_residuals.append(item.behavior_residual)
            schedule.update(np.asarray(plan.final_permutation, dtype="<i8").tobytes())
            candidate_count += int(plan.mix_candidate)
            mixed_count += int(plan.mixed)
            rejected_count += int(plan.mix_candidate and not plan.mixed)
        values = np.asarray(accepted_residuals, dtype=np.float64)
        candidate_values = np.asarray(candidate_residuals, dtype=np.float64)
        def residual_summary(rows: np.ndarray) -> dict[str, Any]:
            return {
                "count": int(rows.size),
                "mean": float(rows.mean()) if rows.size else 0.0,
                "median": float(np.median(rows)) if rows.size else 0.0,
                "p90": float(np.quantile(rows, 0.90)) if rows.size else 0.0,
                "p99": float(np.quantile(rows, 0.99)) if rows.size else 0.0,
                "max": float(rows.max()) if rows.size else 0.0,
                "sha256": _array_sha256(rows),
            }
        return {
            "schema": SCHEMA,
            "match_kind": MATCH_KIND,
            "seed": self.seed,
            "mix_probability_requested": self.mix_probability,
            "contributor_count": self.contributor_count,
            "max_behavior_residual": self.max_behavior_residual,
            "examples_audited": count,
            "mix_candidate_examples": candidate_count,
            "mixed_examples": mixed_count,
            "mixed_fraction_observed": mixed_count / count,
            "rejected_candidate_examples": rejected_count,
            "candidate_donor_uses": candidate_donor_counts,
            "accepted_donor_uses": accepted_donor_counts,
            "donor_sessions_all_source_only": True,
            "total_unit_count_equals_anchor_for_every_plan": True,
            "schedule_sha256": schedule.hexdigest(),
            "candidate_endpoint_residual": residual_summary(candidate_values),
            "accepted_endpoint_residual": residual_summary(values),
        }


class CebraPseudoSessionDataModule(Dandi688MultiSessionDataModule):
    """Ordinary evaluation DataModule with a pseudo-session train dataset only."""

    def __init__(
        self,
        *args: Any,
        pseudo_mix_probability: float = 0.5,
        pseudo_contributor_count: int = 3,
        pseudo_max_behavior_residual: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pseudo_mix_probability = float(pseudo_mix_probability)
        self.pseudo_contributor_count = int(pseudo_contributor_count)
        self.pseudo_max_behavior_residual = float(pseudo_max_behavior_residual)
        self._ordinary_train_dataset: Dandi688MultiSessionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        if stage not in (None, "fit"):
            return
        if isinstance(self.train_dataset, SizeMatchedPseudoSessionDataset):
            return
        require(isinstance(self.train_dataset, Dandi688MultiSessionDataset),
                "ordinary train dataset missing after setup")
        require(self.task.upper() == "CO", "first pseudo-session contract is CO only")
        require(self.signal_view == "sua", "first pseudo-session contract is sorted SUA only")
        require(self.window_size == 50 and self.calibration_n_trials == 30,
                "first pseudo-session contract requires A2 M30/W50")
        require(not self.random_calibration, "first pseudo-session contract requires chronological M30")
        require(self.side_feature_group in {"t4", "z4"}, "T4/Z4 sibling is mandatory")
        self._ordinary_train_dataset = self.train_dataset
        self.train_dataset = SizeMatchedPseudoSessionDataset(
            self._ordinary_train_dataset,
            seed=self.seed,
            mix_probability=self.pseudo_mix_probability,
            contributor_count=self.pseudo_contributor_count,
            max_behavior_residual=self.pseudo_max_behavior_residual,
        )

    def pseudo_session_manifest(self, *, max_examples: int | None = None) -> Mapping[str, Any]:
        require(isinstance(self.train_dataset, SizeMatchedPseudoSessionDataset),
                "pseudo-session train dataset has not been constructed")
        return self.train_dataset.audit_manifest(max_examples=max_examples)
