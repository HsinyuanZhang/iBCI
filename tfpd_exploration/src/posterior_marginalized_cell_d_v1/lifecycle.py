"""Narrow composition seam over the audited equal-session Cell-D backend.

This review-stage module deliberately does *not* implement a physical launch,
artifact writer, source adapter, or GPU device policy.  It provides the one
safe hook a later reviewed backend needs: prewarm PMC carriers before the
single inherited DataLoader iterator is created, then replace only the side
tensor yielded to the inherited Cell-D ``train_step``.  It neither changes
the equal-session scheduler nor reimplements the Cell-D optimizer/loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from src import cell_d_equal_session_v1 as equal_session

from . import plan
from .core import PMCError, PosteriorMarginalizedSideCache, PosteriorSideReplacingIterator


class PMCLifecycleError(RuntimeError):
    """Fail closed when the later composition seam is used incorrectly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PMCLifecycleError(message)


class EqualSessionBackendSurface(Protocol):
    def begin_epoch(self, runtime: Mapping[str, Any], epoch: int, flags: Any) -> Mapping[str, object]: ...
    def train_step(self, runtime: Mapping[str, Any], *, global_step: int, expected_lr: float,
                   require_full_proof: bool, flags: Any) -> Any: ...


@dataclass
class PosteriorMarginalizedEqualSessionAdapter:
    """Inject a cached PMC side into the immutable equal-session backend.

    The adapter is intentionally too narrow to execute alone.  A future
    physical backend must first construct the strict-27 source authority and
    the device-compatible cache, then pass the inherited runtime mapping here.
    ``begin_epoch`` prewarms *before* the base creates its only iterator.
    ``train_step`` delegates byte-for-byte to the inherited implementation.
    """

    base: EqualSessionBackendSurface
    cache: PosteriorMarginalizedSideCache

    def begin_epoch(self, runtime: Mapping[str, Any], epoch: int, flags: Any) -> Mapping[str, object]:
        if type(epoch) is not int or not 0 <= epoch < plan.EPOCHS:
            raise PMCLifecycleError("PMC logical epoch drift")
        if runtime.get("epoch_iterator") is not None:
            raise PMCLifecycleError("PMC must prewarm before the inherited DataLoader iterator exists")
        self.cache.fit_all_source_posteriors()
        entries = self.cache.prewarm_epoch(epoch)
        _require(len(entries) == len(self.cache.roster), "PMC prewarm did not build every source session")
        # The source-side posterior work and the sole host->device transfer
        # happen before the inherited backend constructs/iterates its loader.
        # ``side_for_optimizer_batch`` consequently rejects any hidden device
        # conversion in the timed optimizer path.
        if "device" not in runtime:
            raise PMCLifecycleError("PMC runtime lacks a reviewed device before cache materialization")
        materialized = self.cache.materialize_epoch_for_device(epoch=epoch, device=runtime["device"])
        _require(len(materialized) == len(self.cache.roster), "PMC device cache materialization drift")
        evidence = dict(self.base.begin_epoch(runtime, epoch, flags))
        inherited_iterator = runtime.get("epoch_iterator")
        if inherited_iterator is None:
            raise PMCLifecycleError("equal-session backend failed to create its one epoch iterator")
        runtime["epoch_iterator"] = PosteriorSideReplacingIterator(
            inherited_iterator, cache=self.cache, epoch=epoch, expected_device=runtime["device"],
        )
        evidence["posterior_marginalized_cache"] = self.cache.epoch_receipt_payload(epoch)
        evidence["pmc_prepared_before_inherited_iterator"] = True
        evidence["pmc_device_cache_materialized_before_inherited_iterator"] = True
        evidence["pmc_only_changed_batch_field"] = "side_features_index_4"
        return evidence

    def train_step(self, runtime: Mapping[str, Any], *, global_step: int, expected_lr: float,
                   require_full_proof: bool, flags: Any) -> Any:
        return self.base.train_step(
            runtime, global_step=global_step, expected_lr=expected_lr,
            require_full_proof=require_full_proof, flags=flags,
        )

    def cache_evidence(self) -> Mapping[str, int]:
        evidence = self.cache.observer()
        evidence.assert_batch_loop_clean()
        return evidence.payload()


def deferred_execution_message() -> str:
    """Make the no-launch boundary machine-readable for a future root audit."""
    return (
        "POSTERIOR_MARGINALIZED_CELL_D_SEED42 physical execution is intentionally unavailable: "
        "it requires a separate reviewed strict-27 source authority, root-selected compatible-device contract, "
        "PMC-specific immutable receipt lifecycle, and an in-process root capability."
    )


def require_no_direct_equal_session_lifecycle_reuse() -> None:
    """Reject a tempting but semantically wrong direct call to sealed runner.

    ``cell_d_equal_session_v1.run_lifecycle`` binds its own cell name and
    receipt schemas.  PMC may reuse its scheduler/step methods through this
    adapter, but cannot honestly publish PMC results through that old route.
    """
    if getattr(equal_session, "CELL", None) != "CELL_D_EQUAL_SESSION_SEED42":
        raise PMCLifecycleError("equal-session dependency identity drift")
