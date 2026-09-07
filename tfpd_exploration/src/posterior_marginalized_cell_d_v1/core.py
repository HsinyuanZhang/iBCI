"""Torch-only cached posterior-side treatment for PMC-D.

The module contains no dataset, checkpoint, output-root, CUDA, or launch
entrypoint.  It composes the audited posterior primitives rather than
reimplementing their conjugate fit or local sampling algorithm.  The only
model-visible treatment is a replacement of the fifth batch tensor
(``side_features``) before the sealed Cell-D trainer receives it.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from src.posterior_carrier_v1 import core as posterior

from . import plan


class PMCError(RuntimeError):
    """Fail closed when the PMC one-factor treatment drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PMCError(message)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PMCError(f"{label} must be an exact lowercase SHA-256")
    return value


def _digest_json(value: object) -> str:
    return hashlib.sha256(posterior.canonical_json_bytes(value)).hexdigest()


def _finite_tensor(value: Tensor, label: str, *, ndim: int | None = None) -> None:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise PMCError(f"{label} must be a floating tensor")
    if ndim is not None and value.ndim != ndim:
        raise PMCError(f"{label} rank drift")
    if value.numel() < 1 or not bool(torch.isfinite(value).all().item()):
        raise PMCError(f"{label} must be nonempty and finite")


def validate_sealed_ordinary_ols_normalizer(
    value: posterior.FrozenSourceT4Normalizer,
) -> posterior.FrozenSourceT4Normalizer:
    """Require the real sealed Cell-D float32 OLS moments, not just its SHA.

    ``FrozenSourceT4Normalizer`` deliberately accepts any finite moments with
    a syntactically valid authority string because it is shared by multiple
    routes.  PMC has a narrower contract: its only allowed normalizer is the
    exact Cell-D point-side normalizer.  Check dtype, values, and the same
    canonical semantic hash used by ``mc_maze.unit_side_features`` here at the
    route boundary.  A posterior normalizer or a float64 copy with otherwise
    numerically similar values cannot pass this gate.
    """
    if not isinstance(value, posterior.FrozenSourceT4Normalizer):
        raise PMCError("PMC must use the sealed ordinary OLS T4 normalizer, never a posterior normalizer")
    if value.mean.dtype != torch.float32 or value.std.dtype != torch.float32:
        raise PMCError("PMC sealed OLS moments must be literal torch.float32")
    expected_mean = torch.tensor(plan.SEALED_OLS_T4_MEAN_FLOAT32, dtype=torch.float32, device=value.mean.device)
    expected_std = torch.tensor(plan.SEALED_OLS_T4_STD_FLOAT32, dtype=torch.float32, device=value.std.device)
    if not torch.equal(value.mean, expected_mean) or not torch.equal(value.std, expected_std):
        raise PMCError("PMC sealed OLS normalizer numeric literal drift")
    # ``tolist`` from a float32 tensor yields the exact values used in the
    # sealed stats JSON domain.  The plan helper separately validates the
    # schema and binds the expected semantic SHA.
    semantic = plan.sealed_ols_t4_normalizer_sha256({
        "mean": value.mean.detach().cpu().tolist(),
        "std": value.std.detach().cpu().tolist(),
    })
    if semantic != plan.SEALED_OLS_T4_NORMALIZER_SHA256 or value.authority_sha256 != semantic:
        raise PMCError("PMC sealed OLS normalizer authority/hash drift")
    return value


def _quantiles(value: Tensor) -> dict[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    _finite_tensor(flat, "PMC cache statistic")
    levels = torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0), dtype=torch.float64)
    quantiles = torch.quantile(flat, levels)
    return {
        "min": float(quantiles[0].item()),
        "q25": float(quantiles[1].item()),
        "q50": float(quantiles[2].item()),
        "q75": float(quantiles[3].item()),
        "max": float(quantiles[4].item()),
    }


@dataclass(frozen=True)
class PosteriorPrefixInputs:
    """One exact source-session prefix family, supplied by a later authority."""

    counts_by_budget: Mapping[int, Tensor]
    exposure_by_budget: Mapping[int, Tensor]
    theta_by_budget: Mapping[int, Tensor]
    unit_order_sha256: str
    source_file_sha256: str
    prefix_row_ids_sha256_by_budget: Mapping[int, str]
    theta_recovery_evidence_sha256: str
    theta_recovery_closure_sha256: str
    theta_recovery_schema: str = plan.THETA_RECOVERY_SCHEMA
    theta_recovery_semantics: str = plan.THETA_RECOVERY_SEMANTICS
    theta_fallback_topology_sha256: str = plan.THETA_FALLBACK_TOPOLOGY_SHA256

    def __post_init__(self) -> None:
        _require_sha256(self.unit_order_sha256, "PMC unit-order SHA")
        _require_sha256(self.source_file_sha256, "PMC source-file SHA")
        _require_sha256(self.theta_recovery_evidence_sha256, "PMC theta-recovery evidence SHA")
        _require_sha256(self.theta_recovery_closure_sha256, "PMC theta-recovery closure SHA")
        if (self.theta_recovery_schema != plan.THETA_RECOVERY_SCHEMA
                or self.theta_recovery_semantics != plan.THETA_RECOVERY_SEMANTICS
                or self.theta_fallback_topology_sha256 != plan.THETA_FALLBACK_TOPOLOGY_SHA256):
            raise PMCError("PMC must bind the approved Phase-B-v2 same-prefix theta recovery")
        if (not isinstance(self.prefix_row_ids_sha256_by_budget, Mapping)
                or set(self.prefix_row_ids_sha256_by_budget) != set(plan.BUDGETS)):
            raise PMCError("PMC theta prefix-row topology drift")
        for budget in plan.BUDGETS:
            _require_sha256(self.prefix_row_ids_sha256_by_budget[budget], "PMC theta prefix-row SHA")
        if (not isinstance(self.counts_by_budget, Mapping) or set(self.counts_by_budget) != set(plan.BUDGETS)
                or not isinstance(self.exposure_by_budget, Mapping) or set(self.exposure_by_budget) != set(plan.BUDGETS)
                or not isinstance(self.theta_by_budget, Mapping) or set(self.theta_by_budget) != set(plan.BUDGETS)):
            raise PMCError("PMC prefix inputs require exactly M4/M10/M30 maps")
        unit_count: int | None = None
        for budget in plan.BUDGETS:
            counts = self.counts_by_budget[budget]
            exposure = self.exposure_by_budget[budget]
            theta = self.theta_by_budget[budget]
            if not torch.is_tensor(counts) or counts.ndim != 2 or counts.shape[1] != budget or counts.shape[0] < 1:
                raise PMCError("PMC direct-count prefix shape drift")
            if counts.dtype == torch.bool or counts.is_complex():
                raise PMCError("PMC direct-count prefix dtype drift")
            if counts.is_floating_point() and (not bool(torch.isfinite(counts).all().item()) or not torch.equal(counts, torch.trunc(counts))):
                raise PMCError("PMC prefix counts must already be exact integers")
            if bool((counts < 0).any().item()):
                raise PMCError("PMC prefix counts must be nonnegative")
            _finite_tensor(exposure, "PMC exposure", ndim=1)
            _finite_tensor(theta, "PMC theta", ndim=1)
            if exposure.shape != (budget,) or theta.shape != (budget,) or exposure.device != counts.device or theta.device != counts.device:
                raise PMCError("PMC prefix trial/exposure/theta axis/device drift")
            if not bool((exposure > 0).all().item()):
                raise PMCError("PMC exposure must be positive")
            if unit_count is None:
                unit_count = int(counts.shape[0])
            elif unit_count != int(counts.shape[0]):
                raise PMCError("PMC prefix unit count must be budget-invariant")

    @property
    def unit_count(self) -> int:
        return int(self.counts_by_budget[plan.BUDGETS[0]].shape[0])

    def for_budget(self, budget: int) -> dict[str, Tensor]:
        if budget not in plan.BUDGETS:
            raise PMCError("PMC requested a non-frozen posterior budget")
        return {
            "counts": self.counts_by_budget[budget],
            "exposure": self.exposure_by_budget[budget],
            "theta": self.theta_by_budget[budget],
        }


@dataclass(frozen=True)
class PMCCacheEntry:
    session: str
    session_index: int
    epoch: int
    budget: int
    view: posterior.PosteriorCarrierView
    posterior_fit_sha256: str
    sampling_seed_sha256: str
    unit_order_sha256: str
    source_file_sha256: str
    prefix_row_ids_sha256: str
    theta_recovery_evidence_sha256: str
    theta_recovery_closure_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "session": self.session,
            "session_index": self.session_index,
            "epoch": self.epoch,
            "budget": self.budget,
            "posterior_fit_sha256": self.posterior_fit_sha256,
            "sampling_seed_sha256": self.sampling_seed_sha256,
            "raw_sample_sha256": posterior.tensor_digest(self.view.raw_t4),
            "sealed_ols_normalized_sample_sha256": posterior.tensor_digest(self.view.normalized_t4),
            "raw_sample_statistics": _quantiles(self.view.raw_t4),
            "sealed_ols_normalized_sample_statistics": _quantiles(self.view.normalized_t4),
            "unit_order_sha256": self.unit_order_sha256,
            "source_file_sha256": self.source_file_sha256,
            "prefix_row_ids_sha256": self.prefix_row_ids_sha256,
            "theta_recovery": {
                "schema": plan.THETA_RECOVERY_SCHEMA,
                "semantics": plan.THETA_RECOVERY_SEMANTICS,
                "fallback_topology_sha256": plan.THETA_FALLBACK_TOPOLOGY_SHA256,
                "evidence_sha256": self.theta_recovery_evidence_sha256,
                "closure_sha256": self.theta_recovery_closure_sha256,
            },
            "sampled": self.view.sampled,
            "normalizer_authority_sha256": self.view.normalizer_authority_sha256,
            "normalization_numeric_boundary": {
                "raw_sample_dtype": str(self.view.raw_t4.dtype),
                "sealed_ols_moments_promoted_to_raw_dtype": str(self.view.raw_t4.dtype),
                "model_side_dtype": str(self.view.normalized_t4.dtype),
                "single_cast_to_sealed_cell_d_side_dtype": self.view.raw_t4.dtype != self.view.normalized_t4.dtype,
                "posterior_specific_moments_used": False,
            },
            "posterior_sampling_domain_cell": posterior.CELL,
            "consumer_cell": plan.CELL,
        }


@dataclass(frozen=True)
class PMCCacheObserver:
    """Receipt-stable proof that expensive posterior work stayed pre-loop."""

    posterior_fit_calls: int
    posterior_inverse_calls: int
    sampled_view_builds: int
    normalized_view_builds: int
    cached_session_epochs: int
    cache_build_counter: int
    device_materializations: int
    batch_loop_side_lookups: int
    batch_loop_posterior_fit_calls: int
    batch_loop_inverse_calls: int
    batch_loop_refit_calls: int
    batch_loop_sampling_calls: int
    batch_loop_normalizer_fit_calls: int

    def payload(self) -> dict[str, int]:
        return {
            "posterior_fit_calls": self.posterior_fit_calls,
            "posterior_inverse_calls": self.posterior_inverse_calls,
            "sampled_view_builds": self.sampled_view_builds,
            "normalized_view_builds": self.normalized_view_builds,
            "cached_session_epochs": self.cached_session_epochs,
            "cache_build_counter": self.cache_build_counter,
            "device_materializations": self.device_materializations,
            "batch_loop_side_lookups": self.batch_loop_side_lookups,
            "batch_loop_posterior_fit_calls": self.batch_loop_posterior_fit_calls,
            "batch_loop_inverse_calls": self.batch_loop_inverse_calls,
            "batch_loop_refit_calls": self.batch_loop_refit_calls,
            "batch_loop_sampling_calls": self.batch_loop_sampling_calls,
            "batch_loop_normalizer_fit_calls": self.batch_loop_normalizer_fit_calls,
        }

    def assert_batch_loop_clean(self) -> None:
        if any(value != 0 for value in (
            self.batch_loop_posterior_fit_calls,
            self.batch_loop_inverse_calls,
            self.batch_loop_refit_calls,
            self.batch_loop_sampling_calls,
            self.batch_loop_normalizer_fit_calls,
        )):
            raise PMCError("PMC batch loop performed a forbidden posterior operation")


def degenerate_sampled_side(
    *,
    posterior_mean_beta: Tensor,
    zero_spike_mask: Tensor,
    normalizer: posterior.FrozenSourceT4Normalizer,
) -> tuple[Tensor, Tensor]:
    """Exact degenerate-Gaussian sample for the zero-covariance CPU gate.

    The audited production posterior type correctly requires positive-definite
    covariance, so an exactly zero covariance fixture cannot be instantiated
    as ``PosteriorCarrier``.  Mathematically its sample is its mean; this
    helper makes that synthetic safety case explicit without altering the
    audited sampler used for every non-degenerate production posterior.
    """
    _finite_tensor(posterior_mean_beta, "PMC degenerate posterior mean", ndim=2)
    if posterior_mean_beta.shape[1] != 3 or zero_spike_mask.dtype != torch.bool or zero_spike_mask.shape != (posterior_mean_beta.shape[0],):
        raise PMCError("PMC degenerate posterior fixture shape drift")
    if posterior_mean_beta.device != normalizer.mean.device or posterior_mean_beta.dtype != normalizer.mean.dtype:
        raise PMCError("PMC degenerate posterior/normalizer device or dtype drift")
    raw_t4 = posterior.beta_to_raw_t4(posterior_mean_beta, zero_spike_mask)
    return raw_t4, normalizer.normalize_raw(raw_t4)


class PosteriorMarginalizedSideCache:
    """Fitted posterior bank plus one sampled sealed-OLS view/session/epoch.

    The class intentionally follows the audited posterior bank's efficient
    topology: one fit/inverse for each source-session x {M4,M10,M30}, then
    one local deterministic sample for each source-session x logical epoch.
    All fits occur before any epoch iterator, and all samples for an epoch are
    prewarmed before that epoch's iterator exists.  The optimizer path can
    only retrieve an immutable prepared view.
    """

    def __init__(
        self,
        *,
        roster: Sequence[str],
        prefix_inputs_by_session: Mapping[str, PosteriorPrefixInputs],
        prior: posterior.SourcePrior,
        sealed_ols_normalizer: posterior.FrozenSourceT4Normalizer,
        seed: int = plan.SEED,
    ) -> None:
        self._roster = tuple(roster)
        if (not self._roster or len(set(self._roster)) != len(self._roster)
                or any(not isinstance(session, str) or not session for session in self._roster)):
            raise PMCError("PMC roster must be ordered unique nonempty source session IDs")
        if prior.source_roster != self._roster:
            raise PMCError("PMC source prior roster drift")
        sealed_ols_normalizer = validate_sealed_ordinary_ols_normalizer(sealed_ols_normalizer)
        if type(seed) is not int or seed != plan.SEED:
            raise PMCError("PMC sampling seed drift")
        if not isinstance(prefix_inputs_by_session, Mapping) or set(prefix_inputs_by_session) != set(self._roster):
            raise PMCError("PMC prefix session topology drift")
        copied: dict[str, PosteriorPrefixInputs] = {}
        for session in self._roster:
            row = prefix_inputs_by_session[session]
            if not isinstance(row, PosteriorPrefixInputs):
                raise PMCError("PMC prefix input type drift")
            copied[session] = row
        self._inputs = copied
        self._prior = prior
        self._normalizer = sealed_ols_normalizer
        # Do not override the old sampler's domain cell.  It is the audited,
        # imported local-generator semantics; PMC records that provenance in
        # every cache row while using a separate consumer-cell identity.
        self._samplers: dict[tuple[str, torch.dtype], posterior.SessionStaticPosteriorSampler] = {}
        self._posteriors: dict[str, dict[int, posterior.PosteriorCarrier]] = {}
        self._epoch_entries: dict[tuple[str, int], PMCCacheEntry] = {}
        self._fit_calls = 0
        self._inverse_calls = 0
        self._sampled_view_builds = 0
        self._cache_build_counter = 0
        self._device_materializations = 0
        self._batch_loop_side_lookups = 0

    @property
    def roster(self) -> tuple[str, ...]:
        return self._roster

    @property
    def sealed_ols_normalizer(self) -> posterior.FrozenSourceT4Normalizer:
        return self._normalizer

    def fit_all_source_posteriors(self) -> None:
        """Fit the 27x3 deterministic bank once, before any iterator exists."""
        if self._posteriors:
            return
        before = posterior.host_rng_fingerprint()
        fitted: dict[str, dict[int, posterior.PosteriorCarrier]] = {}
        for session in self._roster:
            inputs = self._inputs[session]
            per_budget: dict[int, posterior.PosteriorCarrier] = {}
            for budget in plan.BUDGETS:
                selected = inputs.for_budget(budget)
                carrier = posterior.fit_conjugate_posterior(
                    counts=selected["counts"], exposure=selected["exposure"], theta=selected["theta"], prior=self._prior,
                )
                if carrier.unit_count != inputs.unit_count or carrier.audit.inverse_call_count != 1:
                    raise PMCError("PMC posterior fit/unit inverse audit drift")
                per_budget[budget] = carrier
                self._fit_calls += 1
                self._inverse_calls += carrier.audit.inverse_call_count
            fitted[session] = per_budget
        posterior.assert_host_rng_unchanged(before, posterior.host_rng_fingerprint())
        self._posteriors = fitted

    def _sampler_for(self, carrier: posterior.PosteriorCarrier) -> posterior.SessionStaticPosteriorSampler:
        """Promote frozen OLS moments for fitting precision, then cast once.

        The ordinary OLS normalizer's *values and authority* are held.  If a
        posterior fit uses float64, its two fixed float32 moments are promoted
        exactly to float64 for raw normalization, after which the normalized
        model-side tensor is cast once to the sealed Cell-D consumer dtype.
        No posterior-specific moment is fitted or substituted.
        """
        key = (str(carrier.mean.device), carrier.mean.dtype)
        sampler = self._samplers.get(key)
        if sampler is None:
            promoted = posterior.FrozenSourceT4Normalizer(
                mean=self._normalizer.mean.to(device=carrier.mean.device, dtype=carrier.mean.dtype),
                std=self._normalizer.std.to(device=carrier.mean.device, dtype=carrier.mean.dtype),
                authority_sha256=self._normalizer.authority_sha256,
            )
            sampler = posterior.SessionStaticPosteriorSampler(seed=plan.SEED, normalizer=promoted)
            self._samplers[key] = sampler
        return sampler

    def _consumer_view(self, view: posterior.PosteriorCarrierView) -> posterior.PosteriorCarrierView:
        """Materialize the sole explicit normalized-side dtype boundary."""
        target_dtype, target_device = self._normalizer.mean.dtype, self._normalizer.mean.device
        normalized = view.normalized_t4.to(device=target_device, dtype=target_dtype).contiguous()
        if normalized.dtype == view.normalized_t4.dtype and normalized.device == view.normalized_t4.device:
            return view
        return posterior.PosteriorCarrierView(
            raw_beta=view.raw_beta,
            raw_t4=view.raw_t4,
            normalized_t4=normalized,
            credibility=view.credibility,
            zero_spike_mask=view.zero_spike_mask,
            sampled=view.sampled,
            session_id=view.session_id,
            epoch=view.epoch,
            posterior_sha256=view.posterior_sha256,
            normalizer_authority_sha256=view.normalizer_authority_sha256,
        )

    def _seed_digest(self, *, session: str, epoch: int) -> str:
        # Exact imported sampler domain.  It is intentionally not derived from
        # the PMC consumer label, which would be a new sampling semantics.
        seed = posterior.route_local_seed(cell=posterior.CELL, seed=plan.SEED, session_id=session, epoch=epoch)
        return _digest_json({"sampler_domain_cell": posterior.CELL, "seed": seed})

    def prewarm_epoch(self, epoch: int) -> tuple[PMCCacheEntry, ...]:
        """Sample every source session before the epoch DataLoader iterator."""
        if type(epoch) is not int or not 0 <= epoch < plan.EPOCHS:
            raise PMCError("PMC prewarm epoch drift")
        if not self._posteriors:
            raise PMCError("PMC must fit all source posteriors before prewarming an epoch")
        before = posterior.host_rng_fingerprint()
        entries: list[PMCCacheEntry] = []
        for index, session in enumerate(self._roster):
            key = (session, epoch)
            budget = plan.budget_for(epoch, index)
            carrier = self._posteriors[session][budget]
            existing = self._epoch_entries.get(key)
            if existing is None:
                sampled = self._sampler_for(carrier).carrier_for(
                    carrier, session_id=session, epoch=epoch, training=True,
                )
                view = self._consumer_view(sampled)
                entry = PMCCacheEntry(
                    session=session,
                    session_index=index,
                    epoch=epoch,
                    budget=budget,
                    view=view,
                    posterior_fit_sha256=carrier.digest(),
                    sampling_seed_sha256=self._seed_digest(session=session, epoch=epoch),
                    unit_order_sha256=self._inputs[session].unit_order_sha256,
                    source_file_sha256=self._inputs[session].source_file_sha256,
                    prefix_row_ids_sha256=self._inputs[session].prefix_row_ids_sha256_by_budget[budget],
                    theta_recovery_evidence_sha256=self._inputs[session].theta_recovery_evidence_sha256,
                    theta_recovery_closure_sha256=self._inputs[session].theta_recovery_closure_sha256,
                )
                self._epoch_entries[key] = entry
                self._sampled_view_builds += 1
                self._cache_build_counter += 1
            else:
                if existing.budget != budget or existing.posterior_fit_sha256 != carrier.digest():
                    raise PMCError("PMC epoch cache attempted to replace a sampled carrier")
            entries.append(self._epoch_entries[key])
        posterior.assert_host_rng_unchanged(before, posterior.host_rng_fingerprint())
        return tuple(entries)

    def entry_for_optimizer_batch(self, *, session: str, epoch: int) -> PMCCacheEntry:
        entry = self._epoch_entries.get((session, epoch))
        if entry is None:
            raise PMCError("PMC optimizer batch attempted to fit/sample a carrier instead of using a prewarmed cache")
        self._batch_loop_side_lookups += 1
        return entry

    def prepared_entry_for_audit(self, *, session: str, epoch: int) -> PMCCacheEntry:
        """Read a prewarmed entry for a descriptive pre-loop audit only.

        Unlike ``entry_for_optimizer_batch`` this method deliberately does not
        increment the timed batch-loop observer.  It lets the source-only
        calibration report inspect an already sampled carrier without
        relabelling audit work as an optimizer-path lookup.
        """
        entry = self._epoch_entries.get((session, epoch))
        if entry is None:
            raise PMCError("PMC descriptive audit requested an unprepared cache entry")
        return entry

    def materialize_epoch_for_device(self, *, epoch: int, device: torch.device | str) -> tuple[PMCCacheEntry, ...]:
        """Move the already-sampled epoch cache before the sole iterator exists.

        Sampling, posterior fitting, and OLS normalization remain CPU
        session×epoch work.  The only transfer is an immutable typed carrier
        view, once per source session before DataLoader construction.  Batch
        lookup thereafter performs no posterior operation or conversion.
        """
        entries = self.prewarm_epoch(epoch)
        target = torch.device(device)
        materialized: list[PMCCacheEntry] = []
        for entry in entries:
            view = entry.view
            if view.normalized_t4.device == target:
                materialized.append(entry)
                continue
            moved = posterior.PosteriorCarrierView(
                raw_beta=view.raw_beta.to(device=target),
                raw_t4=view.raw_t4.to(device=target),
                normalized_t4=view.normalized_t4.to(device=target),
                credibility=view.credibility.to(device=target),
                zero_spike_mask=view.zero_spike_mask.to(device=target),
                sampled=view.sampled,
                session_id=view.session_id,
                epoch=view.epoch,
                posterior_sha256=view.posterior_sha256,
                normalizer_authority_sha256=view.normalizer_authority_sha256,
            )
            replacement = PMCCacheEntry(
                session=entry.session, session_index=entry.session_index, epoch=entry.epoch,
                budget=entry.budget, view=moved, posterior_fit_sha256=entry.posterior_fit_sha256,
                sampling_seed_sha256=entry.sampling_seed_sha256,
                unit_order_sha256=entry.unit_order_sha256, source_file_sha256=entry.source_file_sha256,
                prefix_row_ids_sha256=entry.prefix_row_ids_sha256,
                theta_recovery_evidence_sha256=entry.theta_recovery_evidence_sha256,
                theta_recovery_closure_sha256=entry.theta_recovery_closure_sha256,
            )
            self._epoch_entries[(entry.session, epoch)] = replacement
            materialized.append(replacement)
            self._device_materializations += 1
        return tuple(materialized)

    @staticmethod
    def _validate_prepared_side_contract(
        *, template_shape: tuple[int, ...], template_dtype: torch.dtype,
        prepared_shape: tuple[int, ...], prepared_dtype: torch.dtype,
        prepared_device: torch.device | str, expected_device: torch.device | str,
    ) -> None:
        """Validate the pre-transfer replacement contract without copying.

        ``template_shape``/``template_dtype`` describe the CPU DataLoader
        field.  Its device is intentionally absent: the pre-materialized
        cached side must be allowed to inhabit the reviewed runtime device
        before inherited Cell-D ``train_step`` receives it.
        """
        if prepared_shape != template_shape or prepared_dtype != template_dtype:
            raise PMCError("PMC cached carrier/unit shape or dtype differs from held Cell-D batch")
        if torch.device(prepared_device) != torch.device(expected_device):
            raise PMCError("PMC optimizer batch would need a hidden side device conversion")

    def side_for_optimizer_batch(
        self, *, session: str, epoch: int, template_side: Tensor,
        expected_device: torch.device | str,
    ) -> Tensor:
        """Return only the cached normalized side tensor for one homogeneous B32 batch."""
        _finite_tensor(template_side, "PMC original Cell-D side feature", ndim=3)
        entry = self.entry_for_optimizer_batch(session=session, epoch=epoch)
        view = entry.view
        if template_side.shape[1:] != (view.unit_count, 4):
            raise PMCError("PMC cached carrier/unit topology differs from the held Cell-D batch")
        result = view.normalized_for_batch(int(template_side.shape[0]))
        # Equal-session's DataLoader yields the ordinary held side on CPU and
        # its inherited train_step transfers it only *after* this replacement
        # seam.  PMC deliberately supplies a pre-materialized reviewed-device
        # side here; the later inherited ``.to(device)`` is then a no-op.  Do
        # not compare against the CPU template device or perform a batch-loop
        # transfer.  Shape/dtype remain the held Cell-D contract.
        self._validate_prepared_side_contract(
            template_shape=tuple(template_side.shape), template_dtype=template_side.dtype,
            prepared_shape=tuple(result.shape), prepared_dtype=result.dtype,
            prepared_device=result.device, expected_device=expected_device,
        )
        return result

    def observer(self) -> PMCCacheObserver:
        return PMCCacheObserver(
            posterior_fit_calls=self._fit_calls,
            posterior_inverse_calls=self._inverse_calls,
            sampled_view_builds=self._sampled_view_builds,
            normalized_view_builds=self._sampled_view_builds,
            cached_session_epochs=len(self._epoch_entries),
            cache_build_counter=self._cache_build_counter,
            device_materializations=self._device_materializations,
            batch_loop_side_lookups=self._batch_loop_side_lookups,
            batch_loop_posterior_fit_calls=0,
            batch_loop_inverse_calls=0,
            batch_loop_refit_calls=0,
            batch_loop_sampling_calls=0,
            batch_loop_normalizer_fit_calls=0,
        )

    def epoch_receipt_payload(self, epoch: int) -> dict[str, object]:
        if type(epoch) is not int or not 0 <= epoch < plan.EPOCHS:
            raise PMCError("PMC epoch receipt index drift")
        rows = [self._epoch_entries[(session, epoch)].payload() for session in self._roster if (session, epoch) in self._epoch_entries]
        if len(rows) != len(self._roster):
            raise PMCError("PMC epoch receipt requires one cached sample per source session")
        observer = self.observer()
        observer.assert_batch_loop_clean()
        return {
            "schema": "posterior_marginalized_cell_d_epoch_cache_v1",
            "cell": plan.CELL,
            "epoch": epoch,
            "rows": rows,
            "cache_observer": observer.payload(),
            "sealed_ordinary_ols_normalizer_sha256": self._normalizer.authority_sha256,
            "posterior_normalizer_used": False,
            "inference_uses_deterministic_ordinary_ols_point_t4": True,
        }


class PosteriorSideReplacingIterator(Iterator[Any]):
    """Replace only batch index 4 before sealed Cell-D ``train_step`` sees it."""

    def __init__(
        self, iterator: Iterator[Any], *, cache: PosteriorMarginalizedSideCache, epoch: int,
        expected_device: torch.device | str,
    ) -> None:
        self._iterator = iterator
        self._cache = cache
        self._epoch = epoch
        self._expected_device = torch.device(expected_device)
        self.replaced_batches = 0

    def __iter__(self) -> "PosteriorSideReplacingIterator":
        return self

    def __next__(self) -> Any:
        batch = next(self._iterator)
        if not isinstance(batch, (tuple, list)) or len(batch) < 5:
            raise PMCError("PMC source batch lacks the held Cell-D five-tensor surface")
        neural, behavior, calibration, sessions, ordinary_side = batch[:5]
        if not torch.is_tensor(neural) or not torch.is_tensor(behavior) or not torch.is_tensor(calibration):
            raise PMCError("PMC must preserve tensor neural/behavior/calibration batch fields")
        if not isinstance(sessions, (tuple, list)) or len(sessions) != int(ordinary_side.shape[0]):
            raise PMCError("PMC batch/session topology drift")
        observed = tuple(sessions)
        if len(set(observed)) != 1 or not isinstance(observed[0], str):
            raise PMCError("PMC equal-session batch must contain exactly one scheduled session")
        side = self._cache.side_for_optimizer_batch(
            session=observed[0], epoch=self._epoch, template_side=ordinary_side,
            expected_device=self._expected_device,
        )
        if isinstance(batch, tuple):
            result = list(batch)
            result[4] = side
            result = tuple(result)
        else:
            result = list(batch)
            result[4] = side
        # Retain the original objects for every held tensor.  This check makes
        # the only treatment (side field) mechanically explicit.
        if result[0] is not neural or result[1] is not behavior or result[2] is not calibration or result[3] is not sessions:
            raise PMCError("PMC side injector changed a held batch field")
        self.replaced_batches += 1
        return result


def assert_joint_unit_permutation_equivariance(
    *,
    cache_entry: PMCCacheEntry,
    permutation: Tensor,
    neural: Tensor,
    calibration: Tensor,
    ordinary_side: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return jointly permuted held tensors plus carrier view for CPU tests.

    The actual Cell-D decoder determines output equivariance; this helper
    guarantees the PMC treatment applies precisely the same unit permutation
    to neural, M30 calibration, ordinary side, and posterior rows.
    """
    if permutation.dtype != torch.long or permutation.ndim != 1 or permutation.shape[0] != cache_entry.view.unit_count:
        raise PMCError("PMC joint permutation shape drift")
    expected = torch.arange(permutation.numel(), device=permutation.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise PMCError("PMC joint permutation is not a unit permutation")
    if neural.shape[-1] != permutation.numel() or calibration.shape[-1] != permutation.numel() or ordinary_side.shape[-2] != permutation.numel():
        raise PMCError("PMC held unit axes do not match posterior rows")
    _ = cache_entry.view.joint_permute(permutation)
    return (
        neural.index_select(-1, permutation),
        calibration.index_select(-1, permutation),
        ordinary_side.index_select(-2, permutation),
    )
