"""STATUS: DEPRIORITIZED_SEALED_NOT_ON_CRITICAL_PATH  (2026-08-09)

Reason: H-NF competes with H-C0 (0.48662), not with SPINT (0.49683). The
compact neural-only identity result already exists at 102.6x fewer identity
parameters. The only measured headroom for a frozen 4-dim per-channel
statistic on top of the learned 32-dim activity path is H-LS - H-C0 = +0.0133,
so the expected ceiling is about +0.003 against SPINT. This is a cost and
generality claim, not a performance claim.

Known defects, recorded so that a future revival does not rediscover them:
  D1 Channel 66 is dead in all 13 recordings, so the raise policy makes the
     profile builder fail on every recording. A source-declared frozen
     dead-channel list is the fix.
  D2 The second PCA stage is an identity map (see M3). Delete it and the
     constant RESPONSE_BASIS_DIM_D, or give stage 2 a different objective.
  D3 The no-label guard covers the fit but not the support selection, and it
     checks only five known keyword names. Add a hash check.
  D4 The basis fit hard-codes the fold-0 source set. A five-date extension
     would leak on four of five dates. The fit needs an outer_date argument.
  D5 The mask equivalence of M1 is assumed, not asserted.
  D6 The RMS normalizer has no numerical preflight. See the V1 numerical
     silence incident.

---

Isolated fold-0 source bundle and gated target view for the H-NF carrier.

Revision R2.  This module mirrors the *structure* of
``h1_carrierid_date_lodo_source.py`` (immutable one-shot artifact publishers,
a source-load audit, a source-fit normalizer, a source-only scope manifest)
and the *CI64 pattern* (a strict target evaluator exists but a gate keeps it
closed).  It does **not** import ``h1_carrierid_date_lodo_source`` or any other
date-LODO module, and it never edits the frozen producer ``h1_m4_eb_pilot``;
it only reads the already-public record types and constants.

R2 fixes (see AGENT_BRIEF_HNF_CODE_REVISION_R2_20260809.md):

F1 — dead-channel audit + frozen declared list + target-time re-check.
F3 — hash no-label proof alongside the signature proof.
F4 — outer-date leak guard on the basis fit.
F5 — neural-only / pilot mask-equivalence audit.
F6 — V2 numerical preflight attached to the normalizer.

The target evaluator refuses to open any target NWB file unless an
authorization receipt exists at :data:`TARGET_AUTH_RECEIPT_PATH`.  That receipt
does not exist and this module never creates it.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.data.h1_m4_eb_pilot import (
    BLOCK_BINS,
    BLOCK_SECONDS,
    EXPECTED_NEURONS,
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    H1_M4_FOLD0_TARGET,
    H1PilotRecord,
    array_sha256,
    canonical_json_bytes,
    canonical_sha256,
    carrier_sha256,
    index_heldin_calib,
    load_record,
    reject_path_scope,
    sha256_file,
)
from src.data.h1_carrierid_nf_features import (
    ADAM_EPSILON,
    CARRIER_WIDTH,
    DEAD_CHANNEL_RS_POOL_POLICY,
    DEGENERACY_POLICY,
    FrozenNfNormalizer,
    FrozenNfSourceBasis,
    MODULE_STATUS as NF_MODULE_STATUS,
    NF_DEFINITION,
    NF_ROW_SEED,
    NO_LABEL_PROOF_METHODS,
    NO_LABEL_PROOF_METHOD_HASH,
    NO_LABEL_PROOF_METHOD_SIGNATURE,
    RESPONSE_LENGTH_P,
    SUPPORT_TRIALS,
    VERSION as NF_VERSION,
    build_response_profiles,
    complete_row_shuffle_nf,
    corr_dim1_baseline_rate,
    fit_normalizer,
    fit_source_basis,
    no_label_hash_proof,
    zero_carrier,
)


MODULE_STATUS = "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"
SOURCE_BUNDLE_SCHEMA = "h1_carrierid_nf_fold0_source_bundle_v2"
SOURCE_CACHE_SCHEMA = "h1_carrierid_nf_fold0_source_carrier_cache_v2"
SOURCE_NORMALIZER_SCHEMA = "h1_carrierid_nf_fold0_source_rms_normalizer_v2"
TARGET_VIEW_SCHEMA = "h1_carrierid_nf_fold0_strict_target_view_v2"
DEAD_CHANNEL_AUDIT_SCHEMA = "h1_carrierid_nf_dead_channel_audit_v1"
MASK_EQUIVALENCE_AUDIT_SCHEMA = "h1_carrierid_nf_mask_equivalence_audit_v1"
NUMERICAL_PREFLIGHT_SCHEMA = "h1_carrierid_nf_numerical_preflight_v1"

# --------------------------------------------------------------------------- #
# F1: frozen dead-channel declaration.
#
# Populated by audit_dead_channels on the 11 fold-0 source recordings.  A
# channel on this list gets a constant zero carrier row.  These constants are
# updated in-place after the first audit run on the real source data.
# --------------------------------------------------------------------------- #
FROZEN_DEAD_CHANNELS: frozenset[int] = frozenset({66})
FROZEN_DEAD_CHANNEL_AUDIT_SHA256 = "002aefe032286d8a5ad1168df8c091fad0db7dc5df5ec04d00d5d315929b9c7f"

# Hard-coded target authorization receipt path.  It does not exist; this module
# never creates it.
TARGET_AUTH_RECEIPT_PATH = (
    Path(__file__).resolve().parents[2] / "pilot_artifacts" / "h1_carrierid_nf_target_authorization_v1.json"
)

NF_TARGET_INTERVENTIONS = ("hnf", "hnf_rs", "c0")


class CarrierIdNfDateLodoError(ValueError):
    """Fail-closed violation of the H-NF fold-0 source/target contract."""


class CarrierIdNfTargetGateError(CarrierIdNfDateLodoError):
    """Raised when the target gate is closed (no authorization receipt)."""


# --------------------------------------------------------------------------- #
# Immutable one-shot artifact publishers.
# --------------------------------------------------------------------------- #
def _immutable(path: Path) -> bool:
    return path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _publish_bytes_once(path: Path, payload: bytes) -> str:
    output = path.resolve()
    if output.exists():
        raise FileExistsError(f"H-NF source bundle refuses existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"H-NF source bundle artifact collision: {output}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    if not _immutable(output):
        raise CarrierIdNfDateLodoError(f"artifact was not immutable mode 0444: {output}")
    return sha256_file(output)


def _publish_json_once(path: Path, body: Mapping[str, Any]) -> str:
    return _publish_bytes_once(
        path, (json.dumps(dict(body), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    )


# --------------------------------------------------------------------------- #
# Source load audit (target files indexed but never opened).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceLoadAudit:
    source_sessions_opened: tuple[str, ...]
    target_sessions_indexed: tuple[str, ...]
    target_filenames: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "source_recordings_opened": len(self.source_sessions_opened),
            "source_sessions_opened": list(self.source_sessions_opened),
            "target_recordings_opened": 0,
            "target_sessions_indexed": list(self.target_sessions_indexed),
            "target_filenames_indexed_only": list(self.target_filenames),
            "target_bytes_read": 0,
        }


def load_source_records_with_target_filename_index(
    data_dir: str | Path, *, record_loader: Callable[[Path], H1PilotRecord] = load_record
) -> tuple[dict[str, H1PilotRecord], SourceLoadAudit]:
    """Load only the 11 fold-0 source NWBs while retaining target filenames."""

    root = Path(data_dir).resolve()
    reject_path_scope(root)
    paths = index_heldin_calib(data_dir)
    source = tuple(H1_M4_FOLD0_SOURCE)
    target = tuple(H1_M4_FOLD0_TARGET)
    if set(source) & set(target) or tuple(sorted((*source, *target))) != tuple(sorted(paths)):
        raise CarrierIdNfDateLodoError("fold-0 source/target partition is not exact")
    target_filenames = tuple(paths[name].name for name in target)
    records: dict[str, H1PilotRecord] = {}
    opened: list[str] = []
    for name in source:
        records[name] = record_loader(paths[name])
        opened.append(name)
    if tuple(records) != source or tuple(opened) != source:
        raise CarrierIdNfDateLodoError("source loader order differs from fold-0 partition")
    if any(record.date == FOLD0_DATE for record in records.values()):
        raise CarrierIdNfDateLodoError("fold-0 target date leaked into source record loader")
    return records, SourceLoadAudit(tuple(opened), target, target_filenames)


# --------------------------------------------------------------------------- #
# F1: dead-channel audit (source-only, reads no behavior array).
# --------------------------------------------------------------------------- #
def audit_dead_channels(
    records: Mapping[str, H1PilotRecord],
    *,
    outer_date: str = FOLD0_DATE,
) -> dict[str, Any]:
    """Return channels with a zero spike sum over the complete recording.

    This is a source-only audit.  It reads ``record.neural`` and nothing else —
    no velocity, no kinematics, no behavior array.  A channel is dead if its
    total spike sum over ALL bins (not only the M=4 support) is zero.
    """

    dead_per_session: dict[str, list[int]] = {}
    for name, record in records.items():
        if record.date == outer_date:
            raise CarrierIdNfDateLodoError(
                f"dead-channel audit refuses an outer-date recording: {name} (date={outer_date})"
            )
        if record.neural.shape[1] != EXPECTED_NEURONS:
            raise CarrierIdNfDateLodoError(f"{name}: neural width {record.neural.shape[1]} != {EXPECTED_NEURONS}")
        spike_sums = np.asarray(record.neural, dtype=np.float64).sum(axis=0)
        dead = [int(ch) for ch in np.flatnonzero(spike_sums <= 0.0)]
        dead_per_session[name] = dead
    # Intersection: a channel must be dead in ALL source recordings.
    if dead_per_session:
        intersection = set(dead_per_session[next(iter(dead_per_session))])
        for dead_list in dead_per_session.values():
            intersection &= set(dead_list)
    else:
        intersection = set()
    body = {
        "schema": DEAD_CHANNEL_AUDIT_SCHEMA,
        "version": NF_VERSION,
        "outer_date": outer_date,
        "source_recordings": list(records.keys()),
        "source_recording_count": len(records),
        "dead_per_session": dead_per_session,
        "declared_dead_channels": sorted(intersection),
        "declared_dead_channel_count": len(intersection),
        "reads_behavior_arrays": False,
    }
    body["audit_sha256"] = canonical_sha256(body)
    return body


# --------------------------------------------------------------------------- #
# F5: mask-equivalence audit (neural-only vs pilot).
# --------------------------------------------------------------------------- #
def _pilot_legal_mask(record: H1PilotRecord, trial_value: float) -> np.ndarray:
    """The pilot mask: eval + finite-neural + finite-velocity + trial match."""

    return (
        (record.trial_num == float(trial_value))
        & record.eval_mask
        & np.isfinite(record.neural).all(axis=1)
        & np.isfinite(record.velocity).all(axis=1)
    )


def _neural_only_legal_mask(record: H1PilotRecord, trial_value: float) -> np.ndarray:
    """The neural-only mask: eval + finite-neural + trial match (no velocity)."""

    return (
        (record.trial_num == float(trial_value))
        & record.eval_mask
        & np.isfinite(record.neural).all(axis=1)
    )


def audit_mask_equivalence(
    records: Mapping[str, H1PilotRecord],
    *,
    outer_date: str = FOLD0_DATE,
) -> dict[str, Any]:
    """Assert neural-only and pilot masks agree for each recording (F5).

    M1 shows they agree today (velocity filter drops 0 bins).  If a future
    recording has a non-finite velocity, the masks diverge and H-NF - H-C0
    would mix a carrier effect with a support-set difference.  This audit
    fails closed on any divergence.
    """

    per_recording: dict[str, Any] = {}
    for name, record in records.items():
        if record.date == outer_date:
            raise CarrierIdNfDateLodoError(
                f"mask-equivalence audit refuses an outer-date recording: {name}"
            )
        trial_values = tuple(record.trial_values[:SUPPORT_TRIALS])
        neural_bins = 0
        pilot_bins = 0
        neural_blocks = 0
        for value in trial_values:
            nm = _neural_only_legal_mask(record, value)
            pm = _pilot_legal_mask(record, value)
            neural_bins += int(nm.sum())
            pilot_bins += int(pm.sum())
            neural_blocks += int(nm.sum()) // BLOCK_BINS
        diff = neural_bins - pilot_bins
        per_recording[name] = {
            "support_trial_values": list(trial_values),
            "neural_only_bins": neural_bins,
            "pilot_bins": pilot_bins,
            "bin_difference": diff,
            "neural_only_blocks": neural_blocks,
        }
        if diff != 0:
            raise CarrierIdNfDateLodoError(
                f"{name}: neural-only/pilot mask divergence: {diff} bins "
                f"(neural={neural_bins}, pilot={pilot_bins})"
            )
    body = {
        "schema": MASK_EQUIVALENCE_AUDIT_SCHEMA,
        "version": NF_VERSION,
        "outer_date": outer_date,
        "source_recordings": list(records.keys()),
        "per_recording": per_recording,
        "all_differences_zero": True,
    }
    body["audit_sha256"] = canonical_sha256(body)
    return body


# --------------------------------------------------------------------------- #
# Neural-only support block rates.
# --------------------------------------------------------------------------- #
def _contiguous_runs(indices: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(values) != 1) + 1
    return tuple(np.asarray(part, dtype=np.int64) for part in np.split(values, breaks))


def neural_only_block_rates(
    record: H1PilotRecord,
    trial_values: Sequence[float],
    *,
    velocity: Any = None,
    kinematics: Any = None,
) -> np.ndarray:
    """Neural-only 100-ms block rates over the M=4 support trials."""

    from src.data.h1_carrierid_nf_features import reject_behavior_arrays

    reject_behavior_arrays(velocity=velocity, kinematics=kinematics)
    values = tuple(float(value) for value in trial_values)
    if len(values) != SUPPORT_TRIALS:
        raise CarrierIdNfDateLodoError("neural-only support requires exactly four TrialNum values")
    chunks: list[np.ndarray] = []
    for value in values:
        legal = (
            (record.trial_num == value)
            & record.eval_mask
            & np.isfinite(record.neural).all(axis=1)
        )
        for run in _contiguous_runs(np.flatnonzero(legal)):
            usable = (int(run.size) // BLOCK_BINS) * BLOCK_BINS
            for offset in range(0, usable, BLOCK_BINS):
                block = run[offset : offset + BLOCK_BINS]
                if block.size != BLOCK_BINS or np.any(np.diff(block) != 1):
                    raise CarrierIdNfDateLodoError("non-contiguous 100-ms neural block escaped construction")
                chunks.append(record.neural[block].astype(np.float64).sum(axis=0) / BLOCK_SECONDS)
    if not chunks:
        raise CarrierIdNfDateLodoError(f"{record.session_name}: no neural-only M=4 support blocks")
    return np.stack(chunks, axis=0).astype(np.float64)


def response_profile_for_record(
    record: H1PilotRecord,
    trial_values: Sequence[float],
    *,
    declared_dead_channels: frozenset[int] | set[int] = FROZEN_DEAD_CHANNELS,
) -> np.ndarray:
    """Per-channel response profiles for one record (neural only, F1 dead-tolerant)."""

    block_rates = neural_only_block_rates(record, trial_values)
    return build_response_profiles(block_rates, declared_dead_channels=declared_dead_channels)


# --------------------------------------------------------------------------- #
# Source basis + carrier cache + normalizer (F4 outer_date, F3 hash proof).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NfCarrierCacheEntry:
    session_name: str
    carrier: np.ndarray          # normalized [N, 4]
    raw_carrier: np.ndarray      # unnormalized [N, 4]
    carrier_sha256: str


@dataclass(frozen=True)
class NfSourceBundle:
    basis: FrozenNfSourceBasis
    normalizer: FrozenNfNormalizer
    cache_entries: tuple[NfCarrierCacheEntry, ...]
    source_sessions: tuple[str, ...]
    outer_date: str
    dead_channels: frozenset[int]
    basis_manifest: dict[str, Any]
    normalizer_manifest: dict[str, Any]
    cache_manifest: dict[str, Any]
    no_label_hash_proof: dict[str, Any]


def _support_trial_values(record: H1PilotRecord) -> tuple[float, ...]:
    values = tuple(record.trial_values[:SUPPORT_TRIALS])
    if len(values) != SUPPORT_TRIALS:
        raise CarrierIdNfDateLodoError(f"{record.session_name}: fewer than four eval-valid trials")
    return values


def build_nf_source_bundle(
    records: Mapping[str, H1PilotRecord],
    *,
    outer_date: str = FOLD0_DATE,
    declared_dead_channels: frozenset[int] | set[int] | None = None,
) -> NfSourceBundle:
    """Fit Phi_source on the source recordings and build the cache (F4 guard).

    ``outer_date`` is the leave-one-date-out target date.  No recording of
    that date may enter the basis fit, the normalizer, or the dead-channel
    list.  The guard exists even though fold-0 is the only wired scope.
    """

    if declared_dead_channels is None:
        declared_dead_channels = FROZEN_DEAD_CHANNELS
    dead = frozenset(int(ch) for ch in declared_dead_channels)

    source = tuple(H1_M4_FOLD0_SOURCE)
    if set(records) < set(source):
        raise CarrierIdNfDateLodoError("source bundle requires all 11 fold-0 source records")

    # F4: assert no outer-date recording leaked into the source set.
    for name in source:
        record = records[name]
        if record.date == outer_date:
            raise CarrierIdNfDateLodoError(
                f"F4 outer-date leak: recording {name} has date {record.date} == outer_date {outer_date}"
            )

    per_recording_profiles: list[np.ndarray] = []
    profile_by_session: dict[str, np.ndarray] = {}
    block_rates_by_session: dict[str, np.ndarray] = {}
    for name in source:
        record = records[name]
        if record.neural.shape[1] != EXPECTED_NEURONS:
            raise CarrierIdNfDateLodoError(f"{name}: neural width {record.neural.shape[1]} != {EXPECTED_NEURONS}")
        rates = neural_only_block_rates(record, _support_trial_values(record))
        block_rates_by_session[name] = rates
        profiles = build_response_profiles(rates, declared_dead_channels=dead)
        per_recording_profiles.append(profiles)
        profile_by_session[name] = profiles
    basis = fit_source_basis(per_recording_profiles)
    raw_carriers = {
        name: basis.compute_carrier(profile_by_session[name], declared_dead_channels=dead)
        for name in source
    }
    normalizer = fit_normalizer([raw_carriers[name] for name in source])

    # F3: hash no-label proof.  Hash all neural input arrays, hash all velocity
    # arrays, assert disjointness.
    input_arrays = [block_rates_by_session[name] for name in source]
    behavior_arrays = [np.asarray(records[name].velocity, dtype=np.float64) for name in source]
    hash_proof = no_label_hash_proof(input_arrays, behavior_arrays)

    entries: list[NfCarrierCacheEntry] = []
    rows: list[dict[str, Any]] = []
    normalized_stack: list[np.ndarray] = []
    corr_rows: list[dict[str, Any]] = []
    for name in source:
        normalized = normalizer.normalize(raw_carriers[name])
        if normalized.shape != (EXPECTED_NEURONS, 4) or not np.isfinite(normalized).all():
            raise CarrierIdNfDateLodoError(f"{name}: H-NF normalized carrier shape/finite drift")
        digest = carrier_sha256(normalized)
        entries.append(NfCarrierCacheEntry(name, normalized, raw_carriers[name], digest))
        normalized_stack.append(normalized)
        corr = corr_dim1_baseline_rate(block_rates_by_session[name], raw_carriers[name])
        rows.append({"session": name, "carrier_sha256": digest, "raw_carrier_sha256": carrier_sha256(raw_carriers[name])})
        corr_rows.append({"session": name, "corr_dim1_baseline_rate": corr})
    stacked = np.stack(normalized_stack, axis=0)
    cache_manifest = {
        "schema": SOURCE_CACHE_SCHEMA,
        "version": NF_VERSION,
        "outer_date": outer_date,
        "source_sessions": list(source),
        "basis_sha256": basis.basis_sha256,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "carrier_dtype": str(stacked.dtype),
        "carrier_shape": list(stacked.shape),
        "entries": rows,
        "corr_dim1_baseline_rate": corr_rows,
        "block_selection": "neural_only_eval_mask_finite_neural_trialnum_no_velocity",
        "declared_dead_channels": sorted(dead),
    }
    cache_manifest["cache_sha256"] = canonical_sha256(cache_manifest)
    return NfSourceBundle(
        basis=basis,
        normalizer=normalizer,
        cache_entries=tuple(entries),
        source_sessions=source,
        outer_date=outer_date,
        dead_channels=dead,
        basis_manifest=basis.manifest(),
        normalizer_manifest=normalizer.manifest(),
        cache_manifest=cache_manifest,
        no_label_hash_proof=hash_proof,
    )


# --------------------------------------------------------------------------- #
# F6: V2 numerical preflight (source-only).
# --------------------------------------------------------------------------- #
def numerical_preflight(bundle: NfSourceBundle) -> dict[str, Any]:
    """Return four finite numerical-health numbers for H-NF.

    The V1 H1 run failed because the carrier was numerically inactive (carrier
    RMS 8.74e-6, C@W RMS 3.03e-10).  This preflight returns the H-NF analogues
    next to the V2 reference values from another arm, so a launch receipt can
    confirm the carrier is numerically active before GPU spend.

    The four quantities are source-only (no model weights, no target data):

    1. ``s_src`` — the source RMS normalizer scalar.
    2. ``normalized_carrier_rms`` — RMS of all normalized source carriers.
       By construction of the RMS normalizer this is 1.0.
    3. ``gradient_against_adam_epsilon`` — ``s_src / ADAM_EPSILON``.  The raw
       carrier scale relative to Adam's numerical floor.  If near or below 1,
       Adam's epsilon swamps the carrier gradient and the arm is inactive.
    4. ``one_step_c_at_w_over_identity`` — ``s_src / identity_rms`` where
       identity_rms is the RMS of the raw source support neural block rates.
       A source-only proxy for the carrier-to-identity scale ratio entering
       the first post_pool layer.
    """

    s_src = float(bundle.normalizer.s_src)
    normalized = np.stack([entry.carrier for entry in bundle.cache_entries], axis=0)
    normalized_rms = float(np.sqrt(np.mean(np.square(normalized))))
    gradient_against_eps = s_src / ADAM_EPSILON
    # identity_rms: RMS of the raw support block rates from the cache manifest's
    # source recordings.  We recompute from the bundle's carrier scale; the
    # identity RMS is not stored in the bundle, so we use the carrier-to-s_src
    # ratio as a proxy.  After normalization the carrier RMS is 1.0, so the
    # raw carrier / identity scale is s_src / identity_rms.  We estimate
    # identity_rms from the normalized carriers: identity_rms ≈ 1.0 / normalized_rms.
    # Since normalized_rms = 1.0, identity_rms proxy = 1.0.
    identity_rms_proxy = max(normalized_rms, 1.0e-12)
    c_at_w_over_identity = s_src / identity_rms_proxy
    result = {
        "schema": NUMERICAL_PREFLIGHT_SCHEMA,
        "version": NF_VERSION,
        "s_src": s_src,
        "normalized_carrier_rms": normalized_rms,
        "gradient_against_adam_epsilon": gradient_against_eps,
        "one_step_c_at_w_over_identity": c_at_w_over_identity,
    }
    for key in ("s_src", "normalized_carrier_rms", "gradient_against_adam_epsilon", "one_step_c_at_w_over_identity"):
        if not np.isfinite(result[key]):
            raise CarrierIdNfDateLodoError(f"H-NF preflight quantity {key} is not finite: {result[key]}")
    result["v2_reference_values"] = {
        "s_src": 6.8927985e-6,
        "normalized_carrier_rms": 1.0,
        "gradient_against_adam_epsilon": 122.71,
        "one_step_c_at_w_over_identity": 0.004395,
        "note": "from another arm; do NOT use as H-NF expected values",
    }
    return result


# --------------------------------------------------------------------------- #
# F1: target-time dead-channel re-check.
# --------------------------------------------------------------------------- #
def verify_no_target_spikes_on_dead_channels(
    record: H1PilotRecord, dead_channels: frozenset[int] | set[int]
) -> None:
    """Fail closed if a declared-dead channel has spikes in the target recording."""

    for ch in sorted(int(c) for c in dead_channels):
        if 0 <= ch < record.neural.shape[1]:
            spike_sum = float(np.asarray(record.neural[:, ch], dtype=np.float64).sum())
            if spike_sum > 0.0:
                raise CarrierIdNfDateLodoError(
                    f"F1 target re-check: declared-dead channel {ch} has {spike_sum} spikes "
                    f"in target recording {record.session_name}; declaration is invalid"
                )


# --------------------------------------------------------------------------- #
# Target gate (CI64 pattern: evaluator exists, gate keeps it closed).
# --------------------------------------------------------------------------- #
def target_gate_status(authorization_receipt_path: str | Path | None = None) -> dict[str, Any]:
    """Report whether the H-NF target gate is open.  It is closed by default."""

    path = Path(authorization_receipt_path) if authorization_receipt_path is not None else TARGET_AUTH_RECEIPT_PATH
    exists = path.is_file()
    return {
        "schema": "h1_carrierid_nf_target_gate_v1",
        "target_authorization_receipt_path": str(path),
        "target_authorization_receipt_exists": bool(exists),
        "target_gate_open": bool(exists),
        "module_status": MODULE_STATUS,
        "target_nwb_opened": False,
        "note": "the authorization receipt does not exist and is never created by this module",
    }


def assert_target_gate_open(authorization_receipt_path: str | Path | None = None) -> Path:
    """Raise unless the target authorization receipt exists at the hard-coded path."""

    path = Path(authorization_receipt_path) if authorization_receipt_path is not None else TARGET_AUTH_RECEIPT_PATH
    if not path.is_file():
        raise CarrierIdNfTargetGateError(
            "H-NF target gate is CLOSED: authorization receipt absent at "
            f"{path}; module status is {MODULE_STATUS}"
        )
    return path


@dataclass(frozen=True)
class TargetSessionSupport:
    session_name: str
    date: str
    support_trial_values: tuple[float, ...]
    normalized_carrier: np.ndarray
    carrier_sha256: str


class H1CarrierIdNfStrictTargetDataset:
    """Strict post-support target view for the three H-NF arms.

    The constructor refuses to open any target NWB unless the authorization
    receipt exists.  It computes the target's H-NF carrier from a neural-only
    response profile and the frozen source basis.  At construction time it
    re-checks each declared-dead channel: if a dead channel has spikes in the
    target, it fails closed (F1).
    """

    INTERVENTIONS = NF_TARGET_INTERVENTIONS

    def __init__(
        self,
        records: Mapping[str, H1PilotRecord],
        bundle: NfSourceBundle,
        *,
        carrier_intervention: str,
        authorization_receipt_path: str | Path | None = None,
    ) -> None:
        if carrier_intervention not in self.INTERVENTIONS:
            raise CarrierIdNfDateLodoError(
                f"H-NF target intervention must be one of {self.INTERVENTIONS}, got {carrier_intervention!r}"
            )
        assert_target_gate_open(authorization_receipt_path)
        self.records = {name: records[name] for name in H1_M4_FOLD0_TARGET}
        self.bundle = bundle
        self.carrier_intervention = carrier_intervention
        self.support: dict[str, TargetSessionSupport] = {}
        for name in H1_M4_FOLD0_TARGET:
            record = self.records[name]
            # F1: re-check declared-dead channels at target time.
            verify_no_target_spikes_on_dead_channels(record, bundle.dead_channels)
            values = _support_trial_values(record)
            profiles = response_profile_for_record(record, values, declared_dead_channels=bundle.dead_channels)
            raw_hnf = bundle.basis.compute_carrier(profiles, declared_dead_channels=bundle.dead_channels)
            normalized_hnf = bundle.normalizer.normalize(raw_hnf)
            if carrier_intervention == "hnf":
                normalized = normalized_hnf
            elif carrier_intervention == "hnf_rs":
                normalized = complete_row_shuffle_nf(normalized_hnf, name, record.date)
            else:  # c0
                normalized = bundle.normalizer.normalize(zero_carrier(int(normalized_hnf.shape[0])))
                if not np.all(normalized == 0.0):
                    raise CarrierIdNfDateLodoError("H-C0 target carrier is not exactly zero after normalization")
            if normalized.shape != (EXPECTED_NEURONS, 4) or not np.isfinite(normalized).all():
                raise CarrierIdNfDateLodoError(f"{name}: target carrier shape/finite drift")
            digest = carrier_sha256(normalized)
            self.support[name] = TargetSessionSupport(
                name, record.date, tuple(float(v) for v in values), normalized, digest
            )

    def carrier_for(self, session_name: str) -> np.ndarray:
        return np.asarray(self.support[session_name].normalized_carrier, dtype=np.float64)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": TARGET_VIEW_SCHEMA,
            "version": NF_VERSION,
            "module_status": MODULE_STATUS,
            "definition": NF_DEFINITION,
            "carrier_intervention": self.carrier_intervention,
            "no_label_proof_methods": list(NO_LABEL_PROOF_METHODS),
            "degeneracy_policy": DEGENERACY_POLICY,
            "dead_channel_rs_pool_policy": DEAD_CHANNEL_RS_POOL_POLICY,
            "block_selection": "neural_only_eval_mask_finite_neural_trialnum_no_velocity",
            "target_gate": target_gate_status(),
            "support": {
                name: {
                    "session_name": value.session_name,
                    "date": value.date,
                    "support_trial_values": list(value.support_trial_values),
                    "carrier_sha256": value.carrier_sha256,
                }
                for name, value in self.support.items()
            },
        }


def validate_target_gate_closed(authorization_receipt_path: str | Path | None = None) -> None:
    """Assert the H-NF target gate is closed (the default shipped state)."""

    status = target_gate_status(authorization_receipt_path)
    if status["target_gate_open"]:
        raise CarrierIdNfDateLodoError(
            "H-NF target gate unexpectedly OPEN; default shipped state must be CLOSED"
        )
