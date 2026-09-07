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

H-NF: a deliberately label-free H1 CarrierID carrier (revision R2, NF-v1).

This module is intentionally a *standalone numerical* module.  It accepts
already-materialized neural block-rate arrays and produces a 4-D per-channel
carrier.  It has no dataset, checkpoint, decoder, or target-file dependency,
and it imports nothing from the frozen live producer graph (no
``h1_m4_eb_pilot``, no date-LODO module).  It is therefore safe to land while a
matched-date matrix is in flight.

Arm definition (frozen in :data:`NF_DEFINITION`; do not make it configurable)::

    e_i = Phi_source^T (p_i - p_mean)     # the carrier for channel i, width 4

* ``p_i`` is the neural-only response profile of channel ``i`` (the calibration
  block-rate trajectory over the M=4 support trials, resampled to P=24 points).
  It is NOT a set of hand-chosen marginal statistics (N4 already tested
  [mean_rate, Fano, lag-1 autocorrelation, population_coupling] and failed).
* ``Phi_source`` holds the top-4 right singular vectors of the centered pooled
  source profiles, fit on the 11 fold-0 source recordings.  It is a genuine
  fitted projection; there is no second stage.
* The target session reads no velocity label, no kinematics, and no behavior
  value of any kind.  Two no-label proofs run: the signature method rejects
  non-None behavior kwargs, and the hash method asserts that no input-array
  hash collides with any behavior-array hash (:data:`NO_LABEL_PROOF_METHODS`).
* The width is 4.  The consumer stays the 58,140-parameter identity encoder.

Dead-channel policy (F1).  A frozen source-declared dead-channel list records
channels with a zero spike sum over the complete source recording.  A channel
on that list gets an explicit constant zero carrier row.  A degenerate channel
NOT on the list still raises.  Declared-dead rows join the H-NF-RS permutation
pool so H-NF and H-NF-RS hold the same multiset of rows.

The four arms:

``H-NF``
    The carrier above with correct channel attachment.
``H-NF-RS``
    A deterministic non-identity complete-row permutation of the normalized
    H-NF carrier.  Whole four-value rows are permuted; columns are never
    permuted independently; the carrier/normalizer are never refit.
``H-C0``
    The exact zero vector at matched width 4.  After the source normalizer it
    stays exactly zero.
reference (``H-C``, ``H-LS``)
    Stay as they are.  This module does not recompute them.

Status string: ``IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED``.

Candidate ``NF-v2`` (NOT built): a basis fit for cross-recording stability via
the W3 alignment-score objective.  W3 defines that objective and W3 has not
run, so NF-v2 is a comment-only candidate.  A change to P or the basis objective
needs a new version string and a new module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


MODULE_STATUS = "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"

# --- Frozen version and representation dimensions (F2) ----------------------
# Version NF-v1 freezes P=24 inside the version string.  A change of P or the
# basis objective requires a new version string and a new module.
VERSION = "NF-v1"
CARRIER_WIDTH = 4
SUPPORT_TRIALS = 4
RESPONSE_LENGTH_P = 24
BLOCK_SECONDS = 0.1
EPS = 1.0e-12
NORMALIZER_FLOOR = 1.0e-6
ADAM_EPSILON = 1.0e-8

# --- No-label proofs (F3) ---------------------------------------------------
# Two methods run side by side.  Method 1 (signature) rejects non-None behavior
# kwargs.  Method 2 (hash) asserts that no input-array hash collides with any
# behavior-array hash, covering positional and unknown-name attack surfaces.
NO_LABEL_PROOF_METHOD_SIGNATURE = "signature_rejects_nonnone_behavior_arrays"
NO_LABEL_PROOF_METHOD_HASH = "hash_disjointness_against_velocity"
NO_LABEL_PROOF_METHODS = (NO_LABEL_PROOF_METHOD_SIGNATURE, NO_LABEL_PROOF_METHOD_HASH)

# --- Dead-channel policy (F1) -----------------------------------------------
# The constant below is populated by audit_dead_channels on the 11 fold-0
# source recordings.  A channel on this list gets a constant zero carrier row.
# Declared-dead rows join the H-NF-RS permutation pool.
DEGENERACY_POLICY = "declared_dead_zero_row_undeclared_raise"
DEAD_CHANNEL_RS_POOL_POLICY = "declared_dead_rows_join_hnf_rs_permutation_pool"

# Placeholder — replaced with the real audit output after running audit_dead_channels
# on the 11 fold-0 source recordings.  See h1_carrierid_nf_date_lodo.py for the
# audit function and the frozen receipt hash.
DECLARED_DEAD_CHANNELS: frozenset[int] = frozenset()
DECLARED_DEAD_CARRIER_ROW = np.zeros(CARRIER_WIDTH, dtype=np.float64)
DEAD_CHANNEL_AUDIT_SHA256 = ""

# Deterministic complete-row-permutation seed schedule for H-NF-RS.
NF_ROW_SEED = 20260809

NF_DEFINITION = f"""\
H-NF label-free carrier (frozen definition, version {VERSION}, P={RESPONSE_LENGTH_P}).

Support: the first M={SUPPORT_TRIALS} calibration trials.  For each channel i,
let r_i be its neural-only block-rate trajectory (Hz) over those trials.  No
velocity, no kinematics, and no behavior value is read at any point.

Per-channel response profile:
    p_i = resample(r_i, length={RESPONSE_LENGTH_P})   (linear interpolation)

Source fit (on the 11 fold-0 source recordings, pooled over all source channels):
    p_mean     = mean of all source p_i                         [{RESPONSE_LENGTH_P}]
    Phi_source = top-{CARRIER_WIDTH} right singular vectors of (P_src - p_mean)
                 stored as [{RESPONSE_LENGTH_P}, {CARRIER_WIDTH}] orthonormal columns

Carrier (width {CARRIER_WIDTH}):
    e_i = Phi_source^T (p_i - p_mean)

There is no second PCA stage.  An earlier two-stage design (Phi_source then
U_source) was removed because U_source was provably the truncated identity.

Normalizer (fits on the source H-NF carriers of the same fold only, then freezes):
    s_src      = sqrt(mean(e_i**2) over all source channels and all 4 dims)
    normalize(C) = C / max(s_src, {NORMALIZER_FLOOR})
This is a pure scalar scale, so the H-C0 zero carrier stays exactly zero.

Dead-channel policy: {DEGENERACY_POLICY}.  A channel on the source-declared
dead-channel list (zero spike sum over the complete source recording) gets a
constant zero carrier row.  {DEAD_CHANNEL_RS_POOL_POLICY}.  A degenerate channel
not on the list still raises; it is never filled with a zero, a median, or any
other silent value.

No-label proofs: (1) {NO_LABEL_PROOF_METHOD_SIGNATURE} — every fit path accepts
behavior arrays only as None and raises on a non-None value.  (2)
{NO_LABEL_PROOF_METHOD_HASH} — a hash of every input array is checked against a
hash of every behavior array, catching positional and unknown-name leaks.

Candidate NF-v2 (not built): a basis fit for cross-recording stability via the
W3 alignment-score objective.  W3 has not run, so NF-v2 is a comment only.

Controls:
    H-NF-RS = a deterministic complete-row permutation of normalize(H-NF), keyed
              by (recording, date, seed={NF_ROW_SEED}).  It permutes whole
              four-value rows; it does not permute columns independently and it
              does not refit the carrier or the normalizer.
    H-C0    = the exact zero vector [0,0,0,0] at matched width {CARRIER_WIDTH}.
    H-C, H-LS = referenced from the frozen producer graph; never recomputed here.
"""


class CarrierNfError(ValueError):
    """Fail-closed violation of the H-NF label-free carrier contract."""


# --------------------------------------------------------------------------- #
# No-label proof method 1: signature rejects non-None behavior kwargs (F3).
# --------------------------------------------------------------------------- #
_BEHAVIOR_KWARG_NAMES = ("labels", "velocity", "kinematics", "behavior", "target_kinematics")


def reject_behavior_arrays(**behavior_kwargs: Any) -> None:
    """Raise if any behavior-named array is not None (signature method)."""

    for name in _BEHAVIOR_KWARG_NAMES:
        value = behavior_kwargs.get(name, None)
        if value is not None:
            raise CarrierNfError(
                f"H-NF fit rejects a non-None behavior array ({name}={type(value).__name__}); "
                f"the carrier is label-free by construction ({NO_LABEL_PROOF_METHOD_SIGNATURE})"
            )


# --------------------------------------------------------------------------- #
# No-label proof method 2: hash disjointness (F3).
# --------------------------------------------------------------------------- #
def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def no_label_hash_proof(
    input_arrays: Sequence[np.ndarray],
    behavior_arrays: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Assert that no input-array hash matches any behavior-array hash.

    This is the hash no-label proof (F3 method 2).  It covers attack surfaces
    that the signature method misses: a behavior array passed as a positional
    argument or under an unknown keyword name.  Every array that enters the
    block selection and the carrier fit must be in ``input_arrays``; every
    behavior array of the recording (velocity at minimum) must be in
    ``behavior_arrays``.  The proof asserts hash-set disjointness.
    """

    input_hashes = {_array_sha256(arr) for arr in input_arrays}
    behavior_hashes = {_array_sha256(arr) for arr in behavior_arrays}
    collisions = input_hashes & behavior_hashes
    if collisions:
        raise CarrierNfError(
            f"H-NF no-label hash proof FAILED: {len(collisions)} input array(s) "
            f"hash-match a behavior array; the carrier must be label-free "
            f"({NO_LABEL_PROOF_METHOD_HASH})"
        )
    return {
        "method": NO_LABEL_PROOF_METHOD_HASH,
        "n_input_arrays_checked": len(input_arrays),
        "n_behavior_arrays_checked": len(behavior_arrays),
        "collisions": 0,
        "passed": True,
    }


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _as_finite_f64(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise CarrierNfError(f"{name} must be {ndim}-D, got {array.ndim}-D")
    if not np.isfinite(array).all():
        raise CarrierNfError(f"{name} must be finite")
    return array


# --------------------------------------------------------------------------- #
# Response profile (neural only, with dead-channel tolerance — F1).
# --------------------------------------------------------------------------- #
def resample_trajectory(trajectory: np.ndarray, *, length: int = RESPONSE_LENGTH_P) -> np.ndarray:
    """Linearly resample a 1-D block-rate trajectory to a fixed length."""

    values = _as_finite_f64(trajectory, name="trajectory", ndim=1)
    if values.size < 2:
        raise CarrierNfError("a response trajectory needs at least two support blocks")
    x_old = np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, int(length), dtype=np.float64)
    return np.asarray(np.interp(x_new, x_old, values), dtype=np.float64)


def build_response_profiles(
    block_rates: np.ndarray,
    *,
    declared_dead_channels: frozenset[int] | set[int] = frozenset(),
    labels: Any = None,
    velocity: Any = None,
    kinematics: Any = None,
) -> np.ndarray:
    """Build per-channel response profiles from neural-only block rates.

    ``block_rates`` has shape ``[n_blocks, n_channels]`` and contains only
    neural block rates (Hz).  A channel on ``declared_dead_channels`` gets a
    zero profile (its carrier row will be set to the constant zero row later).
    A degenerate channel NOT on the declared list raises according to the
    frozen degeneracy policy.
    """

    reject_behavior_arrays(labels=labels, velocity=velocity, kinematics=kinematics)
    rates = _as_finite_f64(block_rates, name="block_rates", ndim=2)
    n_blocks, n_channels = rates.shape
    if n_blocks < 2:
        raise CarrierNfError("response profile needs at least two support blocks per channel")
    if n_channels == 0:
        raise CarrierNfError("response profile needs at least one channel")
    if np.any(rates < -EPS):
        raise CarrierNfError("block rates must be nonnegative (spike counts over a fixed window)")
    dead = frozenset(int(ch) for ch in declared_dead_channels)
    profiles = np.empty((n_channels, RESPONSE_LENGTH_P), dtype=np.float64)
    for channel in range(n_channels):
        column = rates[:, channel]
        if channel in dead:
            profiles[channel] = 0.0
            continue
        if not np.any(column > 0.0):
            raise CarrierNfError(
                f"channel {channel}: zero support spikes (all-zero trajectory) and "
                f"NOT on declared-dead list; degeneracy policy is {DEGENERACY_POLICY}"
            )
        if np.ptp(column) <= EPS:
            raise CarrierNfError(
                f"channel {channel}: constant trajectory (zero variance) and "
                f"NOT on declared-dead list; degeneracy policy is {DEGENERACY_POLICY}"
            )
        profiles[channel] = resample_trajectory(column)
    return profiles


# --------------------------------------------------------------------------- #
# Source basis fit (one-stage PCA — F2) + carrier computation.
# --------------------------------------------------------------------------- #
def _stack_profiles(per_recording_profiles: Sequence[np.ndarray]) -> np.ndarray:
    if len(per_recording_profiles) < 2:
        raise CarrierNfError("the source basis needs at least two source recordings")
    stacked: list[np.ndarray] = []
    for index, profiles in enumerate(per_recording_profiles):
        array = _as_finite_f64(profiles, name=f"source_profiles[{index}]", ndim=2)
        if array.shape[1] != RESPONSE_LENGTH_P:
            raise CarrierNfError(
                f"source_profiles[{index}] width {array.shape[1]} != frozen P={RESPONSE_LENGTH_P}"
            )
        if array.shape[0] == 0:
            raise CarrierNfError(f"source_profiles[{index}] has no channels")
        stacked.append(array)
    pooled = np.concatenate(stacked, axis=0)
    if pooled.shape[0] < CARRIER_WIDTH:
        raise CarrierNfError(
            f"pooled source channels ({pooled.shape[0]}) fewer than width={CARRIER_WIDTH}"
        )
    return pooled


@dataclass(frozen=True)
class FrozenNfSourceBasis:
    """Frozen source-fit response projection (Phi_source, one-stage PCA).

    The carrier is ``e_i = Phi_source^T (p_i - p_mean)`` with no second stage.
    """

    p_mean: np.ndarray       # [P]
    phi_source: np.ndarray   # [P, CARRIER_WIDTH], orthonormal columns
    source_recording_count: int
    source_channel_count: int
    profile_hashes: tuple[str, ...]
    basis_sha256: str

    def __post_init__(self) -> None:
        p_mean = _as_finite_f64(self.p_mean, name="p_mean", ndim=1)
        phi = _as_finite_f64(self.phi_source, name="phi_source", ndim=2)
        if p_mean.shape[0] != RESPONSE_LENGTH_P:
            raise CarrierNfError(f"p_mean length {p_mean.shape[0]} != P={RESPONSE_LENGTH_P}")
        if phi.shape != (RESPONSE_LENGTH_P, CARRIER_WIDTH):
            raise CarrierNfError(f"phi_source shape {phi.shape} != [P, {CARRIER_WIDTH}]")
        if not np.allclose(phi.T @ phi, np.eye(CARRIER_WIDTH), atol=1.0e-10):
            raise CarrierNfError("phi_source columns must be orthonormal")
        object.__setattr__(self, "p_mean", np.ascontiguousarray(p_mean))
        object.__setattr__(self, "phi_source", np.ascontiguousarray(phi))

    def project(self, profiles: np.ndarray) -> np.ndarray:
        values = _as_finite_f64(profiles, name="profiles", ndim=2)
        if values.shape[1] != RESPONSE_LENGTH_P:
            raise CarrierNfError(f"profiles width {values.shape[1]} != P={RESPONSE_LENGTH_P}")
        return (values - self.p_mean[None, :]) @ self.phi_source

    def compute_carrier(
        self,
        profiles: np.ndarray,
        *,
        declared_dead_channels: frozenset[int] | set[int] = frozenset(),
        labels: Any = None,
        velocity: Any = None,
        kinematics: Any = None,
    ) -> np.ndarray:
        reject_behavior_arrays(labels=labels, velocity=velocity, kinematics=kinematics)
        carrier = self.project(profiles)
        dead = frozenset(int(ch) for ch in declared_dead_channels)
        for ch in sorted(dead):
            if 0 <= ch < carrier.shape[0]:
                carrier[ch] = DECLARED_DEAD_CARRIER_ROW
        if carrier.shape[1] != CARRIER_WIDTH or not np.isfinite(carrier).all():
            raise CarrierNfError("H-NF carrier must be finite [N, 4]")
        return np.asarray(carrier, dtype=np.float64)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "h1_carrierid_nf_frozen_source_basis_v1",
            "version": VERSION,
            "module_status": MODULE_STATUS,
            "definition": NF_DEFINITION,
            "carrier_width": CARRIER_WIDTH,
            "support_trials": SUPPORT_TRIALS,
            "response_length_p": RESPONSE_LENGTH_P,
            "source_recording_count": int(self.source_recording_count),
            "source_channel_count": int(self.source_channel_count),
            "degeneracy_policy": DEGENERACY_POLICY,
            "dead_channel_rs_pool_policy": DEAD_CHANNEL_RS_POOL_POLICY,
            "no_label_proof_methods": list(NO_LABEL_PROOF_METHODS),
            "p_mean_sha256": _array_sha256(self.p_mean),
            "phi_source_sha256": _array_sha256(self.phi_source),
            "p_mean_shape": list(self.p_mean.shape),
            "phi_source_shape": list(self.phi_source.shape),
            "source_profile_hashes": list(self.profile_hashes),
            "basis_sha256": self.basis_sha256,
        }


def fit_source_basis(
    per_recording_profiles: Sequence[np.ndarray],
    *,
    labels: Any = None,
    velocity: Any = None,
    kinematics: Any = None,
) -> FrozenNfSourceBasis:
    """Fit Phi_source (top-4 right singular vectors) on pooled source profiles.

    Purely unsupervised (one-stage truncated SVD / PCA).  No label is read.
    """

    reject_behavior_arrays(labels=labels, velocity=velocity, kinematics=kinematics)
    pooled = _stack_profiles(per_recording_profiles)
    p_mean = pooled.mean(axis=0)
    centered = pooled - p_mean[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    phi_source = np.asarray(vt[:CARRIER_WIDTH].T, dtype=np.float64)
    profile_hashes = tuple(_array_sha256(profiles) for profiles in per_recording_profiles)
    body = {
        "version": VERSION,
        "p_mean_sha256": _array_sha256(p_mean),
        "phi_source_sha256": _array_sha256(phi_source),
        "source_profile_hashes": list(profile_hashes),
        "source_recording_count": len(per_recording_profiles),
        "source_channel_count": int(pooled.shape[0]),
    }
    return FrozenNfSourceBasis(
        p_mean=p_mean,
        phi_source=phi_source,
        source_recording_count=len(per_recording_profiles),
        source_channel_count=int(pooled.shape[0]),
        profile_hashes=profile_hashes,
        basis_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest(),
    )


def corr_dim1_baseline_rate(
    block_rates: np.ndarray, carrier: np.ndarray
) -> float:
    """Pearson correlation between carrier dimension 1 and the baseline rate.

    The first principal component of a rate trajectory is close to the mean
    rate, so dimension 1 of the carrier is probably the baseline rate.  This
    correlation records the attribution so a later reader cannot call H-NF dim 1
    new information when it is only the baseline rate (B4 alone is 0.287273).
    """

    rates = _as_finite_f64(block_rates, name="block_rates", ndim=2)
    carr = _as_finite_f64(carrier, name="carrier", ndim=2)
    if rates.shape[1] != carr.shape[0]:
        raise CarrierNfError("block_rates columns must match carrier rows (channels)")
    baseline = rates.mean(axis=0)
    dim1 = carr[:, 0]
    if np.std(baseline) < EPS or np.std(dim1) < EPS:
        raise CarrierNfError("baseline rate or carrier dim 1 has zero variance; correlation undefined")
    return float(np.corrcoef(baseline, dim1)[0, 1])


# --------------------------------------------------------------------------- #
# Source normalizer (fits on source H-NF carriers, then freezes).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrozenNfNormalizer:
    s_src: float
    source_carrier_count: int
    normalizer_sha256: str

    @property
    def denominator(self) -> float:
        return max(float(self.s_src), NORMALIZER_FLOOR)

    def normalize(self, carrier: np.ndarray) -> np.ndarray:
        value = _as_finite_f64(carrier, name="carrier", ndim=2)
        if value.shape[1] != CARRIER_WIDTH:
            raise CarrierNfError(f"carrier width {value.shape[1]} != {CARRIER_WIDTH}")
        return np.asarray(value / self.denominator, dtype=np.float64)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "h1_carrierid_nf_source_rms_normalizer_v1",
            "version": VERSION,
            "formula": "s_src = sqrt(mean(carrier**2)) over source channels/dims; C / max(s_src, floor)",
            "floor": NORMALIZER_FLOOR,
            "s_src": float(self.s_src),
            "denominator": float(self.denominator),
            "source_carrier_count": int(self.source_carrier_count),
            "normalizer_sha256": self.normalizer_sha256,
            "preserves_zero_carrier": True,
        }


def fit_normalizer(
    per_recording_carriers: Sequence[np.ndarray],
    *,
    labels: Any = None,
    velocity: Any = None,
    kinematics: Any = None,
) -> FrozenNfNormalizer:
    """Fit the source RMS scalar normalizer on source H-NF carriers."""

    reject_behavior_arrays(labels=labels, velocity=velocity, kinematics=kinematics)
    if len(per_recording_carriers) == 0:
        raise CarrierNfError("normalizer needs at least one source carrier array")
    stacked: list[np.ndarray] = []
    for index, carrier in enumerate(per_recording_carriers):
        array = _as_finite_f64(carrier, name=f"source_carriers[{index}]", ndim=2)
        if array.shape[1] != CARRIER_WIDTH:
            raise CarrierNfError(f"source_carriers[{index}] width {array.shape[1]} != {CARRIER_WIDTH}")
        stacked.append(array)
    pooled = np.concatenate(stacked, axis=0)
    scalar = float(np.sqrt(np.mean(np.square(pooled), dtype=np.float64)))
    if not np.isfinite(scalar) or scalar < 0.0:
        raise CarrierNfError("source H-NF RMS is undefined")
    body = {
        "version": VERSION,
        "s_src": scalar,
        "source_carrier_count": int(pooled.shape[0]),
    }
    return FrozenNfNormalizer(
        s_src=scalar,
        source_carrier_count=int(pooled.shape[0]),
        normalizer_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest(),
    )


# --------------------------------------------------------------------------- #
# Control arms.
# --------------------------------------------------------------------------- #
def zero_carrier(n_channels: int) -> np.ndarray:
    """H-C0: the exact zero vector at matched width 4 (stays zero after normalize)."""

    if int(n_channels) <= 0:
        raise CarrierNfError("H-C0 needs a positive channel count")
    return np.zeros((int(n_channels), CARRIER_WIDTH), dtype=np.float64)


def complete_row_shuffle_nf(
    carrier: np.ndarray, recording: str, date: str, *, seed: int = NF_ROW_SEED
) -> np.ndarray:
    """H-NF-RS: a deterministic non-identity complete-row permutation.

    Whole four-value rows are permuted (including declared-dead zero rows, per
    the dead-channel RS pool policy).  The four columns are never permuted
    independently and the carrier/normalizer are never refit.  The schedule is
    keyed by (recording, date, seed) and the seed namespace is independent of
    every KS4/TS4/H-RS schedule.
    """

    values = _as_finite_f64(carrier, name="carrier", ndim=2)
    if values.shape[1] != CARRIER_WIDTH:
        raise CarrierNfError(f"H-NF-RS carrier width {values.shape[1]} != {CARRIER_WIDTH}")
    if values.shape[0] < 2:
        raise CarrierNfError("a complete row shuffle needs at least two channels")
    token = hashlib.sha256(f"{seed}|hnf-row|{recording}|{date}".encode("utf-8")).digest()
    permutation = np.random.default_rng(int.from_bytes(token[:8], "big")).permutation(values.shape[0])
    if np.array_equal(permutation, np.arange(values.shape[0])):
        permutation = np.roll(permutation, 1)
    shuffled = values[permutation]
    if np.array_equal(shuffled, values):
        raise CarrierNfError("deterministic H-NF-RS row shuffle was not an intervention")
    return np.asarray(shuffled, dtype=np.float64)


def permutation_schedule_manifest(recording: str, date: str, *, seed: int = NF_ROW_SEED) -> dict[str, Any]:
    """Record the (recording, date, seed) H-NF-RS permutation schedule for receipts."""

    token = hashlib.sha256(f"{seed}|hnf-row|{recording}|{date}".encode("utf-8")).digest()
    schedule_seed = int.from_bytes(token[:8], "big")
    return {
        "arm": "H-NF-RS",
        "recording": str(recording),
        "date": str(date),
        "seed": int(seed),
        "schedule_seed": schedule_seed,
        "schedule_namespace": "independent of KS4/TS4/H-RS",
        "columns_permuted_independently": False,
        "carrier_refit": False,
        "normalizer_refit": False,
        DEAD_CHANNEL_RS_POOL_POLICY: True,
    }
