"""Strict-source posterior inputs and deferred Cell-D training adapter.

No source file is opened at import time.  The physical constructor below is
called only by a later root-reviewed backend.  Pure ``PosteriorSessionInput``
and bank builders are used by CPU tests to prove deterministic posterior and
normalizer semantics without touching an NWB, cache, CUDA device, checkpoint,
or result root.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from . import core, phase_b
from .plan import BUDGETS, CELL


class SourceAdapterError(phase_b.PhaseBError):
    """Fail-closed strict-source adapter error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAdapterError(message)


def _json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _verify_source_lineage_file(path: Path, row: Mapping[str, object]) -> None:
    """Route-local same-FD strict-source file proof.

    We intentionally do not import the historical TFSR ``source_smoke``
    module here.  Its package import initializes an unrelated model route and
    creates an undeclared staging dependency.  This is the small, exact
    provenance predicate this route actually needs.
    """
    expected_keys = {"feature_version", "path", "session", "sha256", "size_bytes", "unit_count"}
    if (not isinstance(row, Mapping) or set(row) != expected_keys
            or not isinstance(row.get("path"), str)):
        raise SourceAdapterError("malformed source-lineage row")
    phase_b._sha(row.get("sha256"), "source lineage file SHA")
    expected = Path(str(row["path"]))
    if path.absolute() != expected or path.is_symlink() or path.resolve(strict=True) != expected:
        raise SourceAdapterError("source path/alias drift against sealed lineage")
    before = os.lstat(path)
    if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
            or before.st_size != row.get("size_bytes")):
        raise SourceAdapterError("source size/type drift against sealed lineage")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise SourceAdapterError("source changed between lstat/open")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if (after.st_dev, after.st_ino, after.st_size) != (before.st_dev, before.st_ino, before.st_size):
        raise SourceAdapterError("source changed during same-FD SHA read")
    if digest.hexdigest() != row["sha256"]:
        raise SourceAdapterError("source SHA drift against sealed lineage")


def _load_stage_python_module(
    *,
    stage_root: Path,
    closure: Mapping[str, object],
    relative: str,
    name: str,
) -> types.ModuleType:
    """Execute one closure-bound helper from descriptor-read stage bytes.

    Loading the two known helper files this way avoids executing
    ``src.tfpd_lane.__init__`` and its unrelated diagnostics.  ``__file__``
    remains the staged pathname so the sealed population builder can resolve
    its explicitly closure-bound streaming component paths.
    """
    stable = phase_b.validate_phase_b_closure(closure)
    expected = stable["sha256_by_path"].get(relative)
    if not isinstance(expected, str):
        raise SourceAdapterError("runtime helper missing from Phase-B closure")
    body, _identity = phase_b.descriptor_read_stage_file(
        Path(stage_root).absolute(), relative, expected_sha256=expected,
    )
    module = types.ModuleType(name)
    module.__file__ = str(Path(stage_root).absolute() / relative)
    module.__package__ = ""
    exec(compile(body, module.__file__, "exec"), module.__dict__)
    return module


def load_stage_runtime_helpers(
    stage_root: Path,
    *,
    closure: Mapping[str, object],
) -> tuple[types.ModuleType, types.ModuleType]:
    """Load only the two explicit Cell-D runtime helpers from stage bytes."""
    root = Path(stage_root).absolute()
    arm_common = _load_stage_python_module(
        stage_root=root, closure=closure,
        relative="tfpd_exploration/src/tfpd_lane/arm_common.py",
        name="posterior_carrier_stage_arm_common",
    )
    pop_robust = _load_stage_python_module(
        stage_root=root, closure=closure,
        relative="tfpd_exploration/src/tfpd_lane/pop_robust.py",
        name="posterior_carrier_stage_pop_robust",
    )
    return arm_common, pop_robust


@dataclass(frozen=True)
class PosteriorSessionInput:
    """Source-only direct-count evidence for one strict-train session.

    Counts are raw integer spikes in the chronological first-30 labelled,
    datamodule-compatible rewarded trials.  Exposure and target directions are
    held at the same trial positions.  No floating firing-rate surrogate can
    enter this typed interface.
    """

    session_id: str
    raw_m30_t4: torch.Tensor
    counts_m30: torch.Tensor
    exposure_m30: torch.Tensor
    theta_m30: torch.Tensor
    prefix_row_ids: tuple[str, ...]
    source_path_sha256: str
    unit_order_sha256: str
    raw_t4_row_order_proof: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise SourceAdapterError("source posterior input session ID drift")
        if not isinstance(self.raw_m30_t4, torch.Tensor) or not self.raw_m30_t4.is_floating_point() or self.raw_m30_t4.ndim != 2:
            raise SourceAdapterError("raw M30 T4 must be floating [units,4]")
        if self.raw_m30_t4.shape[0] < 1 or self.raw_m30_t4.shape[1] != 4 or not bool(torch.isfinite(self.raw_m30_t4).all().item()):
            raise SourceAdapterError("raw M30 T4 values drift")
        try:
            direct = core._as_direct_integer_counts(self.counts_m30)
        except core.PosteriorCarrierError as error:
            raise SourceAdapterError("source posterior count semantics drift") from error
        units = self.raw_m30_t4.shape[0]
        if direct.shape != (units, 30):
            raise SourceAdapterError("source posterior counts must be [units,30]")
        if (not isinstance(self.exposure_m30, torch.Tensor) or not self.exposure_m30.is_floating_point()
                or self.exposure_m30.shape != (30,) or self.exposure_m30.device != direct.device
                or not bool(torch.isfinite(self.exposure_m30).all().item())
                or not bool((self.exposure_m30 > 0).all().item())):
            raise SourceAdapterError("source posterior exposure must be finite positive [30]")
        if (not isinstance(self.theta_m30, torch.Tensor) or not self.theta_m30.is_floating_point()
                or self.theta_m30.shape != (30,) or self.theta_m30.device != direct.device
                or not bool(torch.isfinite(self.theta_m30).all().item())):
            raise SourceAdapterError("source posterior theta must be finite [30]")
        if (not isinstance(self.prefix_row_ids, tuple) or len(self.prefix_row_ids) != 30
                or any(not isinstance(value, str) or not value for value in self.prefix_row_ids)
                or len(set(self.prefix_row_ids)) != 30):
            raise SourceAdapterError("source posterior prefix rows must be 30 unique ordered IDs")
        phase_b._sha(self.source_path_sha256, "source posterior source-file SHA")
        phase_b._sha(self.unit_order_sha256, "source posterior unit-order SHA")
        expected_proof = {
            "feature_group", "feature_version", "pool_size", "signal_view", "source_unit_count",
            "channel_ids_are_exact_arange", "raw_row_count", "unit_order_sha256",
            "row_semantics",
        }
        proof = self.raw_t4_row_order_proof
        if (not isinstance(proof, Mapping) or set(proof) != expected_proof
                or proof["feature_group"] != "t4" or proof["feature_version"] != 1 or proof["pool_size"] != 30
                or proof["signal_view"] != "sua" or proof["source_unit_count"] != units
                or proof["channel_ids_are_exact_arange"] is not True
                or proof["raw_row_count"] != units
                or proof["unit_order_sha256"] != self.unit_order_sha256
                or proof["row_semantics"] != "closure_bound_compute_unit_side_features_uncached_sua_rows_follow_nwb_units_order"):
            raise SourceAdapterError("raw M30 T4/unit-row order proof drift")

    def payload(self) -> dict[str, object]:
        prefix_evidence = {
            str(budget): {
                "counts_sha256": core.tensor_digest(self.counts_m30[:, :budget]),
                "exposure_sha256": core.tensor_digest(self.exposure_m30[:budget]),
                "theta_sha256": core.tensor_digest(self.theta_m30[:budget]),
            }
            for budget in BUDGETS
        }
        return {
            "session_id": self.session_id,
            "raw_m30_t4_sha256": core.tensor_digest(self.raw_m30_t4),
            "counts_m30_sha256": core.tensor_digest(self.counts_m30),
            "exposure_m30_sha256": core.tensor_digest(self.exposure_m30),
            "theta_m30_sha256": core.tensor_digest(self.theta_m30),
            # These prefix digests are the explicit bridge from each M4/M10/M30
            # posterior bank entry back to the physical direct-count evidence.
            # A bank digest alone cannot prove that it consumed the same prefix.
            "prefix_evidence_by_budget": prefix_evidence,
            "prefix_row_ids": list(self.prefix_row_ids),
            "prefix_rows_sha256": _json_sha(list(self.prefix_row_ids)),
            "source_path_sha256": self.source_path_sha256,
            "unit_order_sha256": self.unit_order_sha256,
            "raw_t4_row_order_proof": dict(self.raw_t4_row_order_proof),
            "unit_count": int(self.raw_m30_t4.shape[0]),
            "direct_integer_counts": True,
            "exposure_semantics": "trial_stop_time_minus_start_time_seconds",
            "theta_semantics": "chronological_labelled_rewarded_trial_target_direction_radians",
        }


def fit_source_session_posteriors(
    *,
    item: PosteriorSessionInput,
    prior: core.SourcePrior,
) -> dict[int, core.PosteriorCarrier]:
    """Fit exactly three deterministic source posteriors once for one session."""
    values: dict[int, core.PosteriorCarrier] = {}
    for budget in BUDGETS:
        values[budget] = core.fit_conjugate_posterior(
            counts=item.counts_m30[:, :budget],
            exposure=item.exposure_m30[:budget],
            theta=item.theta_m30[:budget],
            prior=prior,
        )
    return values


def build_source_posterior_bank(
    *,
    roster: Sequence[str],
    inputs: Mapping[str, PosteriorSessionInput],
    seed: int = 42,
) -> tuple[core.SourcePrior, phase_b.PosteriorEpochCarrierBank, dict[str, dict[int, core.PosteriorCarrier]]]:
    """Build prior, 27×3 deterministic fits, and the source-only normalizer.

    This is deliberately pre-loop work.  A caller can subsequently prewarm one
    sampled carrier per session/logical epoch, but no optimizer batch may call
    this function or ``fit_conjugate_posterior``.
    """
    strict_roster = phase_b._immutable_roster(roster)
    if not isinstance(inputs, Mapping) or set(inputs) != set(strict_roster):
        raise SourceAdapterError("strict source posterior inputs roster drift")
    ordered = [inputs[session] for session in strict_roster]
    if any(not isinstance(item, PosteriorSessionInput) or item.session_id != session for item, session in zip(ordered, strict_roster, strict=True)):
        raise SourceAdapterError("strict source posterior input order drift")
    raw_m30 = torch.cat([item.raw_m30_t4.detach().to(device="cpu", dtype=torch.float64) for item in ordered], dim=0)
    prior = core.SourcePrior.from_raw_m30_t4(raw_m30, strict_roster)
    posterior_by_session_budget = {
        session: fit_source_session_posteriors(item=inputs[session], prior=prior)
        for session in strict_roster
    }
    bank = phase_b.build_posterior_epoch_bank(
        roster=strict_roster,
        posterior_by_session_budget=posterior_by_session_budget,
        seed=seed,
    )
    return prior, bank, posterior_by_session_budget


@dataclass
class PhysicalPosteriorSourceAdapter:
    """Deferred source-only adapter; never constructed by the dry CLI."""

    dataset: Any
    sampler: Any
    roster: tuple[str, ...]
    bank: phase_b.PosteriorEpochCarrierBank
    prior: core.SourcePrior
    session_inputs: Mapping[str, PosteriorSessionInput]
    source_authority_metadata: Mapping[str, object]
    behavior_normalizer_semantic_sha256: str
    preparation_seconds: float
    train_files: tuple[Path, ...]

    def prewarm_epoch(self, epoch: int) -> None:
        self.bank.prewarm_epoch(epoch)

    def materialize_epoch_for_device(self, *, epoch: int, device: torch.device | str) -> None:
        self.bank.materialize_epoch_for_device(epoch=epoch, device=device)

    def view_for_optimizer_batch(
        self,
        *,
        session: str,
        epoch: int,
        device: torch.device | str,
    ) -> core.PosteriorCarrierView:
        return self.bank.optimizer_batch_view(session=session, epoch=epoch, device=device)

    def source_authority_payload(self, *, closure: Mapping[str, object]) -> dict[str, object]:
        metadata = dict(self.source_authority_metadata)
        lineage = metadata.get("source_lineage")
        source_data_root = metadata.get("source_data_root")
        if (not isinstance(lineage, Mapping) or not isinstance(lineage.get("body_sha256"), str)
                or not isinstance(source_data_root, Mapping)):
            raise SourceAdapterError("strict source lineage metadata schema drift")
        return {
            "schema": "posterior_carrier_source_authority_v1",
            "cell": CELL,
            "source_only": True,
            "target_opened": False,
            "within_opened": False,
            "external_opened": False,
            "formal_opened": False,
            "h1_opened": False,
            "roster": list(self.roster),
            "roster_sha256": _json_sha(list(self.roster)),
            # Repeat the target-free identity fields at the physical receipt
            # boundary.  The full metadata body remains present as the source
            # of the posterior-input file rows, while these literals make a
            # later launch/physical authority substitution fail before step 1.
            "strict_source_metadata_sha256": _json_sha(metadata),
            "manifest_sha256": metadata.get("manifest_sha256"),
            "ordinary_raw_t4_semantic_sha256": metadata.get("side_semantic_sha256"),
            "behavior_normalizer_semantic_sha256": self.behavior_normalizer_semantic_sha256,
            "source_lineage_sha256": lineage.get("body_sha256"),
            "source_data_root": dict(source_data_root),
            "source_authority_metadata": metadata,
            "source_raw_m30_t4_sha256": self.prior.raw_m30_t4_sha256,
            "posterior_prior": self.prior.payload(),
            "posterior_normalizer": self.bank.normalizer.payload(),
            "posterior_inputs": {session: self.session_inputs[session].payload() for session in self.roster},
            "posterior_bank": self.bank.source_posterior_digest_payload(),
            "posterior_preparation_seconds": self.preparation_seconds,
            "normalizer_phase_b_amendment": phase_b.PHASE_B_NORMALIZER_AMENDMENT,
            "m30_b3s_activity_prefix": "held; carrier label budgets only vary M4/M10/M30",
            "cache_read_or_write": False,
            "closure": phase_b.validate_phase_b_closure(closure),
        }


@dataclass(frozen=True)
class Strict27StageAuthority:
    """Descriptor-read strict-27 receipts bound to a separate source root."""

    stage_root: Path
    stage_identity: tuple[int, int]
    source_data: phase_b.SourceDataRootCapability
    payload_value: Mapping[str, object]
    asset_identities: Mapping[str, Mapping[str, int]]

    @property
    def manifest_path(self) -> Path:
        return self.stage_root / phase_b.STRICT27_MANIFEST_RELATIVE

    def payload(self) -> dict[str, object]:
        return dict(self.payload_value)

    def revalidate(self) -> None:
        info = os.lstat(self.stage_root)
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or (info.st_dev, info.st_ino) != self.stage_identity):
            raise SourceAdapterError("stage root identity drift")
        self.source_data.validate()
        for relative, expected_sha, expected_mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS:
            _body, actual = phase_b.descriptor_read_stage_file(
                self.stage_root, relative, expected_sha256=expected_sha, expected_mode=expected_mode,
            )
            if dict(actual) != dict(self.asset_identities[relative]):
                raise SourceAdapterError("stage authority file identity drift")


def _stage_sealed_asset(
    stage_root: Path,
    *,
    relative: str,
    expected_sha256: str,
    expected_mode: int | None,
) -> tuple[bytes, dict[str, int]]:
    body, identity = phase_b.descriptor_read_stage_file(
        stage_root, relative, expected_sha256=expected_sha256, expected_mode=expected_mode,
    )
    if expected_mode == 0o444:
        sidecar, _sidecar_identity = phase_b.descriptor_read_stage_file(
            stage_root,
            f"{relative}.sha256",
            expected_mode=0o444,
        )
        if sidecar != f"{expected_sha256}  {Path(relative).name}\n".encode("ascii"):
            raise SourceAdapterError("stage immutable authority sidecar drift")
    return body, identity


def _load_strict27_stage_authority(
    stage_root: Path,
    *,
    source_data: phase_b.SourceDataRootCapability,
) -> Strict27StageAuthority:
    """Read only staged authority bytes; never open an NWB or cache.

    Unlike the historical helper, lineage rows are checked against the explicit
    external ``SourceDataRootCapability`` rather than assuming source NWBs sit
    below the fresh code-stage root.  This is the crucial staging separation.
    """
    stage = Path(stage_root).absolute()
    stage_info = os.lstat(stage)
    if not stat.S_ISDIR(stage_info.st_mode) or stat.S_ISLNK(stage_info.st_mode):
        raise SourceAdapterError("stage root must be a canonical non-symlink directory")
    source_data.validate()
    assets: dict[str, bytes] = {}
    identities: dict[str, Mapping[str, int]] = {}
    for relative, expected_sha, expected_mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS:
        body, identity = _stage_sealed_asset(
            stage, relative=relative, expected_sha256=expected_sha, expected_mode=expected_mode,
        )
        assets[relative] = body
        identities[relative] = identity
    admission = json.loads(assets[phase_b.SOURCE_AUTHORITY_ASSET_SPECS[0][0]])
    theta_receipt = json.loads(assets[phase_b.SOURCE_AUTHORITY_ASSET_SPECS[1][0]])
    lineage = json.loads(assets[phase_b.SOURCE_AUTHORITY_ASSET_SPECS[3][0]])
    manifest = json.loads(assets[phase_b.STRICT27_MANIFEST_RELATIVE])
    contract = admission.get("data_contract", {})
    roster = phase_b._immutable_roster(contract.get("roster", ()), expected_count=phase_b.SOURCE_SESSION_COUNT)
    normalized_hashes = admission.get("t4_authority_sha256")
    if (admission.get("schema") != "tfpd_admission_arm_v1_preflight"
            or admission.get("status") != "ARM_PREFLIGHT_PASSED"
            or not isinstance(normalized_hashes, Mapping)
            or set(normalized_hashes) != set(roster)
            or any(not isinstance(value, str) or len(value) != 64 for value in normalized_hashes.values())
            or contract.get("manifest_sha256") != phase_b.STRICT27_MANIFEST_SHA
            or contract.get("n_train_windows") != 1_086_007
            or contract.get("steps_per_epoch") != phase_b.SOURCE_STEPS_PER_EPOCH):
        raise SourceAdapterError("staged strict-27 admission authority drift")
    normalizers = contract.get("normalizers", {})
    if (not isinstance(normalizers, Mapping)
            or normalizers.get("side_feature_semantic_sha256") != phase_b.STRICT27_SIDE_SEMANTIC_SHA
            or normalizers.get("behavior_semantic_sha256") != phase_b.STRICT27_BEHAVIOR_SEMANTIC_SHA):
        raise SourceAdapterError("staged strict-27 normalizer authority drift")
    proof = theta_receipt.get("alignment_proof")
    if (theta_receipt.get("schema") != "tfpd_sparsification_theta_authority_v1"
            or theta_receipt.get("status") != "THETA_AUTHORITY_SEALED"
            or theta_receipt.get("artifact", {}).get("sha256") != phase_b.STRICT27_THETA_ARTIFACT_SHA
            or not isinstance(proof, Mapping) or set(proof) != set(roster)):
        raise SourceAdapterError("staged theta authority drift")
    for session in roster:
        entry = proof[session]
        if (not isinstance(entry, Mapping) or set(entry) != {
                "aligned", "datamodule_channels", "n_undefined", "n_valid_directions", "side_rows", "theta_units"
        } or entry.get("aligned") is not True
                or any(type(entry[key]) is not int for key in entry if key != "aligned")
                or entry["theta_units"] != entry["side_rows"]
                or entry["theta_units"] != entry["datamodule_channels"]):
            raise SourceAdapterError("staged theta per-session alignment drift")
    descriptor, rows = lineage.get("descriptor"), lineage.get("source_files")
    expected_row_keys = {"feature_version", "path", "session", "sha256", "size_bytes", "unit_count"}
    if (lineage.get("receipt_kind") != "misleading_identity_swap_v2_strict27_m30_source_lineage"
            or lineage.get("schema_version") != 1 or lineage.get("source_only") is not True
            or lineage.get("target_nwb_opened") is not False or lineage.get("validation_nwb_opened") is not False
            or lineage.get("formal_subc_test_nwb_opened") is not False
            or lineage.get("strict_manifest_sha256") != phase_b.STRICT27_MANIFEST_SHA
            or lineage.get("source_train_sessions") != list(roster)
            or lineage.get("source_train_session_count") != phase_b.SOURCE_SESSION_COUNT
            or lineage.get("matching_authority_consumed_bytes_sha256") != phase_b.STRICT27_MATCHING_AUTHORITY_CONSUMED_SHA
            or not isinstance(descriptor, Mapping) or not isinstance(rows, list) or len(rows) != len(roster)):
        raise SourceAdapterError("staged strict-27 lineage receipt drift")
    expected_descriptor = {
        "feature_group": "t4", "pool_size": 30, "bin_size_ms": 20, "window_size_bins": 50,
        "support": "chronological_first_30_rewarded_trials", "signal_view": "sua",
        "columns": ["m_cos_phi", "m_sin_phi", "m", "b"],
        "normalizer_mean_float32": phase_b.STRICT27_SOURCE_T4_MEAN,
        "normalizer_std_float32": phase_b.STRICT27_SOURCE_T4_STD,
        "normalizer_value_sha256": phase_b.STRICT27_SIDE_SEMANTIC_SHA,
    }
    if any(descriptor.get(key) != value for key, value in expected_descriptor.items()):
        raise SourceAdapterError("staged strict-27 lineage descriptor drift")
    rows_by_session: dict[str, dict[str, object]] = {}
    for position, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != expected_row_keys
                or row.get("session") != roster[position] or row.get("feature_version") != 1
                or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64
                or type(row.get("size_bytes")) is not int or row["size_bytes"] <= 0
                or type(row.get("unit_count")) is not int or row["unit_count"] <= 0):
            raise SourceAdapterError("staged strict-27 lineage row drift")
        # This checks only the authoritative pathname; no NWB is opened here.
        source_data.session_path(session=roster[position], lineage_row=row)
        rows_by_session[roster[position]] = dict(row)
    splits = manifest.get("session_splits") if isinstance(manifest, Mapping) else None
    if (not isinstance(splits, Mapping) or manifest.get("schema_version") != 1 or manifest.get("task") != "CO"
            or manifest.get("max_units_exclusive") != 100 or manifest.get("split_counts") != [27, 6, 6]
            or splits.get("train") != list(roster) or not isinstance(splits.get("val"), list)
            or not isinstance(splits.get("test"), list) or len(splits["val"]) != 6 or len(splits["test"]) != 6):
        raise SourceAdapterError("staged strict train/val manifest drift")
    payload = {
        "admission_preflight": {"path": phase_b.SOURCE_AUTHORITY_ASSET_SPECS[0][0], "body_sha256": phase_b.STRICT27_ADMISSION_SHA},
        "theta_receipt": {"path": phase_b.SOURCE_AUTHORITY_ASSET_SPECS[1][0], "body_sha256": phase_b.STRICT27_THETA_RECEIPT_SHA},
        "theta_artifact": {"path": phase_b.SOURCE_AUTHORITY_ASSET_SPECS[2][0], "body_sha256": phase_b.STRICT27_THETA_ARTIFACT_SHA},
        "roster": list(roster), "normalized_t4_sha256": dict(normalized_hashes),
        "raw_authority_sha256": phase_b.STRICT27_THETA_ARTIFACT_SHA, "normalizer_authority_sha256": phase_b.STRICT27_ADMISSION_SHA,
        "side_semantic_sha256": phase_b.STRICT27_SIDE_SEMANTIC_SHA,
        "behavior_semantic_sha256": phase_b.STRICT27_BEHAVIOR_SEMANTIC_SHA,
        "manifest_sha256": phase_b.STRICT27_MANIFEST_SHA,
        "stage_manifest_relative": phase_b.STRICT27_MANIFEST_RELATIVE,
        "source_data_root": source_data.payload(),
        "stage_authority_assets": {
            relative: {"sha256": expected_sha, "descriptor_identity": dict(identities[relative])}
            for relative, expected_sha, _mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS
        },
        "source_lineage": {
            "path": phase_b.SOURCE_AUTHORITY_ASSET_SPECS[3][0],
            "body_sha256": phase_b.STRICT27_SOURCE_LINEAGE_SHA,
            "matching_authority_consumed_bytes_sha256": phase_b.STRICT27_MATCHING_AUTHORITY_CONSUMED_SHA,
            "rows_by_session": rows_by_session,
        },
    }
    return Strict27StageAuthority(
        stage_root=stage,
        stage_identity=(int(stage_info.st_dev), int(stage_info.st_ino)),
        source_data=source_data,
        payload_value=payload,
        asset_identities=identities,
    )


def verify_strict27_source_metadata(
    root: Path,
    *,
    source_data: phase_b.SourceDataRootCapability | None = None,
) -> Mapping[str, object]:
    """Metadata-only staged-authority audit; it never opens a source NWB."""
    capability = source_data
    if capability is None:
        capability = phase_b._issue_source_data_root_capability(
            source_data_root=Path(phase_b.CANONICAL_SOURCE_DATA_ROOT),
        )
    authority = _load_strict27_stage_authority(Path(root).absolute(), source_data=capability)
    return authority.payload()


def build_target_free_source_identity(
    root: Path,
    *,
    source_data: phase_b.SourceDataRootCapability | None = None,
) -> phase_b.RunIdentity:
    """Build the reviewed source-only identity without opening any NWB or CUDA.

    It binds immutable strict-27 metadata and the exact Phase-B code closure.
    The posterior normalizer is intentionally absent here: it can only be
    derived later from source rows after the immutable attempt/launch exists.
    """
    capability = source_data
    if capability is None:
        capability = phase_b._issue_source_data_root_capability(
            source_data_root=Path(phase_b.CANONICAL_SOURCE_DATA_ROOT),
        )
    authority = verify_strict27_source_metadata(root, source_data=capability)
    roster = phase_b._immutable_roster(authority["roster"], expected_count=phase_b.SOURCE_SESSION_COUNT)
    source_identity = {
        "schema": "posterior_carrier_target_free_source_identity_v1",
        "roster": list(roster),
        "roster_sha256": _json_sha(list(roster)),
        "strict_source_metadata_sha256": _json_sha(authority),
        "manifest_sha256": authority["manifest_sha256"],
        "ordinary_raw_t4_semantic_sha256": authority["side_semantic_sha256"],
        "behavior_normalizer_semantic_sha256": authority["behavior_semantic_sha256"],
        "source_lineage_sha256": authority["source_lineage"]["body_sha256"],
        "source_data_root": capability.payload(),
        "source_only": True,
        "target_opened": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "h1_opened": False,
    }
    return phase_b.RunIdentity(
        source_authority=source_identity,
        closure=phase_b.phase_b_closure(Path(root).absolute()),
        remote_device=dict(phase_b.REMOTE_TORCH_AUTHORITY),
    )


def _prepend_sua_package(root: Path) -> None:
    source = str(Path(root).absolute() / "sua_exploration")
    if source not in sys.path:
        sys.path.insert(0, source)


def _direct_train_path(
    *,
    session: str,
    authority: Mapping[str, object],
    source_data: phase_b.SourceDataRootCapability,
) -> Path:
    rows = authority.get("source_lineage", {}).get("rows_by_session", {})
    if not isinstance(rows, Mapping) or session not in rows or not isinstance(rows[session], Mapping):
        raise SourceAdapterError("source lineage row missing")
    row = rows[session]
    return source_data.session_path(session=session, lineage_row=row)


def _construct_train_only_datamodule(
    *,
    root: Path,
    authority: Mapping[str, object],
    source_data: phase_b.SourceDataRootCapability,
    num_workers: int,
    on_source_opened: Callable[[], None] | None = None,
) -> tuple[Any, Any, tuple[str, ...], tuple[Path, ...]]:
    """Build only the strict train split, avoiding the shared val/test resolver."""
    _prepend_sua_package(root)
    import mc_maze.a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

    roster = phase_b._immutable_roster(authority["roster"], expected_count=phase_b.SOURCE_SESSION_COUNT)
    data_root = source_data.validate()
    train_files = tuple(_direct_train_path(session=session, authority=authority, source_data=source_data) for session in roster)
    if len(set(train_files)) != len(train_files):
        raise SourceAdapterError("strict train path topology drift")
    dm = Dandi688MultiSessionDataModule(
        data_dir=str(data_root), task="CO", split_counts=(27, 6, 6), batch_size=phase_b.SOURCE_BATCH_SIZE,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=num_workers, pin_memory=True, random_calibration=False, seed=42,
        max_units_exclusive=100, cache_dir=None, signal_view="sua", side_feature_group="t4",
        side_feature_pool_size=30,
        # The closure-bound manifest is read from the fresh stage root; source
        # NWBs are exclusively direct children of the external source root.
        train_val_manifest_path=str(Path(root).absolute() / str(authority["stage_manifest_relative"])),
    )
    # Do not invoke `_initialize_splits`: the shared helper constructs
    # validation metadata before callers can clear it.  This route injects the
    # canonical train list only and uses an identity sentinel to suppress
    # setup('fit')'s empty-val dataset construction.
    dm.session_files = {"train": list(train_files), "val": [], "test": []}
    dm.session_splits = {"train": list(roster), "val": [], "test": []}
    dm.session_unit_counts = {}
    dm._splits_initialized = True
    sentinel = object()
    dm.val_dataset = sentinel
    try:
        # This exact shared setup call creates source records/NWB metadata.  A
        # caller-provided progress hook records that source access has begun so
        # a failure thrown inside setup cannot be misreported as zero access.
        if on_source_opened is not None:
            on_source_opened()
        dm.setup("fit")
        if dm.val_dataset is not sentinel:
            raise SourceAdapterError("strict train-only setup constructed a validation dataset")
    finally:
        if dm.val_dataset is sentinel:
            dm.val_dataset = None
    if (dm.train_dataset is None or dm.val_dataset is not None or dm.test_dataset is not None
            or dm.session_files["val"] or dm.session_files["test"] or dm.cache_dir is not None):
        raise SourceAdapterError("strict train-only datamodule boundary drift")
    return dm, a2, roster, train_files


def _source_counts_exposure_theta(
    *,
    path: Path,
    session: str,
    expected_units: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Read direct integer source counts only after a reviewed backend opens source."""
    import numpy as np
    from pynwb import NWBHDF5IO
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials

    trials = list_datamodule_rewarded_trials(path, bin_size_ms=20, window_size=50, trial_result_filter="R")
    if len(trials) < 30:
        raise SourceAdapterError("source session lacks 30 chronological rewarded trials")
    prefix = trials[:30]
    if any(item.get("target_dir") is None for item in prefix):
        raise SourceAdapterError("posterior source prefix has an unlabeled target direction")
    starts = np.asarray([float(item["start_time"]) for item in prefix], dtype=np.float64)
    stops = np.asarray([float(item["stop_time"]) for item in prefix], dtype=np.float64)
    if np.any(stops <= starts):
        raise SourceAdapterError("posterior source prefix exposure is nonpositive")
    theta = np.asarray([float(item["target_dir"]) for item in prefix], dtype=np.float64)
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise SourceAdapterError("source NWB has no units table")
        units = nwb.units.to_dataframe()
        if len(units) != expected_units:
            raise SourceAdapterError("source NWB unit count drift")
        counts = np.empty((expected_units, 30), dtype=np.int64)
        for index in range(expected_units):
            spikes = np.asarray(units.iloc[index]["spike_times"], dtype=np.float64)
            if spikes.size and not np.all(spikes[:-1] <= spikes[1:]):
                raise SourceAdapterError("source unit spike ordering drift")
            counts[index] = np.searchsorted(spikes, stops, side="left") - np.searchsorted(spikes, starts, side="left")
    row_ids = tuple(f"{session}:trial:{int(item['trial_index'])}" for item in prefix)
    return (
        torch.from_numpy(counts),
        torch.from_numpy(stops - starts),
        torch.from_numpy(theta),
        row_ids,
    )


def _raw_m30_t4_and_unit_order(
    *,
    path: Path,
    session: str,
    expected_units: int,
    record_source_unit_count: int,
    record_channel_ids: object,
) -> tuple[torch.Tensor, str, dict[str, object]]:
    """Compute ordinary raw M30 T4 only as the source prior substrate."""
    import numpy as np
    from mc_maze.unit_side_features import compute_unit_side_features_uncached

    raw, metadata = compute_unit_side_features_uncached(
        path, feature_group="t4", pool_size=30, bin_size_ms=20, window_size=50,
        trial_result_filter="R", signal_view="sua",
    )
    expected_channel_ids = np.arange(expected_units, dtype=np.int64)
    observed_channel_ids = np.asarray(record_channel_ids, dtype=np.int64)
    if (getattr(metadata, "feature_group", None) != "t4" or getattr(metadata, "feature_version", None) != 1
            or getattr(metadata, "pool_size", None) != 30
            or raw.shape != (expected_units, 4) or record_source_unit_count != expected_units
            or observed_channel_ids.shape != (expected_units,)
            or not np.array_equal(observed_channel_ids, expected_channel_ids)):
        raise SourceAdapterError("ordinary raw M30 T4 source-prior substrate drift")
    raw64 = torch.from_numpy(np.asarray(raw, dtype=np.float64))
    unit_order = torch.arange(expected_units, dtype=torch.int64)
    unit_order_sha = core.tensor_digest(unit_order)
    # ``compute_unit_side_features_uncached`` is closure-bound in Phase B and,
    # for ``signal_view='sua'``, emits one feature row per NWB units-table row.
    # The datamodule independently emits that same sorted-unit axis as exact
    # ``arange(source_unit_count)``.  Record both sides rather than merely
    # assuming that equal matrix shapes imply an equal unit ordering.
    proof = {
        "feature_group": "t4",
        "feature_version": 1,
        "pool_size": 30,
        "signal_view": "sua",
        "source_unit_count": expected_units,
        "channel_ids_are_exact_arange": True,
        "raw_row_count": int(raw64.shape[0]),
        "unit_order_sha256": unit_order_sha,
        "row_semantics": "closure_bound_compute_unit_side_features_uncached_sua_rows_follow_nwb_units_order",
    }
    return raw64, unit_order_sha, proof


def build_physical_source_adapter(
    root: Path,
    *,
    source_data: phase_b.SourceDataRootCapability,
    num_workers: int = 4,
    on_source_opened: Callable[[], None] | None = None,
) -> PhysicalPosteriorSourceAdapter:
    """Deferred physical source-only construction; never call from dry paths/tests."""
    started = time.monotonic()
    _prepend_sua_package(root)
    from mc_maze.multisession_datamodule import SessionBatchSampler

    stage_authority = _load_strict27_stage_authority(Path(root).absolute(), source_data=source_data)
    authority = stage_authority.payload()
    dm, a2, roster, train_files = _construct_train_only_datamodule(
        root=Path(root).absolute(), authority=authority, source_data=source_data, num_workers=num_workers,
        on_source_opened=on_source_opened,
    )
    # Verify direct source files before *and* after source-only posterior
    # extraction.  This is a narrow source surface; no val/test path is built.
    source_inputs: dict[str, PosteriorSessionInput] = {}
    for session, path in zip(roster, train_files, strict=True):
        row = authority["source_lineage"]["rows_by_session"][session]
        _verify_source_lineage_file(path, row)
        record = dm.train_dataset.sessions[session]
        expected_units = int(row["unit_count"])
        if (record.source_unit_count != expected_units or record.neural.shape[1] != expected_units
                or record.channel_ids is None or not torch.equal(torch.as_tensor(record.channel_ids), torch.arange(expected_units))):
            raise SourceAdapterError("source dataset/unit order drift")
        counts, exposure, theta, row_ids = _source_counts_exposure_theta(
            path=path, session=session, expected_units=expected_units,
        )
        raw_t4, unit_order_sha, row_order_proof = _raw_m30_t4_and_unit_order(
            path=path, session=session, expected_units=expected_units,
            record_source_unit_count=int(record.source_unit_count), record_channel_ids=record.channel_ids,
        )
        _verify_source_lineage_file(path, row)
        source_inputs[session] = PosteriorSessionInput(
            session_id=session, raw_m30_t4=raw_t4, counts_m30=counts,
            exposure_m30=exposure, theta_m30=theta, prefix_row_ids=row_ids,
            source_path_sha256=str(row["sha256"]), unit_order_sha256=unit_order_sha,
            raw_t4_row_order_proof=row_order_proof,
        )
    prior, bank, _posteriors = build_source_posterior_bank(roster=roster, inputs=source_inputs, seed=42)
    # This is the ordinary *behaviour* normalizer, which remains held.  The
    # ordinary side/T4 normalizer is explicitly not used as a model input.
    behavior_sha = a2.normalizer_value_sha256(*dm._behavior_stats)
    if behavior_sha != authority["behavior_semantic_sha256"]:
        raise SourceAdapterError("strict source behaviour normalizer drift")
    sampler = SessionBatchSampler(dm.train_dataset, batch_size=phase_b.SOURCE_BATCH_SIZE, shuffle=True, seed=42)
    if len(sampler) != phase_b.SOURCE_STEPS_PER_EPOCH:
        raise SourceAdapterError("strict source B32 batch topology drift")
    stage_authority.revalidate()
    return PhysicalPosteriorSourceAdapter(
        dataset=dm.train_dataset, sampler=sampler, roster=roster, bank=bank, prior=prior,
        session_inputs=source_inputs, source_authority_metadata=authority,
        behavior_normalizer_semantic_sha256=behavior_sha,
        preparation_seconds=time.monotonic() - started, train_files=train_files,
    )


@dataclass
class _PhysicalSmokeRuntime:
    """Opaque execution state returned only after source-only preparation."""

    torch: Any
    adapter: PhysicalPosteriorSourceAdapter
    wrapper: core.CellDPosteriorWrapper
    optimizer: Any
    arm_common: Any
    pop_robust: Any
    iterator: Any
    device: Any
    remote_device: Mapping[str, object]
    posterior_prepare_seconds: float
    progress: phase_b.SmokeExecutionProgress
    source_opened: bool = True
    remote_initialized: bool = True


def _posterior_statistics(bank: phase_b.PosteriorEpochCarrierBank) -> dict[str, object]:
    values: list[torch.Tensor] = []
    zero_count = 0
    standardized_zero_rows: list[torch.Tensor] = []
    for session in bank.roster:
        for budget in BUDGETS:
            view = bank.deterministic_mean_view(session=session, budget=budget)
            values.append(view.credibility.detach().cpu())
            zero_count += int(view.zero_spike_mask.sum().item())
            if view.zero_spike_mask.any().item():
                standardized_zero_rows.append(view.normalized_t4[view.zero_spike_mask].detach().cpu())
    credibility = torch.cat(values)
    zero_standardized = torch.cat(standardized_zero_rows) if standardized_zero_rows else torch.empty((0, 4), dtype=credibility.dtype)
    return {
        "credibility_min": float(credibility.min().item()),
        "credibility_max": float(credibility.max().item()),
        "credibility_mean": float(credibility.mean().item()),
        "zero_spike_unit_rows_across_m4_m10_m30": zero_count,
        "zero_raw_t4_standardized_not_clamped": True,
        "zero_standardized_rows_sha256": core.tensor_digest(zero_standardized),
        "posterior_normalizer_sha256": bank.normalizer.authority_sha256,
    }


class RemotePosteriorSourceSmokeBackend:
    """The deferred physical 5070-Ti source-only smoke backend.

    The class is intentionally not selected by the public CLI.  Its constructor
    does not open data or initialize CUDA; those actions occur only inside
    ``prepare`` after the outer lifecycle has already published an immutable
    attempt and launch under a root-reviewed capability.
    """

    def __init__(self, root: Path, *, source_data: phase_b.SourceDataRootCapability, num_workers: int = 4,
                 nvml_status: str = "UNAVAILABLE_DRIVER_LIBRARY_MISMATCH") -> None:
        self.root = Path(root).absolute()
        self.source_data = source_data
        self.num_workers = num_workers
        self.nvml_status = nvml_status

    def prepare(self, spec: phase_b.SourceSmokeSpec, identity: phase_b.RunIdentity) -> _PhysicalSmokeRuntime:
        if spec != phase_b.SMOKE_SPEC:
            raise SourceAdapterError("physical posterior backend accepts only the fixed source smoke")
        phase_b.validate_run_identity(identity)
        if self.source_data.payload() != identity.source_authority.get("source_data_root"):
            raise SourceAdapterError("launch/external source-data root capability drift")
        self.source_data.validate()
        if self.num_workers != 4:
            raise SourceAdapterError("posterior source smoke holds predecessor worker count at four")
        # This physical backend can also be called by a direct in-process
        # reviewer.  Recompute the descriptor-safe stage closure before any
        # source or CUDA operation, rather than trusting an earlier dict.
        live_closure = phase_b.phase_b_closure(self.root)
        if live_closure != phase_b.validate_phase_b_closure(identity.closure):
            raise SourceAdapterError("physical source smoke live stage closure drift")
        progress = phase_b.SmokeExecutionProgress()

        def mark_source_opened() -> None:
            nonlocal progress
            progress = progress.merge(phase_b.SmokeExecutionProgress(source_opened=True))

        try:
            # Source preparation constructs the posterior bank and its immutable
            # source-only normalizer before any model batch loop.  It has no
            # val/test/formal path and no cache read/write.
            adapter = build_physical_source_adapter(
                self.root, source_data=self.source_data, num_workers=self.num_workers,
                on_source_opened=mark_source_opened,
            )
            if tuple(adapter.roster) != tuple(identity.source_authority.get("roster", ())):
                raise SourceAdapterError("launch/source strict-27 roster drift")
            if identity.source_authority.get("strict_source_metadata_sha256") != _json_sha(adapter.source_authority_metadata):
                raise SourceAdapterError("launch/source strict metadata authority drift")

            import numpy as np
            import random
            from torch.utils.data import DataLoader
            arm_common, pop_robust = load_stage_runtime_helpers(self.root, closure=live_closure)

            # The only execution-time hardware predicate is the reviewed remote
            # Torch authority.  NVML is recorded as the declared mismatch, never
            # queried for fabricated UUID/BDF/memory evidence.
            if not torch.cuda.is_available():
                raise SourceAdapterError("posterior source smoke requires one visible CUDA device")
            progress = progress.merge(phase_b.SmokeExecutionProgress(
                source_opened=progress.source_opened, remote_initialized=True,
            ))
            attestation = phase_b.attest_remote_torch_only(torch, nvml_status=self.nvml_status)
            if torch.is_autocast_enabled() or bool(torch.backends.cuda.matmul.allow_tf32) or bool(torch.backends.cudnn.allow_tf32):
                raise SourceAdapterError("posterior source smoke forbids AMP/TF32")
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            model = pop_robust.build_population_robustness_model(seed=42, cell="D")
            wrapper = core.CellDPosteriorWrapper(model)
            preservation = wrapper.preservation_audit()
            if (preservation.base_live_parameter_count != 3_510_842 or preservation.wrapper_new_parameter_count != 0
                    or preservation.dynamic_dropout is not True):
                raise SourceAdapterError("posterior Cell-D graph/parameter/dropout preservation drift")
            device = torch.device("cuda:0")
            wrapper.to(device)
            optimizer = torch.optim.Adam(
                wrapper.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0.0, amsgrad=False,
            )
            # Build all epoch-0 samples before the DataLoader iterator and before
            # the timed optimizer core.  Every later batch only retrieves its
            # already cached session/epoch view.
            cache_started = time.monotonic()
            adapter.prewarm_epoch(0)
            adapter.materialize_epoch_for_device(epoch=0, device=device)
            # `adapter.preparation_seconds` covers direct source extraction,
            # prior/normalizer fitting, and all 27×3 inverses.  Include the
            # session-static sampling/device-cache boundary too, but keep this
            # complete preparation interval outside the timed optimizer core.
            posterior_prepare_seconds = adapter.preparation_seconds + (time.monotonic() - cache_started)
            loader = DataLoader(adapter.dataset, batch_sampler=adapter.sampler, num_workers=self.num_workers, pin_memory=True)
            iterator = iter(loader)
            torch.cuda.reset_peak_memory_stats(0)
            return _PhysicalSmokeRuntime(
                torch=torch, adapter=adapter, wrapper=wrapper, optimizer=optimizer,
                arm_common=arm_common, pop_robust=pop_robust, iterator=iterator, device=device,
                remote_device=dict(attestation.payload), posterior_prepare_seconds=posterior_prepare_seconds,
                progress=progress,
            )
        except BaseException as error:
            if isinstance(error, phase_b.SourceSmokeExecutionError):
                raise
            raise phase_b.SourceSmokeExecutionError(stage="prepare", progress=progress, cause=error) from error

    def source_authority(self, runtime: _PhysicalSmokeRuntime,
                         identity: phase_b.RunIdentity) -> Mapping[str, object]:
        payload = runtime.adapter.source_authority_payload(closure=identity.closure)
        payload.update({
            "posterior_preparation_seconds": runtime.posterior_prepare_seconds,
            "remote_torch_authority": dict(runtime.remote_device),
            "posterior_credibility_statistics": _posterior_statistics(runtime.adapter.bank),
            "optimizer": {
                "class": "Adam", "lr_constructor": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                "weight_decay": 0.0, "amsgrad": False,
                "schedule": "arm_common.lr_at_step(48,33925)",
            },
            "execution_policy": {"amp": False, "tf32": False, "torch_compile": False, "batch_size": 32},
        })
        return payload

    def run_steps(self, runtime: _PhysicalSmokeRuntime,
                  spec: phase_b.SourceSmokeSpec) -> phase_b.SmokeStepSummary:
        torch_module = runtime.torch
        losses: list[float] = []
        core_seconds = 0.0
        dropout_law = {
            "dynamic_dropout": True,
            "low": 0.0,
            "high": 1.0,
            "semantics": "complete_fused_unit_token_placeholder_with_inverse_probability_gain",
            "extra_dropout_draws": 0,
        }
        try:
            for global_step in range(spec.optimizer_steps):
                try:
                    batch = next(runtime.iterator)
                except StopIteration as error:
                    raise SourceAdapterError("source sampler exhausted before 100-step smoke completed") from error
                neural, behavior, calib, sessions = batch[:4]
                names = tuple(sessions)
                if len(names) != spec.batch_size or len(set(names)) != 1 or names[0] not in runtime.adapter.roster:
                    raise SourceAdapterError("source smoke batch is not session-homogeneous strict source data")
                view = runtime.adapter.view_for_optimizer_batch(session=names[0], epoch=0, device=runtime.device)
                neural = neural.to(runtime.device, non_blocking=True)
                behavior = behavior.to(runtime.device, non_blocking=True)
                calib = calib.to(runtime.device, non_blocking=True)
                valid = (behavior != -1.0).all(dim=-1)
                if not bool(valid.any().item()):
                    raise SourceAdapterError("source smoke batch has no valid dense-loss bins")
                lr = runtime.arm_common.lr_at_step(global_step, phase_b.SOURCE_EPOCHS, phase_b.SOURCE_STEPS_PER_EPOCH)
                runtime.optimizer.param_groups[0]["lr"] = lr
                runtime.wrapper.train(True)
                runtime.optimizer.zero_grad(set_to_none=True)
                # The timed core intentionally contains only model/loss/backward/
                # optimizer work.  Receipt scalar conversion, full-state hashing,
                # and posterior construction are outside it.
                torch_module.cuda.synchronize(0)
                started = time.perf_counter()
                prediction, _identity = runtime.wrapper(neural, calib_trials_m30=calib, carrier=view)
                loss = (((prediction - behavior).square().sum(dim=-1) * valid).sum()
                        / (valid.sum() * behavior.shape[-1]))
                loss.backward()
                runtime.optimizer.step()
                torch_module.cuda.synchronize(0)
                core_seconds += time.perf_counter() - started
                # An optimizer step has now occurred even if a subsequent
                # finite/scalar audit fails.  Keep this exact monotone count
                # in the runtime so a step-1..99 failure cannot be reported
                # as zero progress.
                runtime.progress = runtime.progress.merge(phase_b.SmokeExecutionProgress(
                    source_opened=runtime.progress.source_opened,
                    remote_initialized=runtime.progress.remote_initialized,
                    optimizer_steps_completed=global_step + 1,
                ))
                if not bool(torch_module.isfinite(loss).item()):
                    raise SourceAdapterError("posterior source smoke produced a nonfinite loss")
                losses.append(float(loss.detach().cpu().item()))
            gradients = phase_b.critical_gradient_proof(runtime.wrapper)
            finite_model, finite_adam = phase_b.finite_model_and_adam(runtime.wrapper, runtime.optimizer)
            if not finite_model or not finite_adam:
                raise SourceAdapterError("posterior source smoke final state is nonfinite")
            torch_module.cuda.synchronize(0)
            current_allocated = int(torch_module.cuda.memory_allocated(0))
            current_reserved = int(torch_module.cuda.memory_reserved(0))
            peak_allocated = int(torch_module.cuda.max_memory_allocated(0))
            peak_reserved = int(torch_module.cuda.max_memory_reserved(0))
            observer = runtime.adapter.bank.observer(expected_full_schedule=False).payload()
            if (observer["batch_loop_inverse_calls"] != 0
                    or observer["epoch_sampled_view_builds"] != len(runtime.adapter.roster)
                    or observer["device_epoch_view_builds"] != len(runtime.adapter.roster)):
                raise SourceAdapterError("posterior batch loop cache/inverse proof drift")
            return phase_b.SmokeStepSummary(
                loss_first=losses[0], loss_last=losses[-1], loss_min=min(losses), loss_max=max(losses),
                nonincreasing_transitions=sum(next_loss <= prior for prior, next_loss in zip(losses, losses[1:])),
                losses_count=len(losses), critical_gradients=gradients, finite_model=finite_model,
                finite_adam=finite_adam, model_state_sha256=phase_b.lazy_safe_state_sha256(runtime.wrapper),
                optimizer_state_sha256=runtime.arm_common.optimizer_sha256(runtime.optimizer),
                posterior_prepare_seconds=runtime.posterior_prepare_seconds, optimizer_core_seconds=core_seconds,
                optimizer_steps_per_second=float(spec.optimizer_steps / core_seconds), peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved, current_allocated_bytes=current_allocated,
                current_reserved_bytes=current_reserved, posterior_cache=observer, dropout=dropout_law,
                remote_device=runtime.remote_device,
            )
        except BaseException as error:
            if isinstance(error, phase_b.SourceSmokeExecutionError):
                raise
            raise phase_b.SourceSmokeExecutionError(
                stage="steps", progress=runtime.progress, cause=error,
            ) from error

    def close(self, runtime: _PhysicalSmokeRuntime | None) -> None:
        # CUDA teardown is deliberately left to the process boundary; this
        # method neither touches another device nor creates a result/cache.
        return None
