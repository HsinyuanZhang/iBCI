"""Explicit future staging plan for the non-governing quick screen.

Importing this module is inert: it uses only the standard library and neither
opens a NWB/checkpoint nor contacts a remote host.  A root-reviewed caller may
later pass an opaque capability and an injected transport.  The public CLI
cannot construct either.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from . import quick_screen as contract


class StageError(RuntimeError):
    """Fail closed before a stage byte is selected or transferred."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise StageError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    try:
        return contract._sha(value, label)
    except contract.QuickScreenError as error:
        raise StageError(str(error)) from error


def _safe_relative(value: object, label: str) -> str:
    try:
        return contract._safe_relative(value, label)
    except contract.QuickScreenError as error:
        raise StageError(str(error)) from error


def _regular_identity(path: Path) -> tuple[tuple[int, int, int], int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise StageError(f"cannot lstat staging source: {path}") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise StageError("staging source must be one regular non-symlink file")
    return (int(info.st_dev), int(info.st_ino), int(info.st_size)), stat.S_IMODE(info.st_mode)


def _read_regular_exact(path: Path, *, byte_count: int, sha256: str, mode: int) -> bytes:
    """Descriptor-read and hash one local leaf immediately before transfer."""
    expected = _sha(sha256, "staging source SHA")
    before, actual_mode = _regular_identity(path)
    if actual_mode != mode or before[2] != byte_count:
        raise StageError("staging source mode/size differs from reviewed plan")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (int(opened.st_dev), int(opened.st_ino), int(opened.st_size)) != before
        ):
            raise StageError("staging source descriptor identity drift")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        body = b"".join(chunks)
    finally:
        os.close(fd)
    after, after_mode = _regular_identity(path)
    if after != before or after_mode != mode or len(body) != byte_count or _digest(body) != expected:
        raise StageError("staging source changed or its SHA differs from authority")
    return body


def _read_regular_for_plan(path: Path) -> tuple[bytes, tuple[int, int, int], int]:
    """Held-FD body read for immutable code/receipt/model plan leaves."""
    before, mode = _regular_identity(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (int(opened.st_dev), int(opened.st_ino), int(opened.st_size)) != before
        ):
            raise StageError("stage-plan source descriptor identity drift")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        body = b"".join(chunks)
    finally:
        os.close(fd)
    after, after_mode = _regular_identity(path)
    if after != before or after_mode != mode or len(body) != before[2]:
        raise StageError("stage-plan source changed during held-FD read")
    return body, before, mode


def full_training_artifact_names() -> tuple[str, ...]:
    return (
        "attempt.json", "launch.json", "source_authority.json", "throughput100.json",
        *(f"epoch-{epoch:02d}.json" for epoch in range(48)),
        *(f"checkpoint-{epoch:02d}.pt" for epoch in (44, 45, 46, 47)),
        "swa_final4.pt", "terminal.json",
    )


@dataclass(frozen=True)
class StageFile:
    source: Path
    destination_relative: str
    byte_count: int
    sha256: str
    role: str
    source_mode: int
    destination_mode: int = 0o444

    def payload(self) -> dict[str, object]:
        destination = _safe_relative(self.destination_relative, "stage destination")
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise StageError("stage byte count must be a nonnegative integer")
        allowed_roles = {
            "code_or_metadata", "base_authority_evidence", "posterior_full_mirror",
            "sealed_cell_d_evidence", "sealed_cell_d_model", "evaluation_nwb", "source_training_nwb",
        }
        if self.role not in allowed_roles or self.destination_mode != 0o444:
            raise StageError("stage role/destination mode drift")
        # The reviewable workspace leaves (code, metadata, immutable receipts,
        # models, and the six selected NWBs) are regular non-symlink files
        # whose observed modes may be 0444, 0600, 0644, or 0664.  Admission is
        # still exact held-FD SHA/size based; every remote destination is
        # independently created as 0444, so accepting local group-writable
        # source metadata never weakens the immutable staged copy.
        allowed_source_modes = {0o444, 0o600, 0o644, 0o664}
        if self.source_mode not in allowed_source_modes:
            raise StageError("stage source mode is not reviewed")
        return {
            "source": str(self.source), "destination_relative": destination, "bytes": self.byte_count,
            "sha256": _sha(self.sha256, "stage file SHA"), "role": self.role,
            "source_mode": format(self.source_mode, "04o"), "destination_mode": "0444",
        }


@dataclass(frozen=True)
class RemoteStagePlan:
    identity: Mapping[str, object]
    stage_root: str
    score_root_relative: str
    files: tuple[StageFile, ...]

    def payload(self) -> dict[str, object]:
        identity = contract._copy_mapping(self.identity, "quick-stage identity")
        if identity.get("classification") != contract.CLASSIFICATION:
            raise StageError("stage identity must be explicitly non-governing")
        if identity.get("v1_failed_predecessor") != contract.validate_v1_failed_predecessor_evidence(
            contract.V1FailedPredecessorEvidence(),
        ):
            raise StageError("stage V1 failed-predecessor binding drift")
        if identity.get("v2_failed_predecessor") != contract.validate_v2_failed_predecessor_evidence(
            contract.V2FailedPredecessorEvidence(),
        ):
            raise StageError("stage V2 failed-predecessor binding drift")
        if self.stage_root != contract.REMOTE_STAGE_ROOT or self.score_root_relative != contract.REMOTE_SCORE_ROOT_RELATIVE:
            raise StageError("remote stage/result root literal drift")
        rows = [item.payload() for item in self.files]
        destinations = [item["destination_relative"] for item in rows]
        if len(destinations) != len(set(destinations)):
            raise StageError("stage contains duplicate destination leaves")
        if any(item["role"] == "source_training_nwb" for item in rows):
            raise StageError("source-training NWB is forbidden from quick-screen stage")
        nwbs = [item for item in rows if item["role"] == "evaluation_nwb"]
        if len(nwbs) != 6 or sum(int(item["bytes"]) for item in nwbs) != contract.selected_transfer_bytes():
            raise StageError("stage must contain exactly six frozen evaluation NWBs")
        expected_nwbs = []
        for surface in contract.SURFACES:
            subject = "sub-C" if surface == contract.WITHIN else "sub-M"
            for _index, _asset_id, _session, frozen_path, byte_count, digest in contract._SELECTED_LITERALS[surface]:
                expected_nwbs.append((
                    f"{contract.REMOTE_EVALUATION_ROOT_RELATIVE}/{subject}/{Path(frozen_path).name}",
                    byte_count, digest,
                ))
        actual_nwbs = sorted((item["destination_relative"], item["bytes"], item["sha256"]) for item in nwbs)
        if actual_nwbs != sorted(expected_nwbs):
            raise StageError("stage evaluation NWB rows differ from frozen selected authorities")
        body = {
            "schema": "posterior_carrier_quick_screen_remote_stage_plan_v3",
            "classification": contract.CLASSIFICATION,
            "cell": contract.CELL,
            "phase": contract.PHASE,
            "identity": identity,
            "v1_failed_predecessor": identity["v1_failed_predecessor"],
            "v2_failed_predecessor": identity["v2_failed_predecessor"],
            "stage_root": self.stage_root,
            "score_root_relative": self.score_root_relative,
            # The transport may create this empty parent with the fresh stage
            # tree, but it must not create the canonical score root itself.
            # ``reserve_result_artifact`` then O_EXCL-reserves that leaf only
            # after the reviewed in-process scoring capability is present.
            "score_result_parent_relative": "tfpd_exploration/results",
            "files": rows,
            "copy_protocol": "local_held_fd_sha256_verified_regular_copy_to_fresh_remote_stage",
            "nwb_assets_in_stage": True,
            "source_training_nwbs_in_stage": False,
            "result_root_created_by_stage": False,
        }
        return {**body, "plan_sha256": _digest(_json(body))}


def _full_rosters() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        (
            "sub-C_ses-CO-20151103", "sub-C_ses-CO-20151104", "sub-C_ses-CO-20151106",
            "sub-C_ses-CO-20151109", "sub-C_ses-CO-20151110", "sub-C_ses-CO-20151112",
        ),
        (
            "sub-M_ses-CO-20140307", "sub-M_ses-CO-20140626", "sub-M_ses-CO-20140627",
            "sub-M_ses-CO-20141203", "sub-M_ses-CO-20150511", "sub-M_ses-CO-20150512",
            "sub-M_ses-CO-20150610", "sub-M_ses-CO-20150611", "sub-M_ses-CO-20150612",
            "sub-M_ses-CO-20150615", "sub-M_ses-CO-20150616", "sub-M_ses-CO-20150617",
            "sub-M_ses-CO-20150623", "sub-M_ses-CO-20150625", "sub-M_ses-CO-20150626",
        ),
    )


def derive_selected_assets_from_existing_authorities(root: Path) -> dict[str, list[dict[str, object]]]:
    """Metadata-only C1/v2 derivation; never used by the public dry path."""
    base = Path(root).absolute()
    path = base / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py"
    spec = importlib.util.spec_from_file_location("_quick_screen_fixed_asset_deriver", path)
    if spec is None or spec.loader is None:
        raise StageError("cannot load fixed asset derivation helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    within, external = _full_rosters()
    try:
        full_rows = module.derive_fixed_input_assets(base, within_roster=within, external_roster=external)
        return contract.select_from_full_authority_rows(full_rows)
    except (module.PhysicalScoreError, contract.QuickScreenError) as error:
        raise StageError(f"fixed selected asset derivation failed: {error}") from error


def _require_descriptor_derived_selection(
    root: Path,
    *,
    reviewed_selection: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    """Prevent a literal/caller row table from replacing C1/v2 authority.

    The literals in the dry identity make the intended subset independently
    inspectable, but the stage must still descriptor-derive the full C1 val-6
    and external-15 authorities at the final pre-transfer boundary.  This
    detects a manifest/ledger substitution even if a supplied three-row map
    remains internally self-consistent.
    """
    supplied = contract.selected_assets_from_payload(reviewed_selection)
    supplied_payload = {
        surface: [item.payload() for item in supplied[surface]] for surface in contract.SURFACES
    }
    derived = derive_selected_assets_from_existing_authorities(root)
    if derived != supplied_payload:
        raise StageError("descriptor-derived C1/v2 selected rows differ from reviewed quick identity")
    return derived


def _pair_files(
    root: Path,
    relative: str,
    *,
    role: str,
    expected_body_sha256: str | None = None,
) -> tuple[StageFile, StageFile]:
    """Return one immutable body+sidecar pair, optionally pinned to a known body.

    A self-consistent ``.sha256`` sidecar is necessary but not sufficient for
    staging: it would otherwise allow a swapped model/receipt pair to enter a
    fresh engineering stage.  Callers use ``expected_body_sha256`` for every
    predecessor whose authoritative digest is already fixed either as a
    literal or inside the descriptor-verified base preflight.
    """
    result: list[StageFile] = []
    body_source = Path(root).absolute() / relative
    body, identity, mode = _read_regular_for_plan(body_source)
    sidecar_source = Path(root).absolute() / f"{relative}.sha256"
    sidecar, sidecar_identity, sidecar_mode = _read_regular_for_plan(sidecar_source)
    digest = _digest(body)
    if expected_body_sha256 is not None and digest != _sha(expected_body_sha256, f"expected stage body SHA {relative}"):
        raise StageError("immutable stage pair body differs from its reviewed predecessor SHA")
    # The repository-wide immutable-pair convention names the basename in a
    # sidecar even if the body is addressed beneath a nested result root.
    # Destination layout remains fully relative; only the sidecar spelling is
    # basename-canonical.
    canonical_name = Path(relative).name
    if mode != 0o444 or sidecar_mode != 0o444 or sidecar != f"{digest}  {canonical_name}\n".encode("ascii"):
        raise StageError("immutable stage pair body/sidecar/mode drift")
    result.append(StageFile(body_source, relative, identity[2], digest, role, mode))
    result.append(StageFile(sidecar_source, f"{relative}.sha256", sidecar_identity[2], _digest(sidecar), role, sidecar_mode))
    return result[0], result[1]


def _immutable_json_pair_for_plan(
    root: Path,
    relative: str,
    *,
    expected_body_sha256: str,
) -> dict[str, object]:
    """Read a small frozen JSON pair without accepting a mutable pathname.

    This is intentionally limited to the base official preflight.  Model and
    checkpoint leaves remain opaque bytes; their expected SHA map is sourced
    from this already digest-pinned authority rather than by deserializing a
    checkpoint during plan construction.
    """
    _pair_files(root, relative, role="base_authority_evidence", expected_body_sha256=expected_body_sha256)
    body, _identity, _mode = _read_regular_for_plan(Path(root).absolute() / relative)
    if _digest(body) != _sha(expected_body_sha256, f"expected JSON authority SHA {relative}"):
        raise StageError("immutable JSON authority changed during plan read")
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise StageError("immutable JSON authority is not a JSON object") from error
    if not isinstance(value, dict):
        raise StageError("immutable JSON authority root must be an object")
    return value


def _full_mirror_sha_map_from_pinned_preflight(root: Path) -> dict[str, str]:
    """Derive the exact opaque full-mirror map from the sealed base authority."""
    preflight = _immutable_json_pair_for_plan(
        root,
        contract.BASE_OFFICIAL_PREFLIGHT_RELATIVE,
        expected_body_sha256=contract.BASE_OFFICIAL_PREFLIGHT_SHA256,
    )
    provenance = preflight.get("full_import_provenance")
    if not isinstance(provenance, Mapping):
        raise StageError("base preflight lacks full import provenance")
    artifact_map = provenance.get("imported_artifact_sha256s")
    names = full_training_artifact_names()
    if not isinstance(artifact_map, Mapping) or set(artifact_map) != set(names):
        raise StageError("base preflight full-mirror artifact topology drift")
    result = {name: _sha(artifact_map[name], f"preflight full-mirror SHA {name}") for name in names}
    if (
        result["terminal.json"] != contract.POSTERIOR_FULL_TERMINAL_SHA256
        or result["swa_final4.pt"] != contract.POSTERIOR_FULL_SWA_SHA256
        or result["source_authority.json"] != contract.POSTERIOR_FULL_SOURCE_AUTHORITY_SHA256
    ):
        raise StageError("base preflight full-mirror anchors differ from quick-screen predecessors")
    return result


def _selected_nwb_files(
    selected_assets: Mapping[str, Sequence[Mapping[str, object]]], source_paths: Mapping[tuple[str, str], Path],
) -> tuple[StageFile, ...]:
    selected = contract.selected_assets_from_payload(selected_assets)
    expected = {(surface, item.session) for surface in contract.SURFACES for item in selected[surface]}
    if set(source_paths) != expected:
        raise StageError("source paths must bind exactly the six frozen selected rows")
    output: list[StageFile] = []
    for surface in contract.SURFACES:
        for asset in selected[surface]:
            source = Path(source_paths[(surface, asset.session)]).absolute()
            identity, mode = _regular_identity(source)
            if identity[2] != asset.byte_count:
                raise StageError("selected local NWB size does not match frozen authority")
            subject = "sub-C" if surface == contract.WITHIN else "sub-M"
            output.append(StageFile(
                source=source,
                destination_relative=f"{contract.REMOTE_EVALUATION_ROOT_RELATIVE}/{subject}/{Path(asset.frozen_path).name}",
                byte_count=asset.byte_count, sha256=asset.sha256, role="evaluation_nwb", source_mode=mode,
            ))
    return tuple(output)


def build_remote_stage_plan(
    *, root: Path, identity: contract.QuickScreenIdentity,
    selected_assets: Mapping[str, Sequence[Mapping[str, object]]],
    source_paths: Mapping[tuple[str, str], Path],
) -> RemoteStagePlan:
    """Build a bounded plan without evaluation-NWB or network access.

    Code, receipt, and model bytes are descriptor-verified because their exact
    transfer map is part of the reviewed plan; no model is deserialized and no
    evaluation NWB body is opened until a later root-reviewed transfer call.
    """
    identity_payload = contract.validate_identity(identity)
    if selected_assets != identity_payload["selected_assets"]:
        raise StageError("selected assets must exactly equal the reviewed identity")
    base = Path(root).absolute()
    derived_selected_assets = _require_descriptor_derived_selection(
        base, reviewed_selection=selected_assets,
    )
    closure = identity.closure.payload()
    files: list[StageFile] = []
    for relative in contract.IMPLEMENTATION_CLOSURE:
        source = base / relative
        body, descriptor, mode = _read_regular_for_plan(source)
        if _digest(body) != closure["sha256_by_path"][relative]:
            raise StageError("stage code/metadata bytes differ from reviewed implementation closure")
        files.append(StageFile(
            source=source, destination_relative=relative, byte_count=descriptor[2],
            sha256=closure["sha256_by_path"][relative], role="code_or_metadata", source_mode=mode,
        ))
    files.extend(_pair_files(
        base,
        contract.BASE_OFFICIAL_PREFLIGHT_RELATIVE,
        role="base_authority_evidence",
        expected_body_sha256=contract.BASE_OFFICIAL_PREFLIGHT_SHA256,
    ))
    files.extend(_pair_files(
        base,
        contract.BASE_ROOT_AUTHORIZATION_RELATIVE,
        role="base_authority_evidence",
        expected_body_sha256=contract.BASE_ROOT_AUTHORIZATION_SHA256,
    ))
    full_mirror_sha256s = _full_mirror_sha_map_from_pinned_preflight(base)
    for name in full_training_artifact_names():
        files.extend(_pair_files(
            base,
            f"{contract.FULL_MIRROR_ROOT}/{name}",
            role="posterior_full_mirror",
            expected_body_sha256=full_mirror_sha256s[name],
        ))
    for relative, role, expected_sha256 in (
        (contract.SEALED_CELL_D_TERMINAL_RELATIVE, "sealed_cell_d_evidence", contract.SEALED_CELL_D_TERMINAL_SHA256),
        (contract.SEALED_CELL_D_SWA_RELATIVE, "sealed_cell_d_model", contract.SEALED_CELL_D_SWA_SHA256),
        (contract.SEALED_CELL_D_BASELINE_RELATIVE, "sealed_cell_d_evidence", contract.SEALED_CELL_D_BASELINE_SHA256),
    ):
        files.extend(_pair_files(base, relative, role=role, expected_body_sha256=expected_sha256))
    files.extend(_selected_nwb_files(derived_selected_assets, source_paths))
    return RemoteStagePlan(identity_payload, contract.REMOTE_STAGE_ROOT, contract.REMOTE_SCORE_ROOT_RELATIVE, tuple(files))


def verify_local_stage_sources(plan: RemoteStagePlan) -> dict[str, str]:
    """Rehash all planned local bytes just before the reviewed transfer."""
    plan_payload = plan.payload()
    result: dict[str, str] = {}
    for item in plan.files:
        body = _read_regular_exact(item.source, byte_count=item.byte_count, sha256=item.sha256, mode=item.source_mode)
        result[item.destination_relative] = _digest(body)
    if len(result) != len(plan.files) or plan_payload["plan_sha256"] != _digest(_json({
        key: value for key, value in plan_payload.items() if key != "plan_sha256"
    })):
        raise StageError("stage plan/source verification drift")
    return result


_STAGE_SEAL = object()


@dataclass(frozen=True)
class RemoteStageCapability:
    plan_sha256: str
    _seal: object = field(repr=False, compare=False)


def _issue_root_review_stage_capability(plan: RemoteStagePlan) -> RemoteStageCapability:
    return RemoteStageCapability(plan.payload()["plan_sha256"], _STAGE_SEAL)


def _require_stage_capability(value: object, *, plan: RemoteStagePlan) -> None:
    if (
        not isinstance(value, RemoteStageCapability) or value._seal is not _STAGE_SEAL
        or value.plan_sha256 != plan.payload()["plan_sha256"]
    ):
        raise StageError("root-reviewed in-process stage capability required")


class StageTransport(Protocol):
    """A separately reviewed remote writer; tests inject a pure fake.

    A physical implementation must reserve ``stage_root`` fresh and
    non-symlinked, create only the declared empty result *parent*, write each
    regular leaf with O_EXCL/fsync and mode 0444, and reject an existing score
    root.  Those remote filesystem operations deliberately remain outside the
    public CLI and require a second root review.
    """

    def begin_fresh_stage(self, *, stage_root: str, manifest: Mapping[str, object]) -> None: ...

    def write_regular(self, *, destination_relative: str, body: bytes, sha256: str, mode: int) -> None: ...

    def finalize_stage(self, *, expected_manifest_sha256: str) -> Mapping[str, object]: ...

    def abort_owned_stage(self) -> None: ...


def stage_reviewed_plan(*, plan: RemoteStagePlan, capability: object, transport: StageTransport) -> Mapping[str, object]:
    """One-shot reviewed staging lifecycle; no public CLI reaches this function."""
    _require_stage_capability(capability, plan=plan)
    payload = plan.payload()
    manifest = {
        "schema": "posterior_carrier_quick_screen_remote_stage_manifest_v3",
        "plan_sha256": payload["plan_sha256"], "stage_root": plan.stage_root,
        "fresh_result_root_relative": plan.score_root_relative,
        "fresh_result_parent_relative": payload["score_result_parent_relative"],
        "result_root_created_by_stage": False,
        "v1_failed_predecessor": payload["v1_failed_predecessor"],
        "v2_failed_predecessor": payload["v2_failed_predecessor"],
        "files": [
            {key: item[key] for key in ("destination_relative", "bytes", "sha256", "role", "destination_mode")}
            for item in payload["files"]
        ],
    }
    manifest_sha = _digest(_json(manifest))
    begun = False
    try:
        transport.begin_fresh_stage(stage_root=plan.stage_root, manifest=manifest)
        begun = True
        for item in plan.files:
            body = _read_regular_exact(item.source, byte_count=item.byte_count, sha256=item.sha256, mode=item.source_mode)
            transport.write_regular(
                destination_relative=item.destination_relative, body=body, sha256=item.sha256, mode=item.destination_mode,
            )
        final = transport.finalize_stage(expected_manifest_sha256=manifest_sha)
        if not isinstance(final, Mapping) or final.get("manifest_sha256") != manifest_sha or final.get("stage_root") != plan.stage_root:
            raise StageError("remote stage final attestation drift")
        return dict(final)
    except BaseException:
        if begun:
            try:
                transport.abort_owned_stage()
            except BaseException:
                pass
        raise
