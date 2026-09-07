"""No-data/no-CUDA tests for the future Posterior full-result importer.

Every remote result in this file is a synthetic byte stream in a temporary
directory or ``BytesIO``.  These tests never contact SSH, open a real result,
load a checkpoint tensor, import Torch, or create the canonical workspace
mirror.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tfpd_exploration" / "src" / "posterior_carrier_v1" / "full_result_import.py"
MATCHED_SOURCE = ROOT / "tfpd_exploration" / "src" / "posterior_carrier_v1" / "matched_score.py"
CLI = ROOT / "tfpd_exploration" / "scripts" / "run_posterior_carrier_full_result_import.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("posterior_carrier_full_result_import_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_matched_contract() -> object:
    spec = importlib.util.spec_from_file_location("posterior_carrier_import_closure_test", MATCHED_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


imp = _load()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("ascii"))


class _FakeTransport:
    def __init__(self, stream: bytes) -> None:
        self.stream = stream
        self.opened = 0
        self.finalized: list[bool] = []

    def open_completed_terminal_stream(self, **_kwargs: object) -> object:
        self.opened += 1
        return imp.TransportHandle(io.BytesIO(self.stream), lambda success: self.finalized.append(success))


def _completed_snapshot(
    *,
    terminal_indent: int | None = 2,
    mutate_manifest: Callable[[dict[str, object]], None] | None = None,
    mutate_leaves: Callable[[dict[str, bytes]], None] | None = None,
) -> tuple[dict[str, object], dict[str, bytes], object, object]:
    """Build a fully synthetic completed remote body/sidecar topology."""
    closure_sha = _sha_text("synthetic-full-closure")
    bodies: dict[str, bytes] = {}
    for name in imp.full_training_artifact_names():
        if name != "terminal.json":
            # Deliberately use a non-canonical pretty terminal below.  The
            # remote manifest must bind its actual digest, not assume a local
            # JSON serialization convention before the leaf arrives.
            bodies[name] = json.dumps({"synthetic": name}, sort_keys=True).encode("utf-8")
    artifact_sha = {name: _sha_bytes(body) for name, body in bodies.items()}
    terminal = {
        "schema": "posterior_carrier_full_train_terminal_v1",
        "cell": imp.CELL,
        "phase": imp.FULL_TRAIN_PHASE,
        "spec": {"synthetic": "fixed-public-full-spec"},
        "identity": {"synthetic": "remote-stage-identity"},
        "final_identity": {"synthetic": "remote-stage-identity"},
        "status": "FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER",
        "attempt_sha256": artifact_sha["attempt.json"],
        "launch_sha256": artifact_sha["launch.json"],
        "source_authority_sha256": artifact_sha["source_authority.json"],
        "artifact_sha256s": {
            name: artifact_sha[name]
            for name in imp.full_training_artifact_names()
            if name != "terminal.json"
        },
        "full_launch_closure": {"closure_sha256": closure_sha},
        "full_final_closure": {"closure_sha256": closure_sha},
        "phase_b_v3_launch_closure": {"closure_sha256": _sha_text("synthetic-v3-closure")},
        "phase_b_v3_final_closure": {"closure_sha256": _sha_text("synthetic-v3-closure")},
        "checkpoint_state_sha256s": {str(epoch): _sha_text(f"state-{epoch}") for epoch in imp.CHECKPOINT_EPOCHS},
        "swa_state_sha256": _sha_text("synthetic-swa-state"),
        "swa_proof": {"synthetic": "checked-later-by-matched-scorer"},
        "boundaries": imp._source_only_boundaries(),
    }
    bodies["terminal.json"] = json.dumps(
        terminal, sort_keys=True, indent=terminal_indent, separators=None if terminal_indent else (",", ":")
    ).encode("utf-8")
    artifact_sha["terminal.json"] = _sha_bytes(bodies["terminal.json"])
    leaves: dict[str, bytes] = {}
    for name in imp.full_training_artifact_names():
        leaves[name] = bodies[name]
        leaves[f"{name}.sha256"] = f"{artifact_sha[name]}  {name}\n".encode("ascii")
    manifest: dict[str, object] = {
        "schema": imp.REMOTE_MANIFEST_SCHEMA,
        "endpoint": imp.RemoteEndpoint().payload(),
        "stage_descriptor_identity": [66308, 991],
        "result_descriptor_identity": [66308, 992],
        "topology": list(imp.full_training_artifact_names()),
        "artifact_sha256s": dict(artifact_sha),
        "leaves": [
            {
                "name": name,
                "bytes": len(leaves[name]),
                "sha256": _sha_bytes(leaves[name]),
                "kind": "regular",
                "mode": "0444",
            }
            for name in imp.remote_leaf_names()
        ],
        "terminal": terminal,
        "terminal_sha256": artifact_sha["terminal.json"],
        "full_closure_sha256": closure_sha,
        "status": "REMOTE_FULL_TERMINAL_SNAPSHOT_READY",
    }
    if mutate_leaves is not None:
        mutate_leaves(leaves)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    authority = imp.RemoteTerminalAuthority(
        stage_identity=imp.DirectoryIdentity(66308, 991),
        result_identity=imp.DirectoryIdentity(66308, 992),
        terminal_sha256=artifact_sha["terminal.json"],
        full_closure_sha256=closure_sha,
    )
    capability = imp._issue_root_review_capability_for_completed_remote_full_result(
        endpoint=imp.RemoteEndpoint(), terminal_authority=authority, host_key_pin=imp.HostKeyPin()
    )
    return manifest, leaves, capability, authority


def _fresh_root(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic_repository"
    (root / "tfpd_exploration" / "results").mkdir(parents=True)
    return root


def _destination(root: Path) -> Path:
    return root / imp.LOCAL_MIRROR_RELATIVE


def _import(root: Path, manifest: Mapping[str, object], leaves: Mapping[str, bytes], capability: object) -> tuple[object, _FakeTransport]:
    transport = _FakeTransport(imp.build_framed_snapshot(manifest, leaves))
    return imp.import_completed_remote_full_result(root, capability=capability, transport=transport), transport


def test_success_is_descriptor_safe_atomic_and_provenance_compatible(tmp_path: Path) -> None:
    manifest, leaves, capability, _authority = _completed_snapshot()
    result, transport = _import(_fresh_root(tmp_path), manifest, leaves, capability)
    assert transport.opened == 1 and transport.finalized == [True]
    assert result.destination == _destination(tmp_path / "synthetic_repository")
    assert {entry.name for entry in result.destination.iterdir()} == set(imp.remote_leaf_names())
    for name in imp.remote_leaf_names():
        assert (result.destination / name).read_bytes() == leaves[name]
        assert (result.destination / name).stat().st_mode & 0o777 == 0o444
    assert imp.validate_imported_full_mirror_provenance(result.provenance) == result.provenance
    assert imp.validate_remote_import_attestation(result.attestation) == result.attestation


@pytest.mark.parametrize("mutation", ["extra", "missing", "failure", "symlink", "mode"])
def test_remote_topology_and_leaf_metadata_fail_before_local_reservation(tmp_path: Path, mutation: str) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        if mutation == "extra":
            manifest["topology"] = [*manifest["topology"], "failure.json"]
        elif mutation == "missing":
            manifest["leaves"] = list(manifest["leaves"])[1:]
        elif mutation == "failure":
            manifest["status"] = "REMOTE_FAILURE"
        elif mutation == "symlink":
            list(manifest["leaves"])[0]["kind"] = "symlink"
        elif mutation == "mode":
            list(manifest["leaves"])[0]["mode"] = "0644"
        else:  # pragma: no cover - exhaustive parameter domain above
            raise AssertionError(mutation)

    manifest, leaves, capability, _authority = _completed_snapshot(mutate_manifest=mutate)
    root = _fresh_root(tmp_path)
    transport = _FakeTransport(imp.build_framed_snapshot(manifest, leaves))
    with pytest.raises(imp.FullResultImportError):
        imp.import_completed_remote_full_result(root, capability=capability, transport=transport)
    assert transport.finalized == [False]
    assert not _destination(root).exists()


def test_incomplete_remote_terminal_graph_fails_before_any_local_publication(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        terminal = dict(manifest["terminal"])
        terminal.pop("swa_proof")
        manifest["terminal"] = terminal

    manifest, leaves, capability, _authority = _completed_snapshot(mutate_manifest=mutate)
    root = _fresh_root(tmp_path)
    transport = _FakeTransport(imp.build_framed_snapshot(manifest, leaves))
    with pytest.raises(imp.FullResultImportError, match="terminal schema"):
        imp.import_completed_remote_full_result(root, capability=capability, transport=transport)
    assert transport.finalized == [False]
    assert not _destination(root).exists()


def test_bad_sidecar_or_body_swap_rolls_back_only_owned_fresh_root(tmp_path: Path) -> None:
    def alter_sidecar(leaves: dict[str, bytes]) -> None:
        leaves["epoch-03.json.sha256"] = b"0" * 64 + b"  epoch-03.json\n"

    manifest, leaves, capability, _authority = _completed_snapshot(mutate_leaves=alter_sidecar)
    # The manifest faithfully describes the malicious stream: SHA transport
    # checks alone pass, then exact body/sidecar semantics force rollback.
    for row in manifest["leaves"]:
        row["sha256"] = _sha_bytes(leaves[row["name"]])
        row["bytes"] = len(leaves[row["name"]])
    root = _fresh_root(tmp_path)
    transport = _FakeTransport(imp.build_framed_snapshot(manifest, leaves))
    with pytest.raises(imp.FullResultImportError, match="sidecar"):
        imp.import_completed_remote_full_result(root, capability=capability, transport=transport)
    assert transport.finalized == [False]
    assert not _destination(root).exists()

    manifest, leaves, capability, _authority = _completed_snapshot()
    leaves["epoch-04.json"] = b"X" * len(leaves["epoch-04.json"])
    root = _fresh_root(tmp_path / "body")
    transport = _FakeTransport(imp.build_framed_snapshot(manifest, leaves))
    with pytest.raises(imp.FullResultImportError, match="SHA"):
        imp.import_completed_remote_full_result(root, capability=capability, transport=transport)
    assert transport.finalized == [False]
    assert not _destination(root).exists()


def test_framed_stream_rejects_truncation_trailing_injection_and_path_traversal(tmp_path: Path) -> None:
    manifest, leaves, capability, _authority = _completed_snapshot()
    full = imp.build_framed_snapshot(manifest, leaves)
    root = _fresh_root(tmp_path / "truncated")
    transport = _FakeTransport(full[:-7])
    with pytest.raises(imp.FullResultImportError, match="truncated"):
        imp.import_completed_remote_full_result(root, capability=capability, transport=transport)
    assert not _destination(root).exists()

    root = _fresh_root(tmp_path / "injected")
    transport = _FakeTransport(full + b"x")
    with pytest.raises(imp.FullResultImportError, match="trailing"):
        imp.import_completed_remote_full_result(root, capability=capability, transport=transport)
    assert not _destination(root).exists()

    reader = imp.FramedSnapshotReader(io.BytesIO(imp.FRAME_MAGIC + imp._frame("manifest", None, imp._json(manifest))))
    reader.read_manifest()
    malicious = imp._frame("leaf", "../terminal.json", b"x")
    reader = imp.FramedSnapshotReader(io.BytesIO(imp.FRAME_MAGIC + imp._frame("manifest", None, imp._json(manifest)) + malicious))
    reader.read_manifest()
    with pytest.raises(imp.FullResultImportError, match="path traversal"):
        reader.copy_next_leaf(expected_name="attempt.json", expected_bytes=1, expected_sha256=_sha_text("x"), write=lambda _x: None)

    class _OverreadingStream:
        def read(self, count: int) -> bytes:
            return b"x" * (count + 1)

    with pytest.raises(imp.FullResultImportError, match="over-read"):
        imp.FramedSnapshotReader(_OverreadingStream()).read_manifest()


def test_destination_collision_symlink_and_partial_stream_never_leave_a_canonical_root(tmp_path: Path) -> None:
    manifest, leaves, capability, _authority = _completed_snapshot()
    root = _fresh_root(tmp_path / "collision")
    destination = _destination(root)
    destination.mkdir()
    transport = _FakeTransport(imp.build_framed_snapshot(manifest, leaves))
    with pytest.raises(imp.FullResultImportError, match="already exists"):
        imp.import_completed_remote_full_result(root, capability=capability, transport=transport)
    assert transport.opened == 0

    root = _fresh_root(tmp_path / "symlink")
    destination = _destination(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination.symlink_to(outside, target_is_directory=True)
    with pytest.raises(imp.FullResultImportError, match="already exists"):
        imp.import_completed_remote_full_result(root, capability=capability, transport=_FakeTransport(b""))
    assert outside.exists() and not list(outside.iterdir())

    root = _fresh_root(tmp_path / "partial")
    stream = imp.build_framed_snapshot(manifest, leaves)
    # Preserve enough frames to reserve/write a few files, then fail exactly
    # inside a later body; rollback must remove only that newly owned root.
    transport = _FakeTransport(stream[: len(stream) // 2])
    with pytest.raises(imp.FullResultImportError):
        imp.import_completed_remote_full_result(root, capability=capability, transport=transport)
    assert not _destination(root).exists()


def test_held_local_parent_and_named_root_identity_gate_rejects_path_swaps(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    transaction = imp.LocalMirrorTransaction.reserve(root)
    parent = root / "tfpd_exploration" / "results"
    moved = root / "tfpd_exploration" / "results_moved"
    parent.rename(moved)
    parent.mkdir()
    try:
        with pytest.raises(imp.FullResultImportError, match="named-parent identity"):
            transaction._assert_identity()
    finally:
        transaction.close()
    assert not _destination(root).exists()


def test_reservation_failure_rolls_back_its_just_created_named_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fresh_root(tmp_path)
    original_open = imp.os.open
    destination_name = Path(imp.LOCAL_MIRROR_RELATIVE).name

    def reject_new_root(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if path == destination_name and dir_fd is not None:
            raise OSError("synthetic post-mkdir open failure")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(imp.os, "open", reject_new_root)
    with pytest.raises(OSError, match="post-mkdir"):
        imp.LocalMirrorTransaction.reserve(root)
    assert not _destination(root).exists()


def test_rollback_never_unlinks_a_same_name_leaf_whose_inode_was_swapped(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    transaction = imp.LocalMirrorTransaction.reserve(root)
    name = "attempt.json"
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=transaction.descriptor)
    try:
        os.write(descriptor, b"owned")
        transaction.created_leaf_identities[name] = (
            os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino,
        )
    finally:
        os.close(descriptor)
    # Keep the old inode alive outside the owned root so the filesystem cannot
    # recycle it for the replacement and mask the adversarial swap.
    os.rename(
        name, "saved_owned_inode", src_dir_fd=transaction.descriptor, dst_dir_fd=transaction.parent_descriptor,
    )
    replacement = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=transaction.descriptor)
    os.close(replacement)
    try:
        with pytest.raises(imp.FullResultImportError, match="ownership identity"):
            transaction.rollback()
    finally:
        transaction.close()
        parent_fd = os.open(root / "tfpd_exploration" / "results", os.O_RDONLY)
        try:
            os.unlink("saved_owned_inode", dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
    # The intentionally adversarial replacement remains; a rollback must not
    # erase a same-name inode it did not create.
    assert (_destination(root) / name).exists()


def test_provenance_and_attestation_substitution_are_rejected(tmp_path: Path) -> None:
    manifest, leaves, capability, _authority = _completed_snapshot()
    result, _transport = _import(_fresh_root(tmp_path), manifest, leaves, capability)
    forged = dict(result.provenance)
    forged["remote_result_root"] = "ssh://wrong/route"
    forged["provenance_sha256"] = imp._digest(imp._json({key: forged[key] for key in forged if key != "provenance_sha256"}))
    with pytest.raises(imp.FullResultImportError, match="remote-root"):
        imp.validate_imported_full_mirror_provenance(forged)
    attestation = dict(result.attestation)
    attestation["result_descriptor_identity"] = [9, 9]
    attestation["attestation_sha256"] = imp._digest(imp._json({key: attestation[key] for key in attestation if key != "attestation_sha256"}))
    with pytest.raises(imp.FullResultImportError, match="binding"):
        imp.validate_remote_import_attestation(attestation)


def test_capability_and_physical_ssh_pin_are_fail_closed_without_live_known_hosts_line() -> None:
    manifest, leaves, capability, authority = _completed_snapshot()
    assert capability.validate()["host_key_pin"]["known_hosts_line_sha256"] is None
    called: list[bool] = []
    transport = imp.SshRemoteTransport(popen_factory=lambda **_kwargs: called.append(True))
    with pytest.raises(imp.FullResultImportError, match="known-hosts"):
        transport.open_completed_terminal_stream(
            endpoint=imp.RemoteEndpoint(), terminal_authority=authority, host_key_pin=imp.HostKeyPin()
        )
    assert called == []
    assert manifest and leaves  # silence fixture-value lint without opening a connection


def test_ssh_command_is_literal_batch_pinned_and_never_uses_a_shell_or_agent() -> None:
    command = imp.SshRemoteTransport.command(
        endpoint=imp.RemoteEndpoint(), known_hosts_path="/tmp/only-root-reviewed-known-hosts"
    )
    assert command[:3] == ("ssh", "-o", "BatchMode=yes")
    rendered = "\0".join(command)
    for required in (
        "StrictHostKeyChecking=yes",
        "UserKnownHostsFile=/tmp/only-root-reviewed-known-hosts",
        "GlobalKnownHostsFile=/dev/null",
        "UpdateHostKeys=no",
        "IdentityAgent=none",
        "IdentitiesOnly=yes",
        "AddKeysToAgent=no",
        "ForwardAgent=no",
        "ClearAllForwardings=yes",
        "ControlMaster=no",
        "xinyuan@xinyuan-ROG",
        "python3\0-",
    ):
        assert required in rendered


def test_generated_remote_program_is_descriptor_based_and_syntax_valid_without_execution() -> None:
    _manifest, _leaves, _capability, authority = _completed_snapshot()
    program = imp._remote_python_program(endpoint=imp.RemoteEndpoint(), terminal_authority=authority)
    compile(program, "<synthetic-remote-import-program>", "exec")
    assert "os.open('/', FLAGS)" in program
    assert "O_NOFOLLOW" in program
    assert "os.listdir(result)" in program
    assert "tarfile" not in program
    assert "shell=True" not in program


def test_dry_cli_is_stdlib_only_no_ssh_no_write() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI)], cwd=ROOT, text=True, capture_output=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"].startswith("DRY_ONLY")
    assert payload["boundaries"]["canonical_mirror_created"] is False
    flags = subprocess.run(
        [sys.executable, str(CLI), "--execute", "--i-have-root-reviewed-remote-import-authorization"],
        cwd=ROOT, text=True, capture_output=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
    )
    assert flags.returncode != 0
    assert "root-reviewed capability" in flags.stderr


def test_static_contract_import_never_imports_torch_or_opens_a_transport() -> None:
    # Other focused tests may legitimately import Torch in the same pytest
    # interpreter.  This dry contract itself must not bind it or any model
    # runtime namespace.
    assert "torch" not in imp.__dict__
    plan = imp.dry_plan()
    assert plan["endpoint"]["host_alias"] == "xinyuan-ROG"
    assert plan["boundaries"]["batch_mode"] is True
    assert plan["boundaries"]["ssh_agent_mutation"] is False


def test_matched_score_implementation_closure_binds_every_import_route_byte() -> None:
    matched = _load_matched_contract()
    closure = matched.implementation_closure(ROOT).payload()
    for relative in (
        "tfpd_exploration/src/posterior_carrier_v1/full_result_import.py",
        "tfpd_exploration/scripts/run_posterior_carrier_full_result_import.py",
        "tfpd_exploration/tests/test_posterior_carrier_full_result_import.py",
    ):
        assert relative in matched.IMPLEMENTATION_CLOSURE
        assert closure["sha256_by_path"][relative] == _sha_bytes((ROOT / relative).read_bytes())
