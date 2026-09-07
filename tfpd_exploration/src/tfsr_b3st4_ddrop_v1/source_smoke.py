"""TF-SR source-only smoke scaffold.  Zero-argument callers never import torch or data APIs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from .contract import IMPLEMENTATION_CLOSURE, compute_live_closure, verify_live_closure

CELL = "TFSR_B3ST4_DDROP_SEED42"
CANONICAL_RECEIPT_RELATIVE_PATH = "tfpd_exploration/results/tfsr_b3st4_ddrop_v1/source_smoke_receipt.json"
PHASE_C_CLOSURE = (
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/source_smoke.py",
    "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_source_smoke.py",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_source_smoke.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json",
)
EXPECTED_STAGE0_SHA = {
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/model.py": "3d4a3a8d4e2a68e9933e84274f2a62308a6eeb5a6978671b53e72af442148fc4",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/__init__.py": "06f3d404f3f73a7bfb69349d71f4adeb24f6306daac99e2a3e7d28a6645f730d",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_stage0.py": "c31f0dfd0190b1a0b09c6fa75ab6c7c3db1045247e161c5af6857bfdb19b92b0",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/contract.py": "c8a7f20aea6982e02ccf9d674100fbe29407403077a5adbce9e95da3d989888f",
    "tfpd_exploration/scripts/preflight_tfsr_b3st4_ddrop_seed42.py": "b1a267d928108a800f5e16d24cf2eaccd9f77bf9d13aac70a0fa72c8520fdeff",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_preflight.py": "9a8a8329ad31a2151ffd9b2b1dddd8b3a96dec697b4cba4a8a6b379607adcace",
}
ADMISSION_PREFLIGHT = "tfpd_exploration/results/admission_arms_v1/preflight_armA.json"
THETA_RECEIPT = "tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority_receipt.json"
THETA_ARTIFACT = "tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority.pt"
ADMISSION_SHA = "2632c6a6a4cfb8a4c0fb2b23e0cc8205ea323240b376903c60d5b27110f59e43"
THETA_RECEIPT_SHA = "d023dd632c4717443f1f55e924a09be1747fc58d5c30a8fb6fa38f4b7b117184"
THETA_ARTIFACT_SHA = "cef39dc8220aa253214963a32e5457dede1045e64b158e37fc267a6fb4146319"
SIDE_SEMANTIC_SHA = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
BEHAVIOR_SEMANTIC_SHA = "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
MANIFEST_SHA = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
SOURCE_LINEAGE = "sua_exploration/results/misleading_identity_swap_v2_source_authority_dev/strict27_m30_source_lineage_v3.json"
SOURCE_LINEAGE_SHA = "7375a37c8c5e59cb7e6786e0c67e930f918d59ff4dc769742391ad6dc97579fa"
MATCHING_AUTHORITY_CONSUMED_SHA = "ccebdf41b3053703c35ad2323665aa6efcd7b91b39e9f5744aa3564745a82f71"
SOURCE_T4_MEAN = [0.04627712443470955, 0.4544036388397217, 1.3432163000106812, 10.150517463684082]
SOURCE_T4_STD = [1.126278281211853, 1.284820556640625, 1.2352101802825928, 9.115250587463379]
FROZEN_DEVICE = {"internal_device": "cuda:0", "cuda_visible_devices": "1",
                 "uuid": "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
                 "bdf": "00000000:03:00.0", "name": "NVIDIA GeForce RTX 3090",
                 "memory_total_mib": 24576}
FROZEN_OPTIMIZER = {"cls": "Adam", "lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8,
                    "weight_decay": 0.0, "amsgrad": False, "schedule": "warmup_cosine",
                    "epochs": 48, "swa": "final_4_epochs", "smoke_first_step_lr": 1e-5}
FROZEN_SMOKE = {"seed": 42, "batch_size": 32, "optimizer_steps": 1, "epochs_authorized": 0}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_stage0_and_phase_c_closures(root: Path) -> dict[str, object]:
    stage0 = compute_live_closure(root)
    for path, expected in EXPECTED_STAGE0_SHA.items():
        if stage0["sha256_by_path"].get(path) != expected:
            raise RuntimeError(f"frozen Stage-0 SHA drift: {path}")
    verify_live_closure(root, stage0["sha256_by_path"], stage0["closure_sha256"])
    phase_c = compute_live_closure(root, PHASE_C_CLOSURE)
    return {"stage0": stage0, "phase_c": phase_c}


def _sealed(root: Path, relative: str, expected: str) -> bytes:
    from .contract import _canonical_regular_bytes
    path = root / relative
    body = _canonical_regular_bytes(path, expected_mode=0o444)
    if hashlib.sha256(body).hexdigest() != expected:
        raise RuntimeError(f"sealed authority SHA drift: {relative}")
    sidecar = _canonical_regular_bytes(Path(str(path) + ".sha256"), expected_mode=0o444)
    if sidecar != f"{expected}  {path.name}\n".encode():
        raise RuntimeError(f"sealed authority sidecar drift: {relative}")
    return body


def verify_canonical_source_authorities(root: Path) -> dict[str, object]:
    """Bind the admitted normalized-T4 and raw-T4 authority chain without data access."""
    admission = json.loads(_sealed(root, ADMISSION_PREFLIGHT, ADMISSION_SHA))
    theta_receipt = json.loads(_sealed(root, THETA_RECEIPT, THETA_RECEIPT_SHA))
    lineage = json.loads(_sealed(root, SOURCE_LINEAGE, SOURCE_LINEAGE_SHA))
    _sealed(root, THETA_ARTIFACT, THETA_ARTIFACT_SHA)
    contract = admission.get("data_contract", {})
    roster = contract.get("roster")
    hashes = admission.get("t4_authority_sha256")
    if admission.get("schema") != "tfpd_admission_arm_v1_preflight" or admission.get("status") != "ARM_PREFLIGHT_PASSED":
        raise RuntimeError("admission authority schema/status drift")
    if (not isinstance(roster, list) or len(roster) != 27 or len(set(roster)) != 27 or
            not all(isinstance(value, str) and value for value in roster) or not isinstance(hashes, dict) or
            set(roster) != set(hashes) or not all(_is_sha256(value) for value in hashes.values())):
        raise RuntimeError("strict-27 normalized-T4 authority roster drift")
    if contract.get("manifest_sha256") != MANIFEST_SHA or contract.get("n_train_windows") != 1086007 or contract.get("steps_per_epoch") != 33925:
        raise RuntimeError("admission manifest/window/sampler authority drift")
    normalizers = contract.get("normalizers", {})
    if normalizers.get("side_feature_semantic_sha256") != SIDE_SEMANTIC_SHA or normalizers.get("behavior_semantic_sha256") != BEHAVIOR_SEMANTIC_SHA:
        raise RuntimeError("admission normalizer semantic authority drift")
    proof = theta_receipt.get("alignment_proof")
    if theta_receipt.get("schema") != "tfpd_sparsification_theta_authority_v1" or theta_receipt.get("status") != "THETA_AUTHORITY_SEALED":
        raise RuntimeError("theta authority schema/status drift")
    if theta_receipt.get("artifact", {}).get("sha256") != THETA_ARTIFACT_SHA or not isinstance(proof, dict) or set(proof) != set(roster):
        raise RuntimeError("theta artifact/proof roster drift")
    for session in roster:
        entry = proof[session]
        if not isinstance(entry, dict) or set(entry) != {"aligned", "datamodule_channels", "n_undefined", "n_valid_directions", "side_rows", "theta_units"}:
            raise RuntimeError("theta alignment proof key drift")
        if entry["aligned"] is not True or any(type(entry[key]) is not int for key in entry if key != "aligned"):
            raise RuntimeError("theta alignment proof type drift")
        if entry["theta_units"] != entry["side_rows"] or entry["theta_units"] != entry["datamodule_channels"]:
            raise RuntimeError("theta alignment proof unit-count drift")
    descriptor = lineage.get("descriptor")
    rows = lineage.get("source_files")
    expected_lineage_keys = {"feature_version", "path", "session", "sha256", "size_bytes", "unit_count"}
    if (lineage.get("receipt_kind") != "misleading_identity_swap_v2_strict27_m30_source_lineage" or
            lineage.get("schema_version") != 1 or lineage.get("source_only") is not True or
            lineage.get("target_nwb_opened") is not False or lineage.get("validation_nwb_opened") is not False or
            lineage.get("formal_subc_test_nwb_opened") is not False or lineage.get("strict_manifest_sha256") != MANIFEST_SHA or
            lineage.get("source_train_sessions") != roster or lineage.get("source_train_session_count") != 27 or
            lineage.get("matching_authority_consumed_bytes_sha256") != MATCHING_AUTHORITY_CONSUMED_SHA or
            not isinstance(descriptor, dict) or not isinstance(rows, list) or len(rows) != 27):
        raise RuntimeError("sealed strict-27 source lineage semantic drift")
    if (descriptor.get("feature_group") != "t4" or descriptor.get("pool_size") != 30 or
            descriptor.get("bin_size_ms") != 20 or descriptor.get("window_size_bins") != 50 or
            descriptor.get("support") != "chronological_first_30_rewarded_trials" or descriptor.get("signal_view") != "sua" or
            descriptor.get("columns") != ["m_cos_phi", "m_sin_phi", "m", "b"] or
            descriptor.get("normalizer_mean_float32") != SOURCE_T4_MEAN or descriptor.get("normalizer_std_float32") != SOURCE_T4_STD or
            descriptor.get("normalizer_value_sha256") != SIDE_SEMANTIC_SHA):
        raise RuntimeError("sealed source lineage T4/M30 semantic drift")
    lineage_by_session: dict[str, dict[str, object]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_lineage_keys or row.get("session") != roster[position]:
            raise RuntimeError("sealed source lineage roster/row-key drift")
        if (row.get("feature_version") != 1 or not isinstance(row.get("path"), str) or not _is_sha256(row.get("sha256")) or
                type(row.get("size_bytes")) is not int or row["size_bytes"] <= 0 or type(row.get("unit_count")) is not int or row["unit_count"] <= 0):
            raise RuntimeError("sealed source lineage row type drift")
        expected_path = root / "sua_exploration/data/dandi_000688/sub-C" / f"{row['session']}_behavior+ecephys.nwb"
        if Path(row["path"]) != expected_path:
            raise RuntimeError("sealed source lineage canonical path drift")
        lineage_by_session[row["session"]] = row
    return {"admission_preflight": {"path": ADMISSION_PREFLIGHT, "body_sha256": ADMISSION_SHA},
            "theta_receipt": {"path": THETA_RECEIPT, "body_sha256": THETA_RECEIPT_SHA},
            "theta_artifact": {"path": THETA_ARTIFACT, "body_sha256": THETA_ARTIFACT_SHA},
            "roster": roster, "normalized_t4_sha256": hashes,
            "raw_authority_sha256": THETA_ARTIFACT_SHA, "normalizer_authority_sha256": ADMISSION_SHA,
            "side_semantic_sha256": SIDE_SEMANTIC_SHA, "behavior_semantic_sha256": BEHAVIOR_SEMANTIC_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "source_lineage": {"path": SOURCE_LINEAGE, "body_sha256": SOURCE_LINEAGE_SHA,
                               "matching_authority_consumed_bytes_sha256": MATCHING_AUTHORITY_CONSUMED_SHA,
                               "rows_by_session": lineage_by_session}}


def verify_source_lineage_file(path: Path, row: Mapping[str, object]) -> None:
    """Same-FD, no-follow streamed source-file provenance proof for live execution."""
    if not isinstance(row, Mapping) or not isinstance(row.get("path"), str) or not _is_sha256(row.get("sha256")):
        raise RuntimeError("malformed source-lineage row")
    expected = Path(row["path"])
    if path.absolute() != expected or path.resolve(strict=True) != expected:
        raise RuntimeError("source path/alias drift against sealed lineage")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_size != row.get("size_bytes"):
        raise RuntimeError("source size/type drift against sealed lineage")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise RuntimeError("source changed between lstat/open")
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
        raise RuntimeError("source changed during same-FD SHA read")
    if digest.hexdigest() != row["sha256"]:
        raise RuntimeError("source SHA drift against sealed lineage")


def recompute_verified_raw_t4(path: Path, row: Mapping[str, object], compute: Any) -> tuple[Any, Any]:
    """Bracket one raw-T4 computation with same-FD source lineage checks."""
    verify_source_lineage_file(path, row)
    raw, metadata = compute(path, feature_group="t4", pool_size=30, bin_size_ms=20, window_size=50,
                            trial_result_filter="R", signal_view="sua")
    verify_source_lineage_file(path, row)
    return raw, metadata


def load_verified_theta_artifact(root: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    """Authorized branch only: parse sealed bytes with weights_only and prove every session join."""
    import io
    import torch
    artifact = torch.load(io.BytesIO(_sealed(root, THETA_ARTIFACT, THETA_ARTIFACT_SHA)), map_location="cpu", weights_only=True)
    receipt = json.loads(_sealed(root, THETA_RECEIPT, THETA_RECEIPT_SHA))
    roster, rows, proof = authority.get("roster"), artifact.get("authority") if isinstance(artifact, dict) else None, receipt.get("alignment_proof")
    if not isinstance(artifact, dict) or artifact.get("kind") != "tfpd_sparsification_theta_authority_v1" or artifact.get("authority_sha256") != receipt.get("authority_sha256"):
        raise RuntimeError("theta artifact kind/authority drift")
    if not isinstance(roster, list) or not isinstance(rows, dict) or not isinstance(proof, dict) or set(rows) != set(roster) or set(proof) != set(roster):
        raise RuntimeError("theta artifact session join drift")
    for session in roster:
        row, aligned = rows[session], proof[session]
        if not isinstance(row, dict) or set(row) != {"theta", "valid", "n_units", "raw_t4_sha256"}:
            raise RuntimeError("theta artifact row key drift")
        units, theta, valid, raw = row["n_units"], row["theta"], row["valid"], row["raw_t4_sha256"]
        if type(units) is not int or units <= 0 or not _is_sha256(raw) or not torch.is_tensor(theta) or not torch.is_tensor(valid):
            raise RuntimeError("theta artifact row type drift")
        if theta.ndim != 1 or valid.ndim != 1 or theta.numel() != units or valid.numel() != units or aligned["theta_units"] != units:
            raise RuntimeError("theta artifact row size drift")
    return artifact


def dry_plan(root: Path) -> dict[str, object]:
    closures = verify_stage0_and_phase_c_closures(root)
    authorities = verify_canonical_source_authorities(root)
    return {
        "cell": CELL,
        "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH",
        "authorization": "none",
        "execution_flags_required_together": ["--execute", "--i-have-source-smoke-authorization"],
        "canonical_receipt_relative_path": CANONICAL_RECEIPT_RELATIVE_PATH,
        "canonical_receipt_must_be_absent_before_execution": True,
        "closures": closures,
        "canonical_source_authorities": authorities,
        "source_smoke_only": {"seed": 42, "batch_size": 32, "steps": 1, "epochs_authorized": 0,
                               "target_or_formal_opened": False, "target_optimizer_steps": 0,
                               "scientific_result": False, "score": False, "authorizes_48_epoch": False},
    }


def raw_tensor_bytes_sha256(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def normalized_t4_authority_sha256(tensor: Any) -> str:
    import torch
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.is_floating_point(): flat = flat + 0
    digest = hashlib.sha256(); digest.update(str(flat.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
    if flat.numel(): digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _ordered_ids(session: str, record: Any, n: int) -> tuple[str, ...]:
    channel_ids = getattr(record, "channel_ids", None)
    if channel_ids is None or len(channel_ids) != n:
        raise RuntimeError("source record lacks exact canonical channel/unit identifiers")
    return tuple(f"{session}:channel:{int(value)}" for value in channel_ids)


def validate_source_batch(batch: Any, record: Any, strict_roster: set[str]) -> dict[str, Any]:
    """Validate the established sampler batch without renormalizing its side tensor."""
    import torch
    if not isinstance(batch, (tuple, list)) or len(batch) < 5:
        raise RuntimeError("source sampler must yield neural, behavior, calib, sessions, side")
    neural, behavior, calib, sessions, side = batch[:5]
    names = list(sessions) if isinstance(sessions, (tuple, list)) else [sessions]
    if not names or len(set(names)) != 1:
        raise RuntimeError("SessionBatchSampler batch must contain exactly one session")
    session = names[0]
    if not isinstance(session, str) or session not in strict_roster:
        raise RuntimeError("batch session is not in exact strict-27 source roster")
    if not all(isinstance(value, torch.Tensor) for value in (neural, behavior, calib, side)):
        raise RuntimeError("source batch tensors are required")
    if neural.ndim != 3 or behavior.ndim != 3 or calib.ndim != 4 or side.ndim != 3:
        raise RuntimeError("source batch rank drift")
    b, t, n = neural.shape
    if b != 32 or t != 50 or behavior.shape != (b, 50, 2) or calib.shape != (b, 30, 100, n) or side.shape != (b, n, 4):
        raise RuntimeError("source batch shape drift")
    if not all(torch.isfinite(value).all().item() for value in (neural, behavior, calib, side)):
        raise RuntimeError("source batch nonfinite tensor")
    source_side = torch.as_tensor(record.side_features, dtype=side.dtype)
    if source_side.shape != (n, 4) or not torch.equal(side, source_side.unsqueeze(0).expand(b, -1, -1)):
        raise RuntimeError("side is not the exact already-normalized source-record T4")
    ids = _ordered_ids(session, record, n)
    return {"session": session, "units": n, "ordered_unit_ids": ids,
            "ordered_unit_digest": _digest_json(ids), "normalized_side_authority_sha256": normalized_t4_authority_sha256(side[0]),
            "side_was_renormalized": False}


def capability_from_verified_side(batch_info: Mapping[str, Any], side: Any, *, raw_authority_sha256: str, normalizer_authority_sha256: str, roster_digest: str, lineage: tuple[str, ...]):
    """Construct only a typed capability from the verified model-visible normalized side."""
    import torch
    from .model import NormalizedT4Batch
    if not isinstance(side, torch.Tensor) or side.ndim != 3 or not isinstance(batch_info.get("ordered_unit_ids"), tuple):
        raise RuntimeError("bare or malformed side tensor")
    if side.shape[1] != len(batch_info["ordered_unit_ids"]) or normalized_t4_authority_sha256(side[0]) != batch_info.get("normalized_side_authority_sha256"):
        raise RuntimeError("side does not match its verified source batch authority")
    return NormalizedT4Batch(side.detach().clone(), raw_authority_sha256, normalizer_authority_sha256,
                             roster_digest, batch_info["ordered_unit_digest"], tuple(batch_info["ordered_unit_ids"]), lineage)


def _receipt_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"


def pre_execution_output_gate(root: Path) -> Path:
    """Static, pre-import gate for the sole canonical execution output.

    This deliberately uses only ``os``/``stat``: an authorized CLI invocation
    must fail here before importing torch, resolving a source path, or opening
    an authority artifact when an old/colliding receipt exists.
    """
    path = root / CANONICAL_RECEIPT_RELATIVE_PATH
    parent = path.parent
    if parent.is_symlink():
        raise RuntimeError("canonical source-smoke receipt parent may not be a symlink")
    if parent.exists() and (not parent.is_dir() or stat.S_ISLNK(os.lstat(parent).st_mode)):
        raise RuntimeError("canonical source-smoke receipt parent is not a regular directory")
    for candidate in (path, Path(str(path) + ".sha256")):
        try:
            value = os.lstat(candidate)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(value.st_mode):
            raise RuntimeError("canonical source-smoke receipt link conflict")
        raise RuntimeError("fresh canonical source-smoke receipt pair required")
    return path


def _full_write(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short receipt write")
        view = view[written:]


def _read_fd_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _receipt_regular_from_dirfd(directory_fd: int, name: str) -> tuple[bytes, os.stat_result]:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o444:
            raise RuntimeError("published receipt is not a 0444 regular file")
        return _read_fd_all(fd), opened
    finally:
        os.close(fd)


def write_smoke_receipt_transactionally(path: Path, payload: Mapping[str, object]) -> str:
    """Publish a 0444 JSON+sidecar pair through one no-follow directory FD.

    The helper is intentionally useful with a temporary test path as well as
    the canonical output.  It never follows the final directory or either
    receipt name, uses ``O_EXCL`` relative to the opened directory, and only
    removes inodes it created if a later write/fsync/validation step fails.
    """
    parent, grandparent = path.parent, path.parent.parent
    if not parent.name or parent.is_symlink() or grandparent.is_symlink():
        raise RuntimeError("receipt parent/ancestor may not be a symlink")
    grand_lstat = os.lstat(grandparent)
    if not stat.S_ISDIR(grand_lstat.st_mode) or stat.S_ISLNK(grand_lstat.st_mode):
        raise RuntimeError("receipt parent ancestor is not a regular directory")
    grand_fd = os.open(grandparent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    directory_fd: int | None = None
    created_parent: tuple[int, int] | None = None
    created: list[tuple[str, int, int]] = []
    try:
        grand_opened = os.fstat(grand_fd)
        if (grand_opened.st_dev, grand_opened.st_ino) != (grand_lstat.st_dev, grand_lstat.st_ino):
            raise RuntimeError("receipt parent ancestor identity changed while opening")
        try:
            parent_lstat = os.stat(parent.name, dir_fd=grand_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir(parent.name, 0o755, dir_fd=grand_fd)
            os.fsync(grand_fd)
            parent_lstat = os.stat(parent.name, dir_fd=grand_fd, follow_symlinks=False)
            if not stat.S_ISDIR(parent_lstat.st_mode) or stat.S_ISLNK(parent_lstat.st_mode):
                raise RuntimeError("created receipt parent is not a regular directory")
            created_parent = (parent_lstat.st_dev, parent_lstat.st_ino)
        if not stat.S_ISDIR(parent_lstat.st_mode) or stat.S_ISLNK(parent_lstat.st_mode):
            raise RuntimeError("receipt parent is not a regular directory")
        directory_fd = os.open(parent.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=grand_fd)
        parent_opened = os.fstat(directory_fd)
        if (parent_opened.st_dev, parent_opened.st_ino) != (parent_lstat.st_dev, parent_lstat.st_ino):
            raise RuntimeError("receipt parent identity changed while opening")
        body_name, sidecar_name = path.name, path.name + ".sha256"
        for name in (body_name, sidecar_name):
            try:
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise RuntimeError("fresh non-symlink smoke receipt pair required")
        body = _receipt_bytes(payload)
        digest = hashlib.sha256(body).hexdigest()
        sidecar_body = f"{digest}  {body_name}\n".encode()
        for name, contents in ((body_name, body), (sidecar_name, sidecar_body)):
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
            try:
                opened = os.fstat(fd)
                if not stat.S_ISREG(opened.st_mode):
                    raise RuntimeError("receipt O_EXCL did not create a regular file")
                created.append((name, opened.st_dev, opened.st_ino))
                _full_write(fd, contents)
                os.fchmod(fd, 0o444)
                os.fsync(fd)
            finally:
                os.close(fd)
        os.fsync(directory_fd)
        # Same-FD reload is both the publication proof and a defense against
        # a path replacement between close and subsequent consumers.
        actual_body, _ = _receipt_regular_from_dirfd(directory_fd, body_name)
        actual_sidecar, _ = _receipt_regular_from_dirfd(directory_fd, sidecar_name)
        if hashlib.sha256(actual_body).hexdigest() != digest or actual_sidecar != sidecar_body:
            raise RuntimeError("post-write receipt body/sidecar drift")
        parsed = json.loads(actual_body)
        if not isinstance(parsed, dict):
            raise RuntimeError("post-write receipt JSON object drift")
        validate_smoke_receipt(parsed)
        parent_after = os.fstat(directory_fd)
        if (parent_after.st_dev, parent_after.st_ino) != (parent_opened.st_dev, parent_opened.st_ino):
            raise RuntimeError("receipt parent identity changed after publication")
        parent_named_after = os.stat(parent.name, dir_fd=grand_fd, follow_symlinks=False)
        if (parent_named_after.st_dev, parent_named_after.st_ino) != (parent_opened.st_dev, parent_opened.st_ino):
            raise RuntimeError("receipt parent was renamed/replaced during publication")
        os.fsync(grand_fd)
        grand_after = os.fstat(grand_fd)
        if (grand_after.st_dev, grand_after.st_ino) != (grand_opened.st_dev, grand_opened.st_ino):
            raise RuntimeError("receipt parent ancestor identity changed after publication")
        return digest
    except BaseException:
        if directory_fd is not None:
            for name, device, inode in reversed(created):
                try:
                    item = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if (item.st_dev, item.st_ino) == (device, inode):
                        os.unlink(name, dir_fd=directory_fd)
                except OSError:
                    pass
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        raise
    finally:
        try:
            if directory_fd is not None:
                os.close(directory_fd)
        finally:
            if created_parent is not None:
                try:
                    current = os.stat(parent.name, dir_fd=grand_fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == created_parent:
                        os.rmdir(parent.name, dir_fd=grand_fd)
                        os.fsync(grand_fd)
                except OSError:
                    pass
            os.close(grand_fd)


_SMOKE_RECEIPT_KEYS = {
    "schema", "status", "cell", "source_smoke", "source_data_opened", "target_data_opened",
    "validation_data_opened", "formal_data_opened", "target_or_formal_opened", "target_optimizer_steps",
    "scientific_result", "score", "authorizes_48_epoch", "environment", "device", "primary_batch",
    "optimizer", "capture_diagnostics", "loss", "critical_gradients", "wall_seconds", "rss_bytes",
    "peak_allocated_bytes", "peak_reserved_bytes", "initial_state_digest", "post_step_state_digest",
    "dropout", "launch_final_closure_equal", "launch_closure", "final_closure", "authorities",
    "live_source_audit",
}


def _validate_closure_payload(closure: Any) -> None:
    if not isinstance(closure, Mapping) or set(closure) != {"stage0", "phase_c", "authorities"}:
        raise RuntimeError("receipt closure/evidence map keys drift")
    for name, paths in (("stage0", IMPLEMENTATION_CLOSURE), ("phase_c", PHASE_C_CLOSURE)):
        entry = closure[name]
        if not isinstance(entry, Mapping) or set(entry) != {"paths", "sha256_by_path", "closure_sha256"}:
            raise RuntimeError("receipt closure entry drift")
        if entry["paths"] != list(paths) or not isinstance(entry["sha256_by_path"], Mapping) or set(entry["sha256_by_path"]) != set(paths):
            raise RuntimeError("receipt closure path map drift")
        if not _is_sha256(entry["closure_sha256"]) or not all(_is_sha256(value) for value in entry["sha256_by_path"].values()):
            raise RuntimeError("receipt closure SHA drift")
    if any(closure["stage0"]["sha256_by_path"].get(path) != expected for path, expected in EXPECTED_STAGE0_SHA.items()):
        raise RuntimeError("receipt frozen Stage-0 closure drift")


def _validate_receipt_authorities(authorities: Any) -> None:
    expected_keys = {"admission_preflight", "theta_receipt", "theta_artifact", "roster", "normalized_t4_sha256",
                     "raw_authority_sha256", "normalizer_authority_sha256", "side_semantic_sha256",
                     "behavior_semantic_sha256", "manifest_sha256", "source_lineage", "theta_raw_t4_sha256"}
    if not isinstance(authorities, Mapping) or set(authorities) != expected_keys:
        raise RuntimeError("receipt authority map keys drift")
    if (authorities["admission_preflight"] != {"path": ADMISSION_PREFLIGHT, "body_sha256": ADMISSION_SHA} or
            authorities["theta_receipt"] != {"path": THETA_RECEIPT, "body_sha256": THETA_RECEIPT_SHA} or
            authorities["theta_artifact"] != {"path": THETA_ARTIFACT, "body_sha256": THETA_ARTIFACT_SHA} or
            authorities["raw_authority_sha256"] != THETA_ARTIFACT_SHA or
            authorities["normalizer_authority_sha256"] != ADMISSION_SHA or
            authorities["side_semantic_sha256"] != SIDE_SEMANTIC_SHA or
            authorities["behavior_semantic_sha256"] != BEHAVIOR_SEMANTIC_SHA or authorities["manifest_sha256"] != MANIFEST_SHA):
        raise RuntimeError("receipt canonical authority binding drift")
    roster = authorities["roster"]
    normalized, raw = authorities["normalized_t4_sha256"], authorities["theta_raw_t4_sha256"]
    if (not isinstance(roster, list) or len(roster) != 27 or len(set(roster)) != 27 or not isinstance(normalized, Mapping) or
            not isinstance(raw, Mapping) or set(normalized) != set(roster) or set(raw) != set(roster) or
            not all(_is_sha256(value) for value in normalized.values()) or not all(_is_sha256(value) for value in raw.values())):
        raise RuntimeError("receipt all-27 T4 authority drift")
    lineage = authorities["source_lineage"]
    if (not isinstance(lineage, Mapping) or set(lineage) != {"path", "body_sha256", "matching_authority_consumed_bytes_sha256", "rows_by_session"} or
            lineage["path"] != SOURCE_LINEAGE or lineage["body_sha256"] != SOURCE_LINEAGE_SHA or
            lineage["matching_authority_consumed_bytes_sha256"] != MATCHING_AUTHORITY_CONSUMED_SHA or
            not isinstance(lineage["rows_by_session"], Mapping) or set(lineage["rows_by_session"]) != set(roster)):
        raise RuntimeError("receipt source-lineage authority drift")


def _validate_live_source_audit(audit: Any, authorities: Mapping[str, Any]) -> None:
    expected = {"roster", "roster_digest", "sessions", "manifest_sha256", "n_train_windows", "steps_per_epoch",
                "batch_size", "seed", "side_normalizer_semantic_sha256", "behavior_normalizer_semantic_sha256", "m30_t4_support"}
    if not isinstance(audit, Mapping) or set(audit) != expected or audit["roster"] != authorities["roster"]:
        raise RuntimeError("receipt live source audit roster drift")
    if (audit["roster_digest"] != _digest_json(tuple(authorities["roster"])) or audit["manifest_sha256"] != MANIFEST_SHA or
            audit["n_train_windows"] != 1086007 or audit["steps_per_epoch"] != 33925 or audit["batch_size"] != 32 or audit["seed"] != 42 or
            audit["side_normalizer_semantic_sha256"] != SIDE_SEMANTIC_SHA or audit["behavior_normalizer_semantic_sha256"] != BEHAVIOR_SEMANTIC_SHA or
            audit["m30_t4_support"] != "chronological_first_30_rewarded_trials" or not isinstance(audit["sessions"], list) or len(audit["sessions"]) != 27):
        raise RuntimeError("receipt live source audit semantic drift")
    expected_row_keys = {"session", "source_path", "source_size_bytes", "source_sha256", "unit_count", "unit_order_digest", "raw_t4_sha256", "normalized_t4_sha256"}
    for index, row in enumerate(audit["sessions"]):
        session = authorities["roster"][index]
        lineage = authorities["source_lineage"]["rows_by_session"][session]
        if (not isinstance(row, Mapping) or set(row) != expected_row_keys or row["session"] != session or
                row["source_path"] != lineage["path"] or row["source_size_bytes"] != lineage["size_bytes"] or
                row["source_sha256"] != lineage["sha256"] or row["unit_count"] != lineage["unit_count"] or
                row["raw_t4_sha256"] != authorities["theta_raw_t4_sha256"][session] or
                row["normalized_t4_sha256"] != authorities["normalized_t4_sha256"][session] or not _is_sha256(row["unit_order_digest"])):
            raise RuntimeError("receipt live source session authority drift")


def validate_smoke_receipt(payload: Mapping[str, Any]) -> None:
    """Exact schema validation for the one engineering-only source-smoke receipt."""
    if not isinstance(payload, Mapping) or set(payload) != _SMOKE_RECEIPT_KEYS:
        raise RuntimeError("smoke receipt exact top-level schema drift")
    boundary = {"schema": "tfsr_b3st4_ddrop_source_smoke_v1", "status": "ENGINEERING_SOURCE_ONLY_SMOKE", "cell": CELL,
                "source_data_opened": True, "target_data_opened": False, "validation_data_opened": False,
                "formal_data_opened": False, "target_or_formal_opened": False, "target_optimizer_steps": 0,
                "scientific_result": False, "score": False, "authorizes_48_epoch": False, "capture_diagnostics": False}
    if any(payload[key] != value for key, value in boundary.items()) or payload["source_smoke"] != FROZEN_SMOKE:
        raise RuntimeError("smoke receipt scientific/engineering boundary drift")
    if payload["device"] != FROZEN_DEVICE or payload["optimizer"] != FROZEN_OPTIMIZER:
        raise RuntimeError("smoke receipt device/optimizer drift")
    environment = payload["environment"]
    if not isinstance(environment, Mapping) or set(environment) != {"python_executable", "python", "torch", "cuda", "cudnn"} or not all(isinstance(value, str) and value for value in environment.values()):
        raise RuntimeError("smoke receipt environment drift")
    batch = payload["primary_batch"]
    if (not isinstance(batch, Mapping) or set(batch) != {"session", "units", "shapes", "unit_order_digest", "raw_t4_sha256", "normalized_t4_sha256"} or
            not isinstance(batch["session"], str) or type(batch["units"]) is not int or batch["units"] <= 0 or not _is_sha256(batch["unit_order_digest"]) or
            not _is_sha256(batch["raw_t4_sha256"]) or not _is_sha256(batch["normalized_t4_sha256"]) or not isinstance(batch["shapes"], Mapping) or
            batch["shapes"] != {"neural": [32, 50, batch["units"]], "behavior": [32, 50, 2], "calib": [32, 30, 100, batch["units"]], "side": [32, batch["units"], 4]}):
        raise RuntimeError("smoke receipt B32 batch drift")
    for key in ("loss", "wall_seconds"):
        if not isinstance(payload[key], (int, float)) or isinstance(payload[key], bool) or not math.isfinite(float(payload[key])) or float(payload[key]) <= 0:
            raise RuntimeError("smoke receipt nonpositive/nonfinite metric")
    for key in ("rss_bytes", "peak_allocated_bytes", "peak_reserved_bytes"):
        if type(payload[key]) is not int or payload[key] <= 0:
            raise RuntimeError("smoke receipt nonpositive memory metric")
    if not _is_sha256(payload["initial_state_digest"]) or not _is_sha256(payload["post_step_state_digest"]) or payload["initial_state_digest"] == payload["post_step_state_digest"]:
        raise RuntimeError("smoke receipt state-digest drift")
    gradients = payload["critical_gradients"]
    if not isinstance(gradients, Mapping) or set(gradients) != set(_CRITICAL_PARAMETER_PREFIXES) or any(value is not True for value in gradients.values()):
        raise RuntimeError("smoke receipt critical-gradient drift")
    dropout = payload["dropout"]
    if not isinstance(dropout, Mapping) or set(dropout) != {"p", "gain", "survivors"} or not isinstance(dropout["p"], (int, float)) or not math.isfinite(float(dropout["p"])) or not 0 <= float(dropout["p"]) <= 1:
        raise RuntimeError("smoke receipt dropout drift")
    if (not isinstance(dropout["gain"], list) or not isinstance(dropout["survivors"], list) or
            len(dropout["gain"]) != 32 or len(dropout["survivors"]) != 32 or
            any(not isinstance(gain_row, list) or not isinstance(survivor_row, list) or len(gain_row) != batch["units"] or len(survivor_row) != batch["units"]
                for gain_row, survivor_row in zip(dropout["gain"], dropout["survivors"], strict=True)) or
            any(not isinstance(gain, (int, float)) or isinstance(gain, bool) or not math.isfinite(float(gain)) or float(gain) < 0 or
                type(survivor) is not bool or survivor != (float(gain) != 0.0)
                for gain_row, survivor_row in zip(dropout["gain"], dropout["survivors"], strict=True) for gain, survivor in zip(gain_row, survivor_row, strict=True))):
        raise RuntimeError("smoke receipt dropout mask drift")
    if payload["launch_final_closure_equal"] is not True or payload["launch_closure"] != payload["final_closure"]:
        raise RuntimeError("smoke receipt launch/final closure drift")
    _validate_closure_payload(payload["launch_closure"])
    _validate_receipt_authorities(payload["authorities"])
    if payload["launch_closure"]["authorities"] != payload["authorities"]:
        raise RuntimeError("smoke receipt closure/evidence authority join drift")
    _validate_live_source_audit(payload["live_source_audit"], payload["authorities"])
    primary_rows = [row for row in payload["live_source_audit"]["sessions"] if row["session"] == batch["session"]]
    if len(primary_rows) != 1 or any(batch[key] != primary_rows[0][key] for key in ("unit_order_digest", "raw_t4_sha256", "normalized_t4_sha256")) or batch["units"] != primary_rows[0]["unit_count"]:
        raise RuntimeError("smoke receipt primary/all-27 authority join drift")


def require_launch_final_closure(launch: Mapping[str, Any], final: Mapping[str, Any]) -> None:
    if launch != final:
        raise RuntimeError("launch/final closure drift")


def require_single_visible_cuda(torch: Any) -> dict[str, object]:
    """Execution-only physical identity gate: one CVD entry maps to internal cuda:0."""
    import subprocess
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != "1" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("physical CVD policy requires exactly one visible GPU for internal cuda:0")
    line = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid,pci.bus_id,name,memory.total", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(line) != 1:
        raise RuntimeError("nvidia-smi/CVD visible-device count mismatch")
    uuid, bdf, name, memory = [part.strip() for part in line[0].split(",", 3)]
    if uuid != "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86" or bdf != "00000000:03:00.0" or name != "NVIDIA GeForce RTX 3090" or int(memory) != 24576:
        raise RuntimeError("physical GPU UUID/BDF identity mismatch")
    return dict(FROZEN_DEVICE)


_CRITICAL_PARAMETER_PREFIXES = {
    "b3s": "b3s.", "activity_encoder": "activity_encoder.", "unit_mlp": "unit_mlp.",
    "query_base": "query_base", "state_query": "state_query.", "attn": "attn.",
    "norm1": "norm1.", "norm2": "norm2.", "ffn": "ffn.", "gru": "gru.", "head": "head.",
}


def require_critical_gradients(model: Any, torch: Any) -> dict[str, bool]:
    """Require a finite, nonzero gradient in every explicitly named block."""
    named = tuple(model.named_parameters())
    results: dict[str, bool] = {}
    for group, prefix in _CRITICAL_PARAMETER_PREFIXES.items():
        matching = [(name, value) for name, value in named if name.startswith(prefix)]
        results[group] = bool(matching) and any(
            value.grad is not None and torch.isfinite(value.grad).all().item() and value.grad.abs().sum().item() > 0
            for _, value in matching
        )
    if not all(results.values()):
        raise RuntimeError("critical gradient missing/nonfinite/zero")
    return results


def _live_source_audit(adapter: Any, authorities: Mapping[str, Any]) -> dict[str, object]:
    """Compact all-27 proof carried by the engineering receipt, never arrays."""
    import torch
    roster = authorities["roster"]
    rows: list[dict[str, object]] = []
    for session in roster:
        lineage = authorities["source_lineage"]["rows_by_session"][session]
        record = adapter.records[session]
        unit_count = lineage["unit_count"]
        ids = _ordered_ids(session, record, unit_count)
        rows.append({"session": session, "source_path": lineage["path"], "source_size_bytes": lineage["size_bytes"],
                     "source_sha256": lineage["sha256"], "unit_count": unit_count,
                     "unit_order_digest": _digest_json(ids), "raw_t4_sha256": adapter.raw_t4_sha256[session],
                     "normalized_t4_sha256": normalized_t4_authority_sha256(torch.as_tensor(record.side_features))})
    return {"roster": list(roster), "roster_digest": _digest_json(tuple(roster)), "sessions": rows,
            "manifest_sha256": MANIFEST_SHA, "n_train_windows": 1086007, "steps_per_epoch": 33925,
            "batch_size": 32, "seed": 42, "side_normalizer_semantic_sha256": SIDE_SEMANTIC_SHA,
            "behavior_normalizer_semantic_sha256": BEHAVIOR_SEMANTIC_SHA,
            "m30_t4_support": "chronological_first_30_rewarded_trials"}


def execute_one_step(adapter: Any, root: Path) -> dict[str, object]:
    """The authorized B32 engineering step; adapter must expose train-only records and first batch."""
    # This must remain before every execution-only import.  The public CLI
    # calls the same gate before constructing the adapter; keeping it here
    # protects direct callers as well.
    pre_execution_output_gate(root)
    import resource, random, sys
    import time
    import numpy as np
    import torch
    from .model import TFSRDecoder
    authorities = verify_canonical_source_authorities(root)
    theta = adapter.theta_authority
    receipt_authorities = dict(authorities)
    receipt_authorities["theta_raw_t4_sha256"] = {session: theta["authority"][session]["raw_t4_sha256"] for session in authorities["roster"]}
    launch = {**verify_stage0_and_phase_c_closures(root), "authorities": receipt_authorities}
    device = require_single_visible_cuda(torch)
    records, first_batch = adapter.records, adapter.first_seed42_batch()
    if set(records) != set(authorities["roster"]): raise RuntimeError("live train-only adapter roster drift")
    for session in authorities["roster"]:
        record = records[session]
        side = torch.as_tensor(record.side_features)
        if normalized_t4_authority_sha256(side) != authorities["normalized_t4_sha256"][session] or side.shape[0] != theta["authority"][session]["n_units"] or adapter.raw_t4_sha256[session] != theta["authority"][session]["raw_t4_sha256"]:
            raise RuntimeError("live normalized/raw authority session join drift")
    info = validate_source_batch(first_batch, records[first_batch[3][0]], set(authorities["roster"]))
    side = first_batch[4].to("cuda:0")
    cap = capability_from_verified_side(info, side, raw_authority_sha256=THETA_ARTIFACT_SHA, normalizer_authority_sha256=ADMISSION_SHA,
                                        roster_digest=_digest_json(tuple(authorities["roster"])), lineage=(ADMISSION_PREFLIGHT, THETA_RECEIPT, THETA_ARTIFACT, info["session"], info["normalized_side_authority_sha256"], theta["authority"][info["session"]]["raw_t4_sha256"]))
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model = TFSRDecoder(capture_diagnostics=False).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN_OPTIMIZER["lr"], betas=tuple(FROZEN_OPTIMIZER["betas"]),
                                  eps=FROZEN_OPTIMIZER["eps"], weight_decay=FROZEN_OPTIMIZER["weight_decay"], amsgrad=False)
    initial = _digest_json({name: normalized_t4_authority_sha256(value) for name, value in model.state_dict().items() if torch.is_tensor(value)})
    neural, behavior, calib = (first_batch[0].to("cuda:0"), first_batch[1].to("cuda:0"), first_batch[2].to("cuda:0"))
    torch.cuda.reset_peak_memory_stats(0); torch.cuda.synchronize(0); start = time.perf_counter(); prediction = model(neural, calib, cap)
    valid = (behavior != -1.0).all(dim=-1); loss = model.dense_valid_bin_mse(prediction, behavior, valid)
    if not torch.isfinite(loss): raise RuntimeError("nonfinite source-smoke loss")
    optimizer.zero_grad(set_to_none=True); loss.backward()
    gradients = require_critical_gradients(model, torch)
    optimizer.step(); torch.cuda.synchronize(0); elapsed = time.perf_counter() - start
    final = {**verify_stage0_and_phase_c_closures(root), "authorities": receipt_authorities}; require_launch_final_closure(launch, final)
    # A completion between the initial gate and publication is a hard stop:
    # this is deliberately after the final-code closure, but before returning
    # any payload that a caller could publish.
    pre_execution_output_gate(root)
    audit = _live_source_audit(adapter, receipt_authorities)
    return {"schema": "tfsr_b3st4_ddrop_source_smoke_v1", "status": "ENGINEERING_SOURCE_ONLY_SMOKE", "cell": CELL,
            "source_smoke": dict(FROZEN_SMOKE), "source_data_opened": True, "target_data_opened": False,
            "validation_data_opened": False, "formal_data_opened": False, "target_or_formal_opened": False,
            "target_optimizer_steps": 0, "scientific_result": False, "score": False, "authorizes_48_epoch": False,
            "environment": {"python_executable": str(sys.executable), "python": str(sys.version), "torch": str(torch.__version__),
                            "cuda": str(torch.version.cuda), "cudnn": str(torch.backends.cudnn.version())}, "device": device,
            "primary_batch": {"session": info["session"], "units": info["units"],
                              "shapes": {"neural": list(neural.shape), "behavior": list(behavior.shape), "calib": list(calib.shape), "side": list(side.shape)},
                              "unit_order_digest": info["ordered_unit_digest"], "raw_t4_sha256": theta["authority"][info["session"]]["raw_t4_sha256"],
                              "normalized_t4_sha256": info["normalized_side_authority_sha256"]},
            "optimizer": dict(FROZEN_OPTIMIZER), "capture_diagnostics": False, "loss": float(loss.item()), "critical_gradients": gradients,
            "wall_seconds": float(elapsed), "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "initial_state_digest": initial, "post_step_state_digest": _digest_json({name: normalized_t4_authority_sha256(value) for name, value in model.state_dict().items() if torch.is_tensor(value)}),
            "dropout": {"p": float(model.last_dropout_p.item()), "gain": model.last_unit_gain_mask.cpu().tolist(), "survivors": model.last_unit_survivor_mask.cpu().tolist()},
            "launch_final_closure_equal": True, "launch_closure": launch, "final_closure": final,
            "authorities": receipt_authorities, "live_source_audit": audit}


class _TrainOnlyAdapter:
    def __init__(self, dataset: Any, sampler: Any, raw_t4_sha256: Mapping[str, str], theta_authority: Mapping[str, Any]):
        self.records, self._dataset, self._sampler = dataset.sessions, dataset, sampler
        self.raw_t4_sha256, self.theta_authority = dict(raw_t4_sha256), dict(theta_authority)
    def first_seed42_batch(self):
        import torch
        batch = next(iter(self._sampler))
        rows = [self._dataset[index] for index in batch]
        return torch.utils.data.default_collate(rows)


def build_train_only_adapter(root: Path, authorities: Mapping[str, Any]) -> _TrainOnlyAdapter:
    """Actual 27-train-only adapter; it never calls the legacy split initializer or resolves val/test paths."""
    import sys
    sys.path.insert(0, str(root / "sua_exploration"))
    from mc_maze import a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule, SessionBatchSampler
    from .contract import _canonical_regular_bytes
    manifest_bytes = _canonical_regular_bytes(root / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json")
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA: raise RuntimeError("same-FD manifest SHA drift")
    manifest = json.loads(manifest_bytes)
    roster = authorities["roster"]
    if manifest.get("session_splits", {}).get("train") != roster:
        raise RuntimeError("same-FD manifest strict-27 roster/order drift")
    theta_authority = load_verified_theta_artifact(root, authorities)
    data_root = Path(a2.SUBC_DATA_ROOT)
    root_stat = os.lstat(data_root)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise RuntimeError("source data root must be a canonical non-symlink directory")
    train_files = []
    for name in roster:
        candidate = data_root / f"{name}_behavior+ecephys.nwb"
        candidate_stat = os.lstat(candidate)
        if (candidate.parent != data_root or not stat.S_ISREG(candidate_stat.st_mode) or
                stat.S_ISLNK(candidate_stat.st_mode)):
            raise RuntimeError("exact train-only NWB path drift")
        verify_source_lineage_file(candidate, authorities["source_lineage"]["rows_by_session"][name])
        train_files.append(candidate)
    dm = Dandi688MultiSessionDataModule(data_dir=str(data_root), task="CO", split_counts=(27, 6, 6), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20, num_workers=0, random_calibration=False,
        seed=42, max_units_exclusive=100, cache_dir=str(a2.SOURCE_CACHE_ROOT), signal_view="sua", side_feature_group="t4",
        side_feature_pool_size=30, train_val_manifest_path=str(a2.MANIFEST_PATH))
    dm.session_files = {"train": train_files, "val": [], "test": []}; dm.session_splits = {"train": list(roster), "val": [], "test": []}; dm._splits_initialized = True
    dm.setup("fit")
    if dm.session_files["val"] or dm.session_files["test"] or dm.test_dataset is not None or dm.val_dataset is None or len(dm.val_dataset) != 0 or len(dm.train_dataset) != 1086007:
        raise RuntimeError("train-only adapter val/test/window drift")
    if a2.normalizer_value_sha256(*dm._side_feature_stats) != SIDE_SEMANTIC_SHA or a2.normalizer_value_sha256(*dm._behavior_stats) != BEHAVIOR_SEMANTIC_SHA:
        raise RuntimeError("live normalizer semantic authority drift")
    sampler = SessionBatchSampler(dm.train_dataset, batch_size=32, shuffle=True, seed=42)
    if len(sampler) != 33925: raise RuntimeError("train-only sampler step drift")
    for name in roster:
        record = dm.train_dataset.sessions.get(name)
        lineage = authorities["source_lineage"]["rows_by_session"][name]
        theta_row = theta_authority["authority"][name]
        if (record is None or record.name != name or record.neural.shape[1] != lineage["unit_count"] or
                record.source_unit_count != lineage["unit_count"] or len(record.channel_ids) != lineage["unit_count"] or
                theta_row["n_units"] != lineage["unit_count"]):
            raise RuntimeError("live source record/unit authority drift")
    from mc_maze.unit_side_features import compute_unit_side_features_uncached
    raw_hashes = {}
    for name, path in zip(roster, train_files, strict=True):
        raw, _ = recompute_verified_raw_t4(path, authorities["source_lineage"]["rows_by_session"][name], compute_unit_side_features_uncached)
        raw_hashes[name] = hashlib.sha256(raw.astype("float32", copy=False).tobytes()).hexdigest()
        if raw_hashes[name] != theta_authority["authority"][name]["raw_t4_sha256"]:
            raise RuntimeError("recomputed raw T4 authority drift")
    return _TrainOnlyAdapter(dm.train_dataset, sampler, raw_hashes, theta_authority)
