"""Append-only v2 guardrails for the DANDI 000688 sub-M CO scorer.

This module intentionally does *not* score an external session in its current
state.  It records the corrections required by adversarial review, seals the
prelaunch bundle that a later authorization must bind, and supplies the
verification half of the future score/aggregate protocol.  In particular, it
does not load a checkpoint, open an NWB file, import a model, make a forward
pass, fit a normalizer, or use a GPU in a dry run or while writing prelaunch
evidence.

The predecessor is deliberately imported only for its already-frozen static
authority chain and endpoint constants.  Its source and prelaunch artifacts
are pinned below and are never written by this module.
"""
from __future__ import annotations

import ast
import base64
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from mc_maze import subm_co_score_only as v1


REPO_ROOT = Path(__file__).resolve().parents[2]

SCOPE_ID = v1.SCOPE_ID
SEEDS = v1.SEEDS
ARMS = v1.ARMS
VIEWS = v1.VIEWS
R2_MULTI_OUTPUT = v1.R2_MULTI_OUTPUT
EXPECTED_ELIGIBLE_SESSION_COUNT = v1.EXPECTED_ELIGIBLE_SESSION_COUNT
EXPECTED_SEALED_METRIC_COUNT = v1.EXPECTED_SEALED_METRIC_COUNT

ScoreOnlyContractError = v1.ScoreOnlyContractError
AuthorizationError = v1.AuthorizationError
RuntimeFenceError = v1.RuntimeFenceError
FilePin = v1.FilePin
FrozenSession = v1.FrozenSession
NormalizerPins = v1.NormalizerPins
ScoreContract = v1.ScoreContract
RuntimeAuditCounters = v1.RuntimeAuditCounters


V2_SCHEMA_VERSION = 2
V2_PRELAUNCH_KIND = "dandi_000688_subm_co_score_only_prelaunch_authorization_draft_v2"
V2_PRELAUNCH_RECEIPT_KIND = "dandi_000688_subm_co_score_only_prelaunch_receipt_v2"
V2_PRELAUNCH_SEAL_KIND = "dandi_000688_subm_co_score_only_prelaunch_artifact_seal_v2"
V2_AUTHORIZATION_KIND = "dandi_000688_subm_co_external_score_only_authorization_v2"
V2_SEAL_KIND = "dandi_000688_subm_co_seal_manifest_v2"
V2_METRIC_KIND = "dandi_000688_subm_co_sealed_session_prediction_metric_v2"

# This tolerance is part of the sealed protocol.  The aggregate independently
# recomputes the TorchMetrics score from the persisted arrays and accepts only
# this bounded serialization/implementation-level difference.  ``1e-6`` is
# tight relative to R2 while allowing the frozen scoring adapter's batched
# float32 TorchMetrics state updates to be checked against the aggregate's one
# CPU update over the exact persisted arrays.
R2_RECOMPUTE_ATOL = 1.0e-6

# An exact Ed25519 public-key file pin is intentionally absent.  This is not a
# string placeholder: until root adds a real public key and SHA-256 pin in a
# later append-only revision, v2 cannot authorize any external score action.
ROOT_ED25519_PUBLIC_KEY_PIN: FilePin | None = None

# Likewise, root must pin a receipt from an already-consumed sub-C development
# session proving the scorer adapter's parity.  This task neither opens sub-C
# nor manufactures that receipt.
SCORER_ADAPTER_PARITY_RECEIPT_PIN: FilePin | None = None
SCORER_ADAPTER_PARITY_RECEIPT_SCHEMA = "dandi_000688_subc_scorer_adapter_parity_receipt_v1"


# These pins make accidental edits to the superseded implementation fail v2
# validation.  They cover precisely the files and artifacts called out by the
# review and are intentionally kept separate from the v2 source snapshot.
V1_IMMUTABLE_PINS: dict[str, FilePin] = {
    "core": FilePin(
        "sua_exploration/mc_maze/subm_co_score_only.py",
        "8e6aa7b9efdeb894dd0f04a06aae0b01226aafb9ea3a6c493b9cb571648a9be4",
    ),
    "entry": FilePin(
        "sua_exploration/scripts/run_dandi688_subm_co_score_only.py",
        "cb417aecd9dd9b9550206df7531c9503fab1f50be9203b2b2abe603db62fd2c9",
    ),
    "prelaunch_writer": FilePin(
        "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch.py",
        "f67fc6067361d4f17876062caa2db1fea5f43451ffd01c1ef91dc5f087547277",
    ),
    "tests": FilePin(
        "sua_exploration/tests/test_dandi688_subm_co_score_only.py",
        "7c537ca328fc58d59aa05ac0b2504ce7305e9b15b835ec3e1c1dc941e102754e",
    ),
    "prelaunch_draft": FilePin(
        "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v1/prelaunch_authorization_draft.json",
        "084484dab27de9b795e0841d5d6eab96483fbd632439bfe5ebb63e7c52782169",
    ),
    "prelaunch_receipt": FilePin(
        "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v1/receipt.json",
        "6a42723245b480507bd3b6d7fc06a942e0e5fbddd8ac93e8688b4129ac4ea95a",
    ),
}

V2_SOURCE_RELATIVE_PATHS = (
    "sua_exploration/mc_maze/subm_co_score_only_v2.py",
    "sua_exploration/scripts/run_dandi688_subm_co_score_only_v2.py",
    "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch_v2.py",
    "sua_exploration/tests/test_dandi688_subm_co_score_only_v2.py",
)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreOnlyContractError(message)


def _authorization_need(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationError(message)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreOnlyContractError(f"cannot parse {label}: {path}") from exc
    _need(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _safe_relative_path(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    _need(not candidate.is_absolute() and ".." not in candidate.parts, f"unsafe {label} relative path: {relative}")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ScoreOnlyContractError(f"{label} escapes output root: {relative}") from exc
    return path


def _is_immutable_mode(path: Path) -> bool:
    return stat.S_IMODE(path.stat().st_mode) == 0o444


def _write_immutable_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    """Write, fsync, then make a JSON artifact read-only (0444)."""

    encoded = _canonical_json(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    _need(_is_immutable_mode(path), f"immutable JSON mode was not applied: {path}")
    return hashlib.sha256(encoded).hexdigest()


def _write_immutable_npz_exclusive(path: Path, *, predictions: np.ndarray, targets: np.ndarray) -> str:
    """Write, fsync, then make a prediction/target NPZ read-only (0444)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        np.savez_compressed(handle, predictions=predictions, targets=targets)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    _need(_is_immutable_mode(path), f"immutable NPZ mode was not applied: {path}")
    return sha256_file(path)


def verify_v1_immutable(root: Path = REPO_ROOT) -> dict[str, str]:
    """Verify every v1 source and stored prelaunch artifact byte-for-byte."""

    root = root.resolve()
    verified: dict[str, str] = {}
    for label, pin in V1_IMMUTABLE_PINS.items():
        path = (root / pin.relative_path).resolve()
        _need(path.is_file() and not path.is_symlink(), f"v1 immutable pin missing: {pin.relative_path}")
        observed = sha256_file(path)
        _need(observed == pin.sha256, f"v1 immutable pin drift: {label}")
        verified[label] = observed
    return verified


def v2_source_snapshot(root: Path = REPO_ROOT) -> dict[str, str]:
    """Return the complete four-file v2 source snapshot used for authorization."""

    root = root.resolve()
    snapshot: dict[str, str] = {}
    for relative in V2_SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve()
        _need(path.is_file() and not path.is_symlink(), f"v2 source missing: {relative}")
        snapshot[relative] = sha256_file(path)
    return snapshot


def validate_authority_chain_v2(
    root: Path = REPO_ROOT,
    *,
    counters: RuntimeAuditCounters | None = None,
) -> ScoreContract:
    """Check the original scientific authority and the immutable v1 baseline.

    This intentionally delegates only metadata/source validation to v1.  That
    validation does not open checkpoints, normalizers, or target NWB assets.
    """

    verify_v1_immutable(root)
    return v1.validate_authority_chain(root, counters=counters)


def expected_query_window_counts_from_preflight(
    contract: ScoreContract,
    root: Path = REPO_ROOT,
) -> dict[str, int]:
    """Derive the exact per-asset query count from the pinned v2 ledger.

    The ledger count is view-invariant: the frozen preflight established the
    valid chronological query starts before either signal view, arm, or seed is
    selected.  A score cell therefore has no discretion to drop windows.
    """

    root = root.resolve()
    preflight_pin = v1.AUTHORITY_PINS["schema_preflight_receipt_v2"]
    receipt_path = v1.verify_file_pin(root, preflight_pin)
    receipt = _load_json(receipt_path, "pinned v2 schema-preflight receipt")
    ledger = receipt.get("asset_disposition_ledger")
    _need(isinstance(ledger, list), "pinned v2 preflight ledger is missing")
    by_asset: dict[str, Mapping[str, Any]] = {}
    for row in ledger:
        _need(isinstance(row, Mapping) and isinstance(row.get("asset_id"), str), "pinned v2 preflight ledger row malformed")
        asset_id = row["asset_id"]
        _need(asset_id not in by_asset, f"pinned v2 preflight ledger duplicate asset ID: {asset_id}")
        by_asset[asset_id] = row
    expected: dict[str, int] = {}
    for session in contract.frozen_sessions:
        row = by_asset.get(session.asset_id)
        _need(row is not None, f"frozen asset missing from pinned v2 preflight ledger: {session.asset_id}")
        _need(row.get("eligible") is True, f"frozen asset is no longer eligible in pinned v2 preflight ledger: {session.asset_id}")
        _need(row.get("session_id") == session.session_id, f"pinned v2 ledger session ID drift: {session.asset_id}")
        _need(row.get("frozen_path") == session.frozen_path, f"pinned v2 ledger frozen path drift: {session.asset_id}")
        feasibility = row.get("datamodule_feasibility")
        _need(isinstance(feasibility, Mapping), f"pinned v2 ledger feasibility block missing: {session.asset_id}")
        count = feasibility.get("complete_query_window_count_strictly_after_first_50")
        _need(isinstance(count, int) and count > 0, f"pinned v2 ledger query-window count malformed: {session.asset_id}")
        expected[session.asset_id] = count
    _need(len(expected) == EXPECTED_ELIGIBLE_SESSION_COUNT, "pinned v2 ledger frozen query-count coverage drift")
    _need(sum(expected.values()) == 708_795, "pinned v2 ledger N=15 query-window total drift")
    return expected


@dataclass(frozen=True)
class ExecutionDeviceBinding:
    """The single execution device that must cover the entire 180-cell matrix."""

    torch_device: str
    kind: str
    matrix_torch_device: str
    host: str | None
    physical_gpu_uuid: str | None
    physical_gpu_pci_bus_id: str | None
    cuda_visible_devices: str | None
    all_180_cells_same_device: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_requested_device(device: str) -> str:
    _authorization_need(isinstance(device, str), "execution device must be a string")
    normalized = device.strip().lower()
    _authorization_need(normalized == "cpu" or re.fullmatch(r"cuda:[0-9]+", normalized) is not None, "device must be cpu or cuda:<nonnegative-index>")
    return normalized


def normalize_physical_gpu_uuid(value: str) -> str:
    """Canonicalize Torch's bare UUID and nvidia-smi's ``GPU-`` UUID alike."""

    _authorization_need(isinstance(value, str) and value == value.strip() and bool(value), "physical GPU UUID must be a nonempty exact string")
    normalized = value if value.startswith("GPU-") else f"GPU-{value}"
    _authorization_need(re.fullmatch(r"GPU-[A-Za-z0-9._-]+", normalized) is not None, "physical GPU UUID malformed")
    return normalized


def _normalize_pci_bus_id(value: str) -> str:
    _authorization_need(isinstance(value, str) and value == value.strip(), "physical GPU PCI bus ID must be an exact string")
    normalized = value.lower()
    _authorization_need(re.fullmatch(r"[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", normalized) is not None, "physical GPU PCI bus ID malformed")
    return normalized


def execution_device_binding_from_mapping(value: Mapping[str, Any]) -> ExecutionDeviceBinding:
    """Validate the exact device-binding schema without touching CUDA."""

    _authorization_need(isinstance(value, Mapping), "authorization execution_device binding is missing")
    expected_keys = {
        "torch_device",
        "kind",
        "matrix_torch_device",
        "host",
        "physical_gpu_uuid",
        "physical_gpu_pci_bus_id",
        "cuda_visible_devices",
        "all_180_cells_same_device",
    }
    _authorization_need(set(value) == expected_keys, "execution_device binding key set drift")
    torch_device = _normalize_requested_device(value.get("torch_device"))
    kind = value.get("kind")
    matrix_torch_device = value.get("matrix_torch_device")
    host = value.get("host")
    physical_gpu_uuid = value.get("physical_gpu_uuid")
    physical_gpu_pci_bus_id = value.get("physical_gpu_pci_bus_id")
    cuda_visible_devices = value.get("cuda_visible_devices")
    all_same = value.get("all_180_cells_same_device")
    _authorization_need(kind in {"cpu", "cuda"}, "execution_device kind must be cpu or cuda")
    _authorization_need(matrix_torch_device == torch_device, "matrix_torch_device must exactly equal torch_device")
    _authorization_need(all_same is True, "all 180 cells must be bound to one execution device")
    if torch_device == "cpu":
        _authorization_need(kind == "cpu", "CPU device requires kind=cpu")
        _authorization_need(host is None and physical_gpu_uuid is None and physical_gpu_pci_bus_id is None and cuda_visible_devices is None, "CPU binding cannot carry CUDA identity fields")
    else:
        _authorization_need(kind == "cuda", "CUDA device requires kind=cuda")
        _authorization_need(isinstance(host, str) and bool(host.strip()), "CUDA binding requires exact host")
        _authorization_need(isinstance(physical_gpu_uuid, str), "CUDA binding requires a physical GPU UUID")
        physical_gpu_uuid = normalize_physical_gpu_uuid(physical_gpu_uuid)
        _authorization_need(isinstance(physical_gpu_pci_bus_id, str), "CUDA binding requires a physical GPU PCI bus ID")
        physical_gpu_pci_bus_id = _normalize_pci_bus_id(physical_gpu_pci_bus_id)
        _authorization_need(cuda_visible_devices is None or (isinstance(cuda_visible_devices, str) and cuda_visible_devices.strip() == cuda_visible_devices), "CUDA_VISIBLE_DEVICES binding must be null or an exact string")
    return ExecutionDeviceBinding(
        torch_device=torch_device,
        kind=str(kind),
        matrix_torch_device=str(matrix_torch_device),
        host=None if host is None else str(host),
        physical_gpu_uuid=None if physical_gpu_uuid is None else str(physical_gpu_uuid),
        physical_gpu_pci_bus_id=None if physical_gpu_pci_bus_id is None else str(physical_gpu_pci_bus_id),
        cuda_visible_devices=None if cuda_visible_devices is None else str(cuda_visible_devices),
        all_180_cells_same_device=True,
    )


def validate_execution_device_binding(
    binding: Mapping[str, Any],
    *,
    requested_device: str,
    runtime_identity: Mapping[str, str] | None = None,
) -> ExecutionDeviceBinding:
    """Compare an authorization device binding to a CLI request/runtime identity.

    ``runtime_identity`` exists so the schema can be tested without a GPU.  A
    later authorized CUDA adapter must obtain it from the actual selected CUDA
    device before it opens a checkpoint or target asset.
    """

    parsed = execution_device_binding_from_mapping(binding)
    requested = _normalize_requested_device(requested_device)
    _authorization_need(parsed.torch_device == requested, "requested --device differs from authorization execution_device")
    if runtime_identity is not None:
        _authorization_need(isinstance(runtime_identity, Mapping), "runtime device identity must be a mapping")
        _authorization_need(runtime_identity.get("torch_device") == parsed.torch_device, "runtime torch device string differs from authorization")
        if parsed.kind == "cuda":
            _authorization_need(runtime_identity.get("host") == parsed.host, "runtime CUDA host differs from authorization")
            runtime_uuid = runtime_identity.get("physical_gpu_uuid")
            _authorization_need(isinstance(runtime_uuid, str) and normalize_physical_gpu_uuid(runtime_uuid) == parsed.physical_gpu_uuid, "runtime CUDA physical GPU UUID differs from authorization")
            runtime_pci = runtime_identity.get("physical_gpu_pci_bus_id")
            _authorization_need(isinstance(runtime_pci, str) and _normalize_pci_bus_id(runtime_pci) == parsed.physical_gpu_pci_bus_id, "runtime CUDA physical GPU PCI bus ID differs from authorization")
            _authorization_need(runtime_identity.get("cuda_visible_devices") == parsed.cuda_visible_devices, "runtime CUDA_VISIBLE_DEVICES differs from authorization")
    return parsed


def _physical_identity_from_nvidia_smi(
    logical_torch_index: int,
    *,
    preferred_uuid: str | None = None,
    preferred_pci: str | None = None,
) -> tuple[str, str]:
    """Resolve/cross-check physical GPU identity with a fixed nvidia-smi query.

    No shell is used and all query arguments are literals.  When Torch exposes
    UUID/PCI properties, they are the preferred mapping and nvidia-smi merely
    corroborates them.  The fallback supports the ordinary unset or numeric
    ``CUDA_VISIBLE_DEVICES`` cases; ambiguous CUDA remapping fails closed.
    """

    executable = Path("/usr/bin/nvidia-smi")
    _authorization_need(executable.is_file() and os.access(executable, os.X_OK), "controlled nvidia-smi executable is unavailable")
    try:
        completed = subprocess.run(
            [str(executable), "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuthorizationError("controlled nvidia-smi identity query failed") from exc
    rows: list[tuple[int, str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        _authorization_need(len(fields) == 3 and fields[0].isdigit(), "controlled nvidia-smi identity output malformed")
        rows.append((int(fields[0]), normalize_physical_gpu_uuid(fields[1]), _normalize_pci_bus_id(fields[2])))
    _authorization_need(rows, "controlled nvidia-smi returned no physical GPUs")
    if preferred_uuid is not None or preferred_pci is not None:
        _authorization_need(preferred_uuid is not None and preferred_pci is not None, "Torch physical GPU identity is incomplete")
        matches = [row for row in rows if row[1] == preferred_uuid and row[2] == preferred_pci]
        _authorization_need(len(matches) == 1, "controlled nvidia-smi cannot corroborate selected Torch physical GPU identity")
        return matches[0][1], matches[0][2]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        matches = [row for row in rows if row[0] == logical_torch_index]
    else:
        tokens = visible.split(",")
        _authorization_need(0 <= logical_torch_index < len(tokens), "CUDA_VISIBLE_DEVICES cannot map the selected logical Torch device")
        token = tokens[logical_torch_index].strip()
        if token.isdigit():
            matches = [row for row in rows if row[0] == int(token)]
        elif token:
            normalized = normalize_physical_gpu_uuid(token)
            matches = [row for row in rows if row[1] == normalized]
        else:
            matches = []
    _authorization_need(len(matches) == 1, "controlled nvidia-smi cannot map selected logical Torch device to one physical GPU")
    return matches[0][1], matches[0][2]


def collect_runtime_execution_identity(binding: ExecutionDeviceBinding) -> dict[str, str | None]:
    """Read the selected future runtime identity before any checkpoint/NWB access.

    This helper is unreachable in the delivered v2 score path because v2 is
    non-authorizable.  When a later root key/parity pin is added, score mode
    invokes it before a checkpoint or NWB path is inspected.  CUDA UUID comes
    from the physical device properties, while ``torch_device`` deliberately
    remains the logical PyTorch string such as ``cuda:0``.
    """

    torch = importlib.import_module("torch")
    if binding.kind == "cpu":
        return {
            "torch_device": "cpu",
            "host": None,
            "physical_gpu_uuid": None,
            "physical_gpu_pci_bus_id": None,
            "cuda_visible_devices": None,
        }
    _authorization_need(torch.cuda.is_available(), "authorized CUDA device is unavailable at runtime")
    index = int(binding.torch_device.split(":", 1)[1])
    _authorization_need(index < int(torch.cuda.device_count()), "authorized CUDA device index is unavailable at runtime")
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    pci_bus_id = getattr(properties, "pci_bus_id", None)
    if isinstance(uuid, bytes):
        uuid = uuid.decode("ascii")
    if isinstance(pci_bus_id, bytes):
        pci_bus_id = pci_bus_id.decode("ascii")
    # Torch 2.5 exposes a bare UUID (without ``GPU-``), whereas nvidia-smi
    # uses the prefixed spelling.  Normalize both and cross-check the selected
    # logical torch index's physical UUID/PCI identity against nvidia-smi.
    if not isinstance(uuid, str) or not isinstance(pci_bus_id, str):
        uuid, pci_bus_id = _physical_identity_from_nvidia_smi(index)
    normalized_uuid = normalize_physical_gpu_uuid(uuid)
    normalized_pci = _normalize_pci_bus_id(pci_bus_id)
    smi_uuid, smi_pci = _physical_identity_from_nvidia_smi(index, preferred_uuid=normalized_uuid, preferred_pci=normalized_pci)
    _authorization_need(smi_uuid == normalized_uuid and smi_pci == normalized_pci, "nvidia-smi physical GPU identity differs from selected Torch device")
    return {
        "torch_device": binding.torch_device,
        "host": socket.gethostname(),
        "physical_gpu_uuid": normalized_uuid,
        "physical_gpu_pci_bus_id": normalized_pci,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def validate_runtime_execution_device(binding: ExecutionDeviceBinding) -> ExecutionDeviceBinding:
    """Perform the actual host/logical-device/physical-UUID comparison."""

    return validate_execution_device_binding(
        binding.as_dict(),
        requested_device=binding.torch_device,
        runtime_identity=collect_runtime_execution_identity(binding),
    )


def realized_ts4_permutation(n_channels: int, seed: int) -> dict[str, Any]:
    """Produce the full, reproducible TS4 permutation—not merely its hash."""

    _need(isinstance(n_channels, int) and n_channels > 0, "TS4 channel count must be positive")
    _need(seed in SEEDS, "TS4 permutation seed is outside the frozen set")
    vector = np.random.RandomState(seed).permutation(n_channels).astype("<i8", copy=False)
    return {
        "algorithm": "numpy.random.RandomState(seed).permutation(N)",
        "seed": seed,
        "channel_count": n_channels,
        "vector": [int(value) for value in vector.tolist()],
        "vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        "identity": bool(np.array_equal(vector, np.arange(n_channels, dtype=np.int64))),
        "identity_is_retained_not_excluded": True,
    }


def validate_realized_ts4_permutation(value: Mapping[str, Any], *, n_channels: int, seed: int) -> dict[str, Any]:
    """Require a sealed vector to equal the exact legacy RandomState result."""

    _need(isinstance(value, Mapping), "TS4 realized permutation is missing")
    expected = realized_ts4_permutation(n_channels, seed)
    _need(set(value) == set(expected), "TS4 realized permutation key set drift")
    _need(value.get("algorithm") == expected["algorithm"], "TS4 permutation algorithm drift")
    _need(value.get("seed") == seed and value.get("channel_count") == n_channels, "TS4 permutation seed/count drift")
    vector = value.get("vector")
    _need(isinstance(vector, list) and len(vector) == n_channels and all(isinstance(item, int) for item in vector), "TS4 full permutation vector malformed")
    observed = np.asarray(vector, dtype="<i8")
    expected_vector = np.asarray(expected["vector"], dtype="<i8")
    _need(np.array_equal(observed, expected_vector), "TS4 realized permutation vector differs from RandomState result")
    _need(value.get("vector_sha256") == expected["vector_sha256"], "TS4 realized permutation SHA-256 drift")
    _need(value.get("identity") == expected["identity"], "TS4 permutation identity flag drift")
    _need(value.get("identity_is_retained_not_excluded") is True, "TS4 identity-retention flag drift")
    return dict(expected)


def static_runner_audit_v2(source_path: Path | None = None) -> dict[str, Any]:
    """AST proof that v2 introduces no target-training operation path."""

    path = source_path or Path(__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    forbidden_import_fragments = ("torch.optim", "lightning", "tensorflow", "sklearn")
    forbidden_calls = {"backward", "zero_grad", "step", "train_dataloader", "fit_behavior_stats", "fit_side_feature_stats"}
    bad_imports = sorted(item for item in imports if any(fragment in item.lower() for fragment in forbidden_import_fragments))
    bad_calls = sorted(item for item in calls if item in forbidden_calls)
    _need(not bad_imports, f"v2 source imports forbidden training subsystem: {bad_imports}")
    _need(not bad_calls, f"v2 source calls forbidden target-training operation: {bad_calls}")
    return {
        "source": str(path),
        "sha256": sha256_file(path),
        "forbidden_imports": bad_imports,
        "forbidden_calls": bad_calls,
        "allowed_score_only_operations_after_future_authorization": [
            "one bound CPU or CUDA device for all cells",
            "frozen-model inference only",
            "TorchMetrics R2Score on sealed arrays",
            "write-once immutable output",
        ],
    }


def build_prelaunch_draft_v2(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Construct the v2 stored, non-authorizing prelaunch draft.

    The literal lack of a root Ed25519 key pin and a consumed sub-C adapter
    parity receipt pin is intentionally surfaced as a blocker rather than
    replaced with a free-form attestation string.
    """

    root = root.resolve()
    contract = validate_authority_chain_v2(root)
    query_window_counts = expected_query_window_counts_from_preflight(contract, root)
    source_snapshot = v2_source_snapshot(root)
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "kind": V2_PRELAUNCH_KIND,
        "status": "NOT_AUTHORIZED_FOR_SCORING",
        "append_only": True,
        "scope_id": SCOPE_ID,
        "v1_immutable_pins": {key: pin.as_dict() for key, pin in sorted(V1_IMMUTABLE_PINS.items())},
        "runner_source_snapshot": source_snapshot,
        "runner_source_snapshot_sha256": canonical_json_sha256(source_snapshot),
        "frozen_common_cohort": [session.as_dict() for session in contract.frozen_sessions],
        "matrix": {
            "N": EXPECTED_ELIGIBLE_SESSION_COUNT,
            "views": list(VIEWS),
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "sealed_cell_count": EXPECTED_SEALED_METRIC_COUNT,
            "query_windows_per_view_at_N15": sum(query_window_counts.values()),
            "query_window_count_by_asset_id": query_window_counts,
            "total_model_windows_across_2x2x3_at_N15": sum(query_window_counts.values()) * len(VIEWS) * len(ARMS) * len(SEEDS),
            "one_device_for_all_180_cells": True,
        },
        "authority_chain": dict(contract.authority_hashes),
        "score_only_protocol": {
            "metric": "torchmetrics.regression.R2Score(multioutput='variance_weighted')",
            "aggregate_recomputation": "CPU TorchMetrics R2Score once per sealed cell from NPZ predictions/targets",
            "aggregate_r2_recompute_atol": R2_RECOMPUTE_ATOL,
            "ts4": "seal and hash the full numpy.random.RandomState(seed).permutation(N) vector",
            "checkpoint_selection": "only fixed C1 epoch_011 for each arm/seed",
            "normalizer_fitting": "FORBIDDEN",
            "optimizer_or_backward": "FORBIDDEN",
        },
        "execution_device_authorization": {
            "cli_flag": "--device",
            "allowed_device_strings": ["cpu", "cuda:<index>"],
            "one_exact_execution_device_for_all_180_cells": True,
            "cpu_binding": {
                "torch_device": "cpu",
                "kind": "cpu",
                "matrix_torch_device": "cpu",
                "host": None,
                "physical_gpu_uuid": None,
                "physical_gpu_pci_bus_id": None,
                "cuda_visible_devices": None,
                "all_180_cells_same_device": True,
            },
            "cuda_binding_requirements": [
                "exact logical PyTorch device string, for example cuda:0",
                "exact host",
                "exact physical GPU UUID distinct from the logical torch device",
                "exact physical GPU PCI bus ID to bind the logical index to the physical GPU",
                "optional exact CUDA_VISIBLE_DEVICES string (null means it must be unset)",
                "all_180_cells_same_device=true",
                "before checkpoint/NWB access, runtime must measure hostname, selected torch device, physical GPU UUID, and CUDA_VISIBLE_DEVICES and compare them exactly",
            ],
            "mixed_device_matrix": "FORBIDDEN",
        },
        "sealed_output_requirements": {
            "prediction_target_npz": "0444 after fsync",
            "metric_json": "0444 after fsync",
            "seal_manifest_json": "0444 after fsync",
            "aggregate_json": "0444 after fsync",
            "aggregate_must_not_trust_metric_json_scalar": True,
        },
        "scorer_adapter_parity_requirement": {
            "required_before_external_scoring": True,
            "receipt_schema": SCORER_ADAPTER_PARITY_RECEIPT_SCHEMA,
            "receipt_must_come_from": "already-consumed sub-C development session",
            "receipt_pin_configured": SCORER_ADAPTER_PARITY_RECEIPT_PIN is not None,
            "this_turn": "NOT_RUN; sub-C/RT not opened",
        },
        "authorization_signature_schema": {
            "schema": "ed25519-detached-canonical-json-v1",
            "required_fields": [
                "schema",
                "key_id",
                "public_key_sha256",
                "signed_payload_sha256",
                "signature_base64",
            ],
            "signed_payload": "canonical JSON authorization excluding the signature field",
            "signature_encoding": "RFC 4648 base64 of a 64-byte Ed25519 detached signature",
            "placeholder_or_free_form_evidence": "REJECTED",
            "root_public_key_pin_configured": ROOT_ED25519_PUBLIC_KEY_PIN is not None,
        },
        "authorization_required_before_scoring": {
            "status": "AUTHORIZED_FOR_SCORING",
            "kind": V2_AUTHORIZATION_KIND,
            "permitted_actions": ["seal_session_predictions_metrics", "aggregate_sealed_metrics"],
            "must_bind": [
                "stored v2 prelaunch draft SHA-256",
                "stored v2 prelaunch receipt SHA-256",
                "stored v2 prelaunch artifact-seal SHA-256",
                "final four-file v2 source snapshot",
                "one exact execution_device binding",
                "scorer_adapter_parity_receipt_sha256",
                "Ed25519 signature envelope",
            ],
            "this_draft_is_an_authorization": False,
        },
        "non_authorizable_blockers": [
            "root has not pinned a real Ed25519 public key file and SHA-256",
            "root has not pinned a consumed sub-C scorer-adapter parity receipt SHA-256",
        ],
        "prohibited_by_this_draft": [
            "checkpoint loading",
            "model forward",
            "prediction or R2 computation on external data",
            "GPU use",
            "optimizer, backward, or target update",
            "normalizer fitting",
            "aggregate opening of external score artifacts",
            "NWB download or sub-C/RT access",
        ],
    }


def build_prelaunch_receipt_v2(root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return deterministic v2 draft/receipt content without any score action."""

    draft = build_prelaunch_draft_v2(root)
    receipt = {
        "schema_version": V2_SCHEMA_VERSION,
        "receipt_kind": V2_PRELAUNCH_RECEIPT_KIND,
        "status": "NOT_AUTHORIZED_FOR_SCORING",
        "append_only": True,
        "scope_id": SCOPE_ID,
        "prelaunch_draft": {
            "filename": "prelaunch_authorization_draft.json",
            "sha256": canonical_json_sha256(draft),
        },
        "runner_source_snapshot": draft["runner_source_snapshot"],
        "runner_source_snapshot_sha256": draft["runner_source_snapshot_sha256"],
        "v1_immutable_pins_verified": verify_v1_immutable(root),
        "dry_run_audit": {
            "checkpoint_files_opened": 0,
            "model_forward_calls": 0,
            "prediction_calls": 0,
            "r2_computations": 0,
            "gpu_used": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "normalizer_fits": 0,
            "sub_c_endpoint_opened": False,
            "rt_endpoint_opened": False,
        },
        "authorization_blocked_until": [
            "root pins an actual Ed25519 public-key file and SHA-256",
            "root pins the SHA-256 of a consumed sub-C scorer-adapter parity receipt",
        ],
    }
    return draft, receipt


def write_prelaunch_artifacts_v2(output_dir: Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Create a fresh three-file v2 prelaunch bundle and make each file 0444."""

    output_dir = output_dir.resolve()
    _need(not output_dir.exists(), f"append-only v2 prelaunch output already exists: {output_dir}")
    draft, receipt = build_prelaunch_receipt_v2(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path = output_dir / "prelaunch_authorization_draft.json"
    receipt_path = output_dir / "receipt.json"
    seal_path = output_dir / "prelaunch_artifact_seal.json"
    draft_sha = _write_immutable_json_exclusive(draft_path, draft)
    _need(draft_sha == receipt["prelaunch_draft"]["sha256"], "internal v2 prelaunch draft hash mismatch")
    receipt_sha = _write_immutable_json_exclusive(receipt_path, receipt)
    seal = {
        "schema_version": V2_SCHEMA_VERSION,
        "kind": V2_PRELAUNCH_SEAL_KIND,
        "status": "NOT_AUTHORIZED_FOR_SCORING",
        "append_only": True,
        "scope_id": SCOPE_ID,
        "prelaunch_draft": {"filename": draft_path.name, "sha256": draft_sha},
        "receipt": {"filename": receipt_path.name, "sha256": receipt_sha},
        "stored_prelaunch_draft_sha256": draft_sha,
        "stored_prelaunch_receipt_sha256": receipt_sha,
        "runner_source_snapshot": draft["runner_source_snapshot"],
        "runner_source_snapshot_sha256": draft["runner_source_snapshot_sha256"],
        "all_artifacts_mode": "0444_after_fsync",
    }
    seal_sha = _write_immutable_json_exclusive(seal_path, seal)
    return {
        "output_dir": str(output_dir),
        "prelaunch_draft": {"path": str(draft_path), "sha256": draft_sha},
        "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "artifact_seal": {"path": str(seal_path), "sha256": seal_sha},
        "status": "NOT_AUTHORIZED_FOR_SCORING",
    }


def load_stored_prelaunch_bundle(prelaunch_dir: Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Read the immutable v2 prelaunch bundle rather than rebuilding a draft."""

    prelaunch_dir = prelaunch_dir.resolve()
    _need(prelaunch_dir.is_dir() and not prelaunch_dir.is_symlink(), "stored v2 prelaunch directory missing or unsafe")
    draft_path = prelaunch_dir / "prelaunch_authorization_draft.json"
    receipt_path = prelaunch_dir / "receipt.json"
    seal_path = prelaunch_dir / "prelaunch_artifact_seal.json"
    for path in (draft_path, receipt_path, seal_path):
        _need(path.is_file() and not path.is_symlink(), f"stored v2 prelaunch file missing or unsafe: {path.name}")
        _need(_is_immutable_mode(path), f"stored v2 prelaunch file is not immutable 0444: {path.name}")
    draft = _load_json(draft_path, "stored v2 prelaunch draft")
    receipt = _load_json(receipt_path, "stored v2 prelaunch receipt")
    seal = _load_json(seal_path, "stored v2 prelaunch artifact seal")
    draft_sha = sha256_file(draft_path)
    receipt_sha = sha256_file(receipt_path)
    seal_sha = sha256_file(seal_path)
    _need(draft.get("kind") == V2_PRELAUNCH_KIND and draft.get("status") == "NOT_AUTHORIZED_FOR_SCORING", "stored v2 draft status/kind drift")
    _need(receipt.get("receipt_kind") == V2_PRELAUNCH_RECEIPT_KIND and receipt.get("status") == "NOT_AUTHORIZED_FOR_SCORING", "stored v2 receipt status/kind drift")
    _need(seal.get("kind") == V2_PRELAUNCH_SEAL_KIND and seal.get("status") == "NOT_AUTHORIZED_FOR_SCORING", "stored v2 artifact seal status/kind drift")
    _need(seal.get("prelaunch_draft") == {"filename": draft_path.name, "sha256": draft_sha}, "stored v2 draft seal binding drift")
    _need(seal.get("receipt") == {"filename": receipt_path.name, "sha256": receipt_sha}, "stored v2 receipt seal binding drift")
    _need(seal.get("stored_prelaunch_draft_sha256") == draft_sha, "stored v2 explicit draft SHA-256 drift")
    _need(seal.get("stored_prelaunch_receipt_sha256") == receipt_sha, "stored v2 explicit receipt SHA-256 drift")
    _need(receipt.get("prelaunch_draft", {}).get("sha256") == draft_sha, "stored v2 receipt draft binding drift")
    snapshot = seal.get("runner_source_snapshot")
    _need(isinstance(snapshot, dict), "stored v2 source snapshot missing")
    _need(seal.get("runner_source_snapshot_sha256") == canonical_json_sha256(snapshot), "stored v2 source snapshot hash drift")
    _need(draft.get("runner_source_snapshot") == snapshot, "stored draft source snapshot differs from artifact seal")
    _need(receipt.get("runner_source_snapshot") == snapshot, "stored receipt source snapshot differs from artifact seal")
    _need(v2_source_snapshot(root) == snapshot, "live v2 source no longer matches stored prelaunch source snapshot")
    return {
        "directory": str(prelaunch_dir),
        "draft": draft,
        "receipt": receipt,
        "seal": seal,
        "draft_sha256": draft_sha,
        "receipt_sha256": receipt_sha,
        "artifact_seal_sha256": seal_sha,
        "runner_source_snapshot": snapshot,
    }


def _authorization_payload_without_signature(authorization: Mapping[str, Any]) -> dict[str, Any]:
    _authorization_need("signature" in authorization, "authorization signature envelope is missing")
    return {str(key): value for key, value in authorization.items() if key != "signature"}


def _validate_signature_envelope(signature: Any, payload: Mapping[str, Any]) -> None:
    """Enforce the exact schema before a root key is ever considered."""

    _authorization_need(isinstance(signature, Mapping), "authorization signature must be an Ed25519 envelope object, not free-form evidence")
    required = {"schema", "key_id", "public_key_sha256", "signed_payload_sha256", "signature_base64"}
    _authorization_need(set(signature) == required, "authorization signature envelope key set drift")
    _authorization_need(signature.get("schema") == "ed25519-detached-canonical-json-v1", "authorization signature schema mismatch")
    for key in ("key_id", "public_key_sha256", "signed_payload_sha256", "signature_base64"):
        _authorization_need(isinstance(signature.get(key), str) and bool(signature[key]), f"authorization signature field missing: {key}")
    _authorization_need(re.fullmatch(r"[0-9a-f]{64}", signature["public_key_sha256"]) is not None, "authorization public-key SHA-256 malformed")
    _authorization_need(signature["signed_payload_sha256"] == canonical_json_sha256(payload), "authorization signed payload SHA-256 mismatch")
    try:
        raw_signature = base64.b64decode(signature["signature_base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise AuthorizationError("authorization signature is not strict base64") from exc
    _authorization_need(len(raw_signature) == 64, "authorization Ed25519 signature must contain 64 bytes")


def _verify_root_ed25519_signature(signature: Mapping[str, Any], payload: Mapping[str, Any], root: Path) -> None:
    """Verify a real root signature, or fail closed while no root key is pinned."""

    _validate_signature_envelope(signature, payload)
    _authorization_need(
        ROOT_ED25519_PUBLIC_KEY_PIN is not None,
        "V2_NON_AUTHORIZABLE_UNTIL_ROOT_ED25519_PUBLIC_KEY_PINNED",
    )
    # This branch is deliberately unreachable in v2 as delivered.  Keeping the
    # check explicit prevents a later configuration change from turning a
    # free-form attestation into authorization.
    assert ROOT_ED25519_PUBLIC_KEY_PIN is not None
    key_path = (root / ROOT_ED25519_PUBLIC_KEY_PIN.relative_path).resolve()
    _authorization_need(key_path.is_file() and not key_path.is_symlink(), "pinned Ed25519 root public key missing")
    _authorization_need(sha256_file(key_path) == ROOT_ED25519_PUBLIC_KEY_PIN.sha256, "pinned Ed25519 root public key SHA-256 drift")
    _authorization_need(signature["public_key_sha256"] == ROOT_ED25519_PUBLIC_KEY_PIN.sha256, "authorization public key does not match root pin")
    try:
        ed25519 = importlib.import_module("cryptography.hazmat.primitives.asymmetric.ed25519")
        serialization = importlib.import_module("cryptography.hazmat.primitives.serialization")
        public_key = serialization.load_pem_public_key(key_path.read_bytes())
        _authorization_need(isinstance(public_key, ed25519.Ed25519PublicKey), "root public key is not Ed25519")
        public_key.verify(base64.b64decode(signature["signature_base64"], validate=True), _canonical_json(payload))
    except AuthorizationError:
        raise
    except Exception as exc:  # cryptography gives several version-specific exceptions.
        raise AuthorizationError("Ed25519 authorization signature verification failed") from exc


def _verify_adapter_parity_receipt(authorization: Mapping[str, Any], root: Path) -> None:
    """Fail closed until a real, consumed sub-C parity receipt is root-pinned."""

    _authorization_need(
        SCORER_ADAPTER_PARITY_RECEIPT_PIN is not None,
        "V2_NON_AUTHORIZABLE_UNTIL_ROOT_PINS_CONSUMED_SUBC_PARITY_RECEIPT",
    )
    assert SCORER_ADAPTER_PARITY_RECEIPT_PIN is not None
    receipt_path = (root / SCORER_ADAPTER_PARITY_RECEIPT_PIN.relative_path).resolve()
    _authorization_need(receipt_path.is_file() and not receipt_path.is_symlink(), "pinned scorer-adapter parity receipt missing")
    receipt_sha = sha256_file(receipt_path)
    _authorization_need(receipt_sha == SCORER_ADAPTER_PARITY_RECEIPT_PIN.sha256, "pinned scorer-adapter parity receipt SHA-256 drift")
    _authorization_need(authorization.get("scorer_adapter_parity_receipt_sha256") == receipt_sha, "authorization scorer-adapter parity receipt binding drift")
    receipt = _load_json(receipt_path, "scorer-adapter parity receipt")
    _authorization_need(receipt.get("schema") == SCORER_ADAPTER_PARITY_RECEIPT_SCHEMA, "scorer-adapter parity receipt schema drift")
    _authorization_need(receipt.get("status") == "PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION", "scorer-adapter parity receipt is not a consumed sub-C parity confirmation")


def require_future_authorization_v2(
    *,
    authorization_path: Path,
    output_root: Path,
    prelaunch_dir: Path,
    requested_device: str,
    permitted_action: str,
    root: Path = REPO_ROOT,
    output_root_must_be_new: bool,
    counters: RuntimeAuditCounters,
) -> tuple[ScoreContract, ExecutionDeviceBinding]:
    """Verify a future v2 authorization before any target/model operation.

    Crucially, bindings are taken from the immutable stored prelaunch bundle,
    never from a freshly rebuilt mutable draft.
    """

    counters.authorization_checks += 1
    root = root.resolve()
    static_runner_audit_v2()
    contract = validate_authority_chain_v2(root, counters=counters)
    stored = load_stored_prelaunch_bundle(prelaunch_dir, root)
    authorization = _load_json(authorization_path.resolve(), "future v2 score authorization")
    _authorization_need(authorization.get("status") == "AUTHORIZED_FOR_SCORING", "authorization status is not AUTHORIZED_FOR_SCORING")
    _authorization_need(authorization.get("kind") == V2_AUTHORIZATION_KIND, "v2 authorization kind mismatch")
    actions = authorization.get("permitted_actions")
    _authorization_need(isinstance(actions, list) and permitted_action in actions, f"authorization omits required action {permitted_action!r}")
    _authorization_need(authorization.get("no_retry_no_overwrite") is True, "authorization must forbid retry/overwrite")
    _authorization_need(authorization.get("normalizer_fitting_permitted") is False, "authorization must forbid normalizer fitting")
    _authorization_need(authorization.get("target_updates_permitted") is False, "authorization must forbid target updates")
    _authorization_need(authorization.get("optimizer_or_backward_permitted") is False, "authorization must forbid optimizer/backward")
    _authorization_need(authorization.get("stored_prelaunch_draft_sha256") == stored["draft_sha256"], "authorization stored draft SHA-256 binding drift")
    _authorization_need(authorization.get("stored_prelaunch_receipt_sha256") == stored["receipt_sha256"], "authorization stored receipt SHA-256 binding drift")
    _authorization_need(authorization.get("stored_prelaunch_artifact_seal_sha256") == stored["artifact_seal_sha256"], "authorization stored artifact-seal SHA-256 binding drift")
    _authorization_need(authorization.get("runner_source_snapshot") == stored["runner_source_snapshot"], "authorization final v2 source snapshot binding drift")
    _authorization_need(
        authorization.get("runner_source_snapshot_sha256") == canonical_json_sha256(stored["runner_source_snapshot"]),
        "authorization final v2 source snapshot SHA-256 binding drift",
    )
    _authorization_need(authorization.get("authority_chain") == contract.authority_hashes, "authorization authority chain binding drift")
    _authorization_need(
        authorization.get("frozen_cohort_asset_ids") == [item.asset_id for item in contract.frozen_sessions],
        "authorization frozen cohort binding drift",
    )
    device = validate_execution_device_binding(authorization.get("execution_device"), requested_device=requested_device)
    _authorization_need(authorization.get("single_use_output_root") == str(output_root.resolve()), "authorization output-root binding drift")
    if output_root_must_be_new:
        _authorization_need(not output_root.exists(), f"score output root already exists; retry/overwrite forbidden: {output_root}")
    else:
        _authorization_need(output_root.is_dir() and not output_root.is_symlink(), f"sealed output root is missing or unsafe: {output_root}")
    _verify_adapter_parity_receipt(authorization, root)
    payload = _authorization_payload_without_signature(authorization)
    _verify_root_ed25519_signature(authorization.get("signature"), payload, root)
    return contract, device


def _validate_prediction_target_arrays(predictions: np.ndarray, targets: np.ndarray, *, expected_count: int | None = None) -> None:
    _need(isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray), "prediction/target NPZ arrays are missing")
    _need(predictions.ndim == 2 and targets.ndim == 2 and predictions.shape == targets.shape, "prediction/target array shape drift")
    _need(predictions.shape[1] == 2, "external endpoint requires two-output velocity")
    _need(predictions.shape[0] > 0, "prediction/target arrays cannot be empty")
    _need(predictions.dtype == np.dtype(np.float32) and targets.dtype == np.dtype(np.float32), "prediction/target arrays must have exact float32 dtype")
    if expected_count is not None:
        _need(predictions.shape[0] == expected_count, "prediction/target query-window count drift")
    _need(np.isfinite(predictions).all() and np.isfinite(targets).all(), "prediction/target arrays contain nonfinite values")


def recompute_torchmetrics_r2_cpu(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute the frozen metric from arrays on CPU only.

    This is called by the aggregate verifier, never by dry-run/prelaunch code.
    The test suite uses only synthetic arrays to exercise this required check.
    """

    _validate_prediction_target_arrays(predictions, targets)
    torch = importlib.import_module("torch")
    metrics = importlib.import_module("torchmetrics.regression")
    device = torch.device("cpu")
    metric = metrics.R2Score(multioutput=R2_MULTI_OUTPUT).to(device)
    metric.update(torch.as_tensor(predictions, dtype=torch.float32, device=device), torch.as_tensor(targets, dtype=torch.float32, device=device))
    result = float(metric.compute().detach().cpu().item())
    _need(math.isfinite(result), "recomputed TorchMetrics R2 is nonfinite")
    return result


class SealedOutputWriterV2:
    """Write 0444 prediction/metric cells, then a 0444 full-matrix seal.

    The class is deliberately usable with synthetic fixtures for verification.
    A future model adapter may call it only after ``require_future_authorization_v2``
    succeeds and proves one execution device for the entire matrix.
    """

    def __init__(
        self,
        root: Path,
        contract: ScoreContract,
        *,
        authorization_sha256: str,
        execution_device: ExecutionDeviceBinding,
        authority_root: Path = REPO_ROOT,
    ) -> None:
        self.root = root.resolve()
        _need(not self.root.exists(), f"sealed v2 output root already exists: {self.root}")
        self.root.mkdir(parents=True, exist_ok=False)
        self.contract = contract
        self.authorization_sha256 = authorization_sha256
        self.execution_device = execution_device
        self.expected_query_window_counts = expected_query_window_counts_from_preflight(contract, authority_root)
        self.records: list[dict[str, Any]] = []
        self._written_keys: set[tuple[str, str, str, int]] = set()
        self._sealed = False
        run_manifest = {
            "schema_version": V2_SCHEMA_VERSION,
            "kind": "dandi_000688_subm_co_score_only_run_manifest_v2",
            "status": "SEALED_METRICS_PENDING",
            "scope_id": SCOPE_ID,
            "authorization_sha256": authorization_sha256,
            "execution_device": execution_device.as_dict(),
            "one_device_for_all_180_cells": True,
            "contract": contract.as_dict(),
            "write_order": "per-session prediction NPZ, then metric JSON, then seal manifest; aggregate is separate",
        }
        self.run_manifest_relative = "sealed/run_manifest.json"
        _write_immutable_json_exclusive(_safe_relative_path(self.root, self.run_manifest_relative, label="run manifest"), run_manifest)

    def write_session_result(
        self,
        *,
        session: FrozenSession,
        view: str,
        arm: str,
        seed: int,
        checkpoint_sha256: str,
        normalizer: NormalizerPins,
        predictions: np.ndarray,
        targets: np.ndarray,
        query_window_count: int,
        ts4_permutation: Mapping[str, Any] | None,
        ts4_feature_channel_count: int | None = None,
    ) -> None:
        _need(not self._sealed, "cannot write after v2 seal manifest")
        _need(view in VIEWS and arm in ARMS and seed in SEEDS, "invalid v2 sealed score identity")
        expected_session = {item.asset_id: item for item in self.contract.frozen_sessions}.get(session.asset_id)
        _need(expected_session == session, "v2 writer session is not the exact frozen cohort session")
        _need(
            checkpoint_sha256 == self.contract.checkpoint_by_key()[(arm, seed)].checkpoint.sha256,
            "v2 writer checkpoint SHA-256 differs from frozen arm/seed pin",
        )
        _need(normalizer.as_dict() == self.contract.normalizers[view].as_dict(), "v2 writer normalizer differs from frozen view pin")
        key = (session.asset_id, view, arm, seed)
        _need(key not in self._written_keys, f"duplicate v2 sealed score cell: {key}")
        _need(query_window_count == self.expected_query_window_counts[session.asset_id], "query-window count differs from pinned v2 preflight ledger")
        _validate_prediction_target_arrays(predictions, targets, expected_count=query_window_count)
        if arm == "shared_t4":
            _need(ts4_permutation is None, "shared_t4 cannot carry a TS4 permutation")
            _need(ts4_feature_channel_count is None, "shared_t4 cannot carry a TS4 feature-channel count")
            permutation = None
        else:
            _need(isinstance(ts4_feature_channel_count, int) and ts4_feature_channel_count > 0, "TS4 feature-channel count is required for a full realized permutation")
            permutation = validate_realized_ts4_permutation(
                ts4_permutation or {},
                n_channels=ts4_feature_channel_count,
                seed=seed,
            )
        prefix = f"sealed/sessions/{session.asset_id}/{view}/{arm}/seed_{seed}"
        prediction_relative = f"{prefix}/predictions_targets.npz"
        metric_relative = f"{prefix}/metric.json"
        prediction_path = _safe_relative_path(self.root, prediction_relative, label="prediction")
        metric_path = _safe_relative_path(self.root, metric_relative, label="metric")
        prediction_sha = _write_immutable_npz_exclusive(
            prediction_path,
            predictions=predictions,
            targets=targets,
        )
        # The score reported in JSON is computed *after reopening* the exact
        # immutable float32 bytes that aggregate will later verify.  This
        # prevents a GPU/resident or pre-cast scalar from becoming authority.
        sealed_predictions, sealed_targets = _load_prediction_target_npz(
            prediction_path,
            expected_shape=predictions.shape,
            expected_count=query_window_count,
        )
        sealed_r2 = recompute_torchmetrics_r2_cpu(sealed_predictions, sealed_targets)
        metric = {
            "schema_version": V2_SCHEMA_VERSION,
            "kind": V2_METRIC_KIND,
            "sealed": True,
            "metric_status": "FINITE_R2",
            "scope_id": SCOPE_ID,
            "asset_id": session.asset_id,
            "session_id": session.session_id,
            "frozen_path": session.frozen_path,
            "view": view,
            "arm": arm,
            "seed": seed,
            "checkpoint_sha256": checkpoint_sha256,
            "source_normalizer": normalizer.as_dict(),
            "execution_device": self.execution_device.as_dict(),
            "evaluation": {"query_window_count": query_window_count},
            "metric": {
                "name": "torchmetrics.regression.R2Score",
                "multioutput": R2_MULTI_OUTPUT,
                "r2": sealed_r2,
                "aggregate_recompute_atol": R2_RECOMPUTE_ATOL,
            },
            "prediction_target_bundle": {
                "relative_path": prediction_relative,
                "sha256": prediction_sha,
                "shape": [int(value) for value in predictions.shape],
            },
            "ts4_realized_permutation": permutation,
        }
        metric_sha = _write_immutable_json_exclusive(metric_path, metric)
        self.records.append(
            {
                "asset_id": session.asset_id,
                "session_id": session.session_id,
                "view": view,
                "arm": arm,
                "seed": seed,
                "metric_relative_path": metric_relative,
                "metric_sha256": metric_sha,
                "prediction_relative_path": prediction_relative,
                "prediction_sha256": prediction_sha,
                "execution_device": self.execution_device.as_dict(),
            }
        )
        self._written_keys.add(key)

    def finalize(self) -> dict[str, Any]:
        _need(not self._sealed, "v2 seal manifest already exists")
        _need(len(self.records) == EXPECTED_SEALED_METRIC_COUNT, "v2 sealed metric cell count is incomplete")
        records = sorted(self.records, key=lambda item: (item["asset_id"], item["view"], item["arm"], item["seed"]))
        seal = {
            "schema_version": V2_SCHEMA_VERSION,
            "kind": V2_SEAL_KIND,
            "status": "SEALED_PER_SESSION_PREDICTIONS_AND_METRICS",
            "scope_id": SCOPE_ID,
            "expected_metric_cell_count": EXPECTED_SEALED_METRIC_COUNT,
            "execution_device": self.execution_device.as_dict(),
            "one_device_for_all_180_cells": True,
            "records": records,
            "aggregate_status": "NOT_OPENED",
        }
        relative = "sealed/seal_manifest.json"
        seal_path = _safe_relative_path(self.root, relative, label="seal manifest")
        seal_sha = _write_immutable_json_exclusive(seal_path, seal)
        self._sealed = True
        return {"path": str(seal_path), "relative_path": relative, "sha256": seal_sha, "status": seal["status"]}


def _load_prediction_target_npz(path: Path, *, expected_shape: Sequence[int], expected_count: int) -> tuple[np.ndarray, np.ndarray]:
    _need(path.is_file() and not path.is_symlink(), f"sealed prediction NPZ missing or unsafe: {path}")
    _need(_is_immutable_mode(path), f"sealed prediction NPZ is not immutable 0444: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            _need(set(loaded.files) == {"predictions", "targets"}, "sealed prediction NPZ array key set drift")
            predictions = np.asarray(loaded["predictions"])
            targets = np.asarray(loaded["targets"])
    except (OSError, ValueError) as exc:
        raise ScoreOnlyContractError(f"cannot open sealed prediction NPZ: {path}") from exc
    _validate_prediction_target_arrays(predictions, targets, expected_count=expected_count)
    _need(list(predictions.shape) == [int(value) for value in expected_shape], "sealed prediction NPZ shape differs from metric metadata")
    return predictions, targets


def _read_and_verify_sealed_metric_grid_v2(
    output_root: Path,
    contract: ScoreContract,
    *,
    expected_query_window_counts: Mapping[str, int],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Internal full-matrix verifier parameterized by a pinned count map."""

    output_root = output_root.resolve()
    seal_path = _safe_relative_path(output_root, "sealed/seal_manifest.json", label="seal manifest")
    _need(seal_path.is_file() and _is_immutable_mode(seal_path), "v2 seal manifest missing or not immutable")
    seal = _load_json(seal_path, "v2 seal manifest")
    _need(seal.get("kind") == V2_SEAL_KIND and seal.get("status") == "SEALED_PER_SESSION_PREDICTIONS_AND_METRICS", "v2 metrics are not sealed")
    _need(seal.get("scope_id") == SCOPE_ID, "v2 seal scope drift")
    _need(seal.get("expected_metric_cell_count") == EXPECTED_SEALED_METRIC_COUNT, "v2 seal expected metric-cell count drift")
    seal_device = execution_device_binding_from_mapping(seal.get("execution_device"))
    _need(seal.get("one_device_for_all_180_cells") is True, "v2 seal allows a mixed-device matrix")
    records = seal.get("records")
    _need(isinstance(records, list) and len(records) == EXPECTED_SEALED_METRIC_COUNT, "v2 sealed record count drift")
    session_index = {session.asset_id: index for index, session in enumerate(contract.frozen_sessions)}
    session_by_asset = {session.asset_id: session for session in contract.frozen_sessions}
    _need(set(expected_query_window_counts) == set(session_index), "expected query-window-count asset set drift")
    _need(all(isinstance(value, int) and value > 0 for value in expected_query_window_counts.values()), "expected query-window-count values malformed")
    seed_index = {seed: index for index, seed in enumerate(SEEDS)}
    expected_keys = {(session.asset_id, view, arm, seed) for session in contract.frozen_sessions for view in VIEWS for arm in ARMS for seed in SEEDS}
    observed_keys: set[tuple[str, str, str, int]] = set()
    grids = {view: {arm: np.full((len(SEEDS), len(contract.frozen_sessions)), np.nan, dtype=np.float64) for arm in ARMS} for view in VIEWS}
    recomputation: list[dict[str, Any]] = []
    for record in records:
        _need(isinstance(record, dict), "malformed v2 sealed record")
        asset_id, view, arm, seed = record.get("asset_id"), record.get("view"), record.get("arm"), record.get("seed")
        _need(asset_id in session_index and view in VIEWS and arm in ARMS and seed in seed_index, "v2 sealed record identity drift")
        key = (str(asset_id), str(view), str(arm), int(seed))
        _need(key not in observed_keys, f"duplicate v2 sealed record: {key}")
        observed_keys.add(key)
        _need(record.get("session_id") == session_by_asset[str(asset_id)].session_id, "v2 sealed record session ID differs from frozen session")
        _need(record.get("execution_device") == seal_device.as_dict(), "v2 record execution device differs from matrix seal")
        metric_relative, prediction_relative = record.get("metric_relative_path"), record.get("prediction_relative_path")
        _need(isinstance(metric_relative, str) and isinstance(prediction_relative, str), "v2 sealed record paths missing")
        canonical_prefix = f"sealed/sessions/{asset_id}/{view}/{arm}/seed_{seed}"
        _need(metric_relative == f"{canonical_prefix}/metric.json", "v2 sealed metric canonical path drift")
        _need(prediction_relative == f"{canonical_prefix}/predictions_targets.npz", "v2 sealed prediction canonical path drift")
        metric_path = _safe_relative_path(output_root, metric_relative, label="sealed metric")
        prediction_path = _safe_relative_path(output_root, prediction_relative, label="sealed prediction")
        _need(metric_path.is_file() and _is_immutable_mode(metric_path), "sealed metric JSON missing or not immutable")
        _need(sha256_file(metric_path) == record.get("metric_sha256"), f"v2 sealed metric hash drift: {metric_relative}")
        _need(sha256_file(prediction_path) == record.get("prediction_sha256"), f"v2 sealed prediction hash drift: {prediction_relative}")
        metric = _load_json(metric_path, "v2 sealed metric")
        _need(metric.get("kind") == V2_METRIC_KIND and metric.get("sealed") is True and metric.get("scope_id") == SCOPE_ID, "v2 sealed metric contract drift")
        _need((metric.get("asset_id"), metric.get("view"), metric.get("arm"), metric.get("seed")) == key, "v2 metric identity differs from seal index")
        session = session_by_asset[str(asset_id)]
        _need(metric.get("session_id") == session.session_id, "v2 metric session ID differs from frozen session")
        _need(metric.get("frozen_path") == session.frozen_path, "v2 metric frozen path differs from frozen session")
        _need(metric.get("execution_device") == seal_device.as_dict(), "v2 metric execution device differs from matrix seal")
        checkpoint = contract.checkpoint_by_key()[(str(arm), int(seed))]
        _need(metric.get("checkpoint_sha256") == checkpoint.checkpoint.sha256, "v2 metric checkpoint binding drift")
        normalizer = metric.get("source_normalizer")
        _need(isinstance(normalizer, dict), "v2 metric normalizer binding missing")
        _need(normalizer == contract.normalizers[str(view)].as_dict(), "v2 metric source normalizer full binding drift")
        evaluation = metric.get("evaluation")
        bundle = metric.get("prediction_target_bundle")
        metric_info = metric.get("metric")
        _need(isinstance(evaluation, dict) and isinstance(bundle, dict) and isinstance(metric_info, dict), "v2 metric payload missing")
        count = evaluation.get("query_window_count")
        _need(isinstance(count, int) and count > 0, "v2 metric query-window count drift")
        _need(count == expected_query_window_counts[str(asset_id)], "v2 metric query-window count differs from pinned preflight ledger")
        _need(bundle.get("relative_path") == prediction_relative and bundle.get("sha256") == record.get("prediction_sha256"), "v2 metric prediction bundle binding drift")
        shape = bundle.get("shape")
        _need(isinstance(shape, list) and len(shape) == 2, "v2 metric prediction shape metadata drift")
        _need(all(isinstance(value, int) and value > 0 for value in shape), "v2 metric prediction shape values malformed")
        predictions, targets = _load_prediction_target_npz(prediction_path, expected_shape=shape, expected_count=count)
        _need(metric.get("metric_status") == "FINITE_R2", "v2 metric status is not finite")
        _need(metric_info.get("name") == "torchmetrics.regression.R2Score" and metric_info.get("multioutput") == R2_MULTI_OUTPUT, "v2 metric definition drift")
        reported = metric_info.get("r2")
        _need(isinstance(reported, (int, float)) and math.isfinite(float(reported)), "v2 reported metric R2 is nonfinite")
        _need(metric_info.get("aggregate_recompute_atol") == R2_RECOMPUTE_ATOL, "v2 metric R2 tolerance drift")
        if arm == "shared_t4":
            _need(metric.get("ts4_realized_permutation") is None, "v2 shared_t4 metric carries TS4 permutation")
        else:
            # The validated sealed record stores the true side-feature channel
            # count.  For v2 synthetic fixtures this metadata explicitly uses
            # the same count as the supplied full vector.
            permutation = metric.get("ts4_realized_permutation")
            _need(isinstance(permutation, dict), "v2 TS4 metric is missing full permutation")
            validate_realized_ts4_permutation(permutation, n_channels=int(permutation.get("channel_count", 0)), seed=int(seed))
        recomputed = recompute_torchmetrics_r2_cpu(predictions, targets)
        _need(abs(float(reported) - recomputed) <= R2_RECOMPUTE_ATOL, "v2 metric JSON R2 differs from CPU TorchMetrics recomputation")
        grids[str(view)][str(arm)][seed_index[int(seed)], session_index[str(asset_id)]] = recomputed
        recomputation.append({"asset_id": asset_id, "view": view, "arm": arm, "seed": seed, "reported_r2": float(reported), "recomputed_r2": recomputed})
    _need(observed_keys == expected_keys, "v2 sealed record set differs from frozen full score matrix")
    return grids, {
        "seal_manifest_sha256": sha256_file(seal_path),
        "execution_device": seal_device.as_dict(),
        "r2_recomputation": {
            "implementation": "torchmetrics.regression.R2Score(multioutput='variance_weighted') on CPU",
            "cells_recomputed": len(recomputation),
            "atol": R2_RECOMPUTE_ATOL,
            "records": recomputation,
        },
    }


def read_and_verify_sealed_metric_grid_v2(
    output_root: Path,
    contract: ScoreContract,
    *,
    root: Path = REPO_ROOT,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Read all 180 sealed cells using ledger-derived count expectations."""

    return _read_and_verify_sealed_metric_grid_v2(
        output_root,
        contract,
        expected_query_window_counts=expected_query_window_counts_from_preflight(contract, root),
    )


def aggregate_sealed_v2(output_root: Path, contract: ScoreContract) -> dict[str, Any]:
    """Aggregate a v2 seal only after NPZ-level validation/recomputation."""

    output_root = output_root.resolve()
    grids, evidence = read_and_verify_sealed_metric_grid_v2(output_root, contract)
    aggregate_relative = "aggregate/endpoint_aggregate.json"
    aggregate_path = _safe_relative_path(output_root, aggregate_relative, label="aggregate")
    _need(not aggregate_path.exists(), "v2 aggregate overwrite/retry is forbidden")
    view_results = {view: v1.summarize_endpoint_view(grids[view]["shared_t4"], grids[view]["shared_ts4"]) for view in VIEWS}
    result = {
        "schema_version": V2_SCHEMA_VERSION,
        "kind": "dandi_000688_subm_co_endpoint_aggregate_v2",
        "status": "ENDPOINT_PASS" if all(view_results[view]["view_pass"] for view in VIEWS) else "ENDPOINT_FAIL_GATE",
        "scope_id": SCOPE_ID,
        "N": EXPECTED_ELIGIBLE_SESSION_COUNT,
        "seed_order": list(SEEDS),
        "frozen_asset_id_order": [session.asset_id for session in contract.frozen_sessions],
        "sealed_evidence": evidence,
        "views": view_results,
        "primary_sua_pass": view_results["sua"]["view_pass"],
        "key_secondary_pseudo_mua_pass": view_results["pseudo_mua"]["view_pass"],
        "overall_mechanism_pass": bool(view_results["sua"]["view_pass"] and view_results["pseudo_mua"]["view_pass"]),
        "aggregate_recomputed_each_cell_from_npz": True,
    }
    aggregate_sha = _write_immutable_json_exclusive(aggregate_path, result)
    return {"status": result["status"], "overall_mechanism_pass": result["overall_mechanism_pass"], "aggregate": {"path": str(aggregate_path), "sha256": aggregate_sha}, "sealed_evidence": evidence}


def build_dry_run_plan_v2(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return only non-executing v2 checks and planned constraints."""

    counters = RuntimeAuditCounters()
    contract = validate_authority_chain_v2(root, counters=counters)
    query_window_counts = expected_query_window_counts_from_preflight(contract, root)
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "mode": "dry_run",
        "status": "NOT_AUTHORIZED_FOR_SCORING",
        "scope_id": SCOPE_ID,
        "frozen_N": len(contract.frozen_sessions),
        "candidate_arms": list(ARMS),
        "seeds": list(SEEDS),
        "sealed_cell_count": EXPECTED_SEALED_METRIC_COUNT,
        "query_windows_per_view_at_N15": sum(query_window_counts.values()),
        "total_model_windows_across_2x2x3_at_N15": sum(query_window_counts.values()) * len(VIEWS) * len(ARMS) * len(SEEDS),
        "device_support": "authorization-bound cpu or cuda:<index>; one exact device for all 180 cells",
        "root_ed25519_key_pin_configured": False,
        "scorer_adapter_parity_receipt_pin_configured": False,
        "scoring_will_not_run": True,
        "no_checkpoint_or_nwb_opened": True,
        "audit_counters": counters.snapshot(),
        "static_audit": static_runner_audit_v2(),
    }


def score_authorized_v2(
    *,
    authorization_path: Path,
    output_root: Path,
    prelaunch_dir: Path,
    nwb_asset_root: Path,
    requested_device: str,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Future score entrypoint; currently fail-closed before external work.

    It intentionally validates authorization before it even inspects the NWB
    directory.  Because v2 has neither a root public-key pin nor a parity
    receipt pin, it always fails before checkpoint/NWB/model/GPU activity.
    """

    del nwb_asset_root  # Explicitly prove this v2 delivery does not inspect it.
    counters = RuntimeAuditCounters()
    _contract, execution_device = require_future_authorization_v2(
        authorization_path=authorization_path,
        output_root=output_root,
        prelaunch_dir=prelaunch_dir,
        requested_device=requested_device,
        permitted_action="seal_session_predictions_metrics",
        root=root,
        output_root_must_be_new=True,
        counters=counters,
    )
    # This is intentionally after signature/parity authorization but before
    # any checkpoint/NWB path is opened.  The delivered v2 blockers make this
    # unreachable today; its placement is the future ordering constraint.
    validate_runtime_execution_device(execution_device)
    raise RuntimeFenceError("v2 authorization unexpectedly passed before a separately reviewed scorer-adapter execution implementation was appended")


def aggregate_authorized_v2(
    *,
    authorization_path: Path,
    output_root: Path,
    prelaunch_dir: Path,
    requested_device: str,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Future aggregate entrypoint; currently fail-closed before seal access."""

    counters = RuntimeAuditCounters()
    contract, _device = require_future_authorization_v2(
        authorization_path=authorization_path,
        output_root=output_root,
        prelaunch_dir=prelaunch_dir,
        requested_device=requested_device,
        permitted_action="aggregate_sealed_metrics",
        root=root,
        output_root_must_be_new=False,
        counters=counters,
    )
    return aggregate_sealed_v2(output_root, contract)
