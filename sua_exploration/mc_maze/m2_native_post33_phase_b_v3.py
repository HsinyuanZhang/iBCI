"""Fail-closed Phase-B contracts for the native-M2 post-33 program.

This module is deliberately score-free.  It defines cell ownership, explicit
source-only selection, paired-SPINT completion receipts, and decoder immutability
checks.  It does not launch training or open any formal evaluation endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
PHASE_ID = "PHASE_B_V3"
ARMS = ("spint", "t4")
SEEDS = (42, 43, 44)
FOLDS = {
    0: "ses-2020-10-19-Run1",
    1: "ses-2020-10-19-Run2",
    2: "ses-2020-10-20-Run1",
    3: "ses-2020-10-20-Run2",
    4: "ses-2020-10-27-Run1",
    5: "ses-2020-10-27-Run2",
    6: "ses-2020-10-28-Run1",
}
EXPECTED_SPINT_EPOCHS = tuple(range(35))
EXPECTED_DECODER_TENSORS = 31


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CellKey:
    protocol_id: str
    arm: str
    fold: int
    seed: int

    def __post_init__(self) -> None:
        if self.protocol_id != PROTOCOL_ID:
            raise ValueError(f"unexpected protocol_id {self.protocol_id!r}")
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS!r}")
        if isinstance(self.fold, bool) or self.fold not in FOLDS:
            raise ValueError("fold must be an integer in [0, 6]")
        if isinstance(self.seed, bool) or self.seed not in SEEDS:
            raise ValueError(f"seed must be one of {SEEDS!r}")

    @property
    def outer_session(self) -> str:
        return FOLDS[self.fold]

    @property
    def source_sessions(self) -> tuple[str, ...]:
        return tuple(session for fold, session in FOLDS.items() if fold != self.fold)

    @property
    def slug(self) -> str:
        return f"arm-{self.arm}/fold-{self.fold}/seed-{self.seed}"


def deterministic_cell_paths(root: str | Path, key: CellKey) -> dict[str, Path]:
    cell = Path(root).resolve() / PROTOCOL_ID / PHASE_ID / key.slug
    return {
        "cell_dir": cell,
        "owner": cell / "ownership.json",
        "started": cell / "status.started.json",
        "completed": cell / "status.completed.json",
        "failed": cell / "status.failed.json",
        "resolved_config": cell / "resolved_config.yaml",
        "selector_records": cell / "selector_records.json",
        "completion_receipt": cell / "spint_completion_receipt.json",
        "result": cell / "result.json",
        "hydra_metadata": cell / ".hydra",
        "checkpoints": cell / "checkpoints",
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write once using O_EXCL; a pre-existing target is a hard failure."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        data = _json_bytes(payload)
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return target


def claim_cell_ownership(root: str | Path, key: CellKey, *, owner_token: str) -> dict[str, Path]:
    if not owner_token or any(char.isspace() for char in owner_token):
        raise ValueError("owner_token must be a non-empty whitespace-free identifier")
    paths = deterministic_cell_paths(root, key)
    write_json_exclusive(
        paths["owner"],
        {
            "schema": "m2_post33_cell_ownership_v3",
            "protocol_id": key.protocol_id,
            "phase_id": PHASE_ID,
            "arm": key.arm,
            "fold": key.fold,
            "seed": key.seed,
            "outer_session": key.outer_session,
            "owner_token": owner_token,
        },
    )
    return paths


def write_status_exclusive(
    root: str | Path,
    key: CellKey,
    *,
    state: str,
    owner_token: str,
    details: Mapping[str, Any] | None = None,
) -> Path:
    if state not in {"started", "completed", "failed"}:
        raise ValueError("state must be started, completed, or failed")
    paths = deterministic_cell_paths(root, key)
    owner = json.loads(paths["owner"].read_text(encoding="utf-8"))
    if owner.get("owner_token") != owner_token:
        raise PermissionError("cell owner token mismatch")
    return write_json_exclusive(
        paths[state],
        {
            "schema": "m2_post33_cell_status_v3",
            "protocol_id": key.protocol_id,
            "phase_id": PHASE_ID,
            "arm": key.arm,
            "fold": key.fold,
            "seed": key.seed,
            "state": state,
            "owner_token": owner_token,
            "details": dict(details or {}),
        },
    )


def validate_source_selector_record(record: Mapping[str, Any], key: CellKey) -> None:
    required = {
        "epoch",
        "metric_name",
        "metric_value",
        "metric_scope",
        "source_sessions",
        "source_totals",
        "outer_session",
        "outer_total",
        "checkpoint_path",
    }
    if set(record) != required:
        raise ValueError(f"selector record keys differ: {set(record) ^ required}")
    epoch = record["epoch"]
    if isinstance(epoch, bool) or epoch not in EXPECTED_SPINT_EPOCHS:
        raise ValueError("selector epoch must be in [0, 34]")
    if record["metric_name"] != "val_source/r2_equal_session_mean":
        raise ValueError("selector metric name is not the v3 source-only metric")
    value = record["metric_value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("selector metric must be finite")
    if record["metric_scope"] != "exact_six_outer_train_source_sessions_only":
        raise ValueError("selector metric scope mismatch")
    if tuple(record["source_sessions"]) != key.source_sessions:
        raise ValueError("selector source session ordering/content mismatch")
    totals = record["source_totals"]
    if set(totals) != set(key.source_sessions):
        raise ValueError("selector source total keys mismatch")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 2 for value in totals.values()):
        raise ValueError("every source session must have integer total > 2")
    if record["outer_session"] != key.outer_session or record["outer_total"] != 0:
        raise ValueError("outer session must be audited with total=0 and excluded")
    checkpoint = Path(str(record["checkpoint_path"]))
    if checkpoint.name != f"epoch_{epoch:03d}.ckpt" or not checkpoint.is_absolute():
        raise ValueError("checkpoint_path must be canonical epoch_NNN.ckpt")


def select_source_checkpoint(
    records: Sequence[Mapping[str, Any]], key: CellKey, *, require_full_35: bool = True
) -> Mapping[str, Any]:
    if not records:
        raise ValueError("selector records are empty")
    for record in records:
        validate_source_selector_record(record, key)
    epochs = [int(record["epoch"]) for record in records]
    if len(epochs) != len(set(epochs)):
        raise ValueError("selector epochs must be unique")
    if require_full_35 and tuple(sorted(epochs)) != EXPECTED_SPINT_EPOCHS:
        raise ValueError("selector must contain exactly epochs 0..34")
    return max(records, key=lambda row: (float(row["metric_value"]), -int(row["epoch"])))


def _canonical_existing_file(path: str | Path) -> Path:
    raw = Path(path)
    canonical = raw.resolve(strict=True)
    if raw.is_symlink() or not canonical.is_file():
        raise ValueError(f"expected canonical regular file, got {raw}")
    return canonical


def build_spint_completion_receipt(
    *,
    key: CellKey,
    selector_records: Sequence[Mapping[str, Any]],
    resolved_config_path: str | Path,
) -> dict[str, Any]:
    if key.arm != "spint":
        raise ValueError("SPINT completion receipt requires arm='spint'")
    selected = select_source_checkpoint(selector_records, key, require_full_35=True)
    checkpoint = _canonical_existing_file(selected["checkpoint_path"])
    config = _canonical_existing_file(resolved_config_path)
    return {
        "schema": "m2_post33_spint_completion_receipt_v3",
        "protocol_id": key.protocol_id,
        "phase_id": PHASE_ID,
        "arm": "spint",
        "fold": key.fold,
        "seed": key.seed,
        "outer_session": key.outer_session,
        "source_sessions": list(key.source_sessions),
        "selected_epoch": int(selected["epoch"]),
        "selector_policy": "max_finite_equal_session_mean_then_earlier_epoch",
        "selector_records": [dict(record) for record in selector_records],
        "checkpoint": {
            "canonical_path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "resolved_config": {
            "canonical_path": str(config),
            "size_bytes": config.stat().st_size,
            "sha256": sha256_file(config),
        },
    }


def validate_spint_completion_receipt(receipt: Mapping[str, Any], key: CellKey) -> Path:
    if key.arm != "spint":
        raise ValueError("receipt lookup key must use arm='spint'")
    if receipt.get("schema") != "m2_post33_spint_completion_receipt_v3":
        raise ValueError("unsupported SPINT completion receipt schema")
    identity = (
        receipt.get("protocol_id"),
        receipt.get("phase_id"),
        receipt.get("arm"),
        receipt.get("fold"),
        receipt.get("seed"),
        receipt.get("outer_session"),
        tuple(receipt.get("source_sessions", [])),
    )
    expected = (
        key.protocol_id,
        PHASE_ID,
        "spint",
        key.fold,
        key.seed,
        key.outer_session,
        key.source_sessions,
    )
    if identity != expected:
        raise ValueError(f"SPINT completion receipt cell mismatch: {identity!r}")
    records = receipt.get("selector_records")
    if not isinstance(records, list):
        raise ValueError("selector_records must be a list")
    selected = select_source_checkpoint(records, key, require_full_35=True)
    if receipt.get("selected_epoch") != selected["epoch"]:
        raise ValueError("selected_epoch does not match explicit selector")
    checkpoint_meta = receipt.get("checkpoint")
    config_meta = receipt.get("resolved_config")
    for label, meta in (("checkpoint", checkpoint_meta), ("resolved_config", config_meta)):
        if not isinstance(meta, Mapping):
            raise ValueError(f"{label} metadata missing")
        path = _canonical_existing_file(meta.get("canonical_path", ""))
        if str(path) != meta.get("canonical_path"):
            raise ValueError(f"{label} path is not canonical")
        if path.stat().st_size != meta.get("size_bytes") or sha256_file(path) != meta.get("sha256"):
            raise ValueError(f"{label} bytes differ from receipt")
    checkpoint = Path(checkpoint_meta["canonical_path"])
    if checkpoint != Path(selected["checkpoint_path"]):
        raise ValueError("receipt checkpoint differs from selected record")
    if receipt.get("selector_policy") != "max_finite_equal_session_mean_then_earlier_epoch":
        raise ValueError("unexpected selector policy")
    return checkpoint


def resolve_paired_teacher_from_receipt(
    receipt_path: str | Path, *, fold: int, seed: int
) -> Path:
    receipt_file = _canonical_existing_file(receipt_path)
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    key = CellKey(PROTOCOL_ID, "spint", fold, seed)
    return validate_spint_completion_receipt(receipt, key)


def tensor_bytes(value: Any) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.numpy().tobytes(order="C")


def decoder_snapshot(decoder: Any, *, expected_count: int = EXPECTED_DECODER_TENSORS) -> dict[str, Any]:
    state = decoder.state_dict()
    names = list(state)
    if len(names) != expected_count:
        raise ValueError(f"decoder tensor count is {len(names)}, expected {expected_count}")
    rows = []
    for name in names:
        value = state[name]
        raw = tensor_bytes(value)
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "num_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    requires_grad = sorted(name for name, value in decoder.named_parameters() if value.requires_grad)
    return {
        "tensor_count": len(rows),
        "tensors": rows,
        "requires_grad_parameter_names": requires_grad,
    }


def compare_decoder_snapshots(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    if reference.get("tensor_count") != EXPECTED_DECODER_TENSORS:
        raise ValueError("reference decoder closure is not 31/31")
    if candidate.get("tensor_count") != EXPECTED_DECODER_TENSORS:
        raise ValueError("candidate decoder closure is not 31/31")
    if reference.get("tensors") != candidate.get("tensors"):
        raise ValueError("decoder key/name/shape/dtype/bytes closure changed")
    if candidate.get("requires_grad_parameter_names"):
        raise ValueError("decoder has requires_grad parameters")


def optimizer_decoder_intersection(optimizer: Any, decoder: Any) -> tuple[int, list[str]]:
    decoder_by_id = {id(parameter): name for name, parameter in decoder.named_parameters()}
    overlap = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in decoder_by_id:
                overlap.append(decoder_by_id[id(parameter)])
    return len(overlap), sorted(set(overlap))


def validate_decoder_lifecycle(
    snapshots: Mapping[str, Mapping[str, Any]], *, optimizer_intersection_count: int
) -> dict[str, Any]:
    expected_stages = ("pretrain", "posttrain", "reload", "prequery")
    if tuple(snapshots) != expected_stages:
        raise ValueError(f"decoder lifecycle stages must be {expected_stages!r}")
    reference = snapshots["pretrain"]
    for stage in expected_stages:
        compare_decoder_snapshots(reference, snapshots[stage])
    if optimizer_intersection_count != 0:
        raise ValueError("decoder parameters intersect optimizer")
    return {
        "tensor_count_expected": EXPECTED_DECODER_TENSORS,
        "tensor_count_compared": EXPECTED_DECODER_TENSORS,
        "stages": list(expected_stages),
        "bit_exact": True,
        "decoder_requires_grad_parameter_count": 0,
        "optimizer_intersection_count": 0,
        "decoder_updated_tensor_count": 0,
    }
