"""Write-once 31/31 decoder lifecycle evidence for Phase-C T4."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    _require_synthetic_test_fixture,
)


STAGES = ("pretrain", "posttrain", "reload", "prequery")
EXPECTED_TENSORS = 31


def _tensor_bytes(value: Any) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes(order="C")


def decoder_snapshot(decoder: Any) -> dict[str, Any]:
    state = decoder.state_dict()
    if len(state) != EXPECTED_TENSORS:
        raise ValueError(f"decoder has {len(state)} state tensors, expected 31")
    rows = []
    for name, value in state.items():
        raw = _tensor_bytes(value)
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


def optimizer_intersection(optimizer: Any | None, decoder: Any) -> list[str]:
    if optimizer is None:
        return []
    decoder_names = {id(parameter): name for name, parameter in decoder.named_parameters()}
    return sorted(
        {
            decoder_names[id(parameter)]
            for group in optimizer.param_groups
            for parameter in group["params"]
            if id(parameter) in decoder_names
        }
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        cursor = 0
        while cursor < len(data):
            cursor += os.write(fd, data[cursor:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def _require_synthetic_fixture_containment(
    fixture_root: str | Path | None,
    *paths: str | Path,
) -> Path:
    """Gate a relaxed lifecycle helper before it can create any artifact."""
    if fixture_root is None:
        raise PermissionError("synthetic lifecycle path requires a tests-only fixture root")
    root = Path(fixture_root).resolve()
    _require_synthetic_test_fixture(root)
    for raw_path in paths:
        path = Path(raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PermissionError(
                "synthetic lifecycle paths must remain inside the registered fixture root"
            ) from exc
    return root


def _write_stage_exclusive(
    stage_dir: str | Path,
    *,
    stage: str,
    decoder: Any,
    optimizer: Any | None,
    cell_identity: Mapping[str, Any],
    synthetic_proof: bool,
    synthetic_fixture_root: str | Path | None = None,
) -> Path:
    if synthetic_proof:
        _require_synthetic_fixture_containment(synthetic_fixture_root, stage_dir)
    if stage not in STAGES:
        raise ValueError(f"unknown decoder lifecycle stage {stage!r}")
    snapshot = decoder_snapshot(decoder)
    if snapshot["requires_grad_parameter_names"]:
        raise ValueError("decoder has requires_grad parameters")
    overlap = optimizer_intersection(optimizer, decoder)
    if overlap:
        raise ValueError(f"decoder intersects optimizer: {overlap}")
    return _write_json_exclusive(
        Path(stage_dir).resolve() / f"{stage}.json",
        {
            "schema": "m2_post33_decoder_lifecycle_stage_v4",
            **dict(cell_identity),
            "stage": stage,
            "synthetic_proof": bool(synthetic_proof),
            "optimizer_intersection_names": overlap,
            "snapshot": snapshot,
        },
    )


def write_stage_exclusive(
    stage_dir: str | Path,
    *,
    stage: str,
    decoder: Any,
    optimizer: Any | None,
    cell_identity: Mapping[str, Any],
) -> Path:
    """Production stage writer: lifecycle records are always non-synthetic."""
    return _write_stage_exclusive(
        stage_dir,
        stage=stage,
        decoder=decoder,
        optimizer=optimizer,
        cell_identity=cell_identity,
        synthetic_proof=False,
    )


def _finalize_lifecycle_evidence(
    stage_dir: str | Path,
    output_path: str | Path,
    *,
    cell_identity: Mapping[str, Any],
    allow_synthetic: bool = False,
    synthetic_fixture_root: str | Path | None = None,
) -> Path:
    if allow_synthetic:
        _require_synthetic_fixture_containment(
            synthetic_fixture_root, stage_dir, output_path
        )
    directory = Path(stage_dir).resolve(strict=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("decoder lifecycle stage directory is invalid")
    if {path.name for path in directory.iterdir()} != {f"{stage}.json" for stage in STAGES}:
        raise ValueError("decoder lifecycle stage exact set mismatch")
    rows = []
    for stage in STAGES:
        path = directory / f"{stage}.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("decoder lifecycle stage must be a regular file")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema") != "m2_post33_decoder_lifecycle_stage_v4" or row.get("stage") != stage:
            raise ValueError("decoder lifecycle stage schema/order mismatch")
        for field, expected in cell_identity.items():
            if row.get(field) != expected:
                raise ValueError("decoder lifecycle stage cell substitution")
        rows.append(row)
    synthetic = any(row["synthetic_proof"] for row in rows)
    if synthetic and not allow_synthetic:
        raise ValueError("synthetic lifecycle stages cannot finalize production evidence")
    snapshots = {row["stage"]: row["snapshot"] for row in rows}
    reference = snapshots["pretrain"]
    if reference["tensor_count"] != EXPECTED_TENSORS:
        raise ValueError("decoder lifecycle reference is not 31/31")
    for stage in STAGES:
        if snapshots[stage] != reference:
            raise ValueError(f"decoder changed at lifecycle stage {stage}")
        if rows[STAGES.index(stage)]["optimizer_intersection_names"]:
            raise ValueError("decoder entered optimizer")
    evidence = {
        "schema": "m2_post33_decoder_lifecycle_evidence_v4",
        **dict(cell_identity),
        "synthetic_proof": synthetic,
        "stages": list(STAGES),
        "tensor_count": EXPECTED_TENSORS,
        "requires_grad_parameter_count": 0,
        "optimizer_intersection_count": 0,
        "updated_tensor_count": 0,
        "bit_exact": True,
        "snapshots": snapshots,
    }
    return _write_json_exclusive(Path(output_path).resolve(), evidence)


def finalize_lifecycle_evidence(
    stage_dir: str | Path,
    output_path: str | Path,
    *,
    cell_identity: Mapping[str, Any],
) -> Path:
    """Production finalizer: synthetic decoder lifecycle evidence is forbidden."""
    return _finalize_lifecycle_evidence(
        stage_dir,
        output_path,
        cell_identity=cell_identity,
        allow_synthetic=False,
    )
