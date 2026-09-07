"""Post-terminal evidence collation for the frozen sparse-event route.

This module does not train, score, select, or mutate a model.  It is invoked
only after the immutable Stage-2 and conditional Stage-3 aggregates exist.  It
re-materializes the reviewed source train/validation surfaces to close digest
fields that were held in memory by the GPU runners, validates every published
SHA sidecar, and emits one immutable final-evidence body.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from . import plan
from .core import array_sha256, require
from .lifecycle import publish_immutable_json
from .stage2_aggregate import _held_json


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sidecar_inventory(root: Path) -> list[dict[str, object]]:
    """Validate every sibling ``*.sha256`` leaf and return a stable inventory."""
    root = Path(root).resolve()
    rows: list[dict[str, object]] = []
    for sidecar in sorted(root.rglob("*.sha256")):
        parts = sidecar.read_text(encoding="ascii").strip().split()
        require(len(parts) == 2, f"malformed SHA sidecar: {sidecar}")
        expected, declared_name = parts
        body = sidecar.with_name(declared_name)
        require(body.parent == sidecar.parent and body.is_file(), f"missing SHA body: {sidecar}")
        actual = _sha256_file(body)
        require(actual == expected, f"SHA body drift: {body}")
        rows.append(
            {
                "body_relative": str(body.relative_to(root)),
                "sha256": actual,
                "size_bytes": int(body.stat().st_size),
                "sidecar_relative": str(sidecar.relative_to(root)),
            }
        )
    require(rows, "empty result SHA inventory")
    return rows


def parse_gpu_dmon(lines: Iterable[str]) -> dict[str, object]:
    """Summarize the documented partial-runtime physical-GPU0 dmon series."""
    samples: list[dict[str, object]] = []
    for raw in lines:
        parts = raw.split()
        if len(parts) != 17 or len(parts[0]) != 8 or not parts[0].isdigit():
            continue
        require(parts[2] == "0", "GPU dmon log includes a nonzero physical index")
        samples.append(
            {
                "date": parts[0],
                "time": parts[1],
                "power_w": float(parts[3]),
                "sm_percent": float(parts[6]),
                "memory_percent": float(parts[7]),
                "framebuffer_mb": float(parts[14]),
            }
        )
    require(samples, "GPU dmon series has no samples")

    def summary(name: str) -> dict[str, float]:
        values = np.asarray([float(row[name]) for row in samples], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    return {
        "physical_gpu_index": 0,
        "gpu_uuid": plan.PHYSICAL_GPU_UUIDS[0],
        "sampling_interval_seconds": 10,
        "coverage": "PARTIAL_RUNTIME_FROM_MONITOR_START__NOT_FULL_EXPERIMENT_RUNTIME",
        "sample_count": len(samples),
        "first_sample_local": f"{samples[0]['date']}T{samples[0]['time']}",
        "last_sample_local": f"{samples[-1]['date']}T{samples[-1]['time']}",
        "sm_percent": summary("sm_percent"),
        "memory_percent": summary("memory_percent"),
        "power_w": summary("power_w"),
        "framebuffer_mb": summary("framebuffer_mb"),
    }


def surface_evidence(surface) -> dict[str, object]:
    """Digest the exact prepared source surface without retaining raw arrays."""
    normalizers = {
        name: {
            "mean_sha256": array_sha256(np.asarray(values["mean"])),
            "scale_sha256": array_sha256(np.asarray(values["scale"])),
        }
        for name, values in sorted(surface.normalizers.items())
    }
    normalizers["parent_m30"] = {
        "mean_sha256": array_sha256(np.asarray(surface.parent_m30_normalizer["mean"])),
        "scale_sha256": array_sha256(np.asarray(surface.parent_m30_normalizer["scale"])),
    }
    sessions: dict[str, dict[str, object]] = {}
    for session_id, row in sorted(surface.sessions.items()):
        receipt = dict(row.receipt)
        sessions[session_id] = {
            "split": row.split,
            "activity_m30_sha256": array_sha256(np.asarray(row.record.calib_trials)),
            "q50_starts_sha256": array_sha256(np.asarray(row.q50_starts)),
            "parent_m30_t4_sha256": array_sha256(np.asarray(row.parent_m30_t4)),
            "whole_m10_raw_sha256": array_sha256(np.asarray(row.whole_m10_raw)),
            "post700_m10_raw_sha256": array_sha256(np.asarray(row.post700_m10_raw)),
            "whole_m10_parentnorm_sha256": array_sha256(np.asarray(row.whole_m10_parentnorm)),
            "whole_m10_t4_sha256": array_sha256(np.asarray(row.whole_m10_t4)),
            "post700_m10_t4_sha256": array_sha256(np.asarray(row.post700_m10_t4)),
            "profile_m10_sha256": array_sha256(np.asarray(row.profile_m10)),
            "descriptor_receipt": receipt,
        }
    require(sum(row["split"] == "train" for row in sessions.values()) == 27, "final evidence train roster drift")
    require(sum(row["split"] == "val" for row in sessions.values()) == 6, "final evidence validation roster drift")
    return {
        "view": surface.signal_view,
        "reliability_mask": list(surface.reliability_mask),
        "normalizers": normalizers,
        "sessions": sessions,
    }


def _successful_seed_receipts(result_parent: Path) -> dict[str, object]:
    """Bind every successful executable root and expose its score/training body."""
    output: dict[str, object] = {}
    for terminal_path in sorted(result_parent.rglob("terminal.json")):
        root = terminal_path.parent
        terminal_sha, terminal = _held_json(root, "terminal.json")
        if not isinstance(terminal, Mapping) or terminal.get("status") != "PASS":
            continue
        relative = str(root.relative_to(result_parent))
        row: dict[str, object] = {"terminal_sha256": terminal_sha, "terminal": terminal}
        for name in ("attempt.json", "training.json", "score.json"):
            if (root / name).is_file():
                digest, body = _held_json(root, name)
                row[name.removesuffix(".json")] = {"sha256": digest, "body": body}
        output[relative] = row
    require(output, "no successful seed receipts found")
    return output


def publish_final_evidence(repo_root: Path, *, gpu_dmon_path: Path) -> dict[str, object]:
    """Publish the immutable final evidence body after the causal graph closes."""
    repo_root = Path(repo_root).resolve()
    result_parent = repo_root / plan.RESULT_PARENT_RELATIVE
    final_root = repo_root / plan.FINAL_AGGREGATE_RELATIVE
    target = final_root / "final_evidence.json"
    require(not target.exists() and not target.with_name("final_evidence.json.sha256").exists(), "final evidence already exists")

    stage2_sha, stage2 = _held_json(repo_root / plan.STAGE2_V3_RELATIVE, "aggregate.json")
    stage3_sha, stage3 = _held_json(final_root, plan.STAGE3_PSEUDO_MUA_AGGREGATE_NAME)
    require(stage2.get("stage") == "stage2_v3_aggregate", "final evidence requires Stage2-v3")
    require(stage3.get("stage") == "stage3_pseudo_mua_aggregate", "final evidence requires Stage3 closure")

    from .admission import admit_stage0_graph
    from .production import prepare_source_surface

    stage0 = admit_stage0_graph(repo_root)
    mask = tuple(bool(value) for value in stage0["reliability_mask"])
    views = {"sua"}
    if any(value.get("status") != "NOT_ADMITTED" for value in stage3["branches"].values()):
        views.add("pseudo_mua")
    surfaces = {
        view: surface_evidence(prepare_source_surface(repo_root, signal_view=view, reliability_mask=mask))
        for view in sorted(views)
    }

    dmon_path = Path(gpu_dmon_path)
    gpu_summary = parse_gpu_dmon(dmon_path.read_text(encoding="utf-8").splitlines())
    payload = {
        "status": "PASS",
        "route": plan.ROUTE_NAME,
        "stage": "final_evidence",
        "scope": "SOURCE_27_TRAIN__SOURCE_6_VALIDATION__NO_TEST_EXTERNAL_EVALAI",
        "authorities": {
            "design_sha256": _sha256_file(repo_root / plan.DESIGN_RELATIVE),
            "workorder_sha256": _sha256_file(repo_root / plan.WORKORDER_RELATIVE),
            "manifest_sha256": _sha256_file(repo_root / plan.MANIFEST_RELATIVE),
            "stage2_v3_aggregate_sha256": stage2_sha,
            "stage3_aggregate_sha256": stage3_sha,
        },
        "stage0_admission": stage0,
        "stage2": stage2,
        "stage3": stage3,
        "surfaces": surfaces,
        "successful_seed_receipts": _successful_seed_receipts(result_parent),
        "gpu0_partial_runtime_monitor": gpu_summary,
        "sha_sidecar_inventory": validate_sidecar_inventory(result_parent),
    }
    require(payload["authorities"]["design_sha256"] == plan.DESIGN_SHA256, "design SHA drift at final evidence")
    require(payload["authorities"]["workorder_sha256"] == plan.WORK_ORDER_SHA256, "work-order SHA drift at final evidence")
    require(payload["authorities"]["manifest_sha256"] == plan.MANIFEST_SHA256, "manifest SHA drift at final evidence")
    digest = publish_immutable_json(final_root, "final_evidence.json", payload)
    return {"final_evidence_sha256": digest, **payload}
