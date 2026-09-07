"""Immutable receipt lifecycles for the smoke, the two arms, and Phase 3.

Every stage is attempt -> launch -> bodies -> terminal under one fresh
``0444``+sidecar root, with attempt published before any NWB open, checkpoint
load, model construction, or forward, and an honest ``failure`` leaf on any
exception after the attempt (the m1_t0c1/cal_aug receipt law).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Callable, Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

from . import plan


class ReceiptError(RuntimeError):
    """Fail closed for stage receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _json_bytes(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class StageSpec:
    """Duck-typed root binding for ``v1.ImmutableArtifactRoot.reserve``."""

    root_relative: str

    def payload(self) -> dict[str, object]:
        return {"schema": "pit_m2_stage_root_v1", "root_relative": self.root_relative}


def assert_fresh(root: Path, relative: str) -> None:
    candidate = Path(root).absolute() / relative
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    raise ReceiptError(f"pit m2 stage root already exists: {relative}")


def ensure_lane_root(root: Path) -> Path:
    """Create (once) the lane's parent directory; never touches a stage root."""
    lane = Path(root).absolute() / plan.RESULT_ROOT_RELATIVE
    lane.mkdir(parents=True, exist_ok=True)
    return lane


def run_stage(
    root: Path, *, relative: str, attempt_payload: Mapping[str, object],
    launch_builder: Callable[[], Mapping[str, object]],
    body_publisher: Callable[[v1.ImmutableArtifactRoot], dict[str, str]],
    terminal_builder: Callable[[Mapping[str, str]], Mapping[str, object]],
    expected_terminal_names: Callable[[], tuple[str, ...]],
    progress: Callable[[], Mapping[str, object]] | None = None,
) -> tuple[dict[str, str], str | None, str | None]:
    """Generic one-stage lifecycle: attempt precedes every backend call."""
    assert_fresh(Path(root), relative)
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), StageSpec(relative))  # type: ignore[arg-type]
    attempt_sha = artifact.publish_json("attempt.json", dict(attempt_payload))
    try:
        launch_sha = artifact.publish_json("launch.json", dict(launch_builder()))
        shas = dict(body_publisher(artifact))
        terminal_payload = dict(terminal_builder(shas))
        terminal_payload["attempt_sha256"] = attempt_sha
        terminal_payload["launch_sha256"] = launch_sha
        terminal_sha = artifact.publish_json("terminal.json", terminal_payload)
        names = expected_terminal_names()
        artifact.validate_live(expected_names=names)
        return shas, terminal_sha, None
    except BaseException as error:
        failure_payload = {
            "schema": f"pit_m2_{Path(relative).name}_failure_v1",
            "status": "FAILED",
            "attempt_sha256": attempt_sha,
            "error_class": type(error).__name__,
            "error_repr": repr(error)[:2000],
            "error_sha256": _sha(repr(error).encode("utf-8")),
            "progress": dict(progress()) if callable(progress) else {},
            "terminal_published": False,
        }
        failure_sha = artifact.publish_json("failure.json", failure_payload)
        return {}, None, failure_sha
    finally:
        artifact.close()


def smoke_stage_names() -> tuple[str, ...]:
    bodies = ("attempt.json", "launch.json", "t0m_smoke.json", "c1m_smoke.json",
              "equality.json", "terminal.json")
    return tuple(item for body in bodies for item in (body, f"{body}.sha256"))


def arm_stage_names() -> tuple[str, ...]:
    bodies = ["attempt.json", "launch.json", "source_authority.json", "stream_head.json",
              "training.json", "checkpoint_best.pt", "checkpoint_last.pt",
              "checkpoint_manifest.json", "terminal.json"]
    bodies += [f"epoch_{index:02d}.json" for index in range(plan.EPOCHS)]
    return tuple(item for body in bodies for item in (body, f"{body}.sha256"))


def phase3_stage_names() -> tuple[str, ...]:
    """One score leaf per (arm, path, surface, session) -- never per surface
    alone (the attempt-1 duplicate-leaf collision)."""
    bodies = ["attempt.json", "launch.json"]
    for arm in plan.ARMS:
        for path in ("static", "fifo"):
            for surface in plan.PHASE3_SURFACES[path]:
                for session in plan.SURFACE_SESSION_ROSTERS[surface]:
                    bodies.append(f"score_{arm}_{path}_{surface}_{session}.json")
    bodies += ["table.json", "terminal.json"]
    names = tuple(item for body in bodies for item in (body, f"{body}.sha256"))
    _require(len(set(names)) == len(names),
             "pit m2 phase3 stage names contain duplicates (leaf collision)")
    return names


def stage_attempt_payload(stage: str, pair_spec_sha256: str,
                          closure: Mapping[str, object]) -> dict[str, object]:
    _require(stage in {"smoke", "t0m", "c1m", "phase3"} and pair_spec_sha256,
             "pit m2 stage attempt payload drift")
    return {
        "schema": "pit_m2_stage_attempt_v1",
        "cell": plan.CELL,
        "phase": plan.PHASE,
        "stage": stage,
        "status": "ATTEMPT_RESERVED",
        "pair_spec_sha256": _sha_of(pair_spec_sha256),
        "closure_sha256": closure.get("closure_sha256"),
        "trainer_binding": plan.TRAINER_BINDING,
        "cycle_law": plan.CYCLE_LAW,
        "dropout_proof": plan.DROPOUT_PROOF,
        "gates": plan.GATES,
        "factorial": {
            "cells": list(plan.FACTORIAL_CELLS),
            "sealed_cells_referenced_not_rerun": list(plan.SEALED_CELLS),
            "f00m_anchor_sha256": plan.F00M_ANCHOR["sha256"],
            "f01m_anchor_sha256": plan.F01M_ANCHOR["sha256"],
        },
        "sealed_smoke_terminal_sha256": plan.SEALED_SMOKE_TERMINAL_SHA256
        if stage in {"t0m", "c1m", "phase3"} else None,
        "deviations": plan.DEVIATIONS,
        "attempt_history_disclosure": dict(plan.PHASE3_ATTEMPT1_DISCLOSURE),
        "data_or_model_accessed": False,
        "stage_kind": ("source_training" if stage in {"smoke", "t0m", "c1m"}
                       else "metric_only_scoring"),
        "operator_resolution": plan.OPERATOR_RESOLUTION,
    }


def _sha_of(value: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, "sha literal drift")
    return value


__all__ = (
    "ReceiptError", "StageSpec", "assert_fresh", "ensure_lane_root", "run_stage",
    "smoke_stage_names",
    "arm_stage_names", "phase3_stage_names", "stage_attempt_payload",
)
