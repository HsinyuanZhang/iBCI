"""Immutable Stage-0 receipt lifecycle. Failure and terminal are exclusive."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Callable, Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

from . import plan


class ReceiptError(RuntimeError):
    """Fail closed for Stage-0 receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class StageSpec:
    root_relative: str


def run_stage0(
    root: Path,
    *,
    attempt_payload: Mapping[str, object],
    launch_builder: Callable[[], Mapping[str, object]],
    body_publisher: Callable[[v1.ImmutableArtifactRoot], dict[str, str]],
    terminal_builder: Callable[[Mapping[str, str]], Mapping[str, object]],
    relative: str = "stage0",
) -> tuple[dict[str, str], str | None, str | None]:
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), StageSpec(relative))  # type: ignore[arg-type]
    attempt_sha = artifact.publish_json("attempt.json", dict(attempt_payload))
    try:
        launch_sha = artifact.publish_json("launch.json", dict(launch_builder()))
        shas = dict(body_publisher(artifact))
        terminal_payload = dict(terminal_builder(shas))
        terminal_payload["attempt_sha256"] = attempt_sha
        terminal_payload["launch_sha256"] = launch_sha
        _require(terminal_payload.get("status") != "FAILED", "terminal cannot be FAILED")
        terminal_sha = artifact.publish_json("terminal.json", terminal_payload)
        names = tuple(sorted(
            list(shas) + [
                "attempt.json", "attempt.json.sha256",
                "launch.json", "launch.json.sha256",
                "terminal.json", "terminal.json.sha256",
            ]
        ))
        # shas keys are body names; validate_live wants all leaf names.
        leaves = set()
        for name in ("attempt.json", "launch.json", "terminal.json"):
            leaves.add(name)
            leaves.add(f"{name}.sha256")
        for name in shas:
            leaves.add(name)
            if not name.endswith(".sha256"):
                leaves.add(f"{name}.sha256")
        artifact.validate_live(expected_names=tuple(sorted(leaves)))
        return shas, terminal_sha, None
    except BaseException as error:
        failure_payload = {
            "schema": "m1_emg_syn3_stage0_failure_v1",
            "status": "FAILED",
            "attempt_sha256": attempt_sha,
            "error_class": type(error).__name__,
            "error_repr": repr(error)[:2000],
            "error_sha256": _sha(repr(error).encode("utf-8")),
            "terminal_published": False,
            "source_accessed": True,
            "target_optimizer_backward_update": 0,
            "cuda_initialized": False,
        }
        failure_sha = artifact.publish_json("failure.json", failure_payload)
        return {}, None, failure_sha
    finally:
        artifact.close()
