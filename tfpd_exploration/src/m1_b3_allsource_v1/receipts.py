"""Immutable receipt roots for the all-source B3 cell."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

from . import plan


class ReceiptError(RuntimeError):
    """Fail closed for successor receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class StageSpec:
    root_relative: str


def refuse_sealed_roots(path: Path | str) -> None:
    parts = Path(path).parts
    for sealed in plan.SEALED_RESULT_ROOTS:
        sealed_parts = Path(sealed).parts
        for index in range(0, len(parts) - len(sealed_parts) + 1):
            if parts[index:index + len(sealed_parts)] == sealed_parts:
                raise ReceiptError(f"refusing sealed result root {sealed}")


def run_stage0(
    root: Path,
    *,
    attempt_payload: Mapping[str, object],
    launch_builder: Callable[[], Mapping[str, object]],
    body_publisher: Callable[[v1.ImmutableArtifactRoot], dict[str, str]],
    terminal_builder: Callable[[Mapping[str, str]], Mapping[str, object]],
    relative: str = "stage0",
    progress: Callable[[], Mapping[str, object]] | None = None,
) -> tuple[dict[str, str], str | None, str | None]:
    refuse_sealed_roots(root)
    refuse_sealed_roots(Path(root) / relative)
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), StageSpec(relative))  # type: ignore[arg-type]
    attempt_sha = artifact.publish_json("attempt.json", dict(attempt_payload))
    try:
        launch_sha = artifact.publish_json("launch.json", dict(launch_builder()))
        shas = dict(body_publisher(artifact))
        published = {key: value for key, value in shas.items() if not str(key).startswith("_")}
        terminal_payload = dict(terminal_builder(shas))
        terminal_payload["attempt_sha256"] = attempt_sha
        terminal_payload["launch_sha256"] = launch_sha
        _require(terminal_payload.get("status") != "FAILED", "terminal cannot be FAILED")
        terminal_sha = artifact.publish_json("terminal.json", terminal_payload)
        leaves = set()
        for name in ("attempt.json", "launch.json", "terminal.json"):
            leaves.add(name)
            leaves.add(f"{name}.sha256")
        for name in published:
            leaves.add(name)
            if not str(name).endswith(".sha256"):
                leaves.add(f"{name}.sha256")
        artifact.validate_live(expected_names=tuple(sorted(leaves)))
        return published, terminal_sha, None
    except BaseException as error:
        failure_payload = {
            "schema": "m1_b3_allsource_failure_v1",
            "status": "FAILED",
            "attempt_sha256": attempt_sha,
            "error_class": type(error).__name__,
            "error_repr": repr(error)[:2000],
            "error_sha256": _sha(repr(error).encode("utf-8")),
            "progress": dict(progress()) if callable(progress) else {},
            "terminal_published": False,
            "target_optimizer_backward_update": 0,
        }
        failure_sha = artifact.publish_json("failure.json", failure_payload)
        return {}, None, failure_sha
    finally:
        artifact.close()
