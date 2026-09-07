"""Small immutable attempt/terminal-xor-failure lifecycle for the PF screen."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as immutable

from . import plan


class LifecycleError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


class _RootSpec:
    def __init__(self, relative: str) -> None:
        self.root_relative = relative

    def payload(self) -> dict[str, object]:
        return {"schema": "m2_postfusion_variant_screen_root_v1", "root_relative": self.root_relative}


def _expected(bodies: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for body in bodies for name in (body, f"{body}.sha256"))


def execute_stage(*, repo_root: Path, root_relative: str, attempt: Mapping[str, object],
                  launch: Callable[[], Mapping[str, object]],
                  bodies: Callable[[immutable.ImmutableArtifactRoot], Mapping[str, str]],
                  terminal: Callable[[Mapping[str, str], str, str], Mapping[str, object]],
                  progress: Callable[[], Mapping[str, object]]) -> tuple[str | None, str | None]:
    """Publish attempt first, then terminal XOR typed failure under a fresh root."""
    root = Path(repo_root).absolute()
    candidate = root / root_relative
    _require(candidate.parent.resolve().is_relative_to(root), "postfusion result path escapes repo")
    # Only the noncanonical lane parent is created here.  The stage root itself
    # remains fresh and is reserved atomically by ImmutableArtifactRoot.
    candidate.parent.mkdir(parents=True, exist_ok=True)
    _require(candidate.parent.is_dir() and not candidate.parent.is_symlink(),
             "postfusion artifact parent is unavailable/symlinked")
    try:
        candidate.lstat()
    except FileNotFoundError:
        pass
    else:
        raise LifecycleError("postfusion stage root is not fresh")
    artifact = immutable.ImmutableArtifactRoot.reserve(root, _RootSpec(root_relative))
    attempt_sha = artifact.publish_json("attempt.json", dict(attempt))
    try:
        launch_sha = artifact.publish_json("launch.json", dict(launch()))
        published = dict(bodies(artifact))
        # Everything that can be meaningfully revalidated must finish before
        # publishing the immutable terminal leaf.  In particular, never catch
        # a post-terminal validation exception and then append ``failure``.
        artifact.validate_live(expected_names=_expected(tuple(["attempt.json", "launch.json", *published.keys()])))
        payload = dict(terminal(published, attempt_sha, launch_sha))
        payload.update({"schema": f"{plan.SCHEMA}_terminal_v1", "status": "TERMINAL",
                        "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
                        "terminal_xor_failure": True})
        terminal_sha = artifact.publish_json("terminal.json", payload)
        return terminal_sha, None
    except BaseException as error:
        failure = {
            "schema": f"{plan.SCHEMA}_failure_v1", "status": "FAILED", "attempt_sha256": attempt_sha,
            "terminal_published": False, "terminal_xor_failure": True,
            "error_class": type(error).__name__, "error_sha256": hashlib.sha256(repr(error).encode()).hexdigest(),
            "progress": dict(progress()),
        }
        return None, artifact.publish_json("failure.json", failure)
    finally:
        artifact.close()


__all__ = ("LifecycleError", "execute_stage")
