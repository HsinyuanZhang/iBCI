"""Small attempt-first immutable lifecycle for the new scorer."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import Callable, Mapping
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as immutable
from . import plan


def execute(repo_root: Path, *, attempt: Mapping[str, object], launch: Callable[[], Mapping[str, object]],
            bodies: Callable[[immutable.ImmutableArtifactRoot], Mapping[str, str]],
            terminal: Callable[[Mapping[str, str]], Mapping[str, object]], progress: Callable[[], Mapping[str, object]],
            root_relative: str = plan.RESULT_ROOT_RELATIVE, schema: str = plan.SCHEMA,
            include_failure_diagnostic: bool = False,
            failure_revalidate: Callable[[], None] | None = None):
    class Spec:
        def __init__(self, relative: str, route_schema: str):
            self.root_relative = relative; self._schema = route_schema
        def payload(self): return {"schema": f"{self._schema}_root_v1", "root_relative": self.root_relative}
    root = Path(repo_root).absolute()
    parent = root / "tfpd_exploration/results"
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError("checkpoint-score artifact parent is unavailable/symlinked")
    artifact = immutable.ImmutableArtifactRoot.reserve(root, Spec(root_relative, schema))
    attempt_sha = artifact.publish_json("attempt.json", dict(attempt))
    published: dict[str, str] = {"attempt.json": attempt_sha}
    try:
        launch_sha = artifact.publish_json("launch.json", dict(launch()))
        published["launch.json"] = launch_sha
        published.update(dict(bodies(artifact)))
        payload = dict(terminal(published)); payload.update({"schema": f"{schema}_terminal_v1", "status": "TERMINAL",
            "attempt_sha256": attempt_sha, "launch_sha256": launch_sha, "terminal_xor_failure": True})
        # ``terminal`` must complete all revalidation before this final
        # immutable publication.  No potentially-failing route operation is
        # performed after the terminal pair exists.
        terminal_sha = artifact.publish_json("terminal.json", payload)
        return terminal_sha, None
    except BaseException as error:
        # ``bodies`` may fail after publishing a valid immutable prefix but
        # before returning its SHA map.  Discover only complete known pairs;
        # do not delete any published evidence to manufacture a clean root.
        for name in ("input_authority.json", "score.json"):
            if ((artifact.path / name).is_file() and (artifact.path / f"{name}.sha256").is_file()
                    and name not in published):
                published[name] = hashlib.sha256((artifact.path / name).read_bytes()).hexdigest()
        failure_payload = {"schema": f"{schema}_failure_v1", "status": "FAILED",
            "attempt_sha256": attempt_sha, "terminal_xor_failure": True, "error_sha256": hashlib.sha256(repr(error).encode()).hexdigest(),
            "published_prefix": dict(published), "progress": dict(progress())}
        if failure_revalidate is not None:
            # A successor profile binds its historical predecessor both at
            # admission and at every terminal *or failure* publication.  A
            # failed revalidation must not suppress the honest original
            # failure receipt; it is recorded as a separate immutable fact.
            try:
                failure_revalidate()
            except BaseException as revalidation_error:
                failure_payload["failure_revalidation"] = {
                    "passed": False,
                    "error_sha256": hashlib.sha256(repr(revalidation_error).encode()).hexdigest(),
                }
            else:
                failure_payload["failure_revalidation"] = {"passed": True}
        if include_failure_diagnostic:
            failure_payload.update({"exception_class": f"{type(error).__module__}.{type(error).__qualname__}",
                                    "diagnostic_message": str(error)[:240]})
        return None, artifact.publish_json("failure.json", failure_payload)
    finally:
        artifact.close()
