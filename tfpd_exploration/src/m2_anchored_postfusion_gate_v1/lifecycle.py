"""Attempt-first immutable lifecycle for APFG V1 production admission."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as immutable

from . import plan


def execute(repo_root: Path, *, attempt: Mapping[str, object], launch: Callable[[], Mapping[str, object]],
            bodies: Callable[[immutable.ImmutableArtifactRoot], Mapping[str, str]],
            terminal: Callable[[Mapping[str, str]], Mapping[str, object]],
            progress: Callable[[], Mapping[str, object]], failure_revalidate: Callable[[], None] | None = None):
    """Publish attempt before every runtime action; keep honest body prefixes."""
    class Spec:
        root_relative = plan.RESULT_ROOT_RELATIVE
        def payload(self): return {"schema": f"{plan.SCHEMA}_root_v1", "root_relative": self.root_relative}
    artifact = immutable.ImmutableArtifactRoot.reserve(Path(repo_root).absolute(), Spec())
    attempt_sha = artifact.publish_json("attempt.json", dict(attempt))
    published: dict[str, str] = {"attempt.json": attempt_sha}
    try:
        launch_sha = artifact.publish_json("launch.json", dict(launch()))
        published["launch.json"] = launch_sha
        published.update(dict(bodies(artifact)))
        payload = dict(terminal(published))
        payload.update({"schema": f"{plan.SCHEMA}_terminal_v1", "status": "TERMINAL", "attempt_sha256": attempt_sha,
                        "launch_sha256": launch_sha, "terminal_xor_failure": True})
        return artifact.publish_json("terminal.json", payload), None
    except BaseException as error:
        revalidation_error = None
        if failure_revalidate is not None:
            try:
                failure_revalidate()
            except BaseException as recheck:
                revalidation_error = f"{type(recheck).__module__}.{type(recheck).__qualname__}:{str(recheck)[:160]}"
        # Existing immutable pairs remain evidence; never delete to create a
        # cosmetically empty failure root.
        for name in ("source_authority.json", "alpha_selection.json", "input_authority.json", "score.json"):
            if (artifact.path / name).is_file() and (artifact.path / f"{name}.sha256").is_file() and name not in published:
                published[name] = hashlib.sha256((artifact.path / name).read_bytes()).hexdigest()
        failure = {"schema": f"{plan.SCHEMA}_failure_v1", "status": "FAILED", "attempt_sha256": attempt_sha,
                   "terminal_xor_failure": True, "error_sha256": hashlib.sha256(repr(error).encode()).hexdigest(),
                   "exception_class": f"{type(error).__module__}.{type(error).__qualname__}",
                   "diagnostic_message": str(error)[:240], "published_prefix": published,
                   "progress": dict(progress())}
        if revalidation_error is not None:
            failure["final_revalidation_error"] = revalidation_error
        return None, artifact.publish_json("failure.json", failure)
    finally:
        artifact.close()


__all__ = ("execute",)
