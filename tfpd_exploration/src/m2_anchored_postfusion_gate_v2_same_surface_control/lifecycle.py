"""Attempt-first V2 immutable prefix lifecycle."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Callable, Mapping
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as immutable
from . import plan
def execute(repo_root: Path, *, attempt: Mapping[str,object], launch: Callable | None=None,
            bodies: Callable, terminal: Callable, progress: Callable, revalidate: Callable):
    class Spec:
        root_relative=plan.RESULT_ROOT_RELATIVE
        def payload(self): return {"schema":plan.SCHEMA,"root_relative":self.root_relative}
    artifact=immutable.ImmutableArtifactRoot.reserve(Path(repo_root).absolute(),Spec()); attempt_sha=artifact.publish_json('attempt.json',dict(attempt)); published={'attempt.json':attempt_sha}
    try:
        if launch is not None:
            launch_payload=dict(launch()); launch_payload.update({'schema':plan.SCHEMA+'_launch_v1','attempt_sha256':attempt_sha})
            published['launch.json']=artifact.publish_json('launch.json',launch_payload)
        published.update(dict(bodies(artifact))); revalidate(); payload=dict(terminal(published)); payload.update({"schema":plan.SCHEMA+'_terminal_v1','status':'TERMINAL','attempt_sha256':attempt_sha,'terminal_xor_failure':True})
        return artifact.publish_json('terminal.json',payload),None
    except BaseException as error:
        try: revalidate()
        except BaseException as check: recheck=f'{type(check).__name__}:{str(check)[:160]}'
        else: recheck=None
        failure={"schema":plan.SCHEMA+'_failure_v1','status':'FAILED','attempt_sha256':attempt_sha,'terminal_xor_failure':True,'exception_class':f'{type(error).__module__}.{type(error).__qualname__}','diagnostic_message':str(error)[:240],'error_sha256':hashlib.sha256(repr(error).encode()).hexdigest(),'published_prefix':published,'progress':dict(progress())}
        if recheck: failure['final_revalidation_error']=recheck
        return None,artifact.publish_json('failure.json',failure)
    finally: artifact.close()
