"""Held-FD, fail-closed validation of the immutable PACD V1 failure graph."""
from __future__ import annotations
import hashlib, json, os, stat
from pathlib import Path
from . import plan

class PredecessorError(RuntimeError): pass
LEAVES = ("attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256")

def _read_fd(fd: int) -> bytes:
    chunks=[]
    while True:
        data=os.read(fd, 1 << 16)
        if not data: return b"".join(chunks)
        chunks.append(data)

def _body(dirfd: int, name: str, expected: str) -> tuple[bytes, dict]:
    st=os.stat(name, dir_fd=dirfd, follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o444:
        raise PredecessorError(f"invalid immutable predecessor leaf: {name}")
    fd=os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dirfd)
    try: body=_read_fd(fd)
    finally: os.close(fd)
    digest=hashlib.sha256(body).hexdigest()
    if digest != expected: raise PredecessorError(f"predecessor body SHA drift: {name}")
    return body, json.loads(body)

def validate_v1_predecessor(root: Path, relative: str = plan.PREDECESSOR_RELATIVE) -> dict[str, object]:
    path=Path(root) / relative
    flags=os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try: dirfd=os.open(path, flags)
    except OSError as error: raise PredecessorError(f"cannot held-open V1 predecessor: {error}") from error
    try:
        names=set(os.listdir(dirfd))
        if names != set(LEAVES): raise PredecessorError(f"V1 predecessor topology drift: {sorted(names)}")
        attempt_bytes, attempt=_body(dirfd,"attempt.json",plan.V1_ATTEMPT_SHA256)
        failure_bytes, failure=_body(dirfd,"failure.json",plan.V1_FAILURE_SHA256)
        for body_name, body_bytes in (("attempt.json",attempt_bytes),("failure.json",failure_bytes)):
            side_name=body_name+".sha256"
            st=os.stat(side_name,dir_fd=dirfd,follow_symlinks=False)
            if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode)!=0o444: raise PredecessorError(f"invalid sidecar: {side_name}")
            fd=os.open(side_name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=dirfd)
            try: side=_read_fd(fd).decode("utf-8")
            finally: os.close(fd)
            expected=f"{hashlib.sha256(body_bytes).hexdigest()}  {body_name}\n"
            if side != expected: raise PredecessorError(f"noncanonical sidecar: {side_name}")
        if attempt.get("source_closure",{}).get("closure_sha256") != plan.V1_CLOSURE_SHA256: raise PredecessorError("V1 closure drift")
        if attempt.get("source_closure",{}).get("files",{}).get("tfpd_exploration/docs/WORKORDER_PACD_V1_20260831.md",{}).get("sha256") != plan.V1_WORKORDER_SHA256: raise PredecessorError("V1 workorder drift")
        if failure.get("status")!="CELL_FAILED" or failure.get("attempt_sha256")!=plan.V1_ATTEMPT_SHA256 or failure.get("terminal_published") is not False or failure.get("target_access") is not False: raise PredecessorError("V1 failure semantics drift")
        if failure.get("no_target_facts",{}).get("source_roster_n") != 27: raise PredecessorError("V1 roster fact drift")
        detail=failure.get("failure",{}).get("detail",""); trace=failure.get("failure",{}).get("traceback","")
        if failure.get("failure",{}).get("kind")!="ValueError" or "uninitialized parameter" not in detail or "core.py" not in trace or "numel" not in trace: raise PredecessorError("V1 lazy-parameter failure lineage drift")
        return {"relative":relative,"attempt_sha256":plan.V1_ATTEMPT_SHA256,"failure_sha256":plan.V1_FAILURE_SHA256,"topology":list(LEAVES)}
    finally: os.close(dirfd)
