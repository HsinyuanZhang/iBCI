"""Immutable attempt-first AJPF lifecycle, independent of physical science code."""
from __future__ import annotations
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from . import plan

class LifecycleError(RuntimeError): pass
def _need(ok: bool, message: str) -> None:
    if not ok: raise LifecycleError(message)

def closure_map(repo_root: Path = plan.REPO_ROOT) -> dict[str, str]:
    plan.validate_static(repo_root)
    answer = {}
    for relative in plan.STATIC_CLOSURE_RELATIVES:
        path = repo_root / relative
        _need(path.is_file() and not path.is_symlink(), f"AJPF closure leaf missing/symlink: {relative}")
        answer[relative] = plan.sha256_file(path)
    return answer

def closure_sha256(mapping: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(mapping), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _pair(root: Path, name: str, body: Mapping[str, Any]) -> str:
    target, side = root / name, root / (name + ".sha256")
    _need(not os.path.lexists(target) and not os.path.lexists(side), f"AJPF immutable leaf exists: {name}")
    raw = (json.dumps(dict(body), sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    fd, temporary = tempfile.mkstemp(dir=root, prefix=f".{name}.")
    try:
        with os.fdopen(fd, "wb") as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444); os.replace(temporary, target)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    with side.open("x", encoding="ascii") as handle: handle.write(f"{digest}  {name}\n")
    os.chmod(side, 0o444)
    return digest

def _topology(root: Path, bodies: tuple[str, ...]) -> None:
    expected = set(bodies) | {name + ".sha256" for name in bodies}
    _need({x.name for x in root.iterdir()} == expected, "AJPF result topology drift")
    for name in bodies:
        body, side = root / name, root / (name + ".sha256")
        for path in (body, side):
            meta = path.stat(follow_symlinks=False)
            _need(stat.S_ISREG(meta.st_mode) and stat.S_IMODE(meta.st_mode) == 0o444 and meta.st_nlink == 1,
                  f"AJPF leaf mode/link drift: {name}")
        digest = hashlib.sha256(body.read_bytes()).hexdigest()
        _need(side.read_bytes() == f"{digest}  {name}\n".encode(), f"AJPF sidecar drift: {name}")


TRAINING_BODIES = ("attempt.json", "launch.json", "source_authority.json", "smoke.json",
                   *(f"epoch_{index:02d}.json" for index in range(1, plan.EPOCHS + 1)),
                   "manifest.json", "terminal.json")
SCORE_BODIES = ("attempt.json", "launch.json", "producer_authority.json", "input_authority.json",
                "score.json", "terminal.json")
CHECKPOINT_BODIES = tuple(f"{arm}_epoch12.pt" for arm in plan.ARM_ORDER)


def publish_binary(root: Path, relative: str, body: bytes) -> dict[str, str]:
    """Publish an immutable nonempty checkpoint body/sidecar atomically."""
    target = root / relative
    _need(body and target.parent.is_dir() and not target.parent.is_symlink(), "AJPF checkpoint body/parent drift")
    side = target.with_name(target.name + ".sha256")
    _need(not os.path.lexists(target) and not os.path.lexists(side), "AJPF immutable checkpoint already exists")
    digest = hashlib.sha256(body).hexdigest()
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444); os.replace(temporary, target)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {target.name}\n")
    os.chmod(side, 0o444)
    _validate_leaf(target, side)
    return {"path": relative, "sha256": digest, "sidecar": relative + ".sha256"}


def _validate_leaf(body: Path, side: Path) -> None:
    for path in (body, side):
        info = path.stat(follow_symlinks=False)
        _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
              f"AJPF immutable leaf mode/link drift: {path.name}")
    digest = hashlib.sha256(body.read_bytes()).hexdigest()
    _need(side.read_bytes() == f"{digest}  {body.name}\n".encode("ascii"), "AJPF immutable sidecar drift")


def validate_training_topology(root: Path, *, terminal: bool) -> None:
    """Exact successful/failure training graph, including its only subdirectory."""
    allowed = set(TRAINING_BODIES if terminal else TRAINING_BODIES[:-1])
    if not terminal:
        allowed.add("failure.json")
    expected = allowed | {name + ".sha256" for name in allowed} | {"checkpoints"}
    _need({item.name for item in root.iterdir()} == expected, "AJPF training root exact topology drift")
    checkpoint_dir = root / "checkpoints"
    info = checkpoint_dir.stat(follow_symlinks=False)
    _need(stat.S_ISDIR(info.st_mode) and not checkpoint_dir.is_symlink(), "AJPF checkpoint directory topology drift")
    expected_cp = set(CHECKPOINT_BODIES) | {name + ".sha256" for name in CHECKPOINT_BODIES}
    _need({item.name for item in checkpoint_dir.iterdir()} == expected_cp, "AJPF checkpoint leaf topology drift")
    for name in allowed:
        _validate_leaf(root / name, root / (name + ".sha256"))
    for name in CHECKPOINT_BODIES:
        _validate_leaf(checkpoint_dir / name, checkpoint_dir / (name + ".sha256"))
    _need((root / "terminal.json").exists() is terminal and (root / "failure.json").exists() is (not terminal),
          "AJPF training terminal/failure XOR drift")


def validate_score_topology(root: Path, *, terminal: bool) -> None:
    allowed = set(SCORE_BODIES if terminal else SCORE_BODIES[:-1])
    if not terminal:
        allowed.add("failure.json")
    expected = allowed | {name + ".sha256" for name in allowed}
    _need({item.name for item in root.iterdir()} == expected, "AJPF score root exact topology drift")
    for name in allowed:
        _validate_leaf(root / name, root / (name + ".sha256"))
    _need((root / "terminal.json").exists() is terminal and (root / "failure.json").exists() is (not terminal),
          "AJPF score terminal/failure XOR drift")


def validate_training_failure_topology(root: Path, *, published: tuple[str, ...], checkpoint_arms: tuple[str, ...]) -> None:
    """Failure may retain only complete ordered prefixes and sealed checkpoints."""
    receipt_prefix = tuple(name for name in TRAINING_BODIES[:-1] if name in published)
    _need(tuple(name for name in published if not name.startswith("checkpoints/")) == receipt_prefix,
          "AJPF training failure receipt prefix is noncanonical")
    expected = set(receipt_prefix) | {name + ".sha256" for name in receipt_prefix} | {"failure.json", "failure.json.sha256"}
    if checkpoint_arms:
        expected.add("checkpoints")
        expected_cp = {f"{arm}_epoch12.pt" for arm in checkpoint_arms}
        _need(tuple(checkpoint_arms) == plan.ARM_ORDER[:len(checkpoint_arms)], "AJPF checkpoint failure prefix drift")
    else:
        expected_cp = set()
    _need({item.name for item in root.iterdir()} == expected, "AJPF training failure root topology drift")
    for name in (*receipt_prefix, "failure.json"):
        _validate_leaf(root / name, root / (name + ".sha256"))
    if expected_cp:
        directory = root / "checkpoints"
        _need(directory.is_dir() and not directory.is_symlink(), "AJPF failure checkpoint directory drift")
        _need({item.name for item in directory.iterdir()} == expected_cp | {name + ".sha256" for name in expected_cp},
              "AJPF failure checkpoint topology drift")
        for name in expected_cp:
            _validate_leaf(directory / name, directory / (name + ".sha256"))


def validate_score_failure_topology(root: Path, *, published: tuple[str, ...]) -> None:
    prefix = tuple(name for name in SCORE_BODIES[:-1] if name in published)
    _need(tuple(published) == prefix, "AJPF score failure receipt prefix is noncanonical")
    expected = set(prefix) | {name + ".sha256" for name in prefix} | {"failure.json", "failure.json.sha256"}
    _need({item.name for item in root.iterdir()} == expected, "AJPF score failure root topology drift")
    for name in (*prefix, "failure.json"):
        _validate_leaf(root / name, root / (name + ".sha256"))

_TOKEN=object(); _USED:set[tuple[int,int,str]]=set()
@dataclass(frozen=True)
class Capability:
    token: object; root: Path; parent_dev: int; parent_ino: int; closure: dict[str,str]; closure_sha256: str

def _issue_capability(*, repo_root: Path, root: Path, reviewed: Mapping[str,str] | None=None, digest: str | None=None) -> Capability:
    current=closure_map(repo_root)
    if reviewed is not None or digest is not None:
        _need(reviewed is not None and digest is not None and dict(reviewed)==current and closure_sha256(current)==digest,
              "AJPF reviewed closure drift")
    _need(not os.path.lexists(root) and root.parent.is_dir() and not root.parent.is_symlink(), "AJPF fresh root/parent required")
    meta=root.parent.stat(follow_symlinks=False)
    return Capability(_TOKEN,root,meta.st_dev,meta.st_ino,current,closure_sha256(current))


def issue_production_capability(*, repo_root: Path, root: Path, reviewed: Mapping[str, str] | None = None,
                                digest: str | None = None) -> Capability:
    """Route-owned issuer; callers cannot mint an arbitrary test capability."""
    return _issue_capability(repo_root=repo_root, root=root, reviewed=reviewed, digest=digest)


def _issue_for_test(*, repo_root: Path, root: Path, reviewed: Mapping[str,str] | None=None,
                    digest: str | None=None) -> Capability:
    """Private test fixture issuer, deliberately absent from production routing."""
    return _issue_capability(repo_root=repo_root, root=root, reviewed=reviewed, digest=digest)


def _held_json(root: Path, name: str) -> tuple[dict[str, Any], str]:
    """Read a validated immutable JSON leaf through the already-held root."""
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        body_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            info = os.fstat(body_fd)
            _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
                  f"AJPF held training leaf mode/link drift: {name}")
            body = b"".join(iter(lambda: os.read(body_fd, 1 << 20), b""))
        finally:
            os.close(body_fd)
        digest = hashlib.sha256(body).hexdigest()
        side_fd = os.open(name + ".sha256", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            side_info = os.fstat(side_fd)
            _need(stat.S_ISREG(side_info.st_mode) and stat.S_IMODE(side_info.st_mode) == 0o444 and side_info.st_nlink == 1,
                  f"AJPF held training sidecar mode/link drift: {name}")
            side = b"".join(iter(lambda: os.read(side_fd, 1 << 20), b""))
        finally:
            os.close(side_fd)
        _need(side == f"{digest}  {name}\n".encode("ascii"), f"AJPF held training sidecar drift: {name}")
        value = json.loads(body.decode("utf-8"))
        _need(isinstance(value, dict), f"AJPF held training JSON schema drift: {name}")
        return value, digest
    finally:
        os.close(directory)


def _held_binary(root: Path, relative: str) -> tuple[bytes, str]:
    """Held-FD/no-follow verification for a nested sealed checkpoint leaf."""
    parent_name, leaf_name = relative.rsplit("/", 1)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parent_fd = os.open(parent_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            body_fd = os.open(leaf_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                info = os.fstat(body_fd)
                _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
                      "AJPF held checkpoint mode/link drift")
                body = b"".join(iter(lambda: os.read(body_fd, 1 << 20), b""))
            finally:
                os.close(body_fd)
            digest = hashlib.sha256(body).hexdigest()
            side_fd = os.open(leaf_name + ".sha256", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                side_info = os.fstat(side_fd)
                _need(stat.S_ISREG(side_info.st_mode) and stat.S_IMODE(side_info.st_mode) == 0o444 and side_info.st_nlink == 1,
                      "AJPF held checkpoint sidecar mode/link drift")
                side = b"".join(iter(lambda: os.read(side_fd, 1 << 20), b""))
            finally:
                os.close(side_fd)
            _need(side == f"{digest}  {leaf_name}\n".encode("ascii"), "AJPF held checkpoint sidecar drift")
            return body, digest
        finally:
            os.close(parent_fd)
    finally:
        os.close(root_fd)


def _held_names(root: Path, *, child: str | None = None) -> tuple[set[str], tuple[int, int]]:
    """Enumerate a directory only through a no-follow held descriptor."""
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = root_fd if child is None else os.open(child, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            info = os.fstat(fd)
            _need(stat.S_ISDIR(info.st_mode), "AJPF held directory type drift")
            return set(os.listdir(fd)), (int(info.st_dev), int(info.st_ino))
        finally:
            if fd != root_fd: os.close(fd)
    finally:
        os.close(root_fd)


def validate_held_training_success_graph(*, root: Path, repo_root: Path) -> dict[str, Any]:
    """Admit the completed producer before score-root reservation or loading."""
    names, before_identity = _held_names(root)
    expected = set(TRAINING_BODIES) | {name + ".sha256" for name in TRAINING_BODIES} | {"checkpoints"}
    _need(names == expected, "AJPF held training root exact topology drift")
    checkpoint_names, _checkpoint_identity = _held_names(root, child="checkpoints")
    expected_cp = set(CHECKPOINT_BODIES) | {name + ".sha256" for name in CHECKPOINT_BODIES}
    _need(checkpoint_names == expected_cp, "AJPF held checkpoint exact topology drift")
    attempt, attempt_sha = _held_json(root, "attempt.json")
    _launch, _launch_sha = _held_json(root, "launch.json")
    _source, _source_sha = _held_json(root, "source_authority.json")
    _smoke, _smoke_sha = _held_json(root, "smoke.json")
    terminal, terminal_sha = _held_json(root, "terminal.json")
    manifest, manifest_sha = _held_json(root, "manifest.json")
    _need(attempt.get("closure") == closure_map(repo_root) and attempt.get("closure_sha256") == closure_sha256(closure_map(repo_root)),
          "AJPF held training closure drift")
    _need(terminal.get("terminal_xor_failure") is True and terminal.get("attempt_sha256") == attempt_sha
          and terminal.get("launch_sha256") == _launch_sha and terminal.get("source_authority_sha256") == _source_sha
          and terminal.get("smoke_sha256") == _smoke_sha and terminal.get("manifest_sha256") == manifest_sha,
          "AJPF held training terminal/manifest link drift")
    _need(manifest.get("smoke_sha256") == _smoke_sha and manifest.get("fixed_epoch") == plan.EPOCHS
          and manifest.get("checkpoint_selection") == "epoch12_only_no_metric_selection",
          "AJPF held training manifest fixed-epoch/smoke drift")
    epochs = manifest.get("epoch_sha256")
    _need(isinstance(epochs, list) and len(epochs) == plan.EPOCHS, "AJPF held training epoch-link cardinality drift")
    for index, digest in enumerate(epochs, 1):
        _body, observed = _held_json(root, f"epoch_{index:02d}.json")
        _need(observed == digest, "AJPF held training epoch link drift")
    checkpoints = manifest.get("checkpoint")
    _need(isinstance(checkpoints, dict) and set(checkpoints) == set(plan.ARM_ORDER),
          "AJPF held training checkpoint manifest topology drift")
    for arm in plan.ARM_ORDER:
        descriptor = checkpoints[arm]
        _need(isinstance(descriptor, dict) and descriptor.get("path") == f"checkpoints/{arm}_epoch12.pt", "AJPF held checkpoint descriptor drift")
        body, observed = _held_binary(root, str(descriptor["path"]))
        _need(observed == descriptor.get("sha256"),
              "AJPF held checkpoint hash link drift")
        descriptor = dict(descriptor)
        proof = descriptor.get("proof")
        _need(isinstance(proof, Mapping) and proof.get("arm") == arm and proof.get("strict_load") is True
              and proof.get("eval_repeated_forward_bitwise") is True and proof.get("state_unchanged") is True
              and proof.get("forward_finite") is True, "AJPF held checkpoint proof drift")
        checkpoints[arm] = descriptor
        descriptor["_held_bytes"] = body
    _after_names, after_identity = _held_names(root)
    _need(before_identity == after_identity, "AJPF held training root swap")
    compact = {"root_identity": list(before_identity), "attempt_sha256": attempt_sha, "launch_sha256": _launch_sha,
               "source_authority_sha256": _source_sha, "smoke_sha256": _smoke_sha,
               "manifest_sha256": manifest_sha, "terminal_sha256": terminal_sha,
               "checkpoint_sha256": {arm: str(checkpoints[arm]["sha256"]) for arm in plan.ARM_ORDER}}
    return {"descriptor_witness": compact, **compact,
            "checkpoint": {arm: {key: value for key, value in checkpoints[arm].items() if key != "_held_bytes"}
                           for arm in plan.ARM_ORDER},
            "checkpoint_bytes": {arm: checkpoints[arm]["_held_bytes"] for arm in plan.ARM_ORDER}}

def consume(cap: Capability, repo_root: Path) -> None:
    _need(isinstance(cap,Capability) and cap.token is _TOKEN,"AJPF opaque capability required")
    key=(cap.parent_dev,cap.parent_ino,str(cap.root)); _need(key not in _USED,"AJPF capability consumed")
    meta=cap.root.parent.stat(follow_symlinks=False)
    _need((meta.st_dev,meta.st_ino)==(cap.parent_dev,cap.parent_ino) and not os.path.lexists(cap.root),"AJPF root parent/freshness drift")
    _need(closure_map(repo_root)==cap.closure,"AJPF closure drift before attempt"); _USED.add(key)

def execute_synthetic(*, cap: Capability, repo_root: Path, source: Mapping[str,Any], smoke: Mapping[str,Any],
                      epochs: tuple[Mapping[str,Any],...], manifest: Mapping[str,Any]) -> dict[str,Any]:
    """Typed production-shaped harness; real runner replaces only body factories."""
    consume(cap,repo_root); root=cap.root; root.mkdir(mode=0o755); published=[]
    try:
        attempt=_pair(root,"attempt.json",{"schema":"m2_anchored_joint_postfusion_v1_attempt","closure":cap.closure,"closure_sha256":cap.closure_sha256,"source_target_access":False,"target_access":False}); published.append("attempt.json")
        launch=_pair(root,"launch.json",{"schema":"m2_anchored_joint_postfusion_v1_launch","gpu_profile":"gpu0_only","target_access":False}); published.append("launch.json")
        source_sha=_pair(root,"source_authority.json",source); published.append("source_authority.json")
        smoke_sha=_pair(root,"smoke.json",smoke); published.append("smoke.json")
        _need(len(epochs)==plan.EPOCHS,"AJPF requires exact 12 epoch receipts")
        epoch_shas=[]
        for index, epoch in enumerate(epochs,1): epoch_shas.append(_pair(root,f"epoch_{index:02d}.json",epoch)); published.append(f"epoch_{index:02d}.json")
        manifest_sha=_pair(root,"manifest.json",manifest); published.append("manifest.json")
        _need(closure_map(repo_root)==cap.closure,"AJPF closure drift before terminal"); _topology(root,tuple(published))
        terminal=_pair(root,"terminal.json",{"schema":"m2_anchored_joint_postfusion_v1_terminal","attempt_sha256":attempt,"launch_sha256":launch,"source_authority_sha256":source_sha,"smoke_sha256":smoke_sha,"epoch_sha256":epoch_shas,"manifest_sha256":manifest_sha,"terminal_xor_failure":True}); published.append("terminal.json"); _topology(root,tuple(published)); return {"terminal_sha256":terminal,"root":str(root)}
    except Exception as error:
        if root.exists() and "terminal.json" not in published:
            failure=_pair(root,"failure.json",{"schema":"m2_anchored_joint_postfusion_v1_failure","published_prefix":published,"exception_class":type(error).__name__,"exception_message":str(error)[:400],"target_access":False}); _topology(root,tuple([*published,"failure.json"])); raise LifecycleError(f"AJPF failed: {failure}") from error
        raise
