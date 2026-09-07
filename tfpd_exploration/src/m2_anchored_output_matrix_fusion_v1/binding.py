"""Held-FD AOF V1 predecessor codec for AOF-M."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from . import plan


class BindingError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def _read(fd: int) -> bytes:
    blocks: list[bytes] = []
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            return b"".join(blocks)
        blocks.append(block)


def validate_aof_v1_graph(root: Path) -> dict[str, object]:
    """Validate the exact 14-leaf terminal graph with held parent FD."""
    nofollow, directory = getattr(os, "O_NOFOLLOW", 0), getattr(os, "O_DIRECTORY", 0)
    _need(isinstance(nofollow, int) and nofollow != 0 and isinstance(directory, int), "AOF-M FD flags unavailable")
    root = Path(root).absolute()
    parent_relative, name = os.path.split(plan.AOF_V1_ROOT_RELATIVE)
    parent_path = root / parent_relative
    parent_fd = os.open(parent_path, os.O_RDONLY | directory | nofollow)
    named_fd = None
    try:
        parent_before = os.fstat(parent_fd)
        named_fd = os.open(name, os.O_RDONLY | directory | nofollow, dir_fd=parent_fd)
        named = os.fstat(named_fd)
        _need(stat.S_ISDIR(named.st_mode), "AOF-M predecessor root type drift")
        expected = set(plan.AOF_V1_BODIES) | {f"{item}.sha256" for item in plan.AOF_V1_BODIES}
        actual = set(os.listdir(named_fd))
        _need(actual == expected, "AOF-M predecessor topology/extra leaf drift")
        bodies: dict[str, object] = {}
        for leaf, wanted_sha in plan.AOF_V1_BODIES.items():
            body_fd = os.open(leaf, os.O_RDONLY | nofollow, dir_fd=named_fd)
            side_fd = os.open(f"{leaf}.sha256", os.O_RDONLY | nofollow, dir_fd=named_fd)
            try:
                for descriptor, label in ((body_fd, leaf), (side_fd, f"{leaf}.sha256")):
                    info = os.fstat(descriptor)
                    _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
                          f"AOF-M predecessor {label} mode/link/type drift")
                body, side = _read(body_fd), _read(side_fd)
            finally:
                os.close(body_fd); os.close(side_fd)
            digest = hashlib.sha256(body).hexdigest()
            _need(digest == wanted_sha and side == f"{digest}  {leaf}\n".encode("ascii"),
                  f"AOF-M predecessor {leaf} body/sidecar drift")
            bodies[leaf] = json.loads(body.decode("utf-8"))
        _need(os.fstat(parent_fd).st_dev == parent_before.st_dev and os.fstat(parent_fd).st_ino == parent_before.st_ino,
              "AOF-M predecessor parent replacement")
    finally:
        if named_fd is not None:
            os.close(named_fd)
        os.close(parent_fd)
    attempt, launch = bodies["attempt.json"], bodies["launch.json"]
    source, paired = bodies["source_authority.json"], bodies["paired_output_authority.json"]
    fit, validation, terminal = (bodies[item] for item in ("fit.json", "validation.json", "terminal.json"))
    _need(attempt.get("schema") == "m2_anchored_output_fusion_v1_attempt"
          and attempt.get("closure_sha256") == plan.AOF_V1_CLOSURE_SHA256
          and launch.get("closure_sha256") == plan.AOF_V1_CLOSURE_SHA256
          and launch.get("uuid") == plan.GPU_UUID and launch.get("logical_device") == 0,
          "AOF-M predecessor attempt/launch semantics drift")
    _need(source.get("checkpoint_sha256") == "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
          and source.get("student_state_sha256") == "2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20"
          and source.get("parameter_updates") == 0 and source.get("target_parameter_updates") == 0,
          "AOF-M predecessor frozen-source semantics drift")
    _need(tuple(paired.get("session_evidence", {})) == plan.SESSIONS
          and paired.get("parameter_updates") == 0 and paired.get("target_parameter_updates") == 0,
          "AOF-M predecessor paired session/order semantics drift")
    for session in plan.SESSIONS:
        row = paired["session_evidence"][session]
        bridge = row.get("official_cpu_to_gpu_bridge")
        _need(row.get("zero_prediction_exact") is True and row.get("model_state_unchanged") is True
              and isinstance(bridge, dict)
              and float(bridge.get("identity_maxabs", float("inf"))) <= 2e-6
              and float(bridge.get("prediction_maxabs", float("inf"))) <= 2e-6
              and float(bridge.get("r2_abs_difference", float("inf"))) <= 2e-7
              and row.get("official_native_prediction_witness", {}).get("exact") is True,
              f"AOF-M predecessor {session} bridge/zero authority drift")
    _need(fit.get("objective") == "equal_session_mse_surrogate_not_direct_r2"
          and validation.get("passed") is False and terminal.get("schema") == "m2_anchored_output_fusion_v1_terminal"
          and terminal.get("status") == "TERMINAL" and terminal.get("attempt_sha256") == plan.AOF_V1_BODIES["attempt.json"]
          and terminal.get("closure_sha256") == terminal.get("final_closure_sha256") == plan.AOF_V1_CLOSURE_SHA256
          and terminal.get("all7_refit_performed") is False and terminal.get("terminal_xor_failure") is True
          and terminal.get("published") == {name: digest for name, digest in plan.AOF_V1_BODIES.items() if name != "terminal.json"},
          "AOF-M predecessor terminal semantics drift")
    return {"root_relative": plan.AOF_V1_ROOT_RELATIVE, "parent_device": int(parent_before.st_dev),
            "parent_inode": int(parent_before.st_ino), "named_device": int(named.st_dev),
            "named_inode": int(named.st_ino), "bodies": dict(plan.AOF_V1_BODIES),
            "closure_sha256": plan.AOF_V1_CLOSURE_SHA256, "paired_evidence": paired["session_evidence"]}
