"""Pre-Torch immutable Stage-0 graph and mask-authority admission checks."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from . import plan
from .core import require


def _held_bytes(path: Path) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"held leaf is not regular: {path}")
    require(stat.S_IMODE(before.st_mode) == 0o444, f"held leaf mode drift: {path}")
    require(before.st_nlink == 1, f"held leaf link-count drift: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        held = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mode, before.st_nlink)
        require(
            (held.st_dev, held.st_ino, held.st_size, held.st_mode, held.st_nlink) == identity,
            f"held leaf changed during open: {path}",
        )
        parts: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            parts.append(chunk)
        after = os.fstat(descriptor)
        require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mode, after.st_nlink) == identity,
            f"held leaf changed while reading: {path}",
        )
        return b"".join(parts)
    finally:
        os.close(descriptor)


def _held_graph(root: Path, bodies: tuple[str, str, str]) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    expected = {name for name in bodies} | {f"{name}.sha256" for name in bodies}
    require(root.is_dir(), f"held graph root unavailable: {root}")
    require({path.name for path in root.iterdir()} == expected, f"held graph topology drift: {root}")
    digests: dict[str, str] = {}
    decoded: dict[str, dict[str, object]] = {}
    for name in bodies:
        body = _held_bytes(root / name)
        digest = hashlib.sha256(body).hexdigest()
        sidecar = _held_bytes(root / f"{name}.sha256").decode("ascii").strip().split()
        require(sidecar == [digest, name], f"held sidecar framing drift: {root / name}")
        parsed = json.loads(body)
        require(isinstance(parsed, dict), f"held JSON body is not an object: {root / name}")
        digests[name] = digest
        decoded[name] = parsed
    return digests, decoded


def _authority_mask(path: Path) -> tuple[tuple[bool, bool, bool, bool], str]:
    body_bytes = _held_bytes(path)
    digest = hashlib.sha256(body_bytes).hexdigest()
    require(digest == plan.MASK_AUTHORITY_SHA256, "mask authority SHA drift")
    body = body_bytes.decode("utf-8")
    match = re.search(
        r"reliability_mask\s*:\s*\[\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)\s*\]",
        body,
        flags=re.IGNORECASE,
    )
    require(match is not None, "mask authority has no exact four-bit vector")
    return tuple(value.lower() == "true" for value in match.groups()), digest


def admit_stage0_graph(repo_root: Path) -> dict[str, object]:
    """Read only small frozen receipts/docs; no Torch, NWB, checkpoint, or CUDA."""
    repo_root = Path(repo_root).resolve()
    primary_root = repo_root / plan.STAGE0_RELATIVE
    primary_digests, primary = _held_graph(
        primary_root, ("attempt.json", "stage0.json", "terminal.json")
    )
    attempt = primary["attempt.json"]
    stage = primary["stage0.json"]
    terminal = primary["terminal.json"]
    require(attempt.get("status") == "STARTED", "Stage-0 attempt status drift")
    require(terminal.get("status") == "PASS", "Stage-0 did not terminally pass")
    require(terminal.get("attempt_sha256") == primary_digests["attempt.json"], "Stage-0 terminal attempt link drift")
    require(terminal.get("stage0_sha256") == primary_digests["stage0.json"], "Stage-0 terminal payload link drift")
    require(stage.get("route") == plan.ROUTE_NAME, "Stage-0 route drift")
    require(stage.get("gpu_opened") is False and stage.get("decoder_opened") is False, "Stage-0 access contract drift")
    require(stage.get("performance_metric_opened") is False, "Stage-0 performance surface was opened")
    static = attempt.get("static_inputs")
    require(
        static == {
            plan.DESIGN_RELATIVE: plan.DESIGN_SHA256,
            plan.MANIFEST_RELATIVE: plan.MANIFEST_SHA256,
            plan.WORKORDER_RELATIVE: plan.WORKORDER_SHA256,
        },
        "Stage-0 frozen input binding drift",
    )

    mask = tuple(bool(value) for value in stage["decision"]["reliability_mask"])
    require(len(mask) == 4, "Stage-0 mask width drift")

    supplement_root = repo_root / plan.STAGE0_SUPPLEMENT_RELATIVE
    supplement_digests, supplement = _held_graph(
        supplement_root, ("attempt.json", "supplement.json", "terminal.json")
    )
    supplement_attempt = supplement["attempt.json"]
    supplement_body = supplement["supplement.json"]
    supplement_terminal = supplement["terminal.json"]
    require(supplement_attempt.get("status") == "STARTED", "supplement attempt status drift")
    require(supplement_terminal.get("status") == "PASS", "supplement did not terminally pass")
    require(supplement_terminal.get("attempt_sha256") == supplement_digests["attempt.json"], "supplement attempt link drift")
    require(supplement_terminal.get("supplement_sha256") == supplement_digests["supplement.json"], "supplement payload link drift")
    require(supplement_terminal.get("primary_stage0_sha256") == primary_digests["stage0.json"], "supplement primary link drift")
    require(supplement_terminal.get("primary_terminal_sha256") == primary_digests["terminal.json"], "supplement primary terminal link drift")
    require(supplement_body.get("mask_or_gate_mutated") is False, "supplement mutated a Stage-0 decision")
    cross = supplement_body.get("reliability_cross_check", {})
    require(cross.get("matches_primary") is True, "supplement reliability cross-check absent")
    require(tuple(bool(value) for value in cross.get("mask", ())) == mask, "supplement mask disagreement")

    authority_mask, authority_sha = _authority_mask(repo_root / plan.MASK_AUTHORITY_RELATIVE)
    require(mask == authority_mask, "Stage-0 receipt/mask authority disagreement")
    require(stage["decision"].get("estimator_route") == "OPEN", "frozen estimator route is not open")
    require(stage["decision"].get("film_route") == "OPEN", "frozen FiLM route is not open")

    return {
        "stage0_root": str(primary_root),
        "attempt_sha256": primary_digests["attempt.json"],
        "stage0_sha256": primary_digests["stage0.json"],
        "terminal_sha256": primary_digests["terminal.json"],
        "supplement_root": str(supplement_root),
        "supplement_attempt_sha256": supplement_digests["attempt.json"],
        "supplement_sha256": supplement_digests["supplement.json"],
        "supplement_terminal_sha256": supplement_digests["terminal.json"],
        "mask_authority_relative": plan.MASK_AUTHORITY_RELATIVE,
        "mask_authority_sha256": authority_sha,
        "reliability_mask": list(mask),
        "estimator_route": stage["decision"]["estimator_route"],
        "film_route": stage["decision"]["film_route"],
        "stage0_attempt_status": attempt["status"],
    }


def admit_stage1_successor(repo_root: Path) -> dict[str, object]:
    """Admit the additive Stage-1 V3 roots without opening Torch/data/CUDA."""
    repo_root = Path(repo_root).resolve()
    stage0 = admit_stage0_graph(repo_root)
    incident_bytes = _held_bytes(repo_root / plan.STAGE1_SUCCESSOR_INCIDENT_RELATIVE)
    incident_sha = hashlib.sha256(incident_bytes).hexdigest()
    require(
        incident_sha == plan.STAGE1_SUCCESSOR_INCIDENT_SHA256,
        "Stage-1 successor incident authority SHA drift",
    )
    v3_incident_bytes = _held_bytes(repo_root / plan.STAGE1_V3_INCIDENT_RELATIVE)
    v3_incident_sha = hashlib.sha256(v3_incident_bytes).hexdigest()
    require(v3_incident_sha == plan.STAGE1_V3_INCIDENT_SHA256, "Stage-1 V3 incident authority SHA drift")

    def failure_witnesses(expected_set, *, version: str, exception_type: str, message: str) -> dict[str, object]:
        witnesses: dict[str, object] = {}
        for seed, expected in sorted(expected_set.items()):
            digests, bodies = _held_graph(
                repo_root / expected["root"], ("attempt.json", "failure.json")
            )
            require(
                digests["attempt.json"] == expected["attempt_sha256"]
                and digests["failure.json"] == expected["failure_sha256"],
                f"Stage-1 {version} seed{seed} failure witness SHA drift",
            )
            failure = bodies["failure.json"]
            require(failure.get("status") == "FAIL", f"Stage-1 {version} seed{seed} is not failed")
            require(
                failure.get("attempt_sha256") == digests["attempt.json"],
                f"Stage-1 {version} seed{seed} failure attempt link drift",
            )
            require(
                failure.get("exception_type") == exception_type
                and failure.get("message") == message
                and failure.get("published_prefix") == ["attempt.json", "attempt.json.sha256"],
                f"Stage-1 {version} seed{seed} failure semantics drift",
            )
            witnesses[str(seed)] = {
                "root": expected["root"],
                "attempt_sha256": digests["attempt.json"],
                "failure_sha256": digests["failure.json"],
            }
        return witnesses

    v1_witnesses = failure_witnesses(
        plan.STAGE1_V1_FAILURE_WITNESSES,
        version="V1",
        exception_type="KeyboardInterrupt",
        message="",
    )
    v2_witnesses = failure_witnesses(
        plan.STAGE1_V2_FAILURE_WITNESSES,
        version="V2",
        exception_type="ModuleNotFoundError",
        message="No module named 'src.models'",
    )
    return {
        "stage0_admission": stage0,
        "successor_incident_relative": plan.STAGE1_SUCCESSOR_INCIDENT_RELATIVE,
        "successor_incident_sha256": incident_sha,
        "v3_incident_relative": plan.STAGE1_V3_INCIDENT_RELATIVE,
        "v3_incident_sha256": v3_incident_sha,
        "v1_failure_witnesses": v1_witnesses,
        "v2_failure_witnesses": v2_witnesses,
        "correction": "ONE_TEMPLATE_DEEPCOPIED_TO_FOUR_FILM_ARMS",
        "production_pythonpath": plan.PRODUCTION_PYTHONPATH,
    }
