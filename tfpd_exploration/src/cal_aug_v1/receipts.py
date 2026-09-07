"""CAL-AUG receipt and runtime-guard helpers beyond ``tfpd_lane.receipt``.

Everything here is additive route surface; the sealed transactional writer
(``tfpd_lane.receipt.write_receipt_transactionally``) is reused verbatim.

* ``verify_pinned_files`` / ``load_pinned_module`` — SHA-256 verification of
  the sealed trainers against ``plan.PINNED_SHA256`` and importlib loading,
  fail-closed BEFORE any data or model access;
* ``gpu_binding`` — the device/UUID binding via ``nvidia-smi`` at launch
  (work order section 4);
* ``publish_attempt`` / ``require_attempt`` — the attempt receipt that must
  exist before any source/model/data access, plus the guard that enforces
  that ordering;
* ``CalAugTimeout`` / ``DeadlineGuard`` / ``DeadlineLoader`` — the hard
  wall-clock guard with per-batch granularity and an atomic ``CELL_FAILED``
  receipt on breach (stage + steps completed + ``terminal_published=false``);
* ``probe_payload`` — the in-run throughput probe arithmetic (steps/s,
  projected per-arm seconds, pair GPU-hours vs the planning ceiling).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from . import plan

CELL_FAILED = "CELL_FAILED"
ATTEMPT_FILE = "attempt.json"
TERMINAL_FILE = "terminal.json"


class ReceiptError(RuntimeError):
    pass


class CalAugTimeout(RuntimeError):
    """Hard wall-clock breach; always published as an atomic CELL_FAILED."""

    def __init__(self, stage: str, steps_completed: int, elapsed_s: float, limit_s: float):
        super().__init__(
            f"CAL-AUG hard wall-clock timeout after {elapsed_s:.1f}s >= {limit_s:.1f}s "
            f"at stage {stage!r} with {steps_completed} optimizer steps completed"
        )
        self.stage = stage
        self.steps_completed = int(steps_completed)
        self.elapsed_s = float(elapsed_s)
        self.limit_s = float(limit_s)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# sealed boundary: pinned literals, verified before anything else
# ---------------------------------------------------------------------------


def verify_pinned_files(repo_root: Path) -> dict[str, str]:
    """Verify every ``plan.PINNED_SHA256`` literal against the live bytes."""
    base = Path(repo_root).absolute()
    table: dict[str, str] = {}
    for relative, expected in plan.PINNED_SHA256.items():
        path = base / relative
        _require(path.is_file(), f"pinned file missing: {relative}")
        digest = sha256_file(path)
        _require(
            digest == expected,
            f"pinned SHA drift for {relative}: {digest} != {expected}",
        )
        table[relative] = digest
    return table


def verify_sealed_predecessors(repo_root: Path) -> dict[str, str]:
    """Sidecar-verify the immutable sealed predecessors (guidance section 2.1)."""
    base = Path(repo_root).absolute()
    table: dict[str, str] = {}
    for relative in plan.SEALED_PREDECESSOR_FILES:
        path = base / relative
        sidecar = Path(str(path) + ".sha256")
        _require(path.is_file() and sidecar.is_file(), f"sealed predecessor missing: {relative}")
        digest = sha256_file(path)
        expected = sidecar.read_text().split()[0]
        _require(digest == expected, f"sealed predecessor sidecar drift: {relative}")
        _require(
            digest == plan.SEALED_PREDECESSOR_SHA256[relative],
            f"sealed predecessor SHA drift: {relative}",
        )
        table[relative] = digest
    return table


def load_pinned_module(name: str, repo_root: Path, relative: str):
    """importlib-load a sealed script after verifying its pinned SHA-256."""
    base = Path(repo_root).absolute()
    path = base / relative
    _require(path.is_file(), f"pinned module missing: {relative}")
    digest = sha256_file(path)
    expected = plan.PINNED_SHA256.get(relative)
    _require(
        expected is not None,
        f"no pinned SHA literal for {relative} (add it to plan.PINNED_SHA256)",
    )
    _require(digest == expected, f"pinned module SHA drift for {relative}: {digest}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sealed_runner_stack(repo_root: Path, tfpd_root: Path) -> dict:
    """Load the sealed runner stack exactly once, reusing its own module names.

    ``run_admission_arm`` importlib-loads ``tfpd_lane_arm_common``,
    ``tfpd_lane_matched_scorer`` and ``tfpd_lane_receipt`` at ITS import time;
    loading it first and then reading those names out of ``sys.modules`` gives
    the successor runner the SAME module objects the sealed runner uses.
    """
    arm_runner = load_pinned_module(
        "tfpd_admission_runner", repo_root, "tfpd_exploration/scripts/run_admission_arm.py"
    )
    pop_runner = load_pinned_module(
        "cal_aug_pop_robust_cell", repo_root, "tfpd_exploration/scripts/run_pop_robust_cell.py"
    )
    _require(
        "tfpd_lane_arm_common" in sys.modules
        and "tfpd_lane_matched_scorer" in sys.modules
        and "tfpd_lane_receipt" in sys.modules,
        "sealed runner did not load the tfpd_lane helper stack",
    )
    spec = importlib.util.spec_from_file_location(
        "tfpd_lane_pop_robust", tfpd_root / "src/tfpd_lane/pop_robust.py"
    )
    if "tfpd_lane_pop_robust" not in sys.modules:
        module = importlib.util.module_from_spec(spec)
        sys.modules["tfpd_lane_pop_robust"] = module
        spec.loader.exec_module(module)
    return {
        "arm_runner": arm_runner,
        "pop_runner": pop_runner,
        "arm_common": sys.modules["tfpd_lane_arm_common"],
        "matched_scorer": sys.modules["tfpd_lane_matched_scorer"],
        "receipt": sys.modules["tfpd_lane_receipt"],
        "pop_robust": sys.modules["tfpd_lane_pop_robust"],
    }


# ---------------------------------------------------------------------------
# GPU binding (work order section 4)
# ---------------------------------------------------------------------------


def gpu_binding(device: str, require_uuid: Optional[str] = plan.BOUND_GPU_UUID) -> dict:
    """Resolve the physical GPU UUID via ``nvidia-smi`` and bind it.

    ``CUDA_VISIBLE_DEVICES`` selects the visible index; the resolved UUID of
    the selected device must equal ``require_uuid`` when a CUDA device is
    requested (the work-order-bound card).  Fails closed on any resolution
    problem — a matched pair must never run on an unbound device.
    """
    import torch

    binding: dict = {
        "requested_device": str(device),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "require_uuid": require_uuid,
    }
    if not str(device).startswith("cuda"):
        binding["mode"] = "cpu (no GPU binding required)"
        return binding
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except Exception as error:  # noqa: BLE001 - any resolution failure is fatal
        raise ReceiptError(f"nvidia-smi GPU binding query failed: {error}") from error
    rows = [
        [cell.strip() for cell in line.split(",")]
        for line in query.stdout.strip().splitlines()
        if line.strip()
    ]
    _require(bool(rows), "nvidia-smi returned no GPU rows")
    table = {row[0]: {"uuid": row[1], "name": row[2], "driver": row[3]} for row in rows}
    visible = binding["cuda_visible_devices"]
    if visible == "":
        visible_indices = list(table)
    else:
        visible_indices = [part for part in visible.split(",") if part != ""]
        _require(
            all(index in table for index in visible_indices),
            f"CUDA_VISIBLE_DEVICES entries not resolved by nvidia-smi: {visible!r}",
        )
    # torch's device index is the ORDINAL within the visible set
    ordinal = 0
    if ":" in str(device):
        try:
            ordinal = int(str(device).split(":", 1)[1])
        except ValueError as error:
            raise ReceiptError(f"unparseable cuda device ordinal: {device!r}") from error
    _require(
        0 <= ordinal < len(visible_indices),
        f"cuda ordinal {ordinal} outside the visible set {visible_indices}",
    )
    physical_index = visible_indices[ordinal]
    selected = table[physical_index]
    binding.update(
        {
            "mode": "cuda",
            "physical_index": physical_index,
            "selected_uuid": selected["uuid"],
            "selected_name": selected["name"],
            "driver_version": selected["driver"],
            "all_gpus": table,
        }
    )
    if require_uuid:
        _require(
            selected["uuid"] == require_uuid,
            f"GPU binding drift: selected {selected['uuid']} != bound {require_uuid}",
        )
    return binding


# ---------------------------------------------------------------------------
# attempt receipt discipline (publish before any data/model access)
# ---------------------------------------------------------------------------


def publish_attempt(out_dir: Path, receipt_module, payload: dict) -> str:
    """Write ``attempt.json`` transactionally; returns its SHA-256."""
    receipt_module.write_receipt_transactionally(Path(out_dir) / ATTEMPT_FILE, payload)
    return sha256_file(Path(out_dir) / ATTEMPT_FILE)


def require_attempt(out_dir: Path, stage: str) -> None:
    """Guard: refuse to proceed at ``stage`` unless the attempt is published."""
    attempt = Path(out_dir) / ATTEMPT_FILE
    _require(
        attempt.is_file(),
        f"attempt receipt missing at stage {stage!r}: data/model access is not "
        "authorized before attempt.json exists",
    )


def attempt_guard(out_dir: Path) -> Callable[[str], None]:
    def guard(stage: str) -> None:
        require_attempt(out_dir, stage)

    return guard


def environment_payload(receipt_module, root: Path, device, gpu: dict) -> dict:
    import os

    import torch

    return {
        "python": sys.version.split()[0],
        "platform": __import__("platform").platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "no_user_site": bool(sys.flags.no_user_site),
        "repo_root": str(Path(root).absolute()),
        "gpu": gpu,
    }


# ---------------------------------------------------------------------------
# hard wall-clock guard
# ---------------------------------------------------------------------------


class DeadlineGuard:
    """Wall-clock deadline with stage/step context for the failure receipt."""

    def __init__(self, timeout_seconds: float, steps_completed: int = 0):
        self.limit_s = float(timeout_seconds)
        self.started = time.monotonic()
        self.steps_completed = int(steps_completed)
        self.stage = "initialization"

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return self.limit_s - self.elapsed()

    def check(self, stage: str, steps_completed: Optional[int] = None) -> None:
        self.stage = stage
        if steps_completed is not None:
            self.steps_completed = int(steps_completed)
        if self.elapsed() >= self.limit_s:
            raise CalAugTimeout(
                stage=stage,
                steps_completed=self.steps_completed,
                elapsed_s=self.elapsed(),
                limit_s=self.limit_s,
            )


class DeadlineLoader:
    """Iterable wrapper giving the sealed ``train_epoch`` a per-batch deadline.

    ``arm_runner.train_epoch`` only iterates ``for batch in loader``, so a
    wrapper iterable adds per-batch granularity (probe + timeout) without
    touching any sealed file.  ``on_step(step_done_in_epoch)`` runs after each
    processed step (generator resumption semantics).  ``steps_offset`` is the
    GLOBAL optimizer-step count at this epoch's start, so the guard's
    ``steps_completed`` stays globally meaningful mid-epoch.
    """

    def __init__(self, loader, guard: DeadlineGuard, on_step: Optional[Callable[[int], None]] = None):
        self.loader = loader
        self.guard = guard
        self.on_step = on_step
        self.steps_offset = 0
        self.steps_yielded = 0

    def __iter__(self):
        step = 0
        for batch in self.loader:
            self.guard.check(f"training_batch_{step}", self.steps_offset + step)
            yield batch
            step += 1
            self.steps_yielded = step
            self.guard.steps_completed = self.steps_offset + step
            if self.on_step is not None:
                self.on_step(step)


def publish_failure(out_dir: Path, receipt_module, payload: dict) -> Optional[str]:
    """Atomic ``CELL_FAILED`` receipt; never masks the original exception."""
    body = {
        "schema": plan.SCHEMA + "_failure",
        "status": CELL_FAILED,
        "terminal_published": False,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    try:
        receipt_module.write_receipt_transactionally(
            Path(out_dir) / TERMINAL_FILE, body
        )
        return sha256_file(Path(out_dir) / TERMINAL_FILE)
    except BaseException as error:  # noqa: BLE001 - report, never mask
        print(f"CELL_FAILED receipt could not be written: {error}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# throughput probe (work order section 4)
# ---------------------------------------------------------------------------


def probe_payload(
    *, arm: str,
    measured_steps: int,
    warmup_steps: int,
    seconds: float,
    total_steps: int = plan.TOTAL_OPTIMIZER_STEPS,
    ceiling_gpu_hours: float = plan.PLANNING_CEILING_GPU_HOURS,
    sealed_anchor_steps_per_second: float = plan.SEALED_ANCHOR_STEPS_PER_SECOND,
) -> dict:
    """steps/s over the measured window plus the projections the gate needs."""
    _require(measured_steps > 0 and seconds > 0.0, "probe needs a positive window")
    steps_per_second = float(measured_steps) / float(seconds)
    projected_arm_seconds = total_steps / steps_per_second
    pair_gpu_hours = 2.0 * projected_arm_seconds / 3600.0
    return {
        "schema": plan.SCHEMA + "_probe",
        "status": "PROBE_PUBLISHED",
        "arm": arm,
        "measured_steps": int(measured_steps),
        "warmup_steps_excluded": int(warmup_steps),
        "measured_seconds": round(float(seconds), 3),
        "steps_per_second": steps_per_second,
        "sealed_anchor_steps_per_second": sealed_anchor_steps_per_second,
        "sealed_anchor_seconds_per_run": plan.SEALED_ANCHOR_SECONDS_PER_RUN,
        "projected_total_steps": int(total_steps),
        "projected_seconds_per_arm": projected_arm_seconds,
        "projected_pair_gpu_hours": pair_gpu_hours,
        "planning_ceiling_gpu_hours": ceiling_gpu_hours,
        "within_planning_ceiling": bool(pair_gpu_hours <= ceiling_gpu_hours),
        "separate_scoring_time_note": (
            "scoring (mechanism + deployment) is measured separately and is not "
            "part of the training-arm ceiling"
        ),
    }


def read_smoke_digests(smoke_path: Path) -> Optional[dict]:
    """Read the smoke receipt's cross-arm digests (fail-closed on drift)."""
    path = Path(smoke_path)
    if not path.is_file():
        return None
    body = json.loads(path.read_text())
    sidecar = Path(str(path) + ".sha256")
    _require(sidecar.is_file(), f"smoke receipt sidecar missing: {sidecar}")
    _require(
        sidecar.read_text().split()[0] == sha256_file(path),
        "smoke receipt SHA drift against its sidecar",
    )
    return body
