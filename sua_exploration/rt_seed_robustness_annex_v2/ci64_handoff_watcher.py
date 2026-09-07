"""Fail-closed local CI64-to-RT-annex campaign handoff.

This module is deliberately inert unless its explicit ``--watch --execute``
route is used.  It never opens NWB files or constructs CUDA objects itself.
Before it may invoke the existing paired-wave supervisor, it requires all of
the following on the local host:

* both receipt-bound H1 CI64 static partition runners are no longer present
  and their exact current-run artifact-completeness contracts pass;
* every current CI64 five-arm terminal receipt is immutable and valid;
* no CI64 trainer, source executor, or static runner remains in ``/proc``;
* GPUs 0 and 1 each have two consecutive empty compute-process samples;
* the current annex implementation snapshot exactly matches the immutable
  user-site-isolated supplemental readiness receipt; and
* the externally supplied campaign root is absent.

Any identity, terminal, process, receipt, snapshot, or root drift is an
error.  The watcher does not signal, terminate, or otherwise manage H1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import uuid
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import launcher
from . import spec
from . import supervisor


ANNEX_ROOT = Path(__file__).resolve().parent
SUPPLEMENTAL_RECEIPT = ANNEX_ROOT / "launch_readiness_supplemental_v2_3.json"
SUPPLEMENTAL_PREDECESSOR = ANNEX_ROOT / "launch_readiness_supplemental_v2_2.json"
DEFAULT_CAMPAIGN_ROOT = Path(
    "/home/xinyuan/rt_seed_robustness_annex_v2_1_campaign_20260810_v3"
)
RECOMMENDED_CONTROL_ROOT = Path(
    "/home/xinyuan/rt_seed_robustness_annex_v2_1_control_20260810_v3"
)
CI_TERMINAL_ROOT = (
    launcher.ROOT
    / "SPINT-main/pilot_artifacts/h1_carrierid_date_lodo_ci/terminal_evaluations"
)
CI_ROOT = launcher.ROOT / "SPINT-main"
CI_RUN_ROOT = CI_ROOT / "pilot_artifacts/h1_carrierid_date_lodo_ci/gpu_runs"
CI_PREFLIGHT_ROOT = CI_ROOT / "pilot_artifacts/h1_carrierid_date_lodo_ci/preflights"
CI_CHECK_ROOT = CI_ROOT / "pilot_artifacts/h1_carrierid_date_lodo_ci/terminal_checks"
CI_LAUNCH_RECEIPT = (
    CI_ROOT
    / "pilot_artifacts/h1_carrierid_date_lodo_ci/"
    "H1_CARRIERID_DATE_LODO_CI64_SLODO_LAUNCH_RECEIPT_v1.json"
)
CI_SOURCE_AGGREGATE = (
    CI_ROOT
    / "pilot_artifacts/h1_carrierid_date_lodo_phase2/"
    "H1_CARRIERID_DATE_LODO_FIVE_DATE_HELDOUT_AGGREGATE_ROUTE_PREREQUISITE_v1.json"
)
CI_TERMINAL_AGGREGATE = (
    CI_ROOT
    / "pilot_artifacts/h1_carrierid_date_lodo_ci/"
    "H1_CARRIERID_DATE_LODO_CI_FIVEDATE_TERMINAL_AGGREGATE_v1.json"
)
CI_DATA_DIR = CI_ROOT / "data/000954"
CI_CHECKER_SCRIPT = CI_ROOT / "scripts/h1_carrierid_date_lodo_ci_terminal_checker.py"
CI_EVALUATOR_SCRIPT = CI_ROOT / "scripts/h1_carrierid_date_lodo_ci_terminal_evaluate.py"
CI_AGGREGATE_SCRIPT = CI_ROOT / "scripts/h1_carrierid_date_lodo_ci_fivedate_aggregate.py"
CI_CHECKER_SCHEMA = "h1_carrierid_date_lodo_ci_five_arm_terminal_check_v1"
CI_CHECKER_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_FIVE_ARM_SOURCE_E49_CHECKPOINTS_NO_TARGET"
CI_TERMINAL_AGGREGATE_SCHEMA = "h1_carrierid_date_lodo_ci_fivedate_terminal_aggregate_v1"
CI_TERMINAL_AGGREGATE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_FIVEDATE_AGGREGATED_WITH_FROZEN_GATE"
CI_SCHEMA = "h1_carrierid_date_lodo_ci_five_arm_terminal_evaluation_v1"
CI_ARMS = ("CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS")
CI_PARTITIONS: dict[str, dict[str, Any]] = {
    "gpu0_h1_static_ci64": {
        "device_index": 0,
        "runner_pid": 2814205,
        "runner_starttime": "160906485",
        # btime + runner_starttime / CLK_TCK, captured from the hard-bound
        # runner identity.  This is deliberately a wall-clock boundary, not
        # a mutable artifact timestamp supplied by a producer.
        "runner_start_wall_time": 1786330301.85,
        "runner_cmdline": "scripts/h1_carrierid_date_lodo_ci_static_partition_runner.sh 0",
        "dates": ("19250108", "19250115", "19250120"),
    },
    "gpu1_h1_static_ci64": {
        "device_index": 1,
        "runner_pid": 2814209,
        "runner_starttime": "160906485",
        "runner_start_wall_time": 1786330301.85,
        "runner_cmdline": "scripts/h1_carrierid_date_lodo_ci_static_partition_runner.sh 1",
        "dates": ("19250113", "19250119"),
    },
}

SOURCE_COMPLETION_SCHEMA = "rt_seed_robustness_annex_v2_ci64_partition_source_completion_v2"
TERMINAL_BINDING_SCHEMA = "rt_seed_robustness_annex_v2_ci64_terminal_binding_v2"
GPU_LEASE_SCHEMA = "rt_seed_robustness_annex_v2_gpu_lease_v1"
RT_AGGREGATE_SCHEMA = "rt_seed_robustness_annex_v2_canonical_20_cell_aggregate_v1"


class HandoffError(RuntimeError):
    """Raised when the handoff cannot prove every launch predicate."""


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise HandoffError(f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str, readonly: bool | None = None) -> os.stat_result:
    """Validate one exact path without following a symlink.

    ``Path.is_file`` follows links, which is not suitable for a provenance
    boundary.  Keep this primitive small because it is used for every sealed
    finalization input.
    """
    try:
        info = path.lstat()
    except OSError as error:
        raise HandoffError(f"required {label} is missing: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise HandoffError(f"required {label} is not a regular non-symlink: {path}")
    if readonly is True and stat.S_IMODE(info.st_mode) != 0o444:
        raise HandoffError(f"required {label} is not immutable 0444: {path}")
    return info


def _ctime(info: os.stat_result) -> float:
    # Linux ctime is the available inode-change timestamp.  Some filesystems
    # also offer st_birthtime; use the later value when it exists, never mtime.
    birth = getattr(info, "st_birthtime", 0.0)
    return max(float(info.st_ctime), float(birth or 0.0))


def _artifact_binding(path: Path, *, label: str, newer_than: float | None = None,
                      readonly: bool | None = None) -> dict[str, Any]:
    info = _regular(path, label=label, readonly=readonly)
    observed_ctime = _ctime(info)
    if newer_than is not None and observed_ctime <= float(newer_than):
        raise HandoffError(f"stale {label}; inode ctime/birth is not after runner start: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "inode_ctime_or_birth": observed_ctime,
            "mode": format(stat.S_IMODE(info.st_mode), "04o")}


def _json_object(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HandoffError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(result, dict):
        raise HandoffError(f"expected JSON object: {path}")
    return result


def _ci_terminal_path(date: str) -> Path:
    return (
        CI_TERMINAL_ROOT
        / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json"
    ).resolve()


def _ci_terminal_status(date: str) -> str:
    return f"PASS_H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_EVALUATED"


def _validate_ci_terminal(path: Path, date: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise HandoffError(f"CI64 terminal missing: {date}")
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise HandoffError(f"CI64 terminal must be immutable 0444: {date}")
    body = _json_object(path)
    if body.get("schema") != CI_SCHEMA:
        raise HandoffError(f"CI64 terminal schema drift: {date}")
    if body.get("status") != _ci_terminal_status(date):
        raise HandoffError(f"CI64 terminal does not prove normal completion: {date}")
    if body.get("outer_date") != date:
        raise HandoffError(f"CI64 terminal outer-date drift: {date}")
    checker = body.get("terminal_checker")
    if not isinstance(checker, Mapping):
        raise HandoffError(f"CI64 terminal checker binding missing: {date}")
    checker_path = Path(str(checker.get("path", ""))).resolve()
    if not checker_path.is_file() or checker_path.is_symlink() or checker.get("sha256") != _sha256(checker_path):
        raise HandoffError(f"CI64 terminal checker binding drift: {date}")
    if path.resolve() == _ci_terminal_path(date):
        if checker_path != _ci_checker_path(date):
            raise HandoffError(f"CI64 terminal checker canonical path drift: {date}")
        _validate_ci_checker(checker_path, date)
    metrics = body.get("metrics")
    # Production receipts are serialized with ``sort_keys=True``.  Key order
    # is therefore not the scientific arm order and must never be used as arm
    # identity; require exact set equality and validate every named row.
    if not isinstance(metrics, Mapping) or set(metrics) != set(CI_ARMS):
        raise HandoffError(f"CI64 terminal five-arm metrics contract drift: {date}")
    scope = body.get("scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(field) is not False
        for field in ("formal_heldout_opened", "minival_opened", "evalai_opened")
    ):
        raise HandoffError(f"CI64 terminal scope drift: {date}")
    updates = body.get("deployment_updates")
    if not isinstance(updates, Mapping) or (
        updates.get("optimizer_steps") != 0
        or updates.get("backward_steps") != 0
        or updates.get("model_state_unchanged") is not True
    ):
        raise HandoffError(f"CI64 terminal deployment-update drift: {date}")
    one_shot = body.get("one_shot")
    if not isinstance(one_shot, Mapping) or (
        one_shot.get("canonical_output_path") != str(path.resolve())
        or one_shot.get("same_date_prior_terminal_evaluation_receipts") != 0
    ):
        raise HandoffError(f"CI64 terminal one-shot binding drift: {date}")
    target = body.get("target")
    if not isinstance(target, Mapping):
        raise HandoffError(f"CI64 terminal target contract missing: {date}")
    sessions = tuple(target.get("sessions", ()))
    shared_query_hash = target.get("shared_query_window_indices_sha256")
    if (
        not sessions
        or not isinstance(shared_query_hash, str)
        or not shared_query_hash
        or target.get("all_query_histories_start_at_or_after_fifth_trial") is not True
    ):
        raise HandoffError(f"CI64 terminal target-window binding drift: {date}")
    files = target.get("files")
    if not isinstance(files, Mapping) or set(files) != set(sessions):
        raise HandoffError(f"CI64 terminal target input-file session set drift: {date}")
    if any(not isinstance(files[session], str) or re.fullmatch(r"[0-9a-f]{64}", files[session]) is None
           for session in sessions):
        raise HandoffError(f"CI64 terminal target input-file SHA contract drift: {date}")
    for arm in CI_ARMS:
        row = metrics[arm]
        if not isinstance(row, Mapping):
            raise HandoffError(f"CI64 terminal metric row malformed: {date}/{arm}")
        pooled_r2 = row.get("pooled_r2")
        per_session = row.get("per_session")
        if (
            not isinstance(pooled_r2, (int, float))
            or not math.isfinite(float(pooled_r2))
            or row.get("query_window_indices_sha256") != shared_query_hash
            or row.get("state_immutable") is not True
            or row.get("state_sha256_before") != row.get("state_sha256_after")
            or not isinstance(per_session, Mapping)
            or tuple(per_session) != sessions
        ):
            raise HandoffError(f"CI64 terminal metric/window/state drift: {date}/{arm}")
        for session in sessions:
            session_row = per_session[session]
            session_r2 = session_row.get("r2") if isinstance(session_row, Mapping) else None
            if (
                not isinstance(session_r2, (int, float))
                or not math.isfinite(float(session_r2))
            ):
                raise HandoffError(
                    f"CI64 terminal per-session metric is non-finite: {date}/{arm}/{session}"
                )


def validate_ci_terminals(
    terminal_paths: Mapping[str, Mapping[str, Path]] | None = None,
) -> dict[str, dict[str, str]]:
    """Validate all five actual CI64 terminal artifacts, never scores."""

    expected = {
        name: {date: _ci_terminal_path(date) for date in part["dates"]}
        for name, part in CI_PARTITIONS.items()
    }
    chosen = expected if terminal_paths is None else terminal_paths
    if set(chosen) != set(expected):
        raise HandoffError("CI64 terminal partition set drift")
    result: dict[str, dict[str, str]] = {}
    for name, expected_dates in expected.items():
        given = chosen.get(name)
        if not isinstance(given, Mapping) or tuple(given) != tuple(expected_dates):
            raise HandoffError(f"CI64 terminal date binding drift: {name}")
        result[name] = {}
        for date, default_path in expected_dates.items():
            path = Path(given[date])
            # Fixtures may supply alternate paths; the production default is
            # nevertheless an exact, non-discovery path per date.
            _validate_ci_terminal(path, date)
            result[name][date] = _sha256(path)
    return result


def _reject_present_ci_terminal_drift_while_running() -> None:
    """Reject a malformed terminal as soon as it appears during the wait.

    A missing terminal is normal only while its receipt-bound static runner is
    still alive.  A terminal which is present but malformed is an error, not a
    reason to keep polling.
    """

    for partition in CI_PARTITIONS.values():
        for date in partition["dates"]:
            path = _ci_terminal_path(date)
            if path.exists() or path.is_symlink():
                _validate_ci_terminal(path, date)


def _runner_state(partition: Mapping[str, Any]) -> str:
    pid = int(partition["runner_pid"])
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return "exited"
    try:
        fields = (proc / "stat").read_text(encoding="utf-8").split()
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except OSError:
        return "exited"
    if (
        len(fields) < 22
        or fields[21] != str(partition["runner_starttime"])
        or str(partition["runner_cmdline"]) not in cmdline
    ):
        return "identity_drift"
    return "active"


def _proc_rows() -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            rows.append(
                (
                    int(proc.name),
                    (proc / "cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode(errors="replace"),
                )
            )
        except (OSError, ValueError):
            continue
    return rows


def ci64_residual_processes(rows: Iterable[tuple[int, str]] | None = None) -> list[dict[str, Any]]:
    """Return every remaining CI64 runner/trainer/source-executor process."""

    needles = (
        "h1_carrierid_date_lodo_ci_static_partition_runner.sh",
        "h1_carrierid_date_lodo_ci_source_executor.py",
        "h1_carrierid_date_lodo_ci64_",
    )
    matches = [
        {"pid": int(pid), "cmdline": command}
        for pid, command in (_proc_rows() if rows is None else rows)
        if any(needle in command for needle in needles)
    ]
    return sorted(matches, key=lambda item: int(item["pid"]))


def _compute_apps(device: int) -> list[str] | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(device),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and line.strip() != "No running processes found"
    ]


def two_empty_compute_samples(
    *,
    devices: Sequence[int] = (0, 1),
    sample: Callable[[int], list[str] | None] = _compute_apps,
    sleep: Callable[[float], None] = time.sleep,
    stability_seconds: float = 1.0,
) -> dict[str, list[list[str] | None]]:
    """Require two consecutive successful empty compute-PID samples per GPU."""

    if not devices or tuple(dict.fromkeys(devices)) != tuple(devices):
        raise HandoffError("compute sample device set must be non-empty and unique")
    first = {str(device): sample(device) for device in devices}
    sleep(stability_seconds)
    second = {str(device): sample(device) for device in devices}
    samples = {
        device: [first[device], second[device]] for device in first
    }
    if any(sample_rows != [[], []] for sample_rows in samples.values()):
        raise HandoffError("GPU0/1 do not have two consecutive empty compute-PID samples")
    return samples


def _ci_preflight_path(date: str) -> Path:
    return (
        CI_PREFLIGHT_ROOT / f"H1_CARRIERID_DATE_LODO_CI_{date}_CPU_PREFLIGHT_v1.json"
    ).resolve()


def _ci_checkpoint_path(date: str, arm: str) -> Path:
    return (CI_RUN_ROOT / date / arm / "checkpoints/fixed_epoch50/epoch_049.ckpt").resolve()


def _ci_run_dir(date: str, arm: str) -> Path:
    """The only accepted source run directory (never discovered by glob)."""
    return (CI_RUN_ROOT / date / arm).resolve()


def _ci_config_path(date: str, arm: str) -> Path:
    return _ci_run_dir(date, arm) / ".hydra" / "config.yaml"


def _source_metadata_paths(date: str, arm: str) -> tuple[Path, ...]:
    """Exact non-score source metadata required by the static runner contract."""
    run = _ci_run_dir(date, arm)
    # These are fixed writer outputs, not a directory scan.  They establish
    # completion of the source side without opening checkpoint bytes.
    return (run / "config_tree.log", run / "tags.log")


def _validate_partition_source_artifacts(name: str) -> dict[str, Any]:
    """Prove this partition produced all its own 5-arm source artifacts now.

    No terminal receipt, score, mtime or checkpoint content is consulted.
    A completed runner cannot supply an exit status retrospectively, so the
    release fact is explicitly *artifact completeness after the fixed runner
    start boundary*, not a claim about an observed process return code.
    """
    partition = CI_PARTITIONS[name]
    boundary = float(partition["runner_start_wall_time"])
    rows: dict[str, Any] = {}
    for date in partition["dates"]:
        by_arm: dict[str, Any] = {}
        for arm in CI_ARMS:
            run_dir = _ci_run_dir(date, arm)
            try:
                run_info = run_dir.lstat()
            except OSError as error:
                raise HandoffError(f"missing exact CI64 run directory: {date}/{arm}") from error
            if stat.S_ISLNK(run_info.st_mode) or not stat.S_ISDIR(run_info.st_mode):
                raise HandoffError(f"CI64 run directory is not a real directory: {date}/{arm}")
            if _ctime(run_info) <= boundary:
                raise HandoffError(f"stale source run directory; inode ctime/birth is not after runner start: {date}/{arm}")
            artifacts = {
                "config": _artifact_binding(_ci_config_path(date, arm), label="source config", newer_than=boundary),
                "checkpoint": _artifact_binding(_ci_checkpoint_path(date, arm), label="source epoch_049 checkpoint", newer_than=boundary),
            }
            # Existing static runner metadata is a prerequisite when present
            # in the contract.  Both fixed files are required here, so an
            # incomplete run cannot be mistaken for a terminal artifact.
            artifacts["metadata"] = [
                _artifact_binding(path, label="source metadata", newer_than=boundary)
                for path in _source_metadata_paths(date, arm)
            ]
            by_arm[arm] = {"run_dir": str(run_dir), "run_dir_ctime_or_birth": _ctime(run_info), "artifacts": artifacts}
        rows[date] = by_arm
    return {
        "partition": name,
        "runner_identity": {key: partition[key] for key in ("runner_pid", "runner_starttime", "runner_cmdline", "runner_start_wall_time")},
        "outcome_completeness": "ALL_EXACT_DATES_X_FIVE_ARMS_FRESH_ARTIFACTS_PRESENT; EXIT_CODE_NOT_OBSERVED_BY_WATCHER",
        "dates": rows,
    }


def _ci_checker_path(date: str) -> Path:
    return (
        CI_CHECK_ROOT / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_CHECK_v1.json"
    ).resolve()


def _validate_ci_checker(path: Path, date: str) -> str:
    if not path.is_file() or path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise HandoffError(f"CI64 terminal checker missing or non-immutable: {date}")
    body = _json_object(path)
    if body.get("schema") != CI_CHECKER_SCHEMA or body.get("status") != CI_CHECKER_STATUS:
        raise HandoffError(f"CI64 terminal checker schema/status drift: {date}")
    if body.get("outer_date") != date:
        raise HandoffError(f"CI64 terminal checker date drift: {date}")
    checkpoints = body.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(CI_ARMS):
        raise HandoffError(f"CI64 terminal checker arm set drift: {date}")
    for arm in CI_ARMS:
        row = checkpoints[arm]
        expected = _ci_checkpoint_path(date, arm)
        expected_config = expected.parent.parent.parent / ".hydra/config.yaml"
        if (
            not isinstance(row, Mapping)
            or Path(str(row.get("checkpoint_path", ""))).resolve() != expected
            or row.get("checkpoint_sha256") != _sha256(expected)
            or Path(str(row.get("config_path", ""))).resolve() != expected_config
            or row.get("config_sha256") != _sha256(expected_config)
        ):
            raise HandoffError(f"CI64 terminal checker checkpoint binding drift: {date}/{arm}")
    preflight = body.get("ci_preflight")
    aggregate = body.get("five_date_aggregate")
    prepared_launch = body.get("prepared_launch_receipt")
    if (
        not isinstance(preflight, Mapping)
        or Path(str(preflight.get("path", ""))).resolve() != _ci_preflight_path(date)
        or preflight.get("sha256") != _sha256(_ci_preflight_path(date))
        or not isinstance(aggregate, Mapping)
        or Path(str(aggregate.get("path", ""))).resolve() != CI_SOURCE_AGGREGATE.resolve()
        or aggregate.get("sha256") != _sha256(CI_SOURCE_AGGREGATE)
        or not isinstance(prepared_launch, Mapping)
        or Path(str(prepared_launch.get("path", ""))).resolve() != CI_LAUNCH_RECEIPT.resolve()
        or prepared_launch.get("sha256") != _sha256(CI_LAUNCH_RECEIPT)
    ):
        raise HandoffError(f"CI64 terminal checker prerequisite binding drift: {date}")
    return _sha256(path)


def _validate_terminal_aggregate(path: Path | None = None) -> str:
    path = CI_TERMINAL_AGGREGATE if path is None else Path(path)
    if not path.is_file() or path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise HandoffError("canonical CI64 five-date terminal aggregate missing or non-immutable")
    body = _json_object(path)
    if (
        body.get("schema") != CI_TERMINAL_AGGREGATE_SCHEMA
        or body.get("status") != CI_TERMINAL_AGGREGATE_STATUS
        or tuple(body.get("required_outer_dates", ())) != ("19250108", "19250113", "19250115", "19250119", "19250120")
        or body.get("all_five_dates_reported") is not True
        or set(body.get("arms", ())) != set(CI_ARMS)
    ):
        raise HandoffError("canonical CI64 five-date terminal aggregate contract drift")
    per_date = body.get("per_date")
    if not isinstance(per_date, Mapping) or set(per_date) != {
        "19250108", "19250113", "19250115", "19250119", "19250120"
    }:
        raise HandoffError("canonical CI64 five-date aggregate date set drift")
    for date in per_date:
        row = per_date[date]
        receipt = row.get("receipt") if isinstance(row, Mapping) else None
        terminal = _ci_terminal_path(date)
        if (
            not isinstance(receipt, Mapping)
            or Path(str(receipt.get("path", ""))).resolve() != terminal
            or receipt.get("sha256") != _sha256(terminal)
        ):
            raise HandoffError(f"canonical CI64 five-date aggregate receipt binding drift: {date}")
    scope = body.get("scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(field) not in (False, 0)
        for field in ("nwb_opened", "checkpoint_opened", "trainer_constructed", "cuda_constructed", "formal_heldout_opened", "minival_opened", "evalai_opened")
    ):
        raise HandoffError("canonical CI64 five-date aggregate scope drift")
    return _sha256(path)


def _validate_supplemental_snapshot(path: Path = SUPPLEMENTAL_RECEIPT) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise HandoffError("supplemental readiness receipt missing")
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise HandoffError("supplemental readiness receipt must be immutable 0444")
    receipt = _json_object(path)
    if receipt.get("schema") != "rt_seed_robustness_annex_v2_1_launch_readiness_supplemental_v1":
        raise HandoffError("supplemental readiness schema drift")
    if receipt.get("status") != (
        "PASS_STATIC_E2E_WAVE_READINESS_USER_SITE_ISOLATED_SUPPLEMENTAL_NOT_AUTHORIZED"
    ):
        raise HandoffError("supplemental readiness status drift")
    if receipt.get("supplemental_revision") != "v2.3":
        raise HandoffError("supplemental readiness revision drift")
    prior = receipt.get("append_only_supplement")
    if not isinstance(prior, Mapping):
        raise HandoffError("supplemental receipt lacks prior-seal binding")
    prior_path = ANNEX_ROOT / "launch_readiness_receipt.json"
    if prior.get("path") != str(prior_path.resolve()) or prior.get("sha256") != _sha256(prior_path):
        raise HandoffError("supplemental prior sealed receipt binding drift")
    predecessor = receipt.get("append_only_predecessor")
    if not isinstance(predecessor, Mapping):
        raise HandoffError("supplemental v2.3 lacks v2.2 predecessor binding")
    _regular(SUPPLEMENTAL_PREDECESSOR, label="supplemental v2.2 predecessor", readonly=True)
    if (
        predecessor.get("path") != str(SUPPLEMENTAL_PREDECESSOR.resolve())
        or predecessor.get("sha256") != _sha256(SUPPLEMENTAL_PREDECESSOR)
        or predecessor.get("mode") != "0444"
        or predecessor.get("preserved_unmodified") is not True
    ):
        raise HandoffError("supplemental v2.2 predecessor binding drift")
    expected = receipt.get("implementation_snapshot")
    if not isinstance(expected, Mapping):
        raise HandoffError("supplemental receipt lacks implementation snapshot")
    if receipt.get("implementation_snapshot_sha256") != launcher.implementation_snapshot_sha256(expected):
        raise HandoffError("supplemental implementation-snapshot digest drift")
    try:
        launcher.assert_implementation_snapshot(expected)
    except launcher.LaunchReadinessError as error:
        raise HandoffError(f"annex implementation snapshot drift: {error}") from error
    if receipt.get("interpreter_isolation_summary", {}).get("exact_python_path") != str(launcher.DEFAULT_PYTHON):
        raise HandoffError("supplemental exact interpreter binding drift")
    if receipt.get("interpreter_isolation_summary", {}).get("python_no_user_site") != "1":
        raise HandoffError("supplemental user-site isolation binding drift")
    audit = receipt.get("shared_change_compatibility_audit")
    if not isinstance(audit, Mapping) or audit.get("conclusion") != (
        "PASS_CURRENT_SHARED_CHANGES_DO_NOT_CHANGE_AFC4_VEL_OR_AFC4_MB4_"
        "TRAIN_OR_SCORE_SEMANTICS"
    ):
        raise HandoffError("supplemental shared-change compatibility audit drift")
    if audit.get("exact_patch_history_recovered") is not True or (
        audit.get("scope_limited_to_fixed_annex_arms") is not True
    ):
        raise HandoffError("supplemental shared-change audit scope drift")
    predecessor_body = _json_object(SUPPLEMENTAL_PREDECESSOR)
    predecessor_snapshot = predecessor_body.get("implementation_snapshot")
    files = audit.get("files")
    changed_paths = {
        "streaming_calibration_exp/scripts/run_rt_clean_nested_loso.py",
        "streaming_calibration_exp/src/data/rt_nested_loso_datamodule.py",
        "streaming_calibration_exp/src/rt_clean_nested_loso_eval.py",
    }
    if not isinstance(predecessor_snapshot, Mapping) or not isinstance(files, Mapping) or set(files) != changed_paths:
        raise HandoffError("supplemental shared-change file set drift")
    for relative in changed_paths:
        row = files.get(relative)
        if (
            not isinstance(row, Mapping)
            or row.get("prior_v2_2_sha256")
            != predecessor_snapshot.get(relative, {}).get("sha256")
            or row.get("current_sha256") != expected.get(relative, {}).get("sha256")
        ):
            raise HandoffError(f"supplemental shared-change hash audit drift: {relative}")
    reconfirmed = receipt.get("annex_contract_reconfirmation")
    if reconfirmed != {
        "arms": ["afc4_vel", "afc4_mb4"],
        "calibration_trials_m": 24,
        "cells": 20,
        "folds": [0, 3, 6, 9, 12],
        "max_epochs": 35,
        "query_start_trial_q": 24,
        "seeds": [43, 44],
        "target_backpropagation": False,
        "target_optimizer_present": False,
        "window_size_bins_w": 50,
    }:
        raise HandoffError("supplemental fixed Annex contract reconfirmation drift")
    roots = receipt.get("handoff_roots")
    if not isinstance(roots, Mapping) or (
        roots.get("campaign_root") != str(DEFAULT_CAMPAIGN_ROOT)
        or roots.get("control_receipt_root") != str(RECOMMENDED_CONTROL_ROOT)
        or roots.get("both_required_fresh_and_absent_before_rearm") is not True
    ):
        raise HandoffError("supplemental recommended root binding drift")
    return receipt


def _validate_campaign_root(root: str | Path) -> Path:
    campaign_root = launcher.validate_artifact_root(root, require_fresh=True)
    if campaign_root.exists():  # Defensive redundancy: no resolution race is accepted.
        raise HandoffError(f"campaign root must be fresh and absent: {campaign_root}")
    return campaign_root


def build_campaign_plan(*, campaign_root: str | Path) -> dict[str, Any]:
    """Construct the exact ten-wave supervisor plan without a GPU launch."""

    root = _validate_campaign_root(campaign_root)
    receipt = _validate_supplemental_snapshot()
    snapshot = receipt["implementation_snapshot"]
    waves = []
    for seed in spec.EXPECTED_SEEDS:
        for fold in spec.SELECTED_FOLDS:
            wave_root = root / f"s{seed}_f{fold}_paired_wave"
            waves.append(
                supervisor.build_wave_plan(
                    seed=seed,
                    fold=fold,
                    artifact_root=wave_root,
                    python_executable=str(launcher.DEFAULT_PYTHON),
                    snapshot=snapshot,
                )
            )
    return {
        "schema": "rt_seed_robustness_annex_v2_ci64_handoff_plan_v1",
        "status": "DRY_RUN_NOT_ARMED",
        "development_only": True,
        "gpu_authorized": False,
        "gpu_launched": False,
        "nwb_read": False,
        "formal_heldout_opened": False,
        "campaign_root": str(root),
        "campaign_nonce": uuid.uuid4().hex,
        "campaign_root_fresh_absent": True,
        "interpreter": str(launcher.DEFAULT_PYTHON),
        "execution_environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": launcher.execution_environment()["PYTHONPATH"],
            "execution_enable_env": f"{launcher.EXECUTION_ENABLE_ENV}=1",
            "authorization_env": (
                f"{launcher.EXECUTION_AUTHORIZATION_ENV}="
                f"{launcher.EXECUTION_AUTHORIZATION_VALUE}"
            ),
        },
        "supplemental_receipt": str(SUPPLEMENTAL_RECEIPT.resolve()),
        "supplemental_receipt_sha256": _sha256(SUPPLEMENTAL_RECEIPT),
        "implementation_snapshot_sha256": receipt["implementation_snapshot_sha256"],
        "watcher_sha256": _sha256(Path(__file__).resolve()),
        "wave_count": len(waves),
        "waves": waves,
        "release_rule": (
            "each exact CI64 static partition independently releases its GPU after source "
            "exit/residual/two-sample checks; canonical checker and forward-only evaluator "
            "terminalize its fixed dates; all five receipts and the canonical aggregate pass; "
            "then GPU0/1 repeat the empty-compute check and the snapshot/fresh root remain unchanged"
        ),
        "failure_policy": "fail_closed_no_supervisor_spawn",
    }


def _write_exclusive_readonly(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink() or os.path.lexists(str(path)):
        raise HandoffError(f"refusing to overwrite handoff receipt: {path}")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise HandoffError(f"refusing to overwrite handoff receipt: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise HandoffError(f"handoff receipt was not published immutable 0444: {path}")


def _receipt_paths(receipt_root: str | Path) -> dict[str, Path]:
    root = launcher.validate_artifact_root(receipt_root)
    paths = {
        "handoff_arm": root / "RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_HANDOFF_ARM_RECEIPT_v1.json",
        "annex_launch": root / "RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_ANNEX_LAUNCH_RECEIPT_v1.json",
        "terminalization_completion": root / "RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_TERMINALIZATION_COMPLETION_RECEIPT_v1.json",
    }
    for name in CI_PARTITIONS:
        token = name.upper()
        paths[f"source_release:{name}"] = root / f"RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_{token}_SOURCE_RELEASE_RECEIPT_v1.json"
        paths[f"terminalization_launch:{name}"] = root / f"RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_{token}_TERMINALIZATION_LAUNCH_RECEIPT_v1.json"
        paths[f"terminalization_completion:{name}"] = root / f"RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_{token}_TERMINALIZATION_COMPLETION_RECEIPT_v1.json"
        for date in CI_PARTITIONS[name]["dates"]:
            paths[f"terminal_invocation:{name}:{date}"] = root / f"RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_{token}_{date}_TERMINAL_INVOCATION_v2.json"
            paths[f"terminal_binding:{name}:{date}"] = root / f"RT_SEED_ROBUSTNESS_ANNEX_V2_CI64_{token}_{date}_TERMINAL_BINDING_v2.json"
    paths["rt_aggregate"] = root / "RT_SEED_ROBUSTNESS_ANNEX_V2_CANONICAL_20_CELL_AGGREGATE_v1.json"
    return paths


def _target_metadata_manifest() -> dict[str, Any]:
    """Hash target *metadata* only; this never opens NWB bytes."""
    if not CI_DATA_DIR.is_dir() or CI_DATA_DIR.is_symlink():
        raise HandoffError("target data directory is not a real directory")
    files: list[dict[str, Any]] = []
    for path in sorted(CI_DATA_DIR.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        info = _regular(path, label="target data metadata")
        files.append({"path": str(path.resolve()), "size": int(info.st_size), "inode": int(info.st_ino),
                      "ctime_or_birth": _ctime(info)})
    if not files:
        raise HandoffError("target data metadata manifest is empty")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"root": str(CI_DATA_DIR.resolve()), "files": files,
            "metadata_sha256": hashlib.sha256(encoded).hexdigest(), "content_opened": False}


def _acquire_gpu_lease(*, receipt_root: Path, device: int, scope: str,
                       compute_sample: Callable[[int], list[str] | None]) -> Path:
    """Atomically occupy a GPU lane after the final empty-PID recheck.

    Lease names are campaign- and scope-specific and are never removed.  A
    second attempt therefore cannot turn a stale control tree into authority.
    """
    if compute_sample(device) != []:
        raise HandoffError(f"GPU{device} gained a compute process before lease acquisition")
    path = receipt_root / f"RT_SEED_ROBUSTNESS_ANNEX_V2_GPU{device}_{scope}_LEASE_v1.json"
    _write_exclusive_readonly(path, {"schema": GPU_LEASE_SCHEMA, "status": "ACQUIRED_EXCLUSIVE",
                                     "device": device, "scope": scope, "compute_pids_immediately_before_acquire": [],
                                     "released_by_deletion": False})
    # A dummy/competitor can appear while publication races; detect it before
    # the caller's Popen.  The immutable lease remains as an audit record.
    if compute_sample(device) != []:
        raise HandoffError(f"GPU{device} lease lost before launch")
    return path


def _write_gpu_release(*, lease: Path, receipt_root: Path, device: int, scope: str) -> Path:
    release = receipt_root / f"RT_SEED_ROBUSTNESS_ANNEX_V2_GPU{device}_{scope}_RELEASE_v1.json"
    _write_exclusive_readonly(release, {"schema": GPU_LEASE_SCHEMA, "status": "RELEASED_LEASE_RETAINED",
                                        "lease_path": str(lease), "lease_sha256": _sha256(lease),
                                        "device": device, "scope": scope, "lease_deleted": False})
    return release


def _runner_summary(
    state: Callable[[Mapping[str, Any]], str] = _runner_state,
) -> dict[str, str]:
    summary = {name: state(partition) for name, partition in CI_PARTITIONS.items()}
    if any(value == "identity_drift" for value in summary.values()):
        raise HandoffError("CI64 static runner identity drift")
    return summary


def _partition_source_residuals(
    name: str, rows: Iterable[tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    """Find CI64 source work bound to exactly one static partition's dates."""

    partition = CI_PARTITIONS[name]
    date_tokens = tuple(f"{date}" for date in partition["dates"])
    own_runner = str(partition["runner_cmdline"])
    matches = []
    for pid, command in (_proc_rows() if rows is None else rows):
        source_executor = "h1_carrierid_date_lodo_ci_source_executor.py" in command
        own_train = "h1_carrierid_date_lodo_ci" in command and any(
            date in command for date in date_tokens
        )
        if own_runner in command or source_executor and any(date in command for date in date_tokens) or own_train:
            matches.append({"pid": int(pid), "cmdline": command})
    return sorted(matches, key=lambda item: int(item["pid"]))


def _partition_source_release_gate(
    *,
    plan: Mapping[str, Any],
    name: str,
    runner_state: Callable[[Mapping[str, Any]], str] = _runner_state,
    process_rows: Iterable[tuple[int, str]] | None = None,
    compute_sample: Callable[[int], list[str] | None] = _compute_apps,
    sleep: Callable[[float], None] = time.sleep,
    stability_seconds: float = 1.0,
) -> dict[str, Any]:
    """Release a partition only after its own source work has truly stopped."""

    receipt = _validate_supplemental_snapshot()
    if plan.get("implementation_snapshot_sha256") != receipt.get("implementation_snapshot_sha256"):
        raise HandoffError("campaign plan supplemental snapshot binding drift")
    if plan.get("watcher_sha256") != _sha256(Path(__file__).resolve()):
        raise HandoffError("campaign plan watcher implementation drift")
    _validate_campaign_root(plan.get("campaign_root", ""))
    partition = CI_PARTITIONS.get(name)
    if partition is None:
        raise HandoffError(f"unknown CI64 partition: {name}")
    state = runner_state(partition)
    if state == "identity_drift":
        raise HandoffError(f"CI64 static runner identity drift: {name}")
    if state == "active":
        _reject_present_ci_terminal_drift_while_running()
        return {"eligible": False, "reason": "WAITING_FOR_PARTITION_SOURCE", "partition": name}
    if state != "exited":
        raise HandoffError(f"CI64 bound static runner is not cleanly absent: {name}")
    residuals = _partition_source_residuals(name, process_rows)
    if residuals:
        raise HandoffError(f"CI64 partition source residual process detected: {name}")
    source_artifacts = _validate_partition_source_artifacts(name)
    device = int(partition["device_index"])
    samples = two_empty_compute_samples(
        devices=(device,), sample=compute_sample, sleep=sleep, stability_seconds=stability_seconds
    )
    _validate_supplemental_snapshot()
    _validate_campaign_root(plan.get("campaign_root", ""))
    return {
        "eligible": True,
        "partition": name,
        "runner_state": state,
        "ci64_source_residual_processes": residuals,
        "source_artifact_completeness": source_artifacts,
        "runner_exit_code_observed": False,
        "compute_pid_samples": samples,
    }


def _release_gate(
    *,
    plan: Mapping[str, Any],
    terminal_paths: Mapping[str, Mapping[str, Path]] | None = None,
    runner_state: Callable[[Mapping[str, Any]], str] = _runner_state,
    process_rows: Iterable[tuple[int, str]] | None = None,
    compute_sample: Callable[[int], list[str] | None] = _compute_apps,
    sleep: Callable[[float], None] = time.sleep,
    stability_seconds: float = 1.0,
) -> dict[str, Any]:
    """Perform one fail-closed release evaluation and return its evidence."""

    # Rebind the sealed input before every decisive check, not only at arm.
    receipt = _validate_supplemental_snapshot()
    if plan.get("implementation_snapshot_sha256") != receipt.get("implementation_snapshot_sha256"):
        raise HandoffError("campaign plan supplemental snapshot binding drift")
    if plan.get("watcher_sha256") != _sha256(Path(__file__).resolve()):
        raise HandoffError("campaign plan watcher implementation drift")
    root = _validate_campaign_root(plan.get("campaign_root", ""))
    states = _runner_summary(runner_state)
    if any(value == "active" for value in states.values()):
        _reject_present_ci_terminal_drift_while_running()
        return {"eligible": False, "reason": "WAITING_FOR_CI64_STATIC_RUNNERS", "runner_states": states}
    if set(states.values()) != {"exited"}:
        raise HandoffError("CI64 bound static runner is not cleanly absent")
    terminal_hashes = validate_ci_terminals(terminal_paths)
    terminal_aggregate_sha = _validate_terminal_aggregate()
    residuals = ci64_residual_processes(process_rows)
    if residuals:
        raise HandoffError("CI64 trainer/runner residual process detected")
    samples = two_empty_compute_samples(
        sample=compute_sample, sleep=sleep, stability_seconds=stability_seconds
    )
    # The root and snapshot are checked once more after the stability delay.
    _validate_campaign_root(root)
    _validate_supplemental_snapshot()
    return {
        "eligible": True,
        "runner_states": states,
        "ci_terminal_sha256": terminal_hashes,
        "ci_terminal_aggregate_sha256": terminal_aggregate_sha,
        "ci64_residual_processes": residuals,
        "compute_pid_samples": samples,
    }


def _require_execution_environment() -> dict[str, str]:
    launcher.require_execution_gate()
    env = launcher.execution_environment()
    if env.get("PYTHONNOUSERSITE") != "1":
        raise HandoffError("PYTHONNOUSERSITE execution environment drift")
    if not launcher.DEFAULT_PYTHON.is_absolute() or not launcher.DEFAULT_PYTHON.is_file():
        raise HandoffError("exact pinned conda Python is unavailable")
    return env


def _supervisor_command(wave: Mapping[str, Any]) -> list[str]:
    root = Path(str(wave["artifact_root"])).resolve(strict=False)
    return [
        str(launcher.DEFAULT_PYTHON),
        "-m",
        "sua_exploration.rt_seed_robustness_annex_v2.supervisor",
        "--seed",
        str(wave["seed"]),
        "--fold",
        str(wave["fold"]),
        "--artifact-root",
        str(root),
        "--output",
        str(root / "wave_receipt.json"),
        "--execute",
    ]


def _assert_exact_campaign_plan(plan: Mapping[str, Any]) -> None:
    """Rebuild all ten deterministic waves and reject any supplied mutation."""
    root = _validate_campaign_root(str(plan.get("campaign_root", "")))
    nonce = plan.get("campaign_nonce")
    if not isinstance(nonce, str) or len(nonce) < 16:
        raise HandoffError("campaign plan lacks a fresh campaign nonce")
    receipt = _validate_supplemental_snapshot()
    if plan.get("implementation_snapshot_sha256") != receipt.get("implementation_snapshot_sha256"):
        raise HandoffError("campaign plan snapshot drift")
    expected: list[dict[str, Any]] = []
    for seed in spec.EXPECTED_SEEDS:
        for fold in spec.SELECTED_FOLDS:
            expected.append(supervisor.build_wave_plan(seed=seed, fold=fold,
                artifact_root=root / f"s{seed}_f{fold}_paired_wave",
                python_executable=str(launcher.DEFAULT_PYTHON), snapshot=receipt["implementation_snapshot"]))
    actual = plan.get("waves")
    if not isinstance(actual, list) or len(actual) != len(expected) or actual != expected:
        raise HandoffError("campaign plan exact ten-wave root/lane/snapshot binding drift")


def _checker_command(date: str) -> list[str]:
    command = [
        str(launcher.DEFAULT_PYTHON), str(CI_CHECKER_SCRIPT),
        "--ci-preflight", str(_ci_preflight_path(date)),
        "--five-date-aggregate", str(CI_SOURCE_AGGREGATE),
        "--launch-receipt", str(CI_LAUNCH_RECEIPT),
        "--output", str(_ci_checker_path(date)),
    ]
    for arm in CI_ARMS:
        command.extend([f"--{arm.lower()}", str(_ci_checkpoint_path(date, arm))])
    return command


def _evaluator_command(date: str) -> list[str]:
    return [
        str(launcher.DEFAULT_PYTHON), str(CI_EVALUATOR_SCRIPT),
        "--terminal-checker", str(_ci_checker_path(date)),
        "--data-dir", str(CI_DATA_DIR),
        "--output", str(_ci_terminal_path(date)),
        "--device", "cuda", "--execute-target-evaluation",
    ]


def _aggregate_command() -> list[str]:
    return [
        str(launcher.DEFAULT_PYTHON), str(CI_AGGREGATE_SCRIPT),
        "--evaluation-dir", str(CI_TERMINAL_ROOT),
        "--output", str(CI_TERMINAL_AGGREGATE),
    ]


def _run_stage(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]], command: Sequence[str], env: Mapping[str, str],
    stage: str, date: str | None = None,
) -> dict[str, Any]:
    completed = runner(
        list(command), cwd=CI_ROOT, env=dict(env), text=True, capture_output=True, check=False,
    )
    row = {
        "stage": stage, "date": date, "command": list(command), "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:],
    }
    if completed.returncode != 0:
        suffix = f" for {date}" if date is not None else ""
        raise HandoffError(f"canonical {stage} failed{suffix}")
    return row


def _terminal_binding_payload(*, date: str, terminal: Path, invocation: Path,
                              plan: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a legacy evaluator receipt to this single v2 invocation.

    The canonical evaluator format predates these fields, so the watcher seals
    them beside it rather than pretending older terminal JSON contained them.
    """
    _validate_ci_terminal(terminal, date)
    terminal_body = _json_object(terminal)
    target = terminal_body["target"]
    invocation_info = _regular(invocation, label="terminal invocation receipt", readonly=True)
    terminal_info = _regular(terminal, label="new canonical terminal receipt", readonly=True)
    if _ctime(terminal_info) <= _ctime(invocation_info):
        raise HandoffError(f"terminal receipt predates this invocation: {date}")
    checker = _validate_ci_checker(_ci_checker_path(date), date)
    checkpoints = {arm: {"checkpoint": _artifact_binding(_ci_checkpoint_path(date, arm), label="checker checkpoint"),
                         "config": _artifact_binding(_ci_config_path(date, arm), label="checker config")}
                   for arm in CI_ARMS}
    return {"schema": TERMINAL_BINDING_SCHEMA, "status": "PASS_NEW_TERMINAL_BOUND_TO_CURRENT_INVOCATION",
            "date": date, "campaign_nonce": plan["campaign_nonce"],
            "terminal": _artifact_binding(terminal, label="new canonical terminal receipt", readonly=True),
            "invocation_receipt": _artifact_binding(invocation, label="terminal invocation receipt", readonly=True),
            "terminal_checker": {"path": str(_ci_checker_path(date)), "sha256": checker},
            "source_checkpoint_and_config": checkpoints,
            "evaluator_source": _artifact_binding(CI_EVALUATOR_SCRIPT, label="evaluator source"),
            "evaluator_produced_target_input_sha256": dict(target["files"]),
            "target_data_metadata_manifest": _target_metadata_manifest(),
            "target_opened_by_watcher": False}


def _validate_terminal_binding(*, path: Path, date: str, terminal: Path,
                               invocation: Path, plan: Mapping[str, Any]) -> str:
    """Recompute every non-terminal binding in a v2 wrapper receipt."""
    _regular(path, label="terminal binding receipt", readonly=True)
    body = _json_object(path)
    expected = _terminal_binding_payload(date=date, terminal=terminal, invocation=invocation, plan=plan)
    # Recomputing is intentionally strict: a modified evaluator source,
    # target metadata, checkpoint/config, checker, terminal, nonce, or
    # invocation receipt changes the canonical JSON binding.
    if body != expected:
        raise HandoffError(f"terminal binding tamper or input drift: {date}")
    return _sha256(path)


def _terminalize_partition(
    *,
    plan: Mapping[str, Any], name: str, receipt_paths: Mapping[str, Path], source_evidence: Mapping[str, Any],
    env: Mapping[str, str], runner: Callable[..., subprocess.CompletedProcess[str]],
    compute_sample: Callable[[int], list[str] | None] = _compute_apps,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run an exited partition's fixed checker/evaluator date sequence once."""

    partition = CI_PARTITIONS[name]
    source_path = receipt_paths[f"source_release:{name}"]
    launch_path = receipt_paths[f"terminalization_launch:{name}"]
    completion_path = receipt_paths[f"terminalization_completion:{name}"]
    _write_exclusive_readonly(source_path, {
        "schema": SOURCE_COMPLETION_SCHEMA,
        "status": "PASS_PARTITION_SOURCE_RELEASED_FOR_TERMINALIZATION",
        "partition": name, "dates": list(partition["dates"]), "device": partition["device_index"],
        "source_release_evidence": dict(source_evidence),
        "implementation_snapshot_sha256": plan["implementation_snapshot_sha256"],
        "watcher_sha256": plan["watcher_sha256"], "gpu_launched_by_watcher": False,
        "campaign_nonce": plan["campaign_nonce"],
        "runner_exit_code_observed": False,
    })
    commands = {
        date: {"checker": _checker_command(date), "evaluator": _evaluator_command(date)}
        for date in partition["dates"]
    }
    _write_exclusive_readonly(launch_path, {
        "schema": "rt_seed_robustness_annex_v2_ci64_terminalization_launch_receipt_v1",
        "status": "PASS_PARTITION_TERMINALIZATION_AUTHORIZED_AFTER_SOURCE_RELEASE",
        "partition": name, "dates": list(partition["dates"]), "device": partition["device_index"],
        "source_release_receipt": str(source_path), "source_release_sha256": _sha256(source_path),
        "commands": commands, "pinned_python": str(launcher.DEFAULT_PYTHON),
        "checker_cuda_visible_devices": "", "evaluator_cuda_visible_devices": str(partition["device_index"]),
        "python_no_user_site": env["PYTHONNOUSERSITE"], "no_glob_mtime_or_score_selection": True,
        "campaign_nonce": plan["campaign_nonce"],
    })
    logs: list[dict[str, Any]] = []
    terminal_hashes: dict[str, str] = {}
    cpu_env = dict(env, CUDA_VISIBLE_DEVICES="")
    evaluator_env = dict(env, CUDA_VISIBLE_DEVICES=str(partition["device_index"]))
    for date in partition["dates"]:
        terminal = _ci_terminal_path(date)
        invocation = receipt_paths[f"terminal_invocation:{name}:{date}"]
        binding = receipt_paths[f"terminal_binding:{name}:{date}"]
        # Fresh v2 authority can never reuse a terminal artifact from an
        # older evaluator invocation, even when it happens to validate.
        if terminal.exists() or terminal.is_symlink():
            raise HandoffError(f"pre-existing canonical terminal is stale for this v2 campaign: {date}")
        checker = _ci_checker_path(date)
        if checker.exists() or checker.is_symlink():
            checker_sha = _validate_ci_checker(checker, date)
            logs.append({"stage": "terminal_checker", "date": date, "skipped_existing_valid_receipt": True, "receipt_sha256": checker_sha})
        else:
            logs.append(_run_stage(runner=runner, command=commands[date]["checker"], env=cpu_env, stage="terminal_checker", date=date))
            checker_sha = _validate_ci_checker(checker, date)
            logs[-1]["receipt_sha256"] = checker_sha
        # The CPU checker itself never initializes CUDA.  Re-probe and acquire
        # a persistent O_EXCL lease immediately before the target-opening
        # evaluator, closing check-to-launch TOCTOU.
        samples = two_empty_compute_samples(devices=(int(partition["device_index"]),), sample=compute_sample, sleep=sleep)
        logs.append({"stage": "pre_evaluator_gpu_recheck", "date": date, "compute_pid_samples": samples})
        _write_exclusive_readonly(invocation, {
            "schema": "rt_seed_robustness_annex_v2_ci64_terminalization_invocation_v2",
            "status": "AUTHORIZED_NEW_TERMINAL_ONLY", "date": date, "partition": name,
            "campaign_nonce": plan["campaign_nonce"], "terminal_must_be_absent": True,
            "checker_sha256": checker_sha, "command": commands[date]["evaluator"],
        })
        lease = _acquire_gpu_lease(receipt_root=launch_path.parent, device=int(partition["device_index"]),
                                   scope=f"terminal_{name}_{date}", compute_sample=compute_sample)
        logs.append(_run_stage(runner=runner, command=commands[date]["evaluator"], env=evaluator_env, stage="terminal_evaluate", date=date))
        _validate_ci_terminal(terminal, date)
        _write_exclusive_readonly(binding, _terminal_binding_payload(date=date, terminal=terminal, invocation=invocation, plan=plan))
        _validate_terminal_binding(path=binding, date=date, terminal=terminal, invocation=invocation, plan=plan)
        _write_gpu_release(lease=lease, receipt_root=launch_path.parent, device=int(partition["device_index"]),
                           scope=f"terminal_{name}_{date}")
        terminal_hashes[date] = _sha256(terminal)
        logs[-1]["receipt_sha256"] = terminal_hashes[date]
    _write_exclusive_readonly(completion_path, {
        "schema": "rt_seed_robustness_annex_v2_ci64_partition_terminalization_completion_receipt_v1",
        "status": "PASS_PARTITION_TERMINALIZATION_COMPLETE",
        "partition": name, "dates": list(partition["dates"]),
        "terminal_receipt_sha256": terminal_hashes, "launch_receipt": str(launch_path),
        "launch_receipt_sha256": _sha256(launch_path), "logs": logs,
        "implementation_snapshot_sha256": plan["implementation_snapshot_sha256"],
        "campaign_nonce": plan["campaign_nonce"],
    })
    return {"partition": name, "completion_receipt": str(completion_path), "terminal_receipt_sha256": terminal_hashes, "logs": logs}


def _finalize_rt_aggregate(*, plan: Mapping[str, Any], receipt_paths: Mapping[str, Path]) -> dict[str, Any]:
    """CPU-only canonical aggregation of exactly ten sealed paired waves."""
    rows: list[Mapping[str, Any]] = []
    waves: list[dict[str, Any]] = []
    for wave in plan["waves"]:
        seed, fold = int(wave["seed"]), int(wave["fold"])
        root = Path(str(wave["artifact_root"])).resolve()
        wave_receipt = root / "wave_receipt.json"
        _regular(wave_receipt, label="wave receipt", readonly=True)
        body = _json_object(wave_receipt)
        if (body.get("schema") != "rt_seed_robustness_annex_v2_1_wave_receipt_v1" or
            body.get("status") != "PASS_WAVE_COMPLETE_EXPLICITLY_AUTHORIZED" or
            int(body.get("seed", -1)) != seed or int(body.get("fold", -1)) != fold or
            body.get("aggregate_allowed") is not True):
            raise HandoffError(f"wave receipt contract mismatch: s{seed} f{fold}")
        cells: list[dict[str, Any]] = []
        for cell in wave["cells"]:
            if int(cell["seed"]) != seed or int(cell["fold"]) != fold:
                raise HandoffError("cell / wave identity mismatch")
            path = Path(str(cell["cell_receipt"])).resolve()
            _regular(path, label="cell receipt", readonly=True)
            cell_body = _json_object(path)
            # Cell receipts contain the metric row at top level in this
            # package.  No target data, CUDA, or checkpoint is opened here.
            if (int(cell_body.get("seed", -1)) != seed or int(cell_body.get("fold", -1)) != fold or
                cell_body.get("arm") != cell["arm"]):
                raise HandoffError(f"cell receipt identity mismatch: s{seed} f{fold}")
            source_initial = Path(str(cell_body.get("source_initial_state_receipt", ""))).resolve()
            selection = Path(str(cell_body.get("selection_receipt", ""))).resolve()
            outer_eval = Path(str(cell_body.get("outer_eval_receipt", ""))).resolve()
            split = Path(str(cell["split_manifest"])).resolve()
            paired = Path(str(cell_body.get("paired_initial_state_receipt", ""))).resolve()
            exact_paths = {
                "source-initial receipt": (source_initial, Path(str(cell["source_initial_receipt"])).resolve()),
                "selection receipt": (selection, Path(str(cell["selection_receipt"])).resolve()),
                "outer-evaluation receipt": (outer_eval, Path(str(cell["outer_eval_receipt"])).resolve()),
            }
            for label, (actual, expected) in exact_paths.items():
                if actual != expected:
                    raise HandoffError(f"cell {label} path drift: s{seed} f{fold}")
                _regular(actual, label=label, readonly=True)
            _regular(split, label="split manifest", readonly=True)
            _regular(paired, label="paired-initial-state receipt", readonly=True)
            if cell_body.get("source_split_manifest_sha256") != _sha256(split):
                raise HandoffError(f"cell split manifest SHA chain drift: s{seed} f{fold}")
            source_body = _json_object(source_initial)
            selection_body = _json_object(selection)
            paired_body = _json_object(paired)
            if (source_body.get("initial_state_hash") != cell_body.get("initial_state_hash") or
                selection_body.get("split_manifest_sha256") != _sha256(split) or
                str(paired_body.get("initial_state_hash", "")) != str(cell_body.get("initial_state_hash", ""))):
                raise HandoffError(f"cell intermediate receipt SHA/provenance chain drift: s{seed} f{fold}")
            rows.append(cell_body)
            cells.append({"cell": _artifact_binding(path, label="cell receipt", readonly=True),
                          "source_initial": _artifact_binding(source_initial, label="source-initial receipt", readonly=True),
                          "selection": _artifact_binding(selection, label="selection receipt", readonly=True),
                          "split": _artifact_binding(split, label="split manifest", readonly=True),
                          "paired": _artifact_binding(paired, label="paired-initial-state receipt", readonly=True),
                          "outer_eval": _artifact_binding(outer_eval, label="outer-evaluation receipt", readonly=True)})
        waves.append({"seed": seed, "fold": fold,
                      "wave_receipt": _artifact_binding(wave_receipt, label="wave receipt", readonly=True),
                      "cell_receipts": cells})
    try:
        aggregate = spec.aggregate_cells(rows)
    except spec.SpecError as error:
        raise HandoffError(f"canonical 20-cell aggregate rejected: {error}") from error
    payload = {"schema": RT_AGGREGATE_SCHEMA, "status": "PASS_COMPLETE_20_CELL_CPU_ONLY_AGGREGATE",
               "campaign_nonce": plan["campaign_nonce"], "wave_count": 10, "cell_count": 20,
               "waves": waves, "aggregate": aggregate,
               "nwb_opened_by_finalizer": False, "cuda_constructed_by_finalizer": False,
               "target_opened_by_finalizer": False}
    _write_exclusive_readonly(receipt_paths["rt_aggregate"], payload)
    return payload


def run_campaign(
    *,
    plan: Mapping[str, Any],
    receipt_root: str | Path,
    poll_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    source_gate: Callable[..., dict[str, Any]] = _partition_source_release_gate,
    annex_gate: Callable[..., dict[str, Any]] = _release_gate,
    terminalizer: Callable[..., dict[str, Any]] = _terminalize_partition,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    compute_sample: Callable[[int], list[str] | None] = _compute_apps,
) -> dict[str, Any]:
    """Terminalize CI64 by released partition, then execute the annex campaign."""

    if plan.get("status") != "DRY_RUN_NOT_ARMED" or int(plan.get("wave_count", -1)) != 10:
        raise HandoffError("campaign plan is not the exact static ten-wave plan")
    if plan.get("gpu_authorized") is not False or plan.get("gpu_launched") is not False:
        raise HandoffError("campaign plan is not inert")
    _assert_exact_campaign_plan(plan)
    env = _require_execution_environment()
    receipt_paths = _receipt_paths(receipt_root)
    campaign_root = Path(str(plan["campaign_root"])).resolve(strict=False)
    receipt_root_resolved = receipt_paths["annex_launch"].parent.resolve(strict=False)
    if (
        receipt_root_resolved == campaign_root
        or campaign_root in receipt_root_resolved.parents
        or receipt_root_resolved in campaign_root.parents
    ):
        raise HandoffError(
            "receipt root must be a separate external tree so it cannot create the fresh campaign root"
        )
    arm_path = receipt_paths["handoff_arm"]
    _write_exclusive_readonly(
        arm_path,
        {
            "schema": "rt_seed_robustness_annex_v2_ci64_handoff_arm_receipt_v1",
            "status": "ARMED_WAITING_FOR_PARTITION_SOURCE_RELEASE",
            "campaign_root": str(campaign_root),
            "receipt_root": str(receipt_root_resolved),
            "wave_count": 10,
            "partitions": CI_PARTITIONS,
            "pinned_python": str(launcher.DEFAULT_PYTHON),
            "python_no_user_site": env["PYTHONNOUSERSITE"],
            "implementation_snapshot_sha256": plan["implementation_snapshot_sha256"],
            "supplemental_receipt_sha256": plan["supplemental_receipt_sha256"],
            "watcher_sha256": plan["watcher_sha256"],
            "campaign_nonce": plan["campaign_nonce"],
            "gpu_launched_by_watcher": False,
            "nwb_read_by_watcher": False,
            "failure_policy": "fail_closed_no_terminalization_or_annex_on_any_drift",
        },
    )
    terminalized: dict[str, Any] = {}
    pending = set(CI_PARTITIONS)
    while pending:
        for name in tuple(sorted(pending)):
            source_evidence = source_gate(
                plan=plan, name=name, stability_seconds=1.0, sleep=sleep,
                compute_sample=compute_sample,
            )
            if source_evidence.get("eligible") is not True:
                if source_evidence.get("reason") != "WAITING_FOR_PARTITION_SOURCE":
                    raise HandoffError("source-release gate returned an unknown non-eligible state")
                continue
            terminalized[name] = terminalizer(
                plan=plan, name=name, receipt_paths=receipt_paths, source_evidence=source_evidence,
                env=env, runner=runner, compute_sample=compute_sample, sleep=sleep,
            )
            pending.remove(name)
        if pending:
            sleep(poll_seconds)
    # Both independently terminalized partitions must now bind all five exact
    # receipts before the CPU-only canonical aggregate may be constructed.
    terminal_hashes = validate_ci_terminals()
    aggregate_logs: list[dict[str, Any]] = []
    if CI_TERMINAL_AGGREGATE.exists() or CI_TERMINAL_AGGREGATE.is_symlink():
        raise HandoffError("pre-existing canonical five-date aggregate is stale for this v2 campaign")
    aggregate_logs.append(_run_stage(
        runner=runner, command=_aggregate_command(), env=dict(env, CUDA_VISIBLE_DEVICES=""), stage="five_date_aggregate",
    ))
    aggregate_sha = _validate_terminal_aggregate()
    aggregate_logs[-1]["receipt_sha256"] = aggregate_sha
    completion_path = receipt_paths["terminalization_completion"]
    _write_exclusive_readonly(completion_path, {
        "schema": "rt_seed_robustness_annex_v2_ci64_terminalization_completion_receipt_v1",
        "status": "PASS_CI64_FIVE_DATE_TERMINALIZATION_AND_CANONICAL_AGGREGATE_COMPLETE",
        "partition_terminalizations": terminalized,
        "terminal_receipt_sha256": terminal_hashes,
        "canonical_five_date_aggregate": {"path": str(CI_TERMINAL_AGGREGATE), "sha256": aggregate_sha},
        "aggregate_logs": aggregate_logs, "implementation_snapshot_sha256": plan["implementation_snapshot_sha256"],
        "score_based_path_selection": False,
        "campaign_nonce": plan["campaign_nonce"],
    })
    evidence = annex_gate(plan=plan, stability_seconds=1.0, sleep=sleep)
    if evidence.get("eligible") is not True:
        raise HandoffError("annex release gate did not pass after canonical terminalization")
    launch_path = receipt_paths["annex_launch"]
    launch_payload = {
        "schema": "rt_seed_robustness_annex_v2_ci64_handoff_launch_receipt_v1",
        "status": "PASS_CI64_TERMINAL_AGGREGATE_AND_SECOND_GPU_GATE_BEFORE_SUPERVISOR_LAUNCH",
        "terminalization_completion_receipt": str(completion_path),
        "terminalization_completion_sha256": _sha256(completion_path),
        "campaign_root": plan["campaign_root"],
        "wave_count": 10,
        "pinned_python": str(launcher.DEFAULT_PYTHON),
        "python_no_user_site": env["PYTHONNOUSERSITE"],
        "implementation_snapshot_sha256": plan["implementation_snapshot_sha256"],
        "watcher_sha256": plan["watcher_sha256"],
        "campaign_nonce": plan["campaign_nonce"],
        "release_evidence": evidence,
        "supervisor_commands": [_supervisor_command(wave) for wave in plan["waves"]],
        "gpu_launched_by_watcher": False,
        "nwb_read_by_watcher": False,
    }
    _write_exclusive_readonly(launch_path, launch_payload)
    logs: list[dict[str, Any]] = []
    for wave in plan["waves"]:
        command = _supervisor_command(wave)
        scope = f"wave_s{wave['seed']}_f{wave['fold']}"
        # This is the final check-to-launch boundary for both lanes.  Leases
        # remain in the external receipt tree after release; they are not a
        # reusable lock-file protocol.
        leases = [_acquire_gpu_lease(receipt_root=receipt_root_resolved, device=device,
                                     scope=scope, compute_sample=compute_sample)
                  for device in (0, 1)]
        completed = runner(
            command,
            cwd=launcher.ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        logs.append(
            {
                "seed": wave["seed"],
                "fold": wave["fold"],
                "command": command,
                "returncode": int(completed.returncode),
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        if completed.returncode != 0:
            raise HandoffError(
                f"paired supervisor wave failed: s{wave['seed']} f{wave['fold']}"
            )
        for device, lease in zip((0, 1), leases):
            _write_gpu_release(lease=lease, receipt_root=receipt_root_resolved, device=device, scope=scope)
    aggregate = _finalize_rt_aggregate(plan=plan, receipt_paths=receipt_paths)
    return {
        "status": "PASS_ALL_TEN_PAIRED_WAVES_SUPERVISED",
        "handoff_arm_receipt": str(arm_path),
        "terminalization_completion_receipt": str(completion_path),
        "launch_receipt": str(launch_path),
        "canonical_20_cell_aggregate_receipt": str(receipt_paths["rt_aggregate"]),
        "aggregate_label": aggregate["aggregate"]["label"],
        "waves": logs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.execute and not args.watch:
        parser.error("--execute requires --watch")
    if args.watch and not args.execute:
        parser.error("--watch requires --execute")
    if args.watch and args.receipt_root is None:
        parser.error("--watch --execute requires --receipt-root")
    if args.dry_run and args.watch:
        parser.error("--dry-run cannot be combined with --watch")
    plan = build_campaign_plan(campaign_root=args.campaign_root)
    if not args.watch:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    result = run_campaign(
        plan=plan,
        receipt_root=args.receipt_root,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
