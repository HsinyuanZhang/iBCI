#!/usr/bin/env python3
"""Score-blind post-completion orchestrator for the C1 shared-Z4 matrix.

The six terminal evaluations are independent, write-once cells.  This parent
process never parses a cell score artifact or evaluator output.  A successful
cell is committed by sealing the artifact and writing a score-blind sidecar
that binds its path, byte count, and SHA-256 to one verified score-blind
terminal completion receipt.  On continuation, an artifact without its sidecar (or vice versa) is
an orphan and fails closed.

Only after all six sidecars validate does this program invoke the existing
three-arm aggregator.  The aggregator is the first process allowed to parse
the shared-Z4 score artifacts or open the T4/TS4 result artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "sua_exploration/scripts"
sys.path.insert(0, str(SCRIPTS))

import shared_zero4_terminal_completion_bridge as completion_bridge


SEEDS = (42, 43, 44)
VIEWS = ("sua", "pseudo_mua")
EXPECTED_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
RUN_NAME = "t4_paired_view_c1_shared_zero4_source_prelaunch_v4_20260805_shared_zero4_s{seed}"
COMMIT_SCHEMA = "paired_view_c1_shared_zero4_terminal_score_blind_commit_v1"
EVALUATOR = SCRIPTS / "eval_paired_view_c1_shared_zero4_terminal.py"
AGGREGATOR = SCRIPTS / "aggregate_t4_paired_view_c1_three_arm_terminal.py"
LEGACY_AGGREGATOR = SCRIPTS / "aggregate_t4_paired_view_c1.py"
DIRECT_RECOVERY_COMPLETION = SCRIPTS / "shared_zero4_direct_recovery_completion.py"
BRIDGE = Path(completion_bridge.__file__).resolve()
ORCHESTRATOR = Path(__file__).resolve()
SEALED_FILE_MODE = 0o444


class OrchestrationError(RuntimeError):
    """The score-blind orchestration contract could not be satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_no_symlink(path: Path, *, role: str, sealed: bool) -> Path:
    if path.is_symlink():
        raise OrchestrationError(f"{role} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise OrchestrationError(f"{role} is missing: {path}") from exc
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or resolved.is_symlink():
        raise OrchestrationError(f"{role} is not a regular file: {resolved}")
    if sealed and stat.S_IMODE(info.st_mode) != SEALED_FILE_MODE:
        raise OrchestrationError(f"{role} mode must be 0444: {resolved}")
    return resolved


def _file_metadata(path: Path, *, role: str, sealed: bool = True) -> dict[str, Any]:
    resolved = _regular_no_symlink(path, role=role, sealed=sealed)
    return {
        "canonical_path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "mode": f"{stat.S_IMODE(resolved.stat().st_mode):04o}",
    }


def _controlled_sources() -> dict[str, dict[str, Any]]:
    """Pin every source that accepts a cell or computes the final aggregate."""

    return {
        "completion_bridge": _file_metadata(
            BRIDGE, role="completion bridge source", sealed=False
        ),
        "direct_recovery_completion": _file_metadata(
            DIRECT_RECOVERY_COMPLETION,
            role="direct-recovery completion verifier source",
            sealed=False,
        ),
        "orchestrator": _file_metadata(
            ORCHESTRATOR, role="post-completion orchestrator source", sealed=False
        ),
        "terminal_evaluator": _file_metadata(
            EVALUATOR, role="fixed terminal evaluator source", sealed=False
        ),
        "three_arm_aggregator": _file_metadata(
            AGGREGATOR, role="fixed three-arm aggregator source", sealed=False
        ),
        "legacy_paired_view_aggregator": _file_metadata(
            LEGACY_AGGREGATOR,
            role="legacy paired-view statistics/provenance source",
            sealed=False,
        ),
    }


def _require_sources_unchanged(expected: Mapping[str, Any], *, stage: str) -> None:
    if _controlled_sources() != dict(expected):
        raise OrchestrationError(f"controlled source drift before {stage}")


def _write_json_exclusive_sealed(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, SEALED_FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
    except BaseException:
        # The fd is owned by fdopen once entered.  Preserve any partial file as
        # evidence; continuation will reject it rather than overwrite it.
        raise
    os.chmod(path, SEALED_FILE_MODE)


def _load_sidecar(path: Path) -> dict[str, Any]:
    resolved = _regular_no_symlink(path, role="cell score-blind commit", sealed=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestrationError(f"invalid score-blind commit: {resolved}") from exc
    if not isinstance(value, dict):
        raise OrchestrationError(f"score-blind commit is not a JSON object: {resolved}")
    return value


def _artifact_path(root: Path, *, seed: int, view: str) -> Path:
    return root / "artifacts" / f"shared_zero4_s{seed}_{view}.json"


def _commit_path(root: Path, *, seed: int, view: str) -> Path:
    return root / "commits" / f"shared_zero4_s{seed}_{view}.commit.json"


def _verify_completion_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise OrchestrationError("terminal completion receipt must not be a symlink")
    try:
        probe = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestrationError("terminal completion receipt is missing or invalid JSON") from exc
    if not isinstance(probe, Mapping):
        raise OrchestrationError("terminal completion receipt must be a JSON object")
    if completion_bridge.is_direct_recovery_payload(probe):
        return completion_bridge.verify_direct_recovery_receipt(path)
    if completion_bridge.is_v3_adapter_payload(probe):
        return completion_bridge.verify_v3_adapter_receipt(path)
    raise OrchestrationError("unsupported terminal completion receipt schema")


def _completion_run_dir(verified: Mapping[str, Any], *, seed: int) -> Path:
    rows = verified.get("rows")
    row = rows.get(str(seed)) if isinstance(rows, Mapping) else None
    metadata = row.get("run_metadata") if isinstance(row, Mapping) else None
    terminal = row.get("terminal_checkpoint") if isinstance(row, Mapping) else None
    if not isinstance(metadata, Mapping) or not isinstance(terminal, Mapping):
        raise OrchestrationError(f"verified adapter lacks seed{seed} run bindings")
    metadata_path = Path(str(metadata.get("canonical_path", ""))).resolve(strict=True)
    terminal_path = Path(str(terminal.get("canonical_path", ""))).resolve(strict=True)
    run_dir = metadata_path.parent
    if (verified.get("kind") == "v3_external_v7_adapter" or seed in (42, 43)) and (
        run_dir.name != RUN_NAME.format(seed=seed)
    ):
        raise OrchestrationError(f"seed{seed} frozen V2 run-directory name drift")
    if terminal_path != run_dir / "epoch_ckpts/epoch_011.ckpt":
        raise OrchestrationError(f"seed{seed} frozen terminal checkpoint path drift")
    completion_bridge.require_supported_run_binding(
        verified,
        run_dir=run_dir,
        metadata_path=metadata_path,
        terminal_checkpoint=terminal_path,
        seed=seed,
    )
    return run_dir


def _expected_commit(
    *,
    seed: int,
    view: str,
    artifact: Path,
    artifact_metadata: Mapping[str, Any],
    completion_metadata: Mapping[str, Any],
    completion_identity: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    run_dir: Path,
    controlled_sources: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    return {
        "schema": COMMIT_SCHEMA,
        "status": "completed_terminal_cell_score_blind",
        "seed": seed,
        "view": view,
        "artifact": dict(artifact_metadata),
        "completion_receipt": dict(completion_metadata),
        "completion_identity": dict(completion_identity),
        "train_val_manifest": dict(manifest_metadata),
        "run_dir": str(run_dir),
        "controlled_sources": dict(controlled_sources),
        "device": device,
        "frozen_evaluator_arguments": {
            "calibration_n": 30,
            "query_start_trial": 50,
            "pool_size": 50,
            "checkpoint": "epoch_ckpts/epoch_011.ckpt",
            "signal_view": view,
        },
        "evaluator_returncode": 0,
        "parent_parsed_artifact": False,
        "parent_parsed_evaluator_stdout_or_stderr": False,
        "partial_score_printed_by_parent": False,
        "formal_or_subm_data_opened_by_parent": False,
        "artifact_write_policy": "exclusive_by_evaluator_then_sealed_0444",
        "continuation_policy": "artifact_and_commit_both_required_exact_digest_no_overwrite",
    }


def _validate_existing_cell(
    *,
    root: Path,
    seed: int,
    view: str,
    completion_metadata: Mapping[str, Any],
    completion_identity: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    run_dir: Path,
    controlled_sources: Mapping[str, Any],
    device: str,
) -> bool:
    """Validate a committed cell without parsing its score JSON."""

    artifact = _artifact_path(root, seed=seed, view=view)
    commit = _commit_path(root, seed=seed, view=view)
    artifact_exists = artifact.exists() or artifact.is_symlink()
    commit_exists = commit.exists() or commit.is_symlink()
    if artifact_exists != commit_exists:
        raise OrchestrationError(f"orphan artifact/commit for seed{seed}/{view}")
    if not artifact_exists:
        return False
    artifact_metadata = _file_metadata(
        artifact, role=f"seed{seed}/{view} terminal artifact", sealed=True
    )
    observed = _load_sidecar(commit)
    expected = _expected_commit(
        seed=seed,
        view=view,
        artifact=artifact,
        artifact_metadata=artifact_metadata,
        completion_metadata=completion_metadata,
        completion_identity=completion_identity,
        manifest_metadata=manifest_metadata,
        run_dir=run_dir,
        controlled_sources=controlled_sources,
        device=device,
    )
    if observed != expected:
        raise OrchestrationError(f"score-blind commit binding drift for seed{seed}/{view}")
    return True


def _run_quiet(command: Sequence[str]) -> int:
    completed = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return int(completed.returncode)


def run(args: argparse.Namespace) -> dict[str, Any]:
    adapter_path = args.matrix_completion_receipt.expanduser()
    # This is the first semantic gate.  No evaluator/aggregator module is
    # imported and no subprocess is called unless the complete transitive V3
    # supported completion validates.
    verified = _verify_completion_receipt(adapter_path)
    if verified.get("kind") not in {
        "v3_external_v7_adapter", "direct_recovery_content_bound_v1"
    }:
        raise OrchestrationError("unsupported verified terminal completion kind")
    adapter_path = adapter_path.resolve(strict=True)
    completion_metadata = _file_metadata(
        adapter_path, role="terminal completion receipt", sealed=True
    )
    completion_identity = {
        "kind": verified["kind"],
        "schema": verified["schema"],
        "status": verified["status"],
        "canonical_path": str(adapter_path),
        "sha256": completion_metadata["sha256"],
    }

    manifest = _regular_no_symlink(
        args.train_val_manifest.expanduser(), role="fixed train/validation manifest", sealed=False
    )
    manifest_metadata = _file_metadata(manifest, role="fixed train/validation manifest", sealed=False)
    if manifest_metadata["sha256"] != EXPECTED_MANIFEST_SHA256:
        raise OrchestrationError("fixed train/validation manifest SHA drift")

    controlled_sources = _controlled_sources()
    root = args.zero4_result_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    aggregate_out = args.aggregate_out.expanduser().resolve()
    if aggregate_out.exists() or aggregate_out.is_symlink():
        raise FileExistsError(f"write-once three-arm aggregate exists: {aggregate_out}")

    run_dirs = {seed: _completion_run_dir(verified, seed=seed) for seed in SEEDS}
    complete: list[tuple[int, str]] = []
    missing: list[tuple[int, str]] = []
    for seed in SEEDS:
        for view in VIEWS:
            if _validate_existing_cell(
                root=root,
                seed=seed,
                view=view,
                completion_metadata=completion_metadata,
                completion_identity=completion_identity,
                manifest_metadata=manifest_metadata,
                run_dir=run_dirs[seed],
                controlled_sources=controlled_sources,
                device=args.device,
            ):
                complete.append((seed, view))
            else:
                missing.append((seed, view))

    python = _regular_no_symlink(
        args.python.expanduser(), role="Python executable", sealed=False
    )
    for seed, view in missing:
        _require_sources_unchanged(
            controlled_sources, stage=f"terminal evaluator seed{seed}/{view}"
        )
        artifact = _artifact_path(root, seed=seed, view=view)
        commit = _commit_path(root, seed=seed, view=view)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        commit.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            str(EVALUATOR),
            "--run-dir", str(run_dirs[seed]),
            "--signal-view", view,
            "--out-path", str(artifact),
            "--matrix-completion-receipt", str(adapter_path),
            "--train-val-manifest", str(manifest),
            "--calibration-n", "30",
            "--query-start-trial", "50",
            "--pool-size", "50",
            "--device", args.device,
        ]
        returncode = _run_quiet(command)
        if returncode != 0:
            raise OrchestrationError(
                f"terminal evaluator failed for seed{seed}/{view} returncode={returncode}"
            )
        if artifact.is_symlink() or not artifact.is_file():
            raise OrchestrationError(
                f"terminal evaluator returned success without regular artifact for seed{seed}/{view}"
            )
        os.chmod(artifact, SEALED_FILE_MODE)
        artifact_metadata = _file_metadata(
            artifact, role=f"seed{seed}/{view} terminal artifact", sealed=True
        )
        sidecar = _expected_commit(
            seed=seed,
            view=view,
            artifact=artifact,
            artifact_metadata=artifact_metadata,
            completion_metadata=completion_metadata,
            completion_identity=completion_identity,
            manifest_metadata=manifest_metadata,
            run_dir=run_dirs[seed],
            controlled_sources=controlled_sources,
            device=args.device,
        )
        _write_json_exclusive_sealed(commit, sidecar)
        complete.append((seed, view))

    if set(complete) != {(seed, view) for seed in SEEDS for view in VIEWS}:
        raise OrchestrationError("internal completion accounting did not reach 6/6")

    # Revalidate all six digests immediately before crossing the disclosure
    # boundary.  No score artifact has been JSON-parsed by this parent.
    for seed in SEEDS:
        for view in VIEWS:
            if not _validate_existing_cell(
                root=root,
                seed=seed,
                view=view,
                completion_metadata=completion_metadata,
                completion_identity=completion_identity,
                manifest_metadata=manifest_metadata,
                run_dir=run_dirs[seed],
                controlled_sources=controlled_sources,
                device=args.device,
            ):
                raise OrchestrationError(f"cell disappeared before aggregation: seed{seed}/{view}")

    _require_sources_unchanged(controlled_sources, stage="three-arm aggregation")
    aggregate_out.parent.mkdir(parents=True, exist_ok=True)
    aggregate_command = [
        str(python),
        str(AGGREGATOR),
        "--t4-ts4-receipt", str(args.t4_ts4_receipt.expanduser().resolve()),
        "--t4-ts4-result-root", str(args.t4_ts4_result_root.expanduser().resolve()),
        "--zero4-result-root", str(root),
        "--zero4-matrix-completion-receipt", str(adapter_path),
        "--out", str(aggregate_out),
    ]
    returncode = _run_quiet(aggregate_command)
    if returncode != 0:
        raise OrchestrationError(f"three-arm aggregator failed returncode={returncode}")
    aggregate_metadata = _file_metadata(
        aggregate_out, role="three-arm terminal aggregate", sealed=True
    )
    return {
        "status": "completed_six_terminal_cells_and_three_arm_aggregate",
        "completed_cell_count": 6,
        "aggregate": aggregate_metadata,
        "controlled_sources": controlled_sources,
        "partial_score_printed_by_parent": False,
        "parent_parsed_cell_score_artifacts": False,
        "formal_or_subm_data_opened_by_parent": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-completion-receipt", type=Path, required=True)
    parser.add_argument("--train-val-manifest", type=Path, required=True)
    parser.add_argument("--zero4-result-root", type=Path, required=True)
    parser.add_argument("--t4-ts4-receipt", type=Path, required=True)
    parser.add_argument("--t4-ts4-result-root", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except Exception as exc:
        # Errors deliberately contain only control-plane identities; evaluator
        # stdout/stderr and cell payloads are never echoed here.
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
