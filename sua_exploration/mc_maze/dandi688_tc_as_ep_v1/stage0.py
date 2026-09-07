"""Source-only Stage-0 constructibility executor for TC-AS-EP V1."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import plan
from .data import (
    load_activity_authority,
    load_strict_source_roster,
    sha256_file,
    valid_query_starts_after_q,
)
from .selector import (
    select_cov_random10,
    selector_stage0_diagnostics,
)


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode("utf-8")


def _write_immutable_json(root: Path, name: str, payload: dict[str, Any]) -> str:
    body = _canonical_json(payload)
    digest = hashlib.sha256(body).hexdigest()
    target = root / name
    sidecar = root / f"{name}.sha256"
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to replace immutable leaf {target}")
    with tempfile.NamedTemporaryFile(dir=root, prefix=f".{name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, target)
    side_body = f"{digest}  {name}\n".encode("ascii")
    with tempfile.NamedTemporaryFile(dir=root, prefix=f".{name}.sha256.", delete=False) as handle:
        temporary_side = Path(handle.name)
        handle.write(side_body)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary_side, 0o444)
    os.replace(temporary_side, sidecar)
    return digest


def _entropy(items: list[tuple[int, ...]]) -> float:
    counts: dict[tuple[int, ...], int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    total = float(len(items))
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _rng_snapshot() -> dict[str, object]:
    payload: dict[str, object] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    try:
        import torch

        payload["torch"] = torch.random.get_rng_state().clone()
    except ModuleNotFoundError:
        payload["torch"] = None
    return payload


def _rng_equal(left: dict[str, object], right: dict[str, object]) -> bool:
    np_left, np_right = left["numpy"], right["numpy"]
    assert isinstance(np_left, tuple) and isinstance(np_right, tuple)
    numpy_equal = (
        np_left[0] == np_right[0]
        and np.array_equal(np_left[1], np_right[1])
        and np_left[2:] == np_right[2:]
    )
    python_equal = left["python"] == right["python"]
    if left["torch"] is None or right["torch"] is None:
        torch_equal = left["torch"] is right["torch"]
    else:
        import torch

        torch_equal = bool(torch.equal(left["torch"], right["torch"]))
    return bool(numpy_equal and python_equal and torch_equal)


def _verify_static_inputs(repo_root: Path) -> dict[str, object]:
    design = repo_root / plan.DESIGN_RELATIVE
    workorder = repo_root / plan.STAGE0_ATTEMPT2_WORKORDER_RELATIVE
    manifest = repo_root / plan.STRICT_MANIFEST_RELATIVE
    teacher = repo_root / plan.TEACHER_RELATIVE
    parity_checkpoint = repo_root / plan.PARITY_CHECKPOINT_RELATIVE
    expected = {
        "design": (design, plan.DESIGN_SHA256),
        "stage0_attempt2_workorder": (
            workorder, plan.STAGE0_ATTEMPT2_WORKORDER_SHA256
        ),
        "strict_manifest": (manifest, plan.STRICT_MANIFEST_SHA256),
        "teacher": (teacher, plan.TEACHER_SHA256),
        "parity_checkpoint": (parity_checkpoint, plan.PARITY_CHECKPOINT_SHA256),
    }
    receipt: dict[str, object] = {}
    for label, (path, expected_sha) in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected_sha:
            raise ValueError(f"{label} SHA drift: {observed} != {expected_sha}")
        receipt[label] = {
            "relative_path": str(path.relative_to(repo_root)),
            "sha256": observed,
        }
    attempt1_root = repo_root / plan.STAGE0_ATTEMPT1_RELATIVE
    for name, expected_sha in plan.STAGE0_ATTEMPT1_BODY_SHA256.items():
        path = attempt1_root / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"Stage-0 attempt-1 predecessor drift: {name}")
    receipt["stage0_attempt1_failure_predecessor"] = {
        "relative_root": plan.STAGE0_ATTEMPT1_RELATIVE,
        "body_sha256": dict(plan.STAGE0_ATTEMPT1_BODY_SHA256),
        "decision": "FAIL",
        "failed_gate": "one_of_six_abs_point_biserial_duration_above_0.20",
    }
    return receipt


def execute_selector_stage0(repo_root: Path, *, result_root: Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    result_root = (
        Path(result_root).resolve()
        if result_root is not None
        else (repo_root / plan.STAGE0_RESULT_RELATIVE)
    )
    if result_root.exists():
        raise FileExistsError(f"Stage-0 root already exists: {result_root}")
    result_root.parent.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(mode=0o755)

    attempt = {
        "schema_version": plan.SCHEMA_VERSION,
        "route": plan.ROUTE_NAME,
        "stage": "selector_stage0_attempt2",
        "status": "STARTED",
        "created_at": datetime.now().astimezone().isoformat(),
        "performance_metric_opened": False,
        "decoder_opened": False,
        "gpu_opened": False,
        "formal_or_external_target_opened": False,
        "candidate_pool_n": plan.CANDIDATE_POOL_N,
        "activity_support_n": plan.ACTIVITY_SUPPORT_N,
        "query_start_trial": plan.QUERY_START_TRIAL,
    }
    attempt_sha = _write_immutable_json(result_root, "attempt.json", attempt)
    try:
        static_inputs = _verify_static_inputs(repo_root)
        roster = load_strict_source_roster(repo_root)
        authorized = tuple(roster["train"] + roster["val"])
        per_session: list[dict[str, object]] = []
        development_diagnostics: dict[str, dict[str, object]] = {}
        random_subsets: dict[str, list[tuple[int, ...]]] = {}
        rng_before = _rng_snapshot()

        for session_id in authorized:
            authority = load_activity_authority(repo_root, session_id)
            query_starts = valid_query_starts_after_q(repo_root, session_id)
            if query_starts.size == 0:
                raise ValueError(f"{session_id}: no Q50 query windows")
            session_receipt = authority.receipt()
            session_receipt.update({
                "split": "train" if session_id in roster["train"] else "val",
                "query_start_trial": plan.QUERY_START_TRIAL,
                "query_window_count": int(query_starts.size),
                "first_query_window_start": int(query_starts[0]),
                "last_query_window_start": int(query_starts[-1]),
            })
            per_session.append(session_receipt)
            if session_id in roster["val"]:
                diagnostics = selector_stage0_diagnostics(authority)
                development_diagnostics[session_id] = diagnostics
                draws = [
                    select_cov_random10(
                        authority,
                        training_seed=plan.PRIMARY_SEED,
                        epoch=0,
                        sample_or_window_id=f"stage0-window-{sample}",
                    ).indices
                    for sample in range(64)
                ]
                random_subsets[session_id] = draws
            print(
                f"STAGE0_SOURCE {session_id} units={authority.rates_hz.shape[1]} "
                f"invalid_dir={int(np.sum(authority.directions == -1))} "
                f"q50_windows={query_starts.size}",
                flush=True,
            )

        rng_after = _rng_snapshot()
        rng_unchanged = _rng_equal(rng_before, rng_after)
        random_receipt = {
            session_id: {
                "draw_count": len(draws),
                "unique_subset_count": len(set(draws)),
                "subset_entropy_nats": _entropy(draws),
                "all_cardinality_10": all(len(indices) == 10 for indices in draws),
                "all_unique": all(len(set(indices)) == 10 for indices in draws),
                "all_sorted": all(tuple(sorted(indices)) == indices for indices in draws),
                "subset_sha256": hashlib.sha256(
                    _canonical_json([list(indices) for indices in draws])
                ).hexdigest(),
            }
            for session_id, draws in random_subsets.items()
        }
        selector_pass = all(
            bool(row["passed"]) for row in development_diagnostics.values()
        )
        random_pass = bool(
            rng_unchanged
            and all(
                row["unique_subset_count"] > 1
                and row["subset_entropy_nats"] > 0.0
                and row["all_cardinality_10"]
                and row["all_unique"]
                and row["all_sorted"]
                for row in random_receipt.values()
            )
        )
        invalid_20150313 = next(
            row["invalid_direction_count"]
            for row in per_session
            if row["session_id"] == "sub-C_ses-CO-20150313"
        )
        authority_pass = bool(
            len(per_session) == 33
            and all(row["candidate_pool_n"] == 50 for row in per_session)
            and all(row["query_window_count"] > 0 for row in per_session)
            and invalid_20150313 == 1
        )
        payload = {
            "schema_version": plan.SCHEMA_VERSION,
            "route": plan.ROUTE_NAME,
            "stage": "selector_and_three_axis_source_authority",
            "status": "PASS" if selector_pass and random_pass and authority_pass else "FAIL",
            "attempt_sha256": attempt_sha,
            "static_inputs": static_inputs,
            "source_roster": {
                "train": list(roster["train"]),
                "val": list(roster["val"]),
                "test_names_not_resolved_or_opened": list(roster["test"]),
            },
            "axis_contract": {
                "candidate_pool_n": plan.CANDIDATE_POOL_N,
                "activity_support_n": plan.ACTIVITY_SUPPORT_N,
                "query_start_trial": plan.QUERY_START_TRIAL,
                "independently_represented": True,
            },
            "per_session": per_session,
            "development_selector_diagnostics": development_diagnostics,
            "random_support": random_receipt,
            "global_rng_streams_unchanged": rng_unchanged,
            "gates": {
                "source_authority_pass": authority_pass,
                "selector_bias_stability_pass": selector_pass,
                "random_support_pass": random_pass,
            },
            "performance_metric_opened": False,
            "decoder_opened": False,
            "gpu_opened": False,
            "formal_or_external_target_opened": False,
        }
        authority_sha = _write_immutable_json(
            result_root, "selector_authority.json", payload
        )
        decision = {
            "schema_version": plan.SCHEMA_VERSION,
            "route": plan.ROUTE_NAME,
            "stage": "selector_stage0_attempt2",
            "decision": payload["status"],
            "attempt_sha256": attempt_sha,
            "selector_authority_sha256": authority_sha,
            "operator_parity_pending": True,
            "gpu_capability_issued": False,
            "decoder_r2_computed": False,
        }
        _write_immutable_json(result_root, "selector_decision.json", decision)
        return {"root": str(result_root), **decision}
    except BaseException as exc:
        failure = {
            "schema_version": plan.SCHEMA_VERSION,
            "route": plan.ROUTE_NAME,
            "stage": "selector_stage0_attempt2",
            "status": "FAILED_EXCEPTION",
            "attempt_sha256": attempt_sha,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "performance_metric_opened": False,
            "decoder_r2_computed": False,
            "gpu_opened": False,
            "formal_or_external_target_opened": False,
        }
        _write_immutable_json(result_root, "failure.json", failure)
        raise
