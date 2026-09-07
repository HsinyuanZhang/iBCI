"""Additive Stage-0 diagnostics that never alter the frozen mask or gates.

Each session is reduced with one call to the reviewed raw-spike interval-rate
primitive.  Candidate, odd/even reliability, independent reference, shuffled
label, and pseudo-MUA views are then refit from disjoint slices of those same
rates.  The small returned arrays are merged in manifest order by the parent.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from . import plan
from .core import array_sha256, fit_source_normalizer, normalize_columns, require, row_shuffle_profile
from .lifecycle import begin_attempt, close_failure, close_terminal, publish_immutable_json
from .reliability import aggregate_column_reliability, rowwise_pearson
from .stage0 import _manifest_roster, _source_path, verify_frozen_inputs


def _shuffle_seed(session_id: str) -> int:
    body = f"{plan.ROUTE_NAME}/STAGE0_DIRECTION_SHUFFLE/{session_id}".encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "little")


def _held_immutable_bytes(path: Path) -> bytes:
    """Read one predecessor leaf without following a replaced symlink."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"held leaf is not regular: {path.name}")
    require(stat.S_IMODE(before.st_mode) == 0o444, f"held leaf mode drift: {path.name}")
    require(before.st_nlink == 1, f"held leaf link-count drift: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        held = os.fstat(descriptor)
        require(
            (held.st_dev, held.st_ino, held.st_size, held.st_mode, held.st_nlink)
            == (before.st_dev, before.st_ino, before.st_size, before.st_mode, before.st_nlink),
            f"held leaf changed during open: {path.name}",
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mode, after.st_nlink)
            == (held.st_dev, held.st_ino, held.st_size, held.st_mode, held.st_nlink),
            f"held leaf changed while reading: {path.name}",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _one_session(task: tuple[str, str, str]) -> dict[str, Any]:
    """Worker payload.  It reads no dense behavior array and opens no model."""
    session_id, split, path_text = task
    path = Path(path_text)
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix
    from .descriptors import (
        _default_electrode_ids,
        classify_era_from_first30,
        refit_from_single_pool,
        single_pool_interval_layout,
    )

    trials = list_datamodule_rewarded_trials(
        path,
        bin_size_ms=plan.BIN_SIZE_MS,
        window_size=plan.WINDOW_SIZE_BINS,
        trial_result_filter=plan.REWARDED_RESULT,
    )
    # Validation consumes only its legal candidate M10 annotations.  Odd/even
    # M30 and trials-50:109 references are train-only audit authorities.
    groups = ("candidate",) if split == "val" else (
        "candidate", "odd_m30", "even_m30", "reference_50_109"
    )
    intervals, slices = single_pool_interval_layout(trials, groups=groups)
    rates, unit_count = _pool_trial_rate_matrix(path, intervals)
    require(rates.ndim == 2 and rates.shape[1] == len(intervals), f"{session_id}: pooled-rate geometry")
    require(int(unit_count) == rates.shape[0], f"{session_id}: unit-count drift")

    whole, post, candidate = refit_from_single_pool(rates, trials, slices, group="candidate")
    odd = even = reference = ref_whole = None
    if split == "train":
        _odd_whole, _odd_post, odd = refit_from_single_pool(rates, trials, slices, group="odd_m30")
        _even_whole, _even_post, even = refit_from_single_pool(rates, trials, slices, group="even_m30")
        ref_whole, _ref_post, reference = refit_from_single_pool(rates, trials, slices, group="reference_50_109")

    direction_seed = _shuffle_seed(session_id)
    _dw, _dp, direction_shuffled = refit_from_single_pool(
        rates,
        trials,
        slices,
        group="candidate",
        direction_permutation_seed=direction_seed,
    )
    electrode_ids = _default_electrode_ids(path)
    _pw, _pp, pseudo = refit_from_single_pool(
        rates,
        trials,
        slices,
        group="candidate",
        signal_view="pseudo_mua",
        electrode_ids=electrode_ids,
    )
    shuffled, permutation = row_shuffle_profile(
        candidate,
        session_id=session_id,
        view="sua",
        training_seed=42,
    )
    repeat, repeat_permutation = row_shuffle_profile(
        candidate,
        session_id=session_id,
        view="sua",
        training_seed=42,
    )
    require(np.array_equal(shuffled, repeat) and np.array_equal(permutation, repeat_permutation), f"{session_id}: row shuffle nondeterministic")
    require(not np.array_equal(permutation, np.arange(permutation.size)), f"{session_id}: row shuffle identity")
    direction_shuffle_correlation = rowwise_pearson(candidate, direction_shuffled)

    return {
        "session": session_id,
        "split": split,
        "era": classify_era_from_first30(trials),
        "candidate": candidate,
        "whole": whole,
        "post": post,
        "odd": odd,
        "even": even,
        "reference": reference,
        "reference_whole": ref_whole,
        "interval_count": len(intervals),
        "interval_layout_sha256": hashlib.sha256(
            json.dumps(
                {key: [value.start, value.stop] for key, value in sorted(slices.items())},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "raw_rate_sha256": array_sha256(rates),
        "direction_shuffle_seed": direction_seed,
        "direction_shuffled_profile_sha256": array_sha256(direction_shuffled),
        "direction_shuffle_column_correlation": [
            float(value) if np.isfinite(value) else None for value in direction_shuffle_correlation
        ],
        "row_shuffle_permutation": permutation.tolist(),
        "row_shuffled_profile_sha256": array_sha256(shuffled),
        "pseudo_mua_profile_sha256": array_sha256(pseudo),
        "pseudo_mua_rows": int(pseudo.shape[0]),
        "sua_rows": int(candidate.shape[0]),
        "dense_velocity_scalars_consumed": 0,
        "audit_labels_after_m10_consumed": split == "train",
    }


def _column_vectors(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, list[float]]:
    output = {name: [] for name in plan.PROFILE_COLUMN_NAMES}
    for row in rows:
        values = rowwise_pearson(np.asarray(row[left]), np.asarray(row[right]))
        for index, name in enumerate(plan.PROFILE_COLUMN_NAMES):
            output[name].append(float(values[index]))
    return output


def _descriptive(values: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, raw in values.items():
        vector = np.asarray(raw, dtype=np.float64)
        finite = vector[np.isfinite(vector)]
        result[name] = {
            "finite_session_count": int(finite.size),
            "mean_raw_r": float(np.mean(finite)) if finite.size else float("nan"),
            "median_raw_r": float(np.median(finite)) if finite.size else float("nan"),
            "positive_session_count": int(np.count_nonzero(finite > 0.0)),
        }
    return result


def _require_primary_match(observed: Mapping[str, Mapping[str, object]], expected: Mapping[str, Mapping[str, object]], label: str) -> None:
    require(set(observed) == set(expected) == set(plan.PROFILE_COLUMN_NAMES), f"{label}: column roster drift")
    for name in plan.PROFILE_COLUMN_NAMES:
        for key in ("finite_session_count", "positive_session_count", "passed"):
            require(observed[name][key] == expected[name][key], f"{label}/{name}/{key}: primary mismatch")
        for key in ("fisher_z_aggregate_r", "median_raw_r"):
            require(np.isclose(float(observed[name][key]), float(expected[name][key]), rtol=0.0, atol=1e-7, equal_nan=True), f"{label}/{name}/{key}: primary mismatch")


def execute_stage0_supplement(
    repo_root: Path,
    result_root: Path,
    *,
    max_workers: int | None = None,
) -> dict[str, object]:
    """Execute the additive source-only supplement in a fresh immutable root."""
    repo_root = Path(repo_root).resolve()
    result_root = Path(result_root).resolve()
    static = verify_frozen_inputs(repo_root)
    roster, splits = _manifest_roster(repo_root)
    primary_root = repo_root / plan.STAGE0_RELATIVE
    expected_leaves = {
        "attempt.json", "attempt.json.sha256", "stage0.json", "stage0.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    require(primary_root.is_dir(), "canonical Stage0 root is unavailable")
    require({path.name for path in primary_root.iterdir()} == expected_leaves, "canonical Stage0 topology drift")
    primary_digests: dict[str, str] = {}
    primary_bodies: dict[str, bytes] = {}
    for name in ("attempt.json", "stage0.json", "terminal.json"):
        body_path = primary_root / name
        body = _held_immutable_bytes(body_path)
        sidecar = _held_immutable_bytes(primary_root / f"{name}.sha256")
        digest = hashlib.sha256(body).hexdigest()
        sidecar_parts = sidecar.decode("ascii").strip().split()
        require(sidecar_parts == [digest, name], f"canonical Stage0 {name} sidecar framing drift")
        declared = sidecar_parts[0]
        require(digest == declared, f"canonical Stage0 {name} body/sidecar drift")
        primary_digests[name] = digest
        primary_bodies[name] = body
    primary_stage_sha = primary_digests["stage0.json"]
    primary = json.loads(primary_bodies["stage0.json"])
    primary_terminal = json.loads(primary_bodies["terminal.json"])
    require(primary_terminal.get("status") == "PASS", "canonical Stage0 did not pass")
    require(primary_terminal.get("attempt_sha256") == primary_digests["attempt.json"], "canonical Stage0 attempt link drift")
    require(primary_terminal.get("stage0_sha256") == primary_stage_sha, "canonical Stage0 payload link drift")

    attempt_sha = begin_attempt(
        result_root,
        {
            "route": plan.ROUTE_NAME,
            "stage": "stage0_supplement_v1",
            "static_inputs": static,
            "primary_stage0_sha256": primary_stage_sha,
            "primary_terminal_sha256": primary_digests["terminal.json"],
            "decoder_opened": False,
            "gpu_opened": False,
            "mask_or_gate_mutated": False,
            "test_not_opened": list(splits["test"]),
        },
    )
    try:
        tasks = [(session, split, str(_source_path(repo_root, session))) for session, split in roster.items()]
        worker_count = min(8, max(1, os.cpu_count() or 1)) if max_workers is None else int(max_workers)
        require(1 <= worker_count <= 8, "supplement worker count outside [1,8]")
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            # executor.map preserves the frozen manifest order.
            rows = list(pool.map(_one_session, tasks))
        require([row["session"] for row in rows] == list(roster), "supplement merge order drift")

        train_rows = [row for row in rows if row["split"] == "train"]
        require(len(train_rows) == 27 and len(rows) == 33, "supplement roster drift")
        normalizer = fit_source_normalizer([np.asarray(row["candidate"]) for row in train_rows])
        split_vectors = _column_vectors(train_rows, "odd", "even")
        reference_vectors = _column_vectors(train_rows, "candidate", "reference")
        split_summary = aggregate_column_reliability(split_vectors, gate="split_half")
        reference_summary = aggregate_column_reliability(reference_vectors, gate="reference")
        _require_primary_match(split_summary, primary["reliability"]["split_half"], "split_half")
        _require_primary_match(reference_summary, primary["reliability"]["deployment_reference"], "deployment_reference")
        observed_mask = [bool(split_summary[name]["passed"] and reference_summary[name]["passed"]) for name in plan.PROFILE_COLUMN_NAMES]
        require(observed_mask == list(primary["decision"]["reliability_mask"]), "supplement mask cross-check failed")

        public_rows = []
        for row in rows:
            normalized = normalize_columns(np.asarray(row["candidate"]), **normalizer)
            corr = rowwise_pearson(np.asarray(row["post"])[:, :2], np.asarray(row["whole"])[:, :2])
            public_rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"candidate", "whole", "post", "odd", "even", "reference", "reference_whole"}
                }
                | {
                    "q_raw_mean": np.mean(row["candidate"], axis=0).tolist(),
                    "q_raw_std": np.std(row["candidate"], axis=0).tolist(),
                    "q_normalized_mean": np.mean(normalized, axis=0).tolist(),
                    "q_normalized_std": np.std(normalized, axis=0).tolist(),
                    "q_raw_sha256": array_sha256(np.asarray(row["candidate"])),
                    "q_normalized_sha256": array_sha256(normalized),
                    "corr_aR_aWhole": float(corr[0]),
                    "corr_cR_cWhole": float(corr[1]),
                }
            )

        era = {}
        for era_name in plan.ERA_EXPECTED_COUNTS:
            era[era_name] = {}
            for split_name in ("train", "val"):
                subset = [row for row in rows if row["era"] == era_name and row["split"] == split_name]
                item: dict[str, object] = {
                    "session_count": len(subset),
                    "q_raw_session_mean": np.mean(
                        np.stack([np.mean(row["candidate"], axis=0) for row in subset]), axis=0
                    ).tolist() if subset else None,
                    "q_raw_session_std": np.std(
                        np.stack([np.mean(row["candidate"], axis=0) for row in subset]), axis=0
                    ).tolist() if subset else None,
                }
                if split_name == "train" and subset:
                    item["split_half"] = _descriptive(_column_vectors(subset, "odd", "even"))
                    item["deployment_reference"] = _descriptive(_column_vectors(subset, "candidate", "reference"))
                else:
                    item["split_half"] = None
                    item["deployment_reference"] = None
                era[era_name][split_name] = item

        payload: dict[str, object] = {
            "route": plan.ROUTE_NAME,
            "stage": "stage0_supplement_v1",
            "attempt_sha256": attempt_sha,
            "primary_stage0_sha256": primary_stage_sha,
            "primary_terminal_sha256": primary_digests["terminal.json"],
            "mask_or_gate_mutated": False,
            "dense_velocity_scalars_consumed": 0,
            "worker_count": worker_count,
            "merge_order": list(roster),
            "raw_rate_pool_calls_per_session": 1,
            "source_train_normalizer": {
                "mean": np.asarray(normalizer["mean"]).tolist(),
                "scale": np.asarray(normalizer["scale"]).tolist(),
            },
            "reliability_cross_check": {
                "split_half": split_summary,
                "deployment_reference": reference_summary,
                "mask": observed_mask,
                "matches_primary": True,
            },
            "per_era": era,
            "per_session": public_rows,
            "pseudo_mua_order": "pool_interval_rates_by_electrode_then_refit_t4",
            "direction_shuffle": "permute_candidate_direction_indices_then_refit_same_rates",
            "row_shuffle": "route_PCG64_nonidentity_permutation",
        }
        supplement_sha = publish_immutable_json(result_root, "supplement.json", payload)
        close_terminal(
            result_root,
            attempt_sha256=attempt_sha,
            payload={
                "supplement_sha256": supplement_sha,
                "primary_stage0_sha256": primary_stage_sha,
                "primary_terminal_sha256": primary_digests["terminal.json"],
                "mask_or_gate_mutated": False,
            },
        )
        return payload
    except BaseException as error:
        close_failure(result_root, attempt_sha256=attempt_sha, error=error)
        raise
