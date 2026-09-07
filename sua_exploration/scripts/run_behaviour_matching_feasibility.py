#!/usr/bin/env python3
"""Measure cross-session behavioural nearest-neighbour matching against the within-session floor.

Frozen protocol: ``sua_exploration/docs/BEHAVIOUR_MATCHING_FEASIBILITY_PROTOCOL_20260814.md``.
Behaviour arrays only: no neural data, no model, no training, CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "streaming_calibration_exp"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sua_exploration.behaviour_matching import core, loaders

DEFAULT_OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/behaviour_matching_feasibility_20260814"
RECEIPT_NAME = "behaviour_matching_feasibility_receipt.json"
PROTOCOL_PATH = REPO_ROOT / core.PROTOCOL_DOCUMENT


COHORTS: tuple[dict[str, Any], ...] = (
    {
        "id": "co_subc_source",
        "label": "centre-out sub-C, frozen 27-session source split",
        "discover": loaders.discover_centre_out_subc,
        "load": loaders.load_centre_out_session,
        "representations": loaders.CO_REPRESENTATIONS,
        "discrete_keys": loaders.CO_DISCRETE_KEYS,
    },
    {
        "id": "co_subm_external",
        "label": "centre-out sub-M, external cohort",
        "discover": loaders.discover_centre_out_subm,
        "load": loaders.load_centre_out_session,
        "representations": loaders.CO_REPRESENTATIONS,
        "discrete_keys": loaders.CO_DISCRETE_KEYS,
    },
    {
        "id": "rt_subc",
        "label": "random-target sub-C, 15 sessions",
        "discover": loaders.discover_rt_subc,
        "load": loaders.load_rt_session,
        "representations": loaders.RT_REPRESENTATIONS,
        "discrete_keys": (),
    },
    {
        "id": "h1_heldin",
        "label": "H1 7-DoF, 13 public held-in recordings",
        "discover": loaders.discover_h1_heldin,
        "load": loaders.load_h1_session,
        "representations": loaders.H1_REPRESENTATIONS,
        "discrete_keys": (),
    },
)


def write_immutable(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, 0o444)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_channel_stats(sessions: Sequence[loaders.SessionSamples]) -> tuple[np.ndarray, np.ndarray]:
    """Pooled per-channel mean/sd over the union of every session's sample windows."""
    total = 0
    summed: np.ndarray | None = None
    squared: np.ndarray | None = None
    for session in sessions:
        core.require(session.channel_accumulator is not None, f"{session.session}: missing channel accumulator")
        count, sums, sumsq = session.channel_accumulator
        total += count
        summed = sums.copy() if summed is None else summed + sums
        squared = sumsq.copy() if squared is None else squared + sumsq
    core.require(total > 1 and summed is not None and squared is not None, "empty cohort accumulator")
    mean = summed / total
    variance = np.maximum(squared / total - np.square(mean), 0.0)
    sd = np.sqrt(variance)
    core.require(bool(np.all(sd >= core.SD_FLOOR)), f"channel sd below floor: {sd.tolist()}")
    return mean, sd


def cohort_common_sizes(sessions: Sequence[loaders.SessionSamples]) -> tuple[int, int]:
    """Protocol amendment A1: one reference/query size for the whole cohort.

    Equal reference-set sizes everywhere are what make ``cross / within`` a paired measure of
    distribution mismatch rather than of how many samples a session happens to hold.
    """
    reference_sizes = []
    query_sizes = []
    for session in sessions:
        reference_pool, query_pool = core.chronological_halves(session.n_samples)
        reference_sizes.append(int(reference_pool.size))
        query_sizes.append(int(query_pool.size))
    return min(core.M_REF, min(reference_sizes)), min(core.M_QUERY, min(query_sizes))


def build_views(
    sessions: Sequence[loaders.SessionSamples],
    *,
    cohort: str,
    representation: str,
    discrete_keys: Sequence[str],
    channel_stats: tuple[np.ndarray, np.ndarray] | None,
    reference_size: int,
    query_size: int,
) -> tuple[dict[str, core.SessionView], np.ndarray, np.ndarray]:
    groups = loaders.representation_groups(representation)
    dim = int(groups.size)
    selections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for session in sessions:
        reference_pool, query_pool = core.chronological_halves(session.n_samples)
        selections[session.session] = (
            core.subsample(reference_pool, reference_size, cohort=cohort, session=session.session, role="reference"),
            core.subsample(query_pool, query_size, cohort=cohort, session=session.session, role="query"),
        )

    if channel_stats is not None:
        channel_mean, channel_sd = channel_stats
        mean = np.asarray(channel_mean, dtype=np.float64)[groups]
        sd = np.asarray(channel_sd, dtype=np.float64)[groups]
    else:
        mean, sd = core.grouped_zscore_stats(
            [
                session.materialize(representation, np.arange(session.n_samples, dtype=np.int64))
                for session in sessions
            ],
            groups,
        )

    views: dict[str, core.SessionView] = {}
    for session in sessions:
        reference_rows, query_rows = selections[session.session]
        views[session.session] = core.SessionView(
            cohort=cohort,
            session=session.session,
            representation=representation,
            dim=dim,
            query=core.apply_zscore(session.materialize(representation, query_rows), mean, sd),
            reference=core.apply_zscore(session.materialize(representation, reference_rows), mean, sd),
            query_labels={key: session.labels[key][query_rows] for key in discrete_keys},
            reference_labels={key: session.labels[key][reference_rows] for key in discrete_keys},
            query_magnitude=session.movement_magnitude(query_rows),
        )
    return views, mean, sd


def sweep_reference_view(
    session: loaders.SessionSamples,
    *,
    cohort: str,
    representation: str,
    size: int,
    mean: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray | None:
    reference_pool, _query_pool = core.chronological_halves(session.n_samples)
    if reference_pool.size < size:
        return None
    rows = core.subsample(reference_pool, size, cohort=cohort, session=session.session, role="reference")
    return core.apply_zscore(session.materialize(representation, rows), mean, sd)


def run(*, output_dir: Path, cohort_filter: Sequence[str] | None, dry_run: bool) -> dict[str, Any]:
    started = time.time()
    results: dict[str, Any] = {}
    scaling_rows: list[dict[str, Any]] = []
    session_inventory: dict[str, Any] = {}

    for spec in COHORTS:
        cohort_id = str(spec["id"])
        if cohort_filter and cohort_id not in cohort_filter:
            continue
        load_started = time.time()
        paths = spec["discover"](REPO_ROOT)
        loader: Callable[..., loaders.SessionSamples] = spec["load"]
        sessions = [loader(path, cohort=cohort_id) for path in paths]
        core.require(len(sessions) >= 2, f"{cohort_id}: need at least two sessions")
        names = [session.session for session in sessions]
        core.require(len(set(names)) == len(names), f"{cohort_id}: duplicate session names")
        load_seconds = time.time() - load_started

        channel_stats = cohort_channel_stats(sessions) if sessions[0].channel_accumulator is not None else None
        session_inventory[cohort_id] = {
            "label": spec["label"],
            "session_count": len(sessions),
            "sessions": [
                {
                    "session": session.session,
                    "path": str(session.path),
                    "samples": session.n_samples,
                    "reference_pool": session.n_samples // 2,
                    "query_pool": session.n_samples - session.n_samples // 2,
                    "notes": session.notes,
                    "bindings": session.bindings,
                }
                for session in sessions
            ],
            "pooled_channel_mean": (
                [float(v) for v in channel_stats[0].tolist()] if channel_stats is not None else None
            ),
            "pooled_channel_sd": (
                [float(v) for v in channel_stats[1].tolist()] if channel_stats is not None else None
            ),
            "load_seconds": load_seconds,
        }
        if cohort_id == "rt_subc":
            session_inventory[cohort_id]["native_direction_degeneracy"] = loaders.rt_native_direction_degeneracy(
                paths
            )

        reference_size, query_size = cohort_common_sizes(sessions)
        core.require(
            reference_size >= core.MIN_POOL and query_size >= core.MIN_POOL,
            f"{cohort_id}: cohort-common sizes {reference_size}/{query_size} below the size floor",
        )
        session_inventory[cohort_id]["cohort_common_reference_size"] = reference_size
        session_inventory[cohort_id]["cohort_common_query_size"] = query_size

        cohort_result: dict[str, Any] = {"label": spec["label"], "representations": {}}
        for representation in spec["representations"]:
            views, zscore_mean, zscore_sd = build_views(
                sessions,
                cohort=cohort_id,
                representation=representation,
                discrete_keys=spec["discrete_keys"],
                channel_stats=channel_stats,
                reference_size=reference_size,
                query_size=query_size,
            )
            usable = {name: view for name, view in views.items() if view.usable()}
            skipped = sorted(set(views) - set(usable))
            core.require(len(usable) >= 2, f"{cohort_id}/{representation}: fewer than two usable sessions")

            floors = {name: core.within_session_floor(view) for name, view in usable.items()}

            # Protocol amendment A2: post-hoc movement-magnitude strata. Both centre-out and RT
            # sample sets are dominated by near-stationary moments, so the unstratified residual
            # is largely the residual of matching "barely moving" to "barely moving".
            magnitudes = np.concatenate([view.query_magnitude for view in usable.values()])
            cut_points = [float(v) for v in np.percentile(magnitudes, [25, 50, 75])]
            n_strata = len(cut_points) + 1
            strata = {
                name: core.stratum_labels(view.query_magnitude, cut_points)
                for name, view in usable.items()
            }
            floor_strata = {
                name: core.stratified_medians(
                    view.query, strata[name], view.reference, dim=view.dim, n_strata=n_strata
                )
                for name, view in usable.items()
            }

            pair_rows: list[dict[str, Any]] = []
            for query_name, query_view in usable.items():
                for reference_name, reference_view in usable.items():
                    if query_name == reference_name:
                        continue
                    row = core.evaluate_pair(
                        query_view,
                        reference_view,
                        within_floor_median=floors[query_name]["floor"]["median"],
                        discrete_keys=spec["discrete_keys"],
                    )
                    cross_strata = core.stratified_medians(
                        query_view.query,
                        strata[query_name],
                        reference_view.reference,
                        dim=query_view.dim,
                        n_strata=n_strata,
                    )
                    row["movement_strata"] = [
                        {
                            "stratum": entry["stratum"],
                            "count": entry["count"],
                            "cross_median_per_dim": entry["median_per_dim"],
                            "within_median_per_dim": floor_strata[query_name][index]["median_per_dim"],
                            "ratio": (
                                float(entry["median"] / floor_strata[query_name][index]["median"])
                                if entry["median"] is not None
                                and floor_strata[query_name][index]["median"]
                                else None
                            ),
                        }
                        for index, entry in enumerate(cross_strata)
                    ]
                    pair_rows.append(row)

            ratios = [row["ratio_cross_over_within"] for row in pair_rows]
            cross_medians = [row["cross"]["median"] for row in pair_rows]
            cross_per_dim = [row["cross"]["median_per_dim"] for row in pair_rows]
            centroid_frac = [row["cross_over_centroid"] for row in pair_rows]
            ratio_spread = core.spread_over_pairs(ratios)
            per_dim_spread = core.spread_over_pairs(cross_per_dim)
            ratio_verdict = core.classify_ratio(ratio_spread["median"], ratio_spread["p90"])
            absolute_verdict = core.classify_absolute(per_dim_spread["median"])

            pooled = np.concatenate(
                [np.concatenate([view.reference, view.query], axis=0) for view in usable.values()], axis=0
            )
            dimensionality = core.dimensionality_report(pooled)

            representation_result: dict[str, Any] = {
                "dim": loaders.representation_dim(representation),
                "sessions_used": sorted(usable),
                "sessions_skipped_below_size_floor": skipped,
                "ordered_pairs": len(pair_rows),
                "primary_reference_size": reference_size,
                "primary_query_size": query_size,
                "requested_reference_size": core.M_REF,
                "requested_query_size": core.M_QUERY,
                "within_session_floor": {
                    "per_session_median": {name: floors[name]["floor"]["median"] for name in sorted(floors)},
                    "per_session_median_per_dim": {
                        name: floors[name]["floor"]["median_per_dim"] for name in sorted(floors)
                    },
                    "achieved_reference_size": {name: floors[name]["reference_size"] for name in sorted(floors)},
                    "achieved_query_size": {name: floors[name]["query_size"] for name in sorted(floors)},
                    "spread_over_sessions": core.spread_over_pairs(
                        [floors[name]["floor"]["median"] for name in sorted(floors)]
                    ),
                    "spread_over_sessions_per_dim": core.spread_over_pairs(
                        [floors[name]["floor"]["median_per_dim"] for name in sorted(floors)]
                    ),
                },
                "cross_session": {
                    "median_l2_spread_over_pairs": core.spread_over_pairs(cross_medians),
                    "median_per_dim_spread_over_pairs": per_dim_spread,
                    "ratio_spread_over_pairs": ratio_spread,
                    "cross_over_centroid_spread_over_pairs": core.spread_over_pairs(centroid_frac),
                },
                "verdict": core.combined_verdict(ratio_verdict, absolute_verdict),
                "dimensionality": dimensionality,
                "movement_strata": {
                    "note": (
                        "post-hoc protocol amendment A2; queries are split by their own movement "
                        "magnitude against cohort-pooled quartiles, references are never stratified"
                    ),
                    "magnitude_quartile_cut_points": cut_points,
                    "magnitude_percentiles": {
                        str(level): float(np.percentile(magnitudes, level))
                        for level in (1, 5, 10, 25, 50, 75, 90, 95, 99)
                    },
                    "per_stratum": [
                        {
                            "stratum": index,
                            "cross_median_per_dim": core.spread_over_pairs(
                                [
                                    row["movement_strata"][index]["cross_median_per_dim"]
                                    for row in pair_rows
                                    if row["movement_strata"][index]["cross_median_per_dim"] is not None
                                ]
                            ),
                            "ratio": core.spread_over_pairs(
                                [
                                    row["movement_strata"][index]["ratio"]
                                    for row in pair_rows
                                    if row["movement_strata"][index]["ratio"] is not None
                                ]
                            ),
                            "within_median_per_dim": core.spread_over_pairs(
                                [
                                    row["movement_strata"][index]["within_median_per_dim"]
                                    for row in pair_rows
                                    if row["movement_strata"][index]["within_median_per_dim"] is not None
                                ]
                            ),
                        }
                        for index in range(n_strata)
                    ],
                },
                "per_pair": pair_rows,
            }

            for key in spec["discrete_keys"]:
                match_rates = [row[f"discrete_{key}"]["discrete_match_rate"] for row in pair_rows]
                restricted = [
                    row[f"discrete_{key}"]["class_restricted"]["median_per_dim"]
                    for row in pair_rows
                    if "class_restricted" in row[f"discrete_{key}"]
                ]
                entry: dict[str, Any] = {
                    "classes": int(
                        np.unique(np.concatenate([view.query_labels[key] for view in usable.values()])).size
                    ),
                    "label_residual": 0.0,
                    "label_residual_note": (
                        "matching on the discrete key alone has exactly zero residual whenever the class "
                        "exists in the reference session; the informative quantity is the continuous "
                        "residual among same-class reference samples, below"
                    ),
                    "match_rate_spread_over_pairs": core.spread_over_pairs(match_rates),
                }
                if restricted:
                    entry["class_restricted_median_per_dim_spread_over_pairs"] = core.spread_over_pairs(restricted)
                    entry["class_restriction_gain_vs_unrestricted"] = float(
                        per_dim_spread["median"] / float(np.median(restricted))
                    )
                representation_result[f"discrete_{key}"] = entry

            cohort_result["representations"][representation] = representation_result

            # protocol section 10: reference-size sweep on a fixed subset of ordered pairs
            by_name = {session.session: session for session in sessions}
            ordered_pairs = [
                (query_name, reference_name)
                for query_name in sorted(usable)
                for reference_name in sorted(usable)
                if query_name != reference_name
            ]
            permutation = np.random.default_rng(
                core.derived_seed(cohort_id, representation, "sweep_pairs")
            ).permutation(len(ordered_pairs))
            selected = [ordered_pairs[int(index)] for index in permutation[: core.MAX_SWEEP_PAIRS]]
            # Only sweep sizes every session's reference pool can supply, so no sweep point is a
            # median over a different, larger-session-biased subset of pairs.
            max_sweep_size = min(
                int(core.chronological_halves(by_name[name].n_samples)[0].size) for name in usable
            )
            sweep: list[dict[str, Any]] = []
            for size in core.SWEEP_REF_SIZES:
                if size > max_sweep_size:
                    continue
                values: list[float] = []
                for query_name, reference_name in selected:
                    reference = sweep_reference_view(
                        by_name[reference_name],
                        cohort=cohort_id,
                        representation=representation,
                        size=size,
                        mean=zscore_mean,
                        sd=zscore_sd,
                    )
                    if reference is None:
                        continue
                    distances = core.nn_distances(usable[query_name].query, reference)
                    values.append(float(np.median(distances)) / np.sqrt(usable[query_name].dim))
                if len(values) < max(2, len(selected) // 2):
                    continue
                sweep.append(
                    {
                        "reference_size": int(size),
                        "pairs": len(values),
                        "median_nn_per_dim": float(np.median(values)),
                        "p25_nn_per_dim": float(np.percentile(values, 25)),
                        "p75_nn_per_dim": float(np.percentile(values, 75)),
                    }
                )
            fit = (
                core.fit_log_slope(
                    [row["reference_size"] for row in sweep], [row["median_nn_per_dim"] for row in sweep]
                )
                if len(sweep) >= 2
                else {"points": len(sweep)}
            )
            common = next(
                (row for row in sweep if row["reference_size"] == core.COMMON_REF_SIZE), None
            )
            representation_result["reference_size_sweep"] = {
                "pairs_sampled": len(selected),
                "max_size_supported_by_every_session": max_sweep_size,
                "points": sweep,
                "fit": fit,
                "common_reference_size": core.COMMON_REF_SIZE,
                "median_nn_per_dim_at_common_size": (common["median_nn_per_dim"] if common else None),
            }
            scaling_rows.append(
                {
                    "cohort": cohort_id,
                    "representation": representation,
                    "nominal_dim": loaders.representation_dim(representation),
                    "participation_ratio": dimensionality["participation_ratio"],
                    "fit": fit,
                }
            )

        results[cohort_id] = cohort_result

    scaling = core.classify_scaling(scaling_rows)
    elapsed = time.time() - started

    receipt: dict[str, Any] = {
        "schema": core.SCHEMA,
        "status": "COMPLETED_CPU_ONLY",
        "date": "2026-08-14",
        "protocol": {
            "document": core.PROTOCOL_DOCUMENT,
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "scope": {
            "behaviour_arrays_only": True,
            "neural_data_read": False,
            "model_or_training": False,
            "cuda_used": False,
            "global_seed": core.GLOBAL_SEED,
            "requested_reference_size": core.M_REF,
            "requested_query_size": core.M_QUERY,
            "amendment_a1_cohort_common_sizes": (
                "reference/query sizes are clamped to min over the cohort's sessions so that "
                "cross-session and within-session residuals always use equal-sized reference "
                "sets; binds only on h1_heldin"
            ),
            "min_pool": core.MIN_POOL,
            "sweep_reference_sizes": list(core.SWEEP_REF_SIZES),
            "max_sweep_pairs": core.MAX_SWEEP_PAIRS,
            "common_reference_size": core.COMMON_REF_SIZE,
        },
        "decision_thresholds": {
            "ratio_tight_median_max": core.RATIO_TIGHT_MEDIAN_MAX,
            "ratio_tight_p90_max": core.RATIO_TIGHT_P90_MAX,
            "ratio_marginal_median_max": core.RATIO_MARGINAL_MEDIAN_MAX,
            "absolute_tight_max": core.ABS_TIGHT_MAX,
            "absolute_marginal_max": core.ABS_MARGINAL_MAX,
            "scaling_tolerance_factor": core.SCALING_TOLERANCE_FACTOR,
        },
        "session_inventory": session_inventory,
        "results": results,
        "minus_one_over_d_scaling": scaling,
        "headline": {
            f"{cohort_id}/{representation}": {
                "dim": block["dim"],
                "participation_ratio": block["dimensionality"]["participation_ratio"],
                "within_session_floor_median_per_dim": block["within_session_floor"][
                    "spread_over_sessions_per_dim"
                ]["median"],
                "cross_session_median_per_dim": block["cross_session"][
                    "median_per_dim_spread_over_pairs"
                ]["median"],
                "ratio_median": block["cross_session"]["ratio_spread_over_pairs"]["median"],
                "ratio_p90": block["cross_session"]["ratio_spread_over_pairs"]["p90"],
                "ratio_min": block["cross_session"]["ratio_spread_over_pairs"]["min"],
                "ratio_max": block["cross_session"]["ratio_spread_over_pairs"]["max"],
                "verdict": block["verdict"],
            }
            for cohort_id, cohort_block in results.items()
            for representation, block in cohort_block["representations"].items()
        },
        "implementation_sha256": {
            "sua_exploration/behaviour_matching/core.py": sha256_file(Path(core.__file__)),
            "sua_exploration/behaviour_matching/loaders.py": sha256_file(Path(loaders.__file__)),
            "sua_exploration/scripts/run_behaviour_matching_feasibility.py": sha256_file(Path(__file__)),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "nice": os.nice(0),
            "elapsed_seconds": elapsed,
        },
    }

    if dry_run:
        return {"status": "DRY_RUN", "elapsed_seconds": elapsed, "headline": receipt["headline"]}

    receipt_path = output_dir / RECEIPT_NAME
    core.require(not receipt_path.exists(), f"refusing to overwrite {receipt_path}")
    receipt_sha = write_immutable(receipt_path, core.canonical_json_bytes(receipt))
    return {
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "elapsed_seconds": elapsed,
        "scaling_verdict": scaling["verdict"],
        "headline": receipt["headline"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cohort", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run(
        output_dir=args.output_dir.resolve(),
        cohort_filter=args.cohort,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
