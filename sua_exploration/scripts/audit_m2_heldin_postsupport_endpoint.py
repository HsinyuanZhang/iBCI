#!/usr/bin/env python3
"""Read-only audit: M2 held-in-calib post-support endpoint feasibility.

Measures trial structure, minival prefix contamination, LOSO session isolation,
directional trial coverage, and the held-out M33 versus held-in post-support
comparison that motivates a precision-calibration replay.  No training, scoring,
optimizer step, checkpoint selection, or hidden EvalAI access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pynwb import NWBHDF5IO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "SPINT-main" / "data"
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m2_heldin_postsupport_endpoint_v1"
M33_AGGREGATE = ROOT / "sua_exploration/results/m2_m33_disjoint_replay_correction_v1/aggregate_heldout.json"
M33_AGGREGATE_SHA256 = "8d56e60825cc3c017b17764146118a0e6f04a33f273f0c6c9191e14de6da70fb"
UNCERTAINTY_AGGREGATE = ROOT / "sua_exploration/results/m2_uncertainty_identifiability_v1/analysis_v1/aggregate_v2.json"
NATIVE_M2_AGGREGATE = ROOT / "sua_exploration/results/native_mua_t4_v1/aggregate_m2.json"
NATIVE_M2_AGGREGATE_SHA256 = "c6eb1727456040b02f333260f67f47f134cade9e0f556ca912169b0da3fc8613"

SUPPORT_TRIALS = 33
WINDOW_SIZE = 50
SUBJECT = "sub-MonkeyN"
DANDISET = "000953"
M33_HELDOUT_MANIFEST = (
    ROOT
    / "streaming_calibration_exp/outputs/streaming_calibration/"
    "m2_m33_disjoint_replay_correction_v1_f0_m2_f1_s42_20260801_115446/split_manifest.json"
)

HASH_SIZE_LIMIT_BYTES = 512 * 1024 * 1024
Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143
MDE_Z_SUM = Z_975 + Z_80

CANDIDATE_BRANCH_SEAL = {
    "binding": True,
    "sealed_branches": [
        "B3TStream+T4",
        "K4",
        "KS4",
        "SSC-T4",
        "B3TS",
        "T4-CFILM",
        "ordinary-T4-versus-clean-SPINT comparison",
    ],
    "forbidden_without_new_protocol": [
        "re-measure",
        "re-run",
        "re-aggregate",
        "re-interpret",
        "compute counterfactual numbers on the new endpoint",
    ],
    "lifting_requires": (
        "a separately pre-declared protocol frozen before any sealed-branch number "
        "is observed on the held-in post-support endpoint"
    ),
    "framing": (
        "The held-in post-support endpoint is a more precise measurement instrument "
        "for calibrating minimum detectable effect of F0 and T4 baselines.  It is not "
        "a second chance at a failed gate or a reopening of closed candidate-branch "
        "verdicts."
    ),
    "danger_of_violation": (
        "Observing a candidate-branch delta on this larger endpoint after a negative "
        "gate invites post-hoc relitigation and multiplies the false-recovery risk "
        "from endpoint shopping."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    record: dict[str, Any] = {"file": path.name, "size_bytes": int(size)}
    if size <= HASH_SIZE_LIMIT_BYTES:
        record["sha256"] = sha256(path)
    else:
        record["sha256"] = None
        record["sha256_skipped_reason"] = "file exceeds the audit hashing limit"
    return record


def trial_table(path: Path) -> dict[str, Any]:
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        return {
            "session_start_time": str(nwb.session_start_time),
            "n_trials": int(len(trials)),
            "start_time": np.asarray(trials["start_time"], dtype=float),
            "stop_time": np.asarray(trials["stop_time"], dtype=float),
            "tgt_loc": trials["tgt_loc"],
        }


def session_of(path: Path) -> str:
    return path.name.split("_ses-")[1].split("_behavior")[0]


def files_for(split: str) -> list[Path]:
    directory = DATA / DANDISET / f"{SUBJECT}-{split}"
    return sorted(directory.glob("*.nwb"))


def m2_directional_counts(tgt_loc_series: Any, start_trial: int, end_trial: int | None) -> dict[str, int]:
    """Count raw and directional trials in a trial-index window.

    M2 centre/rest targets at (0.5, 0.5) are unlabeled and excluded from the
  T4 cosine fit, matching ``calibration_target_angles``.
    """
    centre = np.asarray([0.5, 0.5], dtype=np.float64)
    n_total = len(tgt_loc_series)
    end = end_trial if end_trial is not None else n_total
    raw = int(max(end - start_trial, 0))
    directional = 0
    for index in range(start_trial, min(end, n_total)):
        point = np.asarray(tgt_loc_series.iloc[index], dtype=np.float64).reshape(-1)
        if point.size != 2 or not np.all(np.isfinite(point)):
            continue
        delta = point - centre
        if np.linalg.norm(delta) <= 1.0e-8:
            continue
        directional += 1
    return {"raw_query_trials": raw, "directional_query_trials": directional}


def audit_heldin_minival_prefix() -> dict[str, Any]:
    support = SUPPORT_TRIALS
    minival = {session_of(p): p for p in files_for("held-in-minival")}
    sessions: dict[str, Any] = {}
    for calib_path in files_for("held-in-calib"):
        session = session_of(calib_path)
        minival_path = minival.get(session)
        if minival_path is None:
            sessions[session] = {"minival": "missing"}
            continue
        calib = trial_table(calib_path)
        val = trial_table(minival_path)
        n = val["n_trials"]
        exact_prefix = bool(
            n <= calib["n_trials"]
            and np.array_equal(val["start_time"], calib["start_time"][:n])
            and np.array_equal(val["stop_time"], calib["stop_time"][:n])
        )
        sessions[session] = {
            "calib": {**file_provenance(calib_path), "n_trials": calib["n_trials"]},
            "minival": {**file_provenance(minival_path), "n_trials": n},
            "same_session_start_time": calib["session_start_time"] == val["session_start_time"],
            "minival_is_exact_prefix_of_calib": exact_prefix,
            "support_prefix_trials": support,
            "scored_trials_inside_support_prefix": bool(exact_prefix and n <= support),
            "query_trials_after_support": int(max(n - support, 0)),
            "calib_support_boundary_stop_time": (
                float(calib["stop_time"][support - 1]) if calib["n_trials"] >= support else None
            ),
            "minival_last_stop_time": float(val["stop_time"][-1]),
        }
    contaminated = [
        k for k, v in sessions.items() if v.get("scored_trials_inside_support_prefix")
    ]
    return {
        "support_prefix_trials": support,
        "sessions": sessions,
        "sessions_scoring_entirely_inside_support": contaminated,
        "contaminated_session_count": len(contaminated),
        "total_session_count": len(sessions),
    }


def audit_heldin_postsupport_endpoint() -> dict[str, Any]:
    support = SUPPORT_TRIALS
    sessions: dict[str, Any] = {}
    for calib_path in files_for("held-in-calib"):
        session = session_of(calib_path)
        table = trial_table(calib_path)
        n = table["n_trials"]
        post_support = int(max(n - support, 0))
        directional = m2_directional_counts(table["tgt_loc"], support, None)
        sessions[session] = {
            **file_provenance(calib_path),
            "total_calib_trials": n,
            "frozen_support_budget_trials": support,
            "post_support_trials": post_support,
            "post_support_directional_trials": directional["directional_query_trials"],
            "post_support_centre_or_rest_trials": int(
                post_support - directional["directional_query_trials"]
            ),
            "has_nonempty_post_support": bool(post_support > 0),
            "support_boundary_stop_time": (
                float(table["stop_time"][support - 1]) if n >= support else None
            ),
            "first_post_support_trial_start_time": (
                float(table["start_time"][support]) if n > support else None
            ),
            "session_end_stop_time": float(table["stop_time"][-1]) if n else None,
        }
    post_support_sessions = [k for k, v in sessions.items() if v["has_nonempty_post_support"]]
    totals = {
        "raw_post_support_trials": sum(sessions[s]["post_support_trials"] for s in post_support_sessions),
        "directional_post_support_trials": sum(
            sessions[s]["post_support_directional_trials"] for s in post_support_sessions
        ),
        "session_count": len(post_support_sessions),
    }
    return {
        "frozen_support_budget_trials": support,
        "sessions": sessions,
        "sessions_with_nonempty_post_support": post_support_sessions,
        "total_session_count": len(sessions),
        "totals": totals,
    }


def audit_heldout_m33_endpoint() -> dict[str, Any]:
    support = SUPPORT_TRIALS
    sessions: dict[str, Any] = {}
    for path in files_for("held-out-calib"):
        table = trial_table(path)
        n = table["n_trials"]
        post_support = int(max(n - support, 0))
        directional = m2_directional_counts(table["tgt_loc"], support, None)
        sessions[session_of(path)] = {
            **file_provenance(path),
            "n_trials": n,
            "support_prefix_trials": support,
            "query_trials_after_support": post_support,
            "directional_query_trials": directional["directional_query_trials"],
            "has_disjoint_query": bool(n > support),
        }
    eligible = [k for k, v in sessions.items() if v["has_disjoint_query"]]
    totals = {
        "raw_query_trials": sum(sessions[s]["query_trials_after_support"] for s in eligible),
        "directional_query_trials": sum(
            sessions[s]["directional_query_trials"] for s in eligible
        ),
        "eligible_session_count": len(eligible),
    }
    return {
        "support_prefix_trials": support,
        "sessions": sessions,
        "eligible_sessions": eligible,
        "eligible_session_count": len(eligible),
        "total_session_count": len(sessions),
        "totals": totals,
    }


def heldout_m33_window_audit() -> dict[str, Any]:
    manifest = json.loads(M33_HELDOUT_MANIFEST.read_text(encoding="utf-8"))
    audit = manifest["heldout_query_window_audit"]
    eligible = {
        session: record
        for session, record in audit.items()
        if int(record.get("query_trials", 0)) > 0
    }
    return {
        "source_manifest": str(M33_HELDOUT_MANIFEST.relative_to(ROOT)),
        "per_session": eligible,
        "totals": {
            "query_trials": sum(int(v["query_trials"]) for v in eligible.values()),
            "eligible_windows": sum(int(v["eligible_windows"]) for v in eligible.values()),
            "session_count": len(eligible),
        },
    }


def postsupport_window_audit_one_session(calib_path: Path) -> dict[str, Any]:
    """CPU-only FalconDataset window enumeration for trials [33, end)."""
    import sys
    from unittest.mock import MagicMock

    sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))
    from falcon_challenge.config import FalconTask  # noqa: E402
    from src.data.falcon_datamodule import FalconDataModule, FalconDataset  # noqa: E402

    dm = FalconDataModule(
        task="m2",
        data_dir=str(DATA / DANDISET) + "/",
        window_size=WINDOW_SIZE,
        calibration_n_trials=SUPPORT_TRIALS,
        num_workers=0,
    )
    trainer = MagicMock()
    trainer.world_size = 1
    dm.trainer = trainer
    session_name = f"ses-{session_of(calib_path)}"
    session_data = dm.prepare_session_data(calib_path, FalconTask.m2, use_intertrials=True)
    sessions_dict = {session_name: session_data}
    calib_sessions_dict = {session_name: session_data}
    dataset = FalconDataset(
        sessions_dict=sessions_dict,
        calib_sessions_dict=calib_sessions_dict,
        window_size=WINDOW_SIZE,
        split="test",
        calibration_n_trials=SUPPORT_TRIALS,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        side_feature_group="none",
        query_start_trial=SUPPORT_TRIALS,
        query_end_trial=None,
        allow_empty_query_sessions=False,
    )
    audit = dataset.query_window_audit[session_name]
    return {
        "session": session_name,
        "scored_windows": int(len(dataset)),
        "query_window_audit": audit,
    }


def postsupport_window_audits() -> dict[str, Any]:
    per_session: dict[str, Any] = {}
    for calib_path in files_for("held-in-calib"):
        record = postsupport_window_audit_one_session(calib_path)
        per_session[record["session"]] = record
    totals = {
        "scored_windows": sum(int(v["scored_windows"]) for v in per_session.values()),
        "query_trials": sum(
            int(v["query_window_audit"]["query_trials"]) for v in per_session.values()
        ),
        "session_count": len(per_session),
    }
    return {"per_session": per_session, "totals": totals}


def datamodule_loso_isolation() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))
    from src.data.falcon_datamodule import FalconDataModule

    def build(loso_fold: int) -> FalconDataModule:
        return FalconDataModule(
            task="m2",
            data_dir=str(DATA / DANDISET) + "/",
            window_size=WINDOW_SIZE,
            calibration_n_trials=SUPPORT_TRIALS,
            random_calibration=False,
            smooth_calibration=False,
            max_trial_length=1024,
            use_intertrials=True,
            use_calib_intertrials=False,
            trial_feature_type="raw",
            interpolate_trials=True,
            interpolate_trials_kind="cubic",
            pad_value=-1.0,
            validation_protocol="loso",
            include_heldout_in_fit=False,
            include_heldout_in_test=False,
            loso_fold=loso_fold,
            num_workers=0,
            side_feature_group="none",
            heldin_query_start_trial=0,
        )

    heldin_sessions = sorted(session_of(p) for p in files_for("held-in-calib"))
    evidence: dict[str, Any] = {}
    for fold in range(len(heldin_sessions)):
        dm = build(fold)
        dm.setup(stage="fit")
        left_out = list(dm.val_heldin_session_names)
        train_sessions = list(dm.train_session_names)
        evidence[f"fold{fold}"] = {
            "train_session_names": train_sessions,
            "val_heldin_session_names": left_out,
            "left_out_session": left_out[0] if len(left_out) == 1 else left_out,
            "left_out_session_absent_from_train": (
                len(left_out) == 1 and left_out[0] not in train_sessions
            ),
        }
    return evidence


def datamodule_guard_probe() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))
    from src.data.falcon_datamodule import FalconDataModule

    try:
        dm = FalconDataModule(
            task="m2",
            data_dir=str(DATA / DANDISET) + "/",
            window_size=WINDOW_SIZE,
            calibration_n_trials=SUPPORT_TRIALS,
            random_calibration=False,
            smooth_calibration=False,
            max_trial_length=1024,
            use_intertrials=True,
            use_calib_intertrials=False,
            trial_feature_type="raw",
            interpolate_trials=True,
            interpolate_trials_kind="cubic",
            pad_value=-1.0,
            validation_protocol="loso",
            include_heldout_in_fit=False,
            include_heldout_in_test=False,
            loso_fold=0,
            num_workers=0,
            side_feature_group="none",
            heldin_query_start_trial=SUPPORT_TRIALS,
        )
        dm.setup(stage="fit")
        return {"raised": False, "val_heldin_windows": len(dm.val_heldin_dataset)}
    except Exception as exc:
        return {
            "raised": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


def implementation_gap() -> dict[str, Any]:
    return {
        "constructible_endpoint_not_yet_reachable": True,
        "current_guard_location": "falcon_datamodule.py setup() ~lines 934-954",
        "current_guard_admits_only": {
            "task": "m1",
            "validation_protocol": "loso",
            "calibration_n_trials": 10,
            "random_calibration": False,
            "include_heldout_in_fit": False,
            "note": (
                "M1 replay protocol uses heldin_query_start_trial=10 with calibration_n_trials=10; "
                "the guard does not explicitly test heldin_query_start_trial but the pairing is frozen "
                "in the M1 disjoint replay and clean-selection protocols."
            ),
        },
        "required_widening_for_m2": {
            "task": "m2",
            "validation_protocol": "loso",
            "calibration_n_trials": 33,
            "heldin_query_start_trial": 33,
            "random_calibration": False,
            "include_heldout_in_fit": False,
        },
        "proposed_guard_clause": (
            "allowed_m1 = (self.hparams.task == 'm1' and self.hparams.validation_protocol == 'loso' "
            "and int(self.hparams.calibration_n_trials) == 10 "
            "and int(self.hparams.heldin_query_start_trial) == 10 "
            "and not self.hparams.random_calibration "
            "and not self.hparams.include_heldout_in_fit); "
            "allowed_m2 = (self.hparams.task == 'm2' and self.hparams.validation_protocol == 'loso' "
            "and int(self.hparams.calibration_n_trials) == 33 "
            "and int(self.hparams.heldin_query_start_trial) == 33 "
            "and not self.hparams.random_calibration "
            "and not self.hparams.include_heldout_in_fit); "
            "if not (allowed_m1 or allowed_m2): raise ValueError(...)"
        ),
        "reason": (
            "val_heldin_dataset loads held-in-minival (2 trials) when heldin_query_start_trial=0. "
            "Scoring held-in-calib trials strictly after the frozen first-33 support requires "
            "heldin_query_start_trial=33 and val_query_sessions sourced from train_calib_heldin_sessions."
        ),
        "do_not_modify_in_this_audit": True,
    }


def finite_sample_sd(values: list[float]) -> float | None:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) >= 2 else None


def mde_80(sd: float, n_clusters: int) -> float:
    return MDE_Z_SUM * sd / math.sqrt(n_clusters)


def power_analysis(
    heldin: dict[str, Any],
    heldout: dict[str, Any],
    heldin_windows: dict[str, Any],
    heldout_windows: dict[str, Any],
) -> dict[str, Any]:
    sigma_seed = 0.038835
    sigma_session_six = 0.083517
    sigma_session_m33_four = 0.028843432642735475
    n_heldout_eligible = heldout["eligible_session_count"]
    n_heldin = heldin["total_session_count"]

    m_old = heldout_windows["totals"]["eligible_windows"] / max(
        heldout_windows["totals"]["session_count"], 1
    )
    m_new = heldin_windows["totals"]["scored_windows"] / max(
        heldin_windows["totals"]["session_count"], 1
    )
    window_ratio = m_new / max(m_old, 1.0)
    within_scale = 1.0 / math.sqrt(window_ratio)

    current_mde = {
        "heldout_m33_four_session_endpoint": {
            "n_session_clusters": n_heldout_eligible,
            "mde_using_m33_cluster_sd": mde_80(sigma_session_m33_four, n_heldout_eligible),
            "mde_using_observed_seed_sd": mde_80(sigma_seed, n_heldout_eligible),
            "observed_cluster_sd_source": "m2_m33_disjoint_replay_correction_v1 aggregate T4-F0",
        },
        "heldout_six_session_reference": {
            "n_session_clusters": 6,
            "mde_using_session_cluster_sd": mde_80(sigma_session_six, 6),
            "mde_using_seed_sd": mde_80(sigma_seed, 6),
            "observed_sd_source": "m2_uncertainty_identifiability_v1 analysis_v1 aggregate_v2",
        },
    }

    projected_between_only = mde_80(sigma_session_six, n_heldin)
    projected_within_only = mde_80(sigma_seed * within_scale, n_heldin)
    projected_combined_upper = mde_80(
        math.sqrt(sigma_session_six**2 + (sigma_seed * within_scale) ** 2), n_heldin
    )
    projected_combined_lower = mde_80(
        max(sigma_session_six, sigma_seed * within_scale) / math.sqrt(2), n_heldin
    )

    return {
        "deployment_sesoi_r2": 0.03,
        "decomposition": {
            "within_session_component": (
                "Paired-delta uncertainty from finite query windows per scored session scales "
                "approximately as 1/sqrt(m) where m is eligible_windows per session."
            ),
            "between_session_component": (
                "Equal-weight session averaging scales between-session spread as 1/sqrt(n_sessions)."
            ),
            "new_endpoint_attacks": (
                "More post-support trials/windows per session reduces the within-session term directly; "
                "seven held-in sessions versus four eligible held-out sessions reduces the between-session term."
            ),
        },
        "measured_window_counts": {
            "heldout_m33_eligible": heldout_windows["totals"],
            "heldin_post_support": {
                "scored_windows": heldin_windows["totals"]["scored_windows"],
                "query_trials": heldin_windows["totals"]["query_trials"],
                "session_count": heldin_windows["totals"]["session_count"],
                "mean_windows_per_session": m_new,
            },
            "mean_windows_per_session_ratio_new_over_old": window_ratio,
            "within_session_noise_scale_factor_1_over_sqrt_m": within_scale,
        },
        "observed_dispersion_inputs": {
            "within_fold_seed_sd": sigma_seed,
            "six_session_cluster_sd": sigma_session_six,
            "four_session_m33_cluster_sd": sigma_session_m33_four,
            "source_note": (
                "Dispersion inputs come from existing M2 artifacts on other endpoints; "
                "they are not measured on the held-in post-support endpoint without a replay."
            ),
        },
        "current_endpoint_mde": current_mde,
        "projected_heldin_postsupport_mde_range": {
            "n_session_clusters": n_heldin,
            "assumptions": [
                "Between-session SD remains at the observed six-session cluster SD (0.083517) until Stage B measures seven held-in folds.",
                "Within-session SD scales with window count via 1/sqrt(m) using measured FalconDataset window ratios.",
                "Seed SD (0.038835) proxies the within-session measurement component from the incomplete historical factorial.",
                "True paired-delta variance decomposition is not identifiable without replay; range spans component-dominant bounds.",
            ],
            "lower_bound_r2": projected_combined_lower,
            "upper_bound_r2": projected_combined_upper,
            "between_session_only_r2": projected_between_only,
            "within_session_scaled_only_r2": projected_within_only,
            "cannot_know_without_replay": [
                "actual paired-delta SD on the held-in post-support endpoint",
                "whether between-session SD transfers from held-out to held-in domains",
                "independent per-window noise correlation structure",
            ],
        },
        "frozen_checkpoint_replay_limit": {
            "cells": ["fold1_seed42", "fold1_seed43", "fold2_seed42"],
            "distinct_left_out_sessions": 2,
            "can_establish": "within-session precision on two left-out held-in sessions",
            "cannot_establish": "seven-session between-session dispersion",
            "stage_b_training_runs_required": 14,
            "stage_b_training_runs_note": "7 LOSO folds × 2 arms (F0 and T4)",
        },
        "stage_b_wall_clock_estimate": {
            "reference_runs": [
                "native_mua_t4_v1_f0_m2_f1_s42_20260729_140518",
                "native_mua_t4_v1_t4_m2_f1_s42_20260729_142047",
            ],
            "reference_log_start_times_hkt": ["2026-07-29 14:05:18", "2026-07-29 14:20:47"],
            "reference_next_log_start_hkt": ["2026-07-29 14:51:54", "2026-07-29 15:23:34"],
            "estimated_minutes_per_run": 15.5,
            "estimated_hours_14_runs": round(14 * 15.5 / 60, 1),
            "method": (
                "Adjacent native_mua_t4_v1 M2 training log directory timestamps on 2026-07-29; "
                "F0 and T4 arms averaged."
            ),
        },
    }


def endpoint_comparison(
    heldin: dict[str, Any],
    heldout: dict[str, Any],
    heldin_windows: dict[str, Any],
    heldout_windows: dict[str, Any],
) -> dict[str, Any]:
    heldin_scored = heldin_windows["totals"]["scored_windows"]
    heldout_scored = heldout_windows["totals"]["eligible_windows"]
    ratios: dict[str, Any] = {
        "session_count_heldin_over_heldout": heldin["totals"]["session_count"] / max(
            heldout["eligible_session_count"], 1
        ),
        "raw_trials_heldin_over_heldout": heldin["totals"]["raw_post_support_trials"] / max(
            heldout["totals"]["raw_query_trials"], 1
        ),
    }
    if heldin_scored is not None and heldout_scored is not None:
        ratios["scored_windows_heldin_over_heldout"] = heldin_scored / max(heldout_scored, 1)
    return {
        "central_motivating_fact": (
            "The corrected M33 held-out endpoint that produced authoritative M2 gate numbers "
            "rests on 36 raw query trials across 4 eligible sessions with 2069 scored windows, "
            "while the unused held-in post-support endpoint offers 1692 raw trials across 7 sessions "
            "with far larger per-session window counts."
        ),
        "heldout_m33_eligible": {
            "session_count": heldout["eligible_session_count"],
            "raw_query_trials": heldout["totals"]["raw_query_trials"],
            "directional_query_trials": heldout["totals"]["directional_query_trials"],
            "scored_windows": heldout_scored,
            "sessions": heldout["eligible_sessions"],
        },
        "heldin_post_support": {
            "session_count": heldin["totals"]["session_count"],
            "raw_post_support_trials": heldin["totals"]["raw_post_support_trials"],
            "directional_post_support_trials": heldin["totals"]["directional_post_support_trials"],
            "scored_windows": heldin_scored,
            "sessions": heldin["sessions_with_nonempty_post_support"],
        },
        "ratios": ratios,
    }


def build_audit(include_window_audit: bool = True, include_datamodule: bool = True) -> dict[str, Any]:
    if sha256(M33_AGGREGATE) != M33_AGGREGATE_SHA256:
        raise ValueError("M33 aggregate SHA-256 drift")
    if sha256(NATIVE_M2_AGGREGATE) != NATIVE_M2_AGGREGATE_SHA256:
        raise ValueError("native_m2 aggregate SHA-256 drift")

    minival = audit_heldin_minival_prefix()
    heldin = audit_heldin_postsupport_endpoint()
    heldout = audit_heldout_m33_endpoint()
    heldout_windows = heldout_m33_window_audit()
    heldin_windows = postsupport_window_audits() if include_window_audit else {
        "per_session": {},
        "totals": {"scored_windows": None, "query_trials": None, "session_count": 0},
    }

    audit: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "m2_heldin_calib_post_support_endpoint_feasibility_audit",
        "scope": (
            "Structural audit of the unused M2 held-in-calib post-support endpoint, "
            "comparison to the corrected held-out M33 endpoint, and precision-calibration "
            "protocol inputs for F0/T4 only.  Read-only; no training, scoring, optimizer "
            "step, checkpoint selection, or hidden EvalAI access."
        ),
        "candidate_branch_seal": CANDIDATE_BRANCH_SEAL,
        "frozen_inputs": {
            "m33_disjoint_replay_aggregate": {
                "path": str(M33_AGGREGATE.relative_to(ROOT)),
                "sha256": M33_AGGREGATE_SHA256,
            },
            "native_m2_internal_loso_aggregate": {
                "path": str(NATIVE_M2_AGGREGATE.relative_to(ROOT)),
                "sha256": NATIVE_M2_AGGREGATE_SHA256,
            },
        },
        "internal_loso_heldin_endpoint": minival,
        "heldin_postsupport_endpoint": heldin,
        "heldout_m33_endpoint": heldout,
        "heldout_m33_window_audit": heldout_windows,
        "heldin_postsupport_window_audit": heldin_windows,
        "endpoint_comparison": endpoint_comparison(heldin, heldout, heldin_windows, heldout_windows),
        "implementation_gap": implementation_gap(),
        "datamodule_guard_probe": datamodule_guard_probe(),
    }
    if include_datamodule:
        audit["loso_session_isolation"] = datamodule_loso_isolation()
    if include_window_audit and heldin_windows["totals"]["scored_windows"] is not None:
        audit["power_analysis"] = power_analysis(heldin, heldout, heldin_windows, heldout_windows)
    audit["explicitly_not_claimed"] = [
        "any candidate-branch number on the held-in post-support endpoint",
        "reopening closed M2 gate verdicts",
        "hidden EvalAI query or test evaluation",
        "seven-session between-session dispersion from frozen-checkpoint replay alone",
    ]
    return audit


def run(out_dir: Path, include_window_audit: bool = True, include_datamodule: bool = True) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing audit directory: {out_dir}")
    audit = build_audit(include_window_audit=include_window_audit, include_datamodule=include_datamodule)
    out_dir.mkdir(parents=True)
    path = out_dir / "audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "audit.sha256").write_text(f"{sha256(path)}  audit.json\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-window-audit", action="store_true")
    parser.add_argument("--skip-datamodule-check", action="store_true")
    args = parser.parse_args()
    path = run(
        args.out,
        include_window_audit=not args.skip_window_audit,
        include_datamodule=not args.skip_datamodule_check,
    )
    print(f"wrote {path}")
    print(f"sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
