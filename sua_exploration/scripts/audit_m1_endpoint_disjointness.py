#!/usr/bin/env python3
"""Read-only audit: does FALCON M1 have any support/query-disjoint local endpoint?

Both M1 endpoints that have been reported so far are checked against a single
structural question: is there any scored window that lies strictly after the
frozen chronological calibration support prefix?

* the local held-out-calib replay, whose scores were already withdrawn for
  resolving `query_start_trial` to 0;
* the internal-LOSO held-in endpoint, whose scores are still cited.

The audit opens NWB trial tables and instantiates the FALCON data module in
test/fit setup only.  It runs no training step, no optimizer, no scoring, and
never touches a hidden EvalAI query file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pynwb import NWBHDF5IO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "SPINT-main" / "data"
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_endpoint_infeasibility_v2"
V1_ARTIFACT = ROOT / "sua_exploration" / "results" / "m1_endpoint_infeasibility_v1"
V1_ARTIFACT_SHA256 = (
    "8fc07e6fc1bb697aabae2487bc3c0d8d50f9ad9a4396ff1f61f7f2f6bc5b4b5a"
)

# Frozen support budgets of the native-MUA T4 screen.  These are protocol
# constants, not choices made by this audit.
SUPPORT_TRIALS = {"m1": 10, "m2": 33}
WINDOW_SIZE = {"m1": 100, "m2": 50}
SUBJECT = {"m1": "sub-MonkeyL", "m2": "sub-MonkeyN"}
DANDISET = {"m1": "000941", "m2": "000953"}

# Artifacts whose reported numbers this audit re-scopes.  Bound by hash so the
# receipt cannot silently drift away from the aggregates it demotes.
FROZEN_INPUTS = {
    "internal_loso_aggregate_m1": (
        ROOT / "sua_exploration/results/native_mua_t4_v1/aggregate_m1.json",
        "aee9297409ec3accae00819b78c06636c7ca0c0d114667e6560db124dded35c2",
    ),
    "withdrawn_local_heldout_replay": (
        ROOT / "sua_exploration/results/native_mua_heldout_t4_v1/aggregate_heldout.json",
        "b44ebca073268b93930cc589c51c82f44b794c928d2f5a4a84662984ae9c1ddc",
    ),
}

# Hashing a 13 GB held-in calibration tree is not proportionate for a read-only
# structural audit; those files are identified by size and trial table instead.
HASH_SIZE_LIMIT_BYTES = 512 * 1024 * 1024


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


def trial_table(path: Path, include_tgt_loc: bool = False) -> dict[str, Any]:
    """Trial boundaries, and for M1 only the scalar target azimuth.

    M2 stores `tgt_loc` as a two-dimensional coordinate, so it is not read here.
    """
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        record = {
            "session_start_time": str(nwb.session_start_time),
            "n_trials": int(len(trials)),
            "start_time": np.asarray(trials["start_time"], dtype=float),
            "stop_time": np.asarray(trials["stop_time"], dtype=float),
        }
        if include_tgt_loc:
            record["tgt_loc"] = np.asarray(trials["tgt_loc"], dtype=float).reshape(-1)
    return record


def session_of(path: Path) -> str:
    return path.name.split("_ses-")[1].split("_behavior")[0]


def files_for(task: str, split: str) -> list[Path]:
    directory = DATA / DANDISET[task] / f"{SUBJECT[task]}-{split}"
    return sorted(directory.glob("*.nwb"))


def audit_heldout_endpoint(task: str) -> dict[str, Any]:
    """Does any held-out-calib session have a trial after the support prefix?"""
    support = SUPPORT_TRIALS[task]
    sessions: dict[str, Any] = {}
    for path in files_for(task, "held-out-calib"):
        table = trial_table(path)
        n = table["n_trials"]
        sessions[session_of(path)] = {
            **file_provenance(path),
            "n_trials": n,
            "support_prefix_trials": support,
            "query_trials_after_support": int(max(n - support, 0)),
            "has_disjoint_query": bool(n > support),
        }
    eligible = [k for k, v in sessions.items() if v["has_disjoint_query"]]
    return {
        "support_prefix_trials": support,
        "sessions": sessions,
        "eligible_sessions": eligible,
        "eligible_session_count": len(eligible),
        "total_session_count": len(sessions),
    }


def audit_heldin_endpoint(task: str) -> dict[str, Any]:
    """Is the minival scoring file a prefix of its own calibration support?"""
    support = SUPPORT_TRIALS[task]
    minival = {session_of(p): p for p in files_for(task, "held-in-minival")}
    sessions: dict[str, Any] = {}
    for calib_path in files_for(task, "held-in-calib"):
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
                float(calib["stop_time"][support - 1])
                if calib["n_trials"] >= support
                else None
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


def audit_m1_constructible_endpoint() -> dict[str, Any]:
    """Post-support held-in-calib trials available under validation_protocol=loso."""
    support = SUPPORT_TRIALS["m1"]
    sessions: dict[str, Any] = {}
    for calib_path in files_for("m1", "held-in-calib"):
        session = session_of(calib_path)
        table = trial_table(calib_path)
        n = table["n_trials"]
        post_support = int(max(n - support, 0))
        sessions[session] = {
            **file_provenance(calib_path),
            "total_calib_trials": n,
            "frozen_support_budget_trials": support,
            "post_support_trials": post_support,
            "has_nonempty_post_support": bool(post_support > 0),
            "support_boundary_stop_time": (
                float(table["stop_time"][support - 1]) if n >= support else None
            ),
            "first_post_support_trial_start_time": (
                float(table["start_time"][support]) if n > support else None
            ),
            "session_end_stop_time": float(table["stop_time"][-1]) if n else None,
        }
    return {
        "frozen_support_budget_trials": support,
        "sessions": sessions,
        "sessions_with_nonempty_post_support": [
            k for k, v in sessions.items() if v["has_nonempty_post_support"]
        ],
        "total_session_count": len(sessions),
        "implementation_gap": {
            "constructible_endpoint_not_yet_reachable": True,
            "reason": (
                "Under validation_protocol=loso the left-out session's held-in-calib "
                "file supplies the first calibration_n_trials (=10) as support while "
                "val_heldin_dataset currently loads held-in-minival (2 trials) with "
                "query_start_trial hardcoded to 0.  query_start_trial is plumbed only "
                "into val_heldout_dataset; train_dataset and val_heldin_dataset fix "
                "query_start_trial=0 (falcon_datamodule.py ~lines 1012 and 1059).  "
                "setup() additionally reserves query_start_trial for an explicit "
                "held-out test-only replay (~lines 897-902), so enabling the "
                "post-support held-in-calib query requires a datamodule plumbing "
                "change distinct from the scientific availability of the trials."
            ),
            "query_start_trial_plumbing": {
                "val_heldout_dataset": "uses self.hparams.query_start_trial",
                "train_dataset": "hardcoded query_start_trial=0",
                "val_heldin_dataset": "hardcoded query_start_trial=0",
            },
            "setup_guard": (
                "query_start_trial is reserved for an explicit held-out test-only replay"
            ),
        },
        "note": (
            "These trials are neither trained on nor scored by the reported "
            "internal-LOSO minival endpoint.  No decoding has been run on them."
        ),
    }


def audit_m1_direction_coverage() -> dict[str, Any]:
    """Directional coverage of the M1 target labels the T4 cosine fit relies on."""
    sessions: dict[str, Any] = {}
    for split in ("held-in-calib", "held-out-calib"):
        for path in files_for("m1", split):
            table = trial_table(path, include_tgt_loc=True)
            tgt = table["tgt_loc"]
            finite = tgt[np.isfinite(tgt)]
            prefix = finite[: SUPPORT_TRIALS["m1"]]
            theta = np.deg2rad(prefix)
            design = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
            rank = int(np.linalg.matrix_rank(design))
            unique = sorted({float(v) for v in finite})
            sessions[f"{split}/{session_of(path)}"] = {
                "unique_tgt_loc_deg": unique,
                "n_unique_directions": len(unique),
                "angular_span_deg": float(max(unique) - min(unique)) if unique else None,
                "support_design_rank": rank,
                "support_design_condition": (
                    float(np.linalg.cond(design)) if rank == 3 else None
                ),
            }
    spans = [v["angular_span_deg"] for v in sessions.values() if v["angular_span_deg"]]
    directions = [d for v in sessions.values() for d in v["unique_tgt_loc_deg"]]
    return {
        "sessions": sessions,
        "max_angular_span_deg": float(max(spans)) if spans else None,
        "min_direction_deg": float(min(directions)) if directions else None,
        "max_direction_deg": float(max(directions)) if directions else None,
        "covers_full_circle": False,
        "all_directions_within_one_half_plane": bool(
            directions and max(directions) - min(directions) < 180.0
        ),
        "note": (
            "Held-in sessions sample eight directions across a 157.5 degree arc and "
            "held-out sessions sample four across 135 degrees, but both stay inside a "
            "single half plane.  The cosine design is rank-3, so the fit is defined, "
            "yet [a, c] is extrapolated outside the sampled arc and the support "
            "condition numbers near 5.2--5.8 follow from that geometry.  This is a "
            "conditioning fact, not by itself a decoding result."
        ),
    }


def datamodule_evidence() -> dict[str, Any]:
    """Machine-checkable confirmation from the production data module itself."""
    import sys

    sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))
    from src.data.falcon_datamodule import FalconDataModule

    def build(**overrides: Any) -> FalconDataModule:
        kwargs: dict[str, Any] = dict(
            task="m1",
            data_dir=str(DATA / "000941") + "/",
            window_size=WINDOW_SIZE["m1"],
            calibration_n_trials=SUPPORT_TRIALS["m1"],
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
            num_workers=0,
            side_feature_group="none",
        )
        kwargs.update(overrides)
        return FalconDataModule(**kwargs)

    evidence: dict[str, Any] = {}

    dm = build(loso_fold=1, include_heldout_in_test=True, query_start_trial=0)
    dm.setup(stage="test")
    evidence["heldout_query_start_0"] = {
        "total_windows": len(dm.val_heldout_dataset),
        "per_session": dm.val_heldout_dataset.query_window_audit,
    }

    try:
        dm = build(
            loso_fold=1, include_heldout_in_test=True, query_start_trial=SUPPORT_TRIALS["m1"]
        )
        dm.setup(stage="test")
        evidence["heldout_query_start_10"] = {
            "raised": False,
            "total_windows": len(dm.val_heldout_dataset),
        }
    except Exception as exc:  # the fail-closed guard is the evidence being recorded
        evidence["heldout_query_start_10"] = {
            "raised": True,
            "error": f"{type(exc).__name__}: {exc}",
        }

    evidence["heldin_internal_loso"] = {}
    for fold in (1, 2):
        dm = build(loso_fold=fold, include_heldout_in_test=False)
        dm.setup(stage="fit")
        evidence["heldin_internal_loso"][f"fold{fold}"] = {
            "validation_sessions": list(dm.val_heldin_session_names),
            "scored_windows": len(dm.val_heldin_dataset),
            "per_session": dm.val_heldin_dataset.query_window_audit,
        }

    heldin_calib_sessions = sorted(session_of(p) for p in files_for("m1", "held-in-calib"))
    evidence["loso_session_isolation"] = {}
    for fold in range(len(heldin_calib_sessions)):
        dm = build(loso_fold=fold, include_heldout_in_test=False)
        dm.setup(stage="fit")
        left_out = list(dm.val_heldin_session_names)
        train_sessions = list(dm.train_session_names)
        evidence["loso_session_isolation"][f"fold{fold}"] = {
            "train_session_names": train_sessions,
            "val_heldin_session_names": left_out,
            "left_out_session": left_out[0] if len(left_out) == 1 else left_out,
            "left_out_session_absent_from_train": (
                len(left_out) == 1 and left_out[0] not in train_sessions
            ),
        }
    return evidence


def build_audit(include_datamodule_check: bool) -> dict[str, Any]:
    frozen: dict[str, Any] = {}
    for name, (path, expected) in FROZEN_INPUTS.items():
        digest = sha256(path)
        if digest != expected:
            raise ValueError(f"{name}: SHA-256 drift, refusing to write a stale receipt")
        frozen[name] = {"path": str(path.relative_to(ROOT)), "sha256": digest}

    m1_heldout = audit_heldout_endpoint("m1")
    m1_heldin = audit_heldin_endpoint("m1")
    m1_constructible = audit_m1_constructible_endpoint()
    m2_heldout = audit_heldout_endpoint("m2")
    m2_heldin = audit_heldin_endpoint("m2")

    audit: dict[str, Any] = {
        "schema_version": 2,
        "purpose": "m1_local_endpoint_support_query_disjointness_audit",
        "scope": (
            "Structural audit of every reported FALCON M1 evaluation endpoint, "
            "whether a support/query-disjoint endpoint can be constructed from "
            "held-in-calib post-support trials, with an M2 cross-check.  "
            "Read-only; no training, scoring, optimizer step, checkpoint "
            "selection, or hidden EvalAI access."
        ),
        "supersedes": {
            "artifact": str(V1_ARTIFACT.relative_to(ROOT)),
            "sha256": V1_ARTIFACT_SHA256,
        },
        "withdrawn_claims": {
            "no_local_disjoint_endpoint": (
                "v1 concluded that M1 has no local support/query-disjoint evaluation "
                "endpoint.  That overclaimed: the audit enumerated only the two "
                "already-reported endpoints and never asked whether a valid endpoint "
                "could be constructed from held-in-calib trials after the frozen "
                "first-10 support prefix."
            ),
            "evalai_only_route": (
                "v1 stated that a FALCON EvalAI submission is the only route to a "
                "support/query-disjoint M1 result.  Withdrawn: roughly 366-404 "
                "post-support trials per held-in session are available locally once "
                "the datamodule plumbing is extended."
            ),
        },
        "frozen_inputs": frozen,
        "m1": {
            "local_heldout_calib_endpoint": m1_heldout,
            "internal_loso_heldin_endpoint": m1_heldin,
            "constructible_endpoint": m1_constructible,
            "direction_coverage": audit_m1_direction_coverage(),
        },
        "m2_cross_check": {
            "local_heldout_calib_endpoint": m2_heldout,
            "internal_loso_heldin_endpoint": m2_heldin,
        },
    }

    if include_datamodule_check:
        audit["m1"]["datamodule_evidence"] = datamodule_evidence()

    constructible_sessions = m1_constructible["sessions_with_nonempty_post_support"]
    post_support_counts = [
        m1_constructible["sessions"][s]["post_support_trials"] for s in constructible_sessions
    ]
    audit["conclusions"] = [
        (
            "Both reported M1 endpoints are support-internal: "
            f"{m1_heldout['eligible_session_count']}/{m1_heldout['total_session_count']} "
            "held-out-calib sessions have a trial after the frozen first-10 support, and "
            f"{m1_heldin['contaminated_session_count']}/{m1_heldin['total_session_count']} "
            "held-in sessions score only trials inside that support prefix via the "
            "held-in-minival file (2 trials, a bit-exact calib prefix)."
        ),
        (
            "The M1 held-in minival file is a bit-exact prefix of its own calibration "
            "file, so the internal-LOSO endpoint scores the very trials supplied as "
            "calibration support, and the T4 descriptor is fit on those same trials."
        ),
        (
            "The overlap direction favours T4, because T4 is scored in-sample with "
            "respect to its own descriptor fit while F0 has no such fit and TS4's "
            "permuted descriptor cannot fit those trials."
        ),
        (
            f"M1 held-out-calib cannot be corrected: "
            f"{m1_heldout['eligible_session_count']}/{m1_heldout['total_session_count']} "
            "sessions contain exactly 10 trials (equal to the frozen support budget), so "
            "0 post-support trials exist and query_start_trial=10 fails closed."
        ),
        (
            "A large uncontaminated M1 endpoint is constructible from held-in-calib "
            f"post-support trials: {len(constructible_sessions)}/"
            f"{m1_constructible['total_session_count']} held-in sessions have "
            f"{min(post_support_counts)}-{max(post_support_counts)} trials after the "
            "frozen first-10 support.  Under validation_protocol=loso the left-out "
            "session is absent from train_session_names.  This endpoint has not been "
            "evaluated and is not yet reachable through the current datamodule path."
        ),
        (
            f"M2 shares the same held-in minival prefix structure "
            f"({m2_heldin['contaminated_session_count']}/{m2_heldin['total_session_count']} "
            "sessions), which is already recorded as a diagnostic-only endpoint, but "
            f"unlike M1 it retains {m2_heldout['eligible_session_count']}/"
            f"{m2_heldout['total_session_count']} held-out sessions with a genuine "
            "future query."
        ),
        (
            "A reduced M1 support budget would leave a query arithmetically, but it "
            "would mismatch checkpoints trained at first-10 support and would be a "
            "protocol change chosen after observing that first-10 fails."
        ),
    ]
    audit["claim_revisions"] = {
        "m1_t4_minus_f0_no_net_gain": (
            "Retained and strengthened.  The reported +0.007837 with 1/3 cells positive "
            "was measured under conditions favourable to T4."
        ),
        "m1_t4_content_distinguishable_from_ts4": (
            "Withdrawn as evidence.  The reported +0.024797 cannot be separated from an "
            "in-sample descriptor-fit advantage on the scored trials."
        ),
        "m1_local_heldout_effect_sizes": (
            "Permanently void.  No corrected re-inference is possible from local data."
        ),
    }
    audit["explicitly_not_claimed"] = [
        "M1 native MUA lacks functional calibration information",
        "the M1 absolute R2 values are wrong",
        "M2 held-out q24/q33 results are affected",
        "any hidden EvalAI query or test file was opened or evaluated",
        "any decoding result on the constructible held-in-calib post-support endpoint",
    ]
    audit["constructible_local_endpoint"] = {
        "status": "available_not_evaluated",
        "source": "held-in-calib trials strictly after the frozen first-10 support prefix",
        "per_session_post_support_trials": {
            session: m1_constructible["sessions"][session]["post_support_trials"]
            for session in constructible_sessions
        },
        "requires_datamodule_plumbing_change": True,
        "evalai_also_remains_available": True,
    }
    return audit


def run(out_dir: Path, include_datamodule_check: bool = True) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing audit directory: {out_dir}")
    audit = build_audit(include_datamodule_check)
    out_dir.mkdir(parents=True)
    path = out_dir / "audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "audit.sha256").write_text(f"{sha256(path)}  audit.json\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-datamodule-check", action="store_true")
    args = parser.parse_args()
    path = run(args.out, include_datamodule_check=not args.skip_datamodule_check)
    print(f"wrote {path}")
    print(f"sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
