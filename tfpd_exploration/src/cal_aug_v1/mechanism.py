"""CAL-AUG section 5: source-roster mechanism readout + registration gate.

New additive source-27 materialization (the source sessions are NOT reachable
through the z1/p4 hard-gated surfaces): the sealed
``run_admission_arm.build_datamodule`` loader provides
``dm.train_dataset.sessions[name]`` per source session — ``neural``,
``behavior``, ``calib_trials [30,100,N]``, ``valid_starts`` and
``side_features`` (the TRAINING M30 T4, byte-bound by
``t4_authority_fingerprint``).

For arm X in {T0, C1} and M in {30, 10, 4} (chronological prefix
``calib[:M]``, T4 = the SAME training side for every M within an arm):

    prefix_degradation_X(M) = R2_X(B3S=M, T4=M30) - R2_X(B3S=M30, T4=M30)
    prefix_robustness_recovery(M) = prefix_degradation_C1(M) - prefix_degradation_T0(M)

Equal-session mean over 27 with every paired per-session delta; last-bin
(governed bin 49) house ``session_r2``; valid mask ``(behavior != -1).all(-1)``;
B3S identity-vector distances are DESCRIPTIVE only (collapse caveat).

Registration gate (mechanism evidence, never a deployment claim):

    >=1 of M4/M10 recovery >= +0.01 R2  AND  positive source sessions >= 18/27
    AND  C1 - T0 at M30 >= -0.01

Boundary rule: exact ``>=`` on the float64 equal-session mean (0.01 passes,
0.01 - 1e-13 fails); the program epsilon 1e-12 is a disclosed boundary band
that never flips a verdict.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from . import plan, schedule
from .receipts import sha256_file


class MechanismError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MechanismError(message)


def _sealed_module(name: str):
    """The sealed lane module the runner stack already loaded (single copy)."""
    module = sys.modules.get(name)
    if module is None:
        raise MechanismError(
            f"sealed module {name!r} is not loaded; load the sealed runner stack first"
        )
    return module


# ---------------------------------------------------------------------------
# pure gate arithmetic (float64; no torch needed)
# ---------------------------------------------------------------------------


def equal_session_mean(values: Sequence[float]) -> float:
    _require(bool(values), "equal-session mean of an empty roster")
    return float(sum(float(v) for v in values) / len(values))


def prefix_degradation(r2_prefix: float, r2_m30: float) -> float:
    """``R2(B3S=M, T4=M30) - R2(B3S=M30, T4=M30)`` for one arm/prefix."""
    return float(r2_prefix) - float(r2_m30)


def margin_verdict(value: float, margin: float, epsilon: float = plan.GATE_BOUNDARY_EPSILON) -> dict:
    """Exact ``>=`` margin test plus the disclosed program-epsilon band."""
    meets = float(value) >= float(margin)
    return {
        "value": float(value),
        "margin": float(margin),
        "meets_margin": bool(meets),
        "within_epsilon_band_of_boundary": bool(
            (not meets) and float(value) >= float(margin) - float(epsilon)
        ),
        "boundary_epsilon": float(epsilon),
        "boundary_rule": plan.GATE_SPEC["boundary_rule"],
    }


def registration_gate(
    *,
    recovery_m10: float,
    recovery_m4: float,
    positive_sessions_m10: int,
    positive_sessions_m4: int,
    c1_minus_t0_m30_mean: float,
    n_source_sessions: int = plan.SOURCE_SESSIONS_REQUIRED,
) -> dict:
    """Section 5 registration gate over the crossed readout's summary rows."""
    m10 = margin_verdict(recovery_m10, plan.MECHANISM_RECOVERY_MARGIN_R2)
    m4 = margin_verdict(recovery_m4, plan.MECHANISM_RECOVERY_MARGIN_R2)
    breadth_m10 = int(positive_sessions_m10) >= plan.MECHANISM_POSITIVE_SOURCE_SESSIONS_REQUIRED
    breadth_m4 = int(positive_sessions_m4) >= plan.MECHANISM_POSITIVE_SOURCE_SESSIONS_REQUIRED
    lead_m10 = bool(m10["meets_margin"] and breadth_m10)
    lead_m4 = bool(m4["meets_margin"] and breadth_m4)
    m30_safety = margin_verdict(c1_minus_t0_m30_mean, plan.MECHANISM_M30_SAFETY_MARGIN_R2)
    passed = bool((lead_m10 or lead_m4) and m30_safety["meets_margin"])
    return {
        "expression": plan.GATE_SPEC["mechanism_registration"]["expression"],
        "scope": plan.GATE_SPEC["mechanism_registration"]["scope"],
        "n_source_sessions": int(n_source_sessions),
        "positive_sessions_required": plan.MECHANISM_POSITIVE_SOURCE_SESSIONS_REQUIRED,
        "recovery_margin_m10": m10,
        "recovery_margin_m4": m4,
        "breadth": {
            "m10_positive_sessions": int(positive_sessions_m10),
            "m10_breadth_passed": bool(breadth_m10),
            "m4_positive_sessions": int(positive_sessions_m4),
            "m4_breadth_passed": bool(breadth_m4),
        },
        "lead_prefix": "m10" if lead_m10 and not lead_m4 else ("m4" if lead_m4 and not lead_m10 else ("both" if lead_m10 and lead_m4 else "none")),
        "m30_safety_c1_minus_t0": m30_safety,
        "passed": passed,
        "disposition": plan.MECHANISM_REGISTERED if passed else plan.MECHANISM_NULL,
        "descriptive_only": {
            "b3s_identity_distance": (
                "reported per session/prefix as descriptive only; collapse can "
                "make a latent distance look artificially small"
            )
        },
    }


def per_session_recovery(
    r2_c1_m: Mapping[str, float], r2_c1_m30: Mapping[str, float],
    r2_t0_m: Mapping[str, float], r2_t0_m30: Mapping[str, float],
) -> dict:
    """Paired per-session recovery + the equal-session mean recovery."""
    roster = tuple(r2_c1_m)
    _require(
        tuple(r2_c1_m30) == roster and tuple(r2_t0_m) == roster and tuple(r2_t0_m30) == roster,
        "mechanism paired roster drift",
    )
    per_session = {
        name: (float(r2_c1_m[name]) - float(r2_c1_m30[name]))
        - (float(r2_t0_m[name]) - float(r2_t0_m30[name]))
        for name in roster
    }
    return {
        "per_session": {k: float(v) for k, v in per_session.items()},
        "equal_session_mean_recovery": equal_session_mean(list(per_session.values())),
        "positive_sessions": int(sum(1 for v in per_session.values() if v > 0.0)),
        "n_sessions": len(roster),
    }


# ---------------------------------------------------------------------------
# source-27 materialization through the sealed training loader
# ---------------------------------------------------------------------------


def materialize_source_sessions(arm_runner, args) -> dict:
    """Source-27 records through the sealed ``arm_runner.build_datamodule``.

    Returns the roster (``dm.session_splits["train"]`` order), the per-session
    records, the ``t4_authority_fingerprint`` and the no-target-path flags.
    The manifest SHA and behavior-normalizer semantic checks mirror the sealed
    runner's own fail-closed gates.
    """
    dm, a2 = arm_runner.build_datamodule(args)
    train_dataset = dm.train_dataset
    roster = tuple(dm.session_splits["train"])
    _require(
        len(roster) == plan.SOURCE_SESSIONS_REQUIRED,
        f"strict-27 roster drift: {len(roster)}",
    )
    manifest_sha = sha256_file(a2.MANIFEST_PATH)
    _require(
        manifest_sha == plan.EXPECTED_MANIFEST_SHA256,
        f"manifest SHA drift: {manifest_sha}",
    )
    behavior_semantic = a2.normalizer_value_sha256(*dm._behavior_stats)
    _require(
        behavior_semantic.startswith(plan.BEHAVIOR_NORMALIZER_SEMANTIC_PREFIX),
        f"source behavior normalizer semantic SHA drift: {behavior_semantic}",
    )
    arm_common = _sealed_module("tfpd_lane_arm_common")
    authority = arm_common.t4_authority_fingerprint(train_dataset.sessions)
    sessions = {
        name: train_dataset.sessions[name]
        for name in roster
    }
    _require(set(sessions) == set(roster), "source session resolution drift")
    return {
        "roster": roster,
        "sessions": sessions,
        "t4_authority_fingerprint": authority,
        "no_target_path": {
            "within_dev_sessions_opened": False,
            "external_sub_m_opened": False,
            "formal_or_organizer_held_data_opened": False,
            "val_paths_resolved": list(dm.session_files["val"]),
            "test_paths_resolved": list(dm.session_files["test"]),
            "val_and_test_empty": dm.session_files["val"] == [] and dm.session_files["test"] == [],
        },
        "n_train_windows": len(train_dataset.window_indices),
    }


def load_arm_swa_model(pop_robust, arm_common, swa_path: Path, device=None):
    """Strict-load an arm's sealed ``swa_final4.pt`` into the Cell-D graph."""
    import torch

    path = Path(swa_path)
    sidecar = Path(str(path) + ".sha256")
    _require(path.is_file() and sidecar.is_file(), f"arm SWA artifact missing: {path}")
    digest = sha256_file(path)
    _require(
        digest == sidecar.read_text().split()[0],
        f"arm SWA SHA drift against its sidecar: {path}",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = pop_robust.build_population_robustness_model(seed=plan.SEED, cell="D")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if device is not None:
        model.to(device)
    return model, digest


def score_prefix(
    model, record, m: int, device, eval_batch: int = 32, pop_robust=None,
) -> dict:
    """One (arm, session, M) cell: chronological ``calib[:M]`` + training T4.

    Mirrors the sealed crossed-cell forward law: chunk-32 batches over the
    training window law (``record.valid_starts``), chronological calibration
    prefix expanded per batch, eval mode, ``torch.no_grad``, the passive
    recorder proving dropout never activates, governed bin 49, house
    ``session_r2`` on the last-bin rows that satisfy the training valid mask
    ``(behavior != -1).all(-1)``.
    """
    import numpy as np
    import torch

    pr = pop_robust if pop_robust is not None else _sealed_module("tfpd_lane_pop_robust")
    m = int(m)
    _require(1 <= m <= plan.MAX_CALIBRATION_TRIALS, f"prefix length drift: {m}")
    starts = np.asarray(record.valid_starts, dtype=np.int64)
    neural_all = np.asarray(record.neural, dtype=np.float32)
    behavior = np.asarray(record.behavior, dtype=np.float32)
    calib_all = np.asarray(record.calib_trials, dtype=np.float32)
    side_all = np.asarray(record.side_features, dtype=np.float32)
    _require(calib_all.shape[0] == plan.MAX_CALIBRATION_TRIALS, "calibration block drift")
    prefix = calib_all[:m]
    was_training = model.training
    model.eval()
    predictions: list = []
    targets: list = []
    digest = hashlib.sha256()
    n_valid = 0
    try:
        with pr.dynamic_dropout_recorder() as recorder:
            with torch.no_grad():
                for offset in range(0, starts.size, eval_batch):
                    chunk = starts[offset:offset + eval_batch]
                    neural = torch.from_numpy(
                        np.stack([neural_all[s:s + 50] for s in chunk])
                    ).to(device)
                    calibration = (
                        torch.from_numpy(prefix)
                        .to(device)
                        .unsqueeze(0)
                        .expand(len(chunk), -1, -1, -1)
                    )
                    side = (
                        torch.from_numpy(side_all)
                        .to(device)
                        .unsqueeze(0)
                        .expand(len(chunk), -1, -1)
                    )
                    output, _identity = model(
                        neural, calib_trials=calibration, side_features=side
                    )
                    cpu = output.detach().cpu().contiguous()
                    _require(
                        tuple(cpu.shape) == (len(chunk), 50, 2)
                        and bool(torch.isfinite(cpu).all().item()),
                        "mechanism output shape/nonfinite drift",
                    )
                    digest.update(cpu.numpy().tobytes())
                    predictions.append(cpu[:, 49, :])
                    targets.append(
                        torch.from_numpy(
                            np.stack([behavior[s + 49] for s in chunk])
                        )
                    )
            _require(
                recorder["uniform_calls"] == 0 and recorder["dropout_calls"] == [],
                "mechanism eval dropout became active",
            )
    finally:
        if was_training:
            model.train()
    prediction = torch.cat(predictions).contiguous()
    target = torch.cat(targets).contiguous()
    valid = (target != -1.0).all(dim=-1)
    n_valid = int(valid.sum().item())
    _require(n_valid > 0, "mechanism last-bin valid mask is empty")
    matched_scorer = _sealed_module("tfpd_lane_matched_scorer")
    r2 = matched_scorer.session_r2(prediction[valid], target[valid])
    return {
        "n_windows": int(starts.size),
        "n_valid_last_bin": n_valid,
        "all_rows_valid": bool(valid.all().item()),
        "r2": float(r2),
        "prediction_sha256": digest.hexdigest(),
        "calib_prefix_sha256": schedule.visible_slice_digest(torch.from_numpy(prefix)),
        "side_sha256": schedule.visible_slice_digest(torch.from_numpy(side_all)),
        "m_prefix": m,
    }


def identity_distance_row(model, record, m: int, device) -> dict:
    """DESCRIPTIVE: B3S identity-vector distance, shortened prefix vs M30."""
    import numpy as np
    import torch

    prefix_m = np.asarray(record.calib_trials, dtype=np.float32)[: int(m)]
    prefix_30 = np.asarray(record.calib_trials, dtype=np.float32)[:30]
    side = torch.from_numpy(np.asarray(record.side_features, dtype=np.float32)).to(device)
    with torch.no_grad():
        identity_m = model.compute_identity(torch.from_numpy(prefix_m).unsqueeze(0).to(device), side_features=side.unsqueeze(0))
        identity_30 = model.compute_identity(torch.from_numpy(prefix_30).unsqueeze(0).to(device), side_features=side.unsqueeze(0))
    diff = (identity_m - identity_30).double()
    base = identity_30.double()
    return {
        "m_prefix": int(m),
        "identity_l2": float(diff.norm().item()),
        "identity_relative_l2": float(
            (diff.norm() / base.norm()).item() if base.norm().item() > 0 else float("nan")
        ),
        "descriptive_only": True,
        "collapse_caveat": (
            "identity collapse can make a distance look artificially small; "
            "never used by any gate"
        ),
    }


def mechanism_readout(
    *, models_by_arm: Mapping[str, object], source: Mapping, device, pop_robust,
) -> dict:
    """The full crossed readout over arms x prefixes x 27 source sessions."""
    roster = tuple(source["roster"])
    _require(
        tuple(sorted(models_by_arm)) == ("c1", "t0"),
        "mechanism readout needs exactly the t0 and c1 models",
    )
    rows: dict[str, dict[int, dict[str, float]]] = {}
    detail: dict[str, dict[int, list[dict]]] = {}
    identity_rows: dict[str, list[dict]] = {}
    started = time.perf_counter()
    for arm in ("t0", "c1"):
        model = models_by_arm[arm]
        rows[arm] = {}
        detail[arm] = {}
        identity_rows[arm] = []
        for m in plan.READOUT_PREFIXES:
            detail[arm][m] = []
        for name in roster:
            record = source["sessions"][name]
            for m in plan.READOUT_PREFIXES:
                cell = score_prefix(model, record, m, device, pop_robust=pop_robust)
                detail[arm][m].append({"session": name, **cell})
                rows[arm].setdefault(m, {})[name] = cell["r2"]
            for m in plan.DEGRADED_PREFIXES:
                identity_rows[arm].append(
                    {"session": name, **identity_distance_row(model, record, m, device)}
                )
    _require(
        all(len(rows[arm][m]) == len(roster) for arm in rows for m in rows[arm]),
        "mechanism cell topology drift",
    )

    matched_scorer = _sealed_module("tfpd_lane_matched_scorer")
    paired_session_stats = matched_scorer.paired_session_stats

    per_arm_summary = {
        arm: {
            str(m): {
                "equal_session_mean_r2": equal_session_mean(list(rows[arm][m].values())),
                "per_session_r2": {k: float(v) for k, v in rows[arm][m].items()},
            }
            for m in plan.READOUT_PREFIXES
        }
        for arm in rows
    }
    degradation = {
        arm: {
            str(m): {
                "equal_session_mean": per_arm_summary[arm][str(m)]["equal_session_mean_r2"]
                - per_arm_summary[arm]["30"]["equal_session_mean_r2"],
                "per_session": {
                    name: prefix_degradation(rows[arm][m][name], rows[arm][30][name])
                    for name in roster
                },
            }
            for m in plan.DEGRADED_PREFIXES
        }
        for arm in rows
    }
    recovery = {
        str(m): per_session_recovery(
            rows["c1"][m], rows["c1"][30], rows["t0"][m], rows["t0"][30]
        )
        for m in plan.DEGRADED_PREFIXES
    }
    m30_delta_sessions = {
        name: float(rows["c1"][30][name]) - float(rows["t0"][30][name]) for name in roster
    }
    m30_stats = paired_session_stats([m30_delta_sessions[name] for name in roster], seed=plan.BOOTSTRAP_SEED)
    m30_stats["per_session_delta"] = {k: float(v) for k, v in m30_delta_sessions.items()}
    recovery_stats = {
        str(m): {
            **recovery[str(m)],
            "bootstrap": paired_session_stats(
                list(recovery[str(m)]["per_session"].values()), seed=plan.BOOTSTRAP_SEED
            ),
        }
        for m in plan.DEGRADED_PREFIXES
    }
    gate = registration_gate(
        recovery_m10=recovery["10"]["equal_session_mean_recovery"],
        recovery_m4=recovery["4"]["equal_session_mean_recovery"],
        positive_sessions_m10=recovery["10"]["positive_sessions"],
        positive_sessions_m4=recovery["4"]["positive_sessions"],
        c1_minus_t0_m30_mean=m30_stats["mean"],
    )
    return {
        "schema": plan.SCHEMA + "_mechanism_readout",
        "n_source_sessions": len(roster),
        "roster": list(roster),
        "t4_authority_fingerprint": dict(source["t4_authority_fingerprint"]),
        "no_target_path": dict(source["no_target_path"]),
        "prefixes": list(plan.READOUT_PREFIXES),
        "t4_mode": "the SAME training M30 side features for every M within an arm",
        "equal_session_summary": per_arm_summary,
        "prefix_degradation": degradation,
        "prefix_robustness_recovery": recovery_stats,
        "c1_minus_t0_at_m30": m30_stats,
        "registration_gate": gate,
        "identity_distance_descriptive": identity_rows,
        "wall_seconds": time.perf_counter() - started,
        "target_optimizer_backward_update": 0,
    }
