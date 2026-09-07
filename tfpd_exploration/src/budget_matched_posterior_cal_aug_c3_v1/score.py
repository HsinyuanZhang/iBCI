"""Matched posterior-input deployment score for C2 and C3.

This successor fixes one important semantic boundary: C2/C3 were trained on
the budget-matched posterior carrier, so their deployment cells must consume
that same estimator.  The older generic CAL-AUG scorer consumes ridge-T4 and
is retained only as a diagnostic; it is never accepted as the E02/E03 score.

Target sessions are materialized once.  Four numerical arms are then decoded
on each identical record: C2, C3-Const, C3-Real, and the same C3-Real model
with only the q rows deterministically permuted.  Existing sealed T0/C1 rows
are descriptor-reloaded as comparators and exact-matched on target, selected
support, calibration prefix, window count, and valid-row count.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from budget_matched_posterior_cal_aug_v1 import c2_full, c2_smoke
from budget_matched_posterior_cal_aug_v1 import source_audit as audit_v1
from budget_matched_posterior_cal_aug_v1.posterior import (
    angular_reliability,
    array_sha256,
    fit_posterior_mean,
)
from budget_matched_posterior_cal_aug_v1.training import (
    posterior_normalizer_from_payload,
    source_prior_from_payload,
)

from . import full
from .features import ReliabilityNormalizer, widen_cell_d_for_reliability


SCHEMA = "budget_matched_posterior_cal_aug_c2_c3_posterior_score_v1"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c2_c3_posterior_score_v1"
)
BASELINE_SCORE_ROOT_RELATIVE = "tfpd_exploration/results/cal_aug_v1/deployment"
BASELINE_TERMINAL_SHA256 = (
    "a6e1b72c42d1956125915c720bbacdc00948655877b5f6bfc949f35f8a1bce68"
)
EXPECTED_FULL_LEAVES = 112
SURFACES = ("within", "external")
BUDGETS = (4, 10, 30)
NUMERICAL_ARMS = ("c2", "c3_constant", "c3_real", "c3_real_q_shuffle")
COMPARATORS = ("t0", "c1")
BOOTSTRAP_SEED = 42
SHUFFLE_DOMAIN = "C3_REAL_Q_ROW_SHUFFLE_EVAL_V1"
WORST_SESSION_SAFETY_DELTA = -0.02

EXECUTION_PATHS = tuple(dict.fromkeys((
    *full.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/score.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_score_v1.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/c2_full.py",
    "tfpd_exploration/src/cal_aug_v1/receipts.py",
    "tfpd_exploration/src/cal_aug_v1/plan.py",
    "tfpd_exploration/src/cal_aug_v1/deployment.py",
    "tfpd_exploration/src/calibration_gap_v1/p4_stream_stats.py",
    "tfpd_exploration/src/calibration_gap_v1/z1_oracle_cells.py",
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/src/low_cost_calibration_v1.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/plan.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/matched_score.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/matched_score_physical.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
    "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
    "tfpd_exploration/src/tfpd_lane/__init__.py",
    "tfpd_exploration/src/tfpd_lane/mech_diag.py",
    "tfpd_exploration/src/tfpd_lane/pregate.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "tfpd_exploration/src/tfpd_lane/receipt.py",
    "tfpd_exploration/scripts/run_pop_robust_cell.py",
    "tfpd_exploration/scripts/run_admission_arm.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)))
REVIEW_PATHS = (
    "tfpd_exploration/docs/WORKORDER_BUDGET_MATCHED_POSTERIOR_CAL_AUG_C3_SCORE_V1_20260830.md",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_score_v1.py",
)


class C3ScoreError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C3ScoreError(message)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_sha(path: Path, expected: str | None = None) -> str:
    _require(path.is_file() and not path.is_symlink(), f"body absent/symlink: {path}")
    sidecar = Path(str(path) + ".sha256")
    _require(sidecar.is_file() and not sidecar.is_symlink(), f"sidecar absent/symlink: {sidecar}")
    body_sha = _file_sha(path)
    words = sidecar.read_text().split()
    _require(len(words) >= 1 and words[0] == body_sha, f"sidecar mismatch: {path}")
    if expected is not None:
        _require(body_sha == expected, f"body digest drift: {path}")
    return body_sha


def _closure(repository_root: Path, paths: Sequence[str], suffix: str) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for relative in paths:
        path = repository_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf absent/symlink: {relative}")
        rows[relative] = {"sha256": _file_sha(path), "bytes": path.stat().st_size}
    payload: dict[str, object] = {"schema": SCHEMA + suffix, "files": rows}
    payload["closure_sha256"] = audit_v1._json_sha(payload)
    return payload


def execution_closure(repository_root: Path) -> dict[str, object]:
    return _closure(repository_root.resolve(), EXECUTION_PATHS, "_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return _closure(repository_root.resolve(), REVIEW_PATHS, "_review_closure")


def review_drift(start: Mapping[str, object], end: Mapping[str, object]) -> dict[str, object]:
    before = start["files"]
    after = end["files"]
    changed = [path for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path)]
    return {
        "status": "NO_REVIEW_DRIFT" if not changed else "ACCEPTED_NON_NUMERIC_DRIFT",
        "changed_paths": changed,
        "start_sha256": start["closure_sha256"],
        "end_sha256": end["closure_sha256"],
        "numerical_acceptance_affected": False,
    }


def _validate_immutable_root(root: Path, expected_leaves: int) -> None:
    _require(root.is_dir() and not root.is_symlink(), f"producer root absent/symlink: {root}")
    entries = list(root.iterdir())
    _require(len(entries) == expected_leaves, f"producer leaf count drift: {root} {len(entries)}")
    for entry in entries:
        info = entry.lstat()
        _require(stat.S_ISREG(info.st_mode), f"non-regular producer leaf: {entry}")
        _require(stat.S_IMODE(info.st_mode) == 0o444, f"producer mode drift: {entry}")


def validate_c2_producer(
    repository_root: Path, *, open_checkpoint_body: bool = False
) -> dict[str, object]:
    root = repository_root / c2_full.RESULT_ROOT_RELATIVE
    _validate_immutable_root(root, EXPECTED_FULL_LEAVES)
    _require(not (root / "failure.json").exists(), "C2 producer has failure receipt")
    terminal_path = root / "terminal.json"
    terminal_sha = _pair_sha(terminal_path)
    terminal = json.loads(terminal_path.read_text())
    _require(terminal.get("status") == "C2_FULL_TRAINING_TERMINAL", "C2 terminal status drift")
    _require(terminal.get("epoch_count") == 48 and terminal.get("optimizer_steps") == 1_628_400,
             "C2 training cardinality drift")
    _require(terminal.get("within_opened") is False and terminal.get("external_opened") is False,
             "C2 producer opened targets")
    _require(terminal.get("target_optimizer_backward_update") == 0, "C2 target update drift")
    swa = terminal.get("swa", {})
    _require(swa.get("name") == "swa_final4.pt", "C2 SWA basename drift")
    _require(swa.get("window_epochs") == [44, 45, 46, 47], "C2 SWA window drift")
    _require(swa.get("strict_reload_finite_forward") is True, "C2 strict reload proof absent")
    swa_path = root / "swa_final4.pt"
    swa_sha = str(swa.get("sha256"))
    _require(len(swa_sha) == 64, "C2 declared SWA digest drift")
    if open_checkpoint_body:
        _require(_pair_sha(swa_path) == swa_sha, "C2 SWA link drift")
    return {
        "root_relative": c2_full.RESULT_ROOT_RELATIVE,
        "terminal_sha256": terminal_sha,
        "attempt_sha256": terminal["attempt_sha256"],
        "launch_sha256": terminal["launch_sha256"],
        "execution_closure_sha256": terminal["closure_sha256"],
        "swa_relative": str(swa_path.relative_to(repository_root)),
        "swa_sha256": swa_sha,
        "swa_state_dict_sha256": swa["state_dict_sha256"],
        "exact_leaf_count": EXPECTED_FULL_LEAVES,
    }


def validate_c3_producer(
    repository_root: Path, arm: str, *, open_checkpoint_body: bool = False
) -> dict[str, object]:
    _require(arm in {"constant", "real"}, "unknown C3 producer arm")
    selected = full.C3Arm(arm)
    root = repository_root / full.RESULT_ROOTS[selected]
    _validate_immutable_root(root, EXPECTED_FULL_LEAVES)
    _require(not (root / "failure.json").exists(), f"C3 {arm} producer has failure receipt")
    terminal_path = root / "terminal.json"
    terminal_sha = _pair_sha(terminal_path)
    terminal = json.loads(terminal_path.read_text())
    _require(terminal.get("status") == "C3_FULL_TRAINING_TERMINAL", f"C3 {arm} status drift")
    _require(terminal.get("arm") == arm, f"C3 {arm} identity drift")
    _require(terminal.get("epoch_count") == 48 and terminal.get("optimizer_steps") == 1_628_400,
             f"C3 {arm} training cardinality drift")
    _require(terminal.get("target_data_opened") is False, f"C3 {arm} opened targets")
    _require(terminal.get("target_optimizer_backward_update") == 0, f"C3 {arm} target update drift")
    swa = terminal.get("swa", {})
    _require(swa.get("name") == "swa_final4.pt", f"C3 {arm} SWA basename drift")
    _require(swa.get("window_epochs") == [44, 45, 46, 47], f"C3 {arm} SWA window drift")
    _require(swa.get("strict_reload_finite_forward") is True, f"C3 {arm} strict reload proof absent")
    swa_path = root / "swa_final4.pt"
    swa_sha = str(swa.get("sha256"))
    _require(len(swa_sha) == 64, f"C3 {arm} declared SWA digest drift")
    if open_checkpoint_body:
        _require(_pair_sha(swa_path) == swa_sha, f"C3 {arm} SWA link drift")
    return {
        "arm": arm,
        "root_relative": full.RESULT_ROOTS[selected],
        "terminal_sha256": terminal_sha,
        "attempt_sha256": terminal["attempt_sha256"],
        "launch_sha256": terminal["launch_sha256"],
        "execution_closure_sha256": terminal["execution_closure_sha256"],
        "swa_relative": str(swa_path.relative_to(repository_root)),
        "swa_sha256": swa_sha,
        "swa_state_dict_sha256": swa["state_dict_sha256"],
        "exact_leaf_count": EXPECTED_FULL_LEAVES,
    }


def validate_baseline_score(repository_root: Path) -> dict[str, object]:
    root = repository_root / BASELINE_SCORE_ROOT_RELATIVE
    _validate_immutable_root(root, 4)
    terminal_path = root / "terminal.json"
    terminal_sha = _pair_sha(terminal_path, BASELINE_TERMINAL_SHA256)
    terminal = json.loads(terminal_path.read_text())
    _require(terminal.get("status") == "DEPLOYMENT_SCORING_COMPLETE", "baseline score status drift")
    arms = terminal.get("arms", {})
    _require(set(arms) == set(COMPARATORS), "baseline arm topology drift")
    for arm in COMPARATORS:
        rows = arms[arm].get("rows", [])
        _require(len(rows) == 63, f"baseline {arm} row cardinality drift")
        _require(arms[arm].get("state_unchanged") is True, f"baseline {arm} state mutation")
    return {
        "root_relative": BASELINE_SCORE_ROOT_RELATIVE,
        "terminal_sha256": terminal_sha,
        "attempt_sha256": terminal["attempt_sha256"],
        "terminal": terminal,
        "exact_leaf_count": 4,
    }


def reliability_normalizer_from_smoke(repository_root: Path) -> tuple[ReliabilityNormalizer, dict[str, object]]:
    predecessor = full.validate_smoke_predecessor(repository_root)
    payload = predecessor["terminal"]["q_normalizer"]
    counts = payload["per_budget_row_count"]
    digests = payload["per_budget_rows_sha256"]
    normalizer = ReliabilityNormalizer(
        mean=float(payload["mean_float64"]),
        std=float(payload["std_float64"]),
        row_count=int(payload["row_count"]),
        rows_sha256=str(payload["rows_sha256"]),
        per_budget_row_count={budget: int(counts[str(budget)]) for budget in BUDGETS},
        per_budget_rows_sha256={budget: str(digests[str(budget)]) for budget in BUDGETS},
    )
    reconstructed = normalizer.payload()
    reconstructed["body_sha256"] = audit_v1._json_sha(reconstructed)
    _require(reconstructed == payload, "q normalizer payload drift")
    return normalizer, predecessor


def posterior_side_for_inputs(
    inputs,
    budget: int,
    *,
    prior,
    normalizer,
    q_normalizer: ReliabilityNormalizer,
) -> dict[str, object]:
    """Build the deployable posterior side on the exact selected support."""

    _require(budget in BUDGETS, "unsupported posterior budget")
    selected = np.ascontiguousarray(np.asarray(inputs.selected_by_budget[budget], dtype=np.int64))
    _require(selected.shape == (budget,), "selected support cardinality drift")
    theta = np.ascontiguousarray(np.asarray(inputs.theta[selected], dtype=np.float64))
    rates = np.ascontiguousarray(np.asarray(inputs.rates[selected], dtype=np.float64).T)
    _require(rates.shape == (inputs.n_units, budget), "posterior rate geometry drift")
    fit = fit_posterior_mean(rates, theta, prior_variance=prior.variance)
    side4 = np.ascontiguousarray(normalizer.normalize(fit.raw_t4), dtype=np.float32)
    q_raw = np.ascontiguousarray(angular_reliability(fit), dtype=np.float64)
    q = q_normalizer.normalize(q_raw)
    _require(side4.shape == (inputs.n_units, 4), "posterior side4 geometry drift")
    _require(q.shape == (inputs.n_units,), "posterior q geometry drift")
    return {
        "selected": selected,
        "fit": fit,
        "side4": side4,
        "side4_sha256": array_sha256(side4),
        "q_raw": q_raw,
        "q_raw_sha256": array_sha256(q_raw),
        "q": q,
        "q_sha256": array_sha256(q),
    }


def q_shuffle_permutation(surface: str, session: str, budget: int, units: int) -> np.ndarray:
    _require(surface in SURFACES and budget in BUDGETS, "shuffle cell drift")
    _require(type(units) is int and units > 1, "shuffle requires at least two units")
    domain = f"{SHUFFLE_DOMAIN}|{BOOTSTRAP_SEED}|{surface}|{session}|M{budget}"
    seed = int.from_bytes(hashlib.sha256(domain.encode("utf-8")).digest()[:8], "little")
    permutation = np.random.default_rng(seed).permutation(units).astype(np.int64)
    if np.array_equal(permutation, np.arange(units, dtype=np.int64)):
        permutation = np.roll(permutation, 1)
    _require(np.array_equal(np.sort(permutation), np.arange(units)), "q permutation topology drift")
    _require(not np.array_equal(permutation, np.arange(units)), "q permutation is identity")
    return np.ascontiguousarray(permutation)


def _state_sha(arm_common, model) -> str:
    return str(arm_common.state_sha256(model))


def _load_models(repository_root: Path, producer: Mapping[str, Mapping[str, object]], stack):
    import torch
    # ``src`` is intentionally owned by ``tfpd_exploration`` in this scorer
    # process.  The streaming model tree is exposed independently by adding
    # ``streaming_calibration_exp/src`` to sys.path, so importing it through
    # ``src.models`` creates a namespace collision before any model is loaded.
    from models.components.streaming_encoders import build_encoder

    pop_robust = stack["pop_robust"]
    arm_common = stack["arm_common"]
    models = {}
    bindings = {}
    for arm in ("c2", "c3_constant", "c3_real"):
        info = producer[arm]
        payload = torch.load(repository_root / str(info["swa_relative"]), map_location="cpu", weights_only=False)
        model = pop_robust.build_population_robustness_model(seed=42, cell="D")
        if arm.startswith("c3_"):
            widen_cell_d_for_reliability(model, build_encoder)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        state = _state_sha(arm_common, model)
        _require(state == info["swa_state_dict_sha256"], f"{arm} strict state digest drift")
        q_weight_nonzero = None
        if arm.startswith("c3_"):
            q_weight = model.id_encoder.post_pool[0].weight[:, -1]
            q_weight_nonzero = int(torch.count_nonzero(q_weight).item())
            if arm == "c3_constant":
                _require(q_weight_nonzero == 0, "C3-Const learned a nonzero q column")
            else:
                _require(q_weight_nonzero > 0, "C3-Real q column is unused/zero")
        models[arm] = model
        bindings[arm] = {
            "swa_relative": info["swa_relative"],
            "swa_sha256": info["swa_sha256"],
            "state_dict_sha256": state,
            "strict_load": True,
            "eval_mode": not model.training,
            "q_weight_nonzero": q_weight_nonzero,
        }
    return models, bindings


def _decode(runtime, inputs, model, activity, side):
    from src.calibration_gap_v1 import p4_stream_stats as p4

    original = runtime._model
    runtime._model = model
    try:
        return p4._decode_static(runtime, inputs, activity, side)
    finally:
        runtime._model = original


def _baseline_table(terminal: Mapping[str, object]) -> dict[tuple[str, str, int], Mapping[str, object]]:
    table: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for arm in COMPARATORS:
        for row in terminal["arms"][arm]["rows"]:
            key = (arm, str(row["surface"]), int(row["budget"]))
            # Session is part of the key below; keep a second map representation.
            table[(arm, str(row["session"]), int(row["budget"]))] = row
    _require(len(table) == 126, "baseline row table cardinality drift")
    return table


def _assert_baseline_input(row: Mapping[str, object], runtime, inputs, budget: int) -> None:
    from src.cal_aug_v1.deployment import calibration_prefix_digest

    _require(row["surface"] == inputs.surface and row["session"] == inputs.session,
             "baseline cell identity drift")
    _require(int(row["budget"]) == budget, "baseline budget drift")
    _require(int(row["n_windows"]) == inputs.n_windows, "baseline window count drift")
    _require(int(row["n_valid_last_bin"]) == int(np.asarray(inputs.last_valid_mask).sum()),
             "baseline valid-row count drift")
    _require(row["target_sha256"] == inputs.target_sha256, "baseline target digest drift")
    _require(row["selected_indices_sha256"] == inputs.selected_sha_by_budget[budget],
             "baseline selected-support digest drift")
    _require(row["calibration_prefix_sha256"] == calibration_prefix_digest(runtime, inputs, budget),
             "baseline calibration-prefix digest drift")


def score_matched_cells(repository_root: Path, *, producer, baseline, source, q_normalizer):
    import torch
    from src.cal_aug_v1 import receipts as cal_receipts
    from src.calibration_gap_v1 import p4_stream_stats as p4
    from src.calibration_gap_v1 import z1_oracle_cells as z1

    cal_receipts.verify_sealed_predecessors(repository_root)
    stack = cal_receipts.load_sealed_runner_stack(repository_root, repository_root / "tfpd_exploration")
    pop_robust = stack["pop_robust"]
    matched_scorer = stack["matched_scorer"]
    arm_common = stack["arm_common"]
    models, bindings = _load_models(repository_root, producer, stack)
    state_before = {arm: _state_sha(arm_common, model) for arm, model in models.items()}
    prior = source_prior_from_payload(source["authority"]["source_prior"])
    posterior_normalizer = posterior_normalizer_from_payload(
        source["authority"]["posterior_normalizer"]
    )
    baseline_table = _baseline_table(baseline["terminal"])
    rows: list[dict[str, object]] = []
    materialized: list[dict[str, object]] = []
    runtime = z1.HonestOracleRuntime(root=repository_root)
    try:
        for surface in SURFACES:
            roster = runtime.within_roster if surface == "within" else runtime.external_roster
            for session in roster:
                inputs = p4.materialize_session(runtime, surface, session)
                materialized.append({
                    "surface": surface,
                    "session": session,
                    "n_windows": inputs.n_windows,
                    "n_units": inputs.n_units,
                    "target_sha256": inputs.target_sha256,
                    "calibration_m30_sha256": inputs.calibration_m30_sha256,
                })
                for budget in BUDGETS:
                    for comparator in COMPARATORS:
                        _assert_baseline_input(
                            baseline_table[(comparator, session, budget)], runtime, inputs, budget
                        )
                    feature = posterior_side_for_inputs(
                        inputs,
                        budget,
                        prior=prior,
                        normalizer=posterior_normalizer,
                        q_normalizer=q_normalizer,
                    )
                    selected = feature["selected"]
                    activity = inputs.calib[list(selected)]
                    side4 = torch.from_numpy(feature["side4"]).unsqueeze(0)
                    q = np.asarray(feature["q"], dtype=np.float32)
                    q_zero = np.zeros_like(q, dtype=np.float32)
                    permutation = q_shuffle_permutation(surface, session, budget, inputs.n_units)
                    q_shuffled = np.ascontiguousarray(q[permutation], dtype=np.float32)
                    side5 = {
                        "c3_constant": torch.from_numpy(
                            np.ascontiguousarray(np.concatenate((feature["side4"], q_zero[:, None]), axis=1))
                        ).unsqueeze(0),
                        "c3_real": torch.from_numpy(
                            np.ascontiguousarray(np.concatenate((feature["side4"], q[:, None]), axis=1))
                        ).unsqueeze(0),
                        "c3_real_q_shuffle": torch.from_numpy(
                            np.ascontiguousarray(np.concatenate((feature["side4"], q_shuffled[:, None]), axis=1))
                        ).unsqueeze(0),
                    }
                    cell_sides = {"c2": side4, **side5}
                    consumed_q = {
                        "c2": None,
                        "c3_constant": q_zero,
                        "c3_real": q,
                        "c3_real_q_shuffle": q_shuffled,
                    }
                    for arm in NUMERICAL_ARMS:
                        model_key = "c3_real" if arm == "c3_real_q_shuffle" else arm
                        started = time.perf_counter()
                        with pop_robust.dynamic_dropout_recorder() as recorder:
                            prediction, _identities = _decode(
                                runtime, inputs, models[model_key], activity, cell_sides[arm]
                            )
                        _require(recorder["uniform_calls"] == 0 and not recorder["dropout_calls"],
                                 f"eval dropout active: {arm} {surface} {session} M{budget}")
                        last = prediction[:, 49, :].contiguous()
                        target = torch.from_numpy(inputs.last_targets.copy())
                        valid = torch.from_numpy(inputs.last_valid_mask.copy())
                        _require(bool(valid.any().item()), "no valid last-bin target rows")
                        r2 = matched_scorer.session_r2(last[valid], target[valid])
                        row = {
                            "arm": arm,
                            "surface": surface,
                            "session": session,
                            "budget": budget,
                            "r2": float(r2),
                            "n_windows": inputs.n_windows,
                            "n_valid_last_bin": int(valid.sum().item()),
                            "prediction_sha256": hashlib.sha256(
                                prediction.detach().contiguous().numpy().tobytes()
                            ).hexdigest(),
                            "target_sha256": inputs.target_sha256,
                            "selected_indices_sha256": inputs.selected_sha_by_budget[budget],
                            "calibration_prefix_sha256": __import__(
                                "src.cal_aug_v1.deployment", fromlist=["calibration_prefix_digest"]
                            ).calibration_prefix_digest(runtime, inputs, budget),
                            "posterior_raw_t4_sha256": feature["fit"].raw_t4_sha256,
                            "posterior_normalized_side4_sha256": feature["side4_sha256"],
                            "q_raw_sha256": feature["q_raw_sha256"],
                            "q_normalized_sha256": feature["q_sha256"],
                            "consumed_q_sha256": (
                                array_sha256(consumed_q[arm])
                                if consumed_q[arm] is not None else None
                            ),
                            "consumed_side_sha256": array_sha256(
                                cell_sides[arm].squeeze(0).detach().numpy()
                            ),
                            "q_permutation_sha256": (
                                array_sha256(permutation) if arm == "c3_real_q_shuffle" else None
                            ),
                            "q_permutation_fixed_points": (
                                int(np.sum(permutation == np.arange(inputs.n_units)))
                                if arm == "c3_real_q_shuffle" else None
                            ),
                            "wall_seconds": time.perf_counter() - started,
                            "target_optimizer_steps": 0,
                            "target_backward_calls": 0,
                            "target_update_calls": 0,
                        }
                        rows.append(row)
    finally:
        runtime.close()
    state_after = {arm: _state_sha(arm_common, model) for arm, model in models.items()}
    _require(state_after == state_before, "scoring changed model state")
    _require(len(rows) == 252 and len(materialized) == 21, "score matrix cardinality drift")
    return {
        "bindings": bindings,
        "state_before": state_before,
        "state_after": state_after,
        "state_unchanged": True,
        "materialized_sessions": materialized,
        "rows": rows,
        "target_optimizer_backward_update": 0,
    }


def _rows_by_arm(rows: Sequence[Mapping[str, object]], baseline_terminal) -> dict[str, list[Mapping[str, object]]]:
    table = {arm: [] for arm in (*COMPARATORS, *NUMERICAL_ARMS)}
    table["t0"] = list(baseline_terminal["arms"]["t0"]["rows"])
    table["c1"] = list(baseline_terminal["arms"]["c1"]["rows"])
    for row in rows:
        table[str(row["arm"])].append(row)
    for arm, arm_rows in table.items():
        _require(len(arm_rows) == 63, f"{arm} score row count drift")
    return table


def paired_summary(
    rows: Sequence[Mapping[str, object]],
    baseline_terminal: Mapping[str, object],
    *,
    matched_scorer,
) -> dict[str, object]:
    by_arm = _rows_by_arm(rows, baseline_terminal)
    comparisons = (
        ("c2_minus_c1", "c1", "c2"),
        ("c2_minus_t0", "t0", "c2"),
        ("constant_minus_c2", "c2", "c3_constant"),
        ("real_minus_c2", "c2", "c3_real"),
        ("real_minus_constant", "c3_constant", "c3_real"),
        ("real_minus_q_shuffle", "c3_real_q_shuffle", "c3_real"),
    )
    result: dict[str, object] = {}
    for label, reference, candidate in comparisons:
        result[label] = {}
        for surface in SURFACES:
            for budget in BUDGETS:
                ref = [r for r in by_arm[reference] if r["surface"] == surface and int(r["budget"]) == budget]
                cand = [r for r in by_arm[candidate] if r["surface"] == surface and int(r["budget"]) == budget]
                _require([r["session"] for r in ref] == [r["session"] for r in cand],
                         f"paired roster drift: {label} {surface} M{budget}")
                deltas = [float(c["r2"]) - float(r["r2"]) for r, c in zip(ref, cand, strict=True)]
                stats = matched_scorer.paired_session_stats(deltas, seed=BOOTSTRAP_SEED)
                key = f"{surface}:m{budget}"
                result[label][key] = {
                    "surface": surface,
                    "budget": budget,
                    "reference": reference,
                    "candidate": candidate,
                    "n_sessions": len(deltas),
                    "reference_equal_session_mean_r2": float(np.mean([float(r["r2"]) for r in ref])),
                    "candidate_equal_session_mean_r2": float(np.mean([float(r["r2"]) for r in cand])),
                    "equal_session_mean_delta": float(np.mean(deltas)),
                    "worst_session_reference_r2": min(float(r["r2"]) for r in ref),
                    "worst_session_candidate_r2": min(float(r["r2"]) for r in cand),
                    "worst_session_delta": (
                        min(float(r["r2"]) for r in cand)
                        - min(float(r["r2"]) for r in ref)
                    ),
                    "positive_sessions": int(stats["n_positive"]),
                    "bootstrap_95_interval": list(stats["bootstrap_95_interval"]),
                    "per_session": [
                        {
                            "session": r["session"],
                            "reference_r2": float(r["r2"]),
                            "candidate_r2": float(c["r2"]),
                            "delta_r2": float(c["r2"]) - float(r["r2"]),
                        }
                        for r, c in zip(ref, cand, strict=True)
                    ],
                }
    return result


def decision_readout(summary: Mapping[str, object]) -> dict[str, object]:
    c2 = summary["c2_minus_c1"]
    e03_c2 = summary["real_minus_c2"]
    e03_const = summary["real_minus_constant"]
    e03_shuffle = summary["real_minus_q_shuffle"]
    numerical_lead = [
        budget for budget in (4, 10)
        if c2[f"external:m{budget}"]["equal_session_mean_delta"] >= 0.02
        and c2[f"external:m{budget}"]["positive_sessions"] >= 10
    ]
    lead = [
        budget for budget in numerical_lead
        if c2[f"external:m{budget}"]["worst_session_delta"]
        >= WORST_SESSION_SAFETY_DELTA
    ]
    c2_safety = bool(
        c2["external:m30"]["equal_session_mean_delta"] >= -0.01
        and c2["within:m30"]["equal_session_mean_delta"] >= -0.01
    )
    e03_budgets = []
    for budget in (4, 10):
        key = f"external:m{budget}"
        if (
            e03_c2[key]["equal_session_mean_delta"] > 0.0
            and e03_const[key]["equal_session_mean_delta"] > 0.0
            and e03_shuffle[key]["equal_session_mean_delta"] > 0.0
            and e03_c2[key]["worst_session_delta"] >= WORST_SESSION_SAFETY_DELTA
            and e03_const[key]["worst_session_delta"] >= WORST_SESSION_SAFETY_DELTA
        ):
            e03_budgets.append(budget)
    e03_safety = bool(
        e03_c2["external:m30"]["equal_session_mean_delta"] >= -0.01
        and e03_const["external:m30"]["equal_session_mean_delta"] >= -0.01
    )
    return {
        "c2_promotion": {
            "numerical_lead_external_budgets_before_tail_safety": numerical_lead,
            "lead_external_budgets": lead,
            "m30_safety_passed": c2_safety,
            "passed": bool(lead and c2_safety),
            "rule": (
                "external M4 or M10 delta_vs_C1>=0.02 and positives>=10/15; "
                "worst-session delta>=-0.02; external+within M30>=-0.01"
            ),
            "worst_session_safety_delta": WORST_SESSION_SAFETY_DELTA,
        },
        "e03_performance_identification": {
            "identified_external_budgets": e03_budgets,
            "m30_safety_passed": e03_safety,
            "passed": bool(e03_budgets and e03_safety),
            "rule": (
                "Real>C2, Real>Const, and Real>same-checkpoint-q-shuffle on M4 or M10; "
                "worst-session delta vs C2/Const>=-0.02; M30 safe"
            ),
            "worst_session_safety_delta": WORST_SESSION_SAFETY_DELTA,
            "source_q_error_correlation_evaluated": False,
            "claim_complete": False,
            "claim_incomplete_reason": "source held-out q/error correlation is a separate required mechanism readout",
        },
        "old_c2_ridge_input_score_disposition": "DIAGNOSTIC_ONLY__NOT_E02_POSTERIOR_INPUT",
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "score requires empty CUDA_VISIBLE_DEVICES")
    source = c2_smoke.validate_source_authority(repository_root)
    q_normalizer, smoke = reliability_normalizer_from_smoke(repository_root)
    producer = {
        "c2": validate_c2_producer(repository_root),
        "c3_constant": validate_c3_producer(repository_root, "constant"),
        "c3_real": validate_c3_producer(repository_root, "real"),
    }
    baseline = validate_baseline_score(repository_root)
    strict = execution_closure(repository_root)
    review = review_closure(repository_root)
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "score root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_authority": {
            key: source[key] for key in (
                "root_relative", "attempt_sha256", "source_authority_sha256",
                "terminal_sha256", "closure_sha256",
            )
        },
        "smoke_predecessor": {
            key: smoke[key] for key in (
                "root_relative", "attempt_sha256", "terminal_sha256", "execution_closure_sha256"
            )
        },
        "producers": producer,
        "baseline_score": {key: baseline[key] for key in (
            "root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count"
        )},
        "execution_closure": strict,
        "review_closure": review,
        "surfaces": list(SURFACES),
        "budgets": list(BUDGETS),
        "numerical_arms": list(NUMERICAL_ARMS),
        "posterior_input_required": True,
        "old_ridge_input_score_is_diagnostic_only": True,
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    stage = "post_attempt"
    try:
        stage = "post_attempt_checkpoint_validation"
        producer_opened = {
            "c2": validate_c2_producer(repository_root, open_checkpoint_body=True),
            "c3_constant": validate_c3_producer(
                repository_root, "constant", open_checkpoint_body=True
            ),
            "c3_real": validate_c3_producer(
                repository_root, "real", open_checkpoint_body=True
            ),
        }
        _require(producer_opened == producer, "producer checkpoint graph drift")
        stage = "matched_scoring"
        scored = score_matched_cells(
            repository_root,
            producer=producer,
            baseline=baseline,
            source=source,
            q_normalizer=q_normalizer,
        )
        import sys
        matched_scorer = sys.modules["tfpd_lane_matched_scorer"]
        summary = paired_summary(scored["rows"], baseline["terminal"], matched_scorer=matched_scorer)
        decision = decision_readout(summary)
        stage = "final_revalidation"
        source_final = c2_smoke.validate_source_authority(repository_root)
        _require(source_final == source, "source authority changed during score")
        producer_final = {
            "c2": validate_c2_producer(repository_root, open_checkpoint_body=True),
            "c3_constant": validate_c3_producer(
                repository_root, "constant", open_checkpoint_body=True
            ),
            "c3_real": validate_c3_producer(
                repository_root, "real", open_checkpoint_body=True
            ),
        }
        _require(producer_final == producer, "producer changed during score")
        baseline_final = validate_baseline_score(repository_root)
        _require(
            {key: baseline_final[key] for key in ("root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count")}
            == {key: baseline[key] for key in ("root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count")},
            "baseline score changed during successor score",
        )
        strict_final = execution_closure(repository_root)
        _require(strict_final["closure_sha256"] == strict["closure_sha256"],
                 "numerical execution closure drift")
        review_final = review_closure(repository_root)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "C2_C3_POSTERIOR_MATCHED_SCORE_TERMINAL",
            "attempt_sha256": attempt_sha,
            "producer": producer,
            "baseline_score": attempt["baseline_score"],
            "execution_closure": {"launch": strict, "final": strict_final, "equal": True},
            "review_closure": review_drift(review, review_final),
            "q_normalizer": smoke["terminal"]["q_normalizer"],
            "posterior_input_semantics": {
                "support": "exact frozen deployment selected rows; D-opt M4, chronological M10/M30",
                "carrier": "source-prior posterior mean fitted on the same selected support",
                "normalizer": "frozen source-only equal-budget posterior normalizer",
                "q": "frozen source-only q normalizer; no target refit",
                "old_c2_ridge_input_score": "diagnostic_only",
            },
            "scored": scored,
            "paired_summary": summary,
            "decision_readout": decision,
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = audit_v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": terminal["status"],
            "terminal_sha256": terminal_sha,
            "decision_readout": decision,
            "review_disposition": terminal["review_closure"]["status"],
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_C3_POSTERIOR_MATCHED_SCORE_FAILED",
            "attempt_sha256": attempt_sha,
            "stage": stage,
            "error_class": type(error).__name__,
            "error_sha256": audit_v1._sha_bytes(f"{type(error).__name__}: {error}".encode()),
            "traceback": traceback.format_exc(),
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_MODEL_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "surfaces": list(SURFACES),
        "budgets": list(BUDGETS),
        "numerical_arms": list(NUMERICAL_ARMS),
        "baseline_comparators": list(COMPARATORS),
        "target_materialization_count": 21,
        "posterior_input_required": True,
        "old_c2_ridge_input_score_is_diagnostic_only": True,
        "review_drift_policy": "ACCEPTED_NON_NUMERIC_DRIFT",
    }
