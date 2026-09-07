"""Z1 — the honest-M oracle missing cells (IMPLEMENTATION; CPU-only).

Cell matrix: {M4, M10, M30} x {external-15, within-6} x C3-support — SIX
executed cells (the runbook scaffold's "9 cells" over-counts: the support
axis has a single level, C3).  The C3
support (full-session labeled carrier) is the leakage-diagnostic convention of
``low_cost_calibration_diagnostics_v3``; the NEW factor is the B3S calibration
ACTIVITY restricted to the first M trials, i.e. the tensor the sealed Cell-D
id-encoder mean-pools over is ``calib_trials[:M]`` instead of all 30 trials.

Everything else is copied verbatim from the diagnostics-v3 machinery BY IMPORT
(``src/low_cost_calibration_v1.py`` scorer/decomposition + the reviewed
provenance/asset loaders of ``posterior_marginalized_cell_d_v1`` +
``mc_maze`` parsers + ``tfpd_lane`` sealed model/scorer):

- sealed Cell-D SWA ``results/pop_robust_v1/cellD_2heads_dynamic_dropout/
  swa_final4.pt`` loaded weights-only through the SHA-verified provenance
  reader (body SHA 626f65d8...), strict-loaded, eval mode, CPU float32;
- source-only ordinary OLS T4 normalizer literals (semantic SHA 293b8a55...)
  and the f062506c behavior normalizer, both via the reviewed helpers;
- identical query inputs (windows exclude the 30-trial calibration prefix),
  fixed governed bin 49, variance-weighted equal-session R2 via
  ``tfpd_lane.matched_scorer.session_r2``;
- zero target optimizer/backward/update, no training, formal data unopened.

Two anchors make the reproduction auditable without a GPU re-run:
1. ``normalized_side_sha256`` — the C3 carrier (bit-exact, pure numpy);
2. the M30 cell must reproduce the recorded diagnostics-v3 C3 row to
   ``ANCHOR_TOLERANCE`` (measured CPU-vs-GPU float drift 1.2e-7).

Verification anchors are checked per session; a failure fails the whole run.

Pre-registered readings (handoff §2 Z1 / oracle_cells runbook):
  honest_M4_oracle ~= 0.4449 (the M30-activity C3 rung)  -> the carrier term
  is REAL at M4 activity; P1 aims correctly.  honest_M4_oracle ~= 0.25 ->
  the activity pathway breaks at M4; P4 becomes the main line.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
# The reviewed provenance/asset/data loaders address artifacts as
# "tfpd_exploration/..." relative to the REPOSITORY root (the diagnostics-v3
# convention), while the ledger receipts are relative to the tfpd root.
REPO_ROOT = ROOT.parent

BUDGETS = (4, 10, 30)
SURFACES = ("external", "within")
SUPPORT = "C3_full_session_oracle"
DIAGNOSTICS_RECEIPT = "results/low_cost_calibration_diagnostics_v3/receipt.json"
# CPU float32 forward vs the recorded GPU float32 forward: measured |dR2| <=
# 1.2e-7 on the anchor sessions; the tolerance leaves an order of magnitude of
# slack while still failing on any real pipeline divergence.
ANCHOR_TOLERANCE = 1.0e-5
DECODE_CHUNK = 32  # the diagnostics-v3 forward batch, kept identical
SURFACE_ENV = {"within": "SUBC_DATA_ROOT", "external": "SUBM_DATA_ROOT"}
SURFACE_ROOT_RELATIVE = {
    "within": "sua_exploration/data/dandi_000688/sub-C",
    "external": "sua_exploration/data/dandi_000688/sub-M",
}
# Priority order for the interim-receipt contingency (handoff Z1 runtime
# discipline): M4 first, M10, then the M30 anchor cell.
PRIORITY_BUDGETS = (4, 10, 30)


class Z1Error(RuntimeError):
    """Raised on any Z1 boundary, anchor, or topology drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Z1Error(message)


def cell_matrix(budgets: Sequence[int] = BUDGETS, surfaces: Sequence[str] = SURFACES) -> list[dict]:
    """The exact cell descriptors this module runs (9 with the defaults)."""
    return [
        {
            "budget": int(budget),
            "surface": str(surface),
            "support": SUPPORT,
            "activity": f"b3s_first_{int(budget)}_trials",
            "estimator": "ordinary_ols_point_t4",
            "leakage_diagnostic": True,
        }
        for budget in budgets
        for surface in surfaces
    ]


@dataclass(frozen=True)
class SessionInputs:
    """One materialized evaluation session, shared across the M budgets."""

    surface: str
    session: str
    n_units: int
    n_trials: int
    n_windows: int
    neural_sha256: str
    calibration_m30_sha256: str
    target_sha256: str
    valid_mask_sha256: str
    side_sha256: str
    selected_indices_sha256: str
    side: Any  # torch float32 [1, N, 4]
    calib: Any  # torch float32 [30, 100, N]
    neural: Any  # numpy float32 [T, N]
    starts: Any  # numpy int64 [W]
    last_targets: Any  # numpy float32 [W, 2]
    last_valid_mask: Any  # numpy bool [W]


class HonestOracleRuntime:
    """CPU-only frozen Cell-D harness reusing the reviewed loaders verbatim.

    The GPU attestation of ``DefaultPhysicalRuntime`` is intentionally NOT
    reused: this route is CPU-only by handoff.  Every data/weight authority
    still flows through the reviewed provenance/asset readers, and the model
    state digest is compared before/after the whole run.
    """

    def __init__(self, root: Path = REPO_ROOT) -> None:
        """``root`` is the REPOSITORY root (the reviewed loaders' convention)."""
        root = Path(root).absolute()
        self.root = root
        _require(
            os.environ.get("CUDA_VISIBLE_DEVICES", None) == "",
            "Z1 runs CPU-only: CUDA_VISIBLE_DEVICES must be the empty string",
        )
        for extra in (
            root / "tfpd_exploration",
            root / "sua_exploration",
            root / "streaming_calibration_exp",
        ):
            value = str(extra)
            if value not in sys.path:
                sys.path.insert(0, value)
        self._timing: dict[str, float] = {}
        import torch

        if torch.cuda.is_available():
            raise Z1Error("CUDA is visible in a CPU-only Z1 run")
        self._torch = torch
        self.device = torch.device("cpu")

        from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p

        t0 = time.perf_counter()
        self.provenance = v1p.load_verified_provenance(root)
        self._v1p = v1p
        self._timing["provenance_load_s"] = time.perf_counter() - t0
        self.within_roster = tuple(row.session for row in self.provenance.sealed.within)
        self.external_roster = tuple(row.session for row in self.provenance.sealed.external)
        self._reader = v1p._load_exact_module(
            "_low_cost_calibration_asset_reader",
            root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        self.assets = v1p._assets_from_rows(
            self._reader.derive_fixed_input_assets(
                root, within_roster=self.within_roster, external_roster=self.external_roster,
            )
        )
        for surface, variable in SURFACE_ENV.items():
            expected = str(root / SURFACE_ROOT_RELATIVE[surface])
            _require(
                os.environ.get(variable) == expected,
                f"{variable} must be exactly {expected} for the Z1 run",
            )
        self._held_roots = {
            surface: self._reader.HeldDataRoot.from_environment(variable)
            for surface, variable in SURFACE_ENV.items()
        }
        self._equal_score = v1p._load_exact_module(
            "_z1_equal_score", root / "tfpd_exploration/src/cell_d_equal_session_score_v1.py"
        )
        self._model = self._build_model()

    # -- model -------------------------------------------------------------
    def _build_model(self):
        import io

        import torch
        from torch.nn.parameter import UninitializedParameter
        from torch.torch_version import TorchVersion

        from src.tfpd_lane import pop_robust

        t0 = time.perf_counter()
        model = pop_robust.build_population_robustness_model(seed=42, cell="D")
        with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
            payload = torch.load(
                io.BytesIO(self.provenance.sealed_material.swa_body),
                map_location="cpu",
                weights_only=True,
            )
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        from src.posterior_marginalized_cell_d_v1 import plan as v1plan
        from torch.nn.parameter import UninitializedParameter as _Lazy

        initialized = 0
        lazy: list[str] = []
        for name, parameter in model.named_parameters():
            if isinstance(parameter, _Lazy):
                lazy.append(name)
            else:
                initialized += int(parameter.numel())
        _require(
            initialized == v1plan.SEALED_CELL_D_INITIALIZED_PARAMETERS
            and tuple(sorted(lazy)) == tuple(v1plan.SEALED_CELL_D_LAZY_KEYS),
            "sealed Cell-D topology drift in the CPU harness",
        )
        _require(
            not model.training
            and all(parameter.grad is None for parameter in model.parameters()),
            "sealed Cell-D eval/gradient boundary drift in the CPU harness",
        )
        from src.tfpd_lane.arm_common import state_sha256

        self.sealed_state_sha256 = state_sha256(model)
        self._timing["model_build_s"] = time.perf_counter() - t0
        return model

    # -- session materialization -------------------------------------------
    def materialize_session(self, surface: str, session_name: str) -> SessionInputs:
        import numpy as np
        import torch
        from mc_maze.d_optimal_calibration_design import (
            direction_indices_from_thetas,
            fit_carriers_from_selected_trials,
        )
        from mc_maze.multisession_datamodule import (
            list_datamodule_rewarded_trials,
            load_dandi688_session,
        )
        from mc_maze.unit_side_features import _pool_trial_rate_matrix
        from src import low_cost_calibration_v1 as lc
        from src.posterior_marginalized_cell_d_v1 import plan as v1plan

        _require(surface in SURFACES, f"unknown surface {surface!r}")
        matches = [a for a in self.assets[surface] if a.session == session_name]
        _require(len(matches) == 1, f"asset resolution drift for {session_name}")
        asset = matches[0]
        held = self._held_roots[surface].open_asset(
            relative=Path(asset.frozen_path).name,
            expected_bytes=asset.bytes,
            expected_sha256=asset.sha256,
            surface=surface,
            session=session_name,
        )
        snapshot = held.private_snapshot()
        try:
            _t4_mean, _t4_std, behavior_mean, behavior_std = (
                self._equal_score._validate_source_normalizer_numerics(np)
            )
            t0 = time.perf_counter()
            record = load_dandi688_session(
                snapshot.path,
                bin_size_ms=20,
                window_size=50,
                calibration_n_trials=30,
                max_trial_length=100,
                pad_value=-1.0,
                interpolate_trials=True,
                behavior_mean=behavior_mean,
                behavior_std=behavior_std,
                trial_result_filter="R",
                exclude_calibration_trials_from_windows=True,
                cache_dir=None,
                signal_view="sua",
            )
            trials = list_datamodule_rewarded_trials(
                snapshot.path, bin_size_ms=20, window_size=50, trial_result_filter="R",
            )
            theta = np.asarray([trial["target_dir"] for trial in trials], dtype=np.float64)
            _require(bool(np.isfinite(theta).all()), f"{session_name}: missing target cue")
            directions = direction_indices_from_thetas(theta)
            rates_by_unit, units = _pool_trial_rate_matrix(snapshot.path, trials)
            rates = np.ascontiguousarray(rates_by_unit.T, dtype=np.float64)
            snapshot.reverify()
        finally:
            snapshot.close()
        held.reverify()
        held.close()
        self._timing.setdefault("session_parse_s", 0.0)
        self._timing["session_parse_s"] += time.perf_counter() - t0

        neural = np.ascontiguousarray(record.neural, dtype=np.float32)
        starts = np.ascontiguousarray(record.valid_starts, dtype=np.int64)
        _require(units == neural.shape[1], f"{session_name}: unit-axis drift")
        calib = np.ascontiguousarray(record.calib_trials, dtype=np.float32)
        _require(
            calib.shape == (30, 100, neural.shape[1]),
            f"{session_name}: calibration shape drift",
        )

        # C3 support carrier: fitted from the FULL session (leakage oracle).
        selected = np.arange(theta.size, dtype=np.int64)
        raw_t4 = np.ascontiguousarray(
            fit_carriers_from_selected_trials(rates, directions, selected), dtype=np.float32,
        )
        mean = torch.tensor(v1plan.SEALED_OLS_T4_MEAN_FLOAT32, dtype=torch.float32)
        std = torch.tensor(v1plan.SEALED_OLS_T4_STD_FLOAT32, dtype=torch.float32)
        side = ((torch.as_tensor(raw_t4) - mean) / std).detach().unsqueeze(0)
        _require(
            tuple(side.shape) == (1, neural.shape[1], 4)
            and bool(torch.isfinite(side).all().item()),
            f"{session_name}: normalized C3 carrier shape/nonfinite drift",
        )

        reviewed = self._reader
        last_targets, last_mask, target_sha, mask_sha, _count = (
            reviewed._valid_last_bin_authority(np, behavior=np.ascontiguousarray(record.behavior), starts=starts)
        )
        return SessionInputs(
            surface=surface,
            session=session_name,
            n_units=int(neural.shape[1]),
            n_trials=int(theta.size),
            n_windows=int(starts.size),
            neural_sha256=lc._array_sha(neural),
            calibration_m30_sha256=lc._array_sha(calib),
            target_sha256=str(target_sha),
            valid_mask_sha256=str(mask_sha),
            side_sha256=lc._array_sha(side.numpy()),
            selected_indices_sha256=lc._array_sha(selected),
            side=side,
            calib=torch.from_numpy(calib),
            neural=neural,
            starts=starts,
            last_targets=last_targets,
            last_valid_mask=last_mask,
        )

    # -- forward -----------------------------------------------------------
    def forward_budget(self, inputs: SessionInputs, budget: int) -> dict[str, Any]:
        """C3 carrier + first-``budget``-trials B3S activity, frozen forward."""
        import numpy as np
        import torch
        from src import low_cost_calibration_v1 as lc
        from src.tfpd_lane.matched_scorer import session_r2

        torch = self._torch
        _require(budget in BUDGETS, f"unknown budget {budget}")
        model = self._model
        activity = inputs.calib[:budget]
        _require(tuple(activity.shape) == (budget, 100, inputs.n_units), "activity prefix shape drift")
        t0 = time.perf_counter()
        predictions = []
        with torch.no_grad():
            identity = model.compute_identity(activity.unsqueeze(0), side_features=inputs.side)
            for offset in range(0, inputs.starts.size, DECODE_CHUNK):
                chunk = inputs.starts[offset:offset + DECODE_CHUNK]
                neural = torch.from_numpy(
                    np.stack([inputs.neural[start:start + 50] for start in chunk])
                )
                predictions.append(model.decode_with_identity(neural, identity).detach())
        prediction = torch.cat(predictions, dim=0)
        wall_s = time.perf_counter() - t0
        last_prediction = prediction[:, 49, :].contiguous()
        target = torch.from_numpy(np.ascontiguousarray(inputs.last_targets, dtype=np.float32))
        _require(
            tuple(last_prediction.shape) == (inputs.n_windows, 2)
            and bool(last_prediction.isfinite().all().item()),
            "fixed-bin prediction shape/nonfinite drift",
        )
        manual = lc.score_decomposition(last_prediction.numpy(), target.numpy())
        governing_r2 = session_r2(last_prediction, target)
        manual_float64 = float(manual["variance_weighted_r2"])
        manual["manual_float64_variance_weighted_r2"] = manual_float64
        manual["manual_float64_minus_governing_r2"] = manual_float64 - governing_r2
        manual["variance_weighted_r2"] = governing_r2
        prediction_sha = hashlib.sha256(
            prediction.detach().contiguous().numpy().tobytes()
        ).hexdigest()
        calib_prefix_sha = lc._array_sha(activity.numpy())
        row = {
            "surface": inputs.surface,
            "session": inputs.session,
            "budget": int(budget),
            "support": SUPPORT,
            "activity": f"b3s_first_{int(budget)}_trials",
            "leakage_diagnostic": True,
            "n_windows": inputs.n_windows,
            "n_units": inputs.n_units,
            "n_session_trials": inputs.n_trials,
            "prediction_sha256": prediction_sha,
            "target_sha256": inputs.target_sha256,
            "normalized_side_sha256": inputs.side_sha256,
            "calibration_prefix_sha256": calib_prefix_sha,
            "calibration_m30_sha256": inputs.calibration_m30_sha256,
            "selected_trial_count": inputs.n_trials,
            "wall_seconds": wall_s,
            **manual,
        }
        self._timing.setdefault("forward_s", 0.0)
        self._timing["forward_s"] += wall_s
        return row

    def state_digest(self) -> str:
        from src.tfpd_lane.arm_common import state_sha256

        return state_sha256(self._model)

    def close(self) -> None:
        for root in reversed(tuple(self._held_roots.values())):
            try:
                root.close()
            except Exception:
                pass

    @property
    def timing(self) -> dict[str, float]:
        return dict(self._timing)


# ---------------------------------------------------------------------------
# anchors against the sealed diagnostics-v3 receipt (loaded via the ledger)
# ---------------------------------------------------------------------------


def anchor_table(ledger) -> dict[tuple[str, str], dict]:
    """Per-session diagnostics-v3 C3 rows (the M30-activity oracle ladder rung).

    The diagnostics receipt recorded C3 at budgets {4, 30} with IDENTICAL
    inputs (the C3 carrier is budget-independent and its activity was always
    the M30 prefix); both recorded cells are cross-checked here.
    """
    rel = "results/low_cost_calibration_diagnostics_v3/receipt.json"
    body = ledger.bodies[rel]
    rows: dict[tuple[str, str], dict] = {}
    budget_by_key: dict[tuple[str, str], set] = {}
    for cell in body["cells"]:
        if cell["support"] != SUPPORT:
            continue
        for row in cell["sessions"]:
            key = (cell["surface"], row["session"])
            if key in rows:
                if row["prediction_sha256"] != rows[key]["prediction_sha256"]:
                    raise Z1Error(f"diagnostics C3 rows disagree across budgets: {key}")
                if abs(float(row["variance_weighted_r2"]) - float(rows[key]["variance_weighted_r2"])) > 0.0:
                    raise Z1Error(f"diagnostics C3 R2 rows disagree across budgets: {key}")
            else:
                rows[key] = row
            budget_by_key.setdefault(key, set()).add(cell["budget"])
    _require(
        all(value == {4, 30} for value in budget_by_key.values()),
        "diagnostics C3 anchor topology drift (expected both budget cells)",
    )
    return rows


def check_anchor(rows: Sequence[Mapping], ledger, tolerance: float = ANCHOR_TOLERANCE) -> dict:
    """M30 cell vs the recorded diagnostics-v3 C3 row, per session."""
    anchors = anchor_table(ledger)
    checked = 0
    max_abs = 0.0
    per_session = []
    for row in rows:
        if row["budget"] != 30:
            continue
        key = (row["surface"], row["session"])
        _require(key in anchors, f"missing diagnostics anchor for {key}")
        anchor = anchors[key]
        _require(
            row["normalized_side_sha256"] == anchor["normalized_side_sha256"],
            f"{key}: C3 carrier SHA anchor FAILED (bit-exact carrier expected)",
        )
        _require(
            int(row["n_windows"]) == int(anchor["n_windows"]),
            f"{key}: n_windows anchor failed",
        )
        delta = float(row["variance_weighted_r2"]) - float(anchor["variance_weighted_r2"])
        max_abs = max(max_abs, abs(delta))
        per_session.append({
            "surface": row["surface"],
            "session": row["session"],
            "z1_r2": float(row["variance_weighted_r2"]),
            "diagnostics_c3_r2": float(anchor["variance_weighted_r2"]),
            "delta_r2": delta,
            "carrier_sha_match": True,
            "n_windows_match": True,
        })
        checked += 1
    _require(
        max_abs <= tolerance,
        f"M30/C3 anchor drift {max_abs:.3e} exceeds tolerance {tolerance:.1e}",
    )
    return {
        "anchor": "diagnostics_v3_C3_full_session_oracle_at_M30_activity",
        "tolerance": tolerance,
        "sessions_checked": checked,
        "max_abs_delta_r2": max_abs,
        "per_session": per_session,
    }


# ---------------------------------------------------------------------------
# the budget ladder (every rung LOADED from SHA-verified receipts)
# ---------------------------------------------------------------------------


def budget_ladder(ledger, surface: str, budget: int, honest_oracle_r2: float | None = None) -> dict:
    """One budget's rung ladder for either surface, loaded via the ledger.

    Mirrors ``ledger.GapLedger.ladder_external`` but surface-parameterized and
    extended with the Z1 honest-M oracle rung when its value is supplied.
    """
    rel_diag = "results/low_cost_calibration_diagnostics_v3/receipt.json"
    rel_factorial = "results/calibration_budget_protocol_factorial_v2/receipt.json"
    rel_comparators = "results/calibration_budget_comparators_v1/receipt.json"
    surface = {"external": "external", "within": "within"}[surface]

    def cell(receipt: str, **match):
        return ledger.cell(receipt, surface=surface, **match)["summary"]["equal_session_mean_r2"]

    def best(estimator: str, regime: str, total: bool):
        found = []
        for support in ("doptimal_first30", "chronological"):
            try:
                found.append((cell(rel_factorial, budget=budget, estimator=estimator,
                                   regime=regime, support=support), f"factorial/{support}"))
            except Exception:
                pass
        system = "cell_d_ridge_t4_fixed_0p1" if estimator == "ridge_fixed_0p1" else "cell_d_ols"
        cmp_regime = "total_calibration_limited" if total else "label_limited_m30_activity"
        try:
            found.append((cell(rel_comparators, budget=budget, system=system, regime=cmp_regime),
                          "comparators/chronological"))
        except Exception:
            if total and budget == 30:
                found.append((cell(rel_comparators, budget=budget, system=system,
                                   regime="label_limited_m30_activity"),
                              "comparators/label_limited(total==full@M30)"))
        _require(bool(found), f"no rung resolves at surface={surface} budget={budget}")
        return max(found, key=lambda item: item[0])

    # C0 is DEFINED as the chronological OLS rung (D-opt must not leak in).
    try:
        c0 = cell(rel_factorial, budget=budget, estimator="ols",
                  regime="label_limited_m30_activity", support="chronological")
        c0_source = "factorial/chronological"
    except Exception:
        c0, c0_source = cell(rel_comparators, budget=budget, system="cell_d_ols",
                             regime="label_limited_m30_activity"), "comparators/chronological"
    best_ll, ll_support = best("ridge_fixed_0p1", "label_limited_m30_activity", total=False)
    honest_total, honest_support = best("ridge_fixed_0p1", "total_selected_calibration", total=True)
    try:
        oracle = cell(rel_diag, budget=budget, support=SUPPORT)
        oracle_budget = budget
    except Exception:
        c3 = [c for c in ledger.bodies[rel_diag]["cells"]
              if c["surface"] == surface and c["support"] == SUPPORT]
        _require(bool(c3), f"no C3 oracle rung for surface={surface}")
        oracle_budget = c3[0]["budget"]
        oracle = float(c3[0]["summary"]["equal_session_mean_r2"])
    result = {
        "surface": surface,
        "budget": budget,
        "c0_chronological_ols": float(c0),
        "c0_source": c0_source,
        "best_label_limited_dopt_ridge": float(best_ll),
        "honest_total_calibration_dopt_ridge": float(honest_total),
        "oracle_full_session_labels_m30_activity": float(oracle),
        "support_used": {"label_limited": ll_support, "honest": honest_support},
        "oracle_cell_budget": oracle_budget,
        "carrier_term_m30_activity": float(oracle) - float(best_ll),
        "activity_term": float(best_ll) - float(honest_total),
        "total_gap": float(oracle) - float(honest_total),
        "closed_fraction_from_c0": (float(best_ll) - float(c0)) / (float(oracle) - float(c0)),
    }
    if honest_oracle_r2 is not None:
        result["honest_m_budget_oracle"] = float(honest_oracle_r2)
        result["activity_cost_of_oracle_at_m_budget"] = float(oracle) - float(honest_oracle_r2)
        result["carrier_term_at_m_budget_activity"] = float(honest_oracle_r2) - float(honest_total)
    return result


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def _aggregate(rows: Sequence[Mapping]) -> dict:
    from src import low_cost_calibration_v1 as lc

    return lc.aggregate_session_rows(rows)


def run_z1(
    ledger,
    runtime: HonestOracleRuntime,
    *,
    budgets: Sequence[int] = BUDGETS,
    surfaces: Sequence[str] = SURFACES,
    max_total_seconds: float | None = None,
    on_progress=None,
) -> dict:
    """Run the honest-M oracle cells (session-outer; budgets in priority order).

    Runtime discipline (handoff Z1): if the projected total exceeds
    ``max_total_seconds``, an interim payload is returned after the first
    session with per-budget partial aggregates and the measured throughput.
    """
    started = time.perf_counter()
    budgets = tuple(sorted({int(b) for b in budgets}, key=PRIORITY_BUDGETS.index))
    surfaces = tuple(surfaces)
    all_rows: list[dict] = []
    per_surface_budget: dict[tuple[str, int], list[dict]] = {}
    sessions_done = 0
    interim: dict | None = None
    for surface in surfaces:
        roster = runtime.external_roster if surface == "external" else runtime.within_roster
        for session_name in roster:
            t_session = time.perf_counter()
            inputs = runtime.materialize_session(surface, session_name)
            session_rows = [runtime.forward_budget(inputs, budget) for budget in budgets]
            all_rows.extend(session_rows)
            for row in session_rows:
                per_surface_budget.setdefault((row["surface"], row["budget"]), []).append(row)
            sessions_done += 1
            if on_progress is not None:
                on_progress(inputs, session_rows)
            elapsed = time.perf_counter() - started
            projected = elapsed / max(sessions_done, 1) * sum(
                len(runtime.external_roster if s == "external" else runtime.within_roster)
                for s in surfaces
            )
            if (
                interim is None
                and max_total_seconds is not None
                and projected > max_total_seconds
                and sessions_done == 1
            ):
                interim = {
                    "status": "INTERIM_PARTIAL",
                    "reason": f"projected total {projected:.0f}s exceeds {max_total_seconds:.0f}s",
                    "measured_seconds_first_session": time.perf_counter() - t_session,
                    "projected_total_seconds": projected,
                    "sessions_completed": sessions_done,
                    "rows": [dict(row) for row in all_rows],
                }
    cells = []
    for surface in surfaces:
        for budget in budgets:
            rows = per_surface_budget.get((surface, budget), [])
            if not rows:
                continue
            cells.append({
                "surface": surface,
                "budget": budget,
                "support": SUPPORT,
                "activity": f"b3s_first_{budget}_trials",
                "estimator": "ordinary_ols_point_t4",
                "leakage_diagnostic": True,
                "sessions": rows,
                "summary": _aggregate(rows),
            })
    by_cell = {(c["surface"], c["budget"]): c for c in cells}
    ladders = {}
    for surface in surfaces:
        for budget in budgets:
            cell = by_cell.get((surface, budget))
            honest_r2 = cell["summary"]["equal_session_mean_r2"] if cell else None
            ladders[f"{surface}_M{budget}"] = budget_ladder(ledger, surface, budget, honest_r2)
    anchor = check_anchor(all_rows, ledger) if any(row["budget"] == 30 for row in all_rows) else None
    payload = {
        "schema": "calibration_gap_z1_oracle_cells_v1",
        "status": "COMPLETE",
        "cell_matrix": cell_matrix(budgets, surfaces),
        "cells": cells,
        "ladders": ladders,
        "anchor_check": anchor,
        "interim": interim,
        "throughput": {
            **runtime.timing,
            "wall_seconds_total": time.perf_counter() - started,
            "sessions": sessions_done,
            "forward_rows": len(all_rows),
        },
        "sealed_state_sha256": runtime.sealed_state_sha256,
        "boundaries": {
            "cpu_only": True,
            "cuda_visible_devices": "",
            "zero_target_optimizer_steps": True,
            "zero_target_backward_calls": True,
            "zero_target_update_calls": True,
            "training_authorized": False,
            "formal_opened": False,
            "frozen_sealed_cell_d_swa_strict_loaded": True,
            "sealed_state_unchanged": None,  # filled by the runner after close
            "c3_support_is_labeled_leakage_oracle": True,
            "oracle_cells_are_leakage_diagnostics_never_deployable": True,
            "same_query_inputs_as_diagnostics_v3": True,
            "sealed_ordinary_ols_normalizer": True,
            "new_factor": "b3s_calibration_activity_restricted_to_first_M_trials",
        },
    }
    return payload


def readings(payload: Mapping) -> dict:
    """The pre-registered Z1 reading, computed from the loaded numbers."""
    cell = {(c["surface"], c["budget"]): c["summary"]["equal_session_mean_r2"] for c in payload["cells"]}
    out: dict[str, Any] = {
        "honest_oracle_by_cell": {f"{s}_M{b}": cell.get((s, b)) for s in ("external", "within") for b in (4, 10, 30)},
    }
    m4 = cell.get(("external", 4))
    ladder_m30 = payload["ladders"].get("external_M4", {}).get("oracle_full_session_labels_m30_activity")
    if m4 is not None and ladder_m30 is not None:
        drop = ladder_m30 - m4
        if drop <= 0.05:
            verdict = (
                "carrier term is REAL at M4 activity (oracle ceiling survives the "
                "M4 activity restriction) -> P1 justified"
            )
        elif m4 >= 0.35:
            verdict = (
                "intermediate: partial ceiling loss to the M4 activity pathway; "
                "P1 retains a real but smaller carrier term, P4 gains weight"
            )
        else:
            verdict = (
                "activity pathway breaks at M4 (ceiling collapses under M4 "
                "activity even with leaked labels) -> P4 becomes the main line"
            )
        out["external_M4_ceiling_drop_from_m30_activity_oracle"] = drop
        out["pre_registered_reading"] = verdict
    return out
