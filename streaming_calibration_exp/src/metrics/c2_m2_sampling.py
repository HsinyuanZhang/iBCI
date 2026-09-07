"""Pure-Python contract utilities for C2 equal-session M2 sampling.

This module performs no Torch, CUDA, NWB, model, or training import.  It
freezes the 2x2x3x3 lattice, evaluates the pre-registered gate, and refuses
sealed sub-C formal-test sessions.  The sampling lever is the existing
``SessionBatchSampler.balance_sessions`` switch; C2 does not invent a sampler.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCREEN_ID = "m2_sampling_objective_c2_v1"
SCHEMA_VERSION = 1
SAMPLINGS = ("legacy", "equal_session")
CARRIERS = ("t4", "z4")
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2)
EPOCH_WINDOW = tuple(range(5, 13))
LOSS_MODE = "task_plus_y_plus_E"
LAMBDA_Y = 1.0
LAMBDA_E = 0.1
PRACTICAL_T4_DELTA = 0.03
PRACTICAL_INTERACTION = 0.03
CROSSED_BOOTSTRAP_SEED = 20260813
CROSSED_BOOTSTRAP_DRAWS = 5_000

SEALED_FORMAL_TEST_SESSIONS: frozenset[str] = frozenset(
    {
        "sub-C_ses-CO-20151113",
        "sub-C_ses-CO-20151116",
        "sub-C_ses-CO-20151117",
        "sub-C_ses-CO-20151119",
        "sub-C_ses-CO-20151120",
        "sub-C_ses-CO-20151201",
    }
)

EXISTING_LEVER = {
    "file": "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "sampler_class_line": 751,
    "balance_sessions_argument_line": 758,
    "equal_session_interpolation_lines": (854, 904),
    "window_budget_per_session_argument_line": 760,
    "datamodule_config_name": "balance_session_batches",
    "datamodule_argument_line": 1001,
    "train_sampler_pass_through_lines": (1497, 1503),
    "validation_sampler_unbalanced_line": 1513,
    "ordinary_m2_config": "streaming_calibration_exp/configs/data/falcon_m2.yaml",
    "ordinary_m2_config_currently_false": True,
    "window_budget_routed_from_ordinary_m2_datamodule": False,
    "sua_a2_sampler_has_switch": False,
    "spint_main_sampler_has_switch": False,
}

SCIENCE_CONFIG_KEYS = (
    "seed",
    "train",
    "test",
    "ckpt_path",
    "no_early_stopping",
    "require_baseline_validation",
    "data",
    "model",
    "trainer",
    "callbacks",
    "optimizer",
    "scheduler",
    "baseline_metrics_path",
)

BINDING_FILES = (
    "sua_exploration/docs/C2_M2_SAMPLING_OBJECTIVE_CONTRACT_20260813.md",
    "streaming_calibration_exp/src/metrics/c2_m2_sampling.py",
    "streaming_calibration_exp/src/data/c2_m2_matched_z4_datamodule.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "streaming_calibration_exp/configs/data/falcon_m2.yaml",
    "streaming_calibration_exp/configs/data/falcon_m2_c2_matched_z4.yaml",
    "streaming_calibration_exp/configs/callbacks/c2_epoch_window.yaml",
    "streaming_calibration_exp/configs/experiment/c2_v1_m2_t4_legacy.yaml",
    "streaming_calibration_exp/configs/experiment/c2_v1_m2_t4_equal_session.yaml",
    "streaming_calibration_exp/configs/experiment/c2_v1_m2_z4_legacy.yaml",
    "streaming_calibration_exp/configs/experiment/c2_v1_m2_z4_equal_session.yaml",
    "streaming_calibration_exp/configs/model/streaming_b3s_t4.yaml",
    "streaming_calibration_exp/configs/model/_streaming_base.yaml",
    "streaming_calibration_exp/scripts/preflight_c2_m2_sampling.py",
    "streaming_calibration_exp/scripts/run_c2_m2_sampling.py",
    "streaming_calibration_exp/scripts/aggregate_c2_m2_sampling.py",
    "streaming_calibration_exp/tests/test_c2_m2_sampling.py",
    "streaming_calibration_exp/tests/test_falcon_sampler.py",
)


class C2ContractError(ValueError):
    """Raised for an incomplete, contaminated, or incomparable C2 matrix."""


@dataclass(frozen=True)
class CellSpec:
    sampling: str
    carrier: str
    fold: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.sampling}/{self.carrier}/f{self.fold}/s{self.seed}"

    @property
    def balance_session_batches(self) -> bool:
        return self.sampling == "equal_session"


def expected_cells() -> tuple[CellSpec, ...]:
    return tuple(
        CellSpec(sampling, carrier, fold, seed)
        for sampling, carrier, fold, seed in product(SAMPLINGS, CARRIERS, FOLDS, SEEDS)
    )


def expected_config_name(sampling: str, carrier: str) -> str:
    if sampling not in SAMPLINGS or carrier not in CARRIERS:
        raise C2ContractError(f"Bad C2 factor level sampling={sampling} carrier={carrier}")
    return f"c2_v1_m2_{carrier}_{sampling}"


def refuse_sealed_sessions(session_names: Iterable[str], *, label: str = "sessions") -> None:
    blocked = sorted({name for name in session_names if name in SEALED_FORMAL_TEST_SESSIONS})
    if blocked:
        raise C2ContractError(
            f"{label}: refusing sealed sub-C formal-test session(s): {blocked}"
        )


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(payload: Mapping[str, Any] | list[Any]) -> str:
    if isinstance(payload, Mapping):
        body = canonical_json_bytes(payload)
    else:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def science_config_projection(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        raise C2ContractError("Science config must be a mapping")
    missing = [key for key in SCIENCE_CONFIG_KEYS if key not in cfg and key not in {"optimizer", "scheduler"}]
    if missing:
        raise C2ContractError(f"Science config missing required keys: {missing}")
    selected = {key: cfg.get(key) for key in SCIENCE_CONFIG_KEYS}
    try:
        return json.loads(canonical_json_bytes(selected))
    except (TypeError, ValueError) as exc:
        raise C2ContractError("Science config is not canonical JSON") from exc


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    path = path.resolve()
    sidecar = Path(f"{path}.sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"C2 immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload) + b"\n"
    digest = hashlib.sha256(body).hexdigest()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    try:
        fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(f"{digest}  {path.name}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    return digest


def load_verified_immutable_json(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise C2ContractError(f"Missing immutable body/sidecar: {path}")
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    expected = sidecar.read_text(encoding="ascii").strip().split(maxsplit=1)
    if not expected or expected[0] != digest:
        raise C2ContractError(f"Immutable digest mismatch: {path}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise C2ContractError(f"Invalid JSON receipt: {path}") from exc
    if not isinstance(payload, dict):
        raise C2ContractError(f"Receipt root must be an object: {path}")
    return payload, digest


def source_bindings(root: Path) -> dict[str, str]:
    root = root.resolve()
    files: dict[str, str] = {}
    for relative in BINDING_FILES:
        path = root / relative
        if not path.is_file():
            raise C2ContractError(f"C2 binding file missing: {relative}")
        files[relative] = sha256_file(path)
    return files


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise C2ContractError("C2 aggregate requires a nonempty finite score collection")
    return float(array.mean())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2ContractError(message)


def validate_cell_receipt(payload: Mapping[str, Any], spec: CellSpec) -> float:
    _require(payload.get("schema_version") == SCHEMA_VERSION, f"{spec.key}: schema drift")
    _require(payload.get("screen_id") == SCREEN_ID, f"{spec.key}: screen_id drift")
    _require(payload.get("sampling") == spec.sampling, f"{spec.key}: sampling drift")
    _require(payload.get("carrier") == spec.carrier, f"{spec.key}: carrier drift")
    _require(int(payload.get("fold", -1)) == spec.fold, f"{spec.key}: fold drift")
    _require(int(payload.get("seed", -1)) == spec.seed, f"{spec.key}: seed drift")
    _require(payload.get("loss_mode") == LOSS_MODE, f"{spec.key}: loss_mode must remain ordinary inherited MSE")
    _require(payload.get("r2_native_loss") is False, f"{spec.key}: batch-level R2 loss is held and forbidden")
    _require(
        payload.get("balance_session_batches") is spec.balance_session_batches,
        f"{spec.key}: sampler flag must match the sampling arm",
    )
    _require(
        payload.get("window_budget_per_session") is None,
        f"{spec.key}: C2 tests interpolation, not window_budget_per_session",
    )
    _require(payload.get("development_scope_only") is True, f"{spec.key}: development-only flag missing")
    _require(payload.get("formal_test_or_external_heldout_opened") is False, f"{spec.key}: held-out opened")
    _require(payload.get("sealed_formal_test_sessions_opened") is False, f"{spec.key}: sealed sessions opened")
    session_name = payload.get("session_name")
    _require(isinstance(session_name, str) and session_name, f"{spec.key}: session_name missing")
    refuse_sealed_sessions([session_name], label=spec.key)
    epoch_window = payload.get("epoch_window")
    _require(list(epoch_window or []) == list(EPOCH_WINDOW), f"{spec.key}: epoch window drift")
    score = payload.get("score")
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise C2ContractError(f"{spec.key}: non-numeric score") from exc
    _require(np.isfinite(value), f"{spec.key}: non-finite score")
    return value


def _normalize_scores(scores: Mapping[CellSpec, float]) -> dict[CellSpec, float]:
    expected = set(expected_cells())
    observed = set(scores)
    if observed != expected:
        missing = sorted(cell.key for cell in expected - observed)
        extra = sorted(cell.key for cell in observed - expected)
        raise C2ContractError(f"incomplete or extra C2 matrix: missing={missing}, extra={extra}")
    normalized: dict[CellSpec, float] = {}
    for spec, raw in scores.items():
        value = float(raw)
        if not np.isfinite(value):
            raise C2ContractError(f"non-finite C2 score for {spec.key}")
        normalized[spec] = value
    return normalized


def _score_lookup(scores: Mapping[CellSpec, float], sampling: str, carrier: str, fold: int, seed: int) -> float:
    return scores[CellSpec(sampling, carrier, fold, seed)]


def cell_contrasts(scores: Mapping[CellSpec, float]) -> list[dict[str, Any]]:
    values = _normalize_scores(scores)
    rows: list[dict[str, Any]] = []
    for fold, seed in product(FOLDS, SEEDS):
        delta_t4 = _score_lookup(values, "equal_session", "t4", fold, seed) - _score_lookup(
            values, "legacy", "t4", fold, seed
        )
        delta_z4 = _score_lookup(values, "equal_session", "z4", fold, seed) - _score_lookup(
            values, "legacy", "z4", fold, seed
        )
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "delta_t4": float(delta_t4),
                "delta_z4": float(delta_z4),
                "interaction": float(delta_t4 - delta_z4),
            }
        )
    return rows


def evaluate_sampling_gates(
    scores: Mapping[CellSpec, float],
    *,
    bootstrap_draws: int = CROSSED_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = CROSSED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate the frozen C2 gate.  Must be able both to pass and to fail."""
    rows = cell_contrasts(scores)
    session_rows = []
    for fold in FOLDS:
        subset = [row for row in rows if row["fold"] == fold]
        session_rows.append(
            {
                "fold": fold,
                "delta_t4": _mean(row["delta_t4"] for row in subset),
                "delta_z4": _mean(row["delta_z4"] for row in subset),
                "interaction": _mean(row["interaction"] for row in subset),
            }
        )
    seed_rows = []
    for seed in SEEDS:
        subset = [row for row in rows if row["seed"] == seed]
        seed_rows.append(
            {
                "seed": seed,
                "delta_t4": _mean(row["delta_t4"] for row in subset),
                "delta_z4": _mean(row["delta_z4"] for row in subset),
                "interaction": _mean(row["interaction"] for row in subset),
            }
        )
    mean_t4 = _mean(row["delta_t4"] for row in rows)
    mean_z4 = _mean(row["delta_z4"] for row in rows)
    mean_interaction = _mean(row["interaction"] for row in rows)
    gates = {
        "t4_mean_delta_at_least_floor": mean_t4 >= PRACTICAL_T4_DELTA,
        "all_session_t4_means_positive": all(row["delta_t4"] > 0.0 for row in session_rows),
        "all_seed_t4_means_positive": all(row["delta_t4"] > 0.0 for row in seed_rows),
        "interaction_mean_at_least_floor": mean_interaction >= PRACTICAL_INTERACTION,
    }
    t4_claim = (
        gates["t4_mean_delta_at_least_floor"]
        and gates["all_session_t4_means_positive"]
        and gates["all_seed_t4_means_positive"]
    )
    passed = t4_claim and gates["interaction_mean_at_least_floor"]
    if passed:
        classification = "sampling_objective_pass"
    elif not t4_claim:
        classification = "t4_session_mean_r2_lift_below_floor_stop"
    else:
        classification = "generic_z4_lift_or_no_carrier_specificity_stop"
    def _grid(field: str) -> np.ndarray:
        return np.asarray(
            [
                [
                    next(row[field] for row in rows if row["fold"] == fold and row["seed"] == seed)
                    for fold in FOLDS
                ]
                for seed in SEEDS
            ],
            dtype=np.float64,
        )

    delta_grid = _grid("delta_t4")
    interaction_grid = _grid("interaction")
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "primary_estimand": "unweighted session-mean R2: equal_session(T4) - legacy(T4)",
        "anti_generic_estimand": "(delta_T4) - (delta_Z4)",
        "r2_native_loss": False,
        "wilcoxon_computed": False,
        "wilcoxon_reason": "exact two-sided Wilcoxon/sign tests are unattainable at n=3 sessions or n=3 seeds",
        "mean_t4_delta": mean_t4,
        "mean_z4_delta": mean_z4,
        "mean_interaction": mean_interaction,
        "per_seed_session": rows,
        "per_session": session_rows,
        "per_seed": seed_rows,
        "gates": gates,
        "passed": passed,
        "classification": classification,
        "bootstrap_interval_descriptive_only": True,
        "t4_delta_bootstrap_ci": _crossed_bootstrap_ci(
            delta_grid, draws=bootstrap_draws, seed=bootstrap_seed
        ),
        "interaction_bootstrap_ci": _crossed_bootstrap_ci(
            interaction_grid, draws=bootstrap_draws, seed=bootstrap_seed
        ),
        "existing_lever": dict(EXISTING_LEVER),
        "development_only": True,
        "formal_test": False,
        "gpu_authorized": False,
        "cell_count": len(expected_cells()),
    }


def _crossed_bootstrap_ci(grid: np.ndarray, *, draws: int, seed: int) -> list[float]:
    if draws <= 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n_seed, n_session = grid.shape
    sampled = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        seeds = rng.integers(0, n_seed, n_seed)
        sessions = rng.integers(0, n_session, n_session)
        sampled[index] = grid[np.ix_(seeds, sessions)].mean()
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def aggregate_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    official_preflight_sha256: str,
    bootstrap_draws: int = CROSSED_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = CROSSED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    expected = {cell.key: cell for cell in expected_cells()}
    if set(receipts) != set(expected):
        missing = sorted(set(expected) - set(receipts))
        extra = sorted(set(receipts) - set(expected))
        raise C2ContractError(f"incomplete or extra C2 receipts: missing={missing}, extra={extra}")
    scores: dict[CellSpec, float] = {}
    session_names: list[str] = []
    for key, spec in expected.items():
        payload = receipts[key]
        if payload.get("official_preflight_sha256") != official_preflight_sha256:
            raise C2ContractError(f"{key}: receipt does not bind this official preflight")
        scores[spec] = validate_cell_receipt(payload, spec)
        session_names.append(str(payload["session_name"]))
    refuse_sealed_sessions(session_names, label="aggregate")
    result = evaluate_sampling_gates(
        scores, bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed
    )
    result["official_preflight_sha256"] = official_preflight_sha256
    result["expected_cell_keys"] = [cell.key for cell in expected_cells()]
    return result


def synthetic_score_matrix(
    *,
    t4_delta: float,
    z4_delta: float,
    legacy_t4: float = 0.40,
    legacy_z4: float = 0.30,
) -> dict[CellSpec, float]:
    """Uniform synthetic lattice used to prove the frozen gate can pass and fail."""
    scores: dict[CellSpec, float] = {}
    for cell in expected_cells():
        if cell.carrier == "t4":
            base = legacy_t4
            delta = t4_delta
        else:
            base = legacy_z4
            delta = z4_delta
        scores[cell] = base + (delta if cell.sampling == "equal_session" else 0.0)
    return scores
