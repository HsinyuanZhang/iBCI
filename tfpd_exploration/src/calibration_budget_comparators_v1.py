"""Matched M4/M10/M30 comparator primitives and reviewed execution route."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M4_M10_M30_COMPARATORS_20260823.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/calibration_budget_comparators_v1"
SCHEMA = "calibration_budget_comparators_v1"
BUDGETS = (4, 10, 30)
RIDGE_T4_FIXED_LAMBDA = 0.1
RIDGE_T4_GCV_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CLASSICAL_NORMALIZED_LAMBDA = 1.0
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 10_000
EVAL_BATCH_SIZE = 128
EXPECTED_ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


class ComparatorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparatorError(message)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": "), ensure_ascii=True) + "\n").encode()


def compact_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return sha256(compact_json({
        "dtype": str(array.dtype), "shape": list(array.shape),
        "bytes_sha256": sha256(array.tobytes()),
    }))


def fit_ridge_t4(
    rates_trials_units: Any,
    theta: Any,
    *,
    normalized_lambda: float,
) -> tuple[Any, dict[str, object]]:
    """Fit [a,c,b] with ridge on a/c and an unpenalized intercept."""
    import numpy as np

    rates = np.asarray(rates_trials_units, dtype=np.float64)
    angles = np.asarray(theta, dtype=np.float64).reshape(-1)
    require(rates.ndim == 2 and rates.shape[0] == angles.size and angles.size >= 3, "ridge-T4 input shape drift")
    require(np.isfinite(rates).all() and np.isfinite(angles).all(), "ridge-T4 input nonfinite")
    require(math.isfinite(normalized_lambda) and normalized_lambda >= 0.0, "ridge-T4 lambda invalid")
    design = np.column_stack((np.cos(angles), np.sin(angles), np.ones(angles.size)))
    penalty = np.diag((angles.size * normalized_lambda, angles.size * normalized_lambda, 0.0))
    system = design.T @ design + penalty
    try:
        coefficients = np.linalg.solve(system, design.T @ rates)
    except np.linalg.LinAlgError as error:
        raise ComparatorError("ridge-T4 normal equations are singular") from error
    predicted = design @ coefficients
    residual_sum = float(np.square(rates - predicted).sum(dtype=np.float64))
    hat = design @ np.linalg.solve(system, design.T)
    trace_hat = float(np.trace(hat))
    denominator = float((angles.size - trace_hat) ** 2)
    gcv = residual_sum / denominator if denominator > 0.0 else float("inf")
    a, c, b = coefficients
    m = np.sqrt(a * a + c * c)
    raw = np.ascontiguousarray(np.column_stack((a, c, m, b)), dtype=np.float32)
    require(raw.shape == (rates.shape[1], 4) and np.isfinite(raw).all(), "ridge-T4 output drift")
    return raw, {
        "normalized_lambda": float(normalized_lambda),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_condition": float(np.linalg.cond(system)),
        "trace_hat": trace_hat,
        "residual_sum_squares": residual_sum,
        "gcv": float(gcv),
        "raw_t4_sha256": array_sha256(raw),
    }


def fit_gcv_ridge_t4(rates_trials_units: Any, theta: Any) -> tuple[Any, dict[str, object]]:
    candidates: list[tuple[Any, dict[str, object]]] = [
        fit_ridge_t4(rates_trials_units, theta, normalized_lambda=value)
        for value in RIDGE_T4_GCV_GRID
    ]
    selected_index = min(range(len(candidates)), key=lambda index: (float(candidates[index][1]["gcv"]), index))
    raw, evidence = candidates[selected_index]
    return raw, {
        **evidence,
        "selection": "prefix_only_gcv",
        "grid": list(RIDGE_T4_GCV_GRID),
        "selected_index": selected_index,
        "candidate_gcv": [float(item[1]["gcv"]) for item in candidates],
    }


@dataclasses.dataclass(frozen=True)
class LinearReadout:
    mean: Any
    scale: Any
    target_mean: Any
    weights: Any
    normalized_lambda: float
    formulation: str


def fit_closed_form_ridge(features: Any, targets: Any, *, normalized_lambda: float = 1.0) -> LinearReadout:
    """Deterministic primal/dual normalized ridge with unpenalized intercept."""
    import numpy as np

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    require(x.ndim == 2 and y.shape == (x.shape[0], 2) and x.shape[0] >= 3, "ridge design shape drift")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "ridge design nonfinite")
    require(math.isfinite(normalized_lambda) and normalized_lambda > 0.0, "ridge lambda invalid")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    z = (x - mean) / scale
    target_mean = y.mean(axis=0)
    centered = y - target_mean
    n, p = z.shape
    if n <= p:
        system = z @ z.T
        system.flat[:: n + 1] += n * normalized_lambda
        weights = z.T @ np.linalg.solve(system, centered)
        formulation = "dual"
    else:
        system = (z.T @ z) / n
        system.flat[:: p + 1] += normalized_lambda
        weights = np.linalg.solve(system, (z.T @ centered) / n)
        formulation = "primal"
    require(np.isfinite(weights).all(), "ridge weights nonfinite")
    return LinearReadout(
        mean=np.ascontiguousarray(mean, dtype=np.float32),
        scale=np.ascontiguousarray(scale, dtype=np.float32),
        target_mean=np.ascontiguousarray(target_mean, dtype=np.float32),
        weights=np.ascontiguousarray(weights, dtype=np.float32),
        normalized_lambda=float(normalized_lambda),
        formulation=formulation,
    )


def predict_closed_form_ridge(features: Any, readout: LinearReadout) -> Any:
    import numpy as np

    x = np.asarray(features, dtype=np.float32)
    require(x.ndim == 2 and x.shape[1] == readout.mean.size, "ridge prediction shape drift")
    value = ((x - readout.mean) / readout.scale) @ readout.weights + readout.target_mean
    require(np.isfinite(value).all(), "ridge prediction nonfinite")
    return np.ascontiguousarray(value, dtype=np.float32)


def session_r2_float32(prediction: Any, target: Any) -> float:
    """TorchMetrics-compatible variance-weighted two-output R2 in FP32."""
    import numpy as np

    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    require(pred.shape == truth.shape and pred.ndim == 2 and pred.shape[1] == 2, "R2 input shape drift")
    residual = np.sum(np.square(truth - pred, dtype=np.float32), axis=0, dtype=np.float32)
    centered = truth - truth.mean(axis=0, dtype=np.float32)
    total = np.sum(np.square(centered, dtype=np.float32), axis=0, dtype=np.float32)
    require(bool(np.all(total > 0.0)), "R2 target variance is zero")
    return float(np.float32(1.0) - residual.sum(dtype=np.float32) / total.sum(dtype=np.float32))


def bootstrap_delta(deltas: Sequence[float]) -> dict[str, object]:
    import numpy as np

    values = np.asarray(deltas, dtype=np.float64)
    require(values.ndim == 1 and values.size > 0 and np.isfinite(values).all(), "bootstrap delta invalid")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975), method="linear")
    return {
        "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
        "lower_95": float(lower), "upper_95": float(upper),
    }


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_GPU_NO_WRITE_NO_LAUNCH",
        "budgets": list(BUDGETS),
        "regimes": ["label_limited_m30_activity", "total_calibration_limited"],
        "live_neural_rows": [
            "cell_d_ols", "arm_a_ols", "cell_d_ridge_t4_fixed_0p1", "cell_d_ridge_t4_gcv",
        ],
        "classical_rows": ["trial_rate_ridge", "dense_w50_ridge", "population_vector"],
        "historical_anchors": ["original_spint_b0", "a2_t4"],
        "result_root": RESULT_ROOT_RELATIVE,
        "formal_opened": False,
        "target_updates": 0,
    }


def _publish_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = json_bytes(payload)
    digest = sha256(body)
    for leaf, content in ((name, body), (name + ".sha256", f"{digest}  {name}\n".encode("ascii"))):
        descriptor = os.open(
            directory / leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        try:
            offset = 0
            while offset < len(content):
                written = os.write(descriptor, content[offset:])
                require(written > 0, f"short immutable write: {leaf}")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(directory / leaf, 0o444)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def implementation_closure(root: Path) -> dict[str, object]:
    paths = (
        WORKORDER_RELATIVE,
        "tfpd_exploration/src/calibration_budget_comparators_v1.py",
        "tfpd_exploration/scripts/run_calibration_budget_comparators_v1.py",
        "tfpd_exploration/tests/test_calibration_budget_comparators_v1.py",
        "tfpd_exploration/src/low_cost_calibration_v1.py",
        "tfpd_exploration/src/tfpd/spintshape_module.py",
        "sua_exploration/mc_maze/subm_v9_f0_pv_ridge.py",
        "sua_exploration/mc_maze/trial_level_ridge_core.py",
        "sua_exploration/mc_maze/multisession_datamodule.py",
        "sua_exploration/mc_maze/unit_side_features.py",
        "tfpd_exploration/src/posterior_marginalized_cell_d_v4/matched_score.py",
        "tfpd_exploration/src/posterior_marginalized_cell_d_v4/matched_score_physical.py",
    )
    hashes = {relative: sha256((Path(root) / relative).read_bytes()) for relative in paths}
    payload: dict[str, object] = {"schema": SCHEMA + "_closure", "sha256_by_path": hashes}
    payload["closure_sha256"] = sha256(compact_json(payload))
    return payload


def _fit_closed_form_ridge_cuda(features: Any, targets: Any, *, normalized_lambda: float) -> LinearReadout:
    import numpy as np
    import torch

    x = torch.as_tensor(np.ascontiguousarray(features, dtype=np.float32), device="cuda:0")
    y = torch.as_tensor(np.ascontiguousarray(targets, dtype=np.float32), device="cuda:0")
    require(x.ndim == 2 and tuple(y.shape) == (x.shape[0], 2) and x.shape[0] >= 3, "CUDA ridge shape drift")
    with torch.no_grad():
        mean = x.mean(0)
        scale = x.std(0, correction=0)
        scale = torch.where(scale < 1.0e-8, torch.ones_like(scale), scale)
        z = (x - mean) / scale
        target_mean = y.mean(0)
        centered = y - target_mean
        n, p = int(z.shape[0]), int(z.shape[1])
        if n <= p:
            system = z @ z.T
            system.diagonal().add_(n * float(normalized_lambda))
            weights = z.T @ torch.linalg.solve(system, centered)
            formulation = "dual_cuda_fp32"
        else:
            system = (z.T @ z) / float(n)
            system.diagonal().add_(float(normalized_lambda))
            weights = torch.linalg.solve(system, (z.T @ centered) / float(n))
            formulation = "primal_cuda_fp32"
        require(bool(torch.isfinite(weights).all().item()), "CUDA ridge weight nonfinite")
        result = LinearReadout(
            mean=mean.cpu().contiguous().numpy(), scale=scale.cpu().contiguous().numpy(),
            target_mean=target_mean.cpu().contiguous().numpy(),
            weights=weights.cpu().contiguous().numpy(),
            normalized_lambda=float(normalized_lambda), formulation=formulation,
        )
    del x, y, z, centered, system, weights
    torch.cuda.empty_cache()
    return result


def _predict_dense_ridge_query(private: Any, readout: LinearReadout) -> Any:
    import numpy as np
    import torch
    from mc_maze import d_optimal_calibration_design as design_core
    from mc_maze import subm_v9_f0_pv_ridge as numerical

    mean = torch.as_tensor(readout.mean, device="cuda:0")
    scale = torch.as_tensor(readout.scale, device="cuda:0")
    weights = torch.as_tensor(readout.weights, device="cuda:0")
    target_mean = torch.as_tensor(readout.target_mean, device="cuda:0")
    predictions: list[Any] = []
    with torch.no_grad():
        for offset in range(0, int(private.starts.size), 1024):
            starts = private.starts[offset:offset + 1024]
            features = numerical.raw_window_features(private.neural, starts)
            tensor = torch.as_tensor(features, device="cuda:0")
            value = ((tensor - mean) / scale) @ weights + target_mean
            predictions.append(value.cpu().contiguous().numpy())
    result = np.ascontiguousarray(np.concatenate(predictions), dtype=np.float32)
    require(result.shape == (private.starts.size, 2), "dense ridge query prediction shape drift")
    return result


def _trial_bounds(trials: Sequence[Mapping[str, object]], count: int) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for index, trial in enumerate(trials[:count]):
        raw_start, raw_stop = float(trial["start"]), float(trial["stop"])
        require(raw_start.is_integer() and raw_stop.is_integer(), f"trial {index} bound is not integral")
        start, stop = int(raw_start), int(raw_stop)
        require(start >= 0 and stop - start >= 50, f"trial {index} bound drift")
        require(not bounds or start >= bounds[-1][1], "trial bounds are not chronological")
        bounds.append((start, stop))
    require(len(bounds) == count, "insufficient calibration trials")
    return bounds


def _support_starts(bounds: Sequence[tuple[int, int]], *, window_size: int = 50) -> Any:
    import numpy as np

    parts = [np.arange(start, stop - window_size + 1, dtype=np.int64) for start, stop in bounds]
    require(bool(parts) and all(part.size > 0 for part in parts), "support trial emitted no valid windows")
    starts = np.ascontiguousarray(np.concatenate(parts), dtype=np.int64)
    require(bool((starts[:-1] < starts[1:]).all()), "support starts are not strictly increasing")
    return starts


def _trial_rate_features(private: Any, bounds: Sequence[tuple[int, int]]) -> tuple[Any, Any]:
    import numpy as np

    rates = np.stack([
        private.neural[start:stop].mean(axis=0, dtype=np.float64) / 0.020
        for start, stop in bounds
    ])
    targets = np.stack([
        private.behavior[start:stop].mean(axis=0, dtype=np.float64)
        for start, stop in bounds
    ])
    return np.ascontiguousarray(rates), np.ascontiguousarray(targets)


def _forward_neural_model(
    *, runtime: Any, model: Any, prepared: Any, side: Any, calibration_trials: int,
) -> tuple[float, str, Any]:
    """Run the sealed query with an explicit side tensor and calibration count."""
    import numpy as np
    import torch
    from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p
    from src.tfpd_lane import pop_robust

    private = prepared.opaque
    require(1 <= calibration_trials <= 30, "calibration trial count drift")
    state_before = runtime.state_digest(model)
    predictions: list[Any] = []
    targets: list[Any] = []
    digest = hashlib.sha256()
    first_probe: Any | None = None
    starts = tuple(int(item) for item in private.starts.tolist())
    for offset in range(0, len(starts), EVAL_BATCH_SIZE):
        chunk = starts[offset:offset + EVAL_BATCH_SIZE]
        neural = torch.from_numpy(np.stack([private.neural[start:start + 50] for start in chunk])).to("cuda:0")
        behavior = torch.from_numpy(np.stack([private.behavior[start:start + 50] for start in chunk])).to("cuda:0")
        calibration = torch.from_numpy(private.calibration[:calibration_trials]).to("cuda:0")
        calibration = calibration.unsqueeze(0).expand(len(chunk), -1, -1, -1)
        expanded_side = side.expand(len(chunk), -1, -1)
        with pop_robust.dynamic_dropout_recorder() as recorder:
            with torch.no_grad():
                output, _identity = model(neural, calib_trials=calibration, side_features=expanded_side)
        require(tuple(output.shape) == (len(chunk), 50, 2), "neural comparator output shape drift")
        require(recorder["uniform_calls"] == 0 and recorder["dropout_calls"] == [], "eval dropout became active")
        require(bool(torch.isfinite(output).all().item()), "neural comparator output nonfinite")
        if first_probe is None:
            with torch.no_grad():
                repeated, _identity = model(neural, calib_trials=calibration, side_features=expanded_side)
            require(torch.equal(output, repeated), "neural comparator repeated first batch differs")
            first_probe = True
        cpu = output.detach().cpu().contiguous()
        digest.update(cpu.numpy().tobytes())
        predictions.append(cpu[:, 49, :])
        targets.append(behavior.detach().cpu().contiguous()[:, 49, :])
    prediction = torch.cat(predictions).contiguous()
    target = torch.cat(targets).contiguous()
    valid = torch.ones(prediction.shape[0], dtype=torch.bool)
    result = v1p.ForwardResult(
        input_token_sha256=prepared.input_token_sha256,
        prediction_sha256=digest.hexdigest(), predictions=None, targets=None,
        valid_mask=None, output_shape=(prepared.n_windows, 50, 2),
        governed_bin=49, last_bin_predictions=prediction,
        last_bin_targets=target, last_bin_valid_mask=valid,
        b3s_m30_recomputed=(calibration_trials == 30),
    )
    score = float(runtime.score_result(result, session=prepared))
    state_after = runtime.state_digest(model)
    require(state_before == state_after, "neural comparator model state changed")
    return score, digest.hexdigest(), prediction.numpy()


def _strict_load_arm_a(root: Path, runtime: Any) -> tuple[Any, dict[str, object]]:
    import io
    import torch
    from torch.nn.parameter import UninitializedParameter
    from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p

    path = root / "tfpd_exploration/results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt"
    body = path.read_bytes()
    # The sealed graph deliberately retains two lazy fc_id_in parameters.  Keep
    # the restricted loader, but scope the one exact required type locally.
    with torch.serialization.safe_globals([UninitializedParameter]):
        payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    require(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping), "Arm A SWA payload drift")
    module = v1p._load_exact_module("_budget_compare_arm_a_builder", root / "tfpd_exploration/src/tfpd/spintshape_module.py")
    model = module.build_spintshape_model(seed=42)
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to("cuda:0").eval()
    require(not model.training and all(parameter.grad is None for parameter in model.parameters()), "Arm A eval boundary drift")
    return model, {
        "path": str(path.relative_to(root)), "sha256": sha256(body),
        "state_sha256": runtime.state_digest(model),
    }


def _historical_anchors(root: Path) -> dict[str, object]:
    """Load matched M30/no-label authorities without fabricating short-budget rows."""
    import numpy as np

    a2_path = root / "tfpd_exploration/results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json"
    a2_body = a2_path.read_bytes()
    a2 = json.loads(a2_body)
    a2_rows = {
        seed: {
            surface: list(a2["results"][f"A2_t4_s{seed}"][surface]["per_session"])
            for surface in ("within", "external")
        }
        for seed in (42, 43, 44)
    }
    b0_external: dict[str, object] = {}
    b0_root = root / "sua_exploration/results/subm_b0_external_score_bridge_v1"
    b0_sha: dict[str, str] = {}
    for seed in (42, 43, 44):
        path = b0_root / f"external_subject_M_b0_s{seed}.json"
        body = path.read_bytes()
        payload = json.loads(body)
        b0_sha[str(seed)] = sha256(body)
        b0_external[str(seed)] = [
            {"session": session, "r2": float(payload["per_session_mean_r2"][session])}
            for session in payload["domain_sessions"]
        ]
    within_path = root / "sua_exploration/results/a11_b0_convergence_full_access_v1/a11_b0_convergence_full_access_v1_full_cpu_forward_2843108a665b53b4.json"
    within_body = within_path.read_bytes()
    within = json.loads(within_body)
    b0_within: dict[str, object] = {}
    for seed in (42, 43, 44):
        epochs = within["cpu_forward_result"]["per_seed"][str(seed)]["per_epoch"]
        names = list(epochs["4"]["per_session_r2"])
        b0_within[str(seed)] = [
            {
                "session": session,
                "r2": float(np.mean([epochs[str(epoch)]["per_session_r2"][session] for epoch in range(4, 12)])),
            }
            for session in names
        ]
    return {
        "a2_t4_m30": {"receipt_sha256": sha256(a2_body), "per_seed": a2_rows},
        "original_spint_b0_m30_activity_no_t4_labels": {
            "external_receipt_sha256s": b0_sha,
            "within_receipt_sha256": sha256(within_body),
            "external_per_seed": b0_external,
            "within_per_seed": b0_within,
            "interpretation": "budget-invariant historical M30-activity/no-T4 system anchor; not an M4/M10 total-calibration row",
        },
    }


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    import numpy as np

    values = np.asarray([float(row["r2"]) for row in rows], dtype=np.float64)
    return {
        "session_count": len(rows), "equal_session_mean_r2": float(values.mean()),
        "equal_session_median_r2": float(np.median(values)),
    }


def _paired_contrasts(cells: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    import numpy as np

    results: list[dict[str, object]] = []
    for surface in ("within", "external"):
        for budget in BUDGETS:
            base_cells = [item for item in cells if item["surface"] == surface and item["budget"] == budget and item["system"] == "cell_d_ols" and item["regime"] == "label_limited_m30_activity"]
            require(len(base_cells) == 1, "paired Cell-D base cell missing")
            base = {str(row["session"]): float(row["r2"]) for row in base_cells[0]["sessions"]}
            for cell in cells:
                if cell["surface"] != surface or cell["budget"] != budget or cell is base_cells[0]:
                    continue
                other = {str(row["session"]): float(row["r2"]) for row in cell["sessions"]}
                require(tuple(other) == tuple(base), "paired roster/order drift")
                delta = np.asarray([other[name] - base[name] for name in base], dtype=np.float64)
                results.append({
                    "surface": surface, "budget": budget,
                    "contrast": f"{cell['system']}__{cell['regime']}__minus_cell_d_ols_label_limited",
                    "mean_delta_r2": float(delta.mean()), "median_delta_r2": float(np.median(delta)),
                    "positive_sessions": int((delta > 0).sum()), "session_count": int(delta.size),
                    "bootstrap_95": bootstrap_delta(delta.tolist()),
                    "per_session": [{"session": name, "delta_r2": float(value)} for name, value in zip(base, delta)],
                })
    return results


def execute_reviewed(root: Path) -> Mapping[str, object]:
    """Execute the complete one-shot Phase-1 matrix and publish immutable receipts."""
    import numpy as np
    import torch
    from mc_maze import d_optimal_calibration_design as design_core
    from mc_maze import subm_v9_f0_pv_ridge as numerical
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix
    from src import low_cost_calibration_v1 as lowcost
    from src.posterior_marginalized_cell_d_v1 import plan
    from src.posterior_marginalized_cell_d_v4 import matched_score as v4score

    root = Path(root).absolute()
    require(all(os.environ.get(key) == value for key, value in EXPECTED_ENVIRONMENT.items()), "launch environment drift")
    require(os.environ.get("SUBC_DATA_ROOT") == str(root / "sua_exploration/data/dandi_000688/sub-C"), "SUBC root drift")
    require(os.environ.get("SUBM_DATA_ROOT") == str(root / "sua_exploration/data/dandi_000688/sub-M"), "SUBM root drift")
    result_root = root / RESULT_ROOT_RELATIVE
    require(not result_root.exists() and not result_root.is_symlink(), "comparator result root is not fresh")
    result_root.mkdir(parents=False, mode=0o700)
    closure = implementation_closure(root)
    lowcost_path = root / "tfpd_exploration/results/low_cost_calibration_diagnostics_v3/receipt.json"
    lowcost_body = lowcost_path.read_bytes()
    lowcost_receipt = json.loads(lowcost_body)
    lowcost_reference = {
        (cell["surface"], int(cell["budget"]), row["session"]): row
        for cell in lowcost_receipt["cells"]
        if cell["support"] == "C0_contiguous" and int(cell["budget"]) in (4, 30)
        for row in cell["sessions"]
    }
    require(len(lowcost_reference) == 42, "low-cost C0 reference topology drift")
    attempt = {
        "schema": SCHEMA + "_attempt", "status": "ATTEMPT_RESERVED",
        "closure": closure, "lowcost_v3_receipt_sha256": sha256(lowcost_body),
        "budgets": list(BUDGETS), "formal_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }
    attempt_sha = _publish_pair(result_root, "attempt.json", attempt)
    backend: Any | None = None
    try:
        profile = plan.validate_compatible_device_profile(plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
        backend, _identity, _authority = lowcost._build_reviewed_backend(root, profile)
        runtime = backend.base.runtime
        cell_d = backend.sealed_model()
        arm_a, arm_a_artifact = _strict_load_arm_a(root, runtime)
        model_states_before = {
            "cell_d": runtime.state_digest(cell_d), "arm_a": runtime.state_digest(arm_a),
        }
        mean = torch.tensor(plan.SEALED_OLS_T4_MEAN_FLOAT32, device="cuda:0", dtype=torch.float32)
        std = torch.tensor(plan.SEALED_OLS_T4_STD_FLOAT32, device="cuda:0", dtype=torch.float32)
        rows_by_cell: dict[tuple[str, int, str, str], list[dict[str, object]]] = {}
        support_audit: list[dict[str, object]] = []

        def add_row(surface: str, budget: int, system: str, regime: str, row: Mapping[str, object]) -> None:
            rows_by_cell.setdefault((surface, budget, system, regime), []).append(dict(row))

        for surface in v4score.SURFACES:
            for session_name, prepared in backend.sessions(surface).items():
                private = prepared.opaque
                snapshot = private.held_asset.private_snapshot()
                try:
                    trials = list_datamodule_rewarded_trials(
                        snapshot.path, bin_size_ms=20, window_size=50, trial_result_filter="R",
                    )
                    theta_all = np.asarray([item["target_dir"] for item in trials], dtype=np.float64)
                    rates_units_trials, unit_count = _pool_trial_rate_matrix(snapshot.path, trials)
                    rates_trials_units = np.ascontiguousarray(rates_units_trials.T, dtype=np.float64)
                    snapshot.reverify()
                finally:
                    snapshot.close()
                require(unit_count == private.neural.shape[1], f"{session_name}: unit-axis drift")
                query_target = np.ascontiguousarray(private.last_targets, dtype=np.float32)
                require(query_target.shape == (prepared.n_windows, 2), f"{session_name}: target shape drift")
                prefix = numerical.prefix_sums(private.neural)
                query_rates = numerical.window_rates_from_prefix(prefix, private.starts)
                per_budget: dict[int, dict[str, object]] = {}
                for budget in BUDGETS:
                    bounds = _trial_bounds(trials, budget)
                    support_starts = _support_starts(bounds)
                    require(support_starts.size > 0, f"{session_name}/M{budget}: no support windows")
                    overlap = np.intersect1d(support_starts, private.starts, assume_unique=True)
                    require(overlap.size == 0, f"{session_name}/M{budget}: support/query overlap")
                    direction_indices = design_core.direction_indices_from_thetas(theta_all)
                    trial_times = np.asarray([item["start_time"] for item in trials], dtype=np.float64)
                    ols = lowcost.carrier_rows(
                        rates=rates_trials_units, thetas=theta_all, directions=direction_indices,
                        trial_times=trial_times, budget=budget,
                    )["C0_contiguous"]["raw_t4"]
                    canonical_theta = np.asarray(
                        [design_core.CANONICAL_DIRECTIONS_RAD[int(index)] for index in direction_indices[:budget]],
                        dtype=np.float64,
                    )
                    fixed, fixed_evidence = fit_ridge_t4(
                        rates_trials_units[:budget], canonical_theta,
                        normalized_lambda=RIDGE_T4_FIXED_LAMBDA,
                    )
                    gcv, gcv_evidence = fit_gcv_ridge_t4(rates_trials_units[:budget], canonical_theta)
                    side = {
                        "ols": ((torch.as_tensor(ols, device="cuda:0") - mean) / std).unsqueeze(0),
                        "ridge_fixed": ((torch.as_tensor(fixed, device="cuda:0") - mean) / std).unsqueeze(0),
                        "ridge_gcv": ((torch.as_tensor(gcv, device="cuda:0") - mean) / std).unsqueeze(0),
                    }
                    if budget in (4, 30):
                        require(torch.equal(side["ols"], private.side_by_budget[budget]), f"{session_name}/M{budget}: OLS side parity drift")

                    # Information-density-matched trial-level ridge.
                    trial_x, trial_y = _trial_rate_features(private, bounds)
                    trial_readout = fit_closed_form_ridge(
                        trial_x, trial_y, normalized_lambda=CLASSICAL_NORMALIZED_LAMBDA,
                    )
                    trial_prediction = predict_closed_form_ridge(query_rates, trial_readout)

                    # Label-dense W50 ridge.
                    dense_x = numerical.raw_window_features(private.neural, support_starts)
                    dense_y = numerical.targets_at_window_end(private.behavior, support_starts)
                    dense_readout = _fit_closed_form_ridge_cuda(
                        dense_x, dense_y, normalized_lambda=CLASSICAL_NORMALIZED_LAMBDA,
                    )
                    dense_prediction = _predict_dense_ridge_query(private, dense_readout)

                    # Classical population vector with a prefix-window affine gain.
                    preferred, zero_modulation = numerical.preferred_directions_from_cosine(
                        ols[:, 0], ols[:, 1], ols[:, 2],
                    )
                    support_rates = numerical.window_rates_from_prefix(prefix, support_starts)
                    support_vectors = numerical.population_vectors(support_rates, preferred, ols[:, 3])
                    gain, intercept, pv_rank = numerical.fit_population_vector_gain(support_vectors, dense_y)
                    query_vectors = numerical.population_vectors(query_rates, preferred, ols[:, 3])
                    pv_prediction = numerical.predict_population_vector(query_vectors, gain, intercept)

                    classical = {
                        "trial_rate_ridge": (trial_prediction, {
                            "support_rows": budget, "label_density": "one_mean_velocity_per_trial",
                            "formulation": trial_readout.formulation,
                            "weights_sha256": array_sha256(trial_readout.weights),
                        }),
                        "dense_w50_ridge": (dense_prediction, {
                            "support_rows": int(support_starts.size), "label_density": "every_valid_prefix_window",
                            "formulation": dense_readout.formulation,
                            "weights_sha256": array_sha256(dense_readout.weights),
                        }),
                        "population_vector": (pv_prediction, {
                            "support_rows": int(support_starts.size), "label_density": "dense_affine_gain",
                            "affine_rank": int(pv_rank), "zero_modulation_units": int(zero_modulation),
                            "gain_sha256": array_sha256(gain),
                        }),
                    }
                    for system, (prediction, evidence) in classical.items():
                        add_row(surface, budget, system, "classical_prefix_fit", {
                            "session": session_name, "n_windows": prepared.n_windows,
                            "r2": session_r2_float32(prediction, query_target),
                            "prediction_sha256": array_sha256(prediction),
                            "target_sha256": array_sha256(query_target),
                            "fit": evidence,
                        })
                    per_budget[budget] = {
                        "side": side, "fixed_evidence": fixed_evidence, "gcv_evidence": gcv_evidence,
                        "support_starts_sha256": array_sha256(support_starts),
                        "support_rows": int(support_starts.size), "support_query_overlap": int(overlap.size),
                        "ols_raw_t4_sha256": array_sha256(ols),
                    }
                    support_audit.append({
                        "surface": surface, "session": session_name, "budget": budget,
                        "support_rows": int(support_starts.size), "support_query_overlap": 0,
                        "support_starts_sha256": array_sha256(support_starts),
                        "ols_raw_t4_sha256": array_sha256(ols),
                        "ridge_fixed": fixed_evidence, "ridge_gcv": gcv_evidence,
                    })

                for budget in BUDGETS:
                    item = per_budget[budget]
                    neural_cells = (
                        ("cell_d_ols", "label_limited_m30_activity", cell_d, item["side"]["ols"], 30),
                        ("cell_d_ols", "total_calibration_limited", cell_d, item["side"]["ols"], budget),
                        ("arm_a_ols", "label_limited_m30_activity", arm_a, item["side"]["ols"], 30),
                        ("cell_d_ridge_t4_fixed_0p1", "label_limited_m30_activity", cell_d, item["side"]["ridge_fixed"], 30),
                        ("cell_d_ridge_t4_gcv", "label_limited_m30_activity", cell_d, item["side"]["ridge_gcv"], 30),
                    )
                    for system, regime, model, side, calibration_count in neural_cells:
                        r2, prediction_sha, _prediction = _forward_neural_model(
                            runtime=runtime, model=model, prepared=prepared, side=side,
                            calibration_trials=calibration_count,
                        )
                        row = {
                            "session": session_name, "n_windows": prepared.n_windows, "r2": r2,
                            "prediction_sha256": prediction_sha,
                            "target_sha256": array_sha256(query_target),
                            "normalized_side_sha256": array_sha256(side.detach().cpu().numpy()),
                            "b3s_calibration_trials": calibration_count,
                        }
                        if system == "cell_d_ridge_t4_fixed_0p1":
                            row["fit"] = item["fixed_evidence"]
                        elif system == "cell_d_ridge_t4_gcv":
                            row["fit"] = item["gcv_evidence"]
                        add_row(surface, budget, system, regime, row)
                        if system == "cell_d_ols" and regime == "label_limited_m30_activity" and budget in (4, 30):
                            reference = lowcost_reference[(surface, budget, session_name)]
                            require(prediction_sha == reference["prediction_sha256"], "Cell-D lowcost prediction parity drift")
                            require(abs(r2 - float(reference["variance_weighted_r2"])) <= 1.0e-7, "Cell-D lowcost R2 parity drift")

        require(runtime.state_digest(cell_d) == model_states_before["cell_d"], "Cell-D final state drift")
        require(runtime.state_digest(arm_a) == model_states_before["arm_a"], "Arm A final state drift")
        cells = [
            {
                "surface": surface, "budget": budget, "system": system, "regime": regime,
                "sessions": rows, "summary": _summary(rows),
            }
            for (surface, budget, system, regime), rows in rows_by_cell.items()
        ]
        expected_cell_count = 2 * len(BUDGETS) * 8
        require(len(cells) == expected_cell_count, f"comparator cell topology drift: {len(cells)}")
        historical = _historical_anchors(root)
        receipt = {
            "schema": SCHEMA + "_receipt", "status": "PHASE1_COMPARATORS_COMPLETE",
            "attempt_sha256": attempt_sha, "closure": closure,
            "lowcost_v3_receipt_sha256": sha256(lowcost_body),
            "arm_a_artifact": arm_a_artifact, "model_states_before_after_equal": True,
            "budgets": list(BUDGETS), "cells": cells,
            "paired_contrasts": _paired_contrasts(cells),
            "support_audit": support_audit, "historical_anchors": historical,
            "boundaries": {
                "query_inputs_shared": True, "support_query_overlap_zero": True,
                "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
                "formal_opened": False, "historical_short_budget_rows_fabricated": False,
            },
        }
        receipt_sha = _publish_pair(result_root, "receipt.json", receipt)
        terminal = {
            "schema": SCHEMA + "_terminal", "status": "TERMINAL",
            "attempt_sha256": attempt_sha, "receipt_sha256": receipt_sha,
            "closure": closure, "formal_opened": False,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        }
        terminal_sha = _publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {"receipt_sha256": receipt_sha, "terminal_sha256": terminal_sha, "receipt": receipt}
    except BaseException as error:
        _publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure", "status": "PHASE1_COMPARATORS_FAILED",
            "attempt_sha256": attempt_sha, "error_class": type(error).__name__,
            "error_sha256": sha256(f"{type(error).__name__}: {error}".encode()),
            "formal_opened": False, "target_optimizer_steps": 0,
            "target_backward_calls": 0, "target_update_calls": 0,
        })
        raise
    finally:
        if backend is not None:
            backend.close()
