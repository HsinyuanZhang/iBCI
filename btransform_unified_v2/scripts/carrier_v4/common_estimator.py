"""Shared source-only latent-state carrier estimator for M1 and H1 v4.

The core deliberately has no NWB, GPU, decoder, or target-file dependency.
Callers provide only already-authorized source arrays.  ``fit_behavior`` defines
where the common state dictionary is fitted; this module never expands that
range.  ``support_behavior`` and ``support_rates`` define the legal per-unit
calibration solve for each record.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import NMF


SCHEMA = "carrier_v4_latent_state3_ridge_intercept_per_column_v1"
TASKS = ("m1", "h1")
RANK = 3
RIDGE = 1.0
SCALE_FLOOR = 1.0e-6
RMS_FLOOR = 1.0e-8


@dataclass(frozen=True)
class V4Fit:
    """Frozen source-only state dictionary and carrier normalizer."""

    task: str
    source_sessions: tuple[str, ...]
    behavior_rms: np.ndarray       # [16] M1 or [7] H1
    dictionary: np.ndarray         # [3, 16] M1 or [3, 14] H1; L2-normalized rows
    normalizer_mean: np.ndarray    # [4]
    normalizer_scale: np.ndarray   # [4], each >= 1e-6
    metadata: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _typed_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        array.dtype.str.encode("utf-8") + str(tuple(array.shape)).encode("utf-8") + array.tobytes()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_sha256() -> str:
    return _file_sha256(Path(__file__).resolve())


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _as_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.ndim == ndim and array.size > 0, f"{name} must be a nonempty {ndim}-D array")
    _require(np.isfinite(array).all(), f"{name} contains nonfinite values")
    return np.ascontiguousarray(array)


def _validate_record(task: str, session: str, record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require(isinstance(record, Mapping), f"{session}: record must be a mapping")
    required = {"fit_behavior", "support_behavior", "support_rates"}
    _require(set(record) >= required, f"{session}: missing required keys {sorted(required - set(record))}")
    raw_dim = 16 if task == "m1" else 7
    fit_behavior = _as_array(record["fit_behavior"], name=f"{session}.fit_behavior", ndim=2)
    support_behavior = _as_array(record["support_behavior"], name=f"{session}.support_behavior", ndim=2)
    support_rates = _as_array(record["support_rates"], name=f"{session}.support_rates", ndim=2)
    _require(fit_behavior.shape[1] == raw_dim, f"{session}: fit_behavior width must be {raw_dim}")
    _require(support_behavior.shape[1] == raw_dim, f"{session}: support_behavior width must be {raw_dim}")
    _require(support_behavior.shape[0] == support_rates.shape[0], f"{session}: support rows must align")
    return fit_behavior, support_behavior, support_rates


def _weights(task: str, behavior: np.ndarray, behavior_rms: np.ndarray) -> np.ndarray:
    """Map task-specific raw behavior to nonnegative common-state inputs."""
    value = _as_array(behavior, name="behavior", ndim=2)
    rms = _as_array(behavior_rms, name="behavior_rms", ndim=1)
    if task == "m1":
        _require(value.shape[1] == 16 and rms.shape == (16,), "M1 behavior/RMS geometry")
        result = np.maximum(value, 0.0) / rms[None, :]
    elif task == "h1":
        _require(value.shape[1] == 7 and rms.shape == (7,), "H1 behavior/RMS geometry")
        scaled = value / rms[None, :]
        result = np.concatenate((np.logaddexp(0.0, scaled), np.logaddexp(0.0, -scaled)), axis=1)
    else:
        raise ValueError(f"task must be one of {TASKS}")
    _require(np.isfinite(result).all() and np.all(result >= 0.0), "state weights must be finite and nonnegative")
    return np.ascontiguousarray(result)


def _fit_behavior_rms(task: str, fit_arrays: list[np.ndarray]) -> np.ndarray:
    pooled = np.concatenate(fit_arrays, axis=0)
    if task == "m1":
        pooled = np.maximum(pooled, 0.0)
    rms = np.sqrt(np.mean(np.square(pooled), axis=0))
    _require(np.isfinite(rms).all(), "behavior RMS is nonfinite")
    return np.maximum(rms, RMS_FLOOR)


def _row_sha256(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def _fit_dictionary(source_states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit source NMF3, L2-normalize rows, then canonicalize by energy and SHA."""
    _require(source_states.ndim == 2 and source_states.shape[1] >= RANK, "state matrix has invalid geometry")
    _require(np.isfinite(source_states).all() and np.all(source_states >= 0.0), "NMF requires finite nonnegative states")
    _require(float(np.max(source_states)) > 0.0, "source state matrix is all zero")
    model = NMF(
        n_components=RANK,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        random_state=42,
        tol=1.0e-5,
        max_iter=1000,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    activations = np.asarray(model.fit_transform(source_states), dtype=np.float64)
    dictionary = np.asarray(model.components_, dtype=np.float64)
    _require(dictionary.shape == (RANK, source_states.shape[1]), "NMF dictionary geometry")
    n_iter = int(getattr(model, "n_iter_", 1000))
    _require(n_iter < 1000, f"NMF did not converge before max_iter=1000 (n_iter={n_iter})")
    _require(np.isfinite(activations).all() and np.isfinite(dictionary).all(), "NMF output is nonfinite")
    norms = np.linalg.norm(dictionary, axis=1)
    _require(np.all(norms > 0.0), "NMF dictionary row vanished")
    dictionary = dictionary / norms[:, None]
    activations = activations * norms[None, :]
    energy = np.sum(np.square(activations), axis=0)
    order = tuple(sorted(range(RANK), key=lambda index: (-float(energy[index]), _row_sha256(dictionary[index]), index)))
    dictionary = np.ascontiguousarray(dictionary[list(order)])
    activations = np.ascontiguousarray(activations[:, list(order)])
    energy = np.ascontiguousarray(energy[list(order)])
    import scipy
    import sklearn
    nmf_diagnostics = {
        "n_iter": n_iter,
        "reconstruction_err": float(getattr(model, "reconstruction_err_", float("nan"))),
        "sklearn_version": str(sklearn.__version__),
        "scipy_version": str(scipy.__version__),
    }
    _require(np.isfinite(nmf_diagnostics["reconstruction_err"]), "NMF reconstruction error is nonfinite")
    return dictionary, activations, energy, nmf_diagnostics


def _nnls_activations(states: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    """Old rSyn NNLS deployment law, including its explicit nonnegative clamp."""
    state = _as_array(states, name="states", ndim=2)
    basis = _as_array(dictionary, name="dictionary", ndim=2)
    _require(basis.shape[0] == RANK and state.shape[1] == basis.shape[1], "state/dictionary geometry")
    output = np.empty((state.shape[0], RANK), dtype=np.float64)
    design = basis.T
    for index, row in enumerate(state):
        coefficient, _ = nnls(design, row)
        output[index] = coefficient
    _require(np.isfinite(output).all() and np.all(output >= -1.0e-12), "NNLS output is invalid")
    # Preserve the sealed rSyn postcondition and float64 representation exactly.
    return np.maximum(output, 0.0)


def _fit_unit_ridge_rsyn_law(scores: np.ndarray, rate: np.ndarray) -> tuple[np.ndarray, float]:
    """Literal rSyn unit ridge law: per-unit GEMV and one 4x4 solve."""
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rate, dtype=np.float64).reshape(-1)
    _require(z.ndim == 2 and z.shape[0] == r.shape[0] >= 1 and z.shape[1] == RANK, "ridge shapes")
    n = float(z.shape[0])
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))
    gram = (design.T @ design) / n
    penalty = np.diag([0.0, RIDGE, RIDGE, RIDGE])
    rhs = (design.T @ r) / n
    beta = np.linalg.solve(gram + penalty, rhs)
    _require(np.isfinite(beta).all(), "ridge nonfinite")
    return beta[1:].copy(), float(beta[0])


def _raw_carrier(states: np.ndarray, rates: np.ndarray, dictionary: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit raw-Hz carrier through the sealed rSyn per-unit ridge operation order."""
    activations = _nnls_activations(states, dictionary)
    response = _as_array(rates, name="rates", ndim=2)
    _require(response.shape[0] == activations.shape[0], "rates and state rows must align")
    weights = np.zeros((response.shape[1], RANK), dtype=np.float64)
    intercepts = np.zeros(response.shape[1], dtype=np.float64)
    for unit in range(response.shape[1]):
        weights[unit], intercepts[unit] = _fit_unit_ridge_rsyn_law(activations, response[:, unit])
    raw = np.column_stack((weights, intercepts))
    _require(raw.shape == (response.shape[1], 4) and np.isfinite(raw).all(), "unit ridge carrier is invalid")
    design = np.column_stack((np.ones(activations.shape[0], dtype=np.float64), activations))
    condition = float(np.linalg.cond((design.T @ design) / float(design.shape[0]) + np.diag([0.0, RIDGE, RIDGE, RIDGE])))
    return np.ascontiguousarray(raw), {
        "design_rows": int(design.shape[0]),
        "design_condition_number": condition,
        "activation_column_std": np.std(activations, axis=0).tolist(),
        "finite_fit": bool(np.isfinite(raw).all() and np.isfinite(condition)),
        "unit_ridge_operation": "per_unit_gemv_and_solve_rsyn_law",
    }


def _fit_metadata(
    task: str,
    names: tuple[str, ...],
    records: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    behavior_rms: np.ndarray,
    dictionary: np.ndarray,
    energy: np.ndarray,
    normalizer_mean: np.ndarray,
    normalizer_scale: np.ndarray,
    nmf_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    source_arrays = {
        name: {
            "fit_behavior_typed_sha256": _typed_sha256(records[name][0]),
            "support_behavior_typed_sha256": _typed_sha256(records[name][1]),
            "support_rates_typed_sha256": _typed_sha256(records[name][2]),
            "fit_behavior_shape": list(records[name][0].shape),
            "support_behavior_shape": list(records[name][1].shape),
            "support_rates_shape": list(records[name][2].shape),
        }
        for name in names
    }
    return {
        "schema": SCHEMA,
        "method": "latent_state3_ridge_intercept/per_column",
        "task": task,
        "source_sessions": list(names),
        "fit_behavior_contract": "caller-declared source range; core does not infer or expand it",
        "state_mapping": "relu_emg16/source_rms" if task == "m1" else "softplus_signed_velocity14/source_rms",
        "dictionary_estimator": {
            "name": "sklearn.decomposition.NMF",
            "rank": RANK,
            "init": "nndsvda",
            "solver": "cd",
            "beta_loss": "frobenius",
            "random_state": 42,
            "tol": 1.0e-5,
            "max_iter": 1000,
            "alpha_W": 0.0,
            "alpha_H": 0.0,
            "l1_ratio": 0.0,
            "postprocess": "each dictionary row L2-normalized; activation-energy descending then row-SHA tie order",
        },
        "deployment_activation": "scipy.optimize.nnls with frozen row-normalized source dictionary",
        "unit_encoding": "raw Hz rates; design [1,z1,z2,z3]; gram/n + diag(0,1,1,1); intercept unpenalized",
        "normalizer": "pooled source unit raw carrier per-column mean/std; std <= 1e-6 becomes 1",
        "source_arrays": source_arrays,
        "array_typed_sha256": {
            "behavior_rms": _typed_sha256(behavior_rms),
            "dictionary": _typed_sha256(dictionary),
            "normalizer_mean": _typed_sha256(normalizer_mean),
            "normalizer_scale": _typed_sha256(normalizer_scale),
        },
        "dictionary_activation_energy": energy.tolist(),
        "nmf_fit": dict(nmf_diagnostics),
        "implementation_sha256": _code_sha256(),
    }



def import_frozen_source_fit(
    task: str,
    source_sessions: tuple[str, ...] | list[str],
    behavior_rms: np.ndarray,
    dictionary: np.ndarray,
    normalizer_mean: np.ndarray,
    normalizer_scale: np.ndarray,
    provenance: Mapping[str, Any],
) -> V4Fit:
    """Create a V4Fit from a sealed source fit without refitting its dictionary.

    This validates supplied numeric arrays only.  The caller is responsible for
    binding their reader, source roster, and artifact hashes in ``provenance``;
    this core neither reads those artifacts nor expands the declared source set.
    """
    _require(task in TASKS, f"task must be one of {TASKS}")
    names = tuple(str(name) for name in source_sessions)
    _require(names and len(set(names)) == len(names) and all(names), "source sessions must be unique nonempty names")
    _require(isinstance(provenance, Mapping), "provenance must be a mapping")
    raw_dim = 16 if task == "m1" else 7
    state_dim = 16 if task == "m1" else 14
    rms = _as_array(behavior_rms, name="behavior_rms", ndim=1)
    basis = _as_array(dictionary, name="dictionary", ndim=2)
    mean = _as_array(normalizer_mean, name="normalizer_mean", ndim=1)
    scale = _as_array(normalizer_scale, name="normalizer_scale", ndim=1)
    _require(rms.shape == (raw_dim,), "imported behavior RMS geometry")
    _require(basis.shape == (RANK, state_dim), "imported dictionary geometry")
    _require(mean.shape == scale.shape == (4,), "imported normalizer geometry")
    _require(np.all(rms >= RMS_FLOOR), "imported behavior RMS must respect floor")
    _require(np.all(basis >= 0.0), "imported dictionary must be nonnegative")
    _require(np.all(np.linalg.norm(basis, axis=1) > 0.0), "imported dictionary row vanished")
    _require(np.all(scale > 0.0), "imported normalizer scale must be positive")
    metadata = {
        "schema": SCHEMA,
        "method": "latent_state3_ridge_intercept/per_column",
        "task": task,
        "source_sessions": list(names),
        "dictionary_origin": "imported_frozen_not_refit",
        "fit_behavior_contract": "external sealed source fit; core did not inspect or refit source behavior",
        "state_mapping": "relu_emg16/source_rms" if task == "m1" else "softplus_signed_velocity14/source_rms",
        "deployment_activation": "scipy.optimize.nnls with imported frozen dictionary; explicit maximum(coefficients,0)",
        "unit_encoding": "raw Hz rates; per-unit [1,z1,z2,z3] GEMV/solve; gram/n + diag(0,1,1,1); intercept unpenalized",
        "normalizer": "imported frozen per-column source normalizer",
        "source_arrays": _jsonable(dict(provenance)),
        "array_typed_sha256": {
            "behavior_rms": _typed_sha256(rms),
            "dictionary": _typed_sha256(basis),
            "normalizer_mean": _typed_sha256(mean),
            "normalizer_scale": _typed_sha256(scale),
        },
        "implementation_sha256": _code_sha256(),
    }
    fit = V4Fit(task, names, rms, basis, mean, scale, metadata)
    _validate_fit_geometry(fit)
    return fit

def fit_source(task: str, source_records: Mapping[str, Mapping[str, Any]]) -> tuple[V4Fit, dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Fit one source-only V4 model and return normalized source carriers.

    ``source_records`` maps a stable session name to exactly the caller-authorized
    arrays: ``fit_behavior[Tfit,Kraw]``, ``support_behavior[Tsupport,Kraw]``, and
    ``support_rates[Tsupport,U]``.  The core never reads any additional data.
    """
    _require(task in TASKS, f"task must be one of {TASKS}")
    _require(isinstance(source_records, Mapping) and bool(source_records), "source_records must be nonempty")
    names = tuple(str(name) for name in source_records)
    _require(len(set(names)) == len(names) and all(name for name in names), "source sessions must be unique nonempty names")
    records = {name: _validate_record(task, name, source_records[name]) for name in names}
    behavior_rms = _fit_behavior_rms(task, [records[name][0] for name in names])
    source_states = np.concatenate([_weights(task, records[name][0], behavior_rms) for name in names], axis=0)
    dictionary, _fit_activations, energy, nmf_diagnostics = _fit_dictionary(source_states)

    raw_carriers: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for name in names:
        _fit_behavior, support_behavior, support_rates = records[name]
        states = _weights(task, support_behavior, behavior_rms)
        raw, diagnostic = _raw_carrier(states, support_rates, dictionary)
        raw_carriers[name] = raw
        diagnostics[name] = {
            **diagnostic,
            "raw_carrier_typed_sha256": _typed_sha256(raw),
            "support_state_typed_sha256": _typed_sha256(states),
        }

    pooled = np.concatenate([raw_carriers[name] for name in names], axis=0)
    normalizer_mean = np.mean(pooled, axis=0)
    normalizer_scale = np.std(pooled, axis=0)
    normalizer_scale = np.where(normalizer_scale <= SCALE_FLOOR, 1.0, normalizer_scale)
    _require(np.isfinite(normalizer_mean).all() and np.isfinite(normalizer_scale).all(), "normalizer is invalid")
    metadata = _fit_metadata(task, names, records, behavior_rms, dictionary, energy, normalizer_mean, normalizer_scale, nmf_diagnostics)
    fit = V4Fit(task, names, behavior_rms, dictionary, normalizer_mean, normalizer_scale, metadata)
    carriers = {
        name: np.asarray((raw_carriers[name] - normalizer_mean[None, :]) / normalizer_scale[None, :], dtype=np.float32)
        for name in names
    }
    for name, carrier in carriers.items():
        _require(carrier.shape == raw_carriers[name].shape and np.isfinite(carrier).all(), f"{name}: normalized carrier invalid")
        diagnostics[name]["carrier_typed_sha256"] = _typed_sha256(carrier)
    return fit, carriers, diagnostics


def deploy(fit: V4Fit, behavior: Any, rates: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use only a frozen fit plus legal support arrays to create one carrier."""
    _validate_fit_geometry(fit)
    behavior_array = _as_array(behavior, name="behavior", ndim=2)
    rates_array = _as_array(rates, name="rates", ndim=2)
    _require(behavior_array.shape[0] == rates_array.shape[0], "behavior/rates support rows must align")
    states = _weights(fit.task, behavior_array, fit.behavior_rms)
    raw, diagnostic = _raw_carrier(states, rates_array, fit.dictionary)
    carrier = np.asarray((raw - fit.normalizer_mean[None, :]) / fit.normalizer_scale[None, :], dtype=np.float32)
    _require(np.isfinite(carrier).all(), "deployed carrier is invalid")
    diagnostic.update({
        "support_state_typed_sha256": _typed_sha256(states),
        "raw_carrier_typed_sha256": _typed_sha256(raw),
        "carrier_typed_sha256": _typed_sha256(carrier),
    })
    return carrier, raw, diagnostic


def _validate_fit_geometry(fit: V4Fit) -> None:
    _require(isinstance(fit, V4Fit) and fit.task in TASKS, "invalid V4Fit")
    raw_dim = 16 if fit.task == "m1" else 7
    state_dim = 16 if fit.task == "m1" else 14
    _require(fit.behavior_rms.shape == (raw_dim,), "fit behavior RMS geometry")
    _require(fit.dictionary.shape == (RANK, state_dim), "fit dictionary geometry")
    _require(fit.normalizer_mean.shape == (4,) and fit.normalizer_scale.shape == (4,), "fit normalizer geometry")
    _require(np.isfinite(fit.behavior_rms).all() and np.all(fit.behavior_rms >= RMS_FLOOR), "fit RMS invalid")
    _require(np.isfinite(fit.dictionary).all() and np.all(fit.dictionary >= 0.0), "fit dictionary invalid")
    _require(np.isfinite(fit.normalizer_mean).all() and np.isfinite(fit.normalizer_scale).all() and np.all(fit.normalizer_scale > 0.0), "fit normalizer invalid")


def save_fit(fit: V4Fit, path_prefix: str | Path) -> tuple[Path, Path]:
    """Write immutable JSON/NPZ fit artifacts and return ``(json_path, npz_path)``."""
    _validate_fit_geometry(fit)
    prefix = Path(path_prefix)
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    _require(not json_path.exists() and not npz_path.exists(), "refuse to overwrite V4 fit artifacts")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "behavior_rms": np.ascontiguousarray(fit.behavior_rms, dtype=np.float64),
        "dictionary": np.ascontiguousarray(fit.dictionary, dtype=np.float64),
        "normalizer_mean": np.ascontiguousarray(fit.normalizer_mean, dtype=np.float64),
        "normalizer_scale": np.ascontiguousarray(fit.normalizer_scale, dtype=np.float64),
    }
    np.savez_compressed(npz_path, **arrays)
    body = {
        **_jsonable(dict(fit.metadata)),
        "schema": SCHEMA,
        "task": fit.task,
        "source_sessions": list(fit.source_sessions),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": _file_sha256(npz_path),
        "array_typed_sha256": {name: _typed_sha256(value) for name, value in arrays.items()},
        "implementation_sha256": _code_sha256(),
    }
    json_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path, npz_path


def load_fit(json_path: str | Path) -> V4Fit:
    """Load only an exact-schema, SHA-bound artifact from the current implementation."""
    receipt_path = Path(json_path)
    _require(receipt_path.is_file(), f"missing V4 receipt: {receipt_path}")
    body = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(isinstance(body, dict) and body.get("schema") == SCHEMA, "V4 receipt schema mismatch")
    _require(body.get("implementation_sha256") == _code_sha256(), "V4 implementation SHA mismatch")
    task = body.get("task")
    _require(task in TASKS, "V4 receipt task mismatch")
    source_sessions = tuple(body.get("source_sessions", ()))
    _require(source_sessions and all(isinstance(item, str) and item for item in source_sessions), "V4 receipt source sessions invalid")
    arrays_path = Path(body.get("npz_path", ""))
    _require(arrays_path.is_file() and body.get("npz_sha256") == _file_sha256(arrays_path), "V4 NPZ SHA mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        required = ("behavior_rms", "dictionary", "normalizer_mean", "normalizer_scale")
        _require(set(archive.files) == set(required), "V4 NPZ keys mismatch")
        arrays = {name: np.ascontiguousarray(archive[name], dtype=np.float64) for name in required}
    expected = body.get("array_typed_sha256")
    _require(isinstance(expected, Mapping), "V4 receipt array hashes missing")
    for name, value in arrays.items():
        _require(expected.get(name) == _typed_sha256(value), f"V4 array SHA mismatch: {name}")
    fit = V4Fit(task, source_sessions, arrays["behavior_rms"], arrays["dictionary"], arrays["normalizer_mean"], arrays["normalizer_scale"], body)
    _validate_fit_geometry(fit)
    return fit


__all__ = ["V4Fit", "fit_source", "deploy", "save_fit", "load_fit", "import_frozen_source_fit"]
