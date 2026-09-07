"""Physical execution for the H1/M1 priority-1--3 analyses.

This route opens public held-in-calibration recordings only.  It performs one
M1 M10 operational DirectRidge reference, a controlled five-date H1 per-DoF
forward diagnostic, M1/H1 effective-rank accounting, and an H1--M2 physical
subspace feasibility gate.  It never opens minival, held-out, EvalAI, or
formal surfaces and never performs a target backward/update step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .core import (
    DirectRidgeSession,
    array_sha256,
    canonical_sha256,
    covariance_spectrum,
    direct_ridge_loso,
    need,
    regression_metrics,
    shared_axis_semantic_gate,
)


ROOT = Path(__file__).resolve().parents[2]
SPINT_ROOT = ROOT / "SPINT-main"
RESULT_ROOT = ROOT / "sua_exploration" / "results" / "h1_m1_priority_v1"
M1_ROOT = SPINT_ROOT / "data" / "000941"
H1_ROOT = SPINT_ROOT / "data" / "000954"
M2_ROOT = SPINT_ROOT / "data" / "000953"
H1_DATES = ("19250108", "19250113", "19250115", "19250119", "19250120")
H1_ARMS = ("CI64-FULL", "CI64-C0")
H1_OUTPUT_NAMES = ("tx", "ty", "tz", "rx", "g1", "g2", "g3")
M2_OUTPUT_NAMES = ("index", "mrs")
M1_EXPECTED_OUTPUT_NAMES = (
    "APL", "BCPs", "DLTa", "DLTp", "ECRB", "ECU", "EDC", "FCR",
    "FCU", "FDI", "FDPr", "FDPu", "Hypoth", "PECmaj", "TCPlat", "Thenar",
)


class ExecutionError(ValueError):
    pass


def _sha_file(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing or symlinked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_pair(path: Path, body: Mapping[str, Any]) -> str:
    """Publish one immutable JSON body and canonical basename sidecar."""

    need(path.parent == RESULT_ROOT, "result must stay in the canonical result root")
    sidecar = path.with_name(path.name + ".sha256")
    need(not path.exists() and not sidecar.exists() and not path.is_symlink() and not sidecar.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(body)
    digest = hashlib.sha256(payload).hexdigest()
    for destination, contents in ((path, payload), (sidecar, f"{digest}  {path.name}\n".encode("ascii"))):
        fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(path.parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(contents); handle.flush(); os.fsync(handle.fileno())
            os.chmod(temporary, 0o444)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    need(stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(sidecar.stat().st_mode) == 0o444, "result mode drift")
    need(_sha_file(path) == digest and sidecar.read_text(encoding="ascii") == f"{digest}  {path.name}\n", "result pair verification failed")
    return digest


def _session_from_m1_path(path: Path) -> str:
    marker = "_ses-"
    need(marker in path.stem, f"cannot parse M1 session: {path}")
    return "ses-" + path.stem.split(marker, 1)[1].split("_behavior", 1)[0]


def _m1_output_names(path: Path) -> tuple[str, ...]:
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        container = nwb.acquisition["preprocessed_emg"]
        names = tuple(str(name) for name in container.time_series.keys())
    need(names == M1_EXPECTED_OUTPUT_NAMES, f"M1 output order drift: {names}")
    return names


def load_m1_sessions() -> dict[str, DirectRidgeSession]:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    directory = M1_ROOT / "sub-MonkeyL-held-in-calib"
    paths = tuple(sorted(directory.glob("*.nwb")))
    need(len(paths) == 4 and all("held-in-calib" in str(path) for path in paths), "M1 requires exactly four held-in-calib recordings")
    sessions: dict[str, DirectRidgeSession] = {}
    expected_names: tuple[str, ...] | None = None
    for path in paths:
        names = _m1_output_names(path)
        expected_names = names if expected_names is None else expected_names
        need(names == expected_names, "M1 output order differs across sessions")
        neural, target, trial_change, eval_mask = load_nwb(path, FalconTask.m1)
        name = _session_from_m1_path(path)
        sessions[name] = DirectRidgeSession(
            name=name,
            neural=np.asarray(neural, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
            eval_mask=np.asarray(eval_mask, dtype=bool),
            trial_change=np.asarray(trial_change, dtype=bool),
            output_names=names,
            input_path=str(path.resolve()),
            input_sha256=_sha_file(path),
        )
        sessions[name].validate()
    return sessions


def run_m1_directridge() -> dict[str, Any]:
    sessions = load_m1_sessions()
    result = direct_ridge_loso(sessions)
    spectra: dict[str, Any] = {}
    for name, session in sessions.items():
        _, target, indices = session.aligned(0, split="query")
        spectra[name] = {
            "query_indices_sha256": array_sha256(indices),
            "spectrum": covariance_spectrum(target),
        }
    participation = [float(spectra[name]["spectrum"]["participation_ratio"]) for name in sorted(spectra)]
    result.update({
        "status": "COMPLETE_M1_M10_DIRECTRIDGE_OPERATIONAL_REFERENCE",
        "surface": "public held-in-calib only",
        "formal_heldout_opened": False,
        "minival_opened": False,
        "target_backward_steps": 0,
        "target_optimizer_steps": 0,
        "m1_output_diagnostics": {
            "per_session": spectra,
            "participation_ratio_equal_session_mean": float(np.mean(participation, dtype=np.float64)),
            "participation_ratio_equal_session_median": float(np.median(participation)),
        },
    })
    return result


def _model_state_sha(model: Any) -> str:
    import torch
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in sorted(model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8")); digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _h1_date_context(date: str, device: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """Restore two matched local H1 arms and the exact historical query."""

    import torch
    from omegaconf import OmegaConf
    import hydra

    evaluation_path = SPINT_ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_ci" / "terminal_evaluations" / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json"
    need(evaluation_path.is_file() and stat.S_IMODE(evaluation_path.stat().st_mode) == 0o444, f"missing immutable H1 evaluation: {date}")
    historical = json.loads(evaluation_path.read_text(encoding="utf-8"))
    checker_path = Path(historical["terminal_checker"]["path"])
    need(_sha_file(checker_path) == historical["terminal_checker"]["sha256"], f"{date}: checker hash drift")
    checker = json.loads(checker_path.read_text(encoding="utf-8"))
    preflight_path = Path(checker["ci_preflight"]["path"])
    need(_sha_file(preflight_path) == checker["ci_preflight"]["sha256"], f"{date}: CI preflight hash drift")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    binding = preflight["source_binding"]
    target_module = importlib.import_module("src.data.h1_carrierid_date_lodo_target")
    ci_target_module = importlib.import_module("src.data.h1_carrierid_date_lodo_ci_target")
    plan, normalizer, _ = target_module.load_target_dependencies(binding["source_manifest_path"], outer_date=date)
    records = target_module.load_outer_date_target_records(H1_ROOT, outer_date=date)
    datasets = {
        "CI64-FULL": ci_target_module.H1CarrierIdDateLodoCiStrictTargetDataset(records, plan, normalizer, outer_date=date, carrier_intervention="full"),
        "CI64-C0": ci_target_module.H1CarrierIdDateLodoCiStrictTargetDataset(records, plan, normalizer, outer_date=date, carrier_intervention="c0"),
    }
    need(datasets["CI64-FULL"].window_indices_sha256 == datasets["CI64-C0"].window_indices_sha256 == historical["target"]["shared_query_window_indices_sha256"], f"{date}: H1 query drift")
    models: dict[str, Any] = {}
    for arm in H1_ARMS:
        row = historical["checkpoints"][arm]
        checkpoint = Path(row["path"]); config_path = Path(row["config_path"])
        need(_sha_file(checkpoint) == row["sha256"] and _sha_file(config_path) == row["config_sha256"], f"{date}/{arm}: model authority drift")
        config = OmegaConf.load(config_path)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = hydra.utils.instantiate(config.model)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device); model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[arm] = model
    return historical, models, datasets, tuple(records)


def _h1_forward(model: Any, dataset: Any, sessions: tuple[str, ...], device: Any) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    before = _model_state_sha(model)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    with torch.inference_mode():
        for neural, target, identity, session, carrier in DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0):
            output = model(
                neural.to(device=device, dtype=torch.float32),
                calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                carrier=carrier.to(device=device, dtype=torch.float32),
            )
            if bool(model.hparams.decode_last_timestep_only):
                output = output[:, -1:, :]
            if bool(model.hparams.predict_scaled_behavior):
                output = output / model.hparams.behavior_scaling_factor
            predictions.append(output[:, -1, :].detach().cpu().numpy())
            targets.append(target[:, -1, :].detach().cpu().numpy())
            names.extend(str(value) for value in session)
    after = _model_state_sha(model)
    need(before == after, "H1 forward mutated model state")
    truth = np.concatenate(targets).astype(np.float64)
    prediction = np.concatenate(predictions).astype(np.float64)
    name_array = np.asarray(names, dtype=object)
    need(tuple(sorted(set(names))) == tuple(sorted(sessions)) and truth.shape == prediction.shape and truth.shape[1] == 7, "H1 forward output/session drift")
    return {
        "truth": truth,
        "prediction": prediction,
        "names": name_array,
        "metrics": regression_metrics(truth, prediction),
        "per_session": {
            name: regression_metrics(truth[name_array == name], prediction[name_array == name]) for name in sessions
        },
        "state_sha256_before": before,
        "state_sha256_after": after,
    }


def _h1_axis_metadata() -> dict[str, Any]:
    """Bind the semantic output labels used by the per-DoF report."""

    from pynwb import NWBHDF5IO
    paths = tuple(sorted((H1_ROOT / "sub-HumanPitt-held-in-calib").glob("*.nwb")))
    need(len(paths) == 13, "H1 axis audit requires exactly 13 held-in-calib recordings")
    rows = {}
    for path in paths:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read(); velocity = nwb.acquisition["OpenLoopKinematicsVelocity"]
            names = tuple(token.strip() for token in str(velocity.description).split(","))
            unit = str(velocity.unit)
        need(names == H1_OUTPUT_NAMES and unit == "arbitrary", f"H1 axis metadata drift: {path.name}")
        rows[path.name] = {"description": ",".join(names), "unit": unit}
    return {"axes": list(H1_OUTPUT_NAMES), "unit": "arbitrary", "recordings": rows, "recording_count": len(rows)}


def _bootstrap_mean(values: np.ndarray, *, seed: int = 42, draws: int = 10000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    need(array.ndim == 1 and array.size >= 2 and np.isfinite(array).all(), "invalid bootstrap values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size), endpoint=False)
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_h1_per_dof(*, device_name: str = "cuda:0") -> dict[str, Any]:
    import torch

    need(str(SPINT_ROOT) in sys.path, "SPINT-main must be on sys.path before H1 execution")
    device = torch.device(device_name)
    need(device.type == "cpu" or torch.cuda.is_available(), "requested H1 device unavailable")
    axis_metadata = _h1_axis_metadata()
    dates: dict[str, Any] = {}
    date_deltas: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    all_full: list[np.ndarray] = []
    all_c0: list[np.ndarray] = []
    historical_parity_abs_max = 0.0
    for date in H1_DATES:
        historical, models, datasets, sessions = _h1_date_context(date, device)
        outputs = {arm: _h1_forward(models[arm], datasets[arm], sessions, device) for arm in H1_ARMS}
        need(np.array_equal(outputs["CI64-FULL"]["truth"], outputs["CI64-C0"]["truth"]), f"{date}: H1 arms do not share targets")
        arm_rows: dict[str, Any] = {}
        for arm in H1_ARMS:
            observed = float(outputs[arm]["metrics"]["pooled_variance_weighted_r2"])
            expected = float(historical["metrics"][arm]["pooled_r2"])
            difference = abs(observed - expected)
            historical_parity_abs_max = max(historical_parity_abs_max, difference)
            need(difference <= 2.0e-6, f"{date}/{arm}: historical pooled-R2 parity drift {difference}")
            arm_rows[arm] = {
                "metrics": outputs[arm]["metrics"],
                "per_session": outputs[arm]["per_session"],
                "historical_pooled_r2": expected,
                "historical_parity_abs_error": difference,
                "state_sha256_before": outputs[arm]["state_sha256_before"],
                "state_sha256_after": outputs[arm]["state_sha256_after"],
            }
        full_r2 = np.asarray(outputs["CI64-FULL"]["metrics"]["r2_per_output"], dtype=np.float64)
        c0_r2 = np.asarray(outputs["CI64-C0"]["metrics"]["r2_per_output"], dtype=np.float64)
        delta = full_r2 - c0_r2
        date_deltas.append(delta)
        truth = outputs["CI64-FULL"]["truth"]
        dates[date] = {
            "sessions": list(sessions),
            "query_window_indices_sha256": datasets["CI64-FULL"].window_indices_sha256,
            "arms": arm_rows,
            "delta_full_minus_c0_per_output": dict(zip(H1_OUTPUT_NAMES, delta.tolist())),
            "target_spectrum": covariance_spectrum(truth),
        }
        all_truth.append(truth); all_full.append(outputs["CI64-FULL"]["prediction"]); all_c0.append(outputs["CI64-C0"]["prediction"])
        del models, datasets, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    deltas = np.stack(date_deltas)
    per_dof: dict[str, Any] = {}
    for index, name in enumerate(H1_OUTPUT_NAMES):
        values = deltas[:, index]
        per_dof[name] = {
            "equal_date_mean_delta_r2": float(values.mean(dtype=np.float64)),
            "equal_date_median_delta_r2": float(np.median(values)),
            "positive_dates": int((values > 0.0).sum()),
            "per_date_delta_r2": dict(zip(H1_DATES, values.tolist())),
            "bootstrap_95ci_equal_date_mean": list(_bootstrap_mean(values, seed=42 + index)),
        }
    truth = np.concatenate(all_truth)
    full = np.concatenate(all_full)
    c0 = np.concatenate(all_c0)
    pooled_full = regression_metrics(truth, full)
    pooled_c0 = regression_metrics(truth, c0)
    return {
        "schema": "h1_ci64_full_vs_c0_per_dof_diagnostic_v1",
        "status": "COMPLETE_H1_CONTROLLED_FIVEDATE_PER_DOF_DIAGNOSTIC",
        "contrast": "CI64-FULL minus CI64-C0; identical width-64 graph and strict M4 query, carrier present versus model-bound zero",
        "scope_warning": "This is a complete local controlled diagnostic; it is not a per-DoF reconstruction of the historical H-C minus H-S mainline.",
        "output_metadata_from_nwb": axis_metadata,
        "dates": dates,
        "per_dof_equal_date": per_dof,
        "pooled_all_windows": {
            "CI64-FULL": pooled_full,
            "CI64-C0": pooled_c0,
            "delta_per_output": (np.asarray(pooled_full["r2_per_output"]) - np.asarray(pooled_c0["r2_per_output"])).tolist(),
        },
        "all_query_target_spectrum": covariance_spectrum(truth),
        "historical_pooled_r2_parity_abs_max": historical_parity_abs_max,
        "target_backward_steps": 0,
        "target_optimizer_steps": 0,
        "formal_heldout_opened": False,
        "minival_opened": False,
    }


@dataclass(frozen=True)
class BehaviorRecord:
    dataset: str
    session: str
    path: str
    file_size: int
    names: tuple[str, ...]
    units: tuple[str, ...]
    description: str
    values: np.ndarray
    eval_mask: np.ndarray


def _behavior_records_h1_m2() -> tuple[dict[str, BehaviorRecord], dict[str, BehaviorRecord]]:
    from pynwb import NWBHDF5IO

    h1: dict[str, BehaviorRecord] = {}
    h1_dir = H1_ROOT / "sub-HumanPitt-held-in-calib"
    for path in sorted(h1_dir.glob("*.nwb")):
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read(); velocity = nwb.acquisition["OpenLoopKinematicsVelocity"]
            values = np.asarray(velocity.data[:], dtype=np.float64)
            mask = np.asarray(nwb.acquisition["eval_mask"].data[:], dtype=bool)
            names = tuple(token.strip() for token in str(velocity.description).split(","))
            units = tuple(str(velocity.unit) for _ in names)
        need(names == H1_OUTPUT_NAMES and values.shape[1] == 7 and values.shape[0] == mask.size, "H1 behavior metadata drift")
        session = path.stem.split("_ses-", 1)[1]
        h1[session] = BehaviorRecord("H1", session, str(path.resolve()), path.stat().st_size, names, units, "OpenLoopKinematicsVelocity", values, mask)
    m2: dict[str, BehaviorRecord] = {}
    m2_dir = M2_ROOT / "sub-MonkeyN-held-in-calib"
    for path in sorted(m2_dir.glob("*.nwb")):
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read(); container = nwb.acquisition["finger_vel"]
            names = tuple(str(name) for name in container.time_series.keys())
            values = np.column_stack([np.asarray(container.time_series[name].data[:], dtype=np.float64) for name in names])
            units = tuple(str(container.time_series[name].unit) for name in names)
            mask = np.asarray(nwb.acquisition["eval_mask"].data[:], dtype=bool)
        need(names == M2_OUTPUT_NAMES and values.shape[1] == 2 and values.shape[0] == mask.size, "M2 behavior metadata drift")
        session = path.stem.split("_ses-", 1)[1].split("_behavior", 1)[0]
        m2[session] = BehaviorRecord("M2", session, str(path.resolve()), path.stat().st_size, names, units, "finger_vel", values, mask)
    need(len(h1) == 13 and len(m2) == 7, "H1/M2 held-in-calib roster drift")
    return h1, m2


def run_h1_m2_subspace_gate() -> dict[str, Any]:
    h1, m2 = _behavior_records_h1_m2()
    diagnostics: dict[str, Any] = {}
    for label, records in (("H1", h1), ("M2", m2)):
        per_session = {}
        for name, record in records.items():
            values = record.values[record.eval_mask]
            need(values.shape[0] > values.shape[1] and np.isfinite(values).all(), f"{label}/{name}: invalid behavior query")
            per_session[name] = {
                "path": record.path,
                "file_size": record.file_size,
                "behavior_array_sha256": array_sha256(values),
                "names": list(record.names),
                "units": list(record.units),
                "source_object": record.description,
                "spectrum": covariance_spectrum(values),
            }
        prs = [row["spectrum"]["participation_ratio"] for row in per_session.values()]
        diagnostics[label] = {
            "per_session": per_session,
            "participation_ratio_equal_session_mean": float(np.mean(prs, dtype=np.float64)),
            "participation_ratio_equal_session_median": float(np.median(prs)),
        }
    h1_names, m2_names = H1_OUTPUT_NAMES, M2_OUTPUT_NAMES
    h1_units = sorted({unit for row in h1.values() for unit in row.units})
    m2_units = sorted({unit for row in m2.values() for unit in row.units})
    semantic = shared_axis_semantic_gate(h1_names, ("arbitrary",) * 7, m2_names, ("AU",) * 2)
    return {
        "schema": "h1_m2_shared_physical_subspace_feasibility_gate_v1",
        "status": "STOP_NO_AUDITABLE_SHARED_PHYSICAL_AXES",
        "decision": "STOP",
        "predeclared_gate": {
            "requires_at_least_two_metadata_identical_physical_axes": True,
            "requires_compatible_physical_units": True,
            "forbids_session_specific_target_fitted_rotation": True,
            "unmatched_dimensions_must_be_marginalized_not_zero_filled": True,
        },
        "semantic_evidence": {
            "H1": {"object": "OpenLoopKinematicsVelocity", "axes": list(h1_names), "units": h1_units, "meaning": "human 3 translation + 1 rotation + 3 grasp coordinates"},
            "M2": {"object": "finger_vel", "axes": list(m2_names), "units": m2_units, "meaning": "nonhuman-primate index and MRS finger-group velocities"},
            "exact_axis_name_matches": semantic["exact_name_and_unit_matches"],
            "unit_sets_identical": h1_units == m2_units,
            "semantic_gate_passed": semantic["passed"],
            "reason": "H1 tx/ty are spatial translation coordinates; M2 index/mrs are finger-group coordinates. Normalized covariance similarity cannot create a physical correspondence.",
        },
        "numeric_rank_gate": diagnostics,
        "numeric_gate_can_override_semantic_failure": False,
        "formal_heldout_opened": False,
        "minival_opened": False,
        "target_backward_steps": 0,
        "target_optimizer_steps": 0,
    }


def execute_all(*, h1_device: str = "cuda:0") -> dict[str, Any]:
    need(not RESULT_ROOT.exists(), f"canonical result root already exists: {RESULT_ROOT}")
    m1 = run_m1_directridge()
    m1_sha = _write_pair(RESULT_ROOT / "m1_directridge_and_output_rank.json", m1)
    h1 = run_h1_per_dof(device_name=h1_device)
    h1_sha = _write_pair(RESULT_ROOT / "h1_per_dof.json", h1)
    gate = run_h1_m2_subspace_gate()
    gate_sha = _write_pair(RESULT_ROOT / "h1_m2_subspace_gate.json", gate)
    terminal = {
        "schema": "h1_m1_priority_1_3_terminal_v1",
        "status": "TERMINAL_COMPLETE_PRIORITY_1_3",
        "artifacts": {
            "m1_directridge_and_output_rank.json": m1_sha,
            "h1_per_dof.json": h1_sha,
            "h1_m2_subspace_gate.json": gate_sha,
        },
        "scientific_decisions": {
            "m1_directridge": "COMPLETE_OPERATIONAL_REFERENCE",
            "h1_per_dof": "COMPLETE_CONTROLLED_DIAGNOSTIC_NOT_MAINLINE_RECONSTRUCTION",
            "h1_m2_shared_physical_subspace": "STOP",
        },
        "surface": "public held-in-calib only",
        "formal_heldout_opened": False,
        "minival_opened": False,
        "target_backward_steps": 0,
        "target_optimizer_steps": 0,
    }
    terminal_sha = _write_pair(RESULT_ROOT / "terminal.json", terminal)
    return {"status": terminal["status"], "result_root": str(RESULT_ROOT), "terminal_sha256": terminal_sha}


def audit_existing_results() -> dict[str, Any]:
    """Descriptor-light final audit binding implementation bytes to results."""

    receipt_names = (
        "m1_directridge_and_output_rank.json",
        "h1_per_dof.json",
        "h1_m2_subspace_gate.json",
        "terminal.json",
    )
    receipts: dict[str, dict[str, Any]] = {}
    receipt_hashes: dict[str, str] = {}
    for name in receipt_names:
        path = RESULT_ROOT / name; sidecar = path.with_name(path.name + ".sha256")
        need(path.is_file() and sidecar.is_file() and not path.is_symlink() and not sidecar.is_symlink(), f"missing result pair: {name}")
        need(stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(sidecar.stat().st_mode) == 0o444, f"mutable result pair: {name}")
        digest = _sha_file(path)
        need(sidecar.read_text(encoding="ascii") == f"{digest}  {name}\n", f"sidecar drift: {name}")
        receipts[name] = json.loads(path.read_text(encoding="utf-8")); receipt_hashes[name] = digest
    m1 = receipts["m1_directridge_and_output_rank.json"]
    h1 = receipts["h1_per_dof.json"]
    gate = receipts["h1_m2_subspace_gate.json"]
    terminal = receipts["terminal.json"]
    need(m1.get("status") == "COMPLETE_M1_M10_DIRECTRIDGE_OPERATIONAL_REFERENCE", "M1 receipt status drift")
    need(m1.get("target_support_or_query_used_for_candidate_selection") is False and len(m1.get("folds", {})) == 4, "M1 nested selection boundary drift")
    need(all(name not in row["source_sessions"] and row["target_excluded_from_hyperparameter_selection"] is True for name, row in m1["folds"].items()), "M1 target leaked into hyperparameter selection")
    need(h1.get("status") == "COMPLETE_H1_CONTROLLED_FIVEDATE_PER_DOF_DIAGNOSTIC" and h1.get("historical_pooled_r2_parity_abs_max", 1.0) < 2.0e-6, "H1 replay/parity drift")
    need(gate.get("decision") == "STOP" and gate.get("semantic_evidence", {}).get("semantic_gate_passed") is False, "H1-M2 gate drift")
    need(terminal.get("status") == "TERMINAL_COMPLETE_PRIORITY_1_3" and terminal.get("artifacts") == {name: receipt_hashes[name] for name in receipt_names[:3]}, "terminal artifact edge drift")
    code_paths = (
        ROOT / "sua_exploration/h1_m1_priority_v1/__init__.py",
        ROOT / "sua_exploration/h1_m1_priority_v1/core.py",
        ROOT / "sua_exploration/h1_m1_priority_v1/execute.py",
        ROOT / "sua_exploration/scripts/run_h1_m1_priority_v1.py",
        ROOT / "sua_exploration/tests/test_h1_m1_priority_v1.py",
    )
    code = {str(path.relative_to(ROOT)): _sha_file(path) for path in code_paths}
    audit = {
        "schema": "h1_m1_priority_1_3_independent_audit_v1",
        "status": "PASS_H1_M1_PRIORITY_1_3_INTEGRITY_AND_SEMANTIC_AUDIT",
        "result_body_sha256": receipt_hashes,
        "implementation_sha256": code,
        "implementation_closure_sha256": canonical_sha256(code),
        "assertions": {
            "all_result_pairs_regular_immutable_0444_and_sidecar_valid": True,
            "m1_nested_loso_target_excluded": True,
            "h1_historical_pooled_r2_parity": True,
            "h1_controlled_contrast_not_mainline_reconstruction": True,
            "h1_m2_semantic_gate_stop": True,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
        },
    }
    audit_sha = _write_pair(RESULT_ROOT / "audit.json", audit)
    final = {
        "schema": "h1_m1_priority_1_3_terminal_v2",
        "status": "TERMINAL_COMPLETE_AND_AUDITED_PRIORITY_1_3",
        "predecessor_terminal_sha256": receipt_hashes["terminal.json"],
        "audit_sha256": audit_sha,
        "implementation_closure_sha256": audit["implementation_closure_sha256"],
        "scientific_decisions": terminal["scientific_decisions"],
        "formal_heldout_opened": False,
        "minival_opened": False,
        "target_backward_steps": 0,
        "target_optimizer_steps": 0,
    }
    final_sha = _write_pair(RESULT_ROOT / "terminal_v2.json", final)
    return {"status": final["status"], "audit_sha256": audit_sha, "terminal_v2_sha256": final_sha}
