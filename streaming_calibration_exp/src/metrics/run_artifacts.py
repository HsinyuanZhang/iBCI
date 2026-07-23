"""Run artifact writers (CSV schema, git state, hardware cost)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.components.streaming_encoders import EncoderCostProfile


METRICS_SUMMARY_FIELDS = [
    "run_id",
    "variant",
    "seed",
    "validation_protocol",
    "fold_id",
    "split",
    "session",
    "M",
    "R2_variance_weighted",
    "R2_delta_vs_matched_baseline",
    "identity_mse",
    "prediction_distill_mse",
    "parameter_count",
    "weight_bytes",
    "MAC_per_trial",
    "MAC_per_session",
    "peak_live_state_bytes",
    "trial_buffer_bytes",
    "support_state_bytes",
    "requires_cubic_interpolation",
    "requires_general_multiplier",
    "requires_divider",
    "quant_format",
    "saturation_count",
]


METRICS_PER_SESSION_FIELDS = METRICS_SUMMARY_FIELDS


@dataclass
class ParsedTestMetrics:
    heldin_mean_r2: Optional[float]
    heldout_mean_r2: Optional[float]
    heldin_identity_mse: Optional[float]
    heldout_identity_mse: Optional[float]
    heldin_prediction_distill_mse: Optional[float]
    heldout_prediction_distill_mse: Optional[float]
    heldin_session_r2: Dict[str, float]
    heldout_session_r2: Dict[str, float]
    heldin_session_identity_mse: Dict[str, float]
    heldout_session_identity_mse: Dict[str, float]
    heldin_session_prediction_distill_mse: Dict[str, float]
    heldout_session_prediction_distill_mse: Dict[str, float]


def checkpoint_sha256(ckpt_path: str | Path) -> str:
    path = Path(ckpt_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_test_metrics(metric_dict: Mapping[str, Any]) -> ParsedTestMetrics:
    heldin_r2: Dict[str, float] = {}
    heldout_r2: Dict[str, float] = {}
    heldin_identity: Dict[str, float] = {}
    heldout_identity: Dict[str, float] = {}
    heldin_pred: Dict[str, float] = {}
    heldout_pred: Dict[str, float] = {}

    for key, value in metric_dict.items():
        key = str(key)
        scalar = _metric_scalar(value)
        if scalar is None:
            continue
        for split_name, store_r2, store_identity, store_pred in (
            ("test_heldin_", heldin_r2, heldin_identity, heldin_pred),
            ("test_heldout_", heldout_r2, heldout_identity, heldout_pred),
        ):
            if not key.startswith(split_name):
                continue
            remainder = key[len(split_name) :]
            if remainder.endswith("/r2"):
                store_r2[remainder[: -len("/r2")]] = scalar
            elif remainder.endswith("/identity_mse"):
                store_identity[remainder[: -len("/identity_mse")]] = scalar
            elif remainder.endswith("/prediction_distill_mse"):
                store_pred[remainder[: -len("/prediction_distill_mse")]] = scalar
            break

    return ParsedTestMetrics(
        heldin_mean_r2=_metric_scalar(metric_dict.get("test_heldin/r2_mean")),
        heldout_mean_r2=_metric_scalar(metric_dict.get("test_heldout/r2_mean")),
        heldin_identity_mse=_metric_scalar(metric_dict.get("test_heldin/identity_mse")),
        heldout_identity_mse=_metric_scalar(metric_dict.get("test_heldout/identity_mse")),
        heldin_prediction_distill_mse=_metric_scalar(metric_dict.get("test_heldin/prediction_distill_mse")),
        heldout_prediction_distill_mse=_metric_scalar(metric_dict.get("test_heldout/prediction_distill_mse")),
        heldin_session_r2=heldin_r2,
        heldout_session_r2=heldout_r2,
        heldin_session_identity_mse=heldin_identity,
        heldout_session_identity_mse=heldout_identity,
        heldin_session_prediction_distill_mse=heldin_pred,
        heldout_session_prediction_distill_mse=heldout_pred,
    )


def load_baseline_session_r2(path: str | Path | None) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {"heldin": {}, "heldout": {}}
    csv_path = Path(path)
    if not csv_path.exists():
        return {"heldin": {}, "heldout": {}}

    baseline = {"heldin": {}, "heldout": {}}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row.get("split", "")
            session = row.get("session", "")
            r2 = row.get("R2_variance_weighted", "")
            if not session or r2 in ("", None):
                continue
            if split in {"test_heldin", "heldin"}:
                baseline["heldin"][session] = float(r2)
            elif split in {"test_heldout", "heldout"}:
                baseline["heldout"][session] = float(r2)
    return baseline


def _delta_r2(candidate: Optional[float], baseline: Optional[float]) -> str:
    if candidate is None or baseline is None:
        return ""
    return f"{candidate - baseline:.8f}"


def _mean_session_delta(session_r2: Mapping[str, float], baseline: Mapping[str, float]) -> str:
    deltas = [session_r2[session] - baseline[session] for session in session_r2 if session in baseline]
    if not deltas:
        return ""
    return f"{sum(deltas) / len(deltas):.8f}"


def _provenance_fields(protocol: str, fold_id: Any) -> Dict[str, Any]:
    return {
        "validation_protocol": protocol,
        "fold_id": "" if fold_id is None else fold_id,
    }


def _design_row(
    run_id: str,
    variant: str,
    seed: int,
    profile: EncoderCostProfile,
    protocol: str = "",
    fold_id: Any = None,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        **_provenance_fields(protocol, fold_id),
        "split": "design",
        "parameter_count": profile.parameter_count,
        "weight_bytes": profile.weight_bytes,
        "MAC_per_trial": profile.mac_per_trial,
        "MAC_per_session": profile.mac_per_session,
        "peak_live_state_bytes": profile.peak_live_state_bytes,
        "trial_buffer_bytes": profile.trial_buffer_bytes,
        "support_state_bytes": profile.support_state_bytes,
        "requires_cubic_interpolation": profile.requires_cubic_interpolation,
        "requires_general_multiplier": profile.requires_general_multiplier,
        "requires_divider": profile.requires_divider,
        "quant_format": "fp32",
    }


def build_test_metric_rows(
    run_id: str,
    variant: str,
    seed: int,
    calibration_trials: int,
    parsed: ParsedTestMetrics,
    profile: EncoderCostProfile,
    baseline: Mapping[str, Mapping[str, float]],
    *,
    validation_protocol: str = "",
    fold_id: Any = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    provenance = _provenance_fields(validation_protocol, fold_id)
    summary_rows: List[Dict[str, Any]] = [_design_row(run_id, variant, seed, profile, validation_protocol, fold_id)]
    per_session_rows: List[Dict[str, Any]] = []

    aggregate_specs = [
        ("test_heldin", parsed.heldin_mean_r2, parsed.heldin_identity_mse, parsed.heldin_prediction_distill_mse, parsed.heldin_session_r2, "heldin"),
        ("test_heldout", parsed.heldout_mean_r2, parsed.heldout_identity_mse, parsed.heldout_prediction_distill_mse, parsed.heldout_session_r2, "heldout"),
    ]
    for split_name, mean_r2, identity_mse, pred_mse, session_r2, baseline_key in aggregate_specs:
        summary_rows.append(
            {
                "run_id": run_id,
                "variant": variant,
                "seed": seed,
                **provenance,
                "split": split_name,
                "M": calibration_trials,
                "R2_variance_weighted": "" if mean_r2 is None else f"{mean_r2:.8f}",
                "R2_delta_vs_matched_baseline": _mean_session_delta(session_r2, baseline.get(baseline_key, {})),
                "identity_mse": "" if identity_mse is None else f"{identity_mse:.8g}",
                "prediction_distill_mse": "" if pred_mse is None else f"{pred_mse:.8g}",
                "parameter_count": profile.parameter_count,
                "weight_bytes": profile.weight_bytes,
                "MAC_per_trial": profile.mac_per_trial,
                "MAC_per_session": profile.mac_per_session,
                "peak_live_state_bytes": profile.peak_live_state_bytes,
                "trial_buffer_bytes": profile.trial_buffer_bytes,
                "support_state_bytes": profile.support_state_bytes,
                "requires_cubic_interpolation": profile.requires_cubic_interpolation,
                "requires_general_multiplier": profile.requires_general_multiplier,
                "requires_divider": profile.requires_divider,
                "quant_format": "fp32",
            }
        )

    session_specs = [
        ("test_heldin", parsed.heldin_session_r2, parsed.heldin_session_identity_mse, parsed.heldin_session_prediction_distill_mse, "heldin"),
        ("test_heldout", parsed.heldout_session_r2, parsed.heldout_session_identity_mse, parsed.heldout_session_prediction_distill_mse, "heldout"),
    ]
    for split_name, session_r2, session_identity, session_pred, baseline_key in session_specs:
        for session, r2 in session_r2.items():
            per_session_rows.append(
                {
                    "run_id": run_id,
                    "variant": variant,
                    "seed": seed,
                    **provenance,
                    "split": split_name,
                    "session": session,
                    "M": calibration_trials,
                    "R2_variance_weighted": f"{r2:.8f}",
                    "R2_delta_vs_matched_baseline": _delta_r2(r2, baseline.get(baseline_key, {}).get(session)),
                    "identity_mse": "" if session not in session_identity else f"{session_identity[session]:.8g}",
                    "prediction_distill_mse": "" if session not in session_pred else f"{session_pred[session]:.8g}",
                    "parameter_count": profile.parameter_count,
                    "weight_bytes": profile.weight_bytes,
                    "MAC_per_trial": profile.mac_per_trial,
                    "MAC_per_session": profile.mac_per_session,
                    "peak_live_state_bytes": profile.peak_live_state_bytes,
                    "trial_buffer_bytes": profile.trial_buffer_bytes,
                    "support_state_bytes": profile.support_state_bytes,
                    "requires_cubic_interpolation": profile.requires_cubic_interpolation,
                    "requires_general_multiplier": profile.requires_general_multiplier,
                    "requires_divider": profile.requires_divider,
                    "quant_format": "fp32",
                }
            )

    return summary_rows, per_session_rows


def write_metrics_table(run_dir: Path, rows: Iterable[Mapping[str, Any]], filename: str, fields: List[str]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = run_dir / filename
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_checkpoint_manifest(
    run_dir: Path,
    source_ckpt_path: str | Path,
    *,
    selected_by_metric: str,
    selected_metric_value: Optional[float] = None,
    copy_checkpoint: bool = True,
) -> Path:
    source = Path(source_ckpt_path)
    manifest: Dict[str, Any] = {
        "source_checkpoint_path": str(source.resolve()),
        "source_checkpoint_sha256": checkpoint_sha256(source),
        "selected_by_metric": selected_by_metric,
        "selected_metric_value": selected_metric_value,
    }
    artifact_ckpt = run_dir / "checkpoints" / "best.ckpt"
    if copy_checkpoint and source.exists():
        artifact_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, artifact_ckpt)
        manifest["artifact_checkpoint_path"] = str(artifact_ckpt.resolve())
        manifest["artifact_checkpoint_sha256"] = checkpoint_sha256(artifact_ckpt)
    (run_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return artifact_ckpt


def ensure_run_dir(base_dir: Path | str, run_id: str) -> Path:
    base = Path(base_dir)
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def write_resolved_config(run_dir: Path, cfg: DictConfig) -> None:
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")


def write_environment(run_dir: Path) -> None:
    lines = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"python={subprocess.check_output(['python', '--version'], text=True).strip()}",
    ]
    try:
        import torch

        lines.append(f"torch={torch.__version__}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"torch_import_error={exc}")
    (run_dir / "environment.txt").write_text("\n".join(lines) + "\n")


def make_run_id(cfg: DictConfig) -> str:
    variant = str(cfg.model.get("variant", "run"))
    seed = int(cfg.get("seed", 0))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = cfg.get("run_id")
    fold = cfg.data.get("loso_fold") if hasattr(cfg, "data") else None
    fold_suffix = f"_f{fold}" if fold not in (None, "", "null") else ""

    if base in (None, "", "null"):
        return f"{variant}{fold_suffix}_s{seed}_{stamp}"
    return f"{base}{fold_suffix}_s{seed}_{stamp}"


def write_run_metadata(
    run_dir: Path,
    *,
    cfg: DictConfig,
    run_id: str,
    artifact_root: Path,
    split_manifest: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    selected_metric: str,
    selected_metric_value: Optional[float],
) -> None:
    from src.metrics.gate2_matrix import _preprocessing_label, extract_best_epoch

    model_cfg = cfg.model
    resolved_cfg = OmegaConf.to_container(cfg, resolve=False)
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "comparison_role": cfg.get("comparison_role"),
        "variant": str(model_cfg.get("variant", "")),
        "loss_mode": str(model_cfg.get("loss_mode", "")),
        "lambda_y": model_cfg.get("lambda_y"),
        "lambda_E": model_cfg.get("lambda_E"),
        "seed": int(cfg.get("seed", 0)),
        "fold_id": split_manifest.get("fold_id", cfg.data.get("loso_fold")),
        "validation_protocol": split_manifest.get("validation_protocol", cfg.data.get("validation_protocol", "")),
        "train_sessions": split_manifest.get("train_sessions", []),
        "validation_sessions": split_manifest.get("validation_sessions", []),
        "heldout_evaluated_in_fit": split_manifest.get("heldout_evaluated_in_fit"),
        "heldout_evaluated_in_test": split_manifest.get("heldout_evaluated_in_test"),
        "preprocessing": _preprocessing_label(resolved_cfg),
        "teacher_seen_validation_session": True,
        "selected_by_metric": selected_metric,
        "selected_metric_value": selected_metric_value,
        "best_epoch": extract_best_epoch(checkpoint_manifest),
        "artifact_dir": str(artifact_root.resolve()),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_split_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    (run_dir / "split_manifest.json").write_text(json.dumps(dict(manifest), indent=2) + "\n")


def write_baseline_reference(run_dir: Path, baseline_path: Path) -> Dict[str, str]:
    destination = run_dir / "baseline_reference.csv"
    shutil.copy2(baseline_path, destination)
    payload = {
        "baseline_metrics_path": str(baseline_path.resolve()),
        "baseline_metrics_sha256": checkpoint_sha256(baseline_path),
        "artifact_copy_path": str(destination.resolve()),
    }
    (run_dir / "baseline_reference.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_source_manifest(run_dir: Path, root_dir: Path) -> None:
    include_suffixes = {".py", ".yaml", ".yml", ".sh"}
    exclude_dirs = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "logs",
        "outputs",
        "third_party",
        ".hydra",
    }
    manifest: Dict[str, str] = {}
    for pattern in ("src", "configs", "scripts", "tests"):
        base = root_dir / pattern
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in include_suffixes:
                continue
            if any(part in exclude_dirs for part in path.parts):
                continue
            rel = path.relative_to(root_dir).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            manifest[rel] = digest.hexdigest()
    (run_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_git_state(run_dir: Path, repo_root: Path) -> None:
    roots = [repo_root, repo_root.parent]
    revision = "unknown"
    status = "git unavailable"
    for root in roots:
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)
            break
        except Exception as exc:  # noqa: BLE001
            status = f"git unavailable at {root}: {exc}"
    (run_dir / "git_state.txt").write_text(f"revision={revision}\n\n{status}\n")


def write_hardware_cost(run_dir: Path, profile: EncoderCostProfile, extra: Mapping[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {
        "variant": profile.variant,
        "parameter_count": profile.parameter_count,
        "weight_bytes": profile.weight_bytes,
        "trial_buffer_bytes": profile.trial_buffer_bytes,
        "support_state_bytes": profile.support_state_bytes,
        "peak_live_state_bytes": profile.peak_live_state_bytes,
        "mac_per_trial": profile.mac_per_trial,
        "mac_per_session": profile.mac_per_session,
        "requires_cubic_interpolation": profile.requires_cubic_interpolation,
        "requires_general_multiplier": profile.requires_general_multiplier,
        "requires_divider": profile.requires_divider,
        "cost_source": "cycle_model_estimate",
    }
    if extra:
        payload.update(dict(extra))
    (run_dir / "hardware_cost.json").write_text(json.dumps(payload, indent=2) + "\n")


def append_metrics_rows(run_dir: Path, rows: Iterable[Mapping[str, Any]], filename: str = "metrics_summary.csv") -> None:
    rows = list(rows)
    if not rows:
        return
    path = run_dir / filename
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRICS_SUMMARY_FIELDS})


def export_teacher_metadata(run_dir: Path, teacher_ckpt_path: str) -> None:
    from src.models.streaming_calibration_module import StreamingCalibrationLitModule

    meta = {
        "teacher_checkpoint_path": str(Path(teacher_ckpt_path).resolve()),
        "teacher_checkpoint_sha256": StreamingCalibrationLitModule.teacher_sha256(teacher_ckpt_path),
    }
    (run_dir / "teacher_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
