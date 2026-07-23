"""Gate2 revised experiment matrix aggregation and R1 decision helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

GATE2_MATRIX_FIELDS: List[str] = [
    "run_id",
    "comparison_role",
    "variant",
    "width_or_RK",
    "loss_mode",
    "lambda_y",
    "lambda_E",
    "seed",
    "fold_id",
    "train_sessions",
    "validation_session",
    "teacher_seen_validation_session",
    "preprocessing",
    "R2",
    "delta_fixed_B0",
    "delta_vs_D512_LOSO",
    "identity_mse",
    "prediction_distill_mse",
    "best_epoch",
    "parameter_count",
    "MAC_per_session",
    "peak_live_state_bytes",
    "trial_buffer_bytes",
    "baseline_sha256",
    "teacher_sha256",
    "checkpoint_sha256",
    "source_manifest_sha256",
    "decision_status",
    "registered_at_utc",
    "artifact_dir",
    "notes",
]

REQUIRED_R1_LOSS_MODES = ("task_only", "task_plus_y", "task_plus_y_plus_E")

LOSS_PRIORITY = {
    "task_only": 0,
    "task_plus_y": 1,
    "task_plus_y_plus_E": 2,
}

D512_GREEN = -0.01
D512_AMBER = -0.03
GATE2_GREEN = -0.01
GATE2_AMBER = -0.02
LOSS_TIE_R2 = 0.005
PROBE_GREEN = -0.03
PROBE_AMBER = -0.05


class ArtifactValidationError(ValueError):
    """Raised when a run artifact directory is incomplete or invalid."""


class AmbiguousMatrixError(ValueError):
    """Raised when multiple matrix rows match a required unique key."""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return _file_sha256(path)


def _finite_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        raise ArtifactValidationError(f"Non-finite metric value: {value!r}")
    return parsed


def extract_best_epoch(checkpoint_manifest: Mapping[str, Any]) -> Optional[int]:
    source = str(checkpoint_manifest.get("source_checkpoint_path", ""))
    match = re.search(r"epoch_(\d+)", source)
    if match:
        return int(match.group(1))
    artifact = str(checkpoint_manifest.get("artifact_checkpoint_path", ""))
    match = re.search(r"epoch_(\d+)", artifact)
    if match:
        return int(match.group(1))
    return None


def _preprocessing_label(resolved_cfg: Mapping[str, Any]) -> str:
    data = resolved_cfg.get("data", {})
    max_t = data.get("max_trial_length", 100)
    if data.get("interpolate_trials", True):
        return f"cubic_T{max_t}"
    if data.get("trial_feature_type") == "raw":
        return "raw_bins"
    return "unknown"


def _width_or_rk(variant: str, resolved_cfg: Mapping[str, Any]) -> str:
    model = resolved_cfg.get("model", {})
    variant = variant.upper()
    if variant == "B2":
        return f"D{model.get('id_hidden_dim', '')}"
    if variant == "B3":
        return f"D{model.get('hidden_dim', '')}"
    if variant == "B5":
        return f"R{model.get('num_emas', '')}"
    if variant == "B6":
        return f"R{model.get('num_filters', '')}K{model.get('kernel_size', '')}"
    if variant in {"B0", "B1"}:
        return "D512"
    if variant == "B4":
        return f"D{model.get('hidden_dim', 64)}"
    return ""


def _read_metrics_row(artifact_dir: Path, split: str = "test_heldin") -> Dict[str, str]:
    summary = artifact_dir / "metrics_summary.csv"
    if not summary.exists():
        return {}
    with summary.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == split:
                return row
    return {}


def _read_session_row(artifact_dir: Path, session: str, split: str = "test_heldin") -> Dict[str, str]:
    per_session = artifact_dir / "metrics_per_session.csv"
    if not per_session.exists():
        return {}
    with per_session.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == split and row.get("session") == session:
                return row
    return {}


def validation_session(artifact_dir: Path) -> str:
    split_manifest = _read_json(artifact_dir / "split_manifest.json")
    sessions = split_manifest.get("validation_sessions") or []
    if sessions:
        return str(sessions[0])
    per_session = artifact_dir / "metrics_per_session.csv"
    if per_session.exists():
        with per_session.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") == "test_heldin" and row.get("session"):
                    return str(row["session"])
    return ""


def validation_session_metrics(artifact_dir: Path) -> Tuple[Optional[float], Optional[float], str]:
    session = validation_session(artifact_dir)
    if not session:
        aggregate = _read_metrics_row(artifact_dir)
        return (
            _finite_float(aggregate.get("R2_variance_weighted")),
            _finite_float(aggregate.get("R2_delta_vs_matched_baseline")),
            "",
        )
    row = _read_session_row(artifact_dir, session)
    return (
        _finite_float(row.get("R2_variance_weighted")),
        _finite_float(row.get("R2_delta_vs_matched_baseline")),
        session,
    )


def decision_status_for_delta(delta: Optional[float], *, green: float, amber: float) -> str:
    if delta is None:
        return "unknown"
    if delta >= green:
        return "green"
    if delta >= amber:
        return "amber"
    return "red"


def _read_resolved_config(artifact_dir: Path) -> Dict[str, Any]:
    path = artifact_dir / "resolved_config.yaml"
    if not path.exists():
        return {}
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)  # type: ignore[return-value]


def _matrix_key(fold_id: Any, seed: Any) -> Tuple[str, str]:
    return str(fold_id), str(seed)


def validate_artifact_complete(artifact_dir: Path) -> Dict[str, Any]:
    """Validate that an artifact directory is complete enough for matrix registration."""
    artifact_dir = artifact_dir.resolve()
    if not artifact_dir.is_dir():
        raise ArtifactValidationError(f"Artifact path is not a directory: {artifact_dir}")

    required_files = [
        "resolved_config.yaml",
        "split_manifest.json",
        "metrics_per_session.csv",
        "metrics_summary.csv",
        "checkpoint_manifest.json",
        "baseline_reference.json",
        "teacher_metadata.json",
        "source_manifest.json",
        "hardware_cost.json",
    ]
    missing_files = [name for name in required_files if not (artifact_dir / name).exists()]
    if missing_files:
        raise ArtifactValidationError(f"Missing required artifact files: {missing_files}")

    resolved = _read_resolved_config(artifact_dir)
    split_manifest = _read_json(artifact_dir / "split_manifest.json")
    checkpoint_manifest = _read_json(artifact_dir / "checkpoint_manifest.json")
    baseline_ref = _read_json(artifact_dir / "baseline_reference.json")
    teacher_meta = _read_json(artifact_dir / "teacher_metadata.json")

    variant = str(resolved.get("model", {}).get("variant", "")).strip()
    if not variant:
        raise ArtifactValidationError("resolved_config.model.variant is missing")

    if split_manifest.get("validation_protocol") != "loso":
        raise ArtifactValidationError("split_manifest.validation_protocol must be loso")
    if split_manifest.get("heldout_evaluated_in_fit") is not False:
        raise ArtifactValidationError("heldout_evaluated_in_fit must be false")
    if split_manifest.get("heldout_evaluated_in_test") is not False:
        raise ArtifactValidationError("heldout_evaluated_in_test must be false")
    if not split_manifest.get("validation_sessions"):
        raise ArtifactValidationError("split_manifest.validation_sessions is empty")
    if len(split_manifest.get("train_sessions", [])) != 6:
        raise ArtifactValidationError("split_manifest.train_sessions must contain exactly six sessions")

    fold_id = split_manifest.get("fold_id", resolved.get("data", {}).get("loso_fold"))
    if fold_id in (None, ""):
        raise ArtifactValidationError("fold_id is missing from split manifest / resolved config")
    seed = resolved.get("seed")
    if seed in (None, ""):
        raise ArtifactValidationError("seed is missing from resolved config")

    session = validation_session(artifact_dir)
    if not session:
        raise ArtifactValidationError("validation session could not be resolved")

    r2, delta_b0, metric_session = validation_session_metrics(artifact_dir)
    if metric_session != session:
        raise ArtifactValidationError("metrics_per_session validation session mismatch")
    if r2 is None or delta_b0 is None:
        raise ArtifactValidationError(f"Finite validation R2/delta missing for session {session}")

    ckpt_path = checkpoint_manifest.get("artifact_checkpoint_path") or checkpoint_manifest.get(
        "source_checkpoint_path"
    )
    if not ckpt_path or not Path(str(ckpt_path)).exists():
        raise ArtifactValidationError("checkpoint path missing or does not exist")
    recorded_ckpt_hash = checkpoint_manifest.get("artifact_checkpoint_sha256", "")
    if not recorded_ckpt_hash:
        raise ArtifactValidationError("checkpoint_manifest.artifact_checkpoint_sha256 is missing")
    if _file_sha256(Path(str(ckpt_path))) != recorded_ckpt_hash:
        raise ArtifactValidationError("checkpoint hash does not match checkpoint_manifest")

    baseline_hash = baseline_ref.get("baseline_metrics_sha256", "")
    teacher_hash = teacher_meta.get("teacher_checkpoint_sha256", "")
    if not baseline_hash:
        raise ArtifactValidationError("baseline_reference.baseline_metrics_sha256 is missing")
    if not teacher_hash:
        raise ArtifactValidationError("teacher_metadata.teacher_checkpoint_sha256 is missing")

    selected_metric = checkpoint_manifest.get("selected_by_metric", "")
    if selected_metric != "val_heldin/r2_mean":
        raise ArtifactValidationError(f"Unexpected selected_by_metric: {selected_metric!r}")
    if extract_best_epoch(checkpoint_manifest) is None:
        raise ArtifactValidationError("best epoch could not be extracted from checkpoint manifest")

    return {
        "artifact_dir": str(artifact_dir),
        "variant": variant,
        "fold_id": fold_id,
        "seed": seed,
        "validation_session": session,
        "r2": r2,
        "delta_fixed_B0": delta_b0,
        "baseline_sha256": baseline_hash,
        "teacher_sha256": teacher_hash,
        "checkpoint_sha256": recorded_ckpt_hash,
    }


def build_matrix_row(
    artifact_dir: Path,
    *,
    comparison_role: str = "",
    d512_r2: Optional[float] = None,
    notes: str = "",
) -> Dict[str, Any]:
    validated = validate_artifact_complete(artifact_dir)
    artifact_dir = artifact_dir.resolve()
    resolved = _read_resolved_config(artifact_dir)
    run_metadata = _read_json(artifact_dir / "run_metadata.json")
    split_manifest = _read_json(artifact_dir / "split_manifest.json")
    checkpoint_manifest = _read_json(artifact_dir / "checkpoint_manifest.json")
    baseline_ref = _read_json(artifact_dir / "baseline_reference.json")
    teacher_meta = _read_json(artifact_dir / "teacher_metadata.json")
    hardware = _read_json(artifact_dir / "hardware_cost.json")
    metrics_row = _read_metrics_row(artifact_dir)

    variant = validated["variant"]
    model_cfg = resolved.get("model", {})
    r2 = validated["r2"]
    delta_b0 = validated["delta_fixed_B0"]
    validation_session_name = validated["validation_session"]
    fold_id = validated["fold_id"]
    seed = validated["seed"]

    delta_d512 = "" if d512_r2 is None else f"{r2 - d512_r2:.8f}"
    role = comparison_role or str(run_metadata.get("comparison_role") or resolved.get("comparison_role") or "")

    row: Dict[str, Any] = {
        "run_id": artifact_dir.name,
        "comparison_role": role,
        "variant": variant,
        "width_or_RK": _width_or_rk(variant, resolved),
        "loss_mode": str(model_cfg.get("loss_mode", "")),
        "lambda_y": model_cfg.get("lambda_y", ""),
        "lambda_E": model_cfg.get("lambda_E", ""),
        "seed": seed,
        "fold_id": fold_id,
        "train_sessions": json.dumps(split_manifest.get("train_sessions", []), separators=(",", ":")),
        "validation_session": validation_session_name,
        "teacher_seen_validation_session": "true",
        "preprocessing": _preprocessing_label(resolved),
        "R2": f"{r2:.8f}",
        "delta_fixed_B0": f"{delta_b0:.8f}",
        "delta_vs_D512_LOSO": delta_d512,
        "identity_mse": metrics_row.get("identity_mse", ""),
        "prediction_distill_mse": metrics_row.get("prediction_distill_mse", ""),
        "best_epoch": extract_best_epoch(checkpoint_manifest) or "",
        "parameter_count": hardware.get("parameter_count", metrics_row.get("parameter_count", "")),
        "MAC_per_session": hardware.get("mac_per_session", metrics_row.get("MAC_per_session", "")),
        "peak_live_state_bytes": hardware.get("peak_live_state_bytes", metrics_row.get("peak_live_state_bytes", "")),
        "trial_buffer_bytes": hardware.get("trial_buffer_bytes", metrics_row.get("trial_buffer_bytes", "")),
        "baseline_sha256": baseline_ref.get("baseline_metrics_sha256", ""),
        "teacher_sha256": teacher_meta.get("teacher_checkpoint_sha256", ""),
        "checkpoint_sha256": checkpoint_manifest.get("artifact_checkpoint_sha256", ""),
        "source_manifest_sha256": _manifest_sha256(artifact_dir / "source_manifest.json"),
        "decision_status": decision_status_for_delta(delta_b0, green=GATE2_GREEN, amber=GATE2_AMBER),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "notes": notes,
    }
    return row


def load_matrix(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_matrix(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GATE2_MATRIX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in GATE2_MATRIX_FIELDS})


def upsert_matrix_row(path: Path, row: Mapping[str, Any]) -> None:
    rows = load_matrix(path)
    run_id = str(row.get("run_id", ""))
    rows = [existing for existing in rows if existing.get("run_id") != run_id]
    rows.append(dict(row))
    write_matrix(path, rows)


def _protocol_control_rows(
    matrix_rows: Sequence[Mapping[str, Any]],
    *,
    fold_id: int,
    seed: int,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for row in matrix_rows:
        if row.get("comparison_role") != "protocol_control":
            continue
        if str(row.get("fold_id")) != str(fold_id):
            continue
        if str(row.get("seed")) != str(seed):
            continue
        if _finite_float(row.get("R2")) is None:
            continue
        matches.append(dict(row))
    return matches


def find_protocol_control_r2(
    matrix_rows: Sequence[Mapping[str, Any]],
    *,
    fold_id: int,
    seed: int,
) -> Optional[float]:
    matches = _protocol_control_rows(matrix_rows, fold_id=fold_id, seed=seed)
    if not matches:
        return None
    if len(matches) > 1:
        run_ids = [row.get("run_id", "") for row in matches]
        raise AmbiguousMatrixError(
            f"Multiple protocol_control rows for fold={fold_id}, seed={seed}: {run_ids}"
        )
    return _finite_float(matches[0].get("R2"))


def refresh_d512_deltas(path: Path) -> None:
    rows = load_matrix(path)
    control_r2: Dict[Tuple[str, str], float] = {}
    for row in rows:
        if row.get("comparison_role") != "protocol_control":
            continue
        key = _matrix_key(row.get("fold_id"), row.get("seed"))
        r2 = _finite_float(row.get("R2"))
        if r2 is None:
            continue
        if key in control_r2:
            raise AmbiguousMatrixError(f"Duplicate protocol_control for key {key}")
        control_r2[key] = r2

    updated: List[Dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if row.get("comparison_role") == "protocol_control":
            updated.append(row)
            continue
        key = _matrix_key(row.get("fold_id"), row.get("seed"))
        r2 = _finite_float(row.get("R2"))
        if r2 is not None and key in control_r2:
            row["delta_vs_D512_LOSO"] = f"{r2 - control_r2[key]:.8f}"
        updated.append(row)
    write_matrix(path, updated)


@dataclass
class R1Readiness:
    ready: bool
    missing_requirements: List[str] = field(default_factory=list)
    baseline_sha256: str = ""
    teacher_sha256: str = ""


@dataclass
class R1Decision:
    r1_ready: bool
    decision_state: str
    d512_delta: Optional[float]
    d512_status: str
    winning_loss: Optional[str]
    loss_r2: Dict[str, float]
    loss_delta: Dict[str, float]
    stop_architecture_sweep: bool
    missing_requirements: List[str]
    notes: List[str]


def _loss_rows(
    matrix_rows: Sequence[Mapping[str, Any]],
    *,
    fold_id: int,
    seed: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in matrix_rows:
        role = row.get("comparison_role", "")
        if role not in {"loss_ablation", "loss_ablation_reference"}:
            continue
        if str(row.get("fold_id")) != str(fold_id):
            continue
        if str(row.get("seed")) != str(seed):
            continue
        selected.append(dict(row))
    return selected


def _loss_row_for_mode(loss_rows: Sequence[Mapping[str, Any]], loss_mode: str) -> Optional[Dict[str, Any]]:
    matches = [row for row in loss_rows if row.get("loss_mode") == loss_mode]
    if len(matches) > 1:
        run_ids = [row.get("run_id", "") for row in matches]
        raise AmbiguousMatrixError(f"Multiple rows for loss_mode={loss_mode}: {run_ids}")
    return matches[0] if matches else None


def check_r1_readiness(
    matrix_rows: Sequence[Mapping[str, Any]],
    *,
    fold_id: int = 0,
    seed: int = 42,
) -> R1Readiness:
    missing: List[str] = []
    d512_matches = _protocol_control_rows(matrix_rows, fold_id=fold_id, seed=seed)
    if not d512_matches:
        missing.append(f"protocol_control fold={fold_id} seed={seed}")
    elif len(d512_matches) > 1:
        raise AmbiguousMatrixError(f"Multiple protocol_control rows for fold={fold_id}, seed={seed}")

    loss_rows = _loss_rows(matrix_rows, fold_id=fold_id, seed=seed)
    for loss_mode in REQUIRED_R1_LOSS_MODES:
        row = _loss_row_for_mode(loss_rows, loss_mode)
        if row is None:
            missing.append(f"loss_mode={loss_mode} fold={fold_id} seed={seed}")
            continue
        if _finite_float(row.get("R2")) is None:
            missing.append(f"finite R2 for loss_mode={loss_mode}")
        if _finite_float(row.get("delta_fixed_B0")) is None:
            missing.append(f"finite delta_fixed_B0 for loss_mode={loss_mode}")

    candidate_rows = d512_matches + loss_rows
    baseline_hashes = {str(row.get("baseline_sha256", "")) for row in candidate_rows if row.get("baseline_sha256")}
    teacher_hashes = {str(row.get("teacher_sha256", "")) for row in candidate_rows if row.get("teacher_sha256")}
    if not baseline_hashes:
        missing.append("baseline_sha256")
    if not teacher_hashes:
        missing.append("teacher_sha256")
    if len(baseline_hashes) > 1:
        missing.append("inconsistent baseline_sha256 across R1 rows")
    if len(teacher_hashes) > 1:
        missing.append("inconsistent teacher_sha256 across R1 rows")

    return R1Readiness(
        ready=not missing,
        missing_requirements=missing,
        baseline_sha256=next(iter(baseline_hashes), ""),
        teacher_sha256=next(iter(teacher_hashes), ""),
    )


def choose_winning_loss(loss_rows: Sequence[Mapping[str, Any]]) -> Tuple[Optional[str], str, List[str]]:
    notes: List[str] = []
    if not loss_rows:
        return None, "not_ready", ["No loss-ablation rows found in matrix."]

    ranked = sorted(
        loss_rows,
        key=lambda row: (_finite_float(row.get("R2")) or float("-inf")),
        reverse=True,
    )
    best = ranked[0]
    best_r2 = _finite_float(best.get("R2"))
    if best_r2 is None:
        return None, "not_ready", ["Loss rows exist but R2 is missing or non-finite."]

    ties = [
        row
        for row in ranked
        if _finite_float(row.get("R2")) is not None and abs(_finite_float(row.get("R2")) - best_r2) <= LOSS_TIE_R2
    ]
    if len(ties) == 1:
        notes.append(f"Selected {best['loss_mode']} with fold R2={best_r2:.6f}.")
        return str(best["loss_mode"]), "winner_selected", notes

    notes.append(
        f"Fold R2 tie within {LOSS_TIE_R2}: {[row['loss_mode'] for row in ties]}. "
        "Run seed=43 fold-0 tie-break before locking loss."
    )
    return None, "tie_requires_seed43", notes


def evaluate_r1(matrix_path: Path, *, fold_id: int = 0, seed: int = 42) -> R1Decision:
    rows = load_matrix(matrix_path)
    readiness = check_r1_readiness(rows, fold_id=fold_id, seed=seed)

    d512_row = _protocol_control_rows(rows, fold_id=fold_id, seed=seed)
    d512_delta = _finite_float(d512_row[0].get("delta_fixed_B0")) if len(d512_row) == 1 else None
    d512_status = decision_status_for_delta(d512_delta, green=D512_GREEN, amber=D512_AMBER)

    loss_rows = _loss_rows(rows, fold_id=fold_id, seed=seed)
    loss_r2 = {str(row["loss_mode"]): _finite_float(row.get("R2")) for row in loss_rows}
    loss_delta = {str(row["loss_mode"]): _finite_float(row.get("delta_fixed_B0")) for row in loss_rows}

    if not readiness.ready:
        return R1Decision(
            r1_ready=False,
            decision_state="not_ready",
            d512_delta=d512_delta,
            d512_status=d512_status,
            winning_loss=None,
            loss_r2={k: v for k, v in loss_r2.items() if v is not None},
            loss_delta={k: v for k, v in loss_delta.items() if v is not None},
            stop_architecture_sweep=False,
            missing_requirements=readiness.missing_requirements,
            notes=["R1 not ready; do not select loss or start conditional round."],
        )

    winning_loss, decision_state, notes = choose_winning_loss(
        [row for mode in REQUIRED_R1_LOSS_MODES if (row := _loss_row_for_mode(loss_rows, mode)) is not None]
    )

    all_loss_red = all(
        loss_delta.get(mode) is not None and loss_delta[mode] < D512_AMBER for mode in REQUIRED_R1_LOSS_MODES
    )
    stop = d512_delta is not None and d512_delta < D512_AMBER and all_loss_red
    if stop:
        notes = list(notes)
        notes.append("R1 stop: D512 < -0.03 and all B3 losses < -0.03. Pause architecture sweep.")
        decision_state = "stop_architecture_sweep"

    return R1Decision(
        r1_ready=True,
        decision_state=decision_state,
        d512_delta=d512_delta,
        d512_status=d512_status,
        winning_loss=winning_loss,
        loss_r2={k: v for k, v in loss_r2.items() if v is not None},
        loss_delta={k: v for k, v in loss_delta.items() if v is not None},
        stop_architecture_sweep=stop,
        missing_requirements=[],
        notes=notes,
    )


def loss_overrides(loss_mode: str) -> Dict[str, Any]:
    if loss_mode == "task_only":
        return {"loss_mode": "task_only", "lambda_y": 0.0, "lambda_E": 0.0}
    if loss_mode == "task_plus_y":
        return {"loss_mode": "task_plus_y", "lambda_y": 1.0, "lambda_E": 0.0}
    if loss_mode == "task_plus_y_plus_E":
        return {"loss_mode": "task_plus_y_plus_E", "lambda_y": 1.0, "lambda_E": 0.1}
    raise ValueError(f"Unsupported loss_mode: {loss_mode}")
