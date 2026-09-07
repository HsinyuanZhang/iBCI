#!/usr/bin/env python3
"""Fail-closed aggregation for the frozen M1 Version-B three-arm pilot.

This program is deliberately read-only with respect to its three input run
directories.  It accepts *explicit* H-S, B-C0 and B-C artifact directories,
validates their frozen fold-0/seed-42/M10 source-LOSO contract, and reports the
three paired terminal-epoch deltas.  It never discovers runs, loads a model,
selects an epoch, or opens a formal/minival/held-out split.

The ordinary streaming artifact writer does not serialize every query index or
the full sampler order.  Consequently the two pairing hashes emitted here are
canonical hashes of the immutable evidence that determines those objects:

* query hash: target file SHA plus the complete query-window audit;
* sampler hash: seed/configuration, ordered source sessions, batch counts and
  query batch count, with a common source-code manifest required across arms.

They are pairing receipts, not claims that a hidden sequence was recovered.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFLIGHT = (
    REPO_ROOT / "sua_exploration/results/m1_version_b_preflight/receipt_v2.json"
)

EXPECTED_TEACHER_CHECKPOINT_SHA256 = (
    "f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be"
)
EXPECTED_SOURCE_SESSIONS = (
    "ses-20120926",
    "ses-20120927",
    "ses-20120928",
)
EXPECTED_TARGET_SESSION = "ses-20120924"
EXPECTED_ELIGIBLE_QUERY_WINDOWS = 26_517
EXPECTED_EVALUATED_QUERY_WINDOWS = 26_496
EXPECTED_QUERY_BATCHES = 828
EXPECTED_BATCH_SIZE = 32
PRIMARY_GATE = 0.03


class ArtifactContractError(ValueError):
    """An input artifact does not satisfy the frozen Version-B contract."""


@dataclass(frozen=True)
class ArmSpec:
    key: str
    experiment: str
    carrier_arm: str
    variant: str


ARM_SPECS = {
    "hs": ArmSpec("hs", "m1_version_b_hs_continuation", "none", "B0"),
    "bc0": ArmSpec("bc0", "m1_version_b_c0", "zero4", "B3S"),
    "bc": ArmSpec("bc", "m1_version_b_c", "full", "B3S"),
}

REQUIRED_FILES = (
    "resolved_config.yaml",
    "split_manifest.json",
    "source_manifest.json",
    "teacher_metadata.json",
    "checkpoint_manifest.json",
    "run_metadata.json",
    "metrics_summary.csv",
    "metrics_per_session.csv",
)


def _fail(message: str) -> None:
    raise ArtifactContractError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"cannot read JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must contain an object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing YAML artifact: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactContractError(f"cannot read YAML artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"YAML artifact must contain a mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"missing CSV artifact: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ArtifactContractError(f"cannot read CSV artifact {path}: {exc}") from exc
    _require(bool(rows), f"CSV artifact is empty: {path}")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{label} must be a number, got {value!r}") from exc
    _require(math.isfinite(parsed), f"{label} must be finite, got {value!r}")
    return parsed


def _as_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{label} must be an integer, got {value!r}") from exc
    _require(str(value).strip() in {str(parsed), f"{parsed}.0"}, f"{label} is not integral: {value!r}")
    return parsed


def _expect(value: Any, expected: Any, label: str) -> None:
    _require(value == expected, f"{label}: expected {expected!r}, got {value!r}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be a sequence",
    )
    return value


def _verify_config(cfg: Mapping[str, Any], spec: ArmSpec, label: str) -> None:
    _expect(cfg.get("task_name"), spec.experiment, f"{label}.task_name")
    _expect(cfg.get("seed"), 42, f"{label}.seed")
    _expect(cfg.get("train"), True, f"{label}.train")
    _expect(cfg.get("test"), True, f"{label}.test")
    _expect(cfg.get("no_early_stopping"), True, f"{label}.no_early_stopping")
    _expect(cfg.get("optimized_metric"), "test_heldin/r2_mean", f"{label}.optimized_metric")

    data = _mapping(cfg.get("data"), f"{label}.data")
    _expect(str(data.get("task")).lower(), "m1", f"{label}.data.task")
    _expect(data.get("loso_fold"), 0, f"{label}.data.loso_fold")
    _expect(list(_sequence(data.get("source_session_names"), f"{label}.data.source_session_names")),
            list(EXPECTED_SOURCE_SESSIONS), f"{label}.data.source_session_names")
    _expect(data.get("calibration_n_trials"), 10, f"{label}.data.calibration_n_trials")
    _expect(data.get("random_calibration"), False, f"{label}.data.random_calibration")
    _expect(data.get("heldin_query_start_trial"), 10, f"{label}.data.heldin_query_start_trial")
    _expect(data.get("heldin_query_end_trial"), 210, f"{label}.data.heldin_query_end_trial")
    _expect(data.get("include_heldout_in_fit"), False, f"{label}.data.include_heldout_in_fit")
    _expect(data.get("include_heldout_in_test"), False, f"{label}.data.include_heldout_in_test")
    _expect(data.get("validation_protocol"), "loso", f"{label}.data.validation_protocol")
    _expect(data.get("afc4_arm"), spec.carrier_arm, f"{label}.data.afc4_arm")
    sampler_seed = data.get("sampler_seed")
    _require(
        sampler_seed == 42 or sampler_seed == "${seed}",
        f"{label}.data.sampler_seed: expected 42 or exact '${{seed}}', got {sampler_seed!r}",
    )
    _expect(data.get("batch_size"), EXPECTED_BATCH_SIZE, f"{label}.data.batch_size")
    _expect(data.get("balance_session_batches"), False, f"{label}.data.balance_session_batches")
    _expect(data.get("reshuffle_train_sampler_each_epoch"), False,
            f"{label}.data.reshuffle_train_sampler_each_epoch")

    model = _mapping(cfg.get("model"), f"{label}.model")
    _expect(model.get("variant"), spec.variant, f"{label}.model.variant")
    _expect(model.get("freeze_decoder"), False, f"{label}.model.freeze_decoder")
    _expect(model.get("loss_mode"), "task_only", f"{label}.model.loss_mode")
    _expect(float(model.get("lambda_y")), 0.0, f"{label}.model.lambda_y")
    _expect(float(model.get("lambda_E")), 0.0, f"{label}.model.lambda_E")

    trainer = _mapping(cfg.get("trainer"), f"{label}.trainer")
    _expect(trainer.get("min_epochs"), 12, f"{label}.trainer.min_epochs")
    _expect(trainer.get("max_epochs"), 12, f"{label}.trainer.max_epochs")
    _expect(trainer.get("limit_val_batches"), 0, f"{label}.trainer.limit_val_batches")
    _expect(trainer.get("num_sanity_val_steps"), 0, f"{label}.trainer.num_sanity_val_steps")

    callbacks = _mapping(cfg.get("callbacks"), f"{label}.callbacks")
    _expect(callbacks.get("early_stopping"), None, f"{label}.callbacks.early_stopping")
    fixed = _mapping(callbacks.get("fixed_last_checkpoint"),
                     f"{label}.callbacks.fixed_last_checkpoint")
    _expect(fixed.get("monitor"), None, f"{label}.checkpoint.monitor")
    _expect(fixed.get("save_last"), False, f"{label}.checkpoint.save_last")
    _expect(fixed.get("save_top_k"), -1, f"{label}.checkpoint.save_top_k")
    _expect(fixed.get("every_n_epochs"), 12, f"{label}.checkpoint.every_n_epochs")


def _verify_data_files(
    split: Mapping[str, Any], preflight: Mapping[str, Any], label: str,
    digest_cache: dict[Path, str],
) -> None:
    expected_files = _mapping(preflight.get("source_files"), "preflight.source_files")
    source_files = _mapping(split.get("source_files"), f"{label}.source_files")
    _expect(list(source_files), list(EXPECTED_SOURCE_SESSIONS), f"{label}.source file order")
    target_file = _mapping(split.get("target_file"), f"{label}.target_file")
    observed = dict(source_files)
    observed[EXPECTED_TARGET_SESSION] = target_file
    for session in (*EXPECTED_SOURCE_SESSIONS, EXPECTED_TARGET_SESSION):
        record = _mapping(observed.get(session), f"{label}.file[{session}]")
        expected = _mapping(expected_files.get(session), f"preflight.source_files[{session}]")
        _expect(record.get("sha256"), expected.get("sha256"), f"{label}.{session}.sha256")
        path = Path(str(record.get("path", ""))).expanduser()
        _require(path.is_file(), f"{label}.{session} recorded data file is unavailable: {path}")
        resolved = path.resolve()
        if resolved not in digest_cache:
            digest_cache[resolved] = _sha256_file(resolved)
        _expect(digest_cache[resolved], expected.get("sha256"), f"{label}.{session} on-disk SHA")
        lowered = path.as_posix().lower()
        _require(
            not any(token in lowered for token in ("minival", "held-out", "heldout", "formal", "evalai")),
            f"{label}.{session} points outside the allowed held-in-calib scope: {path}",
        )


def _verify_split(
    split: Mapping[str, Any], spec: ArmSpec, preflight: Mapping[str, Any],
    label: str, digest_cache: dict[Path, str],
) -> tuple[str, str]:
    _expect(split.get("schema"), "m1_version_b_source_loso_v1", f"{label}.schema")
    _expect(split.get("task"), "m1", f"{label}.task")
    _expect(split.get("outer_fold"), 0, f"{label}.outer_fold")
    _expect(split.get("outer_left_out"), EXPECTED_TARGET_SESSION, f"{label}.outer_left_out")
    _expect(list(_sequence(split.get("train_sessions"), f"{label}.train_sessions")),
            list(EXPECTED_SOURCE_SESSIONS), f"{label}.train_sessions")
    _expect(list(_sequence(split.get("validation_sessions"), f"{label}.validation_sessions")),
            [EXPECTED_TARGET_SESSION], f"{label}.validation_sessions")
    _expect(split.get("source_only"), True, f"{label}.source_only")
    for key in (
        "formal", "evalai", "heldout_files_opened", "heldout_values_used",
        "minival_files_opened", "minival_values_used", "target_backpropagation",
        "target_query_values_used_for_optimizer_or_checkpoint_selection",
    ):
        _expect(split.get(key), False, f"{label}.{key}")
    _expect(split.get("target_query_values_read_by_validation_or_evaluator"), True,
            f"{label}.target_query_values_read_by_validation_or_evaluator")
    _expect(split.get("carrier_arm"), spec.carrier_arm, f"{label}.carrier_arm")
    _expect(split.get("support_trials"), [0, 10], f"{label}.support_trials")
    _expect(split.get("query_trials"), [10, 210], f"{label}.query_trials")
    _expect(split.get("calibration_n_trials"), 10, f"{label}.calibration_n_trials")
    _expect(split.get("checkpoint_selection"), "fixed_last_epoch_11_train_source_only",
            f"{label}.checkpoint_selection")
    _verify_data_files(split, preflight, label, digest_cache)

    audits = _mapping(split.get("query_window_audit"), f"{label}.query_window_audit")
    _expect(list(audits), [EXPECTED_TARGET_SESSION], f"{label}.query audit sessions")
    target_audit = _mapping(audits[EXPECTED_TARGET_SESSION], f"{label}.target query audit")
    _expect(target_audit.get("support_trials"), 10, f"{label}.query support")
    _expect(target_audit.get("query_start_trial"), 10, f"{label}.query start")
    _expect(target_audit.get("query_end_trial"), 210, f"{label}.query end")
    _expect(target_audit.get("query_trials"), 200, f"{label}.query trial count")
    _expect(target_audit.get("eligible_windows"), EXPECTED_ELIGIBLE_QUERY_WINDOWS,
            f"{label}.eligible query window count")
    _expect(target_audit.get("full_window_disjoint"), True, f"{label}.query disjointness")
    _expect(target_audit.get("ineligible_reason"), None, f"{label}.query eligibility")

    target_file = _mapping(split["target_file"], f"{label}.target_file")
    query_hash = _canonical_sha256({
        "schema": "m1_version_b_query_pairing_v1",
        "target_session": EXPECTED_TARGET_SESSION,
        "target_file_sha256": target_file["sha256"],
        "query_window_audit": target_audit,
    })
    sampler_hash = _canonical_sha256({
        "schema": "m1_version_b_sampler_pairing_v1",
        "seed": 42,
        "source_sessions": list(EXPECTED_SOURCE_SESSIONS),
        "train_batch_counts": split.get("train_batch_counts"),
        "query_batch_count": split.get("query_batch_count"),
    })
    _require(isinstance(split.get("train_batch_counts"), Mapping),
             f"{label}.train_batch_counts must be present")
    query_batch_count = _as_int(split.get("query_batch_count"), f"{label}.query_batch_count")
    _expect(query_batch_count, EXPECTED_QUERY_BATCHES, f"{label}.query_batch_count")
    _expect(
        query_batch_count * EXPECTED_BATCH_SIZE,
        EXPECTED_EVALUATED_QUERY_WINDOWS,
        f"{label}.evaluated query window count",
    )
    return query_hash, sampler_hash


def _verify_teacher(
    teacher: Mapping[str, Any], cfg: Mapping[str, Any], label: str,
    digest_cache: dict[Path, str], expected_checkpoint_sha256: str,
) -> None:
    _expect(teacher.get("teacher_checkpoint_sha256"), expected_checkpoint_sha256,
            f"{label}.teacher checkpoint SHA")
    teacher_path = Path(str(teacher.get("teacher_checkpoint_path", ""))).expanduser()
    _require(teacher_path.is_file(), f"{label}.teacher checkpoint is unavailable: {teacher_path}")
    resolved = teacher_path.resolve()
    if resolved not in digest_cache:
        digest_cache[resolved] = _sha256_file(resolved)
    _expect(digest_cache[resolved], expected_checkpoint_sha256,
            f"{label}.teacher on-disk SHA")
    model = _mapping(cfg.get("model"), f"{label}.model")
    configured_raw = str(model.get("teacher_ckpt_path", ""))
    root_prefix = "${paths.root_dir}/"
    if configured_raw.startswith(root_prefix):
        paths = _mapping(cfg.get("paths"), f"{label}.paths")
        root_dir = Path(str(paths.get("root_dir", ""))).expanduser()
        _require(root_dir.is_absolute(), f"{label}.paths.root_dir must be absolute")
        configured = root_dir / configured_raw[len(root_prefix):]
    else:
        _require("${" not in configured_raw,
                 f"{label}.configured teacher contains an unsupported interpolation")
        configured = Path(configured_raw).expanduser()
    _require(configured.is_file(), f"{label}.configured teacher is unavailable: {configured}")
    _expect(configured.resolve(), resolved, f"{label}.configured teacher path")


def _terminal_epoch_from_path(path: str) -> int | None:
    name = Path(path).name
    # Lightning's default auto_insert_metric_name may render our
    # ``epoch_{epoch:03d}`` template as ``epoch_epoch=011.ckpt``.
    matches = re.findall(r"epoch(?:_epoch=|_|=)?(\d+)", name)
    return int(matches[-1]) if matches else None


def _verify_checkpoint(
    manifest: Mapping[str, Any], metadata: Mapping[str, Any], run_dir: Path, label: str,
) -> str:
    source_path = str(manifest.get("source_checkpoint_path", ""))
    _expect(_terminal_epoch_from_path(source_path), 11, f"{label}.terminal source checkpoint epoch")
    _expect(manifest.get("selected_metric_value"), None, f"{label}.selected_metric_value")
    selected_by = manifest.get("selected_by_metric")
    _require(
        selected_by in {
            "val_heldin/r2_mean",  # generic writer fallback when monitor=None
            "fixed_last_epoch_11_train_source_only",
            "fixed_terminal_epoch_11",
        },
        f"{label}.selected_by_metric is not a fixed-terminal writer value: {selected_by!r}",
    )
    source_sha = manifest.get("source_checkpoint_sha256")
    artifact_sha = manifest.get("artifact_checkpoint_sha256")
    _expect(source_sha, artifact_sha, f"{label}.source/artifact checkpoint SHA")
    _require(isinstance(artifact_sha, str) and len(artifact_sha) == 64,
             f"{label}.artifact checkpoint SHA is missing")
    # The writer copies to checkpoints/best.ckpt.  Resolve there rather than
    # trusting an absolute path that may refer to another machine.
    artifact_copy = run_dir / "checkpoints" / "best.ckpt"
    _require(artifact_copy.is_file(), f"{label}.terminal artifact copy is missing: {artifact_copy}")
    _expect(_sha256_file(artifact_copy), artifact_sha, f"{label}.terminal checkpoint on-disk SHA")

    # The generic run writer's legacy epoch parser understands epoch_011 but
    # not Lightning's epoch_epoch=011 spelling.  The source checkpoint above
    # remains authoritative; if metadata did parse an epoch, it must agree.
    _require(metadata.get("best_epoch") in (None, 11),
             f"{label}.run_metadata.best_epoch: expected 11 or legacy null, "
             f"got {metadata.get('best_epoch')!r}")
    _expect(metadata.get("selected_metric_value"), None, f"{label}.run_metadata.selected_metric_value")
    _expect(metadata.get("selected_by_metric"), selected_by, f"{label}.run_metadata.selected_by_metric")
    _expect(metadata.get("no_early_stopping"), True, f"{label}.run_metadata.no_early_stopping")
    _expect(metadata.get("max_epochs"), 12, f"{label}.run_metadata.max_epochs")
    return artifact_sha


def _metric_identity(row: Mapping[str, str], spec: ArmSpec, run_id: str, label: str) -> None:
    _expect(row.get("run_id"), run_id, f"{label}.run_id")
    _expect(row.get("variant"), spec.variant, f"{label}.variant")
    _expect(_as_int(row.get("seed"), f"{label}.seed"), 42, f"{label}.seed")
    _expect(row.get("validation_protocol"), "loso", f"{label}.validation_protocol")
    _expect(_as_int(row.get("fold_id"), f"{label}.fold_id"), 0, f"{label}.fold_id")
    _expect(_as_int(row.get("M"), f"{label}.M"), 10, f"{label}.M")


def _verify_metrics(run_dir: Path, spec: ArmSpec, metadata: Mapping[str, Any], label: str) -> float:
    summary = _read_csv(run_dir / "metrics_summary.csv")
    heldin = [row for row in summary if row.get("split") == "test_heldin"]
    _expect(len(heldin), 1, f"{label}.terminal held-in summary row count")
    # A generic writer may emit an empty held-out summary row.  Any numeric
    # held-out value, or any other scored split, is forbidden.
    for row in summary:
        if row is heldin[0]:
            continue
        _require(row.get("R2_variance_weighted") in (None, ""),
                 f"{label} contains an out-of-scope scored summary row: {row.get('split')!r}")
    run_id = str(metadata.get("run_id", ""))
    _expect(run_id, run_dir.name, f"{label}.run_metadata.run_id")
    _metric_identity(heldin[0], spec, run_id, f"{label}.metrics_summary")
    score = _finite_float(heldin[0].get("R2_variance_weighted"), f"{label}.R2")

    per_session = _read_csv(run_dir / "metrics_per_session.csv")
    scored = [row for row in per_session if row.get("R2_variance_weighted") not in (None, "")]
    _expect(len(scored), 1, f"{label}.scored per-session row count")
    row = scored[0]
    _expect(row.get("split"), "test_heldin", f"{label}.per_session.split")
    _expect(row.get("session"), EXPECTED_TARGET_SESSION, f"{label}.per_session.session")
    _metric_identity(row, spec, run_id, f"{label}.metrics_per_session")
    session_score = _finite_float(row.get("R2_variance_weighted"), f"{label}.session R2")
    _require(abs(score - session_score) <= 5e-9,
             f"{label}.summary/session R2 mismatch: {score} vs {session_score}")
    return score


def _load_arm(
    run_dir: Path, spec: ArmSpec, preflight: Mapping[str, Any], digest_cache: dict[Path, str],
    expected_teacher_checkpoint_sha256: str,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    label = spec.key
    _require(run_dir.is_dir(), f"{label} run directory does not exist: {run_dir}")
    for filename in REQUIRED_FILES:
        _require((run_dir / filename).is_file(), f"{label} missing required artifact {filename}")
    cfg = _read_yaml(run_dir / "resolved_config.yaml")
    split = _read_json(run_dir / "split_manifest.json")
    source_manifest = _read_json(run_dir / "source_manifest.json")
    teacher = _read_json(run_dir / "teacher_metadata.json")
    checkpoint = _read_json(run_dir / "checkpoint_manifest.json")
    metadata = _read_json(run_dir / "run_metadata.json")

    _verify_config(cfg, spec, label)
    query_hash, sampler_hash = _verify_split(split, spec, preflight, label, digest_cache)
    _verify_teacher(
        teacher, cfg, label, digest_cache, expected_teacher_checkpoint_sha256
    )
    checkpoint_sha = _verify_checkpoint(checkpoint, metadata, run_dir, label)
    _expect(metadata.get("variant"), spec.variant, f"{label}.run_metadata.variant")
    _expect(metadata.get("seed"), 42, f"{label}.run_metadata.seed")
    _expect(metadata.get("fold_id"), 0, f"{label}.run_metadata.fold_id")
    _expect(metadata.get("validation_protocol"), "loso", f"{label}.run_metadata.validation_protocol")
    _expect(metadata.get("train_sessions"), list(EXPECTED_SOURCE_SESSIONS),
            f"{label}.run_metadata.train_sessions")
    _expect(metadata.get("validation_sessions"), [EXPECTED_TARGET_SESSION],
            f"{label}.run_metadata.validation_sessions")
    score = _verify_metrics(run_dir, spec, metadata, label)
    return {
        "run_dir": str(run_dir),
        "score": score,
        "query_pairing_sha256": query_hash,
        "sampler_pairing_sha256": sampler_hash,
        "source_manifest": source_manifest,
        "source_manifest_sha256": _sha256_file(run_dir / "source_manifest.json"),
        "checkpoint_sha256": checkpoint_sha,
        "model_config": dict(_mapping(cfg.get("model"), f"{label}.model")),
        "teacher_checkpoint_path": teacher["teacher_checkpoint_path"],
    }


def aggregate(
    hs_run_dir: Path, bc0_run_dir: Path, bc_run_dir: Path,
    preflight_receipt: Path = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    """Validate and aggregate one explicit three-arm Version-B cell."""
    run_dirs = {
        "hs": Path(hs_run_dir).expanduser().resolve(),
        "bc0": Path(bc0_run_dir).expanduser().resolve(),
        "bc": Path(bc_run_dir).expanduser().resolve(),
    }
    _require(len(set(run_dirs.values())) == 3, "the three run directories must be distinct")
    preflight = _read_json(Path(preflight_receipt).expanduser().resolve())
    _expect(preflight.get("schema"), "m1_version_b_preflight_v2", "preflight.schema")
    scope = _mapping(preflight.get("scope"), "preflight.scope")
    _expect(scope.get("fold"), 0, "preflight.scope.fold")
    _expect(scope.get("source_sessions"), list(EXPECTED_SOURCE_SESSIONS),
            "preflight.scope.source_sessions")
    _expect(scope.get("target_session"), EXPECTED_TARGET_SESSION, "preflight.scope.target_session")
    for key in ("formal_test_opened", "heldout_files_opened", "minival_files_opened",
                "target_backpropagation"):
        _expect(scope.get(key), False, f"preflight.scope.{key}")

    model_inv = _mapping(preflight.get("model_invariants"), "preflight.model_invariants")
    checks = _mapping(preflight.get("checks"), "preflight.checks")
    expected_teacher_checkpoint_sha256 = checks.get(
        "teacher_checkpoint_sha256", EXPECTED_TEACHER_CHECKPOINT_SHA256
    )
    _require(
        isinstance(expected_teacher_checkpoint_sha256, str)
        and len(expected_teacher_checkpoint_sha256) == 64,
        "preflight teacher checkpoint SHA is invalid",
    )
    c0_init = model_inv.get("b3s_c0_initial_encoder_sha256")
    c_init = model_inv.get("b3s_c_initial_encoder_sha256")
    _require(isinstance(c0_init, str) and len(c0_init) == 64,
             "preflight B-C0 initial encoder SHA is missing")
    _expect(c_init, c0_init, "preflight B-C/B-C0 initial encoder SHA")
    decoder_hashes = _mapping(model_inv.get("decoder_sha256"), "preflight.decoder_sha256")
    expected_decoder = model_inv.get("teacher_net_sha256")
    _require(isinstance(expected_decoder, str) and len(expected_decoder) == 64,
             "preflight teacher decoder SHA is missing")
    for spec in ARM_SPECS.values():
        _expect(decoder_hashes.get(spec.experiment), expected_decoder,
                f"preflight decoder SHA for {spec.experiment}")

    digest_cache: dict[Path, str] = {}
    arms = {
        key: _load_arm(
            run_dirs[key], ARM_SPECS[key], preflight, digest_cache,
            expected_teacher_checkpoint_sha256,
        )
        for key in ("hs", "bc0", "bc")
    }

    for field in ("query_pairing_sha256", "sampler_pairing_sha256"):
        values = {key: arm[field] for key, arm in arms.items()}
        _require(len(set(values.values())) == 1, f"three-arm {field} mismatch: {values}")
    source_manifests = {key: arm["source_manifest"] for key, arm in arms.items()}
    _require(source_manifests["hs"] == source_manifests["bc0"] == source_manifests["bc"],
             "three arms were not run from the same source manifest")
    _expect(arms["bc0"]["model_config"], arms["bc"]["model_config"],
            "B-C0/B-C resolved model configuration")
    _expect(arms["bc0"]["teacher_checkpoint_path"], arms["bc"]["teacher_checkpoint_path"],
            "B-C0/B-C teacher path")

    scores = {key: arms[key]["score"] for key in arms}
    deltas = {
        "bc_minus_bc0": scores["bc"] - scores["bc0"],
        "bc_minus_hs": scores["bc"] - scores["hs"],
        "bc0_minus_hs": scores["bc0"] - scores["hs"],
    }
    return {
        "schema": "m1_version_b_pilot_aggregate_v1",
        "status": "valid_terminal_three_arm_result",
        "scope": {
            "task": "m1",
            "fold": 0,
            "seed": 42,
            "M": 10,
            "support_trials": [0, 10],
            "query_trials": [10, 210],
            "target_session": EXPECTED_TARGET_SESSION,
            "eligible_query_window_count": EXPECTED_ELIGIBLE_QUERY_WINDOWS,
            "test_sample_count": EXPECTED_EVALUATED_QUERY_WINDOWS,
            "dropped_tail_window_count": (
                EXPECTED_ELIGIBLE_QUERY_WINDOWS - EXPECTED_EVALUATED_QUERY_WINDOWS
            ),
            "checkpoint_epoch": 11,
            "development_only": True,
            "formal_test_opened": False,
            "heldout_opened": False,
            "minival_opened": False,
        },
        "pairing": {
            "hash_semantics": "canonical_immutable_artifact_contract_v1",
            "query_sha256": arms["hs"]["query_pairing_sha256"],
            "sampler_sha256": arms["hs"]["sampler_pairing_sha256"],
            "source_manifest_sha256_by_arm": {
                key: arms[key]["source_manifest_sha256"] for key in arms
            },
            "teacher_checkpoint_sha256": expected_teacher_checkpoint_sha256,
            "teacher_decoder_state_sha256": expected_decoder,
            "bc0_bc_initial_encoder_sha256": c0_init,
        },
        "arms": {
            key: {
                "run_dir": arms[key]["run_dir"],
                "R2": scores[key],
                "checkpoint_sha256": arms[key]["checkpoint_sha256"],
            }
            for key in arms
        },
        "paired_deltas": deltas,
        "primary_gate": {
            "metric": "B-C - B-C0",
            "threshold": PRIMARY_GATE,
            "value": deltas["bc_minus_bc0"],
            "passed": deltas["bc_minus_bc0"] >= PRIMARY_GATE,
        },
        "selection_rule": "fixed terminal epoch 11 only; no per-epoch argmax",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hs-run-dir", type=Path, required=True)
    parser.add_argument("--bc0-run-dir", type=Path, required=True)
    parser.add_argument("--bc-run-dir", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument(
        "--output", type=Path,
        help="optional fresh JSON output path; stdout is always emitted",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = aggregate(
            args.hs_run_dir, args.bc0_run_dir, args.bc_run_dir, args.preflight_receipt
        )
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            output = args.output.expanduser().resolve()
            if output.exists():
                _fail(f"refusing to overwrite existing output: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
        sys.stdout.write(encoded)
        return 0
    except ArtifactContractError as exc:
        print(f"M1 Version-B aggregation rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
