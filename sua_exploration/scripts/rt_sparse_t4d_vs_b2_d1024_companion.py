#!/usr/bin/env python3
"""Fail-closed RT T4d versus matched SPINT-structured B2-D1024 companion.

This is a read-only terminal consumer.  It cannot launch training/evaluation,
write a receipt, or modify the frozen Stage-2 matrix.  It first invokes the
independent Stage-2 verifier, then validates legacy B2-D1024 artifacts fold by
fold. Historic B2 outer receipts predate strong query digests. A current-code
CPU replay establishes *implementation compatibility*, not historical ordered
query identity. This consumer accepts a historical B2 score only with an
archived-source reconstruction witness (currently fold 0) or a future
forward-only re-evaluation receipt. Equal window count is never a substitute
for digest equality.

The default prospective subset is folds 4--14.  Fold 3 can join only via an
externally supplied immutable pre-protocol unopened-score attestation; this
tool never infers it from a timestamp or a human assertion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

import yaml

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import rt_sparse_endpoint_stage2_terminal_verify as stage2


FOLDS = tuple(range(15))
PROSPECTIVE_DEFAULT = tuple(range(4, 15))
T4D = "rt_sparse_endpoint_t4d"
B2 = "zero4"
SEED = 42
B2_AGGREGATE = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/RT_STAGE_R_D1024_FULL15_AGGREGATE_v1.json"
B2_AGGREGATE_SHA256 = "95bca578cd9ac412c88eb29b96e22c3eda5968ecb39b8f21dfc8c3fff5b536b8"
DEFAULT_NWB_ROOT = WORKSPACE / "sua_exploration/data/dandi_000688/sub-C"
FOLD0_REMOTE_RECONSTRUCTION = (
    WORKSPACE
    / "sua_exploration/results/rt_sparse_t4d_b2_companion_v1"
    / "RT_B2_FOLD0_REMOTE_PROVENANCE_RECONSTRUCTION_v1.json"
)


class CompanionError(ValueError):
    """A missing legacy fact is an error, not a permissive warning."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CompanionError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"missing JSON artifact: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(result, dict), f"JSON root must be an object: {path}")
    return result


def _yaml(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"missing YAML artifact: {path}")
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    _need(isinstance(result, dict), f"YAML root must be a mapping: {path}")
    return result


def _finite(value: Any, label: str) -> float:
    _need(isinstance(value, (float, int)) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _summary(values: Mapping[str, float]) -> dict[str, Any]:
    _need(bool(values), "empty paired contrast")
    names = sorted(values)
    scores = [float(values[name]) for name in names]
    mean = sum(scores) / len(scores)
    ordered = sorted(scores)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    removed = max(range(len(scores)), key=lambda index: (abs(scores[index]), -index))
    kept = scores[:removed] + scores[removed + 1 :]
    positive = sum(score > 0.0 for score in scores)
    return {
        "ordered": [{"session": session, "delta": float(values[session])} for session in names],
        "n": len(scores), "mean": mean, "median": median,
        "positive": positive, "zero": sum(score == 0.0 for score in scores),
        "negative": sum(score < 0.0 for score in scores),
        "leave_largest_absolute_out_mean": sum(kept) / len(kept) if kept else None,
        "removed_session": names[removed],
        "mean_ge_003": mean >= 0.03, "median_ge_003": median >= 0.03,
        "gate_mean_positive_median_positive_sign_majority": bool(
            mean > 0.0 and median > 0.0 and positive > len(scores) / 2.0
        ),
    }


def _legacy_b2_rows(path: Path = B2_AGGREGATE) -> dict[int, dict[str, Any]]:
    """Return immutable legacy B2 rows, including their original file hashes."""

    _need(path.is_file(), f"missing frozen B2 aggregate: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, "B2 aggregate must be immutable mode 0444")
    _need(_sha(path) == B2_AGGREGATE_SHA256, "B2 aggregate SHA differs from frozen 95bca receipt")
    receipt = _json(path)
    _need(receipt.get("schema") == "rt_stage_r_rc_vs_rs_b2_d1024_paired_aggregate_v1", "B2 aggregate schema drift")
    _need(receipt.get("status") == "PASS_RT_STAGE_R_RC_VS_RS_B2_D1024_ALL_15_PAIRED", "B2 aggregate status drift")
    rows: dict[int, dict[str, Any]] = {}
    for row in receipt.get("rows", []):
        _need(isinstance(row, Mapping), "B2 aggregate row is not an object")
        fold = row.get("fold")
        _need(isinstance(fold, int) and fold in FOLDS and fold not in rows, "B2 aggregate fold grid drift")
        _need(isinstance(row.get("r_s_files"), Mapping), f"B2 fold {fold} file bindings missing")
        _need(_finite(row.get("r_s_b2_d1024_r2"), f"B2 fold {fold} R2") is not None, "unreachable")
        if fold == 10:
            _need(
                row.get("r_s_source") == "local_3090_fold10_race_recovery_v1",
                "B2 fold 10 is not bound to the isolated race-recovery cell",
            )
        rows[fold] = dict(row)
    _need(tuple(sorted(rows)) == FOLDS, "B2 aggregate does not bind exactly 15 folds")
    return rows


def _bound_legacy_file(row: Mapping[str, Any], name: str) -> Path:
    files = row.get("r_s_files")
    _need(isinstance(files, Mapping) and isinstance(files.get(name), Mapping), f"B2 {name} file binding missing")
    binding = files[name]
    path = Path(str(binding.get("path", "")))
    _need(path.is_file(), f"B2 bound {name} file missing: {path}")
    _need(_sha(path) == binding.get("sha256"), f"B2 bound {name} SHA drift")
    return path


def _teacher_metadata_for_fold(fold: int) -> Path | None:
    """Find an existing per-run immutable teacher SHA witness, never fabricate one."""

    roots = (
        WORKSPACE / "streaming_calibration_exp/outputs/rt_stage_r_b2_local3090/gpu_runs_zero4_v2/_artifacts",
        WORKSPACE / "streaming_calibration_exp/outputs/rt_stage_r_b2_local3090/supervisor_folds03_14_v1/_artifacts",
        WORKSPACE / "streaming_calibration_exp/outputs/rt_stage_r_b2_local3090/fold10_race_recovery_v1/_artifacts",
        WORKSPACE / "streaming_calibration_exp/outputs/rt_stage_r_b2_imported_remote/_artifacts",
    )
    marker = f"_f{fold}_s42_"

    # Fold 10 is exceptional for a documented reason: the immutable aggregate
    # binds it to the isolated race-recovery run.  A supervisor witness from
    # the raced run is evidence of an alternative execution, not interchangeable
    # evidence.  Look *only* in that frozen root, and retain the normal
    # uniqueness requirement within it.  In particular, do not fall back to a
    # supervisor witness if the recovery witness is absent.
    if fold == 10:
        recovery_root = roots[2]
        candidates = [p for p in recovery_root.glob(f"*{marker}*/teacher_metadata.json")] if recovery_root.is_dir() else []
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise CompanionError(
                f"B2 fold 10: ambiguous race-recovery teacher metadata witnesses: {candidates}"
            )
        raise CompanionError(
            "B2 fold 10: missing freeze-bound race-recovery teacher metadata witness"
        )

    candidates = [p for root in roots if root.is_dir() for p in root.glob(f"*{marker}*/teacher_metadata.json")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise CompanionError(f"B2 fold {fold}: ambiguous teacher metadata witnesses: {candidates}")
    return None


def _validate_b2_self(fold: int, row: Mapping[str, Any], *, stage_teacher_sha: str,
                      nwb_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Audit a legacy B2 source fit without assuming that old receipts are modern."""

    outer_path = _bound_legacy_file(row, "outer")
    selection_path = _bound_legacy_file(row, "selection")
    split_path = _bound_legacy_file(row, "split")
    config_path = _bound_legacy_file(row, "config")
    outer, selection, split, config = _json(outer_path), _json(selection_path), _json(split_path), _yaml(config_path)
    _need((outer.get("arm"), outer.get("outer_loso_fold"), outer.get("seed")) == (B2, fold, SEED), f"B2 fold {fold}: outer identity mismatch")
    _need((selection.get("arm"), selection.get("outer_loso_fold"), selection.get("seed")) == (B2, fold, SEED), f"B2 fold {fold}: selection identity mismatch")
    _need(outer.get("status") == "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP", f"B2 fold {fold}: outer did not pass")
    _need(selection.get("status") == "PASS_FIT_INNER_SELECTION_ONLY", f"B2 fold {fold}: selection did not pass")
    _need(selection.get("selected_by_metric") == "val_heldin/r2_mean" and selection.get("selected_metric_scope") == "inner_validation_session_only", f"B2 fold {fold}: checkpoint rule mismatch")
    _need(selection.get("formal_heldout_opened") is False and selection.get("outer_target_loaded_during_fit") is False and selection.get("outer_target_query_labels_read_during_fit") is False, f"B2 fold {fold}: fit target exclusion failed")
    _need(split.get("validation_protocol") == "nested_loso" and split.get("outer_loso_fold") == fold, f"B2 fold {fold}: split protocol mismatch")
    _need(split.get("requested_side_feature_group") == B2, f"B2 fold {fold}: not zero4")
    _need(split.get("calibration", {}).get("budget_trials") == 24, f"B2 fold {fold}: M24 mismatch")
    _need(split.get("query", {}).get("query_start_trial") == 24 and split.get("query", {}).get("window_size_bins") == 50, f"B2 fold {fold}: q/W mismatch")
    for name in ("target_backpropagation", "optimizer_present", "model_training_mode", "target_query_labels_used_for_calibration", "target_query_labels_used_for_normalization", "target_query_labels_used_for_checkpoint_selection"):
        _need(outer.get(name) is False, f"B2 fold {fold}: target-state contract failed: {name}")
    _need(outer.get("model_state_unchanged") is True, f"B2 fold {fold}: target state changed")
    _need(outer.get("model_state_sha256_before") == outer.get("model_state_sha256_after"), f"B2 fold {fold}: state SHA mismatch")

    data, model, trainer = config.get("data"), config.get("model"), config.get("trainer")
    _need(isinstance(data, Mapping) and isinstance(model, Mapping) and isinstance(trainer, Mapping), f"B2 fold {fold}: config sections missing")
    _need(config.get("seed") == SEED and config.get("no_early_stopping") is True, f"B2 fold {fold}: seed/early-stop drift")
    _need(config.get("optimized_metric") == "val_heldin/r2_mean", f"B2 fold {fold}: monitor drift")
    for field, expected in (("calibration_n_trials", 24), ("window_size", 50), ("max_trial_length", 100), ("session_window_budget", 4096), ("loso_fold", fold), ("outer_loso_fold", fold), ("side_feature_group", B2)):
        _need(data.get(field) == expected, f"B2 fold {fold}: config {field} mismatch")
    _need(data.get("query_start_trial") in (24, "${.calibration_n_trials}"), f"B2 fold {fold}: config query start mismatch")
    _need(data.get("sampler_seed") in (42, "${seed}"), f"B2 fold {fold}: sampler seed mismatch")
    _need(data.get("random_calibration") is False and data.get("smooth_calibration") is False, f"B2 fold {fold}: calibration chronology mismatch")
    _need(trainer.get("max_epochs") == 35, f"B2 fold {fold}: epoch budget mismatch")
    _need(model.get("variant") == "B2" and model.get("id_hidden_dim") == 1024 and model.get("freeze_decoder") is False, f"B2 fold {fold}: B2-D1024/joint-decoder mismatch")
    optimizer = model.get("optimizer")
    _need(isinstance(optimizer, Mapping) and optimizer.get("_target_") == "torch.optim.Adam" and float(optimizer.get("lr")) == 1e-4, f"B2 fold {fold}: optimizer/lr mismatch")

    target = str(outer.get("outer_target_session", ""))
    _need(target == split.get("target_session"), f"B2 fold {fold}: outer target/split mismatch")
    nwb = nwb_rows.get(target)
    _need(isinstance(nwb, Mapping), f"B2 fold {fold}: target absent from Stage2 NWB allowlist")
    nwb_path = DEFAULT_NWB_ROOT / Path(str(nwb["path"])).name
    _need(nwb_path.is_file() and _sha(nwb_path) == nwb.get("sha256"), f"B2 fold {fold}: target NWB SHA mismatch")
    metadata = _teacher_metadata_for_fold(fold)
    teacher_path: Path
    if metadata is not None:
        teacher = _json(metadata)
        teacher_path = metadata
    elif fold == 0:
        # The historical remote run was imported without a local teacher witness.
        # Its immutable CPU-only reconstruction is the only accepted substitute;
        # do not weaken the other folds' requirement.
        _need(FOLD0_REMOTE_RECONSTRUCTION.is_file(), "B2 fold 0: remote reconstruction receipt missing")
        _need(stat.S_IMODE(FOLD0_REMOTE_RECONSTRUCTION.stat().st_mode) == 0o444, "B2 fold 0: reconstruction receipt must be immutable mode 0444")
        reconstruction = _json(FOLD0_REMOTE_RECONSTRUCTION)
        _need(
            reconstruction.get("schema") == "rt_sparse_t4d_b2_fold0_remote_provenance_reconstruction_v1"
            and reconstruction.get("status") == "PASS_CRYPTOGRAPHIC_REMOTE_RECONSTRUCTION_LIMITED_HISTORICAL_SOURCE_ATTESTATION",
            "B2 fold 0: reconstruction receipt status/schema mismatch",
        )
        teacher = reconstruction.get("remote_teacher_metadata")
        _need(isinstance(teacher, Mapping), "B2 fold 0: reconstruction has no remote teacher witness")
        teacher_path = FOLD0_REMOTE_RECONSTRUCTION
    else:
        raise CompanionError(f"B2 fold {fold}: no immutable teacher-SHA witness; forward-only re-eval or archived attestation required")
    _need(teacher.get("teacher_checkpoint_sha256") == stage_teacher_sha, f"B2 fold {fold}: teacher SHA mismatch")
    return {"fold": fold, "outer": outer, "split": split, "config": config, "files": {"outer": outer_path, "selection": selection_path, "split": split_path, "config": config_path, "teacher_metadata": teacher_path}}


def _reconstruct_b2_query_identity(evidence: Mapping[str, Any], *, nwb_root: Path | None = None,
                                   expected_target_nwb_sha256: str | None = None) -> tuple[str, str, str]:
    """CPU-only implementation replay; this is not a historical attestation."""

    streaming = WORKSPACE / "streaming_calibration_exp"
    if str(streaming) not in sys.path:
        sys.path.insert(0, str(streaming))
    from src.data.falcon_datamodule import SessionBatchSampler  # pylint: disable=import-outside-toplevel
    from src.data.rt_nested_loso_datamodule import build_outer_target_dataset  # pylint: disable=import-outside-toplevel

    config, split, outer = evidence["config"], evidence["split"], evidence["outer"]
    data = config["data"]
    # Bind every value that this replayer would otherwise silently hard-code.
    _need(data.get("interpolate_trials") is True, "B2 query replay requires interpolate_trials=True")
    _need(data.get("interpolate_trials_kind") == "cubic", "B2 query replay requires cubic interpolation")
    _need(float(data.get("pad_value")) == -1.0, "B2 query replay requires pad_value=-1")
    _need(data.get("batch_size") == 32, "B2 query replay requires batch_size=32")
    data_dir = str(outer.get("data_dir", DEFAULT_NWB_ROOT))
    if nwb_root is not None:
        # A new receipt may be generated on the preserved remote Stage2 tree.
        # Its absolute data_dir is not portable and is never trusted locally.
        target_name = Path(str(outer.get("outer_target_path", ""))).name
        _need(bool(target_name), "B2 query replay override requires target NWB basename")
        local_target = nwb_root / target_name
        _need(local_target.is_file(), f"B2 query replay local target missing: {local_target}")
        if expected_target_nwb_sha256 is not None:
            _need(_sha(local_target) == expected_target_nwb_sha256, "B2 query replay local target NWB SHA mismatch")
        data_dir = str(nwb_root)
    dataset, reconstructed, target_path = build_outer_target_dataset(
        data_dir=data_dir, outer_loso_fold=int(outer["outer_loso_fold"]),
        side_feature_group=B2, side_feature_shuffle_seed=SEED, calibration_n_trials=24,
        query_start_trial=24, window_size=50, max_trial_length=100,
        interpolate_trials=True, interpolate_trials_kind=str(data.get("interpolate_trials_kind", "cubic")),
        pad_value=float(data.get("pad_value", -1.0)), expected_session_count=15,
    )
    _need(reconstructed.outer_target_session == outer.get("outer_target_session") == split.get("target_session"), "B2 reconstructed target session mismatch")
    _need(target_path.name == Path(str(outer.get("outer_target_path", ""))).name, "B2 reconstructed target NWB mismatch")
    if expected_target_nwb_sha256 is not None:
        _need(_sha(target_path) == expected_target_nwb_sha256, "B2 reconstructed target NWB SHA mismatch")
    audit = getattr(dataset, "query_window_audit", {}).get(reconstructed.outer_target_session)
    _need(isinstance(audit, Mapping), "B2 query reconstruction produced no audit")
    eligible = int(audit.get("eligible_windows", -1))
    batch_size = int(data.get("batch_size", 0))
    _need(batch_size > 0, "B2 query reconstruction lacks batch size")
    expected = eligible - (eligible % batch_size)
    _need(outer.get("query_windows_evaluated") == expected, "B2 outer count differs from frozen complete-batch sampler rule")
    replay = stage2.evaluated_query_identity_from_sampler(
        dataset, SessionBatchSampler(dataset, batch_size, shuffle=False)
    )
    _need(replay["evaluated_windows"] == expected, "B2 sampler replay count mismatch")
    result = tuple(
        str(replay[field])
        for field in (
            "ordered_window_start_sha256",
            "ordered_target_covariate_evalmask_sha256",
            "ordered_query_identity_sha256",
        )
    )
    _need(all(_is_sha(value) for value in result), "B2 reconstruction lacks strong evaluated digest")
    return result  # type: ignore[return-value]


def _historically_bound_b2_query_identity(evidence: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return a historical witness only; fail rather than relabel a replay.

    Fold 0 has an immutable archived-source CPU replay receipt. Other B2 folds
    require a forward-only re-evaluation receipt after Stage-2 terminal
    closure; matching today's code is useful diagnosis, not enough to pair an
    old score with the new T4d score.
    """

    fold = int(evidence["fold"])
    if fold != 0:
        raise CompanionError(
            f"B2 fold {fold}: no archived-source ordered-query witness; "
            "run the forward-only B2 re-evaluation after Stage-2 terminal closure"
        )
    _need(FOLD0_REMOTE_RECONSTRUCTION.is_file(), "B2 fold 0: remote reconstruction receipt missing")
    _need(stat.S_IMODE(FOLD0_REMOTE_RECONSTRUCTION.stat().st_mode) == 0o444, "B2 fold 0: reconstruction receipt must be immutable mode 0444")
    receipt = _json(FOLD0_REMOTE_RECONSTRUCTION)
    _need(
        receipt.get("schema") == "rt_sparse_t4d_b2_fold0_remote_provenance_reconstruction_v1"
        and receipt.get("status") == "PASS_CRYPTOGRAPHIC_REMOTE_RECONSTRUCTION_LIMITED_HISTORICAL_SOURCE_ATTESTATION",
        "B2 fold 0: remote reconstruction receipt status/schema mismatch",
    )
    replay = receipt.get("historical_b2_query_replay")
    _need(isinstance(replay, Mapping), "B2 fold 0: receipt has no historical replay")
    digests = replay.get("evaluated_digests")
    _need(isinstance(digests, Mapping), "B2 fold 0: receipt has no evaluated digests")
    result = tuple(
        str(digests.get(field, ""))
        for field in (
            "ordered_window_start_sha256",
            "ordered_target_covariate_evalmask_sha256",
            "ordered_query_identity_sha256",
        )
    )
    _need(all(_is_sha(value) for value in result), "B2 fold 0: invalid historical evaluated digest")
    return result  # type: ignore[return-value]


def _stage_records(artifact_root: Path, *, mappings: Sequence[tuple[Path, Path]], nwb_root: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Read stage closures only after the independent verifier has passed."""

    terminal = stage2.verify_terminal_bundle(artifact_root, path_mappings=mappings)
    manifest_path = artifact_root / stage2.MANIFEST_NAME
    manifest = stage2._load_json(manifest_path)
    manifest_sha = stage2._sha256(manifest_path)
    resolver = stage2.PathResolver(mappings)
    records: dict[int, dict[str, Any]] = {}
    for cell in stage2.validate_manifest_schema(manifest):
        if cell["arm"] != T4D:
            continue
        closure = stage2._load_json(artifact_root / "cells" / f"{cell['output_key']}.json")
        checked = stage2.validate_cell_closure(cell, closure, matrix_manifest_sha256=manifest_sha, resolver=resolver)
        replay = stage2.replay_evaluated_query_identity(checked, nwb_root=nwb_root)
        checked["evaluated_query_digests"] = tuple(
            replay[field]
            for field in (
                "ordered_window_start_sha256",
                "ordered_target_covariate_evalmask_sha256",
                "ordered_query_identity_sha256",
            )
        )
        records[int(cell["fold"])] = checked
    _need(tuple(sorted(records)) == FOLDS, "terminal Stage2 does not contain all 15 T4d cells")
    nwbs = {Path(str(row["path"])).name.removeprefix("sub-C_").removesuffix("_behavior+ecephys.nwb"): row for row in manifest["nwb_allowlist"]}
    return records, terminal, nwbs


def _same_order(left: Any, right: Any, label: str) -> None:
    _need(isinstance(left, list) and isinstance(right, list) and left == right, f"{label} ordered session list mismatch")


def aggregate_companion(stage_records: Mapping[int, Mapping[str, Any]], b2_evidence: Mapping[int, Mapping[str, Any]],
                        *, reconstructed_query: Callable[[Mapping[str, Any]], tuple[str, str, str]],
                        prospective_folds: Sequence[int] = PROSPECTIVE_DEFAULT) -> dict[str, Any]:
    """Pure paired aggregation used by the CLI and synthetic tests."""

    _need(tuple(sorted(stage_records)) == FOLDS and tuple(sorted(b2_evidence)) == FOLDS, "companion needs exactly 15 paired folds")
    prospective = tuple(prospective_folds)
    _need(
        prospective == PROSPECTIVE_DEFAULT,
        "fold 3 has no independently verifiable unopened-score attestation; prospective subset is frozen to folds 4-14",
    )
    deltas: dict[str, float] = {}
    for fold in FOLDS:
        stage, legacy = stage_records[fold], b2_evidence[fold]
        outer, split = legacy["outer"], legacy["split"]
        _need(stage["session"] == outer.get("outer_target_session") == split.get("target_session"), f"fold {fold}: target session mismatch")
        _same_order(split.get("inner_train_sessions"), stage.get("inner_train_sessions"), f"fold {fold}: inner train")
        _need(split.get("inner_validation_session") == stage.get("inner_validation_session"), f"fold {fold}: inner validation mismatch")
        _need(
            reconstructed_query(legacy) == stage.get("evaluated_query_digests", stage["query_digests"]),
            f"fold {fold}: reconstructed actual B2 query digest mismatches T4d",
        )
        session = str(stage["session"])
        _need(session not in deltas, f"duplicate target session: {session}")
        deltas[session] = float(stage["r2"]) - _finite(outer.get("r2_variance_weighted"), f"B2 fold {fold} R2")
    prospective_sessions = {str(stage_records[fold]["session"]) for fold in prospective}
    return {
        "schema": "rt_sparse_t4d_vs_matched_spint_structured_b2_d1024_companion_v1",
        "status": "PASS_TERMINAL_COMPANION_READ_ONLY",
        "comparator": "matched SPINT-structured B2-D1024 (not released-code original SPINT)",
        "primary_stage2_gate_changed": False,
        "full15_descriptive": _summary(deltas),
        "prospective_subset": {"folds": list(prospective), "score_opening_basis": "fold3 lacks independent unopened attestation; default folds4-14", "statistics": _summary({k: v for k, v in deltas.items() if k in prospective_sessions})},
    }


def run_terminal(artifact_root: Path, *, mappings: Sequence[tuple[Path, Path]] = ()) -> dict[str, Any]:
    raise CompanionError(
        "historical B2 scores are diagnostic-only: use the uniform 15-fold "
        "rt_sparse_t4d_b2_forward_reeval_terminal.py workflow after Stage2 PASS"
    )
    # Kept below as a separately testable legacy-audit implementation.  It is
    # deliberately unreachable so an archived fold can never become a special
    # score acceptance path after the uniform forward-only decision.
    stage_records, terminal, nwb_rows = _stage_records(
        artifact_root, mappings=mappings, nwb_root=DEFAULT_NWB_ROOT
    )
    teacher_sha = str(terminal["workspace_bindings"]["teacher_sha256"])
    legacy_rows = _legacy_b2_rows()
    b2 = {fold: _validate_b2_self(fold, legacy_rows[fold], stage_teacher_sha=teacher_sha, nwb_rows=nwb_rows) for fold in FOLDS}
    # Stage split receipts are already hash-validated by Stage2's verifier.
    for fold, record in stage_records.items():
        closure = stage2._load_json(artifact_root / "cells" / f"f{fold:02d}_{T4D}.json")
        split_path = stage2.PathResolver(mappings).resolve(closure["artifact_paths"]["split_manifest"])
        split = stage2._load_json(split_path)
        record["inner_train_sessions"] = split.get("inner_train_sessions")
        record["inner_validation_session"] = split.get("inner_validation_session")
    report = aggregate_companion(stage_records, b2, reconstructed_query=_historically_bound_b2_query_identity)
    report["stage2_terminal_verifier"] = terminal
    report["b2_aggregate_sha256"] = B2_AGGREGATE_SHA256
    report["non_interference"] = {"gpu_started": False, "training_started": False, "target_backpropagation": False, "artifact_written": False}
    return report


def _parse_map(value: str) -> tuple[Path, Path]:
    _need("=" in value, "--path-map requires REMOTE=LOCAL")
    left, right = value.split("=", 1)
    _need(bool(left) and bool(right), "--path-map sides must be nonempty")
    return Path(left), Path(right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="REMOTE=LOCAL")
    args = parser.parse_args()
    result = run_terminal(args.artifact_root, mappings=tuple(_parse_map(value) for value in args.path_map))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except CompanionError as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(2) from error
