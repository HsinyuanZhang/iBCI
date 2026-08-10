#!/usr/bin/env python3
"""Independent, read-only terminal verifier for the RT sparse Stage-2 matrix.

The matrix supervisor owns execution.  This module deliberately does not import
the supervisor and never launches a child process, opens an NWB payload, imports
Torch, or writes a receipt.  It verifies a transported 45-cell artifact bundle,
recomputes both paired contrasts from the raw outer-evaluation R2 values, and
prints its independent result to stdout.

When artifacts were copied from another filesystem, repeat ``--path-map`` as
``REMOTE_PREFIX=LOCAL_PREFIX`` so the immutable absolute paths in cell closures
can be resolved without editing those closures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "STAGE2_MATRIX_MANIFEST_v1.json"
AGGREGATE_NAME = "STAGE2_MATRIX_AGGREGATE_v1.json"
FOLDS = tuple(range(15))
ARMS = ("rt_sparse_endpoint_t4d", "afc4_vel", "zero4")
SEED = 42

CONTRACT_REL = Path(
    "sua_exploration/docs/RT_SPARSE_ENDPOINT_STAGE2_THREE_ARM_CONTRACT_20260810.md"
)
READINESS_REL = Path(
    "sua_exploration/results/rt_simple_label_v1/stage2_preflight/"
    "RT_SPARSE_ENDPOINT_STAGE2_ROOT_READINESS_REVIEW_v2.json"
)
SURFACE_RELS: dict[str, Path] = {
    "matrix_supervisor": Path("sua_exploration/scripts/rt_sparse_endpoint_stage2_matrix.py"),
    "runner": Path("streaming_calibration_exp/scripts/run_rt_clean_nested_loso.py"),
    "datamodule": Path("streaming_calibration_exp/src/data/rt_nested_loso_datamodule.py"),
    "falcon_dataset": Path("streaming_calibration_exp/src/data/falcon_datamodule.py"),
    "t4d_loader": Path("streaming_calibration_exp/src/data/rt_sparse_endpoint_loader.py"),
    "outer_evaluator": Path("streaming_calibration_exp/src/rt_clean_nested_loso_eval.py"),
    "base_experiment": Path(
        "streaming_calibration_exp/configs/experiment/rt_clean_nested_loso_m24.yaml"
    ),
    "data_config": Path("streaming_calibration_exp/configs/data/rt_nested_loso_m24.yaml"),
}
CONFIG_TREE_REL = Path("streaming_calibration_exp/configs")
SOURCE_TREE_REL = Path("streaming_calibration_exp/src")
TEACHER_REL = Path(
    "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/"
    "best_ckpt/epoch_034.ckpt"
)
DEFAULT_NWB_REL = Path("sua_exploration/data/dandi_000688/sub-C")


class VerificationError(ValueError):
    """A fail-closed terminal verification error."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSON {path}: {error}") from error
    _need(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise VerificationError(f"cannot read YAML {path}: {error}") from error
    _need(isinstance(value, dict), f"YAML root must be a mapping: {path}")
    return value


def _tree_digest(root: Path, pattern: str, workspace_root: Path) -> str:
    """Reproduce the matrix supervisor's ordered tree digest independently."""

    digest = hashlib.sha256()
    files = [path for path in sorted(root.glob(pattern)) if path.is_file()]
    _need(files, f"bound source tree contains no matching files: {root}/{pattern}")
    for path in files:
        try:
            relative = path.resolve().relative_to(workspace_root.resolve())
        except ValueError as error:
            raise VerificationError(f"tree member escapes workspace: {path}") from error
        digest.update(relative.as_posix().encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _expected_cells() -> list[dict[str, Any]]:
    return [
        {
            "fold": fold,
            "arm": arm,
            "seed": SEED,
            "run_id": f"rt_stage2_{arm}_f{fold:02d}_s42",
            "fresh_fit": True,
            "exactly_once_outer_eval": True,
            "output_key": f"f{fold:02d}_{arm}",
        }
        for fold in FOLDS
        for arm in ARMS
    ]


def _row_sha(row: Any, *, label: str) -> str:
    _need(isinstance(row, Mapping), f"{label} binding must be an object")
    value = row.get("sha256")
    _need(_is_sha256(value), f"{label} binding lacks a SHA-256")
    return str(value)


def validate_manifest_schema(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the immutable 45-cell manifest without opening bound payloads."""

    _need(
        manifest.get("schema") == "rt_sparse_endpoint_stage2_matrix_manifest_v1",
        "matrix manifest schema mismatch",
    )
    _need(manifest.get("status") == "PREPARED_NOT_LAUNCHED", "matrix manifest status mismatch")
    _row_sha(manifest.get("contract"), label="contract")
    _row_sha(manifest.get("root_readiness"), label="root readiness")

    surfaces = manifest.get("surfaces")
    _need(isinstance(surfaces, Mapping), "matrix surfaces binding missing")
    _need(set(surfaces) == set(SURFACE_RELS), "matrix surfaces set mismatch")
    for name in SURFACE_RELS:
        _row_sha(surfaces[name], label=f"surface {name}")

    launch_inputs = manifest.get("launch_inputs")
    _need(isinstance(launch_inputs, Mapping), "matrix launch_inputs missing")
    _need(
        set(launch_inputs) == {"configs_yaml_tree", "src_py_tree", "teacher_checkpoint"},
        "matrix launch_inputs set mismatch",
    )
    for name, row in launch_inputs.items():
        _row_sha(row, label=f"launch input {name}")
    teacher = launch_inputs["teacher_checkpoint"]
    _need(isinstance(teacher.get("bytes"), int) and teacher["bytes"] > 0, "teacher byte count invalid")

    nwbs = manifest.get("nwb_allowlist")
    _need(isinstance(nwbs, list) and len(nwbs) == 15, "NWB allowlist must contain 15 rows")
    nwb_names: list[str] = []
    for index, row in enumerate(nwbs):
        _need(isinstance(row, Mapping), f"NWB allowlist row {index} is not an object")
        path = Path(str(row.get("path", "")))
        _need(
            path.name.startswith("sub-C_ses-RT-")
            and path.name.endswith("_behavior+ecephys.nwb"),
            f"NWB allowlist row {index} is not an RT file",
        )
        _need(isinstance(row.get("bytes"), int) and row["bytes"] > 0, "NWB byte count invalid")
        _row_sha(row, label=f"NWB {path.name}")
        nwb_names.append(path.name)
    _need(len(set(nwb_names)) == 15, "NWB allowlist contains duplicate filenames")

    matrix = manifest.get("matrix")
    _need(isinstance(matrix, Mapping), "matrix section missing")
    _need(matrix.get("folds") == list(FOLDS), "matrix fold list mismatch")
    _need(matrix.get("arms") == list(ARMS), "matrix arm list/order mismatch")
    _need(matrix.get("seed") == SEED, "matrix seed is not 42")
    _need(matrix.get("fresh_fit_count") == 45, "matrix fresh-fit count is not 45")
    _need(matrix.get("outer_eval_count") == 45, "matrix outer-eval count is not 45")
    _need(matrix.get("base_experiment") == "rt_clean_nested_loso_m24", "base experiment drift")
    _need(matrix.get("arm_override_only") is True, "matrix is not arm-override-only")
    cells = matrix.get("cells")
    expected = _expected_cells()
    _need(cells == expected, "matrix cells differ from the exact frozen 15x3xseed42 schedule")
    return expected


def _verify_bound_file(path: Path, row: Mapping[str, Any], *, label: str) -> None:
    _need(path.is_file(), f"bound {label} is missing locally: {path}")
    _need(_sha256(path) == _row_sha(row, label=label), f"bound {label} SHA drift")


def verify_workspace_bindings(
    manifest: Mapping[str, Any],
    *,
    workspace_root: Path,
    nwb_root: Path,
) -> dict[str, Any]:
    """Verify current source/config/teacher/NWB bytes against the launch manifest."""

    workspace = workspace_root.resolve()
    contract = workspace / CONTRACT_REL
    readiness_path = workspace / READINESS_REL
    _verify_bound_file(contract, manifest["contract"], label="contract")
    _verify_bound_file(readiness_path, manifest["root_readiness"], label="root readiness")
    readiness = _load_json(readiness_path)
    _need(
        readiness.get("status") == "PASS_ROOT_REVIEW_STAGE2_MATRIX_READY_NOT_LAUNCHED",
        "root readiness is not matrix-ready",
    )
    readiness_bound = readiness.get("bound_files")
    _need(isinstance(readiness_bound, Mapping), "root readiness bound_files missing")
    _need(
        _row_sha(readiness_bound.get("contract"), label="readiness contract")
        == _row_sha(manifest["contract"], label="manifest contract"),
        "contract SHA differs between readiness and matrix manifest",
    )

    for name, relative in SURFACE_RELS.items():
        path = workspace / relative
        row = manifest["surfaces"][name]
        _verify_bound_file(path, row, label=f"surface {name}")
        _need(
            _row_sha(readiness_bound.get(name), label=f"readiness {name}")
            == _row_sha(row, label=f"manifest {name}"),
            f"surface {name} differs between readiness and matrix manifest",
        )

    launch = manifest["launch_inputs"]
    config_digest = _tree_digest(workspace / CONFIG_TREE_REL, "**/*.yaml", workspace)
    source_digest = _tree_digest(workspace / SOURCE_TREE_REL, "**/*.py", workspace)
    _need(config_digest == launch["configs_yaml_tree"]["sha256"], "config YAML tree digest drift")
    _need(source_digest == launch["src_py_tree"]["sha256"], "source Python tree digest drift")
    for name in ("configs_yaml_tree", "src_py_tree"):
        _need(
            _row_sha(readiness_bound.get(name), label=f"readiness {name}")
            == _row_sha(launch[name], label=f"manifest {name}"),
            f"{name} differs between readiness and matrix manifest",
        )

    teacher_path = workspace / TEACHER_REL
    teacher_row = launch["teacher_checkpoint"]
    _verify_bound_file(teacher_path, teacher_row, label="teacher checkpoint")
    _need(teacher_path.stat().st_size == teacher_row["bytes"], "teacher byte count drift")
    _need(
        _row_sha(readiness_bound.get("teacher_checkpoint"), label="readiness teacher")
        == _row_sha(teacher_row, label="manifest teacher"),
        "teacher differs between readiness and matrix manifest",
    )

    local_nwbs = {path.name: path for path in nwb_root.glob("sub-C_ses-RT-*_behavior+ecephys.nwb")}
    _need(len(local_nwbs) == 15, f"local NWB root must contain exactly 15 RT files: {nwb_root}")
    for row in manifest["nwb_allowlist"]:
        name = Path(str(row["path"])).name
        _need(name in local_nwbs, f"allowlisted NWB is missing locally: {name}")
        local = local_nwbs[name]
        _need(local.stat().st_size == row["bytes"], f"NWB byte count drift: {name}")
        _need(_sha256(local) == row["sha256"], f"NWB SHA drift: {name}")

    return {
        "contract_sha256": manifest["contract"]["sha256"],
        "root_readiness_sha256": manifest["root_readiness"]["sha256"],
        "config_tree_sha256": config_digest,
        "source_tree_sha256": source_digest,
        "teacher_sha256": teacher_row["sha256"],
        "nwb_count": 15,
    }


class PathResolver:
    """Resolve immutable remote artifact paths without rewriting receipts."""

    def __init__(self, mappings: Sequence[tuple[Path, Path]] = ()) -> None:
        self._mappings = tuple((old.resolve(), new.resolve()) for old, new in mappings)

    def resolve(self, raw: Any) -> Path:
        value = Path(str(raw))
        if value.is_file():
            return value.resolve()
        if value.is_absolute():
            for old, new in self._mappings:
                try:
                    suffix = value.relative_to(old)
                except ValueError:
                    continue
                candidate = new / suffix
                if candidate.is_file():
                    return candidate.resolve()
        raise VerificationError(f"bound artifact path cannot be resolved: {value}")


def _raw_path_equal(left: Any, right: Any, *, label: str) -> None:
    _need(str(left) == str(right), f"{label} original path chain mismatch")


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        _need(isinstance(value, Mapping) and key in value, f"config field missing: {'.'.join(keys)}")
        value = value[key]
    return value


def _config_signature(config: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    """Validate fixed cell fields and return fields that must match all arms/folds."""

    _need(config.get("run_id") == cell["run_id"], "resolved config run_id mismatch")
    _need(config.get("seed") == SEED, "resolved config seed mismatch")
    _need(config.get("train") is True and config.get("test") is False, "train/test mode drift")
    _need(config.get("no_early_stopping") is True, "35-epoch no-early-stop rule missing")
    _need(config.get("optimized_metric") == "val_heldin/r2_mean", "checkpoint metric drift")
    data = _nested_get(config, "data")
    model = _nested_get(config, "model")
    trainer = _nested_get(config, "trainer")
    _need(isinstance(data, Mapping) and isinstance(model, Mapping) and isinstance(trainer, Mapping), "config sections invalid")
    _need(data.get("side_feature_group") == cell["arm"], "config arm mismatch")
    _need(data.get("loso_fold") == cell["fold"], "config LOSO fold mismatch")
    _need(data.get("outer_loso_fold") == cell["fold"], "config outer fold mismatch")
    _need(data.get("calibration_n_trials") == 24, "config calibration budget is not M24")
    _need(data.get("query_start_trial") in (24, "${.calibration_n_trials}"), "query boundary drift")
    _need(data.get("window_size") == 50, "window size drift")
    _need(data.get("max_trial_length") == 100, "max trial length drift")
    _need(data.get("session_window_budget") == 4096, "session pool drift")
    _need(data.get("sampler_seed") in (42, "${seed}"), "sampler seed drift")
    _need(data.get("random_calibration") is False, "random calibration is forbidden")
    _need(data.get("smooth_calibration") is False, "smoothed calibration is forbidden")
    _need(trainer.get("max_epochs") == 35, "epoch budget drift")
    _need(model.get("variant") == "B3S", "model variant drift")
    _need(model.get("side_dim") == 4, "side width drift")
    _need(model.get("freeze_decoder") is False, "source decoder must be jointly retrained")
    _need(model.get("loss_mode") == "task_only", "loss mode drift")
    _need(float(model.get("lambda_y")) == 0.0 and float(model.get("lambda_E")) == 0.0, "auxiliary loss drift")
    optimizer = model.get("optimizer")
    _need(isinstance(optimizer, Mapping), "optimizer config missing")
    teacher_ref = model.get("teacher_ckpt_path")
    _need(teacher_ref == "${paths.teacher_ckpt_path}", "model teacher binding drift")
    paths = config.get("paths")
    _need(isinstance(paths, Mapping), "config paths section missing")
    teacher_path = str(paths.get("teacher_ckpt_path", ""))
    _need(teacher_path.endswith(TEACHER_REL.as_posix()), "config teacher path does not name frozen teacher")
    return {
        "data_target": data.get("_target_"),
        "data_dir": data.get("data_dir"),
        "batch_size": data.get("batch_size"),
        "window_size": data.get("window_size"),
        "calibration_n_trials": data.get("calibration_n_trials"),
        "query_start_trial": data.get("query_start_trial"),
        "max_trial_length": data.get("max_trial_length"),
        "interpolate_trials": data.get("interpolate_trials"),
        "interpolate_trials_kind": data.get("interpolate_trials_kind"),
        "session_window_budget": data.get("session_window_budget"),
        "sampler_seed": data.get("sampler_seed"),
        "model_target": model.get("_target_"),
        "variant": model.get("variant"),
        "hidden_dim": model.get("hidden_dim"),
        "side_dim": model.get("side_dim"),
        "freeze_decoder": model.get("freeze_decoder"),
        "loss_mode": model.get("loss_mode"),
        "lambda_y": model.get("lambda_y"),
        "lambda_E": model.get("lambda_E"),
        "optimizer": dict(optimizer),
        "scheduler": model.get("scheduler"),
        "teacher_ref": teacher_ref,
        "teacher_path": teacher_path,
        "trainer_max_epochs": trainer.get("max_epochs"),
        "checkpoint_metric": config.get("optimized_metric"),
    }


def _strong_query_identity(
    outer: Mapping[str, Any], *, batch_size: int
) -> tuple[str, dict[str, Any], tuple[str, str, str]]:
    target_session = outer.get("outer_target_session")
    _need(isinstance(target_session, str) and target_session, "outer target session missing")
    query = outer.get("matched_query_window_identity")
    _need(isinstance(query, Mapping) and set(query) == {target_session}, "outer query identity scope mismatch")
    audit = query[target_session]
    _need(isinstance(audit, Mapping), "outer query audit is not an object")
    digest_names = (
        "ordered_window_start_sha256",
        "ordered_target_covariate_evalmask_sha256",
        "ordered_query_identity_sha256",
    )
    digests = tuple(str(audit.get(name, "")) for name in digest_names)
    _need(all(_is_sha256(value) for value in digests), "strong ordered query digest missing")
    _need(audit.get("support_trials") == 24, "query audit support budget drift")
    _need(audit.get("query_start_trial") == 24, "query audit start drift")
    _need(audit.get("window_size") == 50, "query audit window drift")
    _need(audit.get("full_window_disjoint") is True, "query windows are not support-disjoint")
    eligible = audit.get("eligible_windows")
    _need(isinstance(eligible, int) and eligible > 0, "query audit has no eligible windows")
    _need(isinstance(batch_size, int) and batch_size > 0, "query replay batch size is invalid")
    # SessionBatchSampler, used by the frozen one-shot evaluator, yields only
    # complete batches.  ``query_window_audit`` reports all eligible windows;
    # the scored/evaluated subset is the sampler's ordered complete batches.
    # Do not equate these two counts: on the real RT folds their remainders are
    # e.g. 24 (fold 0), 14 (fold 1), and 7 (fold 2).
    expected_evaluated = eligible - (eligible % batch_size)
    _need(
        outer.get("query_windows_evaluated") == expected_evaluated,
        "evaluated window count is not the frozen SessionBatchSampler complete-batch count",
    )
    return target_session, dict(audit), digests  # type: ignore[return-value]


def validate_cell_closure(
    cell: Mapping[str, Any],
    closure: Mapping[str, Any],
    *,
    matrix_manifest_sha256: str,
    resolver: PathResolver,
) -> dict[str, Any]:
    """Validate one complete closure and return independently parsed evidence."""

    _need(closure.get("schema") == "rt_sparse_endpoint_stage2_cell_closure_v2", "cell closure schema mismatch")
    _need(closure.get("matrix_manifest_sha256") == matrix_manifest_sha256, "cell/manifest SHA mismatch")
    _need(closure.get("cell") == cell, "cell closure identity mismatch")
    paths = closure.get("artifact_paths")
    required_paths = {"selection_receipt", "config", "checkpoint", "split_manifest", "outer_receipt"}
    _need(isinstance(paths, Mapping) and set(paths) == required_paths, "cell artifact path set mismatch")
    sha_fields = {
        "selection_receipt": "selection_receipt_sha256",
        "config": "config_sha256",
        "checkpoint": "checkpoint_sha256",
        "split_manifest": "split_manifest_sha256",
        "outer_receipt": "outer_receipt_sha256",
    }
    resolved: dict[str, Path] = {}
    for name, field in sha_fields.items():
        expected_sha = closure.get(field)
        _need(_is_sha256(expected_sha), f"cell {cell['output_key']} lacks {field}")
        resolved[name] = resolver.resolve(paths[name])
        _need(_sha256(resolved[name]) == expected_sha, f"cell {cell['output_key']} {name} SHA drift")

    selection = _load_json(resolved["selection_receipt"])
    split = _load_json(resolved["split_manifest"])
    outer = _load_json(resolved["outer_receipt"])
    embedded_outer = closure.get("outer_receipt")
    _need(isinstance(embedded_outer, Mapping) and dict(embedded_outer) == outer, "embedded outer receipt differs from file")
    config = _load_yaml(resolved["config"])

    _need(selection.get("schema") == "rt_clean_nested_loso_selection_receipt_v1", "selection schema mismatch")
    _need(selection.get("status") == "PASS_FIT_INNER_SELECTION_ONLY", "selection receipt did not pass")
    _need(selection.get("run_id") == cell["run_id"], "selection run_id is not the fresh matrix run")
    _need(selection.get("arm") == cell["arm"], "selection arm mismatch")
    _need(selection.get("outer_loso_fold") == cell["fold"], "selection fold mismatch")
    _need(selection.get("seed") == SEED, "selection seed mismatch")
    _need(selection.get("selected_by_metric") == "val_heldin/r2_mean", "selection metric drift")
    _need(selection.get("selected_metric_scope") == "inner_validation_session_only", "selection scope drift")
    _need(selection.get("formal_heldout_opened") is False, "formal heldout was opened")
    _need(selection.get("outer_target_loaded_during_fit") is False, "outer target was opened during fit")
    _need(selection.get("outer_target_query_labels_read_during_fit") is False, "outer query labels read during fit")
    _raw_path_equal(selection.get("selection_receipt_path"), paths["selection_receipt"], label="selection self")
    _raw_path_equal(selection.get("best_model_path"), paths["checkpoint"], label="selection checkpoint")
    _raw_path_equal(selection.get("config_path"), paths["config"], label="selection config")
    _raw_path_equal(selection.get("split_manifest_path"), paths["split_manifest"], label="selection split")
    _need(selection.get("best_model_sha256") == closure["checkpoint_sha256"], "selection checkpoint SHA chain mismatch")
    _need(selection.get("config_sha256") == closure["config_sha256"], "selection config SHA chain mismatch")
    _need(selection.get("split_manifest_sha256") == closure["split_manifest_sha256"], "selection split SHA chain mismatch")
    expected_run_marker = f"rid-{cell['run_id']}_f{cell['fold']}_s42"
    _need(expected_run_marker in Path(str(selection.get("run_dir", ""))).name, "selection run directory is not the fresh cell run")
    _need(Path(str(paths["selection_receipt"])).parent == Path(str(selection.get("run_dir"))), "selection run-dir chain mismatch")

    signature = _config_signature(config, cell)

    _need(split.get("validation_protocol") == "nested_loso", "split is not nested LOSO")
    _need(split.get("requested_side_feature_group") == cell["arm"], "split arm mismatch")
    _need(split.get("outer_loso_fold") == cell["fold"], "split fold mismatch")
    _need(split.get("formal_heldout_opened") is False, "split opened formal heldout")
    nested = split.get("nested_selection")
    _need(isinstance(nested, Mapping) and nested.get("clean") is True, "split clean-selection proof missing")
    _need(nested.get("outer_target_loaded_during_fit") is False, "split target-loader exclusion failed")
    _need(nested.get("outer_target_query_labels_read_during_fit") is False, "split query-label exclusion failed")
    _need(nested.get("inner_validation_only_for_checkpoint_selection") is True, "split checkpoint scope failed")
    calibration = split.get("calibration")
    query = split.get("query")
    _need(isinstance(calibration, Mapping) and calibration.get("budget_trials") == 24, "split M24 budget missing")
    _need(calibration.get("target_calibration_optimizer_steps") == 0, "target calibration optimizer steps are nonzero")
    _need(isinstance(query, Mapping) and query.get("query_start_trial") == 24, "split query boundary drift")
    _need(query.get("window_size_bins") == 50, "split query window drift")

    _need(outer.get("schema") == "rt_clean_nested_loso_outer_eval_v1", "outer receipt schema mismatch")
    _need(outer.get("status") == "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP", "outer evaluation did not pass")
    _need(outer.get("run_id") == cell["run_id"], "outer run_id mismatch")
    _need(outer.get("arm") == cell["arm"], "outer arm mismatch")
    _need(outer.get("outer_loso_fold") == cell["fold"], "outer fold mismatch")
    _need(outer.get("seed") == SEED, "outer seed mismatch")
    _need(outer.get("outer_target_session") == split.get("target_session"), "outer/split target mismatch")
    _need(outer.get("checkpoint_sha256") == closure["checkpoint_sha256"], "outer checkpoint SHA chain mismatch")
    _raw_path_equal(outer.get("selection_receipt_path"), paths["selection_receipt"], label="outer selection")
    _raw_path_equal(outer.get("config_path"), paths["config"], label="outer config")
    _raw_path_equal(outer.get("fit_split_manifest"), paths["split_manifest"], label="outer split")
    _need(outer.get("query_start_trial") == 24 and outer.get("window_size") == 50, "outer query contract drift")
    _need(outer.get("target_query_labels_used_for_scoring_only") is True, "query scoring-label proof missing")
    for field in (
        "target_query_labels_used_for_calibration",
        "target_query_labels_used_for_normalization",
        "target_query_labels_used_for_checkpoint_selection",
        "target_backpropagation",
        "optimizer_present",
        "model_training_mode",
    ):
        _need(outer.get(field) is False, f"outer no-target-update proof failed: {field}")
    _need(outer.get("model_state_unchanged") is True, "outer model state changed")
    _need(outer.get("model_state_three_point_unchanged") is True, "three-point state proof failed")
    state_hashes = tuple(
        outer.get(name)
        for name in (
            "model_state_sha256_before",
            "model_state_sha256_after_target_carrier",
            "model_state_sha256_after",
        )
    )
    _need(all(_is_sha256(value) for value in state_hashes), "model state SHA proof missing")
    _need(state_hashes[0] == state_hashes[1] == state_hashes[2], "model state SHA values differ")
    score = outer.get("r2_variance_weighted")
    _need(isinstance(score, (int, float)) and math.isfinite(float(score)), "outer raw R2 is invalid")
    batch_size = signature.get("batch_size")
    _need(isinstance(batch_size, int), "resolved config batch_size missing")
    session, query_audit, query_digests = _strong_query_identity(outer, batch_size=batch_size)
    return {
        "cell": dict(cell),
        "session": session,
        "r2": float(score),
        "query_audit": query_audit,
        "query_digests": query_digests,
        "evaluated_window_count": int(outer["query_windows_evaluated"]),
        "config_signature": signature,
        "checkpoint_sha256": closure["checkpoint_sha256"],
    }


def _summarize(values: Mapping[str, float]) -> dict[str, Any]:
    _need(len(values) == 15, "paired contrast does not contain 15 sessions")
    names = sorted(values)
    numbers = [float(values[name]) for name in names]
    mean = statistics.fmean(numbers)
    median = statistics.median(numbers)
    # Largest absolute value; ties resolve to the earliest sorted session.
    removed_index = max(range(len(names)), key=lambda index: (abs(numbers[index]), -index))
    kept = numbers[:removed_index] + numbers[removed_index + 1 :]
    positive = sum(value > 0.0 for value in numbers)
    zero = sum(value == 0.0 for value in numbers)
    negative = sum(value < 0.0 for value in numbers)
    return {
        "ordered": [
            {"session": name, "delta": float(values[name])}
            for name in names
        ],
        "mean": mean,
        "median": median,
        "positive": positive,
        "zero": zero,
        "negative": negative,
        "leave_largest_absolute_out_mean": statistics.fmean(kept),
        "removed_session": names[removed_index],
        "mean_ge_003": mean >= 0.03,
        "median_ge_003": median >= 0.03,
        "primary_gate_pass": mean > 0.0 and median > 0.0 and positive > len(numbers) / 2,
    }


def _digest_query_values(values: Sequence[Any]) -> str:
    """Byte-identical digest format used by FalconDataset's query audit."""

    # Keep NumPy lazy: schema-only tests deliberately do not import the data
    # stack, while the terminal replay needs exactly the production arrays.
    import numpy as np  # pylint: disable=import-outside-toplevel

    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def evaluated_query_identity_from_sampler(dataset: Any, sampler: Any) -> dict[str, Any]:
    """Digest the exact ordered indices consumed by a one-shot DataLoader.

    This function intentionally receives the *actual* frozen
    ``SessionBatchSampler`` rather than deriving a prefix from a count.  It
    proves the sampler's drop-last behavior and detects a same-count sequence
    with different window indices.
    """

    import numpy as np  # pylint: disable=import-outside-toplevel

    batches = [list(batch) for batch in sampler]
    _need(batches, "query sampler yielded no complete batches")
    batch_size = getattr(sampler, "batch_size", None)
    _need(isinstance(batch_size, int) and batch_size > 0, "sampler batch size invalid")
    _need(all(len(batch) == batch_size for batch in batches), "sampler yielded a non-complete batch")
    indices = [index for batch in batches for index in batch]
    names = {dataset.window_indices[index][0] for index in indices}
    _need(len(names) == 1, "outer target sampler must consume exactly one session")
    session = next(iter(names))
    starts = np.asarray([dataset.window_indices[index][1] for index in indices], dtype=np.int64)
    window_size = int(getattr(dataset, "window_size"))
    target_indices = starts + window_size - 1
    target_rows = np.asarray(dataset.covariate_data[session][target_indices], dtype=np.float32)
    target_mask = np.asarray(dataset.eval_mask[session][target_indices], dtype=bool)
    return {
        "session": session,
        "evaluated_windows": len(indices),
        "ordered_window_start_sha256": _digest_query_values((starts,)),
        "ordered_target_covariate_evalmask_sha256": _digest_query_values(
            (target_indices, target_rows, target_mask)
        ),
        "ordered_query_identity_sha256": _digest_query_values(
            (starts, target_indices, target_rows, target_mask)
        ),
        "sampler_complete_batch_size": batch_size,
        "sampler_batch_count": len(batches),
    }


def replay_evaluated_query_identity(
    record: Mapping[str, Any], *, nwb_root: Path
) -> dict[str, Any]:
    """CPU-only replay of actual Stage-2 query indices for one closure.

    The replay uses a `zero4` target dataset solely to reconstruct the common
    neural/covariate window schedule.  Before using its actual batch indices,
    it proves its *all-eligible* three digests equal the recorded T4d audit;
    this prevents the zero4 reconstruction from silently standing in for a
    different T4d data path.  No model, optimizer, Trainer, or CUDA object is
    constructed.
    """

    streaming_root = ROOT / "streaming_calibration_exp"
    if str(streaming_root) not in sys.path:
        sys.path.insert(0, str(streaming_root))
    from src.data.falcon_datamodule import SessionBatchSampler  # pylint: disable=import-outside-toplevel
    from src.data.rt_nested_loso_datamodule import build_outer_target_dataset  # pylint: disable=import-outside-toplevel

    cell = record["cell"]
    signature = record["config_signature"]
    _need(isinstance(signature, Mapping), "record config signature missing")
    batch_size = signature.get("batch_size")
    _need(isinstance(batch_size, int) and batch_size > 0, "record batch size missing")
    dataset, split, _ = build_outer_target_dataset(
        data_dir=nwb_root,
        outer_loso_fold=int(cell["fold"]),
        side_feature_group="zero4",
        side_feature_shuffle_seed=SEED,
        calibration_n_trials=24,
        query_start_trial=24,
        window_size=50,
        max_trial_length=100,
        interpolate_trials=bool(signature.get("interpolate_trials")),
        interpolate_trials_kind=str(signature.get("interpolate_trials_kind")),
        pad_value=-1.0,
        expected_session_count=15,
    )
    _need(split.outer_target_session == record["session"], "query replay target session mismatch")
    audit = dataset.query_window_audit.get(record["session"])
    _need(isinstance(audit, Mapping), "query replay has no target audit")
    full = tuple(
        str(audit.get(name, ""))
        for name in (
            "ordered_window_start_sha256",
            "ordered_target_covariate_evalmask_sha256",
            "ordered_query_identity_sha256",
        )
    )
    _need(full == record["query_digests"], "zero4 replay all-eligible query digest differs from recorded T4d audit")
    sampler = SessionBatchSampler(dataset, batch_size, shuffle=False)
    replay = evaluated_query_identity_from_sampler(dataset, sampler)
    _need(replay["session"] == record["session"], "query replay sampler session mismatch")
    _need(
        replay["evaluated_windows"] == record["evaluated_window_count"],
        "query replay evaluated count differs from outer receipt",
    )
    return replay


def _compare_exact_structure(expected: Any, actual: Any, *, path: str = "aggregate") -> None:
    """Compare every aggregate field, allowing only 1e-12 float roundoff."""

    if isinstance(expected, Mapping):
        _need(isinstance(actual, Mapping), f"{path} is not an object")
        _need(set(actual) == set(expected), f"{path} field set mismatch")
        for key in expected:
            _compare_exact_structure(expected[key], actual[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        _need(isinstance(actual, list) and len(actual) == len(expected), f"{path} list mismatch")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_exact_structure(left, right, path=f"{path}[{index}]")
        return
    if isinstance(expected, float):
        _need(
            isinstance(actual, (int, float))
            and math.isfinite(float(actual))
            and math.isclose(expected, float(actual), rel_tol=0.0, abs_tol=1.0e-12),
            f"{path} float mismatch: expected {expected!r}, got {actual!r}",
        )
        return
    _need(type(actual) is type(expected) and actual == expected, f"{path} mismatch")


def recompute_terminal_aggregate(
    records: Sequence[Mapping[str, Any]], *, manifest_sha256: str
) -> dict[str, Any]:
    _need(len(records) == 45, "terminal record count is not 45")
    by_cell: dict[tuple[int, str], Mapping[str, Any]] = {}
    sessions_by_fold: dict[int, str] = {}
    t4d_minus_zero: dict[str, float] = {}
    t4d_minus_full: dict[str, float] = {}
    for record in records:
        cell = record["cell"]
        key = (int(cell["fold"]), str(cell["arm"]))
        _need(key not in by_cell, f"duplicate terminal cell: {key}")
        by_cell[key] = record
    _need(set(by_cell) == {(fold, arm) for fold in FOLDS for arm in ARMS}, "terminal cell grid mismatch")

    for fold in FOLDS:
        arm_records = [by_cell[(fold, arm)] for arm in ARMS]
        sessions = {record["session"] for record in arm_records}
        _need(len(sessions) == 1, f"fold {fold}: outer target session differs across arms")
        session = next(iter(sessions))
        _need(session not in sessions_by_fold.values(), f"fold {fold}: duplicate outer session {session}")
        sessions_by_fold[fold] = session
        # Terminal execution attaches actual complete-batch digest replays.
        # The fallback keeps schema-only synthetic fixtures focused on the
        # closure schema; it is never used by a local-binding terminal run.
        digests = [record.get("evaluated_query_digests", record["query_digests"]) for record in arm_records]
        _need(digests[0] == digests[1] == digests[2], f"fold {fold}: ordered evaluated query/target/mask digest mismatch")
        audits = [record["query_audit"] for record in arm_records]
        _need(audits[0] == audits[1] == audits[2], f"fold {fold}: query audit mismatch")
        r2 = {record["cell"]["arm"]: float(record["r2"]) for record in arm_records}
        t4d_minus_zero[session] = r2["rt_sparse_endpoint_t4d"] - r2["zero4"]
        t4d_minus_full[session] = r2["rt_sparse_endpoint_t4d"] - r2["afc4_vel"]

    signatures = {
        json.dumps(record["config_signature"], sort_keys=True, separators=(",", ":"))
        for record in records
    }
    _need(len(signatures) == 1, "matched training configuration differs across cells")
    return {
        "schema": "rt_sparse_endpoint_stage2_matrix_aggregate_v1",
        "status": "PASS_MATRIX_TERMINAL",
        "manifest_sha256": manifest_sha256,
        "cells": 45,
        "t4d_minus_zero4": _summarize(t4d_minus_zero),
        "t4d_minus_full": _summarize(t4d_minus_full),
    }


def verify_terminal_bundle(
    artifact_root: Path,
    *,
    workspace_root: Path = ROOT,
    nwb_root: Path | None = None,
    path_mappings: Sequence[tuple[Path, Path]] = (),
    verify_local_bindings: bool = True,
) -> dict[str, Any]:
    """Verify a complete transported matrix bundle without mutating it."""

    root = artifact_root.resolve()
    manifest_path = root / MANIFEST_NAME
    aggregate_path = root / AGGREGATE_NAME
    _need(manifest_path.is_file(), f"matrix manifest missing: {manifest_path}")
    _need(aggregate_path.is_file(), f"matrix aggregate missing: {aggregate_path}")
    manifest = _load_json(manifest_path)
    cells = validate_manifest_schema(manifest)
    manifest_sha = _sha256(manifest_path)
    bindings: dict[str, Any] = {"verification": "schema_only_in_test"}
    if verify_local_bindings:
        bindings = verify_workspace_bindings(
            manifest,
            workspace_root=workspace_root,
            nwb_root=(nwb_root or workspace_root / DEFAULT_NWB_REL),
        )

    resolver = PathResolver(path_mappings)
    records: list[dict[str, Any]] = []
    for cell in cells:
        closure_path = root / "cells" / f"{cell['output_key']}.json"
        _need(closure_path.is_file(), f"terminal closure missing: {cell['output_key']}")
        closure = _load_json(closure_path)
        records.append(
            validate_cell_closure(
                cell,
                closure,
                matrix_manifest_sha256=manifest_sha,
                resolver=resolver,
            )
        )

    # The frozen evaluator scores only complete SessionBatchSampler batches.
    # Replay the actual ordered indices before recomputing paired contrasts;
    # raw all-eligible audit hashes alone are insufficient when a tail batch is
    # dropped.  Schema-only transported-fixture tests intentionally skip this
    # payload replay, but a normal terminal verification never does.
    if verify_local_bindings:
        replay_root = nwb_root or workspace_root / DEFAULT_NWB_REL
        for record in records:
            replay = replay_evaluated_query_identity(record, nwb_root=replay_root)
            record["evaluated_query_digests"] = (
                replay["ordered_window_start_sha256"],
                replay["ordered_target_covariate_evalmask_sha256"],
                replay["ordered_query_identity_sha256"],
            )
            record["evaluated_query_replay"] = replay

    recomputed = recompute_terminal_aggregate(records, manifest_sha256=manifest_sha)
    reported = _load_json(aggregate_path)
    _compare_exact_structure(recomputed, reported)
    return {
        "schema": "rt_sparse_endpoint_stage2_terminal_verification_v1",
        "status": "PASS_INDEPENDENT_TERMINAL_VERIFICATION_READ_ONLY",
        "matrix_manifest_sha256": manifest_sha,
        "reported_aggregate_sha256": _sha256(aggregate_path),
        "verified_cells": 45,
        "verified_folds": 15,
        "verified_arms": list(ARMS),
        "seed": SEED,
        "historical_full_used": False,
        "workspace_bindings": bindings,
        "recomputed": recomputed,
        "non_interference": {
            "gpu_opened": False,
            "nwb_bytes_hashed_for_identity": bool(verify_local_bindings),
            "nwb_payload_parsed": bool(verify_local_bindings),
            "subprocess_started": False,
            "artifact_written": False,
        },
    }


def _synthetic_manifest() -> dict[str, Any]:
    """A no-data fixture used only to exercise the frozen schema."""

    digest = "a" * 64
    return {
        "schema": "rt_sparse_endpoint_stage2_matrix_manifest_v1",
        "status": "PREPARED_NOT_LAUNCHED",
        "contract": {"path": str(CONTRACT_REL), "sha256": digest},
        "root_readiness": {"path": str(READINESS_REL), "sha256": digest},
        "surfaces": {
            name: {"path": str(path), "sha256": digest}
            for name, path in SURFACE_RELS.items()
        },
        "launch_inputs": {
            "configs_yaml_tree": {"path": str(CONFIG_TREE_REL), "sha256": digest},
            "src_py_tree": {"path": str(SOURCE_TREE_REL), "sha256": digest},
            "teacher_checkpoint": {"path": str(TEACHER_REL), "bytes": 1, "sha256": digest},
        },
        "nwb_allowlist": [
            {
                "path": f"sub-C_ses-RT-{index:08d}_behavior+ecephys.nwb",
                "bytes": 1,
                "sha256": digest,
            }
            for index in range(15)
        ],
        "environment": {"schema_only": True},
        "matrix": {
            "folds": list(FOLDS),
            "arms": list(ARMS),
            "seed": SEED,
            "cells": _expected_cells(),
            "fresh_fit_count": 45,
            "outer_eval_count": 45,
            "base_experiment": "rt_clean_nested_loso_m24",
            "arm_override_only": True,
        },
    }


def preflight_schema_only() -> dict[str, Any]:
    """Validate schema logic with synthetic metadata and an explicitly empty result."""

    cells = validate_manifest_schema(_synthetic_manifest())
    empty_result_rejected = False
    try:
        recompute_terminal_aggregate([], manifest_sha256="a" * 64)
    except VerificationError:
        empty_result_rejected = True
    _need(empty_result_rejected, "schema preflight failed to reject an empty terminal result")
    return {
        "schema": "rt_sparse_endpoint_stage2_terminal_verifier_preflight_v1",
        "status": "PASS_SCHEMA_ONLY_NO_ARTIFACT_OR_TARGET_ACCESS",
        "synthetic_cells": len(cells),
        "empty_terminal_result_rejected": True,
        "imports_torch": False,
        "opens_nwb_payload": False,
        "starts_subprocess": False,
        "writes_artifact": False,
    }


def _parse_path_mapping(value: str) -> tuple[Path, Path]:
    _need("=" in value, "--path-map must be REMOTE_PREFIX=LOCAL_PREFIX")
    old, new = value.split("=", 1)
    _need(bool(old) and bool(new), "--path-map prefixes must be non-empty")
    return Path(old), Path(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--nwb-root", type=Path)
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="REMOTE=LOCAL",
        help="rebase immutable remote artifact paths after transport; repeat as needed",
    )
    parser.add_argument("--preflight-schema-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_schema_only:
        _need(args.artifact_root is None, "schema-only preflight must not receive --artifact-root")
        _need(not args.path_map and args.nwb_root is None, "schema-only preflight accepts no data paths")
        result = preflight_schema_only()
    else:
        _need(args.artifact_root is not None, "terminal verification requires --artifact-root")
        mappings = tuple(_parse_path_mapping(value) for value in args.path_map)
        result = verify_terminal_bundle(
            args.artifact_root,
            workspace_root=args.workspace_root,
            nwb_root=args.nwb_root,
            path_mappings=mappings,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except VerificationError as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(2) from error
