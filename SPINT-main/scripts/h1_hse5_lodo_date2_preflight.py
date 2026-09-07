#!/usr/bin/env python3
"""Fail-closed CPU preflight for an independent H-SE5 LODO date-2 pair.

No CUDA or Lightning Trainer is constructed.  The script first runs the dated
implementation on fold-0 and requires exact source/target parity with the
sealed fold-0 H-SE5 implementation.  It then audits the 19250108 source split
and strict query boundary before writing an immutable launch receipt.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1
from src.data.h1_sparse_event_endpoint import H1SparseEventDataModule, build_sparse_target_dataset
from src.data.h1_sparse_event_endpoint_dated import (
    FIXED_BATCH_SIZE,
    FIXED_EPOCHS,
    FIXED_SEED,
    H1SparseEventDatedBatchSampler,
    H1SparseEventDatedDataModule,
    H1SparseEventDatedSourceDataset,
    build_dated_sparse_source_assets,
    build_dated_sparse_target_dataset,
    lodo_source_sessions,
    lodo_target_sessions,
)
from src.data.h1_sparse_event_source_snapshot_dated import (
    DatedSparseSourceSnapshot,
    load_snapshot,
    write_snapshot,
)
from src.h1_m4_eb_normalized_v2_contract import sha256_file, write_immutable_json


OUTER_DATE = "19250108"
RECEIPT_SCHEMA = "h1_hse5_lodo_date2_cpu_preflight_v1"
FULL_EXPERIMENT = "h1_hse5_lodo_full_19250108"
ZERO_EXPERIMENT = "h1_hse5_lodo_zero5_19250108"
SEALED_FULL_EXPERIMENT = "h1_sparse_event_endpoint_full"
SEALED_ZERO_EXPERIMENT = "h1_sparse_event_endpoint_zero"
ARTIFACT_ROOT = ROOT / "pilot_artifacts/h1_hse5_lodo_19250108"
OFFICIAL_PREFLIGHT = ARTIFACT_ROOT / "H1_HSE5_LODO_DATE2_PREFLIGHT_v1.json"
OFFICIAL_SNAPSHOT = ARTIFACT_ROOT / "H1_HSE5_LODO_19250108_SOURCE_v1.npz"
OFFICIAL_SNAPSHOT_RECEIPT = ARTIFACT_ROOT / "H1_HSE5_LODO_19250108_SOURCE_v1.json"
OFFICIAL_CACHE_DIR = ARTIFACT_ROOT / "shared_source_cache"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _compose(experiment: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])


def _flat(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flat(value[key], name)) if isinstance(value[key], dict) else out.setdefault(name, value[key])
        return out
    return {prefix: value}


def _experiment_payload(name: str) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(ROOT / "configs/experiment" / f"{name}.yaml"), resolve=False)  # type: ignore[return-value]


def _config_checks() -> dict[str, Any]:
    full, zero = _compose(FULL_EXPERIMENT), _compose(ZERO_EXPERIMENT)
    sealed_full, sealed_zero = _compose(SEALED_FULL_EXPERIMENT), _compose(SEALED_ZERO_EXPERIMENT)
    allowed_pair = {"task_name", "tags", "pilot.arm", "pilot.zero_carrier"}
    left = _flat(_experiment_payload(FULL_EXPERIMENT)); right = _flat(_experiment_payload(ZERO_EXPERIMENT))
    differences = {key: {"full": left.get(key), "zero": right.get(key)} for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)}
    illegal_pair = {key: value for key, value in differences.items() if key not in allowed_pair}
    checks = {
        "full_composed_fold_date": str(full.pilot.fold_date) == OUTER_DATE,
        "zero_composed_fold_date": str(zero.pilot.fold_date) == OUTER_DATE,
        "full_sparse_dated_data": str(full.data._target_) == "src.data.h1_sparse_event_endpoint_dated.H1SparseEventDatedDataModule",
        "zero_sparse_dated_data": str(zero.data._target_) == "src.data.h1_sparse_event_endpoint_dated.H1SparseEventDatedDataModule",
        "full_sparse_dated_model": str(full.model._target_) == "src.models.h1_sparse_event_dated_module.H1SparseEventDatedLitModule",
        "zero_sparse_dated_model": str(zero.model._target_) == "src.models.h1_sparse_event_dated_module.H1SparseEventDatedLitModule",
        "arm_semantics": str(full.pilot.arm) == "full" and not bool(full.pilot.zero_carrier) and str(zero.pilot.arm) == "zero" and bool(zero.pilot.zero_carrier),
        "full_zero_pair_only_permitted_differences": not illegal_pair,
        "source_training_exactly_seed42_m4_50": all((int(item.seed) == FIXED_SEED, int(item.pilot.calibration_n_trials) == 4,
                                                         int(item.pilot.batch_size) == FIXED_BATCH_SIZE, int(item.pilot.fixed_terminal_epochs) == FIXED_EPOCHS,
                                                         int(item.trainer.max_epochs) == FIXED_EPOCHS, int(item.trainer.min_epochs) == FIXED_EPOCHS,
                                                         str(item.trainer.precision) == "32-true") for item in (full, zero)),
        "not_ctxv2_placeholder": all("ctxv2" not in str(item.data._target_).lower() and "context" not in str(item.data._target_).lower() for item in (full, zero)),
        "sealed_fold0_is_distinct": str(sealed_full.data._target_) != str(full.data._target_) and str(sealed_zero.model._target_) != str(zero.model._target_),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"pass": all(checks.values()), "checks": checks, "pair_differences": differences, "illegal_pair_differences": illegal_pair,
            "config_shas": {name: sha256_file(ROOT / "configs/experiment" / f"{name}.yaml") for name in (FULL_EXPERIMENT, ZERO_EXPERIMENT)}}


def _fold0_fidelity(data_dir: Path) -> dict[str, Any]:
    """Prove date-parameterised code reproduces fold-0 without delegation."""
    old = H1SparseEventDataModule(task="h1", data_dir=str(data_dir), cache_dir=str(ROOT / "pilot_artifacts/h1_sparse_event_endpoint/fidelity_old"))
    old.setup("fit")
    with tempfile.TemporaryDirectory(prefix="hse5_fold0_fidelity_") as temp_dir:
        dated_live = H1SparseEventDatedDataModule(task="h1", data_dir=str(data_dir), cache_dir=temp_dir, fold_date="19250101")
        dated_live.setup("fit")
        dated_live_manifest_sha = dated_live.pilot_manifest_sha256
    records, sessions, basis, cache, normalizer = build_dated_sparse_source_assets(data_dir=data_dir, fold_date="19250101")
    dataset = H1SparseEventDatedSourceDataset(records, cache, normalizer)
    sampler = H1SparseEventDatedBatchSampler(dataset)
    manifest = {"basis": basis.manifest(), "cache": cache.manifest, "normalizer": normalizer.manifest,
                "source_window_indices_sha256": dataset.window_indices_sha256, "batch_order_sha256": sampler.batch_order_sha256,
                "calibration_schedule_sha256": sampler.schedule_sha256, "source_sessions": list(cache.source_sessions)}
    old_target = build_sparse_target_dataset(data_dir=data_dir, source_module=old)
    class Proxy:
        _setup_done = True
        hparams = type("Params", (), {"fold_date": "19250101"})()
        def __init__(self): self.basis, self.normalizer = basis, normalizer
    dated_target = build_dated_sparse_target_dataset(data_dir=data_dir, source_module=Proxy())
    old_hashes = old_target.support_and_carrier_hashes(); dated_hashes = dated_target.support_and_carrier_hashes()
    checks = {
        "source_session_order": tuple(cache.source_sessions) == tuple(old.carrier_cache.starts_by_session),
        "basis_manifest": basis.manifest() == old.basis.manifest(),
        "cache_sha": cache.manifest["cache_sha256"] == old.carrier_cache.manifest["cache_sha256"],
        "normalizer_manifest": normalizer.manifest == old.normalizer.manifest,
        "source_window_sha": dataset.window_indices_sha256 == old.train_dataset.window_indices_sha256,
        "batch_order_sha": sampler.batch_order_sha256 == old.train_batch_sampler.batch_order_sha256,
        "schedule_sha": sampler.schedule_sha256 == old.train_batch_sampler.schedule_sha256,
        "target_order": tuple(dated_target.target_sessions) == tuple(old_target.records),
        "target_window_count": len(dated_target) == len(old_target),
        "target_window_sha": dated_target.window_indices_sha256 == old_target.window_indices_sha256,
        "target_support_carrier_hashes": dated_hashes == old_hashes,
        "full_dated_manifest_sha": dated_live_manifest_sha == old.pilot_manifest_sha256,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"pass": all(checks.values()), "checks": checks, "dated": manifest,
            "sealed": {"source_manifest_sha256": old.pilot_manifest_sha256, "basis": old.basis.manifest(),
                       "cache_sha": old.carrier_cache.manifest["cache_sha256"], "normalizer": old.normalizer.manifest,
                       "source_window_sha": old.train_dataset.window_indices_sha256,
                       "batch_order_sha": old.train_batch_sampler.batch_order_sha256,
                       "schedule_sha": old.train_batch_sampler.schedule_sha256,
                       "target_window_sha": old_target.window_indices_sha256,
                       "full_manifest_sha": old.pilot_manifest_sha256},
            "dated_live_manifest_sha": dated_live_manifest_sha}


def _date2_assets(source_module: H1SparseEventDatedDataModule) -> dict[str, Any]:
    """Audit the exact live source module that will be snapshotted.

    In particular, this function must never rebuild PCA/cache state.  A fresh
    SVD in a helper process was the source of the v6/v7 binding failure: a
    separately generated snapshot can differ at the floating-point bit level
    from the source state that CPU preflight inspected.
    """

    _need(source_module._setup_done, "date-2 source module must finish source-only setup before audit")
    _need(not str(source_module.hparams.source_snapshot_receipt), "preflight must audit a live, unsnapshotted source module")
    records = source_module.records
    sessions = source_module.event_sessions
    basis = source_module.basis
    cache = source_module.carrier_cache
    normalizer = source_module.normalizer
    dataset = source_module.train_dataset
    sampler = source_module.train_batch_sampler
    _need(str(source_module.hparams.fold_date) == OUTER_DATE, "date-2 live source fold drift")
    dataset = H1SparseEventDatedSourceDataset(records, cache, normalizer); sampler = H1SparseEventDatedBatchSampler(dataset)
    class Proxy:
        _setup_done = True
        hparams = type("Params", (), {"fold_date": OUTER_DATE})()
        def __init__(self): self.basis, self.normalizer = basis, normalizer
    target = build_dated_sparse_target_dataset(data_dir=Path(str(source_module.hparams.data_dir)), source_module=Proxy())
    per_session = {name: {"query_windows": sum(session == name for session, _start in target.window_indices),
                          "support_events": len(tuple(event for event in target.event_sessions[name].events if event.trial_index < 4)),
                          "support_rank": int(np.linalg.matrix_rank(np.column_stack((np.ones(sum(event.trial_index < 4 for event in target.event_sessions[name].events)),
                              basis.transform(np.stack([event.displacement for event in target.event_sessions[name].events if event.trial_index < 4]))))))}
                   for name in target.target_sessions}
    checks = {
        "exact_target_sessions": target.target_sessions == ("ses-19250108T110520", "ses-19250108T111022", "ses-19250108T111455"),
        "exact_source_sessions": cache.source_sessions == lodo_source_sessions(OUTER_DATE) and len(cache.source_sessions) == 10,
        "source_has_no_target_date": all(event_v1.session_date(name) != OUTER_DATE for name in cache.source_sessions),
        "target_has_only_target_date": all(event_v1.session_date(name) == OUTER_DATE for name in target.target_sessions),
        "source_support_blocks": len(cache.entries) == 109,
        "batches_per_epoch": len(sampler) == 3356,
        "query_windows": len(target) == 13107 and target.window_indices_sha256 == "b0cd153750cb484af1237b7af1861600aca69a9b201244c22d140b42b1da7f6e",
        "query_per_recording": {name: item["query_windows"] for name, item in per_session.items()} == {"ses-19250108T110520": 8330, "ses-19250108T111022": 2508, "ses-19250108T111455": 2269},
        "support_events_and_rank": {name: (item["support_events"], item["support_rank"]) for name, item in per_session.items()} == {"ses-19250108T110520": (19, 5), "ses-19250108T111022": (23, 5), "ses-19250108T111455": (22, 5)},
        "basis": basis.basis_sha256 == "d07975b99004b2a5f61c403d86b3151057e4609e80d344b8e7657eebfcea155b" and basis.source_event_count == 697 and np.isclose(basis.retained_variance, 0.7225052486427831, rtol=0.0, atol=1e-15),
        "post_support_query": all(start >= target.support[name].query_first_bin and target.records[name].eval_mask[start + 699] for name, start in target.window_indices),
        "controls_distinct": all(len(set(item.carrier_sha256.values())) == 4 for item in target.support.values()),
        "all_source_support_fits_finite_rank5": all(
            np.isfinite(item.carrier).all() and int(np.linalg.matrix_rank(np.column_stack((np.ones(sum(event.trial_index in range(item.start_index, item.start_index + 4) for event in sessions[item.session_name].events)),
                basis.transform(np.stack([event.displacement for event in sessions[item.session_name].events if item.start_index <= event.trial_index < item.start_index + 4])))))) == 5
            for item in cache.entries),
        "minimum_source_support_events": min(sum(item.start_index <= event.trial_index < item.start_index + 4 for event in sessions[item.session_name].events) for item in cache.entries) == 17,
        "endpoint_parser_executable_path_never_names_dense_velocity": _endpoint_parser_ast_no_dense_velocity(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"pass": all(checks.values()), "checks": checks, "basis": basis.manifest(), "source_cache": cache.manifest,
            "normalizer": normalizer.manifest, "source_window_indices_sha256": dataset.window_indices_sha256,
            "batch_order_sha256": sampler.batch_order_sha256, "calibration_schedule_sha256": sampler.schedule_sha256,
            "source_manifest_sha256": source_module.pilot_manifest_sha256,
            "source_manifest": source_module.pilot_manifest(),
            "target": {"sessions": list(target.target_sessions), "query_window_indices_sha256": target.window_indices_sha256,
                       "support_and_carrier_hashes": target.support_and_carrier_hashes(), "per_session": per_session}}


def _endpoint_parser_ast_no_dense_velocity() -> bool:
    """Inspect executable string constants; comments/docstrings do not count."""
    parser_source = Path(event_v1.__file__).read_text(encoding="utf-8")
    estimator_source = Path(__import__("sua_exploration.mc_maze.h1_sparse_event_endpoint_v2", fromlist=["__file__"]).__file__).read_text(encoding="utf-8")
    def strings(source: str) -> list[str]:
        tree = ast.parse(source)
        return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    parser_strings, estimator_strings = strings(parser_source), strings(estimator_source)
    forbidden = "OpenLoopKinematicsVelocity"
    return not any(forbidden in value for value in parser_strings + estimator_strings)


def _snapshot_requirement() -> dict[str, Any]:
    """Check that both arms are wired to the sole official authority path."""
    # Hydra compose outside a runtime cannot resolve ${paths.work_dir}; check
    # the literal config interpolation and its common suffix instead.
    payloads = [_experiment_payload(name) for name in (FULL_EXPERIMENT, ZERO_EXPERIMENT)]
    values = [str(item["pilot"]["source_snapshot_receipt"]) for item in payloads]
    checks = {
        "same_configured_receipt_path": len(set(values)) == 1,
        "receipt_is_date2_authority_path": values[0].endswith("H1_HSE5_LODO_19250108_SOURCE_v1.json"),
        "data_config_receipt_wiring": all(
            "source_snapshot_receipt: ${pilot.source_snapshot_receipt}" in (ROOT / "configs/data/falcon_h1_sparse_event_endpoint_dated.yaml").read_text(encoding="utf-8")
            for _item in payloads
        ),
    }
    return {"pass": all(checks.values()), "checks": checks, "configured_receipt": values[0]}


def _binding_from_live_source(source_module: H1SparseEventDatedDataModule) -> dict[str, str]:
    """The seven source-derived values that training must consume verbatim."""

    manifest = source_module.pilot_manifest()
    return {
        "source_manifest_sha256": source_module.pilot_manifest_sha256,
        "basis_sha256": str(manifest["basis"]["basis_sha256"]),
        "carrier_cache_sha256": str(manifest["carrier_cache_sha256"]),
        "normalizer_sha256": str(manifest["normalizer_sha256"]),
        "source_window_indices_sha256": str(manifest["source_window_indices_sha256"]),
        "batch_order_sha256": str(manifest["batch_order_sha256"]),
        "schedule_sha256": str(manifest["calibration_schedule_sha256"]),
    }


def _binding_from_snapshot(snapshot: DatedSparseSourceSnapshot) -> dict[str, str]:
    return {
        "source_manifest_sha256": snapshot.manifest_sha256,
        "basis_sha256": snapshot.basis.basis_sha256,
        "carrier_cache_sha256": snapshot.cache.manifest["cache_sha256"],
        "normalizer_sha256": snapshot.normalizer.normalizer_sha256,
        "source_window_indices_sha256": snapshot.source_window_indices_sha256,
        "batch_order_sha256": snapshot.batch_order_sha256,
        "schedule_sha256": snapshot.schedule_sha256,
        "snapshot_receipt_sha256": snapshot.receipt_sha256,
        "snapshot_sha256": snapshot.snapshot_sha256,
    }


def validate_preflight_snapshot_binding(preflight: dict[str, Any], snapshot: DatedSparseSourceSnapshot) -> None:
    """Fail closed unless the launch authority is exactly the audited snapshot.

    This is also imported by the launch script.  Keeping the comparison here
    gives one testable definition rather than two subtly diverging shell/Python
    checks.
    """

    _need(preflight.get("status") == "PASS_HSE5_LODO_DATE2_PREFLIGHT", "preflight receipt status is not PASS")
    _need(str(preflight.get("outer_date")) == snapshot.basis.outer_date, "preflight/snapshot fold-date mismatch")
    expected = preflight.get("source_snapshot_binding")
    _need(isinstance(expected, dict), "preflight lacks source snapshot binding")
    observed = _binding_from_snapshot(snapshot)
    _need(expected == observed, "preflight/source snapshot binding mismatch")


def _path_contract(args: argparse.Namespace) -> None:
    """Official publication is only possible from this preflight process.

    A temp bundle is allowed for tests and probes, but it remains write-once
    and never aliases the official artifact names.  The standalone builder is
    deliberately prevented from creating the official authority.
    """

    paths = (args.output.resolve(), args.snapshot.resolve(), args.snapshot_receipt.resolve())
    official = (OFFICIAL_PREFLIGHT.resolve(), OFFICIAL_SNAPSHOT.resolve(), OFFICIAL_SNAPSHOT_RECEIPT.resolve())
    if args.temp_mode:
        _need(paths != official and all(path not in official for path in paths), "--temp-mode cannot write official H-SE5 authority paths")
    else:
        _need(paths == official, "official preflight requires the fixed official output/snapshot/receipt paths")
        _need(args.cache_dir.resolve() == OFFICIAL_CACHE_DIR.resolve(), "official preflight requires the fixed shared source cache path")
    _need(len(set(paths)) == 3, "preflight, snapshot, and snapshot receipt paths must be distinct")
    _need(not any(path.exists() for path in paths), "H-SE5 preflight/snapshot artifacts are write-once; choose a new temp bundle for tests")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--output", type=Path, default=OFFICIAL_PREFLIGHT)
    parser.add_argument("--snapshot", type=Path, default=OFFICIAL_SNAPSHOT)
    parser.add_argument("--snapshot-receipt", type=Path, default=OFFICIAL_SNAPSHOT_RECEIPT)
    parser.add_argument("--cache-dir", type=Path, default=OFFICIAL_CACHE_DIR)
    parser.add_argument("--temp-mode", action="store_true", help="write a non-official write-once test bundle")
    args = parser.parse_args(); data_dir = args.data_dir.resolve()
    args.output, args.snapshot, args.snapshot_receipt, args.cache_dir = (
        args.output.resolve(), args.snapshot.resolve(), args.snapshot_receipt.resolve(), args.cache_dir.resolve())
    _path_contract(args)
    config, fidelity, snapshot_requirement = _config_checks(), _fold0_fidelity(data_dir), _snapshot_requirement()
    # One live source-only module is the sole input to both the audit and the
    # final immutable snapshot.  Do not replace it with build_* helper output.
    live_source = H1SparseEventDatedDataModule(
        task="h1", data_dir=str(data_dir), cache_dir=str(args.cache_dir), fold_date=OUTER_DATE,
    )
    live_source.setup("fit")
    date2 = _date2_assets(live_source)
    passed = config["pass"] and fidelity["pass"] and date2["pass"] and snapshot_requirement["pass"]
    source_snapshot_binding: dict[str, str] | None = None
    snapshot_publication: dict[str, Any] | None = None
    publication_error: str | None = None
    if passed:
        try:
            live_binding = _binding_from_live_source(live_source)
            snapshot_publication = write_snapshot(
                snapshot_path=args.snapshot, receipt_path=args.snapshot_receipt, source_module=live_source,
                builder_path=Path(__file__), expected_manifest_sha256=live_binding["source_manifest_sha256"],
            )
            published_snapshot = load_snapshot(args.snapshot_receipt)
            source_snapshot_binding = _binding_from_snapshot(published_snapshot)
            _need({key: source_snapshot_binding[key] for key in live_binding} == live_binding,
                  "published source snapshot differs from the live source module audited by preflight")
        except Exception as error:  # preserve a STOP receipt; never overwrite partial authority artifacts
            passed, publication_error = False, f"{type(error).__name__}: {error}"
    receipt = {"schema": RECEIPT_SCHEMA, "status": "PASS_HSE5_LODO_DATE2_PREFLIGHT" if passed else "STOP_HSE5_LODO_DATE2_PREFLIGHT",
               "outer_date": OUTER_DATE, "config": config, "fold0_fidelity": fidelity, "date2": date2,
               "snapshot_requirement": snapshot_requirement, "source_snapshot_binding": source_snapshot_binding,
               "snapshot_publication": snapshot_publication, "snapshot_publication_error": publication_error,
               "implementation": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in {
                   "dated_data": ROOT / "src/data/h1_sparse_event_endpoint_dated.py", "dated_module": ROOT / "src/models/h1_sparse_event_dated_module.py",
                   "preflight": Path(__file__), "full_config": ROOT / "configs/experiment" / f"{FULL_EXPERIMENT}.yaml",
                   "zero_config": ROOT / "configs/experiment" / f"{ZERO_EXPERIMENT}.yaml",
                   "data_config": ROOT / "configs/data/falcon_h1_sparse_event_endpoint_dated.yaml",
                   "model_config": ROOT / "configs/model/falcon_h1_sparse_event_endpoint_dated.yaml",
                   "launch": ROOT / "scripts/h1_hse5_lodo_date2_launch.sh",
                   "snapshot_module": ROOT / "src/data/h1_sparse_event_source_snapshot_dated.py",
                   "snapshot_builder": ROOT / "scripts/build_h1_sparse_event_source_snapshot_dated.py"}.items()},
               "scope": {"cuda_used": False, "training_launched": False, "target_optimizer_steps": 0, "target_backward_steps": 0,
                         "minival_opened": False, "heldout_opened": False, "formal_opened": False, "evalai_opened": False}}
    path, digest = write_immutable_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(path), "sha256": digest}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
