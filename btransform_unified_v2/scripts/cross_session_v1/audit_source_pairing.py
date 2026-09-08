#!/usr/bin/env python3
"""Fail-closed source-pairing audit for the 33-cell cross-session experiment.

This is deliberately metadata-only.  It opens JSON receipts, metadata and
manifests and hashes those files, but never maps or loads neural/target arrays
or target scoring artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from tfpd_exploration.src.m2_dual_track_v1 import plan as m2_plan


ARMS = ("Z_NONE", "B_ACTIVITY_ONLY", "D_JOINT")
EXPECTED = {
    "m1": {"ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928"},
    "m2": {"source7_ext4"},
    "h1": {"19250101", "19250108", "19250113", "19250115", "19250119", "19250120"},
}
H1_SHARED_META = (
    "epochs", "batch", "optimizer", "clip_norm", "ema", "schedule", "precision",
    "backend", "unit_dropout", "behavior_scale", "support", "source_train",
    "source_val", "target", "native_trial_ids", "source_manifest_sha256", "source_code_sha256",
)
M2_CORE_FILES = ("X_store.npy", "target_store.npy", "eligible_starts.npy", "mapping.json")
M2_SOURCE_SESSIONS = tuple(m2_plan.HELDIN_SESSIONS)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, bound: dict[str, str]) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required metadata file is missing: {path}")
    key = str(path.resolve())
    bound[key] = sha(path)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON: {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def nonempty_hash(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"{label} must be a nonempty lowercase SHA-256")
    return value


def hash_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        require(bool(value), f"{label} must not be empty")
        for key, item in value.items():
            require(isinstance(key, str) and key, f"{label} has an invalid key")
            hash_tree(item, f"{label}.{key}")
    else:
        nonempty_hash(value, label)


def exact_same(values: dict[str, Any], label: str) -> Any:
    first = values[ARMS[0]]
    require(all(values[arm] == first for arm in ARMS[1:]), f"{label} differs across Z/B/D")
    return first


def nonempty_value(value: Any, label: str) -> Any:
    require(value is not None, f"{label} must be present and non-null")
    if isinstance(value, (str, list, dict, tuple)):
        require(bool(value), f"{label} must not be empty")
    return value


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def program_cells(root: Path, bound: dict[str, str]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    program = load_json(root / "program.json", bound)
    cells = program.get("cells")
    require(program.get("schema") == "cross_session_calibration_program_v1", "unexpected program schema")
    require(program.get("primary_cell_count") == 33 and isinstance(cells, list) and len(cells) == 33,
            "program.json must canonically enumerate exactly 33 primary cells")
    require(all(isinstance(cell, dict) and cell.get("status") == "COMPLETED" for cell in cells),
            "all 33 program cells must be COMPLETED; partial auditing is forbidden")
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    seen: set[tuple[str, str, str, int]] = set()
    for cell in cells:
        task, fold, arm, seed = cell.get("dataset"), cell.get("fold"), cell.get("arm"), cell.get("seed")
        require(task in EXPECTED and fold in EXPECTED[task] and arm in ARMS and seed == 42,
                f"invalid primary cell identity: {cell}")
        key = (task, fold, arm, seed)
        require(key not in seen, f"duplicate primary cell: {key}")
        seen.add(key)
        grouped.setdefault((task, fold), {})[arm] = cell
    require(len(seen) == 33 and set(grouped) == {(task, fold) for task, folds in EXPECTED.items() for fold in folds},
            "program has a missing or unexpected task/fold")
    require(all(set(arms) == set(ARMS) for arms in grouped.values()), "every fold needs exactly Z/B/D")
    return grouped


def run_path(cell: dict[str, Any], task: str, fold: str, root: Path) -> Path:
    """Require an absolute, existing program-owned completed-run directory."""
    supplied = cell.get("run")
    require(isinstance(supplied, str) and supplied, f"{task}/{fold}/{cell.get('arm')}: COMPLETED cell lacks run path")
    path = Path(supplied)
    require(path.is_absolute() and path.is_dir(), f"{task}/{fold}/{cell.get('arm')}: run must be an existing absolute directory")
    return path


def audit_m1(fold: str, cells: dict[str, dict[str, Any]], root: Path, bound: dict[str, str]) -> dict[str, Any]:
    receipts = {arm: load_json(run_path(cells[arm], "m1", fold, root) / "train_receipt.json", bound) for arm in ARMS}
    for arm, receipt in receipts.items():
        require(receipt.get("schema") == "cross_session_m1_v1" and receipt.get("status") == "COMPLETED", f"M1 {fold}/{arm}: incomplete receipt")
        require(receipt.get("arm") == arm and receipt.get("fold") == fold and receipt.get("seed") == 42, f"M1 {fold}/{arm}: identity mismatch")
        for key in ("actual_source_arrays", "batch_order_sha256", "initialization_hash", "data_contract", "recipe", "source_code_sha256"):
            require(key in receipt, f"M1 {fold}/{arm}: missing {key}")
        hash_tree(receipt["actual_source_arrays"], f"M1 {fold}/{arm}.actual_source_arrays")
        nonempty_hash(receipt["batch_order_sha256"], f"M1 {fold}/{arm}.batch_order_sha256")
        nonempty_hash(receipt["initialization_hash"], f"M1 {fold}/{arm}.initialization_hash")
        hash_tree(receipt["source_code_sha256"], f"M1 {fold}/{arm}.source_code_sha256")
        require(isinstance(receipt["data_contract"], dict) and receipt["data_contract"], f"M1 {fold}/{arm}: data_contract must be nonempty")
        require(isinstance(receipt["recipe"], dict) and receipt["recipe"], f"M1 {fold}/{arm}: recipe must be nonempty")
    evidence = exact_same({arm: receipts[arm]["actual_source_arrays"] for arm in ARMS}, f"M1 {fold} actual_source_arrays")
    order = exact_same({arm: receipts[arm]["batch_order_sha256"] for arm in ARMS}, f"M1 {fold} batch order")
    data_contract = exact_same({arm: receipts[arm]["data_contract"] for arm in ARMS}, f"M1 {fold} data contract")
    recipe = exact_same({arm: receipts[arm]["recipe"] for arm in ARMS}, f"M1 {fold} recipe")
    source_code = exact_same({arm: receipts[arm]["source_code_sha256"] for arm in ARMS}, f"M1 {fold} source code")
    require(receipts["B_ACTIVITY_ONLY"]["initialization_hash"] == receipts["D_JOINT"]["initialization_hash"], f"M1 {fold}: B/D initialization differs")
    return {"status": "PASS", "source_arrays": evidence, "batch_order_sha256": order, "data_contract": data_contract,
            "recipe": recipe, "source_code_sha256": source_code,
            "B_D_initialization_hash": receipts["B_ACTIVITY_ONLY"]["initialization_hash"]}


def audit_h1(fold: str, cells: dict[str, dict[str, Any]], root: Path, bound: dict[str, str]) -> dict[str, Any]:
    iso_fold = f"{fold[:4]}-{fold[4:6]}-{fold[6:]}"
    manifest_path = root / "h1_prepared_v2" / iso_fold / "source" / "manifest.json"
    # Parse and bind the manifest rather than trusting only its receipt hash.
    load_json(manifest_path, bound)
    manifest_sha = bound[str(manifest_path.resolve())]
    trains: dict[str, dict[str, Any]] = {}
    metas: dict[str, dict[str, Any]] = {}
    bindings: dict[str, str] = {}
    for arm in ARMS:
        run = run_path(cells[arm], "h1", fold, root)
        trains[arm] = load_json(run / "train_receipt.json", bound)
        metas[arm] = load_json(run / "run_meta.json", bound)
        train, meta = trains[arm], metas[arm]
        require(train.get("schema") == "h1_lodo_train_receipt_v2", f"H1 {fold}/{arm}: train schema mismatch")
        require(meta.get("schema") == "h1_lodo_train_v2" and meta.get("arm") == arm and meta.get("seed") == 42,
                f"H1 {fold}/{arm}: run metadata identity mismatch")
        require(str(meta.get("fold")) in (fold, f"{fold[:4]}-{fold[4:6]}-{fold[6:]}"), f"H1 {fold}/{arm}: fold mismatch")
        meta_without_binding = {key: value for key, value in meta.items() if key != "recipe_binding"}
        require(train.get("meta") == meta_without_binding, f"H1 {fold}/{arm}: receipt metadata binding mismatch")
        recorded_binding = nonempty_hash(meta.get("recipe_binding"), f"H1 {fold}/{arm}.run_meta.recipe_binding")
        receipt_binding = nonempty_hash(train.get("recipe_binding"), f"H1 {fold}/{arm}.train_receipt.recipe_binding")
        recomputed_binding = canonical_sha(meta_without_binding)
        require(recorded_binding == receipt_binding == recomputed_binding,
                f"H1 {fold}/{arm}: recipe_binding does not bind the receipt/run metadata recipe")
        bindings[arm] = recorded_binding
        nonempty_hash(meta.get("source_manifest_sha256"), f"H1 {fold}/{arm}.source_manifest_sha256")
        require(meta["source_manifest_sha256"] == manifest_sha, f"H1 {fold}/{arm}: source manifest content SHA mismatch")
        nonempty_hash(meta.get("initial_parameter_sha256"), f"H1 {fold}/{arm}.initial_parameter_sha256")
        hash_tree(meta.get("source_code_sha256"), f"H1 {fold}/{arm}.source_code_sha256")
        hash_tree(train.get("source_files"), f"H1 {fold}/{arm}.source_files")
        curve = train.get("curve")
        require(isinstance(curve, list) and len(curve) == 32, f"H1 {fold}/{arm}: must contain exactly 32 epoch rows")
        for epoch, row in enumerate(curve, 1):
            require(isinstance(row, dict) and row.get("epoch") == epoch, f"H1 {fold}/{arm}: missing or unordered epoch {epoch}")
            nonempty_hash(row.get("sampler_endpoint_keep_sha256"), f"H1 {fold}/{arm}.curve[{epoch}]")
    shared = {}
    for key in H1_SHARED_META:
        values = {arm: nonempty_value(metas[arm].get(key), f"H1 {fold}/{arm}.{key}") for arm in ARMS}
        shared[key] = exact_same(values, f"H1 {fold} shared metadata {key}")
    source_files = exact_same({arm: trains[arm]["source_files"] for arm in ARMS}, f"H1 {fold} source file hashes")
    curves = {arm: [row["sampler_endpoint_keep_sha256"] for row in trains[arm]["curve"]] for arm in ARMS}
    sampler = exact_same(curves, f"H1 {fold} epoch sampler/dropout hashes")
    require(metas["B_ACTIVITY_ONLY"]["initial_parameter_sha256"] == metas["D_JOINT"]["initial_parameter_sha256"], f"H1 {fold}: B/D initialization differs")
    return {"status": "PASS", "shared_recipe_fields": shared, "source_files": source_files,
            "epoch_sampler_endpoint_keep_sha256": {str(i): sampler[i - 1] for i in range(1, 33)},
            "B_D_initial_parameter_sha256": metas["B_ACTIVITY_ONLY"]["initial_parameter_sha256"],
            "verified_recipe_binding_by_arm": bindings}


def audit_m2(fold: str, cells: dict[str, dict[str, Any]], root: Path, bound: dict[str, str]) -> dict[str, Any]:
    metas = {arm: load_json(run_path(cells[arm], "m2", fold, root) / "run_meta.json", bound) for arm in ARMS}
    for arm, meta in metas.items():
        require(meta.get("schema") == "cross_session_m2_train_v1" and meta.get("status") == "FORMAL", f"M2 {arm}: metadata schema/status mismatch")
        require(meta.get("arm") == arm and meta.get("seed") == 42 and meta.get("epochs") == 24, f"M2 {arm}: identity mismatch")
        hash_tree(meta.get("source_hashes"), f"M2 {arm}.source_hashes")
        require(isinstance(meta.get("split_contract"), dict) and isinstance(meta["split_contract"].get("manifest"), dict), f"M2 {arm}: split contract/24-epoch manifest missing")
        manifest = meta["split_contract"]["manifest"]
        require(int(manifest.get("epochs", -1)) == 24, f"M2 {arm}: manifest is not 24 epochs")
        batches = manifest.get("batches")
        require(isinstance(batches, dict) and set(batches) == {str(epoch) for epoch in range(1, 25)},
                f"M2 {arm}: manifest batches must have exactly epochs 1..24")
        require(all(isinstance(batches[str(epoch)], list) and batches[str(epoch)] for epoch in range(1, 25)),
                f"M2 {arm}: every manifest epoch must have nonempty batches")
        caches = meta.get("cache_hashes", {}).get("source_train")
        require(isinstance(caches, dict) and set(caches) == set(M2_SOURCE_SESSIONS), f"M2 {arm}: source cache roster drift")
        for session, row in caches.items():
            require(isinstance(session, str) and isinstance(row, dict), f"M2 {arm}: invalid source cache row")
            for name in M2_CORE_FILES:
                nonempty_hash(row.get(name), f"M2 {arm}.{session}.{name}")
        if arm in ("B_ACTIVITY_ONLY", "D_JOINT"):
            for session, row in caches.items():
                nonempty_hash(row.get("calib_activity_sha256"), f"M2 {arm}.{session}.calib_activity_sha256")
    split = exact_same({arm: metas[arm]["split_contract"] for arm in ARMS}, "M2 full split contract")
    sources = exact_same({arm: metas[arm]["source_hashes"] for arm in ARMS}, "M2 source code hashes")
    core = {arm: {s: {name: row[name] for name in M2_CORE_FILES} for s, row in metas[arm]["cache_hashes"]["source_train"].items()} for arm in ARMS}
    core_shared = exact_same(core, "M2 source X/target/eligible-start/mapping hashes")
    calib = {arm: {s: row["calib_activity_sha256"] for s, row in metas[arm]["cache_hashes"]["source_train"].items()} for arm in ("B_ACTIVITY_ONLY", "D_JOINT")}
    require(calib["B_ACTIVITY_ONLY"] == calib["D_JOINT"], "M2 B/D calibration activity hashes differ")
    manifest = split["manifest"]
    split_summary = {
        "canonical_json_sha256": canonical_sha(split),
        "schema": split.get("schema"), "source_surface": split.get("source_surface"),
        "selection_surface": split.get("selection_surface"), "support_trials": split.get("support_trials"),
        "validation_fraction": split.get("validation_fraction"), "rounding": split.get("rounding"),
        "session_ids": sorted(split.get("sessions", {})),
        "manifest_digest": nonempty_hash(manifest.get("digest"), "M2 split manifest.digest"),
        "manifest_schema": manifest.get("schema"), "manifest_epochs": manifest.get("epochs"),
        "manifest_batch_counts": {str(epoch): len(manifest["batches"][str(epoch)]) for epoch in range(1, 25)},
    }
    return {"status": "PASS", "split_contract_summary": split_summary, "source_hashes": sources,
            "core_source_cache_hashes": core_shared, "B_D_calib_activity_sha256": calib["B_ACTIVITY_ONLY"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="cross_session_v1 results root")
    parser.add_argument("--out", type=Path, required=True, help="new audit JSON path")
    args = parser.parse_args()
    root = args.root.resolve()
    require(root.is_dir(), f"results root does not exist: {root}")
    require(not args.out.exists(), f"refusing to overwrite audit output: {args.out}")
    bound: dict[str, str] = {}
    cells = program_cells(root, bound)
    evidence: dict[str, dict[str, Any]] = {"m1": {}, "m2": {}, "h1": {}}
    for (task, fold), trio in sorted(cells.items()):
        evidence[task][fold] = {"m1": audit_m1, "m2": audit_m2, "h1": audit_h1}[task](fold, trio, root, bound)
    payload = {
        "schema": "cross_session_source_pairing_audit_v1",
        "status": "PASS",
        "scope": "metadata-only source pairing; no source/target numeric arrays, predictions, coordinates, R2, or statistical inference were read",
        "program_contract": {"primary_cell_count": 33, "seed": 42, "arms": list(ARMS), "folds": {task: sorted(folds) for task, folds in EXPECTED.items()}},
        "audit_script_sha256": sha(Path(__file__).resolve()),
        "evidence": evidence,
        "bound_input_sha256": dict(sorted(bound.items())),
        "root_verification": "Run only after program.json marks every canonical cell COMPLETED. A PASS establishes paired source inputs/recipes/samplers and B/D initialization evidence; target-score validity and date aggregation remain the responsibility of summarize_figure.py.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(args.out)


if __name__ == "__main__":
    main()
