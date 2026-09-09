#!/usr/bin/env python3
"""Create a read-only, fail-closed receipt for the original M2 last-two-date report.

This program never trains, scores, copies arrays, or chooses an epoch.  It
binds the completed source7 M2 Z/B/D runs to their already-audited EXT4 score
artifacts and reports only the fixed 2020-11-18 and 2020-11-19 targets.  The
two 2020-10-30 runs remain explicitly recorded as a supplement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from scripts.cross_session_v1 import summarize_figure as original  # noqa: E402
from scripts.cross_session_v1 import audit_source_pairing as pairing_impl  # noqa: E402
from tfpd_exploration.src.m2_dual_track_v1 import plan  # noqa: E402


ARMS = ("Z_NONE", "B_ACTIVITY_ONLY", "D_JOINT")
FOLD = "source7_ext4"
FIXED_TARGETS = ("ses-2020-11-18-Run1", "ses-2020-11-19-Run1")
SUPPLEMENT_TARGETS = ("ses-2020-10-30-Run1", "ses-2020-10-30-Run2")
DATE = re.compile(r"^ses-(\d{4}-\d{2}-\d{2})-Run[12]$")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required JSON is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def date_of(session: str) -> str:
    match = DATE.fullmatch(session)
    if not match:
        raise RuntimeError(f"unrecognised M2 session identifier: {session}")
    return match.group(1)


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError(f"{label} must be a lowercase SHA-256")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def m2_cells(old_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    continuation_path = old_root / "program_continuation.json"
    program = read_json(continuation_path)
    cells = program.get("cells")
    if not isinstance(cells, list):
        raise RuntimeError("old continuation cells must be a list")
    found = [cell for cell in cells if isinstance(cell, dict) and cell.get("dataset") == "m2"]
    if len(found) != 3:
        raise RuntimeError("old continuation must contain exactly three M2 cells")
    result = {str(cell.get("arm")): cell for cell in found}
    if set(result) != set(ARMS) or any(cell.get("status") != "COMPLETED" or cell.get("fold") != FOLD
                                      or cell.get("seed") != 42 for cell in result.values()):
        raise RuntimeError("old continuation M2 cell identity/status mismatch")
    snapshot = old_root.parent / "chronological_last2_v1" / "previous_program_continuation.json"
    snapshot_payload = read_json(snapshot)
    snap_cells = snapshot_payload.get("cells")
    snap_m2 = [cell for cell in snap_cells if isinstance(cell, dict) and cell.get("dataset") == "m2"] if isinstance(snap_cells, list) else []
    if sorted(snap_m2, key=lambda x: x["arm"]) != sorted(found, key=lambda x: x["arm"]):
        raise RuntimeError("preserved continuation M2 ledger differs from current continuation")
    return result, {"continuation_path": str(continuation_path.resolve()), "continuation_sha256": sha(continuation_path),
                    "snapshot_path": str(snapshot.resolve()), "snapshot_sha256": sha(snapshot)}


def source_pairing_audit(old_root: Path) -> dict[str, Any]:
    path = old_root / "m2_source_pairing_audit.json"
    audit = read_json(path)
    required = ("same_all_epoch_batch_manifest", "same_source_code_hashes",
                "same_core_source_cache_hashes", "same_B_D_activity_cache_hashes")
    if audit.get("status") != "PASS" or audit.get("dataset") != "m2" or audit.get("fold") != FOLD or audit.get("seed") != 42:
        raise RuntimeError("M2 source-pairing audit identity/status mismatch")
    if any(audit.get(key) is not True for key in required):
        raise RuntimeError("M2 source-pairing audit does not establish paired source protocol")
    hashes = audit.get("run_metadata_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(ARMS):
        raise RuntimeError("M2 source-pairing audit lacks all-arm metadata hashes")
    for arm in ARMS:
        require_hash(hashes[arm], f"source-pairing audit {arm} metadata hash")
    return {"path": str(path.resolve()), "sha256": sha(path), "receipt": audit}


def paired_artifact_audit(old_root: Path, pairing_sha: str) -> dict[str, Any]:
    path = old_root / "m2_zbd_paired_artifact_audit.json"
    audit = read_json(path)
    if audit.get("status") != "PASSED" or audit.get("source_pairing_audit_sha256") != pairing_sha:
        raise RuntimeError("M2 paired-artifact audit identity/source-audit binding mismatch")
    arms = audit.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise RuntimeError("M2 paired-artifact audit lacks all arms")
    return {"path": str(path.resolve()), "sha256": sha(path), "receipt": audit}


def zero_query_inventory(dest: Path) -> dict[str, Any]:
    """Bind the root-owned proof that 11/24 cannot supply a query target."""
    path = dest.parent / "m2_zero_query_date_inventory.json"
    inventory = read_json(path)
    rows = inventory.get("rows")
    expected = set(plan.EXCLUDED_EXTERNAL_SESSIONS)
    if inventory.get("status") != "VERIFIED_FROM_NWB_TRIAL_METADATA" or not isinstance(rows, list):
        raise RuntimeError("M2 zero-query inventory identity/status mismatch")
    by_session = {row.get("session"): row for row in rows if isinstance(row, dict)}
    if set(by_session) != expected:
        raise RuntimeError("M2 zero-query inventory session roster mismatch")
    for session in expected:
        row = by_session[session]
        if row.get("available_trials") != 33 or row.get("M33_support_trials") != 33 or row.get("post_M33_query_trials") != 0:
            raise RuntimeError(f"M2 zero-query inventory does not establish M33 exclusion: {session}")
    return {"path": str(path.resolve()), "sha256": sha(path), "receipt": inventory}


def checked_rows(cell: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the existing strict artifact validator before reading receipt rows."""
    checked = original.extract(cell)
    run = Path(checked["run"])
    score_path = run / "score_receipt.json"
    score = read_json(score_path)
    if score.get("status") != "COMPLETED" or score.get("schema") != "cross_session_m2_ext4_score_audit_v1":
        raise RuntimeError(f"strict score receipt identity/status mismatch: {run}")
    if score.get("target_query_labels_used_for_selection") is not False or score.get("target_query_labels_used_for_gradients") is not False:
        raise RuntimeError(f"target-label leak in original score receipt: {run}")
    selected, fixed = score.get("selected_ema_ext4"), score.get("predeclared_e24_ext4_sensitivity")
    if not isinstance(selected, dict) or not isinstance(fixed, dict):
        raise RuntimeError(f"M2 selected/fixed reports absent: {run}")
    if score.get("source_trial_validation_selection") != read_json(run / "train_receipt.json").get("selection"):
        raise RuntimeError(f"original selected epoch is not bound to source training receipt: {run}")
    return checked, {"path": str(score_path.resolve()), "sha256": sha(score_path), "receipt": score,
                     "selected": selected, "fixed_e24": fixed}


def scientific_trains(cells: dict[str, dict[str, Any]], old_root: Path) -> dict[str, Any]:
    """Validate the current three completed training receipts and source caches."""
    bound: dict[str, str] = {}
    pairing = pairing_impl.audit_m2(FOLD, cells, old_root, bound)
    if pairing.get("status") != "PASS":
        raise RuntimeError("in-process M2 source-pairing audit did not pass")
    receipts: dict[str, Any] = {}
    for arm in ARMS:
        run = Path(cells[arm]["run"])
        meta, train = read_json(run / "run_meta.json"), read_json(run / "train_receipt.json")
        if train.get("schema") != "cross_session_m2_train_receipt_v1" or train.get("status") != "COMPLETED":
            raise RuntimeError(f"scientific M2 training receipt identity/status mismatch: {arm}")
        # ``train.selection`` is the realized source-validation winning row;
        # ``run_meta`` instead holds the configured selection surface/rule.
        # They are intentionally different schemas and are bound below through
        # the full 24-epoch source curve and the score receipt.
        if any(train.get(key) != meta.get(key) for key in ("cell", "arm", "seed", "epochs", "source_hashes", "cache_hashes")):
            raise RuntimeError(f"scientific M2 training receipt/meta binding mismatch: {arm}")
        if meta.get("source_train_only_for_gradients") is not True or meta.get("epoch_selection") != "earliest_best_source_train_last20pct_whole_trials_ema":
            raise RuntimeError(f"scientific M2 source-only metadata contract mismatch: {arm}")
        if (train.get("target_query_labels_used_for_selection") is not False or train.get("target_query_labels_used_for_gradients") is not False
                or meta.get("target_query_labels_used_for_selection") is not False or meta.get("target_query_labels_used_for_gradients") is not False):
            raise RuntimeError(f"scientific M2 training target-label leak: {arm}")
        curve, selection = train.get("source_trial_val_ema_by_epoch"), train.get("selection")
        if not isinstance(curve, dict) or set(curve) != {str(epoch) for epoch in range(1, 25)} or not isinstance(selection, dict):
            raise RuntimeError(f"scientific M2 source selection evidence missing: {arm}")
        epoch = selection.get("epoch")
        if not isinstance(epoch, int) or not 1 <= epoch <= 24 or selection.get("equal_session_mean") != curve[str(epoch)].get("equal_session_mean"):
            raise RuntimeError(f"scientific M2 selected source row does not bind its curve: {arm}")
        receipts[arm] = {"path": str((run / "train_receipt.json").resolve()), "sha256": sha(run / "train_receipt.json"),
                         "run_meta_path": str((run / "run_meta.json").resolve()), "run_meta_sha256": sha(run / "run_meta.json")}
    return {"in_process_audit": pairing, "in_process_bound_input_sha256": dict(sorted(bound.items())),
            "train_receipt_schema_note": "cross_session_m2_train_receipt_v1 has no split_contract; its source hashes/cache hashes are directly bound to run_meta, while audit_m2 verifies the current run_meta split_contract and 24-epoch manifest across Z/B/D",
            "train_receipts": receipts, "source_cache_inputs": pairing["core_source_cache_hashes"]}


def package_files(score: dict[str, Any], arm: str, selected_epoch: int) -> dict[str, Any]:
    packages = score.get("ema_packages")
    if not isinstance(packages, dict):
        raise RuntimeError(f"{arm}: EMA package declarations absent")
    selected, e24 = packages.get("selected_source_trial_val"), packages.get("e24")
    if not isinstance(selected, dict) or not isinstance(e24, dict):
        raise RuntimeError(f"{arm}: selected/e24 package declarations absent")
    if selected.get("exact_tensor_equal_to_declared_epoch_package") is not True or selected.get("declared_source_validation_epoch") != selected_epoch:
        raise RuntimeError(f"{arm}: selected EMA package provenance mismatch")
    checks = (("selected", selected.get("path"), selected.get("sha256_before_score"), selected.get("sha256_after_score")),
              ("declared_selected_epoch", selected.get("declared_epoch_path"), selected.get("declared_epoch_sha256"), selected.get("declared_epoch_sha256")),
              ("e24", e24.get("path"), e24.get("sha256_before_score"), e24.get("sha256_after_score")))
    verified: dict[str, Any] = {}
    for label, raw_path, before, after in checks:
        path = Path(str(raw_path))
        if not path.is_absolute() or not path.is_file() or require_hash(before, f"{arm}/{label} before") != require_hash(after, f"{arm}/{label} after") or sha(path) != before:
            raise RuntimeError(f"{arm}: declared {label} EMA package file/SHA mismatch")
        verified[label] = {"path": str(path), "sha256": before}
    if e24.get("declared_epoch") != 24:
        raise RuntimeError(f"{arm}: fixed package is not e24")
    return verified


def summary_for(endpoint: dict[str, dict[str, dict[str, Any]]], sessions: tuple[str, ...]) -> dict[str, Any]:
    """Per-date equal-session values and equal-date mean; never pool windows."""
    by_date: dict[str, list[str]] = {}
    for session in sessions:
        by_date.setdefault(date_of(session), []).append(session)
    dates: dict[str, dict[str, float]] = {}
    for date, members in sorted(by_date.items()):
        arms = {arm: sum(float(endpoint[arm][session]["r2"]) for session in members) / len(members) for arm in ARMS}
        dates[date] = {**arms, "B_minus_Z": arms["B_ACTIVITY_ONLY"] - arms["Z_NONE"], "D_minus_B": arms["D_JOINT"] - arms["B_ACTIVITY_ONLY"],
                       "equal_session_count": len(members)}
    means = {arm: sum(row[arm] for row in dates.values()) / len(dates) for arm in ARMS}
    return {"per_date_equal_session": dates,
            "equal_date_mean": {**means, "B_minus_Z": means["B_ACTIVITY_ONLY"] - means["Z_NONE"],
                                "D_minus_B": means["D_JOINT"] - means["B_ACTIVITY_ONLY"], "date_count": len(dates)}}


def endpoint_rows(report: dict[str, Any], checked: dict[str, Any], endpoint: str) -> dict[str, dict[str, Any]]:
    rows = report.get("per_session")
    if not isinstance(rows, dict) or set(rows) != set(plan.EXT4_SESSIONS):
        raise RuntimeError(f"{endpoint}: EXT4 roster mismatch")
    bindings = checked["sb"] if endpoint == "selected_source_epoch" else checked["fb"]
    result: dict[str, dict[str, Any]] = {}
    for session, row in rows.items():
        if not isinstance(row, dict):
            raise RuntimeError(f"{endpoint}/{session}: row is invalid")
        artifact = Path(str(row.get("artifact", ""))).resolve()
        if not artifact.is_file() or not artifact.is_absolute():
            raise RuntimeError(f"{endpoint}/{session}: absolute artifact path required")
        artifact_sha = require_hash(row.get("artifact_sha256"), f"{endpoint}/{session} artifact SHA")
        target_sha = require_hash(row.get("target_sha256"), f"{endpoint}/{session} target SHA")
        starts_sha = require_hash(row.get("eligible_starts_sha256"), f"{endpoint}/{session} eligible-start SHA")
        # extract() recomputed the canonical target and coordinate hashes from
        # this exact NPZ, verified its recorded artifact checksum, and
        # recomputed the recorded R2.  Bind the receipt's eligible-start hash
        # to the actual named coordinate array as well, so cross-arm equality
        # below means byte equality, rather than merely equal labels.
        with original.np.load(artifact, allow_pickle=False) as arrays:
            actual_starts_sha = hashlib.sha256(original.np.ascontiguousarray(arrays["eligible_starts"]).tobytes()).hexdigest()
        if target_sha != bindings[session][0] or starts_sha != actual_starts_sha:
            raise RuntimeError(f"{endpoint}/{session}: receipt array hashes differ from validated NPZ")
        result[session] = {"artifact_absolute_path": str(artifact), "artifact_sha256": artifact_sha,
                           "target_sha256": target_sha, "eligible_starts_sha256": starts_sha,
                           "window_count": row.get("window_count"), "r2": row.get("r2")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=ROOT / "results/cross_session_v1")
    parser.add_argument("--dest", type=Path, default=ROOT / "results/chronological_last2_v1/m2")
    args = parser.parse_args()
    old_root, dest = args.old_root.resolve(), args.dest.resolve()
    if dest.exists():
        raise RuntimeError(f"destination must be fresh: {dest}")
    cells, continuation_ledger = m2_cells(old_root)
    pairing = source_pairing_audit(old_root)
    artifact_audit = paired_artifact_audit(old_root, pairing["sha256"])
    scientific_training = scientific_trains(cells, old_root)
    zero_query = zero_query_inventory(dest)
    extracted: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        extracted[arm], receipts[arm] = checked_rows(cells[arm])
        run = Path(extracted[arm]["run"])
        if sha(run / "run_meta.json") != pairing["receipt"]["run_metadata_sha256"][arm]:
            raise RuntimeError(f"source-pairing audit metadata binding changed: {arm}")
        if (extracted[arm]["receipt_sha256"] != receipts[arm]["sha256"] or cells[arm].get("score_receipt") != receipts[arm]["path"]
                or cells[arm].get("score_sha256") != receipts[arm]["sha256"]):
            raise RuntimeError(f"continuation score ledger mismatch: {arm}")
        audited = artifact_audit["receipt"]["arms"][arm]
        if audited.get("receipt_sha256") != receipts[arm]["sha256"] or audited.get("selected_epoch") != extracted[arm]["epoch"]:
            raise RuntimeError(f"paired-artifact audit binding mismatch: {arm}")

    source_dates = [date_of(session) for session in plan.HELDIN_SESSIONS]
    fixed_dates = [date_of(session) for session in FIXED_TARGETS]
    if not all(source < target for source in source_dates for target in fixed_dates):
        raise RuntimeError("source7 must be strictly earlier than every fixed chronological target")
    endpoint_data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for endpoint, report_key in (("selected_source_epoch", "selected"), ("fixed_e24", "fixed_e24")):
        endpoint_data[endpoint] = {arm: endpoint_rows(receipts[arm][report_key], extracted[arm], endpoint) for arm in ARMS}
        for session in plan.EXT4_SESSIONS:
            pairs = {(endpoint_data[endpoint][arm][session]["target_sha256"], endpoint_data[endpoint][arm][session]["eligible_starts_sha256"])
                     for arm in ARMS}
            if len(pairs) != 1:
                raise RuntimeError(f"{endpoint}/{session}: Z/B/D target or eligible-start arrays differ")
    # The main report is fixed before any target arrays are considered: exactly
    # one run on each of the last two query dates, with 10/30 retained only as a supplement.
    for endpoint in endpoint_data:
        if set(endpoint_data[endpoint][ARMS[0]]) != set(FIXED_TARGETS) | set(SUPPLEMENT_TARGETS):
            raise RuntimeError(f"{endpoint}: original full-source EXT4 roster drift")

    selected_epochs = {arm: extracted[arm]["epoch"] for arm in ARMS}
    summaries = {endpoint: {"fixed_last2": summary_for(endpoint_data[endpoint], FIXED_TARGETS),
                            "oct30_supplement_only": summary_for(endpoint_data[endpoint], SUPPLEMENT_TARGETS)}
                 for endpoint in endpoint_data}
    payload = {
        "schema": "chronological_m2_reuse_v1", "status": "COMPLETED", "mode": "reference_existing_arrays_no_copy",
        "old_root": str(old_root), "continuation_ledger": continuation_ledger,
        "source_pairing_audit": pairing,
        "paired_artifact_audit": artifact_audit,
        "scientific_training": scientific_training,
        "post_m33_zero_query_inventory": zero_query,
        "source_files_sha256": {str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                                str((ROOT / "scripts/cross_session_v1/summarize_figure.py").resolve()): sha(ROOT / "scripts/cross_session_v1/summarize_figure.py"),
                                str((ROOT.parent / "tfpd_exploration/src/m2_dual_track_v1/plan.py").resolve()): sha(ROOT.parent / "tfpd_exploration/src/m2_dual_track_v1/plan.py")},
        "source7": {"sessions": list(plan.HELDIN_SESSIONS), "dates": source_dates,
                    "semantics": "original completed source7 training; fixed source-only selected EMA and predeclared e24 sensitivity"},
        "target_policy": {"main_fixed_sessions": list(FIXED_TARGETS), "main_fixed_dates": fixed_dates,
                          "identity_count": 2, "supplement_sessions": list(SUPPLEMENT_TARGETS),
                          "supplement_dates": sorted({date_of(s) for s in SUPPLEMENT_TARGETS}),
                          "excluded_after_m33": list(plan.EXCLUDED_EXTERNAL_SESSIONS),
                          "strict_source_earlier_than_main_targets": True},
        "arms": {},
        "report_summaries": summaries,
        "paired_target_array_gate": "for each endpoint and EXT4 session, all Z/B/D target_sha256 and eligible_starts_sha256 are byte-identical",
    }
    for arm in ARMS:
        score = receipts[arm]["receipt"]
        verified_packages = package_files(score, arm, selected_epochs[arm])
        payload["arms"][arm] = {
            "old_run": extracted[arm]["run"], "old_score_receipt": {"path": receipts[arm]["path"], "sha256": receipts[arm]["sha256"]},
            "source_selected_epoch": selected_epochs[arm], "source_selection": score["source_trial_validation_selection"],
            "verified_ema_package_files": verified_packages,
            "selected_source_epoch_full_ext4": endpoint_data["selected_source_epoch"][arm],
            "fixed_e24_full_ext4_sensitivity": endpoint_data["fixed_e24"][arm],
            "main_fixed_last2": {endpoint: {session: endpoint_data[endpoint][arm][session] for session in FIXED_TARGETS}
                                for endpoint in endpoint_data},
            "oct30_supplement": {endpoint: {session: endpoint_data[endpoint][arm][session] for session in SUPPLEMENT_TARGETS}
                                  for endpoint in endpoint_data},
        }
    atomic_json(dest / "m2_reuse_receipt.json", payload)
    print(json.dumps({"status": "COMPLETED", "receipt": str((dest / "m2_reuse_receipt.json").resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
