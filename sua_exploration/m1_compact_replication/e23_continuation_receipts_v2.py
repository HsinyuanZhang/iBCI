#!/usr/bin/env python3
"""Write an append-only matched-budget e23 aggregate receipt.

This is deliberately separate from the running continuation/evaluator
scripts.  It consumes only the three immutable gate receipts.  A valid gate
whose metric is below the fixed threshold is recorded as a metric STOP; a
receipt or scope/integrity problem is recorded separately as EVIDENCE_INVALID.
In either case an immutable aggregate is written, so a negative result cannot
silently disappear because a pass-only helper raised an exception.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "sua_exploration/m1_compact_replication/results"
THRESHOLD = -0.03
DELTA_ATOL = 1.0e-12
OUTPUT = RESULTS / "M1_COMPACT_B3S_F0_F1_F2_S42_E23_AGGREGATE_v2.json"


class ReceiptError(RuntimeError):
    """Aggregate receipt invariant failed (including unsafe reuse)."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked path: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_receipt(path: Path, label: str) -> dict[str, Any]:
    """Read an immutable, canonical receipt or fail closed."""
    need(path.is_file() and not path.is_symlink(), f"{label} missing/symlinked")
    need(path.stat().st_mode & 0o777 == 0o444, f"{label} mutable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReceiptError(f"{label} unreadable: {exc}") from exc
    need(isinstance(value, dict), f"{label} is not an object")
    try:
        expected = canonical({k: v for k, v in value.items() if k != "canonical_content_sha256"})
    except (TypeError, ValueError) as exc:
        raise ReceiptError(f"{label} contains non-canonical/non-finite JSON: {exc}") from exc
    need(value.get("canonical_content_sha256") == expected, f"{label} canonical hash drift")
    return value


def read_untrusted(path: Path, label: str) -> tuple[dict[str, Any], list[str]]:
    """Parse a gate while retaining enough data to emit EVIDENCE_INVALID."""
    errors: list[str] = []
    value: dict[str, Any] = {}
    if not path.is_file() or path.is_symlink():
        return value, [f"{label} missing/symlinked"]
    if path.stat().st_mode & 0o777 != 0o444:
        errors.append(f"{label} mutable")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            value = parsed
        else:
            errors.append(f"{label} is not an object")
    except (OSError, ValueError) as exc:
        errors.append(f"{label} unreadable: {exc}")
        return value, errors
    try:
        expected = canonical({k: v for k, v in value.items() if k != "canonical_content_sha256"})
        if value.get("canonical_content_sha256") != expected:
            errors.append(f"{label} canonical hash drift")
    except (TypeError, ValueError) as exc:
        errors.append(f"{label} non-canonical/non-finite JSON: {exc}")
    return value, errors


def immutable(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Atomically create a mode-0444 receipt; never overwrite."""
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(body)
    value["canonical_content_sha256"] = canonical(value)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return read_receipt(path, "new e23 aggregate receipt")


def gate_candidates(fold: int) -> list[Path]:
    name = f"M1_COMPACT_B3S_F{fold}_S42_E23_GATE_v1.json"
    if fold == 0:
        # Fold 0 is the pre-existing accepted continuation and may still be in
        # its immutable pull bundle rather than at the results root.
        return [RESULTS / name, RESULTS / "m1_e23_remote_pull_20260810_035000/receipts" / name]
    return [RESULTS / name]


def expected_schema(fold: int) -> str:
    return f"m1_compact_b3s_f{fold}_s42_e23_gate_v1"


def allowed_statuses(fold: int) -> tuple[set[str], set[str], set[str]]:
    """Return (all, pass, metric-failure) exact status tokens per fold."""
    if fold == 0:
        # Legacy e23_evaluate.py emits PASS/FAIL (not the newer STOP token).
        passed = {"PASS_M1_COMPACT_B3S_F0_E23_NONINFERIORITY"}
        failed = {"FAIL_M1_COMPACT_B3S_F0_E23_NONINFERIORITY"}
    else:
        passed = {f"PASS_M1_COMPACT_B3S_F{fold}_S42_E23_NONINFERIORITY"}
        failed = {f"STOP_M1_COMPACT_B3S_F{fold}_S42_E23_NONINFERIORITY"}
    return passed | failed, passed, failed


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def checkpoint_binding_errors(bindings: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(bindings, dict):
        return ["bindings missing/not an object"]
    for arm in ("b0", "b3s_zero4"):
        terminal = (bindings.get(arm) or {}).get("terminal_checkpoint")
        if not isinstance(terminal, dict):
            errors.append(f"{arm} terminal checkpoint binding missing")
            continue
        # Epoch and step are checked by fold_row; verify a supplied payload
        # path/SHA pair as well, without requiring payload transfer to this
        # workspace (the remote evaluator has these files).
        has_path = "path" in terminal
        has_sha = "sha256" in terminal
        if has_path or has_sha:
            if not (has_path and has_sha):
                errors.append(f"{arm} checkpoint path/SHA incomplete")
            else:
                checkpoint = Path(str(terminal.get("path", "")))
                try:
                    actual = sha(checkpoint)
                except ReceiptError as exc:
                    errors.append(f"{arm} checkpoint path invalid: {exc}")
                else:
                    if actual != terminal.get("sha256"):
                        errors.append(f"{arm} checkpoint SHA drift")
    return errors


def load_gate(fold: int) -> tuple[Path, dict[str, Any], list[str]]:
    candidates = gate_candidates(fold)
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    value, errors = read_untrusted(path, f"fold-{fold} e23 gate")
    return path, value, errors


def fold_row(fold: int, gate_path: Path, gate: dict[str, Any], initial_errors: list[str]) -> tuple[dict[str, Any], bool, bool]:
    errors = list(initial_errors)
    scope = gate.get("scope") if isinstance(gate.get("scope"), dict) else {}
    metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
    bindings = gate.get("bindings")
    if gate.get("schema") != expected_schema(fold):
        errors.append("gate schema drift")
    all_statuses, pass_statuses, failure_statuses = allowed_statuses(fold)
    status = gate.get("status")
    if status not in all_statuses:
        errors.append("gate status invalid")
    if scope.get("task") != "m1" or scope.get("fold") != fold or scope.get("seed") != 42:
        errors.append("scope task/fold/seed drift")
    target_session = scope.get("target_session")
    if not isinstance(target_session, str) or not target_session.strip():
        errors.append("target session missing/empty")
    if scope.get("support_trials") != [0, 10] or scope.get("query_trials") != [10, 210]:
        errors.append("query window drift")
    if (
        scope.get("formal_opened") is not False
        or scope.get("minival_opened") is not False
        or scope.get("heldout_opened") is not False
        or scope.get("target_backward_steps") != 0
        or scope.get("target_optimizer_steps") != 0
        or scope.get("target_checkpoint_selection") is not False
    ):
        errors.append("target-training or heldout policy drift")
    if isinstance(metrics, dict) and metrics.get("gate_threshold") != THRESHOLD:
        errors.append("gate threshold drift")
    b0_metrics = metrics.get("b0") if isinstance(metrics.get("b0"), dict) else {}
    b3_metrics = metrics.get("b3s_zero4") if isinstance(metrics.get("b3s_zero4"), dict) else {}
    b0_r2 = b0_metrics.get("pooled_variance_weighted_r2")
    b3_r2 = b3_metrics.get("pooled_variance_weighted_r2")
    delta_value = metrics.get("b3s_zero4_minus_b0")
    if not finite(b0_r2):
        errors.append("B0 R2 missing/non-finite")
    if not finite(b3_r2):
        errors.append("B3S R2 missing/non-finite")
    if not finite(delta_value):
        errors.append("delta missing/non-finite")
    if finite(b0_r2) and finite(b3_r2) and finite(delta_value):
        if not math.isclose(float(b3_r2) - float(b0_r2), float(delta_value), rel_tol=0.0, abs_tol=DELTA_ATOL):
            errors.append("delta inconsistent with B3S R2 minus B0 R2")
    checkpoint_errors = checkpoint_binding_errors(bindings)
    errors.extend(checkpoint_errors)
    terminals = bindings if isinstance(bindings, dict) else {}
    terminal_epochs: list[Any] = []
    terminal_steps: list[Any] = []
    for arm in ("b0", "b3s_zero4"):
        terminal = (terminals.get(arm) or {}).get("terminal_checkpoint")
        terminal = terminal if isinstance(terminal, dict) else {}
        terminal_epochs.append(terminal.get("epoch"))
        terminal_steps.append(terminal.get("global_step"))
    if terminal_epochs != [23, 23]:
        errors.append("terminal epoch drift")
    if not all(isinstance(step, int) and not isinstance(step, bool) and step > 0 for step in terminal_steps):
        errors.append("terminal global-step missing/non-positive")
    elif terminal_steps[0] != terminal_steps[1]:
        errors.append("arm global-step mismatch")
    row = {
        "fold": fold,
        "target_session": target_session,
        "delta": float(delta_value) if finite(delta_value) else None,
        "b0_r2": float(b0_r2) if finite(b0_r2) else None,
        "b3s_zero4_r2": float(b3_r2) if finite(b3_r2) else None,
        "global_step": terminal_steps[0] if terminal_steps[0] == terminal_steps[1] else None,
        "epoch": terminal_epochs[0] if terminal_epochs[0] == terminal_epochs[1] else None,
        "gate": {
            "path": str(gate_path.resolve()),
            "sha256": sha(gate_path) if gate_path.is_file() and not gate_path.is_symlink() else None,
            "schema": gate.get("schema"),
            "status": status,
        },
        "integrity_errors": errors,
    }
    valid_pass = bool(
        finite(delta_value) and not errors and status in pass_statuses and float(delta_value) >= THRESHOLD
    )
    valid_failure = bool(
        finite(delta_value) and not errors and status in failure_statuses and float(delta_value) < THRESHOLD
    )
    row["gate_result"] = "PASS" if valid_pass else "METRIC_FAILURE" if valid_failure else "EVIDENCE_INVALID"
    return row, valid_pass, valid_failure


def verify_existing(existing: dict[str, Any], current_shas: dict[str, str | None]) -> dict[str, Any]:
    rows = existing.get("per_fold")
    need(isinstance(rows, list) and len(rows) == 3, "existing aggregate per-fold rows incomplete")
    bound: dict[str, Any] = {}
    for row in rows:
        need(isinstance(row, dict) and isinstance(row.get("fold"), int), "existing aggregate row malformed")
        fold = str(row["fold"])
        need(fold in {"0", "1", "2"} and fold not in bound, "existing aggregate fold map malformed")
        gate = row.get("gate")
        need(isinstance(gate, dict), f"existing aggregate fold-{fold} gate binding missing")
        bound[fold] = gate.get("sha256")
    need(set(bound) == {"0", "1", "2"}, "existing aggregate does not bind all folds")
    need(bound == {str(fold): current_shas[str(fold)] for fold in (0, 1, 2)}, "existing aggregate gate SHA bindings drifted")
    return existing


def aggregate() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pass_flags: list[bool] = []
    failure_flags: list[bool] = []
    for fold in (0, 1, 2):
        gate_path, gate, read_errors = load_gate(fold)
        row, valid_pass, valid_failure = fold_row(fold, gate_path, gate, read_errors)
        rows.append(row)
        pass_flags.append(valid_pass)
        failure_flags.append(valid_failure)
    # A target session is a fold identity, not a free-form metric field.
    sessions: dict[str, list[int]] = {}
    for row in rows:
        session = row.get("target_session")
        if isinstance(session, str) and session.strip():
            sessions.setdefault(session, []).append(int(row["fold"]))
    for session, folds in sessions.items():
        if len(folds) > 1:
            for fold in folds:
                row = rows[fold]
                row["integrity_errors"].append(f"target session duplicate: {session}")
                row["gate_result"] = "EVIDENCE_INVALID"
                pass_flags[fold] = False
                failure_flags[fold] = False
    current_shas = {str(row["fold"]): row["gate"]["sha256"] for row in rows}
    if OUTPUT.exists() or OUTPUT.is_symlink():
        existing = read_receipt(OUTPUT, "existing e23 aggregate receipt")
        return verify_existing(existing, current_shas)
    deltas = [row["delta"] for row in rows if finite(row["delta"])]
    evidence_invalid = any(not errors == [] for errors in (row["integrity_errors"] for row in rows))
    metric_failure = any(failure_flags)
    all_pass = bool(all(pass_flags) and len(deltas) == 3)
    if evidence_invalid:
        status = "STOP_M1_COMPACT_B3S_F0_F1_F2_E23_EVIDENCE_INVALID"
    elif metric_failure:
        status = "STOP_M1_COMPACT_B3S_F0_F1_F2_E23_STOP_GATE_FAILED"
    elif all_pass:
        status = "PASS_M1_COMPACT_B3S_F0_F1_F2_E23_ALL_NONINFERIORITY"
    else:
        status = "STOP_M1_COMPACT_B3S_F0_F1_F2_E23_EVIDENCE_INVALID"
    body: dict[str, Any] = {
        "schema": "m1_compact_b3s_f0_f1_f2_s42_e23_aggregate_v2",
        "status": status,
        "scope": {
            "task": "m1",
            "seed": 42,
            "folds": [0, 1, 2],
            "threshold": THRESHOLD,
            "matched_terminal_epoch": 23,
            "formal_opened": False,
            "minival_opened": False,
            "heldout_opened": False,
            "development_only": True,
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "target_checkpoint_selection": False,
        },
        "aggregate": {
            "all_folds_noninferior": all_pass,
            "noninferiority_count_delta_ge_threshold": sum(finite(row["delta"]) and float(row["delta"]) >= THRESHOLD for row in rows),
            "positive_count_delta_gt_0": sum(finite(row["delta"]) and float(row["delta"]) > 0 for row in rows),
            "deltas": [row["delta"] for row in rows],
            "min_delta": min(deltas) if deltas else None,
            "max_delta": max(deltas) if deltas else None,
            "mean_delta": sum(deltas) / len(deltas) if deltas else None,
            "median_delta": sorted(deltas)[len(deltas) // 2] if deltas else None,
        },
        "per_fold": rows,
        "interpretation": {
            "claim_limit": "development evidence only; not formal held-out superiority",
            "improvement_claim": False,
            "noninferiority_claim": (
                "B3S-Zero4 is within 0.03 R2 of B0 on every listed development fold at matched terminal epoch 23"
                if all_pass
                else "No aggregate non-inferiority claim: at least one gate failed or evidence was invalid"
            ),
            "same_epoch_budget": True,
            "no_formal_heldout_or_quantization_or_new_seed": True,
        },
        "created_at_epoch": time.time(),
    }
    return immutable(OUTPUT, body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    need(args.aggregate, "use --aggregate")
    receipt = aggregate()
    print(json.dumps({"status": receipt["status"], "path": str(OUTPUT.resolve()), "sha256": sha(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
