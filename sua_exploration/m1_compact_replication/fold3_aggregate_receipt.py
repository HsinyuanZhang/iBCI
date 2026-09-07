#!/usr/bin/env python3
"""Build a fail-closed descriptive aggregate for M1 folds 0/1/2.

This receipt intentionally treats the fold-0 E23 continuation, recovered fold 1,
and fold 2 as a descriptive development series.  It reports the frozen
non-inferiority threshold and never labels that threshold as an improvement
test or a significance test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping


SCHEMA = "m1_compact_b3s_f0_f1_f2_s42_aggregate_v1"
THRESHOLD = -0.03


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked path: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def immutable(path: Path, body: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(body)
    value["canonical_content_sha256"] = canonical(value)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o444)
        os.link(temp, path)
        temp.unlink()
    finally:
        if temp.exists():
            temp.unlink()
    return sha(path)


def read_gate(path: Path, fold: int, expected_schema: str, expected_status: str) -> dict[str, Any]:
    path = path.resolve()
    need(path.is_file() and not path.is_symlink(), f"gate missing: {path}")
    need(path.stat().st_mode & 0o777 == 0o444, f"gate mutable: {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    need(body.get("schema") == expected_schema, f"fold {fold} gate schema drift")
    need(body.get("status") == expected_status, f"fold {fold} gate status drift")
    need(
        body.get("canonical_content_sha256")
        == canonical({k: v for k, v in body.items() if k != "canonical_content_sha256"}),
        f"fold {fold} gate canonical drift",
    )
    scope = body.get("scope", {})
    need(scope.get("task") == "m1" and int(scope.get("fold")) == fold and int(scope.get("seed")) == 42, f"fold {fold} scope drift")
    for key in ("formal_opened", "minival_opened", "heldout_opened"):
        need(scope.get(key) is False, f"fold {fold} formal scope drift: {key}")
    metrics = body.get("metrics", {})
    b0 = float(metrics["b0"]["pooled_variance_weighted_r2"])
    b3s = float(metrics["b3s_zero4"]["pooled_variance_weighted_r2"])
    delta = float(metrics["b3s_zero4_minus_b0"])
    need(abs(delta - (b3s - b0)) < 1e-12, f"fold {fold} delta arithmetic drift")
    need("gate_threshold" in metrics and float(metrics["gate_threshold"]) == THRESHOLD, f"fold {fold} threshold drift")
    bindings = body.get("bindings", {})
    checkpoint_sha = {
        "b0": bindings.get("b0", {}).get("terminal_checkpoint", {}).get("sha256"),
        "b3s_zero4": bindings.get("b3s_zero4", {}).get("terminal_checkpoint", {}).get("sha256"),
    }
    need(all(isinstance(value, str) and len(value) == 64 for value in checkpoint_sha.values()), f"fold {fold} checkpoint binding missing")
    return {
        "fold": fold,
        "path": str(path),
        "sha256": sha(path),
        "schema": body["schema"],
        "status": body["status"],
        "target_session": scope.get("target_session"),
        "b0_r2": b0,
        "b3s_zero4_r2": b3s,
        "delta": delta,
        "checkpoint_sha256": checkpoint_sha,
        "epochs": int(bindings.get("b0", {}).get("terminal_checkpoint", {}).get("fit_loop", {}).get("epoch_processed", 0)),
        "global_step": int(bindings.get("b0", {}).get("terminal_checkpoint", {}).get("global_step", 0)),
    }


def build(paths: Mapping[int, Path], output: Path) -> dict[str, Any]:
    specs = {
        0: ("m1_compact_b3s_f0_s42_e23_gate_v1", "PASS_M1_COMPACT_B3S_F0_E23_NONINFERIORITY"),
        1: ("m1_compact_b3s_f1_s42_gate_v2", "PASS_M1_COMPACT_B3S_F1_S42_NONINFERIORITY"),
        2: ("m1_compact_b3s_f2_s42_gate_v2", "PASS_M1_COMPACT_B3S_F2_S42_NONINFERIORITY"),
    }
    folds = [read_gate(paths[fold], fold, *specs[fold]) for fold in (0, 1, 2)]
    deltas = [item["delta"] for item in folds]
    noninferiority_count = sum(delta >= THRESHOLD for delta in deltas)
    positive_count = sum(delta > 0.0 for delta in deltas)
    # Fold 0 is an E23 continuation (24 processed epochs), whereas folds 1/2
    # are fixed 12-epoch fresh fits.  Keep this mismatch explicit in the
    # interpretation instead of silently presenting a matched-budget claim.
    epochs = {str(item["fold"]): item["epochs"] for item in folds}
    body = {
        "schema": SCHEMA,
        "status": "PASS_M1_COMPACT_B3S_F0_F1_F2_ALL_NONINFERIORITY" if noninferiority_count == 3 else "STOP_M1_COMPACT_B3S_F0_F1_F2_NONINFERIORITY",
        "scope": {
            "task": "m1",
            "folds": [0, 1, 2],
            "seed": 42,
            "formal_opened": False,
            "minival_opened": False,
            "heldout_opened": False,
            "threshold": THRESHOLD,
            "development_only": True,
        },
        "per_fold": folds,
        "aggregate": {
            "deltas": deltas,
            "mean_delta": statistics.fmean(deltas),
            "median_delta": statistics.median(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "positive_count_delta_gt_0": positive_count,
            "noninferiority_count_delta_ge_threshold": noninferiority_count,
            "all_folds_noninferior": noninferiority_count == 3,
        },
        "interpretation": {
            "noninferiority_claim": "B3S-Zero4 is within 0.03 R2 of B0 on every listed development fold" if noninferiority_count == 3 else "all-fold non-inferiority not established",
            "improvement_claim": False,
            "improvement_gate": "not tested; the frozen -0.03 gate is non-inferiority, not a +0.03 improvement test",
            "positive_delta_count_is_descriptive_only": True,
            "significance": "no p-value or significance claim; n=3 folds",
            "training_budget_warning": {
                "epochs_by_fold": epochs,
                "same_epoch_budget": len(set(epochs.values())) == 1,
                "note": "fold 0 is E23/24-epoch continuation; folds 1 and 2 are fresh 12-epoch fits; aggregate is descriptive and not a strictly matched-training-budget comparison",
            },
            "claim_limit": "development evidence only; not formal held-out superiority",
        },
    }
    immutable(output.resolve(), body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--fold2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body = build({0: args.fold0, 1: args.fold1, 2: args.fold2}, args.output)
    print(json.dumps({"status": body["status"], "aggregate": body["aggregate"], "output": str(args.output.resolve()), "sha256": sha(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
