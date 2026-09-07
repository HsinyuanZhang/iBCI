#!/usr/bin/env python3
"""Write append-only transfer and matched-budget aggregate receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import tempfile
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "sua_exploration/m1_compact_replication/results"
THRESHOLD = -0.03


class ReceiptError(RuntimeError):
    pass


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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path: Path, label: str) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{label} missing/symlinked")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"{label} is not an object")
    need(path.stat().st_mode & 0o777 == 0o444, f"{label} mutable")
    need(value.get("canonical_content_sha256") == canonical({k: v for k, v in value.items() if k != "canonical_content_sha256"}), f"{label} canonical hash drift")
    return value


def immutable(path: Path, body: Mapping[str, Any]) -> str:
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
        return sha(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def gate_path(fold: int) -> Path:
    return RESULTS / f"M1_COMPACT_B3S_F{fold}_S42_E23_GATE_v1.json"


def execution_path(fold: int) -> Path:
    return RESULTS / f"M1_COMPACT_B3S_F{fold}_S42_E23_CONTINUATION_EXECUTION_v2.json"


def transfer_path(fold: int) -> Path:
    return RESULTS / f"M1_COMPACT_B3S_F{fold}_S42_E23_TRANSFER_v1.json"


def transfer(fold: int) -> dict[str, Any]:
    gate_file = gate_path(fold)
    state_file = execution_path(fold)
    gate = read(gate_file, f"fold-{fold} e23 gate")
    state = read(state_file, f"fold-{fold} e23 execution")
    need(gate.get("status", "").endswith(("NONINFERIORITY",)), f"fold-{fold} e23 gate status invalid")
    need(state.get("status", "").endswith("PAIR_TERMINAL"), f"fold-{fold} e23 execution not terminal")
    terminals = state.get("terminal_checkpoints", {})
    files: dict[str, Any] = {}
    for key in ("b0", "b3s_zero4"):
        item = terminals[key]
        ckpt = Path(item["path"])
        config = Path(item["artifact"]["resolved_config"]["path"])
        manifest = Path(item["source_only_manifest"]["path"])
        files[key] = {"checkpoint": {"path": str(ckpt.resolve()), "sha256": sha(ckpt), "bytes": ckpt.stat().st_size}, "resolved_config": {"path": str(config.resolve()), "sha256": sha(config), "bytes": config.stat().st_size}, "source_only_manifest": {"path": str(manifest.resolve()), "sha256": sha(manifest), "bytes": manifest.stat().st_size}}
    body = {"schema": f"m1_compact_b3s_f{fold}_s42_e23_transfer_v1", "status": f"PASS_M1_COMPACT_B3S_F{fold}_S42_E23_TRANSFER_MANIFEST", "scope": {"task": "m1", "fold": fold, "seed": 42, "target_session": gate.get("scope", {}).get("target_session"), "development_only": True, "formal_opened": False, "heldout_opened": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_checkpoint_selection": False}, "remote": {"hostname": platform.node(), "root": str(ROOT.resolve()), "transfer_direction": "remote_5070ti_to_workspace_receipt", "payload_copy_performed": False, "note": "Receipt binds immutable remote artifacts; no target NWB bytes are copied or hashed by this manifest."}, "bindings": {"gate": {"path": str(gate_file.resolve()), "sha256": sha(gate_file)}, "execution": {"path": str(state_file.resolve()), "sha256": sha(state_file)}}, "files": files, "metrics": gate.get("metrics", {}), "created_at_epoch": time.time()}
    output = transfer_path(fold)
    if output.exists():
        existing = read(output, f"fold-{fold} transfer receipt")
        need(canonical(body) == existing.get("canonical_content_sha256"), f"fold-{fold} existing transfer receipt differs")
        return existing
    immutable(output, body)
    return read(output, f"fold-{fold} transfer receipt")


def aggregate() -> dict[str, Any]:
    rows = []
    # Fold 0 is the already accepted e23 continuation; folds 1/2 are bound to
    # the new matched-budget continuation gates produced by this task.
    fold0_candidates = [
        RESULTS / "M1_COMPACT_B3S_F0_S42_E23_GATE_v1.json",
        RESULTS / "m1_e23_remote_pull_20260810_035000/receipts/M1_COMPACT_B3S_F0_S42_E23_GATE_v1.json",
    ]
    for fold in (0, 1, 2):
        candidates = fold0_candidates if fold == 0 else [gate_path(fold)]
        gate_file = next((p for p in candidates if p.exists()), None)
        need(gate_file is not None, f"fold-{fold} e23 gate missing")
        gate = read(gate_file, f"fold-{fold} e23 gate")
        need(gate.get("status", "").endswith("NONINFERIORITY"), f"fold-{fold} e23 non-inferiority failed")
        metrics = gate.get("metrics", {})
        delta = float(metrics["b3s_zero4_minus_b0"])
        need(delta >= THRESHOLD, f"fold-{fold} e23 gate below threshold")
        scope = gate.get("scope", {})
        need(scope.get("seed") == 42 and scope.get("formal_opened") is False and scope.get("heldout_opened") is False and scope.get("target_backward_steps") == 0 and scope.get("target_optimizer_steps") == 0 and scope.get("target_checkpoint_selection") is False, f"fold-{fold} gate scope drift")
        terminals = gate.get("bindings", {})
        b0 = terminals.get("b0", {}).get("terminal_checkpoint", {})
        b3 = terminals.get("b3s_zero4", {}).get("terminal_checkpoint", {})
        need(b0.get("epoch") == 23 and b3.get("epoch") == 23, f"fold-{fold} terminal epoch drift")
        need(b0.get("global_step") == b3.get("global_step"), f"fold-{fold} arm step mismatch")
        rows.append({"fold": fold, "target_session": scope.get("target_session"), "delta": delta, "b0_r2": metrics.get("b0", {}).get("pooled_variance_weighted_r2"), "b3s_zero4_r2": metrics.get("b3s_zero4", {}).get("pooled_variance_weighted_r2"), "global_step": b0.get("global_step"), "epoch": 23, "gate": {"path": str(gate_file.resolve()), "sha256": sha(gate_file), "schema": gate.get("schema"), "status": gate.get("status")}})
    deltas = [row["delta"] for row in rows]
    body = {"schema": "m1_compact_b3s_f0_f1_f2_s42_e23_aggregate_v1", "status": "PASS_M1_COMPACT_B3S_F0_F1_F2_E23_ALL_NONINFERIORITY", "scope": {"task": "m1", "seed": 42, "folds": [0, 1, 2], "threshold": THRESHOLD, "matched_terminal_epoch": 23, "formal_opened": False, "minival_opened": False, "heldout_opened": False, "development_only": True, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_checkpoint_selection": False}, "aggregate": {"all_folds_noninferior": all(delta >= THRESHOLD for delta in deltas), "noninferiority_count_delta_ge_threshold": sum(delta >= THRESHOLD for delta in deltas), "positive_count_delta_gt_0": sum(delta > 0 for delta in deltas), "deltas": deltas, "min_delta": min(deltas), "max_delta": max(deltas), "mean_delta": sum(deltas) / len(deltas), "median_delta": sorted(deltas)[len(deltas) // 2]}, "per_fold": rows, "interpretation": {"claim_limit": "development evidence only; not formal held-out superiority", "improvement_claim": False, "noninferiority_claim": "B3S-Zero4 is within 0.03 R2 of B0 on every listed development fold at matched terminal epoch 23", "same_epoch_budget": True, "no_formal_heldout_or_quantization_or_new_seed": True}, "created_at_epoch": time.time()}
    output = RESULTS / "M1_COMPACT_B3S_F0_F1_F2_S42_E23_AGGREGATE_v1.json"
    if output.exists():
        existing = read(output, "e23 aggregate receipt")
        need(canonical(body) == existing.get("canonical_content_sha256"), "existing aggregate differs")
        return existing
    immutable(output, body)
    return read(output, "e23 aggregate receipt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", type=int, choices=(1, 2))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    need((args.transfer is not None) ^ args.aggregate, "choose exactly one --transfer FOLD/--aggregate")
    body = transfer(args.transfer) if args.transfer is not None else aggregate()
    print(json.dumps({"status": body["status"], "path": str((transfer_path(args.transfer) if args.transfer is not None else RESULTS / "M1_COMPACT_B3S_F0_F1_F2_S42_E23_AGGREGATE_v1.json").resolve()), "sha256": sha(transfer_path(args.transfer) if args.transfer is not None else RESULTS / "M1_COMPACT_B3S_F0_F1_F2_S42_E23_AGGREGATE_v1.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
