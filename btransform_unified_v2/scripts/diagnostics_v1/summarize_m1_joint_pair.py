#!/usr/bin/env python3
"""Strict receipt-only M1 joint B/D summary on the frozen visible HO-M10 face.

The script never trains or scores.  It CPU-loads completed checkpoints only to
verify their embedded identity and finite raw/EMA state before reading scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
TRAINER = ROOT / "scripts/rift_v1/m1_joint_train.py"
CELL = "M1-RIFT-R100-D4-JOINT-B3S-CONCAT-V1"
ARMS = {"B": "B_ACTIVITY_ONLY", "D": "D_JOINT"}
SEED = 42
EPOCHS, UPDATES_PER_EPOCH, TOTAL_UPDATES = 24, 6665, 159960
SOURCE_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HO_SESSIONS = ("20121004", "20121017", "20121024")
HO_WINDOWS = {"20121004": 1305, "20121017": 1295, "20121024": 1281}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path}: {exc}"
    return (value, None) if isinstance(value, dict) else (None, f"{path}: JSON object required")


def run_dir(root: Path, arm: str) -> Path:
    return root / f"results/rift_v1/m1_r100_joint_{arm.lower()}_s42_formal_v1"


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite receipt value")
    return result


def digest_ok(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def tensors_finite(values: Any, torch: Any) -> bool:
    """Require a nonempty tensor mapping with finite floating/complex tensors."""
    if not isinstance(values, dict) or not values:
        return False
    for value in values.values():
        if not torch.is_tensor(value) or value.numel() == 0:
            return False
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            return False
    return True


def contract_ok(contract: Any) -> bool:
    if not isinstance(contract, dict):
        return False
    sampler = contract.get("sampler")
    banks, queries = contract.get("bank_hashes"), contract.get("query_hashes")
    if not (contract.get("sessions") == list(SOURCE_SESSIONS) and contract.get("heldout_sessions") == [] and
            contract.get("total_windows") == 213336 and contract.get("updates_per_epoch") == UPDATES_PER_EPOCH and
            sampler == {"batch": 32, "seed": SEED, "shuffle": True, "balance": False, "reshuffle_each_epoch": False} and
            isinstance(banks, dict) and isinstance(queries, dict) and set(banks) == set(SOURCE_SESSIONS) and set(queries) == set(SOURCE_SESSIONS)):
        return False
    required_query_hashes = ("eval_mask_sha256", "window_starts_sha256")
    optional_query_hashes = ("padded_covariate_sha256", "padded_neural_sha256", "query_target_sha256", "m10_calib_sha256")
    if not digest_ok(contract.get("sampler_batch_sha256")):
        return False
    if not all(banks[s].get("budget") == 10 and digest_ok(banks[s].get("e0_sha256")) and digest_ok(banks[s].get("carrier_sha256")) and
               queries[s].get("query_pad_bins") == 99 and queries[s].get("window_count") == contract.get("windows_by_session", {}).get(s) and
               all(digest_ok(queries[s].get(key)) for key in required_query_hashes) and
               all(key not in queries[s] or digest_ok(queries[s][key]) for key in optional_query_hashes)
               for s in SOURCE_SESSIONS):
        return False
    raw_m10 = contract.get("raw_m10_calib_sha256")
    return raw_m10 is None or (isinstance(raw_m10, dict) and set(raw_m10) == set(SOURCE_SESSIONS) and all(digest_ok(raw_m10[s]) for s in SOURCE_SESSIONS))


def ho_ok(contract: Any) -> bool:
    required = ("body_sha256", "carrier_sha256", "e0_sha256", "starts_sha256", "target_sha256", "neural_sha256", "covariate_sha256", "calib10_sha256")
    return isinstance(contract, dict) and contract.get("sessions") == list(HO_SESSIONS) and contract.get("total_windows") == 3881 and contract.get("query_pad_bins") == 99 and isinstance(contract.get("per_session"), dict) and all(contract["per_session"].get(session, {}).get("window_count") == count and all(digest_ok(contract["per_session"][session].get(key)) for key in required) for session, count in HO_WINDOWS.items())


def concat_ho_ok(contract: Any) -> bool:
    common = ("body_sha256", "carrier_sha256", "e0_sha256", "starts_sha256", "target_sha256")
    return isinstance(contract, dict) and contract.get("sessions") == list(HO_SESSIONS) and contract.get("total_windows") == 3881 and contract.get("query_pad_bins") == 99 and isinstance(contract.get("per_session"), dict) and all(contract["per_session"].get(session, {}).get("window_count") == count and all(digest_ok(contract["per_session"][session].get(key)) for key in common) for session, count in HO_WINDOWS.items())


def metadata_ok(meta: dict[str, Any], arm: str) -> bool:
    fields = {
        "schema": "m1_rift_joint_train_v1", "status": "FORMAL", "cell": CELL, "task": "m1", "arm": ARMS[arm],
        "variant": "recency", "identity_interface": "live_b3s_concat", "identity_e0_dim": 100,
        "concat_token_width": 120, "seed": SEED, "sampler_seed": SEED, "context_bins": 100,
        "query_pad_bins": 99, "layer_windows": [25, 25, 25, 24], "depth": 4, "width": 256,
        "proj_dim": None, "attention_backend": "local", "epochs": EPOCHS, "batch": 32,
        "updates_per_epoch": UPDATES_PER_EPOCH, "total_updates": TOTAL_UPDATES, "ema_decay": 0.9995,
        "source_train_only_for_gradients": True, "development_surface": "held-out-calib M10, post-training EMA scan only",
        "official_test_used": False,
    }
    b3s = meta.get("b3s")
    return all(meta.get(key) == value for key, value in fields.items()) and contract_ok(meta.get("source_contract")) and isinstance(meta.get("source_hashes"), dict) and bool(meta["source_hashes"]) and all(digest_ok(value) for value in meta["source_hashes"].values()) and digest_ok(meta.get("initialization_sha256")) and isinstance(b3s, dict) and isinstance(b3s.get("sfix_path"), str) and digest_ok(b3s.get("sfix_sha256"))


def validate_checkpoint_identity(path: Path, meta: dict[str, Any], epoch: int) -> str | None:
    """Mirror the trainer's checkpoint contract without importing the trainer."""
    try:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch reports corrupt/incompatible serialization here.
        return f"checkpoint CPU load failed: {path}: {exc}"
    if not isinstance(state, dict):
        return f"checkpoint object is not a mapping: {path}"
    expected = {"schema": "m1_rift_joint_epoch_checkpoint_v1", "cell": CELL, "smoke": False, "epoch": epoch, "global_step": epoch * UPDATES_PER_EPOCH,
                "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "initialization_sha256": meta["initialization_sha256"]}
    if any(state.get(key) != value for key, value in expected.items()):
        return f"checkpoint identity mismatch: {path}"
    expected_config = {"arm": meta["arm"], "seed": SEED, "sampler_seed": SEED, "context_bins": 100, "proj_dim": None, "epochs": EPOCHS, "batch": 32,
                       "lr": 1e-4, "bias_mode": "recency", "attention_backend": "local", "b3s": meta["b3s"]}
    if state.get("config") != expected_config:
        return f"checkpoint configuration mismatch: {path}"
    raw, ema = state.get("raw_state_dict"), state.get("ema")
    if not tensors_finite(raw, torch) or not isinstance(ema, dict) or ema.get("decay") != 0.9995 or ema.get("n_updates") != epoch * UPDATES_PER_EPOCH or not tensors_finite(ema.get("shadow"), torch) or not set(ema["shadow"]).issubset(set(raw)):
        return f"checkpoint raw/EMA state is empty or non-finite: {path}"
    return None


def validate_arm(root: Path, arm: str) -> tuple[dict[str, Any] | None, list[str]]:
    run = run_dir(root, arm)
    paths = {name: run / name for name in ("run_meta.json", "train_receipt.json", "score_receipt.json")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return None, missing
    values: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, path in paths.items():
        value, error = load_json(path)
        if error:
            errors.append(error)
        elif value is not None:
            values[name] = value
    if errors:
        return None, errors
    meta, train, score = values["run_meta.json"], values["train_receipt.json"], values["score_receipt.json"]
    if not metadata_ok(meta, arm):
        return None, [f"invalid formal M1 metadata: {run}"]
    sfix = Path(meta["b3s"]["sfix_path"])
    if not sfix.is_file() or sha(sfix) != meta["b3s"]["sfix_sha256"]:
        return None, [f"B3S Sfix hash drift: {sfix}"]
    expected_train = {"schema": "m1_rift_joint_train_receipt_v1", "status": "COMPLETED", "cell": CELL, "arm": ARMS[arm], "seed": SEED, "sampler_seed": SEED, "epochs": EPOCHS, "steps": TOTAL_UPDATES, "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "b3s": meta["b3s"], "post_training_scoring_required": True}
    if any(train.get(key) != value for key, value in expected_train.items()):
        return None, [f"invalid completed train receipt: {run}"]
    expected_score = {"schema": "m1_rift_joint_ho_calib_epoch_scan_v1", "status": "COMPLETED", "cell": CELL, "arm": ARMS[arm], "seed": SEED, "sampler_seed": SEED, "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "b3s": meta["b3s"], "official_test_used": False}
    if any(score.get(key) != value for key, value in expected_score.items()) or not ho_ok(score.get("ho_contract")):
        return None, [f"invalid completed visible-HO score receipt: {run}"]
    rows = score.get("ema_by_epoch")
    checkpoint_hashes = score.get("checkpoint_sha256_by_epoch")
    if not isinstance(rows, dict) or not isinstance(checkpoint_hashes, dict) or set(rows) != {str(i) for i in range(1, EPOCHS + 1)} or set(checkpoint_hashes) != set(rows):
        return None, [f"receipt must contain all 24 EMA checkpoint rows: {run}"]
    means: dict[int, float] = {}
    for epoch in range(1, EPOCHS + 1):
        row, checkpoint = rows[str(epoch)], run / f"epoch_{epoch:03d}.pt"
        if not isinstance(row, dict) or not checkpoint.is_file() or checkpoint_hashes[str(epoch)] != sha(checkpoint):
            return None, [f"checkpoint hash/file mismatch: {checkpoint}"]
        checkpoint_error = validate_checkpoint_identity(checkpoint, meta, epoch)
        if checkpoint_error:
            return None, [checkpoint_error]
        per = row.get("per_session")
        if row.get("partial") is not False or row.get("n_windows") != 3881 or not isinstance(per, dict) or set(per) != set(HO_SESSIONS):
            return None, [f"incomplete visible-HO epoch row: {checkpoint}"]
        try:
            scores = [finite(per[s]["r2"]) for s in HO_SESSIONS]
            if any(per[s].get("window_count") != HO_WINDOWS[s] for s in HO_SESSIONS):
                raise ValueError("HO window count")
            mean = finite(row["equal_session_mean"])
        except (KeyError, TypeError, ValueError):
            return None, [f"non-finite/invalid visible-HO value: {checkpoint}"]
        if abs(mean - sum(scores) / len(scores)) > 1e-12:
            return None, [f"forged equal-session mean: {checkpoint}"]
        means[epoch] = mean
    selected_epoch = score.get("selection", {}).get("epoch")
    best = min(range(1, EPOCHS + 1), key=lambda epoch: (-means[epoch], epoch))
    if selected_epoch != best or score.get("selection", {}).get("rule") != "earliest maximum equal-session mean EMA":
        return None, [f"selection is not earliest maximum EMA: {run}"]
    for source, digest in meta["source_hashes"].items():
        path = Path(source)
        if not path.is_file() or sha(path) != digest:
            return None, [f"runtime source hash drift: {path}"]
    return {"run": str(run), "meta": meta, "score": score, "selected_epoch": best,
            "receipt_sha256": {name: sha(path) for name, path in paths.items()},
            "checkpoint_sha256_by_epoch": checkpoint_hashes}, []


def validate_concat_reference(root: Path) -> tuple[dict[str, str] | None, list[str]]:
    run = root / "results/rift_v1/m1_r100_concat_s42_formal_v1"
    paths = {name: run / name for name in ("run_meta.json", "train_receipt.json", "score_receipt.json")}
    if any(not path.is_file() for path in paths.values()):
        return None, [f"missing frozen concat receipt: {path}" for path in paths.values() if not path.is_file()]
    meta, train, score = (json.loads(paths[name].read_text(encoding="utf-8")) for name in paths)
    if not (meta.get("schema") == "m1_rift_concat_train_v1" and meta.get("status") == "FORMAL" and meta.get("seed") == SEED and meta.get("context_bins") == 100 and meta.get("concat_token_width") == 120 and contract_ok(meta.get("source_contract")) and train.get("status") == "COMPLETED" and train.get("epochs") == EPOCHS and train.get("steps") == TOTAL_UPDATES and score.get("schema") == "m1_rift_concat_ho_calib_epoch_scan_v1" and score.get("status") == "COMPLETED" and score.get("official_test_used") is False and concat_ho_ok(score.get("ho_contract"))):
        return None, [f"invalid frozen full-concat reference: {run}"]
    return {"receipt_sha256": {name: sha(path) for name, path in paths.items()}, "meta": meta, "score": score}, []


def common_ho(contract: dict[str, Any]) -> dict[str, Any]:
    keys = ("body_sha256", "carrier_sha256", "e0_sha256", "starts_sha256", "target_sha256", "window_count")
    return {"sessions": contract["sessions"], "total_windows": contract["total_windows"], "query_pad_bins": contract["query_pad_bins"], "per_session": {session: {key: contract["per_session"][session][key] for key in keys} for session in HO_SESSIONS}}


def common_source(contract: dict[str, Any]) -> dict[str, Any]:
    """Fields emitted by both concat and joint runners; joint adds raw-M10 hashes."""
    keys = ("sessions", "heldout_sessions", "total_windows", "windows_by_session", "updates_per_epoch", "sampler_batch_sha256", "sampler", "bank_hashes", "bank_report")
    query_keys = ("eval_mask_sha256", "query_pad_bins", "validity", "window_count", "window_starts_sha256")
    return {**{key: contract[key] for key in keys}, "query_hashes": {session: {key: contract["query_hashes"][session][key] for key in query_keys} for session in SOURCE_SESSIONS}}


def inputs_comparable(b: dict[str, Any], d: dict[str, Any], concat: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if b["meta"]["source_contract"] != d["meta"]["source_contract"]:
        errors.append("B/D source_contract drift")
    if b["meta"]["b3s"] != d["meta"]["b3s"]:
        errors.append("B/D B3S provenance drift")
    if b["score"]["ho_contract"] != d["score"]["ho_contract"]:
        errors.append("B/D held-out input contract drift")
    if common_source(b["meta"]["source_contract"]) != common_source(concat["meta"]["source_contract"]) or common_source(d["meta"]["source_contract"]) != common_source(concat["meta"]["source_contract"]):
        errors.append("joint/frozen-concat common source contract drift")
    concat_ho = common_ho(concat["score"]["ho_contract"])
    if common_ho(b["score"]["ho_contract"]) != concat_ho or common_ho(d["score"]["ho_contract"]) != concat_ho:
        errors.append("joint/frozen-concat common held-out input contract drift")
    return errors


def summarize(root: Path) -> dict[str, Any]:
    base = {"schema": "m1_joint_pair_summary_v1", "summarizer": {"path": str(SELF), "sha256": sha(SELF)},
            "trainer_source": {"path": str(TRAINER), "sha256": sha(TRAINER)},
            "scope": {"source_sessions": list(SOURCE_SESSIONS), "source_calibration": "M10", "source_windows": 213336,
                      "architecture": "full R100/D4 concat; live B3S encoder and decoder jointly trained", "seed": SEED,
                      "epochs": EPOCHS, "updates": TOTAL_UPDATES, "heldout_surface": "visible HO-calib M10; query_start_trial=0; support may overlap query recordings", "heldout_sessions": list(HO_SESSIONS), "heldout_windows": 3881,
                      "selection": "earliest maximum equal-session mean EMA across all 24 exact checkpoint files", "official_test_used": False,
                      "independence_unit": "one seed-42 paired B/D comparison; three sessions are within-one-seed repeated measurements", "no_cross_seed_ci_or_significance": True}}
    concat_hashes, errors = validate_concat_reference(root)
    rows: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        row, arm_errors = validate_arm(root, arm)
        errors.extend(arm_errors)
        if row is not None:
            rows[arm] = row
    if not errors and set(rows) == set(ARMS):
        errors.extend(inputs_comparable(rows["B"], rows["D"], concat_hashes))
    if errors:
        return {**base, "status": "PENDING", "missing_or_invalid": sorted(errors), "completed_arms": sorted(rows), "reason": "a paired mean is withheld until both completed B/D receipt triplets and the frozen concat reference validate"}
    b, d = rows["B"], rows["D"]
    def deltas(epoch_b: int, epoch_d: int) -> dict[str, Any]:
        b_row, d_row = b["score"]["ema_by_epoch"][str(epoch_b)], d["score"]["ema_by_epoch"][str(epoch_d)]
        sessions = {s: finite(d_row["per_session"][s]["r2"]) - finite(b_row["per_session"][s]["r2"]) for s in HO_SESSIONS}
        return {"D_minus_B_equal_session_mean": finite(d_row["equal_session_mean"]) - finite(b_row["equal_session_mean"]), "per_session_D_minus_B": sessions}
    return {**base, "status": "COMPLETED", "frozen_concat_receipt_sha256": concat_hashes["receipt_sha256"],
            "arms": {arm: {"run": rows[arm]["run"], "selected_epoch": rows[arm]["selected_epoch"], "receipt_sha256": rows[arm]["receipt_sha256"], "checkpoint_sha256_by_epoch": rows[arm]["checkpoint_sha256_by_epoch"]} for arm in ARMS},
            "independent_pick": {"B_epoch": b["selected_epoch"], "D_epoch": d["selected_epoch"], **deltas(b["selected_epoch"], d["selected_epoch"])},
            "fixed_epoch_24": deltas(24, 24), "interpretation": "One seed-42 paired comparison only. Per-session values are repeated measurements within that seed; no cross-seed CI, p-value, or significance claim is reported."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
