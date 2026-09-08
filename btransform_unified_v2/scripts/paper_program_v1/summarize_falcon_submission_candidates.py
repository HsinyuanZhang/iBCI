#!/usr/bin/env python3
"""Build a read-only, receipt-bound local FALCON candidate inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

EPOCHS = {str(e) for e in range(1, 25)}
M1 = {"20121004": 1305, "20121017": 1295, "20121024": 1281}
M2 = {"ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1", "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2"}
M2_EXT4 = {"ses-2020-10-30-Run1": 519, "ses-2020-10-30-Run2": 490, "ses-2020-11-18-Run1": 425, "ses-2020-11-19-Run1": 635}
PAIR_KEYS = ("run_meta_sha256", "train_receipt_sha256", "score_receipt_sha256")
CONTROLS = {"C": "m2_r50_mechanism_c_s42_formal_v1", "shuffle": "m2_r50_mechanism_shuffle_s42_formal_v1", "mean": "m2_r50_mechanism_mean_s42_formal_v1", "nonattn": "m2_r50_mechanism_nonattn_s42_formal_v1"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def digest_ok(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def m1_source_contract_ok(contract: Any) -> bool:
    sessions = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
    if not isinstance(contract, dict) or not (
        contract.get("sessions") == list(sessions) and contract.get("heldout_sessions") == []
        and contract.get("total_windows") == 213336 and contract.get("updates_per_epoch") == 6665
        and contract.get("sampler") == {"batch": 32, "seed": 42, "shuffle": True, "balance": False, "reshuffle_each_epoch": False}
        and digest_ok(contract.get("sampler_batch_sha256"))
    ):
        return False
    banks, queries = contract.get("bank_hashes"), contract.get("query_hashes")
    return isinstance(banks, dict) and isinstance(queries, dict) and set(banks) == set(sessions) and set(queries) == set(sessions) and all(
        banks[session].get("budget") == 10 and digest_ok(banks[session].get("e0_sha256")) and digest_ok(banks[session].get("carrier_sha256"))
        and queries[session].get("query_pad_bins") == 99 and queries[session].get("window_count") == contract.get("windows_by_session", {}).get(session)
        and digest_ok(queries[session].get("eval_mask_sha256")) and digest_ok(queries[session].get("window_starts_sha256"))
        for session in sessions
    )


def m1_ho_contract_ok(contract: Any, joint: bool) -> bool:
    common = ("body_sha256", "carrier_sha256", "e0_sha256", "starts_sha256", "target_sha256")
    extra = ("neural_sha256", "covariate_sha256", "calib10_sha256") if joint else ()
    return isinstance(contract, dict) and contract.get("sessions") == list(M1) and contract.get("total_windows") == 3881 and contract.get("query_pad_bins") == 99 and isinstance(contract.get("per_session"), dict) and all(
        contract["per_session"].get(session, {}).get("window_count") == count and all(digest_ok(contract["per_session"][session].get(key)) for key in common + extra)
        for session, count in M1.items()
    )


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def finite(value: Any, label: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid {label}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {label}")
    return value


def receipts(run: Path) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths = {key: run / key for key in ("run_meta.json", "train_receipt.json", "score_receipt.json")}
    return paths, {key: load(path) for key, path in paths.items()}


def receipt_sha(paths: dict[str, Path]) -> dict[str, str]:
    return {key: sha(path) for key, path in paths.items()}


def rows(score: dict[str, Any], counts: dict[str, int]) -> tuple[int, dict[str, Any], float]:
    table = score.get("ema_by_epoch")
    if not isinstance(table, dict) or set(table) != EPOCHS:
        raise RuntimeError("EMA rows must be exactly epochs 1..24")
    means: dict[int, float] = {}
    for epoch in range(1, 25):
        row = table[str(epoch)]
        if not isinstance(row, dict) or row.get("partial") is not False or row.get("n_windows") != sum(counts.values()):
            raise RuntimeError(f"invalid EMA epoch {epoch}")
        per = row.get("per_session")
        if not isinstance(per, dict) or set(per) != set(counts):
            raise RuntimeError(f"incomplete session surface at epoch {epoch}")
        values = []
        for session, count in counts.items():
            item = per[session]
            if not isinstance(item, dict) or item.get("window_count") != count:
                raise RuntimeError(f"window count mismatch at epoch {epoch}/{session}")
            values.append(finite(item.get("r2"), f"epoch {epoch}/{session}"))
        means[epoch] = sum(values) / len(values)
        if abs(finite(row.get("equal_session_mean"), f"epoch {epoch} mean") - means[epoch]) > 1e-12:
            raise RuntimeError(f"equal-session mean mismatch at epoch {epoch}")
    selected = min(range(1, 25), key=lambda epoch: (-means[epoch], epoch))
    if score.get("selection", {}).get("epoch") != selected:
        raise RuntimeError("selection is not earliest best EMA mean")
    return selected, table[str(selected)], means[selected]


def m1_candidate(root: Path, name: str, dirname: str, schema: str, arm: str | None) -> dict[str, Any]:
    run = root / "results/rift_v1" / dirname
    paths, values = receipts(run); meta, train, score = values.values()
    joint = arm is not None
    expected_cell = "M1-RIFT-R100-D4-JOINT-B3S-CONCAT-V1" if joint else "M1-RIFT-R100-D4-CONCAT-E100-RECENCY-V1"
    train_schema = "m1_rift_joint_train_receipt_v1" if joint else "m1_rift_concat_train_receipt_v1"
    score_schema = "m1_rift_joint_ho_calib_epoch_scan_v1" if joint else "m1_rift_concat_ho_calib_epoch_scan_v1"
    if not (meta.get("schema") == schema and meta.get("status") == "FORMAL" and meta.get("cell") == expected_cell and meta.get("task") == "m1" and meta.get("variant") == "recency" and meta.get("identity_interface") == ("live_b3s_concat" if joint else "concat") and meta.get("identity_e0_dim") == 100 and meta.get("concat_token_width") == 120 and meta.get("seed") == 42 and meta.get("context_bins") == 100 and meta.get("query_pad_bins") == 99 and meta.get("epochs") == 24 and meta.get("updates_per_epoch") == 6665 and meta.get("total_updates") == 159960 and meta.get("ema_decay") == 0.9995 and meta.get("source_train_only_for_gradients") is True and meta.get("official_test_used") is False and m1_source_contract_ok(meta.get("source_contract")) and isinstance(meta.get("source_hashes"), dict) and bool(meta["source_hashes"]) and all(digest_ok(value) for value in meta["source_hashes"].values()) and (not joint or (meta.get("arm") == arm and meta.get("sampler_seed") == 42)) and
            (not joint or train.get("schema") == train_schema) and train.get("status") == "COMPLETED" and train.get("cell") == expected_cell and train.get("epochs") == 24 and train.get("steps") == 159960 and (not joint or (train.get("source_hashes") == meta["source_hashes"] and train.get("source_contract") == meta["source_contract"] and train.get("arm") == arm and train.get("seed") == 42 and train.get("sampler_seed") == 42)) and
            score.get("schema") == score_schema and score.get("status") == "COMPLETED" and score.get("cell") == expected_cell and score.get("official_test_used") is False and m1_ho_contract_ok(score.get("ho_contract"), joint) and (not joint or (score.get("source_hashes") == meta["source_hashes"] and score.get("source_contract") == meta["source_contract"] and score.get("arm") == arm and score.get("seed") == 42 and score.get("sampler_seed") == 42))):
        raise RuntimeError(f"invalid M1 formal receipt triplet: {run}")
    epoch, row, mean = rows(score, M1)
    hashes = score.get("checkpoint_sha256_by_epoch")
    if not isinstance(hashes, dict) or set(hashes) != EPOCHS:
        raise RuntimeError(f"M1 needs 24 checkpoint hashes: {run}")
    for text in sorted(EPOCHS, key=int):
        checkpoint = run / f"epoch_{int(text):03d}.pt"
        if not checkpoint.is_file() or sha(checkpoint) != hashes[text]:
            raise RuntimeError(f"M1 checkpoint binding failed: {checkpoint}")
    if score.get("selection", {}).get("rule") != "earliest maximum equal-session mean EMA":
        raise RuntimeError(f"invalid M1 EMA selection rule: {run}")
    for source, digest in meta["source_hashes"].items():
        source_path = Path(source)
        if not source_path.is_file() or sha(source_path) != digest:
            raise RuntimeError(f"M1 source hash drift: {source_path}")
    if joint:
        b3s = meta.get("b3s")
        if not isinstance(b3s, dict) or not isinstance(b3s.get("sfix_path"), str) or not digest_ok(b3s.get("sfix_sha256")) or not Path(b3s["sfix_path"]).is_file() or sha(Path(b3s["sfix_path"])) != b3s["sfix_sha256"]:
            raise RuntimeError(f"M1 joint B3S source binding failed: {run}")
        source_encoder = {"load": "load the selected checkpoint EMA state into the joint decoder; that EMA state includes its trainable B3S encoder", "identity_interface": "live_b3s_concat", "initial_b3_sfix_path": b3s["sfix_path"], "initial_b3_sfix_sha256": b3s["sfix_sha256"]}
    else:
        b3_path = root.parent / "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_fix/epoch_011.pt"
        b3_sha = "7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a"
        if not b3_path.is_file() or sha(b3_path) != b3_sha:
            raise RuntimeError(f"M1 frozen concat B3S source binding failed: {b3_path}")
        source_encoder = {"load": "load the selected checkpoint EMA state into the concat decoder; construct the frozen B3 Sfix e11 student.id_encoder through btransform_unified_v1.m1_projadd.load_b3_id_encoder", "identity_interface": "concat", "frozen_b3_sfix_path": str(b3_path), "frozen_b3_sfix_sha256": b3_sha}
    return {"name": name, "dataset": "M1", "weight_view": "EMA", "selected_epoch": epoch, "local_score": mean, "selection_rule": score["selection"]["rule"], "checkpoint_path": str(run / f"epoch_{epoch:03d}.pt"), "checkpoint_sha256": hashes[str(epoch)], "all_checkpoint_sha256": hashes, "source_encoder": source_encoder, "source_data_surface": "visible HO3, trial 0, M10; support-overlapping", "receipt_sha256": receipt_sha(paths), "per_session": row["per_session"], "surface": "visible HO3, trial 0, M10; support-overlapping", "official_test_used": False}


def m2_formal(root: Path, arm: str, seed: int) -> dict[str, str]:
    run = root / "results/rift_v1" / f"m2_r50_joint_{arm.lower()}_s{seed}_formal_v1"
    paths, values = receipts(run); meta, train, score = values.values(); expected_arm = "B_ACTIVITY_ONLY" if arm == "B" else "D_JOINT"
    if not (meta.get("schema") == "m2_rift_joint_train_v2" and meta.get("status") == "FORMAL" and meta.get("cell") == "M2-RIFT-R50-D4-JOINT-FILM-M33-V1" and meta.get("arm") == expected_arm and meta.get("seed") == seed and meta.get("sampler_seed") == 42 and meta.get("epochs") == 24 and meta.get("official_test_used") is False and isinstance(meta.get("source_hashes"), dict) and isinstance(meta.get("cache_hashes"), dict) and
            train.get("schema") == "m2_rift_joint_train_receipt_v2" and train.get("status") == "COMPLETED" and train.get("cell") == meta["cell"] and train.get("arm") == expected_arm and train.get("seed") == seed and train.get("sampler_seed") == 42 and train.get("epochs") == 24 and train.get("global_step") == 75960 and train.get("source_hashes") == meta["source_hashes"] and train.get("cache_hashes") == meta["cache_hashes"] and
            score.get("schema") == "m2_rift_joint_ext4_epoch_scan_v1" and score.get("status") == "COMPLETED" and score.get("cell") == meta["cell"] and score.get("arm") == expected_arm and score.get("seed") == seed and score.get("sampler_seed") == 42 and score.get("official_test_used") is False and score.get("source_hashes") == meta["source_hashes"] and score.get("cache_hashes") == meta["cache_hashes"]):
        raise RuntimeError(f"invalid M2 formal EXT4 triplet: {run}")
    rows(score, M2_EXT4)
    if score.get("selection", {}).get("rule") != "earliest best equal_session_mean":
        raise RuntimeError(f"invalid M2 formal EXT4 selection rule: {run}")
    for text in sorted(EPOCHS, key=int):
        checkpoint = run / f"epoch_{int(text):03d}.pt"
        if not checkpoint.is_file() or score["ema_by_epoch"][text].get("checkpoint_sha256") != sha(checkpoint):
            raise RuntimeError(f"M2 formal checkpoint binding failed: {checkpoint}")
    for source, digest in meta["source_hashes"].items():
        source_path = Path(source)
        if not source_path.is_file() or sha(source_path) != digest:
            raise RuntimeError(f"M2 formal source hash drift: {source_path}")
    return {key: sha(paths[key.replace("_sha256", ".json")]) for key in PAIR_KEYS}


def m2_candidate(root: Path, name: str, formal_name: str, pick_name: str, *, arm: str | None, seed: int) -> dict[str, Any]:
    formal = root / "results/rift_v1" / formal_name; pick = root / "results/rift_v1" / pick_name
    formal_paths, formal_values = receipts(formal); meta, train, _ = formal_values.values()
    manifest_path, score_path, audit_path = pick / "manifest.json", pick / "score_receipt.json", pick / "validation_audit.json"
    manifest, score, audit = load(manifest_path), load(score_path), load(audit_path)
    concat = arm is None
    expected_schema = "m2_rift_concat_train_v1" if concat else "m2_rift_joint_train_v2"
    expected_train_schema = "m2_rift_concat_train_receipt_v1" if concat else "m2_rift_joint_train_receipt_v2"
    expected_cell = "M2-RIFT-R50-D4-CONCAT-E50-RECENCY-V1" if concat else "M2-RIFT-R50-D4-JOINT-FILM-M33-V1"
    expected_audit_schema = "m2_concat_ext6_full_curve_validation_v1" if concat else f"m2_joint_d_s{seed}_ext6_full_curve_validation_v1"
    expected_arm = None if concat else "D_JOINT"
    picker_schema = "m2_rift_ext6_epoch_pick_v2" if seed == 42 else "m2_rift_ext6_epoch_pick_multiseed_v1"
    if not (meta.get("schema") == expected_schema and meta.get("cell") == expected_cell and meta.get("seed") == seed and meta.get("status") == "FORMAL" and meta.get("official_test_used") is False and isinstance(meta.get("source_hashes"), dict) and bool(meta["source_hashes"]) and all(digest_ok(digest) for digest in meta["source_hashes"].values()) and
            (not concat or (meta.get("task") == "m2" and meta.get("variant") == "recency" and meta.get("identity_interface") == "concat" and meta.get("identity_e0_dim") == 50 and meta.get("context_bins") == 50 and meta.get("depth") == 4 and meta.get("width") == 256 and meta.get("attention_backend") == "local" and meta.get("epochs") == 24 and meta.get("updates_per_epoch") == 3165 and meta.get("ema_decay") == 0.9995 and meta.get("source_train_only_for_gradients") is True and isinstance(meta.get("frozen_cache_hashes"), dict) and bool(meta["frozen_cache_hashes"]))) and
            (concat or (meta.get("arm") == expected_arm and meta.get("sampler_seed") == 42)) and
            train.get("schema") == expected_train_schema and train.get("status") == "COMPLETED" and train.get("cell") == meta["cell"] and train.get("source_hashes") == meta.get("source_hashes") and
            (concat and train.get("epochs") == 24 and train.get("global_step") == 75960 and train.get("frozen_cache_hashes") == meta.get("frozen_cache_hashes") or not concat and train.get("arm") == expected_arm and train.get("seed") == seed and train.get("sampler_seed") == 42 and train.get("cache_hashes") == meta.get("cache_hashes")) and
            manifest.get("schema") == picker_schema + "_manifest" and manifest.get("run") == str(formal) and manifest.get("run_schema") == expected_schema and manifest.get("cell") == expected_cell and manifest.get("seed") == seed and manifest.get("view") == "EMA" and manifest.get("epochs") == list(range(1, 25)) and manifest.get("official_test_used") is False and manifest.get("evalai_opened") is False and
            manifest.get("run_meta_sha256") == sha(formal_paths["run_meta.json"]) and manifest.get("train_receipt_sha256") == sha(formal_paths["train_receipt.json"]) and
            score.get("schema") == picker_schema + "_selection" and score.get("status") == "COMPLETED" and score.get("view") == "EMA" and score.get("official_test_used") is False and score.get("evalai_opened") is False and score.get("manifest_sha256") == canonical_digest(manifest) and
            audit.get("schema") == expected_audit_schema and audit.get("status") == "PASSED" and audit.get("failures") == [] and isinstance(audit.get("checks"), dict) and bool(audit["checks"]) and all(isinstance(check, dict) and check.get("passed") is True for check in audit["checks"].values()) and isinstance(audit.get("scope"), dict) and audit["scope"].get("read_only") is True and audit["scope"].get("evalai_opened") is False and audit["scope"].get("official_test_opened") is False):
        raise RuntimeError(f"invalid M2 ext6 receipt binding: {pick}")
    assets = manifest.get("query_assets", {}).get("sessions") if isinstance(manifest.get("query_assets"), dict) else None
    if not isinstance(assets, dict) or set(assets) != M2:
        raise RuntimeError(f"M2 sealed manifest lacks six query assets: {pick}")
    counts = {key: value.get("window_count") if isinstance(value, dict) else None for key, value in assets.items()}
    if any(not isinstance(value, int) or value <= 0 for value in counts.values()) or sum(counts.values()) != 15403:
        raise RuntimeError(f"M2 sealed manifest total is not 15403: {pick}")
    for source, digest in meta["source_hashes"].items():
        source_path = Path(source)
        if not source_path.is_file() or sha(source_path) != digest:
            raise RuntimeError(f"M2 formal source hash drift: {source_path}")
    manifest_sources = manifest.get("source_sha256")
    if not isinstance(manifest_sources, dict) or not manifest_sources:
        raise RuntimeError(f"M2 manifest source hashes absent: {pick}")
    for source, digest in manifest_sources.items():
        source_path = Path(source)
        if not digest_ok(digest) or not source_path.is_file() or sha(source_path) != digest:
            raise RuntimeError(f"M2 manifest source hash drift: {source_path}")
    epoch, row, mean = rows(score, counts); checkpoints = manifest.get("checkpoint_bytes")
    if not isinstance(checkpoints, dict) or set(checkpoints) != EPOCHS:
        raise RuntimeError(f"M2 manifest needs 24 checkpoint hashes: {pick}")
    hashes: dict[str, str] = {}
    for text in sorted(EPOCHS, key=int):
        item = checkpoints[text]
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise RuntimeError(f"invalid M2 checkpoint row {text}: {pick}")
        checkpoint = Path(item["path"])
        if not checkpoint.is_file() or sha(checkpoint) != item["sha256"] or score["ema_by_epoch"][text].get("checkpoint_sha256") != item["sha256"] or score["ema_by_epoch"][text].get("view") != "EMA":
            raise RuntimeError(f"M2 checkpoint binding failed: {checkpoint}")
        hashes[text] = item["sha256"]
    if score.get("selected") != row or score.get("selection", {}).get("epoch") != epoch or score.get("selection", {}).get("rule") != "earliest maximum finite unweighted equal_session_mean" or row.get("checkpoint_sha256") != hashes[str(epoch)]:
        raise RuntimeError(f"M2 selected EMA/checkpoint binding failed: {pick}")
    ema = score.get("selected_ema_state")
    if not isinstance(ema, dict) or ema.get("raw_state_serialized") is not False or not isinstance(ema.get("path"), str) or not isinstance(ema.get("sha256"), str):
        raise RuntimeError(f"M2 selected EMA-only state invalid: {pick}")
    package = Path(ema["path"])
    if Path(ema["path"]).resolve() != (pick / "selected_ema.pt").resolve() or not package.is_file() or sha(package) != ema["sha256"]:
        raise RuntimeError(f"M2 selected EMA package hash mismatch: {package}")
    artifacts = audit.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError(f"M2 validation artifact hashes absent: {pick}")
    required_artifacts = {"manifest.json", "score_progress.json", "score_receipt.json", "selected_ema.pt", "selected_ema_receipt.json"}
    if set(artifacts) != required_artifacts:
        raise RuntimeError(f"M2 validation artifact set mismatch: {pick}")
    for filename, expected in artifacts.items():
        artifact = pick / filename
        if not isinstance(expected, str) or not artifact.is_file() or sha(artifact) != expected:
            raise RuntimeError(f"M2 validation artifact mismatch: {artifact}")
    selected_receipt = load(pick / "selected_ema_receipt.json")
    if selected_receipt != score:
        raise RuntimeError(f"M2 selected EMA receipt differs from selection receipt: {pick}")
    if concat:
        source_encoder = {"load": "load selected_ema.pt into the concat decoder after building its frozen session identity banks", "identity_interface": "concat", "ema_package_path": str(package), "ema_package_sha256": ema["sha256"]}
    else:
        film = next(((source, digest) for source, digest in meta["source_hashes"].items() if source.endswith("film_states.pt")), None)
        if not isinstance(meta.get("identity_provider"), str) or not meta["identity_provider"] or film is None:
            raise RuntimeError(f"M2 joint source encoder provenance absent: {formal}")
        source_encoder = {"load": "load selected_ema.pt into the joint decoder; the EMA state includes the jointly trained FiLM identity encoder", "identity_provider": meta["identity_provider"], "initial_film_state_path": film[0], "initial_film_state_sha256": film[1], "ema_package_path": str(package), "ema_package_sha256": ema["sha256"]}
    return {"name": name, "dataset": "M2", "weight_view": "EMA", "selected_epoch": epoch, "local_score": mean, "selection_rule": score["selection"]["rule"], "checkpoint_path": checkpoints[str(epoch)]["path"], "checkpoint_sha256": hashes[str(epoch)], "all_checkpoint_sha256": hashes, "ema_only_package_path": str(package), "ema_only_package_sha256": ema["sha256"], "source_encoder": source_encoder, "source_data_surface": "ext6 six sessions, query trial 0, 15403 windows", "receipt_sha256": {"run_meta.json": sha(formal_paths["run_meta.json"]), "train_receipt.json": sha(formal_paths["train_receipt.json"]), "manifest.json": sha(manifest_path), "score_receipt.json": sha(score_path), "validation_audit.json": sha(audit_path)}, "per_session": row["per_session"], "surface": "ext6 six sessions, query trial 0, 15403 windows", "official_test_used": False}


def check_m1_summary(path: Path | None, expected: dict[str, dict[str, Any]]) -> None:
    if path is None:
        raise RuntimeError("missing --m1-pair-summary")
    summary = load(path)
    scope = summary.get("scope")
    if not (summary.get("schema") == "m1_joint_pair_summary_v1" and summary.get("status") == "COMPLETED" and isinstance(scope, dict) and scope.get("seed") == 42 and scope.get("epochs") == 24 and scope.get("updates") == 159960 and scope.get("heldout_sessions") == list(M1) and scope.get("heldout_windows") == 3881 and scope.get("official_test_used") is False and scope.get("selection") == "earliest maximum equal-session mean EMA across all 24 exact checkpoint files"):
        raise RuntimeError("M1 paired summary is not completed")
    arms = summary.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"B", "D"}:
        raise RuntimeError("M1 paired summary arms are invalid")
    for arm in ("B", "D"):
        candidate = expected[arm]
        if arms[arm].get("receipt_sha256") != candidate["receipt_sha256"]:
            raise RuntimeError(f"M1 paired summary {arm} receipt binding mismatch")
        selected_epoch = candidate["selected_epoch"]
        if arms[arm].get("selected_epoch") != selected_epoch or arms[arm].get("checkpoint_sha256_by_epoch", {}).get(str(selected_epoch)) != candidate["checkpoint_sha256"]:
            raise RuntimeError(f"M1 paired summary {arm} selected checkpoint binding mismatch")
    if summary.get("frozen_concat_receipt_sha256") != expected["concat"]["receipt_sha256"]:
        raise RuntimeError("M1 paired summary frozen concat receipt binding mismatch")


def check_m2_summary(path: Path | None, expected: dict[tuple[str, int], dict[str, str]]) -> None:
    if path is None:
        raise RuntimeError("missing --m2-pair-summary")
    summary = load(path)
    if not (summary.get("schema") == "m2_joint_paired_seed_effects_v1" and summary.get("status") == "COMPLETED" and summary.get("surface") == "ext4 mechanism control only; never ext6" and summary.get("arms") == {"B": "B_ACTIVITY_ONLY", "D": "D_JOINT"} and summary.get("seeds") == [42, 43, 44] and summary.get("independence_unit") == "seed-paired B versus D; sessions are within-seed repeated measurements" and summary.get("official_test_used") is False and summary.get("evalai_opened") is False and isinstance(summary.get("three_seed_summary"), dict)):
        raise RuntimeError("M2 paired EXT4 summary is not completed")
    pairs = summary.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise RuntimeError("M2 paired summary must have three seed pairs")
    seen: set[int] = set()
    for pair in pairs:
        if not isinstance(pair, dict) or pair.get("seed") not in (42, 43, 44) or pair["seed"] in seen:
            raise RuntimeError("invalid M2 paired summary seed")
        seen.add(pair["seed"])
        for arm in ("B", "D"):
            row = pair.get(arm)
            actual = {key: row.get(key) for key in PAIR_KEYS} if isinstance(row, dict) else None
            if actual != expected[(arm, pair["seed"])]:
                raise RuntimeError(f"M2 paired summary receipt binding mismatch: {arm}{pair['seed']}")


def controls(root: Path) -> dict[str, dict[str, str]]:
    answer: dict[str, dict[str, str]] = {}
    semantics = {
        "C": ("C_CARRIER_ONLY", "slots", "attention"),
        "shuffle": ("D_SHUFFLE", "slots", "attention"),
        "mean": ("D_JOINT", "mean", "attention"),
        "nonattn": ("D_JOINT", "slots", "nonattention"),
    }
    for label, dirname in CONTROLS.items():
        paths, values = receipts(root / "results/rift_v1" / dirname); meta, train, score = values.values()
        arm, aggregation, temporal = semantics[label]
        if not (meta.get("schema") == "m2_rift_mechanism_train_v1" and meta.get("status") == "FORMAL" and meta.get("cell") == "M2-RIFT-R50-MECHANISM-V1" and meta.get("arm") == arm and meta.get("aggregation") == aggregation and meta.get("temporal") == temporal and meta.get("seed") == 42 and meta.get("sampler_seed") == 42 and meta.get("epochs") == 24 and meta.get("context_bins") == 50 and meta.get("source_train_only_for_gradients") is True and meta.get("official_test_used") is False and isinstance(meta.get("source_hashes"), dict) and bool(meta["source_hashes"]) and isinstance(meta.get("cache_hashes"), dict) and bool(meta["cache_hashes"]) and meta.get("optimizer", {}).get("total_updates") == 75960 and
                train.get("schema") == "m2_rift_mechanism_train_receipt_v1" and train.get("status") == "COMPLETED" and all(train.get(key) == meta.get(key) for key in ("cell", "arm", "aggregation", "temporal", "seed", "sampler_seed", "epochs", "source_hashes", "cache_hashes")) and train.get("global_step") == 75960 and
                score.get("schema") == "m2_rift_mechanism_ext4_epoch_scan_v1" and score.get("status") == "COMPLETED" and score.get("official_test_used") is False and all(score.get(key) == meta.get(key) for key in ("cell", "arm", "aggregation", "temporal", "seed", "sampler_seed", "source_hashes", "cache_hashes"))):
            raise RuntimeError(f"invalid formal M2 control receipt: {dirname}")
        rows(score, M2_EXT4)
        if score.get("selection", {}).get("rule") != "earliest best equal_session_mean":
            raise RuntimeError(f"invalid M2 control selection rule: {dirname}")
        for text in sorted(EPOCHS, key=int):
            checkpoint = root / "results/rift_v1" / dirname / f"epoch_{int(text):03d}.pt"
            if not checkpoint.is_file() or score["ema_by_epoch"][text].get("checkpoint_sha256") != sha(checkpoint):
                raise RuntimeError(f"M2 control checkpoint binding failed: {checkpoint}")
        for source, digest in meta["source_hashes"].items():
            source_path = Path(source)
            if not source_path.is_file() or sha(source_path) != digest:
                raise RuntimeError(f"M2 control source hash drift: {source_path}")
        answer[label] = receipt_sha(paths)
    return answer


def h1(root: Path) -> dict[str, Any]:
    directory = root / "results/rift_v1/h1_r300_official_receipt_20260907"
    get_path, result_path, receipt_path = directory / "submission_get.json", directory / "official_result.json", directory / "retrieval_receipt.json"
    got, result, receipt = load(get_path), load(result_path), load(receipt_path)
    expected_sha = "52e88d57b25c47f1648b73d4b8dad431a01fe49c4ea017153652de0266cc54d8"
    split = result.get("test_split_h1")
    required_metrics = ("Normalized Latency", "Held Out R2 Mean", "Held Out R2 Std.", "Held In R2 Mean", "Held In R2 Std.")
    response = got.get("response")
    if not (got.get("submission_id") == 582073 and isinstance(response, dict) and response.get("id") == 582073 and response.get("status") == "finished" and receipt.get("submission_id") == 582073 and receipt.get("status") == "finished" and receipt.get("sha256") == expected_sha == sha(result_path) and receipt.get("source_url") == response.get("submission_result_file") and isinstance(split, dict) and set(split) == set(required_metrics) and all(math.isfinite(float(split[key])) for key in required_metrics)):
        raise RuntimeError("H1 official receipt is not bound to submission 582073")
    return {"status": "REFERENCE_FROZEN", "submission_id": 582073, "official_result_path": str(result_path), "official_result_sha256": sha(result_path), "submission_get_sha256": sha(get_path), "retrieval_receipt_sha256": sha(receipt_path), "recommendation": "retain frozen 582073; do not recommend resubmission"}


def as_markdown(out: dict[str, Any]) -> str:
    lines = ["# FALCON local candidate inventory", "", f"Status: `{out['status']}`", "", "## H1", "", f"Frozen official submission: `{out['h1']['submission_id']}`.", "", "## Candidates", "", "| Dataset | Candidate | EMA epoch | Local equal-session mean |", "| --- | --- | ---: | ---: |"]
    lines += [f"| {row['dataset']} | {row['name']} | {row['selected_epoch']} | {row['local_score']:.10f} |" for row in out["candidates"]]
    if out["status"] == "COMPLETED":
        lines += ["", "## Final ranking and recommendation", ""]
        for dataset in ("M1", "M2"):
            ranked = out["ranking"][dataset]
            lines.append(f"### {dataset}")
            lines.append("")
            for position, row in enumerate(ranked, start=1):
                lines.append(f"{position}. `{row['name']}` — EMA epoch `{row['selected_epoch']}`, score `{row['local_score']:.10f}`; checkpoint `{row['checkpoint_path']}` (SHA-256 `{row['checkpoint_sha256']}`); source encoder `{row['source_encoder']}`; data surface `{row['source_data_surface']}`; selection rule `{row['selection_rule']}`.")
            recommendation = out["recommendations"][0 if dataset == "M1" else 1]["candidate"]
            lines += ["", f"Recommended {dataset} candidate: `{recommendation['name']}` using its EMA epoch `{recommendation['selected_epoch']}` checkpoint `{recommendation['checkpoint_path']}` (SHA-256 `{recommendation['checkpoint_sha256']}`)."]
    if out["status"] != "COMPLETED":
        lines += ["", "## Pending evidence", ""] + [f"- {reason}" for reason in out["pending_reasons"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--m1-pair-summary", type=Path); parser.add_argument("--m2-pair-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.output.exists() or (args.markdown_output is not None and args.markdown_output.exists()):
        raise RuntimeError("refusing to overwrite an existing output")
    candidates: list[dict[str, Any]] = []; problems: list[str] = []; m1_candidates: dict[str, dict[str, Any]] = {}
    for name, dirname, schema, arm, key in (("frozen_concat42", "m1_r100_concat_s42_formal_v1", "m1_rift_concat_train_v1", None, "concat"), ("jointB42", "m1_r100_joint_b_s42_formal_v1", "m1_rift_joint_train_v1", "B_ACTIVITY_ONLY", "B"), ("jointD42", "m1_r100_joint_d_s42_formal_v1", "m1_rift_joint_train_v1", "D_JOINT", "D")):
        try:
            candidate = m1_candidate(root, name, dirname, schema, arm); candidates.append(candidate); m1_candidates[key] = candidate
        except RuntimeError as exc: problems.append(str(exc))
    formal: dict[tuple[str, int], dict[str, str]] = {}
    for seed in (42, 43, 44):
        for arm in ("B", "D"):
            try: formal[(arm, seed)] = m2_formal(root, arm, seed)
            except RuntimeError as exc: problems.append(str(exc))
    for name, train, pick, arm, seed in (("concat42", "m2_r50_concat_s42_formal_v1", "m2_r50_concat_s42_ext6_pick_v1", None, 42),
                                         ("jointD42", "m2_r50_joint_d_s42_formal_v1", "m2_r50_joint_d_s42_ext6_pick_v1", "D", 42),
                                         ("jointD43", "m2_r50_joint_d_s43_formal_v1", "m2_r50_joint_d_s43_ext6_pick_v1", "D", 43),
                                         ("jointD44", "m2_r50_joint_d_s44_formal_v1", "m2_r50_joint_d_s44_ext6_pick_v1", "D", 44)):
        try: candidates.append(m2_candidate(root, name, train, pick, arm=arm, seed=seed))
        except RuntimeError as exc: problems.append(str(exc))
    bound_controls: dict[str, dict[str, str]] = {}
    try: bound_controls = controls(root)
    except RuntimeError as exc: problems.append(str(exc))
    try:
        if set(m1_candidates) == {"concat", "B", "D"}: check_m1_summary(args.m1_pair_summary, m1_candidates)
        else: problems.append("M1 candidate receipt triplets are incomplete")
    except RuntimeError as exc: problems.append(str(exc))
    try:
        if len(formal) == 6: check_m2_summary(args.m2_pair_summary, formal)
        else: problems.append("M2 formal EXT4 receipt triplets are incomplete")
    except RuntimeError as exc: problems.append(str(exc))
    base = {"schema": "falcon_submission_candidate_recommendations_v2", "policy": "Read-only local development-selection inventory; never submits, packages, scores, or accesses EvalAI.", "h1": h1(root), "candidates": candidates, "m2_control_receipt_sha256": bound_controls}
    if problems:
        out = {**base, "status": "PENDING", "ranking": [], "recommendations": [], "pending_reasons": sorted(set(problems)), "paper_owned_experiments_complete": False}
    else:
        fixed_order = {"M1": {"frozen_concat42": 0, "jointB42": 1, "jointD42": 2}, "M2": {"concat42": 0, "jointD42": 1, "jointD43": 2, "jointD44": 3}}
        ranking = {data: sorted((row for row in candidates if row["dataset"] == data), key=lambda row: (-row["local_score"], fixed_order[data][row["name"]])) for data in ("M1", "M2")}
        out = {**base, "status": "COMPLETED", "ranking": ranking, "recommendations": [{"dataset": data, "candidate": rows[0], "scope": "development-selection only; no official-superiority or mechanism-effect claim"} for data, rows in ranking.items()], "paper_owned_experiments_complete": True}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True); args.markdown_output.write_text(as_markdown(out), encoding="utf-8")
    print(json.dumps({"status": out["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
