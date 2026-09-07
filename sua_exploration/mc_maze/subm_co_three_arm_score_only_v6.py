"""Cryptographically grant-bound V6 control plane for external sub-M scoring.

Production remains blocked.  This module never imports Torch and its blocked
runner never opens an NWB, normalizer, or checkpoint.  A future run can reach
the ledger only through :func:`verify_authorization_and_claim`, which verifies
an Ed25519 authorization, a <=15 minute validity window, all frozen digests,
reviewed terminal closures, runtime/root bindings, and an atomic nonce claim.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import sys
from typing import Any, Mapping, Sequence
import zipfile


ARMS = ("shared_t4", "shared_zero4", "shared_ts4")
VIEWS = ("sua", "pseudo_mua")
SEEDS = (42, 43, 44)
COMPARISONS = (("t4_minus_zero4", "shared_zero4"), ("t4_minus_ts4", "shared_ts4"))
N = 15
CELL_COUNT = 270
QUERY_WINDOWS_PER_VIEW = 708_795
OUTPUT_DIM = 2
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 68_820_260_805
MAX_AUTH_SECONDS = 900
R2_ATOL = 1e-12

CONTRACT_SCHEMA = "dandi_000688_subm_three_arm_score_contract_v6"
AUTH_SCHEMA = "dandi_000688_subm_three_arm_external_authorization_v6"
AUTH_STATUS = "AUTHORIZED_FOR_ONE_EXTERNAL_SUBM_SCORE_V6"
AUTH_ACTION = "score_publish_resume_aggregate_exact_270_v6"
ENVELOPE_SCHEMA = "ed25519_detached_authorization_envelope_v6"
GRANT_SCHEMA = "dandi_000688_subm_three_arm_verified_grant_v6"
CELL_SCHEMA = "dandi_000688_subm_three_arm_cell_result_v6"
AGGREGATE_SCHEMA = "dandi_000688_subm_three_arm_aggregate_v6"

CPU_POLICY = {
    "device": "cpu", "cuda_visible_devices": "", "torch_num_threads": 1,
    "torch_num_interop_threads": 1, "deterministic_algorithms": True,
    "tf32": False, "autocast": False, "normalizer_fitting": False,
    "optimizer_or_backward": False, "target_updates": False,
}
SCORE_PROTOCOL = {
    "support_trials": 50,
    "activity_identity_trials": 30,
    "query": "all valid 50-bin windows strictly after rewarded trial 50",
    "target": "last behavior bin of each query window",
    "output_dim": OUTPUT_DIM,
    "behavior_scaling_factor_applied_once": 5.0,
    "r2": "variance_weighted_internal_from_safe_npz",
    "caller_supplied_r2": "FORBIDDEN",
    "session_drop_after_scoring": "FORBIDDEN",
}
CLAIM_SEPARATION = {
    "t4_minus_zero4": "descriptor-present versus direct neutral coordinate",
    "t4_minus_ts4": "attachment/content permutation mechanism",
    "cross_claim_substitution": "FORBIDDEN",
    "overall_three_arm_claim_requires_both_comparisons_all_views": True,
}
COMPARISON_GATES = {
    name: {
        "grand_paired_mean_minimum_r2": 0.03,
        "all_three_seed_means_strictly_positive": True,
        "session_cross_seed_positive_fraction_minimum": 0.75,
        "hierarchical_session_seed_bootstrap_lower_95_strictly_positive": True,
        "shared_t4_absolute_grand_and_seed_means_strictly_positive": True,
        "views_evaluated_separately": list(VIEWS),
    }
    for name, _ in COMPARISONS
}
BOOTSTRAP_POLICY = {
    "replicates": BOOTSTRAP_REPLICATES,
    "rng": f"numpy.random.Generator(numpy.random.PCG64({BOOTSTRAP_SEED}))",
    "draw_order": [f"{comparison}/{view}" for comparison, _ in COMPARISONS for view in VIEWS],
    "session_resample_count": N, "seed_resample_count_within_each_session": 3,
    "quantiles": [0.025, 0.975], "quantile_method": "linear",
    "cross_comparison_pooling": "FORBIDDEN", "cross_view_rescue": "FORBIDDEN",
}

COHORT_KEYS = {"asset_id", "session_id", "frozen_path", "nwb_sha256", "nwb_bytes"}
SLOT_KEYS = {"arm", "seed", "epoch", "path", "sha256", "bytes", "mode", "status", "closure"}
CLOSURE_PIN_KEYS = {"kind", "path", "sha256", "bytes", "mode", "authority_receipt"}
FILE_PIN_KEYS = {"path", "sha256", "bytes", "mode"}
POLICY_KEYS = {
    "schema", "frozen_predecessor_sha256", "cohort", "query_counts",
    "reviewed_checkpoint_slots", "reviewed_checkpoint_slots_sha256",
    "verified_closure_bundle_sha256", "source_snapshot_sha256",
    "parity_bundle_sha256", "runtime_identity", "cpu_policy",
    "repository_root", "closure_root", "output_parent", "external_parent",
    "claim_root", "public_key",
}
AUTH_KEYS = {
    "schema", "status", "permitted_action", "issued_at", "expires_at", "nonce",
    "output_root", "external_nwb_root", "cell_count", "bindings",
}
BINDING_KEYS = {
    "frozen_predecessor_sha256", "contract_sha256", "cohort_sha256",
    "query_map_sha256", "reviewed_checkpoint_slots_sha256",
    "verified_closure_bundle_sha256", "source_snapshot_sha256",
    "parity_bundle_sha256", "runtime_identity", "cpu_policy",
}
CELL_KEYS = {
    "schema", "status", "verified_grant_sha256", "cell", "r2",
    "query_window_count", "prediction_target_artifact", "contract_sha256",
}
CELL_ID_KEYS = {"session_id", "asset_id", "view", "seed", "arm"}
ARTIFACT_PIN_KEYS = {
    "path", "sha256", "bytes", "mode", "verified_grant_sha256", "arrays",
}
AGGREGATE_KEYS = {
    "schema", "status", "verified_grant_sha256", "contract_sha256",
    "statistics_source", "verified_cell_count", "bootstrap_policy",
    "comparisons", "overall_three_arm_claim_pass",
}


class V6Error(RuntimeError):
    pass


class V6BlockedError(V6Error):
    pass


class V6AuthorizationError(PermissionError):
    pass


class V6LedgerError(V6Error):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise V6Error(message)


def auth_require(condition: bool, message: str) -> None:
    if not condition:
        raise V6AuthorizationError(message)


def ledger_require(condition: bool, message: str) -> None:
    if not condition:
        raise V6LedgerError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def assert_under(path: Path, root: Path, label: str) -> tuple[Path, Path]:
    candidate, base = lexical_absolute(path), lexical_absolute(root)
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise V6LedgerError(f"{label} escapes fixed root") from exc
    return candidate, base


def assert_existing_chain_no_symlink(path: Path, root: Path) -> None:
    candidate, base = assert_under(path, root, "path")
    chain = [base]
    current = base
    for part in candidate.relative_to(base).parts:
        current /= part; chain.append(current)
    for component in chain:
        try:
            mode = component.lstat().st_mode
        except OSError as exc:
            raise V6LedgerError(f"missing path component: {component}") from exc
        if stat.S_ISLNK(mode):
            raise V6LedgerError(f"symlink path component forbidden: {component}")


def assert_absolute_chain_no_symlink(path: Path) -> None:
    candidate = lexical_absolute(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise V6LedgerError(f"missing absolute path component: {current}") from exc
        if stat.S_ISLNK(mode):
            raise V6LedgerError(f"symlink absolute path component forbidden: {current}")


@dataclass(frozen=True)
class FDRead:
    raw: bytes
    sha256: str
    bytes: int
    mode: str


def fd_read_regular(
    path: Path, *, root: Path, expected_mode: str | None = None,
    expected_sha256: str | None = None, expected_bytes: int | None = None,
    max_bytes: int = 512 * 1024 * 1024,
) -> FDRead:
    """O_NOFOLLOW/fstat reader with before/after identity checks."""
    candidate, base = assert_under(path, root, "file")
    assert_absolute_chain_no_symlink(base)
    assert_existing_chain_no_symlink(candidate.parent, base)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise V6LedgerError(f"cannot safely open regular file: {candidate}") from exc
    try:
        before = os.fstat(descriptor)
        ledger_require(stat.S_ISREG(before.st_mode), "opened file is not regular")
        ledger_require(before.st_size <= max_bytes, "file exceeds fixed safety limit")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    ledger_require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "file changed during fd-based read",
    )
    leaf = candidate.lstat()
    ledger_require(not stat.S_ISLNK(leaf.st_mode) and (leaf.st_dev, leaf.st_ino) == (after.st_dev, after.st_ino), "path/file identity changed after read")
    raw = b"".join(chunks)
    mode = f"0{stat.S_IMODE(after.st_mode):03o}"
    digest = hashlib.sha256(raw).hexdigest()
    if expected_mode is not None:
        ledger_require(mode == expected_mode, "file mode drift")
    if expected_bytes is not None:
        ledger_require(len(raw) == expected_bytes, "file byte-size drift")
    if expected_sha256 is not None:
        ledger_require(digest == expected_sha256, "file SHA-256 drift")
    return FDRead(raw=raw, sha256=digest, bytes=len(raw), mode=mode)


def read_canonical_json_fd(
    path: Path, *, root: Path, pin: Mapping[str, Any] | None = None,
    expected_mode: str | None = None, max_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if pin is not None:
        ledger_require(set(pin) == FILE_PIN_KEYS, "file pin schema drift")
        kwargs = {"expected_mode": pin["mode"], "expected_sha256": pin["sha256"], "expected_bytes": pin["bytes"]}
    elif expected_mode is not None:
        kwargs = {"expected_mode": expected_mode}
    observed = fd_read_regular(path, root=root, max_bytes=max_bytes, **kwargs)
    try:
        value = json.loads(observed.raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V6LedgerError("malformed JSON artifact") from exc
    ledger_require(isinstance(value, dict) and observed.raw == canonical_bytes(value), "JSON is not canonical exact object")
    return value


def checkpoint_path(arm: str, seed: int) -> str:
    if arm in {"shared_t4", "shared_ts4"}:
        return f"sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_{arm}_s{seed}/epoch_ckpts/epoch_011.ckpt"
    return f"sua_exploration/checkpoints/t4_paired_view_c1_shared_zero4_source_prelaunch_v4_20260805_{arm}_s{seed}/epoch_ckpts/epoch_011.ckpt"


_KNOWN_CHECKPOINTS = {
    ("shared_t4", 42): ("ab9df840a07d7aeb6cc417bb684f1f5e0265d50f98168400ac915647cdfd7b9f", 64_768_898),
    ("shared_t4", 43): ("05c05b3ab82a2fba43c55aca523248982a954faf5f0363a0235a29d64e57ab22", 64_768_898),
    ("shared_t4", 44): ("a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6", 64_769_167),
    ("shared_ts4", 42): ("a21da5a72a991bd2665af50572a4132998ac79d7f801879048553efcdc8281b2", 64_768_898),
    ("shared_ts4", 43): ("c8dd22dfadb2bc11555fc21abe464316886d221e2dbcd71bf20a6bffe9cb158e", 64_768_898),
    ("shared_ts4", 44): ("a2d877ac81a4e553e5221c54e465db26eba8592888b8cb5339e9dfc4acd66ced", 64_769_167),
}
_KNOWN_EVIDENCE = {
    ("shared_t4", 42): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s42/shared_t4_s42_sua.json", "689a692af59010e4799fe26aa900f9fba8a8cd78238b14e4eae36fce5f16d1aa", 127_124, "0664"),
    ("shared_t4", 43): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s43/shared_t4_s43_sua.json", "e1e9b5453d67535268ae099226cd61320c869e95f02355e02bd273c2261f295d", 127_125, "0664"),
    ("shared_t4", 44): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s44/shared_t4_s44_sua.json", "add3206d4ac5ad28dcb81132eb156d024f99d73fdbc4e84f0b2a7577393c1712", 127_367, "0664"),
    ("shared_ts4", 42): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_ts4_s42/shared_ts4_s42_sua.json", "f687fb5842438586c2bbef970306fe6410773097188f13e84e11aee9f31a5521", 127_169, "0664"),
    ("shared_ts4", 43): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_ts4_s43/shared_ts4_s43_sua.json", "b214fa4bbf596fba7a971a6200b0851f4f6b0a925dc615de0e071bee3a6a4a36", 127_183, "0664"),
    ("shared_ts4", 44): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_ts4_s44/shared_ts4_s44_sua.json", "b65c10a75e066437e8962d181c442f6a3dbd793bdce0a1cec9a13807aabba969", 127_408, "0664"),
}


def slot_binding(row: Mapping[str, Any]) -> str:
    return canonical_sha256({key: row[key] for key in ("arm", "seed", "epoch", "path", "sha256", "bytes", "mode")})


def blocked_checkpoint_slots() -> list[dict[str, Any]]:
    result = []
    for arm in ARMS:
        for seed in SEEDS:
            path = checkpoint_path(arm, seed)
            if (arm, seed) not in _KNOWN_CHECKPOINTS:
                result.append({"arm": arm, "seed": seed, "epoch": 11, "path": path, "sha256": None, "bytes": None, "mode": None, "status": "MISSING_SEALED_TERMINAL_CLOSURE", "closure": None})
                continue
            digest, size = _KNOWN_CHECKPOINTS[(arm, seed)]
            evidence_path, evidence_hash, evidence_bytes, evidence_mode = _KNOWN_EVIDENCE[(arm, seed)]
            result.append({
                "arm": arm, "seed": seed, "epoch": 11, "path": path,
                "sha256": digest, "bytes": size, "mode": "0444",
                "status": "REVIEWED_EXISTING_TERMINAL_EVIDENCE",
                "closure": {
                    "kind": "existing_terminal_evidence", "path": evidence_path,
                    "sha256": evidence_hash, "bytes": evidence_bytes, "mode": evidence_mode,
                    "authority_receipt": None,
                },
            })
    return result


def validate_slots(slots: Any, *, require_complete: bool) -> list[dict[str, Any]]:
    require(isinstance(slots, list) and len(slots) == 9, "reviewed checkpoint slots must contain nine rows")
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}; observed = set(); result = []
    for row in slots:
        require(isinstance(row, Mapping) and set(row) == SLOT_KEYS, "checkpoint slot exact schema drift")
        key = (row.get("arm"), row.get("seed"))
        require(key in expected and key not in observed, "checkpoint slot arm/seed duplicate or drift")
        observed.add(key)
        require(row.get("epoch") == 11 and row.get("path") == checkpoint_path(*key), "checkpoint path/epoch drift")
        if row.get("closure") is None:
            require(key[0] == "shared_zero4" and all(row.get(k) is None for k in ("sha256", "bytes", "mode")) and row.get("status") == "MISSING_SEALED_TERMINAL_CLOSURE", "invalid incomplete slot")
        else:
            require(is_sha256(row.get("sha256")) and positive_int(row.get("bytes")) and row.get("mode") == "0444", "complete checkpoint pin malformed")
            closure = row["closure"]
            require(isinstance(closure, Mapping) and set(closure) == CLOSURE_PIN_KEYS, "closure pin exact schema drift")
            require(isinstance(closure.get("path"), str) and is_sha256(closure.get("sha256")) and positive_int(closure.get("bytes")) and re.fullmatch(r"0[0-7]{3}", str(closure.get("mode"))), "closure file pin malformed")
            if key[0] == "shared_zero4":
                require(row.get("status") == "REVIEWED_FUTURE_SEALED_TERMINAL" and closure.get("kind") == "future_zero4_sealed_terminal" and isinstance(closure.get("authority_receipt"), Mapping) and set(closure["authority_receipt"]) == FILE_PIN_KEYS, "zero4 requires future closure and authority receipt")
            else:
                require((row["sha256"], row["bytes"]) == _KNOWN_CHECKPOINTS[key] and (closure["path"], closure["sha256"], closure["bytes"], closure["mode"]) == _KNOWN_EVIDENCE[key] and closure["kind"] == "existing_terminal_evidence" and closure["authority_receipt"] is None, "existing terminal evidence drift")
        result.append(dict(row))
    require(observed == expected, "checkpoint slot map incomplete")
    if require_complete and any(row["closure"] is None for row in result):
        raise V6BlockedError("BLOCKED_MISSING_ZERO4_TERMINALS")
    return result


def _resolve_pin_path(pin_path: str, *, repository_root: Path, closure_root: Path, future: bool) -> tuple[Path, Path]:
    root = lexical_absolute(closure_root if future else repository_root)
    candidate = Path(pin_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return lexical_absolute(candidate), root


def verify_closure_bundle(slots: Sequence[Mapping[str, Any]], *, repository_root: Path, closure_root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = validate_slots(list(slots), require_complete=True)
    verified: list[dict[str, Any]] = []
    for row in rows:
        closure = row["closure"]
        future = row["arm"] == "shared_zero4"
        path, root = _resolve_pin_path(str(closure["path"]), repository_root=repository_root, closure_root=closure_root, future=future)
        pin = {key: closure[key] for key in FILE_PIN_KEYS}
        observed = fd_read_regular(path, root=root, expected_mode=pin["mode"], expected_sha256=pin["sha256"], expected_bytes=pin["bytes"])
        if future:
            try:
                payload = json.loads(observed.raw.decode())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise V6AuthorizationError("future zero4 closure malformed") from exc
            auth_require(isinstance(payload, dict) and observed.raw == canonical_bytes(payload), "future zero4 closure not canonical")
            expected_keys = {"schema", "status", "arm", "seed", "epoch", "checkpoint", "slot_binding_sha256", "authority_receipt"}
            auth_require(set(payload) == expected_keys and payload["schema"] == "sealed_terminal_checkpoint_closure_v6" and payload["status"] == "SEALED_TERMINAL_CLOSURE_VERIFIED", "future zero4 closure schema/status drift")
            auth_require((payload["arm"], payload["seed"], payload["epoch"]) == (row["arm"], row["seed"], row["epoch"]), "future closure slot identity drift")
            checkpoint = {key: row[key] for key in ("path", "sha256", "bytes", "mode")}
            auth_require(payload["checkpoint"] == checkpoint and payload["slot_binding_sha256"] == slot_binding(row), "future closure checkpoint binding drift")
            receipt_pin = closure["authority_receipt"]
            auth_require(payload["authority_receipt"] == receipt_pin, "future closure authority pin drift")
            receipt_path, receipt_root = _resolve_pin_path(str(receipt_pin["path"]), repository_root=repository_root, closure_root=closure_root, future=True)
            receipt = read_canonical_json_fd(receipt_path, root=receipt_root, pin=receipt_pin)
            receipt_keys = {"schema", "status", "arm", "seed", "epoch", "checkpoint", "slot_binding_sha256", "issuer"}
            auth_require(set(receipt) == receipt_keys and receipt["schema"] == "terminal_checkpoint_authority_receipt_v6" and receipt["status"] == "AUTHORIZED_SEALED_TERMINAL" and (receipt["arm"], receipt["seed"], receipt["epoch"]) == (row["arm"], row["seed"], row["epoch"]) and receipt["checkpoint"] == checkpoint and receipt["slot_binding_sha256"] == slot_binding(row) and isinstance(receipt["issuer"], str) and receipt["issuer"], "future closure authority receipt drift")
        verified.append({"arm": row["arm"], "seed": row["seed"], "checkpoint_slot_binding_sha256": slot_binding(row), "closure": dict(closure)})
    return canonical_sha256(verified), verified


def validate_cohort(cohort: Any) -> list[dict[str, Any]]:
    require(isinstance(cohort, list) and len(cohort) == N, "frozen cohort must be N=15")
    assets = set(); sessions = set(); result = []
    for row in cohort:
        require(isinstance(row, Mapping) and set(row) == COHORT_KEYS, "cohort row exact schema drift")
        require(isinstance(row["asset_id"], str) and row["asset_id"] not in assets, "cohort asset duplicate")
        require(isinstance(row["session_id"], str) and row["session_id"].startswith("sub-M_ses-") and row["session_id"] not in sessions, "cohort session duplicate/scope drift")
        require(isinstance(row["frozen_path"], str) and row["frozen_path"].startswith("sub-M/") and is_sha256(row["nwb_sha256"]) and positive_int(row["nwb_bytes"]), "cohort data pin malformed")
        assets.add(row["asset_id"]); sessions.add(row["session_id"]); result.append(dict(row))
    return result


def validate_policy_shape(policy: Mapping[str, Any], *, require_complete: bool) -> None:
    require(isinstance(policy, Mapping) and set(policy) == POLICY_KEYS and policy.get("schema") == "dandi_000688_subm_three_arm_frozen_policy_v6", "frozen policy exact schema drift")
    cohort = validate_cohort(policy["cohort"])
    query = policy["query_counts"]
    require(isinstance(query, Mapping) and set(query) == {row["asset_id"] for row in cohort} and all(positive_int(v) for v in query.values()) and sum(query.values()) == QUERY_WINDOWS_PER_VIEW, "fixed query map/totals drift")
    slots = validate_slots(policy["reviewed_checkpoint_slots"], require_complete=require_complete)
    require(canonical_sha256(slots) == policy["reviewed_checkpoint_slots_sha256"], "reviewed slot digest drift")
    require(policy["cpu_policy"] == CPU_POLICY and is_sha256(policy["frozen_predecessor_sha256"]) and is_sha256(policy["source_snapshot_sha256"]) and is_sha256(policy["parity_bundle_sha256"]), "fixed policy binding drift")
    if require_complete:
        require(is_sha256(policy["verified_closure_bundle_sha256"]), "verified closure bundle missing")
        for key in ("repository_root", "closure_root", "output_parent", "external_parent", "claim_root"):
            auth_require(isinstance(policy[key], str) and Path(policy[key]).is_absolute(), f"policy {key} missing")
        auth_require(isinstance(policy["public_key"], Mapping) and set(policy["public_key"]) == FILE_PIN_KEYS, "public-key pin missing")


def build_expected_contract(policy: Mapping[str, Any]) -> dict[str, Any]:
    validate_policy_shape(policy, require_complete=False)
    cohort = [dict(row) for row in policy["cohort"]]
    query = {str(k): int(v) for k, v in policy["query_counts"].items()}
    slots = [dict(row) for row in policy["reviewed_checkpoint_slots"]]
    contract: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA, "scope": "external_subM_CO_held_out_score_only",
        "frozen_predecessor_sha256": policy["frozen_predecessor_sha256"],
        "N": N, "cohort": cohort, "cohort_sha256": canonical_sha256(cohort),
        "arms": list(ARMS), "views": list(VIEWS), "seeds": list(SEEDS), "cell_count": CELL_COUNT,
        "query_window_count_by_asset_id": query, "query_map_sha256": canonical_sha256(query),
        "query_windows_per_view": QUERY_WINDOWS_PER_VIEW,
        "total_model_windows": QUERY_WINDOWS_PER_VIEW * len(VIEWS) * len(SEEDS) * len(ARMS),
        "reviewed_checkpoint_slots": slots,
        "reviewed_checkpoint_slots_sha256": canonical_sha256(slots),
        "verified_closure_bundle_sha256": policy["verified_closure_bundle_sha256"],
        "score_protocol": SCORE_PROTOCOL, "runtime": CPU_POLICY,
        "claim_separation": CLAIM_SEPARATION, "comparison_gates": COMPARISON_GATES,
        "bootstrap_policy": BOOTSTRAP_POLICY,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_expected_contract(contract: Mapping[str, Any], policy: Mapping[str, Any], *, require_complete: bool) -> dict[str, Any]:
    validate_policy_shape(policy, require_complete=require_complete)
    expected = build_expected_contract(policy)
    auth_require(dict(contract) == expected, "contract is not exact reconstruction from frozen predecessor/protocol/reviewed slots")
    return expected


def runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    raw = executable.read_bytes()
    return {"host": socket.gethostname(), "python": {"path": str(executable), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "implementation": getattr(sys.implementation, "name", ""), "version": sys.version}}


def _parse_time(value: Any, label: str) -> datetime:
    auth_require(isinstance(value, str), f"{label} is not ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise V6AuthorizationError(f"{label} malformed") from exc
    auth_require(parsed.tzinfo is not None, f"{label} lacks timezone")
    return parsed.astimezone(timezone.utc)


def authorization_bindings(policy: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frozen_predecessor_sha256": policy["frozen_predecessor_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "cohort_sha256": contract["cohort_sha256"],
        "query_map_sha256": contract["query_map_sha256"],
        "reviewed_checkpoint_slots_sha256": contract["reviewed_checkpoint_slots_sha256"],
        "verified_closure_bundle_sha256": contract["verified_closure_bundle_sha256"],
        "source_snapshot_sha256": policy["source_snapshot_sha256"],
        "parity_bundle_sha256": policy["parity_bundle_sha256"],
        "runtime_identity": policy["runtime_identity"], "cpu_policy": policy["cpu_policy"],
    }


_GRANT_MINT_TOKEN = object()
GRANT_PROOF_KEYS = {
    "schema", "authorization", "signature_envelope", "public_key", "nonce",
    "nonce_claim", "output_root", "external_nwb_root", "issued_at",
    "expires_at", "bindings",
}


class VerifiedGrant:
    """Capability object; direct/ordinary construction is rejected."""
    __slots__ = ("_mint", "_sealed", "_proof", "verified_grant_sha256")

    def __init__(self, *, _mint: object | None = None, proof: Mapping[str, Any] | None = None):
        if _mint is not _GRANT_MINT_TOKEN or proof is None:
            raise TypeError("VerifiedGrant can only be minted by the Ed25519 verifier")
        object.__setattr__(self, "_mint", _mint)
        object.__setattr__(self, "_proof", dict(proof))
        object.__setattr__(self, "verified_grant_sha256", canonical_sha256(proof))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("VerifiedGrant is immutable")
        object.__setattr__(self, name, value)

    @property
    def proof(self) -> dict[str, Any]:
        return dict(self._proof)

    @property
    def output_root(self) -> Path:
        return Path(self._proof["output_root"])

    @property
    def external_nwb_root(self) -> Path:
        return Path(self._proof["external_nwb_root"])

    @property
    def contract_sha256(self) -> str:
        return str(self._proof["bindings"]["contract_sha256"])


def _assert_verified_grant(grant: VerifiedGrant, policy: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    auth_require(type(grant) is VerifiedGrant and getattr(grant, "_mint", None) is _GRANT_MINT_TOKEN, "real verifier-minted grant required")
    auth_require(grant.verified_grant_sha256 == canonical_sha256(grant._proof), "verified grant digest drift")
    expected = validate_expected_contract(contract, policy, require_complete=True)
    proof = grant._proof
    auth_require(set(proof) == GRANT_PROOF_KEYS and proof.get("schema") == GRANT_SCHEMA and proof.get("bindings") == authorization_bindings(policy, expected), "verified grant/contract binding drift")
    for label in ("authorization", "signature_envelope", "public_key"):
        auth_require(isinstance(proof.get(label), Mapping) and set(proof[label]) == FILE_PIN_KEYS, f"verified grant {label} pin drift")
    authorization_pin = proof["authorization"]
    envelope_pin = proof["signature_envelope"]
    public_pin = proof["public_key"]
    authorization_read = fd_read_regular(Path(authorization_pin["path"]), root=Path(authorization_pin["path"]).parent, expected_mode=authorization_pin["mode"], expected_sha256=authorization_pin["sha256"], expected_bytes=authorization_pin["bytes"], max_bytes=1024 * 1024)
    envelope_read = fd_read_regular(Path(envelope_pin["path"]), root=Path(envelope_pin["path"]).parent, expected_mode=envelope_pin["mode"], expected_sha256=envelope_pin["sha256"], expected_bytes=envelope_pin["bytes"], max_bytes=1024 * 1024)
    public_read = fd_read_regular(Path(public_pin["path"]), root=Path(public_pin["path"]).parent, expected_mode=public_pin["mode"], expected_sha256=public_pin["sha256"], expected_bytes=public_pin["bytes"], max_bytes=1024 * 1024)
    try:
        authorization = json.loads(authorization_read.raw.decode())
        envelope = json.loads(envelope_read.raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V6AuthorizationError("verified grant authorization proof malformed") from exc
    auth_require(
        isinstance(authorization, dict) and authorization_read.raw == canonical_bytes(authorization)
        and set(authorization) == AUTH_KEYS and authorization["schema"] == AUTH_SCHEMA
        and authorization["status"] == AUTH_STATUS and authorization["permitted_action"] == AUTH_ACTION
        and authorization["nonce"] == proof["nonce"]
        and authorization["output_root"] == proof["output_root"]
        and authorization["external_nwb_root"] == proof["external_nwb_root"]
        and authorization["issued_at"] == proof["issued_at"]
        and authorization["expires_at"] == proof["expires_at"]
        and authorization["bindings"] == proof["bindings"],
        "verified grant authorization content drift",
    )
    auth_require(
        isinstance(envelope, dict) and envelope_read.raw == canonical_bytes(envelope)
        and set(envelope) == {"schema", "algorithm", "authorization_sha256", "signature_b64"}
        and envelope["schema"] == ENVELOPE_SCHEMA and envelope["algorithm"] == "Ed25519"
        and envelope["authorization_sha256"] == authorization_read.sha256,
        "verified grant signature envelope drift",
    )
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        signature = base64.b64decode(envelope["signature_b64"], validate=True)
        key = serialization.load_pem_public_key(public_read.raw)
        auth_require(isinstance(key, Ed25519PublicKey) and len(signature) == 64, "verified grant Ed25519 proof type/length drift")
        key.verify(signature, authorization_read.raw)
    except (ValueError, TypeError, InvalidSignature) as exc:
        raise V6AuthorizationError("verified grant Ed25519 proof invalid") from exc
    claim_pin = proof.get("nonce_claim")
    auth_require(isinstance(claim_pin, Mapping) and set(claim_pin) == FILE_PIN_KEYS, "nonce claim pin missing")
    claim_path = Path(str(claim_pin["path"])); claim_root = Path(str(policy["claim_root"]))
    claim = read_canonical_json_fd(claim_path, root=claim_root, pin=claim_pin)
    auth_require(claim == {"authorization_sha256": authorization_read.sha256, "nonce": proof["nonce"], "schema": "atomic_nonce_claim_v6"}, "nonce claim content drift")


def _claim_nonce(nonce: str, authorization_sha256: str, claim_root: Path) -> dict[str, Any]:
    auth_require(is_sha256(nonce), "nonce malformed")
    claim_root = lexical_absolute(claim_root)
    assert_absolute_chain_no_symlink(claim_root.parent)
    if not claim_root.exists():
        claim_root.mkdir(mode=0o700)
    assert_absolute_chain_no_symlink(claim_root)
    path = claim_root / f"{nonce}.claimed.json"
    raw = canonical_bytes({"authorization_sha256": authorization_sha256, "nonce": nonce, "schema": "atomic_nonce_claim_v6"})
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise V6AuthorizationError("authorization nonce already claimed") from exc
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor); os.fchmod(descriptor, 0o444)
        metadata = os.fstat(descriptor)
        auth_require(stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444, "nonce claim sealing failed")
    finally:
        os.close(descriptor)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": "0444"}


def verify_authorization_and_claim(
    *, authorization_path: Path, signature_envelope_path: Path,
    output_root: Path, external_nwb_root: Path, policy: Mapping[str, Any],
    contract: Mapping[str, Any], now: datetime,
) -> VerifiedGrant:
    """The only supported grant constructor; performs real Ed25519 verify."""
    auth_require(now.tzinfo is not None, "verification time must be timezone-aware")
    expected = validate_expected_contract(contract, policy, require_complete=True)
    closure_digest, _ = verify_closure_bundle(
        policy["reviewed_checkpoint_slots"], repository_root=Path(policy["repository_root"]),
        closure_root=Path(policy["closure_root"]),
    )
    auth_require(closure_digest == policy["verified_closure_bundle_sha256"] == expected["verified_closure_bundle_sha256"], "live closure bundle digest drift")
    auth_require(policy["runtime_identity"] == runtime_identity() and policy["cpu_policy"] == CPU_POLICY, "runtime identity/CPU policy drift")
    auth_require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CUDA_VISIBLE_DEVICES must be empty")

    public_pin = policy["public_key"]
    public_path = Path(str(public_pin["path"])); public_root = public_path.parent
    public = fd_read_regular(public_path, root=public_root, expected_mode=public_pin["mode"], expected_sha256=public_pin["sha256"], expected_bytes=public_pin["bytes"])
    auth_pin_root = authorization_path.parent
    authorization_read = fd_read_regular(authorization_path, root=auth_pin_root, expected_mode="0444", max_bytes=1024 * 1024)
    envelope_read = fd_read_regular(signature_envelope_path, root=signature_envelope_path.parent, expected_mode="0444", max_bytes=1024 * 1024)
    try:
        authorization = json.loads(authorization_read.raw.decode())
        envelope = json.loads(envelope_read.raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V6AuthorizationError("authorization/signature JSON malformed") from exc
    auth_require(isinstance(authorization, dict) and authorization_read.raw == canonical_bytes(authorization) and set(authorization) == AUTH_KEYS, "authorization exact canonical schema drift")
    auth_require(isinstance(envelope, dict) and envelope_read.raw == canonical_bytes(envelope) and set(envelope) == {"schema", "algorithm", "authorization_sha256", "signature_b64"}, "signature envelope exact schema drift")
    auth_require(authorization["schema"] == AUTH_SCHEMA and authorization["status"] == AUTH_STATUS and authorization["permitted_action"] == AUTH_ACTION and authorization["cell_count"] == CELL_COUNT, "authorization scope/status drift")
    issued, expires = _parse_time(authorization["issued_at"], "issued_at"), _parse_time(authorization["expires_at"], "expires_at")
    auth_require(issued <= now.astimezone(timezone.utc) <= expires, "authorization expired or not yet valid")
    auth_require(0 < (expires - issued).total_seconds() <= MAX_AUTH_SECONDS, "authorization validity exceeds 15 minutes")
    auth_require(is_sha256(authorization["nonce"]), "authorization nonce malformed")
    expected_output = lexical_absolute(output_root); expected_external = lexical_absolute(external_nwb_root)
    output_parent, external_parent = Path(policy["output_parent"]), Path(policy["external_parent"])
    assert_under(expected_output, output_parent, "output root"); assert_under(expected_external, external_parent, "external root")
    auth_require(expected_output != lexical_absolute(output_parent), "output root must be a child of fixed parent")
    assert_absolute_chain_no_symlink(expected_output.parent); assert_absolute_chain_no_symlink(expected_external)
    auth_require(authorization["output_root"] == str(expected_output) and authorization["external_nwb_root"] == str(expected_external), "authorization root binding drift")
    auth_require(authorization["bindings"] == authorization_bindings(policy, expected) and set(authorization["bindings"]) == BINDING_KEYS, "authorization digest bindings drift")
    auth_require(envelope["schema"] == ENVELOPE_SCHEMA and envelope["algorithm"] == "Ed25519" and envelope["authorization_sha256"] == authorization_read.sha256, "signature envelope binding drift")
    try:
        signature = base64.b64decode(envelope["signature_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise V6AuthorizationError("signature base64 malformed") from exc
    auth_require(len(signature) == 64, "Ed25519 signature length drift")
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        key = serialization.load_pem_public_key(public.raw)
        auth_require(isinstance(key, Ed25519PublicKey), "public key is not Ed25519")
        key.verify(signature, authorization_read.raw)
    except (ValueError, InvalidSignature) as exc:
        raise V6AuthorizationError("invalid Ed25519 authorization signature") from exc
    claim_pin = _claim_nonce(authorization["nonce"], authorization_read.sha256, Path(policy["claim_root"]))
    proof = {
        "schema": GRANT_SCHEMA,
        "authorization": {"path": str(lexical_absolute(authorization_path)), "sha256": authorization_read.sha256, "bytes": authorization_read.bytes, "mode": authorization_read.mode},
        "signature_envelope": {"path": str(lexical_absolute(signature_envelope_path)), "sha256": envelope_read.sha256, "bytes": envelope_read.bytes, "mode": envelope_read.mode},
        "public_key": {"path": str(lexical_absolute(public_path)), "sha256": public.sha256, "bytes": public.bytes, "mode": public.mode},
        "nonce": authorization["nonce"],
        "nonce_claim": claim_pin, "output_root": str(expected_output),
        "external_nwb_root": str(expected_external), "issued_at": authorization["issued_at"],
        "expires_at": authorization["expires_at"], "bindings": authorization["bindings"],
    }
    return VerifiedGrant(_mint=_GRANT_MINT_TOKEN, proof=proof)


@dataclass(frozen=True, order=True)
class CellKey:
    session_id: str
    asset_id: str
    view: str
    seed: int
    arm: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_cell_keys(contract: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[CellKey, ...]:
    validate_expected_contract(contract, policy, require_complete=True)
    keys = tuple(CellKey(row["session_id"], row["asset_id"], view, seed, arm) for row in contract["cohort"] for view in VIEWS for seed in SEEDS for arm in ARMS)
    auth_require(len(keys) == len(set(keys)) == CELL_COUNT, "exact 270 cell map drift")
    return keys


def _ensure_output_root(grant: VerifiedGrant, policy: Mapping[str, Any], contract: Mapping[str, Any]) -> Path:
    _assert_verified_grant(grant, policy, contract)
    root = lexical_absolute(grant.output_root); parent = Path(policy["output_parent"])
    assert_under(root, parent, "grant output root"); assert_absolute_chain_no_symlink(root.parent)
    if root.exists() or root.is_symlink():
        assert_existing_chain_no_symlink(root, root); ledger_require(root.is_dir(), "output root is not a directory")
    else:
        root.mkdir(mode=0o700)
    return root


def _ensure_parent(path: Path, root: Path) -> None:
    candidate, base = assert_under(path, root, "output")
    current = base
    for part in candidate.parent.relative_to(base).parts:
        current /= part
        if current.exists() or current.is_symlink():
            assert_existing_chain_no_symlink(current, base); ledger_require(current.is_dir(), "output parent not directory")
        else:
            current.mkdir(mode=0o700)


def _write_exclusive(path: Path, raw: bytes, root: Path) -> dict[str, Any]:
    _ensure_parent(path, root)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise V6LedgerError(f"duplicate output artifact: {path}") from exc
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor); os.fchmod(descriptor, 0o444)
        metadata = os.fstat(descriptor)
        ledger_require(stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444 and metadata.st_size == len(raw), "fd-based immutable write failed")
    finally:
        os.close(descriptor)
    observed = fd_read_regular(path, root=root, expected_mode="0444", expected_sha256=hashlib.sha256(raw).hexdigest(), expected_bytes=len(raw))
    return {"path": str(path.relative_to(root)), "sha256": observed.sha256, "bytes": observed.bytes, "mode": "0444"}


def _artifact_path(root: Path, key: CellKey) -> Path:
    return root / "artifacts" / key.session_id / key.view / f"seed_{key.seed}" / f"{key.arm}.predictions_targets.npz"


def _cell_path(root: Path, key: CellKey) -> Path:
    return root / "cells" / key.session_id / key.view / f"seed_{key.seed}" / f"{key.arm}.json"


def _validate_arrays(prediction: Any, target: Any, expected_rows: int) -> tuple[Any, Any]:
    import numpy as np
    ledger_require(isinstance(prediction, np.ndarray) and isinstance(target, np.ndarray), "prediction/target must be numpy arrays")
    ledger_require(prediction.dtype == target.dtype == np.dtype("float32"), "prediction/target dtype must be exact float32")
    ledger_require(prediction.shape == target.shape == (expected_rows, OUTPUT_DIM), "prediction/target shape or row count drift")
    ledger_require(prediction.flags.c_contiguous and target.flags.c_contiguous, "prediction/target must be C-contiguous")
    ledger_require(bool(np.isfinite(prediction).all()) and bool(np.isfinite(target).all()), "prediction/target contains NaN/Inf")
    return prediction, target


def _npz_bytes(prediction: Any, target: Any) -> bytes:
    buffer = io.BytesIO()
    import numpy as np
    np.savez_compressed(buffer, prediction=prediction, target=target)
    return buffer.getvalue()


def _safe_npz_from_raw(raw: bytes, *, expected_rows: int) -> tuple[Any, Any]:
    import numpy as np
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            ledger_require([info.filename for info in infos] == ["prediction.npy", "target.npy"], "NPZ members must be exact prediction/target arrays")
            ledger_require(all(not info.is_dir() and not (info.flag_bits & 0x1) and info.file_size <= expected_rows * OUTPUT_DIM * 4 + 4096 for info in infos), "unsafe NPZ member")
        with np.load(io.BytesIO(raw), allow_pickle=False) as package:
            ledger_require(package.files == ["prediction", "target"], "NPZ array key drift")
            prediction = package["prediction"]
            target = package["target"]
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        raise V6LedgerError("unsafe or malformed NPZ") from exc
    return _validate_arrays(prediction, target, expected_rows)


def _internal_variance_weighted_r2(prediction: Any, target: Any) -> float:
    import numpy as np
    pred = prediction.astype(np.float64, copy=False); truth = target.astype(np.float64, copy=False)
    residual = float(np.square(truth - pred).sum(dtype=np.float64))
    centered = truth - truth.mean(axis=0, dtype=np.float64)
    total = float(np.square(centered).sum(dtype=np.float64))
    ledger_require(math.isfinite(residual) and math.isfinite(total) and total > 0.0, "variance-weighted R2 undefined/nonfinite")
    value = 1.0 - residual / total
    ledger_require(math.isfinite(value) and value <= 1.0 + R2_ATOL, "computed R2 exceeds mathematical upper bound")
    return min(value, 1.0)


def publish_prediction_target_artifact(
    grant: VerifiedGrant, policy: Mapping[str, Any], contract: Mapping[str, Any],
    *, key: CellKey, prediction: Any, target: Any,
) -> dict[str, Any]:
    root = _ensure_output_root(grant, policy, contract)
    ledger_require(key in set(expected_cell_keys(contract, policy)), "unknown cell key")
    rows = contract["query_window_count_by_asset_id"][key.asset_id]
    prediction, target = _validate_arrays(prediction, target, rows)
    raw = _npz_bytes(prediction, target)
    pin = _write_exclusive(_artifact_path(root, key), raw, root)
    pin.update({
        "verified_grant_sha256": grant.verified_grant_sha256,
        "arrays": {
            "prediction": {"dtype": "float32", "shape": [rows, OUTPUT_DIM]},
            "target": {"dtype": "float32", "shape": [rows, OUTPUT_DIM]},
        },
    })
    return pin


def _load_artifact(pin: Mapping[str, Any], *, root: Path, key: CellKey, grant: VerifiedGrant, contract: Mapping[str, Any]) -> tuple[Any, Any]:
    ledger_require(isinstance(pin, Mapping) and set(pin) == ARTIFACT_PIN_KEYS, "artifact exact pin schema drift")
    expected_path = str(_artifact_path(root, key).relative_to(root))
    ledger_require(pin["path"] == expected_path and pin["mode"] == "0444" and pin["verified_grant_sha256"] == grant.verified_grant_sha256, "artifact path/mode/grant binding drift")
    rows = contract["query_window_count_by_asset_id"][key.asset_id]
    expected_arrays = {"prediction": {"dtype": "float32", "shape": [rows, OUTPUT_DIM]}, "target": {"dtype": "float32", "shape": [rows, OUTPUT_DIM]}}
    ledger_require(pin["arrays"] == expected_arrays and is_sha256(pin["sha256"]) and positive_int(pin["bytes"]), "artifact array/hash/size binding drift")
    observed = fd_read_regular(root / expected_path, root=root, expected_mode="0444", expected_sha256=pin["sha256"], expected_bytes=pin["bytes"])
    return _safe_npz_from_raw(observed.raw, expected_rows=rows)


def publish_cell_result(
    grant: VerifiedGrant, policy: Mapping[str, Any], contract: Mapping[str, Any],
    *, key: CellKey, prediction_target_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    root = _ensure_output_root(grant, policy, contract)
    ledger_require(key in set(expected_cell_keys(contract, policy)), "unknown cell key")
    prediction, target = _load_artifact(prediction_target_artifact, root=root, key=key, grant=grant, contract=contract)
    value = _internal_variance_weighted_r2(prediction, target)
    payload = {
        "schema": CELL_SCHEMA, "status": "COMPLETE",
        "verified_grant_sha256": grant.verified_grant_sha256,
        "cell": key.as_dict(), "r2": value,
        "query_window_count": contract["query_window_count_by_asset_id"][key.asset_id],
        "prediction_target_artifact": dict(prediction_target_artifact),
        "contract_sha256": contract["contract_sha256"],
    }
    return _write_exclusive(_cell_path(root, key), canonical_bytes(payload), root)


def _read_cell(path: Path, *, root: Path, key: CellKey, grant: VerifiedGrant, contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_canonical_json_fd(path, root=root, expected_mode="0444", max_bytes=4 * 1024 * 1024)
    ledger_require(set(payload) == CELL_KEYS and payload["schema"] == CELL_SCHEMA and payload["status"] == "COMPLETE", "cell exact schema/status drift")
    ledger_require(isinstance(payload["cell"], Mapping) and set(payload["cell"]) == CELL_ID_KEYS and payload["cell"] == key.as_dict(), "cell identity drift")
    ledger_require(payload["verified_grant_sha256"] == grant.verified_grant_sha256 and payload["contract_sha256"] == contract["contract_sha256"], "cell grant/contract binding drift")
    rows = contract["query_window_count_by_asset_id"][key.asset_id]
    ledger_require(payload["query_window_count"] == rows, "cell query-window count drift")
    prediction, target = _load_artifact(payload["prediction_target_artifact"], root=root, key=key, grant=grant, contract=contract)
    expected_r2 = _internal_variance_weighted_r2(prediction, target)
    value = payload["r2"]
    ledger_require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) <= 1.0 + R2_ATOL and abs(float(value) - expected_r2) <= R2_ATOL, "cell R2 is not exact internal recomputation")
    return payload


def _scan_base(grant: VerifiedGrant, policy: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[Path, tuple[CellKey, ...], dict[CellKey, dict[str, Any]]]:
    root = _ensure_output_root(grant, policy, contract); keys = expected_cell_keys(contract, policy)
    allowed_top = {"artifacts", "cells", "aggregate"}
    for child in root.iterdir():
        ledger_require(not child.is_symlink() and child.name in allowed_top and child.is_dir(), "unknown/symlink/non-directory output topology")
    expected_cells = {key: _cell_path(root, key) for key in keys}; expected_artifacts = {_artifact_path(root, key) for key in keys}
    observed_cells = set(); cells_root = root / "cells"
    if cells_root.exists() or cells_root.is_symlink():
        assert_existing_chain_no_symlink(cells_root, root)
        allowed_dirs = {parent for path in expected_cells.values() for parent in path.parents if parent == cells_root or cells_root in parent.parents}
        for path in cells_root.rglob("*"):
            ledger_require(not path.is_symlink(), "symlink in cell topology")
            if path.is_file(): observed_cells.add(path)
            elif path.is_dir(): ledger_require(path in allowed_dirs, "unknown cell directory")
    ledger_require(observed_cells <= set(expected_cells.values()), "unknown cell file")
    payloads = {}
    for key, path in expected_cells.items():
        if path.exists() or path.is_symlink(): payloads[key] = _read_cell(path, root=root, key=key, grant=grant, contract=contract)
    referenced = {root / payload["prediction_target_artifact"]["path"] for payload in payloads.values()}
    observed_artifacts = set(); artifacts_root = root / "artifacts"
    if artifacts_root.exists() or artifacts_root.is_symlink():
        assert_existing_chain_no_symlink(artifacts_root, root)
        allowed_dirs = {parent for path in expected_artifacts for parent in path.parents if parent == artifacts_root or artifacts_root in parent.parents}
        for path in artifacts_root.rglob("*"):
            ledger_require(not path.is_symlink(), "symlink in artifact topology")
            if path.is_file(): observed_artifacts.add(path)
            elif path.is_dir(): ledger_require(path in allowed_dirs, "unknown artifact directory")
    ledger_require(observed_artifacts == referenced, "partial/unknown artifact state")
    return root, keys, payloads


def _bootstrap(delta: Any, rng: Any) -> tuple[float, float]:
    import numpy as np
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 10_000):
        count = min(10_000, BOOTSTRAP_REPLICATES - start)
        sessions = rng.integers(0, N, size=(count, N)); seeds = rng.integers(0, 3, size=(count, N, 3))
        values[start:start + count] = delta[sessions[:, :, None], seeds].mean(axis=(1, 2))
    result = np.quantile(values, [0.025, 0.975], method="linear")
    return float(result[0]), float(result[1])


def _reconstruct(payloads: Mapping[CellKey, Mapping[str, Any]], contract: Mapping[str, Any], grant: VerifiedGrant) -> dict[str, Any]:
    import numpy as np
    sessions = [(row["session_id"], row["asset_id"]) for row in contract["cohort"]]
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)); comparisons = {}
    for name, comparator in COMPARISONS:
        views = {}
        for view in VIEWS:
            t4 = np.asarray([[payloads[CellKey(session, asset, view, seed, "shared_t4")]["r2"] for seed in SEEDS] for session, asset in sessions], dtype=np.float64)
            other = np.asarray([[payloads[CellKey(session, asset, view, seed, comparator)]["r2"] for seed in SEEDS] for session, asset in sessions], dtype=np.float64)
            delta = t4 - other; seed_means = delta.mean(0); session_means = delta.mean(1); t4_seeds = t4.mean(0)
            lower, upper = _bootstrap(delta, rng); positive = int((session_means > 0).sum()); grand = float(delta.mean())
            gates = {
                "grand_paired_mean_at_least_0p03": grand >= 0.03,
                "all_three_seed_means_strictly_positive": bool((seed_means > 0).all()),
                "at_least_12_of_15_session_means_positive": positive >= 12,
                "hierarchical_bootstrap_lower_95_strictly_positive": lower > 0,
                "shared_t4_absolute_grand_and_seed_means_strictly_positive": bool(t4.mean() > 0 and (t4_seeds > 0).all()),
            }
            views[view] = {
                "grand_paired_mean_r2": grand,
                "seed_means_r2": {str(seed): float(seed_means[i]) for i, seed in enumerate(SEEDS)},
                "session_cross_seed_means_r2": {sessions[i][0]: float(value) for i, value in enumerate(session_means)},
                "positive_session_count": positive, "positive_session_required": 12,
                "hierarchical_bootstrap_95": {"lower": lower, "upper": upper},
                "shared_t4_absolute_grand_r2": float(t4.mean()),
                "shared_t4_absolute_seed_means_r2": {str(seed): float(t4_seeds[i]) for i, seed in enumerate(SEEDS)},
                "gates": gates, "view_pass": all(gates.values()),
            }
        comparisons[name] = {"views": views, "comparison_pass": all(views[v]["view_pass"] for v in VIEWS), "cross_view_rescue_used": False}
    return {
        "schema": AGGREGATE_SCHEMA, "status": "RECONSTRUCTED_FROM_270_VERIFIED_CELLS",
        "verified_grant_sha256": grant.verified_grant_sha256,
        "contract_sha256": contract["contract_sha256"],
        "statistics_source": "internal_r2_reconstructed_from_safe_npz_and_270_verified_cells",
        "verified_cell_count": CELL_COUNT, "bootstrap_policy": BOOTSTRAP_POLICY,
        "comparisons": comparisons,
        "overall_three_arm_claim_pass": all(comparisons[name]["comparison_pass"] for name, _ in COMPARISONS),
    }


def scan_resume_state(grant: VerifiedGrant, policy: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    root, keys, payloads = _scan_base(grant, policy, contract)
    aggregate_path = root / "aggregate" / "aggregate.json"
    if aggregate_path.parent.exists() or aggregate_path.parent.is_symlink():
        assert_existing_chain_no_symlink(aggregate_path.parent, root)
        entries = list(aggregate_path.parent.iterdir())
        ledger_require(entries == [aggregate_path], "partial/unknown aggregate state")
        ledger_require(len(payloads) == CELL_COUNT, "existing aggregate without 270 verified cells")
        observed = read_canonical_json_fd(aggregate_path, root=root, expected_mode="0444", max_bytes=16 * 1024 * 1024)
        ledger_require(set(observed) == AGGREGATE_KEYS, "aggregate schema/mode drift")
        expected = _reconstruct(payloads, contract, grant)
        ledger_require(observed == expected, "existing aggregate is not canonical exact recomputation")
    complete = set(payloads); missing = [key for key in keys if key not in complete]
    return {"expected_cell_count": CELL_COUNT, "complete_cell_count": len(complete), "missing_cell_count": len(missing), "complete": [key.as_dict() for key in keys if key in complete], "missing": [key.as_dict() for key in missing]}


def publish_full_aggregate(grant: VerifiedGrant, policy: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    root, _, payloads = _scan_base(grant, policy, contract)
    ledger_require(len(payloads) == CELL_COUNT, "aggregate forbidden before all 270 verified cells")
    payload = _reconstruct(payloads, contract, grant)
    return _write_exclusive(root / "aggregate" / "aggregate.json", canonical_bytes(payload), root)


def refuse_blocked_execution() -> None:
    raise V6BlockedError("BLOCKED_MISSING_ZERO4_TERMINALS")
