"""Grant-bound append-only ledger for the external sub-M three-arm V5 plan.

This module is a control plane.  Importing it does not import Torch, open a
checkpoint/NWB file, run a model, or compute R2.  The production package is
blocked until the three shared-zero4 terminal closures exist and an external
verifier issues an AuthorizationGrant whose digests bind the finished
contract.  Synthetic tests exercise the ledger and aggregate machinery only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


ARMS = ("shared_t4", "shared_zero4", "shared_ts4")
VIEWS = ("sua", "pseudo_mua")
SEEDS = (42, 43, 44)
COMPARISONS = (
    ("t4_minus_zero4", "shared_zero4"),
    ("t4_minus_ts4", "shared_ts4"),
)
EXPECTED_SESSION_COUNT = 15
EXPECTED_CELL_COUNT = 270
EXPECTED_QUERY_WINDOWS_PER_VIEW = 708_795
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 68_820_260_805

CONTRACT_SCHEMA = "dandi_000688_subm_three_arm_score_contract_v5"
GRANT_SCHEMA = "dandi_000688_subm_three_arm_verified_external_grant_v5"
GRANT_STATUS = "VERIFIED_EXTERNAL_AUTHORIZATION"
GRANT_ACTION = "publish_resume_and_aggregate_exact_270_cells_v5"
CELL_SCHEMA = "dandi_000688_subm_three_arm_cell_result_v5"
AGGREGATE_SCHEMA = "dandi_000688_subm_three_arm_reconstructed_aggregate_v5"

CONTRACT_KEYS = {
    "schema", "scope", "N", "cohort", "cohort_sha256", "arms", "views",
    "seeds", "cell_count", "query_window_count_by_asset_id",
    "query_map_sha256", "query_windows_per_view", "total_model_windows",
    "checkpoint_slots", "checkpoint_slots_sha256", "comparison_gates",
    "bootstrap_policy", "claim_separation", "score_protocol", "runtime",
    "contract_sha256",
}
COHORT_KEYS = {"asset_id", "session_id", "frozen_path", "nwb_sha256", "nwb_bytes"}
SLOT_KEYS = {"arm", "seed", "epoch", "path", "sha256", "bytes", "mode", "status", "closure"}
CLOSURE_KEYS = {"schema", "status", "path", "sha256", "bytes", "mode", "slot_binding_sha256"}
CELL_KEYS = {
    "schema", "status", "grant_binding", "cell", "r2",
    "query_window_count", "prediction_target_artifact", "contract_sha256",
}
CELL_ID_KEYS = {"session_id", "asset_id", "view", "seed", "arm"}
ARTIFACT_KEYS = {"path", "sha256", "bytes", "mode"}
AGGREGATE_KEYS = {
    "schema", "status", "grant_binding", "contract_sha256",
    "statistics_source", "verified_cell_count", "bootstrap_policy",
    "comparisons", "overall_three_arm_claim_pass",
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
    "session_resample_count": EXPECTED_SESSION_COUNT,
    "seed_resample_count_within_each_session": len(SEEDS),
    "quantiles": [0.025, 0.975],
    "quantile_method": "linear",
    "cross_comparison_pooling": "FORBIDDEN",
    "cross_view_rescue": "FORBIDDEN",
}


class ThreeArmV5Error(RuntimeError):
    pass


class ThreeArmV5BlockedError(ThreeArmV5Error):
    pass


class ThreeArmV5AuthorizationError(PermissionError):
    pass


class ThreeArmV5LedgerError(ThreeArmV5Error):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeArmV5Error(message)


def auth_require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeArmV5AuthorizationError(message)


def ledger_require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeArmV5LedgerError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def checkpoint_path(arm: str, seed: int) -> str:
    if arm in {"shared_t4", "shared_ts4"}:
        return (
            "sua_exploration/checkpoints/"
            f"t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_{arm}_s{seed}/"
            "epoch_ckpts/epoch_011.ckpt"
        )
    return (
        "sua_exploration/checkpoints/"
        f"t4_paired_view_c1_shared_zero4_source_prelaunch_v4_20260805_{arm}_s{seed}/"
        "epoch_ckpts/epoch_011.ckpt"
    )


_KNOWN_CHECKPOINTS: dict[tuple[str, int], tuple[str, int]] = {
    ("shared_t4", 42): ("ab9df840a07d7aeb6cc417bb684f1f5e0265d50f98168400ac915647cdfd7b9f", 64_768_898),
    ("shared_t4", 43): ("05c05b3ab82a2fba43c55aca523248982a954faf5f0363a0235a29d64e57ab22", 64_768_898),
    ("shared_t4", 44): ("a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6", 64_769_167),
    ("shared_ts4", 42): ("a21da5a72a991bd2665af50572a4132998ac79d7f801879048553efcdc8281b2", 64_768_898),
    ("shared_ts4", 43): ("c8dd22dfadb2bc11555fc21abe464316886d221e2dbcd71bf20a6bffe9cb158e", 64_768_898),
    ("shared_ts4", 44): ("a2d877ac81a4e553e5221c54e465db26eba8592888b8cb5339e9dfc4acd66ced", 64_769_167),
}

_KNOWN_CLOSURES: dict[tuple[str, int], tuple[str, str, int, str]] = {
    ("shared_t4", 42): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s42/shared_t4_s42_sua.json", "689a692af59010e4799fe26aa900f9fba8a8cd78238b14e4eae36fce5f16d1aa", 127_124, "0664"),
    ("shared_t4", 43): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s43/shared_t4_s43_sua.json", "e1e9b5453d67535268ae099226cd61320c869e95f02355e02bd273c2261f295d", 127_125, "0664"),
    ("shared_t4", 44): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s44/shared_t4_s44_sua.json", "add3206d4ac5ad28dcb81132eb156d024f99d73fdbc4e84f0b2a7577393c1712", 127_367, "0664"),
    ("shared_ts4", 42): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_ts4_s42/shared_ts4_s42_sua.json", "f687fb5842438586c2bbef970306fe6410773097188f13e84e11aee9f31a5521", 127_169, "0664"),
    ("shared_ts4", 43): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_ts4_s43/shared_ts4_s43_sua.json", "b214fa4bbf596fba7a971a6200b0851f4f6b0a925dc615de0e071bee3a6a4a36", 127_183, "0664"),
    ("shared_ts4", 44): ("sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_ts4_s44/shared_ts4_s44_sua.json", "b65c10a75e066437e8962d181c442f6a3dbd793bdce0a1cec9a13807aabba969", 127_408, "0664"),
}


def _slot_binding(arm: str, seed: int, epoch: int, path: str, digest: str, size: int, mode: str) -> str:
    return canonical_sha256({
        "arm": arm, "seed": seed, "epoch": epoch, "path": path,
        "sha256": digest, "bytes": size, "mode": mode,
    })


def checkpoint_slots_v5() -> list[dict[str, Any]]:
    """Return metadata pins only; never stat or open a checkpoint."""
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            path = checkpoint_path(arm, seed)
            known = _KNOWN_CHECKPOINTS.get((arm, seed))
            if known is None:
                rows.append({
                    "arm": arm, "seed": seed, "epoch": 11, "path": path,
                    "sha256": None, "bytes": None, "mode": None,
                    "status": "MISSING_SEALED_TERMINAL_CLOSURE", "closure": None,
                })
                continue
            digest, size = known
            closure_path, closure_hash, closure_bytes, closure_mode = _KNOWN_CLOSURES[(arm, seed)]
            rows.append({
                "arm": arm, "seed": seed, "epoch": 11, "path": path,
                "sha256": digest, "bytes": size, "mode": "0444",
                "status": "PINNED_EXISTING_TERMINAL_EVIDENCE_REQUIRES_0444_AT_EXECUTION",
                "closure": {
                    "schema": "existing_terminal_score_evidence_pin_v5",
                    "status": "EXISTING_TERMINAL_EVIDENCE_PINNED",
                    "path": closure_path, "sha256": closure_hash,
                    "bytes": closure_bytes, "mode": closure_mode,
                    "slot_binding_sha256": _slot_binding(arm, seed, 11, path, digest, size, "0444"),
                },
            })
    return rows


def missing_checkpoint_slots(slots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in slots if row.get("closure") is None]


def validate_checkpoint_slots(slots: Sequence[Mapping[str, Any]], *, require_complete: bool) -> list[dict[str, Any]]:
    require(isinstance(slots, list) and len(slots) == 9, "checkpoint slot count must be nine")
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed: set[tuple[str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for raw in slots:
        require(isinstance(raw, Mapping) and set(raw) == SLOT_KEYS, "checkpoint slot schema drift")
        arm, seed = raw.get("arm"), raw.get("seed")
        require(arm in ARMS and seed in SEEDS, "checkpoint slot arm/seed drift")
        key = (str(arm), int(seed))
        require(key not in observed, "duplicate checkpoint slot")
        observed.add(key)
        require(raw.get("epoch") == 11 and raw.get("path") == checkpoint_path(*key), "checkpoint path/epoch drift")
        closure = raw.get("closure")
        if closure is None:
            require(
                key[0] == "shared_zero4"
                and raw.get("sha256") is None and raw.get("bytes") is None and raw.get("mode") is None
                and raw.get("status") == "MISSING_SEALED_TERMINAL_CLOSURE",
                "incomplete slot is not an exact zero4 closure placeholder",
            )
        else:
            require(is_sha256(raw.get("sha256")), "checkpoint SHA-256 malformed")
            require(_is_positive_int(raw.get("bytes")), "checkpoint byte size malformed")
            require(raw.get("mode") == "0444", "checkpoint immutable mode drift")
            require(isinstance(closure, Mapping) and set(closure) == CLOSURE_KEYS, "terminal closure schema drift")
            require(is_sha256(closure.get("sha256")), "terminal closure SHA-256 malformed")
            require(_is_positive_int(closure.get("bytes")), "terminal closure byte size malformed")
            require(re.fullmatch(r"0[0-7]{3}", str(closure.get("mode"))) is not None, "terminal closure mode malformed")
            require(
                closure.get("slot_binding_sha256") == _slot_binding(
                    key[0], key[1], 11, str(raw["path"]), str(raw["sha256"]), int(raw["bytes"]), "0444"
                ),
                "terminal closure does not bind the complete checkpoint slot",
            )
            if key[0] == "shared_zero4":
                require(
                    raw.get("status") == "PINNED_FUTURE_SEALED_TERMINAL_CLOSURE"
                    and closure.get("schema") == "sealed_terminal_checkpoint_closure_v5"
                    and closure.get("status") == "SEALED_TERMINAL_CLOSURE_VERIFIED"
                    and closure.get("mode") == "0444",
                    "zero4 must come from a future sealed terminal closure",
                )
            else:
                expected_checkpoint = _KNOWN_CHECKPOINTS[key]
                expected_closure = _KNOWN_CLOSURES[key]
                require(
                    (raw.get("sha256"), raw.get("bytes")) == expected_checkpoint
                    and (
                        closure.get("path"), closure.get("sha256"), closure.get("bytes"), closure.get("mode")
                    ) == expected_closure,
                    "known terminal evidence drift",
                )
        normalized.append(dict(raw))
    require(observed == expected, "checkpoint arm/seed map incomplete")
    if require_complete and missing_checkpoint_slots(normalized):
        raise ThreeArmV5BlockedError("BLOCKED_MISSING_ZERO4_TERMINALS")
    return normalized


def validate_frozen_cohort(cohort: Any) -> list[dict[str, Any]]:
    require(isinstance(cohort, list) and len(cohort) == EXPECTED_SESSION_COUNT, "frozen cohort must contain N=15")
    assets: set[str] = set()
    sessions: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in cohort:
        require(isinstance(raw, Mapping) and set(raw) == COHORT_KEYS, "frozen cohort row schema drift")
        asset, session = raw["asset_id"], raw["session_id"]
        require(isinstance(asset, str) and asset and asset not in assets, "cohort asset duplicate/drift")
        require(isinstance(session, str) and session.startswith("sub-M_ses-") and session not in sessions, "cohort session duplicate/drift")
        require(isinstance(raw["frozen_path"], str) and raw["frozen_path"].startswith("sub-M/"), "cohort path scope drift")
        require(is_sha256(raw["nwb_sha256"]) and _is_positive_int(raw["nwb_bytes"]), "cohort NWB pin malformed")
        assets.add(asset); sessions.add(session); rows.append(dict(raw))
    return rows


def build_contract_v5(
    *, cohort: Sequence[Mapping[str, Any]], query_counts: Mapping[str, int],
    checkpoint_slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frozen = validate_frozen_cohort(list(cohort))
    slots = validate_checkpoint_slots(list(checkpoint_slots), require_complete=False)
    require(set(query_counts) == {row["asset_id"] for row in frozen}, "query-count/cohort map drift")
    require(all(_is_positive_int(v) for v in query_counts.values()), "query count malformed")
    require(sum(query_counts.values()) == EXPECTED_QUERY_WINDOWS_PER_VIEW, "N15 query total drift")
    query_map = {str(k): int(v) for k, v in query_counts.items()}
    contract: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "scope": "external_subM_CO_held_out_score_only",
        "N": EXPECTED_SESSION_COUNT,
        "cohort": frozen,
        "cohort_sha256": canonical_sha256(frozen),
        "arms": list(ARMS), "views": list(VIEWS), "seeds": list(SEEDS),
        "cell_count": EXPECTED_CELL_COUNT,
        "query_window_count_by_asset_id": query_map,
        "query_map_sha256": canonical_sha256(query_map),
        "query_windows_per_view": EXPECTED_QUERY_WINDOWS_PER_VIEW,
        "total_model_windows": EXPECTED_QUERY_WINDOWS_PER_VIEW * len(VIEWS) * len(SEEDS) * len(ARMS),
        "checkpoint_slots": slots,
        "checkpoint_slots_sha256": canonical_sha256(slots),
        "comparison_gates": COMPARISON_GATES,
        "bootstrap_policy": BOOTSTRAP_POLICY,
        "claim_separation": {
            "t4_minus_zero4": "descriptor-present versus direct neutral coordinate",
            "t4_minus_ts4": "attachment/content permutation mechanism",
            "cross_claim_substitution": "FORBIDDEN",
            "overall_three_arm_claim_requires_both_comparisons_all_views": True,
        },
        "score_protocol": {
            "support_trials": 50, "activity_identity_trials": 30,
            "query": "all valid 50-bin windows strictly after rewarded trial 50",
            "target": "last behavior bin of each query window",
            "behavior_scaling_factor_applied_once": 5.0,
            "r2": "variance_weighted", "session_drop_after_scoring": "FORBIDDEN",
        },
        "runtime": {
            "device": "cpu", "cuda_visible_devices": "", "deterministic_algorithms": True,
            "tf32": False, "autocast": False, "normalizer_fitting": False,
            "optimizer_or_backward": False, "target_updates": False,
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


@dataclass(frozen=True)
class AuthorizationGrant:
    schema: str
    status: str
    permitted_action: str
    authorization_sha256: str
    nonce: str
    output_root: Path
    contract_sha256: str
    cohort_sha256: str
    query_map_sha256: str
    checkpoint_slots_sha256: str
    preimport_authorization_complete: bool

    def binding(self) -> dict[str, Any]:
        return {
            "authorization_sha256": self.authorization_sha256,
            "nonce": self.nonce,
            "contract_sha256": self.contract_sha256,
        }


def validate_grant(grant: AuthorizationGrant) -> None:
    auth_require(isinstance(grant, AuthorizationGrant), "external verified grant required")
    auth_require(grant.schema == GRANT_SCHEMA and grant.status == GRANT_STATUS, "grant schema/status drift")
    auth_require(grant.permitted_action == GRANT_ACTION, "grant action drift")
    auth_require(is_sha256(grant.authorization_sha256) and is_sha256(grant.nonce), "grant identity malformed")
    auth_require(
        all(is_sha256(value) for value in (
            grant.contract_sha256, grant.cohort_sha256,
            grant.query_map_sha256, grant.checkpoint_slots_sha256,
        )),
        "grant digest binding malformed",
    )
    auth_require(grant.preimport_authorization_complete is True, "grant was not externally verified")
    auth_require(grant.output_root.is_absolute(), "grant output root must be absolute")


def validate_contract_against_grant(contract: Mapping[str, Any], grant: AuthorizationGrant, *, require_complete_slots: bool = True) -> None:
    validate_grant(grant)
    auth_require(isinstance(contract, Mapping) and set(contract) == CONTRACT_KEYS, "contract exact schema drift")
    declared = contract.get("contract_sha256")
    body = dict(contract); body.pop("contract_sha256", None)
    recomputed = canonical_sha256(body)
    auth_require(declared == recomputed, "contract declared/content hash mismatch")
    # The external grant is authoritative.  A caller cannot mutate a cohort and
    # bless it by recomputing the contract's own declared digest.
    auth_require(recomputed == grant.contract_sha256, "contract is not bound by external verified grant")
    cohort = validate_frozen_cohort(contract.get("cohort"))
    auth_require(canonical_sha256(cohort) == contract.get("cohort_sha256") == grant.cohort_sha256, "cohort grant binding drift")
    query_map = contract.get("query_window_count_by_asset_id")
    auth_require(isinstance(query_map, Mapping), "query map missing")
    auth_require(canonical_sha256(query_map) == contract.get("query_map_sha256") == grant.query_map_sha256, "query-map grant binding drift")
    slots = validate_checkpoint_slots(contract.get("checkpoint_slots"), require_complete=require_complete_slots)
    auth_require(canonical_sha256(slots) == contract.get("checkpoint_slots_sha256") == grant.checkpoint_slots_sha256, "checkpoint-slot grant binding drift")
    auth_require(
        contract.get("N") == 15 and contract.get("cell_count") == 270
        and contract.get("arms") == list(ARMS) and contract.get("views") == list(VIEWS)
        and contract.get("seeds") == list(SEEDS)
        and contract.get("comparison_gates") == COMPARISON_GATES
        and contract.get("bootstrap_policy") == BOOTSTRAP_POLICY,
        "fixed experiment contract drift",
    )


@dataclass(frozen=True, order=True)
class CellKey:
    session_id: str
    asset_id: str
    view: str
    seed: int
    arm: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_cell_keys(contract: Mapping[str, Any], grant: AuthorizationGrant) -> tuple[CellKey, ...]:
    validate_contract_against_grant(contract, grant)
    result = tuple(
        CellKey(str(row["session_id"]), str(row["asset_id"]), view, seed, arm)
        for row in contract["cohort"] for view in VIEWS for seed in SEEDS for arm in ARMS
    )
    auth_require(len(result) == len(set(result)) == EXPECTED_CELL_COUNT, "exact 270-cell map drift")
    return result


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_under(path: Path, root: Path, label: str) -> tuple[Path, Path]:
    candidate, base = _lexical_absolute(path), _lexical_absolute(root)
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ThreeArmV5LedgerError(f"{label} escapes grant output root") from exc
    return candidate, base


def _assert_no_symlink_chain(path: Path, root: Path, *, allow_missing_leaf: bool = False) -> None:
    candidate, base = _assert_under(path, root, "path")
    chain = [base]
    relative = candidate.relative_to(base)
    current = base
    for part in relative.parts:
        current = current / part; chain.append(current)
    for index, component in enumerate(chain):
        if not component.exists() and not component.is_symlink():
            if allow_missing_leaf and index == len(chain) - 1:
                return
            raise ThreeArmV5LedgerError(f"missing path component: {component}")
        try:
            mode = component.lstat().st_mode
        except OSError as exc:
            raise ThreeArmV5LedgerError(f"cannot lstat path component: {component}") from exc
        if stat.S_ISLNK(mode):
            raise ThreeArmV5LedgerError(f"symlink path component forbidden: {component}")


def _assert_absolute_existing_chain_no_symlink(path: Path) -> None:
    """Check every existing component from filesystem root through ``path``."""
    candidate = _lexical_absolute(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            raise ThreeArmV5LedgerError(f"missing absolute path component: {current}")
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ThreeArmV5LedgerError(f"symlink absolute path component forbidden: {current}")


def _ensure_output_root(grant: AuthorizationGrant) -> Path:
    validate_grant(grant)
    root = _lexical_absolute(grant.output_root)
    parent = root.parent
    if not parent.exists() or parent.is_symlink() or not parent.is_dir():
        raise ThreeArmV5LedgerError("grant output-root parent missing/unsafe")
    _assert_absolute_existing_chain_no_symlink(parent)
    if root.exists() or root.is_symlink():
        _assert_no_symlink_chain(root, root)
        ledger_require(root.is_dir(), "grant output root is not a directory")
    else:
        root.mkdir(mode=0o755)
        _assert_no_symlink_chain(root, root)
    ledger_require(_lexical_absolute(root) == _lexical_absolute(grant.output_root), "grant output-root binding drift")
    return root


def _ensure_safe_parent(path: Path, root: Path) -> None:
    candidate, base = _assert_under(path, root, "output")
    _assert_no_symlink_chain(base, base)
    current = base
    for part in candidate.parent.relative_to(base).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            _assert_no_symlink_chain(current, base)
            ledger_require(current.is_dir(), "non-directory output parent")
        else:
            current.mkdir(mode=0o755)
            _assert_no_symlink_chain(current, base)


def _write_exclusive(path: Path, raw: bytes, root: Path) -> dict[str, Any]:
    _ensure_safe_parent(path, root)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise ThreeArmV5LedgerError(f"duplicate output artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.chmod(path, 0o444, follow_symlinks=False)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    _assert_no_symlink_chain(path, root)
    metadata = path.lstat()
    ledger_require(stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444, "immutable regular output failed")
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": "0444"}


def _cell_path(root: Path, key: CellKey) -> Path:
    return root / "cells" / key.session_id / key.view / f"s{key.seed}" / f"{key.arm}.json"


def _artifact_path(root: Path, key: CellKey) -> Path:
    return root / "artifacts" / key.session_id / key.view / f"s{key.seed}" / f"{key.arm}.predictions_targets.npz"


def publish_prediction_target_artifact(
    grant: AuthorizationGrant, contract: Mapping[str, Any], *, key: CellKey, content: bytes,
) -> dict[str, Any]:
    """Synthetic/runtime sink for already-produced prediction/target bytes.

    The function does not parse or create predictions and does not calculate a
    statistic.  It only publishes exact caller bytes under the grant-bound path.
    """
    root = _ensure_output_root(grant)
    expected = set(expected_cell_keys(contract, grant))
    ledger_require(key in expected, "wrong arm/seed/view/cohort cell")
    ledger_require(isinstance(content, bytes) and content, "empty/non-byte prediction artifact")
    return _write_exclusive(_artifact_path(root, key), content, root)


def _validate_regular_pin(path: Path, root: Path, pin: Mapping[str, Any], label: str) -> None:
    ledger_require(isinstance(pin, Mapping) and set(pin) == ARTIFACT_KEYS, f"{label} pin schema drift")
    _assert_no_symlink_chain(path, root)
    metadata = path.lstat()
    ledger_require(stat.S_ISREG(metadata.st_mode), f"{label} is not regular")
    ledger_require(stat.S_IMODE(metadata.st_mode) == 0o444 == int(str(pin.get("mode")), 8), f"mutable {label}")
    ledger_require(_is_positive_int(pin.get("bytes")) and metadata.st_size == pin.get("bytes"), f"{label} byte-size drift")
    ledger_require(is_sha256(pin.get("sha256")) and sha256_file(path) == pin.get("sha256"), f"{label} digest drift")


def _validate_cell_payload(
    payload: Any, *, key: CellKey, contract: Mapping[str, Any], grant: AuthorizationGrant, root: Path,
) -> None:
    ledger_require(isinstance(payload, Mapping) and set(payload) == CELL_KEYS, "cell exact schema drift")
    ledger_require(payload.get("schema") == CELL_SCHEMA and payload.get("status") == "COMPLETE", "cell status/schema drift")
    ledger_require(payload.get("grant_binding") == grant.binding(), "cell authorization grant drift")
    ledger_require(isinstance(payload.get("cell"), Mapping) and set(payload["cell"]) == CELL_ID_KEYS and payload["cell"] == key.as_dict(), "cell identity drift")
    value = payload.get("r2")
    ledger_require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)), "cell R2 must be finite")
    expected_query = contract["query_window_count_by_asset_id"][key.asset_id]
    ledger_require(payload.get("query_window_count") == expected_query, "wrong query-window count")
    ledger_require(payload.get("contract_sha256") == grant.contract_sha256, "cell contract binding drift")
    pin = payload.get("prediction_target_artifact")
    expected_relative = str(_artifact_path(root, key).relative_to(root))
    ledger_require(isinstance(pin, Mapping) and pin.get("path") == expected_relative, "cell artifact path drift")
    _validate_regular_pin(root / expected_relative, root, pin, "prediction/target artifact")


def publish_cell_result(
    grant: AuthorizationGrant, contract: Mapping[str, Any], *, key: CellKey, r2: float,
    prediction_target_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    root = _ensure_output_root(grant)
    expected = set(expected_cell_keys(contract, grant))
    ledger_require(key in expected, "wrong arm/seed/view/cohort cell")
    artifact = dict(prediction_target_artifact)
    payload = {
        "schema": CELL_SCHEMA, "status": "COMPLETE", "grant_binding": grant.binding(),
        "cell": key.as_dict(), "r2": float(r2),
        "query_window_count": contract["query_window_count_by_asset_id"][key.asset_id],
        "prediction_target_artifact": artifact,
        "contract_sha256": grant.contract_sha256,
    }
    _validate_cell_payload(payload, key=key, contract=contract, grant=grant, root=root)
    return _write_exclusive(_cell_path(root, key), canonical_bytes(payload), root)


def _read_canonical_json(path: Path, root: Path, label: str) -> Mapping[str, Any]:
    _assert_no_symlink_chain(path, root)
    metadata = path.lstat()
    ledger_require(stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444, f"mutable/nonregular {label}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThreeArmV5LedgerError(f"malformed {label}") from exc
    ledger_require(isinstance(payload, Mapping) and raw == canonical_bytes(payload), f"noncanonical {label}")
    return payload


def scan_resume_state(grant: AuthorizationGrant, contract: Mapping[str, Any]) -> dict[str, Any]:
    root = _ensure_output_root(grant)
    keys = expected_cell_keys(contract, grant)
    expected_cells = {key: _cell_path(root, key) for key in keys}
    expected_artifacts = {key: _artifact_path(root, key) for key in keys}
    allowed_top = {"cells", "artifacts", "aggregate"}
    for child in root.iterdir():
        if child.is_symlink():
            raise ThreeArmV5LedgerError("symlink in output topology")
        if child.name not in allowed_top:
            raise ThreeArmV5LedgerError("partial/unknown output artifact present")
    observed_cells: set[Path] = set()
    cells_root = root / "cells"
    if cells_root.exists() or cells_root.is_symlink():
        _assert_no_symlink_chain(cells_root, root)
        allowed_cell_dirs = {
            _lexical_absolute(parent)
            for path in expected_cells.values()
            for parent in path.parents
            if parent == cells_root or cells_root in parent.parents
        }
        for path in cells_root.rglob("*"):
            if path.is_symlink():
                raise ThreeArmV5LedgerError("symlink in cell ledger")
            if path.is_file():
                observed_cells.add(_lexical_absolute(path))
            elif path.is_dir() and _lexical_absolute(path) not in allowed_cell_dirs:
                raise ThreeArmV5LedgerError("partial/unknown cell directory present")
    expected_cell_paths = {_lexical_absolute(path) for path in expected_cells.values()}
    ledger_require(not (observed_cells - expected_cell_paths), "partial/unknown cell artifact present")

    complete: list[CellKey] = []
    payloads: dict[CellKey, Mapping[str, Any]] = {}
    referenced_artifacts: set[Path] = set()
    for key, path in expected_cells.items():
        if not path.exists() and not path.is_symlink():
            continue
        payload = _read_canonical_json(path, root, "cell artifact")
        _validate_cell_payload(payload, key=key, contract=contract, grant=grant, root=root)
        complete.append(key); payloads[key] = payload
        referenced_artifacts.add(_lexical_absolute(root / payload["prediction_target_artifact"]["path"]))

    observed_artifacts: set[Path] = set()
    artifacts_root = root / "artifacts"
    if artifacts_root.exists() or artifacts_root.is_symlink():
        _assert_no_symlink_chain(artifacts_root, root)
        allowed_artifact_dirs = {
            _lexical_absolute(parent)
            for path in expected_artifacts.values()
            for parent in path.parents
            if parent == artifacts_root or artifacts_root in parent.parents
        }
        for path in artifacts_root.rglob("*"):
            if path.is_symlink():
                raise ThreeArmV5LedgerError("symlink in artifact ledger")
            if path.is_file():
                observed_artifacts.add(_lexical_absolute(path))
            elif path.is_dir() and _lexical_absolute(path) not in allowed_artifact_dirs:
                raise ThreeArmV5LedgerError("partial/unknown artifact directory present")
    ledger_require(observed_artifacts == referenced_artifacts, "partial/unknown prediction artifact present")
    ledger_require(observed_artifacts <= {_lexical_absolute(p) for p in expected_artifacts.values()}, "unknown prediction artifact present")

    aggregate_path = root / "aggregate" / "aggregate.json"
    aggregate_root = aggregate_path.parent
    if aggregate_root.exists() or aggregate_root.is_symlink():
        _assert_no_symlink_chain(aggregate_root, root)
        entries = list(aggregate_root.iterdir())
        ledger_require(entries == [aggregate_path], "partial/unknown aggregate artifact present")
        aggregate = _read_canonical_json(aggregate_path, root, "aggregate artifact")
        ledger_require(set(aggregate) == AGGREGATE_KEYS, "aggregate exact schema drift")
        ledger_require(
            aggregate.get("schema") == AGGREGATE_SCHEMA
            and aggregate.get("status") == "RECONSTRUCTED_FROM_VERIFIED_CELLS"
            and aggregate.get("grant_binding") == grant.binding()
            and aggregate.get("contract_sha256") == grant.contract_sha256
            and aggregate.get("statistics_source") == "reconstructed_from_270_verified_cells"
            and aggregate.get("verified_cell_count") == EXPECTED_CELL_COUNT
            and aggregate.get("bootstrap_policy") == BOOTSTRAP_POLICY
            and set(aggregate.get("comparisons", {})) == {name for name, _ in COMPARISONS},
            "aggregate semantic/grant binding drift",
        )
    complete_set = set(complete)
    missing = [key for key in keys if key not in complete_set]
    return {
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "complete_cell_count": len(complete), "missing_cell_count": len(missing),
        "complete": [key.as_dict() for key in complete],
        "missing": [key.as_dict() for key in missing],
        "_payloads": payloads,
    }


def _bootstrap_interval(delta: Any, rng: Any) -> tuple[float, float]:
    import numpy as np

    output = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    cursor = 0
    chunk_size = 10_000
    while cursor < BOOTSTRAP_REPLICATES:
        count = min(chunk_size, BOOTSTRAP_REPLICATES - cursor)
        session_index = rng.integers(0, EXPECTED_SESSION_COUNT, size=(count, EXPECTED_SESSION_COUNT))
        seed_index = rng.integers(0, len(SEEDS), size=(count, EXPECTED_SESSION_COUNT, len(SEEDS)))
        sampled = delta[session_index[:, :, None], seed_index]
        output[cursor:cursor + count] = sampled.mean(axis=(1, 2))
        cursor += count
    lower, upper = np.quantile(output, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def reconstruct_aggregate(grant: AuthorizationGrant, contract: Mapping[str, Any]) -> dict[str, Any]:
    state = scan_resume_state(grant, contract)
    ledger_require(state["complete_cell_count"] == EXPECTED_CELL_COUNT, "aggregate forbidden before all 270 cells")
    import numpy as np

    payloads: Mapping[CellKey, Mapping[str, Any]] = state["_payloads"]
    sessions = [(str(row["session_id"]), str(row["asset_id"])) for row in contract["cohort"]]
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    comparisons: dict[str, Any] = {}
    for comparison, comparator in COMPARISONS:
        views: dict[str, Any] = {}
        for view in VIEWS:
            t4 = np.asarray([
                [float(payloads[CellKey(session, asset, view, seed, "shared_t4")]["r2"]) for seed in SEEDS]
                for session, asset in sessions
            ], dtype=np.float64)
            other = np.asarray([
                [float(payloads[CellKey(session, asset, view, seed, comparator)]["r2"]) for seed in SEEDS]
                for session, asset in sessions
            ], dtype=np.float64)
            delta = t4 - other
            seed_means = delta.mean(axis=0)
            session_means = delta.mean(axis=1)
            grand = float(delta.mean())
            t4_seed_means = t4.mean(axis=0)
            lower, upper = _bootstrap_interval(delta, rng)
            positive_sessions = int((session_means > 0.0).sum())
            gates = {
                "grand_paired_mean_at_least_0p03": grand >= 0.03,
                "all_three_seed_means_strictly_positive": bool((seed_means > 0.0).all()),
                "at_least_12_of_15_session_means_positive": positive_sessions >= 12,
                "hierarchical_bootstrap_lower_95_strictly_positive": lower > 0.0,
                "shared_t4_absolute_grand_and_seed_means_strictly_positive": bool(t4.mean() > 0.0 and (t4_seed_means > 0.0).all()),
            }
            views[view] = {
                "grand_paired_mean_r2": grand,
                "seed_means_r2": {str(seed): float(seed_means[index]) for index, seed in enumerate(SEEDS)},
                "session_cross_seed_means_r2": {sessions[index][0]: float(value) for index, value in enumerate(session_means)},
                "positive_session_count": positive_sessions,
                "positive_session_required": 12,
                "hierarchical_bootstrap_95": {"lower": lower, "upper": upper},
                "shared_t4_absolute_grand_r2": float(t4.mean()),
                "shared_t4_absolute_seed_means_r2": {str(seed): float(t4_seed_means[index]) for index, seed in enumerate(SEEDS)},
                "gates": gates,
                "view_pass": all(gates.values()),
            }
        comparisons[comparison] = {
            "views": views,
            "comparison_pass": all(views[view]["view_pass"] for view in VIEWS),
            "cross_view_rescue_used": False,
        }
    return {
        "schema": AGGREGATE_SCHEMA,
        "status": "RECONSTRUCTED_FROM_VERIFIED_CELLS",
        "grant_binding": grant.binding(),
        "contract_sha256": grant.contract_sha256,
        "statistics_source": "reconstructed_from_270_verified_cells",
        "verified_cell_count": EXPECTED_CELL_COUNT,
        "bootstrap_policy": BOOTSTRAP_POLICY,
        "comparisons": comparisons,
        "overall_three_arm_claim_pass": all(comparisons[name]["comparison_pass"] for name, _ in COMPARISONS),
    }


def publish_full_aggregate(grant: AuthorizationGrant, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and publish aggregate; caller-supplied statistics are impossible."""
    root = _ensure_output_root(grant)
    payload = reconstruct_aggregate(grant, contract)
    return _write_exclusive(root / "aggregate" / "aggregate.json", canonical_bytes(payload), root)


def public_resume_state(grant: AuthorizationGrant, contract: Mapping[str, Any]) -> dict[str, Any]:
    state = scan_resume_state(grant, contract)
    state.pop("_payloads", None)
    return state


def refuse_blocked_execution() -> None:
    raise ThreeArmV5BlockedError("BLOCKED_MISSING_ZERO4_TERMINALS")
