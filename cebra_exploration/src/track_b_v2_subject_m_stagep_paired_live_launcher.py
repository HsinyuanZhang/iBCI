"""Fail-closed isolated launcher for the paired Subject-M Stage-P pilot.

The public path is a no-write dry plan.  A future authorised parent launches
one clean child for SUA and, only after its immutable terminal is validated,
one clean child for pMUA.  This module never treats the older 13-path live
executor list as complete: each view owns a parent-published launch contract
and the producer owns ``source_materialization.json``, so each cell has fifteen
immutable prospective output pairs plus one global paired completion pair.

No function in this module is authorised merely by its existence.  Real
execution additionally requires the immutable producer addendum, the separate
root-minted launcher preflight, dual CLI flags, ``ROOT_GO_AND_GPU_AUTHORIZED=1``
and an isolated conda child environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_development_target_materializer as target_materializer
import track_b_v2_subject_m_stagep_paired_real_producer as producer
import track_b_v2_subject_m_stagep_runtime as sealed_runtime


SCHEMA_DRY_PLAN = "track_b_v2_subject_m_stagep_paired_live_launcher_dry_v1"
SCHEMA_PREFLIGHT = "track_b_v2_subject_m_stagep_paired_live_launcher_preflight_v1"
SCHEMA_LAUNCH_CONTRACT = "track_b_v2_subject_m_stagep_child_launch_contract_v1"
SCHEMA_FAILURE_COMPLETION = "track_b_v2_subject_m_stagep_child_failure_completion_v1"
SCHEMA_FAILURE_TERMINAL = "track_b_v2_subject_m_stagep_parent_finalization_failure_terminal_v1"
STATUS_DRY = "NO_GO__LAUNCHER_PREFLIGHT_AND_PRODUCER_ADDENDUM_REQUIRED__NO_WRITE"
STATUS_PREFLIGHT = "ROOT_REVIEWED_LAUNCHER_PREFLIGHT__DOES_NOT_ALONE_AUTHORIZE_EXECUTION"

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_subject_m_stagep_paired_live_launcher.py"
TEST = REPO_ROOT / "cebra_exploration/tests/test_track_b_v2_subject_m_stagep_paired_live_launcher.py"
NOTE = REPO_ROOT / "cebra_exploration/docs/TRACK_B_V2_SUBJECT_M_STAGEP_PAIRED_LIVE_LAUNCHER.md"
CONDA_PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
RESULT_ROOT = producer.RESULT_ROOT
LAUNCHER_PREFLIGHT = RESULT_ROOT / "launcher_preflight.json"
PAIRED_COMPLETION = RESULT_ROOT / "aggregate/paired_completion.json"
COST_RECEIPT = (
    REPO_ROOT / "cebra_exploration/results"
    / "track_b_v2_fixed_gpu_cost_sua_firstfold_d8it250_s42_gpu1_v1/receipt.json"
)
COST_RECEIPT_SHA256 = "ec7096a5e54e444fd6cdafa241aaa88a0720143e3c4e42e3565f022ee662c8e2"
PAIR_ORDER = producer.PAIR_ORDER
ROOT_ENV = "ROOT_GO_AND_GPU_AUTHORIZED"
INTERNAL_TOKEN_ENV = "TRACK_B_STAGEP_INTERNAL_CHILD_TOKEN"
CHILD_VIEW_ENV = "TRACK_B_STAGEP_CHILD_VIEW"
FORBIDDEN_CHILD_ENV = ("PYTHONHOME", "PYTHONUSERBASE", "NVIDIA_VISIBLE_DEVICES",
                       "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
PRIVATE_SNAPSHOT_POLICY = {
    "during_parse": "owned_private_snapshot_inode_mode0600_then_verified_0444",
    "success": "unlink_only_after_target_encoder_embedding_and_six_score_hashes_are_published",
    "failure": "retain_verified_0444_for_forensics__cell_is_nonreusable_due_failure_completion",
    "published_as_result": False,
}


class StagePPairedLiveLauncherError(RuntimeError):
    """Raised before an unauthorised boundary or on exact-chain drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StagePPairedLiveLauncherError(message)


def _canonical_bytes(value: Any) -> bytes:
    return base.canonical_json_bytes(value)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical_bytes(value))


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _absolute(value: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(value).expanduser())))


def implementation_closure() -> dict[str, Any]:
    paths = {
        "launcher_core": Path(__file__), "launcher_cli": CLI, "launcher_tests": TEST,
        "launcher_note": NOTE, "paired_real_producer": Path(producer.__file__),
        "sealed_stagep_runtime": Path(sealed_runtime.__file__),
        "canonical_target_materializer": Path(target_materializer.__file__),
    }
    files = {role: producer._source_binding(path, label=f"launcher closure {role}")
             for role, path in paths.items()}
    return {"files": files, "closure_sha256": _sha_json(files)}


def _prospective_paths(view: str) -> tuple[Path, ...]:
    topology = producer._topology(view)
    launch_contract = Path(topology["cell_root"]) / "launch_contract.json"
    score_paths = topology["scores"]
    paths = (
        launch_contract, Path(topology["start"]), Path(topology["source"]), Path(topology["target"]),
        Path(topology["encoder"]), Path(topology["checkpoint"]), Path(topology["embeddings"]),
        *(Path(score_paths[f"{route}__{decoder}"])
          for route in producer.ROUTES for decoder in producer.DECODERS),
        Path(topology["completion"]), Path(topology["terminal"]),
    )
    require(len(paths) == 15 and len(set(paths)) == 15,
            f"{view} prospective topology must contain exact 15 distinct outputs including launch contract/source")
    return tuple(_absolute(path) for path in paths)


def canonical_topology() -> dict[str, Any]:
    per_view = {view: producer._topology(view) for view in PAIR_ORDER}
    all_paths = [path for view in PAIR_ORDER for path in _prospective_paths(view)] + [PAIRED_COMPLETION]
    require(len(all_paths) == 31 and len(set(map(_absolute, all_paths))) == 31,
            "paired output topology has alias/collision")
    return {
        "per_view": per_view, "prospective_pair_count_per_view": 15,
        "launch_contract_pair_is_mandatory_and_parent_published_before_spawn": True,
        "source_receipt_is_mandatory_and_independently_published": True,
        "global_paired_completion": str(PAIRED_COMPLETION),
        "total_prospective_pair_count": 31,
        "private_snapshot_policy": dict(PRIVATE_SNAPSHOT_POLICY),
    }


def _child_argv(view: str) -> list[str]:
    require(view in PAIR_ORDER, "child view invalid")
    return [str(CONDA_PYTHON), str(CLI), "--execute", "--i-have-authorization",
            "--internal-child", "--view", view]


def _child_env_contract(view: str) -> dict[str, str]:
    require(view in PAIR_ORDER, "child environment view invalid")
    return {
        "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "1",
        ROOT_ENV: "1", CHILD_VIEW_ENV: view,
    }


def _launch_contract_path(view: str) -> Path:
    require(view in PAIR_ORDER, "launch-contract view invalid")
    return Path(producer._topology(view)["cell_root"]) / "launch_contract.json"


def _target_authority_bindings(
        *, capabilities: Mapping[str, producer.ViewExecutionCapability]) -> dict[str, Any]:
    """Resolve only canonical no-array A2-ledger plans after output freshness."""
    bindings: dict[str, Any] = {}
    for view in PAIR_ORDER:
        plan = target_materializer.build_development_target_materializer_dry_plan(
            dataset="subject_m", view=view,
            outer_fold_id=str(capabilities[view].cell["outer_fold_id"]),
            target_session_id=producer.TARGET_SESSION_ID)
        require(plan.get("status") ==
                "CANONICAL_DEVELOPMENT_AUTHORITY_AND_SUBM_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS" and
                plan.get("dataset") == "subject_m" and plan.get("view") == view and
                plan.get("target_session_id") == producer.TARGET_SESSION_ID and
                plan.get("target_data_opened") is False,
                f"{view} A2-ledger target materializer authority drift")
        bindings[view] = plan
    return bindings


def build_dry_plan() -> dict[str, Any]:
    """Static no-write/no-data plan; it does not load either required preflight."""
    closure = implementation_closure()
    return {
        "schema": SCHEMA_DRY_PLAN, "status": STATUS_DRY,
        "pair_order": list(PAIR_ORDER), "topology": canonical_topology(),
        "isolated_child": {
            "python": str(CONDA_PYTHON), "cwd": str(REPO_ROOT),
            "argv_by_view": {view: _child_argv(view) for view in PAIR_ORDER},
            "environment_by_view": {view: _child_env_contract(view) for view in PAIR_ORDER},
            "parent_interpreter_direct_scientific_execution_permitted": False,
            "one_independent_process_per_view": True,
        },
        "ordered_state_rule": "SUA_terminal_verified_before_pMUA_subprocess_start",
        "producer_addendum": {"path": str(producer.ADDENDUM_PATH),
                              "present": os.path.lexists(producer.ADDENDUM_PATH)},
        "launcher_preflight": {"path": str(LAUNCHER_PREFLIGHT),
                               "present": os.path.lexists(LAUNCHER_PREFLIGHT)},
        "launcher_implementation_closure": closure,
        "execute_enabled": False, "subprocess_started": False, "target_path_resolved": False,
        "target_opened": False, "formal_data_opened": False, "torch_imported": False,
        "cebra_imported": False, "gpu_queried": False, "write_performed": False,
    }


def _read_pair(path: Path, *, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    body = producer._read_same_fd(path, label=f"{label} body", required_mode=0o444)
    side = producer._read_same_fd(Path(f"{path}.sha256"), label=f"{label} sidecar", required_mode=0o444)
    require(side.raw == f"{body.sha256}  {path.name}\n".encode("ascii") and
            (expected_sha256 is None or body.sha256 == expected_sha256), f"{label} pair drift")
    payload = producer._json_from_verified(body, label=label)
    return {"payload": payload, "body_sha256": body.sha256, "sidecar_sha256": side.sha256,
            "path": str(body.path)}


def _cost_identity() -> dict[str, Any]:
    receipt = _read_pair(COST_RECEIPT, label="fixed GPU cost", expected_sha256=COST_RECEIPT_SHA256)
    identity = receipt["payload"].get("cuda_identity")
    smi = identity.get("nvidia_smi_identity") if isinstance(identity, Mapping) else None
    require(isinstance(identity, Mapping) and isinstance(smi, Mapping) and
            identity.get("cuda_visible_devices") == "1" and
            identity.get("logical_device") == "cuda:0" and
            smi.get("physical_device_index") == "1" and _valid_sha(identity.get("torch_sha256")),
            "fixed-cost CUDA identity drift")
    return {
        "cost_body_sha256": receipt["body_sha256"], "physical_index": 1,
        "physical_uuid": smi["device_uuid"], "physical_pci_bus_id": smi["pci_bus_id"],
        "driver_version": smi["driver_version"], "device_name": smi["device_name"],
        "total_memory_mib": smi["total_memory_mib"], "torch_version": identity["torch_version"],
        "torch_cuda_runtime": identity["torch_cuda_version"], "torch_path": identity["torch_path"],
        "torch_sha256": identity["torch_sha256"],
    }


def _fresh_admissions() -> dict[str, Mapping[str, Any]]:
    return {view: sealed_runtime.build_stagep_live_admission(view=view) for view in PAIR_ORDER}


def _launcher_preflight_payload(
        *, admissions: Mapping[str, Mapping[str, Any]],
        capabilities: Mapping[str, producer.ViewExecutionCapability],
        plan: Mapping[str, Any], addendum: Mapping[str, Any],
        target_authority_bindings: Mapping[str, Any]) -> dict[str, Any]:
    closure = implementation_closure()
    payload = {
        "schema": SCHEMA_PREFLIGHT, "status": STATUS_PREFLIGHT,
        "canonical_path": str(LAUNCHER_PREFLIGHT), "pair_order": list(PAIR_ORDER),
        "canonical_topology": canonical_topology(),
        "producer_addendum_body_sha256": addendum["body_sha256"],
        "producer_closure_sha256": plan["implementation_closure"]["closure_sha256"],
        "launcher_implementation_closure": closure,
        "admission_sha256_by_view": {view: capabilities[view].admission_sha256 for view in PAIR_ORDER},
        "official_preflight_body_sha256_by_view": {
            view: capabilities[view].official_preflight_body_sha256 for view in PAIR_ORDER},
        "cost_bound_cuda_identity": _cost_identity(),
        "target_authority_bindings_by_view": dict(target_authority_bindings),
        "v9_preflight_sha256": producer.V9_PREFLIGHT_SHA256,
        "child_argv_by_view": {view: _child_argv(view) for view in PAIR_ORDER},
        "child_environment_by_view": {view: _child_env_contract(view) for view in PAIR_ORDER},
        "root_environment_tripwire": f"{ROOT_ENV}=1",
        "authorizes_target_or_GPU_by_itself": False, "target_opened": False,
        "formal_data_opened": False, "torch_imported": False, "cebra_imported": False,
    }
    return payload | {"launcher_preflight_payload_sha256": _sha_json(payload)}


def build_launcher_preflight_candidate() -> dict[str, Any]:
    """Root-only candidate; requires the already-minted exact producer addendum."""
    admissions = _fresh_admissions()
    plan = producer.build_no_target_review_plan()
    capabilities = producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
    addendum = producer.load_execution_addendum(admissions=admissions)
    target_bindings = _target_authority_bindings(capabilities=capabilities)
    return _launcher_preflight_payload(
        admissions=admissions, capabilities=capabilities, plan=plan, addendum=addendum,
        target_authority_bindings=target_bindings)


def publish_launcher_preflight(*, i_have_independent_root_review: bool = False) -> dict[str, Any]:
    """Canonical root-only publisher; never exposed by the public CLI."""
    require(i_have_independent_root_review is True,
            "launcher preflight mint requires independent root review")
    require(not os.path.lexists(LAUNCHER_PREFLIGHT) and
            not os.path.lexists(Path(f"{LAUNCHER_PREFLIGHT}.sha256")),
            "launcher preflight body/sidecar must both be fresh")
    _assert_real_parent_chain(LAUNCHER_PREFLIGHT.parent)
    payload = build_launcher_preflight_candidate()
    binding = producer.publish_json_pair(LAUNCHER_PREFLIGHT, payload)
    try:
        require(build_launcher_preflight_candidate() == payload,
                "launcher preflight launch/final closure drift")
        loaded = load_launcher_preflight()
        require(loaded["payload"] == payload, "launcher preflight reload drift")
        return binding
    except BaseException:
        producer._rollback_owned_raw_pair(binding)
        raise


def load_launcher_preflight(*, validate_target_bindings: bool = True) -> dict[str, Any]:
    loaded = _read_pair(LAUNCHER_PREFLIGHT, label="launcher preflight")
    if validate_target_bindings:
        expected = build_launcher_preflight_candidate()
    else:
        admissions = _fresh_admissions()
        plan = producer.build_no_target_review_plan()
        capabilities = producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
        addendum = producer.load_execution_addendum(admissions=admissions)
        frozen = loaded["payload"].get("target_authority_bindings_by_view")
        require(isinstance(frozen, Mapping) and set(frozen) == set(PAIR_ORDER),
                "launcher preflight target authority binding set drift")
        expected = _launcher_preflight_payload(
            admissions=admissions, capabilities=capabilities, plan=plan, addendum=addendum,
            target_authority_bindings=frozen)
    require(loaded["payload"] == expected and
            loaded["payload"].get("canonical_path") == str(LAUNCHER_PREFLIGHT),
            "launcher preflight differs from current exact closure/admissions")
    return loaded


def _validate_real_existing_chain(path: Path) -> None:
    """Reject every symlink/non-directory component without creating anything."""
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if not os.path.lexists(current):
            break
        info = current.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                f"existing ancestor is not a real directory: {current}")


def _assert_real_parent_chain(path: Path) -> None:
    _validate_real_existing_chain(path)
    current = _absolute(path)
    pending: list[Path] = []
    while not current.exists():
        pending.append(current)
        require(current != current.parent, "cannot prepare filesystem root")
        current = current.parent
    info = current.lstat()
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            f"existing ancestor is not a real directory: {current}")
    for directory in reversed(pending):
        directory.mkdir()
        info = directory.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                f"created parent is not a real directory: {directory}")


def require_all_outputs_fresh(*, only_view: str | None = None,
                              existing_launch_contract: Mapping[str, Any] | None = None) -> None:
    """Exact 15×2+1 freshness gate, optionally admitting one verified parent contract."""
    require(only_view is None or only_view in PAIR_ORDER, "freshness view scope invalid")
    seen: set[Path] = set()
    views = PAIR_ORDER if only_view is None else (only_view,)
    for view in views:
        for path in _prospective_paths(view):
            _validate_real_existing_chain(path.parent)
            require(path not in seen, "prospective output path collision")
            seen.add(path)
            if existing_launch_contract is not None and path == _absolute(_launch_contract_path(view)):
                require(existing_launch_contract.get("path") == str(path) and
                        _valid_sha(existing_launch_contract.get("body_sha256")),
                        "existing launch contract binding drift")
            else:
                require(not os.path.lexists(path) and not os.path.lexists(Path(f"{path}.sha256")),
                        f"prospective output body/sidecar not fresh: {path}")
        snapshot = _absolute(producer._topology(view)["private_snapshot"])
        _validate_real_existing_chain(snapshot.parent)
        require(not os.path.lexists(snapshot), f"private snapshot not fresh: {view}")
    paired = _absolute(PAIRED_COMPLETION)
    _validate_real_existing_chain(paired.parent)
    require(paired not in seen and not os.path.lexists(paired) and
            not os.path.lexists(Path(f"{paired}.sha256")), "paired completion not fresh")


@dataclass(frozen=True)
class ExecutionPrerequisites:
    admissions: Mapping[str, Mapping[str, Any]]
    capabilities: Mapping[str, producer.ViewExecutionCapability]
    producer_addendum: Mapping[str, Any]
    launcher_preflight: Mapping[str, Any]
    cost_identity: Mapping[str, Any]
    target_authority_bindings: Mapping[str, Any]


def validate_pretarget_prerequisites(
        *, only_view: str | None = None,
        existing_launch_contract: Mapping[str, Any] | None = None) -> ExecutionPrerequisites:
    """Validate every authority and freshness gate before target resolution."""
    admissions = _fresh_admissions()
    reviewed_plan = producer.build_no_target_review_plan()
    capabilities = producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=reviewed_plan)
    addendum = producer.load_execution_addendum(admissions=admissions)
    preflight = load_launcher_preflight(validate_target_bindings=False)
    require(preflight["payload"]["producer_addendum_body_sha256"] == addendum["body_sha256"],
            "launcher preflight/producer addendum binding drift")
    cost = _cost_identity()
    require(cost == preflight["payload"]["cost_bound_cuda_identity"], "cost identity/preflight drift")
    # The complete 15×view output set (including launch contract/source) is checked before
    # the A2-ledger materializer is allowed to resolve its canonical target
    # asset path.
    require_all_outputs_fresh(only_view=only_view, existing_launch_contract=existing_launch_contract)
    target_bindings = _target_authority_bindings(capabilities=capabilities)
    require(target_bindings == preflight["payload"]["target_authority_bindings_by_view"],
            "launcher preflight/live A2 target authority binding drift")
    return ExecutionPrerequisites(admissions, capabilities, addendum, preflight, cost,
                                  target_bindings)


def _launch_contract_payload(*, view: str, token: str,
                             prerequisites: ExecutionPrerequisites,
                             parent_pid: int | None = None) -> dict[str, Any]:
    require(view in PAIR_ORDER and token, "launch contract token/view invalid")
    preflight = prerequisites.launcher_preflight
    payload = {
        "schema": SCHEMA_LAUNCH_CONTRACT,
        "status": "PARENT_PRESPAWN_CONTRACT__SUBPROCESS_NOT_YET_STARTED",
        "canonical_path": str(_launch_contract_path(view)),
        "cell": dict(prerequisites.capabilities[view].cell),
        "view": view, "parent_pid": os.getpid() if parent_pid is None else int(parent_pid),
        "token_sha256": _sha_bytes(token.encode("utf-8")),
        "actual_argv_required": _child_argv(view),
        "actual_environment_required": _child_env_contract(view),
        "actual_cwd_required": str(REPO_ROOT),
        "actual_python_required": str(CONDA_PYTHON),
        "launcher_preflight_body_sha256": preflight["body_sha256"],
        "launcher_preflight_payload_sha256": preflight["payload"][
            "launcher_preflight_payload_sha256"],
        "launcher_closure_sha256": preflight["payload"][
            "launcher_implementation_closure"]["closure_sha256"],
        "producer_addendum_body_sha256": prerequisites.producer_addendum["body_sha256"],
        "admission_sha256": prerequisites.capabilities[view].admission_sha256,
        "cost_bound_cuda_identity": dict(prerequisites.cost_identity),
        "subprocess_started": False, "actual_started": False,
        "target_path_resolved": False, "target_opened": False,
        "formal_data_opened": False, "token_plaintext_persisted": False,
    }
    return payload | {"launch_contract_payload_sha256": _sha_json(payload)}


def _publish_launch_contract(*, view: str, token: str,
                             prerequisites: ExecutionPrerequisites) -> dict[str, Any]:
    path = _launch_contract_path(view)
    require(not os.path.lexists(path) and not os.path.lexists(Path(f"{path}.sha256")),
            f"{view} launch contract pair must be fresh")
    return _publish(path, _launch_contract_payload(
        view=view, token=token, prerequisites=prerequisites))


def _load_child_launch_contract(*, view: str, token: str) -> dict[str, Any]:
    loaded = _load_output(_launch_contract_path(view), schema=SCHEMA_LAUNCH_CONTRACT,
                          label=f"{view} parent launch contract")
    payload = loaded["payload"]
    require(payload.get("canonical_path") == str(_launch_contract_path(view)) and
            payload.get("cell") == sealed_runtime.StagePCell.from_view(view).as_dict() and
            payload.get("view") == view and payload.get("parent_pid") == os.getppid() and
            payload.get("token_sha256") == _sha_bytes(token.encode("utf-8")) and
            payload.get("token_plaintext_persisted") is False and
            payload.get("subprocess_started") is False and payload.get("actual_started") is False and
            payload.get("launch_contract_payload_sha256") == _sha_json({
                key: value for key, value in payload.items()
                if key != "launch_contract_payload_sha256"}),
            "internal child launch contract/token/parent drift")
    return loaded


def _verify_child_environment(*, view: str, token: str,
                              launch_contract: Mapping[str, Any]) -> None:
    require(view in PAIR_ORDER and token and os.environ.get(INTERNAL_TOKEN_ENV) == token,
            "internal child token/view drift")
    expected = _child_env_contract(view)
    require(all(os.environ.get(key) == value for key, value in expected.items()) and
            all(key not in os.environ for key in FORBIDDEN_CHILD_ENV) and
            Path(sys.executable).resolve() == CONDA_PYTHON.resolve() and
            Path.cwd().resolve() == REPO_ROOT.resolve(), "isolated child environment drift")
    payload = launch_contract.get("payload")
    require(isinstance(payload, Mapping) and
            payload.get("actual_argv_required") == _child_argv(view) and
            payload.get("actual_environment_required") == expected and
            Path(str(payload.get("actual_cwd_required"))).resolve() == REPO_ROOT.resolve() and
            Path(str(payload.get("actual_python_required"))).resolve() == CONDA_PYTHON.resolve() and
            [sys.executable, *sys.argv] == _child_argv(view),
            "actual child argv/env/CWD/python differs from parent contract")


def _verify_gpu_against_cost(cost: Mapping[str, Any]) -> dict[str, Any]:
    observed = producer.verify_isolated_cuda_identity()
    torch_file = producer._read_same_fd(
        Path(str(observed.get("torch_module_path"))), label="live isolated Torch module", required_mode=None)
    require(observed.get("physical_uuid") == cost.get("physical_uuid") and
            observed.get("physical_pci_bus_id") == cost.get("physical_pci_bus_id") and
            observed.get("driver_version") == cost.get("driver_version") and
            observed.get("name") == cost.get("device_name") and
            observed.get("torch_version") == cost.get("torch_version") and
            observed.get("torch_cuda_runtime") == cost.get("torch_cuda_runtime") and
            Path(str(observed.get("torch_module_path"))).resolve() == Path(str(cost.get("torch_path"))).resolve() and
            torch_file.sha256 == cost.get("torch_sha256"),
            "live isolated GPU/Torch identity differs from fixed-cost receipt")
    return dict(observed) | {"torch_module_sha256": torch_file.sha256,
                             "cost_receipt_body_sha256": cost["cost_body_sha256"]}


def _publish(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _assert_real_parent_chain(path.parent)
    return producer.publish_json_pair(path, payload)


def _start_payload(*, capability: producer.ViewExecutionCapability, prerequisites: ExecutionPrerequisites,
                   gpu_identity: Mapping[str, Any], launch_contract: Mapping[str, Any]) -> dict[str, Any]:
    base_payload = producer.build_start_payload(
        capability=capability, addendum_sha256=str(prerequisites.producer_addendum["body_sha256"]),
        live_closure=producer.implementation_closure())
    payload = dict(base_payload)
    payload.pop("start_payload_sha256")
    payload.update({
        "launcher_preflight_body_sha256": prerequisites.launcher_preflight["body_sha256"],
        "launcher_preflight_payload_sha256": prerequisites.launcher_preflight["payload"][
            "launcher_preflight_payload_sha256"],
        "launcher_closure_sha256": prerequisites.launcher_preflight["payload"][
            "launcher_implementation_closure"]["closure_sha256"],
        "parent_launch_contract_path": launch_contract["path"],
        "parent_launch_contract_body_sha256": launch_contract["body_sha256"],
        "parent_launch_contract_payload_sha256": launch_contract["payload"][
            "launch_contract_payload_sha256"],
        "subprocess_started": True, "actual_started": True, "child_pid": os.getpid(),
        "parent_pid": os.getppid(), "actual_argv": [sys.executable, *sys.argv],
        "actual_cwd": str(Path.cwd()),
        "actual_python": sys.executable,
        "actual_environment": {key: os.environ.get(key) for key in
                               ("PYTHONNOUSERSITE", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                                ROOT_ENV, CHILD_VIEW_ENV)},
        "cost_bound_gpu_identity": dict(gpu_identity),
        "canonical_paths": producer._topology(capability.view),
        "private_snapshot_policy": dict(PRIVATE_SNAPSHOT_POLICY),
    })
    require(payload["actual_argv"] == _child_argv(capability.view) and
            payload["actual_environment"] == _child_env_contract(capability.view) and
            Path(payload["actual_cwd"]).resolve() == REPO_ROOT.resolve() and
            Path(payload["actual_python"]).resolve() == CONDA_PYTHON.resolve(),
            "start actual argv/env/CWD/python contract drift")
    return payload | {"start_payload_sha256": _sha_json(payload)}


def _load_output(path: Path, *, schema: str, label: str) -> dict[str, Any]:
    loaded = _read_pair(path, label=label)
    require(loaded["payload"].get("schema") == schema, f"{label} schema drift")
    return loaded


def _load_raw_output(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    body = producer._read_same_fd(path, label=f"{label} body", required_mode=0o444)
    side = producer._read_same_fd(Path(f"{path}.sha256"), label=f"{label} sidecar", required_mode=0o444)
    require(body.sha256 == expected_sha256 and
            side.raw == f"{body.sha256}  {path.name}\n".encode("ascii"), f"{label} raw pair drift")
    return {"path": str(body.path), "body_sha256": body.sha256,
            "sidecar_sha256": side.sha256, "mode": "0444"}


def _require_self_sha(payload: Mapping[str, Any], *, key: str, label: str) -> None:
    bare = dict(payload)
    declared = bare.pop(key, None)
    require(_valid_sha(declared) and declared == _sha_json(bare), f"{label} self SHA drift")


def _validate_start_chain(*, view: str, start: Mapping[str, Any],
                          capability: producer.ViewExecutionCapability,
                          preflight: Mapping[str, Any], addendum: Mapping[str, Any],
                          launch_contract: Mapping[str, Any]) -> None:
    payload = start["payload"]
    frozen = preflight["payload"]
    expected_env = _child_env_contract(view)
    _require_self_sha(payload, key="start_payload_sha256", label=f"{view} start")
    require(payload.get("cell") == capability.cell and
            payload.get("admission_sha256") == capability.admission_sha256 and
            payload.get("official_preflight_body_sha256") == capability.official_preflight_body_sha256 and
            payload.get("execution_addendum_body_sha256") == addendum["body_sha256"] and
            payload.get("producer_implementation_closure") == producer.implementation_closure() and
            payload.get("launcher_preflight_body_sha256") == preflight["body_sha256"] and
            payload.get("launcher_preflight_payload_sha256") ==
            frozen["launcher_preflight_payload_sha256"] and
            payload.get("launcher_closure_sha256") ==
            frozen["launcher_implementation_closure"]["closure_sha256"] and
            payload.get("parent_launch_contract_path") == launch_contract["path"] and
            payload.get("parent_launch_contract_body_sha256") == launch_contract["body_sha256"] and
            payload.get("parent_launch_contract_payload_sha256") ==
            launch_contract["payload"]["launch_contract_payload_sha256"] and
            payload.get("actual_argv") == _child_argv(view) and
            payload.get("actual_environment") == expected_env and
            Path(str(payload.get("actual_cwd"))).resolve() == REPO_ROOT.resolve() and
            Path(str(payload.get("actual_python"))).resolve() == CONDA_PYTHON.resolve() and
            payload.get("parent_pid") == launch_contract["payload"].get("parent_pid") and
            payload.get("subprocess_started") is True and payload.get("actual_started") is True and
            payload.get("target_opened") is False,
            f"{view} start/capability/preflight/actual child contract drift")
    gpu = payload.get("cost_bound_gpu_identity")
    cost = frozen["cost_bound_cuda_identity"]
    require(isinstance(gpu, Mapping) and
            gpu.get("cost_receipt_body_sha256") == cost["cost_body_sha256"] and
            gpu.get("physical_uuid") == cost["physical_uuid"] and
            gpu.get("physical_pci_bus_id") == cost["physical_pci_bus_id"] and
            gpu.get("driver_version") == cost["driver_version"] and
            gpu.get("name") == cost["device_name"] and
            gpu.get("torch_version") == cost["torch_version"] and
            gpu.get("torch_cuda_runtime") == cost["torch_cuda_runtime"] and
            gpu.get("torch_module_sha256") == cost["torch_sha256"],
            f"{view} start GPU identity/cost binding drift")


def _validate_scientific_output_chain(
        *, view: str, capability: producer.ViewExecutionCapability,
        start: Mapping[str, Any], source: Mapping[str, Any], target: Mapping[str, Any],
        encoder: Mapping[str, Any], scores: Mapping[str, Mapping[str, Any]]) -> None:
    cell = capability.cell
    sp, tp, ep = source["payload"], target["payload"], encoder["payload"]
    for payload, key, label in (
            (sp, "source_payload_sha256", "source"),
            (tp, "target_payload_sha256", "target"),
            (ep, "encoder_payload_sha256", "encoder")):
        _require_self_sha(payload, key=key, label=f"{view} {label}")
    require(sp.get("status") == "STRICT27_SOURCE_MATERIALIZED_AND_EXACT_AUTHORITY_MATCH" and
            sp.get("cell") == cell and sp.get("admission_sha256") == capability.admission_sha256 and
            sp.get("source_authority_set_sha256") == capability.source_authority_set_sha256 and
            sp.get("source_session_count") == 27 and
            len(sp.get("ordered_source_session_ids", ())) == 27 and
            sp.get("source_authority_exact_rebuild_match") is True and
            sp.get("historical_selector_plan_executed_or_selected") is False and
            sp.get("target_opened") is False and sp.get("query_opened") is False,
            f"{view} source capability/admission/authority semantics drift")
    query = tp.get("query")
    snapshot = tp.get("private_snapshot")
    require(tp.get("status") == "TARGET_M50_AND_SPARSE_V9_QUERY_MATERIALIZED" and
            tp.get("cell") == cell and
            tp.get("start_payload_sha256") == start["payload"]["start_payload_sha256"] and
            isinstance(query, Mapping) and
            query.get("semantics") == "SPARSE_EXACT_V9_ENDPOINT_GATHER_FROM_CONTINUOUS_SUFFIX_TRANSFORM" and
            query.get("query_row_count") == producer.V9_QUERY_COUNT and
            query.get("valid_starts_int64_sha256") == producer.V9_VALID_STARTS_SHA256 and
            query.get("ordered_target_behavior_float32_sha256") == producer.V9_TARGET_SHA256 and
            query.get("sealed_v9_preflight_sha256") == producer.V9_PREFLIGHT_SHA256 and
            _valid_sha(query.get("ordered_prediction_endpoint_int64_sha256")) and
            _valid_sha(query.get("ordered_offset10_RF_int64_sha256")) and
            query.get("not_contiguous_5_to_minus5_crop") is True and
            query.get("each_RF_is_range_endpoint_minus5_to_endpoint_plus5_exclusive") is True and
            query.get("every_RF_wholly_inside_held_suffix") is True and
            query.get("every_RF_support_disjoint") is True and
            query.get("query_neural_or_auxiliary_enters_any_fit") is False and
            isinstance(snapshot, Mapping) and
            snapshot.get("parser_consumed_continuously_held_fd") is True and
            snapshot.get("pathname_reopen_permitted") is False and
            tp.get("source_only_behavior_normalizer_used") is True and
            tp.get("target_query_neural_or_auxiliary_entered_fit") is False,
            f"{view} target start/cell/V9/snapshot semantics drift")
    fit = ep.get("fit_proof")
    require(ep.get("status") == "JOINT_ENCODER_PERSISTED_AND_RELOADED_EXACT" and
            ep.get("cell") == cell and
            ep.get("start_payload_sha256") == start["payload"]["start_payload_sha256"] and
            ep.get("source_payload_sha256") == sp["source_payload_sha256"] and
            ep.get("source_receipt_body_sha256") == source["body_sha256"] and
            ep.get("target_payload_sha256") == tp["target_payload_sha256"] and
            isinstance(fit, Mapping) and fit.get("fit_stream_count") == 28 and
            fit.get("fit_call_count") == 1 and fit.get("source_session_count") == 27 and
            fit.get("target_support_session_count") == 1 and
            fit.get("target_query_enters_fit") is False and
            fit.get("model_contract") == producer.MODEL_CONTRACT and
            ep.get("same_encoder_services_all_six_readouts") is True and
            ep.get("cross_view_encoder_reuse") is False,
            f"{view} encoder upstream/cell/fit semantics drift")
    expected_roles = {f"{route}__{decoder}" for route in producer.ROUTES for decoder in producer.DECODERS}
    require(set(scores) == expected_roles, f"{view} score role set drift")
    for role, loaded in scores.items():
        payload = loaded["payload"]
        route, decoder = role.split("__", 1)
        _require_self_sha(payload, key="score_payload_sha256", label=f"{view} score {role}")
        readout = payload.get("readout_proof")
        metric = payload.get("metric")
        require(payload.get("status") == "ROUTE_DECODER_SCORE_COMPLETE" and
                payload.get("cell") == cell and payload.get("readout_route") == route and
                payload.get("decoder") == decoder and
                payload.get("target_payload_sha256") == tp["target_payload_sha256"] and
                payload.get("encoder_payload_sha256") == ep["encoder_payload_sha256"] and
                payload.get("query_row_count") == producer.V9_QUERY_COUNT and
                payload.get("target_float32_sha256") == producer.V9_TARGET_SHA256 and
                isinstance(readout, Mapping) and readout.get("query_enters_fit") is False and
                readout.get("query_block_receipt_sha256") == _sha_json(query) and
                readout.get("sparse_query_semantics") ==
                "exact_V9_endpoint_gather__not_contiguous_crop" and
                isinstance(metric, Mapping) and
                metric.get("implementation") == "torchmetrics.regression.R2Score" and
                metric.get("version") == "1.5.1" and metric.get("dtype") == "float32" and
                metric.get("device") == "cpu" and metric.get("multioutput") == "variance_weighted" and
                metric.get("update_scope") ==
                "one_complete_ordered_external_target_session_query_then_compute_once" and
                metric.get("update_call_count") == 1 and metric.get("compute_call_count") == 1 and
                metric.get("target_float32_bytes_sha256") == producer.V9_TARGET_SHA256 and
                metric.get("custom_numpy_float64_pooled_r2_used") is False,
                f"{view} score role/upstream/query/TorchMetrics semantics drift: {role}")


def _cleanup_private_snapshot_after_success(*, view: str,
                                            target: producer.TargetProducerOutput) -> None:
    """Remove only the verified canonical snapshot after all scientific hashes exist."""
    path = _absolute(producer._topology(view)["private_snapshot"])
    contract = target.payload_inputs.get("private_snapshot")
    require(isinstance(contract, Mapping) and
            _absolute(str(contract.get("private_snapshot_path"))) == path and
            _valid_sha(contract.get("private_snapshot_sha256")),
            "private snapshot cleanup contract drift")
    verified = producer._read_same_fd(path, label=f"{view} successful private snapshot cleanup",
                                      required_mode=0o444)
    require(verified.sha256 == contract["private_snapshot_sha256"],
            "private snapshot cleanup bytes differ from parser-held inode")
    named = path.lstat()
    require((named.st_dev, named.st_ino) == (verified.device, verified.inode) and
            stat.S_ISREG(named.st_mode) and not stat.S_ISLNK(named.st_mode),
            "private snapshot cleanup pathname identity drift")
    path.unlink()
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                        getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _require_sua_terminal_before_pmua(*, launcher_preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Same-FD predecessor gate used by the pMUA child before GPU/target access."""
    topology = producer._topology("sua")
    source = _load_output(Path(topology["source"]), schema=producer.SCHEMA_SOURCE,
                          label="SUA predecessor source")
    completion = _load_output(Path(topology["completion"]), schema=producer.SCHEMA_COMPLETION,
                              label="SUA predecessor completion")
    terminal = _load_output(Path(topology["terminal"]), schema=producer.SCHEMA_TERMINAL,
                            label="SUA predecessor terminal")
    payload = terminal["payload"]
    _require_self_sha(source["payload"], key="source_payload_sha256",
                      label="SUA predecessor source")
    _require_self_sha(completion["payload"], key="completion_payload_sha256",
                      label="SUA predecessor completion")
    _require_self_sha(payload, key="terminal_payload_sha256", label="SUA predecessor terminal")
    source_binding = completion["payload"].get("immutable_output_bindings", {}).get("source", {})
    require(payload.get("status") == "TERMINAL_SUCCESS__DEVELOPMENT_PILOT_CELL" and
            payload.get("cell") == sealed_runtime.StagePCell.from_view("sua").as_dict() and
            source["payload"].get("cell") == payload.get("cell") and
            completion["payload"].get("cell") == payload.get("cell") and
            source_binding.get("body_sha256") == source["body_sha256"] and
            completion["payload"].get("source_receipt_body_sha256") == source["body_sha256"] and
            payload.get("completion_payload_sha256") ==
            completion["payload"].get("completion_payload_sha256") and
            payload.get("completion_receipt_body_sha256") == completion["body_sha256"] and
            payload.get("source_receipt_body_sha256") == source["body_sha256"] and
            payload.get("launcher_preflight_body_sha256") == launcher_preflight["body_sha256"] and
            payload.get("launcher_launch_final_live_exact_equal") is True and
            payload.get("producer_launch_final_live_exact_equal") is True and
            payload.get("actual_exit_code") == 0 and
            payload.get("formal_data_opened") is False and
            payload.get("target_query_updates") == 0,
            "pMUA start forbidden before exact immutable SUA source/completion/terminal chain")
    return terminal


def execute_internal_child(*, view: str, token: str) -> int:
    """Actual child chain.  Public callers cannot reach it without parent token."""
    launch_contract = _load_child_launch_contract(view=view, token=token)
    _verify_child_environment(view=view, token=token, launch_contract=launch_contract)
    if view == "pseudo_mua":
        early_preflight = load_launcher_preflight(validate_target_bindings=False)
        require(early_preflight["body_sha256"] ==
                launch_contract["payload"]["launcher_preflight_body_sha256"],
                "pMUA launch contract/preflight drift before SUA predecessor gate")
        _require_sua_terminal_before_pmua(launcher_preflight=early_preflight)
    prerequisites = validate_pretarget_prerequisites(
        only_view=view, existing_launch_contract=launch_contract)
    require(launch_contract["payload"] == _launch_contract_payload(
        view=view, token=token, prerequisites=prerequisites, parent_pid=os.getppid()),
        "parent launch contract differs from current exact prerequisites")
    capability = prerequisites.capabilities[view]
    gpu_identity = _verify_gpu_against_cost(prerequisites.cost_identity)
    topology = producer._topology(view)
    start_payload = _start_payload(capability=capability, prerequisites=prerequisites,
                                   gpu_identity=gpu_identity, launch_contract=launch_contract)
    _publish(Path(topology["start"]), start_payload)

    source = producer.materialize_and_verify_strict27_source(capability)
    _publish(Path(topology["source"]), source.payload)
    target = producer.materialize_target_from_private_snapshot(capability, source)

    vendor = str(REPO_ROOT / "cebra_exploration/third_party/cebra")
    require(vendor not in sys.path, "vendored CEBRA path unexpectedly preloaded before child fit")
    sys.path.insert(0, vendor)
    encoder = producer.fit_one_joint_encoder(source=source, target=target)
    persistence = producer.persist_sklearn_checkpoint_and_embeddings(
        capability=capability, estimator=encoder.estimator,
        embedding_arrays=encoder.persisted_embedding_arrays,
        reload_probe_inputs=encoder.reload_probe_inputs,
        cebra_loader=type(encoder.estimator).load)
    target_payload = producer.finalize_target_payload(
        target=target, encoder=encoder, start_sha256=start_payload["start_payload_sha256"])
    _publish(Path(topology["target"]), target_payload)
    encoder_payload = producer.build_encoder_payload(
        capability=capability, start_sha256=start_payload["start_payload_sha256"],
        source_sha256=source.payload["source_payload_sha256"],
        target_sha256=target_payload["target_payload_sha256"],
        checkpoint_binding=persistence["checkpoint"], embedding_binding=persistence["embeddings"],
        fit_proof=encoder.fit_proof)
    encoder_payload = dict(encoder_payload) | {
        "source_receipt_path": topology["source"],
        "source_receipt_body_sha256": _read_pair(Path(topology["source"]), label="source receipt")["body_sha256"],
    }
    encoder_payload["encoder_payload_sha256"] = _sha_json(
        {key: value for key, value in encoder_payload.items() if key != "encoder_payload_sha256"})
    _publish(Path(topology["encoder"]), encoder_payload)
    scores = producer.score_all_six_readouts(
        encoder=encoder, target_payload=target_payload, encoder_payload=encoder_payload)
    for role, payload in scores.items():
        _publish(Path(topology["scores"][role]), payload)
    _cleanup_private_snapshot_after_success(view=view, target=target)
    return 0


def _failure_completion(*, view: str, returncode: int, start_exists: bool,
                        stdout: bytes, stderr: bytes, argv: Sequence[str],
                        env: Mapping[str, str], progress: Mapping[str, bool],
                        failure_stage: str, launch_contract: Mapping[str, Any],
                        validation_error: str = "") -> dict[str, Any]:
    payload = {
        "schema": SCHEMA_FAILURE_COMPLETION, "status": "TERMINAL_CHILD_FAILURE__CELL_NONREUSABLE",
        "cell": sealed_runtime.StagePCell.from_view(view).as_dict(),
        "subprocess_started": True, "actual_started": True,
        "start_receipt_published": start_exists,
        "parent_launch_contract": {
            key: launch_contract[key] for key in ("path", "body_sha256", "sidecar_sha256")},
        "actual_exit_code": int(returncode), "actual_argv": list(argv),
        "failure_stage": failure_stage,
        "post_exit_validation_error_sha256": _sha_bytes(validation_error.encode("utf-8")),
        "actual_environment": {key: env.get(key) for key in
                               ("PYTHONNOUSERSITE", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                                ROOT_ENV, CHILD_VIEW_ENV)},
        "stdout_sha256": _sha_bytes(stdout), "stderr_sha256": _sha_bytes(stderr),
        "pMUA_start_permitted": False, "target_updates": 0, "formal_data_opened": False,
        "observed_progress_at_parent_reap": dict(progress),
        "target_opened_or_may_have_opened": bool(
            progress.get("target") or progress.get("private_snapshot_present")),
        "private_snapshot_failure_policy": PRIVATE_SNAPSHOT_POLICY["failure"],
    }
    return payload | {"failure_completion_payload_sha256": _sha_json(payload)}


def _finalize_success(*, view: str, returncode: int, argv: Sequence[str]) -> dict[str, Any]:
    require(returncode == 0, "cannot finalize nonzero child as success")
    # The first decisive operation is a same-FD reload plus full live rebuild of
    # the root-minted launcher preflight.  A long fit may not silently adopt a
    # new launcher/producer/admission closure at terminal publication.
    preflight = load_launcher_preflight()
    admissions = _fresh_admissions()
    plan = producer.build_no_target_review_plan()
    capabilities = producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
    addendum = producer.load_execution_addendum(admissions=admissions)
    live_launcher_closure = implementation_closure()
    require(preflight["payload"]["launcher_implementation_closure"] == live_launcher_closure and
            preflight["payload"]["producer_closure_sha256"] ==
            producer.implementation_closure()["closure_sha256"] and
            preflight["payload"]["producer_addendum_body_sha256"] == addendum["body_sha256"],
            "launch/final/live implementation closure or addendum drift")
    capability = capabilities[view]
    topology = producer._topology(view)
    launch_contract = _load_output(_launch_contract_path(view), schema=SCHEMA_LAUNCH_CONTRACT,
                                   label=f"{view} launch contract final")
    start = _load_output(Path(topology["start"]), schema=producer.SCHEMA_START, label=f"{view} start")
    source = _load_output(Path(topology["source"]), schema=producer.SCHEMA_SOURCE, label=f"{view} source")
    target = _load_output(Path(topology["target"]), schema=producer.SCHEMA_TARGET, label=f"{view} target")
    encoder = _load_output(Path(topology["encoder"]), schema=producer.SCHEMA_ENCODER, label=f"{view} encoder")
    require(list(argv) == _child_argv(view), f"{view} parent-observed child argv drift")
    _validate_start_chain(view=view, start=start, capability=capability, preflight=preflight,
                          addendum=addendum, launch_contract=launch_contract)
    checkpoint = _load_raw_output(
        Path(topology["checkpoint"]),
        expected_sha256=str(encoder["payload"]["checkpoint"]["body_sha256"]),
        label=f"{view} checkpoint")
    embeddings = _load_raw_output(
        Path(topology["embeddings"]),
        expected_sha256=str(encoder["payload"]["embedding_bundle"]["body_sha256"]),
        label=f"{view} embeddings")
    score_shas: dict[str, str] = {}
    score_bindings: dict[str, Any] = {}
    loaded_scores: dict[str, Mapping[str, Any]] = {}
    for route in producer.ROUTES:
        for decoder in producer.DECODERS:
            role = f"{route}__{decoder}"
            loaded = _load_output(Path(topology["scores"][role]), schema=producer.SCHEMA_SCORE,
                                  label=f"{view} score {role}")
            loaded_scores[role] = loaded
            score_shas[role] = loaded["payload"]["score_payload_sha256"]
            score_bindings[role] = {key: loaded[key] for key in
                                    ("path", "body_sha256", "sidecar_sha256")}
    _validate_scientific_output_chain(
        view=view, capability=capability, start=start, source=source, target=target,
        encoder=encoder, scores=loaded_scores)
    completion = producer.build_completion_payload(
        capability=capability, start_sha256=start["payload"]["start_payload_sha256"],
        target_sha256=target["payload"]["target_payload_sha256"],
        encoder_sha256=encoder["payload"]["encoder_payload_sha256"],
        score_sha256_by_role=score_shas)
    completion = dict(completion)
    completion.pop("completion_payload_sha256")
    completion.update({"subprocess_started": True, "actual_started": True,
                       "actual_exit_code": 0, "actual_argv": list(argv),
                       "parent_launch_contract_body_sha256": launch_contract["body_sha256"],
                       "launcher_preflight_body_sha256": preflight["body_sha256"],
                       "launcher_closure_sha256_at_launch": start["payload"]["launcher_closure_sha256"],
                       "launcher_closure_sha256_at_final": live_launcher_closure["closure_sha256"],
                       "launcher_launch_final_live_exact_equal": True,
                       "source_receipt_body_sha256": source["body_sha256"],
                       "immutable_output_bindings": {
                           "start": {key: start[key] for key in ("path", "body_sha256", "sidecar_sha256")},
                           "source": {key: source[key] for key in ("path", "body_sha256", "sidecar_sha256")},
                           "target": {key: target[key] for key in ("path", "body_sha256", "sidecar_sha256")},
                           "encoder": {key: encoder[key] for key in ("path", "body_sha256", "sidecar_sha256")},
                           "checkpoint": checkpoint, "embeddings": embeddings,
                           "scores": score_bindings,
                       },
                       "formal_data_opened": False, "target_updates": 0})
    completion["completion_payload_sha256"] = _sha_json(completion)
    completion_binding = _publish(Path(topology["completion"]), completion)
    terminal = producer.build_terminal_payload(
        capability=capability, completion_payload=completion,
        live_closure=producer.implementation_closure())
    terminal = dict(terminal) | {
        "launcher_preflight_body_sha256": preflight["body_sha256"],
        "launcher_preflight_payload_sha256": preflight["payload"]["launcher_preflight_payload_sha256"],
        "launcher_closure_sha256": live_launcher_closure["closure_sha256"],
        "launcher_closure_sha256_at_launch": start["payload"]["launcher_closure_sha256"],
        "launcher_closure_sha256_at_final": live_launcher_closure["closure_sha256"],
        "launcher_launch_final_live_exact_equal": True,
        "producer_closure_sha256_at_launch": start["payload"][
            "producer_implementation_closure"]["closure_sha256"],
        "producer_closure_sha256_at_final": producer.implementation_closure()["closure_sha256"],
        "producer_launch_final_live_exact_equal": True,
        "parent_launch_contract_body_sha256": launch_contract["body_sha256"],
        "actual_exit_code": 0, "source_receipt_body_sha256": source["body_sha256"],
        "completion_receipt_body_sha256": completion_binding["body_sha256"],
        "private_snapshot_policy": dict(PRIVATE_SNAPSHOT_POLICY),
    }
    terminal["terminal_payload_sha256"] = _sha_json(
        {key: value for key, value in terminal.items() if key != "terminal_payload_sha256"})
    _publish(Path(topology["terminal"]), terminal)
    return terminal


ChildRunner = Callable[[str, Sequence[str], Mapping[str, str]], tuple[int, bytes, bytes]]


def _subprocess_child_runner(view: str, argv: Sequence[str], env: Mapping[str, str]) -> tuple[int, bytes, bytes]:
    completed = subprocess.run(list(argv), cwd=REPO_ROOT, env=dict(env), capture_output=True, check=False)
    return int(completed.returncode), bytes(completed.stdout), bytes(completed.stderr)


def _run_one_child(*, view: str, token: str, runner: ChildRunner,
                   prerequisites: ExecutionPrerequisites) -> dict[str, Any]:
    argv = _child_argv(view)
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONHOME", "CUDA_VISIBLE_DEVICES", INTERNAL_TOKEN_ENV,
                          CHILD_VIEW_ENV, ROOT_ENV, *FORBIDDEN_CHILD_ENV}}
    env.update(_child_env_contract(view))
    env[INTERNAL_TOKEN_ENV] = token
    launch_binding = _publish_launch_contract(
        view=view, token=token, prerequisites=prerequisites)
    returncode, stdout, stderr = runner(view, argv, env)
    topology = producer._topology(view)
    start_exists = (os.path.lexists(topology["start"]) and
                    os.path.lexists(Path(f"{topology['start']}.sha256")))
    if returncode != 0:
        progress = {
            role: (os.path.lexists(path) and os.path.lexists(Path(f"{path}.sha256")))
            for role, path in {
                "start": topology["start"], "source": topology["source"],
                "target": topology["target"], "encoder": topology["encoder"],
            }.items()
        }
        progress["launch_contract"] = True
        progress["private_snapshot_present"] = os.path.lexists(topology["private_snapshot"])
        failure = _failure_completion(view=view, returncode=returncode, start_exists=start_exists,
                                      stdout=stdout, stderr=stderr, argv=argv, env=env,
                                      progress=progress, failure_stage="isolated_child_exit",
                                      launch_contract=launch_binding)
        completion_path = Path(topology["completion"])
        if not os.path.lexists(completion_path) and not os.path.lexists(Path(f"{completion_path}.sha256")):
            _publish(completion_path, failure)
        raise StagePPairedLiveLauncherError(
            f"{view} isolated child failed with exit code {returncode}; pMUA is blocked")
    try:
        return _finalize_success(view=view, returncode=returncode, argv=argv)
    except BaseException as exc:
        completion_path = Path(topology["completion"])
        if not os.path.lexists(completion_path) and not os.path.lexists(Path(f"{completion_path}.sha256")):
            progress = {
                role: (os.path.lexists(path) and os.path.lexists(Path(f"{path}.sha256")))
                for role, path in {
                    "start": topology["start"], "source": topology["source"],
                    "target": topology["target"], "encoder": topology["encoder"],
                }.items()
            }
            progress["launch_contract"] = True
            progress["private_snapshot_present"] = os.path.lexists(topology["private_snapshot"])
            failure = _failure_completion(
                view=view, returncode=0, start_exists=start_exists, stdout=stdout, stderr=stderr,
                argv=argv, env=env, progress=progress, failure_stage="post_exit_immutable_validation",
                launch_contract=launch_binding,
                validation_error=f"{type(exc).__name__}:{exc}")
            _publish(completion_path, failure)
        else:
            terminal_path = Path(topology["terminal"])
            if not os.path.lexists(terminal_path) and not os.path.lexists(Path(f"{terminal_path}.sha256")):
                failure_terminal = {
                    "schema": SCHEMA_FAILURE_TERMINAL,
                    "status": "POST_EXIT_FINALIZATION_FAILED__CELL_NONREUSABLE",
                    "cell": sealed_runtime.StagePCell.from_view(view).as_dict(),
                    "actual_exit_code": 0, "actual_started": True,
                    "completion_pair_present": True,
                    "parent_launch_contract": {
                        key: launch_binding[key]
                        for key in ("path", "body_sha256", "sidecar_sha256")},
                    "validation_error_sha256": _sha_bytes(
                        f"{type(exc).__name__}:{exc}".encode("utf-8")),
                    "pMUA_start_permitted": False, "formal_data_opened": False,
                    "target_updates": 0, "target_opened_or_may_have_opened": True,
                }
                failure_terminal["failure_terminal_payload_sha256"] = _sha_json(failure_terminal)
                _publish(terminal_path, failure_terminal)
        raise StagePPairedLiveLauncherError(
            f"{view} child exited zero but immutable output finalization failed; pMUA is blocked") from exc


def _orchestrate_authorized_pair(*, runner: ChildRunner) -> dict[str, Any]:
    require(os.environ.get(ROOT_ENV) == "1", f"real execution requires {ROOT_ENV}=1")
    prerequisites = validate_pretarget_prerequisites()
    sua_terminal = _run_one_child(
        view="sua", token=secrets.token_hex(32), runner=runner, prerequisites=prerequisites)
    immutable_sua = _load_output(
        Path(producer._topology("sua")["terminal"]), schema=producer.SCHEMA_TERMINAL,
        label="SUA terminal before pMUA admission")
    require(immutable_sua["payload"] == sua_terminal and
            sua_terminal.get("status") == "TERMINAL_SUCCESS__DEVELOPMENT_PILOT_CELL" and
            sua_terminal.get("cell", {}).get("view") == "sua",
            "pMUA start forbidden before exact SUA terminal success")
    pmua_prerequisites = validate_pretarget_prerequisites(only_view="pseudo_mua")
    pmua_terminal = _run_one_child(
        view="pseudo_mua", token=secrets.token_hex(32), runner=runner,
        prerequisites=pmua_prerequisites)
    terminal_bindings = {
        view: _load_output(Path(producer._topology(view)["terminal"]),
                           schema=producer.SCHEMA_TERMINAL, label=f"{view} terminal final")
        for view in PAIR_ORDER
    }
    require(terminal_bindings["sua"]["payload"] == sua_terminal and
            terminal_bindings["pseudo_mua"]["payload"] == pmua_terminal,
            "returned terminal payload differs from immutable terminal pair")
    final_preflight = load_launcher_preflight()
    final_closure = implementation_closure()
    require(final_preflight["payload"]["launcher_implementation_closure"] == final_closure and
            all(binding["payload"].get("launcher_preflight_body_sha256") ==
                final_preflight["body_sha256"] and
                binding["payload"].get("launcher_closure_sha256_at_launch") ==
                final_closure["closure_sha256"] and
                binding["payload"].get("launcher_closure_sha256_at_final") ==
                final_closure["closure_sha256"] and
                binding["payload"].get("launcher_launch_final_live_exact_equal") is True and
                binding["payload"].get("producer_launch_final_live_exact_equal") is True
                for binding in terminal_bindings.values()),
            "paired completion launch/final/live terminal closure drift")
    paired = producer.build_paired_completion_payload(
        sua_terminal=sua_terminal, pmua_terminal=pmua_terminal)
    paired = dict(paired) | {
        "launcher_preflight_body_sha256": final_preflight["body_sha256"],
        "launcher_closure_sha256": final_closure["closure_sha256"],
        "launcher_launch_final_live_exact_equal": True,
        "terminal_receipt_body_sha256_by_view": {
            view: terminal_bindings[view]["body_sha256"] for view in PAIR_ORDER},
    }
    paired["paired_completion_payload_sha256"] = _sha_json(
        {key: value for key, value in paired.items() if key != "paired_completion_payload_sha256"})
    _publish(PAIRED_COMPLETION, paired)
    return paired


def execute_authorized_parent() -> dict[str, Any]:
    """Fixed production entry: no injectable runner or caller paths."""
    return _orchestrate_authorized_pair(runner=_subprocess_child_runner)
