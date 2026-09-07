"""Seed-43 TF-SR production contract: seed pinning, lineage, and build disclosure.

This contract binds the matched-replication seed-43 run
``TFSR_B3ST4_DDROP_SEED43``.  It supersedes the frozen seed-42 production
contract (``src/tfsr_b3st4_ddrop_v1/contract.py``) *as the production route for
the matched replication seeds 43/44 only*; the seed-42 route itself stays
frozen and unmodified, and its live artifacts are never read or written here.

Two things are new relative to the seed-42 contract and both are load-bearing:

1. the device authority is physical GPU0 (the seed-42 live run occupies GPU1);
2. the executed forward is the accelerated build v2 (jit-scripted recurrent
   step) whose evidence is the sealed throughput-v2 engineering receipt.
   That adoption is disclosed in full in :data:`BUILD_DISCLOSURE`.

The module imports only torch-free frozen helpers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.tfsr_b3st4_ddrop_v1.contract import _canonical_regular_bytes, compute_live_closure, verify_canonical_evidence
from src.tfsr_b3st4_ddrop_v1.source_smoke import EXPECTED_STAGE0_SHA


CELL_43 = "TFSR_B3ST4_DDROP_SEED43"
SEED_43 = 43
SUPERSEDED_CELL = "TFSR_B3ST4_DDROP_SEED42"
SUPERSEDED_CONTRACT_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/contract.py"

# Fresh, currently-absent canonical output root for the seed-43 training run.
TRAIN_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_train_v1"

# The frozen seed-42 live run occupies physical GPU1 under CVD=1.  Seed 43 is
# ordered onto physical GPU0 exposed as the sole logical CUDA device.  The
# identity fields below are the GPU0 authority of the frozen throughput
# benchmark (``throughput_benchmark.FROZEN_GPU0``); a test pins the two
# definitions together so this route cannot silently drift from it.
FROZEN_DEVICE_43 = {
    "internal_device": "cuda:0",
    "cuda_visible_devices": "0",
    "uuid": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
    "bdf": "00000000:01:00.0",
    "name": "NVIDIA GeForce RTX 3090",
    "memory_total_mib": 24576,
}

# --------------------------------------------------------------------------- #
# Sealed accelerated-build evidence: the throughput-v2 engineering receipt
# --------------------------------------------------------------------------- #

THROUGHPUT_V2_RECEIPT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v2/receipt.json"
THROUGHPUT_V2_RECEIPT_SIDECAR_RELATIVE = THROUGHPUT_V2_RECEIPT_RELATIVE + ".sha256"
THROUGHPUT_V2_RECEIPT_SHA256 = "3438b9a5c264cb497315c6e2650c46f2892a8fea3c97d3af433d7e22f10b6f57"

# Exact measured evidence for the only lever this route adopts.  These are
# pinned to the digit; verify_throughput_v2_receipt requires the sealed receipt
# to reproduce them exactly before any identity can be minted.
JIT_SCRIPTED_STEP_EVIDENCE = {
    "64": {
        "median_step_wall_seconds": 0.09636688558384776,
        "speedup_vs_production": 1.496591690875804,
        "first_step_forward_max_abs": 0.0,
        "first_step_loss_abs": 0.0,
        "first_step_gradient_max_abs": 9.313225746154785e-10,
        "trajectory_forward_max_abs": 0.00010414421558380127,
        "trajectory_gradient_max_abs": 4.9874186515808105e-05,
        "trajectory_model_state_max_abs": 2.896413207054138e-06,
        "trajectory_optimizer_state_max_abs": 3.212830051779747e-06,
    },
    "128": {
        "median_step_wall_seconds": 0.15797596611082554,
        "speedup_vs_production": 1.5632337219162493,
        "first_step_forward_max_abs": 0.0,
        "first_step_loss_abs": 0.0,
        "first_step_gradient_max_abs": 1.8812716007232666e-07,
        "trajectory_forward_max_abs": 0.00012637674808502197,
        "trajectory_gradient_max_abs": 5.324650555849075e-05,
        "trajectory_model_state_max_abs": 1.0132789611816406e-05,
        "trajectory_optimizer_state_max_abs": 6.650574505329132e-06,
    },
}
FROZEN_GRADIENT_BAND = 1e-06

BUILD_DISCLOSURE_STATEMENT = (
    "seed 43 executes the accelerated build (jit-scripted recurrent step; throughput v2 "
    "receipt SHA bound; first-step forward bitwise-equal to the frozen build, gradient "
    "within the frozen band, 20-step trajectory diverges from the frozen build at fp32 "
    "reduction-order noise level — the same property validation hoisting has; disclosed "
    "as build v2 for seeds 43/44)"
)

TRAJECTORY_AMPLIFICATION_NOTE = (
    "The throughput-v2 receipt separates the first paired step from the 20-step trajectory "
    "because a recurrent fp32 trajectory amplifies backward reduction-order differences "
    "beyond the single-step tolerance map.  Every measured candidate with an exact/in-band "
    "first step shows this amplification, including validation hoisting, which changes no "
    "mathematics at all; jit_scripted_step's 20-step forward drift (~1.0e-4 at units=64, "
    "~1.3e-4 at units=128) is that same reduction-order property, not a different model.  "
    "Seed 43 therefore discloses build v2 rather than claiming bitwise trajectory equality "
    "with the frozen build."
)

BUILD_DISCLOSURE: dict[str, Any] = {
    "build": "v2_accelerated_jit_scripted_step",
    "statement": BUILD_DISCLOSURE_STATEMENT,
    "throughput_v2_receipt_relative": THROUGHPUT_V2_RECEIPT_RELATIVE,
    "throughput_v2_receipt_sha256": THROUGHPUT_V2_RECEIPT_SHA256,
    "jit_scripted_step_evidence": {units: dict(values) for units, values in JIT_SCRIPTED_STEP_EVIDENCE.items()},
    "frozen_gradient_band": FROZEN_GRADIENT_BAND,
    "trajectory_amplification_note": TRAJECTORY_AMPLIFICATION_NOTE,
    "applies_to_seeds": [43, 44],
}

SUPERSEDES_NOTE = (
    "This route supersedes the frozen seed-42 production contract "
    "(tfpd_exploration/src/tfsr_b3st4_ddrop_v1/contract.py, cell TFSR_B3ST4_DDROP_SEED42) "
    "as the production route for the matched replication seeds 43/44 ordered by section 0 "
    "of HANDOFF_TASK_FRAME_STATEFUL_READIN_20260819.md.  The training recipe is exactly the "
    "seed-42 recipe with seed 43: identical strict-27 source roster, M30 normalized-T4 "
    "authority, B3S identity path, model graph, Adam optimizer, warmup/cosine schedule, "
    "48 epochs, final-four SWA, loss, and Cell-D whole-unit dropout law.  Only the seed "
    "differs.  The frozen seed-42 package is imported, never modified; the seed-42 run "
    "roots are never read or written by this route."
)

LINEAGE_KEYS = (
    "supersedes_cell",
    "supersedes_contract_relative",
    "supersedes_note",
    "throughput_v2_receipt_relative",
    "throughput_v2_receipt_sha256",
    "frozen_route_closure_sha256",
    "seed42_run_resumed",
)

# --------------------------------------------------------------------------- #
# Route closures
# --------------------------------------------------------------------------- #

# The explicit, ordered, non-globbed closure of this new route.
SEED43_CLOSURE = (
    "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/__init__.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/contract_43.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/accelerated_forward.py",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/train_43.py",
    "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_train.py",
    "tfpd_exploration/scripts/preflight_tfsr_b3st4_ddrop_seed43.py",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_seed43_v1.py",
)

# Frozen seed-42-route files this route imports or executes against.  The
# throughput-benchmark trio is pinned to the exact hashes recorded inside the
# sealed throughput-v2 receipt, so this route provably executes the audited
# accelerated build rather than a later edit of it.
EXPECTED_FROZEN_ROUTE_SHA = {
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/contract.py":
        "c8a7f20aea6982e02ccf9d674100fbe29407403077a5adbce9e95da3d989888f",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/__init__.py":
        "06f3d404f3f73a7bfb69349d71f4adeb24f6306daac99e2a3e7d28a6645f730d",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/model.py":
        "3d4a3a8d4e2a68e9933e84274f2a62308a6eeb5a6978671b53e72af442148fc4",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/source_smoke.py":
        "a4882edc8754b6d4cd09b7e86809dcfb9703797f9b94c95020d9c7bd29fb7cfd",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/train.py":
        "dfbc6fa4c9e016304d439a960af32b82271356031d24220daa6f720a7e520512",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark.py":
        "1f2dc760fe1848069c8b03c5ec237ab8720f5c0c4afbd28a22d7343f3d429add",
    "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark_v2.py":
        "fc96b567f4eaf7ff6b6e28b1d65ab56e97bd2ee79bcf0f572d06bb6b263dd766",
    "tfpd_exploration/src/tfpd/bilinear_readin.py":
        "2576e91baa2ecbae7d9b5734d2aad6618563fe5cd8dbb5cf0f15358caf244ac0",
    "tfpd_exploration/src/tfpd_lane/arm_common.py":
        "9df0af9ee238b41ce290624aad071da626e68302f1940c05d07d6481caa3c2e8",
    "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_train.py":
        "448ad77c62db9e84cab003543068c4494b1cbb27339d9df88a2b240f9aebd56e",
    "tfpd_exploration/scripts/preflight_tfsr_b3st4_ddrop_seed42.py":
        "b1a267d928108a800f5e16d24cf2eaccd9f77bf9d13aac70a0fa72c8520fdeff",
    "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_source_smoke.py":
        "2ec1f9823c2da81bf2564d4e68ab80041e12f607ab5703158484442239f101f7",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_train.py":
        "6532bdec4c6bb2ce9d348ed3f5d5a003bd666c6b0ae9e867a809d0916703271b",
    "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_source_smoke.py":
        "6274ef8d2a81864a2f27faa01da23f93b8109cd0bf86650dffda8feb85902446",
    "tfpd_exploration/docs/WORKORDER_TFSR_PHASE_D_SUCCESSOR_V2_20260819.md":
        "5286a648e9ddf889fc2cd98475fa6c428a3a52f52264711c528f6cfef01d6b20",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_exact_float(value: object, expected: float, label: str) -> None:
    if type(value) is not float or value != expected:
        raise RuntimeError(f"throughput-v2 evidence numeric drift: {label}")


def compute_seed43_closure(root: Path) -> dict[str, object]:
    """Descriptor-safely hash the explicit seed-43 implementation closure."""
    return compute_live_closure(root, SEED43_CLOSURE)


def verify_frozen_route(root: Path) -> dict[str, object]:
    """Fail closed when any frozen seed-42 file this route depends on drifts."""
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_FROZEN_ROUTE_SHA.items():
        body = _canonical_regular_bytes(root.absolute() / relative)
        observed[relative] = _sha(body)
        if observed[relative] != expected:
            raise RuntimeError(f"frozen seed-42 route SHA drift: {relative}")
    for relative, expected in EXPECTED_STAGE0_SHA.items():
        if observed.get(relative, expected) != expected:
            raise RuntimeError(f"frozen Stage-0 SHA drift: {relative}")
    payload = json.dumps({"files": dict(observed)}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "supersedes_cell": SUPERSEDED_CELL,
        "supersedes_contract_relative": SUPERSEDED_CONTRACT_RELATIVE,
        "sha256_by_path": observed,
        "frozen_route_closure_sha256": _sha(payload),
    }


def verify_throughput_v2_receipt(root: Path) -> dict[str, object]:
    """Descriptor-verify the sealed accelerated-build evidence receipt."""
    path = root.absolute() / THROUGHPUT_V2_RECEIPT_RELATIVE
    body = _canonical_regular_bytes(path, expected_mode=0o444)
    if _sha(body) != THROUGHPUT_V2_RECEIPT_SHA256:
        raise RuntimeError("throughput-v2 receipt body SHA drift")
    sidecar = _canonical_regular_bytes(root.absolute() / THROUGHPUT_V2_RECEIPT_SIDECAR_RELATIVE, expected_mode=0o444)
    if sidecar != f"{THROUGHPUT_V2_RECEIPT_SHA256}  receipt.json\n".encode("ascii"):
        raise RuntimeError("throughput-v2 receipt sidecar drift")
    try:
        receipt = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("throughput-v2 receipt is not valid JSON") from error
    if not isinstance(receipt, Mapping):
        raise RuntimeError("throughput-v2 receipt root is not an object")
    if (receipt.get("schema") != "tfsr_b3st4_ddrop_throughput_engineering_v2"
            or receipt.get("status") != "ENGINEERING_BENCHMARK_COMPLETE"
            or receipt.get("purpose") != "ENGINEERING_ONLY"
            or receipt.get("authorizes_training") is not False):
        raise RuntimeError("throughput-v2 receipt boundary/schema drift")
    cells = receipt.get("matrix", {}).get("cells") if isinstance(receipt.get("matrix"), Mapping) else None
    if not isinstance(cells, list):
        raise RuntimeError("throughput-v2 receipt matrix is missing")
    measured: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        if isinstance(cell, Mapping) and cell.get("kind") == "jit_scripted_step":
            if cell.get("status") != "MEASURED" or cell.get("execution") != "eager":
                raise RuntimeError("throughput-v2 jit_scripted_step cell status drift")
            units = str(cell.get("units"))
            if units in measured:
                raise RuntimeError("throughput-v2 jit_scripted_step duplicate units")
            measured[units] = cell
    if set(measured) != {"64", "128"}:
        raise RuntimeError("throughput-v2 jit_scripted_step unit sweep drift")
    for units, expected_values in JIT_SCRIPTED_STEP_EVIDENCE.items():
        cell = measured[units]
        equivalence = cell.get("equivalence")
        if not isinstance(equivalence, Mapping) or equivalence.get("first_step_within_tolerance") is not True:
            raise RuntimeError(f"throughput-v2 jit_scripted_step u{units} first-step band drift")
        first = equivalence.get("first_step_differences")
        trajectory = equivalence.get("differences")
        timing = cell.get("timing")
        if not isinstance(first, Mapping) or not isinstance(trajectory, Mapping) or not isinstance(timing, Mapping):
            raise RuntimeError(f"throughput-v2 jit_scripted_step u{units} evidence shape drift")
        _require_exact_float(first.get("forward_max_abs"), expected_values["first_step_forward_max_abs"],
                             f"u{units}.first_step.forward_max_abs")
        _require_exact_float(first.get("loss_abs"), expected_values["first_step_loss_abs"], f"u{units}.first_step.loss_abs")
        _require_exact_float(first.get("gradient_max_abs"), expected_values["first_step_gradient_max_abs"],
                             f"u{units}.first_step.gradient_max_abs")
        if not expected_values["first_step_gradient_max_abs"] <= FROZEN_GRADIENT_BAND:
            raise RuntimeError("pinned first-step gradient lies outside the frozen band")
        _require_exact_float(trajectory.get("forward_max_abs"), expected_values["trajectory_forward_max_abs"],
                             f"u{units}.trajectory.forward_max_abs")
        _require_exact_float(trajectory.get("gradient_max_abs"), expected_values["trajectory_gradient_max_abs"],
                             f"u{units}.trajectory.gradient_max_abs")
        _require_exact_float(trajectory.get("model_state_max_abs"), expected_values["trajectory_model_state_max_abs"],
                             f"u{units}.trajectory.model_state_max_abs")
        _require_exact_float(trajectory.get("optimizer_state_max_abs"), expected_values["trajectory_optimizer_state_max_abs"],
                             f"u{units}.trajectory.optimizer_state_max_abs")
        _require_exact_float(timing.get("median_step_wall_seconds"), expected_values["median_step_wall_seconds"],
                             f"u{units}.timing.median_step_wall_seconds")
    conclusion = receipt.get("conclusion")
    fastest = conclusion.get("fastest_first_step_within_tolerance") if isinstance(conclusion, Mapping) else None
    if not isinstance(fastest, Mapping) or fastest.get("kind") != "jit_scripted_step":
        raise RuntimeError("throughput-v2 conclusion does not rank jit_scripted_step first")
    return {
        "relative_path": THROUGHPUT_V2_RECEIPT_RELATIVE,
        "sidecar_relative_path": THROUGHPUT_V2_RECEIPT_SIDECAR_RELATIVE,
        "body_sha256": THROUGHPUT_V2_RECEIPT_SHA256,
        "mode": "0444",
        "schema": receipt["schema"],
        "status": receipt["status"],
        "jit_scripted_step_evidence": {units: dict(values) for units, values in JIT_SCRIPTED_STEP_EVIDENCE.items()},
        "conclusion_fastest_kind": "jit_scripted_step",
    }


def verify_canonical_evidence_43(root: Path) -> dict[str, object]:
    """Bind the frozen scientific evidence plus the accelerated-build receipt."""
    frozen = verify_canonical_evidence(root)  # handoff + terminal AM/IM matrix
    return {
        "cell": CELL_43,
        "seed": SEED_43,
        "frozen_evidence": frozen,
        "frozen_route": verify_frozen_route(root),
        "throughput_v2_receipt": verify_throughput_v2_receipt(root),
        "supersedes_note": SUPERSEDES_NOTE,
        "build_disclosure": {key: (dict(value) if isinstance(value, dict) else value) for key, value in BUILD_DISCLOSURE.items()},
    }


def public_lineage(root: Path) -> dict[str, object]:
    """Static lineage block; the frozen-route digest is measured, not trusted."""
    frozen = verify_frozen_route(root)
    return {
        "supersedes_cell": SUPERSEDED_CELL,
        "supersedes_contract_relative": SUPERSEDED_CONTRACT_RELATIVE,
        "supersedes_note": SUPERSEDES_NOTE,
        "throughput_v2_receipt_relative": THROUGHPUT_V2_RECEIPT_RELATIVE,
        "throughput_v2_receipt_sha256": THROUGHPUT_V2_RECEIPT_SHA256,
        "frozen_route_closure_sha256": frozen["frozen_route_closure_sha256"],
        "seed42_run_resumed": False,
    }


def validate_lineage(value: object, *, exact: Mapping[str, object] | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(LINEAGE_KEYS):
        raise RuntimeError("seed-43 lineage schema drift")
    if (value.get("supersedes_cell") != SUPERSEDED_CELL
            or value.get("supersedes_contract_relative") != SUPERSEDED_CONTRACT_RELATIVE
            or value.get("supersedes_note") != SUPERSEDES_NOTE
            or value.get("throughput_v2_receipt_relative") != THROUGHPUT_V2_RECEIPT_RELATIVE
            or value.get("throughput_v2_receipt_sha256") != THROUGHPUT_V2_RECEIPT_SHA256
            or not isinstance(value.get("frozen_route_closure_sha256"), str)
            or len(value.get("frozen_route_closure_sha256")) != 64
            or value.get("seed42_run_resumed") is not False):
        raise RuntimeError("seed-43 lineage semantic drift")
    if exact is not None and json.loads(json.dumps(dict(value), sort_keys=True)) != json.loads(json.dumps(dict(exact), sort_keys=True)):
        raise RuntimeError("seed-43 exact lineage binding drift")
    return value


def validate_build_disclosure(value: object) -> dict[str, object]:
    """Require the full production build disclosure, never a summary of it."""
    if not isinstance(value, Mapping):
        raise RuntimeError("build disclosure must be a mapping")
    expected_keys = {"build", "statement", "throughput_v2_receipt_relative", "throughput_v2_receipt_sha256",
                     "jit_scripted_step_evidence", "frozen_gradient_band", "trajectory_amplification_note",
                     "applies_to_seeds"}
    if set(value) != expected_keys:
        raise RuntimeError("build disclosure schema drift")
    if (value.get("build") != BUILD_DISCLOSURE["build"]
            or value.get("statement") != BUILD_DISCLOSURE_STATEMENT
            or value.get("throughput_v2_receipt_relative") != THROUGHPUT_V2_RECEIPT_RELATIVE
            or value.get("throughput_v2_receipt_sha256") != THROUGHPUT_V2_RECEIPT_SHA256
            or value.get("trajectory_amplification_note") != TRAJECTORY_AMPLIFICATION_NOTE
            or value.get("frozen_gradient_band") != FROZEN_GRADIENT_BAND
            or value.get("applies_to_seeds") != [43, 44]):
        raise RuntimeError("build disclosure semantic drift")
    evidence = value.get("jit_scripted_step_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(JIT_SCRIPTED_STEP_EVIDENCE):
        raise RuntimeError("build disclosure evidence unit sweep drift")
    for units, pinned in JIT_SCRIPTED_STEP_EVIDENCE.items():
        row = evidence.get(units)
        if not isinstance(row, Mapping) or set(row) != set(pinned):
            raise RuntimeError(f"build disclosure evidence schema drift: u{units}")
        for key, expected in pinned.items():
            _require_exact_float(row.get(key), expected, f"disclosure.u{units}.{key}")
        if not row["first_step_forward_max_abs"] == 0.0 or not row["first_step_gradient_max_abs"] <= FROZEN_GRADIENT_BAND:
            raise RuntimeError(f"build disclosure first-step band drift: u{units}")
    return json.loads(json.dumps(dict(BUILD_DISCLOSURE), sort_keys=True))


def dry_plan_43(evidence: Mapping[str, object], closure: Mapping[str, object]) -> dict[str, object]:
    """Static seed-43 training declaration; measurement stays separate."""
    return {
        "cell": CELL_43,
        "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH",
        "authorization": "none",
        "seed": SEED_43,
        "supersedes": {
            "cell": SUPERSEDED_CELL,
            "contract_relative": SUPERSEDED_CONTRACT_RELATIVE,
            "note": SUPERSEDES_NOTE,
            "frozen_route_modified": False,
            "seed42_run_roots_touched": False,
        },
        "build_disclosure": {key: (dict(value) if isinstance(value, dict) else value) for key, value in BUILD_DISCLOSURE.items()},
        "device": dict(FROZEN_DEVICE_43),
        "evidence": dict(evidence),
        "seed43_implementation_closure": dict(closure),
        "training": {
            "epochs": 48, "batch_size": 32, "steps_per_epoch": 33925,
            "checkpoint_epochs": [44, 45, 46, 47], "throughput_probe_steps": 100,
            "seed": SEED_43, "capture_diagnostics": False,
            "optimizer": {"cls": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
                          "weight_decay": 0.0, "amsgrad": False, "gradient_clipping": None},
            "schedule": {"authority": "tfpd_lane.arm_common.lr_at_step", "warmup_epochs": 2,
                         "warmup_start_lr": 1e-5, "warmup_end_lr": 1e-4, "cosine_final_lr": 1e-6},
            "data_order": "SessionBatchSampler(seed=43) over the frozen train-only strict-27 adapter",
            "initialization_rng": "random.seed(43); numpy.random.seed(43); torch.manual_seed(43); torch.cuda.manual_seed_all(43)",
            "dropout_law": "identical Cell-D whole-unit law; one Python random.uniform draw and one F.dropout([B,N]) per step, shared across all 50 time bins",
            "swa": "arithmetic_checkpoint_state_mean_final_4",
            "validation_or_target": "forbidden",
            "behavior_scaling_factor": None,
        },
        "canonical_output": {
            "root_relative": TRAIN_ROOT_RELATIVE,
            "must_be_fresh_before_execution": True,
        },
        "forbidden": ["seed-42 route modification", "seed-42 run-root writes", "teacher", "unit/session tables",
                      "target/formal data", "GPU1 (live seed-42 run device)", "validation"],
    }
