"""Execution boundary for the bounded PACD V1 source-only smoke.

Importing this module is inert: it uses only the standard library and the
route's static :mod:`plan`.  Torch, the sealed Cell-D loaders, the canonical
checkpoint, source arrays, and CUDA are imported only inside ``execute``
*after* a fresh immutable ``attempt.json`` has been published.

This boundary intentionally has no scoring path.  It only runs the authorised
P0/P1/P2 source feasibility smoke and records enough evidence for an
independent reviewer to decide whether a later job may be launched.
"""

from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import subprocess
import sys
import time
import traceback
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import plan


class SmokeError(RuntimeError):
    """A PACD smoke preflight or invariant failure."""


@dataclass(frozen=True)
class ExecutionProfile:
    """Lineage-only seam; V1 remains the default science/executor profile."""

    cell: str
    schema: str
    result_root_relative: str
    bound_patterns: tuple[str, ...]
    review_evidence_paths: tuple[str, ...]
    expected_sealed_sha256: dict[str, str]
    work_order_relative: str
    predecessor_validator: Callable[[Path], dict[str, object]] | None = None


def v1_execution_profile() -> ExecutionProfile:
    return ExecutionProfile(
        cell=plan.CELL,
        schema=plan.SCHEMA,
        result_root_relative=plan.RESULT_ROOT_RELATIVE,
        bound_patterns=plan.BOUND_PATTERNS,
        review_evidence_paths=plan.REVIEW_EVIDENCE_PATHS,
        expected_sealed_sha256=plan.EXPECTED_SEALED_SHA256,
        work_order_relative=plan.WORK_ORDER_RELATIVE,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def load_stdlib_receipt_module(root: Path) -> Any:
    """File-path load the stdlib-only receipt module before ``attempt.json``.

    Importing ``src.tfpd_lane.receipt`` first executes the package's eager
    ``__init__``, which imports diagnostics that in turn import Torch.  The
    pre-attempt path must not traverse that package namespace.
    """
    path = root / "tfpd_exploration/src/tfpd_lane/receipt.py"
    _require(path.is_file(), f"PACD receipt module missing: {path}")
    spec = importlib.util.spec_from_file_location("pacd_stdlib_receipt", path)
    _require(spec is not None and spec.loader is not None, "PACD receipt module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_source_closure(root: Path, receipt: Any, patterns: tuple[str, ...] = plan.BOUND_PATTERNS) -> dict[str, object]:
    """Require every explicit bound path exactly once; no silent omissions."""
    closure = receipt.source_closure(root, patterns)
    expected = set(patterns)
    observed = set(closure.get("files", {}))
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    _require(not missing and not extra,
             f"PACD execution closure path mismatch: missing={missing}, extra={extra}")
    return closure


def dry_payload() -> dict[str, object]:
    """Return a static plan without checking files, importing Torch, or CUDA."""
    return plan.dry_payload()


def parse_nvidia_smi_rows(text: str) -> list[list[str]]:
    """Parse an unquoted ``nvidia-smi --format=csv,noheader`` response."""
    return [
        [item.strip() for item in line.split(",")]
        for line in text.splitlines()
        if line.strip()
    ]


def preflight_gpu0_idle(
    *,
    environ: dict[str, str] | None = None,
    run: Callable[..., Any] = subprocess.run,
    no_user_site: bool | None = None,
) -> dict[str, object]:
    """Strictly bind a future process to the authorised, currently idle GPU0.

    This function is intentionally stdlib-only.  It does not import Torch or
    initialise CUDA.  Queries are restricted to physical GPU0 so the route
    neither enumerates nor interprets the state of the active GPU1 job.
    """
    env = os.environ if environ is None else environ
    _require(
        env.get("CUDA_VISIBLE_DEVICES") == plan.EXPECTED_CUDA_VISIBLE_DEVICES,
        "PACD requires CUDA_VISIBLE_DEVICES=0 exactly",
    )
    if no_user_site is None:
        no_user_site = bool(sys.flags.no_user_site)
    _require(bool(no_user_site), "PYTHONNOUSERSITE=1 is mandatory")
    try:
        identity = run(
            [
                "nvidia-smi",
                "-i",
                plan.EXPECTED_GPU_PHYSICAL_INDEX,
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except Exception as error:  # noqa: BLE001 - device ambiguity is fatal
        raise SmokeError(f"GPU0 identity query failed: {error}") from error
    rows = parse_nvidia_smi_rows(identity.stdout)
    _require(len(rows) == 1 and len(rows[0]) == 4, "GPU0 identity query was ambiguous")
    index, uuid, name, driver = rows[0]
    _require(index == plan.EXPECTED_GPU_PHYSICAL_INDEX, "GPU0 physical-index drift")
    _require(uuid == plan.EXPECTED_GPU_UUID, "GPU0 UUID drift")

    try:
        applications = run(
            [
                "nvidia-smi",
                "-i",
                plan.EXPECTED_GPU_PHYSICAL_INDEX,
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except Exception as error:  # noqa: BLE001 - uncertain idleness is fatal
        raise SmokeError(f"GPU0 compute-process query failed: {error}") from error
    processes = parse_nvidia_smi_rows(applications.stdout)
    _require(not processes, "GPU0 is not idle; PACD smoke must not share it")
    return {
        "physical_index": index,
        "uuid": uuid,
        "name": name,
        "driver_version": driver,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "compute_processes": [],
        "gpu0_idle": True,
        "preflight_uses_torch": False,
    }


def _require_fresh_out_dir(out_dir: Path) -> None:
    _require(not out_dir.exists(), f"fresh PACD result root required: {out_dir}")


def canonical_out_dir(root: Path, result_root_relative: str = plan.RESULT_ROOT_RELATIVE) -> Path:
    """The single immutable result destination, without resolving it yet."""
    return root / result_root_relative


def require_canonical_fresh_out_dir(root: Path, out_dir: Path, result_root_relative: str = plan.RESULT_ROOT_RELATIVE) -> Path:
    """Reject alternate paths and symlink roots before any mutable action."""
    expected = canonical_out_dir(root, result_root_relative)
    _require(out_dir.absolute() == expected.absolute(), "PACD out_dir must equal plan.RESULT_ROOT_RELATIVE")
    root_absolute = root.absolute()
    for component in expected.absolute().relative_to(root_absolute).parts:
        root_absolute = root_absolute / component
        _require(not root_absolute.is_symlink(), "PACD result root may not traverse a symlink")
    _require_fresh_out_dir(out_dir)
    return expected


def _review_evidence(root: Path, receipt: Any, paths: tuple[str, ...] = plan.REVIEW_EVIDENCE_PATHS) -> dict[str, str]:
    """Hash review-only bytes once for the attempt without binding run closure."""
    return {relative: receipt.sha256_file(root / relative) for relative in paths}


def verify_expected_sealed_files(root: Path, receipt: Any, expected_sha256: dict[str, str] = plan.EXPECTED_SEALED_SHA256) -> dict[str, str]:
    """Verify static sealed-file literals before any predecessor payload load."""
    observed: dict[str, str] = {}
    for relative, expected in expected_sha256.items():
        path = root / relative
        sidecar = Path(str(path) + ".sha256")
        _require(path.is_file() and sidecar.is_file(), f"sealed predecessor missing: {relative}")
        actual = receipt.sha256_file(path)
        sidecar_sha = sidecar.read_text().split()[0]
        _require(actual == expected, f"sealed predecessor SHA drift: {relative}")
        _require(sidecar_sha == expected, f"sealed predecessor sidecar drift: {relative}")
        observed[relative] = actual
    return observed


def recheck_gpu0_after_attempt(
    before_attempt: dict[str, object],
    *,
    preflight: Callable[[], dict[str, object]] = preflight_gpu0_idle,
) -> dict[str, object]:
    """Make GPU0 idleness/identity a two-point launch invariant."""
    after_attempt = preflight()
    _require(after_attempt == before_attempt, "GPU0 changed between attempt and CUDA binding")
    return after_attempt


def import_torch_after_gpu_recheck(
    before_attempt: dict[str, object],
    *,
    preflight: Callable[[], dict[str, object]] = preflight_gpu0_idle,
    import_torch: Callable[[], Any] = lambda: __import__("torch"),
) -> tuple[dict[str, object], Any]:
    """The ordering trap: no Torch import is permitted before the recheck."""
    after_attempt = recheck_gpu0_after_attempt(before_attempt, preflight=preflight)
    return after_attempt, import_torch()


def publish_failure(out_dir: Path, receipt: Any, payload: dict[str, object]) -> None:
    """Publish only ``failure.json``; ``terminal.json`` is success-exclusive."""
    _require((out_dir / "attempt.json").is_file(), "PACD failure requires an immutable attempt")
    _require(not (out_dir / "terminal.json").exists(), "terminal.json is reserved for success")
    _require(not (out_dir / "failure.json").exists(), "failure.json is immutable once published")
    receipt.write_receipt_transactionally(out_dir / "failure.json", payload)


def _launch_payload(
    *,
    gpu: dict[str, object],
    closure: dict[str, object],
    review_evidence: dict[str, str],
    profile: ExecutionProfile,
    predecessor: dict[str, object] | None,
    args: Any,
) -> dict[str, object]:
    """Pure metadata for ``attempt.json``; intentionally no tensor access."""
    return {
        "schema": profile.schema + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "cell": profile.cell,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ordering": (
            "published before torch import, CUDA initialisation, source arrays, "
            "the canonical checkpoint, or any target path"
        ),
        "arms": [arm.payload() for arm in plan.ARM_SPECS],
        "source_only": True,
        "target_access": False,
        "full_training_authorized": False,
        "budget": {
            "seed": args.seed,
            "steps_per_arm": args.steps,
            "train_batch_size": args.train_batch_size,
            "num_workers": args.num_workers,
            "cpu_threads": 1,
        },
        "device": {
            **gpu,
            "logical_device": "cuda:0",
            "strict_binding": True,
        },
        "governance": {
            "work_order": profile.work_order_relative,
            "design": plan.DESIGN_RELATIVE,
            "initial_state": plan.INITIAL_STATE_RELATIVE,
            "sealed_terminal": plan.SEALED_TERMINAL_RELATIVE,
            "sealed_swa": plan.SEALED_SWA_RELATIVE,
        },
        "sealed_expected_sha256": dict(profile.expected_sealed_sha256),
        "review_evidence_sha256_attempt_only": review_evidence,
        "predecessor": predecessor,
        "source_closure": closure,
        "disclosures": {
            "new_trainable_parameters": 0,
            "inference_graph_changed": False,
            "prediction_consistency_loss": False,
            "posterior_t4": False,
            "reliability_feature": False,
            "learned_gate": False,
            "target_update": False,
            "dopt_selection": False,
            "cdm_state": False,
        },
    }


def _load_execution_stack(root: Path) -> dict[str, Any]:
    """Load the sealed existing source-only stack after attempt publication."""
    from src.cal_aug_v1 import receipts

    # CAL-AUG's helper verifies its own fixed sealed pins before the loader is
    # importlib-loaded; PACD uses it as an unmodified reader/builder only.
    pinned = receipts.verify_pinned_files(root)
    sealed = receipts.verify_sealed_predecessors(root)
    stack = receipts.load_sealed_runner_stack(root, root / "tfpd_exploration")
    stack["pinned_sha256"] = pinned
    stack["sealed_predecessors"] = sealed
    return stack


def _assert_cuda_binding_after_attempt(torch: Any) -> Any:
    """Bind logical ``cuda:0`` only after the immutable attempt exists."""
    _require(torch.cuda.is_available(), "CUDA is unavailable after PACD attempt publication")
    _require(torch.cuda.device_count() == 1, "CUDA visibility drift: expected exactly one device")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _finite_optimizer_state(torch: Any, optimizer: Any) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                return False
    return True


def inherited_lr_at_smoke_step(arm_common: Any, *, step: int, steps_per_epoch: int) -> float:
    """The sealed direct-T4 LR law, evaluated at a global smoke step."""
    return float(arm_common.lr_at_step(int(step), plan.EPOCHS_FOR_LR, int(steps_per_epoch)))


def synchronize_cuda(torch: Any, device: Any) -> None:
    """Route-owned timing boundary; never measure queued GPU work as elapsed."""
    torch.cuda.synchronize(device)


def _run_arm(
    *,
    arm: Any,
    args: Any,
    device: Any,
    stack: dict[str, Any],
    torch: Any,
    dm: Any,
    train_dataset: Any,
    initial_artifact_sha256: str,
) -> dict[str, object]:
    """Run one source arm from the exact canonical initial state.

    The sealed ``build_datamodule`` has already been audited to remove its
    development split.  This function neither constructs a target loader nor
    calls a scorer.
    """
    from mc_maze.multisession_datamodule import SessionBatchSampler
    from torch.utils.data import DataLoader
    import lightning.pytorch as pl

    from .core import paired_train_step, tensor_sha256

    arm_runner = stack["arm_runner"]
    pop_robust = stack["pop_robust"]
    arm_common = stack["arm_common"]
    receipt = stack["receipt"]

    # The caller materializes this source-only datamodule exactly once.  Every
    # arm still resets stochastic streams, model, optimizer and sampler.
    _require(train_dataset is dm.train_dataset, "PACD arm dataset identity drift")
    pl.seed_everything(args.seed, workers=True)

    initial_path = Path(args.initial_state)
    sidecar = Path(str(initial_path) + ".sha256")
    _require(initial_path.is_file() and sidecar.is_file(), "canonical initial state is absent")
    _require(receipt.sha256_file(initial_path) == initial_artifact_sha256,
             "canonical initial state changed after attempt verification")
    initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    model = pop_robust.build_population_robustness_model(seed=args.seed, cell="D")
    model.load_state_dict(initial_payload["state_dict"], strict=True)
    state_before = arm_common.state_sha256(model)
    _require(state_before == initial_payload["state_sha256"], "canonical strict-load SHA drift")
    model.to(device).train()

    encoder_parameters, decoder_parameters = arm_common.param_groups_by_branch(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=arm_common.ADAM_CONSTRUCTOR["lr"],
        betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
        eps=arm_common.ADAM_CONSTRUCTOR["eps"],
        weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
        amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
    )
    sampler = SessionBatchSampler(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, seed=args.seed
    )
    loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    steps_per_epoch = len(sampler)

    rows: list[dict[str, object]] = []
    batch_digests: list[str] = []
    synchronize_cuda(torch, device)
    arm_started = time.perf_counter()
    for step, batch in enumerate(itertools.islice(loader, args.steps)):
        neural, behavior, calibration, session_names, side_features = batch[:5]
        # ``session_names`` is kept as source-order evidence but never used to
        # resolve a filesystem path.
        batch_digests.append(
            hashlib.sha256(
                (tensor_sha256(neural) + tensor_sha256(behavior) + repr(tuple(session_names))).encode("utf-8")
            ).hexdigest()
        )
        lr = inherited_lr_at_smoke_step(arm_common, step=step, steps_per_epoch=steps_per_epoch)
        optimizer.param_groups[0]["lr"] = lr
        step_started = time.perf_counter()
        record = paired_train_step(
            model=model,
            optimizer=optimizer,
            neural=neural.to(device),
            behavior=behavior.to(device),
            calibration=calibration.to(device),
            side_features=side_features.to(device),
            short_m=arm.short_m,
            pad_value=arm_runner.PAD_VALUE,
            encoder_parameters=encoder_parameters,
            decoder_parameters=decoder_parameters,
        )
        record["step"] = step
        record["lr"] = lr
        record["wall_seconds"] = round(time.perf_counter() - step_started, 6)
        record["source_sessions"] = list(session_names)
        rows.append(record)
    synchronize_cuda(torch, device)
    _require(len(rows) == args.steps, "source loader yielded fewer batches than PACD smoke budget")
    _require(_finite_optimizer_state(torch, optimizer), "non-finite Adam state")
    _require(all(row["optimizer_steps"] == 1 for row in rows), "paired optimizer-step drift")
    _require(all(row["valid_bins"] > 0 for row in rows), "invalid zero-valid-bin pair")
    _require(all(row["dropout_pair_equal"] for row in rows), "dropout pair equality drift")
    _require(all(row["rng_short_transition_equal"] for row in rows), "RNG pair equality drift")
    if arm.name == "p0":
        _require(all(row["prediction_pair_equal"] for row in rows), "P0 prediction equality drift")
        _require(all(row["identity_pair_equal"] for row in rows), "P0 identity equality drift")
    arm_elapsed = time.perf_counter() - arm_started
    return {
        "arm": arm.payload(),
        "initial_state_sha256": state_before,
        "steps": rows,
        "steps_per_epoch": steps_per_epoch,
        "batch_sequence_sha256": hashlib.sha256(
            json.dumps(batch_digests, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "optimizer_state_finite": True,
        "final_state_sha256": arm_common.state_sha256(model),
        "wall_seconds": round(arm_elapsed, 6),
        "optimizer_steps_per_second": float(len(rows)) / max(arm_elapsed, 1.0e-12),
        "forwards_per_second": float(2 * len(rows)) / max(arm_elapsed, 1.0e-12),
    }


def execute(*, root: Path, out_dir: Path, args: Any, profile: ExecutionProfile | None = None) -> int:
    """Run the bounded source smoke; callers must explicitly select execution.

    This function deliberately has no fallback to a CPU or another GPU.  A
    resource conflict is a stop condition, not an invitation to share or move
    the work.
    """
    profile = v1_execution_profile() if profile is None else profile
    _require(args.num_workers == 0, "PACD smoke fixes num_workers=0")
    predecessor = None if profile.predecessor_validator is None else profile.predecessor_validator(root)
    require_canonical_fresh_out_dir(root, out_dir, profile.result_root_relative)
    gpu_before_attempt = preflight_gpu0_idle()
    # File-path receipt loading is intentionally package-free here: importing
    # src.tfpd_lane would eagerly import Torch before immutable attempt.
    receipt = load_stdlib_receipt_module(root)

    closure_launch = exact_source_closure(root, receipt, profile.bound_patterns)
    review_evidence = _review_evidence(root, receipt, profile.review_evidence_paths)
    out_dir.mkdir(parents=True)
    attempt_payload = _launch_payload(
        gpu=gpu_before_attempt,
        closure=closure_launch,
        review_evidence=review_evidence,
        profile=profile,
        predecessor=predecessor,
        args=args,
    )
    receipt.write_receipt_transactionally(out_dir / "attempt.json", attempt_payload)
    attempt_sha = receipt.sha256_file(out_dir / "attempt.json")
    started = time.monotonic()
    stack: dict[str, Any] | None = None
    torch_module: Any | None = None
    device: Any | None = None
    sealed_observed: dict[str, str] | None = None
    gpu_after_attempt: dict[str, object] | None = None
    closure_final: dict[str, object] | None = None
    no_target_facts: dict[str, object] = {
        "within_dev_sessions_opened": False,
        "external_sub_m_opened": False,
        "formal_or_organizer_held_data_opened": False,
        "scorer_called": False,
    }
    try:
        # This helper is the ordering trap: it rechecks GPU0 and immediately
        # imports Torch, with no data/checkpoint/CUDA action in between.
        gpu_after_attempt, torch = import_torch_after_gpu_recheck(gpu_before_attempt)
        torch_module = torch
        torch.set_num_threads(1)
        device = _assert_cuda_binding_after_attempt(torch)
        torch.cuda.reset_peak_memory_stats(device)
        stack = _load_execution_stack(root)
        sealed_observed = verify_expected_sealed_files(root, stack["receipt"], profile.expected_sealed_sha256)
        arm_runner = stack["arm_runner"]
        # Materialize source-only windows once; all three arms reuse this
        # immutable dataset and rebuild only their sampler/model/optimizer.
        dm, _a2 = arm_runner.build_datamodule(args)
        _require(
            len(dm.session_splits["train"]) == plan.SOURCE_SESSIONS_REQUIRED,
            "source roster drift: PACD requires the sealed 27-session source roster",
        )
        _require(dm.session_files["val"] == [] and dm.session_files["test"] == [],
                 "non-source path appeared in the PACD datamodule")
        train_dataset = dm.train_dataset
        no_target_facts.update(
            {
                "val_paths_resolved": list(dm.session_files["val"]),
                "test_paths_resolved": list(dm.session_files["test"]),
                "single_source_datamodule_materialization": True,
                "source_roster_n": len(dm.session_splits["train"]),
            }
        )
        initial_artifact_sha256 = sealed_observed[plan.INITIAL_STATE_RELATIVE]
        results = [
            _run_arm(
                arm=arm,
                args=args,
                device=device,
                stack=stack,
                torch=torch,
                dm=dm,
                train_dataset=train_dataset,
                initial_artifact_sha256=initial_artifact_sha256,
            )
            for arm in plan.ARM_SPECS
        ]
        reference_batches = results[0]["batch_sequence_sha256"]
        _require(
            all(item["batch_sequence_sha256"] == reference_batches for item in results[1:]),
            "P0/P1/P2 source batch order drift",
        )
        predecessor_final = None if profile.predecessor_validator is None else profile.predecessor_validator(root)
        _require(predecessor_final == predecessor, "PACD predecessor drifted before terminal")
        closure_final = exact_source_closure(root, receipt, profile.bound_patterns)
        _require(
            closure_final["closure_sha256"] == closure_launch["closure_sha256"],
            "PACD implementation closure drifted while the smoke was running",
        )
        # Synchronize once more immediately before terminal time and peak
        # sampling so the receipt never reports queued GPU work as completed.
        synchronize_cuda(torch, device)
        terminal_wall_seconds = round(time.monotonic() - started, 3)
        terminal = {
            "schema": profile.schema + "_terminal",
            "status": "PACD_SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE",
            "cell": profile.cell,
            "attempt_sha256": attempt_sha,
            "source_only": True,
            "target_access": False,
            "full_training_authorized": False,
            "wall_seconds": terminal_wall_seconds,
            "arms": results,
            "device": {
                "selected": str(device),
                "current_device": int(torch.cuda.current_device()),
                "pre_attempt": gpu_before_attempt,
                "post_attempt": gpu_after_attempt,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
            "sealed_predecessors": {
                "pacd_static_expected_and_observed": sealed_observed,
                "cal_aug_helper_verified": stack["sealed_predecessors"],
            },
            "pinned_sha256": stack["pinned_sha256"],
            "no_target_facts": no_target_facts,
            "source_closure": {"launch": closure_launch, "final": closure_final},
            "predecessor": predecessor_final,
        }
        receipt.write_receipt_transactionally(out_dir / "terminal.json", terminal)
        return 0
    except BaseException as error:  # noqa: BLE001 - failure receipt is mandatory
        if closure_final is None:
            try:
                closure_final = exact_source_closure(root, receipt, profile.bound_patterns)
            except BaseException as closure_error:  # noqa: BLE001
                closure_final = {"rehash_error": f"{type(closure_error).__name__}: {closure_error}"}
        predecessor_final = None
        if profile.predecessor_validator is not None:
            try:
                predecessor_final = profile.predecessor_validator(root)
                _require(predecessor_final == predecessor, "PACD predecessor drifted before failure")
            except BaseException as predecessor_error:  # noqa: BLE001
                predecessor_final = {"revalidation_error": f"{type(predecessor_error).__name__}: {predecessor_error}"}
        failure = {
            "schema": profile.schema + "_failure",
            "status": "CELL_FAILED",
            "cell": profile.cell,
            "attempt_sha256": attempt_sha,
            "terminal_published": False,
            "source_only": True,
            "target_access": False,
            "wall_seconds": round(time.monotonic() - started, 3),
            "device": {
                "pre_attempt": gpu_before_attempt,
                "post_attempt": gpu_after_attempt,
                "selected": None if device is None else str(device),
                "peak_allocated_bytes": (
                    None if torch_module is None or device is None
                    else int(torch_module.cuda.max_memory_allocated(device))
                ),
                "peak_reserved_bytes": (
                    None if torch_module is None or device is None
                    else int(torch_module.cuda.max_memory_reserved(device))
                ),
            },
            "pinned_sha256": None if stack is None else stack.get("pinned_sha256"),
            "sealed_predecessors": (
                None if stack is None
                else {
                    "pacd_static_expected_and_observed": sealed_observed,
                    "cal_aug_helper_verified": stack.get("sealed_predecessors"),
                }
            ),
            "no_target_facts": no_target_facts,
            "source_closure": {"launch": closure_launch, "final": closure_final},
            "predecessor": predecessor_final,
            "failure": {
                "kind": type(error).__name__,
                "detail": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        try:
            publish_failure(out_dir, receipt, failure)
        except BaseException as receipt_error:  # noqa: BLE001
            print(f"PACD failure receipt could not be published: {receipt_error}", file=sys.stderr)
        raise
