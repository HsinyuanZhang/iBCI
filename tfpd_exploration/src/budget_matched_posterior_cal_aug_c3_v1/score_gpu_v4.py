"""GPU-accelerated, tolerance-anchored successor to the V3 CPU score."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from budget_matched_posterior_cal_aug_v1 import c2_smoke
from budget_matched_posterior_cal_aug_v1 import source_audit as audit_v1
from budget_matched_posterior_cal_aug_v1.posterior import array_sha256

from . import score as v1
from . import score_v2 as v2
from . import score_v3 as v3


SCHEMA = "budget_matched_posterior_cal_aug_c2_c3_gpu_score_v4"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/"
    "c2_c3_posterior_gpu_score_v4"
)
WORK_ORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_BUDGET_MATCHED_POSTERIOR_C2_C3_GPU_SCORE_V4_20260830.md"
)
PHYSICAL_GPU_INDEX = 1
EXPECTED_GPU_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090"
VISIBLE_DEVICE = "1"
GPU_BATCH_CANDIDATES = (1024, 512, 128)
CPU_PARITY_BATCH = 32
MAX_ABS_TOLERANCE = 2.0e-6
R2_TOLERANCE = 2.0e-7
EXPECTED_CUBLAS_WORKSPACE = ":4096:8"

EXECUTION_PATHS = tuple(dict.fromkeys((
    *v3.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/score_gpu_v4.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_gpu_score_v4.py",
)))
REVIEW_PATHS = (
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_gpu_score_v4.py",
)


class GPUScoreV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GPUScoreV4Error(message)


def execution_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), EXECUTION_PATHS, "_gpu_v4_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return v1._closure(repository_root.resolve(), REVIEW_PATHS, "_gpu_v4_review_closure")


def _module_origin(name: str) -> str:
    spec = importlib.util.find_spec(name)
    _require(spec is not None and spec.origin is not None, f"module spec absent: {name}")
    return str(Path(spec.origin).resolve())


def _nvidia_smi_profile() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    _require(executable is not None, "nvidia-smi absent")
    completed = subprocess.run(
        [
            executable,
            "--id=" + EXPECTED_GPU_UUID,
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [value.strip() for value in completed.stdout.strip().split(",")]
    _require(len(fields) == 5, "nvidia-smi profile shape drift")
    index, name, uuid, bus, memory_mib = fields
    _require(int(index) == PHYSICAL_GPU_INDEX, "physical GPU index drift")
    _require(name == EXPECTED_GPU_NAME, "GPU name drift")
    _require(uuid == EXPECTED_GPU_UUID, "GPU UUID drift")
    _require(int(memory_mib) >= 24000, "GPU memory profile drift")
    return {
        "physical_index": int(index),
        "name": name,
        "uuid": uuid,
        "pci_bus_id": bus,
        "memory_total_mib": int(memory_mib),
    }


def pre_attempt_runtime_contract(repository_root: Path) -> dict[str, object]:
    repository_root = repository_root.resolve()
    expected_subc = str(repository_root / v3.SUBC_RELATIVE)
    expected_subm = str(repository_root / v3.SUBM_RELATIVE)
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == VISIBLE_DEVICE,
             "CUDA_VISIBLE_DEVICES must be exactly 1")
    _require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
             "CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == EXPECTED_CUBLAS_WORKSPACE,
             "CUBLAS_WORKSPACE_CONFIG drift")
    _require(os.environ.get("SUBC_DATA_ROOT") == expected_subc, "SUBC_DATA_ROOT drift")
    _require(os.environ.get("SUBM_DATA_ROOT") == expected_subm, "SUBM_DATA_ROOT drift")
    for value in (expected_subc, expected_subm):
        info = Path(value).lstat()
        _require(stat.S_ISDIR(info.st_mode) and not Path(value).is_symlink(),
                 f"canonical data root identity drift: {value}")
    expected_src = str((repository_root / "tfpd_exploration/src/__init__.py").resolve())
    expected_models = str((
        repository_root
        / "streaming_calibration_exp/src/models/components/streaming_encoders.py"
    ).resolve())
    src_origin = _module_origin("src")
    model_origin = _module_origin("models.components.streaming_encoders")
    _require(src_origin == expected_src, f"src namespace origin drift: {src_origin}")
    _require(model_origin == expected_models, f"models namespace origin drift: {model_origin}")
    _require("torch" not in sys.modules, "pre-attempt namespace probe imported Torch")
    return {
        "cuda_visible_devices": VISIBLE_DEVICE,
        "cuda_device_order": "PCI_BUS_ID",
        "cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE,
        "subc_data_root": expected_subc,
        "subm_data_root": expected_subm,
        "src_origin": src_origin,
        "streaming_encoder_origin": model_origin,
        "gpu": _nvidia_smi_profile(),
        "torch_imported": False,
    }


class GPUAssetRuntime:
    """The reviewed held-asset reader without the legacy CPU-only model guard."""

    def __init__(self, root: Path, torch_module) -> None:
        from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p
        from src.calibration_gap_v1 import z1_oracle_cells as z1

        self.root = root.resolve()
        self._torch = torch_module
        self.device = torch_module.device("cuda:0")
        self._timing: dict[str, float] = {}
        started = time.perf_counter()
        self.provenance = v1p.load_verified_provenance(self.root)
        self._v1p = v1p
        self._timing["provenance_load_s"] = time.perf_counter() - started
        self.within_roster = tuple(row.session for row in self.provenance.sealed.within)
        self.external_roster = tuple(row.session for row in self.provenance.sealed.external)
        self._reader = v1p._load_exact_module(
            "_gpu_v4_low_cost_calibration_asset_reader",
            self.root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
        )
        self.assets = v1p._assets_from_rows(
            self._reader.derive_fixed_input_assets(
                self.root,
                within_roster=self.within_roster,
                external_roster=self.external_roster,
            )
        )
        self._held_roots = {
            surface: self._reader.HeldDataRoot.from_environment(variable)
            for surface, variable in z1.SURFACE_ENV.items()
        }
        self._equal_score = v1p._load_exact_module(
            "_gpu_v4_equal_score",
            self.root / "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
        )
        self._model = None

    def close(self) -> None:
        for root in reversed(tuple(self._held_roots.values())):
            try:
                root.close()
            except Exception:
                pass

    @property
    def timing(self) -> dict[str, float]:
        return dict(self._timing)


def _decode_on_device(runtime, inputs, model, activity, side, *, device, batch_size: int):
    torch = runtime._torch
    _require(type(batch_size) is int and batch_size > 0, "decode batch invalid")
    with torch.no_grad():
        identity = model.compute_identity(
            activity.unsqueeze(0).to(device), side_features=side.to(device)
        )
        predictions = []
        for offset in range(0, inputs.starts.size, batch_size):
            chunk = inputs.starts[offset:offset + batch_size]
            neural = torch.from_numpy(
                np.stack([inputs.neural[start:start + 50] for start in chunk])
            ).to(device)
            predictions.append(model.decode_with_identity(neural, identity).detach().cpu())
    return torch.cat(predictions, dim=0).contiguous()


def _decode_gpu_with_fallback(runtime, inputs, model, activity, side):
    torch = runtime._torch
    failures: list[dict[str, object]] = []
    for batch_size in GPU_BATCH_CANDIDATES:
        try:
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            prediction = _decode_on_device(
                runtime, inputs, model, activity, side,
                device=runtime.device, batch_size=batch_size,
            )
            torch.cuda.synchronize(0)
            return prediction, {
                "logical_batch_size": batch_size,
                "fallbacks": failures,
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)),
            }
        except torch.cuda.OutOfMemoryError as error:
            failures.append({
                "logical_batch_size": batch_size,
                "error_class": type(error).__name__,
            })
            torch.cuda.empty_cache()
    raise GPUScoreV4Error("all declared GPU decode batches exhausted memory")


def _feature_bundle(inputs, budget: int, *, prior, posterior_normalizer, q_normalizer):
    feature = v1.posterior_side_for_inputs(
        inputs,
        budget,
        prior=prior,
        normalizer=posterior_normalizer,
        q_normalizer=q_normalizer,
    )
    torch = __import__("torch")
    side4 = torch.from_numpy(feature["side4"]).unsqueeze(0)
    q = np.asarray(feature["q"], dtype=np.float32)
    q_zero = np.zeros_like(q, dtype=np.float32)
    permutation = v1.q_shuffle_permutation(
        inputs.surface, inputs.session, budget, inputs.n_units
    )
    q_shuffled = np.ascontiguousarray(q[permutation], dtype=np.float32)
    side5 = {
        "c3_constant": torch.from_numpy(np.ascontiguousarray(
            np.concatenate((feature["side4"], q_zero[:, None]), axis=1)
        )).unsqueeze(0),
        "c3_real": torch.from_numpy(np.ascontiguousarray(
            np.concatenate((feature["side4"], q[:, None]), axis=1)
        )).unsqueeze(0),
        "c3_real_q_shuffle": torch.from_numpy(np.ascontiguousarray(
            np.concatenate((feature["side4"], q_shuffled[:, None]), axis=1)
        )).unsqueeze(0),
    }
    return feature, {"c2": side4, **side5}, {
        "c2": None,
        "c3_constant": q_zero,
        "c3_real": q,
        "c3_real_q_shuffle": q_shuffled,
    }, permutation


def _r2(matched_scorer, prediction, inputs) -> float:
    torch = __import__("torch")
    last = prediction[:, 49, :].contiguous()
    target = torch.from_numpy(inputs.last_targets.copy())
    valid = torch.from_numpy(inputs.last_valid_mask.copy()).bool()
    _require(bool(valid.any().item()), "no valid last-bin target rows")
    return float(matched_scorer.session_r2(last[valid], target[valid]))


def _parity_evidence(
    cpu_prediction,
    gpu_prediction_a,
    gpu_prediction_b,
    *,
    inputs,
    matched_scorer,
    enforce: bool = True,
    max_abs_tolerance: float = MAX_ABS_TOLERANCE,
    r2_tolerance: float = R2_TOLERANCE,
):
    _require(type(max_abs_tolerance) is float and max_abs_tolerance > 0.0,
             "max-abs parity tolerance invalid")
    _require(type(r2_tolerance) is float and r2_tolerance > 0.0,
             "R2 parity tolerance invalid")
    _require(cpu_prediction.shape == gpu_prediction_a.shape == gpu_prediction_b.shape,
             "parity prediction shape drift")
    repeated = bool(np.array_equal(gpu_prediction_a.numpy(), gpu_prediction_b.numpy()))
    max_abs = float((cpu_prediction - gpu_prediction_a).abs().max().item())
    cpu_r2 = _r2(matched_scorer, cpu_prediction, inputs)
    gpu_r2 = _r2(matched_scorer, gpu_prediction_a, inputs)
    r2_abs = abs(gpu_r2 - cpu_r2)
    evidence = {
        "surface": inputs.surface,
        "session": inputs.session,
        "budget": 4,
        "arm": "c2",
        "cpu_batch": CPU_PARITY_BATCH,
        "gpu_batch_candidates": list(GPU_BATCH_CANDIDATES),
        "gpu_repeated_bitwise_equal": repeated,
        "max_abs_prediction_difference": max_abs,
        "max_abs_tolerance": max_abs_tolerance,
        "cpu_r2": cpu_r2,
        "gpu_r2": gpu_r2,
        "absolute_r2_difference": r2_abs,
        "r2_tolerance": r2_tolerance,
        "cpu_prediction_sha256": hashlib.sha256(cpu_prediction.numpy().tobytes()).hexdigest(),
        "gpu_prediction_sha256": hashlib.sha256(gpu_prediction_a.numpy().tobytes()).hexdigest(),
    }
    evidence["passed"] = bool(
        repeated and max_abs <= max_abs_tolerance and r2_abs <= r2_tolerance
    )
    if enforce:
        _require(evidence["passed"], "CPU/GPU parity smoke failed")
    return evidence


def score_gpu_cells(
    repository_root: Path,
    *,
    producer,
    baseline,
    source,
    q_normalizer,
    parity_only: bool = False,
    enforce_parity: bool = True,
    max_abs_tolerance: float = MAX_ABS_TOLERANCE,
    r2_tolerance: float = R2_TOLERANCE,
):
    import torch
    from src.cal_aug_v1 import receipts as cal_receipts
    from src.calibration_gap_v1 import p4_stream_stats as p4

    cal_receipts.verify_sealed_predecessors(repository_root)
    stack = cal_receipts.load_sealed_runner_stack(
        repository_root, repository_root / "tfpd_exploration"
    )
    matched_scorer = stack["matched_scorer"]
    arm_common = stack["arm_common"]
    models, bindings = v1._load_models(repository_root, producer, stack)
    state_before = {arm: v1._state_sha(arm_common, model) for arm, model in models.items()}
    prior = v1.source_prior_from_payload(source["authority"]["source_prior"])
    posterior_normalizer = v1.posterior_normalizer_from_payload(
        source["authority"]["posterior_normalizer"]
    )
    baseline_table = v1._baseline_table(baseline["terminal"])
    runtime = GPUAssetRuntime(repository_root, torch)
    rows: list[dict[str, object]] = []
    materialized: list[dict[str, object]] = []
    try:
        first_surface = v1.SURFACES[0]
        first_session = runtime.within_roster[0]
        first_inputs = p4.materialize_session(runtime, first_surface, first_session)
        feature0, sides0, _q0, _perm0 = _feature_bundle(
            first_inputs, 4,
            prior=prior,
            posterior_normalizer=posterior_normalizer,
            q_normalizer=q_normalizer,
        )
        activity0 = first_inputs.calib[list(feature0["selected"])]
        cpu_prediction = _decode_on_device(
            runtime, first_inputs, models["c2"], activity0, sides0["c2"],
            device=torch.device("cpu"), batch_size=CPU_PARITY_BATCH,
        )
        for model in models.values():
            model.to(runtime.device)
            model.eval()
        gpu_prediction_a, gpu_meta_a = _decode_gpu_with_fallback(
            runtime, first_inputs, models["c2"], activity0, sides0["c2"]
        )
        gpu_prediction_b, gpu_meta_b = _decode_gpu_with_fallback(
            runtime, first_inputs, models["c2"], activity0, sides0["c2"]
        )
        parity = _parity_evidence(
            cpu_prediction, gpu_prediction_a, gpu_prediction_b,
            inputs=first_inputs,
            matched_scorer=matched_scorer,
            enforce=enforce_parity,
            max_abs_tolerance=max_abs_tolerance,
            r2_tolerance=r2_tolerance,
        )
        parity["first_gpu_forward"] = gpu_meta_a
        parity["repeat_gpu_forward"] = gpu_meta_b

        if parity_only:
            return {
                "bindings": bindings,
                "state_before": state_before,
                "parity_smoke": parity,
                "matrix_executed": False,
                "rows": [],
                "runtime_timing": runtime.timing,
                "target_optimizer_backward_update": 0,
            }

        for surface in v1.SURFACES:
            roster = runtime.within_roster if surface == "within" else runtime.external_roster
            for session in roster:
                inputs = (
                    first_inputs
                    if surface == first_surface and session == first_session
                    else p4.materialize_session(runtime, surface, session)
                )
                materialized.append({
                    "surface": surface,
                    "session": session,
                    "n_windows": inputs.n_windows,
                    "n_units": inputs.n_units,
                    "target_sha256": inputs.target_sha256,
                    "calibration_m30_sha256": inputs.calibration_m30_sha256,
                })
                for budget in v1.BUDGETS:
                    for comparator in v1.COMPARATORS:
                        v1._assert_baseline_input(
                            baseline_table[(comparator, session, budget)],
                            runtime,
                            inputs,
                            budget,
                        )
                    feature, cell_sides, consumed_q, permutation = _feature_bundle(
                        inputs,
                        budget,
                        prior=prior,
                        posterior_normalizer=posterior_normalizer,
                        q_normalizer=q_normalizer,
                    )
                    activity = inputs.calib[list(feature["selected"])]
                    for arm in v1.NUMERICAL_ARMS:
                        model_key = "c3_real" if arm == "c3_real_q_shuffle" else arm
                        if (
                            surface == first_surface
                            and session == first_session
                            and budget == 4
                            and arm == "c2"
                        ):
                            prediction = gpu_prediction_a
                            gpu_meta = gpu_meta_a
                        else:
                            prediction, gpu_meta = _decode_gpu_with_fallback(
                                runtime, inputs, models[model_key], activity, cell_sides[arm]
                            )
                        r2 = _r2(matched_scorer, prediction, inputs)
                        target = torch.from_numpy(inputs.last_targets.copy())
                        valid = torch.from_numpy(inputs.last_valid_mask.copy()).bool()
                        _require(bool(torch.isfinite(prediction).all().item()), "GPU prediction nonfinite")
                        rows.append({
                            "arm": arm,
                            "surface": surface,
                            "session": session,
                            "budget": budget,
                            "r2": r2,
                            "n_windows": inputs.n_windows,
                            "n_valid_last_bin": int(valid.sum().item()),
                            "prediction_sha256": hashlib.sha256(prediction.numpy().tobytes()).hexdigest(),
                            "target_sha256": inputs.target_sha256,
                            "selected_indices_sha256": inputs.selected_sha_by_budget[budget],
                            "calibration_prefix_sha256": __import__(
                                "src.cal_aug_v1.deployment", fromlist=["calibration_prefix_digest"]
                            ).calibration_prefix_digest(runtime, inputs, budget),
                            "posterior_raw_t4_sha256": feature["fit"].raw_t4_sha256,
                            "posterior_normalized_side4_sha256": feature["side4_sha256"],
                            "q_raw_sha256": feature["q_raw_sha256"],
                            "q_normalized_sha256": feature["q_sha256"],
                            "consumed_q_sha256": (
                                array_sha256(consumed_q[arm]) if consumed_q[arm] is not None else None
                            ),
                            "consumed_side_sha256": array_sha256(
                                cell_sides[arm].squeeze(0).numpy()
                            ),
                            "q_permutation_sha256": (
                                array_sha256(permutation) if arm == "c3_real_q_shuffle" else None
                            ),
                            "q_permutation_fixed_points": (
                                int(np.sum(permutation == np.arange(inputs.n_units)))
                                if arm == "c3_real_q_shuffle" else None
                            ),
                            "device": "cuda:0",
                            "logical_batch_size": gpu_meta["logical_batch_size"],
                            "batch_fallbacks": gpu_meta["fallbacks"],
                            "wall_seconds": gpu_meta["wall_seconds"],
                            "peak_cuda_bytes": gpu_meta["peak_cuda_bytes"],
                            "target_optimizer_steps": 0,
                            "target_backward_calls": 0,
                            "target_update_calls": 0,
                        })
        state_after = {arm: v1._state_sha(arm_common, model) for arm, model in models.items()}
        _require(state_after == state_before, "GPU scoring changed model state")
        _require(len(rows) == 252 and len(materialized) == 21, "GPU score matrix cardinality drift")
        return {
            "bindings": bindings,
            "state_before": state_before,
            "state_after": state_after,
            "state_unchanged": True,
            "parity_smoke": parity,
            "materialized_sessions": materialized,
            "rows": rows,
            "runtime_timing": runtime.timing,
            "target_optimizer_backward_update": 0,
        }
    finally:
        runtime.close()


def _post_attempt_device_contract(torch) -> dict[str, object]:
    _require(torch.cuda.is_available(), "CUDA unavailable after attempt")
    _require(torch.cuda.device_count() == 1, "visible CUDA device count drift")
    properties = torch.cuda.get_device_properties(0)
    _require(properties.name == EXPECTED_GPU_NAME, "Torch GPU name drift")
    torch_uuid = str(properties.uuid)
    canonical_uuid = torch_uuid if torch_uuid.startswith("GPU-") else "GPU-" + torch_uuid
    _require(canonical_uuid == EXPECTED_GPU_UUID, "Torch GPU UUID drift")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "visible_device": 0,
        "name": properties.name,
        "uuid": canonical_uuid,
        "torch_uuid_raw": torch_uuid,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def execute_reviewed(repository_root: Path) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    predecessor = {
        "v1": v2.validate_failed_v1(repository_root),
        "v2": v3.validate_failed_v2(repository_root),
    }
    runtime_contract = pre_attempt_runtime_contract(repository_root)
    source = c2_smoke.validate_source_authority(repository_root)
    q_normalizer, smoke = v1.reliability_normalizer_from_smoke(repository_root)
    producer = {
        "c2": v1.validate_c2_producer(repository_root),
        "c3_constant": v1.validate_c3_producer(repository_root, "constant"),
        "c3_real": v1.validate_c3_producer(repository_root, "real"),
    }
    baseline = v1.validate_baseline_score(repository_root)
    strict = execution_closure(repository_root)
    review = review_closure(repository_root)
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "V4 result root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "failed_predecessors": predecessor,
        "pre_attempt_runtime_contract": runtime_contract,
        "source_authority": {key: source[key] for key in (
            "root_relative", "attempt_sha256", "source_authority_sha256",
            "terminal_sha256", "closure_sha256",
        )},
        "smoke_predecessor": {key: smoke[key] for key in (
            "root_relative", "attempt_sha256", "terminal_sha256", "execution_closure_sha256",
        )},
        "producers": producer,
        "baseline_score": {key: baseline[key] for key in (
            "root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count",
        )},
        "execution_closure": strict,
        "review_closure": review,
        "surfaces": list(v1.SURFACES),
        "budgets": list(v1.BUDGETS),
        "numerical_arms": list(v1.NUMERICAL_ARMS),
        "gpu_batch_candidates": list(GPU_BATCH_CANDIDATES),
        "parity_tolerances": {
            "max_abs_prediction": MAX_ABS_TOLERANCE,
            "absolute_r2": R2_TOLERANCE,
        },
        "concurrent_v3_is_not_a_predecessor": True,
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    stage = "post_attempt"
    try:
        import torch
        stage = "device_attestation"
        device = _post_attempt_device_contract(torch)
        stage = "checkpoint_validation"
        producer_opened = {
            "c2": v1.validate_c2_producer(repository_root, open_checkpoint_body=True),
            "c3_constant": v1.validate_c3_producer(repository_root, "constant", open_checkpoint_body=True),
            "c3_real": v1.validate_c3_producer(repository_root, "real", open_checkpoint_body=True),
        }
        _require(producer_opened == producer, "producer checkpoint graph drift")
        stage = "parity_then_matched_scoring"
        scored = score_gpu_cells(
            repository_root,
            producer=producer,
            baseline=baseline,
            source=source,
            q_normalizer=q_normalizer,
        )
        matched_scorer = sys.modules["tfpd_lane_matched_scorer"]
        summary = v1.paired_summary(
            scored["rows"], baseline["terminal"], matched_scorer=matched_scorer
        )
        decision = v1.decision_readout(summary)
        stage = "final_revalidation"
        _require(v2.validate_failed_v1(repository_root) == predecessor["v1"], "V1 failure drift")
        _require(v3.validate_failed_v2(repository_root) == predecessor["v2"], "V2 failure drift")
        _require(pre_attempt_runtime_contract(repository_root) == runtime_contract, "runtime drift")
        _require(c2_smoke.validate_source_authority(repository_root) == source, "source authority drift")
        producer_final = {
            "c2": v1.validate_c2_producer(repository_root, open_checkpoint_body=True),
            "c3_constant": v1.validate_c3_producer(repository_root, "constant", open_checkpoint_body=True),
            "c3_real": v1.validate_c3_producer(repository_root, "real", open_checkpoint_body=True),
        }
        _require(producer_final == producer, "producer drift")
        baseline_final = v1.validate_baseline_score(repository_root)
        keys = ("root_relative", "terminal_sha256", "attempt_sha256", "exact_leaf_count")
        _require({k: baseline_final[k] for k in keys} == {k: baseline[k] for k in keys},
                 "baseline drift")
        strict_final = execution_closure(repository_root)
        _require(strict_final["closure_sha256"] == strict["closure_sha256"],
                 "numerical execution closure drift")
        review_final = review_closure(repository_root)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "C2_C3_POSTERIOR_GPU_SCORE_V4_TERMINAL",
            "attempt_sha256": attempt_sha,
            "failed_predecessors": predecessor,
            "pre_attempt_runtime_contract": runtime_contract,
            "post_attempt_device_contract": device,
            "source_authority": attempt["source_authority"],
            "producers": producer,
            "baseline_score": attempt["baseline_score"],
            "execution_closure": {"launch": strict, "final": strict_final, "equal": True},
            "review_closure": v1.review_drift(review, review_final),
            "scored": scored,
            "paired_summary": summary,
            "decision_readout": decision,
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = audit_v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": terminal["status"],
            "terminal_sha256": terminal_sha,
            "parity_smoke": scored["parity_smoke"],
            "decision_readout": decision,
            "review_disposition": terminal["review_closure"]["status"],
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_C3_POSTERIOR_GPU_SCORE_V4_FAILED",
            "attempt_sha256": attempt_sha,
            "failed_predecessors": predecessor,
            "stage": stage,
            "error_class": type(error).__name__,
            "error_sha256": audit_v1._sha_bytes(f"{type(error).__name__}: {error}".encode()),
            "traceback_sha256": audit_v1._sha_bytes(traceback.format_exc().encode()),
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_MODEL_NO_CUDA_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "visible_device": VISIBLE_DEVICE,
        "expected_gpu_uuid": EXPECTED_GPU_UUID,
        "gpu_batch_candidates": list(GPU_BATCH_CANDIDATES),
        "parity_tolerances": {
            "max_abs_prediction": MAX_ABS_TOLERANCE,
            "absolute_r2": R2_TOLERANCE,
        },
        "execute_requires": ["--execute", "--root-reviewed"],
    }
