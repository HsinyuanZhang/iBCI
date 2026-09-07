"""Opaque root-only APFG admission and route-owned production coordinator.

The implementation candidate may validate static source bytes and a *fresh*
canonical root parent, but it does not open a predecessor, model, checkpoint,
data source, CUDA runtime, or result root.  Those operations belong after an
immutable attempt in a separately reviewed executor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import stat
import copy
from typing import Mapping

from . import plan


class AdmissionError(RuntimeError):
    pass


_ISSUER_TOKEN = object()
_ISSUED_IDS: set[int] = set()


@dataclass(frozen=True)
class _SyntheticProductionRuntime:
    """Private typed test seam; public admission never accepts callbacks.

    The seam is intentionally a concrete object rather than a caller mapping.
    It exercises the production capability consumption, immutable attempt and
    terminal/failure topology without permitting a public execution API.
    """
    launch_payload: Mapping[str, object]
    body_builder: object
    terminal_builder: object
    final_validator: object | None = None


@dataclass(frozen=True)
class _FutureCapability:
    """Opaque one-shot admission record; constructor token is private."""
    repo_root: Path
    result_root: Path
    parent_identity: tuple[int, int]
    closure_sha256: str
    corrected_body_sha256: tuple[tuple[str, str], ...]
    _issuer: object
    consumed: bool = False


def static_admission(repo_root: Path) -> dict[str, object]:
    authority = plan.validate_static(Path(repo_root))
    closure = plan.closure(Path(repo_root))
    return {"schema": plan.SCHEMA, "stage": "stage0_no_data_no_cuda", "authority": authority,
            "closure_sha256": plan.sha256_bytes(plan.canonical_json(closure)),
            "public_cli_can_mint": False, "root_only_capability_possible": plan.LIVE_PRODUCER_LITERALS is not None,
            "reason": "runtime remains root-only and attempt-first"}


def _canonical_fresh_root(repo_root: Path) -> tuple[Path, tuple[int, int]]:
    root = Path(repo_root).absolute()
    target = root / plan.RESULT_ROOT_RELATIVE
    parent = target.parent
    try:
        info = os.lstat(parent)
    except FileNotFoundError as error:
        raise AdmissionError("APFG canonical result parent does not exist") from error
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or os.path.lexists(target)):
        raise AdmissionError("APFG canonical result root is unavailable or not fresh")
    return target, (int(info.st_dev), int(info.st_ino))


def _require_gpu0_environment() -> None:
    # No CUDA API is touched during issuance.  Future post-attempt runtime
    # attestation must additionally bind physical GPU0 UUID/device state.
    observed = {name: os.environ.get(name) for name in plan.LIVE_ENV}
    if observed != plan.LIVE_ENV:
        raise AdmissionError(f"APFG live environment drift: {observed}")


def issue_live_capability(repo_root: Path, *, token: object) -> _FutureCapability:
    """Root-only pre-attempt issuer; public CLI has no access to ``token``."""
    if token is not _ISSUER_TOKEN:
        raise AdmissionError("opaque APFG issuer token mismatch")
    if plan.LIVE_PRODUCER_LITERALS is None:
        raise AdmissionError("APFG live literals are deliberately unbound")
    _require_gpu0_environment()
    plan.validate_static(Path(repo_root))
    target, parent_identity = _canonical_fresh_root(Path(repo_root))
    closure_sha = plan.sha256_bytes(plan.canonical_json(plan.closure(Path(repo_root))))
    cap = _FutureCapability(repo_root=Path(repo_root).absolute(), result_root=target,
                            parent_identity=parent_identity, closure_sha256=closure_sha,
                            corrected_body_sha256=tuple(sorted(plan.CORRECTED_BODIES.items())),
                            _issuer=_ISSUER_TOKEN)
    _ISSUED_IDS.add(id(cap))
    return cap


def _mint_synthetic_capability(repo_root: Path) -> _FutureCapability:
    """Private test seam only; it is deliberately absent from ``__all__``."""
    return issue_live_capability(repo_root, token=_ISSUER_TOKEN)


def consume_for_post_attempt_runtime(capability: _FutureCapability) -> None:
    """Check one-shot/current-closure/root identity before future runtime work."""
    if (not isinstance(capability, _FutureCapability) or capability._issuer is not _ISSUER_TOKEN
            or id(capability) not in _ISSUED_IDS or capability.consumed):
        raise AdmissionError("APFG capability invalid, forged, or reused")
    _require_gpu0_environment()
    observed = plan.sha256_bytes(plan.canonical_json(plan.closure(capability.repo_root)))
    if observed != capability.closure_sha256:
        raise AdmissionError("APFG current closure drift")
    target, parent_identity = _canonical_fresh_root(capability.repo_root)
    if target != capability.result_root or parent_identity != capability.parent_identity:
        raise AdmissionError("APFG canonical root/parent identity drift")
    if capability.corrected_body_sha256 != tuple(sorted(plan.CORRECTED_BODIES.items())):
        raise AdmissionError("APFG corrected predecessor literal drift")
    object.__setattr__(capability, "consumed", True)


def _revalidate_reserved(capability: _FutureCapability) -> None:
    """Final/failure check after immutable root reservation (never fresh)."""
    _require_gpu0_environment()
    plan.validate_static(capability.repo_root)
    observed = plan.sha256_bytes(plan.canonical_json(plan.closure(capability.repo_root)))
    if observed != capability.closure_sha256:
        raise AdmissionError("APFG final closure drift")
    info = os.lstat(capability.result_root)
    parent = os.lstat(capability.result_root.parent)
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != capability.parent_identity):
        raise AdmissionError("APFG reserved root/parent identity drift")


def _gpu0_after_attempt() -> dict[str, object]:
    """Physical-GPU0-only runtime attestation; no fallback is possible."""
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise AdmissionError("APFG GPU0 runtime is unavailable")
    torch.cuda.init()
    if torch.cuda.current_device() != 0 or torch.cuda.device_count() != 1:
        raise AdmissionError("APFG logical CUDA device topology drift")
    props = torch.cuda.get_device_properties(0)
    raw = str(props.uuid)
    canonical = raw if raw.startswith("GPU-") else f"GPU-{raw}"
    expected = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
    if canonical != expected:
        raise AdmissionError("APFG physical GPU0 UUID drift")
    return {"cuda_visible_devices": "0", "logical_device": 0, "physical_device": 0,
            "device_uuid_raw": raw, "device_uuid_canonical": canonical,
            "cuda_initialized": bool(torch.cuda.is_initialized())}


def _target_records_once(*, datamodule: object, pooled: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Reuse the reviewed narrow held-out builder, never another PIT stack."""
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as physical
    from tfpd_exploration.src.cdm_p1_m2_local_v1 import replay as replay
    append = physical._append_heldout_to_prepared_datamodule(datamodule=datamodule)
    surfaces = {
        "external_post30_local": (datamodule.val_heldout_dataset, datamodule.val_calib_heldout_sessions),
        "within_post30": (datamodule.train_dataset, datamodule.train_calib_heldin_sessions),
    }
    records: dict[str, object] = {}
    for surface, (dataset, raw_sessions) in surfaces.items():
        sessions = tuple(sorted(dataset.calib_trialized_neural_features))
        expected = 6 if surface == "external_post30_local" else 7
        if len(sessions) != expected:
            raise AdmissionError(f"APFG {surface} roster drift")
        for session in sessions:
            record = physical._record_from_session_material(surface=surface, session=session, dataset=dataset,
                                                              raw_sessions=raw_sessions, replay=replay)
            comparator = pooled.get(str(record["key"]))
            if not isinstance(comparator, Mapping):
                raise AdmissionError("APFG sealed POOLED comparator record absent")
            physical._bind_pooled_comparator(record, comparator)
            records[str(record["key"])] = record
    if len(records) != 13:
        raise AdmissionError("APFG exact 13-session materialization drift")
    return {"records": records, "append_heldout_evidence": append}


def _move_frozen_target_models_to_cpu(*, torch: object, zero_module: object, learned_module: object,
                                      selection_module: object, selection_optimizer: object | None = None) -> dict[str, object]:
    """End the GPU-only source phase before any target materialization.

    The sealed POOLED comparator is CPU-scored.  Its byte-level ZERO sentinel
    is meaningful only on that same CPU arithmetic surface, so no target
    neural/identity/decode tensor is permitted on CUDA.
    """
    torch.cuda.synchronize(torch.device("cuda:0"))
    before_allocated = int(torch.cuda.memory_allocated(torch.device("cuda:0")))
    before_reserved = int(torch.cuda.memory_reserved(torch.device("cuda:0")))
    alpha = float(learned_module.student.id_encoder.alpha.detach().cpu())
    zero_alpha = float(zero_module.student.id_encoder.alpha.detach().cpu())
    if zero_alpha != 0.0 or math.copysign(1.0, zero_alpha) < 0.0:
        raise AdmissionError("APFG zero module lost exact positive-zero anchor before CPU handoff")
    for module in (zero_module, learned_module, selection_module):
        # ``set_alpha_training(False)`` controls the adapter's branch mode,
        # not Parameter.requires_grad.  Target scoring is strictly inference:
        # freeze every inherited/scalar parameter and discard any source grad
        # before moving the three resident source models off GPU0.
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        module.to(torch.device("cpu")); module.eval()
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise AdmissionError("APFG target CPU model lost frozen topology")
        if any(str(parameter.device) != "cpu" for parameter in module.parameters()):
            raise AdmissionError("APFG target model did not fully transfer to CPU")
    if selection_optimizer is not None:
        # The selection optimizer is source-only evidence.  Clearing its state
        # releases alpha's GPU moment tensors and makes target use impossible.
        selection_optimizer.state.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(torch.device("cuda:0"))
    after_allocated = int(torch.cuda.memory_allocated(torch.device("cuda:0")))
    after_reserved = int(torch.cuda.memory_reserved(torch.device("cuda:0")))
    return {"source_training_device": "cuda:0", "target_scoring_device": "cpu",
            "cuda_initialized_after_source": bool(torch.cuda.is_initialized()),
            "target_cuda_tensors_permitted": False, "learned_refit_alpha_before_cpu_score": alpha,
            "zero_alpha_positive_zero_before_cpu_score": True,
            "target_cpu_decode_batch_size": plan.TARGET_CPU_DECODE_BATCH_SIZE,
            "gpu0_allocator_before_release": {"allocated": before_allocated, "reserved": before_reserved},
            "gpu0_allocator_after_release": {"allocated": after_allocated, "reserved": after_reserved},
            "released_source_gpu_objects": ["selection_module", "selection_optimizer_state", "zero_module", "refit_module"]}


def _execute_synthetic_production(capability: _FutureCapability, runtime: _SyntheticProductionRuntime) -> tuple[str | None, str | None]:
    """Actual production admission/lifecycle with a typed no-data runtime."""
    consume_for_post_attempt_runtime(capability)
    from . import lifecycle
    progress: dict[str, object] = {"stage": "reserved", "corrected_graph_attempted": False,
                                   "corrected_graph_verified": False, "cuda_attempted": False,
                                   "cuda_verified": False, "checkpoint_attempted": False,
                                   "checkpoint_opened": False, "source_attempted": False,
                                   "source_complete": False, "pooled_attempted": False,
                                   "pooled_verified": False, "target_attempted": False,
                                   "target_opened": False, "target_complete": False}
    attempt = {"schema": f"{plan.SCHEMA}_attempt_v1", "status": "ATTEMPT_RESERVED",
               "closure_sha256": capability.closure_sha256, "corrected_predecessor": dict(capability.corrected_body_sha256),
               "cuda_initialized": False, "target_opened": False, "checkpoint_opened": False,
               "gpu_profile": {"cuda_visible_devices": "0", "physical_device": 0}}
    def launch() -> Mapping[str, object]:
        _revalidate_reserved(capability); progress["cuda_attempted"] = True; progress["cuda_verified"] = True
        return dict(runtime.launch_payload)
    def bodies(artifact: object) -> Mapping[str, str]:
        return runtime.body_builder(artifact, progress)
    def terminal(published: Mapping[str, str]) -> Mapping[str, object]:
        _revalidate_reserved(capability)
        if runtime.final_validator is not None:
            runtime.final_validator()
        return dict(runtime.terminal_builder(published, progress))
    return lifecycle.execute(capability.repo_root, attempt=attempt, launch=launch, bodies=bodies,
                             terminal=terminal, progress=lambda: dict(progress),
                             failure_revalidate=(lambda: _revalidate_reserved(capability)))


def execute_production(capability: _FutureCapability, *, _synthetic_runtime: _SyntheticProductionRuntime | None = None) -> tuple[str | None, str | None]:
    """Run the single APFG graph; public CLI intentionally cannot call this.

    The implementation imports model/data modules only inside ``bodies``,
    after the lifecycle has made ``attempt.json`` immutable.  It preserves one
    PIT construction for source fitting and the later 13-record deployment.
    """
    if _synthetic_runtime is not None:
        return _execute_synthetic_production(capability, _synthetic_runtime)
    consume_for_post_attempt_runtime(capability)
    from . import lifecycle
    progress: dict[str, object] = {"stage": "reserved", "target_opened": False, "checkpoint_opened": False,
                                   "cuda_initialized": False, "published_prefix": [],
                                   "corrected_graph_attempted": False, "corrected_graph_verified": False,
                                   "cuda_attempted": False, "cuda_verified": False,
                                   "source_attempted": False, "source_complete": False,
                                   "pooled_attempted": False, "pooled_verified": False,
                                   "target_attempted": False, "target_complete": False}
    attempt = {"schema": f"{plan.SCHEMA}_attempt_v1", "status": "ATTEMPT_RESERVED",
               "closure_sha256": capability.closure_sha256, "corrected_predecessor": dict(capability.corrected_body_sha256),
               "cuda_initialized": False, "target_opened": False, "checkpoint_opened": False,
               "gpu_profile": {"cuda_visible_devices": "0", "physical_device": 0}}
    payload: dict[str, object] = {}

    def launch() -> Mapping[str, object]:
        progress["cuda_attempted"] = True
        device = _gpu0_after_attempt(); progress.update(device); progress["cuda_verified"] = True; progress["stage"] = "launch"
        return {"schema": f"{plan.SCHEMA}_launch_v1", "attempt_closure_sha256": capability.closure_sha256, **device}

    def bodies(artifact: object) -> Mapping[str, str]:
        from . import binding, laws, runtime, scoring, source_replay, training
        from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import binding as pooled_binding
        progress["corrected_graph_attempted"] = True
        binding.validate_operator_corrected_graph(capability.repo_root)
        progress["corrected_graph_verified"] = True
        progress["checkpoint_attempted"] = True
        progress["source_attempted"] = True
        import torch
        # This is intentionally after immutable attempt/launch publication and
        # immediately before the one PIT construction.  Receipt evidence makes
        # plan.SEED=42 an executed law rather than a plan-only literal.
        seed_evidence = runtime.establish_post_attempt_seed_evidence(torch=torch)
        prepared = training.prepare_selected_t4_apfg_after_attempt(repo_root=capability.repo_root, device="cuda:0",
                                                                    launch_attestation=dict(progress))
        progress["checkpoint_opened"] = True
        source_dataset = prepared.datamodule.train_dataset
        source_raw = prepared.datamodule.train_calib_heldin_sessions
        names = tuple(sorted(source_dataset.calib_trialized_neural_features))
        if len(names) != plan.SOURCE_SESSION_COUNT:
            raise AdmissionError("APFG source roster is not exact seven")
        materials = {name: source_replay.materialize_pooled_g00m_source_session(dataset=source_dataset,
                    raw_sessions=source_raw, session=name) for name in names}
        zero_module = copy.deepcopy(prepared.module)
        trained = training.train_select_and_refit(prepared=prepared, materials=materials)
        progress["source_complete"] = True
        runtime_evidence = {
            "attempt_published_before_checkpoint_data_cuda": True, "pit_materializations": 1,
            "strict_load_before_adapter": True, "inherited_trainable_parameter_names": [],
            "alpha_trainable_parameter_names": ["id_encoder.alpha"], "inherited_eval": True,
            "dropout_calls": 0, "optimizer_parameter_names": ["id_encoder.alpha"],
            "source_split": {"fit": list(trained["fit_sessions"]), "validation": list(trained["validation_sessions"])},
            "selection_epochs": list(range(1, plan.EPOCHS + 1)), "refit_alpha_positive_zero": True,
            "training_loss": plan.TRAINING_LOSS, "behavior_scaling_factor": plan.BEHAVIOR_SCALING_FACTOR,
            "predict_scaled_behavior": True, "teacher_forward_calls": 0, "seed_evidence": seed_evidence,
        }
        runtime.validate_post_attempt_runtime_evidence(runtime_evidence)
        source_payload = source_replay.source_authority_summary(tuple(materials.values()))
        source_payload.update({"schema": f"{plan.SCHEMA}_source_authority_v1", "selected_t4_checkpoint_sha256":
                               plan.SELECTED_T4_POOLED_CHECKPOINT_SHA256, "training_loss": plan.TRAINING_LOSS,
                               "teacher_forward_calls": 0, "source_selection": trained["selection"],
                               "selection_rows": trained["selection_rows"], "runtime_evidence": runtime_evidence,
                               "runtime_evidence_validated": True, "prepared_evidence": prepared.evidence})
        source_sha = artifact.publish_json("source_authority.json", source_payload)
        alpha_payload = {key: value for key, value in trained.items() if key not in {"refit_module", "refit_adapter"}}
        alpha_payload.update({"schema": f"{plan.SCHEMA}_alpha_selection_v1", "refit_alpha": trained["refit_alpha"]})
        alpha_sha = artifact.publish_json("alpha_selection.json", alpha_payload)
        progress["published_prefix"] = ["attempt.json", "launch.json", "source_authority.json", "alpha_selection.json"]
        if trained["selection"]["source_safety_gate_passed"] is not True:
            raise AdmissionError("APFG source safety gate failed before target materialization")
        zero_module.student.id_encoder.set_alpha_training(False)
        if not __import__("tfpd_exploration.src.m2_anchored_postfusion_gate_v1.adapter", fromlist=["exact_positive_zero"]).exact_positive_zero(zero_module.student.id_encoder.alpha):
            raise AdmissionError("APFG zero module lost exact positive-zero anchor before target")
        progress["pooled_attempted"] = True
        pooled_witness = pooled_binding.validate_pooled_comparator_score(
            capability.repo_root / "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1")
        progress["pooled_verified"] = True
        # CPU is the historical POOLED numeric authority.  Transfer before a
        # target dataset/window is materialized so target never enters GPU0.
        device_evidence = _move_frozen_target_models_to_cpu(torch=torch, zero_module=zero_module,
                                                             learned_module=trained["refit_module"],
                                                             selection_module=prepared.module,
                                                             selection_optimizer=prepared.optimizer)
        # A handoff failure is still source-only.  Mark target attempted only
        # once all resident source objects have been frozen and released.
        progress["target_attempted"] = True
        materialized = _target_records_once(datamodule=prepared.datamodule, pooled=pooled_witness["comparators"])
        progress["target_opened"] = True
        progress["target_complete"] = True
        public_records = {name: {key: value for key, value in record.items() if key != "_runtime"}
                          for name, record in materialized["records"].items()}
        input_payload = {"schema": f"{plan.SCHEMA}_input_authority_v1", "records": public_records,
                         "append_heldout_evidence": materialized["append_heldout_evidence"],
                         "source_authority_sha256": source_sha, "activity_authority": plan.ACTIVITY_AUTHORITY}
        input_sha = artifact.publish_json("input_authority.json", input_payload)
        rows = scoring.score_locked_records(torch=torch, zero_module=zero_module, learned_module=trained["refit_module"],
                                            records=materialized["records"], pooled_comparators=pooled_witness["comparators"],
                                            device=torch.device("cpu"), decode_batch_size=plan.TARGET_CPU_DECODE_BATCH_SIZE)
        derived = laws.recompute(rows, pooled_witness["comparators"])
        score_sha = artifact.publish_json("score.json", {"schema": f"{plan.SCHEMA}_score_v1", "rows": rows, **derived,
            "source_authority_sha256": source_sha, "alpha_selection_sha256": alpha_sha,
            "target_scoring_parameter_updates": 0, "target_updates": 0, "cuda_initialized": True,
            "device_separation": device_evidence})
        progress["published_prefix"] = ["attempt.json", "launch.json", "source_authority.json", "alpha_selection.json", "input_authority.json", "score.json"]
        payload.update({"source_authority.json": source_sha, "alpha_selection.json": alpha_sha,
                        "input_authority.json": input_sha, "score.json": score_sha, "row_count": len(rows),
                        "governing_gate": derived["governing_gate"], "refit_alpha": trained["refit_alpha"],
                        "source_alpha_optimizer_updates": int(trained["fit_optimizer_updates"]) + int(trained["refit_optimizer_updates"]),
                        "device_separation": device_evidence})
        return {key: str(value) for key, value in payload.items() if key.endswith(".json")}

    def terminal(published: Mapping[str, str]) -> Mapping[str, object]:
        _revalidate_reserved(capability)
        from . import binding
        binding.validate_operator_corrected_graph(capability.repo_root)
        from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import binding as pooled_binding
        pooled_final = pooled_binding.validate_pooled_comparator_score(
            capability.repo_root / "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1")
        if len(pooled_final.get("comparators", {})) != 13:
            raise AdmissionError("APFG final POOLED comparator witness drift")
        return {"current_closure_sha256": capability.closure_sha256, "corrected_predecessor": dict(capability.corrected_body_sha256),
                "source_authority_sha256": published["source_authority.json"], "alpha_selection_sha256": published["alpha_selection.json"],
                "input_authority_sha256": published["input_authority.json"], "score_sha256": published["score.json"],
                "row_count": payload["row_count"], "source_alpha_optimizer_updates": payload["source_alpha_optimizer_updates"],
                "target_parameter_updates": 0, "target_updates": 0,
                "cuda_initialized": True, "source_training_device": "cuda:0", "target_scoring_device": "cpu",
                "target_cuda_tensors_permitted": False, "gpu_profile": {"physical_device": 0, "cuda_visible_devices": "0"},
                "governing_gate": payload["governing_gate"], "limitations": ["single_scalar_alpha", "exploratory_unmatched_pooled_control"]}
    def failure_revalidate() -> None:
        _revalidate_reserved(capability)
        from . import binding
        binding.validate_operator_corrected_graph(capability.repo_root)
        from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import binding as pooled_binding
        pooled_binding.validate_pooled_comparator_score(
            capability.repo_root / "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1")
    return lifecycle.execute(capability.repo_root, attempt=attempt, launch=launch, bodies=bodies,
                             terminal=terminal, progress=lambda: dict(progress), failure_revalidate=failure_revalidate)


__all__ = ("AdmissionError", "static_admission", "issue_live_capability", "consume_for_post_attempt_runtime", "execute_production")
