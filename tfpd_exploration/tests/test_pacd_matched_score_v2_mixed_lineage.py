"""Focused no-data/no-CUDA tests for the deferred mixed-lineage profile."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paired_anchored_calibration_dropout_score_v1 import binding as v1_binding, lifecycle, score
from src.paired_anchored_calibration_dropout_score_v2_mixed_lineage import binding, plan, smoke


class _Calib:
    def __getitem__(self, key):
        return ("calib", tuple(key))


class _Inputs:
    def __init__(self, surface, session):
        self.surface, self.session, self.calib = surface, session, _Calib()
        self.selected_by_budget = {30: tuple(range(30)), 10: tuple(range(10)), 4: tuple(range(4))}
        self.selected_sha_by_budget = {m: f"selected-{session}-{m}" for m in plan.BUDGET_ORDER}
        self.side_sha_by_budget = {m: f"side-{session}-{m}" for m in plan.BUDGET_ORDER}
        self.ridge_fit_by_budget = {m: {"raw_t4_sha256": f"raw-{session}-{m}"} for m in plan.BUDGET_ORDER}
        self.target_sha256, self.valid_mask_sha256 = f"target-{session}", f"valid-{session}"
        self.neural_sha256, self.query_window_starts_sha256 = f"neural-{session}", f"starts-{session}"


class _Runtime:
    within_roster = tuple(f"within-{i}" for i in range(6))
    external_roster = tuple(f"external-{i}" for i in range(15))


class _Receipt:
    def write_receipt_transactionally(self, path, payload):
        from src.tfpd_lane.receipt import write_receipt_transactionally
        write_receipt_transactionally(Path(path), payload)


def _sealed(receipt, path: Path, payload: dict) -> tuple[str, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_receipt_transactionally(path, payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, {"name": path.name, "sha256": digest, "sidecar": path.name + ".sha256"}


def _v4_attestation(arm, *, active=False):
    return {
        "stage": "active_current_pid" if active else "idle_pre_cuda",
        "physical_index": arm.physical_index, "uuid": arm.expected_uuid,
        "name": "NVIDIA GeForce RTX 3090", "driver_version": "535.309.01",
        "pci_bus_id": arm.expected_pci_bus_id, "cuda_visible_devices": arm.cuda_visible_devices,
        "compute_processes": ([{"pid": 123, "process_name": "python", "used_memory": "1 MiB"}] if active else []),
        "idle": not active, "preflight_uses_torch": False,
    }


def _v4_context():
    return {"stage": "active_current_pid", "visible_device_count": 1, "current_logical_device": 0,
            "logical_device": "cuda:0", "cuda_initialized": True, "current_pid": 123,
            "selected_gpu_compute_pids": [123]}


def _v4_scheduler(arm, *, tracked=None):
    profile = binding._scheduler_profile(arm)
    tracked = tracked or {"identity": "none", "pid": None, "live": False, "observed_affinity": []}
    def observation(monotonic_ns):
        return {"observed_affinity": list(profile["logical_cpu_affinity"]),
                "tracked_other_route": dict(tracked),
                "overlap": sorted(set(profile["logical_cpu_affinity"]).intersection(tracked["observed_affinity"])),
                "host_pressure": {"schema": "pacd_v4_host_pressure_v1", "observed_monotonic_ns": monotonic_ns,
                    "mem_available_bytes": 17_179_869_184, "mem_available_floor_bytes": 17_179_869_184,
                    "memory_psi_some_avg10": 0.1, "memory_psi_some_avg10_ceiling": 0.1,
                    "memory_psi_full_avg10": 0.0, "memory_psi_full_avg10_ceiling": 0.0,
                    "swap_total_bytes": 100, "swap_free_bytes": 0, "pass": True}}
    before, after, final = observation(10), observation(20), observation(30)
    return {"attempt": {"profile": profile, "before_attempt": before},
            "launch": {"profile": profile, "after_attempt": after},
            "terminal": {"profile": profile, "before_attempt": before,
                         "after_attempt": after, "final": final}}


def _v4_fixture(tmp_path: Path, *, arm_name="P1", p1_mismatches=0, rng_drift=False,
                checkpoint_epoch_override=None, bad_context=False, scheduler_mutation=None, tracked=None,
                terminal_device_mutation=None):
    """Write one actual-shaped V4 artifact graph, but no tensors/data/CUDA."""
    receipt = _Receipt()
    base = plan.P1_ROOT if arm_name == "P1" else plan.P2_ROOT
    out = tmp_path / base
    p0_relative = plan.P0_ROOT
    (tmp_path / p0_relative).mkdir(parents=True)
    digest = "a" * 64
    raw_arm = (
        binding.V4AdmissionProducerArm("P1", plan.P1_ROOT, digest, digest, digest, digest, digest, digest,
            (digest,) * 4, (digest,) * 48, digest, 4, 0,
            "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9", "00000000:01:00.0", "0")
        if arm_name == "P1" else
        binding.V4AdmissionProducerArm("P2", plan.P2_ROOT, digest, digest, digest, digest, digest, digest,
            (digest,) * 4, (digest,) * 48, digest, 10, 1,
            "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86", "00000000:03:00.0", "1")
    )
    identity = binding._held_directory_identity(tmp_path, p0_relative)
    p0 = {"root_relative": p0_relative, "attempt_sha256": digest, "launch_sha256": digest,
          "source_authority_sha256": digest, "terminal_sha256": digest, "manifest_sha256": digest,
          "swa_sha256": digest, "epoch_sha256s": [digest] * 48, "checkpoint_sha256s": [digest] * 4,
          "historical_v3_closure_sha256": plan.V3_HISTORICAL_CLOSURE, "v2_failure_sha256": plan.V2_FAILURE_SHA,
          "v2_failure": {"relative": "v2/failure", "failure_sha256": plan.V2_FAILURE_SHA, "topology": [], "v1_predecessor": {}},
          "root_identity": identity}
    predecessor = {
        "schema": "pacd_p1_p2_admission_multigpu_v4_predecessor_v1", "v2_failure": p0["v2_failure"],
        "required_v3_p0": {
            "schema": "pacd_v3_p0_admission_witness_v1", "root_relative": p0_relative, "root_identity": identity,
            "attempt": {"name": "attempt.json", "sha256": digest}, "launch": {"name": "launch.json", "sha256": digest},
            "source_authority": {"name": "source_authority.json", "sha256": digest},
            "epochs": [{"name": f"epoch{i:03d}.json", "sha256": digest} for i in range(48)],
            "checkpoints": [{"name": f"epoch{i:03d}.pt", "sha256": digest, "epoch": i} for i in (44, 45, 46, 47)],
            "swa": {"name": "swa_final4.pt", "sha256": digest}, "manifest": {"name": "manifest.json", "sha256": digest},
            "terminal": {"name": "terminal.json", "sha256": digest},
            "historical_v3_closure_sha256": plan.V3_HISTORICAL_CLOSURE, "v2_failure_sha256": plan.V2_FAILURE_SHA,
        },
    }
    profile = {"identity": f"v4-{arm_name.lower()}-gpu{raw_arm.physical_index}", "physical_index": str(raw_arm.physical_index), "expected_uuid": raw_arm.expected_uuid,
               "expected_name": "NVIDIA GeForce RTX 3090", "expected_driver": "535.309.01",
               "expected_pci_bus_id": raw_arm.expected_pci_bus_id, "cuda_visible_devices": raw_arm.cuda_visible_devices, "logical_device": "cuda:0"}
    closure = {"closure_sha256": "b" * 64}
    scheduler = _v4_scheduler(raw_arm, tracked=tracked)
    if scheduler_mutation:
        scheduler_mutation(scheduler)
    attempt_body = {"schema": "pacd_p1_p2_admission_multigpu_v4_attempt", "status": "ATTEMPT_PUBLISHED",
                    "cell": "PACD_P1_P2_ADMISSION_MULTIGPU_V4", "arm": arm_name.lower(),
                    "arm_identity": {"short_m": raw_arm.short_m, "root": base}, "predecessor": predecessor,
                    "budget": {"epochs": 48, "steps_per_epoch": 33925, "total_steps": 1628400,
                               "seed": 42, "batch_size": 32, "num_workers": 0}, "target_access": False,
                    "source_closure": closure,
                    "device": {"profile": profile,
                    "observer_kind": "extended-fixed-device-production-v1", "issuance_idle": _v4_attestation(raw_arm),
                    "pre_attempt_idle": _v4_attestation(raw_arm)}, "scheduler": scheduler["attempt"]}
    attempt_sha, attempt_desc = _sealed(receipt, out / "attempt.json", attempt_body)
    authority_body = {"attempt_sha256": attempt_sha, "target_access": False, "source_roster_n": 27, "val": [], "test": [],
                      "ordered_source_roster": [f"source-{i}" for i in range(27)],
                      "normalizers": {"behavior": "n" * 64}, "t4_authority_sha256": "t" * 64,
                      "window": {"window_size": 50}, "initial_state": {"state_sha256": "s" * 64},
                      "sampler": {"class": "SessionBatchSampler", "batch_size": 32, "shuffle": True,
                                  "seed": 42, "num_workers": 0, "steps_per_epoch": 33925,
                                  "window_indices_sha256": "c" * 64, "batched_indices_sha256": "d" * 64}}
    p0["source_contract"] = {key: authority_body[key] for key in ("ordered_source_roster", "normalizers", "t4_authority_sha256", "window", "initial_state", "sampler")}
    authority_sha, authority_desc = _sealed(receipt, out / "source_authority.json", authority_body)
    context = _v4_context()
    if bad_context:
        context["logical_device"] = "cuda:1"
    launch_body = {"schema": "pacd_p1_p2_admission_multigpu_v4_launch", "attempt": attempt_desc,
                   "source_authority": authority_desc, "device": {"profile": profile,
                   "observer_kind": "extended-fixed-device-production-v1", "post_attempt_idle": _v4_attestation(raw_arm),
                   "context_bound": context}, "scheduler": scheduler["launch"]}
    launch_sha, launch_desc = _sealed(receipt, out / "launch.json", launch_body)
    epoch_descs, epoch_shas = [], []
    for epoch in range(48):
        body = {"epoch": epoch, "optimizer_steps": 33925, "cumulative_optimizer_steps": (epoch + 1) * 33925,
                "adam_finite": True, "rng_violations": int(rng_drift and epoch == 0), "prefix_mutations": 0,
                "p0_prediction_mismatches": p1_mismatches, "p0_identity_mismatches": p1_mismatches,
                "parameter_finiteness": {"violations": 0}, "gradient_coverage": {"zero_decoder_steps": 0,
                "positive_encoder_steps": 1, "positive_decoder_steps": 1, "zero_encoder_steps": 0,
                "accepted_zero_encoder_steps": 0, "zero_encoder_reason": "all_units_dropped_valid_zero"},
                "sentinels": [{}, {}, {}, {}], "sampler_order": {"window_indices_sha256": "c" * 64,
                "batched_indices_sha256": "d" * 64}}
        sha, desc = _sealed(receipt, out / f"epoch{epoch:03d}.json", body)
        epoch_shas.append(sha); epoch_descs.append(desc)
    checkpoint_descs, checkpoint_shas = [], []
    for epoch in (44, 45, 46, 47):
        sha, desc = _sealed(receipt, out / f"epoch{epoch:03d}.pt", {"opaque": epoch})
        desc["epoch"] = checkpoint_epoch_override if epoch == 44 and checkpoint_epoch_override is not None else epoch
        checkpoint_shas.append(sha); checkpoint_descs.append(desc)
    swa_sha, swa_desc = _sealed(receipt, out / "swa_final4.pt", {"opaque": "swa"})
    strict_state = "e" * 64
    swa_loaded = {**swa_desc, "strict_loaded_state_sha256": strict_state}
    inherited = {"components": [{"path": f"/opaque/{item['name']}", "sha256": item["sha256"]} for item in checkpoint_descs],
                 "floating_tensor_count": 27, "buffer_tensor_count": 2, "uninitialized_lazy_tensor_count": 2,
                 "fp64_arithmetic": True, "optimizer_state_included": False,
                 "strict_reload_verified": True, "finite_forward_tensors": True}
    manifest_sha, manifest_desc = _sealed(receipt, out / "manifest.json", {"attempt": attempt_desc,
        "checkpoints": checkpoint_descs, "swa": swa_loaded, "inherited_final_four": inherited,
        "implementation_closure_sha256": closure["closure_sha256"]})
    swa_terminal = {**swa_loaded,
        "proof": {"strict_load": True, "eval": True, "no_grad": True,
                  "repeated_forward_bitwise_equal": True, "output_finite": True,
                  "dynamic_dropout_calls": 0, "state_before_sha256": strict_state, "state_after_sha256": strict_state}}
    terminal_device = {"profile": profile, "observer_kind": "extended-fixed-device-production-v1",
                       "issuance_idle": _v4_attestation(raw_arm), "pre_attempt_idle": _v4_attestation(raw_arm),
                       "post_attempt_idle": _v4_attestation(raw_arm), "context_bound": context,
                       "final": {"stage": "active_current_pid", "current_pid": 123, "selected_gpu_compute_pids": [123]},
                       "peak_allocated": 1, "peak_reserved": 1,
                       "tf32": {"matmul_allow_tf32": False, "cudnn_allow_tf32": True}}
    if terminal_device_mutation:
        terminal_device_mutation(terminal_device)
    terminal_body = {"schema": "pacd_p1_p2_admission_multigpu_v4_terminal", "status": "PACD_FULL_TRAINING_COMPLETE",
        "arm": arm_name.lower(), "target_access": False, "attempt": attempt_desc, "launch": launch_desc,
        "source_authority": authority_desc, "predecessor": predecessor, "epochs": epoch_descs,
        "checkpoints": checkpoint_descs, "swa": swa_terminal, "manifest": manifest_desc,
        "source_closure": {"launch": closure, "final": closure},
        "progress": {"epochs_published": 48, "checkpoints_published": 4, "swa_published": True},
        "device": terminal_device, "scheduler": scheduler["terminal"]}
    terminal_sha, _ = _sealed(receipt, out / "terminal.json", terminal_body)
    arm = binding.V4AdmissionProducerArm(arm_name, base, terminal_sha, swa_sha, manifest_sha, attempt_sha, launch_sha,
        authority_sha, tuple(checkpoint_shas), tuple(epoch_shas), closure["closure_sha256"], raw_arm.short_m, raw_arm.physical_index,
        raw_arm.expected_uuid, raw_arm.expected_pci_bus_id, raw_arm.cuda_visible_devices)
    return arm, p0, binding.MixedPACDProducerBinding.synthetic(), terminal_body


def _payload():
    runtime = _Runtime()
    authority, input_authority = score.materialize_authority(
        runtime, materialize_session=lambda _runtime, surface, session: _Inputs(surface, session),
        tensor_digest=repr,
    )

    def dig(value):
        return hashlib.sha256(value.encode()).hexdigest()

    def forward(system, _inputs, record):
        value = {"P0": .4, "P1": .44, "P2": .43, "T0": .39, "C1": .41, "SD": .38}[system]
        state = dig("state-" + system)
        return {"r2": value, "prediction_sha256": dig("p" + system + record.input_record_sha256),
                "identity_sha256": dig("i" + system), "repeated_identity_sha256": dig("i" + system),
                "repeated_prediction_sha256": dig("p" + system + record.input_record_sha256),
                "sentinel_coordinates": (0,), "sentinel_prediction_sha256": dig("s" + system),
                "repeated_sentinel_prediction_sha256": dig("s" + system), "n_windows": 2,
                "valid_last_bin_count": 1, "model_state_before_sha256": state,
                "model_state_after_sha256": state, "model_swa_sha256": dig("swa" + system),
                "strict_load": True, "eval_no_dropout_no_grad": True, "repeated_forward_equal": True,
                "identity_repeat_equal": True, "finite_prediction": True,
                "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0}

    rows = score.score_session_outer(authority, rosters={"within": runtime.within_roster,
                                                          "external": runtime.external_roster}, system_forward=forward)
    return {"rows": rows, "input_authority": input_authority}


def test_mixed_binding_is_typed_and_v4_arm_device_mapping_is_immutable():
    mixed = binding.MixedPACDProducerBinding.synthetic()
    assert [arm.arm for arm in mixed.arms] == ["P0", "P1", "P2"]
    assert mixed.p1.short_m == 4 and mixed.p1.physical_index == 0
    assert mixed.p2.short_m == 10 and mixed.p2.physical_index == 1
    with pytest.raises(binding.MixedBindingError, match="synthetic"):
        mixed.require_live()
    with pytest.raises(binding.MixedBindingError, match="V4 arm/device mapping"):
        binding.V4AdmissionProducerArm("P1", plan.P1_ROOT, "a" * 64, "a" * 64, "a" * 64, "a" * 64,
                                       "a" * 64, "a" * 64, ("a" * 64,) * 4, ("a" * 64,) * 48, "a" * 64, 4, 1,
                                       "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9", "00000000:01:00.0", "0")


def test_live_binding_requires_the_frozen_complete_literal_mapping(monkeypatch):
    synthetic = binding.MixedPACDProducerBinding.synthetic()
    candidate = replace(synthetic, accepted_v2_smoke_terminal_sha256=plan.V2_SMOKE_TERMINAL_SHA,
                        accepted_v2_smoke_sha256s=(plan.V2_SMOKE_ATTEMPT_SHA, plan.V2_SMOKE_TERMINAL_SHA))
    expected = candidate.payload()
    expected["mode"] = "live"
    monkeypatch.setattr(plan, "LIVE_MIXED_PRODUCER_LITERALS", expected)
    live = replace(candidate, mode="live")
    live.require_live()
    forged_p1 = binding.V4AdmissionProducerArm(
        live.p1.arm, live.p1.root_relative, "b" * 64, live.p1.swa_sha256, live.p1.manifest_sha256,
        live.p1.attempt_sha256, live.p1.launch_sha256, live.p1.source_authority_sha256,
        live.p1.checkpoint_sha256s, live.p1.epoch_sha256s, live.p1.closure_sha256,
        live.p1.short_m, live.p1.physical_index, live.p1.expected_uuid, live.p1.expected_pci_bus_id,
        live.p1.cuda_visible_devices)
    with pytest.raises(binding.MixedBindingError, match="mixed live binding literal drift"):
        replace(live, p1=forged_p1)


def test_v2_profile_executes_the_one_shared_atomic_lifecycle(tmp_path):
    mixed = binding.MixedPACDProducerBinding.synthetic()
    cap = lifecycle._mint_synthetic_capability(
        lifecycle._SECRET, mixed, repository_root=tmp_path, root_relative="mixed",
        execution_profile=smoke.V2_MIXED_SCORE_PROFILE,
    )
    result = lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "mixed", binding=mixed,
                                      capability=cap, receipt=_Receipt(), materialize_then_score=_payload,
                                      allow_synthetic=True, execution_profile=smoke.V2_MIXED_SCORE_PROFILE)
    assert len(result["rows"]) == 756
    terminal = tmp_path / "mixed" / "complete" / "terminal.json"
    assert terminal.is_file() and not (tmp_path / "mixed" / "failure.json").exists()
    assert plan.SCHEMA in terminal.read_text()
    with pytest.raises(lifecycle.LifecycleError, match="consumed"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "mixed", binding=mixed,
                                 capability=cap, receipt=_Receipt(), materialize_then_score=_payload,
                                 allow_synthetic=True, execution_profile=smoke.V2_MIXED_SCORE_PROFILE)


def test_profile_substitution_and_live_mint_fail_closed(tmp_path):
    mixed = binding.MixedPACDProducerBinding.synthetic()
    cap = lifecycle._mint_synthetic_capability(lifecycle._SECRET, mixed, repository_root=tmp_path,
                                                root_relative="mixed", execution_profile=smoke.V2_MIXED_SCORE_PROFILE)
    with pytest.raises(lifecycle.LifecycleError, match="binding-type/profile"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "mixed", binding=mixed,
                                 capability=cap, receipt=_Receipt(), materialize_then_score=_payload,
                                 allow_synthetic=True)
    with pytest.raises(binding.MixedBindingError, match="synthetic"):
        lifecycle.prepare_live_authority(mixed, repository_root=tmp_path, receipt=_Receipt(),
                                         execution_profile=smoke.V2_MIXED_SCORE_PROFILE)
    with pytest.raises(lifecycle.LifecycleError, match="binding-type"):
        lifecycle._mint_synthetic_capability(lifecycle._SECRET, v1_binding.PACDProducerBinding.synthetic(),
                                             repository_root=tmp_path, root_relative="wrong-profile",
                                             execution_profile=smoke.V2_MIXED_SCORE_PROFILE)


def test_v2_dry_cli_is_inert_and_torch_free():
    command = [sys.executable, "-S", str(ROOT / "scripts/run_pacd_matched_score_v2_mixed_lineage.py"), "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True,
                            env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0 and "producer_literals_deferred" in result.stdout
    probe = subprocess.run([sys.executable, "-S", "-c", "import runpy,sys; runpy.run_path(sys.argv[1],run_name='x'); assert 'torch' not in sys.modules", command[2]],
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    assert probe.returncode == 0, probe.stderr


def test_v2_closure_is_explicit_and_excludes_tests_and_results():
    first = lifecycle.source_closure(ROOT.parent, smoke.V2_MIXED_SCORE_PROFILE)
    assert first == lifecycle.source_closure(ROOT.parent, smoke.V2_MIXED_SCORE_PROFILE)
    assert plan.WORK_ORDER_RELATIVE in first["files"]
    assert first["files"][plan.V4_PARALLEL_ROUTE_ISOLATION_RELATIVE]["sha256"] == plan.V4_PARALLEL_ROUTE_ISOLATION_SHA256
    assert first["files"][plan.V4_SHARED_LIFECYCLE_SEAM_RELATIVE]["sha256"] == plan.V4_SHARED_LIFECYCLE_SEAM_SHA256
    assert "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/predecessor.py" in first["files"]
    assert "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/plan.py" in first["files"]
    assert all("/tests/" not in key and "/results/" not in key for key in first["files"])


def test_v2_clean_import_trace_is_torch_free_and_covered_by_closure():
    modules = [
        "src.paired_anchored_calibration_dropout_full_v3.predecessor",
        "src.paired_anchored_calibration_dropout_full_v3.plan",
        "src.paired_anchored_calibration_dropout_full_v2.plan",
        "src.paired_anchored_calibration_dropout_full_v1.plan",
        "src.paired_anchored_calibration_dropout_v1.plan",
    ]
    code = (
        "import sys; from src.paired_anchored_calibration_dropout_score_v2_mixed_lineage import binding; "
        f"names={modules!r}; "
        "assert 'torch' not in sys.modules; "
        "[print(sys.modules[name].__file__) for name in names]"
    )
    result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "",
                                 "PYTHONPATH": str(ROOT)})
    assert result.returncode == 0, result.stderr
    closure = lifecycle.source_closure(ROOT.parent, smoke.V2_MIXED_SCORE_PROFILE)["files"]
    for path in result.stdout.splitlines():
        relative = str(Path(path).resolve().relative_to(ROOT.parent))
        assert relative in closure


def test_v4_held_graph_validates_every_leaf_and_semantic_contract(tmp_path):
    arm, p0, mixed, _terminal = _v4_fixture(tmp_path)
    witness = binding._verify_v4_arm(tmp_path, arm, p0, mixed)
    assert witness["terminal_sha256"] == arm.terminal_sha256
    assert witness["checkpoint_sha256s"] == list(arm.checkpoint_sha256s)


def test_accepted_v2_smoke_descriptor_validates_the_actual_immutable_four_leaf_graph():
    """Integration is receipt-only: no tensor/data/CUDA path is reachable."""
    mixed = replace(binding.MixedPACDProducerBinding.synthetic(),
                    accepted_v2_smoke_terminal_sha256=plan.V2_SMOKE_TERMINAL_SHA,
                    accepted_v2_smoke_sha256s=(plan.V2_SMOKE_ATTEMPT_SHA, plan.V2_SMOKE_TERMINAL_SHA))
    witness = binding._verify_v2_smoke(ROOT.parent, mixed)
    assert witness["topology"] == ["attempt.json", "attempt.json.sha256", "terminal.json", "terminal.json.sha256"]
    assert witness["attempt_sha256"] == plan.V2_SMOKE_ATTEMPT_SHA


@pytest.mark.parametrize("mutation, expected", [
    ("epoch-mode", "artifact mode/type"),
    ("epoch-sidecar", "artifact sidecar drift"),
    ("epoch-evidence", "V4 paired finite law"),
    ("checkpoint-link", "V4 checkpoint descriptor literal"),
])
def test_v4_held_graph_rejects_leaf_and_evidence_drift(tmp_path, mutation, expected):
    arm, p0, mixed, terminal = _v4_fixture(tmp_path, rng_drift=(mutation == "epoch-evidence"),
                                            checkpoint_epoch_override=43 if mutation == "checkpoint-link" else None)
    out = tmp_path / arm.root_relative
    if mutation == "epoch-mode":
        os.chmod(out / "epoch000.json", 0o644)
    elif mutation == "epoch-sidecar":
        side = out / "epoch000.json.sha256"
        os.chmod(side, 0o644)
        side.write_text("x" * 64 + "  epoch000.json\n")
        os.chmod(side, 0o444)
    elif mutation in {"epoch-evidence", "checkpoint-link"}:
        pass
    with pytest.raises((binding.MixedBindingError, v1_binding.BindingError), match=expected):
        binding._verify_v4_arm(tmp_path, arm, p0, mixed)


def test_v4_short_arm_accepts_nonzero_descriptive_p0_mismatch_accounting(tmp_path):
    arm, p0, mixed, _ = _v4_fixture(tmp_path, p1_mismatches=33925)
    assert binding._verify_v4_arm(tmp_path, arm, p0, mixed)["terminal_sha256"] == arm.terminal_sha256


def test_v4_scheduler_exact_p1_p2_profiles_and_historical_live_cdm_parser_pass(tmp_path):
    """The old live-CDM shape remains parser coverage for synthetic receipts only."""
    arm, p0, mixed, _ = _v4_fixture(tmp_path / "p1")
    assert binding._verify_v4_arm(tmp_path / "p1", arm, p0, mixed)["terminal_sha256"] == arm.terminal_sha256
    cdm = {"identity": "cdm_p1_cross_v1", "pid": 804509, "live": True,
           "observed_affinity": list(range(4, 16)) + list(range(20, 32))}
    arm, p0, mixed, _ = _v4_fixture(tmp_path / "p1-live-cdm", tracked=cdm)
    assert binding._verify_v4_arm(tmp_path / "p1-live-cdm", arm, p0, mixed)["terminal_sha256"] == arm.terminal_sha256
    arm, p0, mixed, _ = _v4_fixture(tmp_path / "p2", arm_name="P2")
    assert binding._verify_v4_arm(tmp_path / "p2", arm, p0, mixed)["terminal_sha256"] == arm.terminal_sha256


def test_v4_live_scheduler_observation_requires_inactive_route_for_future_producer():
    """A non-overlapping historical CDM route is never admissible in live mode."""
    arm = binding.MixedPACDProducerBinding.synthetic().p1
    profile = binding._scheduler_profile(arm)
    historical = _v4_scheduler(
        arm,
        tracked={"identity": "cdm_p1_cross_v1", "pid": 804509, "live": True,
                 "observed_affinity": list(range(4, 16)) + list(range(20, 32))},
    )["attempt"]["before_attempt"]
    with pytest.raises(binding.MixedBindingError, match="live tracked route inadmissible"):
        binding._scheduler_observation(historical, profile=profile, label="live", allow_historical_live=False)


def test_v4_host_pressure_nearly_full_swap_is_descriptive_when_psi_passes(tmp_path):
    """Swap occupancy alone must not reject a low-pressure P1 admission."""
    def mutate(scheduler):
        for stage, observation_key in (("attempt", "before_attempt"), ("launch", "after_attempt")):
            pressure = scheduler[stage][observation_key]["host_pressure"]
            pressure["swap_total_bytes"] = 1_000_000
            pressure["swap_free_bytes"] = 1
        for observation_key in ("before_attempt", "after_attempt", "final"):
            pressure = scheduler["terminal"][observation_key]["host_pressure"]
            pressure["swap_total_bytes"] = 1_000_000
            pressure["swap_free_bytes"] = 1
    arm, p0, mixed, _ = _v4_fixture(tmp_path, scheduler_mutation=mutate)
    assert binding._verify_v4_arm(tmp_path, arm, p0, mixed)["terminal_sha256"] == arm.terminal_sha256


def test_v4_host_pressure_swap_disabled_is_valid_when_memory_and_psi_pass(tmp_path):
    def mutate(scheduler):
        observations = (
            scheduler["attempt"]["before_attempt"], scheduler["launch"]["after_attempt"],
            scheduler["terminal"]["before_attempt"], scheduler["terminal"]["after_attempt"],
            scheduler["terminal"]["final"],
        )
        for observation in observations:
            observation["host_pressure"]["swap_total_bytes"] = 0
            observation["host_pressure"]["swap_free_bytes"] = 0
    arm, p0, mixed, _ = _v4_fixture(tmp_path, scheduler_mutation=mutate)
    assert binding._verify_v4_arm(tmp_path, arm, p0, mixed)["terminal_sha256"] == arm.terminal_sha256


@pytest.mark.parametrize("kind, expected", [
    ("missing", "V4 attempt scheduler profile"),
    ("extra", "V4 attempt scheduler profile"),
    ("swapped", "V4 attempt scheduler profile"),
    ("expanded", "V4 attempt scheduler profile"),
    ("terminal-drift", "V4 terminal scheduler historical evidence"),
])
def test_v4_scheduler_rejects_profile_and_terminal_drift(tmp_path, kind, expected):
    def mutate(scheduler):
        if kind == "missing":
            del scheduler["attempt"]["profile"]["logical_cpu_affinity"]
        elif kind == "extra":
            scheduler["attempt"]["profile"]["extra"] = "forbidden"
        elif kind == "swapped":
            scheduler["attempt"]["profile"]["identity"] = "v4-p2-gpu1-cpu4-15-20-31"
        elif kind == "expanded":
            scheduler["attempt"]["profile"]["logical_cpu_affinity"].append(4)
            scheduler["attempt"]["before_attempt"]["observed_affinity"].append(4)
        else:
            scheduler["terminal"]["before_attempt"] = {"observed_affinity": [4],
                "tracked_other_route": {"identity": "none", "pid": None, "live": False, "observed_affinity": []},
                "overlap": []}
    arm, p0, mixed, _ = _v4_fixture(tmp_path, scheduler_mutation=mutate)
    with pytest.raises(binding.MixedBindingError, match=expected):
        binding._verify_v4_arm(tmp_path, arm, p0, mixed)


def test_v4_scheduler_rejects_overlap_and_live_identity_pid_inconsistency(tmp_path):
    cdm = {"identity": "cdm_p1_cross_v1", "pid": 804509, "live": True,
           "observed_affinity": list(range(4, 16)) + list(range(20, 32))}
    arm, p0, mixed, _ = _v4_fixture(tmp_path / "overlap", arm_name="P2", tracked=cdm)
    with pytest.raises(binding.MixedBindingError, match="scheduler overlap"):
        binding._verify_v4_arm(tmp_path / "overlap", arm, p0, mixed)
    inconsistent = {"identity": "cdm_p1_cross_v1", "pid": None, "live": True, "observed_affinity": [4]}
    arm, p0, mixed, _ = _v4_fixture(tmp_path / "inconsistent", tracked=inconsistent)
    with pytest.raises(binding.MixedBindingError, match="live tracked route"):
        binding._verify_v4_arm(tmp_path / "inconsistent", arm, p0, mixed)


@pytest.mark.parametrize("kind, expected", [
    ("missing", "host-pressure schema"),
    ("extra", "host-pressure schema"),
    ("nonfinite", "host-pressure PSI"),
    ("low-memory", "host-pressure gate"),
    ("some-psi", "host-pressure gate"),
    ("full-psi", "host-pressure gate"),
    ("bad-pass", "host-pressure pass"),
    ("nonmonotonic", "monotonic host-pressure time"),
])
def test_v4_scheduler_rejects_host_pressure_and_time_adversaries(tmp_path, kind, expected):
    def mutate(scheduler):
        if kind == "nonmonotonic":
            scheduler["terminal"]["final"]["host_pressure"]["observed_monotonic_ns"] = 15
            return
        pressure = scheduler["attempt"]["before_attempt"]["host_pressure"]
        if kind == "missing":
            del pressure["swap_free_bytes"]
        elif kind == "extra":
            pressure["extra"] = 1
        elif kind == "nonfinite":
            pressure["memory_psi_some_avg10"] = float("nan")
        elif kind == "low-memory":
            pressure["mem_available_bytes"] = 17_179_869_183
            pressure["pass"] = False
        elif kind == "some-psi":
            pressure["memory_psi_some_avg10"] = 0.100000001
            pressure["pass"] = False
        elif kind == "full-psi":
            pressure["memory_psi_full_avg10"] = 0.001
            pressure["pass"] = False
        else:
            pressure["pass"] = False
    arm, p0, mixed, _ = _v4_fixture(tmp_path, scheduler_mutation=mutate)
    with pytest.raises(binding.MixedBindingError, match=expected):
        binding._verify_v4_arm(tmp_path, arm, p0, mixed)


def test_v4_held_graph_rejects_p0_root_swap_closure_and_device_stage_drift(tmp_path):
    arm, p0, mixed, _ = _v4_fixture(tmp_path)
    old = tmp_path / p0["root_relative"]
    moved = old.with_name("p0-replaced")
    os.rename(old, moved)
    old.mkdir()
    with pytest.raises(binding.MixedBindingError, match="V4 P0 held root identity"):
        binding._verify_v4_arm(tmp_path, arm, p0, mixed)

    arm, p0, mixed, _ = _v4_fixture(tmp_path / "closure")
    wrong_closure_arm = binding.V4AdmissionProducerArm(
        arm.arm, arm.root_relative, arm.terminal_sha256, arm.swa_sha256, arm.manifest_sha256,
        arm.attempt_sha256, arm.launch_sha256, arm.source_authority_sha256, arm.checkpoint_sha256s,
        arm.epoch_sha256s, "f" * 64, arm.short_m, arm.physical_index, arm.expected_uuid,
        arm.expected_pci_bus_id, arm.cuda_visible_devices)
    with pytest.raises(binding.MixedBindingError, match="V4 closure equality"):
        binding._verify_v4_arm(tmp_path / "closure", wrong_closure_arm, p0, mixed)

    arm, p0, mixed, _ = _v4_fixture(tmp_path / "device", bad_context=True)
    with pytest.raises(binding.MixedBindingError, match="V4 launch context evidence"):
        binding._verify_v4_arm(tmp_path / "device", arm, p0, mixed)


@pytest.mark.parametrize("kind, expected", [
    ("issuance", "V4 terminal issuance-attestation drift"),
    ("pre-attempt", "V4 terminal pre-attempt-attestation drift"),
    ("post-attempt", "V4 terminal post-attempt-attestation drift"),
    ("final-pid", "V4 final/context PID drift"),
])
def test_v4_terminal_device_receipt_cross_links_reject_actual_shaped_drift(tmp_path, kind, expected):
    """Each mutation preserves the terminal device key topology and reseals it."""
    def mutate(device):
        if kind == "issuance":
            device["issuance_idle"]["stage"] = "drifted_idle_stage"
        elif kind == "pre-attempt":
            device["pre_attempt_idle"]["preflight_uses_torch"] = True
        elif kind == "post-attempt":
            device["post_attempt_idle"]["name"] = "NVIDIA GeForce RTX 3090 drift"
        else:
            # Self-consistent final evidence must still match the context PID.
            device["final"]["current_pid"] = 456
            device["final"]["selected_gpu_compute_pids"] = [456]
    arm, p0, mixed, _ = _v4_fixture(tmp_path, terminal_device_mutation=mutate)
    with pytest.raises(binding.MixedBindingError, match=expected):
        binding._verify_v4_arm(tmp_path, arm, p0, mixed)
