"""Typed deferred V3-P0 / V4-P1-P2 producer binding.

Live literals intentionally remain unavailable.  The immutable shape below is
still useful to prove that no all-V3 binding can accidentally enter V2.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import stat
from pathlib import Path
from typing import Any

from . import plan
from src.paired_anchored_calibration_dropout_score_v1 import binding as v1
from src.paired_anchored_calibration_dropout_full_v3 import predecessor as v3_predecessor


class MixedBindingError(v1.BindingError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise MixedBindingError(message)


def _hex(value: str, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), label)
    return value


def _relative(value: str, label: str) -> str:
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and str(path) == value, label)
    return value


@dataclass(frozen=True)
class _Arm:
    arm: str
    root_relative: str
    terminal_sha256: str
    swa_sha256: str
    manifest_sha256: str
    attempt_sha256: str
    launch_sha256: str
    source_authority_sha256: str
    checkpoint_sha256s: tuple[str, str, str, str]
    epoch_sha256s: tuple[str, ...]
    closure_sha256: str

    def _check_common(self) -> None:
        _relative(self.root_relative, "producer root")
        for item in (self.terminal_sha256, self.swa_sha256, self.manifest_sha256,
                     self.attempt_sha256, self.launch_sha256, self.source_authority_sha256):
            _hex(item, "producer body")
        _require(len(self.checkpoint_sha256s) == 4, "final-four checkpoints")
        for item in self.checkpoint_sha256s:
            _hex(item, "checkpoint body")
        _require(len(self.epoch_sha256s) == 48, "epoch receipt topology")
        for item in self.epoch_sha256s:
            _hex(item, "epoch receipt")
        _hex(self.closure_sha256, "producer closure")


@dataclass(frozen=True)
class V3P0ProducerArm(_Arm):
    def __post_init__(self) -> None:
        _require(self.arm == "P0" and self.root_relative == plan.P0_ROOT, "V3 P0 identity")
        self._check_common()
        _require(self.closure_sha256 == plan.V3_HISTORICAL_CLOSURE, "V3 historical closure literal")


@dataclass(frozen=True)
class V4AdmissionProducerArm(_Arm):
    short_m: int
    physical_index: int
    expected_uuid: str
    expected_pci_bus_id: str
    cuda_visible_devices: str
    logical_device: str = "cuda:0"

    def __post_init__(self) -> None:
        expected = {
            "P1": (plan.P1_ROOT, 4, 0, "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9", "00000000:01:00.0", "0"),
            "P2": (plan.P2_ROOT, 10, 1, "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86", "00000000:03:00.0", "1"),
        }
        _require(self.arm in expected, "V4 arm")
        _require((self.root_relative, self.short_m, self.physical_index, self.expected_uuid,
                  self.expected_pci_bus_id, self.cuda_visible_devices) == expected[self.arm], "V4 arm/device mapping")
        _require(self.logical_device == "cuda:0", "V4 logical device")
        self._check_common()


@dataclass(frozen=True)
class MixedPACDProducerBinding:
    TYPE_IDENTITY = "MixedPACDProducerBinding:v2"
    p0: V3P0ProducerArm
    p1: V4AdmissionProducerArm
    p2: V4AdmissionProducerArm
    mode: str
    accepted_v2_smoke_terminal_sha256: str
    accepted_v2_smoke_root_relative: str
    # The accepted V2 smoke has exactly two immutable receipt bodies
    # (attempt/terminal) and their two sidecars.  The terminal literal is
    # retained separately because the score receipt exposes it publicly.
    accepted_v2_smoke_sha256s: tuple[str, str]
    v2_full_failure_sha256: str
    v2_full_failure_root_relative: str

    def __post_init__(self) -> None:
        _require(self.mode in {"synthetic", "live"}, "binding mode")
        _require((self.p0.arm, self.p1.arm, self.p2.arm) == ("P0", "P1", "P2"), "mixed arm order")
        _hex(self.accepted_v2_smoke_terminal_sha256, "V2 smoke")
        _hex(self.v2_full_failure_sha256, "V2 failure")
        _relative(self.accepted_v2_smoke_root_relative, "V2 smoke root")
        _relative(self.v2_full_failure_root_relative, "V2 failure root")
        _require(len(self.accepted_v2_smoke_sha256s) == 2, "V2 smoke 4-leaf graph")
        for item in self.accepted_v2_smoke_sha256s:
            _hex(item, "V2 smoke body")
        _require(self.accepted_v2_smoke_sha256s[1] == self.accepted_v2_smoke_terminal_sha256,
                 "V2 smoke terminal literal")
        _require(self.v2_full_failure_sha256 == plan.V2_FAILURE_SHA, "V2 failure literal")
        if self.mode == "live":
            _require(self.accepted_v2_smoke_root_relative
                     == "tfpd_exploration/results/paired_anchored_calibration_dropout_v2/smoke_seed42"
                     and self.accepted_v2_smoke_sha256s == (plan.V2_SMOKE_ATTEMPT_SHA, plan.V2_SMOKE_TERMINAL_SHA),
                     "accepted V2 smoke literal")
            _require(plan.LIVE_MIXED_PRODUCER_LITERALS is not None, "mixed live literals are deferred")
            _require(plan.LIVE_MIXED_PRODUCER_LITERALS == self._live_literal_payload(),
                     "mixed live binding literal drift")

    @property
    def arms(self) -> tuple[_Arm, _Arm, _Arm]:
        return self.p0, self.p1, self.p2

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "mixed_pacd_producer_binding_v2", "mode": self.mode,
            "accepted_v2_smoke_terminal_sha256": self.accepted_v2_smoke_terminal_sha256,
            "accepted_v2_smoke_root_relative": self.accepted_v2_smoke_root_relative,
            "accepted_v2_smoke_sha256s": list(self.accepted_v2_smoke_sha256s),
            "v2_full_failure_sha256": self.v2_full_failure_sha256,
            "v2_full_failure_root_relative": self.v2_full_failure_root_relative,
            "arms": [arm.__dict__ for arm in self.arms],
        }

    def require_live(self) -> None:
        _require(self.mode == "live", "synthetic mixed binding cannot score")
        _require(plan.LIVE_MIXED_PRODUCER_LITERALS is not None, "mixed live literals are deferred")
        _require(plan.LIVE_MIXED_PRODUCER_LITERALS == self._live_literal_payload(),
                 "mixed live binding literal drift")

    def _live_literal_payload(self) -> dict[str, Any]:
        """Canonical mapping compared before any live descriptor is opened."""
        payload = self.payload()
        payload["mode"] = "live"
        return payload

    @classmethod
    def synthetic(cls, *, digest: str = "a" * 64) -> "MixedPACDProducerBinding":
        return cls(
            V3P0ProducerArm("P0", plan.P0_ROOT, digest, digest, digest, digest, digest, digest, (digest,) * 4, (digest,) * 48, plan.V3_HISTORICAL_CLOSURE),
            V4AdmissionProducerArm("P1", plan.P1_ROOT, digest, digest, digest, digest, digest, digest, (digest,) * 4, (digest,) * 48, digest,
                                   4, 0, "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9", "00000000:01:00.0", "0"),
            V4AdmissionProducerArm("P2", plan.P2_ROOT, digest, digest, digest, digest, digest, digest, (digest,) * 4, (digest,) * 48, digest,
                                   10, 1, "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86", "00000000:03:00.0", "1"),
            "synthetic", digest,
            "tfpd_exploration/results/paired_anchored_calibration_dropout_v2/smoke_seed42",
            (digest, digest),
            plan.V2_FAILURE_SHA,
            "tfpd_exploration/results/paired_anchored_calibration_dropout_full_v2/p0_fullfull_seed42",
        )


def verify_live_mixed_producers(root: Path, binding: MixedPACDProducerBinding) -> dict[str, Any]:
    """Fail closed until exact immutable mixed producer literals are reviewed.

    This intentionally performs no descriptor/data/tensor access while the
    workorder's live literal block remains absent.
    """
    binding.require_live()
    v2_smoke = _verify_v2_smoke(root, binding)
    lower_v2_failure = _verify_v2_full_failure(root, binding)
    p0 = _verify_v3_p0(root, binding)
    p1 = _verify_v4_arm(root, binding.p1, p0, binding)
    p2 = _verify_v4_arm(root, binding.p2, p0, binding)
    return {"schema": "mixed_pacd_producer_witness_v2", "accepted_v2_smoke": v2_smoke,
            "v2_full_failure": lower_v2_failure, "p0": p0, "p1": p1, "p2": p2}


def _verify_v2_smoke(root: Path, binding: MixedPACDProducerBinding) -> dict[str, Any]:
    """Validate the real immutable V2 attempt/terminal four-leaf graph.

    This is deliberately route-local: Score V1 retained an older synthetic
    smoke codec, while the accepted V2 receipt has a top-level
    ``terminal.attempt_sha256`` link and no launch/authority leaves.
    """
    base = binding.accepted_v2_smoke_root_relative
    names = ("attempt.json", "terminal.json")
    expected = set(names) | {name + ".sha256" for name in names}
    _require(v1._held_leaf_set(root, base) == expected, "V2 smoke exact root topology")
    bodies = [v1.descriptor_json(root, base + "/" + name, digest)
              for name, digest in zip(names, binding.accepted_v2_smoke_sha256s, strict=True)]
    attempt, terminal = bodies
    predecessor = {
        "attempt_sha256": "e1d6cd813b2bc49d89121b3341271a0bf94176abe3811d2f3b6f6ffa4a28b986",
        "failure_sha256": "82ca6850cdd50f0b5ea5073eb89809a0c6ea426ac3f8f7d9e03fc44d1bd7e273",
        "relative": "tfpd_exploration/results/paired_anchored_calibration_dropout_v1/smoke_seed42",
        "topology": ["attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"],
    }
    arms = (("p0", 30), ("p1", 4), ("p2", 10))
    _require(attempt.get("schema") == "paired_anchored_calibration_dropout_v2_attempt"
             and attempt.get("status") == "ATTEMPT_PUBLISHED"
             and attempt.get("cell") == "PAIRED_ANCHORED_CALIBRATION_DROPOUT_V2"
             and attempt.get("source_only") is True and attempt.get("target_access") is False
             and attempt.get("full_training_authorized") is False, "V2 smoke attempt semantics")
    _require(attempt.get("budget") == {"cpu_threads": 1, "num_workers": 0, "seed": 42,
                                        "steps_per_arm": 8, "train_batch_size": 32}
             and attempt.get("predecessor") == predecessor, "V2 smoke attempt budget/predecessor")
    attempt_arms = attempt.get("arms")
    _require(isinstance(attempt_arms, list) and [(item.get("name"), item.get("anchor_m"), item.get("short_m"),
                                                   item.get("loss_weights")) for item in attempt_arms]
             == [(name, 30, short_m, {"anchor": 0.5, "short": 0.5}) for name, short_m in arms],
             "V2 smoke attempt arms")
    _require(terminal.get("schema") == "paired_anchored_calibration_dropout_v2_terminal"
             and terminal.get("status") == "PACD_SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE"
             and terminal.get("cell") == "PAIRED_ANCHORED_CALIBRATION_DROPOUT_V2"
             and terminal.get("source_only") is True and terminal.get("target_access") is False
             and terminal.get("full_training_authorized") is False, "V2 smoke terminal semantics")
    _require(terminal.get("attempt_sha256") == binding.accepted_v2_smoke_sha256s[0]
             and terminal.get("predecessor") == predecessor, "V2 smoke attempt link")
    _require(terminal.get("no_target_facts") == {
        "external_sub_m_opened": False, "formal_or_organizer_held_data_opened": False,
        "scorer_called": False, "single_source_datamodule_materialization": True,
        "source_roster_n": 27, "test_paths_resolved": [], "val_paths_resolved": [],
        "within_dev_sessions_opened": False}, "V2 smoke no-target facts")
    _require(attempt.get("source_closure", {}).get("closure_sha256") == plan.V2_SMOKE_CLOSURE_SHA
             and terminal.get("source_closure", {}).get("launch", {}).get("closure_sha256") == plan.V2_SMOKE_CLOSURE_SHA
             and terminal.get("source_closure", {}).get("final", {}).get("closure_sha256") == plan.V2_SMOKE_CLOSURE_SHA,
             "V2 smoke closure")
    _require(terminal.get("source_closure", {}).get("launch") == terminal.get("source_closure", {}).get("final"),
             "V2 smoke closure equality")
    observed = terminal.get("arms")
    _require(isinstance(observed, list) and len(observed) == 3, "V2 smoke terminal arms")
    for item, (name, short_m) in zip(observed, arms, strict=True):
        record = item.get("arm", {}) if isinstance(item, dict) else {}
        steps = item.get("steps", []) if isinstance(item, dict) else []
        _require(record.get("name") == name and record.get("anchor_m") == 30 and record.get("short_m") == short_m
                 and isinstance(steps, list) and [step.get("step") for step in steps] == list(range(8)),
                 "V2 smoke terminal arm topology")
        for step in steps:
            _require(step.get("optimizer_steps") == 1 and step.get("valid_bins", 0) > 0
                     and step.get("dropout_pair_equal") is True and step.get("rng_pair_transition_equal") is True
                     and step.get("rng_short_transition_equal") is True
                     and step.get("parameter_finiteness") == {"materialized": 29, "skipped_uninitialized_lazy": 2},
                     "V2 smoke paired-step evidence")
            if name == "p0":
                _require(step.get("prediction_pair_equal") is True and step.get("identity_pair_equal") is True,
                         "V2 smoke P0 equality")
    return {"root_relative": base, "attempt_sha256": binding.accepted_v2_smoke_sha256s[0],
            "topology": sorted(expected),
            "terminal_sha256": binding.accepted_v2_smoke_terminal_sha256}


def _verify_v2_full_failure(root: Path, binding: MixedPACDProducerBinding) -> dict[str, Any]:
    """Reuse the reviewed V3 held-FD validator; do not reproduce its codec."""
    _require(binding.v2_full_failure_root_relative == v3_predecessor.plan.V2_FAILURE_RELATIVE,
             "V2 failure root literal")
    _require(binding.v2_full_failure_sha256 == v3_predecessor.plan.V2_FAILURE_SHAS["failure.json"],
             "V2 failure body literal")
    try:
        witness = v3_predecessor.validate_v2_failure(root, binding.v2_full_failure_root_relative)
    except v3_predecessor.PredecessorError as exc:
        raise MixedBindingError("V2 full failure graph") from exc
    _require(witness.get("failure_sha256") == binding.v2_full_failure_sha256
             and witness.get("relative") == binding.v2_full_failure_root_relative, "V2 failure witness")
    return witness


def _descriptor(root: Path, base: str, name: str, digest: str) -> dict:
    return v1.descriptor_sha256(root, base + "/" + name, digest)


def _exact_descriptor(item: Any, *, name: str, digest: str, epoch: int | None = None,
                      extra: frozenset[str] = frozenset(), label: str) -> None:
    """Validate receipt descriptors before trusting their fixed pathname.

    The terminal must not be allowed to redirect a literal-bound body through a
    similarly hashed neighbouring receipt.  Epoch/checkpoint ``epoch`` is a
    receipt property, rather than a terminal alias, in the V4 codec.
    """
    _require(isinstance(item, dict), label + " descriptor object")
    expected = {"name": name, "sha256": digest, "sidecar": name + ".sha256"}
    if epoch is not None:
        expected["epoch"] = epoch
    _require(set(item) == set(expected) | set(extra), label + " descriptor schema")
    _require(all(item.get(key) == value for key, value in expected.items()), label + " descriptor literal")


def _attestation(attestation: Any, *, arm: V4AdmissionProducerArm, active: bool,
                 label: str) -> None:
    """Exact V4 selected-device attestation codec (never the scorer's PID)."""
    _require(isinstance(attestation, dict), label + " attestation object")
    required = {"stage", "physical_index", "uuid", "name", "driver_version", "pci_bus_id",
                "cuda_visible_devices", "compute_processes", "idle", "preflight_uses_torch"}
    _require(set(attestation) == required, label + " attestation schema")
    _require(attestation.get("physical_index") == arm.physical_index
             and attestation.get("uuid") == arm.expected_uuid
             and attestation.get("name") == "NVIDIA GeForce RTX 3090"
             and attestation.get("driver_version") == "535.309.01"
             and attestation.get("pci_bus_id") == arm.expected_pci_bus_id
             and attestation.get("cuda_visible_devices") == arm.cuda_visible_devices
             and attestation.get("preflight_uses_torch") is False, label + " static device profile")
    processes = attestation.get("compute_processes")
    _require(isinstance(processes, list), label + " process list")
    for process in processes:
        _require(isinstance(process, dict) and set(process) == {"pid", "process_name", "used_memory"}
                 and isinstance(process.get("pid"), int), label + " process descriptor")
    _require(attestation.get("stage") in {"idle_pre_cuda", "active_current_pid"}, label + " stage")
    if active:
        _require(attestation.get("stage") == "active_current_pid" and attestation.get("idle") is False
                 and len(processes) == 1, label + " active process")
    else:
        _require(attestation.get("stage") == "idle_pre_cuda" and attestation.get("idle") is True
                 and processes == [], label + " idle process")


def _context(context: Any, *, label: str) -> None:
    _require(isinstance(context, dict), label + " context object")
    _require(set(context) == {"stage", "visible_device_count", "current_logical_device", "logical_device",
                              "cuda_initialized", "current_pid", "selected_gpu_compute_pids"},
             label + " context schema")
    _require(context.get("stage") == "active_current_pid" and context.get("visible_device_count") == 1
             and context.get("current_logical_device") == 0 and context.get("logical_device") == "cuda:0"
             and context.get("cuda_initialized") is True and isinstance(context.get("current_pid"), int)
             and context.get("selected_gpu_compute_pids") == [context.get("current_pid")], label + " context evidence")


def _final_active(final: Any, *, label: str) -> None:
    _require(isinstance(final, dict) and set(final) == {"stage", "current_pid", "selected_gpu_compute_pids"},
             label + " final schema")
    _require(final.get("stage") == "active_current_pid" and isinstance(final.get("current_pid"), int)
             and final.get("selected_gpu_compute_pids") == [final.get("current_pid")], label + " final evidence")


def _scheduler_profile(arm: V4AdmissionProducerArm) -> dict[str, Any]:
    affinities = {
        "P1": ("v4-p1-gpu0-cpu0-3-16-19", [0, 1, 2, 3, 16, 17, 18, 19]),
        "P2": ("v4-p2-gpu1-cpu4-15-20-31", list(range(4, 16)) + list(range(20, 32))),
    }
    identity, affinity = affinities[arm.arm]
    return {"identity": identity, "logical_cpu_affinity": affinity, "num_workers": 0,
            "thread_limits": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}}


def _host_pressure(value: Any, *, label: str) -> int:
    """Exact Torch-free `/proc`-derived receipt codec from the addendum."""
    required = {"schema", "observed_monotonic_ns", "mem_available_bytes", "mem_available_floor_bytes",
                "memory_psi_some_avg10", "memory_psi_some_avg10_ceiling", "memory_psi_full_avg10",
                "memory_psi_full_avg10_ceiling", "swap_total_bytes", "swap_free_bytes", "pass"}
    _require(isinstance(value, dict) and set(value) == required, label + " host-pressure schema")
    _require(value.get("schema") == "pacd_v4_host_pressure_v1"
             and value.get("mem_available_floor_bytes") == 17_179_869_184
             and value.get("memory_psi_some_avg10_ceiling") == 0.1
             and value.get("memory_psi_full_avg10_ceiling") == 0.0, label + " host-pressure constants")
    for key in ("observed_monotonic_ns", "mem_available_bytes"):
        _require(isinstance(value.get(key), int) and not isinstance(value.get(key), bool) and value[key] > 0,
                 label + " host-pressure positive integer")
    _require(isinstance(value.get("swap_total_bytes"), int) and not isinstance(value.get("swap_total_bytes"), bool)
             and value["swap_total_bytes"] >= 0, label + " host-pressure swap-total")
    _require(isinstance(value.get("swap_free_bytes"), int) and not isinstance(value.get("swap_free_bytes"), bool)
             and 0 <= value["swap_free_bytes"] <= value["swap_total_bytes"], label + " host-pressure swap")
    for key in ("memory_psi_some_avg10", "memory_psi_full_avg10"):
        number = value.get(key)
        _require(isinstance(number, (int, float)) and not isinstance(number, bool)
                 and math.isfinite(number) and number >= 0.0, label + " host-pressure PSI")
    passed = (value["mem_available_bytes"] >= value["mem_available_floor_bytes"]
              and value["memory_psi_some_avg10"] <= value["memory_psi_some_avg10_ceiling"]
              and value["memory_psi_full_avg10"] == value["memory_psi_full_avg10_ceiling"])
    _require(value.get("pass") is passed, label + " host-pressure pass")
    # The receipt is an admission/final-success observation, not merely a
    # diagnostic sample.  A coherently recorded failed sample belongs only in
    # the V4 failure branch and can never validate a producer terminal.
    _require(passed, label + " host-pressure gate")
    return value["observed_monotonic_ns"]


def _scheduler_observation(value: Any, *, profile: dict[str, Any], label: str,
                           allow_historical_live: bool) -> int:
    """Validate the addendum's narrow tracked-route isolation evidence."""
    _require(isinstance(value, dict) and set(value) == {"observed_affinity", "tracked_other_route", "overlap", "host_pressure"},
             label + " scheduler observation schema")
    affinity = profile["logical_cpu_affinity"]
    _require(value.get("observed_affinity") == affinity, label + " scheduler affinity")
    tracked = value.get("tracked_other_route")
    _require(isinstance(tracked, dict) and set(tracked) == {"identity", "pid", "live", "observed_affinity"},
             label + " tracked route schema")
    other_affinity = tracked.get("observed_affinity")
    _require(isinstance(other_affinity, list) and all(isinstance(cpu, int) and cpu >= 0 for cpu in other_affinity)
             and len(other_affinity) == len(set(other_affinity)), label + " tracked route affinity")
    if tracked.get("live") is False:
        _require(tracked == {"identity": "none", "pid": None, "live": False, "observed_affinity": []},
                 label + " inactive tracked route")
    else:
        _require(tracked.get("live") is True and tracked.get("identity") == "cdm_p1_cross_v1"
                 and isinstance(tracked.get("pid"), int) and tracked["pid"] > 0 and other_affinity,
                 label + " live tracked route")
        _require(allow_historical_live, label + " live tracked route inadmissible")
    overlap = sorted(set(affinity).intersection(other_affinity))
    _require(value.get("overlap") == overlap and overlap == [], label + " scheduler overlap")
    return _host_pressure(value["host_pressure"], label=label)


def _scheduler(value: Any, *, arm: V4AdmissionProducerArm, placement: str,
               attempt_observation: dict[str, Any] | None = None,
               launch_observation: dict[str, Any] | None = None,
               allow_historical_live: bool = False) -> None:
    profile = _scheduler_profile(arm)
    required = {
        "attempt": {"profile", "before_attempt"},
        "launch": {"profile", "after_attempt"},
        "terminal": {"profile", "before_attempt", "after_attempt", "final"},
    }[placement]
    _require(isinstance(value, dict) and set(value) == required and value.get("profile") == profile,
             "V4 " + placement + " scheduler profile")
    if placement == "attempt":
        _scheduler_observation(value["before_attempt"], profile=profile, label="V4 before-attempt",
                               allow_historical_live=allow_historical_live)
    elif placement == "launch":
        _scheduler_observation(value["after_attempt"], profile=profile, label="V4 after-attempt",
                               allow_historical_live=allow_historical_live)
    else:
        _require(value.get("before_attempt") == attempt_observation
                 and value.get("after_attempt") == launch_observation, "V4 terminal scheduler historical evidence")
        before_ns = _scheduler_observation(value["before_attempt"], profile=profile, label="V4 terminal before-attempt",
                                           allow_historical_live=allow_historical_live)
        after_ns = _scheduler_observation(value["after_attempt"], profile=profile, label="V4 terminal after-attempt",
                                          allow_historical_live=allow_historical_live)
        final_ns = _scheduler_observation(value["final"], profile=profile, label="V4 final scheduler",
                                          allow_historical_live=allow_historical_live)
        _require(before_ns <= after_ns <= final_ns, "V4 scheduler monotonic host-pressure time")


def _final_four_manifest(manifest: Any, checkpoints: list[dict[str, Any]], *, label: str) -> None:
    """Bind arithmetic SWA provenance, not merely a reload proof."""
    _require(isinstance(manifest, dict), label + " manifest object")
    inherited = manifest.get("inherited_final_four")
    _require(isinstance(inherited, dict), label + " final-four manifest")
    components = inherited.get("components")
    _require(isinstance(components, list) and [item.get("sha256") for item in components if isinstance(item, dict)]
             == [item["sha256"] for item in checkpoints], label + " final-four components")
    _require(inherited.get("fp64_arithmetic") is True and inherited.get("optimizer_state_included") is False
             and inherited.get("strict_reload_verified") is True and inherited.get("finite_forward_tensors") is True,
             label + " final-four arithmetic")
    for key in ("floating_tensor_count", "buffer_tensor_count", "uninitialized_lazy_tensor_count"):
        _require(isinstance(inherited.get(key), int) and inherited[key] >= 0, label + " final-four tensor count")
    _require(inherited["floating_tensor_count"] > 0, label + " final-four floating tensors")


def _verify_v3_p0(root: Path, binding: MixedPACDProducerBinding) -> dict[str, Any]:
    arm = binding.p0
    base = arm.root_relative
    terminal = v1.descriptor_json(root, base + "/terminal.json", arm.terminal_sha256)
    adapter = v1.ProducerArm("P0", base, arm.terminal_sha256, arm.swa_sha256, arm.manifest_sha256,
                             arm.attempt_sha256, arm.launch_sha256, arm.source_authority_sha256,
                             arm.checkpoint_sha256s)
    _require(terminal.get("schema") == "pacd_matched_full_training_v3_terminal" and terminal.get("status") == "PACD_FULL_TRAINING_COMPLETE", "V3 P0 terminal")
    _require(terminal.get("arm") == "p0" and terminal.get("target_access") is False, "V3 P0 source-only")
    _require(terminal.get("source_closure", {}).get("launch") == terminal.get("source_closure", {}).get("final"), "V3 P0 closure equality")
    _require(terminal.get("source_closure", {}).get("final", {}).get("closure_sha256") == plan.V3_HISTORICAL_CLOSURE, "V3 historical closure")
    predecessor = terminal.get("predecessor")
    _require(isinstance(predecessor, dict) and predecessor.get("failure_sha256") == binding.v2_full_failure_sha256 and predecessor.get("relative") == binding.v2_full_failure_root_relative, "V3 direct V2 lineage")
    lower_v2_failure = _verify_v2_full_failure(root, binding)
    _require(predecessor == {"failure_sha256": lower_v2_failure["failure_sha256"],
                             "relative": lower_v2_failure["relative"],
                             "topology": lower_v2_failure["topology"],
                             "v1_predecessor": lower_v2_failure["v1_predecessor"]},
             "V3 exact nested V2 witness")
    v1.validate_v3_full_arm_graph(root, adapter, terminal)
    # The shared V1 graph checker establishes terminal topology and receipt
    # semantics.  Bind the opaque final-four and SWA bytes here as well: the
    # mixed successor must never accept a P0 witness whose terminal merely
    # names a missing/replaced checkpoint tensor.
    for epoch, digest in zip((44, 45, 46, 47), arm.checkpoint_sha256s, strict=True):
        _descriptor(root, base, f"epoch{epoch:03d}.pt", digest)
    _descriptor(root, base, "swa_final4.pt", arm.swa_sha256)
    _descriptor(root, base, "manifest.json", arm.manifest_sha256)
    epoch_items = terminal.get("epochs", [])
    _require([item.get("sha256") for item in epoch_items] == list(arm.epoch_sha256s), "V3 P0 epoch literals")
    authority = v1.descriptor_json(root, base + "/source_authority.json", arm.source_authority_sha256)
    sampler = authority.get("sampler", {})
    _require(isinstance(authority.get("ordered_source_roster"), list) and len(authority["ordered_source_roster"]) == 27
             and isinstance(authority.get("normalizers"), dict) and isinstance(authority.get("t4_authority_sha256"), str)
             and isinstance(authority.get("window"), dict) and isinstance(authority.get("initial_state"), dict),
             "V3 P0 source authority detail")
    for epoch, item in enumerate(epoch_items):
        _require(isinstance(item, dict) and set(item) == {"name", "sha256", "sidecar"}
                 and item.get("name") == f"epoch{epoch:03d}.json"
                 and item.get("sidecar") == f"epoch{epoch:03d}.json.sha256", "V3 P0 epoch descriptor")
        receipt = v1.descriptor_json(root, base + "/" + item["name"], item["sha256"])
        coverage = receipt.get("gradient_coverage", {})
        _require(receipt.get("adam_finite") is True and receipt.get("rng_violations") == 0
                 and receipt.get("prefix_mutations") == 0
                 and receipt.get("p0_prediction_mismatches") == 0
                 and receipt.get("p0_identity_mismatches") == 0, "V3 P0 epoch evidence")
        _require(coverage.get("zero_decoder_steps") == 0
                 and coverage.get("positive_encoder_steps", 0) > 0
                 and coverage.get("positive_decoder_steps", 0) > 0
                 and coverage.get("accepted_zero_encoder_steps") == coverage.get("zero_encoder_steps")
                 and coverage.get("zero_encoder_reason") == "all_units_dropped_valid_zero", "V3 P0 gradient coverage")
        _require(receipt.get("sampler_order") == {"window_indices_sha256": sampler.get("window_indices_sha256"),
                                                   "batched_indices_sha256": sampler.get("batched_indices_sha256")},
                 "V3 P0 sampler order")
    p0_manifest = v1.descriptor_json(root, base + "/manifest.json", arm.manifest_sha256)
    p0_checkpoints = [{"name": f"epoch{i:03d}.pt", "sha256": digest, "sidecar": f"epoch{i:03d}.pt.sha256", "epoch": i}
                      for i, digest in zip((44, 45, 46, 47), arm.checkpoint_sha256s, strict=True)]
    _final_four_manifest(p0_manifest, p0_checkpoints, label="V3 P0")
    root_identity = _held_directory_identity(root, base)
    return {"root_relative": base, "attempt_sha256": arm.attempt_sha256, "launch_sha256": arm.launch_sha256,
            "source_authority_sha256": arm.source_authority_sha256, "terminal_sha256": arm.terminal_sha256,
            "manifest_sha256": arm.manifest_sha256, "swa_sha256": arm.swa_sha256,
            "epoch_sha256s": list(arm.epoch_sha256s), "checkpoint_sha256s": list(arm.checkpoint_sha256s), "historical_v3_closure_sha256": plan.V3_HISTORICAL_CLOSURE,
            "v2_failure_sha256": binding.v2_full_failure_sha256, "v2_failure": lower_v2_failure,
            "root_identity": root_identity,
            "source_contract": {key: authority[key] for key in ("ordered_source_roster", "normalizers", "t4_authority_sha256", "window", "initial_state", "sampler")}}


def _held_directory_identity(root: Path, relative: str) -> dict[str, int | str]:
    """Snapshot parent and named-directory identity through O_NOFOLLOW FDs."""
    path = Path(root) / _relative(relative, "held directory relative")
    try:
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise MixedBindingError("held directory parent") from exc
    try:
        try:
            directory_fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                   dir_fd=parent_fd)
        except OSError as exc:
            raise MixedBindingError("held named directory") from exc
        try:
            parent_stat, directory_stat = os.fstat(parent_fd), os.fstat(directory_fd)
            return {"parent_device": int(parent_stat.st_dev), "parent_inode": int(parent_stat.st_ino),
                    "directory_device": int(directory_stat.st_dev), "directory_inode": int(directory_stat.st_ino),
                    "name": path.name}
        finally:
            os.close(directory_fd)
    finally:
        os.close(parent_fd)


def _verify_v4_arm(root: Path, arm: V4AdmissionProducerArm, p0: dict[str, Any], binding: MixedPACDProducerBinding) -> dict[str, Any]:
    base = arm.root_relative
    terminal = v1.descriptor_json(root, base + "/terminal.json", arm.terminal_sha256)
    _require(terminal.get("schema") == "pacd_p1_p2_admission_multigpu_v4_terminal"
             and terminal.get("status") == "PACD_FULL_TRAINING_COMPLETE"
             and terminal.get("arm") == arm.arm.lower() and terminal.get("target_access") is False,
             "V4 terminal semantics")
    names = {"attempt.json", "launch.json", "source_authority.json", "terminal.json", "manifest.json", "swa_final4.pt",
             *[f"epoch{i:03d}.json" for i in range(48)], *[f"epoch{i:03d}.pt" for i in (44, 45, 46, 47)]}
    _require(v1._held_leaf_set(root, base) == names | {name + ".sha256" for name in names},
             "V4 exact root topology")
    attempt = v1.descriptor_json(root, base + "/attempt.json", arm.attempt_sha256)
    launch = v1.descriptor_json(root, base + "/launch.json", arm.launch_sha256)
    authority = v1.descriptor_json(root, base + "/source_authority.json", arm.source_authority_sha256)
    _require(attempt.get("schema") == "pacd_p1_p2_admission_multigpu_v4_attempt"
             and attempt.get("status") == "ATTEMPT_PUBLISHED"
             and attempt.get("cell") == "PACD_P1_P2_ADMISSION_MULTIGPU_V4"
             and attempt.get("arm") == arm.arm.lower()
             and attempt.get("arm_identity") == {"short_m": arm.short_m, "root": base}, "V4 attempt")
    _require(attempt.get("budget") == {"epochs": 48, "steps_per_epoch": 33925, "total_steps": 1628400,
                                        "seed": 42, "batch_size": 32, "num_workers": 0}
             and attempt.get("target_access") is False, "V4 inherited attempt budget")
    _require(launch.get("schema") == "pacd_p1_p2_admission_multigpu_v4_launch"
             and launch.get("attempt", {}).get("sha256") == arm.attempt_sha256
             and launch.get("source_authority", {}).get("sha256") == arm.source_authority_sha256,
             "V4 launch links")
    _require(authority.get("attempt_sha256") == arm.attempt_sha256 and authority.get("target_access") is False
             and authority.get("source_roster_n") == 27 and authority.get("val") == [] and authority.get("test") == [],
             "V4 source authority")
    sampler = authority.get("sampler", {})
    _require(isinstance(sampler, dict) and all(sampler.get(k) == v for k, v in
             (("class", "SessionBatchSampler"), ("batch_size", 32), ("shuffle", True), ("seed", 42),
              ("num_workers", 0), ("steps_per_epoch", 33925))), "V4 sampler")
    _require({key: authority.get(key) for key in ("ordered_source_roster", "normalizers", "t4_authority_sha256", "window", "initial_state", "sampler")}
             == p0["source_contract"], "V4/P0 source authority parity")
    profile = {"identity": f"v4-{arm.arm.lower()}-gpu{arm.physical_index}", "physical_index": str(arm.physical_index),
               "expected_uuid": arm.expected_uuid, "expected_name": "NVIDIA GeForce RTX 3090",
               "expected_driver": "535.309.01", "expected_pci_bus_id": arm.expected_pci_bus_id,
               "cuda_visible_devices": arm.cuda_visible_devices, "logical_device": "cuda:0"}
    attempt_device = attempt.get("device", {})
    _require(isinstance(attempt_device, dict) and set(attempt_device) == {"profile", "observer_kind", "issuance_idle", "pre_attempt_idle"},
             "V4 attempt device schema")
    _require(attempt_device.get("profile") == profile
             and attempt_device.get("observer_kind") == "extended-fixed-device-production-v1", "V4 attempt device")
    _attestation(attempt_device.get("issuance_idle"), arm=arm, active=False, label="V4 issuance")
    _attestation(attempt_device.get("pre_attempt_idle"), arm=arm, active=False, label="V4 pre-attempt")
    allow_historical_live = binding.mode == "synthetic"
    _scheduler(attempt.get("scheduler"), arm=arm, placement="attempt",
               allow_historical_live=allow_historical_live)
    launch_device = launch.get("device", {})
    _require(isinstance(launch_device, dict) and set(launch_device) == {"profile", "observer_kind", "post_attempt_idle", "context_bound"},
             "V4 launch device schema")
    _require(launch_device.get("profile") == profile
             and launch_device.get("observer_kind") == "extended-fixed-device-production-v1", "V4 launch device")
    _attestation(launch_device.get("post_attempt_idle"), arm=arm, active=False, label="V4 post-attempt")
    _context(launch_device.get("context_bound"), label="V4 launch")
    _scheduler(launch.get("scheduler"), arm=arm, placement="launch",
               allow_historical_live=allow_historical_live)

    predecessor = terminal.get("predecessor", {})
    p0w = predecessor.get("required_v3_p0", {}) if isinstance(predecessor, dict) else {}
    _require(isinstance(predecessor, dict) and set(predecessor) == {"schema", "v2_failure", "required_v3_p0"}
             and predecessor.get("schema") == "pacd_p1_p2_admission_multigpu_v4_predecessor_v1"
             and isinstance(p0w, dict) and p0w.get("schema") == "pacd_v3_p0_admission_witness_v1", "V4 predecessor schema")
    _require(attempt.get("predecessor") == predecessor, "V4 attempt predecessor")
    _require(predecessor.get("v2_failure") == p0["v2_failure"], "V4 nested V2 witness")
    _require(p0w.get("root_relative") == p0["root_relative"]
             and p0w.get("attempt") == {"name": "attempt.json", "sha256": p0["attempt_sha256"]}
             and p0w.get("launch") == {"name": "launch.json", "sha256": p0["launch_sha256"]}
             and p0w.get("source_authority") == {"name": "source_authority.json", "sha256": p0["source_authority_sha256"]}
             and p0w.get("terminal") == {"name": "terminal.json", "sha256": p0["terminal_sha256"]}
             and p0w.get("manifest") == {"name": "manifest.json", "sha256": p0["manifest_sha256"]}
             and p0w.get("swa") == {"name": "swa_final4.pt", "sha256": p0["swa_sha256"]}
             and p0w.get("historical_v3_closure_sha256") == plan.V3_HISTORICAL_CLOSURE
             and p0w.get("v2_failure_sha256") == binding.v2_full_failure_sha256
             and p0w.get("root_identity") == p0["root_identity"], "V4 P0 witness")
    _require(set(p0w) == {"schema", "root_relative", "root_identity", "attempt", "launch", "source_authority",
                           "epochs", "checkpoints", "swa", "manifest", "terminal",
                           "historical_v3_closure_sha256", "v2_failure_sha256"}, "V4 P0 witness keys")
    _require(p0w.get("epochs") == [{"name": f"epoch{i:03d}.json", "sha256": digest}
                                     for i, digest in enumerate(p0["epoch_sha256s"])]
             and p0w.get("checkpoints") == [{"name": f"epoch{i:03d}.pt", "sha256": digest, "epoch": i}
                                              for i, digest in zip((44, 45, 46, 47), p0["checkpoint_sha256s"], strict=True)],
             "V4 P0 witness arrays")
    _require(_held_directory_identity(root, p0["root_relative"]) == p0["root_identity"], "V4 P0 held root identity")

    _require(terminal.get("attempt") == {"name": "attempt.json", "sha256": arm.attempt_sha256, "sidecar": "attempt.json.sha256"}
             and terminal.get("launch") == {"name": "launch.json", "sha256": arm.launch_sha256, "sidecar": "launch.json.sha256"}
             and terminal.get("source_authority") == {"name": "source_authority.json", "sha256": arm.source_authority_sha256, "sidecar": "source_authority.json.sha256"},
             "V4 terminal descriptor links")
    _require(terminal.get("source_closure", {}).get("launch") == terminal.get("source_closure", {}).get("final")
             and terminal.get("source_closure", {}).get("final", {}).get("closure_sha256") == arm.closure_sha256,
             "V4 closure equality")
    _require(attempt.get("source_closure", {}).get("closure_sha256") == arm.closure_sha256,
             "V4 attempt closure")
    _require(terminal.get("progress", {}).get("epochs_published") == 48
             and terminal.get("progress", {}).get("checkpoints_published") == 4
             and terminal.get("progress", {}).get("swa_published") is True, "V4 completion")
    epochs = terminal.get("epochs", [])
    _require(isinstance(epochs, list) and len(epochs) == 48, "V4 epoch descriptors")
    for epoch, (item, digest) in enumerate(zip(epochs, arm.epoch_sha256s, strict=True)):
        _exact_descriptor(item, name=f"epoch{epoch:03d}.json", digest=digest, label="V4 epoch")
        receipt = v1.descriptor_json(root, base + "/" + item["name"], digest)
        _require(receipt.get("epoch") == epoch and receipt.get("optimizer_steps") == 33925
                 and receipt.get("cumulative_optimizer_steps") == (epoch + 1) * 33925
                 and receipt.get("adam_finite") is True, "V4 epoch step/Adam law")
        # P1/P2 deliberately change the short calibration view.  Their
        # prediction/identity mismatch counters are descriptive accounting,
        # not P0's matched-compute equality gate; requiring zero here would
        # reject the intended M4/M10 treatment arms.  The paired RNG/mask,
        # prefix, finite, and gradient contracts remain hard gates.
        _require(receipt.get("rng_violations") == 0 and receipt.get("prefix_mutations") == 0
                 and all(isinstance(receipt.get(key), int) and receipt.get(key) >= 0
                         for key in ("p0_prediction_mismatches", "p0_identity_mismatches"))
                 and receipt.get("parameter_finiteness", {}).get("violations") == 0, "V4 paired finite law")
        coverage = receipt.get("gradient_coverage", {})
        _require(coverage.get("zero_decoder_steps") == 0 and coverage.get("positive_encoder_steps", 0) > 0
                 and coverage.get("positive_decoder_steps", 0) > 0
                 and coverage.get("accepted_zero_encoder_steps") == coverage.get("zero_encoder_steps")
                 and coverage.get("zero_encoder_reason") == "all_units_dropped_valid_zero", "V4 gradient coverage")
        _require(isinstance(receipt.get("sentinels"), list) and len(receipt["sentinels"]) == 4,
                 "V4 sentinel topology")
        _require(receipt.get("sampler_order") == {"window_indices_sha256": sampler.get("window_indices_sha256"),
                                                   "batched_indices_sha256": sampler.get("batched_indices_sha256")},
                 "V4 sampler-order evidence")
    checkpoints = terminal.get("checkpoints", [])
    _require(isinstance(checkpoints, list) and len(checkpoints) == 4, "V4 checkpoint descriptors")
    for epoch, item, digest in zip((44, 45, 46, 47), checkpoints, arm.checkpoint_sha256s, strict=True):
        _exact_descriptor(item, name=f"epoch{epoch:03d}.pt", digest=digest, epoch=epoch, label="V4 checkpoint")
        _descriptor(root, base, item["name"], digest)
    _exact_descriptor(terminal.get("swa"), name="swa_final4.pt", digest=arm.swa_sha256,
                      extra=frozenset({"proof", "strict_loaded_state_sha256"}), label="V4 SWA")
    _exact_descriptor(terminal.get("manifest"), name="manifest.json", digest=arm.manifest_sha256, label="V4 manifest")
    manifest = v1.descriptor_json(root, base + "/manifest.json", arm.manifest_sha256)
    _descriptor(root, base, "swa_final4.pt", arm.swa_sha256)
    manifest_swa = {key: value for key, value in terminal["swa"].items() if key != "proof"}
    _require(manifest.get("attempt") == terminal.get("attempt") and manifest.get("swa") == manifest_swa
             and manifest.get("checkpoints") == checkpoints
             and manifest.get("implementation_closure_sha256") == arm.closure_sha256,
             "V4 manifest links")
    _final_four_manifest(manifest, checkpoints, label="V4")
    proof = terminal.get("swa", {}).get("proof", {})
    _require(proof.get("strict_load") is True and proof.get("eval") is True and proof.get("no_grad") is True
             and proof.get("repeated_forward_bitwise_equal") is True and proof.get("output_finite") is True
             and proof.get("dynamic_dropout_calls") == 0
             and proof.get("state_before_sha256") == proof.get("state_after_sha256")
             == terminal.get("swa", {}).get("strict_loaded_state_sha256"), "V4 SWA proof")
    final_device = terminal.get("device", {})
    _require(isinstance(final_device, dict) and set(final_device) == {"profile", "observer_kind", "issuance_idle",
             "pre_attempt_idle", "post_attempt_idle", "context_bound", "final", "peak_allocated", "peak_reserved", "tf32"},
             "V4 terminal device schema")
    _require(final_device.get("profile") == profile
             and final_device.get("observer_kind") == "extended-fixed-device-production-v1", "V4 terminal device")
    # The terminal repeats immutable admission observations.  Re-attesting the
    # terminal values alone is insufficient: a receipt could otherwise splice
    # distinct, independently valid device facts across lifecycle stages.
    _require(final_device.get("issuance_idle") == attempt_device.get("issuance_idle"),
             "V4 terminal issuance-attestation drift")
    _require(final_device.get("pre_attempt_idle") == attempt_device.get("pre_attempt_idle"),
             "V4 terminal pre-attempt-attestation drift")
    _require(final_device.get("post_attempt_idle") == launch_device.get("post_attempt_idle"),
             "V4 terminal post-attempt-attestation drift")
    _attestation(final_device.get("issuance_idle"), arm=arm, active=False, label="V4 terminal issuance")
    _attestation(final_device.get("pre_attempt_idle"), arm=arm, active=False, label="V4 terminal pre-attempt")
    _attestation(final_device.get("post_attempt_idle"), arm=arm, active=False, label="V4 terminal post-attempt")
    _context(final_device.get("context_bound"), label="V4 terminal context")
    _final_active(final_device.get("final"), label="V4 terminal final")
    _require(final_device["final"]["current_pid"] == final_device["context_bound"]["current_pid"]
             and final_device["final"]["selected_gpu_compute_pids"]
             == final_device["context_bound"]["selected_gpu_compute_pids"], "V4 final/context PID drift")
    _scheduler(terminal.get("scheduler"), arm=arm, placement="terminal",
               attempt_observation=attempt["scheduler"]["before_attempt"],
               launch_observation=launch["scheduler"]["after_attempt"],
               allow_historical_live=allow_historical_live)
    _require(final_device.get("context_bound") == launch_device.get("context_bound"), "V4 context stage drift")
    _require(isinstance(final_device.get("peak_allocated"), int) and final_device.get("peak_allocated") >= 0
             and isinstance(final_device.get("peak_reserved"), int) and final_device.get("peak_reserved") >= 0
             and final_device.get("tf32") == {"matmul_allow_tf32": False, "cudnn_allow_tf32": True}, "V4 terminal runtime evidence")
    return {"root_relative": base, "terminal_sha256": arm.terminal_sha256, "swa_sha256": arm.swa_sha256,
            "manifest_sha256": arm.manifest_sha256, "checkpoint_sha256s": list(arm.checkpoint_sha256s)}
