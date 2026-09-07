"""Typed artifact bindings; no path discovery and no producer-literal guessing."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import plan


class BindingError(RuntimeError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise BindingError(message)


def _sha(value: str, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), label)
    return value


def _root_relative(value: str, label: str) -> str:
    candidate = Path(value)
    _require(not candidate.is_absolute() and ".." not in candidate.parts and str(candidate) == value, label)
    return value


@dataclass(frozen=True)
class ProducerArm:
    arm: str
    root_relative: str
    terminal_sha256: str
    swa_sha256: str
    manifest_sha256: str
    attempt_sha256: str
    launch_sha256: str
    source_authority_sha256: str
    checkpoint_sha256s: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        expected = {"P0": "p0_fullfull_seed42", "P1": "p1_m4_seed42", "P2": "p2_m10_seed42"}
        _require(self.arm in expected, "unknown PACD arm")
        _require(self.root_relative.startswith("tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/"), "PACD V3 producer family")
        _require(self.root_relative.endswith("/" + expected[self.arm]), "PACD arm/root identity")
        _root_relative(self.root_relative, "PACD producer root")
        for label, value in (("terminal", self.terminal_sha256), ("swa", self.swa_sha256),
                             ("manifest", self.manifest_sha256), ("attempt", self.attempt_sha256),
                             ("launch", self.launch_sha256), ("source authority", self.source_authority_sha256)):
            _sha(value, label)
        _require(len(self.checkpoint_sha256s) == 4, "final-four checkpoint topology")
        for value in self.checkpoint_sha256s:
            _sha(value, "checkpoint")


@dataclass(frozen=True)
class PACDProducerBinding:
    """Immutable P0/P1/P2 producer identity, either deferred-live or test-only."""

    TYPE_IDENTITY = "PACDProducerBinding:v1"

    p0: ProducerArm
    p1: ProducerArm
    p2: ProducerArm
    mode: str
    accepted_v2_smoke_terminal_sha256: str
    accepted_v2_smoke_root_relative: str
    v2_full_failure_sha256: str
    v2_full_failure_root_relative: str

    def __post_init__(self) -> None:
        _require(self.mode in {"synthetic", "live"}, "binding mode")
        _sha(self.accepted_v2_smoke_terminal_sha256, "V2 smoke predecessor")
        _root_relative(self.accepted_v2_smoke_root_relative, "V2 smoke predecessor root")
        _sha(self.v2_full_failure_sha256, "V2 full-failure predecessor")
        _root_relative(self.v2_full_failure_root_relative, "V2 full-failure predecessor root")
        if self.mode == "live":
            # This branch is intentionally impossible until reviewed exact
            # literals are inserted in the immutable plan.
            _require(plan.LIVE_PACD_PRODUCER_LITERALS is not None, "PACD live producer literals are deferred")

    @property
    def arms(self) -> tuple[ProducerArm, ProducerArm, ProducerArm]:
        return (self.p0, self.p1, self.p2)

    def payload(self) -> dict:
        return {
            "schema": "pacd_producer_binding_v1", "mode": self.mode,
            "accepted_v2_smoke_terminal_sha256": self.accepted_v2_smoke_terminal_sha256,
            "accepted_v2_smoke_root_relative": self.accepted_v2_smoke_root_relative,
            "v2_full_failure_sha256": self.v2_full_failure_sha256,
            "v2_full_failure_root_relative": self.v2_full_failure_root_relative,
            "arms": [arm.__dict__ for arm in self.arms],
        }

    def require_live(self) -> None:
        _require(self.mode == "live", "synthetic PACD producer binding cannot score")
        _require(plan.LIVE_PACD_PRODUCER_LITERALS is not None, "PACD live producer literals are deferred")

    @classmethod
    def synthetic(cls, *, digest: str = "a" * 64) -> "PACDProducerBinding":
        """Typed no-data test fixture; never accepted by live capability minting."""
        return cls(
            *(ProducerArm(arm, f"tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/{suffix}", digest, digest, digest, digest, digest, digest,
                           (digest, digest, digest, digest))
              for arm, suffix in (("P0", "p0_fullfull_seed42"), ("P1", "p1_m4_seed42"), ("P2", "p2_m10_seed42"))),
            mode="synthetic", accepted_v2_smoke_terminal_sha256=digest,
            accepted_v2_smoke_root_relative="tfpd_exploration/results/paired_anchored_calibration_dropout_v2/smoke_seed42",
            v2_full_failure_sha256=digest,
            v2_full_failure_root_relative="tfpd_exploration/results/paired_anchored_calibration_dropout_full_v2/p0_fullfull_seed42",
        )


def _read_fd(fd: int) -> bytes:
    parts: list[bytes] = []
    while part := os.read(fd, 65_536):
        parts.append(part)
    return b"".join(parts)


def _descriptor_body(root: Path, relative: str, expected_sha256: str) -> tuple[bytes, str]:
    """Read one regular body/sidecar through no-follow descriptors."""
    relative = _root_relative(relative, "artifact relative path")
    _sha(expected_sha256, "artifact expected SHA")
    path = Path(root) / relative
    parent = path.parent
    try:
        dfd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise BindingError("artifact parent descriptor") from exc
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        except OSError as exc:
            raise BindingError("artifact body descriptor") from exc
        try:
            st = os.fstat(fd)
            _require(stat.S_ISREG(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o444, "artifact mode/type")
            body = _read_fd(fd)
        finally:
            os.close(fd)
        actual = hashlib.sha256(body).hexdigest()
        _require(actual == expected_sha256, "artifact digest drift")
        side_name = path.name + ".sha256"
        try:
            sfd = os.open(side_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        except OSError as exc:
            raise BindingError("artifact sidecar descriptor") from exc
        try:
            sst = os.fstat(sfd)
            _require(stat.S_ISREG(sst.st_mode) and stat.S_IMODE(sst.st_mode) == 0o444, "artifact sidecar mode/type")
            canonical = _read_fd(sfd).decode("utf-8")
        finally:
            os.close(sfd)
    finally:
        os.close(dfd)
    _require(canonical == f"{expected_sha256}  {path.name}\n", "artifact sidecar drift")
    return body, side_name


def descriptor_sha256(root: Path, relative: str, expected_sha256: str) -> dict:
    """Held-descriptor, no-follow artifact rehash; content is not deserialized."""
    body, side = _descriptor_body(root, relative, expected_sha256)
    del body
    return {"relative": relative, "sha256": expected_sha256, "sidecar": side}


def descriptor_json(root: Path, relative: str, expected_sha256: str) -> dict:
    """JSON receipt decoded from the already SHA-checked held descriptor bytes."""
    body, _side = _descriptor_body(root, relative, expected_sha256)
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BindingError("receipt JSON drift") from exc
    _require(isinstance(value, dict), "receipt JSON object")
    return value


def verify_historical_comparators(root: Path) -> dict:
    """Rehash and route-validate historical final-four producers.

    T0/C1 are CAL-AUG terminals while SD is a POP-ROBUST terminal; sharing a
    loose ``dict`` codec would make a schema drift look like a valid control.
    """
    verified: dict[str, dict] = {}
    for system, value in plan.HISTORICAL_SYSTEMS.items():
        terminal = descriptor_json(root, value["terminal"], value["terminal_sha256"])
        swa = descriptor_sha256(root, value["swa"], value["swa_sha256"])
        _validate_historical_terminal(system, terminal, value, root)
        verified[system] = {"terminal": descriptor_sha256(root, value["terminal"], value["terminal_sha256"]), "swa": swa,
                            "checkpoint_sha256s": [item["sha256"] for item in terminal["checkpoints"]],
                            "schema": terminal["schema"], "source_closure": terminal["source_closure"]}
    return verified


def _validate_historical_terminal(system: str, terminal: Mapping[str, object], _value: Mapping[str, str], root: Path) -> None:
    if system in {"T0", "C1"}:
        _require(terminal.get("schema") == "cal_aug_v1_cell" and terminal.get("status") == "CAL_AUG_CELL_TERMINAL", "CAL-AUG terminal schema")
        _require(terminal.get("arm") == system.lower(), "CAL-AUG arm")
    else:
        _require(terminal.get("schema") == "tfpd_pop_robust_cell_v1" and terminal.get("status") == "CELL_TERMINAL", "SD terminal schema")
    _require(terminal.get("epochs_run") == 48, "historical 48 epochs")
    budget = terminal.get("budget")
    _require(isinstance(budget, dict) and budget.get("steps_per_epoch") == 33925 and budget.get("total_optimizer_steps") == 1628400, "historical fixed step budget")
    optimizer = budget.get("optimizer", {}) if isinstance(budget, dict) else {}
    _require(isinstance(optimizer, dict) and optimizer == {"amsgrad": False, "betas": [0.9, 0.999], "cls": "torch.optim.Adam", "eps": 1e-08, "lr": 0.0001, "weight_decay": 0.0}, "historical Adam contract")
    schedule = budget.get("schedule", {}) if isinstance(budget, dict) else {}
    _require(isinstance(schedule, dict) and all(schedule.get(key) == value for key, value in (("kind", "warmup_then_cosine"), ("steps_per_epoch", 33925), ("total_steps", 1628400), ("warmup_epochs", 2), ("warmup_steps", 67850), ("n_epochs", 48), ("phase_local_steps", True))), "historical schedule contract")
    data_contract = terminal.get("data_contract")
    _require(isinstance(data_contract, dict) and all(data_contract.get(key) == value for key, value in (("roster_n", 27), ("external_sub_m_opened", False), ("within_dev_sessions_opened", False), ("formal_or_organizer_held_data_opened", False))), "historical source-only contract")
    if system in {"T0", "C1"}:
        _require(data_contract.get("val_and_test_empty") is True and data_contract.get("val_paths_resolved") == [] and data_contract.get("test_paths_resolved") == [], "CAL-AUG val/test contract")
        disclosures = terminal.get("disclosures")
        _require(isinstance(disclosures, dict) and disclosures.get("target_updates_gradients_or_optimizer_steps") == 0, "CAL-AUG target-update disclosure")
    _require(terminal.get("invariant_failures") == [], "historical invariant failures")
    checkpoints = terminal.get("checkpoints")
    _require(isinstance(checkpoints, list) and len(checkpoints) == 4 and [item.get("epoch") for item in checkpoints if isinstance(item, dict)] == [44, 45, 46, 47], "historical final-four checkpoints")
    _require(all(isinstance(item, dict) and set(item) == {"epoch", "file", "sha256"} and isinstance(item["file"], str) and item["file"].endswith(f"epoch{int(item['epoch']):03d}.ckpt") and len(str(item["sha256"])) == 64 for item in checkpoints), "historical checkpoint descriptors")
    swa = terminal.get("swa")
    _require(isinstance(swa, dict) and swa.get("window_epochs") == [44, 45, 46, 47] and swa.get("sha256") == _value["swa_sha256"] and swa.get("path") == str((Path(root) / _value["swa"]).resolve()), "historical final-four SWA")
    _require(swa.get("strict_reload_finite_forward_smoke") is True, "historical SWA smoke")
    manifest = swa.get("manifest", {}) if isinstance(swa, dict) else {}
    _require(isinstance(manifest, dict) and manifest.get("strict_reload_verified") is True and manifest.get("finite_forward_tensors") is True and manifest.get("optimizer_state_included") is False, "historical SWA manifest")
    components = manifest.get("components", []) if isinstance(manifest, dict) else []
    expected_component_paths = [str((Path(root) / _value["terminal"]).parent / str(item["file"])) for item in checkpoints]
    _require(isinstance(components, list) and len(components) == 4 and [item.get("sha256") for item in components if isinstance(item, dict)] == [item["sha256"] for item in checkpoints] and [item.get("path") for item in components if isinstance(item, dict)] == expected_component_paths, "historical SWA components")
    closure = terminal.get("source_closure")
    _require(isinstance(closure, dict) and closure.get("launch_final_closure_equal") is True and closure.get("launch", {}).get("closure_sha256") == closure.get("final", {}).get("closure_sha256"), "historical launch/final closure")


def _held_leaf_set(root: Path, relative: str) -> set[str]:
    relative = _root_relative(relative, "held root relative")
    dfd = os.open(Path(root) / relative, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return set(os.listdir(dfd))
    finally:
        os.close(dfd)


def _declared_descriptor(root: Path, base: str, item: Mapping[str, object], label: str) -> dict:
    _require(set(item).issuperset({"name", "sha256", "sidecar"}), label + " descriptor fields")
    name, digest, side = item["name"], item["sha256"], item["sidecar"]
    _require(isinstance(name, str) and isinstance(digest, str) and side == name + ".sha256", label + " descriptor canonical")
    return descriptor_sha256(root, base + "/" + name, digest)


def _validate_v2_smoke_terminal(body: Mapping[str, object]) -> None:
    _require(body.get("schema") == "paired_anchored_calibration_dropout_v2_terminal" and body.get("status") == "PACD_SMOKE_COMPLETE", "V2 smoke terminal semantics")
    _require(body.get("target_access") is False and body.get("terminal_published") is True, "V2 smoke source-only")
    closure = body.get("source_closure")
    _require(isinstance(closure, dict) and closure.get("launch") == closure.get("final"), "V2 smoke closure")


def _validate_v2_full_failure(body: Mapping[str, object]) -> None:
    _require(body.get("schema") == "pacd_matched_full_training_v2_failure" and body.get("status") == "CELL_FAILED", "V2 full failure semantics")
    progress = body.get("progress")
    _require(isinstance(progress, dict) and progress.get("epochs_published") == 0 and progress.get("checkpoints_published") == 0 and progress.get("swa_published") is False, "V2 failure pre-epoch")
    _require(body.get("target_access") is False, "V2 failure no target")
    closure = body.get("source_closure")
    _require(isinstance(closure, dict) and closure.get("launch", {}).get("closure_sha256") == closure.get("final", {}).get("closure_sha256"), "V2 failure closure")


def verify_live_v3_producers(root: Path, binding: PACDProducerBinding) -> dict:
    """Descriptor-validate V3 terminals plus their V2 smoke/failure lineage.

    This intentionally reads receipts and artifact bytes only.  It does not
    open a target asset or deserialize a model/SWA tensor.  The live binding is
    impossible to mint until reviewed producer literals are available.
    """
    binding.require_live()
    smoke_body = descriptor_json(root, binding.accepted_v2_smoke_root_relative + "/terminal.json",
                                 binding.accepted_v2_smoke_terminal_sha256)
    _validate_v2_smoke_terminal(smoke_body)
    smoke = descriptor_sha256(root, binding.accepted_v2_smoke_root_relative + "/terminal.json",
                              binding.accepted_v2_smoke_terminal_sha256)
    failure_body = descriptor_json(root, binding.v2_full_failure_root_relative + "/failure.json",
                                   binding.v2_full_failure_sha256)
    _validate_v2_full_failure(failure_body)
    failure = descriptor_sha256(root, binding.v2_full_failure_root_relative + "/failure.json",
                                binding.v2_full_failure_sha256)
    verified = {"accepted_v2_smoke": smoke, "v2_full_failure": failure, "arms": {}}
    for arm in binding.arms:
        base = arm.root_relative
        terminal = descriptor_sha256(root, base + "/terminal.json", arm.terminal_sha256)
        manifest = descriptor_sha256(root, base + "/manifest.json", arm.manifest_sha256)
        attempt = descriptor_sha256(root, base + "/attempt.json", arm.attempt_sha256)
        launch = descriptor_sha256(root, base + "/launch.json", arm.launch_sha256)
        authority = descriptor_sha256(root, base + "/source_authority.json", arm.source_authority_sha256)
        swa = descriptor_sha256(root, base + "/swa_final4.pt", arm.swa_sha256)
        checkpoints = [descriptor_sha256(root, base + f"/epoch{epoch:03d}.pt", digest)
                       for epoch, digest in zip((44, 45, 46, 47), arm.checkpoint_sha256s, strict=True)]
        body = descriptor_json(root, base + "/terminal.json", arm.terminal_sha256)
        _require(body.get("schema") == "pacd_matched_full_training_v3_terminal" and body.get("status") == "PACD_FULL_TRAINING_COMPLETE", "V3 terminal semantics")
        _require(body.get("arm") == arm.arm.lower() and len(body.get("epochs", [])) == 48 and len(body.get("checkpoints", [])) == 4, "V3 epoch/final-four topology")
        budget = body.get("attempt", {}).get("sha256") == arm.attempt_sha256 and body.get("launch", {}).get("sha256") == arm.launch_sha256 and body.get("source_authority", {}).get("sha256") == arm.source_authority_sha256
        _require(budget, "V3 descriptor links")
        _require(body.get("progress", {}).get("epochs_published") == 48 and body.get("progress", {}).get("checkpoints_published") == 4 and body.get("progress", {}).get("swa_published") is True, "V3 completion progress")
        _require([item.get("sha256") for item in body.get("checkpoints", [])] == list(arm.checkpoint_sha256s), "V3 checkpoint body links")
        _require(body.get("swa", {}).get("sha256") == arm.swa_sha256 and body.get("manifest", {}).get("sha256") == arm.manifest_sha256, "V3 SWA/manifest links")
        swa_proof = body.get("swa", {}).get("proof", {})
        _require(swa_proof.get("strict_load") is True and swa_proof.get("state_before_sha256") == swa_proof.get("state_after_sha256"), "V3 SWA strict proof")
        _require(body.get("target_access") is False and body.get("source_closure", {}).get("launch") == body.get("source_closure", {}).get("final"), "V3 source-only/closure")
        predecessor = body.get("predecessor", {})
        _require(predecessor.get("relative") == binding.v2_full_failure_root_relative and predecessor.get("failure_sha256") == binding.v2_full_failure_sha256, "V3 nested V2 full-failure lineage")
        _validate_v3_arm_graph(root, arm, body)
        verified["arms"][arm.arm] = {"terminal": terminal, "manifest": manifest, "attempt": attempt,
                                     "launch": launch, "source_authority": authority, "swa": swa,
                                     "checkpoints": checkpoints}
    return verified


def _validate_v3_arm_graph(root: Path, arm: ProducerArm, terminal: Mapping[str, object]) -> None:
    """Validate all published V3 source-only training evidence by descriptor.

    The score path never deserializes the checkpoint/SWA tensors.  It does,
    however, refuse a terminal that merely names missing epoch receipts or a
    graph that contains an extra mutable leaf.
    """
    base = arm.root_relative
    epoch_items = terminal.get("epochs")
    checkpoint_items = terminal.get("checkpoints")
    _require(isinstance(epoch_items, list) and isinstance(checkpoint_items, list), "V3 terminal descriptor arrays")
    expected_epoch_names = [f"epoch{epoch:03d}.json" for epoch in range(48)]
    _require([item.get("name") for item in epoch_items if isinstance(item, dict)] == expected_epoch_names, "V3 epoch receipt order")
    _require([item.get("name") for item in checkpoint_items if isinstance(item, dict)] == [f"epoch{epoch:03d}.pt" for epoch in (44, 45, 46, 47)], "V3 checkpoint receipt order")
    expected = {"attempt.json", "launch.json", "source_authority.json", "terminal.json", "manifest.json", "swa_final4.pt", *expected_epoch_names,
                *[f"epoch{epoch:03d}.pt" for epoch in (44, 45, 46, 47)]}
    expected_leaves = expected | {name + ".sha256" for name in expected}
    _require(_held_leaf_set(root, base) == expected_leaves, "V3 exact root topology")
    attempt = descriptor_json(root, base + "/attempt.json", arm.attempt_sha256)
    launch = descriptor_json(root, base + "/launch.json", arm.launch_sha256)
    authority = descriptor_json(root, base + "/source_authority.json", arm.source_authority_sha256)
    _require(attempt.get("schema") == "pacd_matched_full_training_v3_attempt" and attempt.get("status") == "ATTEMPT_PUBLISHED", "V3 attempt semantics")
    budget = attempt.get("budget", {})
    _require(isinstance(budget, dict) and all(budget.get(key) == value for key, value in (("epochs", 48), ("steps_per_epoch", 33925), ("total_steps", 1628400), ("seed", 42), ("batch_size", 32), ("num_workers", 0))), "V3 fixed budget")
    _require(attempt.get("target_access") is False and attempt.get("arm") == arm.arm.lower(), "V3 attempt source-only/arm")
    _require(launch.get("schema") == "pacd_matched_full_training_v3_launch" and launch.get("target_access") is False and launch.get("attempt", {}).get("sha256") == arm.attempt_sha256 and launch.get("source_authority", {}).get("sha256") == arm.source_authority_sha256, "V3 launch links")
    _require(authority.get("attempt_sha256") == arm.attempt_sha256 and authority.get("target_access") is False and authority.get("val") == [] and authority.get("test") == [] and authority.get("source_roster_n") == 27, "V3 source authority")
    sampler = authority.get("sampler", {})
    _require(isinstance(sampler, dict) and all(sampler.get(key) == value for key, value in (("class", "SessionBatchSampler"), ("batch_size", 32), ("shuffle", True), ("seed", 42), ("num_workers", 0), ("steps_per_epoch", 33925))), "V3 sampler law")
    for epoch, item in enumerate(epoch_items):
        _require(isinstance(item, dict), "V3 epoch descriptor")
        receipt = descriptor_json(root, base + "/" + str(item["name"]), str(item["sha256"]))
        _require(receipt.get("epoch") == epoch and receipt.get("optimizer_steps") == 33925 and receipt.get("cumulative_optimizer_steps") == (epoch + 1) * 33925, "V3 epoch optimizer law")
        _require(receipt.get("p0_prediction_mismatches") == 0 and receipt.get("p0_identity_mismatches") == 0 and receipt.get("rng_violations") == 0 and receipt.get("prefix_mutations") == 0, "V3 paired/RNG law")
        finite = receipt.get("parameter_finiteness", {})
        _require(isinstance(finite, dict) and finite.get("violations") == 0, "V3 finite law")
        sentinels = receipt.get("sentinels", [])
        _require(isinstance(sentinels, list) and len(sentinels) == 4, "V3 sentinel topology")
    manifest = descriptor_json(root, base + "/manifest.json", arm.manifest_sha256)
    _require(manifest.get("attempt", {}).get("sha256") == arm.attempt_sha256 and manifest.get("swa", {}).get("sha256") == arm.swa_sha256, "V3 manifest links")
    _require([item.get("sha256") for item in manifest.get("checkpoints", []) if isinstance(item, dict)] == list(arm.checkpoint_sha256s), "V3 manifest checkpoint links")
    proof = terminal.get("swa", {}).get("proof", {}) if isinstance(terminal.get("swa"), dict) else {}
    _require(isinstance(proof, dict) and proof.get("strict_load") is True and proof.get("eval") is True and proof.get("no_grad") is True and proof.get("repeated_forward_bitwise_equal") is True and proof.get("output_finite") is True and proof.get("dynamic_dropout_calls") == 0 and proof.get("state_before_sha256") == proof.get("state_after_sha256"), "V3 SWA proof")


def validate_v3_full_arm_graph(root: Path, arm: ProducerArm, terminal: Mapping[str, object]) -> None:
    """Public strict graph primitive for lineage successors.

    It deliberately performs no route selection and retains the existing V1
    codec; callers must separately enforce whether the arm is P0.
    """
    _validate_v3_arm_graph(root, arm, terminal)
