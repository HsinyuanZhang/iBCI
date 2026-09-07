"""Attempt-first AOF-M source-only lifecycle; public CLI remains inert."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import binding, core, plan


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(value: object, label: str) -> str:
    _need(isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
          f"AOF-M {label} must be a lowercase SHA-256")
    return value


def _pair_names(bodies: dict[str, str]) -> tuple[str, ...]:
    return tuple(sorted((*bodies, *(f"{name}.sha256" for name in bodies))))


def execute(root: Path, *, reviewed_closure_sha256: str, _lifecycle=None, _gpu_attestor=None, _runner_factory=None):
    """Future GPU0-only source route.  Test seams are private keyword-only."""
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as default_lifecycle
    lifecycle = default_lifecycle if _lifecycle is None else _lifecycle
    root = Path(root).absolute()
    _need({key: os.environ.get(key) for key in plan.ENV} == plan.ENV, "AOF-M environment")
    _need(plan.file_sha256(root / plan.DESIGN_RELATIVE) == plan.DESIGN_SHA256, "AOF-M design SHA")
    _need(plan.file_sha256(root / plan.WORKORDER_RELATIVE) == plan.WORKORDER_SHA256, "AOF-M workorder SHA")
    closure = plan.closure(root)
    closure_sha = plan.sha256_bytes(plan.json_bytes(closure))
    reviewed_closure_sha256 = _sha256(reviewed_closure_sha256, "reviewed closure")
    _need(reviewed_closure_sha256 == closure_sha, "AOF-M reviewed closure mismatch")
    class Spec:
        root_relative = plan.ROOT_RELATIVE
        def payload(self):
            return {"schema": plan.SCHEMA + "_root", "root_relative": self.root_relative}
    artifact = lifecycle.ImmutableArtifactRoot.reserve(root, Spec())
    attempt = artifact.publish_json("attempt.json", {"schema": plan.SCHEMA + "_attempt",
        "design_sha256": plan.DESIGN_SHA256, "workorder_sha256": plan.WORKORDER_SHA256,
        "closure_sha256": closure_sha, "reviewed_closure_sha256": reviewed_closure_sha256,
        "closure": closure, "source_target_access_authorized": True,
        "source_target_access": False,
        "hidden_external_evalai_target_access": False, "predecessor_opened": False})
    published = {"attempt.json": attempt}
    progress = {"stage": "attempt_published", "predecessor_attempted": False, "predecessor_opened": False,
                "gpu_attempted": False, "gpu_opened": False, "source_prepare_attempted": False,
                "source_opened": False, "source_target_access_attempted": False, "source_target_access": False,
                "pairs_attempted": False, "pairs_complete": False,
                "hidden_external_evalai_target_access": False}
    terminal_published = False
    try:
        progress["stage"] = "predecessor"; progress["predecessor_attempted"] = True
        predecessor = binding.validate_aof_v1_graph(root); progress["predecessor_opened"] = True
        published["predecessor_authority.json"] = artifact.publish_json("predecessor_authority.json", predecessor)
        from tfpd_exploration.src.m2_anchored_output_fusion_v1.runner import gpu0
        progress["stage"] = "gpu_attestation"; progress["gpu_attempted"] = True
        launch = (gpu0 if _gpu_attestor is None else _gpu_attestor)(); progress["gpu_opened"] = True
        _need(launch.get("uuid") == plan.GPU_UUID and launch.get("logical_device") == 0, "AOF-M GPU0 identity")
        published["launch.json"] = artifact.publish_json("launch.json", {**launch, "closure_sha256": closure_sha,
            "reviewed_closure_sha256": reviewed_closure_sha256})
        from .runner import MatrixSourceMaterializer
        progress["stage"] = "source_construct"
        runner = (MatrixSourceMaterializer if _runner_factory is None else _runner_factory)(root, launch, predecessor)
        progress["stage"] = "source_prepare"; progress["source_prepare_attempted"] = True
        # PIT setup eagerly loader-preloads source behavior arrays.  Mark this
        # before the call so a prepare-time exception cannot claim otherwise.
        progress["source_target_access_attempted"] = True; progress["source_target_access"] = True
        source = runner.prepare(); progress["source_opened"] = True
        published["source_authority.json"] = artifact.publish_json("source_authority.json", {**source,
            "closure_sha256": closure_sha, "source_target_access_authorized": True,
            "source_target_access": progress["source_target_access"],
            "hidden_external_evalai_target_access": False})
        progress["stage"] = "one_forward"; progress["pairs_attempted"] = True
        pairs, evidence = runner.materialize_once(); progress["pairs_complete"] = True
        oof = core.leave_one_session_out(pairs)
        published["oof.json"] = artifact.publish_json("oof.json", {"oof": oof, "session_evidence": evidence,
            "arrays_process_local_only": True, "closure_sha256": closure_sha,
            "source_target_access_authorized": True, "source_target_access": progress["source_target_access"],
            "hidden_external_evalai_target_access": False})
        if oof["passed"]:
            all7 = core.fit_matrix(pairs, expected_sessions=plan.SESSIONS)
            published["all7_matrix.json"] = artifact.publish_json("all7_matrix.json", {
                key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in all7.items()})
        final = plan.closure(root)
        _need({key: os.environ.get(key) for key in plan.ENV} == plan.ENV
              and final == closure and plan.sha256_bytes(plan.json_bytes(final)) == closure_sha == reviewed_closure_sha256,
              "AOF-M final environment/closure drift")
        final_predecessor = binding.validate_aof_v1_graph(root)
        _need(final_predecessor == predecessor, "AOF-M final predecessor witness drift")
        artifact.validate_live(expected_names=_pair_names(published))
        terminal_payload = {"schema": plan.SCHEMA + "_terminal", "status": "TERMINAL",
            "attempt_sha256": attempt, "published": published, "oof_gate": {key: oof[key] for key in (
                "mean_matrix_minus_native", "positive_sessions", "worst_matrix_minus_native", "mean_matrix_minus_scalar", "passed")},
            "all7_refit_performed": bool(oof["passed"]), "closure_sha256": closure_sha,
            "reviewed_closure_sha256": reviewed_closure_sha256, "final_closure_sha256": closure_sha,
            "hidden_external_evalai_target_access": False,
            "source_target_access_authorized": True, "source_target_access": progress["source_target_access"],
            "source_target_access_attempted": progress["source_target_access_attempted"],
            "final_predecessor": final_predecessor,
            "terminal_xor_failure": True}
        terminal = artifact.publish_json("terminal.json", terminal_payload)
        terminal_published = True
        artifact.validate_live(expected_names=_pair_names({**published, "terminal.json": terminal}))
        return terminal, None
    except BaseException as error:
        if terminal_published:
            # Never publish failure after terminal.  A post-terminal topology
            # race is an unrecoverable external integrity failure.
            raise
        final_predecessor = None
        if progress["predecessor_opened"]:
            try:
                candidate = binding.validate_aof_v1_graph(root)
                if candidate == predecessor:
                    final_predecessor = candidate
            except BaseException:
                # Preserve the original exception; receipt honestly states
                # whether the safe recheck completed.
                final_predecessor = None
        artifact.validate_live(expected_names=_pair_names(published))
        failure_payload = {"schema": plan.SCHEMA + "_failure", "attempt_sha256": attempt,
            "published_prefix": published, "progress": progress, "exception_class": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error)[:300], "error_sha256": hashlib.sha256(repr(error).encode()).hexdigest(),
            "closure_sha256": closure_sha, "reviewed_closure_sha256": reviewed_closure_sha256,
            "hidden_external_evalai_target_access": False,
            "source_target_access_authorized": True, "source_target_access": progress["source_target_access"],
            "source_target_access_attempted": progress["source_target_access_attempted"],
            "final_predecessor": final_predecessor, "terminal_xor_failure": True}
        failure = artifact.publish_json("failure.json", failure_payload)
        artifact.validate_live(expected_names=_pair_names({**published, "failure.json": failure}))
        return None, failure
    finally:
        artifact.close()
