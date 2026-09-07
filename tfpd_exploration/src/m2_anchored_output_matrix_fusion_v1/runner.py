"""Thin one-forward AOF-M source materializer, composed from AOF V1."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import plan


class RunnerError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


class MatrixSourceMaterializer:
    """One AOF V1 preparation and exactly one native/post decode per session."""

    def __init__(self, root: Path, launch: dict[str, object], predecessor: dict[str, object]):
        self.root, self.launch, self.predecessor = Path(root), dict(launch), predecessor
        self._aof = None

    def prepare(self) -> dict[str, object]:
        from tfpd_exploration.src.m2_anchored_output_fusion_v1.runner import StaticAOFRunner
        self._aof = StaticAOFRunner(self.root, self.launch)
        authority = self._aof.prepare()
        _need(authority.get("pit_materializations") == 1 and authority.get("optimizer_constructed") is False
              and authority.get("parameter_updates") == 0 and authority.get("target_parameter_updates") == 0,
              "AOF-M inherited one-prepare/frozen authority drift")
        return authority

    def materialize_once(self) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, object]]:
        _need(self._aof is not None, "AOF-M prepare required before pairs")
        from tfpd_exploration.src.m2_anchored_output_fusion_v1.runner import _digest as aof_digest
        pairs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        evidence: dict[str, object] = {}
        prior = self.predecessor["paired_evidence"]
        for session in plan.SESSIONS:
            native, post, target, receipt = self._aof.pairs(session)
            witness = prior[session]
            exact_keys = ("native_prediction_sha256", "post_prediction_sha256", "target_sha256", "starts_sha256",
                          "native_identity_sha256", "post_identity_sha256", "ordered_first30_activity_sha256",
                          "ordered_support_indices_sha256", "normalized_t4_sha256", "raw_t4_sha256",
                          "model_state_before_sha256", "model_state_after_sha256")
            _need(all(receipt.get(key) == witness.get(key) for key in exact_keys)
                  and receipt.get("windows") == witness.get("windows")
                  and receipt.get("official_cpu_to_gpu_bridge") == witness.get("official_cpu_to_gpu_bridge")
                  and receipt.get("zero_prediction_exact") is True and receipt.get("parameter_updates") == 0
                  and receipt.get("target_parameter_updates") == 0,
                  f"{session}: AOF-M rematerialized predecessor authority drift")
            _need(np.isfinite(native).all() and np.isfinite(post).all() and np.isfinite(target).all(),
                  f"{session}: AOF-M arrays nonfinite")
            # The V1 receipt's strings are not themselves sufficient evidence:
            # recompute the exact inherited framing over the arrays actually
            # retained for the CPU OOF solves.
            for array, field in ((native, "native_prediction_sha256"),
                                 (post, "post_prediction_sha256"),
                                 (target, "target_sha256")):
                digest = aof_digest(array)
                _need(digest == receipt.get(field) == witness.get(field),
                      f"{session}: AOF-M returned {field} array/receipt drift")
            pairs[session] = (native, post, target)
            evidence[session] = receipt
        return pairs, evidence
