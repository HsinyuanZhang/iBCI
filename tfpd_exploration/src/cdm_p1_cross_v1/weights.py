"""The weight-swap law of Part A: strict load + state-digest proof.

The pattern is the sealed ``src/cal_aug_v1/deployment._swap_runtime_model`` law,
applied to the frozen activity-only Stage-P runtime instead of the Z1 oracle
runtime.  Two runtime seams hold a model reference and BOTH are swapped:

* ``state.model`` -- the governing/full-system forwards
  (``_forward_full`` / ``_forward_horizon_once``);
* ``state.executor_state.model`` -- the four held-group pseudo forwards
  (``ConcreteCellDFourGroupExecutor._torch_variable_prefix_forward``), which is
  what the P1 direction measurement reads.

Swapping only one of the two would leave F11's carrier measurement silently
running on the other arm's weights, so the swap asserts they alias the same
object before and after, and the restore re-verifies the sealed digest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from . import plan


class WeightSwapError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WeightSwapError(message)


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _purity_proof(state: Any, model: Any) -> dict[str, Any]:
    """One zeros double-forward: finite, repeat-deterministic, state unchanged."""
    torch = state.modules["torch"]
    pop = state.modules["pop_robust"]
    arm = state.modules["arm_common"]
    device = state.device
    neural = torch.zeros((1, 50, 4), dtype=torch.float32, device=device)
    calib = torch.zeros((1, 1, 100, 4), dtype=torch.float32, device=device)
    side = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
    before = arm.state_sha256(model)
    with pop.dynamic_dropout_recorder() as recorder:
        with torch.no_grad():
            first, _ = model(neural, calib_trials=calib, side_features=side)
            second, _ = model(neural, calib_trials=calib, side_features=side)
    _require(
        bool(torch.isfinite(first).all().item()) and torch.equal(first, second),
        "the swapped model failed the finite/repeat-determinism purity proof",
    )
    _require(
        recorder.get("uniform_calls") == 0 and recorder.get("dropout_calls") == [],
        "the swapped model activated dropout during the purity proof",
    )
    _require(arm.state_sha256(model) == before, "the purity proof mutated the swapped model")
    return {
        "finite_forward": True,
        "repeated_fixed_forward_bitwise_equal": True,
        "dynamic_dropout_calls": 0,
        "model_state_unchanged": True,
    }


def swap_runtime_weights(
    runtime: Any,
    *,
    swa_path: Path,
    expected_artifact_sha256: str,
    expected_state_sha256: str,
) -> dict[str, Any]:
    """Strict-load the arm SWA into a fresh model and swap it into the runtime.

    Returns the binding receipt plus the retained sealed model object under
    ``"sealed_model"`` (caller-owned; pass it to :func:`restore_runtime_weights`).
    """
    state = runtime._require_state()
    torch = state.modules["torch"]
    pop = state.modules["pop_robust"]
    arm = state.modules["arm_common"]
    swa_path = Path(swa_path)
    artifact = artifact_sha256(swa_path)
    _require(artifact == expected_artifact_sha256, "the C1 SWA artifact hash drifted")
    sealed_model = state.model
    _require(
        state.executor_state.model is sealed_model,
        "the runtime model seams diverged before the swap (governing != executor)",
    )
    sealed_digest = arm.state_sha256(sealed_model)
    payload = torch.load(swa_path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping) and "state_dict" in payload,
             "the C1 SWA payload root drift (no state_dict)")
    model = pop.build_population_robustness_model(seed=42, cell="D")
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(state.device).eval()
    loaded_digest = arm.state_sha256(model)
    _require(
        loaded_digest == expected_state_sha256,
        "the strict-loaded C1 model state digest disagrees with the sealed deployment binding",
    )
    purity = _purity_proof(state, model)
    state.model = model
    state.executor_state.model = model
    _require(
        state.model is model and state.executor_state.model is model,
        "the swap failed to bind both runtime model seams",
    )
    return {
        "artifact_path": str(swa_path),
        "artifact_sha256": artifact,
        "expected_artifact_sha256": expected_artifact_sha256,
        "strict_load": True,
        "fresh_build": "pop_robust.build_population_robustness_model(seed=42, cell='D')",
        "eval_mode": not model.training,
        "arm_state_sha256": loaded_digest,
        "expected_state_sha256": expected_state_sha256,
        "sealed_state_sha256_before_swap": sealed_digest,
        "purity_proof": purity,
        "swa_manifest_present": "swa_manifest" in payload,
        "sealed_model": sealed_model,
    }


def restore_runtime_weights(
    runtime: Any,
    *,
    sealed_model: Any,
    expected_sealed_state_sha256: str,
) -> dict[str, Any]:
    """Restore the retained sealed model and re-verify its state digest."""
    state = runtime._require_state()
    arm = state.modules["arm_common"]
    digest = arm.state_sha256(sealed_model)
    _require(
        digest == expected_sealed_state_sha256,
        "the retained sealed model digest drifted while the C1 arm ran",
    )
    state.model = sealed_model
    state.executor_state.model = sealed_model
    _require(
        state.model is sealed_model and state.executor_state.model is sealed_model,
        "the restore failed to bind both runtime model seams",
    )
    return {
        "restored_state_sha256": digest,
        "expected_sealed_state_sha256": expected_sealed_state_sha256,
        "sealed_model_restored": True,
    }


def verify_model_digest(runtime: Any, *, expected: str, label: str) -> dict[str, Any]:
    """State-digest proof of the CURRENT model at an arbitrary checkpoint."""
    state = runtime._require_state()
    arm = state.modules["arm_common"]
    governing = arm.state_sha256(state.model)
    executor = arm.state_sha256(state.executor_state.model)
    _require(
        governing == executor,
        f"the runtime model seams disagree at checkpoint {label}",
    )
    _require(governing == expected, f"model state digest drift at checkpoint {label}")
    return {"label": label, "model_state_sha256": governing, "matches_expected": True}
