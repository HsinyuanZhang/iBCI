"""Four information-arm wrapper for the carrier-v4 M1 fusion decoder.

The underlying `JointM1FusionDecoder` remains the authoritative live-B3S
implementation.  This wrapper only controls the two information channels:
E0 and direct T.  It deliberately leaves the neural query decoder and chosen
fusion frontend unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[2]
for _entry in (V2 / "src", V2.parent / "btransform_unified_v1" / "src", HERE.parent):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from btransform_unified_v1.bank import TaskBank
from fusion_model import JointM1FusionDecoder
from btransform_unified_v2.joint_m1_model import ARM_B, ARM_D, JointM1ConcatDecoder


INFORMATION_ARMS = ("full", "activity_only", "carrier_only", "none")
_LEGACY_ARM = {"full": ARM_D, "activity_only": ARM_B, "carrier_only": ARM_B, "none": ARM_B}


class M1InformationArmDecoder(JointM1FusionDecoder):
    """M1 fusion decoder with an explicit E0/T information-arm contract.

    * `full`: live B3S receives `(raw, T)` and T is direct input.
    * `activity_only`: live B3S receives `(raw, 0)`; direct T is zero.
    * `carrier_only`: E0 is a literal zero [64, 100], no encoder call; T is
      direct input.
    * `none`: E0 and direct T are literal zero, with no encoder call.

    The first two arms call the inherited implementation verbatim, making
    concat full/activity initialization and behavior comparable to legacy D/B.
    """

    def __init__(self, information_arm: str, *, fusion: str = "concat",
                 proj_dim: int = 16, seed: int = 42) -> None:
        if information_arm not in INFORMATION_ARMS:
            raise ValueError(f"unknown M1 information arm {information_arm!r}")
        self.information_arm = information_arm
        super().__init__(_LEGACY_ARM[information_arm], fusion=fusion, proj_dim=proj_dim, seed=seed)
        self._none_sessions: set[str] = set()
        if information_arm in ("carrier_only", "none"):
            # Retain the superclass module topology for stable checkpoint keys,
            # but ensure inactive B3S parameters cannot enter optimizer/EMA
            # updates.  _identity below never invokes this module.
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    @property
    def encoder_active(self) -> bool:
        return self.information_arm in ("full", "activity_only")

    @property
    def legacy_arm(self) -> str:
        return _LEGACY_ARM[self.information_arm]

    def install_session_memory(self, banks: Mapping[str, TaskBank],
                               calib_by_session: Mapping[str, np.ndarray | Tensor]) -> None:
        if self.encoder_active:
            super().install_session_memory(banks, calib_by_session)
            return
        # CARRIER_ONLY needs only T. NONE does not need either source, but
        # storing its validated T makes accidental dependence auditable.
        for name, bank in banks.items():
            session = bank.session_id
            if self.information_arm == "none":
                # NONE records only the session key.  It neither reads nor
                # retains T, so a future accidental direct route cannot use it.
                self._none_sessions.add(session)
                continue
            carrier = np.ascontiguousarray(np.asarray(bank.carrier, dtype=np.float32))
            if carrier.shape != (64, 4) or not np.isfinite(carrier).all():
                raise ValueError(f"{session}: carrier geometry/nonfinite drift")
            key = session.replace("-", "_")
            attribute = f"carrier_{key}"
            if hasattr(self, attribute):
                old = getattr(self, attribute).detach().cpu().numpy()
                if not np.array_equal(old, carrier):
                    raise ValueError(f"cross-surface carrier drift {session}")
            else:
                self.register_buffer(attribute, torch.from_numpy(carrier), persistent=False)
            self._carrier[session] = getattr(self, attribute)

    def _identity(self, sessions: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        if self.encoder_active:
            # Exact inherited B/D paths: full maps to D, activity-only to B.
            return super()._identity(sessions, device)
        e0: dict[str, Tensor] = {}
        direct: dict[str, Tensor] = {}
        for session in dict.fromkeys(sessions):
            if self.information_arm == "none":
                if session not in self._none_sessions:
                    raise RuntimeError(f"missing session memory for {session}")
                e0[session] = torch.zeros((64, 100), device=device, dtype=torch.float32)
                direct[session] = torch.zeros((64, 4), device=device, dtype=torch.float32)
                continue
            carrier = self._carrier.get(session)
            if carrier is None:
                raise RuntimeError(f"missing carrier memory for {session}")
            value = carrier.to(device=device, dtype=torch.float32)
            e0[session] = torch.zeros((64, 100), device=device, dtype=torch.float32)
            direct[session] = value if self.information_arm == "carrier_only" else torch.zeros_like(value)
        return torch.stack([e0[session] for session in sessions]), torch.stack([direct[session] for session in sessions])


def arm_contract(information_arm: str) -> dict[str, Any]:
    if information_arm not in INFORMATION_ARMS:
        raise ValueError(f"unknown M1 information arm {information_arm!r}")
    return {
        "information_arm": information_arm,
        "legacy_arm": _LEGACY_ARM[information_arm],
        "e0": "live_b3s(raw,T)" if information_arm == "full"
              else "live_b3s(raw,zero_T)" if information_arm == "activity_only"
              else "literal_zero_64x100",
        "direct_t": "T" if information_arm in ("full", "carrier_only") else "literal_zero_64x4",
        "encoder_called": information_arm in ("full", "activity_only"),
        "encoder_trainable": information_arm in ("full", "activity_only"),
    }


def arm_invariant_smoke(device: str | torch.device = "cpu") -> dict[str, Any]:
    """NWB-free structural test for all E0/T routes and legacy B/D identity."""
    resolved_device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(0x4D314152)
    carrier = np.arange(256, dtype=np.float32).reshape(64, 4) / 31.0
    e0 = np.zeros((64, 100), dtype=np.float32)
    bank = TaskBank(
        session_id="synthetic-m1", E0=e0, carrier=carrier,
        unit_mask=np.ones(64, dtype=np.bool_),
        X_store=np.zeros((1, 100, 64), dtype=np.float32),
        target_store=np.zeros((1, 16), dtype=np.float32), window_ids=np.zeros(1, dtype=np.int64),
        calibration_meta={"shape": [64, 100], "trial_count": 10, "estimator": "synthetic",
                          "array_sha256": hashlib.sha256(e0.tobytes()).hexdigest(), "budget": 10},
    )
    calib = {bank.session_id: torch.randn((10, 1024, 64), generator=generator).numpy().astype(np.float32)}
    query = torch.randn((2, 100, 64), generator=generator, dtype=torch.float32, device="cpu").to(resolved_device)
    valid = torch.ones((2, 100), dtype=torch.bool, device=resolved_device)
    results: dict[str, Any] = {}
    for arm in INFORMATION_ARMS:
        model = M1InformationArmDecoder(arm, seed=42).to(resolved_device).eval()
        model.install_session_memory({bank.session_id: bank}, calib if model.encoder_active else {})
        if not model.encoder_active:
            def _forbidden(*_args: Any, **_kwargs: Any) -> Tensor:
                raise AssertionError("inactive encoder was called")
            model.encoder.forward = _forbidden  # type: ignore[method-assign]
        identity_e0, identity_t = model._identity([bank.session_id], resolved_device)
        expected_e0_zero = arm in ("carrier_only", "none")
        expected_t_zero = arm in ("activity_only", "none")
        if expected_e0_zero and int(torch.count_nonzero(identity_e0)) != 0:
            raise AssertionError(f"{arm}: E0 zero invariant failed")
        if expected_t_zero and int(torch.count_nonzero(identity_t)) != 0:
            raise AssertionError(f"{arm}: direct T zero invariant failed")
        if arm == "carrier_only" and not torch.equal(identity_t[0], torch.from_numpy(carrier).to(resolved_device)):
            raise AssertionError("carrier_only: direct T drift")
        results[arm] = {"e0_nonzero": int(torch.count_nonzero(identity_e0)),
                        "direct_t_nonzero": int(torch.count_nonzero(identity_t)),
                        "encoder_active": model.encoder_active}
    for arm, legacy_arm in (("full", ARM_D), ("activity_only", ARM_B)):
        wrapped = M1InformationArmDecoder(arm, seed=42).to(resolved_device).eval()
        legacy = JointM1ConcatDecoder(legacy_arm, seed=42).to(resolved_device).eval()
        wrapped.install_session_memory({bank.session_id: bank}, calib)
        legacy.install_session_memory({bank.session_id: bank}, calib)
        wrapped_parameters = dict(wrapped.named_parameters())
        legacy_parameters = dict(legacy.named_parameters())
        if set(wrapped_parameters) != set(legacy_parameters) or any(
            not torch.equal(wrapped_parameters[name], legacy_parameters[name]) for name in wrapped_parameters
        ):
            raise AssertionError(f"{arm}: inherited legacy parameter drift")
        with torch.no_grad():
            wrapped_output = wrapped(query, bank, input_valid_mask=valid)
            legacy_output = legacy(query, bank, input_valid_mask=valid)
        if not torch.equal(wrapped_output, legacy_output):
            raise AssertionError(f"{arm}: inherited legacy full-forward drift")
        results[arm]["legacy_parameter_byte_equal"] = True
        results[arm]["legacy_forward_byte_equal"] = True

    # The B activity path must pass literal zero side columns into B3S.  Patch
    # the exact module-global imported by the inherited _identity method, then
    # delegate to the real encoder helper so its normal output is preserved.
    import btransform_unified_v2.joint_m1_model as old_joint
    activity = M1InformationArmDecoder("activity_only", seed=42).to(resolved_device).eval()
    activity.install_session_memory({bank.session_id: bank}, calib)
    observed_sides: list[Tensor] = []
    original_encode = old_joint.encode_b3s
    def _observing_encode(encoder: Any, raw: Tensor, side: Tensor) -> Tensor:
        observed_sides.append(side.detach().clone())
        return original_encode(encoder, raw, side)
    old_joint.encode_b3s = _observing_encode
    try:
        activity._identity([bank.session_id], resolved_device)
    finally:
        old_joint.encode_b3s = original_encode
    if len(observed_sides) != 1 or int(torch.count_nonzero(observed_sides[0])) != 0:
        raise AssertionError("activity_only: B3S side was not literal zero")
    return {"schema": "m1_carrier_v4_information_arm_smoke_v1", "status": "PASSED", "device": str(resolved_device), "arms": results,
            "activity_encoder_side_literal_zero": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="NWB-free M1 information-arm invariant smoke")
    parser.add_argument("--arm-invariant-smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.arm_invariant_smoke:
        parser.error("--arm-invariant-smoke is required")
    import json
    print(json.dumps(arm_invariant_smoke(args.device), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["INFORMATION_ARMS", "M1InformationArmDecoder", "arm_contract"]
