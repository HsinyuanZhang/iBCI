"""Flat-bias M1 live-B3S concat RIFT decoder for the isolated ablation."""
from __future__ import annotations

from btransform_unified_v1.m1_b3s_joint import load_trainable_b3s_from_sfix
from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.joint_m1_model import ARMS, JointM1ConcatDecoder


class FlatJointM1ConcatDecoder(JointM1ConcatDecoder):
    """The joint M1 decoder with the only temporal change: zero recency slopes."""

    def __init__(self, arm: str, *, seed: int = 42) -> None:
        if arm not in ARMS:
            raise ValueError("unknown M1 joint arm")
        # Deliberately bypass JointM1ConcatDecoder.__init__, whose RIFT call is
        # fixed to recency.  The inherited identity/session-memory/frontend
        # methods remain byte-for-byte shared with that frozen implementation.
        RiftConcatDecoder.__init__(self, "m1", context_bins=100, bias_mode="flat", seed=seed)
        self.arm = arm
        self.encoder = load_trainable_b3s_from_sfix()
        self._calib = {}
        self._carrier = {}


__all__ = ["FlatJointM1ConcatDecoder"]
