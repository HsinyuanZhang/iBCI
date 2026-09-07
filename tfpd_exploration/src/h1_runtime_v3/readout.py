"""Compact frozen M3 binding with unchanged native float64 map arithmetic.

Compared with the audited V1 decorator, this implementation does not retain
a duplicate complete checkpoint state after verifying the live engine. It
also reads canonical bank authority once, not on every observed bin, and
hashes each live bank tensor once per check instead of twice. Ordinary and
NumPy-alias bank mutations remain detectable. No model/label fitting lives
in this module. Artifact/parameter replacement requires constructing a new
binding; session reset cannot silently change the frozen model.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from tfpd_exploration.src.h1_m3_runtime_v1.runtime import (
    FrozenM3Readout,
    M3NativeReadoutStream,
    _array_sha256,
    _matrix,
    _sha256,
    _vector,
)


def load_frozen_readout(receipt_path, *, checkpoint_path, plain_ema_state_path):
    """Strictly load one of the explicitly supported model-specific map tags.

    A V6 map is not relabelled as a V4 map to reuse the old loader. All file,
    numeric-law, per-session map-array and frozen-model bindings are checked
    here, then the immutable runtime-only value type is reused.
    """
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    allowed = {"v4full24_canonical_m3_mat7_readout_v1": "v4full24_canonical_m3_mat7_maps_v1",
               "v6_canonical_m3_mat7_readout_v1": "v6_canonical_m3_mat7_maps_v1",
               "v7_canonical_m3_mat7_readout_v1": "v7_canonical_m3_mat7_maps_v1"}
    schema = receipt.get("schema")
    if schema not in allowed or receipt.get("status") != "FITTED_FIXED_CONTRACT":
        raise ValueError("unsupported or unfrozen M3 receipt tag")
    v6 = schema == "v6_canonical_m3_mat7_readout_v1"
    v7 = schema == "v7_canonical_m3_mat7_readout_v1"
    legacy_v4 = not (v6 or v7)
    if (receipt.get("family") != "MAT7" or receipt.get("ridge") != 0.0 or
            (not v6 and receipt.get("fit_dtype") != "float64") or receipt.get("scale_floor") != 1e-6):
        raise ValueError("M3 fixed numerical law drift")
    checkpoint_path, plain_ema_state_path = Path(checkpoint_path).resolve(), Path(plain_ema_state_path).resolve()
    if _sha256(checkpoint_path) != receipt.get("checkpoint_sha256"):
        raise ValueError("checkpoint hash does not match frozen M3 receipt")
    plain_digest_field = "frozen_plain_ema_state_sha256" if legacy_v4 else "plain_ema_sha256"
    if _sha256(plain_ema_state_path) != receipt.get(plain_digest_field):
        raise ValueError("plain EMA state hash does not match frozen M3 receipt")
    maps_path = Path(receipt["maps"]).resolve()
    if _sha256(maps_path) != receipt.get("maps_sha256"):
        raise ValueError("per-session maps file hash drift")
    payload = torch.load(maps_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != allowed[schema]:
        raise ValueError("per-session maps tag does not match the exact receipt tag")
    if (payload.get("family") != "MAT7" or payload.get("ridge") != 0.0 or
            (legacy_v4 and payload.get("scale_floor") != 1e-6)):
        raise ValueError("per-session maps numerical law drift")
    if v7:
        contracts = {"full_v4": "v4_causal_full_window_no_query_temporal_contract",
                     "t_v6": "v6_recency_query_temporal_contract4"}
        arm = receipt.get("arm")
        if (arm not in contracts or payload.get("arm") != arm or
                receipt.get("operator_contract") != contracts[arm]):
            raise ValueError("V7 arm/operator/map binding mismatch")
    raw_maps, sessions = payload.get("maps"), receipt.get("sessions")
    if not isinstance(raw_maps, dict) or not isinstance(sessions, dict) or set(raw_maps) != set(sessions):
        raise ValueError("per-session map roster drift")
    maps = {}
    for session, raw in raw_maps.items():
        if not isinstance(session, str) or not isinstance(raw, dict):
            raise ValueError("invalid per-session map")
        if raw.get("family") != "MAT7" or raw.get("ridge") != 0.0:
            raise ValueError("per-session map family/ridge drift")
        # V6's producer uses different hash-field names and commits the whole
        # maps artifact instead of duplicate per-array hashes. Validate its
        # actual stored dtype, rather than inventing absent receipt metadata
        # or silently casting an unexpected reduced-precision map.
        if not legacy_v4 and any(np.asarray(raw.get(name)).dtype != np.float64 for name in
                      ("p_mean", "p_scale", "y_mean", "y_scale", "intercept", "weight")):
            raise ValueError("V6/V7 maps must actually store float64 arrays")
        mapping = {name: _vector(raw.get(name), f"{session}.{name}", positive=name.endswith("scale"))
                   for name in ("p_mean", "p_scale", "y_mean", "y_scale", "intercept")}
        mapping["weight"] = _matrix(raw.get("weight"), f"{session}.weight")
        expected = sessions[session].get("map_array_sha256", {})
        if legacy_v4 and any(_array_sha256(value) != expected.get(name) for name, value in mapping.items()):
            raise ValueError("per-session map-array hash drift")
        for value in mapping.values():
            value.setflags(write=False)
        maps[session] = MappingProxyType(mapping)
    return FrozenM3Readout(MappingProxyType(maps), receipt_path, maps_path, plain_ema_state_path,
                           receipt["checkpoint_sha256"], receipt[plain_digest_field])


class CompactM3NativeReadoutStream(M3NativeReadoutStream):
    """Exact map application with only small immutable binding metadata."""

    def __init__(self, engine, readout, *, session_id):
        receipt = json.loads(readout.receipt_path.read_text())
        arrays = receipt.get("source_cache_authority", {}).get("arrays", {}).get("train", {})
        if not isinstance(arrays, dict):
            raise ValueError("receipt lacks canonical train-bank authority")
        bank_authorities = {}
        for session, row in arrays.items():
            if not isinstance(row, dict):
                raise ValueError("invalid canonical bank authority")
            bank_authorities[session] = MappingProxyType({key: row.get(key) for key in
                                                         ("bank_e0_sha256", "bank_hc_sha256")})
        self._bank_authorities = MappingProxyType(bank_authorities)
        self._verified_model_token = None
        super().__init__(engine, readout, session_id=session_id)
        self.discarded_verified_state_tensor_bytes = sum(
            value.numel() * value.element_size() for value in self._expected_state.values())
        self._verified_model_token = self._model_token
        # In V1 this state_dict remained resident for every session decorator.
        # The exact live-state comparison has finished; no tensor copy is
        # needed to detect later ordinary model mutation/replacement.
        self._expected_state = None

    def _bank_digests(self, session_id):
        row = self._bank_authorities.get(session_id)
        if row is None or any(not isinstance(value, str) for value in row.values()):
            raise ValueError("receipt lacks canonical train-bank digest for selected session")
        return row

    def _bank_token_for(self, session_id):
        bank = getattr(self.engine, "bank", None)
        roster = tuple(getattr(self.engine, "unit_ids", ()))
        if bank is None or not all(hasattr(bank, key) for key in ("E0", "T", "unit_mask")):
            raise ValueError("wrapped engine must expose its H1 bank and explicit unit roster")
        expected = self._bank_digests(session_id)
        e0_digest, t_digest = _array_sha256(bank.E0), _array_sha256(bank.T)
        if e0_digest != expected["bank_e0_sha256"] or t_digest != expected["bank_hc_sha256"]:
            raise ValueError("engine E0/T bank does not match the canonical M3 session binding")
        n = int(bank.E0.shape[-2])
        if roster != tuple(range(n)) or bank.T.shape[-2] != n or bank.unit_mask.shape[-1] != n:
            raise ValueError("engine roster/bank shape is not the canonical ordered M3 roster")
        return (roster, e0_digest, t_digest, _array_sha256(bank.unit_mask),
                id(bank.E0), id(bank.T), id(bank.unit_mask),
                bank.E0._version, bank.T._version, bank.unit_mask._version)

    def _assert_live_binding(self):
        if self._verified_model_token is None:
            return super()._assert_live_binding()
        self._assert_frozen_model()

    def _assert_frozen_model(self):
        model = self.engine.model
        if model.training or self._tensor_token(model) != self._verified_model_token:
            raise RuntimeError("wrapped model changed after M3 binding; construct a new verified stream")
        if getattr(self.engine, "task", None) != "h1" or getattr(self.engine, "divisor", None) != 20.0:
            raise RuntimeError("wrapped engine native H1 unit contract changed")

    def _assert_fresh(self):
        self._assert_frozen_model()
        if self._bank_token_for(self.session_id) != self._bank_token:
            raise RuntimeError("wrapped bank/roster changed after M3 binding; reset with a verified session binding")

    def reset(self, *, session_id, **kwargs):
        # Reject a changed model BEFORE the underlying session is advanced or
        # reset. The full model artifact is no longer resident in this object.
        self._assert_frozen_model()
        return super().reset(session_id=session_id, **kwargs)

    @property
    def readout_array_bytes(self):
        return sum(value.nbytes for mapping in self.readout.maps.values() for value in mapping.values())

    @property
    def state_bytes(self):
        # This includes all retained map arrays, not only the selected session.
        # They may be shared by multiple decorators; reports must deduplicate
        # object/storage identities before claiming a process total.
        return self.engine.state_bytes + self.readout_array_bytes

    @property
    def memory_breakdown(self):
        return {"engine_reported_state_bytes": self.engine.state_bytes,
                "retained_map_arrays_bytes_shared": self.readout_array_bytes,
                "retained_duplicate_checkpoint_tensor_bytes": 0,
                "duplicate_checkpoint_tensor_bytes_removed_vs_v1": self.discarded_verified_state_tensor_bytes,
                "note": "Python binding metadata, interpreter, transient verification state and engine model/bank are separate."}
