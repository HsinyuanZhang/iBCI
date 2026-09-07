"""Runtime-only application of the frozen V4-FULL24 canonical M3 MAT7 maps.

This module neither imports a cache/NWB reader nor contains any fitting code.
The wrapped engine must already produce *native H1 velocity*; this decorator
does not apply a second /20 conversion.  Maps are evaluated in float64 and a
fresh float32 result is returned from both public prediction accessors.

One decorator owns one engine/session trajectory.  It snapshots model and bank
identity/version at binding and rejects ordinary later mutations; device/dtype
migration is intentionally rejected rather than silently reusing a map that is
specific to the frozen CPU plain-EMA state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch


DIM = 7
EXPECTED_RECEIPT_SCHEMA = "v4full24_canonical_m3_mat7_readout_v1"
EXPECTED_MAP_SCHEMA = "v4full24_canonical_m3_mat7_maps_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("M3 runtime only accepts CPU-bound frozen model/bank tensors")
        return value.detach().numpy()
    return np.asarray(value)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(_as_numpy(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _vector(value: Any, name: str, *, positive: bool = False) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value), dtype=np.float64)
    if array.shape != (DIM,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float64 [{DIM}]")
    if positive and not bool((array > 0.0).all()):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value), dtype=np.float64)
    if array.shape != (DIM, DIM) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float64 [{DIM},{DIM}]")
    return array


@dataclass(frozen=True)
class FrozenM3Readout:
    """Verified per-session MAT7 maps tied to one checkpoint and EMA state."""

    maps: Mapping[str, Mapping[str, np.ndarray]]
    receipt_path: Path
    maps_path: Path
    plain_ema_state_path: Path
    checkpoint_sha256: str
    plain_ema_state_sha256: str

    @classmethod
    def load(
        cls,
        receipt_path: str | Path,
        *,
        checkpoint_path: str | Path,
        plain_ema_state_path: str | Path,
        maps_path: str | Path | None = None,
    ) -> "FrozenM3Readout":
        """Load only the sealed artifacts and reject any binding drift.

        ``checkpoint_path`` and ``plain_ema_state_path`` are supplied by the
        caller's model loader.  Their hashes bind a generic supplied stream to
        this model-specific map without causing this module to load model data.
        """
        receipt_file = Path(receipt_path).resolve()
        if not receipt_file.is_file():
            raise FileNotFoundError(receipt_file)
        receipt = json.loads(receipt_file.read_text())
        if receipt.get("schema") != EXPECTED_RECEIPT_SCHEMA or receipt.get("status") != "FITTED_FIXED_CONTRACT":
            raise ValueError("unrecognized or unfrozen M3 receipt")
        if receipt.get("family") != "MAT7" or float(receipt.get("ridge", float("nan"))) != 0.0:
            raise ValueError("M3 map family/ridge binding drift")
        if receipt.get("fit_dtype") != "float64" or float(receipt.get("scale_floor", float("nan"))) != 1e-6:
            raise ValueError("M3 numeric contract drift")
        checkpoint_file = Path(checkpoint_path).resolve()
        plain_file = Path(plain_ema_state_path).resolve()
        if not checkpoint_file.is_file() or not plain_file.is_file():
            raise FileNotFoundError("bound checkpoint or plain EMA state is missing")
        if _sha256(checkpoint_file) != receipt.get("checkpoint_sha256"):
            raise ValueError("checkpoint hash does not match frozen M3 receipt")
        if _sha256(plain_file) != receipt.get("frozen_plain_ema_state_sha256"):
            raise ValueError("plain EMA state hash does not match frozen M3 receipt")
        selected_maps = Path(maps_path).resolve() if maps_path is not None else Path(receipt["maps"]).resolve()
        if not selected_maps.is_file():
            raise FileNotFoundError(selected_maps)
        if _sha256(selected_maps) != receipt.get("maps_sha256"):
            raise ValueError("per-session M3 maps hash does not match receipt")
        # The checked hash makes this trusted, local artifact safe to decode.
        payload = torch.load(selected_maps, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("schema") != EXPECTED_MAP_SCHEMA:
            raise ValueError("per-session map artifact schema drift")
        if payload.get("family") != "MAT7" or float(payload.get("ridge", float("nan"))) != 0.0:
            raise ValueError("per-session map artifact family/ridge drift")
        if float(payload.get("scale_floor", float("nan"))) != float(receipt["scale_floor"]):
            raise ValueError("per-session map scale-floor drift")
        raw_maps = payload.get("maps")
        receipt_sessions = receipt.get("sessions")
        if not isinstance(raw_maps, dict) or not isinstance(receipt_sessions, dict) or set(raw_maps) != set(receipt_sessions):
            raise ValueError("per-session M3 roster drift")
        verified: dict[str, Mapping[str, np.ndarray]] = {}
        for session, raw in raw_maps.items():
            if not isinstance(session, str) or not isinstance(raw, dict):
                raise ValueError("invalid map session payload")
            if raw.get("family") != "MAT7" or float(raw.get("ridge", float("nan"))) != 0.0:
                raise ValueError(f"{session}: map family/ridge drift")
            mapping = {
                "p_mean": _vector(raw.get("p_mean"), f"{session}.p_mean"),
                "p_scale": _vector(raw.get("p_scale"), f"{session}.p_scale", positive=True),
                "y_mean": _vector(raw.get("y_mean"), f"{session}.y_mean"),
                "y_scale": _vector(raw.get("y_scale"), f"{session}.y_scale", positive=True),
                "weight": _matrix(raw.get("weight"), f"{session}.weight"),
                "intercept": _vector(raw.get("intercept"), f"{session}.intercept"),
            }
            expected = receipt_sessions[session].get("map_array_sha256")
            if not isinstance(expected, dict) or any(_array_sha256(mapping[name]) != expected.get(name) for name in mapping):
                raise ValueError(f"{session}: map-array hash drift")
            for array in mapping.values():
                array.setflags(write=False)
            verified[session] = MappingProxyType(mapping)
        return cls(MappingProxyType(verified), receipt_file, selected_maps, plain_file,
                   receipt["checkpoint_sha256"], receipt["frozen_plain_ema_state_sha256"])

    def for_session(self, session_id: str) -> Mapping[str, np.ndarray]:
        if session_id not in self.maps:
            raise KeyError(f"no frozen M3 map for session {session_id!r}")
        return self.maps[session_id]

    def apply(self, session_id: str, native_prediction: Any) -> np.ndarray:
        """Apply the saved native-space map and return a detached float32 copy."""
        mapping = self.for_session(session_id)
        if isinstance(native_prediction, torch.Tensor):
            native_prediction = native_prediction.detach().cpu().numpy()
        prediction = np.ascontiguousarray(np.asarray(native_prediction), dtype=np.float64)
        was_vector = prediction.ndim == 1
        if was_vector:
            prediction = prediction[None, :]
        if prediction.ndim != 2 or prediction.shape[1] != DIM or not np.isfinite(prediction).all():
            raise ValueError("wrapped engine must return finite native H1 [B,7] prediction")
        output = ((prediction - mapping["p_mean"]) / mapping["p_scale"] @ mapping["weight"] + mapping["intercept"])
        output = output * mapping["y_scale"] + mapping["y_mean"]
        if not np.isfinite(output).all():
            raise ValueError("frozen M3 readout produced nonfinite output")
        copied = np.array(output, dtype=np.float32, copy=True, order="C")
        return copied[0] if was_vector else copied


class M3NativeReadoutStream:
    """Decorate an exact stream while retaining its temporal/session semantics."""

    def __init__(self, engine: Any, readout: FrozenM3Readout, *, session_id: str) -> None:
        self.engine = engine
        self.readout = readout
        self.session_id = session_id
        self._map = readout.for_session(session_id)  # explicit selection before use
        engine_session = getattr(engine, "session_id", session_id)
        if engine_session != session_id:
            raise ValueError("wrapped engine session and explicit M3 map session disagree")
        self._expected_state = self._load_plain_ema_state()
        self._assert_live_binding()
        self._model_token = self._tensor_token(self.engine.model)
        self._bank_token = self._bank_token_for(session_id)

    def _load_plain_ema_state(self) -> Mapping[str, torch.Tensor]:
        # Hash verification happened in FrozenM3Readout.load before this trusted
        # local tensor-only state is decoded.  No labels, NWB, or cache are read.
        state = torch.load(self.readout.plain_ema_state_path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError("frozen plain EMA artifact is not a tensor state_dict")
        return state

    @staticmethod
    def _tensor_token(module: torch.nn.Module) -> tuple:
        values = []
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            values.append((name, id(tensor), tensor.data_ptr(), getattr(tensor, "_version", None),
                           tuple(tensor.shape), tensor.dtype, tensor.device))
        return tuple(values)

    def _bank_token_for(self, session_id: str) -> tuple:
        bank = getattr(self.engine, "bank", None)
        roster = tuple(getattr(self.engine, "unit_ids", ()))
        if bank is None or not hasattr(bank, "E0") or not hasattr(bank, "T") or not hasattr(bank, "unit_mask"):
            raise ValueError("wrapped engine must expose its H1 bank and explicit unit roster")
        expected = self._bank_digests(session_id)
        if _array_sha256(bank.E0) != expected["bank_e0_sha256"] or _array_sha256(bank.T) != expected["bank_hc_sha256"]:
            raise ValueError("engine E0/T bank does not match the canonical M3 session binding")
        n_units = int(bank.E0.shape[-2])
        if tuple(roster) != tuple(range(n_units)) or bank.T.shape[-2] != n_units or bank.unit_mask.shape[-1] != n_units:
            raise ValueError("engine roster/bank shape is not the canonical ordered M3 roster")
        return (roster, _array_sha256(bank.E0), _array_sha256(bank.T), _array_sha256(bank.unit_mask),
                id(bank.E0), id(bank.T), id(bank.unit_mask),
                getattr(bank.E0, "_version", None), getattr(bank.T, "_version", None), getattr(bank.unit_mask, "_version", None))

    def _bank_digests(self, session_id: str) -> Mapping[str, str]:
        source = json.loads(self.readout.receipt_path.read_text()).get("source_cache_authority", {})
        arrays = source.get("arrays", {}) if isinstance(source, dict) else {}
        # Fitting used held-in calibration / train banks.  The receipt binds E0
        # and T there; its source authority does not publish a bank-unit-mask
        # digest, so the mask is shape/roster checked then drift-snapshotted.
        row = arrays.get("train", {}).get(session_id) if isinstance(arrays.get("train", {}), dict) else None
        if not isinstance(row, dict) or not isinstance(row.get("bank_e0_sha256"), str) or not isinstance(row.get("bank_hc_sha256"), str):
            raise ValueError("receipt lacks canonical train-bank digest for selected session")
        return row

    def _assert_live_binding(self) -> None:
        model = getattr(self.engine, "model", None)
        if not isinstance(model, torch.nn.Module) or model.training:
            raise ValueError("wrapped engine must expose the frozen eval() H1 model")
        if getattr(self.engine, "task", None) != "h1" or float(getattr(self.engine, "divisor", float("nan"))) != 20.0:
            raise ValueError("wrapped engine must expose the native H1 divisor-20 contract")
        version = getattr(model, "frontend_contract_version", None)
        version = version.item() if isinstance(version, torch.Tensor) and version.numel() == 1 else version
        if int(version) != 4:
            raise ValueError("wrapped engine frontend contract is not V4")
        actual = model.state_dict()
        if actual.keys() != self._expected_state.keys():
            raise ValueError("live model state keys do not match frozen plain EMA state")
        for name, expected in self._expected_state.items():
            value = actual[name]
            if value.device.type != "cpu" or value.dtype != expected.dtype or not torch.equal(value.detach().cpu(), expected):
                raise ValueError(f"live model differs from frozen plain EMA state at {name}")

    def _assert_fresh(self) -> None:
        if self._tensor_token(self.engine.model) != self._model_token:
            raise RuntimeError("wrapped model changed after M3 binding; construct a new verified stream")
        if self._bank_token_for(self.session_id) != self._bank_token:
            raise RuntimeError("wrapped bank/roster changed after M3 binding; reset with a verified session binding")

    def _require_same_session(self, session_id: str | None) -> None:
        if session_id is not None and session_id != self.session_id:
            raise ValueError("session change requires explicit reset with a new M3 map")

    def observe(self, observation: Any, *, session_id: str | None = None) -> Any:
        self._require_same_session(session_id)
        self._assert_fresh()
        return self.engine.observe(observation, session_id=session_id)

    def current_prediction(self) -> np.ndarray:
        self._assert_fresh()
        return self.readout.apply(self.session_id, self.engine.current_prediction())

    def predict(self, observation: Any, *, session_id: str | None = None) -> np.ndarray:
        self._require_same_session(session_id)
        self._assert_fresh()
        return self.readout.apply(self.session_id, self.engine.predict(observation, session_id=session_id))

    def reset(self, *, session_id: str, **kwargs: Any) -> Any:
        # Verify/select before mutating the underlying stream; failed map lookup
        # therefore cannot partly reset an active trajectory.
        selected = self.readout.for_session(session_id)
        result = self.engine.reset(session_id=session_id, **kwargs)
        self.session_id, self._map = session_id, selected
        self._assert_live_binding()
        self._model_token = self._tensor_token(self.engine.model)
        self._bank_token = self._bank_token_for(session_id)
        return result

    def set_bank(self, bank: Any) -> Any:
        raise RuntimeError("set_bank requires an explicit reset(session_id=...) to re-verify M3 bank binding")

    def on_done(self, **kwargs: Any) -> Any:
        # Delegates exactly: the normal underlying default keeps finite history.
        self._assert_fresh()
        return self.engine.on_done(**kwargs)

    @property
    def state_bytes(self) -> int:
        return self.engine.state_bytes
