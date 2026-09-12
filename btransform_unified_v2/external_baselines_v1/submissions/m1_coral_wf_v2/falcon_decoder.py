"""NumPy-only streaming runtime for fair_v2 linear WF packs.

Payload is numeric only: no labels, raw NWB, pickles, or workspace imports.
Each bin: 12-tap causal WF → session z-score → affine aligner → 10-bin
newest-first history → unpenalized-intercept ridge.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from falcon_challenge.interface import BCIDecoder

PAYLOAD_SCHEMA = "fair_v2_linear_falcon_payload_v1"
HISTORY = 10
N_TAPS = 12


class FairV2LinearFalconDecoder(BCIDecoder):
    def __init__(
        self,
        task_config,
        model_path: str = "/data/payload.npz",
        *,
        manifest_path: str = "/data/manifest.json",
        batch_size: int = 1,
    ):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self._validate_manifest()
        self.payload = np.load(self.model_path, allow_pickle=False)
        self._load_common()
        self._sessions = self.manifest["sessions"]
        self._session_arrays = self._cache_session_arrays()
        self._active = 0
        self._prefixes: list[str] = []
        self._allocate_state()

    def _task_name(self) -> str:
        value = getattr(self.task_config, "task", None)
        return getattr(value, "name", str(value)).split(".")[-1].lower()

    def _output_dim(self) -> int:
        if hasattr(self.task_config, "out_dim"):
            return int(self.task_config.out_dim)
        return int(self.task_config.n_outputs)

    def _validate_manifest(self) -> None:
        m = self.manifest
        if m.get("schema") != PAYLOAD_SCHEMA:
            raise ValueError("unsupported payload schema")
        expected = m.get("expected_prediction_fields")
        if not isinstance(expected, dict):
            raise ValueError("manifest has no expected_prediction_fields")
        if m.get("task") != self._task_name():
            raise ValueError("payload task does not match FalconConfig")
        for field, observed in (("channels", self.task_config.n_channels), ("outputs", self._output_dim())):
            if int(expected.get(field, -1)) != int(observed):
                raise ValueError("manifest %s does not match FalconConfig" % field)
        self.history = int(expected.get("history", -1))
        self.outputs = int(expected["outputs"])
        if self.history != HISTORY:
            raise ValueError("this runtime requires exactly 10 causal bins")
        if int(m.get("max_batch", self.batch_size)) < 1:
            raise ValueError("invalid manifest max_batch")
        if self.batch_size > int(m.get("max_batch", self.batch_size)):
            raise ValueError("configured batch exceeds payload max_batch")
        if not isinstance(m.get("sessions"), dict) or not m["sessions"]:
            raise ValueError("manifest sessions must be a nonempty tag mapping")

    def _array(self, key: str, *, dtype=np.float64) -> np.ndarray:
        if key not in self.payload.files:
            raise ValueError("payload lacks %s" % key)
        value = np.asarray(self.payload[key], dtype=dtype)
        if not np.isfinite(value).all():
            raise ValueError("payload %s is nonfinite" % key)
        return np.ascontiguousarray(value)

    def _load_common(self) -> None:
        self.wf_kernel = self._array("wf_kernel")
        self.ridge_coef = self._array("ridge_coef")
        self.ridge_intercept = self._array("ridge_intercept")
        if self.wf_kernel.shape != (N_TAPS,):
            raise ValueError("wf_kernel must be the official 12-tap FALCON kernel")
        if abs(float(self.wf_kernel.sum()) - 1.0) > 1e-12:
            raise ValueError("wf_kernel must be normalized")
        if self.ridge_coef.ndim != 2 or self.ridge_coef.shape[1] != self.outputs:
            raise ValueError("bad ridge coefficient shape")
        if self.ridge_intercept.shape != (self.outputs,):
            raise ValueError("bad ridge intercept shape")
        if self.ridge_coef.shape[0] % self.history:
            raise ValueError("ridge feature width is not divisible by history")
        self.feature_dim = self.ridge_coef.shape[0] // self.history
        self.wf_kernel.setflags(write=False)
        self.ridge_coef.setflags(write=False)
        self.ridge_intercept.setflags(write=False)

    def _cache_session_arrays(self) -> dict[str, dict[str, np.ndarray]]:
        cached: dict[str, dict[str, np.ndarray]] = {}
        channels = self.task_config.n_channels
        for tag, record in self._sessions.items():
            if not isinstance(record, dict) or not isinstance(record.get("array_prefix"), str):
                raise ValueError("manifest session record is invalid for " + str(tag))
            prefix = record["array_prefix"]
            if not prefix or prefix in cached:
                raise ValueError("manifest has duplicate/empty array_prefix")
            arrays = {
                "z_mean": self._array(prefix + "_z_mean"),
                "z_scale": self._array(prefix + "_z_scale"),
                "A": self._array(prefix + "_A"),
                "b": self._array(prefix + "_b"),
            }
            valid = (
                arrays["z_mean"].shape == (channels,)
                and arrays["z_scale"].shape == (channels,)
                and not np.any(arrays["z_scale"] == 0.0)
                and arrays["A"].shape == (channels, self.feature_dim)
                and arrays["b"].shape == (self.feature_dim,)
            )
            if not valid:
                raise ValueError("invalid cached transform shapes for " + str(tag))
            for value in arrays.values():
                value.setflags(write=False)
            cached[prefix] = arrays
        return cached

    def _allocate_state(self) -> None:
        self._filter = np.zeros((N_TAPS, self.batch_size, self.task_config.n_channels), dtype=np.float64)
        self._history = np.zeros((self.history, self.batch_size, self.feature_dim), dtype=np.float64)
        self._last = np.zeros((self.batch_size, self.outputs), dtype=np.float32)

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        tags = [self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        if not tags or len(tags) > self.batch_size:
            raise ValueError("reset tags must contain 1..batch_size entries")
        unknown = [tag for tag in tags if tag not in self._sessions]
        if unknown:
            raise ValueError("payload has no transform for dataset tag(s): %s" % unknown)
        self._prefixes = [str(self._sessions[tag]["array_prefix"]) for tag in tags]
        self._active = len(tags)
        self._filter.fill(0.0)
        self._history.fill(0.0)
        self._last.fill(0.0)

    def _advance(self, neural_observations: np.ndarray) -> None:
        raw = np.ascontiguousarray(neural_observations, dtype=np.float64)
        if self._active == 0:
            raise RuntimeError("reset(dataset_tags) must be called before inference")
        if raw.ndim != 2 or raw.shape != (self._active, self.task_config.n_channels):
            raise ValueError("neural_observations must be [active_batch, n_channels]")
        if not np.isfinite(raw).all():
            raise ValueError("neural_observations must be finite")
        self._filter[1:, : self._active] = self._filter[:-1, : self._active]
        self._filter[0, : self._active] = raw
        filtered = np.einsum("t,tbc->bc", self.wf_kernel, self._filter[:, : self._active])
        transformed = np.empty((self._active, self.feature_dim), dtype=np.float64)
        for lane, prefix in enumerate(self._prefixes):
            arrays = self._session_arrays[prefix]
            z = (filtered[lane] - arrays["z_mean"]) / arrays["z_scale"]
            transformed[lane] = z @ arrays["A"] + arrays["b"]
        self._history[1:, : self._active] = self._history[:-1, : self._active]
        self._history[0, : self._active] = transformed
        flat = self._history[:, : self._active].transpose(1, 0, 2).reshape(self._active, -1)
        self._last[: self._active] = (flat @ self.ridge_coef + self.ridge_intercept).astype(np.float32)

    def observe(self, neural_observations: np.ndarray):
        self._advance(neural_observations)

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        self._advance(neural_observations)
        return self._last[: self._active].copy()

    def on_done(self, dones: np.ndarray):
        return None

    def set_batch_size(self, batch_size: int):
        batch_size = int(batch_size)
        if not 1 <= batch_size <= int(self.manifest["max_batch"]):
            raise ValueError("batch_size is outside the payload limit")
        self.batch_size = batch_size
        self._active = 0
        self._prefixes = []
        self._allocate_state()

    def payload_sha256(self) -> str:
        return hashlib.sha256(self.model_path.read_bytes()).hexdigest()
