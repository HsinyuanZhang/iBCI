"""Frozen, NumPy-only streaming runtime for external GF baselines.

The payload is deliberately numeric only.  In particular it contains neither
training code nor calibration/query labels, and this module never imports the
workspace, sklearn, or pickle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from falcon_challenge.interface import BCIDecoder


PAYLOAD_SCHEMA = "external_gf_falcon_payload_v1"


class ExternalGFFalconDecoder(BCIDecoder):
    """Causal 10-bin CORAL/AlignedFA + frozen Wiener readout."""

    def __init__(self, task_config, model_path: str = "/data/payload.npz", *,
                 manifest_path: str = "/data/manifest.json", batch_size: int = 1):
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
        self._load_common_weights()
        self._sessions = self.manifest["sessions"]
        self._session_arrays = self._cache_session_arrays()
        self._active = 0
        self._prefixes: list[str] = []
        self._allocate_state()

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
        if self.history != 10:
            raise ValueError("this frozen runtime requires exactly 10 causal bins")
        if int(m.get("max_batch", self.batch_size)) < 1:
            raise ValueError("invalid manifest max_batch")
        if self.batch_size > int(m.get("max_batch", self.batch_size)):
            raise ValueError("configured batch exceeds payload max_batch")
        if not isinstance(m.get("sessions"), dict) or not m["sessions"]:
            raise ValueError("manifest sessions must be a nonempty tag mapping")

    def _task_name(self) -> str:
        value = getattr(self.task_config, "task", None)
        return getattr(value, "name", str(value)).split(".")[-1].lower()

    def _output_dim(self) -> int:
        # Published falcon_challenge calls this ``out_dim``.  The fallback is
        # useful for tiny interface doubles in offline runtime tests.
        if hasattr(self.task_config, "out_dim"):
            return int(self.task_config.out_dim)
        return int(self.task_config.n_outputs)

    def _array(self, key: str, *, dtype=np.float64) -> np.ndarray:
        if key not in self.payload.files:
            raise ValueError("payload lacks %s" % key)
        value = np.asarray(self.payload[key], dtype=dtype)
        if not np.isfinite(value).all():
            raise ValueError("payload %s is nonfinite" % key)
        return np.ascontiguousarray(value)

    def _load_common_weights(self) -> None:
        self.wf_mean = self._array("wf_mean")
        self.wf_scale = self._array("wf_scale")
        self.wf_coef = self._array("wf_coef")
        if self.wf_mean.ndim != 1 or self.wf_scale.shape != self.wf_mean.shape:
            raise ValueError("bad Wiener mean/scale shapes")
        if np.any(self.wf_scale == 0.0):
            raise ValueError("Wiener scale contains zero")
        if self.wf_coef.shape != (self.wf_mean.size + 1, self.outputs):
            raise ValueError("bad Wiener coefficient shape")
        if self.wf_mean.size % self.history:
            raise ValueError("Wiener feature width is not divisible by history")
        self.feature_dim = self.wf_mean.size // self.history
        for value in (self.wf_mean, self.wf_scale, self.wf_coef):
            value.setflags(write=False)

    def _cache_session_arrays(self) -> dict[str, dict[str, np.ndarray]]:
        """Read compressed NPZ arrays once; never do I/O in a streaming bin."""
        cached: dict[str, dict[str, np.ndarray]] = {}
        for tag, record in self._sessions.items():
            if not isinstance(record, dict) or not isinstance(record.get("array_prefix"), str):
                raise ValueError("manifest session record is invalid for " + str(tag))
            prefix = record["array_prefix"]
            if not prefix or prefix in cached:
                raise ValueError("manifest has duplicate/empty array_prefix")
            if self.manifest["method"] == "coral":
                names = ("target_mean", "source_mean", "target_invroot", "source_root")
                arrays = {name: self._array(prefix + "_" + name) for name in names}
                c = self.task_config.n_channels
                valid = (arrays["target_mean"].shape == (c,) and arrays["source_mean"].shape == (c,)
                         and arrays["target_invroot"].shape == (c, c) and arrays["source_root"].shape == (c, c)
                         and self.feature_dim == c)
            elif self.manifest["method"] == "aligned_fa":
                names = ("target_fa_mean", "Wpsi", "cov_z", "rotation")
                arrays = {name: self._array(prefix + "_" + name) for name in names}
                k = arrays["Wpsi"].shape[0] if arrays["Wpsi"].ndim == 2 else -1
                c = self.task_config.n_channels
                valid = (arrays["target_fa_mean"].shape == (c,) and arrays["Wpsi"].shape == (k, c)
                         and arrays["cov_z"].shape == (k, k) and arrays["rotation"].shape == (k, k)
                         and self.feature_dim == k)
            else:
                raise ValueError("unsupported method in manifest")
            if not valid:
                raise ValueError("invalid cached transform shapes for " + str(tag))
            for value in arrays.values():
                value.setflags(write=False)
            cached[prefix] = arrays
        return cached

    def _allocate_state(self) -> None:
        self._history = np.zeros((self.history, self.batch_size, self.feature_dim), dtype=np.float32)
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
        if any(prefix not in self._session_arrays for prefix in self._prefixes):
            raise ValueError("manifest contains an empty array_prefix")
        self._active = len(tags)
        self._history.fill(0.0)  # literal transformed-space zero padding
        self._last.fill(0.0)

    def _transform_one(self, raw: np.ndarray, prefix: str) -> np.ndarray:
        arrays = self._session_arrays[prefix]
        if self.manifest["method"] == "coral":
            target_mean = arrays["target_mean"]
            source_mean = arrays["source_mean"]
            target_invroot = arrays["target_invroot"]
            source_root = arrays["source_root"]
            value = (raw.astype(np.float64) - target_mean) @ target_invroot @ source_root + source_mean
        elif self.manifest["method"] == "aligned_fa":
            mean = arrays["target_fa_mean"]
            wpsi = arrays["Wpsi"]
            cov_z = arrays["cov_z"]
            rotation = arrays["rotation"]
            value = (raw.astype(np.float64) - mean) @ wpsi.T @ cov_z @ rotation
        else:
            raise ValueError("unsupported method in manifest")
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (self.feature_dim,):
            raise ValueError("session transform has wrong feature width")
        return value

    def _advance(self, neural_observations: np.ndarray) -> None:
        raw = np.ascontiguousarray(neural_observations, dtype=np.float32)
        if self._active == 0:
            raise RuntimeError("reset(dataset_tags) must be called before inference")
        if raw.ndim != 2 or raw.shape != (self._active, self.task_config.n_channels):
            raise ValueError("neural_observations must be [active_batch, n_channels]")
        if not np.isfinite(raw).all():
            raise ValueError("neural_observations must be finite")
        transformed = np.empty((self._active, self.feature_dim), dtype=np.float32)
        for lane, prefix in enumerate(self._prefixes):
            transformed[lane] = self._transform_one(raw[lane], prefix)
        self._history[:-1, :self._active] = self._history[1:, :self._active]
        self._history[-1, :self._active] = transformed
        flat = self._history[:, :self._active].transpose(1, 0, 2).reshape(self._active, -1)
        standardized = (flat.astype(np.float64) - self.wf_mean) / self.wf_scale
        design = np.empty((self._active, standardized.shape[1] + 1), dtype=np.float64)
        design[:, :-1] = standardized
        design[:, -1] = 1.0
        self._last[:self._active] = (design @ self.wf_coef).astype(np.float32)

    def observe(self, neural_observations: np.ndarray):
        self._advance(neural_observations)

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        self._advance(neural_observations)
        # FalconEvaluator retains every returned step until it stacks a full
        # stream.  Return an owned snapshot rather than a view of _last.
        return self._last[:self._active].copy()

    def on_done(self, dones: np.ndarray):
        # FALCON's M2/M1/H1 continual evaluator does not call this; its history
        # intentionally crosses trial boundaries.  Keep the state for forwards compatibility.
        return None

    def set_batch_size(self, batch_size: int):
        """Honor the optional BCIDecoder API without retaining stale lanes."""
        batch_size = int(batch_size)
        if not 1 <= batch_size <= int(self.manifest["max_batch"]):
            raise ValueError("batch_size is outside the payload limit")
        self.batch_size = batch_size
        self._active = 0
        self._prefixes = []
        self._allocate_state()

    def payload_sha256(self) -> str:
        return hashlib.sha256(self.model_path.read_bytes()).hexdigest()
