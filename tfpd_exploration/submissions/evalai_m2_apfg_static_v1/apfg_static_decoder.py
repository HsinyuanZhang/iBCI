"""FALCON runtime for the static-pool APFG M2 cached-identity submission.

Runtime semantics are those of the validated EvalAI M2 cached-identity decoder
line (images 578221/581359-581362/581644/581658/581659): ``reset(dataset_tags)``
selects the per-session cached identity, ``predict`` decodes one binned window
with the frozen coupled SPINT decoder, and ``on_done`` is an explicit no-op.

The only thing this deployment changes versus ``act30_dopt4`` is how the cached
identities were produced offline: each identity is the APFG gated identity
``E = native + tanh(alpha) * (post - native)`` with the frozen V2 refit alpha
applied at build time over the same D-opt-4 / first-30 static pool.  No gate,
adapter, optimizer, or trial metadata exists inside this runtime module: all
state carried across ``predict`` calls consists of frozen weights, the cached
identity tensors, and the fixed neural history buffers.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from falcon_challenge.interface import BCIDecoder

from third_party.falcon_challenge.filtering import (
    NEURAL_TAU_MS,
    apply_exponential_filter,
)

PAYLOAD_SCHEMA = "e8_t4_m2_apfg_static_cached_identity_v1"
PAYLOAD_ARM = "apfg_static_act30_dopt4"
PAYLOAD_LABEL_BUDGET = 4
PAYLOAD_ACTIVITY_BUDGET = 30
EXPECTED_FROZEN_ALPHA = -0.20759029686450958
EXPECTED_SESSION_COUNT = 13


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 records ndarray helpers below ``numpy._core`` while the
        # validated EvalAI base image ships NumPy 1.x (``numpy.core``).
        # These are equivalent for this plain numeric payload.
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(
                io.BytesIO(value), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


class ApfgStaticM2Decoder(BCIDecoder):
    """Decode with one APFG-gated static identity per session.

    The payload contains the frozen coupled SPINT decoder and ``E[N,50]``
    identities produced offline from each session's D-opt-4 selected
    calibration trials, the label-free first-30 calibration activity pool, and
    the frozen anchored post-fusion scalar gate.  No calibration data,
    selection law, gate parameter, optimizer state, or trial metadata is
    needed or read at runtime.
    """

    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        with open(model_path, "rb") as handle:
            payload = CPUUnpickler(handle).load()
        if payload.get("schema_version") != PAYLOAD_SCHEMA:
            raise ValueError("Unsupported or non-APFG decoder payload")
        if payload.get("task") != task_config.task:
            raise ValueError("APFG payload task does not match evaluator task")
        metadata = payload.get("metadata", {})
        # Bind the runtime to the static-pool APFG package so a payload from
        # any other arm cannot be silently served by this image.
        if metadata.get("arm") != PAYLOAD_ARM:
            raise ValueError(
                f"payload arm {metadata.get('arm')!r} is not the frozen {PAYLOAD_ARM!r} package"
            )
        if int(metadata.get("label_budget", -1)) != PAYLOAD_LABEL_BUDGET:
            raise ValueError("payload label budget is not the D-opt-4 selection")
        if int(metadata.get("activity_budget", -1)) != PAYLOAD_ACTIVITY_BUDGET:
            raise ValueError("payload activity budget is not the first-30 calibration pool")
        if float(metadata.get("frozen_alpha", "nan")) != EXPECTED_FROZEN_ALPHA:
            raise ValueError("payload frozen alpha is not the learned V2 gate")
        self.decoder = payload["decoder"].eval()
        self.identity_by_dataset_tag = payload["identity_by_dataset_tag"]
        self.window_size = int(payload["window_size"])
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        self.smooth_observations = bool(payload.get("smooth_observations", False))
        if len(self.identity_by_dataset_tag) != EXPECTED_SESSION_COUNT:
            raise ValueError("M2 APFG payload must cover exactly 13 calibration sessions")
        max_history = int(NEURAL_TAU_MS / task_config.bin_size_ms) * 5
        self.raw_history_buffer = np.zeros(
            (max_history, self.batch_size, task_config.n_channels), dtype=np.float32
        )
        self.observation_buffer = np.zeros(
            (self.window_size, self.batch_size, task_config.n_channels), dtype=np.float32
        )
        self.device = torch.device("cpu")
        self.local_identities: list[torch.Tensor] = []
        # Frozen-weight fingerprint for the no-online-update contract proof.
        self._parameter_fingerprint = self._fingerprint_parameters()

    # -- contract instrumentation (read-only; never used by predict) --------

    def _fingerprint_parameters(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        with torch.no_grad():
            for parameter in self.decoder.parameters():
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def parameter_fingerprint(self) -> str:
        """SHA-256 over every frozen decoder parameter (contract audit hook)."""
        return self._fingerprint_parameters()

    def identity_fingerprint(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for tag in sorted(self.identity_by_dataset_tag):
            array = np.ascontiguousarray(self.identity_by_dataset_tag[tag])
            digest.update(tag.encode())
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    # -- BCIDecoder interface ----------------------------------------------

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        hashed_tags = [
            self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags
        ]
        missing = [tag for tag in hashed_tags if tag not in self.identity_by_dataset_tag]
        if missing:
            raise ValueError(
                f"Dataset tags {missing} are absent from the APFG calibration payload"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = self.decoder.to(self.device).eval()
        self.local_identities = [
            torch.as_tensor(
                self.identity_by_dataset_tag[tag], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            for tag in hashed_tags
        ]
        self.raw_history_buffer.fill(0.0)
        self.observation_buffer.fill(0.0)

    def on_done(self, dones: np.ndarray):
        # Explicit no-op.  For a continual task (h1/m1/m2) the official
        # evaluator never calls on_done (sealed audit
        # results/cdm_p1_m2_v1/audit.json), and even if boundaries were
        # delivered they would reach no state machine here.
        return None

    def _decode_with_identity(
        self, neural: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        source = neural.permute(0, 2, 1) + identity
        source = self.decoder.fc_in(source)
        query = self.decoder.fc_in(self.decoder.rep).to(source)
        transformed, _ = self.decoder.transformer(
            query.repeat(source.shape[0], 1, 1), source
        )
        return self.decoder.fc_out(transformed).permute(0, 2, 1)

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if not self.local_identities:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        self.observe(neural_observations)
        decoder_input = torch.as_tensor(
            self.observation_buffer.copy().transpose(1, 0, 2),
            dtype=torch.float32,
            device=self.device,
        )
        predictions = []
        with torch.inference_mode():
            for index, identity in enumerate(self.local_identities):
                predictions.append(
                    self._decode_with_identity(
                        decoder_input[index : index + 1], identity
                    )
                )
        prediction = torch.cat(predictions, dim=0)
        return (
            prediction[:, -1, :].cpu().numpy() / self.behavior_scaling_factor
        )

    def observe(self, neural_observations: np.ndarray):
        observations = np.asarray(neural_observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.task_config.n_channels:
            raise ValueError(
                f"Expected neural observations [B,{self.task_config.n_channels}], "
                f"got {observations.shape}"
            )
        if observations.shape[0] > self.batch_size:
            raise ValueError("Evaluator batch exceeds configured decoder batch size")
        if observations.shape[0] < self.batch_size:
            observations = np.pad(
                observations,
                ((0, self.batch_size - observations.shape[0]), (0, 0)),
            )
        self.raw_history_buffer = np.roll(self.raw_history_buffer, -1, axis=0)
        self.raw_history_buffer[-1] = observations
        if self.smooth_observations:
            history, batch, channels = self.raw_history_buffer.shape
            flat = self.raw_history_buffer.reshape(history, batch * channels)
            smoothed = apply_exponential_filter(
                flat,
                tau=NEURAL_TAU_MS,
                bin_size=self.task_config.bin_size_ms,
            )
            latest = smoothed[-1].reshape(batch, channels).astype(np.float32)
        else:
            latest = observations
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = latest
