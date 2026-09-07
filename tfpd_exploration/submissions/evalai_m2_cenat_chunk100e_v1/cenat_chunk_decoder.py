"""FALCON runtime for the Ce-NAT chunk100e continual TTA M2 submission.

Deployment law (bitwise-matched to the local scorer that produced 0.37022):

- per dataset tag, the payload carries the frozen jointly fine-tuned SPINT
  decoder, the frozen id encoder, the first-30 calibration seed pool
  [30,100,96] and the normalized D-opt4-carrier side features [96,4];
- at ``reset`` each batch slot gets INDEPENDENT state: the seed pool
  (chronological seed prefix-4 rows are permanently protected; FIFO eviction
  of the oldest non-protected row beyond 30), the candidate chunk buffer
  pre-seeded with EXACTLY 49 zero rows (the dataset's W50 zero pre-history),
  the candidate-rate history (EVERY candidate enters the history; the running
  median of past candidates is the acceptance threshold), and the identity
  computed from the seed pool;
- ``predict`` per bin, strictly causal decode-before-commit:
    1. roll the observed bin into the 50-bin decode window;
    2. decode the current bin with the CURRENT identity;
    3. only then append the observed bin to the 100-bin candidate buffer;
    4. when the candidate completes: compute its mean rate, compare against
       the median of PREVIOUS candidate rates, append this rate to the
       history, and if accepted commit the chunk to the pool, evict, and
       recompute the identity (effective from the NEXT bin);
- ``on_done`` is an explicit no-op (continual M2 never calls it, and the law
  never uses boundaries).

No labels, no eval mask, no trial metadata exist in this module.
"""

from __future__ import annotations

import io
import pickle
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from falcon_challenge.interface import BCIDecoder

from third_party.falcon_challenge.filtering import NEURAL_TAU_MS  # noqa: F401  (parity with prior images)

PAYLOAD_SCHEMA = "e8_t4_m2_cenat_chunk100e_tta_v1"
PAYLOAD_ARM = "cenat_chunk100e_tta"
CHUNK_BINS = 100
POOL_CAPACITY = 30
PROTECTED_PREFIX = 4
PREHISTORY_ZEROS = 49
EXPECTED_SESSION_COUNT = 13


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(
                io.BytesIO(value), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


class PayloadIdEncoder(torch.nn.Module):
    """Container-side reconstruction of the frozen B3S identity encoder.

    Mirrors ``SideFeatureEarlyPoolEncoder`` (streaming_encoders.py) op-for-op:
    per-trial ``pre_pool`` (Linear+ReLU over time) with an arrival-order
    running sum, then mean, side concat, and the 3-layer ``post_pool`` MLP.
    Weights come from the exported state dict (strict load).
    """

    def __init__(self, trial_length: int = 100, window_size: int = 50,
                 hidden_dim: int = 64, side_dim: int = 4, num_post_layers: int = 3):
        super().__init__()
        self.trial_length = trial_length
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.side_dim = side_dim
        self.pre_pool = torch.nn.Sequential(
            torch.nn.Linear(trial_length, hidden_dim), torch.nn.ReLU())
        post_in = hidden_dim + side_dim
        layers = [torch.nn.Linear(post_in, hidden_dim)]
        if num_post_layers > 1:
            layers.append(torch.nn.ReLU())
            for _ in range(num_post_layers - 2):
                layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])
            layers.append(torch.nn.Linear(hidden_dim, window_size))
        self.post_pool = torch.nn.Sequential(*layers)

    def forward_batch(self, calib_trials, trial_lengths=None, side_features=None, electrode_ids=None):
        assert calib_trials.dim() == 4, "expected [B,M,T,N] calibration"
        x = calib_trials.permute(0, 1, 3, 2)  # [B,M,N,T]
        batch, trials, neurons, length = x.shape
        total = torch.zeros(batch, neurons, self.hidden_dim,
                            device=x.device, dtype=x.dtype)
        for index in range(trials):  # arrival-order running sum (matches push_trial)
            total = total + self.pre_pool(x[:, index])
        mean_feat = total / float(trials)
        features = torch.cat([mean_feat, side_features.expand(-1, neurons, -1)], dim=-1)
        return self.post_pool(features)


class _SlotState:
    """Independent continual state for one batch slot."""

    def __init__(self, seed_pool: np.ndarray, side: np.ndarray, decoder, id_encoder):
        self.pool = deque(np.ascontiguousarray(seed_pool, dtype=np.float32))
        self.side = torch.as_tensor(np.ascontiguousarray(side, dtype=np.float32)).unsqueeze(0)
        self.decoder = decoder
        self.id_encoder = id_encoder
        # The dataset stream carries a 49-row zero pre-history before the
        # first real bin (W50 convention); the candidate buffer starts with
        # EXACTLY those zero rows so chunk completion bins match the local
        # scorer bit-for-bit.
        self.candidate = np.zeros((PREHISTORY_ZEROS, self.pool[0].shape[1]), dtype=np.float32)
        self.rate_history: list[float] = []
        self.accepted_count = 0
        with torch.inference_mode():
            support = torch.from_numpy(self.stack()).unsqueeze(0)
            self.identity = id_encoder.forward_batch(support, side_features=self.side)

    def stack(self) -> np.ndarray:
        return np.ascontiguousarray(np.stack(list(self.pool)), dtype=np.float32)

    def recompute_identity(self):
        with torch.inference_mode():
            support = torch.from_numpy(self.stack()).unsqueeze(0)
            self.identity = self.id_encoder.forward_batch(support, side_features=self.side)

    def append_bin(self, row: np.ndarray):
        self.candidate = np.concatenate(
            [self.candidate, np.ascontiguousarray(row, dtype=np.float32).reshape(1, -1)], axis=0)
        if self.candidate.shape[0] == CHUNK_BINS:
            rate = float(self.candidate.mean(dtype=np.float64))
            accepted = bool(len(self.rate_history) == 0
                            or rate >= float(np.median(self.rate_history)))
            self.rate_history.append(rate)
            if accepted:
                self.accepted_count += 1
                self.pool.append(self.candidate)
                if len(self.pool) > POOL_CAPACITY:
                    del self.pool[PROTECTED_PREFIX]  # oldest non-protected row
                self.recompute_identity()
            self.candidate = np.zeros((0, self.candidate.shape[1]), dtype=np.float32)


class CenatChunkTTADecoder(BCIDecoder):
    """Decode with a continual, label-free, energy-gated activity memory."""

    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.batch_size = int(batch_size)
        with open(model_path, "rb") as handle:
            payload = CPUUnpickler(handle).load()
        if payload.get("schema_version") != PAYLOAD_SCHEMA:
            raise ValueError("Unsupported or non-Ce-NAT TTA payload")
        if payload.get("task") != task_config.task:
            raise ValueError("payload task does not match evaluator task")
        metadata = payload.get("metadata", {})
        if metadata.get("arm") != PAYLOAD_ARM:
            raise ValueError(f"payload arm {metadata.get('arm')!r} is not {PAYLOAD_ARM!r}")
        if not bool(metadata.get("test_time_adaptive", False)):
            raise ValueError("payload does not declare test-time adaptation")
        if int(metadata.get("label_budget", -1)) != 4:
            raise ValueError("payload label budget drift")
        self.decoder = payload["decoder"].eval()
        self.id_encoder = PayloadIdEncoder()
        self.id_encoder.load_state_dict(payload["id_encoder_state"], strict=True)
        self.id_encoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.id_encoder.parameters():
            parameter.requires_grad_(False)
        self.window_size = int(payload["window_size"])
        self.behavior_scaling_factor = float(payload["behavior_scaling_factor"])
        self.seed_by_tag = payload["seed_pool_by_dataset_tag"]
        self.side_by_tag = payload["side_by_dataset_tag"]
        if len(self.seed_by_tag) != EXPECTED_SESSION_COUNT:
            raise ValueError("payload must cover exactly 13 calibration sessions")
        self.raw_history_buffer = np.zeros(
            (int(NEURAL_TAU_MS / task_config.bin_size_ms) * 5, self.batch_size, task_config.n_channels),
            dtype=np.float32)
        self.observation_buffer = np.zeros(
            (self.window_size, self.batch_size, task_config.n_channels), dtype=np.float32)
        self.device = torch.device("cpu")
        self.slots: list[_SlotState | None] = [None] * self.batch_size
        self._parameter_fingerprint = self._fingerprint_parameters()

    # -- contract instrumentation ------------------------------------------

    def _fingerprint_parameters(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        with torch.no_grad():
            for parameter in self.decoder.parameters():
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            for parameter in self.id_encoder.parameters():
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def parameter_fingerprint(self) -> str:
        return self._fingerprint_parameters()

    def slot_audit(self) -> list[dict]:
        return [{"rate_history_len": len(s.rate_history),
                 "accepted_commits": s.accepted_count,
                 "pool_len": len(s.pool),
                 "candidate_rows": int(s.candidate.shape[0])} for s in self.slots]

    # -- BCIDecoder interface ----------------------------------------------

    @staticmethod
    def _stem(value) -> str:
        return Path(value).stem

    def reset(self, dataset_tags: Iterable[Path] = (Path(""),)):
        hashed_tags = [self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        missing = [tag for tag in hashed_tags if tag not in self.seed_by_tag]
        if missing:
            raise ValueError(f"dataset tags {missing} absent from the payload")
        self.device = torch.device("cpu")
        self.decoder = self.decoder.to(self.device).eval()
        self.id_encoder = self.id_encoder.to(self.device).eval()
        self.slots = []
        for tag in hashed_tags:
            self.slots.append(_SlotState(
                self.seed_by_tag[tag], self.side_by_tag[tag], self.decoder, self.id_encoder))
        self.raw_history_buffer.fill(0.0)
        self.observation_buffer.fill(0.0)

    def on_done(self, dones: np.ndarray):
        return None  # continual task: never called; the law never uses boundaries

    def _decode_slot(self, index: int) -> torch.Tensor:
        state = self.slots[index]
        window = np.ascontiguousarray(self.observation_buffer[:, index, :].T, dtype=np.float32)  # [N,W]
        source = torch.from_numpy(window).unsqueeze(0)  # [1,N,W]
        source = source + state.identity
        source = self.decoder.fc_in(source)
        query = self.decoder.fc_in(self.decoder.rep).to(source)
        transformed, _ = self.decoder.transformer(query.repeat(source.shape[0], 1, 1), source)
        return self.decoder.fc_out(transformed).permute(0, 2, 1)

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if any(slot is None for slot in self.slots):
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        observations = np.asarray(neural_observations, dtype=np.float32)
        n = len(self.slots)  # one slot per file in THIS evaluator wave (7 or 6)
        if observations.shape[0] != n:
            raise ValueError(
                f"evaluator batch rows {observations.shape[0]} != active wave slots {n}")
        predictions = np.zeros((n, 2), dtype=np.float32)
        # 1) ONE roll per step for the whole batch (rolling per slot would
        #    wrap other slots' windows and corrupt their history).
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        for index in range(n):
            self.observation_buffer[-1, index] = observations[index]
        # 2) decode BEFORE any commit, with the pre-step identity
        with torch.inference_mode():
            for index in range(n):
                predictions[index] = (
                    self._decode_slot(index)[:, -1, :].cpu().numpy().astype(np.float32, copy=False)
                    / self.behavior_scaling_factor)[0]
        # 3) only now update the continual states.  Padded suffix slots
        #    (all-zero rows, strictly a suffix of the batch) also advance:
        #    the gate rejects zero-rate candidates, so they are inert by
        #    design and cannot affect earlier real-bin predictions.
        for index in range(n):
            self.slots[index].append_bin(observations[index])
        return predictions

    def observe(self, neural_observations: np.ndarray):
        raise NotImplementedError("official continual M2 never calls observe")
