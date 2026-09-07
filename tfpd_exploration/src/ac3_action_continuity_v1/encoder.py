"""The tiny AC3-0 encoders and direction readouts (CPU, deterministic).

Two model families, both inside the predeclared parameter budget of
``plan.MODELS`` and both fitted ONLY on the training source sessions of a fold:

* **R2** -- an ordinary supervised circular MLP: the encoder is trained
  end-to-end with a circular direction loss against the source direction
  labels (the E7-mirror control of section 23 amendment 3).
* **R3/R4/R5/RS** -- contrastive encoders trained with an InfoNCE objective on
  the section-9 pair batches, explicit speed/phase-matched opposite-direction
  hard negatives included.  The direction estimate of a contrastive row comes
  from a *linear circular probe* fitted by closed-form ridge on the training
  sessions with a frozen encoder; a label-free soft-prototype readout is
  recorded alongside as a diagnostic.

Torch is pinned to CPU and single-threaded, and every fit is seeded: the AC3-0
process gate forbids GPU training and this module never builds a CUDA tensor.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Sequence

import numpy as np

from . import plan
from . import summaries as su


class AC3EncoderError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3EncoderError(message)


EPOCHS = 100
STEPS_PER_EPOCH = 8
BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
HIDDEN = 32
PROBE_RIDGE = float(plan.MODELS["probe_ridge_lambda"])
EMBEDDING_GRID = tuple(int(item) for item in plan.MODELS["embedding_grid"])
TEMPERATURE_GRID = tuple(float(item) for item in plan.MODELS["temperature_grid"])
TORCH_SEED = int(plan.SAMPLING["seeds"]["torch"])
MAX_PARAMETERS = 2000


def torch_cpu():
    import torch

    torch.set_num_threads(1)
    return torch


def parameter_digest(module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(parameter.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def parameter_count(module) -> int:
    return int(sum(int(parameter.numel()) for parameter in module.parameters()))


def build_encoder(embedding_dim: int, seed: int = TORCH_SEED):
    torch = torch_cpu()
    _require(embedding_dim > 0, "embedding dim must be positive")
    torch.manual_seed(int(seed))
    module = torch.nn.Sequential(
        torch.nn.Linear(su.SUMMARY_DIM, HIDDEN),
        torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, embedding_dim),
    )
    count = parameter_count(module)
    _require(count <= MAX_PARAMETERS, f"AC3-0 encoder exceeded the parameter budget: {count}")
    return module


def embed_numpy(module, features: np.ndarray, *, differentiable: bool = False):
    torch = torch_cpu()
    matrix = np.asarray(features, dtype=np.float32)
    _require(matrix.ndim == 2 and matrix.shape[1] == su.SUMMARY_DIM, "encoder input shape drift")
    tensor = torch.as_tensor(matrix, dtype=torch.float32)
    output = module(tensor)
    if differentiable:
        norm = torch.clamp(torch.linalg.norm(output, dim=1, keepdim=True), min=1.0e-8)
        return output / norm
    with torch.no_grad():
        norm = torch.clamp(torch.linalg.norm(output, dim=1, keepdim=True), min=1.0e-8)
        return (output / norm).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# R2: ordinary supervised circular MLP.
# ---------------------------------------------------------------------------


def train_supervised(
    features: np.ndarray,
    target_thetas: np.ndarray,
    *,
    embedding_dim: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = TORCH_SEED,
) -> dict[str, object]:
    """Fit encoder + linear direction head with the circular loss."""
    torch = torch_cpu()
    matrix = np.asarray(features, dtype=np.float32)
    thetas = np.asarray(target_thetas, dtype=np.float64)
    _require(matrix.shape[0] == thetas.shape[0] and matrix.shape[0] >= batch_size,
             "supervised fit needs at least one batch of labelled states")
    _require(bool(np.isfinite(thetas).all()), "supervised fit received undefined direction labels")
    module = build_encoder(embedding_dim, seed=seed)
    torch.manual_seed(int(seed) + 1)
    head = torch.nn.Linear(embedding_dim, 2)
    parameters = list(module.parameters()) + list(head.parameters())
    _require(parameter_count(module) + parameter_count(head) <= MAX_PARAMETERS,
             "AC3-0 supervised model exceeded the parameter budget")
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    cos_target = torch.as_tensor(np.cos(thetas), dtype=torch.float32)
    sin_target = torch.as_tensor(np.sin(thetas), dtype=torch.float32)
    features_tensor = torch.as_tensor(matrix, dtype=torch.float32)
    count = int(matrix.shape[0])
    generator = torch.Generator().manual_seed(int(seed) + 2)
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        for _step in range(max(1, count // int(batch_size))):
            index = torch.randint(0, count, (int(batch_size),), generator=generator)
            z = embed_numpy(module, features_tensor[index].numpy(), differentiable=True)
            unit = head(z)
            unit = unit / torch.clamp(torch.linalg.norm(unit, dim=1, keepdim=True), min=1.0e-8)
            loss = (1.0 - (unit[:, 0] * cos_target[index] + unit[:, 1] * sin_target[index])).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return {
        "module": module,
        "head": head,
        "embedding_dim": int(embedding_dim),
        "final_loss": float(losses[-1]),
        "encoder_parameter_sha256": parameter_digest(module),
        "head_parameter_sha256": parameter_digest(head),
        "parameter_count": parameter_count(module) + parameter_count(head),
        "steps": len(losses),
    }


def supervised_directions(fit: dict[str, object], features: np.ndarray) -> np.ndarray:
    """Direction estimates of the supervised head for a feature matrix."""
    torch = torch_cpu()
    with torch.no_grad():
        z = embed_numpy(fit["module"], features)
        unit = fit["head"](torch.as_tensor(z, dtype=torch.float32)).numpy().astype(np.float64)
    return np.arctan2(unit[:, 1], unit[:, 0])


# ---------------------------------------------------------------------------
# R3/R4/R5/RS: contrastive encoders + linear circular probe.
# ---------------------------------------------------------------------------


def train_contrastive(
    sampler,
    *,
    arm: str,
    embedding_dim: int,
    temperature: float,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    batch_size: int = BATCH_SIZE,
    seed: int = TORCH_SEED,
) -> dict[str, object]:
    """Fit an encoder with InfoNCE on the section-9 pair batches."""
    torch = torch_cpu()
    _require(temperature > 0.0, "temperature must be positive")
    module = build_encoder(embedding_dim, seed=seed)
    optimizer = torch.optim.Adam(module.parameters(), lr=LEARNING_RATE)
    losses: list[float] = []
    audit_counts: list[dict[str, object]] = []
    step_index = 0
    for epoch in range(int(epochs)):
        for _step in range(int(steps_per_epoch)):
            batch = sampler.sample(arm=arm, batch_size=int(batch_size),
                                   seed=int(seed) + 1000 + epoch * int(steps_per_epoch) + _step)
            z_anchor = embed_numpy(module, batch.anchor_features, differentiable=True)
            z_positive = embed_numpy(module, batch.positive_features, differentiable=True)
            logits = z_anchor @ z_positive.T / float(temperature)
            positive_logit = torch.diagonal(logits)
            mask = torch.eye(int(batch.size), dtype=torch.bool)
            negatives = logits.masked_fill(mask, -1.0e9)
            hard = hard_logits_from_features(
                z_anchor, module, batch.hard_features, temperature,
            )
            denominator = torch.logsumexp(
                torch.cat([positive_logit[:, None], negatives, hard], dim=1), dim=1,
            )
            loss = -(positive_logit - denominator).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if step_index % 50 == 0:
                audit_counts.append(dict(batch.counts))
            step_index += 1
    return {
        "module": module,
        "embedding_dim": int(embedding_dim),
        "temperature": float(temperature),
        "arm": arm,
        "final_loss": float(losses[-1]),
        "encoder_parameter_sha256": parameter_digest(module),
        "parameter_count": parameter_count(module),
        "steps": len(losses),
        "sampler_audit_sample": audit_counts,
    }


def hard_logits_from_features(z_anchor, module, hard_features: np.ndarray, temperature: float):
    """Explicit hard-negative logits ``[B, k]`` from a feature tensor.

    ``hard_features`` is ``[B, k, 18]``; rows whose hard negative is absent
    carry an all-zero feature block and are masked to ``-1e9``.
    """
    torch = torch_cpu()
    features = np.asarray(hard_features, dtype=np.float32)
    _require(features.ndim == 3 and features.shape[2] == su.SUMMARY_DIM,
             "hard-negative feature block shape drift")
    batch, slots = int(features.shape[0]), int(features.shape[1])
    flat = features.reshape(batch * slots, su.SUMMARY_DIM)
    embeddings = embed_numpy(module, flat, differentiable=True).reshape(batch, slots, -1)
    logits = (z_anchor[:, None, :] * embeddings).sum(dim=2) / float(temperature)
    absent = ~torch.as_tensor(np.any(np.asarray(hard_features, dtype=bool) != 0, axis=2))
    # Absent slots must not contribute: their similarity is forced to -1e9.
    return logits.masked_fill(absent, -1.0e9)


def fit_linear_probe(embeddings: np.ndarray, target_thetas: np.ndarray,
                     ridge: float = PROBE_RIDGE) -> dict[str, object]:
    """Closed-form ridge probe ``d -> (cos, sin)`` on the training states."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    thetas = np.asarray(target_thetas, dtype=np.float64)
    _require(matrix.ndim == 2 and matrix.shape[0] == thetas.shape[0] and matrix.shape[0] >= 2,
             "linear probe needs at least two labelled states")
    _require(bool(np.isfinite(thetas).all()), "linear probe received undefined direction labels")
    design = np.concatenate([matrix, np.ones((matrix.shape[0], 1))], axis=1)
    target = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    regularizer = ridge * np.eye(design.shape[1])
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return {
        "weights": weights,
        "ridge": float(ridge),
        "n": int(matrix.shape[0]),
        "weights_sha256": hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest(),
    }


def probe_directions(probe: dict[str, object], embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    design = np.concatenate([matrix, np.ones((matrix.shape[0], 1))], axis=1)
    output = design @ np.asarray(probe["weights"], dtype=np.float64)
    return np.arctan2(output[:, 1], output[:, 0])


def linear_probe_accuracy(features: np.ndarray, labels: np.ndarray,
                          ridge: float = 1.0e-3) -> dict[str, object]:
    """Multiclass linear probe accuracy (the section-16 session-ID audit)."""
    matrix = np.asarray(features, dtype=np.float64)
    classes = np.asarray(labels, dtype=np.int64)
    _require(matrix.shape[0] == classes.shape[0] and matrix.shape[0] >= 2,
             "linear probe audit needs aligned features/labels")
    unique = np.unique(classes)
    design = np.concatenate([matrix, np.ones((matrix.shape[0], 1))], axis=1)
    one_hot = np.zeros((classes.shape[0], unique.size), dtype=np.float64)
    for index, value in enumerate(unique):
        one_hot[:, index] = (classes == value).astype(np.float64)
    regularizer = ridge * np.eye(design.shape[1])
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ one_hot)
    scores = design @ weights
    predicted = unique[np.argmax(scores, axis=1)]
    majority = unique[int(np.bincount(classes).argmax())]
    return {
        "accuracy": float((predicted == classes).mean()),
        "majority_class_accuracy": float((majority == classes).mean()),
        "n_classes": int(unique.size),
        "n": int(classes.shape[0]),
    }
