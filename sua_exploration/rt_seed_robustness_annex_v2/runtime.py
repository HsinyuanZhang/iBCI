"""Runtime-only hooks for the RT seed-robustness annex v2.

This module is deliberately isolated from the shared RT runner.  The hook is
added to a source-training command through a Hydra callback override; it does
not monkeypatch Lightning, the data module, or the evaluator.  It records the
student state immediately before the first optimizer step and emits the small
hardware/accounting profile needed by the annex.  Target evaluation is
handled by :mod:`launcher` after the existing one-shot evaluator has written
its legacy receipt.

The module contains no NWB access and cannot launch a Trainer by itself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

try:  # Keep static/spec tests importable if Lightning is not installed.
    from lightning.pytorch import Callback
    _LIGHTNING_IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - exercised only in minimal envs
    Callback = object  # type: ignore[misc,assignment]
    _LIGHTNING_IMPORT_ERROR = error


ALLOWED_ARMS = ("afc4_vel", "afc4_mb4")
ACCOUNTING_BATCH_SIZE = 1
ACCOUNTING_NUM_UNITS = 64
PROVENANCE_HASH_FIELDS = (
    "implementation_snapshot_sha256",
    "train_entry_sha256",
    "runner_source_sha256",
    "evaluator_source_sha256",
    "experiment_config_sha256",
    "data_config_sha256",
    "model_config_sha256",
    "selection_callback_sha256",
    "runtime_hook_sha256",
)


class AnnexRuntimeError(RuntimeError):
    """Raised when the isolated annex hook cannot prove its contract."""


def stable_state_hash(state: Mapping[str, Any]) -> str:
    """Hash a tensor state dictionary with names, dtype, shape, and bytes.

    Including dtype and shape avoids a false equality when two tensors happen
    to have the same raw byte stream under different interpretations.  The
    canonical order is independent of Python insertion order.
    """

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not hasattr(value, "detach") or not hasattr(value, "shape"):
            raise AnnexRuntimeError(f"state entry {name!r} is not a tensor-like value")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(int(x) for x in tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AnnexRuntimeError(f"refusing to overwrite annex receipt: {path}")
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_int(value: Any, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AnnexRuntimeError(f"{label} must be an integer") from error
    return result


def _sha256(value: str, *, label: str) -> str:
    result = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise AnnexRuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return result


class AnnexInitialStateCallback(Callback):
    """Record the initialized student before source optimization starts.

    The callback is deliberately narrow.  It does not alter the optimizer or
    model.  Both paired arms receive the same fold/seed and are required to
    produce the same initial hash; the isolated postflight validator checks
    that equality after both source fits finish.
    """

    def __init__(
        self,
        output_path: str,
        arm: str,
        fold: int,
        seed: int,
        implementation_snapshot_sha256: str,
        train_entry_sha256: str,
        runner_source_sha256: str = "",
        evaluator_source_sha256: str = "",
        experiment_config_sha256: str = "",
        data_config_sha256: str = "",
        model_config_sha256: str = "",
        selection_callback_sha256: str = "",
        runtime_hook_sha256: str = "",
        initial_state_phase: str = "",
        run_id: str = "",
    ) -> None:
        if _LIGHTNING_IMPORT_ERROR is not None:
            raise AnnexRuntimeError(
                "Lightning is required by the runtime callback"
            ) from _LIGHTNING_IMPORT_ERROR
        if arm not in ALLOWED_ARMS:
            raise AnnexRuntimeError(f"unsupported annex arm: {arm!r}")
        self.output_path = Path(output_path)
        self.arm = str(arm)
        self.fold = _as_int(fold, label="fold")
        self.seed = _as_int(seed, label="seed")
        provenance = {
            "implementation_snapshot_sha256": implementation_snapshot_sha256,
            "train_entry_sha256": train_entry_sha256,
            "runner_source_sha256": runner_source_sha256,
            "evaluator_source_sha256": evaluator_source_sha256,
            "experiment_config_sha256": experiment_config_sha256,
            "data_config_sha256": data_config_sha256,
            "model_config_sha256": model_config_sha256,
            "selection_callback_sha256": selection_callback_sha256,
            "runtime_hook_sha256": runtime_hook_sha256,
        }
        self.provenance = {
            field: _sha256(value, label=field) for field, value in provenance.items()
        }
        if initial_state_phase != "before_first_optimizer_step":
            raise AnnexRuntimeError(
                "initial_state_phase must be before_first_optimizer_step"
            )
        self.initial_state_phase = initial_state_phase
        self.run_id = str(run_id)

    @staticmethod
    def _accounting(student: Any) -> dict[str, int | str]:
        if not hasattr(student, "decoder_cost_comparison_receipt"):
            raise AnnexRuntimeError(
                "shared StreamingSpintModel has no decoder_cost_comparison_receipt"
            )
        cost = student.decoder_cost_comparison_receipt(
            batch_size=ACCOUNTING_BATCH_SIZE,
            num_neurons=ACCOUNTING_NUM_UNITS,
        )
        if not isinstance(cost, Mapping) or cost.get("active_mode") != "coupled":
            raise AnnexRuntimeError(
                "RT seed annex requires the ordinary coupled decoder accounting"
            )
        coupled = cost.get("coupled")
        if not isinstance(coupled, Mapping):
            raise AnnexRuntimeError("coupled decoder accounting is missing")
        total = _as_int(coupled.get("total"), label="coupled.total")
        cached = _as_int(
            coupled.get("persistent_state_bytes_fp32"),
            label="coupled.persistent_state_bytes_fp32",
        )
        parameter_count = sum(int(parameter.numel()) for parameter in student.parameters())
        if parameter_count <= 0 or total <= 0 or cached < 0:
            raise AnnexRuntimeError("non-positive runtime accounting")
        return {
            "parameter_count": parameter_count,
            "macs_per_decode_call": total,
            "cached_state_bytes": cached,
            "accounting_batch_size": ACCOUNTING_BATCH_SIZE,
            "accounting_num_units": ACCOUNTING_NUM_UNITS,
            "accounting_mode": "coupled_decoder_receipt",
        }

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        student = getattr(pl_module, "student", None)
        if student is None or not hasattr(student, "state_dict"):
            raise AnnexRuntimeError(
                "student must be initialized before AnnexInitialStateCallback.on_fit_start"
            )
        state = student.state_dict()
        initial_hash = stable_state_hash(state)
        accounting = self._accounting(student)
        payload: dict[str, Any] = {
            "schema": "rt_seed_robustness_annex_v2_source_initial_state_v1",
            "status": "PASS_SOURCE_INITIAL_STATE_RECORDED",
            "development_only": True,
            "formal_heldout_opened": False,
            "arm": self.arm,
            "fold": self.fold,
            "seed": self.seed,
            "run_id": self.run_id,
            "initial_state_hash": initial_hash,
            "initial_state_scope": "student.state_dict immediately before source fit",
            "initial_state_phase": self.initial_state_phase,
            "source_training_optimizer": "allowed_and_expected",
            "trainer_default_root_dir": str(getattr(trainer, "default_root_dir", "")),
            "trainer_log_dir": str(getattr(trainer, "log_dir", "")),
            **self.provenance,
            "accounting": accounting,
        }
        _write_exclusive(self.output_path, payload)
