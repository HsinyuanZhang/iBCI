#!/usr/bin/env python3
"""Runtime-only compatibility wrapper for the frozen C1 evidence finalizer.

The completed C1 evidence contains two runtime-receipt representation differences:

* PyTorch records a bare GPU UUID while ``nvidia-smi`` prefixes the same UUID
  with ``GPU-``; and
* shared-view post-run cost receipts record a single visible Torch device plus
  the physical ``nvidia-smi`` inventory instead of repeating the start status's
  full-host runtime dictionary.

This wrapper does not alter evidence files, model/data inputs, artifacts, scores,
or aggregation rules.  It loads the hash-pinned frozen finalizer, canonicalizes
only UUID spelling for its existing full-runtime normalizer, and validates the
legacy scoped cost schema before presenting the already-validated start runtime
to the original equality check.  Every other finalizer function remains the
original function object.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
FROZEN_FINALIZER_PATH = ROOT / "sua_exploration/scripts/finalize_t4_paired_view_c1_evidence.py"
FROZEN_FINALIZER_SHA256 = "48164ee4e97dfeff87911fa6190f9c277fe59b872f3e6b3e5d424c4d815bc485"


class RuntimeCompatibilityError(RuntimeError):
    """Raised before the frozen finalizer is loaded when its hash has drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_gpu_uuid(value: Any) -> Any:
    """Canonicalize only the reversible NVIDIA ``GPU-`` prefix and ASCII case."""

    if not isinstance(value, str):
        return value
    if value[:4].lower() == "gpu-":
        value = value[4:]
    return value.lower()


def canonicalize_full_runtime_environment(environment: Any) -> Any:
    """Copy a full runtime dictionary and canonicalize only its GPU UUID fields."""

    if not isinstance(environment, dict):
        return environment
    normalized = copy.deepcopy(environment)
    torch_rows = normalized.get("torch_gpus")
    if isinstance(torch_rows, list):
        for row in torch_rows:
            if isinstance(row, dict) and "uuid" in row:
                row["uuid"] = canonicalize_gpu_uuid(row["uuid"])
    smi_rows = normalized.get("nvidia_smi")
    if isinstance(smi_rows, list):
        for index, raw in enumerate(smi_rows):
            if not isinstance(raw, str):
                continue
            pieces = [part.strip() for part in raw.split(",", 4)]
            if len(pieces) != 5:
                continue
            pieces[2] = canonicalize_gpu_uuid(pieces[2])
            smi_rows[index] = ", ".join(pieces)
    return normalized


def _label_parts(label: str) -> tuple[str, str | None]:
    if not isinstance(label, str):
        return str(label), None
    for phase in ("start", "closure", "cost"):
        suffix = f" {phase}"
        if label.endswith(suffix):
            return label[: -len(suffix)], phase
    return label, None


def _parse_scoped_smi_rows(
    rows: Any,
    *,
    label: str,
    need: Callable[[bool, str], None],
) -> dict[str, dict[str, str]]:
    need(isinstance(rows, list) and rows, f"scoped runtime nvidia-smi inventory missing for {label}")
    parsed: dict[str, dict[str, str]] = {}
    for position, raw in enumerate(rows):
        need(isinstance(raw, str), f"scoped runtime nvidia-smi row malformed for {label}: {position}")
        pieces = [part.strip() for part in raw.split(",", 3)]
        need(
            len(pieces) == 4 and all(pieces),
            f"scoped runtime nvidia-smi row malformed for {label}: {position}",
        )
        physical_index, name, uuid, driver = pieces
        need(
            physical_index.isdigit() and int(physical_index) >= 0,
            f"scoped runtime physical GPU index invalid for {label}: {physical_index}",
        )
        need(physical_index not in parsed, f"scoped runtime duplicate physical GPU for {label}")
        parsed[physical_index] = {
            "index": physical_index,
            "name": name,
            "uuid": canonicalize_gpu_uuid(uuid),
            "driver": driver,
        }
    return parsed


class RuntimeCompatibilityAdapter:
    """Adapter installed only over the frozen runtime-normalization function."""

    def __init__(
        self,
        original_normalize: Callable[[Any, str], dict[str, Any]],
        need: Callable[[bool, str], None],
    ) -> None:
        self._original_normalize = original_normalize
        self._need = need
        self._starts: dict[str, dict[str, Any]] = {}

    def normalize(self, environment: Any, label: str) -> dict[str, Any]:
        base_label, phase = _label_parts(label)
        is_scoped = isinstance(environment, dict) and (
            "torch_visible_devices" in environment or "nvidia_smi_gpus" in environment
        )
        if is_scoped:
            self._need(phase == "cost", f"scoped runtime schema is permitted only for cost evidence: {label}")
            self._need(base_label in self._starts, f"scoped runtime has no validated start evidence: {label}")
            return self._validate_scoped_cost(
                environment=environment,
                start=self._starts[base_label],
                label=label,
            )

        canonical_environment = canonicalize_full_runtime_environment(environment)
        normalized = self._original_normalize(canonical_environment, label)
        if phase == "start":
            self._starts[base_label] = {
                "environment": canonical_environment,
                "normalized": copy.deepcopy(normalized),
            }
        return normalized

    def _validate_scoped_cost(
        self,
        *,
        environment: dict[str, Any],
        start: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        need = self._need
        start_environment = start["environment"]
        start_normalized = start["normalized"]

        need(
            "torch_gpus" not in environment and "nvidia_smi" not in environment,
            f"scoped runtime mixes full and scoped GPU schemas for {label}",
        )
        need(
            environment.get("hostname") == start_normalized["hostname"],
            f"scoped runtime hostname mismatch for {label}",
        )
        need(
            environment.get("pytorch") == start_normalized["pytorch"],
            f"scoped runtime PyTorch mismatch for {label}",
        )
        need(
            environment.get("pytorch_cuda") == start_normalized["pytorch_cuda"],
            f"scoped runtime PyTorch CUDA mismatch for {label}",
        )
        need(
            isinstance(environment.get("cuda_available"), bool)
            and environment.get("cuda_available") == start_environment.get("cuda_available"),
            f"scoped runtime CUDA availability mismatch for {label}",
        )

        visible_raw = environment.get("cuda_visible_devices")
        need(isinstance(visible_raw, str) and visible_raw, f"scoped runtime CUDA device missing for {label}")
        visible_parts = [part.strip() for part in visible_raw.split(",")]
        need(
            len(visible_parts) == 1 and visible_parts[0].isdigit(),
            f"scoped runtime must bind exactly one physical CUDA device for {label}",
        )
        physical_index = visible_parts[0]

        scoped_smi = _parse_scoped_smi_rows(
            environment.get("nvidia_smi_gpus"),
            label=label,
            need=need,
        )
        start_smi = {row["index"]: row for row in start_normalized["nvidia_smi"]}
        need(
            set(scoped_smi) == set(start_smi),
            f"scoped runtime physical GPU inventory mismatch for {label}",
        )
        for index, scoped_row in scoped_smi.items():
            full_row = start_smi[index]
            need(
                (
                    scoped_row["name"],
                    scoped_row["uuid"],
                    scoped_row["driver"],
                )
                == (
                    full_row["name"],
                    full_row["uuid"],
                    full_row["driver"],
                ),
                f"scoped runtime physical GPU row mismatch for {label}: {index}",
            )
        need(physical_index in scoped_smi, f"scoped runtime visible physical GPU absent for {label}")
        visible_smi = scoped_smi[physical_index]

        visible_torch = environment.get("torch_visible_devices")
        need(
            isinstance(visible_torch, list) and len(visible_torch) == 1,
            f"scoped runtime must contain exactly one visible Torch GPU for {label}",
        )
        torch_row = visible_torch[0]
        need(isinstance(torch_row, dict), f"scoped runtime visible Torch GPU malformed for {label}")
        need(torch_row.get("logical_index") == 0, f"scoped runtime Torch logical index mismatch for {label}")

        start_torch_matches = [
            row for row in start_normalized["torch_gpus"] if row["uuid"] == visible_smi["uuid"]
        ]
        need(
            len(start_torch_matches) == 1,
            f"scoped runtime visible GPU does not map uniquely to start evidence for {label}",
        )
        start_torch = start_torch_matches[0]
        need(
            start_torch["name"] == visible_smi["name"],
            f"scoped runtime start Torch/nvidia GPU name mismatch for {label}",
        )
        need(
            torch_row.get("name") == start_torch["name"],
            f"scoped runtime visible GPU name mismatch for {label}",
        )
        need(
            torch_row.get("total_memory_bytes") == start_torch["total_memory_bytes"],
            f"scoped runtime visible GPU memory mismatch for {label}",
        )
        if "uuid" in torch_row:
            need(
                canonicalize_gpu_uuid(torch_row["uuid"]) == visible_smi["uuid"],
                f"scoped runtime visible Torch UUID mismatch for {label}",
            )
        if "physical_index" in torch_row:
            need(
                str(torch_row["physical_index"]) == physical_index,
                f"scoped runtime visible Torch physical index mismatch for {label}",
            )

        # The original finalizer still performs its equality check.  Returning the
        # validated start representation records that the scoped receipt describes
        # the same device without pretending that the two JSON schemas are equal.
        return copy.deepcopy(start_normalized)


def load_frozen_finalizer() -> Any:
    observed = sha256_file(FROZEN_FINALIZER_PATH)
    if observed != FROZEN_FINALIZER_SHA256:
        raise RuntimeCompatibilityError(
            "frozen C1 finalizer hash drift: "
            f"expected={FROZEN_FINALIZER_SHA256} observed={observed}"
        )
    module_name = "c1_frozen_finalizer_runtime_compat"
    spec = importlib.util.spec_from_file_location(module_name, FROZEN_FINALIZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeCompatibilityError(f"cannot load frozen finalizer: {FROZEN_FINALIZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_runtime_compat(finalizer: Any) -> RuntimeCompatibilityAdapter:
    existing = getattr(finalizer, "_c1_runtime_compat_adapter", None)
    if existing is not None:
        return existing
    adapter = RuntimeCompatibilityAdapter(finalizer.normalize_runtime_environment, finalizer.need)
    finalizer.normalize_runtime_environment = adapter.normalize
    finalizer._c1_runtime_compat_adapter = adapter
    return adapter


def main() -> int:
    try:
        finalizer = load_frozen_finalizer()
    except (OSError, RuntimeCompatibilityError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    install_runtime_compat(finalizer)
    return int(finalizer.main())


if __name__ == "__main__":
    raise SystemExit(main())
