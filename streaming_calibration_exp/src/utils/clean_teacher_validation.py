"""Fail-closed validation for the M2 held-in-only clean teacher.

This module is intentionally used both by the external receipt finalizer and
by the streaming student before it is allowed to deserialize a teacher.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import torch


LEGACY_HELDOUT_SELECTED_TEACHER_SHA256 = (
    "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
)
PROTOCOL = "m2_clean_teacher_v1"
SELECTION_METRIC = "val_heldin/r2_mean"


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
  raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
  return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(f"clean teacher {name} is missing: {path}")
  try:
    parsed = json.loads(path.read_text())
  except json.JSONDecodeError as error:
    raise ValueError(f"clean teacher {name} is invalid JSON: {path}") from error
  if not isinstance(parsed, dict):
    raise ValueError(f"clean teacher {name} must be a JSON object")
  return parsed


def load_teacher_checkpoint(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(f"clean teacher checkpoint is missing: {path}")
  payload = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(payload, dict):
    raise ValueError("clean teacher checkpoint payload must be a mapping")
  return payload


def validate_selection_callbacks(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Require checkpointed selection to contain only held-in monitored callbacks."""
  callbacks = payload.get("callbacks")
  if not isinstance(callbacks, Mapping):
    raise ValueError("clean teacher checkpoint has no callback state")
  checkpoint_callbacks: list[Mapping[str, Any]] = []
  for name, state in callbacks.items():
    if not isinstance(state, Mapping):
      continue
    # Lightning persists ModelCheckpoint.monitor in its state but does not
    # persist ``mode``; EarlyStopping persists neither.  Its checkpoint key
    # contains both constructor fields, so read the serialized key rather than
    # treating a missing state field as permission to skip the selector check.
    callback_name = str(name)
    key_monitor = re.search(r"'monitor': '([^']+)'", callback_name)
    key_mode = re.search(r"'mode': '([^']+)'", callback_name)
    monitor = state.get("monitor", key_monitor.group(1) if key_monitor else None)
    mode = state.get("mode", key_mode.group(1) if key_mode else None)
    if monitor is None and mode is None:
      continue
    if monitor != SELECTION_METRIC or mode != "max":
      raise ValueError(f"clean teacher callback {name!r} is not held-in/max selected")
    if "ModelCheckpoint" in callback_name:
      checkpoint_callbacks.append(state)
  if len(checkpoint_callbacks) != 1:
    raise ValueError("clean teacher checkpoint must contain exactly one held-in ModelCheckpoint")
  return {
    "monitored_callback_count": sum(
      isinstance(state, Mapping) and state.get("monitor") is not None
      for state in callbacks.values()
    ),
    "model_checkpoint_count": len(checkpoint_callbacks),
    "selection_metric": SELECTION_METRIC,
    "selection_mode": "max",
  }


def validate_checkpoint_provenance(checkpoint_path: Path) -> dict[str, Any]:
  """Validate a clean checkpoint independently of a receipt and return facts."""
  checkpoint_path = checkpoint_path.resolve()
  checkpoint_sha256 = sha256_file(checkpoint_path)
  if checkpoint_sha256 == LEGACY_HELDOUT_SELECTED_TEACHER_SHA256:
    raise ValueError("legacy epoch_034 held-out-selected teacher is permanently forbidden")
  payload = load_teacher_checkpoint(checkpoint_path)
  provenance = payload.get("clean_teacher_provenance_v1")
  if not isinstance(provenance, dict):
    raise ValueError("teacher checkpoint lacks clean_teacher_provenance_v1")
  if provenance.get("protocol") != PROTOCOL:
    raise ValueError("teacher checkpoint has the wrong clean-teacher protocol")
  if provenance.get("task") != "m2" or provenance.get("calibration_n_trials") != 24:
    raise ValueError("teacher checkpoint is not M2/M24")
  if provenance.get("include_heldout_in_fit") is not False or provenance.get("include_heldout_in_test") is not False:
    raise ValueError("teacher checkpoint permits held-out data")
  if provenance.get("heldout_dataset_created") is not False:
    raise ValueError("teacher checkpoint recorded a held-out dataset")
  if provenance.get("selection_metric") != SELECTION_METRIC or provenance.get("selection_mode") != "max":
    raise ValueError("teacher checkpoint selection provenance is not held-in/max")
  runtime_manifest = provenance.get("runtime_manifest")
  if not isinstance(runtime_manifest, dict) or provenance.get("runtime_manifest_equals_external") is not True:
    raise ValueError("teacher checkpoint lacks an externally matched runtime manifest")
  if provenance.get("runtime_manifest_sha256") != canonical_json_sha256(runtime_manifest):
    raise ValueError("teacher checkpoint runtime manifest hash mismatch")
  manifest_path = Path(str(provenance.get("input_manifest_path", ""))).resolve()
  external_manifest = _load_json(manifest_path, name="input manifest")
  if external_manifest != runtime_manifest:
    raise ValueError("teacher checkpoint runtime manifest no longer matches external manifest")
  if provenance.get("input_manifest_sha256") != sha256_file(manifest_path):
    raise ValueError("teacher checkpoint external manifest SHA mismatch")
  files = runtime_manifest.get("files")
  if not isinstance(files, list) or len(files) != 14:
    raise ValueError("teacher checkpoint runtime manifest does not bind exactly fourteen inputs")
  expected_paths = sorted(str(Path(str(row.get("path", ""))).resolve()) for row in files if isinstance(row, dict))
  accessed = sorted(str(Path(path).resolve()) for path in provenance.get("accessed_input_paths", []))
  if len(expected_paths) != 14 or accessed != expected_paths:
    raise ValueError("teacher checkpoint did not access exactly the bound fourteen inputs")
  score = provenance.get("selection_score")
  if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
    raise ValueError("teacher checkpoint has no finite held-in selection score")
  return {
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_sha256": checkpoint_sha256,
    "checkpoint_epoch": int(provenance.get("epoch", payload.get("epoch", -1))),
    "selection_score": float(score),
    "provenance": provenance,
    "provenance_sha256": canonical_json_sha256(provenance),
    "callback_contract": validate_selection_callbacks(payload),
  }


def validate_clean_teacher_receipt(checkpoint_path: str | Path, receipt_path: str | Path) -> dict[str, Any]:
  """Return verified receipt facts or fail before the streaming student loads teacher weights."""
  checkpoint = validate_checkpoint_provenance(Path(checkpoint_path))
  receipt_path = Path(receipt_path).resolve()
  receipt = _load_json(receipt_path, name="receipt")
  if receipt.get("protocol") != PROTOCOL or receipt.get("schema_version") != 1:
    raise ValueError("clean teacher receipt protocol/schema mismatch")
  selected = receipt.get("selected_checkpoint")
  if not isinstance(selected, dict):
    raise ValueError("clean teacher receipt lacks selected checkpoint")
  if Path(str(selected.get("path", ""))).resolve() != Path(checkpoint["checkpoint_path"]):
    raise ValueError("clean teacher receipt checkpoint path mismatch")
  if selected.get("sha256") != checkpoint["checkpoint_sha256"]:
    raise ValueError("clean teacher receipt checkpoint SHA mismatch")
  if selected.get("epoch") != checkpoint["checkpoint_epoch"]:
    raise ValueError("clean teacher receipt epoch mismatch")
  if selected.get("selection_score") != checkpoint["selection_score"]:
    raise ValueError("clean teacher receipt selection score mismatch")
  if receipt.get("checkpoint_provenance_sha256") != checkpoint["provenance_sha256"]:
    raise ValueError("clean teacher receipt checkpoint provenance mismatch")
  if receipt.get("callback_contract") != checkpoint["callback_contract"]:
    raise ValueError("clean teacher receipt callback contract mismatch")
  manifest = receipt.get("input_manifest")
  if not isinstance(manifest, dict):
    raise ValueError("clean teacher receipt lacks input manifest binding")
  provenance = checkpoint["provenance"]
  if manifest.get("path") != provenance["input_manifest_path"] or manifest.get("sha256") != provenance["input_manifest_sha256"]:
    raise ValueError("clean teacher receipt input manifest binding mismatch")
  if receipt.get("resolved_config", {}).get("sha256") is None:
    raise ValueError("clean teacher receipt lacks resolved config hash")
  if not isinstance(receipt.get("source_manifest"), dict):
    raise ValueError("clean teacher receipt lacks source manifest")
  return {"receipt_path": str(receipt_path), "receipt": receipt, **checkpoint}
