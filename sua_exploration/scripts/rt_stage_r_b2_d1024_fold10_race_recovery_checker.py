#!/usr/bin/env python3
"""Receipt-only integrity checker for the isolated RT Stage-R fold-10 retry.

Fold 10 of the original folds-03--14 supervisor was terminalized while its
source fit was still running.  That terminal is historical evidence, not a
restart point.  This module validates a *new*, one-shot recovery cell without
opening RT data, importing a training stack, or launching a subprocess.

It is intentionally usable by the paired aggregate: the aggregate may accept
fold 10 only when this checker binds the fresh cell, its selected checkpoint,
and every source/outer receipt to one immutable recovery terminal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping


WORKSPACE = Path(__file__).resolve().parents[2]
OLD_SUPERVISOR_ROOT = (
    WORKSPACE
    / "streaming_calibration_exp/outputs/rt_stage_r_b2_local3090/supervisor_folds03_14_v1"
)
DEFAULT_RECOVERY_ROOT = (
    WORKSPACE
    / "streaming_calibration_exp/outputs/rt_stage_r_b2_local3090/fold10_race_recovery_v1"
)
FOLD = 10
SEED = 42
ARM = "zero4"
OLD_TERMINAL_SCHEMA = "rt_stage_r_b2_d1024_fold_terminal_v2"
OLD_TERMINAL_STATUS = "STOP_PARTIAL_FIT_WITHOUT_SELECTION"
RECOVERY_TERMINAL_SCHEMA = "rt_stage_r_b2_d1024_fold10_race_recovery_terminal_v1"
RECOVERY_TERMINAL_STATUS = "PASS_FRESH_CPU_ONE_SHOT_OUTER_EVAL"


class RecoveryError(RuntimeError):
    """Raised when the failed cell or the isolated retry cannot be bound."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"required receipt is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(value, dict), f"receipt must be a JSON object: {path}")
    return value


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"{label} is missing: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} must be immutable mode 0444: {path}")


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _resolved_child(path: Path, root: Path, *, label: str) -> Path:
    resolved, resolved_root = path.resolve(), root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise RecoveryError(f"{label} escapes isolated recovery root: {path}") from error
    return resolved


def old_failure_terminal(*, supervisor_root: Path = OLD_SUPERVISOR_ROOT) -> Path:
    """The one historical partial-fit terminal that recovery must preserve."""

    return (
        supervisor_root
        / "cells/b2_d1024_zero4/fold_10/seed_42/cell_terminal.json"
    )


def recovery_paths(recovery_root: Path) -> dict[str, Path]:
    """Return the only legal location of the fresh fold-10 cell."""

    cell = recovery_root.resolve() / "cells/b2_d1024_zero4/fold_10/seed_42"
    fit = cell / "fit"
    return {
        "root": recovery_root.resolve(),
        "cell": cell,
        "launch": cell / "recovery_launch.json",
        "preflight": cell / "constructibility_preflight.json",
        "log": cell / "recovery.log",
        "fit": fit,
        "selection": fit / "rt_nested_selection_receipt.json",
        "config": fit / ".hydra/config.yaml",
        "split": fit / "split_manifest.json",
        "outer": cell / "outer_target_eval.json",
        "terminal": cell / "recovery_terminal.json",
    }

def validate_old_failure(*, terminal_path: Path = old_failure_terminal()) -> dict[str, Any]:
    """Validate and bind the immutable non-restartable historical evidence."""

    _immutable(terminal_path, "historical fold-10 partial-fit terminal")
    receipt = _json(terminal_path)
    required = {
        "schema": OLD_TERMINAL_SCHEMA,
        "status": OLD_TERMINAL_STATUS,
        "fold": FOLD,
        "seed": SEED,
        "arm": ARM,
        "formal_heldout_opened": False,
    }
    for field, expected in required.items():
        _need(receipt.get(field) == expected, f"historical fold-10 receipt drift: {field}")
    return {"path": str(terminal_path.resolve()), "sha256": _sha256(terminal_path)}


def _validate_file_binding(
    files: Mapping[str, Any], *, name: str, expected: Path, recovery_root: Path
) -> dict[str, Any]:
    entry = files.get(name)
    _need(isinstance(entry, Mapping), f"recovery terminal lacks {name} binding")
    _need(entry.get("path") == str(expected.resolve()), f"recovery terminal {name} path drift")
    _need(_is_sha(entry.get("sha256")), f"recovery terminal {name} SHA malformed")
    _resolved_child(expected, recovery_root, label=f"{name} receipt")
    _need(expected.is_file(), f"recovery {name} receipt is missing: {expected}")
    _need(entry.get("sha256") == _sha256(expected), f"recovery terminal {name} SHA mismatch")
    return dict(entry)


def validate_recovery_terminal(
    *, recovery_root: Path = DEFAULT_RECOVERY_ROOT,
    terminal_path: Path | None = None,
    old_terminal_path: Path = old_failure_terminal(),
) -> dict[str, Any]:
    """Fail closed unless fresh fit, config, selection, and outer receipts agree.

    This is deliberately narrower than the full semantic RT validator.  The
    aggregate performs the latter.  Here we establish retry lineage and ensure
    no stale checkpoint or source file from the partial cell can enter it.
    """

    paths = recovery_paths(recovery_root)
    terminal_path = paths["terminal"] if terminal_path is None else terminal_path.resolve()
    _need(terminal_path == paths["terminal"], "recovery terminal is not at the unique fresh-cell path")
    _immutable(terminal_path, "fold-10 recovery terminal")
    terminal = _json(terminal_path)
    required = {
        "schema": RECOVERY_TERMINAL_SCHEMA,
        "status": RECOVERY_TERMINAL_STATUS,
        "fold": FOLD,
        "seed": SEED,
        "arm": ARM,
        "formal_heldout_opened": False,
    }
    for field, expected in required.items():
        _need(terminal.get(field) == expected, f"recovery terminal drift: {field}")

    historical = validate_old_failure(terminal_path=old_terminal_path)
    prior = terminal.get("preserved_old_failure")
    _need(isinstance(prior, Mapping), "recovery terminal omits preserved old failure")
    _need(dict(prior) == historical, "recovery terminal does not bind the historical failure hash")

    freshness = terminal.get("freshness")
    _need(isinstance(freshness, Mapping), "recovery terminal lacks fresh-run audit")
    expected_freshness = {
        "recovery_root": str(paths["root"]),
        "cell": str(paths["cell"]),
        "old_cell_reused": False,
        "warmstart_forbidden": True,
        "configured_ckpt_path": None,
        "recovery_fit_attempts": 1,
    }
    for field, expected in expected_freshness.items():
        _need(freshness.get(field) == expected, f"recovery freshness audit drift: {field}")

    files = terminal.get("files")
    _need(isinstance(files, Mapping), "recovery terminal lacks file bindings")
    bindings = {
        name: _validate_file_binding(files, name=name, expected=paths[name], recovery_root=paths["root"])
        for name in ("preflight", "selection", "config", "split", "outer")
    }
    selection = _json(paths["selection"])
    _need(selection.get("config_sha256") == bindings["config"]["sha256"], "recovery selection config binding drift")
    _need(selection.get("split_manifest_sha256") == bindings["split"]["sha256"], "recovery selection split binding drift")
    checkpoint_text = selection.get("best_model_path")
    _need(isinstance(checkpoint_text, str) and checkpoint_text, "recovery selection lacks selected checkpoint path")
    checkpoint = Path(checkpoint_text).resolve()
    _resolved_child(checkpoint, paths["fit"], label="selected checkpoint")
    _need(checkpoint.is_file(), "recovery selected checkpoint is missing")
    _need(selection.get("best_model_sha256") == _sha256(checkpoint), "recovery selected checkpoint SHA mismatch")
    binding = _validate_file_binding(files, name="selected_checkpoint", expected=checkpoint, recovery_root=paths["root"])
    _need(binding["sha256"] == selection.get("best_model_sha256"), "recovery terminal checkpoint lineage drift")
    outer = _json(paths["outer"])
    _need(outer.get("checkpoint_sha256") == selection.get("best_model_sha256"), "recovery outer receipt uses a non-selected checkpoint")

    return {
        "terminal": terminal,
        "terminal_path": str(terminal_path),
        "terminal_sha256": _sha256(terminal_path),
        "old_failure": historical,
        "files": {**bindings, "selected_checkpoint": binding},
    }
