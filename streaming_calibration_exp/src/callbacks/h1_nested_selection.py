"""Checkpoint-selection contract for the isolated H1 clean nested-LOSO path.

The ordinary Falcon module logs one R2 metric for every held-in session.  A
fit fold, however, exposes exactly one inner-validation session; all other
per-session metrics remain empty and the aggregate mean can become ``-inf``.
This module binds early stopping and best-checkpoint callbacks to that one
non-empty key and rejects aggregate/outer/held-out monitors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from lightning.pytorch import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


_SESSION_RE = re.compile(r"^ses-[A-Za-z0-9T_-]+$")


def h1_inner_validation_monitor(session_name: str) -> str:
    """Return the exact single-session validation metric for H1 selection."""

    session = str(session_name)
    if not _SESSION_RE.fullmatch(session):
        raise ValueError(f"invalid H1 inner-validation session name: {session_name!r}")
    return f"val_heldin_{session}/r2"


def validate_h1_checkpoint_monitor(
    monitor: str,
    *,
    inner_validation_session: str,
    outer_target_session: str | None = None,
) -> str:
    """Fail closed unless ``monitor`` is exactly the inner-val session key."""

    expected = h1_inner_validation_monitor(inner_validation_session)
    observed = str(monitor)
    if observed != expected:
        raise ValueError(
            "H1 clean nested-LOSO checkpoint monitor must be the exact inner-validation "
            f"session key {expected!r}, got {observed!r}"
        )
    if outer_target_session is not None and str(outer_target_session) in observed:
        raise ValueError("H1 checkpoint monitor names the outer target")
    if "heldout" in observed or observed in {"val_heldin/r2_mean", "val_heldin/r2_std"}:
        raise ValueError("H1 checkpoint monitor may not be an aggregate/held-out metric")
    return expected


@dataclass(frozen=True)
class H1SelectionContract:
    """JSON-safe selection metadata shared by teacher and all students."""

    arm: str
    role: str
    inner_validation_session: str
    outer_target_session: str

    @property
    def monitor(self) -> str:
        return h1_inner_validation_monitor(self.inner_validation_session)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": str(self.arm),
            "role": str(self.role),
            "monitor": self.monitor,
            "selection_scope": "inner_validation_session_only",
            "inner_validation_session": str(self.inner_validation_session),
            "outer_target_session": str(self.outer_target_session),
            "outer_target_used": False,
        }


def build_h1_selection_contract(split: Any, *, arm: str, role: str) -> H1SelectionContract:
    """Build and validate a selection contract from a nested-LOSO split."""

    contract = H1SelectionContract(
        arm=str(arm),
        role=str(role),
        inner_validation_session=str(split.inner_validation_session),
        outer_target_session=str(split.outer_target_session),
    )
    validate_h1_checkpoint_monitor(
        contract.monitor,
        inner_validation_session=contract.inner_validation_session,
        outer_target_session=contract.outer_target_session,
    )
    return contract


def audit_h1_teacher_student_monitors(
    contracts: Iterable[Mapping[str, Any]], *, inner_validation_session: str, outer_target_session: str
) -> dict[str, Any]:
    """Require teacher/Full/B4/Zero all select on one inner-val key."""

    required_roles = {"teacher", "afc4_h1q3", "afc4_h1_b4", "zero4"}
    rows = [dict(row) for row in contracts]
    observed_roles = {str(row.get("role", row.get("arm", ""))) for row in rows}
    if not required_roles.issubset(observed_roles):
        raise ValueError(f"H1 teacher/student monitor audit missing roles: {sorted(required_roles - observed_roles)}")
    expected = h1_inner_validation_monitor(inner_validation_session)
    for row in rows:
        validate_h1_checkpoint_monitor(
            str(row.get("monitor", "")),
            inner_validation_session=inner_validation_session,
            outer_target_session=outer_target_session,
        )
        if str(row.get("selection_scope", "")) != "inner_validation_session_only":
            raise ValueError("H1 monitor audit found a non-inner selection scope")
        if bool(row.get("outer_target_used", True)):
            raise ValueError("H1 monitor audit found outer-target checkpoint selection")
    return {
        "status": "PASS_H1_INNER_VALIDATION_MONITOR_AUDIT",
        "monitor": expected,
        "selection_scope": "inner_validation_session_only",
        "outer_target_used": False,
        "roles": sorted(observed_roles & required_roles),
    }


class H1NestedSelectionMonitor(Callback):
    """Bind configured checkpoint/early-stop callbacks to the fold's key.

    The callback is intentionally strict: callbacks must either leave
    ``monitor`` unset or already contain the exact key.  It never silently
    rewrites an aggregate monitor, which would hide a bad composed config.
    """

    def __init__(
        self,
        *,
        require_callback_types: bool = True,
        manifest_path: str | None = None,
        role: str = "student",
    ) -> None:
        super().__init__()
        self.require_callback_types = bool(require_callback_types)
        self.manifest_path = None if manifest_path in (None, "", "null") else Path(str(manifest_path))
        self.role = str(role)
        self.monitor: str | None = None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        datamodule = getattr(trainer, "datamodule", None)
        split = getattr(datamodule, "split", None)
        if split is None:
            raise RuntimeError("H1 nested selection callback requires a prepared H1 split")
        expected = h1_inner_validation_monitor(split.inner_validation_session)
        validate_h1_checkpoint_monitor(
            expected,
            inner_validation_session=split.inner_validation_session,
            outer_target_session=split.outer_target_session,
        )
        seen = 0
        for callback in getattr(trainer, "callbacks", ()):  # pragma: no branch - tiny callback list
            if not isinstance(callback, (ModelCheckpoint, EarlyStopping)):
                continue
            seen += 1
            observed = getattr(callback, "monitor", None)
            if observed is None:
                callback.monitor = expected
            else:
                validate_h1_checkpoint_monitor(
                    str(observed),
                    inner_validation_session=split.inner_validation_session,
                    outer_target_session=split.outer_target_session,
                )
        if self.require_callback_types and seen == 0:
            raise RuntimeError("H1 nested selection callback found no checkpoint/early-stopping callback")
        self.monitor = expected
        if self.manifest_path is not None:
            if not hasattr(datamodule, "get_split_manifest"):
                raise RuntimeError("H1 selection manifest requires a manifest-producing DataModule")
            manifest = dict(datamodule.get_split_manifest())
            configured_teacher_sha = manifest.get("teacher_checkpoint_sha256")
            teacher_path = getattr(pl_module, "_teacher_ckpt_path", None)
            actual_teacher_sha = None
            if teacher_path not in (None, "", "null") and Path(str(teacher_path)).is_file():
                actual_teacher_sha = self._sha256(Path(str(teacher_path)))
            if configured_teacher_sha not in (None, "", "null") and actual_teacher_sha is not None:
                if str(configured_teacher_sha) != actual_teacher_sha:
                    raise ValueError("H1 configured teacher checkpoint SHA disagrees with selected teacher path")
            manifest["teacher_checkpoint_sha256"] = actual_teacher_sha or configured_teacher_sha
            manifest["selection_contract"] = H1SelectionContract(
                arm=str(manifest.get("arm", "")),
                role=self.role,
                inner_validation_session=str(split.inner_validation_session),
                outer_target_session=str(split.outer_target_session),
            ).as_dict()
            manifest["checkpoint_selection"] = {
                **dict(manifest.get("checkpoint_selection", {})),
                "monitor": expected,
                "scope": "inner_validation_session_only",
                "outer_target_used": False,
            }
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if self.manifest_path.exists():
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if existing.get("source_plan_sha256") != manifest.get("source_plan_sha256"):
                    raise ValueError("H1 pre-optimizer manifest source-plan SHA changed before fit")
                if existing.get("checkpoint_selection", {}).get("monitor") != expected:
                    raise ValueError("H1 pre-optimizer manifest monitor is not the exact inner-val key")
            else:
                self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
