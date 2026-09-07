"""Phase-3 2x2x2 readout for the 50-epoch pair at epoch 50 and the selected epoch.

Scorers, session opening, and table aggregation are imported verbatim from the
sealed 20-epoch ``phase3``.  This module's own arm-binding loader does **not**
pin that lane's sealed terminal digests or root paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as base_physical
from tfpd_exploration.src.m1_t0c1_prefix_v1.phase3 import (
    build_table as predecessor_build_table,
    open_session_dataset,
    score_cdm_fifo,
    score_static,
)

from . import plan


class Phase3Error(RuntimeError):
    """Fail closed for the 50-epoch 2x2x2 readout."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase3Error(message)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ArmBinding50:
    """This lane's arm binding: runtime-verified, not pinned to 20-epoch digests."""

    arm: str
    root_relative: str
    terminal_sha256: str
    checkpoints: Mapping[int, Mapping[str, object]]
    best_source_train_loss: Mapping[str, object]


def load_arm_binding(root: Path, arm: str) -> ArmBinding50:
    """Descriptor-read this lane's completed arm terminal/manifest (runtime digest)."""
    _require(arm in plan.ARMS, "phase3 arm drift")
    directory = Path(root).absolute() / plan.ARM_ROOT_RELATIVE[arm]
    _require(directory.is_dir(), f"phase3 arm root absent: {arm}")
    for name in ("checkpoint_manifest.json", "terminal.json"):
        body = (directory / name).read_bytes()
        digest = _sha(body)
        sidecar = (directory / f"{name}.sha256").read_text(encoding="ascii")
        _require(sidecar == f"{digest}  {name}\n", f"phase3 arm sidecar drift: {arm}/{name}")
    terminal_digest = _sha((directory / "terminal.json").read_bytes())
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    terminal = json.loads((directory / "terminal.json").read_text(encoding="utf-8"))
    _require(terminal.get("status") == plan.THIS_LANE_ARM_STATUS
             and manifest.get("arm") == arm
             and manifest.get("swa_enabled") is False
             and manifest.get("swa_artifact_forbidden") is True,
             f"phase3 arm terminal semantics drift: {arm}")
    raw_checkpoints = manifest.get("checkpoints")
    _require(isinstance(raw_checkpoints, Mapping), f"phase3 checkpoint manifest topology drift: {arm}")
    best = raw_checkpoints.get("best_source_train_loss")
    _require(isinstance(best, Mapping), f"phase3 best-train-loss checkpoint absent: {arm}")
    epoch_checkpoints: dict[int, Mapping[str, object]] = {}
    for epoch_index in plan.CHECKPOINT_EPOCH_INDICES:
        entry = None
        for value in raw_checkpoints.values():
            if isinstance(value, Mapping) and int(value.get("epoch_index_0based", -1)) == epoch_index:
                entry = value
                break
        _require(isinstance(entry, Mapping), f"phase3 epoch checkpoint {epoch_index} absent: {arm}")
        expected_name = plan.epoch_checkpoint_filename(epoch_index)
        _require(entry.get("filename") == expected_name and entry.get("strict_reload") is True,
                 f"phase3 epoch checkpoint filename/strict_reload drift: {arm}/{epoch_index}")
        epoch_checkpoints[epoch_index] = dict(entry)
    return ArmBinding50(
        arm=arm,
        root_relative=plan.ARM_ROOT_RELATIVE[arm],
        terminal_sha256=terminal_digest,
        checkpoints=epoch_checkpoints,
        best_source_train_loss=dict(best),
    )


def strict_load_epoch_checkpoint(
    root: Path, root_relative: str, filename: str, expected_state_sha256: str, device: str,
) -> tuple[Any, str]:
    """Strict reload of one epoch checkpoint; filename is not hardcoded to best-loss."""
    _require(isinstance(root_relative, str) and root_relative and isinstance(filename, str)
             and filename.endswith(".pt") and Path(filename).name == filename,
             "epoch checkpoint path drift")
    path = Path(root).absolute() / root_relative / filename
    body = path.read_bytes()
    sidecar = (path.parent / f"{filename}.sha256").read_text(encoding="ascii")
    digest = _sha(body)
    _require(sidecar == f"{digest}  {filename}\n",
             f"epoch checkpoint sidecar drift: {root_relative}/{filename}")
    captured: list[Any] = []

    def factory() -> Any:
        model = base_physical.load_exact_m1_spint_model(Path(root)).to(device)
        base_physical.materialize_exact_m1_model(model, device=device)
        captured.append(model)
        return model

    observed = base_physical.strict_reload_checkpoint_bytes(
        body, expected_state_sha256=expected_state_sha256,
        model_factory=factory, device=device,
    )
    _require(len(captured) == 1 and observed == expected_state_sha256,
             f"phase3 strict epoch checkpoint reload drift: {filename}")
    model = captured[0]
    model.eval()
    _require(model.training is False, "phase3 epoch model must be in eval mode")
    return model, observed


def stamp_table(
    table: Mapping[str, object],
    *,
    epoch_label: str,
    alias_of_epoch50: bool,
    selection_surface: str | None = None,
    selected_by_arm: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Overwrite predecessor phase/schema stamps so the table belongs to this lane."""
    stamped = dict(table)
    stamped["schema"] = "m1_t0c1_prefix_v1_50ep_phase3_table_v1"
    stamped["phase"] = plan.PHASE
    stamped["epoch_label"] = epoch_label
    stamped["alias_of_epoch50"] = bool(alias_of_epoch50)
    stamped["cross_recipe_note"] = plan.CROSS_RECIPE_NOTE
    if selection_surface is not None:
        stamped["selection_surface"] = selection_surface
    if selected_by_arm is not None:
        stamped["selected_epoch_index_by_arm"] = dict(selected_by_arm)
    return stamped


def build_table(cells: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Predecessor aggregation, then this lane's phase stamp."""
    return stamp_table(predecessor_build_table(cells), epoch_label="unspecified", alias_of_epoch50=False)


def reference_delta(table: Mapping[str, object]) -> dict[str, object]:
    """Cross-recipe delta of one 2x2x2 table against spec §5.1.

    Every delta is labelled ``cross_recipe: True``.  The 20-epoch line used
    constant ``lr=1e-5`` with no scheduler.
    """
    _require(isinstance(table, Mapping) and isinstance(table.get("rows"), list),
             "reference_delta needs a table with rows")
    entries: list[dict[str, object]] = []
    for row in table["rows"]:  # type: ignore[union-attr]
        arm = str(row["arm"])
        deployment = str(row["deployment"])
        surface = str(row["surface"])
        this_mean = float(row["equal_session_mean"])
        try:
            reference = float(plan.REFERENCE_20EP_EQUAL_SESSION_MEAN[arm][deployment][surface])
        except KeyError as error:
            raise Phase3Error(f"reference_delta missing spec §5.1 cell: {arm}/{deployment}/{surface}") from error
        entries.append({
            "arm": arm,
            "deployment": deployment,
            "surface": surface,
            "this_lane_equal_session_mean": this_mean,
            "reference_20ep_equal_session_mean": reference,
            "delta_this_minus_20ep": this_mean - reference,
            "cross_recipe": True,
            "peak_lr": plan.LR_WARMUP_END,
            "schedule": plan.LR_SCHEDULE_KIND,
            "note": plan.CROSS_RECIPE_NOTE,
        })
    _require(len(entries) == 8, "reference_delta expected the 2x2x2 = 8 cells")
    _require(all(entry["cross_recipe"] is True for entry in entries),
             "reference_delta lost the cross_recipe label")
    return {
        "schema": "m1_t0c1_prefix_v1_50ep_reference_delta_v1",
        "cross_recipe": True,
        "note": plan.CROSS_RECIPE_NOTE,
        "reference_20ep_arm_terminal_sha256": dict(plan.PREDECESSOR_20EP_ARM_TERMINAL_SHA256),
        "reference_20ep_phase3_terminal_sha256": plan.PREDECESSOR_20EP_PHASE3_TERMINAL_SHA256,
        "deltas": entries,
    }


def _score_grid(
    models: Mapping[str, Any],
    opened: Mapping[str, Any],
    *,
    device: str,
    extra: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    cells: dict[str, dict[str, object]] = {}
    for arm in plan.ARMS:
        for deployment, scorer in (("static_m10", score_static),
                                   ("cdm_activity_fifo_m10", score_cdm_fifo)):
            for session_id in plan.SCORE_ORDER:
                cell = dict(scorer(models[arm], opened[session_id], device=device))
                key = f"{arm}_{deployment}_{session_id}"
                cell.update({
                    "schema": "m1_t0c1_prefix_v1_50ep_phase3_cell_v1",
                    "arm": arm, "deployment": deployment, "session_id": session_id,
                    **dict(extra),
                })
                cells[key] = cell
    return cells


def run_phase3(
    root: Path, *, source_root: Path, device: str, selected_by_arm: Mapping[str, int],
) -> dict[str, object]:
    """Rebuild the 2x2x2 at epoch 50 and at the selected epoch.  Sessions opened once."""
    _require(set(selected_by_arm) == set(plan.ARMS)
             and all(int(selected_by_arm[arm]) in plan.CHECKPOINT_EPOCH_INDICES for arm in plan.ARMS),
             "phase3 selected-epoch map drifted")
    selected = {arm: int(selected_by_arm[arm]) for arm in plan.ARMS}
    bindings = {arm: load_arm_binding(Path(root), arm) for arm in plan.ARMS}
    opened = {
        session_id: open_session_dataset(Path(root), Path(source_root), session_id)
        for session_id in plan.SCORE_ORDER
    }
    models_epoch50: dict[str, Any] = {}
    state_epoch50: dict[str, str] = {}
    for arm in plan.ARMS:
        binding = bindings[arm]
        filename = plan.epoch_checkpoint_filename(plan.EPOCH_50_INDEX)
        expected = str(binding.checkpoints[plan.EPOCH_50_INDEX]["state_sha256"])
        model, state_sha = strict_load_epoch_checkpoint(
            Path(root), binding.root_relative, filename, expected, device,
        )
        models_epoch50[arm] = model
        state_epoch50[arm] = state_sha
    cells_epoch50 = _score_grid(
        models_epoch50, opened, device=device,
        extra={"epoch_index": plan.EPOCH_50_INDEX, "epoch_label": "epoch50",
               "arm_checkpoint_state_sha256_by_arm": dict(state_epoch50)},
    )
    # Per-cell extra is shared; stamp the per-arm state on each cell.
    for arm in plan.ARMS:
        for deployment in plan.PHASE3_DEPLOYMENTS:
            for session_id in plan.SCORE_ORDER:
                key = f"{arm}_{deployment}_{session_id}"
                cells_epoch50[key]["arm_checkpoint_state_sha256"] = state_epoch50[arm]
                cells_epoch50[key]["arm_terminal_sha256"] = bindings[arm].terminal_sha256
    table_epoch50 = stamp_table(
        predecessor_build_table(cells_epoch50), epoch_label="epoch50", alias_of_epoch50=False,
    )
    both_select_50 = all(selected[arm] == plan.EPOCH_50_INDEX for arm in plan.ARMS)
    if both_select_50:
        cells_selected = {
            key: dict(cell, epoch_label="selected", alias_of_epoch50=True)
            for key, cell in cells_epoch50.items()
        }
        table_selected = stamp_table(
            predecessor_build_table(cells_epoch50), epoch_label="selected", alias_of_epoch50=True,
            selection_surface=plan.SELECTION_SURFACE, selected_by_arm=selected,
        )
    else:
        models_selected: dict[str, Any] = {}
        state_selected: dict[str, str] = {}
        for arm in plan.ARMS:
            binding = bindings[arm]
            epoch_index = selected[arm]
            filename = plan.epoch_checkpoint_filename(epoch_index)
            expected = str(binding.checkpoints[epoch_index]["state_sha256"])
            model, state_sha = strict_load_epoch_checkpoint(
                Path(root), binding.root_relative, filename, expected, device,
            )
            models_selected[arm] = model
            state_selected[arm] = state_sha
        cells_selected = _score_grid(
            models_selected, opened, device=device,
            extra={"epoch_label": "selected", "alias_of_epoch50": False},
        )
        for arm in plan.ARMS:
            for deployment in plan.PHASE3_DEPLOYMENTS:
                for session_id in plan.SCORE_ORDER:
                    key = f"{arm}_{deployment}_{session_id}"
                    cells_selected[key]["epoch_index"] = selected[arm]
                    cells_selected[key]["arm_checkpoint_state_sha256"] = state_selected[arm]
                    cells_selected[key]["arm_terminal_sha256"] = bindings[arm].terminal_sha256
        table_selected = stamp_table(
            predecessor_build_table(cells_selected), epoch_label="selected", alias_of_epoch50=False,
            selection_surface=plan.SELECTION_SURFACE, selected_by_arm=selected,
        )
    delta = reference_delta(table_epoch50)
    return {
        "cells_epoch50": cells_epoch50,
        "cells_selected": cells_selected,
        "table_epoch50": table_epoch50,
        "table_selected": table_selected,
        "reference_delta": delta,
        "selected_by_arm": dict(selected),
        "selection_surface": plan.SELECTION_SURFACE,
        "alias_of_epoch50": both_select_50,
    }


__all__ = (
    "Phase3Error", "ArmBinding50", "load_arm_binding", "strict_load_epoch_checkpoint",
    "open_session_dataset", "score_static", "score_cdm_fifo", "build_table",
    "reference_delta", "stamp_table", "run_phase3",
)
