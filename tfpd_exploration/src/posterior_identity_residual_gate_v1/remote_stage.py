"""Static staging description for a future PIRG review.

It intentionally has no transport implementation.  The source NWBs remain at
the audited strict-27 external root; a future stage may carry reviewed code,
five small source-adapter authorities, and the three explicit sealed Cell-D
initialization body+sidecar pairs only after a separate root audit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from . import plan
from . import train
from .train import IMPLEMENTATION_CLOSURE, implementation_closure


REMOTE_STAGE_ROOT = "/home/xinyuan/Work_host/posterior_identity_residual_gate_stage_v1"
REMOTE_RESULT_ROOT_RELATIVE = "tfpd_exploration/results/posterior_identity_residual_gate_v1"
REMOTE_SCORE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_identity_residual_gate_score_v1"


class PIRGStagePlanError(RuntimeError):
    """Raised when the future PIRG transport manifest drifts before launch."""


_INIT_BODY_ROLE = "immutable_sealed_cell_d_initialization_body"
_INIT_SIDECAR_ROLE = "immutable_sealed_cell_d_initialization_sidecar"


def _sealed_cell_d_initialization_stage_files() -> list[dict[str, object]]:
    """Build the exact six leaves needed by the frozen matched Cell-D loader."""
    result: list[dict[str, object]] = []
    for asset in plan.sealed_cell_d_init_assets_payload():
        result.extend((
            {
                "relative_path": asset["relative_path"],
                "sha256": asset["sha256"],
                "source_mode": asset["body_mode"],
                "destination_mode": "0444",
                "role": _INIT_BODY_ROLE,
            },
            {
                "relative_path": asset["sidecar_relative_path"],
                "sha256": asset["sidecar_sha256"],
                "contents": asset["sidecar_contents"],
                "source_mode": asset["sidecar_mode"],
                "destination_mode": "0444",
                "role": _INIT_SIDECAR_ROLE,
            },
        ))
    return result


def validate_remote_stage_plan(payload: Mapping[str, object]) -> dict[str, object]:
    """Fail closed if a future source stage omits or swaps a Cell-D init leaf.

    This intentionally validates only the fixed initialization transport
    subgraph.  The existing matched loader remains responsible for parsing the
    terminal, SWA, and governing baseline semantics once those exact bytes are
    present on the fresh stage.
    """
    if not isinstance(payload, Mapping):
        raise PIRGStagePlanError("PIRG remote stage plan must be a mapping")
    if payload.get("schema") != "posterior_identity_residual_gate_remote_stage_plan_v1":
        raise PIRGStagePlanError("PIRG remote stage plan schema drift")
    expected_assets = plan.sealed_cell_d_init_assets_payload()
    if payload.get("sealed_cell_d_initialization_assets") != expected_assets:
        raise PIRGStagePlanError("PIRG sealed Cell-D initialization asset table drift")
    stage_files = payload.get("stage_files")
    if not isinstance(stage_files, list) or not all(isinstance(item, Mapping) for item in stage_files):
        raise PIRGStagePlanError("PIRG remote stage file schema drift")
    actual = [
        dict(item) for item in stage_files
        if item.get("role") in {_INIT_BODY_ROLE, _INIT_SIDECAR_ROLE}
    ]
    if actual != _sealed_cell_d_initialization_stage_files():
        raise PIRGStagePlanError("PIRG sealed Cell-D initialization body/sidecar transport drift")
    return dict(payload)


def score_overlay_paths() -> tuple[str, ...]:
    """Return the additive score files absent from the source-stage closure.

    The score is staged only after a source terminal has been independently
    accepted.  It adds code/metadata leaves to the same reviewed stage; it
    neither replaces a source-closure byte nor copies an NWB or a source
    result.  Keeping this list static makes the future overlay auditable
    without resolving V3 or evaluation assets at dry-plan time.
    """
    source = set(IMPLEMENTATION_CLOSURE)
    return tuple(path for path in train.SCORE_IMPLEMENTATION_CLOSURE if path not in source)


def build_remote_stage_plan(root: Path) -> dict[str, object]:
    """Return an explicit code/authority plan; it performs no NWB I/O."""
    closure = implementation_closure(Path(root)).payload()
    payload: dict[str, object] = {
        "schema": "posterior_identity_residual_gate_remote_stage_plan_v1",
        "cell": plan.CELL,
        "stage_root": REMOTE_STAGE_ROOT,
        "result_root_relative": REMOTE_RESULT_ROOT_RELATIVE,
        "score_root_relative": REMOTE_SCORE_ROOT_RELATIVE,
        "closure": closure,
        "stage_files": [
            {"relative_path": path, "sha256": closure["sha256_by_path"][path], "role": "code_or_metadata"}
            for path in closure["paths"]
        ] + [
            {
                "relative_path": relative,
                "sha256": sha256,
                "role": "immutable_source_adapter_authority",
                "source_mode": (None if mode is None else f"{mode:04o}"),
                "destination_mode": "0444",
                "paired_sidecar": (None if mode is None else f"{sha256}  {Path(relative).name}\\n"),
            }
            for relative, sha256, mode in plan.SOURCE_ADAPTER_AUTHORITY_ASSETS
        ] + _sealed_cell_d_initialization_stage_files(),
        "nwb_assets_in_stage": False,
        "source_data_policy": {
            "strict27_source_adapter": "compose_existing_audited_posterior_source_adapter_v2",
            "copy_source_nwb": False,
            "symlink_or_bind_source_nwb": False,
            "target_within_external_opened_by_training": False,
        },
        "source_adapter_authority_assets": [
            {"relative_path": relative, "sha256": sha256, "source_mode": mode}
            for relative, sha256, mode in plan.SOURCE_ADAPTER_AUTHORITY_ASSETS
        ],
        "sealed_cell_d_initialization_assets": plan.sealed_cell_d_init_assets_payload(),
        "quick_score_substrate": {
            "v3_stage_root": plan.V3_QUICK_STAGE_ROOT,
            "v3_input_authority_sha256": plan.V3_QUICK_INPUT_AUTHORITY_SHA256,
            "v3_score_sha256": plan.V3_QUICK_SCORE_SHA256,
            "v3_terminal_sha256": plan.V3_QUICK_TERMINAL_SHA256,
            "reuse_completed_v3_held_evaluation_assets_in_place": True,
            "copy_or_restage_evaluation_nwb": False,
            "new_pirg_score_cells_reuse_v3_rows": False,
        },
        "execution": "NOT_AUTHORIZED_BY_PLAN__requires_future_in_process_root_review_capability",
    }
    return validate_remote_stage_plan(payload)


def build_remote_score_overlay_plan(root: Path) -> dict[str, object]:
    """Future-only score-overlay plan; no transport and no target I/O.

    Calling this descriptor-hashes the reviewed score closure, so it is not
    part of the public dry CLI or the source-only test route.  The physical
    scorer continues to resolve its six evaluation assets through the
    immutable V3 stage in place; the overlay carries code and metadata only.
    """
    # This import is deliberately deferred.  The source stage carries this
    # plan module but not the score-only module; the overlay is constructed
    # from a full reviewed local tree before those score leaves are copied.
    from . import score

    closure = score.score_implementation_closure(Path(root)).payload()
    overlay = score_overlay_paths()
    if tuple(path for path in closure["paths"] if path not in set(IMPLEMENTATION_CLOSURE)) != overlay:
        raise RuntimeError("PIRG score overlay topology drift")
    return {
        "schema": "posterior_identity_residual_gate_score_overlay_plan_v1",
        "cell": plan.CELL,
        "base_stage_root": REMOTE_STAGE_ROOT,
        "requires_completed_source_terminal": True,
        "source_result_relative": REMOTE_RESULT_ROOT_RELATIVE,
        "score_result_relative": REMOTE_SCORE_ROOT_RELATIVE,
        "score_closure": closure,
        "overlay_files": [
            {"relative_path": path, "sha256": closure["sha256_by_path"][path], "role": "score_code_or_metadata"}
            for path in overlay
        ],
        "source_closure_bytes_replaced": False,
        "source_result_bytes_copied": False,
        "evaluation_nwb_copied_or_re_staged": False,
        "evaluation_input_route": "immutable_v3_stage_held_assets_in_place",
        "execution": "NOT_AUTHORIZED_BY_PLAN__requires_future_in_process_root_review_capability",
    }
