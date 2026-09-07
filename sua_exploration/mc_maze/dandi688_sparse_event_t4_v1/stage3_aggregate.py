"""CPU-only pseudo-MUA aggregation and final immutable Stage-3 receipt.

This module never imports Torch, a DataModule, or the DANDI reader.  It only
opens immutable Stage-2/3 receipt leaves after the GPU work has terminated.
"""
from __future__ import annotations

from pathlib import Path

from . import plan
from .core import require
from .lifecycle import publish_immutable_json
from .scoring import baseline_claim_gate, pseudo_mua_estimator_gate, pseudo_mua_film_gate
from .stage2_aggregate import (
    _contrast,
    _held_json,
    _read_seed,
    admit_stage3_branch,
)


def _latest_stage2_aggregate(repo_root: Path) -> tuple[str, str, dict[str, object]]:
    """Return the newest published SUA authority, mirroring Stage-3 admission."""
    repo_root = Path(repo_root).resolve()
    candidates = (
        (plan.STAGE2_V3_RELATIVE, "stage2_v3_aggregate"),
        (plan.STAGE2_V2_RELATIVE, "stage2_v2_aggregate"),
        (plan.RESULT_PARENT_RELATIVE + "/stage2", "stage2_aggregate"),
    )
    for relative, expected_stage in candidates:
        root = repo_root / relative
        if (root / "aggregate.json").exists():
            digest, body = _held_json(root, "aggregate.json")
            require(
                body.get("status") == "PASS"
                and body.get("route") == plan.ROUTE_NAME
                and body.get("stage") == expected_stage,
                "latest Stage2 aggregate identity drift",
            )
            return relative, digest, body
    raise FileNotFoundError("no published Stage2 aggregate exists")


def _branch_roots(root: Path, branch: str) -> dict[int, Path]:
    return {
        seed: root / branch / "pseudo_mua" / f"seed{seed}"
        for seed in plan.SEEDS
    }


def _aggregate_admitted_branch(
    repo_root: Path,
    *,
    root: Path,
    branch: str,
) -> dict[str, object]:
    """Held-validate all three Stage-3 roots for a Stage-2-admitted branch."""
    admission = admit_stage3_branch(repo_root, branch)
    roots = _branch_roots(root, branch)
    require(all(path.is_dir() for path in roots.values()), f"Stage3 {branch} seed topology is incomplete")
    rows = {
        seed: _read_seed(path, branch=branch, seed=seed, stage_label="stage3", view="pseudo_mua")
        for seed, path in roots.items()
    }
    for seed, row in rows.items():
        require(
            row["attempt_stage2_admission"] == admission,
            f"Stage3 {branch} seed{seed} parent aggregate binding drift",
        )
    if branch == "estimator":
        summary = _contrast(rows, "POST700-T4", "WHOLE-T4")
        gate = pseudo_mua_estimator_gate(summary)
        summaries = {"estimator": summary}
    else:
        summaries = {
            "semantic": _contrast(rows, "SE-T4", "EMPTY"),
            "baseline": _contrast(rows, "SE-T4", "PHASE-R"),
            "attachment": _contrast(rows, "SE-T4", "ROW-SHUFFLE"),
            "product": _contrast(rows, "SE-T4", "WHOLE-NATIVE"),
            "capacity": _contrast(rows, "EMPTY", "WHOLE-NATIVE"),
        }
        gate = pseudo_mua_film_gate(summaries["semantic"], summaries["attachment"])
    masks = {tuple(rows[seed]["reliability_mask"]) for seed in plan.SEEDS}
    require(len(masks) == 1, f"Stage3 {branch} reliability masks disagree")
    primary = summaries["estimator"] if branch == "estimator" else summaries["semantic"]
    result = {
        "status": "PASS" if gate else "FAIL",
        "gate": bool(gate),
        "admission": admission,
        "summaries": summaries,
        "absolute_threshold_0p015_reported": bool(float(primary["grand_mean"]) >= .015),
        "reliability_mask": list(next(iter(masks))),
        "seed_terminal_sha256": {str(seed): rows[seed]["terminal_sha256"] for seed in plan.SEEDS},
    }
    if branch == "film":
        result["baseline_claim_gate"] = baseline_claim_gate(
            summaries["baseline"],
            delta_b_retained=bool(next(iter(masks))[3]),
        )
    return result


def _not_admitted_branch(root: Path, branch: str, parent: dict[str, object]) -> dict[str, object]:
    """Enforce the work-order rule that a failed SUA branch creates no root."""
    require(not (root / branch).exists(), f"Stage3 {branch} root exists although SUA branch was not admitted")
    return {
        "status": "NOT_ADMITTED",
        "gate": None,
        "parent_stage2_branch_status": parent.get("status"),
        "parent_stage2_branch_gate": parent.get("gate"),
        "summaries": None,
        "seed_terminal_sha256": None,
    }


def _outcome(estimator: dict[str, object], film: dict[str, object]) -> str:
    """Conservative within-view outcome label; never compares pseudo-MUA to SUA."""
    estimator_positive = estimator.get("gate") is True
    film_positive = film.get("gate") is True
    if film_positive:
        return "PSEUDO_MUA_FILM_REPLICATION_POSITIVE"
    if film.get("status") == "NOT_ADMITTED":
        return "PSEUDO_MUA_ESTIMATOR_ONLY__FILM_NOT_ADMITTED_BY_SUA"
    if estimator_positive:
        return "PSEUDO_MUA_ESTIMATOR_REPLICATION_POSITIVE__FILM_NULL"
    if estimator.get("status") == "NOT_ADMITTED" and film.get("status") == "NOT_ADMITTED":
        return "NO_PSEUDO_MUA_BRANCH_ADMITTED_BY_SUA"
    return "PSEUDO_MUA_CONDITIONAL_REPLICATION_NULL"


def aggregate_stage3(repo_root: Path) -> dict[str, object]:
    """Publish the one final pseudo-MUA three-seed result receipt.

    A branch runs iff its corresponding frozen SUA Stage-2 gate passed.  The
    aggregate documents explicitly non-admitted branches instead of treating
    their absence as missing work.
    """
    repo_root = Path(repo_root).resolve()
    stage2_relative, stage2_sha, stage2 = _latest_stage2_aggregate(repo_root)
    root = repo_root / plan.STAGE3_RELATIVE
    final_root = repo_root / plan.FINAL_AGGREGATE_RELATIVE
    target = final_root / plan.STAGE3_PSEUDO_MUA_AGGREGATE_NAME
    require(not target.exists() and not target.with_name(target.name + ".sha256").exists(), "Stage3 final aggregate already exists")
    require(not root.exists() or root.is_dir(), "Stage3 root is not a directory")

    outputs: dict[str, dict[str, object]] = {}
    for branch in ("estimator", "film"):
        parent = stage2.get(branch)
        require(isinstance(parent, dict), f"Stage2 {branch} branch body missing")
        if parent.get("status") == "PASS" and parent.get("gate") is True:
            outputs[branch] = _aggregate_admitted_branch(repo_root, root=root, branch=branch)
        else:
            outputs[branch] = _not_admitted_branch(root, branch, parent)

    if outputs["estimator"].get("status") != "NOT_ADMITTED" and outputs["film"].get("status") != "NOT_ADMITTED":
        require(
            outputs["estimator"].get("reliability_mask") == outputs["film"].get("reliability_mask"),
            "Stage3 estimator/FiLM reliability mask drift",
        )
    final_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS",
        "route": plan.ROUTE_NAME,
        "stage": "stage3_pseudo_mua_aggregate",
        "view": "pseudo_mua",
        "seeds": list(plan.SEEDS),
        "source_stage2_aggregate": {
            "relative": stage2_relative + "/aggregate.json",
            "sha256": stage2_sha,
            "stage": stage2["stage"],
        },
        "branches": outputs,
        "within_view_outcome": _outcome(outputs["estimator"], outputs["film"]),
        "scope": "PSEUDO_MUA_SOURCE_27_TRAIN__PSEUDO_MUA_SOURCE_6_VALIDATION__NO_TEST_EXTERNAL_EVALAI",
    }
    digest = publish_immutable_json(final_root, plan.STAGE3_PSEUDO_MUA_AGGREGATE_NAME, payload)
    return {"aggregate_sha256": digest, **payload}
