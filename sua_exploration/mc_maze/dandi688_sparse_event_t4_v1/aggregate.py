"""Immutable three-seed Stage-1-v3 aggregation and opening receipt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import plan
from .admission import _held_bytes
from .core import require, stage1_film_opening_decision
from .lifecycle import publish_immutable_json
from .scoring import average_paired_seeds


_TRAINED_ARMS = ("EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE")
_SCORED_ARMS = (
    "WHOLE-NATIVE",
    "WHOLE-PARENTNORM",
    "WHOLE-M10NORM",
    "POST700-M10NORM",
    *_TRAINED_ARMS,
)


def _held_json(root: Path, name: str) -> tuple[str, dict[str, object]]:
    body = _held_bytes(root / name)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = _held_bytes(root / f"{name}.sha256").decode("ascii").strip().split()
    require(sidecar == [digest, name], f"held sidecar drift: {root / name}")
    decoded = json.loads(body)
    require(isinstance(decoded, dict), f"held JSON is not an object: {root / name}")
    return digest, decoded


def _read_seed(root: Path, seed: int) -> dict[str, object]:
    artifact_names = tuple(f"stage1_{arm.lower()}.pt" for arm in _TRAINED_ARMS)
    body_names = ("attempt.json", "training.json", "score.json", "terminal.json")
    expected = {name for name in (*artifact_names, *body_names)} | {
        f"{name}.sha256" for name in (*artifact_names, *body_names)
    }
    require(root.is_dir() and {path.name for path in root.iterdir()} == expected, f"seed{seed} topology drift")
    digests: dict[str, str] = {}
    bodies: dict[str, dict[str, object]] = {}
    for name in body_names:
        digests[name], bodies[name] = _held_json(root, name)
    artifact_digests = {}
    for name in artifact_names:
        body = _held_bytes(root / name)
        digest = hashlib.sha256(body).hexdigest()
        sidecar = _held_bytes(root / f"{name}.sha256").decode("ascii").strip().split()
        require(sidecar == [digest, name], f"seed{seed} artifact sidecar drift")
        artifact_digests[name] = digest

    attempt, training, score, terminal = (
        bodies["attempt.json"],
        bodies["training.json"],
        bodies["score.json"],
        bodies["terminal.json"],
    )
    require(attempt.get("status") == "STARTED" and attempt.get("stage") == "stage1_v3", f"seed{seed} attempt drift")
    require(terminal.get("status") == "PASS", f"seed{seed} did not terminally pass")
    require(terminal.get("attempt_sha256") == digests["attempt.json"], f"seed{seed} terminal/attempt link drift")
    require(terminal.get("training_sha256") == digests["training.json"], f"seed{seed} terminal/training link drift")
    require(terminal.get("score_sha256") == digests["score.json"], f"seed{seed} terminal/score link drift")
    require(training.get("attempt_sha256") == digests["attempt.json"], f"seed{seed} training/attempt link drift")
    require(score.get("attempt_sha256") == digests["attempt.json"], f"seed{seed} score/attempt link drift")
    require(training.get("epochs") == plan.EPOCHS, f"seed{seed} epoch count drift")
    require(training.get("average_epochs_zero_based") == list(plan.AVERAGE_EPOCHS_ZERO_BASED), f"seed{seed} averaging law drift")
    require(training.get("head_template_factory_calls") == 1 and training.get("initial_head_states_identical") is True, f"seed{seed} paired head initialization absent")
    initial = training.get("initial_head_state_sha256")
    require(isinstance(initial, dict) and set(initial) == set(_TRAINED_ARMS) and len(set(initial.values())) == 1, f"seed{seed} initial head SHA drift")
    checkpoints = training.get("checkpoints")
    require(isinstance(checkpoints, dict) and set(checkpoints) == set(_TRAINED_ARMS), f"seed{seed} checkpoint roster drift")
    for arm in _TRAINED_ARMS:
        name = f"stage1_{arm.lower()}.pt"
        require(checkpoints[arm] == artifact_digests[name], f"seed{seed} {arm} artifact link drift")
    zero = training.get("zero_anchor")
    require(isinstance(zero, dict) and zero.get("passed") is True, f"seed{seed} zero anchor failed")
    require(training.get("parent_state_before") == training.get("parent_state_after_training"), f"seed{seed} parent changed in training")
    require(score.get("parent_state_before") == score.get("parent_state_after_training") == score.get("parent_state_after_score"), f"seed{seed} parent changed by scoring")

    rows = score.get("per_session_r2")
    evidence = score.get("per_session_evidence")
    require(isinstance(rows, dict) and set(rows) == set(_SCORED_ARMS), f"seed{seed} score arm roster drift")
    require(isinstance(evidence, dict) and set(evidence) == set(_SCORED_ARMS), f"seed{seed} evidence arm roster drift")
    sessions = tuple(sorted(rows["WHOLE-NATIVE"]))
    require(len(sessions) == 6, f"seed{seed} validation roster drift")
    for arm in _SCORED_ARMS:
        require(tuple(sorted(rows[arm])) == sessions == tuple(sorted(evidence[arm])), f"seed{seed} {arm} session roster drift")
        require(all(np.isfinite(float(rows[arm][session])) for session in sessions), f"seed{seed} {arm} nonfinite R2")
    for session in sessions:
        targets = {evidence[arm][session]["target_sha256"] for arm in _SCORED_ARMS}
        queries = {evidence[arm][session]["query_sha256"] for arm in _SCORED_ARMS}
        windows = {int(evidence[arm][session]["window_count"]) for arm in _SCORED_ARMS}
        require(len(targets) == len(queries) == len(windows) == 1, f"seed{seed} {session} scoring surface mismatch")
    return {
        "terminal_sha256": digests["terminal.json"],
        "training_sha256": digests["training.json"],
        "score_sha256": digests["score.json"],
        "scores": rows,
    }


def aggregate_stage1_v2(repo_root: Path) -> dict[str, object]:
    """Validate all three Stage-1-v3 seeds, compute weak opening, and publish once.

    The public function name is retained for the route-local caller that was
    established before the additive v3 successor root was introduced.
    """
    repo_root = Path(repo_root).resolve()
    root = repo_root / plan.STAGE1_SUCCESSOR_RELATIVE
    target = root / "aggregate.json"
    require(not target.exists() and not target.with_name("aggregate.json.sha256").exists(), "Stage1-v3 aggregate already exists")
    seeds = {
        seed: _read_seed(root / "sua" / f"seed{seed}", seed)
        for seed in plan.SEEDS
    }
    scores = {seed: value["scores"] for seed, value in seeds.items()}

    def contrast(candidate: str, baseline: str) -> dict[str, object]:
        return average_paired_seeds(
            {seed: scores[seed][candidate] for seed in plan.SEEDS},
            {seed: scores[seed][baseline] for seed in plan.SEEDS},
        )

    summaries = {
        "estimator_ood": contrast("POST700-M10NORM", "WHOLE-M10NORM"),
        "semantic_ood": contrast("SE-T4", "EMPTY"),
        "baseline_ood": contrast("SE-T4", "PHASE-R"),
        "attachment_ood": contrast("SE-T4", "ROW-SHUFFLE"),
        "product_ood": contrast("SE-T4", "WHOLE-NATIVE"),
        "capacity_or_global_correction_ood": contrast("EMPTY", "WHOLE-NATIVE"),
    }
    semantic = summaries["semantic_ood"]["per_session_seed_average"]
    attachment = summaries["attachment_ood"]["per_session_seed_average"]
    opened = stage1_film_opening_decision(semantic, attachment)
    opening = {
        "status": "OPEN" if opened else "CLOSED",
        "semantic_mean": float(summaries["semantic_ood"]["grand_mean"]),
        "attachment_mean": float(summaries["attachment_ood"]["grand_mean"]),
        "semantic_positive_sessions": int(summaries["semantic_ood"]["positive_sessions"]),
        "predicate": "semantic_mean>0__attachment_mean>0__semantic_positive_sessions>=4",
    }
    payload = {
        "status": "PASS",
        "route": plan.ROUTE_NAME,
        "stage": "stage1_v3_aggregate",
        "seeds": list(plan.SEEDS),
        "stage1_terminal_sha256": {str(seed): seeds[seed]["terminal_sha256"] for seed in plan.SEEDS},
        "stage1_training_sha256": {str(seed): seeds[seed]["training_sha256"] for seed in plan.SEEDS},
        "stage1_score_sha256": {str(seed): seeds[seed]["score_sha256"] for seed in plan.SEEDS},
        "summaries": summaries,
        "stage2_film_opening": opening,
        "scope": "SOURCE_27_TRAIN__SOURCE_6_VALIDATION__NO_TEST_EXTERNAL_EVALAI",
    }
    digest = publish_immutable_json(root, "aggregate.json", payload)
    return {"aggregate_sha256": digest, **payload}
