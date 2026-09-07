"""CPU-only immutable aggregation and Stage-3 admission for Stage-2 SUA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import plan
from .admission import _held_bytes
from .core import require
from .lifecycle import publish_immutable_json
from .scoring import average_paired_seeds, baseline_claim_gate, estimator_gate, film_gate


_ESTIMATOR_ARMS = ("WHOLE-T4", "POST700-T4")
_FILM_ARMS = ("WHOLE-NATIVE", "EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE")


def _held_json(root: Path, name: str) -> tuple[str, dict[str, object]]:
    body = _held_bytes(root / name)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = _held_bytes(root / f"{name}.sha256").decode("ascii").strip().split()
    require(sidecar == [digest, name], f"Stage2 held sidecar drift: {root / name}")
    decoded = json.loads(body)
    require(isinstance(decoded, dict), f"Stage2 JSON body is not an object: {root / name}")
    return digest, decoded


def _read_seed(root: Path, *, branch: str, seed: int, stage_label: str = "stage2", view: str = "sua") -> dict[str, object]:
    require(branch in {"estimator", "film"}, "unsupported Stage2 branch")
    arms = _ESTIMATOR_ARMS if branch == "estimator" else _FILM_ARMS
    prefix = f"{stage_label}_{branch}"
    artifacts = tuple(f"{prefix}_{arm.lower()}.pt" for arm in arms)
    bodies = ("attempt.json", "training.json", "score.json", "terminal.json")
    expected = set(artifacts) | set(bodies) | {f"{name}.sha256" for name in (*artifacts, *bodies)}
    require(root.is_dir() and {item.name for item in root.iterdir()} == expected, f"Stage2 {branch} seed{seed} topology drift")
    digests, parsed = {}, {}
    for name in bodies:
        digests[name], parsed[name] = _held_json(root, name)
    artifact_digests = {}
    for name in artifacts:
        body = _held_bytes(root / name)
        digest = hashlib.sha256(body).hexdigest()
        sidecar = _held_bytes(root / f"{name}.sha256").decode("ascii").strip().split()
        require(sidecar == [digest, name], f"Stage2 {branch} seed{seed} artifact sidecar drift")
        artifact_digests[name] = digest

    attempt, training, score, terminal = (parsed[name] for name in bodies)
    stage = f"{stage_label}_{branch}"
    require(
        attempt.get("status") == "STARTED"
        and attempt.get("stage") == stage
        and attempt.get("seed") == seed
        and attempt.get("view") == view,
        f"Stage2 {branch} seed{seed} attempt drift",
    )
    require(terminal.get("status") == "PASS", f"Stage2 {branch} seed{seed} did not pass")
    require(terminal.get("attempt_sha256") == digests["attempt.json"], f"Stage2 {branch} seed{seed} terminal/attempt link drift")
    require(terminal.get("training_sha256") == digests["training.json"], f"Stage2 {branch} seed{seed} terminal/training link drift")
    require(terminal.get("score_sha256") == digests["score.json"], f"Stage2 {branch} seed{seed} terminal/score link drift")
    require(training.get("attempt_sha256") == digests["attempt.json"] == score.get("attempt_sha256"), f"Stage2 {branch} seed{seed} attempt binding drift")
    require(training.get("average_epochs_zero_based") == list(plan.AVERAGE_EPOCHS_ZERO_BASED), f"Stage2 {branch} seed{seed} average law drift")
    require(isinstance(training.get("epoch_receipts"), list) and len(training["epoch_receipts"]) == plan.EPOCHS, f"Stage2 {branch} seed{seed} epoch receipt drift")
    require(all(set(item.get("optimizer_steps", {})) == set(arms) for item in training["epoch_receipts"]), f"Stage2 {branch} seed{seed} paired arm receipt drift")
    require(all(len(set(item["optimizer_steps"].values())) == 1 and next(iter(item["optimizer_steps"].values())) > 0 for item in training["epoch_receipts"]), f"Stage2 {branch} seed{seed} paired step drift")
    artifacts_receipt = training.get("full_state_artifacts")
    require(isinstance(artifacts_receipt, dict) and set(artifacts_receipt) == set(arms), f"Stage2 {branch} seed{seed} artifact receipt drift")
    for arm in arms:
        artifact = artifacts_receipt[arm]
        require(isinstance(artifact, dict) and artifact.get("artifact_sha256") == artifact_digests[f"{prefix}_{arm.lower()}.pt"], f"Stage2 {branch} seed{seed} {arm} artifact link drift")
    require(terminal.get("full_state_artifacts") == artifacts_receipt, f"Stage2 {branch} seed{seed} terminal artifact link drift")
    if branch == "film":
        initial = training.get("initial_head_state_sha256")
        require(training.get("head_template_factory_calls") == 1 and training.get("initial_head_states_identical") is True, f"Stage2 film seed{seed} head template receipt drift")
        require(isinstance(initial, dict) and set(initial) == set(_FILM_ARMS[1:]) and len(set(initial.values())) == 1, f"Stage2 film seed{seed} initial head SHA drift")
        zero = training.get("zero_anchor")
        require(isinstance(zero, dict) and zero.get("passed") is True, f"Stage2 film seed{seed} zero anchor drift")

    rows = score.get("per_session")
    require(isinstance(rows, dict) and set(rows) == set(arms), f"Stage2 {branch} seed{seed} score arm roster drift")
    sessions = tuple(sorted(rows[arms[0]]))
    require(len(sessions) == 6, f"Stage2 {branch} seed{seed} validation roster drift")
    for arm in arms:
        require(tuple(sorted(rows[arm])) == sessions, f"Stage2 {branch} seed{seed} {arm} session roster drift")
        require(all(np.isfinite(float(rows[arm][session]["r2"])) for session in sessions), f"Stage2 {branch} seed{seed} {arm} R2 drift")
    for session in sessions:
        target = {rows[arm][session].get("target_sha256") for arm in arms}
        query = {rows[arm][session].get("query_sha256") for arm in arms}
        windows = {int(rows[arm][session].get("window_count", -1)) for arm in arms}
        require(len(target) == len(query) == len(windows) == 1, f"Stage2 {branch} seed{seed} {session} Q50 surface mismatch")
    masks = []
    def collect_masks(value):
        if isinstance(value, dict):
            mask = value.get("reliability_mask")
            if isinstance(mask, list) and len(mask) == 4:
                masks.append(tuple(bool(item) for item in mask))
            for child in value.values():
                collect_masks(child)
        elif isinstance(value, list):
            for child in value:
                collect_masks(child)
    collect_masks(attempt)
    require(masks and len(set(masks)) == 1, f"Stage2 {branch} seed{seed} reliability-mask binding drift")
    return {
        "terminal_sha256": digests["terminal.json"],
        "training_sha256": digests["training.json"],
        "score_sha256": digests["score.json"],
        "scores": {arm: {session: float(rows[arm][session]["r2"]) for session in sessions} for arm in arms},
        "reliability_mask": list(masks[0]),
        "attempt_stage2_admission": attempt.get("stage2_aggregate_admission"),
    }


def _contrast(seeds: dict[int, dict[str, object]], candidate: str, baseline: str) -> dict[str, object]:
    return average_paired_seeds(
        {seed: seeds[seed]["scores"][candidate] for seed in plan.SEEDS},
        {seed: seeds[seed]["scores"][baseline] for seed in plan.SEEDS},
    )


def aggregate_stage2(repo_root: Path) -> dict[str, object]:
    """Aggregate terminal SUA Stage-2 seed roots without opening data or Torch."""
    repo_root = Path(repo_root).resolve()
    # A later successor always supersedes an earlier namespace.  This ordering
    # makes the public aggregate API incapable of reopening an obsolete
    # authority after a controller-approved interruption.
    if (repo_root / plan.STAGE2_V3_RELATIVE).exists():
        return aggregate_stage2_v3(repo_root)
    # Once a successor root exists, there is deliberately only one authority:
    # the mixed V1/V2 aggregate.  Do not allow callers to accidentally publish
    # a second V1 aggregate while the interrupted execution is being repaired.
    if (repo_root / plan.STAGE2_V2_RELATIVE).exists():
        return aggregate_stage2_v2(repo_root)
    root = repo_root / plan.RESULT_PARENT_RELATIVE / "stage2"
    target = root / "aggregate.json"
    require(not target.exists() and not target.with_name("aggregate.json.sha256").exists(), "Stage2 aggregate already exists")
    estimator = {seed: _read_seed(root / "estimator" / "sua" / f"seed{seed}", branch="estimator", seed=seed) for seed in plan.SEEDS}
    estimator_summary = _contrast(estimator, "POST700-T4", "WHOLE-T4")
    estimator_pass = estimator_gate(estimator_summary)
    estimator_masks = {tuple(estimator[seed]["reliability_mask"]) for seed in plan.SEEDS}
    require(len(estimator_masks) == 1, "Stage2 estimator seed reliability masks disagree")
    film_roots = [root / "film" / "sua" / f"seed{seed}" for seed in plan.SEEDS]
    present = [item.exists() for item in film_roots]
    require(not any(present) or all(present), "Stage2 FiLM seed topology is partial")
    film_payload = {"status": "NOT_EXECUTED", "gate": None, "summaries": None, "seed_terminal_sha256": None}
    if all(present):
        film = {seed: _read_seed(path, branch="film", seed=seed) for seed, path in zip(plan.SEEDS, film_roots)}
        summaries = {
            "semantic": _contrast(film, "SE-T4", "EMPTY"),
            "baseline": _contrast(film, "SE-T4", "PHASE-R"),
            "attachment": _contrast(film, "SE-T4", "ROW-SHUFFLE"),
            "product": _contrast(film, "SE-T4", "WHOLE-NATIVE"),
            "capacity": _contrast(film, "EMPTY", "WHOLE-NATIVE"),
        }
        film_masks = {tuple(film[seed]["reliability_mask"]) for seed in plan.SEEDS}
        require(len(film_masks) == 1 and film_masks == estimator_masks, "Stage2 film/estimator reliability masks disagree")
        delta_b_retained = bool(next(iter(film_masks))[3])
        film_pass = film_gate(summaries["semantic"], summaries["attachment"])
        film_payload = {"status": "PASS" if film_pass else "FAIL", "gate": bool(film_pass), "baseline_claim_gate": baseline_claim_gate(summaries["baseline"], delta_b_retained=delta_b_retained), "summaries": summaries, "seed_terminal_sha256": {str(seed): film[seed]["terminal_sha256"] for seed in plan.SEEDS}, "reliability_mask": list(next(iter(film_masks))), "delta_b_retained": delta_b_retained}
    payload = {
        "status": "PASS",
        "route": plan.ROUTE_NAME,
        "stage": "stage2_aggregate",
        "seeds": list(plan.SEEDS),
        "estimator": {"status": "PASS" if estimator_pass else "FAIL", "gate": bool(estimator_pass), "summary": estimator_summary, "seed_terminal_sha256": {str(seed): estimator[seed]["terminal_sha256"] for seed in plan.SEEDS}, "reliability_mask": list(next(iter(estimator_masks)))},
        "film": film_payload,
        "scope": "SUA_SOURCE_27_TRAIN__SUA_SOURCE_6_VALIDATION__NO_TEST_EXTERNAL_EVALAI",
    }
    digest = publish_immutable_json(root, "aggregate.json", payload)
    return {"aggregate_sha256": digest, **payload}


def admit_stage3_branch(repo_root: Path, branch: str) -> dict[str, object]:
    """Held-read a successful Stage-2 aggregate before a Stage-3 lifecycle attempt."""
    require(branch in {"estimator", "film"}, "unsupported Stage3 branch")
    repo_root = Path(repo_root).resolve()
    v3_root = repo_root / plan.STAGE2_V3_RELATIVE
    v2_root = repo_root / plan.STAGE2_V2_RELATIVE
    root = (
        v3_root if (v3_root / "aggregate.json").exists()
        else v2_root if (v2_root / "aggregate.json").exists()
        else repo_root / plan.RESULT_PARENT_RELATIVE / "stage2"
    )
    digest, aggregate = _held_json(root, "aggregate.json")
    require(aggregate.get("status") == "PASS" and aggregate.get("route") == plan.ROUTE_NAME and aggregate.get("stage") in {"stage2_aggregate", "stage2_v2_aggregate", "stage2_v3_aggregate"}, "Stage2 aggregate identity drift")
    branch_body = aggregate.get(branch)
    require(isinstance(branch_body, dict) and branch_body.get("status") == "PASS" and branch_body.get("gate") is True, f"Stage3 {branch} is not admitted by Stage2")
    relative = (
        plan.STAGE2_V3_RELATIVE + "/aggregate.json" if root == v3_root
        else plan.STAGE2_V2_RELATIVE + "/aggregate.json" if root == v2_root
        else plan.RESULT_PARENT_RELATIVE + "/stage2/aggregate.json"
    )
    return {"stage2_aggregate_relative": relative, "stage2_aggregate_sha256": digest, "branch": branch, "branch_gate": branch_body}


def admit_stage2_v2_successor(repo_root: Path, branch: str) -> dict[str, object]:
    """Hold the external-context incident and its exact abandoned V1 attempts."""
    require(branch in {"estimator", "film"}, "unsupported Stage2-v2 branch")
    repo_root = Path(repo_root).resolve()
    incident = _held_bytes(repo_root / plan.STAGE2_V2_INCIDENT_RELATIVE)
    incident_sha = hashlib.sha256(incident).hexdigest()
    require(incident_sha == plan.STAGE2_V2_INCIDENT_SHA256, "Stage2-v2 incident SHA drift")
    held = {}
    for old_branch, by_seed in plan.STAGE2_V1_INTERRUPTED_ATTEMPTS.items():
        for seed, expected_sha in by_seed.items():
            root = repo_root / plan.RESULT_PARENT_RELATIVE / "stage2" / old_branch / "sua" / f"seed{seed}"
            require(root.is_dir() and {item.name for item in root.iterdir()} == {"attempt.json", "attempt.json.sha256"}, f"abandoned Stage2 {old_branch} seed{seed} topology drift")
            digest, attempt = _held_json(root, "attempt.json")
            require(digest == expected_sha and attempt.get("status") == "STARTED" and attempt.get("stage") == f"stage2_{old_branch}", f"abandoned Stage2 {old_branch} seed{seed} attempt drift")
            held[f"{old_branch}:{seed}"] = digest
    return {"incident_relative": plan.STAGE2_V2_INCIDENT_RELATIVE, "incident_sha256": incident_sha, "held_interrupted_attempt_sha256": held, "branch": branch}


def admit_stage2_v3_successor(repo_root: Path) -> dict[str, object]:
    """Hold the user-requested V2 pre-CUDA interruption before V3 attempt."""
    repo_root = Path(repo_root).resolve()
    incident = _held_bytes(repo_root / plan.STAGE2_V3_INCIDENT_RELATIVE)
    incident_sha = hashlib.sha256(incident).hexdigest()
    require(incident_sha == plan.STAGE2_V3_INCIDENT_SHA256, "Stage2-v3 incident SHA drift")
    root = repo_root / plan.STAGE2_V2_RELATIVE / "estimator" / "sua" / "seed44"
    expected = {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}
    require(root.is_dir() and {item.name for item in root.iterdir()} == expected, "interrupted Stage2-v2 estimator topology drift")
    attempt_sha, attempt = _held_json(root, "attempt.json")
    failure_sha, failure = _held_json(root, "failure.json")
    require(attempt_sha == plan.STAGE2_V2_INTERRUPTED_ESTIMATOR["attempt_sha256"], "interrupted Stage2-v2 attempt SHA drift")
    require(failure_sha == plan.STAGE2_V2_INTERRUPTED_ESTIMATOR["failure_sha256"], "interrupted Stage2-v2 failure SHA drift")
    require(
        attempt.get("status") == "STARTED"
        and attempt.get("stage") == "stage2_v2_estimator"
        and attempt.get("seed") == 44
        and attempt.get("view") == "sua"
        and attempt.get("gpu_initialized") is False,
        "interrupted Stage2-v2 attempt identity drift",
    )
    require(
        failure.get("status") == "FAIL"
        and failure.get("attempt_sha256") == attempt_sha
        and failure.get("exception_type") == "KeyboardInterrupt"
        and failure.get("published_prefix") == ["attempt.json", "attempt.json.sha256"],
        "interrupted Stage2-v2 failure identity drift",
    )
    return {
        "incident_relative": plan.STAGE2_V3_INCIDENT_RELATIVE,
        "incident_sha256": incident_sha,
        "held_interrupted_v2_estimator": {
            "attempt_sha256": attempt_sha,
            "failure_sha256": failure_sha,
        },
    }


def aggregate_stage2_v2(repo_root: Path) -> dict[str, object]:
    """Publish the one immutable mixed-authority Stage2-v2 aggregate."""
    repo_root = Path(repo_root).resolve()
    root = repo_root / plan.STAGE2_V2_RELATIVE
    target = root / "aggregate.json"
    require(not target.exists() and not target.with_name("aggregate.json.sha256").exists(), "Stage2-v2 aggregate already exists")
    incident = admit_stage2_v2_successor(repo_root, "estimator")
    estimator = {
        42: _read_seed(repo_root / plan.RESULT_PARENT_RELATIVE / "stage2" / "estimator" / "sua" / "seed42", branch="estimator", seed=42),
        43: _read_seed(repo_root / plan.RESULT_PARENT_RELATIVE / "stage2" / "estimator" / "sua" / "seed43", branch="estimator", seed=43),
        44: _read_seed(root / "estimator" / "sua" / "seed44", branch="estimator", seed=44, stage_label="stage2_v2"),
    }
    estimator_summary = _contrast(estimator, "POST700-T4", "WHOLE-T4")
    estimator_masks = {tuple(estimator[seed]["reliability_mask"]) for seed in plan.SEEDS}
    require(len(estimator_masks) == 1, "Stage2-v2 estimator reliability masks disagree")
    film = {seed: _read_seed(root / "film" / "sua" / f"seed{seed}", branch="film", seed=seed, stage_label="stage2_v2") for seed in plan.SEEDS}
    film_masks = {tuple(film[seed]["reliability_mask"]) for seed in plan.SEEDS}
    require(len(film_masks) == 1 and film_masks == estimator_masks, "Stage2-v2 film/estimator reliability masks disagree")
    summaries = {"semantic": _contrast(film, "SE-T4", "EMPTY"), "baseline": _contrast(film, "SE-T4", "PHASE-R"), "attachment": _contrast(film, "SE-T4", "ROW-SHUFFLE"), "product": _contrast(film, "SE-T4", "WHOLE-NATIVE"), "capacity": _contrast(film, "EMPTY", "WHOLE-NATIVE")}
    delta_b_retained = bool(next(iter(film_masks))[3])
    payload = {
        "status": "PASS", "route": plan.ROUTE_NAME, "stage": "stage2_v2_aggregate", "seeds": list(plan.SEEDS),
        "successor_incident": incident,
        "estimator": {"status": "PASS" if estimator_gate(estimator_summary) else "FAIL", "gate": bool(estimator_gate(estimator_summary)), "summary": estimator_summary, "reliability_mask": list(next(iter(estimator_masks))), "seed_terminal_sha256": {str(seed): estimator[seed]["terminal_sha256"] for seed in plan.SEEDS}, "authority": {"42": "stage2_v1", "43": "stage2_v1", "44": "stage2_v2"}},
        "film": {"status": "PASS" if film_gate(summaries["semantic"], summaries["attachment"]) else "FAIL", "gate": bool(film_gate(summaries["semantic"], summaries["attachment"])), "baseline_claim_gate": baseline_claim_gate(summaries["baseline"], delta_b_retained=delta_b_retained), "summaries": summaries, "reliability_mask": list(next(iter(film_masks))), "delta_b_retained": delta_b_retained, "seed_terminal_sha256": {str(seed): film[seed]["terminal_sha256"] for seed in plan.SEEDS}, "authority": {str(seed): "stage2_v2" for seed in plan.SEEDS}},
        "scope": "SUA_SOURCE_27_TRAIN__SUA_SOURCE_6_VALIDATION__NO_TEST_EXTERNAL_EVALAI",
    }
    digest = publish_immutable_json(root, "aggregate.json", payload)
    return {"aggregate_sha256": digest, **payload}


def aggregate_stage2_v3(repo_root: Path) -> dict[str, object]:
    """Publish the one final V1/V3 estimator and V2 FiLM aggregate."""
    repo_root = Path(repo_root).resolve()
    root = repo_root / plan.STAGE2_V3_RELATIVE
    target = root / "aggregate.json"
    require(not target.exists() and not target.with_name("aggregate.json.sha256").exists(), "Stage2-v3 aggregate already exists")
    incident = admit_stage2_v3_successor(repo_root)
    v1_root = repo_root / plan.RESULT_PARENT_RELATIVE / "stage2"
    v2_root = repo_root / plan.STAGE2_V2_RELATIVE
    estimator = {
        42: _read_seed(v1_root / "estimator" / "sua" / "seed42", branch="estimator", seed=42),
        43: _read_seed(v1_root / "estimator" / "sua" / "seed43", branch="estimator", seed=43),
        44: _read_seed(root / "estimator" / "sua" / "seed44", branch="estimator", seed=44, stage_label="stage2_v3"),
    }
    estimator_summary = _contrast(estimator, "POST700-T4", "WHOLE-T4")
    estimator_masks = {tuple(estimator[seed]["reliability_mask"]) for seed in plan.SEEDS}
    require(len(estimator_masks) == 1, "Stage2-v3 estimator reliability masks disagree")
    film = {
        seed: _read_seed(v2_root / "film" / "sua" / f"seed{seed}", branch="film", seed=seed, stage_label="stage2_v2")
        for seed in plan.SEEDS
    }
    film_masks = {tuple(film[seed]["reliability_mask"]) for seed in plan.SEEDS}
    require(len(film_masks) == 1 and film_masks == estimator_masks, "Stage2-v3 film/estimator reliability masks disagree")
    summaries = {
        "semantic": _contrast(film, "SE-T4", "EMPTY"),
        "baseline": _contrast(film, "SE-T4", "PHASE-R"),
        "attachment": _contrast(film, "SE-T4", "ROW-SHUFFLE"),
        "product": _contrast(film, "SE-T4", "WHOLE-NATIVE"),
        "capacity": _contrast(film, "EMPTY", "WHOLE-NATIVE"),
    }
    delta_b_retained = bool(next(iter(film_masks))[3])
    payload = {
        "status": "PASS", "route": plan.ROUTE_NAME, "stage": "stage2_v3_aggregate", "seeds": list(plan.SEEDS),
        "successor_incident": incident,
        "estimator": {
            "status": "PASS" if estimator_gate(estimator_summary) else "FAIL",
            "gate": bool(estimator_gate(estimator_summary)), "summary": estimator_summary,
            "reliability_mask": list(next(iter(estimator_masks))),
            "seed_terminal_sha256": {str(seed): estimator[seed]["terminal_sha256"] for seed in plan.SEEDS},
            "authority": {"42": "stage2_v1", "43": "stage2_v1", "44": "stage2_v3"},
        },
        "film": {
            "status": "PASS" if film_gate(summaries["semantic"], summaries["attachment"]) else "FAIL",
            "gate": bool(film_gate(summaries["semantic"], summaries["attachment"])),
            "baseline_claim_gate": baseline_claim_gate(summaries["baseline"], delta_b_retained=delta_b_retained),
            "summaries": summaries, "reliability_mask": list(next(iter(film_masks))),
            "delta_b_retained": delta_b_retained,
            "seed_terminal_sha256": {str(seed): film[seed]["terminal_sha256"] for seed in plan.SEEDS},
            "authority": {str(seed): "stage2_v2" for seed in plan.SEEDS},
        },
        "scope": "SUA_SOURCE_27_TRAIN__SUA_SOURCE_6_VALIDATION__NO_TEST_EXTERNAL_EVALAI",
    }
    digest = publish_immutable_json(root, "aggregate.json", payload)
    return {"aggregate_sha256": digest, **payload}
