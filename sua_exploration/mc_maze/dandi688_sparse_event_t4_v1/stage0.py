"""CPU-only Stage-0 receipt assembly.  This module never writes a result root."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from . import plan
from .core import require, sha256_file
from .lifecycle import begin_attempt, close_failure, close_terminal, publish_immutable_json
from .reliability import (
    SourceReliabilityAudit,
    aggregate_column_reliability,
    assert_audit_is_disjoint,
    retention_mask,
    rowwise_pearson,
    stage0_decision,
)

if TYPE_CHECKING:
    from .descriptors import SparseEventMaterialization


def verify_frozen_inputs(repo_root: Path) -> dict[str, str]:
    """Stdlib-only Stage-0 admission hashes; must precede the attempt leaf."""
    checks = {
        plan.DESIGN_RELATIVE: plan.DESIGN_SHA256,
        plan.WORKORDER_RELATIVE: plan.WORKORDER_SHA256,
        plan.MANIFEST_RELATIVE: plan.MANIFEST_SHA256,
    }
    observed: dict[str, str] = {}
    for relative, expected in checks.items():
        location = Path(repo_root) / relative
        require(location.is_file(), f"missing frozen input {relative}")
        digest = sha256_file(location)
        require(digest == expected, f"frozen input SHA drift {relative}: {digest}")
        observed[relative] = digest
    return observed


def validate_era_counts(per_session: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    for row in per_session:
        split, era = str(row["split"]), str(row["era"])
        require(split in counts and era in plan.ERA_EXPECTED_COUNTS, "unknown split/era")
        counts[split][era] += 1
    result = {era: {split: int(counts[split][era]) for split in ("train", "val")} for era in plan.ERA_EXPECTED_COUNTS}
    require(result == plan.ERA_EXPECTED_COUNTS, f"frozen era count drift: {result}")
    return result


def _column_correlations(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], *, columns: Sequence[str] = plan.PROFILE_COLUMN_NAMES) -> dict[str, list[float]]:
    require(tuple(left) == tuple(right), "source audit roster drift")
    require(all(index < next(iter(left.values())).shape[1] for index, _ in enumerate(columns)), "requested reliability columns exceed descriptor width")
    result = {name: [] for name in columns}
    for session_id in left:
        values = rowwise_pearson(left[session_id], right[session_id])
        for index, name in enumerate(columns):
            result[name].append(float(values[index]))
    return result


def assemble_stage0_receipt(
    *,
    candidate: Mapping[str, Any],
    session_split: Mapping[str, str],
    split_half_left: Mapping[str, np.ndarray],
    split_half_right: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    estimator_reference: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Create a receipt in memory; caller owns any future immutable publication.

    ``reference`` is source-only audit data.  It is checked for memory isolation
    and omitted from the candidate handoff portion of the returned receipt.
    """
    require(set(candidate) == set(session_split), "candidate/split roster drift")
    train_ids = [session_id for session_id, split in session_split.items() if split == "train"]
    require(len(train_ids) == 27 and len(candidate) == 33, "expected exact 27+6 source roster")
    audit = SourceReliabilityAudit(split_half={key: np.asarray(split_half_left[key]) for key in train_ids}, deployment_reference={key: np.asarray(reference[key]) for key in train_ids})
    assert_audit_is_disjoint([value.raw_profile for value in candidate.values()], audit)
    split_corr = _column_correlations({key: split_half_left[key] for key in train_ids}, {key: split_half_right[key] for key in train_ids})
    reference_corr = _column_correlations({key: candidate[key].raw_profile for key in train_ids}, {key: reference[key] for key in train_ids})
    split_summary = aggregate_column_reliability(split_corr, gate="split_half")
    reference_summary = aggregate_column_reliability(reference_corr, gate="reference")
    mask = retention_mask(split_summary, reference_summary)
    # Whole/Post700 comparison uses its own whole-trial T4 reference (source
    # trials 50--109), separate from the q_SE deployment reference above.
    # Neither reference namespace is exposed in a model handoff.
    assert_audit_is_disjoint([value.whole_t4 for value in candidate.values()], SourceReliabilityAudit(split_half={}, deployment_reference={key: estimator_reference[key] for key in train_ids}))
    whole_vs_ref = _column_correlations({key: candidate[key].whole_t4[:, :2] for key in train_ids}, {key: estimator_reference[key][:, :2] for key in train_ids}, columns=("a_R", "c_R"))
    post_vs_ref = _column_correlations({key: candidate[key].post700_t4[:, :2] for key in train_ids}, {key: estimator_reference[key][:, :2] for key in train_ids}, columns=("a_R", "c_R"))
    estimator_a = float(np.nanmean(post_vs_ref["a_R"])) > float(np.nanmean(whole_vs_ref["a_R"]))
    estimator_c = float(np.nanmean(post_vs_ref["c_R"])) > float(np.nanmean(whole_vs_ref["c_R"]))
    per_session = []
    for session_id, materialized in candidate.items():
        row = dict(materialized.receipt)
        row["split"] = session_split[session_id]
        per_session.append(row)
    era_counts = validate_era_counts(per_session)
    return {
        "schema_version": plan.SCHEMA_VERSION, "route": plan.ROUTE_NAME, "stage": "stage0_cpu_constructibility_information_audit",
        "gpu_opened": False, "decoder_opened": False, "performance_metric_opened": False,
        "axis_contract": {"candidate_pool_n": plan.CANDIDATE_POOL_N, "activity_support_n": plan.ACTIVITY_SUPPORT_N, "carrier_profile_label_horizon": plan.CARRIER_PROFILE_LABEL_HORIZON, "query_start_trial": plan.QUERY_START_TRIAL, "independently_represented": True},
        "per_session": per_session, "era_counts": era_counts,
        "reliability": {"split_half": split_summary, "deployment_reference": reference_summary, "mask": list(mask)},
        "estimator_reference_comparison": {"whole": whole_vs_ref, "post700": post_vs_ref, "post700_exceeds_whole_a": estimator_a, "post700_exceeds_whole_c": estimator_c},
        "decision": stage0_decision(estimator_a_pass=estimator_a, estimator_c_pass=estimator_c, mask=mask),
        "audit_arrays_not_in_candidate_handoff": True,
    }


def _manifest_roster(repo_root: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    manifest = json.loads((Path(repo_root) / plan.MANIFEST_RELATIVE).read_text())
    splits = manifest["session_splits"]
    require(len(splits["train"]) == 27 and len(splits["val"]) == 6 and len(splits["test"]) == 6, "strict manifest counts drift")
    mapping = {name: split for split in ("train", "val") for name in splits[split]}
    return mapping, splits


def _source_path(repo_root: Path, session_id: str) -> Path:
    found = sorted((Path(repo_root) / plan.DATA_RELATIVE).glob(f"{session_id}_behavior+ecephys.nwb"))
    require(len(found) == 1, f"expected one source NWB for {session_id}, got {len(found)}")
    return found[0]


def execute_stage0(repo_root: Path, result_root: Path, *, materializer=None) -> dict[str, object]:
    """Formal source-only Stage-0 executor; call only after root admission.

    It creates `attempt.json` before the first NWB materializer call, and any
    failure is terminalized in the same otherwise immutable root.
    """
    repo_root, result_root = Path(repo_root).resolve(), Path(result_root).resolve()
    static = verify_frozen_inputs(repo_root)
    session_split, splits = _manifest_roster(repo_root)
    attempt_sha = begin_attempt(result_root, {"schema_version": plan.SCHEMA_VERSION, "route": plan.ROUTE_NAME, "stage": "stage0", "static_inputs": static, "source_roster": {"train": splits["train"], "val": splits["val"], "test_not_opened": splits["test"]}, "decoder_opened": False, "gpu_opened": False, "test_opened": False, "external_opened": False})
    try:
        # Import reviewed CPU primitives only after the immutable attempt.  They
        # may transitively import CPU Torch/Lightning but never construct a
        # decoder, load a checkpoint, or initialize CUDA in this path.
        if materializer is None:
            from .descriptors import materialize_sparse_event_t4
            materializer = materialize_sparse_event_t4
        candidate: dict[str, Any] = {}
        split_left: dict[str, np.ndarray] = {}
        split_right: dict[str, np.ndarray] = {}
        reference: dict[str, np.ndarray] = {}
        estimator_reference: dict[str, np.ndarray] = {}
        for session_id, split in session_split.items():
            source = _source_path(repo_root, session_id)
            candidate[session_id] = materializer(source, signal_view="sua", namespace="candidate")
            if split == "train":
                left = materializer(source, signal_view="sua", support_positions=tuple(range(0, 30, 2)), namespace="reliability_audit")
                right = materializer(source, signal_view="sua", support_positions=tuple(range(1, 30, 2)), namespace="reliability_audit")
                ref = materializer(source, signal_view="sua", support_positions=tuple(range(50, 110)), namespace="reliability_audit")
                split_left[session_id], split_right[session_id] = left.raw_profile.copy(), right.raw_profile.copy()
                reference[session_id], estimator_reference[session_id] = ref.raw_profile.copy(), ref.whole_t4.copy()
        receipt = assemble_stage0_receipt(candidate=candidate, session_split=session_split, split_half_left=split_left, split_half_right=split_right, reference=reference, estimator_reference=estimator_reference)
        stage_sha = publish_immutable_json(result_root, "stage0.json", receipt)
        close_terminal(result_root, attempt_sha256=attempt_sha, payload={"stage0_sha256": stage_sha, "decision": receipt["decision"], "root_rehash_required_before_next_stage": True})
        return receipt
    except BaseException as error:
        close_failure(result_root, attempt_sha256=attempt_sha, error=error)
        raise
