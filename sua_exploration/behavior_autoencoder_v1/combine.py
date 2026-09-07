"""Combine disjoint source-selected manifold families without target selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import canonical_sha256, need


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def combine_family_results(paths: Sequence[Path]) -> dict[str, Any]:
    need(len(paths) >= 2 and len(set(paths)) == len(paths), "need distinct family results")
    families: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        need(path.is_file() and not path.is_symlink(), f"missing family result: {path}")
        body = json.loads(path.read_text(encoding="utf-8"))
        need(body.get("schema") == "m1_m10_source_frozen_behavior_manifold_nested_loso_v1", "family schema drift")
        need(body.get("target_support_or_query_used_for_candidate_selection") is False, "family leaked target selection")
        families.append((path, body))
    reference = families[0][1]
    sessions = tuple(reference["sessions"]); output_names = tuple(reference["output_names"])
    for _, body in families[1:]:
        need(tuple(body["sessions"]) == sessions and tuple(body["output_names"]) == output_names, "family surface drift")
        need(body["directridge_equal_session"] == reference["directridge_equal_session"], "family baseline drift")
        need(body["lambda_grid_per_sample"] == reference["lambda_grid_per_sample"], "family lambda grid drift")
    chosen: dict[str, Any] = {}; scores: list[float] = []; baseline_scores: list[float] = []
    pooled_sse = np.zeros(len(output_names), dtype=np.float64)
    pooled_tss = np.zeros(len(output_names), dtype=np.float64)
    all_spec_names: set[str] = set()
    family_bindings: list[dict[str, Any]] = []
    for path, body in families:
        family_specs = {row["name"] if "name" in row else (
            f"pca_q{row['latent_dim']}" if row["kind"] == "pca" else
            f"mlp_q{row['latent_dim']}_h{row['hidden_dim']}_{row['activation']}_raw{str(row['raw_loss_fraction']).replace('.', 'p')}"
        ) for row in body["candidate_specs"]}
        need(not all_spec_names.intersection(family_specs), "family candidate overlap")
        all_spec_names.update(family_specs)
        family_bindings.append({"path": str(path.resolve()), "sha256": _sha_file(path), "result_content_sha256": canonical_sha256(body), "spec_names": sorted(family_specs)})
    for session in sessions:
        alternatives = []
        for family_index, (path, body) in enumerate(families):
            fold = body["folds"][session]
            alternatives.append({
                "family_index": family_index, "path": str(path.resolve()),
                "selected": fold["selected"], "target_metrics": fold["target_metrics"],
                "baseline_target_metrics": fold["baseline_target_metrics"],
                "final_manifold_fit": fold["final_manifold_fit"],
                "input_path": fold["input_path"], "input_sha256": fold["input_sha256"],
            })
        # Source validation score only. Family order is the deterministic tie break.
        winner_index = max(range(len(alternatives)), key=lambda index: (alternatives[index]["selected"]["equal_source_session_mean_r2"], -index))
        winner = alternatives[winner_index]
        metric = winner["target_metrics"]; baseline = winner["baseline_target_metrics"]
        score = float(metric["pooled_variance_weighted_r2"])
        baseline_score = float(baseline["pooled_variance_weighted_r2"])
        scores.append(score); baseline_scores.append(baseline_score)
        pooled_sse += np.asarray(metric["sse_per_output"], dtype=np.float64)
        pooled_tss += np.asarray(metric["tss_per_output"], dtype=np.float64)
        chosen[session] = {
            "selected_by_source_validation_only": True,
            "winner": winner,
            "family_alternatives": [{"family_index": row["family_index"], "path": row["path"], "selected": row["selected"]} for row in alternatives],
            "delta_vs_directridge": score - baseline_score,
        }
    scores_array = np.asarray(scores); baseline_array = np.asarray(baseline_scores); delta = scores_array - baseline_array
    per_output_r2 = 1.0 - pooled_sse / pooled_tss
    return {
        "schema": "m1_m10_source_frozen_behavior_manifold_combined_selection_v1",
        "status": "COMPLETE_SOURCE_ONLY_COMBINED_SELECTION",
        "selection": "best already-inner-selected family per outer fold using equal-source-session validation R2 only",
        "target_used_for_selection": False,
        "families": family_bindings,
        "candidate_spec_count": len(all_spec_names),
        "sessions": list(sessions), "output_names": list(output_names), "folds": chosen,
        "equal_session": {"mean_r2": float(scores_array.mean()), "median_r2": float(np.median(scores_array)), "per_session_r2": dict(zip(sessions, scores_array.tolist()))},
        "directridge_equal_session": reference["directridge_equal_session"],
        "paired_delta_vs_directridge": {"mean": float(delta.mean()), "median": float(np.median(delta)), "positive_sessions": int((delta > 0.0).sum()), "total_sessions": len(sessions), "per_session": dict(zip(sessions, delta.tolist()))},
        "pooled_from_session_sufficient_statistics": {
            "definition": "1-sum_session,output(SSE)/sum_session,output(TSS); chosen family target predictions",
            "sse_per_output": pooled_sse.tolist(), "tss_per_output": pooled_tss.tolist(),
            "r2_per_output": per_output_r2.tolist(),
            "pooled_variance_weighted_r2": 1.0 - float(pooled_sse.sum() / pooled_tss.sum()),
            "equal_output_mean_r2": float(per_output_r2.mean()),
        },
    }

