#!/usr/bin/env python3
"""Run exactly one non-scientific Track-B v2 strict27 SUA CPU fit.

This engineering-only command has no target, formal, score, candidate, grid,
fold, or geometry option.  It revalidates the six settled source authorities,
materialises the canonical first source pseudo-target fold, and executes only
the frozen d=3/iterations=250/seed=42 joint fit.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
sys.path[:] = [entry for entry in sys.path if "/.local/lib/python" not in entry]

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
VENDOR = REPO_ROOT / "cebra_exploration" / "third_party" / "cebra"
for _entry in (str(SRC), str(VENDOR), str(REPO_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np  # noqa: E402
import track_b_v2_actual_cpu_route as route  # noqa: E402
import track_b_v2_source_adapter as source  # noqa: E402


AUTHORITY_ROOT = (
    REPO_ROOT / "cebra_exploration" / "results"
    / "track_b_v2_source_authority_20260814_strict27_sua_continuous_v2_dev"
)
AUTHORITY_FILES = {
    "behavior": ("source_behavior_auxiliary_scaler_authority.json",
                 "53e64c55da7b839e018a3ae87e3b2979f752b41aaa3c42cda98ace7f0d0358e7",
                 "track_b_v2_source_behavior_auxiliary_scaler_authority_v1"),
    "coverage": ("source_coverage.json",
                 "3828e766f21f5a2b8fc5d9ebcf9e73cef9a0cf7152b7fdc25b37b60a1805f5c7",
                 "track_b_v2_source_coverage_receipt_v1"),
    "neural": ("source_neural_input_authority.json",
               "e860b4a05f3b1de4d6f5f3af0da51bf9da8a7645779eadb4ff64734786db2983",
               "track_b_v2_source_neural_input_authority_v1"),
    "selector_plan": ("source_only_dual_geometry_selection_plan.json",
                      "7d51cfd0d2d1359acaada0a126b4f8695762c729fdbd4bd2bc9303d8fea19f3d",
                      "track_b_v2_source_only_dual_geometry_execution_plan_v1"),
    "embedding": ("source_readout_embedding_identity_authority.json",
                  "ebc6f09c3af9516245496c1454e9ca2a4ca059a78588aae1ca22eeb9dbd3e1cd",
                  "track_b_v2_source_readout_embedding_identity_authority_v1"),
    "roster": ("source_roster.json",
               "f812ad5dab601b230d72c91770d6e18864f77cb070d376a3fa260acd45ce4d92",
               "track_b_v2_source_roster_receipt_v1"),
}
HELD_SOURCE_ID = "sub-C_ses-CO-20131003"
FOLD_ID = f"source_pseudo_target_support_query__{HELD_SOURCE_ID}"
SUPPORT_TRIALS = 50
GEOMETRY = route.Geometry(3, 250)
SEED = 42


def _payload_sha(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    declared = body.pop("receipt_payload_sha256", None)
    observed = route.sha256_bytes(route.canonical_json_bytes(body))
    route.require(declared == observed, "authority internal payload SHA drift")
    return observed


def load_settled_authorities() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for label, (filename, expected_sha, schema) in AUTHORITY_FILES.items():
        payload, binding = route.load_immutable_json((AUTHORITY_ROOT / filename).absolute(), expected_schema=schema)
        route.require(binding["sha256"] == expected_sha, f"settled {label} authority body SHA drift")
        route.require(payload.get("dataset") == "subject_m" and payload.get("view") == "sua",
                      f"settled {label} authority scope drift")
        route.require(payload.get("target_data_opened") is False and
                      payload.get("target_query_opened") is False,
                      f"settled {label} authority reports target access")
        route.require(payload.get("gpu_used") is False and payload.get("score_emitted") is False,
                      f"settled {label} authority reports forbidden execution")
        _payload_sha(payload)
        payloads[label] = payload
        bindings[label] = binding
    roster = payloads["roster"]["source_session_ids"]
    route.require(isinstance(roster, list) and len(roster) == 27 and roster[0] == HELD_SOURCE_ID,
                  "strict27 SUA roster/canonical first fold drift")
    plan = payloads["selector_plan"]["inner_source_folds"]
    route.require(isinstance(plan, list) and len(plan) == 27 and
                  plan[0].get("inner_fold_id") == FOLD_ID and
                  plan[0].get("pseudo_target_source_session_id") == HELD_SOURCE_ID and
                  plan[0].get("pseudo_target_support", {}).get("support_budget_trials") == SUPPORT_TRIALS,
                  "canonical first pseudo-target plan drift")
    return payloads, bindings


def _route_array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "route_header_plus_bytes_sha256": route.array_sha256(array),
            "raw_contiguous_bytes_sha256": route.raw_array_sha256(array)}


def materialize_first_fold(payloads: Mapping[str, Mapping[str, Any]]) -> tuple[
        route.SourcePseudoTargetFold, dict[str, Any], tuple[Any, ...]]:
    roster = tuple(payloads["roster"]["source_session_ids"])
    request, rows = source.materialize_canonical_source_sessions(
        dataset="subject_m", view="sua", source_session_ids=roster, source_only_smoke=False)
    route.require(tuple(request["source_session_ids"]) == roster, "materializer source order drift")
    route.require(request["canonical_loader_behavior_scaler"] ==
                  payloads["behavior"]["behavior_auxiliary_scaler"],
                  "materialized behavior scaler differs from settled authority")
    expected_coverage = payloads["coverage"]["source_sessions"]
    route.require(len(rows) == len(expected_coverage) == 27, "materialized strict27 coverage count drift")
    for row, expected in zip(rows, expected_coverage, strict=True):
        route.require(row.as_coverage_dict() == expected,
                      f"materialized source coverage differs for {row.session_id}")

    held = rows[0]
    route.require(held.session_id == HELD_SOURCE_ID, "materialized held-source order drift")
    evaluator = importlib.import_module("scripts.eval_adaptation_dandi688")
    scaler = payloads["behavior"]["behavior_auxiliary_scaler"]
    mean = np.asarray(scaler["mean_float32"], dtype=np.float32)
    std = np.asarray(scaler["std_float32"], dtype=np.float32)
    record = evaluator.load_session_with_trials(
        Path(held.source_path), 20, 50, 50, 100, -1.0, mean, std,
        trial_result_filter="R", signal_view="sua")
    route.require(record.get("name") == HELD_SOURCE_ID and record.get("signal_view") == "sua",
                  "held-source replay scope drift")
    neural = np.asarray(record["neural"], dtype=np.float32)
    behavior = np.asarray(record["behavior"], dtype=np.float32)
    route.require(np.array_equal(neural, held.neural) and np.array_equal(behavior, held.dense_behavior),
                  "held-source replay arrays differ from strict27 materialization")
    trials = record["trials"]
    route.require(isinstance(trials, list) and len(trials) == held.source_trial_count and
                  len(trials) > SUPPORT_TRIALS, "held-source rewarded trial roster drift")
    boundaries = [(int(trial["trial_index"]), int(trial["start"]), int(trial["stop"])) for trial in trials]
    route.require(all(0 <= start < stop <= neural.shape[0] for _, start, stop in boundaries),
                  "held-source trial boundary out of range")
    route.require(all(boundaries[i][1] <= boundaries[i + 1][1] for i in range(len(boundaries) - 1)),
                  "held-source rewarded trials are not chronological")
    support_start = 0
    support_stop = boundaries[SUPPORT_TRIALS - 1][2]
    query_start = boundaries[SUPPORT_TRIALS][1]
    query_stop = boundaries[-1][2]
    route.require(0 < support_stop <= query_start < query_stop <= neural.shape[0],
                  "support/query continuous boundary invalid")
    peer_rows = rows[1:]
    fold = route.SourcePseudoTargetFold(
        fold_id=FOLD_ID, held_source_session_id=HELD_SOURCE_ID,
        peer_source_session_ids=tuple(row.session_id for row in peer_rows),
        peer_neural=tuple(row.neural for row in peer_rows),
        peer_auxiliary=tuple(row.dense_behavior for row in peer_rows),
        held_support_neural=np.ascontiguousarray(neural[support_start:support_stop]),
        held_support_auxiliary=np.ascontiguousarray(behavior[support_start:support_stop]),
        held_query_neural=np.ascontiguousarray(neural[query_start:query_stop]),
        held_query_auxiliary=np.ascontiguousarray(behavior[query_start:query_stop]),
        support_trial_count=SUPPORT_TRIALS, expected_support_trial_count=SUPPORT_TRIALS).validated()
    boundary_array = np.asarray(boundaries, dtype="<i8")
    proof = {
        "fold_id": FOLD_ID,
        "held_source_session_id": HELD_SOURCE_ID,
        "peer_source_session_ids": list(fold.peer_source_session_ids),
        "peer_full_continuous_session_count": len(peer_rows),
        "peer_full_continuous_total_rows": sum(row.neural.shape[0] for row in peer_rows),
        "support": {"start_inclusive": support_start, "stop_exclusive": support_stop,
                    "rewarded_trial_count": SUPPORT_TRIALS,
                    "all_intervening_raw_rows_included": True,
                    "arrays": {"neural": _route_array_record(fold.held_support_neural),
                               "auxiliary": _route_array_record(fold.held_support_auxiliary)}},
        "query": {"start_inclusive": query_start, "stop_exclusive": query_stop,
                  "rewarded_trial_count": len(trials) - SUPPORT_TRIALS,
                  "strictly_after_M50": True, "all_intervening_raw_rows_included": True,
                  "arrays": {"neural": _route_array_record(fold.held_query_neural),
                             "auxiliary": _route_array_record(fold.held_query_auxiliary)}},
        "support_query_gap_rows": query_start - support_stop,
        "rewarded_trial_boundary_array": _route_array_record(boundary_array),
        "held_query_neural_in_fit": False,
        "held_query_auxiliary_in_fit": False,
    }
    return fold, proof, rows


def execute(output: Path) -> dict[str, Any]:
    output = output.expanduser().absolute()
    sidecar = output.with_name(f"{output.name}.sha256")
    route.require(not os.path.lexists(output) and not os.path.lexists(sidecar),
                  "microbenchmark output body/sidecar must be fresh before authority or data access")
    script = Path(__file__).resolve()
    core = Path(route.__file__).resolve()
    adapter = Path(source.__file__).resolve()
    evaluator = REPO_ROOT / "sua_exploration" / "scripts" / "eval_adaptation_dandi688.py"
    multisession = REPO_ROOT / "sua_exploration" / "mc_maze" / "multisession_datamodule.py"
    launch_closure = route.snapshot_file_closure({
        "actual_cpu_route": core,
        "entrypoint": script,
        "source_adapter": adapter,
        "canonical_evaluator": evaluator,
        "multisession_datamodule": multisession,
    })
    started = time.monotonic()
    payloads, bindings = load_settled_authorities()
    authority_seconds = time.monotonic() - started
    materialize_started = time.monotonic()
    fold, fold_proof, rows = materialize_first_fold(payloads)
    materialize_seconds = time.monotonic() - materialize_started

    backend = route.VendoredCebra061Backend()
    fit_started = time.monotonic()
    run = backend.run_arm(
        arm="cebra_joint_behavior", peer_neural=fold.peer_neural,
        peer_auxiliary=fold.peer_auxiliary,
        target_support_neural=fold.held_support_neural,
        target_support_auxiliary=fold.held_support_auxiliary,
        target_query_neural=fold.held_query_neural,
        geometry=GEOMETRY, seed=SEED, transform_peers=False)
    fit_transform_seconds = time.monotonic() - fit_started
    route.require(len(run.fit_calls) == 1 and run.fit_calls[0]["label"] == "joint_multisession_fit",
                  "microbenchmark must execute exactly one CEBRA.fit call")
    route.require(run.peer_fit_embeddings == (), "microbenchmark unexpectedly transformed peer source arrays")
    route.require(not run.source_query_neural_seen_by_fit and not run.source_query_auxiliary_seen_by_fit,
                  "held-source query entered fit")
    alignment = route.derive_contiguous_query_alignment(
        model_alignment=run.model_alignment,
        input_start=fold_proof["query"]["start_inclusive"],
        input_stop=fold_proof["query"]["stop_exclusive"],
        embedding_row_count=run.target_query_embedding.shape[0],
        support_stop=fold_proof["support"]["stop_exclusive"])
    session_records = [{"session_id": row.session_id, "source_path": row.source_path,
                        "source_nwb_sha256": row.source_nwb_sha256,
                        "neural": _route_array_record(row.neural),
                        "dense_behavior": _route_array_record(row.dense_behavior),
                        "source_trial_count": row.source_trial_count} for row in rows]
    import torch
    route.require(not torch.cuda.is_initialized(), "source microbenchmark initialized CUDA")
    total_seconds = time.monotonic() - started
    receipt = {
        "schema": route.SCHEMA_SOURCE_MICROBENCH,
        "status": "ENGINEERING_MICROBENCH_NOT_SCIENTIFIC",
        "official": False,
        "scientific_score_emitted": False,
        "selector_candidate_emitted": False,
        "selector_winner_selected": False,
        "dataset": "subject_m", "view": "sua",
        "source_only": True, "strict27_source_session_count": 27,
        "outer_target_discovered": False, "outer_target_path_resolved": False,
        "outer_target_opened": False, "formal_data_opened": False,
        "gpu_used": False, "cuda_initialized": False,
        "geometry": GEOMETRY.as_dict(), "selector_seed": SEED,
        "cebra_fit_call_count": len(run.fit_calls), "cebra_fit_calls": list(run.fit_calls),
        "query_neural_in_fit": False, "query_auxiliary_in_fit": False,
        "settled_source_authorities": bindings,
        "settled_source_authority_payload_sha256": {
            label: payload["receipt_payload_sha256"] for label, payload in payloads.items()},
        "source_fold": fold_proof,
        "source_sessions": session_records,
        "embedding_outputs": {
            "support": _route_array_record(run.target_support_embedding),
            "query_full_padded": _route_array_record(run.target_query_embedding),
            "query_valid_alignment_authority": alignment,
        },
        "runtime": {
            "authority_validation_wall_clock_s": authority_seconds,
            "strict27_materialization_and_held_replay_wall_clock_s": materialize_seconds,
            "single_fit_plus_support_query_transform_wall_clock_s": fit_transform_seconds,
            "total_wall_clock_s": total_seconds,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "os_cpu_count": os.cpu_count(),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE")},
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "python_executable": sys.executable,
        },
        "backend_identity": dict(backend.identity),
        "implementation_bindings_at_launch": launch_closure,
        "launch_snapshot_exact_equal_to_final_live": True,
        "economic_use_only": True,
        "prohibited_interpretation": "No R2, source winner, model-quality, or target claim may be inferred.",
    }
    route.require_file_closure_unchanged(launch_closure)
    published = route.write_immutable_pair(output, receipt)
    return {"receipt": receipt, "published": published}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = execute(args.output)
    receipt = result["receipt"]
    print(json.dumps({"status": receipt["status"], "published": result["published"],
                      "cebra_fit_call_count": receipt["cebra_fit_call_count"],
                      "runtime": receipt["runtime"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
