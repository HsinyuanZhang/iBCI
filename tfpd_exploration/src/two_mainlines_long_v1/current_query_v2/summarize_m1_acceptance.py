"""Bind completed independent M1 replay audits to trained-state CPU receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    paths = {
        "full_selected": root / "shared/m1_full_selected_ema_complete_stream_v1.json",
        "full_epoch12": root / "shared/m1_full_epoch12_ema_complete_stream_v1.json",
        "t_selected": root / "shared/m1_t_selected_complete_stream_v1.json",
        "t_epoch12": root / "shared/m1_t_epoch12_complete_stream_v1.json",
    }
    replays = {key: json.loads(path.read_text()) for key, path in paths.items()}
    for key, replay in replays.items():
        if replay["status"] != "PASS" or replay["n"] != 31252 or replay["outer_query_opened"]:
            raise ValueError(f"{key}: complete source-only replay required")
        if sha(replay["stream_export"]) != replay["stream_export_sha256"]:
            raise ValueError("replayed prediction archive hash drift")
    audits = {key: value["independent_native_source_reference_audit"] for key, value in replays.items()}
    full, query = audits["full_selected"], audits["t_selected"]
    for audit in audits.values():
        for field in ("target_sha256", "frozen_dev_window_ids_sha256", "frozen_train_window_ids_sha256", "carrier_npz_sha256"):
            if audit[field] != full[field]:
                raise ValueError(f"paired source authority differs: {field}")
    delta = query["equal_session_r2"]["model"] - full["equal_session_r2"]["model"]
    session_delta = {name: value["model"] - full["per_session"][name]["model"]
                     for name, value in query["per_session"].items()}
    accuracy_pass = delta >= -.005 and min(session_delta.values()) >= -.020

    records, bindings = [], {}
    common = None
    for index in range(4):
        path = root / f"shared/m1_full_t_selected_actualapi2048_process{index}.json"
        report = json.loads(path.read_text())
        if report["initialization_only"] or report["task"] != "m1":
            raise ValueError("trained M1 benchmark required")
        identity = {key: report[key] for key in ("weights", "operator_code_sha256", "inputs", "torch", "torch_threads", "interop_threads", "affinity", "cpu")}
        if common is not None and identity != common:
            raise ValueError("benchmark state/code/input/runtime context drift across processes")
        common = identity
        for arm, replay_key in (("full", "full_selected"), ("query", "t_selected")):
            if report["weights"][arm]["file_sha256"] != replays[replay_key]["reference"]["plain_ema_model_state_sha256"]:
                raise ValueError("benchmark weights differ from independently audited selected state")
        modes = {row["mode"]: row for row in report["records"]}
        if set(modes) != {"full", "full_exact", "query_cached"}:
            raise ValueError("all matched measured modes required")
        if any(row["calls"] != 2048 or row["batch"] != 1 for row in modes.values()):
            raise ValueError("2048 actual B1 calls per measured mode required")
        base, exact, cached = (modes[key]["public_predict_wall"] for key in ("full", "full_exact", "query_cached"))
        records.append({"process": index, "timings": {key: row["public_predict_wall"] for key, row in modes.items()},
                        "E_p95_reduction_fraction": 1 - exact["p95_ms"] / base["p95_ms"],
                        "T_p95_speedup_vs_FULL_recompute": base["p95_ms"] / cached["p95_ms"],
                        "T_p95_speedup_vs_E": exact["p95_ms"] / cached["p95_ms"],
                        "E_speed_gate_pass": exact["p95_ms"] <= .8 * base["p95_ms"] and exact["p99_ms"] <= base["p99_ms"] and exact["p95_ms"] <= 15 and exact["p99_ms"] <= 20,
                        "T_speed_gate_pass": cached["p95_ms"] <= base["p95_ms"] / 3 and cached["p95_ms"] <= 15 and cached["p99_ms"] <= 20})
        bindings[str(path)] = sha(path)
    result = {
        "schema": "m1_root_independent_local_acceptance_v1",
        "status": "PASS" if accuracy_pass and all(r["E_speed_gate_pass"] and r["T_speed_gate_pass"] for r in records) else "FAIL",
        "scope": "Source-development numerical/model-speed acceptance only; no untouched outer-test, Docker/Falcon package or EvalAI claim.",
        "accuracy": {key: {"equal_session_r2": audit["equal_session_r2"], "pooled_r2": audit["pooled"], "per_session": audit["per_session"]} for key, audit in audits.items()},
        "T_retention": {"status": "PASS" if accuracy_pass else "FAIL", "equal_session_delta": delta, "minimum_allowed_equal_delta": -.005,
                         "per_session_delta": session_delta, "minimum_session_delta": min(session_delta.values()), "minimum_allowed_session_delta": -.020},
        "stream_replay": {key: {"n": value["n"], "equal_session_r2_delta": value["equal_session_r2_delta"], "max_abs_native_error": value["max_abs_error"]} for key, value in replays.items()},
        "cpu_benchmark": {"context": common, "processes": records,
                          "disclosure": "Original three-process suite 0/1/2 retained; process3 is an additive confirmation without task data-preparation work. Process0 may overlap the tail of other GPU validation/data preparation; no latency outlier was removed. All four satisfy the declared gates. Quantiles are per-process, not percentiles pooled across runs. The T >=3x comparison is to matched FULL recomputation; its gain over already-optimized E is much smaller and separately shown."},
        "receipt_sha256": {**{str(path): sha(path) for path in paths.values()}, **bindings},
        "generator_sha256": sha(__file__),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.output)
    print(json.dumps({"status": result["status"], "T_retention": result["T_retention"]}, indent=2))
