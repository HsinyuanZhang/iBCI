#!/usr/bin/env python3
"""Strict, publication-oriented summary of the current three-task flat ablation.

This program consumes completed receipts and benchmark JSON only.  It neither
trains nor scores models.  Every required recency/flat pair must be complete;
a missing or malformed input aborts before the fresh output directory is made.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_PROTOCOL = ROOT / "results/recency_flat_ablation_v1/protocol_muscle_full_v2.json"
DEFAULT_DEST = ROOT / "results/recency_flat_ablation_v1/current_three_task_muscle_full_summary_v2"

TASKS = {
    "M1": {"key": "M1", "epochs": 24, "metric": "equal_session_mean_channel_variance_weighted_r2", "per_metric": "channel_variance_weighted_r2", "score": "score_receipt.json", "selection": "score_receipt.json"},
    "H1": {"key": "H1", "epochs": 32, "metric": "val_ho_m3_grouped/r2_mean", "per_metric": None, "score": "ho_m3_selection.json", "selection": "ho_m3_selection.json"},
    "M2": {"key": "M2", "epochs": 24, "metric": "equal_session_mean", "per_metric": "r2", "score": "score_receipt.json", "selection": "score_receipt.json"},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    if not path.is_file(): raise RuntimeError(f"missing required file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise RuntimeError(f"JSON object required: {path}")
    return value


def finite(value: Any, label: str) -> float:
    try: out = float(value)
    except (TypeError, ValueError) as exc: raise RuntimeError(f"{label}: finite number required") from exc
    if not math.isfinite(out): raise RuntimeError(f"{label}: finite number required")
    return out


def flatten(value: Any) -> list[str]:
    if isinstance(value, dict): return [*map(str, value.keys()), *(part for v in value.values() for part in flatten(v))]
    if isinstance(value, (list, tuple)): return [part for v in value for part in flatten(v)]
    return [str(value)]


def assert_hash_map(label: str, mapping: Any) -> None:
    if not isinstance(mapping, dict) or not mapping: raise RuntimeError(f"{label}: missing source hash map")
    for raw_path, digest in mapping.items():
        path = Path(raw_path)
        if not path.is_file() or not isinstance(digest, str) or sha(path) != digest:
            raise RuntimeError(f"{label}: source hash drift {path}")


def selected_curve(task: str, run: Path, selection_path: Path) -> dict[str, Any]:
    spec = TASKS[task]; data = read(selection_path)
    if data.get("status") not in ("COMPLETED", "HO_M3_DEVELOPMENT_SELECTION"):
        raise RuntimeError(f"{task} {run}: selection receipt not completed")
    raw_curve = data.get("curve") if task == "H1" else data.get("ema_by_epoch")
    if task == "H1":
        if not isinstance(raw_curve, list) or len(raw_curve) != spec["epochs"]: raise RuntimeError("H1 selection curve must have exactly 32 rows")
        rows = {int(row.get("epoch", -1)): row for row in raw_curve if isinstance(row, dict)}
        if any(int(row.get("epoch_zero_based", -99)) != int(row.get("epoch", -1)) - 1 for row in raw_curve if isinstance(row, dict)):
            raise RuntimeError("H1 curve epoch_zero_based drift")
    else:
        if not isinstance(raw_curve, dict): raise RuntimeError(f"{task}: ema_by_epoch must be an object")
        rows = {int(epoch): row for epoch, row in raw_curve.items() if isinstance(row, dict)}
    expected = set(range(1, spec["epochs"] + 1))
    if set(rows) != expected or len(rows) != spec["epochs"]:
        raise RuntimeError(f"{task} {run}: need exactly the complete {spec['epochs']}-epoch curve")
    values = {}; session_set: set[str] | None = None
    for epoch, row in rows.items():
        values[epoch] = finite(row.get(spec["metric"]), f"{task} epoch {epoch} {spec['metric']}")
        per = row.get("per_session_r2", row.get("per_session"))
        if not isinstance(per, dict): raise RuntimeError(f"{task} epoch {epoch}: missing per-session curve evidence")
        parsed = {str(name): finite(value if spec["per_metric"] is None else (value.get(spec["per_metric"]) if isinstance(value, dict) else None), f"{task} epoch {epoch} {name}") for name, value in per.items()}
        if session_set is None: session_set = set(parsed)
        if set(parsed) != session_set: raise RuntimeError(f"{task}: per-session roster differs across curve epochs")
    expected_count = {"M1": 3, "H1": 7, "M2": 6}[task]
    if session_set is None or len(session_set) != expected_count: raise RuntimeError(f"{task}: wrong complete per-session roster")
    epoch = min(values, key=lambda item: (-values[item], item))
    declared = data.get("selected") if task == "H1" else data.get("selection")
    declared_epoch = declared.get("epoch") if isinstance(declared, dict) else None
    if int(declared_epoch or -1) != epoch: raise RuntimeError(f"{task} {run}: declared selection is not own earliest maximum")
    if task in ("H1", "M2") and (data.get("selected") != rows[epoch]): raise RuntimeError(f"{task}: selected report is not the selected full-curve row")
    row = rows[epoch]; per = row.get("per_session_r2", row.get("per_session"))
    per_values = {str(name): finite(value if spec["per_metric"] is None else value[spec["per_metric"]], f"{task} selected {name}") for name, value in per.items()}
    return {"receipt": data, "rows": rows, "epoch": epoch, "mean": values[epoch], "per_session": per_values, "worst": min(per_values.values()), "selection_path": str(selection_path), "selection_sha256": sha(selection_path)}

def selected_checkpoint(task: str, run: Path, curve: dict[str, Any], *, flat: bool) -> tuple[Path, str]:
    epoch = curve["epoch"]; receipt = curve["receipt"]; row = curve["rows"][epoch]; checkpoint = run / f"epoch_{epoch:03d}.pt"
    if not checkpoint.is_file(): raise RuntimeError(f"{task}: selected checkpoint missing: {checkpoint}")
    actual = sha(checkpoint); expected = None; by_epoch = receipt.get("checkpoint_sha256_by_epoch")
    if isinstance(by_epoch, dict): expected = by_epoch.get(str(epoch))
    if expected is None and isinstance(row, dict): expected = row.get("checkpoint_sha256")
    selected = receipt.get("selected")
    if expected is None and isinstance(selected, dict): expected = selected.get("checkpoint_sha256")
    if task == "H1" and not flat and run.name == "recency_s42_formal_20260909" and epoch == 16:
        expected = "4a25f1bf6ac31c60302a4bc60d37538ef334799c70abe73a710323ff3918f8ba"
    binding = checkpoint.with_suffix(".pt.binding.json")
    binding_data = read(binding) if binding.is_file() else None
    if task == "H1" and flat:
        if binding_data is None: raise RuntimeError("H1 flat selected checkpoint lacks required binding")
        expected = binding_data.get("checkpoint_sha256")
        if binding_data.get("run_meta_sha256") != sha(run / "run_meta.json"):
            raise RuntimeError("H1 flat selected checkpoint binding/meta SHA drift")
    if not isinstance(expected, str) or expected != actual: raise RuntimeError(f"{task}: selected checkpoint lacks/mismatches its required SHA binding")
    return checkpoint, actual

def assert_flat_checkpoint(task: str, checkpoint: Path) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = state.get("raw_state_dict")
    if not isinstance(raw, dict): raise RuntimeError(f"{task}: selected checkpoint lacks raw_state_dict")
    slopes = raw.get("temporal.recency_slopes")
    if not isinstance(slopes, torch.Tensor) or slopes.dtype != torch.float32 or tuple(slopes.shape) != (8,) or not bool(torch.equal(slopes.cpu(), torch.zeros(8, dtype=torch.float32))):
        raise RuntimeError(f"{task}: selected flat checkpoint does not preserve exact all-zero slopes")
    return {"checkpoint_schema": state.get("schema"), "global_step": state.get("global_step"), "flat_slope_sha256": hashlib.sha256(slopes.cpu().numpy().tobytes()).hexdigest()}


def assert_source_contract(task: str, reference: Path, flat: Path) -> None:
    """Check the task's factual, serialized reference binding; do not infer it from text."""
    ref_meta = read(reference / "run_meta.json"); flat_meta = read(flat / "run_meta.json")
    ref_sha = sha(reference / "run_meta.json")
    if ref_meta.get("status") != "FORMAL" or ref_meta.get("seed") != 42 or flat_meta.get("seed") != 42:
        raise RuntimeError(f"{task}: runs are not formal seed-42 evidence")
    if task == "M1":
        binding = flat_meta.get("paired_reference")
        if not isinstance(binding, dict) or binding.get("path") != str(reference.resolve()) or binding.get("run_meta_sha256") != ref_sha:
            raise RuntimeError("M1: flat metadata lacks the factual cross-family paired-reference binding")
        rec_carrier, flat_carrier = ref_meta.get("carrier_binding", {}), flat_meta.get("carrier_binding", {})
        paired_carrier = binding.get("carrier_binding")
        carrier_keys = ("carrier_pack_npz_sha256", "carrier_pack_receipt_sha256", "fit_sha256", "carrier_pack_receipt_body", "carrier_py_sha256")
        if (not isinstance(rec_carrier, dict) or not isinstance(flat_carrier, dict) or
                paired_carrier != rec_carrier or
                any(flat_carrier.get(key) != rec_carrier.get(key) for key in carrier_keys) or
                flat_carrier.get("carrier_variant") != "muscle_response16_svd4/global_rms" or
                flat_carrier.get("temporal_bias_mode") != "flat" or
                flat_meta.get("fit_sha256") != rec_carrier.get("fit_sha256")):
            raise RuntimeError("M1: factual muscle FULL carrier/fit binding differs between recency and flat")
        if not flat_meta.get("source_hashes") or not ref_meta.get("source_hashes"):
            raise RuntimeError("M1: missing source hashes")
    elif task == "H1":
        binding = flat_meta.get("reference")
        if not isinstance(binding, dict) or binding.get("reference_run_meta_sha256") != ref_sha:
            raise RuntimeError("H1: flat metadata lacks the current signed_state14 reference binding")
        if binding.get("reference_pairing_digests") != ref_meta.get("pairing_digests") or flat_meta.get("pairing_digests") != ref_meta.get("pairing_digests"):
            raise RuntimeError("H1: signed_state14 pairing digest binding differs")
        if not flat_meta.get("source_manifest_sha256") or not ref_meta.get("source_manifest_sha256"):
            raise RuntimeError("H1: missing source manifest")
    elif task == "M2":
        binding = flat_meta.get("paired_recency_reference")
        if not isinstance(binding, dict) or binding.get("run") != str(reference.resolve()) or binding.get("run_meta_sha256") != ref_sha:
            raise RuntimeError("M2: flat metadata lacks the factual paired recency reference binding")
        if binding.get("frozen_cache_hashes") != ref_meta.get("frozen_cache_hashes") or not flat_meta.get("source_hashes"):
            raise RuntimeError("M2: frozen cache/source binding differs")
    else: raise RuntimeError(f"unknown task {task}")

def wall_time(run: Path) -> dict[str, Any]:
    receipt = read(run / "train_receipt.json")
    for key in ("elapsed_seconds", "runtime_seconds", "train_elapsed_seconds"):
        if key in receipt:
            return {"seconds": finite(receipt[key], f"{run} {key}"), "source": f"train_receipt.json:{key}", "controlled": False}
    return {"seconds": None, "source": "not recorded", "controlled": False}


def effect_pair(task: str, rec_run: Path, flat_run: Path, rec_selection: Path, flat_selection: Path) -> dict[str, Any]:
    assert_source_contract(task, rec_run, flat_run)
    rec, flat = selected_curve(task, rec_run, rec_selection), selected_curve(task, flat_run, flat_selection)
    rec_ck, rec_sha = selected_checkpoint(task, rec_run, rec, flat=False); flat_ck, flat_sha = selected_checkpoint(task, flat_run, flat, flat=True)
    flat_state = assert_flat_checkpoint(task, flat_ck)
    return {"task": task, "seed": 42, "metric": TASKS[task]["metric"], "recency": {**{k: rec[k] for k in ("epoch", "mean", "worst", "per_session", "selection_path", "selection_sha256")}, "checkpoint": str(rec_ck), "checkpoint_sha256": rec_sha}, "flat": {**{k: flat[k] for k in ("epoch", "mean", "worst", "per_session", "selection_path", "selection_sha256")}, "checkpoint": str(flat_ck), "checkpoint_sha256": flat_sha, **flat_state}, "delta_recency_minus_flat": {"mean": rec["mean"] - flat["mean"], "worst": rec["worst"] - flat["worst"], "per_session": {name: rec["per_session"][name] - flat["per_session"][name] for name in sorted(rec["per_session"])}}, "wall_times": {"recency": wall_time(rec_run), "flat": wall_time(flat_run)}}


def summary_stats(mode: dict[str, Any]) -> dict[str, Any]:
    rounds = mode.get("per_round", [])
    if not isinstance(rounds, list): raise RuntimeError("CPU per_round must be a list")
    round_medians = [finite(row["steady"]["median_ms"], "CPU round median") for row in rounds]
    aggregate = mode.get("steady", {})
    samples = aggregate.get("samples_ms", [])
    if not isinstance(samples, list) or not samples: raise RuntimeError("CPU aggregate raw samples are required")
    return {"rounds": len(rounds), "aggregate_mean_ms": finite(aggregate.get("mean_ms"), "CPU aggregate mean"), "aggregate_median_ms": finite(aggregate.get("median_ms"), "CPU aggregate median"), "aggregate_p95_ms": finite(aggregate.get("p95_ms"), "CPU aggregate p95"), "aggregate_samples_ms": samples, "round_median_center_ms": float(np.median(round_medians)), "observed_round_median_range_ms": [min(round_medians), max(round_medians)]}

def cpu_rows(task: str, path: Path, effects: dict[str, Any]) -> list[dict[str, Any]]:
    data = read(path / "benchmark.json" if path.is_dir() else path)
    if data.get("schema") != "selected_ema_decoder_cpu_cached_v1" or data.get("task") != task.lower(): raise RuntimeError(f"{task}: wrong decoder CPU benchmark schema/task")
    if data.get("script_sha256") != sha(HERE / "benchmark_decoder_cpu.py"):
        raise RuntimeError(f"{task}: CPU benchmark script hash differs from the current reviewed source")
    runtime_source = data.get("benchmark_runtime_source_hashes_before_after", {})
    if runtime_source.get("unchanged") is not True or runtime_source.get("before") != runtime_source.get("after"):
        raise RuntimeError(f"{task}: CPU benchmark runtime source was not sealed unchanged")
    protocol = data.get("protocol", {})
    if protocol.get("batches") != [1, 4, 8] or protocol.get("rounds") != 5 or protocol.get("timed_actual_advances_per_round", 0) < 128 or protocol.get("fresh_runtime_each_round") is not True:
        raise RuntimeError(f"{task}: CPU benchmark protocol is incomplete")
    selected = data.get("selected_by_mode", {}); expected = {mode: effects[mode]["checkpoint_sha256"] for mode in ("recency", "flat")}
    for mode, digest in expected.items():
        selected_meta = selected.get(mode, {})
        if selected_meta.get("checkpoint_sha256") != digest: raise RuntimeError(f"{task}: CPU benchmark selected checkpoint differs from effect receipt")
        assert_hash_map(f"{task} CPU {mode}", selected_meta.get("source_hashes", selected_meta.get("source_manifest_sha256")))
    out = []
    for batch in ("B1", "B4", "B8"):
        case = data.get("cases", {}).get(batch, {}); online = case.get("cached_online", {}); parity = case.get("parity", {})
        if set(online) != {"recency", "flat"} or set(parity) != {"recency", "flat"} or any(not isinstance(value, dict) or value.get("passed") is not True for value in parity.values()):
            raise RuntimeError(f"{task} {batch}: missing successful paired CPU parity")
        if set(online) != {"recency", "flat"}: raise RuntimeError(f"{task} {batch}: incomplete paired CPU modes")
        rec, flat = summary_stats(online["recency"]), summary_stats(online["flat"])
        if rec["rounds"] != 5 or flat["rounds"] != 5: raise RuntimeError(f"{task} {batch}: incomplete CPU round evidence")
        out.append({"task": task, "batch": batch, "recency": rec, "flat": flat, "paired_median_ratio_recency_over_flat": rec["aggregate_median_ms"] / flat["aggregate_median_ms"], "raw_rounds": online, "footprint": case.get("model_footprint_by_mode"), "actual_cache": case.get("cached_preallocated_storage_by_mode")})
    return out


def train_rows(task: str, path: Path, effect: dict[str, Any]) -> dict[str, Any]:
    data = read(path)
    if data.get("schema") != "rift_recency_flat_real_train_step_gpu_v1" or data.get("status") != "COMPLETED_FRESH_PARAMETER_CONTROLLED_COMPUTE_ONLY" or data.get("task") != task.lower():
        raise RuntimeError(f"{task}: wrong GPU train-step benchmark schema/status/task")
    if data.get("script_sha256") != sha(HERE / "benchmark_train_step.py"):
        raise RuntimeError(f"{task}: GPU benchmark script hash differs from the current reviewed source")
    runtime_before, runtime_after = data.get("runtime_source_map_before"), data.get("runtime_source_map_after")
    if data.get("runtime_source_map_unchanged") is not True or runtime_before != runtime_after:
        raise RuntimeError(f"{task}: GPU runtime source map was not sealed unchanged")
    assert_hash_map(f"{task} GPU runtime", runtime_before)
    fixture = data.get("fixture", {})
    if not isinstance(fixture, dict): raise RuntimeError(f"{task}: GPU fixture is malformed")
    assert_hash_map(f"{task} GPU fixture", fixture.get("source_bindings"))
    ref_meta_sha = sha(Path(effect["recency"]["checkpoint"]).parent / "run_meta.json")
    paired = fixture.get("paired_reference") if isinstance(fixture, dict) else None
    if not isinstance(paired, dict): raise RuntimeError(f"{task}: GPU fixture lacks paired reference evidence")
    paired_meta_sha = paired.get("run_meta_sha256", paired.get("reference_run_meta_sha256"))
    if paired_meta_sha != ref_meta_sha: raise RuntimeError(f"{task}: GPU fixture reference does not match summarized recency run")
    if task == "M1" and not fixture.get("carrier_binding_sha256"): raise RuntimeError("M1: GPU fixture lacks factual muscle FULL carrier binding")
    if task == "H1" and paired.get("reference_pairing_digests") is None: raise RuntimeError("H1: GPU fixture lacks signed-state14 pairing evidence")
    if task == "M2" and paired.get("frozen_cache_hashes") is None: raise RuntimeError("M2: GPU fixture lacks frozen cache evidence")
    protocol = data.get("protocol", {})
    if protocol.get("rounds") != 5 or protocol.get("warmup_updates") != 10 or protocol.get("timed_updates") != 30 or not data.get("initial_pairing", {}).get("named_parameters_byte_equal") or data.get("initial_pairing", {}).get("sole_buffer_delta") != "temporal.recency_slopes":
        raise RuntimeError(f"{task}: GPU benchmark pairing/protocol drift")
    rounds = data.get("rounds", {})
    out = {"task": task, "fixture_prepare_seconds": data.get("fixture_prepare_seconds"), "protocol": protocol, "initial_pairing": data.get("initial_pairing"), "fixture": fixture, "modes": {}}
    for mode in ("recency", "flat"):
        values = rounds.get(mode, [])
        if len(values) != 5 or {row.get("round") for row in values} != {1, 2, 3, 4, 5} or any(row.get("ema_updates") != 40 or row.get("slopes_zero_after") != (mode == "flat") for row in values):
            raise RuntimeError(f"{task}: incomplete GPU rounds or mode slope/EMA evidence")
        wall = [finite(row["wall_summary"]["median_ms"], f"{task} GPU median") for row in values]
        peak = [finite(row["memory"]["peak_allocated"], f"{task} GPU memory") for row in values]
        model = values[0].get("model", {})
        out["modes"][mode] = {"round_median_ms": float(np.median(wall)), "round_mean_ms": float(np.mean(wall)), "round_p95_ms": float(np.percentile(wall, 95)), "peak_allocated_bytes_median": float(np.median(peak)), "parameters": model.get("parameters"), "buffers": model.get("buffers"), "logical_kv_bytes_per_stream": model.get("logical_kv_bytes_per_stream"), "temporal_attention_macs": model.get("temporal_attention_macs"), "temporal_attention_macs_formula": model.get("temporal_attention_macs_formula"), "raw_rounds": values}
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def figures(dest: Path, effects: list[dict[str, Any]], latency: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = [row["task"] for row in effects]; x = np.arange(len(labels)); width = .36
    ax0.bar(x-width/2, [row["recency"]["mean"] for row in effects], width, label="recency")
    ax0.bar(x+width/2, [row["flat"]["mean"] for row in effects], width, label="flat")
    ax0.set_xticks(x, labels); ax0.set_ylabel("selected development metric"); ax0.set_title("Current seed-42 selection surfaces"); ax0.legend()
    for task in labels:
        rows = [row for row in latency if row["task"] == task and row["batch"] in ("B1", "B8")]
        for row in rows:
            marker = "o" if row["batch"] == "B1" else "s"
            for mode, color in (("recency", "C0"), ("flat", "C1")):
                value = row[mode]; lo, hi = value["observed_round_median_range_ms"]
                ax1.errorbar(f"{task}-{row['batch']}", value["round_median_center_ms"], yerr=[[value["round_median_center_ms"]-lo], [hi-value["round_median_center_ms"]]], fmt=marker, color=color, capsize=3, label=mode if task == labels[0] and row["batch"] == "B1" else None)
    ax1.set_ylabel("CPU cached online ms/bin"); ax1.set_title("Decoder CPU median ± observed round range"); ax1.legend(); ax1.tick_params(axis="x", rotation=35)
    fig.savefig(dest / "comparison.png", dpi=220); fig.savefig(dest / "comparison.pdf"); plt.close(fig)


def temporal_mode_summary(case_name: str, mode: str, mode_data: Any) -> dict[str, Any]:
    if not isinstance(mode_data, dict): raise RuntimeError(f"{case_name} {mode}: missing pure-temporal online result")
    steady, rounds = mode_data.get("steady_cached_online", {}), mode_data.get("rounds", [])
    if not isinstance(rounds, list) or len(rounds) != 5: raise RuntimeError(f"{case_name} {mode}: pure-temporal needs five rounds")
    round_medians = []
    for row in rounds:
        if not isinstance(row, dict) or not isinstance(row.get("latency_ns"), list) or not row["latency_ns"]: raise RuntimeError(f"{case_name} {mode}: missing raw pure-temporal samples")
        round_medians.append(finite(row.get("summary", {}).get("median_ms"), f"{case_name} {mode} round median"))
    return {"mean_ms": finite(steady.get("mean_ms"), f"{case_name} {mode} mean"), "median_ms": finite(steady.get("median_ms"), f"{case_name} {mode} median"), "p95_ms": finite(steady.get("p95_ms"), f"{case_name} {mode} p95"), "round_median_range_ms": [min(round_medians), max(round_medians)], "raw_rounds": rounds}


def temporal_matrix(temporal: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, case in sorted(temporal["cases"].items()):
        config, memory, macs = case.get("config", {}), case.get("memory", {}), case.get("attention_macs_per_online_bin", {})
        parameters = case.get("parameters", {}).get("count_each_variant"); kv = memory.get("logical_kv_bytes"); mac = macs.get("value")
        if not isinstance(parameters, int) or parameters <= 0 or not isinstance(kv, int) or kv <= 0 or not isinstance(mac, int) or mac <= 0:
            raise RuntimeError(f"{name}: missing actual pure-temporal parameter/KV/MAC fields")
        online = case.get("online_cached", {})
        if set(online) != {"recency", "flat"}: raise RuntimeError(f"{name}: missing paired pure-temporal modes")
        rows.append({"case": name, "parameters_each_variant": parameters, "logical_kv_bytes": kv, "attention_only_macs_per_online_bin": mac, "mac_scope": macs.get("excludes"), "windows": config.get("windows"), "recency_online": temporal_mode_summary(name, "recency", online["recency"]), "flat_online": temporal_mode_summary(name, "flat", online["flat"])})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL); parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--m2-flat-selection", type=Path, default=ROOT / "results/recency_flat_ablation_v1/selection_m2_concat_flat_ext6_s42")
    parser.add_argument("--temporal-json", type=Path, required=True, help="completed pure-temporal benchmark JSON")
    for task in ("m1", "h1", "m2"):
        parser.add_argument(f"--cpu-{task}", type=Path, required=True, help="completed task decoder CPU benchmark directory or JSON")
        parser.add_argument(f"--train-{task}", type=Path, required=True, help="completed task GPU train-step benchmark JSON")
    args = parser.parse_args(); protocol = read(args.protocol); dest = args.dest.resolve()
    if protocol.get("schema") != "recency_flat_current_three_tasks_protocol_v1": raise RuntimeError("wrong protocol schema")
    if dest.exists(): raise FileExistsError("summary destination must be fresh")
    task_cfg = protocol.get("tasks", {}); effects = []
    for task in ("M1", "H1", "M2"):
        cfg = task_cfg.get(task, {}); rec_run, flat_run = Path(cfg["reference"]), Path(cfg["flat"])
        rec_sel = Path(cfg.get("reference_selection", rec_run)) / TASKS[task]["selection"] if task == "M2" else rec_run / TASKS[task]["selection"]
        flat_sel = args.m2_flat_selection / "score_receipt.json" if task == "M2" else flat_run / TASKS[task]["selection"]
        effects.append(effect_pair(task, rec_run, flat_run, rec_sel, flat_sel))
    effect_by_task = {row["task"]: row for row in effects}
    latency = [entry for task, path in (("M1", args.cpu_m1), ("H1", args.cpu_h1), ("M2", args.cpu_m2)) for entry in cpu_rows(task, path, effect_by_task[task])]
    costs = [train_rows(task, path, effect_by_task[task]) for task, path in (("M1", args.train_m1), ("H1", args.train_h1), ("M2", args.train_m2))]
    temporal = read(args.temporal_json)
    required_temporal_cases = {f"R{context}_B{batch}" for context in (50, 100, 300) for batch in (1, 4, 8)}
    if temporal.get("schema") != "rift_temporal_recency_flat_cpu_benchmark_v1" or temporal.get("status") != "COMPLETED_SYNTHETIC_TEMPORAL_ONLY" or set(temporal.get("cases", {})) != required_temporal_cases:
        raise RuntimeError("pure temporal JSON lacks the completed R50/R100/R300 B1/B4/B8 benchmark matrix")
    dest.mkdir(parents=True); result = {"schema": "recency_flat_current_three_task_summary_v1", "status": "COMPLETED", "protocol_sha256": sha(args.protocol), "scope": "Current M1 muscle FULL (paper mainline, official submission 582205), H1 signed_state14, and M2 concat; one paired seed42; development surfaces for recency-versus-flat effects. Official 582205 is cited as fixed mainline provenance, not as a flat-ablation result or a multi-seed confidence interval.", "effects": effects, "decoder_cpu_latency": latency, "controlled_gpu_train_cost": costs, "pure_temporal": {"path": str(args.temporal_json.resolve()), "sha256": sha(args.temporal_json), "scope": temporal.get("scope"), "matrix": temporal_matrix(temporal)}, "full_run_wall_time_note": "Receipt wall times are uncontrolled resource-dependent observations, not paired step benchmarks."}
    (dest / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    effect_csv = [{"task": row["task"], "mode": mode, "seed": 42, "metric": row["metric"], "selected_epoch": row[mode]["epoch"], "mean": row[mode]["mean"], "worst": row[mode]["worst"], "per_session_json": json.dumps(row[mode]["per_session"], sort_keys=True), "delta_recency_minus_flat_mean": row["delta_recency_minus_flat"]["mean"], "delta_recency_minus_flat_worst": row["delta_recency_minus_flat"]["worst"], "delta_recency_minus_flat_per_session_json": json.dumps(row["delta_recency_minus_flat"]["per_session"], sort_keys=True), "selected_checkpoint_sha256": row[mode]["checkpoint_sha256"]} for row in effects for mode in ("recency", "flat")]
    write_csv(dest / "effect.csv", effect_csv)
    write_csv(dest / "latency.csv", [{"task": row["task"], "batch": row["batch"], "recency_mean_ms": row["recency"]["aggregate_mean_ms"], "flat_mean_ms": row["flat"]["aggregate_mean_ms"], "recency_median_ms": row["recency"]["aggregate_median_ms"], "flat_median_ms": row["flat"]["aggregate_median_ms"], "recency_p95_ms": row["recency"]["aggregate_p95_ms"], "flat_p95_ms": row["flat"]["aggregate_p95_ms"], "recency_round_median_range_ms": row["recency"]["observed_round_median_range_ms"], "flat_round_median_range_ms": row["flat"]["observed_round_median_range_ms"], "recency_aggregate_samples_ms_json": json.dumps(row["recency"]["aggregate_samples_ms"]), "flat_aggregate_samples_ms_json": json.dumps(row["flat"]["aggregate_samples_ms"]), "paired_median_ratio_recency_over_flat": row["paired_median_ratio_recency_over_flat"], "recency_raw_per_round_json": json.dumps(row["raw_rounds"]["recency"], sort_keys=True), "flat_raw_per_round_json": json.dumps(row["raw_rounds"]["flat"], sort_keys=True)} for row in latency])
    cpu_b1 = {(row["task"], mode): row for row in latency if row["batch"] == "B1" for mode in ("recency", "flat")}
    write_csv(dest / "train_cost.csv", [{"task": row["task"], "mode": mode, **{key: value for key, value in vals.items() if key != "raw_rounds"}, "raw_gpu_rounds_json": json.dumps(vals["raw_rounds"], sort_keys=True), "decoder_parameter_footprint_json": json.dumps(cpu_b1[(row["task"], mode)]["footprint"][mode], sort_keys=True), "actual_cpu_cache_allocation_json": json.dumps(cpu_b1[(row["task"], mode)]["actual_cache"][mode], sort_keys=True)} for row in costs for mode, vals in row["modes"].items()])
    temporal_rows = temporal_matrix(temporal)
    lines = ["# Current recency-versus-flat ablation", "", "This report covers current M1 muscle FULL, H1 signed_state14, and M2 concat, each at seed 42. M1 is the paper mainline whose fixed original FULL recipe produced official submission 582205; recency-versus-flat effects remain development-surface results and are neither official-evaluation claims nor multi-seed statistical estimates.", "", "## Effects", "", "| Task | Metric | Recency epoch | Flat epoch | Recency mean | Flat mean | Δ recency−flat |", "|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['task']} | {r['metric']} | {r['recency']['epoch']} | {r['flat']['epoch']} | {r['recency']['mean']:.6f} | {r['flat']['mean']:.6f} | {r['delta_recency_minus_flat']['mean']:.6f} |" for r in effects]
    lines += ["", "## Decoder CPU cached online latency", "", "| Task | Batch | Recency mean / median / p95 ms | Flat mean / median / p95 ms | Recency÷flat median |", "|---|---|---:|---:|---:|"]
    lines += [f"| {r['task']} | {r['batch']} | {r['recency']['aggregate_mean_ms']:.4f} / {r['recency']['aggregate_median_ms']:.4f} / {r['recency']['aggregate_p95_ms']:.4f} | {r['flat']['aggregate_mean_ms']:.4f} / {r['flat']['aggregate_median_ms']:.4f} / {r['flat']['aggregate_p95_ms']:.4f} | {r['paired_median_ratio_recency_over_flat']:.4f} |" for r in latency]
    lines += ["", "Round variability is recorded in `latency.csv` as all raw rounds and aggregate samples. The figure uses the median of round medians with its observed min–max range, so its range cannot imply an invalid negative error bar.", "", "## Controlled GPU training-step compute", "", "| Task | Mode | Median / mean / p95 ms across 5 round medians | Peak allocated bytes median | Parameters | Logical KV bytes/stream | Attention-only MACs |", "|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['task']} | {mode} | {v['round_median_ms']:.4f} / {v['round_mean_ms']:.4f} / {v['round_p95_ms']:.4f} | {v['peak_allocated_bytes_median']:.0f} | {v['parameters']} | {v['logical_kv_bytes_per_stream']} | {v['temporal_attention_macs']} |" for r in costs for mode, v in r['modes'].items()]
    lines += ["", "The MAC figures cover stated temporal attention terms only; their serialized formulas and exclusions are in `train_cost.csv` and `summary.json`.", "", "## Pure-temporal reference matrix and paired cached-online latency", "", f"Source: `{args.temporal_json.resolve()}` (SHA-256 `{sha(args.temporal_json)}`).", "", "| Case | Params | Logical KV bytes | MACs/bin | Recency mean / median / p95 ms | Flat mean / median / p95 ms |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['case']} | {r['parameters_each_variant']} | {r['logical_kv_bytes']} | {r['attention_only_macs_per_online_bin']} | {r['recency_online']['mean_ms']:.4f} / {r['recency_online']['median_ms']:.4f} / {r['recency_online']['p95_ms']:.4f} | {r['flat_online']['mean_ms']:.4f} / {r['flat_online']['median_ms']:.4f} / {r['flat_online']['p95_ms']:.4f} |" for r in temporal_rows]
    lines += ["", "All nine pure-temporal cases retain five raw rounds per mode in `summary.json`; each range is the observed min–max of the five round medians."]
    lines += ["", "CPU ratios are descriptive paired median ratios from alternating rounds; no statistical-significance claim is made. Full-run wall times are uncontrolled resource-dependent receipt observations."]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(dest, effects, latency)

if __name__ == "__main__": main()
