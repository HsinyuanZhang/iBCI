#!/usr/bin/env python3
"""Create an audited descriptive summary of four completed M1 muscle scans.

This command reads only ``run_meta.json``, ``train_receipt.json``, and
``score_receipt.json`` from the four explicitly supplied run directories.  It
never discovers runs, opens prediction artifacts, data, NWB files, NPZ files,
or official-test material.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

EPOCHS = tuple(range(1, 25))
HO = ("20121004", "20121017", "20121024")
WINDOWS = {"20121004": 1305, "20121017": 1295, "20121024": 1281}
TOTAL_WINDOWS = sum(WINDOWS.values())
TOTAL_UPDATES = 24 * 6665
ORIGINAL_VARIANT = "muscle_response16_svd4/global_rms"
CANDIDATE_VARIANT = "muscle_response_svd3_mean_rate4_matched_scale"
OLD_TRAIN_SCHEMA = "m1_muscle_r100_train_v1"
NEW_TRAIN_SCHEMA = "m1_muscle_r100_multiseed_train_v1"
OLD_SCORE_SCHEMA = "m1_muscle_r100_ho_calib_epoch_scan_v1"
NEW_SCORE_SCHEMA = "m1_muscle_r100_multiseed_ho_calib_epoch_scan_v1"
OLD_CELL = "M1-MUSCLE-R100-D4-JOINT-B3S-CONCAT-V1"
NEW_CELL = "M1-MUSCLE-R100-D4-JOINT-B3S-CONCAT-MULTISEED-V1"
ARM = "D_JOINT"
METRIC = "equal_session_mean_channel_variance_weighted_r2"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"required receipt missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _without_carrier(value: Any) -> Any:
    """Preserve source/sampler facts while removing carrier-dependent fields."""
    if isinstance(value, Mapping):
        return {
            str(key): _without_carrier(item)
            for key, item in value.items()
            if "carrier" not in str(key).lower()
            and str(key) not in {"bank_hashes", "bank_report"}
        }
    if isinstance(value, list):
        return [_without_carrier(item) for item in value]
    return value


def _variant(binding: Mapping[str, Any]) -> str:
    body = binding.get("carrier_pack_receipt_body", {})
    _need(isinstance(body, Mapping), "carrier receipt body required")
    value = binding.get("carrier_variant", body.get("carrier_variant", body.get("method")))
    _need(isinstance(value, str), "carrier variant/method missing")
    return value


def _finite(value: Any, label: str) -> float:
    result = float(value)
    _need(math.isfinite(result), f"nonfinite {label}")
    return result


def _selected_epoch(curve: Mapping[str, Mapping[str, Any]]) -> int:
    return min(EPOCHS, key=lambda epoch: (-float(curve[str(epoch)][METRIC]), epoch))


def _endpoint(curve: Mapping[str, Mapping[str, Any]], epoch: int) -> dict[str, Any]:
    row = curve[str(epoch)]
    worst_session = min(HO, key=lambda session: float(row["per_session"][session]["channel_variance_weighted_r2"]))
    return {
        "epoch": epoch,
        "equal_session_mean_channel_variance_weighted_r2": row[METRIC],
        "worst_session": worst_session,
        "worst_session_r2": row["per_session"][worst_session]["channel_variance_weighted_r2"],
        "per_session": {
            session: {
                "channel_variance_weighted_r2": row["per_session"][session]["channel_variance_weighted_r2"],
                "window_count": row["per_session"][session]["window_count"],
                "target_sha256": row["per_session"][session]["target_sha256"],
                "starts_sha256": row["per_session"][session]["starts_sha256"],
            }
            for session in HO
        },
    }


def _validate_curve(score: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = score.get("ema_by_epoch")
    _need(isinstance(rows, Mapping) and set(rows) == {str(epoch) for epoch in EPOCHS}, f"{label}: exactly epochs 1..24 required")
    normalized: dict[str, dict[str, Any]] = {}
    for epoch in EPOCHS:
        row = rows[str(epoch)]
        _need(isinstance(row, Mapping), f"{label}: epoch {epoch} row required")
        _need(row.get("partial") is False and int(row.get("n_windows", -1)) == TOTAL_WINDOWS, f"{label}: epoch {epoch} must be full HO3 scoring")
        per = row.get("per_session")
        _need(isinstance(per, Mapping) and set(per) == set(HO), f"{label}: epoch {epoch} HO roster drift")
        values: list[float] = []
        cleaned: dict[str, Any] = {}
        for session in HO:
            item = per[session]
            _need(isinstance(item, Mapping), f"{label}: epoch {epoch}/{session} row required")
            value = _finite(item.get("channel_variance_weighted_r2"), f"{label} epoch {epoch}/{session} metric")
            target_sha, starts_sha = item.get("target_sha256"), item.get("starts_sha256")
            _need(isinstance(target_sha, str) and len(target_sha) == 64, f"{label}: invalid target SHA at epoch {epoch}/{session}")
            _need(isinstance(starts_sha, str) and len(starts_sha) == 64, f"{label}: invalid starts SHA at epoch {epoch}/{session}")
            _need(int(item.get("window_count", -1)) == WINDOWS[session], f"{label}: window count drift at epoch {epoch}/{session}")
            values.append(value)
            cleaned[session] = {"channel_variance_weighted_r2": value, "target_sha256": target_sha,
                                "starts_sha256": starts_sha, "window_count": WINDOWS[session]}
        mean = _finite(row.get(METRIC), f"{label} epoch {epoch} equal-session metric")
        _need(abs(mean - sum(values) / len(values)) <= 1e-12, f"{label}: equal-session mean drift at epoch {epoch}")
        normalized[str(epoch)] = {METRIC: mean, "per_session": cleaned}
    return normalized


def _read_run(path: Path, *, run_id: str, seed: int, variant: str, old_schema: bool) -> dict[str, Any]:
    root = path.resolve()
    meta_path, train_path, score_path = (root / "run_meta.json", root / "train_receipt.json", root / "score_receipt.json")
    meta, train, score = _read_json(meta_path), _read_json(train_path), _read_json(score_path)
    expected_train, expected_score = (OLD_TRAIN_SCHEMA, OLD_SCORE_SCHEMA) if old_schema else (NEW_TRAIN_SCHEMA, NEW_SCORE_SCHEMA)
    expected_cell = OLD_CELL if old_schema else NEW_CELL
    _need(meta.get("schema") == expected_train and meta.get("status") == "FORMAL", f"{run_id}: formal run metadata schema/status mismatch")
    _need(train.get("schema") == expected_train.replace("_train_v1", "_train_receipt_v1") and train.get("status") == "COMPLETED", f"{run_id}: completed training receipt schema/status mismatch")
    _need(score.get("schema") == expected_score and score.get("status") == "COMPLETED", f"{run_id}: completed score receipt schema/status mismatch")
    for name, receipt in (("run metadata", meta), ("training receipt", train), ("score receipt", score)):
        _need(receipt.get("cell") == expected_cell, f"{run_id}: {name} cell mismatch")
        _need(receipt.get("arm") == ARM, f"{run_id}: {name} arm must be {ARM}")
        _need(receipt.get("seed") == seed, f"{run_id}: {name} seed mismatch")
        _need(receipt.get("sampler_seed") == 42, f"{run_id}: {name} sampler seed must be frozen seed 42")
    _need(meta.get("epochs") == 24 and train.get("epochs") == 24 and train.get("steps") == TOTAL_UPDATES, f"{run_id}: incomplete formal training")
    _need(meta.get("total_updates") == TOTAL_UPDATES and meta.get("updates_per_epoch") == 6665, f"{run_id}: update schedule drift")
    _need(meta.get("official_test_used") is False and score.get("official_test_used") is False, f"{run_id}: official data flag drift")
    binding = meta.get("carrier_binding")
    _need(isinstance(binding, Mapping) and _variant(binding) == variant, f"{run_id}: carrier variant mismatch")
    _need(score.get("carrier_binding") == binding, f"{run_id}: score carrier binding mismatch")
    _need(train.get("carrier_binding") == binding, f"{run_id}: training carrier binding mismatch")
    _need(score.get("source_contract") == meta.get("source_contract") == train.get("source_contract"), f"{run_id}: source contract receipt mismatch")
    _need(score.get("source_hashes") == meta.get("source_hashes") == train.get("source_hashes"), f"{run_id}: source hashes receipt mismatch")
    _need(score.get("b3s") == meta.get("b3s") == train.get("b3s"), f"{run_id}: B3S provenance receipt mismatch")
    _need(score.get("fit_sha256") == meta.get("fit_sha256") == train.get("fit_sha256"), f"{run_id}: fit SHA receipt mismatch")
    _need(score.get("ho_contract") is not None, f"{run_id}: HO contract missing")
    curve = _validate_curve(score, run_id)
    selected = _selected_epoch(curve)
    _need(int(score.get("selection", {}).get("epoch", -1)) == selected, f"{run_id}: selection epoch is not earliest maximum")
    _need(score.get("selection", {}).get("metric") == "channel_variance_weighted_r2", f"{run_id}: selection metric drift")
    return {
        "id": run_id, "path": str(root), "seed": seed, "carrier_variant": variant,
        "input_receipts_sha256": {"run_meta.json": _sha(meta_path), "train_receipt.json": _sha(train_path), "score_receipt.json": _sha(score_path)},
        "source_noncarrier_contract": _without_carrier(meta["source_contract"]),
        "ho_noncarrier_contract": _without_carrier(score["ho_contract"]),
        "source_hashes": meta["source_hashes"], "b3s": meta["b3s"], "fit_sha256": meta["fit_sha256"],
        "carrier_pack_npz_sha256": binding.get("carrier_pack_npz_sha256"),
        "curve": curve, "fixed_epoch_3": _endpoint(curve, 3), "selected": _endpoint(curve, selected),
        "selection_rule": "earliest maximum equal-session mean channel variance-weighted R2 on visible HO3 calibration",
    }


def _same_contract(runs: list[Mapping[str, Any]], key: str, label: str) -> dict[str, Any]:
    reference = runs[0][key]
    for run in runs[1:]:
        _need(run[key] == reference, f"{label} mismatch: {run['id']}")
    return {"all_runs_identical": True, "sha256": hashlib.sha256(_canonical_json(reference).encode()).hexdigest(), "value": reference}


def _same_targets_and_starts(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    reference: dict[str, dict[str, Any]] | None = None
    for run in runs:
        for epoch in EPOCHS:
            per = run["curve"][str(epoch)]["per_session"]
            current = {session: {key: per[session][key] for key in ("target_sha256", "starts_sha256", "window_count")} for session in HO}
            if reference is None:
                reference = current
            else:
                _need(current == reference, f"HO target/starts/window contract mismatch: {run['id']} epoch {epoch}")
    _need(reference is not None, "missing HO target/starts contract")
    return {"all_runs_all_epochs_identical": True, "per_session": reference}


def _stats(values: list[float]) -> dict[str, Any]:
    _need(len(values) == 3, "original aggregate requires exactly three seeds")
    mean = sum(values) / len(values)
    return {"n_seeds": 3, "mean": mean,
            "sample_std_ddof1": math.sqrt(sum((value - mean) ** 2 for value in values) / 2),
            "min": min(values), "max": max(values), "values": values}


def _original_aggregate(originals: list[Mapping[str, Any]]) -> dict[str, Any]:
    curve: dict[str, Any] = {}
    for epoch in EPOCHS:
        values = [run["curve"][str(epoch)][METRIC] for run in originals]
        curve[str(epoch)] = {
            "equal_session_mean_channel_variance_weighted_r2": _stats(values),
            "per_session": {
                session: _stats([run["curve"][str(epoch)]["per_session"][session]["channel_variance_weighted_r2"] for run in originals])
                for session in HO
            },
        }
    endpoint = {}
    for key in ("fixed_epoch_3", "selected"):
        endpoint[key] = {
            "individual_run_endpoints": {run["id"]: run[key] for run in originals},
            "metric_summary": _stats([run[key][METRIC] for run in originals]),
            "worst_session_r2_summary": _stats([run[key]["worst_session_r2"] for run in originals]),
            "worst_session_by_run": {run["id"]: run[key]["worst_session"] for run in originals},
            "per_session": {
                session: _stats([run[key]["per_session"][session]["channel_variance_weighted_r2"] for run in originals])
                for session in HO
            },
        }
    return {"description": "Descriptive aggregate across exactly the three original-carrier seeds; selected epochs are independently selected per run.",
            "curve_by_epoch": curve, "endpoint_summaries": endpoint}


def _difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = {METRIC: left[METRIC] - right[METRIC], "per_session": {}}
    for session in HO:
        result["per_session"][session] = left["per_session"][session]["channel_variance_weighted_r2"] - right["per_session"][session]["channel_variance_weighted_r2"]
    return result


def _original_identity(originals: list[Mapping[str, Any]]) -> dict[str, Any]:
    reference = originals[0]
    for run in originals[1:]:
        _need(run["carrier_pack_npz_sha256"] == reference["carrier_pack_npz_sha256"], f"original carrier pack SHA mismatch: {run['id']}")
        _need(run["fit_sha256"] == reference["fit_sha256"], f"original fit SHA mismatch: {run['id']}")
        _need(run["b3s"] == reference["b3s"], f"original B3S initialization source mismatch: {run['id']}")
    return {"original_carrier_pack_sha256": reference["carrier_pack_npz_sha256"],
            "original_fit_sha256": reference["fit_sha256"],
            "original_b3s_initialization_source": reference["b3s"],
            "all_three_original_runs_identical": True}


def _write_csv(path: Path, runs: list[Mapping[str, Any]]) -> None:
    fields = ("run_id", "seed", "carrier_variant", "epoch", METRIC, *[f"{session}_channel_variance_weighted_r2" for session in HO])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for epoch in EPOCHS:
                point = run["curve"][str(epoch)]
                writer.writerow({"run_id": run["id"], "seed": run["seed"], "carrier_variant": run["carrier_variant"], "epoch": epoch,
                                 METRIC: point[METRIC], **{f"{session}_channel_variance_weighted_r2": point["per_session"][session]["channel_variance_weighted_r2"] for session in HO}})
    temporary.replace(path)


def _write_plot(path_png: Path, path_svg: Path, originals: list[Mapping[str, Any]], candidate: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    colors = {"original-s42": "#1f77b4", "original-s43": "#4c9ed9", "original-s44": "#82bce8", "candidate-s42": "#d62728"}
    for run in [*originals, candidate]:
        left.plot(EPOCHS, [run["curve"][str(epoch)][METRIC] for epoch in EPOCHS], label=run["id"], color=colors[run["id"]], linewidth=2)
    left.set(title="Complete 24-epoch HO3 curves", xlabel="Epoch", ylabel="Equal-session channel variance-weighted R²", xticks=(1, 3, 6, 12, 18, 24))
    left.grid(alpha=.25); left.legend(fontsize=8, frameon=False)
    for position, endpoint_name, display in ((0, "fixed_epoch_3", "Fixed epoch 3"), (1, "selected", "Per-run selected")):
        original_values = [run[endpoint_name][METRIC] for run in originals]
        mean, spread = sum(original_values) / 3, math.sqrt(sum((value - sum(original_values) / 3) ** 2 for value in original_values) / 2)
        right.scatter([position - .12] * 3, original_values, color="#1f77b4", s=38, label="Original seeds" if position == 0 else None, zorder=3)
        right.errorbar(position - .12, mean, yerr=spread, color="#0e4f8a", marker="_", capsize=4, linewidth=1.5, label="Original mean ± sample SD" if position == 0 else None)
        right.scatter(position + .12, candidate[endpoint_name][METRIC], color="#d62728", marker="D", s=44, label="Candidate s42" if position == 0 else None, zorder=3)
    right.set(title="Fixed e3 and independently selected endpoints", xlabel="Endpoint", ylabel="Equal-session channel variance-weighted R²", xticks=(0, 1), xticklabels=("Fixed epoch 3", "Selected per run"))
    right.grid(axis="y", alpha=.25); right.legend(fontsize=8, frameon=False, loc="best")
    fig.savefig(path_png, dpi=180); fig.savefig(path_svg)
    plt.close(fig)


def _readme() -> str:
    return """# M1 muscle multiseed summary

This directory is an auditable descriptive aggregation of four explicitly supplied, completed 24-epoch M1 HO3-calibration scans: original-carrier seeds 42, 43, and 44, and candidate-carrier seed 42. The command reads only each run's `run_meta.json`, `train_receipt.json`, and `score_receipt.json`; it does not open prediction artifacts, data, NPZ, NWB, or official-test material.

`summary.json` records SHA-256 values for every input receipt, validates the frozen source/sampler noncarrier contract, validates identical HO targets, starts, and window counts at every epoch, and records fixed epoch 3 and each run's independently selected epoch, including the lowest-scoring HO session at each endpoint. Selection means the earliest maximum of equal-session channel variance-weighted R² on visible HO3 calibration.

`curves.csv` contains only the complete recorded 24-epoch scores. `curves.png` and `curves.svg` show those real curves and fixed-e3/independently-selected endpoints. The original summary uses exactly three seeds and reports mean, sample standard deviation (`ddof=1`), minimum, and maximum.

Candidate-minus-original-s42 differences are descriptive comparisons at the common fixed epoch 3 and at each run's own selected epoch. They do not establish a mechanism and do not report an official-test improvement.
"""


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    _need(not output.exists(), f"--output-dir must be new: {output}")
    original42 = _read_run(args.original_s42, run_id="original-s42", seed=42, variant=ORIGINAL_VARIANT, old_schema=True)
    original43 = _read_run(args.original_s43, run_id="original-s43", seed=43, variant=ORIGINAL_VARIANT, old_schema=False)
    original44 = _read_run(args.original_s44, run_id="original-s44", seed=44, variant=ORIGINAL_VARIANT, old_schema=False)
    candidate42 = _read_run(args.candidate_s42, run_id="candidate-s42", seed=42, variant=CANDIDATE_VARIANT, old_schema=False)
    originals, all_runs = [original42, original43, original44], [original42, original43, original44, candidate42]
    source_contract = _same_contract(all_runs, "source_noncarrier_contract", "source/sampler noncarrier contract")
    ho_contract = _same_contract(all_runs, "ho_noncarrier_contract", "HO noncarrier contract")
    original_identity = _original_identity(originals)
    targets = _same_targets_and_starts(all_runs)
    comparison = {
        "scope": "descriptive only; no mechanism claim and no official-test improvement claim",
        "fixed_epoch_3_candidate_minus_original_s42": {
            "candidate": candidate42["fixed_epoch_3"], "original_s42": original42["fixed_epoch_3"],
            "candidate_minus_original_s42": _difference(candidate42["fixed_epoch_3"], original42["fixed_epoch_3"]),
        },
        "independently_selected_candidate_minus_original_s42": {
            "note": "Each selected epoch is independently chosen on visible HO3 calibration; this is not a common-epoch comparison.",
            "candidate": candidate42["selected"], "original_s42": original42["selected"],
            "candidate_minus_original_s42": _difference(candidate42["selected"], original42["selected"]),
        },
    }
    body = {
        "schema": "m1_muscle_r100_multiseed_summary_v1", "status": "COMPLETED",
        "input_policy": {"runs": [run["id"] for run in all_runs], "receipt_files_per_run": ["run_meta.json", "train_receipt.json", "score_receipt.json"],
                         "read_scope": "JSON receipts only; no prediction artifacts, data, NPZ, NWB, GPU, or official-test material"},
        "invariants": {"formal_epochs": 24, "formal_updates": TOTAL_UPDATES, "frozen_sampler_seed": 42,
                       "source_and_sampler_noncarrier_contract": source_contract, "ho_noncarrier_contract": ho_contract,
                       "source_code_hashes_not_compared_across_old_and_new": True,
                       "three_original_carrier_and_b3s_identity": original_identity,
                       "ho_targets_starts_window_counts": targets},
        "runs": {run["id"]: {key: value for key, value in run.items() if key not in {"source_noncarrier_contract", "ho_noncarrier_contract"}} for run in all_runs},
        "original_three_seed_aggregate": _original_aggregate(originals), "candidate_vs_original_s42": comparison,
        "selection_interpretation": "Selected epochs are independently selected per run on visible HO3 calibration. The original-carrier aggregate contains exactly three seeds.",
    }
    output.mkdir(parents=True)
    _atomic_json(output / "summary.json", body)
    _write_csv(output / "curves.csv", all_runs)
    _atomic_text(output / "README.md", _readme())
    _write_plot(output / "curves.png", output / "curves.svg", originals, candidate42)
    return {"status": "COMPLETED", "summary": str(output / "summary.json"), "csv": str(output / "curves.csv"),
            "plot_png": str(output / "curves.png"), "plot_svg": str(output / "curves.svg")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-s42", type=Path, required=True)
    parser.add_argument("--original-s43", type=Path, required=True)
    parser.add_argument("--original-s44", type=Path, required=True)
    parser.add_argument("--candidate-s42", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
