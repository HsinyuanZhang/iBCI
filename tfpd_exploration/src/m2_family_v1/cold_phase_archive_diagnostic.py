"""Archive-only, fixed-endpoint analysis of the completed cold-history 2x2.

No training/model/data-loader imports. No selection, fitting, or new inference.
The all-1011 equal-session score remains primary; startup partitions are
descriptive and cannot be substituted for that surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "tfpd_exploration/results/m2/family_v1"
PHASE = FAMILY / "cold_history_2x2_phase_v1"
OUTPUT = FAMILY / "cold_history_2x2_archive_diagnostic_v1.json"
FINAL = FAMILY / "finalize_pair_v1/receipt.json"
FINAL_SHA = "3be4a096f98bd9f77726094501271e2836e37c6f61c06ce136ce72505488cf47"
BASELINE = FAMILY / "source_minival_e8_spint_m30_replay_v1.json"
BASELINE_NPZ = BASELINE.with_suffix(".npz")
CELLS = ("FLAT_CONTROL", "FLAT_PREFIX", "ROUTE_CONTROL", "ROUTE_PREFIX")
COUNTS = (173, 129, 117, 116, 141, 141, 194)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def metrics(target, prediction, sessions) -> dict:
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.ndim != 2 or target.shape != prediction.shape or target.shape[1] != 2 or len(target) != len(sessions):
        raise RuntimeError("metric array shape drift")
    if not len(target) or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise RuntimeError("empty/nonfinite metric arrays")

    def one(y, p):
        denominator = float(np.square(y - y.mean(0)).sum())
        if denominator <= 0:
            raise RuntimeError("undefined zero-variance diagnostic")
        return {"count": len(y), "r2": float(1 - np.square(y - p).sum() / denominator),
                "mse": float(np.square(y - p).mean())}

    per = {str(s): one(target[sessions == s], prediction[sessions == s]) for s in dict.fromkeys(sessions.tolist())}
    return {"pooled": one(target, prediction), "per_session": per,
            "equal_session_r2": float(np.mean([r["r2"] for r in per.values()]))}


def check_arrays(arrays: dict, reference: dict) -> None:
    if set(arrays) != {"prediction", "target", "session", "start"}:
        raise RuntimeError("phase archive field drift")
    if arrays["prediction"].shape != (1011, 2) or not np.isfinite(arrays["prediction"]).all():
        raise RuntimeError("1011 finite native predictions required")
    for key in ("target", "session", "start"):
        if not np.array_equal(arrays[key], reference[key]):
            raise RuntimeError(f"phase archive exact {key} identity drift")


def load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def summarize(arrays: dict) -> dict:
    y, p, s, start = (arrays[k] for k in ("target", "prediction", "session", "start"))
    masks = {"all1011": np.ones(len(start), dtype=bool), "startup343_start_lt49": start < 49,
             "full_history668_start_ge49": start >= 49}
    if [int(mask.sum()) for mask in masks.values()] != [1011, 343, 668]:
        raise RuntimeError("fixed diagnostic partition count drift")
    return {name: metrics(y[mask], p[mask], s[mask]) for name, mask in masks.items()}


def effect(prefix: dict, control: dict) -> dict:
    result = {}
    for partition in prefix:
        a, b = prefix[partition], control[partition]
        if a["per_session"].keys() != b["per_session"].keys():
            raise RuntimeError("effect session alignment drift")
        deltas = {s: a["per_session"][s]["r2"] - b["per_session"][s]["r2"] for s in a["per_session"]}
        result[partition] = {"equal_session_r2_delta": a["equal_session_r2"] - b["equal_session_r2"],
                             "pooled_r2_delta": a["pooled"]["r2"] - b["pooled"]["r2"],
                             "pooled_mse_delta": a["pooled"]["mse"] - b["pooled"]["mse"],
                             "per_session_r2_delta": deltas,
                             "positive_session_count": sum(v > 0 for v in deltas.values())}
    return result


def run(phase: Path = PHASE, output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(output)
    # Must fail before even reading historical arrays if the run is unfinished.
    report_path = phase / "report.json"
    report = read(report_path)
    if (report.get("status") != "COMPLETE_FIXED_ENDPOINT2_DIAGNOSTIC"
            or report.get("selected_or_promoted_model") is not None
            or report.get("old_experiment_modified") is not False):
        raise RuntimeError("completed diagnostic-only fixed endpoint2 report required")
    if sha(FINAL) != FINAL_SHA or report["authority"]["finalizer_receipt_sha256"] != FINAL_SHA:
        raise RuntimeError("original finalizer authority drift")
    original = read(FINAL)
    if sha(BASELINE) != original["baseline_receipt_sha256"]:
        raise RuntimeError("historical baseline receipt drift")
    if sha(BASELINE_NPZ) != read(BASELINE)["stream_npz_sha256"]:
        raise RuntimeError("historical baseline prediction archive drift")
    protocol = report["authority"]["protocol"]
    if protocol["epochs"] != 2 or protocol["cells"] != list(CELLS) or protocol["updates_per_epoch_per_cell"] != 3165:
        raise RuntimeError("fixed experiment cardinality drift")
    pretrain_path = phase / "pretrain_authority.json"
    pretrain = read(pretrain_path)
    if pretrain["authority"] != report["authority"]:
        raise RuntimeError("pretrain/report authority mismatch")
    for arm in ("FLAT", "ROUTE"):
        if pretrain["initial_cell_sha256"][arm + "_CONTROL"] != pretrain["initial_cell_sha256"][arm + "_PREFIX"]:
            raise RuntimeError("within-arm initialization mismatch")

    files = {str(p): sha(p) for p in (Path(__file__), FINAL, BASELINE, BASELINE_NPZ, report_path, pretrain_path)}

    def bind(path, expected):
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"artifact hash drift: {path.name}")
        files[str(path)] = actual

    historical = load_arrays(BASELINE_NPZ)
    reference = {key: historical[key] for key in ("target", "start", "session")}
    names = list(dict.fromkeys(reference["session"].tolist()))
    if tuple(int((reference["session"] == s).sum()) for s in names) != COUNTS:
        raise RuntimeError("historical seven-session cardinality drift")
    for name, count in zip(names, COUNTS, strict=True):
        if not np.array_equal(reference["start"][reference["session"] == name], np.arange(count)):
            raise RuntimeError("historical within-session starts drift")
    analyses, endpoint_arrays = {}, {}
    for epoch in (1, 2):
        path = phase / f"epoch_{epoch:03d}_metrics.json"
        recorded = read(path)
        if recorded != report["history"][str(epoch)] or recorded["epoch"] != epoch or set(recorded["cells"]) != set(CELLS):
            raise RuntimeError("epoch metric/report history mismatch")
        files[str(path)] = sha(path)
        analyses[str(epoch)] = {}
        for cell in CELLS:
            row = recorded["cells"][cell]
            bind(phase / cell / f"epoch_{epoch:03d}.pt", row["checkpoint_sha256"])
            if set(row["scores"]) != {"RAW", "EMA"}:
                raise RuntimeError("all RAW and EMA scores required")
            analyses[str(epoch)][cell] = {}
            for kind in ("RAW", "EMA"):
                score = row["scores"][kind]
                path = phase / cell / f"epoch_{epoch:03d}_{kind.lower()}_native.npz"
                bind(path, score["archive_sha256"])
                arrays = load_arrays(path)
                check_arrays(arrays, reference)
                summary = summarize(arrays)
                if (abs(summary["all1011"]["equal_session_r2"] - score["equal_session_r2"]) > 1e-10
                        or abs(summary["all1011"]["pooled"]["r2"] - score["pooled_r2"]) > 1e-10):
                    raise RuntimeError("archive metric reproduction mismatch")
                analyses[str(epoch)][cell][kind] = summary
                if epoch == 2 and kind == "EMA":
                    endpoint_arrays[cell] = arrays
    for cell in CELLS:
        export = report["strict_fixed_endpoint_exports"][cell]
        expected_path = phase / cell / "endpoint2_ema_state.pt"
        if export["export_path"] != str(expected_path):
            raise RuntimeError("fixed endpoint export path drift")
        bind(expected_path, export["export_sha256"])
        path = phase / cell / "endpoint2_ema_replay_native.npz"
        bind(path, export["replay_archive_sha256"])
        arrays = load_arrays(path)
        check_arrays(arrays, reference)
        np.testing.assert_allclose(arrays["prediction"], endpoint_arrays[cell]["prediction"], atol=1e-5, rtol=1e-5)

    effects = {str(epoch): {arm: {kind: effect(analyses[str(epoch)][arm + "_PREFIX"][kind],
                                              analyses[str(epoch)][arm + "_CONTROL"][kind])
                                        for kind in ("RAW", "EMA")} for arm in ("FLAT", "ROUTE")}
               for epoch in (1, 2)}
    for arm in ("FLAT", "ROUTE"):
        if abs(effects["2"][arm]["EMA"]["all1011"]["equal_session_r2_delta"] - report["primary_prefix_minus_control"][arm]) > 1e-10:
            raise RuntimeError("primary effect mismatch")
    post = {path: sha(Path(path)) for path in files}
    if files != post:
        raise RuntimeError("archive/code authority changed during diagnostic")
    result = {"schema": "m2_cold_history_2x2_archive_diagnostic_v1", "status": "COMPLETE_ARCHIVE_ONLY_NO_SELECTION",
              "primary": "fixed epoch2 EMA all1011 equal-session PREFIX minus matched CONTROL",
              "qualification": "post-hoc source-minival-motivated training diagnostic; partitions descriptive, not untouched generalization or formal noninferiority",
              "inputs_sha256_pre": files, "inputs_sha256_post": post, "analysis": analyses,
              "prefix_minus_control": effects,
              "historical": {name: summarize({**reference, "prediction": historical[name + "_prediction"]}) for name in ("e8", "spint")},
              "model_forwards": 0, "parameter_updates": 0, "selection_or_promotion": None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(output=args.output)
    print(json.dumps({"status": result["status"], "fixed_epoch2_ema_effect": {
        arm: result["prefix_minus_control"]["2"][arm]["EMA"]["all1011"] for arm in ("FLAT", "ROUTE")}}))
