"""Root-operated audit and date-equal collection of all nine temporal cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from protocol import ARMS, DATASETS, OUT, ROOT, atomic_json, protocol, sha

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "btransform_unified_v1/src"))
from btransform_unified_v1.r2 import variance_weighted_r2


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def array_hash(value):
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.dtype.str.encode() + str(arr.shape).encode() + arr.tobytes()).hexdigest()


def collect(dest: Path):
    summary_path, audit_path = dest / "summary.json", dest / "collection_receipt.json"
    require(not summary_path.exists() and not audit_path.exists(), "refuse replacing completed collection")
    bound = {}

    def bind(path, expected=None):
        path = path.resolve()
        value = sha(path)
        require(expected is None or value == expected, f"changed input: {path}")
        bound[str(path)] = value
        return value

    bind(HERE / "collect_results.py")
    protocol_sha = bind(dest / "protocol.json")
    require(read(dest / "protocol.json") == protocol(), "temporal protocol changed")
    program = read(dest / "program.json")
    bind(dest / "program.json")
    require(program.get("status") == "COMPLETED" and program.get("protocol_sha256") == protocol_sha, "all temporal jobs must complete before collection")
    cells = {(c["dataset"], c["arm"]): c for c in program["cells"]}
    require(len(program["cells"]) == 9 and set(cells) == {(d, a) for d in DATASETS for a in ARMS}, "incomplete nine-cell roster")
    require(all(c["status"] == "COMPLETED" and c.get("pid") is None for c in cells.values()), "unfinished temporal cell")
    for files in program["dataset_code_sha256"].values():
        for raw, digest in files.items():
            bind(Path(raw), digest)
    for (dataset, arm), cell in cells.items():
        if dataset == "m2":
            bind(Path(cell["reuse_receipt"]), cell["reuse_receipt_sha256"])
        else:
            run = Path(cell["run"])
            bind(run / "preflight.json", cell["preflight_sha256"])
            bind(run / "train_receipt.json", cell["train_receipt_sha256"])
            for raw, digest in cell["score_receipt_sha256"].items():
                bind(Path(raw), digest)

    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONWARNINGS="ignore", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    commands = [
        ("m1_paired_audit", [sys.executable, "-u", str(HERE / "m1_audit.py"), "--run-root", str(dest / "m1"), "--dest", str(dest / "m1_paired_audit.json")]),
        ("h1_paired_audit", [sys.executable, "-u", str(HERE / "h1_audit.py"), "--prepared", str(dest / "h1/prepared"), "--run-root", str(dest / "h1"), "--arms", *ARMS, "--dest", str(dest / "h1_paired_audit.json")]),
    ]
    for name, command in commands:
        require(not (dest / f"{name}.json").exists(), f"existing {name}: inspect before any rerun")
        with (dest / "logs" / f"{name}.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT.parent, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        require(read(dest / f"{name}.json").get("status") == "PASSED", f"{name} failed")
        bind(dest / f"{name}.json")
        print(f"{name}: PASSED", flush=True)

    m1 = read(dest / "m1_paired_audit.json")
    rows = list(m1["rows"])
    require(len(rows) == 6, "M1 audit must provide both dates for all three arms")
    h1_inputs = {}
    for arm in ARMS:
        path = dest / "h1" / arm / "target_score.json"
        score = read(path)
        bind(path)
        for date in DATASETS["h1"]["target_dates"]:
            sessions = [s for s in DATASETS["h1"]["target_sessions"] if DATASETS["h1"]["session_dates"][s] == date]
            values = {}
            for label, key in (("selected_ema", "selected_r2"), ("fixed_e32_ema", "fixed_r2")):
                target = score[label]["target"]
                value = float(np.mean([target["per_session"][s]["r2"] for s in sessions]))
                require(math.isfinite(value) and abs(value - target["per_date"][date]) <= 1e-12, "H1 date aggregation drift")
                values[key] = value
                for session in sessions:
                    item = target["per_session"][session]
                    bind(Path(item["artifact"]), item["artifact_sha256"])
            rows.append({"dataset": "h1", "date": date, "arm": arm, **values, "selected_epoch": score["selected_source_epoch"], "session_count": len(sessions)})
        h1_inputs[arm] = {"score_sha256": sha(path), "selected_source_epoch": score["selected_source_epoch"]}

    reuse = read(dest / "m2/m2_reuse_receipt.json")
    bind(dest / "m2/m2_reuse_receipt.json")
    require(reuse.get("schema") == "chronological_m2_reuse_v1" and reuse.get("status") == "COMPLETED", "unverified M2 reuse")
    require(reuse["source7"]["sessions"] == DATASETS["m2"]["source_sessions"] and reuse["target_policy"]["main_fixed_sessions"] == DATASETS["m2"]["target_sessions"], "M2 reuse temporal roster changed")
    for raw, digest in reuse["source_files_sha256"].items():
        bind(Path(raw), digest)
    for raw, digest in reuse["scientific_training"]["in_process_bound_input_sha256"].items():
        bind(Path(raw), digest)
    for name in ("source_pairing_audit", "paired_artifact_audit"):
        entry = reuse[name]
        bind(Path(entry["path"]), entry["sha256"])
    m2_pairs = {}
    for arm in ARMS:
        entry = reuse["arms"][arm]
        bind(Path(entry["old_score_receipt"]["path"]), entry["old_score_receipt"]["sha256"])
        training = reuse["scientific_training"]["train_receipts"][arm]
        bind(Path(training["path"]), training["sha256"])
        bind(Path(training["run_meta_path"]), training["run_meta_sha256"])
        for package in entry["verified_ema_package_files"].values():
            bind(Path(package["path"]), package["sha256"])
        for session in DATASETS["m2"]["target_sessions"]:
            values = {}
            for endpoint, key in (("selected_source_epoch", "selected_r2"), ("fixed_e24", "fixed_r2")):
                item = entry["main_fixed_last2"][endpoint][session]
                path = Path(item["artifact_absolute_path"])
                bind(path, item["artifact_sha256"])
                with np.load(path, allow_pickle=False) as z:
                    require(set(z.files) == {"prediction", "target", "eligible_starts"}, "M2 prediction artifact schema changed")
                    target, prediction, starts = z["target"], z["prediction"], z["eligible_starts"]
                    require(target.shape == prediction.shape and len(target) == len(starts) and len(target) > 0 and np.isfinite(target).all() and np.isfinite(prediction).all(), "M2 prediction geometry/nonfinite")
                    value = float(variance_weighted_r2(target, prediction))
                    require(math.isfinite(value) and abs(value - item["r2"]) <= 1e-12, "M2 R2 recomputation changed")
                    payload = (array_hash(target), array_hash(starts))
                    require(m2_pairs.setdefault(session, payload) == payload, "M2 labels/coordinates differ across arms or endpoints")
                values[key] = value
            rows.append({"dataset": "m2", "date": DATASETS["m2"]["session_dates"][session], "arm": arm, **values, "selected_epoch": entry["source_selected_epoch"], "session_count": 1})

    expected = {(dataset, date, arm) for dataset in DATASETS for date in DATASETS[dataset]["target_dates"] for arm in ARMS}
    require(len(rows) == 18 and {(r["dataset"], r["date"], r["arm"]) for r in rows} == expected, "summary must contain all 18 date/arm rows")
    rows.sort(key=lambda r: (tuple(DATASETS).index(r["dataset"]), r["date"], ARMS.index(r["arm"])))
    summary = {"schema": "chronological_last2_summary_v1", "status": "PASSED", "protocol_sha256": protocol_sha,
               "scope": "Public temporal transfer, one shared source fit per arm for the last two scoreable dates. Equal-session within date, then equal-date means. Source-only EMA selection, zero target optimizer steps. Single seed42; descriptive paired increments, no statistical noninferiority or hidden-test claim.", "rows": rows}
    atomic_json(summary_path, summary)
    receipt = {"schema": "chronological_last2_collection_receipt_v1", "status": "PASSED", "protocol_sha256": protocol_sha,
               "summary_sha256": sha(summary_path), "bound_input_sha256": dict(sorted(bound.items())), "source_fits": 9, "fresh_fits": 6, "reused_fits": 3, "date_arm_rows": 18,
               "h1_inputs": h1_inputs, "scope": summary["scope"]}
    atomic_json(audit_path, receipt)
    print(json.dumps({"summary": str(summary_path), "sha256": receipt["summary_sha256"], "status": "PASSED"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=OUT)
    arguments = parser.parse_args()
    collect(arguments.dest.resolve())
