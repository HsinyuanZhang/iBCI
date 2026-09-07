"""Evaluate a user-stopped continuation's latest common committed EMA epoch.

This is a separate evaluation, not completion of the interrupted 24-epoch run.
It never optimizes, changes a checkpoint, selects on complete scores, resumes
training, or calls a public/official evaluator. All input bytes remain frozen.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import continue_prefix_train as c
from . import continue_prefix_launcher as launcher
from .formal_prefix_score import score_selection_cached_ema, score_complete_cached_ema
from tfpd_exploration.src.family_runtime_v1 import compare_h1_frozen_quality as quality

p, torch = c.parent, c.torch
GO = "H1_USER_STOP_LOCAL_EVAL_GO"
LIMIT = 1800
SCHEMA = "h1_user_stop_common_epoch_local_eval_authorization_v1"
EVAL_GPU, EVAL_CPU = 1, "16-19"


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError("JSON object required")
    return value


def manifest(root):
    return {str(path): launcher.sha(path) for path in sorted(Path(root).rglob("*")) if path.is_file()}


def latest_common_epoch(rows):
    if set(rows) != set(c.ARMS):
        raise RuntimeError("both arms required")
    common = set.intersection(*(set(r) for r in rows.values()))
    if not common:
        raise RuntimeError("no common committed epoch")
    epoch = max(common)
    for arm, records in rows.items():
        if sorted(records) != list(range(13, max(records) + 1)):
            raise RuntimeError("noncontiguous committed continuation history")
        for index, row in records.items():
            score = row.get("selection", {})
            if (row.get("epoch") != index or score.get("n_bins") != 2908
                    or not isinstance(score.get("r2_concat_float64"), (int, float))
                    or not np.isfinite(score["r2_concat_float64"])):
                raise RuntimeError("committed selection record drift")
    return epoch


def collect_authority(stopped, output, stop_receipt):
    stopped, output, stop_receipt = map(launcher.canon, (stopped, output, stop_receipt))
    if launcher.inside(stopped, output) or launcher.inside(output, stopped):
        raise RuntimeError("evaluation output must be separate from stopped run")
    frozen = c.require_authority(stopped)
    parent = Path(frozen["authority"]["bindings"]["parent_formal"])
    if launcher.inside(parent, output) or launcher.inside(output, parent):
        raise RuntimeError("evaluation cannot overlap parent")
    stop = read(stop_receipt)
    failure = read(stopped / "FAILED_OR_INCOMPLETE.json")
    if (stop.get("status") != "STOPPED_BY_USER_NO_AUTO_RESUME"
            or stop.get("stopped_run") != str(stopped)
            or failure.get("reason") != "KeyboardInterrupt()"
            or failure.get("children") != {"flat": -15, "route": -15}
            or (stopped / "receipt.json").exists()):
        raise RuntimeError("expected user-stopped incomplete continuation")
    for pid in stop["terminated_pids"]:
        if Path(f"/proc/{int(pid)}").exists():
            raise RuntimeError("recorded training process still exists")
    rows = {arm: {} for arm in c.ARMS}
    ready = {arm: read(stopped / "barrier" / f"{arm}.ready.json") for arm in c.ARMS}
    for arm in c.ARMS:
        for file in sorted((stopped / "workers").glob(f"{arm}_epoch_*.json")):
            row = read(file)
            epoch = row["epoch"]
            path = stopped / "checkpoints" / f"{arm}_epoch_{epoch:03d}.pt"
            if (row["checkpoint"] != str(path) or launcher.sha(path) != row["checkpoint_sha256"]
                    or row["identities"] != ready[arm]["identities"][str(epoch)]
                    or row["parent_checkpoint_sha256"] != ready[arm]["parent_epoch12_checkpoint_sha256"]):
                raise RuntimeError("stopped checkpoint/record identity drift")
            rows[arm][epoch] = row
    epoch = latest_common_epoch(rows)
    if epoch != stop["latest_common_committed_epoch"]:
        raise RuntimeError("stop snapshot common epoch drift")
    if ready["flat"]["identities"][str(epoch)] != ready["route"]["identities"][str(epoch)]:
        raise RuntimeError("paired common-epoch schedule mismatch")
    files = {Path(__file__), stop_receipt, quality.ORIGINAL_DIR / "receipt.json",
             quality.ORIGINAL_DIR / "input_authority.json",
             quality.ORIGINAL_DIR / "original_h1_minival_native_float64.npz", quality.C2_PATH,
             Path(quality.__file__), Path(quality.__file__).with_name("complete_h1_family_source.py"),
             Path(quality.__file__).with_name("h1_replay_contract.py")}
    return {"schema": SCHEMA, "status": "ROOT_REVIEW_GO", "stopped_run": str(stopped),
            "output": str(output), "stop_receipt": str(stop_receipt),
            "continuation_authority_sha256": c.digest(frozen), "code_closure": c.code_closure(),
            "stopped_artifacts": manifest(stopped), "inputs": {str(x): launcher.sha(x) for x in sorted(files)},
            "epoch": epoch, "checkpoints": {a: rows[a][epoch] for a in c.ARMS},
            "latest_committed_per_arm": {a: max(rows[a]) for a in c.ARMS},
            "protocol": {"selection": "latest common fully committed epoch, fixed before complete evaluation",
                         "weights": "EMA only; reproduce recorded 2908 score then score all20325",
                         "training_updates": 0, "calibration_changes": 0, "official_calls": 0,
                         "limits": {"wall_seconds": LIMIT, "rss_and_gpu_bytes": 22 << 30},
                         "execution": {"physical_gpu": EVAL_GPU, "cpu_affinity": EVAL_CPU, "serial_arms": True},
                         "comparison": "descriptive local pooled/equal-session/per-session; no held-out inference",
                         "stopping": "no automatic training or successor after evaluation"}}


def require_authority(output):
    sidecar = read(Path(output) / "input_authority.json")
    authority = sidecar["authority"]
    path = launcher.canon(sidecar["authorization_path"])
    if launcher.sha(path) != sidecar["authorization_sha256"] or read(path) != authority:
        raise RuntimeError("evaluation external authority drift")
    if collect_authority(Path(authority["stopped_run"]), Path(output), Path(authority["stop_receipt"])) != authority:
        raise RuntimeError("evaluation input/code/selection drift")
    return authority


def worker(*, arm, output):
    started = time.monotonic()
    authority = require_authority(output)
    physical = EVAL_GPU
    if (os.environ.get(GO) != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical)
            or not torch.cuda.is_available() or torch.cuda.current_device() != 0):
        raise RuntimeError("explicit local-evaluation GO and exact GPU required")
    lo, hi = map(int, EVAL_CPU.split("-"))
    if os.sched_getaffinity(0) != set(range(lo, hi + 1)):
        raise RuntimeError("evaluation CPU affinity drift")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()

    def guard():
        p.resource_guard(started, device)
        if time.monotonic() - started > LIMIT:
            raise TimeoutError("bounded stopped-run evaluation")

    stopped = Path(authority["stopped_run"])
    old = c.require_authority(stopped)
    ready = read(stopped / "barrier" / f"{arm}.ready.json")
    epoch = authority["epoch"]
    record = authority["checkpoints"][arm]
    guard(); cache = p.load_readonly_cache(); guard()
    model, optimizer, ema, shared = c._new_model(arm, device)
    if shared != ready["shared_init_sha256"]:
        raise RuntimeError("same model initialization identity required")
    checkpoint = Path(record["checkpoint"])
    if launcher.sha(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA drift before load")
    guard(); payload = torch.load(checkpoint, map_location="cpu", weights_only=False); guard()
    c.strict_restore(payload=payload, model=model, optimizer=optimizer, ema=ema, epoch=epoch,
                     identities=ready["identities"][str(epoch)], arm=arm, shared_init_sha256=shared,
                     bindings=old["authority"]["bindings"],
                     parent_epoch12_sha256=ready["parent_epoch12_checkpoint_sha256"])
    state_before = {"raw": p.state_digest(model.state_dict()), "optimizer": c.tree_digest(optimizer.state_dict()),
                    "ema": c.tree_digest(ema.checkpoint_state()), "rng": c.tree_digest(c.rng_state())}
    selection = p._guarded_score(model, lambda: score_selection_cached_ema(model, ema, cache, device, guard), guard=guard)
    if abs(selection["r2_concat_float64"] - record["selection"]["r2_concat_float64"]) > 1e-5:
        raise RuntimeError("saved selection does not reproduce")
    p.atomic_json({"status": "SELECTION_REPRODUCED_COMPLETE_RUNNING", "epoch": epoch, "selection": selection},
                  output / f"{arm}_live.json")
    complete = p._guarded_score(model, lambda: score_complete_cached_ema(model, ema, cache, device, guard), guard=guard)
    arrays = {key.removeprefix("_"): complete.pop(key) for key in ("_prediction", "_target", "_session_id", "_end")}
    archive = output / "exports" / f"{arm}_epoch_{epoch:03d}_complete_native_float64.npz"
    plain = output / "exports" / f"{arm}_epoch_{epoch:03d}_plain_ema.pt"
    if archive.exists() or plain.exists():
        raise FileExistsError("fresh evaluation exports required")
    with tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    state.update({k: v.detach().cpu().clone() for k, v in ema.shadow.items()})
    if any(not bool(torch.isfinite(v).all()) for v in state.values()):
        raise RuntimeError("nonfinite plain EMA export")
    p.atomic_torch_save(state, plain)
    if not c.recursive_same(state, torch.load(plain, map_location="cpu", weights_only=True)):
        raise RuntimeError("plain EMA disk reproduction drift")
    with np.load(archive, allow_pickle=False) as check:
        if set(check.files) != set(arrays) or any(not np.array_equal(check[k], arrays[k]) for k in arrays):
            raise RuntimeError("prediction archive disk reproduction drift")
    after = {"raw": p.state_digest(model.state_dict()), "optimizer": c.tree_digest(optimizer.state_dict()),
             "ema": c.tree_digest(ema.checkpoint_state()), "rng": c.tree_digest(c.rng_state())}
    if after != state_before:
        raise RuntimeError("evaluation changed training state or RNG")
    guard(); require_authority(output); guard()
    report = {"status": "COMPLETE_LOCAL_EMA_EVALUATION_ONLY", "arm": arm, "epoch": epoch,
              "checkpoint": str(checkpoint), "checkpoint_sha256": launcher.sha(checkpoint),
              "selection_reproduced": selection, "complete": complete,
              "archive": str(archive), "archive_sha256": launcher.sha(archive),
              "plain_ema": str(plain), "plain_ema_sha256": launcher.sha(plain),
              "state_before": state_before, "state_after": after, "training_updates": 0,
              "elapsed_seconds": time.monotonic() - started}
    p.atomic_json(report, output / f"{arm}_report.json")


def compare(output, reports):
    receipt_path = quality.ORIGINAL_DIR / "receipt.json"
    archive_path = quality.ORIGINAL_DIR / "original_h1_minival_native_float64.npz"
    original_receipt = read(receipt_path)
    original_receipt.update({"_path": str(receipt_path), "_npz_path": str(archive_path)})
    original_arrays = quality.validate_archive(archive_path)
    original = quality._original_metric(original_receipt, original_arrays)
    if read(quality.ORIGINAL_DIR / "input_authority.json").get("pre") != original_receipt["pre"]:
        raise RuntimeError("original input authority drift")
    cache_authority = read(p.CACHE_AUTHORITY)
    if (original_receipt["pre"]["cache"]["sha256"] != launcher.sha(p.CACHE)
            or original_receipt["pre"]["authority"]["sha256"] != launcher.sha(p.CACHE_AUTHORITY)):
        raise RuntimeError("original and candidate cache identity drift")
    c2 = quality._c2(read(quality.C2_PATH), cache_authority, original["per_session_r2_float64"])
    tables = {"Original_local_reference": original, "C2_historical_local_aggregate": c2}
    for arm, report in reports.items():
        arrays = quality.validate_archive(Path(report["archive"]))
        quality.same_surface(arrays, original_arrays)
        observed = quality.metric(arrays)
        quality._same_metrics(observed, report["complete"])
        tables[arm] = observed
    return {"tables": tables,
            "deltas_vs_c2": {a: quality._deltas(a, tables[a], c2) for a in c.ARMS},
            "deltas_vs_original": {a: quality._deltas(a, tables[a], original) for a in c.ARMS},
            "c2_prediction_npz_available": False,
            "interpretation": "local descriptive comparison only; not matched training/calibration, no official held-out inference or formal NI",
            "continuation": "PAUSED_NO_AUTO_RESUME"}


def run(*, stopped, output, stop_receipt, authorization, authorization_sha256):
    started = time.monotonic()
    if os.environ.get(GO) != "1" or output.exists():
        raise RuntimeError("explicit GO and fresh output required")
    fresh = collect_authority(stopped, output, stop_receipt)
    authorization = launcher.canon(authorization)
    if launcher.inside(output, authorization) or launcher.inside(stopped, authorization):
        raise RuntimeError("external evaluation authorization required")
    if launcher.sha(authorization) != authorization_sha256 or read(authorization) != fresh:
        raise RuntimeError("exact evaluation authority required")
    if subprocess.check_output(["nvidia-smi", "-i", str(EVAL_GPU), "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("assigned evaluation GPU must be free")
    output.mkdir(); (output / "exports").mkdir()
    p.atomic_json({"authorization_path": str(authorization), "authorization_sha256": authorization_sha256,
                   "authority": fresh}, output / "input_authority.json")
    sidecar_sha = launcher.sha(output / "input_authority.json")
    children, handles = {}, []
    try:
        for arm in c.ARMS:
            if subprocess.check_output(["nvidia-smi", "-i", str(EVAL_GPU), "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                raise RuntimeError("assigned evaluation GPU occupied before next arm")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(EVAL_GPU), "PYTHONNOUSERSITE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            log = (output / f"{arm}_evaluation.log").open("x"); handles.append(log)
            code = "from pathlib import Path;import sys;from tfpd_exploration.src.h1_queryage_family_v1.stopped_eval import worker;worker(arm=sys.argv[1],output=Path(sys.argv[2]))"
            children[arm] = subprocess.Popen(["taskset", "-c", EVAL_CPU, sys.executable,
                                             "-c", code, arm, str(output)], env=env,
                                            cwd=p.ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            while children[arm].poll() is None:
                if time.monotonic() - started > LIMIT:
                    raise TimeoutError("local evaluation supervisor deadline")
                time.sleep(.2)
            if children[arm].returncode != 0:
                raise RuntimeError("local evaluation child failed")
        if any(child.returncode != 0 for child in children.values()):
            raise RuntimeError("local evaluation child failed")
        launcher.close_handles(handles)
        reports = {arm: read(output / f"{arm}_report.json") for arm in c.ARMS}
        for arm, report in reports.items():
            if (report["status"] != "COMPLETE_LOCAL_EMA_EVALUATION_ONLY" or report["epoch"] != fresh["epoch"]
                    or report["checkpoint_sha256"] != fresh["checkpoints"][arm]["checkpoint_sha256"]
                    or launcher.sha(report["archive"]) != report["archive_sha256"]
                    or launcher.sha(report["plain_ema"]) != report["plain_ema_sha256"]):
                raise RuntimeError("evaluation report/export drift")
        comparison = compare(output, reports)
        if require_authority(output) != fresh or launcher.sha(output / "input_authority.json") != sidecar_sha:
            raise RuntimeError("post evaluation authority drift")
        result = {"schema": "h1_user_stopped_common_epoch_local_eval_v1",
                  "status": "COMPLETE_LOCAL_EVALUATION_TRAINING_PAUSED", "epoch": fresh["epoch"],
                  "authority": fresh, "reports": reports, "comparison": comparison,
                  "elapsed_seconds": time.monotonic() - started, "owned_artifacts": manifest(output)}
        p.atomic_json(result, output / "receipt.json")
        return result
    except BaseException as error:
        launcher.stop(children)
        p.atomic_json({"status": "LOCAL_EVALUATION_FAILED_TRAINING_STILL_PAUSED", "reason": repr(error)},
                      output / "FAILED.json")
        raise
    finally:
        launcher.close_handles(handles)


def main():
    parser = argparse.ArgumentParser()
    for name in ("stopped", "output", "stop-receipt", "authorization"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
