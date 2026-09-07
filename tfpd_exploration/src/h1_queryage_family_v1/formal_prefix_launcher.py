"""Root-reviewed supervisor for the additive H1 QueryAge+prefix candidate.

Importing this module constructs no model, opens no cache, and launches no job.
Smoke and formal authorizations are distinct, externally hash-bound artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"flat": 0, "route": 1}
CPU = {"flat": "12-15", "route": "8-11"}
GO = "H1_QUERYAGE_FORMAL_PREFIX_GO"
SCHEMA = "h1_queryage_formal_prefix_root_authorization_v1"
SELECTION_BINS = 2908
SOURCE = ROOT / "results/family_runtime_v1"
SOURCE_AUTH = {
    "smoke20": ("h1_queryage_source_smoke20_authorization_v1.json", "c2db3a4a12a3181a4b20b1700f9ee1b7cf9a122140e396023065747b01f1645f"),
    "capacity260": ("h1_queryage_source_capacity260_authorization_v1.json", "c1dfbc25134f050808662a95a10147350775142c05319aa8ec04c65dda69bef1"),
    "extend1040": ("h1_queryage_source_extend1040_authorization_v1.json", "fd7cb61c7a8b444c21f3b557e3cb07419738a71a7487831a1c3f5fbe20b81c3c"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError("JSON object required")
    return value


def atomic(path, value):
    with tempfile.NamedTemporaryFile("w", dir=Path(path).parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def canonical(path):
    p = Path(path)
    if not p.is_absolute() or p.resolve() != p:
        raise RuntimeError("canonical absolute path required")
    return p


def inside(parent, child):
    return os.path.commonpath((str(parent), str(child))) == str(parent)


def capacity_audit(receipt, checkpoint):
    """Recompute the actual complete1040 gate and its three-stage provenance."""
    from . import source_capacity as capacity
    receipt, checkpoint = canonical(receipt), canonical(checkpoint)
    smoke_path = SOURCE / "h1_queryage_source_smoke20_v1/receipt.json"
    prior_path = SOURCE / "h1_queryage_source_capacity260_v1/receipt.json"
    prior_checkpoint = prior_path.with_name("capacity_checkpoint.pt")
    body = read(receipt)
    current = capacity.bindings(receipt.parent, mode="extend1040", physical_gpu=0,
                                smoke_receipt=smoke_path, resume_checkpoint=prior_checkpoint,
                                capacity_receipt=prior_path)
    smoke = capacity.validate_smoke(smoke_path, current)
    prior = capacity.validate_prior(prior_path, prior_checkpoint, current)
    if (body.get("schema") != "h1_queryage_source_capacity_v1"
            or body.get("status") != "COMPLETE_SOURCE_CAPACITY_NO_FORMAL"
            or body.get("mode") != "extend1040" or body.get("updates") != 1040
            or body.get("pre") != current or body.get("post") != current
            or body.get("minival_rows_used") is not False
            or len(body.get("losses", [])) != 780
            or body.get("history_losses") != prior["history_losses"] + body["losses"]
            or body.get("checkpoint") != {"path": str(checkpoint), "sha256": sha(checkpoint)}):
        raise RuntimeError("complete1040 source/checkpoint/history provenance drift")
    capacity._owned_receipt(body, receipt)
    if (not capacity.capacity_eligible("extend1040", body["scores"], body["history_losses"])
            or body.get("eligible_for_next_stage") is not True):
        raise RuntimeError("recomputed BOTH-arm1040 gate failed")
    authorization_files = {}
    for mode, stage in (("smoke20", smoke), ("capacity260", prior), ("extend1040", body)):
        name, expected = SOURCE_AUTH[mode]; path = SOURCE / name
        auth = read(path)
        if sha(path) != expected or auth.get("status") != "ROOT_REVIEW_GO" or auth.get("bindings") != stage["pre"]:
            raise RuntimeError("source-stage external authorization drift")
        authorization_files[str(path)] = expected
    return {"status": "PASS_ACTUAL_BOTH1040_ELIGIBILITY_ONLY", "receipt": str(receipt),
            "receipt_sha256": sha(receipt), "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint), "source_authorizations": authorization_files,
            "pooled": {a: body["scores"][a]["pooled"] for a in ARMS},
            "capacity_model_state_used_for_formal_warmstart": False}


def stop_children(children):
    """Only stop processes created by this supervisor; never discover/kill peers."""
    for child in children.values():
        if child.poll() is None:
            child.terminate()
    for child in children.values():
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(timeout=15)


def check_children(children, started, limit):
    if any(child.poll() not in (None, 0) for child in children.values()):
        raise RuntimeError("owned child failed; no automatic restart")
    if time.monotonic() - started > limit:
        raise TimeoutError("global supervisor wall budget exceeded")


def check_external(authorization, expected_sha, expected_body):
    if sha(authorization) != expected_sha or read(authorization) != expected_body:
        raise RuntimeError("external root authorization changed")


def _spawn(module, function, arm, output, extra, handles):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(ARMS[arm]), GO: "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"}
    # Values are JSON-decoded, not interpolated into executable Python syntax.
    arguments = {"arm": arm, "output": str(output), "physical_gpu": ARMS[arm], **extra}
    script = ("import json,sys; from pathlib import Path; "
              f"from tfpd_exploration.src.h1_queryage_family_v1.{module} import {function}; "
              "a=json.loads(sys.argv[1]); "
              "a={k:Path(v) if k in ('output','start_marker','authorization','peer_ready','capacity_receipt','capacity_checkpoint') else v for k,v in a.items()}; "
              f"{function}(**a)")
    handle = (output / f"{arm}_{function}.log").open("x")
    handles.append(handle)
    return subprocess.Popen(["taskset", "-c", CPU[arm], sys.executable, "-c", script,
                             json.dumps(arguments)], cwd=ROOT.parent, env=env,
                            stdout=handle, stderr=subprocess.STDOUT)


def _start_and_wait(children, ready_paths, start_marker, started, limit, validate_ready):
    while not all(p.is_file() for p in ready_paths.values()):
        check_children(children, started, limit)
        if any(p.poll() == 0 for p in children.values()):
            raise RuntimeError("child exited before paired-ready barrier")
        time.sleep(.1)
    ready = {arm: read(path) for arm, path in ready_paths.items()}
    validate_ready(ready)
    atomic(start_marker, {"status": "ROOT_RELEASED_MATCHED_PAIR", "ready_sha256": {a: sha(p) for a, p in ready_paths.items()}})
    while any(p.poll() is None for p in children.values()):
        check_children(children, started, limit); time.sleep(.2)
    check_children(children, started, limit)
    return ready


def manifest(output):
    return {str(p): sha(p) for p in sorted(Path(output).rglob("*"))
            if p.is_file() and p != Path(output) / "receipt.json"}


def collect_smoke_authority(output, capacity_receipt, capacity_checkpoint, arm_authorizations):
    from . import formal_prefix_smoke as smoke
    from . import formal_prefix_train as train
    output = canonical(output)
    if set(arm_authorizations) != set(ARMS):
        raise RuntimeError("both external arm authorizations required")
    arms = {}
    for arm, gpu in ARMS.items():
        item = arm_authorizations[arm]; path = canonical(item["path"])
        binding = smoke.bindings(output / arm, capacity_receipt=capacity_receipt,
                                 capacity_checkpoint=capacity_checkpoint, physical_gpu=gpu)
        smoke._authorize(binding, path, item["sha256"])
        if inside(output, path):
            raise RuntimeError("smoke arm authorization must be external to output")
        arms[arm] = {"path": str(path), "sha256": item["sha256"], "bindings": binding}
    return {"schema": SCHEMA, "status": "ROOT_REVIEW_GO", "mode": "smoke",
            "output": str(output), "capacity_audit": capacity_audit(capacity_receipt, capacity_checkpoint),
            "code_closure": train.code_closure(), "arms": arms}


def validate_ready(ready):
    import math
    if set(ready) != set(ARMS):
        raise RuntimeError("exact paired readiness required")
    for arm, item in ready.items():
        ids = item.get("identities")
        if (item.get("arm") != arm or item.get("physical_gpu") != ARMS[arm]
                or item.get("g0_parity", {}).get("max_abs_diff") != 0.
                or not math.isfinite(float(item.get("route_gate_gradient_l1", 0)))
                or float(item.get("route_gate_gradient_l1", 0)) <= 0
                or not isinstance(ids, dict) or set(ids) != {str(i) for i in range(1, 13)}
                or any(set(v) != {"sampler_sha256", "keep_sha256", "prefix_sha256"} for v in ids.values())):
            raise RuntimeError("exact fresh paired readiness/identity evidence missing")
    if (ready["flat"]["shared_init_sha256"] != ready["route"]["shared_init_sha256"]
            or ready["flat"]["identities"] != ready["route"]["identities"]):
        raise RuntimeError("paired initialization/sampler/dropout/prefix mismatch")


def _smoke_results(output):
    import math
    from . import formal_prefix_smoke as smoke
    ready = {a: read(output / a / "ready.json") for a in ARMS}
    validate_ready(ready)
    results, forecasts = {}, []
    for arm, gpu in ARMS.items():
        p = output / arm / "receipt.json"; value = read(p)
        if (value.get("schema") != smoke.SCHEMA or value.get("status") != "PASS_DISPOSABLE_SOURCE_ONLY_SMOKE"
                or value.get("arm") != arm or value.get("authority") != value.get("authority_post")
                or value.get("parameter_updates") != 20 or value.get("checkpoint_retained") is not False
                or value.get("capacity_state_used_for_warmstart") is not False
                or len(value.get("losses", [])) != 20 or not all(math.isfinite(v) for v in value["losses"])
                or not 0 < value.get("peak_memory_bytes", 0) <= 22 << 30
                or not 0 < value.get("elapsed_seconds", 0) <= 900
                or (output / arm / "known_disposable_smoke_checkpoint.pt").exists()):
            raise RuntimeError("actual disposable smoke result/resource contract drift")
        forecast = smoke._forecast(value["timing"], value["source208_seconds_per_endpoint"], value["reload_seconds_measured"])
        if forecast != value.get("forecast") or forecast["conservative_total_seconds"] > 21600:
            raise RuntimeError("exact-candidate conservative formal resource gate failed")
        for file, key in ((output / arm / "ready.json", "ready_sha256"),
                          (output / arm / "input_authority.json", "input_authority_sha256"),
                          (output / ("route" if arm == "flat" else "flat") / "ready.json", "peer_ready_sha256")):
            if sha(file) != value.get(key):
                raise RuntimeError("smoke own/peer receipt artifact drift")
        results[arm] = {"path": str(p), "sha256": sha(p), "result": value}
        forecasts.append(forecast["conservative_total_seconds"])
    return results, max(forecasts)


def audit_smoke(receipt, capacity_receipt, capacity_checkpoint):
    receipt = canonical(receipt); output = receipt.parent; value = read(receipt)
    if (value.get("schema") != "h1_queryage_formal_prefix_pair_smoke_v1"
            or value.get("status") != "PASS_PAIRED_SOURCE_ONLY_RESOURCE_SMOKE"
            or value.get("owned_artifact_sha256") != manifest(output)):
        raise RuntimeError("completed paired smoke receipt/manifest required")
    external = canonical(value["authorization_path"])
    authority = collect_smoke_authority(output, capacity_receipt, capacity_checkpoint, value["authority"]["arms"])
    check_external(external, value["authorization_sha256"], authority)
    arms, forecast = _smoke_results(output)
    if value.get("authority") != authority or value.get("authority_post") != authority or value.get("arms") != arms or value.get("conservative_formal_seconds") != forecast:
        raise RuntimeError("fresh completed smoke closure/results drift")
    for arm in ARMS:
        if arms[arm]["result"]["authority"] != authority["arms"][arm]["bindings"]:
            raise RuntimeError("paired smoke arm is not externally authorized candidate")
    return {"receipt": str(receipt), "receipt_sha256": sha(receipt),
            "conservative_formal_seconds": forecast, "authority": authority,
            "arms": {a: {"path": v["path"], "sha256": v["sha256"]} for a, v in arms.items()}}


def collect_formal_authority(output, capacity_receipt, capacity_checkpoint, smoke_receipt):
    from . import formal_prefix_train as train
    return {"schema": SCHEMA, "status": "ROOT_REVIEW_GO", "mode": "formal",
            "bindings": train.collect_bindings(canonical(output), capacity_receipt=canonical(capacity_receipt),
                                                capacity_checkpoint=canonical(capacity_checkpoint), smoke_receipt=canonical(smoke_receipt)),
            "capacity_audit": capacity_audit(capacity_receipt, capacity_checkpoint),
            "smoke_audit": audit_smoke(smoke_receipt, capacity_receipt, capacity_checkpoint)}


def freeze_selection(output, ready):
    from . import formal_prefix_train as train
    selected, epoch12, completions = {}, {}, {}
    for arm in ARMS:
        completion = read(output / "workers" / f"{arm}_complete.json")
        records = completion.get("epochs", [])
        pick = train.earliest_argmax([(r["epoch"], r["selection"]["r2_concat_float64"]) for r in records])
        if (completion.get("arm") != arm or completion.get("shared_init_sha256") != ready[arm]["shared_init_sha256"]
                or completion.get("identities") != ready[arm]["identities"]
                or pick != (completion.get("selected_epoch"), completion.get("selected_ema_r2_float64"))):
            raise RuntimeError("worker complete all-epoch selection/identity drift")
        for row in records:
            epoch = row["epoch"]; p = output / "checkpoints" / f"{arm}_epoch_{epoch:03d}.pt"
            if (row.get("checkpoint") != str(p) or row.get("checkpoint_sha256") != sha(p)
                    or row.get("selection", {}).get("n_bins") != SELECTION_BINS
                    or row.get("identities") != ready[arm]["identities"][str(epoch)]
                    or read(output / "workers" / f"{arm}_epoch_{epoch:03d}.json") != row):
                raise RuntimeError("epoch checkpoint/selection identity drift")
        def entry(row):
            return {"epoch": row["epoch"], "ema_r2_float64": row["selection"]["r2_concat_float64"],
                    "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"]}
        selected[arm], epoch12[arm], completions[arm] = entry(records[pick[0] - 1]), entry(records[-1]), completion
    if completions["flat"]["identities"] != completions["route"]["identities"]:
        raise RuntimeError("final cross-arm recipe identity drift")
    freeze = {"schema": "h1_queryage_formal_prefix_selection_freeze_v1", "selected": selected, "epoch12": epoch12}
    atomic(output / "selection_freeze.json", freeze)
    return freeze


def run(mode, output, authorization, authorization_sha, capacity_receipt, capacity_checkpoint, smoke_receipt=None):
    started = time.monotonic(); output, authorization = canonical(output), canonical(authorization)
    if os.environ.get(GO) != "1" or mode not in ("smoke", "formal") or output.exists() or inside(output, authorization):
        raise RuntimeError("explicit GO/mode/fresh output/external authority required")
    actual = read(authorization)
    if mode == "smoke":
        fresh = collect_smoke_authority(output, capacity_receipt, capacity_checkpoint, actual["arms"])
    else:
        fresh = collect_formal_authority(output, capacity_receipt, capacity_checkpoint, smoke_receipt)
    check_external(authorization, authorization_sha, fresh)
    if actual != fresh:
        raise RuntimeError("root-reviewed fresh authority mismatch")
    # Only the two dedicated GPUs are in scope. Refuse a launch if either is busy.
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("both dedicated GPUs must be free before paired launch")
    output.mkdir(parents=True); children, handles = {}, []
    input_body = {"authorization_path": str(authorization), "authorization_sha256": authorization_sha,
                  "authority": fresh, "bindings": fresh.get("bindings"), "capacity_audit": fresh["capacity_audit"]}
    atomic(output / "input_authority.json", input_body); input_sha = sha(output / "input_authority.json")
    limit = 900 if mode == "smoke" else 21600
    try:
        if mode == "smoke":
            for arm in ARMS:
                child_auth = fresh["arms"][arm]
                children[arm] = _spawn("formal_prefix_smoke", "run_smoke", arm, output,
                                       {"authorization": child_auth["path"], "authorization_sha": child_auth["sha256"],
                                        "peer_ready": str(output / ("route" if arm == "flat" else "flat") / "ready.json"),
                                        "start_marker": str(output / "START"), "capacity_receipt": str(capacity_receipt),
                                        "capacity_checkpoint": str(capacity_checkpoint)}, handles)
            _start_and_wait(children, {a: output / a / "ready.json" for a in ARMS}, output / "START", started, limit, validate_ready)
            arms, forecast = _smoke_results(output)
            post = collect_smoke_authority(output, capacity_receipt, capacity_checkpoint, fresh["arms"])
            if any(arms[a]["result"]["authority"] != fresh["arms"][a]["bindings"] for a in ARMS):
                raise RuntimeError("smoke arm results do not match reviewed candidate")
            result = {"schema": "h1_queryage_formal_prefix_pair_smoke_v1", "status": "PASS_PAIRED_SOURCE_ONLY_RESOURCE_SMOKE",
                      "arms": arms, "conservative_formal_seconds": forecast}
        else:
            for name in ("barrier", "workers", "checkpoints", "exports"):
                (output / name).mkdir()
            for arm in ARMS:
                children[arm] = _spawn("formal_prefix_train", "worker_run", arm, output,
                                       {"start_marker": str(output / "barrier/START")}, handles)
            ready = _start_and_wait(children, {a: output / "barrier" / f"{a}.ready.json" for a in ARMS},
                                    output / "barrier/START", started, limit, validate_ready)
            freeze = freeze_selection(output, ready); freeze_sha = sha(output / "selection_freeze.json")
            for arm in ARMS:
                children[arm] = _spawn("formal_prefix_train", "finalizer_run", arm, output, {}, handles)
            while any(p.poll() is None for p in children.values()):
                check_children(children, started, limit); time.sleep(.2)
            check_children(children, started, limit)
            finals = {a: read(output / "workers" / f"{a}_final.json") for a in ARMS}
            for arm, final in finals.items():
                if final.get("status") != "COMPLETE_POST_FREEZE" or final.get("arm") != arm or set(final.get("reports", {})) != {"selected", "epoch12"}:
                    raise RuntimeError("complete paired finalizer receipts required")
                for label, report in final["reports"].items():
                    if (report.get("epoch") != freeze[label][arm]["epoch"] or report.get("complete", {}).get("n_bins") != 20325
                            or report.get("checkpoint_sha256") != freeze[label][arm]["checkpoint_sha256"]
                            or sha(report["complete_archive"]) != report.get("complete_archive_sha256")
                            or report.get("plain_ema_path") != str(output / "exports" / f"{arm}_{label}_plain_ema.pt")
                            or sha(report["plain_ema_path"]) != report.get("plain_ema_sha256")):
                        raise RuntimeError("post-freeze complete result/export drift")
            if sha(output / "selection_freeze.json") != freeze_sha:
                raise RuntimeError("selection freeze mutated")
            post = collect_formal_authority(output, capacity_receipt, capacity_checkpoint, smoke_receipt)
            result = {"schema": "h1_queryage_formal_prefix_complete_pair_v1", "status": "COMPLETE_FIXED_FORMAL_NO_PROMOTION",
                      "selection_freeze": freeze, "selection_freeze_sha256": freeze_sha, "finals": finals}
        if post != fresh or sha(output / "input_authority.json") != input_sha:
            raise RuntimeError("fresh root pre/post or owned input authority drift")
        check_external(authorization, authorization_sha, fresh)
        check_children(children, started, limit)
        for handle in handles:
            handle.close()
        result.update({"authority": fresh, "authority_post": post, "authorization_path": str(authorization),
                       "authorization_sha256": authorization_sha, "input_authority_sha256": input_sha,
                       "elapsed_seconds": time.monotonic() - started, "owned_artifact_sha256": manifest(output)})
        atomic(output / "receipt.json", result)
        return result
    except BaseException as exc:
        stop_children(children)
        atomic(output / "FAILED_OR_INCOMPLETE.json", {"status": "FAILED_OR_INCOMPLETE_NO_RESTART", "reason": repr(exc),
                                                       "children": {a: p.returncode for a, p in children.items()}})
        raise
    finally:
        for handle in handles:
            handle.close()


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=("smoke", "formal"), required=True)
    for name in ("output", "authorization", "capacity-receipt", "capacity-checkpoint"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--smoke-receipt", type=Path); p.add_argument("--authorization-sha", required=True)
    a = p.parse_args(argv)
    return run(a.mode, a.output, a.authorization, a.authorization_sha, a.capacity_receipt, a.capacity_checkpoint, a.smoke_receipt)


if __name__ == "__main__":
    main()
