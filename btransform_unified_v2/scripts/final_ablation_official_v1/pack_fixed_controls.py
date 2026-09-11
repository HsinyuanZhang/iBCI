#!/usr/bin/env python3
"""Build, host-check, container-check, and describe one completed fixed control.

This local helper never imports the submit helper, pushes an image, registers a
submission, or calls an API.  It is intentionally fail-closed: formal queue
stages must already be completed, all packer contracts remain authoritative,
and every output owned by this helper must be fresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path("/home/xinyuan/Work_host/SPINT")
QUEUE = ROOT / "btransform_unified_v2/results/final_ablation_official_v1/root_formal_queue.json"
PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
HELPER = Path(__file__).resolve().parent
PROFILES: dict[str, dict[str, str]] = {
    "m2_activity_only": {"task":"m2", "arm":"ACTIVITY_ONLY", "image_stem":"m2-rift-activity-only-r50", "pack":"tfpd_exploration/submissions/evalai_m2_rift_r50_ablation_v1/pack_and_verify.py", "runtime":"m2_rift_falcon_decoder.py", "payload":"artifacts/m2_rift_r50_ablation.pkl", "build":"artifacts/local_pack_receipt.json", "host":"artifacts/host_verify.json", "fixture":"artifacts/smoke_window.npz", "method_name":"M2 RIFT R50 ACTIVITY_ONLY", "budget":"Formal 24-epoch M2 ACTIVITY_ONLY RIFT-R50 concat; frozen M33 activity encoder is rematerialized with zero side and direct T=0; public local EXT6 HO-label selection chooses EMA, never an official metric; IsHeldOutZeroShot=false."},
    "m2_none": {"task":"m2", "arm":"NONE", "image_stem":"m2-rift-none-r50", "pack":"tfpd_exploration/submissions/evalai_m2_rift_r50_ablation_v1/pack_and_verify.py", "runtime":"m2_rift_falcon_decoder.py", "payload":"artifacts/m2_rift_r50_ablation.pkl", "build":"artifacts/local_pack_receipt.json", "host":"artifacts/host_verify.json", "fixture":"artifacts/smoke_window.npz", "method_name":"M2 RIFT R50 NONE", "budget":"Formal 24-epoch M2 NONE RIFT-R50 concat; E0/T are literal zero and no identity material is read; public local EXT6 HO-label selection chooses EMA, never an official metric; IsHeldOutZeroShot=false."},
    "h1_activity_only": {"task":"h1", "arm":"ACTIVITY_ONLY", "image_stem":"h1-rift-activity-only-r300", "pack":"tfpd_exploration/submissions/evalai_h1_rift_r300_ablation_v1/pack_and_verify.py", "runtime":"h1_rift_falcon_decoder.py", "payload":"payload.pkl", "build":"receipt.json", "host":"host_verify.json", "fixture":"smoke_window.npz", "method_name":"H1 RIFT R300 ACTIVITY_ONLY", "budget":"Formal 32-epoch H1 ACTIVITY_ONLY RIFT-R300; C2 is rematerialized with zero side and T=0, using 7/5/4/3 train budgets and M3 official identity; public local HO-label development selection is not official; IsHeldOutZeroShot=false."},
    "h1_none": {"task":"h1", "arm":"NONE", "image_stem":"h1-rift-none-r300", "pack":"tfpd_exploration/submissions/evalai_h1_rift_r300_ablation_v1/pack_and_verify.py", "runtime":"h1_rift_falcon_decoder.py", "payload":"payload.pkl", "build":"receipt.json", "host":"host_verify.json", "fixture":"smoke_window.npz", "method_name":"H1 RIFT R300 NONE", "budget":"Formal 32-epoch H1 NONE RIFT-R300; E0/T are literal zero; public local HO-label development selection is not official; IsHeldOutZeroShot=false."},
    "m1_none": {"task":"m1", "arm":"NONE", "image_stem":"m1-rift-none-r100", "pack":"tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/pack_and_verify.py", "runtime":"m1_rift_none_falcon_decoder.py", "payload":"artifacts/m1_rift_none_r100_selected.pkl", "build":"artifacts/build_receipt.json", "host":"artifacts/host_verify.json", "fixture":"artifacts/smoke_window.npz", "method_name":"M1 RIFT R100 NONE", "budget":"Formal 24-epoch M1 NONE RIFT-R100; E0/T are literal zero and no calibration is used for identity, while all-24 local selection reads labeled public HO3 calibration; it is not an official metric; IsHeldOutZeroShot=false."},
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise RuntimeError(f"object JSON required: {path}")
    return value

def need_file(path: Path) -> Path:
    if not path.is_file(): raise FileNotFoundError(path)
    return path

def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists(): raise FileExistsError(f"refusing to overwrite helper artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)

def env() -> dict[str, str]:
    value = dict(os.environ)
    value.update({"PYTHONNOUSERSITE":"1", "CUDA_VISIBLE_DEVICES":"", "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2", "OPENBLAS_NUM_THREADS":"2", "NUMEXPR_NUM_THREADS":"2"})
    return value

def run(argv: list[str], *, cwd: Path, log: Path) -> subprocess.CompletedProcess[str]:
    if log.exists(): raise FileExistsError(f"refusing to overwrite log: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    # Open exclusively before launching so ROOT can monitor real-time output;
    # never buffer a long pack/host/Docker transcript only in process memory.
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(argv, cwd=cwd, env=env(), text=True, stdout=handle, stderr=subprocess.STDOUT)
        returncode = process.wait()
    stdout = log.read_text(encoding="utf-8")
    result = subprocess.CompletedProcess(argv, returncode, stdout=stdout)
    if result.returncode != 0: raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return result

def queue_jobs(profile: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue = read_json(need_file(QUEUE))
    rows = [x for x in queue.get("jobs", []) if isinstance(x, dict) and x.get("name") == profile]
    expected = ["train", "score"] if profile.startswith(("m1_", "m2_")) else ["train_and_score"]
    if [x.get("stage") for x in rows] != expected: raise RuntimeError(f"{profile}: queue stages drift: {rows}")
    for row in rows:
        if row.get("status") != "COMPLETED" or row.get("returncode") != 0: raise RuntimeError(f"{profile}/{row.get('stage')}: formal stage is not COMPLETED returncode=0")
    return queue, rows

def option(argv: list[Any], name: str) -> str:
    try: return str(argv[argv.index(name) + 1])
    except (ValueError, IndexError) as exc: raise RuntimeError(f"queue argv lacks {name}") from exc

def base_image(dockerfile: Path) -> str:
    match = re.search(r"^ARG\s+BASE_IMAGE=([^\s]+)$", dockerfile.read_text(encoding="utf-8"), flags=re.M)
    if not match: raise RuntimeError(f"Dockerfile lacks fixed BASE_IMAGE: {dockerfile}")
    return match.group(1)

def output_paths(dest: Path) -> tuple[Path, Path, Path]:
    # M2/H1 packers require a non-existent destination at build entry.  These
    # success artifacts are intentionally addressed inside it but not created
    # until after build/host/docker-context have made the package.
    root = dest / "artifacts"
    return root / "root_container_verify.json", root / "fixed_control_manifest.json", root / "pack_fixed_controls_failure.json"

def live_paths(dest: Path) -> tuple[Path, Path]:
    """External live logs/failure record that cannot pre-create fresh --dest."""
    root = dest.parent / f".{dest.name}.pack_fixed_controls"
    return root / "logs", root / "failure.json"

def pack(profile: str, dest: Path, rows: list[dict[str, Any]], logs: Path) -> None:
    spec = PROFILES[profile]; packer = ROOT / spec["pack"]
    train = rows[0]; run_dir = Path(str(train["run_dir"])); argv = list(train.get("argv", []))
    if profile.startswith("m2_"):
        bank = option(argv, "--bank-cache-root"); score = rows[1]
        selection = str(score["dest"])
        build = [str(PYTHON), str(packer), "--stage", "build", "--arm", spec["arm"], "--bank-cache-root", bank, "--run-dir", str(run_dir), "--selection-dir", selection, "--dest", str(dest)]
        host = [str(PYTHON), str(packer), "--stage", "host", "--arm", spec["arm"], "--bank-cache-root", bank, "--run-dir", str(run_dir), "--selection-dir", selection, "--dest", str(dest), "--public-cache-root", bank]
    elif profile.startswith("h1_"):
        bank = option(argv, "--banks")
        build = [str(PYTHON), str(packer), "--stage", "build", "--arm", spec["arm"], "--banks", bank, "--run-dir", str(run_dir), "--dest", str(dest)]
        host = [str(PYTHON), str(packer), "--stage", "host", "--arm", spec["arm"], "--banks", bank, "--run-dir", str(run_dir), "--dest", str(dest)]
    else:
        fixed = ROOT / "tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1"
        if dest.resolve() != fixed.resolve(): raise RuntimeError("m1_none --dest must be its packer's fixed package directory")
        build = [str(PYTHON), str(packer), "--stage", "build", "--run-dir", str(run_dir)]
        host = [str(PYTHON), str(packer), "--stage", "host", "--run-dir", str(run_dir)]
    run(build, cwd=ROOT, log=logs / "01_build.log")
    run(host, cwd=ROOT, log=logs / "02_host.log")
    if profile.startswith("h1_"):
        docker_context = [str(PYTHON), str(packer), "--stage", "docker-context", "--arm", spec["arm"], "--banks", bank, "--run-dir", str(run_dir), "--dest", str(dest)]
        run(docker_context, cwd=ROOT, log=logs / "03_docker_context.log")

def selected_epoch(build: Mapping[str, Any]) -> int:
    for key in ("selected_epoch", "selection_epoch"):
        if key in build: return int(build[key])
    selection = build.get("selection")
    if isinstance(selection, Mapping) and "epoch" in selection: return int(selection["epoch"])
    raise RuntimeError("build receipt has no selected epoch")

def expected_labels(spec: Mapping[str, str], payload_sha: str) -> dict[str, str]:
    """Static labels from the three checked-in Dockerfiles, before PASS."""
    task = spec["task"]
    if task == "m1":
        return {"ai.eval.task":"m1", "ai.eval.architecture":"RIFT", "ai.eval.backend":"cached", "ai.eval.identity":"literal_zero_e0_t_none_concat", "ai.eval.arm":"NONE", "ai.eval.payload.sha256":payload_sha}
    if task == "m2":
        return {"ai.eval.task":"m2", "ai.eval.architecture":"RIFT-R50-concat", "ai.eval.backend":"cached-pytorch-cpu", "ai.eval.identity":"static-ablation", "ai.eval.arm":spec["arm"], "ai.eval.payload.sha256":payload_sha}
    return {"ai.eval.task":"h1", "ai.eval.architecture":"RIFT", "ai.eval.backend":"cached", "ai.eval.identity":spec["arm"], "ai.eval.ablation.arm":spec["arm"], "ai.eval.payload.sha256":payload_sha}

def host_package_files(dest: Path, spec: Mapping[str, str]) -> dict[str, str]:
    pkg = dest / "artifacts/pkg"
    if not pkg.is_dir(): raise FileNotFoundError(pkg)
    payload = need_file(dest / spec["payload"]); runtime = need_file(dest / spec["runtime"]); decode = need_file(dest / "decode.py")
    files = {"/data/decoder.pkl": payload, f"/{spec['runtime']}": runtime, "/decode.py": decode}
    for path in sorted(pkg.rglob("*.py")):
        files["/pkg/" + str(path.relative_to(pkg))] = path
    return {name: sha(path) for name, path in files.items()}

def docker_build_and_verify(profile: str, dest: Path, build: Mapping[str, Any], logs: Path, receipt_path: Path) -> tuple[dict[str, Any], Path]:
    spec = PROFILES[profile]; payload = need_file(dest / spec["payload"]); payload_sha = sha(payload); epoch = selected_epoch(build)
    tag = f"{spec['image_stem']}-e{epoch}:v1"
    probe = subprocess.run(["docker", "image", "inspect", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if probe.returncode == 0: raise RuntimeError(f"refusing to overwrite existing image tag: {tag}")
    dockerfile = need_file(dest / "Dockerfile"); base = base_image(dockerfile)
    base_info = run(["docker", "image", "inspect", base, "--format", "{{.Id}}"], cwd=ROOT, log=logs / "04_base_image_inspect.log")
    base_id = base_info.stdout.strip()
    run(["docker", "build", "--build-arg", f"BASE_IMAGE={base}", "--build-arg", f"PAYLOAD_SHA256={payload_sha}", "--build-arg", f"ARM={spec['arm']}", "-t", tag, str(dest)], cwd=ROOT, log=logs / "05_docker_build.log")
    inspect = run(["docker", "image", "inspect", tag], cwd=ROOT, log=logs / "08_image_inspect.log")
    info = json.loads(inspect.stdout)[0]; image_id = str(info["Id"]); labels = dict((info.get("Config", {}).get("Labels", {}) or {}))
    for key, value in expected_labels(spec, payload_sha).items():
        if labels.get(key) != value: raise RuntimeError(f"Docker label drift {key}: {labels.get(key)!r} != {value!r}")
    fixture = need_file(dest / spec["fixture"]); host_hashes = host_package_files(dest, spec)
    smoke = ["docker", "run", "--rm", "--network", "none", "--mount", f"type=bind,src={fixture.resolve()},dst=/fixture/smoke_window.npz,readonly", "--entrypoint", "python", tag, f"/{spec['runtime']}", "--smoke-payload", "/data/decoder.pkl", "--smoke-window", "/fixture/smoke_window.npz"]
    smoke_result = run(smoke, cwd=ROOT, log=logs / "06_container_smoke.log")
    code = "import hashlib,json,pathlib; root=pathlib.Path('/pkg'); paths=['/data/decoder.pkl','/decode.py','/" + spec['runtime'] + "']+[str(p) for p in root.rglob('*.py')]; print(json.dumps({p:hashlib.sha256(open(p,'rb').read()).hexdigest() for p in sorted(paths)},sort_keys=True))"
    hashes = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", tag, "-c", code], cwd=ROOT, log=logs / "07_container_hashes.log")
    container_hashes = json.loads(hashes.stdout)
    if container_hashes != host_hashes: raise RuntimeError("container runtime/decode/pkg/payload SHA map differs from host")
    body = {"schema":"root_final_ablation_container_verify_v1", "status":"PASSED", "profile":profile, "arm":spec["arm"], "image_tag":tag, "image_id":image_id, "base_image":base, "base_image_id":base_id, "payload":str(payload), "payload_sha256":payload_sha, "selected_epoch":epoch, "bytes_exact":True, "smoke_exit_code":smoke_result.returncode, "smoke_fixture":str(fixture), "smoke_fixture_sha256":sha(fixture), "file_hashes":host_hashes, "logs":{name:str(logs/name) for name in ("04_base_image_inspect.log","05_docker_build.log","06_container_smoke.log","07_container_hashes.log","08_image_inspect.log")}, "labels":labels, "external_actions":False}
    write_new(receipt_path, body)
    return body, payload

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--dest", type=Path, required=True, help="fresh M2/H1 package destination; fixed M1 package directory for m1_none")
    args = parser.parse_args(); profile = args.profile; dest = args.dest.resolve(); spec = PROFILES[profile]
    queue, rows = queue_jobs(profile)
    if profile != "m1_none" and dest.exists(): raise FileExistsError(f"fresh --dest required: {dest}")
    container_receipt, manifest_path, stale_failure_path = output_paths(dest)
    logs, failure_path = live_paths(dest)
    if any(path.exists() for path in (container_receipt, manifest_path, stale_failure_path, failure_path)): raise FileExistsError("helper receipt/manifest/failure target already exists")
    if logs.exists(): raise FileExistsError(f"helper logs already exist: {logs}")
    try:
        pack(profile, dest, rows, logs)
        build = read_json(need_file(dest / spec["build"])); host = read_json(need_file(dest / spec["host"]))
        if build.get("status") != "BUILT_NOT_HOST_VERIFIED": raise RuntimeError("packer build receipt is not formal unverified build")
        required_host = "HOST_VERIFIED" if profile == "m1_none" else ("HOST_REAL_PUBLIC_PARITY_PASS" if profile.startswith("m2_") else "HOST_PACK_VERIFY_PASS")
        if host.get("status") != required_host: raise RuntimeError("packer host receipt status drift")
        payload_before_docker = need_file(dest / spec["payload"])
        if host.get("payload_sha256") != sha(payload_before_docker): raise RuntimeError("formal host receipt does not bind packaged payload")
        container, payload = docker_build_and_verify(profile, dest, build, logs, container_receipt)
        manifest = {"schema":"root_final_ablation_fixed_control_manifest_v1", "status":"LOCAL_PACKAGE_READY_NOT_SUBMITTED", "profile":profile, "arm":spec["arm"], "formal":True, "formal_queue":str(QUEUE), "payload_path":str(payload), "payload_sha256":sha(payload), "build_receipt_path":str(dest/spec["build"]), "build_receipt_sha256":sha(dest/spec["build"]), "host_receipt_path":str(dest/spec["host"]), "host_receipt_sha256":sha(dest/spec["host"]), "container_receipt_path":str(container_receipt), "container_receipt_sha256":sha(container_receipt), "image_tag":container["image_tag"], "image_id":container["image_id"], "method_label":container["labels"].get("ai.eval.method"), "method_name":spec["method_name"], "method_description":f"{container['labels'].get('ai.eval.method')} | formal {profile} selected epoch {container['selected_epoch']}; local build/host/container verified.", "budget_disclosure":spec["budget"], "state_path":str(dest/"artifacts/root_fixed_control_submit_state.json"), "official_submission_id":None, "official_result":None, "external_actions":False}
        if not isinstance(manifest["method_label"],str) or not manifest["method_label"]: raise RuntimeError("Docker image lacks actual ai.eval.method label")
        write_new(manifest_path, manifest); print(json.dumps(manifest, indent=2, sort_keys=True)); return 0
    except Exception as exc:
        write_new(failure_path, {"schema":"root_final_ablation_fixed_control_pack_failure_v1", "status":"FAILED", "profile":profile, "dest":str(dest), "error":str(exc), "logs":str(logs), "external_actions":False})
        raise

if __name__ == "__main__": raise SystemExit(main())
