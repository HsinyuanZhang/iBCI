#!/usr/bin/env python3
"""Build the M1 RIFT R100 proj_add P16 learned_slope default-ladder e3 payload.

Host-verify against CpuLearnableRecencyRuntime, then docker-pack and container smoke test.
Does not EvalAI-submit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/home/xinyuan/Work_host/SPINT")
DEST = Path(__file__).resolve().parent
RUN = ROOT / "btransform_unified_v2/learnable_recency_v1/results/m1_projadd_learned_slope_default_s42"
CKPT = RUN / "epoch_003.pt"
CKPT_SHA = "b76a48d98820b0540ca738f7c8cebbe82acc9d3ff6a1fa304ff67c8608c88a37"
SCORE = RUN / "score_receipt.json"
BANK_PAYLOAD = ROOT / "tfpd_exploration/submissions/evalai_m1_projadd_depth2_ort_v1/artifacts/m1_projadd_p16_d2_s42_ema_e21.pkl"
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
LEARNABLE_SRC = ROOT / "btransform_unified_v2/learnable_recency_v1/src/learnable_recency_v1"
IMAGE_TAG = "m1-rift-r100-projadd-learnable-default-e3:v1"
METHOD_LABEL = (
    "M1 RIFT R100 proj_add P16 learned_slope default-ladder e3 EMA cached CPU; "
    "architecture=RIFT backend=cached identity=proj_add; HO-M10 pick 0.707514; "
    "not BT-EORT; not ORT; not SPINT"
)
METHOD_NAME = "M1 RIFT R100 proj_add learned default e3"
METHOD_DESCRIPTION = (
    "M1 RIFT (Recency-biased Incremental Finite-context Transformer) R100, D4/width256, "
    "proj_add P16 identity, per-layer learned recency slopes on the default half-life ladder, "
    "cached CPU KV. Same official 7-tag banks. HO-M10 pick e3 mean 0.707514. "
    "Not BT-EORT, not ORT, not SPINT."
)
N_STEPS = 80
GATE = 1.0e-5
SKIP_PKG = {"joint_m2_model.py", "m2_mechanism_model.py", "joint_m1_model.py"}
SAFE_INIT = '''"""Container-safe RIFT decoder package. Joint/mechanism modules are omitted."""

from .config import RiftTemporalConfig
from .model import RiftDecoder
from .streaming import RiftStreamDecoder
from .temporal import RiftTemporal, RiftTemporalState
from .cpu_temporal import CpuRiftTemporalRuntime

__all__ = [
    "CpuRiftTemporalRuntime",
    "RiftDecoder",
    "RiftStreamDecoder",
    "RiftTemporal",
    "RiftTemporalConfig",
    "RiftTemporalState",
]
'''
LEARNABLE_SAFE_INIT = '''"""Container-safe learnable recency package. wrap.py / JointM1 are omitted."""

from .config import LearnableRecencyConfig, config_from_run_meta
from .cpu_temporal import CpuLearnableRecencyRuntime
from .temporal import LearnableRecencyTemporal

__all__ = [
    "CpuLearnableRecencyRuntime",
    "LearnableRecencyConfig",
    "LearnableRecencyTemporal",
    "config_from_run_meta",
]
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _copy_pkg() -> Path:
    pkg = DEST / "artifacts" / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    for src, name in ((V1_SRC / "btransform_unified_v1", "btransform_unified_v1"), (V2_SRC / "btransform_unified_v2", "btransform_unified_v2")):
        dst = pkg / name
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*.py"):
            if item.name in SKIP_PKG:
                continue
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    (pkg / "btransform_unified_v2" / "__init__.py").write_text(SAFE_INIT)
    lr = pkg / "learnable_recency_v1"
    lr.mkdir(parents=True, exist_ok=True)
    for name in ("config.py", "temporal.py", "cpu_temporal.py"):
        shutil.copy2(LEARNABLE_SRC / name, lr / name)
    (lr / "__init__.py").write_text(LEARNABLE_SAFE_INIT)
    return pkg


def _use_packed_pkg(pkg: Path) -> None:
    for path in (str(ROOT), str(pkg), str(DEST)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _build_payload() -> tuple[Path, str, dict]:
    from btransform_unified_v1.ema import DecoderEMA
    from learnable_recency_v1.config import config_from_run_meta
    from m1_rift_falcon_decoder import CPUUnpickler, PAYLOAD_SCHEMA, build_decoder

    if sha256(CKPT) != CKPT_SHA:
        raise RuntimeError("epoch_003.pt SHA drift")
    with open(BANK_PAYLOAD, "rb") as handle:
        bt = CPUUnpickler(handle).load()
    banks = bt["bank_by_dataset_tag"]
    if len(banks) != 7:
        raise RuntimeError(f"expected 7 official tags, got {len(banks)}")
    for tag, row in banks.items():
        row["e0_sha256"] = array_sha(row["E0"])
        row["t_sha256"] = array_sha(row["T"])

    score = json.loads(SCORE.read_text())
    selected = score["selection"]
    if int(selected["epoch"]) != 3:
        raise RuntimeError(f"score receipt selection drifted: {selected}")
    e3_score = score["ema_by_epoch"]["3"]
    if abs(float(e3_score["equal_session_mean"]) - 0.7075138586588596) > 1.0e-12:
        raise RuntimeError(f"e3 mean drifted: {e3_score}")
    meta = json.loads((RUN / "run_meta.json").read_text())
    recency_cfg = config_from_run_meta(meta, "m1")
    if recency_cfg.ladder != "default" or recency_cfg.tier != "learned_slope" or not recency_cfg.per_layer:
        raise RuntimeError(f"run_meta learnable_config drifted: {recency_cfg}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != "m1_projadd_learnable_epoch_checkpoint_v1":
        raise RuntimeError("checkpoint schema is not M1 proj_add learnable")
    if meta.get("identity_interface") != "proj_add" or int(meta.get("proj_dim", -1)) != 16:
        raise RuntimeError("meta is not proj_add P16")
    if int(ckpt.get("epoch", -1)) != 3 or ckpt.get("tier") != "learned_slope":
        raise RuntimeError("checkpoint is not learned_slope e3")
    model = build_decoder(recency_cfg)
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    ema = DecoderEMA(model, decay=0.9995)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(model)
    model.eval()
    ema_state = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()}
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "task": "m1",
        "context_bins": 100,
        "bias_mode": "learnable_learned_slope",
        "identity_interface": "proj_add",
        "proj_dim": 16,
        "tier": "learned_slope",
        "ladder": "default",
        "per_layer": True,
        "behavior_scaling_factor": 1.0,
        "checkpoint": str(CKPT),
        "checkpoint_sha256": CKPT_SHA,
        "bank_source_payload_sha256": sha256(BANK_PAYLOAD),
        "learnable_config": dict(meta["learnable_config"]),
        "bank_by_dataset_tag": banks,
        "ema_state_dict": ema_state,
        "selection": {
            "surface": "visible HO3 M10",
            "epoch": 3,
            "equal_session_mean": float(e3_score["equal_session_mean"]),
            "n_windows": 3881,
            "official_test_used": False,
        },
    }
    dest = DEST / "artifacts" / "m1_rift_r100_projadd_learnable_default_e3.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return dest, sha256(dest), payload["selection"]


def _load_neural_traces() -> dict[str, np.ndarray]:
    for p in (ROOT, ROOT / "streaming_calibration_exp", V1_SRC, ROOT / "src", ROOT / "btransform_unified_v1/scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    script_path = ROOT / "btransform_unified_v2/scripts/rift_v1/m1_train.py"
    spec = importlib.util.spec_from_file_location("m1_rift_train", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dataset, _ = mod.legacy.build_fullsession_face()
    sources = {s.removeprefix("ses-"): np.ascontiguousarray(dataset.neural_data[s][99:99 + N_STEPS], dtype=np.float32) for s in mod.SOURCE_SESSIONS}
    ho_mat = mod._ho_material()
    heldouts = {s: np.ascontiguousarray(ho_mat[s]["dataset"].neural_data[s][99:99 + N_STEPS], dtype=np.float32) for s in mod.HO}
    return {**sources, **heldouts}


def _host_verify(payload_path: Path) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask
    from m1_rift_falcon_decoder import (
        CpuLearnableRiftRuntime,
        M1RiftCachedFalconDecoder,
        _task_bank,
        build_decoder,
        load_payload,
        recency_config_from_payload,
    )

    payload = load_payload(payload_path)
    config = FalconConfig(task=FalconTask.m1)
    report = {}
    traces = _load_neural_traces()
    plans = (
        ("B1", ["20121004"]),
        ("B7", ["20120924", "20120926", "20120927", "20120928", "20121004", "20121017", "20121024"]),
    )
    recency_cfg = recency_config_from_payload(payload)
    for name, tags in plans:
        missing = [tag for tag in tags if tag not in payload["bank_by_dataset_tag"]]
        if missing:
            raise RuntimeError(f"tags missing from payload: {missing}")
        streams = [traces[tag] for tag in tags]
        stems = [f"sub-MonkeyL-held-out-calib_ses-{tag}_behavior+ecephys" if tag in {"20121004", "20121017", "20121024"} else f"sub-MonkeyL-ses-{tag}_behavior+ecephys" for tag in tags]
        packed = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=len(tags))
        packed.reset(dataset_tags=stems)
        model = build_decoder(recency_cfg)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in payload["ema_state_dict"].items()}, strict=True)
        model.eval()
        banks = [_task_bank(tag, payload["bank_by_dataset_tag"][tag]) for tag in tags]
        ref = CpuLearnableRiftRuntime(model, banks, [str(tag) for tag in tags])
        max_abs = 0.0
        last = None
        for step in range(N_STEPS):
            batch_x = np.stack([row[step] for row in streams], axis=0)
            got = packed.predict(batch_x)
            want = ref.advance(torch.from_numpy(batch_x)).numpy() / 1.0
            last = got
            max_abs = max(max_abs, float(np.max(np.abs(got - want))))
            if max_abs > GATE:
                raise RuntimeError(f"{name} host gate fail t={step} max_abs={max_abs}")
        report[name] = {
            "steps": N_STEPS,
            "max_abs": max_abs,
            "last_shape": list(last.shape),
            "sessions": tags,
        }

    smoke = DEST / "artifacts" / "smoke_window.npz"
    tag = "20121004"
    stem = f"sub-MonkeyL-held-out-calib_ses-{tag}_behavior+ecephys"
    neural = np.ascontiguousarray(traces[tag][:40], dtype=np.float32)
    packed = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    packed.reset(dataset_tags=[stem])
    pred = None
    for row in neural:
        pred = packed.predict(row.reshape(1, -1))
    np.savez(smoke, tag_stem=np.asarray(stem), window=neural, expected=pred)
    report["smoke_window"] = str(smoke)
    report["status"] = "HOST_PACK_VERIFY_PASS"
    return report


def _docker_build(payload_sha: str) -> dict:
    cmd = [
        "docker",
        "build",
        "--build-arg",
        f"PAYLOAD_SHA256={payload_sha}",
        "--build-arg",
        f"METHOD_LABEL={METHOD_LABEL}",
        "-t",
        IMAGE_TAG,
        str(DEST),
    ]
    subprocess.run(cmd, check=True)
    inspect = subprocess.check_output(["docker", "image", "inspect", IMAGE_TAG, "--format", "{{.Id}} {{.Size}}"], text=True).strip()
    image_id, size = inspect.split()
    return {"image_tag": IMAGE_TAG, "image_id": image_id, "image_size": int(size)}


def _container_smoke(smoke_path: Path) -> dict:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{smoke_path.resolve()}:/tmp/smoke.npz:ro",
        IMAGE_TAG,
        "python",
        "/m1_rift_falcon_decoder.py",
        "--smoke-payload",
        "/data/decoder.pkl",
        "--smoke-window",
        "/tmp/smoke.npz",
    ]
    raw = subprocess.check_output(cmd, text=True).strip()
    last = raw.splitlines()[-1]
    parsed = json.loads(last)
    if parsed.get("status") != "CONTAINER_SMOKE_PASS":
        raise RuntimeError(f"container smoke unexpected: {raw}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()
    print("[pack] 1/5 copying package modules...", flush=True)
    pkg = _copy_pkg()
    _use_packed_pkg(pkg)

    print("[pack] 2/5 building payload pkl...", flush=True)
    payload_path, payload_sha, selection = _build_payload()
    print(f"[pack] payload {payload_path.name} sha256={payload_sha}", flush=True)

    print("[pack] 3/5 running host-verify (B1 & B7)...", flush=True)
    host_report = _host_verify(payload_path)
    (DEST / "artifacts" / "host_verify.json").write_text(json.dumps(host_report, indent=2, sort_keys=True) + "\n")
    print(f"[pack] host verify PASS: {host_report}", flush=True)

    if args.skip_docker:
        print("[pack] --skip-docker set, exiting after host verify", flush=True)
        return

    print("[pack] 4/5 building Docker image...", flush=True)
    build_info = _docker_build(payload_sha)
    print(f"[pack] docker build ok: {build_info}", flush=True)

    print("[pack] 5/5 running container smoke test...", flush=True)
    smoke_info = _container_smoke(Path(host_report["smoke_window"]))
    print(f"[pack] container smoke PASS: {smoke_info}", flush=True)

    candidate = {
        "schema_version": "m1_rift_evalai_candidate_v1",
        "arm": "m1_rift_r100_projadd_learnable_default_e3_cached",
        "image_tag": IMAGE_TAG,
        "image_id": build_info["image_id"],
        "image_size": build_info["image_size"],
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "method_name": METHOD_NAME,
        "method_description": METHOD_DESCRIPTION,
        "method_label": METHOD_LABEL,
        "selection": selection,
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "receipt_path": str(DEST / "artifacts" / "evalai_submit_receipt.json"),
    }
    (DEST / "artifacts" / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print(f"[pack] candidate manifest written to artifacts/evalai_candidate.json", flush=True)


if __name__ == "__main__":
    main()
