#!/usr/bin/env python3
"""Build the H1 RIFT R300 cached payload, host-verify, and docker-pack.

Does not EvalAI-submit. Does not overwrite 582044 artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/xinyuan/Work_host/SPINT")
DEST = Path(__file__).resolve().parent
CKPT = ROOT / "btransform_unified_v2/results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/formal/epoch_022.pt"
BT_PAYLOAD = ROOT / "tfpd_exploration/submissions/evalai_h1_c2_cal1_b2_ort_v1/artifacts/h1_c2_cal1_b2_s42_ema_e18_L200.pkl"
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
PY = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
IMAGE_TAG = "h1-rift-r300-cached-e22:v1"
METHOD_LABEL = (
    "H1 RIFT R300 recency e22 EMA cached CPU; architecture=RIFT backend=cached; not BT-EORT; not SPINT"
)
METHOD_NAME = "H1 RIFT R300 recency e22 cached"
N_STEPS = 80
GATE = 1.0e-5

for path in (str(V2_SRC), str(V1_SRC), str(DEST), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_pkg() -> None:
    pkg = DEST / "artifacts" / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    for src, name in ((V1_SRC / "btransform_unified_v1", "btransform_unified_v1"), (V2_SRC / "btransform_unified_v2", "btransform_unified_v2")):
        dst = pkg / name
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*.py"):
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _build_payload() -> tuple[Path, str, dict]:
    from btransform_unified_v1.ema import DecoderEMA
    from btransform_unified_v2.model import RiftDecoder
    from h1_rift_falcon_decoder import CPUUnpickler, PAYLOAD_SCHEMA

    with open(BT_PAYLOAD, "rb") as handle:
        bt = CPUUnpickler(handle).load()
    banks = bt["bank_by_dataset_tag"]
    if len(banks) != 27:
        raise RuntimeError(f"expected 27 official tags, got {len(banks)}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("variant") != "recency" or int(ckpt.get("context_bins", 0)) != 300:
        raise RuntimeError("checkpoint is not formal H1 R300 recency")
    model = RiftDecoder("h1", context_bins=300, bias_mode="recency", seed=42, proj_dim=16)
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    ema = DecoderEMA(model, decay=0.9995)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(model)
    model.eval()
    ema_state = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()}
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "task": "h1",
        "context_bins": 300,
        "bias_mode": "recency",
        "proj_dim": 16,
        "behavior_scaling_factor": 20.0,
        "checkpoint": str(CKPT),
        "checkpoint_sha256": sha256(CKPT),
        "bank_source_payload_sha256": sha256(BT_PAYLOAD),
        "bank_by_dataset_tag": banks,
        "ema_state_dict": ema_state,
        "selection": {
            "surface": "C2 HO-M3 development",
            "epoch": 22,
            "mean": 0.3748232633647425,
            "worst_session": "S7",
            "worst": 0.20587240655236508,
            "official_test_used": False,
        },
    }
    dest = DEST / "artifacts" / "h1_rift_r300_recency_e22.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return dest, sha256(dest), payload["selection"]


def _host_verify(payload_path: Path) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask
    from btransform_unified_v1 import adapters
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
    from btransform_unified_v1.bank import TaskBank
    from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
    from btransform_unified_v2.model import RiftDecoder
    from h1_rift_falcon_decoder import H1RiftCachedFalconDecoder, _task_bank, load_payload

    payload = load_payload(payload_path)
    cache = adapters._h1_source_cache()
    config = FalconConfig(task=FalconTask.h1)
    report = {}
    for batch, n_steps in ((1, N_STEPS), (8, N_STEPS)):
        sessions = list(HELDIN_SESSIONS[:batch])
        stems = [f"sub-HumanPitt-held-in-minival_{name}" for name in sessions]
        tags = [config.hash_dataset(Path(stem).stem) for stem in stems]
        missing = [tag for tag in tags if tag not in payload["bank_by_dataset_tag"]]
        if missing:
            raise RuntimeError(f"hashed tags missing from payload: {missing}")
        streams = [
            np.ascontiguousarray(cache["minival"][name]["neural"][:n_steps], dtype=np.float32)
            for name in sessions
        ]
        packed = H1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=batch)
        packed.reset(dataset_tags=stems)
        model = RiftDecoder("h1", context_bins=300, bias_mode="recency", seed=42, proj_dim=16)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in payload["ema_state_dict"].items()}, strict=True)
        model.eval()
        banks = [_task_bank(tag, payload["bank_by_dataset_tag"][tag]) for tag in tags]
        ref = CpuRiftRuntime(model, banks, [str(tag) for tag in tags], temporal_backend="cached")
        max_abs = 0.0
        last = None
        for step in range(n_steps):
            batch_x = np.stack([row[step] for row in streams], axis=0)
            got = packed.predict(batch_x)
            want = ref.advance(torch.from_numpy(batch_x)).numpy() / 20.0
            last = got
            max_abs = max(max_abs, float(np.max(np.abs(got - want))))
            if max_abs > GATE:
                raise RuntimeError(f"B{batch} host gate fail t={step} max_abs={max_abs}")
        report[f"B{batch}"] = {
            "steps": n_steps,
            "max_abs": max_abs,
            "last_shape": list(last.shape),
            "sessions": sessions,
            "tags": tags,
        }
    smoke = DEST / "artifacts" / "smoke_window.npz"
    session = HELDIN_SESSIONS[0]
    stem = f"sub-HumanPitt-held-in-minival_{session}"
    neural = np.ascontiguousarray(cache["minival"][session]["neural"][:40], dtype=np.float32)
    packed = H1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
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
        "-t",
        IMAGE_TAG,
        str(DEST),
    ]
    subprocess.run(cmd, check=True)
    inspect = subprocess.check_output(["docker", "image", "inspect", IMAGE_TAG, "--format", "{{.Id}} {{.Size}}"], text=True).strip()
    image_id, size = inspect.split()
    return {"image_tag": IMAGE_TAG, "image_id": image_id, "image_size": int(size)}


def main() -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _copy_pkg()
    payload_path, payload_sha, selection = _build_payload()
    host = _host_verify(payload_path)
    docker = _docker_build(payload_sha)
    candidate = {
        "arm": "h1_rift_r300_recency_e22_cached",
        "budget_disclosure": (
            "H1 C2-CAL-1 B2 official banks (27 tags, deploy M3). RIFT R300 recency "
            "seed42 EMA e22 after 32-epoch 13-session train. Local HO-M3 is development "
            "selection, not official HO. Cached CPU runtime, no TTA, no ORT."
        ),
        "evalai_opened": False,
        "image_id": docker["image_id"],
        "image_size": docker["image_size"],
        "image_tag": docker["image_tag"],
        "method_description": (
            "H1 RIFT (Recency-biased Incremental Finite-context Transformer) R300, "
            "D4/P16/width256, recency half-lives, cached CPU KV. Same C2-CAL-1 B2 banks "
            f"as 582044. HO-M3 pick e22 mean {selection['mean']:.4f} worst {selection['worst']:.4f}. "
            "Not BT-EORT, not SPINT."
        ),
        "method_label": METHOD_LABEL,
        "method_name": METHOD_NAME,
        "payload_sha256": payload_sha,
        "register": False,
        "selection_mean": selection["mean"],
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "window": 300,
        "host_verify": host,
    }
    (DEST / "artifacts" / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PACKED_NOT_REGISTERED", **{k: candidate[k] for k in ("image_tag", "image_id", "payload_sha256", "image_size")}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
