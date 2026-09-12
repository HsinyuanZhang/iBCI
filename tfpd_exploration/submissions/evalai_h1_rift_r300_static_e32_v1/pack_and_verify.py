#!/usr/bin/env python3
"""Build the H1 RIFT R300 static_identity learned_slope e32 payload.

Seals last-epoch EMA (pre-registered, HO labels do not pick). Host-verify and
docker-smoke. Does not EvalAI-submit. Does not overwrite 582196 / 582241 / 582278.
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
RUN = ROOT / "btransform_unified_v2/learnable_recency_v1/results/h1_static_s42"
CKPT = RUN / "epoch_032.pt"
CKPT_SHA = "b79adeda51955c14bb60e4643835a5d8d6862cdb2480134fd9fd42a04a765b51"
SCORE = RUN / "local_ho_static_report.json"
RUN_META = RUN / "run_meta.json"
MASK_PAYLOAD = ROOT / "tfpd_exploration/submissions/evalai_h1_c2_cal1_b2_ort_v1/artifacts/h1_c2_cal1_b2_s42_ema_e18_L200.pkl"
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
LEARNABLE_SRC = ROOT / "btransform_unified_v2/learnable_recency_v1/src/learnable_recency_v1"
IMAGE_TAG = "h1-rift-r300-static-e32:v1"
METHOD_LABEL = (
    "H1 RIFT R300 static_identity learned_slope default-ladder e32 EMA cached CPU; "
    "architecture=RIFT backend=cached identity=static recency=learned_slope_default; "
    "ho-m3 report 0.209015; not BT-EORT; not ORT; not SPINT; not 582196; not 582241"
)
METHOD_NAME = "H1 RIFT R300 static e32"
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
LEARNABLE_SAFE_INIT = '''"""Container-safe learnable recency package."""

from .config import LearnableRecencyConfig, config_from_run_meta
from .cpu_temporal import CpuLearnableRecencyRuntime
from .static_model import StaticLearnableRiftDecoder
from .temporal import LearnableRecencyTemporal
from .wrap import LearnableRiftDecoder

__all__ = [
    "CpuLearnableRecencyRuntime",
    "LearnableRecencyConfig",
    "LearnableRecencyTemporal",
    "LearnableRiftDecoder",
    "StaticLearnableRiftDecoder",
    "config_from_run_meta",
]
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    for name in ("config.py", "temporal.py", "cpu_temporal.py", "static_model.py"):
        shutil.copy2(LEARNABLE_SRC / name, lr / name)
    shutil.copy2(DEST / "wrap_safe.py", lr / "wrap.py")
    (lr / "__init__.py").write_text(LEARNABLE_SAFE_INIT)
    return pkg


def _use_packed_pkg(pkg: Path) -> None:
    for path in (str(pkg), str(DEST)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _build_payload() -> tuple[Path, str, dict]:
    from btransform_unified_v1.ema import DecoderEMA
    from learnable_recency_v1.config import config_from_run_meta
    from h1_rift_falcon_decoder import CPUUnpickler, PAYLOAD_SCHEMA, build_decoder

    if sha256(CKPT) != CKPT_SHA:
        raise RuntimeError("epoch_032.pt SHA drift")
    score = json.loads(SCORE.read_text())
    selected = score["selected"]
    if int(selected["epoch"]) != 32:
        raise RuntimeError(f"static selection drifted: {selected}")
    if abs(float(selected["val_ho_m3_grouped/r2_mean"]) - 0.20901458725628483) > 1.0e-12:
        raise RuntimeError(f"HO-M3 e32 mean drifted: {selected}")
    if score.get("local_ho_labels_used_for_selection") is not False:
        raise RuntimeError("static HO labels were used for selection")
    meta = json.loads(RUN_META.read_text())
    if meta.get("status") != "FORMAL" or meta.get("variant") != "STATIC":
        raise RuntimeError(f"run_meta is not formal STATIC: {meta.get('status')} {meta.get('variant')}")
    recency_cfg = config_from_run_meta(meta, "h1")
    with open(MASK_PAYLOAD, "rb") as handle:
        official = CPUUnpickler(handle).load()["bank_by_dataset_tag"]
    if len(official) != 27:
        raise RuntimeError(f"expected 27 official tags, got {len(official)}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("variant") != "static" or int(ckpt.get("epoch", -1)) != 32:
        raise RuntimeError("checkpoint is not formal static e32")
    model = build_decoder(recency_cfg)
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    ema = DecoderEMA(model, decay=0.9995)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(model)
    model.eval()
    if bool(model.static_carrier.any()):
        raise RuntimeError("static carrier is not literal zero")
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "task": "h1",
        "context_bins": 300,
        "bias_mode": "learnable_learned_slope",
        "identity": "static",
        "identity_interface": "static_identity_table",
        "proj_dim": 16,
        "tier": "learned_slope",
        "ladder": "default",
        "behavior_scaling_factor": 20.0,
        "checkpoint": str(CKPT),
        "checkpoint_sha256": CKPT_SHA,
        "learnable_config": dict(meta["learnable_config"]),
        "official_dataset_tags": sorted(official),
        "ema_state_dict": {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()},
        "selection": {
            "surface": "C2 HO-M3 development report only",
            "epoch": 32,
            "mean": float(selected["val_ho_m3_grouped/r2_mean"]),
            "worst": float(selected["worst_session_r2"]),
            "rule": score["selection_rule"],
            "official_test_used": False,
            "local_ho_labels_used_for_selection": False,
        },
    }
    dest = DEST / "artifacts" / "h1_rift_r300_static_e32.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return dest, sha256(dest), payload["selection"]


def _host_verify(payload_path: Path) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
    from h1_rift_falcon_decoder import (
        CpuStaticRiftRuntime,
        H1RiftLearnableFalconDecoder,
        build_decoder,
        load_payload,
        recency_config_from_payload,
    )

    payload = load_payload(payload_path)
    cache = torch.load(
        ROOT / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/source_cache.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = FalconConfig(task=FalconTask.h1)
    recency_cfg = recency_config_from_payload(payload)
    report = {}
    for batch, n_steps in ((1, N_STEPS), (8, N_STEPS)):
        sessions = list(HELDIN_SESSIONS[:batch])
        stems = [f"sub-HumanPitt-held-in-minival_{name}" for name in sessions]
        tags = [config.hash_dataset(Path(stem).stem) for stem in stems]
        missing = [tag for tag in tags if tag not in payload["official_dataset_tags"]]
        if missing:
            raise RuntimeError(f"hashed tags missing from payload: {missing}")
        streams = [
            np.ascontiguousarray(cache["minival"][name]["neural"][:n_steps], dtype=np.float32)
            for name in sessions
        ]
        packed = H1RiftLearnableFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=batch)
        packed.reset(dataset_tags=stems)
        model = build_decoder(recency_cfg)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in payload["ema_state_dict"].items()}, strict=True)
        model.eval()
        ref = CpuStaticRiftRuntime(model, [str(tag) for tag in tags])
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
    packed = H1RiftLearnableFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    packed.reset(dataset_tags=[stem])
    pred = None
    for row in neural:
        pred = packed.predict(row.reshape(1, -1))
    np.savez(smoke, tag_stem=np.asarray(stem), window=neural, expected=pred)
    report["smoke_window"] = str(smoke)
    report["status"] = "HOST_PACK_VERIFY_PASS"
    return report


def _docker_build(payload_sha: str) -> dict:
    cmd = ["docker", "build", "--build-arg", f"PAYLOAD_SHA256={payload_sha}", "-t", IMAGE_TAG, str(DEST)]
    subprocess.run(cmd, check=True)
    inspect = subprocess.check_output(["docker", "image", "inspect", IMAGE_TAG, "--format", "{{.Id}} {{.Size}}"], text=True).strip()
    image_id, size = inspect.split()
    return {"image_tag": IMAGE_TAG, "image_id": image_id, "image_size": int(size)}


def _container_smoke(image_tag: str, smoke_window: Path) -> dict:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{smoke_window}:/tmp/smoke_window.npz:ro",
        image_tag,
        "/bin/bash",
        "-c",
        "python /h1_rift_falcon_decoder.py --smoke-payload /data/decoder.pkl --smoke-window /tmp/smoke_window.npz",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return json.loads(out)


def main() -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    os.chdir(ROOT)
    pkg = _copy_pkg()
    _use_packed_pkg(pkg)
    payload_path, payload_sha, selection = _build_payload()
    host = _host_verify(payload_path)
    docker = _docker_build(payload_sha)
    container = _container_smoke(docker["image_tag"], Path(host["smoke_window"]))
    candidate = {
        "arm": "h1_rift_r300_static_e32_cached",
        "budget_disclosure": (
            "H1 RIFT R300 static_identity table, no E0/no carrier/no calibration bank. "
            "Prespecified last-epoch EMA e32 after 32-epoch source-only train. "
            f"HO-M3 report {selection['mean']:.6f} is not epoch selection. "
            "Cached CPU learnable runtime, no TTA, no ORT."
        ),
        "evalai_opened": False,
        "image_id": docker["image_id"],
        "image_size": docker["image_size"],
        "image_tag": docker["image_tag"],
        "method_description": (
            "H1 RIFT R300 P16 static_identity (shared 176x16 table, literal-zero carrier), "
            "per-layer learned_slope on the default half-life ladder, cached CPU KV. "
            f"Prespecified e32 EMA. HO-M3 report {selection['mean']:.6f}. "
            "Not BT-EORT, not ORT, not SPINT, not 582196, not 582241."
        ),
        "method_label": METHOD_LABEL,
        "method_name": METHOD_NAME,
        "payload_sha256": payload_sha,
        "register": False,
        "selection_mean": selection["mean"],
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "window": 300,
        "host_verify": host,
        "container_smoke": container,
    }
    (DEST / "artifacts" / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PACKED_NOT_REGISTERED",
                **{k: candidate[k] for k in ("image_tag", "image_id", "payload_sha256", "image_size")},
                "host_verify": {k: host[k] for k in host if k != "smoke_window"},
                "container_smoke": container,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
