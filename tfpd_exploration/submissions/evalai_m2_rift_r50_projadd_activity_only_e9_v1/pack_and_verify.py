#!/usr/bin/env python3
"""Build the M2 RIFT R50 proj_add P16 activity_only (carrier T=0) learned_slope e9 payload.

Host-verify against CpuLearnableRecencyRuntime, then docker-pack and container smoke test.
Does not EvalAI-submit.
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
RUN = ROOT / "btransform_unified_v2/learnable_recency_v1/results/m2_projadd_activity_only_s42"
CKPT = RUN / "epoch_009.pt"
CKPT_SHA = "0ba428019134c79ce2ff444e027719b0c45d907637d7c240a79466420b9939ac"
SCORE = ROOT / "btransform_unified_v2/learnable_recency_v1/results/selection_m2_projadd_activity_only_s42_ext6/score_receipt.json"
BANK_PAYLOAD = ROOT / "tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
CACHE = ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train"
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
LEARNABLE_SRC = ROOT / "btransform_unified_v2/learnable_recency_v1/src/learnable_recency_v1"
IMAGE_TAG = "m2-rift-r50-projadd-activity-only-e9:v1"
METHOD_LABEL = (
    "M2 RIFT R50 proj_add P16 activity_only (carrier T=0) learned_slope default-ladder e9 EMA cached CPU; "
    "architecture=RIFT backend=cached identity=activity_only; ext6 pick 0.359483; "
    "not BT-EORT; not ORT; not SPINT"
)
METHOD_NAME = "M2 RIFT R50 proj_add activity_only e9"
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
    for path in (str(pkg), str(DEST)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _build_payload() -> tuple[Path, str, dict]:
    from btransform_unified_v1.ema import DecoderEMA
    from learnable_recency_v1.config import config_from_run_meta
    from m2_rift_falcon_decoder import CPUUnpickler, PAYLOAD_SCHEMA, build_decoder

    if sha256(CKPT) != CKPT_SHA:
        raise RuntimeError("epoch_009.pt SHA drift")
    with open(BANK_PAYLOAD, "rb") as handle:
        bt = CPUUnpickler(handle).load()
    raw_banks = bt["bank_by_dataset_tag"]
    if len(raw_banks) != 13:
        raise RuntimeError(f"expected 13 official tags, got {len(raw_banks)}")
    
    # Apply activity_only identity transform: carrier T is zeroed out, E0 is untouched
    banks = {}
    for tag, row in raw_banks.items():
        t = np.asarray(row["T"])
        t_zero = np.zeros(t.shape, dtype=np.float32, order="C")
        banks[tag] = {
            "session": row.get("session") or tag,
            "E0": np.array(row["E0"], dtype=np.float32, order="C", copy=True),
            "T": t_zero,
            "unit_mask": np.array(row["unit_mask"], dtype=np.bool_, copy=True),
            "e0_sha256": row.get("e0_sha256") or hashlib.sha256(np.ascontiguousarray(row["E0"]).tobytes()).hexdigest(),
            "t_sha256": hashlib.sha256(t_zero.tobytes()).hexdigest(),
        }

    score = json.loads(SCORE.read_text())
    selected = score["selection"]
    if int(selected["epoch"]) != 9:
        raise RuntimeError(f"ext6 score receipt selection drifted: {selected}")
    if abs(float(selected["equal_session_mean"]) - 0.3594833960225496) > 1.0e-12:
        raise RuntimeError(f"ext6 e9 mean drifted: {selected}")
    meta = json.loads((RUN / "run_meta.json").read_text())
    if meta.get("identity") != "activity_only":
        raise RuntimeError(f"run_meta identity is not activity_only: {meta.get('identity')}")
    recency_cfg = config_from_run_meta(meta, "m2")
    if recency_cfg.ladder != "default" or recency_cfg.tier != "learned_slope" or not recency_cfg.per_layer:
        raise RuntimeError(f"run_meta learnable_config drifted: {recency_cfg}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != "m2_rift_projadd_ablation_epoch_checkpoint_v1":
        raise RuntimeError("checkpoint schema is not M2 proj_add ablation")
    if ckpt.get("identity") != "activity_only":
        raise RuntimeError("checkpoint identity is not activity_only")
    if ckpt.get("identity_interface") != "proj_add" or int(ckpt.get("proj_dim", -1)) != 16:
        raise RuntimeError("checkpoint is not proj_add P16")
    if int(ckpt.get("epoch", -1)) != 9 or ckpt.get("tier") != "learned_slope":
        raise RuntimeError("checkpoint is not learned_slope e9")
    model = build_decoder(recency_cfg)
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    ema = DecoderEMA(model, decay=0.9995)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(model)
    model.eval()
    ema_state = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()}
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "task": "m2",
        "context_bins": 50,
        "bias_mode": "learnable_learned_slope",
        "identity": "activity_only",
        "identity_interface": "proj_add",
        "proj_dim": 16,
        "tier": "learned_slope",
        "ladder": "default",
        "per_layer": True,
        "behavior_scaling_factor": 5.0,
        "checkpoint": str(CKPT),
        "checkpoint_sha256": CKPT_SHA,
        "bank_source_payload_sha256": sha256(BANK_PAYLOAD),
        "learnable_config": dict(meta["learnable_config"]),
        "bank_by_dataset_tag": banks,
        "ema_state_dict": ema_state,
        "selection": {
            "surface": "ext6 development",
            "epoch": 9,
            "equal_session_mean": float(selected["equal_session_mean"]),
            "pooled_r2": float(score["ema_by_epoch"]["9"]["pooled_r2"]),
            "n_windows": 15403,
            "official_test_used": False,
        },
    }
    dest = DEST / "artifacts" / "m2_rift_r50_projadd_activity_only_e9.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return dest, sha256(dest), payload["selection"]


def _tag_session(tag: str) -> str:
    run, ymd = tag.split("_")
    return f"ses-{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}-{run}"


def _host_verify(payload_path: Path) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask
    from m2_rift_falcon_decoder import (
        CpuLearnableRiftRuntime,
        M2RiftCachedFalconDecoder,
        _task_bank,
        build_decoder,
        load_payload,
        recency_config_from_payload,
    )

    payload = load_payload(payload_path)
    config = FalconConfig(task=FalconTask.m2)
    report = {}
    plans = (("B1", ["Run1_20201019"]), ("B7", [
        "Run1_20201019",
        "Run2_20201019",
        "Run1_20201020",
        "Run2_20201020",
        "Run1_20201027",
        "Run2_20201027",
        "Run1_20201028",
    ]))
    recency_cfg = recency_config_from_payload(payload)
    for name, tags in plans:
        missing = [tag for tag in tags if tag not in payload["bank_by_dataset_tag"]]
        if missing:
            raise RuntimeError(f"hashed tags missing from payload: {missing}")
        streams = [
            np.ascontiguousarray(np.load(CACHE / _tag_session(tag) / "X_store.npy")[:N_STEPS], dtype=np.float32)
            for tag in tags
        ]
        packed = M2RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=len(tags))
        packed.reset(dataset_tags=tags)
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
            want = ref.advance(torch.from_numpy(batch_x)).numpy() / 5.0
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
    tag = "Run1_20201019"
    neural = np.ascontiguousarray(np.load(CACHE / _tag_session(tag) / "X_store.npy")[:40], dtype=np.float32)
    packed = M2RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    packed.reset(dataset_tags=[tag])
    pred = None
    for row in neural:
        pred = packed.predict(row.reshape(1, -1))
    np.savez(smoke, tag_stem=np.asarray(tag), window=neural, expected=pred)
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


def _container_smoke(image_tag: str, payload_path: Path, smoke_window: Path) -> dict:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{smoke_window}:/tmp/smoke_window.npz:ro",
        image_tag,
        "/bin/bash",
        "-c",
        "python /m2_rift_falcon_decoder.py --smoke-payload /data/decoder.pkl --smoke-window /tmp/smoke_window.npz",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return json.loads(out)


def main() -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    pkg = _copy_pkg()
    _use_packed_pkg(pkg)
    payload_path, payload_sha, selection = _build_payload()
    host = _host_verify(payload_path)
    docker = _docker_build(payload_sha)
    container = _container_smoke(docker["image_tag"], payload_path, Path(host["smoke_window"]))
    candidate = {
        "arm": "m2_rift_r50_projadd_activity_only_e9_cached",
        "budget_disclosure": (
            "Official 13-tag M2 banks with native E0 and zero carrier (carrier T=0, activity_only). "
            "RIFT R50 proj_add P16 seed42 EMA e9 after 24-epoch source-seven train, "
            "per-layer learned_slope on the unscaled default half-life ladder. "
            f"ext6 equal-session {selection['equal_session_mean']:.6f} is development selection, "
            "not official HO. Cached CPU learnable runtime, no TTA, no ORT."
        ),
        "evalai_opened": False,
        "image_id": docker["image_id"],
        "image_size": docker["image_size"],
        "image_tag": docker["image_tag"],
        "method_description": (
            "M2 RIFT (Recency-biased Incremental Finite-context Transformer) R50, "
            "D4/width256, proj_add P16 identity with carrier T=0 (activity_only), "
            "per-layer learned recency slopes on the default (H1-scale) half-life ladder, cached CPU KV. "
            f"Official 13-tag banks with zero carrier. ext6 pick e9 mean {selection['equal_session_mean']:.6f}. "
            "Not BT-EORT, not ORT, not SPINT."
        ),
        "method_label": METHOD_LABEL,
        "method_name": METHOD_NAME,
        "payload_sha256": payload_sha,
        "register": False,
        "selection_mean": selection["equal_session_mean"],
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "window": 50,
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
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
