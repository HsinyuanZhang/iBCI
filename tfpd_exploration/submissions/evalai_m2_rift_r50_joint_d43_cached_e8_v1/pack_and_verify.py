#!/usr/bin/env python3
"""Build the M2 RIFT R50 joint D43 e8 cached payload, host-verify, and docker-pack.

Seals live FiLM E0 from the D43 EMA encoder on public M33. Keeps official
MOVE-T4. Does not EvalAI-submit. Does not overwrite 582047, 582189, or H1 582073.
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
CKPT = ROOT / "btransform_unified_v2/results/rift_v1/m2_r50_joint_d_s43_formal_v1/epoch_008.pt"
CKPT_SHA = "931a3f8b4f929db0037a472a4c6f85222c8faddf2fca6a9d76acec4f4eb2aeec"
SELECTED_EMA = ROOT / "btransform_unified_v2/results/rift_v1/m2_r50_joint_d_s43_ext6_pick_v1/selected_ema.pt"
SELECTED_EMA_SHA = "9523e5ea5e8d4049307d20618ba7cad43c31e18816ea549dbdd8c6151ebccf92"
SCORE = ROOT / "btransform_unified_v2/results/rift_v1/m2_r50_joint_d_s43_ext6_pick_v1/score_receipt.json"
BANK_PAYLOAD = ROOT / "tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
CACHE = ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
JOINT_M33 = ROOT / "btransform_unified_v2/results/rift_v1/m2_joint_ext6_raw_m33_v1"
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
IMAGE_TAG = "m2-rift-r50-joint-d43-cached-e8:v1"
METHOD_LABEL = (
    "M2 RIFT R50 joint D43 e8 EMA cached CPU; architecture=RIFT backend=cached "
    "identity=live_film_e0_sealed; ext6 pick; not BT-EORT; not ORT; not SPINT"
)
METHOD_NAME = "M2 RIFT R50 joint D43 e8 cached"
N_STEPS = 80
GATE = 1.0e-5
MODEL_SEED = 43
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
TAG_SESSION = {
    "Run1_20201019": "ses-2020-10-19-Run1",
    "Run1_20201020": "ses-2020-10-20-Run1",
    "Run1_20201027": "ses-2020-10-27-Run1",
    "Run1_20201028": "ses-2020-10-28-Run1",
    "Run1_20201030": "ses-2020-10-30-Run1",
    "Run1_20201118": "ses-2020-11-18-Run1",
    "Run1_20201119": "ses-2020-11-19-Run1",
    "Run1_20201124": "ses-2020-11-24-Run1",
    "Run2_20201019": "ses-2020-10-19-Run2",
    "Run2_20201020": "ses-2020-10-20-Run2",
    "Run2_20201027": "ses-2020-10-27-Run2",
    "Run2_20201030": "ses-2020-10-30-Run2",
    "Run2_20201124": "ses-2020-11-24-Run2",
}
TAG_SURFACE = {
    "Run1_20201019": "source_train",
    "Run1_20201020": "source_train",
    "Run1_20201027": "source_train",
    "Run1_20201028": "source_train",
    "Run1_20201030": "ext4",
    "Run1_20201118": "ext4",
    "Run1_20201119": "ext4",
    "Run1_20201124": "heldout_calib_nov24",
    "Run2_20201019": "source_train",
    "Run2_20201020": "source_train",
    "Run2_20201027": "source_train",
    "Run2_20201030": "ext4",
    "Run2_20201124": "heldout_calib_nov24",
}

for path in (str(V2_SRC), str(V1_SRC), str(DEST), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _copy_pkg() -> None:
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


def _calib_path(tag: str) -> Path:
    session = TAG_SESSION[tag]
    surface = TAG_SURFACE[tag]
    if surface == "heldout_calib_nov24":
        path = JOINT_M33 / session / "calib_activity.npy"
    else:
        path = CACHE / surface / session / "calib_activity.npy"
        joint = JOINT_M33 / session / "calib_activity.npy"
        if joint.is_file():
            if not np.array_equal(np.load(path), np.load(joint)):
                raise RuntimeError(f"ext6 M33 drift vs dual-track at {session}")
    if not path.is_file():
        raise RuntimeError(f"missing public M33 for {tag}: {path}")
    return path


def _install(model, session: str, carrier: np.ndarray, calib_path: Path) -> None:
    raw = np.ascontiguousarray(np.load(calib_path), dtype=np.float32)
    if raw.shape != (33, 100, 96):
        raise RuntimeError(f"raw M33 geometry drift {session}: {raw.shape}")
    key = session.replace("-", "_")
    if hasattr(model, f"calib_{key}"):
        prior = getattr(model, f"calib_{key}").detach().cpu().numpy()
        if not np.array_equal(prior, raw):
            raise RuntimeError(f"cross-surface calibration drift {session}")
    else:
        model.register_buffer(f"calib_{key}", torch.from_numpy(raw), persistent=False)
        model.register_buffer(f"carrier_{key}", torch.from_numpy(np.ascontiguousarray(carrier, dtype=np.float32)), persistent=False)
    model._calib[session] = getattr(model, f"calib_{key}")
    model._carrier[session] = getattr(model, f"carrier_{key}")


def _build_payload() -> tuple[Path, str, dict]:
    from btransform_unified_v1.ema import DecoderEMA
    from btransform_unified_v2.joint_m2_model import ARM_D, JointM2RiftDecoder
    from btransform_unified_v2.model import RiftDecoder
    from m2_rift_falcon_decoder import CPUUnpickler, PAYLOAD_SCHEMA

    if sha256(CKPT) != CKPT_SHA:
        raise RuntimeError("epoch_008.pt SHA drift")
    if sha256(SELECTED_EMA) != SELECTED_EMA_SHA:
        raise RuntimeError("selected_ema.pt SHA drift")
    with open(BANK_PAYLOAD, "rb") as handle:
        bt = CPUUnpickler(handle).load()
    official = bt["bank_by_dataset_tag"]
    if set(official) != set(TAG_SESSION):
        raise RuntimeError(f"official M2 tag set drifted: {sorted(official)}")
    score = json.loads(SCORE.read_text())
    selected = score["selection"]
    if int(selected["epoch"]) != 8:
        raise RuntimeError(f"ext6 score receipt selection drifted: {selected}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != "m2_rift_joint_checkpoint_v2" or ckpt.get("arm") != ARM_D:
        raise RuntimeError("checkpoint is not formal M2 joint D")
    if int(ckpt.get("epoch", -1)) != 8 or int(ckpt.get("seed", -1)) != MODEL_SEED:
        raise RuntimeError("checkpoint is not formal M2 joint D43 e8")
    joint = JointM2RiftDecoder(ARM_D, seed=MODEL_SEED)
    joint.load_state_dict(ckpt["model"], strict=True)
    ema = DecoderEMA(joint, decay=0.9995)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(joint)
    picked = torch.load(SELECTED_EMA, map_location="cpu", weights_only=False)
    current = joint.state_dict()
    for name, value in current.items():
        if name not in picked:
            raise RuntimeError(f"selected_ema missing {name}")
        if not torch.equal(value.cpu(), picked[name].cpu().float()):
            raise RuntimeError(f"selected_ema drift at {name}")
    joint.eval()

    sealed = {}
    for tag, row in official.items():
        session = TAG_SESSION[tag]
        official_t = np.ascontiguousarray(row["T"], dtype=np.float32)
        cache_t = CACHE / TAG_SURFACE[tag] / session / "T.npy" if TAG_SURFACE[tag] != "heldout_calib_nov24" else None
        if cache_t is not None and cache_t.is_file():
            if not np.array_equal(official_t, np.ascontiguousarray(np.load(cache_t), dtype=np.float32)):
                raise RuntimeError(f"official MOVE-T4 drift vs dual-track at {tag}")
        _install(joint, session, official_t, _calib_path(tag))
        with torch.inference_mode():
            e0, carrier = joint._identity([session], torch.device("cpu"))
        e0_np = np.ascontiguousarray(e0[0].cpu().numpy(), dtype=np.float32)
        t_np = np.ascontiguousarray(carrier[0].cpu().numpy(), dtype=np.float32)
        if not np.array_equal(t_np, official_t):
            raise RuntimeError(f"live carrier T drift vs official MOVE-T4 at {tag}")
        sealed[tag] = {
            "E0": e0_np,
            "T": official_t,
            "unit_mask": np.ascontiguousarray(row["unit_mask"], dtype=np.bool_),
            "session": session,
            "e0_sha256": _array_sha(e0_np),
            "t_sha256": _array_sha(official_t),
        }
        if np.array_equal(e0_np, np.ascontiguousarray(row["E0"], dtype=np.float32)):
            raise RuntimeError(f"live FiLM E0 collapsed to native E0 at {tag}")

    rift = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=MODEL_SEED)
    allowed = set(rift.state_dict())
    ema_state = {k: v.detach().cpu().float().clone() for k, v in joint.state_dict().items() if k in allowed}
    missing, unexpected = rift.load_state_dict(ema_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"rift state mismatch missing={missing} unexpected={unexpected}")
    rift.eval()

    payload = {
        "schema": PAYLOAD_SCHEMA,
        "task": "m2",
        "context_bins": 50,
        "bias_mode": "recency",
        "identity_interface": "live_film_e0_sealed",
        "behavior_scaling_factor": 5.0,
        "checkpoint": str(CKPT),
        "checkpoint_sha256": CKPT_SHA,
        "selected_ema_sha256": SELECTED_EMA_SHA,
        "bank_source_payload_sha256": sha256(BANK_PAYLOAD),
        "bank_by_dataset_tag": sealed,
        "ema_state_dict": {k: v.detach().cpu().float().clone() for k, v in rift.state_dict().items()},
        "selection": {
            "surface": "ext6 development",
            "epoch": 8,
            "equal_session_mean": float(selected["equal_session_mean"]),
            "pooled_r2": float(score["ema_by_epoch"]["8"]["pooled_r2"]),
            "n_windows": 15403,
            "official_test_used": False,
        },
    }
    dest = DEST / "artifacts" / "m2_rift_r50_joint_d43_e8.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return dest, sha256(dest), payload["selection"]


def _tag_session(tag: str) -> str:
    return TAG_SESSION[tag]


def _host_verify(payload_path: Path) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask
    from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
    from btransform_unified_v2.model import RiftDecoder
    from m2_rift_falcon_decoder import M2RiftCachedFalconDecoder, _task_bank, load_payload

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
    for name, tags in plans:
        missing = [tag for tag in tags if tag not in payload["bank_by_dataset_tag"]]
        if missing:
            raise RuntimeError(f"hashed tags missing from payload: {missing}")
        streams = [
            np.ascontiguousarray(np.load(CACHE / "source_train" / _tag_session(tag) / "X_store.npy")[:N_STEPS], dtype=np.float32)
            for tag in tags
        ]
        packed = M2RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=len(tags))
        packed.reset(dataset_tags=tags)
        model = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=MODEL_SEED)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in payload["ema_state_dict"].items()}, strict=True)
        model.eval()
        banks = [_task_bank(tag, payload["bank_by_dataset_tag"][tag]) for tag in tags]
        ref = CpuRiftRuntime(model, banks, [str(tag) for tag in tags], temporal_backend="cached")
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
    neural = np.ascontiguousarray(np.load(CACHE / "source_train" / _tag_session(tag) / "X_store.npy")[:40], dtype=np.float32)
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


def main() -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _copy_pkg()
    payload_path, payload_sha, selection = _build_payload()
    host = _host_verify(payload_path)
    docker = _docker_build(payload_sha)
    candidate = {
        "arm": "m2_rift_r50_joint_d43_e8_cached",
        "budget_disclosure": (
            "Official 13-tag M2 MOVE-T4 (same sealed T as 582047). Live FiLM E0 "
            "sealed from joint-D43 EMA e8 on public M33. RIFT R50 proj_add seed43 "
            "after 24-epoch source-seven train. ext6 equal-session "
            f"{selection['equal_session_mean']:.6f} is development selection, not official HO. "
            "Cached CPU runtime, no TTA, no ORT."
        ),
        "evalai_opened": False,
        "image_id": docker["image_id"],
        "image_size": docker["image_size"],
        "image_tag": docker["image_tag"],
        "method_description": (
            "M2 RIFT joint D43 (live FiLM M33 + proj_add R50/D4), recency, cached CPU KV. "
            f"Same MOVE-T4 as 582047; E0 is trained FiLM on public M33. ext6 pick e8 mean "
            f"{selection['equal_session_mean']:.6f}. Not BT-EORT, not ORT, not SPINT, not 582189."
        ),
        "method_label": METHOD_LABEL,
        "method_name": METHOD_NAME,
        "payload_sha256": payload_sha,
        "register": False,
        "selection_mean": selection["equal_session_mean"],
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "window": 50,
        "host_verify": host,
    }
    (DEST / "artifacts" / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PACKED_NOT_REGISTERED",
                **{k: candidate[k] for k in ("image_tag", "image_id", "payload_sha256", "image_size")},
                "host_verify": {k: host[k] for k in host if k != "smoke_window"},
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
