#!/usr/bin/env python3
"""Build the M1 RIFT R100 static_identity learned_slope e19 payload.

HO3 all24 earliest-max equal_session_mean pick. Host-verify streaming vs
offline and docker-smoke. Does not EvalAI-submit. Not muscle, not e24.
"""
from __future__ import annotations

import argparse
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
RUN = ROOT / "btransform_unified_v2/learnable_recency_v1/results/m1_static_s42"
CKPT = RUN / "epoch_019.pt"
CKPT_SHA = "e96544a73d251e2da0ed21da98546cb5452533feba557d80ffe9a76b2e9babce"
SCORE = RUN / "ho3_selection.json"
RUN_META = RUN / "run_meta.json"
DATA_ROOT = ROOT / "SPINT-main/data/000941"
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
LEARNABLE_SRC = ROOT / "btransform_unified_v2/learnable_recency_v1/src/learnable_recency_v1"
IMAGE_TAG = "m1-rift-r100-static-e19:v1"
METHOD_LABEL = (
    "M1 RIFT R100 static_identity learned_slope default-ladder e19 EMA cached CPU; "
    "architecture=RIFT backend=cached identity=static recency=learned_slope_default; "
    "HO3 all24 earliest-max equal_session_mean -0.142517; not muscle; not e24; not WF"
)
METHOD_NAME = "M1 RIFT R100 static e19"
N_STEPS = 80
GATE = 1.0e-5
EXPECTED_MEAN = -0.14251702679952535
SOURCE = ("20120924", "20120926", "20120927", "20120928")
HO = ("20121004", "20121017", "20121024")
OFFICIAL_TAGS = set(SOURCE + HO)
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    for path in (str(pkg), str(DEST), str(ROOT), str(ROOT / "SPINT-main"), str(ROOT / "streaming_calibration_exp")):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _stem(tag: str) -> str:
    kind = "held-in-calib" if tag.startswith("201209") else "held-out-calib"
    return f"sub-MonkeyL-{kind}_ses-{tag}_behavior+ecephys"


def _nwb_path(tag: str) -> Path:
    kind = "held-in-calib" if tag.startswith("201209") else "held-out-calib"
    return DATA_ROOT / f"sub-MonkeyL-{kind}" / f"{_stem(tag)}.nwb"


def _load_raw_neural(tag: str, n_steps: int) -> np.ndarray:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    neural, *_ = load_nwb(_nwb_path(tag), FalconTask.m1)
    raw = np.ascontiguousarray(neural[:n_steps], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 64 or raw.shape[0] < n_steps:
        raise RuntimeError(f"M1 neural shape drift for {tag}: {raw.shape}")
    return raw


def _offline_step(model, history: np.ndarray, context: int, scale: float) -> np.ndarray:
    take = min(history.shape[0], context)
    window = np.zeros((context, history.shape[1]), dtype=np.float32)
    valid = np.zeros((context,), dtype=bool)
    window[-take:] = history[-take:]
    valid[-take:] = True
    x = torch.from_numpy(window).unsqueeze(0)
    v = torch.from_numpy(valid).unsqueeze(0)
    with torch.inference_mode():
        pred = model(x, input_valid_mask=v)
    return pred.detach().cpu().numpy()[0] / scale


def _build_payload() -> tuple[Path, str, dict]:
    from btransform_unified_v1.ema import DecoderEMA
    from learnable_recency_v1.config import config_from_run_meta
    from m1_rift_falcon_decoder import PAYLOAD_SCHEMA, build_decoder

    if sha256(CKPT) != CKPT_SHA:
        raise RuntimeError("epoch_019.pt SHA drift")
    if CKPT.name == "epoch_024.pt":
        raise RuntimeError("refusing last-epoch sidecar e24")
    score = json.loads(SCORE.read_text())
    selected = score["selection"]
    if score.get("schema") != "m1_static_ho3_earliest_max_score_v1":
        raise RuntimeError(f"selection schema drift: {score.get('schema')}")
    if int(selected["epoch"]) != 19:
        raise RuntimeError(f"static selection drifted: {selected}")
    if abs(float(selected["equal_session_mean"]) - EXPECTED_MEAN) > 1.0e-12:
        raise RuntimeError(f"HO3 e19 mean drifted: {selected}")
    if score.get("official_test_used") is not False:
        raise RuntimeError("official test must not be used")
    row = score["ema_by_epoch"]["19"]
    if row.get("checkpoint_sha256") != CKPT_SHA or int(row.get("n_windows", -1)) != 3881:
        raise RuntimeError("e19 score row inventory drift")
    meta = json.loads(RUN_META.read_text())
    if meta.get("status") != "FORMAL" or meta.get("identity_interface") != "static_identity":
        raise RuntimeError(f"run_meta is not formal STATIC: {meta.get('status')} {meta.get('identity_interface')}")
    recency_cfg = config_from_run_meta(meta, "m1")
    if recency_cfg.tier != "learned_slope" or recency_cfg.ladder != "default" or not recency_cfg.per_layer:
        raise RuntimeError(f"run_meta learnable_config drifted: {recency_cfg}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != "m1_static_epoch_checkpoint_v1" or int(ckpt.get("epoch", -1)) != 19:
        raise RuntimeError("checkpoint is not formal static e19")
    if list(ckpt.get("static_identity_shape")) != [64, 16]:
        raise RuntimeError("checkpoint static_identity is not [64,16]")
    model = build_decoder(recency_cfg)
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    ema = DecoderEMA(model, decay=0.9995)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(model)
    model.eval()
    if bool(model.static_carrier.any()):
        raise RuntimeError("static carrier is not literal zero")
    if any(name.startswith("frontend.e0_proj") for name in model.state_dict()):
        raise RuntimeError("static pack must not contain e0_proj")
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "task": "m1",
        "context_bins": 100,
        "bias_mode": "learnable_learned_slope",
        "identity": "static",
        "identity_interface": "static_identity_table",
        "proj_dim": 16,
        "tier": "learned_slope",
        "ladder": "default",
        "per_layer": True,
        "seed": 42,
        "behavior_scaling_factor": 1.0,
        "checkpoint": str(CKPT),
        "checkpoint_sha256": CKPT_SHA,
        "learnable_config": dict(meta["learnable_config"]),
        "official_dataset_tags": sorted(OFFICIAL_TAGS),
        "ema_state_dict": {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()},
        "selection": {
            "surface": "HO3 query all24 earliest-max",
            "metric": "equal_session_mean",
            "epoch": 19,
            "seed": 42,
            "equal_session_mean": float(selected["equal_session_mean"]),
            "n_windows": 3881,
            "official_test_used": False,
            "local_ho_labels_used_for_selection": True,
            "rule": selected["rule"],
        },
    }
    dest = DEST / "artifacts" / "m1_rift_r100_static_e19.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return dest, sha256(dest), payload["selection"]


def _host_verify(payload_path: Path) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask
    from m1_rift_falcon_decoder import (
        CpuStaticRiftRuntime,
        M1RiftCachedFalconDecoder,
        build_decoder,
        load_payload,
        recency_config_from_payload,
    )

    payload = load_payload(payload_path)
    config = FalconConfig(task=FalconTask.m1)
    recency_cfg = recency_config_from_payload(payload)
    report: dict = {
        "tolerance": {
            "stream_vs_offline_gate": GATE,
            "stream_vs_cached_gate": GATE,
            "note": "left-pad bins are invalid in the offline window; Falcon predict bins are valid observations",
        }
    }
    needed = list(dict.fromkeys(list(HO) + list(SOURCE)))
    traces = {tag: _load_raw_neural(tag, N_STEPS) for tag in needed}
    plans = (("B1", ["20121004"]), ("B3", list(HO)), ("B4", list(SOURCE)))
    for name, tags in plans:
        stems = [_stem(tag) for tag in tags]
        hashed = [config.hash_dataset(Path(stem).stem) for stem in stems]
        missing = [tag for tag in hashed if tag not in payload["official_dataset_tags"]]
        if missing:
            raise RuntimeError(f"hashed tags missing from payload: {missing}")
        packed = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=len(tags))
        packed.reset(dataset_tags=stems)
        model = build_decoder(recency_cfg)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in payload["ema_state_dict"].items()}, strict=True)
        model.eval()
        ref = CpuStaticRiftRuntime(model, [str(tag) for tag in hashed])
        max_stream_offline = 0.0
        max_stream_cached = 0.0
        last = None
        histories = [np.zeros((0, 64), dtype=np.float32) for _ in tags]
        for step in range(N_STEPS):
            batch_x = np.stack([traces[tag][step] for tag in tags], axis=0)
            got = packed.predict(batch_x)
            want_cached = ref.advance(torch.from_numpy(batch_x)).numpy() / 1.0
            offline = []
            for i, tag in enumerate(tags):
                histories[i] = np.concatenate([histories[i], traces[tag][step : step + 1]], axis=0)
                offline.append(_offline_step(model, histories[i], 100, 1.0))
            want_offline = np.stack(offline, axis=0)
            last = got
            max_stream_cached = max(max_stream_cached, float(np.max(np.abs(got - want_cached))))
            max_stream_offline = max(max_stream_offline, float(np.max(np.abs(got - want_offline))))
            if max_stream_cached > GATE or max_stream_offline > GATE:
                raise RuntimeError(
                    f"{name} host gate fail t={step} stream_vs_cached={max_stream_cached} stream_vs_offline={max_stream_offline}"
                )
        report[name] = {
            "steps": N_STEPS,
            "stream_vs_cached_max_abs": max_stream_cached,
            "stream_vs_offline_max_abs": max_stream_offline,
            "last_shape": list(last.shape),
            "sessions": tags,
            "tags": hashed,
        }
    smoke = DEST / "artifacts" / "smoke_window.npz"
    tag = "20121004"
    neural = np.ascontiguousarray(traces[tag][:40], dtype=np.float32)
    packed = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    packed.reset(dataset_tags=[_stem(tag)])
    pred = None
    for row in neural:
        pred = packed.predict(row.reshape(1, -1))
    np.savez(smoke, tag_stem=np.asarray(_stem(tag)), window=neural, expected=pred)
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


def _container_smoke(image_tag: str, smoke_window: Path) -> dict:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-v",
        f"{smoke_window}:/tmp/smoke_window.npz:ro",
        image_tag,
        "python",
        "/m1_rift_falcon_decoder.py",
        "--smoke-payload",
        "/data/decoder.pkl",
        "--smoke-window",
        "/tmp/smoke_window.npz",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    parsed = json.loads(out.splitlines()[-1])
    if parsed.get("status") != "CONTAINER_SMOKE_PASS":
        raise RuntimeError(f"container smoke unexpected: {out}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    os.chdir(ROOT)
    pkg = _copy_pkg()
    _use_packed_pkg(pkg)
    payload_path, payload_sha, selection = _build_payload()
    host = _host_verify(payload_path)
    (DEST / "artifacts" / "host_verify.json").write_text(json.dumps(host, indent=2, sort_keys=True) + "\n")
    if args.skip_docker:
        print(json.dumps({"status": "HOST_ONLY", "payload_sha256": payload_sha, "host_verify": {k: host[k] for k in host if k != "smoke_window"}}, indent=2))
        return 0
    docker = _docker_build(payload_sha)
    container = _container_smoke(docker["image_tag"], Path(host["smoke_window"]))
    candidate = {
        "arm": "m1_rift_r100_static_e19_cached",
        "budget_disclosure": (
            "M1 RIFT R100 static_identity[64,16], no E0 / no e0_proj / no muscle carrier / no calibration bank. "
            "Source-only train 24 epochs; HO3 query labels used only to pick earliest-max EMA epoch 19 "
            f"(equal_session_mean {selection['equal_session_mean']:.6f}). "
            "This query-label epoch pick is different from source-selected linear WF. "
            "No official test, no TTA, no ORT, not fair_v2 diag-z/CORAL/WF."
        ),
        "evalai_opened": False,
        "image_id": docker["image_id"],
        "image_size": docker["image_size"],
        "image_tag": docker["image_tag"],
        "method_description": (
            "M1 RIFT R100 P16 static_identity (shared 64x16 table, literal-zero carrier), "
            "per-layer learned_slope on the default half-life ladder, cached CPU KV. "
            f"HO3 all24 earliest-max e19 equal_session_mean {selection['equal_session_mean']:.6f}. "
            "Not muscle, not e24, not BT-EORT, not ORT, not SPINT, not WF."
        ),
        "method_label": METHOD_LABEL,
        "method_name": METHOD_NAME,
        "payload_sha256": payload_sha,
        "register": False,
        "selection": selection,
        "selection_mean": selection["equal_session_mean"],
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "window": 100,
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
