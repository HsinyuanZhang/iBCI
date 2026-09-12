#!/usr/bin/env python3
"""Build the H1 RIFT R300 flat (zero-slope) e15 cached payload, host-verify, and docker-pack.

Seals signed_state14 T + rematerialized C2 E0 + plain RiftTemporal zero slopes.
Does not EvalAI-submit. Does not overwrite 582196.
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
CKPT = ROOT / "btransform_unified_v2/learnable_recency_v1/results/h1_flat_p16_s42/epoch_015.pt"
CKPT_SHA = "3fa99979792272cabedcab6ce118baa51f4c2b97c04b178cafeb3676d9bb66b4"
SCORE = ROOT / "btransform_unified_v2/learnable_recency_v1/results/h1_flat_p16_s42/ho_m3_selection.json"
RUN_META = ROOT / "btransform_unified_v2/learnable_recency_v1/results/h1_flat_p16_s42/run_meta.json"
BANKS_DIR = ROOT / "btransform_unified_v2/results/h1_signed_state_r300_v1/banks_official13_20260909"
BANKS_NPZ = BANKS_DIR / "banks_27.npz"
BANKS_SHA = "6cff806fdf55daa0cf2bd6ac09b6db94d66188d5941490fcff10a403d1e8686b"
MASK_PAYLOAD = ROOT / "tfpd_exploration/submissions/evalai_h1_c2_cal1_b2_ort_v1/artifacts/h1_c2_cal1_b2_s42_ema_e18_L200.pkl"
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
LEARNABLE_SRC = ROOT / "btransform_unified_v2/learnable_recency_v1/src/learnable_recency_v1"
IMAGE_TAG = "h1-rift-r300-flat-e15:v1"
METHOD_LABEL = (
    "H1 RIFT R300 flat zero-slope e15 EMA cached CPU; architecture=RIFT backend=cached "
    "identity=signed_state14 recency=flat_fixed; not BT-EORT; not ORT; not SPINT; not 582196"
)
METHOD_NAME = "H1 RIFT R300 flat e15 cached"
METHOD_DESCRIPTION = (
    "H1 RIFT R300 with per-layer fixed zero slopes (flat control), epoch 15 EMA. "
    "HO-M3 grouped-seven pick e15 r2_mean 0.4658 worst 0.2241. Cached PyTorch FP32 CPU backend. "
    "Not BT-EORT, not ORT, not SPINT, not 582196."
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

LEARNABLE_SAFE_INIT = '''"""Container-safe learnable-recency package."""

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

for path in (str(LEARNABLE_SRC.parent), str(V2_SRC), str(V1_SRC), str(DEST), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


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
    from h1_rift_falcon_decoder import CPUUnpickler, PAYLOAD_SCHEMA, build_decoder

    if sha256(CKPT) != CKPT_SHA:
        raise RuntimeError(f"epoch_015.pt SHA drift: {sha256(CKPT)} != {CKPT_SHA}")
    if sha256(BANKS_NPZ) != BANKS_SHA:
        raise RuntimeError("banks_27.npz SHA drift")
    score = json.loads(SCORE.read_text())
    selected = score["selected"]
    if int(selected["epoch"]) != 15:
        raise RuntimeError(f"HO-M3 selection drifted: {selected['epoch']}")
    run_meta = json.loads(RUN_META.read_text())
    recency_cfg = config_from_run_meta(run_meta, "h1")
    if recency_cfg.tier != "fixed" or recency_cfg.ladder != "default" or not recency_cfg.per_layer:
        raise RuntimeError(f"run_meta learnable_config drifted: {recency_cfg}")
    if any(half is not None for half in recency_cfg.half_life_seconds):
        raise RuntimeError("flat pack requires all half-lives None")
    receipt = json.loads((BANKS_DIR / "receipt.json").read_text())
    if receipt.get("schema") != "h1_signed_state_r300_27tag_v1" or receipt.get("tag_count") != 27:
        raise RuntimeError("signed-state bank receipt drift")
    with open(MASK_PAYLOAD, "rb") as handle:
        official = CPUUnpickler(handle).load()["bank_by_dataset_tag"]
    if set(official) != set(receipt["tags"]):
        raise RuntimeError(f"official 27-tag set drifted vs signed-state banks: {sorted(set(official) ^ set(receipt['tags']))}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("variant") != "learnable_fixed" or int(ckpt.get("context_bins", 0)) != 300:
        raise RuntimeError("checkpoint is not formal H1 R300 learnable_fixed")
    if int(ckpt.get("epoch", -1)) != 15:
        raise RuntimeError("checkpoint is not e15")
    model = build_decoder(recency_cfg)
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    ema = DecoderEMA(model, decay=0.9995)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(model)
    model.eval()

    sealed = {}
    with np.load(BANKS_NPZ) as handle:
        for tag, row in official.items():
            e0 = np.ascontiguousarray(handle[f"E0/{tag}"], dtype=np.float32)
            t = np.ascontiguousarray(handle[f"T/{tag}"], dtype=np.float32)
            old_t = np.ascontiguousarray(row["T"], dtype=np.float32)
            if np.array_equal(t, old_t):
                raise RuntimeError(f"{tag}: signed-state T collapsed to H-C T")
            sealed[tag] = {
                "E0": e0,
                "T": t,
                "unit_mask": np.ascontiguousarray(row["unit_mask"], dtype=np.bool_),
                "session": receipt["tags"][tag]["session"],
                "e0_sha256": _array_sha(e0),
                "t_sha256": _array_sha(t),
            }

    payload = {
        "schema": PAYLOAD_SCHEMA,
        "task": "h1",
        "context_bins": 300,
        "bias_mode": "recency",
        "recency_tier": "fixed",
        "recency_ladder": "default",
        "identity_interface": "signed_state14",
        "proj_dim": 16,
        "behavior_scaling_factor": 20.0,
        "checkpoint": str(CKPT),
        "checkpoint_sha256": CKPT_SHA,
        "banks_27_sha256": BANKS_SHA,
        "bank_source_payload_sha256": sha256(MASK_PAYLOAD),
        "learnable_config": dict(run_meta["learnable_config"]),
        "bank_by_dataset_tag": sealed,
        "ema_state_dict": {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()},
        "selection": {
            "surface": "C2 HO-M3 development",
            "epoch": 15,
            "mean": float(selected["val_ho_m3_grouped/r2_mean"]),
            "worst_session": "S7",
            "worst": float(selected["worst_session_r2"]),
            "official_test_used": False,
        },
    }
    dest = DEST / "artifacts" / "h1_rift_r300_flat_e15.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return dest, sha256(dest), payload["selection"]


def _host_verify(payload_path: Path) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask
    from btransform_unified_v1 import adapters
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
    from h1_rift_falcon_decoder import (
        H1RiftLearnableFalconDecoder,
        CpuLearnableRiftRuntime,
        _task_bank,
        build_decoder,
        load_payload,
        recency_config_from_payload,
    )

    payload = load_payload(payload_path)
    cache = adapters._h1_source_cache()
    config = FalconConfig(task=FalconTask.h1)
    recency_cfg = recency_config_from_payload(payload)
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
        packed = H1RiftLearnableFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=batch)
        packed.reset(dataset_tags=stems)
        model = build_decoder(recency_cfg)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in payload["ema_state_dict"].items()}, strict=True)
        model.eval()
        banks = [_task_bank(tag, payload["bank_by_dataset_tag"][tag]) for tag in tags]
        ref = CpuLearnableRiftRuntime(model, banks, [str(tag) for tag in tags])
        max_abs = 0.0
        last = None
        for step in range(n_steps):
            batch_x = np.stack([row[step] for row in streams], axis=0)
            got = packed.predict(batch_x)
            want = ref.advance(torch.from_numpy(batch_x)).numpy() / 20.0
            last = got
            diff = float(np.max(np.abs(got - want)))
            if diff > max_abs:
                max_abs = diff
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


def _docker_build_and_smoke(payload_path: Path, payload_sha: str, selection: dict, host_report: dict) -> dict:
    smoke_npz = Path(host_report["smoke_window"]).resolve()
    if not smoke_npz.is_file():
        raise RuntimeError(f"smoke file {smoke_npz} missing")
    build_cmd = [
        "docker",
        "build",
        "-t",
        IMAGE_TAG,
        "--build-arg",
        f"PAYLOAD_SHA256={payload_sha}",
        str(DEST),
    ]
    print(f"[docker] running {' '.join(build_cmd)}")
    subprocess.run(build_cmd, check=True)
    img_cmd = ["docker", "image", "inspect", IMAGE_TAG, "--format", "{{.Id}}"]
    image_id = subprocess.check_output(img_cmd, text=True).strip()
    smoke_code = f"""
import numpy as np
from pathlib import Path
from falcon_challenge.config import FalconConfig, FalconTask
from h1_rift_falcon_decoder import H1RiftLearnableFalconDecoder

data = np.load('/tmp/smoke.npz')
stem = str(data['tag_stem'])
window = data['window']
expected = data['expected']
config = FalconConfig(task=FalconTask.h1)
decoder = H1RiftLearnableFalconDecoder(task_config=config, model_path='/data/decoder.pkl', batch_size=1)
decoder.reset(dataset_tags=[Path(stem)])
last = None
for row in window:
    last = decoder.predict(row.reshape(1, -1))
diff = float(np.max(np.abs(last - expected)))
print(f'CONTAINER_SMOKE_PASS diff={{diff}}')
if diff > 1.0e-5:
    raise SystemExit(1)
"""
    run_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{smoke_npz}:/tmp/smoke.npz:ro",
        IMAGE_TAG,
        "python",
        "-c",
        smoke_code,
    ]
    print(f"[docker] running container smoke test against {IMAGE_TAG}")
    proc = subprocess.run(run_cmd, capture_output=True, text=True)
    if proc.returncode != 0 or "CONTAINER_SMOKE_PASS" not in proc.stdout:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"container smoke failed with exit code {proc.returncode}")
    print(proc.stdout.strip())

    candidate = {
        "arm": "h1_rift_r300_flat_e15_cached",
        "task": "h1",
        "image_tag": IMAGE_TAG,
        "image_id": image_id,
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "method_label": METHOD_LABEL,
        "method_name": METHOD_NAME,
        "method_description": METHOD_DESCRIPTION,
        "budget_disclosure": "Falcon H1 standard evaluation protocol: 27 official dataset tags, 3 trial calibration budget per test session.",
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "selection": selection,
        "host_verification": host_report,
        "container_smoke": {
            "status": "PASS",
            "output": proc.stdout.strip(),
        },
    }
    candidate_path = DEST / "artifacts" / "evalai_candidate.json"
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    print(f"[candidate] wrote {candidate_path}")
    return candidate


def main() -> None:
    print("[1/4] Copying pkg tree...")
    pkg = _copy_pkg()
    print("[2/4] Building sealed payload...")
    payload_path, payload_sha, selection = _build_payload()
    print(f"      payload: {payload_path} ({payload_sha})")
    print(f"      selection: epoch={selection['epoch']} ho_m3_r2={selection['mean']:.4f} worst={selection['worst']:.4f}")
    print("[3/4] Running host parity verification (B1 and B8)...")
    host_report = _host_verify(payload_path)
    for b in ("B1", "B8"):
        print(f"      {b}: max_abs={host_report[b]['max_abs']:.3e}")
    print("[4/4] Building Docker image and running container smoke test...")
    candidate = _docker_build_and_smoke(payload_path, payload_sha, selection, host_report)
    print("\nSUCCESS! Ready to submit via submit.py:")
    print(f"  python submit.py --manifest {DEST / 'artifacts' / 'evalai_candidate.json'} --dry-run")
    print(f"  python submit.py --manifest {DEST / 'artifacts' / 'evalai_candidate.json'} --execute --confirm-image-id {candidate['image_id']} --confirm-payload-sha256 {payload_sha}")


if __name__ == "__main__":
    main()
