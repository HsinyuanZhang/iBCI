#!/usr/bin/env python3
"""Pack B3S concat-route M1/M2 learned+full / flat EvalAI images.

Calibrate E0 with ConcatSideTrunk, then strip trunk keys and serve the
existing Falcon proj_add RiftDecoder cached runtime. Does not submit.
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
from typing import Any

import numpy as np
import torch

ROOT = Path("/home/xinyuan/Work_host/SPINT")
SUB = ROOT / "tfpd_exploration/submissions"
LR = ROOT / "btransform_unified_v2/learnable_recency_v1"
SCRIPTS = LR / "scripts"
RESULTS = LR / "results"
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
LEARNABLE_SRC = LR / "src/learnable_recency_v1"
M2_BANK = SUB / "evalai_m2_small_concat_ort_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
M1_BANK = SUB / "evalai_m1_rift_r100_projadd_muscle_flat_e2_v1/artifacts/m1_rift_r100_projadd_muscle_flat.pkl"
M2_CACHE = ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train"
CARRIER_SHA = "68dc80c8734f4310c0ec378f599fd041a439906e90973db4351d582c9a30887e"
N_STEPS = 80
GATE = 1.0e-5
DROP_PREFIX = ("identity_encoder.",)
DROP_EXACT = {"zero_carrier"}
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

ARMS = {
    "m2_full": {
        "task": "m2",
        "dest": SUB / "evalai_m2_rift_r50_concat_full_e8_v1",
        "run": RESULTS / "m2_concat_full_stage2_s42",
        "score": RESULTS / "selection_m2_concat_full_stage2_s42_ext6/score_receipt.json",
        "epoch": 8,
        "table_mean": 0.3832,
        "tier": "learned_slope",
        "payload_name": "m2_rift_r50_concat_full_e8.pkl",
        "payload_schema": "m2_rift_r50_concat_full_e8_cached_v1",
        "image_tag": "m2-rift-r50-concat-full-e8:v1",
        "arm": "m2_rift_r50_concat_full_e8_cached",
        "method_name": "M2 RIFT R50 concat-full e8",
        "method_label": (
            "M2 RIFT R50 concat-side E0 + proj_add P16 learned_slope default-ladder e8 EMA cached CPU; "
            "architecture=RIFT backend=cached identity=proj_add e0_source=concat_side_trunk; "
            "EXT6 pick; not BT-EORT; not ORT; not SPINT; not 582189; not 582240"
        ),
    },
    "m2_flat": {
        "task": "m2",
        "dest": SUB / "evalai_m2_rift_r50_concat_flat_e7_v1",
        "run": RESULTS / "m2_concat_flat_stage2_s42",
        "score": RESULTS / "selection_m2_concat_flat_stage2_s42_ext6/score_receipt.json",
        "epoch": 7,
        "table_mean": 0.3622,
        "tier": "fixed",
        "payload_name": "m2_rift_r50_concat_flat_e7.pkl",
        "payload_schema": "m2_rift_r50_concat_flat_e7_cached_v1",
        "image_tag": "m2-rift-r50-concat-flat-e7:v1",
        "arm": "m2_rift_r50_concat_flat_e7_cached",
        "method_name": "M2 RIFT R50 concat-flat e7",
        "method_label": (
            "M2 RIFT R50 concat-side E0 + proj_add P16 flat zero-slope e7 EMA cached CPU; "
            "architecture=RIFT backend=cached identity=proj_add e0_source=concat_side_trunk; "
            "EXT6 pick; not BT-EORT; not ORT; not SPINT; not 582189; not 582292"
        ),
    },
    "m1_full": {
        "task": "m1",
        "dest": SUB / "evalai_m1_rift_r100_concat_full_e2_v1",
        "run": RESULTS / "m1_concat_full_stage2_s42",
        "score": RESULTS / "selection_m1_concat_full_stage2_s42_ho3/score_receipt.json",
        "epoch": 2,
        "table_mean": 0.7124,
        "tier": "learned_slope",
        "payload_name": "m1_rift_r100_concat_full_e2.pkl",
        "payload_schema": "m1_rift_r100_concat_full_e2_cached_v1",
        "image_tag": "m1-rift-r100-concat-full-e2:v1",
        "arm": "m1_rift_r100_concat_full_e2_cached",
        "method_name": "M1 RIFT R100 concat-full e2",
        "method_label": (
            "M1 RIFT R100 concat-side E0 + proj_add P16 muscle learned_slope e2 EMA cached CPU; "
            "architecture=RIFT backend=cached identity=proj_add e0_source=concat_side_trunk "
            "carrier=muscle_response16_svd4/global_rms"
        ),
    },
    "m1_flat": {
        "task": "m1",
        "dest": SUB / "evalai_m1_rift_r100_concat_flat_e2_v1",
        "run": RESULTS / "m1_concat_flat_stage2_s42",
        "score": RESULTS / "selection_m1_concat_flat_stage2_s42_ho3/score_receipt.json",
        "epoch": 2,
        "table_mean": 0.7195,
        "tier": "fixed",
        "payload_name": "m1_rift_r100_concat_flat_e2.pkl",
        "payload_schema": "m1_rift_r100_concat_flat_e2_cached_v1",
        "image_tag": "m1-rift-r100-concat-flat-e2:v1",
        "arm": "m1_rift_r100_concat_flat_e2_cached",
        "method_name": "M1 RIFT R100 concat-flat e2",
        "method_label": (
            "M1 RIFT R100 concat-side E0 + proj_add P16 muscle flat zero-slope e2 EMA cached CPU; "
            "architecture=RIFT backend=cached identity=proj_add e0_source=concat_side_trunk "
            "carrier=muscle_response16_svd4/global_rms"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    x = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(x.dtype.str.encode())
    digest.update(str(x.shape).encode())
    digest.update(x.tobytes())
    return digest.hexdigest()


def _use_paths(dest: Path, *, packed: bool = False) -> None:
    paths = [
        str(SCRIPTS),
        str(LR / "src"),
        str(V2_SRC),
        str(V1_SRC),
        str(ROOT),
        str(dest),
    ]
    if packed:
        paths.insert(0, str(dest / "artifacts" / "pkg"))
    for path in reversed(paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = _CPUUnpickler(handle).load()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} is not a dict payload")
    return payload


def m2_tag(session: str) -> str:
    body = session.removeprefix("ses-")
    date, run = body.rsplit("-", 1)
    year, month, day = date.split("-")
    return f"{run}_{year}{month}{day}"


def m2_session(tag: str) -> str:
    run, ymd = tag.split("_")
    return f"ses-{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}-{run}"


def m1_tag(session: str) -> str:
    return session.removeprefix("ses-")


def _copy_pkg(dest: Path) -> Path:
    pkg = dest / "artifacts" / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    for src, name in ((V1_SRC / "btransform_unified_v1", "btransform_unified_v1"), (V2_SRC / "btransform_unified_v2", "btransform_unified_v2")):
        dst = pkg / name
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*.py"):
            if item.name in SKIP_PKG:
                continue
            target = dst / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    (pkg / "btransform_unified_v2" / "__init__.py").write_text(SAFE_INIT)
    learnable = pkg / "learnable_recency_v1"
    learnable.mkdir(parents=True, exist_ok=True)
    for name in ("config.py", "temporal.py", "cpu_temporal.py"):
        shutil.copy2(LEARNABLE_SRC / name, learnable / name)
    (learnable / "__init__.py").write_text(LEARNABLE_SAFE_INIT)
    return pkg


def _strip_trunk(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    kept = {}
    dropped = []
    for key, value in state.items():
        if key in DROP_EXACT or key.startswith(DROP_PREFIX):
            dropped.append(key)
            continue
        kept[key] = value.detach().cpu().float().clone()
    if len(dropped) < 9:
        raise RuntimeError(f"expected at least 9 concat-trunk keys, dropped {dropped}")
    return kept


def _build_payload(name: str) -> tuple[Path, str, dict[str, Any], dict[str, np.ndarray]]:
    spec = ARMS[name]
    dest: Path = spec["dest"]
    run: Path = spec["run"]
    task = spec["task"]
    epoch = int(spec["epoch"])
    _use_paths(dest, packed=False)
    import activity_data
    import activity_full_score as score_mod
    import activity_full_train as train
    from learnable_recency_v1.config import config_from_run_meta

    meta = json.loads((run / "run_meta.json").read_text())
    score = json.loads(spec["score"].read_text())
    selected = score["selection"]
    if int(selected["epoch"]) != epoch:
        raise RuntimeError(f"{name}: score receipt epoch {selected['epoch']} != {epoch}")
    curve = score["ema_by_epoch"][str(epoch)]
    mean = float(curve["equal_session_mean"])
    if abs(mean - float(spec["table_mean"])) > 5.0e-4:
        raise RuntimeError(f"{name}: local mean {mean} drifted from table {spec['table_mean']}")
    if meta.get("schema") != train.SCHEMA or meta.get("identity_interface") != "concat_route":
        raise RuntimeError(f"{name}: run_meta is not B3S concat_route")
    if str(meta.get("trunk_side")) != "concat" or str(meta.get("stage")) != "2_b3s":
        raise RuntimeError(f"{name}: run is not concat stage-2")
    recency_cfg = config_from_run_meta(meta, task)
    if recency_cfg.ladder != "default" or recency_cfg.tier != spec["tier"] or not recency_cfg.per_layer:
        raise RuntimeError(f"{name}: recency config drifted: {recency_cfg}")
    if spec["tier"] == "fixed" and any(half is not None for half in recency_cfg.half_life_seconds):
        raise RuntimeError(f"{name}: flat pack requires all half-lives None")
    ckpt_path = run / f"epoch_{epoch:03d}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != train.CKPT_SCHEMA or int(ckpt.get("epoch", -1)) != epoch:
        raise RuntimeError(f"{name}: checkpoint contract mismatch")
    if str(ckpt.get("stage")) != "2_b3s" or str(ckpt.get("trunk_side")) != "concat":
        raise RuntimeError(f"{name}: checkpoint is not concat stage-2")

    if task == "m2":
        train.resolve_m2_ext6_root()
    data = activity_data.load_task_data(task, include_eval=True)
    surface = {**data.get("train", {}), **data.get("evaluation", {})}
    if task == "m2" and set(m2_tag(s) for s in surface) != set(_load_pickle(M2_BANK)["bank_by_dataset_tag"]):
        raise RuntimeError("M2 activity sessions do not cover the official 13 tags")
    if task == "m1" and {m1_tag(s) for s in surface} != set(_load_pickle(M1_BANK)["bank_by_dataset_tag"]):
        raise RuntimeError("M1 activity sessions do not cover the official 7 tags")
    carriers, _binding = train.load_task_carriers(
        task,
        train_sessions=sorted(data.get("train", {})),
        eval_sessions=sorted(data.get("evaluation", {})),
        m1_pack=Path(meta.get("carrier", {}).get("binding", {}).get("carrier_pack_npz", train.M1_CARRIER_PACK_DEFAULT)),
    )
    recorded = meta.get("carrier", {}).get("sha256_per_session") or {}
    for session in sorted(data.get("train", {})):
        if session in recorded and recorded[session] != train._ah(carriers[session]):
            raise RuntimeError(f"{name}: train carrier bytes drifted: {session}")

    model = train.B3SFullRiftDecoder(
        task,
        recency_cfg,
        context_bins=int(meta["context_bins"]),
        seed=int(meta["seed"]),
        support_bins=int(meta["support_bins"]),
        identity_hidden=int(meta["identity_hidden"]),
        trunk_side="concat",
    )
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    model.freeze_trunk()
    if model.trunk_parameter_sha256() != meta["trunk"]["source"]["trunk_weights_sha256"]:
        raise RuntimeError(f"{name}: frozen trunk bytes differ from run_meta")
    score_mod._ema_into(model, ckpt)
    model.eval()
    if spec["tier"] == "fixed":
        slopes = model.temporal.recency_slopes.detach().cpu().float()
        if slopes.shape != (8,) or float(slopes.abs().max()) > 0.0:
            raise RuntimeError(f"{name}: flat EMA slopes are not exact zeros: {slopes.tolist()}")

    old_banks = _load_pickle(M2_BANK if task == "m2" else M1_BANK)["bank_by_dataset_tag"]
    sealed: dict[str, dict[str, Any]] = {}
    traces: dict[str, np.ndarray] = {}
    context = int(data["metadata"]["context"])
    for session, item in surface.items():
        tag = m2_tag(session) if task == "m2" else m1_tag(session)
        old = old_banks[tag]
        carrier_np = np.ascontiguousarray(carriers[session], dtype=np.float32)
        old_t = np.ascontiguousarray(old["T"], dtype=np.float32)
        if carrier_np.shape != old_t.shape or not np.array_equal(carrier_np, old_t):
            raise RuntimeError(f"{name}: official T drift for {session}/{tag}")
        activity, trial_mask = train._support(item, torch.device("cpu"))
        identity = model.calibrate(activity, trial_mask, carrier=torch.as_tensor(carrier_np))
        e0 = np.ascontiguousarray(identity.detach().cpu().numpy(), dtype=np.float32)
        mask = np.ascontiguousarray(old["unit_mask"], dtype=np.bool_)
        sealed[tag] = {
            "E0": e0,
            "T": carrier_np,
            "unit_mask": mask,
            "session": session,
            "e0_sha256": array_sha(e0),
            "t_sha256": array_sha(carrier_np),
        }
        raw = np.ascontiguousarray(item["X"], dtype=np.float32)
        traces[tag] = raw[context - 1 : context - 1 + N_STEPS]

    ema_state = _strip_trunk({k: v for k, v in model.state_dict().items()})
    payload = {
        "schema": spec["payload_schema"],
        "task": task,
        "context_bins": int(meta["context_bins"]),
        "bias_mode": "recency" if spec["tier"] == "fixed" else f"learnable_{spec['tier']}",
        "identity_interface": "proj_add",
        "e0_source": "concat_side_trunk",
        "train_identity_interface": "concat_route",
        "proj_dim": 16,
        "tier": spec["tier"],
        "ladder": "default",
        "per_layer": True,
        "seed": int(meta["seed"]),
        "behavior_scaling_factor": 5.0 if task == "m2" else 1.0,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256(ckpt_path),
        "learnable_config": dict(meta["learnable_config"]),
        "bank_by_dataset_tag": sealed,
        "ema_state_dict": ema_state,
        "selection": {
            "surface": "ext6 development" if task == "m2" else "visible HO3",
            "epoch": epoch,
            "equal_session_mean": mean,
            "official_test_used": False,
        },
    }
    if task == "m1":
        payload["carrier_variant"] = "muscle_response16_svd4/global_rms"
        payload["carrier_pack_sha256"] = CARRIER_SHA
    dest_pkl = dest / "artifacts" / spec["payload_name"]
    dest_pkl.parent.mkdir(parents=True, exist_ok=True)
    if dest_pkl.exists():
        dest_pkl.unlink()
    with dest_pkl.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return dest_pkl, sha256(dest_pkl), payload["selection"], traces


def _host_verify(name: str, payload_path: Path, traces: dict[str, np.ndarray]) -> dict[str, Any]:
    spec = ARMS[name]
    dest: Path = spec["dest"]
    task = spec["task"]
    for key in list(sys.modules):
        if key in {"m1_rift_falcon_decoder", "m2_rift_falcon_decoder"} or key.startswith(
            ("m1_rift_falcon_decoder.", "m2_rift_falcon_decoder.")
        ):
            del sys.modules[key]
    _use_paths(dest, packed=True)
    from falcon_challenge.config import FalconConfig, FalconTask

    if task == "m2":
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
        recency_cfg = recency_config_from_payload(payload)
        decoder_cls = M2RiftCachedFalconDecoder
        scale = 5.0
        plans = (
            ("B1", ["Run1_20201019"]),
            ("B7", [
                "Run1_20201019",
                "Run2_20201019",
                "Run1_20201020",
                "Run2_20201020",
                "Run1_20201027",
                "Run2_20201027",
                "Run1_20201028",
            ]),
        )
        smoke_tag = "Run1_20201019"
    else:
        from m1_rift_falcon_decoder import (
            CpuLearnableRiftRuntime,
            M1RiftCachedFalconDecoder,
            _task_bank,
            build_decoder,
            load_payload,
            recency_config_from_payload,
        )

        def _m1_handle(tag: str) -> str:
            kind = "held-in-calib" if tag.startswith("201209") else "held-out-calib"
            return f"sub-MonkeyL-{kind}_ses-{tag}_behavior+ecephys"

        payload = load_payload(payload_path)
        config = FalconConfig(task=FalconTask.m1)
        recency_cfg = recency_config_from_payload(payload)
        decoder_cls = M1RiftCachedFalconDecoder
        scale = 1.0
        plans = (
            ("B1", ["20121004"]),
            ("B3", ["20121004", "20121017", "20121024"]),
            ("B4", ["20120924", "20120926", "20120927", "20120928"]),
        )
        smoke_tag = "20121004"

    report: dict[str, Any] = {}
    for plan_name, tags in plans:
        packed = decoder_cls(task_config=config, model_path=str(payload_path), batch_size=len(tags))
        reset_tags = tags if task == "m2" else [_m1_handle(tag) for tag in tags]
        packed.reset(dataset_tags=reset_tags)
        if task == "m2":
            ref_model = build_decoder(recency_cfg)
        else:
            ref_model = build_decoder(recency_cfg, int(payload["seed"]))
        ref_model.load_state_dict({k: torch.as_tensor(v) for k, v in payload["ema_state_dict"].items()}, strict=True)
        ref_model.eval()
        banks = [_task_bank(tag, payload["bank_by_dataset_tag"][tag]) for tag in tags]
        ref = CpuLearnableRiftRuntime(ref_model, banks, [str(tag) for tag in tags])
        max_abs = 0.0
        last = None
        for step in range(N_STEPS):
            batch_x = np.stack([traces[tag][step] for tag in tags], axis=0)
            got = packed.predict(batch_x)
            want = ref.advance(torch.from_numpy(batch_x)).numpy() / scale
            last = got
            max_abs = max(max_abs, float(np.max(np.abs(got - want))))
            if max_abs > GATE:
                raise RuntimeError(f"{name} {plan_name} host gate fail t={step} max_abs={max_abs}")
        report[plan_name] = {"steps": N_STEPS, "max_abs": max_abs, "last_shape": list(last.shape), "sessions": tags}

    smoke = dest / "artifacts" / "smoke_window.npz"
    neural = np.ascontiguousarray(traces[smoke_tag][:40], dtype=np.float32)
    packed = decoder_cls(task_config=config, model_path=str(payload_path), batch_size=1)
    packed.reset(dataset_tags=[smoke_tag if task == "m2" else _m1_handle(smoke_tag)])
    pred = None
    for row in neural:
        pred = packed.predict(row.reshape(1, -1))
    np.savez(smoke, tag_stem=np.asarray(smoke_tag), window=neural, expected=pred)
    report["smoke_window"] = str(smoke)
    report["status"] = "HOST_PACK_VERIFY_PASS"
    return report


def _docker_build(spec: dict[str, Any], payload_sha: str) -> dict[str, Any]:
    dest: Path = spec["dest"]
    cmd = [
        "docker",
        "build",
        "--build-arg",
        f"PAYLOAD_SHA256={payload_sha}",
        "--build-arg",
        f"METHOD_LABEL={spec['method_label']}",
        "-t",
        spec["image_tag"],
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    inspect = subprocess.check_output(
        ["docker", "image", "inspect", spec["image_tag"], "--format", "{{.Id}} {{.Size}}"],
        text=True,
    ).strip()
    image_id, size = inspect.split()
    return {"image_tag": spec["image_tag"], "image_id": image_id, "image_size": int(size)}


def pack_one(name: str) -> dict[str, Any]:
    spec = ARMS[name]
    dest: Path = spec["dest"]
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _copy_pkg(dest)
    payload_path, payload_sha, selection, traces = _build_payload(name)
    host = _host_verify(name, payload_path, traces)
    docker = _docker_build(spec, payload_sha)
    if spec["task"] == "m2":
        budget = (
            "Official 13-tag M2 banks (ConcatSideTrunk E0 + sealed MOVE-T4). "
            f"RIFT R50 seed42 EMA e{selection['epoch']} after B3S concat stage-2. "
            f"ext6 earliest-max equal-session {selection['equal_session_mean']:.6f} is development selection, "
            "not official HO. Cached CPU learnable runtime, no TTA, no ORT."
        )
        description = (
            "M2 RIFT (Recency-biased Incremental Finite-context Transformer) R50, "
            "D4/width256, concat-side trunk E0 plus proj_add P16 token fusion, "
            f"{'learned recency slopes' if spec['tier'] == 'learned_slope' else 'flat zero-slope control'}, "
            "cached CPU KV. "
            f"ext6 pick e{selection['epoch']} mean {selection['equal_session_mean']:.6f}. "
            "Not BT-EORT, not ORT, not SPINT."
        )
    else:
        budget = (
            "Official 7-tag M1 banks (ConcatSideTrunk E0 + muscle_response16_svd4/global_rms T). "
            f"RIFT R100 seed42 EMA e{selection['epoch']} after B3S concat stage-2. "
            f"HO3 earliest-max equal-session {selection['equal_session_mean']:.6f} is development selection, "
            "not official HO. Cached CPU learnable runtime, no TTA, no ORT."
        )
        description = (
            "M1 RIFT R100, concat-side trunk E0 plus proj_add P16 token fusion, "
            "muscle_response16_svd4/global_rms carrier, "
            f"{'learned recency slopes' if spec['tier'] == 'learned_slope' else 'flat zero-slope control'}, "
            "cached CPU KV. "
            f"HO3 pick e{selection['epoch']} mean {selection['equal_session_mean']:.6f}. "
            "Not BT-EORT, not ORT, not SPINT."
        )
    candidate = {
        "arm": spec["arm"],
        "budget_disclosure": budget,
        "evalai_opened": True,
        "image_id": docker["image_id"],
        "image_size": docker["image_size"],
        "image_tag": docker["image_tag"],
        "is_public": True,
        "method_description": description,
        "method_label": spec["method_label"],
        "method_name": spec["method_name"],
        "payload_sha256": payload_sha,
        "register": False,
        "selection_mean": selection["equal_session_mean"],
        "state_path": str(dest / "artifacts" / "evalai_push_state.json"),
        "window": 50 if spec["task"] == "m2" else 100,
        "host_verify": host,
        "team": "HKU-ECE",
        "team_id": 41975,
    }
    (dest / "artifacts" / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return {
        "status": "PACKED_NOT_REGISTERED",
        "arm": name,
        "image_tag": docker["image_tag"],
        "image_id": docker["image_id"],
        "payload_sha256": payload_sha,
        "image_size": docker["image_size"],
        "selection_mean": selection["equal_session_mean"],
        "host_verify": {k: host[k] for k in host if k != "smoke_window"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", choices=sorted(ARMS) + ["all"])
    args = parser.parse_args()
    names = list(ARMS) if "all" in args.arms else args.arms
    reports = [pack_one(name) for name in names]
    print(json.dumps(reports, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
