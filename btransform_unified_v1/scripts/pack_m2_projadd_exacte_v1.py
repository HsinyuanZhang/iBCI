#!/usr/bin/env python3
"""Pack the M2 proj_add ext6 pick into an exact-E Docker image. Does not register."""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from falcon_challenge.config import FalconConfig, FalconTask

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import plan
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity
from tfpd_exploration.src.m2_dual_track_v1.champion import tensor_state_sha256
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime import PAYLOAD_SCHEMA, constants as C
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.banks import bank_receipt, collect_official_banks
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.load_weights import export_state_numpy
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.payload import sha256_file

PICK = PACKAGE_ROOT / "results/m2_projadd/20260906_104300/ext6_epoch_pick/selection.json"
DEST = WORKSPACE_ROOT / "tfpd_exploration/submissions/evalai_m2_projadd_exacte_v1"
GEOMETRY = {
    "task": "m2",
    "window": 50,
    "prefix": 0,
    "units": 96,
    "e0_dim": 50,
    "carrier_dim": 4,
    "out_dim": 2,
}


def _apply_ema(model: BTransformerUnifiedDecoderIdentity, ckpt: dict) -> None:
    model.load_state_dict(ckpt["raw_state_dict"])
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    plan.require(set(named) == set(shadow), "EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))


def main() -> dict:
    plan.require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    pick = json.loads(PICK.read_text(encoding="utf-8"))
    plan.require(pick.get("evalai_opened") is False, "pick contacted EvalAI")
    selected = pick["selected"]
    epoch = int(selected["epoch"])
    ckpt_path = Path(selected["ckpt"])
    plan.require(ckpt_path.is_file(), f"missing {ckpt_path}")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts").mkdir(parents=True, exist_ok=True)
    payload_name = f"m2_projadd_s42_ema_e{epoch:02d}_ext6.pkl"
    payload_path = DEST / "artifacts" / payload_name
    plan.require(not payload_path.exists(), f"refusing to overwrite {payload_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BTransformerUnifiedDecoderIdentity(GEOMETRY, seed=42, identity_mode="proj_add")
    _apply_ema(model, ckpt)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weight_sha = tensor_state_sha256({k: v.detach() for k, v in model.state_dict().items()})
    banks = collect_official_banks()
    config = FalconConfig(task=FalconTask.m2)
    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "task": config.task,
        "kind": "proj_add",
        "state_dict": export_state_numpy(model),
        "bank_by_dataset_tag": {
            tag: {
                "E0": np.ascontiguousarray(row["E0"], dtype=np.float32),
                "T": np.ascontiguousarray(row["T"], dtype=np.float32),
                "unit_mask": np.ascontiguousarray(row["unit_mask"], dtype=np.bool_),
            }
            for tag, row in banks.items()
        },
        "window_size": C.WINDOW,
        "behavior_scaling_factor": C.BEHAVIOR_SCALE,
        "smooth_observations": False,
        "metadata": {
            "kind": "proj_add",
            "cell": "M2-PROJADD-V1",
            "seed": 42,
            "epoch": epoch,
            "view": "EMA",
            "ckpt": str(ckpt_path),
            "weight_sha256": weight_sha,
            "selection_surface": pick["selection_surface"],
            "selection_mean": selected["equal_session_mean"],
            "label_budget": 33,
            "activity_budget": 33,
            "online_kv_cache": False,
            "old_spint_decoder": False,
            "tta": False,
            "exact_e": True,
            "dataloader_workers": 0,
            "official_tags": list(C.OFFICIAL_TAGS),
            "evalai_opened": False,
        },
    }
    with payload_path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    payload_sha = sha256_file(payload_path)

    from tfpd_exploration.src.m2_dual_track_v1 import data as dual_data, plan as dual_plan
    from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.banks import session_tag_map
    from btransform_unified_v1.adapters import build_m2_bank

    session = dual_plan.HELDIN_SESSIONS[0]
    tag = session_tag_map()[session]
    date = tag.split("_", 1)[1]
    run = tag.split("_", 1)[0]
    stem = f"sub-MonkeyN{run}_{date}_held_in_eval"
    bank = dual_data.load_session_bank("source_minival", session, device="cpu")
    store = np.asarray(bank.X_store, dtype=np.float32)
    if store.ndim == 3:
        store = store.reshape(-1, store.shape[-1])
    window = np.ascontiguousarray(store[:50], dtype=np.float32)
    task_bank = build_m2_bank("source_minival", session, budget=33)
    x = torch.from_numpy(window[None])
    with torch.inference_mode():
        train_pred = (model(x, task_bank) / C.BEHAVIOR_SCALE).detach().cpu().numpy()

    sys.path.insert(0, str(DEST))
    from trf_falcon_decoder import TrfFalconDecoder

    decoder = TrfFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    decoder.reset(dataset_tags=[stem])
    pred = None
    for row in window:
        pred = decoder.predict(row.reshape(1, -1))
    delta = float(np.max(np.abs(pred - train_pred)))
    plan.require(np.isfinite(pred).all(), "non-finite exact-E pred")
    plan.require(delta <= 1.0e-5, f"exact-E vs train forward max|Δ|={delta}")

    smoke_window = DEST / "artifacts" / "smoke_window.npz"
    np.savez(smoke_window, tag_stem=np.asarray(stem), window=window, expected=np.asarray(pred))

    (DEST / "artifacts" / "payload.receipt.json").write_text(
        json.dumps(
            {
                "schema": "m2_projadd_exacte_payload_v1",
                "payload_path": str(payload_path),
                "payload_sha256": payload_sha,
                "bytes": payload_path.stat().st_size,
                "weight_sha256": weight_sha,
                "selected": selected,
                "parity_exact_e_vs_train_max_abs": delta,
                "banks": bank_receipt(banks),
                "evalai_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (DEST / ".dockerignore").write_text(
        "**\n!.dockerignore\n!Dockerfile\n!decode.py\n!trf_falcon_decoder.py\n"
        f"!artifacts/\n!artifacts/{payload_name}\n",
        encoding="utf-8",
    )
    method_name = f"M2 proj_add Transformer EMA e{epoch} s42 ext6-epochpick exact-E w0"
    method_label = (
        f"M2 proj_add CausalPE4 (seed42 EMA e{epoch}) exact-E w0; "
        "6-session official held-out-calib epoch-pick"
    )
    image_tag = f"spint-t4-m2:projadd-s42-ema-e{epoch}-ext6-w0-{payload_sha[:8]}"
    (DEST / "Dockerfile").write_text(
        f"ARG BASE_IMAGE={C.BASE_IMAGE}\n"
        "FROM ${BASE_IMAGE}\n"
        f"ARG PAYLOAD_SHA256={payload_sha}\n"
        f'LABEL ai.eval.method="{method_label}"\n'
        'LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"\n'
        'LABEL ai.eval.label_budget="33"\n'
        'LABEL ai.eval.activity_budget="33"\n'
        f'LABEL ai.eval.candidate="projadd_s42_ema_e{epoch}_ext6_w0"\n'
        'LABEL ai.eval.old_spint_decoder="false"\n'
        'LABEL ai.eval.exact_e="true"\n'
        'LABEL ai.eval.dataloader_workers="0"\n'
        f"COPY artifacts/{payload_name} /data/decoder.pkl\n"
        "COPY trf_falcon_decoder.py /trf_falcon_decoder.py\n"
        "COPY decode.py /decode.py\n"
        "ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7 "
        "PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
        'CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC '
        '--model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]\n',
        encoding="utf-8",
    )

    subprocess.run(
        ["docker", "build", "-t", image_tag, "--build-arg", f"PAYLOAD_SHA256={payload_sha}", str(DEST)],
        check=True,
    )
    inspect = subprocess.check_output(
        ["docker", "image", "inspect", image_tag, "--format", "{{.Id}} {{.Size}}"],
        text=True,
    ).strip()
    image_id, size = inspect.split()
    smoke_out = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-v",
            f"{payload_path.resolve()}:/data/decoder.pkl:ro",
            "-v",
            f"{smoke_window.resolve()}:/data/smoke_window.npz:ro",
            "-v",
            f"{(DEST / 'trf_falcon_decoder.py').resolve()}:/trf_falcon_decoder.py:ro",
            image_tag,
            "python",
            "/trf_falcon_decoder.py",
            "--smoke-payload",
            "/data/decoder.pkl",
            "--smoke-window",
            "/data/smoke_window.npz",
        ],
        text=True,
    )
    last = [line for line in smoke_out.splitlines() if line.strip()][-1]
    smoke_report = json.loads(last)
    plan.require(smoke_report.get("status") == "CONTAINER_SMOKE_PASS", f"smoke failed: {smoke_out}")

    candidate = {
        "arm": f"projadd_s42_ema_e{epoch}_ext6_w0",
        "image_id": image_id,
        "image_tag": image_tag,
        "image_size": int(size),
        "method_name": method_name,
        "method_label": method_label,
        "payload_sha256": payload_sha,
        "parity_exact_e_vs_train_max_abs": delta,
        "selection_mean": selected["equal_session_mean"],
        "register": False,
        "evalai_opened": False,
    }
    (DEST / "artifacts" / "evalai_candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (DEST / "artifacts" / "evalai_push_state.json").write_text(
        json.dumps(
            {
                "arm": candidate["arm"],
                "image_tag": image_tag,
                "payload_sha256": payload_sha,
                "schema_version": "m2_trf_evalai_push_state",
                "register": False,
                "created": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PACKED_NOT_REGISTERED", "candidate": candidate}, indent=2), flush=True)
    return candidate


if __name__ == "__main__":
    main()
