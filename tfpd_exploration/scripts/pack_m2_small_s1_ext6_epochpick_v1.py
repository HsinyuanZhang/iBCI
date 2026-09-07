#!/usr/bin/env python3
"""Pack the M2 ext6 epoch-pick into an exact-E w0 image. Does not register."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from falcon_challenge.config import FalconConfig, FalconTask

from tfpd_exploration.src.m2_b_small_stability_v1.decoder import SmallTransformerDecoder
from tfpd_exploration.src.m2_b_small_stability_v1.score import apply_view
from tfpd_exploration.src.m2_dual_track_v1.champion import tensor_state_sha256
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime import PAYLOAD_SCHEMA
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime import constants as C
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.banks import bank_receipt, collect_official_banks
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.load_weights import export_state_numpy
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.payload import sha256_file


ROOT = Path("/home/xinyuan/Work_host/SPINT")
PICK = ROOT / "tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/selection.json"
SRC = ROOT / "tfpd_exploration/submissions/evalai_m2_small_trf_e_opt_v1"
DEST = ROOT / "tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> dict:
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    pick = json.loads(PICK.read_text(encoding="utf-8"))
    require(pick["status"] == "SELECTED_BEFORE_EVALAI", "pick is not sealed")
    require(pick["evalai_opened"] is False, "pick contacted EvalAI")
    selected = pick["selected"]
    seed = int(selected["seed"])
    epoch = int(selected["epoch_one_based"])
    ckpt_path = Path(selected["ckpt"])
    require(ckpt_path.is_file(), f"missing selected ckpt {ckpt_path}")
    require(sha256_file(ckpt_path) == selected["ckpt_sha256"], "selected ckpt drift")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts").mkdir(parents=True, exist_ok=True)
    payload_name = f"m2_small_trf_s{seed}_ema_e{epoch:02d}_ext6.pkl"
    payload_path = DEST / "artifacts" / payload_name
    require(not payload_path.exists(), f"refusing to overwrite {payload_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SmallTransformerDecoder(seed=seed)
    apply_view(model, ckpt, "EMA")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weight_sha = tensor_state_sha256({k: v.detach() for k, v in model.state_dict().items()})
    banks = collect_official_banks()
    config = FalconConfig(task=FalconTask.m2)
    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "task": config.task,
        "kind": "small",
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
            "kind": "small",
            "cell": "S1-SMALL-COS",
            "seed": seed,
            "epoch": epoch,
            "view": "EMA",
            "ckpt": str(ckpt_path),
            "weight_sha256": weight_sha,
            "selection_surface": pick["selection_surface"],
            "selection_mean": selected["external_equal_session_mean"],
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
    require(payload_sha not in C.KNOWN_PAYLOAD_SHA256, f"payload digest collides: {payload_sha}")
    receipt = {
        "schema": "m2_small_s1_ext6_epochpick_payload_v1",
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "bytes": payload_path.stat().st_size,
        "weight_sha256": weight_sha,
        "selected": selected,
        "banks": bank_receipt(banks),
        "evalai_opened": False,
    }
    (DEST / "artifacts" / "payload.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    shutil.copy2(SRC / "trf_falcon_decoder.py", DEST / "trf_falcon_decoder.py")
    shutil.copy2(SRC / "decode.py", DEST / "decode.py")
    (DEST / ".dockerignore").write_text(
        "**\n!.dockerignore\n!Dockerfile\n!decode.py\n!trf_falcon_decoder.py\n"
        f"!artifacts/\n!artifacts/{payload_name}\n",
        encoding="utf-8",
    )
    method_name = (
        f"M2 small Transformer S1-SMALL-COS EMA e{epoch} s{seed} ext6-epochpick exact-E w0"
    )
    method_label = (
        f"M2 small causal Transformer (S1-SMALL-COS seed{seed} EMA e{epoch}) "
        "exact-E w0; 6-session official held-out-calib epoch-pick; not 581971"
    )
    image_tag = f"spint-t4-m2:small-trf-s{seed}-ema-e{epoch}-ext6-w0-{payload_sha[:8]}"
    (DEST / "Dockerfile").write_text(
        f"ARG BASE_IMAGE={C.BASE_IMAGE}\n"
        "FROM ${BASE_IMAGE}\n"
        f"ARG PAYLOAD_SHA256={payload_sha}\n"
        f"ARG CHECKPOINT_SHA256={C.CHAMPION_CKPT_SHA256}\n"
        f'LABEL ai.eval.method="{method_label}"\n'
        'LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"\n'
        'LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"\n'
        'LABEL ai.eval.label_budget="33"\n'
        'LABEL ai.eval.activity_budget="33"\n'
        f'LABEL ai.eval.candidate="small_trf_s{seed}_ema_e{epoch}_ext6_w0"\n'
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

    smoke_window = DEST / "artifacts" / "smoke_window.npz"
    from tfpd_exploration.src.m2_dual_track_v1 import data as dual_data, plan
    from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.banks import session_tag_map

    session = plan.HELDIN_SESSIONS[0]
    tag = session_tag_map()[session]
    date = tag.split("_", 1)[1]
    run = tag.split("_", 1)[0]
    stem = f"sub-MonkeyN{run}_{date}_held_in_eval"
    bank = dual_data.load_session_bank("source_minival", session, device="cpu")
    store = np.asarray(bank.X_store, dtype=np.float32)
    if store.ndim == 3:
        store = store.reshape(-1, store.shape[-1])
    window = np.ascontiguousarray(store[:50], dtype=np.float32)
    import sys

    sys.path.insert(0, str(DEST))
    from trf_falcon_decoder import TrfFalconDecoder

    decoder = TrfFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    decoder.reset(dataset_tags=[stem])
    pred = None
    for row in window:
        pred = decoder.predict(row.reshape(1, -1))
    np.savez(smoke_window, tag_stem=np.asarray(stem), window=window, expected=np.asarray(pred))

    subprocess.run(["docker", "build", "-t", image_tag, "--build-arg", f"PAYLOAD_SHA256={payload_sha}", str(DEST)], check=True)
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
    require(smoke_report.get("status") == "CONTAINER_SMOKE_PASS", f"smoke failed: {smoke_out}")

    candidate = {
        "arm": f"small_trf_s{seed}_ema_e{epoch}_ext6_w0",
        "budget_disclosure": (
            "33 public calibration trials for native E0 and MOVE-T4; static banks; no TTA. "
            "Epoch picked on six locally visible official held-out-calib sessions "
            "(query_start_trial=0), not on ext-4-only and not on 581971."
        ),
        "image_id": image_id,
        "image_tag": image_tag,
        "image_size": int(size),
        "method_description": (
            f"M2 SMALL causal Transformer S1-SMALL-COS seed {seed} EMA epoch {epoch}, "
            f"picked by the MOVE-T4 581919 rule on the six visible official held-out-calib "
            f"sessions (equal-session mean {selected['external_equal_session_mean']:.6f}, "
            f"worst {selected['external_worst_session']}="
            f"{selected['external_worst_session_r2']:.6f}). Exact-E runtime and "
            "dataloader_workers=0. Not registered."
        ),
        "method_label": method_label,
        "method_name": method_name,
        "payload_sha256": payload_sha,
        "register": False,
        "selection_path": str(PICK),
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
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
