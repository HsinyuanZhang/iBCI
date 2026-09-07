#!/usr/bin/env python3
"""Pack the H1 stage-2 winner into an exact-E Docker image. Does not register."""
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

from btransform_unified_v1 import adapters, h1_config, plan
from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
from tfpd_exploration.src.m2_dual_track_v1.champion import tensor_state_sha256
from tfpd_exploration.src.two_mainlines_long_v1.h1_runtime import constants as C
from tfpd_exploration.src.two_mainlines_long_v1.h1_runtime.banks import bank_receipt, collect_official_banks
from tfpd_exploration.src.two_mainlines_long_v1.h1_runtime.load_weights import export_state_numpy
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.payload import sha256_file

DEST = Path(
    os.environ.get(
        "H1_PACK_DEST",
        str(WORKSPACE_ROOT / "tfpd_exploration/submissions/evalai_h1_projadd_exacte_v1"),
    )
)
PAYLOAD_SCHEMA = "h1_projadd_exacte_falcon_payload_v1"
SCALE = h1_config.TARGET_MULTIPLIER


def _apply_ema(model: BTransformerUnifiedDecoderIdentity, ckpt: dict) -> None:
    model.load_state_dict(ckpt["raw_state_dict"])
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    plan.require(set(named) == set(shadow), "EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))


def _make_geometry(window: int) -> dict:
    geometry = dict(plan.TASK_GEOMETRY["h1"])
    geometry["e0_dim"] = h1_config.matrix_e0_dim("proj_add")
    geometry["task"] = f"h1-stage2-L{window}-proj_add"
    return geometry


def main() -> dict:
    plan.require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    parser_dest = os.environ.get("H1_STAGE2_WINNER_JSON", "")
    winner_path = Path(parser_dest) if parser_dest else None
    if winner_path is None:
        latest = PACKAGE_ROOT / "results/h1_stage2_full13/LATEST_PAIR.txt"
        plan.require(latest.is_file(), "missing LATEST_PAIR.txt and H1_STAGE2_WINNER_JSON")
        winner_path = Path(latest.read_text(encoding="utf-8").strip()) / "winner.json"
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    plan.require(winner.get("winner"), f"no winner in {winner_path}")
    arm = winner["arms"][winner["winner"]]
    plan.require(arm.get("status") == "COMPLETED", f"winner arm not completed: {arm}")
    window = int(arm["window"])
    ckpt_path = Path(arm["ckpt"])
    plan.require(ckpt_path.is_file(), f"missing {ckpt_path}")
    plan.require(window in (200, 250), f"winner window {window}")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts").mkdir(parents=True, exist_ok=True)
    pack_kind = "plus8" if "extend8" in str(winner_path) else "e24"
    payload_name = f"h1_projadd_stage2_s42_ema_{pack_kind}_L{window}.pkl"
    payload_path = DEST / "artifacts" / payload_name
    plan.require(not payload_path.exists(), f"refusing to overwrite {payload_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BTransformerUnifiedDecoderIdentity(
        _make_geometry(window), seed=42, override_prefix=0, override_window=window, identity_mode="proj_add"
    )
    _apply_ema(model, ckpt)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weight_sha = tensor_state_sha256({k: v.detach() for k, v in model.state_dict().items()})
    banks = collect_official_banks()
    config = FalconConfig(task=FalconTask.h1)
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
        "window_size": window,
        "behavior_scaling_factor": SCALE,
        "smooth_observations": False,
        "metadata": {
            "kind": "proj_add",
            "cell": f"H1-STAGE2-L{window}-PROJ-ADD",
            "seed": 42,
            "epoch": 24,
            "view": "EMA",
            "ckpt": str(ckpt_path),
            "weight_sha256": weight_sha,
            "selection_surface": "stage-2 endpoint24 EMA (no surface pick)",
            "selection_mean": arm["minival13_ema_equal_session_mean"],
            "winner_rule": winner["rule"],
            "online_kv_cache": False,
            "old_spint_decoder": False,
            "exact_e": True,
            "dataloader_workers": 0,
            "official_tags": list(C.OFFICIAL_TAGS),
            "evalai_opened": False,
        },
    }
    with payload_path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    payload_sha = sha256_file(payload_path)

    session = HELDIN_SESSIONS[0]
    stem = f"sub-HumanPitt-held-in-minival_{session}"
    tag = config.hash_dataset(stem)
    plan.require(tag in banks, f"hashed tag {tag} missing from official banks")
    cache = adapters._h1_source_cache()
    neural = np.ascontiguousarray(cache["minival"][session]["neural"], dtype=np.float32)
    stream = neural[:window]
    plan.require(stream.shape == (window, 176), f"smoke stream shape {stream.shape}")
    official = banks[tag]
    task_bank = TaskBank(
        session_id=session,
        E0=official["E0"],
        carrier=official["T"],
        unit_mask=official["unit_mask"],
        X_store=stream[None],
        target_store=np.zeros((1, 7), dtype=np.float32),
        window_ids=np.array([0], dtype=np.int64),
        calibration_meta={
            "shape": list(official["E0"].shape),
            "trial_count": 3,
            "estimator": "official_c2_m3_for_parity",
            "array_sha256": array_sha256(official["E0"]),
            "budget": 3,
        },
    )
    x = torch.from_numpy(stream[None])
    with torch.inference_mode():
        train_pred = (model(x, task_bank) / SCALE).detach().cpu().numpy()

    sys.path.insert(0, str(DEST))
    from h1_trf_falcon_decoder import H1ProjAddFalconDecoder

    decoder = H1ProjAddFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    decoder.reset(dataset_tags=[stem])
    pred = None
    for row in stream:
        pred = decoder.predict(row.reshape(1, -1))
    delta = float(np.max(np.abs(pred - train_pred)))
    plan.require(np.isfinite(pred).all(), "non-finite exact-E pred")
    plan.require(delta <= 1.0e-5, f"exact-E vs train forward max|Δ|={delta}")

    smoke_window = DEST / "artifacts" / "smoke_window.npz"
    np.savez(smoke_window, tag_stem=np.asarray(stem), window=stream, expected=np.asarray(pred))

    (DEST / "artifacts" / "payload.receipt.json").write_text(
        json.dumps(
            {
                "schema": "h1_projadd_stage2_exacte_payload_v1",
                "payload_path": str(payload_path),
                "payload_sha256": payload_sha,
                "bytes": payload_path.stat().st_size,
                "weight_sha256": weight_sha,
                "winner": winner,
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
        "**\n!.dockerignore\n!Dockerfile\n!decode.py\n!h1_trf_falcon_decoder.py\n"
        f"!artifacts/\n!artifacts/{payload_name}\n",
        encoding="utf-8",
    )
    method_name = f"H1 proj_add stage-2 EMA {pack_kind} L{window} exact-E w0"
    method_label = (
        f"H1 proj_add CausalPE4 stage-2 full-13 retrain (seed42 EMA e24 L={window}) "
        "exact-E w0; not SPINT; not a C2 identity swap"
    )
    method_description = (
        f"H1 proj_add CausalPE4 stage-2 full-13 retrain, seed 42, endpoint24 EMA, "
        f"L={window}. CAL-1 deploy M3 (budget=3). Official HO unread; minival13 is "
        "diagnostic after the all-13 retrain. Exact-E, dataloader_workers=0."
    )
    budget_disclosure = (
        "H1 CAL-1 deploy M3 (budget=3); C2 e15 frozen materializer; static banks; no TTA. "
        "Checkpoint is preregistered endpoint24 EMA after optional +8. Local minival is diagnostic only."
    )
    image_tag = f"spint-t4-h1:projadd-s42-ema-{pack_kind}-L{window}-w0-{payload_sha[:8]}"
    (DEST / "Dockerfile").write_text(
        f"ARG BASE_IMAGE={C.BASE_IMAGE}\n"
        "FROM ${BASE_IMAGE}\n"
        f"ARG PAYLOAD_SHA256={payload_sha}\n"
        f'LABEL ai.eval.method="{method_label}"\n'
        'LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"\n'
        'LABEL ai.eval.task="h1"\n'
        'LABEL ai.eval.old_spint_decoder="false"\n'
        'LABEL ai.eval.exact_e="true"\n'
        'LABEL ai.eval.dataloader_workers="0"\n'
        f'LABEL ai.eval.candidate="h1_projadd_stage2_s42_ema_e24_L{window}_w0"\n'
        f"COPY artifacts/{payload_name} /data/decoder.pkl\n"
        "COPY h1_trf_falcon_decoder.py /h1_trf_falcon_decoder.py\n"
        "COPY decode.py /decode.py\n"
        "ENV EVALUATION_LOC=remote TASK=h1 PHASE=test BATCH_SIZE=8 "
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
            f"{(DEST / 'h1_trf_falcon_decoder.py').resolve()}:/h1_trf_falcon_decoder.py:ro",
            image_tag,
            "python",
            "/h1_trf_falcon_decoder.py",
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
        "arm": f"h1_projadd_stage2_s42_ema_e24_L{window}_w0",
        "image_id": image_id,
        "image_tag": image_tag,
        "image_size": int(size),
        "method_name": method_name,
        "method_label": method_label,
        "method_description": method_description,
        "budget_disclosure": budget_disclosure,
        "payload_sha256": payload_sha,
        "parity_exact_e_vs_train_max_abs": delta,
        "selection_mean": arm["minival13_ema_equal_session_mean"],
        "window": window,
        "register": False,
        "evalai_opened": False,
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "hold_reason": "packed for authorized overnight H1 submit before EvalAI daily reset",
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
                "schema_version": "h1_trf_evalai_push_state",
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
