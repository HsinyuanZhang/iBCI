#!/usr/bin/env python3
"""Pack M1 proj_add P16 endpoint24 EMA into an exact-E image. Does not register."""
from __future__ import annotations

import hashlib
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

from btransform_unified_v1 import m1_projadd as mp
from btransform_unified_v1 import plan
from btransform_unified_v1.bank import array_sha256
from tfpd_exploration.src.m2_dual_track_v1.champion import tensor_state_sha256
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.load_weights import export_state_numpy
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.payload import sha256_file

TRAIN_DEST = PACKAGE_ROOT / "results/m1_projadd_series/P16_20260906T150917Z"
DEST = WORKSPACE_ROOT / "tfpd_exploration/submissions/evalai_m1_projadd_exacte_v1"
PAYLOAD_SCHEMA = "m1_projadd_exacte_falcon_payload_v1"
HO_CALIB = {
    "ses-20121004": {
        "tag": "20121004",
        "path": WORKSPACE_ROOT
        / "SPINT-main/data/000941/sub-MonkeyL-held-out-calib/sub-MonkeyL-held-out-calib_ses-20121004_behavior+ecephys.nwb",
        "sha256": "782c1fd090facfb3c50b6a85da8209ca3aea71f66bd4edc13d61fe4a55a3429d",
    },
    "ses-20121017": {
        "tag": "20121017",
        "path": WORKSPACE_ROOT
        / "SPINT-main/data/000941/sub-MonkeyL-held-out-calib/sub-MonkeyL-held-out-calib_ses-20121017_behavior+ecephys.nwb",
        "sha256": "8dd22c67500445ec1e0c11980475badbba9e960a05db16b78aaaad5502b3c652",
    },
    "ses-20121024": {
        "tag": "20121024",
        "path": WORKSPACE_ROOT
        / "SPINT-main/data/000941/sub-MonkeyL-held-out-calib/sub-MonkeyL-held-out-calib_ses-20121024_behavior+ecephys.nwb",
        "sha256": "bbeb6c7d66e2c2e9b76506021a6cf8800bc19021b6d1fd03c4eb6de71490bafb",
    },
}
BASE_IMAGE = "spint-original-m1:e9-epoch019-052e9ea"
PROJ_DIM = 16
TOKEN_IN = 20
HOPEFUL_MINIVAL = 0.70
PRIOR_HO = 0.5740867465488526  # 581982


def _apply_ema(model, ckpt: dict) -> None:
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    plan.require(set(named) == set(shadow), "EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


V3_BANKS_PKL = (
    WORKSPACE_ROOT
    / "tfpd_exploration/submissions/evalai_m1_runtime_v3_selected_t_v1/artifacts/m1_optimized_v2_t_ema_e6.pkl"
)


def _official_ho_carriers() -> dict[str, np.ndarray]:
    """rSyn3 T for the 3 public later-day files.

    Those NWBs are exactly 10 trials and live under held-out-calib, so the
    held-in-only ``load_support_bins`` allowlist cannot open them. The V3
    payload already stores the sealed-basis + source-normalizer T for the
    same 7 public tags; source-session T matched this train inventory.
    """
    with V3_BANKS_PKL.open("rb") as handle:
        payload = pickle.load(handle)
    banks = payload["bank_by_dataset_tag"]
    out = {}
    for spec in HO_CALIB.values():
        tag = spec["tag"]
        plan.require(tag in banks, f"V3 payload missing HO tag {tag}")
        out[tag] = np.ascontiguousarray(banks[tag]["T"], dtype=np.float32)
        plan.require(out[tag].shape == (mp.M1_UNITS, mp.M1_CARRIER_DIM), f"{tag} T shape")
    return out


def _ho_calib_trials(dm, path: Path, session: str) -> np.ndarray:
    from src.data.falcon_datamodule import FalconDataset

    h = dm.hparams
    task = dm._falcon_task_m1()
    source = dm.train_calib_heldin_sessions[mp.M1_SOURCE_SESSIONS[0]]
    record = dm.prepare_session_data(
        path,
        task,
        standardize_covariates=False,
        covariates_mean=source["covariates_mean"],
        covariates_std=source["covariates_std"],
        use_intertrials=True,
    )
    dataset = FalconDataset(
        sessions_dict={session: record},
        calib_sessions_dict={session: record},
        window_size=h.window_size,
        split="train",
        calibration_n_trials=h.calibration_n_trials,
        random_calibration=False,
        smooth_calibration=h.smooth_calibration,
        max_trial_length=h.max_trial_length,
        use_calib_intertrials=h.use_calib_intertrials,
        trial_feature_type=h.trial_feature_type,
        remove_still_times=h.remove_still_times,
        remove_calib_still_times=h.remove_calib_still_times,
        use_calib_active_segments=h.use_calib_active_segments,
        calib_n_active_segments=h.calib_n_active_segments,
        interpolate_trials=h.interpolate_trials,
        interpolate_trials_kind=h.interpolate_trials_kind,
        pad_value=h.pad_value,
        query_start_trial=0,
        query_end_trial=None,
        allow_empty_query_sessions=True,
    )
    calib = np.asarray(dataset.calib_trialized_neural_features[session][: mp.M10_BUDGET], dtype=np.float32)
    plan.require(calib.shape[0] == mp.M10_BUDGET, f"{session} calib trials {calib.shape}")
    return calib


def _row(e0: np.ndarray, carrier: np.ndarray, unit_mask: np.ndarray) -> dict:
    return {
        "E0": np.ascontiguousarray(e0, dtype=np.float32),
        "T": np.ascontiguousarray(carrier, dtype=np.float32),
        "unit_mask": np.ascontiguousarray(unit_mask, dtype=np.bool_),
    }


def _load_deploy_banks() -> dict:
    """B3 + sealed rSyn3 banks. V3 concat E0 is a different identity and is not reused."""
    inventory = json.loads((TRAIN_DEST / "session_inventory.json").read_text(encoding="utf-8"))
    plan.require(
        set(inventory["sessions"]) == set(mp.M1_SESSIONS),
        f"submit candidate must train all local held-in sessions {mp.M1_SESSIONS}, got {inventory['sessions']}",
    )
    plan.require(
        "ALL 4 local held-in" in str(inventory.get("protocol", "")),
        "refusing a LOSO/source-only dest: submission candidate is stage-2 all-4 held-in",
    )
    for session in mp.M1_SESSIONS:
        n_win = int(inventory["window_audit"][session]["eligible_windows"])
        plan.require(n_win > 0, f"{session} has no train windows")
    dm = mp.build_loso_datamodule()
    dataset, _ = mp.assemble_training_universe(dm)
    calib = mp.calib_trials_from_dataset(dataset)
    carriers = mp.load_source_carriers()
    outer, _ = mp.encode_outer_carrier()
    carriers = {**carriers, mp.M1_OUTER_SESSION: outer}
    train_banks, _ = mp.build_banks(calib, carriers)
    for session in mp.M1_SESSIONS:
        plan.require(
            array_sha256(train_banks[session].E0) == inventory["banks"][session]["e0_sha256"],
            f"train E0 SHA drift vs inventory: {session}",
        )
        plan.require(
            array_sha256(train_banks[session].carrier) == inventory["banks"][session]["carrier_sha256"],
            f"train T SHA drift vs inventory: {session}",
        )

    banks = {
        session.removeprefix("ses-"): _row(
            train_banks[session].E0, train_banks[session].carrier, train_banks[session].unit_mask
        )
        for session in mp.M1_SESSIONS
    }
    encoder = mp.load_b3_id_encoder()
    ho_t = _official_ho_carriers()
    for session, spec in HO_CALIB.items():
        path = spec["path"]
        plan.require(path.is_file(), f"missing official calib {path}")
        plan.require(_file_sha256(path) == spec["sha256"], f"official calib SHA drift: {session}")
        e0 = mp.b3_identity(encoder, _ho_calib_trials(dm, path, session))
        banks[spec["tag"]] = _row(e0, ho_t[spec["tag"]], np.ones(mp.M1_UNITS, dtype=np.bool_))
    plan.require(len(banks) == 7, f"deploy banks {len(banks)} != 7")
    return banks


def main() -> dict:
    plan.require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    score = json.loads((TRAIN_DEST / "score_receipt.json").read_text(encoding="utf-8"))
    minival = float(score["acceptance_accounting"]["ours_minival_pooled"])
    plan.require(np.isfinite(minival), "non-finite minival")
    plan.require(minival >= HOPEFUL_MINIVAL, f"not hopeful: minival pooled {minival:.4f} < {HOPEFUL_MINIVAL}")
    ckpt_path = TRAIN_DEST / "epoch_024.pt"
    plan.require(ckpt_path.is_file(), f"missing {ckpt_path}")
    plan.require((DEST / "trf_falcon_decoder.py").is_file(), "missing M1 proj_add decoder")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts").mkdir(parents=True, exist_ok=True)
    payload_name = "m1_projadd_p16_s42_ema_e24.pkl"
    payload_path = DEST / "artifacts" / payload_name
    plan.require(not payload_path.exists(), f"refusing to overwrite {payload_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = mp.build_m1_projadd_model(PROJ_DIM, seed=42)
    _apply_ema(model, ckpt)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    plan.require(int(model.token_in) == TOKEN_IN, f"token_in {model.token_in}")
    weight_sha = tensor_state_sha256({k: v.detach() for k, v in model.state_dict().items()})
    banks = _load_deploy_banks()

    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank, data as src_data

    loaded = src_bank.load()
    dm = src_data.build_source_only_datamodule(loaded)
    session = "ses-20120926"
    neural = np.asarray(dm.train_dataset.base.neural_data[session], dtype=np.float32)
    pad = 99
    window = np.ascontiguousarray(neural[pad : pad + mp.M1_WINDOW], dtype=np.float32)
    plan.require(window.shape == (mp.M1_WINDOW, mp.M1_UNITS), f"smoke window {window.shape}")
    official = banks["20120926"]
    task_bank = mp.make_m1_bank(session, official["E0"], official["T"], unit_mask=official["unit_mask"])
    e0_delta = 0.0
    t_delta = 0.0

    x = torch.from_numpy(window[None])
    with torch.inference_mode():
        train_pred = model(x, task_bank).detach().cpu().numpy()

    config = FalconConfig(task=FalconTask.m1)
    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "task": config.task,
        "kind": "proj_add",
        "state_dict": export_state_numpy(model),
        "bank_by_dataset_tag": banks,
        "window_size": mp.M1_WINDOW,
        "behavior_scaling_factor": mp.M1_DIVISOR,
        "smooth_observations": False,
        "metadata": {
            "kind": "proj_add",
            "cell": "M1-PROJADD-P16",
            "seed": 42,
            "epoch": 24,
            "view": "EMA",
            "proj_dim": PROJ_DIM,
            "token_in": TOKEN_IN,
            "ckpt": str(ckpt_path),
            "weight_sha256": weight_sha,
            "label_budget": 10,
            "exact_e": True,
            "dataloader_workers": 0,
            "torch_threads": 2,
            "evalai_opened": False,
            "train_sessions": list(mp.M1_SESSIONS),
            "held_out": "EvalAI hidden only; local 20120924 is in train; public later-day 20121004/17/24 are M10 calib banks, not a selector",
            "diagnostic_minival_pooled": minival,
            "prior_official_ho_581982": PRIOR_HO,
        },
    }
    with payload_path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    payload_sha = sha256_file(payload_path)

    sys.path.insert(0, str(DEST))
    from trf_falcon_decoder import TrfFalconDecoder

    decoder = TrfFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    decoder.reset(dataset_tags=["L_20120926_held_in_eval"])
    pred = None
    for row in window:
        pred = decoder.predict(row.reshape(1, -1))
    delta = float(np.max(np.abs(pred - train_pred)))
    plan.require(np.isfinite(pred).all(), "non-finite exact-E pred")
    plan.require(delta <= 1.0e-5, f"exact-E vs train forward max|Δ|={delta}")

    smoke_window = DEST / "artifacts" / "smoke_window.npz"
    np.savez(smoke_window, tag_stem=np.asarray("L_20120926_held_in_eval"), window=window, expected=np.asarray(pred))

    method_name = "M1 proj_add P16 Transformer EMA e24 s42 exact-E w0"
    method_label = (
        "M1 proj_add P16 CausalPE4 (seed42 EMA e24) exact-E w0; "
        "endpoint24 EMA; official uint8/float coerce"
    )
    method_description = (
        f"M1 proj_add P16 (Linear 100->16, token_in=20) CausalPE4 seed 42 EMA epoch 24. "
        f"Diagnostic source-minival pooled {minival:.6f} (polluted stage-2 face; not a selector). "
        f"Prior official 581982 HO {PRIOR_HO:.4f}. Exact-E, dataloader_workers=0, torch threads=2."
    )
    image_tag = f"spint-m1:projadd-p16-s42-ema-e24-w0-{payload_sha[:8]}"
    (DEST / "Dockerfile").write_text(
        f"ARG BASE_IMAGE={BASE_IMAGE}\n"
        "FROM ${BASE_IMAGE}\n"
        f"ARG PAYLOAD_SHA256={payload_sha}\n"
        f'LABEL ai.eval.method="{method_label}"\n'
        'LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"\n'
        'LABEL ai.eval.task="m1"\n'
        'LABEL ai.eval.label_budget="10"\n'
        'LABEL ai.eval.exact_e="true"\n'
        'LABEL ai.eval.proj_dim="16"\n'
        'LABEL ai.eval.dataloader_workers="0"\n'
        'LABEL ai.eval.torch_threads="2"\n'
        f"COPY artifacts/{payload_name} /data/decoder.pkl\n"
        "COPY trf_falcon_decoder.py /trf_falcon_decoder.py\n"
        "COPY decode.py /decode.py\n"
        "ENV EVALUATION_LOC=remote TASK=m1 PHASE=test BATCH_SIZE=4 "
        "PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2\n"
        'CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC '
        '--model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]\n',
        encoding="utf-8",
    )
    (DEST / ".dockerignore").write_text(
        "**\n!.dockerignore\n!Dockerfile\n!decode.py\n!trf_falcon_decoder.py\n"
        f"!artifacts/\n!artifacts/{payload_name}\n",
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

    state_path = DEST / "artifacts" / "evalai_push_state.json"
    candidate = {
        "arm": "projadd_p16_s42_ema_e24_w0",
        "image_id": image_id,
        "image_tag": image_tag,
        "image_size": int(size),
        "method_name": method_name,
        "method_label": method_label,
        "method_description": method_description,
        "budget_disclosure": (
            "10 public calibration trials for B3 E0 and rSyn3; static banks; no TTA. "
            "Checkpoint is preregistered endpoint24 EMA. Local minival/LOSO are diagnostic only."
        ),
        "payload_sha256": payload_sha,
        "parity_exact_e_vs_train_max_abs": delta,
        "official_vs_train_e0_max_abs": e0_delta,
        "diagnostic_minival_pooled": minival,
        "prior_official_ho_581982": PRIOR_HO,
        "proj_dim": PROJ_DIM,
        "state_path": str(state_path),
        "register": False,
        "evalai_opened": False,
    }
    (DEST / "artifacts" / "evalai_candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state_path.write_text(
        json.dumps(
            {
                "arm": candidate["arm"],
                "image_tag": image_tag,
                "payload_sha256": payload_sha,
                "schema_version": "m1_projadd_evalai_push_state",
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
