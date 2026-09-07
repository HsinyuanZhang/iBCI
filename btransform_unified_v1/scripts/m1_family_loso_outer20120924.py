"""Score frozen M1 family FLAT on the fold-0 outer session ses-20120924.

Family was trained only on 26/27/28. This is the local true LOSO the source
protocol already defined (OUTER_SESSION). Compares family vs Original on the
same left-out held-in-calib query (trials 10..210). CPU-only. No EvalAI.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/xinyuan/Work_host/SPINT")
OUT = ROOT / "btransform_unified_v1/results/m1_loso_outer20120924_v1/family_flat.json"
RUN = ROOT / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1/family_v1/p1_queryage16_pair_chron80_v2"
STATE = RUN / "finalized_p1/flat_selected_ema_state.pt"
OUTER = "ses-20120924"


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    denom = np.square(target - target.mean(0)).sum()
    if not np.isfinite(denom) or denom <= 0:
        raise RuntimeError("zero/nonfinite target variance")
    return float(1.0 - np.square(pred - target).sum() / denom)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _streaming_path() -> None:
    p = str(ROOT / "streaming_calibration_exp")
    if p not in sys.path:
        sys.path.insert(0, p)


def _encode_outer_carrier() -> np.ndarray:
    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank
    from tfpd_exploration.src.m1_optimized_v2 import plan as m1_plan
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.carrier_bank import _encode_session
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data, plan as parent_plan, syn3

    loaded = src_bank.load()
    path = parent_data.require_source_path(ROOT / parent_plan.SOURCE_RELATIVE[OUTER])
    record = fold_data.load_fold_session(path, role="target")
    blob = np.load(m1_plan.BANK_NPZ, allow_pickle=False)
    basis = syn3.SourceBasis(
        kind="nnmf",
        scale=np.asarray(blob["scale"]),
        dictionary=np.asarray(blob["d0"]),
        activations=np.asarray(blob["activations"]),
        order=tuple(int(v) for v in blob["nmf_order"]),
        reconstruction_digest=str(blob["reconstruction_digest"][0]),
        library={"source": "rSyn3-refit-v1.sealed"},
        extra={},
    )
    raw = _encode_session(record, basis)
    return np.ascontiguousarray(
        syn3.normalize_carriers(raw, loaded["normalizer_mean"], loaded["normalizer_scale"]),
        dtype=np.float32,
    )


def _family_bank(calib: torch.Tensor, carrier: np.ndarray):
    """Load B3 id_encoder from S_FIX without the missing Lightning teacher ckpt."""
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank
    from tfpd_exploration.src.m1_optimized_v2 import plan
    from streaming_calibration_exp.src.models.components.streaming_encoders import EarlyPoolEncoder

    if _sha(plan.S_FIX_PATH) != plan.S_FIX_SHA256:
        raise RuntimeError("frozen B3 checkpoint checksum drift")
    payload = torch.load(plan.S_FIX_PATH, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    incoming = {k[len("student.id_encoder."):]: v for k, v in state.items() if k.startswith("student.id_encoder.")}
    encoder = EarlyPoolEncoder(trial_length=1024, window_size=100, hidden_dim=64, num_post_layers=3)
    encoder.load_state_dict(incoming, strict=True)
    encoder.eval()
    trials = calib[0] if calib.dim() == 4 else calib
    stream = encoder.reset_stream(1, plan.N_UNITS, trials.device, trials.dtype)
    with torch.no_grad():
        for trial in trials:
            encoder.push_trial(stream, trial.unsqueeze(0))
        identity = encoder.finalize_identity(stream)
    if identity.dim() == 3:
        identity = identity[0]
    if tuple(identity.shape) != (plan.N_UNITS, 100):
        raise RuntimeError(f"B3 identity shape drift: {tuple(identity.shape)}")
    t4 = torch.from_numpy(np.ascontiguousarray(carrier, dtype=np.float32))
    return M1Bank(
        E0=identity.float().contiguous(),
        T=t4,
        unit_mask=torch.ones(plan.N_UNITS, dtype=torch.bool),
    )


def _loso_module():
    _streaming_path()
    from src.data.m1_version_b_source_loso_datamodule import M1VersionBSourceLOSODataModule
    from tfpd_exploration.src.m1_optimized_v2 import plan

    dm = M1VersionBSourceLOSODataModule(
        task="m1",
        data_dir=str(plan.DATA_DIR),
        source_session_names=list(plan.SOURCE_SESSIONS),
        heldin_session_names=list(plan.SOURCE_SESSIONS),
        batch_size=32,
        window_size=plan.WINDOW,
        calibration_n_trials=10,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        validation_protocol="loso",
        loso_fold=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=10,
        heldin_query_end_trial=210,
        allow_empty_heldout_query=False,
        num_workers=0,
        pin_memory=False,
        sampler_seed=plan.SEED,
        balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False,
        afc4_arm="none",
    )
    dm.setup("test")
    if dm.outer_left_out != OUTER:
        raise RuntimeError("fold-0 outer session drift")
    return dm


def main() -> dict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    torch.set_num_threads(2)
    if not STATE.is_file():
        raise FileNotFoundError(STATE)
    from tfpd_exploration.src.m1_optimized_v2.model import build

    carrier = _encode_outer_carrier()
    dm = _loso_module()
    val = dm.val_heldin_dataset
    if val is None:
        raise RuntimeError("LOSO val_heldin_dataset missing")
    calib = torch.as_tensor(val.calib_trialized_neural_features[OUTER][:10][None, ...], dtype=torch.float32)
    bank = _family_bank(calib, carrier)
    model = build("flat")
    model.load_state_dict(torch.load(STATE, map_location="cpu", weights_only=False), strict=True)
    model.eval()
    preds, targets, windows = [], [], []
    sampler = dm.val_heldin_batch_sampler
    with torch.inference_mode():
        for ids in sampler:
            neural, target, _calib, sessions = next(iter(DataLoader(val, batch_sampler=[ids])))[:4]
            names = [s.decode() if isinstance(s, bytes) else str(s) for s in sessions]
            if any(n != OUTER for n in names):
                raise RuntimeError("non-outer batch in LOSO val")
            out = model.forward_last(neural.float(), bank)
            preds.append(out.cpu().numpy().astype(np.float32))
            targets.append(target[:, -1, :].numpy().astype(np.float32))
            windows.append(np.ascontiguousarray(neural.numpy(), dtype=np.float32))
    pred = np.concatenate(preds)
    y = np.concatenate(targets)
    x = np.concatenate(windows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    query = OUT.parent / "outer_query_windows.npz"
    np.savez(query, windows=x, target=y, session=np.asarray([OUTER]))
    original = _score_original_docker(query)
    family_r2 = _r2(pred, y)
    original_r2 = float(original["pooled_r2"])
    result = {
        "schema": "m1_family_loso_outer20120924_v1",
        "status": "COMPLETE_INFERENCE_ONLY",
        "outer_session": OUTER,
        "n_points": int(len(y)),
        "family": {
            "system": "M1 family FLAT QueryAge16 selected EMA",
            "state_path": str(STATE),
            "state_sha256": _sha(STATE),
            "pooled_r2": family_r2,
        },
        "original": original,
        "same_surface_gap_family_minus_original": family_r2 - original_r2,
        "source_sessions_used_in_training": ["ses-20120926", "ses-20120927", "ses-20120928"],
        "source_minival_family_flat_selected_pooled": 0.811652,
        "source_minival_original_pooled": 0.809289,
        "official_family_queryage": 0.574,
        "official_original": 0.649,
        "official_gap": -0.075,
        "parameter_updates": 0,
        "note": (
            "This is fold-0 held-in day LOSO (20120924), not official October held-out. "
            "If family already drops ~0.07 vs Original here, local 31,252 is not a selector. "
            "If they stay tied, official drop is longer-horizon session shift."
        ),
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(OUT)
    OUT.with_suffix(".json.sha256").write_text(_sha(OUT) + "  " + OUT.name + "\n")
    return result


def _score_original_docker(query: Path) -> dict:
    out = OUT.parent / "original.json"
    cmd = [
        "docker", "run", "--rm", "--network", "none", "--cpus=2",
        "-e", "CUDA_VISIBLE_DEVICES=",
        "-e", "OMP_NUM_THREADS=2",
        "-v", f"{ROOT}/btransform_unified_v1/scripts/inner_original_history_mask.py:/in/run.py:ro",
        "-v", f"{query}:/in/archive.npz:ro",
        "-v", f"{out.parent}:/out",
        "--entrypoint", "python",
        "spint-original-m1:e9-epoch019-052e9ea",
        "/in/run.py", "--task", "m1_outer", "--cache", "/in/archive.npz",
        "--archive", "/in/archive.npz", "--out", "/out/original.json", "--batch", "32",
    ]
    subprocess.check_call(cmd)
    return json.loads(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
