"""LOSO-matched evaluation protocol for B3 quantization validation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BASELINE_R2_TOLERANCE = 1e-5


@dataclass
class LosoSplit:
    fold_id: int
    train_sessions: List[str]
    validation_sessions: List[str]
    heldout_session: str


def load_split_manifest(path: Path) -> LosoSplit:
    data = json.loads(path.read_text(encoding="utf-8"))
    val_sessions = list(data["validation_sessions"])
    if len(val_sessions) != 1:
        raise ValueError(f"Expected exactly one validation session, got {val_sessions}")
    return LosoSplit(
        fold_id=int(data["fold_id"]),
        train_sessions=list(data["train_sessions"]),
        validation_sessions=val_sessions,
        heldout_session=val_sessions[0],
    )


def build_loso_datamodule(exp_root: Path, data_dir: Path, fold_id: int = 0):
    import importlib.util
    import sys

    exp_root = exp_root.resolve()
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))

    dm_path = exp_root / "src" / "data" / "falcon_datamodule.py"
    spec = importlib.util.spec_from_file_location("b3_falcon_datamodule", dm_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FalconDataModule from {dm_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    FalconDataModule = mod.FalconDataModule

    dm = FalconDataModule(
        task="m2",
        data_dir=str(data_dir.resolve()),
        heldin_session_names=[""],
        batch_size=32,
        window_size=50,
        calibration_n_trials=33,
        random_calibration=True,
        smooth_calibration=False,
        max_trial_length=100,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        validation_protocol="loso",
        loso_fold=fold_id,
        rotation_id=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("test")
    return dm


def get_dataset_for_session(dm, session_name: str, split: str = "val"):
    if split == "val":
        return dm.val_heldin_dataset
    if split == "train":
        return dm.train_dataset
    raise ValueError(split)


def _session_window_indices_full_batches(dataset, session_name: str, batch_size: int = 32) -> List[int]:
    """Match SessionBatchSampler (shuffle=False): only windows in complete batches."""
    session_indices: List[int] = []
    for idx, (sess, _) in enumerate(dataset.window_indices):
        if sess == session_name:
            session_indices.append(idx)
    kept: List[int] = []
    for start in range(0, len(session_indices), batch_size):
        batch = session_indices[start : start + batch_size]
        if len(batch) == batch_size:
            kept.extend(batch)
    return kept


def collect_session_windows(
    dataset,
    session_name: str,
    *,
    batch_size: int = 32,
    drop_partial_last_batch: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Windows for session: neural [S,W,N], behavior [S,W,C].

    When drop_partial_last_batch=True (default), only windows included in complete
    SessionBatchSampler batches are returned — matching Lightning test/val R².
    """
    if drop_partial_last_batch:
        indices = _session_window_indices_full_batches(dataset, session_name, batch_size=batch_size)
    else:
        indices = [idx for idx, (sess, _) in enumerate(dataset.window_indices) if sess == session_name]
    if not indices:
        raise ValueError(f"No windows for session {session_name}")
    neural_list, behavior_list = [], []
    for idx in indices:
        neural_w, cov_w, _calib, _name = dataset[idx]
        neural_list.append(neural_w.numpy() if hasattr(neural_w, "numpy") else np.asarray(neural_w))
        behavior_list.append(cov_w.numpy() if hasattr(cov_w, "numpy") else np.asarray(cov_w))
    return np.stack(neural_list, axis=0).astype(np.float32), np.stack(behavior_list, axis=0).astype(np.float32)


def get_full_calib_pool(dataset, session_name: str) -> np.ndarray:
    """Full trial pool [M_total, T, N] for a session."""
    pool = dataset.calib_trialized_neural_features[session_name]
    return np.asarray(pool, dtype=np.float32)


def sample_calib_draw(
    pool: np.ndarray,
    *,
    num_trials: int = 33,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select num_trials indices from pool; return (indices, calib[M,T,N])."""
    rng = np.random.default_rng(seed)
    m_total = pool.shape[0]
    if num_trials > m_total:
        raise ValueError(f"Cannot sample {num_trials} from pool size {m_total}")
    indices = np.sort(rng.choice(m_total, size=num_trials, replace=False))
    return indices.astype(np.int64), pool[indices].astype(np.float32)


def stable_session_seed(base_seed: int, session_name: str) -> int:
    """Deterministic per-session seed (unlike built-in hash())."""
    h = int(hashlib.md5(session_name.encode("utf-8")).hexdigest()[:8], 16)
    return int(base_seed) + (h % 10000)
    return hashlib.sha256(calib.tobytes()).hexdigest()


def variance_weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Match torchmetrics R2Score(multioutput='variance_weighted') on last-timestep vectors."""
    import torch
    from torchmetrics.regression import R2Score

    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    if y_true.ndim == 3:
        y_true = y_true[:, -1, :]
        y_pred = y_pred[:, -1, :]
    metric = R2Score(multioutput="variance_weighted")
    metric.update(torch.from_numpy(y_pred), torch.from_numpy(y_true))
    return float(metric.compute().item())


def load_student_from_ckpt(ckpt_path: Path, exp_root: Path):
    import torch
    import sys

    if str(exp_root.resolve()) not in sys.path:
        sys.path.insert(0, str(exp_root.resolve()))
    from src.models.streaming_calibration_module import StreamingCalibrationLitModule

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = dict(ckpt.get("hyper_parameters", {}))
    teacher_path = Path(hp.get("teacher_ckpt_path", ""))
    if not teacher_path.is_file():
        for cand in [
            exp_root / hp.get("teacher_ckpt_path", ""),
            exp_root.parent / "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt",
        ]:
            if cand.is_file():
                hp["teacher_ckpt_path"] = str(cand.resolve())
                break
    lit = StreamingCalibrationLitModule(**hp)
    lit.setup("test")
    lit.load_state_dict(ckpt["state_dict"], strict=False)
    lit.eval()
    assert lit.student is not None
    return lit.student


def decode_session_with_E(
    student,
    neural: np.ndarray,
    E: np.ndarray,
    *,
    behavior_scale: float = 5.0,
) -> np.ndarray:
    """neural [S,W,N], E [N,W] -> predictions [S,W,C] scaled like training."""
    import torch

    device = next(student.parameters()).device
    preds = []
    batch_size = 64
    with torch.no_grad():
        e_t = torch.from_numpy(E).unsqueeze(0).to(device)
        for start in range(0, neural.shape[0], batch_size):
            n_t = torch.from_numpy(neural[start : start + batch_size]).to(device)
            y = student.decode_with_identity(n_t, e_t).cpu().numpy()
            y = y[:, -1:, :] / behavior_scale
            preds.append(y)
    return np.concatenate(preds, axis=0)


def session_r2_with_E(
    student,
    neural: np.ndarray,
    behavior: np.ndarray,
    E: np.ndarray,
    *,
    behavior_scale: float = 5.0,
) -> float:
    y_pred = decode_session_with_E(student, neural, E, behavior_scale=behavior_scale)
    y_true = behavior[:, -1:, :]
    return variance_weighted_r2(y_true, y_pred)


def session_r2_fp32_encoder(student, neural: np.ndarray, behavior: np.ndarray, calib_mtN: np.ndarray, *, behavior_scale: float = 5.0) -> float:
    import torch

    with torch.no_grad():
        calib_t = torch.from_numpy(calib_mtN).unsqueeze(0)
        E = student.compute_identity(calib_t).squeeze(0).cpu().numpy()
    return session_r2_with_E(student, neural, behavior, E, behavior_scale=behavior_scale)


def check_baseline_r2(
    measured: float,
    expected: float,
    *,
    session: str,
    tolerance: float = BASELINE_R2_TOLERANCE,
) -> Dict[str, Any]:
    delta = abs(measured - expected)
    ok = delta <= tolerance
    return {
        "session": session,
        "measured_r2": measured,
        "expected_r2": expected,
        "abs_delta": delta,
        "tolerance": tolerance,
        "pass": ok,
    }
