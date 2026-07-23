"""Shared bridge between TCN student and the m2-research mask-based eval protocol.

Single source for: held-in stats, gap-safe windowing, frozen-TCN inference.

Stats unification (one allframe_stats call):
  held_in_stats() returns (mu, sigma, cov) from held-in all frames. mu/sigma serve
  BOTH the z-score basis AND the CORAL reference (they are bit-identical, since
  channel_stats and allframe_stats derive the same mu/sigma from the same input).
  Do NOT call channel_stats and allframe_stats separately -- that recomputes the
  same numbers twice and created the illusion of "two held-in stat sets".

Alignment regimes (see b3_tcn_crossday.py):
  own_zscore : train on z-scored held-in; test = CORAL->zscore (== own-zscore;
               the held-in reference cancels, by construction).
  ref_raw    : train on RAW held-in; test = CORAL(today -> held-in-raw-ref), NO
               post-zscore. CORAL(held-in -> held-in-raw-ref) = identity, so the
               train/test representations match, and the held-in per-channel scale
               genuinely shapes the warp (the only diagonal design that does not
               collapse to a scalar standardization).
"""
from __future__ import annotations

import numpy as np
import torch

import _env  # noqa: F401  bootstraps sys.path to m2-research
from lib.rest_align import allframe_stats, apply_fixed_zscore  # noqa: E402


def held_in_stats(smooth_hi: np.ndarray):
    """Single allframe_stats(smooth_hi) -> (mu, sigma, cov).

    mu/sigma are used for BOTH z-score and CORAL reference (identical values);
    cov is the CORAL reference covariance if a full-CORAL variant is ever added.
    """
    mu, sigma, cov, _ = allframe_stats(smooth_hi)
    return mu, sigma, cov


def window_segment(x_seg: np.ndarray, y_seg: np.ndarray, window: int):
    """Sliding windows over ONE contiguous segment (no gap handling here).

    (T,C) x_seg -> (N,C,window) with N = T-window+1; targets are last-frame y.
    """
    T, C = x_seg.shape
    if T < window:
        empty_x = np.empty((0, C, window), dtype=np.float32)
        empty_y = np.empty((0, y_seg.shape[1]), dtype=np.float32)
        return empty_x, empty_y
    X = np.lib.stride_tricks.sliding_window_view(x_seg, window, axis=0)  # (N,C,window)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y_w = np.ascontiguousarray(y_seg[window - 1:], dtype=np.float32)    # last frame target
    return X, y_w


def _contiguous_runs(mask: np.ndarray):
    idx = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]])))
    return [(int(idx[i]), int(idx[i + 1])) for i in range(0, len(idx), 2)]


def valid_window_indices(mask: np.ndarray, window: int) -> np.ndarray:
    """Indices i in mask where the full window [i-window+1 .. i] lies inside mask.

    Windows per contiguous run of mask, dropping the first (window-1) frames of
    each run as window-end targets. This is gap-safe: a window never straddles a
    fold boundary or an internal gap. Fixes both test->calib fold leakage and
    windowing across held-in concatenation gaps at fold edges.

    Note: contiguous runs longer than window still contribute; this does NOT split
    two sessions that are concatenated without a mask gap between them (shared
    limitation with the ridge pipeline, which also windows across concat points).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    out = []
    for s, e in _contiguous_runs(mask):
        if e - s >= window:
            out.append(np.arange(s + window - 1, e, dtype=np.int64))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def make_train_windows(t: torch.Tensor, idx: torch.Tensor, window: int) -> torch.Tensor:
    """Gather (B, C, window) windows ending (inclusive) at each idx from (T, C) t.

    idx must already be gap/fold-safe (see valid_window_indices). idx is a 1-D
    long tensor on the same device as t.
    """
    offs = torch.arange(window - 1, -1, -1, device=t.device)           # (window,)
    gi = idx[:, None] - offs[None, :]                                  # (B, window)
    return t[gi].permute(0, 2, 1).contiguous()                         # (B, C, window)


def window_masked(smooth, vel, mask, mu_z, sigma_z, window):
    """Window each contiguous run of mask separately (gap-safe), z-scoring inside.

    Kept for callers that want numpy windows with z-score applied per run.
    """
    Xs, ys = [], []
    for s, e in _contiguous_runs(mask):
        x_seg = smooth[s:e]
        y_seg = vel[s:e]
        x_norm = apply_fixed_zscore(x_seg, mu_z, sigma_z)
        X, y_w = window_segment(x_norm, y_seg, window)
        if len(X):
            Xs.append(X); ys.append(y_w)
    if not Xs:
        c = smooth.shape[1]
        return np.empty((0, c, window), dtype=np.float32), np.empty((0, vel.shape[1]), dtype=np.float32)
    return np.concatenate(Xs), np.concatenate(ys)


@torch.no_grad()
def predict_frozen_segment(
    tcn: torch.nn.Module,
    x_seg: np.ndarray,
    y_seg: np.ndarray,
    window: int,
    device=None,
    batch: int = 8192,
):
    """Frozen TCN over ONE contiguous already-transformed segment.

    Caller is responsible for the full align+represent transform (regime-specific)
    producing x_seg in the exact space the TCN was trained on. This helper only
    slides windows and runs the frozen forward -- replacing inline unfold.
    Returns (pred, y_w) concatenated; caller computes R2.
    """
    T = x_seg.shape[0]
    if T < window:
        return np.empty((0, y_seg.shape[1])), np.empty((0, y_seg.shape[1]))
    X, y_w = window_segment(x_seg, y_seg, window)
    tcn.eval()
    was_training = tcn.training
    dev = device if device is not None else next(tcn.parameters()).device
    preds = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(dev)
        preds.append(tcn(xb).cpu().numpy())
    if was_training:
        tcn.train()
    return np.concatenate(preds, axis=0), y_w


if __name__ == "__main__":
    # --- window_segment ---
    x = np.random.randn(1000, 96).astype(np.float32)
    y = np.random.randn(1000, 2).astype(np.float32)
    Xw, yw = window_segment(x, y, 50)
    assert Xw.shape == (951, 96, 50) and yw.shape == (951, 2), "window_segment shape"
    assert np.allclose(Xw[0, :, -1], x[49]) and np.allclose(yw[0], y[49]), "last-frame align"
    print("window_segment OK")

    # --- held_in_stats identity: mu/sigma == channel_stats ---
    from lib.rest_align import channel_stats
    mu, sigma, cov = held_in_stats(x)
    mu_c, sigma_c = channel_stats(x)
    assert np.allclose(mu, mu_c) and np.allclose(sigma, sigma_c), "held_in_stats != channel_stats"
    assert cov.shape == (96, 96), "cov shape"
    print("held_in_stats OK (mu/sigma identical to channel_stats)")

    # --- valid_window_indices: fold-boundary leak guard ---
    mask = np.zeros(1000, dtype=bool)
    mask[100:600] = True                       # one run
    idx = valid_window_indices(mask, 50)
    assert idx.min() == 149 and idx.max() == 599, "run edges drop first window-1"
    assert idx.shape == (451,)
    # two runs with a gap -> windows must not cross the gap
    mask2 = np.zeros(1000, dtype=bool)
    mask2[100:300] = True
    mask2[400:600] = True
    idx2 = valid_window_indices(mask2, 50)
    runs = _contiguous_runs(mask2)
    for s, e in runs:
        seg = idx2[(idx2 >= s) & (idx2 < e)]
        assert seg.min() == s + 49 and seg.max() == e - 1, "per-run gap-safe"
    print("valid_window_indices OK (gap + fold-boundary safe)")

    # --- make_train_windows (torch) ---
    xt = torch.tensor(x)
    it = torch.tensor(idx[:8], dtype=torch.long)
    W = make_train_windows(xt, it, 50)
    assert W.shape == (8, 96, 50), "make_train_windows shape"
    assert torch.allclose(W[0, :, -1], xt[149]), "first window last col = frame 149"
    assert torch.allclose(W[0, :, 0], xt[100]), "first window first col = frame 100"
    print("make_train_windows OK")

    # --- predict_frozen_segment ---
    from models.tcn_student import DepthwiseTemporalConvStudent
    tcn = DepthwiseTemporalConvStudent(n_ch=96, window=50, kernel=9, hidden=32)
    xseg = np.random.randn(600, 96).astype(np.float32)
    yseg = np.random.randn(600, 2).astype(np.float32)
    pred, yt = predict_frozen_segment(tcn, xseg, yseg, 50)
    assert pred.shape == yt.shape == (551, 2), "predict_frozen_segment shape"
    print("predict_frozen_segment OK")
    print("\nlib_planb self-check PASSED")
