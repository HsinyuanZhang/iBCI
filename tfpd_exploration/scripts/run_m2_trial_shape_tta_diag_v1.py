#!/usr/bin/env python3
"""Trial-shaped TTA diagnostic V1 (the honest cut).

Held-in calib recordings, 33-trial seed, query = trials 34+ of the SAME
recording.  One label-free segmenter: silence-gated trial-shaped fragments
from the raw 20 ms stream, resampled to 100 bins, pushed into the pool.
Compare same-surface static (seed-only identity) vs TTA.

Pre-registered single segmenter configuration (no sweep):
  active bin      : bin rate >= 0.25 x median(seed-row mean rates)
  fragment        : maximal active run, gaps <= 25 bins tolerated,
                    length clamped to [20, 300] bins
  commit          : fragment closes (silence or cap) -> resample to 100 bins
                    (linear) -> push;  decode-before-commit always
  capacity        : seed 33 rows fixed; pushed rows accumulate (no eviction),
                    commit counts disclosed

Pass condition (pre-registered): mean(TTA - static) >= +0.01 over the 7
held-in sessions with majority positive, else the online-update route on
this encoder is closed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tfpd_exploration/results/m2_trial_shape_tta_diag_v1"
PKG = ROOT / "tfpd_exploration/submissions/evalai_m2_cenat_chunk100e_v1"
AJPF_C_TRAINING = ROOT / "tfpd_exploration/results/m2_ajpf_c_v2"

CHECKPOINT_BODIES = {
    "pooled": "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e",
    "Ce-NAT": "c5672a2bfc94f55729d64e5eeb2f6917162c345713e7102dec4ad371cd147ea2",
}
SEED_TRIALS = 33
TRIAL_BINS = 100
GAP_TOLERANCE = 25
MIN_FRAGMENT = 20
MAX_FRAGMENT = 300
ACTIVE_FACTOR = 0.25
R2_ABS_TOLERANCE = 1.0e-7


class DiagError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resample_100(fragment: np_module) -> np_module:
    """Linear resample of a [T,96] fragment to [100,96] (per channel)."""
    import numpy as np

    t = fragment.shape[0]
    if t == TRIAL_BINS:
        return np.ascontiguousarray(fragment, dtype=np.float32)
    pos = np.linspace(0.0, t - 1.0, TRIAL_BINS)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, t - 1)
    frac = (pos - lo).reshape(-1, 1)
    return np.ascontiguousarray(
        fragment[lo] * (1.0 - frac) + fragment[hi] * frac, dtype=np.float32)


def main() -> None:
    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "diagnostics must run CPU-only")

    for path in (ROOT, PKG, ROOT / "streaming_calibration_exp",
                 ROOT / "tfpd_exploration/submissions/evalai_m2_apfg_static_v1"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import src.models.streaming_calibration_module  # noqa: F401
    sys.path.append(str(ROOT / "SPINT-main"))

    import copy
    import io

    import numpy as np
    import torch

    from falcon_challenge.config import FalconConfig, FalconTask
    from cenat_chunk_decoder import PayloadIdEncoder
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data, manual_decode
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import variance_weighted_r2

    torch.set_num_threads(min(8, torch.get_num_threads()))
    started = time.monotonic()

    base, data_module, _tc, meta = load_frozen_model_and_data()
    require(meta["checkpoint_sha256"] == CHECKPOINT_BODIES["pooled"], "base checkpoint drift")
    cenat_body = AJPF_C_TRAINING / "checkpoints" / "Ce-NAT_epoch12.pt"
    require(sha256_file(cenat_body) == CHECKPOINT_BODIES["Ce-NAT"], "Ce-NAT body drift")
    cenat = copy.deepcopy(base)
    cenat.load_state_dict(
        torch.load(io.BytesIO(cenat_body.read_bytes()), map_location="cpu", weights_only=True),
        strict=True)
    cenat.eval()
    for parameter in cenat.parameters():
        parameter.requires_grad_(False)
    cenat_decoder = cenat.student.decoder.cpu().eval()
    cenat_encoder = PayloadIdEncoder()
    cenat_encoder.load_state_dict(copy.deepcopy(cenat.student.id_encoder.state_dict()), strict=True)
    cenat_encoder.eval()

    with (PKG / "artifacts/t4_m2_seed42_cenat_chunk100e_tta.pkl").open("rb") as handle:
        import pickle
        payload = pickle.load(handle)
    payload_decoder = payload["decoder"]

    dataset = data_module.train_dataset
    sessions = sorted(dataset.calib_trialized_neural_features)
    require(len(sessions) == 7, "held-in session drift")

    report = {"schema": "m2_trial_shape_tta_diag_v1",
              "segmenter": {"active_factor": ACTIVE_FACTOR, "gap_tolerance": GAP_TOLERANCE,
                            "min_fragment": MIN_FRAGMENT, "max_fragment": MAX_FRAGMENT,
                            "seed_trials": SEED_TRIALS, "resample": "linear_to_100"},
              "sessions": {}}
    deltas = []
    for session in sessions:
        angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
        calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
        seed_rows = [np.ascontiguousarray(row, dtype=np.float32) for row in calibration[:SEED_TRIALS]]
        laws_mod = __import__("laws")
        side, _ = laws_mod.sealed_fit_ridge_side(
            dataset, session, laws_mod.select_dopt4_support(angles))
        side_t = torch.from_numpy(np.ascontiguousarray(side)).unsqueeze(0)

        neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
        targets = np.asarray(dataset.covariate_data[session], dtype=np.float32)
        trial_starts = np.asarray(dataset.trial_start_indices[session], dtype=np.int64)
        require(trial_starts.size >= SEED_TRIALS + 1, f"{session}: trial metadata drift")
        query_origin = int(trial_starts[SEED_TRIALS])  # first bin of trial 34

        all_starts = np.asarray(
            [x for name, x in dataset.window_indices if name == session], dtype=np.int64)
        starts = all_starts[all_starts >= query_origin - 49]
        require(starts.size > 0, f"{session}: no query windows")
        windows = np.ascontiguousarray(
            neural[starts[:, None] + np.arange(50)[None, :]], dtype=np.float32)
        targets_q = np.ascontiguousarray(targets[starts + 49], dtype=np.float32)

        # segmenter threshold from the seed rows (label-free)
        seed_rates = np.asarray([float(r.mean(dtype=np.float64)) for r in seed_rows])
        tau_active = float(ACTIVE_FACTOR * np.median(seed_rates))
        rate_per_bin = neural.mean(axis=1)

        # segment the stream from query_origin
        fragments: list = []
        active = rate_per_bin >= tau_active
        t = query_origin
        frag_start = None
        last_active = None
        while t < neural.shape[0]:
            if active[t]:
                if frag_start is None:
                    frag_start = t
                last_active = t
            elif frag_start is not None and (t - last_active) > GAP_TOLERANCE:
                length = last_active - frag_start + 1
                if length >= MIN_FRAGMENT:
                    fragments.append((frag_start, min(last_active, frag_start + MAX_FRAGMENT - 1)))
                frag_start = None
            t += 1
        if frag_start is not None:
            length = last_active - frag_start + 1
            if length >= MIN_FRAGMENT:
                fragments.append((frag_start, min(last_active, frag_start + MAX_FRAGMENT - 1)))

        # static arm: seed-only identity
        pool = [r.copy() for r in seed_rows]
        with torch.inference_mode():
            identity = cenat_encoder.forward_batch(
                torch.from_numpy(np.stack(pool)).unsqueeze(0), side_features=side_t)
            static_preds = manual_decode(
                cenat_decoder, torch.from_numpy(windows), identity)[:, -1, :].numpy() / 5.0

        # TTA arm: batch-1 bin-by-bin decode-before-commit, trial-shaped pushes
        pool = [r.copy() for r in seed_rows]
        with torch.inference_mode():
            identity = cenat_encoder.forward_batch(
                torch.from_numpy(np.stack(pool)).unsqueeze(0), side_features=side_t)
        commits: list = []
        frag_iter = list(fragments)
        frag_idx = 0
        tta_preds = np.zeros((int(neural.shape[0]), 2), dtype=np.float32)
        for t in range(int(neural.shape[0])):
            with torch.inference_mode():
                window = np.ascontiguousarray(neural[max(0, t - 49): t + 1], dtype=np.float32)
                w = np.zeros((50, neural.shape[1]), dtype=np.float32)
                w[50 - window.shape[0]:] = window
                tta_preds[t] = manual_decode(
                    cenat_decoder, torch.from_numpy(w).unsqueeze(0), identity)[0, -1, :].numpy() / 5.0
            # after decode: close any fragment that ends at this bin
            while frag_idx < len(frag_iter) and frag_iter[frag_idx][1] == t:
                lo, hi = frag_iter[frag_idx]
                pushed = resample_100(neural[lo : hi + 1])
                pool.append(pushed)
                commits.append(t)
                with torch.inference_mode():
                    identity = cenat_encoder.forward_batch(
                        torch.from_numpy(np.stack(pool)).unsqueeze(0), side_features=side_t)
                frag_idx += 1

        tta_r2 = float(variance_weighted_r2(targets_q, tta_preds[starts]))
        static_r2 = float(variance_weighted_r2(targets_q, static_preds))
        delta = tta_r2 - static_r2
        deltas.append(delta)
        report["sessions"][session] = {
            "static_seed33_r2": static_r2, "tta_r2": tta_r2, "delta": delta,
            "fragments_committed": len(commits), "commit_bins": commits,
            "query_windows": int(starts.size), "tau_active": tau_active,
        }
        print(json.dumps({"session": session, "static": round(static_r2, 5),
                          "tta": round(tta_r2, 5), "delta": round(delta, 5),
                          "commits": len(commits)}), flush=True)

    deltas_arr = np.asarray(deltas)
    report["summary"] = {
        "mean_delta": float(deltas_arr.mean()),
        "median_delta": float(np.median(deltas_arr)),
        "positive": int((deltas_arr > 0).sum()),
        "sessions": len(deltas_arr),
        "gate": bool(deltas_arr.mean() >= 0.01 and (deltas_arr > 0).sum() > len(deltas_arr) / 2),
        "wall_seconds": time.monotonic() - started,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target = RESULT_ROOT / "diag.json"
    target.write_bytes(body)
    target.chmod(0o444)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(hashlib.sha256(body).hexdigest() + "  diag.json\n", encoding="ascii")
    sidecar.chmod(0o444)
    print(json.dumps(report["summary"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
