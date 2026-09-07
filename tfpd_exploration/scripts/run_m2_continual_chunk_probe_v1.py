#!/usr/bin/env python3
"""Continual-legal growing-memory probe for the AJPF checkpoints.

Question.  V4's positive result lives on the decode-before-commit growing-pool
protocol, which the official continual M2 contract cannot drive (no on_done,
no trial boundaries).  The sealed A0 chunk experiment showed that 100-bin
raw-window commits POISON the pretrained native model (chunk < static).  But
A0 used a model that had never seen a chunk row -- the same train/deploy
operator-mismatch class that cost the PF line 0.08-0.09 R2.  This probe asks:
do the JOINTLY TRAINED AJPF checkpoints (whose encoder/decoder/gate
co-adapted to growing pools) tolerate boundary-free chunk commits?

Arms (frozen, CPU, one process, same inputs):
  pooled    pretrained Selected-T4 (25d7...) -- sealed act30_dopt4 anchor
  J-NATIVE  AJPF epoch-12 jointly fine-tuned native (7b5be4f6...)
  J-R1      AJPF epoch-12 anchored gate (47736531..., alpha=-0.1838...)

Pool laws (identical first-30 seed + D-opt4 selected-support4 carrier):
  static    no commits (deployment-identical to the scored act30 images)
  chunk100  A0-faithful: tumbling 100-bin windows of the observed raw
            stream, committed when complete, decoded only from chunks
            completed strictly before the scored endpoint; capacity 30 with
            FIFO eviction over non-support rows (support4 protected,
            matching the AJPF training law).
  chunk100e energy-gated: same geometry, but a window commits only if its
            mean multi-unit rate >= the running median of past candidate
            rates (label-free, causal selectivity against idle bins).

Chunk-stream origin: bin 0 of the eval stream on the external surface
(deployment-realistic: EvalAI streams the whole eval session); the first-30
boundary on the within surface (A0 convention).

Receipt: tfpd_exploration/results/m2_continual_chunk_probe_v1/replay.json
(0444 + sidecar).  Read-only with respect to every AJPF/A0 result root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = ROOT / "tfpd_exploration"
RESULT_ROOT = TFPD_ROOT / "results/m2_continual_chunk_probe_v1"
AJPF_V2_TRAINING = TFPD_ROOT / "results/m2_anchored_joint_postfusion_v2/training"

CHECKPOINT_BODIES = {
    "J-NATIVE": {
        "sha256": "40c46a95b9f4ebd735d3fcd2011dcd2e328fa5bd994e6313ab3efbbeabb164f5",
        "student_state_sha256": "7b5be4f62fc98433bc309ee904188fb289fdb9a1b56ec8e4955b5398e3149de3",
    },
    "J-R1": {
        "sha256": "fd7beac07c3df674639560d724d9be8a798b0af27f0e091c96667086e7b4c64d",
        "student_state_sha256": "47736531ae74e3560ba5f5a074e22a16f05a7c36f05895e0d8b658fd6624ff5d",
    },
}
SEALED_SCREEN_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
SEALED_CELL = "ridge_activity30_m4"
R2_ABS_TOLERANCE = 1.0e-7
CHUNK_BINS = 100
POOL_CAPACITY = 30
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
LAWS = ("static", "chunk100", "chunk100e")


class ProbeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_ci(values, seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = values[rng.integers(0, values.size, values.size)].mean()
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive": int((values > 0).sum()),
        "n": int(values.size),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def chunk_commit_bins(neural, origin, energy_gated):
    """Walk raw bins from `origin`; return bins at which a 100-bin chunk commits.

    A bin index b in the returned array means: the window [b-99, b] completed
    at bin b and was committed to the pool.  Causality for consumers: the chunk
    is available to any decode at endpoint t > b (strictly before, A0 law).
    """
    import numpy as np

    total = int(neural.shape[0])
    commits = []
    rates = []
    position = origin
    while position + CHUNK_BINS <= total:
        window = neural[position : position + CHUNK_BINS]
        rate = float(window.mean(dtype=np.float64))
        if energy_gated and rates:
            commit = rate >= float(np.median(np.asarray(rates, dtype=np.float64)))
        else:
            commit = True
        if commit:
            commits.append(position + CHUNK_BINS - 1)
            rates.append(rate)
        else:
            rates.append(rate)  # candidate seen, not committed; still updates the median
        position += CHUNK_BINS
    return commits


class SeededPool:
    """First-30 seed with support4-protected FIFO capacity 30 (AJPF law)."""

    def __init__(self, seed_rows, support_count):
        self.rows = deque(seed_rows)
        self.support_count = int(support_count)
        require(len(self.rows) <= POOL_CAPACITY, "seed exceeds pool capacity")

    def commit(self, row):
        self.rows.append(row)
        if len(self.rows) > POOL_CAPACITY:
            # evict the oldest NON-support row (index support_count; the
            # first `support_count` seed rows are never removed)
            del self.rows[self.support_count]

    def stack(self):
        import numpy as np

        return np.ascontiguousarray(np.stack(list(self.rows)), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "schema": "m2_continual_chunk_probe_dry_v1",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE",
            "arms": ["pooled", "J-NATIVE", "J-R1"],
            "laws": list(LAWS),
            "capacity": POOL_CAPACITY,
            "chunk_bins": CHUNK_BINS,
        }, sort_keys=True))
        return

    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "probe must run CPU-only")

    streaming_root = ROOT / "streaming_calibration_exp"
    for path in (ROOT, TFPD_ROOT / "submissions/evalai_m2_apfg_static_v1", streaming_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import src.models.streaming_calibration_module  # noqa: F401 - pins `src`
    sys.path.append(str(ROOT / "SPINT-main"))

    import copy
    import io

    import numpy as np
    import torch

    from laws import sealed_fit_ridge_side, select_dopt4_support, select_first30_activity_pool
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1.adapter import install_after_strict_load
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1.physical import _student_state_sha
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts,
        variance_weighted_r2,
    )

    torch.set_num_threads(min(8, torch.get_num_threads()))
    started = time.monotonic()

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    require(metadata["checkpoint_sha256"] == "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e",
            "pretrained checkpoint drift")
    sealed_screen = json.loads((ROOT / SEALED_SCREEN_RELATIVE).read_text(encoding="utf-8"))
    sealed_rows = {
        (row["surface"], row["session"]): row
        for row in sealed_screen["rows"] if row["cell"] == SEALED_CELL
    }
    require(len(sealed_rows) == 13, "sealed act30_dopt4 row coverage drift")

    pooled_module = model
    jnative_module = copy.deepcopy(pooled_module)
    jr1_module = copy.deepcopy(pooled_module)
    arm_modules = {"pooled": pooled_module}
    for arm in ("J-NATIVE", "J-R1"):
        body_path = AJPF_V2_TRAINING / "checkpoints" / f"{arm}_epoch12.pt"
        digest = sha256_file(body_path)
        require(digest == CHECKPOINT_BODIES[arm]["sha256"], f"{arm} checkpoint body drift: {digest}")
        state = torch.load(io.BytesIO(body_path.read_bytes()), map_location="cpu", weights_only=True)
        module = jnative_module if arm == "J-NATIVE" else jr1_module
        if arm == "J-R1":
            install_after_strict_load(module.student, "J-R1")
        module.load_state_dict(state, strict=True)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        got = _student_state_sha(module)
        require(got == CHECKPOINT_BODIES[arm]["student_state_sha256"], f"{arm} student state drift: {got}")
        arm_modules[arm] = module

    with torch.no_grad():
        alpha = float(jr1_module.student.id_encoder.alpha.item())

    surfaces_spec = (
        ("external_official_query", data_module.val_heldout_dataset, 6, "external"),
        ("within_post30", data_module.train_dataset, 7, "within"),
    )
    rows_out = []
    for surface_name, dataset, expected, short in surfaces_spec:
        require(dataset is not None, f"{surface_name} dataset missing")
        sessions = sorted(dataset.calib_trialized_neural_features)
        require(len(sessions) == expected, f"{surface_name} session count drift")
        for session in sessions:
            angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
            selected = select_dopt4_support(angles)
            calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
            seed_rows = [np.ascontiguousarray(row, dtype=np.float32)
                         for row in select_first30_activity_pool(calibration)]
            side, _evidence = sealed_fit_ridge_side(dataset, session, selected)
            side_tensor = torch.from_numpy(np.ascontiguousarray(side)).unsqueeze(0)

            starts = np.asarray(
                [s for name, s in dataset.window_indices if name == session], dtype=np.int64
            )
            if surface_name == "within_post30":
                starts = select_common_post30_window_starts(starts, dataset.trial_start_indices[session])
            require(starts.size > 0, f"{session}: no query windows")
            neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
            targets_all = np.asarray(dataset.covariate_data[session], dtype=np.float32)
            endpoints = starts + 49
            windows = np.ascontiguousarray(neural[starts[:, None] + np.arange(50)[None, :]], dtype=np.float32)
            targets = np.ascontiguousarray(targets_all[starts + 49], dtype=np.float32)

            origin = 0
            if short == "within":
                trial_starts = np.asarray(dataset.trial_start_indices[session], dtype=np.int64)
                require(trial_starts.size > 30, f"{session}: trial metadata drift")
                origin = int(trial_starts[30])
            commit_bins = {
                "static": [],
                "chunk100": chunk_commit_bins(neural, origin, energy_gated=False),
                "chunk100e": chunk_commit_bins(neural, origin, energy_gated=True),
            }

            for law in LAWS:
                law_commits = commit_bins[law]
                # number of chunks committed strictly before each endpoint
                committed_before = np.searchsorted(np.asarray(law_commits, dtype=np.int64), endpoints, side="left")
                # partition endpoints into contiguous segments of equal commit count
                for arm, module in arm_modules.items():
                    student = module.student
                    predictions = np.zeros((starts.size, 2), dtype=np.float32)
                    segment_start = 0
                    pool = SeededPool(seed_rows, support_count=len(selected))
                    committed_upto = 0
                    identity = None
                    identity_count = -1
                    while segment_start < starts.size:
                        stop = segment_start
                        while stop < starts.size and committed_before[stop] == committed_before[segment_start]:
                            stop += 1
                        count = int(committed_before[segment_start])
                        while committed_upto < count:  # advance the pool incrementally
                            commit_bin = law_commits[committed_upto]
                            row_start = commit_bin - CHUNK_BINS + 1
                            pool.commit(np.ascontiguousarray(neural[row_start : commit_bin + 1], dtype=np.float32))
                            committed_upto += 1
                        if count != identity_count:  # pool changed (or first segment)
                            support = torch.from_numpy(pool.stack()).unsqueeze(0)
                            with torch.inference_mode():
                                if arm == "J-R1":
                                    identity = student.id_encoder.forward_batch(support, side_features=side_tensor)
                                else:
                                    identity = student.compute_identity(support, side_features=side_tensor)
                            identity_count = count
                        with torch.inference_mode():
                            for offset in range(segment_start, stop, 1024):
                                batch = torch.from_numpy(windows[offset : min(offset + 1024, stop)])
                                output = student.decode_with_identity(batch, identity)
                                predictions[offset : min(offset + 1024, stop)] = (
                                    output[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False) / 5.0
                                )
                        segment_start = stop
                    r2 = float(variance_weighted_r2(targets, predictions))
                    rows_out.append({
                        "arm": arm, "law": law, "surface": surface_name, "session": session,
                        "window_count": int(starts.size),
                        "commits_used": int(len(law_commits)) if law != "static" else 0,
                        "r2": r2,
                        "sealed_act30_dopt4_r2": float(sealed_rows[(surface_name, session)]["r2"]),
                    })
                    print(json.dumps(rows_out[-1], sort_keys=True), flush=True)

    for row in rows_out:
        if row["arm"] == "pooled" and row["law"] == "static":
            delta = abs(row["r2"] - row["sealed_act30_dopt4_r2"])
            require(delta <= R2_ABS_TOLERANCE,
                    f"lineage anchor drift {row['surface']}/{row['session']}: {delta}")

    def arm_law_surface(arm, law, surface):
        return {r["session"]: r["r2"] for r in rows_out
                if r["arm"] == arm and r["law"] == law and r["surface"] == surface}

    summary = {"schema": "m2_continual_chunk_probe_v1", "alpha_jr1": alpha,
               "laws": list(LAWS), "capacity": POOL_CAPACITY, "chunk_bins": CHUNK_BINS,
               "lineage_anchor_max_abs_delta": max(
                   abs(r["r2"] - r["sealed_act30_dopt4_r2"]) for r in rows_out
                   if r["arm"] == "pooled" and r["law"] == "static"),
               "surfaces": {}, "rows": rows_out}
    for surface in ("external_official_query", "within_post30"):
        sessions = sorted({r["session"] for r in rows_out if r["surface"] == surface})
        sealed = {s: sealed_rows[(surface, s)]["r2"] for s in sessions}
        block = {"equal_session_mean": {}, "contrasts": {}}
        for arm in arm_modules:
            for law in LAWS:
                per = arm_law_surface(arm, law, surface)
                require(set(per) == set(sessions), f"coverage drift {arm}/{law}/{surface}")
                block["equal_session_mean"][f"{arm}/{law}"] = float(np.mean([per[s] for s in sessions]))
        contrasts = {}
        for law in LAWS:
            for name, (a_arm, a_law, b_arm, b_law) in {
                f"jr1_{law}_minus_pooled_static": ("J-R1", law, "pooled", "static"),
                f"jr1_{law}_minus_jnative_{law}": ("J-R1", law, "J-NATIVE", law),
                f"jnative_{law}_minus_pooled_static": ("J-NATIVE", law, "pooled", "static"),
                f"pooled_{law}_minus_pooled_static": ("pooled", law, "pooled", "static"),
            }.items():
                left = arm_law_surface(a_arm, a_law, surface)
                right = arm_law_surface(b_arm, b_law, surface)
                deltas = np.asarray([left[s] - right[s] for s in sessions])
                contrasts[name] = {"per_session": {s: float(d) for s, d in zip(sessions, deltas)},
                                   **bootstrap_ci(deltas)}
        block["contrasts"] = contrasts
        summary["surfaces"][surface] = block

    summary["wall_seconds"] = time.monotonic() - started
    RESULT_ROOT.mkdir(parents=True, exist_ok=False)
    body = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target = RESULT_ROOT / "replay.json"
    target.write_bytes(body)
    target.chmod(0o444)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(hashlib.sha256(body).hexdigest() + "  replay.json\n", encoding="ascii")
    sidecar.chmod(0o444)

    slim = {surface: {"equal_session_mean": block["equal_session_mean"],
                      "contrasts": {k: {kk: vv for kk, vv in v.items() if kk != "per_session"}
                                    for k, v in block["contrasts"].items()}}
            for surface, block in summary["surfaces"].items()}
    print(json.dumps(slim, indent=2, sort_keys=True))
    print(json.dumps({"alpha_jr1": alpha, "receipt": str(target)}, sort_keys=True))


if __name__ == "__main__":
    main()
