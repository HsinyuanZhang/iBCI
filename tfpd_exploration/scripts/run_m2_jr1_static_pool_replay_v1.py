#!/usr/bin/env python3
"""Static-pool official-surface replay of the AJPF V4 checkpoints.

Question.  AJPF V4's positive held-out result (+0.0131 J-R1 - J-NATIVE, CI
fully positive) was measured under the decode-before-commit growing-pool
protocol.  The official EvalAI M2 surface is a STATIC first-30 calibration
pool with cached identities.  Before spending the last rationed submission on
a J-R1 candidate, this cell measures how much of the V4 method effect survives
on the deployment surface.

Arms (all CPU, one process, same inputs):
  - ``pooled``   : the pretrained Selected-T4 checkpoint (25d7...).  Its
                   replay must reproduce the sealed act30_dopt4 screen rows
                   (bitwise externally; the lineage anchor of the harness).
  - ``J-NATIVE`` : AJPF V2 epoch-12 jointly fine-tuned native checkpoint
                   (state SHA 7b5be4f6...).  Shows what joint fine-tuning
                   alone does to static pools.
  - ``J-R1``     : AJPF V2 epoch-12 jointly trained anchored-gate checkpoint
                   (state SHA 47736531..., learned alpha).  The candidate.

The static identity is computed once per session from the first-30
calibration block under the sealed act30_dopt4 law (greedy D-opt k=4 support,
ridge lambda=0.1 selected-support4 T4 carrier, first-30 label-free pool) --
bit-identical to the officially scored act30_dopt4 deployment law.

Outputs an immutable 0444+sidecar receipt under
``tfpd_exploration/results/m2_jr1_static_pool_replay_v1/``.  Read-only with
respect to every AJPF result root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = ROOT / "tfpd_exploration"
RESULT_ROOT = TFPD_ROOT / "results/m2_jr1_static_pool_replay_v1"
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
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "schema": "m2_jr1_static_pool_replay_dry_v1",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE",
            "arms": ["pooled", "J-NATIVE", "J-R1"],
            "pool": "static_first30_act30_law",
            "surface": "official_windows_13_sessions",
        }, sort_keys=True))
        return

    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "replay must run CPU-only")

    # -- imports (streaming root pinned first, exactly like the validated replay)
    streaming_root = ROOT / "streaming_calibration_exp"
    here = Path(__file__).resolve().parent
    for path in (ROOT, TFPD_ROOT / "submissions/evalai_m2_apfg_static_v1", streaming_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import src.models.streaming_calibration_module  # noqa: F401 - pins `src`
    sys.path.append(str(ROOT / "SPINT-main"))

    import numpy as np
    import torch
    import torch.serialization

    from laws import select_dopt4_support, select_first30_activity_pool  # APFG-S package laws
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1.adapter import install_after_strict_load
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1.physical import _student_state_sha
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts,
        variance_weighted_r2,
    )

    torch.set_num_threads(min(8, torch.get_num_threads()))
    started = time.monotonic()

    model, data_module, task_config, metadata = load_frozen_model_and_data()
    require(metadata["checkpoint_sha256"] == "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e",
            "pretrained checkpoint drift")
    sealed_screen = json.loads((ROOT / SEALED_SCREEN_RELATIVE).read_text(encoding="utf-8"))
    sealed_rows = {
        (row["surface"], row["session"]): row
        for row in sealed_screen["rows"] if row["cell"] == SEALED_CELL
    }
    require(len(sealed_rows) == 13, "sealed act30_dopt4 row coverage drift")

    import copy
    import io

    pooled_module = model
    jnative_module = copy.deepcopy(pooled_module)
    jr1_module = copy.deepcopy(pooled_module)

    # -- load AJPF checkpoints (body SHA + strict load + state SHA proof)
    arm_modules: dict[str, object] = {"pooled": pooled_module}
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
        require(got == CHECKPOINT_BODIES[arm]["student_state_sha256"],
                f"{arm} student state drift: {got}")
        arm_modules[arm] = module

    with torch.no_grad():
        alpha = float(jr1_module.student.id_encoder.alpha.item())
    require(alpha != 0.0, "J-R1 alpha unexpectedly zero")

    # -- per-session static identities and decode replay
    surfaces_spec = (
        ("external_official_query", data_module.val_heldout_dataset, 6),
        ("within_post30", data_module.train_dataset, 7),
    )
    rows: list[dict] = []
    for surface_name, dataset, expected in surfaces_spec:
        require(dataset is not None, f"{surface_name} dataset missing")
        sessions = sorted(dataset.calib_trialized_neural_features)
        require(len(sessions) == expected, f"{surface_name} session count drift")
        for session in sessions:
            angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
            selected = select_dopt4_support(angles)
            calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
            activity30 = select_first30_activity_pool(calibration)
            from laws import sealed_fit_ridge_side
            side, _evidence = sealed_fit_ridge_side(dataset, session, selected)

            support = torch.from_numpy(np.ascontiguousarray(activity30)).unsqueeze(0)
            side_tensor = torch.from_numpy(np.ascontiguousarray(side)).unsqueeze(0)

            starts = np.asarray(
                [s for name, s in dataset.window_indices if name == session], dtype=np.int64
            )
            if surface_name == "within_post30":
                starts = select_common_post30_window_starts(starts, dataset.trial_start_indices[session])
            require(starts.size > 0, f"{session}: no query windows")
            neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
            targets_all = np.asarray(dataset.covariate_data[session], dtype=np.float32)
            windows = np.ascontiguousarray(
                neural[starts[:, None] + np.arange(50)[None, :]], dtype=np.float32
            )
            targets = np.ascontiguousarray(targets_all[starts + 49], dtype=np.float32)

            for arm, module in arm_modules.items():
                student = module.student
                with torch.inference_mode():
                    if arm == "J-R1":
                        identity = student.id_encoder.forward_batch(support, side_features=side_tensor)
                    else:
                        identity = student.compute_identity(support, side_features=side_tensor)
                    pieces = []
                    for offset in range(0, len(windows), 1024):
                        output = student.decode_with_identity(
                            torch.from_numpy(windows[offset:offset + 1024]), identity
                        )
                        pieces.append(output[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False) / 5.0)
                predictions = np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)
                r2 = float(variance_weighted_r2(targets, predictions))
                rows.append({
                    "arm": arm,
                    "surface": surface_name,
                    "session": session,
                    "window_count": int(starts.size),
                    "r2": r2,
                    "sealed_act30_dopt4_r2": float(sealed_rows[(surface_name, session)]["r2"]),
                })

    # -- lineage anchor: pooled static replay must equal the sealed rows
    for row in rows:
        if row["arm"] == "pooled":
            delta = abs(row["r2"] - row["sealed_act30_dopt4_r2"])
            require(delta <= R2_ABS_TOLERANCE,
                    f"lineage anchor drift {row['surface']}/{row['session']}: {delta}")

    # -- summaries
    def arm_surface(arm: str, surface: str):
        return {r["session"]: r["r2"] for r in rows if r["arm"] == arm and r["surface"] == surface}

    summary: dict = {"schema": "m2_jr1_static_pool_replay_v1", "alpha_jr1": alpha,
                     "lineage_anchor_max_abs_delta": max(
                         abs(r["r2"] - r["sealed_act30_dopt4_r2"]) for r in rows if r["arm"] == "pooled"),
                     "surfaces": {}}
    for surface in ("external_official_query", "within_post30"):
        per = {arm: arm_surface(arm, surface) for arm in arm_modules}
        sessions = sorted(per["pooled"])
        sealed = {s: sealed_rows[(surface, s)]["r2"] for s in sessions}
        sealed_values = np.asarray([sealed[s] for s in sessions])
        contrasts = {}
        for name, (a, b) in {
            "jr1_minus_jnative": ("J-R1", "J-NATIVE"),
            "jr1_minus_pooled_sealed": ("J-R1", "pooled"),
            "jnative_minus_pooled_sealed": ("J-NATIVE", "pooled"),
        }.items():
            deltas = np.asarray([per[a][s] - (per[b][s] if b != "pooled" else sealed[s]) for s in sessions])
            contrasts[name] = {
                "per_session": {s: float(d) for s, d in zip(sessions, deltas)},
                **bootstrap_ci(deltas),
            }
        summary["surfaces"][surface] = {
            "equal_session_mean": {arm: float(np.mean([per[arm][s] for s in sessions])) for arm in per},
            "sealed_pooled_mean": float(sealed_values.mean()),
            "contrasts": contrasts,
        }

    summary["wall_seconds"] = time.monotonic() - started

    RESULT_ROOT.mkdir(parents=True, exist_ok=False)
    body = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target = RESULT_ROOT / "replay.json"
    target.write_bytes(body)
    target.chmod(0o444)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(hashlib.sha256(body).hexdigest() + "  replay.json\n", encoding="ascii")
    sidecar.chmod(0o444)

    print(json.dumps(summary["surfaces"], indent=2, sort_keys=True))
    print(json.dumps({"alpha_jr1": alpha, "receipt": str(target)}, sort_keys=True))


if __name__ == "__main__":
    main()
