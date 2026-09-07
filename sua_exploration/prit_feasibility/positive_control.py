"""Positive control for our PRI-T transplant, on the upstream repo's own example data.

Purpose: if target inference scores poorly on our DANDI decoder outputs, we must be able to
distinguish "PRI-T does not transfer to our offline setting" from "we implemented PRI-T
wrong".  This script runs the identical inference code path against
`prit_upstream/examples/exampledat.mat`, which ships ground-truth `targetPos` alongside the
cursor kinematics, and reports how well inference recovers those targets.

Note on the label constraint: the constraint that forbids reading measured kinematics applies
to our DANDI 000688 evaluation, where the cursor stream would leak the label we claim to
infer.  This file reads kinematics only from the upstream repo's bundled demo array, which is
their data and carries no claim of ours.  No DANDI file is opened here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.io as sio

import fast_prit
import prit_gate

_HERE = Path(__file__).resolve().parent
N_DIRECTIONS = 8
CANONICAL = np.array([-3.0 * math.pi / 4.0 + k * (math.pi / 4.0) for k in range(N_DIRECTIONS)])


def main() -> None:
    mat = sio.loadmat(_HERE / "prit_upstream" / "examples" / "exampledat.mat")
    cursor_pos = np.asarray(mat["cursorPos"], dtype=float)
    cursor_vel = np.asarray(mat["cursorVel"], dtype=float)
    target_pos = np.asarray(mat["targetPos"], dtype=float)
    print("description:", str(mat["description"][0])[:400])
    print(f"pos range x=[{cursor_pos[:,0].min():.3f},{cursor_pos[:,0].max():.3f}] "
          f"y=[{cursor_pos[:,1].min():.3f},{cursor_pos[:,1].max():.3f}]")
    print(f"n_unique_targets={np.unique(target_pos, axis=0).shape[0]}  T={cursor_pos.shape[0]}")

    # Their own recommended defaults, and the same grid construction our gate uses.
    results = []
    for grid_size in (20,):
        locs, _ = prit_gate.build_states(f"grid{grid_size}", prit_gate.TARGET_RADIUS)
        # exampledat lives on a screen normalized differently from our pseudo-trajectory;
        # rescale the grid to the observed cursor extent so states cover the workspace.
        span = np.abs(cursor_pos).max()
        locs_scaled = locs / np.abs(locs).max() * span
        for kappa in (2.0, 4.0, 8.0):
            for stay in (0.999,):
                hmm = prit_gate.make_hmm(
                    locs_scaled, stay, kappa, 0.10 * 2 * span, 32.0 / (2 * span), True,
                )
                obs_log = fast_prit.emission_log_prob(hmm, cursor_vel, cursor_pos)
                states = fast_prit.viterbi(obs_log, stay, locs_scaled.shape[0])
                inferred_loc = locs_scaled[states.astype(int)]

                # Per-timestep target-location error, normalized by workspace span.
                loc_err = np.linalg.norm(inferred_loc - target_pos, axis=1) / (2 * span)

                # Direction agreement: angle from cursor to inferred target vs to true target.
                to_inferred = inferred_loc - cursor_pos
                to_true = target_pos - cursor_pos
                far = np.linalg.norm(to_true, axis=1) > 0.05 * span
                ang_inf = np.arctan2(to_inferred[far, 1], to_inferred[far, 0])
                ang_true = np.arctan2(to_true[far, 1], to_true[far, 0])
                delta = (ang_inf - ang_true + math.pi) % (2 * math.pi) - math.pi
                row = {
                    "grid_size": grid_size,
                    "vm_kappa": kappa,
                    "stay_prob": stay,
                    "median_target_loc_err_frac_of_span": float(np.median(loc_err)),
                    "frac_timesteps_within_10pct_span": float(np.mean(loc_err < 0.10)),
                    "median_abs_direction_err_deg": float(np.median(np.abs(np.degrees(delta)))),
                    "frac_direction_within_22p5_deg": float(
                        np.mean(np.abs(np.degrees(delta)) < 22.5)
                    ),
                    "lambda_expected_cos_delta": float(np.mean(np.cos(delta))),
                }
                results.append(row)
                print(f"  grid{grid_size} kappa={kappa} stay={stay}: "
                      f"median_loc_err={row['median_target_loc_err_frac_of_span']:.4f} span "
                      f"within10%={row['frac_timesteps_within_10pct_span']:.4f} "
                      f"median_dir_err={row['median_abs_direction_err_deg']:.2f}deg "
                      f"dir_within_22.5deg={row['frac_direction_within_22p5_deg']:.4f} "
                      f"lambda={row['lambda_expected_cos_delta']:.4f}")

    best = max(results, key=lambda r: r["frac_direction_within_22p5_deg"])
    out = {
        "source": "prit_upstream/examples/exampledat.mat (upstream repo demo data)",
        "reads_dandi_kinematics": False,
        "n_timesteps": int(cursor_pos.shape[0]),
        "all": results,
        "best": best,
    }
    (_HERE / "positive_control.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nbest: dir_within_22.5deg={best['frac_direction_within_22p5_deg']:.4f} "
          f"lambda={best['lambda_expected_cos_delta']:.4f}")
    print(f"Saved {_HERE / 'positive_control.json'}")


if __name__ == "__main__":
    main()
