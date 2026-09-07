"""PRI-T target inference over SPINT's own decoded velocity: the feasibility gate.

Uses the upstream ``PRIT.prit.HMMRecalibration`` class verbatim (cloned from
github.com/guyhwilson/PRI-T) so the von Mises observation model, the Viterbi search and
the forward-backward marginals are the authors' own code, not a reimplementation.

Observations handed to the HMM:
  * ``cursorVel`` = the frozen SPINT decoder's own output velocity (per 20 ms window).
  * ``cursorPos`` = the running integral of that same decoded velocity, starting at the
    workspace origin at the first decodable bin of each trial.

The NWB ``cursor_vel`` / ``cursor_pos`` / ``cursor_acc`` streams of the target sessions are
never read (see decode_forward.py).  ``target_dir`` enters only as the scoring label.

Hidden states are candidate target *locations*, which are task structure:
  * ``task8``  -- the 8 canonical center-out targets at the measured radius.
  * ``task9``  -- those 8 plus the workspace centre.
  * ``gridN``  -- the paper's default NxN screen discretization (their recommended >= 20).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "prit_upstream"))
from PRIT.prit import HMMRecalibration  # noqa: E402

import fast_prit  # noqa: E402

N_DIRECTIONS = 8
CANONICAL_DIRECTIONS_RAD = np.array(
    [-3.0 * math.pi / 4.0 + k * (math.pi / 4.0) for k in range(N_DIRECTIONS)]
)
BIN_SIZE_S = 0.020
TARGET_RADIUS = 8.0  # from target_corners geometry; every finite row has radius exactly 8
SESSIONS = [
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
]


def snap_to_direction_index(angles: np.ndarray) -> np.ndarray:
    delta = (CANONICAL_DIRECTIONS_RAD[None, :] - np.asarray(angles)[:, None] + math.pi)
    delta = delta % (2.0 * math.pi) - math.pi
    return np.argmin(np.abs(delta), axis=1)


def build_states(mode: str, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (target_locations [S,2], direction_index_per_state [S])."""
    if mode in {"task8", "task9"}:
        locs = radius * np.stack(
            [np.cos(CANONICAL_DIRECTIONS_RAD), np.sin(CANONICAL_DIRECTIONS_RAD)], axis=1
        )
        dir_index = np.arange(N_DIRECTIONS)
        if mode == "task9":
            locs = np.vstack([locs, np.zeros((1, 2))])
            dir_index = np.concatenate([dir_index, [-1]])
        return locs, dir_index
    if mode.startswith("grid"):
        # Same meshgrid construction as upstream generateTargetGrid; called inline because
        # that helper's own assertions reject any explicit x_bounds/y_bounds, so only its
        # is_simulated [-0.5, 0.5] box is reachable through the public signature.
        grid_size = int(mode[4:])
        extent = 1.25 * radius
        x_loc, y_loc = np.meshgrid(
            np.linspace(-extent, extent, grid_size),
            np.linspace(-extent, extent, grid_size),
        )
        locs = np.vstack([np.ravel(x_loc), np.ravel(y_loc)]).T
        dir_index = snap_to_direction_index(np.arctan2(locs[:, 1], locs[:, 0]))
        # A state at the exact origin has no direction; mark it unusable.
        dir_index[np.hypot(locs[:, 0], locs[:, 1]) < 1e-9] = -1
        return locs, dir_index
    raise ValueError(f"unknown state mode {mode!r}")


def smooth_velocity(vel: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return vel
    kernel = np.ones(width) / width
    padded = np.pad(vel, ((width, width), (0, 0)), mode="edge")
    out = np.stack(
        [np.convolve(padded[:, c], kernel, mode="same") for c in range(vel.shape[1])],
        axis=1,
    )
    return out[width : width + vel.shape[0]]


def make_hmm(locs: np.ndarray, stay_prob: float, kappa: float,
             inflection: float, exponent: float, use_adjust: bool) -> HMMRecalibration:
    n_states = locs.shape[0]
    transitions = np.eye(n_states) * stay_prob
    for row in range(n_states):
        other = np.setdiff1d(np.arange(n_states), row)
        transitions[row, other] = (1.0 - stay_prob) / (n_states - 1)
    prior = np.zeros((n_states, 1)) + 1.0 / n_states
    adjust = None
    if use_adjust:
        adjust = lambda x: 1.0 / (1.0 + np.exp(-1.0 * (x - inflection) * exponent))  # noqa: E731
    return HMMRecalibration(transitions, locs, prior, kappa, adjustKappa=adjust)


def run_config(
    decoded: dict[str, dict],
    state_mode: str,
    kappa: float,
    smooth_width: int,
    gain: float,
    stay_prob: float,
    use_adjust: bool,
    radius: float,
    speed_gate: float = 0.0,
    engine: str = "fast",
) -> dict:
    locs, dir_index = build_states(state_mode, radius)
    n_states = locs.shape[0]
    inflection = 0.10 * (2.0 * radius)   # paper: ~10% of screen width
    exponent = 32.0 / (2.0 * radius)     # paper's slope, rescaled to our units
    hmm = make_hmm(locs, stay_prob, kappa, inflection, exponent, use_adjust)

    per_session = {}
    for name in SESSIONS:
        record = decoded[name]
        n_correct_vit = n_correct_occ = n_correct_vote = n_total = 0
        angle_errors = []
        for vel_raw, true_dir in zip(record["vel"], record["true_dir"]):
            if not np.isfinite(true_dir):
                continue
            true_index = int(snap_to_direction_index(np.array([true_dir]))[0])
            pseudo_pos = gain * BIN_SIZE_S * np.cumsum(vel_raw, axis=0)
            pseudo_pos = np.vstack([np.zeros((1, 2)), pseudo_pos[:-1]])
            observed_vel = smooth_velocity(vel_raw, smooth_width)

            if speed_gate > 0.0:
                # Decoder-output-only reliability gate: keep the fastest timesteps, where
                # the decoded direction is meaningful.  Uses no measured kinematics.
                speed = np.hypot(observed_vel[:, 0], observed_vel[:, 1])
                keep = speed >= np.percentile(speed, 100.0 * speed_gate)
                if keep.sum() < 5:
                    keep = np.ones_like(keep)
                observed_vel = observed_vel[keep]
                pseudo_pos = pseudo_pos[keep]

            if engine == "upstream":
                viterbi, occupancy = hmm.predict([pseudo_pos], [observed_vel])
                states = viterbi.astype(int)
            else:
                obs_log = fast_prit.emission_log_prob(hmm, observed_vel, pseudo_pos)
                states = fast_prit.viterbi(obs_log, stay_prob, n_states)
                occupancy = fast_prit.forward_backward(obs_log, stay_prob, n_states)

            # Viterbi: modal state over the trial, restricted to direction-bearing states.
            usable = dir_index[states] >= 0
            if usable.any():
                counts = np.bincount(dir_index[states][usable], minlength=N_DIRECTIONS)
                pred_vit = int(np.argmax(counts))
            else:
                pred_vit = -1
            # Marginals: total occupancy mass per direction across the trial.
            mass = np.zeros(N_DIRECTIONS)
            for d in range(N_DIRECTIONS):
                columns = np.where(dir_index == d)[0]
                if columns.size:
                    mass[d] = occupancy[:, columns].sum()
            pred_occ = int(np.argmax(mass))
            # HMM-free reference: nearest canonical direction of the mean decoded velocity.
            mean_vel = vel_raw.mean(axis=0)
            pred_vote = int(snap_to_direction_index(
                np.array([math.atan2(mean_vel[1], mean_vel[0])])
            )[0])

            n_correct_vit += pred_vit == true_index
            n_correct_occ += pred_occ == true_index
            n_correct_vote += pred_vote == true_index
            n_total += 1
            delta = (CANONICAL_DIRECTIONS_RAD[pred_vit] - true_dir + math.pi) % (2 * math.pi) - math.pi
            angle_errors.append(abs(math.degrees(delta)))

        per_session[name] = {
            "n_trials": n_total,
            "viterbi_accuracy": n_correct_vit / n_total,
            "occupancy_accuracy": n_correct_occ / n_total,
            "mean_velocity_vote_accuracy": n_correct_vote / n_total,
            "median_abs_angle_error_deg": float(np.median(angle_errors)),
            "majority_class_rate": record["majority_class_rate"],
        }
    return {
        "state_mode": state_mode,
        "n_states": int(locs.shape[0]),
        "vm_kappa": kappa,
        "smooth_width_bins": smooth_width,
        "velocity_gain": gain,
        "stay_prob": stay_prob,
        "adjust_kappa": use_adjust,
        "speed_gate_quantile": speed_gate,
        "engine": engine,
        "target_radius": radius,
        "kappa_logistic_inflection": inflection,
        "kappa_logistic_exponent": exponent,
        "per_session": per_session,
        "mean_viterbi_accuracy": float(
            np.mean([v["viterbi_accuracy"] for v in per_session.values()])
        ),
        "mean_occupancy_accuracy": float(
            np.mean([v["occupancy_accuracy"] for v in per_session.values()])
        ),
        "mean_vote_accuracy": float(
            np.mean([v["mean_velocity_vote_accuracy"] for v in per_session.values()])
        ),
    }


def load_decoded(tag: str) -> dict[str, dict]:
    out = {}
    for name in SESSIONS:
        data = np.load(_HERE / "decoded" / f"{name}__{tag}.npz")
        indices = data["trial_indices"]
        true_dir = data["true_dir"]
        finite = np.isfinite(true_dir)
        snapped = snap_to_direction_index(true_dir[finite])
        counts = np.bincount(snapped, minlength=N_DIRECTIONS)
        out[name] = {
            "vel": [data[f"vel_{i}"].astype(np.float64) for i in indices],
            "true_dir": true_dir,
            "majority_class_rate": float(counts.max() / counts.sum()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", default="t4,z4,t4_zeroid")
    parser.add_argument("--state_modes", default="task8,task9,grid20")
    parser.add_argument("--kappas", default="0.25,1,2,4,8")
    parser.add_argument("--smooth_widths", default="1,5,25")
    parser.add_argument("--gains", default="1.0")
    parser.add_argument("--stay_probs", default="0.999")
    parser.add_argument("--adjust", default="1,0")
    parser.add_argument("--speed_gates", default="0.0")
    parser.add_argument("--engine", default="fast", choices=["fast", "upstream"])
    parser.add_argument("--out", default="gate_results.json")
    args = parser.parse_args()

    tags = args.tags.split(",")
    grid = list(itertools.product(
        args.state_modes.split(","),
        [float(v) for v in args.kappas.split(",")],
        [int(v) for v in args.smooth_widths.split(",")],
        [float(v) for v in args.gains.split(",")],
        [float(v) for v in args.stay_probs.split(",")],
        [bool(int(v)) for v in args.adjust.split(",")],
        [float(v) for v in args.speed_gates.split(",")],
    ))

    results = {
        "created_at": datetime.now().astimezone().isoformat(),
        "chance_accuracy_8_class": 1.0 / N_DIRECTIONS,
        "bin_size_s": BIN_SIZE_S,
        "target_radius": TARGET_RADIUS,
        "prit_source": "github.com/guyhwilson/PRI-T (cloned; HMMRecalibration used verbatim)",
        "observation_model": "von Mises on angle(decoded velocity, target - integrated decoded position)",
        "target_session_kinematics_used": "none",
        "configs": {},
    }
    for tag in tags:
        decoded = load_decoded(tag)
        rows = []
        for state_mode, kappa, smooth, gain, stay, adjust, gate in grid:
            row = run_config(
                decoded, state_mode, kappa, smooth, gain, stay, adjust, TARGET_RADIUS,
                speed_gate=gate, engine=args.engine,
            )
            rows.append(row)
            print(
                f"[{tag}] {state_mode:7s} k={kappa:<5g} sm={smooth:<3d} g={gain:<4g} "
                f"adj={int(adjust)} gate={gate:<4g} -> viterbi={row['mean_viterbi_accuracy']:.4f} "
                f"occ={row['mean_occupancy_accuracy']:.4f} vote={row['mean_vote_accuracy']:.4f}",
                flush=True,
            )
        best = max(rows, key=lambda r: r["mean_viterbi_accuracy"])
        results["configs"][tag] = {"all": rows, "best_by_mean_viterbi": best}

    out_path = _HERE / args.out
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
