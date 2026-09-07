"""Aggregate the PRI-T feasibility gate into the deliverable table.

Reports, per validation session and per identity arm:
  * inferred-vs-true 8-class direction agreement under (a) the paper's own default
    configuration and (b) the best configuration found in our sweep;
  * the 1/8 chance rate and the per-session majority-class rate;
  * the angular error structure of the inferred labels;
  * lambda = E[cos(inferred - true)], the first-order attenuation that this label noise
    would impose on T4's per-unit cosine coefficients [a, c].  Because PRI-T's labels are
    per trial and therefore shared across units, this attenuation is (to first order)
    a single scalar common to every unit, which the pipeline's per-column z-scoring
    largely absorbs; the accompanying variance inflation is not absorbed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
N_DIRECTIONS = 8
CANONICAL = np.array([-3.0 * math.pi / 4.0 + k * (math.pi / 4.0) for k in range(N_DIRECTIONS)])
SESSIONS = [
    "sub-C_ses-CO-20151103", "sub-C_ses-CO-20151104", "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109", "sub-C_ses-CO-20151110", "sub-C_ses-CO-20151112",
]
# The paper's stated defaults: N = 20 screen grid, epsilon = 0.999, distance-modulated
# kappa, raw (unsmoothed) decoder velocity, every timestep observed.
PAPER_DEFAULT = dict(state_mode="grid20", smooth_width_bins=1, adjust_kappa=True,
                     speed_gate_quantile=0.0)


def snap(angles: np.ndarray) -> np.ndarray:
    delta = (CANONICAL[None, :] - np.asarray(angles)[:, None] + math.pi) % (2 * math.pi) - math.pi
    return np.argmin(np.abs(delta), axis=1)


def label_noise_geometry(tag: str, config: dict) -> dict:
    """Recompute inferred labels for one config to characterize their error structure."""
    import prit_gate

    decoded = prit_gate.load_decoded(tag)
    offsets_deg: list[float] = []
    cos_deltas: list[float] = []
    per_session: dict[str, dict] = {}
    locs, dir_index = prit_gate.build_states(config["state_mode"], prit_gate.TARGET_RADIUS)
    radius = prit_gate.TARGET_RADIUS
    hmm = prit_gate.make_hmm(
        locs, config["stay_prob"], config["vm_kappa"],
        0.10 * 2 * radius, 32.0 / (2 * radius), config["adjust_kappa"],
    )
    import fast_prit

    for name in SESSIONS:
        record = decoded[name]
        session_offsets = []
        for vel_raw, true_dir in zip(record["vel"], record["true_dir"]):
            if not np.isfinite(true_dir):
                continue
            pseudo_pos = config["velocity_gain"] * prit_gate.BIN_SIZE_S * np.cumsum(vel_raw, axis=0)
            pseudo_pos = np.vstack([np.zeros((1, 2)), pseudo_pos[:-1]])
            observed = prit_gate.smooth_velocity(vel_raw, config["smooth_width_bins"])
            gate = config["speed_gate_quantile"]
            if gate > 0.0:
                speed = np.hypot(observed[:, 0], observed[:, 1])
                keep = speed >= np.percentile(speed, 100.0 * gate)
                if keep.sum() >= 5:
                    observed, pseudo_pos = observed[keep], pseudo_pos[keep]
            obs_log = fast_prit.emission_log_prob(hmm, observed, pseudo_pos)
            states = fast_prit.viterbi(obs_log, config["stay_prob"], locs.shape[0])
            usable = dir_index[states] >= 0
            if usable.any():
                pred = int(np.argmax(np.bincount(dir_index[states][usable], minlength=N_DIRECTIONS)))
            else:
                pred = 0
            delta = (CANONICAL[pred] - true_dir + math.pi) % (2 * math.pi) - math.pi
            session_offsets.append(math.degrees(delta))
        session_offsets = np.asarray(session_offsets)
        offsets_deg.extend(session_offsets.tolist())
        cos_deltas.extend(np.cos(np.radians(session_offsets)).tolist())
        per_session[name] = {
            "n_trials": int(session_offsets.size),
            "accuracy": float(np.mean(np.abs(session_offsets) < 1e-6)),
            "lambda_expected_cos_delta": float(np.mean(np.cos(np.radians(session_offsets)))),
            "median_abs_offset_deg": float(np.median(np.abs(session_offsets))),
        }
    offsets = np.asarray(offsets_deg)
    bins = {}
    for step in (0, 45, 90, 135, 180):
        bins[f"offset_{step}_deg_fraction"] = float(
            np.mean(np.abs(np.abs(offsets) - step) < 1e-6)
        )
    return {
        "per_session": per_session,
        "pooled_offset_distribution": bins,
        "pooled_lambda_expected_cos_delta": float(np.mean(cos_deltas)),
        "pooled_accuracy": float(np.mean(np.abs(offsets) < 1e-6)),
    }


def pick(rows: list[dict], **constraints) -> dict | None:
    matches = [r for r in rows if all(r[k] == v for k, v in constraints.items())]
    if not matches:
        return None
    return max(matches, key=lambda r: r["mean_viterbi_accuracy"])


def main() -> None:
    task = json.loads((_HERE / "gate_results_task.json").read_text())
    grid = json.loads((_HERE / "gate_results_grid.json").read_text())

    out: dict = {
        "chance_accuracy_8_class": 1.0 / N_DIRECTIONS,
        "sessions": SESSIONS,
        "note": (
            "Best configurations are selected on these same six A2 development sessions; "
            "there is no held-out hyperparameter selection, so best-config numbers are an "
            "optimistic ceiling, which is the correct direction for a feasibility gate. "
            "The formal test sessions were never opened."
        ),
        "arms": {},
    }
    for tag in ("t4", "z4", "t4_zeroid"):
        rows = task["configs"][tag]["all"] + grid["configs"][tag]["all"]
        paper = pick(rows, **PAPER_DEFAULT)
        best = max(rows, key=lambda r: r["mean_viterbi_accuracy"])
        entry: dict = {}
        for label, row in (("paper_default_grid20", paper), ("best_swept", best)):
            if row is None:
                entry[label] = None
                continue
            entry[label] = {
                "config": {
                    k: row[k] for k in (
                        "state_mode", "n_states", "vm_kappa", "smooth_width_bins",
                        "velocity_gain", "stay_prob", "adjust_kappa",
                        "speed_gate_quantile",
                    )
                },
                "mean_viterbi_accuracy": row["mean_viterbi_accuracy"],
                "mean_occupancy_accuracy": row["mean_occupancy_accuracy"],
                "mean_hmm_free_velocity_vote_accuracy": row["mean_vote_accuracy"],
                "per_session": {
                    name: {
                        "viterbi_accuracy": value["viterbi_accuracy"],
                        "occupancy_accuracy": value["occupancy_accuracy"],
                        "hmm_free_velocity_vote_accuracy": value["mean_velocity_vote_accuracy"],
                        "majority_class_rate": value["majority_class_rate"],
                        "n_trials": value["n_trials"],
                    }
                    for name, value in row["per_session"].items()
                },
            }
        entry["label_noise_geometry_best_swept"] = label_noise_geometry(tag, best)
        out["arms"][tag] = entry

    (_HERE / "gate_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"chance (8 classes) = {1/N_DIRECTIONS:.4f}\n")
    for tag, entry in out["arms"].items():
        print(f"===== identity arm: {tag}")
        for label in ("paper_default_grid20", "best_swept"):
            row = entry[label]
            if row is None:
                print(f"  {label}: not run")
                continue
            cfg = row["config"]
            print(f"  {label}: {cfg['state_mode']} S={cfg['n_states']} kappa={cfg['vm_kappa']} "
                  f"smooth={cfg['smooth_width_bins']} adj={cfg['adjust_kappa']} "
                  f"gate={cfg['speed_gate_quantile']}")
            print(f"    mean viterbi={row['mean_viterbi_accuracy']:.4f}  "
                  f"occupancy={row['mean_occupancy_accuracy']:.4f}  "
                  f"hmm-free vote={row['mean_hmm_free_velocity_vote_accuracy']:.4f}")
            for name, value in row["per_session"].items():
                print(f"      {name}  n={value['n_trials']:3d}  "
                      f"viterbi={value['viterbi_accuracy']:.3f}  "
                      f"vote={value['hmm_free_velocity_vote_accuracy']:.3f}  "
                      f"majority={value['majority_class_rate']:.3f}")
        geo = entry["label_noise_geometry_best_swept"]
        print(f"    label noise (best config): lambda=E[cos delta]="
              f"{geo['pooled_lambda_expected_cos_delta']:.4f}  "
              f"pooled acc={geo['pooled_accuracy']:.4f}")
        print(f"      offsets: {geo['pooled_offset_distribution']}")
        print()
    print(f"Saved {_HERE / 'gate_summary.json'}")


if __name__ == "__main__":
    main()
