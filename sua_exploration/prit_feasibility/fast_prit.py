"""O(T*S) exact Viterbi / forward-backward for PRI-T's specific transition structure.

PRI-T's ``generateTransitionMatrix`` builds the only transition matrix the method uses:
``stayProb`` on the diagonal and a single uniform off-diagonal value
``(1 - stayProb) / (nStates - 1)``.  For that structure both recursions collapse from
O(S^2) to O(S) per timestep, which is what makes the paper's recommended 400-state screen
grid affordable here.  This is the same model, not an approximation -- verify_equivalence()
checks it against the upstream implementation numerically.
"""
from __future__ import annotations

import numpy as np


def emission_log_prob(hmm, cursor_vel: np.ndarray, cursor_pos: np.ndarray) -> np.ndarray:
    """Reuse the upstream von Mises observation model verbatim."""
    return hmm.get_posterior_prob(cursor_vel, cursor_pos, None)


def viterbi(obs_log: np.ndarray, stay_prob: float, n_states: int) -> np.ndarray:
    """Return the Viterbi state sequence for a uniform-off-diagonal transition matrix."""
    log_stay = np.log(stay_prob)
    log_off = np.log((1.0 - stay_prob) / (n_states - 1))
    length = obs_log.shape[0]

    back = np.zeros((length, n_states), dtype=np.int64)
    v = np.full(n_states, np.log(1.0 / n_states))
    for t in range(length):
        best = int(np.argmax(v))
        best_val = v[best]
        masked = v.copy()
        masked[best] = -np.inf
        second = int(np.argmax(masked))
        second_val = masked[second]

        # For target state i: come from i (stay) or from the best other state (jump).
        jump_val = np.full(n_states, best_val + log_off)
        jump_src = np.full(n_states, best, dtype=np.int64)
        jump_val[best] = second_val + log_off
        jump_src[best] = second
        stay_val = v + log_stay

        take_stay = stay_val >= jump_val
        back[t] = np.where(take_stay, np.arange(n_states), jump_src)
        v = obs_log[t] + np.where(take_stay, stay_val, jump_val)

    states = np.zeros(length, dtype=np.int64)
    states[-1] = int(np.argmax(v))
    for t in range(length - 1, 0, -1):
        states[t - 1] = back[t][states[t]]
    return states


def forward_backward(obs_log: np.ndarray, stay_prob: float, n_states: int) -> np.ndarray:
    """Return [T, S] posterior state marginals P(H_t | O_1..O_T)."""
    off = (1.0 - stay_prob) / (n_states - 1)
    diff = stay_prob - off
    length = obs_log.shape[0]
    obs = np.exp(obs_log - obs_log.max(axis=1, keepdims=True))

    fs = np.zeros((length + 1, n_states))
    scale = np.ones(length + 1)
    fs[0] = 1.0 / n_states
    for t in range(1, length + 1):
        propagated = off * fs[t - 1].sum() + diff * fs[t - 1]
        fs[t] = obs[t - 1] * propagated
        scale[t] = fs[t].sum()
        fs[t] /= scale[t]

    bs = np.ones((length + 1, n_states))
    for t in range(length - 1, -1, -1):
        weighted = bs[t + 1] * obs[t]
        bs[t] = (off * weighted.sum() + diff * weighted) / scale[t + 1]

    posterior = fs[1:] * bs[1:]
    return posterior / posterior.sum(axis=1, keepdims=True)


def verify_equivalence(seed: int = 0) -> dict:
    """Check the fast recursions against upstream PRIT on random data."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent / "prit_upstream"))
    from PRIT.prit import HMMRecalibration

    rng = np.random.RandomState(seed)
    report = {}
    for n_states, length, stay in [(8, 60, 0.999), (9, 80, 0.99), (25, 40, 0.9999)]:
        locs = rng.randn(n_states, 2) * 4.0
        transitions = np.eye(n_states) * stay
        for row in range(n_states):
            other = np.setdiff1d(np.arange(n_states), row)
            transitions[row, other] = (1.0 - stay) / (n_states - 1)
        prior = np.zeros((n_states, 1)) + 1.0 / n_states
        adjust = lambda x: 1.0 / (1.0 + np.exp(-1.0 * (x - 1.6) * 2.0))  # noqa: E731
        hmm = HMMRecalibration(transitions, locs, prior, 2.0, adjustKappa=adjust)

        vel = rng.randn(length, 2)
        pos = np.cumsum(vel, axis=0) * 0.02
        up_states, _ = hmm.viterbi_search(vel, pos)
        up_post, _ = hmm.decode(vel, pos)

        obs_log = emission_log_prob(hmm, vel, pos)
        fast_states = viterbi(obs_log, stay, n_states)
        fast_post = forward_backward(obs_log, stay, n_states)

        report[f"S{n_states}_T{length}"] = {
            "viterbi_state_agreement": float(np.mean(up_states.astype(int) == fast_states)),
            "max_abs_posterior_difference": float(
                np.abs(up_post / up_post.sum(axis=1, keepdims=True) - fast_post).max()
            ),
        }
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(verify_equivalence(), indent=2))
