"""The AC3-U injection seam around ``policy.build_construction_predictions``.

Work order §6: the frozen ``src/learned_gate_p2prime_v1/{physical,policy}.py``
and ``src/causal_dual_memory_cell_d_v1/core.py`` files are NOT edited.  The
driver installs a process-local wrapper on the ``policy`` module attribute,
which the frozen rollout resolves at call time.

* the two new construction names (``group_ensemble`` for UGE, ``r2_head`` for
  U2) pre-rotate the four complementary views by the trial's frozen direction
  estimate and then delegate the ORIGINAL function with ``construction="raw"``;
* every frozen construction name (``raw``, ``smoothed_causal``,
  ``smoothed_zero_phase``, ``true``) is forwarded unchanged and the original
  function's own return value is returned as-is -- U0 therefore runs the exact
  frozen path and anchors bit-exactly against sealed stage-cop O0.

Trial binding is deterministic, not a closure over mutable RNG: the frozen
oracle loop calls the seam exactly once per query trial in order, so wrapper
call ``n`` maps to the ``n``-th trial of the bound session.  That mapping is
verified two ways -- by the call count, and by the predeclared drift guard of
``plan.DRIFT_GUARD`` which compares every incoming group's net-displacement
angle with the frozen ``trajectories.npz`` row of the mapped trial.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np

from src.learned_gate_p2prime_v1 import filters as p2filters

from . import plan, rotation


class AC3USeamError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3USeamError(message)


class SessionBinding:
    """One session's frozen rows + the row's per-trial direction estimates."""

    def __init__(
        self,
        *,
        row_id: str,
        construction: str,
        session: str,
        trial_ids: Sequence[str],
        theta: np.ndarray,
        frozen_velocity: np.ndarray,
        row_starts: np.ndarray,
        row_counts: np.ndarray,
    ) -> None:
        self.row_id = str(row_id)
        self.construction = str(construction)
        self.session = str(session)
        self.trial_ids = tuple(str(item) for item in trial_ids)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.frozen_velocity = np.asarray(frozen_velocity, dtype=np.float64)
        self.row_starts = np.asarray(row_starts, dtype=np.int64)
        self.row_counts = np.asarray(row_counts, dtype=np.int64)
        _require(
            self.theta.shape == (len(self.trial_ids),),
            "the direction array must align with the session's trial roster",
        )
        _require(
            self.row_starts.shape == self.row_counts.shape == (len(self.trial_ids),),
            "the frozen CSR offsets must align with the session's trial roster",
        )
        _require(
            self.frozen_velocity.ndim == 3 and self.frozen_velocity.shape[1:] == (2, 4),
            "the frozen velocity cache must be [P, 2, 4]",
        )

    def trial_slice(self, index: int) -> np.ndarray:
        start = int(self.row_starts[index])
        stop = start + int(self.row_counts[index])
        _require(0 <= start <= stop <= self.frozen_velocity.shape[0], "frozen CSR slice out of range")
        return self.frozen_velocity[start:stop]


class ConstructionSeam:
    """Callable drop-in for ``policy.build_construction_predictions``."""

    def __init__(self, *, original: Callable[..., Any], policy_module: Any) -> None:
        _require(callable(original), "the seam must wrap the original callable")
        self._original = original
        self._policy_module = policy_module
        self._installed = False
        self._binding: Optional[SessionBinding] = None
        self._cursor = 0
        self._records: list[dict[str, object]] = []
        self._current: dict[str, Any] = {}
        self.passthrough_calls = 0
        self.rotated_calls = 0

    # -- lifecycle -----------------------------------------------------------

    def install(self) -> None:
        _require(not self._installed, "the seam is already installed")
        _require(
            self._policy_module.build_construction_predictions is self._original,
            "the policy seam target moved before installation",
        )
        self._policy_module.build_construction_predictions = self
        self._installed = True

    def restore(self) -> None:
        if self._installed:
            self._policy_module.build_construction_predictions = self._original
            self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def original(self) -> Callable[..., Any]:
        return self._original

    # -- session binding -----------------------------------------------------

    def begin_session(self, binding: SessionBinding) -> None:
        _require(self._installed, "the seam must be installed before a session binds")
        _require(self._binding is None, "the previous session was not closed")
        self._binding = binding
        self._cursor = 0
        self._current = {
            "row": binding.row_id,
            "construction": binding.construction,
            "session": binding.session,
            "expected_trials": len(binding.trial_ids),
            "wrapper_calls": 0,
            "passthrough_calls": 0,
            "rotated_calls": 0,
            "undefined_theta_fallback_trials": 0,
            "groups_rotated": 0,
            "abs_deltas_rad": [],
            "angle_gaps_rad": [],
            "norm_ratios": [],
            "validity_objects_reused": 0,
            "validity_objects_rebuilt": 0,
            "trial_ids": [],
        }

    def end_session(self) -> dict[str, object]:
        _require(self._binding is not None, "no session is bound")
        record = self._current
        binding = self._binding
        _require(
            int(record["wrapper_calls"]) == len(binding.trial_ids),
            "seam call count disagrees with the bound session's trial roster: "
            f"{record['wrapper_calls']} != {len(binding.trial_ids)}",
        )
        gaps = np.asarray(record["angle_gaps_rad"], dtype=np.float64)
        ratios = np.asarray(record["norm_ratios"], dtype=np.float64)
        deltas = np.asarray(record["abs_deltas_rad"], dtype=np.float64)
        finite = np.isfinite(gaps)
        gaps = gaps[finite]
        ratios = ratios[np.isfinite(ratios)]
        guard = plan.DRIFT_GUARD
        median_gap = float(np.median(gaps)) if gaps.size else 0.0
        fraction_within = float((gaps <= 1.5).mean()) if gaps.size else 1.0
        guard_passed = bool(
            gaps.size > 0
            and median_gap <= float(guard["session_median_max_rad"])
            and fraction_within >= float(guard["session_fraction_within_1p5_rad_min"])
        )
        summary = {
            "row": record["row"],
            "construction": record["construction"],
            "session": record["session"],
            "wrapper_calls": int(record["wrapper_calls"]),
            "expected_trials": int(record["expected_trials"]),
            "passthrough_calls": int(record["passthrough_calls"]),
            "rotated_calls": int(record["rotated_calls"]),
            "undefined_theta_fallback_trials": int(record["undefined_theta_fallback_trials"]),
            "groups_rotated": int(record["groups_rotated"]),
            "validity_objects_reused": int(record["validity_objects_reused"]),
            "validity_objects_rebuilt": int(record["validity_objects_rebuilt"]),
            "trial_id_sequence_sha256": hashlib.sha256(
                json.dumps(list(record["trial_ids"]), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "rotation_delta_rad": {
                "mean_abs": float(deltas.mean()) if deltas.size else None,
                "max_abs": float(deltas.max()) if deltas.size else None,
                "count": int(deltas.size),
            },
            "frozen_cache_drift": {
                "comparison": guard["comparison"],
                "n_compared": int(gaps.size),
                "n_zero_displacement_pairs_skipped": int((~finite).sum()),
                "median_rad": median_gap,
                "mean_rad": float(gaps.mean()) if gaps.size else None,
                "max_rad": float(gaps.max()) if gaps.size else None,
                "fraction_within_1p5_rad": fraction_within,
                "displacement_norm_ratio_min": float(ratios.min()) if ratios.size else None,
                "displacement_norm_ratio_max": float(ratios.max()) if ratios.size else None,
                "guard_median_max_rad": float(guard["session_median_max_rad"]),
                "guard_fraction_min": float(guard["session_fraction_within_1p5_rad_min"]),
                "guard_passed": guard_passed,
            },
        }
        _require(
            guard_passed,
            "the predeclared drift guard fired: the incoming group velocities do "
            "not track the frozen trajectories.npz rows of the mapped trials "
            f"(session={record['session']}, row={record['row']}, "
            f"median={median_gap:.6f} rad, fraction_within_1p5={fraction_within:.4f}); "
            "stopping instead of proceeding on a wrong cursor mapping",
        )
        self._records.append(summary)
        self._binding = None
        self._current = {}
        self._cursor = 0
        return summary

    @property
    def session_records(self) -> tuple[dict[str, object], ...]:
        return tuple(self._records)

    # -- the seam itself -----------------------------------------------------

    def _audit(self, group_predictions: Sequence[Any], index: int) -> None:
        """Compare the incoming views with the frozen cache row of trial `index`."""
        binding = self._binding
        _require(binding is not None, "the seam was called with no session bound")
        _require(
            0 <= index < len(binding.trial_ids),
            "the seam cursor ran past the bound session's trial roster",
        )
        frozen = binding.trial_slice(index)
        incoming = [np.asarray(item.velocity, dtype=np.float64) for item in group_predictions]
        _require(
            all(item.shape == frozen.shape[:2] for item in incoming) and frozen.shape[1:] == (2, 4),
            f"frozen/incoming trajectory shape drift at trial {binding.trial_ids[index]}",
        )
        for group in range(len(incoming)):
            left = rotation.net_displacement(incoming[group])
            right = rotation.net_displacement(frozen[:, :, group])
            left_norm = float(np.linalg.norm(left))
            right_norm = float(np.linalg.norm(right))
            if left_norm > 0.0 and right_norm > 0.0:
                gap = abs(rotation.wrap_rad(
                    math.atan2(float(left[1]), float(left[0]))
                    - math.atan2(float(right[1]), float(right[0])),
                ))
                self._current["angle_gaps_rad"].append(float(gap))
                self._current["norm_ratios"].append(left_norm / right_norm)
            else:
                self._current["angle_gaps_rad"].append(float("nan"))
                self._current["norm_ratios"].append(float("nan"))

    def __call__(
        self,
        *,
        group_predictions: Sequence[Any],
        construction: str,
        behavior_rows: Any = None,
        behavior_mean: Sequence[float] = (),
        behavior_std: Sequence[float] = (),
        alpha: float = p2filters.ALPHA,
    ):
        binding = self._binding
        if construction in plan.ROTATING_CONSTRUCTIONS:
            _require(binding is not None, f"rotating construction {construction} with no session bound")
            _require(
                construction == binding.construction,
                f"rotating construction {construction} disagrees with the bound row "
                f"{binding.row_id} ({binding.construction})",
            )
            index = self._cursor
            self._cursor += 1
            theta = float(binding.theta[index])
            self._audit(group_predictions, index)
            self._current["wrapper_calls"] = int(self._current["wrapper_calls"]) + 1
            self._current["rotated_calls"] = int(self._current["rotated_calls"]) + 1
            self._current["trial_ids"].append(binding.trial_ids[index])
            rotated, record = rotation.rotate_group_predictions(
                group_predictions, theta, row_id=binding.row_id,
            )
            if not record["applied"]:
                self._current["undefined_theta_fallback_trials"] = (
                    int(self._current["undefined_theta_fallback_trials"]) + 1
                )
            else:
                self._current["groups_rotated"] = int(self._current["groups_rotated"]) + len(rotated)
                self._current["abs_deltas_rad"].extend(abs(float(item)) for item in record["deltas_rad"])
                # The law reuses the incoming validity evidence object verbatim;
                # a rebuilt evidence object is a violation, not a variant.
                reused = sum(
                    1 for before, after in zip(group_predictions, rotated)
                    if after.validity is before.validity
                )
                self._current["validity_objects_reused"] = (
                    int(self._current["validity_objects_reused"]) + reused
                )
                self._current["validity_objects_rebuilt"] = (
                    int(self._current["validity_objects_rebuilt"]) + (len(rotated) - reused)
                )
            self.rotated_calls += 1
            return self._original(
                group_predictions=rotated, construction="raw",
                behavior_rows=behavior_rows, behavior_mean=behavior_mean,
                behavior_std=behavior_std, alpha=alpha,
            )
        _require(
            construction in plan.FROZEN_CONSTRUCTIONS,
            f"unknown construction reached the AC3-U seam: {construction}",
        )
        if binding is not None:
            index = self._cursor
            self._cursor += 1
            self._audit(group_predictions, index)
            self._current["wrapper_calls"] = int(self._current["wrapper_calls"]) + 1
            self._current["passthrough_calls"] = int(self._current["passthrough_calls"]) + 1
            self._current["trial_ids"].append(binding.trial_ids[index])
        self.passthrough_calls += 1
        return self._original(
            group_predictions=group_predictions, construction=construction,
            behavior_rows=behavior_rows, behavior_mean=behavior_mean,
            behavior_std=behavior_std, alpha=alpha,
        )


def build_seam() -> ConstructionSeam:
    """Install-point helper: capture the original BEFORE any assignment."""
    from src.learned_gate_p2prime_v1 import policy as policy_module

    return ConstructionSeam(
        original=policy_module.build_construction_predictions, policy_module=policy_module,
    )
