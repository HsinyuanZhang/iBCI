"""The AC3-0 matrix: row constructions, folds, metrics, shortcut controls.

Statics (``R0``/``R0.5``/``R1``/``O2``) are read straight off the materialized
four-group trajectories and the frozen pseudo-direction payloads of the P2'
pipeline.  Learned rows (``R2``/``R3``/``R4``/``R5``/``RS``) are fitted under
leave-one-source-session-out folds, whole trials in one split, with every
fold-local object (feature normalizer, encoder, probe) fitted on the training
sessions of that fold only.

The metric layer reports per-session and session-mean values so the binding
gates of ``plan.GATES`` compare like with like with the immutable P2' receipt.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from . import circular as circ
from . import encoder as enc
from . import plan
from . import sampler as spl
from . import summaries as su


class AC3MatrixError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3MatrixError(message)


# ---------------------------------------------------------------------------
# The trial set: what the materialization stage persisted.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialSet:
    sessions: tuple[str, ...]
    trial_ids: tuple[str, ...]
    trial_session: np.ndarray
    group_velocity: tuple[tuple[np.ndarray, ...], ...]
    group_valid: tuple[tuple[np.ndarray, ...], ...]
    true_velocity: tuple[np.ndarray, ...]
    pseudo: Mapping[str, Mapping[str, np.ndarray]]
    true_direction: Mapping[str, np.ndarray]

    @property
    def n_trials(self) -> int:
        return len(self.trial_ids)

    def session_trials(self, session_index: int) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.trial_session) == int(session_index))

    def true_thetas(self) -> np.ndarray:
        return np.asarray(self.true_direction["theta_raw_rad"], dtype=np.float64)

    def true_accepted(self) -> np.ndarray:
        return np.asarray(self.true_direction["accepted"], dtype=bool)

    def trial_true_speed(self) -> np.ndarray:
        return np.asarray([
            float(np.linalg.norm(np.asarray(item, dtype=np.float64), axis=1).mean())
            for item in self.true_velocity
        ], dtype=np.float64)

    def trial_duration(self) -> np.ndarray:
        return np.asarray([
            int(np.asarray(mask, dtype=bool).sum()) for mask in self.group_valid
        ], dtype=np.float64)

    def payload(self) -> dict[str, object]:
        digest = hashlib.sha256()
        for trial_id, session_index in zip(self.trial_ids, np.asarray(self.trial_session).tolist()):
            digest.update(str(trial_id).encode("utf-8"))
            digest.update(int(session_index).to_bytes(4, "little"))
        return {
            "n_sessions": len(self.sessions),
            "sessions": list(self.sessions),
            "n_trials": self.n_trials,
            "trials_per_session": {
                self.sessions[index]: int((np.asarray(self.trial_session) == index).sum())
                for index in range(len(self.sessions))
            },
            "trial_membership_sha256": digest.hexdigest(),
            "true_accepted_trials": int(self.true_accepted().sum()),
        }


def trial_set_from_arrays(
    *,
    sessions: Sequence[str],
    trial_ids: Sequence[str],
    trial_session: np.ndarray,
    group_velocity: Sequence[Sequence[np.ndarray]],
    group_valid: Sequence[Sequence[np.ndarray]],
    true_velocity: Sequence[np.ndarray],
    pseudo: Mapping[str, Mapping[str, np.ndarray]],
    true_direction: Mapping[str, np.ndarray],
) -> TrialSet:
    trial_count = len(trial_ids)
    _require(len(sessions) >= 2, "the AC3-0 screen needs at least two source sessions")
    _require(len(np.asarray(trial_session)) == trial_count, "trial session index length drift")
    _require(
        len(group_velocity) == len(group_valid) == len(true_velocity) == trial_count,
        "trial array length drift",
    )
    for views, masks in zip(group_velocity, group_valid):
        _require(len(views) == len(masks) == plan.GROUP_COUNT, "each trial needs four group views")
    return TrialSet(
        sessions=tuple(str(item) for item in sessions),
        trial_ids=tuple(str(item) for item in trial_ids),
        trial_session=np.asarray(trial_session, dtype=np.int32),
        group_velocity=tuple(tuple(np.asarray(item, dtype=np.float64) for item in views)
                             for views in group_velocity),
        group_valid=tuple(tuple(np.asarray(item, dtype=bool) for item in masks)
                          for masks in group_valid),
        true_velocity=tuple(np.asarray(item, dtype=np.float64) for item in true_velocity),
        pseudo={key: dict(value) for key, value in pseudo.items()},
        true_direction=dict(true_direction),
    )


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    body = {
        "dtype": str(array.dtype),
        "shape": [int(item) for item in array.shape],
        "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
    return hashlib.sha256(plan.canonical_json_bytes(body)).hexdigest()


def raw_array_payload(value: np.ndarray) -> dict[str, object]:
    """The materialize receipt's own array digest structure (byte-exact)."""
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": [int(item) for item in array.shape],
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Row estimates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowEstimate:
    row: str
    theta: np.ndarray                 # [T] trial-level estimate (nan = undefined)
    credibility: np.ndarray           # [T]
    representation: np.ndarray        # [T, d] retrieval representation
    per_view_theta: np.ndarray        # [T, 4]
    leakage_label: Optional[str] = None

    def payload(self) -> dict[str, object]:
        return {
            "row": self.row,
            "n_defined": int(np.isfinite(self.theta).sum()),
            "theta_sha256": array_digest(self.theta),
            "credibility_sha256": array_digest(self.credibility),
            "representation_sha256": array_digest(self.representation),
            "leakage_label": self.leakage_label,
        }


def _unit(theta: float) -> np.ndarray:
    if not math.isfinite(float(theta)):
        return np.zeros(2, dtype=np.float64)
    return np.asarray([math.cos(float(theta)), math.sin(float(theta))], dtype=np.float64)


def _view_thetas(trial_set: TrialSet, construction: str) -> np.ndarray:
    payload = trial_set.pseudo[construction]
    thetas = np.asarray(payload["theta_raw_rad"], dtype=np.float64)
    accepted = np.asarray(payload["accepted"], dtype=bool)
    matrix = np.where(accepted, thetas, np.nan)
    _require(matrix.shape == (trial_set.n_trials, plan.GROUP_COUNT),
             "pseudo-direction view matrix shape drift")
    return matrix


def static_row(trial_set: TrialSet, row: str) -> RowEstimate:
    """R0 / R0.5 / R1 / O2 constructions."""
    count = trial_set.n_trials
    if row in ("R0", "R1"):
        construction = plan.ROW_SPECS[row]["construction"]
        views = _view_thetas(trial_set, construction)
        # Single-view estimate: the first complementary group (deterministic).
        theta = views[:, 0].copy()
        credibility = np.asarray([
            (float(circ.resultant_length(row_views[np.isfinite(row_views)]))
             if int(np.isfinite(row_views).sum()) >= 1 else float("nan"))
            for row_views in views
        ], dtype=np.float64)
        representation = np.stack([_unit(value) for value in theta], axis=0)
        return RowEstimate(row=row, theta=theta, credibility=credibility,
                           representation=representation, per_view_theta=views)
    if row == "R0.5":
        views = _view_thetas(trial_set, "raw")
        theta = np.full(count, np.nan, dtype=np.float64)
        credibility = np.full(count, np.nan, dtype=np.float64)
        for index in range(count):
            finite = views[index][np.isfinite(views[index])]
            if finite.size < 2:
                continue
            ensemble = circ.group_ensemble(finite)
            if ensemble["theta_rad"] is not None:
                theta[index] = float(ensemble["theta_rad"])
            credibility[index] = float(ensemble["rho"])
        representation = np.stack([_unit(value) for value in theta], axis=0)
        return RowEstimate(row=row, theta=theta, credibility=credibility,
                           representation=representation, per_view_theta=views)
    if row == "O2":
        true_theta = trial_set.true_thetas()
        accepted = trial_set.true_accepted()
        theta = np.where(accepted, true_theta, np.nan)
        representation = np.stack([_unit(value) for value in theta], axis=0)
        return RowEstimate(
            row=row, theta=theta, credibility=np.ones(count, dtype=np.float64),
            representation=representation, per_view_theta=np.repeat(theta[:, None], 4, axis=1),
            leakage_label=plan.ROW_SPECS["O2"]["leakage_label"],
        )
    raise AC3MatrixError(f"static_row does not build row {row}")


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _session_mean(values: Mapping[str, object]) -> Optional[float]:
    finite = [float(value) for value in values.values()
              if value is not None and math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else None


def _per_view_session_rows(trial_set: TrialSet, estimate: RowEstimate) -> dict[str, dict[str, object]]:
    """P2'-comparable rows: every accepted view of every trial is one comparison."""
    true_theta = trial_set.true_thetas()
    true_accepted = trial_set.true_accepted()
    views = np.asarray(estimate.per_view_theta, dtype=np.float64)
    rows: dict[str, dict[str, object]] = {}
    for index, session in enumerate(trial_set.sessions):
        trials = trial_set.session_trials(index)
        errors: list[float] = []
        mismatch = 0
        compared = 0
        for trial in trials:
            if not bool(true_accepted[trial]):
                continue
            true_index = circ.nearest_canonical(float(true_theta[trial]))[0]
            for view in views[trial]:
                if not math.isfinite(float(view)):
                    continue
                errors.append(float(circ.circular_distance(float(view), float(true_theta[trial]))))
                compared += 1
                if circ.nearest_canonical(float(view))[0] != true_index:
                    mismatch += 1
        rows[session] = {
            "trials": int(trials.size),
            "groups_compared": compared,
            "mean_circular_error_rad": float(np.mean(errors)) if errors else None,
            "snap_mismatch_rate": float(mismatch / compared) if compared else None,
        }
    return rows


def _per_trial_session_rows(trial_set: TrialSet, estimate: RowEstimate) -> dict[str, dict[str, object]]:
    true_theta = trial_set.true_thetas()
    true_accepted = trial_set.true_accepted()
    theta = np.asarray(estimate.theta, dtype=np.float64)
    rows: dict[str, dict[str, object]] = {}
    for index, session in enumerate(trial_set.sessions):
        trials = trial_set.session_trials(index)
        selected = trials[true_accepted[trials] & np.isfinite(theta[trials])]
        errors = [
            float(circ.circular_distance(float(theta[trial]), float(true_theta[trial])))
            for trial in selected
        ]
        mismatch = sum(
            1 for trial in selected
            if circ.nearest_canonical(float(theta[trial]))[0]
            != circ.nearest_canonical(float(true_theta[trial]))[0]
        )
        rows[session] = {
            "trials": int(trials.size),
            "compared": int(selected.size),
            "mean_circular_error_rad": float(np.mean(errors)) if errors else None,
            "snap_mismatch_rate": float(mismatch / selected.size) if selected.size else None,
        }
    return rows


def _calibration_for(trial_set: TrialSet, estimate: RowEstimate) -> dict[str, object]:
    """REQUIRED credibility calibration (section 23 amendment 4), pooled + per session."""
    true_theta = trial_set.true_thetas()
    true_accepted = trial_set.true_accepted()
    theta = np.asarray(estimate.theta, dtype=np.float64)
    credibility = np.asarray(estimate.credibility, dtype=np.float64)

    def _curve(trials: np.ndarray) -> Optional[dict[str, object]]:
        selected = trials[true_accepted[trials] & np.isfinite(theta[trials]) & np.isfinite(credibility[trials])]
        if selected.size == 0:
            return None
        errors = np.asarray([
            float(circ.circular_distance(float(theta[trial]), float(true_theta[trial])))
            for trial in selected
        ], dtype=np.float64)
        flags = np.asarray([
            0 if circ.nearest_canonical(float(theta[trial]))[0]
            == circ.nearest_canonical(float(true_theta[trial]))[0] else 1
            for trial in selected
        ], dtype=np.float64)
        return circ.calibration_curve(credibility[selected], errors, flags)

    pooled = _curve(np.arange(trial_set.n_trials))
    per_session = {}
    for index, session in enumerate(trial_set.sessions):
        curve = _curve(trial_set.session_trials(index))
        if curve is not None:
            per_session[session] = curve
    return {"pooled": pooled, "per_session": per_session}


def row_metrics(trial_set: TrialSet, estimate: RowEstimate, *, per_view: bool = False) -> dict[str, object]:
    """Circular error, snap mismatch, retrieval, dispersion, calibration."""
    per_session = (
        _per_view_session_rows(trial_set, estimate) if per_view
        else _per_trial_session_rows(trial_set, estimate)
    )
    pooled_error = _session_mean({s: r["mean_circular_error_rad"] for s, r in per_session.items()})
    pooled_mismatch = _session_mean({s: r["snap_mismatch_rate"] for s, r in per_session.items()})
    views = np.asarray(estimate.per_view_theta, dtype=np.float64)
    dispersion = np.asarray([
        1.0 - float(circ.resultant_length(row_views[np.isfinite(row_views)]))
        if int(np.isfinite(row_views).sum()) >= 2 else float("nan")
        for row_views in views
    ], dtype=np.float64)
    return {
        "row": estimate.row,
        "aggregation": ("per_view_groups" if per_view else "per_trial"),
        "leakage_label": estimate.leakage_label,
        "per_session": per_session,
        "session_mean_circular_error_rad": pooled_error,
        "session_mean_snap_mismatch_rate": pooled_mismatch,
        "session_mean_cross_group_direction_dispersion": float(np.nanmean(dispersion))
        if bool(np.isfinite(dispersion).any()) else None,
        "retrieval": retrieval_metrics(trial_set, estimate),
        "credibility_calibration": _calibration_for(trial_set, estimate),
        "estimate_digest": estimate.payload(),
    }


def retrieval_metrics(trial_set: TrialSet, estimate: RowEstimate) -> dict[str, object]:
    """Cross-session nearest-neighbour retrieval in the row's own representation.

    Also computes the speed/duration-only confound baseline of section 16.5: if
    retrieving by speed and duration alone explains the agreement rate, the
    direction representation has not earned it.
    """
    representation = np.asarray(estimate.representation, dtype=np.float64)
    _require(representation.shape[0] == trial_set.n_trials, "retrieval representation length drift")
    true_theta = trial_set.true_thetas()
    true_accepted = trial_set.true_accepted()
    speeds = trial_set.trial_true_speed()
    durations = trial_set.trial_duration()
    session_array = np.asarray(trial_set.trial_session)
    speed_scale = float(np.std(speeds)) or 1.0
    duration_scale = float(np.std(durations)) or 1.0
    confound = np.stack([
        (speeds - speeds.mean()) / speed_scale,
        (durations - durations.mean()) / duration_scale,
    ], axis=1)
    agree = 0
    confound_agree = 0
    error_sum = 0.0
    compared = 0
    for trial in range(trial_set.n_trials):
        if not (bool(true_accepted[trial]) and math.isfinite(float(true_theta[trial]))):
            continue
        candidates = np.flatnonzero((session_array != session_array[trial]) & true_accepted)
        if candidates.size == 0:
            continue
        similarities = representation[candidates] @ representation[trial]
        neighbour = int(candidates[int(np.argmax(similarities))])
        distances = np.linalg.norm(confound[candidates] - confound[trial], axis=1)
        confound_neighbour = int(candidates[int(np.argmin(distances))])
        query_index = circ.nearest_canonical(float(true_theta[trial]))[0]
        compared += 1
        if circ.nearest_canonical(float(true_theta[neighbour]))[0] == query_index:
            agree += 1
        if circ.nearest_canonical(float(true_theta[confound_neighbour]))[0] == query_index:
            confound_agree += 1
        error_sum += float(circ.circular_distance(float(true_theta[neighbour]), float(true_theta[trial])))
    return {
        "compared": compared,
        "true_direction_agreement_rate": float(agree / compared) if compared else None,
        "mean_circular_error_to_retrieved_rad": float(error_sum / compared) if compared else None,
        "speed_duration_only_baseline_agreement_rate": float(confound_agree / compared) if compared else None,
        "beats_speed_duration_confound_baseline": bool(agree > confound_agree) if compared else None,
        "note": "nearest neighbour restricted to trials of OTHER source sessions",
    }


# ---------------------------------------------------------------------------
# Folds and learned rows.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSpec:
    held_out_session: str
    held_out_index: int
    train_trials: np.ndarray
    test_trials: np.ndarray


def session_folds(trial_set: TrialSet) -> tuple[FoldSpec, ...]:
    folds = []
    session_array = np.asarray(trial_set.trial_session)
    for index, session in enumerate(trial_set.sessions):
        test = trial_set.session_trials(index)
        train = np.flatnonzero(session_array != index)
        _require(test.size >= 1 and train.size >= 1, f"degenerate fold for session {session}")
        folds.append(FoldSpec(session, index, train, test))
    return tuple(folds)


def _fold_state_mask(table: spl.StateTable, fold: FoldSpec) -> np.ndarray:
    train_trials = {int(item) for item in fold.train_trials}
    return np.asarray([int(item) in train_trials for item in table.trial_idx], dtype=bool)


def _completed_trial_features(trial_set: TrialSet, trials: Sequence[int]) -> np.ndarray:
    """The four per-view completed-trial summaries of the given trials."""
    rows: list[np.ndarray] = []
    for trial in trials:
        views = trial_set.group_velocity[int(trial)]
        masks = trial_set.group_valid[int(trial)]
        for group in plan.GROUPS:
            rows.append(su.completed_trial_summary(views[group], masks[group]))
    return np.stack(rows, axis=0)


def _completed_trial_labels(trial_set: TrialSet, trials: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Completed-trial direction labels, one per (trial, view) row.

    The estimand of EVERY matrix row is the frozen completed-trial integrated
    direction (the same ``pseudo_direction_from_velocity`` pipeline that
    produces R0/R-GE/O2).  Feeding a readout bin-level endpoint labels would
    change the estimand: on this cohort the last-bin true direction sits a mean
    2.34 rad from the integrated completed-trial direction, so the readout is
    fitted on the completed-trial label at the ``t = T`` state.  Trials the
    frozen true pipeline rejects carry no label and are excluded from the fit.
    """
    true_theta = trial_set.true_thetas()
    accepted = trial_set.true_accepted()
    labels: list[float] = []
    valid: list[bool] = []
    for trial in trials:
        trial = int(trial)
        okay = bool(accepted[trial]) and math.isfinite(float(true_theta[trial]))
        for _group in plan.GROUPS:
            labels.append(float(true_theta[trial]) if okay else float("nan"))
            valid.append(okay)
    return np.asarray(labels, dtype=np.float64), np.asarray(valid, dtype=bool)


LEARNED_ARM = {"R2": None, "R3": "C-Time", "R4": "C-Action", "R5": "C-Hybrid", "RS": "shuffle"}


def learned_row_estimates(
    trial_set: TrialSet,
    table: spl.StateTable,
    *,
    row: str,
    embedding_dim: int,
    temperature: float = 1.0,
    seed: int = enc.TORCH_SEED,
) -> dict[str, object]:
    """Leave-one-source-session-out fits for one learned row.

    Every fold fits its own normalizer, encoder and probe on the training
    sessions only; the held-out session's trials are then evaluated at
    ``t = T`` through the completed-trial summary.
    """
    _require(row in plan.LEARNED_ROWS, f"learned_row_estimates does not build row {row}")
    folds = session_folds(trial_set)
    per_view_theta = np.full((trial_set.n_trials, plan.GROUP_COUNT), np.nan, dtype=np.float64)
    pooled_embedding = np.zeros((trial_set.n_trials, int(embedding_dim)), dtype=np.float64)
    credibility = np.full(trial_set.n_trials, np.nan, dtype=np.float64)
    receipts: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for fold in folds:
        state_mask = _fold_state_mask(table, fold)
        train_states = np.flatnonzero(state_mask & table.eligible())
        _require(train_states.size >= enc.BATCH_SIZE, "a fold lacks enough training states")
        normalizer = su.FeatureNormalizer.fit(table.features[train_states])
        features = normalizer.transform(table.features)
        train_features = normalizer.transform(
            _completed_trial_features(trial_set, fold.train_trials)
        )
        train_labels, train_valid = _completed_trial_labels(trial_set, fold.train_trials)
        _require(int(train_valid.sum()) >= 32, "a fold lacks labelled completed trials")
        if row == "R2":
            fit = enc.train_supervised(
                train_features[train_valid], train_labels[train_valid],
                embedding_dim=int(embedding_dim), seed=seed,
                batch_size=min(enc.BATCH_SIZE, int(train_valid.sum())),
            )
            trial_features = normalizer.transform(_completed_trial_features(trial_set, fold.test_trials))
            view_thetas = enc.supervised_directions(fit, trial_features)
            receipts.append({
                "held_out_session": fold.held_out_session, "row": row, "arm": None,
                "embedding_dim": int(embedding_dim),
                "supervision": "completed-trial direction labels at the t=T state",
                "n_labelled_completed_trials": int(train_valid.sum()),
                "encoder_parameter_sha256": fit["encoder_parameter_sha256"],
                "head_parameter_sha256": fit["head_parameter_sha256"],
                "parameter_count": fit["parameter_count"], "final_loss": fit["final_loss"],
                "normalizer": normalizer.payload(),
            })
        else:
            arm = LEARNED_ARM[row]
            fold_sampler = spl.ContrastiveSampler(table, eligible_mask=state_mask)
            fit = enc.train_contrastive(
                fold_sampler, arm=arm, embedding_dim=int(embedding_dim),
                temperature=float(temperature), seed=seed,
            )
            train_embeddings = enc.embed_numpy(fit["module"], train_features)
            probe = enc.fit_linear_probe(
                train_embeddings[train_valid], train_labels[train_valid],
            )
            trial_features = normalizer.transform(_completed_trial_features(trial_set, fold.test_trials))
            trial_embeddings = enc.embed_numpy(fit["module"], trial_features)
            view_thetas = enc.probe_directions(probe, trial_embeddings)
            last_audit = fit["sampler_audit_sample"][-1] if fit["sampler_audit_sample"] else None
            audits.append({
                "held_out_session": fold.held_out_session, "arm": arm, "counts": last_audit,
            })
            receipts.append({
                "held_out_session": fold.held_out_session, "row": row, "arm": arm,
                "embedding_dim": int(embedding_dim), "temperature": float(temperature),
                "probe_supervision": "completed-trial direction labels at the t=T state",
                "n_labelled_completed_trials": int(train_valid.sum()),
                "encoder_parameter_sha256": fit["encoder_parameter_sha256"],
                "parameter_count": fit["parameter_count"], "final_loss": fit["final_loss"],
                "probe": {key: value for key, value in probe.items() if key != "weights"},
                "normalizer": normalizer.payload(),
            })
            for row_index, trial in enumerate(fold.test_trials):
                pooled_embedding[int(trial)] = trial_embeddings[
                    row_index * plan.GROUP_COUNT:(row_index + 1) * plan.GROUP_COUNT
                ].mean(axis=0)
        for row_index, trial in enumerate(fold.test_trials):
            per_view_theta[int(trial), :] = view_thetas[
                row_index * plan.GROUP_COUNT:(row_index + 1) * plan.GROUP_COUNT
            ]
        for trial in fold.test_trials:
            row_views = per_view_theta[int(trial)]
            finite = row_views[np.isfinite(row_views)]
            if finite.size >= 1:
                credibility[int(trial)] = float(circ.resultant_length(finite))
    theta = np.asarray([
        _ensemble_theta(per_view_theta[index]) for index in range(trial_set.n_trials)
    ], dtype=np.float64)
    norms = np.linalg.norm(pooled_embedding, axis=1)
    representation = np.stack([
        (vector / norm) if norm > 0 else np.zeros(pooled_embedding.shape[1], dtype=np.float64)
        for vector, norm in zip(pooled_embedding, norms)
    ], axis=0)
    estimate = RowEstimate(
        row=row, theta=theta, credibility=credibility, representation=representation,
        per_view_theta=per_view_theta,
    )
    return {"estimate": estimate, "receipts": receipts, "sampler_audits": audits}


def _ensemble_theta(row_views: np.ndarray) -> float:
    finite = row_views[np.isfinite(row_views)]
    if finite.size < 1:
        return float("nan")
    value = circ.group_ensemble(finite)["theta_rad"]
    return float("nan") if value is None else float(value)


# ---------------------------------------------------------------------------
# Section 16 shortcut-control audits.
# ---------------------------------------------------------------------------


def session_id_probe(table: spl.StateTable, features_by_state: np.ndarray) -> dict[str, object]:
    """Control 1: how well does a linear probe recover the session ID?"""
    return enc.linear_probe_accuracy(features_by_state, table.session_idx)


def fold_split_leakage(trial_set: TrialSet) -> dict[str, object]:
    """Control 2: no fold's train/test sets share a trial id."""
    overlaps = {}
    for fold in session_folds(trial_set):
        shared = {trial_set.trial_ids[int(i)] for i in fold.train_trials} & {
            trial_set.trial_ids[int(i)] for i in fold.test_trials
        }
        overlaps[fold.held_out_session] = len(shared)
    return {
        "train_test_trial_overlap": overlaps,
        "any_overlap": any(value > 0 for value in overlaps.values()),
        "split_unit": "source session (whole trials)",
    }


def leave_one_session_influence(
    trial_set: TrialSet, row_metrics_by_row: Mapping[str, Mapping[str, object]],
    baseline_row: str, candidate_row: str,
) -> dict[str, object]:
    """Control 5 / gate (d): does the improvement survive dropping any session?"""
    baseline = row_metrics_by_row[baseline_row]["per_session"]
    candidate = row_metrics_by_row[candidate_row]["per_session"]
    improvements = {}
    for session in trial_set.sessions:
        if session not in baseline or session not in candidate:
            improvements[session] = None
            continue
        base = baseline[session]["mean_circular_error_rad"]
        cand = candidate[session]["mean_circular_error_rad"]
        improvements[session] = None if (base is None or cand is None) else float(base) - float(cand)
    drops: dict[str, Optional[float]] = {}
    sessions = [s for s in trial_set.sessions if improvements.get(s) is not None]
    for dropped in sessions:
        remaining = [value for session, value in improvements.items() if session != dropped]
        drops[dropped] = float(np.mean(remaining)) if remaining else None
    pooled = float(np.mean([improvements[s] for s in sessions])) if sessions else None
    return {
        "per_session_improvement_rad": improvements,
        "pooled_improvement_rad": pooled,
        "improvement_after_dropping_each_session_rad": drops,
        "min_over_drops": (float(np.min([v for v in drops.values() if v is not None]))
                           if any(v is not None for v in drops.values()) else None),
    }
