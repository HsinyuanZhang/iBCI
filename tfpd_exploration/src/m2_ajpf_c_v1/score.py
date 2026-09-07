"""Score the AJPF-C modules under their own training law on the 13 sessions.

Deployment-protocol replay (probe pattern): per session one identity per
chunk-state, windows decoded in state segments, `pooled/static` must
reproduce the sealed `act30_dopt4` rows to 1e-7.  Emits the pre-registered
external contrasts and gates from plan.py.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import chunk_law, plan, source_stream


class ScoreError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreError(message)


def bootstrap_ci(values, seed: int = plan.BOOTSTRAP_SEED, resamples: int = plan.BOOTSTRAP_RESAMPLES):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = values[rng.integers(0, values.size, values.size)].mean()
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "positive": int((values > 0).sum()), "n": int(values.size),
            "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975))}


def _decode_session(torch, student, neural, windows, endpoints, commits, material, law, gated_arm):
    """Segment decode under a chunk law; returns predictions [n,2]."""
    import numpy as np

    committed_before = chunk_law.committed_before(commits, endpoints)
    predictions = np.zeros((len(endpoints), 2), dtype=np.float32)
    segment_start = 0
    committed_upto = 0
    identity = None
    identity_count = -1
    seed_rows = material.seed_rows
    support_count = material.support_count
    while segment_start < len(endpoints):
        stop = segment_start
        while stop < len(endpoints) and committed_before[stop] == committed_before[segment_start]:
            stop += 1
        count = int(committed_before[segment_start])
        stack = chunk_law.pool_rows_for_state(neural, seed_rows, support_count, commits, count)
        if count != identity_count:
            support = torch.from_numpy(stack).unsqueeze(0)
            side = torch.from_numpy(np.ascontiguousarray(material.normalized_side)).unsqueeze(0)
            with torch.inference_mode():
                if gated_arm:
                    identity = student.id_encoder.forward_batch(support, side_features=side)
                else:
                    identity = student.compute_identity(support, side_features=side)
            identity_count = count
        with torch.inference_mode():
            for offset in range(segment_start, stop, 1024):
                batch = torch.from_numpy(windows[offset : min(offset + 1024, stop)])
                output = student.decode_with_identity(batch, identity)
                predictions[offset : min(offset + 1024, stop)] = (
                    output[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False) / 5.0)
        segment_start = stop
    return predictions


def score_all(*, torch, repo_root, data_module, modules: dict[str, Any], progress) -> dict[str, Any]:
    import numpy as np

    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts,
        variance_weighted_r2,
    )

    sealed = json.loads((repo_root / plan.SEALED_SCREEN_RELATIVE).read_text(encoding="utf-8"))
    sealed_rows = {(row["surface"], row["session"]): float(row["r2"])
                   for row in sealed["rows"] if row["cell"] == plan.SEALED_CELL}
    require(len(sealed_rows) == 13, "sealed row coverage drift")

    surfaces = (("external_official_query", data_module.val_heldout_dataset, 6, "external"),
                ("within_post30", data_module.train_dataset, 7, "within"))
    rows_out = []
    for surface_name, dataset, expected, short in surfaces:
        sessions = sorted(dataset.calib_trialized_neural_features)
        require(len(sessions) == expected, f"{surface_name} session drift")
        for session in sessions:
            material = source_stream.build_session_material(dataset, session)
            starts = np.asarray(
                [s for name, s in dataset.window_indices if name == session], dtype=np.int64)
            if short == "within":
                starts = select_common_post30_window_starts(starts, dataset.trial_start_indices[session])
            require(starts.size > 0, f"{session}: no windows")
            neural = material.neural
            endpoints = starts + 49
            windows = np.ascontiguousarray(neural[starts[:, None] + np.arange(50)[None, :]], dtype=np.float32)
            targets = np.ascontiguousarray(material.targets[starts + 49], dtype=np.float32)
            for arm, module in modules.items():
                law = plan.LAW_OF_MODULE.get(arm, "static")  # pooled anchor: static law
                commits = material.commits[law] if law != "static" else []
                predictions = _decode_session(torch, module.student, neural, windows,
                                              endpoints, commits, material, law,
                                              gated_arm=arm.endswith("R1"))
                r2 = float(variance_weighted_r2(targets, predictions))
                rows_out.append({"arm": arm, "law": law, "surface": surface_name,
                                 "session": session, "window_count": int(starts.size), "r2": r2})
                progress(json.dumps(rows_out[-1], sort_keys=True))

    # lineage anchor: pooled/static under the pooled module
    for row in rows_out:
        if row["arm"] == "pooled":
            delta = abs(row["r2"] - sealed_rows[(row["surface"], row["session"])])
            require(delta <= plan.R2_ABS_TOLERANCE,
                    f"anchor drift {row['surface']}/{row['session']}: {delta}")

    def pick(arm, surface):
        return {r["session"]: r["r2"] for r in rows_out if r["arm"] == arm and r["surface"] == surface}

    summary = {"schema": plan.SCHEMA + "_score_v1", "rows": rows_out, "surfaces": {}}
    for surface in ("external_official_query", "within_post30"):
        sessions = sorted({r["session"] for r in rows_out if r["surface"] == surface})
        pooled_static = pick("pooled", surface)
        block = {"equal_session_mean": {}, "contrasts": {}}
        for arm in modules:
            per = pick(arm, surface)
            law = plan.LAW_OF_MODULE.get(arm, "static")
            block["equal_session_mean"][f"{arm}/{law}"] = float(np.mean([per[s] for s in sessions]))
        contrasts = {}
        for arm in ("C-NAT", "C-R1", "Ce-NAT", "Ce-R1"):
            per = pick(arm, surface)
            for name, comparator in ((f"{arm}_minus_pooled_static", pooled_static),):
                deltas = np.asarray([per[s] - comparator[s] for s in sessions])
                contrasts[name] = {"per_session": {s: float(d) for s, d in zip(sessions, deltas)},
                                   **bootstrap_ci(deltas)}
        for law, native, gated in (("chunk100", "C-NAT", "C-R1"), ("chunk100e", "Ce-NAT", "Ce-R1")):
            left, right = pick(gated, surface), pick(native, surface)
            deltas = np.asarray([left[s] - right[s] for s in sessions])
            contrasts[f"{gated}_minus_{native}"] = {"per_session": {s: float(d) for s, d in zip(sessions, deltas)},
                                                    **bootstrap_ci(deltas)}
        block["contrasts"] = contrasts
        summary["surfaces"][surface] = block

    external = summary["surfaces"]["external_official_query"]["contrasts"]
    gates = {}
    for tier, arm in (("candidate", "C-R1"), ("sensitivity", "Ce-R1")):
        contrast = external[f"{arm}_minus_pooled_static"]
        gates[tier] = {
            "mean": contrast["mean"], "positive": contrast["positive"],
            "candidate_gate": bool(contrast["mean"] >= plan.GATE_CANDIDATE_MEAN
                                   and contrast["positive"] >= 4),
            "paper_gate": bool(contrast["mean"] >= plan.GATE_PAPER_MEAN
                               and contrast["positive"] >= 4
                               and min(contrast["per_session"].values()) >= plan.GATE_PAPER_WORST),
        }
    summary["gates"] = gates
    return summary
