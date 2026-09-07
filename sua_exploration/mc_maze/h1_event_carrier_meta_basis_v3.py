"""CPU meta-learned sparse event-label basis for H1.

The basis is optimized offline on source sessions through the same analytic
target-carrier fit used at deployment.  Target sessions still perform only a
five-parameter-per-channel ridge solve and no neural-network backward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sua_exploration.mc_maze import h1_event_carrier_design_screen as v1screen
from sua_exploration.mc_maze import h1_event_carrier_nested_context_v2 as v2
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1


SCHEMA = "h1_event_carrier_meta_basis_cpu_v3"
PROTOCOL = "h1_event_carrier_meta_forward_transfer_20260811_v3"
RANK = 4
TARGET_RIDGE = 3.0
OPTIMIZER_STEPS = 200
LEARNING_RATE = 0.03
SUBSPACE_TETHER = 1.0e-3
TORCH_DTYPE = torch.float64
ARMS: tuple[tuple[str, str], ...] = (
    ("meta_context_mid_q4", "context_mid"),
    ("meta_tag_delta_q4", "tag_delta"),
)


@dataclass(frozen=True)
class MetaMap:
    name: str
    feature_family: str
    source_sessions: tuple[str, ...]
    raw_dim: int
    active_mask: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    latent_scale: np.ndarray
    initial_projection: np.ndarray
    initial_loss: float
    final_loss: float
    source_event_count: int
    map_sha256: str

    @property
    def configuration(self) -> v2.Configuration:
        return v2.Configuration(self.name, self.feature_family, 1.0, TARGET_RIDGE)

    def transform(
        self,
        events: Sequence[v1screen.ContextEvent],
        *,
        tag_overrides: Sequence[str] | None = None,
    ) -> np.ndarray:
        raw = v2.raw_features(events, self.feature_family, tag_overrides=tag_overrides)
        x = (raw[:, self.active_mask] - self.mean[None, :]) / self.scale[None, :]
        z = x @ self.projection / self.latent_scale[None, :]
        event_v1._need(z.shape == (len(events), RANK) and np.isfinite(z).all(), "invalid meta-basis latent")
        return z

    def manifest(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "feature_family": self.feature_family,
            "source_sessions": list(self.source_sessions),
            "source_event_count": self.source_event_count,
            "raw_dim": self.raw_dim,
            "active_dim": int(self.active_mask.sum()),
            "optimizer": {
                "steps": OPTIMIZER_STEPS,
                "learning_rate": LEARNING_RATE,
                "subspace_tether": SUBSPACE_TETHER,
                "dtype": str(TORCH_DTYPE),
                "initial_loss": self.initial_loss,
                "final_loss": self.final_loss,
            },
            "array_sha256": {
                "active_mask": event_v1.array_sha256(self.active_mask),
                "mean": event_v1.array_sha256(self.mean),
                "scale": event_v1.array_sha256(self.scale),
                "initial_projection": event_v1.array_sha256(self.initial_projection),
                "projection": event_v1.array_sha256(self.projection),
                "latent_scale": event_v1.array_sha256(self.latent_scale),
            },
            "map_sha256": self.map_sha256,
        }


def _canonical_projection(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value, np.float64).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0:
            output[:, column] *= -1.0
    return output


def _initial_configuration(family: str) -> v2.Configuration:
    if family == "context_mid":
        return v2.Configuration("meta_init_context", "context_mid", 1.0, TARGET_RIDGE)
    if family == "tag_delta":
        return v2.Configuration("meta_init_tag_delta", "tag_delta", 1.0, TARGET_RIDGE)
    raise event_v1.SparseEventEndpointError(f"unsupported meta family {family}")


def _loss(
    parameter: torch.Tensor,
    initial_q: torch.Tensor,
    pooled_x: torch.Tensor,
    source_rows: Sequence[tuple[torch.Tensor, torch.Tensor, np.ndarray]],
) -> torch.Tensor:
    q = torch.linalg.qr(parameter, mode="reduced").Q
    latent_scale = torch.std(pooled_x @ q, dim=0, correction=0).clamp_min(1.0e-8)
    losses: list[torch.Tensor] = []
    penalty_base = torch.diag(torch.tensor([0.0] + [1.0] * RANK, dtype=TORCH_DTYPE))
    for x, response, trial_index in source_rows:
        z = x @ q / latent_scale
        for budget in (3, 4):
            support = torch.as_tensor(trial_index < budget, dtype=torch.bool)
            query = torch.as_tensor(trial_index >= budget, dtype=torch.bool)
            zs, zq = z[support], z[query]
            ys, yq = response[support], response[query]
            design = torch.cat((torch.ones((zs.shape[0], 1), dtype=TORCH_DTYPE), zs), dim=1)
            coefficient = torch.linalg.solve(
                design.T @ design + penalty_base * (zs.shape[0] * TARGET_RIDGE),
                design.T @ ys,
            )
            prediction = torch.cat((torch.ones((zq.shape[0], 1), dtype=TORCH_DTYPE), zq), dim=1) @ coefficient
            centered = yq - yq.mean(dim=0, keepdim=True)
            total = torch.sum(centered.square(), dim=0)
            residual = torch.sum((prediction - yq).square(), dim=0)
            defined = total > 1.0e-12
            losses.append(torch.sum(residual[defined]) / torch.sum(total[defined]))
    projector = q @ q.T
    initial_projector = initial_q @ initial_q.T
    tether = torch.mean((projector - initial_projector).square())
    return torch.stack(losses).mean() + SUBSPACE_TETHER * tether


def fit_meta_map(
    sessions: Mapping[str, v1screen.ContextSession],
    *,
    source_names: Sequence[str],
    arm_name: str,
    feature_family: str,
) -> MetaMap:
    names = tuple(source_names)
    initialization = v2.fit_context_map(
        sessions,
        source_names=names,
        configuration=_initial_configuration(feature_family),
    )
    active = initialization.active_mask
    mean, scale = initialization.mean, initialization.scale
    pooled_events = [event for name in names for event in sessions[name].events]
    raw = v2.raw_features(pooled_events, feature_family)
    pooled_x_np = (raw[:, active] - mean[None, :]) / scale[None, :]
    pooled_x = torch.as_tensor(pooled_x_np, dtype=TORCH_DTYPE)
    source_rows: list[tuple[torch.Tensor, torch.Tensor, np.ndarray]] = []
    offset = 0
    for name in names:
        count = len(sessions[name].events)
        x = torch.as_tensor(pooled_x_np[offset : offset + count], dtype=TORCH_DTYPE)
        response = torch.as_tensor(
            np.stack([event.base.log_rates for event in sessions[name].events]), dtype=TORCH_DTYPE,
        )
        trial_index = np.asarray([event.base.trial_index for event in sessions[name].events], np.int64)
        source_rows.append((x, response, trial_index))
        offset += count
    event_v1._need(offset == len(pooled_events), "meta source slicing drift")
    initial = torch.as_tensor(initialization.projection, dtype=TORCH_DTYPE)
    parameter = torch.nn.Parameter(initial.clone())
    initial_q = torch.linalg.qr(initial.detach(), mode="reduced").Q
    optimizer = torch.optim.Adam([parameter], lr=LEARNING_RATE)
    with torch.no_grad():
        initial_loss = float(_loss(parameter, initial_q, pooled_x, source_rows).item())
    for _step in range(OPTIMIZER_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(parameter, initial_q, pooled_x, source_rows)
        event_v1._need(bool(torch.isfinite(loss).item()), "meta-basis loss became nonfinite")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        q = torch.linalg.qr(parameter, mode="reduced").Q
        final_loss = float(_loss(parameter, initial_q, pooled_x, source_rows).item())
        projection = q.cpu().numpy()
    projection = _canonical_projection(projection)
    latent_scale = np.maximum((pooled_x_np @ projection).std(axis=0), 1.0e-8)
    body = {
        "protocol": PROTOCOL,
        "arm": arm_name,
        "feature_family": feature_family,
        "source_sessions": list(names),
        "active_mask": event_v1.array_sha256(active),
        "mean": event_v1.array_sha256(mean),
        "scale": event_v1.array_sha256(scale),
        "initial_projection": event_v1.array_sha256(initialization.projection),
        "projection": event_v1.array_sha256(projection),
        "latent_scale": event_v1.array_sha256(latent_scale),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
    }
    return MetaMap(
        name=arm_name,
        feature_family=feature_family,
        source_sessions=names,
        raw_dim=raw.shape[1],
        active_mask=np.asarray(active, bool),
        mean=np.asarray(mean, np.float64),
        scale=np.asarray(scale, np.float64),
        projection=np.asarray(projection, np.float64),
        latent_scale=np.asarray(latent_scale, np.float64),
        initial_projection=np.asarray(initialization.projection, np.float64),
        initial_loss=initial_loss,
        final_loss=final_loss,
        source_event_count=len(pooled_events),
        map_sha256=event_v1.canonical_sha256(body),
    )


def run_screen(sessions: Mapping[str, v1screen.ContextSession]) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, Any]]] = {
        arm_name: {"M3": {}, "M4": {}} for arm_name, _family in ARMS
    }
    baselines: dict[str, dict[str, Any]] = {"M3": {}, "M4": {}}
    maps: dict[str, dict[str, Any]] = {arm_name: {} for arm_name, _family in ARMS}
    for outer_date in event_v1.H1_DATES:
        source_names = tuple(
            name for name in event_v1.H1_HELDIN_SESSIONS
            if event_v1.session_date(name) != outer_date
        )
        baseline_map = v2.fit_pca_baseline(sessions, source_names=source_names)
        trained = {
            arm_name: fit_meta_map(
                sessions, source_names=source_names, arm_name=arm_name, feature_family=family,
            )
            for arm_name, family in ARMS
        }
        for arm_name, mapping in trained.items():
            maps[arm_name][outer_date] = mapping.manifest()
        for name in event_v1.H1_HELDIN_SESSIONS:
            if event_v1.session_date(name) != outer_date:
                continue
            for budget in (3, 4):
                baselines[f"M{budget}"][name] = v2.evaluate(sessions[name], baseline_map, budget=budget)
                for arm_name, mapping in trained.items():
                    rows[arm_name][f"M{budget}"][name] = v2.evaluate(sessions[name], mapping, budget=budget)

    arms: dict[str, Any] = {}
    passing: list[str] = []
    for arm_name, family in ARMS:
        aggregates = {
            budget: v2.compare_rows(rows[arm_name][budget], baselines[budget])
            for budget in ("M3", "M4")
        }
        gates: dict[str, Any] = {}
        for budget in ("M3", "M4"):
            summary = aggregates[budget]
            material = summary["candidate_minus_baseline"]
            clauses = {
                "mean_delta_at_least_0p02": material["mean"] >= 0.02,
                "median_delta_at_least_0p01": material["median"] >= 0.01,
                "positive_sessions_at_least_10": material["positive"] >= 10,
                "leave_largest_delta_positive": material["leave_largest_absolute_out_mean"] > 0,
                "beats_intercept_robustly": v1screen._positive_control(summary["candidate_minus_intercept"]),
                "beats_label_shuffle_robustly": v1screen._positive_control(summary["candidate_minus_label_shuffle"]),
                "beats_tag_shuffle_robustly": v1screen._positive_control(summary["candidate_minus_tag_shuffle"]),
            }
            gates[budget] = {"clauses": clauses, "passed": all(clauses.values())}
        passed = gates["M3"]["passed"] and gates["M4"]["passed"]
        if passed:
            passing.append(arm_name)
        arms[arm_name] = {
            "feature_family": family,
            "rows": rows[arm_name],
            "aggregate": aggregates,
            "gate": {"budgets": gates, "passed": passed},
            "maps": maps[arm_name],
        }
    selected = None
    if passing:
        def key(name: str) -> tuple[float, float, str]:
            aggregate = arms[name]["aggregate"]
            medians = [aggregate[budget]["candidate_minus_baseline"]["median"] for budget in ("M3", "M4")]
            means = [aggregate[budget]["candidate_minus_baseline"]["mean"] for budget in ("M3", "M4")]
            return min(medians), min(means), name
        selected = max(passing, key=key)
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "frozen_constants": {
            "rank": RANK,
            "target_ridge": TARGET_RIDGE,
            "optimizer_steps": OPTIMIZER_STEPS,
            "learning_rate": LEARNING_RATE,
            "subspace_tether": SUBSPACE_TETHER,
            "arms": [name for name, _family in ARMS],
        },
        "baseline_rows": baselines,
        "arms": arms,
        "passing_arms": passing,
        "selected_arm": selected,
        "status": "PASS_CPU_META_BASIS_MATERIAL" if selected else "STOP_CPU_META_BASIS_NOT_MATERIAL",
        "gpu_authorized_by_this_screen": False,
        "scope_interpretation": {
            "offline_source_backward_used": True,
            "target_session_backward_used": False,
            "target_carrier_fit_is_closed_form": True,
            "decoder_constructed": False,
        },
    }
