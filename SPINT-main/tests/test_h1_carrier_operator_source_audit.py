"""Mocked/no-NWB contracts for the gated H1 carrier-operator source audit."""
from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import h1_carrier_operator_source_audit as audit
from src.data.h1_carrier_operator_candidate import FrozenCarrierOperator, fit_o1


SOURCE = tuple(f"ses-source-{index:02d}" for index in range(11))


@dataclass(frozen=True)
class _Trial:
    rates: np.ndarray
    velocity: np.ndarray


class _Record:
    def __init__(self, name: str, *, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.session_name = name
        self.date = "19250108"
        self.trial_values = tuple(float(index) for index in range(6))
        self.num_neurons = 8
        self.trials = tuple(
            _Trial(
                rates=rng.uniform(1.0, 15.0, size=(3, self.num_neurons)),
                velocity=rng.normal(size=(3, 2)),
            )
            for _ in self.trial_values
        )
        self._by_value = dict(zip(self.trial_values, self.trials))

    def blocks_for(self, value: float) -> _Trial:
        return self._by_value[float(value)]


def _operator() -> FrozenCarrierOperator:
    rng = np.random.default_rng(77)
    return FrozenCarrierOperator(
        mean=rng.normal(size=8), scale=rng.uniform(0.7, 1.6, size=8), pcs=rng.normal(size=(3, 8)),
        ridge_lambda=2.0, U=rng.normal(size=(2, 4)), mu=rng.normal(size=4), tau2=0.8,
    )


def _literal(record: _Record, plan: SimpleNamespace, values: tuple[float, ...]) -> dict[str, np.ndarray]:
    rates = np.concatenate([record.blocks_for(value).rates for value in values], axis=0)
    labels = np.concatenate([record.blocks_for(value).velocity for value in values], axis=0)
    op = FrozenCarrierOperator(plan.mean, plan.scale, plan.pcs[: plan.q], plan.ridge_lambda, plan.U, plan.mu, plan.tau2)
    # The audit only requires the active result shape/keys; a tiny independent
    # literal expression makes the mock exercise comparison plumbing.
    z = ((rates - op.mean) / op.scale) @ op.pcs.T
    design = np.column_stack((np.ones(z.shape[0]), z))
    regularizer = np.eye(design.shape[1]) * op.ridge_lambda
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    beta = np.linalg.solve(system, design.T @ labels)
    rss = np.square(labels - design @ beta).sum(axis=0)
    hat_trace = float(np.trace(design @ np.linalg.solve(system, design.T)))
    sigma2 = rss / (len(design) - hat_trace)
    G = np.linalg.solve(system, design.T @ design) @ np.linalg.inv(system)
    G = (G + G.T) / 2.0
    raw_rows = (op.pcs.T @ beta[1:]) / op.scale[:, None]
    raw_carrier = raw_rows @ op.U
    projection = op.pcs.T
    factor = ((projection @ G[1:, 1:]) * projection).sum(axis=1) / np.square(op.scale)
    projected_variance = factor * np.trace(op.U.T @ np.diag(sigma2) @ op.U) / 4.0
    weight = op.tau2 / (op.tau2 + projected_variance)
    carrier = op.mu[None, :] + weight[:, None] * (raw_carrier - op.mu[None, :])
    return {"carrier": carrier, "raw_carrier": raw_carrier, "raw_rows": raw_rows, "beta": beta, "G": G,
            "sigma2": sigma2, "projected_variance": projected_variance, "weight": weight}


def _active_stub(records: dict[str, _Record], operator: FrozenCarrierOperator) -> SimpleNamespace:
    plan = SimpleNamespace(
        source_sessions=SOURCE, mean=operator.mean, scale=operator.scale, pcs=operator.pcs, q=operator.projection_dim,
        ridge_lambda=operator.ridge_lambda, U=operator.U, mu=operator.mu, tau2=operator.tau2,
        transform_sha256="a" * 64, raw_receipt_sha256="b" * 64, eb_receipt_sha256="c" * 64,
    )
    return SimpleNamespace(
        H1_M4_FOLD0_SOURCE=SOURCE, FOLD0_DATE="19250101", SUPPORT_TRIALS=4, BLOCK_SECONDS=0.1,
        load_source_records=lambda _root: records,
        reconstruct_frozen_plan=lambda observed, _raw, _eb: plan if observed is records else (_ for _ in ()).throw(AssertionError()),
        fit_frozen_carrier=lambda record, fitted_plan, values: _literal(record, fitted_plan, values),
    )


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data = tmp_path / "000954"
    data.mkdir()
    raw, eb = tmp_path / "raw.json", tmp_path / "eb.json"
    raw.write_text("{}\n", encoding="utf-8")
    eb.write_text("{}\n", encoding="utf-8")
    return data, raw, eb, tmp_path / "receipt.json"


def test_gate_is_closed_before_any_active_loader_or_path_access(monkeypatch, tmp_path: Path):
    data, raw, eb, output = _paths(tmp_path)
    monkeypatch.setattr(audit, "_active_h1_module", lambda: (_ for _ in ()).throw(AssertionError("must not import loader")))
    with pytest.raises(audit.SourceAuditError, match="explicit --run-source-audit"):
        audit.run_source_audit(data_dir=data, raw_receipt=raw, eb_receipt=eb, output=output, execute_source_audit=False)
    assert not output.exists()


def test_rejects_heldout_formal_and_non_000954_data_paths_without_importing_loader(monkeypatch, tmp_path: Path):
    _, raw, eb, output = _paths(tmp_path)
    calls = {"active": 0}
    monkeypatch.setattr(audit, "_active_h1_module", lambda: calls.__setitem__("active", calls["active"] + 1))
    for forbidden in (tmp_path / "heldout" / "000954", tmp_path / "formal" / "000954", tmp_path / "not_000954"):
        with pytest.raises(audit.SourceAuditError):
            audit.run_source_audit(data_dir=forbidden, raw_receipt=raw, eb_receipt=eb, output=output,
                                   execute_source_audit=True)
    assert calls["active"] == 0


def test_mocked_source_audit_writes_atomic_immutable_11_source_only_receipt(monkeypatch, tmp_path: Path):
    data, raw, eb, output = _paths(tmp_path)
    records = {name: _Record(name, seed=index) for index, name in enumerate(SOURCE)}
    monkeypatch.setattr(audit, "_active_h1_module", lambda: _active_stub(records, _operator()))
    result = audit.run_source_audit(data_dir=data, raw_receipt=raw, eb_receipt=eb, output=output,
                                    execute_source_audit=True)
    assert result["status"] == audit.AUDIT_STATUS and result["records"] == 11
    body = json.loads(output.read_text(encoding="utf-8"))
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert body["schema"] == audit.AUDIT_SCHEMA
    assert body["scope"]["allowed_recordings"] == list(SOURCE)
    assert body["scope"]["source_recordings_opened"] == 11
    assert body["scope"]["fold0_target_recordings_opened"] == 0
    assert body["scope"]["heldout_recordings_opened"] == 0
    assert body["operator"]["packed_o2_state_floats"] == 1 + 10 + 8 + 3
    assert set(body["per_record"]) == set(SOURCE)
    first = body["per_record"][SOURCE[0]]
    assert first["n"] == 12 and first["system_condition_2norm"] > 0.0
    assert set(first["max_abs_vs_active_literal"]["o1"]) == set(audit.COMMON_QUANTITIES)
    assert set(first["ranges"]) >= {"z", "DtD", "Dty", "beta", "raw_rows", "projected_variance", "weight"}
    assert body["source_firing_rate_and_crossover"]["crossover_raw_operation_descriptor"]["firing_rate_crossover_hz"] == 10.0
    with pytest.raises(audit.SourceAuditError, match="overwrite"):
        audit.run_source_audit(data_dir=data, raw_receipt=raw, eb_receipt=eb, output=output,
                               execute_source_audit=True)


def test_loader_cannot_expand_the_exact_eleven_recording_scope(monkeypatch, tmp_path: Path):
    data, raw, eb, output = _paths(tmp_path)
    records = {name: _Record(name, seed=index) for index, name in enumerate(SOURCE)}
    records["ses-19250101T111740"] = _Record("ses-19250101T111740", seed=99)
    active = _active_stub(records, _operator())
    reconstruct_calls = {"count": 0}
    active.reconstruct_frozen_plan = lambda *_args: reconstruct_calls.__setitem__("count", reconstruct_calls["count"] + 1)
    monkeypatch.setattr(audit, "_active_h1_module", lambda: active)
    with pytest.raises(audit.SourceAuditError, match="exact frozen eleven-recording scope"):
        audit.run_source_audit(data_dir=data, raw_receipt=raw, eb_receipt=eb, output=output,
                               execute_source_audit=True)
    assert reconstruct_calls["count"] == 0 and not output.exists()


def test_runner_does_not_offer_quantization_or_trainer_paths():
    source = (Path(__file__).parents[1] / "scripts/h1_carrier_operator_source_audit.py").read_text(encoding="utf-8").lower()
    assert "bf16" not in source and "int8" not in source and "quantization" not in source
    assert "import torch" not in source and "pynwb" not in source
