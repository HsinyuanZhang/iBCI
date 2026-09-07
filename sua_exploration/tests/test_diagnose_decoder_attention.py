"""Focused synthetic tests for the repaired A12 real-forward architecture.

These are intentionally tensor/model fixtures, not synthetic A12 *receipts*.
They exercise the new coupled-B3S capture core and its metadata-only runner
plan without opening NWB files, loading a checkpoint, or performing a real
forward audit on project data.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = ROOT / "sua_exploration"
STREAMING_ROOT = ROOT / "streaming_calibration_exp"
for location in (str(STREAMING_ROOT), str(SUA_ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

import torch
import torch.nn as nn

from mc_maze import a12_descriptive_attention_audit as core  # noqa: E402
from mc_maze import decoder_attention_diagnostic as diagnostic  # noqa: E402
from src.models.components.spint import MultiLayerCrossAttention  # noqa: E402


class TinyDecoder(nn.Module):
    """Minimal object graph matching the actual coupled B3S decode path."""

    def __init__(self, *, units_window: int = 50, model_dim: int = 8, heads: int = 2) -> None:
        super().__init__()
        self.fc_in = nn.Linear(units_window, model_dim, bias=False)
        self.rep = nn.Parameter(torch.randn(2, units_window))
        self.transformer = MultiLayerCrossAttention(
            num_layers=1,
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=16,
            dropout=0.0,
        )
        self.fc_out = nn.Linear(model_dim, 1, bias=False)


class TinyB3SStudent(nn.Module):
    """Synthetic tensor fixture using the real coupled cross-attention class."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder_mode = "coupled"
        self.fixed_slot_router = None
        self.decoupled_transformer = None
        self.decoder = TinyDecoder()

    def forward(
        self,
        neural: torch.Tensor,
        *,
        calib_trials: torch.Tensor,
        side_features: torch.Tensor,
        electrode_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del electrode_ids
        diagnostic.assert_b3s_input_shapes(neural, calib_trials, side_features)
        batch, width, units = neural.shape
        # A deterministic fixture identity with exactly [B,N,W] geometry.  It
        # is never zeroed or overridden by the capture path.
        identity_base = calib_trials.mean(dim=(1, 2)) + side_features.mean(dim=-1)
        identity = identity_base.unsqueeze(-1).expand(batch, units, width)
        src = neural.permute(0, 2, 1) + identity
        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        output, _ = self.decoder.transformer(rep.repeat(batch, 1, 1), src)
        return self.decoder.fc_out(output).permute(0, 2, 1), identity


class TinyLitWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.student = TinyB3SStudent()


def _tiny_inputs(*, batch: int = 3, units: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    neural = torch.randn(batch, 50, units)
    calibration = torch.randn(batch, 30, 100, units)
    side = torch.randn(batch, units, 4)
    return neural, calibration, side


def _load_runner_module():
    script = SUA_ROOT / "scripts" / "diagnose_decoder_attention.py"
    spec = importlib.util.spec_from_file_location("a12_forward_runner_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_aggregate_module():
    script = SUA_ROOT / "scripts" / "aggregate_decoder_attention_diagnostic.py"
    spec = importlib.util.spec_from_file_location("a12_forward_aggregate_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _forward_receipt_fixture(*, seed: int = 42, epoch: int = 5) -> dict[str, Any]:
    """A structural receipt fixture only; never written as an operational result."""

    digest = "a" * 64
    def session_row(session: str) -> dict[str, Any]:
        return {
            "input_shape_contract": diagnostic.INPUT_SHAPE_CONTRACT,
            "input_shapes": [
                {
                    "shape": {
                        "neural": [4, 50, 17],
                        "calibration": [4, 30, 100, 17],
                        "side": [4, 17, 4],
                    },
                    "batches": 1,
                }
            ],
            "session_provenance_sha256": digest,
            "support_trial_index_sha256": digest,
            "query_trial_index_sha256": digest,
            "query_window_start_sha256": digest,
            "model_input_sha256": digest,
            "num_query_trials": 31,
            "num_query_windows": 4,
            "qkv_order": ["Q", "K", "V"],
            "value_projection_used_for_head_contribution": True,
            "capture_output_parity_exact": True,
            "attention_summary": {
                "mean_normalized_entropy": 0.5,
                "mean_effective_attended_units": 3.0,
                "head_contribution_l2_mean": 1.0,
                "mean_pairwise_head_cosine": 0.25,
                "variance_across_windows": 0.01,
                "variance_across_covariates": 0.02,
                "attention_tensor_sha256": digest,
                "query_normalized_sha256": digest,
                "key_normalized_sha256": digest,
                "value_normalized_sha256": digest,
                "q_projection_sha256": digest,
                "k_projection_sha256": digest,
                "v_projection_sha256": digest,
                "qkv_order": ["Q", "K", "V"],
                "value_projection_used_for_head_contribution": True,
                "num_windows": 4,
                "num_units": 17,
            },
        }

    def arm_row(arm: str, checkpoint_digest: str) -> dict[str, Any]:
        return {
            "arm": arm,
            "checkpoint_path": f"/tmp/{arm}_epoch_004.ckpt",
            "checkpoint_sha256": checkpoint_digest,
            "run_metadata_path": f"/tmp/{arm}_run_metadata.json",
            "run_metadata_sha256": "b" * 64,
            "normalizer": {
                "source_train_only": True,
                "target_or_validation_refit_performed": False,
                "normalization_base_feature_group": "t4",
                "side_feature_group": arm,
                "side_feature_normalizer_sha256": core.EXPECTED_NORMALIZER_SHA256,
                "behavior_train_stats_sha256": "c" * 64,
                "support_direction_labels_used_for_t4_carrier": True,
                **core.QUERY_BEHAVIOR_FORWARD_CONTRACT,
            },
            "model_state_sha256_pre": "d" * 64,
            "model_state_sha256_post": "d" * 64,
            "model_state_unchanged": True,
            "qkv_order": ["Q", "K", "V"],
            "capture_output_parity_exact": True,
            "sessions": {session: session_row(session) for session in core.DEFAULT_VALIDATION_SESSIONS},
            "preflight_result_sha256": "e" * 64,
        }

    return {
        "schema_version": core.SCHEMA_VERSION,
        "kind": core.FORWARD_KIND,
        "status": "COMPLETED_DESCRIPTIVE_CPU_FORWARD_ONLY",
        "seed": seed,
        "epoch": epoch,
        "official_metadata_preflight": {
            "path": "/tmp/official_metadata_preflight.json",
            "sha256": "f" * 64,
            "receipt_body_sha256": "1" * 64,
        },
        "canonical_scope": {
            "dataset": "DANDI 000688 sub-C / CO / sorted SUA",
            "variant": "B3S",
            "arms": ["t4", "z4"],
            "historical_t4_reference_qualification": "historical T4 paired with v10 Z4",
            "independently_trained_checkpoint_comparison": True,
            "fresh_matched_common_training_claimed": False,
        },
        "execution_scope": {
            "cpu_only": True,
            "cuda_visible_devices": "",
            "python_no_user_site": "1",
            "gpu_used": False,
            "training_performed": False,
            "backward_gradients": False,
            "decoder_weight_updates": False,
            "checkpoint_weight_updates": False,
            "sealed_formal_test_sessions_opened": False,
            **core.QUERY_BEHAVIOR_FORWARD_CONTRACT,
            "support_direction_labels_used_for_t4_carrier": True,
            "whole_identity_zeroing_performed": False,
            "carrier_forward_ablation_performed": False,
            "same_checkpoint_carrier_ablation_performed": False,
            "descriptive_not_causal": True,
            "output_capture_parity_checked": True,
            "cpu_forward_batch_size": core.CPU_FORWARD_BATCH_SIZE,
            "cpu_forward_batch_contract_sha256": core.cpu_forward_batch_contract_sha256(),
            "torch_version": "fixture",
            "torch_path": "/tmp/torch.py",
        },
        "carrier_contrast": "paired_frozen_t4_vs_z4_checkpoints_no_within_forward_intervention",
        "pairing": {
            "same_unit_paired": True,
            "sessions": list(core.DEFAULT_VALIDATION_SESSIONS),
            "validation_unit_counts": {session: 17 for session in core.DEFAULT_VALIDATION_SESSIONS},
            "proof": "fixture same-unit proof",
            "support_query_trial_disjoint": True,
        },
        "arms": {"t4": arm_row("t4", "2" * 64), "z4": arm_row("z4", "3" * 64)},
    }


def test_stack_windows_preserves_bwn_without_transpose() -> None:
    first = torch.arange(50 * 3, dtype=torch.float32).reshape(50, 3)
    second = first + 1000.0
    stacked = diagnostic.stack_windows_bwn([first, second])
    assert tuple(stacked.shape) == (2, 50, 3)
    assert torch.equal(stacked[0], first)
    with pytest.raises(diagnostic.A12ForwardError, match="axis 0"):
        diagnostic.stack_windows_bwn([first.transpose(0, 1)])


def test_actual_coupled_b3s_capture_has_exact_output_parity_and_qkv() -> None:
    torch.manual_seed(5)
    model = TinyLitWrapper().eval()
    neural, calibration, side = _tiny_inputs()
    state_before = diagnostic.model_state_sha256(model)
    plain_prediction, plain_identity = diagnostic.plain_b3s_forward(model, neural, calibration, side)
    captured = diagnostic.forward_b3s_with_capture(model, neural, calibration, side)
    diagnostic.assert_output_parity(plain_prediction, captured.prediction)
    assert torch.equal(plain_identity, captured.identity)
    assert len(captured.layers) == 1
    layer = captured.layers[0]
    assert tuple(layer.attention.shape) == (3, 2, 2, 5)
    assert tuple(layer.q.shape) == (3, 2, 8)
    assert tuple(layer.k.shape) == tuple(layer.v.shape) == (3, 5, 8)
    assert diagnostic.model_state_sha256(model) == state_before


def test_project_qkv_and_head_contribution_use_v_not_k() -> None:
    mha = nn.MultiheadAttention(4, 2, batch_first=True, dropout=0.0)
    with torch.no_grad():
        mha.in_proj_weight.zero_()
        mha.in_proj_bias.zero_()
        mha.in_proj_weight[:4].fill_(1.0)       # Q
        mha.in_proj_weight[4:8].fill_(2.0)      # K
        mha.in_proj_weight[8:12].fill_(3.0)     # V
        mha.out_proj.weight.copy_(torch.eye(4))
        mha.out_proj.bias.zero_()
    query = torch.ones(1, 2, 4)
    key = torch.full((1, 3, 4), 2.0)
    value = torch.full((1, 3, 4), 5.0)
    q, k, v = diagnostic.project_qkv(mha, query, key, value)
    assert torch.all(q == 4.0)
    assert torch.all(k == 16.0)
    assert torch.all(v == 60.0)
    attention = torch.zeros(1, 2, 2, 3)
    attention[..., 0] = 1.0
    contribution_v = diagnostic.head_contribution_norms(mha, v, attention)
    contribution_k = diagnostic.head_contribution_norms(mha, k, attention)
    assert not torch.equal(contribution_v, contribution_k)
    # Each head owns two V dimensions of 60, so the L2 norm is sqrt(2)*60.
    assert torch.allclose(contribution_v, torch.full_like(contribution_v, 60.0 * (2.0 ** 0.5)))


def test_capture_refuses_decoupled_and_fixed_slot_paths() -> None:
    model = TinyLitWrapper().eval()
    model.student.decoder_mode = "decoupled"
    neural, calibration, side = _tiny_inputs()
    with pytest.raises(diagnostic.A12ForwardError, match="ordinary coupled"):
        diagnostic.forward_b3s_with_capture(model, neural, calibration, side)
    model.student.decoder_mode = "coupled"
    model.student.fixed_slot_router = object()
    with pytest.raises(diagnostic.A12ForwardError, match="fixed-slot"):
        diagnostic.forward_b3s_with_capture(model, neural, calibration, side)


def test_attention_summary_exposes_descriptive_metrics_and_v_provenance() -> None:
    torch.manual_seed(19)
    model = TinyLitWrapper().eval()
    neural, calibration, side = _tiny_inputs(batch=4)
    captured = diagnostic.forward_b3s_with_capture(model, neural, calibration, side)
    summary = diagnostic.summarize_attention_captures(captured.layers)
    assert summary["qkv_order"] == ["Q", "K", "V"]
    assert summary["value_projection_used_for_head_contribution"] is True
    for key in (
        "query_normalized_sha256",
        "key_normalized_sha256",
        "value_normalized_sha256",
        "q_projection_sha256",
        "k_projection_sha256",
        "v_projection_sha256",
    ):
        assert len(summary[key]) == 64
    for key in (
        "mean_normalized_entropy",
        "mean_effective_attended_units",
        "head_contribution_l2_mean",
        "mean_pairwise_head_cosine",
        "variance_across_windows",
        "variance_across_covariates",
    ):
        assert isinstance(summary[key], float)
    assert summary["num_windows"] == 4


def test_streaming_attention_summary_is_partition_invariant_on_tiny_synthetic_capture() -> None:
    torch.manual_seed(23)
    model = TinyLitWrapper().eval()
    neural, calibration, side = _tiny_inputs(batch=7)
    captured = diagnostic.forward_b3s_with_capture(model, neural, calibration, side)
    batch_summary = diagnostic.summarize_attention_captures(captured.layers)
    accumulator = diagnostic.AttentionSummaryAccumulator(expected_num_windows=7)
    # Feed the exact same capture in two batches to exercise cross-batch
    # sufficient-statistic accumulation without retaining the tensors.  This
    # is deliberately a tiny synthetic tensor fixture: it tests execution
    # partitioning only, never target metrics or project data.
    first = captured.layers[0]
    split = 3
    def split_record(start: int, stop: int) -> diagnostic.CapturedCrossAttention:
        return diagnostic.CapturedCrossAttention(
            layer_index=first.layer_index,
            mha=first.mha,
            query_normalized=first.query_normalized[start:stop],
            key_normalized=first.key_normalized[start:stop],
            value_normalized=first.value_normalized[start:stop],
            q=first.q[start:stop],
            k=first.k[start:stop],
            v=first.v[start:stop],
            attention=first.attention[start:stop],
        )
    accumulator.update((split_record(0, split),))
    accumulator.update((split_record(split, 7),))
    streaming_summary = accumulator.finalize()
    for key in (
        "mean_normalized_entropy",
        "mean_effective_attended_units",
        "mean_inverse_participation_ratio",
        "mean_max_probability",
        "head_contribution_l2_mean",
        "mean_pairwise_head_cosine",
        "variance_across_windows",
        "variance_across_covariates",
    ):
        assert streaming_summary[key] == pytest.approx(batch_summary[key], rel=1e-6, abs=1e-7)
    assert streaming_summary["head_contribution_l2_per_head"] == pytest.approx(
        batch_summary["head_contribution_l2_per_head"], rel=1e-6, abs=1e-7
    )
    for key in (
        "attention_tensor_sha256",
        "query_normalized_sha256",
        "key_normalized_sha256",
        "value_normalized_sha256",
        "q_projection_sha256",
        "k_projection_sha256",
        "v_projection_sha256",
    ):
        assert streaming_summary[key] == batch_summary[key]


def test_tiny_cpu_forward_is_invariant_to_a_fixed_batch_partition() -> None:
    """Exercise the actual capture path with tiny tensors, never project data."""

    torch.manual_seed(29)
    model = TinyLitWrapper().eval()
    neural, calibration, side = _tiny_inputs(batch=7)

    def run_partition(batch_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
        predictions: list[torch.Tensor] = []
        accumulator = diagnostic.AttentionSummaryAccumulator(expected_num_windows=7)
        for start in range(0, 7, batch_size):
            stop = min(start + batch_size, 7)
            plain, _ = diagnostic.plain_b3s_forward(
                model, neural[start:stop], calibration[start:stop], side[start:stop]
            )
            captured = diagnostic.forward_b3s_with_capture(
                model, neural[start:stop], calibration[start:stop], side[start:stop]
            )
            diagnostic.assert_output_parity(plain, captured.prediction)
            predictions.append(captured.prediction)
            accumulator.update(captured.layers)
        return torch.cat(predictions, dim=0), accumulator.finalize()

    # 512 is the frozen production batch cap; the tiny fixture therefore has
    # one batch.  A different *synthetic* partition must preserve the model's
    # numerical result and per-batch capture parity.  Raw tensor digests are
    # intentionally not expected to match: CPU BLAS reduction order can vary
    # at float32-ulp scale across a batch partition, which is precisely why
    # A12 freezes and records the operational batch size.
    full_prediction, full_summary = run_partition(core.CPU_FORWARD_BATCH_SIZE)
    split_prediction, split_summary = run_partition(3)
    # BLAS reduction order can differ across CPU batch partitions by a few
    # float32 ulps; this is numerical, not a semantic/model-path difference.
    torch.testing.assert_close(full_prediction, split_prediction, rtol=1e-5, atol=1e-6)
    for key in (
        "mean_normalized_entropy",
        "mean_effective_attended_units",
        "mean_inverse_participation_ratio",
        "mean_max_probability",
        "head_contribution_l2_mean",
        "mean_pairwise_head_cosine",
        "variance_across_windows",
        "variance_across_covariates",
    ):
        assert split_summary[key] == pytest.approx(full_summary[key], rel=1e-6, abs=1e-7)


def test_forward_receipt_validator_rejects_false_provenance_claims() -> None:
    receipt = _forward_receipt_fixture()
    core.validate_pair_forward_receipt(receipt)
    mutated = copy.deepcopy(receipt)
    mutated["arms"]["t4"]["sessions"][core.DEFAULT_VALIDATION_SESSIONS[0]]["qkv_order"] = ["Q", "V", "K"]
    with pytest.raises(core.A12AuditError, match="Q/K/V"):
        core.validate_pair_forward_receipt(mutated)
    mutated = copy.deepcopy(receipt)
    mutated["arms"]["z4"]["checkpoint_sha256"] = mutated["arms"]["t4"]["checkpoint_sha256"]
    with pytest.raises(core.A12AuditError, match="independently trained"):
        core.validate_pair_forward_receipt(mutated)
    mutated = copy.deepcopy(receipt)
    mutated["execution_scope"]["whole_identity_zeroing_performed"] = True
    with pytest.raises(core.A12AuditError, match="whole_identity"):
        core.validate_pair_forward_receipt(mutated)
    mutated = copy.deepcopy(receipt)
    mutated["execution_scope"]["query_behavior_loaded_by_shared_session_loader"] = False
    with pytest.raises(core.A12AuditError, match="query_behavior_loaded_by_shared_session_loader"):
        core.validate_pair_forward_receipt(mutated)
    mutated = copy.deepcopy(receipt)
    mutated["arms"]["t4"]["normalizer"]["query_behavior_used_for_attention_metrics"] = True
    with pytest.raises(core.A12AuditError, match="query_behavior_used_for_attention_metrics"):
        core.validate_pair_forward_receipt(mutated)
    mutated = copy.deepcopy(receipt)
    mutated["execution_scope"]["cpu_forward_batch_size"] = 32
    with pytest.raises(core.A12AuditError, match="cpu_forward_batch_size drift"):
        core.validate_pair_forward_receipt(mutated)
    mutated = copy.deepcopy(receipt)
    mutated["execution_scope"]["cpu_forward_batch_contract_sha256"] = "0" * 64
    with pytest.raises(core.A12AuditError, match="cpu_forward_batch_contract_sha256 drift"):
        core.validate_pair_forward_receipt(mutated)
    mutated = copy.deepcopy(receipt)
    mutated["arms"]["t4"]["sessions"][core.DEFAULT_VALIDATION_SESSIONS[0]]["input_shapes"][0]["shape"]["neural"][0] = 513
    mutated["arms"]["t4"]["sessions"][core.DEFAULT_VALIDATION_SESSIONS[0]]["input_shapes"][0]["shape"]["calibration"][0] = 513
    mutated["arms"]["t4"]["sessions"][core.DEFAULT_VALIDATION_SESSIONS[0]]["input_shapes"][0]["shape"]["side"][0] = 513
    with pytest.raises(core.A12AuditError, match="recorded batch dimension violates frozen B=512"):
        core.validate_pair_forward_receipt(mutated)
    mutated = copy.deepcopy(receipt)
    shape = mutated["arms"]["t4"]["sessions"][core.DEFAULT_VALIDATION_SESSIONS[0]]["input_shapes"][0]
    shape["shape"]["neural"][0] = 2
    shape["shape"]["calibration"][0] = 2
    shape["shape"]["side"][0] = 2
    shape["batches"] = 2
    with pytest.raises(core.A12AuditError, match="more than one non-full batch"):
        core.validate_pair_forward_receipt(mutated)


def test_runner_has_no_legacy_synthetic_m2_h1_or_spint_main_surface() -> None:
    source = (SUA_ROOT / "scripts" / "diagnose_decoder_attention.py").read_text(encoding="utf-8")
    # Mentioning a rejected legacy path in an error comment is harmless; the
    # executable surface must not construct its old root/import route.
    for forbidden in ("SPINT_ROOT =", "--synthetic", "run_synthetic_receipt", "DEFAULT_M2", "DEFAULT_H1", "zero_identity"):
        assert forbidden not in source


def test_runner_dry_plan_never_imports_torch_or_opens_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner_module()
    preflight_path = tmp_path / "official_metadata_preflight.json"
    preflight_path.write_text("fixture", encoding="utf-8")
    digest = "f" * 64
    checkpoint = {"checkpoint_path": "/tmp/checkpoint.ckpt", "checkpoint_sha256_observed": digest}
    preflight = {
        "pairs": {
            "42": {
                "pairing": {"validation_sessions": list(core.DEFAULT_VALIDATION_SESSIONS)},
                "arms": {
                    arm: {
                        "metadata": {"metadata_path": f"/tmp/{arm}.json", "metadata_sha256": "a" * 64},
                        "result": {"epoch_checkpoints": {"5": checkpoint}},
                    }
                    for arm in ("t4", "z4")
                },
            }
        }
    }
    monkeypatch.setattr(runner.core, "sha256_file", lambda path: "b" * 64)
    before_torch = set(name for name in sys.modules if name == "torch" or name.startswith("torch."))
    plan = runner.build_dry_run_plan(preflight, preflight_path=preflight_path, seed=42, epoch=5)
    after_torch = set(name for name in sys.modules if name == "torch" or name.startswith("torch."))
    assert plan["status"] == "DRY_RUN_ONLY__NO_TORCH_NWB_OR_FORWARD"
    assert plan["scope"]["torch_imported"] is False
    assert plan["scope"]["shared_session_loader_invoked"] is False
    assert plan["scope"]["forward_query_behavior_contract"] == core.QUERY_BEHAVIOR_FORWARD_CONTRACT
    assert before_torch == after_torch


def test_real_forward_requires_explicit_root_review_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner_module()
    preflight_path = tmp_path / "official_metadata_preflight.json"
    preflight_path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(runner, "load_official_metadata_preflight", lambda path: ({}, "a" * 64))
    monkeypatch.delenv(runner.FORWARD_REVIEW_ENV, raising=False)
    status = runner.main(
        [
            "--official-metadata-preflight", str(preflight_path),
            "--seed", "42",
            "--epoch", "5",
            "--run-forward",
            "--output", str(tmp_path / "would_not_write.json"),
        ]
    )
    assert status == 2
    assert not (tmp_path / "would_not_write.json").exists()


def test_runner_rejects_any_unfrozen_batch_size_before_a_real_forward() -> None:
    runner = _load_runner_module()
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            [
                "--official-metadata-preflight", "/tmp/official_metadata_preflight.json",
                "--seed", "42",
                "--epoch", "5",
                "--batch-size", "32",
            ]
        )
    accepted = runner.build_parser().parse_args(
        [
            "--official-metadata-preflight", "/tmp/official_metadata_preflight.json",
            "--seed", "42",
            "--epoch", "5",
            "--batch-size", str(core.CPU_FORWARD_BATCH_SIZE),
        ]
    )
    assert accepted.batch_size == core.CPU_FORWARD_BATCH_SIZE


def test_programmatic_runner_rejects_batch_drift_before_importing_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    def should_not_import_runtime() -> None:
        raise AssertionError("batch drift reached the real Torch/NWB runtime boundary")

    monkeypatch.setattr(runner, "_prepare_actual_b3s_imports", should_not_import_runtime)
    with pytest.raises(core.A12AuditError, match="batch size drift"):
        runner.run_pair_forward(
            preflight_path=Path("/tmp/unused-preflight.json"),
            preflight={},
            preflight_sha256="a" * 64,
            seed=42,
            epoch=5,
            batch_size=32,
        )


def test_runner_rejects_noncanonical_official_preflight_path(tmp_path: Path) -> None:
    runner = _load_runner_module()
    fake = tmp_path / "official_metadata_preflight.json"
    fake.write_text("{}\n", encoding="utf-8")
    with pytest.raises(runner.A12RunnerError, match="canonical official metadata preflight path"):
        runner.load_official_metadata_preflight(fake)


def test_aggregate_reports_only_paired_descriptive_t4_minus_z4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _load_aggregate_module()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text("fixture-first\n", encoding="utf-8")
    second_path.write_text("fixture-second\n", encoding="utf-8")
    first = _forward_receipt_fixture(seed=42, epoch=5)
    second = _forward_receipt_fixture(seed=43, epoch=5)
    session = core.DEFAULT_VALIDATION_SESSIONS[0]
    second["arms"]["z4"]["sessions"][session]["attention_summary"]["mean_normalized_entropy"] = 0.8
    preflight_path = aggregate.CANONICAL_PREFLIGHT_PATH
    fake_preflight = {
        "fixture": True,
        "cpu_forward_batch_contract": {
            **core.cpu_forward_batch_contract(),
            "contract_sha256": core.cpu_forward_batch_contract_sha256(),
        },
    }

    def fake_load(path: Path):
        if path == first_path.resolve():
            return first, fake_preflight, preflight_path, "a" * 64
        if path == second_path.resolve():
            return second, fake_preflight, preflight_path, "a" * 64
        raise AssertionError(path)

    monkeypatch.setattr(aggregate, "_load_pair_receipt", fake_load)
    result = aggregate.aggregate_pair_receipts([first_path, second_path])
    assert result["execution_scope"]["descriptive_not_causal"] is True
    assert result["execution_scope"]["causal_gate_performed"] is False
    assert result["execution_scope"]["inferential_test_performed"] is False
    assert result["complete_canonical_seed_epoch_matrix"] is False
    assert result["partial_matrix_is_pilot_only"] is True
    assert result["execution_scope"]["delta_definition"] == "t4_minus_z4"
    assert result["cpu_forward_batch_contract"] == fake_preflight["cpu_forward_batch_contract"]
    entropy = result["paired_session_summary"][session]["t4_minus_z4"]["mean_normalized_entropy"]
    assert entropy["n_paired_receipts"] == 2
    assert entropy["mean"] == pytest.approx((0.0 - 0.3) / 2.0)

    with pytest.raises(aggregate.A12AggregateError, match="duplicate"):
        aggregate.aggregate_pair_receipts([first_path, first_path])


def test_aggregate_rejects_batch_drift_even_if_receipt_loading_is_mocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _load_aggregate_module()
    path = tmp_path / "drifted.json"
    path.write_text("fixture-drifted\n", encoding="utf-8")
    receipt = _forward_receipt_fixture()
    receipt["execution_scope"]["cpu_forward_batch_size"] = 32
    preflight = {
        "cpu_forward_batch_contract": {
            **core.cpu_forward_batch_contract(),
            "contract_sha256": core.cpu_forward_batch_contract_sha256(),
        }
    }
    monkeypatch.setattr(
        aggregate,
        "_load_pair_receipt",
        lambda _: (receipt, preflight, aggregate.CANONICAL_PREFLIGHT_PATH, "a" * 64),
    )
    with pytest.raises(aggregate.A12AggregateError, match="CPU forward batch size drift"):
        aggregate.aggregate_pair_receipts([path])


def test_aggregate_rejects_noncanonical_preflight_before_loading_it() -> None:
    aggregate = _load_aggregate_module()
    receipt = _forward_receipt_fixture()
    with pytest.raises(aggregate.A12AggregateError, match="noncanonical"):
        aggregate._preflight_for_receipt(receipt)
