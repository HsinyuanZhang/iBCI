"""Focused CPU/no-CUDA tests for PACD V1.

These tests intentionally use a small synthetic model rather than Cell-D or
any project loader.  They must remain safe to run with no source data, no
checkpoint, no result root, and no CUDA initialisation.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.paired_anchored_calibration_dropout_v1 import plan, smoke  # noqa: E402
from src.paired_anchored_calibration_dropout_v1.core import (  # noqa: E402
    PACDError,
    assert_finite_materialized_parameters,
    paired_train_step,
    supervised_valid_loss,
)


class _ToyPairedModel(torch.nn.Module):
    """A CPU-only stand-in with Cell-D-shaped dynamic whole-unit dropout."""

    def __init__(self, units: int = 5, covariates: int = 2):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 4)
        self.decoder = torch.nn.Linear(4, covariates)
        self.units = units

    def forward(self, neural, *, calib_trials, side_features):
        del side_features
        batch, width = neural.shape[:2]
        unit_mask = torch.ones(batch, self.units, dtype=neural.dtype, device=neural.device)
        p = random.uniform(0.15, 0.35)
        unit_mask = torch.nn.functional.dropout(unit_mask, p=p, training=self.training)
        query_stat = neural.mean(dim=(1, 2))
        calib_stat = (calib_trials.mean(dim=2) * unit_mask.unsqueeze(1)).mean(dim=(1, 2))
        hidden = self.encoder(torch.stack((query_stat, calib_stat), dim=-1))
        prediction = self.decoder(hidden).unsqueeze(1).expand(-1, width, -1)
        return prediction, hidden


def _batch(batch_size: int = 3, width: int = 7, units: int = 5):
    generator = torch.Generator(device="cpu").manual_seed(192)
    neural = torch.randn(batch_size, width, units, generator=generator)
    behavior = torch.randn(batch_size, width, 2, generator=generator)
    calibration = torch.randn(batch_size, 30, 4, units, generator=generator)
    side = torch.randn(batch_size, units, 4, generator=generator)
    return neural, behavior, calibration, side


def _run_toy(short_m: int, *, collect_branch_evidence: bool = False):
    torch.manual_seed(55)
    random.seed(55)
    model = _ToyPairedModel()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    neural, behavior, calibration, side = _batch()
    before = calibration.clone()
    result = paired_train_step(
        model=model,
        optimizer=optimizer,
        neural=neural,
        behavior=behavior,
        calibration=calibration,
        side_features=side,
        short_m=short_m,
        pad_value=-100.0,
        encoder_parameters=list(model.encoder.parameters()),
        decoder_parameters=list(model.decoder.parameters()),
        collect_branch_evidence=collect_branch_evidence,
    )
    return result, calibration, before, optimizer, model


def test_optional_branch_evidence_is_default_off_and_update_equivalent():
    off, _, _, _, off_model = _run_toy(4, collect_branch_evidence=False)
    on, _, _, _, on_model = _run_toy(4, collect_branch_evidence=True)
    assert "short_encoder_grad_norm" not in off
    assert "zero_encoder_policy" not in off
    assert on["short_encoder_grad_norm"] > 0.0
    assert on["short_decoder_grad_norm"] > 0.0
    assert -1.0 <= on["encoder_branch_gradient_cosine"] <= 1.0
    assert off["loss_anchor"] == on["loss_anchor"]
    assert off["loss_short"] == on["loss_short"]
    assert off["prediction_anchor_sha256"] == on["prediction_anchor_sha256"]
    for left, right in zip(off_model.parameters(), on_model.parameters()):
        assert torch.equal(left, right)


def test_p0_is_exact_paired_rng_and_prediction_control_cpu_only():
    cuda_before = torch.cuda.is_initialized()
    result, calibration, before, optimizer, _model = _run_toy(30)
    assert torch.equal(calibration, before)
    assert result["short_m"] == 30
    assert result["dropout_pair_equal"] is True
    assert result["rng_short_transition_equal"] is True
    assert result["rng_pair_transition_equal"] is True
    assert result["prediction_pair_equal"] is True
    assert result["identity_pair_equal"] is True
    assert result["optimizer_steps"] == 1
    assert result["anchor_encoder_grad_norm"] > 0.0
    assert result["anchor_decoder_grad_norm"] > 0.0
    assert result["combined_encoder_grad_norm"] > 0.0
    assert result["combined_decoder_grad_norm"] > 0.0
    assert result["dropout"]["total"] == 15
    assert optimizer.state
    assert torch.cuda.is_initialized() is cuda_before


@pytest.mark.parametrize("short_m", [4, 10])
def test_short_prefix_arms_replay_mask_and_preserve_input_cpu_only(short_m):
    cuda_before = torch.cuda.is_initialized()
    result, calibration, before, _optimizer, _model = _run_toy(short_m)
    assert torch.equal(calibration, before)
    assert result["short_m"] == short_m
    assert result["calibration_anchor_sha256"] != result["calibration_short_sha256"]
    assert result["dropout_pair_equal"] is True
    assert result["rng_short_transition_equal"] is True
    assert result["rng_pair_transition_equal"] is True
    assert result["valid_bins"] > 0
    assert result["loss_combined"] == pytest.approx(
        0.5 * (result["loss_anchor"] + result["loss_short"])
    )
    assert torch.cuda.is_initialized() is cuda_before


def test_valid_mask_requires_nonempty_and_uses_all_behavior_channels():
    prediction = torch.zeros(1, 2, 2)
    behavior = torch.full_like(prediction, -100.0)
    with pytest.raises(PACDError, match="no valid"):
        supervised_valid_loss(prediction, behavior, pad_value=-100.0)
    behavior[0, 1] = torch.tensor([2.0, -1.0])
    loss, valid = supervised_valid_loss(prediction, behavior, pad_value=-100.0)
    assert valid == 1
    assert float(loss) == pytest.approx(2.5)


def test_real_cell_d_inactive_lazy_parameter_regression_is_cpu_only():
    """Reproduce the failed smoke's old ``p.numel()`` path on real Cell-D."""
    from src.tfpd_lane import pop_robust
    from torch.nn.parameter import UninitializedParameter

    cuda_before = torch.cuda.is_initialized()
    model = pop_robust.build_population_robustness_model(seed=42, cell="D").train()
    batch, units = 2, 5
    neural = torch.randn(batch, 50, units)
    calibration = torch.randn(batch, 30, 100, units)
    side = torch.randn(batch, units, 4)
    # This is a real Cell-D coupled forward, not a mock.  Its inactive lazy
    # branch remains unmaterialized even after the ordinary training forward.
    with torch.no_grad():
        prediction, identity = model(neural, calib_trials=calibration, side_features=side)
    assert prediction.shape == (batch, 50, 2)
    assert identity.shape == (batch, units, 50)
    assert any(isinstance(parameter, UninitializedParameter) for parameter in model.parameters())
    # Exact legacy line from the failed PACD smoke: it raises ValueError.
    with pytest.raises(ValueError, match="uninitialized parameter"):
        for parameter in model.parameters():
            if parameter.numel():
                pass
    # The repaired route checks all real parameters, skips only inactive lazy
    # parameters, and does not initialize CUDA.
    report = assert_finite_materialized_parameters(model)
    assert report["materialized"] > 0
    assert report["skipped_uninitialized_lazy"] >= 1
    assert torch.cuda.is_initialized() is cuda_before


def test_real_cell_d_sentinel_evidence_skips_inactive_lazy_parameters_cpu_only():
    """The V2 repair is exercised on a real Cell-D paired optimizer update."""
    from src.tfpd_lane import arm_common, pop_robust
    from torch.nn.parameter import UninitializedParameter

    def run(evidence_on: bool):
        torch.manual_seed(501); random.seed(501)
        model = pop_robust.build_population_robustness_model(seed=42, cell="D").train()
        neural, behavior, _calibration, side = _batch(batch_size=2, width=50, units=5)
        calibration = torch.randn(2, 30, 100, 5, generator=torch.Generator().manual_seed(502))
        encoder, decoder = arm_common.param_groups_by_branch(model)
        lazy = [p for p in model.parameters() if isinstance(p, UninitializedParameter)]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = paired_train_step(
            model=model, optimizer=optimizer, neural=neural, behavior=behavior,
            calibration=calibration, side_features=side, short_m=4, pad_value=-100.0,
            encoder_parameters=encoder + lazy, decoder_parameters=decoder,
            collect_branch_evidence=evidence_on,
        )
        state = [parameter.detach().clone() for parameter in model.parameters()
                 if not isinstance(parameter, UninitializedParameter)]
        return result, lazy, state
    evidence, lazy, state_on = run(True)
    baseline, lazy_off, state_off = run(False)
    counts = evidence["branch_evidence_parameter_counts"]
    assert counts["encoder"]["skipped_uninitialized_lazy"] == len(lazy)
    assert counts["decoder"]["skipped_uninitialized_lazy"] == len(lazy)
    assert evidence["short_encoder_grad_norm"] > 0.0
    assert all(isinstance(p, UninitializedParameter) for p in lazy)
    assert all(isinstance(p, UninitializedParameter) for p in lazy_off)
    assert "branch_evidence_parameter_counts" not in baseline
    assert len(state_on) == len(state_off) and all(torch.equal(a, b) for a, b in zip(state_on, state_off))


def test_real_cell_d_all_units_dropped_is_typed_valid_zero_encoder_cpu_only():
    """V3 admits the real Cell-D stochastic disconnect, then updates decoder.

    This is deliberately a real coupled Cell-D forward.  Setting the existing
    dynamic-dropout interval to [1, 1] only makes the otherwise stochastic
    all-units-dropped realization deterministic for this no-data regression.
    It does not modify a production model or checkpoint.
    """
    from src.tfpd_lane import arm_common, pop_robust
    from torch.nn.parameter import UninitializedParameter

    cuda_before = torch.cuda.is_initialized()
    torch.manual_seed(811); random.seed(811)
    model = pop_robust.build_population_robustness_model(seed=42, cell="D").train()
    model.decoder.dynamic_dropout_low = 1.0
    model.decoder.dynamic_dropout_high = 1.0
    neural, behavior, _calibration, side = _batch(batch_size=2, width=50, units=5)
    calibration = torch.randn(2, 30, 100, 5, generator=torch.Generator().manual_seed(812))
    encoder, decoder = arm_common.param_groups_by_branch(model)
    encoder_before = [p.detach().clone() for p in encoder]
    decoder_before = [p.detach().clone() for p in decoder if not isinstance(p, UninitializedParameter)]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = paired_train_step(
        model=model, optimizer=optimizer, neural=neural, behavior=behavior,
        calibration=calibration, side_features=side, short_m=30, pad_value=-100.0,
        encoder_parameters=encoder, decoder_parameters=decoder,
        zero_encoder_policy="accept_if_paired_unit_mask_empty",
    )
    decoder_after = [p.detach().clone() for p in decoder if not isinstance(p, UninitializedParameter)]
    assert result["dropout"]["retained"] == 0
    assert result["combined_encoder_grad_norm"] == 0.0
    assert result["combined_decoder_grad_norm"] > 0.0
    assert result["combined_encoder_zero_accepted"] is True
    assert result["combined_encoder_zero_reason"] == "all_units_dropped_valid_zero"
    assert result["paired_unit_mask_empty"] is True
    assert result["optimizer_steps"] == 1 and optimizer.state
    assert all(torch.equal(before, after) for before, after in zip(encoder_before, encoder))
    assert any(not torch.equal(before, after) for before, after in zip(decoder_before, decoder_after))
    assert torch.cuda.is_initialized() is cuda_before


class _DisconnectedEncoderRetainedModel(torch.nn.Module):
    """Cell-D-shaped mask evidence with a deliberately disconnected encoder."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 4)
        self.decoder = torch.nn.Linear(1, 2)

    def forward(self, neural, *, calib_trials, side_features):
        del calib_trials, side_features
        # This is the same two-dimensional all-ones unit-mask observation
        # surface as Cell-D.  p=0 retains every unit, so V3 must not waive a
        # disconnected encoder merely because the decoder is trainable.
        mask = torch.nn.functional.dropout(
            torch.ones(neural.shape[0], neural.shape[-1], dtype=neural.dtype),
            p=0.0, training=True,
        )
        output = self.decoder(neural.mean(dim=(1, 2), keepdim=True)).expand(-1, neural.shape[1], -1)
        return output, mask


def test_zero_encoder_with_retained_units_fails_before_optimizer_step_cpu_only():
    torch.manual_seed(813); random.seed(813)
    model = _DisconnectedEncoderRetainedModel().train()
    neural, behavior, calibration, side = _batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(PACDError, match="encoder gradient is zero"):
        paired_train_step(
            model=model, optimizer=optimizer, neural=neural, behavior=behavior,
            calibration=calibration, side_features=side, short_m=30, pad_value=-100.0,
            encoder_parameters=list(model.encoder.parameters()),
            decoder_parameters=list(model.decoder.parameters()),
            zero_encoder_policy="accept_if_paired_unit_mask_empty",
        )
    assert not optimizer.state


def test_dry_payload_is_static_and_declares_no_torch_import():
    payload = smoke.dry_payload()
    assert payload["no_torch_import"] is True
    assert payload["target_access"] is False
    assert payload["full_training_authorized"] is False
    assert [(arm["name"], arm["short_m"]) for arm in payload["arms"]] == [
        ("p0", 30), ("p1", 4), ("p2", 10)
    ]
    assert plan.EXPECTED_SEALED_SHA256[plan.INITIAL_STATE_RELATIVE].startswith("b0a340fe")


def test_public_dry_cli_is_inert_and_machine_readable():
    command = [sys.executable, "-S", str(ROOT / "scripts/run_pacd_smoke_v1.py"), "--dry-run"]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    assert payload["cell"] == plan.CELL
    assert payload["no_torch_import"] is True
    assert completed.stderr == ""


def test_pre_attempt_file_path_receipt_loader_leaves_torch_absent_in_clean_process():
    code = """
import json, sys
from pathlib import Path
sys.path.insert(0, 'tfpd_exploration')
from src.paired_anchored_calibration_dropout_v1 import smoke
smoke.load_stdlib_receipt_module(Path('.').resolve())
print(json.dumps({'torch_present': 'torch' in sys.modules}))
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == {"torch_present": False}


def test_cli_refuses_execution_without_explicit_root_acknowledgement(tmp_path):
    out_dir = tmp_path / "would_be_result_root"
    command = [
        sys.executable,
        "-S",
        str(ROOT / "scripts/run_pacd_smoke_v1.py"),
        "--execute",
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "execution-authorized-by-root" in completed.stderr
    assert not out_dir.exists()


def test_cli_fixes_num_workers_at_zero_before_any_execution():
    command = [
        sys.executable,
        "-S",
        str(ROOT / "scripts/run_pacd_smoke_v1.py"),
        "--execute",
        "--num-workers",
        "1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "num-workers=0" in completed.stderr


def test_reused_receipt_writer_keeps_attempt_and_terminal_immutable(tmp_path):
    from src.tfpd_lane import receipt

    attempt = tmp_path / "attempt.json"
    receipt.write_receipt_transactionally(attempt, {"status": "ATTEMPT_PUBLISHED"})
    assert attempt.is_file()
    assert attempt.with_suffix(".json.sha256").is_file()
    with pytest.raises(SystemExit) as error:
        receipt.write_receipt_transactionally(attempt, {"status": "MUTATED"})
    assert error.value.code == 2
    terminal = tmp_path / "terminal.json"
    receipt.write_receipt_transactionally(terminal, {"status": "COMPLETE"})
    assert terminal.is_file()


def test_failure_topology_is_distinct_from_success_terminal(tmp_path):
    from src.tfpd_lane import receipt

    receipt.write_receipt_transactionally(tmp_path / "attempt.json", {"status": "ATTEMPT_PUBLISHED"})
    smoke.publish_failure(tmp_path, receipt, {"status": "CELL_FAILED"})
    assert (tmp_path / "failure.json").is_file()
    assert not (tmp_path / "terminal.json").exists()
    with pytest.raises(smoke.SmokeError, match="immutable"):
        smoke.publish_failure(tmp_path, receipt, {"status": "CELL_FAILED_AGAIN"})


def test_canonical_out_dir_rejects_alternate_and_symlink_roots(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    expected = root / plan.RESULT_ROOT_RELATIVE
    assert smoke.require_canonical_fresh_out_dir(root, expected) == expected
    with pytest.raises(smoke.SmokeError, match="must equal"):
        smoke.require_canonical_fresh_out_dir(root, tmp_path / "elsewhere")
    target = tmp_path / "symlink-target"
    target.mkdir()
    expected.parent.mkdir(parents=True)
    expected.symlink_to(target, target_is_directory=True)
    with pytest.raises(smoke.SmokeError, match="symlink"):
        smoke.require_canonical_fresh_out_dir(root, expected)


def test_gpu_recheck_requires_exact_same_two_point_preflight_and_has_no_torch():
    before = {
        "physical_index": "0", "uuid": plan.EXPECTED_GPU_UUID,
        "name": "RTX", "driver_version": "x", "cuda_visible_devices": "0",
        "compute_processes": [], "gpu0_idle": True, "preflight_uses_torch": False,
    }
    calls = []

    def same():
        calls.append("post-attempt")
        return dict(before)

    assert smoke.recheck_gpu0_after_attempt(before, preflight=same) == before
    assert calls == ["post-attempt"]
    with pytest.raises(smoke.SmokeError, match="changed"):
        smoke.recheck_gpu0_after_attempt(
            before, preflight=lambda: {**before, "gpu0_idle": False}
        )


def test_torch_import_ordering_trap_runs_preflight_first():
    before = {"gpu0_idle": True}
    events = []

    def preflight():
        events.append("preflight")
        return {"gpu0_idle": True}

    def import_torch():
        events.append("torch-import")
        return object()

    observed, module = smoke.import_torch_after_gpu_recheck(
        before, preflight=preflight, import_torch=import_torch
    )
    assert observed == before
    assert module is not None
    assert events == ["preflight", "torch-import"]


def test_inherited_lr_helper_delegates_exactly_to_sealed_law():
    calls = []

    class ArmCommon:
        @staticmethod
        def lr_at_step(step, epochs, steps_per_epoch):
            calls.append((step, epochs, steps_per_epoch))
            return 0.125 + step / 1000

    assert smoke.inherited_lr_at_smoke_step(ArmCommon, step=7, steps_per_epoch=33925) == pytest.approx(0.132)
    assert calls == [(7, plan.EPOCHS_FOR_LR, 33925)]


def test_execution_closure_is_repo_relative_and_excludes_review_only_bytes():
    receipt = smoke.load_stdlib_receipt_module(REPO)

    closure = smoke.exact_source_closure(REPO, receipt)
    assert closure["files"]
    assert all(not path.startswith("../") for path in closure["files"])
    assert plan.WORK_ORDER_RELATIVE in closure["files"]
    assert "tfpd_exploration/src/__init__.py" in closure["files"]
    assert "tfpd_exploration/src/cal_aug_v1/__init__.py" in closure["files"]
    assert plan.DESIGN_RELATIVE not in closure["files"]
    assert "tfpd_exploration/tests/test_paired_anchored_calibration_dropout_v1.py" not in closure["files"]
    assert "sua_exploration/mc_maze/multisession_datamodule.py" in closure["files"]
    assert "streaming_calibration_exp/src/models/components/streaming_spint.py" in closure["files"]
    assert "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py" in closure["files"]
    assert "sua_exploration/mc_maze/datamodule.py" in closure["files"]
    assert "sua_exploration/mc_maze/__init__.py" in closure["files"]
    assert "streaming_calibration_exp/src/models/__init__.py" in closure["files"]
    assert "streaming_calibration_exp/src/models/components/__init__.py" in closure["files"]
    assert "streaming_calibration_exp/src/models/components/rt_ld_gain.py" in closure["files"]


def test_execution_closure_rejects_silently_missing_explicit_path():
    class IncompleteReceipt:
        @staticmethod
        def source_closure(root, patterns):
            del root, patterns
            return {"files": {plan.BOUND_PATTERNS[0]: {"sha256": "x"}}}

    with pytest.raises(smoke.SmokeError, match="missing"):
        smoke.exact_source_closure(REPO, IncompleteReceipt)


def test_cuda_synchronization_boundary_delegates_exact_device():
    calls = []

    class FakeCuda:
        @staticmethod
        def synchronize(device):
            calls.append(device)

    class FakeTorch:
        cuda = FakeCuda()

    smoke.synchronize_cuda(FakeTorch, "cuda:0")
    assert calls == ["cuda:0"]


def test_gpu0_preflight_accepts_only_bound_idle_identity_without_torch():
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        if "--query-gpu=index,uuid,name,driver_version" in command:
            return SimpleNamespace(
                stdout="0, GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9, RTX 3090, 535.1\n"
            )
        assert "--query-compute-apps=pid,process_name,used_memory" in command
        return SimpleNamespace(stdout="")

    payload = smoke.preflight_gpu0_idle(
        environ={"CUDA_VISIBLE_DEVICES": "0"}, run=fake_run, no_user_site=True
    )
    assert payload["gpu0_idle"] is True
    assert payload["uuid"] == plan.EXPECTED_GPU_UUID
    assert len(calls) == 2
    assert all(["-i", "0"] == command[1:3] for command in calls)


def test_gpu0_preflight_fails_closed_for_other_visibility_or_compute_process():
    with pytest.raises(smoke.SmokeError, match="CUDA_VISIBLE_DEVICES"):
        smoke.preflight_gpu0_idle(environ={"CUDA_VISIBLE_DEVICES": "1"}, no_user_site=True)

    def occupied_run(command, **kwargs):
        del kwargs
        if "--query-gpu=index,uuid,name,driver_version" in command:
            return SimpleNamespace(
                stdout="0, GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9, RTX 3090, 535.1\n"
            )
        return SimpleNamespace(stdout="12345, unrelated-process, 500 MiB\n")

    with pytest.raises(smoke.SmokeError, match="not idle"):
        smoke.preflight_gpu0_idle(
            environ={"CUDA_VISIBLE_DEVICES": "0"}, run=occupied_run, no_user_site=True
        )
