"""CPU contracts for fold-local EMG-rSyn3 Stage-1. No live GPU in this module."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan
from tfpd_exploration.src.m1_h1_activity_headroom_v1 import core as headroom_core


ROOT = Path(__file__).resolve().parents[2]
STAGE1_CLI = ROOT / "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_stage1.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
FOLD0_M10_CARRIER_DIGEST = "2d1a638e4816aa3254f5b1817601ddc4515cd6a24f4d5aa03e61ee7a8513ffe7"


def test_stage1_public_cli_is_dry_and_execute_gpu_errors() -> None:
    dry = subprocess.run(
        [PYTHON, str(STAGE1_CLI)],
        check=True, capture_output=True, text=True, cwd=str(ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
             "PYTHONPATH": str(ROOT)},
    )
    payload = json.loads(dry.stdout)
    assert payload["opens_nwb_or_checkpoint"] is False
    assert payload["public_gpu_capability"] is False
    assert payload["public_execution_authorized"] is False
    assert payload["ls4"]["enabled_in_stage1_pilot"] is False
    gpu = subprocess.run(
        [PYTHON, str(STAGE1_CLI), "--execute-gpu"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
             "PYTHONPATH": str(ROOT)},
    )
    assert gpu.returncode != 0
    assert "cannot mint a GPU capability" in (gpu.stderr + gpu.stdout)


def test_stage1_refuses_sealed_result_roots() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import receipts as fold_receipts

    with pytest.raises(fold_receipts.ReceiptError, match="syn3_fcm_v1"):
        fold_receipts.refuse_sealed_roots(ROOT / "tfpd_exploration/results/m1_emg_syn3_fcm_v1/pilot/z_fix")
    with pytest.raises(fold_receipts.ReceiptError, match="rsyn3_fcm_v1"):
        fold_receipts.refuse_sealed_roots(ROOT / "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/pilot/z_fix")
    fold_receipts.refuse_sealed_roots(ROOT / "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot/z_fix")


def test_post_fc_in_injection_zero_p_is_null_and_nonzero_gets_grad() -> None:
    torch = pytest.importorskip("torch")
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import injection

    hidden = torch.ones(2, 3, 6)
    carrier = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    weight = injection.zero_projection_weight(6, torch)
    out_zero = injection.apply_post_fc_in(hidden, carrier, weight, mask)
    assert torch.equal(out_zero, hidden * mask.unsqueeze(-1))
    weight = torch.nn.Parameter(torch.zeros(6, 4))
    live = injection.apply_post_fc_in(hidden, carrier, weight, mask)
    live.sum().backward()
    assert weight.grad is not None
    assert float(weight.grad.abs().sum()) > 0.0
    with torch.no_grad():
        weight[0, 0] = 1.0
    shifted = injection.apply_post_fc_in(hidden, carrier, weight, mask)
    assert not torch.equal(shifted.detach(), out_zero)


def test_s_acyc_prefix_cycle_keeps_carrier_digest() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import prefix as stage1_prefix
    from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3

    carrier = np.arange(32, dtype=np.float64).reshape(8, 4)
    digest = rsyn3.array_digest(carrier)
    operator = stage1_prefix.ActivityPrefixOperator("S-Acyc")
    assert plan.ACTIVITY_CYCLE == (10, 5, 2)
    assert [operator.m_for_index(i) for i in range(6)] == [10, 5, 2, 10, 5, 2]
    for _ in range(3):
        assert rsyn3.array_digest(carrier) == digest
    fixed = stage1_prefix.ActivityPrefixOperator("S-Fix")
    assert all(fixed.m_for_index(i) == 10 for i in range(4))


def test_cdm_a_selection_is_rolling_fixed_m_causal() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import score as stage1_score

    assert stage1_score.CDM_A_ARM is headroom_core.ActivityArm.ROLLING_FIXED_M
    selection = headroom_core.selection_for_output_trial(
        headroom_core.ActivityArm.ROLLING_FIXED_M,
        output_trial_index=15, total_trials=210, support_trials=10,
    )
    assert selection == tuple(range(5, 15))
    assert all(index < 15 for index in selection)


def test_gpu_bind_refuses_compute_apps() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import gpu as stage1_gpu

    def busy(_arguments):
        query = " ".join(_arguments)
        if "compute-apps" in query:
            return f"{plan.GPU1_UUID}, 9, python, 1024\n"
        return (
            f"0, {plan.GPU0_UUID}, 453, 0\n"
            f"1, {plan.GPU1_UUID}, 23, 0\n"
        )

    with pytest.raises(stage1_gpu.GpuError, match="compute apps"):
        stage1_gpu.assert_target_gpu_launchable(1, cmdline_runner=busy)


def test_pack_allows_one_same_route_occupant_and_refuses_a_third() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import gpu as stage1_gpu

    own = "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_stage1.py --arm Z-Fix"
    rows = (
        f"0, {plan.GPU0_UUID}, 453, 0\n"
        f"1, {plan.GPU1_UUID}, 987, 54\n"
    )

    def runner(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return f"{plan.GPU1_UUID}, 11, python, 958\n"
        return rows

    receipt = stage1_gpu.assert_target_gpu_launchable(
        1, pack=True, cmdline_runner=runner, pid_cmdline=lambda pid: own if pid == "11" else "other",
    )
    assert receipt["pack"] is True
    assert receipt["own_occupants"] == 1

    def two_own(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return (
                f"{plan.GPU1_UUID}, 11, python, 958\n"
                f"{plan.GPU1_UUID}, 12, python, 958\n"
            )
        return rows

    with pytest.raises(stage1_gpu.GpuError, match="pack limit"):
        stage1_gpu.assert_target_gpu_launchable(
            1, pack=True, cmdline_runner=two_own,
            pid_cmdline=lambda pid: own,
        )


def test_pack_refuses_foreign_compute_app() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import gpu as stage1_gpu

    def runner(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return f"{plan.GPU1_UUID}, 99, python, 1024\n"
        return (
            f"0, {plan.GPU0_UUID}, 453, 0\n"
            f"1, {plan.GPU1_UUID}, 987, 54\n"
        )

    with pytest.raises(stage1_gpu.GpuError, match="foreign"):
        stage1_gpu.assert_target_gpu_launchable(
            1, pack=True, cmdline_runner=runner,
            pid_cmdline=lambda pid: "python tfpd_exploration/scripts/run_pit_m2_v1.py",
        )


def test_fixed_last_checkpoint_resolves_under_real_lightning_naming(tmp_path) -> None:
    """Lightning 2.4 writes ``epoch_epoch=011.ckpt``; resolution must not guess the name."""
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("lightning.pytorch")
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset

    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import stage1

    class Tiny(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)

        def training_step(self, batch, _index):
            features, target = batch
            return ((self.layer(features) - target) ** 2).mean()

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    loader = DataLoader(TensorDataset(torch.zeros(8, 2), torch.zeros(8, 1)), batch_size=4)
    callback = ModelCheckpoint(
        dirpath=str(tmp_path), filename="epoch_{epoch:03d}", monitor=None,
        save_top_k=-1, every_n_epochs=plan.STAGE1_EPOCHS, save_last=False,
    )
    trainer = pl.Trainer(
        accelerator="cpu", devices=1,
        max_epochs=plan.STAGE1_EPOCHS, min_epochs=plan.STAGE1_EPOCHS,
        limit_val_batches=0, num_sanity_val_steps=0, logger=False,
        enable_checkpointing=True, callbacks=[callback],
        enable_progress_bar=False, enable_model_summary=False, deterministic=True,
    )
    trainer.fit(Tiny(), train_dataloaders=loader)

    selected = f"epoch_{plan.FIXED_LAST_EPOCH_INDEX:03d}"
    written = sorted(path.name for path in tmp_path.glob("*.ckpt"))
    assert len(written) == 1, written
    # The legacy hardcoded path is exactly what failed in pilot_r2.
    assert not (tmp_path / f"{selected}.ckpt").is_file() or written == [f"{selected}.ckpt"]

    resolved, payload = stage1.resolve_fixed_last_checkpoint(tmp_path, selected, torch)
    assert resolved.name == written[0]
    assert int(payload["epoch"]) == plan.FIXED_LAST_EPOCH_INDEX
    assert "state_dict" in payload

    with pytest.raises(stage1.Stage1Error, match="selection token drift"):
        stage1.resolve_fixed_last_checkpoint(tmp_path, "epoch_010", torch)

    (tmp_path / "epoch_epoch=005.ckpt").write_bytes(b"decoy")
    with pytest.raises(stage1.Stage1Error, match="exactly one checkpoint"):
        stage1.resolve_fixed_last_checkpoint(tmp_path, selected, torch)


def test_fixed_last_checkpoint_rejects_wrong_epoch_payload(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import stage1

    selected = f"epoch_{plan.FIXED_LAST_EPOCH_INDEX:03d}"
    good = f"epoch_epoch={plan.FIXED_LAST_EPOCH_INDEX:03d}.ckpt"
    torch.save({"epoch": plan.FIXED_LAST_EPOCH_INDEX - 1, "state_dict": {}}, tmp_path / good)
    with pytest.raises(stage1.Stage1Error, match="payload epoch"):
        stage1.resolve_fixed_last_checkpoint(tmp_path, selected, torch)

    (tmp_path / good).unlink()
    torch.save({"epoch": plan.FIXED_LAST_EPOCH_INDEX}, tmp_path / good)
    with pytest.raises(stage1.Stage1Error, match="state_dict"):
        stage1.resolve_fixed_last_checkpoint(tmp_path, selected, torch)

    (tmp_path / good).unlink()
    torch.save({"epoch": plan.FIXED_LAST_EPOCH_INDEX, "state_dict": {}}, tmp_path / "no_digits.ckpt")
    with pytest.raises(stage1.Stage1Error, match="epoch digits"):
        stage1.resolve_fixed_last_checkpoint(tmp_path, selected, torch)


def test_fold0_carrier_bank_target_m10_digest_and_query_unread() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import carrier_bank
    from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3

    bank = carrier_bank.build_fold0_carrier_bank(ROOT)
    assert bank["isolation"]["target_query_values_read"] is False
    assert bank["target_session"] == plan.FOLD0_TARGET_SESSION
    target = bank["normalized"][plan.FOLD0_TARGET_SESSION]["rSyn3"]
    zero = bank["normalized"][plan.FOLD0_TARGET_SESSION]["Zero4"]
    assert rsyn3.array_digest(bank["raw"][plan.FOLD0_TARGET_SESSION]) == FOLD0_M10_CARRIER_DIGEST
    assert np.array_equal(zero, np.zeros_like(zero))
    for name in plan.FOLD0_SOURCE_SESSIONS:
        source = bank["normalized"][name]["rSyn3"]
        assert source.shape[1] == 4
        assert np.isfinite(source).all()
        assert not np.allclose(source, 0.0)
