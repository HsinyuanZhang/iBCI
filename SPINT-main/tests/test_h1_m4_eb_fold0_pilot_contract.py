from __future__ import annotations

import json
from pathlib import Path
import random

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from src.data.h1_m4_eb_fold0_datamodule import H1M4EBFold0DataModule
from src.data.h1_m4_eb_pilot import (
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    H1_M4_FOLD0_TARGET,
    H1M4EBPairedBatchSampler,
    H1M4EBSourceDataset,
    H1M4EBStrictTargetDataset,
    carrier_sha256,
    legal_contiguous_starts,
    load_target_records,
    query_mutation_invariance,
    reject_path_scope,
    validate_target_receipt_binding,
)
from src.h1_m4_eb_pilot_contract import load_and_validate_terminal_checkpoint, state_hash
from src.models.components.h1_m4_eb_residual_spint import H1M4EBResidualSpint


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA = ROOT / "data/000954"
RAW = WORKSPACE / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
EB = WORKSPACE / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"


@pytest.fixture(scope="session")
def real_dm(tmp_path_factory):
    cache = tmp_path_factory.mktemp("h1_m4_eb_cache")
    module = H1M4EBFold0DataModule(
        task="h1",
        data_dir=str(DATA),
        raw_receipt_path=str(RAW),
        eb_receipt_path=str(EB),
        cache_dir=str(cache),
    )
    module.setup("fit")
    return module


@pytest.fixture(scope="session")
def real_target(real_dm):
    records = load_target_records(DATA)
    validate_target_receipt_binding(records, real_dm.plan, RAW, EB)
    return records, H1M4EBStrictTargetDataset(records, real_dm.plan, "full")


def _small_net(train_residual: bool):
    return H1M4EBResidualSpint(
        train_residual=train_residual,
        model_dim=32,
        num_covariates=7,
        window_size=8,
        num_heads=4,
        num_layers=1,
        num_id_layers=2,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
        dropout_rate=0.0,
        dynamic_dropout=True,
        dynamic_dropout_low=0.0,
        dynamic_dropout_high=1.0,
        tf_drop_rate=0.1,
        readin_layer_type="mlp",
    )


def _materialize(net, calibration):
    lazy = net.fc_id_in[0]
    if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
        lazy.initialize_parameters(calibration.permute(0, 1, 3, 2))


def test_date_exclusion_scope_and_receipt_binding(real_dm):
    assert tuple(real_dm.records) == H1_M4_FOLD0_SOURCE
    assert real_dm.plan.source_sessions == H1_M4_FOLD0_SOURCE
    assert all(record.date != FOLD0_DATE for record in real_dm.records.values())
    assert real_dm.pilot_manifest()["target_nwb_opened_during_training_setup"] is False
    assert real_dm.plan.raw_receipt_sha256.startswith("660f78")
    assert real_dm.plan.eb_receipt_sha256.startswith("130045")
    with pytest.raises(ValueError):
        reject_path_scope(DATA / "sub-HumanPitt-held-in-minival/x.nwb")


def test_all_legal_starts_cached_and_trialnum_identity_carrier_alignment(real_dm):
    dataset = real_dm.train_dataset
    for name in H1_M4_FOLD0_SOURCE:
        expected = legal_contiguous_starts(real_dm.records[name])
        assert real_dm.carrier_cache.starts_by_session[name] == expected
        for start in expected:
            entry = real_dm.carrier_cache.get(name, start)
            assert entry.trial_values == real_dm.records[name].trial_values[start : start + 4]
            index = next(i for i, row in enumerate(dataset.window_indices) if row[0] == name)
            sample = dataset[(index, start)]
            assert sample[2].shape == (4, 1024, 176)
            assert sample[4].shape == (176, 4)
            np.testing.assert_array_equal(sample[4], entry.carrier.astype(np.float32))
            break


def _independent_frozen_reference(record, plan, values):
    trials = [record.blocks_for(value) for value in values]
    rates = np.concatenate([trial.rates for trial in trials])
    labels = np.concatenate([trial.velocity for trial in trials])
    z = ((rates - plan.mean) / plan.scale) @ plan.pcs[: plan.q].T
    design = np.c_[np.ones(len(z)), z]
    penalty = np.eye(design.shape[1]) * plan.ridge_lambda
    penalty[0, 0] = 0.0
    system = design.T @ design + penalty
    beta = np.linalg.solve(system, design.T @ labels)
    residual = labels - design @ beta
    sigma2 = np.square(residual).sum(0) / (len(design) - np.trace(design @ np.linalg.solve(system, design.T)))
    G = np.linalg.solve(system, design.T @ design) @ np.linalg.inv(system)
    G = (G + G.T) / 2
    raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]
    raw = raw_rows @ plan.U
    projection = plan.pcs[: plan.q].T
    factor = ((projection @ G[1:, 1:]) * projection).sum(1) / np.square(plan.scale)
    variance = factor * np.trace(plan.U.T @ np.diag(sigma2) @ plan.U) / 4
    weight = plan.tau2 / (plan.tau2 + variance)
    return plan.mu + weight[:, None] * (raw - plan.mu)


def test_representative_cached_carrier_matches_independent_frozen_estimator(real_dm):
    name = H1_M4_FOLD0_SOURCE[0]
    start = real_dm.carrier_cache.starts_by_session[name][3]
    entry = real_dm.carrier_cache.get(name, start)
    reference = _independent_frozen_reference(real_dm.records[name], real_dm.plan, entry.trial_values)
    np.testing.assert_array_equal(entry.carrier, reference)


def test_ordered_batches_and_m4_schedule_are_tied(real_dm):
    first = H1M4EBPairedBatchSampler(real_dm.train_dataset, cache_dir=None)
    second = H1M4EBPairedBatchSampler(real_dm.train_dataset, cache_dir=None)
    assert first.batch_order_sha256 == second.batch_order_sha256
    assert first.schedule_sha256 == second.schedule_sha256
    assert np.array_equal(first.schedule, second.schedule)
    for left, right in zip(first, second):
        assert left == right
        for window_index, start in left:
            session = real_dm.train_dataset.window_indices[window_index][0]
            assert start in real_dm.carrier_cache.starts_by_session[session]
        break


def test_strict_target_boundary_interventions_and_query_rebuild_invariance(real_target):
    records, full = real_target
    hashes = full.support_and_carrier_hashes()
    for name in H1_M4_FOLD0_TARGET:
        support = full.support[name]
        assert support.trial_values == records[name].trial_values[:4]
        assert support.fifth_trial == records[name].trial_values[4]
        assert all(start >= support.query_first_bin for session, start in full.window_indices if session == name)
        assert len(set(hashes[name]["carrier_sha256"].values())) == 4
    for intervention in full.INTERVENTIONS:
        variant = full.with_intervention(intervention)
        assert variant.window_indices_sha256 == full.window_indices_sha256
        assert variant.support_and_carrier_hashes() == hashes
    evidence = query_mutation_invariance(full)
    assert evidence["invariant"] and evidence["query_data_changed"]
    assert evidence["original"] == evidence["rebuilt_after_trial5plus_mutation"]


def test_zero_residual_shared_state_and_dynamic_dropout_parity_after_lazy_materialization():
    random.seed(42)
    torch.manual_seed(42)
    base = _small_net(False)
    constructor_rng = torch.get_rng_state()
    random.seed(42)
    torch.manual_seed(42)
    joint = _small_net(True)
    x = torch.randn(2, 8, 5)
    calibration = torch.randn(2, 4, 8, 5)
    carrier = torch.randn(2, 5, 4)
    torch.set_rng_state(constructor_rng)
    _materialize(base, calibration)
    lazy_rng = torch.get_rng_state()
    torch.set_rng_state(constructor_rng)
    _materialize(joint, calibration)
    assert state_hash(base.state_dict()) == state_hash(joint.state_dict())
    base.train()
    joint.train()
    random.seed(99)
    torch.manual_seed(99)
    base_output = base(x, calibration, carrier)
    random.seed(99)
    torch.manual_seed(99)
    joint_output = joint(x, calibration, carrier)
    assert torch.equal(base_output, joint_output)
    assert torch.count_nonzero(base.eb_residual).item() == 0
    assert base.eb_residual.requires_grad is False and joint.eb_residual.requires_grad is True


def test_real_data_source_batch_cpu_smoke(real_dm):
    real_dm.train_batch_sampler.reset_epoch()
    batch = next(iter(real_dm.train_dataloader()))
    assert len(batch) == 5
    assert batch[0].shape == (32, 700, 176)
    assert batch[1].shape == (32, 700, 7)
    assert batch[2].shape == (32, 4, 1024, 176)
    assert batch[4].shape == (32, 176, 4)
    assert len(set(batch[3])) == 1


@pytest.mark.parametrize("arm", ["base", "joint"])
def test_hydra_compose_and_instantiate(arm, tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs"), job_name="pilot_test"):
        config = compose(
            config_name="train.yaml",
            overrides=[
                f"experiment=h1_m4_eb_fold0_exploratory_pilot_{arm}",
                "trainer.accelerator=cpu",
                f"pilot.shared_cache_dir={tmp_path}",
                f"paths.root_dir={ROOT}",
                f"paths.work_dir={ROOT}",
                f"paths.data_dir={ROOT / 'data'}",
                f"paths.output_dir={tmp_path / 'output'}",
            ],
        )
    assert config.pilot.arm == arm and config.trainer.accelerator == "cpu"
    datamodule = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    assert isinstance(datamodule, H1M4EBFold0DataModule)
    assert model.pilot_arm == arm


def test_terminal_checkpoint_failure_cases(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("pilot: invalid\n")
    checkpoint = tmp_path / "invalid.ckpt"
    torch.save({"epoch": 48, "global_step": 10, "state_dict": {"net.eb_residual": torch.zeros(4, 700)}}, checkpoint)
    with pytest.raises(ValueError, match="epoch 49"):
        load_and_validate_terminal_checkpoint(checkpoint, config, expected_arm="base")
    torch.save({"epoch": 49, "global_step": 0, "state_dict": {"net.eb_residual": torch.zeros(4, 700)}}, checkpoint)
    with pytest.raises(ValueError, match="positive real training"):
        load_and_validate_terminal_checkpoint(checkpoint, config, expected_arm="base")


def test_terminal_evaluator_rejects_invalid_device_before_target_access(tmp_path):
    from scripts.h1_m4_eb_fold0_pilot_evaluate import evaluate_terminal_pilot

    with pytest.raises(ValueError, match="cuda or cpu"):
        evaluate_terminal_pilot(
            data_dir=DATA,
            raw_receipt_path=RAW,
            eb_receipt_path=EB,
            shared_cache_dir=tmp_path,
            base_checkpoint_path=tmp_path / "missing-base.ckpt",
            joint_checkpoint_path=tmp_path / "missing-joint.ckpt",
            base_config_path=tmp_path / "missing-base.yaml",
            joint_config_path=tmp_path / "missing-joint.yaml",
            output_path=tmp_path / "must-not-exist.json",
            device="tpu",
        )
    assert not (tmp_path / "must-not-exist.json").exists()
