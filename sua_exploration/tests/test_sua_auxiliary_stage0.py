"""CPU-only contracts for SUA auxiliary Stage-0 mechanisms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.sua_auxiliary_stage0 import (  # noqa: E402
    ParameterMatchedConcatMLP,
    ParameterMatchedNoGroupMLP,
    SegmentedMeanResidual,
    ZeroInitLowRankFiLM,
    calibration_quality_features,
    design_rank_and_condition,
    deterministic_membership_shuffle,
    deterministic_row_shuffle,
)


def test_quality_features_are_finite_and_include_rank_condition_contract():
    quality = calibration_quality_features(
        p2p=np.array([2.0, np.nan]), noise_std=np.array([1.0, np.inf]),
        snr=np.array([2.0, 3.0]), waveform_residual_cv=np.array([0.1, 0.2]),
        waveform_template_drift=np.array([0.3, 0.4]), spike_exposure=np.array([10, 0]),
        t4_relative_residual=np.array([0.5, 0.6]), design_condition=float("inf"), rank_valid=False,
    )
    assert quality.shape == (2, 9)
    assert np.isfinite(quality).all()
    assert np.array_equal(quality[:, -1], np.zeros(2, dtype=np.float32))

    rank, condition = design_rank_and_condition(np.array([0, 1, 2]), [0.0, 1.0, 2.0])
    assert rank == 3 and np.isfinite(condition)
    deficient_rank, deficient_condition = design_rank_and_condition(np.array([0, 0]), [0.0])
    assert deficient_rank == 1 and np.isinf(deficient_condition)


def test_row_and_membership_controls_keep_their_declared_marginals():
    rows = np.arange(24).reshape(6, 4)
    shuffled = deterministic_row_shuffle(rows, seed=43)
    assert np.array_equal(np.sort(shuffled, axis=0), np.sort(rows, axis=0))
    ids = np.array([9, 9, 9, 2, 2, 5])
    membership = deterministic_membership_shuffle(ids, seed=43)
    assert sorted(np.unique(membership, return_counts=True)[1].tolist()) == [1, 2, 3]
    assert not np.array_equal(
        membership[:, None] == membership[None, :], ids[:, None] == ids[None, :]
    )


def test_film_and_concat_are_exact_baselines_at_initialization_and_near_parameter_matched():
    torch.manual_seed(0)
    activity = torch.randn(2, 7, 64)
    confidence = torch.randn(2, 7, 9)
    film = ZeroInitLowRankFiLM(64, 9, rank=8)
    concat = ParameterMatchedConcatMLP(64, 9, film_rank=8)
    assert torch.equal(film(activity, confidence), activity)
    assert torch.equal(concat(activity, confidence), activity)
    # Integer hidden widths cannot exactly match the FiLM count without dummy
    # parameters; the real nonlinear control stays within a declared 10%.
    assert abs(concat.parameter_gap) / concat.target_parameter_count < 0.10


def test_segmented_relation_is_permutation_equivariant_and_exactly_singleton_degenerate():
    torch.manual_seed(1)
    activity = torch.randn(1, 5, 6)
    confidence = torch.randn(1, 5, 3)
    relation = SegmentedMeanResidual(6, 3, relation_dim=4)
    # Make the head non-zero: singleton behaviour must still be exact because its input is zero.
    with torch.no_grad():
        relation.output.weight.fill_(0.25)
    singletons = torch.arange(5).view(1, 5)
    assert torch.equal(relation(activity, confidence, singletons), activity)
    groups = torch.tensor([[3, 3, 1, 1, 7]])
    order = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.argsort(order)
    observed = relation(activity, confidence, groups)
    permuted = relation(activity[:, order], confidence[:, order], groups[:, order])
    assert torch.allclose(permuted[:, inverse], observed)
    no_group = ParameterMatchedNoGroupMLP(6, 3, relation_dim=4)
    assert sum(p.numel() for p in no_group.parameters()) == sum(p.numel() for p in relation.parameters())


def test_singleton_boundary_survives_an_optimizer_step():
    torch.manual_seed(11)
    relation = SegmentedMeanResidual(5, 2, relation_dim=3)
    activity = torch.randn(1, 4, 5)
    context = torch.randn(1, 4, 2)
    optimizer = torch.optim.Adam(relation.parameters(), lr=1e-2)
    # Non-singleton update makes the relation head genuinely nonzero.
    groups = torch.tensor([[0, 0, 1, 1]])
    loss = relation(activity, context, groups).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert not torch.equal(relation.output.weight, torch.zeros_like(relation.output.weight))
    singletons = torch.arange(4).view(1, 4)
    assert torch.equal(relation(activity, context, singletons), activity)


def test_strict_manifest_initialization_never_discovers_or_counts_test_nwbs(monkeypatch):
    """The strict route may resolve train/val paths but must not touch test files."""
    from mc_maze import multisession_datamodule as md

    manifest = Path(__file__).resolve().parents[1] / "configs/subc_co_27_6_strict_train_val_manifest.json"
    data_dir = Path(__file__).resolve().parents[1] / "data/dandi_000688/sub-C"
    monkeypatch.setattr(md, "discover_nwb_files", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("discovery forbidden")))
    monkeypatch.setattr(md, "nwb_unit_count", lambda _path: 1)
    dm = md.Dandi688MultiSessionDataModule(
        data_dir=str(data_dir), task="CO", split_counts=(27, 6, 6), max_units_exclusive=100,
        train_val_manifest_path=str(manifest), num_workers=0,
    )
    dm._initialize_splits()
    assert len(dm.session_files["train"]) == 27
    assert len(dm.session_files["val"]) == 6
    assert dm.session_files["test"] == []
    assert dm.session_splits["test"] == [
        "sub-C_ses-CO-20151113", "sub-C_ses-CO-20151116", "sub-C_ses-CO-20151117",
        "sub-C_ses-CO-20151119", "sub-C_ses-CO-20151120", "sub-C_ses-CO-20151201",
    ]


def test_relation_epoch_window_contract_keeps_train10_and_eval30_distinct():
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from eval_epoch_window_generic_dandi688 import validate_relation_calibration_contract

    metadata = {
        "side_features": {"uses_equality_only_relation_membership": True},
        "training": {"calibration_n_trials": 10},
    }
    validate_relation_calibration_contract(metadata, 30)
    with pytest.raises(ValueError, match="evaluation.forward_calibration_n=30"):
        validate_relation_calibration_contract(metadata, 10)
    with pytest.raises(ValueError, match="training.activity_calibration_n_trials=10"):
        validate_relation_calibration_contract(
            {"side_features": {"uses_equality_only_relation_membership": True}, "training": {"calibration_n_trials": 30}},
            30,
        )
    with pytest.raises(ValueError, match="evaluation.forward_calibration_n=30"):
        validate_relation_calibration_contract(
            {
                "variant": "B3SERN",
                "side_features": {
                    "group": "t4rel_nogroup",
                    "uses_equality_only_relation_membership": False,
                },
                "training": {"calibration_n_trials": 10},
            },
            10,
        )


def test_epoch_window_pool_must_cover_calibration_prefix():
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from eval_epoch_window_generic_dandi688 import validate_prefix_budget

    validate_prefix_budget(30, 30)
    with pytest.raises(ValueError, match="pool_size"):
        validate_prefix_budget(30, 29)
    with pytest.raises(ValueError, match="calibration_n"):
        validate_prefix_budget(0, 30)


def test_no_group_dataset_has_no_membership_batch_field(monkeypatch, tmp_path):
    """The parameter-matched no-group arm receives only T4 side features."""
    from mc_maze import multisession_datamodule as md
    from mc_maze import unit_side_features as usf

    train_path = tmp_path / "train.nwb"
    val_path = tmp_path / "val.nwb"
    membership = np.array([9, 9, 3, 3], dtype=np.int64)

    def fake_record(path, **_kwargs):
        return md.SessionRecord(
            name=path.stem,
            neural=np.zeros((60, 4), dtype=np.float32),
            behavior=np.zeros((60, 2), dtype=np.float32),
            calib_trials=np.zeros((10, 100, 4), dtype=np.float32),
            valid_starts=np.array([0], dtype=np.int64),
            channel_ids=np.arange(4, dtype=np.int64),
        )

    monkeypatch.setattr(md, "discover_nwb_files", lambda *_args, **_kwargs: [train_path, val_path])
    monkeypatch.setattr(md, "chronological_session_split", lambda *_args, **_kwargs: ([train_path], [val_path], []))
    monkeypatch.setattr(md, "session_name_from_path", lambda path: path.stem)
    monkeypatch.setattr(md, "nwb_unit_count", lambda _path: 4)
    monkeypatch.setattr(md, "fit_behavior_stats", lambda *_args, **_kwargs: (np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)))
    monkeypatch.setattr(md, "load_dandi688_session", fake_record)
    monkeypatch.setattr(usf, "fit_side_feature_stats", lambda *_args, **_kwargs: (np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)))
    monkeypatch.setattr(usf, "load_unit_side_features", lambda *_args, **_kwargs: (np.zeros((4, 4), dtype=np.float32), None))
    monkeypatch.setattr(usf, "load_session_electrode_ids", lambda _path: membership.copy())

    def build(token, seed=None):
        dm = md.Dandi688MultiSessionDataModule(
            data_dir=str(tmp_path), task="CO", split_counts=(1, 1, 0),
            max_units_exclusive=None, window_size=50, batch_size=1, num_workers=0,
            side_feature_group=token, side_permutation_seed=seed,
        )
        dm.setup("fit")
        return dm

    relation = build("t4rel")
    shuffled = build("t4rel_membership_shuffled", seed=42)
    no_group = build("t4rel_nogroup")
    relation_ids = relation.train_dataset.sessions[train_path.stem].electrode_ids
    shuffled_ids = shuffled.train_dataset.sessions[train_path.stem].electrode_ids
    assert len(relation.train_dataset[0]) == 6
    assert len(shuffled.train_dataset[0]) == 6
    assert len(no_group.train_dataset[0]) == 5
    assert no_group.train_dataset.sessions[train_path.stem].electrode_ids is None
    assert sorted(np.unique(relation_ids, return_counts=True)[1].tolist()) == sorted(
        np.unique(shuffled_ids, return_counts=True)[1].tolist()
    )
    assert not np.array_equal(
        relation_ids[:, None] == relation_ids[None, :],
        shuffled_ids[:, None] == shuffled_ids[None, :],
    )
    assert not usf.uses_electrode_relation_membership("t4rel_nogroup")
    assert not usf.uses_electrode_ids("t4rel_nogroup")
    # REL-NG must reuse the exact T4 normalization/cache substrate while
    # receiving neither membership nor absolute electrode IDs.
    assert usf.base_feature_group("t4rel_nogroup") == "t4"


def test_strict_manifest_hash_drift_is_rejected(tmp_path):
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from eval_epoch_window_generic_dandi688 import (
        sha256_file,
        validate_strict_manifest_provenance,
    )

    manifest = tmp_path / "strict.json"
    manifest.write_text('{"revision": 1}\n', encoding="utf-8")
    metadata = {
        "train_val_manifest": str(manifest.resolve()),
        "train_val_manifest_sha256": sha256_file(manifest),
    }
    validate_strict_manifest_provenance(metadata, manifest.resolve())
    manifest.write_text('{"revision": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_strict_manifest_provenance(metadata, manifest.resolve())


def test_relation_pilot_aggregator_uses_relation_as_each_treatment():
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from aggregate_sua_electrode_relation_pilot import compute_relation_control_deltas

    records = {
        "t4": {"session": 1.0},
        "relation": {"session": 4.0},
        "membership_shuffle": {"session": 2.0},
        "no_group": {"session": 3.0},
    }
    observed = compute_relation_control_deltas(records)
    assert observed["REL_minus_T4"]["mean_paired_delta_r2"] == 3.0
    assert observed["REL_minus_REL_MS"]["treatment"] == "relation"
    assert observed["REL_minus_REL_MS"]["control"] == "membership_shuffle"
    assert observed["REL_minus_REL_MS"]["mean_paired_delta_r2"] == 2.0
    assert observed["REL_minus_REL_NG"]["control"] == "no_group"
    assert observed["REL_minus_REL_NG"]["mean_paired_delta_r2"] == 1.0


def test_relation_multiseed_decision_requires_all_three_relation_wins():
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from aggregate_sua_electrode_relation_multiseed import (
        compute_multiseed_decision,
    )

    sessions = [f"session_{idx}" for idx in range(6)]
    records = {}
    for seed, jitter in ((42, 0.0), (43, 0.002), (44, -0.002)):
        records[seed] = {
            "t4": {session: 0.50 + jitter for session in sessions},
            "relation": {session: 0.56 + jitter for session in sessions},
            "membership_shuffle": {
                session: 0.51 + jitter for session in sessions
            },
            "no_group": {session: 0.52 + jitter for session in sessions},
        }

    observed = compute_multiseed_decision(records)
    assert observed["paired_deltas"]["REL_minus_T4"]["mean_delta_r2"] == pytest.approx(0.06)
    assert observed["paired_deltas"]["REL_minus_REL_MS"]["mean_delta_r2"] == pytest.approx(0.05)
    assert observed["paired_deltas"]["REL_minus_REL_NG"]["mean_delta_r2"] == pytest.approx(0.04)
    assert all(
        pair["verdict"] == "effective"
        for pair in observed["paired_deltas"].values()
    )
    assert observed["relation_group_verdict"]["verdict"] == "effective"
    assert observed["relation_group_verdict"]["advance_to_relative_amplitude"] is True

    # Losing even one necessary matched control rejects the group and amplitude
    # advancement, even if REL beats T4 by a large margin.
    for seed in records:
        records[seed]["no_group"] = {
            session: records[seed]["relation"][session]
            for session in sessions
        }
    observed = compute_multiseed_decision(records)
    assert observed["paired_deltas"]["REL_minus_REL_NG"]["verdict"] == "ineffective"
    assert observed["relation_group_verdict"]["verdict"] == "ineffective"
    assert observed["relation_group_verdict"]["advance_to_relative_amplitude"] is False
