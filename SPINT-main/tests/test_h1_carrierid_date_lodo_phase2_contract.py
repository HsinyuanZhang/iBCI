"""No-GPU contract tests for H1 CarrierID date-LODO Phase-2 source wrappers."""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.h1_carrierid_date_lodo_phase2 import (
    CarrierIdDateLodoPhase2Error,
    H1CarrierIdDateLodoSchedule,
    H1CarrierIdDateLodoSourceDataModule,
    PHASE2_SOURCE_BINDING_SCHEMA,
    Phase2SourceBinding,
    _expected_normalizer_digest,
)
from src.models.h1_carrierid_date_lodo_phase2_module import (
    CarrierIdDateLodoPhase2ModelError,
    H1CarrierIdDateLodoPhase2LitModule,
)
from scripts.h1_carrierid_date_lodo_phase2_paired_launcher import (
    _compose_arm_config,
    _compose_experiment,
    _experiment_for_arm,
    _validate_arm_config,
    run as prepare_paired_source_launch,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file


class _TinySpint(torch.nn.Module):
    def forward(self, x, calib_trialized_neural_features=None, carrier=None):
        del calib_trialized_neural_features, carrier
        return x[..., :7]


class _TinyLazySpint(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = torch.nn.LazyLinear(7)

    def forward(self, x, calib_trialized_neural_features=None, carrier=None):
        del carrier
        pooled = calib_trialized_neural_features.mean(dim=(1, 2))
        return x[..., :7] + self.identity(pooled).unsqueeze(1)


def _tiny_module(*, arm: str = "H-S", outer_date: str = "19250108", seed: int = 42):
    return H1CarrierIdDateLodoPhase2LitModule(
        arm=arm,
        outer_date=outer_date,
        fixed_seed=seed,
        task="h1",
        net=_TinySpint(),
        decode_last_timestep_only=True,
        predict_scaled_behavior=False,
        behavior_scaling_factor=20.0,
        optimizer=partial(torch.optim.Adam, lr=1e-4),
        scheduler=None,
        compile=False,
        clean_teacher=True,
        scheduler_monitor="val_heldin/r2_mean",
    )


def _synthetic_binding() -> Phase2SourceBinding:
    session = "ses-19250108T110520"
    source_windows = tuple((session, index) for index in range(32))
    cache = SimpleNamespace(
        source_sessions=(session,),
        manifest={"cache_sha256": "f" * 64, "normalized_cache_sha256": "0" * 64},
    )
    return Phase2SourceBinding(
        outer_date="19250108",
        preflight_path=Path("/tmp/preflight.json"), preflight_sha256="a" * 64,
        source_manifest_path=Path("/tmp/manifest.json"), source_manifest_sha256="b" * 64,
        records={session: SimpleNamespace(input_sha256="1" * 64)}, cache=cache,
        normalizer=SimpleNamespace(normalizer_sha256="2" * 64),
        source_windows=source_windows, source_window_indices_sha256="c" * 64,
        batch_order=np.arange(32, dtype=np.int64), batch_order_sha256="d" * 64,
        calibration_schedule=np.full((50, 32), 7, dtype=np.int16), calibration_schedule_sha256="e" * 64,
        target_filename_index={"target_filenames_indexed_only": [], "target_recordings_opened": 0, "target_bytes_read": 0},
    )


def test_phase2_datamodule_rejects_fold0_and_any_nonfixed_source_contract() -> None:
    common = dict(task="h1", data_dir="/tmp/000954", phase1_preflight_path="/tmp/phase1.json")
    with pytest.raises(CarrierIdDateLodoPhase2Error, match="outer_date"):
        H1CarrierIdDateLodoSourceDataModule(**common, outer_date="19250101")
    with pytest.raises(CarrierIdDateLodoPhase2Error, match="fixed source data"):
        H1CarrierIdDateLodoSourceDataModule(**common, outer_date="19250108", fixed_epochs=49)
    with pytest.raises(CarrierIdDateLodoPhase2Error, match="fixed source data"):
        H1CarrierIdDateLodoSourceDataModule(**common, outer_date="19250108", seed=43)


def test_phase2_schedule_uses_immutable_e49_table_and_never_mixes_sessions() -> None:
    binding = _synthetic_binding()
    dataset = SimpleNamespace(window_indices=binding.source_windows)
    sampler = H1CarrierIdDateLodoSchedule(dataset, binding)
    first_epoch = list(iter(sampler))
    assert len(first_epoch) == 1
    assert [request[1] for request in first_epoch[0]] == [7] * 32
    for _ in range(49):
        list(iter(sampler))
    with pytest.raises(RuntimeError, match="epoch 49"):
        list(iter(sampler))


def test_phase2_model_rejects_fold0_wrong_seed_and_checkpoint_warm_start() -> None:
    with pytest.raises(CarrierIdDateLodoPhase2ModelError, match="outer_date"):
        _tiny_module(outer_date="19250101")
    with pytest.raises(CarrierIdDateLodoPhase2ModelError, match="seed=42"):
        _tiny_module(seed=43)
    with pytest.raises(CarrierIdDateLodoPhase2ModelError, match="warm-start"):
        _tiny_module().on_load_checkpoint({"epoch": 49})
    with pytest.raises(CarrierIdDateLodoPhase2ModelError, match="only the fixed terminal e49"):
        _tiny_module().on_save_checkpoint({"epoch": 48})


def test_phase2_model_hs_ignores_carrier_while_hc_requires_it() -> None:
    neural = torch.zeros((2, 700, 7), dtype=torch.float32)
    identity = torch.zeros((2, 4, 1024, 7), dtype=torch.float32)
    assert _tiny_module(arm="H-S")(neural, identity, None).shape == (2, 700, 7)
    with pytest.raises(CarrierIdDateLodoPhase2ModelError, match="requires normalized"):
        _tiny_module(arm="H-C")(neural, identity, None)


def test_phase2_first_source_batch_materializes_hs_lazy_parameters_before_initial_hashing() -> None:
    module = _tiny_module(arm="H-S")
    module.net = _TinyLazySpint()
    batch = (
        torch.zeros((2, 700, 7)), torch.zeros((2, 700, 7)), torch.zeros((2, 4, 1024, 7)),
        ["ses-19250108T110520", "ses-19250108T110520"], torch.zeros((2, 7, 4)),
    )
    assert isinstance(module.net.identity.weight, torch.nn.parameter.UninitializedParameter)
    module._materialize_fresh_parameters(batch)
    assert not isinstance(module.net.identity.weight, torch.nn.parameter.UninitializedParameter)


def test_phase2_binding_manifest_is_source_only_and_has_distinct_schema() -> None:
    binding = _synthetic_binding()
    manifest = binding.manifest()
    assert manifest["schema"] == PHASE2_SOURCE_BINDING_SCHEMA
    assert manifest["target_recordings_opened"] == 0
    assert manifest["target_bytes_read"] == 0
    assert manifest["warm_start_forbidden"] is True


def test_normalizer_digest_excludes_derived_denominator_but_binds_source_scalar() -> None:
    body = {
        "schema": "h1_carrierid_date_lodo_source_rms_normalizer_v1",
        "formula": "s_src=sqrt(mean(C_src_raw**2)); C_norm=C_raw/max(s_src,1e-12)",
        "floor": 1e-12,
        "source_cache_sha256": "a" * 64,
        "entries": 5,
        "rows": 176,
        "dims": 4,
        "s_src": 0.25,
        "denominator": 0.25,
    }
    digest = _expected_normalizer_digest(body)
    assert len(digest) == 64
    changed = dict(body, s_src=0.5)
    assert _expected_normalizer_digest(changed) != digest


def test_phase2_sources_do_not_reference_target_loader_or_cce_residual_path() -> None:
    root = Path(__file__).resolve().parents[1]
    combined = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "src/data/h1_carrierid_date_lodo_phase2.py",
            "src/models/h1_carrierid_date_lodo_phase2_module.py",
            "scripts/h1_carrierid_date_lodo_phase2_preflight.py",
            "scripts/h1_carrierid_date_lodo_phase2_paired_launcher.py",
        )
    )
    assert "load_target_records_for_date" not in combined
    assert "h1_m4_cce_spint" not in combined
    assert "import subprocess" not in combined
    assert "subprocess.run" not in combined
    assert '"--execute"' not in combined


def test_hs_runtime_probe_is_date_generic_explicit_gpu_source_only_and_keeps_cpu_preflight_separate() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/h1_carrierid_date_lodo_phase2_preflight.py").read_text(encoding="utf-8")
    assert "def run_hs_runtime_init_probe" in source
    assert "--run-hs-runtime-init-probe" in source
    assert "max_steps=1" in source
    assert "capture_precedes_training_step_and_first_optimizer_step" in source
    assert '"optimizer_steps_before_capture": 0' in source
    assert '"target_recordings_opened": 0' in source
    assert "Phase-2 CPU preflight requires CUDA_VISIBLE_DEVICES to be unset" in source
    assert 'outer_date = str(pair.get("outer_date", ""))' in source
    assert '"outer_date": outer_date' in source
    assert "runtime-init recovery is limited to the legacy 19250108" not in source


def test_pair_configs_compose_with_fixed_e49_no_validation_and_matched_19250108_binding() -> None:
    pair_preflight = (
        Path(__file__).resolve().parents[1] /
        "pilot_artifacts/h1_carrierid_date_lodo_phase2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIR_CPU_PREFLIGHT_v2.json"
    )
    if not pair_preflight.is_file():
        pytest.skip("real source-only CPU preflight is intentionally local, not a test fixture")
    for arm, experiment in (("H-S", "h1_carrierid_date_lodo_hs_19250108"), ("H-C", "h1_carrierid_date_lodo_hc_19250108")):
        row = _validate_arm_config(
            _compose_experiment(experiment), arm=arm, outer_date="19250108", pair_preflight=pair_preflight.resolve(),
        )
        assert row["fresh_seed"] == 42
        assert row["fixed_terminal_epoch_zero_based"] == 49
        assert row["calibration_support_trials"] == 4
        assert row["strict_target_query_starts_at_trial"] == 5
        assert row["limit_val_batches"] == row["num_sanity_val_steps"] == 0


@pytest.mark.parametrize("outer_date", CONFIRMATORY_DATES[1:])
def test_generic_pair_configs_take_the_date_and_pair_path_only_from_launcher_overrides(
    tmp_path: Path, outer_date: str,
) -> None:
    """This is a Hydra compose-only test: it constructs no DataModule or NWB view."""

    pair_preflight = tmp_path / f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_PAIR_CPU_PREFLIGHT.json"
    phase1_preflight = tmp_path / "phase1_source_only.json"
    for arm in ("H-S", "H-C"):
        experiment, cfg, overrides = _compose_arm_config(
            arm=arm, outer_date=outer_date, pair_preflight=pair_preflight,
            phase1_preflight=phase1_preflight,
        )
        assert experiment == _experiment_for_arm(arm=arm, outer_date=outer_date)
        assert experiment.endswith("_phase2")
        row = _validate_arm_config(
            cfg, arm=arm, outer_date=outer_date, pair_preflight=pair_preflight,
            phase1_preflight=phase1_preflight, experiment=experiment, compose_overrides=overrides,
        )
        assert row["fresh_seed"] == 42
        assert row["fixed_terminal_epoch_zero_based"] == 49
        assert row["epochs"] == 50
        assert row["calibration_support_trials"] == 4
        assert row["strict_target_query_starts_at_trial"] == 5
        assert row["limit_val_batches"] == row["num_sanity_val_steps"] == 0
        assert f"phase2.outer_date={outer_date}" in row["planned_command"]
        assert f"phase2.pair_preflight_path={pair_preflight}" in row["planned_command"]
        assert f"phase2.phase1_preflight_path={phase1_preflight}" in row["planned_command"]


def test_19250108_uses_the_existing_byte_frozen_configs_not_the_generic_template() -> None:
    assert _experiment_for_arm(arm="H-S", outer_date="19250108") == "h1_carrierid_date_lodo_hs_19250108"
    assert _experiment_for_arm(arm="H-C", outer_date="19250108") == "h1_carrierid_date_lodo_hc_19250108"


@pytest.mark.parametrize("outer_date", CONFIRMATORY_DATES[1:])
def test_generic_launcher_prepares_an_immutable_target_free_launch_receipt(tmp_path: Path, outer_date: str) -> None:
    """Exercise the full launcher using synthetic receipts; no DataModule is constructed."""

    phase1 = tmp_path / "phase1_source_only.json"
    phase1.write_text('{"source_only": true}\n', encoding="utf-8")
    phase1.chmod(0o444)
    source = {
        "outer_date": outer_date,
        "preflight_path": str(phase1),
        "preflight_sha256": sha256_file(phase1),
        "source_manifest_sha256": "b" * 64,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }
    pair = tmp_path / f"pair_{outer_date}.json"
    pair.write_text(json.dumps({
        "schema": "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1",
        "status": "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED",
        "outer_date": outer_date,
        "source_binding": source,
        "source_binding_sha256": canonical_sha256(source),
        "scope": {
            "target_recordings_opened": 0, "target_bytes_read": 0,
            "trainer_constructed_or_launched": False, "checkpoint_created_or_loaded": False,
            "cuda_constructed_or_launched": False,
        },
        "phase2_training_contract": {
            "arms": ["H-S", "H-C"], "fresh_seed": 42,
            "fixed_terminal_epoch_zero_based": 49, "epochs": 50,
            "checkpoint_warm_start_forbidden": True, "target_evaluator_status": "NOT_IMPLEMENTED",
        },
    }, sort_keys=True), encoding="utf-8")
    pair.chmod(0o444)
    result = prepare_paired_source_launch(pair_preflight=pair, output=tmp_path / "launch.json")
    assert result["outer_date"] == outer_date
    assert result["scope"]["target_recordings_opened"] == 0
    assert result["scope"]["cuda_constructed_or_launched"] is False
    for arm in ("H-S", "H-C"):
        row = result["pair"]["arms"][arm]
        assert row["experiment_config"].endswith("_phase2")
        assert row["fresh_seed"] == 42 and row["epochs"] == 50
        assert row["fixed_terminal_epoch_zero_based"] == 49
        assert row["calibration_support_trials"] == 4
        assert row["strict_target_query_starts_at_trial"] == 5
