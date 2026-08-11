"""Synthetic, no-NWB contracts for the independent H1 CI64 terminal verifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest
import torch

from scripts import h1_carrierid_date_lodo_ci_fivedate_terminal_verifier_v2 as verifier


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


def _write(path: Path, value: object, *, immutable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    if immutable:
        path.chmod(0o444)
    return path


def _write_text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _chmod_mutable(path: Path) -> None:
    path.chmod(0o644)


def _remote(local: Path, root: Path) -> str:
    return "/remote-ci/" + str(local.resolve().relative_to(root.resolve())).replace("\\", "/")


def _resolved_config(*, arm: str, date: str, phase1: str, preflight: str, aggregate: str) -> str:
    width = 32 if arm == "CI32-FULL" else 64
    intervention = arm.split("-", 1)[1].lower()
    zero = "true" if arm == "CI64-C0" else "false"
    token = arm.lower().replace("-", "_")
    return "\n".join((
        f"protocol_id: h1_carrierid_date_lodo_{token}_{date}_source_only_v1", "train: true", "test: false",
        "ckpt_path: null", "seed: 42", "phase_ci:", f"  outer_date: '{date}'", f"  arm: '{arm}'",
        f"  carrier_intervention: '{intervention}'", f"  phase1_preflight_path: '{phase1}'", f"  ci_preflight_path: '{preflight}'",
        f"  five_date_aggregate_path: '{aggregate}'", "  warm_start_forbidden: true", "data:",
        "  _target_: src.data.h1_carrierid_date_lodo_ci.H1CarrierIdDateLodoCiDataModule", f"  ci_arm: '{arm}'",
        f"  carrier_intervention: '{intervention}'", f"  ci_preflight_path: '{preflight}'",
        f"  five_date_aggregate_path: '{aggregate}'", "model:",
        "  _target_: src.models.h1_carrierid_date_lodo_ci_module.H1CarrierIdDateLodoCiLitModule", f"  arm: '{arm}'",
        f"  outer_date: '{date}'", "  fixed_seed: 42", f"  ci_preflight_path: '{preflight}'",
        f"  five_date_aggregate_path: '{aggregate}'", "  net:",
        "    _target_: src.models.components.h1_carrierid_ci_spint.H1CarrierIdCiSpint",
        "    carrier_hidden_dim: 32", f"    carrier_interface_dim: {width}", f"    zero_carrier: {zero}",
        "trainer:", "  max_epochs: 50", "  min_epochs: 50", "  accelerator: gpu", "  devices: 1", "  precision: 32-true",
        "  limit_val_batches: 0", "  num_sanity_val_steps: 0",
        "callbacks:", "  fixed_epoch50:", "    monitor: null", "    every_n_epochs: 50", "    save_top_k: -1", "    save_last: false",
    )) + "\n"


def _static_config(*, arm: str) -> str:
    """The arm selector, intentionally distinct from a resolved run config."""

    token = arm.lower().replace("-", "_")
    experiment = verifier.STATIC_EXPERIMENTS[arm]["experiment"]
    model_choice = verifier.STATIC_EXPERIMENTS[arm]["model_choice"]
    intervention = arm.split("-", 1)[1].lower()
    if arm == "CI32-FULL":
        defaults = (
            "defaults:",
            "  - override /data: falcon_h1_carrierid_date_lodo_ci",
            f"  - override /model: {model_choice}",
            "  - override /callbacks: h1_carrierid_date_lodo_phase2_terminal",
        )
        base = (
            "seed: 42", "train: true", "test: false", "ckpt_path: null", "logger: false",
            "phase_ci:", "  outer_date: __REQUIRED_EXPLICIT_OUTER_DATE__",
            f"  arm: {arm}", f"  carrier_intervention: {intervention}",
            "  ci_preflight_path: __REQUIRED_IMMUTABLE_CI_SOURCE_PREFLIGHT_PATH__",
            "  five_date_aggregate_path: __REQUIRED_IMMUTABLE_FIVE_DATE_AGGREGATE_PATH__",
            "  warm_start_forbidden: true",
            "trainer:", "  accelerator: gpu", "  devices: 1", "  max_epochs: 50", "  min_epochs: 50", "  precision: 32-true",
        )
    else:
        parent = "h1_carrierid_date_lodo_ci32_full" if arm == "CI64-FULL" else "h1_carrierid_date_lodo_ci64_full"
        defaults = ("defaults:", f"  - {parent}", f"  - override /model: {model_choice}")
        base = ("phase_ci:", f"  arm: {arm}", f"  carrier_intervention: {intervention}")
    return "\n".join((*defaults,
                      f"protocol_id: h1_carrierid_date_lodo_{token}_${{phase_ci.outer_date}}_source_only_v1",
                      f"task_name: h1_carrierid_date_lodo_{token}_${{phase_ci.outer_date}}", *base)) + "\n"


def _hydra_sidecars(*, arm: str, date: str, run_dir: Path, phase1: str, preflight: str, aggregate: str) -> tuple[str, str]:
    task = [
        f"experiment={verifier.STATIC_EXPERIMENTS[arm]['experiment']}",
        f"phase_ci.outer_date={date}",
        f"phase_ci.phase1_preflight_path={phase1}",
        f"phase_ci.ci_preflight_path={preflight}",
        f"phase_ci.five_date_aggregate_path={aggregate}",
        "seed=42", "ckpt_path=null", "train=true", "test=false",
    ]
    hydra = {
        "hydra": {
            "run": {"dir": str(run_dir)},
            "runtime": {"output_dir": str(run_dir), "choices": {
                "experiment": verifier.STATIC_EXPERIMENTS[arm]["experiment"],
                "data": "falcon_h1_carrierid_date_lodo_ci",
                "model": verifier.STATIC_EXPERIMENTS[arm]["model_choice"],
                "callbacks": "h1_carrierid_date_lodo_phase2_terminal",
                "trainer": "gpu",
            }},
            "overrides": {"task": task},
        },
    }
    return _yaml(hydra), _yaml(task)


def _yaml(value: object) -> str:
    from omegaconf import OmegaConf
    return OmegaConf.to_yaml(OmegaConf.create(value), resolve=False)


def _fixture(tmp_path: Path, *, width_delta: float = 0.04, c0_delta: float = 0.04,
             ls_delta: float = 0.04, rs_delta: float = 0.04,
             static_equals_resolved: bool = False, static_arm_swap: bool = False,
             resolved_mutation: tuple[str, str] | None = None,
             nonadjacent_resolved_config: bool = False, noncanonical_checkpoint: bool = False) -> dict[str, object]:
    """Build a remote-path-mapped 25-cell provenance bundle without NWB."""

    aggregate = _write(tmp_path / "upstream_aggregate.json", {"schema": "synthetic"}, immutable=True)
    phase1 = _write(tmp_path / "phase1.json", {"schema": "synthetic-phase1"})
    evaluation_dir = tmp_path / "terminal_evaluations"
    for date in verifier.DATES:
        source_manifest = _write(tmp_path / "source" / date / "manifest.json", {"date": date})
        binding = {
            "outer_date": date, "seed": 42, "epochs": 50, "target_recordings_opened": 0,
            "target_bytes_read": 0, "warm_start_forbidden": True,
            "source_manifest_path": _remote(source_manifest, tmp_path), "source_manifest_sha256": _sha_bytes(source_manifest.read_bytes()),
            "preflight_path": _remote(phase1, tmp_path), "preflight_sha256": _sha_bytes(phase1.read_bytes()),
            "batch_order_sha256": _sha_text(f"batch-{date}"), "calibration_schedule_sha256": _sha_text(f"schedule-{date}"),
            "normalizer_sha256": _sha_text(f"normalizer-{date}"), "source_cache_sha256": _sha_text(f"source-{date}"),
            "normalized_cache_sha256": _sha_text(f"normalized-{date}"), "source_window_indices_sha256": _sha_text(f"window-{date}"),
        }
        preflight_path = tmp_path / "preflights" / f"{date}.json"
        configurations: dict[str, object] = {}
        fresh: dict[str, object] = {}
        shared = _sha_text(f"shared-{date}")
        ci64_state = _sha_text(f"ci64-state-{date}")
        static_configs: dict[str, Path] = {}
        for arm in verifier.ARMS:
            static = _write_text(tmp_path / "configs" / "experiment" / verifier.STATIC_EXPERIMENTS[arm]["filename"],
                                 _static_config(arm=arm))
            static_configs[arm] = static
            width, intervention, zero = verifier._expected_arm(arm)
            fresh[arm] = {"arm": arm, "fresh_seed": 42, "interface_dim": width,
                          "carrier_intervention": intervention, "zero_carrier_at_model_boundary": zero,
                          "initial_state_sha256": ci64_state if arm.startswith("CI64-") else _sha_text(f"ci32-{date}"),
                          "shared_backbone_initial_state_sha256": shared}
        for arm in verifier.ARMS:
            static = static_configs["CI64-RS" if static_arm_swap and arm == "CI64-FULL" else arm]
            if static_equals_resolved:
                legacy_resolved = tmp_path / "runs" / date / arm / ".hydra" / "config.yaml"
                _write_text(legacy_resolved, _resolved_config(
                    arm=arm, date=date, phase1=_remote(phase1, tmp_path),
                    preflight=_remote(preflight_path, tmp_path), aggregate=_remote(aggregate, tmp_path),
                ))
                configurations[arm] = {"path": _remote(legacy_resolved, tmp_path),
                                       "sha256": _sha_bytes(legacy_resolved.read_bytes())}
            else:
                configurations[arm] = {"path": _remote(static, tmp_path), "sha256": _sha_bytes(static.read_bytes())}
        preflight = {
            "schema": verifier.PREFLIGHT_SCHEMA, "status": verifier.PREFLIGHT_STATUS, "outer_date": date,
            "scope": {"target_recordings_opened": 0, "target_bytes_read": 0, "cuda_constructed_or_launched": False},
            "source_controls": {"all_arms": list(verifier.ARMS), "same_source_windows": True,
                                "same_source_schedule": True, "same_source_normalizer": True, "same_fresh_seed": 42,
                                "fixed_terminal_epoch_zero_based": 49, "epochs": 50},
            "fresh_models": fresh, "configuration": configurations, "source_binding": binding,
            "code_sha256": {key: _sha_bytes((verifier.ROOT / rel).read_bytes()) for key, rel in verifier.PREFLIGHT_CODE_FILES.items()},
        }
        _write(preflight_path, preflight, immutable=True)
        preflight_sha = _sha_bytes(preflight_path.read_bytes())
        checkpoints: dict[str, object] = {}
        for arm in verifier.ARMS:
            run_dir = tmp_path / "runs" / date / arm
            config = run_dir / ".hydra" / "config.yaml"
            config_text = _resolved_config(arm=arm, date=date, phase1=_remote(phase1, tmp_path),
                                           preflight=_remote(preflight_path, tmp_path), aggregate=_remote(aggregate, tmp_path))
            if resolved_mutation is not None and arm == "CI64-FULL" and date == verifier.DATES[0]:
                config_text = config_text.replace(*resolved_mutation)
            _write_text(config, config_text)
            phase1_remote = _remote(phase1, tmp_path)
            hydra_yaml, overrides_yaml = _hydra_sidecars(
                arm=arm, date=date, run_dir=run_dir, phase1=phase1_remote,
                preflight=_remote(preflight_path, tmp_path), aggregate=_remote(aggregate, tmp_path),
            )
            _write_text(run_dir / ".hydra" / "hydra.yaml", hydra_yaml)
            _write_text(run_dir / ".hydra" / "overrides.yaml", overrides_yaml)
            checker_config = (run_dir / "alternate" / "config.yaml") if nonadjacent_resolved_config and arm == "CI64-FULL" and date == verifier.DATES[0] else config
            if checker_config != config:
                _write_text(checker_config, config.read_text(encoding="utf-8"))
            checkpoint = (run_dir / "checkpoints" / "wrong_epoch" / "epoch_049.ckpt") if noncanonical_checkpoint and arm == "CI64-FULL" and date == verifier.DATES[0] else (run_dir / "checkpoints" / "fixed_epoch50" / "epoch_049.ckpt")
            meta = {
                "schema": verifier.CHECKPOINT_SCHEMA, "arm": arm, "outer_date": date, "fresh_seed": 42,
                "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
                "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
                "initial_state_sha256": fresh[arm]["initial_state_sha256"],
                "component_initial_state_sha256": fresh[arm]["initial_state_sha256"],
                "shared_backbone_initial_state_sha256": fresh[arm]["shared_backbone_initial_state_sha256"],
                "ci_source_binding_sha256": _sha_text(f"ci-binding-{date}-{arm}"),
                "phase2_base_source_binding_sha256": verifier._canonical_sha(binding),
                "phase1_source_manifest_sha256": binding["source_manifest_sha256"],
                "phase1_preflight_sha256": binding["preflight_sha256"], "ci_preflight_sha256": preflight_sha,
                "five_date_aggregate_sha256": _sha_bytes(aggregate.read_bytes()), "config_sha256": _sha_bytes(checker_config.read_bytes()),
                "target_optimizer_steps": 0, "target_backward_steps": 0, "checkpoint_warm_start": False,
            }
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {"weight": torch.ones(1)}, "epoch": 49, "global_step": 1,
                        "h1_carrierid_date_lodo_ci": meta}, checkpoint)
            checkpoints[arm] = {"checkpoint_path": _remote(checkpoint, tmp_path), "checkpoint_sha256": _sha_bytes(checkpoint.read_bytes()),
                                "config_path": _remote(checker_config, tmp_path), "config_sha256": _sha_bytes(checker_config.read_bytes()),
                                "metadata": meta}
        checker_path = tmp_path / "checks" / f"{date}.json"
        checker = {
            "schema": verifier.CHECKER_SCHEMA, "status": verifier.CHECKER_STATUS, "outer_date": date,
            "ci_preflight": {"path": _remote(preflight_path, tmp_path), "sha256": preflight_sha},
            "five_date_aggregate": {"path": _remote(aggregate, tmp_path), "sha256": _sha_bytes(aggregate.read_bytes())},
            "checkpoints": checkpoints,
            "code_sha256": {relative: _sha_bytes((verifier.ROOT / relative).read_bytes()) for relative in verifier.CHECKER_CLOSURE_FILES},
        }
        _write(checker_path, checker, immutable=True)
        query = _sha_text(f"query-{date}")
        sessions = [f"session-{date}-a", f"session-{date}-b"]
        target_files = {session: _sha_text(f"file-{session}") for session in sessions}
        pooled = {"CI32-FULL": 0.40, "CI64-FULL": 0.40 + width_delta,
                  "CI64-C0": 0.40 + width_delta - c0_delta,
                  "CI64-LS": 0.40 + width_delta - ls_delta,
                  "CI64-RS": 0.40 + width_delta - rs_delta}
        metrics = {
            arm: {"pooled_r2": score, "per_session": {session: {"r2": score} for session in sessions},
                  "query_window_indices_sha256": query, "state_immutable": True,
                  "state_sha256_before": _sha_text(f"state-{date}-{arm}"),
                  "state_sha256_after": _sha_text(f"state-{date}-{arm}")}
            for arm, score in pooled.items()
        }
        evaluation_path = evaluation_dir / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json"
        evaluation = {
            "schema": verifier.EVALUATION_SCHEMA,
            "status": f"PASS_H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_EVALUATED", "outer_date": date,
            "terminal_checker": {"path": _remote(checker_path, tmp_path), "sha256": _sha_bytes(checker_path.read_bytes())},
            "checkpoints": {arm: {"path": row["checkpoint_path"], "sha256": row["checkpoint_sha256"],
                                   "config_path": row["config_path"], "config_sha256": row["config_sha256"]}
                            for arm, row in checkpoints.items()},
            "target": {"sessions": sessions, "files": target_files, "shared_query_window_indices_sha256": query},
            "metrics": metrics, "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
            "one_shot": {"canonical_output_path": _remote(evaluation_path, tmp_path), "same_date_prior_terminal_evaluation_receipts": 0},
            "scope": {"formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False},
        }
        _write(evaluation_path, evaluation, immutable=True)
    return {"root": tmp_path, "evaluation_dir": evaluation_dir, "output": tmp_path / "terminal_verifier.json",
            "path_maps": verifier.parse_path_maps([f"/remote-ci={tmp_path}"])}


def _run(bundle: dict[str, object]) -> dict[str, object]:
    return verifier.verify(evaluation_dir=bundle["evaluation_dir"], output=bundle["output"], path_maps=bundle["path_maps"])


def test_full_grid_revalidates_chain_with_remote_path_map_and_immutable_output(tmp_path: Path):
    bundle = _fixture(tmp_path)
    result = _run(bundle)
    payload = json.loads(Path(bundle["output"]).read_text(encoding="utf-8"))
    assert result["decision"] == "H1_H64_SLODO_ELIGIBLE_NOT_AUTHORIZED"
    assert payload["exact_grid_cells"] == 25 and payload["inference_unit"] == "outer_date"
    assert payload["historical_mechanism_gate"]["passed"] is True
    assert payload["paper_practical_width_gate"]["passed"] is True
    assert stat.S_IMODE(Path(bundle["output"]).stat().st_mode) == 0o444
    assert payload["scope"]["nwb_opened"] == 0 and payload["scope"]["cuda_constructed"] is False
    evidence = payload["per_date"][verifier.DATES[0]]["configuration_evidence"]["CI64-FULL"]
    assert evidence["static_experiment_config"]["historical_receipt_binding"] is True
    assert evidence["static_experiment_config"]["path"] != evidence["resolved_checkpoint_adjacent_config"]["path"]
    assert evidence["resolved_checkpoint_adjacent_config"]["checkpoint_adjacent"] is True
    assert evidence["hydra_sidecars"]["historical_receipt_binding"] is False
    assert "stored" in evidence["hydra_sidecars"]["historical_receipt_binding_note"]


@pytest.mark.parametrize("kind,match", [
    ("legacy_same_file", "static experiment config canonical suffix/arm drift"),
    ("arm_swap", "static experiment config canonical suffix/arm drift"),
    ("byte_drift", "static experiment config byte SHA drift"),
])
def test_static_experiment_config_is_arm_specific_historically_bound_and_never_resolved_config(
        tmp_path: Path, kind: str, match: str):
    bundle = _fixture(
        tmp_path,
        static_equals_resolved=kind == "legacy_same_file",
        static_arm_swap=kind == "arm_swap",
    )
    if kind == "byte_drift":
        static = tmp_path / "configs" / "experiment" / verifier.STATIC_EXPERIMENTS["CI64-FULL"]["filename"]
        static.write_text(static.read_text(encoding="utf-8") + "# byte drift\n", encoding="utf-8")
    with pytest.raises(verifier.CiTerminalVerifierError, match=match):
        _run(bundle)
    assert not Path(bundle["output"]).exists()


@pytest.mark.parametrize("mutation,match", [
    (("train: true", "train: false"), "resolved config violates"),
    (("accelerator: gpu", "accelerator: cpu"), "resolved config violates"),
    (("devices: 1", "devices: 2"), "resolved config violates"),
    (("precision: 32-true", "precision: 16-mixed"), "resolved config violates"),
    (("warm_start_forbidden: true", "warm_start_forbidden: false"), "resolved config violates"),
])
def test_resolved_config_semantics_fail_after_full_synthetic_rehash(tmp_path: Path, mutation: tuple[str, str], match: str):
    # The invalid resolved bytes, checkpoint metadata hash, checker hash and
    # terminal-evaluation hash are all created consistently.  This proves the
    # verifier is not merely relying on a stale digest.
    bundle = _fixture(tmp_path, resolved_mutation=mutation)
    with pytest.raises(verifier.CiTerminalVerifierError, match=match):
        _run(bundle)
    assert not Path(bundle["output"]).exists()


@pytest.mark.parametrize("fixture_kw,match", [
    ({"nonadjacent_resolved_config": True}, "not checkpoint-adjacent"),
    ({"noncanonical_checkpoint": True}, "not exact checkpoints/fixed_epoch50/epoch_049.ckpt"),
])
def test_checkpoint_and_resolved_config_must_have_exact_adjacent_terminal_paths(
        tmp_path: Path, fixture_kw: dict[str, bool], match: str):
    bundle = _fixture(tmp_path, **fixture_kw)
    with pytest.raises(verifier.CiTerminalVerifierError, match=match):
        _run(bundle)
    assert not Path(bundle["output"]).exists()


def _mutate_sidecar_task(path: Path, mutate) -> None:
    from omegaconf import OmegaConf
    payload = OmegaConf.load(path)
    if path.name == "hydra.yaml":
        task = list(payload.hydra.overrides.task)
        mutate(task)
        payload.hydra.overrides.task = task
    else:
        task = list(payload)
        mutate(task)
        payload = OmegaConf.create(task)
    OmegaConf.save(payload, path, resolve=False)


@pytest.mark.parametrize("sidecar,mutate", [
    ("hydra.yaml", lambda rows: rows.pop()),
    ("hydra.yaml", lambda rows: rows.append("unexpected=true")),
    ("hydra.yaml", lambda rows: rows.__setitem__(0, "experiment=h1_carrierid_date_lodo_ci64_rs")),
    ("overrides.yaml", lambda rows: rows.pop()),
    ("overrides.yaml", lambda rows: rows.append("unexpected=true")),
    ("overrides.yaml", lambda rows: rows.__setitem__(0, "experiment=h1_carrierid_date_lodo_ci64_rs")),
])
def test_hydra_override_lineage_rejects_missing_extra_and_wrong_task_overrides(
        tmp_path: Path, sidecar: str, mutate):
    bundle = _fixture(tmp_path)
    date, arm = verifier.DATES[0], "CI64-FULL"
    path = tmp_path / "runs" / date / arm / ".hydra" / sidecar
    _mutate_sidecar_task(path, mutate)
    with pytest.raises(verifier.CiTerminalVerifierError, match="Hydra .*overrides.*missing/extra/wrong"):
        _run(bundle)
    assert not Path(bundle["output"]).exists()


@pytest.mark.parametrize("key,value", [
    ("experiment", "h1_carrierid_date_lodo_ci64_rs"),
    ("model", "falcon_h1_carrierid_date_lodo_ci64_rs"),
])
def test_hydra_choice_lineage_rejects_wrong_choice(tmp_path: Path, key: str, value: str):
    from omegaconf import OmegaConf
    bundle = _fixture(tmp_path)
    path = tmp_path / "runs" / verifier.DATES[0] / "CI64-FULL" / ".hydra" / "hydra.yaml"
    payload = OmegaConf.load(path)
    payload.hydra.runtime.choices[key] = value
    OmegaConf.save(payload, path, resolve=False)
    with pytest.raises(verifier.CiTerminalVerifierError, match="Hydra choice lineage drift"):
        _run(bundle)
    assert not Path(bundle["output"]).exists()


@pytest.mark.parametrize("mutation,match", [
    ("missing_date", "terminal evaluation must be a regular file"),
    ("missing_arm", "target/five-arm metric grid"),
    ("query", "query or state identity"),
    ("sha", "config SHA drift"),
])
def test_grid_and_identity_drift_fail_closed_without_output(tmp_path: Path, mutation: str, match: str):
    bundle = _fixture(tmp_path)
    date = verifier.DATES[0]
    receipt = Path(bundle["evaluation_dir"]) / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json"
    if mutation == "missing_date":
        receipt.unlink()
    elif mutation == "sha":
        config = tmp_path / "runs" / date / "CI64-FULL" / ".hydra" / "config.yaml"
        config.write_text(config.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    else:
        _chmod_mutable(receipt)
        body = json.loads(receipt.read_text(encoding="utf-8"))
        if mutation == "missing_arm":
            body["metrics"].pop("CI64-RS")
        else:
            body["metrics"]["CI64-LS"]["query_window_indices_sha256"] = _sha_text("wrong-query")
        _write(receipt, body, immutable=True)
    with pytest.raises(verifier.CiTerminalVerifierError, match=match):
        _run(bundle)
    assert not Path(bundle["output"]).exists()


@pytest.mark.parametrize("field,value,match", [
    ("epoch", 48, "not e49"),
    ("fresh_seed", 43, "metadata drift at fresh_seed"),
    ("selected_by", "validation_best", "metadata drift at selected_by"),
    ("checkpoint_warm_start", True, "metadata drift at checkpoint_warm_start"),
    ("target_backward_steps", 1, "metadata drift at target_backward_steps"),
])
def test_checkpoint_epoch_seed_selection_warm_start_and_target_bp_fail_closed(tmp_path: Path, field: str, value: object, match: str):
    bundle = _fixture(tmp_path)
    date, arm = verifier.DATES[0], "CI64-FULL"
    checkpoint = tmp_path / "runs" / date / arm / "checkpoints" / "fixed_epoch50" / "epoch_049.ckpt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if field == "epoch":
        payload["epoch"] = value
    else:
        payload["h1_carrierid_date_lodo_ci"][field] = value
    torch.save(payload, checkpoint)
    # Retie the direct checkpoint SHA fields so the test exercises semantic
    # verification rather than failing at the first superficial digest.
    checker = tmp_path / "checks" / f"{date}.json"; _chmod_mutable(checker)
    checker_body = json.loads(checker.read_text(encoding="utf-8"))
    checker_body["checkpoints"][arm]["checkpoint_sha256"] = _sha_bytes(checkpoint.read_bytes())
    _write(checker, checker_body, immutable=True)
    receipt = Path(bundle["evaluation_dir"]) / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json"; _chmod_mutable(receipt)
    receipt_body = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_body["terminal_checker"]["sha256"] = _sha_bytes(checker.read_bytes())
    receipt_body["checkpoints"][arm]["sha256"] = _sha_bytes(checkpoint.read_bytes())
    _write(receipt, receipt_body, immutable=True)
    with pytest.raises(verifier.CiTerminalVerifierError, match=match):
        _run(bundle)
    assert not Path(bundle["output"]).exists()


def test_historical_and_practical_gates_are_forced_separately(tmp_path: Path):
    small = _fixture(tmp_path / "small", width_delta=0.01)
    _run(small)
    body = json.loads(Path(small["output"]).read_text(encoding="utf-8"))
    assert body["historical_mechanism_gate"]["passed"] is True
    assert body["paper_practical_width_gate"]["passed"] is False
    assert body["h64_escalation_gate"]["decision"] == "STOP_PRACTICAL_WIDTH_THRESHOLD_FAILED"

    boundary = _fixture(tmp_path / "boundary", width_delta=0.03)
    _run(boundary)
    body = json.loads(Path(boundary["output"]).read_text(encoding="utf-8"))
    assert body["paper_practical_width_gate"]["passed"] is True
    assert body["h64_escalation_gate"]["decision"] == "H1_H64_SLODO_ELIGIBLE_NOT_AUTHORIZED"


def test_practical_pass_cannot_bypass_c0_or_ls_mechanism_gate(tmp_path: Path):
    bundle = _fixture(tmp_path, width_delta=0.04, c0_delta=-0.01, ls_delta=0.04)
    _run(bundle)
    body = json.loads(Path(bundle["output"]).read_text(encoding="utf-8"))
    assert body["paper_practical_width_gate"]["passed"] is True
    assert body["historical_mechanism_gate"]["passed"] is False
    assert body["h64_escalation_gate"]["decision"] == "STOP_HISTORICAL_MECHANISM_GATE_FAILED"


def test_rs_changes_statistics_but_never_the_gate(tmp_path: Path):
    baseline = _fixture(tmp_path / "baseline", rs_delta=0.04)
    _run(baseline)
    changed = _fixture(tmp_path / "changed", rs_delta=-0.50)
    _run(changed)
    first = json.loads(Path(baseline["output"]).read_text(encoding="utf-8"))
    second = json.loads(Path(changed["output"]).read_text(encoding="utf-8"))
    assert first["h64_escalation_gate"] == second["h64_escalation_gate"]
    assert first["paired_deltas"]["ci64_full_minus_ci64_rs"] != second["paired_deltas"]["ci64_full_minus_ci64_rs"]
    assert second["rs_control"]["hard_gate"] is False


def test_schema_only_and_default_cli_are_read_only_and_data_free(monkeypatch, tmp_path: Path):
    before = sorted(tmp_path.iterdir())
    payload = verifier.preflight_schema_only()
    assert payload["exact_grid_cells"] == 25
    assert payload["scope"] == {"torch_imported": False, "nwb_opened": 0, "target_data_opened": 0,
                                "cuda_constructed": False, "writes": 0}
    assert sorted(tmp_path.iterdir()) == before
    monkeypatch.setattr(sys, "argv", ["h1_carrierid_date_lodo_ci_fivedate_terminal_verifier_v2.py"])
    with pytest.raises(SystemExit, match="refusing terminal artifact access"):
        verifier.main()
