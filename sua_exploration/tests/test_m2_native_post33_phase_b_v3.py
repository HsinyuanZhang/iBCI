"""Focused, score-free tests for native-M2 post-33 Phase-B v3."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
import yaml

from sua_exploration.mc_maze import m2_native_post33_phase_b_v3 as contract


ROOT = Path(__file__).resolve().parents[2]
SPINT_ROOT = ROOT / "SPINT-main"
STREAMING_ROOT = ROOT / "streaming_calibration_exp"


def _spint_metric_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SPINT_ROOT))
    return importlib.import_module("src.models.falcon_post33_confirm_v3_module")


def _records(tmp_path: Path, key: contract.CellKey) -> list[dict[str, object]]:
    records = []
    for epoch in range(35):
        score = 0.2 + epoch / 1000
        if epoch in {2, 4}:
            score = 0.9
        records.append(
            {
                "epoch": epoch,
                "metric_name": "val_source/r2_equal_session_mean",
                "metric_value": score,
                "metric_scope": "exact_six_outer_train_source_sessions_only",
                "source_sessions": list(key.source_sessions),
                "source_totals": {session: 11 + index for index, session in enumerate(key.source_sessions)},
                "outer_session": key.outer_session,
                "outer_total": 0,
                "checkpoint_path": str((tmp_path / f"epoch_{epoch:03d}.ckpt").resolve()),
            }
        )
    return records


def test_source_metric_accepts_exact_six_and_equal_session_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _spint_metric_module(monkeypatch)
    sources, outer = module.fold_sessions(0)
    values = {session: float(index) / 10 for index, session in enumerate(sources)}
    totals = {session: 3 + index for index, session in enumerate(sources)}
    result = module.aggregate_exact_source_metrics(
        expected_sources=sources,
        outer_session=outer,
        values=values,
        totals=totals,
        outer_total=0,
    )
    assert result == pytest.approx(sum(values.values()) / 6)


@pytest.mark.parametrize("defect", ["outer_sample", "missing", "extra", "nonfinite", "low_total"])
def test_source_metric_synthetic_scope_defects_fail(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    module = _spint_metric_module(monkeypatch)
    sources, outer = module.fold_sessions(1)
    values = {session: 0.1 for session in sources}
    totals = {session: 10 for session in sources}
    outer_total = 0
    if defect == "outer_sample":
        outer_total = 1
    elif defect == "missing":
        values.pop(sources[0])
    elif defect == "extra":
        values["ses-extra"] = 0.2
        totals["ses-extra"] = 10
    elif defect == "nonfinite":
        values[sources[0]] = float("nan")
    elif defect == "low_total":
        totals[sources[0]] = 2
    with pytest.raises(ValueError):
        module.aggregate_exact_source_metrics(
            expected_sources=sources,
            outer_session=outer,
            values=values,
            totals=totals,
            outer_total=outer_total,
        )


def test_outer_test_has_one_real_session_and_no_six_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _spint_metric_module(monkeypatch)
    sources, outer = module.fold_sessions(2)
    assert module.validate_exact_outer_test(
        outer_session=outer, values={outer: 0.3}, totals={outer: 100}
    ) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        module.validate_exact_outer_test(
            outer_session=outer,
            values={outer: 0.3, sources[0]: float("-inf")},
            totals={outer: 100, sources[0]: 0},
        )


def test_explicit_selector_covers_0_to_34_and_tie_chooses_earlier(tmp_path: Path) -> None:
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
    selected = contract.select_source_checkpoint(_records(tmp_path, key), key)
    assert selected["epoch"] == 2
    incomplete = _records(tmp_path, key)[:-1]
    with pytest.raises(ValueError, match="0..34"):
        contract.select_source_checkpoint(incomplete, key)


def test_same_cell_concurrent_claim_has_exactly_one_winner(tmp_path: Path) -> None:
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 3, 43)

    def attempt(index: int) -> bool:
        try:
            contract.claim_cell_ownership(tmp_path, key, owner_token=f"owner-{index}")
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 7


def test_different_cells_have_noncolliding_deterministic_paths(tmp_path: Path) -> None:
    keys = [
        contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42),
        contract.CellKey(contract.PROTOCOL_ID, "spint", 1, 42),
        contract.CellKey(contract.PROTOCOL_ID, "t4", 0, 42),
        contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 43),
    ]
    paths = [contract.claim_cell_ownership(tmp_path, key, owner_token=f"owner-{i}") for i, key in enumerate(keys)]
    assert len({row["cell_dir"] for row in paths}) == len(keys)
    assert all("M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1/PHASE_B_V3" in str(row["cell_dir"]) for row in paths)


def test_status_is_write_once_and_requires_cell_owner(tmp_path: Path) -> None:
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 0, 42)
    contract.claim_cell_ownership(tmp_path, key, owner_token="owner")
    contract.write_status_exclusive(tmp_path, key, state="started", owner_token="owner")
    with pytest.raises(FileExistsError):
        contract.write_status_exclusive(tmp_path, key, state="started", owner_token="owner")
    with pytest.raises(PermissionError):
        contract.write_status_exclusive(tmp_path, key, state="completed", owner_token="wrong")


def test_completion_receipt_is_paired_hash_bound_and_write_once(tmp_path: Path) -> None:
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 4, 44)
    records = _records(tmp_path, key)
    selected_path = Path(records[2]["checkpoint_path"])
    selected_path.write_bytes(b"paired-spint-checkpoint")
    config_path = (tmp_path / "resolved_config.yaml").resolve()
    config_path.write_text("protocol_id: fixture\n", encoding="utf-8")
    receipt = contract.build_spint_completion_receipt(
        key=key, selector_records=records, resolved_config_path=config_path
    )
    receipt_path = tmp_path / "receipt.json"
    contract.write_json_exclusive(receipt_path, receipt)
    assert contract.resolve_paired_teacher_from_receipt(
        receipt_path, fold=4, seed=44
    ) == selected_path
    with pytest.raises(FileExistsError):
        contract.write_json_exclusive(receipt_path, receipt)
    with pytest.raises(ValueError, match="cell mismatch"):
        contract.resolve_paired_teacher_from_receipt(receipt_path, fold=3, seed=44)
    selected_path.write_bytes(b"mutated-paired-checkpoint")
    with pytest.raises(ValueError, match="bytes differ"):
        contract.resolve_paired_teacher_from_receipt(receipt_path, fold=4, seed=44)


def test_streaming_local_resolver_accepts_only_the_matching_pair(tmp_path: Path) -> None:
    key = contract.CellKey(contract.PROTOCOL_ID, "spint", 6, 43)
    records = _records(tmp_path, key)
    selected_path = Path(records[2]["checkpoint_path"])
    selected_path.write_bytes(b"local-resolver-checkpoint")
    config_path = (tmp_path / "resolved_config.yaml").resolve()
    config_path.write_text("protocol_id: fixture\n", encoding="utf-8")
    receipt_path = (tmp_path / "receipt.json").resolve()
    contract.write_json_exclusive(
        receipt_path,
        contract.build_spint_completion_receipt(
            key=key, selector_records=records, resolved_config_path=config_path
        ),
    )
    code = f"""
from src.utils.post33_paired_teacher_v3 import resolve_paired_spint_teacher
path = resolve_paired_spint_teacher({str(receipt_path)!r}, loso_fold=6, seed=43)
assert str(path) == {str(selected_path)!r}
try:
    resolve_paired_spint_teacher({str(receipt_path)!r}, loso_fold=5, seed=43)
except ValueError:
    pass
else:
    raise AssertionError('mismatched fold was accepted')
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=STREAMING_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


class _Decoder31(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.values = nn.ParameterList(
            [nn.Parameter(torch.tensor([float(index)], dtype=torch.float32), requires_grad=False) for index in range(31)]
        )


def test_decoder_31_of_31_lifecycle_is_exact_and_optimizer_disjoint() -> None:
    decoder = _Decoder31()
    snapshot = contract.decoder_snapshot(decoder)
    snapshots = {stage: copy.deepcopy(snapshot) for stage in ("pretrain", "posttrain", "reload", "prequery")}
    outside = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([outside], lr=0.1)
    count, names = contract.optimizer_decoder_intersection(optimizer, decoder)
    assert (count, names) == (0, [])
    evidence = contract.validate_decoder_lifecycle(snapshots, optimizer_intersection_count=count)
    assert evidence["tensor_count_compared"] == 31
    assert evidence["decoder_updated_tensor_count"] == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("name", "wrong.name"),
        ("shape", [999]),
        ("dtype", "torch.float64"),
        ("num_bytes", 999),
        ("sha256", "0" * 64),
    ],
)
def test_decoder_lifecycle_negative_tensor_closure_mutations(field: str, replacement: object) -> None:
    decoder = _Decoder31()
    reference = contract.decoder_snapshot(decoder)
    mutated = copy.deepcopy(reference)
    mutated["tensors"][0][field] = replacement
    snapshots = {stage: copy.deepcopy(reference) for stage in ("pretrain", "posttrain", "reload", "prequery")}
    snapshots["posttrain"] = mutated
    with pytest.raises(ValueError, match="closure changed"):
        contract.validate_decoder_lifecycle(snapshots, optimizer_intersection_count=0)


def test_decoder_lifecycle_negative_requires_grad_and_optimizer_overlap() -> None:
    decoder = _Decoder31()
    reference = contract.decoder_snapshot(decoder)

    trainable = _Decoder31()
    trainable.values[0].requires_grad_(True)
    trainable_snapshot = contract.decoder_snapshot(trainable)
    with pytest.raises(ValueError, match="requires_grad"):
        contract.compare_decoder_snapshots(trainable_snapshot, trainable_snapshot)
    optimizer = torch.optim.SGD([trainable.values[0]], lr=0.1)
    count, names = contract.optimizer_decoder_intersection(optimizer, trainable)
    assert count == 1 and names == ["values.0"]
    with pytest.raises(ValueError, match="intersect"):
        contract.validate_decoder_lifecycle(
            {stage: copy.deepcopy(reference) for stage in ("pretrain", "posttrain", "reload", "prequery")},
            optimizer_intersection_count=count,
        )


def test_t4_v3_static_config_has_mandatory_receipt_and_no_teacher_fallback() -> None:
    model_path = STREAMING_ROOT / "configs/model/streaming_b3s_t4_post33_paired_v3.yaml"
    cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    assert cfg["paired_spint_completion_receipt"] == "???"
    assert "teacher_ckpt_path" not in cfg
    all_v3_configs = list((SPINT_ROOT / "configs").rglob("*v3.yaml")) + list(
        (STREAMING_ROOT / "configs").rglob("*v3.yaml")
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in all_v3_configs)
    assert "epoch_034.ckpt" not in text


def test_spint_hydra_compose_binds_source_module_and_unique_paths(tmp_path: Path) -> None:
    cell = (tmp_path / "spint-cell").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(SPINT_ROOT / "configs")):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=m2_native_post33_confirm_v3_spint",
                "data.loso_fold=2",
                "seed=43",
                "cell_owner_token=owner",
                f"cell_paths.cell_dir={cell}",
                f"cell_paths.owner={cell / 'ownership.json'}",
                f"cell_paths.selector_records={cell / 'selector_records.json'}",
                f"cell_paths.checkpoints={cell / 'checkpoints'}",
            ],
        )
    assert cfg.model._target_.endswith("M2Post33SourceOnlyFalconLitModule")
    assert cfg.callbacks.source_selector_v3._target_.endswith("Post33SourceSelectorV3")
    assert cfg.model.scheduler_monitor == "val_source/r2_equal_session_mean"
    assert str(cfg.hydra.run.dir) == str(cell)
    assert str(cfg.callbacks.source_selector_v3.checkpoint_dir) == str(cell / "checkpoints")


def test_t4_hydra_compose_requires_paired_receipt_and_has_no_legacy_teacher(tmp_path: Path) -> None:
    cell = (tmp_path / "t4-cell").resolve()
    receipt = (tmp_path / "spint-receipt.json").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(STREAMING_ROOT / "configs")):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=m2_native_post33_confirm_v3_t4",
                "data.loso_fold=2",
                "seed=43",
                "cell_owner_token=owner",
                f"cell_paths.cell_dir={cell}",
                f"cell_paths.owner={cell / 'ownership.json'}",
                f"cell_paths.selector_records={cell / 'selector_records.json'}",
                f"cell_paths.checkpoints={cell / 'checkpoints'}",
                f"model.paired_spint_completion_receipt={receipt}",
            ],
        )
    assert cfg.model._target_.endswith("M2Post33PairedT4LitModuleV3")
    assert "teacher_ckpt_path" not in cfg.model
    assert str(cfg.model.paired_spint_completion_receipt) == str(receipt)
    assert str(cfg.hydra.run.dir) == str(cell)

    with initialize_config_dir(version_base="1.3", config_dir=str(STREAMING_ROOT / "configs")):
        missing = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=m2_native_post33_confirm_v3_t4",
                "data.loso_fold=2",
                "seed=43",
            ],
        )
    with pytest.raises(Exception):
        OmegaConf.to_container(missing.model, resolve=True, throw_on_missing=True)


def test_t4_v3_dedicated_setup_does_not_mutate_public_hparams_source() -> None:
    source = (STREAMING_ROOT / "src/data/falcon_post33_confirm_v3_datamodule.py").read_text(
        encoding="utf-8"
    )
    assert "self.hparams.validation_protocol =" not in source
    assert "self.hparams.loso_fold =" not in source
    assert "super().setup(" not in source
    assert "dedicated_no_generic_setup_delegation" in source
    assert "generic Falcon split resolution is forbidden" in source


def test_t4_v3_public_hparams_remain_loso_at_runtime_and_generic_consumer_fails() -> None:
    code = """
from src.data.falcon_post33_confirm_v3_datamodule import M2Post33ConfirmT4DataModuleV3
m = M2Post33ConfirmT4DataModuleV3(
    task='m2', data_dir='/tmp/not-opened-during-init',
    validation_protocol='loso', loso_fold=5,
)
assert m.hparams.validation_protocol == 'loso'
assert m.hparams.loso_fold == 5
assert m.hparams.heldin_query_start_trial == 33
assert m.hparams.query_start_trial == 0
assert dict(m.public_endpoint_contract) == {
    'validation_protocol': 'loso', 'loso_fold': 5,
    'heldin_query_start_trial': 33, 'query_start_trial': 0,
}
try:
    m._resolve_train_val_sessions([])
except RuntimeError as exc:
    assert 'generic Falcon split resolution is forbidden' in str(exc)
else:
    raise AssertionError('generic consumer did not fail closed')
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=STREAMING_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_phase_b_sources_import_no_scorer_evalai_or_formal_sua() -> None:
    paths = [
        ROOT / "sua_exploration/mc_maze/m2_native_post33_phase_b_v3.py",
        ROOT / "sua_exploration/scripts/preflight_m2_native_post33_phase_b_v3_cell.py",
        ROOT / "sua_exploration/scripts/finalize_m2_native_post33_spint_v3_receipt.py",
        SPINT_ROOT / "src/models/falcon_post33_confirm_v3_module.py",
        SPINT_ROOT / "src/callbacks/post33_source_selector_v3.py",
        STREAMING_ROOT / "src/data/falcon_post33_confirm_v3_datamodule.py",
        STREAMING_ROOT / "src/models/streaming_post33_paired_t4_v3_module.py",
        STREAMING_ROOT / "src/utils/post33_paired_teacher_v3.py",
    ]
    lowered = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    assert "import evalai" not in lowered
    assert "import scorer" not in lowered
    assert "dandi" not in lowered
    assert "import formal_sua" not in lowered
