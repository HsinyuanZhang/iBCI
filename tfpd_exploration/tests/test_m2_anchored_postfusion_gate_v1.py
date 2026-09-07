"""Focused no-data/no-CUDA tests for the APFG V1 Stage-0 contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import numpy as np

REPO = Path(__file__).resolve().parents[2]
STREAMING_ROOT = REPO / "streaming_calibration_exp"
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import plan
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.adapter import (
    AdapterError,
    AnchoredPostFusionGate,
    alpha_only_adam,
    exact_positive_zero,
    freeze_and_install,
    parameter_evidence,
)
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import driver as apfg_driver
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.driver import AdmissionError, issue_live_capability, static_admission
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.pools import (
    PoolError,
    causal_pool_state,
    group_coordinate_states,
    requested_pool_size,
    validate_canonical_m30_batch,
    validate_support_indices,
)
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.selection import (
    EpochValidation,
    SelectionError,
    lexical_source_split,
    select_earliest_best,
)
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.runtime import (
    FrozenIdentityCacheKey,
    RuntimeContractError,
    build_source_training_contract,
    establish_post_attempt_seed_evidence,
    task_only_scaled_last_bin_mse,
    validate_post_attempt_runtime_evidence,
)
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.training import TrainingError, expand_identity_for_windows, validate_cuda_launch_attestation
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import lifecycle
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.source_replay import (
    SourceReplayError,
    SourceCoordinate,
    SourceSessionMaterial,
    OrderedRawActivityPool,
    canonical_m30_source_batches,
    pool_for_controller_coordinate,
    source_authority_summary,
    _r2,
)
from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import laws


def _native() -> SideFeatureEarlyPoolEncoder:
    torch.manual_seed(7)
    return SideFeatureEarlyPoolEncoder(trial_length=100, window_size=50, hidden_dim=6, side_dim=4, num_post_layers=2)


class _Student(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_encoder = _native()
        self.decoder = torch.nn.Linear(50, 2)
        self._decoder_frozen = True


def _batch(members: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    return torch.randn(2, members, 100, 3), torch.randn(2, 3, 4)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _pool(*, session: str = "s", requested_size: int, support: tuple[int, ...], completed: tuple[int, ...],
          query_trial: int, endpoint: int):
    needed = plan.NON_SUPPORT_COMPLETIONS[requested_size]
    members = support + (completed[-needed:] if needed else ())
    return causal_pool_state(session_id=session, requested_size=requested_size, support_trial_ids=support,
                             completed_non_support_trial_ids=completed, query_trial_index=query_trial,
                             query_endpoint=endpoint, ordered_activity_sha256=tuple(_digest(item) for item in members),
                                 selected_support4_carrier_hz_sha256=_digest(999), normalized_side_sha256=_digest(998),
                             normalizer_sha256=_digest(997))


def test_zero_anchor_is_exact_in_eval_and_training_keeps_alpha_gradient() -> None:
    native = _native()
    adapter = AnchoredPostFusionGate(native)
    calibration, side = _batch()
    native.eval()
    adapter.eval()
    with torch.no_grad():
        expected = native.forward_batch(calibration, side_features=side)
        observed = adapter.forward_batch(calibration, side_features=side)
    assert torch.equal(expected, observed)

    adapter.eval()
    adapter.set_alpha_training(True)
    output = adapter.forward_batch(calibration, side_features=side)
    output.square().mean().backward()
    assert adapter.alpha.grad is not None
    assert float(adapter.alpha.grad.abs()) > 0.0


def test_freeze_and_install_allows_only_scalar_alpha_adam() -> None:
    student = _Student()
    adapter = freeze_and_install(student)
    evidence = parameter_evidence(adapter)
    optimizer = alpha_only_adam(adapter)
    assert evidence["trainable_parameter_names"] == ["alpha"]
    assert optimizer.param_groups[0]["params"] == [adapter.alpha]
    assert all(not parameter.requires_grad for parameter in adapter.native.parameters())
    with pytest.raises(AdapterError, match="optimizer"):
        adapter.native.pre_pool[0].weight.requires_grad = True
        alpha_only_adam(adapter)


def test_negative_zero_is_not_an_exact_positive_zero_anchor() -> None:
    adapter = AnchoredPostFusionGate(_native())
    with torch.no_grad():
        adapter.alpha.copy_(torch.tensor(-0.0, dtype=torch.float32))
    assert not exact_positive_zero(adapter.alpha)
    with torch.no_grad():
        adapter.alpha.copy_(torch.tensor(+0.0, dtype=torch.float32))
    assert exact_positive_zero(adapter.alpha)


def test_causal_pool_uses_recent_completed_non_support_without_future_or_padding() -> None:
    support = (2, 4, 8, 11)
    completed = tuple(range(30, 56))
    m4 = _pool(requested_size=4, support=support, completed=completed, query_trial=56, endpoint=800)
    m10 = _pool(requested_size=10, support=support, completed=completed, query_trial=56, endpoint=800)
    m30 = _pool(requested_size=30, support=support, completed=completed, query_trial=56, endpoint=800)
    assert m4.member_trial_ids == support
    assert m10.member_trial_ids == support + tuple(range(50, 56))
    assert m30.member_trial_ids == support + tuple(range(30, 56))
    with pytest.raises(PoolError, match="not yet"):
        _pool(requested_size=10, support=support, completed=tuple(range(30, 35)), query_trial=35, endpoint=1)
    with pytest.raises(PoolError, match="future"):
        _pool(requested_size=4, support=support, completed=(30, 57), query_trial=57, endpoint=1)
    with pytest.raises(PoolError, match="chronological"):
        _pool(requested_size=4, support=support, completed=(31, 30), query_trial=40, endpoint=1)
    with pytest.raises(PoolError, match="first thirty"):
        validate_support_indices((0, 1, 2, 30))


def test_batch_coordinates_group_only_identical_pool_states() -> None:
    first = _pool(session="a", requested_size=10, support=(1, 2, 3, 4), completed=tuple(range(30, 40)), query_trial=40, endpoint=1)
    same = _pool(session="a", requested_size=10, support=(1, 2, 3, 4), completed=tuple(range(30, 40)), query_trial=40, endpoint=1)
    later = _pool(session="a", requested_size=10, support=(1, 2, 3, 4), completed=tuple(range(30, 41)), query_trial=41, endpoint=2)
    cross_session = _pool(session="b", requested_size=10, support=(1, 2, 3, 4), completed=tuple(range(30, 40)), query_trial=40, endpoint=1)
    groups = group_coordinate_states({("a", 100): first, ("b", 100): same, ("a", 200): later, ("other", 100): cross_session})
    assert sorted(map(len, groups.values())) == [1, 1, 2]
    assert first.digest() != cross_session.digest()


def test_full_pool_receipt_binds_window_endpoint_and_side_normalizer_while_identity_groups_same_trial() -> None:
    first = _pool(session="s1", requested_size=30, support=(1, 2, 3, 4),
                  completed=tuple(range(30, 56)), query_trial=56, endpoint=500)
    later_endpoint = _pool(session="s1", requested_size=30, support=(1, 2, 3, 4),
                           completed=tuple(range(30, 56)), query_trial=56, endpoint=501)
    assert first.digest() != later_endpoint.digest()
    assert first.identity_digest() == later_endpoint.identity_digest()
    with pytest.raises(PoolError, match="normalized side"):
        causal_pool_state(session_id="s1", requested_size=4, support_trial_ids=(1, 2, 3, 4),
                          completed_non_support_trial_ids=(), query_trial_index=30, query_endpoint=500,
                          ordered_activity_sha256=tuple(_digest(item) for item in (1, 2, 3, 4)),
                          selected_support4_carrier_hz_sha256=_digest(1), normalized_side_sha256="bad", normalizer_sha256=_digest(2))


def test_m30_eligible_canonical_controller_cycles_each_law_four_times() -> None:
    state = _pool(requested_size=30, support=(1, 2, 3, 4), completed=tuple(range(30, 56)), query_trial=56, endpoint=100)
    sibling = _pool(requested_size=30, support=(1, 2, 3, 4), completed=tuple(range(30, 56)), query_trial=56, endpoint=101)
    assert len(validate_canonical_m30_batch((state, sibling))) == 2
    values = [requested_pool_size(epoch, 0) for epoch in range(1, 13)]
    assert {value: values.count(value) for value in plan.POOL_CYCLE} == {4: 4, 10: 4, 30: 4}
    with pytest.raises(PoolError, match="M30"):
        validate_canonical_m30_batch((_pool(requested_size=10, support=(1, 2, 3, 4), completed=tuple(range(30, 40)),
                                            query_trial=40, endpoint=10),))


def test_lexical_5_2_selection_and_refit_from_zero() -> None:
    split = lexical_source_split(("s7", "s3", "s1", "s5", "s2", "s4", "s6"))
    assert split == {"fit": ("s1", "s2", "s3", "s4", "s5"), "validation": ("s6", "s7")}
    rows = []
    for epoch in range(1, 13):
        score = 0.10 if epoch in (4, 5) else 0.05
        rows.append(EpochValidation(epoch, {"s6": score, "s7": score}, {"s6": 0.09, "s7": 0.09}))
    selected = select_earliest_best(rows, split["validation"])
    assert selected["selected_epoch"] == 4
    assert selected["source_safety_gate_passed"] is True
    assert selected["refit"]["alpha_initialization"] == "+0.0"
    with pytest.raises(SelectionError, match="every epoch"):
        select_earliest_best(rows[:-1], split["validation"])


def test_canonical_batches_are_semantic_not_digest_ordered() -> None:
    material = SourceSessionMaterial(
        session="s", support_indices=(1, 2, 3, 4), selected_support4_carrier_hz=np.zeros((30, 4), np.float64), raw_m30_hz_audit=np.zeros((30,4),np.float32),
        normalized_side=np.zeros((30, 4), np.float32), normalizer_sha256=_digest(91),
        activities=np.zeros((57, 100, 2), np.float32), query_rows=(),
        coordinates=(
            SourceCoordinate("s", 300, 56, _pool(session="s", requested_size=30, support=(1,2,3,4), completed=tuple(range(30,56)), query_trial=56, endpoint=349), _digest(1), _digest(2), _digest(3)),
            SourceCoordinate("s", 100, 56, _pool(session="s", requested_size=30, support=(1,2,3,4), completed=tuple(range(30,56)), query_trial=56, endpoint=149), _digest(1), _digest(2), _digest(3)),
        ),
    )
    batches = canonical_m30_source_batches(material)
    assert [[item.window_start for item in batch] for batch in batches] == [[100, 300]]


def test_deployment_laws_recompute_exact_zero_learned_gate() -> None:
    rows = []
    for surface, count in (("external_post30_local", 6), ("within_post30", 7)):
        for session_index in range(count):
            for law in laws.LAWS:
                for system, value in (("APFG-ZERO", 0.10), ("APFG-LEARNED", 0.12 if surface.startswith("external") else 0.101)):
                    rows.append({"system": system, "memory_law": law, "surface": surface,
                                 "session": f"{surface}-{session_index}", "budget": 4,
                                 "input_authority_key": f"{surface}|{session_index}", "r2": value,
                                 "prediction_sha256": _digest(session_index + (1 if system.endswith("ZERO") else 20)),
                                 "target_sha256": _digest(42), "query_starts_sha256": _digest(43),
                                 "window_count": 3, "commits": 2, "evictions": 0,
                                 "causal_trace_sha256": _digest(44), "model_state_before_sha256": _digest(45),
                                 "model_state_after_sha256": _digest(45), "parameter_updates": 0,
                                 "target_updates": 0, "activity_authority": plan.ACTIVITY_AUTHORITY})
    pooled = {f"{surface}|{session_index}": {"surface": surface, "session_id": f"{surface}-{session_index}", "r2": 0.10}
              for surface, count in (("external_post30_local", 6), ("within_post30", 7)) for session_index in range(count)}
    result = laws.recompute(rows, pooled)
    assert result["governing_gate"]["passed"] is True
    rows[0]["target_updates"] = 1
    with pytest.raises(laws.LawError, match="mutation"):
        laws.validate_rows(rows)
    rows[0]["target_updates"] = 0
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(laws.LawError, match="sequence"):
        laws.validate_rows(rows)


def test_attempt_first_prefix_preserving_lifecycle_success_and_failure(tmp_path: Path) -> None:
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True)
    terminal, failure = lifecycle.execute(
        tmp_path, attempt={"phase": "attempt"}, launch=lambda: {"phase": "launch"},
        bodies=lambda artifact: {"source_authority.json": artifact.publish_json("source_authority.json", {"ok": True})},
        terminal=lambda published: {"source_authority_sha256": published["source_authority.json"]},
        progress=lambda: {"stage": "success"},
    )
    assert terminal and failure is None
    root = tmp_path / plan.RESULT_ROOT_RELATIVE
    assert {item.name for item in root.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "terminal.json", "terminal.json.sha256"}
    failed_root = tmp_path / "retry"
    (failed_root / "tfpd_exploration/results").mkdir(parents=True)
    terminal, failure = lifecycle.execute(
        failed_root, attempt={"phase": "attempt"}, launch=lambda: {"phase": "launch"},
        bodies=lambda artifact: (_ for _ in ()).throw(RuntimeError("bounded synthetic failure")),
        terminal=lambda _published: {}, progress=lambda: {"stage": "bodies"},
    )
    assert terminal is None and failure
    names = {item.name for item in (failed_root / plan.RESULT_ROOT_RELATIVE).iterdir()}
    assert "failure.json" in names and "terminal.json" not in names and "attempt.json" in names


def test_static_authority_closure_and_inert_live_admission() -> None:
    authority = plan.validate_static(REPO)
    assert authority["design_sha256"] == plan.DESIGN_SHA256
    closure = plan.closure(REPO)
    assert plan.WORKORDER_RELATIVE in closure
    assert len(closure) == len(plan.CLOSURE_RELATIVES)
    admitted = static_admission(REPO)
    assert admitted["public_cli_can_mint"] is False
    assert admitted["root_only_capability_possible"] is True
    with pytest.raises(AdmissionError, match="token"):
        issue_live_capability(REPO, token=object())


def test_runtime_contract_freezes_inherited_path_and_prescribes_one_materialization() -> None:
    contract = build_source_training_contract(("s7", "s1", "s6", "s2", "s5", "s3", "s4"))
    assert contract.source_fit_sessions == ("s1", "s2", "s3", "s4", "s5")
    assert contract.source_validation_sessions == ("s6", "s7")
    evidence = {
        "attempt_published_before_checkpoint_data_cuda": True, "pit_materializations": 1,
        "strict_load_before_adapter": True, "inherited_trainable_parameter_names": [],
        "alpha_trainable_parameter_names": ["id_encoder.alpha"], "inherited_eval": True,
        "dropout_calls": 0, "optimizer_parameter_names": ["id_encoder.alpha"],
        "source_split": {"fit": list(contract.source_fit_sessions), "validation": list(contract.source_validation_sessions)},
        "selection_epochs": list(range(1, 13)), "refit_alpha_positive_zero": True,
        "training_loss": plan.TRAINING_LOSS, "behavior_scaling_factor": 5.0,
        "predict_scaled_behavior": True, "teacher_forward_calls": 0,
        "seed_evidence": {"seed": 42, "python_random_seeded": True, "numpy_random_seeded": True,
                          "torch_manual_seeded": True, "torch_cuda_manual_seed_all": True,
                          "cudnn_deterministic": True, "cudnn_benchmark": False,
                          "deterministic_algorithms_forced": False},
    }
    validate_post_attempt_runtime_evidence(evidence)
    evidence["optimizer_parameter_names"] = ["id_encoder.alpha", "id_encoder.native.pre_pool.0.weight"]
    with pytest.raises(RuntimeContractError, match="optimizer"):
        validate_post_attempt_runtime_evidence(evidence)
    evidence["optimizer_parameter_names"] = ["id_encoder.alpha"]
    evidence["seed_evidence"] = {**evidence["seed_evidence"], "torch_cuda_manual_seed_all": False}
    with pytest.raises(RuntimeContractError, match="seed/determinism"):
        validate_post_attempt_runtime_evidence(evidence)


def test_post_attempt_seed_evidence_precedes_pit_and_binds_seed42() -> None:
    events: list[tuple[str, object]] = []
    class Random:
        def seed(self, value): events.append(("python", value))
    class Numpy:
        random = type("NumpyRandom", (), {"seed": lambda _self, value: events.append(("numpy", value))})()
    class Cuda:
        def manual_seed_all(self, value): events.append(("cuda", value))
    class Cudnn:
        deterministic = False
        benchmark = True
    class Torch:
        cuda = Cuda()
        backends = type("Backends", (), {"cudnn": Cudnn()})()
        def manual_seed(self, value): events.append(("torch", value))
    evidence = establish_post_attempt_seed_evidence(
        torch=Torch(), random_module=Random(), numpy_module=Numpy())
    assert events == [("python", 42), ("numpy", 42), ("torch", 42), ("cuda", 42)]
    assert evidence == {"seed": 42, "python_random_seeded": True, "numpy_random_seeded": True,
                        "torch_manual_seeded": True, "torch_cuda_manual_seed_all": True,
                        "cudnn_deterministic": True, "cudnn_benchmark": False,
                        "deterministic_algorithms_forced": False}


def test_target_records_once_accepts_mapping_comparator_and_binds_each_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the production comparator Mapping branch, not only annotations."""
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as pooled_physical
    class Dataset:
        def __init__(self, names): self.calib_trialized_neural_features = {name: object() for name in names}
    external = tuple(f"ext-{index}" for index in range(6))
    within = tuple(f"in-{index}" for index in range(7))
    datamodule = type("DataModule", (), {
        "val_heldout_dataset": Dataset(external), "train_dataset": Dataset(within),
        "val_calib_heldout_sessions": {name: object() for name in external},
        "train_calib_heldin_sessions": {name: object() for name in within},
    })()
    bound: list[str] = []
    monkeypatch.setattr(pooled_physical, "_append_heldout_to_prepared_datamodule", lambda **_kwargs: {"same": True})
    monkeypatch.setattr(pooled_physical, "_record_from_session_material",
                        lambda *, surface, session, **_kwargs: {"key": f"{surface}|{session}"})
    monkeypatch.setattr(pooled_physical, "_bind_pooled_comparator",
                        lambda record, comparator: bound.append(f"{record['key']}:{comparator['marker']}"))
    pooled = {f"external_post30_local|{name}": {"marker": name} for name in external}
    pooled.update({f"within_post30|{name}": {"marker": name} for name in within})
    result = apfg_driver._target_records_once(datamodule=datamodule, pooled=pooled)
    assert len(result["records"]) == 13
    assert len(bound) == 13


def test_task_only_loss_uses_scaled_last_bin_without_teacher_path() -> None:
    class Student:
        def __init__(self) -> None:
            self.calls = 0
        def decode_with_identity(self, neural, identity):
            self.calls += 1
            return neural + identity
        def teacher(self, *_args):
            raise AssertionError("teacher must not be called")
    student = Student()
    neural = torch.zeros(2, 3, 2)
    identity = torch.full((2, 3, 2), 5.0)
    target = torch.ones(2, 3, 2)
    loss = task_only_scaled_last_bin_mse(torch=torch, student=student, neural=neural, identity=identity, target=target)
    assert torch.equal(loss, torch.tensor(0.0))
    assert student.calls == 1


def test_governed_r2_is_exact_reviewed_variance_weighted_operator() -> None:
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core
    target = np.asarray([[0.0, 0.0], [1.0, 100.0], [2.0, 200.0]], dtype=np.float32)
    prediction = np.asarray([[0.0, 20.0], [1.5, 110.0], [3.0, 205.0]], dtype=np.float32)
    governed = pseudo_core.variance_weighted_r2(target, prediction)
    assert _r2(target, prediction) == governed


def test_cache_key_rejects_noncanonical_digest_case() -> None:
    key = FrozenIdentityCacheKey(session_id="s", pool_state_sha256="A" * 64,
                                 ordered_activity_sha256=("a" * 64,) * 4,
                                     selected_support4_carrier_hz_sha256="b" * 64, normalized_side_sha256="c" * 64,
                                 normalizer_sha256="d" * 64, query_trial_index=30, requested_pool_size=4)
    with pytest.raises(RuntimeContractError, match="digest"):
        key.validate()


def test_real_streaming_spint_same_pool_identity_expands_to_batch_without_backbone_grad() -> None:
    torch.manual_seed(19)
    decoder = SpintModel(model_dim=8, num_covariates=2, window_size=50, num_heads=2, num_layers=1,
                         num_id_layers=1, dropout_rate=0.0, dynamic_dropout=False, tf_drop_rate=0.0)
    # Materialize the sealed decoder's historical LazyLinear before the real
    # StreamingSpint freeze path enumerates decoder parameters.
    _ = decoder(torch.randn(1, 50, 5), torch.randn(1, 4, 100, 5))
    native = SideFeatureEarlyPoolEncoder(trial_length=100, window_size=50, hidden_dim=8, side_dim=4,
                                         num_post_layers=1)
    student = StreamingSpintModel(decoder, native)
    student.freeze_decoder()
    wrapper = torch.nn.Module()
    wrapper.student = student
    adapter = freeze_and_install(wrapper.student)
    wrapper.eval()
    adapter.set_alpha_training(True)
    calibration = torch.randn(1, 4, 100, 5)
    side = torch.randn(1, 5, 4)
    one = adapter.forward_batch(calibration, side_features=side)
    identity = expand_identity_for_windows(identity=one, batch_size=3, channels=5, window_bins=50)
    neural = torch.randn(3, 50, 5)
    target = torch.zeros(3, 50, 2)
    loss = task_only_scaled_last_bin_mse(torch=torch, student=student, neural=neural, identity=identity, target=target)
    loss.backward()
    assert tuple(identity.shape) == (3, 5, 50)
    assert adapter.alpha.grad is not None and torch.isfinite(adapter.alpha.grad)
    assert all(parameter.grad is None for parameter in adapter.native.parameters())
    assert all(parameter.grad is None for parameter in student.decoder.parameters())


def test_source_replay_controller_is_m30_ready_then_derives_exact_cycle() -> None:
    state = _pool(session="s1", requested_size=30, support=(1, 2, 3, 4),
                  completed=tuple(range(30, 56)), query_trial=56, endpoint=500)
    coordinates = tuple(SourceCoordinate("s1", 100 + index, 56, state, _digest(999), _digest(998), _digest(997)) for index in range(33))
    material = SourceSessionMaterial("s1", (1, 2, 3, 4), np.zeros((96, 4), dtype=np.float64),
                                     np.zeros((96, 4), dtype=np.float32), np.zeros((96, 4), dtype=np.float32), _digest(997),
                                     np.zeros((60, 100, 96), dtype=np.float32), tuple(), coordinates)
    batches = canonical_m30_source_batches(material)
    assert [len(item) for item in batches] == [32, 1]
    derived = [pool_for_controller_coordinate(material=material, coordinate=coordinates[0],
                                               epoch_one_indexed=epoch, canonical_batch_ordinal=0)
               for epoch in range(1, 13)]
    assert {item.requested_size: sum(candidate.requested_size == item.requested_size for candidate in derived)
            for item in derived} == {4: 4, 10: 4, 30: 4}
    assert derived[0].member_trial_ids == (1, 2, 3, 4)
    assert derived[1].member_trial_ids == (1, 2, 3, 4, 50, 51, 52, 53, 54, 55)
    assert len(derived[2].member_trial_ids) == 30
    all_material = tuple(SourceSessionMaterial(f"s{index}", (1, 2, 3, 4), np.zeros((96, 4), dtype=np.float64),
                                               np.zeros((96, 4), dtype=np.float32), np.zeros((96, 4), dtype=np.float32), _digest(997),
                                               np.zeros((60, 100, 96), dtype=np.float32), tuple(), (SourceCoordinate(f"s{index}", 100, 56,
                                                                          _pool(session=f"s{index}", requested_size=30,
                                                                                support=(1, 2, 3, 4),
                                                                                completed=tuple(range(30, 56)),
                                                                                query_trial=56, endpoint=500), _digest(999), _digest(998), _digest(997)),))
                         for index in range(1, 8))
    summary = source_authority_summary(all_material)
    assert summary["activity_authority"] == "pooled_g00m_linear"
    assert summary["m30_causal_coordinate_count"] == 7


def test_raw_activity_pool_preserves_support_and_fixed_vs_uncapped_fifo_law() -> None:
    support = [np.full((100, 3), index, dtype=np.float32) for index in range(4)]
    fixed = OrderedRawActivityPool(support_trial_ids=(1, 2, 3, 4), support_activities=support, capacity=30)
    uncapped = OrderedRawActivityPool(support_trial_ids=(1, 2, 3, 4), support_activities=support, capacity=None)
    for trial in range(30, 61):
        activity = np.full((100, 3), trial, dtype=np.float32)
        fixed.commit_completed(trial_id=trial, activity=activity)
        uncapped.commit_completed(trial_id=trial, activity=activity)
    assert fixed.member_trial_ids[:4] == (1, 2, 3, 4)
    assert fixed.count == 30 and fixed.evictions == 5
    assert uncapped.count == 35 and uncapped.evictions == 0
    assert np.array_equal(fixed.stack()[:4], np.stack(support))
    with pytest.raises(SourceReplayError, match="duplicate"):
        fixed.commit_completed(trial_id=60, activity=np.zeros((100, 3), dtype=np.float32))


def test_public_cli_is_inert_and_clean_package_import_does_not_import_torch() -> None:
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": str(REPO)})
    command = [sys.executable, "-S", "tfpd_exploration/scripts/run_m2_anchored_postfusion_gate_v1.py", "--dry-run",
               "--repo-root", str(REPO)]
    completed = subprocess.run(command, cwd=REPO, env=environment, text=True, capture_output=True, check=True)
    payload = json.loads(completed.stdout)
    assert payload["status"] == "INERT_PUBLIC_CLI__ROOT_ONLY_OPAQUE_CAPABILITY_REQUIRED"
    import_command = [sys.executable, "-S", "-c",
                      "import sys; import tfpd_exploration.src.m2_anchored_postfusion_gate_v1 as p; "
                      "assert 'torch' not in sys.modules; print(p.SCHEMA)"]
    imported = subprocess.run(import_command, cwd=REPO, env=environment, text=True, capture_output=True, check=True)
    assert imported.stdout.strip() == plan.SCHEMA


def test_actual_corrected_predecessor_graph_is_read_only_and_torch_free() -> None:
    """Read exactly the completed historical ten-leaf graph in a clean process."""
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": str(REPO)})
    command = [sys.executable, "-S", "-c", "\n".join((
        "import sys",
        "from pathlib import Path",
        "from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.binding import validate_operator_corrected_graph",
        f"w=validate_operator_corrected_graph(Path({str(REPO)!r}))",
        "assert w['closure_sha256'] == '95b8e9e07e39700089e8364f67b15a020d17eb723a56cb18c2222305a1b25225'",
        "assert 'torch' not in sys.modules",
        "print(w['body_sha256']['terminal.json'])",
    ))]
    completed = subprocess.run(command, cwd=REPO, env=environment, text=True, capture_output=True, check=True)
    assert completed.stdout.strip() == plan.CORRECTED_BODIES["terminal.json"]


def _synthetic_repo_with_closure(tmp_path: Path) -> Path:
    """Build a temp canonical root whose closure leaves are read-only links."""
    for relative in plan.CLOSURE_RELATIVES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(REPO / relative, destination)
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _issue_temp_capability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for name, value in plan.LIVE_ENV.items():
        monkeypatch.setenv(name, value)
    root = _synthetic_repo_with_closure(tmp_path)
    return root, apfg_driver._mint_synthetic_capability(root)


def test_actual_production_execute_typed_synthetic_success_topology(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, capability = _issue_temp_capability(monkeypatch, tmp_path)
    events: list[str] = []
    def bodies(artifact, progress):
        events.append("bodies")
        progress.update({"corrected_graph_attempted": True, "corrected_graph_verified": True,
                         "checkpoint_attempted": True, "checkpoint_opened": True,
                         "source_attempted": True, "source_complete": True,
                         "pooled_attempted": True, "pooled_verified": True,
                         "target_attempted": True, "target_opened": True, "target_complete": True})
        return {name: artifact.publish_json(name, {"name": name}) for name in
                ("source_authority.json", "alpha_selection.json", "input_authority.json", "score.json")}
    runtime = apfg_driver._SyntheticProductionRuntime(
        launch_payload={"schema": "synthetic_launch", "cuda_visible_devices": "0", "physical_device": 0},
        body_builder=bodies,
        terminal_builder=lambda published, progress: {"source_authority_sha256": published["source_authority.json"],
                                                       "alpha_selection_sha256": published["alpha_selection.json"],
                                                       "input_authority_sha256": published["input_authority.json"],
                                                       "score_sha256": published["score.json"], "progress": dict(progress)},
    )
    terminal, failure = apfg_driver.execute_production(capability, _synthetic_runtime=runtime)
    assert terminal and failure is None and events == ["bodies"]
    names = {item.name for item in (root / plan.RESULT_ROOT_RELATIVE).iterdir()}
    assert names == {f"{name}{suffix}" for name in ("attempt.json", "launch.json", "source_authority.json", "alpha_selection.json", "input_authority.json", "score.json", "terminal.json") for suffix in ("", ".sha256")}
    with pytest.raises(AdmissionError, match="reused"):
        apfg_driver.execute_production(capability, _synthetic_runtime=runtime)


def test_actual_production_execute_source_safety_and_zero_drift_fail_before_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, capability = _issue_temp_capability(monkeypatch, tmp_path)
    target_calls: list[str] = []
    def safety_failure(_artifact, progress):
        progress.update({"source_attempted": True, "source_complete": True,
                         "source_safety_gate_passed": False})
        raise RuntimeError("source safety gate failed before target")
    runtime = apfg_driver._SyntheticProductionRuntime(
        launch_payload={"schema": "synthetic_launch"}, body_builder=safety_failure,
        terminal_builder=lambda _published, _progress: {})
    terminal, failure = apfg_driver.execute_production(capability, _synthetic_runtime=runtime)
    assert terminal is None and failure and target_calls == []
    failure_payload = json.loads((root / plan.RESULT_ROOT_RELATIVE / "failure.json").read_text())
    assert failure_payload["progress"]["target_attempted"] is False
    assert failure_payload["progress"]["source_safety_gate_passed"] is False
    # A fresh root verifies a comparator/selection drift is also terminal-xor-failure.
    root2, cap2 = _issue_temp_capability(monkeypatch, tmp_path / "second")
    runtime2 = apfg_driver._SyntheticProductionRuntime(
        launch_payload={"schema": "synthetic_launch"},
        body_builder=lambda _artifact, progress: (progress.update({"pooled_attempted": True, "pooled_verified": True}),
                                                   (_ for _ in ()).throw(RuntimeError("ZERO/FIXED30 comparator digest drift")))[1],
        terminal_builder=lambda _published, _progress: {})
    terminal, failure = apfg_driver.execute_production(cap2, _synthetic_runtime=runtime2)
    assert terminal is None and failure
    assert not (root2 / plan.RESULT_ROOT_RELATIVE / "terminal.json").exists()


def test_actual_production_execute_final_witness_drift_is_failure_xor_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, capability = _issue_temp_capability(monkeypatch, tmp_path)
    def bodies(artifact, progress):
        progress.update({"source_attempted": True, "source_complete": True, "pooled_attempted": True,
                         "pooled_verified": True, "target_attempted": True, "target_opened": True})
        return {"source_authority.json": artifact.publish_json("source_authority.json", {"source_alpha_optimizer_updates": 12}),
                "alpha_selection.json": artifact.publish_json("alpha_selection.json", {"selected_epoch": 2}),
                "input_authority.json": artifact.publish_json("input_authority.json", {"records": []}),
                "score.json": artifact.publish_json("score.json", {"target_scoring_parameter_updates": 0})}
    runtime = apfg_driver._SyntheticProductionRuntime(
        launch_payload={"schema": "synthetic_launch"}, body_builder=bodies,
        terminal_builder=lambda _published, _progress: {"source_alpha_optimizer_updates": 12,
                                                        "target_parameter_updates": 0, "target_updates": 0},
        final_validator=lambda: (_ for _ in ()).throw(RuntimeError("corrected/POOLED witness drift")),
    )
    terminal, failure = apfg_driver.execute_production(capability, _synthetic_runtime=runtime)
    assert terminal is None and failure
    root_path = root / plan.RESULT_ROOT_RELATIVE
    assert (root_path / "failure.json").exists() and not (root_path / "terminal.json").exists()
    failure_payload = json.loads((root_path / "failure.json").read_text())
    assert failure_payload["published_prefix"].keys() >= {"attempt.json", "launch.json", "score.json"}


def test_capability_rejects_environment_and_named_root_drift_before_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, capability = _issue_temp_capability(monkeypatch, tmp_path)
    runtime = apfg_driver._SyntheticProductionRuntime(launch_payload={}, body_builder=lambda *_: {}, terminal_builder=lambda *_: {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(AdmissionError, match="environment drift"):
        apfg_driver.execute_production(capability, _synthetic_runtime=runtime)
    # A separate cap cannot be redirected to a named root created after issue.
    for name, value in plan.LIVE_ENV.items():
        monkeypatch.setenv(name, value)
    root2, cap2 = _issue_temp_capability(monkeypatch, tmp_path / "root-swap")
    (root2 / plan.RESULT_ROOT_RELATIVE).mkdir()
    with pytest.raises(AdmissionError, match="unavailable|fresh"):
        apfg_driver.execute_production(cap2, _synthetic_runtime=runtime)


def test_target_cpu_split_releases_all_three_source_models_and_optimizer_state() -> None:
    class Parameter:
        def __init__(self, value=0.0, trainable=False):
            self.requires_grad = trainable; self.device = "cuda:0"; self.grad = object(); self.value = value
        def requires_grad_(self, value): self.requires_grad = value; return self
        def detach(self): return self
        def cpu(self): return self
        def __float__(self): return self.value
    class Adapter:
        def __init__(self, alpha): self.alpha = alpha
    class Student:
        def __init__(self, alpha): self.id_encoder = Adapter(alpha)
    class Module:
        def __init__(self, alpha):
            self.student = Student(alpha); self.training = True
            self._parameters = [Parameter(), Parameter(), alpha]
        def to(self, device):
            for parameter in self._parameters: parameter.device = str(device)
            return self
        def eval(self): self.training = False; return self
        def parameters(self): return tuple(self._parameters)
    class Cuda:
        def __init__(self): self.calls = []
        def synchronize(self, device): self.calls.append(("synchronize", str(device)))
        def memory_allocated(self, device): self.calls.append(("allocated", str(device))); return 64
        def memory_reserved(self, device): self.calls.append(("reserved", str(device))); return 128
        def empty_cache(self): self.calls.append(("empty_cache",))
        def is_initialized(self): return True
    class Torch:
        def __init__(self): self.cuda = Cuda()
        @staticmethod
        def device(value): return value
    torch_fake = Torch(); optimizer = type("Opt", (), {"state": {"alpha": object()}})()
    zero, learned, selection = Module(Parameter(0.0, trainable=True)), Module(Parameter(0.375, trainable=True)), Module(Parameter(0.125, trainable=True))
    # This is the exact pre-fix condition: disabling adapter training leaves
    # alpha.requires_grad=True, so the prior frozen assertion would reject.
    assert zero.student.id_encoder.alpha.requires_grad is True
    evidence = apfg_driver._move_frozen_target_models_to_cpu(
        torch=torch_fake, zero_module=zero, learned_module=learned, selection_module=selection,
        selection_optimizer=optimizer)
    assert all(not module.training and all(str(item.device) == "cpu" and not item.requires_grad and item.grad is None
                                           for item in module.parameters())
               for module in (zero, learned, selection))
    assert optimizer.state == {}
    assert evidence["learned_refit_alpha_before_cpu_score"] == 0.375
    assert evidence["zero_alpha_positive_zero_before_cpu_score"] is True
    assert float(zero.student.id_encoder.alpha) == 0.0
    assert np.signbit(float(zero.student.id_encoder.alpha)) is np.False_
    assert evidence["target_scoring_device"] == "cpu" and evidence["target_cpu_decode_batch_size"] == 1024
    assert evidence["gpu0_allocator_before_release"] == {"allocated": 64, "reserved": 128}
    assert evidence["gpu0_allocator_after_release"] == {"allocated": 64, "reserved": 128}
    assert ("empty_cache",) in torch_fake.cuda.calls


def test_prepare_requires_same_initialized_gpu0_launch_attestation() -> None:
    class Cuda:
        def __init__(self, initialized=True, current=0, count=1): self.initialized, self.current, self.count = initialized, current, count
        def is_available(self): return True
        def is_initialized(self): return self.initialized
        def current_device(self): return self.current
        def device_count(self): return self.count
    class Torch:
        def __init__(self, cuda): self.cuda = cuda
    attestation = {"logical_device": 0, "physical_device": 0, "cuda_visible_devices": "0", "cuda_initialized": True}
    validate_cuda_launch_attestation(torch=Torch(Cuda()), device="cuda:0", launch_attestation=attestation)
    with pytest.raises(TrainingError, match="initialized"):
        validate_cuda_launch_attestation(torch=Torch(Cuda(initialized=False)), device="cuda:0", launch_attestation=attestation)
    with pytest.raises(TrainingError, match="topology"):
        validate_cuda_launch_attestation(torch=Torch(Cuda(count=2)), device="cuda:0", launch_attestation=attestation)
