from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import pytest

from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import binding, laws, physical, plan
from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import lifecycle, driver
from tfpd_exploration.src.m2_postfusion_probe_v1 import memory

ROOT = Path(__file__).resolve().parents[2]


def _publish(path: Path, body: bytes) -> str:
    path.write_bytes(body); path.chmod(0o444)
    digest = hashlib.sha256(body).hexdigest()
    side = path.with_name(path.name + ".sha256"); side.write_text(f"{digest}  {path.name}\n", encoding="ascii"); side.chmod(0o444)
    return digest


def _producer(tmp_path: Path):
    root = tmp_path / "screen"; root.mkdir()
    _publish(root / "attempt.json", b'{"schema":"attempt"}')
    _publish(root / "launch.json", b'{"schema":"launch"}')
    _publish(root / "source_authority.json", b'{"source_heldin_in_sample_monitor":true}')
    checkpoint_bodies = {arm: (arm + "-state").encode() for arm in plan.ARMS}
    checkpoints = {}
    digests = {}
    for arm in plan.ARMS:
        filename = f"source_best_{arm.lower().replace('-', '_')}.pt"
        digest = _publish(root / filename, checkpoint_bodies[arm]); digests[filename] = digest
        checkpoints[arm] = {"epoch": 12, "student_state_sha256": "a" * 64, "filename": filename, "sha256": digest}
    screen = {"matched_prefusion_control_trained": False, "source_best_checkpoints": checkpoints}
    _publish(root / "screen.json", json.dumps(screen, sort_keys=True).encode())
    terminal = {"status": "TERMINAL", "terminal_xor_failure": True}
    terminal_sha = _publish(root / "terminal.json", json.dumps(terminal, sort_keys=True).encode())
    all_names = ["attempt.json", "launch.json", "source_authority.json", "screen.json", "terminal.json",
                 "source_best_pf_mean.pt", "source_best_pf_r1.pt", "source_best_pf_r50.pt"]
    shas = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in all_names}
    return root, {"screen_root_relative": str(root), "sha256": shas}


def test_static_authority_and_deferred_live_literals() -> None:
    assert plan.validate_static(ROOT)["workorder_sha256"] == plan.WORKORDER_SHA256
    assert binding.require_live_literals()["screen_root_relative"].endswith("/screen")


def test_cpu_admission_requires_every_frozen_cpu_environment_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    with pytest.raises(driver.DriverError, match="environment drift"):
        driver._require_cpu_environment()
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    driver._require_cpu_environment()


def test_held_screen_graph_accepts_exact_three_checkpoint_topology_and_rejects_extra(tmp_path: Path) -> None:
    root, literals = _producer(tmp_path)
    witness = binding.validate_screen_graph(root, literals)
    assert set(witness["digests"]) >= {"terminal.json", "source_best_pf_mean.pt"}
    _publish(root / "extra.json", b"{}")
    with pytest.raises(binding.BindingError): binding.validate_screen_graph(root, literals)


def test_cpu_single_prepare_strict_loads_actual_immutable_screen_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production reconstruction seam is source-only and CPU-only.

    This intentionally descriptor-reads the one audited producer graph and
    its three checkpoint *bytes*, but does not instantiate a scorer root,
    target loader, score surface, or CUDA context.
    """
    import torch
    from tfpd_exploration.src.pit_m2_v1 import trainer

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    screen = ROOT / str(plan.LIVE_PRODUCER_LITERALS["screen_root_relative"])
    held = binding.validate_screen_graph(screen, plan.LIVE_PRODUCER_LITERALS,
                                         capture_checkpoint_bytes=True)
    original = trainer.PitM2ArmedRunner.prepare
    calls = 0

    def counted(self, *, attach_operator: bool = True):
        nonlocal calls
        calls += 1
        return original(self, attach_operator=attach_operator)

    monkeypatch.setattr(trainer.PitM2ArmedRunner, "prepare", counted)
    prepared = physical.prepare_three_frozen_arms_from_held_screen(
        repo_root=ROOT, checkpoint_bytes=held["checkpoint_bytes"])
    assert calls == 1 == prepared["pit_prepare_calls"]
    assert set(prepared["modules"]) == set(plan.ARMS)
    assert set(prepared["adapters"]) == set(plan.ARMS)
    assert prepared["datamodule"].val_heldout_dataset is None
    for arm in plan.ARMS:
        evidence = prepared["evidence"][arm]
        assert evidence["strict_load"] is True
        assert evidence["all_parameters_frozen"] is True
        assert evidence["student_state_after_load_sha256"] == plan.LIVE_PRODUCER_LITERALS["checkpoints"][arm]["student_state_sha256"]
        assert all(not parameter.requires_grad for parameter in prepared["modules"][arm].parameters())
    pooled, pooled_evidence = physical._strict_sealed_pooled_clone(repo_root=ROOT,
                                                                     base_module=prepared["base_module"])
    assert pooled_evidence["strict_load"] is True
    assert len(pooled_evidence["selected_t4_checkpoint_sha256"]) == 64
    assert all(not parameter.requires_grad for parameter in pooled.parameters())
    assert not torch.cuda.is_initialized()


def test_bootstrap_is_session_deterministic_and_row_topology_is_exact() -> None:
    values = {f"s{i}": float(i) / 10 for i in range(6)}
    assert laws.session_bootstrap(values) == laws.session_bootstrap(values)
    rows = []
    for arm in plan.ARMS:
        for law in plan.LAWS:
            for surface, count in plan.ROSTER_SIZES.items():
                for index in range(count):
                    rows.append({"arm": arm, "memory_law": law, "surface": surface,
                                 "session": f"{surface}-{index}", "budget": 4, "r2": 0.1,
                                 "input_authority_key":"x", "prediction_sha256":"a"*64,"target_sha256":"b"*64,
                                 "window_count":1,"initial_members":4,"final_members":4,"commits":0,"evictions":0,
                                 "causal_trace_sha256":"c"*64,"model_state_before_sha256":"d"*64,"model_state_after_sha256":"d"*64,
                                 "parameter_updates":0,"target_updates":0})
    laws.validate_rows(rows)
    with pytest.raises(laws.LawError): laws.validate_rows(rows[:-1])


def test_causal_decode_before_commit_and_uncapped_fixed_parity_before_eviction() -> None:
    class Identity:
        def __init__(self, value): self.value = float(value)
        def __add__(self, other): return Identity(self.value + other.value)
        def __sub__(self, other): return Identity(self.value - other.value)
        def __truediv__(self, n): return Identity(self.value / n)
    support = [Identity(1), Identity(3), Identity(5), Identity(7)]
    fixed = memory.PostFusionUniformIdentityPool(support_identities=support, capacity=30)
    uncapped = memory.PostFusionUniformIdentityPool(support_identities=support, capacity=None)
    make = lambda value: Identity(float(np.asarray(value)[0, 0]))
    decode = lambda identity: np.asarray([[identity.value]], dtype=np.float32)
    activity = [np.asarray([[9 + i]], dtype=np.float32) for i in range(3)]
    a = physical.causal_rollout(pool=fixed, support_identities=support, query_activities=activity, decode=decode, make_identity=make)
    b = physical.causal_rollout(pool=uncapped, support_identities=support, query_activities=activity, decode=decode, make_identity=make)
    assert a["predictions"] == b["predictions"] and a["evictions"] == 0 and b["evictions"] == 0
    assert a["trace"][0][1] == 4 and a["trace"][0][2] == 5


def test_materialize_13_inputs_uses_one_prepared_datamodule_and_binds_same_input_pooled() -> None:
    """Actual-shape synthetic authority: no arm decode or score root exists."""
    class Dataset:
        window_size = 50
        side_feature_mean = np.zeros(4, dtype=np.float32)
        side_feature_std = np.ones(4, dtype=np.float32)

        def __init__(self, sessions):
            self.window_indices = [(session, 60) for session in sessions]
            self.calib_trialized_neural_features = {
                session: np.full((30, 100, 2), index + 1, dtype=np.float32)
                for index, session in enumerate(sessions)
            }
            self.covariate_data = {
                session: np.arange(320, dtype=np.float32).reshape(160, 2)
                for session in sessions
            }
            self.neural_data = {session: np.arange(320, dtype=np.float32).reshape(160, 2)
                                for session in sessions}
            self.eval_mask = {session: np.ones(160, dtype=bool) for session in sessions}
            self.side_features = {session: np.full((2, 4), index + 1, dtype=np.float32)
                                  for index, session in enumerate(sessions)}

    within = tuple(f"wi{index}" for index in range(7))
    external = tuple(f"ex{index}" for index in range(6))
    train = Dataset(within)

    class DataModule:
        train_dataset = train
        train_session_names = within
        train_calib_heldin_sessions = {
            session: {"covariates_mean": np.zeros(2, dtype=np.float32),
                      "covariates_std": np.ones(2, dtype=np.float32)} for session in within
        }
        native_t4_normalization = {"feature_group": "t4", "train_sessions": list(within),
                                   "mean": np.zeros(4, dtype=np.float32), "std": np.ones(4, dtype=np.float32)}
        val_heldout_dataset = None

        def __init__(self):
            self.build_calls = 0
            self.val_calib_heldout_sessions = None

        def _build_heldout_dataset(self, cov_mean, cov_std, task_config, *, side_feature_group, side_feature_mean, side_feature_std):
            assert task_config.task is not None and side_feature_group == "t4"
            assert np.array_equal(cov_mean, np.zeros(2)) and np.array_equal(cov_std, np.ones(2))
            assert np.array_equal(side_feature_mean, self.native_t4_normalization["mean"])
            assert np.array_equal(side_feature_std, self.native_t4_normalization["std"])
            self.build_calls += 1
            self.val_calib_heldout_sessions = {session: {} for session in external}
            self.val_heldout_dataset = Dataset(external)

    class Replay:
        @staticmethod
        def _g_session_views(raw_sessions, *, session): return {"session": session}
        @staticmethod
        def g_support_material(*, session, views):
            return {"selected": np.asarray([0, 2, 5, 9], dtype=np.int64),
                    "activities": np.full((30, 100, 2), len(session), dtype=np.float32),
                    "raw_m30_hz": np.full((2, 4), len(session), dtype=np.float32)}
        @staticmethod
        def _side_from_raw(raw_t4, dataset):
            return (np.asarray(raw_t4, dtype=np.float32) - dataset.side_feature_mean) / dataset.side_feature_std
        @staticmethod
        def g_query_rows(*, ds, session, views):
            count = 2 if session.startswith("ex") else 27
            return tuple({"trial_id": f"{session}:trial:{30 + index}", "position": 30 + index,
                           "activity": np.full((100, 2), index + 1, dtype=np.float32),
                           "metric_starts": np.asarray([60 + index], dtype=np.int64),
                           "causal_starts": np.asarray([60 + index], dtype=np.int64)}
                         for index in range(count))

    comparator_calls: list[str] = []
    def comparator(record):
        comparator_calls.append(record["key"])
        return {"r2": 0.1, "window_count": record["window_count"],
                "query_starts_sha256": record["query_starts_sha256"],
                "target_sha256": record["target_sha256"], "prediction_sha256": "f" * 64}

    dm = DataModule()
    output = physical.materialize_13_inputs_and_pooled_comparators(
        prepared={"datamodule": dm}, pooled_comparator=comparator, replay_module=Replay)
    assert dm.build_calls == 1
    assert output["materializations"] == 13 and output["pooled_comparators_bound"] == 13
    assert output["postfusion_arm_rollouts"] == 0
    assert len(comparator_calls) == 13 == len(output["records"])
    assert output["append_heldout_evidence"]["source_preserved_exact"] is True
    for record in output["records"].values():
        assert record["support_indices"] == [0, 2, 5, 9]
        assert record["pooled_comparator"]["target_sha256"] == record["target_sha256"]
        assert "_runtime" in record

    import torch
    class Adapter(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls = 0
        def forward_batch(self, calib_trials, *, side_features=None):
            self.calls += int(calib_trials.shape[1])
            # Distinct, trainable-adapter path; not the historical stream API.
            values = calib_trials.mean(dim=2).mean(dim=1).unsqueeze(-1)
            return values.repeat(1, 1, 50)
    class Student(torch.nn.Module):
        def __init__(self, adapter): super().__init__(); self.id_encoder = adapter
        def decode_with_identity(self, neural, identity):
            base = identity.mean(dim=(1, 2))[:, None, None] + neural.mean(dim=2, keepdim=True)
            return torch.cat((base, base + 0.25), dim=2)
    class Model(torch.nn.Module):
        def __init__(self, adapter): super().__init__(); self.student = Student(adapter)
    adapters = {arm: Adapter() for arm in plan.ARMS}
    modules = {arm: Model(adapters[arm]).eval() for arm in plan.ARMS}
    for model in modules.values():
        for parameter in model.parameters(): parameter.requires_grad_(False)
    rows = physical.score_78_rows_from_materialized(
        prepared={"modules": modules, "adapters": adapters}, materialized=output,
        expected_channels=2)
    assert len(rows) == 78
    assert all(row["support_never_evicted"] and row["parameter_updates"] == 0 for row in rows)
    assert all(row["support_retained_object_order_exact"] for row in rows)
    assert all(len(row["support_identity_order_sha256"]) == 64 for row in rows)
    external = [row for row in rows if row["surface"] == "external_post30_local"]
    assert all(row["evictions"] == 0 for row in external)
    for arm in plan.ARMS:
        paired = [row for row in external if row["arm"] == arm]
        assert paired[0]["prediction_sha256"] == paired[1]["prediction_sha256"]
        assert adapters[arm].calls == 13 * 4 + (6 * 2 + 7 * 27)
    within_fixed = [row for row in rows if row["surface"] == "within_post30" and row["memory_law"] == "FIXED30"]
    assert all(row["evictions"] == 1 for row in within_fixed)
    sample = next(iter(output["records"].values()))
    bad_activity = dict(sample); bad_activity["query_activity_sha256"] = "0" * 64
    with pytest.raises(physical.PhysicalError, match="query activity"):
        physical._validate_runtime_record(bad_activity)
    bad_reorder = dict(sample); bad_runtime = dict(sample["_runtime"])
    bad_runtime["query_rows"] = tuple(reversed(bad_runtime["query_rows"])); bad_reorder["_runtime"] = bad_runtime
    with pytest.raises(physical.PhysicalError, match="trial-order|trace"):
        physical._validate_runtime_record(bad_reorder)
    bad_t4 = dict(sample); bad_runtime = dict(sample["_runtime"]); bad_support = dict(bad_runtime["support"])
    bad_support["raw_m30_hz"] = np.asarray(bad_support["raw_m30_hz"], dtype=np.float32) + 1.0
    bad_runtime["support"] = bad_support; bad_t4["_runtime"] = bad_runtime
    with pytest.raises(physical.PhysicalError, match="support/T4"):
        physical._validate_runtime_record(bad_t4)


def test_cpu_decode_chunking_parity_and_reviewed_metric_digest() -> None:
    """CPU decode batch 1024 is a fixed throughput law, not a new operator."""
    import torch
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    assert plan.CPU_DECODE_BATCH_SIZE == 1024
    assert not hasattr(physical, "reconstruct_frozen_arm")

    class Student:
        def decode_with_identity(self, neural, identity):
            # Nontrivial dependence on both batched windows and the identity.
            feature = neural.square().mean(dim=2, keepdim=True) + identity.mean().to(neural.dtype)
            return torch.cat((feature, feature * 0.25 + 0.125), dim=2)

    class Model:
        student = Student()

    rng = np.random.default_rng(42)
    windows = rng.normal(size=(1031, 50, 2)).astype(np.float32)
    identity = torch.from_numpy(rng.normal(size=(1, 2, 50)).astype(np.float32))
    pred32 = physical._decode_identity_cpu(torch=torch, model=Model(), neural_windows=windows,
                                           identity=identity, batch_size=32)
    pred1024 = physical._decode_identity_cpu(torch=torch, model=Model(), neural_windows=windows,
                                             identity=identity, batch_size=plan.CPU_DECODE_BATCH_SIZE)
    target = rng.normal(size=pred32.shape).astype(np.float32)
    # Chunk partitioning may change CPU reduction scheduling in other Torch
    # builds.  The route reports a numeric tolerance rather than claiming
    # impossible bitwise invariance across chunks.
    assert np.max(np.abs(pred32 - pred1024)) <= 1e-6
    assert abs(physical._r2(target, pred32) - physical._r2(target, pred1024)) <= 1e-6

    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    assert physical._governed_array_sha256(values) == pseudo_core.array_sha256(values)
    assert physical._governed_array_sha256(values) != physical._array_sha256(values)


def test_support_retention_requires_exact_object_content_order() -> None:
    import torch

    support = [torch.full((1, 2, 50), value, dtype=torch.float32) for value in (1.0, 2.0)]

    class Pool:
        def __init__(self, members): self._members = members
        def member_identities(self): return list(self._members)

    ordered = Pool([support[0], support[1], torch.zeros((1, 2, 50))])
    assert len(physical._assert_support_retained(pool=ordered, support_identities=support)) == 2
    reordered = Pool([support[1], support[0]])
    with pytest.raises(physical.PhysicalError, match="object"):
        physical._assert_support_retained(pool=reordered, support_identities=support)


def test_trained_identity_enforces_production_96_by_50_contract() -> None:
    import torch

    class Adapter:
        def forward_batch(self, trials, *, side_features=None):
            assert tuple(trials.shape) == (1, 1, 100, 96)
            assert tuple(side_features.shape) == (1, 96, 4)
            return torch.ones((1, 96, 50), dtype=torch.float32)

    side = torch.zeros((1, 96, 4), dtype=torch.float32)
    identity = physical.trained_trial_identity(adapter=Adapter(), torch=torch,
                                                activity=np.zeros((100, 96), dtype=np.float32),
                                                side_tensor=side)
    assert tuple(identity.shape) == (1, 96, 50)
    with pytest.raises(physical.PhysicalError, match="activity row shape"):
        physical.trained_trial_identity(adapter=Adapter(), torch=torch,
                                        activity=np.zeros((100, 95), dtype=np.float32), side_tensor=side)


def test_materialize_13_inputs_fails_closed_if_narrow_heldout_build_mutates_source() -> None:
    class Dataset:
        window_indices = [("s", 60)]
        side_feature_mean = np.zeros(4, dtype=np.float32)
        side_feature_std = np.ones(4, dtype=np.float32)
        calib_trialized_neural_features = {"s": np.zeros((30, 100, 2), dtype=np.float32)}

    class DataModule:
        train_dataset = Dataset()
        train_session_names = ("s",)
        train_calib_heldin_sessions = {"s": {"covariates_mean": np.zeros(2), "covariates_std": np.ones(2)}}
        native_t4_normalization = {"feature_group": "t4", "train_sessions": ["s"],
                                   "mean": np.zeros(4), "std": np.ones(4)}
        val_heldout_dataset = None
        def _build_heldout_dataset(self, *_args, **_kwargs):
            self.train_dataset.side_feature_mean[0] = 1.0

    with pytest.raises(physical.PhysicalError, match="source authority"):
        physical.materialize_13_inputs_and_pooled_comparators(
            prepared={"datamodule": DataModule()}, pooled_comparator={}, replay_module=object())


def test_attempt_terminal_xor_failure_lifecycle(tmp_path: Path) -> None:
    terminal, failure = lifecycle.execute(tmp_path, attempt={"stage": "attempt"}, launch=lambda: {"cpu": True},
        bodies=lambda artifact: {"score.json": artifact.publish_json("score.json", {"rows": []})},
        terminal=lambda shas: {"score_sha256": shas["score.json"]}, progress=lambda: {"rows": 0})
    assert terminal is not None and failure is None


def test_private_synthetic_lifecycle_is_attempt_first_prefix_preserving_and_one_shot(tmp_path: Path) -> None:
    for relative in plan.CLOSURE_RELATIVES:
        source, target = ROOT / relative, tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True)
    capability = driver._mint_synthetic(tmp_path)
    terminal, failure = driver._execute_synthetic_for_test(capability=capability)
    assert terminal is not None and failure is None
    score_root = tmp_path / plan.RESULT_ROOT_RELATIVE
    score = json.loads((score_root / "score.json").read_text())
    assert score["rows"] == [] and (score_root / "terminal.json").exists()
    assert not (score_root / "failure.json").exists()
    assert not hasattr(driver, "execute_deferred")
    with pytest.raises(driver.DriverError): driver._execute_synthetic_for_test(capability=capability)

    for relative in plan.CLOSURE_RELATIVES:
        source, target = ROOT / relative, tmp_path / "failure" / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    (tmp_path / "failure/tfpd_exploration/results").mkdir(parents=True)
    failed = driver._mint_synthetic(tmp_path / "failure")
    terminal, failure = driver._execute_synthetic_for_test(capability=failed, fail_after="input_authority")
    assert terminal is None and failure is not None
    failed_root = tmp_path / "failure" / plan.RESULT_ROOT_RELATIVE
    assert (failed_root / "attempt.json").exists() and (failed_root / "launch.json").exists()
    assert (failed_root / "input_authority.json").exists() and (failed_root / "failure.json").exists()
    assert not (failed_root / "score.json").exists() and not (failed_root / "terminal.json").exists()
    failure_body = json.loads((failed_root / "failure.json").read_text())
    assert set(failure_body["published_prefix"]) == {"attempt.json", "launch.json", "input_authority.json"}


def _pooled_predecessor_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An actual-shaped immutable JSON body copied into a temporary held root."""
    source = ROOT / plan.POOLED_COMPARATOR_ROOT_RELATIVE / plan.POOLED_COMPARATOR_BODY
    root = tmp_path / "pooled"; root.mkdir()
    body = source.read_bytes()
    digest = _publish(root / plan.POOLED_COMPARATOR_BODY, body)
    monkeypatch.setattr(plan, "POOLED_COMPARATOR_SHA256", digest)
    return root


def test_held_pooled_predecessor_actual_descriptor_cpu_only() -> None:
    """The immutable historical POOLED comparator is JSON-only and no-CUDA."""
    code = """from pathlib import Path
import sys
from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1.binding import validate_pooled_comparator_score
r=validate_pooled_comparator_score(Path(sys.argv[1]))
assert r['body_sha256'] == '455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6'
assert len(r['comparators']) == 13
assert 'torch' not in sys.modules
"""
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    root = ROOT / plan.POOLED_COMPARATOR_ROOT_RELATIVE
    run = subprocess.run([sys.executable, "-S", "-c", code, str(root)], env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr


def test_held_pooled_predecessor_rejects_topology_and_semantic_adversaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _pooled_predecessor_fixture(tmp_path, monkeypatch)
    witness = binding.validate_pooled_comparator_score(root)
    assert len(witness["comparators"]) == 13
    (root / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(binding.BindingError, match="missing/extra"):
        binding.validate_pooled_comparator_score(root)
    (root / "extra.json").unlink()
    (root / plan.POOLED_COMPARATOR_BODY).chmod(0o644)
    with pytest.raises(binding.BindingError, match="mode/type"):
        binding.validate_pooled_comparator_score(root)

    # Fresh fixtures bind their own body SHA so the semantic codec—not a
    # deliberate byte mismatch—is the observed failure.
    for mutator, pattern in (
        (lambda x: x["rows"].__setitem__(0, x["rows"][1]), "canonical row order"),
        (lambda x: x["rows"][7].__setitem__("session_id", x["rows"][17]["session_id"]), "canonical row order"),
        (lambda x: x["rows"][7].__setitem__("query_starts_sha256", "0" * 64), "governed evidence"),
        (lambda x: x["rows"][7].__setitem__("parameter_updates", 1), "forbidden update"),
    ):
        staged = tmp_path / f"semantic-{pattern[:3]}-{len(list(tmp_path.iterdir()))}"; staged.mkdir()
        payload = json.loads((ROOT / plan.POOLED_COMPARATOR_ROOT_RELATIVE / plan.POOLED_COMPARATOR_BODY).read_text())
        mutator(payload)
        digest = _publish(staged / plan.POOLED_COMPARATOR_BODY, json.dumps(payload, sort_keys=True).encode())
        monkeypatch.setattr(plan, "POOLED_COMPARATOR_SHA256", digest)
        with pytest.raises(binding.BindingError, match=pattern):
            binding.validate_pooled_comparator_score(staged)


def _pf_rows_against_pooled(comparators: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Mean and R1 intentionally tie above the nomination threshold: frozen
    # complexity must choose PF-MEAN, not incidental dictionary ordering.
    offsets = {"PF-MEAN": .011, "PF-R1": .011, "PF-R50": .004}
    for surface in plan.SURFACES:
        for session in plan.POOLED_SESSION_ORDER[surface]:
            comparator = comparators[f"{surface}|{session}"]
            for arm in plan.ARMS:
                for law in plan.LAWS:
                    rows.append({"arm": arm, "memory_law": law, "surface": surface, "session": session,
                                 "budget": 4, "r2": float(comparator["r2"]) + offsets[arm],
                                 "input_authority_key": f"{surface}|{session}",
                                 "prediction_sha256": "a" * 64, "target_sha256": comparator["target_sha256"],
                                 "window_count": comparator["window_count"], "initial_members": 4,
                                 "final_members": 4, "commits": 0, "evictions": 0,
                                 "causal_trace_sha256": "c" * 64, "model_state_before_sha256": "d" * 64,
                                 "model_state_after_sha256": "d" * 64, "parameter_updates": 0, "target_updates": 0})
    return rows


def test_recompute_pooled_contrasts_bootstrap_and_exploratory_nomination() -> None:
    held = binding.validate_pooled_comparator_score(ROOT / plan.POOLED_COMPARATOR_ROOT_RELATIVE)
    comparators = held["comparators"]
    assert isinstance(comparators, dict)
    result = laws.recompute(_pf_rows_against_pooled(comparators), comparators)
    pooled_contrasts = [key for key in result["contrasts"] if "minus_POOLED" in key]
    assert len(pooled_contrasts) == 12
    sample = result["contrasts"]["PF-MEAN|UNCAPPED_minus_POOLED|external_post30_local"]
    assert sample["mean_delta"] == pytest.approx(.011)
    assert sample["positive_sessions"] == 6 and sample["bootstrap"]["unit"] == "session"
    nomination = result["nomination"]
    assert nomination["nominated_arm"] == "PF-MEAN"
    assert nomination["matched_prefusion_control_trained"] is False and nomination["exploratory_only"] is True
    bad = {key: dict(value) for key, value in comparators.items()}
    bad["external_post30_local|ses-2020-10-30-Run1"]["bootstrap_unit"] = "window"
    with pytest.raises(laws.LawError, match="window bootstrap"):
        laws.recompute(_pf_rows_against_pooled(comparators), bad)
    bad = {key: dict(value) for key, value in comparators.items()}
    bad["external_post30_local|ses-2020-10-30-Run1"]["window_count"] = 2
    with pytest.raises(laws.LawError, match="target/window"):
        laws.recompute(_pf_rows_against_pooled(comparators), bad)


def test_inert_cli_imports_no_torch_under_python_s() -> None:
    code = "import importlib.util,sys;p=sys.argv[1];s=importlib.util.spec_from_file_location('x',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert m.main(['--dry-run'])==0;assert 'torch' not in sys.modules"
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    run = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT / "tfpd_exploration/scripts/run_m2_postfusion_checkpoint_score_v1.py")], env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
