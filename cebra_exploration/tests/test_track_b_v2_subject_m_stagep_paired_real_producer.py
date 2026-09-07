"""No-target adversarial tests for the additive paired real producers."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_subject_m_stagep_paired_real_producer as producer  # noqa: E402
import track_b_v2_subject_m_stagep_runtime as runtime  # noqa: E402


CLI = ROOT / "cebra_exploration/scripts/run_track_b_v2_subject_m_stagep_paired_real_producer.py"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _admission(view: str, *, common: str = "same") -> dict:
    return {
        "status": "STAGEP_LIVE_ADMISSION_VALID__SEPARATE_ROOT_REVIEWED_EXECUTION_LAUNCH_REQUIRED",
        "cell": runtime.StagePCell.from_view(view).as_dict(),
        "target_path_resolution_permitted": False,
        "official_stagep_preflight_body_sha256": _sha(f"official-{view}"),
        "official_stagep_preflight_sidecar_sha256": _sha(f"official-side-{view}"),
        "root_authorization_pair": {"body_sha256": _sha(f"root-{common}"), "sidecar_sha256": _sha("rs")},
        "fixed_runtime_control_pair": {"body_sha256": _sha(f"control-{common}"), "sidecar_sha256": _sha("cs")},
        "fixed_d8it250_gpu_cost_gate": {"canonical_body_sha256": _sha(f"cost-{common}")},
        "implementation_closure_sha256": _sha(f"closure-{view}"),
    }


def _capability(view: str) -> producer.ViewExecutionCapability:
    return producer.ViewExecutionCapability(
        view=view, cell=runtime.StagePCell.from_view(view).as_dict(),
        admission_sha256=_sha(f"admission-{view}"),
        official_preflight_body_sha256=_sha(f"official-{view}"),
        implementation_closure_sha256=_sha(f"closure-{view}"),
        source_authority_set_sha256=_sha(f"source-{view}"),
        v9_preflight_sha256=producer.V9_PREFLIGHT_SHA256,
    )


def test_live_no_target_plan_binds_exact_source_and_v9_authorities() -> None:
    before = {name: name in sys.modules for name in ("numpy", "torch", "cebra", "pynwb")}
    plan = producer.build_no_target_review_plan()
    assert plan["status"] == producer.STATUS_REVIEW
    assert plan["pair_order"] == ["sua", "pseudo_mua"]
    assert len(plan["eight_producers"]) == 8
    assert plan["v9_post50_authority"]["sha256"] == producer.V9_PREFLIGHT_SHA256
    assert plan["v9_post50_authority"]["by_view"]["sua"] == {
        "target_float32_raw_sha256": producer.V9_TARGET_SHA256,
        "valid_starts_int64_raw_sha256": producer.V9_VALID_STARTS_SHA256,
        "query_window_count": 24708,
    }
    assert all(len(plan["source_authorities"][view]["body_bindings"]) == 6
               for view in producer.PAIR_ORDER)
    assert all(plan["source_authorities"][view]["historical_selector_plan_role"].endswith(
        "NOT_EXECUTABLE__NOT_SELECTED__NOT_AUTHORIZING") for view in producer.PAIR_ORDER)
    assert plan["target_opened"] is plan["source_opened"] is plan["gpu_used"] is False
    assert {name: name in sys.modules for name in before} == before


def test_sidecarless_v9_authority_rejects_sidecar_or_hash_drift(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copied = tmp_path / "preflight.json"
    copied.write_bytes(producer.V9_PREFLIGHT.read_bytes())
    copied.chmod(0o444)
    monkeypatch.setattr(producer, "V9_PREFLIGHT", copied)
    producer.load_v9_query_authority()
    Path(f"{copied}.sha256").write_text("poison")
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="sidecarless"):
        producer.load_v9_query_authority()
    Path(f"{copied}.sha256").unlink()
    copied.chmod(0o644)
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="mode 0444"):
        producer.load_v9_query_authority()


def test_capability_binding_rejects_pair_order_common_authority_and_plan_tamper(
        monkeypatch: pytest.MonkeyPatch) -> None:
    plan = producer.build_no_target_review_plan()
    admissions = {view: _admission(view) for view in producer.PAIR_ORDER}
    monkeypatch.setattr(producer.sealed_runtime, "build_stagep_live_admission",
                        lambda *, view: copy.deepcopy(admissions[view]))
    caps = producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)
    assert tuple(caps) == producer.PAIR_ORDER
    reversed_admissions = {"pseudo_mua": admissions["pseudo_mua"], "sua": admissions["sua"]}
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="ordered"):
        producer.bind_execution_capabilities(admissions=reversed_admissions, reviewed_plan=plan)
    fabricated = copy.deepcopy(admissions)
    fabricated["sua"]["official_stagep_preflight_body_sha256"] = _sha("fabricated")
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="freshly rebuilt"):
        producer.bind_execution_capabilities(admissions=fabricated, reviewed_plan=plan)
    drift = dict(admissions)
    drift["pseudo_mua"] = _admission("pseudo_mua", common="drift")
    monkeypatch.setattr(producer.sealed_runtime, "build_stagep_live_admission",
                        lambda *, view: copy.deepcopy(drift[view]))
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="share root"):
        producer.bind_execution_capabilities(admissions=drift, reviewed_plan=plan)
    monkeypatch.setattr(producer.sealed_runtime, "build_stagep_live_admission",
                        lambda *, view: copy.deepcopy(admissions[view]))
    poisoned_plan = copy.deepcopy(plan)
    poisoned_plan["fit_and_readout_contract"]["target_query_neural_or_auxiliary_in_any_fit"] = True
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="differs"):
        producer.bind_execution_capabilities(admissions=admissions, reviewed_plan=poisoned_plan)


def test_real_current_sealed_admissions_bind_exactly_without_target() -> None:
    admissions = {view: producer.sealed_runtime.build_stagep_live_admission(view=view)
                  for view in producer.PAIR_ORDER}
    capabilities = producer.bind_execution_capabilities(
        admissions=admissions, reviewed_plan=producer.build_no_target_review_plan())
    assert tuple(capabilities) == producer.PAIR_ORDER
    assert all(capabilities[view].admission_sha256 == producer._sha_json(admissions[view])
               for view in producer.PAIR_ORDER)


def test_sparse_query_block_is_exact_endpoint_gather_not_contiguous_crop() -> None:
    np = pytest.importorskip("numpy")
    capability = _capability("sua")
    suffix_start = 100
    support_stop = 95
    starts = np.array([101, 104, 110], dtype=np.int64)
    endpoints = starts + 49
    rows = 80
    neural = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
    behavior = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
    embedding = np.arange(rows * 8, dtype=np.float32).reshape(rows, 8)
    target = np.ascontiguousarray(behavior[endpoints - suffix_start], dtype=np.float32)
    block = producer.build_sparse_v9_query_block(
        capability=capability, suffix_neural=neural, suffix_behavior=behavior,
        suffix_embedding_full_length=embedding, suffix_start_raw=suffix_start,
        support_stop_exclusive=support_stop, reconstructed_valid_starts=starts,
        expected_target_sha256=producer._raw_array_sha(target),
        expected_valid_starts_sha256=producer._raw_array_sha(starts), expected_count=3,
    )
    assert block["receipt"]["not_contiguous_5_to_minus5_crop"] is True
    assert np.array_equal(block["embedding"], embedding[endpoints - suffix_start])
    assert np.array_equal(block["receptive_fields"][0], np.arange(endpoints[0] - 5, endpoints[0] + 5))
    assert block["receipt"]["query_neural_or_auxiliary_enters_any_fit"] is False


@pytest.mark.parametrize("mutation", ("start_hash", "count", "target_hash", "support_cross", "suffix_overrun"))
def test_sparse_query_block_adversarial_drift_fails(mutation: str) -> None:
    np = pytest.importorskip("numpy")
    capability = _capability("sua")
    suffix_start, support_stop, rows = 100, 95, 80
    starts = np.array([101, 104, 110], dtype=np.int64)
    neural = np.ones((rows, 4), dtype=np.float32)
    behavior = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
    embedding = np.ones((rows, 8), dtype=np.float32)
    expected_starts = producer._raw_array_sha(starts)
    target = behavior[starts + 49 - suffix_start]
    expected_target = producer._raw_array_sha(target)
    kwargs = dict(capability=capability, suffix_neural=neural, suffix_behavior=behavior,
                  suffix_embedding_full_length=embedding, suffix_start_raw=suffix_start,
                  support_stop_exclusive=support_stop, reconstructed_valid_starts=starts,
                  expected_target_sha256=expected_target,
                  expected_valid_starts_sha256=expected_starts, expected_count=3)
    if mutation == "start_hash": kwargs["expected_valid_starts_sha256"] = "0" * 64
    elif mutation == "count": kwargs["expected_count"] = 4
    elif mutation == "target_hash": kwargs["expected_target_sha256"] = "0" * 64
    elif mutation == "support_cross": kwargs["support_stop_exclusive"] = 103
    elif mutation == "suffix_overrun": kwargs["reconstructed_valid_starts"] = np.array([101, 104, 130])
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError):
        producer.build_sparse_v9_query_block(**kwargs)


def test_contiguous_fit_blocks_crop_each_block_independently() -> None:
    np = pytest.importorskip("numpy")
    capability = _capability("sua")
    first = producer.build_contiguous_fit_block(
        capability=capability, block_role="source_session", session_id="source-a",
        neural=np.ones((20, 3), np.float32), auxiliary=np.ones((20, 2), np.float32),
        embedding_full_length=np.arange(160, dtype=np.float32).reshape(20, 8), raw_start=0)
    second = producer.build_contiguous_fit_block(
        capability=capability, block_role="held_target_M50_support", session_id="target",
        neural=np.ones((15, 4), np.float32), auxiliary=np.ones((15, 2), np.float32),
        embedding_full_length=np.arange(120, dtype=np.float32).reshape(15, 8), raw_start=20)
    assert first["embedding"].shape == (10, 8)
    assert second["embedding"].shape == (5, 8)
    assert first["receipt"]["padded_edges_enter_fit"] is False
    assert second["receipt"]["raw_start_inclusive"] == 20


def test_joint_fit_wrapper_uses_sparse_v9_query_and_never_fits_query(
        monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")
    cap = _capability("sua")
    rows = tuple(SimpleNamespace(
        session_id=f"sub-C_ses-CO-{index:08d}",
        neural=np.ones((30, 3), np.float32) * (index + 1),
        dense_behavior=np.ones((30, 2), np.float32) * index,
    ) for index in range(27))
    source = producer.SourceProducerOutput(
        capability=cap, request={}, rows=rows,
        payload={"ordered_source_session_ids": [row.session_id for row in rows]})
    support_x = np.ones((20, 3), np.float32)
    support_y = np.ones((20, 2), np.float32)
    suffix_x = np.ones((80, 3), np.float32)
    suffix_y = np.arange(160, dtype=np.float32).reshape(80, 2)
    starts = np.array([101, 104, 110], np.int64)
    target = producer.TargetProducerOutput(
        capability=cap, neural_support=support_x, behavior_support=support_y,
        neural_suffix=suffix_x, behavior_suffix=suffix_y, valid_starts=starts,
        payload_inputs={"suffix_start_raw": 100, "support_stop_exclusive": 95})
    endpoint_target = suffix_y[starts + 49 - 100]
    monkeypatch.setattr(producer, "V9_QUERY_COUNT", 3)
    monkeypatch.setattr(producer, "V9_VALID_STARTS_SHA256", producer._raw_array_sha(starts))
    monkeypatch.setattr(producer, "V9_TARGET_SHA256", producer._raw_array_sha(endpoint_target))
    observed = {}

    def fake_fit(**kwargs):
        observed.update(kwargs)
        return {
            "estimator": object(),
            "source_embeddings": tuple(np.zeros((30, 8), np.float32) + index
                                       for index in range(27)),
            "held_support_embedding": np.zeros((20, 8), np.float32),
            "held_query_embedding": np.arange(640, dtype=np.float32).reshape(80, 8),
            "encoder_state_sha256": _sha("state"), "embedding_bundle_sha256": _sha("embed"),
            "fit_stream_count": 28, "fit_count": 1, "target_query_entered_fit": False,
            "fitted_offset": [5, 5],
        }

    monkeypatch.setattr(producer.sealed_runtime,
                        "_future_fit_primary_joint_encoder_after_all_live_gates", fake_fit)
    monkeypatch.setattr(producer, "verify_isolated_cuda_identity", lambda: {
        "logical_device": "cuda:0", "physical_index": 1, "CUDA_VISIBLE_DEVICES": "1"})
    output = producer.fit_one_joint_encoder(source=source, target=target)
    assert observed["held_query_neural"] is suffix_x
    assert output.fit_proof["target_query_enters_fit"] is False
    assert output.query_block["receipt"]["not_contiguous_5_to_minus5_crop"] is True
    assert output.query_block["embedding"].shape == (3, 8)
    assert len(output.source_blocks) == 27


def test_isolated_cuda_identity_requires_physical_gpu1_and_logical_cuda0(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 1
        @staticmethod
        def current_device(): return 0
        @staticmethod
        def get_device_properties(_index):
            return SimpleNamespace(name="Reviewed GPU", total_memory=24 * 1024**3)

    fake_torch = SimpleNamespace(
        __file__="/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/torch/__init__.py",
        __version__="2.5.1+cu121", version=SimpleNamespace(cuda="12.1"), cuda=FakeCuda())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    identity = producer._validate_isolated_cuda_identity(
        torch_module=fake_torch,
        nvidia_smi_line="1, GPU-abcdef, 00000000:02:00.0, Reviewed GPU, 24576, 535.54")
    assert identity["logical_device"] == "cuda:0" and identity["physical_index"] == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="physical index 1"):
        producer._validate_isolated_cuda_identity(
            torch_module=fake_torch,
            nvidia_smi_line="1, GPU-abcdef, 00000000:02:00.0, Reviewed GPU, 24576, 535.54")


def test_six_readouts_use_sparse_query_and_exact_torchmetrics151(
        monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")
    cap = _capability("sua")
    source_blocks = []
    for index in range(27):
        z = np.arange(48, dtype=np.float32).reshape(6, 8) + index
        y = np.arange(12, dtype=np.float32).reshape(6, 2) + index
        source_blocks.append({"embedding": z, "auxiliary": y, "receipt": {
            "semantics": "CONTIGUOUS_FIT_BLOCK__INDEPENDENT_OFFSET5_5_CROP"}})
    support = {"embedding": np.arange(64, dtype=np.float32).reshape(8, 8),
               "auxiliary": np.arange(16, dtype=np.float32).reshape(8, 2),
               "receipt": {"semantics": "CONTIGUOUS_FIT_BLOCK__INDEPENDENT_OFFSET5_5_CROP"}}
    target = np.array([[0, 0], [1, 1], [2, 3], [4, 8]], np.float32)
    query = {"embedding": np.arange(32, dtype=np.float32).reshape(4, 8), "auxiliary": target,
             "receipt": {"semantics":
                 "SPARSE_EXACT_V9_ENDPOINT_GATHER_FROM_CONTINUOUS_SUFFIX_TRANSFORM",
                 "ordered_target_behavior_float32_sha256": producer._raw_array_sha(target)}}
    encoder = producer.JointEncoderOutput(
        capability=cap, estimator=object(), source_blocks=source_blocks,
        support_block=support, query_block=query, persisted_embedding_arrays={},
        reload_probe_inputs={}, fit_proof={})
    monkeypatch.setattr(producer, "V9_QUERY_COUNT", 4)
    monkeypatch.setattr(producer, "V9_TARGET_SHA256", producer._raw_array_sha(target))
    target_payload = {"target_payload_sha256": _sha("target"), "query": query["receipt"]}
    encoder_payload = {"encoder_payload_sha256": _sha("encoder")}
    outputs = producer.score_all_six_readouts(
        encoder=encoder, target_payload=target_payload, encoder_payload=encoder_payload)
    assert set(outputs) == {f"{route}__{decoder}" for route in producer.ROUTES
                            for decoder in producer.DECODERS}
    assert all(value["metric"]["implementation"] == "torchmetrics.regression.R2Score"
               and value["metric"]["version"] == "1.5.1" for value in outputs.values())
    assert all(value["readout_proof"]["query_enters_fit"] is False for value in outputs.values())


def test_payload_builders_reject_query_in_fit_or_incomplete_six_score_topology() -> None:
    cap = _capability("sua")
    query = {"receipt": {
        "semantics": "SPARSE_EXACT_V9_ENDPOINT_GATHER_FROM_CONTINUOUS_SUFFIX_TRANSFORM",
        "query_row_count": 24708,
        "valid_starts_int64_sha256": producer.V9_VALID_STARTS_SHA256,
        "ordered_target_behavior_float32_sha256": producer.V9_TARGET_SHA256,
        "query_neural_or_auxiliary_enters_any_fit": True,
    }}
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="query block"):
        producer.build_target_payload(
            capability=cap, start_sha256=_sha("start"), asset={},
            support_receipt={"continuous_raw_prefix_start_inclusive": 0,
                             "through_rewarded_trial": 50,
                             "all_intervening_raw_rows_retained": True},
            query_block=query,
            private_snapshot={"parser_consumed_continuously_held_fd": True,
                              "pathname_reopen_permitted": False})
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="exact six"):
        producer.build_completion_payload(
            capability=cap, start_sha256=_sha("s"), target_sha256=_sha("t"),
            encoder_sha256=_sha("e"), score_sha256_by_role={})


def test_raw_pair_writer_is_0444_and_rolls_back_body_on_sidecar_conflict(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = tmp_path / "artifact.bin"
    binding = producer.publish_immutable_raw_pair(body, b"checkpoint")
    assert body.stat().st_mode & 0o777 == 0o444
    assert Path(f"{body}.sha256").stat().st_mode & 0o777 == 0o444
    assert binding["body_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    conflict = tmp_path / "conflict.bin"
    Path(f"{conflict}.sha256").write_text("poison")
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="fresh"):
        producer.publish_immutable_raw_pair(conflict, b"x")
    assert not conflict.exists()


def test_raw_pair_writer_rolls_back_only_owned_body_on_sidecar_creation_race(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = tmp_path / "raced.bin"
    original_open = producer.os.open
    injected = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal injected
        if path == "raced.bin.sha256" and not injected:
            injected = True
            fd = original_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                               dir_fd=kwargs["dir_fd"])
            os.write(fd, b"foreign")
            os.close(fd)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(producer.os, "open", racing_open)
    with pytest.raises(OSError):
        producer.publish_immutable_raw_pair(body, b"owned")
    assert not body.exists()
    assert Path(f"{body}.sha256").read_bytes() == b"foreign"


def test_raw_pair_writer_rejects_parent_directory_rename_swap(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "canonical"
    parent.mkdir()
    moved = tmp_path / "moved-original"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    body = parent / "artifact.bin"
    original_open = producer.os.open
    swapped = False

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "artifact.bin.sha256" and not swapped:
            swapped = True
            os.rename(parent, moved)
            os.rename(replacement, parent)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(producer.os, "open", swapping_open)
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="parent identity"):
        producer.publish_immutable_raw_pair(body, b"owned")
    assert not (moved / "artifact.bin").exists()
    assert not (moved / "artifact.bin.sha256").exists()
    assert not body.exists() and not Path(f"{body}.sha256").exists()


def test_target_private_snapshot_parser_and_pmua_replay_capability_are_exact(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")
    cap = _capability("pseudo_mua")
    source = producer.SourceProducerOutput(
        capability=cap, request={}, rows=(),
        payload={"schema": producer.SCHEMA_SOURCE, "source_authority_exact_rebuild_match": True})
    raw = b"synthetic-held-fd-target-bytes"
    asset_path = tmp_path / "target.nwb"
    asset_path.write_bytes(raw)
    snapshot = tmp_path / "cell/private/held.nwb"
    monkeypatch.setattr(producer, "_topology", lambda _view: {"private_snapshot": str(snapshot)})
    fake_mean = np.array([0.0, 0.0], np.float32)
    fake_std = np.array([1.0, 1.0], np.float32)
    monkeypatch.setattr(producer, "load_source_authority_bundle", lambda _view: ({}, {
        "source_behavior_normalizer": {
            "mean_float32": fake_mean.tolist(), "std_float32": fake_std.tolist(),
            "mean_array_sha256": producer._raw_array_sha(fake_mean),
            "std_array_sha256": producer._raw_array_sha(fake_std),
            "fit_scope": "strict27_subc_co_train_only"}}))
    monkeypatch.setattr(producer.target_materializer,
                        "build_development_target_materializer_dry_plan", lambda **_kwargs: {
        "status": "CANONICAL_DEVELOPMENT_AUTHORITY_AND_SUBM_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS",
        "target_session_id": producer.TARGET_SESSION_ID, "view": "pseudo_mua",
        "target_asset_ledger_gate": {"target_asset": {
            "session_id": producer.TARGET_SESSION_ID,
            "a2_official_local_nwb_path": str(asset_path),
            "expected_sha256": hashlib.sha256(raw).hexdigest(), "expected_bytes": len(raw)}}})
    rows = 100
    sua = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
    pmua = np.ascontiguousarray(sua[:, [0, 2]], dtype=np.float32)
    behavior = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
    trials = [{"start": 0, "stop": 60} for _ in range(51)]
    trials[49] = {"start": 0, "stop": 20}
    trials[50] = {"start": 20, "stop": 80}
    starts = np.array([21, 23, 25], dtype=np.int64)

    class FakeIO:
        def __init__(self, fd_path, _mode):
            assert Path(fd_path).read_bytes() == raw
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self):
            return SimpleNamespace(units=SimpleNamespace(to_dataframe=lambda: object()))

    class FakeLoader:
        NWBHDF5IO = FakeIO
        @staticmethod
        def load_session_with_trials(fd_path, *_args, signal_view, **kwargs):
            assert Path(fd_path).read_bytes() == raw
            assert kwargs["cache_dir"] is None
            return {"signal_view": signal_view, "neural": pmua if signal_view == "pseudo_mua" else sua,
                    "behavior": behavior, "trials": trials}

    class FakeMulti:
        @staticmethod
        def _compute_valid_starts(_trials, _window): return starts
        @staticmethod
        def electrode_ids_from_units(_units): return np.array([0, 1, 2, 3], np.int64)
        @staticmethod
        def pool_spikes_by_electrode(values, _ids):
            return np.ascontiguousarray(values[:, [0, 2]], np.float32), np.array([0, 2], np.int64)

    monkeypatch.setattr(producer, "_canonical_subject_m_modules", lambda: (FakeLoader, FakeMulti))
    monkeypatch.setattr(producer, "V9_QUERY_COUNT", 3)
    monkeypatch.setattr(producer, "V9_VALID_STARTS_SHA256", producer._raw_array_sha(starts))
    monkeypatch.setattr(producer, "V9_TARGET_SHA256",
                        producer._raw_array_sha(behavior[starts + 49]))
    output = producer.materialize_target_from_private_snapshot(cap, source)
    assert np.array_equal(output.neural_support, pmua[:20])
    assert np.array_equal(output.neural_suffix, pmua[20:])
    assert output.payload_inputs["pmua_replay"]["replay_exact_equal"] is True
    assert output.payload_inputs["private_snapshot"]["parser_consumed_continuously_held_fd"] is True
    assert snapshot.stat().st_mode & 0o777 == 0o444


@pytest.mark.parametrize("view", producer.PAIR_ORDER)
def test_real_source_only_strict27_rebuild_exact(view: str) -> None:
    """Integration evidence: opens strict27 source only, never the target."""
    output = producer.materialize_and_verify_strict27_source(_capability(view))
    assert len(output.rows) == 27
    assert output.payload["source_authority_exact_rebuild_match"] is True
    assert output.payload["target_opened"] is False


def test_real_vendored_cebra061_sklearn_same_fd_state_and_transform_reload(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")
    vendor = ROOT / "cebra_exploration/third_party/cebra"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    import cebra
    assert cebra.__version__ == "0.6.1"
    rng = np.random.default_rng(42)
    xs = [rng.normal(size=(80, 3)).astype(np.float64) for _ in range(2)]
    ys = [rng.normal(size=(80, 2)).astype(np.float64) for _ in range(2)]
    model = cebra.CEBRA(model_architecture="offset10-model", device="cpu", batch_size=16,
                        learning_rate=3e-4, output_dimension=8, num_hidden_units=8,
                        max_iterations=2, verbose=False)
    model.fit(xs, ys)
    arrays = {f"session_{i}": np.asarray(model.transform(x, session_id=i), np.float32)
              for i, x in enumerate(xs)}
    cell = tmp_path / "cell"
    cell.mkdir()
    monkeypatch.setattr(producer, "_topology", lambda _view: {
        "checkpoint": str(cell / "model.pt"), "embeddings": str(cell / "embeddings.npz")})
    result = producer.persist_sklearn_checkpoint_and_embeddings(
        capability=_capability("sua"), estimator=model, embedding_arrays=arrays,
        reload_probe_inputs={f"session_{i}": (x, i) for i, x in enumerate(xs)},
        cebra_loader=cebra.CEBRA.load)
    assert result["checkpoint"]["solver_state_sha256_before_save"] == \
        result["checkpoint"]["solver_state_sha256_after_load"]
    assert result["checkpoint"]["all_transform_probes_exact_equal"] is True
    assert result["embeddings"]["same_fd_reload_exact"] is True


def test_addendum_publication_rolls_back_exact_owned_pair_on_final_closure_drift(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "root-only/execution_addendum.json"
    monkeypatch.setattr(producer, "ADDENDUM_PATH", path)
    calls = 0

    def drifting_candidate(*, admissions):
        nonlocal calls
        calls += 1
        return {"schema": producer.SCHEMA_EXECUTION_ADDENDUM,
                "canonical_path": str(path), "generation": calls}

    monkeypatch.setattr(producer, "build_execution_addendum_candidate", drifting_candidate)
    with pytest.raises(producer.TrackBV2SubjectMRealProducerError, match="launch/final"):
        producer.publish_execution_addendum(admissions={}, i_have_independent_root_review=True)
    assert not path.exists() and not Path(f"{path}.sha256").exists()


def test_root_reviewed_addendum_temp_publication_and_exact_reload(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "root-only/execution_addendum.json"
    monkeypatch.setattr(producer, "ADDENDUM_PATH", path)
    admissions = {view: producer.sealed_runtime.build_stagep_live_admission(view=view)
                  for view in producer.PAIR_ORDER}
    binding = producer.publish_execution_addendum(
        admissions=admissions, i_have_independent_root_review=True)
    loaded = producer.load_execution_addendum(admissions=admissions)
    assert path.stat().st_mode & 0o777 == 0o444
    assert Path(f"{path}.sha256").stat().st_mode & 0o777 == 0o444
    assert binding["body_sha256"] == loaded["body_sha256"]
    assert loaded["payload"]["authorizes_target_or_GPU_by_itself"] is False


def test_cli_default_is_no_write_no_ml_and_dual_flag_execute_stops_pre_target() -> None:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(SRC)
    completed = subprocess.run([sys.executable, str(CLI)], cwd=ROOT, env=env,
                               text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["status"] == producer.STATUS_REVIEW
    assert plan["receipt_minted"] is plan["target_opened"] is plan["gpu_used"] is False
    for single in ("--execute", "--i-have-independent-root-review"):
        rejected = subprocess.run([sys.executable, str(CLI), single], cwd=ROOT, env=env,
                                  text=True, capture_output=True, check=False)
        assert rejected.returncode != 0 and "requires both" in rejected.stderr
    refused = subprocess.run([sys.executable, str(CLI), "--execute",
                              "--i-have-independent-root-review"], cwd=ROOT, env=env,
                             text=True, capture_output=True, check=False)
    assert refused.returncode != 0
    assert "disabled pending independent review" in refused.stderr


def test_fresh_import_does_not_import_torch_cebra_or_target_parser_runtime() -> None:
    probe = """
import sys
before={n:(n in sys.modules) for n in ('torch','cebra','pynwb')}
import track_b_v2_subject_m_stagep_paired_real_producer
after={n:(n in sys.modules) for n in before}
assert before == after, (before,after)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    done = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env,
                          text=True, capture_output=True, check=False)
    assert done.returncode == 0, done.stderr
