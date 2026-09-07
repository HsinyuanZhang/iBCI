"""No-data, no-CUDA adversarial tests for the Phase-E matched-score scaffold."""
from __future__ import annotations

from contextlib import contextmanager
import json
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration"))
from src.tfsr_b3st4_ddrop_v1 import score


def _sha(label: str | bytes) -> str:
    return score._sha(label if isinstance(label, bytes) else label.encode("ascii"))


# Keep this independent literal rather than deriving it from the scorer.  It
# is the paired-view C1 val[6] table that durable Phase-E preflight must bind.
WITHIN = (
    "sub-C_ses-CO-20151103", "sub-C_ses-CO-20151104", "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109", "sub-C_ses-CO-20151110", "sub-C_ses-CO-20151112",
)
EXTERNAL = tuple(f"external-{index:02d}" for index in range(15))

# This is intentionally an independent literal, rather than a value derived
# from the scorer.  It catches the v2 receipt's misleading
# ``eligible_session_ids`` UUID field being accidentally treated as the
# session roster again.
REAL_EXTERNAL_SUBM_15 = (
    "sub-M_ses-CO-20140307", "sub-M_ses-CO-20140626", "sub-M_ses-CO-20140627",
    "sub-M_ses-CO-20141203", "sub-M_ses-CO-20150511", "sub-M_ses-CO-20150512",
    "sub-M_ses-CO-20150610", "sub-M_ses-CO-20150611", "sub-M_ses-CO-20150612",
    "sub-M_ses-CO-20150615", "sub-M_ses-CO-20150616", "sub-M_ses-CO-20150617",
    "sub-M_ses-CO-20150623", "sub-M_ses-CO-20150625", "sub-M_ses-CO-20150626",
)

# Independent literals from the sealed strict-27 A2 training-path cache.  Do
# not derive these from score.py: this asserts that the scorer's closure-bound
# behavior normalizer remains tied to the reviewed float32 source values.
SEALED_SOURCE_BEHAVIOR_MEAN = (-0.001148765324614942, 0.002653369214385748)
SEALED_SOURCE_BEHAVIOR_STD = (8.63547420501709, 8.086690902709961)

# Independent literals from the sealed Cell-D terminal.  Keep these in the
# tests rather than deriving them from score.py so resource disclosure cannot
# silently follow a drifted implementation constant.
SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS = 3_510_842
SEALED_CELL_D_LAZY_KEYS = (
    "decoder.fc_id_in.0.weight",
    "decoder.fc_id_in.0.bias",
)
SEALED_CELL_D_LAZY_ROLE = "dead_decoder_fc_id_in_identity_path"

# Independent literals from the revised Phase-E device contract.  The two
# memory values intentionally have different units and must never be derived
# from one another by flooring or rounding.
SEALED_SCORE_DEVICE = {
    "cuda_visible_devices": "1",
    "internal_device": "cuda:0",
    "torch_version": "2.5.1.post303",
    "uuid": "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
    "bdf": "00000000:03:00.0",
    "name": "NVIDIA GeForce RTX 3090",
    "nvidia_smi_memory_total_mib": 24576,
    "torch_total_memory_bytes": 25438126080,
}

# Independent literal for the exact eager package chain reached by
# ``validate_phase_d_training_terminal -> train.lr_for_step``.  Keeping this
# outside the scorer prevents the test from silently accepting a future
# closure edit that drops an initializer-only dependency.
TFPD_LANE_LR_IMPORT_CLOSURE = (
    "tfpd_exploration/src/tfpd_lane/__init__.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/mech_diag.py",
    "tfpd_exploration/src/tfpd_lane/pregate.py",
    "tfpd_exploration/src/tfpd_lane/receipt.py",
)


def _bindings(tmp_path: Path) -> tuple[score.ExternalAssetBinding, ...]:
    return tuple(score.ExternalAssetBinding(
        asset_id=f"asset-{index:02d}", session=name,
        local_path=tmp_path / f"{name}.nwb", expected_bytes=7,
        expected_sha256=_sha(f"asset:{name}"), frozen_path=f"sub-M/{name}.nwb",
    ) for index, name in enumerate(EXTERNAL))


def _within_bindings(tmp_path: Path) -> tuple[score.EvaluationAssetBinding, ...]:
    return tuple(score.EvaluationAssetBinding(
        surface="within", asset_id=str(row["asset_id"]), session=str(row["session"]),
        local_path=tmp_path / Path(str(row["frozen_path"])).name,
        expected_bytes=int(row["bytes"]), expected_sha256=str(row["sha256"]),
        frozen_path=str(row["frozen_path"]),
    ) for row in score.sealed_within_assets())


def _identity() -> score.ScoreIdentity:
    closure = {"paths": ["synthetic"], "sha256_by_path": {"synthetic": _sha("code")},
               "closure_sha256": _sha("closure")}
    authorization = score.PhaseEAuthorization(
        preflight_sha256=_sha("pre"), root_authorization_sha256=_sha("auth"),
        preflight={"synthetic": "preflight"}, root_authorization={"synthetic": "authorization"},
        expected_closure=closure,
    )
    return score.ScoreIdentity(
        fixed_authorities={"synthetic": {"body_sha256": _sha("fixed")}},
        training_terminal_sha256=_sha("terminal"), training_swa_sha256=_sha("swa"),
        training_swa_state_digest=_sha("swa-state"),
        launch_closure=closure,
        phase_e_authorization=authorization.payload(),
    )


def _phase_e_closure() -> dict[str, object]:
    by_path = {path: _sha(f"closure:{path}") for path in score.PHASE_E_CLOSURE}
    immutable_input_bindings = {
        "within_paired_view_manifest": dict(score.WITHIN_PAIRED_VIEW_CLOSURE_BINDING),
    }
    aggregate = _sha(json.dumps(
        {"files": by_path, "immutable_input_bindings": immutable_input_bindings},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))
    return {
        "paths": list(score.PHASE_E_CLOSURE),
        "sha256_by_path": by_path,
        "immutable_input_bindings": immutable_input_bindings,
        "closure_sha256": aggregate,
    }


def _training_evidence() -> score.TrainingEvidence:
    return score.TrainingEvidence(
        terminal_sha256=_sha("terminal"), swa_sha256=_sha("swa"), swa_state_digest=_sha("swa-state"),
        closure={"paths": ["phase-d"], "sha256_by_path": {"phase-d": _sha("phase-d")},
                 "closure_sha256": _sha("phase-d-closure")},
        checkpoint_sha256={str(epoch): _sha(f"checkpoint:{epoch}") for epoch in (44, 45, 46, 47)},
        training_peak_memory_bytes=123,
    )


def _fixed_bindings() -> dict[str, dict[str, object]]:
    return {
        item.name: {"relative_path": item.relative, "body_sha256": item.sha256,
                    "mode": format(item.mode, "04o"), "descriptor_identity": [1, 2, 3]}
        for item in score.FIXED_AUTHORITIES
    }


def _normalizers() -> dict[str, object]:
    return {
        "source_normalizer_body_sha256": score.SOURCE_NORMALIZER_BODY_SHA,
        "t4_semantic_sha256": score.T4_NORMALIZER_SEMANTIC_SHA,
        "behavior_semantic_sha256": score.BEHAVIOR_NORMALIZER_SEMANTIC_SHA,
        "t4_mean": list(score.SOURCE_T4_MEAN), "t4_std": list(score.SOURCE_T4_STD),
        "behavior_mean": list(SEALED_SOURCE_BEHAVIOR_MEAN),
        "behavior_std": list(SEALED_SOURCE_BEHAVIOR_STD),
    }


def _sealed_cell_d_terminal() -> dict[str, object]:
    """Small exact-schema authority slice for no-data Cell-D tests."""
    return {
        "cell": "D",
        "initial_state": {
            "bitwise_equality_proof": {
                "D": {
                    "trainable_parameters": SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
                    "strict_load": True,
                    "state_keys_equal_to_canonical": True,
                    "num_heads": 2,
                    "dynamic_dropout": True,
                },
            },
        },
        "swa": {"manifest": {"uninitialized_lazy_tensor_count": len(SEALED_CELL_D_LAZY_KEYS)}},
    }


def _cell_d_parameter_accounting() -> dict[str, object]:
    return {
        "schema": "tfsr_phase_e_cell_d_lazy_safe_parameter_accounting_v1",
        "count_semantics": "initialized_trainable_parameters_only",
        "initialized_trainable_parameters": SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
        "sealed_initialized_trainable_parameters": SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
        "uninitialized_lazy_parameter_count": len(SEALED_CELL_D_LAZY_KEYS),
        "sealed_uninitialized_lazy_parameter_count": len(SEALED_CELL_D_LAZY_KEYS),
        "uninitialized_lazy_parameter_keys": list(SEALED_CELL_D_LAZY_KEYS),
        "uninitialized_lazy_parameter_role": SEALED_CELL_D_LAZY_ROLE,
    }


def _preflight() -> tuple[score.TrainingEvidence, dict[str, dict[str, object]], dict[str, object]]:
    training = _training_evidence()
    fixed = _fixed_bindings()
    preflight = score.build_target_free_preflight(
        root=ROOT, training=training, fixed_authorities=fixed, closure=_phase_e_closure(), within_roster=WITHIN,
        normalizers=_normalizers(),
    )
    return training, fixed, preflight


def _execution_capability(identity: score.ScoreIdentity) -> score.ExecutionCapability:
    payload = identity.phase_e_authorization
    authorization = score.PhaseEAuthorization(
        preflight_sha256=payload["preflight_sha256"],
        root_authorization_sha256=payload["root_authorization_sha256"],
        preflight={"synthetic": "preflight"}, root_authorization={"synthetic": "authorization"},
        expected_closure=payload["expected_closure"],
    )
    return score._issue_execution_capability(authorization)


def _d_value(surface: str, index: int) -> float:
    return (0.30 if surface == "external" else 0.50) + index / 1000.0


def _sealed_table() -> dict[str, object]:
    def block(surface: str, names: tuple[str, ...]) -> dict[str, object]:
        rows = [{"session": name, "n_windows": 100 + index, "r2": _d_value(surface, index)}
                for index, name in enumerate(names)]
        return {"per_session": rows, "mean_r2": sum(item["r2"] for item in rows) / len(rows)}
    return {"results": {"D_swa": {"within": block("within", WITHIN), "external": block("external", EXTERNAL)}}}


def _session_score(surface: str, name: str, index: int, value: float, input_sha: str) -> score.SessionScore:
    return score.SessionScore(
        session=name, n_windows=100 + index, governing_r2=value, full_window_r2=value - 0.01,
        output_sha256=_sha(f"output:{surface}:{name}:{value}"), input_authority_sha256=input_sha,
    )


def _mode(system: str, surface: str, mode: str, input_sha: str, *, offset: float = 0.04) -> score.ModeEvidence:
    names = WITHIN if surface == "within" else EXTERNAL
    scores = tuple(_session_score(surface, name, index,
                                  _d_value(surface, index) + (0.0 if system == "cell_d" else offset), input_sha)
                   for index, name in enumerate(names))
    common: dict[str, Any] = {
        "system": system, "surface": surface, "mode": mode, "sessions": scores,
        "state_before_sha256": _sha(f"state:{system}:{surface}:{mode}"),
        "state_after_sha256": _sha(f"state:{system}:{surface}:{mode}"),
        "eval_mode": True, "gradients_none": True, "finite_output": True,
        "output_shape": (2, 50, 2), "latency_ms": 1.0, "peak_memory_bytes": 1,
    }
    if system == "tfsr":
        control: dict[str, object] = {"mode": mode, "post_normalization": True, "b3s_recomputed": True}
        if mode == "zero":
            control["exact_zeros_like"] = True
        if mode == "wrong_pair":
            control.update({"permutation_is_derangement": True, "permutation_sha256": _sha("permutation")})
        common.update({"repeat_bitwise_equal": mode == "aligned", "eval_no_mask": True,
                       "capture_diagnostics": False, "b3s_recomputed": True, "t4_control": control})
    return score.ModeEvidence(**common)


def _input_evidence(
    within: tuple[score.EvaluationAssetBinding, ...],
    bindings: tuple[score.ExternalAssetBinding, ...],
) -> score.InputAuthorityEvidence:
    records: list[score.SessionInputAuthority] = []
    tensor = {"dtype": "torch.float32", "shape": [3, 4], "bytes_sha256": _sha("tensor")}
    def record_for(*, session: str, surface: str, asset: Mapping[str, object]) -> score.SessionInputAuthority:
        raw_proof = {
            "schema": "tfsr_phase_e_raw_t4_sua_axis_proof_v1",
            "session": session,
            "signal_view": "sua",
            "channel_ids_dtype": "int64",
            "channel_ids_sha256": _sha(f"channel-ids:{surface}:{session}"),
            "source_unit_count": 3,
            "raw_t4_shape": [3, 4],
            "feature_group": "t4",
            "pool_size": 30,
            "mapping": "raw_t4_row_k_equals_sua_neural_column_k_for_contiguous_channel_ids",
            "closure_bound_functions": [
                "mc_maze.multisession_datamodule.load_dandi688_session",
                "mc_maze.unit_side_features.compute_unit_side_features_uncached",
            ],
        }
        return score.SessionInputAuthority(
            session=session,
            surface=surface,
            asset=asset,
            ordered_unit_digest=_sha("units"),
            unit_count=3,
            raw_t4=tensor,
            normalized_t4=tensor,
            calibration_sha256=_sha("cal"),
            query_start_sha256=_sha("start"),
            neural_sha256=_sha("neural"),
            behavior_sha256=_sha("beh"),
            valid_mask_sha256=_sha("mask"),
            target_last_bin_sha256=_sha(f"target:{surface}:{session}"),
            last_bin_valid_mask_sha256=_sha(f"last-mask:{surface}:{session}"),
            last_bin_valid_count=100,
            n_windows=100,
            raw_t4_sua_axis_proof=raw_proof,
            raw_t4_sua_axis_proof_sha256=score._sha(score._json_bytes(raw_proof)),
        )
    for binding in within:
        records.append(record_for(session=binding.session, surface="within", asset=binding.payload()))
    for binding in bindings:
        # Runtime validation speaks the single cross-surface target-asset
        # schema, so the external ledger row is explicitly promoted before it
        # appears in evidence (rather than relying on the old narrower
        # ExternalAssetBinding payload).
        external = score._as_evaluation_asset(binding)
        records.append(record_for(session=external.session, surface="external", asset=external.payload()))
    return score.InputAuthorityEvidence(
        records=tuple(sorted(records, key=lambda item: (item.surface, item.session))),
        source_normalizer_body_sha256=score.SOURCE_NORMALIZER_BODY_SHA,
        t4_normalizer_semantic_sha256=score.T4_NORMALIZER_SEMANTIC_SHA,
        behavior_normalizer_semantic_sha256=score.BEHAVIOR_NORMALIZER_SEMANTIC_SHA,
        strict_manifest_sha256=next(item.sha256 for item in score.FIXED_AUTHORITIES if item.name == "strict_manifest"),
        raw_to_normalized_exact=True, no_cache_readonly_adapter=True,
    )


class MockBackend:
    def __init__(self, within: tuple[score.EvaluationAssetBinding, ...], bindings: tuple[score.ExternalAssetBinding, ...], *, failure: str | None = None,
                 d_mismatch: bool = False) -> None:
        self.within, self.bindings, self.failure, self.d_mismatch = within, bindings, failure, d_mismatch
        self.closed = False
        self.calls: list[str] = []

    def resolve_inputs(self, *, flags: score.ScoreFlags, **_: Any) -> score.InputAuthorityEvidence:
        self.calls.append("resolve")
        flags.within_opened = flags.external_opened = True
        if self.failure == "resolve":
            raise RuntimeError("synthetic resolve failure")
        return _input_evidence(self.within, self.bindings)

    def score_cell_d(self, *, surface: str, input_authority_sha256: str, flags: score.ScoreFlags) -> score.ModeEvidence:
        self.calls.append(f"d:{surface}")
        flags.record_forward("cell_d", surface, "aligned")
        if self.failure == f"d:{surface}":
            raise RuntimeError("synthetic Cell-D failure")
        item = _mode("cell_d", surface, "aligned", input_authority_sha256)
        if self.d_mismatch and surface == "external":
            changed = list(item.sessions)
            first = changed[0]
            changed[0] = score.SessionScore(first.session, first.n_windows, first.governing_r2 + 0.1,
                                              first.full_window_r2, first.output_sha256, first.input_authority_sha256)
            item = score.ModeEvidence(**{**item.__dict__, "sessions": tuple(changed)})
        return item

    def score_tfsr(self, *, surface: str, mode: str, input_authority_sha256: str, flags: score.ScoreFlags) -> score.ModeEvidence:
        self.calls.append(f"tf:{surface}:{mode}")
        flags.record_forward("tfsr", surface, mode)
        if self.failure == f"tf:{surface}:{mode}":
            raise RuntimeError("synthetic TF-SR failure")
        offset = 0.04 if mode == "aligned" else (-0.03 if mode == "zero" else -0.06)
        return _mode("tfsr", surface, mode, input_authority_sha256, offset=offset)

    def reverify_after_forwards(self, *, flags: score.ScoreFlags) -> None:
        self.calls.append("reverify")
        if self.failure == "reverify":
            raise RuntimeError("synthetic reverify failure")

    def resource_disclosure(self) -> dict[str, object]:
        self.calls.append("resource")
        if self.failure == "resource":
            raise RuntimeError("synthetic resource failure")
        return {
                "cell_d_parameters": SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
                "cell_d_parameter_accounting": _cell_d_parameter_accounting(),
                "tfsr_parameters": 10, "tfsr_analytic_macs": 20,
                "persistent_state": {"per_window": "zero"}, "training_peak_memory_bytes": 1,
                "score_peak_memory_bytes": 1, "aligned_latency_ms": 1.0,
                "runtime": {**score.FROZEN_SCORE_DEVICE, "torchmetrics_version": "1.5.1"}}

    def close(self) -> None:
        self.closed = True


def _artifact(tmp_path: Path) -> score.ArtifactRoot:
    return score.reserve_artifact_root(tmp_path, "score")


def _run(tmp_path: Path, *, failure: str | None = None, d_mismatch: bool = False,
         final_drift: bool = False):
    bindings = _bindings(tmp_path)
    within = _within_bindings(tmp_path)
    backend = MockBackend(within, bindings, failure=failure, d_mismatch=d_mismatch)
    identity = _identity()
    artifact = _artifact(tmp_path)
    closure = identity.launch_closure if not final_drift else {"closure_sha256": _sha("drift")}
    return artifact, backend, lambda: score.run_score_lifecycle(
        artifact=artifact, identity=identity, execution_capability=_execution_capability(identity),
        within_roster=WITHIN, external_sessions=EXTERNAL,
        within_roster_factory=lambda: within,
        external_roster_factory=lambda: bindings, sealed_cell_d_table=_sealed_table(),
        a2_pooled={"within": {name: 0.4 for name in WITHIN}, "external": {name: 0.2 for name in EXTERNAL}},
        backend=backend, final_reverify=lambda: closure,
    )


def _mode_bits(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_zero_arg_dry_cli_imports_no_torch_or_data_and_touches_no_canonical_score_root(tmp_path: Path):
    (tmp_path / "torch.py").write_text("raise RuntimeError('TORCH_IMPORTED')\n")
    script = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_score.py"
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(tmp_path), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    done = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    plan = json.loads(done.stdout)
    assert plan["authorization"] == "none"
    assert "TORCH_IMPORTED" not in done.stdout + done.stderr
    assert not (ROOT / score.SCORE_ROOT_RELATIVE).exists()


@pytest.mark.parametrize("arguments", (("--execute",), ("--i-have-phase-e-root-authorization",), ("--bad",),
                                        ("--execute", "--execute")))
def test_partial_or_duplicate_execution_flags_fail_before_torch(arguments: tuple[str, ...], tmp_path: Path):
    (tmp_path / "torch.py").write_text("raise RuntimeError('TORCH_IMPORTED')\n")
    script = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_score.py"
    env = os.environ.copy(); env.update({"PYTHONPATH": str(tmp_path), "PYTHONNOUSERSITE": "1"})
    done = subprocess.run([sys.executable, str(script), *arguments], cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode != 0
    assert "TORCH_IMPORTED" not in done.stdout + done.stderr


def test_artifact_publication_reload_modes_collision_and_named_root_identity(tmp_path: Path):
    artifact = _artifact(tmp_path)
    digest = artifact.publish_json("attempt.json", {"ok": True})
    assert artifact.reload_json("attempt.json", digest) == {"ok": True}
    assert _mode_bits(artifact.directory / "attempt.json") == 0o444
    assert _mode_bits(artifact.directory / "attempt.json.sha256") == 0o444
    with pytest.raises(score.FailClosedError):
        artifact.publish_json("attempt.json", {"other": True})
    moved = tmp_path / "moved"
    artifact.directory.rename(moved)
    artifact.directory.mkdir()
    with pytest.raises(score.FailClosedError, match="identity"):
        artifact.reload_pair("attempt.json")


@pytest.mark.parametrize("leaf", ("attempt.json", "attempt.json.sha256"))
def test_output_partial_collision_is_rejected_before_factory(leaf: str, tmp_path: Path):
    target = tmp_path / "score"; target.mkdir()
    (target / leaf).write_text("partial")
    with pytest.raises(score.FailClosedError, match="fresh canonical score root"):
        score.reserve_artifact_root(tmp_path, "score")


def test_complete_mock_lifecycle_publishes_exact_pairs_and_clear_go(tmp_path: Path):
    artifact, backend, run = _run(tmp_path)
    terminal = run()
    assert terminal["verdict"] == "CLEAR_GO"
    assert backend.closed is True
    expected = set(score.SCORE_TOPOLOGY) - {"failure.json"}
    assert {item.name for item in artifact.directory.iterdir() if not item.name.endswith(".sha256")} == expected
    for name in expected:
        assert _mode_bits(artifact.directory / name) == 0o444
        artifact.reload_pair(name)
    receipt = artifact.reload_json("score.json")
    assert receipt["a2_contextual"]["non_gating"] is True
    assert receipt["controls"]["external"]["aligned_minus_zero"]["non_rescuing"] is True
    assert receipt["resources"]["runtime"] == {**SEALED_SCORE_DEVICE, "torchmetrics_version": "1.5.1"}
    assert backend.calls.index("resolve") < backend.calls.index("d:within")


def test_reloaded_score_and_terminal_reject_nested_semantic_or_cross_sha_tampering(tmp_path: Path):
    artifact, _backend, run = _run(tmp_path)
    terminal = run()
    identity = _identity()
    score_payload = artifact.reload_json("score.json", terminal["score_sha256"])
    score.validate_score_payload(
        score_payload, identity=identity, input_authority_sha256=terminal["input_authority_sha256"],
    )
    forged_score = json.loads(json.dumps(score_payload))
    forged_score["tfsr"]["external"]["aligned"]["sessions"][0]["governing_r2"] += 0.01
    with pytest.raises(score.FailClosedError, match="mode evidence|paired contrast"):
        score.validate_score_payload(
            forged_score, identity=identity, input_authority_sha256=terminal["input_authority_sha256"],
        )
    forged_terminal = dict(terminal)
    forged_terminal["score_sha256"] = _sha("wrong-score")
    with pytest.raises(score.FailClosedError, match="exact reloaded score SHA"):
        score.validate_terminal_payload(
            forged_terminal, identity, score_payload, expected_score_sha=terminal["score_sha256"],
        )


@pytest.mark.parametrize("failure", ("resolve", "d:within", "d:external", "tf:within:aligned", "tf:within:zero",
                                      "tf:within:wrong_pair", "tf:external:aligned", "tf:external:zero",
                                      "tf:external:wrong_pair", "reverify", "resource"))
def test_injected_failures_after_attempt_publish_honest_failure_only(failure: str, tmp_path: Path):
    artifact, backend, run = _run(tmp_path, failure=failure)
    with pytest.raises(RuntimeError, match="synthetic"):
        run()
    payload = artifact.reload_json("failure.json")
    score.validate_failure_payload(payload)
    assert payload["terminal_published"] is False
    assert not (artifact.directory / "terminal.json").exists()
    assert backend.closed is True


def test_cell_d_exact_replay_mismatch_blocks_all_tfsr_interpretation(tmp_path: Path):
    artifact, backend, run = _run(tmp_path, d_mismatch=True)
    with pytest.raises(score.FailClosedError, match="Cell-D exact"):
        run()
    assert "tf:within:aligned" not in backend.calls
    assert artifact.reload_json("failure.json")["stage"] == "cell_d_external"


def test_launch_final_closure_drift_fails_after_score_without_terminal(tmp_path: Path):
    artifact, backend, run = _run(tmp_path, final_drift=True)
    with pytest.raises(score.FailClosedError, match="launch/final closure"):
        run()
    # score/terminal are one transactional publication group; a closure drift
    # before terminal construction must leave neither scientific leaf.
    assert not (artifact.directory / "score.json").exists()
    assert not (artifact.directory / "terminal.json").exists()
    assert artifact.reload_json("failure.json")["stage"] == "terminal_revalidation"
    assert backend.closed is True


@pytest.mark.parametrize("group_write", (5, 6, 7, 8), ids=("score_body", "score_sidecar", "terminal_body", "terminal_sidecar"))
def test_lifecycle_group_stream_failures_leave_only_honest_failure(
    group_write: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Attempt/input account for four writes; the scientific group starts at 5."""
    artifact, backend, run = _run(tmp_path)
    original = score._write_full
    calls = 0

    def failing_write(descriptor: int, body: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == group_write:
            raise OSError("synthetic lifecycle group stream failure")
        original(descriptor, body)

    monkeypatch.setattr(score, "_write_full", failing_write)
    with pytest.raises(OSError, match="synthetic lifecycle group"):
        run()
    failure = artifact.reload_json("failure.json")
    score.validate_failure_payload(failure)
    assert failure["stage"] == "terminal_revalidation"
    assert failure["terminal_published"] is False
    assert not artifact.has_name("score.json")
    assert not artifact.has_name("terminal.json")
    assert backend.closed is True


def test_execution_capability_is_required_before_attempt_or_target_binding(tmp_path: Path):
    artifact, _backend, unused_run = _run(tmp_path)
    with pytest.raises(score.FailClosedError, match="execution capability"):
        score.run_score_lifecycle(
            artifact=artifact, identity=_identity(), execution_capability=object(),
            within_roster=WITHIN, external_sessions=EXTERNAL,
            within_roster_factory=lambda: (_ for _ in ()).throw(AssertionError("must not resolve target")),
            external_roster_factory=lambda: (_ for _ in ()).throw(AssertionError("must not resolve target")),
            sealed_cell_d_table=_sealed_table(), a2_pooled={"within": {}, "external": {}},
            backend=score.NoLiveBackend(), final_reverify=lambda: _identity().launch_closure,
        )
    assert not artifact.has_name("attempt.json")


def test_failure_receipt_reports_exact_system_surface_mode_matrix_and_honest_target_opens(tmp_path: Path):
    artifact, _backend, run = _run(tmp_path, failure="tf:external:zero")
    with pytest.raises(RuntimeError, match="synthetic TF-SR"):
        run()
    failure = artifact.reload_json("failure.json")
    score.validate_failure_payload(failure)
    assert failure["opened"] == {"source": False, "within": True, "external": True, "formal": False}
    assert failure["resolved"] == {"source": False, "within": True, "external": True, "formal": False}
    assert failure["forward_calls"] == {
        "cell_d": {"within": {"aligned": 1}, "external": {"aligned": 1}},
        "tfsr": {
            "within": {"aligned": 1, "zero": 1, "wrong_pair": 1},
            "external": {"aligned": 1, "zero": 1, "wrong_pair": 0},
        },
    }
    assert failure["backward_calls"] == failure["optimizer_calls"] == 0


@pytest.mark.parametrize(
    ("external", "within", "expected"),
    (({"mean": -0.0001, "median": 1.0, "n_positive": 15, "n_total": 15},
      {"mean": 0.2, "median": 0.2, "n_positive": 6, "n_total": 6}, "STOP"),
     ({"mean": 0.03, "median": 0.001, "n_positive": 9, "n_total": 15},
      {"mean": -0.03, "median": 0.0, "n_positive": 3, "n_total": 6}, "CLEAR_GO"),
     ({"mean": 0.02, "median": 0.001, "n_positive": 9, "n_total": 15},
      {"mean": 0.0, "median": 0.0, "n_positive": 3, "n_total": 6}, "HOLD"),
     ({"mean": 0.05, "median": -0.001, "n_positive": 8, "n_total": 15},
      {"mean": 0.0, "median": 0.0, "n_positive": 3, "n_total": 6}, "HOLD"),
     ({"mean": 0.1, "median": 0.1, "n_positive": 15, "n_total": 15},
      {"mean": -0.031, "median": 0.0, "n_positive": 3, "n_total": 6}, "STOP")),
)
def test_gate_precedence_and_boundaries(external: dict[str, object], within: dict[str, object], expected: str):
    assert score.decide_verdict(external=external, within=within) == expected


def test_paired_statistics_order_and_fixed_seed_are_deterministic():
    first = score.paired_statistics([0.1, -0.2, 0.3])
    second = score.paired_statistics([0.1, -0.2, 0.3])
    assert first == second
    assert first["exact_sign_pattern"] == "+-+"
    assert first["bootstrap_seed"] == 42 and first["bootstrap_draws"] == 10_000


def test_strict_manifest_roster_and_formal_name_rejection():
    manifest = {"session_splits": {"train": ["a"], "val": list(WITHIN), "test": list(score.FORMAL_TEST_SESSION_NAMES)}}
    assert score.extract_within_roster(manifest, formal_names=score.FORMAL_TEST_SESSION_NAMES) == WITHIN
    bad = {"session_splits": {"train": ["a"], "val": list(WITHIN[:-1]) + [score.FORMAL_TEST_SESSION_NAMES[0]],
                              "test": list(score.FORMAL_TEST_SESSION_NAMES)}}
    with pytest.raises(score.FailClosedError, match="formal"):
        score.extract_within_roster(bad)


def _external_authorities() -> tuple[dict[str, object], dict[str, object]]:
    ledger, downloads, selected = [], [], []
    for index, session in enumerate(EXTERNAL):
        asset = f"asset-{index:02d}"; frozen = f"sub-M/{session}.nwb"; digest = _sha(asset)
        ledger.append({"asset_id": asset, "session_id": session, "eligible": True, "disposition": "ELIGIBLE", "frozen_path": frozen})
        downloads.append({"asset_id": asset, "bytes": index + 1, "sha256": digest,
                          "size_and_sha256_verified_before_nwb_open": True, "path": "/forbidden/cache"})
        selected.append({"asset_id": asset, "session_id": session, "path": frozen, "sha256": digest, "size": index + 1})
    return ({"asset_disposition_ledger": ledger, "verified_downloads": downloads,
             "eligible_session_ids": [f"asset-{index:02d}" for index in range(15)],
             "eligible_session_count": 15}, {"selected_assets": selected})


def test_fixed_v2_external_join_uses_asset_id_and_canonical_basename_only(tmp_path: Path):
    ledger, scope = _external_authorities()
    items = score.join_external_assets(ledger, scope, tmp_path)
    assert tuple(item.session for item in items) == EXTERNAL
    assert all(item.local_path.parent == tmp_path.absolute() for item in items)
    assert all(str(item.local_path) != "/forbidden/cache" for item in items)
    forged = json.loads(json.dumps(ledger)); forged["verified_downloads"][0]["bytes"] += 1
    with pytest.raises(score.FailClosedError, match="integrity"):
        score.join_external_assets(forged, scope, tmp_path)


def test_real_fixed_metadata_v2_uuid_join_matches_a2_roster_without_opening_nwb(tmp_path: Path):
    """The actual sealed metadata provides an integration-level UUID guard.

    Only the three JSON authorities are descriptor-read here; no data root,
    checkpoint, model, CUDA API, or parser is reached.
    """
    specs = {item.name: item for item in score.FIXED_AUTHORITIES}
    manifest = score.verify_authority(ROOT, specs["strict_manifest"]).value
    ledger = score.verify_authority(ROOT, specs["external_asset_ledger"]).value
    scope = score.verify_authority(ROOT, specs["external_scope"]).value
    a2 = score.verify_authority(ROOT, specs["a2_matched_reference"]).value
    assert manifest is not None and ledger is not None and scope is not None and a2 is not None
    within = score.extract_within_roster(manifest)
    external = score.extract_external_sessions(ledger)
    pooled = score.extract_a2_pooled_tables(a2, within_sessions=within, external_sessions=external)
    assert external == REAL_EXTERNAL_SUBM_15
    assert tuple(sorted(pooled["external"])) == REAL_EXTERNAL_SUBM_15
    assert tuple(sorted(pooled["within"])) == tuple(sorted(within))
    # This join only forms canonical local basenames under a fresh empty temp
    # root; it never stats or opens one of those potential NWB files.
    bindings = score.join_external_assets(ledger, scope, tmp_path)
    assert tuple(item.session for item in bindings) == REAL_EXTERNAL_SUBM_15
    # This non-empty UUID assertion distinguishes the receipt semantics from
    # a coincidentally correct session-name list.
    assert all("-" in item for item in ledger["eligible_session_ids"])


def test_target_free_preflight_root_authorization_and_closure_are_exact():
    training, fixed, preflight = _preflight()
    assert score.validate_target_free_preflight(preflight, training=training, fixed_authorities=fixed) == preflight
    preflight_sha = score._sha(score._json_bytes(preflight))
    authorization = score.build_root_authorization(
        official_preflight_sha256=preflight_sha, preflight=preflight,
    )
    assert score.validate_root_authorization(
        authorization, official_preflight_sha256=preflight_sha, preflight=preflight,
    ) == authorization
    closure_drift = json.loads(json.dumps(preflight))
    closure_drift["phase_e_closure"]["sha256_by_path"][score.PHASE_E_CLOSURE[0]] = _sha("forged")
    with pytest.raises(score.FailClosedError, match="aggregate SHA"):
        score.validate_target_free_preflight(closure_drift, training=training, fixed_authorities=fixed)
    input_binding_drift = json.loads(json.dumps(preflight))
    input_binding_drift["phase_e_closure"]["immutable_input_bindings"]["within_paired_view_manifest"]["mode"] = "0644"
    with pytest.raises(score.FailClosedError, match="immutable input"):
        score.validate_target_free_preflight(input_binding_drift, training=training, fixed_authorities=fixed)
    authorization_drift = json.loads(json.dumps(authorization))
    authorization_drift["official_preflight_sha256"] = _sha("other-preflight")
    with pytest.raises(score.FailClosedError, match="root authorization"):
        score.validate_root_authorization(
            authorization_drift, official_preflight_sha256=preflight_sha, preflight=preflight,
        )


def test_phase_e_closure_binds_full_tfpd_lane_lr_import_chain_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Phase-D receipt validator executes this eager package import chain.

    This is deliberately an independent literal: adding one path to the
    scorer's own list cannot make this test pass unless every initializer-only
    dependency is actually bound.  The real closure calculation reads code
    only; it does not resolve an NWB, checkpoint, or CUDA device.
    """
    assert score.TFPD_LANE_LR_IMPORT_CLOSURE == TFPD_LANE_LR_IMPORT_CLOSURE
    live = score.phase_e_closure(ROOT)
    assert score._validate_phase_e_closure_payload(live) == live
    assert score._closure_matches_authorization(ROOT, live) == live
    assert all(path in live["paths"] for path in TFPD_LANE_LR_IMPORT_CLOSURE)

    # A syntactically well-formed aggregate with any one package leaf omitted
    # remains invalid against the frozen ordered path set, so it cannot enter
    # a target-free preflight.
    for omitted in TFPD_LANE_LR_IMPORT_CLOSURE:
        forged = json.loads(json.dumps(live))
        forged["paths"].remove(omitted)
        del forged["sha256_by_path"][omitted]
        forged["closure_sha256"] = _sha(json.dumps(
            {
                "files": forged["sha256_by_path"],
                "immutable_input_bindings": forged["immutable_input_bindings"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        with pytest.raises(score.FailClosedError, match="path map drift"):
            score._validate_phase_e_closure_payload(forged)

    # Simulate a descriptor-observed byte change after a reviewed preflight;
    # the live launch/final recheck must reject it before any score can exist.
    original = score._canonical_regular_bytes

    def byte_drift(root: Path, relative: str, *, expected_mode: int):
        body, identity = original(root, relative, expected_mode=expected_mode)
        if relative == TFPD_LANE_LR_IMPORT_CLOSURE[1]:
            return body + b"\\nsynthetic closure byte drift", identity
        return body, identity

    monkeypatch.setattr(score, "_canonical_regular_bytes", byte_drift)
    with pytest.raises(score.FailClosedError, match="implementation closure drift"):
        score._closure_matches_authorization(ROOT, live)


def test_phase_e_cli_bootstrap_resolves_tfpd_lane_before_phase_d_validation() -> None:
    """The actual CLI supplies both package roots without ambient PYTHONPATH.

    The first half reproduces the former parent-only import failure.  The
    second half invokes the real CLI loader, then executes exactly
    ``train.lr_for_step`` -- the import reached by the metadata-only
    Phase-D receipt validator -- without opening a checkpoint/NWB or calling
    CUDA.
    """
    script = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_score.py"
    code = r'''
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

root = Path(sys.argv[1]).resolve()
script = Path(sys.argv[2]).resolve()
repo_paths = {str(root), str(root / "tfpd_exploration"), str(root / "tfpd_exploration" / "src")}
sys.path[:] = [item for item in sys.path if item not in repo_paths]

# Reproduce the old synthetic-package path before the CLI added deterministic
# repository roots.  It must fail at the top-level ``tfpd_lane`` import.
package = ModuleType("_phasee_without_bootstrap")
package.__path__ = [str(root / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1")]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    package.__name__ + ".score",
    root / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/score.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
train = importlib.import_module(package.__name__ + ".train")
try:
    train.lr_for_step(0)
except ModuleNotFoundError as error:
    assert error.name == "tfpd_lane"
else:
    raise AssertionError("missing bootstrap unexpectedly resolved tfpd_lane")

# Now exercise the actual CLI bootstrap and the same metadata-only LR import.
cli_spec = importlib.util.spec_from_file_location("_phasee_cli", script)
cli = importlib.util.module_from_spec(cli_spec)
sys.modules[cli_spec.name] = cli
cli_spec.loader.exec_module(cli)
route = cli._load_static()
validated_train = importlib.import_module("_tfsr_phase_e_static.train")
assert str(root / "tfpd_exploration") in sys.path
assert str(root / "tfpd_exploration/src") in sys.path
assert validated_train.lr_for_step(0) == 1e-5
assert "torch" in sys.modules
print(route.PHASE)
'''
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(script)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == score.PHASE


def test_within_paired_view_manifest_is_descriptor_bound_and_not_caller_configurable(
    monkeypatch: pytest.MonkeyPatch,
):
    """The only legal within six come from the sealed mode-0600 C1 manifest."""
    assets, binding = score.canonical_within_assets_from_paired_view_manifest(ROOT, within_roster=WITHIN)
    assert tuple(item["session"] for item in assets) == WITHIN
    assert assets == score.sealed_within_assets()
    assert binding.payload()["relative_path"] == score.WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE
    assert binding.payload()["body_sha256"] == score.WITHIN_PAIRED_VIEW_MANIFEST_SHA256
    assert binding.payload()["mode"] == "0600"
    assert binding.payload()["selected_split"] == "val"
    assert binding.payload()["read_once"] is True

    # The builder has no old mapping injection point.  A seemingly valid
    # alternative table cannot enter a durable target-free preflight.
    training, fixed, _preflight_payload = _preflight()
    with pytest.raises(TypeError):
        score.build_target_free_preflight(
            root=ROOT,
            training=training,
            fixed_authorities=fixed,
            closure=_phase_e_closure(),
            within_roster=WITHIN,
            normalizers=_normalizers(),
            within_assets=[{"session": "forbidden"}],
        )

    original_reader = score._canonical_regular_bytes
    body, identity = original_reader(
        ROOT,
        score.WITHIN_PAIRED_VIEW_MANIFEST_RELATIVE,
        expected_mode=score.WITHIN_PAIRED_VIEW_MANIFEST_MODE,
    )
    monkeypatch.setattr(score, "_canonical_regular_bytes", lambda *_args, **_kwargs: (b"{}\n", identity))
    with pytest.raises(score.FailClosedError, match="SHA drift"):
        score.canonical_within_assets_from_paired_view_manifest(ROOT, within_roster=WITHIN)
    monkeypatch.setattr(score, "_canonical_regular_bytes", original_reader)

    # A semantic row replacement is rejected even in this test-only scenario
    # where the outer SHA guard is deliberately made to pass.
    forged_manifest = json.loads(body)
    forged_manifest["file_inventory"]["val"][0]["bytes"] += 1
    forged_body = json.dumps(forged_manifest, sort_keys=True).encode("utf-8")
    original_sha = score._sha

    def sha_with_manifest_override(value: bytes) -> str:
        return score.WITHIN_PAIRED_VIEW_MANIFEST_SHA256 if value == forged_body else original_sha(value)

    monkeypatch.setattr(score, "_canonical_regular_bytes", lambda *_args, **_kwargs: (forged_body, identity))
    monkeypatch.setattr(score, "_sha", sha_with_manifest_override)
    with pytest.raises(score.FailClosedError, match="semantic row drift"):
        score.canonical_within_assets_from_paired_view_manifest(ROOT, within_roster=WITHIN)


def test_preflight_rejects_paired_view_row_and_manifest_identity_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    training, fixed, preflight = _preflight()
    row_drift = json.loads(json.dumps(preflight))
    row_drift["within_assets"][0]["sha256"] = _sha("substituted-within-row")
    with pytest.raises(score.FailClosedError, match="paired-view asset authority"):
        score.validate_target_free_preflight(row_drift, training=training, fixed_authorities=fixed)

    # Durable validation fixes the descriptor identity shape; lifecycle
    # verification additionally compares it against a fresh descriptor read.
    artifact_parent, _ = score.canonical_authority_parent(tmp_path)
    artifact_parent.mkdir(parents=True)
    artifact = score.reserve_authority_artifact(tmp_path, score.issue_root_publication_capability())
    cap = score.issue_root_publication_capability()
    preflight_sha = score.publish_target_free_preflight(
        artifact, cap, preflight, training=training, fixed_authorities=fixed,
    )
    authorization = score.build_root_authorization(
        official_preflight_sha256=preflight_sha,
        preflight=preflight,
    )
    score.publish_root_authorization(artifact, cap, authorization, training=training, fixed_authorities=fixed)
    materials = {
        item.name: score.AuthorityMaterial(item, b"{}", {}, (1, 2, 3))
        for item in score.FIXED_AUTHORITIES
    }
    strict_spec = next(item for item in score.FIXED_AUTHORITIES if item.name == "strict_manifest")
    materials["strict_manifest"] = score.AuthorityMaterial(
        strict_spec,
        b"{}",
        {"session_splits": {"train": ["synthetic-train"], "val": list(WITHIN),
                            "test": list(score.FORMAL_TEST_SESSION_NAMES)}},
        (1, 2, 3),
    )
    observed = score._within_manifest_binding_from_payload(preflight["within_paired_view_manifest"])
    replacement = score.WithinPairedViewManifestBinding(
        relative_path=observed.relative_path,
        body_sha256=observed.body_sha256,
        mode=observed.mode,
        descriptor_identity=(observed.descriptor_identity[0] + 1, observed.descriptor_identity[1], observed.descriptor_identity[2]),
        selected_split=observed.selected_split,
        read_once=True,
    )
    monkeypatch.setattr(
        score,
        "canonical_within_assets_from_paired_view_manifest",
        lambda _root, *, within_roster: (score.sealed_within_assets(), replacement),
    )
    monkeypatch.setattr(score, "_closure_matches_authorization", lambda _root, expected: dict(expected))
    with pytest.raises(score.FailClosedError, match="paired-view within authority changed"):
        score.verify_phase_e_authorization(tmp_path, training, materials)


def test_dual_memory_authorities_propagate_exactly_through_preflight_and_root_authorization():
    training, fixed, preflight = _preflight()
    assert score.FROZEN_SCORE_DEVICE == SEALED_SCORE_DEVICE
    assert preflight["device_contract"] == SEALED_SCORE_DEVICE
    preflight_sha = score._sha(score._json_bytes(preflight))
    authorization = score.build_root_authorization(
        official_preflight_sha256=preflight_sha, preflight=preflight,
    )
    assert authorization["device_contract"] == SEALED_SCORE_DEVICE
    assert score.validate_root_authorization(
        authorization, official_preflight_sha256=preflight_sha, preflight=preflight,
    ) == authorization

    # Schema validation rejects each legacy or unit-confused replacement
    # before any backend could resolve an evaluation asset or forward a model.
    for field, replacement in (
        ("nvidia_smi_memory_total_mib", SEALED_SCORE_DEVICE["torch_total_memory_bytes"]),
        ("torch_total_memory_bytes", SEALED_SCORE_DEVICE["nvidia_smi_memory_total_mib"]),
        ("torch_total_memory_bytes", 24259),  # floor(actual Torch bytes / MiB)
        ("torch_total_memory_bytes", 24260 * 1024 * 1024),  # rounded MiB, converted back to bytes
    ):
        forged = dict(SEALED_SCORE_DEVICE)
        forged[field] = replacement
        with pytest.raises(score.FailClosedError, match="device contract"):
            score._validate_device_contract(forged)


@pytest.mark.parametrize(
    ("torch_total_memory", "nvidia_smi_memory_mib"),
    (
        # The two authoritative values are swapped across their interfaces.
        (SEALED_SCORE_DEVICE["nvidia_smi_memory_total_mib"], SEALED_SCORE_DEVICE["torch_total_memory_bytes"]),
        # A plausible but wrong device's Torch byte count is substituted.
        (25435111424, SEALED_SCORE_DEVICE["nvidia_smi_memory_total_mib"]),
        # Floor-MiB and rounded-MiB values are not byte authorities.
        (24259, SEALED_SCORE_DEVICE["nvidia_smi_memory_total_mib"]),
        (24260 * 1024 * 1024, SEALED_SCORE_DEVICE["nvidia_smi_memory_total_mib"]),
        # The nominal nvidia-smi field is independently checked after Torch
        # has supplied the exact byte authority.
        (SEALED_SCORE_DEVICE["torch_total_memory_bytes"], SEALED_SCORE_DEVICE["torch_total_memory_bytes"]),
    ),
    ids=("swapped", "substituted", "floor_mib", "rounded_mib", "nominal_substituted"),
)
def test_physical_dual_memory_attestation_fails_before_data_or_model_import(
    monkeypatch: pytest.MonkeyPatch,
    torch_total_memory: int,
    nvidia_smi_memory_mib: int,
):
    """Exercise the device gate with fakes only: no Torch/CUDA/data/model import."""
    calls: list[str] = []

    class FakeProperties:
        name = SEALED_SCORE_DEVICE["name"]
        total_memory = torch_total_memory

    class FakeCuda:
        def is_available(self) -> bool:
            calls.append("cuda.is_available")
            return True

        def device_count(self) -> int:
            calls.append("cuda.device_count")
            return 1

        def current_device(self) -> int:
            calls.append("cuda.current_device")
            return 0

        def get_device_properties(self, index: int) -> FakeProperties:
            assert index == 0
            calls.append("cuda.get_device_properties")
            return FakeProperties()

    class FakeTorch:
        __version__ = SEALED_SCORE_DEVICE["torch_version"]
        cuda = FakeCuda()

    class FakeTorchMetrics:
        __version__ = "1.5.1"

    def fake_nvidia_smi() -> tuple[str, ...]:
        calls.append("nvidia-smi")
        return (
            f"{SEALED_SCORE_DEVICE['uuid']}, {SEALED_SCORE_DEVICE['bdf']}, "
            f"{SEALED_SCORE_DEVICE['name']}, {nvidia_smi_memory_mib}",
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(score.FailClosedError, match="memory"):
        score._attest_physical_score_runtime(
            torch=FakeTorch(), torchmetrics=FakeTorchMetrics(), nvidia_smi_query=fake_nvidia_smi,
        )
    assert calls
    assert not any("data" in call or "model" in call or "forward" in call for call in calls)

    # The live loader invokes this pure gate before its decoder and NWB imports.
    loader = inspect.getsource(score.PhysicalMatchedScoreBackend._load_runtime)
    gate = loader.index("_attest_physical_score_runtime(")
    assert gate < loader.index("from . import train")
    assert gate < loader.index("from mc_maze.multisession_datamodule import load_dandi688_session")


def test_preflight_requires_exact_sealed_behavior_normalizer_float32_values():
    training, fixed, preflight = _preflight()
    assert score.SOURCE_BEHAVIOR_MEAN == SEALED_SOURCE_BEHAVIOR_MEAN
    assert score.SOURCE_BEHAVIOR_STD == SEALED_SOURCE_BEHAVIOR_STD
    assert preflight["normalizers"]["behavior_mean"] == list(SEALED_SOURCE_BEHAVIOR_MEAN)
    assert preflight["normalizers"]["behavior_std"] == list(SEALED_SOURCE_BEHAVIOR_STD)
    for field, replacement in (("behavior_mean", [0.0, 0.0]), ("behavior_std", [1.0, 1.0])):
        forged = json.loads(json.dumps(preflight))
        # Preserve every source/semantic authority string: only the actual
        # numbers are adversarially changed.
        forged["normalizers"][field] = replacement
        with pytest.raises(score.FailClosedError, match="behavior source normalizer numeric drift"):
            score.validate_target_free_preflight(forged, training=training, fixed_authorities=fixed)


def test_within_preflight_binding_rejects_symlink_root_and_preserves_roster(tmp_path: Path):
    _training, _fixed, preflight = _preflight()
    root = tmp_path / "subc"; root.mkdir()
    bindings = score.join_within_assets(preflight, root)
    assert tuple(item.session for item in bindings) == tuple(sorted(WITHIN))
    assert all(item.local_path.parent == root.absolute() for item in bindings)
    alias = tmp_path / "subc-alias"; alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(score.FailClosedError, match="canonical directory"):
        score.join_within_assets(preflight, alias)


def test_root_only_authority_publication_is_ordered_and_pair_immutable(tmp_path: Path):
    training, fixed, preflight = _preflight()
    artifact = score.reserve_artifact_root(tmp_path, "authority", score.AUTHORITY_TOPOLOGY)
    preflight_sha = score._sha(score._json_bytes(preflight))
    authorization = score.build_root_authorization(
        official_preflight_sha256=preflight_sha, preflight=preflight,
    )
    with pytest.raises(score.FailClosedError, match="only root"):
        score.publish_target_free_preflight(
            artifact, object(), preflight, training=training, fixed_authorities=fixed,
        )
    capability = score.issue_root_publication_capability()
    malformed_preflight = json.loads(json.dumps(preflight))
    del malformed_preflight["status"]
    with pytest.raises(score.FailClosedError, match="preflight schema/binding drift"):
        score.publish_target_free_preflight(
            artifact, capability, malformed_preflight, training=training, fixed_authorities=fixed,
        )
    assert not artifact.has_name("official_preflight.json")
    assert not artifact.has_name("root_authorization.json")
    arbitrary_behavior = json.loads(json.dumps(preflight))
    arbitrary_behavior["normalizers"]["behavior_mean"] = [0.0, 0.0]
    with pytest.raises(score.FailClosedError, match="behavior source normalizer numeric drift"):
        score.publish_target_free_preflight(
            artifact, capability, arbitrary_behavior, training=training, fixed_authorities=fixed,
        )
    assert not artifact.has_name("official_preflight.json")
    assert not artifact.has_name("root_authorization.json")
    assert score.publish_target_free_preflight(
        artifact, capability, preflight, training=training, fixed_authorities=fixed,
    ) == preflight_sha
    with pytest.raises(score.FailClosedError, match="requires durable preflight"):
        # An independent root with no preflight cannot publish authorization.
        empty = score.reserve_artifact_root(tmp_path, "authority-empty", score.AUTHORITY_TOPOLOGY)
        score.publish_root_authorization(
            empty, capability, authorization, training=training, fixed_authorities=fixed,
        )
    malformed_authorization = json.loads(json.dumps(authorization))
    del malformed_authorization["status"]
    with pytest.raises(score.FailClosedError, match="root authorization binding drift"):
        score.publish_root_authorization(
            artifact, capability, malformed_authorization, training=training, fixed_authorities=fixed,
        )
    assert not artifact.has_name("root_authorization.json")
    wrong_preflight_sha = score.build_root_authorization(
        official_preflight_sha256=_sha("wrong-preflight"), preflight=preflight,
    )
    with pytest.raises(score.FailClosedError, match="root authorization binding drift"):
        score.publish_root_authorization(
            artifact, capability, wrong_preflight_sha, training=training, fixed_authorities=fixed,
        )
    assert not artifact.has_name("root_authorization.json")
    assert artifact.reload_json("official_preflight.json", preflight_sha) == preflight
    authorization_sha = score.publish_root_authorization(
        artifact, capability, authorization, training=training, fixed_authorities=fixed,
    )
    assert artifact.reload_json("root_authorization.json", authorization_sha) == authorization
    with pytest.raises(score.FailClosedError, match="collision"):
        score.publish_root_authorization(
            artifact, capability, authorization, training=training, fixed_authorities=fixed,
        )


def test_root_authorization_revalidates_durable_preflight_before_its_own_publish(tmp_path: Path):
    """A pre-existing invalid pair cannot be upgraded into root authorization."""
    training, fixed, preflight = _preflight()
    malformed_durable = json.loads(json.dumps(preflight))
    malformed_durable["normalizers"]["behavior_std"] = [1.0, 1.0]
    artifact = score.reserve_artifact_root(tmp_path, "authority-invalid-durable", score.AUTHORITY_TOPOLOGY)
    # This intentionally bypasses the root helper only to model a corrupted
    # old/foreign durable pair.  The publication function under test must
    # descriptor-reload and reject it before it can create root_authorization.
    durable_sha = artifact.publish_json("official_preflight.json", malformed_durable)
    authorization = score.build_root_authorization(
        official_preflight_sha256=durable_sha, preflight=malformed_durable,
    )
    with pytest.raises(score.FailClosedError, match="behavior source normalizer numeric drift"):
        score.publish_root_authorization(
            artifact,
            score.issue_root_publication_capability(),
            authorization,
            training=training,
            fixed_authorities=fixed,
        )
    assert artifact.reload_json("official_preflight.json", durable_sha) == malformed_durable
    assert not artifact.has_name("root_authorization.json")


def test_reloaded_authority_pairs_bind_preflight_root_decision_and_live_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    training, fixed, preflight = _preflight()
    capability = score.issue_root_publication_capability()
    authority_parent, _name = score.canonical_authority_parent(tmp_path)
    authority_parent.mkdir(parents=True)
    artifact = score.reserve_authority_artifact(tmp_path, capability)
    preflight_sha = score.publish_target_free_preflight(
        artifact, capability, preflight, training=training, fixed_authorities=fixed,
    )
    authorization = score.build_root_authorization(
        official_preflight_sha256=preflight_sha, preflight=preflight,
    )
    score.publish_root_authorization(
        artifact, capability, authorization, training=training, fixed_authorities=fixed,
    )
    materials = {
        item.name: score.AuthorityMaterial(item, b"{}", {}, (1, 2, 3))
        for item in score.FIXED_AUTHORITIES
    }
    strict_spec = next(item for item in score.FIXED_AUTHORITIES if item.name == "strict_manifest")
    materials["strict_manifest"] = score.AuthorityMaterial(
        strict_spec,
        b"{}",
        {"session_splits": {"train": ["synthetic-train"], "val": list(WITHIN),
                            "test": list(score.FORMAL_TEST_SESSION_NAMES)}},
        (1, 2, 3),
    )
    manifest_binding = score._within_manifest_binding_from_payload(
        preflight["within_paired_view_manifest"],
    )
    monkeypatch.setattr(
        score,
        "canonical_within_assets_from_paired_view_manifest",
        lambda _root, *, within_roster: (score.sealed_within_assets(), manifest_binding),
    )
    monkeypatch.setattr(score, "_closure_matches_authorization", lambda _root, expected: dict(expected))
    loaded = score.verify_phase_e_authorization(tmp_path, training, materials)
    assert loaded.preflight_sha256 == preflight_sha
    assert loaded.expected_closure == preflight["phase_e_closure"]

    sidecar = artifact.directory / "root_authorization.json.sha256"
    os.chmod(sidecar, 0o644)
    sidecar.write_text("forged\n")
    os.chmod(sidecar, 0o444)
    with pytest.raises(score.FailClosedError, match="sidecar"):
        score.verify_phase_e_authorization(tmp_path, training, materials)


def test_phase_d_v2_terminal_validation_fails_before_any_target_path_when_root_is_absent(tmp_path: Path):
    assert score.PHASE_D_TRAIN_ROOT_RELATIVE.endswith("train_v2")
    with pytest.raises(score.FailClosedError, match="Phase-D training root absent"):
        score.validate_phase_d_training_terminal(tmp_path)


def test_training_evidence_rejects_incomplete_final_four_or_non_sha_binding():
    with pytest.raises(ValueError, match="final-four"):
        score.TrainingEvidence(
            terminal_sha256=_sha("terminal"), swa_sha256=_sha("swa"), swa_state_digest=_sha("state"),
            closure={}, checkpoint_sha256={"44": _sha("44")},
        )
    with pytest.raises(ValueError, match="final-four"):
        score.TrainingEvidence(
            terminal_sha256=_sha("terminal"), swa_sha256=_sha("swa"), swa_state_digest=_sha("state"),
            closure={}, checkpoint_sha256={str(epoch): _sha(str(epoch)) for epoch in (44, 45, 46, 47)}
            | {"48": _sha("48")},
        )


def test_physical_backend_construction_is_inert_and_public_route_has_no_backend_substitution(tmp_path: Path):
    training, _fixed, preflight = _preflight()
    materials = {
        item.name: score.AuthorityMaterial(item, b"{}", {}, (1, 2, 3))
        for item in score.FIXED_AUTHORITIES
    }
    authorization = score.PhaseEAuthorization(
        preflight_sha256=_sha("preflight"), root_authorization_sha256=_sha("authorization"),
        preflight=preflight, root_authorization={}, expected_closure=_phase_e_closure(),
    )
    backend = score.PhysicalMatchedScoreBackend(
        root=tmp_path, fixed_authorities=materials, training=training, authorization=authorization,
    )
    assert backend._runtime is None
    assert backend._sessions == {"within": {}, "external": {}}
    assert "backend" not in inspect.signature(score.execute_authorized).parameters


@pytest.mark.parametrize("failure_call", (1, 2, 3, 4), ids=("score_body", "score_sidecar", "terminal_body", "terminal_sidecar"))
def test_group_publication_write_failure_rolls_back_every_owned_leaf(
    failure_call: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    artifact = _artifact(tmp_path)
    original = score._write_full
    calls = 0

    def failing_write(descriptor: int, body: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("synthetic group stream failure")
        original(descriptor, body)

    monkeypatch.setattr(score, "_write_full", failing_write)
    with pytest.raises(OSError, match="synthetic group"):
        artifact.publish_group({"score.json": b"score", "terminal.json": b"terminal"})
    assert not artifact.has_name("score.json")
    assert not artifact.has_name("terminal.json")


def test_group_post_publish_or_reload_failure_rolls_back_both_scientific_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    artifact = _artifact(tmp_path)
    with pytest.raises(RuntimeError, match="post-score"):
        artifact.publish_group(
            {"score.json": b"score", "terminal.json": b"terminal"},
            post_publish=lambda _bodies, _digests: (_ for _ in ()).throw(RuntimeError("post-score closure drift")),
        )
    assert not artifact.has_name("score.json")
    assert not artifact.has_name("terminal.json")

    original = score.ArtifactRoot.reload_pair

    def terminal_reload_failure(self: score.ArtifactRoot, name: str, expected_sha: str | None = None) -> bytes:
        if name == "terminal.json":
            raise score.FailClosedError("synthetic terminal reload failure")
        return original(self, name, expected_sha)

    monkeypatch.setattr(score.ArtifactRoot, "reload_pair", terminal_reload_failure)
    with pytest.raises(score.FailClosedError, match="terminal reload"):
        artifact.publish_group({"score.json": b"score", "terminal.json": b"terminal"})
    assert not artifact.has_name("score.json")
    assert not artifact.has_name("terminal.json")


def test_held_fd_hash_identity_and_rename_swap_attack_fail_closed(tmp_path: Path):
    body = b"payload"
    root = tmp_path / "subm"; root.mkdir()
    path = root / "external-00.nwb"; path.write_bytes(body)
    binding = score.ExternalAssetBinding(
        "asset", "external-00", path, len(body), score._sha(body), "sub-M/external-00.nwb",
        data_root=score.DataRootCapability.from_directory(root, label="synthetic"),
    )
    held = score.hold_verified_external_asset(binding)
    try:
        moved = tmp_path / "moved.nwb"; path.rename(moved); path.write_bytes(body)
        with pytest.raises(score.FailClosedError, match="pathname identity"):
            held.reverify()
    finally:
        held.close()


def test_held_asset_rejects_data_root_rename_swap_before_or_after_open(tmp_path: Path):
    body = b"payload"
    root = tmp_path / "subm"; root.mkdir()
    path = root / "external-00.nwb"; path.write_bytes(body)
    binding = score.EvaluationAssetBinding(
        surface="external", asset_id="asset", session="external-00", local_path=path,
        expected_bytes=len(body), expected_sha256=score._sha(body), frozen_path="sub-M/external-00.nwb",
        data_root=score.DataRootCapability.from_directory(root, label="synthetic"),
    )
    held = score.hold_verified_asset(binding)
    try:
        moved = tmp_path / "subm-moved"; root.rename(moved); root.mkdir()
        (root / path.name).write_bytes(body)
        with pytest.raises(score.FailClosedError, match="target data root"):
            held.reverify()
    finally:
        held.close()


def test_real_synthetic_cell_d_lazy_state_digest_is_stable_and_topology_bound(
    monkeypatch: pytest.MonkeyPatch,
):
    """Exercise the real Cell-D graph, never a checkpoint/data/GPU surrogate."""
    import torch
    from torch.nn.parameter import UninitializedParameter
    from src.tfpd_lane import arm_common
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    model = build_population_robustness_model(seed=42, cell="D")
    lazy_keys = tuple(
        name for name, value in model.state_dict().items()
        if isinstance(value, UninitializedParameter)
    )
    assert lazy_keys == (
        "decoder.fc_id_in.0.weight",
        "decoder.fc_id_in.0.bias",
    )
    before = score._physical_model_state_digest(model, torch)
    # The explicit Phase-E algorithm is deliberately compatible with the
    # established shared state digest, including its lazy sentinel treatment.
    assert before == arm_common.state_sha256(model)

    model.eval()
    neural = torch.zeros(1, 50, 3)
    calibration = torch.zeros(1, 30, 100, 3)
    side = torch.zeros(1, 3, 4)
    with torch.no_grad():
        prediction, identity = model(neural, calib_trials=calibration, side_features=side)
    assert tuple(prediction.shape) == (1, 50, 2)
    assert tuple(identity.shape) == (1, 3, 50)
    after_forward = score._physical_model_state_digest(model, torch)
    assert after_forward == before
    assert after_forward == arm_common.state_sha256(model)
    assert tuple(
        name for name, value in model.state_dict().items()
        if isinstance(value, UninitializedParameter)
    ) == lazy_keys

    class StateOnly:
        def __init__(self, state: dict[str, object]) -> None:
            self._state = state

        def state_dict(self) -> dict[str, object]:
            return self._state

    # Dropping or materializing an otherwise dead lazy entry changes the
    # sorted-key/sentinel representation and therefore cannot masquerade as
    # an unchanged Cell-D model state.
    dropped_lazy = dict(model.state_dict())
    dropped_lazy.pop(lazy_keys[0])
    assert score._physical_model_state_digest(StateOnly(dropped_lazy), torch) != before
    materialized_lazy = dict(model.state_dict())
    materialized_lazy[lazy_keys[0]] = torch.zeros(1)
    assert score._physical_model_state_digest(StateOnly(materialized_lazy), torch) != before

    original_sentinel = score._UNINITIALIZED_LAZY_STATE_SENTINEL
    monkeypatch.setattr(score, "_UNINITIALIZED_LAZY_STATE_SENTINEL", b"|altered-lazy-sentinel|")
    assert score._physical_model_state_digest(model, torch) != before
    monkeypatch.setattr(score, "_UNINITIALIZED_LAZY_STATE_SENTINEL", original_sentinel)

    initialized = next(
        value for value in model.state_dict().values()
        if not isinstance(value, UninitializedParameter) and value.is_floating_point() and value.numel() > 0
    )
    with torch.no_grad():
        initialized.reshape(-1)[0].add_(1.0)
    assert score._physical_model_state_digest(model, torch) != before
    assert tuple(
        name for name, value in model.state_dict().items()
        if isinstance(value, UninitializedParameter)
    ) == lazy_keys


def test_real_synthetic_cell_d_final_resource_disclosure_is_lazy_safe(
    monkeypatch: pytest.MonkeyPatch,
):
    """The live resource helper must count real Cell-D without touching lazy numel."""
    import torch
    from torch.nn.parameter import UninitializedParameter
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    class TinyTfsr(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3))

        def accounting(self, *, representative_n: int) -> dict[str, int]:
            assert representative_n == 128
            return {"analytic_mac_estimate_n": 123, "persistent_state_bytes": 0}

    def disclose(model: object, *, terminal: object | None = None) -> dict[str, object]:
        return score._build_physical_resource_disclosure(
            cell_d=model,
            tfsr=TinyTfsr(),
            torch=torch,
            torchmetrics_version="1.5.1",
            training_peak_memory_bytes=7,
            score_peak_memory_bytes=11,
            aligned_latency_ms=0.25,
            sealed_cell_d_terminal=_sealed_cell_d_terminal() if terminal is None else terminal,
        )

    model = build_population_robustness_model(seed=42, cell="D")
    assert tuple(
        name for name, parameter in model.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ) == SEALED_CELL_D_LAZY_KEYS
    # The physical route uses ``Module.to(device)`` before strict state load.
    # CPU placement proves that this model-level operation preserves—not
    # materializes—the same two lazy entries.
    model = model.to("cpu")
    assert tuple(
        name for name, parameter in model.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ) == SEALED_CELL_D_LAZY_KEYS

    def refuse_lazy_numel(self: UninitializedParameter) -> int:
        raise AssertionError("resource accounting called numel() on a lazy parameter")

    monkeypatch.setattr(UninitializedParameter, "numel", refuse_lazy_numel)
    receipt = disclose(model)
    assert receipt["cell_d_parameters"] == SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS
    assert receipt["cell_d_parameter_accounting"] == _cell_d_parameter_accounting()
    assert tuple(
        name for name, parameter in model.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ) == SEALED_CELL_D_LAZY_KEYS

    # A missing lazy key, a materialized lazy key, or an initialized-count
    # change cannot pass as a smaller (or zero-counted) Cell-D graph.
    dropped = build_population_robustness_model(seed=42, cell="D")
    dropped.decoder.fc_id_in[0]._parameters.pop("weight")
    with pytest.raises(score.FailClosedError, match="lazy parameter topology"):
        disclose(dropped)

    materialized = build_population_robustness_model(seed=42, cell="D")
    materialized.decoder.fc_id_in[0]._parameters["weight"] = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(score.FailClosedError, match="lazy parameter topology"):
        disclose(materialized)

    count_drift = build_population_robustness_model(seed=42, cell="D")
    first_initialized = next(
        parameter for parameter in count_drift.parameters()
        if not isinstance(parameter, UninitializedParameter)
    )
    first_initialized.requires_grad_(False)
    with pytest.raises(score.FailClosedError, match="initialized trainable parameter count"):
        disclose(count_drift)

    terminal_drift = _sealed_cell_d_terminal()
    terminal_drift["swa"]["manifest"]["uninitialized_lazy_tensor_count"] = 3
    with pytest.raises(score.FailClosedError, match="lazy-parameter authority"):
        disclose(build_population_robustness_model(seed=42, cell="D"), terminal=terminal_drift)


def test_cell_d_weights_only_loader_scopes_exact_lazy_allowlist(
    monkeypatch: pytest.MonkeyPatch,
):
    """A lazy checkpoint needs exactly one temporary weights-only allowlist entry."""
    import io
    import torch
    from torch.nn.parameter import UninitializedParameter
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    # This is the real Cell-D graph, but its checkpoint is manufactured in
    # memory from a freshly initialized CPU model.  It never opens the sealed
    # SWA or any target/data artifact.
    source = build_population_robustness_model(seed=42, cell="D")
    buffer = io.BytesIO()
    torch.save({"state_dict": source.state_dict()}, buffer)
    body = buffer.getvalue()

    # This is the failure reproduced against the sealed Cell-D SWA.  The test
    # uses only an in-memory synthetic payload, never an artifact path.
    with pytest.raises(Exception):
        torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)

    before_globals = tuple(torch.serialization.get_safe_globals())
    original_safe_globals = torch.serialization.safe_globals
    original_load = torch.load
    observed_allowlists: list[tuple[object, ...]] = []
    observed_weights_only: list[object] = []

    @contextmanager
    def recording_safe_globals(values: object):
        observed_allowlists.append(tuple(values))
        with original_safe_globals(values):
            yield

    def recording_load(*args: object, **kwargs: object):
        observed_weights_only.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch.serialization, "safe_globals", recording_safe_globals)
    monkeypatch.setattr(torch, "load", recording_load)
    state = score._load_cell_d_swa_weights_only_state(torch, body)
    assert observed_allowlists == [(UninitializedParameter,)]
    assert observed_weights_only == [True]
    assert tuple(torch.serialization.get_safe_globals()) == before_globals

    target = build_population_robustness_model(seed=42, cell="D").to("cpu")
    assert tuple(
        name for name, parameter in target.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ) == SEALED_CELL_D_LAZY_KEYS
    target.load_state_dict(state, strict=True)
    assert tuple(
        name for name, parameter in target.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ) == SEALED_CELL_D_LAZY_KEYS


def test_typed_t4_controls_are_post_normalization_zero_and_cyclic_derangement():
    import torch
    from src.tfsr_b3st4_ddrop_v1.model import NormalizedT4Batch, ordered_unit_digest
    ids = ("u0", "u1", "u2")
    cap = NormalizedT4Batch(
        tensor=torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]),
        raw_authority_sha256=_sha("raw"), normalizer_authority_sha256=_sha("norm"), roster_digest=_sha("roster"),
        ordered_unit_digest=ordered_unit_digest(ids), ordered_unit_ids=ids, lineage=("synthetic",), diagnostic_mode="aligned",
    )
    controls = score.make_t4_controls(cap)
    assert torch.equal(controls.zero.tensor, torch.zeros_like(cap.tensor))
    assert controls.permutation == (1, 2, 0)
    assert all(index != item for index, item in enumerate(controls.permutation))
    assert torch.equal(controls.wrong_pair.tensor, cap.tensor[:, [1, 2, 0], :])
    with pytest.raises(score.FailClosedError, match="capability"):
        score.make_t4_controls(cap.tensor)


def test_raw_m30_t4_sua_axis_proof_rejects_channel_metadata_and_source_count_drift():
    import numpy as np

    class Metadata:
        feature_group = "t4"
        pool_size = 30

    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    proof = score._raw_t4_sua_axis_proof_payload(
        np,
        session=WITHIN[0],
        neural_unit_count=3,
        channel_ids=np.arange(3, dtype=np.int64),
        source_unit_count=3,
        raw_t4=raw,
        metadata=Metadata(),
    )
    assert proof["channel_ids_dtype"] == "int64"
    assert proof["source_unit_count"] == 3
    assert proof["raw_t4_shape"] == [3, 4]
    assert proof["closure_bound_functions"] == [
        "mc_maze.multisession_datamodule.load_dandi688_session",
        "mc_maze.unit_side_features.compute_unit_side_features_uncached",
    ]
    for kwargs in (
        {"channel_ids": np.asarray([1, 0, 2], dtype=np.int64)},
        {"channel_ids": np.asarray([0, 1, 2], dtype=np.int32)},
        {"source_unit_count": 4},
        {"metadata": type("BadMetadata", (), {"feature_group": "z4", "pool_size": 30})()},
        {"metadata": type("BadMetadata", (), {"feature_group": "t4", "pool_size": 29})()},
    ):
        arguments = {
            "session": WITHIN[0],
            "neural_unit_count": 3,
            "channel_ids": np.arange(3, dtype=np.int64),
            "source_unit_count": 3,
            "raw_t4": raw,
            "metadata": Metadata(),
        }
        arguments.update(kwargs)
        with pytest.raises(score.FailClosedError, match="channel-order proof"):
            score._raw_t4_sua_axis_proof_payload(np, **arguments)


def test_query_level_last_bin_authority_rejects_target_mask_count_and_receipt_omission(tmp_path: Path):
    import numpy as np

    behavior = np.arange(160, dtype=np.float32).reshape(80, 2)
    starts = np.asarray([0, 10, 20], dtype=np.int64)
    target, mask, target_sha, mask_sha, count = score._last_bin_authority_from_behavior(
        np,
        behavior=behavior,
        starts=starts,
    )
    assert target.shape == (3, 2)
    assert mask.tolist() == [1, 1, 1]
    assert count == 3

    record = _input_evidence(_within_bindings(tmp_path), _bindings(tmp_path)).records[0]
    authority = score.SessionInputAuthority(
        **{
            **record.__dict__,
            "target_last_bin_sha256": target_sha,
            "last_bin_valid_mask_sha256": mask_sha,
            "last_bin_valid_count": count,
            "n_windows": len(starts),
        },
    )
    score._validate_last_bin_authority_arrays(np, target=target, valid_mask=mask, authority=authority)
    changed_target = target.copy()
    changed_target[0, 0] += 1.0
    with pytest.raises(score.FailClosedError, match="last-bin target/mask authority"):
        score._validate_last_bin_authority_arrays(np, target=changed_target, valid_mask=mask, authority=authority)
    changed_mask = mask.copy()
    changed_mask[0] = 0
    with pytest.raises(score.FailClosedError, match="last-bin target/mask authority"):
        score._validate_last_bin_authority_arrays(np, target=target, valid_mask=changed_mask, authority=authority)
    with pytest.raises(ValueError, match="window/count"):
        score.SessionInputAuthority(**{**authority.__dict__, "last_bin_valid_count": count - 1})

    evidence = _input_evidence(_within_bindings(tmp_path), _bindings(tmp_path))
    payload = score._input_authority_payload(evidence, _identity())
    del payload["records"][0]["last_bin_valid_mask_sha256"]
    with pytest.raises(score.FailClosedError, match="nested evidence"):
        score.validate_input_authority_payload(payload, _identity())


def test_last_bin_and_full_window_dense_reference_semantics():
    target = [[[float(t), float(t + 1)] for t in range(50)], [[float(t + 1), float(t + 2)] for t in range(50)]]
    prediction = [[[value + 0.5 for value in row] for row in window] for window in target]
    mask = [[True] * 50, [True] * 48 + [False, False]]
    expected_last = score.variance_weighted_r2([prediction[0][-1], prediction[1][-3]], [target[0][-1], target[1][-3]])
    assert score.last_bin_r2(prediction, target, mask) == expected_last
    dense_p = [prediction[window][time] for window in range(2) for time in range(50) if mask[window][time]]
    dense_t = [target[window][time] for window in range(2) for time in range(50) if mask[window][time]]
    assert score.full_window_r2(prediction, target, mask) == score.variance_weighted_r2(dense_p, dense_t)


def test_mode_evidence_rejects_missing_eval_mask_repeat_or_b3s_proof():
    good = _mode("tfsr", "within", "aligned", _sha("input"))
    with pytest.raises(ValueError, match="repeat"):
        score.ModeEvidence(**{**good.__dict__, "repeat_bitwise_equal": False})
    with pytest.raises(ValueError, match="no-mask"):
        score.ModeEvidence(**{**good.__dict__, "eval_no_mask": False})
    with pytest.raises(ValueError, match="B3S"):
        score.ModeEvidence(**{**good.__dict__, "b3s_recomputed": False})


def test_authority_reader_rejects_sidecar_mode_and_symlink(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    body = b"{}\n"; item = root / "authority.json"; item.write_bytes(body); os.chmod(item, 0o444)
    digest = score._sha(body); side = root / "authority.json.sha256"; side.write_bytes(f"{digest}  authority.json\n".encode()); os.chmod(side, 0o444)
    spec = score.AuthoritySpec("synthetic", "authority.json", digest, 0o444, True)
    assert score.verify_authority(root, spec).body == body
    os.chmod(side, 0o644)
    with pytest.raises(score.FailClosedError, match="mode"):
        score.verify_authority(root, spec)
    os.chmod(side, 0o444); side.unlink(); side.symlink_to(item.name)
    with pytest.raises(score.FailClosedError):
        score.verify_authority(root, spec)
