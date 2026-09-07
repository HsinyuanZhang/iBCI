"""No-data/no-CUDA unit tests for the static-pool APFG M2 EvalAI probe.

Covers the ``submissions/evalai_m2_apfg_static_v1`` package:

- the frozen alpha literal and its tamper-evident binding to the immutable
  V2 same-surface result graph (terminal + score sidecars on disk);
- the selection/activity/ridge law mirrors being constant- and
  behaviour-identical to the already officially scored ``act30_dopt4`` package;
- the APFG adapter semantics on a synthetic B3S stub: exact ``+0.0`` is a
  bitwise operational no-op and the frozen learned alpha reproduces the
  residual expression ``native + tanh(alpha) * (post - native)``;
- the decoder payload schema/arm isolation (no silent cross-serving) and the
  packaging wiring (Dockerfile/decode/exporter dry-run).

Only synthetic tensors are used; no checkpoint, no NWB data, no CUDA, no
network.  Torch-dependent tests skip when torch is unavailable (system
python3); the sealed ``spint`` interpreter runs them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
for _path in (REPO_ROOT, REPO_ROOT / "SPINT-main"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PACKAGE = TFPD_ROOT / "submissions/evalai_m2_apfg_static_v1"
ACT30_PACKAGE = TFPD_ROOT / "submissions/evalai_m2_act30_dopt4_v1"
V2_RESULT_ROOT = TFPD_ROOT / "results/m2_anchored_postfusion_gate_v2_same_surface_control"

FROZEN_ALPHA_LITERAL = -0.20759029686450958
V2_TERMINAL_SHA256 = (
    "9ea88d3a4e9048c484a4e67575a3b1c9a1323fbf0558d739acb065217f2a7b63"
)
V2_SCORE_SHA256 = (
    "5dd7c3e911ca00709e028590af93c065a38f5ea1bd1369dc27fabda87b595d6e"
)
PAYLOAD_SCHEMA = "e8_t4_m2_apfg_static_cached_identity_v1"
PAYLOAD_ARM = "apfg_static_act30_dopt4"
PAYLOAD_NAME = "t4_m2_seed42_dopt4_act30_apfg_identity.pkl"

_LAWS_CACHE: dict[str, object] = {}
_MODULE_CACHE: dict[str, object] = {}


def _load_file_module(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def laws():
    return _load_file_module(PACKAGE / "laws.py", "laws_apfg_static_v1")


def act30_laws():
    return _load_file_module(ACT30_PACKAGE / "laws.py", "laws_act30_dopt4_v1_mirror_source")


def exporter():
    return _load_file_module(
        PACKAGE / "export_apfg_static_payload.py", "export_apfg_static_payload_v1"
    )


def decoder_module():
    pytest.importorskip("torch")
    pytest.importorskip("falcon_challenge")
    return _load_file_module(
        PACKAGE / "apfg_static_decoder.py", "apfg_static_decoder"
    )


def _synthetic_angles(seed: int, n_pool: int = 30, finite: int = 12, total: int = 35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.full(total, np.nan, dtype=np.float64)
    positions = rng.permutation(n_pool)[:finite]
    angles[positions] = (
        -3.0 * np.pi / 4.0 + rng.integers(0, 8, finite) * (np.pi / 4.0)
        + rng.normal(0.0, 0.05, finite)
    )
    return angles


def _synthetic_calib(seed: int, trials: int = 33) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(trials, 100, 96)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Frozen alpha and its V2 provenance binding.
# ---------------------------------------------------------------------------


def test_frozen_alpha_literal_is_exact() -> None:
    module = laws()
    assert module.FROZEN_ALPHA == FROZEN_ALPHA_LITERAL
    assert np.float64(module.FROZEN_ALPHA) == np.float64(FROZEN_ALPHA_LITERAL)


def test_frozen_alpha_binds_the_immutable_v2_terminal_on_disk() -> None:
    module = laws()
    evidence = module.verify_v2_alpha_provenance(REPO_ROOT)
    assert evidence["terminal_sha256"] == V2_TERMINAL_SHA256
    assert evidence["score_sha256"] == V2_SCORE_SHA256
    assert evidence["terminal_sidecar_verified"] is True
    assert evidence["score_sidecar_verified"] is True
    assert evidence["learned_refit_alpha"] == FROZEN_ALPHA_LITERAL
    assert evidence["row_count"] == 65
    assert evidence["parameter_updates"] == 0
    assert evidence["target_updates"] == 0
    assert evidence["cuda_initialized"] is False
    assert evidence["terminal_xor_failure"] is True
    assert evidence["schema"] == (
        "m2_anchored_postfusion_gate_v2_same_surface_control_terminal_v1"
    )


def test_v2_binding_rejects_alpha_drift(tmp_path: Path) -> None:
    """A mutated terminal body must fail the sidecar and the alpha check."""
    import os
    import shutil

    (tmp_path / "tfpd_exploration/results").mkdir(parents=True)
    destination = tmp_path / V2_RESULT_ROOT.relative_to(TFPD_ROOT)
    shutil.copytree(V2_RESULT_ROOT, destination, copy_function=shutil.copy2)
    body = destination / "terminal.json"
    payload = json.loads(body.read_text(encoding="utf-8"))
    payload["learned_refit_alpha"] = -0.5
    body.chmod(0o644)
    body.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    module = laws()
    with pytest.raises(module.LawError):
        module.verify_v2_alpha_provenance(tmp_path)


def test_v2_terminal_carries_the_six_body_v1_predecessor() -> None:
    module = laws()
    body = json.loads((V2_RESULT_ROOT / "terminal.json").read_text(encoding="utf-8"))
    predecessor = body["v1_failure_predecessor"]
    assert set(predecessor) == set(module.V1_PREDECESSOR_BODIES)
    for name, sha in module.V1_PREDECESSOR_BODIES.items():
        assert predecessor[name] == sha


# ---------------------------------------------------------------------------
# 2. Law mirrors stay identical to the officially scored act30_dopt4 package.
# ---------------------------------------------------------------------------


def test_constants_match_the_officially_scored_act30_dopt4_law() -> None:
    module, mirror = laws(), act30_laws()
    for name in (
        "ACTIVITY_HORIZON",
        "RIDGE_NORMALIZED_LAMBDA",
        "CHANNELS",
        "TRIAL_LENGTH",
        "SUPPORT_BUDGET",
        "ACTIVITY_BUDGET",
    ):
        assert getattr(module, name) == getattr(mirror, name), name


@pytest.mark.parametrize("seed", (0, 1, 2, 3, 4))
def test_dopt4_selection_is_behaviour_identical_to_act30_dopt4(seed: int) -> None:
    module, mirror = laws(), act30_laws()
    angles = _synthetic_angles(seed)
    assert np.array_equal(module.select_dopt4_support(angles), mirror.select_dopt4_support(angles))


def test_activity_pool_is_behaviour_identical_to_act30_dopt4() -> None:
    module, mirror = laws(), act30_laws()
    calibration = _synthetic_calib(7)
    assert np.array_equal(
        module.select_first30_activity_pool(calibration),
        mirror.select_first30_activity_pool(calibration),
    )


def test_selection_fails_closed_below_four_candidates() -> None:
    module = laws()
    angles = np.full(35, np.nan)
    angles[[3, 9, 21]] = [0.0, np.pi / 2, np.pi]
    with pytest.raises(module.LawError):
        module.select_dopt4_support(angles)


def test_selection_fails_closed_without_first30_metadata() -> None:
    module = laws()
    with pytest.raises(module.LawError):
        module.select_dopt4_support(np.full(29, np.nan))


# ---------------------------------------------------------------------------
# 3. APFG adapter semantics on a synthetic B3S stub.
# ---------------------------------------------------------------------------


class _StubNative:
    """Minimal B3S-shaped stand-in: deterministic, input-dependent, eval-only."""

    variant = "B3S"
    trial_length = 100
    window_size = 50
    side_dim = 4
    electrode_embed_dim = 0
    hidden_dim = 8

    def __init__(self, neurons: int = 6) -> None:
        import torch

        self._torch = torch
        self._neurons = neurons
        rng = torch.Generator().manual_seed(11)
        self._w = torch.randn(1 + self.side_dim, self.window_size, generator=rng) / 16.0

    def forward_batch(self, calib_trials, trial_lengths=None, side_features=None, electrode_ids=None):
        import torch

        pooled = calib_trials.mean(dim=(1, 2))  # [B, N]
        window = pooled.unsqueeze(-1).expand(-1, self._neurons, self.window_size) / 4.0
        if side_features is not None:
            window = window + side_features.mean(dim=-1, keepdim=True) * 0.125
        return window.contiguous()

    def pre_pool(self, calib_trials):
        # Matches the real B3S contract: [B,M,N,100] in -> [B,M,N,1] per-trial rate.
        return calib_trials.mean(dim=3, keepdim=True)

    def post_pool(self, features):
        import torch

        batch, trials, neurons, dim = features.shape
        flat = features.reshape(batch * trials * neurons, dim)
        projected = flat @ self._w.to(flat.dtype)
        # [B,M,N,W]; the adapter performs the trial reduction itself.
        return projected.reshape(batch, trials, neurons, self.window_size)


@pytest.mark.parametrize("alpha_value,expect_noop", ((0.0, True), (FROZEN_ALPHA_LITERAL, False)))
def test_adapter_zero_alpha_is_bitwise_native_and_learned_is_residual(
    alpha_value: float, expect_noop: bool
) -> None:
    torch = pytest.importorskip("torch")
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import adapter as adapter_source
    native = _StubNative()
    adapter = adapter_source.AnchoredPostFusionGate(native)
    generator = torch.Generator().manual_seed(5)
    calib = torch.randn(2, 7, 100, 6, generator=generator)
    side = torch.randn(2, 6, 4, generator=generator)

    with torch.no_grad():
        adapter.alpha.fill_(alpha_value)
        learned = adapter.forward_batch(calib, side_features=side)
        native_reference = native.forward_batch(calib, side_features=side)
    if expect_noop:
        assert torch.equal(learned, native_reference)
    else:
        assert not torch.equal(learned, native_reference)
        post = adapter._postfusion_identity(calib, side)
        residual = native_reference + torch.tanh(adapter.alpha) * (post - native_reference)
        assert torch.equal(learned, residual)


def test_adapter_alpha_literal_float32_round_trip_is_stable() -> None:
    torch = pytest.importorskip("torch")
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import adapter as adapter_source
    adapter = adapter_source.AnchoredPostFusionGate(_StubNative())
    with torch.no_grad():
        adapter.alpha.fill_(FROZEN_ALPHA_LITERAL)
    assert float(adapter.alpha.item()) == FROZEN_ALPHA_LITERAL
    assert adapter_source.exact_positive_zero(adapter.alpha) is False


def test_exact_positive_zero_rejects_negative_zero_and_nonscalar() -> None:
    torch = pytest.importorskip("torch")
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import adapter as adapter_source
    adapter = adapter_source.AnchoredPostFusionGate(_StubNative())
    with torch.no_grad():
        adapter.alpha.fill_(0.0)
    assert adapter_source.exact_positive_zero(adapter.alpha) is True
    with torch.no_grad():
        adapter.alpha.fill_(-0.0)
    assert adapter_source.exact_positive_zero(adapter.alpha) is False
    with pytest.raises(adapter_source.AdapterError):
        adapter_source.exact_positive_zero(torch.zeros(1))


# ---------------------------------------------------------------------------
# 4. Decoder payload isolation and packaging wiring.
# ---------------------------------------------------------------------------


def test_decoder_schema_and_arm_are_package_local() -> None:
    module = decoder_module()
    assert module.PAYLOAD_SCHEMA == PAYLOAD_SCHEMA
    assert module.PAYLOAD_ARM == PAYLOAD_ARM
    assert module.EXPECTED_SESSION_COUNT == 13
    act30 = _load_file_module(
        ACT30_PACKAGE / "act30_dopt4_decoder.py", "act30_dopt4_decoder_mirror"
    )
    assert module.PAYLOAD_SCHEMA != act30.PAYLOAD_SCHEMA
    assert module.PAYLOAD_ARM != act30.PAYLOAD_ARM


class _PicklableDummyDecoder:
    """Module-level stand-in so pickle can bind it (real payloads carry a nn.Module)."""

    def parameters(self):
        return iter(())

    def eval(self):
        return self


def _minimal_payload(tmp_path: Path, task: object = "falcon_m2_reach", **metadata_overrides) -> Path:
    torch = pytest.importorskip("torch")

    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "task": task,
        "decoder": _PicklableDummyDecoder(),
        "identity_by_dataset_tag": {},
        "window_size": 50,
        "behavior_scaling_factor": 5.0,
        "smooth_observations": False,
        "metadata": {
            "arm": PAYLOAD_ARM,
            "label_budget": 4,
            "activity_budget": 30,
            "frozen_alpha": FROZEN_ALPHA_LITERAL,
            "session_count": 13,
            **metadata_overrides,
        },
    }
    path = tmp_path / "payload.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def test_decoder_rejects_foreign_arm_and_schema(tmp_path: Path) -> None:
    import pickle

    module = decoder_module()
    from falcon_challenge.config import FalconConfig, FalconTask

    config = FalconConfig(task=FalconTask.m2)
    foreign_arm = _minimal_payload(tmp_path, task=FalconTask.m2, arm="dopt4_static_act30")
    with pytest.raises(ValueError, match="arm"):
        module.ApfgStaticM2Decoder(task_config=config, model_path=str(foreign_arm))

    foreign_schema_path = tmp_path / "foreign_schema.pkl"
    foreign_schema_path.write_bytes(_minimal_payload(tmp_path / "seed", task=FalconTask.m2).read_bytes())
    payload = pickle.loads(foreign_schema_path.read_bytes())
    payload["schema_version"] = "e8_t4_m2_cached_identity_v1"
    with foreign_schema_path.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="payload"):
        module.ApfgStaticM2Decoder(task_config=config, model_path=str(foreign_schema_path))


def test_decode_entrypoint_wires_the_apfg_decoder() -> None:
    text = (PACKAGE / "decode.py").read_text(encoding="utf-8")
    assert "apfg_static_decoder" in text
    assert "ApfgStaticM2Decoder" in text


def test_dockerfile_binds_payload_checkpoint_and_labels() -> None:
    text = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    assert f"artifacts/{PAYLOAD_NAME}" in text
    assert "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e" in text
    assert "falcon_challenge==1.0.2" in text or "BASE_IMAGE" in text
    assert "apfg" in text.lower()


def test_exporter_dry_run_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / PAYLOAD_NAME
    completed = subprocess.run(
        [sys.executable, str(PACKAGE / "export_apfg_static_payload.py"), "--output", str(output)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["status"].startswith("DRY_")
    assert report["arm"] == PAYLOAD_ARM
    assert not output.exists()


def test_exporter_refuses_to_overwrite(tmp_path: Path) -> None:
    module = exporter()
    output = tmp_path / PAYLOAD_NAME
    output.write_bytes(b"existing")
    with pytest.raises(module.ExportError):
        module.require_output_absent(output)


@pytest.mark.skipif(
    not (PACKAGE / "artifacts" / PAYLOAD_NAME).exists(),
    reason="real payload not exported yet",
)
def test_exported_payload_and_receipt_are_bound() -> None:
    module = exporter()
    payload_path = PACKAGE / "artifacts" / PAYLOAD_NAME
    receipt = json.loads(
        (PACKAGE / "artifacts" / (PAYLOAD_NAME.replace(".pkl", ".receipt.json"))).read_text(
            encoding="utf-8"
        )
    )
    digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    assert receipt["payload_sha256"] == digest
    assert receipt["arm"] == PAYLOAD_ARM
    assert receipt["frozen_alpha"] == FROZEN_ALPHA_LITERAL
    assert receipt["session_count"] == 13
    assert receipt["zero_equals_native_sessions"] == 13
    for record in receipt["session_records"].values():
        assert record["zero_identity_sha256"] == record["native_identity_sha256"]
        assert record["learned_identity_sha256"] != record["native_identity_sha256"]
