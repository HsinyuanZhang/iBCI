"""Tests for cell W support in scripts/run_subpop_score.py (the matched scorer).

Covers HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3 "Cell W"
(output head and within-window structure, built to
HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md §5 Priority 2) as scored by the
sub-population matched scorer:

- W terminal-receipt validation: a faithful fixture (shaped on the sealed W
  smoke terminal receipt) passes, including the strict=False canonical-load
  discipline (missing keys EXACTLY the 8 W-head keys, unexpected keys empty,
  shared-subset byte equality), the zero-init bitwise launch proofs, the
  added-parameter manifest and the delta wake-up diagnostics; every tampered
  field refuses; a missing receipt stays ``not_landed`` (non-fatal);
- the W curve exclusion: the Step 0C wrapper replicates the PARENT decode
  path and demonstrably DROPS the W residual (the wrong graph), so a W curve
  must raise — ``refuse_cell_curve`` — and W is structurally excluded by
  ``curve_cells_for``;
- the W model wiring: ``build_cell_model`` builds the temporal-residual
  graph, the scorer's strip-then-strict-load path works on it (verified
  against the sealed W smoke SWA when present), the delta is exactly zero at
  initialisation and there is no perturbation to leak;
- the W reading assembly on synthetic contrasts: labelled
  EXPLORATORY__NO_PREREGISTERED_GATE, verdict-free, both granularities, with
  the seeds-43/44 + Z4-sibling adoption note;
- the T/G p-stream SHA-equality assertion is unaffected by W's presence (W
  has no p stream at all and records n/a);
- a --dry-run shows W as ``not_landed`` before it lands (never a silent
  score of the partial launch-receipt-only directory).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat as stat_mod
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "subpop_score_w_under_test", ROOT / "scripts/run_subpop_score.py"
)
score = importlib.util.module_from_spec(spec)
sys.modules["subpop_score_w_under_test"] = score
spec.loader.exec_module(score)

matched_scorer = score._load_module(
    "tfpd_lane_matched_scorer_subpop_score_w_test",
    ROOT / "src/tfpd_lane/matched_scorer.py",
)
trc = score._temporal_residual_module()

W_SMOKE_TERMINAL = (
    ROOT / "results/subpop_v1/cellW_temporal_residual_smoke/terminal_receipt.json"
)
W_SMOKE_SWA = ROOT / "results/subpop_v1/cellW_temporal_residual_smoke/swa_final4.pt"
CANONICAL = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ArmCommonStub:
    sha256_file = staticmethod(_sha_file)


_CANONICAL_STUB = {"state_sha256": "canonical-state-sha"}
_THETA_STUB = {"authority_sha256": "theta-authority-sha"}


# ---------------------------------------------------------------------------
# faithful W terminal-receipt fixture (shaped on the sealed W smoke receipt)
# ---------------------------------------------------------------------------
def _w_manifest() -> dict:
    return {
        "names": sorted(trc.W_HEAD_STATE_KEYS),
        "components": [
            {"name": key, "shape": [1], "numel": 1}
            for key in sorted(trc.W_HEAD_STATE_KEYS)
        ],
        "total_parameters": trc.W_HEAD_PARAMETER_COUNT,
        "total_trainable_head_parameters": trc.W_HEAD_PARAMETER_COUNT,
        "counts_match": True,
    }


def _w_probe(woke: bool = True) -> dict:
    return {
        "delta_abs_mean": 0.0058 if woke else 0.0,
        "delta_abs_max": 0.046 if woke else 0.0,
        "delta_exactly_zero": not woke,
        "delta_to_base_scale_ratio": 0.10 if woke else 0.0,
        "base_abs_mean": 0.0548,
        "w_head_param_norms": {
            key: 1.0 for key in sorted(trc.W_HEAD_STATE_KEYS)
        },
        "value_head_weight_max_abs": 0.00056 if woke else 0.0,
        "value_head_bias_max_abs": 0.0003 if woke else 0.0,
    }


def _w_integrity() -> dict:
    return {
        "cell": "W",
        "num_heads": 2,
        "head_structure": {
            "unit_tokens": "fc_in(activity + identity) - the existing path, unchanged",
            "latent_k": (
                "cross_attention(K=8 learned queries, unit_tokens); ONE "
                "nn.MultiheadAttention(512, 2, batch_first=True), dropout=0.0"
            ),
            "delta": "delta[t, c] = sum_k temporal_basis[t, k] * value_head(latent_k)[k, c]",
            "prediction": "base_output + delta",
            "k_latent_frozen": 8,
            "k_sweep": "none (K = 8 frozen)",
            "value_head": "nn.Linear(512, 2), weight AND bias zero-initialized",
        },
        "initialization": {
            "value_head.weight": "zeros -> delta EXACTLY zero at init",
            "value_head.bias": "zeros -> delta EXACTLY zero at init",
            "queries": "normal(std=512**-0.5)",
            "temporal_basis": "standard normal [window_size=50, K=8]",
            "cross_attention": "PyTorch default, dropout=0.0",
        },
        "zero_init_guarantee": (
            "delta is exactly zero at initialization, so the W forward is "
            "bitwise equal to the Arm A parent path"
        ),
        "non_causal_caveat": trc.NON_CAUSAL_CAVEAT,
        "gradient_flow": trc.GRADIENT_FLOW_NOTE,
        "rng_note": (
            "the new head consumes no RNG (cross_attention dropout = 0.0)"
        ),
        "loss": (
            "EXACTLY Arm A's masked MSE: ((diff2 * valid_rows).sum() / "
            "(valid_rows.sum() * C)); no masking, no consistency term, no "
            "dropout beyond the built-in decoder train-mode dropout"
        ),
        "dropout_structure": {
            "tf_drop_rate": 0.1, "decoder_dropout_rate": 0.0,
            "dynamic_dropout": False,
            "temporal_head_cross_attention_dropout": 0.0,
        },
        "w_head_parameters": _w_manifest(),
        "behavior_scaling_convention": (
            "unscaled standardized behavior (exact Arm A replica)"
        ),
    }


def _w_diagnostics(woke: bool = True) -> list[dict]:
    return [
        {"epoch": 46, "temporal_head_probe": _w_probe(woke=True),
         "value_head_woke": True},
        {"epoch": 47, "temporal_head_probe": _w_probe(woke=woke),
         "value_head_woke": woke},
    ]


def _w_fixture(tmp_path: Path, tamper: str | None = None):
    """Faithful landed-cell-W fixture: 0444 receipt + sidecar + sealed SWA."""
    directory = tmp_path / "results/subpop_v1" / score.CELL_SPECS["W"]["directory"]
    directory.mkdir(parents=True)
    swa = directory / "swa_final4.pt"
    swa.write_bytes(b"w-swa-bytes")
    swa_sha = _sha_file(swa)
    if tamper == "swa_bytes":
        swa.write_bytes(b"tampered")
    integrity = _w_integrity()
    diagnostics = _w_diagnostics()
    initial_state = {
        "path": str(CANONICAL),
        "artifact_sha256": _sha_file(CANONICAL),
        "state_dict_sha256": "canonical-state-sha",
        "strict_load": False,
        "missing_keys_exactly_w_head": True,
        "unexpected_keys_empty": True,
        "missing_keys": sorted(trc.W_HEAD_STATE_KEYS),
        "added_keys": sorted(trc.W_HEAD_STATE_KEYS),
        "shared_subset_state_sha256": "canonical-state-sha",
        "shared_subset_matches_artifact_state_sha256": True,
        "compared": "state SHA recomputed over the canonical-named key subset",
        "graph_parity_vs_canonical": {
            "shared_keys_shapes_equal": True,
            "added_keys": sorted(trc.W_HEAD_STATE_KEYS),
        },
        "eval_path_bitwise_parent_equal": True,
        "zero_init_delta_exactly_zero": True,
    }
    launch_proofs = {
        "zero_init_bitwise": {
            "torch_equal": True, "tensor_sha256_equal": True,
            "raw_bytes_equal": True, "delta_exactly_zero": True,
            "delta_nonzero_count": 0, "delta_abs_mean": 0.0,
            "delta_abs_max": 0.0, "prediction_shape": [8, 50, 2],
            "bitwise_convention": "torch.equal AND equal tensor_sha256",
        }
    }
    payload = {
        "schema": "tfpd_temporal_residual_cell_v1",
        "status": "CELL_TERMINAL",
        "cell": "W",
        "cell_name": score.CELL_SPECS["W"]["directory"],
        "integrity": integrity,
        "smoke": False,
        "max_train_steps": None,
        "epochs_run": 48,
        "invariant_failures": [],
        "source_closure": {"launch_final_closure_equal": True},
        "initial_state": initial_state,
        "launch_proofs": launch_proofs,
        "data_contract": {
            "behavior_normalizer_semantic_sha256":
                "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391",
            "roster_n": 27,
        },
        "diagnostics_per_epoch": diagnostics,
        "swa": {"sha256": swa_sha, "window_epochs": [44, 45, 46, 47]},
    }
    # ---- tampers ------------------------------------------------------------
    if tamper == "status":
        payload["status"] = "CELL_FAILED"
    elif tamper == "smoke_flag":
        payload["smoke"] = True
    elif tamper == "epochs":
        payload["epochs_run"] = 5
    elif tamper == "swa_sha":
        payload["swa"]["sha256"] = "0" * 64
    elif tamper == "strict_load":
        payload["initial_state"]["strict_load"] = True
    elif tamper == "missing_keys":
        payload["initial_state"]["missing_keys"] = (
            sorted(trc.W_HEAD_STATE_KEYS)[:-1])
        payload["initial_state"]["missing_keys_exactly_w_head"] = False
    elif tamper == "missing_keys_flag":
        payload["initial_state"]["missing_keys_exactly_w_head"] = False
    elif tamper == "shared_subset":
        payload["initial_state"][
            "shared_subset_matches_artifact_state_sha256"] = False
    elif tamper == "zero_init_delta":
        payload["initial_state"]["zero_init_delta_exactly_zero"] = False
    elif tamper == "zero_init_proof":
        payload["launch_proofs"]["zero_init_bitwise"]["torch_equal"] = False
    elif tamper == "never_woke":
        payload["diagnostics_per_epoch"] = _w_diagnostics(woke=False)
    elif tamper == "manifest_counts":
        payload["integrity"]["w_head_parameters"]["counts_match"] = False
    elif tamper == "manifest_names":
        payload["integrity"]["w_head_parameters"]["names"] = (
            sorted(trc.W_HEAD_STATE_KEYS)[:-1])
    elif tamper == "manifest_total":
        payload["integrity"]["w_head_parameters"]["total_parameters"] = 8
    elif tamper == "manifest_missing":
        del payload["integrity"]["w_head_parameters"]
    elif tamper == "k_latent":
        payload["integrity"]["head_structure"]["k_latent_frozen"] = 4
    elif tamper == "k_sweep":
        payload["integrity"]["head_structure"]["k_sweep"] = "grid 4/8/16"
    elif tamper == "value_head_init":
        payload["integrity"]["head_structure"]["value_head"] = "default init"
    elif tamper == "gradient_flow":
        del payload["integrity"]["gradient_flow"]
    elif tamper == "zero_init_guarantee":
        del payload["integrity"]["zero_init_guarantee"]
    elif tamper == "loss_masking":
        payload["integrity"]["loss"] = "consistency-masked MSE with unit dropout"
    elif tamper == "scaling":
        payload["integrity"]["behavior_scaling_convention"] = "scaled x5"
    body = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    receipt = directory / "terminal_receipt.json"
    receipt.write_text(body)
    os.chmod(receipt, stat_mod.S_IRUSR | stat_mod.S_IRGRP | stat_mod.S_IROTH)
    sidecar = directory / "terminal_receipt.json.sha256"
    sidecar_text = _sha_file(receipt) + "  terminal_receipt.json\n"
    if tamper == "sidecar":
        sidecar_text = "0" * 64 + "  terminal_receipt.json\n"
    sidecar.write_text(sidecar_text)
    os.chmod(sidecar, stat_mod.S_IRUSR | stat_mod.S_IRGRP | stat_mod.S_IROTH)
    if tamper == "mode":
        os.chmod(receipt, 0o644)
    return receipt


# ---------------------------------------------------------------------------
# W terminal-receipt validation
# ---------------------------------------------------------------------------
def test_validate_w_terminal_passes_on_faithful_fixture(tmp_path):
    _w_fixture(tmp_path)
    record = score.validate_cell_terminal(
        "W", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
    )
    assert record["cell"] == "W"
    assert record["cell_name"] == "cellW_temporal_residual"
    assert record["mode_0444"] and record["non_symlink"]
    assert record["launch_final_closure_equal"]
    assert record["initial_state_reconciled"]
    assert record["normalizer_reconciled_f062506c"]
    assert record["epochs_run"] == 48 and record["num_heads"] == 2
    # W has NO p stream: the receipt states n/a rather than an absent field
    assert record["p_sequence_sha256"].startswith("n/a (no p stream")
    assert str(record["p_stream_total_draws"]).startswith("n/a (no p stream")
    assert record["p_draws_per_step"] == 0
    assert "output head" in record["theta_authority_reconciled"]
    # the W head contract is carried in the perturbation-law record
    law = record["perturbation_law"]
    for field in ("head_structure", "initialization", "zero_init_guarantee",
                  "gradient_flow", "w_head_parameters", "non_causal_caveat",
                  "loss", "rng_note"):
        assert field in law, field
    assert law["head_structure"]["k_latent_frozen"] == 8
    assert law["w_head_parameters"]["counts_match"] is True
    # the delta wake-up diagnostics (the residual leaving exact zero)
    stats = record["realized_perturbation_statistics"]
    assert stats["available"] is True and stats["n_epochs"] == 2
    assert stats["final_epoch"]["value_head_woke"] is True
    assert stats["per_epoch_series"][0]["delta_abs_mean"] > 0


def test_validate_w_terminal_missing_receipt_means_not_landed(tmp_path):
    record = score.validate_cell_terminal(
        "W", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
    )
    assert record is None  # recorded as not_landed and omitted, never fatal


@pytest.mark.parametrize(
    "tamper",
    ["status", "smoke_flag", "epochs", "sidecar", "mode", "swa_sha",
     "swa_bytes", "strict_load", "missing_keys", "missing_keys_flag",
     "shared_subset", "zero_init_delta", "zero_init_proof", "never_woke",
     "manifest_counts", "manifest_names", "manifest_total", "manifest_missing",
     "k_latent", "k_sweep", "value_head_init", "gradient_flow",
     "zero_init_guarantee", "loss_masking", "scaling"],
)
def test_validate_w_terminal_refuses_each_tampering(tmp_path, tamper):
    _w_fixture(tmp_path, tamper=tamper)
    with pytest.raises(SystemExit):
        score.validate_cell_terminal(
            "W", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
        )


def test_w_head_keys_come_from_the_route_module_never_duplicated():
    assert sorted(score.w_head_state_keys()) == sorted(trc.W_HEAD_STATE_KEYS)
    assert len(score.w_head_state_keys()) == 8  # the 8-key W-head missing set
    assert score.w_head_parameter_count() == 1_056_146


# ---------------------------------------------------------------------------
# the W curve exclusion (the wrong-graph path cannot run)
# ---------------------------------------------------------------------------
def test_refuse_cell_curve_raises_for_w_and_only_w():
    with pytest.raises(SystemExit, match="cell W curve refused"):
        score.refuse_cell_curve("W")
    score.refuse_cell_curve("T")     # perturbation cells curve normally
    score.refuse_cell_curve("C")
    score.refuse_cell_curve("G")
    score.refuse_cell_curve(None)    # armA / D checkpoints
    assert "DROPS the W temporal residual" in score.CURVE_EXCLUDED_CELLS["W"]


def test_curve_cells_for_structurally_excludes_w():
    curve_cells, excluded = score.curve_cells_for(["T", "C", "W"])
    assert curve_cells == ["T", "C"]
    assert set(excluded) == {"W"} and excluded["W"]
    assert score.curve_cells_for(["T", "G"]) == (["T", "G"], {})
    assert score.curve_cells_for([]) == ([], {})


def test_the_step0c_wrapper_is_demonstrably_the_wrong_graph_for_w():
    """Why W is native-only: the 0C wrapper replicates the PARENT decode and
    silently drops the W residual.  Constructing it on a W model succeeds (no
    structural protection there), so the scorer's refuse_cell_curve guard is
    what must stand between W and the curve."""
    step0c = score._load_module(
        "tfpd_subpop_step0c_machinery_w_test", ROOT / "scripts/run_subpop_step0c.py"
    )
    model = score.build_cell_model(None, None, "W", trc).eval()
    # wake the residual: value_head nonzero -> delta != 0 (post-training state)
    with torch.no_grad():
        model.temporal_head.value_head.weight.normal_(std=0.05)
        model.temporal_head.value_head.bias.normal_(std=0.05)
    wrapper = step0c.SubpopEvalModel(model, min_keep=step0c.MIN_KEEP).eval()
    torch.manual_seed(0)
    neural = torch.randn(2, 50, 12)
    calib = torch.randn(2, 30, 100, 12)
    side = torch.randn(2, 12, 4)
    with torch.no_grad():
        mine, _ = model(neural, calib_trials=calib, side_features=side)
        wrapped, _ = wrapper(neural, calib_trials=calib, side_features=side)
        delta = model.delta_only(
            neural, model.compute_identity(calib, side_features=side)
        )
    assert int(torch.count_nonzero(delta).item()) > 0   # the residual is live
    assert not torch.equal(mine, wrapped)               # the wrapper DROPS it
    # and the scorer refuses exactly this
    with pytest.raises(SystemExit, match="curve refused"):
        score.refuse_cell_curve("W")


def test_the_curve_loop_carries_the_guard_and_the_exclusion():
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    assert 'refuse_cell_curve(spec["cell"])' in source
    assert "curve_cells_for(landed)" in source
    assert '"cells_native_only"' in source
    assert '"native_only_cells": dict(CURVE_EXCLUDED_CELLS)' in source


# ---------------------------------------------------------------------------
# the W model wiring (build + strip + strict load, native-only scoring)
# ---------------------------------------------------------------------------
def _tiny_inputs(n_units=12, batch=2):
    torch.manual_seed(0)
    return (
        torch.randn(batch, 50, n_units),
        torch.randn(batch, 30, 100, n_units),
        torch.randn(batch, n_units, 4),
    )


def test_build_cell_model_w_builds_the_temporal_residual_graph():
    model = score.build_cell_model(None, None, "W", trc)
    assert isinstance(model, trc.TemporalResidualStreamingSpintModel)
    added = {k for k in model.state_dict() if k.startswith("temporal_head.")}
    assert sorted(added) == sorted(trc.W_HEAD_STATE_KEYS)
    assert score.perturbation_disabled(model)  # W has no perturbation switches
    neural, calib, side = _tiny_inputs()
    model.eval()
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        delta = model.delta_only(neural, identity)
    assert int(torch.count_nonzero(delta).item()) == 0  # zero-init residual


def test_w_load_path_strict_loads_the_sealed_w_smoke_swa():
    """The scorer's exact load path (strip 'model.' conditionally, strict load)
    against the sealed W smoke SWA — the same state layout the live W run
    writes to results/subpop_v1/cellW_temporal_residual/swa_final4.pt."""
    if not W_SMOKE_SWA.is_file():
        pytest.skip("W smoke SWA not present in this checkout")
    torch.serialization.add_safe_globals(
        [torch.nn.parameter.UninitializedParameter]
    )
    state = torch.load(W_SMOKE_SWA, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    state = {(k[len("model."):] if k.startswith("model.") else k): v
             for k, v in state.items()}
    assert not any(k.startswith("model.") for k in state)  # strip is a no-op
    model = score.build_cell_model(None, None, "W", trc)
    model.load_state_dict(state, strict=True)  # the W keys are present
    model.eval()
    neural, calib, side = _tiny_inputs()
    with torch.no_grad():
        behavior, _ = model(neural, calib_trials=calib, side_features=side)
        base, delta = model.decode_components(
            neural, model.compute_identity(calib, side_features=side)
        )
    assert bool(torch.isfinite(behavior).all())
    assert int(torch.count_nonzero(delta).item()) > 0  # the trained head is live


def test_w_canonical_load_uses_the_eight_key_missing_set():
    """The strict=False discipline the terminal receipt records: loading the
    canonical Arm A state into the W graph leaves EXACTLY the 8 W-head keys
    missing and nothing unexpected."""
    if not CANONICAL.is_file():
        pytest.skip("canonical initial state not present in this checkout")
    payload = torch.load(CANONICAL, map_location="cpu", weights_only=False)
    model = score.build_cell_model(None, None, "W", trc)
    incompat = model.load_state_dict(payload["state_dict"], strict=False)
    assert sorted(incompat.missing_keys) == sorted(trc.W_HEAD_STATE_KEYS)
    assert incompat.unexpected_keys == []
    report = trc.load_canonical_initial_state(
        model, payload["state_dict"], payload["state_sha256"]
    )
    assert report["strict_load"] is False
    assert report["missing_keys_exactly_w_head"] is True


# ---------------------------------------------------------------------------
# the W reading: exploratory, verdict-free, both granularities
# ---------------------------------------------------------------------------
def _synthetic_contrasts() -> dict:
    stats = matched_scorer.paired_session_stats(
        [0.05, 0.02, -0.01, 0.03, 0.04, 0.01, -0.02, 0.06, 0.0, 0.03, 0.02,
         0.05, 0.01, 0.04, 0.02], seed=42)
    contrasts = {}
    for granularity in ("governing_last_bin", "diagnostic_full_window"):
        for model_a, model_b in (("W_swa", "armA_swa"), ("W_swa", "D_swa")):
            for surface in ("within", "external"):
                contrasts[f"{model_a}_minus_{model_b}_{surface}_{granularity}"] = {
                    **stats,
                    "contrast": f"{model_a} - {model_b} ({surface}, {granularity})",
                }
    return contrasts


def test_w_reading_is_exploratory_and_verdict_free():
    reading = score.w_reading(_synthetic_contrasts())
    assert reading["label"] == score.W_READING_LABEL == (
        "EXPLORATORY__NO_PREREGISTERED_GATE"
    )
    assert reading["verdict"] is None and reading["pass"] is None  # verdict-free
    assert "no gate was pre-registered" in reading["rule"]
    # both granularities, both references (Arm A and D, live same-engine)
    for key in ("w_minus_armA_external", "w_minus_armA_within",
                "w_minus_d_external", "w_minus_d_within"):
        assert reading[key]["n_total"] == 15
        assert "bootstrap_95_interval" in reading[key]  # descriptive only
    diagnostic = reading["diagnostic_full_window"]
    assert set(diagnostic) == {
        "w_minus_armA_external", "w_minus_armA_within",
        "w_minus_d_external", "w_minus_d_within"}
    for key, block in diagnostic.items():
        assert "diagnostic_full_window" in block["contrast"]
        surface = "external" if key.endswith("_external") else "within"
        assert f"({surface}," in block["contrast"]
    # the adoption note names the required follow-up from the directions handoff
    assert "seeds 43/44" in reading["adoption_note"]
    assert "Z4 sibling" in reading["adoption_note"]
    assert "no adoption decision may be taken from this receipt" in (
        reading["adoption_note"]
    )
    # both granularities are reported for W specifically
    assert "within-window profile" in reading["granularity_note"]


def test_w_reading_refuses_a_missing_contrast():
    contrasts = _synthetic_contrasts()
    del contrasts["W_swa_minus_D_swa_within_diagnostic_full_window"]
    with pytest.raises(SystemExit, match="missing contrast"):
        score.w_reading(contrasts)


def test_w_has_no_preregistered_gate_in_the_scorer_source():
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    # the pre-registered rules stay T/G/C only; W appears solely as the
    # exploratory reading
    assert "W_TRICHOTOMY" not in source and "W_GATE" not in source
    assert "W_BAND" not in source
    assert score.W_READING_LABEL == "EXPLORATORY__NO_PREREGISTERED_GATE"


# ---------------------------------------------------------------------------
# the T/G p-stream SHA assertion is unaffected by W's presence
# ---------------------------------------------------------------------------
def test_t_g_p_stream_sha_assertion_unaffected_by_w_presence():
    good = {
        "T": {"p_sequence_sha256": "a" * 64, "p_draws_per_step": 1},
        "G": {"p_sequence_sha256": "a" * 64, "p_draws_per_step": 1},
        "C": {"p_sequence_sha256": "c" * 64, "p_draws_per_step": 2},
        "W": {
            "p_sequence_sha256": "n/a (no p stream: cell W is an output-head change; no masking)",
            "p_draws_per_step": 0,
        },
    }
    score.verify_p_stream_binding(good)  # W present, same SHAs -> no raise
    score.verify_p_stream_binding({"T": good["T"], "W": good["W"]})  # no G
    score.verify_p_stream_binding({"W": good["W"]})                  # W alone
    bad = {**good, "G": {"p_sequence_sha256": "b" * 64}}
    with pytest.raises(SystemExit, match="T/G p-stream SHA inequality"):
        score.verify_p_stream_binding(bad)


# ---------------------------------------------------------------------------
# dry-run: W is not_landed before it lands (never a partial-artifact score)
# ---------------------------------------------------------------------------
def test_dry_run_reports_w_not_landed_before_it_lands():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_subpop_score.py"),
         "--device", "cpu", "--dry-run", "--cells", "W"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRY_RUN__NO_NWB_OPENED"
    assert payload["cells_requested"] == ["W"]
    assert set(payload["curve"]["native_only_cells"]) == {"W"}
    # before the terminal receipt exists: not_landed, never scored; after it
    # lands: full provenance — never a silent partial score either way
    if "W" in payload["cells_not_landed"]:
        assert payload["cells_landed"] == []
        assert "terminal receipt not present yet" in (
            payload["not_landed"]["W"]["note"]
        )
    else:
        assert payload["cells_landed"] == ["W"]
        assert payload["landed_cell_provenance"]["W"]["swa_sha256"]
        assert payload["landed_cell_provenance"]["W"]["epochs_run"] == 48


def test_realized_perturbation_statistics_w_series_shape():
    stats = score.realized_perturbation_statistics("W", [
        {"epoch": 46, "temporal_head_probe": _w_probe(True), "value_head_woke": True},
        {"epoch": 47, "temporal_head_probe": _w_probe(True), "value_head_woke": True},
    ])
    assert stats["available"] is True and stats["n_epochs"] == 2
    row = stats["per_epoch_series"][0]
    for field in ("delta_abs_mean", "delta_abs_max", "delta_exactly_zero",
                  "delta_to_base_scale_ratio", "value_head_woke",
                  "value_head_weight_max_abs", "value_head_bias_max_abs"):
        assert field in row, field
    empty = score.realized_perturbation_statistics("W", [])
    assert empty["available"] is False and "no per-epoch diagnostics" in empty["note"]
