"""Test suite for Cell W, the temporal latent residual decoder (pre-launch gate).

Covers HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3 (cell W)
built to HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md §5 Priority 2:

- the exact added-parameter set (8 keys under ``temporal_head.``), count
  (1,056,146) and shapes;
- the canonical-initial-state discipline: strict=False load with missing keys
  EXACTLY the W-head keys, unexpected keys empty, shared-subset byte equality
  (arm_common.state_sha256 semantics over the canonical-named subset) —
  against the sealed artifact AND against small tampered/extended fixtures;
- the core property: bitwise equality to the plain parent decode path at
  initialization (torch.equal AND equal signed-zero-normalized tensor SHA),
  delta EXACTLY zero at init, in eval and train modes;
- the delta formula and [B, W, C] layout, verified against a manual einsum;
- wake-up: value_head weight/bias receive nonzero gradient, delta becomes
  nonzero after one optimizer step; queries/temporal_basis/cross-attention
  receive EXACTLY zero gradients at init (documented zero-init residual
  topology), stay bitwise unchanged after step 1, and are live at step 2;
- permutation invariance over units (the only unit-mixing ops are the
  permutation-invariant attentions) on a fixture, up to floating-point
  summation order;
- the SWA path: build_swa_final_four over W checkpoints strict-reloads into
  the W model (extra keys present) with exact mean values;
- runner surface: fresh-directory refusal and frozen contract tokens.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# the sibling roots are merged into the same `src` package by
# temporal_residual_cell._ensure_component_paths() at build time (the
# test_subpop_cells.py pattern); inserting them here would shadow src.tfpd_lane

from src.tfpd_lane import arm_common, matched_scorer
from src.tfpd_lane import temporal_residual_cell as trc

CANONICAL = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"


def _canonical_payload():
    return torch.load(CANONICAL, map_location="cpu", weights_only=False)


def _tiny_inputs(n_units=10, batch=3):
    torch.manual_seed(0)
    return (
        torch.rand(batch, 50, n_units),
        torch.rand(batch, 30, 100, n_units),
        torch.randn(batch, n_units, 4),
    )


def _built():
    """A W model loaded with the canonical initial bytes (W discipline)."""
    payload = _canonical_payload()
    model = trc.build_temporal_residual_model(seed=42)
    trc.load_canonical_initial_state(model, payload["state_dict"],
                                     payload["state_sha256"])
    return model


# ---- added parameters / graph -----------------------------------------------
def test_builder_determinism_and_head_key_set():
    a = trc.build_temporal_residual_model(seed=42)
    b = trc.build_temporal_residual_model(seed=42)
    # the head draw order is fixed after the parent draws: seed-deterministic
    assert arm_common.state_sha256(a) == arm_common.state_sha256(b)
    assert a.temporal_head.k_latent == 8 == trc.K_LATENT  # FROZEN, no sweep
    assert a.decoder.transformer.layers[0].cross_attn.num_heads == 2
    assert a.temporal_head.num_heads == 2
    assert a.temporal_head.cross_attention.dropout == 0.0  # no new dropout
    assert a.decoder.dynamic_dropout is False  # Arm A: route adds no masking
    assert a.decoder.tf_drop_rate == 0.1  # parent dropout structure unchanged
    # exactly ONE new MultiheadAttention; the decoder transformer is untouched
    new_mha = [m for m in a.temporal_head.modules()
               if isinstance(m, torch.nn.MultiheadAttention)]
    assert len(new_mha) == 1 and new_mha[0] is a.temporal_head.cross_attention
    # the added state keys are exactly the eight W-head keys
    assert sorted(trc.W_HEAD_STATE_KEYS) == list(trc.W_HEAD_STATE_KEYS)
    assert set(trc.W_HEAD_STATE_KEYS) == set(a.state_dict()) - set(
        _canonical_payload()["state_dict"]
    )


def test_head_parameter_manifest_counts_and_shapes():
    model = trc.build_temporal_residual_model(seed=42)
    manifest = trc.head_parameter_manifest(model)
    assert manifest["counts_match"] is True
    assert manifest["total_parameters"] == 1_056_146 == trc.W_HEAD_PARAMETER_COUNT
    by_name = {entry["name"]: entry for entry in manifest["components"]}
    assert by_name["temporal_head.queries"]["shape"] == [8, 512]
    assert by_name["temporal_head.queries"]["numel"] == 8 * 512
    assert by_name["temporal_head.temporal_basis"]["shape"] == [50, 8]
    assert by_name["temporal_head.temporal_basis"]["numel"] == 50 * 8
    assert by_name["temporal_head.cross_attention.in_proj_weight"]["shape"] == [1536, 512]
    assert by_name["temporal_head.value_head.weight"]["shape"] == [2, 512]
    assert by_name["temporal_head.value_head.bias"]["shape"] == [2]
    counts = {name: entry["numel"] for name, entry in by_name.items()}
    assert sum(counts.values()) == 1_056_146
    assert counts["temporal_head.cross_attention.in_proj_bias"] == 1536
    assert counts["temporal_head.cross_attention.out_proj.weight"] == 262_144
    assert counts["temporal_head.cross_attention.out_proj.bias"] == 512
    assert counts["temporal_head.value_head.weight"] == 1024


# ---- canonical load discipline ----------------------------------------------
def test_canonical_load_discipline_against_sealed_artifact():
    payload = _canonical_payload()
    model = trc.build_temporal_residual_model(seed=42)
    report = trc.load_canonical_initial_state(
        model, payload["state_dict"], payload["state_sha256"]
    )
    assert report["missing_keys_exactly_w_head"] is True
    assert report["missing_keys"] == sorted(trc.W_HEAD_STATE_KEYS)
    assert report["unexpected_keys_empty"] is True
    assert report["added_keys"] == sorted(trc.W_HEAD_STATE_KEYS)
    assert report["shared_subset_matches_artifact_state_sha256"] is True
    # the full-W state SHA differs from the canonical SHA (extra head bytes)
    assert arm_common.state_sha256(model) != payload["state_sha256"]


def test_canonical_load_discipline_with_fixture_tamper_and_extra_key():
    source = trc.build_temporal_residual_model(seed=42)
    canonical = {
        k: v for k, v in source.state_dict().items()
        if k not in trc.W_HEAD_STATE_KEYS
    }
    recorded = trc.state_sha256_over_keys(canonical, canonical.keys())

    # 1. a clean fixture loads: missing == W keys, unexpected == [], bytes equal
    fresh = trc.build_temporal_residual_model(seed=42)
    report = trc.load_canonical_initial_state(fresh, canonical, recorded)
    assert report["shared_subset_state_sha256"] == recorded

    # 2. tampering one shared tensor breaks the shared-subset byte equality
    tampered = dict(canonical)
    tampered["decoder.rep"] = canonical["decoder.rep"] + 1e-3
    with pytest.raises(RuntimeError, match="shared-subset state SHA"):
        trc.load_canonical_initial_state(
            trc.build_temporal_residual_model(seed=42), tampered, recorded
        )

    # 3. an unexpected key in the "canonical" fixture is rejected
    extended = dict(canonical)
    extended["decoder.fc_out.stray"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected keys"):
        trc.load_canonical_initial_state(
            trc.build_temporal_residual_model(seed=42), extended, recorded
        )

    # 4. a missing shared key surfaces as a missing-key drift (not W-head-only)
    shrunken = {k: v for k, v in canonical.items() if k != "decoder.rep"}
    with pytest.raises(RuntimeError, match="missing keys are not exactly"):
        trc.load_canonical_initial_state(
            trc.build_temporal_residual_model(seed=42), shrunken,
            trc.state_sha256_over_keys(shrunken, shrunken.keys()),
        )


def test_state_sha256_over_keys_matches_arm_common_on_full_state():
    model = trc.build_temporal_residual_model(seed=42)
    keys = set(model.state_dict())
    # restricted to the FULL key set, the helper is exactly arm_common's digest
    assert trc.state_sha256_over_keys(model.state_dict(), keys) == \
        arm_common.state_sha256(model)
    # restricted to the canonical subset of a canonical-loaded model, it equals
    # the artifact's recorded SHA (the byte-exactness proof)
    payload = _canonical_payload()
    loaded = _built()
    assert trc.state_sha256_over_keys(
        loaded.state_dict(), payload["state_dict"].keys()
    ) == payload["state_sha256"]


# ---- the core property: bitwise parent equality at init ---------------------
def test_bitwise_parent_equality_and_delta_exactly_zero_at_init():
    model = _built()
    model.eval()
    neural, calib, side = _tiny_inputs(n_units=12, batch=4)
    with torch.no_grad():
        mine, _ = model(neural, calib_trials=calib, side_features=side)
        identity = model.compute_identity(calib, side_features=side)
        parent = trc.parent_decode(model, neural, identity)
        delta = model.delta_only(neural, identity)
        head_delta = model.temporal_head(model.decoder.fc_in(
            neural.permute(0, 2, 1) + identity
        ))
    assert torch.equal(mine, parent)  # value equality
    assert arm_common.tensor_sha256(mine) == arm_common.tensor_sha256(parent)
    assert int(torch.count_nonzero(delta).item()) == 0  # delta EXACTLY zero
    assert torch.equal(delta, head_delta)
    # the shared helper proves the same facts and records the byte convention
    proof = trc.prove_zero_init_bitwise(model, neural, calib, side)
    assert proof["torch_equal"] and proof["tensor_sha256_equal"]
    assert proof["delta_exactly_zero"] and proof["delta_nonzero_count"] == 0
    assert proof["delta_abs_max"] == 0.0
    assert "signed-zero" in proof["bitwise_convention"]


def test_zero_init_holds_in_train_mode_and_head_consumes_no_rng():
    model = _built()
    neural, calib, side = _tiny_inputs(n_units=8, batch=2)
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        delta = model.delta_only(neural, identity)
        assert int(torch.count_nonzero(delta).item()) == 0  # no head dropout
    # train mode: the parent decoder's built-in dropout consumes the global
    # torch RNG; the W head must consume exactly as much as the plain parent
    # decode path and not one draw more
    model.train()
    torch.manual_seed(1234)
    with torch.no_grad():
        model(neural, calib_trials=calib, side_features=side)
    after_w = torch.rand(64)
    torch.manual_seed(1234)
    with torch.no_grad():
        trc.parent_decode(model, neural, identity)
    after_parent = torch.rand(64)
    assert torch.equal(after_w, after_parent)  # identical RNG footprint


# ---- delta formula / layout --------------------------------------------------
def test_delta_formula_layout_and_shape():
    model = _built()
    torch.nn.init.normal_(model.temporal_head.value_head.weight)
    torch.nn.init.normal_(model.temporal_head.value_head.bias)
    model.eval()
    neural, calib, side = _tiny_inputs(n_units=12, batch=4)
    with torch.no_grad():
        base, delta = model.decode_components(
            neural, model.compute_identity(calib, side_features=side)
        )
        unit_tokens = model.decoder.fc_in(
            neural.permute(0, 2, 1) + model.compute_identity(calib, side_features=side)
        )
        query = model.temporal_head.queries.unsqueeze(0).expand(unit_tokens.size(0), -1, -1)
        latent, _ = model.temporal_head.cross_attention(query, unit_tokens, unit_tokens)
        values = model.temporal_head.value_head(latent)
        manual = torch.einsum(
            "tk,bkc->btc", model.temporal_head.temporal_basis, values
        )
    assert base.shape == delta.shape == (4, 50, 2)  # [B, W, C] added to [B, W, C]
    assert torch.equal(delta, manual)  # delta[t,c] = sum_k basis[t,k]*values[k,c]
    assert int(torch.count_nonzero(delta).item()) > 0  # nonzero once awake
    # base matches the plain parent path: the existing decode is unchanged
    with torch.no_grad():
        parent = trc.parent_decode(
            model, neural, model.compute_identity(calib, side_features=side)
        )
    assert torch.equal(base, parent)


# ---- wake-up: gradient reaches the head --------------------------------------
def test_one_step_wake_up_and_dead_at_init_grad_documentation():
    model = _built()
    neural, calib, side = _tiny_inputs(n_units=10, batch=3)
    torch.manual_seed(7)
    target = torch.randn(3, 50, 2)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0,
    )
    head = model.temporal_head
    before = {
        name: param.detach().clone()
        for name, param in head.named_parameters()
    }

    def _loss():
        pred, _ = model(neural, calib_trials=calib, side_features=side)
        return ((pred - target) ** 2).mean()

    # step 1: value_head grads nonzero; everything upstream EXACTLY zero
    loss = _loss()
    loss.backward()
    assert int(torch.count_nonzero(head.value_head.weight.grad).item()) > 0
    assert int(torch.count_nonzero(head.value_head.bias.grad).item()) > 0
    for name, param in head.named_parameters():
        if name.startswith("value_head"):
            continue
        assert param.grad is not None
        assert int(torch.count_nonzero(param.grad).item()) == 0, name
    optimizer.step()
    assert int(torch.count_nonzero(head.value_head.weight).item()) > 0  # woke
    assert int(torch.count_nonzero(head.value_head.bias).item()) > 0
    for name, param in head.named_parameters():
        if name.startswith("value_head"):
            continue
        # Adam makes no move on exactly-zero grads (weight_decay = 0)
        assert torch.equal(param.detach(), before[name]), name
    # delta is now nonzero
    model.eval()
    with torch.no_grad():
        delta = model.delta_only(neural, model.compute_identity(calib, side_features=side))
    assert float(delta.abs().max().item()) > 0.0

    # step 2: the formerly dead parameters are live (grads reach them now)
    optimizer.zero_grad(set_to_none=True)
    loss2 = _loss()
    loss2.backward()
    for name, param in head.named_parameters():
        if name.startswith("value_head"):
            continue
        assert param.grad is not None
        assert int(torch.count_nonzero(param.grad).item()) > 0, (
            f"{name} stayed dead after value_head woke"
        )
    optimizer.step()
    for name, param in head.named_parameters():
        assert not torch.equal(param.detach(), before[name]), name


# ---- permutation invariance over units ---------------------------------------
def test_permutation_invariance_over_units():
    """Permuting the unit axis of every per-unit input leaves the prediction
    unchanged.  Exact in exact arithmetic (the only unit-mixing ops are the
    permutation-invariant attentions); on hardware the equality is up to
    floating-point summation order over the softmax/weighted-value reductions,
    exactly as for the parent decoder."""
    model = trc.build_temporal_residual_model(seed=42)
    torch.nn.init.normal_(model.temporal_head.value_head.weight)
    torch.nn.init.normal_(model.temporal_head.value_head.bias)
    model.eval()
    neural, calib, side = _tiny_inputs(n_units=16, batch=3)
    perm = torch.randperm(16)
    with torch.no_grad():
        base, delta = model.decode_components(
            neural, model.compute_identity(calib, side_features=side)
        )
        base_p, delta_p = model.decode_components(
            neural[:, :, perm],
            model.compute_identity(calib[:, :, :, perm], side_features=side[:, perm]),
        )
    assert torch.allclose(base, base_p, atol=1e-5, rtol=1e-5)
    assert torch.allclose(delta, delta_p, atol=1e-5, rtol=1e-4)
    assert float((base - base_p).abs().max()) < 1e-5
    # delta carries O(1)-magnitude entries: bound the drift relative to scale
    assert float((delta - delta_p).abs().max()) < 1e-5 * max(
        1.0, float(delta.abs().max())
    )
    # at zero init the prediction is bitwise the parent's, so the parent-level
    # invariance carries over with the same floating-point caveat
    model0 = _built()
    model0.eval()
    with torch.no_grad():
        pred = model0(neural, calib_trials=calib, side_features=side)[0]
        pred_p = model0(neural[:, :, perm], calib_trials=calib[:, :, :, perm],
                        side_features=side[:, perm])[0]
    assert torch.allclose(pred, pred_p, atol=1e-5, rtol=1e-5)


# ---- SWA over W checkpoints ---------------------------------------------------
def test_swa_final_four_strict_reloads_with_extra_keys(tmp_path):
    model = trc.build_temporal_residual_model(seed=42)
    torch.manual_seed(3)
    paths = []
    states = []
    for i in range(4):
        state = {k: (v.clone() if not isinstance(
            v, torch.nn.parameter.UninitializedParameter) else v)
            for k, v in model.state_dict().items()}
        # perturb W-head and parent tensors so the mean is checkable
        state["temporal_head.value_head.weight"] += 0.01 * (i + 1)
        state["temporal_head.queries"] += 0.001 * (i + 1)
        state["decoder.rep"] += 0.0001 * (i + 1)
        path = tmp_path / f"epoch{i:03d}.ckpt"
        torch.save({"state_dict": state}, path)
        paths.append(path)
        states.append(state)
    swa_path = tmp_path / "swa_final4.pt"
    manifest = matched_scorer.build_swa_final_four(paths, swa_path)
    assert manifest["strict_reload_verified"] is True
    swa = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
    # every W-head key survived the average (extra keys present)
    for key in trc.W_HEAD_STATE_KEYS:
        assert key in swa
    expected_value = torch.stack(
        [s["temporal_head.value_head.weight"] for s in states]
    ).double().mean(0).to(swa["temporal_head.value_head.weight"].dtype)
    assert torch.equal(swa["temporal_head.value_head.weight"], expected_value)
    expected_queries = torch.stack(
        [s["temporal_head.queries"] for s in states]
    ).double().mean(0).to(swa["temporal_head.queries"].dtype)
    assert torch.equal(swa["temporal_head.queries"], expected_queries)
    # strict reload into a fresh W model (all keys, including the head)
    fresh = trc.build_temporal_residual_model(seed=42)
    fresh.load_state_dict(swa, strict=True)
    fresh.eval()
    neural, calib, side = _tiny_inputs(n_units=8, batch=2)
    with torch.no_grad():
        pred, _ = fresh(neural, calib_trials=calib, side_features=side)
    assert torch.isfinite(pred).all()
    assert int(torch.count_nonzero(
        fresh.temporal_head.value_head.weight).item()) > 0  # woke in the SWA


# ---- probe diagnostics --------------------------------------------------------
def test_head_probe_diagnostics_fields():
    model = _built()
    neural, calib, side = _tiny_inputs(n_units=8, batch=2)
    probe = trc.head_probe_diagnostics(model, neural, calib, side)
    assert probe["delta_exactly_zero"] is True
    assert probe["delta_abs_max"] == 0.0 and probe["delta_abs_mean"] == 0.0
    assert set(probe["w_head_param_norms"]) == set(trc.W_HEAD_STATE_KEYS)
    assert probe["value_head_weight_max_abs"] == 0.0
    assert probe["value_head_bias_max_abs"] == 0.0
    assert probe["delta_to_base_scale_ratio"] == 0.0
    # after waking the head the same probe sees nonzero delta everywhere
    torch.nn.init.normal_(model.temporal_head.value_head.weight)
    torch.nn.init.normal_(model.temporal_head.value_head.bias)
    probe2 = trc.head_probe_diagnostics(model, neural, calib, side)
    assert probe2["delta_exactly_zero"] is False
    assert probe2["delta_abs_max"] > 0.0
    assert probe2["value_head_weight_max_abs"] > 0.0


# ---- runner surface -----------------------------------------------------------
def test_runner_cli_rejects_existing_root_and_contract_tokens(tmp_path):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    (tmp_path / "cellW_temporal_residual").mkdir()
    stale = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_temporal_residual_cell.py"),
         "--device", "cpu", "--output-root", str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert stale.returncode == 2 and "fresh cell output directory" in stale.stderr
    source = (ROOT / "scripts/run_temporal_residual_cell.py").read_text()
    for token in ("33,925", "strict=False", "swa_final4.pt",
                  "build_swa_final_four", "PYTHONNOUSERSITE=1 is mandatory",
                  "CELL_SMOKE_COMPLETE__NON_AUTHORITATIVE",
                  "canonical_initial_state.pt"):
        assert token in source
    assert "results/subpop_v1" in source
    assert "cellW_temporal_residual" in source
    module_source = (ROOT / "src/tfpd_lane/temporal_residual_cell.py").read_text()
    # the non-causal caveat recorded verbatim (handoff line): the runtime
    # constant IS the exact sentence; the source carries its two fragments
    assert trc.NON_CAUSAL_CAVEAT == (
        "temporal_basis mixes the whole window, so the residual is non-causal, "
        "matching the current decoder"
    )
    assert "so the residual is non-causal" in module_source
    assert "matching the current decoder" in module_source
    assert "non_causal_caveat" in source  # recorded in the integrity block
    # integrity-block fields
    for field in ("num_heads", "head_structure", "initialization",
                  "zero_init_guarantee", "non_causal_caveat", "gradient_flow",
                  "loss", "w_head_parameters", "behavior_scaling_convention"):
        assert field in source
    # the budget guard and canonical-load discipline are structural
    assert "steps_per_epoch != 33925 or epochs != 48" in source
    assert "load_canonical_initial_state" in source
    # added-parameter count recorded in the integrity block
    assert "1_056_146" in (ROOT / "src/tfpd_lane/temporal_residual_cell.py").read_text()
