"""Step 0C tests (CPU-only, no data download, no GPU launch).

Covers HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3 Step 0C
for ``scripts/run_subpop_step0c.py``:

- wrapper eval parity: with NO intervention the wrapper output is bitwise
  equal to the parent decode path (tiny ``StreamingSpintModel``, both the full
  forward and a manual reconstruction of the parent path);
- the intervention sits at the EXACT site (after `src = activity + identity`,
  before `fc_in`), proven by an ``fc_in`` pre-hook against a re-drawn mask;
- zero_nogain / zero_gain arithmetic on a seeded fixture, fraction-0 identity,
  and the f = 1 all-zero (no division) case;
- padding excludes units: ``key_padding_mask`` is [B, N] bool with True =
  excluded, masking a unit changes the output, and a masked key is exactly the
  single-surviving-key attention path (``CrossAttentionLayer`` plumbing);
- the min_keep floor for padding (no fully-masked softmax);
- ensembling average math is exact and the member mask law keeps ~1-p;
- the receipt writer refuses to overwrite;
- checkpoint integrity binds the terminal-receipt SHAs and hard-fails;
- the sealed ``src/tfpd_lane/sparsification.py`` is never imported.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "subpop_step0c_under_test", ROOT / "scripts/run_subpop_step0c.py"
)
step0c = importlib.util.module_from_spec(spec)
sys.modules["subpop_step0c_under_test"] = step0c
spec.loader.exec_module(step0c)

from src.models.components.spint import CrossAttentionLayer, SpintModel  # noqa: E402
from src.models.components.streaming_encoders import build_encoder  # noqa: E402
from src.models.components.streaming_spint import StreamingSpintModel  # noqa: E402


# ---------------------------------------------------------------------------
def _tiny_parent() -> StreamingSpintModel:
    """Tiny coupled B3S graph; dynamic_dropout=True so parity is proven with
    D's flag present (the flag is train-mode-only and inert at eval)."""
    torch.manual_seed(0)
    decoder = SpintModel(
        model_dim=16, num_covariates=2, window_size=8, num_heads=2, num_layers=1,
        num_id_layers=1, use_learnable_id=True, learnable_id_type="mlp",
        learnable_rep=True, dynamic_dropout=True, dynamic_dropout_low=0.0,
        dynamic_dropout_high=1.0,
    )
    id_encoder = build_encoder(
        "B3S", window_size=8, trial_length=20, id_hidden_dim=8, hidden_dim=4, side_dim=4
    )
    return StreamingSpintModel(
        decoder=decoder, id_encoder=id_encoder, decoder_mode="coupled"
    )


def _tiny_inputs(batch: int = 3, units: int = 5, window: int = 8):
    torch.manual_seed(1)
    return (
        torch.rand(batch, window, units),
        torch.rand(batch, 4, 20, units),
        torch.randn(batch, units, 4),
    )


# ---------------------------------------------------------------------------
def test_wrapper_bitwise_equal_to_parent_decode_path():
    parent = _tiny_parent().eval()
    wrapper = step0c.SubpopEvalModel(parent).eval()
    neural, calib, side = _tiny_inputs()
    with torch.no_grad():
        mine, identity_mine = wrapper(neural, calib_trials=calib, side_features=side)
        theirs, identity_parent = parent(neural, calib_trials=calib, side_features=side)
        # manual parent path through the same modules in the same order
        src = neural.permute(0, 2, 1) + identity_parent
        src = parent.decoder.fc_in(src)
        rep = parent.decoder.fc_in(parent.decoder.rep).to(src)
        out, _ = parent.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        manual = parent.decoder.fc_out(out).permute(0, 2, 1)
    assert torch.equal(identity_mine, identity_parent)
    assert torch.equal(mine, theirs)          # full forward, bitwise
    assert torch.equal(mine, manual)          # decode path reconstruction


def test_fraction_zero_is_bitwise_identity_for_both_zero_kinds():
    parent = _tiny_parent().eval()
    wrapper = step0c.SubpopEvalModel(parent).eval()
    neural, calib, side = _tiny_inputs()
    with torch.no_grad():
        native, _ = wrapper(neural, calib_trials=calib, side_features=side)
        for kind in ("zero_nogain", "zero_gain"):
            wrapper.set_intervention(kind, 0.0, torch.Generator().manual_seed(123))
            out, _ = wrapper(neural, calib_trials=calib, side_features=side)
            assert torch.equal(out, native), kind


def test_intervention_sits_between_identity_add_and_fc_in():
    parent = _tiny_parent().eval()
    wrapper = step0c.SubpopEvalModel(parent).eval()
    neural, calib, side = _tiny_inputs()
    captured = {}

    def capture_unit_tokens(module, args):
        # fc_in is called twice per decode (unit tokens, then the rep query);
        # only the FIRST call carries [B, N, W].  Must return None: a non-None
        # pre-hook return value REPLACES the module input.
        if "src" not in captured:
            captured["src"] = args[0].detach().clone()

    fraction, seed = 0.4, 99
    wrapper.set_intervention("zero_gain", fraction, torch.Generator().manual_seed(seed))
    handle = wrapper.decoder.fc_in.register_forward_pre_hook(capture_unit_tokens)
    with torch.no_grad():
        wrapper(neural, calib_trials=calib, side_features=side)
    handle.remove()
    with torch.no_grad():
        identity = wrapper.compute_identity(calib, side_features=side)
        pre_site = neural.permute(0, 2, 1) + identity
        mask = torch.bernoulli(
            torch.full(pre_site.shape[:2], 1.0 - fraction),
            generator=torch.Generator().manual_seed(seed),
        )
        expected = pre_site * mask.unsqueeze(-1) / (1.0 - fraction)
        nogain_expected = pre_site * mask.unsqueeze(-1)
    torch.testing.assert_close(captured["src"], expected, rtol=0, atol=0)
    # zero_nogain at the same seed differs exactly by the gain factor
    wrapper.set_intervention("zero_nogain", fraction, torch.Generator().manual_seed(seed))
    captured.clear()
    handle = wrapper.decoder.fc_in.register_forward_pre_hook(capture_unit_tokens)
    with torch.no_grad():
        wrapper(neural, calib_trials=calib, side_features=side)
    handle.remove()
    torch.testing.assert_close(captured["src"], nogain_expected, rtol=0, atol=0)


def test_zero_kinds_math_and_fully_removed_population_is_finite():
    wrapper = step0c.SubpopEvalModel(_tiny_parent().eval()).eval()
    torch.manual_seed(2)
    src = torch.randn(6, 40, 12)
    for kind, fraction in (("zero_nogain", 0.25), ("zero_gain", 0.25)):
        wrapper.set_intervention(kind, fraction, torch.Generator().manual_seed(5))
        out, mask_out = wrapper.apply_intervention(src)
        keep = torch.bernoulli(
            torch.full((6, 40), 0.75), generator=torch.Generator().manual_seed(5)
        )
        expected = src * keep.unsqueeze(-1)
        if kind == "zero_gain":
            expected = expected / 0.75
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert mask_out is None
        dropped_rows = keep == 0
        if kind == "zero_nogain":
            assert torch.equal(out[dropped_rows], torch.zeros_like(out[dropped_rows]))
    # f == 1: the whole population is removed without any division (no NaN/inf)
    wrapper.set_intervention("zero_gain", 1.0, torch.Generator().manual_seed(5))
    out, _ = wrapper.apply_intervention(src)
    assert torch.equal(out, torch.zeros_like(out))
    assert bool(torch.isfinite(out).all())


def test_padding_key_padding_mask_shape_semantics_and_effect():
    # (a) plumbing: excluding a key is exactly the single-surviving-key path
    torch.manual_seed(3)
    layer = CrossAttentionLayer(d_model=8, nhead=2, dropout=0.0).eval()
    query, keys = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
    with torch.no_grad():
        base, _ = layer(query, keys)
        masked, _ = layer(query, keys, key_padding_mask=torch.tensor([[True, False]]))
        single, _ = layer(query, keys[:, 1:2])
    assert not torch.equal(base, masked)                      # masking changes it
    torch.testing.assert_close(masked, single, rtol=1e-6, atol=1e-6)

    # (b) the wrapper's padding intervention: [B, N] bool, True = excluded,
    #     src NOT zeroed, output differs from native and stays finite
    parent = _tiny_parent().eval()
    wrapper = step0c.SubpopEvalModel(parent).eval()
    neural, calib, side = _tiny_inputs()
    with torch.no_grad():
        native, _ = wrapper(neural, calib_trials=calib, side_features=side)
    captured = {}

    def hook(module, args, kwargs, output):
        captured["args"] = args
        captured["kpm"] = kwargs.get("key_padding_mask")

    handle = wrapper.decoder.transformer.register_forward_hook(hook, with_kwargs=True)
    wrapper.set_intervention("padding", 0.5, torch.Generator().manual_seed(7))
    with torch.no_grad():
        padded, _ = wrapper(neural, calib_trials=calib, side_features=side)
    handle.remove()
    kpm = captured["kpm"]
    assert kpm is not None and tuple(kpm.shape) == (3, 5) and kpm.dtype == torch.bool
    assert 0 < int(kpm.sum()) < kpm.numel()            # some, never all, excluded
    assert not torch.equal(padded, native)
    assert bool(torch.isfinite(padded).all())
    # src is not zeroed: the padded token set fed to fc_in equals the native one
    wrapper.set_intervention("none", 0.0)
    assert step0c.SubpopEvalModel(parent).intervention == "none"


def test_padding_min_keep_floor_prevents_fully_masked_softmax():
    wrapper = step0c.SubpopEvalModel(_tiny_parent().eval(), min_keep=4)
    keep = torch.zeros(2, 10)
    keep[0, 5] = 1.0                    # 1 survivor  -> floored
    keep[1, [0, 1, 2, 3, 4]] = 1.0      # 5 survivors -> untouched
    fixed = wrapper._enforce_min_keep(keep.clone())
    assert fixed[0].tolist() == [1.0] * 4 + [0.0] * 6     # lowest canonical indices
    assert torch.equal(fixed[1], keep[1])
    assert wrapper.mask_stats["n_samples_min_keep_floored"] == 1
    # an over-tight floor on a toy population keeps every unit (no NaN path)
    toy = step0c.SubpopEvalModel(_tiny_parent().eval(), min_keep=4).eval()
    toy.set_intervention("padding", 0.9, torch.Generator().manual_seed(3))
    neural, calib, side = _tiny_inputs(units=3)
    with torch.no_grad():
        out, _ = toy(neural, calib_trials=calib, side_features=side)
    assert bool(torch.isfinite(out).all())


def test_ensemble_forward_averages_members_exactly():
    parent = _tiny_parent().eval()
    wrapper = step0c.SubpopEvalModel(parent).eval()
    neural, calib, side = _tiny_inputs()
    k, seed_idx, label = 3, 0, "ens_probe_K3"
    forward, p_values = step0c.ensemble_forward(wrapper, "zero_gain", k, seed_idx, label)
    assert len(p_values) == k and all(0.0 <= p < 1.0 for p in p_values)
    with torch.no_grad():
        averaged, identity = forward(neural, calib, side)
        manual = None
        for p_value, member in zip(p_values, range(k)):
            generator = torch.Generator().manual_seed(
                step0c.intervention_generator_seed(label, seed_idx, member_idx=member)
            )
            wrapper.set_intervention("zero_gain", p_value, generator)
            member_out = wrapper.decode_with_identity(neural, identity)
            manual = member_out if manual is None else manual + member_out
    torch.testing.assert_close(averaged, manual / k, rtol=0, atol=0)
    # the p stream is a separate, reproducible generator
    _, p_again = step0c.ensemble_forward(wrapper, "zero_gain", k, seed_idx, label)
    assert p_again == p_values
    _, p_other = step0c.ensemble_forward(wrapper, "zero_gain", k, seed_idx + 1, label)
    assert p_other != p_values


def test_member_mask_law_kept_fraction_tracks_one_minus_p():
    wrapper = step0c.SubpopEvalModel(_tiny_parent().eval()).eval()
    for fraction in (0.1, 0.5, 0.75):
        wrapper.reset_mask_stats()
        wrapper.set_intervention("zero_nogain", fraction, torch.Generator().manual_seed(11))
        src = torch.randn(64, 2000, 50)
        with torch.no_grad():
            wrapper.apply_intervention(src)
        stats = wrapper.consume_mask_stats()
        realized = stats["realized_kept_fraction_mean"]
        assert realized == pytest.approx(1.0 - fraction, abs=0.01), (fraction, realized)
        assert 0.0 <= stats["kept_fraction_min"] <= stats["kept_fraction_max"] <= 1.0


def test_wrapper_is_inference_only():
    wrapper = step0c.SubpopEvalModel(_tiny_parent())
    wrapper.train()
    neural, calib, side = _tiny_inputs()
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="inference-only"):
            wrapper(neural, calib_trials=calib, side_features=side)


def test_seed_rule_deterministic_and_pairwise_distinct():
    rule = step0c.intervention_generator_seed
    assert rule("zero_nogain_f0.25", 0) == rule("zero_nogain_f0.25", 0)
    assert rule("zero_nogain_f0.25", 0) != rule("zero_nogain_f0.25", 1)
    assert rule("zero_nogain_f0.25", 0) != rule("zero_nogain_f0.5", 0)
    assert rule("ens_zero_gain_K4", 0, member_idx=2) != rule("ens_zero_gain_K4", 0)
    assert 0 <= rule("x", 0) < 2**63 - 1
    gen = torch.Generator().manual_seed(rule("cond", 1))
    assert torch.rand(3, generator=gen).tolist() == torch.rand(
        3, generator=torch.Generator().manual_seed(rule("cond", 1))
    ).tolist()


def test_receipt_writer_refuses_overwrite(tmp_path):
    receipt_mod = step0c._load_module(
        "tfpd_lane_receipt_step0c_test", ROOT / "src/tfpd_lane/receipt.py"
    )
    target = tmp_path / "step0c_receipt.json"
    receipt_mod.write_receipt_transactionally(target, {"schema": "tfpd_subpop_step0c_v1"})
    assert target.is_file()
    with pytest.raises(SystemExit) as excinfo:
        receipt_mod.write_receipt_transactionally(target, {"schema": "overwritten"})
    assert excinfo.value.code == 2
    assert json.loads(target.read_text())["schema"] == "tfpd_subpop_step0c_v1"
    sidecar = target.with_suffix(target.suffix + ".sha256")
    assert sidecar.read_text().startswith(
        hashlib.sha256(target.read_bytes()).hexdigest()
    )


def _checkpoint_fixture(tmp_path: Path, tamper: str | None = None):
    directory = tmp_path / "results/pop_robust_v1/cellD_2heads_dynamic_dropout"
    directory.mkdir(parents=True)
    ckpt = directory / "swa_final4.pt"
    torch.save({"state_dict": {}}, ckpt)
    payload = {
        "status": "CELL_TERMINAL",
        "swa": {"sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()},
        "epochs_run": 48,
    }
    if tamper == "sha":
        payload["swa"]["sha256"] = "0" * 64
    elif tamper == "status":
        payload["status"] = "CELL_FAILED"
    elif tamper == "missing_key":
        payload.pop("swa")
    receipt = directory / "terminal_receipt.json"
    receipt.write_text(json.dumps(payload))
    return ckpt, receipt


def test_checkpoint_integrity_binds_terminal_receipt_shas(tmp_path, monkeypatch):
    ckpt, _receipt = _checkpoint_fixture(tmp_path)
    spec = {
        "path": ckpt,
        "bound_sha256": None,   # D: the value is READ from the terminal receipt
        "terminal_receipt": _receipt,
        "terminal_status": "CELL_TERMINAL",
        "note": "fixture",
    }
    monkeypatch.setattr(step0c, "CHECKPOINTS", {"D": spec})
    integrity = step0c.verify_checkpoints(step0c._load_module(
        "tfpd_lane_arm_common_step0c_test", ROOT / "src/tfpd_lane/arm_common.py"
    ).sha256_file)
    assert integrity["D"]["sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert integrity["D"]["bound_from"] == "terminal receipt"


@pytest.mark.parametrize("tamper", ["sha", "status", "missing_key"])
def test_checkpoint_integrity_hard_fails_on_tampering(tmp_path, monkeypatch, tamper):
    ckpt, receipt = _checkpoint_fixture(tmp_path, tamper=tamper)
    monkeypatch.setattr(step0c, "CHECKPOINTS", {"D": {
        "path": ckpt, "bound_sha256": None, "terminal_receipt": receipt,
        "terminal_status": "CELL_TERMINAL", "note": "fixture",
    }})
    with pytest.raises(SystemExit, match="integrity failed"):
        step0c.verify_checkpoints(lambda _p: hashlib.sha256(_p.read_bytes()).hexdigest())


def test_arm_a_bound_sha_matches_the_sealed_artifact():
    path = Path(step0c.CHECKPOINTS["armA"]["path"])
    if not path.is_file():
        pytest.skip("sealed arm A SWA not present in this checkout")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        step0c.CHECKPOINTS["armA"]["bound_sha256"]


def test_reference_receipts_sha_bound_and_loadable():
    for key, spec in step0c.REFERENCE_RECEIPTS.items():
        path = Path(spec["path"])
        if not path.is_file():
            pytest.skip(f"sealed reference receipt missing: {path}")
        body = hashlib.sha256(path.read_bytes()).hexdigest()
        assert body == spec["sha256"], key
        node = json.loads(path.read_text())
        for step in spec["pointer"]:
            node = node[step]
        assert isinstance(node, float) and 0.0 < node < 1.0
    values = step0c.load_reference_values(
        step0c._load_module(
            "tfpd_lane_arm_common_step0c_ref", ROOT / "src/tfpd_lane/arm_common.py"
        ).sha256_file
    )
    assert values["armA_external_native"]["value"] == pytest.approx(0.2604, abs=1e-4)
    assert values["D_external_native"]["value"] == pytest.approx(0.4179, abs=1e-4)
    with pytest.raises(SystemExit, match="SHA mismatch"):
        step0c.load_reference_values(lambda _p: "0" * 64)


def test_sealed_sparsification_module_is_not_imported_and_grid_is_frozen():
    source = (ROOT / "scripts/run_subpop_step0c.py").read_text()
    for forbidden in (
        "from src.tfpd_lane import sparsification",
        "from src.tfpd_lane.sparsification",
        "import src.tfpd_lane.sparsification",
        'ROOT / "src/tfpd_lane/sparsification.py"',
    ):
        assert forbidden not in source
    # the route owns its own wrapper: the class and the site live in this file
    assert "class SubpopEvalModel" in source
    assert "src/tfpd_lane/sparsification" in source  # the documented non-import
    assert step0c.SPEC_FRACTIONS == (0.0, 0.1, 0.25, 0.5, 0.75)
    assert step0c.SPEC_N_MASK_SEEDS == 3
    assert step0c.SPEC_ENSEMBLE_K == (1, 4, 8)
    assert step0c.MIN_KEEP == 4
    assert step0c.CURVE_KINDS[0] == "zero_nogain"      # PRIMARY
    assert step0c.ENSEMBLE_KINDS[0] == "zero_gain"     # primary for ensembling
    assert step0c.date_block("sub-M_ses-CO-20140307") == "2014"
    assert step0c.date_block("sub-M_ses-CO-20150625") == "2015"
    assert step0c.date_block(step0c.SEPARATE_SESSION) == "2014"  # split by name
