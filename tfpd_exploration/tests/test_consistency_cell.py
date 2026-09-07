"""Cell C test suite (pre-launch gate): paired-subset consistency.

Covers HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3 items
1-6 against the route-owned implementation:

- initial-state / graph / head-count / parameter-count parity vs the canonical
  Arm A artifact (strict load, bitwise state SHA);
- evaluation masks are exact ones and the forward is bitwise equal to the
  unsparsified parent path; scoring does no gradient/update;
- the exact D site (after identity addition, before fc_in) via the inherited
  parent decode; whole-unit [B, N] mask structure; gain 1/(1-p) per branch;
  p == 1 -> all-zero branch tensor without division;
- two INDEPENDENT branch masks, each identical in law to D's mask (kept ~ 1-p,
  law matched against F.dropout on an all-ones tensor), branch-2 stream
  unaffected by branch-1 redraws, global torch RNG untouched;
- Jaccard overlap recorder math on hand-computed fixtures (both-empty skip);
- loss composition: with lambda = 0 and identical masks, loss = 2 x Arm A
  loss; consistency zero when predictions are equal; gradients flow through
  BOTH branches (clean detach construction);
- skip floor 1: one branch empty for a sample -> excluded, no NaN; both
  branches empty -> zero loss for the sample; whole-branch p = 1;
- one real CPU training step: two forwards, one optimizer step, finite losses,
  recorded components, Jaccard counts;
- the R-Drop structure: two train() forwards with identical masks still
  disagree (independent internal dropout passes);
- runner budget guard (33,925 x 48), fresh-directory CLI rejection, sampler
  contract, sealed shared files byte-frozen.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane import consistency_cell as cc
from src.tfpd_lane import sparsification as sp

CANONICAL = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"


def _canonical_state():
    payload = torch.load(CANONICAL, map_location="cpu", weights_only=False)
    return payload["state_dict"], payload["state_sha256"]


def _tiny_inputs(n_units=10, batch=3):
    torch.manual_seed(0)
    return (
        torch.rand(batch, 50, n_units),
        torch.rand(batch, 30, 100, n_units),
        torch.randn(batch, n_units, 4),
    )


# ---------------------------------------------------------------------------
# graph parity + eval policy
# ---------------------------------------------------------------------------
def test_frozen_constants_and_shared_definitions():
    assert cc.LAMBDA == 0.1 and cc.LAMBDA_FROZEN is True
    assert cc.BRANCH1_KEEP_SEED == 42_004
    assert cc.BRANCH2_KEEP_SEED == 42_005
    assert cc.DRAWS_PER_STEP == 2
    assert cc.SKIP_FLOOR == 1
    assert cc.NUM_HEADS == 2
    # PStream and the route-owned model pattern are imported, never duplicated
    assert cc.PStream is sp.PStream
    assert issubclass(cc.PairedConsistencySpintModel, sp.SparsifiedStreamingSpintModel)
    assert "33,925" in cc.COMPUTE_OPTION and "compute-matched D control at 96 epochs" in cc.PREREGISTRATION


def test_initial_state_graph_head_and_param_parity():
    canonical, canonical_sha = _canonical_state()
    from src.tfpd_lane import arm_common
    from torch.nn.parameter import UninitializedParameter

    model = cc.build_consistency_model(seed=42)
    model.load_state_dict(canonical, strict=True)
    assert arm_common.state_sha256(model) == canonical_sha  # bitwise parity
    assert model.decoder.transformer.layers[0].cross_attn.num_heads == 2
    assert sorted(model.state_dict().keys()) == sorted(canonical.keys())
    mine = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad and not isinstance(p, UninitializedParameter)
    )
    reference = sp.build_sparsified_model(seed=42, cell="R")
    ref_count = sum(
        p.numel() for p in reference.parameters()
        if p.requires_grad and not isinstance(p, UninitializedParameter)
    )
    assert mine == ref_count  # parameter-count parity with the Arm A / R / S2 graph
    assert model.decoder.dynamic_dropout is False  # route mask replaces D's flag


def test_exact_site_and_branch_mask_structure():
    canonical, _ = _canonical_state()
    # the paired model inherits the parent decode unchanged; the single mask
    # insertion stays between the identity addition and fc_in
    assert "decode_with_identity" not in vars(cc.PairedConsistencySpintModel)
    body = inspect.getsource(sp.SparsifiedStreamingSpintModel.decode_with_identity)
    assert body.index("src = src + identity") < body.index("self.apply_sparsification(src)") \
        < body.index("self.decoder.fc_in(src)")

    model = cc.build_consistency_model(seed=42)
    model.load_state_dict(canonical, strict=True)
    model.eval()
    neural, calib, side = _tiny_inputs()
    masker = cc.PairedSubsetMasker()
    p = 0.4
    keep = masker.draw(1, neural.shape[0], neural.shape[-1], p)
    assert keep.shape == (neural.shape[0], neural.shape[-1])  # whole-unit [B, N]
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        src = neural.permute(0, 2, 1) + identity + 0.5  # strictly positive rows
        model.current_branch = 1
        model.current_p = p
        model.current_keep_mask = keep
        model.sparsification_enabled = True
        out = model.apply_sparsification(src)
        model.sparsification_enabled = False
    expected = src * keep.unsqueeze(-1) / (1.0 - p)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # whole-unit structure: each unit row is either exactly zero across W or
    # fully scaled (no partial units)
    for b in range(out.shape[0]):
        row_zero = (out[b] == 0).all(dim=-1)
        row_kept = keep[b] == 1
        assert bool((row_zero == (row_kept == 0)).all())
    # kept-entry amplification is exactly 1/(1-p), no clamp/floor
    kept = keep == 1
    assert kept.any()
    ratio = out[kept] / src[kept]
    torch.testing.assert_close(ratio, torch.full_like(ratio, 1.0 / (1.0 - p)),
                               rtol=0, atol=1e-6)


def test_p_equals_one_all_zero_branch_without_division():
    canonical, _ = _canonical_state()
    model = cc.build_consistency_model(seed=42)
    model.load_state_dict(canonical, strict=True)
    neural, calib, side = _tiny_inputs()
    masker = cc.PairedSubsetMasker()
    keep = masker.draw(2, neural.shape[0], neural.shape[-1], 1.0)
    assert torch.count_nonzero(keep).item() == 0
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        src = neural.permute(0, 2, 1) + identity
        model.current_branch = 2
        model.current_p = 1.0
        model.current_keep_mask = keep
        model.sparsification_enabled = True
        out = model.apply_sparsification(src)
        model.sparsification_enabled = False
    assert torch.count_nonzero(out).item() == 0
    assert torch.isfinite(out).all()  # no division by zero anywhere


def test_eval_mask_exact_ones_and_bitwise_parent_equality():
    canonical, _ = _canonical_state()
    model = cc.build_consistency_model(seed=42)
    model.load_state_dict(canonical, strict=True)
    model.eval()
    neural, calib, side = _tiny_inputs()
    model.sparsification_enabled = False
    with torch.no_grad():
        mine, _ = model(neural, calib_trials=calib, side_features=side)
        # the unsparsified parent path through the same modules in the same order
        identity = model.compute_identity(calib, side_features=side)
        src = neural.permute(0, 2, 1) + identity  # mask would go here: none
        src = model.decoder.fc_in(src)
        rep = model.decoder.fc_in(model.decoder.rep).to(src)
        out, _ = model.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        parent = model.decoder.fc_out(out).permute(0, 2, 1)
    assert torch.equal(mine, parent)
    assert model.sparsification_enabled is False  # never leaks into eval paths
    assert all(p.grad is None for p in model.parameters())  # no update from eval


# ---------------------------------------------------------------------------
# two independent masks, D law, Jaccard recorder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7])
def test_branch_mask_law_matches_D(p):
    torch.manual_seed(0)
    d_mask = torch.nn.functional.dropout(torch.ones(64, 500), p=p, training=True)
    d_kept = float((d_mask != 0).float().mean().item())
    survivors = d_mask[d_mask != 0]
    torch.testing.assert_close(survivors, torch.full_like(survivors, 1.0 / (1.0 - p)),
                               rtol=1e-6, atol=0)  # D's exact gain law
    masker = cc.PairedSubsetMasker()
    for branch in (1, 2):
        kept = float(masker.draw(branch, 64, 500, p).mean().item())
        assert abs(kept - (1.0 - p)) < 0.01, (branch, p, kept)
    assert abs(d_kept - (1.0 - p)) < 0.01  # same law as D's realized mask


def test_branches_independent_streams_and_global_rng_untouched():
    state_before = torch.random.get_rng_state()
    a = cc.PairedSubsetMasker()
    a.draw(1, 8, 50, 0.5)
    second_a = a.draw(2, 8, 50, 0.5)
    # redrawing branch 1 any number of times never changes branch 2's stream
    b = cc.PairedSubsetMasker()
    for _ in range(3):
        b.draw(1, 8, 50, 0.3)
    second_b = b.draw(2, 8, 50, 0.5)
    torch.testing.assert_close(second_a, second_b, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), state_before)  # global RNG untouched
    # cross-branch independence: P(unit kept by both) ~ (1-p1)(1-p2)
    masker = cc.PairedSubsetMasker()
    k1 = masker.draw(1, 32, 500, 0.5)
    k2 = masker.draw(2, 32, 500, 0.5)
    overlap = float(((k1 == 1) & (k2 == 1)).float().mean().item())
    assert abs(overlap - 0.25) < 0.01


def test_jaccard_recorder_hand_fixtures():
    recorder = cc.JaccardRecorder()
    k1 = torch.zeros(4, 6)
    k2 = torch.zeros(4, 6)
    k1[0, :3] = 1.0                     # {0,1,2} vs {1,2,3}: |.|=2, |u|=4 -> 0.5
    k2[0, 1:4] = 1.0
    k1[1, :2] = 1.0                     # identical sets -> 1.0
    k2[1, :2] = 1.0
    k1[2, 0] = 1.0                      # disjoint singletons -> 0.0
    k2[2, 1] = 1.0
    # row 3: both empty -> skipped, never averaged in
    recorder.update(k1, k2)
    summary = recorder.summary()
    assert summary["n"] == 3 and summary["samples_seen"] == 4
    assert summary["skipped_both_empty"] == 1
    assert summary["mean"] == pytest.approx(0.5, abs=1e-12)
    assert summary["min"] == 0.0 and summary["max"] == 1.0
    assert summary["median"] == pytest.approx(0.5, abs=1e-12)
    recorder.reset()
    assert recorder.summary() == {"n": 0, "samples_seen": 0, "skipped_both_empty": 0}


# ---------------------------------------------------------------------------
# loss composition, gradients, skip floor
# ---------------------------------------------------------------------------
def _toy_batch(batch=5, windows=7, channels=2, units=9):
    torch.manual_seed(3)
    behavior = torch.randn(batch, windows, channels)
    pred_1 = torch.randn(batch, windows, channels)
    pred_2 = torch.randn(batch, windows, channels)
    valid = torch.ones(batch, windows, dtype=torch.bool)
    valid[2, :] = False  # one fully padded sample, exactly Arm A's pad law
    keep = (torch.rand(batch, units) > 0.3).float()
    keep[:, 0] = 1.0  # guarantee non-empty rows for the composition test
    return behavior, pred_1, pred_2, valid, keep


def test_loss_composition_lambda_zero_identical_masks():
    behavior, pred, _, valid, keep = _toy_batch()
    total, comp = cc.paired_consistency_loss(pred, pred, behavior, keep, keep,
                                             valid, lambda_=0.0)
    arm_a = cc.arm_a_masked_mse(pred, behavior, valid)
    assert torch.equal(total, 2 * arm_a)          # exactly 2 x Arm A loss
    assert float(comp["consistency"]) == 0.0      # zero when predictions equal
    total_lambda, comp_lambda = cc.paired_consistency_loss(
        pred, pred, behavior, keep, keep, valid, lambda_=cc.LAMBDA
    )
    assert torch.equal(total_lambda, 2 * arm_a)   # consistency contributes zero
    assert comp_lambda["rows_branch1"] == comp_lambda["rows_branch2"] \
        == comp_lambda["rows_consistency"] == int(valid.sum())


def test_arm_a_masked_mse_formula_is_exact_arm_a():
    PAD = cc.PAD_VALUE
    behavior = torch.randn(4, 6, 2)
    behavior[2] = PAD
    pred = torch.randn(4, 6, 2)
    valid = (behavior != PAD).all(dim=-1)
    mine = cc.arm_a_masked_mse(pred, behavior, valid)
    diff2 = ((pred - behavior) ** 2).sum(dim=-1)
    reference = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
    assert torch.equal(mine, reference)


def test_gradients_flow_through_both_branches():
    torch.manual_seed(5)
    target = torch.randn(6, 1, 3)  # [B, W=1, C]: the loss's valid mask is per window
    keep = torch.ones(6, 8)
    valid = torch.ones(6, 1, dtype=torch.bool)

    w1 = torch.zeros(3, 3, requires_grad=True)
    w2 = torch.zeros(3, 3, requires_grad=True)
    x = torch.randn(6, 1, 3)
    pred_1, pred_2 = x @ w1, x @ w2
    loss, _ = cc.paired_consistency_loss(pred_1, pred_2, target, keep, keep,
                                         valid, lambda_=cc.LAMBDA)
    loss.backward()
    assert w1.grad is not None and float(w1.grad.abs().sum()) > 0
    assert w2.grad is not None and float(w2.grad.abs().sum()) > 0

    # branch-2-only path: detaching branch 1 leaves branch 2 supervised by its
    # own behavior loss AND the consistency term
    w3 = torch.zeros(3, 3, requires_grad=True)
    w4 = torch.zeros(3, 3, requires_grad=True)
    pred_3, pred_4 = x @ w3, x @ w4
    assert pred_3.shape == target.shape
    loss_b, _ = cc.paired_consistency_loss(pred_3.detach(), pred_4, target,
                                           keep, keep, valid, lambda_=cc.LAMBDA)
    loss_b.backward()
    assert w3.grad is None
    assert w4.grad is not None and float(w4.grad.abs().sum()) > 0
    # symmetric: branch 1 alone still learns
    w5 = torch.zeros(3, 3, requires_grad=True)
    loss_c, _ = cc.paired_consistency_loss(x @ w5, pred_4.detach(), target,
                                           keep, keep, valid, lambda_=cc.LAMBDA)
    loss_c.backward()
    assert w5.grad is not None and float(w5.grad.abs().sum()) > 0


def test_skip_floor_one_branch_empty():
    torch.manual_seed(7)
    behavior, pred_1, pred_2, valid, keep_2 = _toy_batch()
    keep_1 = keep_2.clone()
    keep_1[0] = 0.0  # sample 0: branch 1 dropped everything
    total, comp = cc.paired_consistency_loss(pred_1, pred_2, behavior,
                                             keep_1, keep_2, valid)
    w1 = valid & (keep_1.sum(dim=1) > 0).unsqueeze(-1)
    w2 = valid & (keep_2.sum(dim=1) > 0).unsqueeze(-1)
    wc = w1 & w2
    assert torch.equal(comp["loss_branch1"], cc.arm_a_masked_mse(pred_1, behavior, w1))
    assert torch.equal(comp["loss_branch2"], cc.arm_a_masked_mse(pred_2, behavior, w2))
    assert torch.equal(comp["consistency"], cc.arm_a_masked_mse(pred_1, pred_2, wc))
    assert int(comp["branch1_skipped_samples"]) == 1
    assert int(comp["branch2_skipped_samples"]) == 0
    assert int(comp["both_branches_skipped_samples"]) == 0
    assert int(comp["rows_consistency"]) == int(wc.sum())
    assert torch.isfinite(total)


def test_skip_floor_both_branches_empty_for_a_sample():
    torch.manual_seed(7)
    behavior, pred_1, pred_2, valid, keep = _toy_batch()
    keep_1, keep_2 = keep.clone(), keep.clone()
    keep_1[0] = 0.0
    keep_2[0] = 0.0  # sample 0 empty in BOTH branches -> zero loss for it
    total, comp = cc.paired_consistency_loss(pred_1, pred_2, behavior,
                                             keep_1, keep_2, valid)
    w1 = valid & (keep_1.sum(dim=1) > 0).unsqueeze(-1)
    assert torch.equal(comp["loss_branch1"], cc.arm_a_masked_mse(pred_1, behavior, w1))
    assert int(comp["both_branches_skipped_samples"]) == 1
    assert int(comp["rows_consistency"]) == int(valid.sum()) - behavior.shape[1]
    assert torch.isfinite(total)


def test_skip_floor_whole_branch_p_one_and_both_p_one():
    torch.manual_seed(7)
    behavior, pred_1, pred_2, valid, keep = _toy_batch()
    empty = torch.zeros_like(keep)
    total, comp = cc.paired_consistency_loss(pred_1, pred_2, behavior,
                                             empty, keep, valid)
    # branch 1 fully excluded; consistency drops out (empty row set -> 0)
    assert float(comp["loss_branch1"]) == 0.0
    assert float(comp["consistency"]) == 0.0
    assert torch.equal(comp["total"], comp["loss_branch2"])
    assert int(comp["branch1_skipped_samples"]) == keep.shape[0]
    total2, comp2 = cc.paired_consistency_loss(pred_1, pred_2, behavior,
                                               empty, empty.clone(), valid)
    assert float(total2) == 0.0  # both branches p == 1: zero loss, no NaN
    assert torch.isfinite(total2)
    assert int(comp2["both_branches_skipped_samples"]) == keep.shape[0]


# ---------------------------------------------------------------------------
# one real CPU training step + the R-Drop structure
# ---------------------------------------------------------------------------
def test_real_model_one_train_step_and_rdrop_structure():
    canonical, _ = _canonical_state()
    model = cc.build_consistency_model(seed=42)
    model.load_state_dict(canonical, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    neural, calib, side = _tiny_inputs()
    behavior = torch.rand(neural.shape[0], 50, 2) * 2.0 - 0.5  # no pad rows
    batch = (neural, behavior, calib, ["ses-fixture"], side)
    p_stream = sp.PStream(seed=sp.P_STREAM_SEED)
    masker = cc.PairedSubsetMasker()
    jaccard = cc.JaccardRecorder()
    lr_fn = lambda step: 1e-4
    stats = cc.train_epoch_paired_consistency(
        model, optimizer, [batch], lr_fn, torch.device("cpu"),
        p_stream, masker, jaccard, phase_step=0, max_steps=1,
    )
    assert stats["optimizer_steps"] == 1 and stats["forwards"] == 2
    assert stats["train_example_windows"] == neural.shape[0]
    assert p_stream.stats()["n"] == 2  # two draws per step
    for key in ("train_loss_mean_per_step", "behavior_loss_sum_mean_per_step",
                "consistency_loss_mean_per_step", "branch1_loss_mean_per_step",
                "branch2_loss_mean_per_step"):
        assert np.isfinite(stats[key]), key
    summary = jaccard.summary()
    assert summary["n"] + summary["skipped_both_empty"] == neural.shape[0]
    mask_summary = masker.summary()
    assert mask_summary["branch1"]["n_draws"] == 1
    assert mask_summary["branch2"]["n_draws"] == 1
    # both branches backpropagate into the shared model
    grad = model.decoder.fc_out.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
    enc_grad = next(model.id_encoder.parameters()).grad
    assert enc_grad is not None
    assert model.sparsification_enabled is False  # never leaks into eval paths

    # R-Drop structure: two train() forwards with IDENTICAL masks (p = 0 ->
    # all-ones mask) still disagree, because each forward's internal decoder
    # dropout is an independent stochastic pass
    model.eval()
    ones_mask = torch.ones(neural.shape[0], neural.shape[-1])
    with torch.no_grad():
        model.sparsification_enabled = True
        model.current_branch = 1
        model.current_p = 0.0
        model.current_keep_mask = ones_mask
        a, _ = model(neural, calib_trials=calib, side_features=side)
        model.current_branch = 2
        b, _ = model(neural, calib_trials=calib, side_features=side)
        model.sparsification_enabled = False
    model.train()
    model.sparsification_enabled = True
    model.current_branch = 1
    model.current_p = 0.0
    model.current_keep_mask = ones_mask
    c, _ = model(neural, calib_trials=calib, side_features=side)
    d, _ = model(neural, calib_trials=calib, side_features=side)
    model.sparsification_enabled = False
    assert torch.equal(a, b)      # eval mode + identical mask: deterministic
    assert not torch.equal(c, d)  # train mode: independent dropout passes


def test_p_stream_two_draws_per_step_sha_documented():
    two_per_step = sp.PStream(seed=sp.P_STREAM_SEED)
    for _ in range(100):
        two_per_step.next()
        two_per_step.next()
    one_per_step = sp.PStream(seed=sp.P_STREAM_SEED)
    for _ in range(100):
        one_per_step.next()
    assert two_per_step.stats()["n"] == 200
    assert one_per_step.stats()["n"] == 100
    # the SHA necessarily differs from the one-draw-per-step cells (R/S2/T/G)
    assert two_per_step.sha256() != one_per_step.sha256()
    # fixed draw order: replaying (branch1, branch2) x 100 reproduces the stream
    replay = sp.PStream(seed=sp.P_STREAM_SEED)
    for _ in range(100):
        p1 = replay.next()
        p2 = replay.next()
        assert 0.0 <= p1 < 1.0 and 0.0 <= p2 < 1.0
    assert replay.sha256() == two_per_step.sha256()


# ---------------------------------------------------------------------------
# runner contract
# ---------------------------------------------------------------------------
def _runner_source() -> str:
    return (ROOT / "scripts/run_consistency_cell.py").read_text()


def test_runner_budget_guard_and_sampler_contract():
    source = _runner_source()
    assert "steps_per_epoch != 33925" in source and "epochs != 48" in source
    assert "SessionBatchSampler(train_dataset, batch_size=32, shuffle=True, seed=args.seed)" in source
    template = (ROOT / "scripts/run_sparsify_cell.py").read_text()
    assert "SessionBatchSampler(train_dataset, batch_size=32, shuffle=True, seed=args.seed)" in template
    for token in ("p_sequence_sha256", "lambda_sweep", "LAMBDA",
                  "consistency_cell.PREREGISTRATION", "consistency_cell.COMPUTE_OPTION",
                  "skip_floor"):
        assert token in source, token
    # the pre-registration and compute option strings live in the module constants
    assert "two-view same-batch" in cc.COMPUTE_OPTION
    assert "compute-matched D control at 96 epochs" in cc.PREREGISTRATION
    assert "33,925 steps/epoch x 48 epochs" in cc.COMPUTE_OPTION


def test_runner_cli_rejects_existing_directory():
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    out_root = ROOT / "cache/test_consistency_cli"
    cell_dir = out_root / "cellC_paired_consistency"
    cell_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_consistency_cell.py"),
             "--device", "cpu", "--output-root", str(out_root)],
            capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
        )
        assert proc.returncode == 2
        assert "fresh cell output directory required" in proc.stderr
    finally:
        cell_dir.rmdir()
        out_root.rmdir()


def test_sealed_shared_files_frozen():
    launch = json.loads(
        (ROOT / "results/admission_arms_v1/armA_direct_t4_48/launch_receipt.json").read_text()
    )
    sealed = launch["source_closure"]["files"]
    for rel in ("scripts/run_admission_arm.py",):
        live = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert live == sealed[rel]["sha256"], f"sealed file drifted: {rel}"
    source = _runner_source()
    for rel in ("src/tfpd_lane/consistency_cell.py",
                "src/tfpd_lane/sparsification.py",
                "src/tfpd_lane/arm_common.py",
                "src/tfpd_lane/matched_scorer.py",
                "src/tfpd_lane/receipt.py",
                "scripts/run_admission_arm.py"):
        assert f'"{rel}"' in source, rel  # bound into the closure, fail-closed
