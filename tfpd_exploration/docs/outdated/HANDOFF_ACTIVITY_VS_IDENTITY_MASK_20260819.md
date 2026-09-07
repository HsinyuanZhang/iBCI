# Handoff: is the sub-population effect activity ablation or identity ablation?

Date: 2026-08-19
Status: research-direction handoff. Defines two cells and pre-registers their readings. Authorizes no
GPU launch by itself.

Predecessor: `HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md` (executed and closed
2026-08-19). All prohibitions in that document's §4 are inherited unchanged.

---

## 1. Why this cell and not another module

The predecessor round attempted four design additions — subset geometry (S2), an explicit invariance
objective (C), temporal capacity (W), head capacity (DH) — and all four failed. The design space
around the perturbation is exhausted at current evidence. What is *not* exhausted is a decomposition
that no cell has yet touched, and it can change the headline claim rather than decorate it.

Established, governing convention (last-bin, equal-session, variance-weighted, final-four SWA,
seed 42):

| system | external | vs Arm A |
|---|---:|---:|
| D — whole-unit ablation | **0.4179** | +0.1576 (14/15) |
| T — true removal, no gain | 0.4081 | +0.1477 (15/15) |
| Arm A — no perturbation | 0.2604 | — |
| R — elementwise, matched amount | 0.1877 | −0.0727 |
| G — gain only | 0.2699 | +0.0095 |

Five controls agree that whole-unit structure is required and that nothing else about the
implementation matters. But **every perturbation so far has been applied to a sum of two very
different signals.**

## 2. The code fact that opens the question

`streaming_calibration_exp/src/models/components/spint.py:445-455`:

```python
src = src + id                              # BxNxW activity  +  BxNxW identity
dropout_mask = F.dropout(torch.ones(B, N), p=p, training=self.training)
src = src * dropout_mask.unsqueeze(-1)      # kills BOTH components together
```

Two facts matter:

1. The mask is applied to `activity + identity`, so a dropped unit loses its activity **and** its
   identity tag simultaneously. No cell has separated them.
2. `id` comes from `id_encoder.forward_batch(calib_trials)`
   (`tfpd_exploration/src/tfpd_lane/subpop_cells.py:322-325`) — it is derived from **calibration
   neural activity**, i.e. a session fingerprint. It is *not* the 4-dim analytic carrier.

Fact 2 is what makes the decomposition load-bearing, because the lane's other strong result is an
identity-source asymmetry: carrier-derived identity transfers externally, while activity-derived
identity is externally *negative* (`spintshape_z4` external −0.2936).

**Live hypothesis:** D's benefit may not be population sub-sampling at all, but **session-fingerprint
ablation** — randomly destroying the calibration-activity-derived identity signal, which is precisely
the component that fails to transfer to held-out sessions. Under that reading, "we train on random
sub-populations" is the wrong mechanism sentence and "we stochastically suppress session-specific
identity" is the right one.

A useful property of `F.dropout`: it returns survivors already scaled by `1/(1-p)`, so the gain is
fused into the mask tensor. The three cells below can therefore share **one identical mask draw** and
differ only in where it is applied. This is a genuine one-factor contrast at the application site,
not a re-bundling.

---

## 3. The two cells

Both reuse the existing harness. The insertion point is
`tfpd_exploration/src/tfpd_lane/subpop_cells.py:384-406` (`decode_with_identity`), where line 392 does
`src = src + identity`. Today `apply_perturbation` receives only the summed `src`; these cells need it
to see the two components while still separate. That is the whole build — a hook signature change and
two one-line variants.

| cell | forward | a dropped unit becomes |
|---|---|---|
| D (reference) | `(activity + identity) * mask` | a constant token: no activity, no identity |
| **AM** | `activity * mask + identity` | present and identifiable, but silent |
| **IM** | `activity + identity * mask` | audible but anonymous |

Requirements common to both:

- The mask draw must be **identical in distribution and code path** to D's: one shared `p ~ U(0,1)`
  per training step, whole-unit Bernoulli, the same `min_keep` clamp, the same `F.dropout` call so
  the `1/(1-p)` scaling is inherited rather than reimplemented. Reuse `UnitRemovalMasker`.
- Evaluation path must be bitwise equal to the parent path, as T's already is
  (`apply_perturbation` returning the input untouched when the perturbation is disabled).
- Everything else held at Arm A / D settings: 2 heads, no pretraining, no teacher checkpoint, no
  width change, no `behavior_scaling_factor` change, seed 42.

### Pre-registered readings

Band for "equal to D": |Δ external| < 0.03 at the governing granularity, consistent with the T−D
band already accepted (−0.0099).

| outcome | reading | consequence |
|---|---|---|
| IM ≈ D and AM ≪ D | the mechanism is **identity/fingerprint ablation** | headline rewrite; the claim becomes far more specific and connects to the carrier thesis that is the lane's actual subject |
| AM ≈ D and IM ≪ D | the mechanism is **activity sub-sampling** | the population story stands as written; this round is a confirmatory control |
| both ≪ D | the mechanism requires **joint** ablation — the token must be fully replaced | population story stands in a stronger form; report as the decomposition's floor |
| both ≈ D | suspicious | conflicts with R, where a matched amount of elementwise noise came in *below* baseline. Do not report as a finding; investigate the mask draw and the evaluation-path equality first |

The fourth row is the sanity check. R is the reason it is a red flag rather than a result: if noise on
either pathway alone suffices, R should not have failed.

---

## 4. Order and prohibitions

Two cells, independent, parallelizable across two GPUs. Neither blocks the other. There is no
zero-GPU step available this round — the existing checkpoints cannot answer this question, because the
distinction is a training-time one.

Inherited prohibitions, all still in force: no dropout-strength sweep; no revival of the 64-head line
(DH −1.1373); no revival of carrier-sector sparsification (S2 gate rejected); no width change; no
pretraining or teacher checkpoint in any training path; no silent `behavior_scaling_factor` change;
**cell E stays closed** (see predecessor §0.2).

Added prohibition: do not sweep a mixing coefficient between AM and IM. A partial-mask interpolation
is a fifth design addition, and the predecessor round established that design additions are currently
a negative-expectation move. Answer the discrete question first.

**Standing claim constraint, unchanged and now twice deferred:** no superiority claim over A2 is
admissible until D (or whichever system becomes the headline) has seeds 43/44, because A2 is a
three-seed reference with an external spread of 0.066. This round does not relieve that requirement.
If the operator wants the paper rather than another mechanism finding, the seeds are the higher-value
GPU allocation and this document should be deprioritized accordingly.

---

## 5. Receipt requirements

Carry forward all six from the predecessor's §5, and note two specifically:

1. Record the **realized** perturbation statistics per cell: the `p` distribution, realized
   surviving-unit counts, the clamp hit rate, and — new — an explicit assertion that the mask tensor
   applied in AM and in IM is drawn by the same code path as D's.
2. Report both granularities and both aggregations, with (last-bin, equal-session,
   variance-weighted) labelled governing, plus the 2014 / 2015 date-block split and
   `sub-M_ses-CO-20141203` separately. D's advantage on that session is +0.4863, so it moves
   aggregates on its own.

Two further items, both inherited defects that must not be reproduced:

3. Include a `supersedes` field. Three receipts in the predecessor round
   (`subpop_score_v1`, `_r1`, `_r2`) lack one; they turned out to be cumulative rather than
   corrective, but that had to be established by diffing contrasts instead of being read off.
4. `scripts/run_tfap_stage3.py:369-374` still has no branch for mechanism-pass-without-
   engineering-pass and mislabels that state `BOTH_GATES_FAIL__STOP_ROUTE`. Do not copy the pattern
   into this round's scorer.

---

## 6. Framing constraints

- The mechanism sentence licensed today is "training on random sub-populations" (T settled this; the
  predecessor's interim restriction is lifted). **This round can revoke it.** If IM ≈ D, the sentence
  must change to identity suppression, and every document asserting the population framing needs
  revision — including the predecessor's §0.1.
- The carrier remains necessary regardless of outcome: A2's Z4 external is −0.1461 / −0.1189 /
  −0.1108. The claim is always sparsification **plus** carrier.
- Report whichever of AM/IM fails as a negative result that narrows the mechanism. The predecessor
  round's value came mostly from its four failures, not from D.
- Any document citing an A2 external advantage of +0.180 is wrong; the matched figure is
  **+0.0857 (8/15, CI [+0.020, +0.165])**.

## 7. Receipts this document relies on

- `results/subpop_score_v1_r2/subpop_score_receipt.json` — C/G/T/W, authoritative
- `results/subpop_step0c_v1/step0c_receipt.json` — unit-loss curves, ensembling excluded
- `results/sparsification_score_v1/sparsification_score_receipt.json` — R and S2
- `results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json` — A2 matched, 3 seeds (v1 VOID, missing `/5.0`)
- `streaming_calibration_exp/src/models/components/spint.py:445-455` — the `src + id` then mask fact
- `tfpd_exploration/src/tfpd_lane/subpop_cells.py:322-325,384-406` — identity provenance and the insertion point
