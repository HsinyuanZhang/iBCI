# HANDOFF: decoder-side directions — A10 ladder, MATCH, POP-HYP, entmax, anchored schedule

**Date:** 2026-08-15
**Status:** design record after re-audit of the three-lane decoder brainstorm. Authorizes nothing by
itself; every GPU arm needs its own frozen contract and CPU preflight. Queue order: swap-v2 (already
frozen, awaiting ROOT GO) keeps priority over every arm in this file.

## 0. Common gate standard (mixing precedent, binding)

- Primary endpoint: **absolute external sub-M T4 lift >= +0.03**. Interaction is reported and
  diagnostic-only; it can never rescue a missed absolute gate. The mixing screen died exactly this
  way (interaction +0.0718, absolute +0.0226).
- Within-sub-C floor: −0.03 (non-inferiority).
- Z4 sibling wherever the design admits one; frozen before launch; no post-hoc tuning.
- House discipline: any gate must be shown, on synthetic inputs, to be able to both pass and fail.

## 1. A10 readout-adaptation ladder — re-audited, Rank 1 (CPU-first)

Closed-form ridge adaptation of the frozen sealed-A2 checkpoints' readout on the target
calibration prefix, in capacity rungs: (a) per-covariate output affine (4 params); (b) per-covariate
additive delta on the scored-bin readout row, `Δr_c ∈ R^H` on top of the shared `fc_out`
(spint.py:103), fitted on prefix windows against dense prefix velocity.

**Re-audit findings — two facts change the design.**

1. A10 v1 was superseded for **two** reasons, both verified in
   `a10_no_backprop_cost_preflight.py:100-104` and the eval cell header: "arms do not use matched
   supervision" AND "the adaptation implementation is incomplete". The v1 preflight appends its
   blocker unconditionally and the eval cell raises on `--launch`. A revival must fix both, not
   just re-run: a frozen supervision ledger (label type, count, density, updated state, query
   boundary, online cost) and a complete, tested adaptation implementation before any GPU cell.
2. The "un-shared `fc_out` row" closure in the 0813 ledger killed per-covariate independent rows
   as a **training-time architecture** (dose-response inverted). Rung (b) is a deployment-time
   additive delta on a frozen shared readout — different estimand — but the adjacency must be
   disclosed, not silently ignored.

**Design.** Stage 0 (CPU, zero GPU): source-side refit no-op check — refit the same rungs by ridge
on source data with the encoder frozen; if source refit is a no-op, any target gain is cleanly
target-specific. Then dense-first: rung (b) with dense prefix velocity labels is the label-richest
upper bound of the whole lane.

**Kill-by-monotonicity (the design's main value):** if the dense-label upper-bound rung gains
< +0.03 external, the entire closed-form readout-adaptation lane — including any sparser or
template-target variant — dies in one screen, because dense prefix velocity dominates every
sparser target set for fitting a linear readout.

**Label-budget honesty.** This is a Tier-2 deployment contract: no backprop (ridge solve), but it
uses target labels (dense prefix velocity) to update parameters. Report with both label counts,
`information_matched=false`, beside — never inside — the Tier-1 cached-T4 conjunction claim.

**Controls.** Label-shuffled prefix refit must not exceed +0.01. Z4 sibling reported (does
adaptation rescue Z4; non-rescuing strengthens the carrier story).

**Kill risks.** Frozen-consumer insensitivity may extend to parameters, not only inputs (the H1
oracle precedent, −0.0029); the M30 prefix may be too short to condition 2H rows; the gain may
collapse into the 4-param affine (a real but small win — report it as such).

## 2. MATCH — closed-form session-moment calibration of the read-in, Rank 2 (CPU pre-gate first)

Per-session diagonal per-bin affine `x̃ = A_s x + b_s` on the neural input, moment-matched to a
frozen leave-one-session-out source-pool statistic, applied **before** the identity add; the
identity path stays byte-identical. `A_s, b_s` fold into `fc_in`'s first layer at deploy
(`W1' = W1 A_s`): zero marginal inference compute, label-free, no backprop.

**Re-audit finding — the premise is live.** Verified in `multisession_datamodule.py`: the SUA A2
path normalizes only the *behavior* (train-only); the neural x enters with raw session moments.
The site MATCH occupies is genuinely unoccupied.

**Escape.** No carrier-derived vector exists in this intervention, so the additive equivalence
class does not speak to it. It is not the closed whitening (that whitened the carrier descriptor /
cross-unit Gram *within* session as a decoding transform; this calibrates the x interface *across*
sessions against the source pool). Not FiLM/gain/gate: parameter-free, closed-form, unconditional.

**Why it can move the never-moved 0.26 within-external gap.** Every prior intervention acted at
source training on *reliance*; mixing proved reliance converts at ~30%. MATCH acts on the target
side of the gap: external units' pre-ReLU activations sit off the manifold `fc_in` was trained on.
It predicts a **Z4 co-lift** (carrier-neutral interface calibration) — that is the expected
signature, not a failure mode.

**Design.** Zero-GPU pre-gates, both killable before any training: (i) synthetic pass/fail — apply
a known per-bin affine to a held-out source session; the diagnostic must flag it, and a same-subject
held-out session must not be flagged; (ii) measure the actual external-session moment shift against
source session-to-session spread — kill before GPU if within ~2σ. Then `{match, none} × {T4, Z4}`,
seed 42. Gate: absolute external T4 >= +0.03; within floor −0.03; interaction reported, expected
≈ 0.

**Kill risks.** Attention LayerNorm may absorb per-session constants — pre-register diagnostics at
the hidden state, not only at x; sparse counts destabilize σ (freeze one variance-stabilizing
transform up front); calibration-to-evaluation non-stationarity within session.

## 3. POP-HYP — population-carrier hypernetwork on `fc_in`'s first layer, Rank 3 (GPU)

Freeze a 14-dim permutation-invariant summary `s` of the session's `[N,4]` carrier cloud
(mean 4 + upper-triangular covariance 10, source-z-scored). Modulate only `W1`:
`W1 ← W1 + U diag(Bs) V`, rank 4, `B` zero-init so `ΔW(0) = 0` and step-0 is parent-exact — the
same zero-init discipline as `carrier_post_pool` (h1_carrierid_spint.py:94-95). Everything else
byte-identical. ~500 new parameters, disclosed.

**Escape, re-verified.** The carrier's contribution is `(U diag(Bs) V) x` — bilinear in (s, x);
it cannot be written as `base(x) + offset(E)` for any E, so it is outside the additive lemma at
both sampled points by construction. Not the per-unit gating family (those were scalar per-unit
fields; this is one shared-operator modulation by a population statistic). Not SetKV (no token
surgery; jointly trained, honoring the frozen-crash lesson).

**Why the RS4 finding supports it.** SetKV's RS4 control proved attention is permutation-invariant
over the K/V set — the decoder's carrier use is population-codebook-like, not per-unit routing.
POP-HYP delivers the carrier exactly as a population statistic. It is the only surviving proposal
whose mechanism matches that measured fact.

**Z4 arm is an exact parent alias** via zero-init — the all-zero-port confound
(`Z4 − B0 = +0.0896`) is structurally cancelled.

**Design.** Zero-GPU pre-gate: `s` across the 27 source sessions must be non-degenerate
(top PC < ~95% of trace), else ΔW is constant and the arm is a capacity confound — kill before
GPU. Then `{hyper, none} × {T4, Z4}`, seed 42, jointly trained from scratch on the A2 lineage.
Gate: absolute external T4 >= +0.03 primary; interaction >= +0.03 as a secondary diagnostic.

**Kill risks.** Direction-balanced protocols may make T4 clouds near-identical across sessions
(the pre-gate exists for exactly this); rank 4 too weak; the hypernetwork ignores `s`.

## 4. Entmax-1.5 attention — demoted below POP-HYP (rank re-audit)

Replace softmax with entmax-1.5 over the K/V axis in an in-house MHA with identical projections.
The escape argument survives (it changes the link function, not carrier delivery; support-set
selection is a combinatorial object no logit residual can emulate) and it directly answers the
measured "softmax has no veto" result.

**Why demoted.** The just-completed value-mask probe measured the decoder's actual allocation:
pruning the bottom-25%-by-m units moves external T4 by only +0.002. If a quarter of the pool,
selected by tuning quality, is already worth nothing to remove, the mass entmax could learn to
veto on the T4 arm may be small — the T4-painted identity plausibly already down-weights the rows
a veto would target. The strong veto demand is on the Z4 arm (external −0.143), which is the
control, not the product.

**Design if run.** Pre-GPU check piggybacks on the mask probe receipts (already in hand — the
answer is unfavorable, which is why this is held). `{softmax, entmax} × {T4, Z4}`, 3 seeds only if
promoted. Gate: absolute external T4 >= +0.03; a Z4-equal gain is generic regularization — kill
per the mixing precedent. Largest implementation cost of the five (custom attention).

## 5. Anchored decoder schedule — Rank 4 (piggyback only)

Freeze the decoder for epochs 0–4 of source training (exactly the epochs the 5–12 scoring window
already discards), train only the identity/identity-carrier encoder; unfreeze for epochs 5–12.
Data, loss, eval, total schedule, and compute byte-identical. Zero new hypers.

**Escape.** Training is already joint (verified in A2 receipts: `freeze_decoder = False`), so
"joint training" is not a novelty — but a schedule *within* joint training is untested anywhere in
the receipts. Mechanism: at epoch 0 the identity tokens are near-random; the jointly-trained
`fc_in` reorganizes around garbage identity before the encoder stabilizes, co-adapting decoder
keys to source activity statistics — the same co-adaptation that goes misleading externally.
Anchoring forces the encoder to shape identity to a fixed decoder.

**Conversion logic.** Unlike mixing/swap it does not try to increase carrier reliance (the
quantity measured to convert poorly); it reduces decoder source-specific co-adaptation — a
generalization fix whose natural signature is absolute external T4.

**Design.** `{anchored, joint} × {T4, Z4}`, seed 42. Gate: absolute external T4 >= +0.03; within
floor −0.03; a Z4-symmetric lift is still an absolute win but is reported as a generic curriculum
effect, not a carrier mechanism. Run only as a rider on another arm's GPU allocation.

**Kill risks.** If the MC-Maze teacher prior is itself the mismatch (P3's blame), anchoring to it
longer hurts everywhere — a clean negative; effect may sit in the ±0.02 band like other
schedule-adjacent arms.

## 6. What was rejected at re-audit (do not re-propose)

- **CKLV (carrier-keyed rows with live values)** — killed algebraically: two rows sharing a value
  satisfy `Σ (e^{q·k} + e^{q·k̃})/D · v = Σ e^{q·k + log(1+e^{q·(k̃−k)})}/D · v` — exactly a
  per-unit additive logit offset, i.e. the closed rank-residual class (−0.0031). The proposing
  agent's own pre-kill of "contrast keys" applies verbatim to its own headline idea.
- **T-FILT (carrier-conditioned temporal filter mixing)** — per-unit carrier conditioning family
  has three negative scalps (gain −0.0021, electrode gate −0.0108, FiLM +0.0034) and RS4 shows
  per-unit routing is not how the decoder uses the carrier.
- **Value centering** — pool-mean subtraction is largely absorbed by output biases; adjacent to
  the whitening closure (+0.0017).
- **The `rep`-query sub-region and `fc_out` architecture** — algebraically closed at depth 1
  (rank-≤C score perturbation + output bias; per-covariate rows measured inverted). The boundary
  agent's additive lemma (pre-ReLU ≡ W-add; post-MLP ≡ A1; a third entry must be non-additive)
  should be recorded in the closure ledger.

## 7. Sequencing

```
Now (CPU, zero GPU):   A10 Stage 0 source-refit no-op + supervision ledger;
                       MATCH pre-gates (synthetic flag test + moment-shift measurement).
GPU queue:             swap-v2 seed-42 first (frozen contract, awaiting ROOT GO).
After swap-v2 verdict: POP-HYP seed-42 (if promoted); anchored as rider on any GPU batch.
Held:                  entmax (unfavorable mask-probe prior; largest implementation cost).
Never:                 CKLV, T-FILT, value centering, query/readout architecture re-opens.
```

The A10 dense-first screen and MATCH pre-gates can run while B0 and swap-v2 occupy the queue;
neither touches a sealed receipt, and each is killable before spending GPU.
