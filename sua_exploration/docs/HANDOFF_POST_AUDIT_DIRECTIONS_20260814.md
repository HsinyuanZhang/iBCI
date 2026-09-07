# HANDOFF: post-audit design directions — misleading identity and carrier routing

**Date:** 2026-08-14
**Status:** design record from a negative-evidence audit plus two independent brainstorm lanes.
**Authorizes:** nothing by itself. Every idea below needs its own frozen contract before any GPU run.

## Root review correction

This handoff is retained as ideation evidence, but the active queue follows these corrections:

- H1 shows strong **NeuronID encoder capacity redundancy** under the matched H-S width experiment; this does not
  establish that carrier headroom is exhausted.  H1 remains open at lower priority.
- A2 external Z4 retains the activity-derived identity path and only zeros the carrier after the ordinary T4
  substrate is formed.  Its negative score therefore does not by itself prove that activity identity is worse than
  no identity.
- Misleading-identity swap-v2 must use one session/epoch-consistent partial donor mapping, matched in a separately
  carried unmasked T4/rate descriptor space and shared byte-for-byte by T4 and Z4 siblings.  Per-window or arbitrary
  within-batch swaps are invalid.
- Held-in-domain improvement is not required for the robustness mechanism.  The gate is within-domain
  non-inferiority plus external absolute T4 lift; an explicit swapped-input robustness diagnostic distinguishes
  learned down-weighting from ordinary regularization.
- Gauge augmentation is not in flight, and all-bin supervision is not an automatic fallback.  A fair size/compute-
  matched CEBRA comparator remains the final Track-B fallback after the internal mechanisms are frozen.

Pseudo-session Stage P is terminal-negative on its frozen absolute-accuracy gate: external T4 lift was
`+0.022609 < +0.03` (within `+0.009485`), so no extra seeds or tuning were run.  SetKV Stage 0 is also terminal:
SetKV-T4 changed R2 by `-0.738673` within subject and `-0.638865` externally, with an effectively zero
T4-versus-row-shuffle attachment contrast.  The active internal order is value-weighted mask followed by the
standalone misleading-identity swap-v2; the H1-excluded CEBRA route continues through independent CPU/source
gates in parallel.

## 1. What the audit of the negative evidence concluded

Four wrongful-kill modes were checked: silent implementation failure, underpowered gates,
confounded controls, and over-broad closure.

- **Zero silent failures.** Every measured-negative arm has an attachment or content control
  proving the intervention fired: A1's TS4 mis-attachment costs ~0.43 R² per session; FiLM's
  TS4 arm loses 0.29; the logit-residual triplet (aligned 0.5956 / additive 0.5916 / shuffled
  0.5890) shows content-specific routing with near-zero net effect. The two known dead code
  paths (`set_carrier_noise_cholesky`, `_subset_last_dim`) never entered any scored receipt.
- **Gates were fixed before use.** The unattainable Wilcoxon n=3 gates were caught at design
  time; no terminal decision rests on an impossible gate.
- **Two underpowered stops, wording only.** A1 (+0.0128, one seed) and B1 (+0.0137, three
  seeds, spread 0.081) failed their +0.03 routing gates correctly, but the papers must say
  "routing pilot below the pre-registered threshold", not "mechanism disproved".
- **No re-runs recommended.** The fusion family (9 arms), the descriptor family (T8/T3K/
  whitening/go-cue), and the M30 budget plateau are three independently saturated families.

## 2. The two problems are different, and directions must name which one they serve

- **H1 (human, 7-DoF): NeuronID capacity is redundant; carrier headroom is unresolved.** The
  architecture-preserving 60K activity-only identity path is non-inferior to the 5.97M path
  (four-date mean delta +0.0364), but this bounds excess encoder capacity rather than the value
  of better carrier information. H1 therefore remains open at lower priority for information,
  objective, and generalization tests; the current result does not close carrier-side work.
- **SUA/RT/M2 (monkey, 2-D): real carrier headroom.** A2 measured external sub-M T4 = +0.341
  against activity-only Z4 = −0.143 (interaction +0.236). Everything below targets this side.

## 3. The unoccupied condition: identity that lies

The sealed A2 external Z4 mean of −0.143 means that under subject shift, activity identity is
not absent and not noisy — it is **validly computed and actively misleading** (worse than a
zero carrier, per the attachment-sensitivity result). No run or held intervention rehearses
that condition:

| Intervention | Co-occurrence | Per-unit correspondence |
|---|---|---|
| Pseudo-session mixing (running) | broken | correct (rows indivisible) |
| CF1 activity dropout (held) | correct | absent |
| B2 carrier corruption (safety) | correct | carrier broken (opposite direction) |
| **Misleading-identity swap (new)** | correct | **broken, valid-looking** |

## 4. Rank 1 training idea — rate-matched misleading-identity swap

During source training, with frozen p=0.5, replace a unit's activity-identity row
(`mean_h` in the identity path) with the row of a **rate-matched donor unit drawn from a
behavior-matched source window**, leaving that unit's query population and T4 row untouched.
Task loss only. The Z4 sibling gets the identical swap schedule.

- **Why it escapes the closed families:** CF1 removes identity; this corrupts it with
  valid-looking content. B2 corrupts the carrier, the opposite direction. No invariance or
  alignment term, so the headroom-screen and 99.7%-decodable falsifications do not apply.
- **The naive version is rejected:** within-batch random swap leaves a detectable
  rate-mismatch cue (x_i vs mean_h_j) that exists in training but not deployment, because
  sub-M identity is self-consistent. The donor must be rate-matched to suppress that cue.
- **Pre-registered diagnostic:** if the gain comes from mismatch detection rather than
  learned down-weighting, held-in-domain evaluation (identity never swapped) should show no
  gain — that outcome kills the mechanism reading.
- **Design:** `{T4,Z4} x {swap,no-swap}`, seed 42 first. Gate: absolute external sub-M T4
  lift >= +0.03, within-sub-C floor −0.03, interaction reported but non-rescuing. Kill on any
  gate failure; no donor-rule or p tuning after scores.
- **Kill risks:** the consumer may still learn a residual detection cue; if the identity path
  is effectively low-capacity the swap degenerates toward dropout and CF1 subsumes it.

## 5. Rank 1 architecture idea — carrier tokens as set members (SetKV dual routing)

Keep the parent additive identity path byte-identical. Additionally evaluate the existing
identity encoder on the carrier-only input, map through the **shared** `fc_in`, and append
these N rows to the K/V set at the attention call. Zero new parameters. Queries then hold
separate routing weights over carrier descriptors, decoupled from the weights over activity.

- **Why it escapes the additive equivalence class:** every closed arm delivers the carrier as
  a per-unit additive offset before the attention dot product, where it collapses into the
  same score/value perturbation. Here softmax competes **between token kinds**, so the
  carrier readout is not gated by corrupted activity keys — the decoder can route around
  poisoned identity instead of overriding it. The latent-query closure does not apply (K/V
  rows are read by queries; latent queries failed for lack of query mixing).
- **Controls, all pre-declared:** Z4 sibling is the carrier-only token from an all-zero input
  (sink present, content absent, so dead-port effects cancel inside the interaction); a
  duplicate-activity-token arm (B15P pattern) proves any gain is not set size; RS4 row
  permutation is the attachment control.
- **Two-stage design caps the downside.** Stage 0 is forward-only on sealed A2 checkpoints:
  informative if positive, non-killing if flat (frozen-consumer insensitivity is not closure).
  Stage 1 is `{W-add, SetKV} x {T4, Z4}` x 3 seeds, gate: interaction >= +0.03, 3/3 seeds
  positive, plus absolute external T4 lift (anti-Z4-crash rule).
- **Implementation note:** on SUA the path lives in the streaming student's B3S identity
  encoder, not the H1 `H1CarrierIdSpint` file. The contract must bind the right module.
- **Kill risks:** source training never practices unreliable activity identity, so the extra
  tokens may be ignored; the carrier-token routing weight is session-static per covariate, so
  if under-exploitation is dynamic, this is flat.

**Terminal update.** Stage 0 produced no usable frozen-forward signal and does not promote Stage 1.  Aggregate
SHA-256: `1d6ea6f9fd12623094341b41157ca18c6be26bf134554a939ce53ce7b4009802`.  This closes the queued SetKV
implementation, not every jointly trained set-token consumer.

## 6. Zero-GPU probes to run first

1. **Carrier mask via `key_padding_mask`**: mask bottom-quantile tuning-gain units by a rule
   frozen on source sessions only. The plumbing exists in `CrossAttentionLayer` and is never
   used. Pre-registered mechanism check — attention mass vs m-rank on source: if weakly
   tuned units already get ~zero mass, predict flat and stop **before** scoring.
2. **SetKV Stage 0** (see above).

Both are forward-only on sealed checkpoints; neither can damage an existing receipt.

## 7. Held, not dead

- **Multi-cohort source pool** (add M2 and RT as source sessions): highest ceiling, genuine
  subject heterogeneity instead of simulated; but it changes the source distribution the A2
  estimand sits on, needs full matched retraining, and its primary endpoint is absolute
  external T4 with carrier interaction secondary.
- **Depth-2, reopened on a structural argument:** at one layer, unit tokens are built
  independently and each routing score depends only on (q_c, k_i), so the carrier field
  cannot jointly condition how any single unit is read; a second hop scores queries against
  values that already summarize all units. This is a computation impossible at depth 1, which
  the closed list never tested. Cost (new teacher lineage, capacity confound) keeps it held.
- **Identity-discordant mixing**: permute identity rows inside a passing pseudo-session mix.
  Conditional on the seed-42 mixing gate passing first.
- **Dropped this round:** inference-time carrier-noise marginalization (frozen consumers have
  twice refused better carrier information; K-fold inference threatens the cached-state
  story); identity-consolidation stop-grad phase (a soft CF1, subsumed).

## 8. CEBRA verdict

The CEBRA-inspired mechanism search has narrowed to the currently queued routing and identity
interventions; H1 is lower priority rather than closed.  Track-B strict-27 source-only SUA and pseudo-MUA
loader/authority construction is now complete, with exact pooling replay in all 27 pseudo-MUA sessions and no
target access, CEBRA fit, GPU, or score.  Target/query lineage and source-only geometry selection remain before
the comparator can run.  Its accuracy family is `MultiSessionSolver`.  `UnifiedSolver` is kept only as a structural
boundary: its input width concatenates all training-session units, so it cannot produce a valid
few-shot accuracy score for a standalone unseen target session.

## 9. Sequencing

```
Step 0 (forward-only): carrier-mask mechanism check; SetKV Stage 0 is terminal-negative.
Step 1 (GPU, one contract): rate-matched swap, seed-42 cell first.
Dependency: if pseudo-session mixing passes seed 42, swap re-parents onto the mix
(discordant phase); if mixing fails, swap stands alone.
Held: multi-cohort pool; depth-2.
```

Pseudo-session Stage P stopped at seed 42, so the swap remains a standalone A2-parent experiment; it is not
re-parented onto pseudo-session mixing.
