# Handoff: SPINT Decoder — Verified Structure and Ranked Improvement Directions

Date: 2026-08-17
Revision: 2 (supersedes revision 1 of the same date)

Status: research-direction handoff. Authorizes no GPU launch, no target/organizer-held scoring,
no formal evaluation. **§5's ranking is resolved as of 2026-08-19 — see the outcome note below
before using it.**

---

**Outcome note (2026-08-19). Priorities 1 and 2 have both executed, with opposite results, and the
ranking in §5 should no longer be used to schedule.**

Priority 1 became the lane's headline result. Run as `dynamic_dropout(0,1)` rather than the frozen
`0.10` proposed here, it lifted external from 0.2604 (Arm A) to **0.4179** — +0.1576, 14/15 positive
— at the governing granularity (last-bin, equal-session, variance-weighted, seed 42), passing A2's
matched 0.3461 outright. Two subsequent rounds isolated the mechanism with five controls: elementwise
corruption at matched amount is *harmful* (R 0.1877), the `1/(1-p)` gain contributes nothing
(G +0.0095), true removal via `key_padding_mask` is equivalent to the placeholder form (T 0.4081,
within band), carrier-ordered subset geometry adds nothing (S2 0.3767), and making the invariance an
explicit objective actively hurts (C 0.3823). The licensed sentence is *training on random
sub-populations is necessary and sufficient; none of its implementation details matter.*

Priority 2 became cell W and is a **negative result**: −0.0902 vs Arm A external (3/15), and worse at
both granularities. The temporal-basis residual design described in §5 was built as specified; the
design does not deliver. Do not reschedule it.

Two of this document's framings are corrected by those rounds:

- Priority 1's rationale here says the dropout is "unit-set robustness regularization … not an
  intervention on activity-identity overfitting". That opposition is **not yet established**, because
  the mask is applied to `x + id` and so ablates activity and identity *together*. Which of the two
  carries the effect is exactly the open question, now scheduled as cells AM/IM in
  `HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md`. If the identity-only mask reproduces D, revision
  1's withdrawn claim is partially reinstated and this paragraph must be rewritten.
- Priority 1 specified "on a pass, add seeds 43/44". It passed decisively. **The seeds were deferred
  twice and are still missing**, which is the only blocker on a formal superiority claim over A2 (a
  three-seed reference with 0.066 external spread).

Governing document for scheduling: `HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md`. Closed and
executed upstream: `WORKORDER_BEAT_A2_SPARSIFICATION_20260818.md`,
`HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md`.

---

**Revision note.** Revision 1 was written against the superseded epoch-window estimator and
before three sets of results were checked: the Gate-3/Gate-4 admission-arm outcomes, the terminal
`A1` and decoupled-K/V verdicts, and the external matrix. Revision 2 keeps §1 and §2, replaces
the evidence base entirely, removes two directions that already have terminal STOP receipts,
demotes identity widening, and re-orders everything behind the in-flight TFAP round. §8 lists
every claim withdrawn from revision 1 so nothing propagates silently.

Audience: whoever schedules the next round. Every load-bearing claim carries a `file:line` or a
receipt path.

---

## 1. What the network actually is

Teacher-free `spintshape` = A2-shaped B3S calibration encoder + SPINT decoder. Verified in
`streaming_calibration_exp/src/models/components/spint.py:430-468` and
`streaming_encoders.py:364-444`:

```text
id_i  = post_pool( cat( mean_M pre_pool(calib_i), T4_i ) )   # B3S, shared over units, M=30
src_i = x_i + id_i                                            # spint.py:445
h_i   = fc_in(src_i)                                          # Linear(50,512) ReLU Linear(512,512)
q     = fc_in(rep)                                            # rep = nn.Parameter(1, C=2, 50)
z     = CrossAttn(Q=q, K=V=h)                                 # spint.py:464
y     = fc_out(z)                                             # Linear(512, 50)
```

Four properties that constrain which changes are possible at all:

1. **Identity is added in window space, not hidden space.** `id_i ∈ R^50` is added directly to the
   binned spike-count window (`spint.py:445`), *before* the shared read-in, so identity must
   express itself as an additive pseudo-spike-waveform.

2. **K and V are the same tensor, with separate learned projections.**
   `CrossAttentionLayer:30-34` passes `key=value=key_value_norm` into `nn.MultiheadAttention`,
   whose `in_proj_weight` supplies distinct W_Q/W_K/W_V. The tying is at the *source*, not the
   projection.

3. **The query slots are "virtual units".** `rep` is a learnable parameter in window space, put
   through the *same* `fc_in` as the units (`spint.py:462`). Slot count is hard-tied to
   `num_covariates = 2`, which is also the output channel count — see §5 Priority 2.

4. **There is no temporal model anywhere.** `fc_out = Linear(512, 50)`: each of the two slots
   emits an entire 50-bin trajectory from one 512-d vector. No recurrence, no within-window causal
   structure.

No per-unit parameters, no unit-embedding table, no fixed unit slots. The model is
permutation-invariant over units and accepts variable `N`.

---

## 2. Parameter budget

Counted from the terminal `spintshape_t4` checkpoint (live total 3,510,842):

| block | params | share |
|---|---:|---:|
| cross-attention block | 3,152,384 | 89.8% |
| ⤷ of which the FFN alone (512→2048→512) | 2,099,712 | 59.8% |
| per-unit read-in `fc_in` | 288,768 | 8.2% |
| output `fc_out` | 25,650 | 0.73% |
| `decoder.fc_id_out` (**dead**, bypassed by B3S) | 25,650 | 0.73% |
| calibration identity encoder (all of B3S) | 18,290 | 0.52% |
| query slots `rep` | 100 | 0.003% |

B3S runs at width 64 (`pre_pool = Linear(100,64)`,
`post_pool = Linear(68,64) → Linear(64,64) → Linear(64,50)`), compresses to 50 dimensions, is added
to spike counts, and only then read into a 512-d decoder. 59.8% of the budget sits in one
feed-forward block; the entire output surface is 0.73%.

**This is an allocation audit, not an attribution.** It licenses a bottleneck *hypothesis* only.
It does not show that any block is under-capacity, and revision 1's inference from this table to a
top-priority widening experiment is withdrawn (§8).

**Config trap to fix while you are in there.** `tfpd_exploration/src/tfpd/spintshape_module.py:43`
passes `id_hidden_dim=128`, but that argument is **dead for B3S**: `build_encoder` routes
`id_hidden_dim` only to B2/`LatePoolEncoder` (`streaming_encoders.py:285-292`), while the B3S
branch receives `hidden_dim=64` (`streaming_encoders.py:3386-3393`). Rename the consumed argument
to something unambiguous (`b3s_hidden_dim`) so no future cell believes it ran width 128.

---

## 3. Current evidence base

**Use these numbers. The epoch-5–12 window means used in revision 1 are superseded** — they
average R² over 8 checkpoints, correspond to no deployable model, and are biased upward relative
to A2's metric. All rows below are final-four SWA under the matched scorer with equal session
weight.

### Within development (6 sessions, `results/gate3_within_screen_v1/within_screen_receipt.json`)

| system (SWA) | within mean R² |
|---|---:|
| **A `direct_t4_48`** | **0.5163** |
| C `direct_t4_33` | 0.5135 |
| spintshape 12-epoch baseline | 0.4734 (sealed 0.47272) |
| B `z4_pretrain_then_t4` | 0.3392 |
| A2 teacher-init (cited, mean-level only) | 0.5750 |

### External sub-M (15 sessions, `results/gate4_arm_external_v1/final_external_matrix_v2_receipt.json`)

| system | external native R² | n+ /15 |
|---|---:|---:|
| A2 teacher-init (cited only, own authority) | **0.341** | — |
| **A `direct_t4_48` SWA** | **0.1610** | 8/15 |
| spintshape 12-epoch T4 SWA | 0.0902 | — |
| C `direct_t4_33` SWA | 0.0775 | 5/15 |
| pv_t4 (**void**, see §6) | −0.0164 | — |
| large_t4 | −0.0848 | — |
| spintshape 12-epoch **Z4** SWA | **−0.2936** | — |
| large_z4 | −0.7367 | — |
| bl_t4 | −0.9597 | — |

Paired external contrasts: `A − spintshape-12ep = +0.0708` (13/15, bootstrap 95%
[+0.046, +0.094], exact sign p = 0.0074); `A − C = +0.0835` (15/15, [+0.057, +0.113], p = 0.0001).

### Four facts that drive the ranking in §5

**(a) The objective is external transfer, and the external spread is far larger than the within
spread.** Within, the best and worst trained T4 systems differ by ~0.18; externally the same
systems span 0.161 down to −0.96. A2 is `+0.0587` ahead of Arm A within but `+0.180` ahead
externally. Any direction that raises within without raising external is not progress.

**(b) Activity-derived identity is a negative asset under subject shift.** `spintshape_z4`
external is **−0.2936**, against +0.0902 for the same system with T4. This is the central reason
identity widening is demoted: widening the calibration-activity encoder plausibly widens session
fingerprint capacity too, and would be expected to raise within while flattening or lowering
external.

**(c) Carrier-derived identity transfers much better than activity-derived identity.** Comparing
the two single-source arms: `large_t4` (carrier-only identity) versus `spintshape_z4`
(calibration-only identity) — within, large is ahead by `0.2562 − 0.1985 = +0.058`; externally it
is ahead by `−0.0848 − (−0.2936) = +0.209`. **The gap widens ~3.6× under subject shift.** This is
confounded by topology and must always be reported as such, but the direction and magnitude are
large, and it is the most interesting mechanism result the lane currently owns. It answers the
contrast that `HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md` §8 #2 had posed as unmeasured.

**(d) Long-schedule direct T4 is the surviving positive engineering result, and exposure order is
harmful.** Going 12 → 48 epochs with warmup+cosine and final-four SWA is worth `+0.043` within
(5/6) and `+0.071` external (13/15). Z4-first pretraining then T4 is `−0.177` within against both
a compute-matched (A) and an exposure-matched (C) control, 0/6 sessions positive against each.
The exposure-matched control C is what makes this interpretable: the loss is caused by the
ordering, not by Arm B's 15 fewer T4 epochs.

**On the T4−Z4 "+0.25 constant".** On the old estimator, T4 minus Z4 was `+0.2589` (spintshape),
`+0.2503` (large), `+0.2490` (A2). The correct reading is narrow: **conversion within the simple
fusion family appears empirically stable, so the prior for another plain concat / add-site / FiLM
variant is low.** It is *not* an information-theoretic ceiling on T4 — those three numbers come
from independently trained systems with different baselines, initializations and optimization, and
they say nothing about T4 under a better objective, better pretraining, or a different decoder
representation. Revision 1's "saturated" claim is withdrawn (§8).

---

## 4. Scheduling reality

Two GPUs are occupied by the in-flight TFAP round (`results/tfap_stage2_v1/finetune_pt4`,
`finetune_pz4`). Throughput reference: 33,925 optimizer steps per epoch, ≈10.4 min/epoch, so a
48-epoch cell is ≈8h20m. **Do not queue a new decoder GPU cell before TFAP terminates and is
scored.**

`scripts/run_tfap_stage3.py` has two defects that must be fixed before those cells terminate: a
CI escape clause in `mechanism_gate` (`:304-308`) and integrity checks that run *after* the
15 external target sessions are opened (`scored` is populated at `:206-252`, Stage-2 terminal /
SWA-SHA / closure verification at `:283-298`). This is tracked as a separate scorer work order and
is a prerequisite for anything in §5.

---

## 5. Ranked directions

### Priority 0 — finish and correctly score TFAP; open no new decoder cell

TFAP is already testing the largest available effect, and §3(a) says that effect is where the
headroom is. Read it as three components, not a binary:

```text
generic pretraining component  = P-Z4 − Arm A
task-frame aligned component   = P-T4 − P-Z4
total engineering component    = P-T4 − Arm A
```

Both components can be positive simultaneously. Mechanism gate must be
`P-T4 − P-Z4 external mean ≥ +0.03` **and** `≥ 10/15` positive, with the bootstrap interval
reported and never used to rescue a sub-threshold mean.

Branching after TFAP:

| outcome | next step |
|---|---|
| `P-T4 − Arm A ≥ +0.03` and `P-T4 − P-Z4 ≥ +0.03`, ≥10/15 | seeds 43/44 on P-T4. Task-frame-aligned whole-model pretraining is the lead training contribution; do not start a decoder sweep. |
| both beat Arm A, but `P-T4 − P-Z4 < +0.03` | keep generic whole-model pretraining as an engineering recipe; do not claim task-frame alignment is the mechanism. |
| neither beats Arm A | proceed to Priority 1 and 2. |

### Priority 1 — one unit-dropout cell, frozen strength, run alone — **EXECUTED, PASSED (see outcome note)**

Cheapest candidate that targets external transfer directly. Already implemented
(`spint.py:449-455`), currently `dropout_rate=0.0`.

Correct rationale: the dropout is applied to `src = x + id` (`spint.py:445`), so it drops the
**whole unit token**, including calibration-derived identity and T4's contribution. It is
therefore **unit-set robustness regularization** — robustness to unit loss, population
composition change, and unit-count variation — not an intervention on activity-identity
overfitting. Revision 1's claim that it targets the Z4 decay is withdrawn (§8).

Freeze one strength (`dropout_rate = 0.10`, `dynamic_dropout = false`), everything else on the
Arm A recipe (T4, 48 epochs, warmup+cosine, final-four SWA, seed 42, matched scorer). Gates:
external delta vs the best current baseline `≥ +0.03`, `≥ 10/15` positive, within delta
`≥ −0.03`. On a pass, add seeds 43/44. On a fail, **do not** sweep 0.05/0.15/0.20.

Common starting point depends on Priority 0: if P-T4 becomes the best system, branch from the
TFAP Stage-1 pretrained state; otherwise from the Arm A scratch recipe.

### Priority 2 — Temporal Latent Residual decoder — **EXECUTED AS CELL W, NEGATIVE (see outcome note)**

Do **not** simply set `C = 8`. `num_covariates` simultaneously sets the `rep` slot count, the
transformer output slot count, and the behaviour output channel count (`spint.py:421-423, 466-467`),
so `C = 8` yields `[B, 50, 8]` and no longer matches the 2-d velocity labels. Revision 1 called
this a config-level change; that was wrong (§8).

The version worth building keeps the current output as an exact baseline and adds a zero-initialized
residual whose slots are a low-rank **temporal basis** rather than behaviour coordinates:

```text
unit_tokens = fc_in(query_activity + identity)
latent_k    = attention(K learned queries, unit_tokens)        # K = 8
delta[t, c] = sum_k temporal_basis[t, k] * value_head(latent_k)[k, c]
prediction  = base_output + zero_init(delta)
```

This directly attacks property 4 of §1 — two coordinate slots emitting a whole 50-bin trajectory
in one shot — instead of changing the T4 recipe again. It stays permutation-invariant over units,
is bitwise equal to the current decoder at initialization, and needs no GRU/SSM. Only if it works
is an SSM refinement worth considering.

Freeze `K = 8`, one seed-42 T4 cell. Do not sweep `K`. Do not combine with Priority 1. On a
performance pass, add seeds 43/44 plus one Z4 sibling to establish whether the gain is generic or
a better consumption of T4.

Caveat to record: `temporal_basis` mixes the whole window, so the residual is non-causal, matching
the current decoder. If streaming/FALCON-style deployment ever becomes a target, this layer needs
redesign.

### Priority 3 — identity capacity **reallocation** at fixed parameter budget

Not a widening sweep. One arm only:

```text
B3S hidden width: 64 -> 256          (18,290 -> 171,314 params)
FFN width:        2048 -> ~1899      (compensating reduction)
total parameters approximately unchanged
```

The hypothesis is explicit and writable: *is ~153K parameters more useful in the per-unit
calibration-identity encoder than in the global query FFN?* Ranked below Priorities 1–2 because of
§3(b): activity-derived identity is externally negative, so added capacity there carries a
fingerprint risk. Read all three outcomes — within up, external up, external **down** — and do
not adopt as the main system on a within-only gain. Do not run a 128/256/512 grid.

### Priority 4 — minimal block localization, only if TFAP's main effect is clearly positive

Do not run the five block-transfer cells proposed in revision 1. Start with two complementary
cells:

| transferred | question |
|---|---|
| B3S / identity encoder only | is the reusable part identity construction? |
| decoder only (`fc_in` + `transformer` + `rep` + `fc_out`) | is it the population decoder / task manifold? |

Only if both are mixed should this be split further into `fc_in`, `transformer`, and
`rep + fc_out`. Practical blocker: `fc_id_in` is `nn.LazyLinear` with input dimension
`trial_length` (`spint.py:406`), and in `spintshape` the decoder's own `fc_id_in`/`fc_id_out` are
bypassed entirely by B3S and left dead in the checkpoint. Resolve that wiring before designing
transfer cells.

---

## 6. Closed — do not schedule, with receipts

| direction | verdict | receipt |
|---|---|---|
| Hidden-space identity/T4 port | `PILOT_ROUTING_STOP`, primary `+0.0128` vs prereg `+0.03`, 4/6 | A1, `sua_exploration/docs/HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` |
| Full decoupled K/V — identity/T4 as keys, activity as values | **`−0.4447` vs coupled**, 0/6 | `kv_e_t4 = 0.1392` vs `coupled_t4 = 0.5838`, `sua_exploration/results/sua_t4_decoupled_kv_v1/aggregate_seed42.json` |
| SetKV extra carrier K/V tokens | `−0.7387` within / `−0.6389` external | `sua_exploration/results/setkv_delta_forward_v1/terminal_forward_aggregate.json` |
| Plain FiLM / multiplicative T4 fusion | low prior; see the fusion-family reading in §3 | — |
| Carrier-admission curriculum (Z4-first then T4) | **FAIL**, `B − A = −0.177` (0/6), `B − C = −0.174` (0/6) | `results/gate3_within_screen_v1/within_screen_receipt.json` |
| Direct `C = 8/16` query-slot sweep | interface invalid; superseded by Priority 2 | `spint.py:421-423, 466-467` |
| State- or time-conditioned queries | pre-gate overall FAIL | `results/pregate_timevarying_v1/receipt.json` |
| 8-checkpoint mean as an architecture gate | superseded by matched scorer + final-four SWA | §3 |
| 12-epoch spintshape as the performance baseline | superseded by Arm A | §3 |

**Two notes so these closures stay accurate.**

*A1 is narrower than the direction it closes.* A1 tested
`h = fc_in(x + activity_identity) + P(T4)` — it moved **T4** into hidden space while calibration
identity stayed in window space. The constraint named in §1 property 1 (calibration identity must
masquerade as a pseudo-spike-waveform) is therefore still strictly untested. It is closed here on
priority grounds — it is a within-capacity move and §3(a) says external transfer is the objective
— not on the grounds that the experiment was already run. Do not record it as "tested".

*The decoupled-K/V closure is narrower than it looks — check which variant is being proposed.*
The receipt refutes **key replacement**: sourcing keys from identity/T4 while values come from
activity (`kv_e_t4 = 0.1392`, `kv_e_only = 0.1740`, against `coupled_t4 = 0.5838`). It does **not**
refute a zero-initialized T4 *residual* on a key that still contains activity, i.e.
`key = W_key(base) + α·W_t4(T4)`, `value = W_value(base)` — which is the successor now proposed in
`HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md` §6. That design remains open. Its prior is poor: the
adjacent token-level residual is A1 at `+0.0128`, and since `W_K`/`W_V` are already separate
projections of one source and T4 already reaches the key through B3S inside `id_i`, the cell tests
only whether an *additional key-restricted* T4 pathway helps. It stays behind §5 here, and its gates
must be restated against Arm A rather than 12-epoch spintshape. A prior draft of that document
proposed the refuted replacement form; the evidence bracket has been written into its §6.

*The `pv` row is void.* `population_vector.py` decomposes `[a,c]/max(m,eps)` on a **z-scored**
carrier; 65.7% of units get `m_z` clamped to `1e-6`, giving direction vectors of median magnitude
3.24×10⁵. The lane has **no valid transparent baseline**. See
`HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md` §2A. Never cite `pv = 0.0012` as evidence about
population vectors.

---

## 7. Confirmed prediction worth keeping

Arm B's receipts record that during its 15 Z4 epochs, `W_side` and both Adam moments stayed
**elementwise exactly zero**, with the visible side tensor bitwise equal to canonical Z4. That is
the invariant predicted from the affine decomposition
`post_pool[0](cat(h, α·T4)) = W_h h + α·W_s T4 + b`: at `α = 0` the gradient w.r.t. the T4 columns
is identically zero, so the phase is an exact freeze of a 256-weight block.

Two consequences to carry forward. First, any future "gradual admission" schedule is a *gain*
schedule on that block, not an information schedule, and under plain Adam (no weight decay,
`spintshape_module.py:94`) the step size is scale-free, so a ramp is largely self-cancelling —
the phase boundary is the only real treatment. Second, Arm B's final `‖W_side‖ = 0.7606` against
A `1.3992` and C `1.3475` says Z4-first pretraining left the T4 pathway underdeveloped at matched
exposure, which is a concrete mechanism for its `−0.174` loss.

---

## 8. Withdrawn from revision 1

Recorded explicitly so no downstream document inherits them.

1. ~~"The calibration-identity path is worth 0.2012 R² — the largest single measured term."~~
   `0.4574 − 0.2562` is a **system contrast** between spintshape and large, confounding identity
   source, fusion operator, GRU presence, per-unit encoder width, parameter allocation and
   optimization. It suggests the calibration-identity/decoder bundle matters; it attributes nothing
   to the identity encoder's capacity. Revision 1's Tier-1 widening priority rested on this and is
   demoted to Priority 3 as a fixed-budget reallocation.
2. ~~"Teacher init + mature topology ≤ 0.1176."~~ Not an upper bound on anything. Under matched
   48-epoch SWA the within gap to A2 is `0.0587` while the external gap is `0.180`; the two differ
   by 3×, so the old figure was mostly protocol, budget and estimator.
3. ~~"The binding constraint is T4's information content, not the operator consuming it"
   ("saturated").~~ Replaced by the narrow fusion-family reading in §3.
4. ~~Unit dropout "directly targets the Z4 decay".~~ It drops the whole unit token including
   identity; it is unit-set robustness regularization (Priority 1).
5. ~~Slot count is a config-level change, and Tier 1 items can be merged into one cell.~~ The
   interface breaks, and merging three interventions makes a positive result uninterpretable.
   Every direction in §5 runs alone.
6. ~~Tier 2 #4 (hidden-space port) and Tier 3 #7 (decoupled K/V) as open directions.~~ Both closed
   in §6.
7. ~~The five block-transfer cells as a schedulable set.~~ Conditional two-cell version in
   Priority 4.
8. ~~The epoch-window within table as the evidence base.~~ Replaced in §3.

---

## 9. Receipts

- `tfpd_exploration/results/gate3_within_screen_v1/within_screen_receipt.json` — matched-scorer within screen, four SWAs
- `tfpd_exploration/results/gate4_arm_external_v1/final_external_matrix_v2_receipt.json` (sha `8cbc4a88…`) — six-cell external matrix
- `tfpd_exploration/results/admission_arms_v1/arm{A,B,C}_*/{launch,terminal}_receipt.json` — A `6cf7d317…`, B `a6b42fbf…`, C `e08cd735…`
- `tfpd_exploration/results/z4_boundary_pilot_v1` — Z4 boundary pilot, terminal `1372887ac976f219…`
- `tfpd_exploration/results/tfap_stage2_v1/finetune_p{t4,z4}` — in flight
- `sua_exploration/results/sua_t4_decoupled_kv_v1/aggregate_seed42.json` — decoupled K/V STOP
- `sua_exploration/results/setkv_delta_forward_v1/terminal_forward_aggregate.json` — SetKV STOP
- `tfpd_exploration/results/pregate_timevarying_v1/receipt.json` — time-varying pre-gate FAIL
- Ledger: `TFPD_RESULTS_LEDGER_20260816.md` §3A (Gates 1–4)
- Related: `HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md` — §0A status added, §1 PV row voided and
  estimator superseded, §6 successor annotated with its evidence bracket
- Related: `HANDOFF_CARRIER_ADMISSION_CURRICULUM_20260816.md` — executed; curriculum gate failed
