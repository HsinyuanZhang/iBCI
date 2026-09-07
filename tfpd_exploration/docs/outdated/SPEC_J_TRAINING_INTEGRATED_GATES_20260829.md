# Spec: J0/J1/J2/J3 Training-Integrated Gates (J-ladder)

Date: 2026-08-29
Status: SPEC ONLY — WRITE, NO EXECUTION. No training of any kind is
authorized by this document.
Authority: `docs/DESIGN_LEARNABLE_CAUSAL_OUTPUT_FILTER_20260829.md` §7, §10.4,
§11 (verbatim gates in §5 below); operator addition: the J0′ sampler-confound
control (§3).
Prerequisite route: `learnable_output_filter_v1` (P0–P4 frozen-output route).
Entry condition (§7.1): the training-integrated route starts ONLY after P1
proves that filtering the stronger activity-only stream has a real matched
effect (B1−B0 on the CDM stream).

---

## 1. Scope and hard boundaries

* The J-ladder RETRAINS the decoder. It therefore never inherits the
  frozen-decoder parity claims of the P0–P4 route: every J receipt must state
  that the decoder parameters change during the run.
* Permitted trainable filter parameters are ONLY (§7.2):
  - one scalar EMA logit;
  - K4 nonnegative FIR logits normalized by softmax; or
  - the F4 scalar-gain logistic parameters already defined in §4 of the design
    (8 whitelisted rotation-invariant features + bias, one scalar gain shared
    by both velocity dimensions).
* No output-coordinate-specific matrix, recurrent hidden network, TCN,
  target conditioner, pseudo-label generation, learned carrier gate, or
  T4-update-rule change is permitted (§2.3, §7.2).
* Output filtering and carrier-policy learning remain separate factorial axes
  in every J receipt and table.
* GPU training cells are J0, J0′, J2, J3 (J1 is inference-only on the exact J0
  checkpoint). Each training cell requires its own work order with receipt-
  bound sequence length, burn-in length, gradient accumulation and peak memory
  (§7.5); none of these may be changed opportunistically mid-run.

## 2. The arm ladder (§7.2, verbatim table)

| Arm | Decoder | Filter during training | Trainable filter | Purpose |
|---|---|---|---|---|
| J0 | accepted baseline | none | no | raw reference |
| J1 | exact J0 checkpoint | yes, post hoc | source-fit only | frozen-output reference |
| J2 | retrained from the accepted initialization | yes | no; freeze the J1 filter | isolate training integration |
| J3 | retrained from the same initialization | yes | yes, constrained | conditional joint-learning test |

J2 is the first GPU cell. J3 is allowed only if J2 is positive AND the
adaptive oracle or source-grouped F3/F4 result shows remaining
filter-parameter headroom.

The initial filter for J2/J3 must be a source-selected fixed filter or an
exact bypass, frozen before the run (for this route: the P2-selected fixed
filter of `results/learnable_output_filter_v1/p2_fixed_filter.json`, payload
SHA bound into the attempt).

## 3. J0′ — the sampler-confound control (operator addition, REQUIRED)

The existing Falcon training path uses `SessionBatchSampler(..., shuffle=True)`;
J2 must instead use the route-owned chronological sequence sampler (§4). A J2
that improves on an old J0 therefore conflates two changes:

1. the integrated causal output filter, and
2. the change of sampling order / sequence exposure itself.

J0′ removes the confound:

| Arm | Decoder | Sampler | Filter during training | Trainable filter | Purpose |
|---|---|---|---|---|---|
| J0′ | retrained from the SAME accepted initialization as J2 | the SAME chronological sequence sampler as J2 (identical sequences, order, resets, burn-in marks, step counts) | NONE — exact bypass (F0, bitwise) | no | isolates the sampling-order effect |

Requirements:

* J0′ uses the identical initialization bytes, identical seed policy, identical
  optimizer/schedule/batch/accumulation configuration and identical
  chronological sequence membership/order/reset/burn-in digests as J2; the ONLY
  difference is the filter: J0′ carries an exact F0 bypass (the filtered and
  raw training predictions must be bitwise equal inside J0′, asserted every
  step).
* The attribution identities are then:

  ```text
  J0′ − J0        = the sampling-order / sequence-exposure effect itself
  J2 − J0′        = the integrated-filter effect with the sampler held fixed
  ```

  Both contrasts are mandatory readings of every J2 receipt. A J2 claim that
  cannot show `J2 − J0′` is descriptive only.
* COST DISCLOSURE: J0′ doubles the training cost of the J2 cell (one extra full
  retrain). This is the price of attribution and is not optional. If only one
  extra run can be afforded, J0′ outranks J3.
* J1 must be recomputed on the J0′ checkpoint as well (`J1′`), because the
  primary estimand `J2_filtered − J1_filtered` (§5) must compare against the
  matched same-sampler chain, not only the legacy J0 checkpoint.

## 4. Chronological training data (§7.4, binding requirements)

The training-integrated route must use a route-owned chronological sequence
sampler with ALL of the following properties:

* each sequence contains last-bin predictions from ordered query windows in
  one session and one uninterrupted segment;
* rows inside a sequence are never shuffled;
* complete sequences or blocks may be shuffled only after their internal order
  and predecessor context are sealed;
* a reset marker is supplied at the exact trial/session/recording/gap boundary
  (TRIAL_RESET is primary while trial IDs are available; STREAM_GAP_RESET only
  after the formal adapter audit);
* no filter state crosses sessions or an ambiguous discontinuity;
* every scored row appears exactly once per logical epoch;
* state burn-in rows are explicitly identified and excluded from the loss;
* source-session exposure and optimizer-step counts remain matched to J0/J0′.

For the internal M2/SUA trial protocol the preferred training unit is a full
ordered trial trajectory with an exact trial reset. For a stream-gap contract,
use chronological blocks with exact predecessor context; never introduce an
artificial block reset and call it deployment-equivalent.

It is NOT valid to replace this with a causal filter over the `W` internal
bins of one model window: the governing stream consists of successive
last-bin predictions from successive windows, and an earlier internal output
of the current window need not equal the prior window's last-bin prediction
(§7.4). An in-window filter would be a different system.

## 5. Loss, gradient contract and gates (§7.3, §7.5, §7.6 — verbatim)

### 5.1 Loss (§7.3)

```text
L_raw      = normalized_session_MSE(r_t, target_t)
L_filtered = normalized_session_MSE(y_hat_t, target_t)
L_joint    = 0.5 * L_raw + 0.5 * L_filtered
```

Both terms use the same valid rows, source-session normalization, and equal
session weighting. The equal `0.5/0.5` weights are primary and are not
selected on external labels. The raw auxiliary loss is mandatory; every
checkpoint retains two source validation curves and every score reports both
`r_t` and `y_hat_t`. A successor may test another source-frozen loss weight
only after the primary equal-weight cell is terminal and only with a separate
work order.

### 5.2 Gradient and state contract (§7.5)

* gradients may flow through `F_theta` and the decoder within one sealed
  chronological training sequence;
* the incoming filter state at a block boundary is reconstructed from exact
  predecessor/burn-in rows and detached before the scored block;
* no gradient crosses a session, trial reset, recording gap, or source fold;
* target-session backward/optimizer/update counts remain zero;
* evaluation uses the identical filter equations, initialization, and reset
  contract as training;
* training and evaluation state-transition digests must be reproducible;
* filter parameters are source-trained and immutable at deployment.

### 5.6-style anti-compensation gate (§7.3)

J2/J3 fail the anti-compensation gate if the source-grouped out-of-fold raw R2
falls by more than `0.01` from J0, even when filtered R2 improves.

### 5.3 Required contrasts and gates (§7.6 — VERBATIM)

The primary training-integration estimand is:

```text
J2_filtered - J1_filtered
```

It asks whether exposing the decoder to a fixed deployed filter during source
training improves on applying that exact filter after training. The contrasts
`J1-J0` and `J2_raw-J0` remain mandatory diagnostics.

Advance J2 beyond source validation only if:

- source grouped-OOF `J2_filtered-J1_filtered >= +0.005`;
- source grouped-OOF `J2_raw-J0 >= -0.01`;
- onset lag, peak lag, and overshoot do not cross their source-frozen safety
  limits;
- no reset, chronology, exposure, or state-digest gate fails.

Claim an external training-integration benefit only if:

- `J2_filtered-J1_filtered >= +0.01` on the predeclared primary budget;
- at least 10/15 external sessions are positive;
- within regression is no worse than `-0.005`;
- J2 uses the same source selection rule and seed policy as J0.

J3 must additionally beat J2 by at least `+0.005` source-grouped OOF before it
may be externally scored. If J2 is positive but J3 is not, the result supports
training with known smoothing dynamics, not learning a more complex filter.

### 5.4 Two-chain requirement (§7.6/§7.7, §10.4)

Because J2/J3 retrain the decoder, they require matched seeds and a matched J0
retraining control. Comparing a new J2 run only with an old sealed checkpoint
is not sufficient to attribute the gain to the integrated filter. With the
operator's J0′ control the full matched chain is:

```text
J0  (accepted recipe, legacy shuffled sampler)
J0′ (same init as J2, chronological sampler, exact F0 bypass)
J1  (post-hoc frozen-output filter on the exact J0 checkpoint)
J1′ (post-hoc frozen-output filter on the exact J0′ checkpoint)
J2  (integrated frozen filter, chronological sampler, same init as J0′)
J3  (integrated constrained-learnable filter, same init)
```

A J2/J3 result that lacks a matched same-seed J0/J0′/J1 chain is descriptive
and cannot support a causal design claim (§10.4).

## 6. Receipt requirements (§11, training-integrated cells)

Every training cell additionally records:

* J0 initialization/checkpoint and same-seed pairing (and, for J2/J3, the
  identical-initialization proof against J0′);
* raw and filtered loss curves and checkpoint-selection values;
* raw and filtered prediction digests;
* filter parameter/state digests at every checkpoint;
* sequence/block membership, order, reset, predecessor, and burn-in digests
  (identical between J0′ and J2, asserted);
* scored versus burn-in row counts;
* source-session exposures and optimizer-step counts (matched to J0/J0′);
* raw/filtered gradient norms at the decoder output;
* raw R2, filtered R2, onset lag, peak lag, and overshoot;
* filter bypass evaluation from the same trained decoder (J0′'s bypass is
  exact by construction; J2/J3 additionally evaluate with the filter bypassed);
* proof that target backward/update counts are zero.

Lane conventions as in §11: immutable 0444 body plus canonical sidecar, fresh
root, explicit closure, attempt-before-data/model, atomic terminal or failure
publication.

## 7. Execution order and stop conditions

First pilot (§9 P6): J0′/J1′/J2 on ONE primary source fold and ONE seed, only
after P1 and the source gates pass; treated as a mechanism pilot, not a
performance claim. J3 is not part of the first pilot; it is enabled only by
positive J2 plus independent filter-parameter headroom (P3/P4 readouts). A
full seed/fold matrix is authorized only after the pilot passes chronology,
raw anti-compensation, lag, and matched filtered-score gates.

Stop conditions that apply verbatim from §13 of the design (training-integrated
subset):

12. J2 improves filtered output only by materially degrading its raw output;
13. J2 fails to beat applying the same filter post hoc to matched J0/J1;
14. J3 fails to beat fixed-filter J2 after source-grouped validation;

and, from §13.11, if chronological source sequences cannot be reconstructed
without changing the governing sample/exposure contract, the route stops
before any J2 launch. Do not respond to a STOP by adding a GRU, longer K, more
layers, target-time fine-tuning, or a larger feature search.

## 8. Interpretation boundary (§7.7)

A positive J2 result is a system-training result, not proof that output jitter
alone was the original bottleneck. A positive J1 with null J2 says the
post-hoc filter is sufficient. A positive J2 with null J1 would be unexpected
and must be treated as a new regularization effect requiring replication, not
as confirmation of the continuity probe. With J0′ in the chain, a positive
`J2 − J0′` with null `J0′ − J0` attributes the gain to the integrated filter
rather than to the sampler; a material `J0′ − J0` must be reported as a
sampler effect and never silently absorbed.
