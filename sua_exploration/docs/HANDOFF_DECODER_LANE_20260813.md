# HANDOFF: the decoder lane

**Date:** 2026-08-13
**Method:** three independent agents, one decoder axis each (temporal path, population path, capacity economics),
each forbidden from the other two. I filtered their output and verified the load-bearing facts in code.
**Authorizes:** nothing. A2 is terminal; B1 Stage P is running.
**Relation to other docs:** this is the decoder-specific expansion of
`HANDOFF_FOUR_LANE_BRAINSTORM_SYNTHESIS_20260813.md`. Section 3 there still governs what is dead.

---

## 1. Three findings that change the decoder picture

### 1.1 Only the last bin is scored, so "time-collapsed attention" is nearly a non-issue

`decode_last_timestep_only` is `true` on both lines that matter — `train_variant_dandi688.py:657,823` and
`configs/model/falcon_m2.yaml:4` — and `_slice_last_timestep` slices **both** prediction and target to
`[:, -1:, :]`.

So rows `0..48` of `fc_out` never receive a gradient. The decoder is **a last-bin regressor that happens to emit
50 bins**, not a trajectory model.

**This withdraws a motivation I have been repeating.** I described "attention weights are constant across the 50
output bins" as the surviving structural limitation and used it to motivate time-indexed queries. When only bin
49 is scored, that constancy is close to tautological. Time-indexed queries lose their stated rationale and drop
out of the priority list.

What the last-bin fact *strengthens* is a different mismatch: identity is painted across a 50-bin waveform that
the loss never looks at, except through bin 49.

### 1.2 Capacity and the carrier are not substitutes — my suspicion was wrong

I asked the third agent to test, not confirm, the hypothesis that a bigger decoder would make the carrier worth
less. It does not survive:

- **CI64 is not decoder capacity.** It widens the compact identity head by `+2,208` parameters with the decoder
  fixed. Its carrier increment actually *grew*: `CI32 Full−C0 = +0.032557` against `CI64 Full−C0 = +0.037980`,
  while the 64-wide system got worse overall. That is weakly complementary, not substitutive.
- **B15 is encoder attention**, wrong path and wrong domain, and it is estimator-fragile: a different checkpoint
  rule moves `B15−B15P` from `+0.006354` to `+0.046200`.
- **Within subject C, activity already works** (`Z4 = 0.326`) and the carrier still adds `+0.249`. Strong
  substitutes would show a near-zero increment there.
- **The external Z4 crash is correspondence failure, not underfitting.** Extra parameters trained on matched
  sub-C batches do not construct a sub-M identity map they never practised; they may overfit source activity and
  make external Z4 *worse*, which would make the carrier worth *more*.

I had conflated "capacity" with the only two capacity-adjacent results available, and neither is decoder
capacity. The suspicion is withdrawn.

### 1.3 The paper's online-cost figure is the read-in term, not the decoder

`03_methodology.tex:277` states `Workload_stream ≈ 921 MMAC/s` at `N = 64`. That is the read-in
(`50 → 512 → 512`, about `18.4M` MAC/window at 50 Hz). The **full coupled decoder** is `57,970,688` MAC/window,
a figure that appears in 204 files including post-run cost receipts. At 50 Hz that is about **2,899 MMAC/s**,
roughly `3.1x` the printed number.

This does not touch the T4-versus-Zero4 bit-identical claim, which is relative and survives any capacity applied
to both arms. It does mean the absolute streaming figure is mislabelled, and any depth or width proposal must be
costed against `~58M` MAC/window, not against `921`.

A second accounting point: `102.6x` is an **identity-path** ratio (`58,140` against `5,965,500`). Whole-model
compression is only `1.54x` (`10.95M` against `16.86M`) because the decoder dominates. Decoder scaling spends
precisely the budget the paper never compressed.

---

## 2. Priority

Every arm below is `{design, parent} x {T4, Z4}`, with a TS4 attachment control whenever the new tensor is
unit-aligned. Primary endpoint is **absolute T4 against the matched parent**, with the requirement that the Z4
sibling does **not** reproduce the lift, plus the interaction `(T4−Z4)_new − (T4−Z4)_parent`. A collapsing Z4 is
never a win. Every gate needs a synthetic must-pass and must-fail input before freezing.

| # | Design | Load-bearing? | Cost |
|---|---|---|---|
| **1** | **E-time-structure diagnostic.** On a frozen consumer, broadcast `mean_t(E)` across `W` or permute `E`'s bins, matched T4 and Z4. Does `fc_in` use `E`'s time axis at all? | Diagnostic | CPU, forward-only, zero parameters |
| **2** | **Query-side T4.** `rep` is a global `nn.Parameter`, so the decoder asks the *same question* of every session and all session specificity sits in K/V. Add `Q_c = fc_in(rep_c) + P_q(set of T4)`, zero-init, permutation-invariant over units. | **Yes by construction** — Z4 has no `[a,c]` to build it from | Source training, no new teacher; ~2k parameters for the mean-pool form |
| **3** | **Last-bin-localized identity add.** Add identity only on the scored bin instead of all 50. Zero extra parameters. | Maybe — Z4 sibling decides | Cheapest GPU item; no new parameters |
| **4** | **Per-query V residual.** `V_{i,c} = h_i + P_c(T4_i)`, keys unchanged. Each covariate reads a different value; softly what PVA does hard. | **Yes** — needs per-direction coefficients | Real plumbing: PyTorch MHA shares V across queries |
| Hold | Spectral live/identity split; T4-gated mix of the existing `C` queries; extra mixed T4-built queries | Mixed | See section 4 |

**Item 1 first, and it replaces a diagnostic that was already killed.** My earlier identity-rank diagnostic was
withdrawn because `E = post_pool([mean_h, T4])` mixes activity and carrier, so its spectrum cannot isolate the
carrier. Scrambling `E`'s time axis avoids that entirely: it does not need to separate the two sources, only to
ask whether destroying the time structure has any consequence. If both arms move less than `0.01`, the waveform
contract is unused, which kills the temporal designs **and** independently supports A1's premise that identity
need not be a waveform.

**Item 2 is the one genuinely new structural observation from this round.** Identity currently enters only by
painting the `N` unit tokens that become shared keys and values; the residual skip starts from a source-trained,
session-blind `Q`. Putting T4 into `Q` is the only way the carrier can change *what is asked* rather than *who
answers*, and it is untouched by all nine prior arms.

---

## 3. Dropped from the decoder lane

- **Time-indexed queries.** Motivation collapsed with section 1.1.
- **Decoder scaling as an accuracy hunt.** It does not address A2's failure mode, it spends the one budget the
  deployment story protects, and a "bigger is better" result would be uninterpretable as evidence about the
  carrier. If it is ever run, it is a defensive `{L=1, L=2} x {T4, Z4}` cell whose purpose is to stop a reviewer
  misreading, with MAC/window co-reported against the `~58M` figure.
- **`G(E)`, a `Linear(W,H)` read of the painted waveform.** Strictly downstream of item 1; pointless if the
  waveform is unused, and a wider worse A1 if it is.
- **Untying the query from neural `fc_in`.** Queries contain no T4, so the prior is that it raises all arms.
- **Richer `fc_out` or a full-window reconstruction loss.** The other bins were never trained on the teacher
  either; using them as targets fits noise, and a behaviour-window loss is extra dense supervision.

Still dead from earlier rounds and not revived here: unmixed latent queries, slot routing, encoder cross-neuron
attention, key or logit residuals, K/V splits that delete `fc_in`, rank-1 `fc_in` on its stated motivation,
signed-readout arguments, PVA as a hard readout constraint, identity-interface widening.

---

## 4. Held, with reasons

**Spectral live/identity split** (`h = fc_in(x_lp + E) + U(x − x_lp)`, keeping identity in the slow band so
transients cannot impersonate it). Conceptually strong and designed to be load-bearing, but it overlaps with
carrier forcing: both aim to stop the activity path from solving the task alone. Run one of them first and see
whether the mechanism moves before paying for the second.

**T4-gated mix of the existing `C` queries.** On SUA and M2, `C = 2`, so the mix is a `2x2` coordination
residual and may be structurally too small to clear `+0.03`. More promising on H1 with `C = 7`, which is a
different substrate and a different contract.

**Extra mixed T4-built queries over intact `N`.** This is the design the latent-query withdrawal actually named
and never tested, and it avoids slot routing's error because `N` is preserved. But putting extra tokens and a
mixing block inside the teacher decoder requires a new teacher. Do not run it before items 1 to 4 report.

---

## 5. What this lane does not claim

None of these are predicted wins. The honest expectation, given that A2 already delivers the large effect
without any decoder change, is a few points on the native remainder. The value of items 1 and 2 is that they are
cheap and that a negative result from either is informative: item 1 would retire an entire class of temporal
designs, and item 2 would establish that session-blind queries are not the constraint.
