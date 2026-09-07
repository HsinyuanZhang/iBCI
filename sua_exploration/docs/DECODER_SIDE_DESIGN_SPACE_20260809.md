# DECODER-SIDE DESIGN SPACE FOR THE CARRIER: four options, one recommendation

**Date:** 2026-08-09
**Status:** root-reviewed design and execution note. Option 1 has been implemented as
the full `4->50` per-bin RT L-D operator and passed its model/data/RNG CPU contracts.
The GPU cell is still unlaunched: the current handoff draft is being superseded because
the three composed Hydra arms did not yet include the mandatory clean nested-LOSO
selection-receipt callback. No RT L-D accuracy result exists yet. Options 2--4 remain
unauthorized; H1/M1 are not reopened by this note.
**Context:** the CPU screen line is closed. Five levers are shut. L-D is the only untouched
structural lever. This note records the decoder/consumer design space that matches the
encoder change, and the evidence that constrains it.

---

## 1. Why the encoder change alone is not enough

SPINT's identity path answers "what does this channel's calibration activity look like."
Our carrier asks "what relation between that activity and the calibration target can be
estimated." The carrier is **not** a strict information superset of activity: it is a
low-dimensional, estimator-dependent projection and may discard activity information.
The pair `(activity, correctly paired carrier)` can nevertheless contain conditional
information that a label-free statistic cannot identify, for example opposite tuning
signs at similar marginal rates. Whether that information survives estimation and helps
held-out decoding is empirical, not information-theoretically guaranteed.

An analytic, deployment-time estimator with controlled label scope and matched nulls can
already constitute a method. A matching decoder-side operator would strengthen the
mechanistic story by testing whether the estimated coefficient is useful specifically as
an activity modulator; it is not required to retroactively legitimize the encoder result.

---

## 2. The core observation

The carrier contains coefficients of an **encoding** model such as
`r_i = b_i + w_i^T v + eps`. This motivates an activity--coefficient product, but the
naive product is not the exact inverse of that model:

```
heuristic:  v_hat(t) = sum_i r_i(t) w_i
linear-Gaussian inverse:
v_hat(t) = (W^T Sigma^-1 W + lambda I)^-1 W^T Sigma^-1 (r(t) - b)
```

The current architecture concatenates activity and carrier inside a nonlinear identity
MLP, then adds the resulting identity token to live activity before another nonlinear
read-in and cross-attention. It can therefore express **implicit** interactions. What it
lacks is an explicit, low-cost multiplicative inductive bias on the live activity path.
The decoder-side question is whether making that bias explicit improves generalization,
not whether multiplication is currently mathematically impossible.

That is the gap. Every option below closes it to a different depth.

---

## 3. The four options, smallest change first

### Option 1 — zero-init multiplicative gain on live activity (recommended pilot)

```
x'_i = x_i * (1 + g(carrier_i))
src_i = x'_i + E_i
```

Use `bias=False` and zero-initialize `g.weight`; construct the common backbone before
attaching `g` so the matched arms consume the same RNG stream. At step zero the gain arm
must be bitwise identical to the additive reference. The cost is task dependent:
`4x50=200` weights on SUA/RT and `4x700=2800` on H1 (plus bias only if explicitly used).
The prior confidence-FiLM result (`+0.003399` versus T4) conditioned on reliability, not
carrier content, so it does not exactly test this hypothesis; it does lower the prior.

### Option 2 — FiLM one layer deeper

Same idea applied to the output of `fc_in`, not to raw activity. It is materially larger:
`4x512=2048` weights on SUA/RT and `4x1024=4096` on H1, before biases. It is not justified
until option 1 passes, because it changes a deeper representation and is harder to
attribute.

### Option 3 — carrier-biased attention logits

SPINT's decoder queries are already behavior dimensions (learned `rep [C, W]`). Letting
the attention score depend on `f(query_c, carrier_i)` would make functional identity
control which units are read. However, the frozen rank-8 SUA T4 attention-logit residual
already gave `0.587131` versus `0.590273` for continuation (`-0.003142`). That does not
rule out every jointly trained variant, but it makes this a low-priority branch rather
than the next experiment.

### Option 4 — carrier-as-basis readout (strongest claim, highest risk)

The decoder stops producing behavior directly. It outputs a time-varying per-channel
gain `g_i(t)`, and the behavior comes from a carrier-constrained skeleton:

```
v_hat(t) = sum_i g_i(t) r_i(t) w_i
```

This hard constraint locks channel `i`'s contribution to the direction `w_i`. With
constant `g_i`, unit preferred-direction vectors, baseline subtraction, and the usual
normalization, it reduces to a weighted population-vector decoder; more generally it is
a carrier-basis-constrained linear readout, not exactly classical PVA. The full decoder
can no longer rotate a channel contribution away from `w_i`, so estimator error and
model mismatch become structural bottlenecks. The exact ridge/SVD inverse above is a
better CPU diagnostic. PV50 is also an important warning: direct use of T4 preferred
directions plus affine calibration was worse than T4 (`T4-PV50 = +0.241454` on SUA).
Therefore this is a headroom/control study, not the main GPU route.

---

## 4. The evidence constraints

| Constraint | Value | Consequence |
|---|---|---|
| H1 fold-0 carrier ablation | `H-C - H-C0 = +0.038895/+0.039647` (seeds 42/43) | A realized ablation, not a theoretical ceiling; wait for five-date H-C0 before a new H1 branch. |
| SUA carrier controls | `T4-Z4 = +0.248968`; `AC4-RS4 = +0.294163` | Large carrier dependence, but not proof that a new operator adds complementary information. |
| RT carrier controls | `Full-MB4 = +0.257510`, 15/15 | Best location for a small operator pilot after the current queues close. |
| SUA formal scope | consumed by G1 | Scheduling only. |
| RT matrix | completed | Do not silently redefine its endpoint; any operator pilot is a new pre-registered experiment. |
| Overlap gate | falsifies, does not confirm | A passing residual is not evidence of a decoder gain. Only a GPU arm answers that. |

The handoff already says it: if a GPU arm runs, it is L-D on SUA or RT, not H1.

---

## 5. Controls that are mandatory for any of these

- Zero-init or equivalent proof of bitwise identity with the current arm at step zero.
- A gain-XLS control: the ordinary aligned Full carrier remains in `E`, while the strong
  XLSv2 carrier is sent only to `g`. This isolates whether the new multiplicative path
  needs correct pairing instead of merely acting as extra regularization.
- Gate against a freshly trained matched reference, never against a sealed absolute
  score. A fresh construction lost `0.06239` on identical windows.
- Build the common backbone before adding the zero-initialized gain module and verify an
  exact-null CPU test, so parameter initialization/RNG drift cannot masquerade as gain.

---

## 6. Recommendation

If another GPU branch is opened, start with **RT M24 option 1 only**. Use three freshly
matched arms: additive Full (`A0`), aligned Full in both `E` and gain (`G-Full`), and
aligned Full in `E` but strong XLSv2 only in gain (`G-XLS`). For fold 0, seed 42, freeze
both gates before launch:

1. `G-Full - A0 >= +0.03 R2`;
2. `G-Full - G-XLS >= +0.03 R2`.

If either fails, stop: no option 2, seed sweep, H1 transfer, or post-hoc operator variant.
If both pass, expand to folds 1--2 and require both aggregate deltas to remain at least
`+0.03` with 3/3 positive before considering all 15 sessions.

This pilot asks for **incremental information beyond the existing nonlinear consumer**.
The existing carrier encoder is already a method; a gain operator should be promoted to
the paper only if both accuracy and pairing-specificity gates pass.

### 6.1 Execution update (2026-08-10)

The original scalar `4->1` implementation was rejected before GPU use and is sealed as
`SUPERSEDED_DO_NOT_LAUNCH`. The corrected v2 uses `Linear(4,50,bias=False)`, zero
initialization, 200 parameters, a `[B,N,50]` calibration-derived state, and applies the
gain before additive Full identity and `decoder.fc_in`. Root independently reran the
L-D, clean-RT, XLS, and receipt suites (`37 + 4` passing tests) and verified all 14
receipt-bound artifact hashes. The v2 receipt/plan SHA-256 prefixes are
`0e3bdb78...b09dfd` and `1e5bbb2e...b26888`; both are queue records, not score evidence.

The RNG issue was subsequently corrected with an already-zero `[50,4]` Parameter and
`F.linear`; the matched A0/G construction now leaves the CPU RNG state bitwise equal.
The first device-handoff draft was rejected because it observed a transient H1 child
rather than the complete static partition runner. Its replacement correctly bound the
outer runner, but root composition audit then showed that the RT L-D Hydra arms lacked
the fit-end `RtNestedSelectionReceipt` callback required by the one-shot outer evaluator.
That replacement is therefore also `DO_NOT_ARM`. A new append-only handoff must bind the
corrected composed configs, exact arm identity (`run_id`, `rt_ld_arm`, gain source),
static-runner release, and absolute run paths before any watcher or GPU cell starts.

---

## 7. What this note does not do

- It does not authorize a GPU run. Route B (L-D on SUA or RT) versus Route C (write the
  existing result) is the owner's decision.
- It does not reopen any closed lever.
- It does not claim that H1 has a known `+0.0389` ceiling. That number is a fold-0
  ablation; the five-date H-C0 attribution control must close before any H1 conclusion.
