# Frozen Protocol: Source-Pooled Ridge (SPR)

**Date predeclared:** 2026-08-12, before any arm was executed
**Status:** CPU-only. No GPU. Another agent session is using both GPUs; do not disturb it.
**Scope:** subject-M (DANDI 000688) SUA and pseudo-MUA views, and optionally FALCON H1.

---

## 1. Why this experiment exists

The paper reports that on subject-M the carrier **loses** to a densely supervised ridge at the
longest calibration budget: `0.3568` against `0.4179` at M50 SUA, and `0.3061` against `0.4102` at
M50 pseudo-MUA.

Two sealed defenses already exist:

1. **Direction-only ridge fails outright.** Given the same trial-direction labels the carrier uses,
   a direct ridge is **negative at every budget**: `-0.4599 / -0.2086 / -0.1220` at M15/M30/M50 SUA.
   The audited finding is that T4 beats it at every budget with 14/15 or 15/15 session signs.
2. **The loss is budget-dependent.** The dense ridge only overtakes T4 at the longest budget. At
   M15 SUA it scores `0.0726` against T4's `0.3381`.

**The unresolved confound in defense 1 is architecture, not labels.** T4 consumes one
trial-direction scalar *plus a source-pretrained decoder*; the direction-only ridge is fit from
scratch on the target session alone. So T4's advantage may be entirely source pretraining.

This experiment removes that confound by giving the ridge a source prior.

**A prior claim was already struck for exactly this reason.** The review removed an unsupported
matched-label T4-Ridge `+0.25 R2` claim and banned matched-information language. Nothing in this
protocol reinstates it. Only a passing SPR result could earn a matched claim, and only in the
narrow form defined in Section 5.

---

## 2. The structural obstacle, which is itself a result

A ridge's parameters are indexed by recorded units. Transferring them across sessions therefore
requires a channel correspondence:

| View | Unit basis | Correspondence across sessions | SPR definable? |
|---|---|---|---|
| subject-M SUA | sorted single units | unit sets change across sessions | **No, not without an alignment step** |
| subject-M pseudo-MUA | electrode-level sums | fixed electrode array | **Yes** |
| FALCON H1 | 176 threshold-crossing channels | fixed channel map | **Yes** |

**Part A of this protocol is a constructibility audit** that measures this rather than assuming it.
If a source-pooled ridge is undefined on SUA without alignment, that is a reportable finding: the
natural route to giving a ridge source information demands a fixed channel map, which is precisely
what a permutation-invariant carrier does not require. Do not manufacture a correspondence on SUA to
force a number.

---

## 3. Estimator — frozen

Per source session `s`, fit a ridge `beta_s` on that session's calibration block using the
established `50*N` representation and the sealed normalisation and intercept conventions. Pool:

```
beta_prior = mean_s(beta_s)          over channel-correspondent source sessions only
```

On the target session, solve the prior-anchored ridge in closed form:

```
beta = (X'X + (lambda + gamma) I)^-1 (X'y + gamma * beta_prior)
```

`gamma = 0` recovers the existing scratch ridge exactly; that identity must be asserted in a test.
The intercept stays unpenalised and is never shrunk toward the prior.

**Selection rule.** `lambda` and `gamma` are selected by nested leave-one-session-out **among source
sessions only**. No target query window, and no target calibration label beyond the declared budget,
may influence either. State in the receipt exactly what each selection consumed.

---

## 4. Arms — frozen

Budgets `M in {15, 30, 50}`, both views, all 15 sessions, matched to the sealed A2a-v2 protocol.

| Arm | Target | Prior | Role |
|---|---|---|---|
| `ridge_dense_scratch` | dense per-bin velocity | none | integrity gate vs sealed |
| `ridge_direction_scratch` | trial direction `[cos, sin]` | none | integrity gate vs sealed |
| `ridge_dense_spr` | dense per-bin velocity | source-pooled | new |
| **`ridge_direction_spr`** | trial direction `[cos, sin]` | source-pooled | **primary candidate** |
| T4 reference | - | - | sealed values, not re-run |

### Integrity gates, required before any new arm is interpreted
- `ridge_dense_scratch` must reproduce `0.0726 / 0.3222 / 0.4179` (SUA) and
  `0.1291 / 0.3414 / 0.4102` (pseudo-MUA).
- `ridge_direction_scratch` must reproduce `-0.4599 / -0.2086 / -0.1220` (SUA) and
  `-0.4244 / -0.1635 / -0.0879` (pseudo-MUA).
- Tolerance `5e-5`, matching the sealed reproduction standard for this family (the sealed receipt
  itself reproduces its references to `2.24e-6`, and is described as high-precision numerical
  reproduction rather than bit-exact).

Sealed T4 references: SUA `0.3381 / 0.3582 / 0.3568`, pseudo-MUA `0.2882 / 0.3053 / 0.3061`.

---

## 5. What each outcome licenses — fixed in advance

Primary contrast: `T4 - ridge_direction_spr`, per view per budget, reported as mean, median, session
sign count out of 15, and a paired bootstrap interval.

- **T4 still exceeds `ridge_direction_spr` at every budget with at least 11/15 session signs.**
  Then the source-pretraining confound is addressed and the paper may state, in this narrow form:
  *given the same sparse trial-direction labels and a source prior, a direct ridge does not reach
  the carrier.* Architecture still differs, so this remains a system-level statement; it is **not**
  a claim that the carrier's estimator is superior.
- **`ridge_direction_spr` reaches or exceeds T4 anywhere.** Then defense 1 weakens and must be
  reported. The budget-dependence defense and the supervision-count claim are unaffected.
- **SPR is undefined on a view** (no channel correspondence). Report
  `SPR_UNDEFINED_NO_CHANNEL_CORRESPONDENCE` with the supporting counts and treat it as a finding
  per Section 2.

Do not relax the 11/15 threshold or add arms afterwards. A result that weakens the paper is the
successful outcome of a fairness check if that is what the data says.

---

## 6. Interpretation limits

- SPR gives the ridge *source information*; it does not give it a permutation-invariant consumer or
  a nonlinear decoder. The comparison remains system-level.
- Any SPR arm still consumes whatever target labels its target arm consumes. The supervision-count
  accounting (750 direction scalars against 299,450 dense coordinates on subject-M) is unchanged by
  this experiment and must be reported alongside accuracy.
- Nothing here licenses reinstating the struck matched-label `+0.25 R2` claim.
- The A2b-v2 within-ridge density result stands unchanged and is not reopened.
