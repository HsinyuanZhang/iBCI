# Carrier-biased attention: root review and corrected RT-only design

**Original proposal:** 2026-08-09  
**Root review:** 2026-08-10  
**Status:** **NO-GO as originally written.** The mechanism is implementable, but it
substantially overlaps a previously negative SUA attention-logit experiment and several
mechanistic, implementation, cost, and interpretation claims in the original note were
incorrect. A corrected experiment is retained only as a conditional, low-priority RT
pilot after the simpler L-D input-gain experiment. This document authorizes no GPU run.

This review applies two screens before proposing more training:

1. **Failure-history screen:** ask whether the operator has already been tested in a more
   expressive form.
2. **Simplicity screen:** prefer the smallest experiment that can distinguish carrier
   content from extra parameters, fresh co-training, and a bad attachment.

---

## 1. What the proposed operator actually is

SPINT uses learned output queries and per-unit/channel neural tokens. For output query
`c` and unit/channel `i`, ordinary cross-attention contains the logit

```text
s[c,i] = Q[c] K[i]^T / sqrt(d_head)
```

The proposal adds a session-calibrated bias:

```text
s'[c,i] = s[c,i] + B[c,i]
B[c,i]  = Phi[c] carrier[i]
```

`carrier[i]` is recomputed from the calibration block and is fixed during streaming.
The direct matrix `Phi` should be zero-initialized and learned only during source
training. Deployment still has no target-session backward pass.

Do **not** parameterize the same bias as `lambda * phi * carrier`. `lambda` and `phi`
are scale- and sign-non-identifiable: rescaling one and inversely rescaling the other
leaves the operator unchanged. Absorb the temperature into one zero-initialized `Phi`
and report the effective bias scale, not separate parameter norms.

The carrier definitions also need to remain task-specific and exact:

- SUA T4 is the four-component fitted descriptor used by the selected T4 branch, not an
  `[a,c,0,0]` placeholder.
- RT Full is `[W_x, W_y, ||W||, b]`; RT MB4 is `[0,0,||W||,b]`.
- H1 CarrierID uses a different population construction and must not be described as
  equivalent to either of the above.

The existing identity path remains active. This is an additional consumption operator,
not a replacement for carrier-conditioned identity tokens or for live neural activity.

---

## 2. What is genuinely attractive about it

The useful hypothesis is narrow:

> A functional carrier may be easier for a source-trained decoder to consume as an
> explicit output-query-to-unit prior than only through the implicit `identity ->
> fc_in -> K/V` path.

That would preserve several desirable deployment properties:

- all channels remain present; there is no token pruning;
- the carrier is still fitted analytically at target time;
- target adaptation remains forward-only, with no target-session gradient state;
- the cached carrier bias has size proportional to `C*N`, independent of model width;
- the operator remains equivariant to a joint permutation of neural tokens, identity
  rows, carrier rows, and the cached bias columns.

The claim must be **carrier-conditioned** attention, not session-invariant attention.
Both the carrier and the neural keys remain session-dependent. Cross-date robustness is
an empirical hypothesis, not an invariance theorem.

---

## 3. The closest prior experiment is already negative

This is not a new mechanism family. The existing
`CoupledT4LogitResidualStreamingSpint` implementation uses a more expressive rank-8
factorized nonlinear T4-to-attention-logit residual, shared across heads. It preserved
the identity path and included aligned, row-shuffled, and matched-additive controls.

The authoritative SUA aggregate is:

| Arm or contrast | R² / delta |
|---|---:|
| selected T4 continuation | 0.590273 |
| aligned rank-8 logit residual | 0.587131 |
| row-shuffled logit residual | 0.580729 |
| matched additive control | 0.583375 |
| aligned - selected T4 | -0.003142, 3/6 sessions positive |
| aligned - row-shuffled | +0.006402 |
| aligned - matched additive | +0.003756 |

It failed the pre-declared `+0.03` and all-session-positive gates. The new direct linear
bias is approximately a restricted member of that family, so it has a low prior on SUA
and does not reopen the consumed SUA formal scope.

The only material difference worth testing is **fresh joint source training on RT**:
the full decoder can learn Q/K/V jointly with the operator from initialization, whereas
the SUA residual experiment trained a small residual around an already selected T4
model. That distinction is enough to retain one conditional RT pilot, but not enough to
call the operator novel by itself.

---

## 4. The signed-linear bias has a mechanistic flaw

For RT, `W_x` and `W_y` are signed tuning coefficients. A static attention score based
linearly on `W_x` tends to prefer one tuning polarity. That is not generally the correct
notion of relevance: both a positively and a negatively x-tuned unit can carry strong
information about x velocity. The sign can be used in the value/readout path, while a
static *selection* prior often needs to represent strength rather than polarity.

Consequently, the original direct signed form is not yet the strongest principled v1.
If this branch is ever opened, a source-only CPU audit must choose and freeze one of:

```text
signed:     rho(W) = [W_x, W_y, ||W||, b]
relevance:  rho(W) = [W_x^2, W_y^2, W_x W_y, ||W||, b]
```

The relevance form is the more defensible attention prior, after train-source-only
normalization. This is not a license to run both after seeing target results. The chosen
map, normalization, and dimension must be fixed from source data before the GPU cell.

This point also limits the interpretation of the previous negative result: its `tanh`
factorization is expressive, but its simplest behavior is predominantly odd in the
signed tuning coefficients. A fresh even-relevance operator is a corrected hypothesis,
not evidence that the old result was positive.

---

## 5. Implementation is not one line

`CrossAttentionLayer.forward` already accepts `attn_mask`, but the main SPINT and
streaming decode entry points do not currently carry a session-specific bias through the
full call chain. A correct implementation must cover all of the following.

### 5.1 Mask shape and session isolation

PyTorch MHA accepts:

- `[C,N]` for a mask shared across the whole batch and all heads;
- `[B*num_heads,C,N]` for example-specific masks;
- **not** `[B,C,N]` directly.

A session-homogeneous batch may safely use `[C,N]`. Any mixed-session batch must expand
the masks correctly and fail closed if a carrier can cross session boundaries.

### 5.2 Exact-null behavior

Passing an all-zero `attn_mask` can change the PyTorch arithmetic path from `bmm` to
`baddbmm`; zero bias is therefore not automatically bitwise identical on every backend.
At `Phi=0`, the model must bypass the mask path, or reuse the custom shared-logit-bias
attention helper already implemented for the SUA residual adapter. Exact equality must
be tested on the production device and dimensions, not inferred from zero initialization.

### 5.3 Dropout semantics

Current dynamic dropout multiplies the combined input token before `fc_in`. It is not a
key-padding mask: `fc_in` bias and subsequent nonlinearities can still produce a key and
value for a zeroed token. Multiplying `B[c,i]` by the raw dropout mask is also wrong,
because PyTorch dropout scales kept entries by `1/(1-p)`.

For v1, either:

- use the binary keep indicator only for the bias while explicitly retaining existing
  SPINT token-dropout semantics; or
- introduce true key masking as a separately named topology change with its own control.

The second choice must not be silently folded into this experiment.

### 5.4 Cache contract

Only `B = Phi rho(carrier)^T` is session-cacheable. The cache key must bind at least the
session/data identity, carrier hash, `C`, `N`, dtype, and operator checkpoint. Cache and
on-the-fly recomputation must agree exactly within the declared numerical tolerance.

---

## 6. Hardware and state accounting

“Zero recurring compute” is incorrect. The carrier-to-bias matrix multiplication can be
performed once per session, but every streaming attention call still reads/adds the bias
for each head before softmax.

For carrier dimension `D`, the session-time preparation is approximately:

```text
C * N * D multiply-accumulates
```

The cached state is:

```text
C * N elements
```

The online bias application is approximately:

```text
B * num_heads * C * N additions/read accesses per window
```

For the common RT shape `B=1`, `C=2`, `N=64`, `num_heads=64`, this is 8,192 bias
additions per window. The cache is 128 elements: 512 bytes in FP32, 256 bytes in FP16,
or 128 bytes in INT8. This is small relative to QK attention, but it is not zero and it
must appear in the MAC/state/latency receipt.

The direct signed operator has `C*4` parameters: 8 for RT/SUA and 28 for H1. The
five-dimensional relevance form has `C*5`: 10 for RT and 35 for H1. There is no extra
deployment optimizer state.

---

## 7. Claims that must not be used

### 7.1 Softmax does not “average away” carrier noise

Carrier errors share labels, design balance, regression conditioning, and normalization,
so they need not be independent across channels. Softmax exponentiates perturbations and
can amplify an extreme noisy channel. At a large bias scale it can create winner's curse,
not averaging. Noise robustness must be measured with clean, independent-noise, and
correlated-noise synthetic controls; it is not a theoretical property of the design.

### 7.2 This does not exactly contain Direction 3

A very large bias approaches top-score carrier routing, subject to ties and query-specific
scores. Direction 3 used a different fixed-slot/assignment construction and value path.
The correct phrase is “a soft carrier-score routing limit,” not “exactly Direction 3.”

### 7.3 L-D can still attend to a quiet channel

The existing identity token can produce keys/values even when instantaneous neural
activity is zero. L-D's multiplicative activity gain has no incremental effect on the
zero activity component, but the complete decoder is not forbidden from attending to
that channel.

### 7.4 H1 `+0.0389` is not a ceiling

`H-C - H-C0 = +0.038895/+0.039647` on fold 0, seeds 42/43, is an implemented ablation,
not a mathematical upper bound on a better carrier consumer. H1 is excluded from the
first pilot because its carrier construction and matched cross-date attribution remain
unresolved, not because a consumer is bounded to a fraction of `0.0389`.

---

## 8. Mandatory CPU gates before any GPU run

All gates are fail-closed:

1. **Exact null:** zero bias reproduces the ordinary decoder bitwise on the production
   device and shapes, using an explicit no-mask branch where necessary.
2. **Shape:** `[C,N]` and `[B*H,C,N]` behave as declared; `[B,C,N]` is rejected.
3. **Session isolation:** a synthetic mixed-session batch cannot reuse or cross-attach a
   cached bias.
4. **Permutation equivariance:** jointly permuting neural, identity, carrier, and bias
   columns leaves the output unchanged.
5. **Cache parity:** cached and recomputed bias paths agree.
6. **Dropout contract:** binary keep behavior and existing token-dropout behavior are
   tested separately.
7. **Trainability:** gradients reach zero-initialized `Phi`; there is no dead two-factor
   initialization.
8. **Noise null:** clean, independent, and correlated carrier perturbations are reported.
9. **Instrumentation:** record bias/logit RMS, attention KL, and entropy. These are
   diagnostics, not substitutes for the accuracy gates.

If any gate fails, no GPU cell starts.

---

## 9. Conditional RT pilot

### 9.1 Scheduling decision

- L-D remains the higher-prior and simpler operator test and runs first if the RT branch
  is reopened.
- Do not run this on SUA: the close rank-8 predecessor is negative and the formal scope
  is consumed.
- Do not run it first on H1 or M1: current evidence does not establish a carrier-content
  gain there that justifies a deeper consumer.
- The earlier RT R4 M6/M12 full-15 matrix is already complete and sealed; this would be a
  separately pre-registered decoder-operator experiment, not a continuation of an
  unfinished matrix.

### 9.2 First GPU cell

Use fresh source training for every arm, identical source data, seed, fold, epochs,
checkpoint rule, and parameter accounting:

1. `A0`: existing Full carrier path, no attention bias;
2. `B-Full`: existing Full carrier path plus the frozen source-selected attention bias;
3. `B-XLS`: correct Full carrier remains in the identity path, but only the bias receives
   the strong XLSv2 mismatched carrier.

The two primary paired gates are frozen before the run:

```text
B-Full - A0    >= +0.03 R²
B-Full - B-XLS >= +0.03 R²
```

Either failure stops the branch. `B-XLS` is the essential causal control: it asks whether
correct content in the *bias attachment* matters while keeping the ordinary identity
carrier correct. A tiny parameter-matched post-attention additive control is added only
after both first-cell gates pass; it must not be used to rescue a failed first cell.

### 9.3 Expansion

Only after both first-cell gates pass, expand to the pre-declared RT folds (for example
`0/7/14`, subject to the existing RT split authority). Require both contrasts to have:

- mean paired delta at least `+0.03 R²`; and
- positive delta in every pre-declared fold.

No search over rank, temperature, head-specific bias, carrier map, normalization, or
entropy threshold is allowed after target results are visible. A full-15 expansion is
considered only after the three-fold gate.

---

## 10. Final disposition

The original Direction-4 rationale overstates both novelty and hardware simplicity. A
nearby, more expressive SUA attention-logit residual has already failed. The remaining
scientifically defensible question is narrower:

> When trained jointly from scratch on RT, can a source-selected, preferably
> sign-invariant carrier-relevance bias improve both the correct-carrier baseline and a
> wrong-bias-attachment control by at least `0.03 R²`?

That is a legitimate conditional experiment. It is not currently the main line, is not
ready for GPU execution until the CPU contracts pass, and should follow the simpler L-D
test. If it fails either paired gate, close carrier-biased attention rather than adding
more fusion variants.
