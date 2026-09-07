# Backprop-free held-out calibration: mechanism and pre-closure idea audit

**Maintained:** 2026-08-04 (Asia/Hong_Kong)  
**Status:** development analysis; it neither authorizes a GPU run nor opens a formal endpoint  
**Scope:** sorted SUA, deterministic pseudo-MUA, and native threshold-crossing MUA

## 1. The contribution should be stated at the correct level

The most defensible emerging contribution is not “a four-number feature improves SPINT.” It is:

> A source-trained decoder can adapt to a new session without target-session backpropagation by
> fitting a small, causally available functional carrier whose sufficient statistics have a known
> transformation law under neural-unit aggregation, then caching the resulting session identity
> for all future query windows.

This statement has four separately testable parts:

1. **functional content:** the descriptor carries movement-related information not recoverable from
   a width-matched zero or row-shuffled input;
2. **causal estimation:** only chronological support and its permitted labels enter calibration;
3. **deployment adaptation:** target-session weights never change and the finalized identity is
   computed once, not rebuilt on every query window;
4. **granularity robustness:** the carrier has a principled SUA-to-MUA transformation rather than
   relying on persistent unit identities.

The current evidence supports the first part most cleanly on SUA, supports an exact controlled
version of the fourth part on pseudo-MUA, and supports end-to-end native-M2 usefulness with an
epoch-confounded official comparison. Fresh C1 and native-M2 Phase C test the two missing causal
links. Native M1 is a negative boundary, not supporting evidence.

## 2. Decompose the method before proposing variants

Every proposal belongs to one of three layers:

| Layer | Question | Current evidence |
|---|---|---|
| carrier | What session-local statistic describes a unit/channel? | attached first-harmonic coefficients `[a,c]` dominate the SUA gain |
| estimator | How is that statistic recovered from a small chronological block? | ordinary analytic fit works at supported budgets; low-budget correction and alternate GLMs did not pass source-only gates |
| consumer | How does the source-trained decoder use the statistic? | ordinary T4 fusion works on SUA; seven additional network-side fusion ideas did not establish a better consumer |

This decomposition prevents a common error: when an estimator is noisy, adding attention to the
consumer cannot create the missing information. Conversely, when the descriptor already has a
large aligned-versus-shuffled effect, replacing it with a higher-capacity latent code gives up
identifiability without first establishing an information deficit.

## 3. Why AC4 is an aggregation-compatible functional carrier

For calibration trial `t`, define the shared design row

```text
x_t = [cos(theta_t), sin(theta_t), 1].
```

For unit `i`, with trial rate vector `r_i`, ordinary least squares gives

```text
beta_i = [a_i, c_i, b_i]^T = (X^T X)^(-1) X^T r_i.
```

For an electrode channel `k` formed by summing units `G_k`, `r_k = sum_i r_i`. Therefore

```text
beta_k = (X^T X)^(-1) X^T sum_i r_i = sum_i beta_i.
```

The equality holds because every unit uses the same trial design and the estimator is linear in
the response. Thus `[a,c,b]` is aggregation-homomorphic. The magnitude
`m=sqrt(a^2+c^2)` must be recomputed after aggregation and is not additive:

```text
m_k = |sum_i (a_i + j c_i)|, generally not sum_i |a_i + j c_i|.
```

This distinction matters mechanistically. Units with opposing preferred directions can cancel in
native or pseudo-MUA even when each SUA unit is strongly tuned. A method that treats per-unit
phase or magnitude as invariant under pooling is structurally wrong; the signed vector `[a,c]`
has the correct merge law. The completed SUA attribution and row-attachment control are therefore
consistent with the algebra rather than merely with a wider side-input layer.

The pseudo-MUA audit proves exact count conservation and numerical coefficient conservation for a
controlled deterministic merge. Native threshold MUA is only approximately additive: threshold
nonlinearity, unsorted multiunit mixtures, refractory effects, and hardware preprocessing can
break the exact equality. This is why pseudo-MUA is a mechanistic bridge but native M2 remains a
necessary empirical endpoint.

## 4. What determines the required number of labels

Conditional on the calibration design and a homoscedastic residual variance `sigma_i^2`, the
coefficient covariance is

```text
Cov(beta_i | X) = sigma_i^2 (X^T X)^(-1).
```

Nominal trial count is therefore not the relevant quantity by itself. The weakest eigenvalue of
the centered directional design determines the worst-estimated direction in coefficient space.
Ten balanced directions can be more informative than many trials confined to a narrow arc.

Three consequences follow:

1. **M1 failure is expected:** ten trials over a half-plane/narrow support produce extrapolation
   and poor conditioning; early overlap-contaminated results were not evidence against this.
2. **low-budget T4 is unstable before `b` is unstable:** the intercept is an average-rate
   statistic, whereas `[a,c]` must resolve a two-dimensional directional vector.
3. **`m` is especially unsafe at low signal-to-noise:** it is a nonlinear norm and acquires a
   positive noise bias even when the true modulation approaches zero.

Existing score-free geometry receipts make this contrast quantitative:

| Quantity | native M1 | native M2 post-33 |
|---|---:|---:|
| directional geometry | eight targets on `0--157.5` degrees | eight targets around the full circle |
| support trials | `10` | `33`, of which `16` are directional and `17` centre/rest |
| balanced design condition number | about `4.807` | about `1.414` |
| observed seven-session first-33 condition range | not applicable to this receipt | `1.584--2.090`, median `1.782` |
| analytic `sd(c)` at the operating point | `1.0438` | `0.3755` |
| intercept--sine noise correlation | `-0.9024` | `0.0150` |

The roughly `2.78x` analytic standard-deviation ratio for the weak M1 coefficient is only part of
the problem. Repeating more M1 trials reduces variance as `1/sqrt(M)` but does not remove the
half-plane design correlation: the reported condition number remains approximately `4.78--4.87`
from 10 through 400 cyclically balanced trials. Thus M1 needs either different target geometry or
a different estimand; merely spending more trials on the same directional arc does not turn its
cosine coefficients into the M2 object.

Evidence sources are the score-free `m1_t4_mechanism_v1/audit.json` and the frozen Phase-A
native-M2 post-33 endpoint receipt. These geometry numbers do not use query R2 and do not authorize
an M1 rescue run.

This also answers the label question precisely. T4 needs one permitted target direction for each
rewarded calibration trial used in the fit, not dense query labels and not target-session gradient
updates. The effective label budget is the directional information in `X`, not only `M`. Ordinary
SPINT/B3 receives the same neural support but no target directions, so a T4-versus-SPINT result is
a valid deployment comparison but not a same-information ablation. T4-versus-row-shuffled T4 and
AC4 attribution are the cleaner mechanism controls.

## 5. Relationship between B3/SPINT activity calibration and T4

T4 does not replace the activity calibration path. In the selected implementation, chronological
neural support first supplies the usual SPINT/B3-style temporal identity information; T4 supplies
an additional fitted functional carrier to a source-trained consumer. The useful distinction is:

```text
B0 / SPINT: neural support -> activity identity -> decoder
T4 model:   neural support -> activity identity --+
             labelled support -> [a,c,m,b] -------+-> jointly source-trained identity consumer
```

The source decoder may be initialized from SPINT and co-trained with the T4-aware encoder. At a
held-out session, however, all network weights are frozen. Calibration consists of sufficient-
statistic accumulation, analytic descriptor fitting, one identity forward pass, and caching. An
implementation that recomputes support identity for every query is numerically similar but does
not satisfy the intended deployment mechanism or cost claim.

State accounting follows the same lifetime. Raw support and T4 coefficients exist while the
session identity is finalized; ordinary cached query decoding then requires only `E` and the live
neural window. Descriptor bytes are therefore calibration-state bytes, not automatically
additional permanent online bytes. An evaluator may retain the full outer dataset in host memory
for scoring, but that process residency must not be confused with the minimal decoder state.

## 6. Divergence: candidate ideas before filtering

The list deliberately includes ideas that should be rejected, so absence of later pursuit is an
evidence-based decision rather than forgotten scope.

| ID | Candidate | Layer | Initial attraction | Existing or likely kill condition |
|---|---|---|---|---|
| I1 | matched native-M2 SPINT/T4 post-33 LOSO | endpoint | removes the official epoch confound | stop after Stage A only under the frozen severe-negative rule |
| I2 | paired SUA/pseudo-MUA co-training | consumer/training | one weight set may learn the known merge transformation | C1 dual-view non-inferiority or attachment gate fails |
| I3 | one-way consistency C2 | training | teacher view may regularize the weaker view | not licensed unless complete C1 passes |
| I4 | rotation-equivariant scalar shrinkage of complex `z=a+jc` | estimator | preserves geometry while reducing variance | current EB/correction evidence is negative; requires independent data |
| I5 | analytic confidence-gated interpolation between activity-only and AC4 | estimator/consumer | abstains when design is ill-conditioned | fixed/static gates and correction maps failed; no current GPU license |
| I6 | confidence-conditioned FiLM | consumer | exposes session-specific uncertainty | new fusion family has repeated negative evidence; stop in current scope |
| I7 | D-optimal or direction-balanced calibration acquisition | protocol | increases information without increasing label count | useful only if deployment controls or can select targets |
| I8 | calibration stopping rule based on the smallest design eigenvalue | protocol | converts nominal `M` into a reproducibility guarantee | improves safety/label efficiency, not necessarily R2 |
| I9 | second-harmonic tuning | estimator | captures non-cosine responses | failed the source-only Experiment-B gate |
| I10 | Poisson/robust GLM | estimator | better spike-count likelihood | failed the source-only Experiment-B gate |
| I11 | learned cross-budget correction MLP | estimator | predicts low-budget coefficient error | target-free development transfer was negative |
| I12 | fixed-K temporal K/V memory | carrier/consumer | retains within-trial dynamics | P20 lost to order-invariant B20 in all four source sessions |
| I13 | population subspace/Procrustes alignment | carrier | avoids individual-unit drift | variable unit axes lack a clean cross-session correspondence; no carrier gate |
| I14 | waveform/SNR/electrode identity | carrier | cheap physical metadata | prior SUA tests show no decoding value; not movement-functional |
| I15 | more decoder-side attention, FiLM, or dynamic weights | consumer | increases interaction capacity | seven network-side variants failed; complexity is not missing information |
| I16 | native simultaneous SUA/MUA coefficient-conservation audit | mechanism | measures where exact pooling law breaks in threshold MUA | depends on paired recordings/data availability |
| I17 | independent-subject AC4 attachment confirmation | endpoint | tests whether sub-C mechanism generalizes | requires untouched subject/formal scope and frozen candidate |
| I18 | matched epoch-34 B0 official control | endpoint | removes decoder-version confound on hidden M2 | requires an isolated package and explicit submission authorization |
| I19 | online label-free intercept/drift update | estimator | tracks nonstationary firing-rate drift after support | must not consume query behavior; separate from AC4 claim |
| I20 | INT8 encoder/descriptor path after accuracy closure | deployment | tests hardware viability | premature until FP32 candidate and cached semantics are frozen |

## 7. Convergence under evidence, simplicity, and publication value

Applying the explain-it, simplicity, stakeholder, and feasibility filters leaves four priorities:

1. **I1 native-M2 matched confirmation — highest causal value.** It directly tests whether the
   label-assisted carrier improves native MUA relative to the same-fold, same-seed source-selected
   SPINT decoder. It is the only remaining local experiment that can deconfound the official M2
   gain.
2. **I2 C1 paired-view co-training — highest granularity value.** It tests whether one shared
   model can serve both sorted units and their deterministic electrode pooling while preserving
   attached AC4 content. Its primary result is non-inferiority and shared deployability, not a
   promise of higher peak R2.
3. **I7/I8 design-aware calibration — highest future label-efficiency value.** The estimator
   covariance identifies a principled control variable: directional design information. This can
   first be audited on CPU, but it should not be turned into a GPU arm in the already-viewed scope.
4. **I17 independent-subject confirmation — highest generalization value.** If a frozen candidate
   is available, external-subject evidence is more informative than another network variant on
   the six reused sub-C development sessions.

I3 remains conditional. I4/I5 are scientifically coherent but currently defeated by stronger
negative evidence and data reuse; they are reserve hypotheses, not active experiments. I6 and
I9--I15 should stop. I16 and I18 are valuable audits/endpoints but depend on data or new external
authorization. I20 follows, rather than precedes, FP32 accuracy closure.

## 8. Two-sentence research pitch

Held-out neural sessions normally require gradient adaptation or stable unit correspondence, both
of which are awkward for low-latency implants. We instead source-train a decoder to consume a
causally fitted, aggregation-equivariant functional identity, then adapt a new SUA or MUA session
with one analytic fit and one cached forward pass—no target-session backpropagation.

## 9. Three decisive validations

### Validation A — native M2 causal bridge

Run exact outer-session LOSO pairs with the same seed, source sessions, source-only checkpoint
selection, chronological 33-trial support, query start, and decoder lineage. SPINT uses neural
support; T4 uses the same support plus permitted trial directions. Report paired R2, sign stability,
finite absolute scores, calibration/backward/update counters, descriptor-fit time/state, one-time
identity time/state, and cached-query time.

### Validation B — controlled granularity bridge

Complete C1 without opening intermediate scores. Compare shared versus separately trained T4 in
both SUA and deterministic pseudo-MUA, and compare shared T4 against shared row-shuffled TS4.
Passing means one weight set is non-inferior in both views and still depends on correct carrier-row
attachment; it does not by itself prove native threshold-MUA equivalence.

### Validation C — independent generalization

Freeze the carrier, consumer, calibration budget, query boundary, and checkpoint rule before using
an untouched subject or separately authorized hidden endpoint. The primary check is attached AC4
or T4 versus its matched activity-only and row-shuffled controls. This is more valuable than tuning
another consumer on already-inspected sub-C sessions.

## 10. Short feasibility program

The immediate program is intentionally narrower than the divergent list:

1. finish C1 and apply its frozen aggregate once;
2. finish Phase-C cached-deployment parity, cost, provenance, and authorization audits;
3. run only native-M2 Stage-A exact14 first, then obey the frozen continuation rule;
4. if the full native-M2 aggregate is positive, freeze the FP32 mechanism before encoder/identity
   INT8 work;
5. reserve external-subject/formal evidence for the single frozen candidate, not an arm search.

No additional network-side fusion, temporal memory, waveform feature, or low-budget correction GPU
run belongs in this program without new independent evidence that reverses its existing kill result.

## 11. Strongest objection and response

**Objection:** T4 is simply supervised target metadata, so improvement over label-free SPINT is
expected and does not establish a better neural decoder.

**Response:** The objection correctly limits the claim: the comparison is not same-information.
The contribution is the amount and deployment form of supervision—small chronological trial-level
labels, an analytic estimator, no target-session optimizer/backward pass, aggregation-compatible
state, and cached online inference. Component attribution, label shuffle, and row attachment test
whether the gain is carried by the claimed functional coefficients rather than width; matched
native-M2 LOSO tests whether that mechanism adds useful held-out-session accuracy under an exact
decoder lineage. Both types of evidence are required.

## 12. Publication-safe outcome map

| C1 | native M2 matched | Allowed central conclusion |
|---|---|---|
| pass | positive | aggregation-compatible bp-free functional calibration works across controlled SUA/pseudo-MUA granularity and improves native M2 under a matched local protocol |
| pass | null/negative | shared granularity robustness is supported, but native-MUA accuracy benefit is dataset/geometry dependent |
| fail | positive | native-M2 benefit is supported, but one universal SUA/pseudo-MUA weight set is not |
| fail | null/negative | retain the clean SUA AC4 mechanism and official end-to-end M2 system result; state that causal native-MUA and shared-granularity confirmation were not obtained |

In every branch, M1 remains an explicit low-budget/narrow-geometry failure boundary. A negative
branch still yields a useful methodological result only if calibration chronology, information
asymmetry, decoder lineage, uncertainty, and cached deployment cost are reported without dilution.
