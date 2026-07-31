# T4, streaming B3T, and sorted-SUA auxiliary information: experiment program

**Status:** proposed validation-only program. This document authorizes neither training
nor formal held-out evaluation. It preserves the occupied `sub-C/CO/27-6-6` test scope.

**Updated:** 2026-07-29.

## 1. Research question: do not optimize the view gap

The motivating observation needs to be challenged. Under the matched pseudo-MUA bridge,
T4 has the following absolute validation-development scores:

| View | F0 | T4 | T4-F0 |
|---|---:|---:|---:|
| sorted SUA | 0.313987 | 0.566749 | +0.252761 |
| electrode-pooled pseudo-MUA | 0.208382 | 0.526121 | +0.317739 |

After T4, `SUA-pseudo-MUA=+0.040627 ± 0.015256 SE` (5/6 sessions, 3/3 seeds
positive). Before T4 it was `+0.105605`. The gain interaction is
`Gamma=(T4-F0)_SUA-(T4-F0)_pMUA=-0.064978 ± 0.030744 SE`, which remains
**indeterminate** under the frozen practical `±0.03` rule. Thus T4 may have recovered
functional information that electrode pooling preserves; alternatively, sorted-SUA
information may be unused; both views may be near a task/protocol ceiling; or more than
one explanation may hold. A smaller or larger gap alone distinguishes none of these.

The program keeps four non-substitutable estimands:

| Estimand | Definition | Legitimate use | Never use it as |
|---|---|---|---|
| Absolute SUA | `R²_SUA(A)` | whether a SUA arm is useful | proof of unique sorting content |
| Absolute pseudo-MUA | `R²_pMUA(A)` | controlled pooling robustness | true threshold-crossing MUA evidence |
| View gap | `G_A=R²_SUA(A)-R²_pMUA(A)` | diagnostic only | optimization target |
| Gain interaction | `I_A=(A-F0)_SUA-(A-F0)_pMUA` | whether gain differs by view | specificity without its own gate |

A candidate earns priority only by controlled improvement of an absolute endpoint, or by
non-inferior absolute performance at lower label/compute cost. A larger gap accompanied by
worse SUA is a failure; so is a larger gap produced only by degrading pseudo-MUA while SUA
stays unchanged. A smaller gap caused by pseudo-MUA recovery can be valuable. Every result
remains validation development evidence; pseudo-MUA is derived from sorted SUA and is not an
external real-MUA replication.

## 2. Fixed calibration accounting and common boundary

Current evidence is **T4@50**, not T4@30. B3 activity identity uses the first
`M_activity=30` calibration trials. T4 fits target-direction-conditioned rates from the
first `M_T4=50` rewarded trials. Matched arms all evaluate after trial 50, even F0; no
evaluation label updates encoder/decoder/feature statistics.

The primary label-budget estimand fixes `M_activity=30`, varies only
`M_T4={5,10,15,20,30,50}`, and evaluates all arms on `[50:]`. This isolates supervised
direction-label cost from activity evidence. A later separately chartered joint-latency
experiment may set `M_activity=M_T4`; it is not pooled with the primary estimate.

For every budget, pre-report direction-count vector, design-matrix rank and condition
number of `[1, cos(theta), sin(theta)]`. Rank 3 is mandatory; the old “at least two
directions” guard is insufficient. A rank-deficient fit is a declared measurement
degeneracy, not an architecture failure. For full-rank but ill-conditioned fits, use a
predeclared train-only ridge/shrinkage rule and a confidence flag. Before any training,
freeze either a balanced direction schedule, where protocol permits, or natural rewarded
ordering plus coverage/condition stratification; do not choose after observing R².

At `N=64,T=100,M_activity=30`, T4 adds 256 learned parameters, 1,024 FP32 bytes and
16,384 MAC/session to B3 (T4: 18,290 parameters; 13,033,472 MAC/session); online 50-Hz
MAC is unchanged. The real cost is labeled calibration directions/latency. Deployment may
stream direction-specific rate sums and counts; it need not preserve all trial rows or run
a generic SVD per channel.

## 3. Audit: what prior auxiliary negatives do and do not close

| Evidence | Fusion, control, budget actually used | Result | Ruled out | Still untested |
|---|---|---|---|---|
| F1 | waveform `[p2p, noise std, SNR]` static concat once before B3S `psi`/3-layer `post_pool`; dimension-matched FS1; pool 50/activity 30; 12 epochs/3 seeds | `F1-F0=-0.0324`; `F1-FS1=-0.0457`; **indeterminate** | this raw static concat | reliability-conditioned use |
| F2 | F1 plus width, peak/trough ratio, repolarization slope; static concat plus FS2 under same protocol | `F2-F0=-0.0012`; `F2-FS2=-0.0367`; **indeterminate** | this six-scalar static concat | shrinkage/FiLM/relation mechanisms |
| F3/FS3 | F2 plus absolute electrode-ID embedding concat | implemented, never formally run | nothing empirically | electrode information independent of F2 |
| T4/TS4 | `[a,c,m,b]` static concat at the same B3S `psi`/3-layer `post_pool`; row shuffle; 50 labeled trials | SUA `+0.2528`, 6/6 and 3/3; **effective** | proves aligned content, not width, matters | component attribution, low budget, confidence |
| T4GATE D | static absolute-ID scalar multiplicative gate on T4 identity, with ID shuffle | `-0.0108`; **ineffective** | this fixed per-array reliability table | relation, session-specific confidence, geometry |
| absolute-ID A/C | A concat embedding; C additive anchor table | implemented but unrun | nothing | no justification to run merely because code exists |

The key corrective is narrow: F1/F2 and T4 both used B3S concat plus the same 3-layer
post-pool MLP and 12-epoch protocol. T4's success proves that concat is not generally
incapable of reading side information. Hence “concat was simple” cannot alone revive F1/F2.
The narrower, falsifiable auxiliary hypothesis is that waveform/amplitude/quality describe
**measurement reliability**, not functional identity content, and may require explicit
shrinkage/FiLM or within-electrode context. It must beat a parameter-matched nonlinear
concat MLP and shuffled controls or be killed.

Implementation facts reinforce the audit. `SideFeatureEarlyPoolEncoder` accumulates activity
trial-by-trial then concatenates side values once before `post_pool`; it has no explicit
multiplicative activity-side interaction. Waveforms are calibration-pool template scalars.
T4 is separately calculated from target-direction-conditioned calibration rates and z-scored
using train sessions only. T4 is supervised, not label-free.

Three electrode hypotheses must remain distinct:

1. **Absolute ID:** “electrode 37 has a persistent learned value.” D is the only completed
   direct T4 test and failed; A/C are unrun and array-specific.
2. **Same-electrode membership relation:** “sorted units partitioned from an electrode offer
   local split/merge/quality context.” It has no global ID table and is permutation equivariant.
3. **Geometry:** “nearby physical sites matter.” Untestable here: coordinates are absent and
   Utah pin number is not geometry. Acquire the laboratory `.cmp` map first.

## 4. Failure/boundary probing, composition/decomposition, simplicity test

The raw idea list is deliberately wider than the final program.

| ID | Raw idea | Decision |
|---|---|---|
| R1 | T4 `[a,c,m,b]` components | retain: attribution prerequisite |
| R2 | target-label shuffle retaining rates | retain: direction versus rate control |
| R3 | `M_T4` sweep separate from activity | retain: label-cost test |
| R4 | condition/residual/spike-exposure confidence | retain |
| R5 | FiLM activity conditioning | retain only low-rank/zero-init |
| R6 | full bilinear tensor | merge into low-rank FiLM; reject unbounded capacity |
| R7 | B3T+T4 composition | retain |
| R8 | truly bin-streaming B3T | implemented 2026-07-31; retain as engineering prerequisite |
| R9 | raw waveform/template encoder | defer |
| R10 | retry raw waveform concat | reject: repeats F1/F2 |
| R11 | absolute electrode table | defer/kill: D ineffective |
| R12 | hierarchical same-electrode set relation | conditional eligibility audit first |
| R13 | geometry | blocked by data |
| R14 | generic attention/larger model/longer training | reject: no isolated mechanism |
| R15 | joint SUA/pseudo-MUA co-training | reserve after a mechanism passes |

The resulting four candidates are below. They are sequential; no large all-mechanism model
is authorized.

### C1. Factorized and uncertainty-aware T4 (first scientific screen)

**Mechanism and minimum form.** Compare `TAC=[a,c]`, `TMB=[m,b]`, full
`T4=[a,c,m,b]`, target-label-shuffled T4, and row-shuffled `TS4`. After that screen only,
add calibration-only confidence: fit residual/deviance, modulation/residual ratio, spike
exposure, direction counts, `log(condition number)` and rank-valid bit. Amplitude/SNR/
waveform-quality may enter this confidence vector only if computed strictly inside the
calibration block; they do not re-enter as raw identity content.

**State, cost and controls.** Store streaming direction rate/count sufficient statistics,
not full trial-rate rows, in a versioned cache namespace that includes `M_T4`, ordering/
schedule, rank rule, confidence definition, signal view and mapping fingerprint. It is
session-rate state, zero online MAC. Controls are T4, TS4, target-label shuffle, same-dimension
confidence row-shuffle/mismatch, and a label-free rate-only matched side vector. pseudo-MUA
must fit T4 after pooling, not average unit rows.

**Kill/branch rule.** If `TMB` is non-inferior to full T4 (paired lower 2SE bound >= -0.03)
and `TAC` is not, stop calling it a directional mechanism. If label shuffle retains full T4,
stop the directional claim. Confidence advances only if it adds +0.03 at fixed budget or
attains non-inferiority to T4@50 using `M_T4<=20/25`; otherwise no larger fusion network.

### C2. Actually bin-streaming B3T+T4 (efficiency factorial)

**Mechanism.** B3T is independently positive (`+0.0418 ± 0.0090`, six seeds,
effective_heterogeneous) and is about -31% parameters/-65% session MAC versus B3. Its
combination with functional T4 tests a principled composition.

**Implemented boundary (2026-07-31).** `TemporalBasisEarlyPoolEncoder/B3T` and its
side-feature subclass `B3TS` now expose the actual bin API
`start_trial -> push_sample -> end_trial` with `supports_bin_streaming=True`. `B3TStream`
is therefore an execution mode of the same learned B3T/B3TS network, not a second model
whose extra name would require redundant accuracy training. At trial start it allocates only
`[B,N,K]` (`K=12`) basis coefficient sums; each chronological valid bin adds
`x_t*basis[:,t]`; at end-of-trial it applies the existing learned
`K->D` projection+ReLU and adds to `sum_feat`. T4/TS4 is still appended only once at
finalization before `psi`. No complete `[B,N,T]` trial appears in streaming state.

For `N=64`, the exact transient accumulator is `64*12*4 = 3,072 B`, versus the former
`64*100*4 = 25,600 B` full-trial assumption, plus the unchanged `64*64*4 = 16,384 B`
support accumulator. MAC remains `N*T*K + N*K*D` per trial plus post-pool/fusion. The
vectorized `push_trial` path is retained only as the training/reference implementation; it
uses the same parameters, honors variable valid lengths, and is checked against bin
execution.

**Controls/tests.** Factorial `B3/B3T/T4/B3TStream+T4/B3TStream+TS4`. The implementation
tests now cover batch versus vectorized full-trial exact equality, bin-stream agreement
(`atol=2e-5`, because accumulation order differs), variable valid length and adversarial
padding tails, chronological/end-of-trial guards, joint unit+side permutation, T4
finalization, absence of retained full trials, and exact state/MAC `cost_profile`. The
targeted CPU suite reports `19 passed`.

**Frozen execution gate (2026-07-31).** The architecture comparison uses matched fresh
training with `M_activity=M_T4=30`, common evaluation start 30, 12 epochs and the fixed
epochs 5--12 window. Seed 42 first runs fresh `T4` and `B3TStream+T4`. Only if their strict
artifacts give `B3TStream+T4−fresh T4 >=−0.03` may the same-seed
`B3TStream+TS4` content control start. The control runner invokes this read-only gate before
creating a run, result or log path. Seeds are restricted to the predeclared `42/43/44`;
checkpoint paths and byte hashes, teacher/manifest/normalization provenance, exact sessions,
formal seal and complete cost receipts are fail-closed.

**Kill rule.** Correct content contrast is mandatory. Advance only when B3TStream+T4 is
non-inferior to the matched fresh T4@30 within -0.03 and exact parameters/MAC are both
>=25% lower. T4@50 remains the reference for the separate label-efficiency question, not
for this architecture comparison. If the aligned seed-42 arm is worse than -0.03, do not run
the TS4 control and do not add FiLM on this substrate. If aligned non-inferiority passes but
the strict T4-vs-TS4 content gate fails, stop after the control.

### C3. Low-rank activity-conditioned functional fusion

**Mechanism/minimum form.** Explicitly condition activity by functional state rather than
unboundedly widening `psi`:

```text
c_i = [selected T4_i, confidence_i, optional calibration-only quality_i]
r_i = U sigma(V c_i)                 # rank r=4 or 8
h'_i = (1 + A r_i) ⊙ h_i + B r_i
E_i = psi(h'_i, selected T4_i)
```

Zero-initialize output paths `A/B` so the first forward equals selected T4+B3T. This has
`O(D*r+r*q)` shared parameters and `O(N*D*r)` session MAC, no new online token/MAC/state.

**Controls.** Same-dimensional content-shuffled/mismatched `c_i`; label-free rate-only
confidence; waveform-quality row shuffle; and parameter-matched nonlinear per-unit concat
MLP with no multiplicative activity condition. If FiLM cannot beat the latter, its structure
has not earned a claim.

**Kill rule.** At `M_T4=50`, require +0.03, 5/6 positive sessions and all seed means positive
versus both selected concat baseline and matched MLP. At low budget, non-inferiority to T4@50
within -0.03 is the alternate efficiency success. Otherwise stop raw waveform/amplitude work.

### C4. Same-electrode relation, conditional on a read-only eligibility audit

**Why this is not a casual revival.** E3/E4 previously deferred same-electrode pooling because
mean units/electrode was only 1.3-1.8 and many electrodes were singleton. A new read-only audit
shows the unit-side coverage is nevertheless material: in validation sessions 20151103/04/06/
09/10/12, multi-unit electrodes are 21.2-39.0% of electrodes, while their sorted-unit shares
are 38.1%, 55.9%, 58.3%, 61.5%, 60.7%, and 57.9% (5/6 above 55%; max group size 3-4). This
supports an eligibility audit, not a positive result. Descriptive correlations of multi-unit
share with existing F0/T4 SUA-pMUA gap are only ~0.026/~0.034 (`n=6`), so multiplicity alone
does not explain merge sensitivity and is not a statistical conclusion.

**Stage-0 eligibility gate.** Before training, read-only decompose existing F0 and T4
SUA-pMUA gap by multi-unit versus singleton source-electrode strata; audit whether within-group
T4/activity/waveform-quality heterogeneity, rather than simple group size, predicts merge
sensitivity. Predefine the stratified aggregation and report coverage/session table. If the
gap is not concentrated in multi-unit strata, or heterogeneity does not explain it at a useful
descriptive level, kill this route. “Not adequately powered” is distinct from negative when
the predeclared stratum has insufficient coverage.

**Minimal relation.** No absolute ID table:

```text
u_i = small_mlp([h_i, T4_i, confidence_i])
g_e = mean_{j: electrode(j)=e}(u_j)
delta_i = W[u_i, g_e, u_i-g_e, log(group_size)]
E_i = E_i^(T4+B3T) + zero_init(delta_i)
```

It is equivariant to unit permutations and invariant within an electrode group. Pseudo-MUA
has singleton channel groups, so `u_i-g_e=0` and the residual must reduce to baseline apart
from a controlled group-size constant: a required boundary test, not an expected gain.

**Controls/cost/kill.** Use membership shuffle preserving per-session group-size histogram,
parameter-matched no-group MLP, within-group feature mismatch, and waveform-quality shuffle
if quality is present. Use segmented sums/counts and `[B,N,r]` state, no `N²` attention nor
global lookup; report exact linear-in-N MAC/state. Require SUA gain over both base and
no-group MLP, no single-session domination, predeclared multi-unit stratum support, and all
controls. Otherwise kill; do not run A/C absolute-ID tables as consolation.

## 5. Staged factorial, seeds, views, and stopping

All runs use V4: unique run directory, `E=12`, no early stopping, epochs 5-12 mean,
train-only normalizers, same-session/same-seed pairing, and no test neural/behaviour/trials.
Arms in Stages 1-3 do **not** share weights; matching numerical seed/data/order makes the
contrast paired. Artifact reuse is allowed only after exact configuration, boundary, split,
feature-statistics and provenance hash verification.

| Stage | Arms | Views/seeds | Primary outcome | Advance rule |
|---|---|---|---|---|
| 0 | contracts; C4 eligibility audit | one train + fixed validation smoke; read-only current-artifact decomposition | no leakage, rank/condition, cache isolation, B3TStream equivalence, C4 coverage/heterogeneity | any failure blocks training; failed C4 audit kills C4 |
| 1 | `F0,T4,TAC,TMB,T4-label-shuffle,TS4` | SUA+pMUA; 42/43/44; activity 30/T4 50; eval `[50:]` | absolute endpoints, content, gap and interaction separately | choose minimal defensible T4 or stop fusion |
| 2 | `B3,B3T,T4,B3TStream+T4,B3TStream+TS4` | both views; 42/43/44 | T4 content retained plus exact cost/non-inferiority | only cost-dominant content-valid substrate reaches C3 |
| 3 | selected concat+confidence, FiLM, matched MLP/shuffles; pilot budgets 10/20/50 first | SUA primary; pMUA boundary at 20/50; 42/43/44 | low-budget non-inferiority or controlled superiority | expand to all budgets only by predeclared rule |
| 4 | C4 relation plus its controls | SUA primary/pMUA singleton boundary; 42/43/44 | relation vs base, membership shuffle and no-group MLP | retain only controlled sorting-specific evidence |

If a planned three-seed comparison is indeterminate, add the complete predeclared seed block
45/46/47 to all necessary factorial arms, never only the apparent winner. V4 `effective`
requires mean delta >=+0.03, 5/6 positive sessions and all seed means positive.
`ineffective` means `mean + 2*paired_SE < +0.03`; otherwise report
`effective_heterogeneous` or `indeterminate`. Non-inferiority requires lower paired 2SE
bound >=-0.03 plus its stated cost reduction. Report paired and unpaired SE, session signs,
epoch and seed dispersion, exact views, common boundary, and group-size strata.

## 6. Compute/data budget and two-week pilot

**Reusable:** T4/TS4 feature extraction, train-only stats, pseudo-MUA pooling/fingerprints,
shuffle utilities, V4 evaluator, raised-cosine basis, existing B3/B3T/T4 artifacts and B3S
finalization location.

**New code/cache/tests:** `t4_component_confidence_v1` namespace carrying budget, ordering,
rank/shrinkage, view and mapping fingerprint; B3TStream factory/config/tests; low-rank and
relation variants; zero-init/permutation/singleton/error/cost tests; runner/aggregator checks
for no-test provenance, stale directories and common boundaries. No existing receipt/cache may
be reused without namespace compatibility.

Stage-1 maximum is 6 arms x 2 views x 3 seeds = 36 cells. Stage-2 maximum is 5 x 2 x 3 =
30 cells, subject to verified reuse. Stage-3 begins only with budgets 10/20/50, not an
unbounded six-budget search. Stage-4 is unscheduled unless C4 eligibility and C3 both pass.

| Days | Deliverable |
|---|---|
| 1-2 | freeze component/confidence/rank policy; label-shuffle/balance/cache tests; C4 read-only eligibility audit |
| 3-4 | B3TStream implementation tests: batch/full/bin, padding, T4 finalize, state/MAC; one train+validation smoke |
| 5-7 | Stage 1, both views, three seeds; choose minimal T4 or stop |
| 8-10 | Stage 2 only; prove stream efficiency and content retention |
| 11-12 | implement C3 plus all matched controls; static tests/smoke |
| 13-14 | C3 pilot at 10/20/50; decide efficiency pass, full predeclared extension, or stop |

## 7. Paper-safe interpretation and objections

| Pattern | Can say | Cannot say |
|---|---|---|
| SUA rises versus all content/matched controls | tested sorted-SUA auxiliary mechanism helps | a larger gap itself is the objective |
| both views rise / gap shrinks | mechanism is robust to pooling | views are equal without equivalence gate |
| no auxiliary gain on T4 | tested mechanism unsupported/indeterminate | all SUA auxiliary information is absent |
| relation helps SUA and pMUA singleton reduces to base | consistent with within-electrode sorted-unit heterogeneity | absolute ID or geometry is useful |
| T4@20 non-inferior to T4@50 | this supervised calibration cost can shrink | system is label-free/activity calibration shrank |

**Objection: T4 is a supervised shortcut.** Accept the boundary: component/label-shuffle/
rate-only controls identify what is used, while `M_T4`, coverage and condition quantify label
cost. No uncontrolled T4@50 comparison proves efficiency.

**Objection: waveform was already negative.** Not quite: F1/F2 are static-concat,
three-seed indeterminate screens. The program only tests the narrower reliability hypothesis,
and kills it if structured fusion loses to parameter-matched concat.

**Objection: relation pooling is another ID lookup.** It is not: it consumes equality/group
membership within each session, not an implant-wide embedding. It has a sharp singleton
boundary and an explicit eligibility/heterogeneity kill gate. Geometry remains blocked.

**Objection: B3T was only trial-streaming.** That objection was valid for the earlier
implementation. Since 2026-07-31, B3T/B3TS expose a true bin API and retain only `[B,N,K]`
current-trial coefficients. The batch/full/bin, padding, state and permutation contracts
pass. Accuracy still has to pass the separately predeclared matched T4/B3T+T4/B3T+TS4
experiment; the execution equivalence result is not itself an R² claim.

These conclusions are constrained by [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) §§J-K,
[`T4_OPTIMIZATION_DIRECTIONS.md`](T4_OPTIMIZATION_DIRECTIONS.md),
[`PSEUDO_MUA_T4_BRIDGE_48H.md`](PSEUDO_MUA_T4_BRIDGE_48H.md),
[`UNIT_SIDE_FEATURE_ABLATION.md`](UNIT_SIDE_FEATURE_ABLATION.md),
[`ELECTRODE_ANCHOR_DESIGNS.md`](ELECTRODE_ANCHOR_DESIGNS.md),
[`E3_E4_ENCODER_PROGRAM.md`](E3_E4_ENCODER_PROGRAM.md),
[`MEASUREMENT_PROTOCOL_V4.md`](MEASUREMENT_PROTOCOL_V4.md), and
[`ASIC_DEPLOYMENT_CHARTER.md`](ASIC_DEPLOYMENT_CHARTER.md), plus inspected implementation in
`streaming_encoders.py` and `unit_side_features.py`.
