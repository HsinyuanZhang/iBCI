# Audit of the Coordinating Agent

**Auditor:** independent reviewer, no stake in the conclusions.
**Date:** 2026-08-14. **Compute:** CPU only, `CUDA_VISIBLE_DEVICES=` on every command, no GPU touched.
**Scope:** the coordinator's own outputs — `COORDINATOR_VERIFIED_FINDINGS.md`, `PROJECT_VISION.md`,
`README.md`, `CODE_INTERFACE.md` — plus the coordinator's accept/reject judgements on subagent claims.
**Nothing in the repository was modified.** All audit code lives in `/tmp/audit/`.

> **Note on a moving target.** `COORDINATOR_VERIFIED_FINDINGS.md` was edited at 00:10:53 while this
> audit was in progress (597 → 636 lines), adding the "F8 addendum" at lines 297–334 which credits a
> concurrent Opus auditor. My F8 probe was written and run before that addendum existed and reaches
> the same conclusions from an independent implementation — their +0.591 for template 1 against my
> +0.582 is a genuine two-implementation replication. Everything else in this audit is against the
> file as of 636 lines. Where the addendum already concedes a point, I say so.

---

## Verdict

**The measurement work is sound; the inference and the readiness claims are not.** Every quantity I
could re-derive reproduced, most of them to the last digit the document prints: F2b (11,171 / 11,619
/ 22,790 parameters, zero shared tensors), F3 (2,400 / 12,003 = 20.0%), F4, F6, F7, F13a/F13b (all
ten weight RMS values), and F14's arithmetic (−0.022145 sample-weighted, −0.004787 equal-weighted,
64%, 3.32×/3.67×). F8's direction and mechanism reproduce, and F15's trend reproduces more strongly
than the document's own evidence supports. This is not a project that invented numbers.

Three things are wrong enough to matter.

1. **F14's central caveat is factually false, and removing it damages the hypothesis the coordinator
   calls the project's most valuable output.** The date-1 receipt does expose per-recording deltas; I
   extracted them in one command. Adding those two points shows the two dates have nearly identical
   mean query length (4483 vs 4369 samples) but differ by 0.036 in mean delta, and date 1's own
   length slope under-predicts date-2's failing recording by 0.031–0.056. The query-horizon story
   survives only as a within-date effect and cannot explain the between-date failure the paper
   reports.
2. **There is a third bias in our favour, larger than either of the two the coordinator caught, and
   F15 walked past it.** The comparator's fixed `NORMALIZED_LAMBDA_FIXED = 1.0` readout costs CEBRA
   **0.19 R²** on the project's own positive control (0.8091 → 0.9979 at λ=1e-2). F15 diagnosed the
   *decoder family* and prescribed reporting kNN; the dominant term is the *untuned penalty*, and at
   a sane λ the linear readout nearly matches kNN.
3. **The primary arm fails its own required gate at the configuration the protocol declares for
   scoring.** The gate hard-codes 250 iterations; `SOURCE_MAX_ITERATIONS = 10000`. At 10,000 the arm
   scores target R² 0.6086 and `assert_positive_control` raises "numbers from this arm are void".

Two smaller structural problems: Track A was closed with **zero** independent verification in a
project whose stated discipline is "facts established by running code", and its summary in the two
most-read documents states something false about the data; and the Track B scoring path does not
exist in any usable form (the primary decoder cannot accept H1's 7-D labels at all, and no code path
evaluates a target *query* block), which makes `PROJECT_VISION.md:133` "the gating question is not
'can we build it' — that is answered" incorrect.

On the three judgement calls: **blocking Track B — right**, and for stronger reasons than the
coordinator gave. **Closing Track A — right outcome, substandard process.** **Deprioritizing CEBRA
for §12 items 1–6 — right**, and consistent with the governing handoff.

---

## Findings, ordered by severity

### 1. F14 — the date-1 check the coordinator declared impossible is possible, and it weakens the query-horizon hypothesis  · **WRONG**

`COORDINATOR_VERIFIED_FINDINGS.md:596`:

> The date-1 receipt (`H1_SE5_M4_FOLD0_TERMINAL_v1.json`) has a different structure and does **not**
> expose comparable per-recording deltas, so the pattern could not be checked on the working date

It does expose them. The keys are renamed, not absent:
`metrics.hse5_same_checkpoint_interventions.full.per_session` and
`metrics.separately_trained_zero5.per_session` each carry `{r2, samples}` per recording — the same
structure as date 2's `full_same_checkpoint_interventions.full.per_session` /
`independently_trained_zero5.per_session`. The coordinator stopped at the naming difference.

```
$ python3  # both receipts, per-recording Full − Zero5
date   recording              qsamples  calib_end      delta
date2  ses-19250108T110520        8330       3027  -0.042554
date1  ses-19250101T111740        6735       3233  +0.025902
date2  ses-19250108T111022        2508       3554  +0.001088
date2  ses-19250108T111455        2269       3456  +0.027104
date1  ses-19250101T112404        2230       3431  +0.035927
```

Date 1's pooled delta from these is +0.028469, matching the receipt margin
`hse5_minus_separately_trained_zero5 = 0.028468583` and the paper's +0.0285 exactly, so these are
the comparable quantities.

**What the two extra points do to the hypothesis.**

- Date 1's long recording is 6735 samples — 81% of date 2's failing 8330 — and its delta is
  **+0.0259, positive**. A 3.0× length ratio within date 1 costs only 0.010 of delta.
- The dates have **almost the same mean query length** (4482.5 vs 4369.0 samples) and differ by
  **0.0357** in mean delta. Length cannot produce a difference that length does not vary on.
- Extrapolating date 1's own within-date slope (−2.225e-6 delta per query sample) to 8330 samples
  predicts **+0.0136** (from the 2269 anchor) or **−0.0119** (from the 2508 anchor). Observed:
  **−0.0426**. The horizon model calibrated on the working date under-predicts the failing recording
  by **0.031–0.056**, i.e. by more than the entire effect being explained.

Rank correlation of length against delta over all five recordings is −0.90, so the *sign* of the
horizon effect is consistent. That is all that survives: a within-date monotone ordering whose
magnitude is roughly a fifth of what the date-2 failure requires. The document's framing —
`:34` "The date-2 failure is one recording holding 64% of samples, 3.3× longer than the others |
suggests query-horizon drift, the most favourable hypothesis", `PROJECT_VISION.md:113-118` "the
strongest being that the failure is a **query-horizon** effect", `PROJECT_VISION.md:149` making it
the number-one next step — is not supported once the available data are used.

**Why this is the most serious finding.** The coordinator names this hypothesis "the most favourable"
and notes it "converts the paper's most awkward negative into a demonstration of its central
advantage." It then declared the one cheap check that could disconfirm it impossible, on a false
premise, and promoted it to first priority. I do not think this was deliberate; the effect is the
same. The corroborating evidence the coordinator *did* have and did not cite is stronger than what it
used: `query_first_bin` is 3027 / 3554 / 3456 on date 2 and 3233 / 3431 on date 1, which is the real
demonstration that the calibration prefix occupies a comparable span in every recording and only the
horizon differs. That number is in the same JSON block as `target_samples`.

**Also on F14, smaller:**

- Pooling is not sample weighting. `:559-561` says "The reported figure tracks the sample-weighted
  mean". The sample-weighted mean of per-recording *Full* R² is 0.4942632 against a pooled 0.4956701,
  so the pooled figure is TSS-weighted; "tracks" is fair, "is" would not be.
- The re-weighting recommendation is uniformly self-serving and was computed on one side only.
  Equal weighting moves date 2 from −0.0226 to −0.0048 **and** date 1 from +0.0285 to +0.0309. Both
  flatter us. `PROJECT_VISION.md:124` lists this as "A number in the paper that needs re-checking"
  having computed only the favourable half and declared the other half uncomputable. Under reporting
  rule 5 the process is the violation, even though the symmetric answer also favours us.
- The remedy is already in the governing document and was not invoked. Reporting rule 2
  (`HANDOFF_COMPARATORS_20260812.md:413`) requires "per-session paired contrasts with sign counts and
  an uncertainty interval… A single mean is not enough." The paper's H1 sentence
  (`bci_paper_overleaf/paper_6pp.tex:670`) reports a pooled mean with no sign counts. "2 of 3
  recordings positive on date 2, 2 of 2 on date 1" is the compliant fix and needs no new estimator.
- `:589` attributes `s_src = 1.070213494715288` to "the same receipt family". It is the **date-1**
  value; date 2's terminal receipt records 1.07223914300781. Immaterial to the argument, but this is
  a document whose authority rests on provenance.
- "The failing **recording** is 3.3–3.7× longer" (`:574`) is true of the query block, not the
  recording. Total bins are 11,357 / 6,062 / 5,725, i.e. 1.9–2.0×.

### 2. The fixed readout penalty is a third bias in our favour, and F15 misattributed its own finding  · **WRONG (attribution) / undeclared bias**

`cebra_comparator.py:40` sets `NORMALIZED_LAMBDA_FIXED = 1.0`, applied through the sealed house ridge
to a 3–8-dimensional unit-norm CEBRA embedding. On the project's own positive control, at the gate's
own settings:

```
--- output_dimension=3, 250 iters, 240 samples ---
  house ridge normalized_lambda=1 (what the arm uses)  source R2 0.8091   target R2 0.7929
  ridge lambda=1e-2                                    source R2 0.9979   target R2 0.9945
  ridge lambda=1e-6 ~ plain least squares              source R2 0.9979   target R2 0.9950
  kNN k=3 cosine (secondary decoder)                   source R2 0.9997   target R2 0.9975
```

**The chosen λ costs CEBRA 0.19 R² on a control where the correct answer is ~1.0.** That is 2–3×
larger than the linear-versus-kNN family gap that F15 diagnoses, and it is invisible in every table
the project prints. Reporting rule 5 — "never tune a comparator where tuning helps us and leave it
fixed where it does not" — is exactly on point, and rule 1 permits selecting λ inside the calibration
block for free.

Re-running F15's grid at three readout strengths, 4 seeds per cell, means over seeds:

| samples | iters | λ=1 (protocol) | λ=1e-2 | kNN |
|---:|---:|---:|---:|---:|
| 240 | 200 | 0.795 | 0.992 | 1.000 |
| 240 | 400 | 0.730 | 0.975 | 1.000 |
| 240 | 800 | 0.693 | 0.927 | 1.000 |
| 400 | 200 | 0.794 | 0.996 | 1.000 |
| 400 | 400 | 0.730 | 0.975 | 1.000 |
| 400 | 800 | 0.678 | 0.907 | 1.000 |

Monotone decreasing in iterations: λ=1 in 8/8 seed×sample cells, λ=1e-2 in 8/8, kNN in **0/8** (kNN
rises). So **F15's trend is real and better supported than its six single-seed points suggested** —
I will not criticise the conclusion. But its diagnosis is wrong in proportion: the family choice
costs CEBRA 0.008–0.09; the untuned penalty costs 0.20–0.23. F15's prescription (`:633-636`) fixes
the iteration count and the decoder family and never mentions λ, which sits three lines above
`KNN_NEIGHBORS` in the same constants block.

The protocol's justification for the λ is a category error rather than a bias:
`TRACK_B_CEBRA_COMPARATOR_PROTOCOL.md:172` argues "Every internal comparator reports the same
linear-readout metric. Changing the head would make CEBRA incomparable to our own table." Consistency
of the *head family* does not require consistency of the *penalty*: our ridge baselines apply λ=1 to
57–176 channels of firing rates; here it is applied to 3 unit-norm coordinates. `normalized_lambda`
is defined relative to standardised features, so the same number means very different shrinkage in
the two feature spaces.

**Secondary, same theme — the readout-fitting rule is the thing that makes F8's negative control
fail, and its bias direction is never declared.** `TRACK_B_CEBRA_COMPARATOR_PROTOCOL.md:177` rules
"Where the readout is fitted. Source only", justified as "Fitting a linear head on the target prefix
would give CEBRA a second target-session estimator (the ridge) on top of the adapted encoder, which
is a different method family." Our own method *is* a frozen decoder plus a closed-form target-session
estimator, so "different method family" is not the real objection; the real objection is that CEBRA
would then get two target-session fits to our one. That is arguable, but it means `ARM_BIAS`'s
description of `cebra_adapt_unaligned` as "Official adapt=True" (`cebra_comparator.py:94`) is a
mislabel: official CEBRA fits its decoder on labelled data of the session being decoded. My oracle
measurement below shows that arm scores 0.64–0.82 under CEBRA's own decoding convention. Every arm
carries a bias declaration; the global readout rule that determines the headline verdict carries
none.

### 3. The primary arm fails its own required gate at the declared scoring configuration  · **WRONG (unnoticed defect)**

`run_positive_control_gate` defaults to `max_iterations=250` (`cebra_comparator.py:131`,
`CODE_INTERFACE.md:61`). The declared scoring budget is `SOURCE_MAX_ITERATIONS = 10000`
(`cebra_comparator.py:35`, `CODE_INTERFACE.md:143`). Running the repo's own gate functions across
that range:

```
gate default:  iterations=250  output_dimension=3  threshold=0.70
declared scoring: SOURCE_MAX_ITERATIONS=10000  OUTPUT_DIMENSION=8

iters=   250  source_r2=+0.7528  target_r2=+0.7502  gate=PASS
iters=  1000  source_r2=+0.6630  target_r2=+0.6621  gate=FAIL -> PositiveControlError: ... target R2=0.662 < 0.7; numbers from this arm are void
iters=  2500  source_r2=+0.6266  target_r2=+0.6258  gate=FAIL
iters= 10000  source_r2=+0.6079  target_r2=+0.6086  gate=FAIL -> PositiveControlError: ... numbers from this arm are void
```

The gate passes only because it is run 40× cheaper than the configuration it is meant to authorise.
This is a direct consequence of F15's own finding — R² falls monotonically with iterations — which
the coordinator documented without checking the gate against the constant. The gate also runs at
`POSITIVE_CONTROL_OUTPUT_DIMENSION = 3` while `OUTPUT_DIMENSION = 8`; that difference alone moves the
house-ridge source R² from 0.8091 to 0.9247, so the gate is not validating the scoring geometry
either. `CODE_INTERFACE.md:56-67` presents the gate as the thing that must pass "before any number is
interpreted" and does not disclose either mismatch.

**Related: the gate's teeth are seed-dependent.** Re-running the repo's own arm code on 8 seeds:

| arm | target R² range | verdict |
|---|---|---|
| `cebra_joint_behavior` | +0.7408 … +0.8227 | 8/8 pass ≥0.70 |
| `cebra_frozen_source_adapt` | +0.7244 … +0.8113 | 8/8 pass, min 0.724 |
| `cebra_adapt_unaligned` (must be <0.20) | **−0.3400 … +0.5579** | **6/8 below 0.20** |

On seeds 3 and 5 the declared negative control scores +0.520 and +0.558, and
`assert_positive_control` would raise `CebraComparatorError: negative control … unexpectedly
recovered the latent`, failing the whole gate. `CODE_INTERFACE.md:66-67` — "Asserting both directions
is what gives the gate teeth" — is true of one seed, not of the construction.

### 4. Track A was closed with no independent verification, and its summary states something false  · **WRONG**

`COORDINATOR_VERIFIED_FINDINGS.md` contains **no** finding verifying any Track A claim. The only
mentions of Track A are `:63` (network reliability) and `:187` (Track A catching the brief's RT
error). In a file whose first line is "Facts established by the coordinating agent by **running
code**, not by reading it", a whole track was closed on the document alone.

The compression then introduced an error. `PROJECT_VISION.md:92-93` and `README.md:37`:

> The rat hippocampus data has four subjects with differing unit counts but **no direction or
> velocity basis**.

The vendored loader docstring (`third_party/cebra/cebra/datasets/hippocampus.py:92-96`) says the
label "is position and the running direction (left, right) of a rat. The behavior label is structured
as 3D array consists of position, right, and left." Track A itself got this right —
`TRACK_A_DATASET_AUDIT.md:131` records "Labels are `position` ∈ R³ = `[position_m, right, left]`" and
rests its verdict on the correct grounds (1-D track, no 8-direction target, φ = [cos θ, sin θ]
inapplicable, wrong species/area/encoding, four animals rather than multiple implant days). The
coordinator's summary replaced a careful argument with a false one, in the two documents a reader
actually reads. "No basis exists" closes the door permanently; "the basis is a binary heading on a
1-D track" leaves a decision that could be revisited.

I did verify the load-bearing Area2_Bump claim independently and Track A is correct:
`monkey_reaching.py:132-139` selects on `ctr_hold_bump`, so `session` ∈ {active, passive, all} is
trial type, not a recording day.

### 5. The scoring path does not exist, so "can we build it — that is answered" is wrong  · **WRONG (readiness claim)**

`PROJECT_VISION.md:133`: "The gating question is not 'can we build it' — that is answered".

- **The primary decoder cannot score H1 at all.** `fit_linear_readout` → sealed `fit_ridge`, which
  requires `y.shape[1] == 2` (`sua_exploration/mc_maze/subm_v9_f0_pv_ridge.py:282`). H1 is 7-DoF.
  ```
  7-D ridge FAILS: ClassicalControlError ridge feature/target shape drift
  2-D ridge OK RidgeReadout
  ```
  `falcon_h1` is a bound dataset with `PRIMARY_DECODER = "linear_ridge"` and no test covers it.
- **No code path evaluates a target query block.** In `run_arm_on_synthetic_fold` the target
  embedding comes from `target_x` (`cebra_comparator.py:1258`, `:1290`, `:1319`) and the score is
  taken against `target_y = ys[target_index]` (`:1190`) — the same rows that entered the fit for the
  joint and frozen arms. `target_query_neural` is accepted (`:1180`) and used only to raise for one
  undefined arm (`:1198`); it is otherwise dead. Whatever the intent, the object that turns an
  embedding into a held-out target-session number is not written. This particular gap biases
  *towards* CEBRA, so it is not a fairness problem — it is a readiness problem, and it is the one
  thing the paper would need.
- **`assert_matched_budget` is wired to nothing.** Its only callers are four lines in
  `tests/test_cebra_comparator.py:43-48`. `CODE_INTERFACE.md:93-104` presents it under "Leak and
  budget guards" as if it guarded a path.
- **The integrity gate is unsatisfiable as configured.** `INTEGRITY_ATOL = 1e-5`, but 6 of the 10
  entries in `SEALED_REFERENCES` are stored to 4 decimal places (`carrier_sua: 0.3568`,
  `dense_ridge_sua: 0.4179`, `carrier: 0.5000`, …). A correct reproduction lands within 1e-5 of a
  4-dp rounded value only by luck. The gate has never run (`results/` is empty), so nobody found out.

`README.md:38` "Skeleton complete and gated" overstates this. `PROJECT_VISION.md:129-130` — "Part A
sample counts are unmeasured, the integrity gate is wired but never executed, and no scoring arm has
touched subject-M, RT, H1 or M2" — is honest and should have been the headline instead.

### 6. F8 reproduces in direction; the magnitude is overstated and the receipt does not exist  · **OVERSTATED / unverifiable as documented**

**What reproduces.** Across 20 configurations (architectures `offset1`/`offset10`, source iterations
250–5000, adapt iterations 250–5000, output dimensions 3 and 8, both templates):

- the joint fit's target R² tracks its source R² to **|gap| ≤ 0.009 in every configuration** — F15's
  ±0.004 claim, and the right diagnostic;
- the adapt path's target R² under a source-fitted readout never approaches the joint fit;
- **more adapt compute makes it worse, not better** (−0.651 at 5000 adapt iterations on `offset1`,
  −1.005 at 2000 on `offset10`), so the coordinator's rejection of the "insufficient adapt
  iterations" explanation is correct;
- **the stated mechanism is right, and I can show it more directly than the document does.**
  Refitting the readout on the target embedding (an oracle the protocol forbids, used here purely as
  a diagnostic) gives **0.64–0.82** — the same quality as the joint fit — while the best linear map
  from the adapted target embedding onto the source embedding reaches **0.44–0.999**. The adapted
  session is a good embedding in the wrong frame, exactly as claimed. Embedding norms are 1.000 in
  every configuration, consistent with the hypersphere argument.

**What does not reproduce: the numbers.** F8's table (`:258-262`) reports 0.9891 / 0.9924 source and
−1.2278 / +0.9940 target. Using the protocol's readout I get source R² 0.60–0.82 at d=3 and 0.92–0.95
at d=8, nowhere near 0.99. The cause is the readout, not CEBRA: at λ=1e-2 the same joint fit gives
**0.9979 source / 0.9945 target**, which is F8's row 3 to three decimals. So **F8 was measured with a
near-unregularised readout that the built comparator does not use**, and F15 then juxtaposes the two
(`:615-617`, "against the broken arm's 0.9891 source / −1.2278 target") without noting that the
0.71–0.81 column and the 0.9891 column come from different decoders. F8 states its construction but
not its architecture, output dimension, iteration counts or readout, and the script that produced it
is not on disk (`scripts/` holds only `probe_cebra_dataset_shapes.py` and the runner; `results/` is
empty). Against `:5` — "Everything here is reproducible with the recipe in `scripts/cebra_env.sh`" —
F8 is the one finding that is not.

**On overstatement.** `:270` "Negative R² is the expected outcome, not a tuning failure" and `:252`
"structurally broken" describe a distribution whose mean over 8 seeds is **+0.0098**, with 2 of 8
seeds at +0.52 and +0.56 and one adapt setting at +0.700. −1.2278 is an extreme draw quoted as
characteristic in four places (`:260`, `PROJECT_VISION.md:97`, `third_party/CEBRA_PROVENANCE.txt`,
`TRACK_B_CEBRA_COMPARATOR_PROTOCOL.md`). The addendum at `:322-324` now concedes exactly this and
the concession is correct; the four downstream copies still say −1.23. The accurate statement is that
the adapt path lands the target in an arbitrarily oriented copy of the source space, so a
source-fitted readout has an expected R² near zero with a spread of roughly ±0.6 — which supports
`:276` ("every dataset would have returned ~0") better than the quoted number does.

**Is the synthetic control rigged to fail?** No. It is *easier* than real data in every respect that
matters: one shared latent, identical labels across sessions, exact sample-to-sample correspondence,
light noise. Nothing in it disadvantages `adapt` specifically, and the oracle measurement proves the
adapt run learned a good embedding. The conclusion transfers.

### 7. F13 — the measurements are exact; "the network did compensate" is an over-read and the 7–12× interval should be withdrawn  · **OVERSTATED**

All ten numbers reproduce to the digit printed, from the four checkpoints on disk:

```
date1_Full   c1=0.23402 c2=0.07264 c3=0.08762 c4=0.32864 c5=0.03253 | carrier 0.18803 activity 0.11406
date2_Full   c1=0.21362 c2=0.03533 c3=0.03715 c4=0.20790 c5=0.03184 | carrier 0.13601 activity 0.11822
date1_Zero / date2_Zero5: all carrier columns exactly 0.00000
```

Shape (32, 37) on all four; F13a's checkpoint-existence correction against C2 is right (I count 2,717
`.ckpt` files against F13a's 2,716 — a file appeared since).

**"The network did *not* passively accept the input imbalance" (`:498`) does not follow.** The carrier
columns are zero-initialised (`h1_sparse_event_spint.py:49`, which F11 cites correctly) and trained
with Adam at `weight_decay: 0.0`. Adam normalises the per-parameter step, so a column's final
magnitude measures how *consistently useful* its gradient direction was, not how much the optimiser
pushed back against its input scale. "The intercept ended smallest" is equally consistent with "the
intercept was least useful" — a different claim with different implications for S1. The coordinator
makes the Adam step-size point itself in F11 (`:400-403`) and then reasons in the opposite direction
in F13b. The downstream conclusion "S1 weakens" rests on this inference.

**The 7–12× composition should be withdrawn as a number, not caveated.** Recomputing it from C2's
per-coordinate input RMS (`TRACK_C_BRAINSTORM_OPUS.md:42-46`) paired coordinate-by-coordinate with my
weight measurements:

| | c1 | c2 | c3 | c4 | intercept | intercept ÷ max tuning | intercept ÷ min tuning |
|---|---:|---:|---:|---:|---:|---:|---:|
| date 1 | 0.00796 | 0.00130 | 0.00157 | 0.00937 | 0.07272 | **7.8×** | **55.9×** |
| date 2 | 0.00726 | 0.00063 | 0.00066 | 0.00593 | 0.07118 | **9.8×** | **112.6×** |

The interval "7–12×" (`:508`) is recoverable only by taking the two largest of four tuning
coordinates and pooling two dates into one range. The four discarded products are 4–15× smaller, and
on date 2 the weakest coordinate's residual imbalance is **112.6×** — larger than anything C2
claimed. The published caveat says only that the composition "mixes C2's input measurement with my
weight measurement"; the selection and the cross-date pooling are not disclosed. The defensible
version keeps the direction and drops the interval: the trained weights partially offset the input
imbalance, leaving roughly one order of magnitude on the strongest tuning coordinates and two on the
weakest.

**F11 mislabels the units it is citing.** `:397` says the intercept arrives "with 32–61× the RMS of
the four signed-tuning coordinates". C2's 32–61× is the **standard-deviation** ratio; the RMS ratio
is 66–125× (2.2355 ÷ {0.0340, 0.0179, 0.0179, 0.0285}), and C2 states both
(`TRACK_C_BRAINSTORM_OPUS.md:51-52`). F13b then corrects "7–12×, not 32–61×" against the mislabelled
figure.

**Was overturning the two brainstorms' convergence justified?** Partly. Insisting on a measurement
over a discussion was right, and the measurement is clean. But the conclusion is stated far more
firmly than a zero-initialised weight snapshot supports, and the coordinator's own replacement
experiment (`:512-517`, an end-to-end effective-contribution probe on all four checkpoints) is
essentially S1's diagnostic done properly. The practical effect of the overturn is to rename the
experiment the two agents converged on, while recording in prose that their idea "weakens". The
caveats at `:503-509` are honest and well placed; the index line and `PROJECT_VISION.md` are not
where a reader would learn that the downgrade rests on an ambiguous inference.

### 8. Smaller inaccuracies in `CODE_INTERFACE.md` and the findings file  · mixed

| Location | Problem |
|---|---|
| `COORDINATOR_VERIFIED_FINDINGS.md:130` | F3 cites `cebra.py:993-1012` for the adapt state-dict split. The local patch shifted it; those lines are now the patched optimizer/solver block. The code is at 1049–1068. |
| `:149` | F4 cites `cebra.py:981-984` for the multisession-adapt guard. That guard is now at 1035–1038; 981–984 is the patch's own `freeze_sessions` error. Both citations were taken pre-patch and now mislead a reviewer who follows them. |
| `:222-225` | F7 presents a verbatim block ending `raise ValueError(...)`. The source raises `NotImplementedError` (`metrics.py:351`). I confirmed the behaviour empirically — 7-D labels are rejected — so the finding is right and the quotation is wrong. |
| `CODE_INTERFACE.md:132-135` | Describes receipts in the present tense ("Receipts carry input SHAs…"). `results/` is empty; no receipt has ever been written. |
| `CODE_INTERFACE.md:29` | `--authorise-scoring` is described as "Deliberately inert"; it raises `SystemExit`. Harmless, but it is not inert. |
| `run_cebra_comparator.py:309-311` | `--audit-only` without `--dataset` silently runs the dry run instead of erroring, because the `args.dataset is None` branch is tested first. |
| `cebra_comparator.py:455` | `AdaptationCost.target_parameters_updated` stores the number of trainable **tensors**, not parameters: measured 24 / 8 / 2 for joint / frozen / adapt against parameter counts 7,945 / 2,851 / 1,216. This field feeds the receipt that would populate the paper's cost-of-no-backprop table. |
| `cebra_comparator.py:1282` vs `:1253` | `cebra_frozen_source_adapt` trains its target encoder for `max_adapt_iterations` (500) while `cebra_joint_behavior` trains everything for `max_source_iterations` (10000) — a 20× asymmetry between two arms that are compared to each other. `ARM_BIAS` declares only "source cannot move". Given F15, the sign of this handicap is not even obvious, which is precisely why it should be declared. |
| `ARM_BIAS["cebra_no_adapt"]` | "Not a handicapped CEBRA." On H1 and M2, where N is fixed, `no_adapt` is the *only* arm the Part A prior marks `CEBRA_DEFINABLE`, and it consumes **zero** target labels while our carrier consumes four trials. As the arm most likely to run first on H1, calling it unhandicapped invites exactly the misreading rule 4 exists to prevent. |
| Budget arms | `assert_matched_budget` accepts a `requested` override, so a declared upper-bound arm ("CEBRA at 4× our supervision") is possible and is not declared anywhere. Reporting rule 4 argues for it. `underdetermined_input_layer` is computed and surfaced but wired to no verdict, so a CEBRA arm starved by the matched budget would report a low number attributed to CEBRA. |

### 9. F2b's inference is right; one clause of it overreaches  · **OVERSTATED (mild)**

The measurement is exact and the "sharpens rather than weakens" reading is correct: with zero shared
tensors and zero shared parameter objects, the honest contrast really is "a full encoder per session
aligned by a loss" against "one shared decoder plus a cached descriptor". I checked the stronger form
too — no parameter is shared by reference, and the criterion is `FixedCosineInfoNCE` with zero
parameters, so there is genuinely nothing tied.

The clause that overreaches is "parameter growth per session: linear" set against "zero" in a table
row about **cost of a new session** (`PROJECT_VISION.md:41-42`, `README.md:30-31`). At deployment
CEBRA needs one encoder for the live session; the linear growth is a training-time and model-zoo
property. Our own per-session cached `E_i` also grows with the unit count. The defensible claim is
that no parameter receives a **gradient** on the target session and no new trainable parameters are
created — which is what F2b's body says and what the table row blurs.

One coherence point: F3's index line still reads "the structural analogue of our identity token"
with no cross-reference to F8. F3 describes the single-session `adapt` path, which this project has
since declared a negative control that "must fail". A reader scanning the index takes away that
CEBRA's per-session input layer is our analogue; the project's own conclusion is that the analogue
lives on a path it will not report.

### 10. The briefs, and where the near-miss actually came from  · minor

The brief carried two load-bearing errors: the RT path (F6, caught by Track A) and the mechanism of
multi-session CEBRA (F2b, caught after Track C1 questioned it). The second one *caused* F8: a design
built on "session-specific input layers over a shared trunk" leads directly to
source-fit → collapse → `adapt`. `:78-80` acknowledges the brief was wrong and the index calls F2b
the "root cause of F8". But `PROJECT_VISION.md:96` frames the episode as "Track B: the first design
would have produced a false conclusion", which puts the near-miss on the subagent rather than on the
brief that specified it. The three-track split itself was sound — the tracks share no state and the
two-model Track C did produce checkable disagreements, which is the strongest structural decision in
the project.

---

## What I reproduced, and what I could not

| Finding | Status | Evidence |
|---|---|---|
| F1 environment | partial | `torch 2.5.1.post303`, `numpy 2.0.1`, `cebra 0.6.1` from the vendored tree, 22/22 tests pass in 5.6 s. The CUDA half is unverifiable here by instruction. |
| F2 multi-session across mismatched N | reproduced | joint fit on 24 + 31 units, both transform to a shared space |
| **F2b zero weight sharing** | **exact** | 11,171 / 11,619 / 22,790 params; identical tensors `[]`; shared parameter objects 0 |
| F3 adapt touches layer 1 only | **exact** | changed tensors `['net.0.weight','net.0.bias']`, 2 of 10; 2,400/12,003 = 20.0% |
| F4 adapt blocked after multi-session | reproduced | `NotImplementedError: The adapt option with a multisession training is not handled` |
| F5 proxy unreliability | not tested | no network work attempted |
| F6 RT is `sub-C` | **exact** | sub-C 68/53/15, sub-M 22/22/0, sub-J 3/3/0; `EXPECTED_FOLDS = 15`; `000129/sub-Indy` has 2 NWBs |
| F7 consistency is 1-D-only | reproduced, quote wrong | 7-D labels rejected; exception is `NotImplementedError`, not `ValueError` |
| **F8 direction + mechanism** | **reproduced, strengthened** | 20 configs; joint gap ≤0.009; oracle 0.64–0.82; linear-map-to-source 0.44–0.999; 5000 adapt iterations makes it worse |
| **F8 specific numbers** | **not reproduced** | 0.9891/0.9924/+0.9940 require λ≈1e-2 (I get 0.9979/0.9945); the protocol readout gives 0.60–0.82. Probe script absent, receipt absent |
| F9 collapse defect | unverifiable | the function was deleted; `test_collapse_helper_was_deleted` enforces its absence |
| F10 RT discovery | reproduced | `discover_rt_sessions` returns 15 `ses-RT-*`; refuses `000129`/`sub-Indy` |
| F11 structural asymmetry | reproduced, one mislabel | `carrier_dim` 4 vs 5; injection lines 42/49/65/67 exact; "32–61× the RMS" is the std ratio |
| F12 V1/V2 estimator split | not re-derived | out of scope; no claim made either way |
| **F13a/F13b weights** | **exact** | all ten RMS values; both zero arms exactly 0.00000; shape (32,37) |
| F13b "network compensated" | over-read | zero-init + Adam confounds compensation with usefulness |
| F13b "7–12×" | **selective** | 7.8×/9.8× on the two strongest coordinates; 55.9×/112.6× on the weakest |
| **F14 arithmetic and receipt** | **exact** | −0.022145 / −0.004787 / 63.55% / 3.32× and 3.67×; receipt names all three recordings |
| **F14 "date 1 cannot be checked"** | **false** | date-1 per-recording deltas extracted: +0.025902 (6735 samples), +0.035927 (2230) |
| **F14 query-horizon hypothesis** | **weakened** | equal mean lengths across dates (4483 vs 4369), 0.036 delta gap; date-1 slope under-predicts by 0.031–0.056 |
| **F15 monotone decline** | **reproduced, stronger** | 8/8 seed×sample cells monotone at λ=1 and at λ=1e-2; kNN 0/8 |
| F15 attribution to decoder family | wrong proportion | λ costs 0.20–0.23; family costs 0.008–0.09 |
| Positive-control gate | passes at its own settings, fails at declared ones | 250 → PASS 0.7502; 1000/2500/10000 → PositiveControlError |
| Gate negative control | seed-dependent | 6/8 seeds below 0.20; seeds 3 and 5 at +0.520/+0.558 would raise |
| Primary decoder on H1 | broken | `ClassicalControlError: ridge feature/target shape drift` on 7-D labels |
| Track A Area2_Bump verdict | reproduced | `monkey_reaching.py:132-139` selects on `ctr_hold_bump` |
| Track A hippocampus summary | false as summarised | loader label is `[position, right, left]` |

**Not attempted:** any GPU work; any real-NWB scoring run; F12's estimator recomputation; the CUDA
claim in F1; network/proxy behaviour in F5; the H1 held-out split (respected the held-in-only rule —
the only H1 artifacts I opened are the two terminal receipts and the four held-in checkpoints).

**Audit code:** `/tmp/audit/{f8_probe,f8_fast,f2b_probe,f13_probe,f15_probe,f15_lambda,readout_probe,gate_seeds,gate_at_scoring_config}.py`,
run as `CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 $CEBRA_PY -s <script>`.

---

## The three judgement calls

**Blocking Track B's arms — correct, and under-justified.** The coordinator blocked on F8 and F15.
The stronger reasons are the ones it did not find: the primary arm fails its own gate at the declared
iteration count, the primary decoder cannot accept H1's label dimension, no code path scores a target
query block, the readout penalty costs CEBRA 0.19 R², and the integrity gate is numerically
unsatisfiable. Had the arms been authorised on the strength of the passing gate, the resulting numbers
would have been void for at least four independent reasons.

**Closing Track A — right outcome, process below the project's own standard.** No CEBRA dataset can
exercise cross-session calibration on a motor task; Track A argues this carefully and its Area2_Bump
claim verifies. But the coordinator produced no verification of its own for an entire track, and its
summary of the one dataset that came closest is false in a way that forecloses a decision on
incorrect grounds. The cheap fix is to restate the hippocampus verdict in Track A's own terms in
`README.md` and `PROJECT_VISION.md`.

**Deprioritizing CEBRA in favour of `HANDOFF_COMPARATORS_20260812.md` §12 items 1–6 — correct.** I
checked the citation: items 1–6 are indeed zero-compute or writing, and §12:444 already declares them
"the highest-value remaining work on the paper". The reasoning at `PROJECT_VISION.md:136-141` is the
best argument in the document: the cost-of-no-backprop measurement is cleaner on our own architecture
than through another implementation's confounds, and post-F8 the cost axis is largely settled in
advance. Given the state of the Track B code, this call is if anything conservative.

**Overturning the two brainstorms' convergence on S1 — the right instinct, the wrong confidence.**
Measuring rather than debating was correct and the measurement is exact. But the inference from
zero-initialised weights under Adam does not distinguish compensation from usefulness, the 7–12×
interval is selectively constructed, and the replacement experiment is S1's own diagnostic under a
new name. This is not over-weighting one's own evidence against two agents — it is over-reading one's
own evidence.

---

## What would change my mind, in order of cost

1. **Recompute F14 with the date-1 per-recording deltas included** (five minutes, no compute). If the
   query-horizon hypothesis survives a five-point fit with a date effect, I withdraw finding 1. On
   the numbers above I do not expect it to.
2. **Rerun the positive-control gate at `SOURCE_MAX_ITERATIONS` and `OUTPUT_DIMENSION`** and either
   fix the constants or state that the gate authorises a different configuration than it validates.
3. **Select the readout λ on source data** and re-run F15's grid. If the linear arm then tracks kNN,
   F15's protocol change reduces to "report both" and the fairness risk mostly disappears.
4. **Restate F8's table with the readout it was measured under**, or re-measure it with the
   protocol's readout, and reduce −1.2278 to a range in all four documents that quote it.
5. **Replace the "7–12×" interval** with the eight paired products above.
6. Before any real-data run: give the sealed ridge a d>2 path or declare H1 out of scope for the
   linear arm; implement a query-block evaluation; wire `assert_matched_budget`; store
   `SEALED_REFERENCES` at full precision or widen `INTEGRITY_ATOL`; rename
   `target_parameters_updated`.
