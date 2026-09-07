# Fixed-K temporal prototype — cross-domain replication audit

**Audited:** 2026-08-02 (Asia/Hong_Kong)  
**Scope:** read-only review of existing source layouts, manifests, result receipts, and implementation contracts. No NWB was opened by this audit; no GPU, decoder, local held-out replay, formal SUA test, EvalAI action, or shared-code change is authorized here.

**Terminal update:** the subsequently executed M1 Gate A2 failed at its P20-versus-B20 content
gate. Therefore the P20 M2/SUA sequence proposed below is not executed. B20 is a new hypothesis
and requires a separate cross-domain protocol; it must not inherit this P20 authorization.

## Verdict

The M1 result justifies one new, independently specified **CPU-only temporal-order carrier** experiment, but not a decoder experiment. It does not justify repeating the M1 slot-shuffle gate, tuning its null after the result, or reviving the closed M1 representation branch.

Two meanings of “best replication” must remain separate:

| Question | Best location | Why |
|---|---|---|
| Lowest incremental-cost cross-domain check | Native FALCON **M2 held-in source only** | Seven source sessions, fixed 96 native-MUA channels, existing raw-bin trialization, and chronological support/query machinery. No decoder or new ingestion regime is needed. |
| Strongest independent generalization check | Sorted SUA **sub-C CO, 27 source + 6 target-free development sessions** | Twenty-seven independent source sessions plus six sessions unused for this candidate; variable session-local unit counts directly test the intended permutation-equivariant claim. |

Recommended sequence: **M2 source-only CPU Gate A first; SUA source-train plus target-free development Gate A only if M2 is positive.** The six M2 files named `held-out-calib` have already been viewed by T4/K4/SSC local development workflows. They are not an unseen confirmation set and are not opened by either Gate A.

## 1. What transfers from the M1 result

The completed M1 source-LOSO proxy reported `prototype - rate_only = +0.100526` in 4/4 sessions, CI95 `[+0.090798,+0.110253]`, and MDE80 `0.012719`. That is useful evidence that compact temporal support structure may contain information beyond support mean, variance, and exposure.

It did not establish coherent cross-session slot semantics. The predeclared session-keyed slot-shuffle contrast was positive by sign, but catastrophic extrapolation in two sessions produced CI95 crossing zero and MDE80 `108.863822`. The frozen M1 decision is therefore `stop_cpu_gate_not_met`; M1 decoder, GPU, held-out, K/r/router search, and quantization remain prohibited. See [M1 fixed-K result audit](M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A_RESULT_AUDIT.md).

The new cross-domain question must be narrower and predeclared:

> Does preserving chronological temporal order in a compact, label-free support carrier predict a later, label-free neural temporal profile beyond a dimension-matched rate/exposure carrier?

The primary content null should be a coherent **within-trial time-order null**, not another post-hoc slot-shuffle rescue. For every support trial, one deterministic non-identity permutation of valid bins is shared by all channels/units in that trial. Apply it to all source and application sessions before EWMA routing, and fit transformed anchors from transformed outer-train support only. This preserves each unit's count marginal, exposure, trial boundary, and instantaneous population-bin vector, while removing temporal adjacency used by the causal EWMA. It has width 20 just like prototype and rate-only carriers.

This is a new temporal-order experiment, not a direct re-analysis of the failed slot-coordinate mechanism. M1 cannot choose its seed, permutation family, K/r/alpha values, support length, target horizon, or acceptance margin after data are seen.

## 2. SUA: 27 source + six target-free development sessions

### 2.1 Feasibility and correct meaning of target-free

Yes. The existing cross-budget receipt already validates the chronological `27 train / 6 validation / 6 sealed formal-test` sub-C CO regime, and opened only train plus development validation. It also demonstrates that a source-fitted object can be applied to the six development sessions without using their teacher target until after prediction.

For this carrier, target-free application must be stricter:

- carrier, anchors, normalization, and ridge receive only the first `M_support` neural trials and trial boundaries;
- object ID, target direction, velocity, reward, electrode ID, waveform/SNR, and later neural bins are forbidden from deployed objects;
- later neural data are scorer-only, used after the application carrier is sealed to form a predeclared **label-free later-neural temporal-profile target**;
- the six formal SUA test names remain sealed and paths unresolved.

This is neural representation stability evidence, not behavior decoding or frozen-decoder benefit. A condition-labelled future-rate oracle would be a separately named diagnostic and cannot be mixed into this cross-domain primary gate.

### 2.2 Variable units and no cross-session unit identity

Variable `N` and absent unit correspondence are legitimate if the model is a shared **per-unit row map**. Fixed anchors map a session-local unit's temporal trace to a 20-D carrier; a source-only shared ridge maps its 20-D row to its own fixed-width later neural target row. Concatenating source rows needs no assertion that unit row `i` is the same physical neuron in another session.

The licensed claim is limited to a session-independent mapping from a session-local unit's support statistics to that same unit's later neural profile. It is not physical-neuron tracking, electrode matching, or a population decoder claim. A within-session unit-row permutation must leave carrier, source readout fit, and score invariant.

There is one required practical correction. SUA sessions have variable unit counts (about 38--91 in the current regime) and valid-bin exposure. Literal concatenation weights large sessions more heavily in anchor fitting and ridge loss. That is not invalid, but it changes the estimand. The protocol must predeclare one policy:

1. **Recommended:** deterministic, source-only equal-session cap/subsample of temporal rows for anchor fit, equal session weight in ridge loss, and equal-session validation aggregation.
2. Exposure-weighted: retain all rows/bins, state the row/bin-weighted estimand, and use exactly the same weighting for all arms.

The first aligns better with a six-session deployment claim. Its cap and seed must be frozen from source-only feasibility metadata, not selected using six development scores.

### 2.3 SUA boundaries

M1 indices cannot be copied blindly. SUA's established regime has activity calibration of 30 trials and common behavior evaluation after trial 50. The initial neural-only carrier should freeze:

```text
support carrier:       first 10 chronological eligible neural trials, no labels
source training:       all 27 source sessions; nested source-LOSO diagnostic
development application:first 10 eligible neural trials from each of six dev sessions
later target horizon:  fixed post-50 neural-only trial interval, scorer-only
```

`M_support=10` preserves the M1 small-calibration hypothesis. It must not become 30 simply because legacy B3 activity state uses 30. If a later decoder experiment is ever earned, its 10-trial prototype state and existing 30-trial B3 state must be separately accounted; this audit does not authorize that integration.

The target should be a fixed-width label-free per-unit temporal-profile statistic, for example a post-50 rate summary plus frozen-anchor temporal distribution. It is built only after support finalization. The mandatory rate-only control prevents marginal firing-rate persistence from being mistaken for temporal structure.

## 3. M2: seven source sessions and six local held-out files

### 3.1 Feasibility

M2 is technically the cheapest cross-domain falsification. There are seven held-in-calibration source NWBs and six held-out-calibration NWBs; existing reliability artifacts document 96 native-MUA channels. The raw native path preserves trial boundaries and raw 20-ms counts. Finger velocity is explicitly forbidden for this neural-only carrier.

The M2 gate opens only exact hashed held-in source paths; it must not invoke a broad datamodule setup that can enumerate held-out files. A small exact-manifest reader analogous to the corrected M1 loader is required. Fixed width reduces implementation risk, but channel-permutation equivariance remains mandatory: do not assume channel `i` is physically comparable across sessions.

Freeze the M2 receipt as:

```text
support carrier: trials [0,24), raw 20-ms neural counts, no behavior labels
source target:    fixed later raw neural interval from trial 24 onward, scorer-only
```

M24 is required: it is the existing all-six-session chronological support budget. M33 would make two local held-out files zero-query and creates an unnecessary support-budget degree of freedom. The source-only gate never opens the six held-out files; the all-six feasibility fact only fixes M24 before a result.

Required primary arms are: ordered prototype, width-20 rate/exposure/variance control, and width-20 time-order-null prototype. An optional zero/mean-only carrier is descriptive only. T4, D4, and K4 are not mechanism controls here: T4 uses direction labels and K4 uses dense velocity, so those contrasts are label-information-confounded.

### 3.2 Why M2 local held-out is not a new confirmation

The M2 `held-out-calib` files have already appeared in local T4/K4/SSC development replays. The completed endpoint audit explicitly seals failed candidate branches against post-hoc reopening. A new prototype must not read these files during source Gate A, cannot select any configuration from them, and cannot describe a later replay as independent formal generalization.

If a new candidate is someday frozen source-only, one local replay would at most be a **previously viewed local development precision check**. Official FALCON/EvalAI is the only external M2 endpoint, and proxy success alone does not authorize it.

## 4. Shared Gate A and serial decision

Freeze the common carrier family before implementation:

```text
K=4; rank=4; causal EWMA alpha=(0.5, .25, .125, .0625)
hard nearest ordered anchors; width=20; ridge=1
explicit trial reset; no support labels, dense behavior, unit-ID table, or query input
```

For both domains report source-only outer-LOSO scores, `prototype-rate_only`, `prototype-time_order_null`, equal-session CI/MDE/positive-session count, complementary Binomial split repeatability, unit-permutation invariance, target/support disjointness, raw/padding checks, label-access receipt, and state/MAC accounting. Resampling seeds are never biological samples.

```text
M1 A2/E4 and M1 DLA: closed ineffective; do not reopen
    |
M2 source-only CPU temporal-order Gate A (7 held-in sessions, M=24)
    |-- fail any content/repeatability/provenance gate -> stop cross-domain branch
    '-- pass -> SUA source-only Gate A (27 sessions, M=10)
                  -> one target-free development application (6 sessions)
                     |-- fail -> stop; no decoder/GPU
                     '-- pass -> separate root review of zero-init decoder compatibility
                                  -> separate GPU selection protocol if approved
                                     -> freeze one candidate
                                        -> one formal endpoint, never for rescue/selection
```

M1 D4, E4/A2, and decoder-latent alignment are closed historical routes. A prototype proxy cannot reopen them or justify a post-hoc M1 decoder variant.

## 5. Kill criteria

The numerical practical margin must be frozen in the later prelaunch receipt; it must not inherit M1's `+0.100526` proxy effect or the unrelated M2 K4 decoder `+0.03` margin. Regardless of that chosen SESOI, stop if any condition below is met:

1. Any label access, target/support overlap, application-session anchor/normalization/readout refit, non-finite carrier, non-identity-null failure, raw-trial/padding failure, or unit-permutation failure invalidates the run rather than relaxing a check.
2. Either count-distribution or prototype-value split-half lower 2.5% quantile is below 0.5, or fewer than 90% of rows are defined in a fixed resample.
3. Prototype fails to have positive paired CI95 lower bound and mean at least observed MDE80 against **both** rate-only and time-order-null.
4. After positive M2, either SUA source-LOSO or six-session target-free development application fails the same two-control criterion.
5. Positive mean with CI crossing zero or mean below MDE is indeterminate/negative for escalation, not a reason to add seeds, refold the same sessions, remove outliers, or widen the carrier.

A pass only establishes later-neural proxy information. Decoder work requires a new zero-init, matched-attachment/null, frozen-state, fixed-seed/epoch protocol; it is never automatic.

## 6. Prohibited before all gates pass

- K/r/EWMA/ridge/support-budget/horizon sweeps; outcome-driven M24/M33 selection; a new slot-shuffle family to make M1 pass.
- Dense M2 velocity; target direction; SUA object ID/reward; waveform/SNR/electrode/static spatial input; T4/D4/K4 fusion; autoencoder; contrastive objective; FiLM; cross-attention; dynamic weight generation.
- M2 local held-out replay, official EvalAI submission, SUA formal-test read, or quantization based on a CPU proxy result.
- Any M1 A2/E4/D4/DLA revival, new M1 report split, or prior official M1 score used to select carrier settings.
- Treating variable-unit SUA rows as matched physical neurons or silently treating row/bin multiplicity as additional independent sessions.

## Recommendation

Authorize only a reviewed prelaunch implementation for the **M2 seven-source, M24, label-free temporal-order Gate A**. It is the lowest-cost falsification. Do not compute until its exact file hashes, source-only reader, target definition, weighting policy, null seed, and practical/MDE rule have passed review.

If it passes, run the stronger SUA `27 + 6` target-free application test. That SUA step—not an M1 rerun and not reuse of M2's viewed local held-out sessions—is the scientifically meaningful gate before GPU training or formal evaluation.

## Reviewed local evidence

- [M1 fixed-K result audit](M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A_RESULT_AUDIT.md)
- [SUA Step-2A independent result audit](SUA_STEP2A_INDEPENDENT_RESULT_AUDIT.md)
- [M1/SUA route root review](SUA_M1_NEXT_EXPERIMENT_ROUTE_ROOT_REVIEW.md)
- [M2 CPU correction and carrier results](M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md)
- `results/m2_heldin_postsupport_endpoint_v1/audit.json` (read-only endpoint inventory and candidate-branch seal)
- `streaming_calibration_exp/src/data/falcon_k4_features.py` (raw M2 contract only; its velocity input is excluded here)
