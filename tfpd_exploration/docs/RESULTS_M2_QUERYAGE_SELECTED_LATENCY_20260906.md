# M2 selected QueryAge: same-host public-call latency receipt summary

## Result in one sentence

On this host and frozen image, the selected cached QueryAge FLAT and ROUTE
runtimes had consistently higher mean latency than the frozen Original wrapper
with one intra-op CPU thread (T1), but consistently lower mean and p95 latency
with two threads (T2). T1 p95 direction varied across repeats.
This is a **within-host, configuration-specific observation**, not an
isolated-service, production, or official latency claim.  It makes no quality,
generalization, or selection claim.

The six formal receipts all have status
`PASS_PUBLIC_STREAM_TIMING_ONLY`.  Immediately before this summary was written,
each receipt's bound authority was revalidated by
`compare_m2_queryage_cached_spint._revalidate` in a CPU-only, taskset 4--7,
one-thread process.  That check re-hashes the active code closure, selected
exports, finalizer and trainer/source authority, Original wrapper/payload,
both complete-stream proofs, the required current smoke receipts, and the
seven-lane source hashes.  All six checks passed.  It does not load or time a
model.

Root separately recomputed all five statistics from all36,864 positive, finite
raw timings (six receipts × three engines ×2,048 calls). Every stored mean,
p50, p95, p99 and maximum matched exactly. Receipt hashes matched the table;
each immutable pre-authority sidecar, its external-authorization hash and
content, the seven-lane source identity, configuration, no-update flag and
reported public-window oracle bounds also passed. This audit did not construct
or time a model and did not replace the source-closure revalidation above.

## Measurement surface and fixed identity

Every formal receipt measures whole public `predict()` calls, not a component
timer.  The three engines are reset, then invoked in rotating
`ORIGINAL -> FLAT -> ROUTE` order (rotated every input) on the same continuous
native FP32 source sequence.  The output contract is an owned, contiguous,
finite `float32 [7, 2]` array.  The measurements include the selected ROUTE
state; FLAT is not used as a proxy for ROUTE.

| Fixed property | Bound value |
| --- | --- |
| Container image | `sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f` |
| Frozen Original SPINT module | `855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519` |
| Batch / source | B7; seven held-in source lanes; FP32 stream `[2177, 7, 96]` |
| Per receipt | 1 initial bin + 128 warm-up calls + 2,048 timed calls |
| Models | Original; selected plain-EMA FLAT epoch 002; selected plain-EMA ROUTE epoch 020 |
| Finalizer receipt | `57c214b12c2097aa6bff74451169d092296f473c1d544dad53344288c23e9dcc` |
| Selected FLAT export | `9d0ba29d6f9f3ab7df4880092902c7bac25a319741c65f8007b64dccd41e2b5f` |
| Selected ROUTE export | `182d3b28e4c4668df43ebf4edd8305d7b23ff9c771013fa5d6a1d5740b68fa33` |
| Complete-stream proof: uncached | `93775aa2b2963386681b93c0fb2265f372062f7ae3d9e1dff445fa6335c1fff7` |
| Complete-stream proof: cached | `bb3116055f7cd289715b4acadd720c17c5a785d89141eac563df78fd25589701` |
| Required smoke receipts | T1 `d23da32f3e432dcef8fbc057f64daab3a6b7528e8f19850a595ad2b00b403021`; T2 `ad9d2119d7c4fa401a03cfcde97015373d6225f227beeec75e1c794b648550ee` |

The bounded timing process used CPU only (`CUDA_VISIBLE_DEVICES=-1`), inter-op
threads 1, CPU affinity 4--7, and the thread count named by T1 or T2.  Each
receipt also checks public-output shape/dtype/ownership/finite values,
parameter immutability, independent rolling-W50 history, and independent
whole-process output oracles.  Consequently, the timing result is tied to the
actual selected public boundary, but it does **not** establish that this host
was otherwise idle or that it matches a deployment service.

## Formal receipts: absolute latency

Values are milliseconds per timed public call.  `p95` is the empirical 95th
percentile within that one 2,048-call receipt.  No rows are dropped and no
best repeat is selected.

| Configuration / repeat | Original mean | Original p95 | FLAT mean | FLAT p95 | ROUTE mean | ROUTE p95 | Receipt SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T1 (1 thread), r1 | 13.283370 | 15.618175 | 13.962413 | 14.998902 | 14.057481 | 15.119870 | `cd9c5c8e0162e6b4a27be1bfcac93b972ae4a3021d1748263ecb150c47bb1ce6` |
| T1 (1 thread), r2 | 12.637992 | 12.781407 | 13.715343 | 13.963178 | 13.798016 | 14.093245 | `5131ca5c6c73b8ba414ff628a4d2184c5ec6bd36483fbfa61d23429d162233b3` |
| T1 (1 thread), r3 | 12.748207 | 13.281007 | 13.689080 | 13.998776 | 13.788723 | 14.149826 | `70326cf5774f936c5fd2c3df09f54b51a2a297fee90371a89a67a24ac8f98285` |
| T2 (2 threads), r1 | 9.692910 | 11.136863 | 8.944719 | 9.767778 | 9.042438 | 9.856859 | `189cdfef8976fb4b7f4f88dc28f186dfe89acd0a87bdf5c2fc7afbd740e77088` |
| T2 (2 threads), r2 | 9.404215 | 9.725240 | 8.686300 | 8.869888 | 8.788515 | 8.967107 | `ef873869ceeb1d012144c5ec67626a2c5f044db8de0385a51448f0cf8dbd33cd` |
| T2 (2 threads), r3 | 9.305388 | 9.511420 | 8.876730 | 9.064241 | 8.964153 | 9.143999 | `0653482ff6575fcc7bab55d4e40313dcd558ee790c5b7f11214ceea21203f38b` |

The raw, self-contained receipt files and their pre-authority sidecars are in
`results/family_runtime_v1/m2_queryage_selected_cached_spint_b7_v1/`:
`t{1,2}_2048_repeat{1,2,3}.json` and
`t{1,2}_2048_repeat{1,2,3}.json.authority_pre.json`.

## Repeat-level comparison to Original

Percentages below are computed separately inside each repeat as
`100 * (candidate / Original - 1)`.  Negative is lower latency.  The median
and range summarize the three repeat percentages for the named thread setting;
they are not a pooled 6,144-call latency distribution and must not be compared
across T1 and T2 as though thread count were unchanged.

| Configuration / candidate | Mean vs Original: r1, r2, r3 | Mean: median [range] | p95 vs Original: r1, r2, r3 | p95: median [range] |
| --- | --- | --- | --- | --- |
| T1 / FLAT | +5.112%, +8.525%, +7.380% | +7.380% [+5.112%, +8.525%] | -3.965%, +9.246%, +5.404% | +5.404% [-3.965%, +9.246%] |
| T1 / ROUTE | +5.828%, +9.179%, +8.162% | +8.162% [+5.828%, +9.179%] | -3.191%, +10.264%, +6.542% | +6.542% [-3.191%, +10.264%] |
| T2 / FLAT | -7.719%, -7.634%, -4.607% | -7.634% [-7.719%, -4.607%] | -12.293%, -8.795%, -4.701% | -8.795% [-12.293%, -4.701%] |
| T2 / ROUTE | -6.711%, -6.547%, -3.667% | -6.547% [-6.711%, -3.667%] | -11.493%, -7.796%, -3.863% | -7.796% [-11.493%, -3.863%] |

## Interpretation and decision

1. **T1 is unfavorable for the candidates.**  The mean is slower in every
   T1 repeat: FLAT by 5.112--8.525% and ROUTE by 5.828--9.179%.  The p95
   direction in r1 is favorable while both r2 and r3 are unfavorable, so the
   T1 tail evidence is visibly less stable than its mean evidence.  A claim
   that cached QueryAge is lower-latency at one thread is not supported.

2. **T2 is favorable on this same-host surface.**  Both candidates are lower
   than Original in all three means and all three p95s.  At T2, FLAT is
   -4.607% to -7.719% in mean and ROUTE is -3.667% to -6.711%; the ROUTE
   result is therefore a direct measurement, not an inference from FLAT.
   This is useful local evidence for the two-thread configuration only.

3. **Do not collapse the configurations.**  The reversal between T1 and T2
   shows that CPU-thread policy and host scheduling materially affect the
   comparison.  The right operational conclusion is to retain the six
   receipts and their configuration labels, not publish one averaged number
   or select the strongest candidate/repeat.

4. **The first T1 tail result is an outlier relative to T1 r2/r3.**  Its
   candidates have higher means but lower p95s than Original, whereas r2/r3
   are higher on both.  This is reported as variability, not used to infer a
   tail win or a performance regression cause.

## Isolation and contemporaneous-work caveat

These receipts came from a local co-hosted process, not an isolated GPU or
production inference service.  Their CPU-only process boundary protects the
specified benchmark code from CUDA execution; it does not prove exclusive
ownership of the machine, stable CPU frequency, absence of kernel/IO pressure,
or independence from other work.

In particular, the H1 GPU smoke work was active late in the T2 sequence,
including the time window of T2 repeat 3.  That overlap does not invalidate
the receipt's exact input/model/code bindings or the within-process oracle
checks, but it prevents attributing the T2 r3 change to the cached runtime
and is an additional reason not to turn these values into an isolated-service
claim.  The six receipt values stand as observed; no correction or exclusion
has been applied for the overlap.

## Explicit non-claims

- Not an official benchmark, SLA, throughput test, cold-start result, or
  service-latency measurement.
- Not evidence of latency on another host, image, CPU topology, thread policy,
  batch size, source roster, or request mix.
- Not a model-quality, generalization, ranking, or model-selection result.
- Not a comparison against a modified model: the frozen Original wrapper and
  the already selected FLAT/ROUTE plain-EMA exports are the only engines timed.
- Not permission to alter the runtime.  Any implementation change requires a
  new selected-export/native/cache-equivalence closure and a new same-boundary
  timing sequence.

The runtime-boundary rationale and candidate-only optimization hypotheses are
documented separately in
`docs/M2_QUERYAGE_CACHED_RUNTIME_PROFILE_PLAN_20260906.md`; that plan does not
replace these whole-call receipts.
