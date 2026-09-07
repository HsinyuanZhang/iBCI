# M2 R10 M24 source gate — root result audit

**Audit date:** 2026-08-02  
**Disposition:** terminal endpoint-infeasibility stop; no R10 performance estimate  
**Scope:** seven already-seen, hash-bound M2 held-in source sessions; CPU only

## 1. Outcome

The prospective experiment asked whether ten quantiles of the first 24 unlabeled trial log-rates
(`R10`) predict a later neural autocorrelation proxy better than the frozen low-order `L10`
control. It never reached that comparison.

| attempt | immutable result | SHA-256 | terminal reason | metrics reached |
|---|---|---|---|---|
| v1 | `results/m2_r10_m24_source_gate_v1/source_gate.json` | `45fefcb8ac45bfba5d6789e38068c02e130ef37d299e2323319868d027c503c1` | `uint8` cannot represent `-1` padding | none |
| v2 mechanical correction | `results/m2_r10_m24_source_gate_v2/source_gate.json` | `8438fb884d915440b7c92bcecb60ad9b96ccfdee513e0a8bdfc2a1c1664f0ef3` | target defined-row fraction below `0.90` | none |

The v1 failure was an implementation-boundary defect. The official binning helper emits unsigned
counts; the production trializer pads with `-1`. The correction was reviewed independently,
covered by a uint8-to-signed regression test, issued under a new prelaunch/review/confirmation,
and wrote to a new output directory. It did not change any scientific feature, target, lag,
session, threshold, or statistic.

The v2 failure is the frozen target-feasibility condition. At least one of the seven exact source
sessions did not have the required whole-row future autocorrelation target defined for `>=90%` of
channels at every required lag. The terminal receipt was written before R10/L10 LOSO scoring.
Because the terminal wrapper intentionally does not expose a post-failure subset analysis, this
audit does not speculate about which session or lag caused the stop.

## 2. What was not computed

No valid value exists for any of the following:

- per-session or mean `R10 - L10` R2;
- operational paired interval or MDE;
- 256-repeat R10 thinning reliability;
- aligned bounded utility `H`;
- any of the 4,095 full-`S_96` attachment schedules or their p-value;
- decoder R2, held-out performance, latency, or INT8 accuracy.

Accordingly, the outcome is neither “R10 works” nor “R10 is worse than L10.” It says that this
predeclared seven-source M2 target cannot identify the question under the frozen coverage rule.

## 3. Integrity and information boundary

Before v2, independent static review confirmed:

- the callable API validates root authorization before any source loader call;
- the formal null is sampled from the complete uniform permutation group, not a derangement-only
  conditional space;
- behavior values and the NWB `eval_mask` are not read;
- trial inclusion is rebuilt only from acquisition timestamps and trial start/stop geometry;
- unit loading materializes spike times, not electrode/waveform/SNR tables;
- whole-row target masks, uint32 overflow, hashes, deadlines, and immutable terminal receipts are
  fail-closed;
- 22 synthetic tests, `py_compile`, and diff checking passed.

Both result receipts bind the exact protocol, manifest, prior-input receipt, production
`FalconDataset`, runner, pure feature module, prelaunch, and root-review hashes. CUDA was disabled;
neither attempt touched held-out, decoder, GPU, EvalAI, or SUA data.

## 4. Scientific disposition

The M1 observation that R10 was close to full B20 remains a source-side, later-neural proxy finding
inside M1. It did not receive cross-domain M2 confirmation. The correct action is therefore:

1. retain M1 R10 as a diagnostic hypothesis, not a validated carrier;
2. do not change the M2 lag grid, target mask, support budget, or session subset after this stop;
3. do not launch the conditional SUA R10 loader/source/dev program;
4. do not construct a decoder, GPU cell, formal held-out run, or INT8 branch for R10;
5. retain the official M2 T4 EvalAI result as the project's positive deployment result; it is
   scientifically separate and unaffected by this R10 endpoint failure.

A future project may define a different neural-only endpoint prospectively, but it would be a new
hypothesis and cannot be called a completion or rescue of this gate.
