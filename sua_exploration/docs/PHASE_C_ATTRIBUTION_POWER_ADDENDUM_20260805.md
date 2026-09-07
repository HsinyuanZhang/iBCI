# Phase-C attribution, power, and recovery addendum

**Status:** publication-method addendum; not part of the frozen Phase-A/Phase-C runtime source
closure.  
**Updated:** 2026-08-05 09:50 HKT

This addendum preserves analysis added after the Phase-A canonical source seal. The historical
closure documents remain byte-stable while Phase-C executes; this file does not authorize data
access, score opening, GPU work, or a change to any frozen gate.

## Power and interpretation

The seven held-out sessions are the primary statistical clusters. Three source-training seeds do
not turn them into 21 or 42 independent target observations. Historical M24 score-free dispersion
gives the following approximate two-sided 5%, 80%-power sensitivity range:

| Historical SD | Normal MDE | t MDE, df=6 |
|---:|---:|---:|
| `0.038835` | `0.0411 R2` | `0.0492 R2` |
| `0.083517` | `0.0884 R2` | `0.1058 R2` |

An MDE of `0.03 R2` would require SD below approximately `0.0283` under the normal approximation
or `0.0237` under the t approximation. At a mean of exactly `+0.03`, the frozen three-seed
stability guard also requires seed-mean SD `<0.02598`. Thus `+0.03` is a deployment-relevant SESOI
and frozen engineering decision threshold, not a power-derived significance boundary. Stage-A is
negative-only futility: survival authorizes completion of the matrix and is not efficacy evidence.
If the full matrix fails, the correct conclusion is that it did not achieve the preregistered
stable deployment-relevant gain, not that every smaller true effect is zero.

## Label and attribution boundaries

Native-M2 T4 receives the same chronological neural calibration block as clean SPINT plus one
direction label for each of 16 directional calibration trials. The comparison is therefore a
supervised, BP-free calibration package versus a neural-only baseline, not an equal-label-
information comparison.

The shared-Z4 receipt records:

```text
old_t4_ts4_initial_digests_available=false
three_arm_initial_bit_exact_claim=not_authorized
```

Accordingly, `T4-Z4` is a matched-interface, matched-width, matched-budget end-to-end descriptor-
package availability comparison. It is not a bit-exact initialization ablation or a pure single-
variable causal proof. It simultaneously includes direction-label availability, descriptor
algebra, and source-trained interface use. `T4-TS4` tests correct row attachment conditional on
descriptor availability. Re-running nine source-only cells would not repair historical initial
digests or create a new independent endpoint, so that rerun remains closed.

## Quantization boundary

Encoder-only QAT seed42 already fails the frozen all-seed maximum-saturation gate:
`0.0292906 > 0.005`, although its SUA/pseudo-MUA accuracy, integer-code parity, overflow, and
byte-identical FP32-decoder checks pass. Seed44 can complete the failure ledger but cannot rescue a
strict three-seed PASS. Current evidence is encoder W8A8/INT32 plus an FP32 decoder; it is not
full-model INT8, silicon PPA, or measured end-to-end hardware latency evidence.

## Phase-C r5 evaluator incident

Both r5 SPINT seed42 folds 0/1 completed all 35 training epochs and then failed only in the
post-test B=1 deployment microbenchmark because Lightning restored train mode. No endpoint payload
or score commitment was sealed or opened. The symmetric repair calls `model.eval()` after
`Trainer.test()` and before `benchmark_online_b1()` in both SPINT and T4 workers; the relevant
Phase-C/capability suite passes `89/89`.

The immutable incident receipt is
`results/m2_native_post33_phase_c_v4_eval_recovery_incident_20260805/incident_and_r6_decision.json`
(SHA-256 `a73d02f6cbc7691f30bd09b6d0f13ca72b0720ae8db56ede3976d2f69529715e`). It
forbids evaluator-only salvage because that would reuse consumed nonces, mutate a failed root, or
relax selector/config/source-cost root binding. The only permitted recovery is an append-only r6
fresh-root exact retrain under the patched source closure.

### r6b continuation incident

The first fresh-root retry, r6b, was intentionally stopped after approximately five minutes and
before any endpoint payload, score commitment, deployment evidence, opened decision, or score
access. It had issued only Stage-A execution and opening capabilities, then destroyed its
ephemeral private key. The production opener writes a Stage-A decision but does not sign it;
production Stage-B requires a detached signature and requires every Stage-B/full-opening
capability to bind that exact decision and signature. Consequently r6b could never continue after
a surviving Stage-A gate. It was stopped with two started/failed SPINT cells, zero completed cells,
and six aggregate epoch checkpoints. r6c must keep the same non-persisted private key in a
fail-closed live process through unconditional decision signing and conditional Stage-B/full-
opening issuance.
