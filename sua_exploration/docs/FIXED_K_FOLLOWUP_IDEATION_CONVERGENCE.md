# Fixed-K temporal prototype: follow-up ideation and convergence

**Date:** 2026-08-02
**Input tension:** M1 source-only prototype beats rate-only by `+0.100526` proxy R2 in 4/4
sessions and is highly repeatable, but the one session-keyed slot-shuffle control has a
heavy-tailed failure distribution and does not pass the predeclared paired mechanism gate.

This document records the candidate space and its execution disposition. Sections 1--6 are
historical protocol ideation, not a live work queue. They do not revise the v3 result or authorize
data access, SUA work, decoder/GPU/formal evaluation, or INT8 work.

## 1. Divergent candidates

| ID | candidate question | initial disposition |
|---|---|---|
| C1 | Does the aligned carrier outrank a frozen distribution of many session-keyed whole-slot permutations? | retain; directly addresses the failed mechanism gate |
| C2 | Does within-trial bin-order destruction remove the gain while preserving per-trial/unit count marginals? | retain; tests temporal order |
| C3 | Can a width-20 quantile/moment/exposure carrier reproduce the gain without temporal state? | retain; simplicity baseline |
| C4 | Does a filter-bank with reversed within-support time provide an acausal upper-bound diagnostic? | reject now; not deployable and weaker than C2 |
| C5 | Do phase-randomized support trains remove the gain? | reserve; complex null with avoidable spectral assumptions |
| C6 | Does a per-unit spike-count histogram match the prototype? | merge into C3 |
| C7 | Does changing `K` from 4 to 2/8 improve the source proxy? | reject until mechanism passes; hyperparameter rescue |
| C8 | Does changing temporal rank/filter constants improve the result? | reject until mechanism passes; same reason |
| C9 | Does a learned router improve slot coherence? | reject; raises complexity before fixed mechanism is established |
| C10 | Does the carrier predict an unlabeled later overall-rate endpoint? | diagnostic reserve; weaker task than category tuning |
| C11 | Does the result reproduce on SUA train/development sessions with variable unit count? | retain conditionally; stronger cross-domain sample size |
| C12 | Does it reproduce on native MUA M2? | retain conditionally; closest hardware/data-domain replication |
| C13 | Can the prototype be concatenated into the existing frozen decoder without training? | reject; prior frozen-space mismatch already failed |
| C14 | Can an offline-trained, modality-dropout decoder use prototype state with zero target backprop? | retain only after C1--C3 and cross-domain evidence |
| C15 | Can a low-rank FiLM or key residual fuse the prototype? | defer behind the simplest decoder integration |
| C16 | Can raw calibration trials be stored as in-context key/value memory? | reject now; state/latency increase and unnecessary before compact state is validated |
| C17 | Can the prototype accumulator and side projection be INT8-quantized? | retain only after decoder accuracy is positive |
| C18 | Should the existing v3 shuffle be re-scored with a sign test or outlier deletion? | reject; post-hoc rescue of a frozen result |

## 2. Convergence

The retained immediate idea is **mechanism decomposition before model integration**:

> A compact, label-free temporal prototype appears to predict later M1 channel tuning beyond
> mean/variance/exposure. A new source-only experiment should determine whether that advantage
> requires chronological temporal structure and a coherent shared slot coordinate, rather than
> arising from nonlinear marginal rate statistics or one pathological shuffle draw.

It resolves three competing explanations with the smallest additional scope:

1. **slot coherence:** aligned carrier versus a predeclared permutation-null distribution;
2. **temporal order:** aligned carrier versus within-trial order destruction;
3. **marginal complexity:** aligned carrier versus a width-matched distributional rate baseline.

## 3. Kill criteria

The branch stops before a decoder if any of the following holds under the new frozen protocol:

- aligned performance does not exceed the marginal carrier by a distinguishable, practically
  nontrivial session-level amount;
- aligned performance is not extreme under the predeclared slot-permutation null;
- temporal-order destruction retains essentially all aligned gain;
- a raw/oracle/unit-permutation/state contract fails;
- the result depends on changing K, rank, filters, ridge, endpoint, or excluding a session after
  seeing the result.

Technical resamples and permutation draws are null/measurement repetitions, not biological
sessions. They cannot replace source-session uncertainty.

## 4. Historical conditional validation sequence (superseded)

The following was the prospective gate order before the B20 characterization and R10 execution.
It is retained for provenance only; the branch is now closed by the outcomes in Sections 7--8 and
does not authorize any of its conditional continuations.

1. M1 exact-four-source CPU Gate A2 with C1--C3.
2. If and only if A2 passes, one source/development cross-domain CPU replication, selected before
   data access from the SUA/M2 feasibility audit.
3. If both pass, one minimal decoder development cell comparing F0, rate-distribution,
   prototype, and the frozen temporal null under matched training/checkpoint rules.
4. Only a predeclared decoder margin opens the remaining development cells and one frozen official
   held-out evaluation.
5. Only a positive frozen decoder candidate receives encoder/prototype-path INT8 PTQ/QAT. Decoder
   quantization remains out of scope because it has already been established elsewhere.

## 5. Strongest objection

**Objection:** the `+0.10` proxy may exploit persistent electrode/channel identity or the scheduled
relationship between the first ten and later category blocks, rather than a portable neural
temporal mechanism.

**Response:** the new gate must preserve channel rows and all per-trial/unit marginals while
destroying only temporal order, and must compare against a high-capacity marginal carrier. A
cross-domain replication with variable unit count then tests whether the effect survives beyond
fixed M1 channel indexing. Failure at either stage ends the claim.

## 6. Historical two-week upper-bound feasibility pilot (superseded)

The actual CPU gate is expected to be much faster, but the outer feasibility envelope is:

- days 1--2: immutable protocol, synthetic tests, exact source-only receipt;
- days 3--4: M1 CPU Gate A2 and independent audit;
- days 5--7: one conditional SUA or M2 CPU replication;
- days 8--10: one decoder cell only if both CPU gates pass;
- days 11--12: remaining frozen development cells only if the first decoder cell passes;
- days 13--14: one official held-out evaluation and encoder-side INT8 only if all earlier gates
  pass.

Every failed gate shortens this schedule; it never redirects compute into a rescue sweep.

## 7. Execution outcome and simplicity pivot

Gate A2 stopped at its first content comparison. B20 exceeded P20 in all four source sessions;
`P20-B20` mean was `-0.041963` with paired 95% CI
`[-0.076004,-0.007923]`. Consequently C1/C2 and every P20 decoder/cross-domain continuation are
closed without running the expensive nulls.

The result elevates a candidate that was originally introduced only as a simplicity control:
order-invariant B20 exceeded legacy rate-only by mean `+0.142489` in 4/4 sessions. This does not
rescue the fixed-K claim. The subsequent component gate was indeterminate (Section 8), so this
source-only observation does not open an M2, SUA, decoder, GPU, formal, or INT8 branch.

## 8. B20 characterization outcome and R10-only follow-up boundary

The guarded M1 B20 characterization reproduced the four Gate-A2 B20 scores with zero numerical
error and found that complete B20 rows are genuinely channel-attached: `A-all`, `A-R`, and `A-Q`
each had zero exceedances among 4,095 random schedules (`p=1/4096`; Monte-Carlo 97.5% upper bound
`0.0009004`). Measurement repeatability was also high.

The component decision was nevertheless
`b20_component_attribution_indeterminate_stop`. Breaking R10 attachment cost
`0.03895--0.08250 R2` across the four sessions, whereas breaking Q10 attachment cost only
`0.00938--0.01745`, below the frozen `0.03` practical threshold in every session. Q10 was still
statistically distinguishable, so the preregistered rule correctly forbids both a full-B20 claim
and a claim that Q10 has no effect.

Standalone R10 was a **new lower-state hypothesis for independent CPU replication**, not a passed
B20 simplification: on the four M1 source sessions it beat rate-only by
`+0.11275/+0.13576/+0.13186/+0.15447 R2`, while full B20 exceeded R10 by only
`+0.00476/+0.01188/+0.00482/+0.01366`. R10 requires trial totals/exposures but no 256-bin
histogram.

That independent M2 source-only gate is now closed without a performance result. Its v1 execution
was invalid at the unsigned-padding check (`OverflowError: Python integer -1 out of bounds for
uint8`). The one authorized mechanical v2 retry was also invalid before scoring because the target
defined-row fraction was below `0.90`. Neither execution produced R2 values, R10--L10 deltas,
reliability, or schedule-null results. These are execution/data-validity stops, **not** an R10
performance-negative finding. Together with `b20_component_attribution_indeterminate_stop`, they
close the B20/R10-to-SUA, decoder, GPU, formal, and INT8 escalation paths. No continuation may be
inferred from the M1 source proxy values above.
