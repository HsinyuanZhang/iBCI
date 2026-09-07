# Independent audit — M1 B20 source characterization v1

**Scope:** read-only audit of the completed result and its frozen protocol. This audit did not
open NWB, rerun source calculations, use GPU, read held-out data, or alter result/code/protocol.

## Verdict

**Verified terminal state: `b20_component_attribution_indeterminate_stop`.** The native result
is internally consistent with the frozen protocol: A2 reproduction, identity-plus-4,095 random
schedule convention, all family ranks, session-median conditions, forced decision precedence,
and diagnostic-only reliability boundary all pass audit.

The result shows strong source-only channel-row attachment for B20, R10, and Q10 in the four
already-seen M1 sessions. But only R10 has a material conditional loss in every source session.
Q10's attachment family is statistically distinguishable, while its conditional loss is below
`0.030 R2` in all four sessions. That is deliberately neither a legal simplification outcome
(A-Q was distinguishable) nor a legal composite-B20 outcome (all `D_Q >= 0.030` fails).

This B20 branch therefore stops before any B20-to-M2, B20-to-SUA, decoder, GPU, formal-held-out,
EvalAI, or quantization extension. This no-go applies only to this B20 branch; it neither
validates nor cancels independently governed M2/T4/SUA work.

## 1. Artifact, binding, and scope

| Check | Verified value | Status |
| --- | --- | --- |
| Result JSON | `results/m1_b20_source_characterization_v1/source_characterization.json` | pass |
| Result SHA-256 | `adebc8c6619503efbd248d8825b7449b608c4667cc8860908575bd59182114ca` | pass |
| Adjacent `.sha256` | identical hash and filename | pass |
| Frozen protocol hash | `809dd76de9e0e3c8829bf21efce7b5872ff0cb6db6f4784dee28c946ea07050d` | pass |
| Current protocol file | same hash as artifact binding | pass |
| Source manifest hash | `da3a6397a0739ddca09e999c8a2a64fc968d3b8ca860e96d6e600570bd624902` | pass |
| Sources | ses-20120924/26/27/28 and all four NWB hashes match manifest | pass |
| Prelaunch receipt | `643481e47b0047691c72b93eaa158a5d94d5a97ea766629028a93869f58f3b76` | pass |
| Root review receipt | `ccf5e89f6c078814a75123e852523c1e1f45d0fc87037cd66a6354ebd0d50009` | pass |
| Contract / runner / exact loader | `f06c2a40…` / `eaff3fba…` / `2454ef7c…`, all equal current and prelaunch hashes | pass |
| Result scope | source-only and M1-already-seen; formal-held-out, decoder, GPU, EvalAI all false | pass |

Static runner review agrees with the receipts: it clears `CUDA_VISIBLE_DEVICES`, uses the
exact-four-source loader only after review, builds carriers from support-only inputs, and passes
future target only into the ridge scorer. The reliability function accepts support/session/repeat
only; it has no target or ridge argument.

## 2. A2 reproduction

The canonical B20 R2 checkpoint is exactly reproduced at the declared `1e-10` tolerance:

| Left-out source session | A2 expected R2 | Observed R2 | Absolute error |
| --- | ---: | ---: | ---: |
| ses-20120924 | 0.8924489340 | 0.8924489340 | 0 |
| ses-20120926 | 0.9425713887 | 0.9425713887 | 0 |
| ses-20120927 | 0.9634109918 | 0.9634109918 | 0 |
| ses-20120928 | 0.8205992848 | 0.8205992848 | 0 |

`H_B = 0.9152934898`; canonical session utilities are 0.9028929055, 0.9456903183,
0.9647024926, and 0.8478882428 in session order above.

## 3. Recomputed row-attachment ranks

I recomputed H from all reported per-fold utilities, the exceedance count, every session
median, and both score checksums. All pass. Every family contains one canonical identity and
4,095 random schedules. There are zero random upper-tail exceedances, ties count against B20:

```text
p = (1 + 0) / 4096 = 0.000244140625
Beta^-1(0.975; 1,4095) = 0.000900419642
Bonferroni threshold = 0.05/3 = 0.0166666667
```

| Family | Random H median | Random H range | `H_B - max(H_random)` | p / MC 97.5% upper | Canonical U > its session median |
| --- | ---: | --- | ---: | --- | --- |
| A-all | 0.479101736 | [0.437761514, 0.511907595] | 0.403385895 | 0.000244141 / 0.000900420 | yes, 4/4 |
| A-R | 0.870226050 | [0.856439668, 0.879712274] | 0.035581216 | 0.000244141 / 0.000900420 | yes, 4/4 |
| A-Q | 0.904215632 | [0.882716222, 0.912294530] | 0.002998960 | 0.000244141 / 0.000900420 | yes, 4/4 |

The independently recomputed session null medians are:

| Family | ses24 | ses26 | ses27 | ses28 |
| --- | ---: | ---: | ---: | ---: |
| A-all | 0.478635645 | 0.482319518 | 0.481825386 | 0.476179365 |
| A-R | 0.868562201 | 0.912096649 | 0.909090761 | 0.792456828 |
| A-Q | 0.895301888 | 0.931437541 | 0.956055610 | 0.835529492 |

The stored little-endian float64 H checksums and compact sorted-JSON per-fold checksums also
match independently for A-all, A-R, and A-Q. Thus all three `distinguishable=true` flags are
correct and no single-shuffle or raw-R2 heavy-tail rule has been used.

## 4. Component decision and precedence

Canonical R2 minus random per-session median R2 gives:

| Loss | ses24 | ses26 | ses27 | ses28 | Frozen practical condition |
| --- | ---: | ---: | ---: | ---: | --- |
| `D_R` (break R attachment) | 0.043776955 | 0.038946428 | 0.063411171 | 0.082497680 | `>=0.030`: pass 4/4 |
| `D_Q` (break Q attachment) | 0.009390620 | 0.016180682 | 0.009375253 | 0.017445129 | `>=0.030`: fail 4/4; `<0.030`: true 4/4 |

The frozen predicate values are:

```text
A-all, A-R, A-Q distinguishable = true, true, true
all(D_R >= .030) = true
all(D_Q >= .030) = false
both-negligible = false          # both component families would have to be non-distinguishable
R10/Q10 simplification = false   # their relevant conditional family is distinguishable
two-block composite = false       # D_Q practical condition fails
```

The both-negligible/R10 lower-state precedence cannot fire, because both component families are
distinguishable. Direct zero-padded R10/Q10 scores are descriptive only and cannot override the
full-width conditional-null predicates. The stored indeterminate stop is therefore the only
legal decision under the protocol.

## 5. Reliability remained diagnostic-only

The result records exactly 1,024 complementary Binomial-0.5 thinnings per source session,
rescaled by two before B20 construction, and a target-free odd/even-bin diagnostic. All B20,
R10, and Q10 defined-row fractions are 1.0. B20 thinning median-row-cosine 2.5%/50%/97.5%:

| Session | B20 | R10 | Q10 |
| --- | --- | --- | --- |
| ses24 | 0.967532 / 0.975649 / 0.983296 | 0.967792 / 0.975996 / 0.983542 | 1.000000 / 1.000000 / 1.000000 |
| ses26 | 0.967619 / 0.974895 / 0.980973 | 0.964190 / 0.973848 / 0.979957 | 1.000000 / 1.000000 / 1.000000 |
| ses27 | 0.956691 / 0.967861 / 0.975688 | 0.950773 / 0.962620 / 0.972373 | 1.000000 / 1.000000 / 1.000000 |
| ses28 | 0.960107 / 0.969310 / 0.976571 | 0.962999 / 0.972003 / 0.979032 | 1.000000 / 1.000000 / 1.000000 |

Odd/even B20 median row cosines are 0.971166, 0.981440, 0.971752, and 0.978773. The receipt
retains per-coordinate Pearson values and writes undefined values as `null`; some Q10 odd/even
coordinates are undefined, so Q10's apparent unit cosine of one is not deployment evidence.

Reliability did not enter any p-value family, practical-loss predicate, or terminal decision.
It adds zero biological samples (`independent_biological_samples_added=0`) and cannot rescue the
indeterminate component attribution.

## 6. Required disposition

Freeze the B20 branch at this audited terminal state:

- do not start B20-related M2, SUA, decoder, GPU, formal-held-out, EvalAI, quantization, or
  parameter-sweep work from this result;
- do not post hoc choose R10 merely because its direct score is close to B20: the frozen
  simplification gate did not fire;
- do not claim Q10 deployment value from its technical reliability; and
- retain only the narrow finding: this already-seen M1 source proxy is row-attached, R has a
  material conditional contribution, and Q's practical contribution is unresolved.

Any revisit needs a new independent protocol and independent data boundary; it cannot treat
these already-seen M1 sources as confirmation.
