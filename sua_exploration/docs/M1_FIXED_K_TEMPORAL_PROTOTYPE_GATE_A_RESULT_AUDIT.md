# M1 fixed-K temporal-prototype CPU Gate-A — root result audit

**Reviewed:** 2026-08-02 (Asia/Hong_Kong)  
**Disposition:** mixed representation signal, but the predeclared Gate A did not pass; stop before
decoder, GPU, held-out, EvalAI, K/r/router search, or quantization.

## 1. Scope and provenance

The authoritative result is
`results/m1_fixed_k_temporal_prototype_gate_a_v3/source_gate_a.json`, SHA-256
`b2b1e9ff3288ef57fc26d2b446fd803155b1ec5e0ce14dc2f9cc1cbddb92f6dd`.
The execution was authorized by a separate root-review file bound to immutable prelaunch SHA-256
`2d7e85013a3519930b36d02356bdd7508fa666c7ced3e48c9d44a1829e6c189d`.

Only the four hash-bound M1 `held-in-calib` source NWBs were opened. The runner did not enumerate
or resolve minival, held-out, formal-test, or EvalAI paths. It used no decoder and forced CUDA off.
The endpoint is an offline source-LOSO **later-neural oracle proxy**, not behavior-decoding R2:
first-ten-trial carriers predict each channel's `[210,end)` category-conditioned neural rate.
Future category labels are used by the scorer only.

The v3 raw-data contracts passed: non-interpolated integer count prefixes, exact `-1` padding
removal, prefix sums equal to production trial spike sums, support `[0,10)`, future target
`[210,end)`, and no retained deployed raw-support matrix. The focused suite passed `19/19` tests.

## 2. Source-LOSO proxy result

| carrier | session R2 values | mean R2 |
|---|---|---:|
| D4 | `0.646740 / 0.689107 / 0.699845 / 0.609199` | 0.661223 |
| rate-only | `0.774948 / 0.794927 / 0.826736 / 0.652464` | 0.762269 |
| fixed-K prototype | `0.880273 / 0.891359 / 0.920905 / 0.758640` | **0.862794** |
| session-keyed slot shuffle | `-104.436467 / 0.789010 / 0.851389 / -1.098782` | -25.973712 |

The content comparison against the dimension-matched rate/exposure control is strong and
internally consistent:

- prototype minus rate-only: mean `+0.100526`, positive in `4/4` sessions;
- paired Student-t 95% CI: `[+0.090798,+0.110253]`;
- measured two-sided-alpha-.05, 80%-power MDE: `0.012719`;
- prototype also exceeds D4 descriptively by mean `+0.201572` in `4/4` sessions, although D4 is
  label-informed and was not one of the two mechanism gates.

This supports a narrow finding: the first-ten-trial temporal distribution contains stable
cross-session information about later channel tuning beyond support mean, variance, and exposure.
It does **not** show behavior-decoder improvement or official held-out generalization.

## 3. Repeatability and deployment receipt

All four sessions passed the frozen split-half requirement. Across 256 complementary within-bin
spike thinnings per session:

- slot-count cosine lower 2.5% quantiles were `0.99947--0.99971`;
- prototype-value cosine lower 2.5% quantiles were `0.99588--0.99667`;
- the minimum value-defined unit fraction was `0.984375`; all other sessions were `1.0`.

At `N=64`, the declared streaming state is 1,728 scalars / 6,912 bytes in FP32, plus 16 shared
anchor scalars. The accounting convention reports 25 multiplications, 37 additions, and 3
comparisons per unit per bin, with zero deployed raw-support elements. These are operation/state
counts, not a measured latency benchmark.

## 4. Why the overall gate still fails

The predeclared mechanism gate also required prototype to beat a session-keyed whole-slot shuffle
with a positive 95% CI lower bound and a mean at least as large as the measured MDE. All four
paired differences were positive, but they were extremely heterogeneous:

```text
+105.316740 / +0.102348 / +0.069516 / +1.857422
```

The shuffle caused catastrophic extrapolation in two outer sessions. Consequently the paired
session SD was `52.326821`, MDE was `108.863822`, and the 95% CI was
`[-56.427143,+110.100156]`. The contrast therefore fails the frozen distinguishability rule even
though its sign is 4/4 positive. With only four biological inference units, the repeatability
seeds cannot repair this uncertainty and must not be counted as extra sample size.

The most defensible interpretation is that the current slot-shuffle control is highly disruptive
but statistically unstable. It does not provide a precise mechanism estimate. Replacing the
predeclared paired-t gate post hoc with a sign test, robust interval, favorable-session deletion,
or a different shuffle family would be a new experiment and cannot rescue v3.

## 5. Frozen decision

The machine decision is `stop_cpu_gate_not_met`; `gpu_authorized=false`. Therefore:

- retain the positive prototype-versus-rate result as **source-only representation evidence**;
- do not claim coherent cross-session slot semantics as established;
- do not start a decoder pilot, formal held-out evaluation, EvalAI submission, K/r/router sweep,
  FiLM/cross-attention fusion, or quantization from this artifact;
- preserve the code and receipt as a potentially useful later hypothesis, but close it in the
  current contracted sequence.

This is not an “everything failed” result. It identifies a stable neural-only carrier with a large
source proxy effect, while also showing that the present four-session mechanism test cannot
support the stronger deployment claim required to spend a GPU/held-out evaluation.
