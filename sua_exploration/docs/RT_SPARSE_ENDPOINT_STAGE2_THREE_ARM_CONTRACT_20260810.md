# RT sparse-endpoint Stage 2 — fresh matched three-arm contract

**Status:** terminal development evidence. The frozen 15-fold × 3-arm matrix completed 45/45 fresh fits and one-shot evaluations; the independent terminal verifier passed. A separate forward-only companion subsequently evaluated the matched B2-D1024 checkpoints on the exact Stage-2 query identities.

**Bound evidence:** protocol addendum SHA `741e41249ab0d5fd771f5298f885afc1e468f09f51d29cb4beb78dd63da89581`; Stage-0B receipt SHA `b88c91ab5cfb30b4a9ef978622e00488193c4ad18b09498c84ba76e10b9943b1`; Stage-1 receipt SHA `9b69eaaa2339610116a5db8efa23ba20ad61459373293d98e80ceec294c3d0e9`; Stage-1 root review SHA `c94352899c9a55a253e3a730d1af6a247c73d3c77df00a1e597f1c4871fb411b`.

Terminal bindings: matrix manifest SHA `93a1aa3549b844c399ab1cc2b9bddb1d93ee2070b51c91e80d60875cee4b3ca4`; matrix aggregate SHA `bb2806953e979180c408fb55744534be6fa470d4144f210cc50917a9b1006b7d`; independent verifier status `PASS_INDEPENDENT_TERMINAL_VERIFICATION_READ_ONLY`; exact-query B2 companion receipt SHA `c36ec0e31ed913ed4e8077f9a4d9d634d53529ce037ad06af1f48d279b16820e`.

## Frozen matrix

Each arm has **15 fresh outer-LOSO fits** and **15 one-shot outer-target evaluations**, for `45` fresh fits and `45` evaluations total. Every arm/fold uses seed `42`, M24 support, query start trial `24`, window size `50`, maximum trial length `100`, the same nested source/validation partition, session-window budget `4096`, **35 epochs**, identical optimizer/learning-rate settings, and the same inner-only checkpoint metric/rule. This is an independently trained system comparison, not a fixed-weight causal descriptor ablation. A fit never opens its outer target. The post-selection target evaluator must have no optimiser, backward pass, gradient update, or target-session parameter update.

| Canonical arm | Required target carrier | Behaviour allowed while constructing the target carrier |
| --- | --- | --- |
| `R-T4d` | endpoint-derived `[a,c,0,0]`; one reach-mean neural response row per retained reach | carrier construction on inner-train, inner-val, and outer target uses only M24 trial events, cursor-position endpoint samples, and spikes; no dense cursor-velocity series, value, validity mask, or integral enters the carrier |
| `R-Full` | current dense velocity `[w_x,w_y,||W||,b]` | existing dense velocity construction is allowed |
| `R-Zero4` | exact zero `[0,0,0,0]` | no target behaviour label is used |

The T4d source-only 4-D normalizer is fit from exactly the 13 inner-train sessions; its padded coordinates remain exact zero before and after normalisation. The outer target carrier is constructed and hashed before the decoder target stream is loaded; a before/after carrier-state hash and runtime access log are mandatory. Dense velocity remains legal for decoder source training and outer-query scoring in all arms, including T4d. It is prohibited only from T4d carrier construction and may not cause target query velocity to enter that carrier.

All three arms must be newly source-trained within the same matrix. The sealed historical `R-Full = 0.44195` is prohibited as a matched comparator, threshold, checkpoint source, or substituted result.

## Terminal outcome

The fresh matched arm means are T4d/Full/Zero4 `0.448176/0.445189/0.179272`. The predeclared primary contrast passed: T4d−Zero4 has mean `+0.268905`, median `+0.241568`, leave-largest-absolute-out mean `+0.259142`, and 15/15 positive folds (exact two-sided sign-test `p=6.1035e-5`). T4d−Full has average `+0.002987`; this is reported only as separate dense-supervision context and does not support superiority, equivalence, or non-inferiority.

The exact-query forward-only B2 companion is not a fourth freshly trained Stage-2 arm. It reuses the sealed matched B2-D1024 checkpoints without retraining or reselection and evaluates them once on the Stage-2 query identities. B2 mean is `0.145148`; T4d−B2 has mean `+0.303028`, median `+0.292102`, leave-largest-absolute-out mean `+0.291237`, and 15/15 positive folds. B2 is a same-pipeline SPINT-structured reference, not a claim to reproduce every released-code RT detail.

## Primary comparisons and reporting

The primary comparison is `R-T4d − R-Zero4`; the secondary descriptive comparison is `R-T4d − R-Full`. For both, aggregate all 15 paired outer-session differences and report equal-session mean, median, positive/zero/negative sign count, the full ordered session values, and leave-largest-out mean (remove the largest absolute session difference; break ties by earliest session name). The frozen primary gate is **mean > 0, median > 0, and a positive-sign majority**. A mean alone is not a pass condition. Report, without gating, whether the mean and median respectively reach `+0.03`; this is a practical-magnitude annotation, not a post-hoc replacement gate.

## Reviewed production implementation

The clean nested runner and data configuration freeze the common M24/query/window/pool/seed/outer-LOSO machinery for all three arms. `RtNestedLossoDataModule` now selects a dedicated `rt_sparse_endpoint_t4d` loader before the shared dense-velocity path. That loader reads the M24 trial events, cursor-position timestamps, only the deduplicated position samples needed to bracket the retained reach endpoints, and spike times; it freezes and hashes `[a,c,0,0]` before calling the ordinary dense decoder-target loader. The latter call remains legal for source decoder training, query construction and scoring, but its values cannot enter the already-frozen T4d carrier.

The source-only T4d normalizer is fit from exactly the 13 inner-train sessions. Its two padded dimensions are forced to mean zero and standard deviation one so that their normalized values remain exact zero. The one-shot outer evaluator constructs the target carrier only after validating the source-only checkpoint receipt and then checks model-state hashes at three points: before target-carrier construction, after target-carrier construction, and after scoring. Any mismatch fails closed.

The immutable production-parity receipt
`results/rt_simple_label_v1/stage2_preflight/RT_SPARSE_ENDPOINT_PRODUCTION_CPU_PARITY_v2.json`
has status `PASS_PRODUCTION_T4D_STAGE1_PARITY_AND_NWB_PROVENANCE_NO_GPU` and SHA-256
`c398eacced3ae09840eec91fdaba8e2154944a8291cef37c9756be16d0935f4a`. It verifies all 15 NWB files before payload access and reproduces the frozen Stage-1 `[a,c]` values with maximum absolute error
`4.407153824104171e-07`, below the fixed `1e-6` tolerance. The production loader actually reads
`2751` two-dimensional endpoint/bracket samples (`5502` scalar coordinates); the separately reported
semantic endpoint payload is `2764` scalar coordinates. Neither quantity is a human-annotation-cost claim.

## CPU preflight terminal rule

The preflight script binds all four receipts/reviews, hashes the current runner/data/evaluator surfaces, confirms the 15-fold × 3-arm schedule and prohibited historical reference, and emits one of two CPU-only states:

* `READY_FOR_SEPARATE_GPU_REVIEW`: all three legal arm adapters and exact evaluator contracts exist.
* `STOP_STAGE2_IMPLEMENTATION_GAP_NO_GPU`: any missing sparse adapter, target no-BP proof, arm mismatch, receipt/hash drift, or schedule drift.

The per-cell preflight historically returned `READY_FOR_SEPARATE_GPU_REVIEW` with an empty implementation blocker list. The matrix supervisor freezes 45 unique `(fold, arm, seed)` cells, hashes the source/config trees, teacher checkpoint and all 15 NWBs, rechecks those inputs before every child launch, uses canonical one-shot target outputs, and resumes without re-evaluating an existing valid outer receipt. Each terminal closure revalidates the source-selection/checkpoint/config/split/outer receipt chain and the target-session no-backprop state proof. Aggregation is refused until all 45 fresh cells exist and, within every fold, the three arms have identical ordered query-window, target-covariate and evaluation-mask digests. The completed terminal verifier confirmed these conditions before emitting the predeclared mean, median, sign, leave-largest-absolute-out and practical-magnitude statistics. The historical readiness state itself was not an accuracy result and never permitted substituting the sealed historical Full score for a fresh matched arm.
