# M2 matched epoch-34 B0 EvalAI eligibility audit — 2026-08-04

**Status:** CONSTRUCTIBLE_BUT_NOT_SUBMISSION_READY

**Scope:** read-only preparation audit. No EvalAI API request, image push, registration, formal submission, training, or new local/hidden-query evaluation was performed.
**Next external action:** requires fresh explicit user/root authorization.

## Decision

A strict epoch-34 B0 control is technically constructible and is the correct formal control for the existing M2 T4 system. It is **not submission-ready today**: no B0 cached-identity payload, B0 EvalAI image, B0 container-parity receipt, or B0-specific guarded submit helper currently exists.

The existing official original-SPINT M2 submission 578218 is not matched: it packages the epoch-27 decoder. The official T4 submission 578221 packages an epoch-34 decoder. Its official difference remains an end-to-end system comparison until a separately packaged epoch-34 B0 control is frozen and, only after explicit authorization, submitted.

The correct B0 is **not zero identity**. Original SPINT B0 still derives a learned, activity-derived identity from the same chronological first 33 neural calibration trials. Zeroing that identity would create a new weaker baseline and would overstate T4.

### Terminology lock

In this audit, **B0** means the epoch-34 teacher/original-SPINT neural identity
shown explicitly below. It is the activity-derived identity used by the local
M33/q33 B0 receipt. A separately trained streaming B3-only/F0 identity is also
non-zero and may be a useful control, but it is a different checkpointed
encoder; it must be called B3/F0 (not B0) unless a new equivalence receipt
proves otherwise. Neither control may receive the T4 direction descriptor.

## Exact intended contrast

Let C[0:33] be the public chronological calibration neural trials after the frozen preprocessing (raw features, cubic interpolation to 100 bins, no calibration smoothing), and let X be one online 50-bin neural window.

~~~text
E_B0 = teacher.fc_id_out(mean_trial(teacher.fc_id_in(C[0:33])))
y_B0 = frozen_epoch34_decoder(X + E_B0)

E_T4 = frozen_T4_B3S_encoder(C[0:33], normalized_T4[a,c,m,b])
y_T4 = frozen_epoch34_decoder(X + E_T4)
~~~

The formal comparison is therefore an **end-to-end value-of-labeled-calibration** contrast:

- Both arms use the same chronological first 33 neural trials, frozen decoder, online 50-bin history, container parent, and output scale.
- B0 reads no calibration target direction and no T4 descriptor.
- T4 additionally uses one direction label per calibration trial to derive its [a,c,m,b] side descriptor.
- Neither arm may use hidden query labels, optimizer updates, backpropagation, online refitting, or a runtime calibration buffer.

This does not isolate “carrier shape” while holding label information fixed. T4 versus TS4 remains the more focused channel-attachment/content control.

## Confirmed frozen facts

| Item | Verified fact | SHA-256 / immutable ID |
|---|---|---|
| Epoch-34 teacher | required B0 decoder source | fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec |
| T4 selected checkpoint | all-held-in M2 seed-42 B3S checkpoint | 25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e |
| Submitted T4 payload | 13 cached identities, each [96,50] | dcc449a15bc478f3380c95add964fc344522a25bc9938c0a5563bf5a75ae0c96 |
| Submitted T4 image | official private submission 578221 | sha256:57b1fb2418ad2fd8f2d4e62f8fddb6d4f75b9c8a072730bf7afb6af53517d260 |
| Common container parent | original validated M2 image | sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8 |
| Local M33/q33 pair receipt | development-only pairing receipt | 37e15bd09af2b5cd83725423510de40444291ceac8d175e78290c7a642f1114d |
| T4 q33 split receipt | chronological support; full 50-bin query disjointness | 42e7183e7e6af2e24e93f5606efdaf15789186a566adf33ee922b11573c41a8d |

A read-only tensor-level verification was run inside the frozen T4 container:

~~~text
teacher net state keys            31
exported T4 decoder state keys    31
missing keys                       0
extra keys                         0
non-equal tensors                  0
all decoder tensors equal       true
T4 identity map entries           13
identity shape             [96,50]
teacher has B0 fc_id layers     true
~~~

Thus the decoder serialized inside the submitted T4 payload is bitwise equal to the epoch-34 teacher network and retains teacher.fc_id_in / teacher.fc_id_out. A B0 cached-identity payload can use that exact decoder object without retraining.

T4 runtime closure currently on disk:

| File | SHA-256 |
|---|---|
| sua_exploration/evalai_t4_m2/export_t4_payload.py | 4cd6914869463881811ad987315ecc2e036b97529f285c27929ccf365a01f077 |
| sua_exploration/evalai_t4_m2/t4_spint_decoder.py | fd1d5b203d9c8daccdda2e212c7efd7720f43717dceef876bd136e99bf7224dc |
| sua_exploration/evalai_t4_m2/decode.py | beda41a330e15ea6327cd229e6a3f5458c2c2660425c98da9b2d34a30848f782 |
| sua_exploration/evalai_t4_m2/Dockerfile | 2218fc41183b8ee2fc6ca535c015292374ea493fdf3718d31d0a50d6c5448408 |
| sua_exploration/evalai_t4_m2/submit_evalai.py | 950eff328037a7627dcb57374b936b501bdf327b63e2e8740841df23dd3747f2 |

## Existing local q33 evidence—and its limit

The frozen local receipt reports the following equal-session means on the four M2 calibration sessions that genuinely retain query trials after trial 33:

| Arm | Local M33/q33 equal-session mean R² |
|---|---:|
| epoch-34 B0 | 0.2314276081 |
| T4 seed 42 | 0.2956301500 |
| TS4 seed 42 | 0.2000452500 |

The paired local values are T4-B0 = +0.0642025419 (3/4 positive) and T4-TS4 = +0.0955849000 (4/4 positive). The two sessions containing exactly 33 trials are correctly excluded rather than assigned an artificial score.

This is positive development evidence only. It is neither a hidden EvalAI test nor an existing EvalAI-ready B0 payload.

## Why the current b0_baseline is insufficient

The existing local b0_baseline artifact is a useful teacher-provenance witness, not a formal deployment package.

| Requirement | Current state | Consequence |
|---|---|---|
| Teacher checkpoint | exact epoch-34 alias, same SHA | valid source only |
| Calibration count | 33 | count compatible |
| Calibration selection | random_calibration: true | incompatible with chronological first-33 deployment |
| Query boundary | no q33 runtime/query receipt | cannot prove deployment support/query boundary |
| Cached identities | absent | no B0 EvalAI runtime input |
| Docker image | absent | no immutable B0 image ID to register |
| Host/container parity | absent | output and latency path unresolved |
| Submission guard | existing helper is T4-specific and its state marks 578221 | must not be reused |

The frozen current-local artifact hashes are:

~~~text
checkpoint alias SHA   fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec
metrics CSV SHA        8c316c48768cb33025e7435d113819d88d3866f1f13fb63bed10e7d7c6a17225
resolved config SHA    2497c5e7f91e7c90277f8a6202c7b6961825d23123071b162dfe7037f789e668
checkpoint manifest    7ebb84f095108ead43b31da36b1290a75f0e3a6919ed2fc0e3126337a6f1e9f1
~~~

It must not be relabeled as a formal q33 B0 candidate.

## Frozen repair configuration

Create new isolated files under a new directory such as sua_exploration/evalai_b0_m2_epoch34/. Do not edit the frozen submitted T4 directory or shared training source.

| Field | Required B0 value |
|---|---|
| Task / target phase | FALCON M2 / private few-shot-test-2319, phase 4599 |
| Decoder | epoch-34 teacher SHA fbcb…13ec; require the 31/31 equality check |
| Identity estimator | teacher fc_id_in → mean across 33 trials → teacher fc_id_out |
| Support | public chronological trials [0:33], exactly 33 for all 13 M2 calibration tags |
| Direction / side labels | never read or retained by B0 |
| Local query-boundary verification | query starts at trial 33; every 50-bin history wholly after it |
| Online window / observations | 50 bins; raw; smooth_observations=false |
| Runtime batch size | 7 |
| Behavior scale | 5.0 |
| Parent image | spint-m2:e8-epoch027-76f0fb2, sha256:b179…aaf8 |
| Persistent online state | one cached E_B0[96,50] per tag; 13 tags |
| Runtime adaptation | none: no optimizer, backward pass, refit, raw calibration buffer, or target-label state |

### Normalizer rule

The T4 normalizer d17f…539e belongs only to T4’s four side features. B0 does not have these features, so applying that normalizer to B0 would be meaningless and would violate the intended ablation.

The matched controls are the neural preprocessing, B0 calibration count and chronology, decoder state, online history, behavior scale, base container, and batch runtime. Record the T4 normalizer only as T4-arm provenance; record side-feature normalizer = not applicable for B0.

## Hash closure required before B0 becomes eligible

Before any authenticated preflight, all of the following must exist in a new B0 receipt:

1. SHA-256 values for B0 exporter, runtime wrapper, entrypoint, Dockerfile, payload, image, and independent submit helper.
2. Teacher checkpoint SHA plus a direct state-dict equality audit between the teacher and packaged decoder: 31 expected tensors, zero missing, extra, or unequal tensors.
3. Per-session records for all 13 public calibration tags: source calibration shape [33,100,96], preprocessing declaration, calibration tensor hash, B0 identity shape [96,50], identity hash, and max direct-teacher-vs-cached-B0 output error.
4. A fail-closed audit proving B0 input includes neither a T4 side tensor nor a direction/query target label.
5. The B0 image must use the same base-image ID and declare its own payload / teacher hash labels.
6. A B0-specific submit state file must refuse the T4 image ID and submission 578221. It must not share evalai_t4_m2/artifacts/evalai_push_state.json.

## Required dry-run gates (no submission)

After a B0 package exists but before asking for any submission authorization:

1. **Static export audit:** validate every hash-closure item above; direct teacher B0 and cached identity must be exactly equal on a fixed public calibration-derived neural window for every public session.
2. **Runtime-path equivalence:** compare reset, observation buffering, source = neural.permute(0,2,1)+identity, fc_in, transformer, fc_out, behavior scaling, and batch handling with T4. Any difference must be schema/arm metadata only, never the online numeric path.
3. **Container smoke test:** import/schema/hash checks and synthetic no-scoring prediction through the new container using the common parent image and matching entrypoint arguments.
4. **Public/local parity, only if separately authorized:** compare host and container output tensors on the declared public M2 staging scope. Do not open an organizer hidden test file or query an EvalAI hidden result.
5. **Latency disclosure:** compare staging runtime only under the identical batch and stream settings. Do not infer the official number from local staging; report official Normalized Latency only after a formal result.

B0’s cached identity has the same [96,50] online state and same decoder MAC path as cached T4. Its one-time offline identity calculation, like T4’s one-time export, is not part of online decoder latency. Any claim must consequently say online decoding latency, not end-to-end calibration-plus-decoding latency.

## Minimum repair sequence

1. Add isolated B0 packaging files only; leave shared training code and all frozen T4 submission files unchanged.
2. Export B0 identities from the frozen epoch-34 teacher and the public first-33 neural support. Fail if a T4 feature or direction label enters the exporter.
3. Clone the frozen online algebra into a B0-specific schema/runtime, prove direct-versus-cached equality, and write the closure receipt.
4. Build a new immutable B0 image from the common parent image; record its real image ID rather than assuming it equals T4’s image ID.
5. Complete the dry-run gates above, then freeze a B0 preparation receipt with actual payload and image hashes.
6. Request a fresh explicit authorization before authenticated EvalAI preflight, ECR push, registration, or private submission.

There is deliberately no executable B0 submission command today: no B0 image ID or B0 helper exists yet, so a guessed tag/hash would be unsafe. Only after steps 1–5 are frozen and the user explicitly authorizes an external submission may the B0 preparation receipt add this exact, now fully instantiated command:

~~~bash
/tmp/spint-e8-evalai-py38/bin/python \
  sua_exploration/evalai_b0_m2_epoch34/submit_evalai.py \
  --execute \
  --confirm-image-id sha256:<actual-frozen-B0-image-id>
~~~

The placeholder is intentionally non-executable and is not authorization.

## Scientific qualification

The repository’s later clean-teacher guard identifies epoch-34 teacher fbcb…13ec as a legacy held-out-selected teacher. A matched B0 submission would remove the decoder-version confound in T4 versus B0, but cannot retrospectively turn that legacy source into a clean held-in-selected teacher. Its correct name is therefore a **matched legacy epoch-34 system ablation**, not independent clean-teacher confirmation.

Live EvalAI phase availability, quotas, and team permissions were intentionally not queried because they are mutable external state. Submission 578221 must never be duplicated.
