# Work Order: TF-SR Phase-E Matched Evaluation

Date: 2026-08-19  
Owner of scientific design, review, and authorization: root/Sol  
Implementation owner: Terra  
Runtime monitor: Luna  
Status: implementation may proceed; target access and score execution are not authorized by this document

## 0. Decision first

Build an additive, fail-closed Phase-E evaluator while the source-only TF-SR seed-42 run is training.
Do not edit the running model, source adapter, Phase-C/Phase-D trainer, their tests, the frozen TF-SR
handoff, any sealed receipt, or any existing result. Do not open source, validation, external, or formal
NWB data during implementation or tests. Do not launch a GPU score.

The evaluator may execute only after all of the following are independently accepted by root:

1. the canonical 48-epoch training terminal and final-four SWA exist and pass exact live validation;
2. the Phase-E implementation and adversarial tests pass independent review;
3. a target-free canonical Phase-E preflight pair is minted;
4. a separate canonical root authorization pair binds that preflight;
5. the canonical score output root is fresh.

## 1. Scientific estimand

The governing comparison is one paired seed-42 system contrast:

```text
TF-SR final-four SWA minus sealed Cell D seed42
```

It must use the same 6 within-development sessions, the same 15 external sub-M sessions, the same
M30/query windows, the same source-only behavior/T4 normalizers, the same last-bin query, and the same
variance-weighted per-session R2 implementation. Sessions are equally weighted. No target update,
backward call, optimizer construction, checkpoint selection, or session selection is allowed.

Device/runtime parity is fixed rather than operator-selectable. The sealed Cell D matched table was
produced with `CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`, Torch `2.5.1.post303`, on physical GPU UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` / BDF `00000000:03:00.0` (RTX 3090). The live scorer must
use that exact mapping with no CPU/other-GPU fallback, after Phase-D is terminal and the device is
free. This is an evaluation-only forward route; it does not authorize training or target updates.

The two memory authorities are deliberately separate and must not be rounded or coerced into one
shared MiB field. On this exact physical GPU, `nvidia-smi` reports the nominal field **24576 MiB**,
whereas `torch.cuda.get_device_properties(0).total_memory` reports exactly **25438126080 bytes**.
Both literal values must be bound by the preflight, root authorization, live device check, resource
receipt, and adversarial tests. A comparison between Torch's floor-MiB value and the nominal
`nvidia-smi` field is forbidden.

The evaluator must replay Cell D on the live evaluation tensors before interpreting TF-SR. Every Cell D
session score, window count, and surface mean must exactly reproduce the sealed governing table. A
failure to reproduce Cell D is a metric/data/runtime mismatch and stops the result; it is not a TF-SR
failure.

## 2. Fixed authorities

The implementation must bind and descriptor-safely verify these canonical artifacts. A path copy,
alias, symlink, mutable replacement, wrong mode, wrong SHA, or malformed sidecar fails closed.

| Authority | Canonical path | Expected SHA-256 |
|---|---|---|
| TF-SR source smoke | `tfpd_exploration/results/tfsr_b3st4_ddrop_v1/source_smoke_receipt.json` | `022ca7a253e208c86c846b593bc684372ac9ab21197db7a974277416df90dfbc` |
| Strict-27 manifest | `sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json` | `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9` |
| A2 matched-score reference | `tfpd_exploration/results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json` | `0ae0f74c6b5d606599d1108378836c8cbb7d1e5df05a453aa83f682255734d4e` |
| Cell D matched table | `tfpd_exploration/results/sparsification_step0_v1/step0_receipt.json` | `92b5cef8c3fd1a023b1c560c95cfc13af739cb948351a4231d7be41bea59fdca` |
| Cell D terminal | `tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json` | `b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442` |
| Cell D SWA | `tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt` | `626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd` |
| External asset ledger | `sua_exploration/results/dandi_000688_subm_co_schema_preflight_v2/receipt.json` | `1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283` |
| External scope manifest | `sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json` | `68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55` |

The accepted source normalizer semantics are fixed:

```text
T4 side normalizer semantic SHA  293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0
behavior normalizer semantic SHA f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391
source normalizer authority body   2632c6a6a4cfb8a4c0fb2b23e0cc8205ea323240b376903c60d5b27110f59e43
accepted Stage-0 closure           0e7154000d670e06edeee19e6c57dfe5a5ae7606e993f62650b83b906d858ae4
accepted Phase-C closure           2be132f3f3e4f8a232e40c02da62f9e34bc9d43966f6279ce3f4e9f05be0b1bc
```

The TF-SR training terminal and SWA SHAs are deliberately not guessed here. The live preflight must
load them from the canonical training root, validate the exact Phase-D schema and all upstream
bindings, reload all 48 epoch receipts and the four checkpoint pairs, and independently verify the
SWA state digest before producing any authorization input.

## 3. Canonical topology

New files only:

```text
tfpd_exploration/src/tfsr_b3st4_ddrop_v1/score.py
tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_score.py
tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_score.py
```

"New files only" is an ownership rule, not a three-file runtime closure. The Phase-E launch closure
must explicitly bind this work order, all three new files, and every repository Python/runtime file
actually imported by the live no-cache adapter, Cell D reconstruction, TF-SR reconstruction, and
held-FD NWB parser. Explicit paths are required; a recursive source glob is not an acceptable
authority. Installed Torch and TorchMetrics are bound separately by exact versions, the two physical
GPU memory authorities above, and the mandatory exact Cell-D live replay; they are not silently
treated as repository-relative source files.

Canonical authority root:

```text
tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_score_authority_v1/
  official_preflight.json
  official_preflight.json.sha256
  root_authorization.json
  root_authorization.json.sha256
```

Canonical score root:

```text
tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_matched_score_v1/
  attempt.json
  attempt.json.sha256
  input_authority.json
  input_authority.json.sha256
  score.json
  score.json.sha256
  terminal.json
  terminal.json.sha256
  failure.json                 # only on failure after a durable attempt
  failure.json.sha256
```

All publications use transactional `O_EXCL`, full writes, file and directory `fsync`, exact `0444`
body/sidecar pairs, owned-inode rollback, and canonical-path/no-symlink checks. The execution route
must reject a body-only or sidecar-only collision before any data path is resolved.

## 4. Ordering and access boundary

The public CLI defaults to a static dry plan: no Torch import, no dataset import, no path discovery,
no model load, no data open, no CUDA call, and no write. Execution requires two explicit flags plus
the exact canonical preflight and root-authorization pairs.

Live ordering is mandatory:

1. validate the canonical training terminal/SWA and every non-data authority;
2. verify the Phase-E launch closure and fresh canonical score topology;
3. publish and reload immutable `attempt.json`;
4. only then resolve source/within/external data paths;
5. open every evaluation NWB with `O_NOFOLLOW`, verify the held descriptor's regular-file identity,
   size, and SHA before an NWB parser sees it, and keep that descriptor or an equivalent verified
   private snapshot alive through parsing;
6. publish and reload `input_authority.json` before model forward;
7. score Cell D and exact-compare the sealed table;
8. score TF-SR aligned, zero-T4, and wrong-pair-T4 modes;
9. verify every held input descriptor and model state again;
10. publish/reload `score.json`, revalidate all upstream pairs and the final live closure, then publish
    `terminal.json`.

The six formal sub-C test session names remain inert. Resolving or opening one is a hard failure. The
external roster is exactly the 15 eligible rows in the fixed v2 asset ledger; every local file must
match the ledger's asset ID, canonical session name, bytes, and SHA before parsing.

Schema warning: in the existing v2 receipt, the field named `eligible_session_ids` contains eligible
**asset UUIDs**, not canonical session strings. The scorer must join each UUID to exactly one eligible
`asset_disposition_ledger` row, then take that row's `session_id`; treating those UUIDs as session
names is forbidden. The 15 derived session names must exact-match the A2 matched reference roster
before any path is formed.

## 5. Exact evaluation tensors and T4 capability

Use a no-cache, read-only evaluation adapter. It must reconstruct the same `SessionRecord` semantics
as the established A2 matched scorer:

```text
signal view                    SUA
calibration                    first 30 chronological rewarded trials
query                          all authorized post-calibration validation/external windows
window                         50 bins
bin size                       20 ms
padding                        -1.0
behavior normalization         strict-27 source-only
T4 normalization               strict-27 source-only
behavior_scaling_factor        None / output scale 1.0
```

For every session, independently recompute raw M30 T4 from the held verified asset, normalize it with
the source-only mean/std, and exact-compare it to the model-visible side tensor produced by the
evaluation adapter. Build a typed `NormalizedT4Batch`; never pass a bare side tensor to TF-SR. The
capability must bind:

- input asset path/bytes/SHA and held-descriptor identity;
- session name and canonical ordered channel IDs;
- ordered-unit digest and unit count;
- raw T4 dtype/shape/bytes digest;
- normalized T4 dtype/shape/bytes digest;
- source normalizer authority and strict-27 roster digest;
- calibration bytes, query-start bytes, neural bytes, behavior bytes, valid-mask bytes, and target
  last-bin bytes.

Record those compact digests in `input_authority.json`; do not put full arrays in JSON.

Controls are post-normalization and change only the T4/unit correspondence:

```text
aligned     exact normalized T4 on the receiving unit axis
zero        exact zeros_like(aligned T4)
wrong_pair  deterministic cyclic shift by one unit; no fixed point; same value multiset
```

Reject sessions with fewer than two units. Record the exact permutation vector digest, prove it is a
derangement, and recompute B3S for each mode. Do not reuse aligned B3S output in a control.

## 6. Model and scorer semantics

TF-SR:

- load only the canonical Phase-D `swa.pt` through its existing strict validator;
- instantiate `TFSRDecoder(capture_diagnostics=False)`, strict-load, set `.eval()`;
- require no dropout draw, all-one evaluation gain/survivor masks, finite `[B,50,2]` output;
- require a repeated fixed-batch aligned forward to be bitwise equal;
- hash model state before and after every surface/mode and require equality;
- require every parameter gradient to remain `None`.

Cell D parity anchor:

- load the canonical Cell D SWA read-only and strict-load its exact graph. The sealed SWA itself
  contains `UninitializedParameter` objects, so plain `torch.load(..., weights_only=True)` is known
  to reject it. Keep `weights_only=True` and use a narrowly scoped
  `torch.serialization.safe_globals([torch.nn.parameter.UninitializedParameter])` context only
  around this already SHA-verified in-memory Cell-D payload. `weights_only=False`, a broader global
  allowlist, pathname loading before SHA verification, or persistent mutation of global safe-state
  is forbidden;
- hash the exact Cell D state with a lazy-safe canonical rule. The sealed graph contains dead
  `UninitializedParameter` entries (`decoder.fc_id_in.0.weight` and `.bias`); these must contribute
  their sorted key plus an explicit `uninitialized-lazy` sentinel and must never be detached,
  materialized, dropped, or initialized merely to make hashing succeed. Initialized tensors retain
  the normal dtype/shape/bytes binding. The same rule is used before and after every forward;
- score it once on each live aligned surface with the same batch size and metric path;
- require every session name, window count, R2 value, and equal-session mean to exactly match the
  sealed `step0_receipt.json` governing table before any TF-SR gate is interpreted.

Cell D resource accounting is also lazy-safe. The sealed Cell-D terminal is the authority for
`trainable_parameters = 3510842` and `swa.manifest.uninitialized_lazy_tensor_count = 2`. The live
scorer must independently sum `numel()` only over initialized parameters and require that total to
equal 3510842, while separately requiring and reporting the exact two dead lazy keys
`decoder.fc_id_in.0.weight` and `decoder.fc_id_in.0.bias`. It must never call `numel()`, `detach()`,
or inspect shape on an `UninitializedParameter`; it must not materialize the lazy entries merely for
resource accounting. Dropping the lazy keys, changing their topology, or treating the two lazy
objects as zero parameters without the separate disclosure fails closed.

Both systems use:

```text
governing metric  torchmetrics R2Score(multioutput="variance_weighted")
governing query   final bin of each valid 50-bin window
aggregation       one R2 per session, then unweighted session mean
diagnostic        full-window valid-bin R2, one per session, equal-session mean
batch size        128
autograd          disabled for the complete scoring lifecycle
```

Accumulate predictions and targets per session before computing R2 so chunk boundaries cannot change
the estimator. Controls are diagnostic and cannot rescue the primary aligned gate.

The matched A2 reference is contextual, not a second live model pass. Read the already sealed
`A2_t4_pooled_within` and `A2_t4_pooled_external` per-session tables from the fixed A2 matched-score
receipt, exact-reproduce their equal-session means, and report TF-SR-minus-A2 paired statistics. Do
not reopen an A2 checkpoint and do not let the A2 contrast alter the Cell-D gate. This compares the
seed-42 TF-SR screen with the matched three-seed pooled A2 reference and must be labelled accordingly.

## 7. Required result and decision

`score.json` must contain, for both within and external surfaces:

- per-session and equal-session-mean Cell D aligned governing/full-window scores;
- per-session and equal-session-mean TF-SR aligned governing/full-window scores;
- TF-SR zero and wrong-pair governing/full-window diagnostics;
- exact query/window counts and all compact input/output digests;
- TF-SR-minus-D paired deltas in sorted session order;
- the sealed matched A2 pooled per-session tables and TF-SR-minus-A2 contextual paired statistics,
  explicitly non-gating and labelled seed42-versus-A2-three-seed-pooled;
- mean, median, positive count, min, max, exact sign pattern, all deltas, and fixed-seed-42 10,000-draw
  paired-session bootstrap interval;
- aligned-minus-zero and aligned-minus-wrong diagnostic contrasts, explicitly non-rescuing;
- model parameters, analytic MACs, persistent state, training peak from the accepted terminal, score
  peak memory, and measured aligned latency;
- zero target optimizer/backward/update evidence and exact state-before/state-after equality.

The terminal must compute exactly one predeclared verdict:

```text
CLEAR_GO
  external mean(TF-SR - D) >= +0.03
  within   mean(TF-SR - D) >= -0.03
  external median(TF-SR - D) > 0
  external positive sessions >= 9/15

HOLD
  external mean delta in [0,+0.03), or a positive mean is minority-driven,
  provided the STOP conditions do not apply

STOP
  external mean delta < 0, or within mean delta < -0.03
```

Precedence is `STOP`, then `CLEAR_GO`, otherwise `HOLD`. Zero/wrong-pair results, A2 values,
full-window diagnostics, latency, or resource measurements cannot alter this verdict. Seed 42 remains
a screen and cannot establish multi-seed superiority.

## 8. Failure honesty

After `attempt.json` exists, every exception path must publish one immutable failure pair unless a
terminal already exists. It records the exact stage, whether source/within/external/formal data were
resolved/opened, number of model forward calls by system/surface/mode, whether any backward or
optimizer operation occurred, and a traceback digest. A failed score is never a scientific result.

No partial `score.json` may be published. A stream or sidecar publication failure must either roll
back only files created by that transaction or leave a separately valid failure pair; it must not
leave an apparently complete terminal topology.

## 9. Implementation tests required before review

All tests are no-data, no-CUDA, no-write-to-canonical-root mocks or synthetic tensors. They must cover:

1. zero-argument dry plan imports no Torch/data modules and touches no canonical result;
2. either execution flag alone fails before authority/data/model imports;
3. missing/incomplete training terminal, wrong SWA SHA/state digest, checkpoint-binding drift, or
   launch/final closure drift fails closed;
4. alias/symlink/copy/mode/SHA/sidecar/rename-swap attacks on every authority fail;
5. output body-only and sidecar-only collisions fail before data-path resolution;
6. no target path helper is called before a durable/reloaded attempt;
7. exact 6/15 roster enforcement and formal-name rejection;
8. forged external ledger row/path/size/SHA or held-FD identity drift fails;
9. raw-to-normalized T4 exact comparison, ordered-unit binding, cyclic derangement, zero control, and
   recomputed-B3S proofs;
10. TF-SR strict SWA reload, eval/no-mask/repeat/state/gradient invariants;
11. Cell D exact replay mismatch blocks the TF-SR verdict;
12. last-bin and full-window metric equality against dense synthetic references;
13. paired statistics and all boundary/precedence cases for CLEAR_GO/HOLD/STOP;
14. injected failures at data authority, Cell D, each TF-SR mode, score publication, and terminal
    finalization produce honest failure semantics;
15. launch-versus-final implementation closure drift fails before score or terminal publication.
16. the nominal `nvidia-smi` 24576-MiB authority and exact Torch 25438126080-byte authority are
    checked independently; substituting, rounding, floor-MiB-converting, or swapping either one
    fails before any evaluation data path or model forward.
17. a real synthetic Cell-D graph containing its two dead `UninitializedParameter` entries passes
    lazy-safe state hashing without materializing them; changing an initialized tensor changes the
    digest, changing the lazy-key/sentinel topology fails, and before/after-forward state equality
    remains enforceable. A mock model with no lazy parameters is not sufficient evidence.
18. the same real Cell-D graph reaches final resource disclosure without calling `numel()` on either
    lazy parameter: initialized live parameters equal the sealed 3510842 authority, the exact two
    lazy keys and sealed lazy count are reported separately, and lazy-key/materialization/count
    drift fails closed. This test must cover the actual final resource-disclosure helper rather than
    only the state-digest helper.
19. a synthetic in-memory `weights_only=True` checkpoint containing an `UninitializedParameter`
    fails under the ordinary restricted loader but succeeds through the exact Cell-D loader's local
    one-class safe-globals context; strict graph loading retains the lazy object. The test must also
    prove `weights_only=False` is never requested and the allowlist is neither broadened nor left
    installed after the load. Root separately repeats this check against the sealed Cell-D SWA
    during independent review; implementation tests must not require opening that result artifact.

## 10. Ownership boundary

Terra owns only the three new files in Section 3. Terra is not alone in the repository and must adapt
to concurrent files without reverting anyone else's work. Terra must not edit:

- `model.py`, `contract.py`, `source_smoke.py`, `train.py`, or package `__init__.py`;
- any existing script or test;
- `HANDOFF_TASK_FRAME_STATEFUL_READIN_20260819.md` or this work order;
- any file under an existing result, checkpoint, data, cache, or manifest root.

Terra may run only isolated synthetic/no-data CPU tests. The public evaluator must remain dry and
non-authorizing. Root performs the independent code/provenance review and is the only role that may
mint the target-free preflight, mint root authorization, or authorize a live score after training
terminal acceptance. Luna continues read-only training monitoring and does not inspect or modify the
Phase-E implementation.
