# TF-SR Phase-E Terminal Runbook

Owner: root/Sol. Terra remains stopped; Luna remains read-only. This runbook is not a launch
authorization and is not part of the frozen Phase-D or Phase-E implementation closure.

## 1. Wait for the real terminal

Do nothing until the original Phase-D v2 process exits naturally and the canonical training root
contains immutable `terminal.json`, `swa.pt`, checkpoints 44--47, and all 48 epoch receipts. Do not
construct an authority from epoch receipts alone.

Root must then run, in this order:

1. `score.verify_fixed_authorities(root)`;
2. `score.validate_phase_d_training_terminal(root)`;
3. confirm the returned terminal/SWA/closure evidence and confirm GPU1 is free;
4. confirm both Phase-E authority and score roots are absent and non-symlinks;
5. rerun the frozen 71-test no-CUDA scorer suite and recompute closure `2772ec4b...`.

Any mismatch stops the route. Do not repair a terminal or choose another checkpoint.

Pre-terminal readiness rehearsal, 2026-08-21 HKT: root independently reran the current frozen
seed-42 scorer suite under no-user-site, `CUDA_VISIBLE_DEVICES=''`, and plugin autoload disabled;
all 71 tests passed in 19.27 s. The descriptor-built 26-path closure still validates as
`2772ec4baa7b61781b49c0889bd3fe9e8567ffafaf5172171f8fc423ea69b270`, and both governing
Phase-E roots remain absent. Root also exercised the real incomplete-training negative path: all
eight sealed fixed authorities validated, the canonical Phase-D artifact attached read-only, and
terminal validation stopped at the expected absent `terminal.json` without creating either output
root. This is readiness evidence only. Step 5 above remains mandatory after the real terminal/SWA
has been accepted, because current success cannot authorize future bytes.

### Optional early-checkpoint boundary

The active frozen run publishes model checkpoints only at epochs 44--47. Epoch receipts 0--43
contain digests and optimization evidence, not recoverable model weights. Therefore do not export a
live in-memory state, modify the running process, or restart a successor merely to obtain an early
checkpoint.

If an early preview is still operationally useful after `checkpoint-44.pt` appears naturally, it
must use a separately reviewed read-only diagnostic consumer and the same last-bin,
variance-weighted, equal-session estimator as Phase-E. Label that output
`NON_GOVERNING_EARLY_DIAGNOSTIC`; it cannot select a checkpoint, trigger early stopping, change the
48-epoch run, satisfy a Phase-E gate, or replace final-four SWA. The existing Phase-E route remains
terminal/SWA-only. Because epoch 44 leaves only three training epochs, this preview is optional and
must not delay the governing terminal score.

### Concurrent seed-43 accelerated run

Seed 43 was separately operator-launched on GPU0 at 2026-08-21 12:45 HKT while seed 42 continued
on GPU1. Do not stop, restart, merge, or redirect either process. Seed 42 remains the pre-registered
discovery/gating run and its Phase-E score must terminalize first.

Seed 43 is not consumable by the frozen seed-42 scorer: it has its own terminal/checkpoint schemas,
lineage, output root, and accelerated-build-v2 disclosure. Its later scorer is governed by
`WORKORDER_TFSR_SEED43_PHASE_E_SUCCESSOR_20260821.md` (SHA `73ccb4d9...`) and must remain a separate
additive route. No seed-43 authority or score may be published until all of the following hold:

1. seed 43 has a complete immutable 48-epoch terminal, checkpoints 44--47, and final-four SWA;
2. its own frozen validators accept every epoch/checkpoint/SWA and launch equals final closure;
3. the seed-42 Phase-E score terminal already exists and is descriptor-bound upstream;
4. the separate seed-43 scorer has passed root's no-data/no-CUDA audit and received an explicit
   execution authorization.

Interpretation is ordered, not pooled opportunistically. If seed 42 is `CLEAR_GO`, seed 43 is a
confirmatory replication. If seed 42 is `HOLD` or `STOP`, seed 43 may be scored and reported only as
`NON_GOVERNING_POST_AUTHORIZATION_REPLICATION`; it cannot revise the seed-42 verdict. A seed42/43
mean is descriptive and must retain each build label. It is neither a three-seed result nor an
authorization for seed 44.

Seed 43 uses the same model graph, source authorities, loss, optimizer, schedule, budget, dropout,
checkpoint policy, and SWA recipe, but its recurrent inner loop is JIT scripted. Therefore never
write `Only the seed differs` in the result narrative. Report `seed43, accelerated build v2`; retain
the sealed first-step equivalence and 20-step FP32 trajectory-drift evidence. The engineering
throughput receipt is evidence only (`authorizes_training=false`), not the operator authorization.

The first and every later seed-43 epoch is accepted only as a complete immutable receipt, never from
the live log or loss alone. Root must descriptor/sidecar-check the 0444 pair, call seed43's frozen
`validate_epoch_receipt`, and independently confirm:

- exact cell/schema, epoch index, 33,925 steps, and cumulative/global progress;
- observed first/last LR equals both receipt expectations and the frozen schedule;
- finite ordered min/mean/max loss (optimization evidence only, never checkpoint selection);
- exact dropout quantile/count algebra and 1,085,600 population examples;
- all eleven named critical-gradient groups true at the final step only;
- finite full model and Adam state plus two valid state digests;
- launch closure and seed43 lineage exactly equal the durable launch receipt and current bytes;
- positive elapsed/throughput and valid peak allocated/reserved/RSS evidence;
- source-only true, capture diagnostics/scientific/score false, target/formal unopened;
- no simultaneous failure/terminal/SWA before its legal lifecycle point.

Failure of any one item is an immediate root alert and stop condition; a decreasing loss cannot
override missing provenance, gradient, state, or access-boundary evidence.

Post-terminal code-hygiene item: do not edit the live seed-43 closure while training is active.
After its terminal/SWA pair is durably accepted, repair
`test_first_train_step_forward_and_loss_bitwise_equal_gradient_within_frozen_band` so the local
test subtracts each parameter's `.grad` tensors rather than subtracting the `Parameter` values while
merely checking that `.grad` is present. Then rerun the frozen-build test set under a successor
closure and record the repair. This test bug is not evidence against the live run: the sealed
throughput-v2 receipt's gradient comparison was produced by the implementation's actual
`_gradient_snapshot(model)` path and remains the governing engineering evidence.

The former seed-43 replication-addendum candidate at closure `d6ed5dfb...` (source/CLI/test
`4f16bc81...` / `5af8aa97...` / `bda6f3e8...`) is **superseded**. A real metadata-only attach
against the live seed-43 artifact root found that `_attach_readonly_seed43_artifact` called the
nonexistent export `train_43._directory_identity`; its 17 focused tests had mocked that attach and
therefore did not cover the physical seam. This did not invalidate the live run: root accepted
epoch 1 through the training route's native validator and exact frozen train-42 identity helper.

The bounded replacement is **GO for preserving the successor bytes only** at source/CLI/test
`b4259919...` / `5af8aa97...` / `fae2e996...`, explicit 35-path closure
`ab1a200d7df4bdcc4cee564cd2de08368962bf4d5131ded6a6c523fd846a24fa`. It uses only the exact
closure-bound frozen train-42 directory-identity implementation and rejects dependency alias,
canonical-root symlink, and post-attachment inode replacement. Root independently reproduced
19/19 focused no-data/no-CUDA tests, py_compile, closure and dry/fresh-root checks, plus a real
metadata-only live epoch-01 attach and native receipt validation. This does not authorize the
target-free authority pair, evaluation NWB access, CUDA scoring, or result publication. Keep both
prospective seed-43 authority/result roots absent until seed 43 has a fully validated 48-epoch
terminal and final-four SWA, the seed-42 Phase-E input/score/terminal triplet is already durable,
and root issues a later reviewed execution capability.

## 2. Publish the target-free authority

Using only the reviewed `score.py` APIs:

1. derive fixed bindings from `AuthorityMaterial.binding()`;
2. derive the within roster from the fixed strict manifest;
3. call `build_target_free_preflight` with `phase_e_closure(root)` and the closure-bound source
   normalizer constants;
4. reserve the canonical authority root with `issue_root_publication_capability()` and
   `reserve_authority_artifact()`;
5. publish and reload `official_preflight.json` with `publish_target_free_preflight()`;
6. build the root decision from that exact durable preflight SHA;
7. publish and reload `root_authorization.json` with `publish_root_authorization()`;
8. call `verify_phase_e_authorization()` and exact-compare its training, closure, preflight, and
   authorization evidence to the in-memory candidates.

This stage reads sealed metadata and training artifacts only. It must not resolve or open an NWB.

## 3. Run exactly one matched score

After rechecking GPU1 UUID/BDF/memory and the two fresh result roots, use the frozen two-flag CLI with
`CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`, the exact SPINT Python, thread counts fixed to one, and
explicit canonical `SUBC_DATA_ROOT` and `SUBM_DATA_ROOT`. Do not provide a model path, checkpoint
path, session list, or alternative output root.

The scorer must publish/reload `attempt.json` before resolving either data root. It must then replay
Cell D exactly, score TF-SR aligned/zero/wrong-pair, and end in either one immutable terminal or one
honest failure receipt. Never retry into the same root.

## 4. Interpret without changing the gate

- `CLEAR_GO`: external delta at least +0.03, within delta at least -0.03, external median positive,
  and at least 9/15 external sessions positive. Only this branch lets the already-authorized seed43
  run count as a confirmatory replication and licenses a future seed44 run.
- `HOLD`: report the system and session heterogeneity; do not launch rescue ablations.
- `STOP`: close the TF-SR route; do not sweep width, heads, GRU size, window length, or fusion.

Zero/wrong-pair controls and the pooled A2 comparison are explanatory and cannot rescue the primary
Cell-D gate.

For operator readability only, the exact sealed Cell-D means translate the delta gate into these
absolute mean boundaries: external Cell D is `0.4179362749059995`, so the external mean must be at
least `0.4479362749059995`; within Cell D is `0.5696851710478464`, so the within mean must be at
least `0.5396851710478464`. The external paired median and 9/15 sign requirements still apply and
cannot be inferred from these means. Pooled A2 is `0.3460880616472827` external and
`0.5776186750994788` within; therefore exceeding A2 external is not close to satisfying the Cell-D
primary gate. These are derived readings of sealed metadata, not replacement thresholds.

### Claim discipline after the immutable verdict

Every branch must report the same evidence before interpretation: Cell-D replay parity; seed42
aligned within/external equal-session means; paired mean, median, sign count, session deltas, and
frozen bootstrap interval versus Cell D; aligned/zero/wrong-pair controls; the non-gating pooled-A2
context; runtime/resources; and all no-update/no-formal boundaries. Never report only the headline
mean or omit a negative session block.

- After `CLEAR_GO`, the licensed claim is system-level: under the frozen source-only training and
  matched zero-target-update evaluation, the complete causal TF-SR system outperformed the sealed
  Cell-D system while retaining within performance. Do not attribute the gain individually to the
  GRU, state queries, concat fusion, B3S, T4, or dynamic dropout; this run changes the whole system
  relative to Cell D. Seed43 may test replication of the system result but does not retroactively
  isolate a component.
- After `HOLD`, report the signed session distribution and controls as an inconclusive system swap.
  Do not call a positive but sub-threshold mean a win, and do not choose a favorable date block,
  mode, or A2 contrast as a replacement endpoint.
- After `STOP`, report that this frozen TF-SR system failed the practical matched gate. Do not claim
  that all causal/state-conditioned set decoders are disproved, but do close this implementation
  family under the current experiment rather than silently rebuilding it from the same result.

Seed43 always retains the label `accelerated build v2`. If seed42 is not `CLEAR_GO`, its later score
is explicitly non-governing and cannot average the discovery verdict into a pass. If seed42 is
`CLEAR_GO`, seed43 can strengthen or weaken replication confidence, but two runs still do not
license a three-seed estimate or a claim that only the seed differed.
