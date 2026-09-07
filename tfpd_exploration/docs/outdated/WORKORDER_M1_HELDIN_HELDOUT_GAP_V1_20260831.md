# M1 Held-In vs Held-Out Fold Paired Gap V1

## Purpose and boundary

The SPINT paper reports Falcon-M1 held-in `0.77±0.03` versus held-out
`0.78±0.04`, i.e. a negligible cross-session gap on M1.  This cell produces
OUR OWN paired number under the frozen CS-WG line: the SAME frozen
source-only decoder, the SAME native M10 calibration budget, and the SAME
governing metric, scored on

* its training (held-in) sessions `20120926`, `20120927`, `20120928`, and
* the fold-defined held-out session `20120924` (the outer target of the
  frozen fold-0 producer; the accepted receipt
  `cross_session_worst_group_m1_fold20120924_heldin_score_v1` calls this same
  session "held-in" in the held-in-calib *split* sense — both namings refer
  to the identical frozen receipt and are recorded side by side here).

It is a descriptive paired readout, not a formal benchmark verdict, not an
EvalAI claim, and not a target-training route.  No target optimizer step,
backward, or update is permitted anywhere.

## Immutable producer and anchor

Sole decoder: the terminal-pinned no-SWA CS-WG fold-20120924 producer

```text
tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa/fold_20120924_cswg
terminal.json                         efd084573843e05ada6c28c050c28e8bbf01976bec443d864bc3a0a7921afdc5
best checkpoint state                 d5d86325e21b5a257a44ee2eff4a9e35ddba9d1d6b0de591e607538bbbb41db7
selected_checkpoint_role              best_source_train_loss
```

Sole anchor receipt (the frozen held-in/split score of session `20120924`):

```text
root                                  tfpd_exploration/results/cross_session_worst_group_m1_fold20120924_heldin_score_v1
input_authority.json                  342c78953a5e7070e3168db7c7644f9d543ee36459243bc9323fe815ce8a9a00
score.json                            d5a08db493ced3c7284fc3d8b295c9e9609326658cf7deb9ef49f9826d6bb605
terminal.json                         0db36a9a5ee314cf4904b80449febe393ff107e510369b0f2b176b9945caa7e5
governing_r2                          0.5679166316986084
prediction_sha256                     781bf769021a67aaedd8839b51b90ee65203e28a8ded366612e51fb8c85f336f
target_sha256                         e913d03a972154a4a7ad3eae9963174fb54dbeefd3bc32b542b5cb76bc0ff8aa
calibration_sha256                    3dfabe28ff6bfdd90f9f866fe6ed9e406a10cd66944b9cc11a54fd8831c4c3cc
n_windows / forward batches           54849 / 429
```

The new scorer must reproduce the anchor EXACTLY (field-wise input digests,
`n_windows`, `governing_r2`, `prediction_sha256`, forward-batch count) on
session `20120924` before any other session's score may be accepted; any
mismatch fails closed.

## Fixed scientific surface

* Graph: frozen M1 `SpintModel`, window 100, 64 units, 16 outputs, native
  M10 calibration `[10,1024,64]`, sealed best source-loss checkpoint.
* Input path: closure-bound direct `FalconDataModule.prepare_session_data`
  plus `FalconDataset` per session, with the session's own calibration rows
  only; no cross-session calibration substitution, no normalizer fit/refit,
  no source refit, no target optimizer/backward/update.
* Evaluation: `model.eval()`, `torch.no_grad()`, dynamic dropout off, batch
  128, final-bin rows only, governing metric
  `torchmetrics.regression.R2Score(multioutput='variance_weighted')` over all
  valid final-bin rows of each session.  Labels enter only the metric.
* Budget: exactly one budget, the native `M10_native`
  (`calibration_trials=10`), identical on both arms.
* Scored sessions and arms (pre-registered):
  * `heldout_fold` arm: `20120924` (fold-0 outer target of this producer).
  * `heldin_training` arm: `20120926`, `20120927`, `20120928` (the
    producer's exact training sessions).
  * Order: `20120924` first (anchor), then `20120926`, `20120927`, `20120928`.
* Fold bookkeeping (transparency only, not arm assignment): the line's data
  configs also define fold-1 target `20120926` and fold-2 target `20120927`;
  for THIS producer those two sessions are training sessions, so they belong
  to the `heldin_training` arm and are recorded with a
  `also_fold_target_of` marker.  No other frozen fold checkpoints exist.

## Blocked surface (recorded, not silently substituted)

The official M1 held-out-calib sessions (`20121004`, `20121017`, `20121024`)
are NOT scored: the frozen recipe pins `official_held_out_sessions_forbidden
= 3`, and the program's sealed audits show each local held-out-calib file
has exactly 10 trials with no query rows after the `[0,10)` support
(`sua_exploration/docs/HELDOUT_BP_FREE_PUBLICATION_EVIDENCE_MATRIX_20260805.md`,
verdict D withdrawn; `sua_exploration/docs/C1_TO_NATIVE_MUA_CLAIM_BRIDGE_AUDIT_20260804.md`).
The paired held-out side is therefore the fold-defined held-out session
above, which is locally available under the sealed four-session metadata
authority.

## Pre-registered statistics and verdict (decided before any scoring)

* Equal-session aggregation: unweighted mean of per-session R² within each
  arm; sd with `ddof=0` (program convention).
* Gap definition: `gap = heldout_fold_mean - heldin_training_mean`
  (negative = decoder worse on the held-out fold session).
* Session bootstrap: `numpy.random.default_rng(42)`, 10,000 draws,
  percentile 2.5/97.5 CI; each draw resamples sessions with replacement
  independently within each arm (`rng.integers(0, n, n)` per arm, held-in
  arm drawn first).  Pre-registered limitation: the held-out arm has a
  single fold-defined session for this producer, so its within-arm
  resampling is degenerate and the CI quantifies training-session sampling
  variability only.
* Verdict rule (ordered, first match wins):
  1. `|gap| <= 0.03` and `ci_low >= -0.03` and `ci_high <= 0.03` →
     `NO_HEADROOM`;
  2. `gap < -0.03` or `ci_high < -0.03` → `HEADROOM_PRESENT` (direction
     `HELD_OUT_WORSE`);
  3. `gap > 0.03` or `ci_low > 0.03` → `HEADROOM_PRESENT` (direction
     `HELD_OUT_BETTER`);
  4. otherwise → `INDETERMINATE` (point gap outside the band but CI crosses
     it).

## Supporting diagnostic (read-only quote)

The frozen-weights activity headroom receipt
`results/m1_h1_activity_headroom_v1/m1_fold20120924.json`
(sha256 `5ea74d3131f0b3bfc4b757ca29748285e48b4978cfdd7b07454f380ea48670d8`,
same fold-20120924 surface, matched-ERM checkpoint) reports
`STATIC_SUPPORT` R² `0.5707439184188843` versus causal
`CAUSAL_GROWING_CAP30` R² `0.5933129191398621`: cross-session calibration
*activity* headroom exists on this surface even while the paper-level
session gap is claimed negligible.  It is quoted only as context; it is not
part of the paired gap or its verdict.

## Lifecycle and receipts

Ordered result lifecycle, one fresh root
`tfpd_exploration/results/m1_heldin_heldout_gap_v1`:

```text
attempt -> launch -> input_authority_<session> -> score_<session> (x4)
        -> paired_table -> terminal
```

`attempt` precedes every NWB descriptor resolution, parse, checkpoint tensor
load, model construction, CUDA initialization, and forward.  All leaves are
immutable `0444` nlink-one body/canonical-sidecar pairs.  Any exception
after `attempt` publishes an immutable `failure` with factual progress.  The
per-session `score_<session>` receipt binds input digests, selected
checkpoint/proof, model state before/after equality, final-bin R², and zero
target optimizer/backward/update.  The public CLI is dry and cannot issue a
capability; only an in-process root-reviewed caller may execute.

## Mandatory no-data tests

Synthetic tests must cover: producer graph tampering fail-closed; anchor
receipt tampering/rehash fail-closed; exact anchor-reproduction enforcement;
session/arm roster and fold-target bookkeeping; attempt-before-backend
ordering; immutable success and failure lifecycles; zero target updates;
paired statistics and pre-registered verdict branches on synthetic arrays;
closure drift; public dry CLI inertness; and no CUDA initialization.
