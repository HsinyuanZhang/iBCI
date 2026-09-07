# H1 QueryAge16 source-capacity gate — prospective execution boundary

Status: preparation/review only. This is one declared member of the already
allowed two-temporal-operator CRST-B4 family, not new formal training authority.
Motivation: the completed fixed dense supervision test did not improve the
full-history H1 score versus its matched CONTROL in either EMA arm. This does
not prove that the temporal operator caused that deficit.

## Model and scientific scope

Both FLAT and ROUTE retain H1's exact existing set-v2-unscaled-dot,
local-FC1-balanced initialization, frontend, readout, bank and mask definitions.
Only both fresh temporal modules change to the existing `QueryTemporalStack`:
W700, width256, heads8, four blocks, FFN512,16 linear age buckets, independently
normalized/projected raw memory and depth-updated current query, without causal
PE. No trained causal weights, M1 shell, signed frontend or shorter W is used.
FLAT/ROUTE retain their sole calibration routing-bias difference.

Factory: `h1_queryage_family_v1.model.make_queryage_localbalanced_pair`.
Names explicitly contain `temporal=FW-QueryAge16` and
`spatial=set-v2-localbalanced`. The model's tested source SHA is
`bef468cf3e7e34f511bd2a94ceab7bcfd53e959475f5ee8f83f3272923e741b9`.

This is a learnability screen, not model selection on minival. It does not
establish the superiority of one temporal member or an improvement over SPINT.
No official or external-development outcome is used.

## Fixed data and update recipe

Use only the13 source-train rows and their16 previously frozen starts each,
exactly208 windows; frozen-ID file SHA:
`da4bf975a5c1023bbffe26da87d0d4977f437d7db1db012283511ef140d96211`.
Source/minival are different recordings. The pre-existing serialized cache
contains both splits; after its complete authority is validated, only train
rows may enter model forwards or updates. Do not claim the minival bytes were
never deserialized, and do not construct/rebuild any cache.

- Seed42, fresh paired initialization, byte-identical non-routing state and
  zero ROUTE gate; actual zero-gate parity and nonzero gate-gradient preflight.
- Sorted13-session cyclic order, one session's full16 frozen endpoints per
  paired update, shared inputs/targets/order across arms.
- AdamW with the existing trusted H1 parameter groups, LR `2e-4`, gradient
  clipping1.0, effective batch16 from four microbatches of4 with loss weight1/4.
- Decoder-raw MSE target equals native velocity×20; score divides predictions
  by20. No dropout, prefix augmentation, dense-position treatment or EMA.
- Record actual source/start/target/order identities, all losses and update
  counts. Scores are RAW source208 native FP64 training-fit summaries only.

## Three bounded stages

1. **smoke20**: disposable fresh paired models,20 real optimizer updates;
   first4 warmup, remaining16 synchronized paired durations. No full training
   selection or minival score. Hard900 s and22 GiB peak allocated CUDA memory.
2. **capacity260**: separate fresh initialization, exactly260 paired updates.
   Requires the smoke's exact current source/model/protocol/code authority and
   forecasts from its actual paired mean: `260×mean×1.5+300≤3600` and
   `1040×mean×1.5+600≤3600` seconds. Hard3600 s/22 GiB for the whole process.
3. **extend1040**: allowed only if at260 at least one arm has pooled source208
   R²≥.10, prediction-std≥.25×target-std, finite losses, and last32 loss mean
   below first32 mean. Resume the exact260 models, optimizers and RNG state;
   finish1040 total updates, not1040 additional updates. Same3600 s/22 GiB bound.

At1040, both arms must have R²≥.50 and prediction-std≥.50×target-std for a
meaningful learnability pass. All arms/scores are retained, including failures.
Failure stops this candidate's screen; pass permits a new review, not automatic
formal training, checkpoint promotion or deployment.

## Execution and artifact requirements

Every process requires an explicit root-reviewed launch authorization binding
its mode, exact source/helper/model/protocol bytes, external input receipts and
absolute fresh output path. Capacity/extension require the matching completed
earlier-stage receipts and checkpoints, with fresh byte revalidation.
Time/resource accounting starts at entry and covers setup, every forward,
backward, optimizer step, source scoring, checkpoint reload and final audit.
Only one leased physical GPU0/1 and one CPU thread are permitted per process.

Checkpoints include RAW models, optimizers, update counts and all RNG state,
are saved atomically to new paths and disk-reloaded for strict verification.
Capture fresh pre/post authority and hash all own final artifacts before the
completion receipt. Preserve every prior experiment, cache, payload, image,
registration and submission. No GPU screen has been authorized by this document
alone; code/actual-artifact preflight and resource smoke are still required.
