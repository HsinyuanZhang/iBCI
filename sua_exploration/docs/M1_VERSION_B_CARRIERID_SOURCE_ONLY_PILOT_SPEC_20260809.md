# M1 Version-B CarrierID pilot: source-only LOSO specification

**Status:** audit and executable specification only.  No GPU cell, formal
held-out endpoint, EvalAI submission, or minival file access is authorized by
this document.

## 1. Audit result and why the old anchor was not reproducible

The strict M1 fold-0 **anchor** checkpoint/config pair is unchanged:

- checkpoint: `streaming_calibration_exp/logs/train/runs/2026-08-06-17-21-50-020564_rid-m1_afc4_emg_full_fold0_q3_dev12_f0_s42/checkpoints/best_ckpt/epoch_011.ckpt`
  (`sha256=16b2500e01c734d445b9464edadbc7809442b0c8fe6cc74686aed45f2de21bf4`);
- the strict post-M10 source-LOSO config used by the anchor has SHA-256
  `7b458f68dd01ab2972a65acbd0b076e1bf100b8b78051f1e71cc476bfd51afc7`;
- the sealed endpoint is held-in-calib source LOSO fold 0, support trials
  `[0,10)`, query trials `[10,210)`, and `26496` scored windows.

The first three forward replays used a plain `DataLoader` and differed from the
sealed score by approximately `-5.95e-5`:

| replay | device | reproduced R2 | difference from sealed |
|---|---|---:|---:|
| `m1_identity_headroom_anchored_v1` | CPU | 0.6622288141 | -5.9315e-5 |
| `m1_identity_headroom_anchored_v2` | CPU | 0.6622285843 | -5.9545e-5 |
| `m1_identity_headroom_anchored_v3_gpu` | CUDA | 0.6622285247 | -5.9605e-5 |

This is not a model or hardware drift.  The sealed Lightning path uses the
data module's `SessionBatchSampler` and `test_step`; the replays did not exactly
reproduce that sampler/endpoint path.  The exact-sampler replay is now closed:

`sua_exploration/results/m1_identity_headroom_anchored_v4_gpu_exact_sampler/result.json`

reports `reproduced_r2=0.6622881293296814`, exactly equal to the sealed score,
`difference=0`, `samples=26496`, unchanged model state, and zero optimizer or
backward steps.  Its endpoint declares `formal_heldout_opened=false` and
`minival_values_used=false`.  The v1-v3 numbers must not be used as a scientific
effect or as a reason to loosen the `1e-6` anchor tolerance.

The zero-identity diagnostic in v4 is `R2=-1.1862240941` and
`full-zero=+1.8485124896`.  This is an identity-**reliance** diagnostic only,
not an attainable headroom bound or a new formal-held-out result.  It rejects
the premise that M1 ignores identity, but it does not predict the B-C gain.

## 2. Data-path correction required before the pilot

The existing `src.data.falcon_emg_afc4_datamodule.M1EMGAFC4DataModule` is **not
safe as the pilot data module**.  Its `setup()` calls `FalconDataModule.setup()`
first.  The base setup discovers and reads public `held-in-minival/*.nwb`
files, after which the subclass replaces the validation dataset with the
held-in-calib LOSO dataset.  Existing AFC4 result receipts correctly say that
minival values were not used for the score, but they do not make the file access
disappear.

The pilot must use a new isolated source-LOSO module (for example
`M1EMGAFC4SourceLOSODataModule`) that does **not** call the broad base `setup()`.
Its contract is:

1. Resolve exactly the four allow-listed native `sub-MonkeyL-held-in-calib`
   files.  Reject paths containing `minival`, `held-out`, `formal`, `EvalAI`, or
   `test` and reject any symlink escaping the directory.
2. For fold 0, train on `ses-20120926`, `ses-20120927`, and
   `ses-20120928`; use `ses-20120924` only for the local held-in-calib query.
   Do not infer the split from a glob.
3. Build the source training dataset and target query dataset directly from
   those held-in-calib records.  The target query windows begin at trial 10 and
   use the exact `[10,210)` endpoint and the exact `SessionBatchSampler`
   semantics used by the v4 anchor (the session-local incomplete batch is
   dropped).
4. For `Full`, construct `SourceFrozenEMGAFC4Plan` from source sessions only,
   then add the left-out held-in-calib session.  The target carrier fit reads
   only target support trials `[0,10)`; the query neural/EMG/behavior values are
   read only by the evaluator after the carrier is frozen.  The source plan may
   reuse `falcon_emg_afc4_features.py`, which already enforces the M10 closed
   form and source-only PCA/normalizer.
5. `Zero4` must return four exact zeros without invoking the target carrier fit.
   The plan/receipt must still bind the same source PCA/normalizer hashes as
   `Full`, so the width control cannot be a separate estimator.
6. The receipt must distinguish `minival_files_opened=false` from
   `minival_values_used=false`; both must be false.  It must also record the
   held-in-calib source file hashes, source/target trial ranges, query-window
   checksum, sampler checksum, and the fact that the held-in query is read by
   validation/reporting but is not used for optimizer steps, early stopping, or
   checkpoint selection.

This is a development LOSO endpoint, not the consumed formal M1 test scope.

## 3. Three matched arms

Run exactly one pilot cell first: fold 0, seed 42, source-only train and the
strict held-in-calib query above.  Every arm uses the same teacher checkpoint,
optimizer, seed, sampler, 12-epoch budget, query boundary, and fixed final
checkpoint (`epoch_011`, or an explicitly equivalent fixed-last checkpoint).
Do not select the checkpoint from target query R2. Per-epoch validation on the
left-out session is disabled (`limit_val_batches=0`, zero sanity steps); the
held-in-calib query is evaluated once after the fixed epoch-11 source terminal.

### H-S-continuation: original SPINT identity plus decoder

- `StreamingCalibrationLitModule`, `variant=B0` (or `B1` only if a receipt
  explicitly records the streaming-vs-batch equivalence), `side_dim=0`;
- copy the teacher's original `fc_id_in/fc_id_out` and decoder, then set
  `freeze_decoder=false`;
- source training uses `loss_mode=task_only`, `lambda_y=0`, `lambda_E=0`,
  Adam `lr=1e-4`, `weight_decay=0`, and fixed 12 epochs;
- no target descriptor, target label, or target optimizer step.

`B0` is the exact original SPINT identity algebra.  `B3` or a zeroed `B3S` must
not be called B0: those are compact identity encoders.

### B-C0: compact matched zero carrier

- `StreamingCalibrationLitModule`, `variant=B3S`, `hidden_dim=64`,
  `side_dim=4`, no electrode table, and `freeze_decoder=false`;
- the source/target data path supplies exact normalized `Zero4` rows;
- all other settings are byte-for-byte the B-C arm.

The existing B3S encoder profile is `80,676` static identity parameters,
`64,339,968` MAC/session at M10, `24,576` support-state bytes, and `417,792`
peak live bytes.  These values are a reference, not a replacement for the
new pilot receipt.

### B-C: compact CarrierID with the EMG-general AFC4 carrier

- the same B3S topology and source training as B-C0;
- the side rows are source-frozen q=3 EMG AFC4 `[w1,w2,w3,b]`, normalized with
  the source-session Full normalizer and attached to the correct target
  channels;
- target fitting is one closed-form ridge solve with `lambda=1`, using the
  first ten target support trials only; no target behavior-direction/kinematic
  label and no target backpropagation are allowed;
- the first three coordinates are the source-frozen EMG PCA scores and the
  fourth is the unpenalized baseline rate, as fixed in
  `M1_EMG_AFC4_FEASIBILITY_AND_MINIMAL_BLUEPRINT.md`.

This is an EMG-general functional carrier, not circular T4 and not the old
categorical D4.  It is still a supervised *calibration signal* in the broad
sense that it uses the support EMG recording; it must not be advertised as a
label-free carrier.

## 4. Fairness and hash checks

`freeze_decoder=false` is required for all three arms.  The original decoder
must not be frozen for H-S while being retrained for B-C0/B-C: that would mix a
carrier effect with a decoder-adaptation effect.  The optimizer in
`StreamingCalibrationLitModule.configure_optimizers()` already checks that the
unfrozen student contains exactly all decoder and encoder parameters.

Before the first optimizer step, the pilot writer must record:

- teacher checkpoint SHA and decoder `state_dict` SHA;
- per-arm decoder-before SHA (all three must match the copied teacher);
- seed, source-session order, source batch-sampler order/checksum, and epoch
  budget;
- B-C0/B-C initial model-state SHA and B-C model-state SHA (their active B3S
  weights must match under the common seed; the B-C side columns are
  zero-initialized by the existing B3S constructor);
- target carrier raw/normalized SHA, source PCA/normalizer SHA, and query
  window SHA;
- after-score model-state SHA and explicit `optimizer_steps=0` for target
  evaluation.

The common initialization teacher for the proposed three arms is the separate
source-only M1 decoder checkpoint
`streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42/checkpoints/best_ckpt/epoch_018.ckpt`
(`sha256=f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be`). Its
model/decoder parameter count is `15,007,496`; the M1 SPINT identity path
accounts for `5,350,500` parameters.  The B3S static identity path is `80,676`
parameters (the fresh CPU preflight below verifies these values), so the
identity-path reduction is about `66.32x`.  The resulting initialized student
counts are `20,357,996` for H-S/B0 and `15,088,172` for either compact B3S
arm.  Whole-model counts and MACs must be reported separately; a compact
encoder does not make the decoder free.

The current CPU preflight receipt is
`sua_exploration/results/m1_version_b_preflight/receipt_v2.json`.  It hashes the
four allow-listed source files, maps the trusted teacher to CPU, verifies that
all three decoder state hashes equal the teacher, verifies byte-identical B3S
initialization for B-C0/B-C under seed 42, and validates the terminal
single-query policy (`limit_val_batches=0`, zero sanity steps, no monitored
checkpoint, save only epoch 11, and `optimized_metric=test_heldin/r2_mean`).
Synthetic regression tests are in
`sua_exploration/tests/test_m1_version_b_source_loso_scope.py` (six pass; they
do not open NWB files).  A separate live source-scope smoke replay opened only
the four held-in-calib NWBs and produced `158,487` source windows and `26,517`
target query windows for fold 0; no minival/held-out path was touched.

## 5. Frozen pilot gates

These are one-cell development gates, not claims of statistical significance:

1. **Validity gate:** all three arms must have identical source/query/sampler
   receipts, no minival/formal access, and fixed final checkpoints.  Otherwise
   the pilot is invalid and is rerun only after a reviewed receipt repair.
2. **Carrier gate:** `B-C - B-C0 >= +0.03 R2` on the pooled fold-0 query.  A
   negative or smaller delta stops the Version-B M1 branch; do not add seeds,
   folds, widths, FiLM paths, or alternate carriers to rescue it.
3. **Compact-vs-SPINT diagnostic:** report `B-C - H-S-continuation` and
   `B-C0 - H-S-continuation` on the same windows.  A result within `-0.03 R2`
   of H-S is non-inferiority evidence only; it is not a superiority claim with
   one fold.
4. **Mechanism sanity:** the B-C side contribution must not disappear when
   the B-C0/B-C initial active weights, source normalizer, and source/query
   boundaries are checked.  If only the two compact arms differ in training
   state or checkpoint selection, stop and repair the pilot.

If the carrier gate passes, the next expansion follows the control-first order
frozen in section 5.1 below; folds 1 and 2 do not start before those mechanism
controls pass.  No formal held-out or EvalAI submission follows automatically.
If it fails, record M1 as a compact-carrier negative/low-headroom development
result and close this branch.

### 5.1 Precommitted expansion order and mechanism gates

This paragraph was frozen while the fresh fold-0 three-arm run was still in
progress and before either `B-C` or `H-S-continuation` had produced a score.  It
prevents a positive `B-C-B-C0` result from being overinterpreted as evidence for
the three EMG-conditioned coefficients when the fourth, baseline-rate coordinate
could be sufficient.

If and only if the fold-0 carrier gate `B-C-B-C0 >= +0.03 R2` passes, keep the
candidate, seed, source split, fixed epoch and query windows unchanged and run
the following separately trained compact controls on fold 0, in this order:

1. `B-B4`: the already-defined post-normalization baseline-only arm
   `[0,0,0,b]`;
2. `B-RS4`: the already-defined deterministic complete-row permutation of the
   full normalized carrier;
3. `B-LS4`: a deterministic nonidentity permutation of the ten support-trial
   EMG-score rows before the per-channel ridge fit, while leaving spike-rate
   rows, source-frozen PCA, normalizer, trial exposure, support/query boundary
   and model topology unchanged.

`B-LS4` is a ten-trial vector-label null, not a dense-bin shuffle: the current
M1 AFC4 estimator fits ten movement-window mean EMG rows to ten movement-window
per-channel mean-rate rows.  Its permutation must be a derangement, must be
fixed from the session name and seed before training, and must be recorded in
the receipt.  It must not refit the EMG PCA on shuffled labels.

The functional-content mechanism gate requires all three paired fold-0 deltas
to be at least `+0.03 R2`:

```text
B-C - B-B4  >= +0.03
B-C - B-RS4 >= +0.03
B-C - B-LS4 >= +0.03
```

Failure against `B-B4` means the observed value is rate-scale conditioning, not
an EMG-tuning carrier.  Failure against `B-RS4` means correct channel attachment
is not established.  Failure against `B-LS4` means correct EMG--rate pairing is
not established.  These outcomes may retain an operational compact-model result,
but they stop the EMG-functional mechanism claim and do not authorize alternate
ranks, wider side inputs, FiLM, or a new label shuffle.

Only after both the original carrier gate and all three mechanism gates pass may
the unchanged six-arm family (`H-S`, `B-C0`, `B-C`, `B-B4`, `B-RS4`, `B-LS4`)
expand to the already approved folds 1 and 2 at seed 42.  Report every fold and
do not select a subset by sign.  An all-source deployment candidate or official
submission remains downstream of that three-fold development aggregate; it is
not selected from a single fold.

## 6. 5070Ti launch procedure (template, not an authorization)

The remote RTX 5070 Ti Laptop has one usable GPU.  Use a dedicated tmux session
and a new result root; never attach to an old RT/H1 process or reuse a checkpoint
root.  A reviewed implementation should expose three explicit Hydra configs and
one source-only receipt before launch.  The command shape is:

```bash
ssh xinyuan@100.103.97.12
cd /data/EDA/proj/digital_workspace/T28_BCI_TAPEOUT/SPINT
tmux new -s m1_version_b_f0_s42
CUDA_VISIBLE_DEVICES=0 python -u streaming_calibration_exp/src/train.py \
  --config-name train \
  experiment=m1_version_b_hs_continuation \
  seed=42 data.loso_fold=0 trainer.max_epochs=12 trainer.min_epochs=12 \
  trainer.limit_val_batches=0 trainer.num_sanity_val_steps=0
```

The B-C0 and B-C commands differ only in their isolated experiment name and
`afc4_arm`; run them sequentially on the same GPU, never concurrently with a
different writer for fold 0.  Before each launch, check `nvidia-smi`, create a
fresh root, and write a launch receipt containing the exact resolved config
hash.  Do not execute this template until the source-only datamodule and all
three configs pass CPU preflight.

The prior M1 B3S 12-epoch fold-0 run took about 45 minutes on a 3090 including
data setup and test.  A 5070Ti Laptop estimate is roughly 35--70 minutes per
compact arm and 50--90 minutes for the larger B0 continuation; budget about
2--4 hours for the three sequential cells, plus CPU preflight.  Record actual
wall time rather than using this estimate as a result.

## 7. Reuse decision

Safe to reuse without semantic changes:

- `StreamingCalibrationLitModule`'s `freeze_decoder=false` optimizer path;
- `B0/B1` original-SPINT identity encoder and `B3S` compact side encoder;
- `SourceFrozenEMGAFC4Plan` and its q=3/M10 estimator, after the new module
  proves the source-only path;
- the teacher decoder checkpoint and its hash.

Not safe to reuse unchanged:

- `M1EMGAFC4DataModule`, because it opens minival before replacing the dataset;
- the existing AFC4 M1 configs, because they freeze the decoder and use
  `task_plus_y_plus_E` rather than the common joint source-training contract;
- the all-source AFC4 module, because it has no LOSO target query;
- any existing formal/EvalAI or minival endpoint.
