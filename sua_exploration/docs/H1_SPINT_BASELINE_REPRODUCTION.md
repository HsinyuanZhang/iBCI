# H1 original-SPINT baseline reproduction (isolated preparation)

> **RT scope clarification:** “RT is sealed” below refers only to the historical dense-velocity
> `afc4_vel` matrix. The separate sparse-endpoint `R-T4d` Stage-2 matrix is not terminal.

**Status:** RT is sealed and both predeclared H1 original-SPINT variants completed their fixed 50-epoch from-scratch runs on 2026-08-07. Both terminal checkpoints report metadata epoch 49 and global step 206650; their immutable public-prefix finalizers and dual terminal seal passed. Exact role packages were built, container-hash checked, and registered privately on EvalAI before either result was opened. Submission `578473` (released-code LR `5e-5`) scored held-out `0.2099167±0.1142316`; `578474` (paper LR `1e-5`) scored `0.2614916±0.1487168`, within `0.02851` mean R2 of the paper's approximately `0.29±0.15`. The roles remain fixed; this is a close, not exact, paper-setting reproduction. This document is separate from the stopped q3-AFC4 pilot. The formal M2/M4 CPU gate decisions remain unchanged. The unnormalized equal-M4 V1 pilot was numerically inactive. Its separately frozen source-RMS V2 then produced strong Full--Zero/row/label effects (`+0.08558/+0.10359/+0.10326`) but Full `0.49388860` remained `0.00294445` below matched SPINT `0.49683305`; it therefore failed only the superiority clause and is stopped without date expansion. A later parameter-efficient CarrierID consumer replaced the 5,965,500-parameter SPINT identity MLP with a 58,140-parameter carrier-aware encoder. Its two-seed single-date result expanded to a positive five-date development aggregate (`H-C−H-S=+0.05629` equal-date mean), then to an all-source private EvalAI submission. Submission `578689` scored held-out `0.274939±0.127206` and held-in `0.473125±0.039341`, exceeding the stronger paper-LR reproduction by `+0.013448/+0.002702` mean R2. This is organizer-held system-level evidence; no per-recording metrics or organizer-held matched carrier controls were exposed.

## Frozen roles and protocol

The two learning-rate variants are named before any result exists and must not be renamed or exchanged after scoring:

| variant | role | Adam LR |
|---|---|---:|
| `released_code_lr_5e-5` | released-configuration sensitivity and implementation primary; **not** the same-hyperparameter paper-number reproduction | `5e-5` |
| `paper_lr_1e-5` | Appendix / paper-number reproduction candidate (the paper/code discrepancy sensitivity) | `1e-5` |

Both variants use the original `FalconLitModule` and `SpintModel`, with only the optimizer LR differing scientifically. The common protocol is:

- all 13 held-in recordings from six dates (`19250101`, `19250108`, `19250113`, `19250115`, `19250119`, `19250120`);
- `calibration_n_trials: 2`, `random_calibration: true`, `window_size: 700`, `max_trial_length: 1024`;
- model dimension 1024, 64 heads, one transformer layer, three ID-MLP layers, dynamic dropout enabled;
- seed 42, public trainer `deterministic: false`, fixed `min_epochs=max_epochs=50`;
- no validation metric selects an epoch. The terminal checkpoint is metadata epoch 49 (normally named `epoch_049.ckpt`), and finalizers report `epochs_completed: 50` and `global_step` explicitly.

Only `paper_lr_1e-5` may be compared as a same-hyperparameter reproduction candidate against the
paper's H1 `0.29 +/- 0.15`. The `5e-5` arm remains a separately named released-configuration
sensitivity even if its local or private score is larger. Neither minival nor a private score may
exchange these roles or select which arm is reported.

## Official-source provenance

A fresh read-only clone of the paper-linked repository `shlizee/SPINT` on 2026-08-07 resolved to
commit `c1784d0f770e6df672af3cfa7391423b5e5884be`. The local H1 data config, H1 model config, complete
`SpintModel` implementation, and streaming `spint_sample.py` are byte-identical to that commit. The
ASTs of `FalconDataset`, `SessionBatchSampler`, `FalconLitModule.forward`,
`FalconLitModule.model_step`, and `FalconLitModule.training_step` are also identical. Consequently,
the neural window/calibration sampling, SPINT forward/loss, and training-step computation used here
are traceable to the official source.

The worktree as a whole is deliberately **not** described as an exact checkout. Local differences
add held-in-only enumeration and manifest guards, suppress an unpopulated held-out validation metric
through `clean_teacher`, pin `num_workers=0`, save a metric-independent terminal checkpoint, make
modern PyTorch deserialization explicit, pin the evaluator version, and bind an explicit package
filename. These controlled differences and their hashes must be carried into the terminal package
receipt. Until that immutable receipt exists, the accurate description is an audited local
derivative of the official H1 implementation, not an exact released-code replay.

The immutable source receipt is
`../results/h1_spint_baseline_reproduction_v1/official_source_provenance_v1.json`, SHA-256
`c01d5d00f78d6fd8aee0632856ce1e1cf878478c99432f373f336055381f0594`, mode `0444`. It records
the exact official/local file hashes, the semantic AST hashes above, and the controlled-difference
classification. The later terminal package receipt additionally binds the exporter and Docker
runtime hashes actually used for each submission.

The baseline's SPINT ID input is neural-only trialized calibration features. H1 has no discrete direction-label stream, but it does expose dense 7-D kinematic velocity covariates as the behavior target. Four separately frozen M=2 CPU programs tested whether those covariates support a reproducible carrier. Signed AFC4 had `0/6` dates pass its split-trial direction-cosine gate; invariant AFC4 and LFMC4 each had `0/6` dates with positive correct held-trial predictive gain. A statistically distinct low-rank population decoder `y <- X` showed correct pairing above a support-label-rotation null q95 on `6/6` dates, but its trial1/2 coefficient-row attachment stability reached cosine `>=0.5` on `0/6` dates. Thus H1 contains weak population-level label association, but no tested M=2 estimator can attach it reproducibly to per-channel identity. A final remaining-hypothesis-space audit found no defensible fifth M=2 family: per-channel lag/kernel/CCA-like candidates reparameterize the stopped encoding/moment programs, while population projection or paired Procrustes still requires the failed row attachment. The current M=2 carrier branch is therefore stopped before GPU decoding. The population-decoder receipt is `../results/h1_population_decoder_carrier_m2_date_lodo_v1/H1_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json`, SHA-256 `71d817b5c32fc8137853d7c60506c4c75eb39eac5bd32c519bdd924986178c0e`, mode `0444`; the closure audit is `H1_REMAINING_HYPOTHESIS_SPACE_AUDIT_M2.md`, SHA-256 `892062ff64861a6335eb9c7ddf211b18339195c452601d5192d14cefd57591cf`.

A separately named M=4 CPU program was then run once under a protocol frozen before its results.
Correct later-trial population-decoder R2 was positive and exceeded the within-support label-rotation
q95 on all `6/6` dates, but independent trials1--2 versus trials3--4 coefficient-row attachment
reached cosine `>=0.5` on only `2/6` dates, below the required `4/6`. It is therefore sealed
`NO_GPU`; the stronger population association is not treated as stable channel identity. Its receipt
is `../results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json`,
SHA-256 `660f78f86ed74b3950ff53946edc2db50979802e8f02a37b211c5e044c0ed4bb`, mode `0444`.

One final separately frozen confidence hypothesis then applied source-only empirical-Bayes shrinkage
to those same M=4 rows, using support-only analytic ridge covariance and no strength sweep. It raised
both direct R2 and split-half cosine over raw M4 on all `6/6` dates, beat fresh label rotations and a
complete row-shuffle on `6/6`, but still reached the absolute cosine `>=0.5` clause on only `2/6`.
It therefore remains a consistent variance-reduction diagnostic, not a passed channel-identity gate
or GPU authorization. Its immutable receipt is
`../results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json`,
SHA-256 `13004595d5d5e28c4fb1316bf7119bd3cdb2197bbf5001abb53eec5d2881c964`, mode `0444`.
Any future kinematic comparison would remain a separate supervised few-shot carrier, must bring
genuinely new target-side identifiability evidence, must not be described as GF-FSU, and must not be
conflated with this neural-only SPINT baseline. An M=4 performance comparison would also require an
equal-four-trial neural-only SPINT comparator and cannot be presented as an M=2 rescue.

A later, independently frozen source-only audit tested the remaining activity-selection and
kinematic-rank hypothesis as a fixed `G0/G1 x q_kin={1,2,3}` matrix. Its sole primary cell was the
RT-style raw-20-ms-bin activity gate with rank three, and every score used the actual reduced-rank
reconstruction `E @ U.T` on all ungated trials-5+ query blocks. The gate retained `46.06%--52.01%`
of M4 support blocks and raised the rank-three absolute split-half attachment count from `2/6` to
`4/6` dates. Nevertheless, reconstructed query R2 was positive on only `2/6` dates; the equal-date
mean fell from `+0.03656` without gating to `-0.02470` with gating, with a negative gated-minus-
ungated delta on all six dates. Thus activity-conditioned coefficient stability did not generalize
to the full streaming query distribution, and the gated/reduced-rank branch is sealed NO-GPU. The
immutable receipt is
`../results/h1_m4_gated_reduced_rank_carrier_date_lodo_v1/H1_M4_GATED_REDUCED_RANK_CARRIER_CPU_RECEIPT.json`,
SHA-256 `e659ca55ebd9d9a89d394ba813efe6bfcf8c8eb98e43ea5751ca7a53604f6232`, mode `0444`.

That separately declared equal-M4 comparison has now been completed for exploratory fold 0. Both
from-scratch terminal checkpoints were bound at epoch 49/global step 180500, with identical runtime
initial-state and source-manifest hashes. On the strict trial5+ query, matched SPINT scored
`0.4968330503`, while joint Full scored `0.4720068382` (`-0.0248262122`). Joint Zero, row-shuffle,
and label-rotation/refit all differed from Full by only `1.09e-9`--`1.37e-9`. Thus the jointly trained
decoder did not measurably consume this EB carrier and did not exceed the equal-budget SPINT base.
The immutable receipt is
`../../SPINT-main/logs/h1_m4_eb_fold0_paired_s42_v1/H1_M4_EB_FOLD0_TERMINAL_GATE_RECEIPT.json`,
SHA-256 `75449da18e5b6c2b9b1e56f167c00134acf16cf3ed012cc5bb601d6c1b506af7`, mode `0444`.
This closes that exploratory M4 system route without changing the original neural-only baseline or
the earlier CPU gate decisions.

A post-terminal numerical audit then narrowed what that STOP means. The target carrier RMS was only
`8.36e-6`--`8.74e-6`; combined with the learned residual-weight RMS `1.314e-5`, the actual injected
identity residual was `2.88e-10`--`3.03e-10` RMS, about `9e-10` of the ordinary SPINT identity.
The implementation had normalized the neural inputs used to fit the ridge model but had not applied
the source-only descriptor normalization required for the final carrier coordinates. V1 therefore
remains a valid negative result for its exact unnormalized system, but Full--control equality cannot
be interpreted as a clean test of carrier content. One separately named V2 numerical-conditioning
repair is allowed: divide all train/deployment carriers and controls by the single global RMS of all
legal source-only M4 carrier-cache rows. This reparameterization changes no estimator, information,
width, seed, optimizer, epoch budget, or target gate and permits no scale sweep. It must pass a
no-target gradient/update/injection preflight before any GPU run, and it cannot revise the M4
attachment STOP or support comparison with original M2 SPINT.

That V2 has now completed its only allowed fold-0 terminal. Both arms reached metadata epoch 49 / global
step 180500 from the same runtime initial state and source schedule. On the same strict 8,965-sample
trial5+ query, matched SPINT scored `0.4968330503`; normalized-V2 Full scored `0.4938885959`, Zero
`0.4083090652`, row-shuffle `0.3903015503`, and label-rotation/refit `0.3906331111`. Hence the
correct carrier improved the same joint checkpoint by `+0.0855795/+0.1035870/+0.1032555` over the
three controls: numerical normalization revealed substantial carrier-content, channel-attachment,
and correct-pairing dependence that V1 could not test. Nevertheless Full--matched-base was
`-0.0029445`; the predeclared all-five conjunction therefore failed and no date expansion is
allowed. This is positive mechanism evidence but negative system-superiority evidence. The immutable
terminal receipt is
`../../SPINT-main/pilot_artifacts/h1_m4_eb_normalized_v2/gpu_runs/h1_m4_eb_normalized_v2_terminal_gate.json`,
SHA-256 `ffa4582a36122f34c9d703376dcd23f21b3f26a2af16ae73d06fc10cd45406b8`, mode `0444`.

The subsequent CarrierID experiment changed the consumer rather than adding another residual to
the large SPINT identity. It replaced the `5,965,500`-parameter identity MLP with a `58,140`-
parameter early-pooling carrier-aware encoder (a `102.606x` reduction in static identity
parameters), retained the same downstream SPINT decoder topology, and used zero target-session
optimizer/backward steps. On the same 8,965 strict post-support windows, seed 42 scored
`H-S/H-C/H-C0=0.49683305/0.52551078/0.48661562`; the independently scheduled seed 43 scored
`0.51838670/0.54095494/0.50130780`. Thus `H-C-H-S` was
`+0.02867773/+0.02256824` (mean `+0.02562299`, 2/2 seeds positive, seed SD `0.00432006`) and
`H-C-H-C0` was `+0.03889516/+0.03964715` (mean `+0.03927115`, 2/2 positive). In seed 43 the
same-checkpoint label/row/zero interventions reduced Full by `0.10421/0.14253/0.08415`, confirming
that the trained consumer uses carrier content and attachment. Both seeds nevertheless show a
positive `H-C-H-S` delta on the larger recording and a negative delta on the smaller recording;
this remains a single-date, two-recording development result rather than session-uniform or
cross-date superiority. The seed-43 terminal and two-seed aggregate receipts have SHA-256
`307528cca84ea108d691ee4f190c79548a4f359cc8bde1c652f0afec60df3855` and
`5b4813efbb09fdc29e00c4915d5297f05e7cf851848e6b6a5e0a8d778940a435`, respectively, both mode
`0444`.

The subsequently frozen five-date source-date LODO program gave equal-date H-C--H-S
`+0.05629` (4/5 dates), H-C--H-LS `+0.03037` (4/5), and H-C--H-C0 `+0.03256` (4/5), with every
reversal retained. The all-source model then used all 13 public held-in calibration recordings and
the fixed epoch-49 endpoint. Its deployment package consumed each evaluator-provided calibration
prefix exactly: M=4 for held-in and M=3 for held-out, with no padding, duplicated trials, target
optimizer, or backward step. Private EvalAI submission `578689` returned held-out
`0.2749391781±0.1272058604`, held-in `0.4731250306±0.0393413347`, and normalized latency
`0.1139191965`. The held-out mean deltas versus `578474/578473` are `+0.0134476/+0.0650225`.
Because EvalAI returned aggregates only, these differences have no paired confidence interval,
sign test, or session-uniform interpretation. The official endpoint validates the deployed system;
carrier-content attribution continues to rely on the source-date H-C0/H-LS and same-checkpoint
corruption controls.
The immutable candidate terminal receipt is
`../../SPINT-main/pilot_artifacts/h1_carrierid_all_source_official_v1/H1_CARRIERID_ALL_SOURCE_EVALAI_TERMINAL_RESULT_RECEIPT_v5.json`,
mode `0444`, SHA-256
`b7f76450494f3c0d0ad169de9e89f662c17f435ef2c08bb3d7062908515b488e`.

## Loader semantics and scope

The upstream loader's `random_calibration` behavior was checked directly: for each training window it samples a uniformly random **contiguous** two-trial block with `random.randint(0, total_trials-2)`. It does not sample arbitrary bins or an independent trial pair. Held-in validation uses the first two trialized calibration records deterministically. Calibration trials are interpolated to 1024 samples with cubic interpolation; the query window remains 700 bins.

The isolated `H1BaselineDataModule` at `SPINT-main/src/data/h1_baseline_datamodule.py` preserves those `FalconDataset` and `SessionBatchSampler` semantics while enumerating only:

- `sub-HumanPitt-held-in-calib` (13 files) for fit;
- matching `sub-HumanPitt-held-in-minival` (13 files) for validation only.

It refuses `test`/`predict`, never scans `sub-HumanPitt-held-out-calib`, and emits an input manifest with per-file SHA-256. The `num_workers: 0` setting is an execution-only deviation: upstream normalizes `null` to `os.cpu_count()-1`, whereas the isolated module explicitly uses zero workers to avoid host-dependent worker fan-out while preserving ordering, support, trial selection, and labels. The held-in baseline reproduction is intentionally all-13 together. Any later generalization analysis should group recordings by the six calendar dates before considering a session-level split; the q3 pilot's session-LOSO partition is not this baseline protocol.

## CPU receipt and tests

The held-in-only source/local preflight writes the superseding immutable v3 receipt at:

`../results/h1_spint_baseline_reproduction_v1/cpu_preflight_receipt_v3.json`

It verifies 26 held-in files, six date groups, random contiguous two-trial training selection, deterministic first-two validation selection, one CPU forward with output shape `[1,700,7]`, and records input-manifest/source/config hashes, the LR role lock, fixed-epoch volume, and the installed `falcon-challenge` 1.0.2 evaluator hash. The older `cpu_preflight_receipt_v2.json` is retained for audit history but is obsolete/superseded because it predates evaluator and staged-runner source-hash closure; `cpu_preflight_receipt.json` is an earlier draft and is not frozen. Paper Appendix A.4.6 reports `<2 GiB` peak on an A40 for H1 batch32 and approximately eight hours for 50 epochs; those are reported references, not local RTX3090 measurements.

The behavior scaling is also intentionally equivalent: paper Table A9 uses a target scale of `0.05`; released code uses `behavior_scaling_factor: 20.0` and divides the forward output by 20 in `FalconLitModule.model_step`, i.e. the same numerical scale. This is not an architecture or target-definition discrepancy.

Targeted tests:

```bash
cd SPINT-main
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  /home/xinyuan/miniconda3/envs/spint/bin/pytest -q \
  tests/test_h1_baseline_spint.py tests/test_h1_baseline_eval.py \
  tests/test_h1_baseline_runner.py
```

The tests cover no-held-out enumeration, random contiguous two-trial versus first-two validation semantics, six-date grouping, exact LR-only resolved-config differences, fixed-50/no-monitor configuration, original-SPINT forward/loss compatibility, terminal checkpoint rejection, remainder-preserving evaluation, day grouping, state immutability, receipt overwrite refusal, and the staged-runner executable/import contract. The complete targeted command was rerun immediately before launch on 2026-08-07 and passed `10/10` tests.

After launch, the terminal package and EvalAI-binding gates were added without changing the live
training process. The combined H1 suite covering baseline, finalizer, explicit 27-file package
allowlist, payload audit, Docker/runtime binding, and immutable external-result receipt now passes
`32/32` tests. This is a packaging/provenance extension, not a model or data-path change.

The staged runner also has a shell syntax/static contract test: it must pin the executable `/home/xinyuan/miniconda3/envs/spint/bin/python` by default (or an explicitly overridden `PYTHON_BIN`), verify that it can import the SPINT runtime, and invoke `src/train.py` through that interpreter rather than relying on a host `python` command.

## Endpoint-scope correction: public minival is not independent

A separate read-only audit opens the 13 public held-in-calibration/minival pairs through the same
FALCON H1 loader and compares every loader-visible field. In **13/13** pairs, the complete minival
`neural`, seven-dimensional `behavior`, `trial_change`, and `eval_mask` arrays are a bit-exact prefix
of the corresponding calibration arrays. Minival contains 1,503--1,831 bins per recording, beginning
at calibration bin zero; calibration contains 6,456--12,267 bins.

The immutable receipt is
`../results/h1_spint_baseline_reproduction_v1/heldin_minival_prefix_audit_v1.json`, SHA-256
`0a729c0c29341401780b49a474e303ed15a6094d0da8738fed4a012fed590769`, mode `0444`. The
reproducible audit script is `SPINT-main/scripts/h1_heldin_prefix_audit.py`, SHA-256
`8d35205c707b76f83c7a0720778ede0e4992197e61c5b693c10a451ed50e47d5`.

Consequently, the staged run remains useful for architecture, optimizer, training-volume, terminal
checkpoint, and scoring-pipeline reproduction, but its local minival R2 is an
**implementation-regression endpoint only**. It is neither independent held-in evidence nor a
reproduction of the paper's EvalAI-private held-in `0.47 +/- 0.06`, and it says nothing about the
paper's formal held-out `0.29 +/- 0.15`. A scientific local development comparison must instead use
a chronological query constructed from calibration bins/trials strictly after its support, and a
paper-level reproduction ultimately requires the private EvalAI endpoint.

The same public files also do not provide a clean high-budget rescue endpoint. A separate immutable
budget receipt at
`../results/h1_spint_baseline_reproduction_v1/calibration_budget_endpoint_audit_v1.json`
(SHA-256 `03d7a27c1953c49f0dc13c455d0c9f0d5e84cdf9e6f91122a59eef5ddd8aade3`, mode
`0444`) finds three recordings with only seven legal trials. M=8 is therefore not a common budget;
M=6 leaves only one future trial in those recordings, including two of three recordings on date
`19250108`. This receipt authorizes no GPU work. M=4 was subsequently evaluated once as a separately
frozen descriptive CPU deployment condition and failed its own attachment gate, as recorded above;
it is not a post-hoc replacement for the failed fair-M=2 carrier question and authorizes no GPU work.

The private route was confirmed reachable by a read-only EvalAI CLI check on 2026-08-07. The
configured account participates in FALCON challenge `2319`; Minival phase `4598` and Test phase
`4599` were active and public, with Test rate-limited to six submissions per day. After both roles
passed their fixed-terminal integrity checks, they were registered privately as `578473` and
`578474`, respectively, before either result was opened. Both are now terminal `finished`; their
exact results are recorded above. The later CarrierID submission `578689` is also terminal
`finished` and is bound by its separate immutable receipt.

The Docker contract is H1-capable: `spint_sample.py` accepts `--split h1`, the image receives
`TASK=h1`, `BATCH_SIZE=8`, and an explicit `MODEL_FILE`, and `.dockerignore` now whitelists the two
predeclared H1 package names. Package roles must remain visible in filenames and image tags:

```bash
# Released-configuration sensitivity package. The gate rejects rolling/nonterminal checkpoints,
# unknown/minival calibration files, role/LR mismatch, and receipt overwrite.
PYTHONPATH=. /home/xinyuan/miniconda3/envs/spint/bin/python \
  scripts/h1_terminal_package.py \
  --data-dir /absolute/data/000954 \
  --run-dir /absolute/h1_baseline_staged_run/released_code_lr_5e-5 \
  --checkpoint /absolute/h1_baseline_staged_run/released_code_lr_5e-5/checkpoints/fixed_epoch50/epoch_049.ckpt \
  --config /absolute/h1_baseline_staged_run/released_code_lr_5e-5/.hydra/config.yaml \
  --variant released_code_lr_5e-5 \
  --package local_data/spint_h1_released_code_lr_5e-5.pkl \
  --receipt /absolute/results/h1_released_code_lr_5e-5_package_receipt.json

docker build --build-arg TASK=h1 --build-arg BATCH_SIZE=8 \
  --build-arg MODEL_FILE=spint_h1_released_code_lr_5e-5.pkl \
  -t spint_h1:released-code-lr-5e-5 \
  -f third_party/falcon_challenge/spint_sample.Dockerfile .
```

The paper-LR reproduction candidate uses `spint_h1_paper_lr_1e-5.pkl`, variant
`paper_lr_1e-5`, its own immutable receipt, and a distinct image tag. Both roles were
frozen before results; hidden scores may not select or rename either role. Packaging
may read the released public H1 calibration files to cache neural-only two-trial ID features, as the
original decoder does; it does not read private query behavior.

`SPINT-main/scripts/h1_evalai_submit.py` performs the live role binding. Read-only mode rechecks the
terminal package/checkpoint/source hashes, exact package basename, actual local image ID, runtime
`TASK=h1/BATCH_SIZE=8/PHASE=test`, and `/data/decoder.pkl` bytes. Execute mode pushes a UUID ECR tag,
requires the returned content-addressed manifest or config digest to equal the local image ID, records
the ECR manifest/config digests and submitted URI, then performs one private phase-4599 registration
with the three required metadata fields. The current Docker-29 legacy-builder path binds through the
registry manifest digest; both registered images have exact local-ID/registry-manifest equality. The
older `h1_evalai_submission_receipt.py` remains an offline compatibility receipt, not the authority
for the live push. The required `MODEL_FILE` mapping is frozen as:

| variant | package / Docker `MODEL_FILE` |
|---|---|
| `released_code_lr_5e-5` | `spint_h1_released_code_lr_5e-5.pkl` |
| `paper_lr_1e-5` | `spint_h1_paper_lr_1e-5.pkl` |

## Staged launch and terminal finalization

The RT prerequisite is now satisfied. The immutable marker
`../results/k4_rt_loso_v1/rt_seed42_clean_nested_seal.marker` has status `PASS_RT_SEALED`, SHA-256
`ef637fc64aaf317deb2111d2daaed2bc7c0aa00e0786a9bf09004b1d82ff2402`, and binds the exact
45-cell RT aggregate SHA-256
`b8e5abcbe6cf7688fb5cb76c3bbd20b835b1e6332de1a59bf0ce2d692974f54a`.

The two predeclared variants were launched at 04:23 HKT in tmux
`h1_spint_baseline_s42_v1`, with run root
`SPINT-main/logs/h1_baseline_staged_20260807_s42_v1`. GPU0 owns
`released_code_lr_5e-5`; GPU1 owns `paper_lr_1e-5`. Both resolved configs completed epoch 0 with
4,133 train batches, approximately 1.2 GiB initial device memory, and no traceback/OOM. Both
epoch-0 rolling `last.ckpt` health snapshots pass archive and `torch.load` checks with metadata
`epoch=0/global_step=4133`; these snapshots are overwritten by later epochs and are not sealed
artifacts. The command used was:

```bash
cd SPINT-main
bash scripts/run_h1_baseline_staged.sh \
  /absolute/path/to/RT_SEALED_MARKER \
  /absolute/path/to/h1_baseline_staged
```

The runner places `released_code_lr_5e-5` on `CUDA_VISIBLE_DEVICES=0` and `paper_lr_1e-5` on `CUDA_VISIBLE_DEVICES=1`; it refuses to start without a marker containing `PASS`, `SEALED`, or `COMPLETE`, and refuses a missing/non-importable `PYTHON_BIN`. By default it uses `/home/xinyuan/miniconda3/envs/spint/bin/python`, because this host has no `python` command. The expected training volume from the CPU preflight is 4,133 full train batches per epoch, 206,650 optimizer steps over 50 epochs, and 627 held-in validation batches per epoch (validation is diagnostic only). Adam parameter/gradient/state lower bound is about 0.25 GiB. Paper Appendix A.4.6 reports `<2 GiB` peak on an A40 and approximately eight hours; local RTX3090 elapsed time and peak memory will be recorded from the live runs rather than inferred.

Both local jobs exited normally after approximately 2 h 22 min, substantially faster than the paper's
A40 wall-time reference. Their terminal facts are:

| variant | checkpoint SHA-256 | epoch / steps | public-prefix six-date mean R2 | package SHA-256 | EvalAI ID |
|---|---|---:|---:|---|---:|
| released-code `5e-5` | `f36091a7...c4596c` | `49 / 206650` | `0.960503` | `20a1d41a...49298` | `578473` |
| paper `1e-5` | `25803244...16cd` | `49 / 206650` | `0.935215` | `40b12314...95f05` | `578474` |

The local public-prefix numbers are not official accuracy evidence: the endpoint audit proves that
these minival arrays are calibration prefixes. They establish load/forward/training sanity and make
no claim relative to the paper's private held-in `0.47+/-0.06` or private held-out `0.29+/-0.15`.
Only phase-4599 results are comparable to the latter. Those results are no longer pending:
`578473/578474/578689` are terminal and their official aggregates are reported above.

After each run, pass the exact metadata-terminal checkpoint and resolved Hydra config to the post-run finalizer; it never searches minival scores:

```bash
PYTHONPATH=. /home/xinyuan/miniconda3/envs/spint/bin/python scripts/h1_baseline_eval.py \
  --data-dir data/000954 \
  --checkpoint /absolute/run/checkpoints/fixed_epoch50/epoch_049.ckpt \
  --config /absolute/run/.hydra/config.yaml \
  --variant released_code_lr_5e-5 \
  --output /absolute/run/h1_terminal_receipt.json \
  --device cuda
```

Use `paper_lr_1e-5` for the sensitivity run. The finalizer preserves all windows, reports 13 recording diagnostics, concatenates recordings sharing `h1_session_date`, and uses the same equal-date mean and population-standard-deviation arithmetic (`np.std(ddof=0)`) as the installed `falcon_challenge.evaluator.compute_metrics_regression`; sample standard deviation is secondary. This arithmetic parity does not make the public prefix endpoint official-comparable evidence. Pooled variance-weighted R² and equal-recording summaries are also secondary. It rejects non-50-epoch checkpoints, held-out paths, mismatched LR roles, validation-selection configs, and existing receipt paths, and chmods successful receipts `0444`.
