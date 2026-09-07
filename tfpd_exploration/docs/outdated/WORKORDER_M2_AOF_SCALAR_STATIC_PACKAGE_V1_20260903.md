# Work order — M2 AOF-S Static Package V1

Date: 2026-09-03  
Status: **authorized for local build/validation only; network submission NO-GO**

Implement the frozen AOF-S design in
`DESIGN_M2_AOF_SCALAR_OOF_CONFIRMATION_V1_20260903.md` as an additive static
package. Reuse the existing `evalai_m2_act30_dopt4_v1` and
`evalai_m2_apfg_static_v1` build/validation seams where their contracts match.

Frozen additive source and output roots:

```text
source package root:
  tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1
immutable package-artifact root:
  tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/artifacts/local_build_v1
result root:
  tfpd_exploration/results/m2_aof_scalar_static_package_v1/local_build
```

The source package root is the closure-bound implementation and therefore must
exist. The package-artifact root and result root must both be absent before
admission. No overwrite, in-place migration, or reuse of a retired/partial
artifact/result root is allowed. Payloads, validation outputs, and image-build
evidence may be published only below `artifacts/local_build_v1`; implementation
source files are never generated or mutated by the build lifecycle.

## Authorized work

- Descriptor-validate the immutable AOF-M **valid terminal, gate-failed** root:
  `tfpd_exploration/results/m2_anchored_output_matrix_fusion_v1/source_oof`.
- Bind terminal SHA
  `3bc012339f73ab207d18fcddd701adf1fb884c73bb34512cffb0376ba32f1a6e`
  and closure
  `eb74bd47cd0e97f10474d1472919f3b591ef2f0b70feededee44dc77329462e8`.
- Validate its exact six-body/twelve-leaf terminal-only topology
  (`attempt`, `predecessor_authority`, `launch`, `source_authority`, `oof`,
  `terminal`), sidecars, `0444`, `nlink=1`, no extras/failure/all7 matrix, and
  all terminal links. Require the governing AOF-M gate `passed=false`,
  `all7_refit_performed=false`, and exact failed matrix-minus-scalar condition;
  never relabel this predecessor as a success.
- Reconstruct the seven scalar OOF rows and the all-seven beta only from the
  sealed OOF sufficient statistics. The exact expected values are:

  ```text
  mean scalar-native = +0.006358371947682961
  positives          = 6/7
  worst              = -0.0021069852144761647
  numerator          = -2.643848775861138e-05
  denominator        =  6.958658366347930e-05
  beta               = -0.3799365677508742
  ```

- Build a new payload by loading the exact `c51b...` payload used by submission
  `581361`, retaining its decoder and 13 native identities, and adding the 13
  post identities. The later `e4ff...` reconstruction may be used only for
  declared numerical-bridge validation, never as the governing native map.
- Freeze `h_post` to the exact chronological float32 operator in the design:
  PIT-cubic first30 activity, normalized D-opt4 side, `pre_pool`, side expansion,
  concatenation, `post_pool`, sequential arrival-order float32 addition, and
  division by 30. Require exact receipt links for all 13 activity, support,
  raw/normalized T4, side, and native-identity digests. Only the seven source
  sessions may carry source-screen parity claims.
- Implement exact two-call decoded-output fusion and a direct-native `+0`
  control in a new route-owned decoder.
- Run CPU/local minival and contract validation, build the Docker image, and
  publish immutable local receipts.

## Exact official comparator and package lineage

Descriptor-bind all of the following rather than an approximate baseline:

```text
official scored terminal:
  sua_exploration/evalai_t4_m2_activity_budget/artifacts/
  evalai_submission_581361_terminal_receipt_v1.json
  SHA 68da5427e2d27169b8b6e1e08a487113b65b734a7e7644a5d1778f993febbb97
official push state:
  sua_exploration/evalai_t4_m2_activity_budget/artifacts/evalai_push_state_m4_v1.json
  SHA 015bdf9ccd3aa2c3959ff34b5b68346271ac59900f79edccb44505a518e33413
official submission / held-out R2:
  581361 / 0.2897439880338965
official payload / image manifest:
  c51b71167d81490927fee8a552785ee27ddff7e90085ed3ff4b18b857afbac40
  sha256:378bbd4868e515a3527418e098db2989e0fc8fbccc929ee029a237ce5e9f291b
official payload receipt:
  sua_exploration/evalai_t4_m2_activity_budget/artifacts/
  t4_m2_seed42_ridge_m4_activity30_identity.receipt.json
  SHA a3c304106ce56105c3bde59b3237604e95388608d740800da4ff2da0fa7186db

later bridge-only reconstruction receipt:
  tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/artifacts/
  t4_m2_seed42_dopt4_act30_identity.receipt.json
  SHA 6f90230f9f8f330edeec970ea0108defde24cd43ca3195fea264083cac6fa583
later bridge-only reconstruction payload / image:
  e4ff17e857c0bab9bbd900bc737ca7c48476a44ed03725cefc5377d92b959261
  sha256:79ac29ca71a84eb97be59411fc80d2a567127e95fd96b63dfa042d22b5a4421f
sealed local score:
  tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json
  SHA 6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce
checkpoint / student state:
  25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
  2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20
```

The two payload/image lineages are not byte-identical. AOF-S must inherit the
`c51b...` decoder/native identities from the exact scored comparator. Receipts
must disclose that `e4ff...` is bridge-only reconstruction evidence. No
cross-lineage SHA-equivalence claim or substitution is allowed.

## Forbidden work

- No EvalAI API call, ECR push, submission registration, hidden test, or
  network mutation.
- No model/gate/identity training, optimizer, gradient, checkpoint mutation,
  beta refit, target adaptation, new fit family, or hyperparameter sweep.
- No reuse of APFG's identity-space `tanh(alpha)` as though it were AOF-S.
  AOF-S combines two decoded outputs with the literal beta above.
- No overwrite of the AOF, AOF-M, `act30_dopt4`, or APFG package/result roots.
- No claim that source OOF is untouched confirmation.

## Required tests and receipts

1. Exact AOF-M predecessor held-FD codec and adversarial topology/body/sidecar
   tests.
2. Repeated sufficient-statistic equality, exact declared-order reconstruction,
   and drift/reorder/duplicate/missing-session rejection.
3. Decoder `+0` direct-native object equality and `-0` rejection.
4. Same-window native/post decode evidence, literal beta, scale 5.0, and zero
   update/state-digest checks.
5. 13-tag payload shape/dtype/finite/digest checks and exact official native
   identity parity.
6. Real local evaluator/minival validation, with no selection based on its R2.
7. Immutable attempt/authority/validation/terminal XOR failure lifecycle,
   externally reviewed non-self-referential closure, and exact result topology.
8. Docker image identity, payload SHA, source closure, environment, dependency,
   and no-network evidence.

## Immutable lifecycle and local container law

The successful result root contains exactly these six immutable body/sidecar
pairs and no extras:

```text
attempt.json
predecessor_authority.json
input_authority.json
build.json
validation.json
terminal.json
```

Every published body is `0444`, `nlink=1`, has a basename-bound `.sha256`
sidecar, and is descriptor-linked forward. Failure is terminal-only for the
greatest already-published immutable prefix: publish `failure.json` instead of
`terminal.json`; never delete or overwrite a published prefix, and never allow
both terminal and failure. The terminal binds the exact package file tree,
payload/receipt/validation hashes, image ID, base-image identity, closure, and
all predecessor authorities.

Docker build/validation is offline only. The Dockerfile must use a pinned base
image that is already present locally: tag
`spint-m2:e8-epoch027-76f0fb2`, exact local image ID
`sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8`.
Before build, prove that exact tag-to-ID mapping exists; fail closed if absent.
Build and container validation run with network disabled (`network=none` /
equivalent), perform no pull, login, push, API request, or registry mutation,
and record that evidence. The package may contain an inert guarded submission
helper, but no local lifecycle capability may authorize network use.

The final handoff may report a locally validated image and the exact guarded
submission command, but it must not execute that command without a new explicit
authorization.
