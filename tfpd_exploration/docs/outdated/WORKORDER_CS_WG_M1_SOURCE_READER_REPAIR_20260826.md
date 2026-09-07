# Work Order: CS-WG M1 Physical Source Reader Repair

Date: 2026-08-26  
Status: bounded successor repair before any source-data or GPU execution  
Scientific role: make the accepted CS-WG Stage-0/source-lifecycle candidate executable on the unchanged native M1 data contract

## 1. Why this repair is required

The no-data CS-WG source-lifecycle candidate passed its focused tests, but it
is not a live-smoke candidate yet.

Two production facts were not represented correctly:

1. the historical source-only datamodule allow-lists only folds 0--2, while
   CS-WG requires all four outer folds; and
2. `FalconDataset.__getitem__` deterministically returns the same sealed
   chronological M10 calibration tensor for every query window from one
   session. The candidate incorrectly requires a unique calibration-owner
   token for every row.

The second condition is not a new form of sharing. It is the existing native
M1 contract: each row carries the correct session-specific M10 tensor, and all
rows from that session have the same sealed calibration content. The unsafe
operation is substituting another session's tensor, a global tensor, or a
caller-provided tensor.

This repair must preserve the network, optimizer, Stage-0 objective, sampler,
and source-only boundary. It must not create a new scientific arm.

The repair is based on these reviewed predecessor facts:

```text
CS-WG Stage-0 work order SHA256:
225fccec7588e28c25d1e4c3240066eb896fcd32c48e11236aaa8d45f8cb491d

CS-WG Stage-0 closure SHA256:
dd1fc152d4f7f900d6707bcea46136e1bde7cf781ca2f99dc12991296be7bb51

Original source-lifecycle work order SHA256:
ca0c3b754c602ec490e4f5b74c5bf85a93764184e6f9a489a10ec4eed892e043

Original no-data source-lifecycle closure SHA256:
2f2078bcd89476b84c42d78abedd6b2430d6114bd6550e1ef5e938352e429bdf
```

No canonical CS-WG source-smoke root was created under the original closure.
Root must descriptor-check that freshness again before any future launch.

## 2. Frozen scientific and network contract

Everything in the original source-lifecycle work order remains frozen unless
this document explicitly corrects it:

- exact existing M1 SPINT graph;
- 15,007,496 live parameters after the `[10,1024,64]` lazy materialization;
- query window `[100,64]` and output `[100,16]`;
- final-bin raw MSE;
- seed 42, Adam `1e-5`, weight decay 0, no scheduler;
- B32, 20 epochs, no AMP/TF32/compile;
- CS-WG `lambda=1.0`, `tau=0.01`, with no search;
- matched ERM paired by graph, seed, optimizer, step count, and checkpoint rule;
- source only, with no outer target, minival, held-out, formal, EvalAI, or
  target gradient/update surface.

No model, layer, dimension, dropout law, calibration transformation, loss
literal, or evaluation rule may change in this repair.

## 3. Correct calibration ownership law

For one source session, the reader must reconstruct exactly one deterministic
chronological M10 calibration tensor with shape `[10,1024,64]`. It must record
its contiguous-float32 SHA256 and session identity.

Every valid source query row must carry a row-local immutable view or copy of
that exact tensor. The following must hold:

```text
row.calibration_session == row.session
row.calibration_sha256 == session.calibration_sha256
all rows in one session have the same calibration_sha256
different sessions cannot substitute calibration tensors
```

The route must not require unique calibration contents or unique semantic
owners per row. A row identifier may remain unique, but it is not evidence
that the calibration data are different.

The mixed B32 concatenation must still stack all 32 row inputs explicitly and
preserve the row/session ownership table. Passing one tensor as a broadcasted
batch-wide calibration, reusing another session's tensor, or dropping the
per-row calibration input remains forbidden.

Receipts must disclose, per source session:

- calibration shape, dtype, contiguous-byte SHA256, and row count;
- that all row digests equal the session digest;
- that no cross-session calibration substitution occurred; and
- that the final concatenated B32 contains one calibration row per query row.

The physical smoke must also make seed 42 operational rather than merely
descriptive. Before model construction it must snapshot the process-local
Python, NumPy, Torch-CPU, and selected Torch-CUDA RNG states; seed all four
domains from the typed run identity; use that state for model initialization
and dynamic whole-unit dropout; and restore the pre-run states on every normal
or failed close. The source-only episode scheduler itself remains independent
of host RNG. Receipts must bind the seed, initial model-state digest, RNG-domain
policy, and restoration proof so CS-WG and matched ERM can begin from the same
initial state law.

## 4. Route-owned four-fold reader

Do not edit the shared historical fold table or datamodule.

Add a route-owned reader that supports all four CS-WG outer folds from the
typed run spec. It must select exactly the three supplied source descriptors
without globbing or resolving the omitted outer target.

The reader must reuse the closure-bound native M1 parsing primitives:

- `FalconDataModule.prepare_session_data` for native NWB loading;
- `FalconDataset` for the exact M1 window, calibration trialization,
  interpolation, eval-mask, and final-bin behavior contract.

Use the frozen source recipe:

```text
task=m1
window_size=100
calibration_n_trials=10
random_calibration=false
smooth_calibration=false
max_trial_length=1024
standardize_covariates=false
use_intertrials=true
use_calib_intertrials=false
trial_feature_type=raw
interpolate_trials=true
interpolate_trials_kind=cubic
pad_value=-1.0
side_feature_group=none
query_start_trial=0
```

The reader may construct the native dataset directly from those two parsing
primitives. It must not call the ordinary `FalconDataModule.setup`, discover
minival, create validation/test datasets, or infer a fold from directory
contents.

The exact selected NWB bodies must be opened through held, no-follow source
descriptors. The parser must consume the held file identity (for example via
the held descriptor path), and the named source identity, size, and SHA256 must
be revalidated after parsing. Symlinks, hard-link aliases, parent replacement,
path escape, source-body drift, unexpected files, or descriptor/session drift
must fail closed.

## 5. Import namespace and closure law

The production process must import this experiment as:

```text
tfpd_exploration.src.cross_session_worst_group_v1
```

It must not claim top-level package name `src`. The unmodified
`streaming_calibration_exp` runtime keeps its historical top-level `src`
package. Do not delete or replace `sys.modules`, monkeypatch globals, or copy
the parser.

The repaired explicit closure must bind every locally executed eager import
in the native M1 parser path, including the relevant
`streaming_calibration_exp/src` package, `falcon_datamodule.py`, its direct
local data-feature imports, and the route-owned reader. Installed external
packages must be version-attested in source authority. A clean subprocess must
prove that both package trees can coexist under these names before any source
or CUDA action.

## 6. Source audit before CUDA

After the no-data candidate is frozen and independently reviewed, root may
authorize one CPU/source-only audit for the smoke fold:

```text
outer target: 20120924
sources: 20120926, 20120927, 20120928
```

The audit occurs after a durable attempt and before model construction or
CUDA. It must report:

- exact selected source file identities;
- valid window counts;
- exact session calibration digests and equality across rows;
- pooled source-only quantile boundaries;
- per-session stratum sets and exact missing/common topology;
- exact paired epoch step count; and
- a constructible step-zero 11/11/10 B32 input authority.

If the three physical source pools do not expose the exact common Stage-0
stratum topology, stop before CUDA with a typed failure receipt. Do not relax
the stratum definition or invent a fallback in this repair.

## 7. Additive and bounded ownership

Terra owns this implementation. Other agents are active in the same worktree;
do not revert or rewrite their changes.

Permitted changes are limited to:

```text
tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_READER_REPAIR_20260826.md
tfpd_exploration/src/cross_session_worst_group_v1/source_reader.py
tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py
tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py
tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_reader_audit.py
tfpd_exploration/tests/test_cross_session_worst_group_m1_source_v1.py
tfpd_exploration/tests/test_cross_session_worst_group_m1_source_reader.py
```

Edits to the two existing route files must be the minimum necessary to:

1. replace the false per-row-unique calibration law with the exact
   session-digest law;
2. expose the route-owned reader through the existing deferred backend; and
3. add the amended work order and parser dependency graph to the closure.

Do not edit the Stage-0 plan/core, shared M1 code/config, model, datamodule,
another experiment, result, checkpoint, or data file.

The original canonical smoke root may remain the future root only if root
independently proves it is absent and that no prior attempt was minted.
Otherwise stop and require a new root version; do not delete or overwrite it.

## 8. Mandatory no-data tests

At minimum, add tests for:

1. exact predecessor work-order/closure and amended-work-order binding;
2. clean coexistence of `tfpd_exploration.src...` and the historical
   `streaming_calibration_exp` top-level `src` package;
3. all four run-spec folds without changing the shared fold table;
4. exact three-descriptor selection with no directory glob and no omitted
   target resolution;
5. exact native reader recipe and direct use of
   `prepare_session_data`/`FalconDataset`;
6. same-session repeated calibration content accepted and cross-session
   substitution rejected;
7. wrong calibration shape/dtype/digest/session and batch broadcast rejected;
8. row/query/target/eval-mask identity preservation;
9. held-descriptor symlink, hard-link, sidecar, inode, parent, size, and body
   substitution failures;
10. actual synthetic native-dataset materialization through the route reader;
11. source audit stops before model/CUDA when strata differ;
12. exact common-strata success constructs one 11/11/10 B32 authority;
13. attempt-before-source ordering and honest failure progress;
14. static audit CLI imports no Torch, opens nothing, writes nothing, and
    cannot issue the in-process source capability;
15. the original Stage-0 and source-lifecycle no-data tests remain green under
    their corrected semantics;
16. exact seed-42 model initialization/dynamic-dropout RNG setup and restoration
    without changing the caller's Python, NumPy, Torch-CPU, or Torch-CUDA RNG;
17. failure receipts distinguish source-resolution, CUDA-attestation, and
    post-model-construction boundaries without fabricating progress;
18. resource validators reject NaN and infinity in every elapsed/rate field and
    require non-negative integer CUDA memory counters; and
19. CUDA remains uninitialized throughout the no-data suite.

## 9. Stop conditions

Stop without GPU launch if any of the following occurs:

- the shared parser must be edited or copied;
- the two package namespaces cannot coexist without module replacement;
- fold 3 requires opening or resolving its omitted target;
- the exact native M10 tensor cannot be bound per session and per row;
- the reader needs minival, held-out, formal, EvalAI, or target data;
- physical source strata do not meet the frozen common-stratum contract;
- the closure omits an executed local import; or
- any canonical root already exists.

This work order authorizes code construction and no-data validation only. It
does not authorize source NWB access, CUDA, GPU smoke, full training, matched
evaluation, H1, root minting, or retry. Root performs independent review and
issues any later bounded source-audit or smoke authorization. Luna monitors a
launched GPU job read-only at 30-minute intervals and never edits, retries,
signals, or intervenes.
