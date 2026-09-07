# Work Order: CDM-D Matched Score V8 Physical-Helper Successor

Date: 2026-08-26  
Owner: Terra (implementation and no-data/no-CUDA tests); root (independent audit, authority, scheduling); Luna (read-only long-run monitoring only)  
Status: implementation authorized; data/GPU execution is not authorized by this document

## 1. Purpose

V7 correctly repaired the V5-wrapper `RuntimeFlags` composition and its sole
run passed reserved-root validation, immutable attempt publication, sealed
checkpoint loading, CUDA attestation, fixed-input materialization, and 4,238
full-system forwards. It then failed closed in the M30 group-control path
because the inherited V1 scorer looked up
`_variable_prefix_array_digest` on the V5 physical wrapper. That wrapper
closure-binds the actual helper module as `v1_physical`; it intentionally does
not re-export V1 private helper functions or `ConcreteCellDFourGroupExecutor`.

V8 is a narrow engineering successor. It must authenticate and use the exact
V1 physical helper dependency behind the V5 wrapper while leaving the accepted
CDM-D science unchanged.

## 2. Immutable V7 predecessor

V8 must descriptor-load and exact-validate the following before reserving or
publishing any V8 authority or result root.

### V7 authority

- root: `tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v7`
- `official_preflight.json` SHA-256:
  `b88ebb2396db494d3e68600468987d64c280587ace6a0046d7d520658d12cd0f`
- `root_authorization.json` SHA-256:
  `320b86117fbb255b369fd080fa180f5c4e3259cdc68c06f4fe31828330bf0db3`
- identity SHA-256:
  `96ba7001660fbf88326e719ea02e364812a985385bf407e576da840c657d4667`
- implementation closure SHA-256:
  `0e5010316cbe164fe6060e58bb42aebb8fbe181f54f72c41e7375c78f961c6f0`

### V7 failed score graph

- root: `tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v7`
- exact topology: `attempt.json`, `input_authority.json`, `failure.json`, and
  their canonical `.sha256` sidecars; no other leaves
- all six leaves regular, non-symlink, mode `0444`
- score-directory `(st_dev, st_ino)`: `(64512, 28863240)`
- score-directory mode: `0755`
- `attempt.json` SHA-256:
  `16e320f00cd6ea7ebcc32ca4cc45e80da8595e41816f8d3930e8c97b30c91cb7`
- `input_authority.json` SHA-256:
  `d5e5ececc27b657473181d39626b94c0c200b8c924caf7f074d136d834372777`
- `failure.json` SHA-256:
  `4eff5f7639026ad8f3abf60477248023fe54872953b2b6854c9d47bfc519f4c8`
- launch log SHA-256:
  `08eba1db54baec7d97af65ee1494a3cd7225ab3150d2ff2ecd2a1493c84664ac`
- failure stage: `budget_m30`
- failure class: `AttributeError`
- failure error SHA-256:
  `7a295bf081bb98796cdf730bb1bcc52cd51c4d685c42d2bbf17f2e532ce0d31a`
- exact missing symbol:
  `source_execute_physical_v5._variable_prefix_array_digest`
- honest live facts: sealed checkpoint opened; CUDA initialized; within and
  external assets opened; input authority published; full-system forward
  count `4238`; group forward count `0`; no score/terminal; target backward,
  optimizer, and update counts all zero

The V6 authority/failure graph, historical V5 authority/empty-root incident,
and accepted V5 88-body source gate remain transitive immutable predecessors.

## 3. Exact repair boundary

The only permitted runtime change is a typed, closure-bound way for the
inherited V1 scorer to select its base source-physical helper module when the
selected executor module is a successor wrapper such as V5.

An acceptable implementation is:

1. add one backward-compatible V1 physical helper-module hook;
2. retain the exact historical V1 behavior by default;
3. V8 alone authenticates `source_execute_physical_v5.v1_physical` by exact
   module object, module name, root-relative path, and current closure SHA;
4. route all three inherited helper uses through that authenticated module:
   `_variable_prefix_array_digest`,
   `ConcreteCellDFourGroupExecutor._normalized_active_t4`, and
   `ConcreteCellDFourGroupExecutor._torch_variable_prefix_forward`;
5. retain the V5 one-shot executor itself for finalized-row capture/consume;
6. do not copy group-forward logic, monkeypatch globals, mutate `sys.modules`,
   use `hasattr` fallback, or catch arbitrary `AttributeError`.

The helper module and executor module are different roles. Receipts/tests must
make that separation explicit so V8 cannot accidentally fall back to the V1
executor and thereby lose the V5 finalized-row/independent-activity semantics.

## 4. Science contract is unchanged

V8 must preserve exactly:

- sealed Cell-D and CDM-D model/checkpoint bindings;
- fixed within-6 and external-15 assets and common post-first-30 query pool;
- independent activity transition and V5 one-shot finalized-row semantics;
- budget order `M30 -> M10 -> M4`;
- two surfaces times two systems per budget, exactly 12 cells;
- complete the matrix even if M30 is negative;
- B128, fixed last-bin variance-weighted per-session R2, equal-session
  summaries, paired bootstrap, thresholds, and gates;
- no target gradients, optimizer steps, state labels, normalizer refit, or
  formal/H1 access.

No network, weight, parameter, parser, carrier, FIFO, K4, metric, roster,
threshold, or target chronology change is permitted.

## 5. Fresh V8 roots

- authority:
  `tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v8`
- result:
  `tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v8`

V5, V6, and V7 authority/result roots are immutable history. Never delete,
rename, reuse, or populate them.

## 6. Required no-data tests

Before freeze, tests must prove at least:

1. exact descriptor-held V7 authority plus six-leaf failure graph is required
   before V8 authority/result reservation and at final revalidation;
2. missing, extra, substituted, recomputed, wrong-mode, reordered-sidecar,
   symlinked, or inode-replaced V7 evidence fails before V8 output;
3. real import proves the V5 physical wrapper lacks the three required
   top-level helper surfaces and binds its exact `v1_physical` dependency;
4. wrong dependency object/name/path/SHA, forged helper, or forged
   `ConcreteCellDFourGroupExecutor` fails before data/checkpoint/CUDA;
5. the authenticated helper provides all three exact required callables;
6. a faithful physical seam executes digest, normalized-T4, and variable-prefix
   forward dispatch through V1 helpers while the finalized-row executor remains
   the exact V5 one-shot type;
7. V1/V5/V6/V7 regression behavior is unchanged;
8. reserve -> held-artifact -> attempt -> prepare ordering remains exact;
9. complete synthetic 12-cell score+terminal and honest post-attempt failure
   paths remain valid;
10. static CLI imports no Torch and cannot mint an opaque capability.

Do not declare the seam tested merely by checking names. At least one CPU or
faithfully injected call must pass arguments through each of the three helper
dispatches and observe the V5 executor role separately.

## 7. Stop conditions and ownership

Terra may edit only the minimal backward-compatible V1 physical helper seam
and new additive V8 workorder/package/CLI/tests. Terra must not open canonical
result roots, NWBs, checkpoint tensors, CUDA/GPU, mint authority, or launch.

After freeze, root independently audits current bytes, all live predecessors,
fixed metadata, fresh V8 roots, and an idle compatible GPU. Only root may mint
and authorize one V8 execution. Any failure is terminal evidence and is not
retried. Luna monitors any long run read-only every 30 minutes without edits,
signals, or retries.
