# Work Order: CDM-D Matched Score V7 Runtime-Flags Successor

Date: 2026-08-26  
Owner: Terra (implementation and no-data/no-CUDA tests); root (independent audit, authority, scheduling); Luna (read-only long-run monitoring only)  
Status: implementation authorized; data/GPU execution is not authorized by this document

## 1. Purpose

V6 correctly repaired the reserved-root lifecycle and durably published an
attempt before physical preparation.  Its sole reviewed execution then
failed closed in `prepare` because the inherited V1 physical runtime asked
the V5 wrapper module for `RuntimeFlags`.  The V5 wrapper imports the actual
V1 source-execution module as `v1`; it does not define `RuntimeFlags` at its
own top level.

V7 is a narrow engineering successor.  It must repair that typed runtime
composition while leaving the accepted CDM-D science unchanged.

## 2. Immutable V6 predecessor

V7 must descriptor-load and exact-validate all of the following before it
reserves or publishes a V7 authority or result root.

### V6 authority

- root: `tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v6`
- `official_preflight.json` SHA-256:
  `52ee183b91bb236131f90e7316404ba6b79c7ba0b8445f242296280d015e7b85`
- `root_authorization.json` SHA-256:
  `dac888d29ee188f187cf6c40085d12ef6bec2401ecf366b797d4ba9909185971`
- V6 identity SHA-256:
  `116e1f2b795767bcced31176393d971b678f07068af23326806fceb4f45e3dac`
- V6 implementation closure SHA-256:
  `20db2a1c473fa1b2986e9b78ecf1d9c61343e6a041e576f5e12bf2cc533b9f88`

### V6 failed score graph

- root: `tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v6`
- exact topology: `attempt.json`, `failure.json`, and their canonical
  `.sha256` sidecars; no other leaves
- all four leaves are regular, non-symlink, mode `0444`
- score-directory `(st_dev, st_ino)`: `(64512, 28863215)`
- score-directory mode: `0755`
- `attempt.json` SHA-256:
  `0c20ad8405768781e8e398c2e8663acddd5939f5249171eba8aa6001fc65c4ee`
- `failure.json` SHA-256:
  `45cf3867933b0c1b2bdd55780ea32e106c734696a26d32364c7163fc3e99f07f`
- launch log SHA-256:
  `74967ab250904d3a7bfebe8e3d29618e83202ae43fca2e7c47a707570b4196ff`
- failure stage: `prepare`
- failure class: `AttributeError`
- failure error SHA-256:
  `09e21f2b3f72be98e5e3b62f8a67bce253e6fa09680411b6a7e22419aca9e825`
- honest access state: sealed checkpoint material opened; no within/external
  assets opened; CUDA not initialized; zero full-system/group forwards; zero
  target backward, optimizer, or update calls; no input authority or terminal
  published

The historical V5 authority and empty V5 score-root incident, and the accepted
V5 88-body source gate, remain transitive immutable predecessors and must
continue to validate.

## 3. Exact repair boundary

The only permitted runtime change is a typed, closure-bound way for the
inherited physical scorer to construct the V1 `RuntimeFlags` object when the
selected source-execution route is a successor wrapper such as V5.

An acceptable implementation is:

1. add one backward-compatible V1 physical method/hook for constructing the
   runtime flags;
2. keep the V1/V5/V6 default behavior unchanged;
3. let the additive V7 physical subclass resolve the V5 wrapper's exact
   closure-bound `v1` dependency and construct `v1.RuntimeFlags` from that
   module;
4. exact-check module name, root-relative path, and closure SHA before use;
5. do not mutate module globals or `sys.modules`, and do not copy the parser,
   evaluator, model, or lifecycle.

Do not add a fallback based only on `hasattr`, do not catch arbitrary
`AttributeError`, and do not alias a foreign flags object into the V5 module.

## 4. Science contract is unchanged

V7 must preserve the V5/V6 experiment byte-for-byte in meaning:

- the same sealed Cell-D and CDM-D SWA/model paths;
- the same fixed within-6 and external-15 assets and input convention;
- the same independent activity-memory semantics;
- budgets in exact order `M30 -> M10 -> M4`;
- two surfaces times two systems per budget, exactly 12 cells;
- complete the nonadaptive matrix even if an earlier budget is negative;
- B128, last-bin variance-weighted per-session R2, equal-session summaries,
  and the same paired bootstrap/gates;
- no target gradients, optimizer steps, updates, labels in state, normalizer
  refit, or formal/H1 access.

No network topology, parameters, weights, carrier math, FIFO law, K4 grouping,
metric, roster, or threshold may change.

## 5. Fresh V7 roots

- authority:
  `tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v7`
- result:
  `tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v7`

V5 and V6 roots are immutable historical evidence.  Never delete, rename,
reuse, or populate them.

## 6. Required no-data tests

Before freeze, tests must prove at least:

1. the exact V6 authority and failed four-leaf result graph are required and
   descriptor-held before V7 authority/result reservation;
2. missing, substituted, recomputed, reordered, wrong-mode, symlinked, or
   inode-replaced V6 leaves fail before V7 output;
3. the V7 runtime-flags factory returns the exact V1 `RuntimeFlags` type from
   the authenticated dependency;
4. top-level V5 `RuntimeFlags` absence is reproduced, while the reviewed V7
   seam succeeds without monkeypatching;
5. wrong dependency module/path/SHA or a forged flags class fails before
   sealed model load or CUDA;
6. a real CPU/no-CUDA or faithfully injected physical-prepare seam reaches
   beyond flags construction and exercises the V5 executor interface;
7. V1/V5/V6 regression behavior remains unchanged;
8. reserve -> held-artifact validation -> durable attempt -> prepare ordering
   remains exact;
9. a complete synthetic 12-cell lifecycle publishes atomic score+terminal,
   while a post-attempt physical failure publishes only an honest failure;
10. static CLI is Torch-free and public flags cannot mint a capability.

## 7. Stop conditions and ownership

Terra may edit only the minimal backward-compatible shared physical seam and
new additive V7 workorder/package/CLI/tests.  Terra must not open canonical
result roots, NWBs, checkpoint tensors, CUDA, or GPU and must not mint or
launch.

After Terra freezes exact bytes, root independently audits the closure,
predecessors, fixed target metadata, fresh roots, and selected idle GPU.  Only
root may mint V7 authority and authorize one launch.  Any failure produces an
immutable failure graph and no automatic retry.  If the run becomes long,
Luna monitors read-only at 30-minute intervals without edits, signals, or
retries.
