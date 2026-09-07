# M2 selected QueryAge cached runtime: read-only profile plan

## Scope and interpretation

This is an operator-level inspection of the selected M2 QueryAge cached
runtime.  It is not a benchmark result, optimization implementation, quality
claim, or speed claim.  The only measurement surface for any later work is
the already-authorized, whole-public-call B7 comparison using the selected
plain-EMA FLAT and ROUTE exports (including the learned ROUTE routing state)
and the same continuous source-train input sequence.  No component timing can
replace that receipt.

The current cached implementation is exact by construction only because the
completed selected-export proof establishes it.  Any code change below would
need a new proof binding to the same selected exports, raw windows, public
outputs, cache invalidation cases, and source/archive authority; it cannot be
assumed equivalent merely because it uses the same weights.

## What remains on every public `predict()`

The cache removes full-window K/V projection for the four QueryReadBlocks.
It does **not** make the entire W50 decoder constant-cost.  For actual B7,
the following work is still on the public path.

| Boundary | Current exact work | Why it remains |
|---|---|---|
| Input and safety audit | NumPy FP32/contiguity/finite check; model/bank/derived-tensor signatures; cache signature checks | The runtime deliberately detects model, bank, mask, or cache mutation rather than reusing stale derived state. |
| W50 rolling state | `torch.cat` shifts `[7,50,96]` raw history; a second `torch.cat` forms `[7,50,256]` frontend history | The finite W50 window is explicit and public output must reflect the new bin plus the shifted left boundary. |
| Local spatial repair | Causal k=5 repair only for positions `0,1,2,3,49`; lifted M2 set frontend/attention for these five tokens | A dropped left bin changes the finite-window zero-boundary prefix, while the new bin changes the newest receptive field.  ROUTE's learned route bonus is part of this exact frontend. |
| Cache rollover | Validates shifted unchanged tokens, projects K/V for the five changed frontend tokens in every one of four blocks, allocates/fills new K/V tensors, and clones current frontend `z` | QueryAge memories are independent per original frontend token.  Only these five K/V positions need projection, but the stored W50 sequence must still be rolled exactly. |
| Current-query reader | Four depth-ordered query projections, Q-to-50 attention reads, output projections, LayerNorms, and 256→512→256 FFNs | The current query changes at each depth; it cannot be reused from the prior call. |
| Output | final norm/readout, native `/5`, owning contiguous FP32 NumPy result | This is the selected model's public B7 boundary. |

The key unavoidable arithmetic after caching is therefore four current-query
reads over 50 memory positions plus four FFNs, not four full W50 K/V
projections.  The spatial frontend and memory-buffer traffic can still be
material on CPU at B7.

## Candidate component boundaries

If a later, separately authorized profiling pass is desired, record elapsed
time only around the following boundaries.  The end-to-end public `predict()`
measurement remains primary; these boundaries are diagnostic only and must be
outside the official public-call timing loop or explicitly accounted for.

1. Runtime audit/signature validation.
2. Raw W50 rollover.
3. `repaired_local_features_linear` and lifted spatial `repair` together.
4. Frontend W50 rollover.
5. `QueryMemoryCache.advance`, subdivided into changed-token K/V projection,
   unchanged-token validation, and K/V rollover/copy allocation if practical.
6. `QueryMemoryCache.predict` / four `QueryReadBlock.read` calls.
7. Final norm/readout plus NumPy ownership copy.

The component receipt should retain B7 dimensions, the selected-arm identity,
the actual nonzero ROUTE routing state, exact call index, and whether a reset
or cache rebuild occurred.  It must not profile FLAT alone and infer ROUTE,
or use a homogeneous/toy bank as deployment evidence.

## Operator-preserving ideas worth evaluating only after a new gate

These are implementation hypotheses, not approved changes.

1. **Preallocated double buffers for raw, frontend, K, V, and cached `z`.**
   Current rollovers use `torch.cat`, `empty_like`, and cloning.  Two
   fixed-shape buffers with explicit copy/rotate operations could reduce
   allocator pressure and temporary objects while retaining the same logical
   W50 tensor contents.  The cache's public mutation/invalidation semantics
   would have to remain intact.  Exact FP32 equality is not automatic: changed
   write ordering and aliases require the full proof.

2. **Avoid advanced-index temporary construction in the unchanged-token
   check.**  `QueryMemoryCache.advance` builds Python index lists and compares
   indexed tensors on every call.  A fixed W50 shift invariant can potentially
   be checked with fixed slices or a proof-gated debug/audit mode.  Removing
   the check outright is not operator-preserving: it is the guard that forces
   a rebuild when an unlisted frontend position changed.

3. **Fuse the frontend roll assignment, not the spatial algebra.**  The five
   repaired rows are already the minimal k=5 affected set.  Replacing the
   frontend `cat` with preallocated slice assignments can be exact at the
   tensor-value level.  Fusing the local-convolution repair, MLP, set
   attention, or ROUTE bias into a different kernel is a separate numerical
   claim and may change FP32 rounding.

4. **Cache static age/bias bookkeeping only when its exact tensor identity is
   maintained.**  Each query read deterministically constructs the W50 age
   bucket vector/bias shape.  A prebuilt CPU FP32/broadcast-compatible bias
   view could remove small repeated setup work.  It must preserve current
   dtype, device, heads, and `valid_mask` semantics exactly.

5. **Treat audit frequency as a safety-policy decision, not a free
   optimization.**  Signature scans are likely small relative to the four
   FFNs but should be measured.  Reducing them changes when external mutation
   is detected, and therefore belongs behind a clear runtime contract and the
   mutation/invalidation proof cases.

## Non-candidates without a new network claim

- Replacing the four depth-updated current-query reads with a causal state,
  one-layer update, or prior-call query is not justified: each layer consumes
  the newly updated query against its own independent memory.
- Skipping the five repaired spatial positions is invalid at the finite W50
  left boundary/newest bin, even though positions 4–48 shift unchanged.
- Freezing or zeroing the ROUTE route bonus, using FLAT timings as a ROUTE
  proxy, or modifying selected weights is outside an operator-safe runtime
  optimization.
- Changing thread count, batch roster, source sequence, Original wrapper, or
  comparing a component timer to public B7 latency would not answer the
  selected benchmark question.

## Decision rule

Before any candidate implementation, first use the existing B7 receipt to
identify a component whose share is material and repeatable under the exact
selected FLAT and ROUTE runtimes.  Then make one isolated implementation
change, run the complete selected-export/native/cache-equivalence proof anew,
and finally rerun the same public B7 smoke/formal protocol.  Until both
closures pass, report no speed conclusion.
