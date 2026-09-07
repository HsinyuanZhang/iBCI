# Optimization notes: CDM-D score/execution throughput (line-level, for the main agent)

Date: 2026-08-26 · Author: operator-side review · Status: SUGGESTIONS ONLY — nothing modified.
Profile (from v8 receipts + live): 13.9 windows/s, peak VRAM 642 MiB / 24 GiB, one CPU core
pegged, GPU 11%. Chunks: 129,214 for 28,494 windows ≈ 4.5 forwards/window. Same pathology
family as the TFSR throughput round (CPU-dispatch bound, tiny effective batches).

## Hot-spot ranking (from reading `src/causal_dual_memory_cell_d_score_v1/physical.py`)

### O1 — compute the B3S identity ONCE per trial-state, not per chunk-forward  [expected: the dominant fix]

`physical.py:890` calls
`state.model(neural[start:stop], calib_trials=calibration[start:stop], side_features=side[start:stop])`.
`calibration` is the expanded `[B,30,1024,64]` stack (`physical.py:880`), so **every chunk
forward re-runs the B3S calibration encoder over the full 30×1024×64 stack**, batch-expanded
to the chunk size — and this is repeated ×2 (repeat check) ×4 (groups) × every chunk × every
trial. The identity is a pure function of (activity stack, side/carrier), which change ONLY
at trial boundaries.

Fix: `StreamingSpintModel.forward` already accepts `identity=` (streaming_spint.py:682-696;
passing it skips the encoder entirely). Per trial-state:
1) `identity = model.compute_identity(calib_stack, side_features=active_side)` once (shape [N,W]
   or expanded per its contract — check `_expand_identity`, streaming_spint.py:66-73);
2) pass `identity=identity` in every chunk/group/repeat forward within that trial;
3) re-derive only at committed transitions (the `transition_records` boundaries already tell
   you when the stack/side changed).
Numerics: identical by construction (same encoder output, same decode path) — the receipt
anchor (per-session governing R² vs the v8 sealed values to ≤1e-7, and
`prediction_sha256` parity where the identity bytes are unchanged) must pass bitwise or
near-bitwise; any drift means the identity derivation differs and must be investigated, not
tolerated.

### O2 — stop paying ×2 on EVERY chunk for the repeat-forward integrity check

`physical.py:891-899` runs the model twice per chunk and asserts `torch.equal(first, second)`
+ dropout counters. This is a per-chunk receipt audit in the hot loop. Keep the audit but
sample it: once per trial-state (first chunk after any committed transition) + once at a
fixed mid-session window, recorded in the evidence block with its coordinates. 2× saving on
everything not audited; the invariant you actually care about (eval determinism per state)
is preserved.

### O3 — group forwards: subsample or interval-gate

`_group_predictions` (`physical.py:920-944`) decodes the trial's full neural windows for
each of the 4 held-group views, every completed trial — this is ~4/5 of all chunks. If the
trust gate consumes only scalar departure statistics from the group trajectories:
(a) compute group evidence on a fixed subsample of windows per trial (e.g. stride-4), or
(b) run group evidence every K-th trial (K=3-5) with the gate holding the last statistics
    between evaluations (disclose the hold), or
(c) at minimum: combine (O1) so group forwards reuse the precomputed identity.
Any change here alters gate-evidence granularity → it is a SYSTEM CHANGE, not pure speed:
must be a new versioned cell (e.g. v9-eval) with its own anchors, not a silent replacement.

### O4 — raise EVAL_BATCH_SIZE

`plan.py:83: EVAL_BATCH_SIZE = 128` with 642 MiB peak of 24 GiB. Raise to 1024-2048 and
measure; the decode path is small (3.5M params) so memory scales gently. Keep the digest
order stable (`physical.py:895` hashes chunk bytes sequentially — larger chunks change the
chunk partition but not the concatenated bytes; confirm `torch.cat(chunks)` equality and
re-derive `prediction_sha256` under the new partition before swapping it into receipts, or
keep the digest at fixed logical boundaries independent of chunking).

### O5 — cross-session/fold process concurrency

Peak 642 MiB → 4-8 concurrent processes per GPU (15 external sessions or the CS-WG four
folds), zero numerics change (each process is independent). Near-linear wall-clock.
This is the only item safe to apply to RECEIPT-PRODUCING runs without re-anchoring.

### O6 — parse/caching tier (CPU side)

Session parse measured ~35% of wall on the CPU runs (Z1: 259 s of 749 s). Materialize each
session's tensors once to a disk cache keyed by the existing session digests; keep the cache
outside result roots with its own disclosure.

### O7 — training tier only (CS-WG): TF32 + fold concurrency

For CS-WG training runs: `torch.backends.cuda.matmul.allow_tf32 = True` +
`cudnn.allow_tf32 = True` (receipts currently pin both False — v8 runtime env). 2-3× on
3090. Governing/scoring runs KEEP fp32 (bitwise receipt discipline). Run the four folds
concurrently (O5). Do NOT attempt torch.compile (triton 3.2.0 inductor incompatibility —
sealed TFSR throughput v2 receipt) or CUDA graphs (measured no-gain, TFSR v3).

## Anchors (mandatory before any optimized path replaces the current one)

1. Per-session governing R² vs the sealed v8 values: max |Δ| ≤ 1e-7 (the standard the Z1/P4
   runs set).
2. `prediction_sha256` parity per session where O1/O2/O4 are pure speed (identity bytes and
   concatenation unchanged → digests must match bitwise).
3. O3 changes evidence granularity → new cell version + fresh anchors vs the current gate
   decisions (transition acceptance sets must be compared, not assumed).
4. Throughput receipts: record windows/s before/after per item so the contributions are
   attributable (the TFSR throughput v1-v3 receipt format is the template).

## Expected combined effect

O1 alone: likely 5-20× (removes the dominant redundant FLOPs). O1+O2+O4: 20-80× on the
decode path. O5: additional 4-8× wall-clock for multi-session runs. Realistic landing:
13.9 → several hundred windows/s single-stream, minutes instead of ~14 min per cell.
