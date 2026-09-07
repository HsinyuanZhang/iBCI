# QueryAge low-R² runtime/selection alignment audit — 2026-09-06

## Scope, method, and conclusion

This is a **read-only static-code audit** of the H1 QueryAge model's
current-query runtime contract against the direct formal selection scorer, plus
a narrow audit of the M2 selected ext4 target/scale contract.  It does not load
a selected checkpoint, construct a stream, forward a model, inspect a live
worker, alter a cache, or change a frozen protocol.  Therefore its conclusions
are conditional on the inspected source being the source actually imported by
the selected deployment and evaluation jobs.

The H1 implementation specifies the same high-level finite-window operation on
both paths:

```text
raw neural history ending at e  ->  W=700 frontend tokens z[0..699]
                                 ->  current query z[699], read all z[0..699]
                                 ->  four QueryAge blocks -> readout / 20
                                 ->  native velocity at endpoint e
```

The formal selection path materializes a complete direct window and invokes
`candidate.forward_last`.  The generic H1 runtime retains a W=700 raw window,
updates its causal frontend/cache, and reads the final frontend token through
the same `QueryTemporalStack`.  On the code inspected, there is no smoothing,
window-size change, target-one-bin shift, different age convention, temporal
reset at trial completion, or second H1 output scaling operation.

That is a **contract-alignment finding**, not evidence that the actual selected
checkpoint was put into this runtime, nor a proof that every selected endpoint
was replayed equivalently.  In particular, a passing cold-start/boundary test
can prove a boundary case but cannot by itself prove all 2,908 selected H1
positions, their bank binding, the selected EMA state, or the caller's input
stream.

For M2, the inspected ext4 evaluator uses `forward_last(...) / 5`, compares
against the native `target_store`, and archives both in FP64.  The premise that
this M2 path has a `SCALE20` conversion is not supported by this code: `/20` is
the H1 contract; M2's decoder-raw/native bridge is 5x.  No apparent double
scaling or target offset exists on the shown M2 ext4 scoring path.

## H1 comparison matrix

| Concern | Formal direct selection surface | Current-query runtime surface | Static assessment |
| --- | --- | --- | --- |
| Raw input/window | `_ends(..., complete=False)` converts frozen `query_starts` to `s + 699`; `_windows` returns `neural[s:s+700]` for those legal starts. | `reset` constructs `[B,700,N]`, left-pads only a short supplied history in raw space, and `observe` shifts in one raw bin. | Both define a W=700 history with the newest observation at index 699. Selection starts are full windows, so its direct windows do not invoke early-history zero padding. |
| Current query | The QueryAge temporal stack takes `z[:, -1:]`. | `QueryMemoryCache.predict` calls `forward_cached(self.z[:, -1:], ...)`. | Same final frontend token is the query. |
| Memory and age | Every QueryAge layer projects the complete `z`; age is `[T-1, ..., 0]`, so the newest/current key has age zero. | Cache stores layer K/V projected from the same full frontend window and rolls/replaces entries after each bin. | Same intended W=700 independent-memory reader and age direction. |
| Attention eligibility | `read` uses additive age bias and `is_causal=False`; all keys are legitimate because each is at or before the current query. | The cached route calls the same `QueryReadBlock.read`. | Same attention mask convention; it is deliberately not the non-square `is_causal=True` convention. |
| Temporal depth/output | Formal model factory installs 256-wide, 8-head, 4-layer, FFN-512, 16-bucket QueryAge, and `forward_last` performs final norm/readout. | Runtime requires `QueryTemporalStack`, asks its cache for the last hidden state, then performs `readout(final_norm(hidden))[:, -1]`. | Same named QueryAge topology and output placement, subject to an actual state-dict/topology binding at deployment. |
| Bank/unit binding | Formal scorer builds `H1Bank(E0,T,unit_mask)` from the frozen row. | Runtime validates explicit unique `unit_ids` against `E0`, `T`, and mask dimensions; a changed bank rebuilds state. | Both consume an H1 bank. Static code cannot prove the caller bound the selected session bank and unit order. |
| Output units/target | Formal scorer uses `forward_last(x, bank) / 20.0`, then native `velocity[end]`. | Runtime fixes the H1 divisor to 20 and returns `readout(...) / divisor`. | One 20x decoder-raw-to-native bridge on each inspected path. |

## Evidence behind the H1 finding

### Direct selected surface: full windows, final-bin targets, and no scoring-time smoothing

`src/h1_queryage_family_v1/formal_prefix_score.py` is the guarded formal
scorer.  Its selection endpoint is exactly `query_starts + (WINDOW - 1)`
(lines 49–67, with `WINDOW = 700` at line 20).  In `_score`, it obtains a
window with `_windows(row["neural"], ends[...])`, runs
`candidate.forward_last(x, bank) / 20.0`, and appends
`row["velocity"][ends]` as the native target (lines 92–140).  It uses exactly
2,908 selection bins and rejects another cardinality (lines 137–140); complete
evaluation is an explicitly separate 20,325-bin mode.

The shared helper `_windows` in `src/h1_optimized_v2/score.py` creates a
right-aligned W=700 array and fills its suffix with the observations ending at
the requested endpoint (lines 13–17).  For a formal selected endpoint
`e=s+699`, `take=min(e+1,700)` is 700 because the frozen selection start is
nonnegative; the result is therefore exactly `neural[s:s+700]`, with no
synthetic left bins.  The H1 source builder independently describes starts as
windows entirely after calibration and appends `last - WINDOW + 1` only when
the last bin is eval-valid (`h1_temporal_decoder_quick_product_v1/data.py`,
lines 82–90).  The prior coordinate audit records that the cached 2,908
selection starts are full legal W700 windows.

There is no smoothing operation in this direct scorer.  It reads cached FP32
neural rows, builds a tensor, applies the selected candidate, and only casts
the finite output to FP64 for metrics.  It also uses the frozen bank topology
`{E0, T, unit_mask}` without target-derived reconstruction (formal scorer,
lines 70–75).

### The QueryAge operator has one current query and uncontextualized history

`src/two_mainlines_long_v1/current_query_v2/core.py` defines the exact reader
shared by the audited paths.  `QueryTemporalStack.forward` verifies `0 <
T <= W`, selects `query = z[:, -1:]`, and passes that query with the entire
original `z` into each layer (lines 97–109).  `QueryReadBlock.project_memory`
projects layer-specific K/V from the full `z` (lines 35–39).  Its reader:

- creates `age = [t-1, ..., 0]` and maps it into the 16 age buckets (lines
  49–54), so position 699/current has age 0 and position 0 has the greatest
  age;
- permits an optional explicit valid mask only; without one, all W positions
  remain available (lines 55–60);
- calls scaled-dot-product attention with the age bias and `is_causal=False`
  (lines 61–68).  The adjacent comment correctly notes that non-square
  `[one query, T keys]` causal masking would retain only key zero, which would
  be a different operation.

This is not ordinary Transformer K/V caching: historical tokens are not
successively contextualized.  The implementation documentation and code make
the historical memory layer-specific projections of the frontend `z`, while
only the current query is depth-updated (module lines 1–8, 74–79).  Thus use of
an all-history attention read is compatible with a current-time prediction; it
does not imply future-bin access.

`h1_queryage_family_v1/model.py` binds the formal H1 family to W700,
width 256, 8 heads, 4 layers, FFN width 512, and 16 buckets (lines 24–27), and
installs that stack in both FLAT and ROUTE models (lines 30–47).  The inherited
H1 `forward_last` performs its readout after `final_norm` (`h1_family_v1/model.py`,
lines 100–104), matching the runtime's explicit final norm/readout sequence.

### Runtime: raw W700 rolling state, no hidden smoothing, explicit reset rules

The runtime entry here is
`src/two_mainlines_long_v1/current_query_v2/streaming.py`.  Its module contract
says input is one binned neural observation already in training input space,
and explicitly excludes smoothing, normalization, masked-bin dropping, and
session-boundary inference (lines 1–7).  It requires the H1 output divisor to
be exactly 20 (lines 41–63), constructs a `FrontendWindowCache` with the model
window and configured causal kernel, and uses `QueryMemoryCache` only for the
current-query operator (lines 67–76).

On session reset it creates an all-zero W700 raw window and, for a short
provided history, writes only the rightmost real raw observations into it
(lines 135–162).  `observe` rejects an implicit session-ID change, validates
one finite `[B,N]` observation, advances the frontend window, then advances
the memory with the changed frontend indices (lines 180–195).  The native
prediction is exactly `readout(final_norm(hidden))[:, -1] / divisor` (lines
197–204).  A trial `on_done` deliberately retains history unless the caller
explicitly requests `reset_session=True` (lines 212–215).

`latency_opt_v2/exact_window.py` specifies the corresponding rolling frontend
contract.  A one-bin shift constructs `raw = old.raw[:,1:] + next_bin`,
recomputes the left `kernel-1` frontend positions and the final position, and
retains the unaffected causal-local positions (lines 105–117).  With `k=5`,
the exposed changed indices are `0..3` and `699`.  `QueryMemoryCache.advance`
then checks that every supposedly unchanged frontend token equals the prior
token shifted by one; otherwise it rebuilds rather than silently retaining
stale K/V (current-query core lines 180–210).  It likewise rebuilds for a
weight/device/dtype/shape change (lines 185–190).  This is a meaningful guard
against a bad frontend-cache boundary claim, not merely a performance cache.

## Important qualifications: what static alignment does *not* establish

1. **Runtime code correctness is not selected-model correctness.**  The
   generic stream accepts a `model` and `bank`; this audit did not deserialize
   the selected EMA export, verify its state-dict against the QueryAge factory,
   call `model.eval()`, or show that a production wrapper selects this stream.
   A correct generic runtime can still be unused, given a wrong model, or fed a
   wrong bank/unit roster.

2. **A boundary parity result is necessary but not sufficient.**  Startup
   zeros, a one-bin shift, and an explicit reset exercise only a few histories.
   They do not prove the direct/replay identity for every frozen selected
   endpoint, every session/bank, every possible reset/rebind route, or an
   uninterrupted caller feed that includes all bins.  The runtime itself says
   to advance even bins excluded by a behavioral scoring mask (line 181), so a
   caller that omits them would violate the intended history even though the
   stream implementation is correct.

3. **The direct selection path is stateless and offline; the runtime is
   stateful.**  Direct selection materializes each W700 window independently.
   The runtime's equivalence depends on its stated preconditions: raw inputs
   are already in the training space, observations arrive in order one per bin,
   bank/unit roster matches, and explicit reset occurs at a true session
   transition.  This audit does not prove those external facts.

4. **No causal claim about data generation is made.**  All-memory access in a
   window is correct only if the window ends at the current observed bin.  This
   audit traces local array/window indices; it does not independently validate
   NWB bin timestamps, the upstream loader, or any host-side preprocessing
   before `observe`.

These qualifications mean the low R² is not explained or ruled out as a
deployment failure.  They identify a focused next validation, if separately
authorized: replay the frozen selected endpoints through the exact production
binding and compare native predictions to direct `forward_last` outputs,
including each session, bank, and reset history.  That would be a new
evaluation action and is intentionally not performed here.

## M2 ext4: target identity, endpoint/window, and scale

The M2 QueryAge family is a distinct W50 system.  Its model declaration binds
`QueryTemporalStack(... window=50, age_buckets=16)` and states that inherited
`forward_last` retains M2's decoder-raw/native-5x API
(`src/m2_queryage_family_v1/model.py`, lines 23–57).  The training protocol is
explicit: the final-bin training target is native ×5 and MSE is in
decoder-raw space (`train_pair.py`, lines 152–176).  Source-minival scoring
divides a raw forecast by `plan.BEHAVIOR_SCALE` before comparing to native
targets (lines 256–271).

The selected ext4 evaluator has the same conversion:

```text
raw[start : start + 50]  ->  model.forward_last(window, bank, unit_mask) / 5
                          ->  native prediction
target_store             ->  native target
```

Specifically, it hashes the required ext4 files before loading the arrays,
loads `X_store.npy`, `eligible_starts.npy`, and `target_store.npy`, and calls
the frozen geometry verifier before a bank is loaded
(`evaluate_selected_ext4.py`, lines 183–207).  During scoring it requires
FP32 raw data/targets and int64 starts; for every start it stacks exactly
`raw[start:start+50]` and divides `forward_last` by **5** (lines 239–285).
The first batch also checks its singleton calculation against the batched
calculation (lines 274–279).  Thus the shown evaluator has neither a start/end
offset (`start+50` is the W50 exclusive endpoint) nor an additional `/20`.

`m2_family_v1/evaluate_frozen_ext4.py` verifies that ext4 starts and targets
match sealed hashes, that support is disjoint, and that the full query window
is disjoint (lines 51–60).  It records the historical protocol's
`native_divisor: 5` (lines 29–36).  The newer evaluator promotes predictions
and targets to FP64 only for R²/archive processing; it retains the exact FP32
source target through a FP32 round-trip/hash check and preserves the ordered
per-session start identity (`evaluate_selected_ext4.py`, lines 314–357).

Consequently, the only scale conclusion warranted by this review is:

| Path | Decoder-raw target/prediction bridge | R² target representation | Finding |
| --- | ---: | --- | --- |
| H1 QueryAge formal selection/runtime | `/20` prediction; training target native ×20 | native velocity | Consistent in inspected H1 code. |
| M2 QueryAge source train/selection | `/5` prediction; training target native ×5 | native target | Consistent in inspected M2 code. |
| M2 selected ext4 | `/5` prediction | sealed native `target_store`, then FP64 archive | No `/20`, double-scale, or visible target-offset operation. |

The M2 result is likewise static.  It does not certify that an ext4 result was
actually run, that its selected export was the intended one, or that source
files at a later execution still had the inspected hashes.  The evaluator is
designed to reject those identity drifts at run time; this audit did not invoke
it.

## Bottom line

The inspected H1 code is internally aligned around an end-at-current W700
QueryAge operation, full selection windows, final-bin native targets, and one
20x output conversion.  The runtime's rolling frontend/KV mechanics include
specific protections for boundary changes, stale weights, bank changes, and
implicit session changes.  This removes several concrete *static* mismatch
hypotheses, but it does not convert sparse boundary evidence into a proof of
the complete selected/replay surface.

The inspected M2 ext4 code instead has a W50/native-5x contract and preserves
sealed start/target identities.  Treating it as `SCALE20` would itself be a
cross-task scale mistake.  Neither conclusion diagnoses the low R²; both
separate code-level contract evidence from the still-unverified selected
runtime execution.
