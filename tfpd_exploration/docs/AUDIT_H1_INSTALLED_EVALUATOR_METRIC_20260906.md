# H1 installed evaluator versus formal QueryAge selection metric

Scope: read-only inspection of the locally installed package, not a statement
about a remote EvalAI service revision. No model, checkpoint, cache artifact,
or GPU was opened.

## Exact installed files inspected

- `/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/evaluator.py`
  SHA-256 `2b848f84ef620eac82ce53d331d03727d9ae4ea9a5f2e5edb6e6557537ce4418`
- `/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py`
  SHA-256 `7638d839ec2085c7dde71ec9ca9676cbf1256c79fdcb327c4ac6487bfe5c9b05`

These are installed local source bytes. They are not proof of which package
revision a remote service is running.

## Finding: the reduction differs

The frozen QueryAge formal selection scorer reports **one pooled concat FP64
R²** across all 2,908 endpoint rows and seven output columns:

- conversion and formula: `formal_prefix_score.py:25-36`;
- ordered 13 minival rows: `:39-46`;
- endpoint extraction: `:49-67`;
- pooled return: `:136-142`.

Its complete-only diagnostics subsequently add unweighted equal-session mean
and a worst session (`:143-149`), but those are not the governing selection
value.

The installed evaluator's movement regression path instead:

1. concatenates datasets, masks predictions with `preds[eval_mask]`, and uses
   the already masked target rows (`evaluator.py:727-735`);
2. reconstructs contiguous boundaries for each reduced H1/M2 session from
   per-dataset lengths and masked-point counts (`:727-736`);
3. calls `sklearn.metrics.r2_score(target_session, prediction_session,
   multioutput='variance_weighted')` independently for each session
   (`:736-737`); and
4. returns `R2 Mean = np.mean(r2_scores)` and standard deviation (`:738-741`).

Therefore the installed evaluator's displayed H1 summary is an **unweighted
mean of per-session, variance-weighted multioutput R² values**. Correction:
for a fixed session with nonzero SST in every output, sklearn's
`variance_weighted` result is algebraically the same as the formal per-session
SSE/SST helper:

`sum_j SST_j * (1 - SSE_j/SST_j) / sum_j SST_j = 1 - sum_j SSE_j/sum_j SST_j`.

The substantive ordinary difference is instead **across sessions**: the
installed `R2 Mean` averages session values unweighted, whereas formal
selection is a single pooled concatenated SSE/SST ratio. Edge cases remain if
an output/session has zero variance, and the installed evaluator's
`reduce_key` dataset-to-session grouping must match the cache grouping before
the two summaries can be compared.

I verified the equality in the installed `spint` environment using a small
deterministic 3×2 NumPy fixture: sklearn `variance_weighted` was
`0.9469572368421053`; direct pooled-per-session SSE/SST was
`0.9469572368421052`. I also read the existing Original 20,325 NPZ only:
for all 13 `session_id` values, direct per-session SSE/SST and sklearn differed
by at most floating-point roundoff (absolute maximum `5.56e-16`).

This is a metric-definition difference, not evidence that the frozen formal
score is erroneous. The formal recipe explicitly governs epoch selection by
pooled native FP64 R²; changing it during the live authorized run would be an
unauthorized protocol change.

## Mask, session, and output-order comparison

- H1 dataloading reads `OpenLoopKinematicsVelocity` and the H1 `eval_mask`
  (`dataloaders.py:93-106`). The mask is a time-axis boolean array; the target
  is whatever channel order the NWB velocity array supplies.
- Prediction collection keeps time-by-output prediction arrays, asserts output
  width equals the configured decoder `out_dim` (`evaluator.py:496-518`), and
  retains each file's target and mask in the same collection ordering
  (`:540-552`). No channel permutation is visible here.
- For H1/M2 multi-dataset sessions, evaluator aggregation maps each datafile
  to a reduced session ID (`evaluator.py:252-266`) using `reduce_key`
  (`evaluator.py:123-133`), then records dataset lengths before concatenating
  and applying the mask (`:283-294, 727-736`). For H1's `S...` labels,
  `reduce_key` returns the part before the first underscore; the separate
  `Run...` branch returns the second underscore-delimited part and must not be
  mistaken for H1's branch. The local H1 held-in reduction keys are `S0_`
  through `S5_` before reduction. This is not a claim that the formal cache's
  13 timestamped session keys are already the same grouping. The Original NPZ contains 13 timestamped
  `ses-...` keys across six dates, with some dates having two or three runs;
  this observation is about local archive bytes, not remote-service behavior.
- Formal scoring orders cache session names lexicographically and emits
  `session_id` plus `end` metadata on complete surfaces
  (`formal_prefix_score.py:39-46, 122-149`). It does not use the installed
  evaluator's `reduce_key` grouping because its cached minival already has 13
  session rows.

Thus there is no static indication of an output-channel reorder or inverse
mask use in either path. There is, however, no authorized current-array proof
in this audit that cache session names/endpoints equal the installed
evaluator's original file grouping; that is a separate data-identity question.

## Consequence for low-R² interpretation

The live formal values around 0.21 are formal pooled-selection R². They should
not be compared numerically to an installed-evaluator `R2 Mean` without
computing both reductions on the same prediction/target/mask rows after the
run. A discrepancy can arise from cross-session weighting, `reduce_key`
grouping, or zero-variance handling; it would not by itself establish a scale,
channel, or alignment bug.

No frozen scorer is modified or recommended to be modified here. A future
post-completion descriptive report can safely include both reductions, clearly
labelled as `formal pooled FP64 R²` and `installed-local evaluator R2 Mean`,
provided it binds the same endpoints and records the local package hashes
above.
