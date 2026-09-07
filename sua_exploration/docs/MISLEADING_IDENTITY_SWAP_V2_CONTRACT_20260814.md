# Misleading-identity swap v2: development Stage-P contract

**Status:** development live-integration implementation with one canonical immutable CPU preflight;
no GPU cell or target-domain scorer has been launched.
**Scope:** DANDI 000688 SUA, sub-C development source/validation and sub-M external development only.
The six formal sub-C test sessions remain sealed and must never be resolved or opened.

## Scientific intervention

The experiment changes only the activity-derived B3S identity row.  For a session and Lightning
`current_epoch`, exactly one deterministic partial involution is applied to
`mean_feat = sum_feat / trial_count` after pooling and before any side feature is concatenated.  Query neural
activity, the unit's visible T4/Z4 carrier row, electrode identifiers, targets, decoder, support chronology,
and optimizer are not permuted by the intervention.

The frozen swap fraction is `0.5`.  The donor map is fixed for every batch/window of a session/epoch.  Internal
epoch identity is the zero-based Lightning `current_epoch`; receipts additionally report the one-based epoch
number.  Selected units are paired as an involution and all unselected units remain fixed points.

## Hidden matching authority

T4 and Z4 siblings consume the same immutable matching-authority bytes.  The matching descriptor is the
source-normalized ordinary T4 vector `[m_cos_phi, m_sin_phi, m, b]` from the same chronological first-30
rewarded support used by the ordinary carrier.  `b` is the rate coordinate; no additional rate column is used.
For Z4, these unmasked values are hidden matching metadata only: the model-visible carrier remains exact zero,
and the hidden descriptor is never concatenated, decoded, optimized, or exposed to the query path.

Every authority records exact descriptor shape/dtype/SHA and, for every session/epoch, the seed, permutation,
selected mask, selected count, actual fraction, and pair-distance summaries.  T4 and Z4 may not independently
rebuild an allegedly equivalent authority after launch.

The source authority is built from exactly the strict 27 source-training NWBs and binds their roster/files,
the M30 ordinary-T4 feature semantics, normalizer values, frozen matched-A2 normalizer SHA, and immutable
authority bytes.  Validation and formal session names remain inert during this source-only build.  Target
diagnostic mappings use a separate truthful schema (`source_only=false`, target opened, zero target updates),
are built once per development domain from its M30 support, and are byte-shared across all four cells and both
visible-carrier arms.  A target authority can never validate as the source-training authority.

## Stage-P matrix

Seed 42 contains four fresh source-training cells under one config, initial-state, runtime, data roster,
normalizer, and matching authority:

| carrier | clean training | swap training |
|---|---|---|
| T4 | `clean_t4` | `swap_t4` |
| Z4 | `clean_z4` | `swap_z4` |

All cells use task-only loss, B3S, M30 chronological activity support, M30 T4 support, 12 training epochs,
and the fixed epoch-5--12 scoring window.  Each unchanged checkpoint bundle is scored on within-sub-C
development and external-sub-M development.  Normal scoring uses clean identity input.  A separate deliberate
swapped-input mode is a diagnostic, not a fifth training arm.

Primary Stage-P gates on clean-input scores are:

1. external T4 lift `R2(swap_t4) - R2(clean_t4) >= +0.03`;
2. within-sub-C T4 delta `R2(swap_t4) - R2(clean_t4) >= -0.03`.

The carrier interaction `(swap_t4-clean_t4) - (swap_z4-clean_z4)` is always reported and cannot rescue either
primary gate.  The deliberate-input robustness diagnostic is

`[(swap_train,test_swap)-(swap_train,test_clean)] - [(clean_train,test_swap)-(clean_train,test_clean)]`

for external T4.  Its sign and per-session values are reported, but it is non-rescuing: it neither passes a
failed primary gate nor authorizes expansion.  Seed 42 stops on either primary failure.  Seeds 43/44 and any
M2/RT migration require a successor contract; they are outside this Stage-P authorization.

## Execution and interpretation boundaries

- Clean/default-off B3S must be bitwise identical to the ordinary parent forward.
- Training swap, clean evaluation, and deliberate swapped-input evaluation are explicit disjoint modes.
- Target-domain labels may be used only by the already disclosed first-30 T4 support construction and final
  metric.  There are zero target optimizer/backward/update steps and no target normalizer fit.
- No donor rule, fraction, descriptor, epoch window, or threshold may be tuned from target scores.
- SetKV, value-mask, A2 sealed files/checkpoints/receipts, shared production files, and formal data are not
  modified by this implementation.  A2 is a read-only protocol/reference lineage, not an output location.
- Four cells must load the same immutable initial-state bytes and record its state-dict SHA before training.
  Each cell writes every epoch checkpoint and only epochs 5--12 enter scoring.  Target-authority preparation,
  GPU training, and GPU scoring are separately authorized operations; dry-run and queue-bridge defaults are inert.
- Every live cell, domain score, target authority, source authority, and aggregate is an O_EXCL immutable body
  plus SHA-256 sidecar.  Any occupied body or sidecar fails closed rather than repairing or overwriting it.
- Cell training, target-authority construction, scoring, and aggregation must all consume the same canonical
  official preflight and exact live implementation closure.  Each domain score must reproduce the complete
  sealed matched-A2 seed-42 trial-30 semantics and source normalizer authority before target access; query
  window count alone is not an adequate parent-protocol binding.
- The canonical official-preflight path is fixed by code; caller-supplied aliases or successor-independent
  copies are forbidden.  It freezes one unique output path for each of the four cells under a common seed-42
  root.  Queue dry-run may report a wholly absent, unlaunched terminal, but any present terminal must be a
  complete immutable pair and pass status, official-preflight, scientific-artifact, and live-closure checks.
- A positive cell is a bounded development result, not a formal result or universal identity claim.
