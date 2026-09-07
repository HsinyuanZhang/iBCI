# HANDOFF TO ROOT: A2 interim review, open problems, and next-round options

> **Historical interim snapshot.**  The terminal A2 aggregate and current B1 execution status are recorded in
> the terminal addendum at the end of this file; they supersede the pending-seed and v8-GO statements retained
> below as audit history.

**Date:** 2026-08-13
**Basis:** independent review of the A2 v2 scoring receipts at seeds 42 and 43, plus a recheck of items left
open by `HANDOFF_TO_ROOT_POST_A4_REVIEW_20260812.md`.
**Authorizes:** nothing. Part A is problems to fix. Part B is options, not instructions.

---

## 0. Where A2 stands

Independently recomputed from the eight scoring receipts, not copied from any aggregate:

| seed | domain | T4 | Z4 | carrier gain |
|---|---|---:|---:|---:|
| 42 | within sub-C, 6 sessions | `0.585301` | `0.338131` | `+0.247170` |
| 42 | external sub-M, 15 sessions | `0.303069` | `-0.174186` | `+0.477255` |
| 43 | within sub-C | `0.576001` | `0.317025` | `+0.258975` |
| 43 | external sub-M | `0.357220` | `-0.089386` | `+0.446606` |

Interaction `+0.230085` and `+0.187631`; two-seed mean `+0.208858` against a `+0.03` gate.

The execution is clean and I verified it rather than trusting the receipts' own summary:
`same_source_cell = true` for all four arm-by-seed combinations, so both domains score the **same** checkpoint;
`backward_gradients` and `decoder_weight_updates` false in every cell; both arms share side normalizer
`293b8a55...`; `target_domain_normalizer_refit_performed: false`; the six sealed sub-C sessions appear by name
only with `formal_test_sessions_resolved_or_opened: false`. The within-subject scores also reproduce the SUA
lattice (`T4 0.574976`, `Z4 0.326008`), which is a good internal-validity signal.

**Nothing here is quotable yet.** Seed 44 is still training and the bootstrap gate is not yet evaluable.

---

## Part A — problems to fix

### A-1. Verify the A2 bootstrap gate can actually fire, before you evaluate it

Three gates in this program have already turned out to be mathematically incapable of doing their job: the
D-optimal `det(X'X)` gate could only pass, the A4 falsification rule could never fire, and the original A2
Wilcoxon gate could never reach `p <= 0.05` at `n = 3`. The crossed seed-by-session bootstrap that replaced the
Wilcoxon gate has **not** been checked for the same defect.

Check it before seed 44 lands, not after.

**Make gate attainability a standing requirement.** Every aggregator test suite currently proves the gate
computes correctly. None proves the gate can both fire and not fire. Add two synthetic inputs per gate — one
that must pass, one that must fail — and assert they give different verdicts. This is cheap and it would have
caught all three earlier defects.

### A-2. The two domains have non-overlapping, unequal session sets

Within-subject is 6 sessions, external is 15, and no session appears in both. The bootstrap therefore **cannot
pair sessions across domains**. It must resample sessions independently within each domain and then form the
interaction. This is the one place the implementation can silently be wrong and still produce a plausible
interval.

### A-3. The absolute Z4 number conflates two failures

Policy is `behavior_authority: strict_subc_source_train_27_only` with target-domain refit forbidden. So the
behavior normalizer fitted on sub-C is applied to sub-M. Part of Z4's `-0.174` is normalizer mismatch, not
identity failure.

**The interaction is unaffected** — the T4 arm carries the identical mismatch and still scores `+0.30` and
`+0.36`. But do not quote `-0.174` on its own as "the activity-only identity fails on a new subject" without
that qualifier. Write the interaction as the claim and the absolute as context.

### A-4. Report the Z4 external seed variance

`-0.174186` versus `-0.089386` is a spread of `0.085` on a 15-session mean, which is larger than the spread
between the two seed-level interactions (`0.042`). The external Z4 arm is unstable across seeds. Report it
rather than letting a reader discover it from the appendix.

### A-5. Still open from the previous handoff

- **Estimator copy is still unguarded.** `d_optimal_calibration_design.py` still defines its own
  `CANONICAL_DIRECTIONS_RAD` and `_fit_cosine_tuning`, and no test cross-checks them against
  `unit_side_features`. This must close before B9 produces a number, because it yields plausible wrong numbers
  rather than an error.
- **Root-level test collection is worse than reported.** A full run now gives `1961 tests collected, 67 errors`.
  Each subtree passes from its own directory. A third-party reviewer will start with a root-level `pytest`.

### A-6. Contract filename no longer matches its content

`A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md` now contains a target-subject-shift estimand. A reviewer
navigating by filename will be misled. Rename it, or put the corrected estimand in the first line of the file.

---

## Part B — options for the next round

These are ideas, not instructions. The common theme: the six A2 source checkpoints are a **reusable asset**, and
several previously expensive experiments are now nearly free because scoring them is forward-only.

### B-1. Subject J is on disk

`sua_exploration/data/dandi_000688/` contains `sub-C`, `sub-M`, **and `sub-J`**. The A2 contract already
mentions scoring on J "if admissible."

This is the highest-value option available. It converts the biggest caveat — one external subject, so no
general cross-subject claim — into a two-point generalization, and it costs one forward-only scoring pass per
existing checkpoint. No new training. Check admissibility first: unit counts, task match, and whether J was ever
touched during any earlier selection.

### B-2. Turn the normalizer confound into a result

Refit only the **behavior** normalizer on target-domain statistics, which needs no labels, and rescore. Two
outcomes and both are useful: if Z4 recovers substantially, you have decomposed identity failure from
normalizer mismatch and the paper reports both cleanly; if Z4 stays negative, the identity failure is
established as real and A-3's qualifier disappears.

### B-3. Build a graded shift axis instead of a two-point contrast

`P3_CROSS_SESSION_ANALYSIS.md` documents a unit-count regime jump within sub-C around 2016-09: train sessions
average 59 units, later sessions average 245. That is a within-subject shift axis that already exists in the
data. Scoring the same checkpoints across early sub-C, late sub-C, then sub-M would turn the current binary
contrast into a **dose-response curve**, with subject M at the far end.

A curve is a much stronger figure than a two-point difference, it is forward-only, and it partially answers
"which component of subject shift does the carrier repair" without needing A3's synthetic breakage ladder.

### B-4. A3 becomes cheap now, and it isolates what A2 bundles

A2 tests target-subject shift, which bundles animal, array, tuning distribution, and unit correspondence. A3's
ladder — unit dropout, electrode pooling, unit subsetting — isolates the correspondence component specifically.
Running it on the A2 checkpoints is forward-only. If B-3 and A3 agree on where the carrier's advantage appears,
the mechanism claim gets much stronger.

### B-5. B1 factorial is staged and is the right next GPU job

`m2_carrier_distillation_interaction_v2` is at `CPU_PREFLIGHT_READY_GPU_NOT_AUTHORIZED` with a matched
`t4 x z4` by loss-mode factorial. Note this design is better than the three-cell sweep originally specified:
the hypothesis is that `lambda_E` pulls the student back toward a phase-poor solution **conditional on a
carrier being present**, so the interaction is the estimand and the main effect alone would not test it. Queue
it behind A2's seed 44.

### B-6. Free reuse of the A2 checkpoints

Three previously-listed items are now near-zero marginal cost because the checkpoints exist and scoring is
forward-only: the A5 transferred and aged carrier control, the identity-token occupancy probe from section 3.5
of `HANDOFF_NEXT_CONTRIBUTION_STRATEGY_20260812.md`, and the A12 descriptive attention audit once its real-data
path is repaired.

---

## Part C — priority

| # | Item | Cost | Why now |
|---|---|---|---|
| 1 | A-1 gate attainability, A-2 bootstrap domain handling | small | Must be right before seed 44 lands |
| 2 | A-3 and A-4 wording | writing | Prevents an overclaim entering the draft |
| 3 | B-1 subject J admissibility check | small | Answers the largest caveat, forward-only |
| 4 | A-5 estimator copy gate | small | Blocks any B9 number |
| 5 | B-2 normalizer decomposition, B-3 graded shift axis | forward-only | Converts a confound and a two-point contrast into results |
| 6 | B-5 B1 factorial | GPU | After A2 releases the GPU |
| 7 | A-5 root-level test collection, A-6 filename | small | Third-party review workflow |

Items 1 and 2 are the only ones that are time-critical. Everything in Part B can wait for A2 to close.

---

## Root review — 2026-08-13

**Disposition:** accept the interim A2 arithmetic and the reporting cautions, but revise the proposed
next-round priority.  This review does not change any A2-bound source, contract, receipt, or running cell.

### Closed before the terminal A2 aggregate

1. **A-1 gate attainability is closed by an independent no-write replay.**  Synthetic positive and null
   interactions respectively pass and fail the frozen primary gate.  The secondary six-session and
   fifteen-session sign/Wilcoxon checks likewise have attainable pass and fail cases.  The frozen A2
   implementation must not be edited after its official preflight merely to add redundant tests.
2. **A-2 domain resampling is correct.**  Seed indices are shared across the two domains, while session
   indices are resampled independently within the disjoint six-session and fifteen-session rosters.  An
   independent replay reproduced the implementation.  There is no cross-domain session pairing.
3. **A-3/A-4 are reporting requirements, not new experiments.**  The primary quantity is the
   external-minus-within interaction in `T4-Z4`.  Absolute external Z4 is secondary and must be accompanied
   by its seed spread and the source-C behavior-normalizer qualifier.

### Corrections to Part B

- **B-1 / subject J: hold as a descriptive stress test, not a second clean external subject.**  J has only
  three CO recordings with 38, 18, and 19 units.  It was already opened for loader/model forward-backward
  smoke testing, and the existing documentation explicitly reserves it for schema/pipeline smoke rather
  than model ranking.  A later score-only contract may report it as a low-unit-count, three-session stress
  test, but it cannot upgrade A2 to robust two-animal generalization and must not modify the frozen A2 scorer.
- **B-2 / target behavior-normalizer refit: diagnostic only and lower priority.**  Behavior mean/std are
  computed from target kinematic outputs.  Therefore this is not label-free deployment adaptation, even
  though it requires no gradients.  It changes A2's strict source-only behavior-normalizer contract and may
  be used only as a separately labelled decomposition diagnostic after A2 closes.
- **B-3 / graded sub-C shift: not free under the present A2 split.**  All 27 source sessions are 2013--2015
  training sessions, and the six clean A2 development sessions are also in the old `<100 units` regime.
  The 2016-09 high-unit regime cited by `P3_CROSS_SESSION_ANALYSIS.md` is outside the frozen A2 evaluation
  roster.  Scoring early source sessions would be held-in; adding the late P3 sessions would change the
  data contract.  A defensible dose-response therefore requires a new temporal split or source-only LODO
  retraining and is not a near-zero-cost A2 checkpoint reuse.
- **B-4 / A3 ladder: rewrite as population-degradation robustness.**  Consistent unit subsetting, dropout,
  or pooling changes available signal and population size but preserves activity-carrier attachment; a
  consistent permutation is an exact null for the permutation-invariant model.  These views do not isolate
  biological unit correspondence.  Existing row/label-shuffle controls already test wrong attachment more
  directly.  Do not launch A3 until a non-null deployment perturbation and a narrower robustness claim are
  frozen.
- **B-5 / B1: accept as the next GPU experiment after A2.**  The estimand must remain the full matched
  `{T4,Z4} x {with-E,no-E}` interaction.  Terra's final settlement repaired the pre-launch freshness
  and direct-scorer output-path gaps; independent Luna review passed all 32 focused tests and minted
  immutable v8 preflight SHA `b1c9c5e...`.  B1 is therefore GO immediately after A2 closes.  Stage P
  remains routing-only and Stage F, if triggered, is the independent confirmatory set.
- **B-6 / reuse: split the list.**  A5 remains non-identifiable on sorted SUA without an independently
  validated cross-date unit map and must not be revived.  A12 and identity-token occupancy are permissible
  CPU-only descriptive audits after the current exact A11 CPU job releases resources; neither is a causal
  carrier-effect experiment.

### Deferred engineering items

- The independent estimator copy remains a blocker only for B9.  B9 is not on the active queue, so this does
  not pre-empt A2 or B1.
- Root-level `pytest` collection spans incompatible subprojects/plugins.  Per-tree authoritative commands are
  already documented in the repository README.  A unified root harness is useful engineering work, not a
  scientific blocker for the current receipts.
- Do not rename `A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md` while it is part of the immutable A2
  implementation binding.  Its opening text already states the corrected target-subject-shift estimand.
  After A2 closure, add a stable target-subject-shift alias or a versioned successor while preserving the
  original path for receipt provenance.

### Revised execution order

1. Finish seed 44 and publish the frozen A2 aggregate; until then the seed-42/43 values remain interim.
2. If A2 passes, update the paper with the interaction, all seed/session summaries, Z4 variance, and the
   normalizer qualifier.
3. Release the two 3090s to B1 Stage P.  Run Stage F only if the frozen Stage-P routing gate passes.
4. Treat J scoring, target-normalizer decomposition, and A3 population-degradation views as optional
   diagnostics after the mainline, not as blockers.
5. Keep B2 staged after B1 because its corruption training must use the B1-selected loss mode.

---

## Terminal addendum — 2026-08-13

The interim restriction above has now expired because seed 44 and the frozen aggregate are complete.  The
authoritative immutable aggregate is
`sua_exploration/results/a2_matched_subject_shift_v2/terminal_aggregate.json`, SHA-256
`5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc`.

| quantity | seed 42 | seed 43 | seed 44 | three-seed mean |
|---|---:|---:|---:|---:|
| within sub-C `T4-Z4` | `+0.247170` | `+0.258975` | `+0.240757` | `+0.248968` |
| external sub-M `T4-Z4` | `+0.477255` | `+0.446606` | `+0.530438` | `+0.484766` |
| subject-shift interaction | `+0.230085` | `+0.187631` | `+0.289681` | `+0.235799` |

The crossed seed-by-session bootstrap interval is `[+0.100852,+0.371768]`; the frozen verdict is
`subject_shift_interaction_effective`, and all primary and secondary gates pass.  Three-seed absolute means are
within `Z4=0.326008`, `T4=0.574976`; external `Z4=-0.143399`, `T4=0.341367`.  The external Z4 seed means
`-0.174186/-0.089386/-0.166624` must remain visible, together with the strict source-C behavior-normalizer
qualifier.  The interaction is increased relative carrier value under the observed shift, not an absolute
T4 lift of `+0.235799` and not a unit-correspondence result.  All six formal sub-C test sessions remained sealed.

The prior execution-order statement that B1 was immediately GO under v8 is also superseded.  Its first T4/Z4
attempt pair produced no scientific cell: Z4 failed in DataModule setup because its wrapper retained a string
`data_dir`, and T4 was externally interrupted without completion/binding/score receipts.  Preserve both invalid
traces.  B1 resumes only after the exact real Z4 setup path passes, an independent audit approves the repair, a
successor preflight is minted, and fresh durable attempt paths are used.  The scientific factorial and its
Stage-P/Stage-F gates are unchanged.
