# CEBRA-inspired size-matched pseudo-session source training

**Date:** 2026-08-14  
**Status:** source-only CPU design gate in progress; this document does not by itself launch a GPU.  
**Machine contract:** `sua_exploration/configs/cebra_pseudosession_sua_v1.json`.

## Hypothesis

A2 showed that the carrier/zero-carrier margin increases under the observed C-to-M subject shift, while the
external Z4 activity-identity path becomes strongly negative. The new intervention tests whether the source
consumer overfits natural within-session unit co-occurrence. It presents behavior-matched units from several
source sessions in one size-matched population, without changing the encoder, decoder, loss, target calibration,
or target update rule.

For an anchor source window with endpoint velocity `v`, choose two distinct source sessions and their nearest
continuous-velocity windows. If all matches pass the source-only residual gate, construct

```text
U_tilde(v) = units(anchor, v) union units(donor_1, v) union units(donor_2, v),
```

with exactly the anchor's original unit count. A unit's neural query, M30 calibration activity, and T4/Z4 row
are indivisible. The final unit permutation is common to all three tensors. The target remains the anchor's
original behavior window and the anchor `SessionBatchSampler` schedule is unchanged.

The deployment path is untouched: M30 trial-level target directions fit T4, source normalizers are reused, and
the source-trained model is evaluated with no target backward pass or weight update. Dense velocity is consumed
only in source training, where the existing supervision-efficiency contract permits it.

## Frozen intervention

- source: the exact A2 27-session sub-C CO roster;
- validation/external: ordinary unmixed A2 sub-C development and sub-M external protocols;
- `mix_probability = 0.5`, `contributors = 3`;
- donor key: normalized continuous last-bin 2-D velocity, never a direction class;
- each contributor supplies one third of the anchor unit count, with integer quotas differing by at most one;
- accept only if both donor residuals are no more than `0.25` behavior SD per dimension RMSE;
- otherwise return the ordinary anchor example exactly;
- T4 and post-standardization Z4 are mandatory sibling arms;
- B3S, M30, W50, task-only, LR `1e-4`, 12 epochs, and mean epochs 5--12 remain unchanged.

The `0.25` residual gate was frozen before GPU execution. In a source-only 100,000-example audit, candidate
matching had a median residual `0.00140` and a rare maximum `45.12` caused by extreme behavior endpoints.
The gate rejected 357 of 50,225 candidate mixtures and retained 49,868. Accepted residuals had median
`0.00137`, P99 `0.0804`, and maximum `0.24985`. This is a validity fallback, not a tunable training arm.

## Attribution and gates

The logical design is `{ordinary, pseudo-session} x {T4, Z4}`. The ordinary cells are the sealed A2 source
checkpoints; only pseudo-session T4/Z4 cells are fresh.

Stage P runs seed 42 only. It expands to seeds 43 and 44 only when:

1. external sub-M absolute `mix-T4 - parent-T4 >= +0.03`; and
2. within-sub-C `mix-T4 - parent-T4 >= -0.03`.

The carrier interaction

```text
(mix-T4 - parent-T4) - (mix-Z4 - parent-Z4)
```

is always reported, but cannot rescue an unchanged T4 score by making Z4 worse. After three seeds, the route is
`carrier-specific` only if the external interaction is at least `+0.03` and positive for every seed, in addition
to the accuracy gates. If T4 and Z4 improve similarly, the result is a generic pseudo-session augmentation, not
a carrier mechanism. If the seed-42 accuracy gates fail, no mix-probability, contributor-count, residual-threshold,
or architecture tuning is allowed.

## CPU evidence required before GPU

- `p=0` item and sampler parity with the ordinary A2 DataModule;
- exact fixed anchor unit count for every mixed plan;
- one shared unit selection/permutation for neural, calibration, and side tensors;
- source-only donors and zero resolved formal-test paths;
- exact A2 T4 normalizer SHA `293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0`;
- T4/Z4 plan parity and exact-zero Z4 rows after the ordinary T4 normalizer;
- a real B3S production batch with finite forward loss and student backward gradients while the teacher stays frozen;
- a full 1,086,007-example immutable source schedule receipt and implementation bindings.

The first seven checks have passed locally. The full immutable preflight is the remaining CPU gate.

## Candidate convergence record

The CEBRA review considered: projection-head consistency, supervised contrastive loss, size-matched
pseudo-sessions, naive union pseudo-sessions, neural/T4 gauge augmentation, all-bin temporal supervision,
session-adversarial alignment, MMD/CORAL alignment, equal-session sampling, carrier corruption, a direct CEBRA
comparator, and RT/M2/H1 migration. Projection consistency was killed by an existing source-only screen: T4
headroom fraction `0.00619`, below its frozen `0.10` gate. Naive union is excluded because population-size change
is a confound. Decoder fusion and carrier corruption address different already-priced questions. The strongest
remaining candidates are:

1. size-matched continuous-behavior pseudo-sessions on SUA/sub-M;
2. all-bin source supervision, if pseudo-session mixing is flat;
3. gauge augmentation as a cheaper invariance control;
4. migration of a proven mechanism to M2 and RT;
5. a fair CEBRA comparator after an internal mechanism has earned the cost.

**Two-sentence pitch.** Functional carriers may fail under subject shift not because their four values are weak,
but because source training never breaks the natural co-occurrence between an activity population and one
session. Behavior-matched pseudo-sessions create that missing cross-session condition while retaining the exact
target label budget and BP-free deployment path.

**Strongest objection.** Any gain could be generic data augmentation rather than better carrier use. The Z4
sibling and the absolute external-T4 endpoint separate those readings: equal T4/Z4 gains are reported as generic,
while carrier specificity requires a positive interaction without a collapsing Z4 control.

## Dataset order

SUA/sub-M is first because it has 27 source sessions, 15 external-subject sessions, dense source behavior for
matching, and the strongest existing subject-shift interaction. A positive mechanism migrates next to M2 and
then RT. H1 remains open but lower priority: its compact-SPINT result demonstrates strong original NeuronID
capacity redundancy, and its sparse 7-DoF multi-phase events require phase-conditioned matching rather than the
2-D continuous key used here.
