# Subject-M B0 external score-only bridge

Status: **terminal CPU score-only bridge**.  All three fixed seed cells and the
aggregate are complete; no rerun, new seed, epoch selection, or target access is
authorized by this document.

The bridge scores no new model and trains nothing.  Its only prospective
runtime is CPU forward evaluation of three archived original-SPINT B0 source
runs on the fixed 15-session subject-M cohort.  It is deliberately separate
from A2's sealed B3S T4/Z4 matrix.

## Frozen inputs and score rule

- Seeds: `42, 43, 44`.
- B0 source directories: `sua_spint_t4_mainline_fp32_v1_b0_dandi688_co_s{seed}`.
- Logical epochs: every epoch `5,6,7,8,9,10,11,12`, mapped exactly to the
  zero-based Lightning files `epoch_004.ckpt` through `epoch_011.ckpt`.
  Per-session R² is the unweighted arithmetic mean over all eight values; no
  target metric can select a checkpoint.
- Strict source lineage: 27 sub-C train / 6 sub-C validation / 6 names-only
  formal sessions, source manifest SHA
  `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`,
  and teacher SHA
  `9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d`.
- Query semantics: chronological first 30 rewarded trials for B0 activity
  calibration; all scored 50-bin windows lie in rewarded trials strictly after
  trial 30.  The bridge compares each session's count/order projection against
  the sealed A2 M30 subject-M receipts.
- Behavior normalization: the exact strict-27 source behavior normalizer,
  semantic SHA `f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391`.
  Target normalizer fitting, optimizer/backward activity, decoder updates, and
  GPU execution are forbidden.
- Subject-M bytes: before resolving any runtime target path, the immutable
  preflight binds all three A2-vetted metadata authorities:
  - A2 official preflight SHA
    `8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd`;
  - sidecarless `0444` v2 score-blind ledger SHA
    `1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283`;
  - sidecarless `0444` frozen v2 scope-manifest SHA
    `68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55`.

  The authority joins A2's ordered 15-session eligibility audit to the frozen
  `session → asset_id → canonical sub-M path → size/SHA-256` map and the
  ledger's independently verified download row.  A prospective scorer first
  stream-hashes each current local NWB through one `O_NOFOLLOW` FD, checks its
  exact byte count/SHA and pathname identity, then passes the pathname to the
  NWB loader.  It repeats the same-FD check immediately after loading and
  records both identities and hashes in the score receipt.  The byte checks
  read raw file bytes only; they do not parse an NWB or calculate a metric.

The archived B0 checkpoints and teacher are mode `0664`, and the normalizer is
mode `0600`, rather than legacy immutable `0444` inputs.  No target score is
permitted merely because a preflight passes: a root-minted immutable
authorization pair must bind the complete source audit digest.  For every
prospective score cell, the bridge reads each source input once from an
`O_NOFOLLOW` file descriptor, validates same-FD identity and SHA, and then:

- sends normalizer bytes directly to `np.load(BytesIO(...))`; and
- writes the verified B0 and teacher bytes to fresh, private `0400` snapshots
  before the historical pathname-only loader can reopen either file.  The
  snapshots are re-verified after loader setup and removed with their private
  directory.

Thus the score path cannot hash one mutable archive pathname and consume a
later replacement.  The immutable preflight also pins every active
bridge/scorer/aggregator/read-only-preflight/root-publisher/A2
adapter/model-loader source byte.  The aggregate re-verifies that source map
against every score receipt, so a later CLI or core change cannot be silently
mixed with an older preflight.  All receipt body/sidecar
writes use parent-symlink rejection, `O_EXCL`, file and directory `fsync`, and
rollback that deletes only the transaction's own inode.

## What B0 is—and is not

CPU source audits require the **original-SPINT identity architecture under the
matched streaming source protocol**: a `BatchReferenceEncoder` with
trainable copied `fc_id_in` and `fc_id_out` stacks (12 identity tensors) and
`side_dim=0`.  B0 consumes no target-direction carrier.  A2 T4 and Z4 are
B3S systems and A2 constructs a target-session carrier for both arms before
the Z4 mask.

This is not a claim about a published-SPINT system or checkpoint.

Therefore the two new outputs are predeclared **system contrasts**:

```text
external T4 − B0
external Z4 − B0
```

They jointly change architecture/topology, source checkpoint lineage, and
target carrier use.  They cannot identify a carrier causal effect and cannot
be used to rank or post-hoc select T4, Z4, an epoch, or a seed.

The immutable A11 B0 CPU-forward receipt is additionally bound as the
within-subject B0 reference:

- `a11_b0_convergence_full_access_v1_full_cpu_forward_2843108a665b53b4.json`
- body SHA `955ebaf8b1ac229317118bb7c3bf3ebdc8c6c62c5f5be2ea1440d32d78724614`
- six development sub-C sessions, fixed logical epochs 5--12, M30,
  query-after-30, and the same behavior-normalizer semantic SHA.

It is accepted only if its per-seed B0 checkpoint SHA map agrees with the
archived B0 sources and its exact query/metric/normalizer/roster receipts pass
validation.  The aggregate then reports, without any selection:

```text
B0 external_subject_M mean and per-seed means
B0 external_subject_M − within_subject shift
[(T4 − B0)external − (T4 − B0)within]
[(Z4 − B0)external − (Z4 − B0)within]
```

Those two latter quantities are explicitly **system-shift interactions**,
not carrier-causal interactions, and are paired only by the predeclared seed;
the external 15-session and within six-session rosters are distinct.  A11
mismatch fails closed before such a shift is emitted.

The sealed A2 carrier-content interaction is also copied, unchanged, from the
immutable A2 terminal aggregate:

```text
[T4 − Z4]external_subject_M − [T4 − Z4]within_subject
```

It explicitly does not include B0.  The A2 terminal body SHA is
`5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc`.

## Terminal result and execution record

All predeclared execution gates completed.  The official preflight and root
execution authorization are immutable under
`results/subm_b0_external_score_bridge_authority_v1/`.  The three immutable
score pairs are:

- seed 42 body SHA `60bf08e91b77af13872da40b687d1daf5b483f60f6c66dc98b650439fe26e1a2`;
- seed 43 body SHA `c96d25eabbf94aa50e99023028da0a7756a17016e7bd1bd308b213a36b47cce5`;
- seed 44 body SHA `1c85013d674edf5a8bdbbf8ea4efd06f9ced8e18146a723f9a38b329843320a3`.

Their external subject-M means are respectively
`−0.1188071719/−0.0963518284/−0.1333917134`; all three are negative and
their fixed-seed mean is `−0.1161835713`.  Against the immutable A11 within
B0 mean `+0.2364165927`, the external-minus-within shift is `−0.3526001640`
and is negative for all three seeds.  The external system contrasts are
`T4−B0=+0.4575510666` and `Z4−B0=−0.0272153523`.

The terminal aggregate is
`results/subm_b0_external_score_bridge_v1/terminal_aggregate.json`, body SHA
`913e58ec8556aff61fb25fe8571a94102136652f467cb190f5a0c14c3cec6211`.
Its body and sidecar are regular non-symlink mode-`0444` files; the aggregator
opened no target data and the formal sub-C test sessions remained sealed.

The result supports a failure-to-transfer interpretation for this
matched-streaming original-SPINT identity topology: it learned useful
within-domain structure but became negative under the observed C-to-M shift.
Its proximity to external Z4 supports Z4 as a reasonable proxy for that
deployment failure, while T4 remains strongly positive.  The scientific
limits in the preceding section remain controlling: these are system
contrasts, not a pure carrier intervention and not a claim about a published
SPINT checkpoint.  Because the canonical outputs already exist under O_EXCL
semantics, the historical launch commands must not be rerun or repurposed.
