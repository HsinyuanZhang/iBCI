# Posterior Carrier Matched Score V1

Status: design and no-data scaffold only. This document does not authorize a
checkpoint load, NWB access, CUDA initialization, score publication, or any
remote command.

## Purpose

Evaluate the completed `POSTERIOR_CARRIER_BUDGETMIX_D_SEED42` full-training
SWA against the sealed Cell-D checkpoint, re-evaluated with closed-form OLS
point carriers at M30/M10/M4. All results use the governing last-bin,
variance-weighted, equal-session metric. The scorer must wait for the
immutable full terminal and final-four SWA, then use one physical,
forward-only input pass per session.

## Attribution boundary

The trained posterior arm is one system: Cell-D initialization plus the
parameter-free posterior carrier wrapper, its credibility attention bias, and
the reviewed 48-epoch budget-mix training recipe. It is compared with a
different sealed system: the original Cell-D checkpoint, re-evaluated with the
ordinary OLS point carrier at the same M30/M10/M4 prefix budgets. This is an
honest system comparison, not a one-factor attribution of the training arm.

The posterior-minus-point headline is therefore **not** produced by feeding a
point carrier into the posterior-trained SWA. Such a forward-only switch, if
ever added, is a separately labelled non-governing inference diagnostic. It
cannot be substituted for the sealed-point-system comparator or used for any
gate below.

## Frozen score matrix

| Role | Carrier mode | Prefix budget | Surfaces | Role in interpretation |
| --- | --- | --- | --- | --- |
| Sealed point system | sealed Cell-D checkpoint + OLS point carrier | M30, M10, M4 | strict within-6, fixed-ledger external sub-M-15 | mandatory native M30 parity anchor; matched point-system baseline at all three budgets |
| Posterior system | posterior-trained SWA + posterior mean + credibility | M30, M10, M4 | same two surfaces | matched distributional-identity system under test |
| Diagnostic only | posterior-system aligned / zero / complete cyclic wrong-pair | M30 only | same two surfaces | attachment/identity diagnostic; explicitly M30-only and never a gate substitute |
| Optional replication | all above only after a distinct H1 authority exists | same | H1 only | separately labelled; never substitutes for strict external-15 |

For every valid 50-bin window, score only the last queried bin using
`tfpd_lane.matched_scorer.session_r2` with variance-weighted two-coordinate
R2. First calculate one R2 per session, then equal-weight sessions. All
primary effects are paired by exact session name. No flattened-window metric,
sample-weighted aggregate, target optimizer step, backward pass, normalizer
refit, or target fitting is allowed.

For both systems and each M30/M10/M4 row, the receipt reports per-session R2,
equal-session mean, and equal-session median. For every budget, it reports
paired `posterior_system - sealed_point_system` deltas, positive-session count,
and a deterministic, domain-separated paired bootstrap 95% interval
(`10,000` draws, seed `42`). It also reports M30-to-M10 and M30-to-M4
degradation for both systems.

The external decision is frozen before observing scores:

```text
STOP if posterior-minus-sealed-point M30 mean < -0.02.
PASS only if M30 safety passes, M4 mean >= +0.03, M4 positives >= 9/15,
posterior M30-to-M4 degradation < sealed-point M30-to-M4 degradation, and
posterior aligned M30 mean > both posterior zero M30 and posterior cyclic
wrong-pair M30 means.
Otherwise HOLD.
```

The zero/wrong controls are M30-only, so anti-triviality is explicitly M30-only.
They cannot rescue a failed M30 safety, M4 headline, breadth, or curve gate.
No receipt may select a favorable budget, diagnostic, or H1 surface after
observing values.

## Execution gates

Before any input pathname or model tensor is opened, the future physical route
must verify all of the following through immutable descriptor-safe evidence:

- the posterior full terminal, source authority, checkpoints 44--47, and SWA;
- the sealed Cell-D terminal/SWA and full per-session governing last-bin table;
- exact strict within-6 and fixed-ledger external-15 authorities;
- the frozen source posterior normalizer and M4/M10/M30 prefix rules;
- matched input order, B3S M30 calibration, model state stability, eval mode,
  no dropout, no gradients, no updates, and fixed-device authority.

The public CLI is dry/fail-closed. An in-process root-reviewed capability and
two explicit execution flags are required later. Any preflight, lineage,
parity, asset, closure, mode, or publication failure must produce an honest
failure terminal and no partial scientific score.

## Physical implementation boundary (reviewed, but not authorized to run)

The physical backend is deliberately a later capability, not a second scorer
design.  The default public route remains dry and neither imports Torch nor
opens a result, input, checkpoint, cache, or device.  A root-reviewed
in-process capability is required before a durable score attempt can be
reserved; visible command-line flags alone do not grant this capability.

### Immutable predecessor and imported mirror

The completed remote full-training result must first be copied byte-for-byte
into the fresh local canonical mirror
`tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_full_train_import_mirror_v1`.
The imported-mirror provenance binds the remote root label, its held-directory
identity, the remote terminal SHA, the complete remote-to-local SHA map, and
the remote full closure.  It explicitly states that the remote stage identity
is not rebuilt locally.  The future scorer may read only this mirror, through
one held `O_NOFOLLOW` directory descriptor; it may never infer a remote
identity from the local score stage.

The mirror topology is exact: attempt, launch, source authority, throughput100,
all 48 epoch receipts, checkpoints 44--47, final-four SWA, terminal, and one
0444 SHA sidecar for every body.  A failure receipt, extra leaf, sidecar/mode
drift, missing epoch, terminal graph mismatch, or any body-map mismatch is a
pre-input NO-GO.  Before model placement, the physical backend strict-loads
every final-four posterior checkpoint on CPU, recomputes each state digest,
rebuilds the FP64 arithmetic SWA, strict-loads it into a fresh posterior
wrapper, and verifies the stored SWA state digest.  The sealed Cell-D terminal,
SWA, and governing last-bin table are separately descriptor-loaded and bound.

### Matched physical inputs and models

The durable target-free preflight derives—not accepts from its caller—fixed
strict-within-6 and fixed-ledger external-15 asset rows before any input
pathname is resolved.  Within rows come only from one descriptor read of
`c1_train_val_33_manifest.json` (the exact mode-0600 C1 val manifest), whose
six literal filenames, byte counts, and hashes are checked.  External rows
come only from the fixed v2 `eligible_session_ids` UUID list joined to exactly
one eligible disposition row, one verified-download row, and one scope row.
The durable receipt records those descriptor identities; every publication and
every physical authorization re-derives the rows and exact-compares them
before an evaluator asset factory exists.  A held root and held asset
descriptor then verify byte count, SHA, named inode, and a private parser
snapshot for every session.  One parsed session object is shared by every
score cell.  Its receipt row binds neural bytes, held B3S M30 calibration
bytes, last-bin target and valid-mask bytes/count, point and posterior prefix
inputs, the common selected prefix-row IDs at M30/M10/M4, same-prefix theta
recovery evidence/fallback count, and both normalized carrier byte digests.

Point rows use the closure-bound ordinary source-only OLS normalizer at
M30/M10/M4.  Posterior rows reconstruct only the unique frozen source prior
and posterior normalizer from the completed full source authority; the
preflight derives its exact payload rather than accepting a self-hashed copy.
It retains that `PosteriorSourceT4Normalizer` as a CPU/float64 immutable
authority.  At physical evaluation it creates a separate device-local,
float64 `FrozenSourceT4Normalizer` view with the identical authority SHA; this
is required because `posterior_mean_view` requires carrier and normalizer to
share device/dtype.  The operation is a copy of the four already validated
moments, never a target fit or a normalizer mutation.  B3S always uses the
same M30 calibration activity.  Raw T4 construction proves exact SUA unit-axis
order (`channel_ids == arange(n_units)`), feature group, pool size, source unit
count, raw bytes, and closure-bound function semantics before a carrier reaches
either model.

For avoidance of a false causal claim, the sealed point rows reuse the
ordinary OLS normalizer from the sealed Cell-D training/replay authority at
all three forward-only budgets.  They are not a point-budget-mix-trained
control.  The posterior arm's M4/M10/M30 distributional normalizer is a
separate frozen full-training source authority.  The receipt names both
normalizer families and this system-comparison distinction explicitly.

The posterior theta adapter may use a same-prefix `target_corners` fallback
only when the already selected rewarded prefix row has missing `target_dir`;
it cannot select a later label.  The scorer does not apply the strict-source
aggregate fallback-topology gate to target sessions.  Instead it binds each
fallback's prefix position and original trial index, and binds the full M30
prefix plus its M10/M4 leading slices for both the posterior and OLS point
systems.  The OLS side's closure-bound T4 function is required to use the
same `list_datamodule_rewarded_trials(... )[:M]` rule.  Any nonzero target
fallback is disclosed in the input authority as a target-local system-detail;
it cannot remain hidden as a short-prefix distributional effect.

The physical route strictly loads two distinct systems: the sealed Cell-D SWA
for every OLS point cell and the posterior full SWA for every posterior and
diagnostic cell.  Both run in `eval`, under `no_grad`, with dropout inactive,
no gradients, and state digests identical before/after every cell.  A repeated
fixed batch must be bitwise equal.  The sealed point M30 replay must reproduce
the entire sealed per-session governing last-bin table before any successor
row is licensed.

### Access order, device, and publication

The immutable score attempt precedes all model and within/external input
access.  The fixed execution contract is GPU0 only (`CUDA_VISIBLE_DEVICES=0`
to logical `cuda:0`), with independently exact nvidia-smi nominal memory,
Torch memory bytes, UUID, BDF, GPU name, Torch/CUDA/cuDNN versions, and
TorchMetrics 1.5.1.  H1 and formal inputs are outside this route.  Target
optimizer, backward, update, and normalizer-refit counts remain zero.

After all forwards, the backend re-verifies held inputs, model state, runtime
and implementation closure, then descriptor-reloads the durable preflight and
root authorization.  Only then may `score.json` and `terminal.json` publish as
one `O_EXCL`/fsync/0444 transactional group.  Any failure after the attempt
leaves an honest immutable failure terminal and rolls back any owned partial
scientific score pair.

### Review tests required before an execution GO

The no-data suite covers matrix order and gates, descriptor/sidecar/mode and
topology drift, remote/local mirror substitution, source-normalizer and
carrier-budget swaps, old/full-window baseline substitution, input-order and
last-bin target/mask/count swaps, model/SWA/checkpoint state drift, early input
access, no-dropout/no-gradient/state stability, final authorization recheck,
and transactional publication rollback.  A synthetic CPU Cell-D/posterior
wrapper forward is an interface test only; it is not a checkpoint load or a
scientific score.  No execution is authorized until a separate audit accepts
the frozen bytes and the completed full terminal/mirror exists.
