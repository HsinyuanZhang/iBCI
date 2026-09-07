# DANDI 000688 sub-M CO mechanism endpoint v2

Status: `candidate_frozen_metadata_only_external_runner_blocked`

This is an append-only candidate freeze for a future, separately authorized external-subject mechanism test. It does not report a score, authorize an NWB download, open the formal endpoint, or authorize GPU work. The intended claim is limited to **same-Dandiset cross-animal confirmation** on sub-M center-out sessions. It is not an independent-dataset or independent-laboratory confirmation. The pseudo-MUA view is deterministic within-session electrode pooling of the same sorted-SUA source, not native threshold-crossing MUA.

## Relationship to v1

V2 supersedes v1 only as the candidate/mechanism specification. It does not modify or replace any v1 file. The generator fail-closes unless the following v1 files retain their exact SHA-256 values and the live metadata replay is byte-identical to the v1 manifest:

| v1 artifact | SHA-256 |
|---|---|
| `scripts/freeze_dandi688_subm_co_scope.py` | `23142a547e6c49cbf6f0e0d723e7c7500b4c8fbc9309424702c21f5522f5a971` |
| `docs/DANDI_000688_SUBM_CO_SCOPE_FREEZE_PREFLIGHT.md` | `28980aebae5b10c7116d47b62a341548dda40636c49b6d8b2519cb9c1b1b8e83` |
| `manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v1.json` | `65a38eee5f5b13029120978884c8d5dc8c00f8001738a322a114be75a5e1b50c` |

The v1 API inventory and its complete 22-asset CO ledger are replayed and copied into v2 without changing scope or eligibility. The ledger contains all 22 public CO assets; a later compatibility preflight may assign a score-blind disposition to each asset but may not silently remove a row.

## Frozen matched candidates

The comparison is `shared_t4` versus `shared_ts4`, using seeds 42, 43, and 44. There is exactly one terminal checkpoint per arm and seed: the completed 12-epoch run's `epoch_011.ckpt`. No target session, label, prediction, or score may select an epoch.

| arm | seed | terminal checkpoint SHA-256 |
|---|---:|---|
| shared_t4 | 42 | `ab9df840a07d7aeb6cc417bb684f1f5e0265d50f98168400ac915647cdfd7b9f` |
| shared_t4 | 43 | `05c05b3ab82a2fba43c55aca523248982a954faf5f0363a0235a29d64e57ab22` |
| shared_t4 | 44 | `a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6` |
| shared_ts4 | 42 | `a21da5a72a991bd2665af50572a4132998ac79d7f801879048553efcdc8281b2` |
| shared_ts4 | 43 | `c8dd22dfadb2bc11555fc21abe464316886d221e2dbcd71bf20a6bffe9cb158e` |
| shared_ts4 | 44 | `a2d877ac81a4e553e5221c54e465db26eba8592888b8cb5339e9dfc4acd66ced` |

For each seed, the generator compares the complete scalar metadata trees. After normalizing timestamps and arm-specific output paths, T4 and TS4 metadata must be identical. The only scientific difference allowed is descriptor-row attachment:

- T4 retains each normalized four-value `[a,c,m,b]` row on its source unit/channel.
- TS4 starts from the identical raw T4 fit, applies the identical view-specific source-train-27 normalizer, then evaluates `features[RandomState(seed).permutation(N), :]`.
- TS4 moves all four descriptor values as a complete row. It does not change row values, the row multiset, labels, electrode IDs, feature width, architecture, teacher, source data, loss, or parameterization.
- The historical control does not force a nonidentity permutation. Every realized permutation must be reported. A realized identity is retained and may not become an exclusion rule.

The view-specific source-only normalizer semantic hashes are frozen as:

- SUA: `ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7`
- pseudo-MUA: `92470ad14062af6cb998e06e7696b94bfdfd20ac5e415615302a5dddc7098fcc`

The manifest also binds the closure metadata, source/train and data manifests, teacher, prelaunch receipt, and control-code hashes. A mismatch in any matched item is a fail-closed error, not a reason to repair or substitute an arm.

The seed-44 TS4 terminal checkpoint was restored into a previously absent canonical local path by a hash-qualified copy from the known original remote C1 stage. Its expected SHA-256 and size were verified. This is artifact recovery for a pre-existing completed run; it is neither retraining nor checkpoint selection, and no existing target was overwritten.

## Frozen endpoint

All eligibility decisions must be made from compatibility/schema information before any checkpoint load, prediction, or score. Both arms, both views, and all seeds use one common eligible-session cohort. Let its size be `N`. If `N < 6`, the runner must return `ENDPOINT_NO_GO_INSUFFICIENT_ELIGIBLE_SESSIONS` before scoring. All 22 assets remain visible in the disposition ledger even when some are score-blindly ineligible.

For each eligible session, view, arm, and seed, compute exactly one R² from all valid windows strictly after chronological rewarded trial 50:

- use the last behavior bin of each 50-bin query window;
- divide decoder output exactly once by `BEHAVIOR_SCALING_FACTOR=5.0`;
- update `torchmetrics.regression.R2Score(multioutput="variance_weighted")` over every query-window prediction and target in that session, then compute once;
- treat any undefined or nonfinite arm score as endpoint failure; do not drop the session.

Pair within the identical session, view, and seed:

`delta = R2(shared_t4) - R2(shared_ts4)`.

Use equal session weights within each seed, equal weights over seeds 42/43/44 for the grand paired mean, and equal seed weights for each session's cross-seed mean. The primary endpoint is the SUA paired delta. The key secondary endpoint is the pseudo-MUA paired delta. Absolute shared-T4 R² is required in each view. Pseudo-MUA cannot rescue a failed SUA primary; the overall mechanism claim requires both views to pass all gates independently.

For **each view**, all of the following are required:

1. Grand paired mean delta is at least `+0.03 R²`.
2. All three seed mean deltas are strictly positive.
3. At least `ceil(0.75*N)` session cross-seed mean deltas are strictly positive.
4. The hierarchical session-by-seed percentile-bootstrap 95% lower bound is strictly positive.
5. Shared-T4 absolute grand mean R² is strictly positive and all three shared-T4 seed mean R² values are strictly positive.

The bootstrap is frozen to 100,000 replicates using `numpy.random.Generator(numpy.random.PCG64(68820260805))`. Each replicate samples `N` sessions with replacement and, independently within every sampled session, samples three seed indices with replacement from seeds 42/43/44. The statistic is the mean of those `N*3` paired deltas. The two-sided percentile interval uses quantiles 0.025 and 0.975 with NumPy's `linear` quantile method.

## Safety boundary and next authorization

V2 authorizes none of the following:

- NWB download or content access;
- score-blind schema preflight;
- checkpoint loading, prediction, scoring, or GPU use;
- retry, overwrite, target-driven selection, backpropagation, optimizer state, or target updates.

The external score-only runner is deliberately absent and blocked. Before it can exist, a separate authorization must first produce an all-22, score-blind compatibility receipt, preserve `N >= 6`, freeze one common cohort, and use a fresh single-use output root. If strict T4/TS4 matching cannot be demonstrated, this endpoint fails closed.

## Reproduction

From the repository root, the metadata-only freeze is created once with:

```bash
python3 sua_exploration/scripts/freeze_dandi688_subm_co_scope_v2.py
```

The immutable output is then reproduced and byte-verified with:

```bash
python3 sua_exploration/scripts/freeze_dandi688_subm_co_scope_v2.py --verify-existing
```

These commands access only public DANDI metadata and already-local metadata/artifact hashes. They do not download or open NWB data, import model weights, compute predictions or scores, or use a GPU.
