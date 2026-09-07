# DANDI 000688 sub-M CO scope freeze and preflight boundary

**Scope ID:** `dandi_000688_v0.250122.1735_subm_co_scope_freeze_v1`

**Candidate binding:** fresh FP32 `shared_t4` C1, seeds 42/43/44, `lambda_consistency=0`

**Current status:** metadata scope may be frozen; NWB compatibility preflight and external scoring remain unauthorized and unimplemented.

## 1. Decision

The only near-term external-subject confirmation scope is every center-out asset for `sub-M` in
the immutable DANDI 000688 published version `0.250122.1735`. This is a cross-animal confirmation
inside the same Dandiset and study family. It is not an independent-dataset or independent-lab
confirmation.

The metadata freeze is intentionally separate from both later stages:

1. a separately authorized, score-blind NWB schema/feasibility preflight; and
2. a separately implemented and authorized external score-only runner.

The scope freeze authorizes neither stage. In particular, it does not authorize downloading an
NWB, opening an NWB, computing a neural/behavioral prediction, or starting a GPU job.

## 2. Metadata-only generator

The generator is
[`freeze_dandi688_subm_co_scope.py`](../scripts/freeze_dandi688_subm_co_scope.py). It uses only the
Python standard library. It does not import PyTorch, PyNWB, h5py, NumPy, or project data modules.
Its network allowlist is the HTTPS production API host `api.dandiarchive.org`, restricted to the
published-version metadata subtree. It rejects `/download` paths and never follows an asset
`contentUrl`.

Run once from the repository root:

```bash
python3 sua_exploration/scripts/freeze_dandi688_subm_co_scope.py
```

The write-once output is:

```text
sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v1.json
```

The writer creates a complete temporary file, fsyncs it, and publishes it with an atomic hard
link. If the destination already exists, normal generation fails rather than replacing it. A
later metadata-drift/reproducibility check is read-only:

```bash
python3 sua_exploration/scripts/freeze_dandi688_subm_co_scope.py --verify-existing
```

The manifest omits a wall-clock creation time. Given identical official metadata, generator code,
and frozen local C1 files, generation is byte-deterministic.

## 3. Real DANDI API contract

The published-version response and asset-detail responses are metadata objects at their JSON
top level. In particular, asset digests are:

```text
digest["dandi:dandi-etag"]
digest["dandi:sha2-256"]
```

They are not fields below `.blob`. The asset-list row supplies `asset_id`, `path`, `size`, `blob`,
and `zarr`; the per-asset metadata response must agree on identifier, path, and content size before
its top-level digests are accepted.

The generator deliberately starts from the complete asset list, not a single `sub-M` query page:

| Frozen inventory fact | Required value |
| --- | ---: |
| Published assets | 111 |
| Published bytes | 13,179,483,710 |
| `page_size` | 100 |
| Page lengths after following `next` | 100, 11 |
| Page `count` fields | 111, `null` |
| All `sub-M` assets | 28 |
| All `sub-M` bytes | 2,915,252,180 |
| Selected CO assets | 22 |
| Selected CO bytes | 2,312,360,648 |
| Excluded RT assets | 6 |
| Excluded RT bytes | 602,891,532 |

The anchored include rule is:

```text
^sub-M/sub-M_ses-CO-[0-9]{8}_behavior\+ecephys\.nwb$
```

Every and only matching row is included. The six rows matching the corresponding `RT` expression
are retained in the manifest as explicit exclusions. An unknown third `sub-M` naming regime,
duplicate asset ID/path, pagination loop, missing page, changed count/byte total, missing digest,
or list/detail disagreement fails closed.

The `count=null` value on page two is the production API's observed schema, not an omitted
validation: the generator requires `111` on page one, `null` on page two, 111 unique accumulated
rows, and the exact accumulated byte total.

For each of the 22 selected assets, the generator also requires the current public metadata
contract: participant identifier `M`, the exact center-out session description, NWB encoding,
and the current `Units`, `Position`, `ElectrodeGroup`, `SpatialSeries`, `BehavioralTimeSeries`, and
`ProcessingModule` categories. Those categories are useful evidence but do not expose table rows.

## 4. What public metadata does not establish

The scope manifest must not be described as an NWB-schema PASS. Without opening NWB content, the
following remain unknown:

- whether every unit row has event-level `spike_times` and a valid `electrodes` reference;
- the per-session unit count and whether it satisfies the frozen `N < 100` rule;
- whether the trials table contains usable `result` and `target_dir` values;
- whether at least 50 chronological rewarded trials survive the exact datamodule filter;
- whether the first-50 cosine design has rank three;
- whether any complete query window remains strictly after trial 50;
- whether `cursor_vel` is a finite 2-D time series with the expected timestamps and `cm/s` units.

These are later raw-schema gates, not facts inferred from DANDI summary labels.

## 5. Frozen C1 binding

This external scope is bound to one deployable terminal FP32 checkpoint per seed. It does not bind
the development epoch-5--12 mean as if that statistic were a physical model, and it forbids target
data from choosing an epoch.

| Seed | File | SHA-256 |
| ---: | --- | --- |
| 42 | `...shared_t4_s42/epoch_ckpts/epoch_011.ckpt` | `ab9df840a07d7aeb6cc417bb684f1f5e0265d50f98168400ac915647cdfd7b9f` |
| 43 | `...shared_t4_s43/epoch_ckpts/epoch_011.ckpt` | `05c05b3ab82a2fba43c55aca523248982a954faf5f0363a0235a29d64e57ab22` |
| 44 | `...shared_t4_s44/epoch_ckpts/epoch_011.ckpt` | `a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6` |

The generator verifies the three closure `run_metadata.json` files and requires all of the
following before emitting a manifest:

- `status=completed`, `training_kind=shared_paired_view`, and the expected seed;
- no held-out/formal access recorded by the fresh C1 run;
- equal SUA/pseudo-MUA task weights, shared heads, and `lambda_consistency=0`;
- full four-dimensional T4 with pool 50 for both views;
- the same frozen source manifest, training/validation manifest, and teacher hashes;
- the terminal checkpoint file exists locally and matches its fixed hash.

The source-only side-feature normalizers are separately bound:

| View | Semantic normalizer SHA-256 | Cache artifact SHA-256 |
| --- | --- | --- |
| SUA T4 | `ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7` | `32d32a7fe1b80a139571aae0ce3c3a1d802aec21c99b23a4cff72b8a60261701` |
| pseudo-MUA T4 | `92470ad14062af6cb998e06e7696b94bfdfd20ac5e415615302a5dddc7098fcc` | `17596b29d90c29ca67efa87437eda909e53dc931fe8f897f513f7ce8e5790236` |

The source-only behavior-stat cache is also bound for both view namespaces; both current copies
have SHA-256 `821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd`.
No target-subject normalizer may be fitted. C2 is closed for this scope and may not be introduced
after target access as an endpoint rescue.

## 6. Future score-blind NWB preflight

A later authorization may permit a CPU-only compatibility pass across all 22 frozen assets. The
eligibility rules must be frozen before that access and applied identically to every asset:

1. downloaded byte count and SHA-256 equal this manifest before the file is opened;
2. NWB subject/session/task identity equals the frozen row;
3. event-level units expose `spike_times` and one resolvable electrode reference per unit;
4. `0 < N < 100`; no post-hoc truncation, padding-policy change, or cap change;
5. trials contain `start_time`, `stop_time`, `result`, and finite `target_dir`;
6. using 20-ms bins, 50-bin windows, and `result == 'R'`, at least 50 usable chronological
   rewarded trials exist;
7. the first-50 `[cos(theta), sin(theta), 1]` matrix has rank three; direction imbalance is
   reported but is not a tunable exclusion threshold;
8. at least one complete query window remains strictly after trial 50;
9. `cursor_vel` is finite, 2-D, timestamp-monotonic, and in `cm/s`;
10. if pseudo-MUA is reported, raw unit activity is pooled by validated electrode before T4 is
    refit; unit-level T4 rows are never averaged.

All 22 assets remain in the ledger. Any incompatibility must be reported; no asset may disappear
because of a model result. This preflight must not run a model forward pass or calculate an R²,
correlation, loss, behavior prediction, or other ranking statistic.

## 7. External score-only runner remains a blocker

No authorized external runner exists yet. The current frozen-manifest `MultiSessionDataModule`
path is tied to the sub-C `(27,6,6)` source/development registry and
`max_units_exclusive=100`; pointing it at `sub-M` is not a valid evaluation procedure.

Before any GPU or score, a new score-only entry point must:

- consume the exact scope manifest and a separately authorized schema-preflight receipt;
- load only the three frozen terminal checkpoints and the bound source-only normalizers;
- expose no target train loader, optimizer, backward call, checkpoint writer, or normalizer fit;
- enforce activity support `[0,30)`, T4 label/rate pool `[0,50)`, and query start 50;
- emit explicit zero target optimizer/backward/update counters and fail if any are nonzero;
- write into a fresh single-use root and refuse retry or overwrite.

Until that runner exists and passes a separate review, the correct state is
`scope_frozen_metadata_only_external_runner_blocked`, not “ready to evaluate.”
