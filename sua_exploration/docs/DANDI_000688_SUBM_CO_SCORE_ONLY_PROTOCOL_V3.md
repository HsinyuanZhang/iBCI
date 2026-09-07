# External sub-M score-only V3 — V5R2-parity-gated CPU prelaunch

Status: `STATIC_V3_SCORE_ONLY_PACKAGE_NOT_AUTHORIZED`

V3 is append-only. It does not change, delete, reopen, or authorize the
sealed score-only v2 package, V5/V5R2 parity packages, the V5R2 consumed
nonce, or the root-reviewed V5R2 success receipt. V3 prelaunch and dry-run do
not open a sub-M asset, NWB, checkpoint, normalizer, model, or prediction
output; they do not execute a forward pass or R2 computation.

## V5R2 compatibility prerequisite

Every V3 authorization binds this one immutable, successful parity execution:

- receipt: `parity_execution_receipt.json`, SHA-256
  `faace6493fcf939e87bf0f5ad219f75df739434b7941fccc7f654c23d78c63e8`;
- input trace: SHA-256
  `c3309f4d9e60de96517ea9c2cfad5d1f89c28398f3644cbb9a379d17779ae859`;
- environment: SHA-256
  `cf2bf2010ba326a3445d409902cb3343c05380ae5b29825d250a997d06345f02`;
- execution seal: SHA-256
  `b5f92c71a0e601fcdbd8866d4d7e1ec882ce9f285936431bfbe27aa7e9a03e18`.

V3 verifies all four files are regular, mode `0444`, sealed together, and
semantically coherent. The receipt must have schema
`dandi_000688_subc_explicit_v5_source_closure_execution_receipt_v5r2`, status
`PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION_PENDING_SEPARATE_ROOT_REVIEW_V5R2`,
and `external_subm_scoring_performed=false`. Reference and adapter must have
identical finite float32 prediction/target digest records, identical R2
`0.4910046458244324`, and the exact verified hashes. The shared observer must
record `input_exact=true` and 456 forward calls. The bridge trace must prove
the raw chronology SHA is unchanged before/after, only `start`/`stop` are
cast, and no bin/selection/T4/query-valid-start semantics changed by the
bridge.

## Frozen score-only matrix

V3 pins the sealed v2 four-source closure and three-file v2 prelaunch bundle;
it reuses its literal fixed cohort without rediscovery:

- N=15 common sub-M sessions;
- views: SUA primary and pseudo-MUA key secondary;
- arms: `shared_t4` and `shared_ts4`;
- seeds: 42, 43, 44;
- rewarded support: the first 50 chronological trials;
- activity identity: `first_n30` / 30 rows;
- T4/TS4 feature pool: 50 rewarded trials;
- query: every valid 50-bin window strictly after rewarded trial 50.

This remains 180 sealed session/view/arm/seed cells. V3 does not add a cohort,
arm, seed, endpoint, or experimental matrix cell.

### Claim boundary

The frozen two-arm endpoint is an attachment/content contrast only:
`shared_t4` (correct T4 attachment) versus `shared_ts4` (the frozen
seeded shuffled attachment). It can support the direction of that contrast
separately for primary SUA and key-secondary pseudo-MUA. It cannot support an
absolute “T4 over SPINT” claim because this matrix contains no same-window
shared-B0 control. V3 does not add B0, relabel either arm as B0, or infer an
absolute claim from the two-arm result. A B0 question requires a separate,
future append-only extension with its own authority, prelaunch, and review.
No same-window shared-B0 control is present in this V3 package.

## Authorization and runtime fence

The runner has no caller policy/key override. It reconstructs policy only
from an immutable V3 prelaunch and fixes the dedicated V5R2 public anchor.
A future score command requires canonical authorization JSON, strict-base64
64-byte detached Ed25519 signature, exact policy/source/parity bindings, a
fresh 256-bit nonce, a unique output root, and issue/expiry times no more than
15 minutes apart. Signature, source, policy, expiry, output, and nonce gates
finish before importing the V5 bridge, V3/V2 score runtime, Torch, NumPy, or
opening any external path.

The only future external operations permitted are CPU frozen-model forward
inference and CPU TorchMetrics R2 on the frozen 180-cell matrix. CUDA, normalizer
fitting, optimizer/gradient/backward, target updates, and retraining are
forbidden. The eventual external fixture constructor must call the parity-proven
V5 `bridge_owner_chronology_for_c1_builder` and the same C1 owner path; it must
not call the older score-only adapter loader that failed the float slice schema.

This V3 prelaunch is not an authorization. A separate root review and new
single-use V3 capability are required before any sub-M path can be accessed.
