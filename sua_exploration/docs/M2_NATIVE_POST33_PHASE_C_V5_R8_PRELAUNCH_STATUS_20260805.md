> **SUPERSEDED — historical evidence only, not current authority.** r8 was retired after a pre-Hydra ABI engineering failure; r9 is now active.
> Current launch state: r9 in [`ACTIVE_EXPERIMENT_CONTROL_BOARD.md`](ACTIVE_EXPERIMENT_CONTROL_BOARD.md); device boundary in [`M2_NATIVE_POST33_PHASE_C_V5_DEVICE_RECOVERY_20260805.md`](M2_NATIVE_POST33_PHASE_C_V5_DEVICE_RECOVERY_20260805.md).
> Preserved as append-only audit evidence; content unchanged.

# M2 Native Post-33 Phase-C v5/r8 — prelaunch status

## Current disposition

**r8 is ready for an independent prelaunch review, but is not authorized to
start a GPU worker yet.** The CPU-only verifier passed and both physical-GPU
matrix dry-runs passed. No r8 cell root exists, so no selector, owner claim,
training/evaluation worker, CUDA context, formal data access, score payload, or
R² value has been materialized.

The prior r7 capability is explicitly retired and must never be used. Its
matrix dry-run exposed a schema-key mismatch before authorization claiming or
worker/CUDA initialization. The immutable retirement record is
[`r7_prelaunch_matrix_contract_retirement.json`](../results/m2_native_post33_phase_c_v5_r7_launch_receipts_20260805/retirement/r7_prelaunch_matrix_contract_retirement.json).
The r8 program binds that retirement record as an upstream constraint and uses
a new root, public key, program receipt, manifests, authorizations, and two new
nonces.

## What r8 is allowed to do

There are exactly two detached-Ed25519 `cell_execution` capabilities, both for
Stage A / seed 42 only:

| Physical GPU | Fold set | Ordered work | Cells |
| --- | --- | --- | ---: |
| `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9` | 0, 2, 4, 6 | SPINT then T4 per fold | 8 |
| `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` | 1, 3, 5 | SPINT then T4 per fold | 6 |

This is exactly 14 cells: `2 arms × 7 folds × seed 42`. No opening
capability, Stage-B capability, full-opening capability, or score-opening
authorization was written.

The fresh r8 files are located under
[`m2_native_post33_phase_c_v5_r8_launch_receipts_20260805`](../results/m2_native_post33_phase_c_v5_r8_launch_receipts_20260805), while the future cell root remains absent at
`sua_exploration/results/m2_native_post33_phase_c_v5_r8_cells_20260805`.

## Device-repair binding

The r8 source program has 117 pinned source/config entries. It explicitly
binds the full V5 device helper
`m2_native_post33_deployment_device_prep_v5.py`
(`3e476e11297e32632fe411cf0723afedd3ed81a03964b7f5abc8eb72e7e4d146`),
the r8 outer wrapper, the r8 cell pipeline, and the r8 matrix launcher.

Both full inner evaluators are literal v5 copies with exactly one AST-level
change: their capability import is redirected from
`m2_native_post33_authorization_v4` to
`m2_native_post33_authorization_v5_r8`. The same one-import-only rule holds
for the two training wrappers. Thus r8 changes the trust/capability lineage,
not the repaired evaluation, chronology, query, score-payload, decoder, or
deployment-device logic.

## Verified evidence

- CPU source/capability verifier: [`cpu_source_capability_verifier.json`](../results/m2_native_post33_phase_c_v5_r8_launch_receipts_20260805/prelaunch/cpu_source_capability_verifier.json)
  - `status = PASS_READY_NOT_LAUNCHED`
  - 14-cell coverage, two distinct fresh nonces, r6d and r7 retirement
    bindings, no cell root/selector, no private-key serialization, and no
    Stage-B/opening authority.
- Fresh public anchor: [`public_anchor.json`](../results/m2_native_post33_phase_c_v5_r8_launch_receipts_20260805/authority/public_anchor.json)
- Signed GPU0 authorization: [`stage_a_execution_gpu0_v5_r8.json`](../results/m2_native_post33_phase_c_v5_r8_launch_receipts_20260805/auth/stage_a_execution_gpu0_v5_r8.json)
- Signed GPU1 authorization: [`stage_a_execution_gpu1_v5_r8.json`](../results/m2_native_post33_phase_c_v5_r8_launch_receipts_20260805/auth/stage_a_execution_gpu1_v5_r8.json)
- Non-executing matrix dry-runs for both shards passed: [`matrix_dry_run_cpu_audit.json`](../results/m2_native_post33_phase_c_v5_r8_launch_receipts_20260805/prelaunch/matrix_dry_run_cpu_audit.json).
  They validate current program/portable/shard closure and record
  `execution_started = false`.

## Remaining authorization boundary

`run_m2_native_post33_phase_c_v5_r8_matrix.py --execute` fails closed unless
the caller supplies a separate independent review receipt. For each shard, the
receipt must contain this exact approval scope (with absolute current paths):

```json
{
  "schema": "m2_post33_phase_c_v5_r8_independent_launch_review_v1",
  "status": "APPROVED_FOR_STAGE_A_LAUNCH",
  "program": {"canonical_path": "<r8 program receipt>"},
  "shard": {"canonical_path": "<that r8 shard manifest>"}
}
```

That review is the only remaining process gate. It should independently
confirm the r8 source/program hashes, the full-v5-only worker binding, the
two-shard 14-cell scope, the r7 retirement, the empty r8 cell root, the
absence of Stage-B/opening authority, and the target GPU identities. It must
not inspect an endpoint/R²; no endpoint has been opened in this preparation.
