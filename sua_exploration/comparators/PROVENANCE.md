# Provenance of copies in `sua_exploration/comparators/`

These files are **copies**. The originals remain the authoritative, receipt-bound artifacts.

The receipts' internal `implementation_binding` fields point at the **original** module paths under
`sua_exploration/mc_maze/` and `sua_exploration/scripts/`, **not** at the copies in this folder.
Therefore a number may only be published citing the original path and SHA recorded inside the
receipt itself.

The code subtree (`core/`, `runners/`, `tests/`, owned by a separate agent) contains copies whose
imports were **deliberately edited**, so those will **not** be byte-identical to their originals.
That is expected. This file does not hash those files; the other agent reports their diffs.

SHA256 values below were computed with `PYTHONNOUSERSITE=1 ~/miniconda3/envs/spint/bin/python` from
the repository root. Originals were hashed before and after the copy; inode, size, and SHA256 were
unchanged.

## Copied-file table

| Destination | Original | SHA256 of original | SHA256 of copy | Match |
|---|---|---|---|---|
| `sua_exploration/comparators/docs/HANDOFF_COMPARATORS_20260812.md` | `sua_exploration/docs/HANDOFF_COMPARATORS_20260812.md` | `0d756f3229caabeb6a5b0181e10756889be502f991f6e491437c88e643df8373` | `0d756f3229caabeb6a5b0181e10756889be502f991f6e491437c88e643df8373` | yes |
| `sua_exploration/comparators/docs/HANDOFF_COMPARATORS_20260812_zh.md` | `sua_exploration/docs/HANDOFF_COMPARATORS_20260812_zh.md` | `03d53949c507ee733b5548b4756155ca6eb097b8014f596a3a20c89d5d7f4760` | `03d53949c507ee733b5548b4756155ca6eb097b8014f596a3a20c89d5d7f4760` | yes |
| `sua_exploration/comparators/docs/SOURCE_POOLED_RIDGE_PROTOCOL_20260812.md` | `sua_exploration/docs/SOURCE_POOLED_RIDGE_PROTOCOL_20260812.md` | `0301714154a28ec2986e2073ddc37145f03e2ce4d92ef34638574fd288fbd0fd` | `0301714154a28ec2986e2073ddc37145f03e2ce4d92ef34638574fd288fbd0fd` | yes |
| `sua_exploration/comparators/receipts/h1_ridge_family/h1_ridge_family_receipt.json` | `sua_exploration/results/h1_ridge_family/h1_ridge_family_receipt.json` | `37e32b01994600588ae544a0726d4e9ddfcc27d476195d59f239e92d21639e84` | `37e32b01994600588ae544a0726d4e9ddfcc27d476195d59f239e92d21639e84` | yes |
| `sua_exploration/comparators/receipts/rt_classical_comparators/rt_classical_comparators_receipt.json` | `sua_exploration/results/rt_classical_comparators/rt_classical_comparators_receipt.json` | `c51cb0ff7dadd3c40ca7861dea92d80f3457c709801ea8ec7ba3e91b9e52b042` | `c51cb0ff7dadd3c40ca7861dea92d80f3457c709801ea8ec7ba3e91b9e52b042` | yes |
| `sua_exploration/comparators/receipts/population_vector_comparator/population_vector_comparator_m2_audit_receipt.json` | `sua_exploration/results/population_vector_comparator/population_vector_comparator_m2_audit_receipt.json` | `de5fb95f904b82a44b0e0a9f20808e60868ff0d9a14f98f991385bdac41bcbd6` | `de5fb95f904b82a44b0e0a9f20808e60868ff0d9a14f98f991385bdac41bcbd6` | yes |
| `sua_exploration/comparators/receipts/population_vector_comparator/QUARANTINE_scope_violation/population_vector_comparator_h1_audit_receipt.json` | `sua_exploration/results/population_vector_comparator/QUARANTINE_scope_violation/population_vector_comparator_h1_audit_receipt.json` | `cab4e268f6521002b82c4f48d1a50515e25b7c2937a005c94ceeaf61ec678a34` | `cab4e268f6521002b82c4f48d1a50515e25b7c2937a005c94ceeaf61ec678a34` | yes |
| `sua_exploration/comparators/receipts/population_vector_comparator/QUARANTINE_scope_violation/README.md` | `sua_exploration/results/population_vector_comparator/QUARANTINE_scope_violation/README.md` | `82db44314d6a584522d030b8b892927694c6ac5e63ea566dbcb080d4c341389e` | `82db44314d6a584522d030b8b892927694c6ac5e63ea566dbcb080d4c341389e` | yes |

The H1 population-vector receipt is retained under the subdirectory name
`QUARANTINE_scope_violation` exactly. It is quarantined for a scope violation and must not be
silently promoted to look valid.

Receipt JSON copies are mode `0444`, matching the originals. The quarantined README is not a
receipt; its copy is mode `664`, matching its original.
