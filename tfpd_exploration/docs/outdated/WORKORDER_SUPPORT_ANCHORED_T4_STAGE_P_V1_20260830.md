# Work Order — Support-Anchored T4 Stage P (deployable pseudo-direction)

```json
{
 "authority": "DESIGN_SUPPORT_ANCHORED_CAUSAL_T4_MEMORY_20260830.md sections 5-8; operator GO after Stage O",
 "cells": "P0 (activity-only frozen T4 reference) / P1 (cross-group circular ensemble direction, anchored block refit) / P2 (trajectory-smoothed cross-group ensemble) / P3 (P2 direction, constant confidence) / P4 (P4 deterministic group/confidence shuffle) / P5 (same-group, self-labeling diagnostic only)",
 "commit_law": "three-factor gate ACTIVE in stage P (measurement confidence + direction coverage + support trust region); hyperparameters source-selected ONLY (within-6 folds); M30 exact no-op; reuse the stage-O package machinery (anchor/block_refit/trust_region/replay) additively",
 "date": "2026-08-30",
 "gates": "promotion: candidate-P0 >= +0.01 external, >=10/15, other low budget >= P0-0.01, within >= P0-0.02, beats constant + shuffle + no-pseudo controls (design 8.1); continuity attribution P2-P1 >= +0.01 (8.2); confidence attribution vs P3/P4 (8.3); stop conditions design 8.4",
 "process": "additive src/support_anchored_t4_stage_p_v1 + driver + no-data/no-CUDA tests; attempt-before-data; 0444 receipts; O0/P0 bit-anchors to sealed activity-only rows; leakage false on all P rows; single idle GPU after review; stop if P1/P2 fail to beat activity-only",
 "schema": "support_anchored_t4_stage_p_workorder_v1",
 "stage_o_anchor": {
  "m10_external_o2_minus_o0": 0.05767866103494063,
  "m4_external_o2_minus_o0": 0.1571351238257375
 },
 "stage_o_decision": "GO_DEPLOYABLE_PSEUDO_DIRECTION_PROGRAM"
}
```
