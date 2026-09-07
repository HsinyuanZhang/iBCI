# Addendum: M2 AJPF-C V2 — source-exposure law amendment

Date: 2026-09-03
Status: successor amendment to WORKORDER_M2_AJPF_C_CONTINUAL_V1_20260903.md
Failed predecessor: `tfpd_exploration/results/m2_ajpf_c_v1` (fail-closed at
source-stream construction: `coordinate budget drift: 28999`; attempt.json +
failure.json 0444 pairs preserved, nothing trained, no GPU work started).

## Amendment (the only change)

Empirical diagnosis after the V1 fail-close: the seven held-in source
sessions contain 116,308 bins total (~39 min), 115,622 valid windows, and
1,166 chunk100 states.  The V1 coordinate law (91,717 coordinates with one
≤32 group per state, group budget 2,900) was derived from an incorrect
session-length assumption and is unsatisfiable on the real data.

V2 source-exposure law (deterministic, per law):

1. use **all** chunk states (state stride 1);
2. within each state, evenly subsample the valid windows to the largest
   multiple of BATCH_SIZE (32) not exceeding `min(state_windows, 96)`;
3. split each state's windows into groups of exactly ≤32 (one batch each);
4. group budget per epoch: 3,600; receipt discloses coordinates, groups,
   states, and discards per session and law;
5. the 91,717 AJPF figure remains the declared *reference* budget, reported
   next to the achieved count; no hard band is enforced because availability,
   not selection, is the binding constraint.

Everything else in the V1 work order (arms, pairing, optimizer, gates,
boundaries, one-shot lifecycle) is unchanged and binding.  Canonical root:
`tfpd_exploration/results/m2_ajpf_c_v2`.
