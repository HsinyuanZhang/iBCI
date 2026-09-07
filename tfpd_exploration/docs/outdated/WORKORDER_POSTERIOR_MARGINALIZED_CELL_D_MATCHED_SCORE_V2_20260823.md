# PMC-D matched scorer V2 successor — 2026-08-23

## Purpose and boundary

This is an additive successor to the failed `posterior_marginalized_cell_d_score_v1`
physical attempt.  V2 has one fresh canonical result root:

`tfpd_exploration/results/posterior_marginalized_cell_d_score_v2`

The failed V1 root is read-only predecessor evidence.  V2 must first hold and
validate its exact four body/sidecar pairs, `failure.stage == "prepare"`, no
`input_authority.json`, and no `terminal.json`.  It must never reserve, write,
or alias the V1 root.  The V2 root must be fresh and is reserved transactionally
before any input derivation.

The frozen V1 body SHA map is `preflight.json=9cfdb7ba…c3b9ddc`,
`authorization.json=ad9ab47b…c2051313`,
`attempt.json=6d05144a…3ad3114`, and
`failure.json=de4ab5a2…bcf50b07`; the implementation also binds the exact
sidecar body SHAs `85e0e5de…13a23`, `695faad7…df901`,
`370cae5f…c1f1a`, and `8b308f27…8fbb4` respectively.

The PMC-D scientific contract is inherited unchanged: Cell-D graph and SWA
provenance, deterministic ordinary OLS point T4 for M4/M10/M30, B3S inputs,
fixed governed bin 49, variance-weighted per-session R2, equal-session
within6/external15 aggregation, target optimizer/backward/update counts zero,
and paired PMC-D minus sealed Cell-D deltas/sign/95% CI.  No posterior sample,
posterior mean, posterior normalizer, credibility field, refit, or update is an
inference operation.

## Actual sealed Cell-D producer seam

The sealed producer body is frozen at SHA-256
`626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd` and is
written by `tfpd_exploration/src/tfpd_lane/matched_scorer.py`.  Its exact
weights-only CPU payload keys are:

`state_dict`, `swa_manifest`

The manifest has exactly `buffer_tensor_count`, `components`,
`floating_tensor_count`, `fp64_arithmetic`, `optimizer_state_included`, and
`uninitialized_lazy_tensor_count`; the terminal appends strict-reload and
finite-forward proof fields.  V2 validates those fields, all four component
hashes, the 29 floating/2 lazy/0 buffer counts, no optimizer state, the Cell-D
state-key and parameter topology, strict CPU reload, and derived state SHA
`c7a8489d5e17583ac210dd06b09e931ef6ba0d6b98a2767cae26d897725cf8fc`.
PMC's existing declared SWA schema remains unchanged and is loaded by the
composed V1 runtime.

## Execution lifecycle

1. Root-side code validates the immutable failed V1 graph.
2. A separately V2-root-reviewed in-process capability selects one exact
   compatible device profile; the public CLI cannot mint or pass it.  A V1
   capability is rejected.  The V2 issuer fresh-loads provenance, recomputes
   the complete descriptor-safe physical V1 closure as the V2 closure base,
   adds the V2 leaves, and binds the resulting V2 closure and identity SHA.
3. Current PMC terminal/final-four/SWA and sealed terminal/SWA/baseline
   provenance and the complete V2 physical code closure are revalidated.
4. V2 reserves its fresh root, publishes preflight/authorization/attempt as
   exact body/sidecar pairs, and only then derives fixed input metadata.
5. A composed backend strictly loads both SWAs, materializes every input once,
   replays all 12 cells, persists/reloads sealed M30 session/order/window/R2
   parity, and writes a terminal or honest failure receipt.
6. Before terminal publication it rechecks the unchanged failed V1 graph,
   current provenance, and exact V2 closure payload.  Every preflight,
   authorization, attempt, score, and terminal identity is derived from that
   V2 closure-bound identity.  Any drift fails closed.

No launch, result-root reservation, NWB access, checkpoint-tensor access,
CUDA initialization, or capability mint occurred while freezing this workorder.

## Additive closure

The V2 closure includes this workorder, `src/posterior_marginalized_cell_d_v2`
source, the V2 dry CLI, the focused no-data/actual-payload regression, the
reviewed V1 physical substrate, exact posterior-carrier descriptor reader, Cell-D
producer/model/metric leaves, and the no-cache data adapter dependencies.
