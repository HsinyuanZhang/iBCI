# Posterior Carrier Quick Screen V3

Status: **NON_GOVERNING_QUICK_SCREEN**. This work order authorizes only a
small remote engineering screen after a separate code audit and explicit
root-reviewed launch capability. It does not replace the completed full
matched scorer, establish a paper claim, select a checkpoint, or authorize a
target update.

## V1/V2 failure lineage and V3 successor boundary

The reviewed V1 remote attempt failed honestly at `prepare`, before an
evaluation NWB was opened and before a model, optimizer, backward pass, or
target update. Its remote result root remains immutable evidence and is not
reused as a V2 or V3 output root:

| V1 immutable leaf | Exact SHA-256 | Required fact |
| --- | --- | --- |
| `attempt.json` | `7620c853d2b4ff5c2aae359859956bc94bca9c48740294103c453674bd0a8e23` | attempt reserved before evaluation-path resolution |
| `failure.json` | `275111598568502cabc7c68350644b2599108c1930cdb622057c27dd7a5cf965` | `stage=prepare`, `PhysicalQuickScreenError`, no input/score/terminal, no target update |

Both V1 bodies and their canonical basename sidecars are regular `0444`
leaves under the prior remote stage.

V2 is also immutable failed-predecessor evidence. It reached `prepare` after
the TorchMetrics compatibility repair but failed before an evaluation NWB,
input authority, model forward, optimizer, backward pass, or target update:

| V2 immutable leaf | Exact SHA-256 | Required fact |
| --- | --- | --- |
| `attempt.json` | `11c586cc6bf837cf21c4972acd797018ebb40cf48b2be12c0619490f45147383` | attempt reserved before evaluation-path resolution |
| `failure.json` | `d06a6f9afc5ba0015010a9cedf1e96ab872944c6851334a6b67308c851a348be` | `stage=prepare`, `PhysicalQuickScreenError`, no input/score/terminal, no target update |

The V2 failure error digest is
`85b311481adb2e5455238d651192d4ab5c9f2cc55baf3d7a374fa4f3306e9456`.
Both V2 leaves and their canonical basename sidecars are `0444`. A V3
identity binds both immutable predecessor roots, exact body SHA-256 values,
sidecar grammar, and failure facts. The private V3 executor descriptor-
validates both four-leaf failed-root topologies before it accepts a V3
reviewed capability or reserves a V3 score root. It must reject a missing,
extra, substituted, symlinked, mode-drifted, or semantically inconsistent
predecessor root. It may never delete, overwrite, or add a predecessor leaf.

## Exact final-four SWA compatibility repair

V2's second failure is an engineering validator mismatch, not permission to
weaken the Posterior full-SWA evidence. The authoritative full-training route
forms every floating final-four state tensor in its native dtype using the
exact ordered algorithm `zeros_like(first)`, four `add_` calls, then
`div(4)`. The frozen general matched scorer instead reconstructs the same
quantity through an FP64 accumulator and casts back. For the completed
Posterior state, the two reconstructions have the same 31-key topology but
the FP64 alternative differs in 27 floating tensors by small rounding
amounts; it is therefore not the serialization algorithm that produced the
sealed full-SWA bytes.

V3 makes a deliberately narrow, auditable repair. Its route-local remote
backend continues to inherit the frozen parser, no-cache input materializer,
model builders, Cell-D validation, carrier construction, batching, forward
execution, and scoring. It has only two route-local overrides:

1. the already reviewed Torch-only 5070 Ti device attestation; and
2. Posterior `_validate_swa_against_checkpoints`, which first runs the frozen
   validator and accepts only its exact known FP64-arithmetic mismatch. It
   then recomputes the four checkpoints with the authoritative native-FP32
   `zeros_like -> ordered add_ -> div(4)` rule, requires bitwise equality to
   the stored SWA state, strict-loads a fresh Posterior wrapper, and requires
   the terminal-bound state digest `cd3df34ce009636837207ef4ad3b7b213e70b4b7a6e4e73ec273cc8512f206e5`.

Any other frozen-validator error remains fatal. The sealed Cell-D SWA stays
on the frozen base validation path and is never relaxed. The V3 score receipt
must state `final4_exact_recompute=PASS__AUTHORITATIVE_FULL_TRAIN_FP32_ACCUMULATION`
and separately disclose that the base FP64 reconstruction is incompatible
with this authoritative FP32 construction; it is not a bypass, a different
checkpoint choice, or an averaging approximation.

## Fixed purpose

The screen answers one operational question cheaply: on a predeclared,
heterogeneous 3+3 subset, is the completed Posterior Carrier full-SWA
directionally better or worse than the sealed Cell-D point system when the
carrier prefix falls from M30 to M4? It is a signal screen, not a governing
result. A null, positive, or negative screen never changes the full scorer's
matrix, decision rule, or authority.

## Frozen selected sessions

The selections are indices into the existing ordered immutable authorities,
not a score-dependent choice:

| Surface | Frozen roster indices | Sessions |
| --- | --- | --- |
| strict within | `[0, 2, 5]` of C1 val-6 | `sub-C_ses-CO-20151103`, `sub-C_ses-CO-20151106`, `sub-C_ses-CO-20151112` |
| fixed-ledger external | `[0, 7, 14]` of sub-M-15 | `sub-M_ses-CO-20140307`, `sub-M_ses-CO-20150611`, `sub-M_ses-CO-20150626` |

The route descriptor-derives the complete C1 val-6 and external-15 tables
from the existing immutable manifest/ledger/scope and then exact-selects those
indices. It never accepts a caller-provided asset mapping.

## Exact four-cell matrix

For every selected session, materialize the input once and evaluate exactly
these cells, in this order:

1. `sealed_cell_d_ols_point`, M30;
2. `posterior_full_swa_aligned`, M30;
3. `sealed_cell_d_ols_point`, M4;
4. `posterior_full_swa_aligned`, M4.

All use the last bin of each valid 50-bin query window and retain the formal
metric label `tfpd_lane.matched_scorer.session_r2` (variance-weighted
two-coordinate R2). The V3 engineering implementation computes that quantity
route-locally as a manual variance-weighted two-coordinate R2 because the
frozen helper constructs the removed TorchMetrics 1.9 `num_outputs` argument.
Before any input path is resolved, V3 requires a deterministic
manual-vs-TorchMetrics 1.9 parity check; CPU tests also compare the manual
implementation with the local TorchMetrics 1.5.1 frozen helper on random,
nondegenerate fixtures at tight tolerance. The receipt explicitly labels this
as an engineering manual-parity implementation, not a new estimator or a
governing result.
The receipt first reports one R2 per session, then the equal-session mean and
median. For each surface and M30/M4 it reports the paired
`posterior - sealed_point` deltas, their mean/median, and positive-session
count. It contains no PASS/STOP/HOLD verdict and cannot be cited as a
governing metric result.

The Cell-D arm uses its sealed ordinary point-carrier normalizer at M30 and
M4; it is a sealed-system replay, not a point-budget-mix-trained control. The
Posterior arm uses its one frozen full-training source prior and posterior
normalizer. Both use identical materialized neural data, behavior, M30 B3S
calibration data, valid window starts, and prefix row IDs. Neither refits a
normalizer or samples a target carrier.

## Explicit exclusions

There is no M10, H1, formal/held-out surface, zero carrier, wrong-pair
carrier, full-window headline, target fitting, backward pass, optimizer step,
normalizer refit, target sampling, score-driven session choice, or checkpoint
selection. No mechanism diagnostic may be silently added later.

## Staging and remote boundary

The only evaluation NWBs eligible for transfer are the six selected rows. The
local source is SHA/size checked against C1 / v2-ledger / scope metadata before
it enters a framed transfer. The remote stage receives a fresh regular `0444`
copy only; it may not use a symlink, bind mount, cache, or a source-training
NWB. The six NWBs total 392,880,348 bytes.

The stage may additionally receive exact immutable code, metadata, sealed
Cell-D terminal/SWA/baseline bytes, and an exact byte-preserving local imported
Posterior full-result mirror. The existing remote posterior training root is
read-only evidence only and is never mutated. A future stage manifest names
every transferred leaf, size, SHA, and role.

The remote RTX 5070 Ti is an engineering device. Its reviewed environment has
an NVML driver/library mismatch, so the route must attest it through Torch
only: one visible `cuda:0`, `torch==2.13.0+cu130`, CUDA `13.0`, cuDNN
`92000`, TorchMetrics `1.9.0`, `NVIDIA GeForce RTX 5070 Ti Laptop GPU`,
capability `[12,0]`, and `12,346,195,968` Torch-reported bytes. Before any
model score, a synthetic manual variance-weighted-R2 parity gate must agree
with the TorchMetrics implementation. It must record
`nvml_status=UNAVAILABLE_DRIVER_LIBRARY_MISMATCH` and literal `null` for
UUID, BDF, and nominal NVIDIA-SMI memory; it must not call `nvidia-smi` or
substitute any local RTX 3090 authority. This engineering receipt does not
claim metric comparability beyond the non-governing screen.

TorchMetrics 1.9 constructs
`R2Score(multioutput="variance_weighted")` without `num_outputs`.

## Lifecycle and access order

The public CLI is dry, no-network, no-Torch, no-data, and no-write. Visible
flags cannot launch the route. A root-only in-process capability is required
for staging or score execution.

A fresh V3 remote stage is
`/home/xinyuan/Work_host/posterior_carrier_quick_screen_stage_v3`; its fresh
score root is `tfpd_exploration/results/posterior_carrier_quick_screen_v3`.
The V1 and V2 stages/roots are read-only predecessors only. The V3 executor
validates both predecessor roots first, then publishes an immutable V3 attempt before any
evaluation asset pathname is resolved. Success publishes `score.json` and `terminal.json`
as one O_EXCL/fsync/0444 group. A post-attempt failure writes one immutable
failure terminal, with honest opened-surface/device/update facts, and never
leaves a partial scientific score. Existing authorities, completed training
results, and running TF-SR routes are never edited or overwritten.

## Review gates before launch

The additive code must have no-data synthetic tests for fixed selection,
matrix order, derived asset substitution, transfer plan identity/size/SHA,
attempt-before-input, no-update boundaries, same-input evidence, aggregation,
atomic success/failure publication, output collision, and dry public CLI.
After the implementation is frozen, an independent root audit must approve
the code and a transfer plan before any remote staging or GPU forward begins.
