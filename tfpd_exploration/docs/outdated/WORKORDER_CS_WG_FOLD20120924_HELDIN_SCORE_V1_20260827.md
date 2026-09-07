# CS-WG Fold-20120924 Held-In Score V1

## Purpose and boundary

This is a narrow, descriptive, one-session held-in score for the completed
CS-WG fold whose unopened outer session is `20120924`.  It answers only what
the accepted source-only model predicts on that held-in session under the
frozen M1 evaluation graph.  It is not a matched ERM comparison, a formal
benchmark claim, a held-out/minival score, or a target-training route.

The implementation stage is code plus synthetic/CPU tests only.  It must not
open an NWB or checkpoint tensor, initialize CUDA, reserve a root, issue a
capability, or execute a score.

## Immutable producer selection

The sole accepted producer root is:

```text
tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa/fold_20120924_cswg
```

Its exact terminal graph has 28 body/sidecar pairs (56 leaves), no failure,
and is bound by:

```text
terminal.json                         efd084573843e05ada6c28c050c28e8bbf01976bec443d864bc3a0a7921afdc5
training.json                         0250b689de465b8777485883e9646c5fc64bfa9d5ea710b8670afffece5406b1
terminal identity canonical SHA       c6b23446bca124b26b47c01485de19354be4d47a42a2bc5e59e43ee0b34b56d3
checkpoint_manifest.json              e190f6c813d5f9403e7d4f505cebbeb4830edb63f1ae2a11d15bdc9907a40550
checkpoint_best_source_train_loss.pt  2cefa5cbeec5653a4fed76cacfb61113ae47bf4546879cf6cc9f6f427890d9e9
best checkpoint state                 d5d86325e21b5a257a44ee2eff4a9e35ddba9d1d6b0de591e607538bbbb41db7
```

The manifest must additionally prove that `checkpoint_last.pt` has the same
body/state and that both best and last select epoch 19.  The score explicitly
selects role `best_source_train_loss`; it must not treat the equality as an
unbound role choice.

The loader holds the root through no-follow descriptors, requires all 56
leaves to be regular `0444` nlink-one body/sidecar pairs with canonical
basename sidecars, rejects any failure/extra/missing leaf, reconstructs the
terminal identity, and revalidates the graph before terminal publication.

## Fixed scientific surface

* Outer held-in metric session: `20120924` only.
* Source producer sessions remain exactly `20120926`, `20120927`, `20120928`.
* Graph: frozen M1 `SpintModel`, window 100, 64 units, output 16, M10
  calibration `[10,1024,64]`, and the selected sealed best source-loss
  checkpoint.
* Input path: closure-bound direct `FalconDataModule.prepare_session_data`
  plus `FalconDataset`, with a route-owned metric-only target capability.
  No normalizer fitting, source refitting, target optimizer, backward, or
  update is permitted.
* Evaluation uses `model.eval()`, `torch.no_grad()`, dynamic dropout off, and
  final-bin prediction/target rows only.
* Governing score is the frozen M1 metric:
  `torchmetrics.regression.R2Score(multioutput="variance_weighted")` over all
  valid final-bin rows of the one session.  Metric labels are read only for
  this computation; target optimizer/backward/update are exactly zero.

An absolute CS-WG R2 may be reported.  A historical raw-head row may appear
only as an explicitly descriptor-bound descriptive reference on the identical
surface; it cannot be presented as paired ERM unless a separately sealed
matched-ERM graph is bound.

## Future lifecycle

The public CLI is dry and cannot issue a capability.  A future root-reviewed
execution must validate the exact producer graph, current closure, fixed
target source-root lexical capability and fresh score root before reservation.
The capability binds the exact absolute source-root string but does not
stat/open it until the durable score attempt; the sealed metadata authority
then supplies the sole `20120924` relative path/SHA descriptor.  The ordered
result lifecycle is:

```text
attempt -> launch -> input_authority -> score -> terminal
```

Any exception after `attempt` publishes an immutable `failure` with factual
progress.  Attempt precedes target descriptor/NWB open, checkpoint tensor
load, model construction, CUDA initialization, and all forwards.  The score
receipt binds target input digests, selected checkpoint/proof, model state
before/after equality, final-bin R2, target metric-only access, and zero
target optimizer/backward/update.

## Mandatory no-data tests

Synthetic tests must cover full-graph topology/SHA/identity/manifest/checkpoint
tampering; selected-role ambiguity; attempt-before-backend; immutable success
and failure lifecycles; no-target-update flags; metric parity on deterministic
CPU tensors; last-bin-only rejection; model eval/no-grad/dropout proof;
closure drift; public dry CLI inertness; and no CUDA initialization.
