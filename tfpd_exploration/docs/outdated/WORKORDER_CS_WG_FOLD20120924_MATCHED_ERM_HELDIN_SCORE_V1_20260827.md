# CS-WG fold-20120924 matched-ERM held-in score v1

## Purpose

Prepare a strictly additive, metric-only held-in scorer for the future
same-fold `MATCHED_ERM` full producer.  It reuses the reviewed fold-20120924
CS-WG scorer's native reader, exact M1 model materialization, strict selected
checkpoint reload, final-bin variance-weighted R2, and immutable
attempt/input/score/terminal lifecycle.  It is not an ERM training route, a
new parser, a matched target-training procedure, or a benchmark verdict.

This implementation boundary is no-data/no-CUDA.  It must not read the future
ERM result graph, an NWB, target rows, or checkpoint tensors; it must not
reserve a root, issue a capability, or execute a score.

## Fixed surface

- Target metric session: `20120924` only.
- Producer source sessions: `20120926`, `20120927`, `20120928` only.
- Model/evaluation graph: M1 `SpintModel`, W=100, U=64, raw output 16, M10
  calibration `[10,1024,64]`, `model.eval()`, `torch.no_grad()`, dynamic
  dropout off, and selected role `best_source_train_loss` only.
- Metric: final-bin
  `torchmetrics.regression.R2Score(multioutput='variance_weighted')`.
- Target labels are metric-only.  Target optimizer, backward, and update
  counts are exactly zero.
- The future ERM producer must be a completed 20-epoch source-only,
  no-SWA graph.  Its only scientific difference from the completed CS-WG
  producer is `system=MATCHED_ERM`, `lambda=0.0`, `tau=0.01`.

## Same-input comparator anchor

The successor must descriptor-hold and exact-reload the accepted CS-WG
`input_authority.json` and `score.json` body/sidecar pairs, then compare the
target-evaluation evidence field-by-field with those receipt bodies.  It must
not compare the two whole input-authority hashes because their producer graph
bindings differ.  The fixed anchor is:

```text
CS-WG input_authority body SHA   342c78953a5e7070e3168db7c7644f9d543ee36459243bc9323fe815ce8a9a00
CS-WG score body SHA             d5a08db493ced3c7284fc3d8b295c9e9609326658cf7deb9ef49f9826d6bb605
n_windows                         54849
target descriptor body SHA        63ee25782c62ff2275dcfbdcaa56552ec4c26fcde00f5a74e5be54785b5c25eb
target descriptor byte count      73077382
calibration SHA                   3dfabe28ff6bfdd90f9f866fe6ed9e406a10cd66944b9cc11a54fd8831c4c3cc
calibration shape                 [10,1024,64]
ordered-window-start SHA          7f8693db5004525fee536698860e04f9f0cb33a3508011da88bc63f6e3bfc12c
ordered-query-identity SHA        a89144a9d223603d243d9493fe7fe04e3dafa309b6014fab206170ecc7cc4f98
ordered-target-evalmask SHA       85e8ad4acae31373e2917bb6939ae1233c6d2f47be12331480f6cc8c7413e8e0
target SHA                        e913d03a972154a4a7ad3eae9963174fb54dbeefd3bc32b542b5cb76bc0ff8aa
reader recipe SHA                 559d86219c24190bc326349239a45a80f466332965b435a68a538a65f905d9b1
```

The accepted CS-WG R2 (`0.5679166316986084`) is only a descriptive reference.
No ERM-versus-CS-WG delta or formal claim is authorized until both immutable
producer and score graphs are available.

## Deferred ERM producer binding

The future selected producer root is
`tfpd_exploration/results/cross_session_worst_group_m1_matched_erm_full_v1_no_swa/fold_20120924_matched_erm`.
No digest is guessed in code.  A root-reviewed `MatchedERMFullBinding` must
explicitly provide, after that graph terminalizes, all exact body/state facts:

1. `attempt.json`, `launch.json`, and `source_authority.json` body SHAs;
2. terminal identity canonical SHA, `training.json`,
   `checkpoint_manifest.json`, and `terminal.json` body SHAs;
3. selected best checkpoint body/state SHA and last checkpoint body/state SHA;
4. best/last epoch indices.

The held no-follow graph loader requires exactly the 28 body/sidecar pairs,
regular `0444` nlink-one leaves, canonical basename sidecars, no failure or
extra leaves, exact terminal links, source-only/no-SWA semantics, and
`MATCHED_ERM/lambda=0.0/tau=0.01`.  Missing binding literals fail before
capability, root reservation, source/target access, checkpoint bytes, model
construction, CUDA, or forwards.

## Lifecycle and implementation seam

The only shared change is a typed profiled lifecycle loop plus a physical
receipt-codec seam.  The default CS-WG profile is used unchanged, preserving
its historical payload and execution behavior.  This successor provides its
own identity, producer binding, receipt schema, capability, and physical
codec; it does not copy the existing lifecycle, parser, reader, model forward,
or metric implementation.

A future root-reviewed run must validate the current successor closure,
typed ERM binding, held producer graph, exact comparator anchor, metadata
authority, lexical target source-root capability, and a fresh score root
before reservation.  The ordered result graph is:

```text
attempt -> launch -> input_authority -> score -> terminal
```

After score, it revalidates the held ERM producer graph and the published
score graph before terminal publication.  Any pre-terminal error produces an
honest immutable failure.  No public CLI can issue a capability or execute.

## No-data acceptance

Synthetic/no-CUDA tests must prove: absent producer literals fail closed;
the exact 56-leaf ERM graph and all named SHA links reload only with supplied
literals; system/lambda/tau, source authority, attempt, manifest, checkpoint,
state, and terminal drift reject; the comparator input fields reject one-field
drift; attempt precedes backend input preparation; success and failure graph
topologies are immutable; the inherited V1 profile remains byte-semantic;
the module import/dry CLI is inert and CUDA remains uninitialized.
