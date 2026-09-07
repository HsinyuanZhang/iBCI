# Work Order — M2 A0 Historical C-Pre Binding V1 (2026-09-02)

## 1. Purpose

This is a narrow lineage repair for the M2 A0 chunk-memory experiment. It does
not change the C-Pre science, A0 science, query coordinates, carrier, activity
state, model, checkpoint, batching, scheduler, or result-root names.

The completed C-Pre V2 metadata audit is an immutable historical predecessor.
A later transport-only GPU UUID normalization changed the current A0 source
closure. That source-only change must not force a rerun or rewrite of the
already accepted CPU-only C-Pre result.

## 2. Inherited authorities

- Parent A0 work order SHA-256:
  `5367edd6d6f3455c59fd892d5dd7113076191a1ca584fb4dfd3c975f02c7c64b`
- C-Pre V2 repair work order SHA-256:
  `24b9ed27884d70b413f18aeadd827a0258d80eb96d717901643eec343510d055`
- C-Pre V2 root:
  `tfpd_exploration/results/m2_cpre_metadata_v2`
- Exact accepted C-Pre V2 bodies:
  - `attempt.json`:
    `64d594c44b1f9b7f4670caa00b3fc754db047a0c8ae592e5c45ee5018dbe4be8`
  - `metadata_inventory.json`:
    `a0c76c912ca6aab10a427b3e28a50b1063dfbd68b7dc09d588f256bc1e5d7af2`
  - `terminal.json`:
    `f98352a6aa95c35fe8fdc76bc5f9b47cdbac27a3c95aecfbf1f6f9a1692981b6`
- Historical C-Pre execution closure SHA-256:
  `c9d7851c97142c75cc3a2c0ac546ba156040e3decd26dffdec0d5558dbe59fe8`

The V1 failed C-Pre predecessor embedded in the accepted V2 graph remains:

- attempt:
  `47dabd4b1f364afb8e83402c1f7f098603e6fc3c6decbe2e1248f07699038e67`
- failure:
  `f91383045b4c7ee63b37272419b753d8779bbb456928d2ada081176e922f150a`

## 3. Required validation law

Before A0 capability issuance, and again before A0 terminal or failure, the
route must descriptor-read the exact six-leaf C-Pre V2 graph with no-follow
semantics and verify:

1. exact body SHA-256 values above;
2. regular files, mode `0444`, canonical sidecars, and no extra leaves;
3. terminal-to-attempt and terminal-to-inventory links;
4. the exact embedded V1 failure predecessor;
5. `target_access=false`, `target_values_read=false`,
   `model_or_checkpoint_opened=false`, and `cuda_initialized=false`;
6. the exact historical C-Pre closure SHA-256 above; and
7. the V2 full-window coordinate law already frozen by the repair work order.

The validator must not require the historical C-Pre closure mapping to equal
the current A0 execution closure mapping. A0 separately binds and revalidates
its current source closure in its own attempt, capability, terminal, or failure
receipts.

## 4. GPU UUID transport representation

For the pinned physical GPU0 only, the runtime may observe either the canonical
UUID `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9` or the same UUID with exactly
the single `GPU-` prefix omitted. It must canonicalize the latter to the former
and record both raw and canonical values. All other representations fail
closed. GPU1 remains forbidden and unqueried.

## 5. Required tests and execution boundary

- A current-source-closure change must not invalidate the exact historical
  C-Pre predecessor.
- Any historical body, sidecar, mode, topology, semantic, predecessor, or
  historical-closure drift must fail closed.
- Any current A0 closure drift must independently fail closed.
- The accepted C-Pre root must not be rewritten, rerun, or replaced.
- No A0 launch is authorized until the repaired no-CUDA suite and current
  closure audit pass.

