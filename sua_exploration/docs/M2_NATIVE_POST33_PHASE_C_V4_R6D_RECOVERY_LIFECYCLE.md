> **SUPERSEDED — historical evidence only, not current authority.** r6d was retired, not repaired.
> Current launch state: r9 in [`ACTIVE_EXPERIMENT_CONTROL_BOARD.md`](ACTIVE_EXPERIMENT_CONTROL_BOARD.md); device boundary in [`M2_NATIVE_POST33_PHASE_C_V5_DEVICE_RECOVERY_20260805.md`](M2_NATIVE_POST33_PHASE_C_V5_DEVICE_RECOVERY_20260805.md).
> Preserved as append-only audit evidence; content unchanged.

# Native-M2 post-33 Phase-C r6d recovery lifecycle

**Status:** source-and-test implementation only.  This document is a control-flow handoff, not
launch authorization.  It does not permit r6d bootstrap, plan construction, trigger creation,
matrix execution, score opening, or formal-endpoint access by itself.

## Why r6d is a fresh lineage

r6c was retired before GPU execution because its ephemeral signer could not prove the required
Stage-B lifecycle.  Its append-only retirement receipt is
`results/m2_native_post33_phase_c_v4_r6c_retirement_20260805/r6c_pre_gpu_retirement.json`
(SHA-256 `22ef77ce1a697a9901c85235b2233c820a5d7015d884fb9b25e463272c89448a`).
No r6c receipt, source map, authorization, or cell root may be reused or mutated.

r6d starts with a new receipt root, cell root, live-signer runtime directory, and public-key
anchor.  Until those fresh resources are deliberately created in the order below, r6d must remain
non-executable.

## Mandatory order

1. **Source GO first.** Independently review the r6d controller, checker, supervisor, rollover
   builder, program closure, and adversarial tests.  Resolve every high/critical concern before
   any filesystem state is created under r6d roots.

2. **Bootstrap only the live anchor and readiness record.** With the r6d receipt root, cell root,
   runtime directory, and r6d public-key path all absent, start the live signer bootstrap.  It
   creates only the new public key and `READY` record while retaining the private key in the live
   process.  It must not sign a capability until a later, independently built trigger arrives.

3. **Immediately pivot verifier source to that new anchor.** Read the newly created public-key
   SHA-256 and make the narrow, reviewed source change in
   `m2_native_post33_authorization_v4.py`: `PUBLIC_KEY` and `PUBLIC_KEY_SHA256` must point to the
   r6d anchor.  Rebuild/review the Phase-C program closure so it seals both this changed
   authorization source and the r6d controller/gate/supervisor/checker sources.

   This deliberate two-step boundary is important: before bootstrap, an r6d public-key literal
   does not exist and the authorization module legitimately still names r6c.  After bootstrap,
   retaining that old literal is a hard failure, not a permissible transitional state.

4. **Only then build the r6d plan.** The builder must fail preflight while the source delta lacks
   the authorization-anchor change.  After the pivot, it must perform all external/history/
   absence checks and scratch validation before writing any r6d target file.  It writes the
   metadata-only launch row, core, and proof before atomically creating the `O_EXCL` trigger last.
   It must never embed a caller-provided matrix command.

   The proof also records the exact rollover-builder source metadata.  This is provenance for the
   pre-trigger construction step only; the builder remains deliberately outside the execution
   runtime source-map roots.

5. **The live signer validates and signs Stage A.** It binds trigger, core, proof, launch metadata,
   source map, and r6d readiness.  It rechecks the proof's complete capability/cell-root absence
   immediately after trigger validation and immediately before the first signature write; a
   capability that appears in that interval aborts the signer.  Conditional Stage-B authority is
   not materialized until the opaque Stage-A decision is opened and the fixed gate returns
   `continue`.

6. **A source-sealed supervisor is the sole GPU front door.** Its public CLI is only
   `--gpu-id 0|1`; it reconstructs the fixed matrix argv internally after signature and source-map
   checks.  The matrix gets a dedicated process group plus `PR_SET_PDEATHSIG(SIGTERM)` and an
   immediate parent-PID recheck.  Its independent watcher deliberately has no PDEATHSIG so it can
   kill a still-identical reparented matrix group.  If that watcher exits successfully while its
   matrix is still alive, the supervisor independently requires a stable, score-free Stage-A
   completion marker; otherwise it kills the verified group and fails.  A user-systemd service with
   `KillMode=control-group` is an additional containment layer, not a replacement for signer or
   source verification.

## Explicit pre-build invariants

- No r6d receipt root, r6d cell root, or r6d execution reservation/start/completion/failure file
  may preexist.
- The retired r6c signer PID must be gone; its cell root and all execution receipt surfaces must
  remain absent.
- r6d build validation is CPU-only and score-free.  It may not access score/formal data or invoke
  CUDA.
- The r6d program source map must include the four r6d runtime sources and the post-bootstrap
  authorization-anchor source.  The expected r6c-to-r6d closure delta is intentionally incomplete
  before Step 3 and the production builder must reject that state.

Any departure from this order requires a new independent review; it is not a recoverable
"convenience" exception.
