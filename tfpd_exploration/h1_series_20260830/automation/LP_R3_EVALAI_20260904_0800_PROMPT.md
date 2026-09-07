# Scheduled task: H1 LP-R3 all-source package and EvalAI submission

Start time: 2026-09-04 08:00 Asia/Hong_Kong (UTC+08:00)

The user explicitly authorizes one H1 EvalAI submission of LP-R3, but reports
that the submission quota is currently exhausted.  Work autonomously toward
the outcome below and keep a complete local audit trail.

## Outcome

Produce a legitimate all-source H1 LP-R3 model, build and offline-validate its
Docker image, then submit exactly once to the H1 EvalAI phase if and only if a
read-only quota check positively confirms that a submission is available.

## Scientific authority

Read and preserve the contracts and results in:

- `tfpd_exploration/h1_series_20260830/docs/DESIGN_H1_SUPPORT_RESAMPLED_POSTPOOL_V1_20260903.md`
- `tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_SUPPORT_RESAMPLED_POSTPOOL_V1_20260903.md`
- `tfpd_exploration/h1_series_20260830/docs/RESULT_H1_SUPPORT_RESAMPLED_POSTPOOL_V1_20260904.md`
- `tfpd_exploration/h1_series_20260830/docs/RESULT_H1_ACTIVITY_CARRIER_RESAMPLING_FACTORIAL_V1_20260904.md`

LP-R3—not the descriptive RF simplification—is the governed selected arm.
Existing evidence is five-fold source-date LODO and is not itself a deployable
all-source checkpoint.  Do not submit an arbitrary fold checkpoint.

## Required sequence

1. Inspect the dirty shared worktree and active processes.  Preserve all
   unrelated edits and never signal, renice, restart, or overwrite another
   process or artifact.
2. Use GPU0 only.  Do not query, inspect, address, or use GPU1.  If GPU0 is
   occupied, wait without interfering; if it does not become safely available,
   stop with an explicit blocked receipt rather than switching devices.
3. Author and freeze a separate all-source LP-R3 design/work order.  Reuse the
   already validated operator, support schedule, seed 42, batch 32, and 12
   epochs.  Initialize from the sealed all-source C1 checkpoint.  Freeze the
   decoder/body exactly as in the selected LODO cell; train only the declared
   58,140-parameter late-pooling branch.  Source training uses the same 50/50
   first-M3 versus deterministic non-first contiguous-M3 paired activity and
   carrier schedule.  Target deployment uses chronological first M3.
4. Use a fresh additive result root.  Never edit or retry the sealed LODO and
   factorial roots.  Require the same gradient, frozen-state, support-pairing,
   no-target-update, and immutable receipt checks before accepting the
   all-source checkpoint.
5. Build a new H1 decoder/package and Docker image from that all-source
   checkpoint.  Follow the installed FALCON H1 evaluator contract and the
   existing sealed H1 C1 package as the implementation reference.  Do not
   change label budget, carrier law, normalizer, window law, or reset law.
6. Run CPU/unit tests, exact checkpoint and payload rehashes, decoder shape
   checks, zero/update sentinels, and an offline Docker minival using the
   official evaluator-compatible path.  Docker must run with network disabled.
   Do not infer readiness from a host-only forward.
7. Query EvalAI quota/status without creating a submission.  If quota remains
   unavailable, stop at `READY_NOT_SUBMITTED_NO_QUOTA`, keep the validated image
   and receipts, and do not attempt a push.
8. Only when every prior gate passes and quota is positively available, push
   exactly one private H1 submission.  Record image digest, payload digest,
   challenge/phase/team identifiers, metadata flags, submission ID, and the
   server response.  Do not make it public and do not submit a second time.
9. Poll the submitted job to a terminal platform state and write a concise
   result/handoff document.  A platform failure is not authorization to retry.

## Hard boundaries

- No GPU1 query or use.
- No target/hidden label access, optimizer step, or model selection.
- No replacement of LP-R3 by LP-AR-CF based on the post-hoc factorial result.
- No new architecture, epoch/LR/seed sweep, continual memory, or FiLM bundled
  into this submission.
- No deletion or mutation of prior immutable artifacts.
- At most one EvalAI submission.

Lead the final report with one of: `SUBMITTED_<id>`,
`READY_NOT_SUBMITTED_NO_QUOTA`, or a precise fail-closed blocker.
