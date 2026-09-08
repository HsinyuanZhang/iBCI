# Cross-session final artifact pipeline

After root decides to start the terminal pipeline, run:

```bash
python scripts/cross_session_v1/finalize_artifacts_when_complete.py \
  --root /absolute/path/to/results/cross_session_v1
```

`--poll-seconds` defaults to 15 and is restricted to 1--30 seconds.  Before
all cells finish, the wrapper only reads `program_continuation.json`, its
cell status/PID fields, and any available heartbeat JSON.  It emits progress
when state changes or at least every 60 seconds.  It never starts, resumes,
retries, kills, or otherwise changes a training or scoring process.

The continuation ledger must enumerate exactly the canonical 33 seed-42 cells
(four M1 folds, one M2 fold, and six H1 folds, each with Z/B/D), all with
`COMPLETED` status.  A `FAILED` cell, a controller that disappears before
completion after one unchanged-ledger reread, a reused PID whose `/proc/<pid>/cmdline` does not identify
`run_program_continuation.py`, or a noncanonical ledger stops the wrapper.
Completion also requires the continuation controller to have naturally exited
and the terminal ledger status to be `PRIMARY_GRID_COMPLETED_AWAITING_FIGURE`.
An empty command line for a still-present `/proc` PID is treated as a transient
exiting state and is waited out; a nonempty mismatched command line fails.

At startup it fixes SHA-256 bindings for `execution_source_seal.json`, this
pipeline wrapper, the
existing source-pairing auditor, the existing figure collector, and the prior
queue handoff audit.  Before execution it checks that those bindings and every
file listed in the execution source seal remain unchanged.  It verifies the
handoff's original-program SHA and confirms that PID 103380's recorded
`run_program_overlap.py` controller is absent; it makes no ledger write before
full completion and the pre-merge guards pass.

Before any ledger write, it calls `summarize_figure.extract` for every complete
cell.  This validates the actual scoring artifacts and requires each receipt
SHA to equal the `score_sha256` recorded in the continuation ledger.  It then
refuses to proceed if any of these final destinations already exists:
`prior_queue_program.json`, `source_pairing_audit.json`, `final_figure/`,
`ledger_merge_receipt.json`, or `post_training_pipeline_receipt.json`.

The merge copies exact bytes: it archives the old `program.json` as
`prior_queue_program.json`, atomically copies `program_continuation.json` to
canonical `program.json`, and writes `ledger_merge_receipt.json` with old,
new, continuation, and archive SHA-256 values plus its timestamp.  It then
uses the current Python executable to invoke, in order:

```text
audit_source_pairing.py --root ROOT --out ROOT/source_pairing_audit.json
summarize_figure.py --root ROOT --out ROOT/final_figure
```

It stops on any error.  On success it writes
`post_training_pipeline_receipt.json` with SHA bindings for the merge receipt,
both ledgers, source-pairing audit, every final-figure file, fixed inputs, and
the 33 verified score receipts.  Its `ARTIFACTS_READY_FOR_REVIEW` status means
only that root may inspect the figure and integrate verified results into the
paper; it does not mean the paper or the overall goal is complete.
