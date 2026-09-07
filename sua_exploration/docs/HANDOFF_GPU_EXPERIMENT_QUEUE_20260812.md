# Handoff: GPU Experiment Queue

**Date:** 2026-08-12
**Purpose:** every remaining experiment that requires GPU, with its rationale, cost, gate, and risk.
**CPU work is not listed here** — see `HANDOFF_COMPARATORS_20260812.md` Section 7 for the CPU queue
and Section 6 for the CPU arms already completed.

Measured cost basis: one H1 arm is 50 fixed epochs, about **126 MB** and **2.1-2.9 hours** depending
on GPU contention. Two RTX 3090s. Disk at 91%, ~79 GB free.

---

## 0. Live state — updated 16:40, nothing is running

**Both GPUs are idle. No `ctxv2` tmux sessions exist. No Stage B checkpoints exist.**

Stage B was launched at 15:12 and **killed at 15:55 by session teardown, not by a crash** — see the
operational incident section below. Under the fixed-epoch no-selection protocol a killed run leaves
no checkpoint, so **Stage B must be re-run from scratch**. Roughly 43 minutes of GPU time on both
cards was lost.

Nothing else from this program is running. The agent's watcher process is also dead.

---

## Queue

### G1 — Stage B: separately trained Context-LS and Context-RS, fold-0 — MUST BE RE-RUN

**Cost: 2 arms, ~2.1-2.9 h each, both GPUs in parallel. Previously launched, produced nothing.**

All CPU prerequisites are already complete and verified, so this can launch immediately once the
detachment requirement below is satisfied:

- preflight `SPINT-main/pilot_artifacts/h1_ctxv2_stage_b/CTXV2_STAGE_B_PREFLIGHT_v1.json`,
  SHA `359f819dc9e28e7ba955effd2cb036ae597a99c0e0682fc9055da37960000662`, status
  `PASS_CTXV2_STAGE_B_CPU`;
- 116/116 source blocks corrupted in both modes, zero fixed points, determinism confirmed, label
  mode verified to refit rather than permute;
- experiment configs `h1_ctxv2_context_{ls,rs}_19250101` and their data configs exist;
- corruption is applied only **after** immutable snapshot validation, so the arm provably consumed
  the audited clean source assets before one auditable transform.

**Launch with full detachment** (`setsid`/`nohup`), `PYTHONNOUSERSITE=1`, and
`PYTHONPATH=<root>/SPINT-main:<root>` — both paths, in that order; a launch with only `SPINT-main`
fails instantly and already cost one wasted attempt. Use **fresh run directories**; `_v1` and `_v2`
are both already taken and contain nothing usable.

**Purpose.** Separately trained Context-LS and Context-RS on fold-0, giving H1 control parity with
center-out (TS4/LS4) and RT (XLSv2). Motivated by the measured finding that same-checkpoint controls
understate content dependence by about 4x.

**Gate.** Protocol clause 5.2: Context Full must exceed both separately trained arms; both margins
positive. Sealed same-checkpoint references are `0.499189` (label) and `0.499152` (row) against
Context Full `0.516518`.

**Expected outcome.** Margins **smaller** than the sealed same-checkpoint values. That is the
prediction, not a failure. Only a sign inversion fails the clause.

**Before reading any number:** confirm both runs reached fixed epoch 50, score only the `_v2` run
directories, and use byte-identical query windows across arms with the query SHA recorded.

---

### G2 — Stage A multi-date Context, date `19250108` — BLOCKED

**Purpose.** The primary generalisation clause. Also the prerequisite for G3.

**Cost.** 3 arms (Context Full, H-SE5, Zero5), two waves, ~6.3 hours.

**BLOCKER — do not launch.** The fidelity gate that was supposed to validate the forked pipeline is
vacuous: it ran on fold-0, which the module routes to the sealed code path, so the dated code that
would train on `19250108` was never exercised. Measured divergence on identical fold-0 inputs:

```
sealed path manifest : c49694d850c426d58c10f3da5271bcb472e9c52d95963d619a7134a48e6adb78
dated  path manifest : db24ac9607dd5e8012bbcc3f69ed761ee362864bc68627821de521d03f8cd85d
DIFFERS: carrier_cache_sha256, normalized_cache_sha256, normalizer, normalizer_sha256
```

The dated path also uses a different batch sampler, which changes batch composition and therefore
the trained model. A result produced now would be uninterpretable: a date-2 difference could not be
attributed to the date rather than the pipeline.

**Unblock procedure** in `HANDOFF_H1_CONTEXT_PROGRAM_STATE_20260812.md` Section 2.4. Blocker file at
`SPINT-main/pilot_artifacts/h1_ctxv2_19250108/STAGE_A_BLOCKED_DO_NOT_LAUNCH.md`. Do not lower the
bit-identity requirement.

**Outcome semantics are predeclared** in `H1_CONTEXT_FULL_UNSEAL_MULTIDATE_GPU_PROTOCOL_20260812.md`
Section 5.4, including what a second non-uniform result would mean. Note that even a 3/3 second date
gives **2/2 dates, 4/5 recordings**, because fold-0 already contains one negative recording.

---

### G3 — Cross-date carrier transfer — highest scientific value, gated behind G2

**Purpose.** The paper asserts at lines 363-364 that the carrier "cannot drift like a transferred
neural fingerprint." That claim is currently **contradicted** by the only evidence bearing on it: a
forward-only transfer between the two fold-0 recordings scored `0.519024` against `full` at
`0.516518` — swapping carriers slightly **improved** performance, on both recordings.

But those two recordings are timestamps six minutes apart on the same day, so that test is
within-day between-block, not between-session. **The meaningful test is cross-date**, which needs a
second validated held-out date.

**Cost.** Forward-only once G2 exists. No training.

**Why it matters.** This is the question a reviewer is most likely to ask about the whole method:
is the carrier session-specific, or effectively a good fixed prior? Running it before submission is
strictly better than discovering the answer in review. Either outcome is publishable — a clear drop
supports session-specificity; no drop means lines 363-364 must go and the method is honestly
described as a content-and-attachment descriptor rather than a drift-tracking one.

**Note:** lines 363-364 must be deleted or qualified **regardless of G3**, on the strength of the
within-day result alone. See `HANDOFF_PAPER_EDITS_20260812.md` P0.1.

---

### G4 — Cost-of-no-backprop baseline — highest reviewer salience, highest risk

**Purpose.** The paper's entire premise is that a backward pass is unaffordable on an implanted
device. It never quantifies what that constraint costs.

**Design.** An arm that **does** adapt on the same calibration prefix, same labels, same query, same
source checkpoint. Two variants, in increasing cost:
1. **readout-only probe** — refit only the final linear read-out on frozen features;
2. **full fine-tuning** — unfreeze and train on the calibration prefix.

Both must use only calibration-block data and the identical evaluation query, with early stopping
(if any) selected on calibration data alone.

**Cost.** Modest per arm; the readout probe is very cheap and should be run first.

**Risk, stated plainly.** If fine-tuning gains a lot, the paper must own the trade-off. That is the
honest content of a hardware-constrained method paper, and the trade-off is more defensible when we
report it than when a reviewer computes it. If fine-tuning gains little, it is the single strongest
statement of the paper's thesis available.

**This is the baseline a strong reviewer is most likely to demand.** It is the only proposed
experiment that directly tests the paper's framing premise rather than its results.

---

### G5 — Decoder-level budget curve on H1 — cheap, honest, low headline value

Forward-only on the sealed H-SE5 checkpoint, same query, same sealed references, varying the number
of calibration events. The CPU version is complete and predicts **monotonic degradation with no
plateau** (at 8 events the carrier retains under 3% of the effect), so expect this to document a
floor rather than produce a headline. Worth running for completeness of the dose-response story,
not for a claim.

---

## Priority

| # | Item | Cost | Blocked by | Value |
|---|---|---|---|---|
| 1 | G1 Stage B evaluation | none, in flight | - | control parity |
| 2 | G4 readout-probe variant | small | - | tests the framing premise |
| 3 | G2 Stage A | ~6.3 h | pipeline fix | generalisation |
| 4 | G3 cross-date transfer | forward-only | G2 | patches a printed claim |
| 5 | G4 full fine-tuning | moderate | - | completes the premise test |
| 6 | G5 budget curve | forward-only | - | completeness |

G4's readout probe is deliberately ranked above Stage A: it needs no pipeline fix, it is cheap, and
it addresses the premise on which the entire paper rests.

---

## Operational incident 2026-08-12 — GPU runs killed by session teardown

Stage B was launched at 15:12 into detached tmux sessions and **died at 15:55 with no checkpoints**,
losing about 43 minutes of GPU time on both cards.

**It did not crash.** Neither training log contains an error or traceback; both stop mid
progress-bar at epoch 11-12 of 50. Memory was fine (4 GB of 62 used), no OOM.

**Diagnosis — CORRECTED 16:45.** An initial diagnosis attributed this to agent-session teardown.
That was probably wrong. **A second agent session is working concurrently in this worktree.** At
16:41 it launched two GPU runs of its own (`h1_hse5_lodo_full_19250108` on GPU 0 and
`h1_hse5_lodo_zero5_19250108` on GPU 1, tmux `hse5_date2_full` / `hse5_date2_zero`, launcher parent
PID 27130, unrelated to this session), having just authored a complete H-SE5 date-2 LODO program
(dated data module, dated snapshot builder, dated model module, configs, preflight, launcher,
terminal evaluator, verifier, recompute script, tests).

The far more likely explanation for Stage B's death is therefore **GPU contention between
concurrent agent sessions**, not process-group teardown. The observations remain as recorded — no
error in either log, tmux server and July-era sessions untouched, no OOM — but the cause should be
read as another actor claiming the devices.

**Coordination hazard, now the dominant operational risk.** Two agents are running GPU experiments
in the same worktree with no locking. Before any launch: check `nvidia-smi` **and** `tmux ls` **and**
`pgrep -af src/train.py`, and do not assume idle GPUs will stay idle. A run started without
coordinating can be killed at any point and, under the fixed-epoch no-selection protocol, leaves no
checkpoint and no partial credit.

**Possible duplicated work.** The other session's `build_h1_sparse_event_source_snapshot_dated.py`
and dated source/snapshot modules overlap substantially with the Stage A fork described in G2.
Before resuming G2, diff the two implementations — the other agent's may already be correct, or may
share the same divergence bug documented in G2. Do not build a third.

**Requirement for every future GPU launch.** `tmux new-session -d` alone is **not** sufficient
isolation here. Fully reparent the job away from the agent session's process group, for example:

```
setsid nohup <command> > <logfile> 2>&1 < /dev/null &
```

and verify after launch that the process's parent is `init`/`systemd` rather than a shell belonging
to the agent session. Then confirm survival by checking again several minutes later.

Because a killed run leaves **no checkpoint at all** under the fixed-epoch, no-selection protocol,
there is no partial credit: the entire run is lost. For runs of this length, consider whether
periodic checkpointing is worth enabling as insurance, noting that the protocol forbids using any
intermediate checkpoint for *selection* — it would be for crash recovery only, and any such change
must be predeclared.

## Standing constraints for any GPU run

- `PYTHONNOUSERSITE=1` always, and `PYTHONPATH=<root>/SPINT-main:<root>` — **both**, in that order.
  A launch with only `SPINT-main` fails instantly; this already cost one wasted launch.
- Interpreter `~/miniconda3/envs/spint/bin/python` (Torch 2.5.1 / CUDA 11.8).
- Never modify a sealed file; subclass instead.
- Never delete or overwrite an existing log, checkpoint, or receipt. Use a new run directory.
- Only the 13 public `sub-HumanPitt-held-in-calib` NWBs; never `held-out`, `heldout`, `minival`,
  `formal`, `private`, `evalai`, `test_ecephys`.
- Predeclare gates before launching. A failed gate is a failure; do not relax thresholds or add arms
  afterwards.
- Verify a run reached its fixed terminal epoch before reading any number from it.
- Every published number cites its receipt SHA and an independent verifier.
