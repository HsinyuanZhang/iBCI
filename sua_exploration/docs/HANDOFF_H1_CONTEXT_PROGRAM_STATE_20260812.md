# Handoff: H1 Context Program — State, Blockers, and Remaining Work

**Date:** 2026-08-12, ~15:30 HKT
**Author role:** outgoing agent. Written for a fresh agent taking over.
**Read with:** `HANDOFF_PAPER_EDITS_20260812.md` (paper edits),
`H1_CONTEXT_FULL_UNSEAL_MULTIDATE_GPU_PROTOCOL_20260812.md` (frozen GPU protocol),
`H1_TAGFREE_POSITION_CONTEXT_CARRIER_PROTOCOL_20260812.md` (closed CPU screen),
`H1_SPARSE_MAINLINE_STRENGTHENING_20260812.md` (CPU screen results).

---

## 0. Scope question answered honestly

**Is the remaining work just completion of earlier experiments rather than new design?**

Substantially yes.

| Remaining item | Nature |
|---|---|
| Stage B — separately trained LS/RS, fold-0 | Control rigour for an existing fold-0 result. Gap-fill. |
| Stage A — second date `19250108` | Generalisation of an existing result. Gap-fill. |
| Cross-date carrier transfer | A new *control* on an old, untested claim. Not a new method. |
| Paper edits | Writing. No new experiment. |

Every genuinely new *design* generated in this session has already been tested and closed:

- **Tag-free position context** (drop the event tag, keep `[delta, midpoint]`): terminal
  `STOP_CPU_PCTX_NOT_MATERIAL`. Best candidate `+0.004720` against a `+0.011550` requirement.
- **K-point within-event trajectory** (read more than two position samples per event): no arm passed;
  best paired contrast `+0.0019` against a `+0.010` gate.

**Exactly one untested new design remains.** See Section 5.

---

## 1. Live process state — READ THIS FIRST

### 1.0 STATUS UPDATE 16:40 — Stage B IS DEAD, NOTHING IS RUNNING

**Everything in Section 1.1 below is historical.** Stage B was killed at 15:55 by session teardown,
not by a crash: neither log contains an error, both stop mid progress-bar at epoch 11-12 of 50, the
tmux server and all pre-existing sessions survived, and memory was fine. Only this agent session's
spawned processes died.

**Consequences:** both GPUs idle, no `ctxv2` tmux sessions, **no checkpoints**, so Stage B must be
re-run from scratch. All of its CPU prerequisites remain valid and verified.

**Requirement for the re-run:** `tmux new-session -d` was insufficient isolation. Use
`setsid`/`nohup` to reparent the job away from the launching session's process group, verify the
parent is `init`/`systemd` after launch, and re-check survival several minutes later. Use fresh run
directories — `_v1` and `_v2` are taken and contain nothing usable.

Full incident record and the GPU queue: `HANDOFF_GPU_EXPERIMENT_QUEUE_20260812.md`.

### 1.1 Stage B launch record (HISTORICAL — this run is dead)

| Arm | tmux | GPU | Run directory |
|---|---|---|---|
| Context-LS (label corruption) | `ctxv2_ls` | 0 | `SPINT-main/logs/h1_ctxv2_context_ls_19250101_m4_s42_v2` |
| Context-RS (row corruption) | `ctxv2_rs` | 1 | `SPINT-main/logs/h1_ctxv2_context_rs_19250101_m4_s42_v2` |

Started 15:12, epoch 3/50 at 15:27, ~15-17 it/s, no errors. **Revised ETA ~18:10-18:20**, not the
17:30 the launching agent estimated — it extrapolated from a single-run 22 it/s, but two concurrent
runs with `num_workers=0` contend on CPU dataloading.

Logs: `SPINT-main/pilot_artifacts/h1_ctxv2_stage_b/logs/ctxv2_{ls,rs}_train_launch2.log`

**A watcher is running** (PID 3151491) that polls every 60 s and exits on `STAGEB_DONE`,
`STAGEB_FAILED`, or `STAGEB_TIMEOUT` (3.5 h). It belongs to the outgoing agent's session; a new
agent should start its own or watch the logs directly.

To stop everything: `tmux kill-session -t ctxv2_ls; tmux kill-session -t ctxv2_rs; kill 3151491`

**Ignore the `_v1` run directories and `ctxv2_{ls,rs}_train.log`.** The first launch died instantly
on a `PYTHONPATH` error and those artifacts are dead. Score only `_v2`.

**Do not be alarmed by `train/r2_mean = -1.42e+18` in the logs.** The sealed fold-0 Context run
reports the identical value at epoch 3 and at epoch 49 where it converged correctly. It is a
degenerate per-batch variance denominator in that logged metric. Use `train/loss` (currently
1.46e-5, sealed run was 1.29e-5 at the same epoch).

### 1.2 What to do when Stage B finishes

Evaluate against the **predeclared secondary clause**, protocol Section 5.2: Context Full must
exceed both separately trained arms, both margins positive.

Sealed fold-0 reference values (same-checkpoint interventions, 8,965-window query):

| Arm | pooled R2 |
|---|---:|
| Context Full | 0.516518 |
| same-checkpoint label shuffle | 0.499189 |
| same-checkpoint row shuffle | 0.499152 |

**Expect the separately trained margins to be SMALLER than these.** We measured that
same-checkpoint controls overstate content dependence by roughly 4x (Section 3.3 below). A shrunken
margin is the predicted outcome, not a failure. Only a **sign inversion** fails the clause.

Verify both runs reached fixed epoch 50 before reading any number. Use the strict evaluator pattern
in `SPINT-main/scripts/h1_sparse_event_endpoint_evaluate.py`, with byte-identical query windows
across arms and the query SHA recorded.

---

## 2. Stage A is BLOCKED by a diagnosed numerical divergence

**Do not launch Stage A.** `SPINT-main/scripts/ctxv2_stage_a_launch_v2.sh` exists and its preflight,
config-parity, fidelity-gate, and snapshot receipts all exist, but the fidelity gate is **vacuous**.

### 2.1 Why the gate is vacuous

`SPINT-main/src/data/h1_context_event_source_dated.py` branches on `_is_fold0()` at lines 175, 197,
212 and 279, routing `outer_date="19250101"` to the **sealed** stack. The fidelity gate ran on
fold-0, so it validated sealed code reproducing itself. The dated code path that would actually run
on `19250108` was never exercised by the gate.

The two paths are different implementations throughout:

| Component | fold-0 branch (sealed) | dated branch (would run on 19250108) |
|---|---|---|
| source assets | `build_context_source_assets` | `build_context_source_assets_dated` |
| dataset | `H1ContextSourceDataset` | `H1ContextDatedSourceDataset` |
| **batch sampler** | `H1M4EBPairedBatchSampler` | `H1ContextDatedBatchSampler` |
| manifest | `build_context_manifest` | `build_context_manifest_dated` |

The batch sampler determines batch composition and ordering, so it directly affects the trained
model.

### 2.2 The measured divergence

Running **both** paths on fold-0 with identical inputs:

```
sealed path manifest SHA : c49694d850c426d58c10f3da5271bcb472e9c52d95963d619a7134a48e6adb78  (matches sealed snapshot)
dated  path manifest SHA : db24ac9607dd5e8012bbcc3f69ed761ee362864bc68627821de521d03f8cd85d  (does NOT match)

Fields that differ: carrier_cache_sha256, normalized_cache_sha256, normalizer, normalizer_sha256
Fields that match : the latent map (it is bound to the sealed receipt)
```

So `build_context_source_assets_dated` computes **different carriers and a different normalizer**
from the sealed builder, on the same sessions, with the same latent map. That is a substantive
numerical bug, not a schema cosmetic.

The launching agent found this mismatch and resolved it by adding the fold-0 delegation branch
rather than fixing the dated builder. Its own log: "Fold-0 manifest mismatch comes from the dated
cache schema — I'll fix `source_dated.py` to delegate to the sealed path for `19250101`."

### 2.3 Why this matters scientifically

If Stage A ran as-is and Context minus H-SE5 differed on date 2, it would be impossible to
attribute the difference to the date rather than to the pipeline. That destroys the comparability
the multi-date test exists to establish — precisely the divergence risk recorded in protocol
Section 5.6.

### 2.4 Required fix before Stage A may launch

1. Fix `build_context_source_assets_dated` and the dated normalizer so the **dated path itself**,
   with the fold-0 delegation branch removed or bypassed, reproduces
   `c49694d850c426d58c10f3da5271bcb472e9c52d95963d619a7134a48e6adb78` exactly.
2. Diagnose the four differing fields specifically. Start with the normalizer, since
   `normalized_cache_sha256` and `carrier_cache_sha256` both differ and the normalizer is fitted
   from the cache.
3. Only then re-run `ctxv2_stage_a_fidelity_gate.py`, with the delegation disabled, and require a
   pass. The existing fidelity-gate receipt
   (`CTXV2_STAGE_A_FIDELITY_GATE_v1.json`, SHA `de683119...c180`) must be treated as **not
   establishing what its name claims** and superseded.
4. The `19250108` snapshot already written
   (manifest SHA `7d3a3961926f91568d014af07748d60d5b65c52eb92aaae5d1e53b0cae1320dc`) was produced
   by the unvalidated dated path and must be rebuilt after the fix.

Do not lower the bit-identity requirement.

---

## 3. Completed results this session

All receipts are immutable, mode `0444`. Every screen reproduces the sealed H-SE5 forward-transfer
numbers to exactly `0.0`, confirming they measure the sealed estimator.

### 3.1 CPU screens

| Screen | Terminal status | Key result |
|---|---|---|
| Event-budget sweep | done | Monotonic, **no plateau**. At 8 events the carrier retains under 3% of the effect. H1 has no sparsity headroom. |
| Conditioning sweep | done | Diversity at fixed count is real (11/13 at k=14, p=0.023). Condition number does **not** predict forward transfer. |
| Trial-aligned + eval-boundary | done | The evaluation boundary explains **0.9%** of the M3-M4 gap; support size explains 99.1%. Sealed M3 and M4 are directly comparable. M1 is infeasible (0/13, below the 8-event floor), so the method's own rule sets a 2-trial minimum. |
| Sample-complexity audit | done | Carrier 4.0 obs/param (overdetermined) vs ridge 0.356 (**underdetermined in 13/13**), ratio 11.4x. |
| Tag-free position context | `STOP_CPU_PCTX_NOT_MATERIAL` | No tag-free path exists. The event tag supplies **67%** of Context's gain at M4. |
| K-point within-event | no arm passed | Best `+0.0019` against a `+0.010` gate. H1 events are spatially near-straight (arc/chord 1.023) but temporally non-uniform (speed CV 0.569). |

### 3.2 Cross-recording carrier transfer — changes a paper claim

Receipt `SPINT-main/pilot_artifacts/h1_cross_session_transfer/H1_CROSS_SESSION_CARRIER_TRANSFER_FOLD0_v1.json`,
SHA `ec0d9802d563d73fc6096bbf3abed33523ba35d2ed038c04970954e13ae68f98`.

| Arm | pooled R2 |
|---|---:|
| full | 0.516518 |
| **cross-recording transfer** | **0.519024** |
| row shuffle | 0.499152 |
| zero | 0.484059 |

Swapping carriers between the two fold-0 recordings slightly **improved** performance. Integrity
clean: `full` reproduced sealed `0.516518` to 3.3e-08, query SHA identical across arms, model state
unchanged, zero optimizer and backward calls.

**Scope caveat that must travel with this number.** `ses-19250101T111740` and `ses-19250101T112404`
are timestamps six minutes apart on the same day — consecutive blocks, not different sessions. This
shows the per-block refit earns nothing at six-minute separation. It does **not** test drift across
days, which is the regime the paper's claim concerns.

Consequence: `paper_6pp.tex` lines 363-364 ("because `c_i` is re-fitted every session it cannot
drift like a transferred neural fingerprint") must be deleted or qualified. See
`HANDOFF_PAPER_EDITS_20260812.md` P0.1.

### 3.3 A methodological finding to carry forward

The sealed same-checkpoint within-trial tag shuffle costs 16% of Context's decoder gain; the
measured true tag contribution is 67%. **Same-checkpoint controls understate content dependence by
about 4x.** This is why Stage B's separately trained arms exist, and it is why a shrunken Stage B
margin is expected rather than alarming.

Related: three proxy statistics have each failed to predict forward transfer — design condition
number, retained variance, and retained energy. Do not use any as a selection criterion.

---

## 4. Recommended order for the incoming agent

| # | Action | Blocking? |
|---|---|---|
| 1 | Let Stage B finish (~18:10-18:20), verify epoch 50, evaluate against clause 5.2 | in flight |
| 2 | Apply the paper edits in `HANDOFF_PAPER_EDITS_20260812.md`, P0 first | no |
| 3 | Fix the dated-path divergence (Section 2.4) before touching Stage A | blocks Stage A |
| 4 | Re-run the fidelity gate with delegation disabled; rebuild the `19250108` snapshot | blocks Stage A |
| 5 | Only then consider Stage A GPU launch | — |

Items 1 and 2 are independent of the Stage A blocker and should not wait behind it.

---

## 5. The one untested new design

**Sub-event splitting.** Split each movement event into sub-events, each contributing its own
`(displacement, rate)` observation pair. This multiplies *observations* rather than feature
dimensions, which is the only lever the budget sweep identified as effective (monotonic starvation,
4.0 obs/param, no plateau). It keeps `q=4` and the `[176,5]` carrier, so the decoder interface is
unchanged, and it costs the same acquisition reads as the K-point screen.

Arithmetic: ~20 support events per session becomes ~40 at a 2-way split (8.0 obs/param) or ~80 at a
4-way split (16 obs/param), moving the estimator out of the marginal regime.

**Known risk that may kill it, stated up front.** Because arc/chord is 1.023, sub-displacements
within an event are nearly parallel. Splitting therefore adds observations along nearly the same
direction, varying mainly in magnitude — improving estimation of tuning *gain* but not *direction*,
and the conditioning sweep showed directional diversity is what drives carrier fidelity. Shorter
windows also mean fewer spikes and noisier rates.

Whether this nets positive depends on whether the estimator is limited by sample count (the budget
sweep says yes) or by directional coverage (the conditioning sweep says yes). Both are real, so the
outcome is genuinely uncertain. That makes it worth a cheap CPU screen and **not** worth GPU time in
advance.

---

## 6. Standing constraints

- **Never modify a sealed file.** Everything under `SPINT-main/src/`, `SPINT-main/configs/`,
  `SPINT-main/scripts/` that predates this session is sealed. Import and subclass.
- `PYTHONNOUSERSITE=1` always. Training needs
  `PYTHONPATH=<root>/SPINT-main:<root>` — both, in that order.
- Interpreter `~/miniconda3/envs/spint/bin/python` (Torch 2.5.1 / CUDA 11.8).
- Only the 13 public `sub-HumanPitt-held-in-calib` NWBs. Never `held-out`, `heldout`, `minival`,
  `formal`, `private`, `evalai`, `test_ecephys`.
- Never read `acquisition/OpenLoopKinematicsVelocity` in a sparse-carrier path.
- Do not `git add .` in this dirty multi-agent worktree.
- Predeclared gates are frozen. If an experiment fails its gate, it fails; do not relax thresholds
  or add arms afterwards.
- Disk is at 91% with ~79 GB free. Each run is ~126 MB.
