# Active experiment control board

**Updated:** 2026-08-11 HKT

**Purpose:** live processes, terminalization state, and cleanup queue only
**Scientific authority:**
[`HANDOFF_MAINLINE_CLOSURE_20260811.md`](HANDOFF_MAINLINE_CLOSURE_20260811.md)

This file is intentionally short. Historical queues, proposals, launch incidents, and closed hypotheses belong in
immutable receipts or Git history, not in the live control surface.

## 1. Active mainline — RT sparse endpoint carrier

| field | frozen value / current state |
|---|---|
| matrix | 15 folds × `R-T4d/R-Full/R-Zero4` × seed42 = 45 cells |
| calibration/evaluation | M24, q24, W50, 35 epochs, fresh source fit, one-shot outer target |
| hardware | remote RTX 5070 Ti |
| completed | **38/45 cells; 12/15 complete paired folds** |
| current cell | fold12 `R-Zero4` training; fold12 T4d and Full are closed but remain unpaired until Zero4 closes |
| terminal aggregate | absent; partial values are nonterminal |
| invariants | identical within-fold query identity; target optimizer absent; no target backward; model state unchanged |
| independent verifier | implemented; terminal-verifier/B2 chain retested 2026-08-11 (`29 passed`); execute only at 45/45 |
| follow-on | exact-query 15-fold B2 forward-only companion; no B2 retraining |

Current early evidence from folds0--11:

| contrast | mean | median | drop-largest mean | positive folds |
|---|---:|---:|---:|---:|
| `T4d−Zero4` | **`+0.294121`** | **`+0.358093`** | **`+0.283988`** | **12/12** |
| `T4d−dense Full` | `+0.005379` | `+0.001802` | `−0.001029` | 6/12 |

Arm means are T4d/Full/Zero4 `0.451688/0.446308/0.157567`. Fold11 closed at
`0.3221854512/0.2674641752/0.0806173878`, with matched target `ses-RT-20150316`, query start trial24,
96,288 windows, and common query hash `c840f4be...`. All 12 paired folds have passed the one-shot/no-backprop/
state-immutability checks. Matrix manifest SHA begins `93a1aa35...b3ca4`.

Do not write these values into the paper as terminal, equivalence, or native annotation-cost evidence.

## 2. Prepared RT terminal sequence

When 45/45 closures exist:

1. run the independent Stage-2 terminal verifier;
2. inspect its exact 15×3 grid, provenance, query, no-BP, and recomputed-statistic verdict;
3. if and only if it passes, run all 15 sealed B2 checkpoints forward-only on exact Stage-2 queries;
4. finalize full-15 and prospective folds4--14 B2 comparisons;
5. update the handoff, `CURRENT_RESULTS.md`, and Overleaf with terminal-only numbers.

The B2 implementation is already committed at GitHub commit `ed03215`. On 2026-08-11, the independent Stage-2
verifier, B2 preflight, B2 forward-only terminal evaluator, and paired companion suites were rerun together with
plugin autoload disabled: **29/29 tests passed**. Fold0 uses a
preservation-only imported checkpoint whose SHA begins `078ac6dc...`; the import receipt SHA begins
`5d00985d...`. Historical B2 outer scores must not be mixed with the exact-query forward pass.

## 3. Secondary active work — H1 CI64

| field | current state |
|---|---|
| matrix | five dates × five arms = 25 fixed-e49 source checkpoints |
| completed | **19/25** |
| active | date19250119 `CI64-RS` on GPU1; date19250120 `CI32-FULL` on GPU0 |
| remaining after active cells | four date19250120 arms under the original GPU0 serial runner |
| target evaluation | unopened; no current CI64 accuracy result |
| role | secondary H1 compact-consumer result; not a sparse-label claim and not a mainline blocker |

A proposed 3+2 split of the final date was cancelled before any explicit cell launch. The GPU0 partition parent was
briefly stopped and resumed without touching either active child; the execution receipt records
`CANCELLED_BEFORE_LAUNCH_ORIGINAL_SERIAL_SCHEDULE_RESTORED`. No duplicate date20 run directory was created.

H1 is low-frequency monitored. Root work is focused on RT and documentation closure.

## 4. Closed results relevant to the active decision

| route | terminal result | disposition |
|---|---|---|
| dense RT Full vs B2 | `0.441950` vs `0.145148`, delta `+0.296802`, 15/15 positive | paper evidence; B2 named SPINT-structured same-pipeline reference |
| RT L-D live-activity gain | `G-Full−A0=−0.002068`; `G-Full−G-XLS=+0.006606` | stopped; no additional seed/fold/fusion |
| H1 all-source official | held-out `0.274939±0.127206` vs paper-LR SPINT `0.261492±0.148717` | system-level dense-covariate evidence |
| M1 carrier content | official T4 below Original; matched EMG-AFC4 `−0.00652` | negative boundary; no rescue |

## 5. Documentation closure queue

1. keep `HANDOFF_MAINLINE_CLOSURE_20260811.md` as the sole current handoff;
2. keep README and ROADMAP concise and pointing to that handoff;
3. keep the already-compressed `CURRENT_RESULTS.md` as the dataset/claim ledger and replace only its explicitly
   nonterminal RT block after terminal verification;
4. mirror the current 12-fold paired aggregate into full and six-page Overleaf only as an explicit nonterminal
   placeholder; replace it after terminal verification and B2 companion completion;
5. demote older `HANDOFF_*`, `AGENT_BRIEF_*`, and dated reviews to historical/archive status.

The terminal RT paper update has a frozen replacement map. A user-requested 12-fold partial score may appear only
inside an explicit `NONTERMINAL PLACEHOLDER`; it is not a claim and must be replaced at closure:

- full paper abstract in `bci_paper_overleaf/main.tex`;
- contribution paragraph in `bci_paper_overleaf/sections/01_introduction.tex`;
- endpoint-derived T4d and supervision-density scope in `bci_paper_overleaf/sections/03_methodology.tex`;
- RT dataset/annotation wording, sparse-status text, and the separate Stage-2 subsection/table in
  `bci_paper_overleaf/sections/04_experiments.tex`;
- RT scope language in `bci_paper_overleaf/sections/05_conclusion.tex`;
- the corresponding abstract, RT subsection/table, pending-evidence list, and conclusion in
  `bci_paper_overleaf/paper_6pp.tex`;
- RT interval/supervision captions in `bci_paper_overleaf/figures/scripts/figure_snippets.tex`.

The dense Full/B2 result remains a separate historical estimand. The sparse T4d update must report matched
T4d--dense Full, T4d--Zero4, and T4d--exact-query B2 statistics without replacing or silently relabelling the dense
matrix. Preserve the Stage0B/Stage1 constructibility result as a separate layer, remove only the terminal-accuracy
`pending` language, and retain the caveat that endpoint-derived directions do not restore a native annotation-cost
claim. Any `single-seed RT` or `RT $\pm2$SE` sentence must name whether it describes historical dense Stage-R or
sparse Stage-2. Broad `sparse trial- or event-level supervision` language must distinguish endpoint-derived RT
evidence from the negative H1 sparse-scalar boundary; historical `+0.2799/+0.2775` continuous-velocity contrasts
remain dense Full--B4/Zero4 results.

The 12-fold placeholder was mirrored to both Overleaf sources and pushed at Overleaf commit `8e703ac`; it must be
replaced, not supplemented, when the terminal bundle and B2 companion close.

No active protocol/script/test/receipt-referenced document may be deleted merely because its conclusion is old.

## 6. Post-terminal cleanup and GitHub queue

Cleanup begins after RT runners/watchers/terminalizers exit. H1 paths remain protected while H1 runs.

- recoverable candidate: 42 periodic `last.ckpt` aliases that are byte-identical to retained best checkpoints,
  totaling 2,665,007,268 bytes (about 2.482 GiB);
- before moving them, rerun process/GPU/tmux/`lsof` and path-reference checks;
- write a quarantine manifest containing source path, retained peer, size, and SHA;
- retain all immutable receipts, terminal aggregates, query digests, and canonical best checkpoints;
- do not commit generated JSON results, checkpoints, logs, `results/`, or `pilot_artifacts/`;
- run focused tests and `git diff --check`;
- stage exact reviewed paths only; never use `git add .`;
- push and verify the remote commit SHA.

Five unreferenced CPU evidence briefs have passed the fresh basename/stem scan and were moved without content
changes from the active docs root to `docs_archive/20260809_cpu_briefs/`. The archive manifest records their
original SHA-256 values and supersession boundary. Historical `git_state.txt` snapshots retain former path strings,
but no current document, script, test, protocol, or paper source depends on them. Further old documents remain in
place until the same reference audit proves they are safe to archive.

## 7. Stop rules

- No new T4/RT architecture or label-budget arms after viewing these target results.
- No CI64-driven H64 escalation in this closure.
- No quantization work before the FP32 scientific result is frozen.
- No M1 rescue, H1 carrier redesign, deeper FiLM, attention, fixed-K, or label-free carrier experiments.
- Any missing provenance/query/no-BP invariant causes fail-closed terminalization, not post-hoc repair of the score.
