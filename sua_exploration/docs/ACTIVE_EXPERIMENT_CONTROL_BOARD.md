# Active experiment control board

**Updated:** 2026-08-11 HKT

**Purpose:** live processes, terminalization state, and cleanup state only
**Scientific authority:**
[`HANDOFF_MAINLINE_CLOSURE_20260811.md`](HANDOFF_MAINLINE_CLOSURE_20260811.md)

**Live state:** no RT, H1, or A2 process is active. The RT/H1 accuracy programs and corrected CPU-only A2a/A2b-v2
supervision-density controls are terminal. The stale dead H1 Phase-2 preflight tmux session was removed during the
final live-state audit.

This file is intentionally short. Historical queues, proposals, launch incidents, and closed hypotheses belong in
immutable receipts or Git history, not in the live control surface.

## 1. Terminal mainline — RT sparse endpoint carrier

| field | frozen value / current state |
|---|---|
| matrix | 15 folds × `R-T4d/R-Full/R-Zero4` × seed42 = 45 cells |
| calibration/evaluation | M24, q24, W50, 35 epochs, fresh source fit, one-shot outer target |
| hardware | completed across remote RTX 5070 Ti and local RTX 3090 execution |
| completed | **45/45 cells; 15/15 complete paired folds** |
| current cell | none; Stage-2 execution and B2 exact-query companion are closed |
| terminal aggregate | `PASS_MATRIX_TERMINAL`; SHA `bb2806953...` |
| invariants | identical within-fold query identity; target optimizer absent; no target backward; model state unchanged |
| independent verifier | `PASS_INDEPENDENT_TERMINAL_VERIFICATION_READ_ONLY` |
| follow-on | exact-query B2 companion complete; final receipt SHA `c36ec0e3...` |

Terminal evidence from all 15 folds:

| contrast | mean | median | drop-largest mean | positive folds |
|---|---:|---:|---:|---:|
| `T4d−Zero4` | **`+0.268905`** | **`+0.241568`** | **`+0.259142`** | **15/15** |
| `T4d−B2-D1024` | **`+0.303028`** | **`+0.292102`** | **`+0.291237`** | **15/15** |

Sparse-mainline arm means are T4d/Zero4/B2 `0.448176/0.179272/0.145148`. T4d--Zero4 and
T4d--B2 are positive in every fold; both exact two-sided sign tests are `p=6.1035e-5`. The prospective
folds4--14 B2 gate also passes with mean/median `+0.311178/+0.304175` and 11/11 positive. Matrix manifest SHA is
`93a1aa3549b844c399ab1cc2b9bddb1d93ee2070b51c91e80d60875cee4b3ca4`; aggregate SHA is
`bb2806953e979180c408fb55744534be6fa470d4144f210cc50917a9b1006b7d`; B2 final SHA is
`c36ec0e31ed913ed4e8077f9a4d9d634d53529ce037ad06af1f48d279b16820e`.

These values may be written as terminal development evidence. Dense Full is separate context only: its mean is
`0.445189` versus T4d `0.448176` (average difference `+0.002987`), without a superiority, equivalence, or
non-inferiority claim. Do not
describe endpoint-derived RT labels as native annotation-cost evidence.

## 2. Completed RT terminal sequence

The frozen sequence completed in order:

1. the independent Stage-2 verifier passed from a complete isolated closure with explicit path mapping;
2. its exact 15x3 grid, provenance, query, no-BP, and recomputed statistics passed;
3. a new B2 output root evaluated all 15 sealed checkpoints exactly once on the Stage-2 queries;
4. full-15 and prospective folds4--14 B2 summaries both closed and passed their stated checks;
5. mainline and Overleaf RT terminal replacement completed and was cross-checked against the terminal receipts;
6. H1 CI64 terminal values and the stop-H64 decision were propagated after independent verification.

The B2 implementation is already committed at GitHub commit `ed03215`. On 2026-08-11, the independent Stage-2
verifier, B2 preflight, B2 forward-only terminal evaluator, and paired companion suites were rerun together with
plugin autoload disabled: **31/31 collected cases passed across 29 test functions**. This is the historical focused
terminal suite, not a claim that the entire current root-worktree candidate is green. Fold0 uses a
preservation-only imported checkpoint whose SHA begins `078ac6dc...`; the import receipt SHA begins
`5d00985d...`. Historical B2 outer scores must not be mixed with the exact-query forward pass.

## 3. Completed secondary work — H1 CI64

| field | current state |
|---|---|
| matrix | five dates × five arms = 25 fixed-e49 source checkpoints |
| completed | **25/25 checkpoints and 5/5 one-shot held-date evaluations** |
| active | none |
| terminal result | CI64-FULL−CI32-FULL `−0.020130`, 2/5 dates positive |
| controls | FULL−C0 `+0.037980` (4/5); FULL−LS `+0.026249` (4/5); FULL−RS `+0.033704` (5/5) |
| verifier | `PASS_H1_CARRIERID_DATE_LODO_CI_FIVEDATE_TERMINAL_VERIFIED_V2`; SHA `29d6a5f6...` |
| decision | stop H64 escalation; no consumer-width improvement |
| role | secondary H1 compact-consumer result; not a sparse-label claim and not a mainline blocker |

The aggregate SHA is `534e7d4f2559e03d3cf89a5f7f0b9641172479664ce8095f2d0866ca1e6a7f53`. The final verifier
checked the exact 5×5 grid, fixed checkpoint/config lineage, shared queries, unchanged state, and zero target
backpropagation. CI64 retains positive content and attachment contrasts, but doubling the interface width is
negative on average and fails both the historical width condition and the predeclared `+0.03` practical gate.

## 4. Closed results and scope boundaries

| route | terminal result | disposition |
|---|---|---|
| dense RT Full vs B2 | `0.441950` vs `0.145148`, delta `+0.296802`, 15/15 positive | paper evidence; B2 named SPINT-structured same-pipeline reference |
| RT L-D live-activity gain | `G-Full−A0=−0.002068`; `G-Full−G-XLS=+0.006606` | stopped; no additional seed/fold/fusion |
| H1 all-source official | held-out `0.274939±0.127206` vs paper-LR SPINT `0.261492±0.148717` | system-level dense-covariate evidence |
| M1 carrier content | official T4 below Original; matched EMG-AFC4 `−0.00652` | negative boundary; no rescue |

## 5. Documentation maintenance state

1. keep `HANDOFF_MAINLINE_CLOSURE_20260811.md` as the sole current handoff;
2. keep README and ROADMAP concise and pointing to that handoff;
3. keep the already-compressed `CURRENT_RESULTS.md` as the dataset/claim ledger; its RT block now carries terminal values;
4. keep the completed terminal sparse-RT replacement consistent across both Overleaf sources;
5. keep the terminal H1 CI64 width result in its secondary compact-consumer role;
6. demote older `HANDOFF_*`, `AGENT_BRIEF_*`, and dated reviews to historical/archive status.

All six items are complete for the RT/H1 scope. The six-page H1 organizer-held wording now states only that the
organizer-reported mean is higher, and the matched H1 content figure includes its five-date bootstrap interval;
Overleaf commit `cb191fd` is synchronized with its remote.

The terminal RT paper update has been applied at the following locations:

- full paper abstract in `bci_paper_overleaf/main.tex`;
- contribution paragraph in `bci_paper_overleaf/sections/01_introduction.tex`;
- endpoint-derived T4d and supervision-density scope in `bci_paper_overleaf/sections/03_methodology.tex`;
- RT dataset/annotation wording, sparse-status text, and the separate Stage-2 subsection/table in
  `bci_paper_overleaf/sections/04_experiments.tex`;
- RT scope language in `bci_paper_overleaf/sections/05_conclusion.tex`;
- the corresponding abstract, RT subsection/table, pending-evidence list, and conclusion in
  `bci_paper_overleaf/paper_6pp.tex`;
- RT interval/supervision captions in `bci_paper_overleaf/figures/scripts/figure_snippets.tex`.

The dense Full/B2 result remains a separate historical estimand. The sparse T4d mainline reports
T4d--Zero4 and T4d--exact-query B2; its comparison with dense Full is retained only as a separate arm-mean context
statement, without fold-wise detail or a superiority, equivalence, or non-inferiority claim. Preserve the
Stage0B/Stage1 constructibility result as a separate layer, remove only the terminal-accuracy
`pending` language, and retain the caveat that endpoint-derived directions do not restore a native annotation-cost
claim. Any `single-seed RT` or `RT $\pm2$SE` sentence must name whether it describes historical dense Stage-R or
sparse Stage-2. Broad `sparse trial- or event-level supervision` language must distinguish endpoint-derived RT
evidence from the negative H1 sparse-scalar boundary; historical `+0.2799/+0.2775` continuous-velocity contrasts
remain dense Full--B4/Zero4 results.

The former 12/14-fold placeholders were removed when the terminal bundle and B2 companion closed; do not
reintroduce a partial-results table beside the terminal table.

No active protocol/script/test/receipt-referenced document may be deleted merely because its conclusion is old.

## 6. Post-terminal cleanup and GitHub state

RT and H1 terminal runners have exited. A fresh process, handle, path-reference, size, and SHA audit found 60
`periodic_ckpt/last.ckpt` aliases under the completed RT-R4 budget-response root that were byte-identical to their
same-run retained `best_ckpt/last.ckpt` peers. All 60 periodic paths were preserved as hardlinks to those peers,
reclaiming 3,886,714,880 allocated bytes (about 3.620 GiB). The pre/post audit, every path pair, SHA, mode, inode
transition, and recovery rule are recorded in
`sua_exploration/manifests/rt_r4_periodic_last_hardlink_dedup_20260811.json`.

A separate audit of the terminal Stage-2 source root then proved 14 same-run `last.ckpt` aliases byte-identical to
retained epoch checkpoints and unreferenced by receipts. Those paths were converted to hardlinks, preserving every
path, SHA, and selected-epoch receipt binding while reclaiming `906,891,264` allocated bytes. Manifest:
`sua_exploration/manifests/rt_terminal_stage2_last_ckpt_hardlink_dedup_20260811.json`, SHA
`e68dd35da75ebde415dcf8af7066f53a217677386d62d6387118dc10d8c949bd`. Source/canonical result bundles, split
manifests, selected epochs, and unique logs remain untouched.

- retain all immutable receipts, terminal aggregates, query digests, and canonical best checkpoints;
- do not commit generated JSON results, checkpoints, logs, `results/`, or `pilot_artifacts/`;
- run focused tests and `git diff --check`;
- stage exact reviewed paths only; never use `git add .`;
- push and verify the remote commit SHA.

Five unreferenced CPU evidence briefs passed the fresh basename/stem scan and remain available in the local-only,
ignored `docs_archive/20260809_cpu_briefs/`; their original SHA-256 values are preserved in its local manifest and
in Git history. No current document, script, test, protocol, or paper source depends on them. Further old documents
remain tracked until the same reference audit proves they are safe to remove from the public repository.

On 2026-08-11, a second read-only basename/stem audit cleared 16 obsolete handoffs, seven agent briefs, and the old
workspace-hygiene report for recoverable local archival. They were moved to ignored
`docs_archive/20260811_retired_handoffs/` with an archive manifest. Seven handoff-named files remain active: the
current mainline handoff, the audited trial-ridge handoff, and five historical files still bound by scripts or
receipts. No active protocol, terminal artifact, or paper source was moved.

## 7. Stop rules

- No new T4/RT architecture or label-budget arms after viewing these target results.
- No CI64-driven H64 escalation in this closure.
- Quantization is deferred and is not authorized by this closure board.
- No M1 rescue, H1 carrier redesign, deeper FiLM, attention, fixed-K, or label-free carrier experiments.
- Any missing provenance/query/no-BP invariant causes fail-closed terminalization, not post-hoc repair of the score.
