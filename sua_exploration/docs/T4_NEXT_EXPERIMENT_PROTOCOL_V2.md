# T4 next-experiment protocol v2

**Status:** root-reviewed design; CPU preflight required; no GPU or formal-test authorization

**Updated:** 2026-08-02

**Scope:** DANDI 000688 sub-C/CO source and reused-development evidence only

**Design origin:** Terra subagent draft, independently reviewed and narrowed by root

This protocol replaces the proposed six-experiment list as the execution contract for the next
SUA/T4 round. The original proposal had useful hypotheses, but it mixed three incompatible
substrates, treated several controls as optional, and would have allowed estimator or co-training
runs before establishing what information in T4 actually matters.

This document does **not** authorize a GPU launch. A machine-readable CPU preflight receipt must
first bind the artifacts, scorer boundary, normalizer, teacher, manifests, code, seeds, costs, and
data-access scope described below. The six formal SUA sessions remain sealed.

## 1. Decision summary

The next program has three stages:

1. **A — M30 component attribution:** determine whether the existing T4 gain comes from pure
   phase `[cos(phi),sin(phi)]`, amplitude-weighted coordinates `[a,c]`, modulation and baseline
   `[m,b]`, baseline `b`, or their combination. This is the only unconditional new scientific
   experiment.
2. **B — one estimator upgrade:** eligible only if A shows that direction coordinates have an
   independent decoder contribution. At most one source-only CPU-selected estimator may reach
   GPU.
3. **C — paired SUA/pseudo-MUA co-training:** a separate shared-weight experiment on the existing
   bridge substrate. It begins with `lambda=0`; consistency is permitted only if shared weights
   are already non-inferior in both views.

No new FiLM, adapter, cross-attention, dynamic-weight, electrode-table, waveform/SNR, K4, D4,
R10/B20, or other network-side fusion branch is part of this program. Those families are either
already stopped or answer a different question.

## 2. Evidence and data locks

Permitted evidence:

- the 27 source-training sessions in the strict sub-C/CO manifest;
- the existing six reused-development sessions;
- existing artifacts that pass exact hash and scorer-trace qualification.

Prohibited actions:

- loading or evaluating the six formal SUA sessions;
- using formal-session results for selection, normalization, debugging, or reruns;
- using M1, M2, EvalAI, native-MUA, D4, K4, R10, or B20 evidence to select an SUA arm;
- changing an arm, threshold, seed set, score window, label budget, or loss after reading the
  relevant development result;
- calling reused-development evidence a formal held-out confirmation.

The only permitted claim scope for A--C is:

> DANDI 000688 sub-C/CO reused-development evidence; formal SUA sessions unopened.

## 3. Substrate audit and hard blockers

### 3.1 Substrate A: strict M30 scientific mainline

Component attribution is fixed to:

```text
result root: sua_exploration/results/sua_spint_t4_mainline_fp32_v1/
signal: sorted SUA
split: 27 source train / 6 reused development / 6 formal sealed
activity support: chronological rewarded/datamodule-usable trials [0:30)
T4 label/rate support: the same usable trials [0:30)
score trials: usable trials [30:end]
training: exactly 12 epochs, no early stopping
reported score: arithmetic mean of epochs 5--12
seeds: 42, 43, 44
```

The existing aggregate is internally consistent and reports:

| Arm | Mean R2 | Role |
|---|---:|---|
| `B0` | 0.236417 | historical side-width-0 SPINT control |
| `T4` | 0.574976 | full `[a,c,m,b]` descriptor |
| `TS4` | 0.284528 | full-row attachment shuffle |

Existing paired effects are `T4-B0=+0.338559` and `T4-TS4=+0.290448`, with 6/6 session
means and 3/3 seed means positive. These establish that the M30 substrate has a strong content
signal. `B0` is **not** a parameter-matched substitute for the new `Z4` arm because its side
width is zero.

The shared T4/TS4 train-only normalizer SHA-256 is:

```text
293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0
```

The teacher and manifest hashes shared by the nine existing artifacts are:

```text
teacher:  9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d
manifest: 4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9
```

Exact artifact hashes that the preflight must recheck:

| Arm | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| `B0` | `24477be67d6855ccaadaf64f2651e949e16a2ff972afd821d881a2f6d3f325fd` | `8cd4872c6419145a8b5c349207d42d7868ae80b4c72e607d33a00e9fd98031ee` | `415dde1d40106ccb0d5481fa413dda123bc1e99f9a3a2464c48475befc0a5f90` |
| `T4` | `b8f659a46ad55eea766cbad1be70e1cc99df4c3c4c6c38863f2a5a5ee3104148` | `e18a52a750b44f426ba1e39f3a8f791806f53228e5a8a69b73a491178b8a09b4` | `704f5a40bb07e53dc3267d4ed70e434255c84cb9e9b60dcfcb8dd7dcecac3e64` |
| `TS4` | `0178e384eb8976b931b0fcc47ce53354e84500a65672ee51fd4d9c19ab41c841` | `ca146f795aa342560a1220c7a6e3218f4370f10493bbbb76f7e808a1591936ba` | `e697e870d40efe5357b9c3ea410eb286fdd2d537e0134007b2b33f19d6641f7d` |

These values document the current files; they do not replace the launch-time receipt.

The corrected v2 CPU preflight has now passed and supersedes its conservative v1 fail-closed
receipt. It reconstructs exact original NWB trial-table indices for all six reused-development
sessions, confirms that the historical runners explicitly passed `--pool_size 30`, and proves an
empty fit/scored intersection:

```text
sua_exploration/results/t4_m30_experiment_a_cpu_preflight_v2_20260802/receipt.json
SHA-256 d78097203d5c46b26b0ad251164607dadb78886230aea23f9ba55230148b240e
```

The v1 audit had mistaken the current evaluator's default `FIXED_POOL_SIZE=50` for an enforced
value and treated `chronological_rewarded_trials` as different from the actual T4 support
contract. Root review corrected both points without overwriting v1. The M30 reference substrate
is eligible for descriptor-contract implementation; this receipt does not authorize training.

### 3.2 Resolved audit: selected `T4@50` score boundary

The selected-architecture receipt states `evaluation_start_trial=50`, while the selected leaf
results use the compact name `first/n=30/pool=50` and generic run metadata describes training
evaluation after `calibration_n_trials=30`. This initially created a real metadata ambiguity.

The CPU scorer audit has now reconstructed the actual original NWB trial-table indices for all
six reused-development sessions and all three selected seeds. It binds the selection receipt,
leaf artifacts, run metadata, launch logs, strict manifest, current scorer path, and T4 feature
path. The resolved semantics are:

```text
activity forward support: usable chronological trials [0:30)
T4 label/rate fit:        usable chronological trials [0:50)
scored validation:        usable chronological trials [50:end]
trials 30:49 scored:      false
T4-fit/scored overlap:    false
```

Receipt:

```text
sua_exploration/results/sua_t4_m50_scorer_boundary_audit_v1_20260802/audit.json
SHA-256 fe8b8391267dce18b44e9107ff35f2a285e8d52c1180b8961e3022aa2fe56836
```

The M50 accuracy anchors are therefore no longer quarantined on score-overlap grounds. They
remain reused-development evidence, not formal SUA confirmation, and they are not substituted
for A's separately frozen M30 substrate. This audit clears only the boundary question; it does
not independently authorize or validate INT8.

### 3.3 Substrate C blocker: bridge budget semantics

The completed SUA/pseudo-MUA bridge uses three distinct quantities:

```text
forward activity support Q = 30
T4 label/rate pool = 50
source training metadata calibration_n_trials = 10
```

It must therefore be named:

> `paired-view bridge-Q30/T4-50-source10`

Before C starts, CPU provenance must verify that the `source10`, forward-30, label-pool-50, and
score-window semantics are real and compatible rather than stale or ambiguous metadata. This
blocks C only; it does not block A.

The first C preflight resolved the budgets and verified score start 50, separate SUA/pseudo-MUA
normalizers, electrode pooling, electrode-rate T4 refitting, and cache isolation. It nevertheless
returned `blocked_fail_closed`: the 18 historical separate-model artifacts did not pin the
historical scorer/feature source bytes, and their training metadata recorded neither strict
manifest path nor manifest SHA. C1 cannot begin until either a historical provenance bridge is
accepted or the separate references are rebuilt under one frozen current-source receipt.

```text
sua_exploration/results/t4_paired_view_c1_preflight_v1/receipt.json
SHA-256 5ba52a393cd53c6e2d3c968986ab37197a31abc643a1b744f4afb1ca941aa76c
```

## 4. Experiment A — M30 T4 component attribution

### 4.1 Scientific question

For each unit $i$, ordinary T4 is:

\[
T4_i=[a_i,c_i,m_i,b_i], \qquad m_i=\sqrt{a_i^2+c_i^2}.
\]

The experiment asks whether decoder gain requires pure preferred-direction phase, whether it
requires the amplitude-weighted directional coordinates `[a,c]`, whether the rotation-invariant
rate quantities `[m,b]` are sufficient, or whether baseline `b` alone carries the effect. This
distinction matters because `[a,c]=[m*cos(phi),m*sin(phi)]` still contains modulation amplitude;
`AC4` alone cannot prove that phase is useful. The experiment changes only the contents of the
existing four-dimensional side input and does not introduce a new fusion mechanism.

### 4.2 Common representation contract

Every new arm uses the existing B3S architecture and `side_dim=4`. Full ordinary T4 is first
standardized with the source-only T4 normalizer. Masks are then applied in standardized space:

\[
\widetilde T4_i=[\tilde a_i,\tilde c_i,\tilde m_i,\tilde b_i].
\]

An exact zero therefore means the source-normalized feature mean, not zero spikes or zero raw
rate. No masked ordinary-T4 arm may fit its own zero-variance normalizer or alter the side width.

`PH4` is the one representation with different raw semantics. Define

\[
p_i=
\begin{cases}
[0,0], & m_i=0,\\
[a_i/m_i,c_i/m_i], & m_i>0.
\end{cases}
\]

Its two phase columns use one separately hashed source-only phase normalizer, then are zero-padded
to width four. No epsilon or modulation threshold may be tuned; the receipt reports the exact
count of `m=0` rows. This arm is needed to isolate phase and is not compared as if it shared the
ordinary-T4 coordinate scale.

The following are fixed across arms: teacher, source/validation split and order, architecture,
optimizer, loss, seed, 12-epoch rule, epoch-5--12 scorer, activity support, label budget,
normalizer, and task targets.

### 4.3 Frozen arms

| Arm | Four-dimensional B3S side input | Purpose |
|---|---|---|
| `T4` | `[ã, ĉ, m̃, b̃]` | existing complete reference |
| `TS4` | complete normalized T4-row permutation | existing correct-row attachment control |
| `Z4` | `[0, 0, 0, 0]` | parameter-matched no-descriptor absolute control |
| `PH4` | `[cos(phi) normalized, sin(phi) normalized, 0, 0]` | pure preferred-direction phase without modulation amplitude |
| `AC4` | `[ã, ĉ, 0, 0]` | directional first-harmonic coefficients |
| `MB4` | `[0, 0, m̃, b̃]` | rotation-invariant modulation plus baseline |
| `B4` | `[0, 0, 0, b̃]` | baseline-rate-only carrier |
| `LS4` | `[ã_LS, ĉ_LS, m̃_LS, b̃_aligned]` | target-direction association control |

`LS4` is generated within each session and seed as follows:

1. keep activity, spike rates, task targets, trial order, and the `[0:30)` support fixed;
2. deterministically permute only `target_dir` across those 30 support trials;
3. refit `a_LS,c_LS` and compute `m_LS=sqrt(a_LS^2+c_LS^2)`;
4. copy the ordinary aligned `b` exactly rather than refitting the intercept;
5. apply the same full-T4 source normalizer.

Copying `b` prevents an imbalanced direction design from turning label shuffle into an accidental
baseline-rate intervention. `LS4` and `TS4` are not interchangeable: LS4 tests the
direction--rate association, while TS4 tests correct descriptor-to-unit attachment.

### 4.4 Fixed run matrix

Because the M30 reference preflight has passed, the next model-side prerequisite is descriptor
implementation plus its immutable invariant/cost receipt. The eventual training matrix remains:

```text
Z4, PH4, AC4, MB4, B4, LS4 x seeds 42, 43, 44 = 18 runs
```

Existing T4 and TS4 may be reused only after exact qualification. Seed 42 is not an arm-selection
screen: all six new arms run for all three seeds or none run. Each run uses a new exclusive
directory and fails closed if any prior checkpoint, event, or result is present; no overwrite or
resume is allowed.

### 4.5 Statistics and three-state decisions

For arm contrast $A-B$, define a paired delta for development session $s$ and training seed
$r$:

\[
d_{s,r}=R^2_{A,s,r}-R^2_{B,s,r}.
\]

The session is the biological replication unit; three seeds are training replicates, not 18
independent subjects. Every aggregate must report:

- all 18 paired deltas;
- six cross-seed session means and three cross-session seed means;
- the overall paired mean;
- paired seed-mean SE and `MDE_2SE=2*SE`;
- a hierarchical seed/session bootstrap interval;
- exact paired Wilcoxon on the six session means;
- per-epoch and per-seed dispersion;
- paired versus unpaired SE and implied seed correlation.

A contrast is:

- **effective** only if mean delta is at least `+0.03`, at least 5/6 session means are positive,
  3/3 seed means are positive, the paired two-SE lower bound is above zero, and the hierarchical
  interval lower bound is above zero;
- **ineffective for a practical +0.03 effect** if `mean + 2*SE < +0.03`;
- **indeterminate** otherwise.

A component is “sufficient” only if both conditions hold:

```text
lower paired bound(component - T4) >= -0.03
component - Z4 is effective
```

Point-estimate proximity alone is not non-inferiority.

### 4.6 Conditional component row shuffles

After the full A aggregate is frozen, each `C` in `{PH4, AC4, MB4, B4}` independently triggers
`C-RS4 x 3 seeds` only when:

1. `C-T4` passes the `-0.03` non-inferiority lower-bound rule;
2. `C-Z4` is effective;
3. provenance, normalization, and scorer audits remain clean.

If several components qualify, all qualifying shuffles run; a winner may not be chosen after
viewing the aggregate. The maximum extension is 12 runs.

### 4.7 Predeclared interpretation exits

| Result | Required interpretation | Next action |
|---|---|---|
| `MB4` sufficient, `B4` not sufficient | modulation depth plus baseline carrier | stop directional-mechanism claim; optimize only rate/modulation estimation if justified |
| `B4` sufficient | baseline-rate carrier | stop direction and harmonic work |
| `PH4-Z4` effective and `T4-MB4` effective | pure phase is useful and direction adds beyond `[m,b]` | direction mechanism is supported; B may proceed if all other gates pass |
| `AC4` effective but `PH4` not effective | amplitude-weighted coefficients help, but pure phase is unproven | use “amplitude-weighted coefficient carrier”; do not claim preferred direction alone |
| `AC4-Z4` effective and `T4-MB4` effective | direction coordinates have independent value | B may enter source-only CPU selection |
| `T4-LS4` positive but `MB4` sufficient | label association affects representation, but direction coordinates are not necessary | do not enter B |
| all components unstable relative to `Z4` | the full-T4 effect is not decomposable at current precision | stop component extrapolation |
| existing M30 reference fails qualification | invalid substrate | no GPU run |

If `[m,b]` or `b` is sufficient, the paper and filenames must stop calling the retained carrier a
“movement-aligned directional signature.” Naming must follow the surviving information.

## 5. Experiment B — one conditional estimator upgrade

### 5.1 Entrance gate

B is eligible only if A jointly establishes:

- `T4-Z4` effective;
- `PH4-Z4` reports a clean, non-leaking phase-specific result; a negative result does not by
  itself veto an amplitude-weighted `[a,c]` estimator, but forbids a pure-phase claim;
- `AC4-Z4` effective;
- `T4-MB4` effective;
- `T4-LS4` positive content evidence;
- no component-row-shuffle or provenance failure.

If `MB4` is non-inferior or the directional component lacks independent value, B stops
permanently. A better coefficient proxy alone is not enough: the existing W3 shrinkage improved
the source future-rate proxy in 27/27 sessions, but gave only `+0.000753 R2` versus ordinary
T4@15 and remained `-0.058089 R2` below its historical M50 reference.

### 5.2 CPU candidates and source-only selection

At most these three estimator families may enter a nested 27-source-session LOSO audit:

1. hierarchical empirical Bayes/ridge with a prior demonstrably different from W3; `a,c` use a
   zero-mean rotationally symmetric prior or learned covariance, while `b` may use a source-only
   nonzero prior;
2. joint first- and second-harmonic fit with design
   `[1, cos(theta), sin(theta), cos(2theta), sin(2theta)]`, while output to the decoder remains
   four-dimensional `[a1,c1,sqrt(a1^2+c1^2),b]`; the second harmonic is a nuisance term, not a
   wider side input;
3. Poisson log-link GLM with the same four-dimensional output. It must be described as
   decoder-gradient-free iterative calibration, not closed-form, and must report IRLS/Newton
   iterations, convergence, failures, latency, and workspace.

Every outer fold fits priors and hyperparameters on the other 26 source sessions, uses the held
out source session's first 30 trials as support, and evaluates only strictly later source trials.
No development or formal result enters selection.

The frozen CPU entrance thresholds are:

```text
mean prospective predictive-deviance ratio <= 0.98
direction-coefficient split-half reliability gain >= +0.02
both quantities non-worse in at least 20/27 outer folds
no increase in rank-deficient or nonconverged-unit proportion
no leakage or cost-receipt failure
```

Tie-break order is lower predictive deviance, then lower calibration MAC, then smaller persistent
state. A remaining tie produces no GPU winner.

### 5.3 One GPU winner and stop

If exactly one estimator `E*` passes, the complete matrix is:

```text
ordinary T4: reuse after qualification
Z4: reuse after A qualification
E* aligned x 3 seeds
E*-LS4 x 3 seeds
E*-TS4 x 3 seeds
```

`E*` is decoder-effective only if `E*-T4` is effective, both label-association and row-attachment
controls remain positive, all M30 boundaries match, and the implementation receipt confirms no
decoder/encoder backpropagation during calibration. After this one GPU round, penalties,
harmonic rank, link function, and iteration cap may not be retuned.

## 6. Experiment C — paired SUA/pseudo-MUA co-training

### 6.1 Purpose and invariants

C asks whether one weight set can support both deterministic signal views without material loss.
It does not claim pseudo-MUA is real threshold-crossing MUA and does not change inference
architecture. SUA and pseudo-MUA retain separate train-only normalizers and feature-cache
namespaces; electrode T4 is refit from pooled electrode rates rather than averaged from unit T4.

### 6.2 Phase C1: shared weights, no consistency loss

Each optimizer step contains paired SUA and pseudo-MUA microbatches from the same source session,
behavior target, and temporal indices. The loss is:

\[
L_{C1}=0.5L_{task}(SUA)+0.5L_{task}(pseudoMUA), \qquad \lambda=0.
\]

This paired average preserves exposure for both views; random single-view batches that halve each
view's exposure are not an equivalent control.

The fixed matrix is:

```text
MV0-T4 x seeds 42,43,44
MV0-TS4 x seeds 42,43,44
```

Six shared-model training runs produce 12 view-specific evaluation records. C1 passes only when:

- the paired lower bound versus the corresponding separately trained T4 is at least `-0.03` in
  **both** views;
- `MV0-T4 - MV0-TS4` retains positive content evidence in both views;
- the upper paired bound on the increase in absolute SUA/pseudo-MUA performance gap is at most
  `+0.03`.

Failure of any condition terminates C; no wider model, view-specific head, extra epoch, or fusion
rescue is permitted.

### 6.3 Phase C2: one-way consistency, conditional on C1

Only after C1 passes may C2 add:

\[
L_{C2}=0.5L_{task}(SUA)+0.5L_{task}(pseudoMUA)
+\lambda\|\hat y_{pseudoMUA}-stopgrad(\hat y_{SUA})\|_2^2.
\]

The direction is intentionally asymmetric because pseudo-MUA is a lossy electrode pooling of SUA.
`lambda` must be fixed before development evaluation. The default proposal is a single
`lambda=0.05` only after explicit task/consistency loss normalization is frozen; otherwise one
lambda may be selected through nested 27-source-only analysis. No development lambda sweep is
allowed.

The matrix is `MVlambda-T4 x 3` and `MVlambda-TS4 x 3`. Success requires SUA non-inferiority to
MV0, at least `+0.03` practical pseudo-MUA improvement over MV0, content evidence in both views,
no increased view gap, unchanged online state/MAC, and one deployment weight copy. If C2 is only
non-inferior, the allowed claim is limited to one shared model supporting both views without
material loss.

## 7. CPU preflight receipt required before any launch

The preflight must fail closed unless it records and verifies:

### 7.1 Reused-artifact identity

```text
artifact and run_metadata SHA-256
teacher path and SHA-256
strict manifest path and SHA-256
source/development session list and order
seed and signal view
architecture, side_dim, feature group
training calibration_n
forward-evaluation calibration_n
label/rate feature pool_n
actual support, feature-fit, and score trial indices
epoch window and checkpoint rule
train-only normalizer hash
permutation family and seed
task/loss settings
formal files opened = false
```

### 7.2 Descriptor invariants

- `Z4` is bitwise zero after normalization;
- `PH4`, `AC4`, `MB4`, and `B4` remain width four;
- `PH4` is computed from raw `a,c,m` before its source-only phase normalization, uses no tuned
  epsilon, and reports all `m=0` rows;
- `LS4.b` is copied bitwise from aligned T4;
- `LS4.a/c/m` change only through direction-label permutation;
- row-shuffle controls permute complete normalized rows only;
- all descriptors are finite and match the unit/electrode count;
- task targets and activity tensors are unchanged by descriptor controls;
- normalizers are fit on source training data only.

For C, tests must additionally prove count conservation under electrode pooling, singleton
electrode identity, one-electrode-per-unit mapping, pooled-rate T4 refitting, view-specific cache
and normalizer isolation, shared weights, exact loss weights, no hidden consistency gradient when
`lambda=0`, and the intended C2 stop-gradient direction.

### 7.3 Cost receipt

Every arm reports learned parameters, FP32 weight bytes, calibration MACs, fit workspace,
persistent descriptor bytes, B3 support-state bytes, online per-bin MAC/workspace, training-only
compute, peak GPU memory, and wall-clock time. For `N` units, T4 persistent descriptor state is
explicitly `4*N*dtype_bytes`; temporary estimator state must not be hidden inside that number.

## 8. Managed execution order

```text
0a. Audit and freeze M30 B0/T4/TS4 artifact eligibility and actual scorer trace.
0b. Independently audit the selected-M50 score-boundary conflict; do not block M30 A.
0c. Independently audit bridge-Q30/T4-50-source10 semantics; block C only.

1. If 0a passes, implement descriptor invariants and unit tests.
2. Generate an immutable prelaunch receipt for A.
3. Run A's fixed 18-run matrix; freeze the complete aggregate.
4. Run every conditionally triggered component row-shuffle, or none if no trigger fires.
5. Enter B only if direction is independently necessary; select at most one CPU winner.
6. Enter C1 only if bridge preflight passes; enter C2 only if C1 passes.
7. Do not open formal SUA sessions under any branch.
```

Current state after the first audit and implementation rounds:

- `0a` M30 reference eligibility: **PASS**; descriptor-contract implementation may begin;
- `0b` selected-M50 boundary: **PASS**; no fit/scored overlap;
- `0c` bridge semantics: resolved, but C provenance: **BLOCKED_FAIL_CLOSED**;
- PH4 and the other A descriptor cores are implemented; descriptor prelaunch v1 was independently
  rejected because runner/evaluator/aggregate/test/cost coverage was incomplete;
- launch-package v2 added a sealed 18-cell runner, two-GPU scheduler, forced-M30 evaluator,
  deterministic output paths, authorization guard, runtime-source hashes, and the exact PH4
  zero-`m` source-row count (`3`), but it also closed **NO-GO**;
- v2 remains insufficient because its aggregate lacks the hierarchical interval, exact paired
  Wilcoxon, paired/unpaired uncertainty and qualified-T4 sufficiency decision, and because the
  static plus mandatory post-run cost contract is incomplete;
- v2 receipt:
  `results/t4_m30_experiment_a_descriptor_prelaunch_v2_20260802/receipt.json`, SHA-256
  `e708c7bb6957cabb11f432d7021e3be89bccaa12f1baf078bcf35df0a1c63b98`;
- the superseded write-once v3-r2 package is complete. Its
  receipt is
  `results/t4_m30_experiment_a_descriptor_prelaunch_v3_r2_20260802/receipt.json`, SHA-256
  `913b78d1eee95a6a412f083b1c095fc859ffb677c70f4c18583f7b9bceeb3796`;
- v3-r2 binds the fixed two-GPU runner, per-cell logs/status, launch-time source-hash rechecks,
  post-run CUDA peak/wall-clock receipts, full seed-by-session statistics, and separate
  descriptor/B3S/decoder cost scopes. Root-local static/hash tests passed, but independent
  adversarial review closed the package **NO-GO**: paired-SE computation contained a hard-coded
  decision field; authorization lacked a fixed root trust anchor; each cell and the aggregate did
  not revalidate the complete frozen provenance/status graph; collision/TOCTOU protection was
  incomplete; and measured cost was not summarized. None is an accuracy result;
- write-once v3-r3 sealed a root-controlled project public key and added Ed25519, expiry, nonce,
  paired-SE, note, and cost-reporting repairs. Its immutable receipt is
  `results/t4_m30_experiment_a_descriptor_prelaunch_v3_r3_20260802/receipt.json`, SHA-256
  `6960f93121796fd5d398242a704844cc99d5f177c3ca620e515a9b88e8621be7`;
- root closed v3-r3 **NO-GO** before independent review because its new verifier was not wired
  into a versioned runner/scheduler and the complete on-disk success/tamper/failure suite was not
  executed;
- v3-r4 then added a versioned runner/scheduler, fixed-path Ed25519 verification, atomic nonce
  claim, cell locks, and status validation. Its immutable receipt is
  `results/t4_m30_experiment_a_descriptor_prelaunch_v3_r4_20260802/receipt.json`, SHA-256
  `a75ad69b68bb62a0ffddbe5e7e95e5c75a9d73356984343895d619a48c545b7c`;
- root also closed v3-r4 **NO-GO** because its scheduler did not invoke an aggregate protected by
  the same fixed auth/claim and the full 24-artifact disk success/tamper/failure suite was absent.
  Write-once v3-r5 then closed those gaps and passed root plus independent CPU audits. Its receipt
  is `results/t4_m30_experiment_a_descriptor_prelaunch_v3_r5_20260802/receipt.json`, SHA-256
  `7bd117aaa5c934c8e2e5444dd358b1452345e51a3ee88ee929f3d7ec7db590c7`;
- the first signed r5 launch consumed its nonce but aborted before training: the scheduler directly
  executed a mode-`0664` runner, so 18/18 cells returned exit `126` with `Permission denied` and
  produced no checkpoints, scores, or GPU work. r5 evidence is immutable and non-reusable;
- write-once r6 recovery is active on entirely new auth/claim/run paths. It must explicitly invoke
  the cell runner through `bash` and pass fresh root plus independent review before reauthorization;
- r6 subsequently passed both reviews and its explicit `bash` invocation reached the two first-cell
  Python entries, but the `nohup` scheduler tree was reclaimed when the short-lived execution
  wrapper exited. No status, epoch checkpoint, result, cost receipt, or GPU allocation was produced;
- write-once r7 recovery therefore uses new paths and must be launched as a persistent managed
  foreground execution session. r5/r6 authorization, nonce, and incident trees remain immutable;
- r7 was rejected before authorization: its receipt was emitted before final scheduler repair,
  became source-stale after that repair, and retained r6 auth/cache metadata. r7 was never signed
  or run; write-once r8 must finalize code/tests first and emit its receipt last;
- r8 corrected runtime/path/incident metadata but was rejected before authorization because its
  source map omitted the recursively imported r5 aggregate and v3 statistics module. r9 was also
  rejected before authorization because its receipt preceded the final runtime chain and sealed
  only nine sources;
- r10 finalized the complete runtime chain before its write-once receipt. Root and independent
  review verified the 45-item source seal, including all 42 recursively reachable local Python
  dependencies. Receipt SHA-256 is
  `16798e80c210badc734c3687836509740e8b4522b8d4b4f4d3064a6374f60804`;
- a fresh r10 single-use Ed25519 authorization was claimed and Experiment A is running in
  persistent managed foreground session `89162`. This authorizes only the frozen A matrix; B and C
  remain conditional/blocked, and no new accuracy result exists until 18/18 cell validation and
  aggregation complete.

GPU idleness is not a scientific gate and cannot override these dependencies.
