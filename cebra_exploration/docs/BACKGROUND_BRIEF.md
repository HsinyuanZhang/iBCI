# Background Brief — CEBRA Exploration

**Date:** 2026-08-13
**Audience:** every subagent working in `cebra_exploration/`. Read this in full before doing anything.
**Status:** authoritative shared context. If your task prompt disagrees with this file, ask; do not guess.

---

## 1. What this project is

We have a conference paper under preparation. This new sub-project asks three separate questions
about **CEBRA** (Schneider, Lee & Mathis, *Nature* 617:360-368, 2023 — "Learnable latent embeddings
for joint behavioural and neural analysis"):

| Track | Question |
|---|---|
| **A** | Can we *use CEBRA's datasets*? Restricted to motor/movement data. |
| **B** | Can we *adapt CEBRA's method* onto our datasets, as a real comparator? |
| **C** | Can *our method borrow ideas from CEBRA* to improve? |

Each track is dispatched to a separate agent. You own exactly one track. Do not do another track's
work — the outputs are cross-reviewed and overlap wastes everyone's time.

---

## 2. The paper, in enough detail to reason about

**Title:** Closed-Form Encoding Signatures for Backpropagation-Free Session Calibration in Invasive
Brain-Computer Interfaces.
**Source:** `bci_paper_overleaf/paper_6pp.tex` (740 lines, IEEE conference format). Read it.

### 2.1 The problem

An implanted decoder is trained once but deployed for months. Between sessions, electrodes drift,
the spike sorter returns a **different unit set** (different count `N`, different order, different
identities), and tuning changes. Retraining per session is the usual fix, but **a backward pass is
exactly what an implanted, power-limited device cannot afford.**

### 2.2 Our method — the encoding-signature carrier

A permutation-invariant decoder maps a variable-size unit set to behaviour via cross-attention. Each
unit `i` carries a **session-local identity token** `E_i ∈ R^W` that is added to its live activity
window. The paper's contribution is about **what content goes into `E_i`**.

Three phases with a hard optimization boundary:

```
source training           →  session calibration              →  online decode
(backprop, past sessions)    (closed-form fit + 1 forward pass)   (cached E_i, frozen weights)
```

**No parameter receives a gradient on the target session.** This is the paper's central constraint.
Every design idea you propose must respect it, or must be explicitly flagged as violating it.

The carrier itself is a closed-form per-unit OLS regression of firing rate onto a task basis
`φ(t)`, fitted on a short labelled calibration prefix:

```
r_i(t) = b_i + w_i^T φ(t) + ε_i(t)      →     β_i = X† r_i     (X = [1 | Φ])
```

For center-out data with a target direction `θ_m` per trial, `φ = [cos θ, sin θ]`, which is the
classical Georgopoulos cosine-tuning model, and the descriptor is `T4_i = [â_i, ĉ_i, m_i, b̂_i]`
with modulation depth `m_i = ||ŵ_i||`. The descriptor is concatenated with an activity summary and
mapped to `E_i` by a small MLP, then **cached for the whole session**.

Two properties matter for Track C:
- **Pooling linearity.** Because `β_i` is linear in `r_i`, pooling units into a channel sums their
  coefficient vectors: `β_j = Σ_{i∈S_j} β_i`. Survival under loss of spike sorting is *predicted*,
  not just observed. (`m_j` is not additive and must be recomputed.)
- **Attachment sensitivity.** Correct carrier content attached to the *wrong* units (row shuffle) is
  **worse than a zero carrier**. The effect is a property of the correct unit-behaviour pairing.

### 2.3 The claim, stated precisely

We do **not** claim state-of-the-art accuracy. On the FALCON H1 leaderboard, NDT2-Multi few-shot
supervised scores 0.52 held-out against our 0.2749. The claim is a **conjunction of axes**:

| Axis | Our position |
|---|---|
| Target-session supervision | sparse labels (event/trial level), not dense |
| Backward pass on new session | **none** |
| Channel correspondence required | **none** (permutation-invariant) |
| Identity path size | 58,140 params, 102.6× smaller than the activity-only baseline |
| Deployment | ~12% lower organizer-measured latency |

No other method in the literature occupies all four at once. Read
`sua_exploration/docs/HANDOFF_COMPARATORS_20260812.md` §5 for the seven-axis frame.

### 2.4 Known weak points — be honest about these

- On H1 the sparse claim **does not hold**: a four-trial sparse check moved from `+0.0285` on one
  date to `−0.0226` on another. H1 uses a *dense* 7-DoF population carrier, and the paper explicitly
  makes no sparse claim there. CPU screens found no sparsity headroom below ~20 events.
- Held-in 0.4731 vs held-out 0.2749 is a ~0.20 gap; FALCON argues oracle models show only ~0.04.
- The "ridge loses because it is underdetermined" argument was **tested and refuted** — reducing the
  ridge's dimension by PCA *hurt* it. Do not reinstate posedness as a causal claim.

---

## 3. Our datasets, and the scope discipline

**This section is the single most common source of agent error in this repo. Violating it voids
results.**

**CORRECTED 2026-08-13** — an earlier version of this table put RT in `data/000129/sub-Indy`. That
was wrong; it is verified below. If you read the earlier version, re-check your RT paths.

**subject-M, subject-C and RT all live in the same DANDI set, `dandi_000688` (12G), Miller lab.**
Verified file counts:

| Subject | total NWB | `ses-CO` (center-out) | `ses-RT` (random target) |
|---|---:|---:|---:|
| `sub-C` | 68 | 53 | **15** |
| `sub-M` | 22 | 22 | 0 |
| `sub-J` | 3 | 3 | 0 |

| Name | Path | What it is |
|---|---|---|
| subject-M | `sua_exploration/data/dandi_000688/sub-M/sub-M_ses-CO-*.nwb` | Center-out reaching, monkey. The **external** cohort. SUA and deterministically pooled pseudo-MUA views. The paper reports 15 sessions. |
| subject-C | `sua_exploration/data/dandi_000688/sub-C/sub-C_ses-CO-*.nwb` | Center-out, the six-session **development** domain. Formal test sessions remain sealed. |
| **RT (random target)** | `sua_exploration/data/dandi_000688/sub-C/sub-C_ses-RT-*.nwb` — **15 sessions** | Continuous random-target cursor control, **same animal as subject-C**. Its recorded trial-table direction field is **degenerate** (one unique value across all 15 sessions), so direction must be derived from cursor endpoints (the T4d estimator). The sealed loader is `sua_exploration/mc_maze/rt_classical_comparators.py`, which hard-rejects any file not matching `sub-C_ses-RT-*`. |
| FALCON M2 | `SPINT-main/data/000953/` (15G) | FALCON benchmark, monkey N. |
| FALCON H1 | `SPINT-main/data/000954/` (98M) | FALCON benchmark, human Pitt, 7-DoF. |

**Present but NOT used by the paper** — do not confuse these with the above:
`sua_exploration/data/000128/sub-Jenkins` (NLB MC_Maze) and
`sua_exploration/data/000129/sub-Indy` (NLB MC_RTT, only a train and a test NWB — this is *not* our
RT dataset), plus `SPINT-main/data/000941` (FALCON M1).

### 3.1 The held-out rule is PER DATASET

An earlier blanket rule said "never open a path containing `held-out`". **That is wrong** and caused
errors in both directions.

| Dataset | Is `held-out-calib` in scope? | Why |
|---|---|---|
| FALCON M2 | **Yes** | It is FALCON's released few-shot calibration data; the sealed M2 arms use exactly it as the M24 budget. |
| FALCON H1 | **No** | The whole H1 evidence chain is scoped to the 13 public **held-in** recordings. Route H1 file discovery through `sua_exploration/mc_maze/h1_sparse_event_endpoint.index_heldin_calib`, **never** a raw glob. `reject_path_scope` fails closed on `held-out`. |
| subject-M, RT | n/a | No such split. |

Never confuse `held-out-calib` (released calibration data) with the private evaluation set. The
latter is never legal. A previous agent globbed `*held-out-calib*.nwb` for H1 and its entire receipt
was quarantined.

### 3.2 Sealed evidence

Many modules under `sua_exploration/mc_maze/` and `SPINT-main/src/` are **sealed**: their numbers are
cited in the paper and bound to receipt SHAs. **Do not modify them.** Import and reuse them. If you
believe a sealed module has a bug, report it — do not fix it.

---

## 4. Where the comparator work already stands

Read `sua_exploration/docs/HANDOFF_COMPARATORS_20260812.md` — it is the single authoritative
comparison document. Summary of coverage:

| Dataset | Ridge | Population vector | Kalman | FA alignment | Carrier (ours) |
|---|---|---|---|---|---|
| subject-M SUA | 0.4179 | 0.1154 | skeleton, unrun | skeleton | 0.3568 |
| subject-M pMUA | 0.4102 | 0.1042 | skeleton, unrun | skeleton | 0.3061 |
| FALCON M2 | 0.1139 | definable, unrun | blocked | n/a (cite FALCON) | 0.2268 |
| RT | 0.2002 | proven inapplicable | skeleton, unrun | skeleton | 0.4482 |
| FALCON H1 | 0.2582 / 0.2889 tuned | quarantined | skeleton, unrun | n/a (cite FALCON) | 0.5000 |

Cited-not-reproduced baselines come from **FALCON Table 1** (same hidden endpoint, so citable):
on H1, NoMAD+WF 0.13, CycleGAN+WF 0.12, Wiener Filter oracle 0.21, NDT2-Multi few-shot 0.52.

**The governing rule:** you may cite another paper's number only if it was computed on the same
data, the same split and the same metric. FALCON M2/H1 qualify. subject-M and RT never do — every
comparator there must be run by us.

**Existing comparator template.** Follow the house pattern exactly:
- protocol doc, frozen and dated *before* any arm runs — see
  `sua_exploration/docs/FA_ALIGNMENT_COMPARATOR_PROTOCOL_20260813.md`
- implementation — `sua_exploration/mc_maze/fa_alignment_comparator.py`
- runner — `sua_exploration/scripts/run_fa_alignment_comparator.py`
- tests — `sua_exploration/tests/test_fa_alignment_comparator.py`
- a **Part A constructibility audit** that runs first and can return a `*_UNDEFINED_*` verdict. A
  method being structurally inapplicable to a dataset is a **finding**, not a failure.
- every number carries an immutable JSON receipt with input SHAs and an integrity gate that
  reproduces the sealed reference for that dataset before any new arm is interpreted.

---

## 5. What CEBRA is

Contrastive self-supervised learning for neural data. Instead of predicting behaviour, it learns an
**embedding** `f(x) ∈ R^d` trained with an InfoNCE objective where the positive-pair distribution is
defined by an auxiliary variable:

- **CEBRA-Behavior** — positives are sampled from time points with *similar behaviour labels*. Uses
  labels, but only to define the sampling distribution, never as a regression target.
- **CEBRA-Time** — positives are temporal neighbours. **Fully unlabelled.**
- **CEBRA-Hybrid** — both.

Decoding is a separate downstream step (typically kNN or linear on the embedding).

**Two properties make CEBRA directly relevant to us:**

1. **Multi-session training with different neuron counts.** Sessions with 20 and 31 units train
   jointly and map into one shared latent space. **Verified working in our environment** (see §6).

   **CORRECTED 2026-08-13.** An earlier version of this brief said CEBRA does this with
   *session-specific input layers* over a shared trunk. **That is wrong.** Multi-session CEBRA builds
   an `nn.ModuleList` of **completely independent full encoders**, one per session, sharing **zero
   weights** — measured: not one tensor is identical between two session encoders after a joint fit.
   Alignment comes entirely from the contrastive loss. Cost grows as
   `O(N_sessions × full encoder)`. See `COORDINATOR_VERIFIED_FINDINGS.md` F2b, which also explains
   why this *strengthens* our efficiency contrast. The "session-specific input layer" pattern does
   exist in CEBRA, but only on the **single-session** `adapt=True` path (F3).
2. **Consistency metric.** CEBRA quantifies cross-session/cross-subject agreement as the R² of a
   linear map between embeddings. That is a ready-made diagnostic for "did the latent space stay put
   across sessions", which is precisely our drift question.

**The axis where CEBRA differs from us:** adapting to a *new* session requires training that
session's input layer, i.e. **a backward pass on the target session**. We perform none. Quantifying
what that backward pass buys is a genuine result either way.

CEBRA's own motor dataset is `cebra/datasets/monkey_reaching.py` (Area2_Bump, monkey somatosensory
area 2 center-out reaching). Its other datasets are rat hippocampus and Allen mouse visual cortex.

---

## 6. Environment — verified, do not re-litigate

**Use the existing `spint` env. Do not create a new conda env** (explicit user instruction: port the
code onto our environment, because we will need to fine-tune the network and deployment).

CEBRA v0.6.1 source is **vendored** at `cebra_exploration/third_party/cebra/` (git
`d1842ccc659bdf2d458ab31784ae029b9d48d21f`), not pip-installed, so we can modify the architecture.

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/xinyuan/Work_host/SPINT/cebra_exploration/third_party/cebra \
/home/xinyuan/miniconda3/envs/spint/bin/python -s your_script.py
```

**`PYTHONNOUSERSITE=1` (or `python -s`) is mandatory.** A user-site `torch 2.12.0+cu130` shadows the
env and reports `cuda False` because it is too new for driver 535.309.01. With user-site disabled you
get `torch 2.5.1.post303` with **CUDA available on 2× RTX 3090**.

Verified working already:
- `import cebra` → 0.6.1, torch 2.5.1, numpy 2.0.1
- single-session `fit`/`transform`
- **multi-session fit with mismatched neuron counts (20 vs 31)** → both transform to a shared 3-D space

Only missing dependency was `literate-dataclasses`, installed `--no-deps` (nothing else changed).

**Network is flaky.** `pypi.org` through the clash proxy on `127.0.0.1:7890` gives intermittent
`SSL: UNEXPECTED_EOF_WHILE_READING`. The reliable recipe is the Tsinghua mirror with the proxy
stripped:

```bash
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python -m pip install \
  --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple <pkg>
```

Always `--no-deps` unless you have checked the transitive set. Breaking the `spint` env breaks the
paper's whole GPU pipeline.

---

## 7. Hard rules for every track

1. **CPU only.** Another agent session may be running GPU jobs; a past incident killed GPU runs
   through contention. Do not launch GPU work. Do not run `nvidia-smi`-gated jobs.
2. **Do not run experiments that produce citable numbers.** Build, unit-test, audit, and report.
   Small smoke tests on synthetic or tiny real slices are fine and encouraged. A full scoring run is
   not. The coordinating agent decides when to run.
3. **Do not modify anything outside `cebra_exploration/`** except where your task explicitly says
   so. In particular do not touch `sua_exploration/mc_maze/`, `SPINT-main/src/`, or the paper.
4. **Respect §3 scope discipline**, especially H1 held-in-only.
5. **No new conda env**, no upgrading existing packages.
6. **Write down what you did not verify.** An honest "I could not check X" is worth more than a
   confident guess. Every claim you make will be cross-reviewed and then audited.
7. **Report structure:** write your findings to the markdown file your task names, under
   `cebra_exploration/docs/`. Code goes under `cebra_exploration/src/`, `scripts/`, `tests/`.
8. **State-of-the-art honesty.** If the answer to your track is "no, this is not worth doing", say
   so plainly and give the reason. A well-argued negative is a successful outcome.

---

## 8. Key files

| Path | What |
|---|---|
| `bci_paper_overleaf/paper_6pp.tex` | The paper |
| `bci_paper_overleaf/references.bib` | Bibliography (CEBRA is already cited as `cebra2023`) |
| `sua_exploration/docs/HANDOFF_COMPARATORS_20260812.md` | Authoritative comparison plan |
| `sua_exploration/docs/FA_ALIGNMENT_COMPARATOR_PROTOCOL_20260813.md` | Template for a comparator protocol |
| `sua_exploration/mc_maze/fa_alignment_comparator.py` | Template for a comparator implementation |
| `sua_exploration/mc_maze/h1_sparse_event_endpoint.py` | H1 scoped loaders — the only legal H1 entry point |
| `sua_exploration/mc_maze/rt_classical_comparators.py` | RT ridge/PV, sealed |
| `cebra_exploration/third_party/cebra/` | Vendored CEBRA 0.6.1 source |
