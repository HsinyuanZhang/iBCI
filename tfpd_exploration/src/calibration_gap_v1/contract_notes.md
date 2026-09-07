# Z6 — FALCON contract questions (to resolve by reading the spec, zero GPU)

Open questions, each gating a named path. Record the verdict + citation
(failure-closed: no verdict = not permitted).

Sources read (2026-08-25):

- **[PAPER]** Karpowicz et al., "Few-shot Algorithms for Consistent Neural
  Decoding (FALCON) Benchmark", bioRxiv 2024.09.15.613126v2 (Oct 31, 2024;
  NeurIPS 2024 Datasets & Benchmarks track),
  https://www.biorxiv.org/content/10.1101/2024.09.15.613126v2
- **[REPO]** official challenge repository snel-repo/falcon-challenge
  (README.md and falcon_challenge/evaluator.py at `main`),
  https://github.com/snel-repo/falcon-challenge
- Local adapters cross-checked: `SPINT-main/third_party/falcon_challenge/spint_sample.py`
  and `spint_decoder.py` (the submission-side interface actually wired in this
  workspace: `reset(dataset_tags) -> observe/predict(neural_observations) -> on_done`).

## Q1 (gates P4, and U in P1): transductive evaluation-stream activity

May a submission compute label-free statistics (normalization/whitening,
neural-covariance subspace U) from the unlabeled evaluation-stream neural
data of the target session, beyond the M calibration trials?

**VERDICT: PERMITTED — explicitly supported, as a named data-use class
(few-shot unsupervised / test-time adaptation), with a causality requirement
and a mandatory class declaration. It is NOT the same column as the
few-shot-supervised / strict-total-calibration regime.**

Clauses:

1. [PAPER §2.2 "Few-shot unsupervised"]: "methods remove the need for
   behavioral data on new days, skipping explicit calibration periods by
   allowing recalibration procedures to be performed using only neural data
   from normal iBCI use."
2. [PAPER §2.2 "Test-time adaptation"]: "these methods use neural data and
   inferred behavioral labels collected during normal iBCI use to perform
   'semi-supervised' decoder recalibration."
3. [PAPER Fig. 1c caption]: "Zero-shot methods use no data from held-out
   sessions, few-shot methods use the calibration splits from held-out
   sessions, and test-time adaptive methods can implement behavior-free,
   unsupervised decoder updates during evaluation."
4. [PAPER §5 Discussion]: "we impose minimal restrictions on training
   strategies: we allow zero-shot, few-shot, or test-time adaptation and
   provide generous compute for model inference."
5. [REPO README §Code]: "During `reset`, `predict`, and `observe` methods,
   your approach has access to a new timestep of neural observations. ...
   Only within-trial data will be considered for evaluation, but you are
   welcome to use data from the entire available time period."
6. [REPO evaluator.py `predict_files`]: for the movement tasks
   (`h1/m1/m2`, `continual = True`) every timestep calls
   `decoder.predict(neural_observations)`; ground-truth targets and eval
   masks never cross the decoder boundary (predictions are pickled and
   compared server-side) — unlabeled eval-stream activity structurally
   reaches the decoder and nothing in the evaluator restricts its use.

Binding conditions on the permission:

- **Causality** — [PAPER §2.1]: "The evaluation server (EvalAI) requires
  causal, open-loop predictions to be made on streaming neural data,
  timestep-by-timestep." A transductive statistic must therefore accumulate
  causally as the stream arrives (streaming normalization / incremental
  covariance), not in a non-causal second pass.
- **Class declaration** — [PAPER Fig. 1c + Table 1 legend: OR/ZS/FSU/FSS/TTA]:
  using eval-stream activity puts a submission in the FSU/TTA column; it must
  be reported as such and never alongside (or in place of) the
  strict-total-calibration regime.
- **Trial-structure caution** — [PAPER §5]: "evaluation may be susceptible to
  promoting models that exploit trial structure implicit in the datasets";
  statistics must be label-free and not cue-driven (see Q2).

Consequence for the route: P4's unlabeled activity statistics and P1's
transductive U are contract-legal, but they are a *regime change* relative to
the strict `total_selected_calibration` honest column: they must be labelled
transductive/TTA in every table, and the strict-regime activity cost stays
the honest-deployment accounting.

## Q2 (gates the D-opt label accounting): cue metadata budget

D-opt-first30 selection consumes the cue identity of the first 30 trials to
choose 4. Are experimenter-side cues (a) free metadata, (b) cheap labels to
count, or (c) unavailable at deployment? Same question generalized: cue
identity across ALL unlabeled trials would allow cue-conditional
mean-activity carriers at zero kinematic-label cost — the cheapest possible
major win IF the contract permits.

**VERDICT: (c) UNAVAILABLE at evaluation for the movement tasks; within the
released calibration split the only behavioral data are the paired covariates
themselves, so any cue identity derived from them is LABEL SPENDING, not free
metadata. Wider unlabeled D-opt pools and cue-conditional carriers at zero
kinematic-label cost are NOT PERMITTED (failure-closed).**

Clauses:

1. [PAPER §5 Extensions and limitations]: "To penalize sensitivity to trial
   structure, FALCON does not provide trial labels in movement decoding
   tasks." (The route's targets — the mc-maze-style reach surfaces — are
   movement tasks.)
2. [PAPER §2.1]: "Lack of trial structure in movement tasks is an important
   training time consideration; decoders trained on trialized data can
   degrade significantly when evaluated continuously (Section A.5.1)."
3. [REPO evaluator.py `predict_files`]: the decoder interface receives only
   binned `neural_observations` (shape [batch, n_channels]) plus hashed
   dataset tags in `reset`; for continual tasks (`continual = True` covers
   h1/m1/m2) the loop's `trial_change` signal is never forwarded —
   `decoder.on_done(trial_delta_obs)` is called only when
   `not self.continual`. No cue/target identity exists anywhere in the
   evaluation interface.
4. [REPO README §Code]: trial-timing access is explicitly routed through
   `on_done` ("To access and make use of trial timing signals, implement the
   `on_done` method"), which movement tasks never invoke.

Accounting consequence: D-opt-first-30 / cue-balanced selection spends the
cue identity of the pool it selects from. Under this contract that identity
is purchasable only through the paired behavioral covariates of the labeled
calibration budget (choosing WHICH M labeled trials to calibrate on is a
legal label-allocation choice; harvesting cue identity from unlabeled trials
is not). The existing factorial's `label_limited_m30_activity` regime —
which assumes the first-30 pool's cues are readable — is therefore itself a
diagnostic convention, not a deployable FALCON movement-task configuration,
and must be labelled as such wherever it is quoted.

## Q3 (gates nothing yet, paper armour): few-shot budget definition

Does the FALCON few-shot definition count (i) labeled calibration trials
only, (ii) all calibration data touched, or (iii) any evaluation-stream
access? The calibration-cost-normalized protocol (Path 4 armour) needs this
typology to place SPINT (unlabeled), our carrier (labeled), Ridge/Wiener
(labeled), and transductive variants on one honest axis.

**VERDICT: the few-shot budget is the RELEASED HELD-OUT CALIBRATION SPLIT as
a paired neural+behavioral volume (definition (ii) restricted to that split):
touching the split at all is few-shot; the behavioral channel makes it
few-shot-supervised (FSS), neural-only makes it few-shot-unsupervised (FSU);
evaluation-stream access is a SEPARATE declared axis (TTA), not counted in
the calibration budget.**

Clauses:

1. [PAPER §2.1]: "An evaluation split of the same length is withheld from
   both held-in and held-out. All remaining data is released for held-in
   sessions while a small fraction of data is released for held-out
   sessions." + "only a small amount of supervised data is released from
   held-out sessions."
2. [PAPER A.3 per-dataset Processing]: M1 — "Ten trials were released for
   each held-out calibration set" (1.1–2.2 min); M2 — "For held-out sessions,
   the first 10% of each session is released as the calibration split"
   (0.8–1.7 min); H1 — "The first 20% of each file is released for each
   held-out session as the calibration split" (1.5–1.8 min); H2 — "only 3
   trials are released for few-shot calibration" (1–2 min); B1 — "3 motifs
   are made available for the few-shot calibration split" (2.7 s).
3. [PAPER A.5.2]: "FALCON aims to enforce the realistic constraint that
   calibration on new sessions will have limited neural and behavioral data.
   To ensure the few-shot problem was well-represented, we established that
   the held-out calibration splits were insufficient to train new linear
   decoders on their own."
4. [PAPER §2.2 + Fig. 1c + Table 1 legend]: the benchmark's own data-use
   typology — ZS (no held-out data), FSU (neural calibration only), FSS
   (paired calibration), TTA (evaluation-stream neural + inferred labels) —
   is the declaration axis the protocol should adopt verbatim.

Protocol consequence (Path 4 armour): report three axes for every system —
(i) labeled calibration volume (trials / minutes), (ii) neural calibration
volume touched, (iii) data-use class (ZS/FSU/FSS/TTA). That grid places
SPINT (unlabeled neural → FSU), our carrier (labeled → FSS), Ridge/Wiener
(labeled → FSS), and any transductive variant of ours (TTA) without letting
unlabeled-vs-labeled activity be silently conflated.

## Verdict log

| Q | verdict | evidence | date |
|---|---|---|---|
| Q1 | PERMITTED as FSU/TTA class, causal accumulation, class-declared; NOT the strict-regime column | PAPER §2.1, §2.2 (FSU + TTA), Fig. 1c caption, §5; REPO README "welcome to use data from the entire available time period"; evaluator.py predict loop (targets never cross the decoder boundary) | 2026-08-25 |
| Q2 | NOT PERMITTED as free metadata — cue/trial identity unavailable in movement tasks; cue use = label spending | PAPER §5 "FALCON does not provide trial labels in movement decoding tasks", §2.1; evaluator.py (on_done never called for continual h1/m1/m2); README (timing only via on_done) | 2026-08-25 |
| Q3 | Budget = released held-out calibration split as a paired neural+behavioral volume; eval-stream access is a separate declared axis (ZS/FSU/FSS/TTA typology) | PAPER §2.1, A.3 per-dataset Processing (10 trials / first 10% / first 20% / 3 trials / 3 motifs), A.5.2, Fig. 1c + Table 1 legend | 2026-08-25 |
