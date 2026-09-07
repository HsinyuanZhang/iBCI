# Handoff: Task-Frame Stateful Read-in (TF-SR)

Date: 2026-08-19  
Status: **design handoff only; no implementation, preflight, receipt, or GPU launch is authorized by
this document**

## 0. Operator decision

Performance is primary. Do not spend the next GPU round on low-upside Cell-D-neighborhood cells such
as last-bin-only loss, add-versus-concat, another head count, another output residual, another dropout
shape, or another consistency objective.

Build one coherent new system first. Treat the first result as a **system-level performance screen**,
not an isolating causal contrast. If the complete system clearly beats sealed Cell D, replicate it and
then run only the minimum mechanism ablations. If it does not clearly beat Cell D, stop the route; do
not rescue it with a silent module sweep.

Immediate order:

1. treat the terminal AM/IM joint-ablation result as the frozen rationale for masking the complete
   fused unit token;
2. build and independently review the TF-SR no-data scaffold;
3. run the single TF-SR seed-42 system screen only under a separate launch authorization;
4. if and only if TF-SR is a clear seed-42 GO, run matched Cell-D and TF-SR seeds 43/44 before any
   multi-seed superiority claim;
5. finish that matched replication before spending GPU time on attribution or paper-inspired
   extensions.

Cell-D seeds 43/44 are deliberately deferred rather than treated as a launch blocker. Existing A2
multi-seed results suggest that this model family is sufficiently stable for a seed-42 development
screen, while the matched three-seed comparison remains mandatory if TF-SR survives that screen.
The A2 evidence is a prior, not proof that Cell D has zero seed variance.

The RPNT/POSSM review below does not reopen a queue of small Cell-D add-ons. In particular,
time-bin masking, lag selection, timestamp-query readout, T4-RoPE, and context-filtered attention
must not be silently inserted into the first TF-SR cell. They are either source-only audits or named
post-success successors.

The proposed system is **TF-SR: Task-Frame Stateful Read-in**:

```text
shared B3S calibration encoder + closed-form normalized T4 unit coordinates
    -> causal per-time fused unit tokens
    -> state-conditioned permutation-invariant population read-in
    -> shared recurrent dynamics
    -> velocity
```

This supersedes the unimplemented `A-CausalMask` / `A-RecurrentHead` proposal in
`HANDOFF_CAUSAL_TASK_FRAME_STATE_DECODER_20260819.md`. That proposal incorrectly assumed that Cell D
retained a temporal axis inside its attention graph. It must not be implemented or launched.

---

## 1. Evidence motivating a system swap

All numbers below use the governing last-bin, variance-weighted R2, equal-session convention unless
stated otherwise.

### 1.1 Cell D is strong, but it is not a temporal model

Sealed Cell D is currently the strongest teacher-free system:

```text
Cell D external                         0.4179
Cell D within                           0.5697
Cell D - Arm A external                +0.1576
Cell T - Cell D external               -0.0099
```

Its success establishes that whole-unit sparsification and the available task/calibration
information are useful. It does not establish that the SPINT decoder topology is optimal.

The live decoder path is:

```text
query activity [B,N,50]
    + B3S/calibration-derived identity [B,N,50]
    -> Linear(50,512) per unit
    -> two coordinate slots attend over N unit tokens
    -> Linear(512,50) per coordinate
    -> [B,50,2]
```

The 50 bins are feature dimensions, not a sequence axis. The first `Linear(50,512)` mixes the whole
window before cross-attention; attention runs over units, not time. Cell D has no recurrent state and
no per-time population read-in. Its governing last-bin output is wall-clock deployable, but the model
does not explicitly represent causal dynamics.

Code anchors:

```text
streaming_calibration_exp/src/models/components/spint.py:401-403  fc_in
streaming_calibration_exp/src/models/components/spint.py:445      activity + identity
streaming_calibration_exp/src/models/components/spint.py:458-466  unit attention then fc_out
```

### 1.2 Large-v1 is the nearest negative ancestor, not a reason to stop

Large-v1 already implemented a per-time T4-only recurrent set decoder:

```text
causal activity features
    * carrier projection                 # Hadamard fusion
    -> four static learned slots
    -> causal GRU(256)
    -> velocity
```

Its relevant results were poor:

```text
Large-v1 within                         0.2562
Large-v1 external native              -0.0848
```

More broadly, every system that discarded the fused B3S+T4 information condition has been externally
negative:

```text
spintshape_z4 external                -0.2936   # calibration-activity identity only
large_t4 external                     -0.0848   # T4-only recurrent set decoder
bl_t4 external                        -0.9597   # T4-only bilinear recurrent decoder
```

These systems differ substantially, so the table is not an isolating B3S effect. It is nevertheless a
strong performance prior: removing the fused calibration source from the headline TF-SR cell would
reimpose the only information condition that has failed three times and has never been externally
positive in this lane.

This rules out the claim that any T4-conditioned recurrent set reader will automatically beat SPINT.
It does not test the TF-SR hypothesis because Large-v1 combines four problematic choices:

1. carrier content multiplicatively destroys or amplifies the activity token;
2. population queries are static rather than conditioned on the current dynamical state;
3. it lacks the whole-unit robustness treatment proven by Cell D;
4. it has no stable fused activity-plus-carrier path: wrong carrier content directly corrupts the only
   unit representation.

TF-SR must not be described as a scaled Large-v1 rerun. It changes the read-in mechanism, not merely
the recurrent backend or training length.

The performance prior is nevertheless severe and must remain visible:

```text
Large-v1 external native                         -0.0848
+ Cell D's isolated gain over Arm A               +0.1576
= optimistic additive Large-plus-dropout value     0.0728
Cell D external                                    0.4179
remaining optimistic gap                           0.3451
actual Large-v1-to-D gap                           0.5027
```

TF-SR therefore does not begin "0.03 behind D." It begins from a recurrent T4-only ancestor roughly
0.50 behind D and must justify why fused calibration identity plus a different read-in can overturn
that result. The `+0.03` gate is a required margin over the already strong D system, not an estimate
of the architectural gap.

### 1.3 Small neighboring treatments have low expected upside

The following are explicitly deferred, not silently included:

- last-bin-only training loss on Cell D;
- add versus concat on the unchanged Cell D graph;
- head-count changes;
- another temporal output residual (Cell W was negative);
- Large-v1 plus dropout as a standalone performance route;
- S4D versus GRU;
- trainable FIR-timescale sweeps;
- four untied slots;
- consistency or contrastive objectives;
- another sparsification geometry.

They may be useful diagnostics later, but none is the next headline performance bet.

### 1.4 AM/IM is terminal: complete fused-unit removal is load-bearing

The AM/IM cells and matched scorer are terminal under:

```text
tfpd_exploration/results/aimask_v1/cellAM_activity_mask
tfpd_exploration/results/aimask_v1/cellIM_identity_mask
tfpd_exploration/results/aimask_score_v1/subpop_score_receipt.json
```

Governing last-bin, equal-session results:

```text
                         external    versus D    versus Arm A    within
Cell D                    0.4179         --          +0.1576      0.5697
AM: activity*mask + id    0.2996       -0.1183       +0.0393      0.4841
IM: activity + id*mask    0.1146       -0.3033       -0.1458      0.4919
```

AM is below D in 13/15 external sessions; its paired 95% interval versus D is
`[-0.1867,-0.0503]`. IM is below D in 15/15 external sessions; its paired interval is
`[-0.3833,-0.2225]`. Both are below D in all 6/6 within sessions. The pre-registered matrix verdict is
`JOINT_ABLATION_REQUIRED`.

The licensed mechanism sentence is:

> Random whole-unit removal is useful when the live activity and calibration-derived identity of a
> unit are removed together. Preserving either half while stochastically perturbing the other breaks
> the fused unit representation and fails to reproduce Cell D.

This rules out identity-only/fingerprint suppression as a sufficient explanation for Cell D. It does
not prove that calibration identity contains no session fingerprint, nor does the larger IM failure
show that identity is intrinsically more important than activity: IM also creates the strongest
activity/identity attachment mismatch.

Together with `T approximately D`, the result supports treating the complete fused unit token as the
atomic population element. TF-SR already follows this rule: it applies the inherited D dropout only
after B3S+T4 identity and causal activity have been fused into `u_i(t)`, using one unit mask shared
over the complete window. The AM/IM interpretive blocker is therefore cleared without changing the
predeclared TF-SR topology. Do not duplicate or restart AM/IM.

Receipt body SHA-256:

```text
597badd24b0003a486bfb6b78c6f52166dc16930ed173d4ce07f94f6af6f8bb6
```

### 1.5 What RPNT's masking result does and does not establish

RPNT Table 8 reports that a random masking ratio sampled from `U(0,1)` outperformed every tested
fixed neuron/temporal ratio on its T-RT and B-CS tasks. This is useful external support for the broad
principle already isolated by Cell D: training across a distribution of corruption difficulty can be
more robust than choosing one fixed rate.

It is not an independent replication of Cell D's exact treatment. RPNT applies stochastic masking to
neural activity over neuron and time dimensions during self-supervised Poisson reconstruction. Cell D
uses supervised behavior loss and zero-placeholder whole-unit dropout with gain semantics. RPNT did
not isolate a supervised whole-time-bin treatment against our Cell-D law.

Therefore:

- cite RPNT as external motivation for the `U(0,1)` difficulty schedule;
- do not claim that RPNT proves Cell D's mechanism;
- do not add time-bin masking to the first TF-SR bundle;
- after a replicated TF-SR win, a matched whole-time-bin or joint neuron/time masking cell may be a
  one-factor robustness successor, not a headline architecture claim.

### 1.6 What the from-scratch result says about pretraining

RPNT trained from scratch on individual downstream sessions outperformed pretrained POSSM/POYO
baselines in the reported LTRCH comparison. This is useful evidence that a strong architecture can be
competitive without inherited weights. It supports keeping TF-SR teacher-free as a clean first test.

It does not prove that architecture generally dominates pretraining. The comparison changes both
architecture and training method, and RPNT itself improves from `0.8356` from-scratch to `0.8515`
FS-SFT and `0.8778` Full-SFT on T-RT. The honest conclusion is:

> Pretraining is not required for a competitive decoder, but it remains a potentially useful
> optimization component and must be evaluated within a matched architecture.

Do not use RPNT to claim that A2's initialization advantage is unimportant.

### 1.7 Output timing and lag are separate from the TF-SR architecture screen

POSSM uses timestamp-encoded output queries over the three most recent hidden states. This cleanly
decouples prediction timestamps from behavior-channel count and permits queries 50 or 100 ms beyond
the current neural chunk. Its lag experiment was on par with or slightly below zero-lag performance;
it demonstrates interface flexibility, not a positive lag effect.

TF-SR already removes Cell D's `num_covariates == output-slot == trajectory-slot` coupling by
emitting two behavior coordinates at every recurrent time step. The first TF-SR cell therefore does
not need POSSM output cross-attention for interface correctness.

Before any lagged GPU experiment, perform a source-only audit that records the repository's current
neural/behavior alignment and evaluates a small predeclared set of nonnegative deployment lags. A
lag may be frozen only from held-source evidence before target access. Any performance comparison
using that lag must retrain both TF-SR and its matched baseline under the same shifted target; otherwise
lag and architecture are confounded.

### 1.8 T4-RoPE is the strongest paper-inspired successor, not licensed evidence

RPNT Table 6 reports MRoPE above sinusoidal, standard RoPE, and learned positional encoding. The
transferable idea is to represent a continuous physical coordinate through rotations of query/key
subspaces rather than only as an additive feature. Our sealed raw T4 authority already supplies
`theta_i = atan2(c_i, a_i)`, making a session-free T4-angle rotary read-in technically well defined.

However, RPNT's MRoPE encodes site position, time, behavior type, subject identity, or recording
time depending on the dataset. Its ablation does not establish that a unit's tuning angle should be a
rotary attention coordinate. TF-SR must not use session, subject, date, site, or dataset metadata.

If the base TF-SR system is replicated, the first architecture successor should be a one-factor
T4-RoPE treatment on the frozen winner:

```text
held: B3S, normalized T4 content, tokens, dropout, queries, GRU, loss, schedule, scorer
changed: rotate a predeclared query/key subspace by raw-T4 theta
forbidden: session/subject/site/date axes or learned metadata tables
required tests: theta+2pi invariance, joint unit permutation invariance, raw-theta authority binding
```

### 1.9 Context-based attention is deferred until a per-time winner exists

RPNT Table 7 shows its largest ablation effect for context-based attention. The operation convolves a
temporal attention-score matrix using a context-generated kernel. It cannot be inserted into Cell D,
whose attention axis is units and whose time bins have already been collapsed into features.

TF-SR would create a real time-indexed sequence of population read-ins, so a strictly prefix-causal
context filter becomes implementable there. It is nevertheless deferred: the first TF-SR already
conditions population queries on recurrent state, and adding score convolution would obscure whether
state-conditioned read-in works. Any later implementation must convolve only along time with weights
shared over units, preserve unit permutation invariance, and pass every-prefix state/output tests.

### 1.10 Cross-paper scores are not comparable to this zero-target-update route

Do not compare RPNT/POSSM values near `0.84-0.99` numerically with Cell D's `0.4179`. The datasets,
trial intervals, preprocessing, split rules, metric aggregation, and training contracts differ. RPNT's
cross-site FS-SFT uses a 20% downstream training split, and POSSM transfer uses unit identification or
full finetuning. Our governing external route performs zero target optimizer steps and zero target
backward calls. RPNT also reports successful reach-period results for center-out tasks, whereas our
headline is last-bin, variance-weighted, and equal-session.

The admissible paper-level comparison is qualitative: their setting demonstrates transfer after
downstream adaptation; ours asks whether a fixed decoder can transfer under sparse calibration and no
target gradient updates.

---

## 2. Scientific hypothesis

Cell D fixes one population read-in for an entire 50-bin window. TF-SR instead asks whether a shared
causal state should determine how a previously unseen neural population is read at each time step,
using a shared calibration encoder and closed-form task correspondence rather than a persistent
learned unit embedding.

The system hypothesis is:

> A shared permutation-equivariant calibration encoder fused with a closed-form task-frame carrier
> can replace persistent unit embeddings as the interface between a variable neural population and a
> shared recurrent decoder. Conditioning the population read-in on the preceding dynamical state
> should improve cross-subject decoding while preserving permutation invariance and causal
> deployment.

The first GPU cell is a system test of that full statement. It does not attribute performance to any
single component.

---

## 3. Non-negotiable boundaries

TF-SR must satisfy all of the following:

- use the exact frozen strict-27 source roster and support/query lineage;
- use the exact normalized M30 T4 authority already used by the current teacher-free route;
- T4 is produced from labeled calibration by a closed-form/no-backprop encoder;
- use the exact shared B3S architecture on the same M=30 calibration trials, with no per-unit
  parameters and no persistent calibration cache beyond the derived per-unit identity;
- no teacher checkpoint, teacher tensor, teacher prediction, or teacher initialization;
- no learned unit embedding table;
- no learned session, subject, date, site, or dataset table;
- no target-data encoder fitting or target optimizer/backward call;
- no decoupled identity-key/activity-value attention route;
- no behavior scaling change; keep `behavior_scaling_factor = None`;
- preserve unit permutation invariance by construction;
- keep active decoder parameters within the Cell-D order of magnitude, with a preferred ceiling of
  3.6M active trainable parameters;
- emit `[B,50,2]` and reuse the existing governing last-bin scorer;
- use causal inputs only: prediction at time `t` may not depend on activity after `t`;
- use one explicit window-local state reset in the first cell; no hidden trial/session state;
- run seed 42 first;
- first-cell training uses 48 epochs, the established warmup/cosine schedule, and final-four SWA;
- no formal data.

Whole-unit sparsification is deliberately carried over as a known performance component. This makes
TF-SR a system swap rather than an architecture-isolation experiment.

---

## 4. Frozen first-cell architecture

Proposed cell name:

```text
TFSR_B3ST4_DDROP_SEED42
```

### 4.1 Inputs

For each training/evaluation window:

```text
x                  [B,50,N]          normalized query neural activity
calib              [B,M=30,100,N]   exact M30 calibration neural trials
c                  [B,N,4]           exact normalized M30 T4 descriptor
```

`c` must be aligned to the same ordered unit axis as `x`. T4 may not be recomputed, renormalized, or
adapted inside TF-SR.

Compute the exact shared B3S identity once per window/session support:

```text
mean_h_i = mean over M=30 of pre_pool(calib_trial_i)
identity_i = post_pool(concat(mean_h_i, c_i))       # [B,N,50]
```

The exact first-cell B3S architecture is the existing shared encoder:

```text
pre_pool        Linear(100,64) -> ReLU
post_pool       Linear(68,64) -> ReLU
                -> Linear(64,64) -> ReLU
                -> Linear(64,50)
active params   18,290
```

B3S is not a unit table. The same weights are applied to every unit, it is permutation-equivariant,
and a new unit is serviced by its M30 calibration trials plus recomputed T4 rather than by learning a
new embedding row. Retaining it does not weaken the "no persistent learned unit identity" claim.

The first system intentionally retains the fused B3S+T4 information condition used by every strong
system in this lane. T4-only is a post-success information-source ablation, not the main performance
cell. Query activity remains fully present at every time step.

### 4.2 Causal activity encoder

Reuse the existing repository-native `CausalActivityEncoder` implementation from:

```text
tfpd_exploration/src/tfpd/bilinear_readin.py
```

First-cell dimensions:

```text
causal trailing window                 20 bins
activity hidden width                 256
activity feature width                 64
```

For each unit and each time step:

```text
a_i(t) = CausalActivityEncoder(x_i[0:t])       # [B,50,N,64]
```

The encoder uses left zero padding only for unavailable pre-window history. It must never read a
future activity bin. Its prefix-causality must be tested through outputs, not inferred from source
inspection alone.

This is a trainable decoder-side activity featurizer. The phrase "no-backprop encoder" refers to the
closed-form T4 encoder; it does not require the decoder's causal activity MLP to be frozen.

### 4.3 Fused task-frame unit token

At every time step, concatenate dynamic activity features with the static fused B3S+T4 identity:

```text
u_i(t) = UnitMLP(concat(a_i(t), identity_i))
```

First-cell dimensions:

```text
input width                            64 + 50 = 114
token width                            256
UnitMLP                                Linear(114,256) -> ReLU -> Linear(256,256)
```

This deliberately replaces Large-v1's Hadamard product. Activity cannot be destroyed merely because
one carrier coordinate is small or wrong. Calibration activity and T4 are first fused by the shared
B3S encoder; the resulting identity and live activity are then fused before keys and values are
formed:

```text
K = V = fused unit token
```

There is no identity-only key and no activity-only value. Concat is an interface choice, not a
separate novelty claim.

### 4.4 Whole-unit robustness treatment

Reuse the exact Cell-D placeholder-and-gain whole-unit dropout law:

```text
one p draw per training step/window
one [B,N] whole-unit mask
the same unit mask at every time step
the same F.dropout gain semantics as Cell D
no key_padding_mask true-removal substitution
no new min-keep rule
evaluation path: no perturbation
```

Apply the mask to the complete fused token `u_i(t)` before population attention. A dropped unit is a
zero placeholder across the entire 50-bin window, exactly matching the known useful robustness law
rather than introducing a third sparsification treatment.

### 4.5 State-conditioned population read-in

Use two coordinate-associated base queries and condition both on the previous recurrent state:

```text
q_x(t) = q_x_base + Q_x(h(t-1))
q_y(t) = q_y_base + Q_y(h(t-1))
```

First-cell dimensions:

```text
number of query slots                   2
slot meaning                            x velocity / y velocity
token and query width                   256
attention heads                         4
attention layers                        1
FFN width                               1024
```

At each time step, cross-attention runs over the current unit set:

```text
o(t) = CrossAttention(
         queries = [q_x(t), q_y(t)],
         keys    = {u_i(t)} over units,
         values  = {u_i(t)} over units
       )
```

This is the central new information flow. The previous state controls how the current population is
read, while T4 supplies a comparable task-frame description for every unit. Permuting units and T4
rows together must leave all predictions unchanged.

### 4.6 Population-mass disclosure channels

Softmax attention hides absolute population size. Append three non-learned channels to the flattened
two-slot observation:

```text
log1p(number of retained units)
retained-unit fraction
log1p(mean absolute causal activity feature)
```

These channels must be computed without target behavior and must be invariant to unit permutation.
They are required correctness information, not a learned session descriptor.

### 4.7 Shared recurrent dynamics

Use one repository-native GRU:

```text
recurrent input       flattened two-slot observation + three mass channels
GRU hidden width      256
layers                1
direction             forward only
state reset           zero at the start of every 50-bin window
output head           Linear(256,2)
```

The per-time update is:

```text
h(t)       = GRU(concat(flatten(o(t)), mass(t)), h(t-1))
y_hat(t)   = Linear(h(t))
```

Do not introduce S4D, S4, Mamba, a bidirectional RNN, or persistent state across windows in the first
cell. GRU is chosen to isolate the system idea from a new unreviewed numerical backend.

### 4.8 Output and loss

Emit a complete causal trajectory:

```text
y_hat              [B,50,2]
```

Use the established valid-bin MSE over all emitted bins. Unlike Cell D, every output bin is now
prefix-causal, so dense all-bin supervision is compatible with online semantics.

Governing evaluation remains:

```text
last bin only
variance-weighted R2
equal session weight
same source-only normalizer and query authority
```

Full-window R2 may be reported as diagnostic only.

---

## 5. Why this is not the revoked CTF-SD bundle

The revoked CTF-SD proposal combined static slots, S4D, width 256, a new dropout law, causal
convolution, hand-picked summaries, B3S removal, and a new loss without one coherent read-in
mechanism.

TF-SR has one system-level mechanism chain:

```text
T4 task correspondence
    -> causal per-time fused unit token
    -> state-conditioned population read-in
    -> shared recurrent dynamics
```

It deliberately excludes:

- four untied global slots;
- S4D or Mamba;
- RPNT attention-score convolution;
- a trainable FIR bank;
- a new sparsification law;
- self-supervised or contrastive pretraining;
- teacher initialization;
- learned unit/session metadata;
- an output residual attached to Cell D.

This is still a multi-factor system swap relative to D. Success may motivate attribution; failure may
not be interpreted as evidence against any single component.

---

## 6. Relationship to prior art

### 6.1 SPINT / Cell D

SPINT collapses the complete window into each unit token before a static population readout. TF-SR
maintains an explicit time axis, reads the population at every step, and updates a recurrent state.
TF-SR retains the small shared B3S calibration encoder because it is useful information, not a
persistent identity table. It removes SPINT's whole-window unit-token read-in and static trajectory
readout. It has no learned unit identity row, fixed unit slot, or session table.

### 6.2 Large-v1

Large-v1 uses multiplicative activity-carrier tokens and static slots. TF-SR uses fused concat tokens,
state-conditioned coordinate queries, and Cell-D whole-unit robustness. Its isolating prediction is
that **state-dependent read-in**, not recurrence alone, is necessary.

### 6.3 POSSM

POSSM motivates per-time population tokenization followed by recurrent dynamics. TF-SR differs in
the unit interface and read-in:

- POSSM uses learned unit embeddings and multi-dataset pretraining;
- TF-SR uses a shared B3S calibration encoder fused with a label-efficient closed-form T4 descriptor,
  with no persistent unit table;
- TF-SR conditions its population queries on the previous dynamical state;
- a new unit is serviced by its M30 calibration trials and recalculated T4, not by learning an
  embedding row.

Do not claim that POSSM establishes TF-SR's performance.

POSSM additionally motivates a timestamp-parameterized output interface over the three most recent
hidden states. TF-SR does not adopt that output module in its first cell; Section 1.7 specifies the
source-only lag audit and the fairness rule for any later lagged treatment.

### 6.4 RPNT

RPNT motivates causal local context and robustness to neural non-stationarity. TF-SR does not copy
RPNT's MRoPE metadata, session/site IDs, global-attention convolution, or SSL objective. Its causal
activity features and whole-unit robustness are the mechanism-level transfer.

The exact quantitative evidence and its limits are frozen in Sections 1.5, 1.6, 1.8, 1.9, and 1.10.
RPNT's random masking, MRoPE, and context-attention ablations motivate successors; none is evidence
that the corresponding TF-SR treatment will improve zero-target-update transfer.

### 6.5 Observation-model terminology

TF-SR is a discriminative carrier-conditioned read-in. It maps neural measurements to a latent
state; it does not define a generative likelihood `p(neural | state)`. Therefore do not write that T4
"defines the observation model" in the strict state-space sense.

The generative Route C remains a separate, higher-risk future architecture. It is not part of the
first TF-SR cell.

Primary references:

```text
sua_exploration/papers/Ryoo_et_al_2025_POSSM_Generalizable_Real_Time_Neural_Decoding.pdf
sua_exploration/papers/Fang_et_al_2026_RPNT_Robust_Pretrained_Neural_Transformer.pdf
tfpd_exploration/src/tfpd_large.py
tfpd_exploration/docs/HANDOFF_POST_BILINEAR_CONSUMER_EXTENSION_20260815.md
tfpd_exploration/docs/HANDOFF_CAUSAL_TASK_FRAME_STATE_DECODER_20260819.md
```

---

## 7. Stage-0 requirements before any GPU request

Stage 0 is CPU/no-data or source-only as appropriate. It authorizes no target or GPU execution.

### 7.1 Structural gates

1. **Prefix causality:** for every `t`, two inputs with identical prefixes and arbitrary different
   future suffixes must produce identical `y_hat[0:t]` and recurrent states `h[0:t]`.
2. **Unit permutation invariance:** jointly permuting query activity, calibration trials, and T4 rows
   must leave the complete trajectory unchanged.
3. **Variable N:** finite outputs and gradients for the frozen supported unit-count range.
4. **No persistent identity:** parameter registry contains zero unit, session, subject, date, site,
   or dataset table.
5. **Calibration/T4 alignment:** wrong query/calibration/T4 unit counts, order, or receipt bindings
   fail closed.
6. **Mask semantics:** one unit mask is shared across all 50 time steps; evaluation is exact no-mask.
7. **Finite gradients:** B3S pre/post pools, causal activity encoder, fused token MLP, state-query
   projection, attention, GRU, and output head all receive finite nonzero gradients on the aligned
   synthetic fixture.
8. **Output/loss:** exact `[B,50,2]`; all-bin valid MSE; last-bin scorer reuses the established code.
9. **State reset:** state is exactly zero at every window start and cannot leak across batches.
10. **Carrier controls and normalization:** synthetic raw carrier values must be transformed by the
    exact live-datamodule-equivalent source normalizer before entering B3S. Record raw and normalized
    carrier SHAs plus the exact mean/std authority. Zero and wrong-pair controls are constructed in
    the normalized model-visible domain and passed through the same B3S path. Aligned, zero, and
    wrong-pair forwards must be finite and must not change parameter state. No synthetic R2 threshold
    is a performance authorization. This gate is load-bearing: a raw-carrier synthetic fixture may
    pass while the real z-scored system fails, as the prior PV route demonstrated.
11. **No unregistered paper-inspired factors:** the first cell contains no time-bin mask, lagged
    labels, timestamp-query output attention, T4-RoPE, or context-score convolution.

### 7.2 Resource gates

The preflight must report:

- exact active/trainable parameter count;
- parameters by B3S, causal activity encoder, token MLP, attention/read-in, GRU, and head;
- per-window dense MAC estimate at representative N;
- persistent state bytes;
- peak activation estimate;
- measured CPU forward latency on a small synthetic batch;
- expected GPU memory envelope.

Preferred active parameter ceiling is 3.6M. Crossing that ceiling requires explicit review but must
never exceed the current decoder order of magnitude.

### 7.3 Data/provenance gates

- exact strict-27 source roster;
- exact M30 T4 authority path/SHA and normalizer path/SHA;
- exact M30 calibration-trial schedule/query support receipt and B3S architecture closure;
- exact source/query schedule lineage;
- target/formal discovery/open flags false;
- zero target optimizer/backward calls;
- source-only synthetic and real-source smoke receipts;
- code/config/runtime closure and initial-state SHA;
- no-user-site and physical/logical GPU identity deferred to a separate launch preflight.

---

## 8. First performance cell

Only one first-round training cell is proposed:

```text
TFSR_B3ST4_DDROP_SEED42
```

Training contract:

```text
source roster                 exact strict-27
carrier                       exact normalized M30 T4
B3S calibration identity      exact shared B3S, M=30, 18,290 params
teacher                       none
unit/session tables           none
target encoder BP             none
behavior_scaling_factor       None
epochs                        48
optimizer/schedule            established warmup/cosine recipe
SWA                           final four checkpoints
seed                          42
formal data                   forbidden
```

The implementation agent may perform engineering smoke runs only after independent code review. A
real GPU launch requires a separate explicit authorization; this handoff is not that authorization.

---

## 9. Governing evaluation and decision

Compare the final-four SWA model against the sealed matched Cell D and the matched A2 reference using
the same query authority and scorer.

Primary comparison:

```text
TF-SR seed42 versus sealed Cell D seed42
```

Required reporting:

- external mean and median paired delta;
- external per-session table and positive-session count;
- paired bootstrap interval;
- within mean and median paired delta;
- last-bin governing scores;
- full-window diagnostic scores;
- aligned, zero-T4, and wrong-pair-T4 forward diagnostics, with B3S identity recomputed through the
  exact same normalized side-feature path for every control;
- latency, MACs, parameters, peak GPU memory, and persistent state;
- no target update/backward evidence.

Predeclared reading:

```text
CLEAR GO:
  external TF-SR - D >= +0.03
  within   TF-SR - D >= -0.03
  external median delta > 0
  at least 9/15 external sessions positive

AMBIGUOUS / HOLD:
  external delta in [0, +0.03), or a positive mean driven by a small minority
  report the system; do not automatically launch ablations or rescue cells

STOP:
  external delta < 0, or within delta < -0.03
  do not sweep heads, width, GRU size, causal window, or fusion to rescue it
```

Seed 42 is only a performance screen. It cannot establish superiority over a multi-seed reference.

---

## 10. What happens only after a clear GO

### 10.1 Replication before mechanism expansion

If TF-SR is a clear seed-42 GO, run both Cell D and the complete TF-SR system at seeds 43 and 44.
The paper-level performance claim requires matched three-seed aggregates and session-paired
uncertainty. Do not spend the two additional Cell-D runs before TF-SR clears the seed-42 screen, and
do not use A2's apparent seed stability as a substitute for the eventual matched replication.

### 10.2 Minimum ablation set

Only after the complete system is replicated should the following be considered:

1. **Static-query ablation:** remove `Q_x(h(t-1))` and `Q_y(h(t-1))`; keep tokens, attention, GRU,
   dropout, and training unchanged. This tests state-dependent population read-in.
2. **No-state ablation:** replace the recurrent update with a parameter-budget-matched per-time head;
   keep the causal tokens and population reader. This tests shared dynamics.
3. **T4-only information-source ablation:** remove the calibration-activity contribution while
   retaining normalized T4 and the complete TF-SR decoder. This is the correct place to test whether
   the new stateful read-in still needs B3S; do not impose the lane's three-times-negative T4-only
   condition on the headline cell.

Aligned versus zero versus wrong-pair T4 remains a mandatory forward-only diagnostic on every
trained system. Retrain a Z4 arm only if the forward diagnostics cannot distinguish carrier use from
generic recurrent robustness.

Do not begin with width, head, concat/add, state size, window length, or backend ablations. Those are
engineering choices unless the three core ablations leave a specific unresolved mechanism.

### 10.3 Ordered paper-inspired successors

Only after a clear replicated TF-SR win and the minimum mechanism set above:

1. **T4-RoPE:** one changed factor on the frozen winner; highest-priority architecture successor.
2. **Timestamp-query readout:** query a fixed recent-state set for predeclared timestamps. Keep lag
   zero for the architectural comparison; test a nonzero lag only in a separately matched pair.
3. **Whole-time-bin or joint neuron/time masking:** a one-factor training-robustness successor, not a
   claim that RPNT's SSL mechanism was replicated.
4. **Strictly causal context-score filtering:** only if state-conditioned read-in is already supported
   and the filter has an explicit every-prefix proof.

This is an ordered dependency list, not a sweep menu. At most one factor is added per successor.

### 10.4 Interpretation discipline

If TF-SR wins but the static-query ablation matches it, the result is a causal recurrent set decoder,
not state-dependent read-in. If the no-state ablation matches it, the result is a per-time set reader,
not shared dynamics. If the T4-only ablation matches it, B3S was protective engineering rather than a
load-bearing information source in TF-SR. If aligned T4 does not beat zero and wrong-pair controls,
the result is not a task-frame decoder.

---

## 11. Failure policy

A negative first system does not prove that POSSM, RPNT, recurrence, attention, or T4 is generally
wrong. It proves only that the frozen TF-SR bundle did not beat D under this protocol.

Nevertheless, the operational route must stop. Do not convert a failed system into an unregistered
sweep. Any successor must begin with a new mechanism prediction, a new handoff, and an explicit
comparison against the Large-v1 failure.

The generative Route C may remain a later research proposal because it makes a different prediction:
carrier-derived tuning defines `p(neural | latent state)` and the read-in is obtained through an
innovation/filtering update. It is a new architecture, not a cheap TF-SR rescue.

---

## 12. Honest contribution language

Do not claim:

```text
"We added an SSM to SPINT."
"T4 defines the generative observation model."
"The first system result proves state-conditioned attention is responsible."
"TF-SR is POSSM without UnitEmb."
```

Prospective two-sentence contribution:

> Existing set decoders collapse an entire neural window before population read-out, while recurrent
> neural decoders commonly rely on persistent learned unit identities. TF-SR instead uses a
> shared permutation-equivariant calibration encoder fused with a label-efficient closed-form
> task-frame carrier to construct causal unit tokens, and lets a shared dynamical state determine how
> an unseen neural population is read at each time step.

The claim becomes admissible only if:

- the complete system clearly and reproducibly improves external decoding;
- zero/wrong-pair controls establish real carrier use;
- the static-query, no-state, and T4-only ablations support the named mechanisms and information
  source;
- no learned unit/session table or teacher path exists;
- the causal and permutation invariance gates pass on the final implementation.

---

## 13. Immediate handoff to an implementation agent

1. Treat this document as design input, not launch authorization.
2. Build only a route-owned additive scaffold; do not modify sealed Cell D, A2, Large-v1, AM/IM, or
   their receipts/checkpoints.
3. Reuse the existing `CausalActivityEncoder`, stable cross-attention primitives, and GRU where their
   semantics match this contract.
4. Implement the exact first-cell dimensions in Section 4; do not expose an architecture sweep CLI.
5. Implement all Stage-0 adversarial tests and a no-data dry plan.
6. Do not open target/formal data, train, mint an official preflight, or launch a GPU.
7. Stop at a natural no-data review boundary and report file SHAs, tests, parameter/MAC/state/latency
   accounting, and every remaining live execution gap.
8. Require an independent read-only audit before any source smoke or GPU request.
9. The scaffold may be built while AM/IM runs, but no official launch authorization may be minted
   until the immutable AM/IM terminal interpretation has been incorporated without changing the
   predeclared TF-SR topology.
