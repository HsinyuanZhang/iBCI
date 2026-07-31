# T4 next-round ideation after negative extension results

**Status:** contingency design plus isolated component readiness; no arm is
authorized before the active v1 → v2 → exact-head/M15 evidence gates finish.

**Evidence boundary:** this document uses completed train/validation results and
source/cost audits already recorded in `CURRENT_RESULTS.md`. It opens no new
dataset and contains no result from formal held-out sessions.

## Starting point

The project is not looking for another way to prove that ordinary T4 works:
ordinary T4 is already strongly positive on SUA and native M2. The unresolved
problem is to obtain one of two stronger outcomes without destroying that
baseline:

1. accuracy: a mechanism that reliably exceeds ordinary T4; or
2. deployment: the same T4 content effect with materially fewer labels or less
   session/online compute.

Three negative results constrain the search:

- electrode-relation lookup did not improve T4;
- confidence-FiLM was only `+0.0034 R²` and did not beat matched controls;
- v1 decoupled K/V destroyed the pretrained activity path, and both static
  E-T4 and dynamic x-only versions failed badly.

The ideation lenses are failure analysis, composition/decomposition and the
simplicity test. The main design rule is therefore:

> Preserve the selected ordinary-T4 function at initialization; add only the
> interaction whose causal role is under test.

## Divergent candidates

The list is retained even when a candidate is rejected so later rounds do not
silently repeat the same idea.

1. **Zero-initialized T4 key residual on the coupled teacher.** Keep
   `K,V = teacher(x+E)` and add a session-static low-rank
   `ΔK_i = U φ(T4_i)` to K only.
2. **Zero-initialized T4 attention-logit bias.** Predict a query/unit bias from
   T4, leaving teacher K and V exactly unchanged.
3. **Post-readin T4 residual.** Add `g(T4_i)` after `fc_in(x_i+E_i)` instead of
   replacing the read-in or separating K/V.
4. **T4-conditioned input gain.** Apply a zero-initialized multiplicative
   residual to `x+E` before the coupled read-in.
5. **Cross-budget identity consistency.** During held-in training, require
   M10/M15/M20 T4 estimates from the same session to yield consistent E and
   predictions; deployment uses only the chosen low budget.
6. **Learned analytic shrinkage.** Map T4 sufficient statistics
   `(X'X, X'y, exposure)` to shrinkage coefficients offline; calibration remains
   a forward/closed-form update with no gradient.
7. **Streaming RLS T4 + B3T.** Maintain cosine-fit sufficient statistics as
   labeled trials arrive and refresh the B3T session state incrementally.
8. **B3TStream distillation from selected B3S+T4.** Preserve the T4 content
   mechanism while reducing calibration state and encoder MAC.
9. **Second-harmonic tuning features.** Extend the cosine fit with
   `cos(2θ), sin(2θ)` to represent bimodal directional tuning.
10. **Poisson/log-link T4.** Replace least-squares rate fitting with a guarded
    Poisson GLM/IRLS estimator.
11. **Posterior-standardized T4.** Feed `a/SE(a), c/SE(c)` rather than adding
    confidence through FiLM.
12. **Unsupervised/T4 mixture with a reliability residual.** Start at T4 and
    learn a confidence-gated residual toward an activity-only identity.
13. **Relative-amplitude normalization before B3.** Use calibration-block
    amplitude/rate scale to normalize activity, not as a static identity table.
14. **Array-coordinate relative bias.** Use relative electrode distance rather
    than electrode-number lookup.
15. **Direction-balanced active label acquisition.** Select a balanced subset
    of calibration trials rather than the chronological first M.
16. **Higher-dimensional condition PSTH summaries.** Encode per-direction
    temporal response differences instead of a four-number rate fit.

## Convergence

| Rank | Candidate | Why it survives | Primary kill condition |
|---:|---|---|---|
| 1 | coupled T4 key residual | exact T4 baseline at init; tests decoder-level identity/activity interaction; small incremental state/MAC | aligned residual fails to beat TS4 residual and T4 continuation |
| 2 | cross-budget consistency + learned shrinkage | directly targets 70% label reduction; calibration still has no backprop | M15 is more than 0.03 below T4@50 or loses T4-vs-TS4 content |
| 3 | B3TStream distillation | already has true bin-streaming state reduction; simple deployment claim | cannot meet non-inferiority to B3S+T4 |
| 4 | second-harmonic or Poisson T4 | changes the estimator rather than adding an identity table | train-only predictive audit is absent or label/design rank becomes unsafe |
| 5 | post-readin T4 residual | baseline-preserving and simple | redundant with ordinary E; no direct content effect |

Candidates 12–14 are deprioritized because the current confidence/electrode
results already make them weak. Candidate 15 is scientifically interesting but
changes the chronological deployment protocol and cannot be mixed into the
current comparison. Candidate 16 increases label/data requirements and should
not precede the simpler estimator tests.

## Winner: coupled teacher + static T4 key residual

Two-sentence pitch:

> Decoupled K/V failed because it removed the pretrained activity read-in and
> changed both attention selection and content. Preserve the selected
> ordinary-T4 decoder exactly, then add a zero-initialized, session-static
> low-rank T4 residual only to its keys so direct functional identity can alter
> whom each behavioral query attends to without replacing activity values.

Minimal mechanism:

```text
z_i = teacher_fc_in(x_i + E_i)
K_i = W_K LN(z_i) + U_K phi(T4_i)        # new residual; zero at init
V_i = W_V LN(z_i)                        # unchanged teacher value
Q_j = W_Q LN(teacher_fc_in(rep_j))       # unchanged teacher query
```

An equivalent lower-state form may add
`Q_j (U_K phi(T4_i))^T / sqrt(d)` directly as a cached low-rank attention-logit
bias. `phi` and `U_K` must be shared across units, with parameter count
independent of N. The first pilot must not add confidence, electrode identity,
token pruning or another value residual.

Why this is different from failed FiLM:

- FiLM altered the pooled activity-to-E path; the decoder still used the same
  entangled K/V representation.
- The proposed residual acts at attention selection while leaving the selected
  T4 encoder, activity read-in and value path intact.
- Zero initialization makes aligned, TS4 and no-residual arms identical to the
  selected T4 function before optimization.

Why this is different from failed decoupled K/V:

- it never removes `teacher_fc_in(x+E)`;
- it never replaces 64 teacher heads with one random/low-rank decoder;
- activity continues to determine both the teacher key and value;
- T4 supplies only an incremental static selection signal.

## Three validation experiments

### Experiment 1 — seed-42 causal screen

Warm-start all arms from the same selected T4 full checkpoint:

1. T4 continuation;
2. aligned `ΔK(T4)`;
3. TS4 `ΔK(T4_π)`, shuffling only residual input rows;
4. optional parameter-matched T4 residual added after attention, with no
   key/logit interaction.

Freeze the selected backbone for the first residual-only diagnostic so every
new arm optimizes exactly the same new tensors. Use the current
`M_activity=30`, `M_T4=50`, eval start 50 and epochs 5–12 rule. Advance only if
aligned residual beats both T4 continuation and TS4 residual; do not use
non-inferiority alone because this candidate adds compute.

### Experiment 2 — optimization attribution

If the residual-only diagnostic contains a positive T4-vs-TS4 effect but is
smaller than `+0.03`, compare two predeclared training policies:

- residual-only; and
- residual plus the teacher attention output projection, with the rest frozen.

This distinguishes insufficient adapter plasticity from a missing mechanism.
It is one optimization round, not an open hyperparameter search.

### Experiment 3 — three-seed effectiveness

Run the selected policy for seeds 42/43/44. `effective` requires:

- mean `ΔR² ≥ +0.03` versus same-seed T4 continuation;
- 3/3 seed means and 6/6 session means positive;
- hierarchical-bootstrap 95% lower bound above zero;
- exact paired session Wilcoxon `p≤.05`;
- aligned residual also passes the same content checks versus TS4 residual.

Only then freeze the FP32 architecture for formal held-out and encoder INT8.

## Two-week-ceiling feasibility pilot

The local implementation is expected to be faster than two weeks; two weeks is
the maximum budget before the branch is killed.

1. Implement the residual as an isolated adapter around the copied teacher
   attention, with cached and on-the-fly equality tests.
2. Prove zero-init bit equality to the selected ordinary-T4 checkpoint.
3. Audit exact incremental parameter, MAC and state receipts at N=64.
4. Run the four-arm seed-42 screen.
5. If the content contrast is absent, kill the branch; if present but small,
   run the single predeclared output-projection optimization above.

## Strongest objection

**Objection:** ordinary T4 is already embedded in E, so a direct T4 key residual
is redundant; confidence-FiLM suggests another conditional path will also be
ignored.

**Response:** redundancy is exactly what the TS4 residual control tests.
Unlike FiLM, this path has direct access to attention selection and starts from
the unchanged successful T4 function. If aligned and shuffled residuals remain
indistinguishable, the branch is rejected after one seed rather than rescued by
more capacity.

## Isolated implementation readiness

The winning component is implemented in
`streaming_spint_t4_key_residual_adapter.py`, without a Lightning selector,
runner or GPU launch. It preserves the existing `fc_in(x+E)`, query, value,
64-head attention, output projection, norms, FFN and readout. The only new
path is a shared `4 → rank → 512` T4 map whose output projection is exactly
zero initialized and whose full-width result can be cached after calibration.
The TS4 control permutes only the T4 rows used by this new residual; the
ordinary identity encoder continues to receive aligned T4.

The focused CPU suite is `6 passed`; the adjacent coupled/decoupled adapter
regression suite is `38 passed`. A no-data smoke against the actual teacher
checkpoint (`D=512`, 64 heads, `W=50`) establishes:

- zero-init residual decode is bitwise identical to ordinary coupled T4;
- cached and on-the-fly residual decode are bitwise identical;
- after freezing the backbone, decoder and identity encoder have zero gradient
  tensors while the residual output factor has a nonzero gradient.

At `N=64, rank=8`, the residual map costs `264,192` calibration-only MAC and
adds a 131,072-byte FP32 full-width cache. The online Linear/attention/FFN MAC
count remains the coupled `57,970,688` plus an elementwise key addition, so this
is explicitly an accuracy candidate rather than an efficiency candidate.
