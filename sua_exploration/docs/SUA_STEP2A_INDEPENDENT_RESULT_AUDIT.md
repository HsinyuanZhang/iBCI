# Independent audit — Step-2A source-only cross-budget T4 receipt

**Audited:** 2026-08-02 (Asia/Hong_Kong)  
**Scope:** independent read-only review of `results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json`. This document does not modify the implementation or result receipt, start a GPU job, resolve/open a formal-test path, or make a decoder R² claim.

## Verdict

**Step-2B: no-go.** The source-LOSO descriptor proxy contains a real-looking, robust signal for the seven implemented unit-level reliability fields, but that incremental signal does **not** generalize distinguishably to the six target-free development-validation sessions. The stated Step-2A CPU kill criterion is therefore met: a correction map does not yet have a reliable, out-of-fold estimator-aware signal beyond the known budget.

**Step-3: unblocked, but not positive.** Under the serialized protocol, this written Step-2 stop/no-go permits the separate CPU-only fixed-K prototype Gate A to be considered. It is not an authorization for a Step-2B GPU pilot or a Step-3 decoder/GPU pilot.

## Boundary and implementation audit

- The receipt is correctly marked source/development only: `no_gpu_launch=true`, `no_formal_test_opened=true`, and `formal_test_paths_resolved=false`. The manifest contains 27 train, 6 development-validation, and 6 sealed formal-test names; the runner resolves only the first two groups.
- All 33 opened sessions use chronological rewarded prefixes, with an evaluation boundary of trial 50. Every budget has a rank-3 fit and finite descriptor rows. Thus the decision is not caused by rank failure or an undefined low-budget fit.
- The cache payload/version includes budgets, SUA signal view, bin/window, reward filter and fit semantics; the result stores the exact 50-trial chronology and cache hash per session. The raw count matrices used for split-half checks are absent from the output and cache receipt.
- In `nested_source_q_error_audit`, each left-out source session is scored by a model fitted from the other 26 source sessions. The held-out `T4@50` enters only after prediction as the offline descriptor-error target. This is target-free at application, not a decoder or deployment performance claim.
- The spike split-half result is a within-trial multinomial partition and rescaled rate, so it preserves the same trial labels/prefix. The 8 resampling seeds within a session are **repeatability diagnostics, not independent biological/session observations**; all inference below uses equal-session aggregates.

### Material semantic limitations (do not over-read the receipt)

1. The fitted `q_plus_m` model contains the seven **unit-level** reliability fields and `M` only. It does *not* include the protocol's proposed global direction-count histogram, direction balance, or design condition. It should consequently be called `q_unit + M`, not the full proposed `q@M` mechanism. This narrows rather than invalidates the no-go: this implemented subset has no reliable validation increment.
2. More fundamentally, the model predicts a scalar squared-error **magnitude**, not the signed four-coordinate residual `T4@50 - T4@M`. Even a successful scalar reliability predictor would say only *whether* a descriptor is unreliable, not *how to correct it*. This receipt therefore cannot by itself establish the proposed vector correction map `g_phi`; it is a necessary-signal preflight only.
3. The receipt does not give the required per-reliability-field degeneracy/distribution table or an ablation/contribution analysis. A scalar `q_unit+M` result cannot identify which causal field, if any, carries transferable information.
4. The target is raw four-coordinate squared error to `T4@50`. Since `m` is determined by `(a,c)`, this endpoint weights directional-amplitude error more than a three-parameter cosine coefficient loss would. It is a legitimate preflight proxy, but it is neither calibrated decoder R² nor evidence that reducing this loss improves decoding.
5. `T4@50` shares the first `M` trials with `T4@M`; this is appropriate for a causal teacher boundary but means the error curve measures approach to the 50-trial estimate, not error to a noise-free ground truth. No future pilot should call it an unbiased ground-truth error.
6. The output has a minor stale type comment saying the reliability array has five columns while the actual named/reported array has seven. The executable shape assertion and result names agree on seven; this is documentation debt, not an observed numerical mismatch.

## Equal-session q-error results

Endpoint: per-unit mean squared four-coordinate `T4@M -> T4@50` error pooled across `M=10,15,20` within each session. Negative paired delta favors `q_unit + M`.

| Evaluation set | Comparison | N | mean paired delta | median | wins | 80%-power paired MDE | normal-approx. 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nested source LOSO | `q_unit+M - M-only` | 27 | -0.7671 | -0.4340 | 21/27 | 0.7668 | [-1.3035, -0.2307] |
| Nested source LOSO | `q_unit+M - constant` | 27 | -0.8566 | -0.4620 | 22/27 | 0.8225 | [-1.4319, -0.2812] |
| Target-free dev validation | `q_unit+M - M-only` | 6 | -0.0226 | -0.0554 | 3/6 | 0.4349 | [-0.3268, 0.2817] |
| Target-free dev validation | `q_unit+M - constant` | 6 | -0.1157 | -0.1179 | 4/6 | 0.5104 | [-0.4726, 0.2413] |

The necessary simpler control generalizes: on the six validation sessions, `constant - M-only` is +0.0931 (all 6 favor `M-only`; 80%-power MDE 0.0823; normal 95% CI [0.0355, 0.1507]). Thus the audit can detect the known budget effect, but not the incremental unit reliability effect.

The apparent source result is not *only* one outlier: removing the largest absolute source delta (`20131023`) gives -0.5360, 20/26 wins, MDE 0.4270; removing the two largest absolute deltas (`20131023`, `20150309`) gives -0.4234, 19/25 wins, MDE 0.2995. It is nevertheless not sufficient for Step-2B because its source-only advantage collapses in the target-free set. On the six validation sessions, dropping the most favorable session (`20151110`) reverses the mean to +0.0976; the validation direction is therefore especially non-robust.

## Cross-budget estimator evidence

All sessions are rank defined at every budget. Equal-session descriptor error declines smoothly with more labels, as it should:

| Set | Budget | mean error to `T4@50` | median | 80%-power MDE |
|---|---:|---:|---:|---:|
| Source train | 10 | 1.2133 | 0.9950 | 0.4185 |
| Source train | 15 | 0.7049 | 0.5554 | 0.2697 |
| Source train | 20 | 0.4676 | 0.3604 | 0.1802 |
| Development validation | 10 | 1.1227 | 1.1048 | 0.3247 |
| Development validation | 15 | 0.6031 | 0.6797 | 0.2235 |
| Development validation | 20 | 0.3753 | 0.3810 | 0.1142 |

Median disjoint-half repeatability also rises monotonically. The table reports component Pearson across units, summarized across the 8 source-session resamples (and shown only as a stability diagnostic):

| Budget | Source median `(a,c)` | Source median `m` | Source median `b` | Dev median `(a,c)` | Dev median `m` | Dev median `b` |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 0.589 | 0.670 | 0.993 | 0.624 | 0.666 | 0.991 |
| 15 | 0.656 | 0.728 | 0.995 | 0.711 | 0.725 | 0.995 |
| 20 | 0.718 | 0.771 | 0.996 | 0.732 | 0.773 | 0.996 |
| 50 | 0.860 | 0.876 | 0.999 | 0.874 | 0.883 | 0.999 |

This supports a modest and useful estimator observation: at low label budgets the direction components are the noisy part, while baseline rate is already near perfectly repeatable. It does **not** establish that a shared correction network can exploit that noise predictably across sessions.

## Required next action

Record Step-2 as **negative/inconclusive for a shared reliability-conditioned correction** and do not launch Step-2B. Preserve the receipt as the causal budget/noise characterization. The next eligible item is the separately specified Step-3 CPU Gate A, with its own source-only, fixed-K, shuffled-slot controls and no decoder or formal-test escalation unless that gate passes.
