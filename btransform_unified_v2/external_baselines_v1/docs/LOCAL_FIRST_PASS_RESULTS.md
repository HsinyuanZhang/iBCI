# Local first-pass external gradient-free controls

## Status and scope

The CPU runs completed in priority order **M2 → M1 → H1**. They are retained historical local public-calibration screens, not formal benchmark results: all receipts say `official_test_used: false`, `target_labels_used_for_fit: false`, and `target_backprop_used: false`.

**Frozen protocol notice.** The archived protocol uses one fixed source session and `ridge=1`, without an official-preprocessing matched control. It is therefore `FROZEN_PROTOCOL_INVALID_FOR_COMPARISON`: none of the numbers below is a fair RIFT comparison, an EvalAI candidate, evidence of an aligner effect, or evidence about RIFT gains. Preserve the receipts and values for provenance only. The replacement plan is [fair_v2](../fair_v2/).

Each task uses one latest fixed source session, a causal 10-bin ridge Wiener decoder (`ridge=1.0`), and the fixed first-pass adapter settings `fa_dim=10`, `fa_max_iter=1000`, `fa_n_init=3`, `fa_stable_fraction=0.5`, `coral_ridge=0.001`, `coral_shrinkage=0.1`.  The matched controls are documented in the [external-baselines README](../README.md): raw WF, source-FA+WF (`fa_wf`), target-FA/closed-form Procrustes+WF (`aligned_fa_wf`), and linear-CORAL+WF (`coral_wf`).  The source decoder uses source labels; target adapters receive target neural calibration only, never target behavior labels, and use no target backpropagation.

| Task | Fixed source session | CPU wall time | Held-out local windows | Receipt |
|---|---|---:|---:|---|
| M2 | `ses-2020-10-28-Run1` | 30.91 s | 15,403 | [corrected metrics](../results/m2_full_v1_corrected/receipt.json), [model/runtime diagnostics](../results/m2_full_v1/receipt.json) |
| M1 | `ses-20120928` | 81.78 s | 3,881 | [receipt](../results/m1_full_v1/receipt.json) |
| H1 | `ses-19250120T115537` | 21.69 s | 33,613 | [receipt](../results/h1_full_v1/receipt.json) |

## M2: historical first-pass values

M2’s corrected receipt reports both historical FULL-compatible flattened R² and standard per-output centered variance-weighted R².  They use different denominators and are displayed side by side, never subtracted across columns.

| Method | FULL-compatible flattened equal-session mean | Standard variance-weighted equal-session mean | FULL-compatible flattened pooled R² | Standard variance-weighted pooled R² |
|---|---:|---:|---:|---:|
| Raw WF | −0.16936 | −0.16937 | −0.14061 | −0.14062 |
| Source FA + WF | −0.00668 | −0.00669 | −0.00847 | −0.00848 |
| AlignedFA + WF | −0.15063 | −0.15064 | −0.13849 | −0.13850 |
| CORAL + WF | **0.08296** | **0.08296** | **0.09270** | **0.09269** |

In this archived table, CORAL has the numerically highest M2 row. That ordering is not a candidate recommendation or an aligner-effect conclusion because the protocol is frozen as invalid for comparison. M2's fixed electrode/channel coordinates remain a data-description fact, not stable-cell evidence; see [CHANNEL_COORDINATES.md](CHANNEL_COORDINATES.md).

## M1: historical first-pass values and FactorAnalysis convergence record

| Method | FULL-compatible flattened equal-session mean | Standard variance-weighted equal-session mean | FULL-compatible flattened pooled R² | Standard variance-weighted pooled R² |
|---|---:|---:|---:|---:|
| Raw WF | −1.64311 | −2.37074 | −1.69012 | −2.45307 |
| Source FA + WF | −1.58309 | −2.32993 | −1.59336 | −2.32886 |
| AlignedFA + WF | 0.47964 | 0.32539 | 0.48117 | 0.33402 |
| CORAL + WF | **0.49715** | **0.35115** | **0.49649** | **0.35369** |

The M1 row ordering is historical only and cannot establish an alignment improvement under this frozen protocol. Both the shared M1 source FA fit and target session `20121017` touch the first-pass `fa_max_iter=1000` budget with `converged: false`. The `20121017` final log-likelihood improvement is `0.002351` (above the configured tolerance `0.001`, but only about `3.76e-9` relative to the log-likelihood magnitude); parameters stayed finite and noise variances positive. M1 has the same fixed-electrode-only coordinate basis as M2, not same-cell evidence.

## H1: exploratory positional-row screen; all methods remain poor

H1’s main metric is **`grouped_seven.r2_mean`**, grouping the 14 recordings into seven sessions.  The 14-recording `spint_recording_mean_r2` is diagnostic only and is not the main result.

| Method | grouped-seven R² mean | grouped-seven population SD | 14-recording diagnostic mean |
|---|---:|---:|---:|
| Raw WF | −8.32531 | 0.79923 | −8.32616 |
| Source FA + WF | **−0.07171** | **0.03330** | −0.07185 |
| AlignedFA + WF | −0.09968 | 0.03997 | −0.09963 |
| CORAL + WF | −4.73425 | 0.12615 | −4.73488 |

All H1 values are historical under the frozen protocol. H1 uses common delivered row positions but lacks a verified cross-date physical channel map. `physical_channel_correspondence_unverified` applies; see [CHANNEL_COORDINATES.md](CHANNEL_COORDINATES.md).

## Replacement work

Do not prioritize, submit, or extend any archived v1 row. The new fair matched experiment belongs in [fair_v2](../fair_v2/); it must supply the required official-preprocessing control before comparison with RIFT.
