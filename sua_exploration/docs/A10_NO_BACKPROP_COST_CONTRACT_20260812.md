# A10 — matched cost of the no-backprop constraint contract

**Date:** 2026-08-12  
**Status:** **SUPERSEDED / INERT DESIGN RECORD.** **Authorizes no GPU run.** The v1 matrix is not a matched
measurement of the cost of backpropagation and must not gate source-trained, target-time zero-backprop
architectures.  
**Screen ID:** `a10_no_backprop_cost_v1`  
**Contract document SHA-256:** bind at launch via `contract_sha256` in the inert runner.

> **Root audit, 2026-08-13.** The three arms do not receive matched supervision: `gradient_free` fits a
> four-value carrier from one direction annotation per calibration trial, whereas the proposed probe and
> fine-tune arms use behaviour targets for weight updates and do not define an equal-annotation sparse loss.
> The implementation is also intentionally incomplete: the non-gradient-free cell entry point terminates
> without producing a scientific result. Therefore the nine-cell matrix below is historical design text,
> not a frozen executable contract. A future replacement must separate (i) a label-richer dense supervised
> upper bound from (ii) an equal-annotation comparison, if the latter can be defined without changing the
> prediction target. A10 may prioritize target-session readout/fine-tuning or other target-time parameter
> adaptation; it is not an upper bound on a new source-trained interface whose deployment still uses no
> gradients or weight updates.

---

## Claim under test (must be able to fail)

On the frozen SUA sub-C M30 mainline, how much R² is forfeited by forbidding backward passes and label-using weight updates during held-out-session calibration, relative to matched readout-only and full fine-tuning on the **same** calibration prefix, labels, query windows, and validation sessions?

---

## ⚠️ Numbers that are NOT the baseline (do not quote as comparable)

The following come from `P3_CROSS_SESSION_ANALYSIS.md` and **must not** be cited as matched baselines for this experiment:

| Hint | Value | Confound |
|---|---:|---|
| Within-session end-to-end ceiling (B3, 80/20) | `0.6937` | Different split; not gradient-free deployment |
| Labelled encoder-only finetune oracle K=20 | `+0.692` vs zero-shot `−0.122` | 4× unit-count regime jump; held-out labels; backward pass |
| Best gradient-free cross-session carrier (subject-M) | `0.356828` | Different subject, protocol, and estimand |

**The superseded A10 v1 design does not replace these hints.** No matched authoritative measurement currently
exists.

---

## Frozen matrix

One shared source checkpoint per seed (T4 B3S trained on sub-C 27 sessions under mainline protocol). Three adaptation arms per seed on the **same** six validation sessions.

| Arm | Backward pass | Trainable parameters during calibration | Behavior labels for weight updates |
|---|---|---|---|
| `gradient_free` | none | none | none |
| `readout_probe` | yes | decoder only | yes (calibration prefix only) |
| `full_finetune` | yes | encoder + decoder | yes (calibration prefix only) |

| Cell ID | Arm | Seeds | Scoring sessions |
|---|---|---|---|
| `gradient_free` | frozen forward-only | 42, 43, 44 | sub-C 6 val |
| `readout_probe` | decoder probe | 42, 43, 44 | sub-C 6 val |
| `full_finetune` | full adaptation | 42, 43, 44 | sub-C 6 val |

**Total GPU cells:** `3 arms × 3 seeds = 9` adaptation evaluations.

Source training may reuse sealed mainline T4 checkpoints when SHA-bound; otherwise `3` additional source-training cells are required (not counted in the 9 adaptation cells).

### Frozen protocol (identical across all arms within a seed)

| Parameter | Frozen value |
|---|---|
| Source variant | B3S + `t4` side features |
| Activity calibration | chronological first `30` trials |
| T4 label pool | chronological first `30` rewarded trials |
| Query / evaluation | trials `[30:]` |
| Source training epochs | `12`, no early stopping, epochs `5–12` mean |
| Adaptation data scope | calibration prefix trials `[0:30]` only |
| Adaptation evaluation | trials `[30:]` only (disjoint from adaptation inputs) |
| Adaptation epochs | `20` (fixed; no validation argmax) |
| Adaptation learning rate | `1e-4` |
| Adaptation batch size | `32` |
| Split manifest | `configs/subc_co_27_6_strict_train_val_manifest.json` |
| Teacher | mainline teacher checkpoint (SHA-bound) |
| Formal test NWBs | **never opened** |

`gradient_free` arm must record `uses_backward_gradients=false` and `uses_behavior_labels_for_weight_updates=false`. Backprop arms must record both as `true`.

---

## Frozen acceptance rules

### Historical headroom gate (inactive)

Define per seed:

```text
headroom_readout[s] = R2(readout_probe,s) − R2(gradient_free,s)
headroom_full[s]     = R2(full_finetune,s) − R2(gradient_free,s)
best_headroom[s]     = max(headroom_readout[s], headroom_full[s])
```

| Gate name | Condition | Verdict if pass |
|---|---|---|
| `headroom_large` | `mean(best_headroom) ≥ +0.03` AND all three `best_headroom[s] > 0` AND paired `σ` gate: `mean − 2·σ_paired > 0` | **Large headroom** — decoder redesign (B13–B16) authorized for separate review |
| `headroom_small` | `mean(best_headroom) + 2·σ_paired < +0.03` | **Small headroom** — thesis confirmed; B13–B16 **foreclosed** |
| `headroom_indeterminate` | otherwise | No redesign authorization; report descriptive headroom only |

`σ_paired = stdev(best_headroom[s], ddof=1) / √3`.

### Secondary descriptive gates (FP32 mainline format)

| Contrast | Rule |
|---|---|
| `readout_probe − gradient_free` | five-gate FP32 mainline rule |
| `full_finetune − gradient_free` | five-gate FP32 mainline rule |

---

## Outcome authorization map

| Outcome | Authorizes | Forecloses |
|---|---|---|
| `headroom_small` | Strong no-backprop thesis; fusion nulls explained by constraint | B13 closed-form last-layer adaptation; B14 signed attention; B15 latent queries; B16 carrier-driven read-in |
| `headroom_large` | Bottleneck identification; separate review for B13–B16 | Claim that backward pass is unaffordable **without** quantified trade-off |
| `readout_probe ≈ full_finetune` | Decoder-limited; B13/B15 priority | Encoder widening |
| `full_finetune >> readout_probe` | Encoder adaptation matters | Readout-only redesign paths |

This inactive gate must not be used to block source-trained, target-time zero-backprop interfaces. A future
auditable A10 replacement may gate only branches whose premise is target-session parameter adaptation or a
quantified label-richer supervised upper bound.

---

## Data isolation boundary

**Sealed formal-test sessions (never load, score, or reference):**

- `sub-C_ses-CO-20151113`
- `sub-C_ses-CO-20151116`
- `sub-C_ses-CO-20151117`
- `sub-C_ses-CO-20151119`
- `sub-C_ses-CO-20151120`
- `sub-C_ses-CO-20151201`

---

## GPU cost

| Item | Count |
|---|---:|
| Adaptation cells | 9 |
| Optional fresh source trains (if mainline SHA reuse fails) | 3 |
| Estimated wall time | ~6–12 GPU-hours |

---

## Implementation bindings

| Artifact | Path |
|---|---|
| Preflight | `scripts/a10_no_backprop_cost_preflight.py` |
| Inert runner | `scripts/run_a10_no_backprop_cost_one_cell.sh` |
| Aggregator | `scripts/aggregate_a10_no_backprop_cost.py` |
| Reference scorer | `scripts/eval_adaptation_dandi688.py` (adaptation modes) |
| Authorization env | `A10_GPU_AUTHORIZATION=I_AUTHORIZE_A10_NO_BACKPROP_COST_GPU` |

**This superseded document authorizes nothing and must not be used to launch the inert runner.**
