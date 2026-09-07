# C1 — teacher-domain ablation contract

**Date:** 2026-08-13
**Status:** pre-registered CPU scaffolding only. **Authorizes no GPU run, no
training, and no target or sealed-session access.** Root may launch the 2×2
only after a source-only teacher compatibility receipt records that a
CO-native teacher can be constructed and loaded into the existing student
contract.
**Screen ID:** `c1_teacher_domain_ablation_v1`
**Contract document SHA-256:** bind at launch via `contract_sha256` in the inert
runner; recompute with `sha256sum` on this file before any authorization.

**Collision note.** This is not the historical paired-view C1 (native-MUA /
shared-Zero4) line. Those artifacts remain sealed and unused here.

---

## Claim under test (must be able to fail)

The current SUA/A2 student copies an MC-Maze-derived decoder initialization
even under `loss_mode=task_only`. Teacher/target mismatch is a **plausible
contributor**, not an already isolated cause (handoff §5.2 item 3). This
experiment tests whether replacing that initialization with a CO-native
teacher, while holding the W-add interface and sampling fixed, improves
absolute external T4 and a matched T4/Z4 interaction.

A positive result does **not** prove that MC-Maze mismatch was the sole limiter.
A negative result drops the printable claim that the inherited teacher domain
is currently costing a few points of external T4. Combining CO-native with
H-add (A1) in the first arm is forbidden: the two factors would be
unattributable.

---

## Frozen matrix

| Factor | Levels | Held fixed |
|---|---|---|
| Teacher domain | `mc_maze` · `co_native` | — |
| Carrier | `t4` · `z4` (`mask_standardized_t4`) | — |
| Add site | **W-add only** | H-add is A1; out of scope |
| Sampling | legacy window-count | C2; out of scope |
| Loss mode | `task_only` | no distillation-term change |
| Decoder | coupled, trainable | no selected-T4 encoder warm-start |

**Source-training cells:** `2 teachers × 2 carriers × 3 seeds = 12` fresh runs.
Each unchanged checkpoint is scored on both `within_subject` (sub-C 6 val) and
`external_subject_M` (15 admissible sub-M sessions). Domain retraining is
forbidden.

Seeds `{42, 43, 44}`. Variant `B3S`. Protocol otherwise matches A2 v2:
M30/trial-30, 12 epochs, no early stopping, checkpoint mean of epochs 5–12.

### Frozen student contract (both teachers)

| Parameter | Frozen value |
|---|---|
| Lightning `task` string | `mc_maze` (existing student constructor; not a data-domain claim) |
| `window_size` / trial length | `50` / `100` |
| Identity MLP input | materialized `Linear(100, 512)` |
| `fc_in` | `Linear(50, 512)` then ReLU then `Linear(512, 512)` — N-free |
| `num_covariates` | `2` |
| `model_dim` / heads / layers / id layers | `512` / `64` / `1` / `3` |
| Encoder warm-start | **none** |
| `freeze_decoder` | `false` |

The student copies `teacher.state_dict()` into a fresh `SpintModel` with
`strict=True` inside `StreamingCalibrationLitModule.setup` **regardless of
`loss_mode`**. B3S does not copy `fc_id_in` / `fc_id_out` into the encoder
(`copy_teacher_id_weights` is a no-op for `SideFeatureEarlyPoolEncoder`).
Section 5.2 item 3 is therefore about decoder initialization, not an explicit
selected-T4 encoder warm-start.

---

## Source-only teacher compatibility receipt (root gate)

**This is the execution gate for C1, not a result.** It must run first, on CPU,
without opening target (sub-M) NWBs or the six sealed sub-C test sessions, and
without training.

It must establish:

1. whether a CO-native teacher **checkpoint of the existing student
   architecture** can be constructed at all;
2. whether that checkpoint **loads** into `StreamingCalibrationLitModule`
   under the frozen student contract above;
3. **exactly** what differs from the current MC-Maze decoder initialization.

It must **not** assume that those differences cause the remaining accuracy
gap. Differences are recorded so the 2×2 can test them.

A trained CO-native teacher does not exist in this scaffolding. The receipt
proves interface constructibility with a matching-architecture synthetic
checkpoint whose weights are not a scientific initialization. GPU source
training remains unauthorized until a later reviewed binding names a real
CO-native teacher trained **only** on the 27 sub-C source-train sessions.

CO-native construction spec (future training; not executed here):

| Field | Required value |
|---|---|
| Data | DANDI 000688 sub-C CO, strict 27-session **train** roster only |
| Behavior | `cursor_vel` (2-D); `num_covariates=2` |
| Units | variable N, `max_units_exclusive=100`; decoder weights remain N-free |
| Architecture | bitwise the production SpintModel hyperparameters above |
| Lightning wrapper | `FalconLitModule` with `task="mc_maze"` so the existing student loader still works |
| Forbidden | sub-M, sealed six, validation-as-train, H-add, equal-session sampling, selected-T4 warm-start |

---

## Frozen acceptance rule

Primary quantities (handoff §5.2 dual reporting):

```text
external_T4_lift[s] = meanR2(CO-native, T4, external, s)
                    - meanR2(MC-Maze,  T4, external, s)

external_Z4_lift[s] = meanR2(CO-native, Z4, external, s)
                    - meanR2(MC-Maze,  Z4, external, s)

external_interaction[s] = (T4 - Z4)_CO-native,external,s
                        - (T4 - Z4)_MC-Maze,external,s

within_T4_delta[s] = meanR2(CO-native, T4, within, s)
                   - meanR2(MC-Maze,  T4, within, s)
```

`teacher_domain_external_t4_effective` if and only if **all** hold:

1. mean external T4 lift `>= +0.03`;
2. all three seed external T4 lifts `> 0`;
3. Z4 does **not** reproduce the lift: it is false that (mean Z4 lift `>= +0.03`
   and all three seed Z4 lifts `> 0`);
4. mean external teacher×carrier interaction `>= +0.03`;
5. all three seed interactions `> 0`;
6. within-subject T4 non-inferiority: mean `within_T4_delta >= -0.03`.

Absolute external T4 is required so a collapsing Z4 cannot mint a deployment
win. The interaction is required so a generic optimization improvement cannot
be reported as greater use of carrier content. The Z4 sibling is the
generic-lift control. Within-subject non-inferiority blocks a transfer win
that damages the development domain.

Bootstrap intervals, if later added, are descriptive only. A three-seed
Wilcoxon is prohibited.

### Synthetic attainability (frozen before any run)

The aggregator ships one synthetic matrix that must pass and one that must
fail, and the tests assert different verdicts. Current examples:

- **Must pass:** external T4 lift `+0.06` on all seeds, Z4 lift near zero,
  interaction `+0.06`, within T4 non-inferior.
- **Must fail (generic lift):** T4 and Z4 both lift `+0.06` — absolute T4
  would pass, but Z4 reproduces the lift and the interaction is ~0.
- **Must fail (Z4 crash):** T4 unchanged, Z4 collapses — `T4-Z4` inflates
  while absolute T4 lift is 0.

---

## Data isolation boundary

**Never load, score, or reference these sealed formal-test sessions:**

- `sub-C_ses-CO-20151113`
- `sub-C_ses-CO-20151116`
- `sub-C_ses-CO-20151117`
- `sub-C_ses-CO-20151119`
- `sub-C_ses-CO-20151120`
- `sub-C_ses-CO-20151201`

The compatibility receipt and CPU preflight open **no NWB**. They may read
the strict manifest JSON for names only. Sub-M is score-only in a future GPU
matrix and is out of scope for the source-only compatibility gate.

---

## Outcome authorization map

| Outcome | Authorizes | Forecloses |
|---|---|---|
| Compatibility receipt fail | nothing | C1 GPU until a repaired CO-native teacher contract exists |
| Compatibility receipt pass, no trained teacher | CPU scaffolding only | GPU |
| 2×2 gate pass (future) | teacher-domain claim as a contributor | treating it as the isolated cause of the native remainder |
| Absolute T4 lift with equal Z4 lift | generic teacher improvement only | carrier-specific interpretation |
| Interaction from Z4 collapse only | nothing | deployment win |
| Gate fail | drop the printable inherited-teacher-penalty claim | not A1, not C2, not CF1 |

---

## Implementation bindings

| Artifact | Path |
|---|---|
| Core / gates | `sua_exploration/mc_maze/c1_teacher_domain_ablation.py` |
| Compatibility probe | `sua_exploration/mc_maze/c1_teacher_compatibility.py` |
| Compatibility receipt CLI | `sua_exploration/scripts/write_c1_teacher_compatibility_receipt.py` |
| CPU preflight | `sua_exploration/scripts/c1_teacher_domain_ablation_preflight.py` |
| Inert runner | `sua_exploration/scripts/run_c1_teacher_domain_ablation_one_cell.sh` |
| Fail-closed aggregator | `sua_exploration/scripts/aggregate_c1_teacher_domain_ablation.py` |
| Config | `sua_exploration/configs/c1_teacher_domain_ablation.json` |
| Tests | `sua_exploration/tests/test_c1_teacher_domain_ablation.py` |

**This document authorizes nothing.** The inert runner refuses `--launch`.
No GPU, no training, no sealed-session access.
