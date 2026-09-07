# Identity encoder content/carrier closure — M2 and H1

Date: 2026-09-05

Status: `ID_ENCODER_GRID_COMPLETE__M2_AND_H1_OFFICIAL_CONFIRMATION_COMPLETE`

This receipt closes the identity-encoder variables requested before the paper:
carrier estimation window, calibration-profile semantic content, row alignment,
adapter capacity, pooling-position disposition, and three-seed stability.  The
separate decoder-family experiment in `FABLE0904_review1.md` §6 is explicitly
owned by the other-GPU decoder axis and is not duplicated here.

## 1. Governing results

### 1.1 M2, historical whole-trial T4 carrier

Matched official-product grid: first M33, `t4_plus_contrast`, 12 epochs,
learning rate `3e-4`, 1,224 trainable adapter parameters, seeds 42/43/44,
frozen base identity path and decoder.

| Arm | External equal-session R2, mean over seeds |
|---|---:|
| REAL hold/reach profile | 0.3172523 |
| EMPTY profile | **0.3221378** |
| fixed ROWSHUFFLE profile | **0.3352646** |

Key paired contrasts are calculated after averaging each session over seeds:

| Contrast | Mean | Positive sessions | Worst | Exact bootstrap 95% lower |
|---|---:|---:|---:|---:|
| REAL − EMPTY | **−0.0048855** | 2/6 | −0.01791 | −0.01396 |
| REAL − ROWSHUFFLE | **−0.0180123** | 0/6 | −0.04412 | −0.03060 |
| EMPTY − native P0 | **+0.0230049** | 5/6 | −0.01929 | **+0.00558** |

Seed 42 REAL reproduces the sealed local official-product result exactly:
`0.3210151173`.  Consequently, the official 581801 improvement is real, but
this final-grid content experiment does **not** attribute it to hold/reach
semantics or unit-row alignment.  It attributes the repeatable local component
to a small M33-trained, T4-conditioned adapter.

### 1.2 M2, corrected movement-window T4 carrier

The carrier-only intervention keeps the frozen checkpoint and M33 budget, but
fits T4 from raw calibration-neural bins `[5,30)` (`100–600 ms` after the
trial boundary).  Window geometry was selected using the seven held-in source
sessions before decoder scoring.  Deployment requires only raw calibration
neural counts, trial boundaries, and the same sparse target directions.

| Carrier | External equal-session R2 |
|---|---:|
| whole-trial T4 | 0.2991329 |
| MOVE-T4 | **0.3254516** |
| MOVE-T4 − whole T4 | **+0.0263187**, 6/6 positive |

This is the direct M2 counterpart to the 688 observation that a task-aligned
T4 estimator can absorb what initially looked like a need for profile
injection.  Submission 581899 confirms transfer to the hidden official stream:

| Official M2 method | Held-out R2 |
|---|---:|
| prior static T4 (578221) | 0.303244 |
| hold/reach FiLM (581801) | 0.320281 |
| **MOVE-T4, no FiLM (581899)** | **0.3275179** |

MOVE-T4 is approximately `+0.02427` over the prior static T4 submission and
`+0.00724` over the FiLM submission.  Thus the corrected sparse carrier is not
merely non-inferior: officially, it removes the need for FiLM to reach the best
score in this comparison.

### 1.3 MOVE-T4 × profile-content grid

The full three-seed content grid was repeated after carrier correction with the
same canonical zero-init state and matched schedule.

| Arm | External equal-session R2, mean over seeds |
|---|---:|
| MOVE-P0, no trained adapter | 0.3254516 |
| MOVE + REAL profile | 0.3426799 |
| MOVE + EMPTY profile | **0.3536106** |
| MOVE + fixed ROWSHUFFLE | **0.3591088** |

| Contrast | Mean | Positive sessions | Worst | Exact bootstrap 95% lower |
|---|---:|---:|---:|---:|
| REAL − EMPTY | **−0.0109307** | 1/6 | −0.02544 | −0.01908 |
| REAL − ROWSHUFFLE | **−0.0164289** | 2/6 | −0.04146 | −0.02856 |
| EMPTY − MOVE-P0 | **+0.0281589** | **6/6** | **+0.01448** | **+0.02140** |
| ROWSHUFFLE − EMPTY | +0.0054983 | 5/6 | −0.02030 | −0.00629 |

The carrier correction does not rescue profile semantics.  It makes the
capacity result cleaner: every external session benefits from a profile-free
adapter, while the real profile is worse than EMPTY.  Fixed row-shuffled input
can act as arbitrary unit/session code or regularization, but its gain over
EMPTY is not bootstrap-positive and must not be called semantic information.

## 2. Cross-dataset interpretation

The earlier `FABLE0904_review1.md` claims that M2 FiLM content was established
and that H1 content remained pending are superseded by the matched controls:

| Dataset | REAL/profile result | EMPTY/ROWSHUFFLE result | Supported interpretation |
|---|---|---|---|
| M2 whole T4 | REAL official product positive | REAL < EMPTY < ROWSHUFFLE locally | T4-conditioned M33 adapter; profile semantics unsupported |
| M2 MOVE-T4 | carrier alone +0.0263 | REAL < EMPTY; EMPTY +0.0282, 6/6 | correct carrier plus profile-free budget adapter |
| H1 C1/M3 EP | FULL +0.0239 | EMPTY +0.0254 (5/5), ROWSHUFFLE +0.0241 | M3-budget adaptation, not speed-profile content |
| 688 full T4 | CP approximately EMPTY | profile content null | carrier already covers the state channel |

The paper-safe common statement is therefore narrower and stronger:

> A zero-initialized conditional slot is a useful low-parameter calibration
> adapter, but improvement cannot be attributed to calibration-profile
> semantics without an EMPTY and row-alignment control.  In the tested M2 and
> H1 grids, the repeatable gain survives without profile content; task-aligned
> carrier estimation remains the interpretable information-bearing change.

FiLM may be described as the implementation of the slot, not as evidence that
hold/reach or speed-profile information was consumed.  The phrase “carrier has
a state gap, so FiLM supplies that state information” is no longer supported by
the final controls and must be removed from the main claim.

## 3. H1 official decomposition update

Submission 581900 removed only the MAT7 readout from the otherwise identical
581866 EP-FiLM package and finished successfully:

| H1 package | Official held-out R2 | Official held-in R2 |
|---|---:|---:|
| 581866, EP-FiLM + MAT7 | 0.2675402 | 0.4514774 |
| 581900, EP-FiLM + identity readout | **0.2980939** | **0.4619134** |
| 581900 − 581866 | **+0.0305537** | **+0.0104361** |

This validates the decision to remove MAT7 but does not reverse the H1 V5
content verdict.  H1 profile content remains unused; 581900 measures a cleaner
deployment stack, not semantic FiLM efficacy.

## 4. Completion audit for the identity-encoder lane

| Paper variable | Governing evidence | Status |
|---|---|---|
| H1 calibration-profile content | FULL / EMPTY / ROWSHUFFLE V5, five-date LODO | complete: content null, budget adapter positive |
| M2 final-grid profile content | three seeds, REAL / EMPTY / ROWSHUFFLE | complete: content and row alignment unsupported |
| M2 carrier estimation window | WHOLE versus MOVE-T4, frozen checkpoint and 581899 | complete: local +0.0263 (6/6); official 0.32752, above 581801 |
| carrier × profile interaction | MOVE-P0 / REAL / EMPTY / ROWSHUFFLE, three seeds | complete: real profile remains inferior to EMPTY |
| adapter capacity/budget effect | EMPTY versus P0 on both carriers | complete: repeatable; MOVE grid 6/6 and bootstrap lower bound > 0 |
| seed stability | seeds 42/43/44, paired schedules | complete |
| pooling position | H1 LP-R3 and LP-FiLM precedents already terminal; current winning path is early-pool | closed; no new late-pool run justified by this grid |
| decoder-family independence | `FABLE0904_review1.md` §6 | separate other-GPU decoder experiment; deliberately not duplicated |

The identity-encoder ablation grid and both requested official submissions are
complete.  The only remaining item in the broader paper package is the
separately owned decoder-family experiment; it is not an unfinished
identity-encoder variable and remains on the other-GPU lane.

## 5. Prepared next submission, not pushed

The strongest honest next M2 candidate is the seed-42 MOVE-T4 + EMPTY adapter,
not REAL or ROWSHUFFLE:

- local external R2: `0.3535113889`;
- 13/13 cached session identities;
- cached-versus-direct decoder max absolute difference: `0.0`;
- container finite-output smoke: pass;
- payload SHA-256:
  `4e4dae8f7239582a26d44cdb449e674710f28223523dd691dd4f8758b05220e0`;
- image ID:
  `sha256:caa98aefff18364df18bf2b5d4f143b9cb82beb8aaeeec1395faec134055752d`;
- status: `READY_NOT_SUBMITTED`.

It is intentionally described as a **profile-free T4-conditioned adapter**.
Submitting it can test whether the local capacity gain transfers, but cannot
support a profile-information claim.

## 6. Evidence bindings

- Whole-carrier content aggregate SHA-256:
  `fdc47e94f26964dd7d62eb72595dc7236013af4e4ec97ecb180cfea607679375`
- MOVE-T4 carrier score SHA-256:
  `295c3852b9d9b2c83690cc5fbbe401b863ee07c72a784e6f3ee534a022861541`
- MOVE-T4 content aggregate SHA-256:
  `dc425f88b07dd3c35fce9a0dfd1156cd4ddce172ae67be631ba05233ed924923`
- Prepared EMPTY payload receipt SHA-256:
  `af97e5f689d5c3ef72afa50034a3c3619f4c1ea6f027d328d5acde49927fd94b`
- H1 581900 official result SHA-256:
  `aba76c6021fc51b881c256b08b1278c89bd338b2f97003928e084bab32e45d11`
- M2 581899 official result SHA-256:
  `24b8391d33b353ab7bafbaf58c113c3042edc14ef3cffdb5eac0b567026226e5`
