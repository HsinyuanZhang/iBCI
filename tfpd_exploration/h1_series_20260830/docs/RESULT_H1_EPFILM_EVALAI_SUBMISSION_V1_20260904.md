# H1 EP-FILM EvalAI Submission V1

Date: 2026-09-04
Status: `FINISHED_POSITIVE_VS_LP_ANCHOR_BELOW_FROZEN_C1_AND_CAL_AUG_LINE`

## Decision

Submit the all-source EP-FILM deployment candidate (frozen C1/M3 carrier-aware
SPINT decoder + early-pooling calibration-profile FiLM, H=32, 648 parameters,
plus the inherited MAT7 readout calibration), cached identities,
`IsTestTimeAdaptive=false`.  Pre-registered purpose (WORKORDER V3 §5): the
official FiLM×pooling quadrant completion on hidden data, explicitly not a
predicted score improvement.  It does not replace LP-R3 as this line's sealed
deployment candidate unless it officially exceeds it.

## Package and container authority

- package/payload SHA-256:
  `df71cb9329a87b5073242044b2933391ced7bf866d1f481e994f71dbd97ba2d7`;
- substrate checkpoint SHA-256 (frozen C1 epoch-49, identical to the cal-aug
  line's published artifact):
  `0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06`;
- frozen decoder model-state SHA-256:
  `bdaf7dbcbae75ea307f20356aaf80066586f7d9afa273712a5e34708b903eb85`;
- all-source EP-FILM state SHA-256:
  `b602c2e090f455cb6259fc76fb9a225bfdb9aa6bf40adcd5ccd10b9a0da13b37`;
- readout selection SHA-256:
  `08169c6c8e3ead47550dd3426930498a45ba7c789c13aec83422f76e24177e18`;
- source authority SHA-256:
  `a92b57350f2dcb04027bb6d848e5582d84e1bef4bf507d6844963ccad3c87bd5`;
- calibration authority SHA-256:
  `ea4b57879afa84508d6e5d89e8047a989bbfe5fddfc28aa6f8473c532d868cfe`;
- final image ID:
  `sha256:3180ac0b2ac117f77dda789245ce63c3ce9aab09509b81dc969856a650d97c7b`;
- private ECR tag: `15d1d8a7-8321-4c2d-b0ff-6e26d28e9bbe`
  (local tag `h1-epfilm-c1:evalai-v1-df71cb93`).

Container CPU smoke `PASS_H1_EP_FILM_CONTAINER_SMOKE`; host/container minival
R2 parity within `1.3e-7` (tight threshold `1e-4`); container minival
Held In R2 Mean `0.97398`.  All-source in-sample eval: session-mean delta
`+0.00486` over the zero-init anchor, preregistered overfit signal absent
(ratio 0.203 < 2.0).  V2 LODO evidence going in: EP-FILM `+0.0239` (4/5
dates), local LODO EP-FILM `0.4299` vs LP-R3 `0.4460`.

## EvalAI registration and terminal result

- challenge: `2319`; phase: `4599` / `few-shot-test-2319`; team: `HKU-ECE` (41975);
- submission ID: `581866`; visibility: private;
- submitted at: `2026-09-04T11:43:54.998230Z`;
- evaluated: `2026-09-04T12:02:06Z`, execution `0.128 s`;
- terminal status: `finished`;
- stderr: benign only (hdmf namespace + dataloader-worker warnings); evaluator
  found 27 files and completed both eval batches; no error;
- official result JSON SHA-256:
  `c666a198f681690c7a6c773f1f26e6a5239f4d7f30a3589560d19bb66f8bc6f0`;
- official stdout SHA-256:
  `85de0dc4fce62a670df36aaaac4b7fd050ead64a27f983c8cbe5225bb0089f5e`.

Official `test_split_h1` metrics:

| Metric | Value |
|---|---:|
| Held Out R2 Mean | `0.2675401890907159` |
| Held Out R2 Std. | `0.23549154768685449` |
| Held In R2 Mean | `0.4514773616359327` |
| Held In R2 Std. | `0.03356742236271655` |
| Normalized Latency | `0.038153259882001696` |

## Matched official comparison

The team runs two EvalAI accounts for quota; the cal-aug line owner registered
581747/581748 and 581812-581814 outside HKU-ECE.  Their sealed
`official_results.json` (branch
`exp/h1-cal-aug-m3-aware-dual-selection-v2-evalai-a1`) is reproduced here as
the comparison frame.  The 2026-09-04 handoff's H1 champion table predated
awareness of 581812-581814 and is superseded by this section.

| Arm (line) | Submission | HO R2 Mean | HO Std | HI R2 Mean | vs 581866 |
|---|---|---:|---:|---:|---:|
| T0 (cal-aug) | 581747 | 0.2411 | 0.1157 | 0.3557 | −0.0264 |
| LP-R3/M3RC (ours) | 581792 | 0.2410 | 0.2452 | 0.4448 | −0.0266 |
| C1, no selection (cal-aug) | 581748 | 0.2841 | 0.1348 | 0.4587 | +0.0166 |
| **EP-FILM (ours, this doc)** | **581866** | **0.2675** | **0.2355** | **0.4515** | — |
| C2-E49, no selection (cal-aug) | 581812 | 0.3050 | 0.1305 | 0.4847 | +0.0375 |
| C2-HI-E45 (cal-aug) | 581813 | 0.3240 | 0.1252 | 0.4926 | +0.0565 |
| C2-HO-E15 (cal-aug) | 581814 | 0.3760 | 0.1258 | 0.5112 | +0.1085 |

## Interpretation

1. Pre-registered criterion vs LP-R3: exceeded (`0.2675 > 0.2410`, `+0.0266`),
   so EP-FILM replaces LP-R3 as this line's sealed deployment candidate.
2. Versus its own frozen substrate (C1, `581748`): `−0.0166`.  The additive
   package (MAT7 readout calibration + EP-FiLM) does not recover the untouched
   substrate.  On this chain the readout calibration alone cost `−0.043`
   (`581792`), and EP-FiLM recovered `+0.0266` of that; FiLM's isolated effect
   officially remains unmeasured (no EP-zero arm was submitted), so "FiLM is
   bad" is not supported — "the additive stack is net negative vs C1" is.
3. Stop rule (HANDOFF_H1_SUCCESSOR_AGENT_20260903; DESIGN
   _H1_CAUSAL_ACTIVITY_COMPLETION_V1_20260903): official HO gain vs C1
   `< +0.020` → stop the H1 accuracy line.  Condition met (`−0.0166`).  The
   line stays closed; reopening is a user decision.
4. Dominance of the cal-aug retraining line: 581866 sits below even C2-E49
   (`0.3050`, no selection, `−0.0375`).  On H1, training M3-awareness into the
   decoder body (prefix-cycle M3 batches) dominates adding modules on a frozen
   substrate.  C2-HI-E45 (`0.3240`, epoch selected on held-in surface only)
   leads 581866 by `+0.0565` with no held-out data involved.
5. Local-surface lesson: our LODO/minival surface overpredicts by ~0.16 and
   flipped the EP/LP ranking; it is not a reliable H1 development surface.
   The cal-aug line's HO-M3 surface (held-out-calibration recordings S6-S12,
   which are part of the official 27-file test set) tracked official closely
   (local session-mean `0.4057` → official `0.3760`) and its epoch selection
   (E15) transferred.  Their preregistration itself labels that surface
   "development/model-selection, not untouched held-out generalization";
   whether selecting on local copies of official test recordings is in-scope
   for the challenge is a team-level rules question, recorded here factually.
6. Allowed claims: official FiLM×pooling quadrant completion within the
   frozen-C1 substrate family only.  No claim that FiLM improves H1 SOTA; no
   claim that readout calibration helps; H1 team SOTA is the cal-aug line
   (`C2-HO-E15`, `0.3760`).

## Successor notes

- Any future FiLM work must re-anchor on the best substrate (C2 checkpoints)
  and pass the zero-init bit-exact anchor first; under the stop rule this
  requires explicit user authorization.
- A deployment-meaningful open question stays: `EP-FiLM + no readout layer` on
  the C1 substrate would isolate FiLM officially, but spending a submission on
  it is not justified while the cal-aug line dominates.
