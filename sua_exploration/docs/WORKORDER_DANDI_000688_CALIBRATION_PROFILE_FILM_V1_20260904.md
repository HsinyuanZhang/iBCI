# Workorder: DANDI 000688 Calibration-Profile FiLM V1

Status: `EXECUTION_AUTHORITY__SOURCE_ONLY__GPU0`

Governing design:
`sua_exploration/docs/DESIGN_DANDI_000688_CALIBRATION_PROFILE_FILM_V1_20260904.md`

## Required sequence

1. Verify the strict 27/6/6 manifest, teacher, selected seed checkpoint, source
   metadata, code closure, empty result root, and isolated GPU0.
2. Publish an immutable attempt receipt before model/DataModule/CUDA access.
3. Prepare exactly one C30, T4@30 source DataModule. It must resolve only the
   27 train and 6 validation files named by the strict manifest.
4. Build M10/M30 low/high-speed profiles and Q50 query starts. Fail closed on
   any nonfinite, degenerate, roster, shape, chronology, or query mismatch.
5. Strict-load the final epoch-11 parent checkpoint and freeze every base
   parameter. Construct four identical zero-init FiLM heads.
6. Prove native/FiLM identity and prediction bitwise equality before the first
   optimizer step.
7. Train `CP10`, `CP30`, `SHUFFLE10`, and `EMPTY` together for exactly 12
   epochs on GPU0, with one input transfer shared by all arms.
8. Score the frozen evaluation matrix on every Q50 validation window, compute
   paired session deltas and the frozen decision law, and write terminal or
   prefix-preserving failure receipts.
9. Run seeds 43/44 only when seed42 passes the noninferiority allocation gate.

## Frozen constants

- Activity/T4 support: 30 chronological rewarded trials.
- Profile horizons: 10 and 30.
- Query: rewarded trials `[50:]` only.
- Profile mask: `[1,1,0,0]`.
- FiLM rank: 8; trainable parameters: 1,224.
- Epochs: 12; batch: 32; LR: 3e-4; Adam; no weight decay.
- Per-source-session epoch cap: 1,024 Q50 windows.
- Seeds: 42, then 43/44 only after seed42 noninferiority.
- Formal-test access, target update, EvalAI push: false.

