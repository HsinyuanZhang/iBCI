# APFC V1 source-only capacity screen

This work order binds the frozen APFC design
`DESIGN_ANCHORED_POSTFUSION_CAPACITY_SCREEN_V1_20260902.md`.

- One process, physical GPU0 only (`CUDA_VISIBLE_DEVICES=0`); GPU1 is out of
  scope and must not be queried, initialized, or scheduled.
- Reuse one PIT Selected-T4 POOLED source DataModule and the G00m-linear
  source activity authority.  No target surface is opened.
- Run exactly A-S1, A-TB4, and A-DC2 for 12 epochs with matched source
  batches, pool cycle `(4,10,30)`, seed 42, frozen inherited branches, and
  Adam only over declared gate parameters.
- Select only from the lexical 5/2 source validation records; fail closed if
  the scalar control or either capacity gate violates the frozen design gate.
- Publish immutable attempt, launch, source authority, screen, and exactly
  one terminal-or-failure receipt.  No target result is authorized.
