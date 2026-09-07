"""SLOT-AUDIT V1 — the 50 already-supervised output slots, zero training.

Work order: ``docs/WORKORDER_SLOT_AUDIT_V1_20260829.md`` (binding); purpose
and reading discipline: ``docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_
AUDIT_20260829.md`` sections 2.2 and 5.

Submodules:
- ``plan``          constants, sealed anchors, the pre-registered reading law;
- ``cache_store``   the npz cache + cumulative sha256-bound manifest;
- ``scoring``       house variance-weighted R2 (torch) + numpy float64 replica;
- ``alignment``     the absolute-bin alignment law, per-slot calibration rows,
                    per-delta redundant-estimate residual curves;
- ``ensembles``     slot-subset ensembles, source-selected slot, readings;
- ``trajalign``     trajectory-alignment re-measurement THROUGH the frozen
                    ``src/continuity_probe_v1`` law (imported, never rebuilt);
- ``targets``       the session behavior-bin target source (parse + anchor);
- ``materialize``   stage 1 (CPU decode loop, parity anchors, cache write);
- ``audit``         stage 2 (pure statistics on the cache) + terminal receipt.
"""
