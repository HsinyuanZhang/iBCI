# C2 Matched Deployment Score V1

Score the completed C2 final-four SWA once on the frozen within-6 and
external-15 deployment surfaces at M4, M10, and M30. Compare it with the
matched T0 checkpoint using the existing sealed Z1/P4 scoring engine.

The numerical execution closure contains runtime files only. Documentation
and tests are held in a separate review closure. Review-only drift is recorded
as `ACCEPTED_NON_NUMERIC_DRIFT` and never invalidates the numerical score.

No target training, backward pass, optimizer step, carrier update, or model
state mutation is allowed. The C2 and T0 producer pairs are revalidated before
and after scoring. The result root must be fresh.
