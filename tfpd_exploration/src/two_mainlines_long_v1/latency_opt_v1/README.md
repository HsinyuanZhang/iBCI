# latency_opt_v1

Queued from `DESIGN_M2_H1_TEMPORAL_DECODER_LATENCY_OPTIMIZATION_V1_20260905.md`.

Do not implement against sealed `evalai_m2_*` or H1 training trees.
Start at E0 baseline in a new result root when the coordinator releases CPU
(H1/M1 train load down). Formal CPU benches are single-flight.
