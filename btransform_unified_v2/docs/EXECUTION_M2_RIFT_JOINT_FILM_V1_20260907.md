# M2 RIFT joint FiLM B/D CPU preflight

The new joint cell uses `HoldContrastFiLMEarlyPoolEncoder(100,50,64,side8,rank8,num_post3,t4_plus_contrast)`, loading local `film_states.pt:p0` and `selected_head.pt` with the canonical overlay. It recomputes M33 identities by FP32 chronological `reset_stream/push_trial/finalize_identity`, retaining an autograd path through all encoder parameters.

Arm B applies `side8=zeros` and direct carrier zeros. Arm D uses `[MOVE-T4, zeros]` and the real MOVE-T4 direct carrier. Both use a fresh RIFT proj-add decoder at seed 42; gradients open only source-seven data. Ext4 is a declared visible development selection surface, not a hidden-test result.

The provider parity receipt covers one source-seven and one ext4 session against frozen E0 and records encoder gradient flow. Both arm CPU one-update preflights completed with finite loss.

## Formal run and selection contract

Formal training has exactly 24 epochs and 3,165 source-only updates per epoch (75,960). Each checkpoint contains the full joint encoder/decoder state, optimizer, EMA shadow state, CPU/CUDA RNG state, exact arm, seed, temporal configuration, and source/cache hashes. Resume accepts only a checkpoint directly under the same formal destination whose metadata, arm, hashes, and configuration match, and rejects completed or smoke checkpoints.

Post-training scoring reloads every epoch's full joint checkpoint, applies its EMA to both encoder and decoder parameters, re-materializes all four ext4 M33 identities through chronological FP32 provider calls, and performs the full 2,069-window ext4 scan. It requires finite pooled/equal-session R² and every exact four-session window count before writing progress. A stored epoch score must retain the identical checkpoint SHA; SHA drift is rejected. Selection is the earliest maximum equal-session mean across epochs 1–24.

The live streaming `frontend_last` also calls the trained identity provider and never falls back to frozen `TaskBank.E0` or carrier. After one real CPU update per arm, startup-zero, left-masked R50 full/stream checks passed: B max absolute difference `4.47e-7`; D `2.38e-7`.
