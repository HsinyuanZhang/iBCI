# Result — M2 Post-Fusion Variant Screen V1 (2026-09-02)

Status: **SOURCE-ONLY SCREEN COMPLETED; HELD-OUT SCORE COMPLETED WITH NO NOMINATION**.

This result closes the pre-registered three-arm, fixed-12-epoch training
screen. The subsequent 78-row local held-out score is reported in
`RESULT_M2_POSTFUSION_CHECKPOINT_SCORE_V2_20260902.md`: no arm passed the
literal V2 nomination gate. PF-MEAN is a clean external negative result;
PF-R1/PF-R50 require the operator-corrected, no-training rescore documented in
`AUDIT_M2_POSTFUSION_TRAIN_SCORE_OPERATOR_MATCH_20260902.md`. The source
selector here remains an explicitly in-sample monitor; no matched pre-fusion
T0 was trained and the official M2 submission surface was not evaluated.

## 1. Execution outcome

The GPU0 run completed naturally without retry:

```text
tmux                 m2_pf_screen_v1_20260902
physical GPU         0
GPU UUID             GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
epochs per arm       12
shared step groups   43,416
total wall           2,272.7556 s (37.88 min)
shared groups/s      19.1028
peak allocated       884,655,616 bytes
peak reserved        1,474,297,856 bytes
hard cap             10,800 s
terminal/failure     1 / 0
target access        false
matched T0 trained   false
```

All three models and optimizers remained independent while consuming the same
resident batch, chronological `(30,10,4)` calibration-prefix decision, and
paired Python/NumPy/Torch/CUDA RNG state before the DataLoader advanced. The
12-step smoke passed and the same epoch-1 iterator continued without replay or
skipping.

GPU utilization during read-only monitoring was generally about 90--96% while
the model footprint remained small (roughly 2.2 GiB process-level observed
usage). Thus low allocated memory did not indicate an idle GPU. One additional,
unattributed Python process briefly appeared on GPU0 for approximately 27
seconds and exited naturally. It was not controlled, signaled, or restarted by
this route. The main run remained healthy and deterministic receipt laws did
not report a violation. GPU1 and the independently owned M1 process/root were
not queried or modified.

## 2. Source monitor disclosure

The inherited source monitor is not validation or grouped OOF. Its seven
session names exactly equal the seven training session names, and all `1,011`
monitor coordinates are contained in the `115,911` training coordinates. The
only honest name is therefore:

```text
source_heldin_in_sample_monitor
```

It may select a source-best checkpoint independently within each arm and give
a descriptive ranking, but it is not evidence of source-session or held-out
generalization.

## 3. Source-only arm results

All three source-best checkpoints occurred at epoch 12:

| arm | added parameters | epoch-12 / best source monitor R2 | epoch11→12 slope | horizon flag |
|---|---:|---:|---:|---|
| PF-MEAN | 0 | 0.6350632 | +0.0008204 | true |
| PF-R1 | 1 | 0.6367036 | +0.0005371 | true |
| PF-R50 | 50 | 0.6333466 | +0.0009087 | true |

PF-R1 is numerically highest by `0.0016404` over PF-MEAN, but this lies within
the pre-registered `0.002` tie band. The fixed complexity tie-break therefore
names **PF-MEAN** the descriptive source winner. This ranking does not suppress
either residual arm: all three checkpoints must be scored once.

All endpoint slopes are small and positive. They record a possible 12-epoch
horizon limitation but do not authorize more epochs, checkpoint replacement,
or target-guided continuation. The matched comparison remains exactly 12
epochs for every arm.

### 3.1 What the residual gates learned

A post-terminal CPU-only, `weights_only=True` inspection of the immutable
checkpoint states initialized no CUDA and found:

```text
PF-R1:
  alpha              -0.7700281
  tanh(alpha)         -0.6469458

PF-R50:
  alpha range         [-0.7119517, -0.1777011]
  alpha mean          -0.4585296
  tanh(alpha) range   [-0.6118992, -0.1758539]
  tanh(alpha) mean    -0.4233312
```

All 50 PF-R50 coefficients are negative. Under
`h = h_native + tanh(alpha) * (h_pf - h_native)`, both residual arms therefore
learned an extrapolation away from the pure Post-Fusion identity, not a
positive interpolation toward it. This is a cautious mechanism signal: the
source objective did not simply prefer more Post-Fusion contribution.
Nevertheless, the source monitor is in-sample and the residual arms may still
regularize transfer differently. This observation cannot replace the fixed
78-row held-out score or terminate an arm before that score.

## 4. Immutable producer evidence

Canonical root:

```text
tfpd_exploration/results/m2_postfusion_variant_screen_v1/screen
```

The root has exactly 16 immutable leaves: eight regular `0444`, `nlink=1`
bodies and their exact basename-bound SHA-256 sidecars. It contains terminal
and no failure.

```text
attempt.json
65b3959cb347f171b67e9c33225583bd72aa116f6a32c74ba945969ab6002a19

launch.json
1d48fcf92bab77713dba79badb174c0a7a8393f4b7a4e31867cd2840fd79e80c

source_authority.json
aaa7407fe974fe628f433236d0050e3a91bb7eb05ac0ce17606395b45baf6ff0

screen.json
3eb237ac5f87c6fb405f70ed7afaf9f8f930a6ab633fc32b92b9a9d969b18003

source_best_pf_mean.pt
c1b4557e036796bd88f406cc4882821186a110001c1a88c47845bcad52c04a03

source_best_pf_r1.pt
48c5ad8f1cbd24f1ad2fa0d3dc4935045c187e8229e5864a868f42bd49b722a8

source_best_pf_r50.pt
2a0ccff4009444af3b72add02c6c11ca09bdd8a969abdb482714701ebfec718b

terminal.json
9919651c93b384c76e89fcc5248c9855f0b931604475b01b48611df104fb2625
```

The terminal, screen descriptors, sidecars, and independently recomputed
checkpoint body hashes agree exactly. The terminal links the attempt, launch,
source authority, screen, and all three checkpoint bodies. The current 55-file
producer execution closure independently reconstructs as:

```text
d1f287671957e4244594530e9dbb2b9bec9555cd6c7fe280741e5b0087c4bf51
```

Selected checkpoint student-state digests:

```text
PF-MEAN  80169f5e82d2d6e38f7fc5a6dfc65b7ed9739fd8e848862c93357d96370c5e2f
PF-R1    9ad23d0d857ee707e10a2b1b32444602a43daa629a4a73517d8f70ff57149a2c
PF-R50   7fc4d233650375f87cd402b84c69fe1396262b3ce6568badc66b44ffc8e06749
```

## 5. Completed next experiment

The separately pre-registered narrow score in
`WORKORDER_M2_POSTFUSION_CHECKPOINT_SCORE_V1_20260902.md`, recovered under the
V2 import-successor work order after a pre-data namespace failure, has now
completed:

```text
B30 / D-opt-k4
3 trained arms x {FIXED30, UNCAPPED}
x {6 external_post30_local, 7 within_post30 sessions}
= 78 rows
```

Every checkpoint was evaluated. PF-MEAN scored `0.1869045` external R2, or
`-0.1121529` versus the sealed POOLED k4 reference, with only `1/6` sessions
positive. The literal PF-R1 and PF-R50 rows were worse, so the score emitted
`NO_NOMINATION_STOP_POSTFUSION_ARCHITECTURE_AXIS`.

A later algebra audit proved PF-MEAN's scorer exactly matches its training
placement at M4/M10/M30. It also proved that PF-R1/PF-R50 were evaluated under
a materially different residual composition than training. Their current rows
therefore cannot close the learned-residual question; they require one fixed
whole-pool CPU rescore before the final architecture-axis decision.

The final interpretation is:

> Post-Fusion training was feasible, stable, and fast. The algebraically clean
> PF-MEAN route improved all seven within sessions under uncapped memory but
> failed to transfer to local held-out sessions. Causal pre-readout activity
> pooling remains the stronger external design. The residual-gate variants
> remain unresolved until their scorer is made identical to the training graph.
