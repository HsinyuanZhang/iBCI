# M2 AJPF V2 training result and score incident

Date: 2026-09-03 HKT  
Training status: `TERMINAL`  
Score status: `FAIL_CLOSED_BEFORE_INPUT_AUTHORITY_AND_NEW_ROWS`  
AJPF held-out result: **not yet evaluated**

## Training result

The V2 outer successor bound the exact V1 pre-step failure and passed the fresh CPU strict-load state regression.  It then completed the unchanged three-arm, twelve-epoch GPU0 training route.  All 12 epochs contained exactly 3,455 shared groups, the shared-mask/RNG and update gates passed, and all three epoch-12 checkpoints were sealed with strict reload/repeated-forward proofs.

The reviewed combined V2/V1 closure was 79 leaves:

```text
e156c087ca7462989a83a7018b18d72384cbc08ba0d799778989bdc56a4d6378
```

Epoch-12 source task loss:

| Arm | Loss |
|---|---:|
| `J-NATIVE` | `4.680893471231684e-05` |
| `J-R1` | `4.68043472210411e-05` |
| `J-MEAN` | `4.71212733827997e-05` |

These are source-training diagnostics, not held-out R² results.

Immutable training bodies:

| Body/artifact | SHA-256 |
|---|---|
| `training/terminal.json` | `d0e913e7a4f765f8c011556cb843753bf4734e290d80125af74c06cf051df4c3` |
| `training/manifest.json` | `7208320350757d9fe5d621272bc048e8ff052d50789bb856e492a72505d6ad68` |
| `training/epoch_12.json` | `dd80470bb8b72a55e980ecb9fd629c1a20fc86364a9c1650380b3629f86ca6d4` |
| `J-NATIVE_epoch12.pt` | `40c46a95b9f4ebd735d3fcd2011dcd2e328fa5bd994e6313ab3efbbeabb164f5` |
| `J-R1_epoch12.pt` | `fd7beac07c3df674639560d724d9be8a798b0af27f0e091c96667086e7b4c64d` |
| `J-MEAN_epoch12.pt` | `7e3c01e8f600cebda27ead7da0061495d8f34f6051390ea2802484e38e587008` |

## Score incident

The fresh CPU scorer successfully published its attempt, launch, and producer-authority receipts.  Before publishing `input_authority.json`, before computing or inspecting any new AJPF R², and before evaluating either promotion gate, it stopped at the exact historical native sentinel:

```text
ProductionError: AJPF exact 25d J-NATIVE FIXED30 historical sentinel drift
```

Score failure graph body SHAs:

| Body | SHA-256 |
|---|---|
| `score/attempt.json` | `68c16f4bc261a4b71dda53f18b076986d4cc3dbc74dc85b05614388f2f28d40e` |
| `score/launch.json` | `a550edd5e65355274d45f40c31f4d856baa7c518246483e14486909b29aec668` |
| `score/producer_authority.json` | `c7c0f4c57df2a1b4393e92f4c92b1289023146e6dbc3d861ca1011081193714a` |
| `score/failure.json` | `84938be8a9f06b98a263dc2419c1d0f2781f35f6a044b93cce71a7d9b7a54e5a` |
| outer `failure.json` | `8bacc519364d4f1aec6035cd9af6212f69b177a4e58355ac03b9701973b6d797` |

No score row, score receipt, score terminal, or outer terminal was published.  Therefore this failure is not a negative AJPF result.

## Bounded reproduction

After the formal process exited naturally, a read-only CPU reproduction used the same sealed V2 checkpoints, the same strict CPU module reconstruction, the same 13-session materializer, the same native POOLED replay, and the same held historical comparator graph.  Across all 13 exact record keys, it compared:

- prediction SHA;
- target SHA;
- governed-start SHA;
- window count;
- R².

Observed mismatches: **0/13 sessions and 0/65 field comparisons**.  The historical mapping and record-key law were therefore not reproducibly wrong, and no threshold or equality was relaxed.

## Permitted next action

Training is complete and must not be repeated.  A score-only successor may:

1. bind this exact successful training graph and both failed V2 score/outer graphs;
2. publish its own immutable attempt before Torch, target, or descriptor use;
3. repeat the five-field historical sentinel and publish per-session equality evidence;
4. proceed only if all 13 sessions pass exactly;
5. invoke the unchanged CPU AJPF scorer and unchanged two preregistered external gates in a fresh result root.

If any sentinel mismatch recurs, it must preserve the exact differing key/field/value evidence and stop.  It may not retrain, relax SHA equality, alter the comparator, or change the gates.

