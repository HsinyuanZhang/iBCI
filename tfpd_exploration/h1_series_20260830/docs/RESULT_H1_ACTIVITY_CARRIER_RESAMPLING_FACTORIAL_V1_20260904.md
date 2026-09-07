# Result: H1 Activity/Carrier Resampling Factorial V1

Date: 2026-09-04  
Status: `COMPLETE_ACTIVITY_DIVERSITY_DOMINANT`

## 1. Conclusion

The triggered 2x2 mechanism follow-up shows that the successful LP-R3
regularization effect is driven almost entirely by **activity-support
diversity**, not by repeatedly refitting the four-dimensional carrier.

Factorial mean effects across five source-grouped outer dates:

- activity resampling: **+0.021744 R2**, positive on **5/5** dates;
- carrier resampling: `+0.000970 R2`, positive on 4/5 but below the frozen
  `0.003` directional threshold;
- matched activity/carrier interaction: `-0.000342 R2`, without a common
  direction.

The registered classification is `ACTIVITY_DIVERSITY_DOMINANT`.

## 2. Factorial scores

`FF` and `RR` are immutable scores from the main experiment.  `RF` and `FR`
are the two new trained crossed cells.

| outer date | FF: fixed A/fixed C | RF: random A/fixed C | FR: fixed A/random C | RR: random A/random C | activity effect | carrier effect | interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19250108 | 0.425709 | 0.453796 | 0.427262 | 0.454789 | +0.027807 | +0.001274 | -0.000560 |
| 19250113 | 0.388135 | 0.399963 | 0.389184 | 0.400947 | +0.011795 | +0.001017 | -0.000066 |
| 19250115 | 0.581183 | 0.587963 | 0.583144 | 0.590018 | +0.006827 | +0.002008 | +0.000094 |
| 19250119 | 0.337304 | 0.361325 | 0.337524 | 0.356367 | +0.021432 | -0.002369 | -0.005178 |
| 19250120 | 0.384112 | 0.422971 | 0.385034 | 0.427894 | +0.040860 | +0.002923 | +0.004002 |
| **mean** | **0.423289** | **0.445203** | **0.424430** | **0.446003** | **+0.021744** | **+0.000970** | **-0.000342** |

Relative to the same frozen-C1 mean `0.405985`, random activity with a fixed
first-M3 carrier (`RF`) gains approximately **+0.039219 R2** and is positive
on all five dates.  It is only `0.000800 R2` below the full matched-resampling
LP-R3 cell (`RR`).

## 3. Carrier interpretation

This result does not say that the H-C carrier is unnecessary or that it may be
zeroed.  Every cell retained a four-dimensional carrier input at training and
deployment.  The result says something narrower and more useful:

- the carrier behaves like a compact conditioning variable whose canonical
  first-M3 estimate is already adequate for identity-branch training;
- the high-variance nuisance is which three neural activity realizations form
  the identity support;
- exposing the late-pool MLP to multiple activity triplets regularizes the
  activity-to-identity mapping;
- refitting the carrier on every source support triplet adds little average
  benefit and has no stable interaction with activity resampling.

Therefore the simplest future all-source candidate is not “more carrier
augmentation.”  It is **random-M3 activity training with a fixed canonical
M3 carrier**, followed by ordinary matched first-M3 deployment.  Because this
factorial was declared descriptive-only, it does not retroactively replace
LP-R3 as the main experiment's selected arm; it supplies a preregisterable
simplification for a new work order.

## 4. Integrity and scope

- only the two missing crossed cells were newly trained;
- each used the exact main 12-epoch, batch32, stride4, seed42, Adam `5e-5`
  branch-only contract;
- decoder/body state hashes were unchanged;
- checkpoints were sealed before opening each outer date;
- frozen-C1 prediction, target, endpoint, and first-M3 support digests matched
  the immutable main fold before factorial values were combined;
- target optimizer/backward/model-update counts were zero;
- formal held-out and EvalAI were not opened;
- only GPU0 was used; GPU1 was not queried or touched by this route.

## 5. Immutable authority

- result root: `tfpd_exploration/h1_series_20260830/results/h1_activity_carrier_resampling_factorial_v1/`
- attempt SHA256: `9d95d24e45528d36169a466a19d3761bfdf9f8abaf060d57dd9645c9f0139290`
- score SHA256: `e939b13a47006ffd7fadbd63273b3055dce0b2fefb6dfb0af050bcbd1c428df7`
- terminal SHA256: `90d23928ba567c1b3059c9a52045f6ffb89ace6f53d832689e7047ba54df65ce`
- design SHA256 at launch: `ac7fbf6dc520d93cd63e289c3954d3a7336ddaeef7ab0ec81305dff71e333284`
- work order SHA256 at launch: `d0eda6a13b9764b4486341566c2969da695c9e5479b817ef0b99efdeb14e28b2`

