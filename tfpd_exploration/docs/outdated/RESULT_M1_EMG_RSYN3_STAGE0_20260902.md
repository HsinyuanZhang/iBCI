# Result: M1 EMG-rSyn3 Stage 0 constructibility PASS

**Date:** 2026-09-02  
**Status:** CPU Stage 0 accepted; GPU capability not issued  
**Successor work order:**
`tfpd_exploration/docs/WORKORDER_M1_EMG_RSYN3_FCM_ONEFOLD_V1_20260902.md`  
**Parent FAIL (unchanged):**
`tfpd_exploration/docs/RESULT_M1_EMG_SYN3_STAGE0_SIGNED_REJECT_20260902.md`  
**Root:** `tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/stage0/`

## 1. Decision

```text
decision = PASS
decoder_r2_computed = false
gpu_capability_issued = false
parent_fail_rewritten = false
rectifier = relu_nonnegative_projection  # np.maximum(x, 0.0)
```

`terminal.json` sha256
`aa98889c41e4c1a839dcf5e51d36008a4cdedc2d442770f31ffde1a7b747788a`

`decision.json` sha256
`bbc6815a551406e94e330965fcd77cc1221a82be06a3c37428ed5066a1b53dad`

The sealed EMG-Syn3 FAIL at
`tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0/decision.json`
remains
`6cc34cdd434917907d8c90b739c3c02003a511c3ae919277baf6de041702bf39`.

A prior successor crash (`n_components=3` on two trial-mean PCA rows at M2) was
operator-retired, not deleted, to
`tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/retired_crash_trialmean_pca_m2/`.
M2 bin-level Syn3 remains constructible; only the historical trial-mean PCA
disclosure is typed unavailable at two trials.

## 2. What the rectifier did

Stored negative fraction was unchanged from the parent FAIL (~1.16e-4 to
1.63e-4). After `R(x)=max(x,0)`:

| session | clipped count | stored min | rectified min | rectified negative fraction |
|---|---:|---:|---:|---:|
| `ses-20120924` | 65 | -0.110 | 0 | 0 |
| `ses-20120926` | 73 | -0.108 | 0 | 0 |
| `ses-20120927` | 63 | -0.082 | 0 | 0 |
| `ses-20120928` | 52 | -0.133 | 0 | 0 |

This is a sparse projection onto the nonnegative cone. It is not a claim that
stored `preprocessed_emg` is a physiological rectified envelope.

## 3. Fold-0 M10 disclosure (development target `ses-20120924`)

| quantity | value |
|---|---|
| valid bins / seconds | 636 / 12.72 |
| design rank / condition | 4 / 7.31 |
| rSyn3 split-half `W` Pearson | 0.911 |
| rSyn3 split-half intercept Pearson | 0.954 |
| trial-mean PCA split-half `W` Pearson | 0.445 |
| M2 rejected for trial count | false |

All four folds × `{10,6,4,2}` produced finite carriers. M10 rSyn3 `W`
split-half is high (0.82–0.93). Matched trial-mean PCA `W` split-half is
weaker (0.25–0.45). That is disclosure, not a decoder R² gate, and it does
not by itself prove NNMF is better than PCA at deployment.

## 4. What this does not authorize

Stage 0 does not compute decoder R², does not open GPU, and does not relax
`S-Fix − Z-Fix >= +0.03`. Paper text still cannot call the stored series a
rectified envelope.
