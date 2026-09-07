# Result: M1 EMG-Syn3 Stage 0 signed-EMG reject

**Date:** 2026-09-02  
**Status:** sealed FAIL; do not modify, overwrite, or re-run into this root  
**Parent work order:**
`tfpd_exploration/docs/WORKORDER_M1_EMG_SYN3_FCM_ONEFOLD_V1_20260902.md`  
**Parent work-order SHA-256:**
`7b4a002b39e4d86808931e2682b8cde2a9b55dc02a557d490da9e944da2c4378`  
**Root:** `tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0/`

This note interprets a completed CPU Stage-0 terminal. It does not reopen
EMG-Syn3, does not authorize GPU, and does not write into the sealed root.

## 1. Sealed receipts

| leaf | sha256 |
|---|---|
| `decision.json` | `6cc34cdd434917907d8c90b739c3c02003a511c3ae919277baf6de041702bf39` |
| `signal_view.json` | `27e681f86aca6bf7a2d7163cb422ba2c6b1f04c6bb8c5414687335c3453858cf` |
| `terminal.json` | `f8436c9e2e1f1e62a3a3fb4f78415cba7bef395c3821437ddb2ee67b99a3f7ba` |
| `attempt.json` | `09959c75df6b26eaf00f4225df53a0617492bfb70a9c70ea881dc10875e13e17` |
| `launch.json` | `c22ba4286388a50b79caa4777a2cb44e7da04c1848f05d64ef75de19e1a901b4` |
| `reliability_table.json` | `1633902c6cbba5e7770dbed172df754a25078bb76efe1f23474edc87f1a47655` |
| `fold_receipts.json` | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| `checkpoint_authority.json` | `a2f6b78b8decfdcec83959e41946cf2f792c0bba84ed32e554f200774b8a2187` |

`decision.json` records:

```text
decision = FAIL
failures = ["stored preprocessed_emg is signed; NNMF rejected without a frozen rectifier"]
decoder_r2_computed = false
```

`terminal.json` records `COMPLETE_STAGE0` with `decision=FAIL` and
`gpu_capability_issued=false`. That pairing is correct: the audit finished, and
the scientific gate failed. A later successor must not convert this FAIL into a
PASS by rewriting these bytes.

## 2. What failed, and only that

The stored `preprocessed_emg` time series is not in the nonnegative orthant
required by NNMF. Direct factorization was rejected, as the parent work order
required when no rectifier was frozen.

Per-session stored-view negative fraction:

| session | negative fraction | minimum |
|---|---:|---:|
| `ses-20120924` | `1.463e-4` | `-0.1102` |
| `ses-20120926` | `1.631e-4` | `-0.1076` |
| `ses-20120927` | `1.588e-4` | `-0.0819` |
| `ses-20120928` | `1.163e-4` | `-0.1333` |

About 0.012%–0.016% of values are negative, so about 99.984%–99.988% are already
nonnegative. That is not the signature of untreated bipolar raw EMG. It is the
signature of an approximately nonnegative stored view with sparse negative
overshoot from interpolation, filtering, or other numerical processing.

This FAIL does **not** prove any of the following:

- a three-dimensional EMG synergy is unconstructible;
- bin-level neural–EMG encoding is invalid;
- an AFC4-like carrier has no information;
- independent carrier injection is invalid;
- carrier-by-CDM-A interaction is invalid.

Those estimands were not computed. `reliability_table.json` and
`fold_receipts.json` are empty because NNMF was not entered.

## 3. What must not be claimed

Until a complete preprocessing provenance is bound, paper language may not call
the stored tensor a “rectified envelope.” The honest stored-view statement is:

> FALCON M1 `preprocessed_emg` is an approximately nonnegative 20-ms series
> with a sparse negative overshoot of order 1e-4; it cannot be passed to NNMF
> without a source-frozen nonnegative projection.

## 4. Successor, not a patch of this FAIL

EMG-Syn3 remains the scientific hypothesis. The parent root stays FAIL. The
correct next experiment is a named successor that freezes one rectifier before
RMS scaling and NNMF:

- design: `tfpd_exploration/docs/DESIGN_M1_EMG_RSYN3_SUCCESSOR_20260902.md`
- work order: `tfpd_exploration/docs/WORKORDER_M1_EMG_RSYN3_FCM_ONEFOLD_V1_20260902.md`
- name: **EMG-rSyn3** (Rectified EMG Synergy-3)
- operator: `x_rect = np.maximum(x_raw, 0.0)`

Do not sweep `abs`, offset, envelope filters, or smoothers after reading this
FAIL. Do not delete or chmod these receipts to make Stage 0 look green.
