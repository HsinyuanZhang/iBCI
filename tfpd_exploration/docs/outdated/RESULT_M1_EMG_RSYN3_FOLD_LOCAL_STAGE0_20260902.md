# Result: M1 EMG-rSyn3 fold-local Stage-0 audit PASS

**Date:** 2026-09-02  
**Status:** CPU fold-local Stage 0 accepted; GPU capability not issued by this
root  
**Work order:**
`tfpd_exploration/docs/WORKORDER_M1_EMG_RSYN3_FOLD_LOCAL_STAGE0_20260902.md`  
**Root:** `tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/stage0/`

## 1. Decision

```text
decision = PASS
decoder_r2_computed = false
gpu_capability_issued = false
parent_fail_rewritten = false
rsyn3_pass_rewritten = false
sealed_pass_comparison_all_equal = true
target_query_values_read = false
ls4_enabled_in_stage1_pilot = false
```

`terminal.json` sha256
`4ce75aea0a4fdd7c09bfd7fee95a8ae013dd3f70b9e0dc42de29408e683a297c`

`decision.json` sha256
`143800cbe78317f4df3b1e28143a0bf219b42044e332ad2bd2aa100466cc6652`

Sealed predecessors are unchanged:

| root | decision sha256 |
|---|---|
| EMG-Syn3 FAIL | `6cc34cdd434917907d8c90b739c3c02003a511c3ae919277baf6de041702bf39` |
| EMG-rSyn3 V1 PASS | `bbc6815a551406e94e330965fcd77cc1221a82be06a3c37428ed5066a1b53dad` |

## 2. Fold-local isolation

Each fold independently:

- source EMG: full session (fold 0 sources `[0,409)`, `[0,376)`, `[0,412)`);
- source neural: `[0,10)`;
- target EMG and neural: `[0,10)` only;
- query trials remain structurally available and unread.

Fold-0 dictionary immutability used **636 target-support bins**, not an
eight-bin source prefix. The frozen dictionary digest was unchanged.

## 3. Comparison with sealed V1 PASS

Carrier, dictionary, coverage, and split-half digests match on all 16
fold×budget rows. V1’s target query EMG read therefore did not change the
constructibility table; it was a scope/disclosure defect, not a silent
change of the M10–M2 carriers.

The PCA diagnostic is labeled **rectified trial-mean PCA3**. It is not
called historical signed PCA.

## 4. LS4

Unequal-length LS4 currently cyclic-repeats or truncates
(`take = arange(dest_n) % source_n`). It is disabled for the first
Z-Fix / S-Fix / S-Acyc GPU pilot and must be repaired before Stage-2
`S-LS4`.

## 5. What this does not authorize

This root does not compute decoder R² and does not mint a GPU capability
by itself. ReLU remains `float64 np.maximum(x_raw, 0.0)` before RMS and
NNMF. The static gate remains `S-Fix − Z-Fix >= +0.03`.
