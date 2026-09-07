# Work Order: H1-M3RC All-Source Package V1

Date: 2026-09-03  
Authority: `DESIGN_H1_M3_READOUT_CALIBRATION_V1_20260903.md` and
`RESULT_H1_M3_READOUT_CALIBRATION_V1_20260903.md`

## Authorized action

Build and locally validate one H1 EvalAI package on physical GPU0.  GPU1 must
not be queried or touched.  Building and validating the package does not by
itself authorize an EvalAI push.

## Frozen method

- neural anchor: all-source C1 epoch-49 checkpoint SHA
  `0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06`;
- model state SHA
  `bdaf7dbcbae75ea307f20356aaf80066586f7d9afa273712a5e34708b903eb85`;
- original carrier authority SHA
  `8ea4bb1174c00ab713843cd7561562d43f81509eaaea6ea12ee80cd4eba95de7`;
- exact carrier transform SHA
  `1c566312152d0203b282fd62a415694d9ddf5845a0208c291e4240e2b9b3ccd7`;
- support: chronological first three trials, with exactly three legal trials
  required for every held-out-calibration session;
- readout: `MAT7`, ridge selected once on all 13 held-in source sessions
  before held-out-calibration files are opened;
- runtime: fixed session payload, no fitting, no backward, no optimizer, no
  query-label access, no continual update.

The builder must use the sealed producer checkout at Git commit
`5dd9bb4a7377a5431b7dbac4f1378e529130eb1a`; substituting an evolved PCA/SVD
implementation is forbidden.  The reconstructed `mean`, `scale`, `pcs`, `U`,
`mu`, `tau2`, source normalizer and transform must all match the historical
authority exactly before packaging continues.

## Required package and checks

The payload contains all 27 FALCON session keys: 13 held-in plus 14 held-out.
Each row contains one M3 activity identity, one M3 four-dimensional carrier,
and one M3-fitted readout map.  The identity, carrier and readout support lists
must be identical and length three.

Before any push:

1. strict model reload and state immutability;
2. source-only selection receipt published before held-out calibration access;
3. 14/14 held-out rows prove exactly three legal trials;
4. batch sizes 1 and 8 exercise `reset/observe/predict`;
5. host and Docker CPU/GPU smokes are finite;
6. local held-in-minival replay reports both the uncorrected anchor and M3RC;
7. official decoder output shape is `[batch, 7]` at each 20 ms call;
8. hidden evaluation data opened: zero; EvalAI submissions: zero.

## Fresh roots

- receipt root:
  `tfpd_exploration/h1_series_20260830/results/h1_m3_readout_calibration_evalai_package_v1/`
- package:
  `SPINT-main/local_data/h1_m3rc_evalai_v1/decoder.pt`

Both roots are single-attempt.  A failure remains immutable and requires an
additive successor; it is never overwritten.
