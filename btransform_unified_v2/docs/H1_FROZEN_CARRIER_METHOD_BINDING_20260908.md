# H1 frozen carrier method binding — 2026-09-08

## Decision

The official H1 RIFT submission **582073** is now bound to the actual upstream H-C construction family.  Its four-dimensional carrier was produced in the all-source M3 pipeline with **source-centering/scaling, PCA `q=12`, intercept-unpenalized ridge `lambda=10.0`, a source output-SVD basis `U` reduced to four coordinates, analytic empirical-Bayes shrinkage, and source-RMS normalization**.  It was then copied unchanged through the EP-FiLM, C2, B2, and RIFT packaging chain.

This corrects the prior provenance boundary.  The manuscript's `q=16`, `lambda_H=100` formula does **not** describe submission 582073.  It must be changed to the bound `q=12`, `lambda=10.0` construction or explicitly identified as a different carrier family.

## Byte chain: source M3 payload → C2 → B2 → RIFT

The upstream source candidate is the immutable EP-FiLM payload `SPINT-main/local_data/h1_epfilm_evalai_v1/decoder.pt` (SHA `df71cb9329a87b5073242044b2933391ced7bf866d1f481e994f71dbd97ba2d7`). Its builder creates each row with exactly the first three public trial numbers, calls `fit_deployment_carrier(record, plan, values)`, divides its four-column result by the sealed source normalizer, and serializes it as `sessions[tag]["carrier"]`.

A read-only comparison of all 27 `float32` contiguous carrier arrays found that the source candidate's `carrier`, C2 M3 `sessions[tag]["carrier"]`, and B2 `bank_by_dataset_tag[tag]["T"]` are byte-identical for every official tag. The canonical ordered comparison-record digest is `a07eead0cae895eed606d9efcfc0048e43fdaa2576619f090198f5d2345ba407`.

For the required concrete anchor, `S0_set_1` in the source candidate has shape `[176,4]`, calibration trials `[3.0,4.0,5.0]`, and raw contiguous float32 SHA-256 `85d4801852554164421218d030404d07d8bcfea7795296e1ae28ffc82d17e2a2`. That is exactly the SHA recorded for C2 and B2, and therefore exactly the direct `T` in frozen RIFT submission 582073.

C2’s `build_payload.py` changes only decoder-state / zero-FiLM data. It inherits the source 27-session M3 rows byte-for-byte. B2’s payload packer turns the C2 row `carrier` into its `T`; RIFT’s packer copies the B2 banks without calling an H-C estimator.

## Sealed H-C construction authority

The actual source authority is in the historical all-source M3 producer checkout:

`/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1/source_authority/`.

Its sealed `plan.json` (SHA `a92b57350f2dcb04027bb6d848e5582d84e1bef4bf507d6844963ccad3c87bd5`) records:

- selected `q=12` and ridge `lambda=10.0`, selected from the sealed source-only grid receipt `selection.json` (SHA `3a9f59f75da95aab056850f70e4b5c47afc093125e02e789f0c2f7db59ce6067`);
- source mean, scale, PCA rows, output basis `U`, and prior mean hashes; `tau2=1.0378493774682498e-10`; transform SHA `1c566312152d0203b282fd62a415694d9ddf5845a0208c291e4240e2b9b3ccd7`;
- all 13 source recording identities and input hashes.

Its sealed `normalizer.json` (SHA `9b35870244a38d0568c3e5f3fd5ab2f10bc2ede815dafe308a130a04e5db6732`) fixes

`C_norm = C_raw / max(s_src, 1e-12)`, where `s_src = sqrt(mean(C_src_raw**2)) = 6.8260113140959355e-06`.

The source package applies this exact normalizer after `fit_deployment_carrier` and casts the result to contiguous `float32` before serialization.

The named construction operator is `SPINT-main/src/data/h1_m4_eb_pilot.py::fit_deployment_carrier` (current audited source SHA `c73c80fcec05d323052d9a4154a4cf7c46e90989a917c89116ce085e76b50a8a`). It accepts exactly three or four unique support trials; this deployment row uses exactly three, with no padding. It projects rate blocks as `((rates-mean)/scale) @ pcs[:q].T`, fits a ridge decoder with an unpenalized intercept, maps coefficients through `pcs[:q].T / scale` and `U`, then applies the analytic EB weight `tau2/(tau2+projected_variance)` about `mu`. This establishes the PCA, ridge, output-SVD, EB-shrinkage, and normalizer stages that C2 itself only consumes.

## C2 and runtime consumption

The frozen C2 materializer consumes its already-computed M3 carrier `[176,4]`, combines it with the three-trial activity identity, and emits `E0`; it does not re-fit H-C. The RIFT decoder uses `E0 [176,700]` as static identity and the same `T [176,4]` as a direct carrier. Direct-carrier ablations holding E0 fixed therefore do not remove H-C information already fused into `E0`.

## Functional replay proof and calibrated construction cost

The original sealed authority retains the parameter receipts but not the original plan-NPZ bytes. A source-only replay therefore rebuilt the numeric plan from the sealed `q=12` / `lambda=10.0` selection and the 13 public source calibrations. Its FP64 `mean` and `scale` hashes agree with the sealed receipt, while the replayed `pcs`, `U`, and `mu` FP64 hashes differ. This is recorded as a provenance difference; it is not treated as a parameter substitution or numerical correction.

The functional test is stronger for the deployed boundary: with that replayed plan, all 27 public-calibration rows use their original first-three trial supports and reproduce the normalized contiguous `float32` carrier bytes of both the source payload and frozen RIFT banks. The verify-27 receipt is `results/diagnostics_v1/h1_source_plan_replay_verify27_v1/report.json` (SHA `5c64b943c2d6758493d88ecaee714edfd1294b9fd28a2eedeec4006fd6461aca`). Thus the replay is functionally equivalent at the deployed `T` boundary, but it does **not** claim recovery of the original plan-NPZ container or identical FP64 PCA/SVD arrays.

The resulting H1 calibration-cost receipt uses this verified replay only after that all-27 gate. It measures 27 sessions × 3 warm rounds with public support records and interpolated M3 activity preloaded: replayed carrier solve → frozen C2 `E0` materialization → static bank construction. All 81 rounds reproduce both frozen `T` and `E0` byte-for-byte. The total warm wall time is `1.001563` ms minimum, `1.085079` ms median, `1.155038` ms P95, and `1.325125` ms maximum. File and report hashing, source-plan construction, raw-record loading, and activity interpolation are outside this timing scope. The cost receipt is `results/diagnostics_v1/h1_frozen_calibration_cost_v2/report.json` (SHA `54d737eac6342f10d0f6517bfe94f48ff6303e35e7a160012727b98b83d53186`); its validation receipt SHA is `20860381e55a781047e8c5e7b6942d6f78764094cfe5d2360e57999a4bc5a03d`.

## Scope and limitation

This is a read-only provenance audit plus read-only public-calibration replay and cost measurement. It did not score a model, use a GPU, open EvalAI, modify frozen payloads, train weights, or access hidden test data. The historical source checkout has advanced past the producer's recorded `LEGACY_HEAD` and does not retain that Git object locally; the binding relies on immutable authority receipts, the source package's recorded construction route, and deployed-array equality. The missing Git object still limits commit-level and byte-identical plan-NPZ reconstruction; the replay evidence establishes functional equivalence only at the 27 deployed normalized `float32` carrier arrays.
