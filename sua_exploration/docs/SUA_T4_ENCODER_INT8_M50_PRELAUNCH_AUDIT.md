# SUA T4@50 encoder-only INT8 prelaunch audit

**Status:** authorized after root and independent Terra review; GPU launch requires the tests and
hash-bound selection below to remain unchanged.

## Selected experiment

The retained FP32 architecture is ordinary coupled `B3S + T4`, not FiLM, decoupled K/V,
factorized-logit residual, low-label shrinkage, or B3TStream. The quantization contract is:

- activity identity support: chronological first 30 trials;
- T4 labelled/rate support: chronological first 50 rewarded trials;
- common validation query: trials `[50:]` only;
- exact seeds: 42, 43, 44;
- strict split: 27 train / 6 validation / 6 unresolved formal-test names;
- four identity-encoder Linear layers: W8A8, INT32 accumulator, integer requantization;
- pooled 64-D activity and normalized 4-D T4 share the post0-input activation scale and are
  concatenated in the integer domain at the real `68 -> 64` layer;
- decoder remains FP32;
- PTQ scale candidates are selected from the 27 training sessions by identity RMSE only;
- validation does not select PTQ scales, a QAT epoch, or a candidate architecture;
- PTQ passes only with mean `delta R2 >= -0.01`, maximum edge saturation `<=0.5%`, zero INT32
  overflow, and exact STE-versus-independent-integer `E` equality;
- a failed PTQ launches exactly eight epochs of encoder-only QAT; any other epoch budget is
  rejected by the entrypoint.

The final architecture receipt is
`manifests/sua_t4_final_architecture_selection_v1.json`, SHA-256
`d4a35d99311a21dbe2b530f2b2c420516c79f7a1bc65c6ae0ee0f51eec149b41`. It binds the strict
three-seed baseline aggregate, all terminal candidate evidence, manifest/teacher hashes, and the
three selected `T4@50` checkpoint, metadata, and validation-result hashes.

The receipt deliberately separates two evidence layers: the strict three-seed M30 B0/T4/TS4
matrix establishes T4 mechanism eligibility, while the three M50 artifacts select and bind the
exact checkpoints to quantize. It does not pretend that the M30 matrix is an M50 comparison. The
authorized endpoint is the paired degradation of each selected M50 checkpoint after encoder
quantization; no M50-vs-B0/TS4 efficacy claim is added by this experiment.

## Corrected blockers

The old dormant INT8 chain was not launched because it targeted `T4@30` and evaluated from trial
30. The corrected chain now requires the final-selection receipt, maps all three seeds to the M50
checkpoint namespace, requires `pool_size=50`, holds activity calibration at 30, and passes
`evaluation_start_trial=50` into the fixed evaluator. PTQ receipts now distinguish between:

- a scale-selection objective that does not use behavioral labels; and
- train-only T4 inputs that do contain target-label/rate estimates from 50 trials.

The aggregate no longer trusts a self-reported `ptq_pass` or `qat_pass`. It recomputes the R2
delta, checks every session delta, the full gate dictionary, saturation, overflow, exact integer
parity, scope, 30/50/50 protocol, selected checkpoint hash, package hash, four W8A8/INT32/integer-
requant layers, real 4-D integer side concatenation, and FP32 decoder scope. It also opens the
hash-bound NPZ with pickle disabled and checks the exact 20-array key set, dtype, and shape for
all four weight/scale/bias/requant groups. Reports and packages are immutable by default; retries
require a new result directory.

## Verification

Before data/GPU launch:

- shell syntax and Python compilation passed;
- `software-to-hardware/test_t4_qat_backward.py`,
  `software-to-hardware/test_b3_qat_backward.py`,
  `test_aggregate_t4_encoder_int8.py`, and
  `test_t4_encoder_int8_protocol.py`: **16 passed**;
- negative tests reject false pass flags with parity error, overflow, excessive saturation,
  excessive R2 loss, a failed gate, T4@30 budget drift, scope drift, a non-W8A8 manifest, or an
  NPZ array-shape mismatch;
- an actual selected seed-42 checkpoint CPU smoke produced `post0=(64,68)`, layer order
  `pre_pool/post0/post1/post2`, and an INT8-engine output `E` of shape `[N,50]`;
- final-selection validation opens no NWB and resolves no formal-test file.

The result namespace is `results/sua_t4_encoder_int8_m50_v1/`. Only the claim
**T4@50 identity encoder INT8 + FP32 decoder** is authorized; this is not full-model INT8.
