# DANDI 000688 sparse-event T4 / FiLM Stage-1 V1 incident and additive V2 authority

Status: `FROZEN_ADDITIVE_SUCCESSOR_AUTHORITY__SOURCE_ONLY`

Date: 2026-09-04 (Asia/Hong_Kong)

This document records a pre-GPU implementation defect in the first Stage-1
launch and authorizes a science-neutral additive successor. It does not alter
the frozen experiment design or work order, does not authorize test/external
data, and does not authorize deletion, retry, or overwrite of any V1 result
root.

## 1. Frozen parent authorities

- Design SHA-256:
  `56982085d4cc7c4e06d29d79a3701e8e6f6d93d08955ceff4c09737bef956705`
- Work-order SHA-256:
  `18cc4dde507a41b3853cb6b5a6bd6f9f5c3d47b0837e26949433803ce427c6d2`
- Stage-0 mask-authority SHA-256:
  `7a5a4cf505a31e12b423059e8a2df0de7ff534d6301ed3409a8064584088bb54`

All scientific laws, arms, seeds, source split, Q50 surface, twelve-epoch
budget, learning rates, epoch-8-through-11 averaging, and opening predicates
remain unchanged.

## 2. Immutable V1 failure witnesses

The following two V1 roots must remain present, immutable, and consist of
exactly `attempt.json`, `attempt.json.sha256`, `failure.json`, and
`failure.json.sha256`, all regular files with mode `0444` and link count one.

### Seed 42

Root:
`sua_exploration/results/dandi688_sparse_event_t4_v1/stage1/sua/seed42`

- attempt body SHA-256:
  `13f057b55b58b4338441bb9409f5ab19d75c9b892390939a5b251157e228b481`
- failure body SHA-256:
  `cc1d71826cf3db5184578b80872a8636f704219c579a384b9c51931103183353`
- failure type: `KeyboardInterrupt`
- published prefix: `attempt.json`, `attempt.json.sha256`

### Seed 43

Root:
`sua_exploration/results/dandi688_sparse_event_t4_v1/stage1/sua/seed43`

- attempt body SHA-256:
  `32bbe68030a7e19101d9610e5675c06696e2e4115912f1144f409978069faa43`
- failure body SHA-256:
  `87712acc8a4ded570dee9c0c5d5f3a1313568e244aea8cb2c12eeb2e37d67218`
- failure type: `KeyboardInterrupt`
- published prefix: `attempt.json`, `attempt.json.sha256`

Both attempts were interrupted during CPU source materialization after root
reservation and before CUDA initialization or any optimizer step. No training,
validation prediction, decoder-performance result, test file, or external file
was observed. GPU 0 and GPU 1 were idle after interruption.

## 3. Defect and correction

The V1 executor constructed the four FiLM heads by calling the random head
factory four times. Although every final projection was zero initialized and
the pre-first-step native sentinel would therefore pass, the hidden-layer
initial states were different across `EMPTY`, `PHASE-R`, `SE-T4`, and
`ROW-SHUFFLE`. This violated the work order's identical-head-initialization law
and weakened paired-arm attribution.

The only authorized correction is:

1. set the frozen seed before head construction;
2. invoke the FiLM head factory exactly once to construct one template;
3. create the four arm heads by `copy.deepcopy(template)`;
4. verify and receipt that all four initial complete state dictionaries have
   the same canonical digest before the first optimizer step;
5. retain separate optimizers and all other frozen laws unchanged.

The same one-template/deep-copy rule applies to the Stage-2 FiLM matrix before
that conditional branch may run. No hyperparameter, profile, arm, data,
training, scoring, or selection change is authorized.

## 4. Additive successor roots

Fresh Stage-1 execution is authorized only under:

```text
sua_exploration/results/dandi688_sparse_event_t4_v1/stage1_v2/sua/seed42/
sua_exploration/results/dandi688_sparse_event_t4_v1/stage1_v2/sua/seed43/
sua_exploration/results/dandi688_sparse_event_t4_v1/stage1_v2/sua/seed44/
```

Before reserving any successor root, the executor must held-file validate this
document and both exact V1 failure graphs. Each V2 attempt must bind their body
SHA-256 values. The three V2 roots remain one-time, immutable lifecycle roots;
failure requires another explicitly frozen additive successor.

Stage-2 and Stage-3 namespaces remain those in the frozen work order because
no Stage-2 or Stage-3 root was created by this incident.

