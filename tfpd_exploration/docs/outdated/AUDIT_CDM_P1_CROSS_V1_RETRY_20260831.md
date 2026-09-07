# Audit: CDM x P1 Cross-Dataset V1 Failure and Retry

Date: 2026-08-31

Status: read-only root audit. The second replay is active. This document does
not authorize a stop, retry, code edit, target access, or scientific claim.

## 1. Executive finding

The first local `cdm_p1_cross_v1 --stage replay` did not terminalize. It failed
after substantial target replay work at a real implementation seam:

```text
replay.py:459
sealed_model_holder["sealed_model"] = swap.pop("sealed_model")
KeyError: 'sealed_model'
```

An external agent then changed the route-owned weight-swap/test bytes, removed
the first canonical result root, recreated the same root name, published a new
attempt, and launched a second replay at 2026-08-31 11:39 HKT.

The current replay is therefore not merely a PID restart. It is a revised-code
retry after target data had already been opened. The current attempt correctly
binds the revised bytes, but it does not bind an immutable predecessor failure
receipt because the first root was removed rather than terminalized with a
failure graph.

## 2. First replay evidence

First replay PID: `799940`

Device: physical GPU1, exact `CUDA_VISIBLE_DEVICES=1`.

The mutable execution log is:

`tfpd_exploration/cdm_p1_cross_v1_replay.log`

Stable post-exit SHA-256:

`9c963b64e0a4377c8705fd445a8e5aa01ea78c68e5ea2e93b4c6d206bfa7b4e4`

The log shows source parsing, CDM computation warnings, and the exact final
traceback above. No aggregate R2, gate, replay receipt, terminal, or failure
receipt was published. The old canonical attempt body is no longer present,
so the log is disclosure evidence rather than a complete immutable lifecycle.

The first attempt had bound at least these route-owned hashes before it was
removed:

```text
weights.py  622ec99f30afeb43b9d94974b9e3de94b505ae79ffe26b8a4caea3c849b2e5df
test.py     59fb48605b61ad7070ccea4ab078888ba3d7e850fc604b651b38fe501221e555
replay.py   fbbd577321a6ebef4300b3fc65b1945a9fcba241623cd6442811e3910fdc0b77
```

The exact first attempt body SHA was not preserved in the canonical root and
must not be invented.

## 3. Second replay evidence

Second replay PID: `804509`

Start: 2026-08-31 11:39:02 HKT.

Device: physical GPU1, exact `CUDA_VISIBLE_DEVICES=1`.

Current attempt body SHA-256:

`d8c5c32cf41e18a42d9cabdcdcbcb69f4524395b1d1fa7ca46573a95c971012b`

The attempt and sidecar are regular mode-`0444` files. It binds the unchanged
work order and Stage-P P1 promotion anchor, plus the revised owned files:

```text
replay.py   fbbd577321a6ebef4300b3fc65b1945a9fcba241623cd6442811e3910fdc0b77
weights.py  fe718cc9883738fe90745b933f82587c657fa7257191ef2a752e8d674243fca3
test.py     3ac9f277d03eb61cf54e927480e8162063b38195a25dbff8f5b958676d79ed71
```

The revised `swap_runtime_weights` result now explicitly includes the retained
`sealed_model` object required by the unchanged replay call site. This is an
engineering interface repair: it changes neither the sealed SWA bytes, model
weights, P1 hyperparameters, carrier law, target metric, nor gate thresholds.
Nevertheless it is execution-critical source drift and correctly required a
new attempt rather than continuation of the old process.

## 4. Scientific admissibility

The second replay may provide useful diagnostic numbers, but its terminal must
not be described as an unqualified clean first execution of V1. Any report must
disclose all of the following:

- a first replay opened target inputs and failed before receipt publication;
- the failure was an implementation-interface error, not a performance gate;
- no aggregate score or target-selected parameter was exposed by the first
  process;
- route-owned execution bytes changed before the second replay;
- the canonical V1 root name was reused after deleting the first attempt;
- no immutable predecessor failure graph links the second attempt to the first.

For a paper-grade canonical result, the safest later option is an additive
successor root that binds this audit and the second attempt/terminal as
historical evidence, then runs once with frozen bytes. That decision should be
made only after seeing whether the current diagnostic replay completes; do not
interrupt it merely to improve lineage aesthetics.

## 5. Isolation from PACD

The first replay materially affected PACD throughput but did not alter PACD
science or bytes. PACD epoch 008, which overlapped the high-CPU first replay,
records:

```text
wall_seconds             3608.5735
paired_steps_per_second  9.40122
combined_mean_loss       0.523603
```

Pre-overlap epochs 000--006 occupied `3300--3389 s` and
`10.0089--10.2802 steps/s`. Epoch 008 is therefore about 6.5% slower than the
previous slowest epoch. All scientific invariants remain exact: 33,925 steps,
zero P0 prediction/identity mismatch, zero RNG/prefix/finiteness violation,
finite Adam, all decoder gradients positive, and all ten zero encoder events
typed as valid all-units-dropped cases.

After epoch 008, root applied a scheduling-only CPU partition:

```text
PACD PID 783126:  logical CPUs 0-3,16-19   (four physical cores)
CDM  PID 804509: logical CPUs 4-15,20-31  (twelve physical cores)
```

The host is one NUMA node with 16 physical cores / 32 logical CPUs. The
partition changes no code, data, model, RNG, optimizer, GPU binding, or result
root. Epoch 009 is the first prospective measurement of whether the partition
restores PACD throughput while allowing the second replay to continue.

## 6. Current disposition

- Do not stop or edit either process.
- Keep PACD on physical GPU0 and CDM on physical GPU1.
- Preserve the CPU partition through the next natural epoch/replay event.
- Treat the current CDM result as diagnostic until retry lineage is explicitly
  handled.
- Never silently delete or recreate another canonical attempt root.

