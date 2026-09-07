# Submission 581713 — M2 APFG Static Probe (exploratory)

Date: 2026-09-03
Status: pushed and registered; official score pending
Decision authority: user overrode the hold recommendation on 2026-09-03
("即使是 NULL 也提交一下吧,毕竟 EvalAI 的数据集不一样") — spending 1 of the 2
rationed submissions to buy an official confirmation point on data the local
replay cannot fully proxy.

## What was submitted

| Field | Value |
|---|---|
| Submission ID | **581713** (challenge 2319, phase few-shot-test-2319 / 4599, team HKU-ECE, private) |
| Arm | `apfg_static_act30_dopt4` (display: M2-APFG-S) |
| Image | `spint-t4-m2:apfg-static-s42-b3d19967`, id `sha256:290757aabe648c7f5f8e3efdae1b941428988b5263a71f8abda379079d2faf3a` |
| Payload | `b3d19967b258fe94584ad4f84a7d394f8c2c83f85bdc3e198b59fc8178b05af5` (13 cached APFG-gated identities + frozen coupled decoder) |
| ECR uuid tag | `15bb3a81-9188-42dd-a5c0-e9a42fb62e68` |
| Attributes | IsHeldOutZeroShot=false, IsTestTimeAdaptive=false, IsPretrained=false |
| Cell receipts | `results/evalai_m2_apfg_static_v1/{attempt,terminal}.json` + `artifacts/evalai_push_state_apfg_static_v1.json` |

Deployment: offline calibration byte-identical to the officially scored
`act30_dopt4` arm; one anchored post-fusion scalar gate applied at build time
(`alpha = -0.20759029686450958`, bound to the immutable V2 same-surface graph);
runtime serves cached identities only (`reset`+`predict`, `on_done` no-op, zero
online updates).

## Pre-registered expectation (from `local_validation_receipt.json`)

- Native arm of the image reproduces the sealed `act30_dopt4` rows bitwise
  (external delta 0.0), so the honest local anchor for the official held-out
  score is that arm's HO 0.2897.
- On the local static 30-trial pool the learned gate is null-to-slightly-
  negative (external mean -0.0007693, 3/6 positive; within +0.0004449, 4/7),
  so the point prediction is **HO approximately 0.289, a descriptive tie with
  `act30_dopt4`**.
- The user's override rationale is recorded: the official evaluation data is
  not the local file set, so the local null is a prediction, not an outcome.

## How to read the official result

- **Tie (|delta| < ~0.003 vs 0.2897):** the local static-pool finding is
  confirmed on official data — the transferred gate adds nothing at pool size
  30. Closes the deployment question cleanly.
- **Materially higher (>= ~+0.005):** the local data under-represented the
  official evaluation distribution; a real gate effect on official data
  exists and warrants a successor note (still not a promotion without the
  pre-registered gate on a fresh study).
- **Materially lower:** pool-composition sensitivity of the transferred
  scalar hurts on official data; report as a negative deployment probe.

In no case does this submission promote APFG: the pre-registered +0.010/4-of-6
promotion gate belongs to the local V2 study and was not passed there.

## Quota state after this push

6/6 today (exhausted), 6/50 this month, 19/100 total at push time. One
submission of the user's rationed two remains.
