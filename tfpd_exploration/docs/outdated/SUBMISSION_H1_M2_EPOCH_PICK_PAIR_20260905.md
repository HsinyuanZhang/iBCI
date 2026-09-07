# H1/M2 visible-held-out epoch-pick EvalAI pair — 2026-09-05

## Status

Two private submissions were frozen, packaged, container-checked, uploaded, and
registered on EvalAI phase 4599.  Epoch selection used only locally visible
held-out development recordings.  Neither hidden EvalAI result was opened
during candidate selection.

| Dataset | Submission | Frozen candidate | Visible selection score | Remote status |
|---|---:|---|---:|---|
| M2 | 581919 | MOVE-T4 + profile-free adapter, seed 44, epoch 8 (one-based) | 0.3604492265, equal-session mean over six visible external sessions | **finished: official HO 0.3494526364** |
| H1 | 581920 | C2 M3-aware prefix-cycle decoder, epoch 15 (zero-based) | 0.4056059026, visible held-out grouped mean | **finished: official HO 0.3759890904** |

The two candidates are not FiLM-content claims.  M2 uses a zero/profile-free
adapter with the movement-window T4 carrier.  H1 deploys the C2 native
early-pooled identity through a bitwise-neutral zero-FiLM runtime shim.

## M2 frozen package

- Selection grid: seeds 42/43/44 × epochs 1–12, fixed before scoring.
- Tie break: higher visible equal-session mean, then higher worst session,
  earlier epoch, then lower seed.
- Selected visible worst-session R2: `0.2340014608`.
- Selection receipt SHA-256:
  `9759e0853655d359146c30f1c6550005f3569b6a391cca5eb371c04749541ee2`.
- Selected adapter-head SHA-256:
  `13551d3fc33d1cc296670c577c11519c04abdb42fa081f7d061191bb5610e6c5`.
- Payload SHA-256:
  `f2f8cd4c046a5880e9716d61981cee4aa0652d33e411212fa7be217cef05b051`.
- Docker image ID:
  `sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f`.
- Container payload hash: verified exactly.
- EvalAI metadata: not zero-shot, not test-time adaptive, not externally
  pretrained.

## H1 frozen package

- Selection authority: visible held-out H1 calibration/development surface.
- Selection authority SHA-256:
  `73e2666b1121685ab0d953a72073c4c8b0bc48fbac9d0ca63b0461985a1bdfdc`.
- Checkpoint SHA-256:
  `ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215`.
- Payload SHA-256:
  `91ef13cc94b9ab865c8f926dcbb6d33e9628bf9cbc9b757665d10e33cfda144a`.
- Docker image ID:
  `sha256:c7a88aaec3b75754e183a25e0a5363a514e313f84be84ae10c7cb39d094baeea`.
- Native identity equality through the zero-FiLM shim: bitwise true.
- Container smoke: `PASS_H1_EP_FILM_CONTAINER_SMOKE`; prediction finite;
  model and shim states remained immutable.
- EvalAI metadata: not zero-shot, not test-time adaptive, not externally
  pretrained.

## Upload and resource receipt

The phase preflight reported `active=0`, `max_concurrent=3`,
`max_per_day=6`, and `today=3` before registration.  Both submissions were
therefore within quota.  M2 registered at `2026-09-04T18:30:25.068987Z`; H1
registered at `2026-09-04T18:31:03.651492Z`.  Existing remote base layers were
reused, so the effective new uploads were the 18.6 MB M2 payload layer and the
128.1 MB H1 payload layer.  No GPU was used for packaging or upload, and the
unrelated GPU1 job was not touched.

## Official results

H1 submission 581920 reached a stable `finished` state after the expected
transient worker/host-scorer handoff:

| H1 metric | Official value |
|---|---:|
| Held Out R2 Mean | **0.37598909036302** |
| Held Out R2 Std. | 0.12575989592244785 |
| Held In R2 Mean | **0.5111673345036892** |
| Held In R2 Std. | 0.030089216314430362 |
| Normalized Latency | 0.035071517532194986 |

The remote H1 result body SHA-256 is
`6a5842908d79321ab62075e8dba01efcb3c045420afba019711d68abf866326a`.

M2 submission 581919 also reached a stable `finished` state:

| M2 metric | Official value |
|---|---:|
| Held Out R2 Mean | **0.3494526364023173** |
| Held Out R2 Std. | 0.09990003111191914 |
| Held In R2 Mean | **0.5972297932379577** |
| Held In R2 Std. | 0.02084621382897675 |
| Normalized Latency | 0.04403732470875482 |

The remote M2 result body SHA-256 is
`3833b3f3b4d01f333a0f05fac20956b7338260d452e28fb431f8dc60be9df2e7`.
Against the immediately preceding MOVE-T4/no-FiLM submission 581899, this is
`+0.0219346990` held-out R2 and `+0.0234078157` held-in R2.  The hidden result
therefore supports transfer of the visible-held-out epoch choice; it does not
support a profile-content FiLM claim because the selected adapter receives the
fixed zero profile.
