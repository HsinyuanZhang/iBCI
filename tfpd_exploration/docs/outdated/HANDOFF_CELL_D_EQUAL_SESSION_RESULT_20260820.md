# Handoff: Cell-D Equal-Session Result

Date: 2026-08-20

## Decision first

Stop the equal-session sampling route. Do not replicate it at seeds 43/44 and do not spend another
GPU round tuning session-sampling weights. The exact one-factor successor did not improve the
governing matched score. Keep the sealed Cell-D seed-42 system as the baseline and continue the
already-running TF-SR structural route.

This is a valid negative result, not a failed run. Training, SWA construction, baseline replay, and
matched scoring all completed under their frozen contracts.

## What changed

Only source-session exposure changed:

- sealed Cell D: source windows are sampled in proportion to each session's number of windows;
- successor: each of the 27 source sessions contributes equally within every epoch.

The model graph, seed 42 initial state, B3S+T4 input, whole-unit dynamic dropout, optimizer,
48-epoch budget, checkpoint window, SWA construction, target-free evaluation, and score metric were
held fixed.

## Authoritative evidence

Training:

- result root: `tfpd_exploration/results/cell_d_equal_session_seed42_v1`;
- terminal status: `CELL_TRAINING_TERMINAL__UNSCORED`;
- terminal SHA-256: `cb0bc529e2db368d968ca758f101d1a31e8c83b8de5f93a76f97f01fa67d80ca`;
- SWA artifact SHA-256: `512af5a75c712e675b666c676619297c64b4b21f476d63c694d0a83b70a332ff`;
- SWA state SHA-256: `e7f8959d8e51dafd6a8a729f6947cdd5e7d65d775d38f3df0667c082fd60672d`;
- 48/48 epochs and 1,628,400 optimizer steps completed.

Matched scoring:

- authority root: `tfpd_exploration/results/cell_d_equal_session_seed42_score_authority_v1`;
- official preflight SHA-256: `26ffe987abfb10047ceef8296da7ea6ee771cb37fb36d6945a0a7ce235062507`;
- root authorization SHA-256: `87512725d3fa15f3258d7f5f269717d5b50af3555a01dfc4c804dcdae7dba1d2`;
- score root: `tfpd_exploration/results/cell_d_equal_session_seed42_score_v1`;
- input-authority SHA-256: `2381d5de27db94d3ae064be9613608e6854ba6af3a5e6ec51c7f87b13f334b1d`;
- score SHA-256: `0c752bb421a5903e4b957f9451be631baeb0df58cd030538e1e3a507164708ac`;
- terminal SHA-256: `933b1ab93918483eeae40bdd703509c276d175ecee54924cc6ac5f0a8d3bea23`;
- terminal verdict: `STOP`.

The scorer first reproduced all 21 sealed Cell-D last-bin session scores exactly. Both models then
used the same no-cache parsed inputs, source normalizers, M30 T4 rows, valid query endpoints, and
last-bin target bytes. No target update, backward pass, optimizer step, normalizer refit, or formal
surface access occurred.

## Governing result

| surface | sealed Cell D | equal-session | delta | paired sign evidence |
|---|---:|---:|---:|---|
| within, 6 sessions | 0.569685 | 0.566522 | -0.003163 | 2/6 positive; median -0.007223 |
| external, 15 sessions | 0.417936 | 0.415124 | -0.002812 | 9/15 positive; median +0.008660 |

The pre-registered stop rule was triggered because the external mean delta was below zero. The
within delta remained safely above the -0.03 retention floor, but there was no performance gain.

External effects were heterogeneous. Most sessions were near zero or modestly positive, but two
large regressions dominated the equal-session mean:

- `sub-M_ses-CO-20140627`: delta -0.223514;
- `sub-M_ses-CO-20150512`: delta -0.116988.

The largest positive sessions were `sub-M_ses-CO-20150626` (+0.121676),
`sub-M_ses-CO-20150616` (+0.086010), and `sub-M_ses-CO-20150615` (+0.056498).

## Interpretation

The previous window-count imbalance was not the governing bottleneck for average zero-shot
transfer. Equalizing source-session exposure changes which target sessions benefit, but it does not
produce a reliable system-level gain.

The lower final source loss from the equal-session training run is not evidence of better transfer.
The sampling distribution changed, so its source objective is differently weighted. The matched
within/external score is the valid comparison, and that comparison is neutral-to-negative.

The positive external median and 9/15 positive sessions mean equal-session sampling is not
universally harmful. However, the route fails the performance-first requirement because the gains
are small and are offset by large regressions. A post-hoc mixture or session-dependent selector is
not authorized by this result and would add target-dependent complexity.

## Paper use

This result may appear as a compact negative control or appendix ablation:

> Balancing source-session exposure did not improve average zero-shot transfer, indicating that the
> benefit of population sparsification is not explained by correcting source-session window-count
> imbalance.

Do not present equal-session sampling as an innovation, a robustness improvement, or a replacement
for Cell D.

## Next work

1. Let the already-running `TFSR_B3ST4_DDROP_SEED42` complete unchanged.
2. Score TF-SR against the same sealed Cell-D last-bin baseline after its valid terminal/SWA exists.
3. Use the TF-SR result—not additional sampling ablations—to decide whether the next performance
   round should refine causal state-conditioned read-in or pivot to another larger architecture bet.
4. Do not launch equal-session seeds 43/44 unless a later independent result provides a new,
   pre-registered mechanism that predicts the two large external regressions.

