# H1 Per-Session Ridge Baseline — Protocol-Corrected v2r2 Handoff

**Date:** 2026-08-12  
**Status:** Complete. CPU-only immutable replay; independent from-source byte-identical verification PASS.  
**Scientific role:** Descriptive four-trial deployment-boundary reference, not an information- or architecture-matched carrier control.

## Authoritative artifacts

| Artifact | SHA-256 | Mode |
|---|---|---:|
| `sua_exploration/results/h1_ridge_baseline_v2r2/h1_ridge_baseline_v2r2_receipt.json` | `2c76d3c51611e4dc9489efeca729a896ac7b0495fc7742fcbdf6f4f78ea33c28` | `0444` |
| `sua_exploration/results/h1_ridge_baseline_v2r2/h1_ridge_baseline_v2r2_arrays.npz` | `39332089ed34957feb165cbdecbf393776da76aae87dd791e28a0af0dc3a7778` | `0444` |
| `sua_exploration/results/h1_ridge_baseline_v2r2/h1_ridge_baseline_v2r2_verification.json` | `980e0383b100f597678535dcf350a7f7087e10a5443898e90bad991895273585` | `0444` |

The verifier reloaded both source NWBs, reconstructed the calibration and actual scored-query rows,
refit the two seven-output ridge models, and required all 24 stored arrays to be byte-identical. Its
terminal status is `PASS_FROM_SOURCE_BYTE_IDENTICAL_REPLAY`.

The earlier v1 receipt (`f12cfe39...00d`) is superseded because calibration and query used different
50-bin target alignment. The v2 receipt (`131553d6...2a6`) has the correct scientific value but is
superseded for reporting by v2r2 because v2 hashed a separately reconstructed query list and copied
the wrong-scope H-SE5 label totals. Neither historical receipt was overwritten.

## Frozen protocol

| Property | v2r2 contract |
|---|---|
| Development scope | H1 fold-0 date `19250101`; two public held-in recordings |
| Calibration | Chronological first four trials per recording |
| Feature | Causal 50-bin × 176-channel spike history, `[t-49:t+1]` |
| Calibration containment | Every history lies entirely within one support trial |
| Target | Dense 7-DoF `OpenLoopKinematicsVelocity` at bin `t` |
| Solver | Uniform normalized ridge, fixed `lambda=1`, unpenalized intercept |
| Hyperparameter role | Canonical fixed comparator; no H1 source- or target-side selection |
| Query | 8,965 actual scored windows, all strictly post-support |
| Query SHA | `665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da`, exact H-SE5 match |
| Metric | Float64 pooled multi-output SSE/TSS, decoder-window last bin |
| Target-session optimization | Zero optimizer steps; zero backward steps |

## Result

| System/reference | Pooled R² | Role |
|---|---:|---|
| H-C | `0.525511` | Source-pretrained decoder + dense carrier |
| **Context Full** | **`0.516518`** | **Source-pretrained decoder + sparse endpoint/event-context carrier; selected H1 sparse representative** |
| H-SE5 | `0.500037` | Endpoint-only sparse ablation |
| H-S | `0.496833` | Source-pretrained SPINT activity-only reference |
| Zero5 | `0.471569` | Source-pretrained compact zero-carrier control |
| **Ridge v2r2** | **`0.258235`** | Separate target-session dense-velocity linear readout |

Ridge v2r2 per recording:

| Recording | Calibration rows | Query windows | R² |
|---|---:|---:|---:|
| `ses-19250101T111740` | 2,960 | 6,735 | `0.237329` |
| `ses-19250101T112404` | 3,128 | 2,230 | `0.317563` |
| **Pooled** | **6,088** | **8,965** | **`0.258235`** |

The v2r2 value exactly matches the protocol-corrected v2 value. The off-by-one and containment
corrections therefore do not change the qualitative ordering, but only v2r2 has the final reporting
contract and replay evidence.

## Fold-0 M4 target-supervision accounting

The H-SE5 counts are read from and bound to immutable
`source_audit_v2r2.json` (SHA-256 `de4c23ac...28a4`), rather than copied from its all-recording M3
summary.

| Estimator/package | Target-session calibration accounting |
|---|---|
| Context Full M4 | 41 movement events; the same 574 acquired endpoint coordinates as H-SE5; displacement and midpoint derived from those endpoints; native event tags; 164 projected q4 inputs |
| H-SE5 M4 endpoint estimator | 41 movement events; 574 acquired endpoint coordinates; 287 derived displacement coordinates; 164 projected q4 inputs |
| Ridge v2r2 | 6,088 dense 7-DoF velocity rows; 42,616 velocity coordinates |

These are different algorithmic target representations. They are **not** independent-sample counts,
effective sample sizes, human-annotation costs, equal-information arms, or a causal explanation of
the accuracy difference. H-SE5 does not read the dense velocity series during target-session carrier
construction; offline source decoder training still uses behavior targets.

## Paper-safe interpretation

The paper may say:

> On the two H1 fold-0 development recordings, a fixed four-trial per-session Ridge50 comparator
> supplied with dense 7-DoF velocity achieved 0.258 pooled R² on the same 8,965 strict post-support
> query windows used by H-S, H-SE5, Context Full, and H-C. Source-pretrained decoder systems achieved 0.497--0.526
> R² under that query boundary. This is a descriptive deployment-boundary reference rather than an
> information- or architecture-matched carrier comparison.

The paper must not claim that:

- v2r2 is an optimized H1 ridge baseline or a ridge-family upper bound;
- `rows < features` has been isolated as the cause of Ridge's lower score;
- the comparison proves carrier superiority or supplies a carrier-content effect;
- H-SE5 consumes one scalar per event;
- the coordinate counts are independent observations or an annotation-cost ratio;
- Ridge v2r2 itself is cross-date, multi-seed, organizer-held, or evidence for/against the separate
  endpoint-sparse Context Full system.

The H1 main claims remain separate: Context Full supplies the pooled endpoint-sparse development
representative; matched five-date H-C supplies carrier/consumer attribution; and the organizer-held
endpoint evaluates the dense H-C complete system. Ridge v2r2 does not reopen H1 architecture search.

## Implementation

- Runner and verifier: `sua_exploration/scripts/run_h1_ridge_baseline_v2r2.py`
- Original v2 runner retained unchanged: `sua_exploration/scripts/run_h1_ridge_baseline_v2.py`
- Numerical core: `sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py`
- H1 loader: `SPINT-main/src/data/h1_m4_eb_pilot.py`

Execution was CPU-only with `CUDA_VISIBLE_DEVICES=""`, four-thread BLAS caps, and process nice value
10. No GPU task, checkpoint, or watched training directory was modified.
