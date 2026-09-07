# DANDI 000688 sparse-event T4/FiLM Stage-2 V1 execution-context incident

**Status:** frozen additive-successor authority.

Two externally managed interactive execution contexts were reclaimed after
publishing their immutable attempts and before the Python process could publish
either a terminal or a Python failure receipt.  This is recorded as
`EXTERNAL_EXECUTION_CONTEXT_LOSS`.  It is an execution-context loss, not a
model result, a training failure, a data failure, or a scientific gate result.

The affected, held roots are exactly:

| Branch | Root | Attempt SHA-256 |
| --- | --- | --- |
| estimator | `stage2/estimator/sua/seed44/` | `4a89bdbf837cdf25c5d39708b5c4511f554a64fa698adcea6c192d55020d42e0` |
| FiLM | `stage2/film/sua/seed42/` | `17571373cde77c77375cf594fd63d80019c34b18b980af1e1d9e755e80d45bb9` |

At incident validation each root had exactly `attempt.json` and
`attempt.json.sha256`, each a regular `0444`, single-link immutable leaf.  No
terminal, failure, checkpoint, score, or training leaf may be added to either
root, and neither root may be reused or overwritten.

The only authorized correction is an additive `stage2_v2` namespace.  It may
reuse the completed V1 estimator terminal roots for seeds 42 and 43, but must
run estimator seed 44 only under `stage2_v2/estimator/sua/seed44/`.  All three
FiLM seeds must run only under `stage2_v2/film/sua/seed42..44/`.  The successor
must held-validate this document and both V1 attempt leaves before publishing
any successor attempt.  Stage-2 aggregation must bind the mixed estimator
authority and the all-successor FiLM authority explicitly.
