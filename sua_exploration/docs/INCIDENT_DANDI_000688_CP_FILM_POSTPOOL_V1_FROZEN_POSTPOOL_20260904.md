# Incident: DANDI CP-FiLM Post-Pool V1 Did Not Unfreeze Post-Pool

Status: `V1_RESULT_INVALID_FOR_COADAPTATION_CLAIM__ROOT_IMMUTABLE`

The source-only seed42 run under
`sua_exploration/results/dandi688_cp_film_postpool_v1/seed42` completed, but it
did not execute the co-adaptation intervention named by its design.

## Cause

The parent loader freezes every student parameter before returning the native
model. `copy.deepcopy(student.id_encoder.post_pool)` retained
`requires_grad=False`. The V1 arm constructor counted all post-pool parameters
instead of counting only trainable parameters, so its 13,050-parameter check did
not detect the error.

The immutable `training.json` proves the defect: for all four arms,
`trainable_names` contains only:

- `film.0.weight`;
- `film.0.bias`;
- `film.2.weight`;
- `film.2.bias`.

No `post_pool.*` parameter is present. The V1 epoch-12 validation values are
also exactly the frozen-head predecessor values, as expected when only the FiLM
head is trained.

## Consequence

The V1 root is valid evidence for a repeat of the frozen-head FiLM training and
for the pre-registered epoch-9--12 parameter-average diagnostic. It is invalid
evidence about FiLM/post-pool co-adaptation and must not close that hypothesis.

The root and its receipts remain immutable. No scientific number will be
deleted or rewritten.

Relevant immutable body hashes:

- `attempt.json`: `f69cd7fd6078cca46769dc3094c6a17df241d7a5cbde3b0cdb57dcb8edc2d7e4`;
- `source_authority.json`: `da3ebf1ff02a2397165c08f55905ec32ac8f09581caf8232e5a239015bb85907`;
- `training.json`: `8ae1ef8af47dd34ae234ba242696cc4e200eb94a8f624a45cdccf9c897d0328d`;
- `score.json`: `89f40f3e3e89cda1877b95115839cad09fb7c34b531e9da5192199847665efcb`.

## Required V2 repair

An additive V2 root must:

1. bind this incident and the V1 body hashes;
2. set every arm-local `post_pool` parameter to `requires_grad=True` after the
   copy is made;
3. require exactly 13,050 trainable parameters and require the complete
   `post_pool.*` trainable-name set;
4. prove the native parent remains frozen;
5. prove each arm's post-pool state changes after training;
6. preserve the V1 science, averaging law, source split, and stop rule.

