# Work Order: CDM-D Activity-Only Quick V1

Date: 2026-08-28

Status: authorized non-governing matched engineering screen

## Question

Does the useful M4/M10 part of CDM-D survive when completed query trials may
update only the B3S activity FIFO and the support-only carrier is immutable?

## Fixed predecessor and inputs

Use the accepted V8 result graph exactly:

- closure `62c685fb0d288107ed43d7a0ef469bbb655c32e68b0d7b677aac8b371ab94939`
- attempt `557bf8071094d6512be862b1b56f0d8a949df57f5fe79356d76771d1fea2a376`
- input `ada5427650d5e612232e74b12159b332721fd475bb218f9894ec3b30ecffdd07`
- score `98ea2bca22b4dbce6ac96b9b517a3774262115b62b6b0e15a1c242191633f77e`
- terminal `80b2dff139a1298e1447c9313ae70548191c7406f2643a1cfb45a1867dec8096`

Materialized records must exactly match the V8 input-record digest. Reuse the
V8 sealed and full-CDM rows as immutable comparators; do not rerun them.

## Runtime change

Keep V8 parser, sealed model, normalizer, support-only fixed-ridge carrier,
variable-prefix B3S forward, query chronology, and metric unchanged. For each
valid completed query trial:

1. append the B3S activity capability to the bounded FIFO;
2. acquire no complementary-group prediction and make no carrier proposal;
3. preserve carrier state and digest exactly.

Run only M10 and M4, within-6 then external-15. M30 is exactly the sealed
support-only system under this policy and is not rerun.

## Decision rule

Use the frozen gates from the full activity-only specification:

- M10 external delta versus sealed >= +0.02 and >=10/15 positive;
- M4 external delta versus sealed >= +0.05 and >=10/15 positive;
- neither within delta below -0.01;
- no external session below sealed by more than 0.20;
- `sub-M_ses-CO-20140626` improves by >=0.30 versus V8 full CDM at both
  budgets, or is no lower than -0.20 versus sealed at both budgets.

This quick result is matched and meaningful but non-governing: it has one
attempt/result terminal and no separate authority-mint ceremony. A positive
result selects the route for a full successor and seed replication. A negative
result stops activity-only refinement.

