# Track-B v2 post-synthetic runtime-control authority

This additive publisher translates the independently audited fixed-geometry
synthetic terminal receipt into the two pre-existing consumer contracts.  It
does not train, score, import CEBRA/Torch, discover target paths, or open source,
target, formal, NWB, or NPZ arrays.

The positive-control threshold is the predeclared `R² = 0.70`; it is not the
post-hoc midpoint proposed by the engineering receipt.  Admission requires all
eight seeds of both aligned positive arms (`cebra_joint_behavior` and
`cebra_frozen_source_adapt`) to be strictly above `0.70` for both normalized
ridge and cosine kNN3 on the target-support-only readout.

The deranged hard-null threshold was deliberately left pending in the earlier
contract.  Root freezes it here, after the immutable eight-seed synthetic smoke
and before any real target access, at `R² = 0.60`.  That rounded threshold lies
above the observed hard-null maximum `0.5055904984474182`, below the separately
predeclared positive threshold, is not the measured positive/hard-null midpoint,
and may never be relaxed or backfilled.  All eight deranged seeds must be
strictly below `0.60` for both decoders.  The unaligned arm is retained only as
a reported diagnostic distribution and cannot pass, fail, rescue, or set either
threshold.

The shared raw-bound evidence is fixed to:

- fixed-cost body SHA `ec7096a5e54e444fd6cdafa241aaa88a0720143e3c4e42e3565f022ee662c8e2`;
- synthetic-v2 terminal body SHA `99269afd2770ac333b5a529768920a1adbd4be560dcd942839697edab2c14de8`;
- terminal execution-closure SHA `025a34913925973cab6faf255ae44c5f008bb4646da5237f4bb4a0571316affe`;
- derangement permutation SHA `b101d5fb8d0d7b4703a0df87377253c055f653e970e799de52b733f7250a9444`;
- 32 arm-runs, 56 CEBRA fits, and 192 measurements.

Because Subject-M and RT already consume incompatible canonical schemas, the
publisher prepares two immutable bodies that share one evidence SHA and one
paired-authority set identifier.  Publication is canonical-only and
transactional across both body/sidecar pairs using `O_EXCL`, `fsync`, mode
`0444`, and rollback on conflict.  The default CLI action is a read-only dry
plan; publication requires two explicit root-review flags.  Publishing these
controls still does not itself authorize target execution.
