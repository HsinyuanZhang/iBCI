# Track-B v2 RT development executor/scorer boundary

This additive route is a no-target scaffold, not an execution authority.  It
consumes the canonical development materializer and the immutable RT
15-session local-asset authority body
`771ab920531a322fbccc9a745d80cc1cb6b3325bcc469283161e2561f7da9352`.
The sole canonical fixed d8/it250 GPU engineering-cost receipt is a hard gate
before held-target path resolution, stat, descriptor open, snapshot creation,
NWB parsing, CEBRA import/fit, readout fit, or score.  The receipt is currently
absent, so the route is explicitly **NO-GO**.

A future separately reviewed live successor must use one fresh joint
14-source-plus-held-M24-support encoder at fixed `d=8`, `iterations=10000`.
It must report normalized ridge `lambda=0.01` and cosine kNN `k=3` separately
over all three mandatory readout routes; no score-driven geometry, seed,
route, or decoder selection is permitted.  Held query rows are strictly
post-M24 and enter no fit.  Prediction targets are exactly valid-window start
plus 49, and every offset10 receptive field is the half-open range
`[endpoint-5, endpoint+5)`, wholly inside query and disjoint from support.
This exposes four strictly future raw bins and is noncausal; any online or
latency-equivalent description is forbidden.

Target parsing must consume a continuously held verified descriptor or an
immutable private snapshot copied from it.  The required sequence is
the canonical
`track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority`
entrypoint, not a caller-supplied ledger mapping, followed by `O_NOFOLLOW`
open, same-FD size/SHA and inode checks, `O_EXCL` private copy,
fsync and mode `0444`, held parser FD rewind, parser consumption through
`/proc/self/fd/<fd>` or an equivalent descriptor API, and post-parser inode
revalidation.  Checking a pathname before and after an ordinary parser reopen
is not sufficient.  This scaffold opens no RT NWB/NPZ, imports/fits no CEBRA,
uses no GPU, emits no score, and mints no receipt.
