# Budget-Matched Posterior CAL-AUG Source Audit V3

Date: 2026-08-30

V3 is the narrow successor of the immutable V2 failure.  V2 correctly added a
finite-direction observation mask, but its physical support-ID builder still
called V1's `float(target_dir)` law and failed when position 27 carried
`target_dir=null`.

V3 binds both immutable predecessor failure graphs.  Its only semantic repair
is that a physical support ID encodes an absent direction as canonical JSON
`null`.  It does not impute an angle.  Activity/carrier prefix, usable-label
mask, source prior, posterior, equal-budget normalizer, rank/DOF gates, and all
data/GPU boundaries remain exactly the V2 law.

The V1 and V2 result roots remain untouched.  V3 uses a fresh result root and
publishes an attempt before source data access.

