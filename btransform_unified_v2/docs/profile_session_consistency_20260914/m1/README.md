# M1 official Full profile consistency

`analyze_m1_h1_profiles.py m1` reads only the deployable `carrier_official4/carrier_pack.npz` and frozen `fit.npz` used by official Full 582413. Each of seven recordings has a `64×4` source-normalized muscle-response SVD4 profile. Source sessions are 2012-09-24/26/27/28; held-out sessions are 2012-10-04/17/24. `pairs.csv` retains all 21 unordered pairs and `profile_rows.csv` retains every profile value.

Rows are compared by fixed input column (0–63), with no per-session fit, rotation, alignment, or query-label access. The fixed-column contract is an engineering correspondence only: these results must not be described as tracking biological single units across sessions. The 200 whole-row shuffles are a descriptive fixed-column reference, not a p-value.
