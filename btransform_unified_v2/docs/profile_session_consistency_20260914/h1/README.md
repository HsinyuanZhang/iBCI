# H1 official Full profile consistency

`analyze_m1_h1_profiles.py h1` reads only the sealed `banks_27.npz` `T/<tag>` arrays and the frozen signed-state14 plan used by official Full 582241. It retains all 27 recordings (13 dates) and all 351 unordered recording pairs. Each profile is `176×4`, projected and normalized with the held-in source plan before this analysis; it does not use `E0`, query labels, or decoder scores.

Same-day recordings remain separate in the pair table. Rows are compared by fixed input column (0–175), with no per-session fit, rotation, or alignment. The correspondence is an engineering contract and cannot establish biological single-unit identity. The 200 whole-row shuffles are descriptive only and no independent-pair p-value is reported.
