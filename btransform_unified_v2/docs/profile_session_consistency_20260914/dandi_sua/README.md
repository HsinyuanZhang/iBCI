# DANDI 2015 SUA MOVE-T4 profile-consistency bundle

This directory is a self-contained audit artifact for 24 prepared source/development sessions: 18 source sessions and 6 development sessions. It quantifies the stability of **session-level empirical distributions** of the four-dimensional MOVE-T4 carrier. It does not establish unit identity across days, estimate decoder accuracy, or use query labels, final data, or model weights. The prepared `carrier_angles` are the development/support directional labels used only to estimate the carrier profile.

`profiles.npz` contains arrays under these keys for each session ID:

- `raw/<session>`: MOVE-T4 estimated from all 33 trial angles/counts.
- `norm/<session>` and `full/<session>`: the same profile after the frozen, source-only carrier normalizer.
- `raw_odd/<session>`, `raw_even/<session>`, `norm_odd/<session>`, and `norm_even/<session>`: estimates from the 17 odd or 16 even trial rows.
- `row_ids/<session>`: local prepared `channel_indices`, matching the rows in every profile above. These are local channel coordinates, not cross-session neuron identities.
- `observed_mask/<session>`: the prepared local-row validity mask; all listed row IDs are observed.

`rows.csv` gives each row's session, real date, source/development split, local row position, and local channel index. `pairs.csv` lists all \(24\choose2=276\) pairs. `split_half.csv` gives the odd/even reference for each session. `summary.json` and `provenance.json` record the exact data inputs, source-normalizer hash and numeric mean/std, metric, and bandwidth rule.

The comparison is a Gaussian-kernel distribution cosine:

\[
S(X,Y) = \frac{\operatorname{mean} k(X,Y)}{\sqrt{\operatorname{mean} k(X,X)\operatorname{mean} k(Y,Y)}},\quad
k(x,y)=\exp[-\lVert x-y\rVert^2/(2\ell^2)].
\]

The bandwidth \(\ell\) is the median nonzero Euclidean distance among all normalized **source** profile rows. This fixed source-only value is used for all source/development pairs and split-half calculations. MMD² is included as a distance companion. `mmd2_unclipped` retains the floating-point calculation; `mmd2` clamps only a negative round-off result to zero. The pair values share sessions, so they are dependent descriptive values; this artifact reports no p-value.

Run `python3 recompute_metrics.py` from any location to recompute all 276 pair values and 24 split-half values from `profiles.npz` and verify them against the CSV/JSON records. Run `python3 plot.py` to recreate `profile_consistency.png` and `profile_consistency.pdf`; it reads only files in this directory. Both scripts require NumPy; `plot.py` also requires Matplotlib. `extract_bundle.py` is the separate, repository-dependent extraction recipe. It is included for provenance and reads the prepared `.sua.npz` carrier counts, carrier angles, local channel indices, observed masks, and frozen source statistics named in `provenance.json`.
