# M1 M10 carrier support stability: visible HO-calib development surface

The audit reconstructs each carrier from the three visible held-out-calibration
NWB recordings using the production public-calibration reader, frozen
all-source NNMF basis and RMS/source normalizer, NNLS activations, and the
production per-unit ridge encoding. Reconstruction is byte/allclose checked
against `encode_heldout_calib_carrier` before any resampling.

This is a same-surface diagnostic: M10 carrier support and the locally visible
query windows belong to the same held-out-calib development recording and can
overlap in time. `query_values_read_for_carrier=false` establishes that carrier
fitting does not consume query targets; it does not establish an independent
post-support evaluation. The receipt binds the window inventory only to make
that surface explicit.

The nominal 50% and 75% treatments use five and eight whole chronological
trials respectively because an M10 support requires an integer count; ten
without-replacement selections use seeds 101--110. The full M10 reference is
evaluated once. The receipt records coverage and condition, support indices,
carrier distance, and separate cold B3 checkpoint/provider load versus three
preloaded B3 identity-forward timings.
